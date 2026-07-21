// launch_impl: reflection-checked argument binding, descriptor-set setup,
// barrier + dispatch recording, queue submission.
#include <vulkore/launch.hpp>

#include <vulkore/check.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace vulkore::detail {

namespace {

std::string arg_label(const KernelInfo& ki, size_t i) {
  std::string label = "argument " + std::to_string(i);
  if (!ki.args[i].name.empty()) label += " ('" + ki.args[i].name + "')";
  return label;
}

uint32_t group_count(const std::string& kernel, char axis, uint32_t global,
                     uint32_t local, uint32_t max_groups) {
  if (global == 0) {
    throw Error("vulkore::launch('" + kernel + "'): grid." + axis +
                " must be nonzero (grid is the global thread count)");
  }
  const uint64_t groups =
      (static_cast<uint64_t>(global) + local - 1) / local;  // round UP
  if (groups > max_groups) {
    throw Error("vulkore::launch('" + kernel + "'): grid." + axis + "=" +
                std::to_string(global) + " needs " + std::to_string(groups) +
                " workgroups; device limit is " + std::to_string(max_groups));
  }
  return static_cast<uint32_t>(groups);
}

}  // namespace

// Records ONE dispatch into an existing command buffer. Any descriptor set it
// ALLOCATES is appended to `out_sets` so the caller can free it when the fence
// signals; sets served from `cache` are owned by the cache and are not.
//
// Split out of launch_impl so a whole batch of dispatches can share one command
// buffer and ONE vkQueueSubmit — submit cost measured at 0.31 ms per dispatch
// on Adreno 840, which for an LLM-shaped workload of ~838 dispatches per token
// is 260 ms of pure overhead.
//
// This function is on the host-recording hot path (an LLM decode step calls it
// 838 times per token), so it deliberately keeps every per-dispatch working
// buffer on the stack. Heap allocation here was measurable.
void record_into(VkCommandBuffer cb, Kernel& kernel, Grid grid,
                 const LaunchArg* args, size_t count,
                 std::vector<VkDescriptorSet>& out_sets, DescriptorCache* cache,
                 bool emit_pre_barrier, bool wide_post) {
  if (!kernel.valid()) {
    throw Error("vulkore::launch: empty Kernel handle");
  }
  Context& ctx = kernel.context();
  const VolkDeviceTable& t = ctx.table();
  const KernelInfo& ki = kernel.info();

  // ---- validate arguments against the reflection --------------------------
  if (count != ki.args.size()) {
    std::string sig;
    for (const KernelArg& a : ki.args) {
      if (!sig.empty()) sig += ", ";
      sig += a.name.empty() ? std::string("<unnamed>") : a.name;
    }
    throw Error("vulkore::launch: kernel '" + ki.name + "' expects " +
                std::to_string(ki.args.size()) + " argument(s) (" + sig +
                "), got " + std::to_string(count));
  }

  // clspv packs every POD into one block; the ABI caps it at 128 bytes and
  // devices seen so far report maxPushConstantsSize 128-256. Stack-allocate the
  // common case, fall back to the heap so an oversized block still works.
  constexpr size_t kPushInline = 256;
  uint8_t push_inline[kPushInline];
  std::vector<uint8_t> push_heap;
  uint8_t* push_data = push_inline;
  const size_t push_size = ki.push_constant_size;
  if (push_size > kPushInline) {
    push_heap.assign(push_size, 0);
    push_data = push_heap.data();
  } else if (push_size) {
    std::memset(push_data, 0, push_size);
  }

  constexpr size_t kMaxBufferArgs = DescriptorCache::kMaxBindings;
  size_t nbuf = 0;
  uint32_t buf_set[kMaxBufferArgs];
  uint32_t buf_binding[kMaxBufferArgs];
  VkBuffer buf_handle[kMaxBufferArgs];

  for (size_t i = 0; i < count; ++i) {
    const KernelArg& ka = ki.args[i];
    const LaunchArg& la = args[i];
    if (ka.kind == ArgKind::StorageBuffer) {
      if (!la.buffer) {
        throw Error("vulkore::launch('" + ki.name + "'): " + arg_label(ki, i) +
                    " is a global-memory pointer — pass an vulkore::Buffer, not "
                    "a POD value");
      }
      if (!la.buffer->valid()) {
        throw Error("vulkore::launch('" + ki.name + "'): " + arg_label(ki, i) +
                    " is an empty (moved-from?) Buffer");
      }
      if (nbuf == kMaxBufferArgs) {
        throw Error("vulkore::launch('" + ki.name + "'): more than " +
                    std::to_string(kMaxBufferArgs) +
                    " buffer arguments is not supported");
      }
      buf_set[nbuf] = ka.set;
      buf_binding[nbuf] = ka.binding;
      buf_handle[nbuf] = la.buffer->handle();
      ++nbuf;
    } else {  // PodPushConstant
      if (!la.pod) {
        throw Error("vulkore::launch('" + ki.name + "'): " + arg_label(ki, i) +
                    " is a POD value in the kernel — an vulkore::Buffer was "
                    "passed");
      }
      if (la.pod_size != ka.size) {
        throw Error("vulkore::launch('" + ki.name + "'): " + arg_label(ki, i) +
                    " is " + std::to_string(ka.size) +
                    " byte(s) in the kernel but a " +
                    std::to_string(la.pod_size) +
                    "-byte value was passed (double literal for a float arg? "
                    "size_t for a uint?)");
      }
      std::memcpy(push_data + ka.offset, la.pod, ka.size);
    }
  }

  // ---- dispatch geometry: round global threads up to whole workgroups -----
  const std::array<uint32_t, 3> local = kernel.local_size();
  const VkPhysicalDeviceLimits& lim = ctx.device_properties().limits;
  const uint32_t gx = group_count(ki.name, 'x', grid.x, local[0],
                                  lim.maxComputeWorkGroupCount[0]);
  const uint32_t gy = group_count(ki.name, 'y', grid.y, local[1],
                                  lim.maxComputeWorkGroupCount[1]);
  const uint32_t gz = group_count(ki.name, 'z', grid.z, local[2],
                                  lim.maxComputeWorkGroupCount[2]);

  // ---- descriptor sets -----------------------------------------------------
  const std::span<const VkDescriptorSetLayout> layouts = kernel.set_layouts();
  constexpr size_t kMaxSets = 8;
  if (layouts.size() > kMaxSets) {
    throw Error("vulkore::launch('" + ki.name + "'): more than " +
                std::to_string(kMaxSets) + " descriptor sets is not supported");
  }
  VkDescriptorSet sets[kMaxSets];
  size_t nsets = 0;
  // Only sets this call allocated; a cache hit contributes nothing to free.
  VkDescriptorSet owned[kMaxSets];
  size_t nowned = 0;
  auto release_owned = [&]() noexcept {
    for (size_t i = 0; i < nowned; ++i) ctx.free_descriptor_set(owned[i]);
  };

  try {
    for (size_t si = 0; si < layouts.size(); ++si) {
      DescriptorCache::Key key;
      key.layout = layouts[si];
      for (size_t i = 0; i < nbuf; ++i) {
        if (buf_set[i] != si) continue;
        key.bindings[key.count] = buf_binding[i];
        key.buffers[key.count] = buf_handle[i];
        ++key.count;
      }

      if (cache) {
        if (VkDescriptorSet* hit = cache->find(key)) {
          sets[nsets++] = *hit;
          continue;
        }
      }

      VkDescriptorSet set = ctx.allocate_descriptor_set(layouts[si]);
      owned[nowned++] = set;
      sets[nsets++] = set;

      VkDescriptorBufferInfo infos[DescriptorCache::kMaxBindings];
      VkWriteDescriptorSet writes[DescriptorCache::kMaxBindings];
      for (uint32_t i = 0; i < key.count; ++i) {
        infos[i] = {key.buffers[i], 0, VK_WHOLE_SIZE};
        writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = key.bindings[i];
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
      }
      if (key.count) {
        t.vkUpdateDescriptorSets(ctx.device(), key.count, writes, 0, nullptr);
      }
      // Handed to the cache only AFTER a successful update, so a throw above
      // never leaves an un-written set reachable.
      if (cache) {
        cache->insert(key, set);
        --nowned;  // the cache owns it now; the fence must not free it
      }
    }

    // ---- record -------------------------------------------------------------
    // Make prior writes (staging copies from Buffer::upload, host writes,
    // previous dispatches on this queue) visible to this dispatch.
    //
    // Only the FIRST dispatch in a command buffer needs this: for every later
    // one the preceding dispatch's post-barrier below already covers
    // COMPUTE/SHADER_WRITE -> ALL_COMMANDS/MEMORY_READ, which subsumes it. That
    // halves the barrier count in a Batch (838 dispatches per LLM token were
    // emitting 1676 barriers).
    if (emit_pre_barrier) {
      VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      pre.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      pre.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      t.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre,
                             0, nullptr, 0, nullptr);
    }

    t.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline());
    if (nsets) {
      t.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                kernel.pipeline_layout(), 0,
                                static_cast<uint32_t>(nsets), sets, 0, nullptr);
    }
    if (push_size) {
      t.vkCmdPushConstants(cb, kernel.pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(push_size), push_data);
    }
    t.vkCmdDispatch(cb, gx, gy, gz);

    // Make this dispatch's writes visible to whatever comes next.
    //
    // A standalone launch() must reach transfer reads (Buffer::download's
    // staging copy) and host reads (a mapped download after the fence), so it
    // pays the wide ALL_COMMANDS|HOST form. Inside a Batch the only thing that
    // can come next in this command buffer is another dispatch, so the narrow
    // COMPUTE -> COMPUTE form is sufficient; Batch::submit() appends ONE wide
    // barrier at the end to cover whatever reads the results. On an Adreno 840
    // the wide form is not free on the GPU either: 838 of them cost ~2.2 ms per
    // token of GPU time on top of the host cost of recording them.
    if (wide_post) {
      VkMemoryBarrier post{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      post.dstAccessMask =
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      t.vkCmdPipelineBarrier(
          cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
          &post, 0, nullptr, 0, nullptr);
    } else {
      VkMemoryBarrier post{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      post.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      t.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &post,
                             0, nullptr, 0, nullptr);
    }

    for (size_t i = 0; i < nowned; ++i) out_sets.push_back(owned[i]);
  } catch (...) {
    release_owned();
    throw;
  }
}

// Single dispatch: its own command buffer and its own submit. Unchanged
// behaviour — every existing call site keeps working.
Fence launch_impl(Kernel& kernel, Grid grid, const LaunchArg* args,
                  size_t count) {
  Context& ctx = kernel.context();
  VkCommandBuffer cb = ctx.begin_one_shot();
  std::vector<VkDescriptorSet> sets;
  try {
    record_into(cb, kernel, grid, args, count, sets);
  } catch (...) {
    ctx.free_command_buffer(cb);
    throw;
  }
  // The Fence recycles the command buffer + descriptor sets once it observes
  // completion (wait()/is_signaled()/destruction).
  return ctx.submit(cb, [ctx_ptr = &ctx, cb, sets]() noexcept {
    for (VkDescriptorSet s : sets) ctx_ptr->free_descriptor_set(s);
    ctx_ptr->free_command_buffer(cb);
  });
}

}  // namespace vulkore::detail

// ---- Batch: many dispatches, ONE submit -----------------------------------
namespace vulkore {

DescriptorCache::~DescriptorCache() { clear(); }

void DescriptorCache::clear() noexcept {
  for (auto& [k, set] : map_) ctx_->free_descriptor_set(set);
  map_.clear();
}

Batch::Batch(Context& ctx) : ctx_(&ctx), cb_(ctx.begin_one_shot()) {}

Batch::Batch(Context& ctx, DescriptorCache& cache)
    : ctx_(&ctx), cache_(&cache), cb_(ctx.begin_one_shot()) {}

Batch::~Batch() {
  if (cb_ != VK_NULL_HANDLE) {          // never submitted — don't leak
    for (VkDescriptorSet s : sets_) ctx_->free_descriptor_set(s);
    ctx_->free_command_buffer(cb_);
  }
}

void Batch::add_impl(Kernel& kernel, Grid grid, const detail::LaunchArg* args,
                     size_t count) {
  if (cb_ == VK_NULL_HANDLE) throw Error("vulkore::Batch: already submitted");
  if (&kernel.context() != ctx_)
    throw Error("vulkore::Batch: kernel belongs to a different Context");
  detail::record_into(cb_, kernel, grid, args, count, sets_, cache_,
                      /*emit_pre_barrier=*/n_ == 0, /*wide_post=*/false);
  ++n_;
}

Fence Batch::submit() {
  if (cb_ == VK_NULL_HANDLE) throw Error("vulkore::Batch: already submitted");
  // One wide barrier for the whole batch: the per-dispatch ones are narrow
  // COMPUTE -> COMPUTE, so this is what makes the results visible to a
  // Buffer::download staging copy or a mapped host read after the fence.
  if (n_) {
    const VolkDeviceTable& t = ctx_->table();
    VkMemoryBarrier post{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    t.vkCmdPipelineBarrier(
        cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
        &post, 0, nullptr, 0, nullptr);
  }
  VkCommandBuffer cb = cb_;
  cb_ = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> sets = std::move(sets_);
  return ctx_->submit(cb, [ctx_ptr = ctx_, cb, sets]() noexcept {
    for (VkDescriptorSet s : sets) ctx_ptr->free_descriptor_set(s);
    ctx_ptr->free_command_buffer(cb);
  });
}

}  // namespace vulkore
