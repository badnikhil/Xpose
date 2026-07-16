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

Fence launch_impl(Kernel& kernel, Grid grid, const LaunchArg* args,
                  size_t count) {
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

  std::vector<uint8_t> push_data(ki.push_constant_size, 0);
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
      std::memcpy(push_data.data() + ka.offset, la.pod, ka.size);
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

  // ---- descriptor sets ------------------------------------------------------
  const std::span<const VkDescriptorSetLayout> layouts = kernel.set_layouts();
  std::vector<VkDescriptorSet> sets;
  sets.reserve(layouts.size());
  auto release_sets = [&]() noexcept {
    for (VkDescriptorSet s : sets) ctx.free_descriptor_set(s);
  };
  try {
    for (VkDescriptorSetLayout layout : layouts) {
      sets.push_back(ctx.allocate_descriptor_set(layout));
    }

    std::vector<VkDescriptorBufferInfo> buffer_infos;
    buffer_infos.reserve(count);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const KernelArg& ka = ki.args[i];
      if (ka.kind != ArgKind::StorageBuffer) continue;
      buffer_infos.push_back(
          {args[i].buffer->handle(), 0, VK_WHOLE_SIZE});
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = sets[ka.set];
      w.dstBinding = ka.binding;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w.pBufferInfo = &buffer_infos.back();
      writes.push_back(w);
    }
    if (!writes.empty()) {
      t.vkUpdateDescriptorSets(ctx.device(),
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    // ---- record + submit ---------------------------------------------------
    VkCommandBuffer cb = ctx.begin_one_shot();
    try {
      // Make prior writes (staging copies from Buffer::upload, host writes,
      // previous dispatches on this queue) visible to this dispatch.
      VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      pre.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      pre.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      t.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre,
                             0, nullptr, 0, nullptr);

      t.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline());
      if (!sets.empty()) {
        t.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  kernel.pipeline_layout(), 0,
                                  static_cast<uint32_t>(sets.size()),
                                  sets.data(), 0, nullptr);
      }
      if (!push_data.empty()) {
        t.vkCmdPushConstants(cb, kernel.pipeline_layout(),
                             VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             static_cast<uint32_t>(push_data.size()),
                             push_data.data());
      }
      t.vkCmdDispatch(cb, gx, gy, gz);

      // Make this dispatch's writes visible to whatever comes next: transfer
      // reads (Buffer::download staging copy), host reads (mapped download
      // after the fence), and later dispatches.
      VkMemoryBarrier post{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      post.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      t.vkCmdPipelineBarrier(
          cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
          &post, 0, nullptr, 0, nullptr);

      // The Fence recycles the command buffer + descriptor sets once it
      // observes completion (wait()/is_signaled()/destruction).
      return ctx.submit(cb, [ctx_ptr = &ctx, cb, sets]() noexcept {
        for (VkDescriptorSet s : sets) ctx_ptr->free_descriptor_set(s);
        ctx_ptr->free_command_buffer(cb);
      });
    } catch (...) {
      ctx.free_command_buffer(cb);
      throw;
    }
  } catch (...) {
    release_sets();
    throw;
  }
}

}  // namespace vulkore::detail
