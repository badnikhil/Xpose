// vulkore::launch — the headline API. CUDA-style kernel dispatch:
//
//   vulkore::Fence f = vulkore::launch(kernel, {n}, y_buf, x_buf, 2.0f, n);
//   f.wait();                       // or let the Fence destructor drain
//
// Grid is the GLOBAL thread count (1/2/3-D). It is rounded UP to whole
// workgroups (kernels bound-check, CUDA-style), matching the
// -uniform-workgroup-size contract the fixtures are compiled with.
//
// Arguments are matched positionally against the kernel's clspv reflection:
//   vulkore::Buffer      -> its storage-buffer descriptor (set/binding)
//   POD value          -> its slice of the push-constant block
// Wrong arity, a Buffer where a POD is expected (or vice versa), and POD size
// mismatches (e.g. a 2.0 double literal for a float arg) throw vulkore::Error
// with the argument's name in the message. Non-POD/pointer arguments fail to
// compile via static_assert.
//
// The recorded command buffer wraps the dispatch in pipeline barriers so
// upload -> launch -> download and launch -> launch chains on the Context
// queue are correctly synchronized. The command buffer and descriptor sets
// are recycled once the returned Fence observes completion (wait/destroy).
#pragma once

#include <vulkore/buffer.hpp>
#include <vulkore/program.hpp>
#include <vulkore/sync.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vulkore {

// Global thread counts (CUDA semantics), NOT workgroup counts.
struct Grid {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

class DescriptorCache;

namespace detail {

struct LaunchArg {
  const Buffer* buffer = nullptr;  // exactly one of buffer/pod is set
  const void* pod = nullptr;
  uint32_t pod_size = 0;
};

Fence launch_impl(Kernel& kernel, Grid grid, const LaunchArg* args,
                  size_t count);

// `cache` may be null (allocate a fresh set per dispatch, freed via out_sets).
// `emit_pre_barrier` is false for every dispatch but the first in `cb`: the
// previous dispatch's post-barrier already covers it.
void record_into(VkCommandBuffer cb, Kernel& kernel, Grid grid,
                 const LaunchArg* args, size_t count,
                 std::vector<VkDescriptorSet>& out_sets,
                 DescriptorCache* cache = nullptr,
                 bool emit_pre_barrier = true, bool wide_post = true);

template <typename T>
LaunchArg make_launch_arg(const T& value) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<U, Buffer>) {
    return LaunchArg{&value, nullptr, 0};
  } else {
    static_assert(!std::is_pointer_v<U>,
                  "vulkore::launch: raw pointers cannot be kernel arguments — "
                  "pass an vulkore::Buffer for global memory");
    static_assert(std::is_trivially_copyable_v<U>,
                  "vulkore::launch: arguments must be vulkore::Buffer or "
                  "trivially-copyable POD values");
    static_assert(sizeof(U) <= 128,
                  "vulkore::launch: POD argument larger than any push-constant "
                  "block — pass it via an vulkore::Buffer");
    return LaunchArg{nullptr, &value, static_cast<uint32_t>(sizeof(U))};
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// DescriptorCache — reuse descriptor sets across submissions.
//
// Default `Batch`/`launch()` allocate a descriptor set per dispatch and free it
// when the Fence signals. That is correct but not free: on an Adreno 840 the
// vkAllocateDescriptorSets + vkUpdateDescriptorSets pair is the single largest
// term in host recording cost, and an LLM decode step records 838 dispatches
// *per token* with the SAME buffers bound every time.
//
// A descriptor set is fully determined by (set layout, the buffers bound into
// it). This caches sets on exactly that key, so a repeating dispatch chain
// allocates on its first token and hits the cache forever after. Cached sets
// are never updated after creation, so re-binding one in a second in-flight
// command buffer is legal.
//
//   vulkore::DescriptorCache dc(ctx);          // lives as long as the buffers do
//   for (;;) {
//     vulkore::Batch b(ctx, dc);
//     ...
//     b.submit().wait();
//   }
//
// LIFETIME CONTRACT — the sharp edge. The key holds raw `VkBuffer` handles, and
// a driver may hand the same handle back for a NEW buffer once an old one is
// destroyed. So:
//   * a DescriptorCache must be destroyed BEFORE the Context, and
//   * `clear()` must be called after destroying any Buffer that was ever bound
//     through it, or a later lookup can return a set pointing at freed memory.
// If in doubt, do not use a cache: plain `Batch(ctx)` is unchanged.
class DescriptorCache {
 public:
  explicit DescriptorCache(Context& ctx) : ctx_(&ctx) {}
  ~DescriptorCache();
  DescriptorCache(const DescriptorCache&) = delete;
  DescriptorCache& operator=(const DescriptorCache&) = delete;

  /** Frees every cached set. Callers must not have any in flight. */
  void clear() noexcept;

  [[nodiscard]] size_t size() const noexcept { return map_.size(); }
  [[nodiscard]] uint64_t hits() const noexcept { return hits_; }
  [[nodiscard]] uint64_t misses() const noexcept { return misses_; }

  // Implementation detail, public only so detail::record_into can reach it.
  static constexpr size_t kMaxBindings = 16;
  struct Key {
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    uint32_t count = 0;
    std::array<uint32_t, kMaxBindings> bindings{};
    std::array<VkBuffer, kMaxBindings> buffers{};
    bool operator==(const Key& o) const noexcept {
      if (layout != o.layout || count != o.count) return false;
      for (uint32_t i = 0; i < count; ++i)
        if (bindings[i] != o.bindings[i] || buffers[i] != o.buffers[i])
          return false;
      return true;
    }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
      uint64_t h = 1469598103934665603ull;
      auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
      };
      mix(reinterpret_cast<uint64_t>(k.layout));
      mix(k.count);
      for (uint32_t i = 0; i < k.count; ++i) {
        mix(k.bindings[i]);
        mix(reinterpret_cast<uint64_t>(k.buffers[i]));
      }
      return static_cast<size_t>(h);
    }
  };

  VkDescriptorSet* find(const Key& k) noexcept {
    auto it = map_.find(k);
    if (it == map_.end()) return nullptr;
    ++hits_;
    return &it->second;
  }
  void insert(const Key& k, VkDescriptorSet set) {
    map_.emplace(k, set);
    ++misses_;
  }

 private:
  Context* ctx_ = nullptr;
  std::unordered_map<Key, VkDescriptorSet, KeyHash> map_;
  uint64_t hits_ = 0, misses_ = 0;
};

// Records + submits a dispatch on the Context's compute queue; returns the
// Fence signaled on completion (asynchronous — wait() before reading results).
template <typename... Args>
Fence launch(Kernel& kernel, Grid grid, const Args&... args) {
  if constexpr (sizeof...(Args) == 0) {
    return detail::launch_impl(kernel, grid, nullptr, 0);
  } else {
    const detail::LaunchArg wrapped[] = {detail::make_launch_arg(args)...};
    return detail::launch_impl(kernel, grid, wrapped, sizeof...(Args));
  }
}

// ---------------------------------------------------------------------------
// Batch — many dispatches, ONE vkQueueSubmit.
//
// launch() submits immediately, which costs a measured 0.31 ms per dispatch on
// an Adreno 840 regardless of how little work the kernel does. That is fine for
// a handful of dispatches and ruinous for anything shaped like an LLM: a Gemma
// 1B decode step issues ~130 matmuls, so 40 ms per token disappears into
// submission before the GPU computes anything.
//
// Batch records them all into one command buffer and submits once:
//
//   vulkore::Batch b(ctx);
//   for (auto& layer : layers) {
//     b.add(qkv,  {cols}, out, in, w, s, rows, cols);
//     b.add(mlp,  {cols}, ...);
//   }
//   b.submit().wait();            // ONE submit for the whole token
//
// Barriers are still recorded between dispatches, so ordering and visibility
// match a sequence of individual launch() calls. Destroying an un-submitted
// Batch releases its command buffer and descriptor sets.
class Batch {
 public:
  explicit Batch(Context& ctx);
  /** As above, but descriptor sets come from `cache` — see DescriptorCache for
   *  the lifetime contract. The cache must outlive the Fence this Batch
   *  returns. */
  Batch(Context& ctx, DescriptorCache& cache);
  ~Batch();
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  template <typename... Args>
  Batch& add(Kernel& kernel, Grid grid, const Args&... args) {
    if constexpr (sizeof...(Args) == 0) {
      add_impl(kernel, grid, nullptr, 0);
    } else {
      const detail::LaunchArg wrapped[] = {detail::make_launch_arg(args)...};
      add_impl(kernel, grid, wrapped, sizeof...(Args));
    }
    return *this;
  }

  /** Submits everything recorded so far. The Batch is spent afterwards. */
  Fence submit();

  [[nodiscard]] size_t size() const noexcept { return n_; }

 private:
  void add_impl(Kernel& kernel, Grid grid, const detail::LaunchArg* args,
                size_t count);

  Context* ctx_ = nullptr;
  DescriptorCache* cache_ = nullptr;
  VkCommandBuffer cb_ = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> sets_;  // owned (uncached) sets only
  size_t n_ = 0;
};

}  // namespace vulkore
