// vulkore::Buffer — VMA-backed VkBuffer with upload/download transfers.
//
// Every buffer gets STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST usage (clspv
// maps OpenCL global pointers to storage buffers). Transfer strategy is keyed
// on MAPPABILITY (HOST_VISIBLE), NOT coherency:
//   - allocation is HOST_VISIBLE (mappable)        -> direct map + memcpy,
//     with vmaFlush/InvalidateAllocation around the copy when the memory is
//     NOT HOST_COHERENT (e.g. Mali HOST_CACHED). Those calls are no-ops on
//     coherent memory, so desktop (lavapipe/UMA iGPU/coherent device-local)
//     behavior is unchanged. Covers lavapipe, UMA iGPUs, every Android SoC.
//   - otherwise (true device-local, not host-visible) -> staging buffer +
//     one-shot vkCmdCopyBuffer; the staging buffer's OWN memory is likewise
//     flushed/invalidated when non-coherent (discrete GPUs, e.g. NVIDIA).
#pragma once

#include <vulkore/context.hpp>

#include <cstddef>
#include <span>

namespace vulkore {

// Where the allocation should live.
enum class Usage {
  DeviceLocal,  // prefer VRAM / device-local memory (kernel-side working set)
  HostVisible,  // prefer host-visible mapped memory (frequent CPU access)
};

class Buffer {
 public:
  Buffer() = default;  // empty; only destruction/moving allowed
  ~Buffer();

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  VkDeviceSize size_bytes() const { return size_; }
  bool valid() const { return buffer_ != VK_NULL_HANDLE; }
  // True when the allocation is directly MAPPABLE (HOST_VISIBLE) — the memcpy
  // path, no staging copy — regardless of HOST_COHERENT. Non-coherent
  // mappable memory (e.g. Mali HOST_CACHED) still reports true; the transfer
  // just flushes/invalidates around the memcpy.
  bool host_visible() const { return mappable_; }

  template <typename T>
  void upload(std::span<const T> data, VkDeviceSize dst_offset_bytes = 0) {
    upload_bytes(data.data(), data.size_bytes(), dst_offset_bytes);
  }
  template <typename T>
  void download(std::span<T> out, VkDeviceSize src_offset_bytes = 0) {
    download_bytes(out.data(), out.size_bytes(), src_offset_bytes);
  }
  void upload_bytes(const void* src, VkDeviceSize size, VkDeviceSize dst_offset = 0);
  void download_bytes(void* dst, VkDeviceSize size, VkDeviceSize src_offset = 0);

  // Test hook: force the staging-copy path even on host-visible allocations,
  // so both transfer paths are exercised on UMA/lavapipe devices too.
  void set_force_staging(bool force) { force_staging_ = force; }

  // Seam for Program/launch (architecture.md §4): descriptor writes need the
  // raw handle + size for VkDescriptorBufferInfo.
  VkBuffer handle() const { return buffer_; }

 private:
  friend class Context;  // Context::alloc_bytes constructs Buffers
  Buffer(Context& ctx, VkDeviceSize size, Usage usage);

  void destroy() noexcept;

  Context* ctx_ = nullptr;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VmaAllocation allocation_ = VK_NULL_HANDLE;
  VkDeviceSize size_ = 0;
  bool mappable_ = false;   // HOST_VISIBLE: map+memcpy directly (no staging).
  bool coherent_ = false;   // HOST_COHERENT: no flush/invalidate needed.
  bool force_staging_ = false;
};

// --- Context::alloc<T> (declared in context.hpp; Buffer must be complete) ---
template <typename T>
Buffer Context::alloc(size_t count, Usage usage) {
  static_assert(sizeof(T) % 4 == 0,
                "storage buffer element types should be 4-byte aligned");
  return alloc_bytes(static_cast<VkDeviceSize>(count) * sizeof(T), usage);
}

template <typename T>
Buffer Context::alloc(size_t count) {
  return alloc<T>(count, Usage::DeviceLocal);
}

}  // namespace vulkore
