// xpose::Buffer — VMA-backed VkBuffer with upload/download transfers.
//
// Every buffer gets STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST usage (clspv
// maps OpenCL global pointers to storage buffers). Transfer strategy:
//   - allocation is HOST_VISIBLE + HOST_COHERENT  -> direct map + memcpy
//     (lavapipe, UMA iGPUs, every Android SoC)
//   - otherwise                                   -> staging buffer + one-shot
//     vkCmdCopyBuffer (discrete GPUs, e.g. the NVIDIA path)
#pragma once

#include <xpose/context.hpp>

#include <cstddef>
#include <span>

namespace xpose {

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
  // True when the allocation is HOST_VISIBLE|HOST_COHERENT (memcpy path).
  bool host_visible() const { return host_visible_coherent_; }

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
  bool host_visible_coherent_ = false;
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

}  // namespace xpose
