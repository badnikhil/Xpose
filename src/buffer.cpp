#include <xpose/buffer.hpp>

#include <xpose/check.hpp>

#include <cstring>
#include <utility>

namespace xpose {

namespace {

constexpr VkBufferUsageFlags kBufferUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

// Small RAII staging buffer: host-visible, persistently mapped.
struct Staging {
  Staging(Context& ctx, VkDeviceSize size) : ctx(&ctx) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    XP_CHECK(vmaCreateBuffer(ctx.allocator(), &bci, &aci, &buffer, &allocation, &info));
    mapped = info.pMappedData;
    // The staging memory itself may be non-coherent (Mali hands back
    // HOST_VISIBLE|HOST_CACHED for HOST_ACCESS_RANDOM). Track coherency so the
    // transfer can flush after the host write-in / invalidate before the
    // read-out. vmaFlush/InvalidateAllocation are no-ops on coherent staging
    // memory (desktop: RADV/NVIDIA/llvmpipe), so this costs nothing there.
    VkMemoryPropertyFlags mem_flags = 0;
    vmaGetAllocationMemoryProperties(ctx.allocator(), allocation, &mem_flags);
    coherent = (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
  }
  ~Staging() { vmaDestroyBuffer(ctx->allocator(), buffer, allocation); }
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;

  Context* ctx;
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  void* mapped = nullptr;
  bool coherent = false;
};

}  // namespace

Buffer::Buffer(Context& ctx, VkDeviceSize size, Usage usage)
    : ctx_(&ctx), size_(size) {
  if (size == 0) throw Error("xpose: zero-sized buffer allocation");

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = kBufferUsage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci{};
  if (usage == Usage::HostVisible) {
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  } else {
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  }
  XP_CHECK(vmaCreateBuffer(ctx.allocator(), &bci, &aci, &buffer_, &allocation_,
                           nullptr));

  // Decide the transfer path from what VMA actually gave us. Capture BOTH
  // properties independently:
  //   - mappable_ (HOST_VISIBLE): can we map+memcpy directly (no staging)?
  //     UMA drivers (lavapipe, iGPUs, Android SoCs) hand back host-visible
  //     memory even for DeviceLocal requests, so this is usually true there.
  //   - coherent_ (HOST_COHERENT): are host writes/reads automatically visible
  //     to the device, or must we vmaFlush/InvalidateAllocation around them?
  // Mali (Usage::HostVisible) yields HOST_VISIBLE|HOST_CACHED that is NOT
  // coherent: still mappable (direct path), just needs explicit flush/invalidate
  // — which is exactly why keying the path on coherency (the old bug) wrongly
  // pushed HostVisible buffers to staging AND left both paths cache-incoherent.
  VkMemoryPropertyFlags mem_flags = 0;
  vmaGetAllocationMemoryProperties(ctx.allocator(), allocation_, &mem_flags);
  mappable_ = (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
  coherent_ = (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

Buffer::~Buffer() { destroy(); }

void Buffer::destroy() noexcept {
  if (buffer_ != VK_NULL_HANDLE) {
    vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation_);
  }
  ctx_ = nullptr;
  buffer_ = VK_NULL_HANDLE;
  allocation_ = VK_NULL_HANDLE;
  size_ = 0;
  mappable_ = false;
  coherent_ = false;
}

Buffer::Buffer(Buffer&& other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      size_(std::exchange(other.size_, 0)),
      mappable_(std::exchange(other.mappable_, false)),
      coherent_(std::exchange(other.coherent_, false)),
      force_staging_(other.force_staging_) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    destroy();
    ctx_ = std::exchange(other.ctx_, nullptr);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    size_ = std::exchange(other.size_, 0);
    mappable_ = std::exchange(other.mappable_, false);
    coherent_ = std::exchange(other.coherent_, false);
    force_staging_ = other.force_staging_;
  }
  return *this;
}

void Buffer::upload_bytes(const void* src, VkDeviceSize size, VkDeviceSize dst_offset) {
  if (size == 0) return;
  if (!valid()) throw Error("xpose: upload to empty buffer");
  if (dst_offset + size > size_) throw Error("xpose: upload out of bounds");

  // Direct path whenever the memory is MAPPABLE (host-visible), coherent or
  // not. Staging is reserved for truly non-host-visible (discrete device-local)
  // memory below.
  if (mappable_ && !force_staging_) {
    void* mapped = nullptr;
    XP_CHECK(vmaMapMemory(ctx_->allocator(), allocation_, &mapped));
    std::memcpy(static_cast<char*>(mapped) + dst_offset, src, size);
    // Flush AFTER the host write so it reaches device memory on non-coherent
    // memory (Mali HOST_CACHED). No-op on coherent memory (desktop unchanged).
    if (!coherent_) {
      XP_CHECK(vmaFlushAllocation(ctx_->allocator(), allocation_, dst_offset, size));
    }
    vmaUnmapMemory(ctx_->allocator(), allocation_);
    return;
  }

  // Staging path (non-host-visible device-local target). The staging buffer's
  // own mapped memory may itself be non-coherent (Mali), so flush the host
  // write into it before the GPU copy reads it.
  Staging staging(*ctx_, size);
  std::memcpy(staging.mapped, src, size);
  if (!staging.coherent) {
    XP_CHECK(vmaFlushAllocation(ctx_->allocator(), staging.allocation, 0, size));
  }
  ctx_->one_shot([&](VkCommandBuffer cb) {
    VkBufferCopy region{0, dst_offset, size};
    ctx_->table().vkCmdCopyBuffer(cb, staging.buffer, buffer_, 1, &region);
  });
}

void Buffer::download_bytes(void* dst, VkDeviceSize size, VkDeviceSize src_offset) {
  if (size == 0) return;
  if (!valid()) throw Error("xpose: download from empty buffer");
  if (src_offset + size > size_) throw Error("xpose: download out of bounds");

  // Direct path whenever the memory is MAPPABLE (host-visible), coherent or
  // not. Callers reading a KERNEL-written buffer must have waited on the launch
  // fence first (unchanged contract) — that fence + launch.cpp's post-dispatch
  // HOST barrier make the device writes available; the invalidate below then
  // refreshes the CPU cache from memory before we read on non-coherent memory.
  if (mappable_ && !force_staging_) {
    void* mapped = nullptr;
    XP_CHECK(vmaMapMemory(ctx_->allocator(), allocation_, &mapped));
    // Invalidate AFTER mapping but BEFORE the read: it refreshes the CPU cache
    // from device memory so we see the device's writes, not a stale HOST_CACHED
    // line (Mali). The order matters — vkInvalidateMappedMemoryRanges requires
    // the memory to be CURRENTLY host-mapped (VUID-VkMappedMemoryRange-memory-
    // 00684); invalidating an unmapped, non-persistently-mapped allocation
    // null-derefs inside the Mali driver. No-op on coherent memory (desktop).
    if (!coherent_) {
      XP_CHECK(vmaInvalidateAllocation(ctx_->allocator(), allocation_, src_offset, size));
    }
    std::memcpy(dst, static_cast<const char*>(mapped) + src_offset, size);
    vmaUnmapMemory(ctx_->allocator(), allocation_);
    return;
  }

  // Staging path (non-host-visible device-local source).
  Staging staging(*ctx_, size);
  ctx_->one_shot([&](VkCommandBuffer cb) {
    VkBufferCopy region{src_offset, 0, size};
    ctx_->table().vkCmdCopyBuffer(cb, buffer_, staging.buffer, 1, &region);
    // Fence wait in one_shot() makes the GPU copy available to the host; on
    // non-coherent staging memory (Mali) we still must invalidate the CPU
    // cache before reading — done after the fence, below.
  });
  if (!staging.coherent) {
    XP_CHECK(vmaInvalidateAllocation(ctx_->allocator(), staging.allocation, 0, size));
  }
  std::memcpy(dst, staging.mapped, size);
}

}  // namespace xpose
