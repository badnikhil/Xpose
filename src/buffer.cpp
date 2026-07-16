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
  }
  ~Staging() { vmaDestroyBuffer(ctx->allocator(), buffer, allocation); }
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;

  Context* ctx;
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  void* mapped = nullptr;
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

  // Decide the transfer path from what VMA actually gave us: UMA drivers
  // (lavapipe, iGPUs, Android SoCs) often hand back host-visible+coherent
  // memory even for DeviceLocal requests -> direct memcpy, no staging.
  VkMemoryPropertyFlags mem_flags = 0;
  vmaGetAllocationMemoryProperties(ctx.allocator(), allocation_, &mem_flags);
  host_visible_coherent_ =
      (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
      (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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
  host_visible_coherent_ = false;
}

Buffer::Buffer(Buffer&& other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      size_(std::exchange(other.size_, 0)),
      host_visible_coherent_(std::exchange(other.host_visible_coherent_, false)),
      force_staging_(other.force_staging_) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    destroy();
    ctx_ = std::exchange(other.ctx_, nullptr);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    size_ = std::exchange(other.size_, 0);
    host_visible_coherent_ = std::exchange(other.host_visible_coherent_, false);
    force_staging_ = other.force_staging_;
  }
  return *this;
}

void Buffer::upload_bytes(const void* src, VkDeviceSize size, VkDeviceSize dst_offset) {
  if (size == 0) return;
  if (!valid()) throw Error("xpose: upload to empty buffer");
  if (dst_offset + size > size_) throw Error("xpose: upload out of bounds");

  if (host_visible_coherent_ && !force_staging_) {
    void* mapped = nullptr;
    XP_CHECK(vmaMapMemory(ctx_->allocator(), allocation_, &mapped));
    std::memcpy(static_cast<char*>(mapped) + dst_offset, src, size);
    vmaUnmapMemory(ctx_->allocator(), allocation_);
    return;
  }

  Staging staging(*ctx_, size);
  std::memcpy(staging.mapped, src, size);
  ctx_->one_shot([&](VkCommandBuffer cb) {
    VkBufferCopy region{0, dst_offset, size};
    ctx_->table().vkCmdCopyBuffer(cb, staging.buffer, buffer_, 1, &region);
  });
}

void Buffer::download_bytes(void* dst, VkDeviceSize size, VkDeviceSize src_offset) {
  if (size == 0) return;
  if (!valid()) throw Error("xpose: download from empty buffer");
  if (src_offset + size > size_) throw Error("xpose: download out of bounds");

  if (host_visible_coherent_ && !force_staging_) {
    void* mapped = nullptr;
    XP_CHECK(vmaMapMemory(ctx_->allocator(), allocation_, &mapped));
    std::memcpy(dst, static_cast<const char*>(mapped) + src_offset, size);
    vmaUnmapMemory(ctx_->allocator(), allocation_);
    return;
  }

  Staging staging(*ctx_, size);
  ctx_->one_shot([&](VkCommandBuffer cb) {
    VkBufferCopy region{src_offset, 0, size};
    ctx_->table().vkCmdCopyBuffer(cb, buffer_, staging.buffer, 1, &region);
    // Fence wait in one_shot() makes the copy visible to the host (coherent
    // staging memory); no explicit HOST barrier needed for this pattern.
  });
  std::memcpy(dst, staging.mapped, size);
}

}  // namespace xpose
