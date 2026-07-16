#include <xpose/sync.hpp>

#include <xpose/check.hpp>

#include <utility>

namespace xpose {

Fence::Fence(const Context& ctx) : ctx_(&ctx) {
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  XP_CHECK(ctx.table().vkCreateFence(ctx.device(), &fci, nullptr, &fence_));
}

Fence::~Fence() { destroy(); }

void Fence::destroy() noexcept {
  if (fence_ != VK_NULL_HANDLE) {
    // Destroying an in-flight fence is invalid; best-effort drain first.
    ctx_->table().vkWaitForFences(ctx_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    ctx_->table().vkDestroyFence(ctx_->device(), fence_, nullptr);
  }
  fence_ = VK_NULL_HANDLE;
  ctx_ = nullptr;
}

Fence::Fence(Fence&& other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)),
      fence_(std::exchange(other.fence_, VK_NULL_HANDLE)) {}

Fence& Fence::operator=(Fence&& other) noexcept {
  if (this != &other) {
    destroy();
    ctx_ = std::exchange(other.ctx_, nullptr);
    fence_ = std::exchange(other.fence_, VK_NULL_HANDLE);
  }
  return *this;
}

bool Fence::wait(uint64_t timeout_ns) const {
  if (fence_ == VK_NULL_HANDLE) return true;  // empty/moved-from: nothing pending
  VkResult r = ctx_->table().vkWaitForFences(ctx_->device(), 1, &fence_,
                                             VK_TRUE, timeout_ns);
  if (r == VK_TIMEOUT) return false;
  XP_CHECK(r);  // throws on VK_ERROR_DEVICE_LOST etc.
  return true;
}

bool Fence::is_signaled() const {
  if (fence_ == VK_NULL_HANDLE) return true;
  VkResult r = ctx_->table().vkGetFenceStatus(ctx_->device(), fence_);
  if (r == VK_NOT_READY) return false;
  XP_CHECK(r);
  return true;
}

}  // namespace xpose
