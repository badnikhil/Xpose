#include <vulkore/sync.hpp>

#include <vulkore/check.hpp>

#include <utility>

namespace vulkore {

Fence::Fence(const Context& ctx) : ctx_(&ctx) {
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  XP_CHECK(ctx.table().vkCreateFence(ctx.device(), &fci, nullptr, &fence_));
}

Fence::~Fence() { destroy(); }

void Fence::fire_on_complete() const noexcept {
  if (!on_complete_) return;
  std::function<void()> hook = std::move(on_complete_);
  on_complete_ = nullptr;
  try {
    hook();
  } catch (...) {
    // Recycling hooks must not throw through wait()/destructors.
  }
}

void Fence::destroy() noexcept {
  if (fence_ != VK_NULL_HANDLE) {
    // Only drain a fence that was actually submitted. An unsubmitted fence
    // (vkQueueSubmit failed after creation) will never signal, so waiting on it
    // here would deadlock this noexcept path. An unsubmitted fence is not
    // in-flight, so destroying it directly is valid; the completion hook (if
    // any) is left to the caller's error path.
    if (submitted_) {
      // Destroying an in-flight fence is invalid; best-effort drain first.
      ctx_->table().vkWaitForFences(ctx_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
      fire_on_complete();
    }
    ctx_->table().vkDestroyFence(ctx_->device(), fence_, nullptr);
  }
  fence_ = VK_NULL_HANDLE;
  ctx_ = nullptr;
  submitted_ = false;
  on_complete_ = nullptr;
}

Fence::Fence(Fence&& other) noexcept
    : ctx_(std::exchange(other.ctx_, nullptr)),
      fence_(std::exchange(other.fence_, VK_NULL_HANDLE)),
      submitted_(std::exchange(other.submitted_, false)),
      on_complete_(std::exchange(other.on_complete_, nullptr)) {}

Fence& Fence::operator=(Fence&& other) noexcept {
  if (this != &other) {
    destroy();
    ctx_ = std::exchange(other.ctx_, nullptr);
    fence_ = std::exchange(other.fence_, VK_NULL_HANDLE);
    submitted_ = std::exchange(other.submitted_, false);
    on_complete_ = std::exchange(other.on_complete_, nullptr);
  }
  return *this;
}

bool Fence::wait(uint64_t timeout_ns) const {
  // empty/moved-from, or created-but-not-submitted: nothing to wait on.
  if (fence_ == VK_NULL_HANDLE || !submitted_) return true;
  VkResult r = ctx_->table().vkWaitForFences(ctx_->device(), 1, &fence_,
                                             VK_TRUE, timeout_ns);
  if (r == VK_TIMEOUT) return false;
  XP_CHECK(r);  // throws on VK_ERROR_DEVICE_LOST etc.
  fire_on_complete();
  return true;
}

bool Fence::is_signaled() const {
  if (fence_ == VK_NULL_HANDLE || !submitted_) return true;
  VkResult r = ctx_->table().vkGetFenceStatus(ctx_->device(), fence_);
  if (r == VK_NOT_READY) return false;
  XP_CHECK(r);
  fire_on_complete();
  return true;
}

}  // namespace vulkore
