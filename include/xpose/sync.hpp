// xpose::Fence — thin RAII wrapper over VkFence.
//
// Returned by Context::submit(); the future launch() will return an Event
// backed by the same mechanism (architecture.md §5). Timeline semaphores are a
// deliberate later nice-to-have.
#pragma once

#include <xpose/context.hpp>

#include <cstdint>
#include <functional>

namespace xpose {

class Fence {
 public:
  Fence() = default;  // empty fence: wait() is a no-op returning true
  ~Fence();

  Fence(const Fence&) = delete;
  Fence& operator=(const Fence&) = delete;
  Fence(Fence&& other) noexcept;
  Fence& operator=(Fence&& other) noexcept;

  // Blocks until signaled or timeout. Returns true if signaled, false on
  // timeout; throws xpose::Error on device loss. Waiting on an already
  // signaled fence (double-wait) returns true immediately.
  bool wait(uint64_t timeout_ns = UINT64_MAX) const;
  bool is_signaled() const;

  VkFence handle() const { return fence_; }

 private:
  friend class Context;  // Context::submit constructs signalable fences
  Fence(const Context& ctx);  // creates an unsignaled VkFence

  void destroy() noexcept;
  // Runs the completion hook exactly once, the first time completion is
  // observed (successful wait()/is_signaled(), or the draining destructor).
  // launch() uses it to recycle its command buffer + descriptor sets.
  void fire_on_complete() const noexcept;

  const Context* ctx_ = nullptr;
  VkFence fence_ = VK_NULL_HANDLE;
  mutable std::function<void()> on_complete_;
};

}  // namespace xpose
