#include <vulkore/vulkore.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

class SyncTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<vulkore::Context>(); }
  static void TearDownTestSuite() { ctx_.reset(); }
  static std::unique_ptr<vulkore::Context> ctx_;
};
std::unique_ptr<vulkore::Context> SyncTest::ctx_;

TEST_F(SyncTest, SubmitEmptyCommandBufferAndWait) {
  VkCommandBuffer cb = ctx_->begin_one_shot();
  vulkore::Fence fence = ctx_->submit(cb);
  EXPECT_TRUE(fence.wait());
  EXPECT_TRUE(fence.is_signaled());
  ctx_->free_command_buffer(cb);
}

TEST_F(SyncTest, DoubleWaitIsSafe) {
  VkCommandBuffer cb = ctx_->begin_one_shot();
  vulkore::Fence fence = ctx_->submit(cb);
  EXPECT_TRUE(fence.wait());
  EXPECT_TRUE(fence.wait());                 // second wait: immediate success
  EXPECT_TRUE(fence.wait(/*timeout_ns=*/0)); // even with zero timeout
  ctx_->free_command_buffer(cb);
}

TEST_F(SyncTest, SubmitCopyCommandBufferAndVerify) {
  const size_t n = 1024;
  std::vector<uint32_t> src_data(n);
  for (size_t i = 0; i < n; ++i) src_data[i] = static_cast<uint32_t>(i * 7 + 3);

  vulkore::Buffer src = ctx_->alloc<uint32_t>(n);
  vulkore::Buffer dst = ctx_->alloc<uint32_t>(n);
  src.upload(std::span<const uint32_t>(src_data));

  // Record the copy through the raw seam (what launch() will do internally).
  VkCommandBuffer cb = ctx_->begin_one_shot();
  VkBufferCopy region{0, 0, n * sizeof(uint32_t)};
  ctx_->table().vkCmdCopyBuffer(cb, src.handle(), dst.handle(), 1, &region);
  vulkore::Fence fence = ctx_->submit(cb);
  EXPECT_TRUE(fence.wait());
  ctx_->free_command_buffer(cb);

  std::vector<uint32_t> got(n, 0);
  dst.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, src_data);
}

TEST_F(SyncTest, MovedFromFenceWaitIsNoOp) {
  VkCommandBuffer cb = ctx_->begin_one_shot();
  vulkore::Fence a = ctx_->submit(cb);
  vulkore::Fence b = std::move(a);
  EXPECT_TRUE(b.wait());
  EXPECT_TRUE(a.wait());  // NOLINT(bugprone-use-after-move) — empty fence: no-op true
  EXPECT_TRUE(a.is_signaled());
  ctx_->free_command_buffer(cb);
}

TEST_F(SyncTest, DefaultFenceIsSignaled) {
  vulkore::Fence f;
  EXPECT_TRUE(f.wait());
  EXPECT_TRUE(f.is_signaled());
}

TEST_F(SyncTest, OneShotHelperRoundtrip) {
  const size_t n = 64;
  std::vector<uint32_t> data(n, 0xC0FFEEu);
  vulkore::Buffer src = ctx_->alloc<uint32_t>(n);
  vulkore::Buffer dst = ctx_->alloc<uint32_t>(n);
  src.upload(std::span<const uint32_t>(data));

  ctx_->one_shot([&](VkCommandBuffer cb) {
    VkBufferCopy region{0, 0, n * sizeof(uint32_t)};
    ctx_->table().vkCmdCopyBuffer(cb, src.handle(), dst.handle(), 1, &region);
  });

  std::vector<uint32_t> got(n, 0);
  dst.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, data);
}

TEST_F(SyncTest, WaitIdleSucceeds) {
  EXPECT_NO_THROW(ctx_->wait_idle());
}

// Exercises the VK_TIMEOUT branch of Fence::wait (sync.cpp:58 -> returns false).
// Builds a large chained-copy GPU workload and polls with a zero timeout so the
// fence is very unlikely to already be signaled. Non-flaky by construction: the
// only hard assertion after observing a timeout is that a BLOCKING wait then
// succeeds (deterministic); if the device drained before we could observe the
// timeout, we GTEST_SKIP rather than risk a false failure.
TEST_F(SyncTest, FenceTimeoutOnUnsignaled) {
  const size_t n = 8 * 1024 * 1024;  // 32 MiB of u32 per buffer
  vulkore::Buffer src = ctx_->alloc<uint32_t>(n, vulkore::Usage::DeviceLocal);
  vulkore::Buffer dst = ctx_->alloc<uint32_t>(n, vulkore::Usage::DeviceLocal);
  std::vector<uint32_t> data(n, 0xABCDEF01u);
  src.upload(std::span<const uint32_t>(data));

  VkCommandBuffer cb = ctx_->begin_one_shot();
  VkBufferCopy region{0, 0, static_cast<VkDeviceSize>(n) * sizeof(uint32_t)};
  // Chain many copies to inflate the workload well past a single DMA burst.
  for (int i = 0; i < 64; ++i) {
    ctx_->table().vkCmdCopyBuffer(cb, src.handle(), dst.handle(), 1, &region);
  }
  vulkore::Fence fence = ctx_->submit(cb);

  const bool timed_out = !fence.wait(/*timeout_ns=*/0);  // poll immediately
  if (!timed_out) {
    // Finished faster than we could observe VK_TIMEOUT — can't test that branch.
    EXPECT_TRUE(fence.wait());
    ctx_->free_command_buffer(cb);
    GTEST_SKIP() << "workload completed too fast to observe a fence timeout";
  }
  // Observed VK_TIMEOUT. A subsequent blocking wait must drain deterministically.
  EXPECT_TRUE(fence.wait());
  EXPECT_TRUE(fence.is_signaled());  // definitely done after a blocking wait
  ctx_->free_command_buffer(cb);
}

// The completion hook (Context::submit(cb, on_complete), context.cpp:248-252)
// must fire EXACTLY ONCE — the first time completion is observed — and never
// again on repeat wait()/is_signaled()/destruction (fire_on_complete moves the
// hook out and nulls it, sync.cpp:16-25). A hook that throws is swallowed.
// This machinery is load-bearing for launch() recycling and had zero coverage.
TEST_F(SyncTest, CompletionHookFiresExactlyOnce) {
  // 1. Observed first via wait(); repeat observations and destruction don't refire.
  int calls = 0;
  {
    VkCommandBuffer cb = ctx_->begin_one_shot();
    vulkore::Fence fence = ctx_->submit(cb, [&calls] { ++calls; });
    EXPECT_TRUE(fence.wait());
    EXPECT_EQ(calls, 1);              // fired on first completion observation
    EXPECT_TRUE(fence.wait());        // repeat wait: must NOT refire
    EXPECT_TRUE(fence.is_signaled()); // nor is_signaled()
    EXPECT_EQ(calls, 1);
    ctx_->free_command_buffer(cb);
  }  // Fence destroyed: draining destructor must NOT refire the spent hook.
  EXPECT_EQ(calls, 1);

  // 2. First observed at DESTRUCTION (no explicit wait): still fires once.
  int calls2 = 0;
  {
    VkCommandBuffer cb = ctx_->begin_one_shot();
    { vulkore::Fence f = ctx_->submit(cb, [&calls2] { ++calls2; }); }  // dtor drains+fires
    ctx_->free_command_buffer(cb);
  }
  EXPECT_EQ(calls2, 1);

  // 3. A throwing hook is swallowed and does not propagate through wait()
  //    (fire_on_complete's catch-all, sync.cpp:22-24).
  {
    VkCommandBuffer cb = ctx_->begin_one_shot();
    vulkore::Fence f = ctx_->submit(cb, [] { throw std::runtime_error("boom"); });
    EXPECT_NO_THROW(f.wait());
    ctx_->free_command_buffer(cb);
  }
}

// Move-assigning over a Fence that owns a live VkFence must drain+destroy the
// old fence before adopting the source (sync.cpp:44-52), and the moved-from
// fence must degrade to a safe no-op (empty fence => wait()/is_signaled() true).
TEST_F(SyncTest, MoveAssignOverLiveFence) {
  VkCommandBuffer cb_a = ctx_->begin_one_shot();
  VkCommandBuffer cb_b = ctx_->begin_one_shot();
  vulkore::Fence a = ctx_->submit(cb_a);
  vulkore::Fence b = ctx_->submit(cb_b);

  a = std::move(b);  // destroys a's old live fence (drains cb_a), adopts b's

  EXPECT_TRUE(a.wait());
  EXPECT_TRUE(a.is_signaled());
  EXPECT_TRUE(b.wait());             // NOLINT(bugprone-use-after-move) empty: no-op true
  EXPECT_TRUE(b.is_signaled());

  ctx_->free_command_buffer(cb_a);   // cb_a drained by the move-assign destroy
  ctx_->free_command_buffer(cb_b);   // cb_b drained by a.wait()
}

}  // namespace
