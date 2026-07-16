#include <xpose/xpose.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace {

class SyncTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<xpose::Context>(); }
  static void TearDownTestSuite() { ctx_.reset(); }
  static std::unique_ptr<xpose::Context> ctx_;
};
std::unique_ptr<xpose::Context> SyncTest::ctx_;

TEST_F(SyncTest, SubmitEmptyCommandBufferAndWait) {
  VkCommandBuffer cb = ctx_->begin_one_shot();
  xpose::Fence fence = ctx_->submit(cb);
  EXPECT_TRUE(fence.wait());
  EXPECT_TRUE(fence.is_signaled());
  ctx_->free_command_buffer(cb);
}

TEST_F(SyncTest, DoubleWaitIsSafe) {
  VkCommandBuffer cb = ctx_->begin_one_shot();
  xpose::Fence fence = ctx_->submit(cb);
  EXPECT_TRUE(fence.wait());
  EXPECT_TRUE(fence.wait());                 // second wait: immediate success
  EXPECT_TRUE(fence.wait(/*timeout_ns=*/0)); // even with zero timeout
  ctx_->free_command_buffer(cb);
}

TEST_F(SyncTest, SubmitCopyCommandBufferAndVerify) {
  const size_t n = 1024;
  std::vector<uint32_t> src_data(n);
  for (size_t i = 0; i < n; ++i) src_data[i] = static_cast<uint32_t>(i * 7 + 3);

  xpose::Buffer src = ctx_->alloc<uint32_t>(n);
  xpose::Buffer dst = ctx_->alloc<uint32_t>(n);
  src.upload(std::span<const uint32_t>(src_data));

  // Record the copy through the raw seam (what launch() will do internally).
  VkCommandBuffer cb = ctx_->begin_one_shot();
  VkBufferCopy region{0, 0, n * sizeof(uint32_t)};
  ctx_->table().vkCmdCopyBuffer(cb, src.handle(), dst.handle(), 1, &region);
  xpose::Fence fence = ctx_->submit(cb);
  EXPECT_TRUE(fence.wait());
  ctx_->free_command_buffer(cb);

  std::vector<uint32_t> got(n, 0);
  dst.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, src_data);
}

TEST_F(SyncTest, MovedFromFenceWaitIsNoOp) {
  VkCommandBuffer cb = ctx_->begin_one_shot();
  xpose::Fence a = ctx_->submit(cb);
  xpose::Fence b = std::move(a);
  EXPECT_TRUE(b.wait());
  EXPECT_TRUE(a.wait());  // NOLINT(bugprone-use-after-move) — empty fence: no-op true
  EXPECT_TRUE(a.is_signaled());
  ctx_->free_command_buffer(cb);
}

TEST_F(SyncTest, DefaultFenceIsSignaled) {
  xpose::Fence f;
  EXPECT_TRUE(f.wait());
  EXPECT_TRUE(f.is_signaled());
}

TEST_F(SyncTest, OneShotHelperRoundtrip) {
  const size_t n = 64;
  std::vector<uint32_t> data(n, 0xC0FFEEu);
  xpose::Buffer src = ctx_->alloc<uint32_t>(n);
  xpose::Buffer dst = ctx_->alloc<uint32_t>(n);
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

}  // namespace
