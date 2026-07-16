#include <xpose/xpose.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace {

// One Context for the whole suite (device selected via XPOSE_DEVICE by the
// test-matrix runner).
class BufferTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<xpose::Context>(); }
  static void TearDownTestSuite() { ctx_.reset(); }
  static std::unique_ptr<xpose::Context> ctx_;
};
std::unique_ptr<xpose::Context> BufferTest::ctx_;

std::vector<uint32_t> pattern_u32(size_t n, uint32_t salt) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint32_t>(i) * 2654435761u ^ salt;
  }
  return v;
}

void roundtrip(xpose::Context& ctx, size_t count, xpose::Usage usage,
               bool force_staging) {
  SCOPED_TRACE("count=" + std::to_string(count) +
               " usage=" + (usage == xpose::Usage::DeviceLocal ? "DeviceLocal" : "HostVisible") +
               " force_staging=" + (force_staging ? "yes" : "no"));
  xpose::Buffer buf = ctx.alloc<uint32_t>(count, usage);
  buf.set_force_staging(force_staging);
  ASSERT_TRUE(buf.valid());
  ASSERT_EQ(buf.size_bytes(), count * sizeof(uint32_t));

  const auto src = pattern_u32(count, 0xA5A5A5A5u);
  buf.upload(std::span<const uint32_t>(src));

  std::vector<uint32_t> dst(count, 0);
  buf.download(std::span<uint32_t>(dst));
  ASSERT_EQ(src, dst);
}

// Element counts deliberately include non-multiples of 4 (odd/prime counts);
// all sizes stay 4-byte-typed as storage buffers require.
const size_t kCounts[] = {1, 3, 7, 64, 1023, 4096, 100003};

TEST_F(BufferTest, RoundtripDeviceLocal) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, xpose::Usage::DeviceLocal, false);
}

TEST_F(BufferTest, RoundtripHostVisible) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, xpose::Usage::HostVisible, false);
}

// Forcing the staging path guarantees the one-shot vkCmdCopyBuffer transfer is
// exercised even on devices where every allocation is host-visible (lavapipe,
// UMA iGPU).
TEST_F(BufferTest, RoundtripForcedStagingDeviceLocal) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, xpose::Usage::DeviceLocal, true);
}

TEST_F(BufferTest, RoundtripForcedStagingHostVisible) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, xpose::Usage::HostVisible, true);
}

TEST_F(BufferTest, HostVisibleUsageIsMappable) {
  xpose::Buffer buf = ctx_->alloc<uint32_t>(64, xpose::Usage::HostVisible);
  EXPECT_TRUE(buf.host_visible());
}

TEST_F(BufferTest, FloatRoundtrip) {
  const size_t n = 1024;
  std::vector<float> src(n);
  for (size_t i = 0; i < n; ++i) src[i] = 0.5f * static_cast<float>(i) - 100.0f;
  src[0] = -0.0f;                                   // IEEE edge inputs survive
  src[1] = std::numeric_limits<float>::quiet_NaN(); // transfers bit-exactly
  src[2] = std::numeric_limits<float>::infinity();

  xpose::Buffer buf = ctx_->alloc<float>(n);
  buf.upload(std::span<const float>(src));
  std::vector<float> dst(n, 0.0f);
  buf.download(std::span<float>(dst));
  // Compare bit patterns (NaN != NaN would fail a value compare).
  EXPECT_EQ(0, std::memcmp(src.data(), dst.data(), n * sizeof(float)));
}

TEST_F(BufferTest, PartialUploadDownloadAtOffset) {
  const size_t n = 256;
  const auto base = pattern_u32(n, 0x11111111u);
  xpose::Buffer buf = ctx_->alloc<uint32_t>(n);
  buf.upload(std::span<const uint32_t>(base));

  // Overwrite the middle quarter at a byte offset.
  const size_t off_elems = 64, sub_elems = 64;
  const auto patch = pattern_u32(sub_elems, 0xDEADBEEFu);
  buf.upload(std::span<const uint32_t>(patch), off_elems * sizeof(uint32_t));

  std::vector<uint32_t> got(n, 0);
  buf.download(std::span<uint32_t>(got));
  for (size_t i = 0; i < n; ++i) {
    const uint32_t want = (i >= off_elems && i < off_elems + sub_elems)
                              ? patch[i - off_elems]
                              : base[i];
    ASSERT_EQ(got[i], want) << "at element " << i;
  }

  // Partial download of just the patched window.
  std::vector<uint32_t> window(sub_elems, 0);
  buf.download(std::span<uint32_t>(window), off_elems * sizeof(uint32_t));
  EXPECT_EQ(window, patch);
}

TEST_F(BufferTest, MoveTransfersOwnership) {
  xpose::Buffer a = ctx_->alloc<uint32_t>(16);
  const auto data = pattern_u32(16, 0x22222222u);
  a.upload(std::span<const uint32_t>(data));

  xpose::Buffer b = std::move(a);
  EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move) — testing moved-from state
  ASSERT_TRUE(b.valid());
  std::vector<uint32_t> got(16, 0);
  b.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, data);

  xpose::Buffer c;
  c = std::move(b);
  EXPECT_FALSE(b.valid());  // NOLINT(bugprone-use-after-move)
  ASSERT_TRUE(c.valid());
  got.assign(16, 0);
  c.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, data);
}

TEST_F(BufferTest, OutOfBoundsTransferThrows) {
  xpose::Buffer buf = ctx_->alloc<uint32_t>(4);
  std::vector<uint32_t> big(5, 0);
  EXPECT_THROW(buf.upload(std::span<const uint32_t>(big)), xpose::Error);
  EXPECT_THROW(buf.download(std::span<uint32_t>(big)), xpose::Error);
  std::vector<uint32_t> ok(4, 0);
  EXPECT_THROW(buf.download(std::span<uint32_t>(ok), 4), xpose::Error);
}

TEST_F(BufferTest, ZeroSizeAllocThrows) {
  EXPECT_THROW(ctx_->alloc<uint32_t>(0), xpose::Error);
}

}  // namespace
