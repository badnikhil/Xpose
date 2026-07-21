// vulkore::Batch and vulkore::DescriptorCache.
//
// Batch had NO coverage in this suite before these tests, which mattered once
// two things about its recording changed for the LLM decode loop
// (agent-docs/llm-decode-loop.md, "cutting the record cost"):
//
//   1. Only the FIRST dispatch in a Batch emits the wide pre-barrier; every
//      later one relies on the previous dispatch's post-barrier. The post
//      barriers inside a Batch are now the narrow COMPUTE -> COMPUTE form, and
//      Batch::submit() appends ONE wide barrier for host/transfer readers.
//   2. Descriptor sets can be served from a DescriptorCache instead of being
//      allocated and freed per dispatch.
//
// Both are silent-corruption changes if wrong — a missed barrier gives a stale
// read, a bad cache key binds the WRONG buffer — so every test here checks
// exact values against a CPU reference, not just "it ran".
#include <vulkore/vulkore.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

std::string batch_kernel_path(const std::string& name) {
  const char* env_dir = std::getenv("VULKORE_KERNEL_DIR");
  std::string dir = env_dir ? env_dir : VULKORE_KERNEL_DIR;
  return dir + "/" + name + ".spv";
}

class BatchTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<vulkore::Context>(); }
  static void TearDownTestSuite() {
    // Programs must die before the Context (see exit-teardown-fix.md).
    programs_.clear();
    ctx_.reset();
  }

  static vulkore::Kernel load(const std::string& module, const char* kernel) {
    programs_.push_back(std::make_unique<vulkore::Program>(
        vulkore::Program::from_file(*ctx_, batch_kernel_path(module))));
    return programs_.back()->kernel(kernel);
  }

  static std::unique_ptr<vulkore::Context> ctx_;
  static std::vector<std::unique_ptr<vulkore::Program>> programs_;
};
std::unique_ptr<vulkore::Context> BatchTest::ctx_;
std::vector<std::unique_ptr<vulkore::Program>> BatchTest::programs_;

// ---------------------------------------------------------------------------
// Barriers: a long chain of dispatches where every one reads what the previous
// one wrote. This is the test that fails if the intra-batch barrier is too weak
// or is dropped — and it fails with WRONG NUMBERS, not a crash.
// ---------------------------------------------------------------------------

// scale_inplace alternates x2 / x0.5 so the expected product is exact in
// binary floating point AND order-sensitive (a dropped or reordered dispatch
// changes the answer rather than cancelling out).
TEST_F(BatchTest, ChainedInPlaceDispatchesSeeEachOther) {
  vulkore::Kernel k = load("scale_inplace", "scale_inplace");
  constexpr uint32_t n = 1000;  // not a multiple of 64: round-up + tail guard
  constexpr int kSteps = 64;

  std::vector<float> host(n);
  for (uint32_t i = 0; i < n; ++i) host[i] = 1.0f + static_cast<float>(i % 13);
  vulkore::Buffer buf = ctx_->alloc<float>(n);
  buf.upload(std::span<const float>(host));

  std::vector<float> expected = host;
  {
    vulkore::Batch b(*ctx_);
    for (int s = 0; s < kSteps; ++s) {
      const float f = (s % 2 == 0) ? 3.0f : 0.25f;
      b.add(k, {n}, buf, f, n);
      for (float& v : expected) v *= f;
    }
    EXPECT_EQ(b.size(), static_cast<size_t>(kSteps));
    EXPECT_TRUE(b.submit().wait());
  }

  std::vector<float> got(n);
  buf.download(std::span<float>(got));  // exercises the batch-final wide barrier
  for (uint32_t i = 0; i < n; ++i) ASSERT_FLOAT_EQ(got[i], expected[i]) << "i=" << i;
}

// Same chain across DIFFERENT pipelines and push-constant layouts, so the
// barrier is proven for dispatch->dispatch transitions that also swap pipeline
// and descriptor-set layout.
TEST_F(BatchTest, ChainedAcrossDifferentKernels) {
  vulkore::Kernel add_one = load("two_kernels", "add_one");
  vulkore::Kernel mul_two = programs_.back()->kernel("mul_two");
  constexpr uint32_t n = 777;
  constexpr int kRounds = 20;

  std::vector<int32_t> host(n);
  for (uint32_t i = 0; i < n; ++i) host[i] = static_cast<int32_t>(i % 7);
  vulkore::Buffer buf = ctx_->alloc<int32_t>(n);
  buf.upload(std::span<const int32_t>(host));

  std::vector<int32_t> expected = host;
  {
    vulkore::Batch b(*ctx_);
    for (int r = 0; r < kRounds; ++r) {
      b.add(add_one, {n}, buf, n);
      b.add(mul_two, {n}, buf, n);
      for (int32_t& v : expected) v = (v + 1) * 2;
    }
    EXPECT_TRUE(b.submit().wait());
  }

  std::vector<int32_t> got(n);
  buf.download(std::span<int32_t>(got));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(got[i], expected[i]) << "i=" << i;
}

// A Batch must also see writes made BEFORE it — that is what the first
// dispatch's wide pre-barrier is for, and it is the one barrier the collapse
// deliberately keeps.
TEST_F(BatchTest, SeesUploadRecordedBeforeTheBatch) {
  vulkore::Kernel k = load("saxpy", "saxpy");
  constexpr uint32_t n = 4096;
  std::vector<float> x(n), y(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = 0.5f * static_cast<float>(i) - 7.0f;
    y[i] = static_cast<float>(i % 31);
  }
  vulkore::Buffer xb = ctx_->alloc<float>(n);
  vulkore::Buffer yb = ctx_->alloc<float>(n);
  xb.upload(std::span<const float>(x));
  yb.upload(std::span<const float>(y));

  vulkore::Batch b(*ctx_);
  b.add(k, {n}, yb, xb, 2.0f, n);
  EXPECT_TRUE(b.submit().wait());

  std::vector<float> got(n);
  yb.download(std::span<float>(got));
  for (uint32_t i = 0; i < n; ++i)
    ASSERT_FLOAT_EQ(got[i], 2.0f * x[i] + y[i]) << "i=" << i;
}

// ---------------------------------------------------------------------------
// DescriptorCache
// ---------------------------------------------------------------------------

// The decode loop's whole premise: the same dispatch chain, resubmitted, must
// allocate descriptor sets only the first time and produce identical results
// every time.
TEST_F(BatchTest, DescriptorCacheHitsOnResubmit) {
  vulkore::Kernel k = load("scale_inplace", "scale_inplace");
  constexpr uint32_t n = 512;
  constexpr int kPerBatch = 8;

  std::vector<float> host(n, 1.0f);
  vulkore::Buffer buf = ctx_->alloc<float>(n);

  vulkore::DescriptorCache dc(*ctx_);
  ASSERT_EQ(dc.size(), 0u);

  for (int token = 0; token < 5; ++token) {
    buf.upload(std::span<const float>(host));
    {
      vulkore::Batch b(*ctx_, dc);
      for (int s = 0; s < kPerBatch; ++s) b.add(k, {n}, buf, 2.0f, n);
      EXPECT_TRUE(b.submit().wait());
    }
    std::vector<float> got(n);
    buf.download(std::span<float>(got));
    // 2^8 = 256, exact in binary floating point.
    for (uint32_t i = 0; i < n; ++i) ASSERT_FLOAT_EQ(got[i], 256.0f) << "i=" << i;

    // Every dispatch in this batch binds the SAME (layout, buffer), so the
    // cache holds exactly one set no matter how many tokens run.
    EXPECT_EQ(dc.size(), 1u) << "token " << token;
  }
  EXPECT_EQ(dc.misses(), 1u);                    // one allocation, ever
  EXPECT_EQ(dc.hits(), 5u * kPerBatch - 1u);     // everything else was a hit
}

// The silent-corruption risk of a cache is a key collision handing back a set
// bound to the WRONG buffer. Same kernel, three different buffers, values
// chosen so any mix-up is visible.
TEST_F(BatchTest, DescriptorCacheKeysOnTheBuffersBound) {
  vulkore::Kernel k = load("scale_inplace", "scale_inplace");
  constexpr uint32_t n = 256;

  vulkore::Buffer a = ctx_->alloc<float>(n);
  vulkore::Buffer b_ = ctx_->alloc<float>(n);
  vulkore::Buffer c = ctx_->alloc<float>(n);
  const std::vector<float> ha(n, 1.0f), hb(n, 10.0f), hc(n, 100.0f);
  a.upload(std::span<const float>(ha));
  b_.upload(std::span<const float>(hb));
  c.upload(std::span<const float>(hc));

  vulkore::DescriptorCache dc(*ctx_);
  {
    vulkore::Batch batch(*ctx_, dc);
    batch.add(k, {n}, a, 2.0f, n);
    batch.add(k, {n}, b_, 4.0f, n);
    batch.add(k, {n}, c, 8.0f, n);
    EXPECT_TRUE(batch.submit().wait());
  }
  EXPECT_EQ(dc.size(), 3u) << "three distinct buffers must not share one set";

  std::vector<float> ga(n), gb(n), gc(n);
  a.download(std::span<float>(ga));
  b_.download(std::span<float>(gb));
  c.download(std::span<float>(gc));
  for (uint32_t i = 0; i < n; ++i) {
    ASSERT_FLOAT_EQ(ga[i], 2.0f) << "i=" << i;
    ASSERT_FLOAT_EQ(gb[i], 40.0f) << "i=" << i;
    ASSERT_FLOAT_EQ(gc[i], 800.0f) << "i=" << i;
  }

  // Re-running the identical chain hits all three, allocating nothing new.
  const uint64_t misses_before = dc.misses();
  {
    vulkore::Batch batch(*ctx_, dc);
    batch.add(k, {n}, a, 1.0f, n);
    batch.add(k, {n}, b_, 1.0f, n);
    batch.add(k, {n}, c, 1.0f, n);
    EXPECT_TRUE(batch.submit().wait());
  }
  EXPECT_EQ(dc.misses(), misses_before);
  EXPECT_EQ(dc.size(), 3u);
}

// A cached and an uncached Batch must compute the same thing. Multi-buffer
// kernel so the cache key covers more than one binding. (vec_add is INT.)
TEST_F(BatchTest, CachedAndUncachedAgree) {
  vulkore::Kernel k = load("vec_add", "vec_add");
  constexpr uint32_t n = 1031;  // prime: round-up + tail guard

  std::vector<int32_t> x(n), y(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = static_cast<int32_t>(i);
    y[i] = static_cast<int32_t>(n - i) * 3;
    expected[i] = x[i] + y[i];
  }

  auto run = [&](vulkore::DescriptorCache* dc) {
    vulkore::Buffer xb = ctx_->alloc<int32_t>(n);
    vulkore::Buffer yb = ctx_->alloc<int32_t>(n);
    vulkore::Buffer ob = ctx_->alloc<int32_t>(n);
    xb.upload(std::span<const int32_t>(x));
    yb.upload(std::span<const int32_t>(y));
    {
      std::unique_ptr<vulkore::Batch> b(
          dc ? new vulkore::Batch(*ctx_, *dc) : new vulkore::Batch(*ctx_));
      b->add(k, {n}, ob, xb, yb, n);
      EXPECT_TRUE(b->submit().wait());
    }
    std::vector<int32_t> got(n);
    ob.download(std::span<int32_t>(got));
    return got;
  };

  const std::vector<int32_t> plain = run(nullptr);
  std::vector<int32_t> cached;
  {
    // The cache is scoped INSIDE the buffers' lifetime by construction here:
    // it is destroyed at the end of this block, before the next run allocates
    // new buffers that could reuse those VkBuffer handles.
    vulkore::DescriptorCache dc(*ctx_);
    cached = run(&dc);
  }
  for (uint32_t i = 0; i < n; ++i) {
    ASSERT_EQ(plain[i], expected[i]) << "i=" << i;
    ASSERT_EQ(cached[i], expected[i]) << "i=" << i;
  }
}

// clear() must actually release the sets, so a cache can be recycled after the
// buffers it saw are gone (the documented way to avoid the handle-reuse trap).
TEST_F(BatchTest, DescriptorCacheClearReleasesSets) {
  vulkore::Kernel k = load("scale_inplace", "scale_inplace");
  constexpr uint32_t n = 64;
  vulkore::DescriptorCache dc(*ctx_);

  for (int round = 0; round < 3; ++round) {
    vulkore::Buffer buf = ctx_->alloc<float>(n);
    const std::vector<float> host(n, 3.0f);
    buf.upload(std::span<const float>(host));
    {
      vulkore::Batch b(*ctx_, dc);
      b.add(k, {n}, buf, 2.0f, n);
      EXPECT_TRUE(b.submit().wait());
    }
    std::vector<float> got(n);
    buf.download(std::span<float>(got));
    for (uint32_t i = 0; i < n; ++i) ASSERT_FLOAT_EQ(got[i], 6.0f);
    EXPECT_EQ(dc.size(), 1u);
    dc.clear();  // buf dies at the end of this scope — drop its set first
    EXPECT_EQ(dc.size(), 0u);
  }
}

// An un-submitted Batch must free its command buffer AND any set it owns.
// With a cache the sets belong to the cache, so this also proves the Batch
// does not double-free them.
TEST_F(BatchTest, AbandonedBatchDoesNotLeakOrDoubleFree) {
  vulkore::Kernel k = load("scale_inplace", "scale_inplace");
  constexpr uint32_t n = 64;
  vulkore::Buffer buf = ctx_->alloc<float>(n);
  vulkore::DescriptorCache dc(*ctx_);
  for (int i = 0; i < 50; ++i) {
    vulkore::Batch plain(*ctx_);
    plain.add(k, {n}, buf, 2.0f, n);
    vulkore::Batch cached(*ctx_, dc);
    cached.add(k, {n}, buf, 2.0f, n);
    // Both destroyed here without submit().
  }
  EXPECT_EQ(dc.size(), 1u);
  // The Context is still usable afterwards.
  const std::vector<float> host(n, 1.0f);
  buf.upload(std::span<const float>(host));
  vulkore::Batch b(*ctx_, dc);
  b.add(k, {n}, buf, 5.0f, n);
  EXPECT_TRUE(b.submit().wait());
  std::vector<float> got(n);
  buf.download(std::span<float>(got));
  for (uint32_t i = 0; i < n; ++i) ASSERT_FLOAT_EQ(got[i], 5.0f);
}

// Many cached batches in flight at once: the sets are shared between
// simultaneously-submitted command buffers, which is legal only because a
// cached set is never updated after creation. Idempotent kernel so the
// unordered completion of the batches does not change the answer.
TEST_F(BatchTest, ManyCachedBatchesInFlight) {
  vulkore::Kernel k = load("vec_add", "vec_add");
  constexpr uint32_t n = 1024;
  std::vector<int32_t> x(n), y(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = static_cast<int32_t>(i);
    y[i] = static_cast<int32_t>(i) * 2;
  }
  vulkore::Buffer xb = ctx_->alloc<int32_t>(n);
  vulkore::Buffer yb = ctx_->alloc<int32_t>(n);
  vulkore::Buffer ob = ctx_->alloc<int32_t>(n);
  xb.upload(std::span<const int32_t>(x));
  yb.upload(std::span<const int32_t>(y));

  vulkore::DescriptorCache dc(*ctx_);
  std::vector<vulkore::Fence> fences;  // Fence dtor DRAINS — hold them all
  fences.reserve(100);
  for (int i = 0; i < 100; ++i) {
    vulkore::Batch b(*ctx_, dc);
    b.add(k, {n}, ob, xb, yb, n);
    fences.push_back(b.submit());
  }
  EXPECT_TRUE(fences.back().wait());
  EXPECT_EQ(dc.size(), 1u);

  std::vector<int32_t> got(n);
  ob.download(std::span<int32_t>(got));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(got[i], x[i] + y[i]) << "i=" << i;
}

}  // namespace
