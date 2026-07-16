// launch(): end-to-end kernel execution against CPU references, dispatch
// round-up geometry, argument mismatch diagnostics, IEEE comparison edge
// cases (testing-plan.md Layer 1; pattern inherited from dcompute_cmptest).
#include <vulkore/vulkore.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

std::string kernel_path(const std::string& name) {
  return std::string(VULKORE_KERNEL_DIR) + "/" + name + ".spv";
}

class LaunchTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<vulkore::Context>(); }
  static void TearDownTestSuite() { ctx_.reset(); }

  static vulkore::Kernel load(const std::string& name) {
    programs_.push_back(std::make_unique<vulkore::Program>(
        vulkore::Program::from_file(*ctx_, kernel_path(name))));
    return programs_.back()->kernel(name);
  }

  static std::unique_ptr<vulkore::Context> ctx_;
  static std::vector<std::unique_ptr<vulkore::Program>> programs_;
};
std::unique_ptr<vulkore::Context> LaunchTest::ctx_;
std::vector<std::unique_ptr<vulkore::Program>> LaunchTest::programs_;

// ---- end-to-end: upload -> launch -> download vs CPU reference -------------

void run_saxpy(vulkore::Context& ctx, vulkore::Kernel& k, uint32_t n, float a) {
  SCOPED_TRACE("saxpy n=" + std::to_string(n));
  std::vector<float> x(n), y(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = 0.25f * static_cast<float>(i) - 100.0f;
    y[i] = 1.5f * static_cast<float>(i % 97);
  }
  std::vector<float> expected(n);
  for (uint32_t i = 0; i < n; ++i) expected[i] = a * x[i] + y[i];

  vulkore::Buffer xb = ctx.alloc<float>(n);
  vulkore::Buffer yb = ctx.alloc<float>(n);
  xb.upload(std::span<const float>(x));
  yb.upload(std::span<const float>(y));

  vulkore::Fence f = vulkore::launch(k, {n}, yb, xb, a, n);
  EXPECT_TRUE(f.wait());

  std::vector<float> result(n);
  yb.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) {
    ASSERT_FLOAT_EQ(result[i], expected[i]) << "i=" << i;
  }
}

TEST_F(LaunchTest, SaxpyEndToEnd) {
  vulkore::Kernel k = load("saxpy");
  // 1000 is deliberately not a multiple of the 64-wide workgroup (round-up).
  run_saxpy(*ctx_, k, 1000, 2.5f);
  run_saxpy(*ctx_, k, 4096, -0.75f);  // exact multiple
  run_saxpy(*ctx_, k, 1, 3.0f);       // single element, single group
}

TEST_F(LaunchTest, VecAddEndToEnd) {
  vulkore::Kernel k = load("vec_add");
  const uint32_t n = 100003;  // prime, way off any workgroup multiple
  std::vector<int32_t> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<int32_t>(i * 2654435761u);
    b[i] = -static_cast<int32_t>(i) * 7 + 13;
    expected[i] = a[i] + b[i];  // wraparound identical on GPU (32-bit)
  }
  vulkore::Buffer ab = ctx_->alloc<int32_t>(n);
  vulkore::Buffer bb = ctx_->alloc<int32_t>(n);
  vulkore::Buffer ob = ctx_->alloc<int32_t>(n);
  ab.upload(std::span<const int32_t>(a));
  bb.upload(std::span<const int32_t>(b));

  vulkore::launch(k, {n}, ob, ab, bb, n).wait();

  std::vector<int32_t> result(n);
  ob.download(std::span<int32_t>(result));
  ASSERT_EQ(result, expected);
}

TEST_F(LaunchTest, RoundUpNeverWritesPastN) {
  // Round-up dispatches extra threads; the kernel's `i < n` guard must keep
  // them away from data. Detectable here because the buffer is longer than n.
  vulkore::Kernel k = load("scale_inplace");
  const uint32_t n = 1000, slack = 64;
  std::vector<float> data(n + slack);
  for (uint32_t i = 0; i < n + slack; ++i) data[i] = static_cast<float>(i);

  vulkore::Buffer buf = ctx_->alloc<float>(n + slack);
  buf.upload(std::span<const float>(data));
  vulkore::launch(k, {n}, buf, 2.0f, n).wait();

  std::vector<float> result(n + slack);
  buf.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) {
    ASSERT_EQ(result[i], 2.0f * data[i]) << "i=" << i;  // exact: *2 is exact
  }
  for (uint32_t i = n; i < n + slack; ++i) {
    ASSERT_EQ(result[i], data[i]) << "guard failed at i=" << i;
  }
}

TEST_F(LaunchTest, GridSweepNonMultiples) {
  vulkore::Kernel k = load("scale_inplace");
  for (uint32_t n : {1u, 7u, 63u, 64u, 65u, 128u, 1000u}) {
    SCOPED_TRACE("n=" + std::to_string(n));
    std::vector<float> data(n);
    for (uint32_t i = 0; i < n; ++i) data[i] = static_cast<float>(i) + 0.5f;
    vulkore::Buffer buf = ctx_->alloc<float>(n);
    buf.upload(std::span<const float>(data));
    vulkore::launch(k, {n}, buf, 4.0f, n).wait();
    std::vector<float> result(n);
    buf.download(std::span<float>(result));
    for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 4.0f * data[i]);
  }
}

TEST_F(LaunchTest, ChainedLaunchesSameBuffer) {
  // dispatch -> dispatch on the same buffer exercises the pipeline barriers.
  vulkore::Kernel k = load("scale_inplace");
  const uint32_t n = 1000;
  std::vector<float> data(n, 1.0f);
  vulkore::Buffer buf = ctx_->alloc<float>(n);
  buf.upload(std::span<const float>(data));

  vulkore::Fence f1 = vulkore::launch(k, {n}, buf, 2.0f, n);
  vulkore::Fence f2 = vulkore::launch(k, {n}, buf, 3.0f, n);  // no wait between
  EXPECT_TRUE(f2.wait());
  EXPECT_TRUE(f1.is_signaled());  // same queue, earlier submission

  std::vector<float> result(n);
  buf.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 6.0f);
}

// ---- comparison kernel: true AND false outcomes + IEEE edge cases ----------

TEST_F(LaunchTest, CompareIeeeEdgeCases) {
  vulkore::Kernel k = load("compare_fp");
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();

  // The three canonical bitwise-comparison traps from testing-plan.md:
  //   -0.0 == +0.0 (bit patterns differ), NaN != NaN (bit patterns equal),
  //   -2.0 < -1.0 (integer compare of sign-magnitude gets negatives backwards)
  // plus plain true/false cases for both operators.
  const std::vector<float> a = {-0.0f, nan, -2.0f, 1.0f, 3.0f,
                                2.0f, -1.0f, 0.0f, inf, -inf};
  const std::vector<float> b = {+0.0f, nan, -1.0f, 2.0f, 3.0f,
                                1.0f, -2.0f, 1.0f, inf, inf};
  const uint32_t n = static_cast<uint32_t>(a.size());

  // CPU reference computed here, not hard-coded (host FPU is IEEE-754).
  std::vector<int32_t> eq_expected(n), lt_expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    eq_expected[i] = a[i] == b[i] ? 1 : 0;
    lt_expected[i] = a[i] < b[i] ? 1 : 0;
  }
  // Sanity-check the traps really are in the set (guards test rot).
  ASSERT_EQ(eq_expected[0], 1);  // -0.0 == +0.0
  ASSERT_EQ(eq_expected[1], 0);  // NaN != NaN
  ASSERT_EQ(lt_expected[2], 1);  // -2.0 < -1.0
  ASSERT_EQ(lt_expected[6], 0);  // -1.0 < -2.0 is false

  vulkore::Buffer ab = ctx_->alloc<float>(n);
  vulkore::Buffer bb = ctx_->alloc<float>(n);
  vulkore::Buffer eqb = ctx_->alloc<int32_t>(n);
  vulkore::Buffer ltb = ctx_->alloc<int32_t>(n);
  ab.upload(std::span<const float>(a));
  bb.upload(std::span<const float>(b));

  vulkore::launch(k, {n}, eqb, ltb, ab, bb, n).wait();

  std::vector<int32_t> eq(n), lt(n);
  eqb.download(std::span<int32_t>(eq));
  ltb.download(std::span<int32_t>(lt));
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_EQ(eq[i], eq_expected[i]) << "== failed for a=" << a[i] << " b=" << b[i];
    EXPECT_EQ(lt[i], lt_expected[i]) << "< failed for a=" << a[i] << " b=" << b[i];
  }
}

// ---- argument validation ----------------------------------------------------

TEST_F(LaunchTest, WrongArgCountThrows) {
  vulkore::Kernel k = load("saxpy");
  vulkore::Buffer buf = ctx_->alloc<float>(16);
  try {
    vulkore::launch(k, {16}, buf, buf, 1.0f);  // missing n
    FAIL() << "expected vulkore::Error";
  } catch (const vulkore::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("expects 4"), std::string::npos) << msg;
    EXPECT_NE(msg.find("got 3"), std::string::npos) << msg;
  }
}

TEST_F(LaunchTest, PodWhereBufferExpectedThrows) {
  vulkore::Kernel k = load("saxpy");
  vulkore::Buffer buf = ctx_->alloc<float>(16);
  try {
    vulkore::launch(k, {16}, buf, 5.0f, 1.0f, 16u);  // arg 1 must be a Buffer
    FAIL() << "expected vulkore::Error";
  } catch (const vulkore::Error& e) {
    EXPECT_NE(std::string(e.what()).find("'x'"), std::string::npos) << e.what();
  }
}

TEST_F(LaunchTest, BufferWherePodExpectedThrows) {
  vulkore::Kernel k = load("saxpy");
  vulkore::Buffer buf = ctx_->alloc<float>(16);
  try {
    vulkore::launch(k, {16}, buf, buf, buf, 16u);  // arg 2 must be a POD float
    FAIL() << "expected vulkore::Error";
  } catch (const vulkore::Error& e) {
    EXPECT_NE(std::string(e.what()).find("'a'"), std::string::npos) << e.what();
  }
}

TEST_F(LaunchTest, PodSizeMismatchThrows) {
  vulkore::Kernel k = load("saxpy");
  vulkore::Buffer buf = ctx_->alloc<float>(16);
  try {
    vulkore::launch(k, {16}, buf, buf, 2.0 /* double! */, 16u);
    FAIL() << "expected vulkore::Error";
  } catch (const vulkore::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("'a'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("8-byte"), std::string::npos) << msg;
  }
}

TEST_F(LaunchTest, ZeroGridThrows) {
  vulkore::Kernel k = load("saxpy");
  vulkore::Buffer buf = ctx_->alloc<float>(16);
  EXPECT_THROW(vulkore::launch(k, {0}, buf, buf, 1.0f, 16u), vulkore::Error);
}

TEST_F(LaunchTest, EmptyBufferArgThrows) {
  vulkore::Kernel k = load("saxpy");
  vulkore::Buffer good = ctx_->alloc<float>(16);
  vulkore::Buffer empty;  // default-constructed
  EXPECT_THROW(vulkore::launch(k, {16}, good, empty, 1.0f, 16u), vulkore::Error);
}

// ---- resource recycling under sustained launching ---------------------------

TEST_F(LaunchTest, ManyLaunchesSteadyState) {
  // 300 launches > descriptor-pool capacity (128 sets/pool): passes only if
  // the Fence completion hook really recycles sets and command buffers, or
  // pool growth works. Both paths are exercised: waits recycle, the burst
  // below outruns recycling.
  vulkore::Kernel k = load("scale_inplace");
  const uint32_t n = 64;
  std::vector<float> data(n, 1.0f);
  vulkore::Buffer buf = ctx_->alloc<float>(n);

  for (int round = 0; round < 3; ++round) {
    buf.upload(std::span<const float>(data));
    std::vector<vulkore::Fence> in_flight;
    for (int i = 0; i < 100; ++i) {
      in_flight.push_back(vulkore::launch(k, {n}, buf, 1.0f, n));  // x*1
    }
    for (auto& f : in_flight) EXPECT_TRUE(f.wait());
    std::vector<float> result(n);
    buf.download(std::span<float>(result));
    for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 1.0f);
  }
}

TEST_F(LaunchTest, UnwaitedFenceDrainsOnDestruction) {
  vulkore::Kernel k = load("scale_inplace");
  const uint32_t n = 1000;
  std::vector<float> data(n, 3.0f);
  vulkore::Buffer buf = ctx_->alloc<float>(n);
  buf.upload(std::span<const float>(data));
  {
    vulkore::Fence f = vulkore::launch(k, {n}, buf, 2.0f, n);
    // dropped without wait(): destructor must drain + recycle
  }
  std::vector<float> result(n);
  buf.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 6.0f);
}

TEST_F(LaunchTest, ForcedStagingTransfersAroundLaunch) {
  // Force the staging path (vkCmdCopyBuffer) on both sides of a dispatch so
  // the transfer->compute->transfer barriers get exercised even on UMA.
  vulkore::Kernel k = load("vec_add");
  const uint32_t n = 1000;
  std::vector<int32_t> a(n), b(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<int32_t>(i);
    b[i] = 2 * static_cast<int32_t>(i) + 1;
  }
  vulkore::Buffer ab = ctx_->alloc<int32_t>(n);
  vulkore::Buffer bb = ctx_->alloc<int32_t>(n);
  vulkore::Buffer ob = ctx_->alloc<int32_t>(n);
  ab.set_force_staging(true);
  bb.set_force_staging(true);
  ob.set_force_staging(true);
  ab.upload(std::span<const int32_t>(a));
  bb.upload(std::span<const int32_t>(b));

  vulkore::launch(k, {n}, ob, ab, bb, n).wait();

  std::vector<int32_t> result(n);
  ob.download(std::span<int32_t>(result));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], a[i] + b[i]);
}

}  // namespace
