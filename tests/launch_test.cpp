// launch(): end-to-end kernel execution against CPU references, dispatch
// round-up geometry, argument mismatch diagnostics, IEEE comparison edge
// cases (testing-plan.md Layer 1; pattern inherited from dcompute_cmptest).
#include <xpose/xpose.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

std::string kernel_path(const std::string& name) {
  const char* env_dir = std::getenv("XPOSE_KERNEL_DIR");
  std::string dir = env_dir ? env_dir : XPOSE_KERNEL_DIR;
  return dir + "/" + name + ".spv";
}

class LaunchTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<xpose::Context>(); }
  static void TearDownTestSuite() {
    // Programs own Vulkan objects (pipelines/layouts/shader module) tied to
    // ctx_'s VkDevice and hold a raw Context* — they MUST be destroyed while
    // that Context is still alive. Clearing here (before ctx_.reset(), and
    // before process-exit static teardown) upholds the documented
    // "Program must not outlive its Context" invariant; otherwise these
    // statics destruct after main returns against a freed device (SIGSEGV).
    programs_.clear();
    ctx_.reset();
  }

  static xpose::Kernel load(const std::string& name) {
    programs_.push_back(std::make_unique<xpose::Program>(
        xpose::Program::from_file(*ctx_, kernel_path(name))));
    return programs_.back()->kernel(name);
  }

  static std::unique_ptr<xpose::Context> ctx_;
  static std::vector<std::unique_ptr<xpose::Program>> programs_;
};
std::unique_ptr<xpose::Context> LaunchTest::ctx_;
std::vector<std::unique_ptr<xpose::Program>> LaunchTest::programs_;

// ---- end-to-end: upload -> launch -> download vs CPU reference -------------

void run_saxpy(xpose::Context& ctx, xpose::Kernel& k, uint32_t n, float a) {
  SCOPED_TRACE("saxpy n=" + std::to_string(n));
  std::vector<float> x(n), y(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = 0.25f * static_cast<float>(i) - 100.0f;
    y[i] = 1.5f * static_cast<float>(i % 97);
  }
  std::vector<float> expected(n);
  for (uint32_t i = 0; i < n; ++i) expected[i] = a * x[i] + y[i];

  xpose::Buffer xb = ctx.alloc<float>(n);
  xpose::Buffer yb = ctx.alloc<float>(n);
  xb.upload(std::span<const float>(x));
  yb.upload(std::span<const float>(y));

  xpose::Fence f = xpose::launch(k, {n}, yb, xb, a, n);
  EXPECT_TRUE(f.wait());

  std::vector<float> result(n);
  yb.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) {
    ASSERT_FLOAT_EQ(result[i], expected[i]) << "i=" << i;
  }
}

TEST_F(LaunchTest, SaxpyEndToEnd) {
  xpose::Kernel k = load("saxpy");
  // 1000 is deliberately not a multiple of the 64-wide workgroup (round-up).
  run_saxpy(*ctx_, k, 1000, 2.5f);
  run_saxpy(*ctx_, k, 4096, -0.75f);  // exact multiple
  run_saxpy(*ctx_, k, 1, 3.0f);       // single element, single group
}

TEST_F(LaunchTest, VecAddEndToEnd) {
  xpose::Kernel k = load("vec_add");
  const uint32_t n = 100003;  // prime, way off any workgroup multiple
  std::vector<int32_t> a(n), b(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<int32_t>(i * 2654435761u);
    b[i] = -static_cast<int32_t>(i) * 7 + 13;
    expected[i] = a[i] + b[i];  // wraparound identical on GPU (32-bit)
  }
  xpose::Buffer ab = ctx_->alloc<int32_t>(n);
  xpose::Buffer bb = ctx_->alloc<int32_t>(n);
  xpose::Buffer ob = ctx_->alloc<int32_t>(n);
  ab.upload(std::span<const int32_t>(a));
  bb.upload(std::span<const int32_t>(b));

  xpose::launch(k, {n}, ob, ab, bb, n).wait();

  std::vector<int32_t> result(n);
  ob.download(std::span<int32_t>(result));
  ASSERT_EQ(result, expected);
}

TEST_F(LaunchTest, RoundUpNeverWritesPastN) {
  // Round-up dispatches extra threads; the kernel's `i < n` guard must keep
  // them away from data. Detectable here because the buffer is longer than n.
  xpose::Kernel k = load("scale_inplace");
  const uint32_t n = 1000, slack = 64;
  std::vector<float> data(n + slack);
  for (uint32_t i = 0; i < n + slack; ++i) data[i] = static_cast<float>(i);

  xpose::Buffer buf = ctx_->alloc<float>(n + slack);
  buf.upload(std::span<const float>(data));
  xpose::launch(k, {n}, buf, 2.0f, n).wait();

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
  xpose::Kernel k = load("scale_inplace");
  for (uint32_t n : {1u, 7u, 63u, 64u, 65u, 128u, 1000u}) {
    SCOPED_TRACE("n=" + std::to_string(n));
    std::vector<float> data(n);
    for (uint32_t i = 0; i < n; ++i) data[i] = static_cast<float>(i) + 0.5f;
    xpose::Buffer buf = ctx_->alloc<float>(n);
    buf.upload(std::span<const float>(data));
    xpose::launch(k, {n}, buf, 4.0f, n).wait();
    std::vector<float> result(n);
    buf.download(std::span<float>(result));
    for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 4.0f * data[i]);
  }
}

TEST_F(LaunchTest, ChainedLaunchesSameBuffer) {
  // dispatch -> dispatch on the same buffer exercises the pipeline barriers.
  xpose::Kernel k = load("scale_inplace");
  const uint32_t n = 1000;
  std::vector<float> data(n, 1.0f);
  xpose::Buffer buf = ctx_->alloc<float>(n);
  buf.upload(std::span<const float>(data));

  xpose::Fence f1 = xpose::launch(k, {n}, buf, 2.0f, n);
  xpose::Fence f2 = xpose::launch(k, {n}, buf, 3.0f, n);  // no wait between
  EXPECT_TRUE(f2.wait());
  EXPECT_TRUE(f1.is_signaled());  // same queue, earlier submission

  std::vector<float> result(n);
  buf.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 6.0f);
}

// ---- comparison kernel: true AND false outcomes + IEEE edge cases ----------

TEST_F(LaunchTest, CompareIeeeEdgeCases) {
  xpose::Kernel k = load("compare_fp");
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

  xpose::Buffer ab = ctx_->alloc<float>(n);
  xpose::Buffer bb = ctx_->alloc<float>(n);
  xpose::Buffer eqb = ctx_->alloc<int32_t>(n);
  xpose::Buffer ltb = ctx_->alloc<int32_t>(n);
  ab.upload(std::span<const float>(a));
  bb.upload(std::span<const float>(b));

  xpose::launch(k, {n}, eqb, ltb, ab, bb, n).wait();

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
  xpose::Kernel k = load("saxpy");
  xpose::Buffer buf = ctx_->alloc<float>(16);
  try {
    xpose::launch(k, {16}, buf, buf, 1.0f);  // missing n
    FAIL() << "expected xpose::Error";
  } catch (const xpose::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("expects 4"), std::string::npos) << msg;
    EXPECT_NE(msg.find("got 3"), std::string::npos) << msg;
  }
}

TEST_F(LaunchTest, PodWhereBufferExpectedThrows) {
  xpose::Kernel k = load("saxpy");
  xpose::Buffer buf = ctx_->alloc<float>(16);
  try {
    xpose::launch(k, {16}, buf, 5.0f, 1.0f, 16u);  // arg 1 must be a Buffer
    FAIL() << "expected xpose::Error";
  } catch (const xpose::Error& e) {
    EXPECT_NE(std::string(e.what()).find("'x'"), std::string::npos) << e.what();
  }
}

TEST_F(LaunchTest, BufferWherePodExpectedThrows) {
  xpose::Kernel k = load("saxpy");
  xpose::Buffer buf = ctx_->alloc<float>(16);
  try {
    xpose::launch(k, {16}, buf, buf, buf, 16u);  // arg 2 must be a POD float
    FAIL() << "expected xpose::Error";
  } catch (const xpose::Error& e) {
    EXPECT_NE(std::string(e.what()).find("'a'"), std::string::npos) << e.what();
  }
}

TEST_F(LaunchTest, PodSizeMismatchThrows) {
  xpose::Kernel k = load("saxpy");
  xpose::Buffer buf = ctx_->alloc<float>(16);
  try {
    xpose::launch(k, {16}, buf, buf, 2.0 /* double! */, 16u);
    FAIL() << "expected xpose::Error";
  } catch (const xpose::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("'a'"), std::string::npos) << msg;
    EXPECT_NE(msg.find("8-byte"), std::string::npos) << msg;
  }
}

TEST_F(LaunchTest, ZeroGridThrows) {
  xpose::Kernel k = load("saxpy");
  xpose::Buffer buf = ctx_->alloc<float>(16);
  EXPECT_THROW(xpose::launch(k, {0}, buf, buf, 1.0f, 16u), xpose::Error);
}

TEST_F(LaunchTest, EmptyBufferArgThrows) {
  xpose::Kernel k = load("saxpy");
  xpose::Buffer good = ctx_->alloc<float>(16);
  xpose::Buffer empty;  // default-constructed
  EXPECT_THROW(xpose::launch(k, {16}, good, empty, 1.0f, 16u), xpose::Error);
}

// ---- resource recycling under sustained launching ---------------------------

TEST_F(LaunchTest, ManyLaunchesSteadyState) {
  // 300 launches > descriptor-pool capacity (128 sets/pool): passes only if
  // the Fence completion hook really recycles sets and command buffers, or
  // pool growth works. Both paths are exercised: waits recycle, the burst
  // below outruns recycling.
  xpose::Kernel k = load("scale_inplace");
  const uint32_t n = 64;
  std::vector<float> data(n, 1.0f);
  xpose::Buffer buf = ctx_->alloc<float>(n);

  for (int round = 0; round < 3; ++round) {
    buf.upload(std::span<const float>(data));
    std::vector<xpose::Fence> in_flight;
    for (int i = 0; i < 100; ++i) {
      in_flight.push_back(xpose::launch(k, {n}, buf, 1.0f, n));  // x*1
    }
    for (auto& f : in_flight) EXPECT_TRUE(f.wait());
    std::vector<float> result(n);
    buf.download(std::span<float>(result));
    for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 1.0f);
  }
}

TEST_F(LaunchTest, UnwaitedFenceDrainsOnDestruction) {
  xpose::Kernel k = load("scale_inplace");
  const uint32_t n = 1000;
  std::vector<float> data(n, 3.0f);
  xpose::Buffer buf = ctx_->alloc<float>(n);
  buf.upload(std::span<const float>(data));
  {
    xpose::Fence f = xpose::launch(k, {n}, buf, 2.0f, n);
    // dropped without wait(): destructor must drain + recycle
  }
  std::vector<float> result(n);
  buf.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], 6.0f);
}

TEST_F(LaunchTest, ForcedStagingTransfersAroundLaunch) {
  // Force the staging path (vkCmdCopyBuffer) on both sides of a dispatch so
  // the transfer->compute->transfer barriers get exercised even on UMA.
  xpose::Kernel k = load("vec_add");
  const uint32_t n = 1000;
  std::vector<int32_t> a(n), b(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<int32_t>(i);
    b[i] = 2 * static_cast<int32_t>(i) + 1;
  }
  xpose::Buffer ab = ctx_->alloc<int32_t>(n);
  xpose::Buffer bb = ctx_->alloc<int32_t>(n);
  xpose::Buffer ob = ctx_->alloc<int32_t>(n);
  ab.set_force_staging(true);
  bb.set_force_staging(true);
  ob.set_force_staging(true);
  ab.upload(std::span<const int32_t>(a));
  bb.upload(std::span<const int32_t>(b));

  xpose::launch(k, {n}, ob, ab, bb, n).wait();

  std::vector<int32_t> result(n);
  ob.download(std::span<int32_t>(result));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(result[i], a[i] + b[i]);
}

// ---- cross-pipeline dispatch->dispatch barrier ------------------------------

TEST_F(LaunchTest, CrossKernelRawChain) {
  // ChainedLaunchesSameBuffer proves the dispatch->dispatch barrier for the
  // SAME kernel; this proves it ACROSS DIFFERENT pipelines/layouts. saxpy
  // (2 buffers + 2 POD) writes yb, then WITHOUT a host wait scale_inplace
  // (1 buffer + 2 POD, a different pipeline) reads+writes the same yb. If the
  // in-cmdbuf barriers didn't cover cross-pipeline hazards, scale would read
  // stale data. Both float, so the CPU reference is exact-ish (FLOAT_EQ).
  xpose::Kernel saxpy = load("saxpy");
  xpose::Kernel scale = load("scale_inplace");
  const uint32_t n = 1000;  // not a workgroup multiple (round-up path)
  const float a = 2.5f, factor = -4.0f;
  std::vector<float> x(n), y0(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = 0.5f * static_cast<float>(i % 37) - 3.0f;
    y0[i] = static_cast<float>(i % 11);
  }
  xpose::Buffer xb = ctx_->alloc<float>(n);
  xpose::Buffer yb = ctx_->alloc<float>(n);
  xb.upload(std::span<const float>(x));
  yb.upload(std::span<const float>(y0));

  xpose::Fence f1 = xpose::launch(saxpy, {n}, yb, xb, a, n);   // yb = a*x + y0
  xpose::Fence f2 = xpose::launch(scale, {n}, yb, factor, n);  // yb *= factor
  EXPECT_TRUE(f2.wait());
  EXPECT_TRUE(f1.is_signaled());  // earlier submission on the same queue

  std::vector<float> result(n);
  yb.download(std::span<float>(result));
  for (uint32_t i = 0; i < n; ++i) {
    const float expected = (a * x[i] + y0[i]) * factor;
    ASSERT_FLOAT_EQ(result[i], expected) << "i=" << i;
  }
}

TEST_F(LaunchTest, InterleavedDifferentKernels) {
  // Interleave two DIFFERENT kernels (different pipelines, descriptor-set
  // layouts, and push-constant sizes) with no waits between, so the
  // descriptor-set / command-buffer recycler juggles mixed layouts in flight.
  // ManyLaunchesSteadyState stresses recycling but with ONE pipeline; this
  // adds the mixed-layout dimension. Concurrency is kept at 100 (the count
  // already proven safe by ManyLaunchesSteadyState) to stay non-flaky.
  xpose::Kernel scale = load("scale_inplace");  // 1 buffer + 2 POD, 8B push
  xpose::Kernel add = load("vec_add");          // 3 buffers + 1 POD, 4B push
  const uint32_t n = 256;
  const int K = 50;  // 2*K = 100 launches in flight

  xpose::Buffer sf = ctx_->alloc<float>(n);
  std::vector<float> ones(n, 1.0f);
  sf.upload(std::span<const float>(ones));

  std::vector<int32_t> a(n), b(n), expected_add(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<int32_t>(i) * 3 - 5;
    b[i] = static_cast<int32_t>(i) + 7;
    expected_add[i] = a[i] + b[i];
  }
  xpose::Buffer ab = ctx_->alloc<int32_t>(n);
  xpose::Buffer bb = ctx_->alloc<int32_t>(n);
  xpose::Buffer ob = ctx_->alloc<int32_t>(n);
  ab.upload(std::span<const int32_t>(a));
  bb.upload(std::span<const int32_t>(b));

  // Alternate *2 / *0.5 so the running product stays in {0.5,1,2} (all exact)
  // AND ordering matters: a broken chain on sf would apply a factor to a stale
  // base and drift off the reference.
  float running = 1.0f;
  std::vector<xpose::Fence> fences;
  for (int it = 0; it < K; ++it) {
    const float factor = (it % 2 == 0) ? 2.0f : 0.5f;
    running *= factor;
    fences.push_back(xpose::launch(scale, {n}, sf, factor, n));
    fences.push_back(xpose::launch(add, {n}, ob, ab, bb, n));  // idempotent
  }
  for (auto& f : fences) EXPECT_TRUE(f.wait());

  std::vector<float> sresult(n);
  sf.download(std::span<float>(sresult));
  for (uint32_t i = 0; i < n; ++i) ASSERT_EQ(sresult[i], running) << "i=" << i;

  std::vector<int32_t> aresult(n);
  ob.download(std::span<int32_t>(aresult));
  ASSERT_EQ(aresult, expected_add);
}

// ---- grid geometry validation -----------------------------------------------

TEST_F(LaunchTest, GridTooLargeThrows) {
  // grid.x that needs more workgroups than maxComputeWorkGroupCount[0]
  // (launch.cpp:29-33). The limit is a real device query, so on drivers where
  // it is too high to exceed within a uint32 grid (e.g. NVIDIA at 2^31-1) we
  // GTEST_SKIP rather than fail — keeps the same binary green on the phone too.
  xpose::Kernel k = load("saxpy");
  const uint32_t local_x = k.local_size()[0];  // 64 by default
  const uint32_t max_groups =
      ctx_->device_properties().limits.maxComputeWorkGroupCount[0];
  // Smallest grid.x whose ceil(grid.x/local_x) == max_groups + 1 (> the limit).
  const uint64_t needed =
      (static_cast<uint64_t>(max_groups) + 1u) * static_cast<uint64_t>(local_x);
  if (needed > 0xFFFFFFFFull) {
    GTEST_SKIP() << "device '" << ctx_->device_name()
                 << "' maxComputeWorkGroupCount[0]=" << max_groups
                 << " is too high to exceed within a uint32 grid";
  }
  const uint32_t grid_x = static_cast<uint32_t>(needed);
  xpose::Buffer xb = ctx_->alloc<float>(16);
  xpose::Buffer yb = ctx_->alloc<float>(16);
  try {
    xpose::launch(k, {grid_x}, yb, xb, 1.0f, 16u);
    FAIL() << "expected throw for grid.x=" << grid_x;
  } catch (const xpose::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("workgroups"), std::string::npos) << msg;
    EXPECT_NE(msg.find("device limit is"), std::string::npos) << msg;
  }
}

// Generalizes GridTooLargeThrows (x-only) to all three axes. For each axis the
// per-axis maxComputeWorkGroupCount is queried at runtime; where it is too high
// to exceed within a uint32 grid (Mali/RADV: UINT32_MAX on every dim) that axis
// is skipped, otherwise a grid that needs one workgroup more than the limit is
// built and launch() must throw naming that axis (guards launch.cpp ~29-33, one
// group_count() call per axis at ~95-100). Runs meaningfully on NVIDIA (y/z=
// 65535; x skipped) and llvmpipe (all three = 65535); skips wholesale on
// Mali/RADV. If no axis is exercisable the whole test skips (never a false pass).
TEST_F(LaunchTest, GridTooLargeThrowsAllAxes) {
  xpose::Kernel k = load("saxpy");
  const std::array<uint32_t, 3> local = k.local_size();  // 64x1x1 by default
  const VkPhysicalDeviceLimits& lim = ctx_->device_properties().limits;
  xpose::Buffer xb = ctx_->alloc<float>(16);
  xpose::Buffer yb = ctx_->alloc<float>(16);

  int tested = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const char axis_ch = "xyz"[axis];
    SCOPED_TRACE(std::string("axis ") + axis_ch);
    const uint32_t local_a = local[axis];
    const uint32_t max_groups = lim.maxComputeWorkGroupCount[axis];
    // Smallest global thread count on this axis whose ceil(global/local) is
    // max_groups + 1 (one workgroup past the limit).
    const uint64_t needed =
        (static_cast<uint64_t>(max_groups) + 1u) * static_cast<uint64_t>(local_a);
    if (needed > 0xFFFFFFFFull) {
      continue;  // can't exceed this axis within a uint32 grid (Mali/RADV)
    }
    const uint32_t g = static_cast<uint32_t>(needed);

    // Keep the other two axes tiny (1 workgroup) so only the target axis trips.
    xpose::Grid grid;  // {1,1,1}
    grid.x = 16;
    switch (axis) {
      case 0: grid.x = g; break;
      case 1: grid.y = g; break;
      case 2: grid.z = g; break;
    }
    try {
      xpose::launch(k, grid, yb, xb, 1.0f, 16u);
      ADD_FAILURE() << "expected throw for grid." << axis_ch << '=' << g;
    } catch (const xpose::Error& e) {
      const std::string msg = e.what();
      EXPECT_NE(msg.find(std::string("grid.") + axis_ch), std::string::npos) << msg;
      EXPECT_NE(msg.find("workgroups"), std::string::npos) << msg;
      EXPECT_NE(msg.find("device limit is"), std::string::npos) << msg;
    }
    ++tested;
  }
  if (tested == 0) {
    GTEST_SKIP() << "device '" << ctx_->device_name()
                 << "' maxComputeWorkGroupCount is too high on all three axes "
                    "to exceed within a uint32 grid (Mali/RADV: UINT32_MAX)";
  }
}

TEST_F(LaunchTest, ZeroGridYZThrows) {
  // ZeroGridThrows only covers x==0; y and z go through the same guard
  // (launch.cpp:22-25, one call per axis) and must throw too.
  xpose::Kernel k = load("saxpy");
  xpose::Buffer xb = ctx_->alloc<float>(16);
  xpose::Buffer yb = ctx_->alloc<float>(16);
  {
    SCOPED_TRACE("grid.y == 0");
    try {
      xpose::launch(k, {16, 0}, yb, xb, 1.0f, 16u);
      FAIL() << "expected throw for grid.y==0";
    } catch (const xpose::Error& e) {
      const std::string msg = e.what();
      EXPECT_NE(msg.find("grid.y"), std::string::npos) << msg;
      EXPECT_NE(msg.find("nonzero"), std::string::npos) << msg;
    }
  }
  {
    SCOPED_TRACE("grid.z == 0");
    try {
      xpose::launch(k, {16, 1, 0}, yb, xb, 1.0f, 16u);
      FAIL() << "expected throw for grid.z==0";
    } catch (const xpose::Error& e) {
      const std::string msg = e.what();
      EXPECT_NE(msg.find("grid.z"), std::string::npos) << msg;
      EXPECT_NE(msg.find("nonzero"), std::string::npos) << msg;
    }
  }
}

}  // namespace
