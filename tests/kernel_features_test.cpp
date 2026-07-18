// New-fixture coverage (2026-07-18): a real 2-D kernel (grid.y > 1 with actual
// data — the biggest prior functional gap), a reqd_work_group_size kernel
// (PropertyRequiredWorkgroupSize reflection + baked local size), a two-entry
// multi-kernel module, and a mixed-scalar-POD push-constant-packing kernel.
// Every fixture gets a reflection assertion and an end-to-end dispatch checked
// against a CPU reference (testing-plan.md Layer 1). Fixtures + reflection
// details: tests/kernels/README.md.
#include <xpose/xpose.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
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

// Shared Context + Program ownership. Programs own Vulkan objects tied to ctx_'s
// VkDevice and hold a raw Context*, so they MUST be destroyed while the Context
// is alive: TearDownTestSuite clears programs_ BEFORE ctx_.reset() (the
// exit-139 UAF lesson from agent-docs/exit-teardown-fix.md).
class KernelFeaturesTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<xpose::Context>(); }
  static void TearDownTestSuite() {
    programs_.clear();
    ctx_.reset();
  }

  // Loads a module and keeps it alive; returns a reference (stable — the
  // unique_ptr's pointee address does not move on vector growth).
  static xpose::Program& load_program(const std::string& name) {
    programs_.push_back(std::make_unique<xpose::Program>(
        xpose::Program::from_file(*ctx_, kernel_path(name))));
    return *programs_.back();
  }
  static xpose::Kernel load(const std::string& name) {
    return load_program(name).kernel(name);
  }

  static std::unique_ptr<xpose::Context> ctx_;
  static std::vector<std::unique_ptr<xpose::Program>> programs_;
};
std::unique_ptr<xpose::Context> KernelFeaturesTest::ctx_;
std::vector<std::unique_ptr<xpose::Program>> KernelFeaturesTest::programs_;

using xpose::ArgKind;
using xpose::KernelArg;

void expect_buffer_arg(const KernelArg& a, uint32_t ordinal, uint32_t set,
                       uint32_t binding, const std::string& name) {
  EXPECT_EQ(a.ordinal, ordinal);
  EXPECT_EQ(a.kind, ArgKind::StorageBuffer);
  EXPECT_EQ(a.set, set);
  EXPECT_EQ(a.binding, binding);
  EXPECT_EQ(a.name, name);
}

void expect_pod_arg(const KernelArg& a, uint32_t ordinal, uint32_t offset,
                    uint32_t size, const std::string& name) {
  EXPECT_EQ(a.ordinal, ordinal);
  EXPECT_EQ(a.kind, ArgKind::PodPushConstant);
  EXPECT_EQ(a.offset, offset);
  EXPECT_EQ(a.size, size);
  EXPECT_EQ(a.name, name);
}

// ======================= transpose_2d (2-D grid) =============================

TEST_F(KernelFeaturesTest, Transpose2dReflection) {
  xpose::Program& prog = load_program("transpose_2d");
  EXPECT_EQ(prog.kernel_names(), std::vector<std::string>{"transpose_2d"});

  xpose::Kernel k = prog.kernel("transpose_2d");
  const xpose::KernelInfo& info = k.info();
  ASSERT_EQ(info.args.size(), 4u);
  expect_buffer_arg(info.args[0], 0, 0, 0, "out");
  expect_buffer_arg(info.args[1], 1, 0, 1, "in");
  expect_pod_arg(info.args[2], 2, 0, 4, "width");
  expect_pod_arg(info.args[3], 3, 4, 4, "height");
  EXPECT_EQ(info.push_constant_size, 8u);
  // No reqd attribute -> default workgroup size via spec constants.
  EXPECT_FALSE(info.has_reqd_workgroup_size);
  EXPECT_EQ(k.local_size(), (std::array<uint32_t, 3>{64, 1, 1}));
}

// Priority test: dispatch with width != height and non-power-of-two dims so the
// (CUDA-style) 2-D grid drives grid.y > 1 with REAL data for the first time and
// rounds grid.x up past a whole workgroup (local {64,1,1}); the kernel's
// x<width && y<height guard must keep the overflow invocations off the data.
TEST_F(KernelFeaturesTest, Grid2DEndToEnd) {
  xpose::Kernel k = load("transpose_2d");

  // {width,height}: small single-x-group (37x53), multi-group in BOTH axes
  // (131x97, 131>2*64), an x-exact-multiple edge (128x30), and a tall/thin one.
  const std::array<std::pair<uint32_t, uint32_t>, 4> dims = {
      {{37u, 53u}, {131u, 97u}, {128u, 30u}, {3u, 200u}}};

  for (auto [w, h] : dims) {
    SCOPED_TRACE("width=" + std::to_string(w) + " height=" +
                 std::to_string(h));
    std::vector<float> in(static_cast<size_t>(w) * h);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        // Distinct, exactly-representable value per cell (w*h < 2^24).
        in[static_cast<size_t>(y) * w + x] =
            static_cast<float>(y * w + x);
      }
    }
    // CPU reference: out is h x w row-major, out[x*h + y] = in[y*w + x].
    std::vector<float> expected(static_cast<size_t>(w) * h);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        expected[static_cast<size_t>(x) * h + y] = in[static_cast<size_t>(y) * w + x];
      }
    }

    xpose::Buffer inb = ctx_->alloc<float>(in.size());
    xpose::Buffer outb = ctx_->alloc<float>(in.size());
    inb.upload(std::span<const float>(in));

    xpose::launch(k, {w, h}, outb, inb, w, h).wait();

    std::vector<float> result(in.size());
    outb.download(std::span<float>(result));
    for (size_t i = 0; i < result.size(); ++i) {
      ASSERT_EQ(result[i], expected[i]) << "linear index " << i;  // exact: moved
    }
  }
}

// ======================= reqd_wgsize (reqd_work_group_size) ===================

TEST_F(KernelFeaturesTest, ReqdWorkGroupSize) {
  xpose::Program& prog = load_program("reqd_wgsize");
  xpose::Kernel k = prog.kernel("reqd_wgsize");

  const xpose::KernelInfo& info = k.info();
  ASSERT_EQ(info.args.size(), 3u);
  expect_buffer_arg(info.args[0], 0, 0, 0, "out");
  expect_buffer_arg(info.args[1], 1, 0, 1, "in");
  expect_pod_arg(info.args[2], 2, 0, 4, "n");
  EXPECT_EQ(info.push_constant_size, 4u);

  // The reqd_work_group_size(32,1,1) attribute -> PropertyRequiredWorkgroupSize
  // reflection -> baked into the pipeline instead of the default 64x1x1.
  EXPECT_TRUE(info.has_reqd_workgroup_size);
  EXPECT_EQ(info.reqd_workgroup_size, (std::array<uint32_t, 3>{32, 1, 1}));
  EXPECT_EQ(k.local_size(), (std::array<uint32_t, 3>{32, 1, 1}));

  // End-to-end: out[i] = in[i] + 1. n = 1000 is not a multiple of 32 -> the
  // grid rounds up and the i<n guard trims the tail.
  const uint32_t n = 1000;
  std::vector<int32_t> in(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    in[i] = static_cast<int32_t>(i) * 3 - 500;
    expected[i] = in[i] + 1;
  }
  xpose::Buffer inb = ctx_->alloc<int32_t>(n);
  xpose::Buffer outb = ctx_->alloc<int32_t>(n);
  inb.upload(std::span<const int32_t>(in));

  xpose::launch(k, {n}, outb, inb, n).wait();

  std::vector<int32_t> result(n);
  outb.download(std::span<int32_t>(result));
  ASSERT_EQ(result, expected);
}

// ======================= two_kernels (multi-kernel module) ===================

TEST_F(KernelFeaturesTest, MultiKernelModule) {
  xpose::Program& prog = load_program("two_kernels");

  // kernel_names() returns both entry points in module order.
  EXPECT_EQ(prog.kernel_names(),
            (std::vector<std::string>{"add_one", "mul_two"}));

  xpose::Kernel add = prog.kernel("add_one");
  xpose::Kernel mul = prog.kernel("mul_two");

  // Distinct pipelines (and layouts) from the SAME module.
  EXPECT_NE(add.pipeline(), VK_NULL_HANDLE);
  EXPECT_NE(mul.pipeline(), VK_NULL_HANDLE);
  EXPECT_NE(add.pipeline(), mul.pipeline());
  EXPECT_NE(add.pipeline_layout(), mul.pipeline_layout());

  // Both take (global int* data, uint n).
  ASSERT_EQ(add.info().args.size(), 2u);
  expect_buffer_arg(add.info().args[0], 0, 0, 0, "data");
  expect_pod_arg(add.info().args[1], 1, 0, 4, "n");
  ASSERT_EQ(mul.info().args.size(), 2u);
  expect_buffer_arg(mul.info().args[0], 0, 0, 0, "data");
  expect_pod_arg(mul.info().args[1], 1, 0, 4, "n");

  // A missing name still throws, listing BOTH kernels the module contains.
  try {
    prog.kernel("sub_three");
    FAIL() << "expected xpose::Error for unknown kernel";
  } catch (const xpose::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("sub_three"), std::string::npos) << msg;
    EXPECT_NE(msg.find("add_one"), std::string::npos) << msg;
    EXPECT_NE(msg.find("mul_two"), std::string::npos) << msg;
  }

  // End-to-end: interleave the two kernels of the ONE Program on a SHARED
  // buffer with no host wait between launches (the in-cmdbuf dispatch->dispatch
  // barrier across two DIFFERENT pipelines from the same module makes each read
  // the previous kernel's writes). Sequence add,mul,add,mul -> ((x+1)*2+1)*2.
  const uint32_t n = 777;  // not a workgroup multiple
  std::vector<int32_t> data(n), expected(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = static_cast<int32_t>(i) - 300;
    expected[i] = ((data[i] + 1) * 2 + 1) * 2;  // exact 32-bit int arithmetic
  }
  xpose::Buffer buf = ctx_->alloc<int32_t>(n);
  buf.upload(std::span<const int32_t>(data));

  xpose::launch(add, {n}, buf, n);
  xpose::launch(mul, {n}, buf, n);
  xpose::launch(add, {n}, buf, n);
  xpose::launch(mul, {n}, buf, n).wait();

  std::vector<int32_t> result(n);
  buf.download(std::span<int32_t>(result));
  ASSERT_EQ(result, expected);
}

// ======================= many_pod (mixed-scalar push constants) ==============

TEST_F(KernelFeaturesTest, MixedPodPacking) {
  xpose::Program& prog = load_program("many_pod");
  xpose::Kernel k = prog.kernel("many_pod");

  const xpose::KernelInfo& info = k.info();
  ASSERT_EQ(info.args.size(), 6u);
  expect_buffer_arg(info.args[0], 0, 0, 0, "out");
  // Mixed types (float/int/uint/float/uint), all 4 bytes, tightly packed.
  expect_pod_arg(info.args[1], 1, 0, 4, "f");
  expect_pod_arg(info.args[2], 2, 4, 4, "i");
  expect_pod_arg(info.args[3], 3, 8, 4, "u");
  expect_pod_arg(info.args[4], 4, 12, 4, "g");
  expect_pod_arg(info.args[5], 5, 16, 4, "n");
  EXPECT_EQ(info.push_constant_size, 20u);

  // End-to-end: out[idx] = f*idx + i - u + g (proves the launch.cpp packing
  // memcpy writes each scalar at the right offset). Values chosen so every
  // intermediate is exactly representable.
  const uint32_t n = 1000;
  const float f = 2.0f;
  const int32_t i = -3;
  const uint32_t u = 7;
  const float g = 0.5f;
  std::vector<float> expected(n);
  for (uint32_t idx = 0; idx < n; ++idx) {
    expected[idx] = f * static_cast<float>(idx) + static_cast<float>(i) -
                    static_cast<float>(u) + g;
  }
  xpose::Buffer outb = ctx_->alloc<float>(n);

  xpose::launch(k, {n}, outb, f, i, u, g, n).wait();

  std::vector<float> result(n);
  outb.download(std::span<float>(result));
  for (uint32_t idx = 0; idx < n; ++idx) {
    ASSERT_FLOAT_EQ(result[idx], expected[idx]) << "idx=" << idx;
  }
}

}  // namespace
