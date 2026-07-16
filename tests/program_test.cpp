// Program module: .spv loading + clspv reflection parsing + pipeline creation.
// Reflection expectations mirror tests/kernels/README.md (the fixture ABI:
// storage buffers at set 0 with bindings in buffer-arg order, PODs packed
// into one push-constant block from offset 0).
#include <vulkore/vulkore.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using vulkore::ArgKind;
using vulkore::KernelArg;

std::string kernel_path(const std::string& name) {
  return std::string(VULKORE_KERNEL_DIR) + "/" + name + ".spv";
}

class ProgramTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<vulkore::Context>(); }
  static void TearDownTestSuite() { ctx_.reset(); }
  static std::unique_ptr<vulkore::Context> ctx_;
};
std::unique_ptr<vulkore::Context> ProgramTest::ctx_;

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

TEST_F(ProgramTest, SaxpyReflection) {
  vulkore::Program prog =
      vulkore::Program::from_file(*ctx_, kernel_path("saxpy"));
  EXPECT_EQ(prog.kernel_names(), std::vector<std::string>{"saxpy"});

  vulkore::Kernel k = prog.kernel("saxpy");
  ASSERT_TRUE(k.valid());
  EXPECT_EQ(k.name(), "saxpy");

  const vulkore::KernelInfo& info = k.info();
  ASSERT_EQ(info.args.size(), 4u);
  expect_buffer_arg(info.args[0], 0, 0, 0, "y");
  expect_buffer_arg(info.args[1], 1, 0, 1, "x");
  expect_pod_arg(info.args[2], 2, 0, 4, "a");
  expect_pod_arg(info.args[3], 3, 4, 4, "n");
  EXPECT_EQ(info.push_constant_size, 8u);
  EXPECT_FALSE(info.has_reqd_workgroup_size);

  EXPECT_NE(k.pipeline(), VK_NULL_HANDLE);
  EXPECT_NE(k.pipeline_layout(), VK_NULL_HANDLE);
  ASSERT_EQ(k.set_layouts().size(), 1u);  // everything lives in set 0
  // No reqd_work_group_size attribute -> runtime default via spec constants.
  EXPECT_EQ(k.local_size(), (std::array<uint32_t, 3>{64, 1, 1}));
}

TEST_F(ProgramTest, VecAddReflection) {
  vulkore::Program prog =
      vulkore::Program::from_file(*ctx_, kernel_path("vec_add"));
  vulkore::Kernel k = prog.kernel("vec_add");
  const vulkore::KernelInfo& info = k.info();
  ASSERT_EQ(info.args.size(), 4u);
  expect_buffer_arg(info.args[0], 0, 0, 0, "out");
  expect_buffer_arg(info.args[1], 1, 0, 1, "a");
  expect_buffer_arg(info.args[2], 2, 0, 2, "b");
  expect_pod_arg(info.args[3], 3, 0, 4, "n");
  EXPECT_EQ(info.push_constant_size, 4u);
}

TEST_F(ProgramTest, ScaleInplaceReflection) {
  vulkore::Program prog =
      vulkore::Program::from_file(*ctx_, kernel_path("scale_inplace"));
  vulkore::Kernel k = prog.kernel("scale_inplace");
  const vulkore::KernelInfo& info = k.info();
  ASSERT_EQ(info.args.size(), 3u);
  expect_buffer_arg(info.args[0], 0, 0, 0, "data");
  expect_pod_arg(info.args[1], 1, 0, 4, "factor");
  expect_pod_arg(info.args[2], 2, 4, 4, "n");
  EXPECT_EQ(info.push_constant_size, 8u);
}

TEST_F(ProgramTest, CompareFpReflection) {
  vulkore::Program prog =
      vulkore::Program::from_file(*ctx_, kernel_path("compare_fp"));
  vulkore::Kernel k = prog.kernel("compare_fp");
  const vulkore::KernelInfo& info = k.info();
  ASSERT_EQ(info.args.size(), 5u);
  expect_buffer_arg(info.args[0], 0, 0, 0, "eq_out");
  expect_buffer_arg(info.args[1], 1, 0, 1, "lt_out");
  expect_buffer_arg(info.args[2], 2, 0, 2, "a");
  expect_buffer_arg(info.args[3], 3, 0, 3, "b");
  expect_pod_arg(info.args[4], 4, 0, 4, "n");
}

TEST_F(ProgramTest, KernelHandleIsCached) {
  vulkore::Program prog =
      vulkore::Program::from_file(*ctx_, kernel_path("saxpy"));
  vulkore::Kernel k1 = prog.kernel("saxpy");
  vulkore::Kernel k2 = prog.kernel("saxpy");
  EXPECT_EQ(k1.pipeline(), k2.pipeline());  // one pipeline per kernel
  EXPECT_EQ(k1.pipeline_layout(), k2.pipeline_layout());
}

TEST_F(ProgramTest, UnknownKernelNameThrowsListingContents) {
  vulkore::Program prog =
      vulkore::Program::from_file(*ctx_, kernel_path("saxpy"));
  try {
    prog.kernel("daxpy");
    FAIL() << "expected vulkore::Error";
  } catch (const vulkore::Error& e) {
    EXPECT_NE(std::string(e.what()).find("daxpy"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("saxpy"), std::string::npos);
  }
}

TEST_F(ProgramTest, MissingFileThrows) {
  EXPECT_THROW(vulkore::Program::from_file(*ctx_, kernel_path("no_such")),
               vulkore::Error);
}

TEST_F(ProgramTest, GarbageBytesThrow) {
  const std::vector<uint8_t> garbage(64, 0xAB);
  EXPECT_THROW(vulkore::Program(*ctx_, garbage), vulkore::Error);

  const std::vector<uint8_t> tiny = {0x03, 0x02, 0x23, 0x07};  // just magic
  EXPECT_THROW(vulkore::Program(*ctx_, tiny), vulkore::Error);
}

TEST_F(ProgramTest, NonClspvSpirvThrows) {
  // Valid header (magic, version 1.3), zero instructions: no reflection set.
  const uint32_t words[5] = {0x07230203u, 0x00010300u, 0, 100, 0};
  std::vector<uint8_t> bytes(sizeof(words));
  std::memcpy(bytes.data(), words, sizeof(words));
  try {
    vulkore::Program prog(*ctx_, bytes);
    FAIL() << "expected vulkore::Error";
  } catch (const vulkore::Error& e) {
    EXPECT_NE(std::string(e.what()).find("ClspvReflection"), std::string::npos);
  }
}

TEST_F(ProgramTest, EmptyKernelHandleThrows) {
  vulkore::Kernel empty;
  EXPECT_FALSE(empty.valid());
  EXPECT_THROW(empty.info(), vulkore::Error);
  EXPECT_THROW(vulkore::launch(empty, {1}), vulkore::Error);
}

}  // namespace
