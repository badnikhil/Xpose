#include <xpose/xpose.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

namespace {

// RAII save/restore for XPOSE_DEVICE so selection tests do not disturb the
// value the outer test-matrix runner exported.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) saved_ = old;
    if (value) {
      ::setenv(name, value, /*overwrite=*/1);
    } else {
      ::unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (saved_) {
      ::setenv(name_.c_str(), saved_->c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> saved_;
};

bool contains_ci(const std::string& haystack, const std::string& needle) {
  auto lower = [](std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
  };
  return lower(haystack).find(lower(needle)) != std::string::npos;
}

TEST(ContextTest, CreatesOnDefaultDevice) {
  xpose::Context ctx;
  EXPECT_FALSE(ctx.device_name().empty());
  EXPECT_NE(ctx.device(), VK_NULL_HANDLE);
  EXPECT_NE(ctx.queue(), VK_NULL_HANDLE);
  EXPECT_NE(ctx.allocator(), VK_NULL_HANDLE);
  EXPECT_NE(ctx.table().vkQueueSubmit, nullptr);
}

TEST(ContextTest, EnumeratesAtLeastThreeDevices) {
  // This machine: AMD RADV RENOIR + NVIDIA RTX 2050 + llvmpipe.
  xpose::Context ctx;
  EXPECT_GE(ctx.all_devices().size(), 3u)
      << "expected RADV + NVIDIA + llvmpipe to all be visible";
  for (const auto& d : ctx.all_devices()) {
    EXPECT_FALSE(d.name.empty());
    EXPECT_GE(d.api_version, VK_API_VERSION_1_1);
  }
}

TEST(ContextTest, EnvVarSelectsLlvmpipe) {
  ScopedEnv env("XPOSE_DEVICE", "llvmpipe");
  xpose::Context ctx;
  EXPECT_TRUE(contains_ci(ctx.device_name(), "llvmpipe"))
      << "got: " << ctx.device_name();
  EXPECT_EQ(ctx.device_properties().deviceType, VK_PHYSICAL_DEVICE_TYPE_CPU);
}

TEST(ContextTest, EnvVarMatchIsCaseInsensitive) {
  ScopedEnv env("XPOSE_DEVICE", "LLVMPIPE");
  xpose::Context ctx;
  EXPECT_TRUE(contains_ci(ctx.device_name(), "llvmpipe"));
}

TEST(ContextTest, EnvVarNoMatchThrows) {
  ScopedEnv env("XPOSE_DEVICE", "definitely-not-a-real-gpu-9000");
  EXPECT_THROW({ xpose::Context ctx; }, xpose::Error);
}

TEST(ContextTest, ExplicitIndexSelectsThatDevice) {
  // Unset XPOSE_DEVICE to prove the explicit index needs no env cooperation,
  // then check the index-0 context reports the enumerated index-0 name.
  ScopedEnv env("XPOSE_DEVICE", nullptr);
  xpose::Context probe;
  const auto& devices = probe.all_devices();
  ASSERT_FALSE(devices.empty());

  xpose::Context ctx(0u);
  EXPECT_EQ(ctx.device_index(), 0u);
  EXPECT_EQ(ctx.device_name(), devices[0].name);
}

TEST(ContextTest, ExplicitIndexBeatsEnvVar) {
  xpose::Context probe;
  const auto& devices = probe.all_devices();
  // Env var points somewhere else; explicit index must still win.
  ScopedEnv env("XPOSE_DEVICE", "llvmpipe");
  const uint32_t last = static_cast<uint32_t>(devices.size() - 1);
  xpose::Context ctx(0u);
  EXPECT_EQ(ctx.device_index(), 0u);
  (void)last;
}

TEST(ContextTest, ExplicitIndexOutOfRangeThrows) {
  EXPECT_THROW({ xpose::Context ctx(1000u); }, xpose::Error);
}

TEST(ContextTest, DefaultPolicyPrefersNonCpuDevice) {
  // With three devices including hardware GPUs, default must not pick the
  // CPU device (llvmpipe). Only meaningful when XPOSE_DEVICE is unset.
  ScopedEnv env("XPOSE_DEVICE", nullptr);
  xpose::Context ctx;
  if (ctx.all_devices().size() >= 2) {
    bool has_gpu = false;
    for (const auto& d : ctx.all_devices()) {
      if (d.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
          d.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        has_gpu = true;
      }
    }
    if (has_gpu) {
      EXPECT_NE(ctx.device_properties().deviceType, VK_PHYSICAL_DEVICE_TYPE_CPU)
          << "default policy picked " << ctx.device_name();
    }
  }
}

TEST(ContextTest, TwoContextsOnDifferentDevicesCoexist) {
  // Per-context VolkDeviceTable: two live devices must not clobber each other.
  ScopedEnv env("XPOSE_DEVICE", nullptr);
  xpose::Context a;  // default (hardware GPU on this machine)
  ScopedEnv env2("XPOSE_DEVICE", "llvmpipe");
  xpose::Context b;
  EXPECT_NE(a.device(), b.device());
  a.wait_idle();
  b.wait_idle();
}

}  // namespace
