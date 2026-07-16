// Custom gtest main: prints which Vulkan device this run selected, so the
// XPOSE_DEVICE multi-driver test matrix output is self-describing.
#include <xpose/xpose.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  const char* sel = std::getenv("XPOSE_DEVICE");
  std::printf("[xpose] XPOSE_DEVICE=%s\n", sel ? sel : "(unset)");
  try {
    xpose::Context ctx;
    std::printf("[xpose] running on device [%u] \"%s\" (%zu devices visible)\n",
                ctx.device_index(), ctx.device_name().c_str(),
                ctx.all_devices().size());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[xpose] FATAL: cannot create Context: %s\n", e.what());
    return 1;
  }

  return RUN_ALL_TESTS();
}
