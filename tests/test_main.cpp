// Custom gtest main: prints which Vulkan device this run selected, so the
// VULKORE_DEVICE multi-driver test matrix output is self-describing.
#include <vulkore/vulkore.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  const char* sel = std::getenv("VULKORE_DEVICE");
  std::printf("[vulkore] VULKORE_DEVICE=%s\n", sel ? sel : "(unset)");
  try {
    vulkore::Context ctx;
    std::printf("[vulkore] running on device [%u] \"%s\" (%zu devices visible)\n",
                ctx.device_index(), ctx.device_name().c_str(),
                ctx.all_devices().size());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[vulkore] FATAL: cannot create Context: %s\n", e.what());
    return 1;
  }

  return RUN_ALL_TESTS();
}
