// Host-only tests for the error-handling backbone: the XP_CHECK macro, the
// vulkore::Error it throws, and vk_result_name(). These touch no Vulkan device
// (VkResult values are compile-time enum constants), so they run anywhere —
// desktop and phone alike — without a Context or a live driver.
#include <vulkore/check.hpp>

#include <gtest/gtest.h>

#include <string>

namespace {

// XP_CHECK on a non-success VkResult throws vulkore::Error carrying the real
// VkResult plus a message with the result name, the checked expression, and
// file:line (check.hpp:29-35 macro + context.cpp:44-47 Error ctor). VK_SUCCESS
// does not throw. This is the backbone of the whole error strategy.
TEST(CheckTest, CheckThrowsCarriesResultAndLocation) {
  // Success path: no throw (check.hpp:32 — only non-VK_SUCCESS throws).
  EXPECT_NO_THROW(XP_CHECK(VK_SUCCESS));

  // Use a variable so the stringized expression ("lost") is DISTINCT from the
  // VkResult name ("VK_ERROR_DEVICE_LOST"); otherwise the two substring checks
  // below would collapse into one.
  const VkResult lost = VK_ERROR_DEVICE_LOST;
  try {
    XP_CHECK(lost);
    FAIL() << "XP_CHECK on a non-success VkResult must throw";
  } catch (const vulkore::Error& e) {
    // Carries the actual VkResult (only XP_CHECK-on-a-live-call errors do; the
    // string Error ctor would carry VK_ERROR_UNKNOWN).
    EXPECT_EQ(e.result(), VK_ERROR_DEVICE_LOST);
    const std::string msg = e.what();
    EXPECT_NE(msg.find("VK_ERROR_DEVICE_LOST"), std::string::npos)
        << "message should name the result: " << msg;
    EXPECT_NE(msg.find("lost"), std::string::npos)
        << "message should quote the checked expression: " << msg;
    EXPECT_NE(msg.find("check_test.cpp:"), std::string::npos)
        << "message should carry file:line: " << msg;
  }
}

// vk_result_name() maps known VkResults to their exact enum-name strings and
// falls back to "VkResult(<n>)" for anything unmapped — guarding the switch's
// default branch (context.cpp:16-42) against a typo'd or missing case label.
TEST(CheckTest, VkResultNameMapsKnownAndUnknown) {
  EXPECT_EQ(vulkore::vk_result_name(VK_SUCCESS), "VK_SUCCESS");
  EXPECT_EQ(vulkore::vk_result_name(VK_NOT_READY), "VK_NOT_READY");
  EXPECT_EQ(vulkore::vk_result_name(VK_TIMEOUT), "VK_TIMEOUT");
  EXPECT_EQ(vulkore::vk_result_name(VK_ERROR_DEVICE_LOST), "VK_ERROR_DEVICE_LOST");
  EXPECT_EQ(vulkore::vk_result_name(VK_ERROR_OUT_OF_DEVICE_MEMORY),
            "VK_ERROR_OUT_OF_DEVICE_MEMORY");
  EXPECT_EQ(vulkore::vk_result_name(VK_ERROR_UNKNOWN), "VK_ERROR_UNKNOWN");

  // Unmapped values -> default branch (context.cpp:40). VkResult is signed, so
  // exercise both a positive and a negative unmapped code.
  EXPECT_EQ(vulkore::vk_result_name(static_cast<VkResult>(123456)),
            "VkResult(123456)");
  EXPECT_EQ(vulkore::vk_result_name(static_cast<VkResult>(-99)), "VkResult(-99)");
}

}  // namespace
