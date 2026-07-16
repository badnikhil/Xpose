// xpose — error strategy: exceptions (hackathon-speed choice, see agent-docs/xpose-core.md).
// Every Vulkan call goes through XP_CHECK, which throws xpose::Error carrying the
// VkResult and its enum name.
#pragma once

#include <volk.h>

#include <stdexcept>
#include <string>

namespace xpose {

// Human-readable name for a VkResult ("VK_ERROR_DEVICE_LOST", ...).
std::string vk_result_name(VkResult r);

class Error : public std::runtime_error {
 public:
  Error(VkResult result, const char* expr, const char* file, int line);
  explicit Error(const std::string& message);  // non-VkResult failures

  VkResult result() const { return result_; }

 private:
  VkResult result_ = VK_ERROR_UNKNOWN;
};

}  // namespace xpose

#define XP_CHECK(expr)                                              \
  do {                                                              \
    VkResult xp_check_result_ = (expr);                             \
    if (xp_check_result_ != VK_SUCCESS) {                           \
      throw ::xpose::Error(xp_check_result_, #expr, __FILE__, __LINE__); \
    }                                                               \
  } while (0)
