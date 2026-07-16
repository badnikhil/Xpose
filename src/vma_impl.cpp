// Single translation unit hosting the VulkanMemoryAllocator implementation.
// Vulkan entry points are provided dynamically by volk (VK_NO_PROTOTYPES);
// VMA_STATIC_VULKAN_FUNCTIONS=0 / VMA_DYNAMIC_VULKAN_FUNCTIONS=1 come from the
// xpose CMake target so every TU sees the same VMA configuration.
#include <volk.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
