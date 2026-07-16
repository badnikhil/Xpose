#include <vulkore/context.hpp>

#include <vulkore/buffer.hpp>
#include <vulkore/check.hpp>
#include <vulkore/sync.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>

namespace vulkore {

// ---- Error / vk_result_name -------------------------------------------------

std::string vk_result_name(VkResult r) {
  switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
    default: return "VkResult(" + std::to_string(static_cast<int>(r)) + ")";
  }
}

Error::Error(VkResult result, const char* expr, const char* file, int line)
    : std::runtime_error(vk_result_name(result) + " from `" + expr + "` at " +
                         file + ":" + std::to_string(line)),
      result_(result) {}

Error::Error(const std::string& message) : std::runtime_error(message) {}

// ---- volk bootstrap ---------------------------------------------------------

namespace {

void ensure_volk_initialized() {
  static std::once_flag once;
  static VkResult init_result = VK_ERROR_INITIALIZATION_FAILED;
  std::call_once(once, [] { init_result = volkInitialize(); });
  if (init_result != VK_SUCCESS) {
    throw Error(init_result, "volkInitialize() — is libvulkan.so.1 present?",
                __FILE__, __LINE__);
  }
}

bool contains_ci(const std::string& haystack, const std::string& needle) {
  auto it = std::search(haystack.begin(), haystack.end(),
                        needle.begin(), needle.end(),
                        [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                        });
  return it != haystack.end();
}

}  // namespace

// ---- Context ----------------------------------------------------------------

Context::Context(std::optional<uint32_t> device_index) {
  ensure_volk_initialized();

  // Vulkan 1.1 minimum target — the Android baseline. Request nothing exotic.
  if (volkGetInstanceVersion() < VK_API_VERSION_1_1) {
    throw Error("vulkore: Vulkan loader does not support Vulkan 1.1");
  }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "vulkore";
  app.pEngineName = "vulkore";
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  XP_CHECK(vkCreateInstance(&ici, nullptr, &instance_));
  // Instance-level functions only; device-level calls go through the
  // per-context VolkDeviceTable so several Contexts can coexist safely.
  volkLoadInstanceOnly(instance_);

  // Enumerate all physical devices (also exposed for tests / diagnostics).
  uint32_t count = 0;
  XP_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
  if (count == 0) throw Error("vulkore: no Vulkan physical devices found");
  std::vector<VkPhysicalDevice> phys(count);
  XP_CHECK(vkEnumeratePhysicalDevices(instance_, &count, phys.data()));

  all_devices_.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(phys[i], &p);
    all_devices_.push_back({i, p.deviceName, p.deviceType, p.apiVersion});
  }

  device_index_ = pick_device(device_index);
  physical_device_ = phys[device_index_];
  vkGetPhysicalDeviceProperties(physical_device_, &props_);
  device_name_ = props_.deviceName;

  // Queue family: prefer a dedicated compute family (compute && !graphics),
  // else the first compute-capable one (graphics+compute).
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &family_count,
                                           families.data());
  std::optional<uint32_t> dedicated, any_compute;
  for (uint32_t i = 0; i < family_count; ++i) {
    const VkQueueFlags f = families[i].queueFlags;
    if (!(f & VK_QUEUE_COMPUTE_BIT)) continue;
    if (!any_compute) any_compute = i;
    if (!(f & VK_QUEUE_GRAPHICS_BIT) && !dedicated) dedicated = i;
  }
  if (!any_compute) {
    throw Error("vulkore: device '" + device_name_ + "' has no compute queue");
  }
  queue_family_ = dedicated.value_or(*any_compute);

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = queue_family_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  // TODO(program/launch): enable features kernels may need (shaderInt64,
  // 8/16-bit storage, variablePointers) after querying support here.
  XP_CHECK(vkCreateDevice(physical_device_, &dci, nullptr, &device_));
  volkLoadDeviceTable(&table_, device_);
  table_.vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
               VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = queue_family_;
  XP_CHECK(table_.vkCreateCommandPool(device_, &cpci, nullptr, &command_pool_));

  // VMA, fed exclusively through the dynamic entry points (volk owns loading).
  VmaVulkanFunctions vf{};
  vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vf.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  VmaAllocatorCreateInfo aci{};
  aci.vulkanApiVersion = VK_API_VERSION_1_1;
  aci.instance = instance_;
  aci.physicalDevice = physical_device_;
  aci.device = device_;
  aci.pVulkanFunctions = &vf;
  XP_CHECK(vmaCreateAllocator(&aci, &allocator_));
}

uint32_t Context::pick_device(std::optional<uint32_t> explicit_index) const {
  // 1. Explicit index wins.
  if (explicit_index) {
    if (*explicit_index >= all_devices_.size()) {
      throw Error("vulkore: device index " + std::to_string(*explicit_index) +
                  " out of range (found " +
                  std::to_string(all_devices_.size()) + " devices)");
    }
    return *explicit_index;
  }
  // 2. VULKORE_DEVICE=<substring of device name>, case-insensitive.
  if (const char* want = std::getenv("VULKORE_DEVICE"); want && *want) {
    for (const DeviceInfo& d : all_devices_) {
      if (contains_ci(d.name, want)) return d.index;
    }
    std::string names;
    for (const DeviceInfo& d : all_devices_) names += "\n  [" + std::to_string(d.index) + "] " + d.name;
    throw Error("vulkore: VULKORE_DEVICE='" + std::string(want) +
                "' matches no device; available:" + names);
  }
  // 3. First discrete, else first integrated, else anything.
  for (const DeviceInfo& d : all_devices_) {
    if (d.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return d.index;
  }
  for (const DeviceInfo& d : all_devices_) {
    if (d.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return d.index;
  }
  return all_devices_.front().index;
}

Context::~Context() {
  if (device_ != VK_NULL_HANDLE) {
    table_.vkDeviceWaitIdle(device_);
    if (allocator_ != VK_NULL_HANDLE) vmaDestroyAllocator(allocator_);
    if (command_pool_ != VK_NULL_HANDLE) {
      table_.vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    table_.vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
}

// ---- buffers ----------------------------------------------------------------

Buffer Context::alloc_bytes(VkDeviceSize size_bytes, Usage usage) {
  return Buffer(*this, size_bytes, usage);
}

// ---- one-shot command buffers -------------------------------------------

VkCommandBuffer Context::begin_one_shot() {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = command_pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  XP_CHECK(table_.vkAllocateCommandBuffers(device_, &ai, &cb));

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  XP_CHECK(table_.vkBeginCommandBuffer(cb, &bi));
  return cb;
}

Fence Context::submit(VkCommandBuffer cb) {
  XP_CHECK(table_.vkEndCommandBuffer(cb));
  Fence fence(*this);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  XP_CHECK(table_.vkQueueSubmit(queue_, 1, &si, fence.handle()));
  return fence;
}

void Context::free_command_buffer(VkCommandBuffer cb) {
  table_.vkFreeCommandBuffers(device_, command_pool_, 1, &cb);
}

void Context::one_shot(const std::function<void(VkCommandBuffer)>& record) {
  VkCommandBuffer cb = begin_one_shot();
  try {
    record(cb);
    Fence fence = submit(cb);
    fence.wait();
  } catch (...) {
    free_command_buffer(cb);
    throw;
  }
  free_command_buffer(cb);
}

void Context::wait_idle() { XP_CHECK(table_.vkDeviceWaitIdle(device_)); }

}  // namespace vulkore
