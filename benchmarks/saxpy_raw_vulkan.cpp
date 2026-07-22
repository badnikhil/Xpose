// saxpy_raw_vulkan.cpp — the SAME SAXPY as saxpy_vulkore.cpp, written directly
// against the raw Vulkan API. Identical benchmark protocol, identical kernel
// binary (clspv-compiled saxpy.spv: set 0 binding 0 = y, binding 1 = x,
// 8-byte push-constant block {float a; uint n}, workgroup size via spec
// constants 0/1/2).
//
// This is the MINIMAL correct version for mobile-class (Adreno/Mali) GPUs:
// it must handle HOST_VISIBLE-but-non-COHERENT memory (flush after CPU
// writes, invalidate before CPU reads) or results are silently wrong on
// device — the classic "works on desktop, broken on the phone" trap.
// Everything here is what a runtime is supposed to absorb: instance/device
// bring-up, queue-family selection, memory-type selection, buffer binding,
// descriptor layouts/pools/sets, pipeline + specialization, command pools,
// barriers, fences, cleanup.
#include <volk.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define VK_CHECK(call)                                                    \
  do {                                                                    \
    VkResult res_ = (call);                                               \
    if (res_ != VK_SUCCESS) {                                             \
      std::fprintf(stderr, "%s:%d: %s -> VkResult %d\n", __FILE__,        \
                   __LINE__, #call, static_cast<int>(res_));              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

namespace {

uint32_t kN = 1u << 20;  // element count; override with argv[2]
constexpr float kA = 2.5f;
constexpr int kSyncIters = 200;
constexpr int kBatch = 64;
constexpr int kRounds = 8;
constexpr uint32_t kWorkgroupSize = 64;

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  const std::streamsize size = f.tellg();
  if (size <= 0 || size % 4 != 0) {
    std::fprintf(stderr, "bad SPIR-V size %lld\n",
                 static_cast<long long>(size));
    std::exit(1);
  }
  std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(words.data()), size);
  return words;
}

bool name_contains_ci(const char* haystack, const char* needle) {
  const std::string h(haystack), n(needle);
  auto lower = [](std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(c));
    return s;
  };
  return lower(h).find(lower(n)) != std::string::npos;
}

// One host-visible storage buffer with its own device memory, persistently
// mapped. `coherent` records whether flush/invalidate are required.
struct GpuBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
  bool coherent = false;
};

struct PushConstants {
  float a;
  uint32_t n;
};

}  // namespace

int main(int argc, char** argv) {
  const std::string spv_path =
      argc > 1 ? argv[1] : "tests/kernels/saxpy.spv";
  if (argc > 2) kN = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));

  // ---- loader + instance ---------------------------------------------------
  VK_CHECK(volkInitialize());

  VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = "saxpy_raw_vulkan";
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo instance_ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_ci.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  VK_CHECK(vkCreateInstance(&instance_ci, nullptr, &instance));
  volkLoadInstance(instance);

  // ---- physical device selection ------------------------------------------
  // Same policy as the Vulkore sample: VULKORE_DEVICE env substring match
  // (case-insensitive), else first discrete GPU, else first integrated,
  // else device 0.
  uint32_t device_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));
  if (device_count == 0) {
    std::fprintf(stderr, "no Vulkan devices\n");
    return 1;
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &device_count,
                                      devices.data()));

  uint32_t picked = UINT32_MAX;
  const char* want = std::getenv("VULKORE_DEVICE");
  VkPhysicalDeviceProperties props{};
  if (want) {
    for (uint32_t i = 0; i < device_count; ++i) {
      vkGetPhysicalDeviceProperties(devices[i], &props);
      if (name_contains_ci(props.deviceName, want)) {
        picked = i;
        break;
      }
    }
    if (picked == UINT32_MAX) {
      std::fprintf(stderr, "no device matching VULKORE_DEVICE=%s\n", want);
      return 1;
    }
  } else {
    for (VkPhysicalDeviceType type :
         {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
          VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU}) {
      for (uint32_t i = 0; i < device_count && picked == UINT32_MAX; ++i) {
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceType == type) picked = i;
      }
      if (picked != UINT32_MAX) break;
    }
    if (picked == UINT32_MAX) picked = 0;
  }
  VkPhysicalDevice phys = devices[picked];
  vkGetPhysicalDeviceProperties(phys, &props);
  std::printf("device: %s\n", props.deviceName);
  if (props.apiVersion < VK_API_VERSION_1_1) {
    std::fprintf(stderr, "device does not support Vulkan 1.1\n");
    return 1;
  }

  // ---- queue family with compute ------------------------------------------
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &family_count,
                                           families.data());
  uint32_t queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < family_count; ++i) {
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      queue_family = i;
      break;
    }
  }
  if (queue_family == UINT32_MAX) {
    std::fprintf(stderr, "no compute queue family\n");
    return 1;
  }

  // ---- logical device + queue ---------------------------------------------
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_ci.queueFamilyIndex = queue_family;
  queue_ci.queueCount = 1;
  queue_ci.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_ci.queueCreateInfoCount = 1;
  device_ci.pQueueCreateInfos = &queue_ci;

  VkDevice device = VK_NULL_HANDLE;
  VK_CHECK(vkCreateDevice(phys, &device_ci, nullptr, &device));
  volkLoadDevice(device);

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, queue_family, 0, &queue);

  // ---- memory-type selection ----------------------------------------------
  // Prefer HOST_VISIBLE | HOST_COHERENT; fall back to any HOST_VISIBLE type
  // (e.g. Adreno/Mali HOST_CACHED) and remember that flush/invalidate are
  // then mandatory around every CPU access.
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
  auto find_memory_type = [&](uint32_t type_bits, VkMemoryPropertyFlags flags,
                              uint32_t* out) -> bool {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
      if ((type_bits & (1u << i)) &&
          (mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
        *out = i;
        return true;
      }
    }
    return false;
  };

  auto create_buffer = [&](VkDeviceSize size) -> GpuBuffer {
    GpuBuffer b;
    b.size = size;
    VkBufferCreateInfo buffer_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_ci.size = size;
    buffer_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &buffer_ci, nullptr, &b.buffer));

    VkMemoryRequirements reqs{};
    vkGetBufferMemoryRequirements(device, b.buffer, &reqs);

    uint32_t type_index = 0;
    if (find_memory_type(reqs.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &type_index)) {
      b.coherent = true;
    } else if (find_memory_type(reqs.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                &type_index)) {
      b.coherent = false;  // non-coherent: flush/invalidate required
    } else {
      std::fprintf(stderr, "no host-visible memory type\n");
      std::exit(1);
    }

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = reqs.size;
    alloc_info.memoryTypeIndex = type_index;
    VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    VK_CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    return b;
  };

  // Flush after host writes / invalidate before host reads. No-ops on
  // coherent memory. The range must be mapped (it is: persistent map) and
  // offset/size must respect nonCoherentAtomSize — offset 0 + VK_WHOLE_SIZE
  // satisfies that without alignment math.
  auto flush_writes = [&](const GpuBuffer& b) {
    if (b.coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = b.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    VK_CHECK(vkFlushMappedMemoryRanges(device, 1, &range));
  };
  auto invalidate_for_reads = [&](const GpuBuffer& b) {
    if (b.coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = b.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    VK_CHECK(vkInvalidateMappedMemoryRanges(device, 1, &range));
  };

  // ---- buffers + input data ------------------------------------------------
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(kN) * sizeof(float);
  GpuBuffer yb = create_buffer(bytes);
  GpuBuffer xb = create_buffer(bytes);

  std::vector<float> x(kN), y(kN);
  for (uint32_t i = 0; i < kN; ++i) {
    x[i] = 0.25f * static_cast<float>(i % 1000) - 100.0f;
    y[i] = 1.5f * static_cast<float>(i % 97);
  }
  std::memcpy(xb.mapped, x.data(), bytes);
  std::memcpy(yb.mapped, y.data(), bytes);
  flush_writes(xb);
  flush_writes(yb);

  // ---- shader module -------------------------------------------------------
  const std::vector<uint32_t> spirv = read_spirv(spv_path);
  VkShaderModuleCreateInfo module_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_ci.codeSize = spirv.size() * sizeof(uint32_t);
  module_ci.pCode = spirv.data();
  VkShaderModule shader_module = VK_NULL_HANDLE;
  VK_CHECK(vkCreateShaderModule(device, &module_ci, nullptr, &shader_module));

  // ---- descriptor set layout: binding 0 = y, binding 1 = x ----------------
  VkDescriptorSetLayoutBinding bindings[2]{};
  for (uint32_t i = 0; i < 2; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo set_layout_ci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_layout_ci.bindingCount = 2;
  set_layout_ci.pBindings = bindings;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &set_layout_ci, nullptr,
                                       &set_layout));

  // ---- pipeline layout: one 8-byte push-constant block --------------------
  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(PushConstants);

  VkPipelineLayoutCreateInfo pipeline_layout_ci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout_ci.setLayoutCount = 1;
  pipeline_layout_ci.pSetLayouts = &set_layout;
  pipeline_layout_ci.pushConstantRangeCount = 1;
  pipeline_layout_ci.pPushConstantRanges = &push_range;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_ci, nullptr,
                                  &pipeline_layout));

  // ---- compute pipeline (workgroup size via spec constants 0/1/2) ---------
  const uint32_t wg[3] = {kWorkgroupSize, 1, 1};
  VkSpecializationMapEntry spec_entries[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    spec_entries[i].constantID = i;
    spec_entries[i].offset = i * sizeof(uint32_t);
    spec_entries[i].size = sizeof(uint32_t);
  }
  VkSpecializationInfo spec_info{};
  spec_info.mapEntryCount = 3;
  spec_info.pMapEntries = spec_entries;
  spec_info.dataSize = sizeof(wg);
  spec_info.pData = wg;

  VkComputePipelineCreateInfo pipeline_ci{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_ci.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_ci.stage.module = shader_module;
  pipeline_ci.stage.pName = "saxpy";
  pipeline_ci.stage.pSpecializationInfo = &spec_info;
  pipeline_ci.layout = pipeline_layout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci,
                                    nullptr, &pipeline));

  // ---- descriptor pool + set + writes -------------------------------------
  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
  VkDescriptorPoolCreateInfo pool_ci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_ci.maxSets = 1;
  pool_ci.poolSizeCount = 1;
  pool_ci.pPoolSizes = &pool_size;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VK_CHECK(vkCreateDescriptorPool(device, &pool_ci, nullptr,
                                  &descriptor_pool));

  VkDescriptorSetAllocateInfo set_alloc{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_alloc.descriptorPool = descriptor_pool;
  set_alloc.descriptorSetCount = 1;
  set_alloc.pSetLayouts = &set_layout;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  VK_CHECK(vkAllocateDescriptorSets(device, &set_alloc, &descriptor_set));

  VkDescriptorBufferInfo buffer_infos[2]{};
  buffer_infos[0] = {yb.buffer, 0, VK_WHOLE_SIZE};
  buffer_infos[1] = {xb.buffer, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet writes[2]{};
  for (uint32_t i = 0; i < 2; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptor_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &buffer_infos[i];
  }
  vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

  // ---- command pool + command buffers + fences ----------------------------
  // kBatch command buffers/fences so the throughput phase can keep kBatch
  // submissions in flight (mirrors what the Vulkore runtime recycles for us).
  VkCommandPoolCreateInfo cmd_pool_ci{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cmd_pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cmd_pool_ci.queueFamilyIndex = queue_family;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VK_CHECK(vkCreateCommandPool(device, &cmd_pool_ci, nullptr, &command_pool));

  std::vector<VkCommandBuffer> cmd_bufs(kBatch);
  VkCommandBufferAllocateInfo cb_alloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cb_alloc.commandPool = command_pool;
  cb_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cb_alloc.commandBufferCount = kBatch;
  VK_CHECK(vkAllocateCommandBuffers(device, &cb_alloc, cmd_bufs.data()));

  std::vector<VkFence> fences(kBatch);
  VkFenceCreateInfo fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  for (int i = 0; i < kBatch; ++i)
    VK_CHECK(vkCreateFence(device, &fence_ci, nullptr, &fences[i]));

  const uint32_t groups = (kN + kWorkgroupSize - 1) / kWorkgroupSize;
  const PushConstants push{kA, kN};

  // Record one dispatch. Barriers: previous dispatches' writes to y must be
  // visible to this dispatch (launch->launch chains during the throughput
  // phase), and this dispatch's writes must be visible to host reads.
  auto record = [&](VkCommandBuffer cb) {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));

    VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    pre.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre, 0,
                         nullptr, 0, nullptr);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &descriptor_set, 0,
                            nullptr);
    vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);
    vkCmdDispatch(cb, groups, 1, 1);

    VkMemoryBarrier post{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &post, 0, nullptr,
                         0, nullptr);
    VK_CHECK(vkEndCommandBuffer(cb));
  };

  auto submit = [&](VkCommandBuffer cb, VkFence fence) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
  };

  auto launch_and_wait = [&]() {
    VK_CHECK(vkResetFences(device, 1, &fences[0]));
    record(cmd_bufs[0]);
    submit(cmd_bufs[0], fences[0]);
    VK_CHECK(vkWaitForFences(device, 1, &fences[0], VK_TRUE, UINT64_MAX));
  };

  // ---- correctness ---------------------------------------------------------
  launch_and_wait();
  invalidate_for_reads(yb);
  const float* out = static_cast<const float*>(yb.mapped);
  for (uint32_t i = 0; i < kN; ++i) {
    const float expected = kA * x[i] + y[i];
    if (out[i] != expected) {
      std::fprintf(stderr, "MISMATCH at %u: got %f want %f\n", i, out[i],
                   expected);
      return 1;
    }
  }
  std::printf("correctness: OK (%u elements)\n", kN);

  // ---- benchmark: sync latency --------------------------------------------
  for (int i = 0; i < 50; ++i) launch_and_wait();  // warmup

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kSyncIters; ++i) launch_and_wait();
  std::printf("sync latency: %.1f us/launch (%d iters)\n",
              1000.0 * ms_since(t0) / kSyncIters, kSyncIters);

  // ---- benchmark: throughput (kBatch launches in flight per round) --------
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < kRounds; ++r) {
    VK_CHECK(vkResetFences(device, kBatch, fences.data()));
    for (int i = 0; i < kBatch; ++i) {
      record(cmd_bufs[i]);
      submit(cmd_bufs[i], fences[i]);
    }
    VK_CHECK(vkWaitForFences(device, kBatch, fences.data(), VK_TRUE,
                             UINT64_MAX));
  }
  std::printf("throughput: %.1f us/launch (%d batches of %d)\n",
              1000.0 * ms_since(t0) / (kRounds * kBatch), kRounds, kBatch);

  // ---- cleanup -------------------------------------------------------------
  VK_CHECK(vkDeviceWaitIdle(device));
  for (VkFence f : fences) vkDestroyFence(device, f, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
  vkDestroyShaderModule(device, shader_module, nullptr);
  for (GpuBuffer* b : {&yb, &xb}) {
    vkUnmapMemory(device, b->memory);
    vkDestroyBuffer(device, b->buffer, nullptr);
    vkFreeMemory(device, b->memory, nullptr);
  }
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return 0;
}
