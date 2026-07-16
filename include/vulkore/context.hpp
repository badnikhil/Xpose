// vulkore::Context — instance / physical device / logical device / compute queue.
//
// Device selection policy (in priority order):
//   1. explicit index passed to the constructor
//   2. VULKORE_DEVICE=<case-insensitive substring of device name> env var
//   3. first discrete GPU, else first integrated GPU, else anything with a
//      compute queue (covers lavapipe)
#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vulkore {

class Buffer;      // buffer.hpp
class Fence;       // sync.hpp
enum class Usage;  // buffer.hpp

struct DeviceInfo {
  uint32_t index = 0;
  std::string name;
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  uint32_t api_version = 0;
};

class Context {
 public:
  // Creates instance + logical device. With no argument, applies the
  // selection policy above; an explicit index overrides everything.
  explicit Context(std::optional<uint32_t> device_index = std::nullopt);
  ~Context();

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;             // Buffers/Fences hold Context*;
  Context& operator=(Context&&) = delete;  // keep it pinned.

  // ---- device info -------------------------------------------------------
  const std::string& device_name() const { return device_name_; }
  const VkPhysicalDeviceProperties& device_properties() const { return props_; }
  // Every physical device visible to this instance (selection candidates).
  const std::vector<DeviceInfo>& all_devices() const { return all_devices_; }
  uint32_t device_index() const { return device_index_; }

  // ---- buffers ------------------------------------------------------------
  // Storage-buffer usage by default; see buffer.hpp for Usage semantics.
  // Defined at the bottom of buffer.hpp (needs the complete Buffer type).
  template <typename T>
  Buffer alloc(size_t count, Usage usage);
  template <typename T>
  Buffer alloc(size_t count);  // Usage::DeviceLocal
  Buffer alloc_bytes(VkDeviceSize size_bytes, Usage usage);

  // ---- one-shot command buffers & submission ------------------------------
  // Allocates a primary command buffer from the pool and begins it with
  // ONE_TIME_SUBMIT. Pair with submit() + free_command_buffer().
  VkCommandBuffer begin_one_shot();
  // Ends recording, submits on the compute queue, returns a Fence signaled on
  // completion. The command buffer stays owned by the caller (free after wait).
  Fence submit(VkCommandBuffer cb);
  // As above, but the Fence runs `on_complete` once completion is observed
  // (wait()/is_signaled()/destruction). launch() recycles its command buffer
  // and descriptor sets through this. The hook must not throw.
  Fence submit(VkCommandBuffer cb, std::function<void()> on_complete);
  void free_command_buffer(VkCommandBuffer cb);
  // begin + record + submit + wait + free, in one call.
  void one_shot(const std::function<void(VkCommandBuffer)>& record);
  void wait_idle();  // vkDeviceWaitIdle — the cudaDeviceSynchronize analog

  // ---- descriptor sets (launch() machinery) --------------------------------
  // Allocates from an internal, on-demand-growing list of descriptor pools
  // (storage-buffer descriptors only — the only kind vulkore v1 kernels use).
  // Return sets with free_descriptor_set(); every pool is destroyed with the
  // Context.
  VkDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
  void free_descriptor_set(VkDescriptorSet set) noexcept;

  // ---- raw handles --------------------------------------------------------
  // Seam for the Program/Kernel/launch modules (architecture.md §3–4): pipeline
  // creation needs device()+table(), descriptor writes need allocator()+queue
  // family, dispatch recording uses begin_one_shot()/submit().
  // TODO(pipeline-cache): add a persistent VkPipelineCache accessor.
  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  uint32_t queue_family() const { return queue_family_; }
  VmaAllocator allocator() const { return allocator_; }
  // Per-device function table (volk). All device-level calls MUST go through
  // this — the process may hold several Contexts on different drivers.
  const VolkDeviceTable& table() const { return table_; }

 private:
  uint32_t pick_device(std::optional<uint32_t> explicit_index) const;
  void add_descriptor_pool();

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VmaAllocator allocator_ = VK_NULL_HANDLE;
  VolkDeviceTable table_{};

  // launch() descriptor pools (created on demand, grown when exhausted) and
  // the pool each live set came from (vkFreeDescriptorSets needs it).
  std::vector<VkDescriptorPool> descriptor_pools_;
  std::unordered_map<VkDescriptorSet, VkDescriptorPool> set_pool_;

  std::vector<DeviceInfo> all_devices_;
  uint32_t device_index_ = 0;
  std::string device_name_;
  VkPhysicalDeviceProperties props_{};
};

}  // namespace vulkore
