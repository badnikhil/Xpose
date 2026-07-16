// xpose::Program / xpose::Kernel — clspv-compiled SPIR-V modules.
//
// Program loads a .spv, parses clspv's NonSemantic.ClspvReflection extended
// instruction set (the same scheme clvk consumes) and exposes kernels by
// entry-point name. Kernel is a lightweight handle onto Program-owned state:
// reflection info + VkPipelineLayout + compute VkPipeline.
//
// Supported argument ABI (the mode our fixtures are compiled with — see
// tests/kernels/README.md):
//   - global/constant pointers -> storage buffers (ArgumentStorageBuffer)
//   - POD scalars/structs      -> one push-constant block
//                                 (ArgumentPodPushConstant, clspv
//                                 -pod-pushconstant)
// Anything else (images, samplers, UBO/SSBO PODs, pointer-to-local, implicit
// push constants, program-scope variables, printf) throws at load with the
// offending reflection instruction named.
#pragma once

#include <xpose/context.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace xpose {

enum class ArgKind {
  StorageBuffer,    // bind an xpose::Buffer (descriptor set/binding below)
  PodPushConstant,  // pass a POD value (offset/size in the push-constant block)
};

struct KernelArg {
  uint32_t ordinal = 0;
  ArgKind kind = ArgKind::StorageBuffer;
  std::string name;      // from ArgumentInfo; may be empty if clspv omitted it
  // ArgKind::StorageBuffer:
  uint32_t set = 0;
  uint32_t binding = 0;
  // ArgKind::PodPushConstant:
  uint32_t offset = 0;   // byte offset inside the push-constant block
  uint32_t size = 0;     // byte size of the value
};

struct KernelInfo {
  std::string name;
  std::vector<KernelArg> args;          // in ordinal order
  uint32_t push_constant_size = 0;      // bytes of POD block (0 = none)
  // reqd_work_group_size kernel attribute, when present
  // (PropertyRequiredWorkgroupSize reflection).
  bool has_reqd_workgroup_size = false;
  std::array<uint32_t, 3> reqd_workgroup_size{1, 1, 1};
};

namespace detail {
struct KernelState;
struct ProgramImpl;
}  // namespace detail

// Cheap copyable handle; the pipeline/layout it points at are owned by the
// Program, so a Kernel must not outlive its Program (nor the Context).
class Kernel {
 public:
  Kernel() = default;  // empty; launch() rejects it

  bool valid() const { return state_ != nullptr; }
  const KernelInfo& info() const;
  const std::string& name() const;
  Context& context() const;

  // Workgroup size the pipeline was specialized with: the kernel's
  // reqd_work_group_size attribute when present, else {64,1,1}. launch()
  // rounds the (CUDA-style, global-thread) grid up to whole workgroups.
  std::array<uint32_t, 3> local_size() const;

  VkPipeline pipeline() const;
  VkPipelineLayout pipeline_layout() const;
  // One layout per descriptor set index, dense from set 0.
  std::span<const VkDescriptorSetLayout> set_layouts() const;

 private:
  friend class Program;
  explicit Kernel(detail::KernelState* state) : state_(state) {}
  detail::KernelState* state_ = nullptr;
};

class Program {
 public:
  // Parses reflection and creates the VkShaderModule (with the non-semantic
  // reflection instructions stripped, so no VK_KHR_shader_non_semantic_info
  // device extension is required). Throws xpose::Error on malformed SPIR-V or
  // unsupported clspv features.
  Program(Context& ctx, std::span<const uint8_t> spirv);
  static Program from_file(Context& ctx, const std::string& path);
  ~Program();

  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  Program(Program&&) noexcept;
  Program& operator=(Program&&) noexcept;

  // Kernel by entry-point name. The compute pipeline is created on first
  // request and cached. Throws if the name is unknown (message lists the
  // kernels the module does contain).
  Kernel kernel(const std::string& name);

  std::vector<std::string> kernel_names() const;

 private:
  std::unique_ptr<detail::ProgramImpl> impl_;
};

}  // namespace xpose
