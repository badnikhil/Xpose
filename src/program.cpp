// SPIR-V loading + clspv reflection parsing + compute pipeline creation.
//
// Reflection format reference: clspv docs/OpenCLCOnVulkan.md ("Embedded
// Reflection Instructions") + SPIRV-Headers NonSemanticClspvReflection.h;
// operand layout cross-checked against clvk's parser (third_party/clvk
// src/program.cpp, parse_reflection()).
#include <vulkore/program.hpp>

#include <vulkore/check.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <utility>

namespace vulkore {

namespace {

// ---- SPIR-V core opcodes we care about --------------------------------------
constexpr uint32_t kOpString = 7;
constexpr uint32_t kOpExtension = 10;
constexpr uint32_t kOpExtInstImport = 11;
constexpr uint32_t kOpExtInst = 12;
constexpr uint32_t kOpTypeInt = 21;
constexpr uint32_t kOpConstant = 43;

// ---- NonSemantic.ClspvReflection instruction numbers -------------------------
// (SPIRV-Headers include/spirv/unified1/NonSemanticClspvReflection.h, rev 7)
enum ClspvReflection : uint32_t {
  kReflKernel = 1,
  kReflArgumentInfo = 2,
  kReflArgumentStorageBuffer = 3,
  kReflArgumentPodPushConstant = 7,
  kReflSpecConstantWorkgroupSize = 12,
  kReflPropertyRequiredWorkgroupSize = 24,
};

const char* reflection_inst_name(uint32_t n) {
  // Names for the instructions we deliberately do NOT support, so load-time
  // errors say what feature the kernel actually used.
  switch (n) {
    case 4: return "ArgumentUniform";
    case 5: return "ArgumentPodStorageBuffer";
    case 6: return "ArgumentPodUniform";
    case 8: return "ArgumentSampledImage";
    case 9: return "ArgumentStorageImage";
    case 10: return "ArgumentSampler";
    case 11: return "ArgumentWorkgroup (pointer-to-local)";
    case 13: return "SpecConstantGlobalOffset";
    case 14: return "SpecConstantWorkDim";
    case 15: return "PushConstantGlobalOffset";
    case 16: return "PushConstantEnqueuedLocalSize";
    case 17: return "PushConstantGlobalSize";
    case 18: return "PushConstantRegionOffset";
    case 19: return "PushConstantNumWorkgroups";
    case 20: return "PushConstantRegionGroupOffset";
    case 21: return "ConstantDataStorageBuffer";
    case 22: return "ConstantDataUniform";
    case 23: return "LiteralSampler";
    case 25: return "SpecConstantSubgroupMaxSize";
    case 26: return "ArgumentPointerPushConstant";
    case 27: return "ArgumentPointerUniform";
    case 28: return "ProgramScopeVariablesStorageBuffer";
    case 29: return "ProgramScopeVariablePointerRelocation";
    case 34: return "ArgumentStorageTexelBuffer";
    case 35: return "ArgumentUniformTexelBuffer";
    case 36: return "ConstantDataPointerPushConstant";
    case 37: return "ProgramScopeVariablePointerPushConstant";
    case 38: return "PrintfInfo";
    case 39: return "PrintfBufferStorageBuffer";
    case 40: return "PrintfBufferPointerPushConstant";
    case 41: return "NormalizedSamplerMaskPushConstant";
    case 42: return "WorkgroupVariableSize";
    default: return "unknown";
  }
}

// Decodes a nul-terminated literal string starting at words[first], bounded by
// the instruction's word count.
std::string literal_string(const uint32_t* words, uint32_t word_count,
                           uint32_t first) {
  if (first >= word_count) return {};
  const char* begin = reinterpret_cast<const char*>(words + first);
  const size_t max_len = static_cast<size_t>(word_count - first) * 4;
  size_t len = 0;
  while (len < max_len && begin[len] != '\0') ++len;
  return std::string(begin, len);
}

constexpr uint32_t kSpirvMagic = 0x07230203u;
constexpr uint32_t kHeaderWords = 5;

}  // namespace

namespace detail {

struct KernelState {
  Context* ctx = nullptr;
  ProgramImpl* program = nullptr;
  KernelInfo info;
  std::array<uint32_t, 3> local_size{1, 1, 1};
  std::vector<VkDescriptorSetLayout> set_layouts;  // index = set number
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  bool built = false;
};

struct ProgramImpl {
  Context* ctx = nullptr;
  VkShaderModule shader_module = VK_NULL_HANDLE;

  // Module-wide workgroup-size spec constant IDs (SpecConstantWorkgroupSize).
  bool has_wg_spec_ids = false;
  std::array<uint32_t, 3> wg_spec_ids{0, 1, 2};

  // Kernels in module order; stable addresses (KernelState is what Kernel
  // handles point at).
  std::vector<std::unique_ptr<KernelState>> kernels;

  KernelState* find(const std::string& name) {
    for (auto& k : kernels) {
      if (k->info.name == name) return k.get();
    }
    return nullptr;
  }

  ~ProgramImpl() {
    if (!ctx) return;
    const VolkDeviceTable& t = ctx->table();
    VkDevice dev = ctx->device();
    for (auto& k : kernels) {
      if (k->pipeline) t.vkDestroyPipeline(dev, k->pipeline, nullptr);
      if (k->pipeline_layout) {
        t.vkDestroyPipelineLayout(dev, k->pipeline_layout, nullptr);
      }
      for (VkDescriptorSetLayout l : k->set_layouts) {
        if (l) t.vkDestroyDescriptorSetLayout(dev, l, nullptr);
      }
    }
    if (shader_module) t.vkDestroyShaderModule(dev, shader_module, nullptr);
  }
};

}  // namespace detail

using detail::KernelState;
using detail::ProgramImpl;

namespace {

// Walks the instruction stream, invoking fn(opcode, words, word_count) per
// instruction. words points at the first word of the instruction.
template <typename Fn>
void for_each_instruction(std::span<const uint32_t> code, Fn&& fn) {
  size_t i = kHeaderWords;
  while (i < code.size()) {
    const uint32_t word_count = code[i] >> 16u;
    const uint32_t opcode = code[i] & 0xFFFFu;
    if (word_count == 0 || i + word_count > code.size()) {
      throw Error("vulkore::Program: malformed SPIR-V (bad instruction size at word " +
                  std::to_string(i) + ")");
    }
    fn(opcode, code.data() + i, word_count);
    i += word_count;
  }
}

struct ParseScratch {
  std::unordered_map<uint32_t, std::string> strings;    // OpString id -> text
  std::unordered_map<uint32_t, uint32_t> constants;     // OpConstant %uint id -> value
  std::unordered_map<uint32_t, std::string> arg_names;  // ArgumentInfo id -> name
  std::unordered_map<uint32_t, KernelState*> kernel_by_id;  // Kernel inst id -> state
  uint32_t uint_type_id = 0;
  uint32_t reflection_set_id = 0;

  uint32_t constant(uint32_t id, const char* what) const {
    auto it = constants.find(id);
    if (it == constants.end()) {
      throw Error(std::string("vulkore::Program: reflection operand '") + what +
                  "' (id " + std::to_string(id) + ") is not a 32-bit integer constant");
    }
    return it->second;
  }
  const std::string& string(uint32_t id, const char* what) const {
    auto it = strings.find(id);
    if (it == strings.end()) {
      throw Error(std::string("vulkore::Program: reflection operand '") + what +
                  "' (id " + std::to_string(id) + ") is not an OpString");
    }
    return it->second;
  }
};

void parse_reflection(std::span<const uint32_t> code, ProgramImpl& impl) {
  ParseScratch s;

  // Pass 1: strings, uint constants, and the reflection ext-inst set id.
  for_each_instruction(code, [&](uint32_t op, const uint32_t* w, uint32_t wc) {
    switch (op) {
      case kOpString:
        if (wc >= 3) s.strings[w[1]] = literal_string(w, wc, 2);
        break;
      case kOpExtInstImport:
        if (wc >= 3 &&
            literal_string(w, wc, 2).rfind("NonSemantic.ClspvReflection", 0) == 0) {
          s.reflection_set_id = w[1];
        }
        break;
      case kOpTypeInt:
        if (wc >= 4 && w[2] == 32 && w[3] == 0) s.uint_type_id = w[1];
        break;
      case kOpConstant:
        if (wc >= 4 && w[1] == s.uint_type_id) s.constants[w[2]] = w[3];
        break;
      default:
        break;
    }
  });

  if (s.reflection_set_id == 0) {
    throw Error(
        "vulkore::Program: no NonSemantic.ClspvReflection instruction set in "
        "module — is this clspv output? (compile kernels with clspv; see "
        "tests/kernels/README.md)");
  }

  // Pass 2: the reflection instructions themselves.
  // OpExtInst layout: w[1]=result type, w[2]=result id, w[3]=set, w[4]=number,
  // w[5..]=operands (ids of OpString / OpConstant / other reflection insts).
  for_each_instruction(code, [&](uint32_t op, const uint32_t* w, uint32_t wc) {
    if (op != kOpExtInst || wc < 5 || w[3] != s.reflection_set_id) return;
    const uint32_t inst = w[4];
    auto kernel_of = [&](uint32_t id) -> KernelState& {
      auto it = s.kernel_by_id.find(id);
      if (it == s.kernel_by_id.end()) {
        throw Error("vulkore::Program: reflection argument references unknown "
                    "kernel (id " + std::to_string(id) + ")");
      }
      return *it->second;
    };
    switch (inst) {
      case kReflKernel: {  // Kernel %fn %name [numArgs flags attributes]
        if (wc < 7) throw Error("vulkore::Program: truncated Kernel reflection");
        auto st = std::make_unique<KernelState>();
        st->ctx = impl.ctx;
        st->program = &impl;
        st->info.name = s.string(w[6], "kernel name");
        s.kernel_by_id[w[2]] = st.get();
        impl.kernels.push_back(std::move(st));
        break;
      }
      case kReflArgumentInfo: {  // ArgumentInfo %name [type addrq accq typeq]
        if (wc >= 6) s.arg_names[w[2]] = s.string(w[5], "argument name");
        break;
      }
      case kReflArgumentStorageBuffer: {
        // %kernel ordinal set binding [arginfo]
        if (wc < 9) throw Error("vulkore::Program: truncated ArgumentStorageBuffer");
        KernelArg a;
        a.kind = ArgKind::StorageBuffer;
        a.ordinal = s.constant(w[6], "ordinal");
        a.set = s.constant(w[7], "descriptor set");
        a.binding = s.constant(w[8], "binding");
        if (wc >= 10) a.name = s.arg_names[w[9]];
        kernel_of(w[5]).info.args.push_back(std::move(a));
        break;
      }
      case kReflArgumentPodPushConstant: {
        // %kernel ordinal offset size [arginfo]
        if (wc < 9) throw Error("vulkore::Program: truncated ArgumentPodPushConstant");
        KernelArg a;
        a.kind = ArgKind::PodPushConstant;
        a.ordinal = s.constant(w[6], "ordinal");
        a.offset = s.constant(w[7], "offset");
        a.size = s.constant(w[8], "size");
        if (wc >= 10) a.name = s.arg_names[w[9]];
        kernel_of(w[5]).info.args.push_back(std::move(a));
        break;
      }
      case kReflSpecConstantWorkgroupSize: {  // xSpecId ySpecId zSpecId
        if (wc < 8) throw Error("vulkore::Program: truncated SpecConstantWorkgroupSize");
        impl.has_wg_spec_ids = true;
        impl.wg_spec_ids = {s.constant(w[5], "x spec id"),
                            s.constant(w[6], "y spec id"),
                            s.constant(w[7], "z spec id")};
        break;
      }
      case kReflPropertyRequiredWorkgroupSize: {  // %kernel x y z
        if (wc < 9) {
          throw Error("vulkore::Program: truncated PropertyRequiredWorkgroupSize");
        }
        KernelState& k = kernel_of(w[5]);
        k.info.has_reqd_workgroup_size = true;
        k.info.reqd_workgroup_size = {s.constant(w[6], "x"),
                                      s.constant(w[7], "y"),
                                      s.constant(w[8], "z")};
        break;
      }
      default:
        throw Error(
            "vulkore::Program: unsupported ClspvReflection instruction " +
            std::to_string(inst) + " (" + reflection_inst_name(inst) +
            ") — vulkore v1 supports storage-buffer args and -pod-pushconstant "
            "PODs only (compile kernels like tests/kernels/README.md)");
    }
  });

  if (impl.kernels.empty()) {
    throw Error("vulkore::Program: module contains no kernels");
  }

  // Normalize each kernel: ordinal order, contiguity, push-constant extent.
  for (auto& kp : impl.kernels) {
    KernelInfo& ki = kp->info;
    std::sort(ki.args.begin(), ki.args.end(),
              [](const KernelArg& a, const KernelArg& b) {
                return a.ordinal < b.ordinal;
              });
    for (uint32_t i = 0; i < ki.args.size(); ++i) {
      if (ki.args[i].ordinal != i) {
        throw Error("vulkore::Program: kernel '" + ki.name +
                    "' has a non-contiguous argument list (unsupported arg "
                    "kind was skipped?)");
      }
      if (ki.args[i].kind == ArgKind::PodPushConstant) {
        ki.push_constant_size = std::max(
            ki.push_constant_size, ki.args[i].offset + ki.args[i].size);
      }
    }
    ki.push_constant_size = (ki.push_constant_size + 3u) & ~3u;
  }
}

// Removes the instructions that need VK_KHR_shader_non_semantic_info, so the
// module loads on any Vulkan 1.1 driver: OpExtension "SPV_KHR_non_semantic_info",
// every OpExtInstImport of a NonSemantic.* set, and every OpExtInst using one.
// (Equivalent to spirv-opt --strip-reflect for clspv output; results of
// non-semantic instructions are only ever consumed by other non-semantic
// instructions, so dropping them wholesale is safe.)
std::vector<uint32_t> strip_non_semantic(std::span<const uint32_t> code) {
  std::vector<uint32_t> out;
  out.reserve(code.size());
  out.insert(out.end(), code.begin(), code.begin() + kHeaderWords);

  std::unordered_map<uint32_t, bool> nonsemantic_sets;
  for_each_instruction(code, [&](uint32_t op, const uint32_t* w, uint32_t wc) {
    bool drop = false;
    switch (op) {
      case kOpExtension:
        drop = literal_string(w, wc, 1) == "SPV_KHR_non_semantic_info";
        break;
      case kOpExtInstImport:
        if (literal_string(w, wc, 2).rfind("NonSemantic.", 0) == 0) {
          nonsemantic_sets[w[1]] = true;
          drop = true;
        }
        break;
      case kOpExtInst:
        drop = wc >= 5 && nonsemantic_sets.count(w[3]) > 0;
        break;
      default:
        break;
    }
    if (!drop) out.insert(out.end(), w, w + wc);
  });
  return out;
}

void build_pipeline(KernelState& st) {
  Context& ctx = *st.ctx;
  const VolkDeviceTable& t = ctx.table();
  const KernelInfo& ki = st.info;

  // Descriptor set layouts, dense from set 0 to the highest set used.
  uint32_t max_set = 0;
  for (const KernelArg& a : ki.args) {
    if (a.kind == ArgKind::StorageBuffer) max_set = std::max(max_set, a.set);
  }
  st.set_layouts.resize(ki.args.empty() ? 0 : max_set + 1, VK_NULL_HANDLE);
  for (uint32_t set = 0; set < st.set_layouts.size(); ++set) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const KernelArg& a : ki.args) {
      if (a.kind != ArgKind::StorageBuffer || a.set != set) continue;
      VkDescriptorSetLayoutBinding b{};
      b.binding = a.binding;
      b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b.descriptorCount = 1;
      b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings.push_back(b);
    }
    VkDescriptorSetLayoutCreateInfo li{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = static_cast<uint32_t>(bindings.size());
    li.pBindings = bindings.data();
    XP_CHECK(t.vkCreateDescriptorSetLayout(ctx.device(), &li, nullptr,
                                           &st.set_layouts[set]));
  }

  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, ki.push_constant_size};
  if (ki.push_constant_size >
      ctx.device_properties().limits.maxPushConstantsSize) {
    throw Error("vulkore::Program: kernel '" + ki.name + "' needs " +
                std::to_string(ki.push_constant_size) +
                " bytes of push constants; device limit is " +
                std::to_string(ctx.device_properties().limits.maxPushConstantsSize));
  }

  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = static_cast<uint32_t>(st.set_layouts.size());
  pli.pSetLayouts = st.set_layouts.data();
  pli.pushConstantRangeCount = ki.push_constant_size > 0 ? 1 : 0;
  pli.pPushConstantRanges = &range;
  XP_CHECK(t.vkCreatePipelineLayout(ctx.device(), &pli, nullptr,
                                    &st.pipeline_layout));

  // Workgroup size: reqd_work_group_size attribute wins; else default 64x1x1
  // (every Vulkan device guarantees >=128 invocations per group). clspv
  // exposes the size as spec constants; specialize them at pipeline creation.
  st.local_size = ki.has_reqd_workgroup_size ? ki.reqd_workgroup_size
                                             : std::array<uint32_t, 3>{64, 1, 1};

  VkSpecializationMapEntry entries[3];
  VkSpecializationInfo spec{};
  ProgramImpl& prog = *st.program;
  if (prog.has_wg_spec_ids) {
    for (uint32_t i = 0; i < 3; ++i) {
      entries[i] = {prog.wg_spec_ids[i], i * 4u, 4u};
    }
    spec.mapEntryCount = 3;
    spec.pMapEntries = entries;
    spec.dataSize = sizeof(st.local_size);
    spec.pData = st.local_size.data();
  }

  VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pci.stage.module = prog.shader_module;
  pci.stage.pName = ki.name.c_str();
  pci.stage.pSpecializationInfo = prog.has_wg_spec_ids ? &spec : nullptr;
  pci.layout = st.pipeline_layout;
  // TODO(pipeline-cache): feed a persistent VkPipelineCache here (big win on
  // Android first-run).
  XP_CHECK(t.vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &pci,
                                      nullptr, &st.pipeline));
  st.built = true;
}

}  // namespace

// ---- Kernel -------------------------------------------------------------

namespace {
const KernelState& checked(const KernelState* s) {
  if (!s) throw Error("vulkore::Kernel: empty handle (default-constructed?)");
  return *s;
}
}  // namespace

const KernelInfo& Kernel::info() const { return checked(state_).info; }
const std::string& Kernel::name() const { return checked(state_).info.name; }
Context& Kernel::context() const { return *checked(state_).ctx; }
std::array<uint32_t, 3> Kernel::local_size() const {
  return checked(state_).local_size;
}
VkPipeline Kernel::pipeline() const { return checked(state_).pipeline; }
VkPipelineLayout Kernel::pipeline_layout() const {
  return checked(state_).pipeline_layout;
}
std::span<const VkDescriptorSetLayout> Kernel::set_layouts() const {
  return checked(state_).set_layouts;
}

// ---- Program ------------------------------------------------------------

Program::Program(Context& ctx, std::span<const uint8_t> spirv)
    : impl_(std::make_unique<ProgramImpl>()) {
  impl_->ctx = &ctx;

  if (spirv.size() < kHeaderWords * 4 || spirv.size() % 4 != 0) {
    throw Error("vulkore::Program: SPIR-V blob has invalid size (" +
                std::to_string(spirv.size()) + " bytes)");
  }
  // SPIR-V words are 32-bit; copy to guarantee alignment.
  std::vector<uint32_t> code(spirv.size() / 4);
  std::memcpy(code.data(), spirv.data(), spirv.size());

  if (code[0] != kSpirvMagic) {
    throw Error(code[0] == 0x03022307u
                    ? "vulkore::Program: byte-swapped SPIR-V is not supported"
                    : "vulkore::Program: not a SPIR-V module (bad magic)");
  }

  parse_reflection(code, *impl_);

  const std::vector<uint32_t> stripped = strip_non_semantic(code);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = stripped.size() * 4;
  smci.pCode = stripped.data();
  XP_CHECK(ctx.table().vkCreateShaderModule(ctx.device(), &smci, nullptr,
                                            &impl_->shader_module));
}

Program Program::from_file(Context& ctx, const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw Error("vulkore::Program: cannot open '" + path + "'");
  const std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (!f.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw Error("vulkore::Program: failed reading '" + path + "'");
  }
  return Program(ctx, bytes);
}

Program::~Program() = default;
Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;

Kernel Program::kernel(const std::string& name) {
  KernelState* st = impl_->find(name);
  if (!st) {
    std::string names;
    for (const auto& k : impl_->kernels) names += " '" + k->info.name + "'";
    throw Error("vulkore::Program: no kernel '" + name +
                "' in module; it contains:" + names);
  }
  if (!st->built) build_pipeline(*st);
  return Kernel(st);
}

std::vector<std::string> Program::kernel_names() const {
  std::vector<std::string> names;
  names.reserve(impl_->kernels.size());
  for (const auto& k : impl_->kernels) names.push_back(k->info.name);
  return names;
}

}  // namespace vulkore
