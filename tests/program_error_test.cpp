// Program module — SPIR-V / NonSemantic.ClspvReflection PARSER error surface.
//
// program_test.cpp covers the happy path + two coarse rejects
// (GarbageBytesThrow, NonClspvSpirvThrows). This file targets each SPECIFIC
// error branch in src/program.cpp and asserts the exact message substring, so a
// passing test proves it hit the INTENDED code path (not a coincidental earlier
// throw).
//
// PORTABILITY: every SPIR-V blob is built from raw uint32 words IN-TEST (the
// `Spirv` builder below) — no spirv-as at runtime, no committed blobs — so this
// same binary runs on the Android phone with no tools present. This is possible
// because every error asserted here is thrown by parse_reflection(), which runs
// BEFORE strip_non_semantic()/vkCreateShaderModule() (program.cpp:496 precedes
// 498-503): the module never has to be Vulkan-valid, only well-formed enough to
// reach the branch under test.
#include <xpose/xpose.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace {

// ---- SPIR-V word-stream builder --------------------------------------------
// Emits a minimal little-endian SPIR-V module (host is little-endian on x86 and
// arm64, matching program.cpp's literal_string byte reads). Instruction word
// layout: word[0] = (word_count << 16) | opcode, followed by word_count-1
// operand words. Reflection OpExtInst layout mirrors program.cpp:230-232:
// w[1]=result type, w[2]=result id, w[3]=ext-inst set id, w[4]=instruction
// number, w[5..]=operands.

constexpr uint32_t kSpirvMagic = 0x07230203u;
constexpr uint32_t kSpirvVersion13 = 0x00010300u;

// SPIR-V literal string: nul-terminated, packed 4 bytes/word little-endian,
// zero-padded to a whole word (a full zero word when the length is a multiple
// of 4). Matches program.cpp:82-90.
std::vector<uint32_t> encode_string(const std::string& s) {
  std::vector<uint32_t> out;
  uint32_t word = 0;
  int shift = 0;
  for (unsigned char c : s) {
    word |= static_cast<uint32_t>(c) << shift;
    shift += 8;
    if (shift == 32) {
      out.push_back(word);
      word = 0;
      shift = 0;
    }
  }
  out.push_back(word);  // trailing word carries the nul terminator + padding
  return out;
}

std::vector<uint8_t> to_bytes(const std::vector<uint32_t>& words) {
  std::vector<uint8_t> b(words.size() * 4u);
  if (!words.empty()) std::memcpy(b.data(), words.data(), b.size());
  return b;
}

class Spirv {
 public:
  // Valid 5-word header: magic, version 1.3, generator 0, id-bound, schema 0.
  Spirv() : words_{kSpirvMagic, kSpirvVersion13, 0u, 4096u, 0u} {}

  // OpExtInstImport %id "name"  (opcode 11).
  void import(uint32_t result_id, const std::string& name) {
    const std::vector<uint32_t> s = encode_string(name);
    push_op(11u, 2u + static_cast<uint32_t>(s.size()));
    words_.push_back(result_id);
    append(s);
  }
  // OpString %id "text"  (opcode 7).
  void op_string(uint32_t result_id, const std::string& text) {
    const std::vector<uint32_t> s = encode_string(text);
    push_op(7u, 2u + static_cast<uint32_t>(s.size()));
    words_.push_back(result_id);
    append(s);
  }
  // OpTypeInt %id width signedness  (opcode 21).
  void type_int(uint32_t result_id, uint32_t width, uint32_t signedness) {
    push_op(21u, 4u);
    words_.push_back(result_id);
    words_.push_back(width);
    words_.push_back(signedness);
  }
  // OpConstant %type %id value  (opcode 43).
  void constant(uint32_t type_id, uint32_t result_id, uint32_t value) {
    push_op(43u, 4u);
    words_.push_back(type_id);
    words_.push_back(result_id);
    words_.push_back(value);
  }
  // OpExtInst %type %id %set number operands...  (opcode 12) — a reflection
  // instruction. word_count = 5 + operands.size().
  void ext_inst(uint32_t result_type, uint32_t result_id, uint32_t set_id,
                uint32_t number, std::initializer_list<uint32_t> operands) {
    push_op(12u, 5u + static_cast<uint32_t>(operands.size()));
    words_.push_back(result_type);
    words_.push_back(result_id);
    words_.push_back(set_id);
    words_.push_back(number);
    for (uint32_t o : operands) words_.push_back(o);
  }
  // Append a raw word verbatim (used to inject a malformed instruction header).
  void raw_word(uint32_t w) { words_.push_back(w); }

  std::vector<uint8_t> bytes() const { return to_bytes(words_); }

 private:
  void push_op(uint32_t opcode, uint32_t word_count) {
    words_.push_back((word_count << 16) | opcode);
  }
  void append(const std::vector<uint32_t>& s) {
    for (uint32_t w : s) words_.push_back(w);
  }
  std::vector<uint32_t> words_;
};

// Constructs a Program from `bytes` expecting an xpose::Error whose message
// contains `needle`; fails the test if no throw or the wrong message.
void expect_program_error(xpose::Context& ctx, const std::vector<uint8_t>& bytes,
                          const std::string& needle) {
  try {
    xpose::Program prog(ctx, bytes);
    ADD_FAILURE() << "expected xpose::Error containing \"" << needle << "\"";
  } catch (const xpose::Error& e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
        << "actual message: " << e.what();
  }
}

std::string write_temp(const std::string& name,
                       const std::vector<uint8_t>& data) {
  const std::string path = ::testing::TempDir() + name;
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!data.empty()) {
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  }
  return path;
}

class ProgramErrorTest : public ::testing::Test {
 protected:
  // Any device on any driver: these tests parse SPIR-V and never dispatch.
  static void SetUpTestSuite() { ctx_ = std::make_unique<xpose::Context>(); }
  static void TearDownTestSuite() { ctx_.reset(); }
  static std::unique_ptr<xpose::Context> ctx_;
};
std::unique_ptr<xpose::Context> ProgramErrorTest::ctx_;

// ---- G5: malformed / truncated SPIR-V guards -------------------------------
// Each sub-case asserts the specific "invalid/malformed/truncated" wording so
// it proves the exact guard fired.
TEST_F(ProgramErrorTest, MalformedSpirvGuards) {
  {  // for_each_instruction: word_count == 0 (program.cpp:161-164).
    SCOPED_TRACE("instruction word-count == 0");
    Spirv m;
    m.raw_word(0u);  // header word with word_count field == 0
    expect_program_error(*ctx_, m.bytes(), "malformed SPIR-V");
  }
  {  // Byte-swapped magic 0x03022307 (program.cpp:490-493).
    SCOPED_TRACE("byte-swapped magic");
    expect_program_error(
        *ctx_,
        to_bytes({0x03022307u, kSpirvVersion13, 0u, 4096u, 0u}),
        "byte-swapped");
  }
  {  // Total size not a multiple of 4, and >20 to hit the %4 branch
     // (program.cpp:482-485).
    SCOPED_TRACE("size not a multiple of 4");
    expect_program_error(*ctx_, std::vector<uint8_t>(22, 0xABu), "invalid size");
  }
  {  // Truncated Kernel reflection: wc 6 < 7 (program.cpp:245).
    SCOPED_TRACE("truncated Kernel");
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.ext_inst(99, 10, 1, /*Kernel*/ 1, {98});  // wc = 6
    expect_program_error(*ctx_, m.bytes(), "truncated Kernel");
  }
  {  // Truncated ArgumentStorageBuffer: wc 8 < 9 (program.cpp:260).
    SCOPED_TRACE("truncated ArgumentStorageBuffer");
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.ext_inst(99, 10, 1, /*ArgumentStorageBuffer*/ 3, {1, 2, 3});  // wc = 8
    expect_program_error(*ctx_, m.bytes(), "truncated ArgumentStorageBuffer");
  }
  {  // Truncated ArgumentPodPushConstant: wc 8 < 9 (program.cpp:272).
    SCOPED_TRACE("truncated ArgumentPodPushConstant");
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.ext_inst(99, 10, 1, /*ArgumentPodPushConstant*/ 7, {1, 2, 3});  // wc = 8
    expect_program_error(*ctx_, m.bytes(), "truncated ArgumentPodPushConstant");
  }
  {  // Truncated SpecConstantWorkgroupSize: wc 7 < 8 (program.cpp:283).
    SCOPED_TRACE("truncated SpecConstantWorkgroupSize");
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.ext_inst(99, 10, 1, /*SpecConstantWorkgroupSize*/ 12, {1, 2});  // wc = 7
    expect_program_error(*ctx_, m.bytes(), "truncated SpecConstantWorkgroupSize");
  }
  {  // Truncated PropertyRequiredWorkgroupSize: wc 8 < 9 (program.cpp:291).
    SCOPED_TRACE("truncated PropertyRequiredWorkgroupSize");
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.ext_inst(99, 10, 1, /*PropertyRequiredWorkgroupSize*/ 24, {1, 2, 3});  // wc = 8
    expect_program_error(*ctx_, m.bytes(),
                         "truncated PropertyRequiredWorkgroupSize");
  }
}

// ---- G7: reflection import present but zero Kernel instructions ------------
// Distinct from NonClspvSpirvThrows (which has NO import): here the
// "no NonSemantic.ClspvReflection" check passes, and the DISTINCT empty-kernels
// guard (program.cpp:310-312) fires.
TEST_F(ProgramErrorTest, NoKernelsThrows) {
  Spirv m;
  m.import(1, "NonSemantic.ClspvReflection.5");
  expect_program_error(*ctx_, m.bytes(), "module contains no kernels");
}

// ---- G14: Program::from_file on empty / non-word-aligned files -------------
// from_file (program.cpp:506-516) reads the bytes and forwards to the size
// guard (program.cpp:482-485).
TEST_F(ProgramErrorTest, EmptyOrTruncatedFile) {
  {
    SCOPED_TRACE("0-byte file");
    const std::string p = write_temp("xpose_prog_empty.spv", {});
    try {
      xpose::Program::from_file(*ctx_, p);
      ADD_FAILURE() << "expected throw on empty file";
    } catch (const xpose::Error& e) {
      EXPECT_NE(std::string(e.what()).find("invalid size"), std::string::npos)
          << e.what();
    }
    std::remove(p.c_str());
  }
  {
    SCOPED_TRACE("non-multiple-of-4 file");
    const std::string p =
        write_temp("xpose_prog_odd.spv", std::vector<uint8_t>(22, 0xABu));
    try {
      xpose::Program::from_file(*ctx_, p);
      ADD_FAILURE() << "expected throw on truncated file";
    } catch (const xpose::Error& e) {
      EXPECT_NE(std::string(e.what()).find("invalid size"), std::string::npos)
          << e.what();
    }
    std::remove(p.c_str());
  }
}

// ---- G1: unsupported reflection instruction throws NAMING the feature ------
// The whole "images/samplers/POD-in-UBO-or-SSBO/pointer-to-local/implicit push
// constants/printf -> throw at load naming the feature" contract
// (reflection_inst_name table program.cpp:40-78 + default throw 301-306). Each
// module is well-formed through a real Kernel, then carries ONE unsupported
// reflection OpExtInst; the message must name the human feature.
TEST_F(ProgramErrorTest, UnsupportedReflectionInstThrows) {
  auto module_with_unsupported = [](uint32_t number) {
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.op_string(2, "k");
    m.ext_inst(99, 10, 1, /*Kernel*/ 1, {98, 2});  // valid Kernel "k"
    m.ext_inst(99, 11, 1, number, {10});           // the unsupported inst
    return m.bytes();
  };

  struct Case {
    uint32_t number;
    const char* needle;  // human feature name from reflection_inst_name()
  };
  // Representative unsupported features across every rejected category:
  const Case cases[] = {
      {4, "ArgumentUniform"},               // UBO buffer arg
      {5, "ArgumentPodStorageBuffer"},      // POD-in-SSBO
      {6, "ArgumentPodUniform"},            // POD-in-UBO
      {8, "ArgumentSampledImage"},          // image/sampler
      {9, "ArgumentStorageImage"},          // storage image
      {10, "ArgumentSampler"},              // sampler
      {11, "ArgumentWorkgroup"},            // pointer-to-local
      {15, "PushConstantGlobalOffset"},     // implicit push constant: offset
      {19, "PushConstantNumWorkgroups"},    // implicit push constant: numgroups
      {23, "LiteralSampler"},               // literal sampler
      {38, "PrintfInfo"},                   // printf
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(std::string("reflection inst ") + std::to_string(c.number) +
                 " -> " + c.needle);
    try {
      xpose::Program prog(*ctx_, module_with_unsupported(c.number));
      ADD_FAILURE() << "expected unsupported-feature throw";
    } catch (const xpose::Error& e) {
      const std::string msg = e.what();
      EXPECT_NE(msg.find("unsupported ClspvReflection instruction"),
                std::string::npos)
          << msg;
      EXPECT_NE(msg.find(c.needle), std::string::npos) << msg;
    }
  }
}

// ---- G6: reflection arg references an unknown kernel id --------------------
// kernel_of() (program.cpp:235-242). ordinal/set/binding are valid constants,
// so kernel_of is the throw site (not an earlier constant() failure).
TEST_F(ProgramErrorTest, UnknownKernelRefThrows) {
  Spirv m;
  m.import(1, "NonSemantic.ClspvReflection.5");
  m.type_int(3, 32, 0);
  m.constant(3, 4, 0);  // reused for ordinal/set/binding = 0
  m.ext_inst(99, 11, 1, /*ArgumentStorageBuffer*/ 3, {999, 4, 4, 4});
  expect_program_error(*ctx_, m.bytes(), "references unknown kernel");
}

// ---- G8: reflection operand has the wrong id kind --------------------------
// constant() (program.cpp:178-185) and string() (186-193).
TEST_F(ProgramErrorTest, NonIntegerReflectionOperandThrows) {
  {  // ordinal id 777 is not an OpConstant -> constant() throws.
    SCOPED_TRACE("ordinal operand is not an OpConstant");
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.ext_inst(99, 11, 1, /*ArgumentStorageBuffer*/ 3, {1, 777, 777, 777});
    expect_program_error(*ctx_, m.bytes(), "is not a 32-bit integer constant");
  }
  {  // kernel name id 888 is not an OpString -> string() throws.
    SCOPED_TRACE("kernel name operand is not an OpString");
    Spirv m;
    m.import(1, "NonSemantic.ClspvReflection.5");
    m.ext_inst(99, 10, 1, /*Kernel*/ 1, {98, 888});
    expect_program_error(*ctx_, m.bytes(), "is not an OpString");
  }
}

// ---- G16: kernel with a non-contiguous ordinal list ------------------------
// Normalization contiguity check (program.cpp:321-326): two args with ordinals
// {0, 2} leave a gap at ordinal 1.
TEST_F(ProgramErrorTest, NonContiguousArgsThrows) {
  Spirv m;
  m.import(1, "NonSemantic.ClspvReflection.5");
  m.op_string(2, "k");
  m.type_int(3, 32, 0);
  m.constant(3, 4, 0);  // value 0
  m.constant(3, 5, 2);  // value 2 (the gap: no ordinal 1)
  m.constant(3, 6, 1);  // value 1
  m.ext_inst(99, 10, 1, /*Kernel*/ 1, {98, 2});           // kernel "k"
  m.ext_inst(99, 11, 1, /*ArgumentStorageBuffer*/ 3,
             {10, 4, 4, 4});  // ordinal 0, set 0, binding 0
  m.ext_inst(99, 12, 1, /*ArgumentStorageBuffer*/ 3,
             {10, 5, 4, 6});  // ordinal 2, set 0, binding 1
  expect_program_error(*ctx_, m.bytes(), "non-contiguous argument list");
}

}  // namespace
