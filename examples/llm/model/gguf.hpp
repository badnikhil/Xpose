// GGUF v3 reader — host-side, mmap'd, zero-copy. No Vulkan, no GPU.
//
// Layout notes and the exact key names live in agent-docs/gguf-loader.md.
// Summary of the on-disk file:
//
//   magic u32 "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64
//   metadata_kv_count x { key:string, value_type:u32, value }
//   tensor_count     x { name:string, n_dims:u32, dims:u64[n_dims],
//                        type:u32, offset:u64 }
//   <pad to general.alignment>
//   tensor data blob   (tensor.offset is relative to the START of this blob)
//
// A GGUF "string" is u64 length + that many bytes, NOT NUL-terminated. We hand
// them out as std::string_view pointing straight into the mapping: a Gemma 3
// vocab is 262144 entries, and copying those into std::string costs ~15 MB of
// heap for no reason on a phone. The consequence is that EVERY view and every
// tensor pointer dies with the GGUFFile — it is move-only and non-copyable to
// make that hard to get wrong.
//
// The whole file is mmap'd MAP_PRIVATE and never read() into a heap buffer:
// a 1B Q4_K_M model is ~700 MB, which is most of a phone's free RAM.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vulkore::llm {

// ---------------------------------------------------------------------------
// ggml tensor types. Values are ABI — they are written into the file.
// ---------------------------------------------------------------------------
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // 4 and 5 are the removed Q4_2/Q4_3 — never reused.
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    COUNT   = 40,
};

struct TypeTraits {
    const char* name;
    uint32_t    block_size;  // elements per block (1 for unquantised types)
    uint32_t    type_size;   // bytes per block
    bool        dequantisable;  // do we implement dequantize() for it?
};

// Never throws; unknown types come back with block_size 0 and name "UNKNOWN".
const TypeTraits& type_traits(GGMLType t);

// ---------------------------------------------------------------------------
// Metadata values.
// ---------------------------------------------------------------------------
enum class MetaType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12,
};

const char* meta_type_name(MetaType t);

// One key/value pair. Scalars are widened into u64/i64/f64 (all three are
// filled for numeric types so callers can read whichever they want without
// caring how the file happened to encode it). Arrays land in exactly one of
// arr_str / arr_num depending on the element type.
struct MetaValue {
    MetaType type      = MetaType::UINT32;
    MetaType elem_type = MetaType::UINT32;  // only meaningful when type == ARRAY

    uint64_t    u = 0;
    int64_t     i = 0;
    double      f = 0.0;
    std::string_view str;

    std::vector<std::string_view> arr_str;
    // Numeric arrays are widened to double. A 262144-entry f32 score array
    // costs 2 MB this way; the simplification is worth it and it is the only
    // large numeric array a Gemma file actually contains.
    std::vector<double> arr_num;

    size_t array_length() const {
        return type == MetaType::ARRAY
                   ? (elem_type == MetaType::STRING ? arr_str.size() : arr_num.size())
                   : 0;
    }
};

// ---------------------------------------------------------------------------
// Tensors.
// ---------------------------------------------------------------------------
struct Tensor {
    std::string_view      name;
    std::vector<uint64_t> shape;   // GGUF order: shape[0] is the FASTEST-varying
                                   // axis (a row length), i.e. reversed vs numpy.
    GGMLType              type   = GGMLType::F32;
    uint64_t              offset = 0;  // byte offset from the tensor-data blob
    const uint8_t*        data   = nullptr;  // absolute pointer into the mapping
    uint64_t              nbytes = 0;

    uint64_t n_elements() const {
        uint64_t n = 1;
        for (uint64_t d : shape) n *= d;
        return n;
    }
};

// ---------------------------------------------------------------------------
// Model config, pulled out of metadata. Field names follow the GGUF keys, not
// the HF config.json names.
// ---------------------------------------------------------------------------
struct ModelConfig {
    std::string arch;             // general.architecture, e.g. "gemma3"
    std::string name;             // general.name

    uint32_t n_layers            = 0;  // <arch>.block_count
    uint32_t n_heads             = 0;  // <arch>.attention.head_count
    uint32_t n_kv_heads          = 0;  // <arch>.attention.head_count_kv
    uint32_t embedding_length    = 0;  // <arch>.embedding_length
    uint32_t feed_forward_length = 0;  // <arch>.feed_forward_length
    uint32_t context_length      = 0;  // <arch>.context_length
    uint32_t key_length          = 0;  // <arch>.attention.key_length   (head dim)
    uint32_t value_length        = 0;  // <arch>.attention.value_length
    uint32_t vocab_size          = 0;  // tokenizer.ggml.tokens length
    // <arch>.attention.sliding_window. 0 == absent, i.e. every layer is global.
    // Gemma 3 sets 512 and alternates 5 windowed layers to 1 global one; the
    // ALTERNATION pattern is not in the file, only the extent.
    uint32_t sliding_window      = 0;

    float rope_freq_base = 10000.0f;   // <arch>.rope.freq_base
    float rms_eps        = 1e-6f;      // <arch>.attention.layer_norm_rms_epsilon

    // head_dim: key_length when present, else embedding_length / n_heads.
    uint32_t head_dim() const {
        if (key_length) return key_length;
        return n_heads ? embedding_length / n_heads : 0;
    }
};

// ---------------------------------------------------------------------------
// The file.
// ---------------------------------------------------------------------------
class GGUFFile {
public:
    GGUFFile() = default;
    ~GGUFFile();

    GGUFFile(const GGUFFile&)            = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;
    GGUFFile(GGUFFile&&) noexcept;
    GGUFFile& operator=(GGUFFile&&) noexcept;

    // Throws std::runtime_error on open/mmap failure or a malformed file.
    static GGUFFile open(const std::string& path);

    uint32_t version() const { return version_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t file_size() const { return size_; }
    uint64_t tensor_data_offset() const { return data_offset_; }

    const std::vector<Tensor>& tensors() const { return tensors_; }
    // nullptr if absent.
    const Tensor* tensor(std::string_view name) const;

    const std::vector<std::pair<std::string_view, MetaValue>>& metadata() const {
        return meta_order_;
    }
    const MetaValue* meta(std::string_view key) const;

    // Typed convenience accessors; nullopt when the key is missing.
    std::optional<uint64_t>         get_u64(std::string_view key) const;
    std::optional<int64_t>          get_i64(std::string_view key) const;
    std::optional<double>           get_f64(std::string_view key) const;
    std::optional<bool>             get_bool(std::string_view key) const;
    std::optional<std::string_view> get_str(std::string_view key) const;

    ModelConfig config() const;

    // ---- dequantisation -------------------------------------------------
    // Writes n_elements floats to dst. n_elements must be a whole number of
    // blocks for the type. Returns false for types we do not implement.
    static bool dequantize(GGMLType type, const void* src, float* dst, size_t n_elements);
    // Whole tensor -> heap. Caller pays n_elements * 4 bytes: do not call this
    // on token_embd of a 262144-vocab model unless you mean it (1.2 GB).
    std::vector<float> dequantize_tensor(const Tensor& t) const;

private:
    void parse();
    void reset();

    const uint8_t* base_ = nullptr;   // mmap base
    size_t         size_ = 0;
    int            fd_   = -1;

    uint32_t version_     = 0;
    uint64_t alignment_   = 32;
    uint64_t data_offset_ = 0;

    std::vector<Tensor> tensors_;
    std::unordered_map<std::string_view, size_t> tensor_index_;

    std::vector<std::pair<std::string_view, MetaValue>> meta_order_;
    std::unordered_map<std::string_view, size_t> meta_index_;
};

// fp16 -> fp32, portable (no _Float16 / F16C assumptions; the NDK's arm64
// target has __fp16 but desktop gcc/clang disagree on the spelling).
float fp16_to_fp32(uint16_t h);

}  // namespace vulkore::llm
