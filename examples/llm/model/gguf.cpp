#include "gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace vulkore::llm {
namespace {

constexpr uint32_t kMagic = 0x46554747u;  // "GGUF" little-endian

// K-quant super-block size. Every *_K type packs 256 elements.
constexpr uint32_t QK_K = 256;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("gguf: " + msg); }

// Bounds-checked cursor over the mapping. Every field read goes through this;
// a truncated or hostile file must throw, not segfault. We are mmap'ing an
// untrusted ~700 MB blob and indexing it with u64 lengths straight out of the
// file, so unchecked reads are the obvious way to get a SIGBUS.
class Reader {
public:
    Reader(const uint8_t* base, size_t size) : p_(base), end_(base + size), base_(base) {}

    size_t pos() const { return static_cast<size_t>(p_ - base_); }

    void need(size_t n) const {
        if (n > static_cast<size_t>(end_ - p_))
            fail("truncated file: wanted " + std::to_string(n) + " bytes at offset " +
                 std::to_string(pos()));
    }

    template <typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }

    std::string_view read_string() {
        uint64_t len = read<uint64_t>();
        need(static_cast<size_t>(len));
        std::string_view s(reinterpret_cast<const char*>(p_), static_cast<size_t>(len));
        p_ += len;
        return s;
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
    const uint8_t* base_;
};

size_t meta_scalar_size(MetaType t) {
    switch (t) {
        case MetaType::UINT8:
        case MetaType::INT8:
        case MetaType::BOOL:    return 1;
        case MetaType::UINT16:
        case MetaType::INT16:   return 2;
        case MetaType::UINT32:
        case MetaType::INT32:
        case MetaType::FLOAT32: return 4;
        case MetaType::UINT64:
        case MetaType::INT64:
        case MetaType::FLOAT64: return 8;
        default: return 0;  // STRING / ARRAY are variable-length
    }
}

// Read one non-array value into v, filling u/i/f for numeric types so callers
// need not care whether the writer chose u32 or u64 for e.g. block_count.
void read_scalar(Reader& r, MetaType t, MetaValue& v) {
    switch (t) {
        case MetaType::UINT8:   { auto x = r.read<uint8_t>();  v.u = x; v.i = x; v.f = x; break; }
        case MetaType::INT8:    { auto x = r.read<int8_t>();   v.u = static_cast<uint64_t>(x); v.i = x; v.f = x; break; }
        case MetaType::UINT16:  { auto x = r.read<uint16_t>(); v.u = x; v.i = x; v.f = x; break; }
        case MetaType::INT16:   { auto x = r.read<int16_t>();  v.u = static_cast<uint64_t>(x); v.i = x; v.f = x; break; }
        case MetaType::UINT32:  { auto x = r.read<uint32_t>(); v.u = x; v.i = x; v.f = x; break; }
        case MetaType::INT32:   { auto x = r.read<int32_t>();  v.u = static_cast<uint64_t>(x); v.i = x; v.f = x; break; }
        case MetaType::FLOAT32: { auto x = r.read<float>();    v.f = x; v.i = static_cast<int64_t>(x); v.u = static_cast<uint64_t>(v.i); break; }
        case MetaType::FLOAT64: { auto x = r.read<double>();   v.f = x; v.i = static_cast<int64_t>(x); v.u = static_cast<uint64_t>(v.i); break; }
        case MetaType::BOOL:    { auto x = r.read<uint8_t>();  v.u = x ? 1 : 0; v.i = v.u; v.f = static_cast<double>(v.u); break; }
        case MetaType::STRING:  { v.str = r.read_string(); break; }
        default: fail("read_scalar on non-scalar type " + std::to_string(static_cast<uint32_t>(t)));
    }
}

double read_num_as_double(Reader& r, MetaType t) {
    MetaValue tmp;
    read_scalar(r, t, tmp);
    return (t == MetaType::FLOAT32 || t == MetaType::FLOAT64) ? tmp.f
           : (t == MetaType::INT8 || t == MetaType::INT16 || t == MetaType::INT32 ||
              t == MetaType::INT64)
               ? static_cast<double>(tmp.i)
               : static_cast<double>(tmp.u);
}

}  // namespace

// ---------------------------------------------------------------------------
// fp16
// ---------------------------------------------------------------------------
float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +/- zero
        } else {
            // Subnormal: renormalise into a float32 normal.
            uint32_t m = mant, e = 0;
            while ((m & 0x400u) == 0) { m <<= 1; ++e; }
            m &= 0x3FFu;
            bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---------------------------------------------------------------------------
// Type table
// ---------------------------------------------------------------------------
const TypeTraits& type_traits(GGMLType t) {
    // Block sizes are ABI, taken from ggml.c's type_traits[]. Getting one of
    // these wrong silently shifts every subsequent tensor, so they are spelled
    // out rather than computed.
    static const TypeTraits table[] = {
        /* 0  F32     */ {"F32",      1,   4,   true},
        /* 1  F16     */ {"F16",      1,   2,   true},
        /* 2  Q4_0    */ {"Q4_0",    32,  18,   true},   // fp16 d + 16B nibbles
        /* 3  Q4_1    */ {"Q4_1",    32,  20,   true},
        /* 4  ---     */ {"REMOVED",  0,   0,   false},
        /* 5  ---     */ {"REMOVED",  0,   0,   false},
        /* 6  Q5_0    */ {"Q5_0",    32,  22,   true},
        /* 7  Q5_1    */ {"Q5_1",    32,  24,   true},
        /* 8  Q8_0    */ {"Q8_0",    32,  34,   true},
        /* 9  Q8_1    */ {"Q8_1",    32,  40,   false},
        /* 10 Q2_K    */ {"Q2_K",   256,  84,   false},
        /* 11 Q3_K    */ {"Q3_K",   256, 110,   false},
        /* 12 Q4_K    */ {"Q4_K",   256, 144,   true},
        /* 13 Q5_K    */ {"Q5_K",   256, 176,   false},
        /* 14 Q6_K    */ {"Q6_K",   256, 210,   true},
        /* 15 Q8_K    */ {"Q8_K",   256, 292,   false},
        /* 16 IQ2_XXS */ {"IQ2_XXS",256,  66,   false},
        /* 17 IQ2_XS  */ {"IQ2_XS", 256,  74,   false},
        /* 18 IQ3_XXS */ {"IQ3_XXS",256,  98,   false},
        /* 19 IQ1_S   */ {"IQ1_S",  256,  50,   false},
        /* 20 IQ4_NL  */ {"IQ4_NL",  32,  18,   false},
        /* 21 IQ3_S   */ {"IQ3_S",  256, 110,   false},
        /* 22 IQ2_S   */ {"IQ2_S",  256,  82,   false},
        /* 23 IQ4_XS  */ {"IQ4_XS", 256, 136,   false},
        /* 24 I8      */ {"I8",       1,   1,   false},
        /* 25 I16     */ {"I16",      1,   2,   false},
        /* 26 I32     */ {"I32",      1,   4,   false},
        /* 27 I64     */ {"I64",      1,   8,   false},
        /* 28 F64     */ {"F64",      1,   8,   false},
        /* 29 IQ1_M   */ {"IQ1_M",  256,  56,   false},
        /* 30 BF16    */ {"BF16",     1,   2,   true},
    };
    static const TypeTraits unknown{"UNKNOWN", 0, 0, false};
    const auto idx = static_cast<uint32_t>(t);
    if (idx >= sizeof(table) / sizeof(table[0])) return unknown;
    return table[idx];
}

const char* meta_type_name(MetaType t) {
    switch (t) {
        case MetaType::UINT8:   return "u8";
        case MetaType::INT8:    return "i8";
        case MetaType::UINT16:  return "u16";
        case MetaType::INT16:   return "i16";
        case MetaType::UINT32:  return "u32";
        case MetaType::INT32:   return "i32";
        case MetaType::FLOAT32: return "f32";
        case MetaType::BOOL:    return "bool";
        case MetaType::STRING:  return "str";
        case MetaType::ARRAY:   return "array";
        case MetaType::UINT64:  return "u64";
        case MetaType::INT64:   return "i64";
        case MetaType::FLOAT64: return "f64";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// GGUFFile
// ---------------------------------------------------------------------------
GGUFFile::~GGUFFile() { reset(); }

void GGUFFile::reset() {
    if (base_) munmap(const_cast<uint8_t*>(base_), size_);
    if (fd_ >= 0) ::close(fd_);
    base_ = nullptr;
    size_ = 0;
    fd_   = -1;
}

GGUFFile::GGUFFile(GGUFFile&& o) noexcept { *this = std::move(o); }

GGUFFile& GGUFFile::operator=(GGUFFile&& o) noexcept {
    if (this == &o) return *this;
    reset();
    base_         = std::exchange(o.base_, nullptr);
    size_         = std::exchange(o.size_, 0);
    fd_           = std::exchange(o.fd_, -1);
    version_      = o.version_;
    alignment_    = o.alignment_;
    data_offset_  = o.data_offset_;
    tensors_      = std::move(o.tensors_);
    tensor_index_ = std::move(o.tensor_index_);
    meta_order_   = std::move(o.meta_order_);
    meta_index_   = std::move(o.meta_index_);
    return *this;
}

GGUFFile GGUFFile::open(const std::string& path) {
    GGUFFile f;
    f.fd_ = ::open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0) fail("open('" + path + "'): " + std::strerror(errno));

    struct stat st{};
    if (::fstat(f.fd_, &st) != 0) fail("fstat: " + std::string(std::strerror(errno)));
    if (st.st_size <= 0) fail("empty file");
    f.size_ = static_cast<size_t>(st.st_size);

    void* m = ::mmap(nullptr, f.size_, PROT_READ, MAP_PRIVATE, f.fd_, 0);
    if (m == MAP_FAILED) fail("mmap: " + std::string(std::strerror(errno)));
    f.base_ = static_cast<const uint8_t*>(m);

    // Weights are streamed once, mostly sequentially, and the kernel should
    // feel free to drop the pages again — this is the whole point of not
    // read()ing 700 MB into the heap on a 6 GB phone.
    ::madvise(m, f.size_, MADV_SEQUENTIAL);

    f.parse();
    return f;
}

void GGUFFile::parse() {
    Reader r(base_, size_);

    if (r.read<uint32_t>() != kMagic) fail("bad magic (not a GGUF file)");
    version_ = r.read<uint32_t>();
    if (version_ != 2 && version_ != 3)
        fail("unsupported version " + std::to_string(version_) + " (want 2 or 3)");

    const uint64_t n_tensors = r.read<uint64_t>();
    const uint64_t n_kv      = r.read<uint64_t>();
    // Sanity-clamp before reserving: these are u64 straight from the file.
    if (n_tensors > (1u << 24) || n_kv > (1u << 24)) fail("implausible tensor/kv count");

    // ---- metadata ----
    meta_order_.reserve(static_cast<size_t>(n_kv));
    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string_view key = r.read_string();
        MetaValue v;
        v.type = static_cast<MetaType>(r.read<uint32_t>());

        if (v.type == MetaType::ARRAY) {
            v.elem_type      = static_cast<MetaType>(r.read<uint32_t>());
            const uint64_t n = r.read<uint64_t>();
            // Every element costs at least one byte on disk, so a length that
            // exceeds the file cannot be honest. Checking here keeps the
            // reserve() below from being handed a u64 straight from the file
            // (and keeps `n * esz` from overflowing size_t).
            if (n > size_) fail("array length " + std::to_string(n) + " exceeds file size");
            if (v.elem_type == MetaType::STRING) {
                // Cheap guard: each entry is at least 8 bytes of length prefix.
                r.need(static_cast<size_t>(n) * 8);
                v.arr_str.reserve(static_cast<size_t>(n));
                for (uint64_t j = 0; j < n; ++j) v.arr_str.push_back(r.read_string());
            } else if (v.elem_type == MetaType::ARRAY) {
                fail("nested arrays are not supported");
            } else {
                const size_t esz = meta_scalar_size(v.elem_type);
                if (esz == 0) fail("array of unknown element type");
                r.need(static_cast<size_t>(n) * esz);
                v.arr_num.reserve(static_cast<size_t>(n));
                for (uint64_t j = 0; j < n; ++j)
                    v.arr_num.push_back(read_num_as_double(r, v.elem_type));
            }
        } else {
            read_scalar(r, v.type, v);
        }

        meta_index_.emplace(key, meta_order_.size());
        meta_order_.emplace_back(key, std::move(v));
    }

    if (auto a = get_u64("general.alignment")) alignment_ = *a;
    if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0)
        fail("general.alignment is not a power of two");

    // ---- tensor descriptors ----
    tensors_.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t t = 0; t < n_tensors; ++t) {
        Tensor ti;
        ti.name              = r.read_string();
        const uint32_t ndims = r.read<uint32_t>();
        if (ndims > 4) fail("tensor '" + std::string(ti.name) + "' has " +
                            std::to_string(ndims) + " dims (max 4)");
        ti.shape.reserve(ndims);
        for (uint32_t d = 0; d < ndims; ++d) ti.shape.push_back(r.read<uint64_t>());
        ti.type   = static_cast<GGMLType>(r.read<uint32_t>());
        ti.offset = r.read<uint64_t>();

        const TypeTraits& tt = type_traits(ti.type);
        if (tt.block_size == 0)
            fail("tensor '" + std::string(ti.name) + "' has unknown type " +
                 std::to_string(static_cast<uint32_t>(ti.type)));
        const uint64_t ne = ti.n_elements();
        if (ne % tt.block_size != 0)
            fail("tensor '" + std::string(ti.name) + "': " + std::to_string(ne) +
                 " elements is not a whole number of " + tt.name + " blocks");
        ti.nbytes = ne / tt.block_size * tt.type_size;

        tensor_index_.emplace(ti.name, tensors_.size());
        tensors_.push_back(std::move(ti));
    }

    // ---- data blob ----
    data_offset_ = (r.pos() + alignment_ - 1) & ~(alignment_ - 1);
    if (data_offset_ > size_) fail("tensor data offset past end of file");

    for (Tensor& t : tensors_) {
        const uint64_t start = data_offset_ + t.offset;
        if (start > size_ || t.nbytes > size_ - start)
            fail("tensor '" + std::string(t.name) + "' data [" + std::to_string(start) +
                 ", +" + std::to_string(t.nbytes) + ") runs past end of file (" +
                 std::to_string(size_) + ")");
        t.data = base_ + start;
    }
}

const Tensor* GGUFFile::tensor(std::string_view name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const MetaValue* GGUFFile::meta(std::string_view key) const {
    auto it = meta_index_.find(key);
    return it == meta_index_.end() ? nullptr : &meta_order_[it->second].second;
}

std::optional<uint64_t> GGUFFile::get_u64(std::string_view k) const {
    if (const MetaValue* v = meta(k)) return v->u;
    return std::nullopt;
}
std::optional<int64_t> GGUFFile::get_i64(std::string_view k) const {
    if (const MetaValue* v = meta(k)) return v->i;
    return std::nullopt;
}
std::optional<double> GGUFFile::get_f64(std::string_view k) const {
    if (const MetaValue* v = meta(k)) return v->f;
    return std::nullopt;
}
std::optional<bool> GGUFFile::get_bool(std::string_view k) const {
    if (const MetaValue* v = meta(k)) return v->u != 0;
    return std::nullopt;
}
std::optional<std::string_view> GGUFFile::get_str(std::string_view k) const {
    const MetaValue* v = meta(k);
    if (!v || v->type != MetaType::STRING) return std::nullopt;
    return v->str;
}

ModelConfig GGUFFile::config() const {
    ModelConfig c;
    c.arch = std::string(get_str("general.architecture").value_or("unknown"));
    c.name = std::string(get_str("general.name").value_or(""));

    // Config keys are namespaced by the architecture string, so a Gemma file
    // says "gemma3.block_count" and a Llama one says "llama.block_count".
    // Building the key from general.architecture is what llama.cpp does and it
    // means this loader is not actually Gemma-specific.
    const std::string p = c.arch + ".";
    auto u32 = [&](const std::string& suffix, uint32_t dflt) -> uint32_t {
        auto v = get_u64(p + suffix);
        return v ? static_cast<uint32_t>(*v) : dflt;
    };
    auto f32 = [&](const std::string& suffix, float dflt) -> float {
        auto v = get_f64(p + suffix);
        return v ? static_cast<float>(*v) : dflt;
    };

    c.n_layers            = u32("block_count", 0);
    c.n_heads             = u32("attention.head_count", 0);
    c.n_kv_heads          = u32("attention.head_count_kv", c.n_heads);
    c.embedding_length    = u32("embedding_length", 0);
    c.feed_forward_length = u32("feed_forward_length", 0);
    c.context_length      = u32("context_length", 0);
    c.key_length          = u32("attention.key_length", 0);
    c.value_length        = u32("attention.value_length", 0);
    c.sliding_window      = u32("attention.sliding_window", 0);
    c.rope_freq_base      = f32("rope.freq_base", 10000.0f);
    c.rms_eps             = f32("attention.layer_norm_rms_epsilon", 1e-6f);

    // Vocab size is the token-array length, NOT a scalar key. Some writers do
    // emit <arch>.vocab_size, most do not; the array is always authoritative.
    if (const MetaValue* toks = meta("tokenizer.ggml.tokens"))
        c.vocab_size = static_cast<uint32_t>(toks->arr_str.size());
    else
        c.vocab_size = u32("vocab_size", 0);

    return c;
}

// ---------------------------------------------------------------------------
// Dequantisation. Block layouts transcribed from ggml-quants.c; the K-quant
// ones are the fiddly bit and are commented block-by-block.
// ---------------------------------------------------------------------------
namespace {

// Q4_K packs 8 six-bit scales and 8 six-bit mins into 12 bytes: the first four
// pairs live in the low 6 bits of scales[0..7], and the remaining four pairs
// are split — low nibble in scales[8..11], high 2 bits borrowed from the top
// of scales[0..7]. This is the single most error-prone routine in the file.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

}  // namespace

bool GGUFFile::dequantize(GGMLType type, const void* src, float* dst, size_t n) {
    const auto* p = static_cast<const uint8_t*>(src);
    const TypeTraits& tt = type_traits(type);
    if (!tt.dequantisable || tt.block_size == 0) return false;
    if (n % tt.block_size != 0) return false;
    const size_t nb = n / tt.block_size;

    switch (type) {
        case GGMLType::F32:
            std::memcpy(dst, p, n * 4);
            return true;

        case GGMLType::F16:
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, p + i * 2, 2);
                dst[i] = fp16_to_fp32(h);
            }
            return true;

        case GGMLType::BF16:
            // bf16 is just the top 16 bits of an fp32.
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, p + i * 2, 2);
                const uint32_t bits = static_cast<uint32_t>(h) << 16;
                std::memcpy(&dst[i], &bits, 4);
            }
            return true;

        case GGMLType::Q4_0:
            // 18 B/block: fp16 scale d, then 16 bytes of packed nibbles. The
            // nibbles are NOT sequential: byte j holds elements j (low) and
            // j+16 (high). Quants are unsigned 0..15, biased by -8.
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t* blk = p + b * 18;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const float d  = fp16_to_fp32(dh);
                const uint8_t* qs = blk + 2;
                float* y = dst + b * 32;
                for (int j = 0; j < 16; ++j) {
                    y[j]      = static_cast<float>(static_cast<int>(qs[j] & 0x0F) - 8) * d;
                    y[j + 16] = static_cast<float>(static_cast<int>(qs[j] >> 4) - 8) * d;
                }
            }
            return true;

        case GGMLType::Q4_1:
            // 20 B/block: fp16 d, fp16 m, 16 B nibbles. Affine rather than
            // symmetric, so quants are unsigned with no -8 bias.
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t* blk = p + b * 20;
                uint16_t dh, mh;
                std::memcpy(&dh, blk, 2);
                std::memcpy(&mh, blk + 2, 2);
                const float d = fp16_to_fp32(dh), m = fp16_to_fp32(mh);
                const uint8_t* qs = blk + 4;
                float* y = dst + b * 32;
                for (int j = 0; j < 16; ++j) {
                    y[j]      = static_cast<float>(qs[j] & 0x0F) * d + m;
                    y[j + 16] = static_cast<float>(qs[j] >> 4) * d + m;
                }
            }
            return true;

        case GGMLType::Q5_0:
        case GGMLType::Q5_1: {
            // 5-bit: the low nibbles live in the 16-byte qs array exactly as in
            // Q4_0, and the 5th bit of all 32 values is spread across a u32
            // bitmask `qh` — bit j is element j's high bit, bit j+16 is element
            // j+16's. Hence the (j) and (j+12) shifts: the second one lands the
            // bit in position 4 after the & 0x10.
            const bool is_1 = (type == GGMLType::Q5_1);
            const size_t stride = is_1 ? 24 : 22;
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t* blk = p + b * stride;
                uint16_t dh, mh = 0;
                std::memcpy(&dh, blk, 2);
                if (is_1) std::memcpy(&mh, blk + 2, 2);
                const float d = fp16_to_fp32(dh);
                const float m = is_1 ? fp16_to_fp32(mh) : 0.0f;
                uint32_t qh;
                std::memcpy(&qh, blk + (is_1 ? 4 : 2), 4);
                const uint8_t* qs = blk + (is_1 ? 8 : 6);
                float* y = dst + b * 32;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t xh0 = static_cast<uint8_t>((qh >> (j + 0)) << 4) & 0x10;
                    const uint8_t xh1 = static_cast<uint8_t>(qh >> (j + 12)) & 0x10;
                    const int q0 = static_cast<int>((qs[j] & 0x0F) | xh0);
                    const int q1 = static_cast<int>((qs[j] >> 4) | xh1);
                    if (is_1) {
                        y[j]      = static_cast<float>(q0) * d + m;
                        y[j + 16] = static_cast<float>(q1) * d + m;
                    } else {
                        y[j]      = static_cast<float>(q0 - 16) * d;
                        y[j + 16] = static_cast<float>(q1 - 16) * d;
                    }
                }
            }
            return true;
        }

        case GGMLType::Q8_0:
            // 34 B/block: fp16 scale, 32 signed int8.
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t* blk = p + b * 34;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const float d = fp16_to_fp32(dh);
                const auto* qs = reinterpret_cast<const int8_t*>(blk + 2);
                float* y = dst + b * 32;
                for (int j = 0; j < 32; ++j) y[j] = static_cast<float>(qs[j]) * d;
            }
            return true;

        case GGMLType::Q4_K:
            // 144 B/super-block of 256 elements:
            //   fp16 d | fp16 dmin | 12 B packed scales/mins | 128 B nibbles
            // The 256 elements are eight 32-element sub-blocks, each with its
            // own 6-bit scale and 6-bit min. Value = d*scale*q - dmin*min, so
            // unlike Q4_0 the quants are unsigned with no bias.
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t* blk = p + b * 144;
                uint16_t dh, mh;
                std::memcpy(&dh, blk, 2);
                std::memcpy(&mh, blk + 2, 2);
                const float d    = fp16_to_fp32(dh);
                const float dmin = fp16_to_fp32(mh);
                const uint8_t* scales = blk + 4;
                const uint8_t* qs     = blk + 16;
                float* y = dst + b * QK_K;
                int is = 0;
                for (uint32_t j = 0; j < QK_K; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, scales, sc, m);
                    const float d1 = d * sc, m1 = dmin * m;
                    get_scale_min_k4(is + 1, scales, sc, m);
                    const float d2 = d * sc, m2 = dmin * m;
                    for (int l = 0; l < 32; ++l)
                        *y++ = d1 * static_cast<float>(qs[l] & 0xF) - m1;
                    for (int l = 0; l < 32; ++l)
                        *y++ = d2 * static_cast<float>(qs[l] >> 4) - m2;
                    qs += 32;
                    is += 2;
                }
            }
            return true;

        case GGMLType::Q6_K:
            // 210 B/super-block of 256:
            //   128 B low nibbles | 64 B high 2-bit pairs | 16 int8 scales | fp16 d
            // Quants are 6-bit unsigned biased by -32, in sixteen 16-element
            // groups each with its own int8 scale.
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t* blk = p + b * 210;
                const uint8_t* ql  = blk;
                const uint8_t* qh  = blk + 128;
                const auto*    sc  = reinterpret_cast<const int8_t*>(blk + 192);
                uint16_t dh;
                std::memcpy(&dh, blk + 208, 2);
                const float d = fp16_to_fp32(dh);
                float* y = dst + b * QK_K;
                for (uint32_t n2 = 0; n2 < QK_K; n2 += 128) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = static_cast<int>((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = static_cast<int>((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = static_cast<int>((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = d * sc[is + 0] * static_cast<float>(q1);
                        y[l + 32] = d * sc[is + 2] * static_cast<float>(q2);
                        y[l + 64] = d * sc[is + 4] * static_cast<float>(q3);
                        y[l + 96] = d * sc[is + 6] * static_cast<float>(q4);
                    }
                    y += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return true;

        default:
            return false;
    }
}

std::vector<float> GGUFFile::dequantize_tensor(const Tensor& t) const {
    const uint64_t n = t.n_elements();
    std::vector<float> out(static_cast<size_t>(n));
    if (!dequantize(t.type, t.data, out.data(), static_cast<size_t>(n)))
        fail("dequantize: unsupported type " + std::string(type_traits(t.type).name) +
             " for tensor '" + std::string(t.name) + "'");
    return out;
}

}  // namespace vulkore::llm
