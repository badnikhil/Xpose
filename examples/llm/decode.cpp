// Host-side decode loop + KV cache manager. See decode.hpp for the kernel
// contract and the reasoning behind the one-submit-per-token design.
#include "decode.hpp"

#include <limits>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>
#include <cerrno>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vulkore::llm {
namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string human_bytes(uint64_t b) {
    char buf[64];
    const double mib = double(b) / (1024.0 * 1024.0);
    if (mib >= 1024.0) std::snprintf(buf, sizeof buf, "%.2f GiB", mib / 1024.0);
    else               std::snprintf(buf, sizeof buf, "%.1f MiB", mib);
    return buf;
}

// Resolve a kernel, and if it is absent say so in terms the kernel author can
// act on: the name we wanted and the module we wanted it from. The kernels
// this file drives were written concurrently with it, so "missing kernel" is
// the expected failure, not an exotic one.
Kernel need_kernel(Program& prog, const std::string& module, const char* name) {
    try {
        return prog.kernel(name);
    } catch (const std::exception& e) {
        std::ostringstream os;
        os << "decode: required kernel '" << name << "' is missing from " << module
           << " (" << e.what() << "). The contract this loop expects is documented "
              "at the top of examples/llm/decode.hpp.";
        throw std::runtime_error(os.str());
    }
}

// ---------------------------------------------------------------------------
// Q4 packing (the matvec_q4 layout)
// ---------------------------------------------------------------------------
// Reads one GGUF row at a time and dequantises it. GGUF stores a linear weight
// as [out][in] row-major with shape[0] == in, so row j IS output column j of
// the [in][out] matrix the kernel wants — the packer is therefore also the
// transpose, and never materialises the full fp32 tensor. That matters for
// exactly one tensor: token_embd is 262144x1152, which is 1.2 GB in fp32.
struct RowReader {
    const Tensor* t    = nullptr;
    uint32_t      rows = 0;  // elements per GGUF row == `in` features

    void check() const {
        const auto& tr = type_traits(t->type);
        if (!tr.dequantisable || tr.block_size == 0)
            throw std::runtime_error(std::string("decode: tensor '") + std::string(t->name) +
                                     "' has type " + tr.name + ", which the GGUF loader "
                                     "cannot dequantise");
        if (rows % tr.block_size != 0)
            throw std::runtime_error(std::string("decode: tensor '") + std::string(t->name) +
                                     "' row length " + std::to_string(rows) +
                                     " is not a multiple of the " + tr.name + " block size " +
                                     std::to_string(tr.block_size) +
                                     " — rows are not block-aligned, so they cannot be read "
                                     "individually");
    }

    void read(uint64_t j, float* dst) const {
        const auto& tr = type_traits(t->type);
        const uint64_t blocks_per_row = rows / tr.block_size;
        const uint8_t* src = t->data + (j * blocks_per_row) * tr.type_size;
        if (!GGUFFile::dequantize(t->type, src, dst, rows))
            throw std::runtime_error("decode: dequantize failed for " + std::string(t->name));
    }
};

constexpr uint32_t kQBlock = 32;
constexpr uint32_t kGroup  = 64;  // GRP in kernels/llm_matvec.cl

// Symmetric int4: q = round(v / s) + 8 with s = max|v| / 7, so the stored
// nibble lands in [1,15] and the kernel's (nibble - 8) recovers [-7,7]. The
// asymmetric -8 slot is left unused; it buys about half a bit and costs a
// branch in every packer and dequantiser that touches it.
inline uint32_t quantise(float v, float inv_scale) {
    return uint32_t(std::clamp(int(std::lround(v * inv_scale)) + 8, 0, 15));
}

// The stored scale is chosen (below) to be exactly representable in bf16, and
// the quantiser divides by that same value, so the GPU multiplies back exactly
// what the host divided by — the bf16 step is absorbed into the nibbles rather
// than added on top of them. Scales are finite and non-negative; no Inf/NaN.
inline uint32_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u >> 16;
}

// Pick the bf16 scale that actually minimises the block's reconstruction
// error, rather than just rounding amax/7 to nearest.
//
// Worth the load-time arithmetic because a scale error behaves quite unlike an
// int4 rounding error. The nibbles' errors are independent, so across a
// 1152-term dot product they average down by sqrt(k); a wrong scale is COMMON
// to all 32 weights in its block and does not average down at all. Measured on
// 20000 synthetic 1152-term dot products: round-to-nearest bf16 is 1.002x the
// fp32-scale error, and choosing between the two neighbouring bf16 values is
// 0.995x — i.e. BETTER than the exact fp32 scale, because amax/7 was never the
// error-minimising choice in the first place. That 0.5% is why this is here
// instead of a one-line round_bf16().
float best_bf16_scale(const float* blk, float s0) {
    if (!(s0 > 0.0f)) return 0.0f;
    uint32_t u;
    std::memcpy(&u, &s0, 4);
    float lo, hi;
    const uint32_t ulo = u & 0xFFFF0000u;
    const uint32_t uhi = ulo + 0x10000u;
    std::memcpy(&lo, &ulo, 4);
    std::memcpy(&hi, &uhi, 4);
    auto err = [&](float s) {
        if (!(s > 0.0f)) return std::numeric_limits<double>::infinity();
        const float inv = 1.0f / s;
        double e = 0.0;
        for (uint32_t t = 0; t < kQBlock; ++t) {
            const double d = double(int(quantise(blk[t], inv)) - 8) * double(s) - double(blk[t]);
            e += d * d;
        }
        return e;
    };
    return err(lo) <= err(hi) ? lo : hi;
}

// [block][col] fp32 scales -> the Km4Bf16 plane: uint2 sc[(b/4)*cols + j],
// blocks 4q..4q+3 in the low/high halves of .x then .y. Emitted as uint32
// because vulkore::Buffer only takes 4-byte element types.
//
// Written for a COLUMN RANGE [j0,j1) rather than the whole tensor, because the
// packer is threaded over column groups and each output word here depends on
// exactly one column. Ranges therefore touch disjoint words and need no
// synchronisation. `out` is the whole-tensor plane.
void pack_scales_bf16_cols(const float* scales, uint32_t nblk, uint32_t cols,
                           uint32_t j0, uint32_t j1, uint32_t* out) {
    for (uint32_t q = 0; q < nblk / 4; ++q)
        for (uint32_t j = j0; j < j1; ++j) {
            const size_t d = (size_t(q) * cols + j) * 2;
            out[d] = 0;
            out[d + 1] = 0;
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t bits = bf16_bits(scales[size_t(q * 4 + i) * cols + j]);
                out[d + (i >> 1)] |= bits << ((i & 1u) * 16);
            }
        }
}

// Packs 64 output columns (== 64 consecutive GGUF rows, since GGUF stores
// [out][in] and the kernels want [in][out] — the packer IS the transpose).
// `cols64[c*rows + k]` is weight k of column group*64 + c.
//
// 64 at a time because that is KM4's group width; the COL layout only needs 8
// and 64 is a multiple of it, so one traversal serves both.
void pack_group(const float* cols64, uint32_t rows, uint32_t cols, uint32_t group,
                MatvecLayout layout, uint32_t* words, float* scales) {
    const uint32_t nblk = rows / kQBlock;
    const uint32_t j0   = group * kGroup;

    float inv[kGroup];
    for (uint32_t b = 0; b < nblk; ++b) {
        for (uint32_t c = 0; c < kGroup; ++c) {
            const float* r = cols64 + size_t(c) * rows + b * kQBlock;
            float amax = 0.0f;
            for (uint32_t t = 0; t < kQBlock; ++t) amax = std::max(amax, std::fabs(r[t]));
            // Rounded BEFORE the nibbles are derived from it, so the GPU
            // multiplies back exactly the value the quantiser divided by and
            // the bf16 step is absorbed rather than added. Measured: identical
            // reconstruction RMS to the fp32-scale path, 1.000x.
            const float s = layout == MatvecLayout::Km4Bf16
                                ? best_bf16_scale(r, amax / 7.0f) : amax / 7.0f;
            scales[size_t(b) * cols + j0 + c] = s;
            inv[c] = s > 0.0f ? 1.0f / s : 0.0f;
        }

        if (layout == MatvecLayout::ColumnNibble) {
            // word[(k*cols + j) >> 3], nibble (j&7)*4.
            const uint32_t words_per_row = cols / 8;
            for (uint32_t t = 0; t < kQBlock; ++t) {
                const uint32_t k = b * kQBlock + t;
                for (uint32_t sub = 0; sub < kGroup / 8; ++sub) {
                    uint32_t word = 0;
                    for (uint32_t c = 0; c < 8; ++c) {
                        const uint32_t cc = sub * 8 + c;
                        word |= quantise(cols64[size_t(cc) * rows + k], inv[cc]) << (c * 4);
                    }
                    words[size_t(k) * words_per_row + group * (kGroup / 8) + sub] = word;
                }
            }
        } else {
            // uint4[(g*nblk + b)*64 + t]; component (k>>3)&3, nibble (k&7)*4.
            // One uint4 is exactly one 32-weight quantisation block, which is
            // why the kernel's scale lookup happens once per 16-byte load.
            const size_t base = (size_t(group) * nblk + b) * kGroup;
            for (uint32_t c = 0; c < kGroup; ++c) {
                uint32_t w[4] = {0, 0, 0, 0};
                for (uint32_t t = 0; t < kQBlock; ++t) {
                    const uint32_t k = b * kQBlock + t;
                    w[(t >> 3) & 3u] |= quantise(cols64[size_t(c) * rows + k], inv[c])
                                        << ((t & 7u) * 4);
                }
                uint32_t* dst = words + (base + c) * 4;
                dst[0] = w[0]; dst[1] = w[1]; dst[2] = w[2]; dst[3] = w[3];
            }
        }
    }
}

}  // namespace

MatvecConfig MatvecConfig::baseline() {
    MatvecConfig m;
    m.layout        = MatvecLayout::ColumnNibble;
    m.module        = "kernels/llm.spv";
    m.kernel        = "matvec_q4";
    m.split_kernel  = "matvec_q4_split";
    m.reduce_kernel = "matvec_reduce";
    return m;
}

uint32_t MatvecConfig::split_for(uint32_t rows, uint32_t cols) const {
    if (split <= 1 || cols >= split_below_cols) return 1;
    uint32_t s = std::min(split, rows / kQBlock);  // never more splits than blocks
    // The workgroup-count limit is real: 262144 columns x 16 splits / 64 is
    // 65536 workgroups against a device limit of 65535 (matvec-optimisation.md).
    while (s > 1 && (uint64_t(cols) * s + 63) / 64 > max_workgroups) --s;
    return std::max(s, 1u);
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

DecodeConfig DecodeConfig::gemma3_1b() { return DecodeConfig{}; }

DecodeConfig DecodeConfig::from_model(const ModelConfig& mc) {
    DecodeConfig c;
    if (mc.n_layers)            c.n_layers   = mc.n_layers;
    if (mc.n_heads)             c.n_heads    = mc.n_heads;
    if (mc.n_kv_heads)          c.n_kv_heads = mc.n_kv_heads;
    if (mc.embedding_length)    c.hidden     = mc.embedding_length;
    if (mc.feed_forward_length) c.ffn        = mc.feed_forward_length;
    if (mc.head_dim())          c.head_dim   = mc.head_dim();
    if (mc.vocab_size)          c.vocab      = mc.vocab_size;
    c.rms_eps           = mc.rms_eps;
    c.rope_theta_global = mc.rope_freq_base;
    // Gemma 3 publishes the window extent (gemma3.attention.sliding_window =
    // 512) but NOT the 5-local-to-1-global alternation, which stays a
    // DecodeConfig policy field. A file without the key leaves the default in
    // place rather than silently disabling the window.
    if (mc.sliding_window) c.sliding_window = mc.sliding_window;
    return c;
}

float DecodeConfig::effective_embed_scale() const {
    return embed_scale > 0.0f ? embed_scale : std::sqrt(float(hidden));
}
float DecodeConfig::effective_attn_scale() const {
    return attn_scale > 0.0f ? attn_scale : 1.0f / std::sqrt(float(head_dim));
}

// ---------------------------------------------------------------------------
// Dispatch plan / cost model — no GPU, no kernels, no model file
// ---------------------------------------------------------------------------

DispatchPlan plan_token(const DecodeConfig& cfg, const MatvecConfig& mv) {
    DispatchPlan p;
    // Gemma 3 wraps BOTH sublayers: input_layernorm + post_attention_norm and
    // ffn_norm + post_ffw_norm. Four RMSNorms per layer, two dispatches each.
    p.norms      = 4 * 2;
    p.qk_norms   = 2;
    p.rope       = 2;
    p.kv_appends = 2;
    // 3 at shallow spans (scores + softmax_rows + apply). Past
    // DecodeConfig::attn_min_span every layer switches to the parallel forms
    // and this becomes 6: scores_mq4 + (softmax partial/finish/scale) +
    // (apply_mq4 + reduce). So a fully deep token issues 78 MORE dispatches
    // than this plan reports — ~0.44 ms at the measured 5.6 us, bought to
    // remove ~17 ms of serial attention. See llm-attention-flatness.md.
    p.attention  = cfg.parallel_softmax ? 6 : 3;
    p.activation = 1;
    p.residuals  = 2;

    const uint32_t q_cols  = cfg.n_heads * cfg.head_dim;
    const uint32_t kv_cols = cfg.n_kv_heads * cfg.head_dim;
    struct MV { uint64_t rows, cols; uint32_t n; };
    const MV mvs[] = {
        {cfg.hidden, q_cols,     cfg.n_layers},
        {cfg.hidden, kv_cols,    cfg.n_layers},  // k
        {cfg.hidden, kv_cols,    cfg.n_layers},  // v
        {q_cols,     cfg.hidden, cfg.n_layers},  // o_proj
        {cfg.hidden, cfg.ffn,    cfg.n_layers},  // gate
        {cfg.hidden, cfg.ffn,    cfg.n_layers},  // up
        {cfg.ffn,    cfg.hidden, cfg.n_layers},  // down
        {cfg.hidden, cfg.vocab,  1},             // lm_head
    };
    for (const MV& m : mvs) {
        const uint64_t nib = m.rows * m.cols / 2 * m.n;
        const uint64_t scale_sz = mv.layout == MatvecLayout::Km4Bf16 ? 2 : 4;
        const uint64_t sca = (m.rows / kQBlock) * m.cols * scale_sz * m.n;
        p.weight_bytes_nibbles += nib;
        p.weight_bytes_scales  += sca;
        if (m.n == 1 && m.cols == cfg.vocab) p.lm_head_bytes = nib + sca;
        // A split matvec is two dispatches: the split pass and the reduce.
        const uint32_t d = mv.split_for(uint32_t(m.rows), uint32_t(m.cols)) > 1 ? 2 : 1;
        // lm_head + the sampling epilogue: one topk4_partial for the candidate
        // harvest, three for the argmax tree, one for the serial fallback.
        if (m.n == 1)
            p.epilogue_head =
                d + (cfg.sampling.mode == SampleMode::Temperature && !cfg.sampling.cpu_full_vocab
                         ? 1u
                         : (cfg.parallel_argmax ? 3u : 1u));
        else          p.matvecs += d;
    }
    p.weight_bytes = p.weight_bytes_nibbles + p.weight_bytes_scales;

    p.per_layer = p.norms + p.qk_norms + p.matvecs + p.rope + p.kv_appends +
                  p.attention + p.activation + p.residuals;
    p.prologue      = 0;  // the embedding is a CPU memcpy into a mapped buffer
    p.epilogue_norm = 2;
    p.total         = p.per_layer * cfg.n_layers + p.prologue + p.epilogue_norm +
                      p.epilogue_head;
    p.total_prefill = p.total - p.epilogue_head;
    p.submits       = 1;
    return p;
}

Machine Machine::for_matvec(const MatvecConfig& mv) {
    Machine m;
    if (mv.layout == MatvecLayout::ColumnNibble) {
        m.name          = "Adreno 840, baseline COL layout (measured, batched)";
        m.bandwidth_gbs = 12.5;
    } else if (mv.split > 1) {
        m.name          = "Adreno 840, KM4+split4 (measured, batched)";
        m.bandwidth_gbs = 30.9;
    } else {
        m.name          = "Adreno 840, KM4 no split (measured, batched)";
        m.bandwidth_gbs = 28.9;  // 23.1 published x 1.25
    }
    return m;
}

Projection project(const DispatchPlan& plan, const Machine& m) {
    Projection r;
    r.weight_ms   = double(plan.weight_bytes) / (m.bandwidth_gbs * 1e9) * 1e3;
    r.overhead_ms = plan.total * m.dispatch_ms;
    r.submit_ms   = plan.submits * m.submit_ms;
    r.total_ms    = r.weight_ms + r.overhead_ms + r.submit_ms;
    r.tokens_per_sec = r.total_ms > 0 ? 1000.0 / r.total_ms : 0.0;
    return r;
}

std::string dry_run_report(const DecodeConfig& cfg, const Machine& m,
                           const MatvecConfig& mv) {
    const DispatchPlan p = plan_token(cfg, mv);
    const Projection   r = project(p, m);
    const Projection   naive = project(
        [&] { DispatchPlan q = p; q.submits = q.total; return q; }(), m);

    std::ostringstream o;
    o << "=== decode dry run (no GPU, no kernels, no model file) ===\n\n"
      << "model     " << cfg.n_layers << " layers, hidden " << cfg.hidden
      << ", ffn " << cfg.ffn << ", " << cfg.n_heads << " q heads / "
      << cfg.n_kv_heads << " kv head" << (cfg.n_kv_heads == 1 ? "" : "s")
      << " x " << cfg.head_dim << ", vocab " << cfg.vocab << "\n\n";

    o << "--- dispatches per token -------------------------------------\n";
    auto row = [&](const char* what, uint32_t n, const char* note) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "  %-22s %5u   %s\n", what, n, note);
        o << buf;
    };
    row("RMSNorm (4 x 2-pass)", p.norms, "sumsq + apply_gemma");
    row("QK-norm", p.qk_norms, "rmsnorm_heads_gemma on q and k");
    row("matvec", p.matvecs, "q k v o gate up down (+ split-k reduces)");
    row("RoPE", p.rope, "in-place, pos/theta as push constants");
    row("kv_append", p.kv_appends, "k and v");
    row("attention", p.attention,
        p.attention == 6 ? "scores_mq4 + softmax x3 + apply_mq4 + reduce (span >= "
                           "attn_min_span; 3 below it)"
                         : "scores + softmax_rows + apply");
    row("GeGLU", p.activation, "");
    row("residual", p.residuals, "");
    {
        char buf[160];
        std::snprintf(buf, sizeof buf, "  %-22s %5u   per layer, x%u layers = %u\n",
                      "TOTAL/layer", p.per_layer, cfg.n_layers, p.per_layer * cfg.n_layers);
        o << buf;
    }
    row("embedding", p.prologue, "CPU memcpy into a HostVisible buffer");
    row("final norm", p.epilogue_norm, "");
    row("lm_head + argmax", p.epilogue_head, "skipped during prefill");
    {
        char buf[200];
        std::snprintf(buf, sizeof buf, "  %-22s %5u   in %u vkQueueSubmit\n",
                      "TOKEN TOTAL", p.total, p.submits);
        o << buf;
        std::snprintf(buf, sizeof buf, "  %-22s %5u   (no lm_head: %s less traffic)\n",
                      "prefill token", p.total_prefill, human_bytes(p.lm_head_bytes).c_str());
        o << buf;
    }

    o << "\n--- matvec plan ----------------------------------------------\n"
      << "  layout " << (mv.layout == MatvecLayout::Km4Bf16 ? "KM4+bf16 scales (mv_km4_bs)"
                          : mv.layout == MatvecLayout::Km4     ? "KM4 (mv_km4)"
                                                                : "COL (matvec_q4)")
      << " from " << mv.module << "\n";
    {
        const uint32_t q_cols  = cfg.n_heads * cfg.head_dim;
        const uint32_t kv_cols = cfg.n_kv_heads * cfg.head_dim;
        struct S { const char* n; uint32_t rows, cols, count; };
        const S shapes[] = {
            {"q",       cfg.hidden, q_cols,     cfg.n_layers},
            {"k",       cfg.hidden, kv_cols,    cfg.n_layers},
            {"v",       cfg.hidden, kv_cols,    cfg.n_layers},
            {"o_proj",  q_cols,     cfg.hidden, cfg.n_layers},
            {"gate",    cfg.hidden, cfg.ffn,    cfg.n_layers},
            {"up",      cfg.hidden, cfg.ffn,    cfg.n_layers},
            {"down",    cfg.ffn,    cfg.hidden, cfg.n_layers},
            {"lm_head", cfg.hidden, cfg.vocab,  1},
        };
        for (const S& s : shapes) {
            const uint32_t sp = mv.split_for(s.rows, s.cols);
            const std::string how = sp > 1
                ? "split-k " + std::to_string(sp) + " + reduce"
                : std::string("single dispatch");
            char buf[160];
            std::snprintf(buf, sizeof buf, "  %-8s %6u x %-7u x%-3u  %s\n", s.n, s.rows,
                          s.cols, s.count, how.c_str());
            o << buf;
        }
    }

    o << "\n--- weight traffic per generated token -----------------------\n"
      << "  int4 nibbles           " << human_bytes(p.weight_bytes_nibbles) << "\n"
      << (mv.layout == MatvecLayout::Km4Bf16 ? "  bf16 block scales      "
                                              : "  fp32 block scales      ")
      << human_bytes(p.weight_bytes_scales)
      << (mv.layout == MatvecLayout::Km4Bf16 ? "   (+12.5%: one bf16 per "
                                             : "   (+25%: one float per ")
      << kQBlock << " weights)\n"
      << "  total                  " << human_bytes(p.weight_bytes) << "\n"
      << "  of which lm_head       " << human_bytes(p.lm_head_bytes) << "\n";

    o << "\n--- KV cache -------------------------------------------------\n"
      << "  " << cfg.max_seq << " positions x " << cfg.n_layers << " layers x "
      << cfg.n_kv_heads << " kv head x " << cfg.head_dim << " x fp32 x2 (K,V)\n"
      << "  per position           " << (2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 4)
      << " B\n"
      << "  total                  " << human_bytes(cfg.kv_cache_bytes())
      << "   (" << (2 * cfg.n_layers) << " buffers of "
      << human_bytes(cfg.kv_cache_bytes() / (2 * cfg.n_layers)) << ")\n";

    // What the KV cache costs in TIME, not just space. This is the table that
    // decides max_seq: the cache is allocated once but READ every token, and
    // the sliding window changes how that read scales with depth.
    //
    //   global layers  read the whole cache      -> grows with position
    //   local  layers  read at most `window`     -> flat past the window
    //
    // Without the window every layer is in the first class, which is why the
    // unmasked loop was not merely wrong past 512 but also, at depth, slower
    // than the model it was imitating.
    {
        const uint64_t per_layer_pos = 2ull * cfg.n_kv_heads * cfg.head_dim * 4;
        const uint32_t nglob = cfg.global_every ? cfg.n_layers / cfg.global_every : cfg.n_layers;
        const uint32_t nloc  = cfg.n_layers - nglob;
        o << "\n--- attention KV traffic per token, by position ---------------\n"
          << "  " << nglob << " global layers read all of it; " << nloc
          << " local layers read <= " << cfg.sliding_window << "\n"
          << "  pos        windowed        unwindowed      of weight traffic\n";
        for (uint32_t P : {512u, 2048u, 8192u, 32768u}) {
            if (P > cfg.max_seq * 4) break;
            const uint64_t win = per_layer_pos *
                (uint64_t(nglob) * P +
                 uint64_t(nloc) * (cfg.sliding_window ? std::min(P, cfg.sliding_window) : P));
            const uint64_t nowin = per_layer_pos * uint64_t(cfg.n_layers) * P;
            char b[200];
            std::snprintf(b, sizeof b, "  %-10u %-15s %-15s %+.0f%%\n", P,
                          human_bytes(win).c_str(), human_bytes(nowin).c_str(),
                          100.0 * double(win) / double(p.weight_bytes));
            o << b;
        }
    }

    o << "\n--- projection (" << m.name << ") ---\n"
      << "  bandwidth " << m.bandwidth_gbs << " GB/s, dispatch " << m.dispatch_ms
      << " ms, submit " << m.submit_ms << " ms\n";
    char buf[200];
    std::snprintf(buf, sizeof buf, "  %-22s %8.2f ms\n", "weight streaming", r.weight_ms);   o << buf;
    std::snprintf(buf, sizeof buf, "  %-22s %8.2f ms  (%u dispatches)\n", "dispatch overhead",
                  r.overhead_ms, p.total); o << buf;
    std::snprintf(buf, sizeof buf, "  %-22s %8.2f ms  (%u submit)\n", "submission",
                  r.submit_ms, p.submits); o << buf;
    std::snprintf(buf, sizeof buf, "  %-22s %8.2f ms  -> %.1f tok/s\n", "TOTAL",
                  r.total_ms, r.tokens_per_sec); o << buf;
    std::snprintf(buf, sizeof buf,
                  "\n  for contrast, one submit per dispatch: %.1f ms -> %.1f tok/s"
                  " (%.1fx worse)\n",
                  naive.total_ms, naive.tokens_per_sec, naive.total_ms / r.total_ms);
    o << buf;
    o << "\n  PROJECTED for the Adreno 840, not measured on it: the bandwidth figure\n"
         "  comes from the matvec benchmark and the per-dispatch figure from the\n"
         "  transformer kernel harness. The loop HAS run end-to-end, but only on an\n"
         "  Adreno X1-85 laptop under turnip (23.8 ms/token) — never on a phone.\n";
    return o.str();
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

namespace {

struct QWeight {
    Buffer   q;      // packed nibbles
    Buffer   scale;  // fp32, one per (block, column)
    uint32_t rows = 0, cols = 0;
    uint64_t bytes = 0;
};

struct LayerWeights {
    Buffer  attn_norm, post_attn_norm, ffn_norm, post_ffn_norm, q_norm, k_norm;
    QWeight wq, wk, wv, wo, w_gate, w_up, w_down;
    Buffer  kcache, vcache;
};

}  // namespace

struct DecodeSession::Impl {
    // Declaration order IS destruction order reversed: buffers must die before
    // the Programs that describe them and the Context that owns them.
    Context ctx;
    // Descriptor sets for the per-token dispatch chain. That chain binds the
    // SAME buffers every token, so after the first token every lookup hits and
    // the vkAllocateDescriptorSets + vkUpdateDescriptorSets pair — the largest
    // single term in host record cost, 838 of them per token — disappears.
    // Declared right after ctx so it is destroyed just before ctx and long
    // after every Buffer below, which is DescriptorCache's lifetime contract.
    DescriptorCache dcache{ctx};
    Program matvec_prog;
    Program tf_prog;
    std::unique_ptr<Program> sampling_prog;  // optional

    Kernel k_matvec, k_matvec_split, k_reduce;
    Kernel k_sumsq, k_apply, k_heads_norm, k_rope, k_geglu;
    Kernel k_scores, k_softmax, k_attn_apply, k_residual, k_kv_append, k_argmax;
    Kernel k_am_partial, k_am_reduce, k_am_final;  // parallel argmax tree
    // Parallel attention: 3-pass softmax + multi-query scores. Both optional —
    // an older llm_transformer.spv simply keeps the serial path.
    Kernel k_sm_part, k_sm_fin, k_sm_scale, k_scores_mq4;
    Kernel k_apply_mq4, k_apply_red;
    bool   has_par_softmax = false, has_mq4 = false;   // present in the module
    bool   par_softmax = false, scores_mq4 = false;    // actually enabled
    Kernel k_topk4;                                // candidate harvest
    bool   fast_argmax = false;
    bool   harvest     = false;  // topk4_partial resolved AND sampling wants it
    MatvecConfig mv;

    GGUFFile  gguf;
    Tokenizer tok;
    RowReader embed_rows;  // token_embd, for the CPU-side embedding gather

    std::vector<LayerWeights> layers;
    Buffer  final_norm;
    QWeight lm_head;

    // Activations
    Buffer x;         // residual stream, HostVisible (CPU writes the embedding)
    Buffer xb, xb2;
    Buffer q_raw, q, k_raw, k, v;
    Buffer att, scores;
    Buffer hb_gate, hb_up, hb;
    Buffer logits, sample_out /* HostVisible, 2 floats */, norm_partial;
    Buffer mv_partial;  // split-k accumulators, [split][cols]
    Buffer am_val, am_idx, am_val2, am_idx2, am_out_idx;  // argmax tree scratch
    Buffer sm_pmax, sm_psum, sm_stats;                    // parallel-softmax scratch
    Buffer att_partial;                                   // apply_mq4 split scratch
    Buffer cand_val, cand_idx;  // HostVisible, 4*topk_parts candidate pairs
    std::vector<float>    cand_val_host;  // reused, so sampling allocates nothing
    std::vector<uint32_t> cand_idx_host;
    std::vector<std::pair<float, uint32_t>> cand_scratch;

    uint64_t weight_bytes = 0;
    double   kv_alloc_ms  = 0;   // wall time spent allocating the KV cache
    LoadStats load_stats;        // where the weight-load seconds went
    std::vector<float> embed_scratch;  // one embedding row
    std::unique_ptr<std::mt19937_64> rng;

    Impl(const Paths& paths)
        : ctx(),
          matvec_prog(Program::from_file(ctx, paths.matvec.module)),
          tf_prog(Program::from_file(ctx, paths.transformer_spv)) {}
};

namespace {

// The host-side result of packing one tensor: exactly the bytes that go into
// the two GPU buffers, and exactly the bytes written to the disk cache. Filling
// one of these is the only work a packer thread does — it never touches
// vulkore::Context, which is single-threaded for allocation, upload AND teardown.
//
// Exactly one of scale_u / scale_f is populated, chosen by the layout. Two
// vectors rather than one reinterpreted blob because a Buffer of floats is
// allocated as floats, and casting a uint32 buffer to float* to feed
// Buffer::upload would be an aliasing violation for no gain.
struct PackedHost {
    std::vector<uint32_t> words;
    std::vector<uint32_t> scale_u;  // Km4Bf16: four bf16 scales per uint2
    std::vector<float>    scale_f;  // Km4 / ColumnNibble: [block][col] fp32

    size_t words_bytes() const { return words.size() * sizeof(uint32_t); }
    size_t scale_bytes() const {
        return scale_u.size() * sizeof(uint32_t) + scale_f.size() * sizeof(float);
    }
    size_t total_bytes() const { return words_bytes() + scale_bytes(); }
    const void* scale_data() const {
        return scale_u.empty() ? static_cast<const void*>(scale_f.data())
                               : static_cast<const void*>(scale_u.data());
    }
    void release() {
        std::vector<uint32_t>().swap(words);
        std::vector<uint32_t>().swap(scale_u);
        std::vector<float>().swap(scale_f);
    }
};

// Uploads a packed weight to the GPU. MUST run on the thread that owns the
// Context.
QWeight upload_packed(Context& ctx, const PackedHost& p, uint32_t rows, uint32_t cols) {
    QWeight w;
    w.rows = rows;
    w.cols = cols;
    w.q    = ctx.alloc<uint32_t>(p.words.size());
    w.q.upload(std::span<const uint32_t>(p.words));
    if (!p.scale_u.empty()) {
        w.scale = ctx.alloc<uint32_t>(p.scale_u.size());
        w.scale.upload(std::span<const uint32_t>(p.scale_u));
    } else {
        w.scale = ctx.alloc<float>(p.scale_f.size());
        w.scale.upload(std::span<const float>(p.scale_f));
    }
    w.bytes = p.total_bytes();
    return w;
}

// One unit of packer work: a range of 64-column groups of one tensor. Column
// groups are the natural grain because every output word of both layouts is a
// function of exactly one column group, so ranges write disjoint memory.
struct PackTask {
    std::string   name;
    const Tensor* src  = nullptr;
    uint32_t      rows = 0, cols = 0;
};

// Packs groups [g0,g1) of `t` into the (whole-tensor) `out`, which the caller
// has already sized. `tmp_scales` is the whole-tensor fp32 scale plane; for the
// bf16 layout this thread converts only its own column range afterwards.
void pack_chunk(const PackTask& t, uint32_t g0, uint32_t g1, MatvecLayout layout,
                float* tmp_scales, PackedHost& out) {
    RowReader rr{t.src, t.rows};
    std::vector<float> scratch(size_t(t.rows) * kGroup);
    for (uint32_t g = g0; g < g1; ++g) {
        for (uint32_t c = 0; c < kGroup; ++c)
            rr.read(uint64_t(g) * kGroup + c, scratch.data() + size_t(c) * t.rows);
        pack_group(scratch.data(), t.rows, t.cols, g, layout,
                   out.words.data(), tmp_scales);
    }
    if (layout == MatvecLayout::Km4Bf16)
        pack_scales_bf16_cols(tmp_scales, t.rows / kQBlock, t.cols, g0 * kGroup,
                              g1 * kGroup, out.scale_u.data());
}

// Whole-tensor convenience form, for `--selftest` (which packs synthetic
// shapes without going near the load path).
std::vector<uint32_t> pack_scales_bf16(const std::vector<float>& scales, uint32_t nblk,
                                       uint32_t cols) {
    std::vector<uint32_t> out(size_t(nblk) / 4 * cols * 2);
    pack_scales_bf16_cols(scales.data(), nblk, cols, 0, cols, out.data());
    return out;
}

const Tensor& need_tensor(const GGUFFile& f, const std::string& name) {
    if (const Tensor* t = f.tensor(name)) return *t;
    throw std::runtime_error("decode: GGUF is missing tensor '" + name + "'");
}

// ---------------------------------------------------------------------------
// Packed-weight disk cache ("<model>.xpack")
// ---------------------------------------------------------------------------
// Repacking 1B parameters costs seconds of single-core CPU every launch, and
// the result is a pure function of (file bytes, layout, packer arithmetic). So
// it is written once and read back.
//
// The failure mode this format is designed against is NOT a corrupt read — it
// is a cache that loads SUCCESSFULLY with weights that belong to something
// else. That produces fluent, plausible, wrong text, which this project has
// already shipped once (the double +1 on norm weights). Every field below
// therefore exists to make a mismatch a hard failure, and ANY doubt at all
// makes the loader repack silently rather than guess:
//
//   * magic/version/sizes    — wrong or older writer
//   * src_size/src_mtime_ns  — the model file changed
//   * src_fingerprint        — the model file was REPLACED with one of the same
//                              size and timestamp (mtime alone is not enough on
//                              a device where files arrive by adb push)
//   * layout/qblock/group/scale_bf16/packer_version — the packing arithmetic or
//                              the KM4 bit layout changed
//   * the entry table        — the tensor names, order, shapes and byte counts
//                              must equal what THIS session is about to ask for
//   * per-entry hash         — the payload is what the writer wrote
//
// The file is built at "<path>.tmp" and rename(2)'d into place, so a partial
// write can never be observed as a valid cache.
constexpr char     kXPackMagic[8]  = {'X', 'P', 'A', 'C', 'K', '0', '0', '1'};
constexpr uint32_t kXPackVersion   = 1;
// Bump when anything that changes the packed BYTES changes: quantise(),
// best_bf16_scale(), pack_group(), pack_scales_bf16_cols(), kQBlock, kGroup.
constexpr uint32_t kPackerVersion  = 1;
constexpr uint32_t kXPackMaxName   = 80;
constexpr uint32_t kXPackMaxEntries = 8192;

struct XPackHeader {
    char     magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t src_size;
    uint64_t src_mtime_ns;
    uint64_t src_fingerprint;
    uint32_t layout;
    uint32_t qblock;
    uint32_t group;
    uint32_t scale_bf16;
    uint32_t packer_version;
    uint32_t n_entries;
    uint64_t table_offset;
    uint64_t total_size;
    uint64_t reserved[6];
};
static_assert(sizeof(XPackHeader) == 128, "XPackHeader must stay 128 bytes");

struct XPackEntry {
    char     name[kXPackMaxName];
    uint32_t rows, cols;
    uint64_t offset;       // start of words; scales follow immediately
    uint64_t words_bytes;
    uint64_t scale_bytes;
    uint64_t hash;         // over words||scales
};
static_assert(sizeof(XPackEntry) == 120, "XPackEntry must stay 120 bytes");

// FNV-1a over 64-bit words with a final avalanche, plus the length so that
// appending zeros cannot go unnoticed. Chosen over a byte-at-a-time hash
// because this runs over 536 MiB on every cache HIT and has to stay cheap
// relative to the read it is validating.
// The payload is hashed in FIXED-SIZE chunks so that both the writer and the
// reader can spread the work over cores. The chunk size is part of the format:
// change it and every existing cache stops validating, which is why it lives
// next to the version constants rather than being a tuning knob.
constexpr size_t kXPackHashChunk = 4u << 20;

uint64_t xpack_hash(const void* data, size_t nbytes) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull ^ uint64_t(nbytes);
    size_t i = 0;
    for (; i + 8 <= nbytes; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 1099511628211ull;
        h ^= h >> 29;
    }
    for (; i < nbytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    h ^= h >> 32;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 32;
    return h;
}

// Chunked, threaded form. Deterministic and independent of the thread count:
// the per-chunk hashes are folded back in chunk order.
uint64_t xpack_hash_mt(const void* data, size_t nbytes, uint32_t nthreads) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const size_t nchunks = (nbytes + kXPackHashChunk - 1) / kXPackHashChunk;
    if (nchunks <= 1 || nthreads <= 1) {
        uint64_t acc = 1469598103934665603ull ^ uint64_t(nbytes);
        for (size_t c = 0; c < nchunks; ++c) {
            const size_t off = c * kXPackHashChunk;
            acc = (acc ^ xpack_hash(p + off, std::min(kXPackHashChunk, nbytes - off))) *
                  1099511628211ull;
        }
        return acc;
    }
    std::vector<uint64_t> part(nchunks);
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    const uint32_t nt = uint32_t(std::min<size_t>(nthreads, nchunks));
    for (uint32_t w = 0; w < nt; ++w)
        pool.emplace_back([&] {
            for (;;) {
                const size_t c = next.fetch_add(1, std::memory_order_relaxed);
                if (c >= nchunks) return;
                const size_t off = c * kXPackHashChunk;
                part[c] = xpack_hash(p + off, std::min(kXPackHashChunk, nbytes - off));
            }
        });
    for (auto& t : pool) t.join();
    uint64_t acc = 1469598103934665603ull ^ uint64_t(nbytes);
    for (uint64_t v : part) acc = (acc ^ v) * 1099511628211ull;
    return acc;
}

// Cheap identity check on the SOURCE model that does not read 806 MiB: the
// first and last 64 KiB plus the size. Combined with size+mtime this is what
// stops a different model file inheriting a stale cache.
bool source_stamp(const std::string& path, uint64_t& size, uint64_t& mtime_ns,
                  uint64_t& fingerprint) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return false;
    size     = uint64_t(st.st_size);
    mtime_ns = uint64_t(st.st_mtim.tv_sec) * 1000000000ull + uint64_t(st.st_mtim.tv_nsec);
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    constexpr size_t kEnd = 64 << 10;
    std::vector<uint8_t> buf(std::min<size_t>(kEnd, size_t(size)));
    uint64_t h = size;
    for (int which = 0; which < 2; ++which) {
        const off_t off = which == 0 ? 0 : off_t(size > buf.size() ? size - buf.size() : 0);
        size_t got = 0;
        while (got < buf.size()) {
            const ssize_t n = ::pread(fd, buf.data() + got, buf.size() - got, off + off_t(got));
            if (n <= 0) break;
            got += size_t(n);
        }
        if (got != buf.size()) { ::close(fd); return false; }
        h = xpack_hash(buf.data(), buf.size()) ^ (h * 1099511628211ull);
    }
    ::close(fd);
    fingerprint = h;
    return true;
}

bool read_exact(int fd, void* dst, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t got = 0;
    while (got < n) {
        const ssize_t r = ::read(fd, p + got, n - got);
        if (r <= 0) return false;
        got += size_t(r);
    }
    return true;
}

bool write_exact(int fd, const void* src, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    size_t put = 0;
    while (put < n) {
        const ssize_t r = ::write(fd, p + put, n - put);
        if (r <= 0) return false;
        put += size_t(r);
    }
    return true;
}

std::string default_cache_path(const Paths& paths) {
    if (!paths.pack_cache.empty()) return paths.pack_cache;
    if (paths.model_gguf.empty()) return {};
    return paths.model_gguf + ".xpack";
}

// Fills `hdr` with what a cache for this session MUST say. Returns false if the
// model file cannot be stamped, in which case caching is disabled entirely
// (better no cache than one keyed on nothing).
bool expected_header(const Paths& paths, MatvecLayout layout, uint32_t n_entries,
                     XPackHeader& hdr) {
    hdr = XPackHeader{};
    std::memcpy(hdr.magic, kXPackMagic, 8);
    hdr.version        = kXPackVersion;
    hdr.header_size    = sizeof(XPackHeader);
    hdr.layout         = uint32_t(layout);
    hdr.qblock         = kQBlock;
    hdr.group          = kGroup;
    hdr.scale_bf16     = layout == MatvecLayout::Km4Bf16 ? 1u : 0u;
    hdr.packer_version = kPackerVersion;
    hdr.n_entries      = n_entries;
    return source_stamp(paths.model_gguf, hdr.src_size, hdr.src_mtime_ns, hdr.src_fingerprint);
}

// Opens a cache file and validates everything that can be validated without
// reading the payload. Returns -1 (and says why, once, at a low volume) if the
// cache is absent or does not match; the caller then repacks.
int open_valid_cache(const std::string& path, const XPackHeader& want,
                     const std::vector<PackTask>& tasks, std::vector<XPackEntry>& table) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return -1;
    auto reject = [&](const char* why) {
        std::fprintf(stderr, "decode: ignoring pack cache (%s), repacking\n", why);
        ::close(fd);
        return -1;
    };
    struct stat st {};
    if (::fstat(fd, &st) != 0) return reject("stat failed");

    XPackHeader h{};
    if (!read_exact(fd, &h, sizeof h))                 return reject("truncated header");
    if (std::memcmp(h.magic, kXPackMagic, 8) != 0)     return reject("bad magic");
    if (h.version != kXPackVersion)                    return reject("version mismatch");
    if (h.header_size != sizeof(XPackHeader))          return reject("header size mismatch");
    if (h.src_size != want.src_size)                   return reject("model size changed");
    if (h.src_mtime_ns != want.src_mtime_ns)           return reject("model mtime changed");
    if (h.src_fingerprint != want.src_fingerprint)     return reject("model contents changed");
    if (h.layout != want.layout)                       return reject("matvec layout differs");
    if (h.qblock != want.qblock || h.group != want.group)
        return reject("quantisation block/group differs");
    if (h.scale_bf16 != want.scale_bf16)               return reject("scale format differs");
    if (h.packer_version != want.packer_version)       return reject("packer version differs");
    if (h.n_entries != want.n_entries)                 return reject("tensor count differs");
    if (h.n_entries == 0 || h.n_entries > kXPackMaxEntries)
        return reject("implausible tensor count");
    if (h.total_size != uint64_t(st.st_size))          return reject("file size mismatch");
    if (h.table_offset < sizeof(XPackHeader) ||
        h.table_offset + uint64_t(h.n_entries) * sizeof(XPackEntry) != h.total_size)
        return reject("table does not close the file");

    table.resize(h.n_entries);
    if (::lseek(fd, off_t(h.table_offset), SEEK_SET) < 0) return reject("seek failed");
    if (!read_exact(fd, table.data(), table.size() * sizeof(XPackEntry)))
        return reject("truncated table");

    // The table must describe EXACTLY the tensors this session is about to
    // ask for, in the same order, with the same shapes and the byte counts
    // those shapes imply. Anything else and we do not know what we are loading.
    uint64_t off = sizeof(XPackHeader);
    for (size_t i = 0; i < tasks.size(); ++i) {
        const XPackEntry& e = table[i];
        if (tasks[i].name.size() >= kXPackMaxName)     return reject("tensor name too long");
        if (std::memcmp(e.name, tasks[i].name.c_str(), tasks[i].name.size() + 1) != 0)
            return reject("tensor name/order mismatch");
        if (e.rows != tasks[i].rows || e.cols != tasks[i].cols)
            return reject("tensor shape mismatch");
        const uint64_t wb = uint64_t(e.rows) * e.cols / 8 * sizeof(uint32_t);
        const uint64_t sb = want.scale_bf16
                                ? uint64_t(e.rows) / kQBlock / 4 * e.cols * 2 * sizeof(uint32_t)
                                : uint64_t(e.rows) / kQBlock * e.cols * sizeof(float);
        if (e.words_bytes != wb || e.scale_bytes != sb)
            return reject("entry byte count disagrees with its shape");
        if (e.offset != off)                           return reject("entry offsets not contiguous");
        off += wb + sb;
    }
    if (off != h.table_offset) return reject("payload does not reach the table");
    if (::lseek(fd, off_t(sizeof(XPackHeader)), SEEK_SET) < 0) return reject("seek failed");
    return fd;
}

// ---------------------------------------------------------------------------
// The load path
// ---------------------------------------------------------------------------

// The full list of quantised tensors, validated (existence, shape, block
// divisibility, dequantisability) BEFORE any thread starts, so a bad model
// throws from the calling thread with a useful message instead of from a
// worker. Order is the contract with the cache file's entry table.
std::vector<PackTask> build_pack_tasks(const GGUFFile& gguf, const DecodeConfig& cfg,
                                       MatvecLayout layout) {
    const uint32_t hidden = cfg.hidden;
    const uint32_t q_dim  = cfg.n_heads * cfg.head_dim;
    const uint32_t kv_dim = cfg.n_kv_heads * cfg.head_dim;
    const uint32_t ffn    = cfg.ffn;

    std::vector<PackTask> tasks;
    auto add = [&](const std::string& name, uint32_t rows, uint32_t cols) {
        const Tensor& t = need_tensor(gguf, name);
        if (t.shape.size() != 2 || t.shape[0] != rows || t.shape[1] != cols)
            throw std::runtime_error("decode: tensor '" + name + "' has unexpected shape (want {" +
                                     std::to_string(rows) + "," + std::to_string(cols) + "})");
        // Km4Bf16 packs four blocks into one uint2, and the kernel's nblk/4
        // loop would SILENTLY DROP a partial quad — wrong weights, no error.
        // Every Gemma 3 1B shape divides (1152/32=36, 1024/32=32, 6912/32=216),
        // but nothing enforced it until this threw.
        if (layout == MatvecLayout::Km4Bf16 && (rows / kQBlock) % 4)
            throw std::runtime_error("decode: '" + name + "' has " +
                                     std::to_string(rows / kQBlock) +
                                     " quantisation blocks, which the bf16 scale plane needs"
                                     " to be a multiple of 4 (rows a multiple of 128)");
        if (rows % kQBlock)
            throw std::runtime_error("decode: '" + name + "' has " + std::to_string(rows) +
                                     " rows, not a multiple of " + std::to_string(kQBlock));
        if (cols % kGroup)
            throw std::runtime_error("decode: '" + name + "' has " + std::to_string(cols) +
                                     " columns, which is not a multiple of the " +
                                     std::to_string(kGroup) + "-column packing group");
        if (name.size() >= kXPackMaxName)
            throw std::runtime_error("decode: tensor name '" + name + "' is too long to cache");
        PackTask task;
        task.name = name;
        task.src  = &t;
        task.rows = rows;
        task.cols = cols;
        RowReader{&t, rows}.check();
        tasks.push_back(std::move(task));
    };

    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add(p + "attn_q.weight",      hidden, q_dim);
        add(p + "attn_k.weight",      hidden, kv_dim);
        add(p + "attn_v.weight",      hidden, kv_dim);
        add(p + "attn_output.weight", q_dim,  hidden);
        add(p + "ffn_gate.weight",    hidden, ffn);
        add(p + "ffn_up.weight",      hidden, ffn);
        add(p + "ffn_down.weight",    ffn,    hidden);
    }
    // Gemma ties the output projection to the embedding table; llama.cpp only
    // emits `output.weight` when they are untied.
    add(gguf.tensor("output.weight") ? "output.weight" : "token_embd.weight", hidden, cfg.vocab);
    return tasks;
}

// Packs one whole tensor, threaded over its column groups. Used by
// --verify-cache; the load path uses the wave scheduler below, which threads
// across tensors as well and so keeps every core busy on the small ones.
void pack_one_tensor(const PackTask& t, MatvecLayout layout, uint32_t nthreads,
                     PackedHost& out);

size_t packed_words_bytes(const PackTask& t) {
    return size_t(t.rows) * t.cols / 8 * sizeof(uint32_t);
}
size_t packed_scale_bytes(const PackTask& t, MatvecLayout layout) {
    return layout == MatvecLayout::Km4Bf16
               ? size_t(t.rows) / kQBlock / 4 * t.cols * 2 * sizeof(uint32_t)
               : size_t(t.rows) / kQBlock * t.cols * sizeof(float);
}

// Sizes the host vectors for one tensor. `tmp_scales` is the fp32 scale plane
// the packer writes into; for the non-bf16 layouts it IS the final buffer, so
// it is aliased onto PackedHost::scale_f rather than copied.
void size_packed(const PackTask& t, MatvecLayout layout, PackedHost& out,
                 std::vector<float>& tmp_scales) {
    out.words.resize(size_t(t.rows) * t.cols / 8);
    const size_t nscale = size_t(t.rows) / kQBlock * t.cols;
    if (layout == MatvecLayout::Km4Bf16) {
        out.scale_u.resize(size_t(t.rows) / kQBlock / 4 * t.cols * 2);
        tmp_scales.resize(nscale);
    } else {
        out.scale_f.resize(nscale);
        tmp_scales.clear();
    }
}

void pack_one_tensor(const PackTask& t, MatvecLayout layout, uint32_t nthreads,
                     PackedHost& out) {
    std::vector<float> tmp;
    size_packed(t, layout, out, tmp);
    float* sc = layout == MatvecLayout::Km4Bf16 ? tmp.data() : out.scale_f.data();
    const uint32_t ngroups = t.cols / kGroup;
    std::atomic<uint32_t> next{0};
    const uint32_t nt = std::max(1u, std::min(nthreads, ngroups));
    std::vector<std::thread> pool;
    for (uint32_t w = 0; w < nt; ++w)
        pool.emplace_back([&] {
            for (;;) {
                const uint32_t g = next.fetch_add(1, std::memory_order_relaxed);
                if (g >= ngroups) return;
                pack_chunk(t, g, g + 1, layout, sc, out);
            }
        });
    for (auto& th : pool) th.join();
}

// Packs `tasks` and uploads them into `dst`, from the cache if there is a
// trustworthy one and from the GGUF otherwise.
//
// THE THREADING RULE, which is the whole design constraint here: vulkore::Context
// is single-threaded — allocation, upload and destruction alike — so the only
// thing worker threads ever touch is a PackedHost, which is plain host memory.
// Uploads happen on this thread, in task order, after a wave has been packed.
//
// Waves exist because the packed model is 536 MiB and the GPU copy is another
// 536 MiB. Packing everything before uploading anything would double peak RSS
// on a phone for no gain; a wave is capped at paths.pack_wave_bytes of packed
// host data.
void load_packed_weights(Context& ctx, const Paths& paths, MatvecLayout layout,
                         const std::vector<PackTask>& tasks,
                         const std::vector<QWeight*>& dst, LoadStats& stats) {
    const double t_start = now_ms();
    stats.threads = paths.pack_threads ? paths.pack_threads
                                       : std::max(1u, std::thread::hardware_concurrency());

    XPackHeader want{};
    const bool stampable = expected_header(paths, layout, uint32_t(tasks.size()), want);
    const std::string cache_path = default_cache_path(paths);
    const bool cache_enabled = paths.use_pack_cache && stampable && !cache_path.empty();
    stats.cache_path = cache_enabled ? cache_path : std::string{};
    if (paths.use_pack_cache && !stampable)
        std::fprintf(stderr, "decode: cannot stat the model file — pack cache disabled\n");

    // ---- fast path: read the packed bytes straight back -------------------
    if (cache_enabled && !paths.force_repack) {
        std::vector<XPackEntry> table;
        const int fd = open_valid_cache(cache_path, want, tasks, table);
        if (fd >= 0) {
            bool ok = true;
            PackedHost host;
            std::vector<float> unused;
            for (size_t i = 0; i < tasks.size() && ok; ++i) {
                size_packed(tasks[i], layout, host, unused);
                const size_t wb = host.words_bytes(), sb = host.scale_bytes();
                if (wb != table[i].words_bytes || sb != table[i].scale_bytes) { ok = false; break; }
                const double r0 = now_ms();
                if (!read_exact(fd, host.words.data(), wb)) { ok = false; break; }
                void* sp = host.scale_u.empty() ? static_cast<void*>(host.scale_f.data())
                                                : static_cast<void*>(host.scale_u.data());
                if (!read_exact(fd, sp, sb)) { ok = false; break; }
                stats.cache_io_ms += now_ms() - r0;
                // Hash both halves as one stream, exactly as they were written.
                const double h0 = now_ms();
                uint64_t h = xpack_hash_mt(host.words.data(), wb, stats.threads);
                h ^= xpack_hash_mt(sp, sb, stats.threads) * 1099511628211ull;
                stats.cache_hash_ms += now_ms() - h0;
                if (h != table[i].hash) {
                    std::fprintf(stderr, "decode: pack cache entry '%s' failed its checksum\n",
                                 tasks[i].name.c_str());
                    ok = false;
                    break;
                }
                const double u0 = now_ms();
                *dst[i] = upload_packed(ctx, host, tasks[i].rows, tasks[i].cols);
                stats.upload_ms += now_ms() - u0;
                stats.cache_bytes += wb + sb;
                if ((i % 32) == 0)
                    std::fprintf(stderr, "\rdecode: loading cached weights %zu/%zu ...", i + 1,
                                 tasks.size());
            }
            ::close(fd);
            if (ok) {
                std::fprintf(stderr, "\rdecode: loaded %s of packed weights from %s        \n",
                             human_bytes(stats.cache_bytes).c_str(), cache_path.c_str());
                stats.from_cache    = true;
                stats.cache_read_ms = now_ms() - t_start;
                return;
            }
            // Anything wrong at all and we repack from the model. The buffers
            // already uploaded above are simply overwritten below — a wasted
            // upload is not a correctness problem, whereas continuing with a
            // half-verified cache would be.
            std::fprintf(stderr, "\rdecode: pack cache unusable, repacking from the model    \n");
            stats.cache_bytes = 0;
        }
    }

    // ---- slow path: pack from the GGUF, in parallel, wave by wave ---------
    const double t_pack0 = now_ms();
    int wfd = -1;
    std::string tmp_path;
    std::vector<XPackEntry> table;
    if (cache_enabled) {
        tmp_path = cache_path + ".tmp";
        wfd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (wfd < 0)
            std::fprintf(stderr, "decode: cannot write %s (%s) — not caching packed weights\n",
                         tmp_path.c_str(), std::strerror(errno));
        else {
            XPackHeader blank{};
            if (!write_exact(wfd, &blank, sizeof blank)) { ::close(wfd); wfd = -1; }
        }
    }
    uint64_t off = sizeof(XPackHeader);

    std::vector<PackedHost>        host(tasks.size());
    std::vector<std::vector<float>> tmp(tasks.size());

    struct Chunk { size_t task; uint32_t g0, g1; };

    size_t wave0 = 0;
    while (wave0 < tasks.size()) {
        // Take tasks until the wave's packed footprint hits the cap. At least
        // one, so a single tensor larger than the cap still makes progress.
        size_t wave1 = wave0;
        uint64_t bytes = 0;
        while (wave1 < tasks.size()) {
            const uint64_t b = packed_words_bytes(tasks[wave1]) +
                               packed_scale_bytes(tasks[wave1], layout) +
                               (layout == MatvecLayout::Km4Bf16
                                    ? uint64_t(tasks[wave1].rows) / kQBlock * tasks[wave1].cols * 4
                                    : 0);
            if (wave1 > wave0 && bytes + b > paths.pack_wave_bytes) break;
            bytes += b;
            ++wave1;
        }

        std::vector<Chunk> chunks;
        for (size_t i = wave0; i < wave1; ++i) {
            size_packed(tasks[i], layout, host[i], tmp[i]);
            const uint32_t ngroups = tasks[i].cols / kGroup;
            // ~2M weights per chunk: small enough that the biggest tensor
            // (1152 x 262144) splits into ~150 pieces and does not become the
            // critical path on its own, large enough that scheduling is free.
            uint32_t gpc = uint32_t(std::max<uint64_t>(
                1, 2000000ull / (uint64_t(tasks[i].rows) * kGroup)));
            gpc = std::min(gpc, ngroups);
            for (uint32_t g = 0; g < ngroups; g += gpc)
                chunks.push_back({i, g, std::min(g + gpc, ngroups)});
        }

        std::atomic<size_t> next{0};
        std::mutex          err_mu;
        std::exception_ptr  err;
        const uint32_t nthreads =
            std::min<uint32_t>(stats.threads, std::max<size_t>(1, chunks.size()));
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (uint32_t w = 0; w < nthreads; ++w) {
            pool.emplace_back([&] {
                for (;;) {
                    const size_t c = next.fetch_add(1, std::memory_order_relaxed);
                    if (c >= chunks.size()) return;
                    {
                        std::lock_guard<std::mutex> lk(err_mu);
                        if (err) return;
                    }
                    try {
                        const Chunk& ck = chunks[c];
                        float* sc = layout == MatvecLayout::Km4Bf16 ? tmp[ck.task].data()
                                                                    : host[ck.task].scale_f.data();
                        pack_chunk(tasks[ck.task], ck.g0, ck.g1, layout, sc, host[ck.task]);
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(err_mu);
                        if (!err) err = std::current_exception();
                        return;
                    }
                }
            });
        }
        for (auto& t : pool) t.join();
        if (err) {
            if (wfd >= 0) { ::close(wfd); ::unlink(tmp_path.c_str()); }
            std::rethrow_exception(err);
        }

        for (size_t i = wave0; i < wave1; ++i) {
            const double u0 = now_ms();
            *dst[i] = upload_packed(ctx, host[i], tasks[i].rows, tasks[i].cols);
            stats.upload_ms += now_ms() - u0;

            if (wfd >= 0) {
                const double c0 = now_ms();
                const size_t wb = host[i].words_bytes(), sb = host[i].scale_bytes();
                XPackEntry e{};
                std::snprintf(e.name, sizeof e.name, "%s", tasks[i].name.c_str());
                e.rows        = tasks[i].rows;
                e.cols        = tasks[i].cols;
                e.offset      = off;
                e.words_bytes = wb;
                e.scale_bytes = sb;
                e.hash        = xpack_hash_mt(host[i].words.data(), wb, stats.threads) ^
                                (xpack_hash_mt(host[i].scale_data(), sb, stats.threads) *
                                 1099511628211ull);
                if (!write_exact(wfd, host[i].words.data(), wb) ||
                    !write_exact(wfd, host[i].scale_data(), sb)) {
                    std::fprintf(stderr, "decode: pack cache write failed (%s) — discarding\n",
                                 std::strerror(errno));
                    ::close(wfd);
                    ::unlink(tmp_path.c_str());
                    wfd = -1;
                } else {
                    table.push_back(e);
                    off += wb + sb;
                }
                stats.cache_write_ms += now_ms() - c0;
            }
            host[i].release();
            std::vector<float>().swap(tmp[i]);
            if ((i % 32) == 0)
                std::fprintf(stderr, "\rdecode: packing %zu/%zu tensors (%u threads) ...", i + 1,
                             tasks.size(), nthreads);
        }
        wave0 = wave1;
    }
    stats.pack_ms = now_ms() - t_pack0 - stats.upload_ms - stats.cache_write_ms;
    std::fprintf(stderr, "\rdecode: packed %zu tensors on %u threads            \n", tasks.size(),
                 stats.threads);

    // ---- finish the cache: table, then the header, then rename ------------
    if (wfd >= 0) {
        const double c0 = now_ms();
        want.table_offset = off;
        want.total_size   = off + uint64_t(table.size()) * sizeof(XPackEntry);
        bool ok = table.size() == tasks.size() &&
                  write_exact(wfd, table.data(), table.size() * sizeof(XPackEntry)) &&
                  ::lseek(wfd, 0, SEEK_SET) == 0 && write_exact(wfd, &want, sizeof want);
        if (ok) ok = ::fsync(wfd) == 0;
        ::close(wfd);
        // rename(2) is the only reason a half-written cache can never be read:
        // the name only ever points at a complete file.
        if (ok && ::rename(tmp_path.c_str(), cache_path.c_str()) == 0) {
            stats.cache_written = true;
            stats.cache_bytes   = want.total_size;
            std::fprintf(stderr, "decode: wrote %s to %s\n",
                         human_bytes(want.total_size).c_str(), cache_path.c_str());
        } else {
            ::unlink(tmp_path.c_str());
            std::fprintf(stderr, "decode: could not finish the pack cache (%s)\n",
                         std::strerror(errno));
        }
        stats.cache_write_ms += now_ms() - c0;
    }
}

}  // namespace

DecodeSession::DecodeSession(const Paths& paths, const DecodeConfig& cfg)
    : cfg_(cfg) {
    impl_ = std::make_unique<Impl>(paths);
    Impl& s = *impl_;

    // ---- kernels: fail loudly and by name --------------------------------
    s.mv = paths.matvec;
    const std::string& mvm = paths.matvec.module;
    s.k_matvec = need_kernel(s.matvec_prog, mvm, s.mv.kernel.c_str());
    if (s.mv.split > 1) {
        s.k_matvec_split = need_kernel(s.matvec_prog, mvm, s.mv.split_kernel.c_str());
        s.k_reduce       = need_kernel(s.matvec_prog, mvm, s.mv.reduce_kernel.c_str());
    }
    const std::string& tfm = paths.transformer_spv;
    s.k_sumsq      = need_kernel(s.tf_prog, tfm, "rmsnorm_sumsq");
    s.k_apply      = need_kernel(s.tf_prog, tfm, "rmsnorm_apply_gemma");
    s.k_heads_norm = need_kernel(s.tf_prog, tfm, "rmsnorm_heads_gemma");
    s.k_rope       = need_kernel(s.tf_prog, tfm, "rope");
    s.k_geglu      = need_kernel(s.tf_prog, tfm, "geglu");
    s.k_scores     = need_kernel(s.tf_prog, tfm, "attention_scores");
    s.k_softmax    = need_kernel(s.tf_prog, tfm, "softmax_rows");
    s.k_attn_apply = need_kernel(s.tf_prog, tfm, "attention_apply");
    s.k_residual   = need_kernel(s.tf_prog, tfm, "add_residual");
    s.k_kv_append  = need_kernel(s.tf_prog, tfm, "kv_append");
    s.k_argmax     = need_kernel(s.tf_prog, tfm, "argmax");

    // PARALLEL ATTENTION (optional; absent from an older module, in which case
    // the serial kernels below are used and nothing else changes).
    //   softmax_rows is one thread per row and there are only n_heads = 4 rows,
    //   so it is the largest depth-dependent term in the token — 2.51 ms per
    //   global layer at span 8288 on an Adreno 840, against 2.05 for scores and
    //   0.68 for apply. attention_scores_mq4 additionally exploits Gemma's
    //   multi-query layout to read each key ONCE instead of once per q head.
    try {
        s.k_sm_part  = need_kernel(s.tf_prog, tfm, "softmax_rows_partial");
        s.k_sm_fin   = need_kernel(s.tf_prog, tfm, "softmax_rows_finish");
        s.k_sm_scale = need_kernel(s.tf_prog, tfm, "softmax_rows_scale");
        s.has_par_softmax = true;
    } catch (const std::exception&) {
        if (cfg_.parallel_softmax)
            std::fprintf(stderr, "[decode] %s has no softmax_rows_partial; "
                         "falling back to the SERIAL softmax_rows (slow at depth)\n",
                         tfm.c_str());
    }
    try {
        s.k_scores_mq4 = need_kernel(s.tf_prog, tfm, "attention_scores_mq4");
        s.k_apply_mq4 = need_kernel(s.tf_prog, tfm, "attention_apply_mq4");
        s.k_apply_red = need_kernel(s.tf_prog, tfm, "attention_apply_reduce");
        s.has_mq4 = true;
    } catch (const std::exception&) {
        if (cfg_.scores_mq4)
            std::fprintf(stderr, "[decode] %s has no attention_scores_mq4; "
                         "falling back to the generic attention_scores\n", tfm.c_str());
    }

    // Parallel argmax if the sampling module is available. The single-thread
    // fallback is CORRECT but costs ~31 ms/token at vocab 262144.
    if (!paths.sampling_spv.empty()) {
        try {
            s.sampling_prog =
                std::make_unique<Program>(Program::from_file(s.ctx, paths.sampling_spv));
            s.k_am_partial = need_kernel(*s.sampling_prog, paths.sampling_spv, "argmax_partial");
            s.k_am_reduce  = need_kernel(*s.sampling_prog, paths.sampling_spv, "argmax_reduce");
            s.k_am_final   = need_kernel(*s.sampling_prog, paths.sampling_spv, "argmax_final");
            s.fast_argmax  = true;
            // Candidate harvest for non-greedy sampling. Resolved separately so
            // an older llm_sampling.spv without it still gets the fast argmax.
            try {
                s.k_topk4 = need_kernel(*s.sampling_prog, paths.sampling_spv, "topk4_partial");
                s.harvest = !cfg.sampling.cpu_full_vocab &&
                            cfg.sampling.mode == SampleMode::Temperature;
            } catch (const std::exception& e2) {
                std::fprintf(stderr,
                             "decode: WARNING no topk4_partial (%s) — non-greedy sampling "
                             "falls back to downloading all %u logits per token\n",
                             e2.what(), cfg.vocab);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "decode: WARNING no parallel argmax (%s) — falling back to the "
                         "single-thread scan, which costs ~31 ms per token at vocab %u\n",
                         e.what(), cfg.vocab);
        }
    }

    // ---- model ------------------------------------------------------------
    if (paths.model_gguf.empty())
        throw std::runtime_error("decode: no model path given (use --dry-run to inspect the "
                                 "dispatch plan without one)");
    s.gguf = GGUFFile::open(paths.model_gguf);
    const ModelConfig mc = s.gguf.config();
    // Shapes come from the file; policy fields (max_seq, sampling, ...) stay.
    DecodeConfig from_file = DecodeConfig::from_model(mc);
    from_file.max_seq      = cfg.max_seq;
    from_file.sampling     = cfg.sampling;
    // Attention-extent policy is a COMMAND-LINE choice, so it has to survive
    // the merge with the file's config. It did not before: from_model() reset
    // sliding_window/global_every to the struct defaults and the caller's
    // values were dropped on the floor. That was invisible while nothing read
    // those fields, and it silently turned `--window` into a no-op the first
    // time something did.
    if (cfg.sliding_window != DecodeConfig{}.sliding_window)
        from_file.sliding_window = cfg.sliding_window;
    from_file.global_every = cfg.global_every;
    from_file.norm_parts   = cfg.norm_parts;
    // Same trap, same fix: these are command-line A/B switches and MUST survive
    // the merge, or `--no-parallel-softmax` becomes a silent no-op and the two
    // arms of a benchmark are the same binary doing the same thing.
    from_file.parallel_softmax = cfg.parallel_softmax;
    from_file.scores_mq4       = cfg.scores_mq4;
    from_file.attn_min_span = cfg.attn_min_span;
    from_file.attn_apply_split = cfg.attn_apply_split;
    from_file.softmax_max_parts = cfg.softmax_max_parts;
    from_file.embed_scale  = cfg.embed_scale;
    from_file.attn_scale   = cfg.attn_scale;
    if (mc.arch != "gemma3")
        std::fprintf(stderr,
                     "decode: WARNING architecture is '%s', not gemma3 — the Gemma-specific "
                     "(1+w) norms, QK-norm and rotate-half RoPE may not apply\n",
                     mc.arch.c_str());
    cfg_ = from_file;
    s.tok = Tokenizer::from_gguf(s.gguf);

    // Decided HERE, after the merge, so the shape checks see the FILE's real
    // n_heads/kv_heads rather than the struct defaults.
    s.par_softmax = s.has_par_softmax && cfg_.parallel_softmax;
    // attention_scores_mq4 keeps one accumulator per head in a register and
    // assumes a single KV head, so it is only valid for exactly this shape.
    s.scores_mq4  = s.has_mq4 && cfg_.scores_mq4 && cfg_.n_heads == 4 &&
                    cfg_.n_kv_heads == 1 && (cfg_.head_dim % 4) == 0;
    if (cfg_.scores_mq4 && s.has_mq4 && !s.scores_mq4)
        std::fprintf(stderr, "[decode] attention_scores_mq4 needs n_heads=4, "
                     "kv_heads=1, head_dim%%4=0; got %u/%u/%u — using the "
                     "generic attention_scores\n",
                     cfg_.n_heads, cfg_.n_kv_heads, cfg_.head_dim);
    cfg_.parallel_softmax = s.par_softmax;
    cfg_.scores_mq4       = s.scores_mq4;

    cfg_.parallel_argmax = s.fast_argmax;
    plan_ = plan_token(cfg_, s.mv);

    const uint32_t hidden  = cfg_.hidden;
    const uint32_t q_dim   = cfg_.n_heads * cfg_.head_dim;
    const uint32_t kv_dim  = cfg_.n_kv_heads * cfg_.head_dim;
    const uint32_t ffn     = cfg_.ffn;

    if (hidden % kQBlock || q_dim % kQBlock || ffn % kQBlock)
        throw std::runtime_error("decode: every matvec `rows` must be a multiple of 32");

    // ---- weights ----------------------------------------------------------
    // Norm weights. The kernels apply (1 + w); a GGUF Gemma file already stores
    // (1 + w), so subtract 1 here or it is applied twice. See
    // DecodeConfig::gguf_norm_includes_bias for what that costs.
    //
    // NOT part of the packed-weight cache: 26 layers x 4 x 1152 + 2 x 256
    // floats is under 500 KiB, dequantised straight out of the mapping in
    // milliseconds. Caching them would add the `gguf_norm_includes_bias`
    // policy flag to everything the cache header has to prove, for no time.
    auto load_f32 = [&](const std::string& name, uint32_t n) {
        const Tensor& t = need_tensor(s.gguf, name);
        if (t.n_elements() != n)
            throw std::runtime_error("decode: norm tensor '" + name + "' has " +
                                     std::to_string(t.n_elements()) + " elements, want " +
                                     std::to_string(n));
        std::vector<float> h = s.gguf.dequantize_tensor(t);
        if (cfg_.gguf_norm_includes_bias)
            for (float& f : h) f -= 1.0f;
        Buffer b = s.ctx.alloc<float>(n);
        b.upload(std::span<const float>(h));
        return b;
    };

    s.layers.resize(cfg_.n_layers);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        LayerWeights& w = s.layers[l];
        w.attn_norm      = load_f32(p + "attn_norm.weight", hidden);
        w.post_attn_norm = load_f32(p + "post_attention_norm.weight", hidden);
        w.ffn_norm       = load_f32(p + "ffn_norm.weight", hidden);
        w.post_ffn_norm  = load_f32(p + "post_ffw_norm.weight", hidden);
        w.q_norm         = load_f32(p + "attn_q_norm.weight", cfg_.head_dim);
        w.k_norm         = load_f32(p + "attn_k_norm.weight", cfg_.head_dim);
    }
    s.final_norm = load_f32("output_norm.weight", hidden);

    // The quantised weights: 26 layers x 7 tensors plus the output head. These
    // are the ~1B parameters, and packing them is the whole of the load time.
    // Every tensor is read at its native quantisation (this file is
    // Q8_0/Q5_0/Q4_K/Q6_K mixed, because K-quants need the fast axis divisible
    // by 256 and Gemma 3's 1152 is 4.5 blocks) and re-quantised to a single
    // uniform int4 format — see agent-docs/llm-decode-loop.md for that trade.
    std::vector<PackTask>  tasks = build_pack_tasks(s.gguf, cfg_, s.mv.layout);
    std::vector<QWeight*>  dst;
    dst.reserve(tasks.size());
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        LayerWeights& w = s.layers[l];
        for (QWeight* q : {&w.wq, &w.wk, &w.wv, &w.wo, &w.w_gate, &w.w_up, &w.w_down})
            dst.push_back(q);
    }
    dst.push_back(&s.lm_head);
    if (dst.size() != tasks.size())
        throw std::runtime_error("decode: internal — pack task/destination count mismatch");

    load_packed_weights(s.ctx, paths, s.mv.layout, tasks, dst, s.load_stats);
    for (const QWeight* q : dst) s.weight_bytes += q->bytes;

    // KV cache: [token][kv_head][head_dim], one K and one V per layer. Timed
    // separately from the weight packing because raising max_seq is a choice
    // about ALLOCATION and RESIDENCY, and the load-time cost of that choice has
    // to be a measured number, not an assumption.
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const double kt0 = now_ms();
        s.layers[l].kcache = s.ctx.alloc<float>(size_t(cfg_.max_seq) * kv_dim);
        s.layers[l].vcache = s.ctx.alloc<float>(size_t(cfg_.max_seq) * kv_dim);
        s.kv_alloc_ms += now_ms() - kt0;
    }

    // The embedding gather stays on the CPU, straight out of the mapping.
    s.embed_rows = RowReader{&need_tensor(s.gguf, "token_embd.weight"), hidden};
    s.embed_rows.check();
    s.embed_scratch.resize(hidden);

    // ---- activations ------------------------------------------------------
    s.x            = s.ctx.alloc<float>(hidden, Usage::HostVisible);
    s.xb           = s.ctx.alloc<float>(hidden);
    s.xb2          = s.ctx.alloc<float>(hidden);
    s.q_raw        = s.ctx.alloc<float>(q_dim);
    s.q            = s.ctx.alloc<float>(q_dim);
    s.k_raw        = s.ctx.alloc<float>(kv_dim);
    s.k            = s.ctx.alloc<float>(kv_dim);
    s.v            = s.ctx.alloc<float>(kv_dim);
    s.att          = s.ctx.alloc<float>(q_dim);
    s.scores       = s.ctx.alloc<float>(size_t(cfg_.n_heads) * cfg_.max_seq);
    // Parallel-softmax scratch, sized for the largest nparts any span can ask
    // for. Tiny: at nparts 256 and 4 heads this is 4 KiB total.
    s.sm_pmax      = s.ctx.alloc<float>(size_t(cfg_.n_heads) * cfg_.softmax_max_parts);
    s.sm_psum      = s.ctx.alloc<float>(size_t(cfg_.n_heads) * cfg_.softmax_max_parts);
    s.sm_stats     = s.ctx.alloc<float>(size_t(cfg_.n_heads) * 2);
    s.att_partial  = s.ctx.alloc<float>(size_t(cfg_.n_heads) * cfg_.head_dim *
                                        cfg_.attn_apply_split);
    s.hb_gate      = s.ctx.alloc<float>(ffn);
    s.hb_up        = s.ctx.alloc<float>(ffn);
    s.hb           = s.ctx.alloc<float>(ffn);
    s.logits       = s.ctx.alloc<float>(cfg_.vocab);
    s.sample_out   = s.ctx.alloc<float>(2, Usage::HostVisible);
    s.norm_partial = s.ctx.alloc<float>(cfg_.norm_parts);

    // One scratch buffer serves every split matvec: they are sequential and
    // separated by Batch's barriers. Size it for the worst split'd shape.
    size_t partial_n = 1;
    for (uint32_t cols : {q_dim, kv_dim, hidden, ffn, cfg_.vocab})
        partial_n = std::max<size_t>(partial_n,
                                     size_t(cols) * s.mv.split_for(ffn, cols));
    s.mv_partial = s.ctx.alloc<float>(partial_n);

    if (s.fast_argmax) {
        s.am_val     = s.ctx.alloc<float>(cfg_.argmax_parts);
        s.am_idx     = s.ctx.alloc<uint32_t>(cfg_.argmax_parts);
        s.am_val2    = s.ctx.alloc<float>(cfg_.argmax_parts2);
        s.am_idx2    = s.ctx.alloc<uint32_t>(cfg_.argmax_parts2);
        s.am_out_idx = s.ctx.alloc<uint32_t>(1);
    }
    if (s.harvest) {
        // HostVisible: the whole point is that the host reads these out of a
        // mapping with NO staging copy and therefore NO extra vkQueueSubmit.
        // 4 * 4096 pairs = 128 KiB, against 1 MiB staged for the full vocab.
        const uint32_t nc = cfg_.sampling.topk_parts * 4u;
        s.cand_val = s.ctx.alloc<float>(nc, Usage::HostVisible);
        s.cand_idx = s.ctx.alloc<uint32_t>(nc, Usage::HostVisible);
        s.cand_val_host.resize(nc);
        s.cand_idx_host.resize(nc);
    }
}

DecodeSession::~DecodeSession() = default;

std::string DecodeSession::device_name() const { return impl_->ctx.device_name(); }
uint64_t DecodeSession::weight_bytes() const { return impl_->weight_bytes; }
double   DecodeSession::kv_alloc_ms() const { return impl_->kv_alloc_ms; }
const LoadStats& DecodeSession::load_stats() const { return impl_->load_stats; }
const Tokenizer& DecodeSession::tokenizer() const { return impl_->tok; }
void DecodeSession::reset() { pos_ = 0; }

// The embedding gather is a CPU dequantise of one GGUF row plus Gemma's
// sqrt(hidden) scale, memcpy'd into the mapped residual buffer. No dispatch, no
// submit, and no 1.2 GB fp32 embedding table resident on the GPU.
void DecodeSession::write_embedding(uint32_t token) {
    Impl& s = *impl_;
    if (token >= cfg_.vocab)
        throw std::runtime_error("decode: token id " + std::to_string(token) +
                                 " is outside the vocabulary");
    s.embed_rows.read(token, s.embed_scratch.data());
    const float sc = cfg_.effective_embed_scale();
    for (float& f : s.embed_scratch) f *= sc;
    s.x.upload(std::span<const float>(s.embed_scratch));
}

// One token, recorded into ONE command buffer. Every dispatch here is ordered
// against the previous one by the barrier Batch inserts, so aliasing rules are
// the ordinary read-after-write ones — except that a kernel must never take
// the same Buffer as both `in` and `out` unless it is documented in-place
// (only `rope` is), because threads within a single dispatch are unordered.
uint32_t DecodeSession::record_token(Batch& b, uint32_t pos, bool want_logits) {
    Impl& s = *impl_;
    const DecodeConfig& cfg = cfg_;
    const uint32_t hidden  = cfg.hidden;
    const uint32_t q_dim   = cfg.n_heads * cfg.head_dim;
    const uint32_t kv_dim  = cfg.n_kv_heads * cfg.head_dim;
    const uint32_t ffn     = cfg.ffn;
    const uint32_t seq_len = pos + 1;
    const uint32_t nparts  = cfg.norm_parts;
    const float    eps     = cfg.rms_eps;
    const float    ascale  = cfg.effective_attn_scale();
    uint32_t n = 0;

    auto norm = [&](Buffer& out, Buffer& in, Buffer& w, uint32_t len) {
        b.add(s.k_sumsq, {nparts}, s.norm_partial, in, len, nparts);
        b.add(s.k_apply, {len}, out, in, w, s.norm_partial, len, nparts, eps);
        n += 2;
    };
    // Split-k costs one extra reduce dispatch and buys parallelism where a
    // thread-per-output-column cannot fill the GPU (2.0x on `down`). Neutral
    // to negative on the wide layers, so split_for() gates it on `cols`.
    auto matvec = [&](Buffer& out, Buffer& in, const QWeight& w) {
        const uint32_t sp = s.mv.split_for(w.rows, w.cols);
        if (sp > 1) {
            b.add(s.k_matvec_split, {w.cols * sp}, s.mv_partial, in, w.q, w.scale,
                  w.rows, w.cols, sp);
            b.add(s.k_reduce, {w.cols}, out, s.mv_partial, w.cols, sp);
            n += 2;
        } else {
            b.add(s.k_matvec, {w.cols}, out, in, w.q, w.scale, w.rows, w.cols);
            ++n;
        }
    };

    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        LayerWeights& w     = s.layers[l];
        const float   theta = cfg.rope_theta(l);

        norm(s.xb, s.x, w.attn_norm, hidden);
        matvec(s.q_raw, s.xb, w.wq);
        matvec(s.k_raw, s.xb, w.wk);
        matvec(s.v,     s.xb, w.wv);

        // Gemma 3 QK-norm, then RoPE, then commit to the cache.
        b.add(s.k_heads_norm, {q_dim},  s.q, s.q_raw, w.q_norm, cfg.n_heads,    cfg.head_dim, eps);
        b.add(s.k_heads_norm, {kv_dim}, s.k, s.k_raw, w.k_norm, cfg.n_kv_heads, cfg.head_dim, eps);
        b.add(s.k_rope, {q_dim / 2},  s.q, cfg.n_heads,    cfg.head_dim, pos, theta);
        b.add(s.k_rope, {kv_dim / 2}, s.k, cfg.n_kv_heads, cfg.head_dim, pos, theta);
        b.add(s.k_kv_append, {kv_dim}, w.kcache, s.k, kv_dim, pos);
        b.add(s.k_kv_append, {kv_dim}, w.vcache, s.v, kv_dim, pos);
        n += 6;

        // SLIDING WINDOW. Gemma 3 makes 5 of every 6 layers local with a
        // 512-position window; only layers 5/11/17/23 (i.e. (i+1)%6==0) see the
        // whole cache. A decode query at `pos` attends t in (pos - window, pos],
        // which is a contiguous SUFFIX of the cache, so it is expressed as an
        // offset+length rather than a mask: no -INFINITY, and the work on a
        // local layer stops growing once seq_len passes the window.
        const uint32_t kv_start = cfg.attn_kv_start(l, seq_len);
        const uint32_t span     = seq_len - kv_start;

        // All three parallel forms are gated on the SAME span threshold. Below
        // it every one of them is measurably SLOWER on an Adreno 840 (0.67x /
        // 0.77x / 0.92x at span 41) because the extra dispatches cost more than
        // the serial work they remove. Flatness must not be bought by making
        // the shallow case worse.
        if (s.scores_mq4 && span >= cfg.attn_min_span) {
            b.add(s.k_scores_mq4, {span}, s.scores, s.q, w.kcache,
                  cfg.head_dim, span, ascale, kv_start);
        } else {
            b.add(s.k_scores, {cfg.n_heads * span}, s.scores, s.q, w.kcache,
                  cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, span, ascale, kv_start);
        }
        ++n;

        // Below the threshold the 3-dispatch form costs more in dispatch
        // overhead than the serial row scan costs in work, so keep the serial
        // kernel there. ~5.6 us x 2 extra dispatches is the break-even.
        const uint32_t nparts = cfg.softmax_parts_for(span);
        if (s.par_softmax && nparts > 1) {
            b.add(s.k_sm_part, {cfg.n_heads * nparts}, s.sm_pmax, s.sm_psum,
                  s.scores, cfg.n_heads, span, nparts);
            b.add(s.k_sm_fin, {cfg.n_heads}, s.sm_stats, s.sm_pmax, s.sm_psum,
                  cfg.n_heads, nparts);
            b.add(s.k_sm_scale, {cfg.n_heads * span}, s.scores, s.sm_stats,
                  cfg.n_heads, span);
            n += 3;
        } else {
            b.add(s.k_softmax, {cfg.n_heads}, s.scores, cfg.n_heads, span);
            ++n;
        }

        if (s.scores_mq4 && span >= cfg.attn_min_span) {
            const uint32_t sp = cfg.attn_apply_split;
            b.add(s.k_apply_mq4, {cfg.head_dim * sp}, s.att_partial, s.scores,
                  w.vcache, cfg.head_dim, span, kv_start, sp);
            b.add(s.k_apply_red, {q_dim}, s.att, s.att_partial, q_dim, sp);
            n += 2;
        } else {
            b.add(s.k_attn_apply, {q_dim}, s.att, s.scores, w.vcache,
                  cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, span, kv_start);
            ++n;
        }

        matvec(s.xb, s.att, w.wo);
        norm(s.xb2, s.xb, w.post_attn_norm, hidden);
        b.add(s.k_residual, {hidden}, s.x, s.xb2, hidden);
        ++n;

        norm(s.xb, s.x, w.ffn_norm, hidden);
        matvec(s.hb_gate, s.xb, w.w_gate);
        matvec(s.hb_up,   s.xb, w.w_up);
        b.add(s.k_geglu, {ffn}, s.hb, s.hb_gate, s.hb_up, ffn);
        ++n;
        matvec(s.xb, s.hb, w.w_down);
        norm(s.xb2, s.xb, w.post_ffn_norm, hidden);
        b.add(s.k_residual, {hidden}, s.x, s.xb2, hidden);
        ++n;
    }

    norm(s.xb, s.x, s.final_norm, hidden);
    if (want_logits) {
        matvec(s.logits, s.xb, s.lm_head);
        if (s.harvest) {
            // ONE dispatch, inside the token's own Batch, writing 128 KiB into
            // HOST-VISIBLE memory. The host then finishes top-k + softmax +
            // multinomial over 16384 candidates with no submit and no
            // full-vocab scan. This REPLACES the argmax tree — a temperature
            // sample never reads sample_out.
            b.add(s.k_topk4, {cfg.sampling.topk_parts}, s.cand_val, s.cand_idx,
                  s.logits, cfg.vocab, cfg.sampling.topk_parts);
            ++n;
        } else if (s.fast_argmax) {
            // vocab -> argmax_parts -> argmax_parts2 -> 1. Three dispatches
            // inside the Batch cost ~0.02 ms against ~31 ms for one serial scan.
            const uint32_t p1 = cfg.argmax_parts, p2 = cfg.argmax_parts2;
            b.add(s.k_am_partial, {p1}, s.am_val, s.am_idx, s.logits, cfg.vocab, p1);
            b.add(s.k_am_reduce, {p2}, s.am_val2, s.am_idx2, s.am_val, s.am_idx, p1, p2);
            b.add(s.k_am_final, {1}, s.sample_out, s.am_out_idx, s.am_val2, s.am_idx2, p2);
            n += 3;
        } else {
            b.add(s.k_argmax, {1}, s.sample_out, s.logits, cfg.vocab);
            ++n;
        }
    }
    return n;
}

// Runs one network pass per prompt token. `want_logits` is false for all but
// the last, which skips the lm_head matvec entirely — 151 MB of traffic, the
// whole difference between a prefill token and a generated one.
//
// This is decode-shaped prefill: one token at a time, reusing the single-token
// path. A real batched prefill would compute all prompt positions in one pass,
// but softmax_rows is one-thread-per-row and attention_scores has no causal
// mask, so both would have to change. See agent-docs/llm-decode-loop.md.
void DecodeSession::prefill(std::span<const uint32_t> tokens) {
    if (tokens.empty()) return;
    Impl& s = *impl_;
    const double t0 = now_ms();
    uint32_t dispatches = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (pos_ >= cfg_.max_seq)
            throw std::runtime_error("decode: prompt exceeds max_seq (" +
                                     std::to_string(cfg_.max_seq) + ")");
        write_embedding(tokens[i]);

        // Only the LAST prompt token needs logits — it is the one whose
        // distribution produces the first generated token.
        const bool last = (i + 1 == tokens.size());
        Batch batch(s.ctx, s.dcache);
        dispatches += record_token(batch, pos_, last);
        Fence f = batch.submit();   // ONE submit; held, not discarded (§5.3)
        f.wait();
        cur_token_ = tokens[i];
        ++pos_;
    }

    const double ms = now_ms() - t0;
    metrics_ = Metrics{ms, 0, ms, 0,
                       ms > 0 ? tokens.size() * 1000.0 / ms : 0.0,
                       dispatches, uint32_t(tokens.size()), pos_};
}

uint32_t DecodeSession::decode_step() {
    Impl& s = *impl_;
    if (pos_ == 0)
        throw std::runtime_error("decode: decode_step() before prefill() — there is no context "
                                 "to condition on");
    if (pos_ >= cfg_.max_seq)
        throw std::runtime_error("decode: KV cache is full (max_seq " +
                                 std::to_string(cfg_.max_seq) + ")");

    const double t0 = now_ms();

    // SAMPLE FIRST, then run the network on what we sampled. The logits sitting
    // on the device were produced by the previous pass — prefill's last prompt
    // token, or the previous decode_step. Running the network first and then
    // sampling would feed the last prompt token through TWICE, once at its own
    // position during prefill and again at the next position here, which
    // corrupts both the KV cache and every subsequent position index.
    uint32_t token = 0;
    if (cfg_.sampling.mode == SampleMode::Greedy) {
        float out[2] = {0, 0};
        s.sample_out.download(std::span<float>(out, 2));  // mapped read, no submit
        token = uint32_t(out[0]);
    } else if (s.harvest) {
        token = sample_candidates();  // mapped read, no submit
    } else {
        std::vector<float> logits(cfg_.vocab);
        s.logits.download(std::span<float>(logits));
        token = sample_cpu(logits);
    }
    const double t_sample = now_ms();
    cur_token_ = token;

    // Forward pass on the sampled token, leaving logits ready for the next call.
    write_embedding(token);
    Batch batch(s.ctx, s.dcache);
    const uint32_t dispatches = record_token(batch, pos_, true);
    const double t_rec = now_ms();

    Fence fence = batch.submit();
    fence.wait();
    const double t1 = now_ms();

    ++pos_;
    metrics_ = Metrics{t1 - t0, t_rec - t_sample, t1 - t_rec, t_sample - t0,
                       (t1 - t0) > 0 ? 1000.0 / (t1 - t0) : 0.0,
                       dispatches, 1, pos_};
    return token;
}

// Temperature + softmax + multinomial over an ALREADY-SELECTED candidate list.
// Both sampling paths end here, so they differ only in HOW the candidates were
// found — which is what makes `--sample-check` a fair comparison.
//
// `cand` is truncated to top_k in place. It need not be sorted on entry.
uint32_t DecodeSession::draw(std::vector<std::pair<float, uint32_t>>& cand) {
    const Sampling& sp = cfg_.sampling;
    const float t = sp.temperature > 0.0f ? sp.temperature : 1.0f;
    if (cand.empty()) return 0;

    // Ties break on the LOWER index, matching argmax_partial / topk4_partial.
    const auto better = [](const std::pair<float, uint32_t>& a,
                           const std::pair<float, uint32_t>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    const uint32_t k = sp.top_k ? std::min<uint32_t>(sp.top_k, uint32_t(cand.size()))
                                : uint32_t(cand.size());
    if (k < cand.size()) {
        std::nth_element(cand.begin(), cand.begin() + k, cand.end(), better);
        cand.resize(k);
    }

    float maxl = -INFINITY;
    for (auto& c : cand) maxl = std::max(maxl, c.first);
    double sum = 0;
    for (auto& c : cand) { c.first = std::exp((c.first - maxl) / t); sum += c.first; }

    // Nucleus (top-p) filter, applied AFTER the temperature softmax so the mass
    // is the real post-temperature distribution. Needs the candidates in
    // descending order; nth_element above only partitioned them.
    if (sp.top_p > 0.0f && sp.top_p < 1.0f && sum > 0) {
        std::sort(cand.begin(), cand.end(), better);
        double acc = 0, cut = sp.top_p * sum;
        size_t keep = 0;
        // `keep` lands on the first candidate that carries the running mass past
        // the threshold, and that one is INCLUDED -- so the kept set always
        // covers >= top_p, and is never empty even if one token holds all mass.
        for (; keep < cand.size(); ++keep) { acc += cand[keep].first; if (acc >= cut) break; }
        cand.resize(std::min(cand.size(), keep + 1));
        sum = 0;
        for (const auto& c : cand) sum += c.first;
    }

    auto& rng = impl_->rng;
    if (!rng) rng = std::make_unique<std::mt19937_64>(sp.seed);
    std::uniform_real_distribution<double> u(0.0, sum);
    double r = u(*rng);
    for (const auto& c : cand) { r -= c.first; if (r <= 0) return c.second; }
    return cand.back().second;
}

// THE FAST PATH. `topk4_partial` already reduced the vocabulary to
// 4*topk_parts candidates inside the token's Batch; this reads them out of
// MAPPED memory (no submit) and finishes on the host.
uint32_t DecodeSession::sample_candidates() {
    Impl& s = *impl_;
    s.cand_val.download(std::span<float>(s.cand_val_host));
    s.cand_idx.download(std::span<uint32_t>(s.cand_idx_host));

    std::vector<std::pair<float, uint32_t>>& cand = s.cand_scratch;
    cand.clear();
    cand.reserve(s.cand_val_host.size());
    for (size_t i = 0; i < s.cand_val_host.size(); ++i) {
        // Sentinel for an unfilled slot (a chain shorter than 4). Cannot occur
        // at vocab 262144 / 4096 chains, but the kernel emits it and dropping
        // it here keeps the two paths agreeing if nparts is ever raised.
        if (s.cand_idx_host[i] == 0xFFFFFFFFu) continue;
        cand.emplace_back(s.cand_val_host[i], s.cand_idx_host[i]);
    }
    return draw(cand);
}

std::vector<std::pair<float, uint32_t>> DecodeSession::last_candidates() const {
    Impl& s = *impl_;
    std::vector<std::pair<float, uint32_t>> out;
    if (!s.harvest) return out;
    std::vector<float>    v(s.cand_val_host.size());
    std::vector<uint32_t> d(s.cand_idx_host.size());
    s.cand_val.download(std::span<float>(v));
    s.cand_idx.download(std::span<uint32_t>(d));
    // RAW, sentinels included: slot j of chain p is at p*4 + j, and
    // `--sample-check` needs slot 0 alone to score what a plain
    // `argmax_partial` harvest (one candidate per chain) would have given.
    out.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i) out.emplace_back(v[i], d[i]);
    return out;
}

bool DecodeSession::candidate_harvest_active() const { return impl_->harvest; }

// THE SLOW PATH, kept UNCHANGED for A/B (`--sample-cpu`) — deliberately not
// refactored through draw(), because it is the measurement baseline and a
// "harmless" rewrite (contiguous pairs instead of an indirect comparator) would
// have made the before number look better than what was actually shipped.
//
// Costs a `vocab`-float download (1 MiB for Gemma) through the staging path —
// an extra vkQueueSubmit — plus an nth_element over 262144 elements with a
// cache-hostile indirect comparator. Measured at ~12 ms/token on an Adreno 840.
uint32_t DecodeSession::sample_cpu(std::vector<float>& logits) {
    const Sampling& sp = cfg_.sampling;
    const float t = sp.temperature > 0.0f ? sp.temperature : 1.0f;

    std::vector<uint32_t> idx;
    const uint32_t k = sp.top_k ? std::min<uint32_t>(sp.top_k, cfg_.vocab) : cfg_.vocab;
    idx.resize(cfg_.vocab);
    std::iota(idx.begin(), idx.end(), 0u);
    if (k < cfg_.vocab) {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                         [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        idx.resize(k);
    }

    float maxl = -INFINITY;
    for (uint32_t i : idx) maxl = std::max(maxl, logits[i]);
    double sum = 0;
    for (uint32_t i : idx) { logits[i] = std::exp((logits[i] - maxl) / t); sum += logits[i]; }

    auto& rng = impl_->rng;
    if (!rng) rng = std::make_unique<std::mt19937_64>(sp.seed);
    std::uniform_real_distribution<double> u(0.0, sum);
    double r = u(*rng);
    for (uint32_t i : idx) { r -= logits[i]; if (r <= 0) return i; }
    return idx.empty() ? 0 : idx.back();
}

// ---------------------------------------------------------------------------
// CPU reference forward pass
// ---------------------------------------------------------------------------
// Ground truth for the GPU path, at full fp32 precision straight from the GGUF
// (no int4 requantisation). Weights are dequantised one row at a time out of
// the mapping, so this needs no more memory than the largest single row —
// slow (~1 GB of dequantisation per token) but exact, and it is the only way
// to tell "the decode loop is wrong" from "int4 requantisation is too lossy".
namespace {

struct CpuModel {
    const GGUFFile* f = nullptr;
    DecodeConfig    cfg;
    std::vector<float> kc, vc;   // [layer][pos][kv_dim]
    std::vector<float> row;
    uint32_t threads = std::max(1u, std::thread::hardware_concurrency());

    void init(const DecodeConfig& c) {
        cfg = c;
        const size_t kv = size_t(cfg.n_kv_heads) * cfg.head_dim;
        kc.assign(size_t(cfg.n_layers) * cfg.max_seq * kv, 0.0f);
        vc.assign(kc.size(), 0.0f);
        row.resize(std::max({cfg.hidden, cfg.ffn, uint32_t(cfg.n_heads * cfg.head_dim)}));
    }

    const Tensor& T(const std::string& n) const { return need_tensor(*f, n); }

    // out[j] = dot(in, GGUF row j). GGUF rows ARE output columns.
    //
    // Threaded over output columns. This is a pure speed change with no effect
    // on the arithmetic — every column is an independent dot product over its
    // own dequantised row, accumulated in double, and each thread owns a
    // private RowReader and row buffer. It exists because verifying the
    // sliding-window fix REQUIRES driving this reference past position 512,
    // and a single-threaded pass is ~2.7 s/token: 600 positions would be 45
    // minutes per run. Threads make the deep-position oracle usable rather
    // than theoretical. Output is bit-identical to the serial form (no
    // reduction is split across threads).
    void matvec(const Tensor& t, const float* in, uint32_t rows, uint32_t cols, float* out) {
        const uint32_t nthreads = std::min<uint32_t>(threads, cols);
        if (nthreads <= 1) {
            RowReader rr{&t, rows};
            for (uint32_t j = 0; j < cols; ++j) {
                rr.read(j, row.data());
                double acc = 0;
                for (uint32_t k = 0; k < rows; ++k) acc += double(in[k]) * double(row[k]);
                out[j] = float(acc);
            }
            return;
        }
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (uint32_t w = 0; w < nthreads; ++w) {
            pool.emplace_back([&, w] {
                RowReader rr{&t, rows};
                std::vector<float> r(rows);
                for (uint32_t j = w; j < cols; j += nthreads) {
                    rr.read(j, r.data());
                    double acc = 0;
                    for (uint32_t k = 0; k < rows; ++k) acc += double(in[k]) * double(r[k]);
                    out[j] = float(acc);
                }
            });
        }
        for (auto& th : pool) th.join();
    }

    std::vector<float> norm_w(const std::string& n, uint32_t len) {
        std::vector<float> w = f->dequantize_tensor(T(n));
        w.resize(len);
        return w;
    }

    // Multiplier is (bias + w). A GGUF Gemma file already stores (1 + w), so
    // bias is 0 there; a true HF checkpoint would need 1.
    void rmsnorm(float* out, const float* in, const float* w, uint32_t n, float eps) const {
        const float bias = cfg.gguf_norm_includes_bias ? 0.0f : 1.0f;
        double ss = 0;
        for (uint32_t i = 0; i < n; ++i) ss += double(in[i]) * double(in[i]);
        const float sc = float(1.0 / std::sqrt(ss / n + eps));
        for (uint32_t i = 0; i < n; ++i) out[i] = in[i] * sc * (bias + w[i]);
    }

    static void rope(float* x, uint32_t n_heads, uint32_t hd, uint32_t pos, float theta) {
        const uint32_t half = hd / 2;
        for (uint32_t h = 0; h < n_heads; ++h)
            for (uint32_t d = 0; d < half; ++d) {
                const float inv = std::exp2(-std::log2(theta) * (float(2 * d) / float(hd)));
                const float ang = float(pos) * inv, c = std::cos(ang), s = std::sin(ang);
                float* p = x + h * hd + d;
                const float lo = p[0], hi = p[half];
                p[0] = lo * c - hi * s;
                p[half] = lo * s + hi * c;
            }
    }

    // `want_logits == false` runs the network for its KV-cache side effect only
    // and skips the lm_head matvec. That matvec is 1152x262144 — bigger than
    // the entire rest of the model — so skipping it is what makes advancing
    // this reference to position 512+ affordable. Returns an empty vector then.
    std::vector<float> forward(uint32_t token, uint32_t pos, bool want_logits = true) {
        const uint32_t H = cfg.hidden, FF = cfg.ffn, HD = cfg.head_dim;
        const uint32_t QD = cfg.n_heads * HD, KV = cfg.n_kv_heads * HD;
        const float eps = cfg.rms_eps;

        std::vector<float> x(H), xb(H), xb2(H), q(QD), k(KV), v(KV), att(QD);
        std::vector<float> gate(FF), up(FF), hb(FF);

        RowReader emb{&T("token_embd.weight"), H};
        emb.read(token, x.data());
        const float es = cfg.effective_embed_scale();
        for (float& e : x) e *= es;

        const bool dbg = std::getenv("VULKORE_LLM_DEBUG") != nullptr;
        auto rms = [](const float* p, uint32_t n) {
            double s = 0;
            for (uint32_t i = 0; i < n; ++i) s += double(p[i]) * double(p[i]);
            return std::sqrt(s / n);
        };
        if (dbg) std::printf("\n  tok %u pos %u  embed rms %.4f\n", token, pos, rms(x.data(), H));

        for (uint32_t l = 0; l < cfg.n_layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            rmsnorm(xb.data(), x.data(), norm_w(p + "attn_norm.weight", H).data(), H, eps);
            matvec(T(p + "attn_q.weight"), xb.data(), H, QD, q.data());
            matvec(T(p + "attn_k.weight"), xb.data(), H, KV, k.data());
            matvec(T(p + "attn_v.weight"), xb.data(), H, KV, v.data());

            const auto qn = norm_w(p + "attn_q_norm.weight", HD);
            const auto kn = norm_w(p + "attn_k_norm.weight", HD);
            for (uint32_t h = 0; h < cfg.n_heads; ++h)
                rmsnorm(q.data() + h * HD, q.data() + h * HD, qn.data(), HD, eps);
            for (uint32_t h = 0; h < cfg.n_kv_heads; ++h)
                rmsnorm(k.data() + h * HD, k.data() + h * HD, kn.data(), HD, eps);

            const float theta = cfg.rope_theta(l);
            rope(q.data(), cfg.n_heads, HD, pos, theta);
            rope(k.data(), cfg.n_kv_heads, HD, pos, theta);

            float* kcl = kc.data() + size_t(l) * cfg.max_seq * KV;
            float* vcl = vc.data() + size_t(l) * cfg.max_seq * KV;
            std::copy(k.begin(), k.end(), kcl + size_t(pos) * KV);
            std::copy(v.begin(), v.end(), vcl + size_t(pos) * KV);

            // SLIDING WINDOW — this reference is the ORACLE for the GPU path,
            // so it has to implement the window independently, not mirror the
            // kernel. Gemma 3's local layers (5 of every 6) attend only
            // t in (pos - window, pos]; global layers see everything. Omitting
            // this here would have made --verify compare two identically wrong
            // implementations and report perfect agreement.
            const uint32_t t0 = cfg.attn_kv_start(l, pos + 1);
            const uint32_t S  = pos + 1 - t0;   // attended keys
            const float scale = cfg.effective_attn_scale();
            const uint32_t group = cfg.n_heads / cfg.n_kv_heads;
            std::vector<float> sc(size_t(cfg.n_heads) * S);
            for (uint32_t h = 0; h < cfg.n_heads; ++h) {
                const uint32_t kvh = h / group;
                for (uint32_t j = 0; j < S; ++j) {
                    const size_t t = size_t(t0) + j;
                    double acc = 0;
                    for (uint32_t d = 0; d < HD; ++d)
                        acc += double(q[h * HD + d]) * double(kcl[(t * cfg.n_kv_heads + kvh) * HD + d]);
                    sc[size_t(h) * S + j] = float(acc * scale);
                }
                float* rowp = sc.data() + size_t(h) * S;
                float m = *std::max_element(rowp, rowp + S);
                double sum = 0;
                for (uint32_t j = 0; j < S; ++j) { rowp[j] = std::exp(rowp[j] - m); sum += rowp[j]; }
                for (uint32_t j = 0; j < S; ++j) rowp[j] /= float(sum);
                for (uint32_t d = 0; d < HD; ++d) {
                    double acc = 0;
                    for (uint32_t j = 0; j < S; ++j)
                        acc += double(rowp[j]) * double(vcl[(size_t(t0) + j) * KV + kvh * HD + d]);
                    att[h * HD + d] = float(acc);
                }
            }

            matvec(T(p + "attn_output.weight"), att.data(), QD, H, xb.data());
            rmsnorm(xb2.data(), xb.data(), norm_w(p + "post_attention_norm.weight", H).data(), H, eps);
            for (uint32_t i = 0; i < H; ++i) x[i] += xb2[i];

            rmsnorm(xb.data(), x.data(), norm_w(p + "ffn_norm.weight", H).data(), H, eps);
            matvec(T(p + "ffn_gate.weight"), xb.data(), H, FF, gate.data());
            matvec(T(p + "ffn_up.weight"), xb.data(), H, FF, up.data());
            for (uint32_t i = 0; i < FF; ++i) {
                const float g = gate[i];
                const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
                hb[i] = 0.5f * g * (1.0f + std::tanh(inner)) * up[i];
            }
            matvec(T(p + "ffn_down.weight"), hb.data(), FF, H, xb.data());
            rmsnorm(xb2.data(), xb.data(), norm_w(p + "post_ffw_norm.weight", H).data(), H, eps);
            for (uint32_t i = 0; i < H; ++i) x[i] += xb2[i];
            if (dbg)
                std::printf("  L%-2u q %.3f k %.3f att %.3f ffn %.3f | x %.3f\n", l,
                            rms(q.data(), QD), rms(k.data(), KV), rms(att.data(), QD),
                            rms(hb.data(), FF), rms(x.data(), H));
        }

        if (!want_logits) return {};
        rmsnorm(xb.data(), x.data(), norm_w("output_norm.weight", H).data(), H, eps);
        std::vector<float> logits(cfg.vocab);
        const char* head = f->tensor("output.weight") ? "output.weight" : "token_embd.weight";
        matvec(T(head), xb.data(), H, cfg.vocab, logits.data());
        return logits;
    }
};

}  // namespace

// Generates with the CPU reference. Slow and exact — the yardstick the GPU path
// is judged against.
void cpu_reference_generate(const std::string& model_path, const std::string& prompt,
                            uint32_t n_tokens, uint32_t max_seq) {
    GGUFFile f = GGUFFile::open(model_path);
    DecodeConfig cfg = DecodeConfig::from_model(f.config());
    cfg.max_seq = max_seq;
    Tokenizer tok = Tokenizer::from_gguf(f);

    CpuModel m;
    m.f = &f;
    m.init(cfg);

    const auto ids = tok.encode(prompt, true);
    std::printf("cpu-ref: %zu prompt tokens\n%s", ids.size(), prompt.c_str());
    std::fflush(stdout);

    std::vector<float> logits;
    uint32_t pos = 0;
    for (int32_t id : ids) logits = m.forward(uint32_t(id), pos++);

    for (uint32_t i = 0; i < n_tokens && pos < max_seq; ++i) {
        const uint32_t best =
            uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (int32_t(best) == tok.eos_id()) break;
        std::printf("%s", tok.decode_one(int32_t(best)).c_str());
        std::fflush(stdout);
        logits = m.forward(best, pos++);
    }
    std::printf("\n");
}

std::vector<float> DecodeSession::last_logits() const {
    std::vector<float> v(cfg_.vocab);
    impl_->logits.download(std::span<float>(v));
    return v;
}

void verify_against_cpu(const Paths& paths, const DecodeConfig& cfg,
                        const std::string& prompt, uint32_t n_steps,
                        uint32_t skip, int32_t cpu_window) {
    DecodeSession sess(paths, cfg);
    GGUFFile f = GGUFFile::open(paths.model_gguf);
    DecodeConfig c = sess.config();
    Tokenizer tok = Tokenizer::from_gguf(f);
    CpuModel m;
    m.f = &f;
    m.init(c);
    // Sensitivity control — see the header. Desyncing the two windows on
    // purpose is the only way to show the comparison can see a window at all.
    if (cpu_window >= 0) m.cfg.sliding_window = uint32_t(cpu_window);

    const auto ids32 = tok.encode(prompt, true);
    std::vector<uint32_t> ids(ids32.begin(), ids32.end());
    std::printf("device %s | %zu prompt tokens | skip %u | comparing %u positions\n",
                sess.device_name().c_str(), ids.size(), skip, n_steps + 1);
    std::printf("sliding window: gpu %u | cpu-ref %u%s | %u of every %u layers "
                "(global: (i+1) %% %u == 0) | cpu-ref threads %u\n\n",
                c.sliding_window, m.cfg.sliding_window,
                cpu_window >= 0 ? "  *** CONTROL: windows deliberately DESYNCED ***" : "",
                c.global_every ? c.global_every - 1 : 0, c.global_every,
                c.global_every, m.threads);

    sess.prefill(ids);
    std::vector<float> cpu;
    uint32_t pos = 0;
    // Only the LAST prompt position needs logits — the rest exist to fill the
    // KV cache. Computing all of them meant a 1152x262144 matvec per prompt
    // token for nothing.
    for (size_t i = 0; i < ids.size(); ++i) cpu = m.forward(ids[i], pos++, i + 1 == ids.size());

    // Advance both paths, uncompared, to get PAST the sliding window.
    for (uint32_t i = 0; i < skip; ++i) {
        const double t0 = now_ms();
        const uint32_t next = sess.decode_step();
        cpu = m.forward(next, pos++, i + 1 == skip);
        if (i % 16 == 0 || i + 1 == skip)
            std::printf("  ...advancing, pos %u (%.1f s/token cpu-ref)\r", pos,
                        (now_ms() - t0) / 1000.0);
        std::fflush(stdout);
    }
    if (skip) std::printf("\n\n");

    std::printf("  %-4s %-6s %-10s %-10s %-9s %-9s %s\n", "pos", "span", "corr", "top1 gpu",
                "top1 cpu", "agree", "top5 overlap");
    uint32_t agreements = 0, compared = 0;
    for (uint32_t step = 0; step <= n_steps; ++step) {
        const std::vector<float> gpu = sess.last_logits();

        // Pearson correlation over the whole vocab, plus top-1/top-5 agreement.
        double mg = 0, mc = 0;
        for (uint32_t i = 0; i < c.vocab; ++i) { mg += gpu[i]; mc += cpu[i]; }
        mg /= c.vocab; mc /= c.vocab;
        double sgg = 0, scc = 0, sgc = 0;
        for (uint32_t i = 0; i < c.vocab; ++i) {
            const double a = gpu[i] - mg, b = cpu[i] - mc;
            sgg += a * a; scc += b * b; sgc += a * b;
        }
        const double corr = sgc / std::sqrt(sgg * scc);

        auto topk = [&](const std::vector<float>& v, uint32_t k) {
            std::vector<uint32_t> idx(v.size());
            std::iota(idx.begin(), idx.end(), 0u);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](uint32_t a, uint32_t b) { return v[a] > v[b]; });
            idx.resize(k);
            return idx;
        };
        const auto tg = topk(gpu, 5), tc = topk(cpu, 5);
        uint32_t overlap = 0;
        for (uint32_t a : tg)
            for (uint32_t b : tc) if (a == b) ++overlap;
        const bool agree = tg[0] == tc[0];
        agreements += agree; ++compared;

        // `span` is what a LOCAL layer actually attends at this position. Once
        // it stops tracking pos+1 the window is live, and only then is this
        // table testing anything the old 4-position --verify could not see.
        const uint32_t span = pos - c.attn_kv_start(0, pos);
        std::printf("  %-4u %-6u %-10.6f %-10u %-9u %-9s %u/5\n", pos - 1, span, corr, tg[0],
                    tc[0], agree ? "yes" : "NO", overlap);

        if (step == n_steps) break;
        // Feed BOTH models the GPU's token so the comparison stays on one
        // sequence — otherwise they diverge and we would be comparing logits
        // for different contexts, which measures nothing.
        const uint32_t next = sess.decode_step();
        cpu = m.forward(next, pos++);
    }
    std::printf("\n  top-1 agreement %u/%u\n", agreements, compared);
}

// ---------------------------------------------------------------------------
// Sampling quality: the GPU candidate harvest against an exact CPU top-k
// ---------------------------------------------------------------------------
// The harvest is an APPROXIMATION and the only honest way to accept it is to
// measure what it loses on real logits. Everything below compares against
// std::partial_sort over the same full-vocab logit vector the GPU produced, so
// the ONLY difference under test is the candidate selection.
void sample_check(const Paths& paths, const DecodeConfig& cfg,
                  const std::string& prompt, uint32_t n_steps) {
    DecodeConfig c = cfg;
    c.sampling.mode = SampleMode::Temperature;  // the harvest only runs here
    c.sampling.cpu_full_vocab = false;
    if (!c.sampling.top_k) c.sampling.top_k = 64;

    DecodeSession sess(paths, c);
    if (!sess.candidate_harvest_active())
        throw std::runtime_error("sample-check: the candidate harvest is not active "
                                 "(topk4_partial missing from the sampling module?)");
    const uint32_t k = c.sampling.top_k;
    const uint32_t vocab = sess.config().vocab;

    const auto ids32 = sess.tokenizer().encode(prompt, true);
    std::vector<uint32_t> ids(ids32.begin(), ids32.end());
    sess.prefill(ids);

    std::printf("device %s | vocab %u | k %u | %u chains x 4 = %u candidates "
                "(%.0f KiB)\n\n",
                sess.device_name().c_str(), vocab, k, c.sampling.topk_parts,
                c.sampling.topk_parts * 4, c.sampling.topk_parts * 4 * 8 / 1024.0);
    // Mass is printed as the DEFICIT (1 - captured) in scientific notation:
    // a "mass 1.000000" column would round away exactly the thing being
    // measured.
    std::printf("  %-4s %-11s %-12s %-11s %-12s\n", "step", "hit@k m=4", "1-mass m=4",
                "hit@k m=1", "1-mass m=1");

    double sum_hit4 = 0, sum_mass4 = 0, sum_hit1 = 0, sum_mass1 = 0;
    uint32_t worst_hit4 = k, worst_hit1 = k;
    double worst_mass4 = 1.0, worst_mass1 = 1.0;

    for (uint32_t step = 0; step < n_steps; ++step) {
        const std::vector<float> logits = sess.last_logits();
        const auto cand = sess.last_candidates();

        // Exact top-k, and the softmax probability of each member. The mass is
        // computed over the top-k only, matching what the sampler normalises.
        std::vector<uint32_t> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0u);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        idx.resize(k);
        const float t = c.sampling.temperature > 0 ? c.sampling.temperature : 1.0f;
        double tot = 0;
        std::vector<double> p(k);
        for (uint32_t i = 0; i < k; ++i) {
            p[i] = std::exp((logits[idx[i]] - logits[idx[0]]) / t);
            tot += p[i];
        }
        for (double& x : p) x /= tot;

        // What each harvest recovers. Selecting top-k out of the candidate set
        // is exactly what sample_candidates() does.
        auto score = [&](uint32_t stride) {
            std::vector<std::pair<float, uint32_t>> cs;
            for (size_t i = 0; i < cand.size(); i += stride)
                if (cand[i].second != 0xFFFFFFFFu) cs.push_back(cand[i]);
            const uint32_t kk = std::min<uint32_t>(k, uint32_t(cs.size()));
            std::partial_sort(cs.begin(), cs.begin() + kk, cs.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first > b.first ||
                                         (a.first == b.first && a.second < b.second);
                              });
            cs.resize(kk);
            uint32_t hit = 0;
            double   mass = 0;
            for (uint32_t i = 0; i < k; ++i)
                for (const auto& e : cs)
                    if (e.second == idx[i]) { ++hit; mass += p[i]; break; }
            return std::pair<uint32_t, double>{hit, mass};
        };
        // stride 1 walks every slot (top-4 per chain); stride 4 takes slot 0
        // only, which is precisely `argmax_partial`'s output.
        const auto [h4, m4] = score(1);
        const auto [h1, m1] = score(4);

        std::printf("  %-4u %2u/%-8u %-12.3e %2u/%-8u %-12.3e\n", step, h4, k, 1.0 - m4,
                    h1, k, 1.0 - m1);
        sum_hit4 += h4; sum_mass4 += m4; sum_hit1 += h1; sum_mass1 += m1;
        worst_hit4 = std::min(worst_hit4, h4); worst_mass4 = std::min(worst_mass4, m4);
        worst_hit1 = std::min(worst_hit1, h1); worst_mass1 = std::min(worst_mass1, m1);

        sess.decode_step();
    }

    const double n = n_steps;
    std::printf("\n  m=4 (shipped)               mean %.3f/%u of the true top-k, "
                "worst %u/%u; mean mass deficit %.3e, worst %.3e\n",
                sum_hit4 / n, k, worst_hit4, k, 1.0 - sum_mass4 / n, 1.0 - worst_mass4);
    std::printf("  m=1 (argmax_partial reuse)  mean %.3f/%u of the true top-k, "
                "worst %u/%u; mean mass deficit %.3e, worst %.3e\n",
                sum_hit1 / n, k, worst_hit1, k, 1.0 - sum_mass1 / n, 1.0 - worst_mass1);
}

// ---------------------------------------------------------------------------
// Self-test: this file's packer against the kernel it drives
// ---------------------------------------------------------------------------
// The whole int4 path hinges on the host packer and the kernel agreeing on a
// bit layout, and a disagreement is silent — the model would simply generate
// nonsense. This checks them against each other on random data at the real
// Gemma shapes, including the split-k pairing, and needs neither a model file
// nor any of the transformer kernels.
double matvec_selftest(const MatvecConfig& mv, bool verbose) {
    Context ctx;
    Program prog = Program::from_file(ctx, mv.module);
    Kernel  k    = need_kernel(prog, mv.module, mv.kernel.c_str());
    Kernel  ksp, kred;
    if (mv.split > 1) {
        ksp  = need_kernel(prog, mv.module, mv.split_kernel.c_str());
        kred = need_kernel(prog, mv.module, mv.reduce_kernel.c_str());
    }
    if (verbose)
        std::printf("device %s | layout %s | %s\n\n", ctx.device_name().c_str(),
                    mv.layout == MatvecLayout::Km4Bf16 ? "KM4+bf16"
                        : mv.layout == MatvecLayout::Km4 ? "KM4" : "COL",
                    mv.module.c_str());

    struct S { const char* n; uint32_t rows, cols; };
    // The real per-token shapes, plus one deliberately awkward case.
    const S shapes[] = {
        {"q",       1152, 1024}, {"k/v",  1152,  256},  {"o_proj", 1024, 1152},
        {"gate/up", 1152, 6912}, {"down", 6912, 1152},  {"lm_head", 1152, 8192},
    };

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    double worst = 0.0;

    for (const S& sh : shapes) {
        const uint32_t rows = sh.rows, cols = sh.cols, nblk = rows / kQBlock;

        // Column-major source: col[c*rows + k], matching what pack_group wants.
        std::vector<float> w(size_t(rows) * cols), in(rows);
        for (float& f : w)  f = u(rng);
        for (float& f : in) f = u(rng);

        std::vector<uint32_t> words(size_t(rows) * cols / 8);
        std::vector<float>    scales(size_t(nblk) * cols);
        for (uint32_t g = 0; g < cols / kGroup; ++g)
            pack_group(w.data() + size_t(g) * kGroup * rows, rows, cols, g, mv.layout,
                       words.data(), scales.data());

        // CPU reference: re-derive the quantised values independently of the
        // packing, so a layout bug cannot cancel itself out.
        std::vector<float> ref(cols, 0.0f);
        for (uint32_t j = 0; j < cols; ++j) {
            const float* col = w.data() + size_t(j) * rows;
            double acc = 0.0;
            for (uint32_t b = 0; b < nblk; ++b) {
                const float s = scales[size_t(b) * cols + j];
                const float inv = s > 0.0f ? 1.0f / s : 0.0f;
                float part = 0.0f;
                for (uint32_t t = 0; t < kQBlock; ++t) {
                    const uint32_t kk = b * kQBlock + t;
                    part += in[kk] * (float(quantise(col[kk], inv)) - 8.0f);
                }
                acc += double(part) * s;
            }
            ref[j] = float(acc);
        }

        Buffer bw  = ctx.alloc<uint32_t>(words.size());
        // The bf16 plane is a different element type AND a different index
        // order, so it is built and uploaded through the same helper the real
        // load path uses — otherwise the selftest would be checking a packing
        // no weight ever gets.
        const std::vector<uint32_t> sbits =
            mv.layout == MatvecLayout::Km4Bf16 ? pack_scales_bf16(scales, nblk, cols)
                                               : std::vector<uint32_t>{};
        Buffer bs  = mv.layout == MatvecLayout::Km4Bf16
                         ? ctx.alloc<uint32_t>(sbits.size())
                         : ctx.alloc<float>(scales.size());
        Buffer bi  = ctx.alloc<float>(rows);
        Buffer bo  = ctx.alloc<float>(cols);
        bw.upload(std::span<const uint32_t>(words));
        if (mv.layout == MatvecLayout::Km4Bf16) bs.upload(std::span<const uint32_t>(sbits));
        else                                    bs.upload(std::span<const float>(scales));
        bi.upload(std::span<const float>(in));

        const uint32_t sp = mv.split_for(rows, cols);
        Buffer bp = ctx.alloc<float>(size_t(cols) * std::max(sp, 1u));
        {
            Batch batch(ctx);
            if (sp > 1) {
                batch.add(ksp, {cols * sp}, bp, bi, bw, bs, rows, cols, sp);
                batch.add(kred, {cols}, bo, bp, cols, sp);
            } else {
                batch.add(k, {cols}, bo, bi, bw, bs, rows, cols);
            }
            batch.submit().wait();
        }

        std::vector<float> got(cols);
        bo.download(std::span<float>(got));

        // Judge the error against the RMS of the output vector, not against
        // each element. A dot product of `rows` random terms lands near zero
        // for some columns purely by cancellation, and those columns are
        // relatively "wrong" by 1e-3 while being absolutely fine — the same
        // effect the transformer kernel harness handles with an atol term. A
        // layout disagreement is an O(1) error and cannot hide under this.
        double max_abs = 0.0, sumsq = 0.0;
        uint32_t untouched = 0;
        for (uint32_t j = 0; j < cols; ++j) {
            if (got[j] == 0.0f && ref[j] != 0.0f) ++untouched;
            max_abs = std::max(max_abs, std::fabs(double(got[j]) - double(ref[j])));
            sumsq  += double(ref[j]) * double(ref[j]);
        }
        const double rms = std::sqrt(sumsq / cols);
        const double err = rms > 0 ? max_abs / rms : max_abs;
        worst = std::max(worst, err);
        if (verbose)
            std::printf("  %-8s %5u x %-6u split %-2u  max_abs %.3e  /rms %.3e  %s\n",
                        sh.n, rows, cols, sp, max_abs, err,
                        err < 1e-4 ? "PASS" : "*** FAIL ***");
        if (err >= 1e-4)
            throw std::runtime_error(std::string("decode: matvec self-test FAILED on shape ") +
                                     sh.n + " (err/rms " + std::to_string(err) +
                                     ") — the host packer and the kernel disagree on the "
                                     "weight layout");
        if (untouched == cols)
            throw std::runtime_error("decode: matvec self-test produced an all-zero output — "
                                     "the kernel did not run (cf. the Mandelbulb that compiled, "
                                     "validated, dispatched and did nothing)");
    }
    if (verbose) std::printf("\nall shapes PASS, worst err/rms %.3e\n", worst);
    return worst;
}

}  // namespace vulkore::llm

// ===========================================================================
// CLI
// ===========================================================================
#ifndef VULKORE_LLM_DECODE_NO_MAIN

namespace {
using namespace vulkore::llm;

// ---------------------------------------------------------------------------
// --verify-cache
// ---------------------------------------------------------------------------
// The dedicated test for the packed-weight cache. A cache that loads the wrong
// bytes does not crash — it produces fluent, plausible, wrong text, which is
// this project's worst failure mode and has happened before. So rather than
// spot-checking generated output, this re-packs every tensor from the model and
// compares the cache's payload BYTE FOR BYTE, reporting the first differing
// offset. It needs no GPU and no kernels.
int pack_cache_verify(const Paths& paths, uint32_t nthreads) {
    if (paths.model_gguf.empty()) {
        std::fprintf(stderr, "--verify-cache needs --model\n");
        return 2;
    }
    GGUFFile gguf = GGUFFile::open(paths.model_gguf);
    const DecodeConfig cfg = DecodeConfig::from_model(gguf.config());
    const MatvecLayout layout = paths.matvec.layout;
    const std::vector<PackTask> tasks = build_pack_tasks(gguf, cfg, layout);

    XPackHeader want{};
    if (!expected_header(paths, layout, uint32_t(tasks.size()), want)) {
        std::fprintf(stderr, "--verify-cache: cannot stat %s\n", paths.model_gguf.c_str());
        return 1;
    }
    const std::string path = default_cache_path(paths);
    std::vector<XPackEntry> table;
    const int fd = open_valid_cache(path, want, tasks, table);
    if (fd < 0) {
        std::fprintf(stderr, "--verify-cache: no valid cache at %s (run once to create it)\n",
                     path.c_str());
        return 1;
    }

    std::printf("verifying %s against a fresh pack of %s\n", path.c_str(),
                paths.model_gguf.c_str());
    std::printf("%-34s %10s %10s  %s\n", "tensor", "words B", "scales B", "result");

    const uint32_t nt = nthreads ? nthreads : std::max(1u, std::thread::hardware_concurrency());
    size_t bad = 0;
    uint64_t checked = 0;
    std::vector<uint8_t> disk;
    for (size_t i = 0; i < tasks.size(); ++i) {
        PackedHost fresh;
        pack_one_tensor(tasks[i], layout, nt, fresh);
        const size_t wb = fresh.words_bytes(), sb = fresh.scale_bytes();
        disk.resize(wb + sb);
        if (!read_exact(fd, disk.data(), disk.size())) {
            std::printf("%-34s %10zu %10zu  SHORT READ\n", tasks[i].name.c_str(), wb, sb);
            ++bad;
            break;
        }
        const char* verdict = "ok";
        if (wb != table[i].words_bytes || sb != table[i].scale_bytes) {
            verdict = "SIZE MISMATCH";
            ++bad;
        } else if (std::memcmp(disk.data(), fresh.words.data(), wb) != 0) {
            size_t k = 0;
            while (k < wb && disk[k] == reinterpret_cast<const uint8_t*>(fresh.words.data())[k]) ++k;
            std::printf("%-34s  words differ at byte %zu\n", tasks[i].name.c_str(), k);
            verdict = "WORDS DIFFER";
            ++bad;
        } else if (std::memcmp(disk.data() + wb, fresh.scale_data(), sb) != 0) {
            verdict = "SCALES DIFFER";
            ++bad;
        }
        checked += wb + sb;
        // One line per tensor is 183 lines; print the head, the tail and every
        // failure, so a passing run stays readable and a failing one cannot be
        // missed.
        if (i < 3 || i + 2 >= tasks.size() || std::strcmp(verdict, "ok") != 0)
            std::printf("%-34s %10zu %10zu  %s\n", tasks[i].name.c_str(), wb, sb, verdict);
        else if (i == 3)
            std::printf("%-34s %10s %10s  %s\n", "... (per-tensor, all ok)", "", "", "");
    }
    ::close(fd);
    std::printf("\n%zu tensors, %s compared, %zu mismatch(es)\n", tasks.size(),
                human_bytes(checked).c_str(), bad);
    std::printf("%s\n", bad ? "FAIL — the cache does NOT match a fresh pack"
                            : "PASS — cached bytes are identical to a fresh pack");
    return bad ? 1 : 0;
}

int usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [--dry-run] [--selftest] [--model FILE] [--prompt TEXT] [-n N]\n"
        "          [--chat] [--matvec-spv F] [--transformer-spv F]\n"
        "          [--matvec-layout km4bf16|km4|col] [--split N] [--max-seq N]\n"
        "          [--temp T] [--top-k K] [--top-p P] [--seed S]\n"
        "          [--bandwidth GBS] [--dispatch-ms MS] [--submit-ms MS]\n\n"
        "  --dry-run   report the dispatch plan, KV cache cost and the projected\n"
        "              ms/token from the cost model. Needs NO GPU, NO kernels and\n"
        "              NO model file.\n"
        "  --selftest  check this file's int4 packer against the matvec kernel it\n"
        "              drives, at the real shapes. Needs a GPU and the matvec .spv\n"
        "              only — no model, no transformer kernels.\n"
        "  --cpu-ref   generate with the fp32 CPU reference instead of the GPU.\n"
        "  --verify    drive both paths over one token sequence and report how\n"
        "              far the logits diverge.\n"
        "  --verify-steps N / --verify-skip N   compare N+1 positions after\n"
        "              advancing N uncompared ones. A sliding-window bug is\n"
        "              INVISIBLE below --window positions, so a deep run needs\n"
        "              --verify-skip well past it.\n"
        "  --prompt-file F  read the prompt from a file (long-context tests).\n"
        "  --window N  sliding-window extent; 0 disables (all layers global).\n"
        "  --verify-cpu-window N   CONTROL: give the CPU reference a DIFFERENT\n"
        "              window from the GPU. Correlation must collapse; if it does\n"
        "              not, --verify is blind to the window and proves nothing.\n"
        "  --chat      wrap the prompt in Gemma's <start_of_turn> template.\n"
        "  --sample-check  score the GPU candidate harvest against an exact CPU\n"
        "              top-k over the same logits (hit@k and probability mass).\n"
        "  --sample-cpu    force the old full-vocab download + host nth_element.\n"
        "  --topk-parts N  chains for the harvest (default 4096, 4 candidates each).\n"
        "  --pack-threads N  packer threads (default hardware_concurrency).\n"
        "  --pack-cache F  path for the packed-weight cache (default <model>.xpack).\n"
        "  --no-pack-cache  neither read nor write the packed-weight cache.\n"
        "  --repack        ignore any existing cache and rewrite it.\n"
        "  --verify-cache  re-pack every tensor from the model and compare the\n"
        "              cache's payload byte for byte. Needs NO GPU.\n"
        "  --curve     print median gpu ms/token bucketed by KV position plus the\n"
        "              deepest/shallowest RATIO, which is the metric that matters\n"
        "              for consistent throughput. Generates past EOS to reach the\n"
        "              deep buckets.\n"
        "  --no-parallel-softmax / --no-scores-mq4   fall back to the serial\n"
        "              softmax_rows / the generic attention_scores+apply. Together\n"
        "              they reproduce the pre-2026-07-19 attention exactly, which\n"
        "              is how the flatness A/B is run from ONE binary.\n"
        "  --attn-min-span N   span below which the serial attention kernels are\n"
        "              kept (default 128; the parallel forms are SLOWER below it).\n"
        "  --attn-apply-split N   span split for attention_apply_mq4 (default 16).\n"
        "  --global-every N   layer i is global iff (i+1) %% N == 0. A value above\n"
        "              the layer count makes EVERY layer local, which is wrong for\n"
        "              the model but is the THERMAL CONTROL for --curve: nothing\n"
        "              then depends on position, so any remaining growth is drift.\n",
        argv0);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace vulkore::llm;
    Paths        paths;
    DecodeConfig cfg = DecodeConfig::gemma3_1b();
    Machine      machine;
    bool         dry = false, selftest = false, chat = false, cpu_ref = false, verify = false;
    bool         samp_check = false, verify_cache = false, curve = false;
    bool         machine_overridden = false;
    uint32_t     verify_steps = 4;
    uint32_t     verify_skip  = 0;
    int32_t      verify_cpu_window = -1;
    std::string  prompt = "The capital of France is";
    uint32_t     n_tokens = 32;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--dry-run")         dry = true;
        else if (a == "--selftest")        selftest = true;
        else if (a == "--cpu-ref")         cpu_ref = true;
        else if (a == "--verify")          verify = true;
        else if (a == "--sample-check")    samp_check = true;
        else if (a == "--curve")           curve = true;
        else if (a == "--no-parallel-softmax") cfg.parallel_softmax = false;
        else if (a == "--no-scores-mq4")       cfg.scores_mq4 = false;
        else if (a == "--softmax-parts")   cfg.softmax_max_parts = std::stoul(next());
        else if (a == "--attn-min-span")  cfg.attn_min_span = std::stoul(next());
        else if (a == "--attn-apply-split") cfg.attn_apply_split = std::stoul(next());
        else if (a == "--sample-cpu")      cfg.sampling.cpu_full_vocab = true;
        else if (a == "--topk-parts")      cfg.sampling.topk_parts = std::stoul(next());
        else if (a == "--chat")            chat = true;
        else if (a == "--model")           paths.model_gguf = next();
        else if (a == "--matvec-spv")      paths.matvec.module = next();
        else if (a == "--transformer-spv") paths.transformer_spv = next();
        else if (a == "--split")           paths.matvec.split = std::stoul(next());
        else if (a == "--matvec-layout") {
            const std::string v = next();
            if      (v == "col") paths.matvec = MatvecConfig::baseline();
            else if (v == "km4") {
                // fp32-scale KM4: the A-side of the bf16-scale comparison.
                paths.matvec.layout        = MatvecLayout::Km4;
                paths.matvec.kernel        = "mv_km4";
                paths.matvec.split_kernel  = "mv_km4_split";
            } else if (v != "km4bf16") {
                std::fprintf(stderr, "unknown layout '%s'\n", v.c_str());
                return 2;
            }
        }
        else if (a == "--verify-steps")    verify_steps = std::stoul(next());
        else if (a == "--verify-skip")     verify_skip = std::stoul(next());
        else if (a == "--verify-cpu-window") verify_cpu_window = std::stol(next());
        else if (a == "--window")          cfg.sliding_window = std::stoul(next());
        // THERMAL CONTROL. layer_is_global() is (i+1) % global_every == 0, so a
        // value above the layer count makes EVERY layer local and therefore
        // position-independent past the window. That is numerically wrong for
        // the model, but as a timing control it is exactly right: any ms/token
        // growth left in a --curve run under it is drift, not depth.
        else if (a == "--global-every")    cfg.global_every = std::stoul(next());
        else if (a == "--pack-cache")      paths.pack_cache = next();
        else if (a == "--no-pack-cache")   paths.use_pack_cache = false;
        else if (a == "--repack")          paths.force_repack = true;
        else if (a == "--pack-threads")    paths.pack_threads = std::stoul(next());
        else if (a == "--verify-cache")    verify_cache = true;
        else if (a == "--prompt")          prompt = next();
        else if (a == "--prompt-file") {
            const std::string path = next();
            std::FILE* fp = std::fopen(path.c_str(), "rb");
            if (!fp) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
            std::string s;
            char buf[4096];
            size_t got;
            while ((got = std::fread(buf, 1, sizeof buf, fp)) > 0) s.append(buf, got);
            std::fclose(fp);
            prompt = s;
        }
        else if (a == "-n")                n_tokens = std::stoul(next());
        else if (a == "--max-seq")         cfg.max_seq = std::stoul(next());
        else if (a == "--temp") { cfg.sampling.temperature = std::stof(next());
                                  cfg.sampling.mode = SampleMode::Temperature; }
        else if (a == "--top-p") { cfg.sampling.top_p = std::stof(next());
                                  cfg.sampling.mode = SampleMode::Temperature; }
        else if (a == "--top-k") { cfg.sampling.top_k = std::stoul(next());
                                   cfg.sampling.mode = SampleMode::Temperature; }
        else if (a == "--seed")            cfg.sampling.seed = std::stoull(next());
        else if (a == "--bandwidth") { machine.bandwidth_gbs = std::stod(next());
                                       machine_overridden = true;
                                       machine.name = "user-supplied"; }
        else if (a == "--dispatch-ms")     machine.dispatch_ms = std::stod(next());
        else if (a == "--submit-ms")       machine.submit_ms = std::stod(next());
        else return usage(argv[0]);
    }

    if (!machine_overridden) {
        const Machine d = Machine::for_matvec(paths.matvec);
        machine.name = d.name;
        machine.bandwidth_gbs = d.bandwidth_gbs;
    }

    if (verify_cache) {
        try {
            return pack_cache_verify(paths, paths.pack_threads);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "\nerror: %s\n", e.what());
            return 1;
        }
    }

    if (samp_check) {
        try {
            if (chat)
                prompt = "<start_of_turn>user\n" + prompt +
                         "<end_of_turn>\n<start_of_turn>model\n";
            sample_check(paths, cfg, prompt, std::max(n_tokens, 1u));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "\nerror: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    if (verify) {
        try {
            // 4 positions was too small a sample to judge a weight-format
            // change on: a greedy top-1 pick at a near-tie is a coin flip, and
            // one flip out of five reads as a regression whether or not it is
            // one. --verify-steps buys more evidence at the cost of CPU time.
            verify_against_cpu(paths, cfg, prompt, verify_steps, verify_skip,
                               verify_cpu_window);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "\nerror: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    if (cpu_ref) {
        try {
            if (chat)
                prompt = "<start_of_turn>user\n" + prompt +
                         "<end_of_turn>\n<start_of_turn>model\n";
            cpu_reference_generate(paths.model_gguf, prompt, n_tokens, cfg.max_seq);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "\nerror: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    if (dry) {
        std::fputs(dry_run_report(cfg, machine, paths.matvec).c_str(), stdout);
        return 0;
    }
    if (selftest) {
        try {
            matvec_selftest(paths.matvec);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "\nerror: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    // Gemma 3 IT expects its turn template; a bare prompt makes an
    // instruction-tuned model ramble. add_space_prefix is false in this file's
    // metadata, and the tokenizer already honours that.
    if (chat)
        prompt = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n";

    try {
        const double t0 = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        DecodeSession sess(paths, cfg);
        const double load_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count() - t0;

        std::printf("device      %s\n", sess.device_name().c_str());
        std::printf("weights     %.1f MiB packed int4+scales (loaded in %.2f s)\n",
                    double(sess.weight_bytes()) / (1024.0 * 1024.0), load_s);
        {
            const LoadStats& ls = sess.load_stats();
            if (ls.from_cache)
                std::printf("load path   cache hit in %.2f s (%.2f read + %.2f verify + "
                            "%.2f upload) %s\n",
                            ls.cache_read_ms / 1e3, ls.cache_io_ms / 1e3,
                            ls.cache_hash_ms / 1e3, ls.upload_ms / 1e3, ls.cache_path.c_str());
            else
                std::printf("load path   packed on %u threads, %.2f s pack + %.2f s upload"
                            "%s\n",
                            ls.threads, ls.pack_ms / 1e3, ls.upload_ms / 1e3,
                            ls.cache_written ? " (cache written)" : "");
        }
        std::printf("kv cache    %.1f MiB (%u positions, allocated in %.1f ms)\n",
                    double(sess.kv_bytes()) / (1024.0 * 1024.0), sess.config().max_seq,
                    sess.kv_alloc_ms());
        std::printf("dispatches  %u per token, 1 submit\n\n", sess.plan().total);

        const auto ids32 = sess.tokenizer().encode(prompt, /*add_bos=*/true);
        std::vector<uint32_t> ids(ids32.begin(), ids32.end());
        sess.prefill(ids);
        std::printf("prefill     %zu tokens in %.1f ms (%.1f tok/s)\n",
                    ids.size(), sess.last_metrics().ms, sess.last_metrics().tokens_per_sec);

        std::printf("\n%s", prompt.c_str());
        std::fflush(stdout);
        // Gemma IT ends a turn with <end_of_turn>, not <eos>; stopping only on
        // eos_id lets it run straight past the answer into the next turn.
        const int32_t eot = sess.tokenizer().id_for("<end_of_turn>");
        double total_ms = 0;
        uint32_t emitted = 0;
        // Per-step samples, not just the last one. A phone GPU shares memory
        // with the rest of the system and drifts thermally, so a single step is
        // not a measurement: matvec-optimisation.md caught the same shape
        // reading 2.66 and 2.01 ms in consecutive runs. Report min AND mean so
        // a regression cannot hide behind either.
        std::vector<double> gpu_ms, rec_ms; std::vector<double> smp_ms;
        std::vector<uint32_t> pos_of;   // --curve: KV position of each sample
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t tok = sess.decode_step();
            // --curve sweeps the depth axis and must reach the deep buckets, so
            // it keeps generating past EOS. Timing is a function of position,
            // not of content; the text is not the artefact here.
            if (!curve && (int32_t(tok) == sess.tokenizer().eos_id() ||
                           int32_t(tok) == eot)) break;
            std::printf("%s", sess.tokenizer().decode_one(int32_t(tok)).c_str());
            std::fflush(stdout);
            total_ms += sess.last_metrics().ms;
            gpu_ms.push_back(sess.last_metrics().gpu_ms);
            rec_ms.push_back(sess.last_metrics().record_ms);
            smp_ms.push_back(sess.last_metrics().sample_ms);
            pos_of.push_back(sess.last_metrics().position);
            ++emitted;
        }
        std::printf("\n\ndecode      %u tokens, %.1f ms/token, %.2f tok/s\n",
                    emitted, emitted ? total_ms / emitted : 0.0,
                    total_ms > 0 ? emitted * 1000.0 / total_ms : 0.0);
        const Metrics m = sess.last_metrics();
        std::printf("last step   record %.2f ms | gpu %.2f ms | sample %.2f ms\n",
                    m.record_ms, m.gpu_ms, m.sample_ms);
        if (!gpu_ms.empty()) {
            auto stats = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                double sum = 0; for (double x : v) sum += x;
                return std::array<double, 3>{v.front(), v[v.size() / 2], sum / double(v.size())};
            };
            const auto g = stats(gpu_ms), r = stats(rec_ms), sm = stats(smp_ms);
            std::printf("gpu ms      min %.2f | median %.2f | mean %.2f   (n=%zu)\n",
                        g[0], g[1], g[2], gpu_ms.size());
            std::printf("record ms   min %.2f | median %.2f | mean %.2f\n", r[0], r[1], r[2]);
            std::printf("sample ms   min %.2f | median %.2f | mean %.2f\n", sm[0], sm[1], sm[2]);
        }
        // THE FLATNESS METRIC. A single median over a long generation hides the
        // whole problem: it averages a cheap shallow token with an expensive
        // deep one. What matters for "consistent tps" is the RATIO between the
        // shallowest and deepest bucket, so print the curve and that ratio.
        if (curve && !gpu_ms.empty()) {
            static const uint32_t kEdges[] = {0, 128, 512, 1024, 2048, 4096, 8192,
                                              16384, 0xFFFFFFFFu};
            std::printf("\n--- depth curve (median gpu ms by KV position) ---\n");
            std::printf("%14s %8s %12s %10s\n", "position", "n", "gpu ms", "tok/s");
            double shallow = 0, deep = 0;
            for (size_t e = 0; e + 1 < sizeof(kEdges) / sizeof(kEdges[0]); ++e) {
                std::vector<double> v;
                for (size_t i = 0; i < gpu_ms.size(); ++i)
                    if (pos_of[i] >= kEdges[e] && pos_of[i] < kEdges[e + 1])
                        v.push_back(gpu_ms[i]);
                if (v.empty()) continue;
                std::sort(v.begin(), v.end());
                const double med = v[v.size() / 2];
                std::printf("%7u-%-6u %8zu %12.2f %10.1f\n", kEdges[e],
                            kEdges[e + 1] == 0xFFFFFFFFu ? 0 : kEdges[e + 1] - 1,
                            v.size(), med, 1000.0 / med);
                if (shallow == 0) shallow = med;
                deep = med;
            }
            if (shallow > 0)
                std::printf("\nFLATNESS  deepest/shallowest = %.2fx   "
                            "(1.00x is the goal; lower is flatter)\n", deep / shallow);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
    return 0;
}

#endif  // VULKORE_LLM_DECODE_NO_MAIN
