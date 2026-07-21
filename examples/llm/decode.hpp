// Host-side decode loop + KV cache for Gemma 3 1B on Vulkore.
//
// This is Track D of agent-docs/llm-on-vulkore.md: the integration layer that
// strings kernels/llm.cl's matvecs and kernels/llm_transformer.cl's small ops
// together into an actual token. It owns the Context, the Programs, every
// weight buffer, the KV cache, and the sampling step.
//
// ===========================================================================
// THE ONE DESIGN RULE: one token == ONE vkQueueSubmit.
// ===========================================================================
// A trivial kernel costs a measured 0.31 ms per `vulkore::launch` on an Adreno
// 840 no matter how little work it does, because launch() does one submit and
// one fence each. A Gemma 3 1B token issues 838 dispatches here. Per-launch
// submission would therefore cost 260 ms/token of pure overhead before a
// single weight byte is read. Everything below is recorded into ONE
// `vulkore::Batch` and submitted once; inside a Batch a small dispatch costs
// ~0.006 ms (measured, agent-docs/llm-transformer-kernels.md).
//
// Two corollaries that shaped the design and are easy to get wrong:
//
//   * A host->device upload is ALSO a submit (Adreno DeviceLocal is not
//     host-visible, so uploads take the staging path: one_shot + wait). Any
//     per-token upload therefore costs more than the dispatch it saves. This
//     is why RoPE uses the `rope` kernel (computes sin/cos inline from a `pos`
//     push constant) rather than `rope_cached` (needs two 128-float tables
//     uploaded per theta per token), and why the token embedding is written
//     straight into a HostVisible residual buffer by memcpy instead of via an
//     `embed_lookup` dispatch over a 1.2 GB fp32 table.
//   * `Fence`'s destructor DRAINS. Discarding the Fence from a launch
//     serialises the queue (pocket-trainer-design.md §5.3). There is exactly
//     one Fence per token here and it is held, then waited.
//
// ===========================================================================
// KERNEL CONTRACT
// ===========================================================================
// Kernels are looked up by name at construction; a missing one throws with the
// kernel name and the .spv path in the message. Grid is the GLOBAL thread
// count (vulkore rounds up to whole workgroups), so every kernel bound-checks.
//
// From the matvec module (default kernels/llm_matvec.spv, KM4 layout):
//
//   mv_km4      (out, in, global const uint4* wq, scale, rows, cols)  grid: cols
//   mv_km4_split(partial, in, wq, scale, rows, cols, split)     grid: cols*split
//   mv_reduce   (out, partial, cols, split)                           grid: cols
//     out[j] = sum_k in[k] * (nibble(k,j) - 8) * scale[(k/32)*cols + j].
//     KM4 packing: uint4[(g*nblk + b)*64 + t] with g = j/64, t = j%64,
//     b = k/32; component (k>>3)&3, nibble (k&7)*4. `rows` a multiple of 32,
//     `cols` a multiple of 64. Split-k is used when cols < 2048.
//     `MatvecConfig::baseline()` selects the older COL layout instead
//     (`matvec_q4` from kernels/llm.spv, word[(k*cols+j)>>3], nibble (j&7)*4).
//     Scales are [block][col] in BOTH layouts.
//
// From the transformer module (default kernels/llm_transformer.spv):
//
//   rmsnorm_sumsq(global float* partial, global const float* in,
//                 uint n, uint nparts)                    grid: nparts
//   rmsnorm_apply_gemma(global float* out, global const float* in,
//                       global const float* w, global const float* partial,
//                       uint n, uint nparts, float eps)   grid: n
//     Two-pass RMSNorm with Gemma's (1 + w) weight convention. Measured 6.4x
//     faster than the one-pass form inside a Batch.
//   rmsnorm_heads_gemma(global float* out, global const float* in,
//                       global const float* w, uint n_heads, uint head_dim,
//                       float eps)                        grid: n_heads*head_dim
//     Gemma 3 QK-norm: independent RMSNorm per head, weight shared.
//   rope(global float* x, uint n_heads, uint head_dim, uint pos, float theta)
//                                                         grid: n_heads*head_dim/2
//     In-place ROTATE-HALF (pairs d with d + head_dim/2), one thread per pair.
//   geglu(global float* out, global const float* gate, global const float* up,
//         uint n)                                         grid: n
//   attention_scores(global float* scores, global const float* q,
//                    global const float* kcache, uint n_heads, uint kv_heads,
//                    uint head_dim, uint span, float scale, uint kv_start)
//                                                         grid: n_heads*span
//   softmax_rows(global float* x, uint n_rows, uint n_cols)  grid: n_rows
//   attention_apply(global float* out, global const float* probs,
//                   global const float* vcache, uint n_heads, uint kv_heads,
//                   uint head_dim, uint span, uint kv_start)
//                                                         grid: n_heads*head_dim
//     SLIDING WINDOW. The attended keys are cache positions
//     [kv_start, kv_start + span), and `scores`/`probs` have row stride `span`.
//     kv_start = 0, span = seq_len is the unwindowed form and is bit-identical
//     to the pre-window signature.
//
//     Gemma 3 alternates FIVE sliding-window layers (extent 512, published as
//     gemma3.attention.sliding_window) to ONE global layer: layer i is global
//     iff (i + 1) % 6 == 0. Ignoring that is not a small error — past 512
//     positions, 22 of 26 layers attend to keys the model was never trained to
//     see. It is expressed as an OFFSET rather than a -INFINITY mask because a
//     decode query is always the newest position, so its window is a
//     contiguous SUFFIX of the cache; that makes the correct version the
//     CHEAPER one, which a mask would not have been.
//
//     `DecodeConfig::attn_kv_start(layer, seq_len)` computes kv_start; the CPU
//     reference implements the same rule independently, because an oracle that
//     shares the loop's mistake is not an oracle.
//   add_residual(global float* x, global const float* y, uint n)  grid: n
//   kv_append(global float* cache, global const float* src, uint n, uint pos)
//                                                         grid: n
//   argmax(global float* out, global const float* logits, uint n)  grid: 1
//     out[0] = index as float, out[1] = value. FALLBACK ONLY — one thread over
//     262144 logits measures 31.5 ms, over half a token. Preferred is the tree
//     below from kernels/llm_sampling.spv, same output ABI:
//
//   argmax_partial(pval, uint* pidx, logits, n, nparts)        grid: nparts
//   argmax_reduce (oval, uint* oidx, pval, pidx, n, nparts)    grid: nparts
//   argmax_final  (out, uint* pidx_out, pval, pidx, nparts)    grid: 1
//     vocab -> 4096 -> 64 -> 1. Three dispatches, ~0.02 ms total.
//
// Deliberately NOT used, though the module provides them: `embed_lookup`
// (needs a 1.2 GB fp32 table for a 262144x1152 vocab — done on the CPU from
// the mmap'd GGUF instead), `rope_cached` (see above), `mul_scalar` (Gemma's
// sqrt(hidden) embedding scale is folded into the CPU-side embedding write),
// `rmsnorm`/`rmsnorm_gemma` one-pass forms, `swiglu`, `add_vec`,
// `geglu_fused`.
//
// ===========================================================================
// LAYOUT CONTRACTS
// ===========================================================================
//   KV cache      k[(t*kv_heads + h)*head_dim + d], one K and one V buffer per
//                 layer, capacity max_seq positions. Appending is a contiguous
//                 write at the end, which is what kv_append does.
//   scores        [q_head][seq_len], row stride == seq_len (rebuilt each step).
//   weights       see matvec_q4 above. GGUF stores linear weights as
//                 [out][in] row-major (shape[0]==in is the fastest axis); the
//                 kernel wants [in][out], so packing TRANSPOSES.
//
// Q/K/V are three separate matvecs rather than one fused 1152x1536, and
// gate/up likewise, because a descriptor binding cannot carry a byte offset:
// `rmsnorm_heads_gemma` and `geglu` need their inputs at the start of a
// buffer. The weight bytes moved are identical either way; it costs 3 extra
// dispatches per layer (~0.5 ms/token), which a `split3` kernel would recover.
#pragma once

#include <vulkore/vulkore.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "model/gguf.hpp"
#include "model/tokenizer.hpp"

namespace vulkore::llm {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum class SampleMode {
    Greedy,       // GPU argmax; downloads 2 floats
    Temperature,  // temperature + top-k
};

struct Sampling {
    SampleMode mode        = SampleMode::Greedy;
    float      temperature = 1.0f;
    uint32_t   top_k       = 0;  // 0 == disabled (full distribution)
    // Nucleus sampling: keep the smallest set of candidates whose probabilities
    // sum to `top_p`, and discard the rest BEFORE drawing. 1.0 disables it.
    //
    // This is the setting that governs confident-sounding nonsense. top-k alone
    // keeps a fixed 64 candidates no matter how sure the model is, so a token
    // the model gives 0.3% to still gets drawn 0.3% of the time. Under top-p a
    // confident step collapses to one or two candidates and an genuinely
    // uncertain one stays wide. Gemma's own recommended decoding is
    // temp 1.0 / top-k 64 / top-p 0.95; we shipped the first two and not this,
    // which is why output was needlessly creative.
    float      top_p       = 1.0f;
    uint64_t   seed        = 1234;

    // How Temperature gets its candidate set.
    //
    //   false (default) — GPU CANDIDATE HARVEST. `topk4_partial` keeps the best
    //     4 of each of `topk_parts` strided chains inside the token's own Batch
    //     (one extra dispatch, no extra submit) and the host finishes top-k +
    //     softmax + multinomial over those 4*topk_parts candidates, read out of
    //     HOST-VISIBLE memory. 128 KiB of mapped reads instead of a 1 MiB
    //     staged download, and ~16k host comparisons instead of ~262k.
    //
    //   true — the original path: download all `vocab` logits (a staging copy,
    //     i.e. an EXTRA vkQueueSubmit) and run nth_element over the whole
    //     vocabulary. Exact, and kept so the fast path can be A/B'd against it
    //     (`--sample-cpu`, and `--sample-check` which runs both).
    bool cpu_full_vocab = false;

    // Chains for the candidate harvest; candidates = 4 * topk_parts.
    //
    // 1024, NOT the 4096 the argmax tree uses. The GPU pass is memory-bound and
    // roughly indifferent between them, but the host cost is linear in the
    // candidate count and it dominates: measured on an Adreno 840, mean
    // sample_ms is 0.94 at 4096 chains, 0.42 at 2048, 0.15 at 1024, 0.05 at
    // 512. Quality does not pay for it — 4 candidates per chain recovered the
    // exact top-64 on every one of 60 measured tokens at BOTH 1024 and 512
    // (`--sample-check`), because losing a true top-k member needs FIVE of them
    // in one chain. 1024 is picked over 512 for margin, not for a measured
    // difference: the 5-collision probability is ~7e-6 against ~1.1e-4.
    uint32_t topk_parts = 1024;
};

struct DecodeConfig {
    uint32_t n_layers   = 26;
    uint32_t hidden     = 1152;
    uint32_t ffn        = 6912;
    uint32_t n_heads    = 4;
    uint32_t n_kv_heads = 1;
    uint32_t head_dim   = 256;
    uint32_t vocab      = 262144;

    // KV cache capacity in tokens. Memory is
    //   2 * n_layers * max_seq * n_kv_heads * head_dim * 4 bytes
    // = 53,248 B per position for Gemma 3 1B (MQA: ONE kv head, which is the
    // only reason a phone context is affordable). See kv_cache_bytes().
    //
    // 8192, raised from 2048 once the sliding window made deep context correct.
    // The number is measured, not chosen because it fits. All figures below are
    // an Adreno 840 (OnePlus 15, Qualcomm driver), taskset to the prime core.
    //
    //   CAPACITY IS ALMOST FREE IN TIME. Nothing in the decode loop scans
    //   max_seq — only the live position — and a sweep confirms it: with a
    //   shallow prompt, median gpu ms/token is 12.90 / 13.54 / 14.79 / 13.20 /
    //   13.32 / 12.60 at max_seq 2048 / 4096 / 8192 / 16384 / 32768 / 65536,
    //   i.e. FLAT inside run-to-run noise across a 32x range. Allocation is
    //   0.2 / 14.3 / 18.7 / 52.2 / 130.7 / 359.1 ms against a 13.8 s weight
    //   load. Even 65536 (3.3 GiB) allocates cleanly on this 16 GB phone.
    //
    //   WHAT COSTS IS DEPTH, and that is independent of capacity. Windowed
    //   decode measures 12.9 ms/token at position ~40, 30.9 at 4176 and 62.6 at
    //   8288 — very close to linear, because the 4 GLOBAL layers still scale
    //   with position even though the other 22 no longer do.
    //
    // So the limit is set by where throughput stops being usable, not by what
    // fits: 8192 costs 416 MiB and still runs 16 tok/s at FULL depth (above
    // reading speed), with 952 MiB total resident alongside the weights.
    // 16384 would be ~9 tok/s at full depth and 32768 ~5 tok/s — offering
    // context that cannot be used interactively. Both work and are correct;
    // pass --max-seq if you want them.
    //
    // The way to earn a larger default is to fix the deep-context cost, and it
    // is NOT bandwidth: going from position ~40 to 4176 adds 56.5 MiB of KV
    // reads and 18 ms, an implied 3.1 GB/s against this device's measured
    // 62.7-70.1 GB/s ceiling. It is serialisation — `softmax_rows` is ONE
    // THREAD PER ROW over a `span`-wide row and there are only 4 rows, and
    // `attention_apply` runs 1024 threads each walking the whole span. Same
    // shape as the single-thread argmax that turned out to be 58% of a token.
    uint32_t max_seq = 8192;

    float rms_eps           = 1e-6f;
    float rope_theta_global = 1e6f;
    float rope_theta_local  = 1e4f;
    // Gemma 3 alternates 5 sliding-window layers to 1 global. Layer i is
    // global iff (i + 1) % global_every == 0.
    uint32_t global_every    = 6;
    uint32_t sliding_window  = 512;

    // 0 => sqrt(hidden) / 1 => 1/sqrt(head_dim), the Gemma defaults.
    float embed_scale = 0.0f;
    float attn_scale  = 0.0f;

    // Strided partial count for the two-pass RMSNorm.
    uint32_t norm_parts = 64;
    // Tree widths for the parallel argmax: vocab -> parts -> parts2 -> 1.
    // 4096/64 is the configuration measured best in llm-sampling-kernels.md.
    uint32_t argmax_parts  = 4096;
    uint32_t argmax_parts2 = 64;
    // Three tree dispatches instead of one serial scan. Set false to plan for
    // the fallback. DecodeSession overwrites this with what it actually got.
    bool     parallel_argmax = true;

    // ---- PARALLEL ATTENTION (the flatness fix) ----------------------------
    // `softmax_rows` is one thread per row and there are only n_heads rows, so
    // its cost is pure exposed memory latency and grows linearly with span. It
    // was the largest depth-dependent term in the token. The replacement is the
    // same two-pass-through-global-memory shape as the parallel argmax tree.
    bool     parallel_softmax = true;
    // One thread per key position accumulating all 4 heads, valid only for
    // Gemma's multi-query shape (n_heads 4, kv_heads 1). Reads each key once
    // instead of once per head and uses float4 loads.
    bool     scores_mq4 = true;
    // Below this span every parallel form is SLOWER on an Adreno 840 — the
    // extra dispatches (~5.6 us each) cost more than the serial work they
    // remove. Measured at span 41: scores 0.67x, softmax 0.77x, apply 0.92x.
    uint32_t attn_min_span = 128;
    // Span split for attention_apply_mq4. Sharing v across the 4 heads alone
    // would drop the grid from 1024 to 256 threads; splitting the span 16 ways
    // puts it back to 4096. 16 measured best at every span >= 512.
    uint32_t attn_apply_split = 16;
    // Upper bound on nparts, and therefore on the scratch buffers.
    uint32_t softmax_max_parts = 256;

    // nparts wants to be ~sqrt(span): pass 1 does span/nparts iterations on
    // n_heads*nparts threads while pass 2 does nparts iterations on n_heads
    // threads, so raising nparts past sqrt just moves the serial scan from the
    // first dispatch into the second. Rounded to a power of two.
    uint32_t softmax_parts_for(uint32_t span) const {
        if (!parallel_softmax || span < attn_min_span) return 1;
        uint32_t p = 1;
        while (p * p < span && p < softmax_max_parts) p *= 2;
        return (p > span) ? span : p;
    }

    // THE BUG THAT COST THE MOST TIME. Gemma stores RMSNorm weights as an
    // offset from identity and HF applies `x_hat * (1 + w)`, which is why the
    // kernels are named *_gemma and hardcode the (1 + w). But llama.cpp's
    // converter (convert_hf_to_gguf.py, Gemma*Model.modify_tensors) does
    // `data_torch + 1` on every tensor ending in `norm.weight` at CONVERSION
    // time, so a GGUF file already holds (1 + w) and the kernel adds a second
    // one. The symptom is not a crash: the model runs, the residual stream
    // blows up ~55x in layer 0, and it emits fluent-looking garbage.
    //
    // Fixed host-side by subtracting 1 at load, so the kernel's (1 + w)
    // reproduces the file's multiplier exactly and no kernel changes.
    // Set false for a checkpoint that stores true HF offsets.
    bool gguf_norm_includes_bias = true;

    Sampling sampling{};

    static DecodeConfig gemma3_1b();
    // Fills shape fields from GGUF metadata, keeps the rest.
    static DecodeConfig from_model(const ModelConfig& mc);

    float effective_embed_scale() const;
    float effective_attn_scale() const;
    bool  layer_is_global(uint32_t layer) const {
        return global_every == 0 || (layer + 1) % global_every == 0;
    }
    float rope_theta(uint32_t layer) const {
        return layer_is_global(layer) ? rope_theta_global : rope_theta_local;
    }
    // Window in effect on `layer`; 0 == unlimited (a global layer).
    uint32_t layer_window(uint32_t layer) const {
        return layer_is_global(layer) ? 0u : sliding_window;
    }
    // First cache position a decode query at seq_len-1 may attend on `layer`.
    //
    // Gemma 3's local layers attend t in (pos - window, pos]. During DECODE the
    // query is always the newest position, so the live keys are a contiguous
    // SUFFIX of the cache and the window needs no mask — just an offset. This
    // is what makes the fix a speedup rather than a tax: past `sliding_window`
    // positions, 5 of every 6 layers stop scanning the whole cache.
    //
    // (Prefill here is decode-shaped — one position per pass — so the same
    // suffix argument holds for every position the loop ever runs. A BATCHED
    // prefill would need the real 2-D mask, which is what
    // `attention_scores_prefill` in kernels/llm_sampling.cl provides.)
    uint32_t attn_kv_start(uint32_t layer, uint32_t seq_len) const {
        const uint32_t w = layer_window(layer);
        return (w == 0 || seq_len <= w) ? 0u : seq_len - w;
    }
    uint64_t kv_cache_bytes() const {
        return 2ull * n_layers * max_seq * n_kv_heads * head_dim * sizeof(float);
    }
};

// Which matvec kernel to drive, and therefore how the weights are packed.
//
// KM4 is the default on the strength of agent-docs/matvec-optimisation.md:
// 2.4x over the `matvec_q4` baseline on the Adreno 840 (11.8 -> 28.5 GB/s).
// It is a pure repacking done once at load, so it costs nothing at run time.
// The baseline layout is kept selectable because it is the simpler contract
// and the one kernels/llm.cl documents.
//
//   ColumnNibble  word[(k*cols + j) >> 3], nibble (j&7)*4        (matvec_q4)
//   Km4           uint4[(g*nblk + b)*64 + t] with g = j/64, t = j%64,
//                 b = k/32; component (k>>3)&3, nibble (k&7)*4    (mv_km4)
//   Km4Bf16       Km4 nibbles, but the block scales are bf16 packed four to a
//                 uint2 at sc[(b/4)*cols + j]                     (mv_km4_bs)
//
// Scales are [block][col] — scale[b*cols + j] — in the first two layouts.
//
// Km4Bf16 is the default. The nibble plane is byte-identical to Km4; only the
// scale plane changes, from one fp32 per 32-weight block to one bf16. That is
// worth more than it sounds: the fp32 scales are 119.2 of the 595.9 MiB a
// Gemma 3 1B token streams (+25% on top of the nibbles), so halving them
// removes 10% of ALL weight traffic, and packing four to a uint2 removes 37.5%
// of the kernel's load instructions — the axis matvec-optimisation.md found to
// be decisive. bf16 (not fp16) so the GPU-side decode is one shift and the
// fp32 exponent range is preserved; 8 mantissa bits is 0.4% relative against
// int4's ~7% quantisation step, and --verify is unchanged to six digits.
enum class MatvecLayout { ColumnNibble, Km4, Km4Bf16 };

struct MatvecConfig {
    MatvecLayout layout = MatvecLayout::Km4Bf16;
    std::string  module        = "kernels/llm_matvec.spv";
    std::string  kernel        = "mv_km4_bs";
    std::string  split_kernel  = "mv_km4_bs_split";
    std::string  reduce_kernel = "mv_reduce";

    // Split-k: extra parallelism for tall/narrow projections, at the cost of a
    // reduce dispatch. Worth 2.0x on `down` (6912x1152) and nothing on the
    // wide layers, which already have more threads than the GPU can use. Only
    // viable now that Batch has amortised submission — it was reverted once,
    // correctly, when a dispatch cost 0.31 ms.
    uint32_t split                = 4;     // 1 disables
    uint32_t split_below_cols     = 2048;  // apply split-k when cols < this
    uint32_t max_workgroups       = 65535; // split is capped to respect this

    // The COL/matvec_q4 pairing, for `--matvec-layout col`.
    static MatvecConfig baseline();
    // Effective split factor for a given shape (1 == no split).
    uint32_t split_for(uint32_t rows, uint32_t cols) const;
};

struct Paths {
    MatvecConfig matvec{};
    std::string  transformer_spv = "kernels/llm_transformer.spv";
    // Optional. When present, the three-level parallel argmax from
    // kernels/llm_sampling.cl replaces llm_transformer.cl's single-thread one.
    // That matters far more than it sounds: the single-thread scan over 262144
    // logits measures 31.5 ms on an Adreno X1-85 — comparable to the entire
    // rest of the token — against 0.08 ms for the tree
    // (agent-docs/llm-sampling-kernels.md). Same output ABI, so it is a
    // genuine drop-in. Missing file => fall back with a warning.
    std::string  sampling_spv = "kernels/llm_sampling.spv";
    std::string  model_gguf;

    // ---- load path (see agent-docs/llm-load-time.md) -----------------------
    // Packing ~1B parameters from the file's native quantisation into the KM4
    // int4 layout is pure CPU work over independent tensors, so it is threaded;
    // 0 means std::thread::hardware_concurrency(). The GPU uploads stay on the
    // owning thread — vulkore::Context is single-threaded, allocation and
    // teardown included — so workers only ever fill host staging vectors.
    uint32_t     pack_threads = 0;
    // Cap on packed-but-not-yet-uploaded host bytes. The packed model is
    // 536 MiB; without a cap the parallel packer would hold all of it in RAM
    // on top of the GPU copy.
    uint64_t     pack_wave_bytes = 192ull << 20;

    // Disk cache of the packed weights. Empty => "<model_gguf>.xpack".
    std::string  pack_cache;
    bool         use_pack_cache   = true;   // read AND write
    bool         force_repack     = false;  // ignore an existing cache, rewrite it
};

// ---------------------------------------------------------------------------
// Dispatch plan / cost model — computable with NO GPU and NO kernels, which is
// what `--dry-run` reports.
// ---------------------------------------------------------------------------

struct DispatchPlan {
    // Per layer
    uint32_t norms          = 0;  // two-pass RMSNorm dispatches
    uint32_t qk_norms       = 0;
    uint32_t matvecs        = 0;  // includes split-k reduce passes
    uint32_t rope           = 0;
    uint32_t kv_appends     = 0;
    uint32_t attention      = 0;  // scores + softmax + apply
    uint32_t activation     = 0;  // geglu
    uint32_t residuals      = 0;
    uint32_t per_layer      = 0;

    uint32_t prologue       = 0;  // embedding is CPU-side: 0 dispatches
    uint32_t epilogue_norm  = 0;  // final RMSNorm (2)
    uint32_t epilogue_head  = 0;  // lm_head matvec + argmax (2)

    uint32_t total          = 0;  // a sampled token
    uint32_t total_prefill  = 0;  // a prompt token (no lm_head, no argmax)
    uint32_t submits        = 1;  // the whole point

    // Weight traffic for one sampled token: nibbles + fp32 block scales.
    uint64_t weight_bytes       = 0;
    uint64_t weight_bytes_nibbles = 0;
    uint64_t weight_bytes_scales  = 0;
    uint64_t lm_head_bytes        = 0;
};

DispatchPlan plan_token(const DecodeConfig& cfg, const MatvecConfig& mv = {});

// Measured constants from agent-docs/llm-on-vulkore.md and
// agent-docs/llm-transformer-kernels.md (Adreno 840 / Adreno X1-85).
//
// NOTE on the bandwidth figures — they are all quoted 25% low upstream.
// Both examples/llm/bench.cpp and examples/llm/matvec_bench.cpp compute GB/s
// from `nbytes = rows*cols/2`, the int4 nibbles ONLY. The kernels also read one
// fp32 scale per 32-weight block, which is 25% more traffic; it was inside the
// measured time but not inside the byte count. So the true achieved figures are
// 1.25x the published ones:
//
//   published (nibbles only)      true (nibbles + scales)
//   11.8  llm-on-vulkore baseline   14.75
//   10.0  matvec_bench reference   12.5
//   24.7  km4+split4               30.9
//   28.5  best-per-shape           35.6
//   39.1  pure-load ceiling        39.1  (a raw stream probe: already honest)
//
// This projection counts nibbles AND scales, so it pairs with the right-hand
// column. Using a published figure with the honest byte count would
// double-count the scales. It also means the matvec work is nearer the
// hardware ceiling than its own doc claims: 35.6/39.1 is 91%, not 73%.
struct Machine {
    const char* name          = "Adreno 840, KM4+split4 (measured, batched)";
    double      bandwidth_gbs = 30.9;   // achieved incl. block scales — see above
    double      dispatch_ms   = 0.006;  // small dispatch inside a Batch
    double      submit_ms     = 0.31;   // one vkQueueSubmit + fence

    // Picks the bandwidth measured for the layout/split actually configured.
    static Machine for_matvec(const MatvecConfig& mv);
};

struct Projection {
    double weight_ms  = 0;
    double overhead_ms = 0;  // dispatches
    double submit_ms  = 0;
    double total_ms   = 0;
    double tokens_per_sec = 0;
};

Projection project(const DispatchPlan& plan, const Machine& m);

// Human-readable dry-run report. No Context, no Program, no model file.
std::string dry_run_report(const DecodeConfig& cfg, const Machine& m,
                           const MatvecConfig& mv = {});

// Packs a random matrix, runs the configured matvec path on the GPU and checks
// it against a CPU reference — i.e. verifies THIS file's packer against the
// kernel it drives. Needs a GPU and the matvec .spv, but no model and none of
// the transformer kernels. Returns the max relative error; throws on failure.
double matvec_selftest(const MatvecConfig& mv, bool verbose = true);

// Full-precision CPU forward pass straight from the GGUF — the yardstick that
// separates "the decode loop is wrong" from "int4 requantisation is too lossy".
// Slow (~1 GB of dequantisation per token); no GPU involved.
void cpu_reference_generate(const std::string& model_path, const std::string& prompt,
                            uint32_t n_tokens, uint32_t max_seq);

// Drives the GPU path and the fp32 CPU reference over the SAME token sequence
// and reports how far the logits diverge. This is the check that separates a
// wrong decode loop from merely lossier int4 weights: a loop bug shows up as
// near-zero correlation, requantisation shows up as high correlation with
// occasional top-1 flips on near-ties.
//
// `skip` generated steps are run BEFORE the first comparison, with the CPU
// reference skipping its lm_head on those (it only needs the KV-cache side
// effect). That is what makes a DEEP comparison affordable, and a deep
// comparison is the only kind that can see a sliding-window bug: below
// `sliding_window` positions the window is a no-op and a masked and an
// unmasked implementation are the same function. `--verify` at 4 positions —
// all under 512 — is structurally blind to it.
//
// `cpu_window` is a SENSITIVITY CONTROL, not a feature. Set >= 0 it overrides
// the sliding window used by the CPU reference ALONE, so the two paths are
// deliberately given different attention extents. A comparison that cannot
// tell those apart is not measuring the window at all, and that is exactly the
// state this file was in: the CPU reference had no window either, so GPU and
// CPU were two copies of the same mistake and agreed perfectly. Run
// `--verify-cpu-window 0` to see the correlation collapse and thereby prove
// the deep table is load-bearing.
void verify_against_cpu(const Paths& paths, const DecodeConfig& cfg,
                        const std::string& prompt, uint32_t n_steps,
                        uint32_t skip = 0, int32_t cpu_window = -1);

// Scores the GPU CANDIDATE HARVEST against an exact CPU top-k over the SAME
// logits, on real model output rather than synthetic data. Reports, per token,
// how much of the true top-k each harvest recovers and how much softmax
// probability MASS it captures — the quantity that actually governs whether
// sampling behaves, since missing the 61st-ranked token is not the same kind of
// error as missing the 2nd.
//
// Scores two harvests at once:
//   top-4/chain — what the decode loop uses (`topk4_partial`).
//   top-1/chain — slot 0 only, i.e. exactly what reusing the existing
//                 `argmax_partial` output as a candidate set would give.
void sample_check(const Paths& paths, const DecodeConfig& cfg,
                  const std::string& prompt, uint32_t n_steps);

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct Metrics {
    double   ms             = 0;  // wall time of the last decode_step()
    double   record_ms      = 0;  // time spent building the Batch
    double   gpu_ms         = 0;  // submit + fence wait
    double   sample_ms      = 0;
    double   tokens_per_sec = 0;
    uint32_t dispatches     = 0;
    uint32_t submits        = 0;
    uint32_t position       = 0;  // KV cache position after the step
};

// ---------------------------------------------------------------------------
// DecodeSession
// ---------------------------------------------------------------------------

// Where the weight-load seconds went. Reported after every load, because the
// two paths (repack vs cache hit) differ by ~6x and a silent fallback to
// repacking would otherwise look like the phone having a bad day.
struct LoadStats {
    bool     from_cache   = false;  // weights came off disk, no packing at all
    bool     cache_written = false;
    uint32_t threads      = 1;      // packer threads actually used
    double   pack_ms      = 0;      // CPU dequant/requantise/repack (wall)
    double   upload_ms    = 0;      // vulkore::Buffer alloc + upload, serialised
    double   cache_read_ms = 0;     // total of the cache-hit path
    double   cache_io_ms   = 0;     // read(2) only
    double   cache_hash_ms = 0;     // checksum verification
    double   cache_write_ms = 0;
    uint64_t cache_bytes  = 0;
    std::string cache_path;
};

class DecodeSession {
public:
    // Loads both SPIR-V modules, resolves every kernel in the contract above
    // (throwing by name if one is missing), packs the GGUF weights into the
    // q4 layout, and allocates the KV cache.
    DecodeSession(const Paths& paths, const DecodeConfig& cfg);
    ~DecodeSession();

    DecodeSession(const DecodeSession&)            = delete;
    DecodeSession& operator=(const DecodeSession&) = delete;

    // Runs the prompt through the network one token at a time, filling the KV
    // cache. The lm_head matvec and argmax are SKIPPED for every token, so a
    // prompt token costs 151 MB less traffic than a generated one.
    void prefill(std::span<const uint32_t> tokens);

    // One generated token: reads the last committed token, runs the network,
    // samples, appends to the KV cache. Returns the sampled token id, which
    // becomes the input to the next call.
    uint32_t decode_step();

    void reset();  // clears the KV cache position; weights stay resident

    Metrics  last_metrics() const { return metrics_; }
    uint32_t position() const { return pos_; }
    const DecodeConfig& config() const { return cfg_; }
    const DispatchPlan& plan() const { return plan_; }
    // Bytes of GPU memory held by weights / KV cache.
    uint64_t weight_bytes() const;
    uint64_t kv_bytes() const { return cfg_.kv_cache_bytes(); }
    // Wall time spent in the KV-cache allocations at construction. The cost of
    // a large max_seq is mostly memory; this is the part that is time.
    double   kv_alloc_ms() const;
    const LoadStats& load_stats() const;
    std::string device_name() const;

    // The tokenizer built from the same GGUF file the weights came from. Its
    // vocabulary borrows the mapping, so it dies with the session.
    const Tokenizer& tokenizer() const;

    // Logits from the most recent forward pass (vocab floats, one download).
    std::vector<float> last_logits() const;

    // The (value, index) candidates `topk4_partial` harvested during the most
    // recent forward pass, straight out of mapped memory. Empty if the harvest
    // is not active. Exposed for `--sample-check`, which scores this set
    // against an exact CPU top-k over the same logits.
    std::vector<std::pair<float, uint32_t>> last_candidates() const;
    bool candidate_harvest_active() const;

private:
    struct Impl;
    // Records ONE token's entire dispatch chain into `b`. Returns the dispatch
    // count so the loop can check it against the plan.
    uint32_t record_token(Batch& b, uint32_t pos, bool want_logits);
    void     write_embedding(uint32_t token);
    uint32_t sample_cpu(std::vector<float>& logits);
    // Finishes sampling from the GPU candidate harvest, read out of mapped
    // memory. No submit, no full-vocab scan.
    uint32_t sample_candidates();
    // Draws from an already-selected (value, index) list: temperature, softmax,
    // multinomial. Shared by both paths so they differ ONLY in how the
    // candidate set was found.
    uint32_t draw(std::vector<std::pair<float, uint32_t>>& cand);

    std::unique_ptr<Impl> impl_;
    DecodeConfig cfg_;
    DispatchPlan plan_;
    Metrics      metrics_{};
    uint32_t     pos_       = 0;
    uint32_t     cur_token_ = 0;
};

}  // namespace vulkore::llm
