// Token sampling + prefill-shaped kernels for the Gemma 3 1B decode pipeline.
//
// Two groups live here:
//
//   PART 1 - SAMPLING. Everything between the lm_head logits and a token id.
//     Gemma's vocab is 262144, so anything single-threaded over the vocab is a
//     real cost on the critical path of every token. kernels/llm_transformer.cl
//     ships a one-thread `argmax` and a one-thread-per-row `softmax_rows`; both
//     were flagged there as deliberately decode-shaped. This file is the
//     parallel replacement.
//
//   PART 2 - PREFILL SHAPES. Prefill processes T positions at once, so every
//     kernel in llm_transformer.cl that bakes in "one position" (rope's `pos`,
//     embed_lookup's `token`, kv_append's `pos`, the single-query attention
//     pair, the single-vector RMSNorms) needs a (row, column)-indexed twin.
//
// ABI (see tests/kernels/README.md): storage buffers only, PODs in ONE push
// constant block <= 128 bytes, uniform workgroup size, grid == GLOBAL thread
// count so every kernel bound-checks.
//
// **`__local` is REJECTED** - clspv emits WorkgroupVariableSize and
// vulkore::Program throws the module out. That has now blocked a workgroup
// reduction three separate times in this project, so every reduction below is
// two-pass through GLOBAL memory: a `_partial` kernel with `nparts` threads
// each walking a STRIDED chain, then a tiny finalizer. Strided (thread p takes
// k = p, p+nparts, ...) not blocked, so a wavefront reads contiguous floats.
//
// Indexing is `uint` throughout, never `size_t` - a 64-bit index makes clspv
// request an Int64 capability the target need not support. The largest product
// computed here is n_rows*n_cols for prefill scores (4 heads * 2048 q * 2048 k
// = 1.7e7), comfortably inside uint.

// ===========================================================================
// PART 1 - SAMPLING
// ===========================================================================

// ---- Parallel argmax ------------------------------------------------------
//
// Replaces llm_transformer.cl's `argmax`, which is ONE thread walking all
// 262144 logits and costs a MEASURED 31.5 ms on the Adreno X1-85.
//
// `argmax_partial` (nparts threads, one strided chain each) then
// `argmax_reduce` (an optional extra tree level) then `argmax_final` (one
// thread). Use THREE levels - 262144 -> 4096 -> 64 -> 1 measures 0.074-0.086 ms
// over five runs, **~380x** the serial kernel. Two levels is much worse than
// it looks (best 0.265 ms at nparts=1024) because the serial finalizer is the same
// ~0.13 us/element walk being replaced, just shorter. See the nparts sweep in
// examples/llm/sampling_test.cpp.
//
// Tie-breaking is LOWEST INDEX WINS, matching a straight-line CPU
// `for (k) if (v > best)` reference exactly:
//   - pass 1 uses strict `>`, so the earliest k in a chain holds the slot;
//   - pass 2 compares (value, index) lexicographically, because chain p is not
//     ordered by index relative to chain p+1.
// Without the index tiebreak the GPU and CPU disagree on any duplicated
// maximum, which is not exotic: quantised logits collide often.
kernel void argmax_partial(global float* pval, global uint* pidx,
                           global const float* logits, uint n, uint nparts) {
    uint p = get_global_id(0);
    if (p >= nparts) return;

    float bv = -INFINITY;
    uint bi = 0u;
    for (uint k = p; k < n; k += nparts) {
        float v = logits[k];
        if (v > bv) { bv = v; bi = k; }
    }
    pval[p] = bv;
    pidx[p] = bi;
}

// One more TREE LEVEL over (value, index) pairs: nparts_out threads each reduce
// a strided chain of the previous level's output.
//
// This exists because the two-level version is finalizer-bound, by a lot.
// Measured on the Adreno X1-85 at vocab 262144 (see sampling_test.cpp):
// pass 1 at nparts=4096 takes 0.034 ms - essentially the 1 MB read at the
// ~30 GB/s this device actually achieves - while the serial argmax_final over
// those 4096 partials takes 0.66 ms, i.e. 19x the useful work. Both serial
// walks cost ~0.13 us/element, so the finalizer is just a shorter version of
// the same bottleneck being replaced. Inserting this level (4096 -> 64 -> 1)
// removes it: three dispatches inside a Batch cost ~0.006 ms each.
//
// The comparison is (value, index) lexicographic, identical to argmax_final,
// so any number of levels composes without changing the tie-break rule.
kernel void argmax_reduce(global float* oval, global uint* oidx,
                          global const float* pval, global const uint* pidx,
                          uint n, uint nparts) {
    uint p = get_global_id(0);
    if (p >= nparts) return;
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint k = p; k < n; k += nparts) {
        float v = pval[k];
        uint  e = pidx[k];
        if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
    }
    oval[p] = bv;
    oidx[p] = bi;
}

// out[0] = winning index (as float), out[1] = winning value.
// Same output ABI as llm_transformer.cl's `argmax`, so this pair is a drop-in
// replacement for it. The index-as-float encoding is exact up to 2^24
// (16.7M) - fine for a 262144 vocab, and the caller usually wants the uint
// form too, so `pidx_out` carries it losslessly.
kernel void argmax_final(global float* out, global uint* pidx_out,
                         global const float* pval, global const uint* pidx,
                         uint nparts) {
    uint i = get_global_id(0);
    if (i >= 1u) return;

    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint p = 0; p < nparts; ++p) {
        float v = pval[p];
        uint  k = pidx[p];
        if (v > bv || (v == bv && k < bi)) { bv = v; bi = k; }
    }
    out[0] = (float)bi;
    out[1] = bv;
    pidx_out[0] = bi;
}

// ---- Temperature ----------------------------------------------------------
//
// logits /= T. The host passes `inv_temp` = 1/T because the reciprocal is one
// scalar computed once, and because T == 0 (greedy) has no reciprocal - the
// host must route T == 0 to argmax instead of calling this with inf.
kernel void apply_temperature(global float* logits, float inv_temp, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    logits[i] = logits[i] * inv_temp;
}

// ---- Numerically stable softmax over the FULL vocabulary -------------------
//
// llm_transformer.cl's `softmax_rows` gives one thread to each row. Over the
// vocabulary there is exactly ONE row, so that kernel degenerates to a single
// thread doing 3 serial passes over 262144 elements. This is the parallel
// form, in five dispatches that never leave the device:
//
//   reduce_max_partial   grid nparts   pmax[p]  = max of chain p
//   reduce_max_final     grid 1        stats[0] = max
//   softmax_exp_partial  grid nparts   out[k]   = exp(x[k]-max), psum[p] = chain sum
//   reduce_sum_final     grid 1        stats[1] = total
//   softmax_normalize    grid n        out[k]  /= total
//
// Five dispatches sounds expensive and is not: llm-transformer-kernels.md
// measured a dispatch inside an vulkore::Batch at ~0.006 ms (0.31 ms is the cost
// of a SUBMIT, not of a dispatch), and the whole chain is one submit.
//
// `stats` is a 2-float device buffer, deliberately never read back: the max
// and the sum stay on the GPU, so a full-vocab softmax costs zero host
// round-trips. That is the entire reason the finalizers exist as kernels
// rather than as three lines of C++.
kernel void reduce_max_partial(global float* pmax, global const float* x,
                               uint n, uint nparts) {
    uint p = get_global_id(0);
    if (p >= nparts) return;
    float m = -INFINITY;
    for (uint k = p; k < n; k += nparts) m = fmax(m, x[k]);
    pmax[p] = m;
}

kernel void reduce_max_final(global float* stats, global const float* pmax,
                             uint nparts) {
    uint i = get_global_id(0);
    if (i >= 1u) return;
    float m = -INFINITY;
    for (uint p = 0; p < nparts; ++p) m = fmax(m, pmax[p]);
    stats[0] = m;
}

// out may alias x (in-place is fine - each thread only touches its own chain).
kernel void softmax_exp_partial(global float* out, global float* psum,
                                global const float* x, global const float* stats,
                                uint n, uint nparts) {
    uint p = get_global_id(0);
    if (p >= nparts) return;
    float m = stats[0];
    float s = 0.0f;
    for (uint k = p; k < n; k += nparts) {
        // x[k] - m <= 0 always, so exp() cannot overflow. It CAN underflow to
        // 0 for logits far below the max, which is the correct answer and is
        // exactly why the max is subtracted at all.
        float e = exp(x[k] - m);
        out[k] = e;
        s += e;
    }
    psum[p] = s;
}

// Tree level for the sum, mirroring argmax_reduce. The max side needs no such
// kernel: `reduce_max_partial` already reads a float array and writes strided
// chain maxima, so it composes with itself directly (262144 -> 4096 -> 64),
// as long as the caller alternates buffers.
//
// Whether a middle level pays depends on nparts. At the sizes here the serial
// finalizer costs ~0.13 us per element, so it starts to dominate above a few
// hundred partials - see the nparts sweep in sampling_test.cpp.
kernel void reduce_sum_partial(global float* out, global const float* in,
                               uint n, uint nparts) {
    uint p = get_global_id(0);
    if (p >= nparts) return;
    float s = 0.0f;
    for (uint k = p; k < n; k += nparts) s += in[k];
    out[p] = s;
}

kernel void reduce_sum_final(global float* stats, global const float* psum,
                             uint nparts) {
    uint i = get_global_id(0);
    if (i >= 1u) return;
    float s = 0.0f;
    for (uint p = 0; p < nparts; ++p) s += psum[p];
    stats[1] = s;
}

kernel void softmax_normalize(global float* out, global const float* stats,
                              uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    out[i] = out[i] / stats[1];
}

// ---- Top-k ----------------------------------------------------------------
//
// Partial selection of the k largest, k typically 40. A full 262144-element
// sort is not needed and would be the wrong tool; this is two stages:
//
//   topk_partial      grid nparts   each thread keeps the top-k of its strided
//                                   chain in a PRIVATE array, emits k candidates
//   topk_rank_select  grid ncand    each candidate counts how many candidates
//                                   beat it; rank < k -> write out[rank]
//
// Stage 2 is O(ncand^2) comparisons but they are spread over ncand threads and
// the candidate array is a few tens of KB (cache-resident), so each thread
// does ncand cheap reads. With nparts=256 and k=40 that is 10240 candidates =
// 10240 reads per thread. It needs no sort, no scan, no __local, and it is
// exact and deterministic.
//
// The correctness argument for stage 1: the chains partition the vocabulary,
// so any global top-k element is in exactly one chain, and it is among that
// chain's top-k. Taking each chain's top-k therefore cannot lose it. (Taking
// each chain's top-1 WOULD lose it, which is why the private array is needed
// and why nparts cannot simply be cranked up instead.)
//
// Ranks are a permutation because (value, index) pairs are unique: chains are
// disjoint so no index repeats. Ties in value are broken by lower index, the
// same rule as argmax, so the output matches a CPU stable sort exactly.
//
// **The output is sorted DESCENDING by construction** - out[0] is the largest.
// That is free here (rank IS the sorted position) and it is what top-p needs.
#define XP_TOPK_MAX 64u

kernel void topk_partial(global float* cval, global uint* cidx,
                         global const float* x, uint n, uint nparts, uint k) {
    uint p = get_global_id(0);
    if (p >= nparts) return;
    if (k > XP_TOPK_MAX) k = XP_TOPK_MAX;

    // Sorted DESCENDING; bv[k-1] is the current admission threshold.
    float bv[XP_TOPK_MAX];
    uint  bi[XP_TOPK_MAX];
    for (uint j = 0; j < k; ++j) { bv[j] = -INFINITY; bi[j] = 0xFFFFFFFFu; }

    for (uint e = p; e < n; e += nparts) {
        float v = x[e];
        // Strict `>` keeps the earliest index on a tie, matching the CPU rule.
        if (!(v > bv[k - 1u])) continue;
        uint j = k - 1u;
        while (j > 0u && v > bv[j - 1u]) {
            bv[j] = bv[j - 1u];
            bi[j] = bi[j - 1u];
            --j;
        }
        bv[j] = v;
        bi[j] = e;
    }

    for (uint j = 0; j < k; ++j) {
        cval[p * k + j] = bv[j];
        cidx[p * k + j] = bi[j];
    }
}

// ---- Top-4-per-chain: the CANDIDATE HARVEST for host-side top-k -----------
//
// `topk_partial` above is the general form, but its `float bv[64]` private
// array is sized at XP_TOPK_MAX regardless of the runtime `k`, and it is
// dynamically indexed - so clspv puts it in a real Private allocation (512 B
// per thread) that the compiler cannot keep in registers. At nparts=4096 that
// is 2 MiB of scratch traffic on top of the logits read, and it turns a
// memory-bound pass into a spill-bound one.
//
// This kernel is the same idea with M FIXED AT 4 and held in four scalar
// register pairs - no array, no dynamic index, no spill. Cost is then the same
// 1 MiB logits read as `argmax_partial` (the measured memory floor for this
// pass) plus a handful of compares per element.
//
// PURPOSE: harvest a small, cheap, near-exact SUPERSET of the vocabulary's
// top-k so the HOST can finish the sampling on a few tens of KB instead of
// downloading all 262144 logits. With nparts=4096 over vocab 262144 each chain
// is 64 elements and emits its best 4, giving 16384 candidates (128 KiB).
//
// EXACTNESS. Chains partition the vocabulary, so a true top-k element is lost
// only if FIVE OR MORE of the true top-k share one chain. Chain membership is
// `index % nparts`, which is uncorrelated with rank, so for k=64 into 4096
// chains that is C(64,5)/4096^4 ~ 3e-8 per token. Taking only the best ONE per
// chain (i.e. plain `argmax_partial`) is a completely different proposition:
// the collision that matters is then a PAIR, ~0.49 expected per token, and
// roughly 40% of tokens lose at least one true top-64 member. See the measured
// `--sample-check` table in agent-docs/llm-sampling-cost.md.
//
// Slots are filled descending; unfilled ones are (-INFINITY, 0xFFFFFFFF), the
// same sentinel `topk_rank_select` skips.
kernel void topk4_partial(global float* cval, global uint* cidx,
                          global const float* x, uint n, uint nparts) {
    uint p = get_global_id(0);
    if (p >= nparts) return;

    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
    uint  i0 = 0xFFFFFFFFu, i1 = 0xFFFFFFFFu, i2 = 0xFFFFFFFFu, i3 = 0xFFFFFFFFu;

    for (uint e = p; e < n; e += nparts) {
        float v = x[e];
        // Strict `>` everywhere keeps the earliest index on a tie, matching
        // argmax_partial and a straight-line CPU reference.
        if (!(v > v3)) continue;
        if (v > v0) {
            v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = v; i0 = e;
        } else if (v > v1) {
            v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v; i1 = e;
        } else if (v > v2) {
            v3 = v2; i3 = i2; v2 = v; i2 = e;
        } else {
            v3 = v; i3 = e;
        }
    }

    uint o = p * 4u;
    cval[o + 0u] = v0; cidx[o + 0u] = i0;
    cval[o + 1u] = v1; cidx[o + 1u] = i1;
    cval[o + 2u] = v2; cidx[o + 2u] = i2;
    cval[o + 3u] = v3; cidx[o + 3u] = i3;
}

kernel void topk_rank_select(global float* oval, global uint* oidx,
                             global const float* cval, global const uint* cidx,
                             uint ncand, uint k) {
    uint i = get_global_id(0);
    if (i >= ncand) return;

    float v = cval[i];
    uint  d = cidx[i];
    // Unfilled slots from stage 1 (fewer than k elements in a chain). They must
    // not consume an output rank.
    if (d == 0xFFFFFFFFu) return;

    uint rank = 0u;
    for (uint j = 0; j < ncand; ++j) {
        float w = cval[j];
        uint  e = cidx[j];
        if (e == 0xFFFFFFFFu) continue;
        if (w > v || (w == v && e < d)) ++rank;
    }
    if (rank < k) { oval[rank] = v; oidx[rank] = d; }
}

// Set every logit strictly below the k-th largest to -INFINITY, so a
// subsequent full-vocab softmax puts exactly zero mass outside the top-k.
// `topk_val` is topk_rank_select's DESCENDING output, so the threshold is
// topk_val[k-1].
//
// This is the "mask then softmax" route, which keeps the sampling distribution
// on the device in its natural 262144-long layout. The alternative - softmax
// only the k survivors - is cheaper and is what the host path does; both are
// provided because the choice depends on whether you are downloading anyway.
kernel void topk_mask(global float* logits, global const float* topk_val,
                      uint k, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float t = topk_val[k - 1u];
    if (logits[i] < t) logits[i] = -INFINITY;
}

// ---- Top-p / nucleus ------------------------------------------------------
//
// See agent-docs/llm-inference.md: the recommendation is to do top-p on
// the HOST, after downloading the k survivors. It needs sorted-descending
// probabilities and a running cumulative sum; the sort is the expensive part
// and top-k already produced it for k elements, and the host must receive the
// sampled id anyway because the next embed_lookup takes the token as a PUSH
// CONSTANT. So a device-side scan buys nothing and costs dispatches.
//
// This kernel exists for the case where you want the pipeline to stay on the
// GPU end to end (e.g. speculative decoding, where the id is consumed by
// another kernel). It is ONE thread - and unlike a one-thread argmax over
// 262144 that is defensible, because k is ~40. Serial over 40 elements is
// nothing; serial over a vocabulary is a millisecond.
//
// Semantics match the usual definition (HF, llama.cpp): keep the shortest
// descending prefix whose cumulative probability REACHES p, i.e. the token
// that crosses the threshold is kept. Zeroed weights drop out of the
// subsequent multinomial draw.
kernel void topp_cutoff(global float* w, uint k, float p) {
    uint i = get_global_id(0);
    if (i >= 1u) return;

    float total = 0.0f;
    for (uint j = 0; j < k; ++j) total += w[j];
    if (!(total > 0.0f)) return;

    float cum = 0.0f;
    for (uint j = 0; j < k; ++j) {
        // Threshold tested on the mass BEFORE this token, so the crossing
        // token survives. j == 0 always survives (cum == 0 < p for p > 0).
        if (cum >= p * total) w[j] = 0.0f;
        else cum += w[j];
    }
}

// ---- Multinomial sampling -------------------------------------------------
//
// Draw from a k-element weight vector given a uniform u in [0,1). Weights need
// NOT be normalised - the running target is u*total - which is what lets this
// run straight on topp_cutoff's output. `idx` maps slot -> vocab id.
// out[0] = chosen vocab id (as float), out[1] = its weight.
//
// One thread, serial over k (~40). Intended for the post-top-k list.
kernel void multinomial_sample(global float* out, global uint* idx_out,
                               global const float* w, global const uint* idx,
                               uint k, float u) {
    uint i = get_global_id(0);
    if (i >= 1u) return;

    float total = 0.0f;
    for (uint j = 0; j < k; ++j) total += w[j];

    float target = u * total;
    float cum = 0.0f;
    uint chosen = 0u;
    float cw = 0.0f;
    for (uint j = 0; j < k; ++j) {
        cum += w[j];
        if (w[j] > 0.0f) { chosen = idx[j]; cw = w[j]; }   // last nonzero so far
        if (cum > target) break;
    }
    out[0] = (float)chosen;
    out[1] = cw;
    idx_out[0] = chosen;
}

// Full-vocabulary multinomial WITHOUT a 262144-long prefix scan.
//
// It reuses the softmax intermediates exactly as `softmax_exp_partial` leaves
// them: `w` = the unnormalised exp values, `psum[p]` = the sum of strided chain
// p, `stats[1]` = the grand total. Sampling is then two nested searches, both
// serial in one thread but both SHORT:
//   1. walk the nparts (=256) chain sums to find the owning chain,
//   2. walk that chain's n/nparts (=1024) elements to find the token.
// 1280 steps instead of 262144 - a 200x cut without any extra dispatch, and
// with no normalisation pass needed at all.
//
// The chains are strided, so this samples from a PERMUTED cumulative
// distribution. That is fine: a multinomial draw does not care about the order
// of the categories.
kernel void multinomial_strided(global float* out, global uint* idx_out,
                                global const float* w, global const float* psum,
                                global const float* stats, uint n, uint nparts,
                                float u) {
    uint i = get_global_id(0);
    if (i >= 1u) return;

    float target = u * stats[1];
    float cum = 0.0f;
    uint chain = nparts - 1u;
    for (uint p = 0; p < nparts; ++p) {
        float s = psum[p];
        if (cum + s > target) { chain = p; break; }
        cum += s;
    }

    uint chosen = chain;
    for (uint k = chain; k < n; k += nparts) {
        chosen = k;
        cum += w[k];
        if (cum > target) break;
    }
    out[0] = (float)chosen;
    out[1] = w[chosen];
    idx_out[0] = chosen;
}

// ===========================================================================
// PART 2 - PREFILL SHAPES
// ===========================================================================
//
// llm_transformer.cl is a DECODE step: one token, one position. Prefill runs T
// positions at once, so kernels that hard-code a single position or a single
// vector need a (row, column)-indexed twin. Elementwise kernels there
// (geglu, add_residual, add_vec, mul_scalar) already work unchanged on a
// T*hidden buffer - they are not repeated here.
//
// Row-major layout everywhere: row r occupies [r*n_cols, (r+1)*n_cols).

// ---- softmax over many rows -----------------------------------------------
//
// llm_transformer.cl's `softmax_rows` is ONE THREAD PER ROW doing three serial
// passes. At decode there are n_heads = 4 rows, so that kernel launches 4
// threads; documented there as deliberate, and it is - the score matrix is 8 KB
// and cache-resident. At prefill the score matrix is [head][query][key], so
// n_rows = n_heads*n_query is thousands of rows and n_cols is the context.
//
// Split into two, so the O(n_rows*n_cols) part is fully (row, col) parallel:
//   softmax_rows_stats  grid n_rows           two serial passes per row -> max, sum
//   softmax_rows_apply  grid n_rows*n_cols    one thread PER ELEMENT
//
// The reduction stays one-thread-per-row on purpose. At prefill that is
// already thousands of threads, which saturates the device; splitting it
// further would add dispatches to parallelise something that is not the
// bottleneck. That is precisely the opposite conclusion to the vocab softmax
// above, and for the opposite reason: there, n_rows is 1.
//
// `stats` is 2*n_rows floats: stats[2r] = row max, stats[2r+1] = row sum.
kernel void softmax_rows_stats(global float* stats, global const float* x,
                               uint n_rows, uint n_cols) {
    uint r = get_global_id(0);
    if (r >= n_rows) return;
    uint base = r * n_cols;

    float m = -INFINITY;
    for (uint j = 0; j < n_cols; ++j) m = fmax(m, x[base + j]);

    // A fully masked row (every entry -INFINITY) would give m = -INFINITY and
    // exp(-inf - -inf) = NaN. Causal and sliding-window masks always leave the
    // diagonal live, so this cannot happen from attention_scores_prefill; the
    // guard is here so a caller-supplied mask cannot produce NaNs silently.
    if (!isfinite(m)) { stats[2u * r] = 0.0f; stats[2u * r + 1u] = 1.0f; return; }

    float s = 0.0f;
    for (uint j = 0; j < n_cols; ++j) s += exp(x[base + j] - m);

    stats[2u * r] = m;
    stats[2u * r + 1u] = s;
}

kernel void softmax_rows_apply(global float* x, global const float* stats,
                               uint n_rows, uint n_cols) {
    uint i = get_global_id(0);
    uint total = n_rows * n_cols;
    if (i >= total) return;
    uint r = i / n_cols;
    x[i] = exp(x[i] - stats[2u * r]) / stats[2u * r + 1u];
}

// ---- prefill attention ----------------------------------------------------
//
// scores[h][qi][t] with row stride seq_len, i.e. the decode layout extended by
// a query axis. `q_offset` is the absolute position of query 0 (nonzero when
// prefilling in chunks or continuing after a decode), so the absolute query
// position is q_offset + qi.
//
// Masking is folded in rather than done as a separate pass - it is one compare
// on a value already in a register, versus a whole extra dispatch over
// n_heads*n_query*seq_len:
//   - CAUSAL: key position t must satisfy t <= q_offset + qi.
//   - SLIDING WINDOW: `window == 0` disables it; otherwise the attended span is
//     the last `window` positions, t > qpos - window. Gemma 3 alternates five
//     local (windowed, theta 1e4) layers to one global (theta 1e6) layer, which
//     llm_transformer.cl had no way to express at all.
// The diagonal t == qpos survives both, so no row is ever fully masked.
kernel void attention_scores_prefill(global float* scores, global const float* q,
                                     global const float* kcache, uint n_heads,
                                     uint kv_heads, uint head_dim, uint n_query,
                                     uint seq_len, uint q_offset, uint window,
                                     float scale) {
    uint i = get_global_id(0);
    uint total = n_heads * n_query * seq_len;
    if (i >= total) return;

    uint t  = i % seq_len;
    uint hq = i / seq_len;
    uint qi = hq % n_query;
    uint h  = hq / n_query;

    uint qpos = q_offset + qi;
    if (t > qpos) { scores[i] = -INFINITY; return; }
    if (window != 0u && t + window <= qpos) { scores[i] = -INFINITY; return; }

    uint group = n_heads / kv_heads;
    uint kvh = h / group;
    uint qb = (h * n_query + qi) * head_dim;
    uint kb = (t * kv_heads + kvh) * head_dim;

    float acc = 0.0f;
    for (uint d = 0; d < head_dim; ++d) acc += q[qb + d] * kcache[kb + d];
    scores[i] = acc * scale;
}

// out[h][qi][d] = sum_t probs[h][qi][t] * v[t][kv_head(h)][d].
// One thread per output element; the decode kernel's grid was n_heads*head_dim,
// this adds the query axis.
kernel void attention_apply_prefill(global float* out, global const float* probs,
                                    global const float* vcache, uint n_heads,
                                    uint kv_heads, uint head_dim, uint n_query,
                                    uint seq_len) {
    uint i = get_global_id(0);
    uint total = n_heads * n_query * head_dim;
    if (i >= total) return;

    uint d  = i % head_dim;
    uint hq = i / head_dim;
    uint qi = hq % n_query;
    uint h  = hq / n_query;

    uint group = n_heads / kv_heads;
    uint kvh = h / group;
    uint pb = (h * n_query + qi) * seq_len;
    uint vstride = kv_heads * head_dim;
    uint vb = kvh * head_dim + d;

    float acc = 0.0f;
    for (uint t = 0; t < seq_len; ++t) acc += probs[pb + t] * vcache[t * vstride + vb];
    out[i] = acc;
}

// ---- prefill RMSNorm ------------------------------------------------------
//
// Same two-pass split that llm-transformer-kernels.md measured at 6.4x the
// one-pass form, extended over rows. Pass 1 is one thread per ROW (T rows at
// prefill, so thousands); pass 2 is one thread per ELEMENT.
kernel void rmsnorm_rows_sumsq(global float* rowss, global const float* in,
                               uint n_rows, uint n) {
    uint r = get_global_id(0);
    if (r >= n_rows) return;
    uint base = r * n;
    float ss = 0.0f;
    for (uint j = 0; j < n; ++j) { float v = in[base + j]; ss += v * v; }
    rowss[r] = ss;
}

// Gemma's (1 + w) convention; `w` is shared across rows.
kernel void rmsnorm_rows_apply_gemma(global float* out, global const float* in,
                                     global const float* w,
                                     global const float* rowss, uint n_rows,
                                     uint n, float eps) {
    uint i = get_global_id(0);
    uint total = n_rows * n;
    if (i >= total) return;
    uint r = i / n;
    uint j = i - r * n;
    float scale = rsqrt(rowss[r] / (float)n + eps);
    out[i] = in[i] * scale * (1.0f + w[j]);
}

// Per-head QK-norm over many tokens: [token][head][head_dim], weight shared.
kernel void rmsnorm_heads_rows_gemma(global float* out, global const float* in,
                                     global const float* w, uint n_tokens,
                                     uint n_heads, uint head_dim, float eps) {
    uint i = get_global_id(0);
    uint total = n_tokens * n_heads * head_dim;
    if (i >= total) return;
    uint d = i % head_dim;
    uint head = i / head_dim;            // flattened (token, head)
    uint base = head * head_dim;

    float ss = 0.0f;
    for (uint j = 0; j < head_dim; ++j) { float v = in[base + j]; ss += v * v; }
    out[i] = in[i] * rsqrt(ss / (float)head_dim + eps) * (1.0f + w[d]);
}

// ---- prefill RoPE ---------------------------------------------------------
//
// llm_transformer.cl's `rope`/`rope_cached` take a single scalar `pos`, so they
// can only rotate one token. Here the position varies per row.
//
// Rotate-half (HF Gemma/Llama): d pairs with d + head_dim/2. One thread per
// PAIR, writing both halves. Layout [token][head][head_dim], absolute position
// of token ti is pos0 + ti.
//
// The angle tables cannot be precomputed per-position the way `rope_cached`
// does for decode (that would need a T x head_dim/2 table rebuilt per chunk),
// so this recomputes inv_freq per thread via exp2/log2 - `pow` is avoided on
// the Mandelbulb precedent recorded in llm-transformer-kernels.md.
kernel void rope_prefill(global float* x, uint n_tokens, uint n_heads,
                         uint head_dim, uint pos0, float theta) {
    uint i = get_global_id(0);
    uint hdhalf = head_dim / 2u;
    uint total = n_tokens * n_heads * hdhalf;
    if (i >= total) return;

    uint d  = i % hdhalf;
    uint th = i / hdhalf;
    uint ti = th / n_heads;

    uint base = th * head_dim + d;
    float frac = (float)(2u * d) / (float)head_dim;
    float inv = exp2(-log2(theta) * frac);
    float ang = (float)(pos0 + ti) * inv;
    float c = cos(ang), s = sin(ang);

    float a = x[base];
    float b = x[base + hdhalf];
    x[base]        = a * c - b * s;
    x[base + hdhalf] = b * c + a * s;
}

// ---- prefill plumbing -----------------------------------------------------

// Append T rows to the KV cache starting at `pos0`. Decode's kv_append writes
// one row; grid here is n_tokens*row.
kernel void kv_append_rows(global float* cache, global const float* src,
                           uint n_tokens, uint row, uint pos0) {
    uint i = get_global_id(0);
    uint total = n_tokens * row;
    if (i >= total) return;
    cache[pos0 * row + i] = src[i];
}

// Gather T embedding rows. Decode's embed_lookup takes the token as a PUSH
// CONSTANT, which cannot express T tokens; the ids come from a buffer instead.
//
// `(uint)token * n` is deliberately NOT widened to size_t - a 64-bit index
// would make clspv request Int64. Gemma's table is 262144 x 1152, so the
// product peaks at 3.0e8, safely inside uint.
kernel void embed_lookup_rows(global float* out, global const float* table,
                              global const uint* tokens, uint n_tokens, uint n) {
    uint i = get_global_id(0);
    uint total = n_tokens * n;
    if (i >= total) return;
    uint ti = i / n;
    uint j = i - ti * n;
    out[i] = table[tokens[ti] * n + j];
}
