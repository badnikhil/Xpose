// Vulkore transformer kernels — everything a Gemma 3 1B single-token decode step
// needs that ISN'T a matvec. The matvecs live in llm.cl and dominate the time
// budget (~500 MB of int4 weights per token); these are the small, awkward,
// bandwidth-trivial ops in between, and their cost is almost entirely DISPATCH
// cost. That shapes every design decision below: prefer ONE dispatch doing
// redundant arithmetic over TWO dispatches that are individually optimal.
// A trivial dispatch measured 0.31 ms on Adreno 840; a 1152-element reduction
// does not come close to costing that.
//
// ABI (tests/kernels/README.md — violations make vulkore::Program throw at load):
//   - storage buffers only, no images/samplers/UBOs
//   - PODs in ONE push-constant block, <= 128 bytes
//   - uniform workgroup size; grid is the GLOBAL thread count rounded UP to
//     whole workgroups, so EVERY kernel bound-checks its id
//   - NO __local arrays. clspv emits WorkgroupVariableSize (reflection
//     instruction 42) for them and Program refuses the module. This has bitten
//     this repo twice (mlp_trainer reduce_sum, a Mandelbulb), so every
//     reduction here is either serial-per-thread or redundant-per-thread
//     through global memory.
//   - buffer element types are 4-byte aligned (vulkore::Buffer static_asserts);
//     everything here is float or uint.
//
// Gemma 3 1B shapes: hidden 1152, ffn 6912, 26 layers, 4 q heads, 1 kv head,
// head_dim 256, RMSNorm (1+w convention), RoPE (rotate-half), GeGLU (tanh gelu).
//
// KV cache layout used throughout: [token][kv_head][head_dim], i.e.
//   k[(t * kv_heads + kvh) * head_dim + d]
// which makes appending a new token a contiguous write at the end. Attention
// scores are [q_head][seq_len], row stride == seq_len (rebuilt every step).

// ===========================================================================
// RMSNorm
// ===========================================================================
// out[i] = in[i] * rsqrt(mean(in^2) + eps) * w[i]
//
// The sum of squares is a reduction, and __local is off the table. This version
// has EVERY thread recompute the whole sum: n threads x n reads, one dispatch.
//
// I wrote this expecting it to WIN — 1.3 M cache-resident loads sounded cheaper
// than a second 0.31 ms submit. Measured on the Adreno X1-85 it loses badly:
//
//   one-pass (n^2, 1 dispatch)   0.1576 ms
//   two-pass (2 dispatches)      0.0245 ms      <- 6.4x faster
//   trivial kernel (floor)       0.0055 ms
//
// (steady state over 3 runs; the first run of a fresh process reads high while
// pipelines warm and clocks ramp.)
//
// Two things were wrong in the estimate. The 0.31 ms figure is the cost of a
// SUBMIT, not a dispatch; inside an vulkore::Batch a dispatch costs ~0.006 ms, so
// a second one really is nearly free. And the redundant sum is not cache-cheap
// the way I assumed — it is 29x the trivial kernel's cost. Over Gemma 3 1B's 52
// norms per token that is 8.2 ms vs 1.3 ms, on a ~42 ms/token budget.
//
// So: prefer rmsnorm_sumsq + rmsnorm_apply_gemma below. This one-pass form is
// kept because it needs no scratch buffer and is the right shape when a norm
// must stand alone outside a Batch (where the submit cost really does dominate),
// and because the phone's Qualcomm driver has a different dispatch cost than
// this laptop's turnip and has not been measured yet.
kernel void rmsnorm(global float* out, global const float* in,
                    global const float* w, uint n, float eps) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float ss = 0.0f;
    for (uint k = 0; k < n; ++k) ss += in[k] * in[k];
    float scale = rsqrt(ss / (float)n + eps);
    out[i] = in[i] * scale * w[i];
}

// Gemma scales by (1 + w) rather than w — the norm weights are stored as an
// offset from identity, so a zero-initialised weight is a no-op. Getting this
// wrong produces near-zero activations and fluent-looking garbage, so it is a
// separate entry point rather than a flag nobody sets.
kernel void rmsnorm_gemma(global float* out, global const float* in,
                          global const float* w, uint n, float eps) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float ss = 0.0f;
    for (uint k = 0; k < n; ++k) ss += in[k] * in[k];
    float scale = rsqrt(ss / (float)n + eps);
    out[i] = in[i] * scale * (1.0f + w[i]);
}

// ---- two-pass RMSNorm ------------------------------------------------------
// The redundant-sum kernels above are ONE dispatch doing n^2 work. These are
// TWO dispatches doing n + n*NPARTS work. Which wins depends entirely on how
// expensive a dispatch is, and that turned out to be device-specific enough to
// be worth measuring rather than assuming — see the numbers in
// agent-docs/llm-transformer-kernels.md. Both are kept because the answer
// differs between the turnip laptop and the Qualcomm phone.
//
// Pass 1: NPARTS threads each sum a strided slice into partial[].
// Pass 2: every thread sums the NPARTS partials (cheap) and applies the norm.
// The partial buffer is NPARTS floats and must be sized by the host; a barrier
// between the two dispatches is what vulkore::Batch already emits.
kernel void rmsnorm_sumsq(global float* partial, global const float* in,
                          uint n, uint nparts) {
    uint p = get_global_id(0);
    if (p >= nparts) return;
    float acc = 0.0f;
    for (uint k = p; k < n; k += nparts) acc += in[k] * in[k];
    partial[p] = acc;
}

// Gemma (1+w) convention, matching rmsnorm_gemma.
kernel void rmsnorm_apply_gemma(global float* out, global const float* in,
                                global const float* w,
                                global const float* partial, uint n,
                                uint nparts, float eps) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float ss = 0.0f;
    for (uint p = 0; p < nparts; ++p) ss += partial[p];
    float scale = rsqrt(ss / (float)n + eps);
    out[i] = in[i] * scale * (1.0f + w[i]);
}

// Gemma 3 applies QK-norm: an RMSNorm over each head's head_dim slice of q and
// k, before RoPE, with a weight shared across heads. Redundant-sum again, but
// now over head_dim (256) instead of n, so the waste is 256x smaller.
// Grid: n_heads * head_dim.
kernel void rmsnorm_heads_gemma(global float* out, global const float* in,
                                global const float* w, uint n_heads,
                                uint head_dim, float eps) {
    uint i = get_global_id(0);
    uint total = n_heads * head_dim;
    if (i >= total) return;
    uint base = (i / head_dim) * head_dim;
    uint d = i - base;
    float ss = 0.0f;
    for (uint k = 0; k < head_dim; ++k) ss += in[base + k] * in[base + k];
    float scale = rsqrt(ss / (float)head_dim + eps);
    out[i] = in[i] * scale * (1.0f + w[d]);
}

// ===========================================================================
// RoPE
// ===========================================================================
// Rotate-half convention (HF Gemma/Llama): the head vector splits in two and
// element d < head_dim/2 pairs with d + head_dim/2:
//   out[d]      = x[d]      * cos - x[d+h] * sin
//   out[d+h]    = x[d+h]    * cos + x[d]   * sin
// with angle = pos * theta^(-2d/head_dim).
//
// NOT the interleaved (d, d+1) convention some GGUF exporters use. Picking the
// wrong one is silent: the model still runs, it just loses positional sense
// after a few tokens.
//
// One thread per PAIR, so grid = n_heads * head_dim/2 and each thread writes
// both halves. Writing one element per thread would make each thread read its
// partner and recompute the same angle for no gain.
//
// theta^(-2d/head_dim) is computed as exp2(-log2(theta) * 2d/head_dim) rather
// than pow(): an 8th-power Mandelbulb in this repo used pow/acos/atan2,
// compiled clean, passed spirv-val, bound successfully and then SILENTLY did
// not execute on Adreno. exp2/log2/sin/cos map to native Adreno instructions.
// This kernel IS verified to write its output (see examples/llm/kernel_tests).
kernel void rope(global float* x, uint n_heads, uint head_dim, uint pos,
                 float theta) {
    uint i = get_global_id(0);
    uint half_dim = head_dim >> 1;
    uint total = n_heads * half_dim;
    if (i >= total) return;

    uint h = i / half_dim;
    uint d = i - h * half_dim;
    uint base = h * head_dim + d;

    float frac = (float)(2u * d) / (float)head_dim;
    float inv = exp2(-log2(theta) * frac);
    float ang = (float)pos * inv;
    float c = cos(ang);
    float s = sin(ang);

    float lo = x[base];
    float hi = x[base + half_dim];
    x[base] = lo * c - hi * s;
    x[base + half_dim] = lo * s + hi * c;
}

// Same rotation with the angle table supplied by the host. A real decode loop
// builds cos/sin once per position (or once for the whole context) and reuses
// it across all 26 layers and both q and k, which removes 2 trig ops per
// element and removes any doubt about transcendental support on a given
// driver. cos_t/sin_t are head_dim/2 long, indexed by d.
kernel void rope_cached(global float* x, global const float* cos_t,
                        global const float* sin_t, uint n_heads,
                        uint head_dim) {
    uint i = get_global_id(0);
    uint half_dim = head_dim >> 1;
    uint total = n_heads * half_dim;
    if (i >= total) return;

    uint h = i / half_dim;
    uint d = i - h * half_dim;
    uint base = h * head_dim + d;

    float c = cos_t[d];
    float s = sin_t[d];
    float lo = x[base];
    float hi = x[base + half_dim];
    x[base] = lo * c - hi * s;
    x[base + half_dim] = lo * s + hi * c;
}

// ===========================================================================
// Gated MLP activations
// ===========================================================================
// tanh is expressed through exp rather than the tanh builtin, and the argument
// is clamped first: exp(-2z) overflows to inf for z <= -44, and (1-inf)/(1+inf)
// is NaN. |z| > 10 is already tanh = +-1 to within fp32, so clamping costs
// nothing and removes the NaN cliff entirely.
static float tanh_approx(float z) {
    z = clamp(z, -10.0f, 10.0f);
    float e = exp(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
}

// GeGLU with the tanh-approximate GELU — this is `gelu_pytorch_tanh`, which is
// what Gemma's config specifies. The exact (erf) GELU differs by ~1e-3 in the
// tails, which is above our 1e-4 parity bar, so the CPU reference must use the
// same approximation. out = gelu(gate) * up.
kernel void geglu(global float* out, global const float* gate,
                  global const float* up, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float g = gate[i];
    float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    float gelu = 0.5f * g * (1.0f + tanh_approx(inner));
    out[i] = gelu * up[i];
}

// SwiGLU (Llama/Mistral): out = (g * sigmoid(g)) * up. Not used by Gemma, but
// the two families differ only in this kernel, so having both makes the decode
// path architecture-agnostic. Same clamp reasoning as tanh_approx.
kernel void swiglu(global float* out, global const float* gate,
                   global const float* up, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + exp(-clamp(g, -30.0f, 30.0f)));
    out[i] = g * sig * up[i];
}

// Fused variant: gate and up projections written into ONE buffer of 2*n by a
// single matvec (gate at [0,n), up at [n,2n)). Saves a dispatch and lets the
// two projections share one weight stream. n here is the HALF length.
kernel void geglu_fused(global float* out, global const float* gate_up,
                        uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float g = gate_up[i];
    float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    float gelu = 0.5f * g * (1.0f + tanh_approx(inner));
    out[i] = gelu * gate_up[n + i];
}

// ===========================================================================
// Attention
// ===========================================================================
// scores[h][j] = scale * dot(q[h], k[kv_start + j][kv_head(h)])  j in [0, span)
//
// Grouped-query attention: Gemma 3 1B has 4 q heads sharing 1 kv head, so
// kv_head(h) = h / (n_heads / kv_heads). With kv_heads == n_heads this
// degenerates to standard MHA at no cost.
//
// SLIDING WINDOW. `kv_start` is the absolute cache index the score row starts
// at and `span` is how many keys it covers, so a windowed layer is expressed
// as a CONTIGUOUS SUB-RANGE of the cache rather than as a full-length row with
// a -INFINITY mask. That matters for both correctness and cost:
//
//   * Gemma 3 makes 5 of every 6 layers local with a 512-position window
//     (layer i is global iff (i+1) % 6 == 0). A decode query at `pos` attends
//     t in (pos - window, pos], i.e. kv_start = max(0, seq_len - window),
//     span = seq_len - kv_start. Passing kv_start = 0, span = seq_len gives
//     the old unmasked global behaviour bit-for-bit.
//   * Because the live keys of a decode step are always the MOST RECENT ones,
//     the window is a suffix and needs no mask at all — the work simply
//     shrinks. Past 512 positions the local layers stop growing, so attention
//     cost per token goes from O(seq_len) on 26 layers to O(seq_len) on 4 and
//     O(512) on 22. Masking with -INFINITY instead would have been correct and
//     just as slow as before.
//
// Grid: n_heads * span — one thread per score, head_dim (256) multiply-adds
// each. The k reads are the only real traffic and they are shared across the
// 4 q heads of a group.
kernel void attention_scores(global float* scores, global const float* q,
                             global const float* kcache, uint n_heads,
                             uint kv_heads, uint head_dim, uint span,
                             float scale, uint kv_start) {
    uint i = get_global_id(0);
    uint total = n_heads * span;
    if (i >= total) return;

    uint h = i / span;
    uint j = i - h * span;
    uint group = n_heads / kv_heads;
    uint kvh = h / group;

    uint qb = h * head_dim;
    uint kb = ((kv_start + j) * kv_heads + kvh) * head_dim;
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; ++d) acc += q[qb + d] * kcache[kb + d];
    scores[i] = acc * scale;
}

// Row-wise softmax over the score matrix, ONE THREAD PER ROW.
//
// This looks wrong and is deliberate. A row is seq_len wide and there are only
// n_heads rows (4 for Gemma 3 1B), so the parallel version needs either a
// __local reduction (rejected by clspv) or three dispatches (max, sum, scale)
// at 0.31 ms each. One thread walking a few-thousand-element row costs far less
// than the two extra submits, and the whole score matrix is n_heads*seq_len
// floats — 8 KB at seq_len 512, i.e. cache-resident.
//
// Numerically the standard max-subtracted form: exp(x - rowmax) never
// overflows and the row sum is >= 1.
kernel void softmax_rows(global float* x, uint n_rows, uint n_cols) {
    uint r = get_global_id(0);
    if (r >= n_rows) return;
    uint base = r * n_cols;

    float m = -INFINITY;
    for (uint j = 0; j < n_cols; ++j) m = fmax(m, x[base + j]);

    float sum = 0.0f;
    for (uint j = 0; j < n_cols; ++j) {
        float e = exp(x[base + j] - m);
        x[base + j] = e;
        sum += e;
    }

    float inv = 1.0f / sum;
    for (uint j = 0; j < n_cols; ++j) x[base + j] *= inv;
}

// ---------------------------------------------------------------------------
// PARALLEL softmax_rows — three passes through GLOBAL memory.
//
// `softmax_rows` above is one thread per row. With n_heads = 4 that is FOUR
// threads on a GPU with thousands of lanes, and each of them walks the row
// three times. Measured per dispatch on an Adreno 840 it is the single largest
// depth-dependent term in a Gemma 3 1B decode step: 0.018 ms at span 41, 0.167
// ms at span 512 (which all 22 local layers reach), 2.51 ms at span 8288. It is
// not bandwidth — 4 threads cannot generate enough outstanding loads to use the
// device's ~64 GB/s, so the whole cost is exposed memory latency.
//
// The shape is the same two-pass-through-global-memory reduction that already
// worked twice in this repo (the parallel argmax tree, 31.5 -> 0.08 ms, and the
// two-pass RMSNorm). `__local` is NOT an option: clspv emits
// WorkgroupVariableSize and vulkore::Program rejects the module.
//
//   softmax_rows_partial   grid n_rows*nparts   per-chunk max + sum-of-exp
//   softmax_rows_finish    grid n_rows          combine chunks -> (max, 1/sum)
//   softmax_rows_scale     grid n_rows*n_cols   one thread per element
//
// Chunks are STRIDED (j += nparts), not contiguous blocks, so adjacent threads
// read adjacent floats and a wavefront's loads coalesce.
//
// `nparts` wants to be ~sqrt(n_cols): pass 1 costs n_cols/nparts iterations on
// n_rows*nparts threads while pass 2 costs nparts iterations on n_rows threads,
// so pushing nparts up just moves the serial scan from pass 1 into pass 2.
kernel void softmax_rows_partial(global float* pmax, global float* psum,
                                 global const float* x, uint n_rows,
                                 uint n_cols, uint nparts) {
    uint i = get_global_id(0);
    if (i >= n_rows * nparts) return;
    uint r = i / nparts;
    uint p = i - r * nparts;
    uint base = r * n_cols;

    float m = -INFINITY;
    for (uint j = p; j < n_cols; j += nparts) m = fmax(m, x[base + j]);

    // An empty chunk (nparts > n_cols) or a fully masked one leaves m at
    // -INFINITY; emit (−INF, 0) and let the finaliser drop it. Computing
    // exp(x - -INF) here would produce NaN.
    float s = 0.0f;
    if (isfinite(m))
        for (uint j = p; j < n_cols; j += nparts) s += exp(x[base + j] - m);

    pmax[i] = m;
    psum[i] = s;
}

// Standard stable online-softmax combine: rescale every chunk's sum from its
// own max onto the global max. stats[2r] = row max, stats[2r+1] = 1/rowsum —
// the RECIPROCAL, so the per-element pass multiplies instead of divides.
kernel void softmax_rows_finish(global float* stats, global const float* pmax,
                                global const float* psum, uint n_rows,
                                uint nparts) {
    uint r = get_global_id(0);
    if (r >= n_rows) return;
    uint b = r * nparts;

    float M = -INFINITY;
    for (uint p = 0; p < nparts; ++p) M = fmax(M, pmax[b + p]);
    if (!isfinite(M)) { stats[2u * r] = 0.0f; stats[2u * r + 1u] = 1.0f; return; }

    float S = 0.0f;
    for (uint p = 0; p < nparts; ++p) {
        float pm = pmax[b + p];
        if (isfinite(pm)) S += psum[b + p] * exp(pm - M);
    }

    stats[2u * r]      = M;
    stats[2u * r + 1u] = (S > 0.0f) ? (1.0f / S) : 1.0f;
}

kernel void softmax_rows_scale(global float* x, global const float* stats,
                               uint n_rows, uint n_cols) {
    uint i = get_global_id(0);
    if (i >= n_rows * n_cols) return;
    uint r = i / n_cols;
    x[i] = exp(x[i] - stats[2u * r]) * stats[2u * r + 1u];
}

// ---------------------------------------------------------------------------
// MULTI-QUERY attention_scores: one thread per KEY POSITION, all heads at once.
//
// `attention_scores` above puts one thread on each (head, key) pair and loops
// over head_dim. Two costs follow, both invisible until the span is thousands
// wide:
//
//   * Gemma 3 1B is multi-query (kv_heads = 1), so all four q heads dot against
//     the SAME key vector — and the generic kernel reads that vector four
//     times. At span 8288 that is 34 MiB of KV traffic per dispatch where 8.5
//     MiB would do.
//   * Adjacent threads differ in the KEY index, so their loads are head_dim*4 =
//     1024 bytes apart. Each thread does consume its whole run, so this is not
//     read amplification — but it is one scalar load instruction per float, and
//     `llm-performance.md` already established on this device that load
//     INSTRUCTION COUNT, not coalescing, is what the Adreno cares about.
//
// So: one thread per key, reading that key ONCE as float4 (4x fewer loads) and
// accumulating four dot products in scalar registers (4x less traffic). The
// four accumulators are separate named variables, not an array — a private
// array with a dynamic index makes clspv allocate real scratch per thread and
// turns a memory-bound pass into a spill-bound one.
//
// REQUIRES n_heads == 4 and kv_heads == 1; the host falls back to the generic
// `attention_scores` for any other shape. head_dim must be a multiple of 4.
// Output layout is identical: scores[h*span + j].
kernel void attention_scores_mq4(global float* scores, global const float4* q,
                                 global const float4* kcache, uint head_dim,
                                 uint span, float scale, uint kv_start) {
    uint j = get_global_id(0);
    if (j >= span) return;

    uint hd4 = head_dim >> 2;
    uint kb = (kv_start + j) * hd4;      // kv_heads == 1, so no head term
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint d = 0; d < hd4; ++d) {
        float4 kv = kcache[kb + d];
        a0 += dot(q[d], kv);
        a1 += dot(q[hd4 + d], kv);
        a2 += dot(q[2u * hd4 + d], kv);
        a3 += dot(q[3u * hd4 + d], kv);
    }
    scores[j]              = a0 * scale;
    scores[span + j]       = a1 * scale;
    scores[2u * span + j]  = a2 * scale;
    scores[3u * span + j]  = a3 * scale;
}

// out[h][d] = sum_j probs[h][j] * v[kv_start + j][kv_head(h)][d]
//
// The `kv_start`/`span` pair is the same sliding-window contract as
// attention_scores above and MUST be passed the identical values: `probs` is
// the softmaxed output of that kernel, so its row stride is `span` and its
// column j means absolute cache position kv_start + j. kv_start = 0,
// span = seq_len is the old global behaviour.
//
// Grid: n_heads * head_dim. Each thread walks the attended span for one output
// element, striding through the v cache by kv_heads*head_dim. The stride hurts
// coalescing, but adjacent threads (adjacent d) hit adjacent floats, so each
// wavefront still reads contiguous 256-float runs.
kernel void attention_apply(global float* out, global const float* probs,
                            global const float* vcache, uint n_heads,
                            uint kv_heads, uint head_dim, uint span,
                            uint kv_start) {
    uint i = get_global_id(0);
    uint total = n_heads * head_dim;
    if (i >= total) return;

    uint h = i / head_dim;
    uint d = i - h * head_dim;
    uint group = n_heads / kv_heads;
    uint kvh = h / group;

    uint pb = h * span;
    uint vstride = kv_heads * head_dim;
    uint vb = (kv_start * kv_heads + kvh) * head_dim + d;
    float acc = 0.0f;
    for (uint j = 0; j < span; ++j) acc += probs[pb + j] * vcache[vb + j * vstride];
    out[i] = acc;
}

// ---------------------------------------------------------------------------
// MULTI-QUERY attention_apply, split over the span.
//
// Same multi-query observation as attention_scores_mq4: with kv_heads = 1 all
// four heads weight the SAME v vectors, and the generic kernel reads the v
// cache once per head. Reading it once and accumulating four outputs cuts the
// traffic 4x.
//
// The generic kernel's grid is n_heads*head_dim = 1024 threads; sharing v
// across heads would drop that to 256 and trade the traffic saving for an
// occupancy loss. So the span is ALSO split `split` ways, which puts the thread
// count back at 256*split and makes each thread walk a shorter run. That needs
// a reduce pass, exactly like the split-k matvecs.
//
// partial layout: partial[(s*4 + h)*head_dim + d], reduced over s.
// REQUIRES n_heads == 4 and kv_heads == 1.
kernel void attention_apply_mq4(global float* partial, global const float* probs,
                                global const float* vcache, uint head_dim,
                                uint span, uint kv_start, uint split) {
    uint i = get_global_id(0);
    if (i >= head_dim * split) return;

    uint s = i / head_dim;
    uint d = i - s * head_dim;

    uint chunk = (span + split - 1u) / split;
    uint j0 = s * chunk;
    uint j1 = j0 + chunk;
    if (j1 > span) j1 = span;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    // Adjacent threads differ in d, i.e. by one float, so each wavefront reads
    // a contiguous run — the property the generic kernel already had and the
    // reason it was the cheapest of the three to begin with.
    uint vb = (kv_start + j0) * head_dim + d;
    for (uint j = j0; j < j1; ++j, vb += head_dim) {
        float v = vcache[vb];
        a0 += probs[j] * v;
        a1 += probs[span + j] * v;
        a2 += probs[2u * span + j] * v;
        a3 += probs[3u * span + j] * v;
    }

    uint ob = s * 4u * head_dim + d;
    partial[ob]                  = a0;
    partial[ob + head_dim]       = a1;
    partial[ob + 2u * head_dim]  = a2;
    partial[ob + 3u * head_dim]  = a3;
}

// Sum the `split` partial vectors. n = n_heads*head_dim.
kernel void attention_apply_reduce(global float* out, global const float* partial,
                                   uint n, uint split) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float acc = 0.0f;
    for (uint s = 0; s < split; ++s) acc += partial[s * n + i];
    out[i] = acc;
}

// ===========================================================================
// Plumbing
// ===========================================================================

// x += y. The residual stream update, twice per layer.
kernel void add_residual(global float* x, global const float* y, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    x[i] += y[i];
}

// out = a + b, non-destructive form for when the residual must be kept.
kernel void add_vec(global float* out, global const float* a,
                    global const float* b, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

// x *= s. Gemma multiplies the token embedding by sqrt(hidden) on the way in;
// also the generic place to fold an attention or logit scale.
kernel void mul_scalar(global float* x, float s, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    x[i] *= s;
}

// Append this token's k (or v) to the cache at `pos`, i.e. copy n floats to
// cache[pos*n .. pos*n+n). A dedicated kernel rather than a host-side copy
// because the source is already on the device and round-tripping it through
// the host would cost more than the dispatch.
kernel void kv_append(global float* cache, global const float* src, uint n,
                      uint pos) {
    uint i = get_global_id(0);
    if (i >= n) return;
    cache[pos * n + i] = src[i];
}

// Gather one row of the embedding table: out[i] = table[token*n + i].
// Vocab is 262144 x 1152 fp32, so this must stay a gather — materialising a
// one-hot matvec would read 1.2 GB to produce 1152 floats.
kernel void embed_lookup(global float* out, global const float* table,
                         uint token, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    // token*n peaks at 262144*1152 = 3.0e8, comfortably inside uint32 — no
    // size_t/Int64 promotion, which clspv would turn into a capability Adreno
    // need not support.
    out[i] = table[token * n + i];
}

// Greedy argmax over the logits, ONE THREAD, serial over vocab.
//
// 262144 elements in one thread is genuinely slow (~1 ms class), but it is one
// dispatch and it is exact. The parallel alternative is a two-pass reduction
// (partial argmax then reduce), which is the right answer if this ever shows up
// in a profile; it does not today, because the lm_head matvec that produces
// these logits reads 300 MB and takes far longer.
// out[0] = best index (as float), out[1] = best value.
kernel void argmax(global float* out, global const float* logits, uint n) {
    uint i = get_global_id(0);
    if (i >= 1u) return;
    uint best = 0;
    float bv = -INFINITY;
    for (uint k = 0; k < n; ++k) {
        float v = logits[k];
        if (v > bv) { bv = v; best = k; }
    }
    out[0] = (float)best;
    out[1] = bv;
}
