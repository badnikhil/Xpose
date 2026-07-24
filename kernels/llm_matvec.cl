// Vulkore LLM matvec — optimisation variants for int4 single-token decode.
//
// The baseline (kernels/llm.cl matvec_q4) measured 11.8 GB/s on an Adreno 840
// against 60-77 GB/s of LPDDR5X: ~18% of the hardware. Every kernel here is an
// attempt to close that gap; examples/llm/matvec_bench.cpp measures all of them
// against each other and checks each one's output against the baseline.
//
// The central problem the baseline has is NOT coalescing, it is LOAD-INSTRUCTION
// COUNT. Its packing puts 8 nibbles per uint along the COLUMN axis, so word
// (k*cols + j)/8 holds columns j..j+7 of row k. Thread j therefore issues one
// 4-byte load per k and uses 4 BITS of it; the other 7 nibbles belong to its 7
// neighbours. Across the matrix that is rows*cols load instructions to move
// rows*cols/2 BYTES — one load instruction per half byte of traffic. The
// coalescing is perfect and the instruction count is 16x what it needs to be.
//
// The fix is to pack along k so a single load carries weights that ONE thread
// consumes, while keeping neighbouring threads on neighbouring addresses. See
// the layout notes below.
//
// ABI per tests/kernels/README.md: storage-buffer args only, PODs in one push
// constant block, uniform workgroup, grid = GLOBAL threads (rounded up, so
// every kernel bound-checks). No __local — clspv emits WorkgroupVariableSize
// and Program throws — so multi-pass reductions go through global memory.

#define BLK 32u   // quantisation block along k: one fp32 scale per 32 weights
#define GRP 64u   // column-group width of the k-major layouts (= default wgsize)

// ---------------------------------------------------------------------------
// LAYOUTS
//
// A "COL" (baseline, kernels/llm.cl): word[(k*cols + j) >> 3], nibble (j&7).
//     1 load per k per thread, 4 useful bits per load.
//
// B "KMN" (naive k-major): word[j*(rows/8) + (k>>3)], nibble (k&7).
//     8 useful nibbles per load, but thread j and j+1 are rows/8 WORDS apart,
//     so a 64-wide wave touches 64 different cache lines per instruction.
//     Included to measure exactly how much that costs.
//
// C "KM" (group-interleaved k-major, uint): with g = j/GRP, t = j%GRP,
//     c = k/8, nc = rows/8:  word[(g*nc + c)*GRP + t], nibble (k&7).
//     8 useful nibbles per load AND consecutive threads on consecutive words
//     (a wave covers 256 contiguous bytes). 8x fewer loads than COL.
//
// D "KM4" (group-interleaved k-major, uint4): c4 = k/32, nc4 = rows/32,
//     uint4[(g*nc4 + c4)*GRP + t]; component (k>>3)&3, nibble (k&7).
//     32 weights per 16-byte load, wave covers 1024 contiguous bytes.
//     32x fewer load instructions than COL. One uint4 == one quant block,
//     so the scale lookup happens once per load.
//
// Scales stay [block][col] in every variant: thread j reads scale[b*cols + j],
// which is already perfectly coalesced.
// ---------------------------------------------------------------------------

// Dot of 8 packed nibbles (biased by 8 -> signed [-8,7]) against 8 floats.
inline float q4dot8(uint w, global const float* v) {
    float s;
    s  = v[0] * ((float)((w      ) & 15u) - 8.0f);
    s += v[1] * ((float)((w >>  4) & 15u) - 8.0f);
    s += v[2] * ((float)((w >>  8) & 15u) - 8.0f);
    s += v[3] * ((float)((w >> 12) & 15u) - 8.0f);
    s += v[4] * ((float)((w >> 16) & 15u) - 8.0f);
    s += v[5] * ((float)((w >> 20) & 15u) - 8.0f);
    s += v[6] * ((float)((w >> 24) & 15u) - 8.0f);
    s += v[7] * ((float)((w >> 28) & 15u) - 8.0f);
    return s;
}

// One uint4 == one BLK=32 quantisation block.
inline float q4dot32(uint4 w, global const float* v) {
    return q4dot8(w.x, v) + q4dot8(w.y, v + 8) +
           q4dot8(w.z, v + 16) + q4dot8(w.w, v + 24);
}

// ---------------------------------------------------------------------------
// A. Baseline, byte-identical maths to kernels/llm.cl matvec_q4.
// Duplicated here so every variant is timed inside one binary against one
// reference, with no cross-module compilation differences.
// ---------------------------------------------------------------------------
kernel void mv_ref(global float* out, global const float* in,
                   global const uint* wq, global const float* scale,
                   uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint shift = (j & 7u) * 4u;
    float acc = 0.0f;
    uint nblk = rows / BLK;
    for (uint b = 0; b < nblk; ++b) {
        float part = 0.0f;
        for (uint t = 0; t < BLK; ++t) {
            uint k = b * BLK + t;
            uint word = wq[(k * cols + j) >> 3];
            part += in[k] * ((float)((word >> shift) & 15u) - 8.0f);
        }
        acc += part * scale[b * cols + j];
    }
    out[j] = acc;
}

// Baseline + split-k. Retried now that vulkore::Batch amortises dispatch cost:
// the earlier attempt lost 40% purely because it doubled a 0.31 ms-per-launch
// submit count, which is no longer what a dispatch costs.
kernel void mv_ref_split(global float* partial, global const float* in,
                         global const uint* wq, global const float* scale,
                         uint rows, uint cols, uint split) {
    uint id = get_global_id(0);
    if (id >= cols * split) return;
    uint j = id % cols;
    uint p = id / cols;
    uint shift = (j & 7u) * 4u;
    uint nblk = rows / BLK;
    uint per = (nblk + split - 1u) / split;
    uint b0 = p * per;
    uint b1 = min(b0 + per, nblk);
    float acc = 0.0f;
    for (uint b = b0; b < b1; ++b) {
        float part = 0.0f;
        for (uint t = 0; t < BLK; ++t) {
            uint k = b * BLK + t;
            uint word = wq[(k * cols + j) >> 3];
            part += in[k] * ((float)((word >> shift) & 15u) - 8.0f);
        }
        acc += part * scale[b * cols + j];
    }
    partial[p * cols + j] = acc;
}

// Second pass for every split-k variant. Global memory, not __local.
kernel void mv_reduce(global float* out, global const float* partial,
                      uint cols, uint split) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    float acc = 0.0f;
    for (uint p = 0; p < split; ++p) acc += partial[p * cols + j];
    out[j] = acc;
}

// ---------------------------------------------------------------------------
// B. Naive k-major: contiguous per thread, strided across the wave.
// ---------------------------------------------------------------------------
kernel void mv_kmn(global float* out, global const float* in,
                   global const uint* wq, global const float* scale,
                   uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint nc = rows >> 3;
    global const uint* row = wq + (size_t)j * nc;
    uint nblk = rows / BLK;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b) {
        uint c = b << 2;   // 4 words per 32-weight block
        float part = q4dot8(row[c    ], in + b * BLK)
                   + q4dot8(row[c + 1], in + b * BLK + 8)
                   + q4dot8(row[c + 2], in + b * BLK + 16)
                   + q4dot8(row[c + 3], in + b * BLK + 24);
        acc += part * scale[b * cols + j];
    }
    out[j] = acc;
}

// ---------------------------------------------------------------------------
// C. Group-interleaved k-major, 4-byte loads.
// ---------------------------------------------------------------------------
kernel void mv_km(global float* out, global const float* in,
                  global const uint* wq, global const float* scale,
                  uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint g = j / GRP, t = j % GRP;
    uint nc = rows >> 3;
    uint nblk = rows / BLK;
    size_t base = (size_t)g * nc * GRP + t;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b) {
        uint c = b << 2;
        float part = q4dot8(wq[base + (size_t)(c    ) * GRP], in + b * BLK)
                   + q4dot8(wq[base + (size_t)(c + 1) * GRP], in + b * BLK + 8)
                   + q4dot8(wq[base + (size_t)(c + 2) * GRP], in + b * BLK + 16)
                   + q4dot8(wq[base + (size_t)(c + 3) * GRP], in + b * BLK + 24);
        acc += part * scale[b * cols + j];
    }
    out[j] = acc;
}

// ---------------------------------------------------------------------------
// D. Group-interleaved k-major, 16-byte loads. The main event.
// ---------------------------------------------------------------------------
kernel void mv_km4(global float* out, global const float* in,
                   global const uint4* wq, global const float* scale,
                   uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint g = j / GRP, t = j % GRP;
    uint nblk = rows / BLK;               // == rows/32 == uint4 count per col
    size_t base = (size_t)g * nblk * GRP + t;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b)
        acc += q4dot32(wq[base + (size_t)b * GRP], in + b * BLK)
             * scale[b * cols + j];
    out[j] = acc;
}

// Same, with the workgroup size pinned instead of the runtime's default 64.
__attribute__((reqd_work_group_size(128, 1, 1)))
kernel void mv_km4_wg128(global float* out, global const float* in,
                         global const uint4* wq, global const float* scale,
                         uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint g = j / GRP, t = j % GRP;
    uint nblk = rows / BLK;
    size_t base = (size_t)g * nblk * GRP + t;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b)
        acc += q4dot32(wq[base + (size_t)b * GRP], in + b * BLK)
             * scale[b * cols + j];
    out[j] = acc;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
kernel void mv_km4_wg256(global float* out, global const float* in,
                         global const uint4* wq, global const float* scale,
                         uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint g = j / GRP, t = j % GRP;
    uint nblk = rows / BLK;
    size_t base = (size_t)g * nblk * GRP + t;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b)
        acc += q4dot32(wq[base + (size_t)b * GRP], in + b * BLK)
             * scale[b * cols + j];
    out[j] = acc;
}

__attribute__((reqd_work_group_size(32, 1, 1)))
kernel void mv_km4_wg32(global float* out, global const float* in,
                        global const uint4* wq, global const float* scale,
                        uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint g = j / GRP, t = j % GRP;
    uint nblk = rows / BLK;
    size_t base = (size_t)g * nblk * GRP + t;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b)
        acc += q4dot32(wq[base + (size_t)b * GRP], in + b * BLK)
             * scale[b * cols + j];
    out[j] = acc;
}

// ---------------------------------------------------------------------------
// D2/D4. Register blocking: one thread owns CPT output columns, so the
// in[] reads and the loop bookkeeping are shared across CPT weight streams and
// CPT loads are in flight at once. Thread i owns columns i, i+T, i+2T, ...
// with T = ceil(cols/CPT), which keeps consecutive threads on consecutive
// columns (and therefore consecutive words) for every one of the CPT streams.
// ---------------------------------------------------------------------------
kernel void mv_km4_c2(global float* out, global const float* in,
                      global const uint4* wq, global const float* scale,
                      uint rows, uint cols) {
    uint i = get_global_id(0);
    uint T = (cols + 1u) / 2u;
    if (i >= T) return;
    uint nblk = rows / BLK;

    uint j0 = i, j1 = i + T;
    size_t b0 = (size_t)(j0 / GRP) * nblk * GRP + (j0 % GRP);
    size_t b1 = (size_t)(j1 / GRP) * nblk * GRP + (j1 % GRP);
    float a0 = 0.0f, a1 = 0.0f;
    bool ok1 = j1 < cols;
    for (uint b = 0; b < nblk; ++b) {
        global const float* v = in + b * BLK;
        size_t off = (size_t)b * GRP;
        a0 += q4dot32(wq[b0 + off], v) * scale[b * cols + j0];
        if (ok1) a1 += q4dot32(wq[b1 + off], v) * scale[b * cols + j1];
    }
    out[j0] = a0;
    if (ok1) out[j1] = a1;
}

// The two winners on the laptop Adreno were wg32 (narrow layers) and 2-column
// blocking (wide layers); this is both at once.
__attribute__((reqd_work_group_size(32, 1, 1)))
kernel void mv_km4_c2_wg32(global float* out, global const float* in,
                           global const uint4* wq, global const float* scale,
                           uint rows, uint cols) {
    uint i = get_global_id(0);
    uint T = (cols + 1u) / 2u;
    if (i >= T) return;
    uint nblk = rows / BLK;

    uint j0 = i, j1 = i + T;
    size_t b0 = (size_t)(j0 / GRP) * nblk * GRP + (j0 % GRP);
    size_t b1 = (size_t)(j1 / GRP) * nblk * GRP + (j1 % GRP);
    float a0 = 0.0f, a1 = 0.0f;
    bool ok1 = j1 < cols;
    for (uint b = 0; b < nblk; ++b) {
        global const float* v = in + b * BLK;
        size_t off = (size_t)b * GRP;
        a0 += q4dot32(wq[b0 + off], v) * scale[b * cols + j0];
        if (ok1) a1 += q4dot32(wq[b1 + off], v) * scale[b * cols + j1];
    }
    out[j0] = a0;
    if (ok1) out[j1] = a1;
}

kernel void mv_km4_c4(global float* out, global const float* in,
                      global const uint4* wq, global const float* scale,
                      uint rows, uint cols) {
    uint i = get_global_id(0);
    uint T = (cols + 3u) / 4u;
    if (i >= T) return;
    uint nblk = rows / BLK;

    uint j[4];
    size_t bs[4];
    float a[4];
    for (uint m = 0; m < 4u; ++m) {
        j[m] = i + m * T;
        uint jj = min(j[m], cols - 1u);          // clamp so the address is safe
        bs[m] = (size_t)(jj / GRP) * nblk * GRP + (jj % GRP);
        a[m] = 0.0f;
    }
    for (uint b = 0; b < nblk; ++b) {
        global const float* v = in + b * BLK;
        size_t off = (size_t)b * GRP;
        for (uint m = 0; m < 4u; ++m)
            a[m] += q4dot32(wq[bs[m] + off], v) * scale[b * cols + min(j[m], cols - 1u)];
    }
    for (uint m = 0; m < 4u; ++m)
        if (j[m] < cols) out[j[m]] = a[m];
}

// ---------------------------------------------------------------------------
// KM4 + split-k. Narrow layers (o_proj is 1152 columns = 18 waves) cannot fill
// an Adreno with one thread per column no matter how good the loads are; split
// the k axis SPLIT ways for cols*SPLIT threads, then reduce with mv_reduce.
// ---------------------------------------------------------------------------
kernel void mv_km4_split(global float* partial, global const float* in,
                         global const uint4* wq, global const float* scale,
                         uint rows, uint cols, uint split) {
    uint id = get_global_id(0);
    if (id >= cols * split) return;
    uint j = id % cols;
    uint p = id / cols;
    uint nblk = rows / BLK;
    uint per = (nblk + split - 1u) / split;
    uint s0 = p * per;
    uint s1 = min(s0 + per, nblk);
    size_t base = (size_t)(j / GRP) * nblk * GRP + (j % GRP);
    float acc = 0.0f;
    for (uint b = s0; b < s1; ++b)
        acc += q4dot32(wq[base + (size_t)b * GRP], in + b * BLK)
             * scale[b * cols + j];
    partial[p * cols + j] = acc;
}

// ---------------------------------------------------------------------------
// Pure bandwidth probe with 16-byte loads: the hard ceiling every matvec
// number above sits under, measured the same way on the same device.
// n is the uint4 count; each thread strides so the wave stays contiguous.
// ---------------------------------------------------------------------------
// The guard on the store MUST depend on acc. The obvious `if (i == 0)` is a
// trap: acc is then dead in every other thread, and a compiler is free to sink
// the whole loop into the branch so 4095 of 4096 threads read nothing at all.
// Mesa turnip does exactly that — the probe reported 419 GB/s on a part whose
// LPDDR5X tops out near 135, which is how the bug was caught. Comparing against
// a value the sum cannot take keeps every thread's loads live without ever
// storing, so the measured traffic is the traffic that actually happened.
kernel void stream4(global float* out, global const uint4* src, uint n,
                    uint threads) {
    uint i = get_global_id(0);
    if (i >= threads) return;
    uint4 acc = (uint4)(0u, 0u, 0u, 0u);
    for (uint k = i; k < n; k += threads) acc += src[k];
    uint s = acc.x + acc.y + acc.z + acc.w;
    if (s == 0xDEADBEEFu) out[i & 3u] = (float)s;
}

// ---------------------------------------------------------------------------
// E. KM4 with BF16 BLOCK SCALES, four scales per uint2 load.
//
// Everything above reads one fp32 scale per 32-weight block, i.e. 4 bytes of
// scale for every 16 bytes of nibbles. That is a flat +25% on ALL weight
// traffic (119.2 of the 595.9 MiB a Gemma 3 1B token moves) and one extra load
// instruction per block, and BOTH are avoidable: the scale is already only
// meaningful to ~4 bits of int4 quantisation error, so fp32 precision on it is
// two orders of magnitude more than the format can use.
//
// bf16 rather than fp16 deliberately. fp16 needs a real decoder (exponent
// rebias, a subnormal normalisation loop) — llm_quant.cl has one and it is
// ~15 instructions. bf16 is the TOP 16 BITS of the fp32 word, so decoding is a
// single shift and it keeps fp32's exponent range, so no scale can overflow or
// flush to zero however the model was trained. The mantissa drops to 8 bits =
// 0.4% relative, against int4's ~7% quantisation step: invisible.
//
// Four scales per uint2 (blocks 4q..4q+3, low/high halves of .x then .y) so the
// scale stream costs ONE load per four weight loads instead of one per one:
// per 128 weights the thread issues 4 uint4 + 1 uint2 = 5 loads for 72 bytes,
// against 8 loads for 80 bytes before. 37.5% fewer load instructions and 10%
// fewer bytes, which is the axis llm-performance.md found to be decisive.
//
// Layout: sc[(b/4)*cols + j] as uint2. Consecutive j are consecutive uint2s, so
// a wave still reads contiguous memory. Requires rows % 128 == 0 (nblk % 4).
// ---------------------------------------------------------------------------

inline float bf16lo(uint w) { return as_float(w << 16); }
inline float bf16hi(uint w) { return as_float(w & 0xFFFF0000u); }

// One group of four blocks: 4 uint4 weight loads against one already-loaded
// uint2 of scales.
inline float km4_quad(global const uint4* wq, size_t base, global const float* in,
                      uint b, uint2 s) {
    float a;
    a  = q4dot32(wq[base + (size_t)(b     ) * GRP], in + (b     ) * BLK) * bf16lo(s.x);
    a += q4dot32(wq[base + (size_t)(b + 1u) * GRP], in + (b + 1u) * BLK) * bf16hi(s.x);
    a += q4dot32(wq[base + (size_t)(b + 2u) * GRP], in + (b + 2u) * BLK) * bf16lo(s.y);
    a += q4dot32(wq[base + (size_t)(b + 3u) * GRP], in + (b + 3u) * BLK) * bf16hi(s.y);
    return a;
}

kernel void mv_km4_bs(global float* out, global const float* in,
                      global const uint4* wq, global const uint2* sc,
                      uint rows, uint cols) {
    uint j = get_global_id(0);
    if (j >= cols) return;
    uint nblk = rows / BLK;
    uint nq   = nblk / 4u;                 // groups of four blocks
    size_t base = (size_t)(j / GRP) * nblk * GRP + (j % GRP);
    float acc = 0.0f;
    for (uint q = 0; q < nq; ++q)
        acc += km4_quad(wq, base, in, q * 4u, sc[(size_t)q * cols + j]);
    out[j] = acc;
}

// Split-k twin. The split is over QUAD groups, not blocks, so every partition
// boundary stays aligned to the scale packing and no thread reloads a uint2 it
// only half uses.
kernel void mv_km4_bs_split(global float* partial, global const float* in,
                            global const uint4* wq, global const uint2* sc,
                            uint rows, uint cols, uint split) {
    uint id = get_global_id(0);
    if (id >= cols * split) return;
    uint j = id % cols;
    uint p = id / cols;
    uint nblk = rows / BLK;
    uint nq   = nblk / 4u;
    uint per  = (nq + split - 1u) / split;
    uint q0 = p * per;
    uint q1 = min(q0 + per, nq);
    size_t base = (size_t)(j / GRP) * nblk * GRP + (j % GRP);
    float acc = 0.0f;
    for (uint q = q0; q < q1; ++q)
        acc += km4_quad(wq, base, in, q * 4u, sc[(size_t)q * cols + j]);
    partial[p * cols + j] = acc;
}
