// GGUF K-quant dequantisation and fused dequant-matvec.
//
// The kernels in llm.cl assume a made-up Q4_0-like layout invented for
// benchmarking: one fp32 scale per (block, column) in a side array, nibbles
// biased by 8. No real GGUF file looks like that. A Gemma 3 1B Q4_K_M holds
// its weights as **Q4_K** and its output/embedding matrix as **Q6_K**, both of
// which are super-block formats with packed 6-bit sub-scales. This file
// implements those two plus Q8_0, straight from the layouts in llama.cpp's
// ggml-quants.c (dequantize_row_q4_K / _q6_K / _q8_0, get_scale_min_k4).
//
// Exact byte layouts: agent-docs/gguf-quant-kernels.md.
//
// TWO ABI CONSTRAINTS SHAPE EVERY LINE HERE (tests/kernels/README.md):
//
//  1. vulkore::Buffer static_asserts a 4-byte-aligned element type, so there is
//     no uchar/ushort buffer. Quantised bytes arrive as a `global const uint*`
//     view of the raw tensor blob and every field is dug out with shifts and
//     masks. Q4_K's super-block is 144 B (36 words) so its blocks stay
//     word-aligned and the 4-bit quants can be read a word at a time; Q6_K
//     (210 B) and Q8_0 (34 B) are NOT multiples of 4, so from the second block
//     onwards every field sits at an arbitrary byte phase and must go through
//     u8()/u16(). That costs a redundant load per byte and is the price of the
//     alignment rule.
//
//  2. `__local` is rejected by the toolchain (clspv emits WorkgroupVariableSize
//     and Program throws), so there are no workgroup reductions: the fused
//     matvecs give one thread a whole output row and keep the accumulator in a
//     register. That is also the right shape for decode, where the matrix is
//     tall and the "vector" is a single token.
//
// Grids are rounded up to whole workgroups, so every kernel bound-checks.

// ---------------------------------------------------------------------------
// Byte / half-word access into a uint view of the blob.
// ---------------------------------------------------------------------------
// `off` is a BYTE offset from the start of the buffer. u16 is deliberately two
// u8 loads rather than a 16-bit read: a Q6_K fp16 scale lands on an odd byte
// often enough that a shifted 16-bit path would be wrong half the time.
static inline uint u8(global const uint* p, uint off) {
    return (p[off >> 2] >> ((off & 3u) * 8u)) & 0xFFu;
}
static inline uint u16(global const uint* p, uint off) {
    return u8(p, off) | (u8(p, off + 1u) << 8);
}
// int8 stored as two's complement in a byte.
static inline int i8(global const uint* p, uint off) {
    return (int)(u8(p, off) ^ 0x80u) - 128;
}

// fp16 -> fp32 by hand. There is no half type to lean on here, and pow/exp2
// tricks are a bad bet on Adreno (pow/acos/atan2 have silently not executed in
// this repo before), so this is pure integer bit surgery finished with a
// bitcast. Subnormals are normalised with a shift loop: GGUF scales are never
// subnormal in practice, but "in practice" is not a reason to be wrong, and
// the loop runs at most 10 times.
static inline float fp16_to_fp32(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint exp  = (h >> 10) & 0x1Fu;
    uint mant = h & 0x3FFu;
    uint bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;                      // +/- 0
        } else {
            uint m = mant;
            int  e = -1;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0u);
            m &= 0x3FFu;                      // drop the implicit bit
            bits = sign | ((uint)(112 - e) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);   // Inf / NaN
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);  // 127 - 15 = 112
    }
    return as_float(bits);
}

// Decode a raw fp16 array. Exists so the decoder above can be swept over all
// 65536 bit patterns on the DEVICE and compared with the host — every other
// kernel here multiplies by an fp16 scale, so a wrong decoder would show up as
// a diffuse scale error that is miserable to localise later.
kernel void fp16_decode(global float* out, global const uint* src, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    out[i] = fp16_to_fp32(u16(src, i * 2u));
}

// ---------------------------------------------------------------------------
// Q8_0 — 34 B block of 32 weights: fp16 d, then 32 int8. w = d * q.
// ---------------------------------------------------------------------------
kernel void dequant_q8_0(global float* out, global const uint* src, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    uint base = (i >> 5) * 34u;               // block index * 34
    float d = fp16_to_fp32(u16(src, base));
    out[i] = d * (float)i8(src, base + 2u + (i & 31u));
}

// out[r] = sum_k w[r][k] * in[k], row length `cols` (multiple of 32).
kernel void matvec_q8_0(global float* out, global const float* in,
                        global const uint* w, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nblk = cols >> 5;
    uint base = r * nblk * 34u;               // start of this row's blocks
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b) {
        float d = fp16_to_fp32(u16(w, base));
        float s = 0.0f;
        for (uint l = 0; l < 32u; ++l)
            s += in[(b << 5) + l] * (float)i8(w, base + 2u + l);
        acc += d * s;                          // scale once per block, not per weight
        base += 34u;
    }
    out[r] = acc;
}

// ---------------------------------------------------------------------------
// Q5_0 — 22 B block of 32 weights:
//   [0:2]  fp16 d
//   [2:6]  qh, a little-endian u32 whose bit e is the 5th bit of weight e
//   [6:22] 16 B of low nibbles, laid out Q4_0-style
// w = d * (q - 16), q the 5-bit unsigned value.
//
// The trap is the INTERLEAVE. Byte j of qs holds weight j in its low nibble
// and weight j+16 in its high nibble — the two halves of the block, not
// adjacent weights. The reference writes that as shifts of (j) and (j+12)
// followed by `& 0x10`, which obscures it; the equivalent statement is simply
// "bit e of qh belongs to weight e", and that is what this code does. Reading
// the high bits as bit j / bit j+1 (i.e. assuming qs byte j holds weights 2j
// and 2j+1) produces a plausible-looking tensor that is quietly wrong for half
// its values, which is exactly the failure mode worth testing hard.
// ---------------------------------------------------------------------------
static inline float q5_0_value(global const uint* p, uint base, uint e) {
    uint qh   = (uint)(u8(p, base + 2u)) | (u8(p, base + 3u) << 8)
              | (u8(p, base + 4u) << 16) | (u8(p, base + 5u) << 24);
    uint byte = u8(p, base + 6u + (e & 15u));
    uint lo   = (e < 16u) ? (byte & 0xFu) : (byte >> 4);
    int  q    = (int)(lo | (((qh >> e) & 1u) << 4)) - 16;
    return (float)q;
}

kernel void dequant_q5_0(global float* out, global const uint* src, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    uint base = (i >> 5) * 22u;
    out[i] = fp16_to_fp32(u16(src, base)) * q5_0_value(src, base, i & 31u);
}

kernel void matvec_q5_0(global float* out, global const float* in,
                        global const uint* w, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nblk = cols >> 5;
    uint base = r * nblk * 22u;
    float acc = 0.0f;
    for (uint b = 0; b < nblk; ++b) {
        float d = fp16_to_fp32(u16(w, base));
        float s = 0.0f;
        for (uint e = 0; e < 32u; ++e)
            s += in[(b << 5) + e] * q5_0_value(w, base, e);
        acc += d * s;
        base += 22u;
    }
    out[r] = acc;
}

// ---------------------------------------------------------------------------
// Q4_K — 144 B super-block of 256 weights (8 sub-blocks of 32):
//   [0:2]   fp16 d      (scale of the sub-block SCALES)
//   [2:4]   fp16 dmin   (scale of the sub-block MINS)
//   [4:16]  12 B holding 8 six-bit scales + 8 six-bit mins
//   [16:144] 128 B of 4-bit quants
// w = d*scale[s]*q - dmin*min[s].  The quants are UNSIGNED with no bias; the
// zero point is carried by the per-sub-block min instead. Getting that wrong
// (biasing by 8 as Q4_0 does) is the classic Q4_K bug.
// ---------------------------------------------------------------------------
// get_scale_min_k4: sub-blocks 0-3 take the low 6 bits of bytes j and j+4;
// sub-blocks 4-7 take their low nibble from byte j+4 and their HIGH 2 bits
// from the top of byte j-4 (scale) / byte j (min). Transcribed verbatim, then
// checked against the reference in the host test for all 8 sub-blocks.
static inline void q4k_scale_min(global const uint* p, uint sc_base, uint j,
                                 float* sc, float* mn) {
    uint d, m;
    if (j < 4u) {
        d = u8(p, sc_base + j) & 63u;
        m = u8(p, sc_base + j + 4u) & 63u;
    } else {
        d = (u8(p, sc_base + j + 4u) & 0xFu) | ((u8(p, sc_base + j - 4u) >> 6) << 4);
        m = (u8(p, sc_base + j + 4u) >> 4)   | ((u8(p, sc_base + j)      >> 6) << 4);
    }
    *sc = (float)d;
    *mn = (float)m;
}

// Element r of a super-block belongs to sub-block r/32 (see the doc for why
// that one-liner is equivalent to the reference's is/j walk), and its nibble
// is byte (r/64)*32 + r%32, low half when (r%64) < 32.
kernel void dequant_q4_k(global float* out, global const uint* src, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    uint base = (i >> 8) * 144u;
    uint r    = i & 255u;

    float d    = fp16_to_fp32(u16(src, base));
    float dmin = fp16_to_fp32(u16(src, base + 2u));
    float sc, mn;
    q4k_scale_min(src, base + 4u, r >> 5, &sc, &mn);

    uint byte = u8(src, base + 16u + ((r >> 6) * 32u) + (r & 31u));
    uint q    = ((r & 63u) < 32u) ? (byte & 0xFu) : (byte >> 4);
    out[i] = d * sc * (float)q - dmin * mn;
}

// Fused. Per sub-block this accumulates sum(in*q) and sum(in) separately, so
// the scale and the min are each applied ONCE per 32 weights instead of per
// weight: w.x = d*sc*sum(q*x) - dmin*mn*sum(x). Quants are read a uint at a
// time (8 nibbles) — legal only because 144 is a multiple of 4, so every
// super-block, and its 16-byte-offset quant array, stays word-aligned.
kernel void matvec_q4_k(global float* out, global const float* in,
                        global const uint* w, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nsb  = cols >> 8;                    // super-blocks per row
    uint base = r * nsb * 144u;
    float acc = 0.0f;
    for (uint b = 0; b < nsb; ++b) {
        float d    = fp16_to_fp32(u16(w, base));
        float dmin = fp16_to_fp32(u16(w, base + 2u));
        uint  qw   = (base + 16u) >> 2;       // word index of the 128 quant bytes
        for (uint s = 0; s < 8u; ++s) {
            float sc, mn;
            q4k_scale_min(w, base + 4u, s, &sc, &mn);
            // NB: `half` is a reserved type name in OpenCL C even when the
            // fp16 extension is off — naming this variable `half` is a syntax
            // error, not a shadowing warning.
            uint  chunk = (s >> 1) * 8u;      // 8 words = 32 bytes per 64 weights
            uint  shift = (s & 1u) ? 4u : 0u; // even sub-block = low nibbles
            uint  k     = (b << 8) + (s << 5);
            float sq = 0.0f, sx = 0.0f;
            for (uint t = 0; t < 8u; ++t) {   // 8 words x 4 nibbles = 32 weights
                uint word = w[qw + chunk + t];
                for (uint e = 0; e < 4u; ++e) {
                    float x = in[k + t * 4u + e];
                    sq += x * (float)((word >> (e * 8u + shift)) & 0xFu);
                    sx += x;
                }
            }
            acc += d * sc * sq - dmin * mn * sx;
        }
        base += 144u;
    }
    out[r] = acc;
}

// ---------------------------------------------------------------------------
// Q6_K — 210 B super-block of 256 weights (16 groups of 16):
//   [0:128]   ql — low 4 bits of every quant
//   [128:192] qh — high 2 bits, four quants packed per byte
//   [192:208] 16 int8 group scales
//   [208:210] fp16 d
// w = d * scale[group] * (q - 32), q the 6-bit unsigned value.
// 210 is not a multiple of 4, so nothing here can be read a word at a time.
// ---------------------------------------------------------------------------
// The 256 elements are two halves of 128. Inside a half, element index
// within = 32*g + l picks one of four interleaved quarters g: the low/high
// nibble of ql[l] or ql[l+32], with high bits from qh[l] >> 2g and scale
// sc[l/16 + 2g]. That interleave is why a naive sequential read is wrong.
static inline float q6k_value(global const uint* p, uint base, uint r) {
    uint hlf = r >> 7;                        // which 128-element half
    uint w   = r & 127u;
    uint l   = w & 31u;
    uint g   = w >> 5;                        // 0..3
    uint ql  = base + hlf * 64u + l + ((g & 1u) * 32u);
    uint qh  = base + 128u + hlf * 32u + l;
    uint sc  = base + 192u + hlf * 8u + (l >> 4) + 2u * g;

    uint lo = (g < 2u) ? (u8(p, ql) & 0xFu) : (u8(p, ql) >> 4);
    int  q  = (int)(lo | (((u8(p, qh) >> (2u * g)) & 3u) << 4)) - 32;
    return fp16_to_fp32(u16(p, base + 208u)) * (float)i8(p, sc) * (float)q;
}

kernel void dequant_q6_k(global float* out, global const uint* src, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    out[i] = q6k_value(src, (i >> 8) * 210u, i & 255u);
}

// Fused. Unlike Q4_K there is nothing to hoist per sub-block cheaply (the
// scale changes every 16 weights and the bias is a constant -32 folded into
// the quant), so this walks the interleave directly and hoists only d.
kernel void matvec_q6_k(global float* out, global const float* in,
                        global const uint* w, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nsb  = cols >> 8;
    uint base = r * nsb * 210u;
    float acc = 0.0f;
    for (uint b = 0; b < nsb; ++b) {
        float d = fp16_to_fp32(u16(w, base + 208u));
        float sb = 0.0f;
        for (uint hlf = 0; hlf < 2u; ++hlf) {
            for (uint g = 0; g < 4u; ++g) {
                for (uint l = 0; l < 32u; ++l) {
                    uint ql = base + hlf * 64u + l + ((g & 1u) * 32u);
                    uint qh = base + 128u + hlf * 32u + l;
                    uint sc = base + 192u + hlf * 8u + (l >> 4) + 2u * g;
                    uint lo = (g < 2u) ? (u8(w, ql) & 0xFu) : (u8(w, ql) >> 4);
                    int  q  = (int)(lo | (((u8(w, qh) >> (2u * g)) & 3u) << 4)) - 32;
                    sb += in[(b << 8) + hlf * 128u + (g << 5) + l]
                          * (float)i8(w, sc) * (float)q;
                }
            }
        }
        acc += d * sb;
        base += 210u;
    }
    out[r] = acc;
}

// ===========================================================================
// REPACKED FAST PATH
// ===========================================================================
// Everything above reads the NATIVE GGUF layout. That is bit-exact and it is
// the reference the code below is validated against, but it measured 0.9-4.1
// GB/s against a ~39 GB/s achievable ceiling, because the alignment rule
// forces one 4-byte load per byte extracted. Decode re-reads every weight once
// per token, so a one-time transform at model load is amortised over thousands
// of tokens: pay it once. The host side is examples/llm/quant_repack.{hpp,cpp}.
//
// The layout follows agent-docs/matvec-optimisation.md, whose finding was that
// the decisive factor is LOAD INSTRUCTION COUNT, not coalescing — the old
// packing was already perfectly coalesced and still spent one load per 4 useful
// BITS. Two rules came out of that doc and both are obeyed here:
//
//   * pack along K so one load carries weights ONE thread consumes;
//   * interleave in groups of 64 so a wave still reads contiguous memory.
//     ("naive k-major", contiguous per THREAD, was the worst k-major variant
//     measured — per-WAVE contiguity is what matters.)
//
// Index for output row r, k-block kb (32 weights), with g = r/64, t = r%64:
//
//     idx = (g * nkb + kb) * 64 + t          nkb = cols / 32
//
// Rows are padded up to a multiple of 64 so the last group is whole.
//
// PLANES, not widened quants. The obvious simplification is to store every
// format as int8; it was rejected because it would roughly DOUBLE the model on
// disk and in RAM (Q4_K would go 4.5 -> 10 bits/weight), which is the entire
// thing quantisation is for. Instead each format keeps its native bit width,
// split into planes that never straddle a word:
//
//   4 bits  -> one uint4 per (thread, k-block): 32 nibbles, exactly 16 B
//   5 bits  -> that uint4 + one uint of 32 high bits
//   6 bits  -> that uint4 + two uints of 32 2-bit high fields
//   8 bits  -> two uint4 per (thread, k-block)
//
// so a thread gets 32 weights from 1-3 loads instead of ~40. Within the nibble
// uint4, component c holds k-offsets 8c..8c+7 at shift 4*(k%8).
//
// SCALES ARE PRE-DECODED to fp32 on the host, one (scale[, bias]) per thread
// per k-block, in the SAME interleaved index. That removes the fp16 decode and
// the 6-bit scale unpacking from the inner loop entirely, and it is what lets
// all four formats share one shape. Only Q4_K needs a bias: its
// w = d*sc*q - dmin*mn has an additive term that is not proportional to the
// scale. The other three fold their zero point into a constant subtraction
// (-16, -32, none), which costs nothing.
//
// `in` is read as float4: with the weight loads this cheap, the INPUT vector
// would otherwise dominate the instruction count at 32 scalar loads per
// k-block. This is why cols must be a multiple of 32.

// One 8-nibble word (unsigned 0..15) against 8 inputs. sx accumulates plain
// sum(x) for the Q4_K bias term; the compiler drops it where it is unused.
#define NIB8(W, XA, XB)                                                        \
    do {                                                                       \
        uint _w = (W);                                                         \
        float4 _a = (XA), _b = (XB);                                           \
        sq += (float)((_w >>  0) & 0xFu) * _a.x                                \
            + (float)((_w >>  4) & 0xFu) * _a.y                                \
            + (float)((_w >>  8) & 0xFu) * _a.z                                \
            + (float)((_w >> 12) & 0xFu) * _a.w                                \
            + (float)((_w >> 16) & 0xFu) * _b.x                                \
            + (float)((_w >> 20) & 0xFu) * _b.y                                \
            + (float)((_w >> 24) & 0xFu) * _b.z                                \
            + (float)((_w >> 28) & 0xFu) * _b.w;                               \
        sx += _a.x + _a.y + _a.z + _a.w + _b.x + _b.y + _b.z + _b.w;           \
    } while (0)

// 8 nibbles + 8 one-bit extensions (bit j of H is weight j's 5th bit), biased.
#define N5_8(W, H, BIAS, XA, XB)                                               \
    do {                                                                       \
        uint _w = (W), _h = (H);                                               \
        float4 _a = (XA), _b = (XB);                                           \
        sq += (float)((int)(((_w >>  0) & 0xFu) | (((_h >> 0) & 1u) << 4)) - BIAS) * _a.x \
            + (float)((int)(((_w >>  4) & 0xFu) | (((_h >> 1) & 1u) << 4)) - BIAS) * _a.y \
            + (float)((int)(((_w >>  8) & 0xFu) | (((_h >> 2) & 1u) << 4)) - BIAS) * _a.z \
            + (float)((int)(((_w >> 12) & 0xFu) | (((_h >> 3) & 1u) << 4)) - BIAS) * _a.w \
            + (float)((int)(((_w >> 16) & 0xFu) | (((_h >> 4) & 1u) << 4)) - BIAS) * _b.x \
            + (float)((int)(((_w >> 20) & 0xFu) | (((_h >> 5) & 1u) << 4)) - BIAS) * _b.y \
            + (float)((int)(((_w >> 24) & 0xFu) | (((_h >> 6) & 1u) << 4)) - BIAS) * _b.z \
            + (float)((int)(((_w >> 28) & 0xFu) | (((_h >> 7) & 1u) << 4)) - BIAS) * _b.w; \
    } while (0)

// 8 nibbles + 8 two-bit extensions (field j of H at shift 2j), biased by 32.
#define N6_8(W, H, XA, XB)                                                     \
    do {                                                                       \
        uint _w = (W), _h = (H);                                               \
        float4 _a = (XA), _b = (XB);                                           \
        sq += (float)((int)(((_w >>  0) & 0xFu) | (((_h >>  0) & 3u) << 4)) - 32) * _a.x \
            + (float)((int)(((_w >>  4) & 0xFu) | (((_h >>  2) & 3u) << 4)) - 32) * _a.y \
            + (float)((int)(((_w >>  8) & 0xFu) | (((_h >>  4) & 3u) << 4)) - 32) * _a.z \
            + (float)((int)(((_w >> 12) & 0xFu) | (((_h >>  6) & 3u) << 4)) - 32) * _a.w \
            + (float)((int)(((_w >> 16) & 0xFu) | (((_h >>  8) & 3u) << 4)) - 32) * _b.x \
            + (float)((int)(((_w >> 20) & 0xFu) | (((_h >> 10) & 3u) << 4)) - 32) * _b.y \
            + (float)((int)(((_w >> 24) & 0xFu) | (((_h >> 12) & 3u) << 4)) - 32) * _b.z \
            + (float)((int)(((_w >> 28) & 0xFu) | (((_h >> 14) & 3u) << 4)) - 32) * _b.w; \
    } while (0)

// 4 signed bytes against one float4.
#define B4(W, X)                                                               \
    do {                                                                       \
        uint _w = (W);                                                         \
        float4 _x = (X);                                                       \
        sq += (float)((int)(((_w >>  0) & 0xFFu) ^ 0x80u) - 128) * _x.x         \
            + (float)((int)(((_w >>  8) & 0xFFu) ^ 0x80u) - 128) * _x.y         \
            + (float)((int)(((_w >> 16) & 0xFFu) ^ 0x80u) - 128) * _x.z         \
            + (float)((int)(((_w >> 24) & 0xFFu) ^ 0x80u) - 128) * _x.w;        \
    } while (0)

// ---- per-format dot over a k-block RANGE ----------------------------------
// Split-k and whole-row kernels call the same helper, so the fast path has one
// implementation and split-k cannot drift from it.

static inline float rq4k_dot(global const float4* in, global const uint4* q,
                             global const float* scale, global const float* bias,
                             uint nkb, uint g, uint t, uint kb0, uint kb1) {
    float acc = 0.0f;
    for (uint kb = kb0; kb < kb1; ++kb) {
        uint idx = (g * nkb + kb) * 64u + t;
        uint xb = kb << 3;
        uint4 w = q[idx];
        float sq = 0.0f, sx = 0.0f;
        NIB8(w.x, in[xb + 0], in[xb + 1]);
        NIB8(w.y, in[xb + 2], in[xb + 3]);
        NIB8(w.z, in[xb + 4], in[xb + 5]);
        NIB8(w.w, in[xb + 6], in[xb + 7]);
        acc += scale[idx] * sq - bias[idx] * sx;
    }
    return acc;
}

static inline float rq50_dot(global const float4* in, global const uint4* q,
                             global const uint* hi, global const float* scale,
                             uint nkb, uint g, uint t, uint kb0, uint kb1) {
    float acc = 0.0f;
    for (uint kb = kb0; kb < kb1; ++kb) {
        uint idx = (g * nkb + kb) * 64u + t;
        uint xb = kb << 3;
        uint4 w = q[idx];
        uint  h = hi[idx];
        float sq = 0.0f;
        N5_8(w.x, h >>  0, 16, in[xb + 0], in[xb + 1]);
        N5_8(w.y, h >>  8, 16, in[xb + 2], in[xb + 3]);
        N5_8(w.z, h >> 16, 16, in[xb + 4], in[xb + 5]);
        N5_8(w.w, h >> 24, 16, in[xb + 6], in[xb + 7]);
        acc += scale[idx] * sq;
    }
    return acc;
}

// Q6_K's scale granularity is 16, not 32, so each k-block carries TWO scales
// and the halves accumulate separately.
static inline float rq6k_dot(global const float4* in, global const uint4* q,
                             global const uint* hi, global const float* scale,
                             uint nkb, uint g, uint t, uint kb0, uint kb1) {
    float acc = 0.0f;
    for (uint kb = kb0; kb < kb1; ++kb) {
        uint idx = (g * nkb + kb) * 64u + t;
        uint xb = kb << 3;
        uint4 w = q[idx];
        uint h0 = hi[idx * 2u + 0u], h1 = hi[idx * 2u + 1u];
        float sq = 0.0f;
        N6_8(w.x, h0 >>  0, in[xb + 0], in[xb + 1]);
        N6_8(w.y, h0 >> 16, in[xb + 2], in[xb + 3]);
        float lo = sq;
        sq = 0.0f;
        N6_8(w.z, h1 >>  0, in[xb + 4], in[xb + 5]);
        N6_8(w.w, h1 >> 16, in[xb + 6], in[xb + 7]);
        acc += scale[idx * 2u + 0u] * lo + scale[idx * 2u + 1u] * sq;
    }
    return acc;
}

static inline float rq80_dot(global const float4* in, global const uint4* q,
                             global const float* scale,
                             uint nkb, uint g, uint t, uint kb0, uint kb1) {
    float acc = 0.0f;
    for (uint kb = kb0; kb < kb1; ++kb) {
        uint idx = (g * nkb + kb) * 64u + t;
        uint xb = kb << 3;
        uint4 a = q[idx * 2u + 0u], b = q[idx * 2u + 1u];
        float sq = 0.0f;
        B4(a.x, in[xb + 0]); B4(a.y, in[xb + 1]);
        B4(a.z, in[xb + 2]); B4(a.w, in[xb + 3]);
        B4(b.x, in[xb + 4]); B4(b.y, in[xb + 5]);
        B4(b.z, in[xb + 6]); B4(b.w, in[xb + 7]);
        acc += scale[idx] * sq;
    }
    return acc;
}

// ---- whole-row kernels ----------------------------------------------------
kernel void matvec_rq4_k(global float* out, global const float4* in,
                         global const uint4* q, global const float* scale,
                         global const float* bias, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nkb = cols >> 5;
    out[r] = rq4k_dot(in, q, scale, bias, nkb, r >> 6, r & 63u, 0u, nkb);
}

kernel void matvec_rq5_0(global float* out, global const float4* in,
                         global const uint4* q, global const uint* hi,
                         global const float* scale, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nkb = cols >> 5;
    out[r] = rq50_dot(in, q, hi, scale, nkb, r >> 6, r & 63u, 0u, nkb);
}

kernel void matvec_rq6_k(global float* out, global const float4* in,
                         global const uint4* q, global const uint* hi,
                         global const float* scale, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nkb = cols >> 5;
    out[r] = rq6k_dot(in, q, hi, scale, nkb, r >> 6, r & 63u, 0u, nkb);
}

kernel void matvec_rq8_0(global float* out, global const float4* in,
                         global const uint4* q, global const float* scale,
                         uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    uint nkb = cols >> 5;
    out[r] = rq80_dot(in, q, scale, nkb, r >> 6, r & 63u, 0u, nkb);
}

// ---- split-k --------------------------------------------------------------
// One thread per output row leaves narrow matrices starved: attn_k is 256 rows,
// i.e. 4 waves, on a GPU that wants thousands of threads. Split-k was reverted
// once in this repo for making decode 40% WORSE — that verdict was correct at
// the time and is now obsolete: it lost because it doubled a dispatch count
// when a submit cost 0.31 ms. With vulkore::Batch amortising submission the extra
// reduce pass is nearly free (matvec-optimisation.md measured 2.0x on the
// tall/narrow `down` shape).
//
// Thread id maps to (p, r) as id = p*rows + r so that adjacent threads still
// hold adjacent ROWS and the interleaved layout stays coalesced.
// partial is [split * rows]; mv_reduce sums it.

#define SPLIT_PROLOGUE                                                         \
    uint id = get_global_id(0);                                                \
    if (id >= rows * split) return;                                            \
    uint r = id % rows, p = id / rows;                                         \
    uint nkb = cols >> 5;                                                      \
    uint per = (nkb + split - 1u) / split;                                     \
    uint kb0 = p * per;                                                        \
    uint kb1 = min(kb0 + per, nkb);                                            \
    if (kb0 >= nkb) { partial[id] = 0.0f; return; }

kernel void matvec_rq4_k_split(global float* partial, global const float4* in,
                               global const uint4* q, global const float* scale,
                               global const float* bias, uint rows, uint cols,
                               uint split) {
    SPLIT_PROLOGUE
    partial[id] = rq4k_dot(in, q, scale, bias, nkb, r >> 6, r & 63u, kb0, kb1);
}

kernel void matvec_rq5_0_split(global float* partial, global const float4* in,
                               global const uint4* q, global const uint* hi,
                               global const float* scale, uint rows, uint cols,
                               uint split) {
    SPLIT_PROLOGUE
    partial[id] = rq50_dot(in, q, hi, scale, nkb, r >> 6, r & 63u, kb0, kb1);
}

kernel void matvec_rq6_k_split(global float* partial, global const float4* in,
                               global const uint4* q, global const uint* hi,
                               global const float* scale, uint rows, uint cols,
                               uint split) {
    SPLIT_PROLOGUE
    partial[id] = rq6k_dot(in, q, hi, scale, nkb, r >> 6, r & 63u, kb0, kb1);
}

kernel void matvec_rq8_0_split(global float* partial, global const float4* in,
                               global const uint4* q, global const float* scale,
                               uint rows, uint cols, uint split) {
    SPLIT_PROLOGUE
    partial[id] = rq80_dot(in, q, scale, nkb, r >> 6, r & 63u, kb0, kb1);
}

// Second pass for every split-k kernel above. No __local, so this is a plain
// global read-and-sum; split is small (4-16) so it is a rounding error next to
// the matvec itself.
kernel void mv_reduce(global float* out, global const float* partial,
                      uint rows, uint split) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    float acc = 0.0f;
    for (uint p = 0; p < split; ++p) acc += partial[p * rows + r];
    out[r] = acc;
}
