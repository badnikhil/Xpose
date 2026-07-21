// Host-side repack: native GGUF block layout -> the GPU-friendly layout that
// kernels/llm_quant.cl's matvec_rq* kernels consume.
//
// WHY THIS EXISTS. The native-layout kernels in llm_quant.cl are bit-exact and
// need no host transform — weights go straight from the mmap to the GPU — but
// they measured 0.9-4.1 GB/s against a ~39 GB/s achievable ceiling. The cause
// (agent-docs/matvec-optimisation.md) is LOAD INSTRUCTION COUNT: vulkore::Buffer
// only takes 4-byte element types, so reading a byte out of a 22/34/210-byte
// block costs a whole 4-byte load. Decode re-reads every weight once per token,
// so a transform paid ONCE at model load is amortised over thousands of tokens.
//
// THE LAYOUT (rows = outputs, cols = reduction length, both from the tensor;
// cols must be a multiple of 32, rows are padded up to a multiple of 64):
//
//   g = r/64, t = r%64, kb = k/32, j = k%32, nkb = cols/32
//   idx = (g*nkb + kb)*64 + t
//
// Pack along K so one load carries weights ONE thread consumes; interleave in
// groups of 64 so a wave still reads contiguous memory. (Naive k-major, which
// is contiguous per THREAD, was the worst k-major variant that agent measured —
// per-WAVE contiguity is what matters.)
//
//   quant[]  4-bit nibble plane, uint4 per idx: uint 4*idx + j/8, shift 4*(j%8)
//            (Q8_0 instead uses TWO uint4 per idx: uint 8*idx + j/4, shift 8*(j%4))
//   high[]   Q5_0: uint per idx, bit j
//            Q6_K: 2 uints per idx: uint 2*idx + j/16, shift 2*(j%16)
//   scale[]  fp32, one per idx (Q6_K: two, at 2*idx + j/16 — its scale
//            granularity is 16, not 32)
//   bias[]   fp32, one per idx, Q4_K ONLY
//
// Scales are PRE-DECODED to fp32 here, which takes the fp16 decode and the
// 6-bit scale unpacking out of the GPU inner loop entirely. Only Q4_K needs a
// bias, because w = d*sc*q - dmin*mn has an additive term not proportional to
// the scale; the others fold their zero point into a constant (-16, -32, none).
//
// Bit widths are PRESERVED. Storing everything as int8 would have been far
// simpler and was rejected: it would roughly double the model in RAM (Q4_K
// 4.5 -> 10 bits/weight), which is what quantisation exists to avoid. The
// growth this layout does cost is metadata only, and repack_growth() reports it.

#pragma once

#include <cstdint>
#include <vector>

#include "gguf.hpp"

namespace vulkore::llm {

struct RepackedTensor {
    uint32_t rows = 0;         // logical output rows
    uint32_t padded_rows = 0;  // rounded up to a multiple of 64
    uint32_t cols = 0;         // reduction length
    uint32_t nkb = 0;          // cols / 32
    GGMLType type = GGMLType::F32;

    std::vector<uint32_t> quant;  // always present
    std::vector<uint32_t> high;   // Q5_0 / Q6_K only
    std::vector<float>    scale;  // always present
    std::vector<float>    bias;   // Q4_K only

    size_t bytes() const {
        return quant.size() * 4 + high.size() * 4 + scale.size() * 4 + bias.size() * 4;
    }
    bool has_high() const { return !high.empty(); }
    bool has_bias() const { return !bias.empty(); }
};

// True on success. Fails for unsupported types, cols % 32 != 0, or a source
// buffer that is not a whole number of blocks.
bool repack(GGMLType type, const void* src, uint32_t rows, uint32_t cols,
            RepackedTensor& out);

// Pull element (r, k) back OUT of the repacked form. This is the oracle the
// repack is tested with: reconstruct() must reproduce a from-spec dequant of
// the same tensor BIT-FOR-BIT, which it can, because the stored scale is
// exactly the fp32 product the reference forms and the arithmetic that follows
// is the same two operations in the same order.
float reconstruct(const RepackedTensor& t, uint32_t r, uint32_t k);

// Bytes the native layout would have used, for the growth report.
size_t native_bytes(GGMLType type, uint32_t rows, uint32_t cols);

}  // namespace vulkore::llm
