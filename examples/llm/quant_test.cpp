// Correctness harness for kernels/llm_quant.cl — GGUF dequantisation on the GPU.
//
// This is a correctness job before it is a performance job, so the harness is
// built around three independent opinions of what each format means:
//
//   1. the KERNEL (kernels/llm_quant.cl),
//   2. a scalar CPU reference written in this file straight from the format
//      spec, element-indexed rather than pointer-walking so a transcription
//      slip in either version shows up as a disagreement rather than being
//      copied into both,
//   3. GGUFFile::dequantize (examples/llm/model/gguf.cpp), the loader agent's
//      transcription, which has already been validated against a REAL model
//      file by cross-quantisation correlation.
//
// All three must agree, and then the whole thing is run again on actual
// tensors pulled out of models/gemma-3-1b-it-Q4_K_M.gguf. Synthetic blocks
// alone are not enough: they cannot catch a wrong block STRIDE or a wrong
// assumption about which axis a row runs along, because both sides of a
// synthetic test share the same assumption.
//
// LIVENESS is checked everywhere with a sentinel prefill. A kernel in this repo
// (an 8th-power Mandelbulb) once compiled clean, passed spirv-val, bound
// successfully and then silently did not execute on Adreno while every API call
// reported success. A clean compile proves nothing; only observed writes do.
//
// Build (examples are not wired into CMake):
//   g++ -std=c++20 -O2 -Iinclude -Iexamples/llm -Iexamples/llm/model \
//       -Ithird_party/Vulkan-Headers/include -Ithird_party/volk \
//       -Ithird_party/VulkanMemoryAllocator/include \
//       -DVMA_STATIC_VULKAN_FUNCTIONS=0 -DVMA_DYNAMIC_VULKAN_FUNCTIONS=1 \
//       examples/llm/quant_test.cpp examples/llm/quant_repack.cpp \
//       examples/llm/model/gguf.cpp build/libvulkore.a -ldl -o build/llm_quant_test
//   ./build/llm_quant_test [kernels/llm_quant.spv] [models/gemma-3-1b-it-Q4_K_M.gguf]

#include <vulkore/vulkore.hpp>

#include "gguf.hpp"
#include "quant_repack.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

using vulkore::llm::GGMLType;
using vulkore::llm::GGUFFile;

static constexpr float kSentinel = -1.0e30f;
static int g_pass = 0, g_fail = 0;

// ---------------------------------------------------------------------------
// Reporting. Dequantisation is EXACT arithmetic in principle — every value is
// a small integer times an fp16 scale — so the bar here is much tighter than
// the transformer harness's: any disagreement beyond fp32 rounding of a
// different multiply order is a layout bug, not noise.
// ---------------------------------------------------------------------------
static constexpr float kRtol = 1e-6f;
static constexpr float kAtol = 1e-9f;

static bool report(const char* name, const std::vector<float>& gpu,
                   const std::vector<float>& cpu, bool liveness = true) {
    double max_abs = 0.0, max_rel = 0.0;
    size_t worst = 0, nbad = 0;
    int untouched = 0, nonfinite = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(gpu[i])) ++nonfinite;
        if (liveness && gpu[i] == kSentinel) ++untouched;
        double a = std::fabs(double(gpu[i]) - double(cpu[i]));
        double r = a / std::fmax(std::fabs(double(cpu[i])), 1e-6);
        if (a > kAtol + double(kRtol) * std::fabs(double(cpu[i]))) {
            if (!nbad) worst = i;
            ++nbad;
        }
        if (a > max_abs) max_abs = a;
        max_rel = std::fmax(max_rel, r);
    }
    bool ok = nbad == 0 && untouched == 0 && nonfinite == 0;
    ok ? ++g_pass : ++g_fail;
    std::printf("  %-30s %s  n=%-9zu max_abs=%.3e  max_rel=%.3e", name,
                ok ? "PASS" : "FAIL", cpu.size(), max_abs, max_rel);
    if (untouched) std::printf("  UNTOUCHED=%d", untouched);
    if (nonfinite) std::printf("  NONFINITE=%d", nonfinite);
    if (nbad)
        std::printf("  bad=%zu worst[%zu] gpu=%.9g cpu=%.9g", nbad, worst,
                    gpu[worst], cpu[worst]);
    std::printf("\n");
    return ok;
}

// ---------------------------------------------------------------------------
// CPU reference — written from the format spec, independently of both the
// kernel and the loader. Deliberately indexed by ELEMENT (given element i,
// where do its bits live?) rather than walking pointers the way ggml does,
// so the two transcriptions fail differently if either is wrong.
// ---------------------------------------------------------------------------
static float ref_fp16(uint32_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            bits = sign | (uint32_t(112 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

namespace ref {

inline uint32_t u8(const uint8_t* p, size_t o) { return p[o]; }
inline uint32_t u16(const uint8_t* p, size_t o) { return p[o] | (p[o + 1] << 8); }
inline int i8(const uint8_t* p, size_t o) { return int(p[o] ^ 0x80u) - 128; }

// Q8_0: 34 B block = fp16 d, 32 int8. w = d*q.
void q8_0(const uint8_t* p, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* b = p + (i / 32) * 34;
        y[i] = ref_fp16(u16(b, 0)) * float(i8(b, 2 + i % 32));
    }
}

// Q5_0: 22 B block = fp16 d, u32 qh, 16 B nibbles. w = d*(q-16).
// Element i takes the low nibble of byte i%16 when i<16 and the high nibble
// of that same byte when i>=16; its 5th bit is bit i of qh. (The reference's
// (j) / (j+12) shift pair says exactly this in a more confusing way.)
void q5_0(const uint8_t* p, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* b = p + (i / 32) * 22;
        size_t e = i % 32;
        uint32_t qh = u8(b, 2) | (u8(b, 3) << 8) | (u8(b, 4) << 16) | (u8(b, 5) << 24);
        uint32_t byte = u8(b, 6 + e % 16);
        uint32_t lo = e < 16 ? (byte & 0xF) : (byte >> 4);
        int q = int(lo | (((qh >> e) & 1u) << 4)) - 16;
        y[i] = ref_fp16(u16(b, 0)) * float(q);
    }
}

// Q4_K: 144 B super-block = fp16 d, fp16 dmin, 12 B packed 6-bit scales/mins,
// 128 B nibbles. w = d*sc[s]*q - dmin*mn[s], q UNSIGNED and unbiased.
void q4k_scale_min(const uint8_t* s, int j, int& d, int& m) {
    if (j < 4) {
        d = s[j] & 63;
        m = s[j + 4] & 63;
    } else {
        d = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}
void q4_k(const uint8_t* p, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* b = p + (i / 256) * 144;
        size_t r = i % 256;
        float d = ref_fp16(u16(b, 0)), dmin = ref_fp16(u16(b, 2));
        int sc, mn;
        q4k_scale_min(b + 4, int(r / 32), sc, mn);
        uint32_t byte = u8(b, 16 + (r / 64) * 32 + r % 32);
        uint32_t q = (r % 64) < 32 ? (byte & 0xF) : (byte >> 4);
        y[i] = d * float(sc) * float(q) - dmin * float(mn);
    }
}

// Q6_K: 210 B super-block = 128 B ql, 64 B qh, 16 int8 scales, fp16 d.
// w = d*sc[g]*(q-32). Within each 128-element half the elements are four
// interleaved quarters, which is the part that bites.
void q6_k(const uint8_t* p, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* b = p + (i / 256) * 210;
        size_t r = i % 256, h = r / 128, w = r % 128, l = w % 32, g = w / 32;
        uint32_t byte = u8(b, h * 64 + l + (g & 1) * 32);
        uint32_t lo = g < 2 ? (byte & 0xF) : (byte >> 4);
        uint32_t hi = (u8(b, 128 + h * 32 + l) >> (2 * g)) & 3u;
        int q = int(lo | (hi << 4)) - 32;
        int sc = i8(b, 192 + h * 8 + l / 16 + 2 * g);
        y[i] = ref_fp16(u16(b, 208)) * float(sc) * float(q);
    }
}

}  // namespace ref

// ---------------------------------------------------------------------------
// Format table. Everything below is driven off this so adding a format later
// means adding one row.
// ---------------------------------------------------------------------------
struct Format {
    const char* name;
    GGMLType    type;
    uint32_t    block;    // elements per block
    uint32_t    stride;   // bytes per block
    uint32_t    dsize;    // bytes of fp16 scale at the START of the block (0 = elsewhere)
    uint32_t    doff;     // byte offset of the fp16 d within the block
    void      (*cpu)(const uint8_t*, float*, size_t);
    const char* deq_kernel;
    const char* mv_kernel;
    const char* rq_kernel;   // repacked fast path; _split suffix for split-k
};

static const Format kFormats[] = {
    {"Q8_0", GGMLType::Q8_0, 32, 34, 2, 0, ref::q8_0, "dequant_q8_0", "matvec_q8_0", "matvec_rq8_0"},
    {"Q5_0", GGMLType::Q5_0, 32, 22, 2, 0, ref::q5_0, "dequant_q5_0", "matvec_q5_0", "matvec_rq5_0"},
    {"Q4_K", GGMLType::Q4_K, 256, 144, 4, 0, ref::q4_k, "dequant_q4_k", "matvec_q4_k", "matvec_rq4_k"},
    {"Q6_K", GGMLType::Q6_K, 256, 210, 2, 208, ref::q6_k, "dequant_q6_k", "matvec_q6_k", "matvec_rq6_k"},
};

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------
// Quantised bytes reach the GPU as a uint view: vulkore::Buffer static_asserts a
// 4-byte-aligned element type, so there is no uchar buffer to upload into.
// Q5_0/Q8_0/Q6_K strides are not multiples of 4, hence the round-up.
static std::vector<uint32_t> as_words(const uint8_t* p, size_t nbytes) {
    std::vector<uint32_t> w((nbytes + 3) / 4, 0);
    std::memcpy(w.data(), p, nbytes);
    return w;
}

// Random block bytes, with the fp16 scale fields forced into a sane range.
// Random BYTES (rather than round-tripping a quantiser) is the point: it
// exercises every scale, every min and every high-bit pattern the format can
// express, including the ones a real quantiser would never emit.
static std::vector<uint8_t> random_blocks(const Format& f, size_t nblocks,
                                          std::mt19937& rng) {
    std::vector<uint8_t> raw(nblocks * f.stride);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : raw) b = uint8_t(byte(rng));
    // fp16 fields: exponent in [1,30] (no zero/subnormal/Inf/NaN) so a bad
    // scale cannot mask a layout bug behind a NaN.
    std::uniform_int_distribution<int> ex(1, 30), mn(0, 1023), sg(0, 1);
    for (size_t b = 0; b < nblocks; ++b) {
        for (uint32_t k = 0; k < f.dsize; k += 2) {
            uint32_t h = uint32_t(sg(rng) << 15) | uint32_t(ex(rng) << 10) | uint32_t(mn(rng));
            // Keep magnitudes near 1 so products stay well inside fp32 range:
            // rebias the exponent into [8,22] i.e. roughly 2^-7 .. 2^7.
            h = (h & 0x83FFu) | (uint32_t(8 + (ex(rng) % 15)) << 10);
            raw[b * f.stride + f.doff + k] = uint8_t(h & 0xFF);
            raw[b * f.stride + f.doff + k + 1] = uint8_t(h >> 8);
        }
    }
    return raw;
}

int main(int argc, char** argv) {
    const char* spv = argc > 1 ? argv[1] : "kernels/llm_quant.spv";
    const char* gguf = argc > 2 ? argv[2] : "models/gemma-3-1b-it-Q4_K_M.gguf";

    vulkore::Context ctx;
    auto prog = vulkore::Program::from_file(ctx, spv);
    std::printf("device: %s\nmodule: %s (%zu kernels)\n", ctx.device_name().c_str(),
                spv, prog.kernel_names().size());

    std::mt19937 rng(20260719);

    // =======================================================================
    // 1. fp16 decode, swept over ALL 65536 bit patterns.
    // Every other kernel here multiplies by an fp16 scale, so a decoder bug
    // would show up downstream as a diffuse scale error that is very hard to
    // localise. Sweeping the whole domain settles it once: subnormals, both
    // zeros, both infinities and every NaN payload included.
    // =======================================================================
    {
        std::printf("\nfp16 decode (exhaustive, all 65536 bit patterns):\n");
        std::vector<uint16_t> pat(65536);
        for (uint32_t i = 0; i < 65536; ++i) pat[i] = uint16_t(i);
        auto words = as_words(reinterpret_cast<const uint8_t*>(pat.data()), 131072);
        auto dsrc = ctx.alloc<uint32_t>(words.size());
        auto dout = ctx.alloc<float>(65536);
        dsrc.upload(std::span<const uint32_t>(words));
        std::vector<float> seed(65536, kSentinel);
        dout.upload(std::span<const float>(seed));
        auto kdec = prog.kernel("fp16_decode");
        vulkore::launch(kdec, {65536u}, dout, dsrc, 65536u).wait();
        std::vector<float> gpu(65536);
        dout.download(std::span<float>(gpu));

        // NaN != NaN, so finite/Inf values are compared numerically and NaNs
        // are compared as "both are NaN" — a NaN scale never reaches the maths
        // anyway, but a decoder that turned NaN into a finite number would be
        // hiding a broken exponent path.
        size_t bad = 0, first = 0, nsub = 0, nnan = 0;
        double max_abs = 0.0;
        for (uint32_t i = 0; i < 65536; ++i) {
            float want = ref_fp16(i);
            bool okv = std::isnan(want) ? std::isnan(gpu[i]) : (gpu[i] == want);
            if (std::isnan(want)) ++nnan;
            if (((i >> 10) & 0x1F) == 0 && (i & 0x3FF)) ++nsub;
            if (!okv) {
                if (!bad) first = i;
                ++bad;
            } else if (std::isfinite(want)) {
                max_abs = std::fmax(max_abs, std::fabs(double(gpu[i]) - double(want)));
            }
        }
        bad == 0 ? ++g_pass : ++g_fail;
        std::printf("  %-30s %s  n=65536    exact=%zu/65536  (subnormals=%zu, "
                    "NaNs=%zu)  max_abs=%.3e\n",
                    "fp16_decode", bad ? "FAIL" : "PASS", 65536 - bad, nsub, nnan,
                    max_abs);
        if (bad)
            std::printf("      first mismatch: pattern 0x%04zx gpu=%.9g cpu=%.9g\n",
                        first, gpu[first], ref_fp16(uint32_t(first)));
    }

    // =======================================================================
    // 2. Synthetic blocks: GPU vs this file's CPU reference vs the loader's.
    // =======================================================================
    std::printf("\nsynthetic random blocks (GPU vs CPU reference in this file):\n");
    for (const Format& f : kFormats) {
        const size_t nblocks = 512;
        const size_t n = nblocks * f.block;
        auto raw = random_blocks(f, nblocks, rng);
        auto words = as_words(raw.data(), raw.size());

        std::vector<float> cpu(n);
        f.cpu(raw.data(), cpu.data(), n);

        // Third opinion: the loader's independently written transcription,
        // already validated against the real model file.
        std::vector<float> loader(n, 0.0f);
        bool have_loader = GGUFFile::dequantize(f.type, raw.data(), loader.data(), n);

        auto dsrc = ctx.alloc<uint32_t>(words.size());
        auto dout = ctx.alloc<float>(n);
        dsrc.upload(std::span<const uint32_t>(words));
        std::vector<float> seed(n, kSentinel);
        dout.upload(std::span<const float>(seed));
        auto kdq = prog.kernel(f.deq_kernel);
        vulkore::launch(kdq, {uint32_t(n)}, dout, dsrc, uint32_t(n)).wait();
        std::vector<float> gpu(n);
        dout.download(std::span<float>(gpu));

        report((std::string(f.name) + " dequant").c_str(), gpu, cpu);
        if (have_loader) {
            size_t diff = 0;
            for (size_t i = 0; i < n; ++i)
                if (loader[i] != cpu[i]) ++diff;
            diff == 0 ? ++g_pass : ++g_fail;
            std::printf("  %-30s %s  n=%-9zu vs GGUFFile::dequantize "
                        "(bit-identical: %s)\n",
                        (std::string(f.name) + " ref cross-check").c_str(),
                        diff ? "FAIL" : "PASS", n, diff ? "no" : "yes");
        }

        // ---- fused dequant-matvec on the same bytes ------------------------
        // Rows are laid out the way GGUF lays them out: shape[0] (the fast
        // axis) is the row length, so one output row's blocks are contiguous.
        const uint32_t cols = f.block * 4;         // 128 or 1024 elements/row
        const uint32_t rows = uint32_t(n / cols);
        std::vector<float> in(cols);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto& e : in) e = u(rng);

        std::vector<float> mvcpu(rows, 0.0f);
        for (uint32_t r = 0; r < rows; ++r) {
            double acc = 0.0;   // double accumulation: the GPU sums in fp32 but
                                // in a different order, and a double reference
                                // makes the printed error the KERNEL's error
                                // rather than a race between two fp32 orders.
            for (uint32_t k = 0; k < cols; ++k) acc += double(cpu[r * cols + k]) * double(in[k]);
            mvcpu[r] = float(acc);
        }

        auto din = ctx.alloc<float>(cols);
        auto dmv = ctx.alloc<float>(rows);
        din.upload(std::span<const float>(in));
        std::vector<float> mvseed(rows, kSentinel);
        dmv.upload(std::span<const float>(mvseed));
        auto kmv = prog.kernel(f.mv_kernel);
        vulkore::launch(kmv, {rows}, dmv, din, dsrc, rows, cols).wait();
        std::vector<float> mvgpu(rows);
        dmv.download(std::span<float>(mvgpu));

        // Reduction order differs from the reference, so this one gets the
        // looser bar a dot product deserves; the dequant test above is the
        // strict one.
        double max_rel = 0.0, max_abs = 0.0;
        int untouched = 0;
        double norm = 0.0;
        for (uint32_t r = 0; r < rows; ++r) {
            if (mvgpu[r] == kSentinel) ++untouched;
            double a = std::fabs(double(mvgpu[r]) - double(mvcpu[r]));
            max_abs = std::fmax(max_abs, a);
            norm = std::fmax(norm, std::fabs(double(mvcpu[r])));
        }
        max_rel = norm > 0 ? max_abs / norm : 0.0;
        bool ok = untouched == 0 && max_rel < 1e-5;
        ok ? ++g_pass : ++g_fail;
        std::printf("  %-30s %s  rows=%-6u cols=%-5u max_abs=%.3e  "
                    "max_abs/|out|_inf=%.3e%s\n",
                    (std::string(f.name) + " matvec (fused)").c_str(),
                    ok ? "PASS" : "FAIL", rows, cols, max_abs, max_rel,
                    untouched ? "  UNTOUCHED!" : "");
    }

    // =======================================================================
    // 3. REAL WEIGHTS. Synthetic blocks share their layout assumption between
    // the test and the kernel, so they cannot catch a wrong block stride or a
    // wrong idea of which axis a row runs along. Actual tensors can.
    // =======================================================================
    std::printf("\nreal tensors from %s:\n", gguf);
    GGUFFile file;
    bool opened = true;
    try {
        file = GGUFFile::open(gguf);
    } catch (const std::exception& e) {
        opened = false;
        std::printf("  SKIPPED — could not open (%s)\n", e.what());
    }

    if (opened) {
        // Census first: the tensor-type mix is the whole reason this file
        // supports the legacy 32-element formats at all. Gemma 3 1B is
        // nominally "Q4_K_M" but its width is 1152, which is not divisible by
        // the 256 a K-quant super-block needs, so most tensors fall back to
        // Q5_0/Q8_0 and a Q4_K-only kernel would cover under 10% of them.
        int count[64] = {0};
        for (const auto& t : file.tensors())
            if (uint32_t(t.type) < 64) ++count[uint32_t(t.type)];
        std::printf("  tensor census: ");
        for (const Format& f : kFormats)
            std::printf("%s=%d  ", f.name, count[uint32_t(f.type)]);
        std::printf("(total %zu)\n", file.tensors().size());

        for (const Format& f : kFormats) {
            // First 2-D tensor of this type: matvec needs a row length.
            const vulkore::llm::Tensor* pick = nullptr;
            for (const auto& t : file.tensors())
                if (t.type == f.type && t.shape.size() >= 2 && t.shape[0] % f.block == 0) {
                    pick = &t;
                    break;
                }
            if (!pick) {
                std::printf("  %-30s SKIP  no 2-D tensor of this type in the file\n",
                            f.name);
                continue;
            }
            const uint32_t cols = uint32_t(pick->shape[0]);
            // Cap the work: token_embd is 262144 x 1152, which is 1.2 GB as
            // fp32. A thousand rows exercises the same code paths, and is also
            // enough threads for the throughput number below to mean something
            // (256 rows = 256 threads leaves an Adreno almost entirely idle).
            // ALSO capped by a hard device limit: the standalone dequant
            // kernels are one thread per ELEMENT, and turnip allows at most
            // 65535 workgroups in x, i.e. 65535*64 = 4193280 threads per
            // launch. A full ffn_down (6912 x 1152 = 7.9 M elements) does not
            // fit in ONE dispatch and would throw. Real users of the dequant
            // path must therefore tile it; the fused matvec has no such
            // problem because it is one thread per ROW.
            uint32_t rows = uint32_t(std::min<uint64_t>(pick->shape[1], 1024));
            const uint32_t max_rows = 4193280u / cols;
            if (rows > max_rows) rows = max_rows;
            const size_t n = size_t(rows) * cols;
            const size_t nbytes = (n / f.block) * f.stride;

            auto words = as_words(pick->data, nbytes);
            std::vector<float> cpu(n), loader(n);
            f.cpu(pick->data, cpu.data(), n);
            GGUFFile::dequantize(f.type, pick->data, loader.data(), n);
            size_t refdiff = 0;
            for (size_t i = 0; i < n; ++i)
                if (loader[i] != cpu[i]) ++refdiff;

            auto dsrc = ctx.alloc<uint32_t>(words.size());
            auto dout = ctx.alloc<float>(n);
            dsrc.upload(std::span<const uint32_t>(words));
            std::vector<float> seed(n, kSentinel);
            dout.upload(std::span<const float>(seed));
            auto kdq = prog.kernel(f.deq_kernel);
            vulkore::launch(kdq, {uint32_t(n)}, dout, dsrc, uint32_t(n)).wait();
            std::vector<float> gpu(n);
            dout.download(std::span<float>(gpu));

            std::string label = std::string(f.name) + " " + std::string(pick->name);
            if (label.size() > 30) label = label.substr(0, 30);
            report(label.c_str(), gpu, cpu);
            if (refdiff)
                std::printf("      WARNING: %zu/%zu values disagree with "
                            "GGUFFile::dequantize\n", refdiff, n);

            // Fused matvec on the real tensor.
            std::vector<float> in(cols);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            for (auto& e : in) e = u(rng);
            std::vector<float> mvcpu(rows);
            for (uint32_t r = 0; r < rows; ++r) {
                double acc = 0.0;
                for (uint32_t k = 0; k < cols; ++k)
                    acc += double(cpu[size_t(r) * cols + k]) * double(in[k]);
                mvcpu[r] = float(acc);
            }
            auto din = ctx.alloc<float>(cols);
            auto dmv = ctx.alloc<float>(rows);
            din.upload(std::span<const float>(in));
            std::vector<float> mvseed(rows, kSentinel);
            dmv.upload(std::span<const float>(mvseed));
            auto kmv = prog.kernel(f.mv_kernel);
            vulkore::launch(kmv, {rows}, dmv, din, dsrc, rows, cols).wait();
            std::vector<float> mvgpu(rows);
            dmv.download(std::span<float>(mvgpu));

            double max_abs = 0.0, norm = 0.0;
            int untouched = 0;
            for (uint32_t r = 0; r < rows; ++r) {
                if (mvgpu[r] == kSentinel) ++untouched;
                max_abs = std::fmax(max_abs, std::fabs(double(mvgpu[r]) - double(mvcpu[r])));
                norm = std::fmax(norm, std::fabs(double(mvcpu[r])));
            }
            double rel = norm > 0 ? max_abs / norm : 0.0;
            bool ok = untouched == 0 && rel < 1e-5;
            ok ? ++g_pass : ++g_fail;
            std::printf("  %-30s %s  rows=%-6u cols=%-5u max_abs=%.3e  "
                        "max_abs/|out|_inf=%.3e%s\n",
                        (std::string(f.name) + " matvec (real)").c_str(),
                        ok ? "PASS" : "FAIL", rows, cols, max_abs, rel,
                        untouched ? "  UNTOUCHED!" : "");

            // No throughput print here. An earlier version timed the native
            // matvec at this point with a mean-of-50 and it disagreed with the
            // best-of-N in section 4 by 3x for the SAME kernel on the SAME
            // shape across consecutive runs — the clock/thermal drift that
            // matvec-optimisation.md warns about, which is exactly what a mean
            // fails to reject. All performance numbers now come from section
            // 4's single best-of-N harness, which times native and repacked
            // back to back so the comparison cannot be contaminated by drift
            // between runs.
        }
    }

    // =======================================================================
    // 4. REPACKED FAST PATH.
    // The native-layout kernels above are correct and slow. This section
    // validates the load-time repack (examples/llm/quant_repack.cpp) and the
    // matvec_rq* kernels that consume it, then measures what the transform
    // actually buys.
    // =======================================================================
    std::printf("\nrepacked fast path — layout oracle (host, must be BIT-EXACT):\n");

    // best-of-N, per matvec-optimisation.md: a GPU sharing memory with the rest
    // of the system throttles under sustained load, so the MEAN drifts between
    // runs. Best-of-N is the reproducible quantity and is what isolates kernel
    // quality from thermal state. Trial 0 is discarded as warm-up, and t0 is
    // taken AFTER the batch is recorded so the number is submit + GPU, not
    // command-buffer construction.
    auto time_best = [&](int reps, auto record) {
        double best = 1e30;
        for (int trial = 0; trial < 8; ++trial) {
            vulkore::Batch b(ctx);
            for (int i = 0; i < reps; ++i) record(b);
            auto t0 = std::chrono::steady_clock::now();
            b.submit().wait();
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count() / reps;
            if (trial) best = std::fmin(best, ms);
        }
        return best;
    };

    struct Case {
        std::string          label;
        std::vector<uint8_t> own;   // synthetic bytes live here
        const uint8_t*       ext = nullptr;  // real tensors point into the mmap
        uint32_t             rows = 0, cols = 0;
        bool                 real = false;
    };

    for (const Format& f : kFormats) {
        std::vector<Case> cases;
        {
            // Synthetic. rows = 130 is deliberately NOT a multiple of 64: the
            // repacked layout interleaves in groups of 64 and pads the last
            // group, and a padding bug would be invisible at a round row count.
            Case c;
            c.label = std::string(f.name) + " synth";
            c.rows = 130;
            c.cols = f.block == 256 ? 1024 : 128;
            c.own = random_blocks(f, size_t(c.rows) * (c.cols / f.block), rng);
            cases.push_back(std::move(c));
        }
        if (opened) {
            for (const auto& t : file.tensors())
                if (t.type == f.type && t.shape.size() >= 2 && t.shape[0] % f.block == 0) {
                    Case c;
                    c.label = std::string(f.name) + " " + std::string(t.name);
                    c.cols = uint32_t(t.shape[0]);
                    c.rows = uint32_t(std::min<uint64_t>(t.shape[1], 1024));
                    c.ext = t.data;
                    c.real = true;
                    cases.push_back(std::move(c));
                    break;
                }
        }

        for (Case& c : cases) {
            const uint8_t* src = c.own.empty() ? c.ext : c.own.data();
            const size_t n = size_t(c.rows) * c.cols;

            std::vector<float> cpu(n);
            f.cpu(src, cpu.data(), n);

            vulkore::llm::RepackedTensor rp;
            auto tr0 = std::chrono::steady_clock::now();
            bool ok = vulkore::llm::repack(f.type, src, c.rows, c.cols, rp);
            double repack_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - tr0).count();
            if (!ok) {
                std::printf("  %-30s FAIL  repack() refused the tensor\n", c.label.c_str());
                ++g_fail;
                continue;
            }

            // ---- host oracle: reconstruct must be BIT-EXACT ----------------
            size_t bad = 0, worst = 0;
            for (uint32_t r = 0; r < c.rows; ++r)
                for (uint32_t k = 0; k < c.cols; ++k) {
                    float got = vulkore::llm::reconstruct(rp, r, k);
                    if (got != cpu[size_t(r) * c.cols + k]) {
                        if (!bad) worst = size_t(r) * c.cols + k;
                        ++bad;
                    }
                }
            bad == 0 ? ++g_pass : ++g_fail;
            std::printf("  %-30s %s  n=%-9zu reconstruct==dequant: %s", c.label.c_str(),
                        bad ? "FAIL" : "PASS", n, bad ? "NO" : "bit-exact");
            if (bad)
                std::printf("  bad=%zu worst[%zu] got=%.9g want=%.9g", bad, worst,
                            vulkore::llm::reconstruct(rp, uint32_t(worst / c.cols),
                                                    uint32_t(worst % c.cols)),
                            cpu[worst]);
            std::printf("\n");

            // ---- GPU: repacked matvec, plain and split-k -------------------
            std::vector<float> in(c.cols);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            for (auto& e : in) e = u(rng);
            std::vector<float> want(c.rows);
            for (uint32_t r = 0; r < c.rows; ++r) {
                double acc = 0.0;
                for (uint32_t k = 0; k < c.cols; ++k)
                    acc += double(cpu[size_t(r) * c.cols + k]) * double(in[k]);
                want[r] = float(acc);
            }

            const uint32_t split = 4;      // correctness check uses 4
            const uint32_t max_split = 16;  // timing sweeps up to this
            auto dq = ctx.alloc<uint32_t>(rp.quant.size());
            auto dh = ctx.alloc<uint32_t>(rp.has_high() ? rp.high.size() : 1);
            auto ds = ctx.alloc<float>(rp.scale.size());
            auto db = ctx.alloc<float>(rp.has_bias() ? rp.bias.size() : 1);
            auto din = ctx.alloc<float>(c.cols);
            auto dout = ctx.alloc<float>(c.rows);
            auto dpart = ctx.alloc<float>(size_t(c.rows) * max_split);
            dq.upload(std::span<const uint32_t>(rp.quant));
            if (rp.has_high()) dh.upload(std::span<const uint32_t>(rp.high));
            ds.upload(std::span<const float>(rp.scale));
            if (rp.has_bias()) db.upload(std::span<const float>(rp.bias));
            din.upload(std::span<const float>(in));

            auto kr = prog.kernel(f.rq_kernel);
            auto krs = prog.kernel(std::string(f.rq_kernel) + "_split");
            auto kred = prog.kernel("mv_reduce");

            auto launch_plain = [&](vulkore::Batch& b) {
                switch (f.type) {
                    case GGMLType::Q4_K: b.add(kr, {c.rows}, dout, din, dq, ds, db, c.rows, c.cols); break;
                    case GGMLType::Q5_0:
                    case GGMLType::Q6_K: b.add(kr, {c.rows}, dout, din, dq, dh, ds, c.rows, c.cols); break;
                    default:             b.add(kr, {c.rows}, dout, din, dq, ds, c.rows, c.cols); break;
                }
            };
            auto launch_split_n = [&](vulkore::Batch& b, uint32_t sp) {
                switch (f.type) {
                    case GGMLType::Q4_K: b.add(krs, {c.rows * sp}, dpart, din, dq, ds, db, c.rows, c.cols, sp); break;
                    case GGMLType::Q5_0:
                    case GGMLType::Q6_K: b.add(krs, {c.rows * sp}, dpart, din, dq, dh, ds, c.rows, c.cols, sp); break;
                    default:             b.add(krs, {c.rows * sp}, dpart, din, dq, ds, c.rows, c.cols, sp); break;
                }
                b.add(kred, {c.rows}, dout, dpart, c.rows, sp);
            };
            auto launch_split = [&](vulkore::Batch& b) { launch_split_n(b, split); };

            auto check = [&](const char* what, auto&& record) {
                std::vector<float> seed(c.rows, kSentinel);
                dout.upload(std::span<const float>(seed));
                vulkore::Batch b(ctx);
                record(b);
                b.submit().wait();
                std::vector<float> gpu(c.rows);
                dout.download(std::span<float>(gpu));
                double max_abs = 0.0, norm = 0.0;
                int untouched = 0;
                for (uint32_t r = 0; r < c.rows; ++r) {
                    if (gpu[r] == kSentinel) ++untouched;
                    max_abs = std::fmax(max_abs, std::fabs(double(gpu[r]) - double(want[r])));
                    norm = std::fmax(norm, std::fabs(double(want[r])));
                }
                double rel = norm > 0 ? max_abs / norm : 0.0;
                bool good = untouched == 0 && rel < 1e-5;
                good ? ++g_pass : ++g_fail;
                std::printf("  %-30s %s  rows=%-5u cols=%-5u max_abs=%.3e  "
                            "max_abs/|out|_inf=%.3e%s\n",
                            (c.label + " " + what).c_str(), good ? "PASS" : "FAIL",
                            c.rows, c.cols, max_abs, rel,
                            untouched ? "  UNTOUCHED!" : "");
            };
            check("rq matvec", launch_plain);
            check("rq matvec split4", launch_split);

            // ---- what the repack bought ------------------------------------
            if (c.real) {
                const size_t native_n = vulkore::llm::native_bytes(f.type, c.rows, c.cols);
                auto native_words = as_words(src, native_n);
                auto dnat = ctx.alloc<uint32_t>(native_words.size());
                dnat.upload(std::span<const uint32_t>(native_words));
                auto knat = prog.kernel(f.mv_kernel);

                const int R = 30;
                double t_nat = time_best(R, [&](vulkore::Batch& b) {
                    b.add(knat, {c.rows}, dout, din, dnat, c.rows, c.cols);
                });
                double t_rq = time_best(R, launch_plain);

                auto gbs = [&](double ms, size_t bytes) {
                    return double(bytes) / (ms * 1e-3) / 1e9;
                };
                std::printf("      native  %7.4f ms %6.2f GB/s  |  repacked %7.4f ms "
                            "%6.2f GB/s (%.1fx vs native)\n",
                            t_nat, gbs(t_nat, native_n), t_rq, gbs(t_rq, rp.bytes()),
                            t_nat / t_rq);
                // Split-k sweep. It is a plain runtime argument, so the right
                // factor is a per-shape choice the decode path can make without
                // recompiling anything: narrow tensors want a lot of it, wide
                // ones already have more parallelism than the GPU can use.
                std::printf("      split ");
                double t_best_split = 1e30;
                uint32_t best_sp = 0;
                for (uint32_t sp : {4u, 8u, 16u}) {
                    double t = time_best(R, [&](vulkore::Batch& b) { launch_split_n(b, sp); });
                    if (t < t_best_split) { t_best_split = t; best_sp = sp; }
                    std::printf(" %2u:%7.4f ms %6.2f GB/s ", sp, t, gbs(t, rp.bytes()));
                }
                std::printf("\n      BEST split%u: %.4f ms  %.2f GB/s  = %.1fx native, "
                            "%.1fx repacked-unsplit\n",
                            best_sp, t_best_split, gbs(t_best_split, rp.bytes()),
                            t_nat / t_best_split, t_rq / t_best_split);
                std::printf("      size %.2f -> %.2f MB (%+.1f%%), repack %.1f ms "
                            "(%.0f MB/s of source)\n",
                            native_n / 1048576.0, rp.bytes() / 1048576.0,
                            100.0 * (double(rp.bytes()) / double(native_n) - 1.0),
                            repack_ms, native_n / 1048576.0 / (repack_ms * 1e-3));
            }
        }
    }

    // =======================================================================
    // 5. What repacking the WHOLE model costs at startup.
    // "A few seconds is acceptable" — so measure it rather than assert it.
    // =======================================================================
    if (opened) {
        std::printf("\nwhole-model repack cost:\n");
        size_t nat_total = 0, rq_total = 0, done = 0, skipped = 0;
        double ms_total = 0.0;
        for (const auto& t : file.tensors()) {
            const Format* f = nullptr;
            for (const Format& cand : kFormats)
                if (cand.type == t.type) f = &cand;
            if (!f || t.shape.size() < 2 || t.shape[0] % f->block != 0) {
                ++skipped;
                continue;
            }
            const uint32_t cols = uint32_t(t.shape[0]);
            const uint32_t rows = uint32_t(t.n_elements() / cols);
            vulkore::llm::RepackedTensor rp;
            auto t0 = std::chrono::steady_clock::now();
            if (!vulkore::llm::repack(t.type, t.data, rows, cols, rp)) {
                ++skipped;
                continue;
            }
            ms_total += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            nat_total += vulkore::llm::native_bytes(t.type, rows, cols);
            rq_total += rp.bytes();
            ++done;
        }
        std::printf("  %zu tensors repacked (%zu skipped: F32 norms, 1-D, or a row\n"
                    "  length not divisible by the block size)\n", done, skipped);
        std::printf("  %.1f MB native -> %.1f MB repacked (%+.1f%%) in %.2f s "
                    "(%.0f MB/s, single-threaded)\n",
                    nat_total / 1048576.0, rq_total / 1048576.0,
                    100.0 * (double(rq_total) / double(nat_total) - 1.0),
                    ms_total / 1000.0, nat_total / 1048576.0 / (ms_total * 1e-3));
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
