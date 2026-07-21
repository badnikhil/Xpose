// matvec_bench — measure every int4 matvec variant in kernels/llm_matvec.cl at
// real Gemma 3 1B decode shapes, and CHECK EACH ONE against the reference
// matvec_q4 in kernels/llm.cl before believing its timing.
//
// The correctness check is not ceremony. A kernel can compile, pass spirv-val,
// bind, dispatch and still silently write nothing on Adreno (that happened with
// pow/acos/atan2 in the Mandelbulb sim: uninitialised output, zero measured
// work, a beautiful bandwidth number for doing nothing). So every variant's
// output buffer is poisoned before the run and compared element-by-element
// against the reference afterwards; a variant that fails is reported FAIL and
// its timing is not counted towards the best-of total.
//
// Usage: matvec_bench <llm.spv> <llm_matvec.spv> [--quick]
#include <vulkore/vulkore.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t BLK = 32;  // quantisation block along k
constexpr uint32_t GRP = 64;  // column-group width of the k-major layouts

struct Shape { const char* name; uint32_t rows, cols, count; };

// Gemma 3 1B: 26 layers, hidden 1152, ffn 6912, 4 heads x 256, 1 kv head.
const Shape kShapes[] = {
    {"qkv",     1152,   1536, 26},
    {"o_proj",  1024,   1152, 26},
    {"gate",    1152,   6912, 26},
    {"up",      1152,   6912, 26},
    {"down",    6912,   1152, 26},
    {"lm_head", 1152, 262144,  1},
};

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Deterministic pseudo-random nibble for weight (k, j). Generated on the fly so
// no layout ever needs a rows*cols byte staging array (lm_head alone is 302 M
// weights). Every layout packs the SAME logical matrix, which is what makes the
// cross-variant comparison meaningful.
inline uint32_t nib(uint32_t k, uint32_t j) {
    uint32_t h = k * 2654435761u ^ (j + 0x9e3779b9u) * 2246822519u;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h & 15u;
}

// --- the four packings (see the layout notes at the top of llm_matvec.cl) ---

// A "COL": word[(k*cols + j)/8], nibble (j&7). What kernels/llm.cl expects.
std::vector<uint32_t> pack_col(uint32_t rows, uint32_t cols) {
    std::vector<uint32_t> w(size_t(rows) * cols / 8, 0);
    for (uint32_t k = 0; k < rows; ++k)
        for (uint32_t j = 0; j < cols; ++j)
            w[(size_t(k) * cols + j) >> 3] |= nib(k, j) << ((j & 7u) * 4u);
    return w;
}

// B "KMN": word[j*(rows/8) + k/8], nibble (k&7).
std::vector<uint32_t> pack_kmn(uint32_t rows, uint32_t cols) {
    std::vector<uint32_t> w(size_t(rows) * cols / 8, 0);
    size_t nc = rows / 8;
    for (uint32_t j = 0; j < cols; ++j)
        for (uint32_t k = 0; k < rows; ++k)
            w[size_t(j) * nc + (k >> 3)] |= nib(k, j) << ((k & 7u) * 4u);
    return w;
}

// C "KM": word[(g*nc + k/8)*GRP + t], nibble (k&7), g = j/GRP, t = j%GRP.
std::vector<uint32_t> pack_km(uint32_t rows, uint32_t cols) {
    std::vector<uint32_t> w(size_t(rows) * cols / 8, 0);
    size_t nc = rows / 8;
    for (uint32_t j = 0; j < cols; ++j) {
        size_t base = size_t(j / GRP) * nc * GRP + (j % GRP);
        for (uint32_t k = 0; k < rows; ++k)
            w[base + size_t(k >> 3) * GRP] |= nib(k, j) << ((k & 7u) * 4u);
    }
    return w;
}

// D "KM4": uint4[(g*nblk + k/32)*GRP + t], component (k>>3)&3, nibble (k&7).
std::vector<uint32_t> pack_km4(uint32_t rows, uint32_t cols) {
    std::vector<uint32_t> w(size_t(rows) * cols / 8, 0);
    size_t nblk = rows / BLK;
    for (uint32_t j = 0; j < cols; ++j) {
        size_t base = (size_t(j / GRP) * nblk * GRP + (j % GRP)) * 4;
        for (uint32_t k = 0; k < rows; ++k)
            w[base + size_t(k / BLK) * GRP * 4 + ((k >> 3) & 3u)] |=
                nib(k, j) << ((k & 7u) * 4u);
    }
    return w;
}

enum class Layout { COL, KMN, KM, KM4 };

struct Variant {
    const char* name;
    const char* kernel;
    Layout layout;
    uint32_t cols_per_thread;  // register blocking factor (1 = none)
    uint32_t split;            // 0 = no split-k; else split factor
    const char* note;
};

const Variant kVariants[] = {
    {"ref (col)",     "mv_ref",        Layout::COL, 1, 0, "baseline, 8 nibbles per uint along cols"},
    {"ref+split8",    "mv_ref_split",  Layout::COL, 1, 8, "baseline layout, k split 8 ways + reduce"},
    {"kmn",           "mv_kmn",        Layout::KMN, 1, 0, "naive k-major: contiguous per thread, strided per wave"},
    {"km (uint)",     "mv_km",         Layout::KM,  1, 0, "group-interleaved k-major, 4-byte loads"},
    {"km4 (uint4)",   "mv_km4",        Layout::KM4, 1, 0, "group-interleaved k-major, 16-byte loads"},
    {"km4 wg32",      "mv_km4_wg32",   Layout::KM4, 1, 0, "km4 with reqd_work_group_size(32)"},
    {"km4 wg128",     "mv_km4_wg128",  Layout::KM4, 1, 0, "km4 with reqd_work_group_size(128)"},
    {"km4 wg256",     "mv_km4_wg256",  Layout::KM4, 1, 0, "km4 with reqd_work_group_size(256)"},
    {"km4 x2cols",    "mv_km4_c2",     Layout::KM4, 2, 0, "km4 + 2 output columns per thread"},
    {"km4 x2 wg32",   "mv_km4_c2_wg32",Layout::KM4, 2, 0, "km4 + 2 cols/thread + wgsize 32"},
    {"km4 x4cols",    "mv_km4_c4",     Layout::KM4, 4, 0, "km4 + 4 output columns per thread"},
    {"km4+split4",    "mv_km4_split",  Layout::KM4, 1, 4, "km4, k split 4 ways + reduce"},
    {"km4+split8",    "mv_km4_split",  Layout::KM4, 1, 8, "km4, k split 8 ways + reduce"},
    {"km4+split16",   "mv_km4_split",  Layout::KM4, 1, 16, "km4, k split 16 ways + reduce"},
};
constexpr size_t kNVariants = sizeof(kVariants) / sizeof(kVariants[0]);

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <llm.spv> <llm_matvec.spv> [--quick]\n", argv[0]);
        return 2;
    }
    bool quick = argc > 3 && std::string(argv[3]) == "--quick";

    vulkore::Context ctx;
    std::printf("device: %s\n", ctx.device_name().c_str());

    auto ref_prog = vulkore::Program::from_file(ctx, argv[1]);
    auto gold_k = ref_prog.kernel("matvec_q4");     // the thing we must match
    auto prog = vulkore::Program::from_file(ctx, argv[2]);

    std::vector<vulkore::Kernel> kernels;
    kernels.reserve(kNVariants);
    for (const auto& v : kVariants) kernels.push_back(prog.kernel(v.kernel));
    auto reduce_k = prog.kernel("mv_reduce");
    auto stream_k = prog.kernel("stream4");

    const int R = quick ? 2 : 7;

    // --- pure-load bandwidth ceiling, same device, same submit path ---------
    // Swept over thread counts, because a single point is not a ceiling: too
    // few threads and you measure occupancy, too many and you measure launch
    // overhead. The BEST of the sweep is the number every matvec below is
    // really competing against — and on the phone it is nowhere near the
    // 60-77 GB/s the LPDDR5X spec sheet promises.
    double stream_best = 0;
    {
        // 256 MB, not 512. At 512 MB the laptop's turnip driver reported
        // 311 GB/s while 64/128/256 MB all scaled linearly at 55-65 — the
        // buffer exceeds maxStorageBufferRange there, so every read past the
        // bound is squashed by robustBufferAccess and returns zero for free.
        // Free reads look like infinite bandwidth. 256 MB scales honestly on
        // both devices (verified by a size sweep before picking this).
        const size_t n4 = 16u << 20;              // 16 M uint4 = 256 MB
        auto src = ctx.alloc<uint32_t>(n4 * 4);
        auto out = ctx.alloc<float>(4);
        // Fill with DISTINCT, high-entropy data. A constant fill made the
        // laptop's turnip driver report 433 GB/s for a part whose LPDDR5X tops
        // out near 135 — something along the path (framebuffer compression or
        // page dedup) was serving the repeats without touching DRAM. Random
        // bytes cannot be compressed, so the number becomes a real one.
        std::vector<uint32_t> h(1u << 20);
        for (size_t off = 0; off < n4 * 4; off += h.size()) {
            for (size_t i = 0; i < h.size(); ++i) {
                uint32_t x = uint32_t(off + i) * 2654435761u;
                x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
                h[i] = x;
            }
            src.upload(std::span<const uint32_t>(h),
                       VkDeviceSize(off) * sizeof(uint32_t));
        }
        std::printf("stream4 pure-load probe (256 MB, 16-byte loads):\n");
        for (uint32_t sh = 12; sh <= 21; sh += 3) {
            const uint32_t threads = 1u << sh;
            vulkore::launch(stream_k, {threads}, out, src, uint32_t(n4), threads).wait();
            double ms = 1e30;
            for (int r = 0; r < R; ++r) {
                double t0 = now_ms();
                vulkore::launch(stream_k, {threads}, out, src, uint32_t(n4), threads).wait();
                ms = std::min(ms, now_ms() - t0);
            }
            double gbs = double(n4) * 16.0 / (ms * 1e6);
            stream_best = std::max(stream_best, gbs);
            std::printf("    %8u threads: %8.2f ms  %7.1f GB/s\n", threads, ms, gbs);
        }
        std::printf("  ceiling: %.1f GB/s\n\n", stream_best);
    }

    // Per-variant accumulators for the best-of-per-shape total.
    double var_total_ms[kNVariants] = {0};
    bool var_ok[kNVariants], var_partial[kNVariants];
    for (size_t v = 0; v < kNVariants; ++v) { var_ok[v] = true; var_partial[v] = false; }

    double best_total_ms = 0, ref_total_ms = 0;
    uint64_t total_bytes = 0;

    for (const auto& s : kShapes) {
        const size_t nweights = size_t(s.rows) * s.cols;
        const size_t nwords = nweights / 8;         // int4 -> 8 per uint
        const size_t nblk = s.rows / BLK;
        const size_t wbytes = nweights / 2;

        std::printf("== %s  rows=%u cols=%u  x%u  (%.1f MB/token)\n",
                    s.name, s.rows, s.cols, s.count, double(wbytes) * s.count / 1e6);

        // Inputs: real random data, NOT a constant fill. A constant weight word
        // makes every packing produce the same answer, so packing bugs hide.
        std::vector<float> hin(s.rows), hs(nblk * s.cols);
        for (uint32_t i = 0; i < s.rows; ++i)
            hin[i] = std::sin(double(i) * 0.37) * 0.5f;
        for (size_t i = 0; i < hs.size(); ++i)
            hs[i] = 0.02f + 0.001f * float(i % 7);

        auto in = ctx.alloc<float>(s.rows);
        auto sc = ctx.alloc<float>(hs.size());
        auto out = ctx.alloc<float>(s.cols);
        in.upload(std::span<const float>(hin));
        sc.upload(std::span<const float>(hs));

        // --- gold: kernels/llm.cl matvec_q4 on the COL packing -------------
        std::vector<float> gold(s.cols);
        auto wcol = ctx.alloc<uint32_t>(nwords);
        {
            auto hw = pack_col(s.rows, s.cols);
            wcol.upload(std::span<const uint32_t>(hw));
            std::vector<float> poison(s.cols, -1e30f);
            out.upload(std::span<const float>(poison));
            vulkore::launch(gold_k, {s.cols}, out, in, wcol, sc, s.rows, s.cols).wait();
            out.download(std::span<float>(gold));
        }
        double gmax = 0;
        for (float g : gold) gmax = std::max(gmax, double(std::fabs(g)));
        if (gmax == 0.0) {
            std::printf("  REFERENCE PRODUCED ALL ZEROS — aborting, nothing to compare against\n");
            return 1;
        }

        // Weight buffers per layout, built lazily and kept for the whole shape.
        // lm_head is 151 MB per layout, so four layouts is ~600 MB resident;
        // they are freed when the shape's scope ends.
        auto wkmn = ctx.alloc<uint32_t>(nwords);
        auto wkm = ctx.alloc<uint32_t>(nwords);
        auto wkm4 = ctx.alloc<uint32_t>(nwords);
        { auto h = pack_kmn(s.rows, s.cols); wkmn.upload(std::span<const uint32_t>(h)); }
        { auto h = pack_km(s.rows, s.cols);  wkm.upload(std::span<const uint32_t>(h)); }
        { auto h = pack_km4(s.rows, s.cols); wkm4.upload(std::span<const uint32_t>(h)); }

        std::printf("  %-14s %9s %9s %9s  %s\n", "variant", "ms/token", "GB/s", "vs ref", "check");

        double shape_best_ms = 1e30;
        const char* shape_best_name = "-";
        double shape_ref_ms = 0;

        for (size_t vi = 0; vi < kNVariants; ++vi) {
            const Variant& v = kVariants[vi];
            vulkore::Buffer* w = nullptr;
            switch (v.layout) {
                case Layout::COL: w = &wcol; break;
                case Layout::KMN: w = &wkmn; break;
                case Layout::KM:  w = &wkm;  break;
                case Layout::KM4: w = &wkm4; break;
            }
            const uint32_t SP = v.split;
            const uint32_t threads =
                SP ? s.cols * SP : (s.cols + v.cols_per_thread - 1) / v.cols_per_thread;

            // maxComputeWorkGroupCount is 65535 on this driver, so split-k over
            // a 262144-column lm_head asks for more workgroups than exist. It
            // would also be pointless there — 262144 columns is already far
            // more parallelism than the GPU can use.
            const uint32_t local = kernels[vi].local_size()[0];
            if (uint64_t(threads + local - 1) / local > 65535ull) {
                std::printf("  %-14s %9s %9s %9s  skipped: %u workgroups exceeds "
                            "the 65535 limit\n", v.name, "-", "-", "-",
                            uint32_t((threads + local - 1) / local));
                var_partial[vi] = true;
                continue;
            }

            // Split-k needs a partials buffer; allocate only when used.
            std::vector<vulkore::Buffer> tmp;
            if (SP) tmp.push_back(ctx.alloc<float>(size_t(s.cols) * SP));

            // --- correctness: poison, run once, compare ---------------------
            std::vector<float> poison(s.cols, -1e30f);
            out.upload(std::span<const float>(poison));
            {
                vulkore::Batch b(ctx);
                if (SP) {
                    b.add(kernels[vi], {threads}, tmp[0], in, *w, sc, s.rows, s.cols, SP);
                    b.add(reduce_k, {s.cols}, out, tmp[0], s.cols, SP);
                } else {
                    b.add(kernels[vi], {threads}, out, in, *w, sc, s.rows, s.cols);
                }
                b.submit().wait();
            }
            std::vector<float> got(s.cols);
            out.download(std::span<float>(got));

            double maxerr = 0;
            size_t untouched = 0;
            for (uint32_t j = 0; j < s.cols; ++j) {
                if (got[j] == -1e30f) ++untouched;
                maxerr = std::max(maxerr, double(std::fabs(got[j] - gold[j])));
            }
            const double rel = maxerr / gmax;
            const bool ok = untouched == 0 && rel < 1e-4;
            if (!ok) var_ok[vi] = false;

            // --- timing: one Batch per token group, ONE submit --------------
            {
                vulkore::Batch warm(ctx);
                for (uint32_t i = 0; i < s.count; ++i) {
                    if (SP) {
                        warm.add(kernels[vi], {threads}, tmp[0], in, *w, sc, s.rows, s.cols, SP);
                        warm.add(reduce_k, {s.cols}, out, tmp[0], s.cols, SP);
                    } else {
                        warm.add(kernels[vi], {threads}, out, in, *w, sc, s.rows, s.cols);
                    }
                }
                warm.submit().wait();
            }
            // BEST of R, not the mean. A phone GPU shares memory with the rest
            // of the system and throttles under sustained load, so the mean
            // drifts run to run (the same shape measured 2.66 and 2.01 ms in
            // two consecutive full runs). The minimum is the reproducible
            // quantity: it is what the kernel achieves when nothing else is in
            // the way, which is what a kernel comparison wants to isolate.
            double ms = 1e30;
            for (int r = 0; r < R; ++r) {
                double t0 = now_ms();
                vulkore::Batch b(ctx);
                for (uint32_t i = 0; i < s.count; ++i) {
                    if (SP) {
                        b.add(kernels[vi], {threads}, tmp[0], in, *w, sc, s.rows, s.cols, SP);
                        b.add(reduce_k, {s.cols}, out, tmp[0], s.cols, SP);
                    } else {
                        b.add(kernels[vi], {threads}, out, in, *w, sc, s.rows, s.cols);
                    }
                }
                b.submit().wait();
                ms = std::min(ms, now_ms() - t0);
            }
            const double gbs = double(wbytes) * s.count / (ms * 1e6);

            if (vi == 0) { shape_ref_ms = ms; ref_total_ms += ms; }
            var_total_ms[vi] += ms;
            if (ok && ms < shape_best_ms) { shape_best_ms = ms; shape_best_name = v.name; }

            char check[64];
            if (untouched)
                std::snprintf(check, sizeof check, "FAIL %zu cols untouched", untouched);
            else if (!ok)
                std::snprintf(check, sizeof check, "FAIL rel=%.2e", rel);
            else
                std::snprintf(check, sizeof check, "ok  (rel %.1e)", rel);

            std::printf("  %-14s %9.2f %9.1f %8.2fx  %s\n",
                        v.name, ms, gbs, shape_ref_ms / ms, check);
        }

        best_total_ms += shape_best_ms;
        total_bytes += uint64_t(wbytes) * s.count;
        std::printf("  best: %s  %.2f ms  (%.2fx ref)\n\n",
                    shape_best_name, shape_best_ms, shape_ref_ms / shape_best_ms);
    }

    // --- summary -----------------------------------------------------------
    std::printf("== per-variant totals across the whole decode step\n");
    std::printf("  %-14s %10s %9s %11s  %s\n",
                "variant", "ms/token", "GB/s", "tokens/sec", "correct");
    for (size_t vi = 0; vi < kNVariants; ++vi) {
        std::printf("  %-14s %10.1f %9.1f %11.1f  %s\n",
                    kVariants[vi].name, var_total_ms[vi],
                    double(total_bytes) / (var_total_ms[vi] * 1e6),
                    1000.0 / var_total_ms[vi],
                    var_partial[vi] ? "n/a (some shapes skipped)"
                                    : (var_ok[vi] ? "yes" : "NO"));
    }
    std::printf("\n  weights/token : %.0f MB (int4)\n", double(total_bytes) / 1e6);
    std::printf("  reference     : %.1f ms  %.1f GB/s  %.1f tok/s\n",
                ref_total_ms, double(total_bytes) / (ref_total_ms * 1e6), 1000.0 / ref_total_ms);
    std::printf("  best-per-shape: %.1f ms  %.1f GB/s  %.1f tok/s  (%.2fx)\n",
                best_total_ms, double(total_bytes) / (best_total_ms * 1e6),
                1000.0 / best_total_ms, ref_total_ms / best_total_ms);
    std::printf("  pure-load probe: %.1f GB/s  ->  best-per-shape reaches %.0f%% of it\n",
                stream_best, 100.0 * double(total_bytes) / (best_total_ms * 1e6) / stream_best);
    return 0;
}
