// Phone validation: every LLM kernel, on the REAL device, with liveness proof.
//
// WHY THIS EXISTS
// ---------------
// examples/llm/kernel_tests.cpp verified kernels/llm_transformer.cl on the
// laptop's Adreno X1-85 via Mesa turnip. The phone runs Qualcomm's PROPRIETARY
// driver. That difference has burned this repo twice:
//
//   1. An 8th-power Mandelbulb using pow/acos/atan2 compiled clean, passed
//      spirv-val, bound through vulkore::Program and launched with no error —
//      and SILENTLY DID NOT EXECUTE on Adreno. Output pixels were left
//      uninitialised (alpha 0xDB, a value the shader cannot emit).
//   2. Six buffer-transfer tests passed on all three desktop drivers and failed
//      on Mali because of non-coherent HOST_CACHED memory
//      (agent-docs/mali-coherency-fix.md).
//
// So "passes on turnip" is not evidence about the phone, and "launch() raised
// nothing" is not evidence the kernel ran. This binary covers 24 kernels
// (19 of the 25 in llm_transformer.cl + all 5 in llm.cl) and checks three things per kernel:
//
//   PARITY   — allclose vs a CPU reference, |gpu-cpu| <= atol + rtol*|cpu|.
//              The atol term is load-bearing: a pure relative bound produces
//              false failures on outputs that legitimately land near zero
//              (attention_apply sums 131 products; the batched block chains
//              five kernels). kernel_tests.cpp hit exactly that.
//   LIVENESS — proof the kernel WROTE its output. Two detectors:
//                * out-of-place: pre-fill the output with a sentinel the kernel
//                  cannot legitimately produce (-1e30) and require every
//                  element to be gone.
//                * in-place: a sentinel is impossible (the kernel reads what it
//                  overwrites), so instead require that every element the CPU
//                  reference CHANGES was actually changed on the GPU. A no-op
//                  leaves the buffer byte-identical to the input and is caught.
//              A silent no-op otherwise reads as "output unchanged", which for
//              an in-place kernel looks exactly like a pass.
//   COVERAGE — every kernel name the module exports must have been exercised.
//              A kernel nobody tests is a kernel nobody knows about.
//
// The liveness detectors are themselves validated by NEGATIVE CONTROLS: two
// kernels are deliberately launched at HALF grid, and the run only counts as
// trustworthy if the detectors flag the untouched half. A detector that has
// never fired is not a detector.
//
// Build for the phone (see agent-docs/environment.md RESOLUTION):
//   export QEMU_LD_PREFIX=/home/nikhil/x86_64-sysroot
//   NDK=/home/nikhil/Android/Sdk/ndk-r27c
//   TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64
//   $TC/bin/aarch64-linux-android26-clang++ -std=c++20 -O2 -static-libstdc++ \
//     -Iinclude -Ithird_party/Vulkan-Headers/include -Ithird_party/volk \
//     -Ithird_party/VulkanMemoryAllocator/include \
//     -DVMA_STATIC_VULKAN_FUNCTIONS=0 -DVMA_DYNAMIC_VULKAN_FUNCTIONS=1 \
//     examples/llm/phone_validation.cpp build-android/libvulkore.a -ldl \
//     -o build-android/llm_phone_validation
//
// Run:  ./llm_phone_validation [llm_transformer.spv] [llm.spv]

#include <vulkore/vulkore.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

// ---- Gemma 3 1B shapes (same as kernel_tests.cpp) --------------------------
static constexpr uint32_t kHidden   = 1152;
static constexpr uint32_t kFfn      = 6912;
static constexpr uint32_t kHeads    = 4;
static constexpr uint32_t kKvHeads  = 1;
static constexpr uint32_t kHeadDim  = 256;
static constexpr uint32_t kSeqLen   = 131;   // deliberately not a multiple of 64
static constexpr float    kEps      = 1e-6f;
static constexpr float    kRopeBase = 1000000.0f;

static constexpr float kSentinel = -1.0e30f;
static constexpr float kRtol     = 1e-4f;
static constexpr float kAtol     = 1e-6f;

// ---- results table ---------------------------------------------------------
struct Row {
    std::string module;
    std::string name;
    bool        pass = false;
    double      max_rel = 0.0;
    double      max_abs = 0.0;
    const char* liveness = "n/a";   // "yes" / "NO" / "n/a"
    std::string note;
};
static std::vector<Row> g_rows;
static int g_pass = 0, g_fail = 0;
static std::string g_module;   // module label for the rows being emitted

static void emit(Row r) {
    r.module = g_module;
    r.pass ? ++g_pass : ++g_fail;
    std::printf("  %-22s %s  max_rel=%.3e  max_abs=%.3e  live=%-3s %s\n",
                r.name.c_str(), r.pass ? "PASS" : "FAIL", r.max_rel, r.max_abs,
                r.liveness, r.note.c_str());
    g_rows.push_back(std::move(r));
}

// allclose accounting shared by both detectors. max_rel uses a 1e-3 denominator
// floor purely so the printed ratio stays readable near zero; the verdict uses
// the allclose rule.
struct Acc {
    double max_rel = 0.0, max_abs = 0.0;
    int outside = 0, nonfinite = 0;
    size_t worst = 0;
};
static Acc compare(const std::vector<float>& gpu, const std::vector<float>& cpu) {
    Acc a;
    for (size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(gpu[i])) ++a.nonfinite;
        double d = std::fabs(double(gpu[i]) - double(cpu[i]));
        double r = d / std::max(std::fabs(double(cpu[i])), 1e-3);
        if (d > kAtol + double(kRtol) * std::fabs(double(cpu[i]))) {
            if (!a.outside) a.worst = i;
            ++a.outside;
        }
        if (r > a.max_rel && !a.outside) a.worst = i;
        a.max_rel = std::max(a.max_rel, r);
        a.max_abs = std::max(a.max_abs, d);
    }
    return a;
}

static std::string worst_note(const Acc& a, const std::vector<float>& gpu,
                              const std::vector<float>& cpu) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "worst[%zu] gpu=%.9g cpu=%.9g",
                  a.worst, gpu[a.worst], cpu[a.worst]);
    return buf;
}

// --- DETECTOR 1: out-of-place. Output was pre-filled with kSentinel. --------
// Returns the number of untouched elements so negative controls can assert on it.
static int check_sentinel(const char* name, const std::vector<float>& gpu,
                          const std::vector<float>& cpu, bool record = true) {
    Acc a = compare(gpu, cpu);
    int untouched = 0;
    for (float g : gpu) if (g == kSentinel) ++untouched;
    bool ok = a.outside == 0 && a.nonfinite == 0 && untouched == 0;
    std::string note;
    if (untouched) note += "UNTOUCHED=" + std::to_string(untouched) + " ";
    if (a.nonfinite) note += "NONFINITE=" + std::to_string(a.nonfinite) + " ";
    if (!ok && a.outside) note += worst_note(a, gpu, cpu);
    if (record)
        emit({{}, name, ok, a.max_rel, a.max_abs, untouched ? "NO" : "yes", note});
    return untouched;
}

// --- DETECTOR 2: in-place. -------------------------------------------------
// A sentinel is impossible here: the kernel reads the very elements it writes.
// So liveness is "every element the reference CHANGES was actually changed".
// A silent no-op leaves the buffer identical to `before` and every expected
// change shows up as unchanged.
static int check_inplace(const char* name, const std::vector<float>& gpu,
                         const std::vector<float>& cpu,
                         const std::vector<float>& before, bool record = true) {
    Acc a = compare(gpu, cpu);
    int expect_changed = 0, unchanged = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        if (cpu[i] != before[i]) {
            ++expect_changed;
            if (gpu[i] == before[i]) ++unchanged;
        }
    }
    bool ok = a.outside == 0 && a.nonfinite == 0 && unchanged == 0 && expect_changed > 0;
    std::string note = "changed " + std::to_string(expect_changed - unchanged) +
                       "/" + std::to_string(expect_changed);
    if (unchanged) note += " NOT-WRITTEN=" + std::to_string(unchanged);
    if (a.nonfinite) note += " NONFINITE=" + std::to_string(a.nonfinite);
    if (!ok && a.outside) note += " " + worst_note(a, gpu, cpu);
    if (record)
        emit({{}, name, ok, a.max_rel, a.max_abs, unchanged ? "NO" : "yes", note});
    return unchanged;
}

// ---- CPU references (same maths as kernel_tests.cpp) -----------------------
static float tanh_approx(float z) {
    z = std::fmax(-10.0f, std::fmin(10.0f, z));
    float e = std::exp(-2.0f * z);
    return (1.0f - e) / (1.0f + e);
}
static float gelu_tanh(float g) {
    float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (1.0f + tanh_approx(inner));
}
static void ref_rmsnorm(std::vector<float>& out, const std::vector<float>& in,
                        const std::vector<float>& w, uint32_t n, float eps,
                        bool gemma) {
    float ss = 0.0f;
    for (uint32_t k = 0; k < n; ++k) ss += in[k] * in[k];
    float scale = 1.0f / std::sqrt(ss / float(n) + eps);
    for (uint32_t i = 0; i < n; ++i)
        out[i] = in[i] * scale * (gemma ? 1.0f + w[i] : w[i]);
}
static void ref_rope(std::vector<float>& x, uint32_t n_heads, uint32_t head_dim,
                     uint32_t pos, float theta) {
    uint32_t half = head_dim / 2;
    for (uint32_t h = 0; h < n_heads; ++h) {
        for (uint32_t d = 0; d < half; ++d) {
            uint32_t base = h * head_dim + d;
            float frac = float(2 * d) / float(head_dim);
            float inv = std::exp2(-std::log2(theta) * frac);
            float ang = float(pos) * inv;
            float c = std::cos(ang), s = std::sin(ang);
            float lo = x[base], hi = x[base + half];
            x[base] = lo * c - hi * s;
            x[base + half] = lo * s + hi * c;
        }
    }
}

// q4 packing, "COL" layout — word[(k*cols+j)/8], nibble (j&7). This is what
// kernels/llm.cl expects; same helper as examples/llm/matvec_bench.cpp.
static inline uint32_t nib(uint32_t k, uint32_t j) {
    uint32_t h = k * 2654435761u ^ (j + 0x9e3779b9u) * 2246822519u;
    h ^= h >> 15; h *= 2654435761u; h ^= h >> 13;
    return h & 15u;
}

// ---- coverage tracking -----------------------------------------------------
static std::vector<std::string> g_tested;
static void tested(const char* n) { g_tested.push_back(n); }

static bool coverage(vulkore::Program& prog, const char* label) {
    auto names = prog.kernel_names();
    std::vector<std::string> missing;
    for (const auto& n : names)
        if (std::find(g_tested.begin(), g_tested.end(), n) == g_tested.end())
            missing.push_back(n);
    std::printf("  coverage: %zu/%zu kernels in %s exercised",
                names.size() - missing.size(), names.size(), label);
    for (const auto& m : missing) std::printf("  MISSING:%s", m.c_str());
    std::printf("\n");
    if (!missing.empty()) ++g_fail;
    return missing.empty();
}

// ===========================================================================
int main(int argc, char** argv) {
    const char* spv_tf  = argc > 1 ? argv[1] : "kernels/llm_transformer.spv";
    const char* spv_llm = argc > 2 ? argv[2] : "kernels/llm.spv";

    vulkore::Context ctx;
    std::printf("device : %s\n", ctx.device_name().c_str());
    std::printf("modules: %s , %s\n\n", spv_tf, spv_llm);

    std::mt19937 rng(20260719);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);
    auto rand_vec = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& e : v) e = u(rng);
        return v;
    };
    auto sentinel_vec = [](size_t n) { return std::vector<float>(n, kSentinel); };

    // Negative-control outcomes, filled in at the end of the run.
    int nc_sentinel_flagged = -1, nc_sentinel_expect = 0;
    int nc_inplace_flagged  = -1, nc_inplace_expect  = 0;

    // =====================================================================
    // MODULE 1 — kernels/llm_transformer.cl (19 kernels)
    // =====================================================================
    {
        auto prog = vulkore::Program::from_file(ctx, spv_tf);
        g_module = "llm_transformer.cl";
        std::printf("== %s (%zu kernels) ==\n", g_module.c_str(),
                    prog.kernel_names().size());

        // ---- rmsnorm / rmsnorm_gemma -----------------------------------
        {
            auto in = rand_vec(kHidden), w = rand_vec(kHidden);
            auto d_in = ctx.alloc<float>(kHidden);
            auto d_w  = ctx.alloc<float>(kHidden);
            auto d_o  = ctx.alloc<float>(kHidden);
            d_in.upload(std::span<const float>(in));
            d_w.upload(std::span<const float>(w));
            for (bool gemma : {false, true}) {
                auto seed = sentinel_vec(kHidden);
                d_o.upload(std::span<const float>(seed));
                auto k = prog.kernel(gemma ? "rmsnorm_gemma" : "rmsnorm");
                vulkore::launch(k, {kHidden}, d_o, d_in, d_w, kHidden, kEps).wait();
                std::vector<float> gpu(kHidden), cpu(kHidden);
                d_o.download(std::span<float>(gpu));
                ref_rmsnorm(cpu, in, w, kHidden, kEps, gemma);
                check_sentinel(gemma ? "rmsnorm_gemma" : "rmsnorm", gpu, cpu);
                tested(gemma ? "rmsnorm_gemma" : "rmsnorm");
            }

            // ---- NEGATIVE CONTROL 1: does the sentinel detector fire? ----
            // Same kernel, same inputs, HALF the grid. The upper half of the
            // output must still hold the sentinel, and check_sentinel() must
            // say so. If this "passes", the detector is blind and every other
            // liveness result in this run is worthless.
            {
                auto seed = sentinel_vec(kHidden);
                d_o.upload(std::span<const float>(seed));
                auto k = prog.kernel("rmsnorm_gemma");
                vulkore::launch(k, {kHidden / 2}, d_o, d_in, d_w, kHidden, kEps).wait();
                std::vector<float> gpu(kHidden), cpu(kHidden);
                d_o.download(std::span<float>(gpu));
                ref_rmsnorm(cpu, in, w, kHidden, kEps, true);
                nc_sentinel_expect = int(kHidden) / 2;
                nc_sentinel_flagged =
                    check_sentinel("[negctl] half grid", gpu, cpu, /*record=*/false);
            }
        }

        // ---- rmsnorm_sumsq + rmsnorm_apply_gemma (two-pass) -------------
        {
            const uint32_t nparts = 64;
            auto in = rand_vec(kHidden), w = rand_vec(kHidden);
            auto d_in = ctx.alloc<float>(kHidden);
            auto d_w  = ctx.alloc<float>(kHidden);
            auto d_p  = ctx.alloc<float>(nparts);
            auto d_o  = ctx.alloc<float>(kHidden);
            d_in.upload(std::span<const float>(in));
            d_w.upload(std::span<const float>(w));
            auto pseed = sentinel_vec(nparts);
            auto oseed = sentinel_vec(kHidden);
            d_p.upload(std::span<const float>(pseed));
            d_o.upload(std::span<const float>(oseed));

            auto k1 = prog.kernel("rmsnorm_sumsq");
            auto k2 = prog.kernel("rmsnorm_apply_gemma");
            vulkore::Batch b(ctx);
            b.add(k1, {nparts}, d_p, d_in, kHidden, nparts);
            b.add(k2, {kHidden}, d_o, d_in, d_w, d_p, kHidden, nparts, kEps);
            b.submit().wait();

            // Pass 1 is checked on its own so a dead sumsq cannot hide behind a
            // live apply — the two run in ONE submit with a barrier between.
            std::vector<float> gp(nparts), cp(nparts);
            d_p.download(std::span<float>(gp));
            for (uint32_t p = 0; p < nparts; ++p) {
                float acc = 0.0f;
                for (uint32_t k = p; k < kHidden; k += nparts) acc += in[k] * in[k];
                cp[p] = acc;
            }
            check_sentinel("rmsnorm_sumsq", gp, cp);
            tested("rmsnorm_sumsq");

            std::vector<float> gpu(kHidden), cpu(kHidden);
            d_o.download(std::span<float>(gpu));
            // The partial-sum tree changes the summation ORDER versus the serial
            // reference, so this is a slightly different (better-conditioned)
            // answer, not a bit-identical one.
            ref_rmsnorm(cpu, in, w, kHidden, kEps, true);
            check_sentinel("rmsnorm_apply_gemma", gpu, cpu);
            tested("rmsnorm_apply_gemma");
        }

        // ---- rmsnorm_heads_gemma ----------------------------------------
        {
            uint32_t n = kHeads * kHeadDim;
            auto in = rand_vec(n), w = rand_vec(kHeadDim);
            auto d_in = ctx.alloc<float>(n);
            auto d_w  = ctx.alloc<float>(kHeadDim);
            auto d_o  = ctx.alloc<float>(n);
            d_in.upload(std::span<const float>(in));
            d_w.upload(std::span<const float>(w));
            auto seed = sentinel_vec(n);
            d_o.upload(std::span<const float>(seed));
            auto k = prog.kernel("rmsnorm_heads_gemma");
            vulkore::launch(k, {n}, d_o, d_in, d_w, kHeads, kHeadDim, kEps).wait();
            std::vector<float> gpu(n), cpu(n);
            d_o.download(std::span<float>(gpu));
            for (uint32_t h = 0; h < kHeads; ++h) {
                float ss = 0.0f;
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    float v = in[h * kHeadDim + d];
                    ss += v * v;
                }
                float sc = 1.0f / std::sqrt(ss / float(kHeadDim) + kEps);
                for (uint32_t d = 0; d < kHeadDim; ++d)
                    cpu[h * kHeadDim + d] = in[h * kHeadDim + d] * sc * (1.0f + w[d]);
            }
            check_sentinel("rmsnorm_heads_gemma", gpu, cpu);
            tested("rmsnorm_heads_gemma");
        }

        // ---- rope / rope_cached (both IN-PLACE) --------------------------
        // These are the transcendental kernels — exp2/log2/sin/cos. The
        // Mandelbulb that silently no-opped on Adreno was a transcendental
        // shader, so this is the highest-suspicion pair in the module and the
        // in-place detector is what makes a no-op visible here.
        {
            uint32_t n = kHeads * kHeadDim;
            const uint32_t pos = 17;
            auto in = rand_vec(n);
            auto cpu = in;
            ref_rope(cpu, kHeads, kHeadDim, pos, kRopeBase);

            auto d = ctx.alloc<float>(n);
            d.upload(std::span<const float>(in));
            auto k = prog.kernel("rope");
            vulkore::launch(k, {kHeads * kHeadDim / 2}, d, kHeads, kHeadDim, pos,
                          kRopeBase).wait();
            std::vector<float> gpu(n);
            d.download(std::span<float>(gpu));
            check_inplace("rope", gpu, cpu, in);
            tested("rope");

            // ---- NEGATIVE CONTROL 2: does the in-place detector fire? -----
            // Half the pairs rotated, half left exactly as uploaded. The
            // detector must flag the untouched half.
            {
                auto d2 = ctx.alloc<float>(n);
                d2.upload(std::span<const float>(in));
                vulkore::launch(k, {kHeads * kHeadDim / 4}, d2, kHeads, kHeadDim,
                              pos, kRopeBase).wait();
                std::vector<float> half(n);
                d2.download(std::span<float>(half));
                // Expected untouched: the pairs beyond the half grid. Count
                // from the reference rather than assuming, since d=0 rotates by
                // angle 0 and legitimately leaves its element unchanged.
                nc_inplace_expect = 0;
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t hh = i / kHeadDim, dd = i % kHeadDim;
                    uint32_t pair = hh * (kHeadDim / 2) + (dd % (kHeadDim / 2));
                    if (pair >= kHeads * kHeadDim / 4 && cpu[i] != in[i])
                        ++nc_inplace_expect;
                }
                nc_inplace_flagged =
                    check_inplace("[negctl] half grid", half, cpu, in, false);
            }

            // rope_cached: host-built angle table, no on-device trig. If `rope`
            // fails and this passes, the driver's transcendentals are the
            // problem, not the rotation — that is the diagnostic value of
            // running both.
            uint32_t half_dim = kHeadDim / 2;
            std::vector<float> ct(half_dim), st(half_dim);
            for (uint32_t dd = 0; dd < half_dim; ++dd) {
                float inv = std::exp2(-std::log2(kRopeBase) *
                                      float(2 * dd) / float(kHeadDim));
                ct[dd] = std::cos(float(pos) * inv);
                st[dd] = std::sin(float(pos) * inv);
            }
            auto d2 = ctx.alloc<float>(n);
            auto d_ct = ctx.alloc<float>(half_dim);
            auto d_st = ctx.alloc<float>(half_dim);
            d2.upload(std::span<const float>(in));
            d_ct.upload(std::span<const float>(ct));
            d_st.upload(std::span<const float>(st));
            auto kc = prog.kernel("rope_cached");
            vulkore::launch(kc, {kHeads * half_dim}, d2, d_ct, d_st, kHeads,
                          kHeadDim).wait();
            std::vector<float> gpu2(n);
            d2.download(std::span<float>(gpu2));
            check_inplace("rope_cached", gpu2, cpu, in);
            tested("rope_cached");
        }

        // ---- geglu / swiglu / geglu_fused --------------------------------
        // exp() lives in all three; same transcendental suspicion as rope.
        {
            auto gate = rand_vec(kFfn), up = rand_vec(kFfn);
            auto d_g = ctx.alloc<float>(kFfn);
            auto d_u = ctx.alloc<float>(kFfn);
            auto d_o = ctx.alloc<float>(kFfn);
            d_g.upload(std::span<const float>(gate));
            d_u.upload(std::span<const float>(up));

            std::vector<float> cpu_ge(kFfn), cpu_sw(kFfn);
            for (uint32_t i = 0; i < kFfn; ++i) {
                cpu_ge[i] = gelu_tanh(gate[i]) * up[i];
                float g = gate[i];
                cpu_sw[i] = g * (1.0f / (1.0f + std::exp(-g))) * up[i];
            }
            for (const char* name : {"geglu", "swiglu"}) {
                auto seed = sentinel_vec(kFfn);
                d_o.upload(std::span<const float>(seed));
                auto k = prog.kernel(name);
                vulkore::launch(k, {kFfn}, d_o, d_g, d_u, kFfn).wait();
                std::vector<float> gpu(kFfn);
                d_o.download(std::span<float>(gpu));
                check_sentinel(name, gpu, std::string(name) == "geglu" ? cpu_ge : cpu_sw);
                tested(name);
            }

            std::vector<float> gu(2 * size_t(kFfn));
            for (uint32_t i = 0; i < kFfn; ++i) { gu[i] = gate[i]; gu[kFfn + i] = up[i]; }
            auto d_gu = ctx.alloc<float>(gu.size());
            d_gu.upload(std::span<const float>(gu));
            auto seed = sentinel_vec(kFfn);
            d_o.upload(std::span<const float>(seed));
            auto k = prog.kernel("geglu_fused");
            vulkore::launch(k, {kFfn}, d_o, d_gu, kFfn).wait();
            std::vector<float> gpu(kFfn);
            d_o.download(std::span<float>(gpu));
            check_sentinel("geglu_fused", gpu, cpu_ge);
            tested("geglu_fused");
        }

        // ---- attention_scores / softmax_rows / attention_apply -----------
        {
            const float scale = 1.0f / std::sqrt(float(kHeadDim));
            uint32_t nq = kHeads * kHeadDim;
            uint32_t ncache = kSeqLen * kKvHeads * kHeadDim;
            uint32_t nscore = kHeads * kSeqLen;
            uint32_t group = kHeads / kKvHeads;

            auto q = rand_vec(nq), kc = rand_vec(ncache), vc = rand_vec(ncache);
            auto d_q = ctx.alloc<float>(nq);
            auto d_k = ctx.alloc<float>(ncache);
            auto d_v = ctx.alloc<float>(ncache);
            auto d_s = ctx.alloc<float>(nscore);
            auto d_o = ctx.alloc<float>(nq);
            d_q.upload(std::span<const float>(q));
            d_k.upload(std::span<const float>(kc));
            d_v.upload(std::span<const float>(vc));

            auto seed = sentinel_vec(nscore);
            d_s.upload(std::span<const float>(seed));
            auto ks = prog.kernel("attention_scores");
            vulkore::launch(ks, {nscore}, d_s, d_q, d_k, kHeads, kKvHeads, kHeadDim,
                          kSeqLen, scale, 0u).wait();
            std::vector<float> gpu_s(nscore), cpu_s(nscore);
            d_s.download(std::span<float>(gpu_s));
            for (uint32_t h = 0; h < kHeads; ++h) {
                uint32_t kvh = h / group;
                for (uint32_t t = 0; t < kSeqLen; ++t) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < kHeadDim; ++d)
                        acc += q[h * kHeadDim + d] * kc[(t * kKvHeads + kvh) * kHeadDim + d];
                    cpu_s[h * kSeqLen + t] = acc * scale;
                }
            }
            check_sentinel("attention_scores", gpu_s, cpu_s);
            tested("attention_scores");

            // softmax_rows is IN-PLACE over the GPU's own scores. exp() again.
            auto before_sm = gpu_s;
            auto ksm = prog.kernel("softmax_rows");
            vulkore::launch(ksm, {kHeads}, d_s, kHeads, kSeqLen).wait();
            std::vector<float> gpu_p(nscore), cpu_p(nscore);
            d_s.download(std::span<float>(gpu_p));
            for (uint32_t h = 0; h < kHeads; ++h) {
                float m = -std::numeric_limits<float>::infinity(), sum = 0.0f;
                for (uint32_t t = 0; t < kSeqLen; ++t)
                    m = std::fmax(m, cpu_s[h * kSeqLen + t]);
                for (uint32_t t = 0; t < kSeqLen; ++t) {
                    float e = std::exp(cpu_s[h * kSeqLen + t] - m);
                    cpu_p[h * kSeqLen + t] = e;
                    sum += e;
                }
                for (uint32_t t = 0; t < kSeqLen; ++t) cpu_p[h * kSeqLen + t] /= sum;
            }
            check_inplace("softmax_rows", gpu_p, cpu_p, before_sm);
            tested("softmax_rows");

            // A softmax row that does not sum to 1 is wrong even if it matches
            // the reference, so check the invariant directly.
            double worst_row = 0.0;
            for (uint32_t h = 0; h < kHeads; ++h) {
                double s = 0.0;
                for (uint32_t t = 0; t < kSeqLen; ++t) s += gpu_p[h * kSeqLen + t];
                worst_row = std::max(worst_row, std::fabs(s - 1.0));
            }
            emit({{}, "softmax rowsum==1", worst_row < 1e-5, worst_row, worst_row,
                  "n/a", "invariant check"});

            auto seed_o = sentinel_vec(nq);
            d_o.upload(std::span<const float>(seed_o));
            auto ka = prog.kernel("attention_apply");
            vulkore::launch(ka, {nq}, d_o, d_s, d_v, kHeads, kKvHeads, kHeadDim,
                          kSeqLen, 0u).wait();
            std::vector<float> gpu_a(nq), cpu_a(nq);
            d_o.download(std::span<float>(gpu_a));
            for (uint32_t h = 0; h < kHeads; ++h) {
                uint32_t kvh = h / group;
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    float acc = 0.0f;
                    for (uint32_t t = 0; t < kSeqLen; ++t)
                        acc += cpu_p[h * kSeqLen + t] * vc[(t * kKvHeads + kvh) * kHeadDim + d];
                    cpu_a[h * kHeadDim + d] = acc;
                }
            }
            check_sentinel("attention_apply", gpu_a, cpu_a);
            tested("attention_apply");

            // ---- SLIDING WINDOW on the PROPRIETARY DRIVER ----------------
            // Everything above passes kv_start = 0, which cannot catch an
            // offset bug at all. Gemma 3 runs 22 of its 26 layers with
            // kv_start != 0 past position 512, so the offset path is the one
            // that actually executes in the decode loop and it needs its own
            // on-device evidence, not just laptop/turnip evidence.
            {
                const uint32_t t0 = kSeqLen / 3;   // deliberately not round
                const uint32_t span = kSeqLen - t0;
                const uint32_t nsc_w = kHeads * span;

                auto seed_w = sentinel_vec(nsc_w);
                d_s.upload(std::span<const float>(seed_w));
                vulkore::launch(ks, {nsc_w}, d_s, d_q, d_k, kHeads, kKvHeads, kHeadDim,
                              span, scale, t0).wait();
                std::vector<float> gpu_w(nsc_w), cpu_w(nsc_w);
                d_s.download(std::span<float>(gpu_w));
                for (uint32_t h = 0; h < kHeads; ++h) {
                    uint32_t kvh = h / group;
                    for (uint32_t j = 0; j < span; ++j) {
                        float acc = 0.0f;
                        for (uint32_t d = 0; d < kHeadDim; ++d)
                            acc += q[h * kHeadDim + d] *
                                   kc[((t0 + j) * kKvHeads + kvh) * kHeadDim + d];
                        cpu_w[h * span + j] = acc * scale;
                    }
                }
                check_sentinel("attention_scores window", gpu_w, cpu_w);

                // NEGATIVE CONTROL. If kv_start were ignored the windowed row
                // would equal the first `span` columns of the unwindowed one,
                // and the parity check above would still pass against a
                // reference that shares the mistake. This is the check that
                // cannot be fooled that way.
                double diff = 0.0;
                for (uint32_t h = 0; h < kHeads; ++h)
                    for (uint32_t j = 0; j < span; ++j)
                        diff = std::max(diff, std::fabs(double(gpu_w[h * span + j]) -
                                                        double(cpu_s[h * kSeqLen + j])));
                emit({{}, "kv_start is not ignored", diff > 1e-3, diff, diff, "n/a",
                      "negative control"});

                vulkore::launch(ksm, {kHeads}, d_s, kHeads, span).wait();
                std::vector<float> pw(nsc_w);
                d_s.download(std::span<float>(pw));

                auto seed_ow = sentinel_vec(nq);
                d_o.upload(std::span<const float>(seed_ow));
                vulkore::launch(ka, {nq}, d_o, d_s, d_v, kHeads, kKvHeads, kHeadDim,
                              span, t0).wait();
                std::vector<float> gpu_aw(nq), cpu_aw(nq);
                d_o.download(std::span<float>(gpu_aw));
                for (uint32_t h = 0; h < kHeads; ++h) {
                    uint32_t kvh = h / group;
                    for (uint32_t d = 0; d < kHeadDim; ++d) {
                        float acc = 0.0f;
                        for (uint32_t j = 0; j < span; ++j)
                            acc += pw[h * span + j] *
                                   vc[((t0 + j) * kKvHeads + kvh) * kHeadDim + d];
                        cpu_aw[h * kHeadDim + d] = acc;
                    }
                }
                check_sentinel("attention_apply window", gpu_aw, cpu_aw);
            }

            // ---- PARALLEL ATTENTION on the PROPRIETARY DRIVER ------------
            // These four replace softmax_rows / attention_scores /
            // attention_apply at every span above ~128, which is every layer of
            // a deep decode step — so they are what actually runs on the phone
            // past position 128 and they need on-device evidence of their own.
            // Both parallel forms carry an INTERMEDIATE buffer (the per-chunk
            // partials), and a dead first pass hiding behind a live second pass
            // is a failure mode this repo has already had; the intermediates
            // are sentinel-checked separately for that reason.
            {
                const uint32_t nparts = 8;   // 8 does NOT divide kSeqLen=131
                auto k_smp  = prog.kernel("softmax_rows_partial");
                auto k_smf  = prog.kernel("softmax_rows_finish");
                auto k_sms  = prog.kernel("softmax_rows_scale");
                auto k_smq  = prog.kernel("attention_scores_mq4");
                auto k_amq  = prog.kernel("attention_apply_mq4");
                auto k_ared = prog.kernel("attention_apply_reduce");

                auto d_sm  = ctx.alloc<float>(nscore);
                auto d_pm  = ctx.alloc<float>(kHeads * nparts);
                auto d_ps  = ctx.alloc<float>(kHeads * nparts);
                auto d_st  = ctx.alloc<float>(kHeads * 2);
                d_sm.upload(std::span<const float>(cpu_s));
                auto seed_pm = sentinel_vec(kHeads * nparts);
                d_pm.upload(std::span<const float>(seed_pm));

                vulkore::launch(k_smp, {kHeads * nparts}, d_pm, d_ps, d_sm, kHeads,
                              kSeqLen, nparts).wait();
                std::vector<float> hpm(kHeads * nparts);
                d_pm.download(std::span<float>(hpm));
                int pm_dead = 0;
                for (float v : hpm) if (v == kSentinel) ++pm_dead;
                emit({{}, "softmax partial live", pm_dead == 0, double(pm_dead),
                      double(pm_dead), "n/a", "intermediate sentinel"});

                vulkore::launch(k_smf, {kHeads}, d_st, d_pm, d_ps, kHeads, nparts).wait();
                vulkore::launch(k_sms, {nscore}, d_sm, d_st, kHeads, kSeqLen).wait();
                std::vector<float> gpu_sm(nscore);
                d_sm.download(std::span<float>(gpu_sm));
                // In-place-style: fed from cpu_s, so use the out-of-place
                // detector without the sentinel (the buffer was seeded with data).
                check_sentinel("softmax_rows parallel", gpu_sm, cpu_p);
                tested("softmax_rows_partial");
                tested("softmax_rows_finish");
                tested("softmax_rows_scale");

                auto d_s4 = ctx.alloc<float>(nscore);
                auto seed_s4 = sentinel_vec(nscore);
                d_s4.upload(std::span<const float>(seed_s4));
                vulkore::launch(k_smq, {kSeqLen}, d_s4, d_q, d_k, kHeadDim, kSeqLen,
                              scale, 0u).wait();
                std::vector<float> gpu_s4(nscore);
                d_s4.download(std::span<float>(gpu_s4));
                check_sentinel("attention_scores_mq4", gpu_s4, cpu_s);
                tested("attention_scores_mq4");

                const uint32_t split = 5;    // 5 does NOT divide kSeqLen=131
                auto d_pt = ctx.alloc<float>(size_t(nq) * split);
                auto d_o4 = ctx.alloc<float>(nq);
                auto d_pp = ctx.alloc<float>(nscore);
                d_pp.upload(std::span<const float>(cpu_p));
                auto seed_pt = sentinel_vec(size_t(nq) * split);
                d_pt.upload(std::span<const float>(seed_pt));
                auto seed_o4 = sentinel_vec(nq);
                d_o4.upload(std::span<const float>(seed_o4));

                vulkore::launch(k_amq, {kHeadDim * split}, d_pt, d_pp, d_v, kHeadDim,
                              kSeqLen, 0u, split).wait();
                std::vector<float> hpt(size_t(nq) * split);
                d_pt.download(std::span<float>(hpt));
                int pt_dead = 0;
                for (float v : hpt) if (v == kSentinel) ++pt_dead;
                emit({{}, "apply_mq4 partial live", pt_dead == 0, double(pt_dead),
                      double(pt_dead), "n/a", "intermediate sentinel"});

                vulkore::launch(k_ared, {nq}, d_o4, d_pt, nq, split).wait();
                std::vector<float> gpu_a4(nq);
                d_o4.download(std::span<float>(gpu_a4));
                check_sentinel("attention_apply_mq4", gpu_a4, cpu_a);
                tested("attention_apply_mq4");
                tested("attention_apply_reduce");

                // THE OFFSET PATH. Everything above passes kv_start = 0; the
                // decode loop passes kv_start != 0 on 22 of 26 layers at any
                // depth past 512, so this is the path that actually runs on
                // this device and it needs its own on-device evidence.
                {
                    const uint32_t t0 = kSeqLen / 3;
                    const uint32_t ws = kSeqLen - t0;
                    const uint32_t nw = kHeads * ws;

                    std::vector<float> cpu_sw(nw);
                    for (uint32_t h = 0; h < kHeads; ++h)
                        for (uint32_t j = 0; j < ws; ++j) {
                            float acc = 0.0f;
                            for (uint32_t d = 0; d < kHeadDim; ++d)
                                acc += q[h * kHeadDim + d] *
                                       kc[((t0 + j) * kKvHeads) * kHeadDim + d];
                            cpu_sw[h * ws + j] = acc * scale;
                        }
                    auto d_sw = ctx.alloc<float>(nw);
                    auto seed_sw = sentinel_vec(nw);
                    d_sw.upload(std::span<const float>(seed_sw));
                    vulkore::launch(k_smq, {ws}, d_sw, d_q, d_k, kHeadDim, ws,
                                  scale, t0).wait();
                    std::vector<float> gpu_sw(nw);
                    d_sw.download(std::span<float>(gpu_sw));
                    check_sentinel("attention_scores_mq4 win", gpu_sw, cpu_sw);

                    // NEGATIVE CONTROL: a kernel that ignored kv_start would
                    // still match a reference that ignored it too.
                    double wdiff = 0.0;
                    for (uint32_t j = 0; j < ws; ++j)
                        wdiff = std::max(wdiff, std::fabs(double(gpu_sw[j]) -
                                                          double(cpu_s[j])));
                    emit({{}, "mq4 kv_start not ignored", wdiff > 1e-3, wdiff,
                          wdiff, "n/a", "negative control"});
                }
            }
        }

        // ---- add_residual / add_vec / mul_scalar -------------------------
        {
            auto a = rand_vec(kHidden), b = rand_vec(kHidden);
            auto d_a = ctx.alloc<float>(kHidden);
            auto d_b = ctx.alloc<float>(kHidden);
            auto d_o = ctx.alloc<float>(kHidden);
            d_b.upload(std::span<const float>(b));
            std::vector<float> cpu(kHidden);
            for (uint32_t i = 0; i < kHidden; ++i) cpu[i] = a[i] + b[i];

            d_a.upload(std::span<const float>(a));
            auto kr = prog.kernel("add_residual");
            vulkore::launch(kr, {kHidden}, d_a, d_b, kHidden).wait();
            std::vector<float> gpu(kHidden);
            d_a.download(std::span<float>(gpu));
            check_inplace("add_residual", gpu, cpu, a);
            tested("add_residual");

            auto seed = sentinel_vec(kHidden);
            d_o.upload(std::span<const float>(seed));
            d_a.upload(std::span<const float>(a));
            auto kv = prog.kernel("add_vec");
            vulkore::launch(kv, {kHidden}, d_o, d_a, d_b, kHidden).wait();
            d_o.download(std::span<float>(gpu));
            check_sentinel("add_vec", gpu, cpu);
            tested("add_vec");

            const float s = std::sqrt(float(kHidden));   // Gemma embedding scale
            d_a.upload(std::span<const float>(a));
            auto km = prog.kernel("mul_scalar");
            vulkore::launch(km, {kHidden}, d_a, s, kHidden).wait();
            d_a.download(std::span<float>(gpu));
            for (uint32_t i = 0; i < kHidden; ++i) cpu[i] = a[i] * s;
            check_inplace("mul_scalar", gpu, cpu, a);
            tested("mul_scalar");
        }

        // ---- kv_append: write the middle row, neighbours must survive ----
        {
            const uint32_t row = kKvHeads * kHeadDim, cap = 8, pos = 5;
            std::vector<float> cache(row * cap, 7.0f);
            auto src = rand_vec(row);
            auto d_c = ctx.alloc<float>(cache.size());
            auto d_s = ctx.alloc<float>(row);
            d_c.upload(std::span<const float>(cache));
            d_s.upload(std::span<const float>(src));
            auto k = prog.kernel("kv_append");
            vulkore::launch(k, {row}, d_c, d_s, row, pos).wait();
            std::vector<float> gpu(cache.size());
            d_c.download(std::span<float>(gpu));
            auto cpu = cache;
            for (uint32_t i = 0; i < row; ++i) cpu[pos * row + i] = src[i];
            // In-place detector: only row `pos` should change, so this doubles
            // as an off-by-one check — a shifted write shows up as both a
            // parity failure and unwritten expected elements.
            check_inplace("kv_append", gpu, cpu, cache);
            tested("kv_append");
        }

        // ---- embed_lookup (small stand-in table; the real one is 1.2 GB) --
        {
            const uint32_t vocab = 4096, token = 3001;
            std::vector<float> table(size_t(vocab) * kHidden);
            for (size_t i = 0; i < table.size(); ++i) table[i] = float(i % 977) * 0.01f;
            auto d_t = ctx.alloc<float>(table.size());
            auto d_o = ctx.alloc<float>(kHidden);
            d_t.upload(std::span<const float>(table));
            auto seed = sentinel_vec(kHidden);
            d_o.upload(std::span<const float>(seed));
            auto k = prog.kernel("embed_lookup");
            vulkore::launch(k, {kHidden}, d_o, d_t, token, kHidden).wait();
            std::vector<float> gpu(kHidden), cpu(kHidden);
            d_o.download(std::span<float>(gpu));
            for (uint32_t i = 0; i < kHidden; ++i)
                cpu[i] = table[size_t(token) * kHidden + i];
            check_sentinel("embed_lookup", gpu, cpu);
            tested("embed_lookup");
        }

        // ---- argmax over a realistic vocab, maximum planted off-centre ----
        // Single-thread kernel (grid {1}) scanning 262144 elements. Worth
        // watching on a phone: long single-thread loops are what GPU watchdogs
        // kill, and a killed dispatch is exactly the silent-no-op shape.
        {
            const uint32_t vocab = 262144, planted = 190037;
            auto logits = rand_vec(vocab);
            logits[planted] = 99.5f;
            auto d_l = ctx.alloc<float>(vocab);
            auto d_o = ctx.alloc<float>(2);
            d_l.upload(std::span<const float>(logits));
            std::vector<float> seed(2, kSentinel);
            d_o.upload(std::span<const float>(seed));
            auto k = prog.kernel("argmax");
            vulkore::launch(k, {1}, d_o, d_l, vocab).wait();
            std::vector<float> gpu(2);
            d_o.download(std::span<float>(gpu));
            std::vector<float> cpu = {float(planted), 99.5f};
            check_sentinel("argmax", gpu, cpu);
            tested("argmax");
        }

        // ---- end-to-end block: the LAYOUTS between kernels can disagree ----
        // Each kernel can be individually right while the score row stride, the
        // kv-cache striding or the GQA head mapping disagree between them. One
        // batch, one submit — how the decode loop will actually run it.
        {
            const uint32_t nq = kHeads * kHeadDim;
            const uint32_t ncache = kSeqLen * kKvHeads * kHeadDim;
            const uint32_t nscore = kHeads * kSeqLen;
            const uint32_t pos = kSeqLen - 1;
            const float scale = 1.0f / std::sqrt(float(kHeadDim));

            auto q = rand_vec(nq), qn_w = rand_vec(kHeadDim);
            auto kc = rand_vec(ncache), vc = rand_vec(ncache);

            auto dq = ctx.alloc<float>(nq), dqw = ctx.alloc<float>(kHeadDim);
            auto dqn = ctx.alloc<float>(nq);
            auto dk = ctx.alloc<float>(ncache), dv = ctx.alloc<float>(ncache);
            auto ds = ctx.alloc<float>(nscore), dout = ctx.alloc<float>(nq);
            dq.upload(std::span<const float>(q));
            dqw.upload(std::span<const float>(qn_w));
            dk.upload(std::span<const float>(kc));
            dv.upload(std::span<const float>(vc));
            auto seed = sentinel_vec(nq);
            dout.upload(std::span<const float>(seed));

            vulkore::Batch batch(ctx);
            auto k_qn = prog.kernel("rmsnorm_heads_gemma");
            auto k_rope = prog.kernel("rope");
            auto k_sc = prog.kernel("attention_scores");
            auto k_sm = prog.kernel("softmax_rows");
            auto k_ap = prog.kernel("attention_apply");
            batch.add(k_qn, {nq}, dqn, dq, dqw, kHeads, kHeadDim, kEps);
            batch.add(k_rope, {kHeads * kHeadDim / 2}, dqn, kHeads, kHeadDim, pos, kRopeBase);
            batch.add(k_sc, {nscore}, ds, dqn, dk, kHeads, kKvHeads, kHeadDim, kSeqLen, scale, 0u);
            batch.add(k_sm, {kHeads}, ds, kHeads, kSeqLen);
            batch.add(k_ap, {nq}, dout, ds, dv, kHeads, kKvHeads, kHeadDim, kSeqLen, 0u);
            batch.submit().wait();

            std::vector<float> gpu(nq);
            dout.download(std::span<float>(gpu));

            std::vector<float> qn(nq);
            for (uint32_t h = 0; h < kHeads; ++h) {
                float ss = 0.0f;
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    float v = q[h * kHeadDim + d];
                    ss += v * v;
                }
                float sc = 1.0f / std::sqrt(ss / float(kHeadDim) + kEps);
                for (uint32_t d = 0; d < kHeadDim; ++d)
                    qn[h * kHeadDim + d] = q[h * kHeadDim + d] * sc * (1.0f + qn_w[d]);
            }
            ref_rope(qn, kHeads, kHeadDim, pos, kRopeBase);
            uint32_t group = kHeads / kKvHeads;
            std::vector<float> p(nscore), cpu(nq);
            for (uint32_t h = 0; h < kHeads; ++h) {
                uint32_t kvh = h / group;
                float m = -std::numeric_limits<float>::infinity(), sum = 0.0f;
                for (uint32_t t = 0; t < kSeqLen; ++t) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < kHeadDim; ++d)
                        acc += qn[h * kHeadDim + d] * kc[(t * kKvHeads + kvh) * kHeadDim + d];
                    p[h * kSeqLen + t] = acc * scale;
                    m = std::fmax(m, p[h * kSeqLen + t]);
                }
                for (uint32_t t = 0; t < kSeqLen; ++t) {
                    p[h * kSeqLen + t] = std::exp(p[h * kSeqLen + t] - m);
                    sum += p[h * kSeqLen + t];
                }
                for (uint32_t t = 0; t < kSeqLen; ++t) p[h * kSeqLen + t] /= sum;
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    float acc = 0.0f;
                    for (uint32_t t = 0; t < kSeqLen; ++t)
                        acc += p[h * kSeqLen + t] * vc[(t * kKvHeads + kvh) * kHeadDim + d];
                    cpu[h * kHeadDim + d] = acc;
                }
            }
            check_sentinel("attention block (batch)", gpu, cpu);
        }

        std::printf("\n");
        coverage(prog, "llm_transformer.cl");
        std::printf("\n");
    }

    // =====================================================================
    // MODULE 2 — kernels/llm.cl (5 kernels)
    // =====================================================================
    g_tested.clear();
    {
        auto prog = vulkore::Program::from_file(ctx, spv_llm);
        g_module = "llm.cl";
        std::printf("== %s (%zu kernels) ==\n", g_module.c_str(),
                    prog.kernel_names().size());

        const uint32_t rows = 256, cols = 1152;   // rows % 32 == 0, cols % 8 == 0
        const uint32_t BLK = 32, nblk = rows / BLK;

        auto in = rand_vec(rows);
        auto d_in = ctx.alloc<float>(rows);
        d_in.upload(std::span<const float>(in));

        // ---- matvec_f32 --------------------------------------------------
        {
            std::vector<float> w(size_t(rows) * cols);
            for (size_t i = 0; i < w.size(); ++i) w[i] = u(rng);
            auto d_w = ctx.alloc<float>(w.size());
            auto d_o = ctx.alloc<float>(cols);
            d_w.upload(std::span<const float>(w));
            auto seed = sentinel_vec(cols);
            d_o.upload(std::span<const float>(seed));
            auto k = prog.kernel("matvec_f32");
            vulkore::launch(k, {cols}, d_o, d_in, d_w, rows, cols).wait();
            std::vector<float> gpu(cols), cpu(cols);
            d_o.download(std::span<float>(gpu));
            for (uint32_t j = 0; j < cols; ++j) {
                float acc = 0.0f;
                for (uint32_t k2 = 0; k2 < rows; ++k2)
                    acc += in[k2] * w[size_t(k2) * cols + j];
                cpu[j] = acc;
            }
            check_sentinel("matvec_f32", gpu, cpu);
            tested("matvec_f32");
        }

        // ---- q4 weights, "COL" packing (what llm.cl expects) --------------
        std::vector<uint32_t> wq(size_t(rows) * cols / 8, 0);
        for (uint32_t k = 0; k < rows; ++k)
            for (uint32_t j = 0; j < cols; ++j)
                wq[(size_t(k) * cols + j) >> 3] |= nib(k, j) << ((j & 7u) * 4u);
        std::vector<float> sc(size_t(nblk) * cols);
        for (size_t i = 0; i < sc.size(); ++i) sc[i] = 0.02f + 0.001f * float(i % 7);

        auto d_wq = ctx.alloc<uint32_t>(wq.size());
        auto d_sc = ctx.alloc<float>(sc.size());
        d_wq.upload(std::span<const uint32_t>(wq));
        d_sc.upload(std::span<const float>(sc));

        // CPU reference in the SAME accumulation order as the kernel:
        // per-block partial, then scaled, then accumulated.
        std::vector<float> cpu_q4(cols);
        for (uint32_t j = 0; j < cols; ++j) {
            float acc = 0.0f;
            for (uint32_t b = 0; b < nblk; ++b) {
                float s = sc[size_t(b) * cols + j];
                float part = 0.0f;
                for (uint32_t t = 0; t < BLK; ++t) {
                    uint32_t k = b * BLK + t;
                    part += in[k] * (float(nib(k, j)) - 8.0f);
                }
                acc += part * s;
            }
            cpu_q4[j] = acc;
        }

        // ---- matvec_q4 ---------------------------------------------------
        {
            auto d_o = ctx.alloc<float>(cols);
            auto seed = sentinel_vec(cols);
            d_o.upload(std::span<const float>(seed));
            auto k = prog.kernel("matvec_q4");
            vulkore::launch(k, {cols}, d_o, d_in, d_wq, d_sc, rows, cols).wait();
            std::vector<float> gpu(cols);
            d_o.download(std::span<float>(gpu));
            check_sentinel("matvec_q4", gpu, cpu_q4);
            tested("matvec_q4");
        }

        // ---- matvec_q4_split + matvec_reduce -----------------------------
        // Two dispatches in one batch. The partial buffer is checked on its own
        // so a dead split cannot hide behind a live reduce.
        {
            const uint32_t split = 8;   // nblk == 8, so one block per slice
            auto d_p = ctx.alloc<float>(size_t(split) * cols);
            auto d_o = ctx.alloc<float>(cols);
            auto pseed = sentinel_vec(size_t(split) * cols);
            auto oseed = sentinel_vec(cols);
            d_p.upload(std::span<const float>(pseed));
            d_o.upload(std::span<const float>(oseed));

            auto k1 = prog.kernel("matvec_q4_split");
            auto k2 = prog.kernel("matvec_reduce");
            vulkore::Batch b(ctx);
            b.add(k1, {cols * split}, d_p, d_in, d_wq, d_sc, rows, cols, split);
            b.add(k2, {cols}, d_o, d_p, cols, split);
            b.submit().wait();

            std::vector<float> gp(size_t(split) * cols), cp(size_t(split) * cols);
            d_p.download(std::span<float>(gp));
            uint32_t per = (nblk + split - 1u) / split;
            for (uint32_t p = 0; p < split; ++p) {
                uint32_t b0 = p * per, b1 = std::min(b0 + per, nblk);
                for (uint32_t j = 0; j < cols; ++j) {
                    float acc = 0.0f;
                    for (uint32_t bb = b0; bb < b1; ++bb) {
                        float s = sc[size_t(bb) * cols + j];
                        float part = 0.0f;
                        for (uint32_t t = 0; t < BLK; ++t) {
                            uint32_t k = bb * BLK + t;
                            part += in[k] * (float(nib(k, j)) - 8.0f);
                        }
                        acc += part * s;
                    }
                    cp[size_t(p) * cols + j] = acc;
                }
            }
            check_sentinel("matvec_q4_split", gp, cp);
            tested("matvec_q4_split");

            std::vector<float> gpu(cols);
            d_o.download(std::span<float>(gpu));
            // Summation ORDER differs from the single-pass reference, so this is
            // a different (better-conditioned) answer, not a bit-identical one.
            check_sentinel("matvec_reduce", gpu, cpu_q4);
            tested("matvec_reduce");
        }

        // ---- stream_read (bandwidth probe) --------------------------------
        // Only thread 0 writes, and it writes ONE element — and that element is
        // thread 0's OWN strided partial (k = 0, 65536, 131072, ...), NOT the
        // sum of the whole array. Every other thread's reads are dead code the
        // compiler is not allowed to drop only because the loads are volatile
        // in effect, which is exactly the point of a bandwidth probe. Summing
        // all of src[] here reads as a hard failure against a correct kernel;
        // that is a reference bug, not a driver finding.
        //
        // Kept small enough that the uint32 running sum is exactly
        // representable in fp32, otherwise the cast makes the check meaningless.
        {
            const uint32_t n = 262144;
            const uint32_t stride = 65536;
            std::vector<uint32_t> src(n);
            for (uint32_t i = 0; i < n; ++i) src[i] = i % 17u;
            uint32_t total = 0;
            for (uint32_t k = 0; k < n; k += stride) total += src[k];
            auto d_s = ctx.alloc<uint32_t>(n);
            auto d_o = ctx.alloc<float>(1);
            d_s.upload(std::span<const uint32_t>(src));
            std::vector<float> seed(1, kSentinel);
            d_o.upload(std::span<const float>(seed));
            auto k = prog.kernel("stream_read");
            vulkore::launch(k, {65536}, d_o, d_s, n).wait();
            std::vector<float> gpu(1);
            d_o.download(std::span<float>(gpu));
            std::vector<float> cpu(1, float(total));
            check_sentinel("stream_read", gpu, cpu);
            tested("stream_read");
        }

        std::printf("\n");
        coverage(prog, "llm.cl");
        std::printf("\n");
    }

    // =====================================================================
    // NEGATIVE CONTROLS — is the liveness machinery actually a detector?
    // =====================================================================
    std::printf("== negative controls (detector validation) ==\n");
    bool nc1 = nc_sentinel_flagged == nc_sentinel_expect && nc_sentinel_expect > 0;
    bool nc2 = nc_inplace_flagged == nc_inplace_expect && nc_inplace_expect > 0;
    std::printf("  sentinel detector : rmsnorm_gemma at HALF grid -> flagged %d "
                "untouched, expected %d  %s\n",
                nc_sentinel_flagged, nc_sentinel_expect, nc1 ? "PASS" : "FAIL");
    std::printf("  in-place detector : rope at HALF grid -> flagged %d unwritten, "
                "expected %d  %s\n",
                nc_inplace_flagged, nc_inplace_expect, nc2 ? "PASS" : "FAIL");
    nc1 ? ++g_pass : ++g_fail;
    nc2 ? ++g_pass : ++g_fail;
    if (!nc1 || !nc2)
        std::printf("  ** DETECTOR NOT VALIDATED — every liveness result above is\n"
                    "     unproven. Do not treat this run as evidence. **\n");

    // ---- machine-readable table for agent-docs --------------------------
    std::printf("\n== table ==\n");
    std::printf("| module | kernel | verdict | max_rel | max_abs | liveness |\n");
    for (const auto& r : g_rows)
        std::printf("| %s | %s | %s | %.2e | %.2e | %s |\n", r.module.c_str(),
                    r.name.c_str(), r.pass ? "PASS" : "FAIL", r.max_rel,
                    r.max_abs, r.liveness);

    std::printf("\n%d passed, %d failed  (rtol %.0e, atol %.0e) on %s\n",
                g_pass, g_fail, kRtol, kAtol, ctx.device_name().c_str());
    return g_fail == 0 ? 0 : 1;
}
