// CPU-parity harness for kernels/llm_transformer.cl.
//
// Every kernel is run on the GPU at REAL Gemma 3 1B shapes and compared against
// a straight-line CPU implementation of the same maths. Two things are checked
// per kernel:
//
//   1. PARITY   — allclose against the CPU reference (rtol 1e-4, atol 1e-6).
//   2. LIVENESS — the output buffer is pre-filled with a sentinel and every
//                 element must have been overwritten.
//
// (2) is not paranoia. An 8th-power Mandelbulb in this repo compiled clean,
// passed spirv-val, bound successfully, and then SILENTLY DID NOT EXECUTE on
// Adreno — leaving its output untouched while every API call reported success.
// A clean compile proves nothing; only observed writes do. Since these kernels
// use exp/sin/cos/log2, the liveness check is the point of the harness as much
// as the parity check is.
//
// Build (examples are not wired into CMake):
//   g++ -std=c++20 -O2 -Iinclude -Ithird_party/Vulkan-Headers/include
//       -Ithird_party/volk -Ithird_party/VulkanMemoryAllocator/include
//       -DVMA_STATIC_VULKAN_FUNCTIONS=0 -DVMA_DYNAMIC_VULKAN_FUNCTIONS=1
//       examples/llm/kernel_tests.cpp build/libvulkore.a -ldl -o build/llm_kernel_tests
//   ./build/llm_kernel_tests kernels/llm_transformer.spv

#include <vulkore/vulkore.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

// ---- Gemma 3 1B ------------------------------------------------------------
static constexpr uint32_t kHidden   = 1152;
static constexpr uint32_t kFfn      = 6912;
static constexpr uint32_t kHeads    = 4;
static constexpr uint32_t kKvHeads  = 1;
static constexpr uint32_t kHeadDim  = 256;
static constexpr uint32_t kSeqLen   = 131;      // deliberately not a multiple of 64
static constexpr float    kEps      = 1e-6f;
static constexpr float    kRopeBase = 1000000.0f;   // Gemma 3 global-layer theta

static constexpr float kSentinel = -1.0e30f;
// numpy.allclose semantics: |gpu - cpu| <= atol + rtol*|cpu|.
//
// The atol term is load-bearing, not slack. attention_apply accumulates 131
// products per output and the batched block chains five kernels, so outputs
// that land near zero carry ~1e-6 of absolute fp32 noise (eps is 1.2e-7) while
// being relatively "wrong" by 1e-3. A pure relative bar flags those as failures
// and says nothing about the values that matter. Both numbers are printed so
// the raw magnitudes are visible regardless of the verdict.
static constexpr float kRtol = 1e-4f;
static constexpr float kAtol = 1e-6f;

static int g_pass = 0, g_fail = 0;

// max_rel uses a 1e-3 denominator floor purely so the printed ratio stays
// readable near zero; the PASS/FAIL verdict uses the allclose rule above.
static void report(const char* name, const std::vector<float>& gpu,
                   const std::vector<float>& cpu, bool check_liveness = true) {
    double max_rel = 0.0, max_abs = 0.0;
    size_t worst = 0;
    int untouched = 0, nonfinite = 0, outside = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(gpu[i])) ++nonfinite;
        if (check_liveness && gpu[i] == kSentinel) ++untouched;
        double a = std::fabs(double(gpu[i]) - double(cpu[i]));
        double r = a / std::max(std::fabs(double(cpu[i])), 1e-3);
        if (a > kAtol + double(kRtol) * std::fabs(double(cpu[i]))) {
            if (!outside) worst = i;
            ++outside;
        }
        if (r > max_rel && !outside) worst = i;
        max_rel = std::max(max_rel, r);
        max_abs = std::max(max_abs, a);
    }
    bool ok = outside == 0 && untouched == 0 && nonfinite == 0;
    ok ? ++g_pass : ++g_fail;
    std::printf("  %-24s %s  n=%-8zu max_rel=%.3e  max_abs=%.3e",
                name, ok ? "PASS" : "FAIL", cpu.size(), max_rel, max_abs);
    if (untouched) std::printf("  UNTOUCHED=%d", untouched);
    if (nonfinite) std::printf("  NONFINITE=%d", nonfinite);
    if (!ok) std::printf("  worst[%zu] gpu=%.9g cpu=%.9g", worst, gpu[worst], cpu[worst]);
    std::printf("\n");
}

// ---- CPU references --------------------------------------------------------
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

int main(int argc, char** argv) {
    const char* spv = argc > 1 ? argv[1] : "kernels/llm_transformer.spv";
    vulkore::Context ctx;
    auto prog = vulkore::Program::from_file(ctx, spv);
    std::printf("device: %s\nmodule: %s (%zu kernels)\n\n",
                ctx.device_name().c_str(), spv, prog.kernel_names().size());

    std::mt19937 rng(20260719);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);
    auto rand_vec = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& e : v) e = u(rng);
        return v;
    };
    auto sentinel_vec = [](size_t n) { return std::vector<float>(n, kSentinel); };

    // ======================= RMSNorm =======================================
    {
        auto in = rand_vec(kHidden), w = rand_vec(kHidden);
        auto dev_in = ctx.alloc<float>(kHidden);
        auto dev_w  = ctx.alloc<float>(kHidden);
        auto dev_out = ctx.alloc<float>(kHidden);
        dev_in.upload(std::span<const float>(in));
        dev_w.upload(std::span<const float>(w));

        for (bool gemma : {false, true}) {
            auto seed = sentinel_vec(kHidden);
            dev_out.upload(std::span<const float>(seed));
            auto k = prog.kernel(gemma ? "rmsnorm_gemma" : "rmsnorm");
            vulkore::launch(k, {kHidden}, dev_out, dev_in, dev_w, kHidden, kEps).wait();
            std::vector<float> gpu(kHidden), cpu(kHidden);
            dev_out.download(std::span<float>(gpu));
            ref_rmsnorm(cpu, in, w, kHidden, kEps, gemma);
            report(gemma ? "rmsnorm_gemma" : "rmsnorm", gpu, cpu);
        }
    }

    // ---- two-pass RMSNorm: same answer, different dispatch/work trade -------
    {
        const uint32_t nparts = 64;
        auto in = rand_vec(kHidden), w = rand_vec(kHidden);
        auto dev_in = ctx.alloc<float>(kHidden);
        auto dev_w = ctx.alloc<float>(kHidden);
        auto dev_p = ctx.alloc<float>(nparts);
        auto dev_out = ctx.alloc<float>(kHidden);
        dev_in.upload(std::span<const float>(in));
        dev_w.upload(std::span<const float>(w));
        auto seed = sentinel_vec(kHidden);
        dev_out.upload(std::span<const float>(seed));

        auto k1 = prog.kernel("rmsnorm_sumsq");
        auto k2 = prog.kernel("rmsnorm_apply_gemma");
        vulkore::Batch b(ctx);
        b.add(k1, {nparts}, dev_p, dev_in, kHidden, nparts);
        b.add(k2, {kHidden}, dev_out, dev_in, dev_w, dev_p, kHidden, nparts, kEps);
        b.submit().wait();

        std::vector<float> gpu(kHidden), cpu(kHidden);
        dev_out.download(std::span<float>(gpu));
        ref_rmsnorm(cpu, in, w, kHidden, kEps, /*gemma=*/true);
        // Note: the partial-sum tree changes the summation ORDER versus the
        // serial reference, so this is a slightly different (better-conditioned)
        // answer, not a bit-identical one.
        report("rmsnorm two-pass", gpu, cpu);
    }

    // ---- per-head QK norm --------------------------------------------------
    {
        uint32_t n = kHeads * kHeadDim;
        auto in = rand_vec(n), w = rand_vec(kHeadDim);
        auto dev_in = ctx.alloc<float>(n);
        auto dev_w = ctx.alloc<float>(kHeadDim);
        auto dev_out = ctx.alloc<float>(n);
        dev_in.upload(std::span<const float>(in));
        dev_w.upload(std::span<const float>(w));
        auto seed = sentinel_vec(n);
        dev_out.upload(std::span<const float>(seed));

        auto k = prog.kernel("rmsnorm_heads_gemma");
        vulkore::launch(k, {n}, dev_out, dev_in, dev_w, kHeads, kHeadDim, kEps).wait();

        std::vector<float> gpu(n), cpu(n);
        dev_out.download(std::span<float>(gpu));
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
        report("rmsnorm_heads_gemma", gpu, cpu);
    }

    // ======================= RoPE ==========================================
    // In-place, so liveness is checked differently: the output must DIFFER from
    // the input (a no-op kernel would leave it byte-identical).
    {
        uint32_t n = kHeads * kHeadDim;
        const uint32_t pos = 17;
        auto in = rand_vec(n);
        auto dev = ctx.alloc<float>(n);
        dev.upload(std::span<const float>(in));

        auto k = prog.kernel("rope");
        vulkore::launch(k, {kHeads * kHeadDim / 2}, dev, kHeads, kHeadDim, pos,
                      kRopeBase).wait();
        std::vector<float> gpu(n);
        dev.download(std::span<float>(gpu));

        int changed = 0;
        for (uint32_t i = 0; i < n; ++i) if (gpu[i] != in[i]) ++changed;
        auto cpu = in;
        ref_rope(cpu, kHeads, kHeadDim, pos, kRopeBase);
        report("rope (in-place)", gpu, cpu, /*check_liveness=*/false);
        std::printf("  %-24s %s  %d/%u elements changed by the rotation\n",
                    "rope liveness", changed > int(n) / 2 ? "PASS" : "FAIL",
                    changed, n);
        changed > int(n) / 2 ? ++g_pass : ++g_fail;

        // Host-built angle table must reproduce the on-device trig exactly
        // enough to be interchangeable across the 26 layers.
        uint32_t half = kHeadDim / 2;
        std::vector<float> ct(half), st(half);
        for (uint32_t d = 0; d < half; ++d) {
            float inv = std::exp2(-std::log2(kRopeBase) * float(2 * d) / float(kHeadDim));
            ct[d] = std::cos(float(pos) * inv);
            st[d] = std::sin(float(pos) * inv);
        }
        auto dev2 = ctx.alloc<float>(n);
        auto dev_ct = ctx.alloc<float>(half);
        auto dev_st = ctx.alloc<float>(half);
        dev2.upload(std::span<const float>(in));
        dev_ct.upload(std::span<const float>(ct));
        dev_st.upload(std::span<const float>(st));
        auto kc = prog.kernel("rope_cached");
        vulkore::launch(kc, {kHeads * half}, dev2, dev_ct, dev_st, kHeads, kHeadDim).wait();
        std::vector<float> gpu2(n);
        dev2.download(std::span<float>(gpu2));
        report("rope_cached", gpu2, cpu, false);
    }

    // ======================= GeGLU / SwiGLU ================================
    {
        auto gate = rand_vec(kFfn), up = rand_vec(kFfn);
        auto dev_g = ctx.alloc<float>(kFfn);
        auto dev_u = ctx.alloc<float>(kFfn);
        auto dev_o = ctx.alloc<float>(kFfn);
        dev_g.upload(std::span<const float>(gate));
        dev_u.upload(std::span<const float>(up));

        std::vector<float> cpu_ge(kFfn), cpu_sw(kFfn);
        for (uint32_t i = 0; i < kFfn; ++i) {
            cpu_ge[i] = gelu_tanh(gate[i]) * up[i];
            float g = gate[i];
            cpu_sw[i] = g * (1.0f / (1.0f + std::exp(-g))) * up[i];
        }

        for (const char* name : {"geglu", "swiglu"}) {
            auto seed = sentinel_vec(kFfn);
            dev_o.upload(std::span<const float>(seed));
            auto k = prog.kernel(name);
            vulkore::launch(k, {kFfn}, dev_o, dev_g, dev_u, kFfn).wait();
            std::vector<float> gpu(kFfn);
            dev_o.download(std::span<float>(gpu));
            report(name, gpu, std::string(name) == "geglu" ? cpu_ge : cpu_sw);
        }

        // Fused layout: [gate | up] in one 2n buffer.
        std::vector<float> gu(2 * size_t(kFfn));
        for (uint32_t i = 0; i < kFfn; ++i) { gu[i] = gate[i]; gu[kFfn + i] = up[i]; }
        auto dev_gu = ctx.alloc<float>(gu.size());
        dev_gu.upload(std::span<const float>(gu));
        auto seed = sentinel_vec(kFfn);
        dev_o.upload(std::span<const float>(seed));
        auto k = prog.kernel("geglu_fused");
        vulkore::launch(k, {kFfn}, dev_o, dev_gu, kFfn).wait();
        std::vector<float> gpu(kFfn);
        dev_o.download(std::span<float>(gpu));
        report("geglu_fused", gpu, cpu_ge);
    }

    // ======================= Attention =====================================
    {
        const float scale = 1.0f / std::sqrt(float(kHeadDim));
        uint32_t nq = kHeads * kHeadDim;
        uint32_t ncache = kSeqLen * kKvHeads * kHeadDim;
        uint32_t nscore = kHeads * kSeqLen;

        auto q = rand_vec(nq), kc = rand_vec(ncache), vc = rand_vec(ncache);
        auto dev_q = ctx.alloc<float>(nq);
        auto dev_k = ctx.alloc<float>(ncache);
        auto dev_v = ctx.alloc<float>(ncache);
        auto dev_s = ctx.alloc<float>(nscore);
        auto dev_o = ctx.alloc<float>(nq);
        dev_q.upload(std::span<const float>(q));
        dev_k.upload(std::span<const float>(kc));
        dev_v.upload(std::span<const float>(vc));

        // ---- scores
        auto seed = sentinel_vec(nscore);
        dev_s.upload(std::span<const float>(seed));
        auto ks = prog.kernel("attention_scores");
        vulkore::launch(ks, {nscore}, dev_s, dev_q, dev_k, kHeads, kKvHeads,
                      kHeadDim, kSeqLen, scale, 0u).wait();
        std::vector<float> gpu_s(nscore), cpu_s(nscore);
        dev_s.download(std::span<float>(gpu_s));

        uint32_t group = kHeads / kKvHeads;
        for (uint32_t h = 0; h < kHeads; ++h) {
            uint32_t kvh = h / group;
            for (uint32_t t = 0; t < kSeqLen; ++t) {
                float acc = 0.0f;
                for (uint32_t d = 0; d < kHeadDim; ++d)
                    acc += q[h * kHeadDim + d] * kc[(t * kKvHeads + kvh) * kHeadDim + d];
                cpu_s[h * kSeqLen + t] = acc * scale;
            }
        }
        report("attention_scores", gpu_s, cpu_s);

        // ---- softmax (in place over the GPU's own scores)
        auto ksm = prog.kernel("softmax_rows");
        vulkore::launch(ksm, {kHeads}, dev_s, kHeads, kSeqLen).wait();
        std::vector<float> gpu_p(nscore), cpu_p(nscore);
        dev_s.download(std::span<float>(gpu_p));
        for (uint32_t h = 0; h < kHeads; ++h) {
            float m = -std::numeric_limits<float>::infinity(), sum = 0.0f;
            for (uint32_t t = 0; t < kSeqLen; ++t) m = std::fmax(m, cpu_s[h * kSeqLen + t]);
            for (uint32_t t = 0; t < kSeqLen; ++t) {
                float e = std::exp(cpu_s[h * kSeqLen + t] - m);
                cpu_p[h * kSeqLen + t] = e;
                sum += e;
            }
            for (uint32_t t = 0; t < kSeqLen; ++t) cpu_p[h * kSeqLen + t] /= sum;
        }
        report("softmax_rows", gpu_p, cpu_p, false);
        // A softmax row that does not sum to 1 is wrong even if it matches the
        // reference, so check the invariant directly.
        double worst_row = 0.0;
        for (uint32_t h = 0; h < kHeads; ++h) {
            double s = 0.0;
            for (uint32_t t = 0; t < kSeqLen; ++t) s += gpu_p[h * kSeqLen + t];
            worst_row = std::max(worst_row, std::fabs(s - 1.0));
        }
        bool rows_ok = worst_row < 1e-5;
        rows_ok ? ++g_pass : ++g_fail;
        std::printf("  %-24s %s  max |rowsum-1| = %.3e\n", "softmax rows sum to 1",
                    rows_ok ? "PASS" : "FAIL", worst_row);

        // ---- weighted sum of v
        auto seed_o = sentinel_vec(nq);
        dev_o.upload(std::span<const float>(seed_o));
        auto ka = prog.kernel("attention_apply");
        vulkore::launch(ka, {nq}, dev_o, dev_s, dev_v, kHeads, kKvHeads, kHeadDim,
                      kSeqLen, 0u).wait();
        std::vector<float> gpu_a(nq), cpu_a(nq);
        dev_o.download(std::span<float>(gpu_a));
        for (uint32_t h = 0; h < kHeads; ++h) {
            uint32_t kvh = h / group;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                float acc = 0.0f;
                for (uint32_t t = 0; t < kSeqLen; ++t)
                    acc += cpu_p[h * kSeqLen + t] * vc[(t * kKvHeads + kvh) * kHeadDim + d];
                cpu_a[h * kHeadDim + d] = acc;
            }
        }
        report("attention_apply", gpu_a, cpu_a);

        // ---- PARALLEL ATTENTION: the flatness kernels ---------------------
        //
        // These replace softmax_rows / attention_scores / attention_apply at
        // spans above ~128, which is every layer of a deep decode step. They
        // are checked against the SAME CPU references above, not against the
        // serial kernels, so a shared misunderstanding cannot pass both.
        {
            // softmax_rows_partial -> _finish -> _scale.
            // nparts deliberately does NOT divide kSeqLen (131): the strided
            // chunking must handle a ragged tail, and 8 chunks of 131 is 16.375.
            auto k_smp  = prog.kernel("softmax_rows_partial");
            auto k_smf  = prog.kernel("softmax_rows_finish");
            auto k_sms  = prog.kernel("softmax_rows_scale");
            auto k_smq  = prog.kernel("attention_scores_mq4");
            auto k_amq  = prog.kernel("attention_apply_mq4");
            auto k_ared = prog.kernel("attention_apply_reduce");
            const uint32_t nparts = 8;
            auto dev_sm = ctx.alloc<float>(nscore);
            dev_sm.upload(std::span<const float>(cpu_s));   // the pre-softmax scores
            auto pmax = ctx.alloc<float>(kHeads * nparts);
            auto psum = ctx.alloc<float>(kHeads * nparts);
            auto stats = ctx.alloc<float>(kHeads * 2);
            // Sentinel the INTERMEDIATE buffers too. A dead first pass hiding
            // behind a live second pass is a real failure mode in this repo.
            auto seed_p = sentinel_vec(kHeads * nparts);
            pmax.upload(std::span<const float>(seed_p));
            psum.upload(std::span<const float>(seed_p));

            vulkore::launch(k_smp, {kHeads * nparts},
                          pmax, psum, dev_sm, kHeads, kSeqLen, nparts).wait();
            std::vector<float> hp(kHeads * nparts);
            pmax.download(std::span<float>(hp));
            int pm_untouched = 0;
            for (float v : hp) if (v == kSentinel) ++pm_untouched;
            (pm_untouched == 0) ? ++g_pass : ++g_fail;
            std::printf("  %-24s %s  %d/%zu partials untouched\n",
                        "softmax partial live", pm_untouched == 0 ? "PASS" : "FAIL",
                        pm_untouched, hp.size());

            vulkore::launch(k_smf, {kHeads}, stats,
                          pmax, psum, kHeads, nparts).wait();
            vulkore::launch(k_sms, {nscore}, dev_sm,
                          stats, kHeads, kSeqLen).wait();
            std::vector<float> gpu_sm(nscore);
            dev_sm.download(std::span<float>(gpu_sm));
            report("softmax_rows parallel", gpu_sm, cpu_p, false);

            // attention_scores_mq4 — same reference as the generic kernel.
            auto dev_s4 = ctx.alloc<float>(nscore);
            auto seed_s4 = sentinel_vec(nscore);
            dev_s4.upload(std::span<const float>(seed_s4));
            vulkore::launch(k_smq, {kSeqLen}, dev_s4,
                          dev_q, dev_k, kHeadDim, kSeqLen, scale, 0u).wait();
            std::vector<float> gpu_s4(nscore);
            dev_s4.download(std::span<float>(gpu_s4));
            report("attention_scores_mq4", gpu_s4, cpu_s);

            // attention_apply_mq4 + reduce. split 5 deliberately does not
            // divide 131 either, so the last chunk is short.
            const uint32_t split = 5;
            auto part = ctx.alloc<float>(size_t(nq) * split);
            auto dev_o4 = ctx.alloc<float>(nq);
            auto seed_o4 = sentinel_vec(nq);
            dev_o4.upload(std::span<const float>(seed_o4));
            auto seed_pt = sentinel_vec(size_t(nq) * split);
            part.upload(std::span<const float>(seed_pt));
            auto dev_p = ctx.alloc<float>(nscore);
            dev_p.upload(std::span<const float>(cpu_p));   // exact softmax probs
            vulkore::launch(k_amq, {kHeadDim * split},
                          part, dev_p, dev_v, kHeadDim, kSeqLen, 0u, split).wait();
            std::vector<float> hpart(size_t(nq) * split);
            part.download(std::span<float>(hpart));
            int pt_untouched = 0;
            for (float v : hpart) if (v == kSentinel) ++pt_untouched;
            (pt_untouched == 0) ? ++g_pass : ++g_fail;
            std::printf("  %-24s %s  %d/%zu partials untouched\n",
                        "apply_mq4 partial live", pt_untouched == 0 ? "PASS" : "FAIL",
                        pt_untouched, hpart.size());
            vulkore::launch(k_ared, {nq}, dev_o4,
                          part, nq, split).wait();
            std::vector<float> gpu_a4(nq);
            dev_o4.download(std::span<float>(gpu_a4));
            report("attention_apply_mq4", gpu_a4, cpu_a);

            // ---- THE OFFSET PATH, which is the one that actually runs ----
            // Everything above passes kv_start = 0. In the decode loop 22 of 26
            // layers pass kv_start != 0 at any depth past 512, so a kv_start bug
            // in the new kernels would be invisible to every check so far. This
            // is the same blind spot llm-sliding-window.md documents.
            const uint32_t t0 = kSeqLen / 3;        // deliberately not round
            const uint32_t wspan = kSeqLen - t0;
            const uint32_t nsc_w = kHeads * wspan;

            std::vector<float> cpu_sw(nsc_w);
            for (uint32_t h = 0; h < kHeads; ++h)
                for (uint32_t j = 0; j < wspan; ++j) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < kHeadDim; ++d)
                        acc += q[h * kHeadDim + d] *
                               kc[((t0 + j) * kKvHeads) * kHeadDim + d];
                    cpu_sw[h * wspan + j] = acc * scale;
                }
            auto dev_sw = ctx.alloc<float>(nsc_w);
            auto seed_sw = sentinel_vec(nsc_w);
            dev_sw.upload(std::span<const float>(seed_sw));
            vulkore::launch(k_smq, {wspan}, dev_sw, dev_q, dev_k, kHeadDim, wspan,
                          scale, t0).wait();
            std::vector<float> gpu_sw(nsc_w);
            dev_sw.download(std::span<float>(gpu_sw));
            report("attention_scores_mq4 window", gpu_sw, cpu_sw);

            // NEGATIVE CONTROL: if kv_start were ignored the windowed row would
            // equal the first `wspan` columns of the unwindowed one. Without
            // this, a kernel that drops the offset passes the parity check
            // against a reference that shares the mistake.
            double wdiff = 0.0;
            for (uint32_t j = 0; j < wspan; ++j)
                wdiff = std::max(wdiff, std::fabs(double(gpu_sw[j]) -
                                                  double(cpu_s[j])));
            (wdiff > 1e-3) ? ++g_pass : ++g_fail;
            std::printf("  %-24s %s  max diff vs unwindowed = %.3e\n",
                        "mq4 kv_start not ignored", wdiff > 1e-3 ? "PASS" : "FAIL",
                        wdiff);

            // apply_mq4 with the same offset, against probs over the window.
            std::vector<float> pw(nsc_w);
            for (uint32_t h = 0; h < kHeads; ++h) {
                float m = -std::numeric_limits<float>::infinity(), sum = 0.0f;
                for (uint32_t j = 0; j < wspan; ++j) m = std::fmax(m, cpu_sw[h * wspan + j]);
                for (uint32_t j = 0; j < wspan; ++j) {
                    float e = std::exp(cpu_sw[h * wspan + j] - m);
                    pw[h * wspan + j] = e; sum += e;
                }
                for (uint32_t j = 0; j < wspan; ++j) pw[h * wspan + j] /= sum;
            }
            std::vector<float> cpu_aw(nq);
            for (uint32_t h = 0; h < kHeads; ++h)
                for (uint32_t d = 0; d < kHeadDim; ++d) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < wspan; ++j)
                        acc += pw[h * wspan + j] *
                               vc[((t0 + j) * kKvHeads) * kHeadDim + d];
                    cpu_aw[h * kHeadDim + d] = acc;
                }
            auto dev_pw = ctx.alloc<float>(nsc_w);
            dev_pw.upload(std::span<const float>(pw));
            auto part_w = ctx.alloc<float>(size_t(nq) * split);
            auto dev_ow = ctx.alloc<float>(nq);
            auto seed_ow = sentinel_vec(nq);
            dev_ow.upload(std::span<const float>(seed_ow));
            vulkore::launch(k_amq, {kHeadDim * split}, part_w, dev_pw, dev_v,
                          kHeadDim, wspan, t0, split).wait();
            vulkore::launch(k_ared, {nq}, dev_ow, part_w, nq, split).wait();
            std::vector<float> gpu_aw(nq);
            dev_ow.download(std::span<float>(gpu_aw));
            report("attention_apply_mq4 window", gpu_aw, cpu_aw);
        }

        // ---- SLIDING WINDOW: kv_start != 0
        //
        // The whole point of the offset form is that a windowed layer reads a
        // contiguous SUFFIX of the cache. Passing kv_start = 0 (everything
        // above) can never catch an indexing mistake in that offset, so this
        // drives a real window and checks it against a CPU reference written
        // in terms of ABSOLUTE cache positions [t0, kSeqLen).
        {
            const uint32_t t0 = kSeqLen / 3;      // deliberately not a nice number
            const uint32_t span = kSeqLen - t0;
            const uint32_t nsc_w = kHeads * span;

            auto seed_w = sentinel_vec(nsc_w);
            dev_s.upload(std::span<const float>(seed_w));
            vulkore::launch(ks, {nsc_w}, dev_s, dev_q, dev_k, kHeads, kKvHeads, kHeadDim,
                          span, scale, t0).wait();
            std::vector<float> gpu_w(nsc_w), cpu_w(nsc_w);
            dev_s.download(std::span<float>(gpu_w));
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
            report("attention_scores window", gpu_w, cpu_w);

            // NEGATIVE CONTROL: the windowed scores must NOT equal the first
            // `span` columns of the unwindowed row. If they did, kv_start
            // would be being ignored — which is exactly the bug this whole
            // change exists to fix, and it would otherwise pass silently.
            bool differs = false;
            for (uint32_t h = 0; h < kHeads && !differs; ++h)
                for (uint32_t j = 0; j < span; ++j)
                    if (gpu_w[h * span + j] != cpu_s[h * kSeqLen + j]) { differs = true; break; }
            std::printf("  %-28s %s\n", "attention_scores kv_start live",
                        differs ? "yes" : "*** NO — kv_start IGNORED ***");
            if (!differs) ++g_fail; else ++g_pass;

            vulkore::launch(ksm, {kHeads}, dev_s, kHeads, span).wait();
            std::vector<float> pw(nsc_w);
            dev_s.download(std::span<float>(pw));

            auto seed_ow = sentinel_vec(nq);
            dev_o.upload(std::span<const float>(seed_ow));
            vulkore::launch(ka, {nq}, dev_o, dev_s, dev_v, kHeads, kKvHeads, kHeadDim,
                          span, t0).wait();
            std::vector<float> gpu_aw(nq), cpu_aw(nq);
            dev_o.download(std::span<float>(gpu_aw));
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
            report("attention_apply window", gpu_aw, cpu_aw);
        }
    }

    // ======================= Plumbing ======================================
    {
        auto a = rand_vec(kHidden), b = rand_vec(kHidden);
        auto dev_a = ctx.alloc<float>(kHidden);
        auto dev_b = ctx.alloc<float>(kHidden);
        auto dev_o = ctx.alloc<float>(kHidden);
        dev_b.upload(std::span<const float>(b));

        dev_a.upload(std::span<const float>(a));
        auto kr = prog.kernel("add_residual");
        vulkore::launch(kr, {kHidden}, dev_a, dev_b, kHidden).wait();
        std::vector<float> gpu(kHidden), cpu(kHidden);
        dev_a.download(std::span<float>(gpu));
        for (uint32_t i = 0; i < kHidden; ++i) cpu[i] = a[i] + b[i];
        report("add_residual", gpu, cpu, false);

        auto seed = sentinel_vec(kHidden);
        dev_o.upload(std::span<const float>(seed));
        dev_a.upload(std::span<const float>(a));
        auto kv = prog.kernel("add_vec");
        vulkore::launch(kv, {kHidden}, dev_o, dev_a, dev_b, kHidden).wait();
        dev_o.download(std::span<float>(gpu));
        report("add_vec", gpu, cpu);

        // Gemma's embedding scale: sqrt(hidden).
        const float s = std::sqrt(float(kHidden));
        dev_a.upload(std::span<const float>(a));
        auto km = prog.kernel("mul_scalar");
        vulkore::launch(km, {kHidden}, dev_a, s, kHidden).wait();
        dev_a.download(std::span<float>(gpu));
        for (uint32_t i = 0; i < kHidden; ++i) cpu[i] = a[i] * s;
        report("mul_scalar", gpu, cpu, false);
    }

    // ---- kv_append: write into the middle of a cache, check the neighbours
    // are untouched (an off-by-one in the offset would corrupt history).
    {
        const uint32_t row = kKvHeads * kHeadDim, cap = 8, pos = 5;
        std::vector<float> cache(row * cap, 7.0f);
        auto src = rand_vec(row);
        auto dev_c = ctx.alloc<float>(cache.size());
        auto dev_s = ctx.alloc<float>(row);
        dev_c.upload(std::span<const float>(cache));
        dev_s.upload(std::span<const float>(src));
        auto k = prog.kernel("kv_append");
        vulkore::launch(k, {row}, dev_c, dev_s, row, pos).wait();
        std::vector<float> gpu(cache.size());
        dev_c.download(std::span<float>(gpu));
        auto cpu = cache;
        for (uint32_t i = 0; i < row; ++i) cpu[pos * row + i] = src[i];
        report("kv_append", gpu, cpu, false);
    }

    // ---- embed_lookup on a small stand-in table (the real one is 262144x1152
    // fp32 = 1.2 GB, above maxStorageBufferRange on the target devices).
    {
        const uint32_t vocab = 4096, token = 3001;
        std::vector<float> table(size_t(vocab) * kHidden);
        for (size_t i = 0; i < table.size(); ++i) table[i] = float(i % 977) * 0.01f;
        auto dev_t = ctx.alloc<float>(table.size());
        auto dev_o = ctx.alloc<float>(kHidden);
        dev_t.upload(std::span<const float>(table));
        auto seed = sentinel_vec(kHidden);
        dev_o.upload(std::span<const float>(seed));
        auto k = prog.kernel("embed_lookup");
        vulkore::launch(k, {kHidden}, dev_o, dev_t, token, kHidden).wait();
        std::vector<float> gpu(kHidden), cpu(kHidden);
        dev_o.download(std::span<float>(gpu));
        for (uint32_t i = 0; i < kHidden; ++i) cpu[i] = table[size_t(token) * kHidden + i];
        report("embed_lookup", gpu, cpu);
    }

    // ---- argmax over a realistic vocab, with the maximum planted off-center
    {
        const uint32_t vocab = 262144, planted = 190037;
        auto logits = rand_vec(vocab);
        logits[planted] = 99.5f;
        auto dev_l = ctx.alloc<float>(vocab);
        auto dev_o = ctx.alloc<float>(2);
        dev_l.upload(std::span<const float>(logits));
        std::vector<float> seed(2, kSentinel);
        dev_o.upload(std::span<const float>(seed));
        auto k = prog.kernel("argmax");
        vulkore::launch(k, {1}, dev_o, dev_l, vocab).wait();
        std::vector<float> gpu(2);
        dev_o.download(std::span<float>(gpu));
        std::vector<float> cpu = {float(planted), 99.5f};
        report("argmax", gpu, cpu);
    }

    // ======================= End-to-end attention block ====================
    // The individual kernels can each be right while the LAYOUTS between them
    // disagree — score row stride, kv-cache striding, the GQA head mapping. So
    // chain norm -> rope -> scores -> softmax -> apply in ONE batch and compare
    // the final output against a CPU model of the whole block.
    {
        const uint32_t nq = kHeads * kHeadDim;
        const uint32_t ncache = kSeqLen * kKvHeads * kHeadDim;
        const uint32_t nscore = kHeads * kSeqLen;
        const uint32_t pos = kSeqLen - 1;
        const float scale = 1.0f / std::sqrt(float(kHeadDim));

        auto q = rand_vec(nq), qn_w = rand_vec(kHeadDim);
        auto kc = rand_vec(ncache), vc = rand_vec(ncache);

        auto dq  = ctx.alloc<float>(nq);
        auto dqw = ctx.alloc<float>(kHeadDim);
        auto dqn = ctx.alloc<float>(nq);
        auto dk  = ctx.alloc<float>(ncache);
        auto dv  = ctx.alloc<float>(ncache);
        auto ds  = ctx.alloc<float>(nscore);
        auto dout = ctx.alloc<float>(nq);
        dq.upload(std::span<const float>(q));
        dqw.upload(std::span<const float>(qn_w));
        dk.upload(std::span<const float>(kc));
        dv.upload(std::span<const float>(vc));
        auto seed = sentinel_vec(nq);
        dout.upload(std::span<const float>(seed));

        // ONE submit for the whole block — this is how the decode loop will run
        // it, and it also exercises the inter-dispatch barriers.
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

        // CPU model of the same block.
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
        report("attention block (batched)", gpu, cpu);
    }

    // ======================= Cost of the reduction trade ===================
    // Correctness does not settle which RMSNorm to use. The one-pass kernel has
    // every thread redo the whole sum of squares (n^2 work, ONE dispatch); the
    // two-pass pair does n + n*nparts work across TWO dispatches. Measure it
    // instead of arguing about it — everything is timed inside a Batch so the
    // comparison is GPU work, not submit overhead.
    {
        const uint32_t nparts = 64;
        const int R = 200;
        auto in = rand_vec(kHidden), w = rand_vec(kHidden);
        auto dev_in = ctx.alloc<float>(kHidden);
        auto dev_w = ctx.alloc<float>(kHidden);
        auto dev_p = ctx.alloc<float>(nparts);
        auto dev_o = ctx.alloc<float>(kHidden);
        dev_in.upload(std::span<const float>(in));
        dev_w.upload(std::span<const float>(w));

        auto k1 = prog.kernel("rmsnorm_gemma");
        auto k2 = prog.kernel("rmsnorm_sumsq");
        auto k3 = prog.kernel("rmsnorm_apply_gemma");
        auto k4 = prog.kernel("add_residual");

        auto time_it = [&](const char* name, auto record) {
            for (int warm = 0; warm < 20; ++warm) {
                vulkore::Batch b(ctx); record(b); b.submit().wait();
            }
            auto t0 = std::chrono::steady_clock::now();
            vulkore::Batch b(ctx);
            for (int r = 0; r < R; ++r) record(b);
            b.submit().wait();
            double per = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count() / R;
            std::printf("  %-30s %.4f ms per norm\n", name, per);
            return per;
        };

        std::printf("\ntiming (hidden=%u, batched, %d reps):\n", kHidden, R);
        double one = time_it("rmsnorm one-pass (n^2, 1 disp)", [&](vulkore::Batch& b) {
            b.add(k1, {kHidden}, dev_o, dev_in, dev_w, kHidden, kEps);
        });
        double two = time_it("rmsnorm two-pass (2 disp)", [&](vulkore::Batch& b) {
            b.add(k2, {nparts}, dev_p, dev_in, kHidden, nparts);
            b.add(k3, {kHidden}, dev_o, dev_in, dev_w, dev_p, kHidden, nparts, kEps);
        });
        double triv = time_it("add_residual (dispatch floor)", [&](vulkore::Batch& b) {
            b.add(k4, {kHidden}, dev_o, dev_in, kHidden);
        });
        std::printf("  -> two-pass is %.2fx the one-pass cost; the floor for ANY\n"
                    "     dispatch here is %.4f ms, and Gemma 3 1B does 52 norms\n"
                    "     per token (2 x 26 layers): %.2f ms vs %.2f ms per token.\n",
                    two / one, triv, one * 52, two * 52);
    }

    std::printf("\n%d passed, %d failed  (rtol %.0e)\n", g_pass, g_fail, kRtol);
    return g_fail == 0 ? 0 : 1;
}
