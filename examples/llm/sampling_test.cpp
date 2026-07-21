// CPU-parity + timing harness for kernels/llm_sampling.cl.
//
// Two things are being established here:
//
//   1. CORRECTNESS of the sampling and prefill kernels against straight-line
//      CPU references, at REAL Gemma 3 1B shapes (vocab 262144).
//   2. THE NUMBER THIS WORK EXISTS FOR: how much faster the two-pass parallel
//      argmax is than llm_transformer.cl's one-thread argmax over the vocab.
//      That kernel sits on the critical path of every generated token.
//
// Every output buffer is pre-filled with a sentinel and any element left
// untouched fails the check. That is not paranoia: an 8th-power Mandelbulb in
// this repo compiled clean, passed spirv-val, bound successfully and then
// SILENTLY DID NOT EXECUTE on Adreno while every API call reported success. A
// clean compile proves nothing; only observed writes do. Timing an unexecuted
// kernel would produce a spectacular and completely fake speedup, so the
// argmax parity check gates the argmax timing deliberately.
//
// Build (examples are not wired into CMake):
//   g++ -std=c++20 -O2 -Iinclude -Ithird_party/Vulkan-Headers/include \
//       -Ithird_party/volk -Ithird_party/VulkanMemoryAllocator/include \
//       -DVMA_STATIC_VULKAN_FUNCTIONS=0 -DVMA_DYNAMIC_VULKAN_FUNCTIONS=1 \
//       examples/llm/sampling_test.cpp build/libvulkore.a -ldl -o build/llm_sampling_test
//   ./build/llm_sampling_test kernels/llm_sampling.spv kernels/llm_transformer.spv

#include <vulkore/vulkore.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <numeric>
#include <random>
#include <span>
#include <vector>

// ---- Gemma 3 1B ------------------------------------------------------------
static constexpr uint32_t kVocab   = 262144;
static constexpr uint32_t kHidden  = 1152;
static constexpr uint32_t kHeads   = 4;
static constexpr uint32_t kKvHeads = 1;
static constexpr uint32_t kHeadDim = 256;
static constexpr float    kEps     = 1e-6f;
static constexpr float    kRopeBase = 1000000.0f;

// Reduction width for the two-pass kernels. 4096 threads over a 262144 vocab is
// 64 elements per thread — enough work per thread to amortise launch overhead,
// while leaving the serial finalizer only 4096 steps (64x less than the vocab).
static constexpr uint32_t kParts = 4096;

static constexpr float kSentinel = -1.0e30f;
static constexpr float kRtol = 1e-4f;
static constexpr float kAtol = 1e-6f;

static int g_pass = 0, g_fail = 0;

static void ok(const char* name, bool good, const char* detail = "") {
    good ? ++g_pass : ++g_fail;
    std::printf("  %-34s %s  %s\n", name, good ? "PASS" : "FAIL", detail);
}

// numpy-allclose semantics: |gpu-cpu| <= atol + rtol*|cpu|.
static void report(const char* name, const std::vector<float>& gpu,
                   const std::vector<float>& cpu, bool check_liveness = true) {
    double max_rel = 0.0, max_abs = 0.0;
    size_t worst = 0;
    int untouched = 0, nonfinite = 0, outside = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(gpu[i]) && std::isfinite(cpu[i])) ++nonfinite;
        if (check_liveness && gpu[i] == kSentinel) ++untouched;
        double a = std::fabs(double(gpu[i]) - double(cpu[i]));
        if (!std::isfinite(cpu[i])) {                 // masked -INFINITY entries
            if (gpu[i] != cpu[i]) { if (!outside) worst = i; ++outside; }
            continue;
        }
        double r = a / std::max(std::fabs(double(cpu[i])), 1e-3);
        if (a > kAtol + double(kRtol) * std::fabs(double(cpu[i]))) {
            if (!outside) worst = i;
            ++outside;
        }
        max_rel = std::max(max_rel, r);
        max_abs = std::max(max_abs, a);
    }
    bool good = outside == 0 && untouched == 0 && nonfinite == 0;
    good ? ++g_pass : ++g_fail;
    std::printf("  %-34s %s  n=%-8zu max_rel=%.3e  max_abs=%.3e",
                name, good ? "PASS" : "FAIL", cpu.size(), max_rel, max_abs);
    if (untouched) std::printf("  UNTOUCHED=%d", untouched);
    if (nonfinite) std::printf("  NONFINITE=%d", nonfinite);
    if (!good) std::printf("  worst[%zu] gpu=%.9g cpu=%.9g", worst, gpu[worst], cpu[worst]);
    std::printf("\n");
}

int main(int argc, char** argv) {
    const char* spv_s = argc > 1 ? argv[1] : "kernels/llm_sampling.spv";
    const char* spv_t = argc > 2 ? argv[2] : "kernels/llm_transformer.spv";

    vulkore::Context ctx;
    auto prog = vulkore::Program::from_file(ctx, spv_s);
    auto tprog = vulkore::Program::from_file(ctx, spv_t);
    std::printf("device: %s\nsampling: %s (%zu kernels)\ntransformer: %s (%zu kernels)\n\n",
                ctx.device_name().c_str(), spv_s, prog.kernel_names().size(),
                spv_t, tprog.kernel_names().size());

    std::mt19937 rng(20260719);
    std::uniform_real_distribution<float> u(-1.5f, 1.5f);
    auto rand_vec = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& e : v) e = u(rng);
        return v;
    };
    auto sentinel_vec = [](size_t n) { return std::vector<float>(n, kSentinel); };

    // vulkore::launch and Batch::add take Kernel& — a temporary from
    // Program::kernel() will not bind. Cache them here rather than at namespace
    // scope: a static Kernel would outlive the Context and use-after-free its
    // VkDevice during exit handlers, which is exactly the bug recorded in
    // agent-docs/exit-teardown-fix.md.
    std::map<std::string, vulkore::Kernel> kcache;
    auto kern = [&](vulkore::Program& p, const std::string& name) -> vulkore::Kernel& {
        std::string key = std::to_string(reinterpret_cast<uintptr_t>(&p)) + "/" + name;
        auto it = kcache.find(key);
        if (it == kcache.end()) it = kcache.emplace(key, p.kernel(name)).first;
        return it->second;
    };

    // ==================================================================
    // Parallel argmax over the full vocabulary
    // ==================================================================
    double t_serial = 0.0, t_par = 0.0;
    {
        const uint32_t planted = 190037;
        auto logits = rand_vec(kVocab);
        logits[planted] = 99.5f;

        auto d_log  = ctx.alloc<float>(kVocab);
        auto d_pval = ctx.alloc<float>(kParts);
        auto d_pidx = ctx.alloc<uint32_t>(kParts);
        auto d_out  = ctx.alloc<float>(2);
        auto d_oidx = ctx.alloc<uint32_t>(1);
        d_log.upload(std::span<const float>(logits));

        auto& k1 = kern(prog, "argmax_partial");
        auto& k2 = kern(prog, "argmax_final");

        auto run_parallel = [&] {
            std::vector<float> seed(2, kSentinel);
            d_out.upload(std::span<const float>(seed));
            vulkore::Batch b(ctx);
            b.add(k1, {kParts}, d_pval, d_pidx, d_log, kVocab, kParts);
            b.add(k2, {1}, d_out, d_oidx, d_pval, d_pidx, kParts);
            b.submit().wait();
            std::vector<float> gpu(2);
            d_out.download(std::span<float>(gpu));
            return gpu;
        };

        auto gpu = run_parallel();
        report("argmax parallel (planted max)", gpu, {float(planted), 99.5f});

        std::vector<uint32_t> gidx(1);
        d_oidx.download(std::span<uint32_t>(gidx));
        ok("argmax uint index buffer", gidx[0] == planted);

        // ---- tie-break: duplicate maxima must resolve to the LOWEST index,
        // the same rule a straight-line CPU loop uses. Chains are strided, so
        // without the (value,index) comparison in argmax_final this fails.
        {
            auto tied = logits;
            tied[planted] = 42.0f;
            tied[7]       = 42.0f;
            tied[kVocab - 3] = 42.0f;
            d_log.upload(std::span<const float>(tied));
            auto g = run_parallel();
            ok("argmax tie -> lowest index", g[0] == 7.0f && g[1] == 42.0f);
            d_log.upload(std::span<const float>(logits));
        }

        // ---- the headline measurement -----------------------------------
        // Both timed inside an vulkore::Batch so this is GPU work, not submit
        // overhead (0.31 ms per SUBMIT is the dominant cost otherwise).
        auto& ks = kern(tprog, "argmax");           // one thread, whole vocab
        auto d_sout = ctx.alloc<float>(2);

        // Correctness gate on the serial kernel too — timing a kernel that did
        // not run is the classic way to fake a speedup on this hardware.
        {
            std::vector<float> seed(2, kSentinel);
            d_sout.upload(std::span<const float>(seed));
            vulkore::launch(ks, {1}, d_sout, d_log, kVocab).wait();
            std::vector<float> g(2);
            d_sout.download(std::span<float>(g));
            report("argmax serial (reference kernel)", g, {float(planted), 99.5f});
        }

        auto time_it = [&](const char* name, int reps, auto record) {
            for (int w = 0; w < 5; ++w) { vulkore::Batch b(ctx); record(b); b.submit().wait(); }
            double best = 1e30;
            for (int trial = 0; trial < 3; ++trial) {
                auto t0 = std::chrono::steady_clock::now();
                vulkore::Batch b(ctx);
                for (int r = 0; r < reps; ++r) record(b);
                b.submit().wait();
                double per = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0).count() / reps;
                best = std::min(best, per);
            }
            std::printf("  %-40s %8.4f ms\n", name, best);
            return best;
        };

        std::printf("\nargmax timing at vocab=%u (batched, min of 3 trials):\n", kVocab);
        t_serial = time_it("serial argmax (1 thread)", 20, [&](vulkore::Batch& b) {
            b.add(ks, {1}, d_sout, d_log, kVocab);
        });
        t_par = time_it("parallel argmax (2-pass, 4096 parts)", 200, [&](vulkore::Batch& b) {
            b.add(k1, {kParts}, d_pval, d_pidx, d_log, kVocab, kParts);
            b.add(k2, {1}, d_out, d_oidx, d_pval, d_pidx, kParts);
        });
        std::printf("  -> parallel argmax is %.1fx faster (%.4f ms saved per token)\n",
                    t_serial / t_par, t_serial - t_par);

        // ---- how much of what is left is the SERIAL finalizer? nparts trades
        // pass-1 parallelism against pass-2 serial length, and the optimum is
        // not obvious a priori: too few parts and pass 1 is starved, too many
        // and argmax_final becomes the new single-threaded walk.
        std::printf("\n  nparts sweep (pass1 + pass2, and pass1 alone):\n");
        for (uint32_t np : {256u, 1024u, 4096u, 16384u, 65536u}) {
            auto d_pv = ctx.alloc<float>(np);
            auto d_pi = ctx.alloc<uint32_t>(np);
            double both = 0.0, only1 = 0.0;
            for (int w = 0; w < 5; ++w) {
                vulkore::Batch b(ctx);
                b.add(k1, {np}, d_pv, d_pi, d_log, kVocab, np);
                b.add(k2, {1}, d_out, d_oidx, d_pv, d_pi, np);
                b.submit().wait();
            }
            for (int trial = 0; trial < 3; ++trial) {
                auto t0 = std::chrono::steady_clock::now();
                { vulkore::Batch b(ctx);
                  for (int r = 0; r < 100; ++r) {
                      b.add(k1, {np}, d_pv, d_pi, d_log, kVocab, np);
                      b.add(k2, {1}, d_out, d_oidx, d_pv, d_pi, np);
                  }
                  b.submit().wait(); }
                double a = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0).count() / 100;
                auto t1 = std::chrono::steady_clock::now();
                { vulkore::Batch b(ctx);
                  for (int r = 0; r < 100; ++r)
                      b.add(k1, {np}, d_pv, d_pi, d_log, kVocab, np);
                  b.submit().wait(); }
                double c = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t1).count() / 100;
                if (trial == 0 || a < both) both = a;
                if (trial == 0 || c < only1) only1 = c;
            }
            std::printf("    nparts=%-6u  total %7.4f ms   pass1 %7.4f ms   "
                        "finalizer %7.4f ms   (%.1fx vs serial)\n",
                        np, both, only1, both - only1, t_serial / both);
            if (both < t_par) t_par = both;
        }
        std::printf("  -> best TWO-level parallel argmax %.4f ms = %.1fx\n", t_par,
                    t_serial / t_par);

        // ---- three levels: 262144 -> 4096 -> 64 -> 1. The middle level turns
        // the 4096-long serial finalizer into a 64-long one, at the cost of one
        // extra dispatch (~0.006 ms inside a Batch).
        {
            const uint32_t L1 = 4096, L2 = 64;
            auto d_v1 = ctx.alloc<float>(L1);
            auto d_i1 = ctx.alloc<uint32_t>(L1);
            auto d_v2 = ctx.alloc<float>(L2);
            auto d_i2 = ctx.alloc<uint32_t>(L2);
            auto& kred = kern(prog, "argmax_reduce");
            auto record3 = [&](vulkore::Batch& b) {
                b.add(k1,   {L1}, d_v1, d_i1, d_log, kVocab, L1);
                b.add(kred, {L2}, d_v2, d_i2, d_v1, d_i1, L1, L2);
                b.add(k2,   {1},  d_out, d_oidx, d_v2, d_i2, L2);
            };
            // Correctness first — a three-level tree that loses the tie-break
            // rule would still produce a plausible token.
            std::vector<float> seed(2, kSentinel);
            d_out.upload(std::span<const float>(seed));
            { vulkore::Batch b(ctx); record3(b); b.submit().wait(); }
            std::vector<float> g3(2);
            d_out.download(std::span<float>(g3));
            report("argmax 3-level (4096->64->1)", g3, {float(planted), 99.5f});

            double t3 = time_it("parallel argmax (3-level tree)", 500, record3);
            std::printf("  -> 3-level argmax is %.1fx faster than serial, and "
                        "%.2fx faster than the best 2-level\n\n",
                        t_serial / t3, t_par / t3);
            t_par = std::min(t_par, t3);
        }
    }

    // ==================================================================
    // Full-vocabulary softmax
    // ==================================================================
    {
        auto logits = rand_vec(kVocab);

        auto d_x     = ctx.alloc<float>(kVocab);
        auto d_out   = ctx.alloc<float>(kVocab);
        auto d_pmax  = ctx.alloc<float>(kParts);
        auto d_psum  = ctx.alloc<float>(kParts);
        auto d_stats = ctx.alloc<float>(2);

        auto& km = kern(prog, "reduce_max_partial");
        auto& kmf = kern(prog, "reduce_max_final");
        auto& ke = kern(prog, "softmax_exp_partial");
        auto& ksf = kern(prog, "reduce_sum_final");
        auto& kn = kern(prog, "softmax_normalize");

        // Three-level reduction on BOTH the max and the sum, for the same
        // reason as the argmax tree: a serial finalizer over kParts partials
        // costs ~0.13 us/element and would dominate the whole softmax.
        // reduce_max_partial composes with itself (it reads floats, writes
        // strided chain maxima); the sum needs reduce_sum_partial.
        const uint32_t L2 = 64;
        auto d_pmax2 = ctx.alloc<float>(L2);
        auto d_psum2 = ctx.alloc<float>(L2);
        auto& ksp = kern(prog, "reduce_sum_partial");

        auto record = [&](vulkore::Batch& b) {
            b.add(km,  {kParts}, d_pmax, d_x, kVocab, kParts);
            b.add(km,  {L2},     d_pmax2, d_pmax, kParts, L2);
            b.add(kmf, {1},      d_stats, d_pmax2, L2);
            b.add(ke,  {kParts}, d_out, d_psum, d_x, d_stats, kVocab, kParts);
            b.add(ksp, {L2},     d_psum2, d_psum, kParts, L2);
            b.add(ksf, {1},      d_stats, d_psum2, L2);
            b.add(kn,  {kVocab}, d_out, d_stats, kVocab);
        };
        auto softmax = [&](const std::vector<float>& in) {
            d_x.upload(std::span<const float>(in));
            auto seed = sentinel_vec(kVocab);
            d_out.upload(std::span<const float>(seed));
            vulkore::Batch b(ctx);
            record(b);
            b.submit().wait();
            std::vector<float> g(kVocab);
            d_out.download(std::span<float>(g));
            return g;
        };
        auto cpu_softmax = [](const std::vector<float>& in) {
            double m = -std::numeric_limits<double>::infinity();
            for (float v : in) m = std::max(m, double(v));
            double s = 0.0;
            std::vector<double> e(in.size());
            for (size_t i = 0; i < in.size(); ++i) { e[i] = std::exp(double(in[i]) - m); s += e[i]; }
            std::vector<float> out(in.size());
            for (size_t i = 0; i < in.size(); ++i) out[i] = float(e[i] / s);
            return out;
        };

        auto gpu = softmax(logits);
        report("softmax vocab", gpu, cpu_softmax(logits));

        double sum = std::accumulate(gpu.begin(), gpu.end(), 0.0);
        char buf[96];
        std::snprintf(buf, sizeof buf, "sum=%.9f  |sum-1|=%.3e", sum, std::fabs(sum - 1.0));
        ok("softmax vocab sums to 1", std::fabs(sum - 1.0) < 1e-4, buf);

        // ---- numerical stability. A softmax without max subtraction overflows
        // to inf/inf = NaN here; the whole point of the two reduce_max passes.
        {
            auto big = logits;
            for (auto& v : big) v += 1000.0f;
            auto g = softmax(big);
            auto c = cpu_softmax(logits);       // shift-invariant: same answer
            report("softmax stability (+1000)", g, c);
            bool finite = std::all_of(g.begin(), g.end(),
                                      [](float v) { return std::isfinite(v) && v >= 0.0f; });
            ok("softmax +1000 all finite", finite);
        }
        // ---- and the other direction, where every exp underflows to zero
        // except near the max.
        {
            auto sharp = logits;
            for (auto& v : sharp) v *= 200.0f;
            auto g = softmax(sharp);
            double s2 = std::accumulate(g.begin(), g.end(), 0.0);
            std::snprintf(buf, sizeof buf, "sum=%.9f (peaked distribution)", s2);
            ok("softmax sharp (x200) sums to 1", std::fabs(s2 - 1.0) < 1e-4, buf);
        }

        // Cost of the whole 7-dispatch chain, for the token budget.
        {
            for (int w = 0; w < 5; ++w) { vulkore::Batch b(ctx); record(b); b.submit().wait(); }
            double best = 1e30;
            for (int trial = 0; trial < 3; ++trial) {
                auto t0 = std::chrono::steady_clock::now();
                vulkore::Batch b(ctx);
                for (int r = 0; r < 200; ++r) record(b);
                b.submit().wait();
                best = std::min(best, std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - t0).count() / 200);
            }
            std::printf("\n  full-vocab softmax (7 dispatches, 3-level)   %8.4f ms\n\n", best);
        }
    }

    // ==================================================================
    // Temperature
    // ==================================================================
    {
        const float T = 0.7f;
        auto logits = rand_vec(kVocab);
        auto d_x = ctx.alloc<float>(kVocab);
        d_x.upload(std::span<const float>(logits));
        auto& k = kern(prog, "apply_temperature");
        vulkore::launch(k, {kVocab}, d_x, 1.0f / T, kVocab).wait();
        std::vector<float> gpu(kVocab), cpu(kVocab);
        d_x.download(std::span<float>(gpu));
        for (uint32_t i = 0; i < kVocab; ++i) cpu[i] = logits[i] / T;
        report("apply_temperature (T=0.7)", gpu, cpu, false);
    }

    // ==================================================================
    // Top-k
    // ==================================================================
    {
        const uint32_t K = 40, NP = 256, NCAND = NP * K;
        auto logits = rand_vec(kVocab);
        // Plant an exact duplicate so the tie rule is exercised inside the set.
        logits[1234] = 3.25f;
        logits[9999] = 3.25f;

        auto d_x    = ctx.alloc<float>(kVocab);
        auto d_cval = ctx.alloc<float>(NCAND);
        auto d_cidx = ctx.alloc<uint32_t>(NCAND);
        auto d_oval = ctx.alloc<float>(K);
        auto d_oidx = ctx.alloc<uint32_t>(K);
        d_x.upload(std::span<const float>(logits));
        auto seed = sentinel_vec(K);
        d_oval.upload(std::span<const float>(seed));

        auto& kp = kern(prog, "topk_partial");
        auto& kr = kern(prog, "topk_rank_select");
        {
            vulkore::Batch b(ctx);
            b.add(kp, {NP},    d_cval, d_cidx, d_x, kVocab, NP, K);
            b.add(kr, {NCAND}, d_oval, d_oidx, d_cval, d_cidx, NCAND, K);
            b.submit().wait();
        }
        std::vector<float> gval(K);
        std::vector<uint32_t> gidx(K);
        d_oval.download(std::span<float>(gval));
        d_oidx.download(std::span<uint32_t>(gidx));

        // CPU reference: stable sort by (value desc, index asc), take K.
        // Sorted to 64, not K — the k-boundary check below reuses this and
        // partial_sort leaves everything past its middle iterator unordered.
        std::vector<uint32_t> order(kVocab);
        std::iota(order.begin(), order.end(), 0u);
        std::partial_sort(order.begin(), order.begin() + 64, order.end(),
                          [&](uint32_t a, uint32_t b) {
                              if (logits[a] != logits[b]) return logits[a] > logits[b];
                              return a < b;
                          });

        bool set_ok = true, val_ok = true, desc_ok = true;
        for (uint32_t j = 0; j < K; ++j) {
            if (gidx[j] != order[j]) set_ok = false;
            if (gval[j] != logits[order[j]]) val_ok = false;
            if (j && gval[j] > gval[j - 1]) desc_ok = false;
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, "k=%u of %u  top=%u(%.4f)  kth=%u(%.4f)",
                      K, kVocab, gidx[0], gval[0], gidx[K - 1], gval[K - 1]);
        ok("top-k index set == CPU sort", set_ok, buf);
        ok("top-k values match", val_ok);
        ok("top-k output sorted descending", desc_ok);
        bool live = std::none_of(gval.begin(), gval.end(),
                                 [](float v) { return v == kSentinel; });
        ok("top-k liveness (all k written)", live);

        // ---- k = 1 must agree with argmax, and k = 64 (XP_TOPK_MAX) must not
        // overrun the private array.
        for (uint32_t kk : {1u, 64u}) {
            auto d_cv = ctx.alloc<float>(NP * kk);
            auto d_ci = ctx.alloc<uint32_t>(NP * kk);
            auto d_ov = ctx.alloc<float>(kk);
            auto d_oi = ctx.alloc<uint32_t>(kk);
            vulkore::Batch b(ctx);
            b.add(kp, {NP},      d_cv, d_ci, d_x, kVocab, NP, kk);
            b.add(kr, {NP * kk}, d_ov, d_oi, d_cv, d_ci, NP * kk, kk);
            b.submit().wait();
            std::vector<uint32_t> gi(kk);
            d_oi.download(std::span<uint32_t>(gi));
            bool m = true;
            for (uint32_t j = 0; j < kk; ++j) if (gi[j] != order[j]) m = false;
            std::snprintf(buf, sizeof buf, "k=%u", kk);
            ok("top-k at k boundary", m, buf);
        }

        // ---- topk_mask + softmax: exactly K tokens carry the whole mass.
        {
            auto d_p     = ctx.alloc<float>(kVocab);
            auto d_pmax  = ctx.alloc<float>(kParts);
            auto d_psum  = ctx.alloc<float>(kParts);
            auto d_stats = ctx.alloc<float>(2);
            d_x.upload(std::span<const float>(logits));
            auto& kmk = kern(prog, "topk_mask");
            auto& km = kern(prog, "reduce_max_partial");
            auto& kmf = kern(prog, "reduce_max_final");
            auto& ke = kern(prog, "softmax_exp_partial");
            auto& ksf = kern(prog, "reduce_sum_final");
            auto& kn = kern(prog, "softmax_normalize");
            vulkore::Batch b(ctx);
            b.add(kmk, {kVocab}, d_x, d_oval, K, kVocab);
            b.add(km,  {kParts}, d_pmax, d_x, kVocab, kParts);
            b.add(kmf, {1},      d_stats, d_pmax, kParts);
            b.add(ke,  {kParts}, d_p, d_psum, d_x, d_stats, kVocab, kParts);
            b.add(ksf, {1},      d_stats, d_psum, kParts);
            b.add(kn,  {kVocab}, d_p, d_stats, kVocab);
            b.submit().wait();
            std::vector<float> p(kVocab);
            d_p.download(std::span<float>(p));

            uint32_t nonzero = 0;
            double mass = 0.0;
            for (uint32_t i = 0; i < kVocab; ++i) { if (p[i] > 0.0f) ++nonzero; mass += p[i]; }
            std::snprintf(buf, sizeof buf, "nonzero=%u (want %u)  mass=%.6f", nonzero, K, mass);
            ok("topk_mask -> softmax support == k", nonzero == K && std::fabs(mass - 1.0) < 1e-4, buf);
        }
    }

    // ==================================================================
    // Top-p and multinomial, on the k-element survivor list
    // ==================================================================
    {
        const uint32_t K = 40;
        // A realistic post-softmax shape: descending, heavy head.
        std::vector<float> probs(K);
        std::vector<uint32_t> ids(K);
        double tot = 0.0;
        for (uint32_t j = 0; j < K; ++j) {
            probs[j] = std::exp(-0.35f * float(j));
            ids[j] = 1000u + j * 37u;
            tot += probs[j];
        }
        for (auto& v : probs) v = float(v / tot);

        auto d_w   = ctx.alloc<float>(K);
        auto d_i   = ctx.alloc<uint32_t>(K);
        auto d_out = ctx.alloc<float>(2);
        auto d_oi  = ctx.alloc<uint32_t>(1);
        d_i.upload(std::span<const uint32_t>(ids));

        // ---- top-p cutoff
        for (float p : {0.5f, 0.9f, 0.99f}) {
            d_w.upload(std::span<const float>(probs));
            vulkore::launch(kern(prog, "topp_cutoff"), {1}, d_w, K, p).wait();
            std::vector<float> g(K);
            d_w.download(std::span<float>(g));

            std::vector<float> c = probs;
            double total = 0.0;
            for (float v : probs) total += v;
            double cum = 0.0;
            for (uint32_t j = 0; j < K; ++j) {
                if (cum >= p * total) c[j] = 0.0f;
                else cum += c[j];
            }
            uint32_t kept = 0;
            for (uint32_t j = 0; j < K; ++j) if (g[j] > 0.0f) ++kept;
            char buf[96];
            std::snprintf(buf, sizeof buf, "p=%.2f kept=%u/%u", p, kept, K);
            std::vector<float> gv(g.begin(), g.end()), cv(c.begin(), c.end());
            report(("topp_cutoff " + std::string(buf)).c_str(), gv, cv, false);
        }

        // ---- multinomial over the survivors, against an identical CPU walk
        {
            d_w.upload(std::span<const float>(probs));
            bool all = true;
            uint32_t checked = 0;
            for (int s = 0; s < 32; ++s) {
                float uu = float(s) / 32.0f;
                vulkore::launch(kern(prog, "multinomial_sample"), {1},
                              d_out, d_oi, d_w, d_i, K, uu).wait();
                std::vector<float> g(2);
                d_out.download(std::span<float>(g));

                double total = 0.0;
                for (float v : probs) total += v;
                double target = double(uu) * total, cum = 0.0;
                uint32_t chosen = 0;
                for (uint32_t j = 0; j < K; ++j) {
                    cum += probs[j];
                    if (probs[j] > 0.0f) chosen = ids[j];
                    if (cum > target) break;
                }
                if (uint32_t(g[0]) != chosen) all = false;
                ++checked;
            }
            char buf[96];
            std::snprintf(buf, sizeof buf, "%u uniforms checked", checked);
            ok("multinomial_sample vs CPU CDF", all, buf);
        }

        // ---- empirical distribution: the sampler must actually be unbiased,
        // not merely agree with a CPU walk on the same code path.
        {
            d_w.upload(std::span<const float>(probs));
            const int N = 4000;
            std::vector<int> hist(K, 0);
            std::mt19937 r2(7);
            std::uniform_real_distribution<float> uu(0.0f, 1.0f);
            for (int s = 0; s < N; ++s) {
                vulkore::launch(kern(prog, "multinomial_sample"), {1},
                              d_out, d_oi, d_w, d_i, K, uu(r2)).wait();
                std::vector<uint32_t> gi(1);
                d_oi.download(std::span<uint32_t>(gi));
                uint32_t slot = (gi[0] - 1000u) / 37u;
                if (slot < K) ++hist[slot];
            }
            double worst = 0.0;
            for (uint32_t j = 0; j < K; ++j)
                worst = std::max(worst, std::fabs(double(hist[j]) / N - probs[j]));
            char buf[96];
            std::snprintf(buf, sizeof buf,
                          "N=%d  max|emp-p|=%.4f  p0=%.3f emp0=%.3f",
                          N, worst, probs[0], double(hist[0]) / N);
            // 3 sigma at p=0.3, N=4000 is ~0.022; 0.03 is a loose but real bar.
            ok("multinomial empirical distribution", worst < 0.03, buf);
        }
    }

    // ==================================================================
    // Full-vocabulary multinomial without a 262144-long scan
    // ==================================================================
    {
        const uint32_t N = 8192, NP = 64;
        std::vector<float> logits(N);
        for (uint32_t i = 0; i < N; ++i)
            logits[i] = (i % 500u == 0u) ? 4.0f : -3.0f + 0.0001f * float(i);

        auto d_x     = ctx.alloc<float>(N);
        auto d_w     = ctx.alloc<float>(N);
        auto d_pmax  = ctx.alloc<float>(NP);
        auto d_psum  = ctx.alloc<float>(NP);
        auto d_stats = ctx.alloc<float>(2);
        auto d_out   = ctx.alloc<float>(2);
        auto d_oi    = ctx.alloc<uint32_t>(1);
        d_x.upload(std::span<const float>(logits));

        auto& km = kern(prog, "reduce_max_partial");
        auto& kmf = kern(prog, "reduce_max_final");
        auto& ke = kern(prog, "softmax_exp_partial");
        auto& ksf = kern(prog, "reduce_sum_final");
        auto build = [&](vulkore::Batch& b) {
            b.add(km,  {NP}, d_pmax, d_x, N, NP);
            b.add(kmf, {1},  d_stats, d_pmax, NP);
            b.add(ke,  {NP}, d_w, d_psum, d_x, d_stats, N, NP);
            b.add(ksf, {1},  d_stats, d_psum, NP);
        };
        { vulkore::Batch b(ctx); build(b); b.submit().wait(); }

        std::vector<float> w(N), psum(NP), stats(2);
        d_w.download(std::span<float>(w));
        d_psum.download(std::span<float>(psum));
        d_stats.download(std::span<float>(stats));

        // CPU replica of the strided two-level search the kernel performs.
        auto cpu_pick = [&](float uu) {
            double target = double(uu) * stats[1], cum = 0.0;
            uint32_t chain = NP - 1;
            for (uint32_t p = 0; p < NP; ++p) {
                if (cum + psum[p] > target) { chain = p; break; }
                cum += psum[p];
            }
            uint32_t chosen = chain;
            for (uint32_t k = chain; k < N; k += NP) {
                chosen = k;
                cum += w[k];
                if (cum > target) break;
            }
            return chosen;
        };

        bool all = true;
        std::mt19937 r3(11);
        std::uniform_real_distribution<float> uu(0.0f, 1.0f);
        for (int s = 0; s < 64; ++s) {
            float uv = uu(r3);
            vulkore::launch(kern(prog, "multinomial_strided"), {1},
                          d_out, d_oi, d_w, d_psum, d_stats, N, NP, uv).wait();
            std::vector<uint32_t> gi(1);
            d_oi.download(std::span<uint32_t>(gi));
            if (gi[0] != cpu_pick(uv)) all = false;
        }
        ok("multinomial_strided vs CPU walk", all, "64 uniforms, n=8192");

        // The 500-spaced spikes carry ~almost all the mass; a correct sampler
        // lands on them nearly every time. This catches a scan that silently
        // samples the wrong chain.
        {
            int spikes = 0;
            const int T = 500;
            for (int s = 0; s < T; ++s) {
                vulkore::launch(kern(prog, "multinomial_strided"), {1},
                              d_out, d_oi, d_w, d_psum, d_stats, N, NP, uu(r3)).wait();
                std::vector<uint32_t> gi(1);
                d_oi.download(std::span<uint32_t>(gi));
                if (gi[0] % 500u == 0u) ++spikes;
            }
            double expected = 0.0, tot = 0.0;
            for (uint32_t i = 0; i < N; ++i) { tot += w[i]; if (i % 500u == 0u) expected += w[i]; }
            expected /= tot;
            char buf[96];
            std::snprintf(buf, sizeof buf, "hit spikes %d/%d, expected %.3f",
                          spikes, T, expected);
            ok("multinomial_strided mass lands right",
               std::fabs(double(spikes) / T - expected) < 0.05, buf);
        }
    }

    // ==================================================================
    // PREFILL SHAPES
    // ==================================================================
    const uint32_t kT = 37;            // deliberately not a multiple of 64
    const uint32_t kSeq = 37;

    // ---- softmax over many rows
    {
        const uint32_t rows = kHeads * kT, cols = kSeq;
        const uint32_t n = rows * cols;
        auto x = rand_vec(n);
        auto d_x = ctx.alloc<float>(n);
        auto d_s = ctx.alloc<float>(2 * rows);
        d_x.upload(std::span<const float>(x));
        {
            auto& kst = kern(prog, "softmax_rows_stats");
            auto& kap = kern(prog, "softmax_rows_apply");
            vulkore::Batch b(ctx);
            b.add(kst, {rows}, d_s, d_x, rows, cols);
            b.add(kap, {n},    d_x, d_s, rows, cols);
            b.submit().wait();
        }
        std::vector<float> gpu(n), cpu(n);
        d_x.download(std::span<float>(gpu));
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t base = r * cols;
            float m = -std::numeric_limits<float>::infinity();
            for (uint32_t j = 0; j < cols; ++j) m = std::fmax(m, x[base + j]);
            float s = 0.0f;
            for (uint32_t j = 0; j < cols; ++j) s += std::exp(x[base + j] - m);
            for (uint32_t j = 0; j < cols; ++j) cpu[base + j] = std::exp(x[base + j] - m) / s;
        }
        report("softmax_rows_prefill", gpu, cpu, false);

        double worst = 0.0;
        for (uint32_t r = 0; r < rows; ++r) {
            double s = 0.0;
            for (uint32_t j = 0; j < cols; ++j) s += gpu[r * cols + j];
            worst = std::max(worst, std::fabs(s - 1.0));
        }
        char buf[80];
        std::snprintf(buf, sizeof buf, "%u rows, max|rowsum-1|=%.3e", rows, worst);
        ok("prefill softmax rows sum to 1", worst < 1e-4, buf);
    }

    // ---- prefill attention: causal mask, sliding window, and the decode
    // equivalence check. The last one is the important one: with n_query=1 and
    // q_offset=pos, the prefill kernels must reproduce llm_transformer.cl's
    // decode kernels bit-for-bit, or the two paths will silently disagree.
    {
        const uint32_t nq = kHeads * kT * kHeadDim;
        const uint32_t nsc = kHeads * kT * kSeq;
        const uint32_t ncache = kSeq * kKvHeads * kHeadDim;
        const float scale = 1.0f / std::sqrt(float(kHeadDim));

        auto q = rand_vec(nq), kc = rand_vec(ncache), vc = rand_vec(ncache);
        auto d_q = ctx.alloc<float>(nq);
        auto d_k = ctx.alloc<float>(ncache);
        auto d_v = ctx.alloc<float>(ncache);
        auto d_s = ctx.alloc<float>(nsc);
        auto d_o = ctx.alloc<float>(nq);
        d_q.upload(std::span<const float>(q));
        d_k.upload(std::span<const float>(kc));
        d_v.upload(std::span<const float>(vc));

        auto& ksc = kern(prog, "attention_scores_prefill");
        uint32_t group = kHeads / kKvHeads;

        for (uint32_t window : {0u, 8u}) {
            auto seed = sentinel_vec(nsc);
            d_s.upload(std::span<const float>(seed));
            vulkore::launch(ksc, {nsc}, d_s, d_q, d_k, kHeads, kKvHeads, kHeadDim,
                          kT, kSeq, 0u, window, scale).wait();
            std::vector<float> gpu(nsc), cpu(nsc);
            d_s.download(std::span<float>(gpu));
            for (uint32_t h = 0; h < kHeads; ++h)
                for (uint32_t qi = 0; qi < kT; ++qi)
                    for (uint32_t t = 0; t < kSeq; ++t) {
                        uint32_t o = (h * kT + qi) * kSeq + t;
                        if (t > qi || (window && t + window <= qi)) {
                            cpu[o] = -std::numeric_limits<float>::infinity();
                            continue;
                        }
                        float acc = 0.0f;
                        for (uint32_t d = 0; d < kHeadDim; ++d)
                            acc += q[(h * kT + qi) * kHeadDim + d] *
                                   kc[(t * kKvHeads + h / group) * kHeadDim + d];
                        cpu[o] = acc * scale;
                    }
            char nm[64];
            std::snprintf(nm, sizeof nm, "attention_scores_prefill w=%u", window);
            report(nm, gpu, cpu);
        }

        // softmax + apply on the causal scores
        {
            auto d_st = ctx.alloc<float>(2 * kHeads * kT);
            auto seedo = sentinel_vec(nq);
            d_o.upload(std::span<const float>(seedo));
            auto& kst = kern(prog, "softmax_rows_stats");
            auto& kap = kern(prog, "softmax_rows_apply");
            auto& kaa = kern(prog, "attention_apply_prefill");
            vulkore::Batch b(ctx);
            b.add(ksc, {nsc}, d_s, d_q, d_k, kHeads, kKvHeads, kHeadDim, kT, kSeq, 0u, 0u, scale);
            b.add(kst, {kHeads * kT}, d_st, d_s, kHeads * kT, kSeq);
            b.add(kap, {nsc},         d_s, d_st, kHeads * kT, kSeq);
            b.add(kaa, {nq}, d_o, d_s, d_v, kHeads, kKvHeads, kHeadDim, kT, kSeq);
            b.submit().wait();

            std::vector<float> gpu(nq), cpu(nq);
            d_o.download(std::span<float>(gpu));
            for (uint32_t h = 0; h < kHeads; ++h) {
                uint32_t kvh = h / group;
                for (uint32_t qi = 0; qi < kT; ++qi) {
                    std::vector<float> p(kSeq, 0.0f);
                    float m = -std::numeric_limits<float>::infinity();
                    for (uint32_t t = 0; t <= qi; ++t) {
                        float acc = 0.0f;
                        for (uint32_t d = 0; d < kHeadDim; ++d)
                            acc += q[(h * kT + qi) * kHeadDim + d] *
                                   kc[(t * kKvHeads + kvh) * kHeadDim + d];
                        p[t] = acc * scale;
                        m = std::fmax(m, p[t]);
                    }
                    float s = 0.0f;
                    for (uint32_t t = 0; t <= qi; ++t) { p[t] = std::exp(p[t] - m); s += p[t]; }
                    for (uint32_t d = 0; d < kHeadDim; ++d) {
                        float acc = 0.0f;
                        for (uint32_t t = 0; t <= qi; ++t)
                            acc += (p[t] / s) * vc[(t * kKvHeads + kvh) * kHeadDim + d];
                        cpu[(h * kT + qi) * kHeadDim + d] = acc;
                    }
                }
            }
            report("prefill attention block (causal)", gpu, cpu);
        }

        // ---- decode equivalence: n_query = 1, q_offset = seq_len-1.
        {
            const uint32_t pos = kSeq - 1;
            const uint32_t n1 = kHeads * kHeadDim, s1 = kHeads * kSeq;
            auto q1 = rand_vec(n1);
            auto d_q1 = ctx.alloc<float>(n1);
            auto d_s1 = ctx.alloc<float>(s1);
            auto d_s2 = ctx.alloc<float>(s1);
            d_q1.upload(std::span<const float>(q1));
            vulkore::launch(ksc, {s1}, d_s1, d_q1, d_k, kHeads, kKvHeads, kHeadDim,
                          1u, kSeq, pos, 0u, scale).wait();
            vulkore::launch(kern(tprog, "attention_scores"), {s1}, d_s2, d_q1, d_k,
                          kHeads, kKvHeads, kHeadDim, kSeq, scale).wait();
            std::vector<float> a(s1), b(s1);
            d_s1.download(std::span<float>(a));
            d_s2.download(std::span<float>(b));
            report("prefill scores == decode scores", a, b, false);
        }
    }

    // ---- prefill RMSNorm, checked against the decode two-pass pair
    {
        const uint32_t n = kT * kHidden;
        auto in = rand_vec(n), w = rand_vec(kHidden);
        auto d_in = ctx.alloc<float>(n);
        auto d_w  = ctx.alloc<float>(kHidden);
        auto d_ss = ctx.alloc<float>(kT);
        auto d_o  = ctx.alloc<float>(n);
        d_in.upload(std::span<const float>(in));
        d_w.upload(std::span<const float>(w));
        auto seed = sentinel_vec(n);
        d_o.upload(std::span<const float>(seed));
        {
            auto& k1 = kern(prog, "rmsnorm_rows_sumsq");
            auto& k2 = kern(prog, "rmsnorm_rows_apply_gemma");
            vulkore::Batch b(ctx);
            b.add(k1, {kT}, d_ss, d_in, kT, kHidden);
            b.add(k2, {n},  d_o, d_in, d_w, d_ss, kT, kHidden, kEps);
            b.submit().wait();
        }
        std::vector<float> gpu(n), cpu(n);
        d_o.download(std::span<float>(gpu));
        for (uint32_t r = 0; r < kT; ++r) {
            double ss = 0.0;
            for (uint32_t j = 0; j < kHidden; ++j) {
                double v = in[r * kHidden + j];
                ss += v * v;
            }
            float sc = 1.0f / std::sqrt(float(ss / kHidden) + kEps);
            for (uint32_t j = 0; j < kHidden; ++j)
                cpu[r * kHidden + j] = in[r * kHidden + j] * sc * (1.0f + w[j]);
        }
        report("rmsnorm_rows_gemma", gpu, cpu);

        // per-head QK-norm over many tokens
        const uint32_t nh = kT * kHeads * kHeadDim;
        auto in2 = rand_vec(nh), w2 = rand_vec(kHeadDim);
        auto d_in2 = ctx.alloc<float>(nh);
        auto d_w2  = ctx.alloc<float>(kHeadDim);
        auto d_o2  = ctx.alloc<float>(nh);
        d_in2.upload(std::span<const float>(in2));
        d_w2.upload(std::span<const float>(w2));
        auto seed2 = sentinel_vec(nh);
        d_o2.upload(std::span<const float>(seed2));
        vulkore::launch(kern(prog, "rmsnorm_heads_rows_gemma"), {nh},
                      d_o2, d_in2, d_w2, kT, kHeads, kHeadDim, kEps).wait();
        std::vector<float> g2(nh), c2(nh);
        d_o2.download(std::span<float>(g2));
        for (uint32_t head = 0; head < kT * kHeads; ++head) {
            double ss = 0.0;
            for (uint32_t j = 0; j < kHeadDim; ++j) {
                double v = in2[head * kHeadDim + j];
                ss += v * v;
            }
            float sc = 1.0f / std::sqrt(float(ss / kHeadDim) + kEps);
            for (uint32_t j = 0; j < kHeadDim; ++j)
                c2[head * kHeadDim + j] = in2[head * kHeadDim + j] * sc * (1.0f + w2[j]);
        }
        report("rmsnorm_heads_rows_gemma", g2, c2);
    }

    // ---- prefill RoPE, checked row-by-row against the decode `rope` kernel at
    // the matching position. Getting rotate-half vs interleaved wrong is silent
    // (the model just loses positional sense), so this compares against the
    // convention already validated in llm_transformer.cl rather than a fresh
    // CPU reference that could repeat the same mistake.
    {
        const uint32_t pos0 = 5;
        const uint32_t per = kHeads * kHeadDim, n = kT * per;
        auto x = rand_vec(n);
        auto d_x = ctx.alloc<float>(n);
        d_x.upload(std::span<const float>(x));
        vulkore::launch(kern(prog, "rope_prefill"), {kT * kHeads * kHeadDim / 2},
                      d_x, kT, kHeads, kHeadDim, pos0, kRopeBase).wait();
        std::vector<float> gpu(n);
        d_x.download(std::span<float>(gpu));

        auto d_row = ctx.alloc<float>(per);
        std::vector<float> cpu(n);
        for (uint32_t t = 0; t < kT; ++t) {
            std::vector<float> row(x.begin() + t * per, x.begin() + (t + 1) * per);
            d_row.upload(std::span<const float>(row));
            vulkore::launch(kern(tprog, "rope"), {kHeads * kHeadDim / 2},
                          d_row, kHeads, kHeadDim, pos0 + t, kRopeBase).wait();
            d_row.download(std::span<float>(std::span<float>(cpu).subspan(t * per, per)));
        }
        report("rope_prefill == decode rope", gpu, cpu, false);

        int changed = 0;
        for (uint32_t i = 0; i < n; ++i) if (gpu[i] != x[i]) ++changed;
        char buf[80];
        std::snprintf(buf, sizeof buf, "%d/%u elements changed", changed, n);
        ok("rope_prefill liveness", changed > int(n) * 9 / 10, buf);
    }

    // ---- prefill plumbing
    {
        const uint32_t row = kKvHeads * kHeadDim, cap = 64, pos0 = 11;
        std::vector<float> cache(row * cap, 7.0f);
        auto src = rand_vec(row * kT);
        auto d_c = ctx.alloc<float>(cache.size());
        auto d_s = ctx.alloc<float>(row * kT);
        d_c.upload(std::span<const float>(cache));
        d_s.upload(std::span<const float>(src));
        vulkore::launch(kern(prog, "kv_append_rows"), {kT * row},
                      d_c, d_s, kT, row, pos0).wait();
        std::vector<float> gpu(cache.size());
        d_c.download(std::span<float>(gpu));
        auto cpu = cache;
        for (uint32_t i = 0; i < kT * row; ++i) cpu[pos0 * row + i] = src[i];
        report("kv_append_rows", gpu, cpu, false);

        // embed_lookup_rows on a stand-in table (the real one is 262144x1152
        // fp32 = 1.2 GB, above maxStorageBufferRange on the target devices).
        const uint32_t vocab = 4096;
        std::vector<float> table(size_t(vocab) * kHidden);
        for (size_t i = 0; i < table.size(); ++i) table[i] = float(i % 977) * 0.01f;
        std::vector<uint32_t> toks(kT);
        for (uint32_t t = 0; t < kT; ++t) toks[t] = (t * 251u + 13u) % vocab;
        auto d_t = ctx.alloc<float>(table.size());
        auto d_k = ctx.alloc<uint32_t>(kT);
        auto d_o = ctx.alloc<float>(kT * kHidden);
        d_t.upload(std::span<const float>(table));
        d_k.upload(std::span<const uint32_t>(toks));
        auto seed = sentinel_vec(kT * kHidden);
        d_o.upload(std::span<const float>(seed));
        vulkore::launch(kern(prog, "embed_lookup_rows"), {kT * kHidden},
                      d_o, d_t, d_k, kT, kHidden).wait();
        std::vector<float> g(kT * kHidden), c(kT * kHidden);
        d_o.download(std::span<float>(g));
        for (uint32_t t = 0; t < kT; ++t)
            for (uint32_t j = 0; j < kHidden; ++j)
                c[t * kHidden + j] = table[size_t(toks[t]) * kHidden + j];
        report("embed_lookup_rows", g, c);
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (t_par > 0.0)
        std::printf("parallel argmax speedup at vocab %u: %.1fx "
                    "(%.4f ms -> %.4f ms)\n", kVocab, t_serial / t_par, t_serial, t_par);
    return g_fail == 0 ? 0 : 1;
}
