// Per-kernel attention profiler: which of attention_scores / softmax_rows /
// attention_apply actually grows with context depth?
//
// The decode loop's ms/token is near-linear in position (12.9 ms at pos 40,
// 62.6 ms at pos 8288) even though 22 of 26 layers are windowed to 512. The
// four global layers are the suspects, and inside a global layer there are
// exactly three O(span) dispatches. This times each ONE IN ISOLATION at real
// Gemma 3 1B shapes across a span sweep, so the attribution is measured rather
// than argued.
//
// Method: R back-to-back dispatches of a single kernel inside ONE vulkore::Batch
// (one submit, so submit latency is amortised, and the Batch's barriers keep
// them serialised so R*t is really R times the kernel). A trivial-kernel arm
// measures the per-dispatch floor so it can be subtracted mentally.
#include <vulkore/vulkore.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kNHeads  = 4;
constexpr uint32_t kKVHeads = 1;
constexpr uint32_t kHeadDim = 256;

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// best-of-N, because a single sample on a phone measures the governor.
struct Timer {
  int reps;
  template <typename F>
  double best(F&& f) {
    double b = 1e30;
    for (int i = 0; i < reps; ++i) {
      double t0 = now_ms();
      f();
      b = std::min(b, now_ms() - t0);
    }
    return b;
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string tf_path  = "kernels/llm_transformer.spv";
  std::vector<uint32_t> spans = {41, 512, 1000, 2000, 4176, 8288};
  uint32_t R = 64;      // dispatches per timed batch
  int best_of = 5;
  // Re-reading ONE kcache R times back to back lets it sit in the system-level
  // cache, which the real decode loop never gets: between two attention
  // dispatches it streams 536 MiB of weights and evicts everything. Rotating
  // over several distinct caches defeats that and is the honest number.
  uint32_t rotate = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* d) { return (i + 1 < argc) ? argv[++i] : d; };
    if (a == "--transformer") tf_path = next(tf_path.c_str());
    else if (a == "--reps")   R = uint32_t(std::stoul(next("64")));
    else if (a == "--best-of") best_of = std::stoi(next("5"));
    else if (a == "--rotate") rotate = uint32_t(std::stoul(next("1")));
    else if (a == "--spans") {
      spans.clear();
      std::string s = next("");
      size_t p = 0;
      while (p < s.size()) {
        size_t c = s.find(',', p);
        if (c == std::string::npos) c = s.size();
        spans.push_back(uint32_t(std::stoul(s.substr(p, c - p))));
        p = c + 1;
      }
    }
  }

  vulkore::Context ctx;
  std::printf("device      %s\n", ctx.device_name().c_str());
  std::printf("shapes      n_heads=%u kv_heads=%u head_dim=%u  (Gemma 3 1B)\n",
              kNHeads, kKVHeads, kHeadDim);
  std::printf("method      %u dispatches per submit, best of %d\n\n", R, best_of);

  auto prog = vulkore::Program::from_file(ctx, tf_path);
  auto k_scores = prog.kernel("attention_scores");
  auto k_soft   = prog.kernel("softmax_rows");
  auto k_apply  = prog.kernel("attention_apply");
  auto k_resid  = prog.kernel("add_residual");   // trivial-dispatch floor probe
  // The new arms live in the SAME module as the old ones, so an A/B here never
  // confounds the kernel change with a .spv regeneration — a trap this repo has
  // already paid for once (llm-sliding-window.md).
  auto k_sm_part = prog.kernel("softmax_rows_partial");
  auto k_sm_fin  = prog.kernel("softmax_rows_finish");
  auto k_sm_sc   = prog.kernel("softmax_rows_scale");
  auto k_mq4     = prog.kernel("attention_scores_mq4");
  auto k_ap_mq4  = prog.kernel("attention_apply_mq4");
  auto k_ap_red  = prog.kernel("attention_apply_reduce");

  Timer timer{best_of};
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.f, 1.f);

  // Per-dispatch floor: add_residual over 64 elements does essentially nothing.
  double floor_ms = 0;
  {
    auto a = ctx.alloc<float>(64), b = ctx.alloc<float>(64);
    std::vector<float> h(64, 0.f);
    a.upload(std::span<const float>(h));
    b.upload(std::span<const float>(h));
    floor_ms = timer.best([&] {
      vulkore::Batch batch(ctx);
      for (uint32_t r = 0; r < R; ++r) batch.add(k_resid, {64}, a, b, 64u);
      batch.submit().wait();
    }) / R;
  }
  std::printf("per-dispatch floor (trivial kernel)   %.4f ms\n\n", floor_ms);

  std::printf("%8s %12s %12s %12s %12s   %10s %10s\n",
              "span", "scores ms", "softmax ms", "apply ms", "TOTAL ms",
              "soft %", "KV MiB");
  std::printf("%s\n", std::string(92, '-').c_str());

  struct NewRow {
    uint32_t span; double scores, soft; uint32_t nparts;
    double sc_maxrel, sm_maxabs, sm_rowsum;
    double old_scores, old_soft;
    double apply; uint32_t split; double ap_maxrel, old_apply;
  };
  std::vector<NewRow> new_rows;

  double base_total = 0;
  for (uint32_t span : spans) {
    const size_t kv_n = size_t(span) * kKVHeads * kHeadDim;
    const size_t q_n  = size_t(kNHeads) * kHeadDim;
    const size_t sc_n = size_t(kNHeads) * span;

    std::vector<float> hq(q_n), hkv(kv_n);
    for (auto& v : hq) v = u(rng);
    for (auto& v : hkv) v = u(rng);

    auto q  = ctx.alloc<float>(q_n);
    std::vector<vulkore::Buffer> kcs, vcs;
    for (uint32_t i = 0; i < rotate; ++i) {
      kcs.push_back(ctx.alloc<float>(kv_n));
      vcs.push_back(ctx.alloc<float>(kv_n));
      kcs.back().upload(std::span<const float>(hkv));
      vcs.back().upload(std::span<const float>(hkv));
    }
    auto& kc = kcs[0];
    auto& vc = vcs[0];
    auto sc = ctx.alloc<float>(sc_n);
    auto out = ctx.alloc<float>(q_n, vulkore::Usage::HostVisible);
    q.upload(std::span<const float>(hq));

    const float ascale = 1.0f / std::sqrt(float(kHeadDim));
    const uint32_t kv_start = 0;

    // --- scores, isolated -------------------------------------------------
    double t_scores = timer.best([&] {
      vulkore::Batch b(ctx);
      for (uint32_t r = 0; r < R; ++r)
        b.add(k_scores, {kNHeads * span}, sc, q, kcs[r % rotate], kNHeads,
              kKVHeads, kHeadDim, span, ascale, kv_start);
      b.submit().wait();
    }) / R;

    // --- softmax, isolated. Run scores once first so the input is realistic.
    {
      vulkore::Batch b(ctx);
      b.add(k_scores, {kNHeads * span}, sc, q, kc, kNHeads, kKVHeads,
            kHeadDim, span, ascale, kv_start);
      b.submit().wait();
    }
    double t_soft = timer.best([&] {
      vulkore::Batch b(ctx);
      for (uint32_t r = 0; r < R; ++r) b.add(k_soft, {kNHeads}, sc, kNHeads, span);
      b.submit().wait();
    }) / R;

    // --- apply, isolated --------------------------------------------------
    double t_apply = timer.best([&] {
      vulkore::Batch b(ctx);
      for (uint32_t r = 0; r < R; ++r)
        b.add(k_apply, {q_n}, out, sc, vcs[r % rotate], kNHeads, kKVHeads,
              kHeadDim, span, kv_start);
      b.submit().wait();
    }) / R;

    // LIVENESS: prove the kernels wrote something. A row of a softmaxed score
    // matrix sums to 1 and the apply output must be finite and non-zero.
    std::vector<float> hsc(sc_n), hout(q_n);
    sc.download(std::span<float>(hsc));
    out.download(std::span<float>(hout));
    double rowsum = 0;
    for (uint32_t j = 0; j < span; ++j) rowsum += hsc[j];
    double omax = 0;
    for (float v : hout) omax = std::max(omax, double(std::fabs(v)));
    bool live = std::fabs(rowsum - 1.0) < 1e-3 && omax > 1e-9 && std::isfinite(omax);

    // ================= NEW ARMS =========================================
    // nparts ~ sqrt(span) balances pass 1 (n_cols/nparts iterations on
    // n_rows*nparts threads) against pass 2 (nparts iterations on n_rows).
    uint32_t nparts = 1;
    while (nparts * nparts < span && nparts < 512) nparts *= 2;
    if (nparts > span) nparts = span;

    auto pmax = ctx.alloc<float>(size_t(kNHeads) * nparts);
    auto psum = ctx.alloc<float>(size_t(kNHeads) * nparts);
    auto stats = ctx.alloc<float>(size_t(kNHeads) * 2);

    // Fresh scores for the new arms, so both softmaxes see the SAME input.
    auto sc_ref = ctx.alloc<float>(sc_n);
    {
      vulkore::Batch b(ctx);
      b.add(k_scores, {kNHeads * span}, sc_ref, q, kc, kNHeads, kKVHeads,
            kHeadDim, span, ascale, kv_start);
      b.submit().wait();
    }
    std::vector<float> h_ref(sc_n);
    sc_ref.download(std::span<float>(h_ref));

    auto sc_new = ctx.alloc<float>(sc_n);
    double t_soft_new = timer.best([&] {
      vulkore::Batch b(ctx);
      for (uint32_t r = 0; r < R; ++r) {
        b.add(k_sm_part, {kNHeads * nparts}, pmax, psum, sc_new, kNHeads, span, nparts);
        b.add(k_sm_fin, {kNHeads}, stats, pmax, psum, kNHeads, nparts);
        b.add(k_sm_sc, {uint32_t(sc_n)}, sc_new, stats, kNHeads, span);
      }
      b.submit().wait();
    }) / R;

    double t_scores_new = timer.best([&] {
      vulkore::Batch b(ctx);
      for (uint32_t r = 0; r < R; ++r)
        b.add(k_mq4, {span}, sc_new, q, kcs[r % rotate], kHeadDim, span,
              ascale, kv_start);
      b.submit().wait();
    }) / R;

    // --- CORRECTNESS, both arms, against the old kernels on the same input --
    // A clean compile proves nothing on this device; check the numbers.
    double sc_maxrel = 0, sm_maxabs = 0, sm_sum = 0;
    {
      // scores: mq4 vs generic
      auto a = ctx.alloc<float>(sc_n), bb = ctx.alloc<float>(sc_n);
      {
        vulkore::Batch b(ctx);
        b.add(k_scores, {kNHeads * span}, a, q, kc, kNHeads, kKVHeads, kHeadDim,
              span, ascale, kv_start);
        b.add(k_mq4, {span}, bb, q, kc, kHeadDim, span, ascale, kv_start);
        b.submit().wait();
      }
      std::vector<float> ha(sc_n), hb(sc_n);
      a.download(std::span<float>(ha));
      bb.download(std::span<float>(hb));
      double rms = 0;
      for (float v : ha) rms += double(v) * v;
      rms = std::sqrt(rms / double(sc_n)) + 1e-30;
      for (size_t i = 0; i < sc_n; ++i)
        sc_maxrel = std::max(sc_maxrel, std::fabs(double(ha[i]) - hb[i]) / rms);

      // softmax: parallel vs serial, both fed h_ref
      auto c = ctx.alloc<float>(sc_n), d = ctx.alloc<float>(sc_n);
      c.upload(std::span<const float>(h_ref));
      d.upload(std::span<const float>(h_ref));
      {
        vulkore::Batch b(ctx);
        b.add(k_soft, {kNHeads}, c, kNHeads, span);
        b.add(k_sm_part, {kNHeads * nparts}, pmax, psum, d, kNHeads, span, nparts);
        b.add(k_sm_fin, {kNHeads}, stats, pmax, psum, kNHeads, nparts);
        b.add(k_sm_sc, {uint32_t(sc_n)}, d, stats, kNHeads, span);
        b.submit().wait();
      }
      std::vector<float> hc(sc_n), hd(sc_n);
      c.download(std::span<float>(hc));
      d.download(std::span<float>(hd));
      for (size_t i = 0; i < sc_n; ++i)
        sm_maxabs = std::max(sm_maxabs, std::fabs(double(hc[i]) - hd[i]));
      for (uint32_t j = 0; j < span; ++j) sm_sum += hd[j];   // row 0 must sum to 1
    }
    // --- apply_mq4 + reduce, sweeping the split factor -------------------
    double t_apply_new = 1e30; uint32_t best_split = 1; double ap_maxrel = 0;
    {
      std::vector<float> ref(q_n);
      {   // generic apply, the reference
        auto o = ctx.alloc<float>(q_n, vulkore::Usage::HostVisible);
        vulkore::Batch b(ctx);
        b.add(k_apply, {q_n}, o, sc_ref, vc, kNHeads, kKVHeads, kHeadDim, span, kv_start);
        b.submit().wait();
        o.download(std::span<float>(ref));
      }
      double rms = 0;
      for (float v : ref) rms += double(v) * v;
      rms = std::sqrt(rms / double(q_n)) + 1e-30;

      for (uint32_t sp : {1u, 2u, 4u, 8u, 16u}) {
        if (sp > span) break;
        auto part = ctx.alloc<float>(q_n * sp);
        auto o = ctx.alloc<float>(q_n, vulkore::Usage::HostVisible);
        double t = timer.best([&] {
          vulkore::Batch b(ctx);
          for (uint32_t r = 0; r < R; ++r) {
            b.add(k_ap_mq4, {kHeadDim * sp}, part, sc_ref, vcs[r % rotate],
                  kHeadDim, span, kv_start, sp);
            b.add(k_ap_red, {uint32_t(q_n)}, o, part, uint32_t(q_n), sp);
          }
          b.submit().wait();
        }) / R;
        std::vector<float> hv(q_n);
        o.download(std::span<float>(hv));
        double mr = 0;
        for (size_t i2 = 0; i2 < q_n; ++i2)
          mr = std::max(mr, std::fabs(double(ref[i2]) - hv[i2]) / rms);
        if (t < t_apply_new) { t_apply_new = t; best_split = sp; ap_maxrel = mr; }
      }
    }
    new_rows.push_back({span, t_scores_new, t_soft_new, nparts, sc_maxrel,
                        sm_maxabs, sm_sum, t_scores, t_soft,
                        t_apply_new, best_split, ap_maxrel, t_apply});

    double total = t_scores + t_soft + t_apply;
    if (base_total == 0) base_total = total;
    double kv_mib = double(kv_n) * 4 * 2 / (1024.0 * 1024.0);

    std::printf("%8u %12.4f %12.4f %12.4f %12.4f   %9.1f%% %10.2f  %s\n",
                span, t_scores, t_soft, t_apply, total,
                100.0 * t_soft / total, kv_mib,
                live ? "" : "!! NOT LIVE (rowsum/out check failed)");
    if (!live)
      std::printf("      liveness detail: rowsum=%.6f  max|out|=%.3e\n", rowsum, omax);
  }

  std::printf("\n--- NEW kernels (same .spv module, so no regeneration confound) ---\n");
  std::printf("%7s %6s %11s %11s %6s %11s  %7s %7s %7s  %10s %10s %10s %9s\n",
              "span", "nparts", "scores_mq4", "softmax x3", "split", "apply_mq4",
              "sc x", "sm x", "ap x", "sc maxrel", "sm maxabs", "ap maxrel",
              "row sum");
  std::printf("%s\n", std::string(132, '-').c_str());
  for (size_t i = 0; i < new_rows.size(); ++i) {
    const auto& n = new_rows[i];
    // old timings live in the first table; recompute speedups from the sweep
    char s1[16], s2[16], s3[16];
    std::snprintf(s1, sizeof s1, "%.2fx", n.old_scores / n.scores);
    std::snprintf(s2, sizeof s2, "%.2fx", n.old_soft / n.soft);
    std::snprintf(s3, sizeof s3, "%.2fx", n.old_apply / n.apply);
    std::printf("%7u %6u %11.4f %11.4f %6u %11.4f  %7s %7s %7s  %10.2e %10.2e "
                "%10.2e %9.6f\n",
                n.span, n.nparts, n.scores, n.soft, n.split, n.apply, s1, s2, s3,
                n.sc_maxrel, n.sm_maxabs, n.ap_maxrel, n.sm_rowsum);
  }

  std::printf("\nPer GLOBAL layer of a token there is one of each of the three "
              "above.\nGemma 3 1B has 4 global layers, so multiply the TOTAL "
              "column by 4\nto get the per-token cost that grows with "
              "position.\n");
  return 0;
}
