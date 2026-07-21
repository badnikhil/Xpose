// TPS ceiling benchmark: run every matvec a Gemma 3 1B decode step performs,
// at the real shapes, and report achieved bandwidth + implied tokens/sec.
#include <vulkore/vulkore.hpp>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

struct Shape { const char* name; uint32_t rows, cols, count; };

// Gemma 3 1B: 26 layers, hidden 1152, ffn 6912, 4 heads x 256, 1 kv head,
// vocab ~262144. Per-token matvecs:
static const Shape kShapes[] = {
    {"qkv",     1152,  1536, 26},
    {"o_proj",  1024,  1152, 26},
    {"gate",    1152,  6912, 26},
    {"up",      1152,  6912, 26},
    {"down",    6912,  1152, 26},
    {"lm_head", 1152, 262144, 1},
};

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    vulkore::Context ctx;
    auto prog = vulkore::Program::from_file(ctx, argv[1]);
    auto q4  = prog.kernel("matvec_q4");
    auto q4s = prog.kernel("matvec_q4_split");
    auto red = prog.kernel("matvec_reduce");
    // Narrow matrices need the k axis split to get enough threads; wide ones
    // already saturate, so splitting only adds a reduce pass.
    // Split-k is DISABLED. It should have helped - narrow matrices are starved
    // of threads - but it made things 40% worse (17.3 -> 10.5 tok/s) because it
    // doubles the dispatch count, and dispatch is what actually costs.
    // Measured: a trivial kernel costs 0.31 ms per launch on Adreno 840, so 130
    // dispatches per token is 40 ms of pure overhead before any work happens.
    // Re-enable only after launch batching lands.
    auto split_for = [](uint32_t) -> uint32_t { return 1; };
    std::printf("device: %s\n\n", ctx.device_name().c_str());

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1.f, 1.f);

    double total_ms = 0;
    uint64_t total_bytes = 0, total_params = 0;

    std::printf("%-9s %6s %8s %5s %10s %9s %9s\n",
                "matvec","rows","cols","xN","weights MB","ms/token","GB/s");
    for (const auto& s : kShapes) {
        size_t nw = size_t(s.rows) * s.cols;
        size_t nbytes = nw / 2;                       // int4
        size_t nblk = s.rows / 32;

        std::vector<float> hin(s.rows);
        for (auto& v : hin) v = u(rng);
        std::vector<uint32_t> hw(nbytes / 4, 0x71717171u);
        std::vector<float> hs(nblk * s.cols, 0.02f);

        auto in    = ctx.alloc<float>(s.rows);
        auto w     = ctx.alloc<uint32_t>(nbytes / 4);
        auto sc    = ctx.alloc<float>(hs.size());
        auto out   = ctx.alloc<float>(s.cols);
        uint32_t SP = split_for(s.cols);
        auto part  = ctx.alloc<float>(size_t(s.cols) * SP);
        in.upload(std::span<const float>(hin));
        w.upload(std::span<const uint32_t>(hw));
        sc.upload(std::span<const float>(hs));

        // Submit the whole per-token group and wait ONCE. Waiting per launch
        // measures round-trip submit latency, not bandwidth: with 26 tiny
        // dispatches per shape that dominates completely and made the phone
        // look slower than the laptop, which is backwards for LPDDR5X.
        // Fence discipline per design §5.3 — hold them all, wait on the last.
        {
            std::vector<vulkore::Fence> warm;
            for (uint32_t i = 0; i < s.count; ++i) {
                if (SP == 1) {
                    warm.push_back(vulkore::launch(q4, {s.cols}, out, in, w, sc, s.rows, s.cols));
                } else {
                    warm.push_back(vulkore::launch(q4s, {s.cols * SP}, part, in, w, sc,
                                                 s.rows, s.cols, SP));
                    warm.push_back(vulkore::launch(red, {s.cols}, out, part, s.cols, SP));
                }
            }
            warm.back().wait();
        }

        // ONE command buffer, ONE submit for the whole per-token group.
        // Individual launch() calls cost 0.31 ms each in submission alone.
        const int R = 5;
        double t0 = now_ms();
        for (int r = 0; r < R; ++r) {
            vulkore::Batch b(ctx);
            for (uint32_t i = 0; i < s.count; ++i)
                b.add(q4, {s.cols}, out, in, w, sc, s.rows, s.cols);
            b.submit().wait();
        }
        double per_token_ms = (now_ms() - t0) / R;
        double gbs = (double(nbytes) * s.count) / (per_token_ms * 1e6);
        total_ms += per_token_ms;
        total_bytes += uint64_t(nbytes) * s.count;
        total_params += uint64_t(nw) * s.count;

        std::printf("%-9s %6u %8u %5u %10.1f %9.2f %9.1f  x%u\n",
                    s.name, s.rows, s.cols, s.count,
                    double(nbytes) * s.count / 1e6, per_token_ms, gbs, SP);
    }

    std::printf("\n  params/token : %.2f B\n", total_params / 1e9);
    std::printf("  weights/token: %.0f MB (int4)\n", total_bytes / 1e6);
    std::printf("  matvec time  : %.1f ms/token\n", total_ms);
    std::printf("  bandwidth    : %.1f GB/s\n", total_bytes / (total_ms * 1e6));
    std::printf("  ==> CEILING  : %.1f tokens/sec  (matvec only)\n", 1000.0 / total_ms);
    return 0;
}
