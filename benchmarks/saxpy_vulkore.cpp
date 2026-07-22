// saxpy_vulkore.cpp — SAXPY (y = a*x + y) on the GPU via the Vulkore runtime.
//
// The entire GPU story — device init, buffer allocation, transfers with
// correct cache management on non-coherent (Adreno/Mali) memory, SPIR-V
// reflection, pipeline creation, descriptor sets, barriers, submission,
// fencing — is the ~10 lines marked CORE below. Compare with
// saxpy_raw_vulkan.cpp, which does the identical job against the raw API.
//
// Benchmark protocol (identical in both programs):
//   correctness: n = 1M floats, verified element-wise vs a CPU reference
//   sync latency: SYNC_ITERS x (launch + wait), mean us/launch
//   throughput:  ROUNDS x (BATCH launches in flight, then wait all)
#include <vulkore/vulkore.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace {
uint32_t kN = 1u << 20;  // element count; override with argv[2]
constexpr float kA = 2.5f;
constexpr int kSyncIters = 200;
constexpr int kBatch = 64;
constexpr int kRounds = 8;

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}
}  // namespace

int main(int argc, char** argv) {
  const std::string spv = argc > 1 ? argv[1] : "tests/kernels/saxpy.spv";
  if (argc > 2) kN = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));

  std::vector<float> x(kN), y(kN);
  for (uint32_t i = 0; i < kN; ++i) {
    x[i] = 0.25f * static_cast<float>(i % 1000) - 100.0f;
    y[i] = 1.5f * static_cast<float>(i % 97);
  }

  // ---- CORE: everything GPU-related happens in these lines ----------------
  vulkore::Context ctx;
  vulkore::Program prog = vulkore::Program::from_file(ctx, spv);
  vulkore::Kernel saxpy = prog.kernel("saxpy");

  vulkore::Buffer xb = ctx.alloc<float>(kN);
  vulkore::Buffer yb = ctx.alloc<float>(kN);
  xb.upload(std::span<const float>(x));
  yb.upload(std::span<const float>(y));

  vulkore::launch(saxpy, {kN}, yb, xb, kA, kN).wait();

  std::vector<float> out(kN);
  yb.download(std::span<float>(out));
  // ---- end CORE ------------------------------------------------------------

  std::printf("device: %s\n", ctx.device_name().c_str());
  for (uint32_t i = 0; i < kN; ++i) {
    const float expected = kA * x[i] + y[i];
    if (out[i] != expected) {
      std::fprintf(stderr, "MISMATCH at %u: got %f want %f\n", i, out[i],
                   expected);
      return 1;
    }
  }
  std::printf("correctness: OK (%u elements)\n", kN);

  for (int i = 0; i < 50; ++i)  // warmup
    vulkore::launch(saxpy, {kN}, yb, xb, kA, kN).wait();

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kSyncIters; ++i)
    vulkore::launch(saxpy, {kN}, yb, xb, kA, kN).wait();
  std::printf("sync latency: %.1f us/launch (%d iters)\n",
              1000.0 * ms_since(t0) / kSyncIters, kSyncIters);

  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < kRounds; ++r) {
    std::vector<vulkore::Fence> fences;
    fences.reserve(kBatch);
    for (int i = 0; i < kBatch; ++i)
      fences.push_back(vulkore::launch(saxpy, {kN}, yb, xb, kA, kN));
    for (auto& f : fences) f.wait();
  }
  std::printf("throughput: %.1f us/launch (%d batches of %d)\n",
              1000.0 * ms_since(t0) / (kRounds * kBatch), kRounds, kBatch);
  return 0;
}
