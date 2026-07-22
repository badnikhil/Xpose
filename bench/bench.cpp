// Vulkore GPU-vs-CPU benchmark harness.
//
// Runs five kernels spanning distinct performance regimes on the Vulkan device
// via vulkore::launch(), against scalar single-threaded and multi-threaded CPU
// baselines. Verifies every GPU result against the CPU reference and emits JSON
// on stdout (human summary on stderr).
#include <vulkore/vulkore.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Timing {
  double best = 0, median = 0;
};

// Runs fn warmup+iters times; returns best and median wall time.
Timing time_it(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = Clock::now();
    fn();
    samples.push_back(ms_since(t0));
  }
  std::sort(samples.begin(), samples.end());
  Timing t;
  t.best = samples.front();
  t.median = samples[samples.size() / 2];
  return t;
}

unsigned cpu_threads() {
  unsigned n = std::thread::hardware_concurrency();
  return n ? n : 1;
}

// Splits [0,n) across all cores.
void parallel_for(size_t n, const std::function<void(size_t, size_t)>& body) {
  const unsigned T = cpu_threads();
  const size_t chunk = (n + T - 1) / T;
  std::vector<std::thread> pool;
  pool.reserve(T);
  for (unsigned t = 0; t < T; ++t) {
    const size_t lo = std::min(n, size_t(t) * chunk);
    const size_t hi = std::min(n, lo + chunk);
    if (lo < hi) pool.emplace_back([&body, lo, hi] { body(lo, hi); });
  }
  for (auto& th : pool) th.join();
}

// ---- accuracy ------------------------------------------------------------
struct Accuracy {
  double max_abs = 0;
  double max_rel = 0;
  size_t mismatches = 0;  // exact-match failures (integer kernels)
  bool exact = false;     // true when integer/exact comparison was used
};

Accuracy compare_f32(const std::vector<float>& gpu, const std::vector<float>& cpu) {
  Accuracy a;
  for (size_t i = 0; i < gpu.size(); ++i) {
    const double g = gpu[i], c = cpu[i];
    const double abs_err = std::fabs(g - c);
    a.max_abs = std::max(a.max_abs, abs_err);
    const double denom = std::max(1e-30, std::fabs(c));
    a.max_rel = std::max(a.max_rel, abs_err / denom);
    if (g != c) ++a.mismatches;
  }
  return a;
}

template <typename T>
Accuracy compare_exact(const std::vector<T>& gpu, const std::vector<T>& cpu) {
  Accuracy a;
  a.exact = true;
  for (size_t i = 0; i < gpu.size(); ++i) {
    if (gpu[i] != cpu[i]) {
      ++a.mismatches;
      a.max_abs = std::max(a.max_abs, std::fabs(double(gpu[i]) - double(cpu[i])));
    }
  }
  return a;
}

// ---- JSON helpers --------------------------------------------------------
std::string j_num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}

template <typename T>
std::string j_samples(const std::vector<T>& v, size_t k) {
  std::string s = "[";
  for (size_t i = 0; i < std::min(k, v.size()); ++i) {
    if (i) s += ", ";
    s += j_num(double(v[i]));
  }
  return s + "]";
}

struct Result {
  std::string name, kind, note, unit;
  size_t elements = 0;
  double work = 0;  // ops (flops or int-ops) for throughput
  Timing gpu, cpu1, cpuN;
  Accuracy acc;
  std::string in_samples, gpu_samples, cpu_samples;
};

std::vector<Result> g_results;

void emit(const Result& r) { g_results.push_back(r); }

}  // namespace

int main() {
  vulkore::Context ctx;
  const std::string dev = ctx.device_name();
  std::fprintf(stderr, "device: %s   cpu threads: %u\n", dev.c_str(), cpu_threads());

  vulkore::Program prog = vulkore::Program::from_file(
      ctx, std::string(VULKORE_BENCH_KERNEL_DIR) + "/bench.spv");

  constexpr int kWarmup = 3;
  constexpr int kIters = 7;

  // ======================= 1. SAXPY (memory bound) =======================
  {
    const uint32_t n = 4000000;  // 4M (Adreno caps at 65535 workgroups/axis)
    const float a = 2.5f;
    std::vector<float> x(n), y0(n), gpu_out(n), cpu_out(n);
    for (uint32_t i = 0; i < n; ++i) {
      x[i] = float(i % 1000) * 0.001f;
      y0[i] = float(i % 777) * 0.01f;
    }
    vulkore::Kernel k = prog.kernel("saxpy");
    vulkore::Buffer bx = ctx.alloc<float>(n), by = ctx.alloc<float>(n);
    bx.upload(std::span<const float>(x));

    auto gpu_run = [&] {
      by.upload(std::span<const float>(y0));
      vulkore::launch(k, {n}, by, bx, a, n).wait();
    };
    Timing gt = time_it(gpu_run, kWarmup, kIters);
    by.download(std::span<float>(gpu_out));

    auto cpu_body = [&](size_t lo, size_t hi) {
      for (size_t i = lo; i < hi; ++i) cpu_out[i] = a * x[i] + y0[i];
    };
    Timing c1 = time_it([&] { cpu_body(0, n); }, 1, 3);
    Timing cN = time_it([&] { parallel_for(n, cpu_body); }, 1, 3);

    Result r;
    r.name = "SAXPY";
    r.kind = "Memory bound";
    r.note = "y = a*x + y over 4M floats — 2 reads + 1 write per 2 flops";
    r.unit = "GB/s";
    r.elements = n;
    r.work = double(n) * 3.0 * sizeof(float);  // bytes moved
    r.gpu = gt; r.cpu1 = c1; r.cpuN = cN;
    r.acc = compare_f32(gpu_out, cpu_out);
    r.in_samples = j_samples(x, 6);
    r.gpu_samples = j_samples(gpu_out, 6);
    r.cpu_samples = j_samples(cpu_out, 6);
    emit(r);
    std::fprintf(stderr, "saxpy done\n");
  }

  // ======================= 2. Matrix multiply ============================
  {
    const uint32_t n = 512;
    const size_t sz = size_t(n) * n;
    std::vector<float> A(sz), B(sz), gpu_out(sz), cpu_out(sz);
    for (size_t i = 0; i < sz; ++i) {
      A[i] = float((i * 31u) % 100) * 0.01f;
      B[i] = float((i * 17u) % 100) * 0.01f;
    }
    vulkore::Kernel k = prog.kernel("matmul");
    vulkore::Buffer ba = ctx.alloc<float>(sz), bb = ctx.alloc<float>(sz),
                  bc = ctx.alloc<float>(sz);
    ba.upload(std::span<const float>(A));
    bb.upload(std::span<const float>(B));

    Timing gt = time_it([&] { vulkore::launch(k, {n, n}, bc, ba, bb, n).wait(); },
                        kWarmup, kIters);
    bc.download(std::span<float>(gpu_out));

    auto cpu_body = [&](size_t lo, size_t hi) {
      for (size_t row = lo; row < hi; ++row)
        for (uint32_t col = 0; col < n; ++col) {
          float s = 0.0f;
          for (uint32_t kk = 0; kk < n; ++kk) s += A[row * n + kk] * B[kk * n + col];
          cpu_out[row * n + col] = s;
        }
    };
    Timing c1 = time_it([&] { cpu_body(0, n); }, 0, 3);
    Timing cN = time_it([&] { parallel_for(n, cpu_body); }, 0, 3);

    Result r;
    r.name = "Matrix multiply";
    r.kind = "Compute bound";
    r.note = "Naive 512x512 SGEMM, 2-D grid — O(n^3), no tiling or shared memory";
    r.unit = "GFLOP/s";
    r.elements = sz;
    r.work = 2.0 * double(n) * n * n;
    r.gpu = gt; r.cpu1 = c1; r.cpuN = cN;
    r.acc = compare_f32(gpu_out, cpu_out);
    r.in_samples = j_samples(A, 6);
    r.gpu_samples = j_samples(gpu_out, 6);
    r.cpu_samples = j_samples(cpu_out, 6);
    emit(r);
    std::fprintf(stderr, "matmul done\n");
  }

  // ======================= 3. Mandelbrot =================================
  {
    const uint32_t W = 1024, H = 1024, iters = 256;
    const size_t sz = size_t(W) * H;
    std::vector<int32_t> gpu_out(sz), cpu_out(sz);
    vulkore::Kernel k = prog.kernel("mandelbrot");
    vulkore::Buffer bo = ctx.alloc<int32_t>(sz);

    Timing gt = time_it(
        [&] { vulkore::launch(k, {W, H}, bo, W, H, iters).wait(); }, kWarmup, kIters);
    bo.download(std::span<int32_t>(gpu_out));

    auto cpu_body = [&](size_t lo, size_t hi) {
      for (size_t py = lo; py < hi; ++py)
        for (uint32_t px = 0; px < W; ++px) {
          float cr = -2.0f + 3.0f * float(px) / float(W);
          float ci = -1.5f + 3.0f * float(py) / float(H);
          float zr = 0.0f, zi = 0.0f;
          uint32_t i = 0;
          while (i < iters && (zr * zr + zi * zi) <= 4.0f) {
            float t = zr * zr - zi * zi + cr;
            zi = 2.0f * zr * zi + ci;
            zr = t;
            ++i;
          }
          cpu_out[py * W + px] = int32_t(i);
        }
    };
    Timing c1 = time_it([&] { cpu_body(0, H); }, 0, 3);
    Timing cN = time_it([&] { parallel_for(H, cpu_body); }, 0, 3);

    Result r;
    r.name = "Mandelbrot";
    r.kind = "Float ALU";
    r.note = "1024x1024, 256 max iterations — divergent trip counts, almost no memory traffic";
    r.unit = "Giter/s";
    r.elements = sz;
    {
      double total = 0;
      for (size_t i = 0; i < sz; ++i) total += cpu_out[i];
      r.work = total;
    }
    r.gpu = gt; r.cpu1 = c1; r.cpuN = cN;
    r.acc = compare_exact(gpu_out, cpu_out);
    r.in_samples = "[]";
    r.gpu_samples = j_samples(gpu_out, 6);
    r.cpu_samples = j_samples(cpu_out, 6);
    emit(r);
    std::fprintf(stderr, "mandelbrot done\n");
  }

  // ======================= 4. 3-point stencil ============================
  {
    const uint32_t n = 4000000;
    std::vector<float> in(n), gpu_out(n), cpu_out(n);
    for (uint32_t i = 0; i < n; ++i) in[i] = float(i % 512) * 0.5f;
    vulkore::Kernel k = prog.kernel("blur3");
    vulkore::Buffer bi = ctx.alloc<float>(n), bo = ctx.alloc<float>(n);
    bi.upload(std::span<const float>(in));

    Timing gt = time_it([&] { vulkore::launch(k, {n}, bo, bi, n).wait(); },
                        kWarmup, kIters);
    bo.download(std::span<float>(gpu_out));

    auto cpu_body = [&](size_t lo, size_t hi) {
      for (size_t i = lo; i < hi; ++i) {
        float l = (i > 0) ? in[i - 1] : in[0];
        float c = in[i];
        float rr = (i + 1 < n) ? in[i + 1] : in[n - 1];
        cpu_out[i] = (l + c + rr) * (1.0f / 3.0f);
      }
    };
    Timing c1 = time_it([&] { cpu_body(0, n); }, 1, 3);
    Timing cN = time_it([&] { parallel_for(n, cpu_body); }, 1, 3);

    Result r;
    r.name = "3-point stencil";
    r.kind = "Bandwidth + branch";
    r.note = "Neighbour-averaging blur over 4M floats — unaligned reads, edge branches";
    r.unit = "GB/s";
    r.elements = n;
    r.work = double(n) * 4.0 * sizeof(float);
    r.gpu = gt; r.cpu1 = c1; r.cpuN = cN;
    r.acc = compare_f32(gpu_out, cpu_out);
    r.in_samples = j_samples(in, 6);
    r.gpu_samples = j_samples(gpu_out, 6);
    r.cpu_samples = j_samples(cpu_out, 6);
    emit(r);
    std::fprintf(stderr, "blur3 done\n");
  }

  // ======================= 5. Integer hash ===============================
  {
    const uint32_t n = 4000000;
    const uint32_t rounds = 32;
    std::vector<uint32_t> in(n), gpu_out(n), cpu_out(n);
    for (uint32_t i = 0; i < n; ++i) in[i] = i * 2654435761u + 12345u;
    vulkore::Kernel k = prog.kernel("hash_rounds");
    vulkore::Buffer bi = ctx.alloc<uint32_t>(n), bo = ctx.alloc<uint32_t>(n);
    bi.upload(std::span<const uint32_t>(in));

    Timing gt = time_it([&] { vulkore::launch(k, {n}, bo, bi, rounds, n).wait(); },
                        kWarmup, kIters);
    bo.download(std::span<uint32_t>(gpu_out));

    auto cpu_body = [&](size_t lo, size_t hi) {
      for (size_t i = lo; i < hi; ++i) {
        uint32_t h = in[i];
        for (uint32_t r = 0; r < rounds; ++r) {
          h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
          h *= 0x846ca68bu; h ^= h >> 16;
        }
        cpu_out[i] = h;
      }
    };
    Timing c1 = time_it([&] { cpu_body(0, n); }, 1, 3);
    Timing cN = time_it([&] { parallel_for(n, cpu_body); }, 1, 3);

    Result r;
    r.name = "Integer hash";
    r.kind = "Integer ALU";
    r.note = "32 rounds of a bit-mixing hash over 4M uints — integer units only, exact result";
    r.unit = "Gop/s";
    r.elements = n;
    r.work = double(n) * rounds * 7.0;
    r.gpu = gt; r.cpu1 = c1; r.cpuN = cN;
    r.acc = compare_exact(gpu_out, cpu_out);
    r.in_samples = j_samples(in, 6);
    r.gpu_samples = j_samples(gpu_out, 6);
    r.cpu_samples = j_samples(cpu_out, 6);
    emit(r);
    std::fprintf(stderr, "hash done\n");
  }

  // ---- JSON out ----------------------------------------------------------
  std::printf("{\n  \"device\": \"%s\",\n  \"cpu_threads\": %u,\n  \"benchmarks\": [\n",
              dev.c_str(), cpu_threads());
  for (size_t i = 0; i < g_results.size(); ++i) {
    const Result& r = g_results[i];
    std::printf(
        "    {\"name\": \"%s\", \"kind\": \"%s\", \"note\": \"%s\", \"unit\": \"%s\",\n"
        "     \"elements\": %zu, \"work\": %s,\n"
        "     \"gpu_ms\": %s, \"cpu1_ms\": %s, \"cpuN_ms\": %s,\n"
        "     \"max_abs\": %s, \"max_rel\": %s, \"mismatches\": %zu, \"exact\": %s,\n"
        "     \"in\": %s, \"gpu_vals\": %s, \"cpu_vals\": %s}%s\n",
        r.name.c_str(), r.kind.c_str(), r.note.c_str(), r.unit.c_str(),
        r.elements, j_num(r.work).c_str(), j_num(r.gpu.best).c_str(),
        j_num(r.cpu1.best).c_str(), j_num(r.cpuN.best).c_str(),
        j_num(r.acc.max_abs).c_str(), j_num(r.acc.max_rel).c_str(),
        r.acc.mismatches, r.acc.exact ? "true" : "false", r.in_samples.c_str(),
        r.gpu_samples.c_str(), r.cpu_samples.c_str(),
        i + 1 < g_results.size() ? "," : "");
  }
  std::printf("  ]\n}\n");

  std::fprintf(stderr, "\n%-18s %10s %10s %10s %8s\n", "kernel", "gpu(ms)",
               "cpu1(ms)", "cpuN(ms)", "speedup");
  for (const Result& r : g_results) {
    std::fprintf(stderr, "%-18s %10.2f %10.2f %10.2f %7.1fx\n", r.name.c_str(),
                 r.gpu.best, r.cpu1.best, r.cpuN.best, r.cpu1.best / r.gpu.best);
  }
  return 0;
}
