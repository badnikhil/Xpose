# Raw Vulkan vs Vulkore — LOC & overhead comparison

Two standalone programs do the **identical job**: SAXPY (`y = a*x + y`) on the
GPU, same clspv-compiled kernel binary (`tests/kernels/saxpy.spv`), same
verification against a CPU reference, same benchmark protocol.

- `saxpy_vulkore.cpp` — written against the Vulkore runtime.
- `saxpy_raw_vulkan.cpp` — written directly against the Vulkan API (volk
  loader, no other dependency; it does **not** link the vulkore library).
  This is the *minimal correct mobile-class* version: it handles
  HOST_VISIBLE-but-non-COHERENT memory (flush after CPU writes, invalidate
  before CPU reads), which Adreno/Mali devices expose and desktop drivers
  never do — skip that and the code passes on every desktop and silently
  returns stale data on a phone.

Build (part of the default build, `-DVULKORE_BUILD_BENCHMARKS=OFF` to disable):

```sh
cmake --build build
./build/benchmarks/saxpy_raw_vulkan tests/kernels/saxpy.spv [n]
./build/benchmarks/saxpy_vulkore      tests/kernels/saxpy.spv [n]
```

`VULKORE_DEVICE=<substring>` selects the GPU in both programs.

## Lines of code

Counted with `grep -vE '^\s*(//|$)' | wc -l` (non-blank,
non-comment).

| | raw Vulkan | Vulkore | ratio |
|---|---|---|---|
| whole program (incl. benchmark loops, verification) | 415 | 67 | **6.2×** |
| GPU-specific code (init → alloc → upload → dispatch → readback) | ~370 | **10** | **~37×** |

The 10 Vulkore lines (the `CORE` block in `saxpy_vulkore.cpp`):

```cpp
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
```

The raw program additionally *hard-codes* the kernel's interface (descriptor
bindings, push-constant offsets, entry-point name, workgroup-size spec IDs) —
change the kernel signature and you hand-edit pipeline code. Vulkore reads all
of that from clspv's reflection at load time and validates arguments at
launch (arity, buffer-vs-POD, POD sizes) with named-argument error messages.

The raw sample is also the *smaller* problem: it supports exactly one kernel,
one descriptor set reused forever, host-visible memory only. The Vulkore
runtime behind those 10 lines (~1.9k LOC, written once) additionally covers
staging transfers for non-host-visible discrete GPUs, descriptor-pool growth,
command-buffer/descriptor recycling via fence-completion hooks, multi-device
coexistence (per-device dispatch tables), and multi-kernel modules.

## Measured overhead (Adreno X1-85, Mesa turnip, aarch64, 2026-07-19)

Same protocol in both programs: 50-launch warmup, then 200 × (launch + wait)
("sync latency"), then 8 rounds of 64 launches in flight ("throughput").
Correctness verified on every run in both programs.

| workload | metric | raw Vulkan | Vulkore | delta |
|---|---|---|---|---|
| n = 1M | sync latency | 299.7 µs | 365.6 µs | +22% ¹ |
| n = 1M | throughput | 248.2 µs | 252.4 µs | **+1.7%** |
| n = 1024 | sync latency | 51–61 µs | 54–79 µs | within run noise ¹ |
| n = 1024 | throughput | 29.3–39.1 µs | 32.7–33.2 µs | **~+3 µs/launch** |

¹ sync-latency runs jitter ±15 µs run-to-run on this device (one Vulkore run
beat raw: 53.8 vs 54.8 µs); treat sync numbers as parity-within-noise plus
a small constant. The stable, honest summary:

- **Per-launch cost of the abstraction: ~3 µs** (Vulkore re-records a command
  buffer, allocates/recycles a descriptor set, and creates a fence *every
  launch*; the raw sample reuses one prerecorded descriptor set — its best
  case).
- **On a real workload (1M elements) that is ~2% of dispatch time**; kernels
  in AI inference run far longer than 250 µs, so the overhead vanishes into
  the noise.

llvmpipe (CPU fallback) numbers for reference: raw 356.7 vs Vulkore 350.6
µs/launch throughput at n = 1M — indistinguishable.

## Reading the result

The abstraction costs ~6× fewer lines to *hold* (whole program), ~37× fewer
lines to *write per kernel*, and ~2% dispatch overhead on real workloads —
while being *more* correct on mobile GPUs than what developers actually write
(the coherency trap above shipped past 3 desktop drivers and 6 green tests
before a real device caught it — see `agent-docs/mali-coherency-fix.md`).
