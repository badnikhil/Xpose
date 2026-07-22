# bench — GPU vs CPU benchmark

Five kernels spanning distinct performance regimes, dispatched through
`vulkore::launch()` and checked element-by-element against a CPU reference
computing the same result.

| Kernel | Regime | Exercises |
|---|---|---|
| `saxpy` | Memory bound | `y = a*x + y`, 2 reads + 1 write per 2 flops |
| `matmul` | Compute bound | Naive 512×512 SGEMM on a 2-D grid, O(n³) |
| `mandelbrot` | Float ALU | Divergent trip counts, almost no memory traffic |
| `blur3` | Bandwidth + branch | Neighbour reads, edge branches |
| `hash_rounds` | Integer ALU | 32 bit-mixing rounds, exact result |

## Build & run

The kernels are compiled ahead of time, same ABI as `tests/kernels`:

```sh
third_party/clspv/build/bin/clspv \
  -cl-std=CL3.0 -inline-entry-points -pod-pushconstant \
  -uniform-workgroup-size -spv-version=1.3 \
  bench/kernels/bench.cl -o bench/kernels/bench.spv
```

The harness links the static library from the normal build (`cmake --build build`):

```sh
g++ -std=c++20 -O3 -march=native \
  -DVMA_STATIC_VULKAN_FUNCTIONS=0 -DVMA_DYNAMIC_VULKAN_FUNCTIONS=1 \
  -DVULKORE_BENCH_KERNEL_DIR='"'$PWD'/bench/kernels"' \
  -Iinclude -Ithird_party/Vulkan-Headers/include -Ithird_party/volk \
  -Ithird_party/VulkanMemoryAllocator/include \
  bench/bench.cpp build/libvulkore.a -ldl -lpthread -o bench/vulkore_bench

./bench/vulkore_bench > run.json        # JSON on stdout, summary table on stderr
VULKORE_DEVICE=llvmpipe ./bench/vulkore_bench   # same binary, CPU rasterizer
```

`bench/vulkore_bench` is a build output and is gitignored.

## Reading the numbers

- GPU timings are best-of-7, CPU best-of-3, after warmup.
- **`saxpy`'s GPU timing includes a host upload on every iteration**; the other
  four time the dispatch only. Its GPU number is therefore pessimistic.
- Bars inside one card share that card's scale and are not comparable across
  cards; the speedup column is the cross-kernel comparison.
- `mandelbrot` disagrees with the CPU on ~0.1% of pixels. Boundary points are
  chaotically sensitive, so a last-bit difference in one multiply changes the
  escape iteration count. Expected for the algorithm, not a runtime defect —
  the integer kernels are bit-exact and the other float kernels agree to ~1 ULP.

## Grid-size ceiling

Sizes are capped at 4,000,000 threads: Adreno reports
`maxComputeWorkGroupCount[0] = 65535`, which at the default 64×1×1 workgroup
gives 4,194,240 threads per dispatch. Larger grids throw from `launch()`.
