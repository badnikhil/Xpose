# Xpose — agent notes

C++20 Vulkan compute runtime with CUDA-like ergonomics. Kernels come from
clspv-compiled SPIR-V. Full design docs + decision log live in
`../agent-docs/` (read `00-INDEX.md` first; `xpose-core.md` and
`xpose-program-launch.md` cover this repo).

## Rules
- Submodule remotes point at their public upstreams — never push those.
- Don't move submodule pins (`third_party/*` SHAs) without recording why here
  and in `../agent-docs/environment.md`.
- `../agent-docs/` is local-only knowledge base — never commit it here.
- **Found a failing edge case (especially on real hardware)? Fix the code, keep/add
  a regression test, AND document the lesson here** — don't just patch and move on.
  A desktop-only green run is NOT proof of device correctness (see coherency note).

## Third-party layout (git submodules at `third_party/`)
- After a fresh clone: `git submodule update --init --recursive`.
- Submodules: `Vulkan-Headers`, `volk`, `VulkanMemoryAllocator`, `googletest`
  (build/test deps); `SPIRV-Tools` (reference clone); `clspv` (kernel
  compiler); `clvk`, `kompute` (reference reading only, not built).
- **clspv's own deps are NOT submodules of clspv** — they're fetched clones:
  run `python3 utils/fetch_sources.py --shallow` inside `third_party/clspv`
  to populate `third_party/clspv/third_party/{llvm,SPIRV-Headers,SPIRV-Tools}`
  (~2.9 GB). `.gitmodules` sets `ignore = untracked` for clspv accordingly.
- `third_party/spirv-tools-build/` is a **gitignored out-of-tree build** of
  clspv's pinned SPIRV-Tools providing `tools/spirv-{as,dis,val}` for the
  fixture pipeline. Rebuild:
  ```sh
  cmake -G Ninja -S third_party/clspv/third_party/SPIRV-Tools \
        -B third_party/spirv-tools-build -DCMAKE_BUILD_TYPE=Release \
        -DSPIRV-Headers_SOURCE_DIR=$PWD/third_party/clspv/third_party/SPIRV-Headers \
        -DSPIRV_SKIP_TESTS=ON
  ninja -C third_party/spirv-tools-build -j6 spirv-as spirv-dis spirv-val
  ```
  (The top-level `SPIRV-Tools` submodule does NOT build against clspv's
  SPIRV-Headers pin — always use the clspv-pinned combo above.)

## Build & test
```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # gcc default; clang also supported
cmake --build build
./build/tests/xpose_tests                                  # default device (first discrete GPU)
for d in llvmpipe RENOIR NVIDIA; do XPOSE_DEVICE=$d ./build/tests/xpose_tests; done  # full matrix
```
- Third-party deps resolve to `third_party/` in-repo; override with `-DXPOSE_THIRD_PARTY_DIR=`.
- No system Vulkan SDK on this machine: headers are vendored, volk dlopens `libvulkan.so.1`.
- **On-device (Android):** `scripts/run-android.sh` cross-builds arm64, pushes the
  test binary + `tests/kernels/*.spv`, and runs the suite on a connected phone over
  adb (`--build` to recompile, `-s <serial>` to select a device). This is the ONLY
  way to exercise the non-coherent-memory path — run it before trusting buffer
  transfers. Runner details in `../agent-docs/android-device-testing.md`.

## Kernels & fixtures
- Test fixtures: `tests/kernels/*.cl` + committed `.spv` (compiled by
  **godbolt's hosted clspv** — no local clspv build yet). `tests/kernels/README.md`
  records the exact compiler id/flags; `regenerate.py` re-derives the binaries
  (needs `third_party/spirv-tools-build/tools/`, see layout section above).
- The supported kernel ABI is fixed by those flags: storage-buffer args +
  `-pod-pushconstant` PODs + `-uniform-workgroup-size`. Program throws on
  anything else (images/samplers/UBO-PODs/implicit push constants...).
- Tests find fixtures via the `XPOSE_KERNEL_DIR` compile definition
  (absolute path — binary is not relocatable; adjust for Android pushes).

## Device portability & memory coherency (READ before touching Buffer/transfers)
- **Desktop-green != device-green.** Every desktop driver (RADV/NVIDIA/llvmpipe)
  hands back HOST_VISIBLE **+ HOST_COHERENT** memory, so the non-coherent path is
  NEVER exercised on desktop. Android GPUs (Mali-G57 verified) expose
  HOST_VISIBLE | **HOST_CACHED (non-coherent)** memory — a real, correct memory
  type, not a bug (cached = faster CPU access). This bit us on the first real
  device run: 6 buffer-transfer tests passed on all 3 desktop drivers, failed on Mali.
- **Non-coherent = manage the CPU cache, NOT copy data.** UMA devices share one
  physical RAM (no PCIe, no staging copy needed); the only hazard is the CPU cache
  being out of sync with RAM. After a host WRITE to mapped memory ->
  `vmaFlushAllocation` (push cache to RAM so the GPU sees it); before a host READ of
  GPU output -> `vmaInvalidateAllocation` (drop stale cache, re-read RAM). Both are
  VMA no-ops on coherent memory, so they cost nothing on desktop.
- **Buffer transfer strategy** (`src/buffer.cpp`) is keyed on *mappability*, not
  coherency: direct map+memcpy (+ flush/invalidate when `!coherent_`) whenever the
  allocation is HOST_VISIBLE; the staging-copy path is reserved for truly
  non-host-visible device-local memory (discrete GPUs). `Buffer::host_visible()`
  means "directly mappable" — true for any HOST_VISIBLE allocation, coherent or not.
- **VMA flush/invalidate require the range be currently host-mapped**
  (VUID-VkMappedMemoryRange-memory-00684): map first, THEN flush/invalidate while
  mapped. Getting the order wrong null-derefs inside the Mali driver; desktop hides
  it because invalidate is a no-op on coherent memory. Full write-up:
  `../agent-docs/mali-coherency-fix.md`.

## Layout
- `include/xpose/` public headers (`xpose.hpp` umbrella), `src/` impl, `tests/` googletest.
- Modules: Context/Buffer/Sync (core) + Program (SPIR-V load, ClspvReflection
  parse, pipelines) + launch() (variadic dispatch, returns Fence).
- Device-level Vulkan calls go through `Context::table()` (per-context `VolkDeviceTable`),
  never global function pointers — multiple Contexts on different drivers must coexist.
- Error strategy: exceptions (`xpose::Error`), every Vk call wrapped in `XP_CHECK`.
- launch() recycles command buffers/descriptor sets via the Fence completion
  hook (`Context::submit(cb, on_complete)`); descriptor pools grow on demand.
- Grid = GLOBAL thread counts (CUDA-style), rounded up to whole workgroups;
  default workgroup size 64x1x1 via spec constants — kernels must bound-check.
