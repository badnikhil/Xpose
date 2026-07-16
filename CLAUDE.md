# Vulkore — agent notes

C++20 Vulkan compute runtime with CUDA-like ergonomics. Kernels come from
clspv-compiled SPIR-V. Full design docs + decision log live in
`../agent-docs/` (read `00-INDEX.md` first; `vulkore-core.md` and
`vulkore-program-launch.md` cover this repo).

## Rules
- Submodule remotes point at their public upstreams — never push those.
- Don't move submodule pins (`third_party/*` SHAs) without recording why here
  and in `../agent-docs/environment.md`.
- `../agent-docs/` is local-only knowledge base — never commit it here.

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
./build/tests/vulkore_tests                                  # default device (first discrete GPU)
for d in llvmpipe RENOIR NVIDIA; do VULKORE_DEVICE=$d ./build/tests/vulkore_tests; done  # full matrix
```
- Third-party deps resolve to `third_party/` in-repo; override with `-DVULKORE_THIRD_PARTY_DIR=`.
- No system Vulkan SDK on this machine: headers are vendored, volk dlopens `libvulkan.so.1`.

## Kernels & fixtures
- Test fixtures: `tests/kernels/*.cl` + committed `.spv` (compiled by
  **godbolt's hosted clspv** — no local clspv build yet). `tests/kernels/README.md`
  records the exact compiler id/flags; `regenerate.py` re-derives the binaries
  (needs `third_party/spirv-tools-build/tools/`, see layout section above).
- The supported kernel ABI is fixed by those flags: storage-buffer args +
  `-pod-pushconstant` PODs + `-uniform-workgroup-size`. Program throws on
  anything else (images/samplers/UBO-PODs/implicit push constants...).
- Tests find fixtures via the `VULKORE_KERNEL_DIR` compile definition
  (absolute path — binary is not relocatable; adjust for Android pushes).

## Layout
- `include/vulkore/` public headers (`vulkore.hpp` umbrella), `src/` impl, `tests/` googletest.
- Modules: Context/Buffer/Sync (core) + Program (SPIR-V load, ClspvReflection
  parse, pipelines) + launch() (variadic dispatch, returns Fence).
- Device-level Vulkan calls go through `Context::table()` (per-context `VolkDeviceTable`),
  never global function pointers — multiple Contexts on different drivers must coexist.
- Error strategy: exceptions (`vulkore::Error`), every Vk call wrapped in `XP_CHECK`.
- launch() recycles command buffers/descriptor sets via the Fence completion
  hook (`Context::submit(cb, on_complete)`); descriptor pools grow on demand.
- Grid = GLOBAL thread counts (CUDA-style), rounded up to whole workgroups;
  default workgroup size 64x1x1 via spec constants — kernels must bound-check.
