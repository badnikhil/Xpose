# Xpose — agent notes

C++20 Vulkan compute runtime with CUDA-like ergonomics. Kernels come from
clspv-compiled SPIR-V. Full design docs + decision log live in
`../agent-docs/` (read `00-INDEX.md` first; `xpose-core.md` and
`xpose-program-launch.md` cover this repo).

## Rules
- LOCAL repo only: **never add a remote, never push.**
- Never touch `../third_party/` git state; `../third_party/clspv/build` belongs to another agent.
- `../agent-docs/` is local-only knowledge base — never commit it here.

## Build & test
```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # gcc default; clang also supported
cmake --build build
./build/tests/xpose_tests                                  # default device (first discrete GPU)
for d in llvmpipe RENOIR NVIDIA; do XPOSE_DEVICE=$d ./build/tests/xpose_tests; done  # full matrix
```
- Third-party deps are sibling checkouts (`../third_party/`); override with `-DXPOSE_THIRD_PARTY_DIR=`.
- No system Vulkan SDK on this machine: headers are vendored, volk dlopens `libvulkan.so.1`.

## Kernels & fixtures
- Test fixtures: `tests/kernels/*.cl` + committed `.spv` (compiled by
  **godbolt's hosted clspv** — no local clspv build yet). `tests/kernels/README.md`
  records the exact compiler id/flags; `regenerate.py` re-derives the binaries
  (needs `../third_party/spirv-tools-build/tools/`).
- The supported kernel ABI is fixed by those flags: storage-buffer args +
  `-pod-pushconstant` PODs + `-uniform-workgroup-size`. Program throws on
  anything else (images/samplers/UBO-PODs/implicit push constants...).
- Tests find fixtures via the `XPOSE_KERNEL_DIR` compile definition
  (absolute path — binary is not relocatable; adjust for Android pushes).

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
