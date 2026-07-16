# Xpose — agent notes

C++20 Vulkan compute runtime with CUDA-like ergonomics. Kernels come from
clspv-compiled SPIR-V. Full design docs + decision log live in
`../agent-docs/` (read `00-INDEX.md` first; `xpose-core.md` covers this repo).

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

## Layout
- `include/xpose/` public headers (`xpose.hpp` umbrella), `src/` impl, `tests/` googletest.
- Device-level Vulkan calls go through `Context::table()` (per-context `VolkDeviceTable`),
  never global function pointers — multiple Contexts on different drivers must coexist.
- Error strategy: exceptions (`xpose::Error`), every Vk call wrapped in `XP_CHECK`.
- Program/Kernel/launch modules are NOT built yet — seams documented in
  `context.hpp` TODOs and `../agent-docs/xpose-core.md`.
