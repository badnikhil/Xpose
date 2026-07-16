# Kernel fixtures

OpenCL C sources (`*.cl`) and the clspv-compiled SPIR-V binaries (`*.spv`)
used by the Program/Launch tests. **The `.spv` binaries are committed** so the
test suite never needs a network connection or a local clspv build.

| Fixture | Signature | Exercises |
|---|---|---|
| `saxpy` | `(global float* y, global const float* x, float a, uint n)` | float math, 2 buffers + 2 PODs |
| `vec_add` | `(global int* out, global const int* a, global const int* b, uint n)` | int math, 3 buffers + 1 POD |
| `scale_inplace` | `(global float* data, float factor, uint n)` | single buffer read+write in place |
| `compare_fp` | `(global int* eq_out, global int* lt_out, global const float* a, global const float* b, uint n)` | per-element `==` / `<` as int results; IEEE edge cases (-0.0==+0.0, NaN!=NaN, -2.0<-1.0) |

Every kernel bound-checks against `n`, so launches may round the grid up to
whole workgroups (CUDA-style) safely.

## How the binaries were produced (regenerable via `./regenerate.py`)

No local clspv build exists yet, so compilation uses **Compiler Explorer's
hosted clspv**:

- Endpoint: `POST https://godbolt.org/api/compiler/clspv/compile`
  (compiler id **`clspv`**, listed name "clspv (trunk)", language `openclc`;
  id verified against `GET /api/compilers/openclc` on 2026-07-16).
- Options (`options.userArguments`):
  `-cl-std=CL3.0 -inline-entry-points -pod-pushconstant -uniform-workgroup-size -spv-version=1.3`
- All output filters disabled in the request — godbolt returns SPIR-V
  **disassembly text**, which is reassembled with `spirv-as` and validated
  with `spirv-val --target-env vulkan1.1` (SPIRV-Tools v2025.5, clspv's
  pinned commit `fb747184`, built at
  `third_party/spirv-tools-build/`; see `regenerate.py --spirv-tools-bin`).

### Why these flags (the ABI contract the Program module relies on)

- `-pod-pushconstant`: every POD argument lands in **one push-constant
  block**, reflected as `ArgumentPodPushConstant(kernel, ordinal, offset,
  size)`. No POD argument buffer, no UBO.
- `-uniform-workgroup-size`: **required** in combination with CL3.0 +
  `-pod-pushconstant`. Without it clspv emits an implicit
  `PushConstantRegionOffset` block (non-uniform work-group support) as a
  *second* `PushConstant` interface variable, which fails Vulkan validation
  (VUID-StandaloneSpirv-OpEntryPoint-06674: max one push-constant block per
  entry point). `vulkore::launch` always dispatches whole workgroups, so the
  uniform-size promise holds.
- `-spv-version=1.3`: SPIR-V 1.3 / Vulkan 1.1 — the Android-floor target the
  library initializes at.

### Reflection scheme observed in these fixtures (NonSemantic.ClspvReflection.5)

- `Kernel %fn %name numArgs flags attributes`
- buffer args: `ArgumentStorageBuffer %kernel ordinal set binding [info]` —
  all at **descriptor set 0**, binding = order of buffer args (0,1,2,...).
- POD args: `ArgumentPodPushConstant %kernel ordinal offset size [info]` —
  tightly packed from **offset 0** (4-byte scalars at 0, 4, ...).
- `SpecConstantWorkgroupSize 0 1 2` — workgroup size via SpecIds 0/1/2
  (default 1,1,1; the runtime specializes them at pipeline creation).
- No implicit push constants, no literal samplers, no images.
