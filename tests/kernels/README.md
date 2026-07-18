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
| `transpose_2d` | `(global float* out, global const float* in, uint width, uint height)` | genuine 2-D kernel using BOTH `get_global_id(0)` and `get_global_id(1)`; drives `launch()`'s `grid.y > 1` with real data + `grid.x` round-up + 2-D bound checks |
| `reqd_wgsize` | `__attribute__((reqd_work_group_size(32,1,1)))` `(global int* out, global const int* in, uint n)` | emits `PropertyRequiredWorkgroupSize` (24) -> `has_reqd_workgroup_size` + baked 32×1×1 local size (not the default 64×1×1) |
| `two_kernels` | `add_one(global int* data, uint n)` **and** `mul_two(global int* data, uint n)` | one source, TWO entry points -> multi-kernel module: `kernel_names()` order, distinct pipelines, interleaved dispatch from one `Program` |
| `many_pod` | `(global float* out, float f, int i, uint u, float g, uint n)` | 5 MIXED-type scalar PODs -> a 20-byte push-constant block, offsets 0/4/8/12/16 (packing beyond the 2-scalar shapes) |

Every kernel bound-checks against `n` (or `width`/`height`), so launches may
round the grid up to whole workgroups (CUDA-style) safely.

### Per-kernel notes / flag deviations

- **No flag deviation was needed for any of these fixtures** — all four compile
  under the SAME standard flag set below.
- `reqd_wgsize`: with the standard flags clspv **does** emit
  `PropertyRequiredWorkgroupSize %kernel 32 1 1` in the reflection AND a
  hard-coded `OpExecutionMode %entry LocalSize 32 1 1` — and it **omits**
  `SpecConstantWorkgroupSize` for this kernel. So the local size comes from the
  execution mode, not spec constants; `Program` sets `local_size = {32,1,1}`
  from the reflection and skips spec specialization (its `has_wg_spec_ids` is
  false here), which agrees with the baked execution mode. Confirmed with
  `spirv-dis` (`-uniform-workgroup-size` does NOT suppress the property).
- `transpose_2d`, `two_kernels`, `many_pod`: default workgroup size via
  `SpecConstantWorkgroupSize 0 1 2` (SpecIds 0/1/2), like the original four.

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
  pinned commit `fb747184` from `third_party/clspv/third_party/SPIRV-Tools`,
  built out-of-tree into the repo's gitignored
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
