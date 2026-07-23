# vulkore — Program / Launch / Batch / DescriptorCache

## Kernel fixture pipeline

- `tests/kernels/regenerate.py` compiles `.cl` → `.spv` and validates.
  It **prefers the locally built clspv** at `third_party/clspv/build/bin/clspv`
  (no network, reproducible, no rate limit when several agents regenerate at
  once) and falls back to Compiler Explorer's hosted clspv (`--godbolt`
  forces it). Godbolt details: compiler id `clspv`, language id `openclc`
  (plain `/api/compilers/opencl` returns `[]`); POST with **all output filters
  disabled** or the OpString debug section is stripped and spirv-as fails on
  undefined ids; output is disassembly text, reassembled with `spirv-as` and
  validated `spirv-val --target-env vulkan1.1`.
- spirv-as/dis/val come from `third_party/spirv-tools-build/` — clspv's OWN
  SPIRV-Tools pin built against clspv's SPIRV-Headers (`repo-layout.md`). The
  top-level `SPIRV-Tools` submodule does NOT build against that pin; don't try.
- **Flags (the fixed kernel ABI):**
  `-cl-std=CL3.0 -inline-entry-points -pod-pushconstant -uniform-workgroup-size
  -spv-version=1.3`. Gotcha: without `-uniform-workgroup-size`, CL3.0 mode
  emits an implicit `PushConstantRegionOffset` block in ADDITION to the POD
  push block → two PushConstant interface variables → validation failure
  (VUID-StandaloneSpirv-OpEntryPoint-06674). Uniform workgroup size is safe
  because launch() always dispatches whole workgroups.
- Different clspv revisions emit differently sized binaries from identical
  source — a regenerated .spv is a confound bundled with any kernel edit;
  separate the two before claiming bit-exactness.
- Fixtures (committed `.cl` + `.spv` in `tests/kernels/`): saxpy, vec_add,
  scale_inplace, compare_fp (per-element float ==/< as int — asserts BOTH
  outcomes and the IEEE traps: `-0.0==+0.0` true, `NaN!=NaN`, `-2.0<-1.0`
  true, ±inf; bitwise-comparison bugs pass naive tests), transpose_2d,
  reqd_wgsize, two_kernels, many_pod.

## Reflection format (NonSemantic.ClspvReflection.5)

Cross-checked against clvk's parser (`third_party/clvk/src/program.cpp`).
Operand values are ids of `OpConstant %uint` / `OpString`; OpExtInst operands
start at word 5.

| Instruction | Operands (after set+number) |
|---|---|
| `Kernel`(1) | fn-id, name-str, numArgs, flags, attributes-str |
| `ArgumentInfo`(2) | name-str [, type, addr-q, access-q, type-q] |
| `ArgumentStorageBuffer`(3) | kernel-ref, ordinal, set, binding [, arginfo] |
| `ArgumentPodPushConstant`(7) | kernel-ref, ordinal, offset, size [, arginfo] |
| `SpecConstantWorkgroupSize`(12) | xSpecId, ySpecId, zSpecId (module-wide) |
| `PropertyRequiredWorkgroupSize`(24) | kernel-ref, x, y, z |

Scheme with the standard flags: all buffers at descriptor **set 0**, bindings =
buffer-arg order; PODs tightly packed in ONE push-constant block from offset 0;
workgroup size via SpecIds 0/1/2; no implicit push constants.
`reqd_work_group_size` kernels instead get `PropertyRequiredWorkgroupSize` + a
baked `OpExecutionMode LocalSize` and NO spec-constant instruction — emitted
fine under the standard flags; `Program` reads the property, sets
`local_size`, and skips spec specialisation, so `k.local_size()` and the
dispatched size agree.

## Program

`Program(Context&, span<const uint8_t>)` / `Program::from_file`. Own minimal
SPIR-V word-stream parser (no SPIRV-Tools runtime dep). **Strips ALL
non-semantic instructions** before `vkCreateShaderModule` (OpExtension
SPV_KHR_non_semantic_info, NonSemantic.* imports and their OpExtInsts) → no
device extension required; Context enables zero features. `kernel("name")`
returns a cheap `Kernel` handle; pipelines are built lazily and cached, one per
kernel; workgroup size = reqd if present, else default **64×1×1** via spec
constants at pipeline creation.

Unsupported reflection instructions **throw at load naming the feature**:
images, samplers, texel buffers, `ArgumentWorkgroup` (pointer-to-local),
POD-in-UBO/SSBO, `ArgumentUniform`, literal samplers, printf, implicit push
constants (global offset / num workgroups / region offset — kernels compiled
with other flag sets, e.g. clvk's `-global-offset`, are rejected). Malformed
modules (bad magic, byte-swapped, truncated instructions, non-contiguous arg
ordinals, unknown kernel refs, non-constant operands) throw with specific
messages — all covered by `tests/program_error_test.cpp`, which bakes SPIR-V
as raw words in-test (no tool dependency, phone-runnable) and can do so
because `parse_reflection()` runs before module creation, so test modules need
not be Vulkan-valid.

## launch()

`launch(kernel, Grid{gx,gy,gz}, args...)` → `Fence`.

- Grid = **GLOBAL threads** (CUDA semantics), rounded UP:
  `groups = ceil(global/local)`; per-axis `maxComputeWorkGroupCount` checked
  (throws), zero grid throws.
- `Buffer&` → descriptor write at the reflected set/binding (bound with
  `VK_WHOLE_SIZE`); POD → memcpy into the push blob at the reflected offset,
  **size must match exactly** (a `2.0` double literal for a float arg throws
  and the message says so). `static_assert` rejects pointers and non-POD
  types; POD block cap 128 bytes (compile-time — cannot be a runtime test).
- Barriers around a standalone launch: wide pre
  (`ALL_COMMANDS/MEMORY_WRITE → COMPUTE/SHADER_READ|WRITE`) and wide post
  (`COMPUTE/SHADER_WRITE → ALL_COMMANDS|HOST / MEMORY_READ|WRITE`) — covers
  upload→dispatch, dispatch→download (both transfer paths), dispatch→dispatch.
- Recycling: `Context::submit(cb, on_complete)` + the Fence completion hook
  (fires exactly once, from wait/is_signaled/draining-dtor) return the command
  buffer and per-launch descriptor sets; descriptor pools grow on demand
  (128 sets/pool, FREE_DESCRIPTOR_SET_BIT). Sustained launching does not leak.

## Batch and DescriptorCache

Motivated by the LLM decode loop (838 dispatches/token — per-launch submits
would be ~260 ms; numbers in `llm-performance.md`).

```cpp
vulkore::DescriptorCache dc(ctx);       // must be destroyed BEFORE ctx
for (;;) {
  vulkore::Batch b(ctx, dc);            // Batch(ctx) without a cache is unchanged
  ...                                 // b.launch(kernel, grid, args...)
  b.submit().wait();
}
```

`Batch` records any number of dispatches into one command buffer, one
`vkQueueSubmit`, preserving ordering semantics of sequential `launch()` calls.

`DescriptorCache` memoises descriptor sets on `(VkDescriptorSetLayout, the
buffers bound)` — a chain that re-binds the same buffers every iteration
allocates once and hits forever. Cached sets are never updated after creation,
which is what makes binding one set in several in-flight command buffers legal.
`clear()/size()/hits()/misses()` are public. Cached sets are owned by the
cache (not the Fence free-list) and freed in `clear()`/dtor.

**LIFETIME CONTRACT — the sharp edge**: the key holds raw `VkBuffer` handles,
and a driver may reuse a handle for a new buffer after the old one is
destroyed. Destroy the cache before the Context, and call `clear()` after
destroying any Buffer that was ever bound through it.

### Barrier scheme inside a Batch

| position | barrier |
|---|---|
| first dispatch in the cmdbuf | wide pre: `ALL_COMMANDS/MEMORY_WRITE → COMPUTE/SHADER_READ\|WRITE` |
| after each dispatch in the Batch | narrow: `COMPUTE/SHADER_WRITE → COMPUTE/SHADER_READ\|WRITE` |
| appended by `Batch::submit()` | wide post: `COMPUTE/SHADER_WRITE → ALL_COMMANDS\|HOST/MEMORY_READ\|WRITE` |
| standalone `launch()` | unchanged — wide pre + wide post |

Why sufficient: the previous dispatch's post-barrier covers
dispatch→dispatch, and the FIRST dispatch's wide pre-barrier is a full memory
barrier over `ALL_COMMANDS`, covering anything submitted before this command
buffer — including a `Buffer::upload` issued while the Batch was being
recorded (upload submits its own command buffer first). A Batch records only
dispatches, so nothing can consume the writes until `submit()` appends the
wide post-barrier. Not free on the GPU either: narrowing 838 wide barriers was
worth ~1.2 ms/token of GPU time on an Adreno 840. `launch()`'s recorded
command buffer is byte-for-byte what it was before Batch existed.

## Test coverage worth knowing about

- `tests/batch_test.cpp` (9 tests): exact-value checks because both Batch
  changes fail SILENTLY when wrong (a missed barrier reads stale data; a bad
  cache key binds the wrong buffer). Includes: 64 alternating in-place scales
  in one Batch (order-sensitive, exact in binary fp); chains across different
  pipelines/layouts/push sizes; cache hit/miss counter assertions; keys-on-
  buffers-bound with distinguishable values; 100 cached Batches in flight
  (same set bound in 100 live command buffers); abandoned un-submitted Batches
  neither leak nor double-free.
- `tests/kernel_features_test.cpp`: 2-D grids with real data (transpose at
  131×97 etc. — multi-group in both axes, x round-up, bound guards),
  reqd_work_group_size end-to-end (n=1000, not a 32-multiple), multi-kernel
  module interleave on one buffer with no host wait, mixed-POD packing at
  offsets 0/4/8/12/16.
- `tests/launch_test.cpp`: CPU-reference checks at sizes {1..100003} incl.
  round-up + slack-buffer no-overwrite guard; the IEEE compare_fp matrix;
  cross-kernel chains without host waits (barrier proof); 100 interleaved
  in-flight launches across mixed pipelines; grid-too-large/zero-grid per-axis
  throws; forced staging around a dispatch; arity/kind/size mismatch messages.
- Known coverage gap: local-Y round-up (`local[1] > 1`) is never exercised —
  the default local size is 64×1×1 so 2-D grids round up only in X; would need
  a 2-D reqd_work_group_size fixture (e.g. 16×16×1).
