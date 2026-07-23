# Mali-G57 device profile (Redmi Note 10T 5G, Dimensity 700)

Measured ground truth from an on-device probe (Android 13, arm64-v8a, Arm
proprietary driver `libGLES_mali.so`). This is the device that exposed the
non-coherent-memory bug (`mali-coherency-fix.md`).

## Core properties

| Field | Value | Note vs desktop |
|---|---|---|
| deviceName | `Mali-G57 MC2` | |
| deviceType | INTEGRATED_GPU | UMA |
| **apiVersion** | **1.1.177** | Exactly Vulkan **1.1** — validates the Context 1.1 floor as the real minimum; any 1.2/1.3 assumption breaks here |
| vendorID / deviceID | 0x13b5 (Arm) / 0x90930010 | |

## Compute limits

| Limit | Value | Note |
|---|---|---|
| maxComputeWorkGroupCount[0..2] | **[UINT32_MAX ×3]** | ALL dims — the launch() grid guard is unreachable within a uint32 grid here. Contrast: RADV RENOIR [UINT32_MAX, 65535, 65535], NVIDIA [2^31−1, 65535, 65535], llvmpipe [65535 ×3] — so `GridTooLargeThrowsAllAxes` runs on desktop and skips only on Mali |
| maxComputeWorkGroupSize | [512, 512, 512] | fixtures use 64/32 ×1×1 |
| maxComputeWorkGroupInvocations | 512 | |
| maxComputeSharedMemorySize | 32 KiB | unused (no `__local` in the ABI) |

## Transfer / descriptor / memory limits

| Limit | Value | Note |
|---|---|---|
| **maxPushConstantsSize** | **256** | ≥ the 128 B launch.hpp POD cap — cap is safe here |
| **maxStorageBufferRange** | **256 MiB** | Far below desktop (~4 GiB). See latent note below |
| maxUniformBufferRange | 64 KiB | irrelevant — no UBOs in the ABI |
| **maxBoundDescriptorSets** | **4** | small (desktop 8–32); the clspv ABI uses set 0 only, so fine — assert if multi-set kernels ever appear |
| minStorageBufferOffsetAlignment | 64 | |
| **nonCoherentAtomSize** | **64** | the flush/invalidate rounding granularity — the key coherency number |
| maxMemoryAllocationCount | UINT32_MAX | no 4096 cap |

## Memory heaps & types (the coherency ground truth)

Single 5578 MiB DEVICE_LOCAL heap (UMA). Three types:

| Type | Flags | Role under VMA |
|---|---|---|
| type[0] | DEVICE_LOCAL \| HOST_VISIBLE \| **HOST_COHERENT** | `Usage::DeviceLocal` lands here → direct path, no flush needed |
| type[1] | DEVICE_LOCAL \| HOST_VISIBLE \| **HOST_CACHED** (non-coherent) | `Usage::HostVisible` + staging land here → requires flush/invalidate. The type that broke the 6 transfer tests |
| type[2] | DEVICE_LOCAL \| LAZILY_ALLOCATED | transient attachments; never requested |

Every type is DEVICE_LOCAL: there is no non-host-visible type, so the staging
path only ever runs via the `set_force_staging(true)` test hook on this phone.

## Subgroup

subgroupSize **16** (quad-based; NVIDIA 32, RADV 64); operations
BASIC|VOTE|ARITHMETIC|BALLOT|SHUFFLE|SHUFFLE_RELATIVE|CLUSTERED|QUAD; quad ops
fragment-only. No kernel depends on subgroup width — a future subgroup
reduction must not assume 32/64.

## Probe results

- **Coherency fix robustness**: 96/96 unaligned / sub-atom (offset,size)
  round-trip combos correct across {HostVisible, DeviceLocal} × {direct,
  forced-staging} × {partial-upload, partial-download} — VMA's atom-rounding of
  flush/invalidate ranges is correct even sub-atom. Locked as
  `BufferTest.BufferUnalignedSubAtomRoundtrip`.
- Over-heap alloc (largest heap + 1 GiB): throws clean `vulkore::Error`
  (`VK_ERROR_OUT_OF_DEVICE_MEMORY`), no crash. Locked as
  `OverHeapAllocThrowsCleanly`.
- Buffer > maxStorageBufferRange, bound with `VK_WHOLE_SIZE`, dispatched over
  its first elements: no error, no corruption. See latent note.
- 4000 small allocations: fine. scale_inplace over N=4,000,037 (prime),
  CPU-checked: 0 mismatches.
- Profile invariants locked as `ContextTest.DeviceProfileInvariants`.

## Latent portability notes (device facts, not vulkore defects)

1. **`VK_WHOLE_SIZE` binding vs 256 MiB `maxStorageBufferRange`**: launch.cpp
   binds every storage buffer with `VK_WHOLE_SIZE`, which is spec-legal (the
   range VUID is gated on `range != VK_WHOLE_SIZE`). A kernel that *logically
   indexes* past 256 MiB in one buffer is undefined on this device; vulkore does
   not guard it (a friendly warning, not a hard error, would be the shape if
   ever wanted). The LLM KV cache deliberately uses one buffer per layer
   partly for this reason.
2. **maxBoundDescriptorSets = 4** — only matters if a >4-set kernel is ever
   added; current clspv ABI is set 0 only.

Suite status on this device: green, exit 0 via `scripts/run-android.sh`
(environmental skips only — single GPU, no llvmpipe, grid guard unreachable).
