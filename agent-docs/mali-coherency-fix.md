# Non-coherent host memory — the Mali transfer fix

The first run on real mobile hardware (Redmi Note 10T 5G, Mali-G57 MC2)
failed 6 buffer-TRANSFER tests while every compute result was correct.
Desktop (RADV/NVIDIA/llvmpipe) had been fully green — **desktop-green is not
device-green**, because desktop host-visible memory is always HOST_COHERENT
and the non-coherent path is never exercised there.

## The memory-type difference

| Usage request | Desktop (RADV/NVIDIA/llvmpipe) | Mali-G57 |
|---|---|---|
| `Usage::DeviceLocal` (AUTO_PREFER_DEVICE) | NVIDIA: DEVICE_LOCAL only (→ staging); RADV/llvmpipe: HOST_VISIBLE\|HOST_COHERENT | HOST_VISIBLE\|HOST_COHERENT (direct path, worked) |
| `Usage::HostVisible` / staging (AUTO_PREFER_HOST + HOST_ACCESS_RANDOM) | HOST_VISIBLE\|**HOST_COHERENT** | HOST_VISIBLE\|**HOST_CACHED, NON-coherent** |

HOST_CACHED non-coherent is a real, correct memory type (cached = faster CPU
access), not a driver bug. On it, host writes sit in the CPU cache and never
reach the GPU without a flush; host reads see stale cache lines without an
invalidate. Corruption signature: "first ~16 words correct, then
zeros/shifted/repeated".

**Non-coherent means manage the CPU cache, NOT copy data.** UMA devices share
one physical RAM — no staging copy is needed; the only hazard is cache/RAM
desync.

## The two bugs and the fix (`src/buffer.cpp`, `include/vulkore/buffer.hpp`)

1. **Path decision was keyed on coherency, not mappability** — a single
   `host_visible_coherent_ = HOST_VISIBLE && HOST_COHERENT` flag made Mali's
   `HostVisible` buffers report `host_visible() == false` AND fall through to
   staging. Split into `mappable_` (HOST_VISIBLE) + `coherent_`
   (HOST_COHERENT); the direct map path runs whenever
   `mappable_ && !force_staging_`; staging is strictly for non-host-visible
   memory. `host_visible()` now means "directly mappable", coherent or not.
2. **Neither path flushed/invalidated.** Now: direct upload = map → memcpy →
   `if (!coherent_) vmaFlushAllocation(range)` while still mapped → unmap;
   direct download = map → `if (!coherent_) vmaInvalidateAllocation(range)`
   while mapped → memcpy → unmap. The staging path flushes its own staging
   allocation after the host memcpy-in and invalidates it after the copy+fence
   before the memcpy-out (the staging buffer is persistently mapped, so those
   calls are always valid).

Desktop behaviour is unchanged: `vmaFlush/InvalidateAllocation` return early
(no driver call) when the memory type is HOST_COHERENT, and the direct path
additionally gates on `!coherent_`.

## The ordering VUID that null-derefs the Mali driver

**VUID-VkMappedMemoryRange-memory-00684: the memory must be currently
host-mapped when flush/invalidate is called.** The first fix attempt
invalidated before mapping; Mali's `vkInvalidateMappedMemoryRanges`
dereferenced its null mapped-pointer → SIGSEGV inside `libGLES_mali.so`.
Desktop can never catch this: on coherent memory the invalidate is a VMA no-op
that never reaches the driver. **Map FIRST, then flush/invalidate while
mapped.**

Symbolizing device crashes without root: `/data/tombstones` is unreadable —
pull the backtrace from `adb logcat -s DEBUG` and symbolize against the
unstripped binary with the NDK's `llvm-symbolizer` (or `ndk-stack`).

## Interaction with launch()

`launch()`'s post-dispatch HOST barrier is the Vulkan *availability* operation
(makes shader writes visible to host reads); the CPU-cache invalidate in
`download_bytes` is orthogonal and complements it. The contract is unchanged:
wait the launch `Fence`, then download. No coherency logic belongs in
launch.cpp — all host access funnels through `Buffer::upload/download_bytes`.

## Status

Verified on the Mali device (all six formerly-failing tests pass, suite exit
0) and robust across 96/96 unaligned/sub-atom transfer combinations
(`mali-g57-device-profile.md`). Locked by
`BufferTest.BufferUnalignedSubAtomRoundtrip` (`vulkore-core.md`). The fix also
carried unmodified to the Adreno 840, where `DeviceLocal` is not host-visible
and takes staging instead (`adreno-840-device-profile.md`) — keying on
mappability covers both worlds.
