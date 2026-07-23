# vulkore-core — Context / Buffer / Sync

## Public API (umbrella: `#include <vulkore/vulkore.hpp>`)

```cpp
namespace vulkore {

class Error : std::runtime_error { VkResult result(); };   // check.hpp
// XP_CHECK(vkCall(...)) — throws Error with VkResult name + expr + file:line

struct DeviceInfo { uint32_t index; std::string name; VkPhysicalDeviceType type; uint32_t api_version; };

class Context {                                            // context.hpp
  explicit Context(std::optional<uint32_t> device_index = std::nullopt);
  const std::string& device_name();  const VkPhysicalDeviceProperties& device_properties();
  const std::vector<DeviceInfo>& all_devices();  uint32_t device_index();
  template<class T> Buffer alloc(size_t count, Usage);     // sizeof(T)%4==0 static_assert
  template<class T> Buffer alloc(size_t count);            // = Usage::DeviceLocal
  Buffer alloc_bytes(VkDeviceSize, Usage);
  VkCommandBuffer begin_one_shot();
  Fence submit(VkCommandBuffer);                           // + submit(cb, on_complete) hook overload
  void free_command_buffer(VkCommandBuffer);
  void one_shot(std::function<void(VkCommandBuffer)>);     // begin+record+submit+wait+free
  void wait_idle();                                        // cudaDeviceSynchronize analog
  // raw seam for Program/launch:
  VkInstance instance(); VkPhysicalDevice physical_device(); VkDevice device();
  VkQueue queue(); uint32_t queue_family(); VmaAllocator allocator();
  const VolkDeviceTable& table();                          // ALL device-level calls go through this
};

enum class Usage { DeviceLocal, HostVisible };             // buffer.hpp

class Buffer {                                             // RAII, move-only
  VkDeviceSize size_bytes(); bool valid(); bool host_visible();  // = directly mappable
  template<class T> void upload(std::span<const T>, VkDeviceSize dst_offset_bytes = 0);
  template<class T> void download(std::span<T>, VkDeviceSize src_offset_bytes = 0);
  void upload_bytes(const void*, VkDeviceSize size, VkDeviceSize off = 0);   // + download_bytes
  void set_force_staging(bool);                            // test hook: force copy path on UMA
  VkBuffer handle();
};

class Fence {                                              // sync.hpp, RAII move-only
  bool wait(uint64_t timeout_ns = UINT64_MAX);             // true=signaled, false=timeout, throws on device loss
  bool is_signaled(); VkFence handle();
  // double-wait safe; moved-from fence => wait() no-op true; dtor drains
};
}
```

## Decisions and why

| Decision | Choice | Why |
|---|---|---|
| Errors | Exceptions (`vulkore::Error` + `XP_CHECK`) | Every Vk call checked; message carries VkResult name + expression + location. |
| Loader | volk, compiled into libvulkore (`volk.c` as a TU) | dlopens `libvulkan.so.1`/`libvulkan.so` — no link-time loader dep; matches Android; no system dev headers assumed. No `VOLK_STATIC_DEFINES` needed for compute-only. |
| Multi-context safety | `volkLoadInstanceOnly` + **per-Context `VolkDeviceTable`** | Global `volkLoadDevice` clobbers device-level pointers when two Contexts on different drivers coexist. **Rule: device-level calls only via `ctx.table()`.** |
| API version | Instance & VMA at Vulkan **1.1** | The Android floor — and the real minimum: Mali-G57 reports exactly 1.1. |
| Device selection | explicit index > `VULKORE_DEVICE` (case-insensitive substring) > first discrete > first integrated > anything | No-match on the env var **throws** and lists devices rather than silently falling back. Explicit index short-circuits before the env check. |
| Queue family | dedicated compute (COMPUTE && !GRAPHICS) if present, else first compute-capable | Both paths exist in the wild (desktop discretes vs llvmpipe/mobile combined queues). One queue, priority 1.0. |
| Memory | VMA `AUTO_PREFER_DEVICE`/`AUTO_PREFER_HOST`; impl in single TU `src/vma_impl.cpp`; `VMA_DYNAMIC_VULKAN_FUNCTIONS=1` fed by volk (PUBLIC compile defs so all TUs agree) | |
| Transfer strategy | Decided per-allocation via `vmaGetAllocationMemoryProperties`: **mappable (HOST_VISIBLE) → map+memcpy**, with flush-after-write / invalidate-before-read when NOT HOST_COHERENT; staging + one-shot `vkCmdCopyBuffer` only for truly non-host-visible memory | Keyed on **mappability, not coherency** — Mali hands back HOST_VISIBLE\|HOST_CACHED non-coherent memory (`mali-coherency-fix.md`). Flush/invalidate are VMA no-ops on coherent memory, so desktop hot paths are unchanged. UMA devices get zero-copy even for `DeviceLocal`; NVIDIA (and Adreno 840 `DeviceLocal`) take staging. |
| `host_visible()` semantics | "directly mappable", true for any HOST_VISIBLE allocation, coherent or not | |
| Buffer usage flags | Always `STORAGE \| TRANSFER_SRC \| TRANSFER_DST` | clspv maps global pointers to storage buffers. |
| Ownership | Buffer/Fence hold `Context*` → Context non-movable; Buffer/Fence move-only RAII; Fence dtor drains before destroy | Destroying in-flight fences is invalid. |
| Compilers | gcc default; clang verified clean. C++20, `-Wall -Wextra -Wno-missing-field-initializers`, third-party TUs `-w` | |

`VkDeviceCreateInfo` enables **no features/extensions** — Program strips
non-semantic instructions so none are needed; enable shaderInt64 / 8-16-bit
storage there if a future kernel wants them (TODO marker in context.cpp).

## Known gaps / invariants

- `Program`/`Kernel`/`Buffer`/`Fence` must not outlive their Context —
  **documented, not enforced** (no alive-flag assert). This has bitten once via
  a test-fixture static (`exit-teardown-fix.md`).
- Context is single-threaded by design — allocation, upload, teardown alike
  (load-path threading works around this by keeping workers on host memory
  only, `llm-inference.md`).
- No timeline semaphores; no images/samplers.

## Test suite

`tests/` googletest; **91 tests** (Check, Context, Buffer, Sync, Program,
ProgramError, Launch, KernelFeatures, Batch). Green with exit 0 on: NVIDIA RTX
2050, RADV RENOIR, llvmpipe (x86_64 box), Adreno X1-85/turnip + llvmpipe
(aarch64 laptop); on phones: 88 pass + 3 env-skips (OnePlus 15 / Adreno 840),
Mali-G57 green via `run-android.sh`. Also clean under `MALLOC_PERTURB_=165`;
ctest treats any nonzero exit as failure (the teardown-SIGSEGV guard).

Test-design rules that keep the ONE binary valid everywhere:

- **Environment-specific assertions are guarded, not deleted**: llvmpipe-based
  tests `GTEST_SKIP` unless an llvmpipe device exists (an unmatched
  `VULKORE_DEVICE` throws, which would otherwise fail on a phone);
  multi-device tests skip below 2 devices. Enumeration is probed once with
  `VULKORE_DEVICE` unset.
- Coverage highlights: roundtrip integrity at odd/prime sizes; both transfer
  paths forced via `set_force_staging(true)` (staging is otherwise dead code on
  UMA); float bit-pattern roundtrips incl. −0.0/NaN/inf; offset partial
  transfers; move-over-live-object semantics; OOB throws; selection policy
  matrix; fence timeout/double-wait/moved-from; completion hook fires exactly
  once and swallows a throwing hook.
- `BufferTest.BufferUnalignedSubAtomRoundtrip` — THE regression guard for the
  coherency fix: sub-`nonCoherentAtomSize`/unaligned (offset,size) pairs
  incl. a runtime-derived `(atom−1, atom+3)` boundary-cross, over
  {HostVisible,DeviceLocal} × {direct,forced-staging} × {upload,download},
  asserting the window byte-exact AND surrounding bytes untouched. Only truly
  exercised on non-coherent (mobile) memory; a no-op pass on desktop.
- `BufferTest.OverHeapAllocThrowsCleanly` — locks the OOM `XP_CHECK`. Sizing
  lesson: request more than the largest heap of ANY type + 1 GiB — sizing off
  the device-local heap alone let VMA fall back to a host heap and COMMIT ~5
  GiB before failing.
- `LaunchTest.GridTooLargeThrowsAllAxes` — per-axis guard proof; runs on
  drivers with y/z = 65535 (RADV/NVIDIA/llvmpipe), skips where all dims are
  UINT32_MAX (Mali).
- `ContextTest.DeviceProfileInvariants` — locks the 1.1 floor, power-of-two
  `nonCoherentAtomSize`, `maxPushConstantsSize ≥ 128` (the launch.hpp POD cap),
  ≥1 HOST_VISIBLE type. Uses the underscore core spelling
  `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2` (the KHR `...PROPERTIES2`
  form does not exist in the vendored headers — a recurring compile gotcha).
- Flaky-by-nature checks are guarded: the fence-timeout test asserts only the
  blocking-wait success and skips if the device drained too fast; the
  "is_signaled false before completion" test was deliberately omitted as
  inherently racy.
