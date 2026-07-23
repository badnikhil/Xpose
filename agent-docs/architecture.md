# Vulkore — what it is and why it is shaped this way

**Vulkore is a pure C++ framework for running GPU compute kernels on Android
(and desktop Vulkan) with CUDA-like ergonomics.** Kernels are written in
OpenCL C — real C, pointers and all — compiled at build time by
**google/clspv** into Vulkan-flavor SPIR-V with embedded reflection; a slim
C++20 Vulkan runtime loads the `.spv`, maps kernel arguments to descriptor
bindings via the reflection, and dispatches:

```
kernel.cl  (OpenCL C)
    │  build time: clspv
    ▼
kernel.spv  (Vulkan/Shader-flavor SPIR-V
             + NonSemantic.ClspvReflection: kernel names, arg→set/binding, spec constants)
    │  runtime: libvulkore
    │    Context → Buffer(VMA) → Program(reflection parse) → launch()/Batch → Fence
    ▼
libvulkan.so  (guaranteed NDK public library on every Android device)
    → Adreno / Mali / desktop RADV / NVIDIA / llvmpipe — anything with Vulkan compute
```

## Why Vulkan and not OpenCL (primary-source verdict)

- **Qualcomm's Adreno OpenCL driver does not ingest SPIR-V IL — on any Adreno
  generation, on any OS.** gpuinfo.org: `cl_khr_il_program` appears in ~50% of
  all OpenCL reports database-wide and **0%** of Adreno reports; no Adreno
  report has a non-empty `CL_DEVICE_IL_VERSION`. Qualcomm's own Snapdragon
  OpenCL guide never mentions `clCreateProgramWithIL`. Adreno OpenCL is rich
  but source/binary-only.
- The only Android OpenCL drivers accepting SPIR-V IL are ARM Mali (SPIR-V 1.0
  only) and one Imagination part — useless on Snapdragon-dominated Android.
  OpenCL library loading is itself OEM roulette (`public.libraries.txt`;
  Pixels ship no OpenCL at all).
- **`libvulkan.so` is a guaranteed NDK public library on every Android
  device** — always in an app's linker namespace.

So on Snapdragon the only stock-driver door for precompiled-IL compute is
Vulkan/Shader-flavor SPIR-V. Vulkan is the requirement, not a preference. (An
earlier plan to add a Vulkan codegen target to dcompute/LDC was dropped: clspv
already is a mature C-to-Vulkan-SPIR-V compiler.)

## Positioning

| | Kernel language | Host API |
|---|---|---|
| clvk | OpenCL C via clspv | full OpenCL 3.0 API (verbose; carries conformance weight) |
| Kompute | GLSL | nice modern C++, but kernels are a shading language |
| **Vulkore** | **OpenCL C — real C kernels** | **CUDA-style `vulkore::launch(kernel, grid, args...)`** |

clvk proves the clspv-on-Vulkan approach and donated reflection-parsing wisdom
(`third_party/clvk/src/program.cpp`); Kompute donated Android packaging prior
art. Vulkore takes the empty quadrant: C kernels AND one-line launches.

## Modules (as built)

- **Context** (`context.{hpp,cpp}`) — instance/device/queue init, device
  selection policy, command pool, VMA allocator, on-demand descriptor pools,
  one-shot command helpers, per-context `VolkDeviceTable`. Single-threaded by
  design (including teardown). Non-movable (Buffers/Fences hold `Context*`).
- **Buffer** (`buffer.{hpp,cpp}`) — VMA-backed storage buffers, typed
  upload/download, transfer strategy keyed on mappability with
  flush/invalidate on non-coherent memory (`mali-coherency-fix.md`).
- **Program / Kernel** (`program.{hpp,cpp}`) — SPIR-V load, ClspvReflection
  parse, pipeline/layout creation (lazy, cached), non-semantic stripping so no
  device extensions are required.
- **launch() / Batch / DescriptorCache** (`launch.{hpp,cpp}`) — variadic
  dispatch returning a `Fence`; `Batch` records many dispatches into one
  submit; `DescriptorCache` memoises descriptor sets. Details in
  `vulkore-program-launch.md`.
- **Sync** (`sync.{hpp,cpp}`) — RAII `Fence` with completion hooks (drives
  command-buffer/descriptor recycling).
- Umbrella header `include/vulkore/vulkore.hpp`; errors via `vulkore::Error` +
  `XP_CHECK` (see `vulkore-core.md`).

## Dependency policy

- `third_party/VulkanMemoryAllocator` — the only hard runtime dep (header-only).
- `third_party/volk` — loader; dlopens `libvulkan.so(.1)`, no link-time Vulkan dep.
- `third_party/Vulkan-Headers` — vendored (no system Vulkan SDK assumed).
- `third_party/clspv` — build-time tool only, never shipped on device.
- `third_party/clvk`, `third_party/kompute` — reference reading only.
- `third_party/SPIRV-Tools` + `spirv-tools-build/` — dev tooling
  (`spirv-as/dis/val`) for the fixture pipeline (`repo-layout.md`).

## Deliberate non-features

- No images/samplers/UBO-PODs/printf — `Program` throws at load naming the
  instruction (`vulkore-program-launch.md` has the full list).
- No timeline semaphores; no pipeline-cache persistence yet (TODO in
  `context.hpp` — a real win on Android where pipeline compile happens
  on-device at first run).
- On-device JIT (linking clspv's C++ API) is possible but pulls LLVM into the
  binary; not pursued.
