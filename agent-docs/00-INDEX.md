# agent-docs index

Shared knowledge base for the **Vulkore** repo — the engineering reference for
future agents and engineers. Read the relevant doc before working in its area;
update it when your work changes or supersedes something here.

## Core runtime

| Doc | One-liner |
|---|---|
| `architecture.md` | What Vulkore is, the clspv→SPIR-V→Vulkan pipeline, why Vulkan (Adreno OpenCL never ingests SPIR-V IL), module inventory, dependency policy, positioning vs clvk/Kompute. |
| `vulkore-core.md` | Context/Buffer/Sync: public API, design decisions (per-Context volk device tables, exceptions, selection policy, mappability-keyed transfers), invariants, test-suite design rules. |
| `vulkore-program-launch.md` | Program/launch/Batch/DescriptorCache: fixture pipeline + clspv flags, ClspvReflection wire format, non-semantic stripping, launch semantics, Batch barrier scheme, DescriptorCache lifetime contract. |

## LLM inference

| Doc | One-liner |
|---|---|
| `llm-inference.md` | Gemma 3 1B end to end: model facts, the (1+w) norm-weight GGUF bug, design rules (one submit/token, no `__local`, uploads are submits), kernel contracts, KV cache + sliding window, sampling pipeline, weight pack/`.xpack` cache, verification surface. |
| `llm-performance.md` | The measured numbers (decode curve, llama.cpp/LiteRT-LM baselines), the dispatch-bound-then-bandwidth-saturated cost model, every optimisation as problem→change→effect, on-device kernel validation, the turnip-proxy rule, measurement traps. |
| `gguf.md` | GGUF v3 parsing, SentencePiece tokenizer, exact quant bit layouts (Q8_0/Q5_0/Q4_K/Q6_K), the Q4_K_M census surprise, native + repacked dequant/matvec kernels and the repack scheme. |

## Device profiles & portability

| Doc | One-liner |
|---|---|
| `mali-coherency-fix.md` | Non-coherent HOST_CACHED memory on Mali: mappability vs coherency, flush-after-write / invalidate-before-read while mapped (VUID-00684 null-derefs the Mali driver otherwise). READ before touching Buffer/transfers. |
| `mali-g57-device-profile.md` | Measured Mali-G57 ground truth: Vulkan 1.1 exactly, nonCoherentAtomSize 64, maxStorageBufferRange 256 MiB, memory types, probe results, latent portability notes. |
| `adreno-840-device-profile.md` | The primary target (OnePlus 15): `DeviceLocal` is NOT host-visible → staging path (unlike Mali UMA), cross-build+run recipe, Adreno/Mali data-point table. |
| `exit-teardown-fix.md` | Exit-139 UAF: Context-dependent objects in statics must be destroyed before the Context; ctest exit-0 guard; MALLOC_PERTURB_ debugging trick. |

## Build, testing & environment

| Doc | One-liner |
|---|---|
| `android-build.md` | NDK cross-compile (zero patches): volk/bionic facts, android-26 rationale, no libvulkan link, symbolization notes. |
| `android-device-testing.md` | `scripts/run-android.sh` (options, mechanics), crash symbolization without root, flaky-adb lessons for long on-device runs. |
| `environment.md` | The two dev machines (x86_64 3-driver box; aarch64 ThinkPad with Adreno X1-85/turnip), NDK-under-qemu setup (`QEMU_LD_PREFIX`), clspv/SPIRV-Tools builds, phones. |
| `repo-layout.md` | Tree structure, third-party submodule pins, fresh-checkout bring-up, submodule rules. |
