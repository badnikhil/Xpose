# Adreno 840 device profile (OnePlus 15, SM8850)

The primary target device: OnePlus 15 (`CPH2745`), Snapdragon 8 Elite Gen 5,
**Adreno 840 with Qualcomm's proprietary Vulkan driver**, Android 16,
arm64-v8a, one Vulkan device visible.

Full test suite: green, exit 0 (environmental skips only — single device, no
llvmpipe). Every buffer-transfer test passed on first contact: the Mali
coherency fix (`mali-coherency-fix.md`) carried without change.

## KEY FINDING — the memory model differs from Mali's

```
HostVisible alloc  -> host_visible() = true
DeviceLocal alloc  -> host_visible() = FALSE
roundtrip upload->download: OK
```

**On Mali (UMA), `DeviceLocal` comes back host-visible and takes the
direct-map path. On the Adreno 840, `DeviceLocal` is NOT host-visible — every
`DeviceLocal` transfer takes the STAGING path** (an extra copy + a transfer
submit). Same correct results, different code path; keying the transfer
strategy on mappability covers both. Do not assume "mobile = UMA = direct
map".

Consequences that have already mattered:

- A host→device upload is a submit (staging = `one_shot` + wait) — this drove
  the LLM decode design (inline RoPE angles, CPU embedding memcpy,
  `HostVisible` buffers for sampling readbacks; `llm-inference.md`).
- The 536 MiB weight upload at model load goes entirely through staging and is
  the largest single term of a warm start (~0.48 s); making it cheaper is a
  `Buffer` question.
- Whether Adreno's HostVisible allocations are coherent is **not established**
  (`Buffer` exposes mappability, not coherency) — unknown, not "no".

## Adreno/Mali data points

| Device | GPU | Driver | Suite |
|---|---|---|---|
| ThinkPad T14s Gen 6 (X1E80100) | Adreno X1-85 | Mesa **turnip** | green |
| OnePlus 15 (SM8850) | **Adreno 840** | **Qualcomm proprietary** | green |
| Redmi Note 10T 5G | Mali-G57 MC2 | Arm proprietary | green (post coherency fix) |

Both Adreno driver stacks — open-source turnip and Qualcomm proprietary — pass
the full suite. For the turnip-vs-proprietary proxy rule (numerics faithful,
performance NOT), see `llm-performance.md`.

Other measured facts on this device, recorded elsewhere: pure-load bandwidth
ceiling 62.7–70.1 GB/s; batched dispatch cost ~5.6 µs; submit cost ~0.31 ms;
best matvec workgroup size wg128 (vs wg32 on turnip); wide→narrow barrier
narrowing worth ~1.2 ms/838 dispatches; the device throttles under sustained
load (inter-run clock spread up to 1.6x, intra-run 0.2%) — all in
`llm-performance.md`.

## Cross-build + run recipe

Built on the aarch64 ThinkPad (NDK under qemu — see `environment.md`):

```sh
export QEMU_LD_PREFIX=/home/nikhil/x86_64-sysroot
NDK=/home/nikhil/Android/Sdk/ndk-r27c
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DCMAKE_BUILD_TYPE=Release
cmake --build build-android

adb shell mkdir -p /data/local/tmp/vulkore/kernels
adb push build-android/tests/vulkore_tests /data/local/tmp/vulkore/
adb push tests/kernels/ /data/local/tmp/vulkore/kernels/
adb shell "cd /data/local/tmp/vulkore && \
  VULKORE_KERNEL_DIR=/data/local/tmp/vulkore/kernels/kernels ./vulkore_tests"
```

Note `adb push tests/kernels/ …/kernels/` nests as `kernels/kernels` — point
`VULKORE_KERNEL_DIR` at the inner path. `scripts/run-android.sh` automates all of
this (`android-device-testing.md`); on the ThinkPad override its NDK default
with `ANDROID_NDK=/home/nikhil/Android/Sdk/ndk-r27c`.
