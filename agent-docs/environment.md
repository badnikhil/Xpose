# environment — dev machines, toolchains, devices

Two development machines exist. **Paths look identical (`/home/nikhil`) but the
machines are not — verify which one you are on before trusting any path.**

| | x86_64 box | aarch64 ThinkPad T14s Gen 6 |
|---|---|---|
| SoC/GPU(s) | AMD RENOIR iGPU (RADV) + NVIDIA RTX 2050 (proprietary) + llvmpipe | Snapdragon X1E80100 → **Adreno X1-85 (Mesa turnip)** + llvmpipe |
| Test matrix | `for d in llvmpipe RENOIR NVIDIA` | `for d in llvmpipe Adreno` |
| NDK | `~/Android/Sdk/ndk/{26.3,27.0,28.2}` (native x86_64) | `~/Android/Sdk/ndk-r27c` (x86_64 binaries under qemu — see below) |
| Notes | gcc 13 + clang 18; adb native | gcc only (no clang); adb native arm64 |

Neither machine has a system Vulkan SDK: no `/usr/include/vulkan/`, no
`libvulkan.so` dev symlink — only the runtime loader. Headers are vendored
(`third_party/Vulkan-Headers`) and volk dlopens the loader; never `-lvulkan`.
`vulkaninfo` is not installed; enumerate devices with the test runner (it
prints the selected device) or a small volk probe.

The ThinkPad's Adreno X1-85 gives Adreno-family coverage without a phone —
but it runs Mesa **turnip** while phones run Qualcomm's proprietary driver,
and driver differences are exactly where the Mali bug lived. Numerics proxy
yes, performance proxy no (`llm-performance.md`). Never a substitute for the
on-device run.

## NDK on the aarch64 ThinkPad (x86_64 emulation)

Google ships the Linux NDK as **x86_64 host binaries only**. They run fine
under `qemu-user` binfmt (cross-build wall time ~1m13s — entirely usable),
with one requirement:

```sh
export QEMU_LD_PREFIX=/home/nikhil/x86_64-sysroot   # REQUIRED
```

Without it every NDK tool dies on the missing `/lib64/ld-linux-x86-64.so.2`.
`~/x86_64-sysroot` is a local unpack of the amd64 libc6/libgcc-s1/libstdc++6/
zlib1g `.deb`s fetched straight from the Ubuntu pool and unpacked with
`dpkg-deb -x` — no sudo, no multiarch (`dpkg --add-architecture amd64` gets
nothing: the arm64 archive does not serve amd64 packages). Under qemu the NDK
clang occasionally dies spuriously (~1 run in 3 on big TUs) — just retry.

Cross-build recipe: `android-build.md`. `scripts/run-android.sh` defaults to
the x86_64 box's NDK path — on the ThinkPad pass
`ANDROID_NDK=/home/nikhil/Android/Sdk/ndk-r27c`.

## clspv & SPIRV-Tools

- clspv's deps are FETCHED clones, not submodules: `python3
  utils/fetch_sources.py --shallow` inside `third_party/clspv` populates
  `third_party/clspv/third_party/{llvm,SPIRV-Headers,SPIRV-Tools}` (~2.9 GB,
  fast with `--shallow`).
- A clspv build is an LLVM build — budget 10–20+ GB of disk, Release only.
  The local binary lands at `third_party/clspv/build/bin/clspv` and
  `regenerate.py` prefers it over godbolt.
- `third_party/spirv-tools-build/` (gitignored) holds `spirv-{as,dis,val}`
  built from **clspv's pinned SPIRV-Tools against clspv's SPIRV-Headers** —
  the only combo that builds; the top-level `SPIRV-Tools` submodule does not
  build against that Headers pin. Rebuild recipe in `CLAUDE.md`; small build,
  ~2 min.

## Phones

| Device | Role |
|---|---|
| OnePlus 15 (SM8850, Adreno 840, Qualcomm proprietary, Android 16) | primary target — `adreno-840-device-profile.md` |
| Redmi Note 10T 5G (Dimensity 700, Mali-G57 MC2, Android 13) | the device that found the coherency bug — `mali-g57-device-profile.md` |

Both are flaky adb targets under load; see the device-handling lessons in
`android-device-testing.md`.

llama.cpp baseline builds live outside the repo in `/home/nikhil/bench/`
(llama.cpp + OpenCL headers/ICD loader cross-built for arm64); build flags and
gotchas in `llm-performance.md`.
