# android-build — cross-compiling Vulkore for Android

**Builds clean with zero patches**: `libvulkore.a` and the `vulkore_tests` binary
compile and link with NDK r27 / arm64-v8a / android-26, unmodified sources, no
CMake changes beyond the standard toolchain trio.

```sh
NDK=<ndk-r27 path>            # e.g. ~/Android/Sdk/ndk/27.0.12077973 or ~/Android/Sdk/ndk-r27c
cmake -G Ninja -S . -B build-android-arm64 \
      -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
      -DCMAKE_BUILD_TYPE=Release
ninja -C build-android-arm64          # lib + gtest + tests; target `vulkore` for lib only
```

`build-*/` dirs are gitignored. On the aarch64 ThinkPad the NDK runs under
qemu — `export QEMU_LD_PREFIX=/home/nikhil/x86_64-sysroot` first
(`environment.md`).

## Facts that make it work (verified, not assumed)

- **The test binary links against `libdl.so libm.so libc.so` ONLY — no
  `libvulkan.so`.** volk's `volk.c` has a native `#elif defined(__ANDROID__)`
  branch: `dlopen("libvulkan.so.1")` then fallback `dlopen("libvulkan.so")`
  (Android ships only the latter, `/system/lib64/libvulkan.so`). No
  `VK_USE_PLATFORM_ANDROID_KHR` needed — that define only unlocks surface/
  AHardwareBuffer entry points, irrelevant for compute.
- `/system/lib64/libvulkan.so` **is in the default linker namespace for plain
  executables run from `/data/local/tmp` via adb** (no classloader-namespace
  restrictions — the standard Vulkan-CTS-style pushed-binary pattern).
- googletest builds on bionic with no flags (its CMake handles Android; the
  "pthread_create not found" configure lines are normal — pthreads live in
  bionic libc).
- No Linux-only assumptions in vulkore: only `std::getenv`/`::setenv` (in bionic
  since API 1); no platform ifdefs; `CMAKE_DL_LIBS` resolves under the NDK.
- **android-26 (Android 8.0) is the baseline**: Vulkan 1.1 guaranteed on
  Vulkan-capable 8.0+ devices, matching Context's requested apiVersion. A
  Vulkan-1.0-only device would fail at Context creation — runtime, not build
  time.
- Reflection is stripped before module creation, so no device extension
  (not even VK_KHR_shader_non_semantic_info) is required on-device.
- The pushed binary is unstripped (~12 MB) — keep it that way for
  `ndk-stack`/`llvm-symbolizer` tombstone symbolization; `llvm-strip` a copy if
  transfer size matters.
- Standalone example binaries (`examples/llm/*`) need `-static-libstdc++` or
  they die on a missing `libc++_shared.so` on-device.

## Kompute comparison (reference-read takeaway)

Kompute's Android loader plumbing is the same idea in vulkan-hpp form
(`vk::DynamicLoader` dlopen + dispatcher init) — nothing to copy. It only
initialises a GLOBAL dispatcher, which has exactly the multi-context
clobbering hazard vulkore's per-Context `VolkDeviceTable` avoids.
