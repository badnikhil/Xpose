# android-device-testing — running the suite (and anything else) on a phone

On-device runs are the ONLY way to exercise the non-coherent-memory path
(`mali-coherency-fix.md`) — run them before trusting buffer transfers.

## `scripts/run-android.sh`

One command from the repo root: preflights adb + device authorization, prints
model/Android/ABI and warns if `/system/lib64/libvulkan.so` is absent,
optionally cross-compiles (the `android-build.md` recipe), pushes the binary +
the `.spv` fixtures to `/data/local/tmp/vulkore`, runs with
`VULKORE_KERNEL_DIR=/data/local/tmp/vulkore/kernels`, streams output, and
**propagates the gtest exit code** as its own. Cleans up via an EXIT trap
unless `--keep`.

| Option | Meaning |
|---|---|
| `-s <serial>` | Target a device (auto-selects when exactly one is attached; errors and lists serials otherwise) |
| `--build` / `--rebuild` | Cross-compile first (reuse / wipe the build dir) |
| `--abi <abi>` | Build ABI (default `arm64-v8a`; build dir `build-android-arm64`) |
| `--android-platform <p>` | API level for the build (default `android-26`) |
| `--device-filter <substr>` | Exported as `VULKORE_DEVICE` on-device (only needed on multi-ICD boards; a phone has one GPU) |
| `--gtest-filter <expr>` | Passed through as `--gtest_filter` |
| `--keep` | Leave `/data/local/tmp/vulkore` in place |

Env overrides: `ANDROID_NDK`/`NDK` (the script's default NDK path is
machine-specific — on the aarch64 ThinkPad set
`ANDROID_NDK=/home/nikhil/Android/Sdk/ndk-r27c`), `JOBS`.

Mechanics it relies on:

- The tests read `VULKORE_KERNEL_DIR` at runtime, falling back to the baked
  compile-definition path only if unset; the pushed `kernels/` dir must contain
  the `.spv` files directly.
- The binary needs nothing else pushed (dlopens the system libvulkan;
  `android-build.md`).
- Success = all tests pass + exit 0. Phones show a handful of environmental
  `GTEST_SKIP`s (single GPU, no llvmpipe; on Mali also the grid-limit test —
  see the device profiles).

Phone one-time setup: enable Developer options + USB debugging, data-capable
cable, accept the debugging dialog with "always allow". On MIUI also "USB
debugging (Security settings)" if prompted.

## Crash symbolization without root

`/data/tombstones` is unreadable on non-rooted phones. Get the backtrace from
`adb logcat -s DEBUG`, then symbolize against the unstripped binary:
`$NDK/ndk-stack -sym build-android-arm64/tests` (or `llvm-symbolizer -e <binary>`).

## Flaky-device lessons (long benchmark/LLM runs)

Collected from sessions where the phone dropped off USB every 10–20 s:

- **`adb push` can exit 0 having transferred nothing.** For large files:
  `split -b 32M`, push each chunk with a size-verified retry loop, `cat` on
  device, then verify md5 end-to-end.
- **Interactive `adb shell <cmd>` gets SIGHUP'd on disconnect.** Push a small
  launcher script, run it under `nohup` detached, poll for a completion marker
  in the output file. `nohup` output is block-buffered — an empty log does not
  mean a stalled process.
- **Do not put `/vendor/lib64` on `LD_LIBRARY_PATH` for the whole adb shell
  command** — `nohup` itself then fails to link
  (`phdr mmap failed: Permission denied`). Set the env inside the detached
  script.
- **`pgrep -f <name>` matches its own launching shell** — judge liveness from
  the output file only.
- **The phone deep-sleeps when USB drops**, freezing long runs:
  `settings put global stay_on_while_plugged_in 7` before any long detached
  run.
- Two background jobs writing the same output file = reading a stale result as
  new. Distinct filenames per run.
- Strip pushed benchmark binaries (`llvm-strip --strip-all`, 129 → 9.8 MB) —
  the difference between minutes and hours on a flaky link. (Keep the TEST
  binary unstripped for symbolization, or keep both.)
