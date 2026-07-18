#!/usr/bin/env bash
# =============================================================================
# run-android.sh — build (optional) and run the Vulkore GoogleTest suite on a
#                  physically-connected Android device over adb.
#
# What it does:
#   1. Preflight: finds adb, starts the server, selects & verifies an
#      authorized device, and prints its model / Android version / ABI.
#   2. (Optional) cross-compiles the arm64 build with the Android NDK.
#   3. Pushes the aarch64 PIE test binary + the committed .spv kernel fixtures
#      to /data/local/tmp/vulkore on the device.
#   4. Runs vulkore_tests with VULKORE_KERNEL_DIR pointing at the pushed kernels,
#      streaming output back to the host and propagating the gtest exit code.
#   5. Cleans up the on-device temp dir (unless --keep).
#
# The test binary links no Vulkan lib; volk dlopens the device's own
# /system/lib64/libvulkan.so at runtime, so nothing else needs pushing.
#
# Examples:
#   # Redmi Note 10T 5G (or any single connected phone), reuse existing build:
#   scripts/run-android.sh
#
#   # First run on a fresh checkout (or after source changes) — cross-compile:
#   scripts/run-android.sh --build
#
#   # Force a clean rebuild of the arm64 build dir first:
#   scripts/run-android.sh --rebuild
#
#   # Several devices attached — target one by adb serial:
#   scripts/run-android.sh -s 1a2b3c4d
#
#   # Only the Launch suite, and force-select a named GPU on a multi-ICD board:
#   scripts/run-android.sh --gtest-filter 'LaunchTest.*' --device-filter Mali
#
# NDK: defaults to the path validated in agent-docs/android-build.md; override
#      with the ANDROID_NDK or NDK environment variable.
# =============================================================================
set -euo pipefail

# --- constants ---------------------------------------------------------------
DEFAULT_NDK="/home/nikhil/Android/Sdk/ndk/27.0.12077973"
DEVICE_DIR="/data/local/tmp/vulkore"          # on-device temp dir

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

# --- defaults (option state) -------------------------------------------------
SERIAL=""
DO_BUILD=0
DO_REBUILD=0
ABI="arm64-v8a"
ANDROID_PLATFORM="android-26"
DEVICE_FILTER=""
GTEST_FILTER=""
KEEP=0

# --- helpers -----------------------------------------------------------------
log()  { printf '[run-android] %s\n' "$*" >&2; }
warn() { printf '[run-android] WARNING: %s\n' "$*" >&2; }
die()  { printf '[run-android] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat >&2 <<EOF
Usage: $(basename "$0") [options]

Build (optionally) and run the Vulkore GoogleTest suite on a connected Android
device over adb.

Options:
  -s <serial>              Target a specific adb device (see 'adb devices').
                           If omitted and exactly one device is connected it is
                           used; if several are connected you must pass -s.
      --build              Cross-compile the arm64 build before running
                           (reuses/refreshes build-android-<abi>).
      --rebuild            Wipe the arm64 build dir and cross-compile from
                           scratch before running.
      --abi <abi>          Android ABI to build for (default: ${ABI}).
      --android-platform <plat>
                           Android platform/API for the build
                           (default: ${ANDROID_PLATFORM}).
      --device-filter <substr>
                           Exported as VULKORE_DEVICE on-device: substring-match
                           to pick a specific GPU (optional; usually 1 GPU).
      --gtest-filter <expr>
                           Passed to the binary as --gtest_filter=<expr>.
      --keep               Do not delete the on-device temp dir after running.
  -h, --help               Show this help and exit.

Environment:
  ANDROID_NDK / NDK        Override the NDK path
                           (default: ${DEFAULT_NDK}).
  JOBS                     Parallel build jobs (default: nproc).

Examples:
  $(basename "$0")                       # single phone, reuse existing build
  $(basename "$0") --build               # cross-compile then run
  $(basename "$0") -s 1a2b3c4d           # pick a device when several attached
EOF
}

# --- argument parsing --------------------------------------------------------
need_value() { [[ $# -ge 2 && -n "${2:-}" ]] || die "option '$1' requires a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s)                 need_value "$@"; SERIAL="$2"; shift 2 ;;
    --build)            DO_BUILD=1; shift ;;
    --rebuild)          DO_REBUILD=1; DO_BUILD=1; shift ;;
    --abi)              need_value "$@"; ABI="$2"; shift 2 ;;
    --android-platform) need_value "$@"; ANDROID_PLATFORM="$2"; shift 2 ;;
    --device-filter)    need_value "$@"; DEVICE_FILTER="$2"; shift 2 ;;
    --gtest-filter)     need_value "$@"; GTEST_FILTER="$2"; shift 2 ;;
    --keep)             KEEP=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    --)                 shift; break ;;
    -*)                 usage; die "unknown option: $1" ;;
    *)                  usage; die "unexpected argument: $1" ;;
  esac
done

# --- derived paths -----------------------------------------------------------
NDK_HOME="${ANDROID_NDK:-${NDK:-$DEFAULT_NDK}}"
# arm64-v8a -> build-android-arm64 (matches agent-docs); other ABIs keep name.
BUILD_DIR="${REPO_ROOT}/build-android-${ABI/-v8a/}"
BIN="${BUILD_DIR}/tests/vulkore_tests"
KERNELS_SRC="${REPO_ROOT}/tests/kernels"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 6)}"

# adb command array — every adb call goes through "${ADB[@]}" so -s is uniform.
ADB=(adb)

# --- cleanup trap ------------------------------------------------------------
DEVICE_READY=0
cleanup() {
  if [[ "$KEEP" -eq 1 ]]; then
    [[ "$DEVICE_READY" -eq 1 ]] && log "--keep set: leaving ${DEVICE_DIR} on device"
    return 0
  fi
  if [[ "$DEVICE_READY" -eq 1 ]]; then
    log "cleaning up ${DEVICE_DIR} on device"
    "${ADB[@]}" shell "rm -rf '${DEVICE_DIR}'" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# =============================================================================
# 1. Preflight: adb present, server up, device selected & authorized.
# =============================================================================
command -v adb >/dev/null 2>&1 || die \
  "'adb' not found on PATH. Install the Android platform-tools (e.g.
   'sudo apt install android-tools-adb', or unzip Google's platform-tools and
   add it to PATH), then retry."

log "starting adb server"
adb start-server >/dev/null 2>&1 || die "failed to start adb server"

if [[ -n "$SERIAL" ]]; then
  ADB+=(-s "$SERIAL")
else
  # Auto-select: exactly one authorized ('device' state) device -> use it.
  ready=()
  others=()
  while IFS=$'\t' read -r s st; do
    [[ -z "$s" ]] && continue
    if [[ "$st" == "device" ]]; then
      ready+=("$s")
    else
      others+=("$s ($st)")
    fi
  done < <("${ADB[@]}" devices | awk 'NR>1 && NF>=2 {print $1"\t"$2}')

  if [[ ${#ready[@]} -eq 1 ]]; then
    SERIAL="${ready[0]}"
    ADB+=(-s "$SERIAL")
  elif [[ ${#ready[@]} -gt 1 ]]; then
    log "multiple authorized devices connected:"
    for s in "${ready[@]}"; do printf '    %s\n' "$s" >&2; done
    die "several devices attached — re-run with -s <serial> to pick one."
  else
    if [[ ${#others[@]} -gt 0 ]]; then
      log "device(s) present but not usable:"
      for o in "${others[@]}"; do printf '    %s\n' "$o" >&2; done
      die "no authorized device. On the phone: enable USB debugging and accept
   the 'Allow USB debugging?' RSA prompt (tap 'Always allow from this
   computer'). If it says 'offline', unplug/replug or run 'adb kill-server'."
    fi
    die "no devices/emulators found. Plug the phone in with a data-capable USB
   cable, enable Developer Options + USB debugging, and accept the RSA prompt."
  fi
fi

# Final authorization gate on the selected device.
state="$("${ADB[@]}" get-state 2>/dev/null || true)"
if [[ "$state" != "device" ]]; then
  die "selected device is in state '${state:-unknown}', not 'device'. Enable USB
   debugging and accept the RSA authorization prompt on the phone, then retry."
fi
log "using device: ${SERIAL:-<default>}"

# =============================================================================
# 2. Device info + ABI sanity.
# =============================================================================
prop() { "${ADB[@]}" shell "getprop $1" 2>/dev/null | tr -d '\r'; }
DEV_MODEL="$(prop ro.product.model)"
DEV_RELEASE="$(prop ro.build.version.release)"
DEV_SDK="$(prop ro.build.version.sdk)"
DEV_ABI="$(prop ro.product.cpu.abi)"
log "model=${DEV_MODEL:-?}  Android=${DEV_RELEASE:-?} (API ${DEV_SDK:-?})  abi=${DEV_ABI:-?}"

if [[ -n "$DEV_ABI" && "$DEV_ABI" != "$ABI" ]]; then
  warn "device primary ABI '${DEV_ABI}' does not match build ABI '${ABI}'. The
   binary may not run. Pass --abi ${DEV_ABI} (and --build) to match."
fi

# Non-fatal heads-up: volk needs the device's Vulkan loader present.
if ! "${ADB[@]}" shell "ls /system/lib64/libvulkan.so" >/dev/null 2>&1; then
  warn "/system/lib64/libvulkan.so not found on device — Context creation will
   fail if the device has no Vulkan driver."
fi

# =============================================================================
# 3. Build (if requested or binary missing).
# =============================================================================
if [[ "$DO_REBUILD" -eq 1 && -d "$BUILD_DIR" ]]; then
  log "wiping build dir ${BUILD_DIR}"
  rm -rf "$BUILD_DIR"
fi

if [[ "$DO_BUILD" -eq 1 || ! -f "$BIN" ]]; then
  if [[ "$DO_BUILD" -ne 1 && ! -f "$BIN" ]]; then
    log "no test binary at ${BIN}; building it now (pass --build explicitly to force)."
  fi
  TOOLCHAIN="${NDK_HOME}/build/cmake/android.toolchain.cmake"
  [[ -f "$TOOLCHAIN" ]] || die \
    "NDK toolchain not found at ${TOOLCHAIN}. Set ANDROID_NDK or NDK to a valid
   Android NDK (default expects ${DEFAULT_NDK})."
  log "cross-compiling: abi=${ABI} platform=${ANDROID_PLATFORM} ndk=${NDK_HOME}"
  cmake -G Ninja -S "$REPO_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
        -DCMAKE_BUILD_TYPE=Release
  ninja -C "$BUILD_DIR" -j"$JOBS"
fi

[[ -f "$BIN" ]] || die \
  "test binary missing at ${BIN}. Re-run with --build to cross-compile it."

# =============================================================================
# 4. Push binary + .spv kernels.
# =============================================================================
shopt -s nullglob
spv_files=("$KERNELS_SRC"/*.spv)
shopt -u nullglob
[[ ${#spv_files[@]} -gt 0 ]] || die \
  "no .spv kernel fixtures found in ${KERNELS_SRC}. The tests need them."

log "preparing ${DEVICE_DIR} on device"
"${ADB[@]}" shell "rm -rf '${DEVICE_DIR}' && mkdir -p '${DEVICE_DIR}/kernels'" \
  || die "failed to create ${DEVICE_DIR} on device"
DEVICE_READY=1   # from here on, cleanup trap will remove the temp dir

log "pushing test binary ($(basename "$BIN"))"
"${ADB[@]}" push "$BIN" "${DEVICE_DIR}/vulkore_tests" >/dev/null \
  || die "failed to push test binary"
"${ADB[@]}" shell "chmod 755 '${DEVICE_DIR}/vulkore_tests'" \
  || die "failed to chmod test binary"

log "pushing ${#spv_files[@]} kernel fixtures -> ${DEVICE_DIR}/kernels/"
for f in "${spv_files[@]}"; do
  "${ADB[@]}" push "$f" "${DEVICE_DIR}/kernels/" >/dev/null \
    || die "failed to push kernel: $f"
done

# =============================================================================
# 5. Run the suite (VULKORE_KERNEL_DIR -> pushed kernels; propagate exit code).
# =============================================================================
remote_cmd="VULKORE_KERNEL_DIR='${DEVICE_DIR}/kernels'"
[[ -n "$DEVICE_FILTER" ]] && remote_cmd+=" VULKORE_DEVICE='${DEVICE_FILTER}'"
remote_cmd+=" '${DEVICE_DIR}/vulkore_tests'"
[[ -n "$GTEST_FILTER" ]] && remote_cmd+=" --gtest_filter='${GTEST_FILTER}'"

log "running: ${remote_cmd}"
log "----------------------------------------------------------------------"
set +e
"${ADB[@]}" shell "$remote_cmd"
rc=$?
set -e
log "----------------------------------------------------------------------"
if [[ "$rc" -eq 0 ]]; then
  log "SUCCESS — gtest exit code 0"
else
  log "FAILURE — gtest exit code ${rc}"
fi

# cleanup runs via the EXIT trap; preserve the gtest exit code.
exit "$rc"
