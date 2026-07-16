#!/usr/bin/env python3
"""Regenerate the .spv kernel fixtures from the .cl sources.

Uses Compiler Explorer's hosted clspv (no local clspv build needed):
  POST https://godbolt.org/api/compiler/<COMPILER_ID>/compile
clspv on godbolt emits SPIR-V *disassembly text*; we reassemble it with
spirv-as and validate with spirv-val (SPIRV-Tools binaries, see SPIRV_TOOLS
below or pass --spirv-tools-bin).

The committed .spv binaries are the source of truth for the test suite so
tests never touch the network; run this only to refresh fixtures.

Usage: ./regenerate.py [--spirv-tools-bin DIR] [kernel.cl ...]
"""

import argparse
import json
import pathlib
import subprocess
import sys
import urllib.request

# Compiler Explorer compiler id (language "openclc"), verified 2026-07-16:
#   id "clspv", name "clspv (trunk)".
COMPILER_ID = "clspv"
GODBOLT = "https://godbolt.org"

# One predictable ABI for every fixture:
#   -pod-pushconstant        POD kernel args -> a single push-constant block
#                            (reflection: ArgumentPodPushConstant)
#   -uniform-workgroup-size  REQUIRED with CL3.0 + -pod-pushconstant: without
#                            it clspv emits an implicit RegionOffset push
#                            constant as a SECOND PushConstant interface ->
#                            fails Vulkan validation (max one PC block per
#                            entry point). vulkore::launch always dispatches
#                            whole workgroups (rounds the grid up; kernels
#                            bound-check), so uniform workgroup size holds.
#   -cl-std=CL3.0            + -inline-entry-points (required by CL3.0 mode)
#   -spv-version=1.3         SPIR-V 1.3 = the Vulkan 1.1 Android-floor target
CLSPV_ARGS = ("-cl-std=CL3.0 -inline-entry-points -pod-pushconstant "
              "-uniform-workgroup-size -spv-version=1.3")

# Default: the clspv-pinned SPIRV-Tools built out-of-tree into the repo's
# gitignored third_party/spirv-tools-build/ (see agent-docs/environment.md);
# override with --spirv-tools-bin. Resolved relative to this script so the
# repo stays relocatable.
SPIRV_TOOLS = str((pathlib.Path(__file__).resolve().parent
                   / "../../third_party/spirv-tools-build/tools").resolve())
TARGET_ENV = "vulkan1.1"


def compile_on_godbolt(source: str) -> str:
    body = json.dumps({
        "source": source,
        "options": {
            "userArguments": CLSPV_ARGS,
            "filters": {
                "binary": False, "commentOnly": False, "demangle": False,
                "directives": False, "execute": False, "intel": False,
                "labels": False, "libraryCode": False, "trim": False,
            },
        },
    }).encode()
    req = urllib.request.Request(
        f"{GODBOLT}/api/compiler/{COMPILER_ID}/compile", data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    if r.get("code") != 0:
        err = "\n".join(l.get("text", "") for l in r.get("stderr", []))
        raise RuntimeError(f"clspv failed (code {r.get('code')}):\n{err}")
    return "\n".join(l.get("text", "") for l in r.get("asm", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spirv-tools-bin", default=SPIRV_TOOLS)
    ap.add_argument("kernels", nargs="*")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    tools = pathlib.Path(args.spirv_tools_bin)
    cls = ([pathlib.Path(k) for k in args.kernels]
           or sorted(here.glob("*.cl")))

    for cl in cls:
        print(f"== {cl.name}: godbolt {COMPILER_ID} [{CLSPV_ARGS}]")
        disasm = compile_on_godbolt(cl.read_text())
        spv = cl.with_suffix(".spv")
        subprocess.run(
            [tools / "spirv-as", "--target-env", TARGET_ENV,
             "--preserve-numeric-ids", "-o", spv, "-"],
            input=disasm.encode(), check=True)
        subprocess.run(
            [tools / "spirv-val", "--target-env", TARGET_ENV, spv],
            check=True)
        print(f"   -> {spv.name} ({spv.stat().st_size} bytes, "
              f"spirv-val {TARGET_ENV} OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
