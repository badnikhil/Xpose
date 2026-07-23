# repo-layout — third-party submodules and tree structure

```
include/vulkore/            public headers (vulkore.hpp umbrella)
src/                      runtime implementation
tests/                    googletest suite + kernels/ fixtures (.cl + committed .spv)
kernels/                  LLM kernel modules (.cl + committed .spv)
examples/llm/             decode loop, model loader, benchmarks/harnesses (not in CMake)
android/                  the chat app (Kotlin/Compose + llm_jni.cpp)
scripts/run-android.sh    on-device test driver
benchmarks/               benchmark suite
agent-docs/               this knowledge base (committed in-repo)
third_party/              git submodules (below) + gitignored spirv-tools-build/
build*/                   gitignored build dirs
```

## Submodule pins (`third_party/`)

| Submodule | Pin | Role |
|---|---|---|
| `Vulkan-Headers` | `8d6039a4` | API headers (no system SDK assumed) |
| `volk` | `3b005543` | loader; dlopens libvulkan at runtime |
| `VulkanMemoryAllocator` | `3aa92122` | the only hard runtime dep |
| `googletest` | `a25f4357` | test framework |
| `clspv` | `0f162d74` | build-time kernel compiler; `.gitmodules` sets `ignore = untracked` for it (its fetched deps are untracked inside it) |
| `SPIRV-Tools` | `58dc0b3e` | reference clone — do NOT build (headers mismatch with clspv's pin) |
| `clvk` | `1ba5a1c7` | reference reading only (reflection parser) |
| `kompute` | `2cf421e6` | reference reading only |

- clspv's own deps (`llvm` `140fc5aa`, `SPIRV-Headers` `b824a462`,
  `SPIRV-Tools` `fb747184`) are **fetched clones, not submodules**:
  `python3 utils/fetch_sources.py --shallow` inside `third_party/clspv`
  (~2.9 GB).
- `third_party/spirv-tools-build/` is a **gitignored** out-of-tree build of
  clspv's pinned SPIRV-Tools providing `tools/spirv-{as,dis,val}` for the
  fixture pipeline — recipe in `CLAUDE.md`. Build dir stays OUTSIDE the clspv
  tree so clspv's git state is untouched.
- Submodule histories may be shallow; `git submodule update --init --recursive`
  on a fresh clone fetches from the public upstreams.

## Fresh-checkout bring-up

```sh
git submodule update --init --recursive
cd third_party/clspv && python3 utils/fetch_sources.py --shallow   # only needed for clspv work
# spirv-as/dis/val: rebuild recipe in CLAUDE.md (§ Third-party layout)
```

## Rules

- Submodule remotes point at their public upstreams — **fetch fine, never
  push, never commit inside them.**
- Don't move submodule pins without recording why in `CLAUDE.md` and here —
  the pin table above is the reproducibility record.
- `agent-docs/` is committed in-repo and must stay current with every
  change/decision/finding.
