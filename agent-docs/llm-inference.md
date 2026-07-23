# LLM inference — Gemma 3 1B end to end on Vulkore

`examples/llm/decode.{hpp,cpp}` drive Gemma 3 1B through the Vulkore runtime:
GGUF load (`examples/llm/model/`, see `gguf.md`) → weight pack/upload → prefill
→ decode loop → sampling. Kernels live in `kernels/llm_transformer.cl` (25),
`kernels/llm_matvec.cl` (16), `kernels/llm_sampling.cl` (27), `kernels/llm.cl`
(5, the original baseline matvecs kept as references). All performance numbers
and the measurement lessons live in `llm-performance.md`; this doc is the
design: what runs, in what order, and why it is shaped this way.

## Model facts (verified against the real file)

`gemma-3-1b-it-Q4_K_M.gguf`: arch `gemma3`, 26 layers, hidden 1152, ffn 6912,
4 q heads, **1 kv head (MQA)**, head_dim **256** (from `attention.key_length`,
NOT hidden/heads), vocab 262144, ctx 32768, rms_eps 1e-6.

- RMSNorm uses the **(1 + w)** weight convention; GeGLU with tanh-approximate
  GELU (`gelu_pytorch_tanh`); QK-norm on q and k before RoPE; RoPE is
  **rotate-half** (pairs `d` with `d + head_dim/2`), not interleaved — choosing
  wrong is silent: the model runs and just loses positional sense.
- **Sliding-window attention**: extent 512 (from `gemma3.attention.
  sliding_window`), layer `i` is global iff `(i+1) % 6 == 0` (layers 5/11/17/23
  see the whole context, the other 22 only the last 512 positions). The 5:1
  alternation is not in the file; it is a `DecodeConfig` policy field.
- rope theta 1e6 on global layers, 1e4 on local ones.
- **No `output.weight`** — the LM head is tied to `token_embd.weight`.
  Attention scale `1/sqrt(256)`; embedding scale `sqrt(1152)`.
- The file has NO `gemma3.rope.scaling.*` metadata: Gemma 3 **1B** is natively
  32k; the factor-8 rope scaling belongs to the 128k 4B/12B/27B variants. Not a
  long-context gap for this model.

## The bug that shaped the tooling: GGUF norm weights already hold (1 + w)

llama.cpp's converter does `data_torch + 1` on every `norm.weight` at
CONVERSION time, so a GGUF already stores the finished multiplier and
`rmsnorm_apply_gemma`'s `(1 + w)` applies a second one. Symptom: fluent,
grammatical, completely wrong text (residual stream growing 55x in layer 0) —
not a crash, not a NaN. The tell was in the weights all along: norm weights
with RMS 6.2 and max 56.75 are finished multipliers, not offsets-from-identity
(~0–1). Fix is host-side: subtract 1.0 from every norm weight at load
(`DecodeConfig::gguf_norm_includes_bias`); the kernels correctly implement the
HF convention and are untouched.

**What found it: a full fp32 CPU forward pass (`--cpu-ref`) that reads the same
GGUF and never touches the GPU.** It reproduced the garbage, which eliminated
the GPU path, the kernels, the packing and the quantisation in one step. A CPU
reference is the bisection tool for this class of work, not a luxury — without
it, "loop wrong" vs "kernels wrong" vs "int4 too lossy" are indistinguishable.
It is committed, threaded over output columns (bit-identical results), and
skips the lm_head on uncompared positions so a deep comparison is usable.

## Design rules (each learned the hard way)

- **One token = ONE `vkQueueSubmit`.** A per-launch submit costs 0.31 ms on an
  Adreno 840; the loop issues 838 dispatches/token, which per-launch would be
  ~260 ms of pure submission. Everything is recorded into one `vulkore::Batch`.
- **An upload is also a submit** (Adreno `DeviceLocal` is not host-visible, so
  uploads take the staging path + wait). Hence: RoPE computes sin/cos inline
  from a `pos` push constant (`rope`), not from per-token uploaded angle tables
  (`rope_cached`); and the **embedding is a CPU memcpy** — one GGUF row
  dequantised on the CPU, scaled by sqrt(hidden), memcpy'd into a `HostVisible`
  residual buffer. Zero dispatches, zero submits (a GPU `embed_lookup` would
  need the 262144×1152 table as fp32 = 1.2 GB resident).
- **No `__local`** — clspv emits `WorkgroupVariableSize` and `Program` throws.
  Every reduction is serial-per-thread or two-pass through global memory. This
  constraint has blocked the textbook workgroup reduction repeatedly; stop
  re-trying it.
- **No `size_t` indexing** (promotes to Int64 capability); `uint` suffices
  (worst case 3.0e8 for the embedding table).
- `tanh`/sigmoid go through clamped `exp` (|z|>10 clamps first; `exp(-2z)`
  overflow otherwise NaNs). `pow` is avoided — rope inverse frequency is
  `exp2(-log2(theta)*frac)` — because `pow`/`acos`/`atan2` have silently not
  executed on Adreno in this repo.
- Grid = GLOBAL thread count, rounded up to whole workgroups: every kernel
  bound-checks. PODs share one push-constant block ≤128 B (largest here is 28 B).
- `Buffer` rejects element types not 4-byte aligned — int4 nibbles are packed
  into `uint`, quantised blobs are viewed as `uint*`.
- **A clean compile proves nothing.** Kernels have compiled, passed spirv-val,
  bound and dispatched without error and silently not executed. Every harness
  here proves writes (see `llm-performance.md` §liveness).
- q/k/v (and gate/up) are separate matvecs, not fused, **because a descriptor
  binding carries no byte offset** and downstream kernels need their inputs at
  buffer start. Identical bytes moved, 3 extra dispatches/layer; a `split3`
  kernel or an offset argument would recover 78 dispatches/token.

## Decode step anatomy

Dispatch budget per generated token (all in one submit):

| stage | dispatches | note |
|---|---|---|
| RMSNorm ×4 (two-pass) | 8 | Gemma 3 wraps BOTH sublayers: input/post_attention + ffn/post_ffw norms |
| QK-norm | 2 | `rmsnorm_heads_gemma` on q and k |
| matvec | 12 | q k v o gate up down; 5 of them split-k + reduce |
| RoPE | 2 | in-place, pos/theta push constants |
| kv_append | 2 | k and v |
| attention | 3 | scores + softmax + apply (6 when the parallel deep-span forms engage) |
| GeGLU | 1 | |
| residual | 2 | |
| **per layer** | **32** | ×26 = 832 |
| final norm | 2 | |
| lm_head + sampling harvest | 4 | skipped during prefill (−180 MiB traffic/token) |
| **total** | **838** | 916 at depth with the parallel attention forms |

Weight traffic per token: **536.3 MiB** (int4 nibbles + bf16 block scales), of
which lm_head is 180 MiB. `--dry-run` prints the full plan and byte model with
no GPU or model file.

### Two-pass RMSNorm is the norm of record

The one-pass `rmsnorm`/`rmsnorm_gemma` (every thread redundantly recomputes the
sum of squares) is 6.4x SLOWER than `rmsnorm_sumsq` + `rmsnorm_apply_gemma`
(8.2 vs 1.3 ms over a token's 52 norms): inside a Batch a dispatch costs
~6 µs, so the second dispatch is nearly free and the redundant O(n²) work is
not. The one-pass forms are kept for norms standing alone outside a Batch,
where submit cost genuinely dominates.

### KV cache

Layout `k[(t*kv_heads + h)*head_dim + d]` — appending a position is the
contiguous write `kv_append` does. **One K and one V buffer per layer** (52
buffers), so no buffer approaches `maxStorageBufferRange`. MQA makes context
affordable: 1 kv head ⇒ 53,248 B/position across all 26 layers (4 heads would
be 4x that).

| max_seq | KV total |
|---|---|
| 2048 | 104 MiB |
| **8192 (default)** | **416 MiB** |
| 32768 (model max) | 1.62 GiB |

Capacity is ~free in TIME (gpu ms/token is flat within noise from max_seq 2048
to 65536 — nothing scans max_seq, only the live position; even 3.3 GiB
allocates cleanly). The default is bounded by **throughput at depth**, not by
what fits: 8192 keeps total residency under 1 GiB (536 MiB weights + 416 MiB
KV) and decodes at ~39 tok/s at full depth; 32768 would land near 5 tok/s.
Raising the default needs a fresh `--curve` run with the thermal control, not
extrapolation. When the cache fills at 8192 the session clears history with a
message instead of failing silently.

### Sliding window: an offset, not a mask

`attention_scores`/`attention_apply` (and the `_mq4` forms) take
`(span, kv_start)`:

```
w        = layer_is_global(layer) ? 0 : sliding_window
kv_start = (w == 0 || seq_len <= w) ? 0 : seq_len - w
span     = seq_len - kv_start
```

A decode query is always the newest position, so its window is a contiguous
**suffix** of the cache — the correct implementation is therefore also the
cheaper one (22 of 26 layers stop growing with depth). A `-INFINITY` mask would
be equally correct and exactly as slow as the bug it replaced. `kv_start=0,
span=seq_len` reproduces unwindowed behaviour bit-for-bit; below position 512
the window is a no-op. (Batched prefill needs the real 2-D mask;
`attention_scores_prefill` has causal + window folded into the score kernel —
one register compare instead of an extra masking dispatch; the diagonal stays
live so a row can never be fully masked into a NaN softmax.)

**The oracle trap, recorded because it generalises:** the sliding window was
declared in `decode.hpp` and read by nothing — and the CPU reference had the
same omission, so a deep GPU-vs-CPU `--verify` compared two identical mistakes
and printed a clean 0.95 correlation. Against a *corrected* oracle the
unwindowed path collapses to 0.744–0.878 corr / 1-of-4 top-1 at position ~1071
(fixed: 0.929–0.957, 3-of-4). Hence `--verify-cpu-window N`, a sensitivity
control that desyncs the two windows on purpose: if the comparison cannot tell
them apart it is not measuring the window. Related trap: `DecodeConfig::
from_model()` copies caller-set policy fields back via an explicit allowlist,
which silently rots when a field is added — `--window` was a no-op until
`sliding_window`/`global_every` were added to it.

### Attention kernels at depth

Serial-over-a-long-span is the recurring poison (see also argmax below). Three
parallel forms engage above `DecodeConfig::attn_min_span = 128` — below it the
serial forms win because 2–3 extra dispatches cost more than the serial work
they remove:

- `softmax_rows_partial/_finish/_scale` — two-pass stable softmax, strided
  chunks, `nparts ≈ sqrt(span)` (pass 1 costs span/nparts on n_rows*nparts
  threads, pass 2 costs nparts; the root balances them).
- `attention_scores_mq4` — exploits `kv_heads == 1`: one thread per key
  position reads the key ONCE as `float4` and accumulates 4 dot products in
  named scalar registers (a private array with a dynamic index makes clspv
  allocate real per-thread scratch).
- `attention_apply_mq4` + `attention_apply_reduce` — same sharing, span split
  16 ways to restore occupancy (1024 threads would starve an Adreno 840).

## Kernel contract

Resolved by name at Program load; a missing kernel throws with the name and
.spv path. Grid = global threads. Full signatures live in the `.cl` sources;
the contracts that bind modules together:

- **Scores layout** `[q_head][span]`, row stride = span, rebuilt every step.
- **GQA mapping** `kv_head(h) = h / (n_heads/kv_heads)`.
- Matvec layouts (`kernels/llm_matvec.cl`): default **KM4 + bf16 scales**
  (`mv_km4_bs`, `mv_km4_bs_split`): nibble plane `uint4[(g*nblk + b)*64 + t]`
  with `g=j/64, t=j%64, b=k/32`, component `(k>>3)&3`, nibble `(k&7)*4`; scales
  bf16 packed four per `uint2` at `sc[(b/4)*cols + j]` (requires rows % 128 == 0
  — the loader THROWS by tensor name on a partial quad; the kernel would
  silently drop it). `--matvec-layout km4|col` selects the fp32-scale or
  column-nibble baselines. `rows % 32 == 0`, `cols % 64 == 0` everywhere.
- Split-k is applied when `cols < 2048` (q/k/v/o_proj/down), skipped for wide
  layers, and capped so `cols*split/64 ≤ 65535` workgroups (the limit that
  makes split16 impossible on lm_head).
- bf16 scale decode is `as_float(w << 16)` — one shift, full fp32 exponent
  range. That is why bf16 and not fp16 (fp16 needs a ~15-instruction decoder
  and can overflow/flush on an unusual scale).
- The packer picks the error-minimising **neighbour** of the two bf16
  candidates per block (`best_bf16_scale`), which measures *better* than the
  exact fp32 scale (0.995x dot-product error) — `amax/7` was never the
  error-minimising choice, and a scale error is common to a whole block so it
  does not average down like independent nibble error.

## Sampling

Greedy: 3-level argmax tree (`argmax_partial` → `argmax_reduce` →
`argmax_final`, 262144 → 4096 → 64 → 1), 8-byte readback from mapped memory.
The single-thread `argmax` fallback is kept but costs 31.5 ms — one serial
dispatch over the vocab once cost more than the other 837 dispatches combined.
"Single-threaded" is not the problem; "single-threaded over the vocab" is
(serial over 40 elements is nothing).

Temperature/top-k (the shipped default — greedy puts Gemma 3 1B into
repetition loops and is not shippable): **`topk4_partial`** keeps the best
FOUR of each of 1024 strided chains in scalar registers and writes 32 KiB into
`HostVisible` memory inside the token's existing Batch — one dispatch, zero
extra submits (it replaces the argmax tree, so the dispatch count drops
838→836). The host finishes top-k + temperature + softmax + multinomial over
4096 candidates.

- Best-4-per-chain is **exact** on every measured token (a true top-64 member
  is lost only if ≥5 land in one chain, ~7e-6); best-1-per-chain (reusing
  `argmax_partial`) is measurably lossy — mean 61.9/64, worst 58/64. The extra
  compares are free because the pass is bounded by reading the 1 MiB of logits.
- The host cost is linear in CANDIDATE count, not in submits — hence 1024
  chains, not 4096 (which costs 0.94 ms of host work). `--topk-parts N`
  overrides.
- `--sample-check` scores both harvests per token against `std::partial_sort`
  on the GPU's own logits; the lossy m=1 column doubles as a negative control
  proving the kernel executes.
- **Top-p runs on the host, by design**: the token id must reach the CPU anyway
  (`embed_lookup`/the embedding memcpy need it), top-k already produced the
  sort (rank-select emits sorted-descending for free), and a full-vocab top-p
  would need a 262144-element sort — the wrong tool on this ABI. Chaining
  top-k→top-p matches llama.cpp/HF practice; the discarded tail mass is
  negligible at p ≤ 0.95 but it IS an approximation.
- Semantics caveat: with the harvest active, `--top-k 0` means "top-4096", not
  full vocabulary; use `--sample-cpu` for a genuine full-vocab draw.
- The full-vocab GPU chain also exists and is verified (stable softmax via
  3-level max/sum trees, 7 dispatches; `multinomial_strided` samples the
  permuted distribution in 1280 steps reusing the softmax intermediates, no
  normalisation pass) — for pipelines that must stay on-GPU end to end.
  `XP_TOPK_MAX = 64`: larger k is silently clamped.

## Weight load, pack, cache

The constructor dequantises the file's mixed Q8_0/Q5_0/Q4_K/Q6_K to fp32 and
re-quantises to uniform int4 (symmetric, `s = max|v|/7`, block 32), packs KM4 +
bf16 scales, uploads. **This double quantisation is the honest weak point**:
logit correlation vs the fp32 oracle is ~0.95 instead of ~0.999 (`token_embd`
is Q8_0 in the file, crushed to 4 bits, and it is also the LM head). It buys
ONE matvec kernel instead of one per format. The per-format repacked kernels in
`kernels/llm_quant.cl` (`gguf.md`) are the fix; wiring them in is the largest
open quality task (measured trade: ~+3.9 ms/token, `llm-performance.md`).

Load path (`llm-performance.md` has the timings; design here):

- **Packing is threaded over 64-column groups, not tensors** — both layouts
  index by `j/64` so a group range writes a disjoint region with no
  synchronisation. Grain matters: `token_embd` is 32% of all packing work in
  one tensor; at tensor grain it would be the critical path on its own.
- Work is issued in **waves capped at 192 MiB of packed host data** so peak RSS
  stays bounded (pack-everything-then-upload would hold the whole 536 MiB twice).
- **`vulkore::Context` is single-threaded — allocation, upload and teardown
  alike.** Workers fill plain host vectors and never see the Context; the
  owning thread uploads in task order.
- Packed bytes are cached to `<model>.xpack`, `rename(2)`'d from a `.tmp` so a
  partial write is unobservable. The header is paranoid because the failure
  mode is not a corrupt read — it is a cache that loads *successfully* with
  the wrong weights (the same fluent-garbage failure as the norm bug):
  magic/version, model size + mtime, model **fingerprint** (first+last 64 KiB
  hashed — catches same-size replacement, plausible with `adb push`), layout /
  scale format / packer version, entry table equal to the session's own task
  list, contiguity (entries tile the payload exactly), per-entry checksum.
  **Any doubt repacks silently.**
- **`kPackerVersion` is a hand-bumped constant: bump it whenever `quantise()`,
  `best_bf16_scale()`, `pack_group()` or the scale packing changes**, or every
  existing `.xpack` silently becomes wrong weights. This is the one manual
  obligation the design carries and the one that will be forgotten.
- Norm weights are deliberately NOT cached (<500 KiB, ms to dequantise, and
  caching them would drag the `gguf_norm_includes_bias` flag into the header).
- `--verify-cache` re-packs from the model and compares byte-for-byte (183
  tensors, 536.3 MiB); a planted bit-flip is caught independently by it and by
  the runtime checksum. `--repack` forces, `--no-pack-cache` disables,
  `--pack-cache F` relocates.
- The `.xpack` costs 536 MiB of disk next to the 806 MiB GGUF. The Android app
  (`android/app/.../llm_jni.cpp`) uses default `Paths` and writes it next to
  the model in app storage.

## Prefill

Prefill is **decode-shaped** — one position per pass reusing the single-token
path, skipping lm_head (the single biggest saving, 180 MiB/position). This is
the known weakness: llama.cpp's batched prefill is ~11x faster.
`kernels/llm_sampling.cl` already ships the batched-prefill kernel twins
(`softmax_rows_stats/_apply`, masked `attention_*_prefill`, row-wise
rmsnorm/rope/kv_append/`embed_lookup_rows` — token ids from a buffer, since a
push constant cannot express T of them), all pinned to the decode kernels at
max_rel 0; a real batched prefill is host-side work with no kernel gap left.
The prefill softmax keeps a one-thread-per-row reduction deliberately: with
`n_rows = n_heads*n_query` in the thousands the device is saturated and the
elementwise bulk is split out per-element — the opposite conclusion to the
vocab softmax, where n_rows == 1.

## Rejected: static command buffer re-submission

Recording the token's command buffer once and re-submitting would remove the
~0.85 ms/token `record` cost, and was rejected: five kernels take
position-dependent push constants (`rope`/`kv_append` take `pos`;
`attention_scores`/`softmax_rows`/`attention_apply` take span), which are baked
into a recorded buffer — they would all need `_dyn` variants reading from a
buffer; `attention_scores`' grid depends on span, so a static buffer needs
bucketed worst-case grids with in-kernel masking; the upside is ≤5% of a token
and the failure mode is fluent, plausible, wrong text. If ever done, that
bucketed-grid + `_dyn`-kernel design is the shape, verified at several sequence
lengths.

## Verification surface

```
--verify [--verify-steps N]    GPU vs fp32 CPU oracle over ONE shared token
                               sequence (diverging sequences compare logits for
                               different contexts and measure nothing)
--verify-skip N --prompt-file  push the comparison past position 512
--verify-cpu-window N          window sensitivity control (see above)
--cpu-ref                      fp32 CPU only; VULKORE_LLM_DEBUG=1 prints per-layer RMS
--selftest                     host packer vs matvec kernel at real shapes, no model
--verify-cache                 .xpack vs fresh pack, byte-for-byte
--sample-check                 GPU top-k harvest vs exact CPU top-k
--dry-run                      dispatch plan + byte model, no GPU
```

Reading `--verify`: top-1 agreement with corr 0.93–0.96 is the signature of a
correct loop on lossier (double-quantised) weights; a loop bug gives corr near
zero. Correlation decays with depth (~0.82–0.95 at position 4176) — that is
accumulated quantisation error, not a loop defect (proved by byte-identical
greedy text across kernel variants at depth). A 4-position greedy probe has a
~0.004 correlation realisation floor (any perturbation reshuffles rounding) —
weight-format changes need `--verify-steps ~20`, not the default.
`--selftest` judges error against the RMS of the output vector, not
per-element: near-zero dot products are relatively "wrong" by 1e-3 while
absolutely fine, and a real layout bug is O(1), not 1e-6.

## Building

`examples/llm/*.cpp` are not wired into CMake; each file's header comment
carries its g++ line. Typical:

```sh
INC="-Iinclude -Iexamples/llm -Ithird_party/Vulkan-Headers/include \
     -Ithird_party/volk -Ithird_party/VulkanMemoryAllocator/include"
g++ -std=c++20 -O2 $INC -DVMA_STATIC_VULKAN_FUNCTIONS=0 \
    -DVMA_DYNAMIC_VULKAN_FUNCTIONS=1 \
    examples/llm/decode.cpp examples/llm/model/gguf.cpp \
    examples/llm/model/tokenizer.cpp build/libvulkore.a -ldl -o build/llm_decode

./build/llm_decode --model models/gemma-3-1b-it-Q4_K_M.gguf --chat \
                   --prompt "Name three prime numbers." -n 60
```

Kernel .spv regeneration: `python3 tests/kernels/regenerate.py kernels/<name>.cl`
(local clspv preferred, godbolt fallback). **A regenerated .spv is a confound
bundled with every kernel edit** — different clspv revisions emit differently
sized binaries from identical source; separate the recompile from the code
change before claiming bit-exactness (run the new host binary against both the
committed and the regenerated module).

## Known gaps

- Per-format quant kernels not wired into decode (see above) — the open
  quality item.
- Prefill not batched (kernels exist, host loop does not); no chunking policy.
- No repetition penalty; `--top-k 0` semantic caveat above.
- fp32 activations throughout (unmeasured as a bottleneck; weights dominate).
- `multinomial_strided` verified at n=8192, not the full 262144 (no
  size-dependent logic, but untested at the real shape).
