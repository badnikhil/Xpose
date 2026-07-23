# LLM performance — measured numbers, optimisation record, baselines

All headline numbers are **phone-measured**: OnePlus 15 (SM8850, Adreno 840,
Qualcomm proprietary Vulkan driver), Gemma 3 1B, `gemma-3-1b-it-Q4_K_M.gguf`.
The ThinkPad's Adreno X1-85 (Mesa turnip) is the iteration device only — see
§turnip-proxy below for exactly how far it can be trusted. Design context for
every kernel named here is in `llm-inference.md`.

## Headline: decode is a curve, not a number

Throughput depends on context depth. One binary, one 8100-token generation,
temp/top-k sampling:

| KV position | tok/s |
|---|---|
| 0–127 | 70.9 |
| 128–511 | 64.1 |
| 512–1023 | 61.1 |
| 1024–2047 | 60.7 |
| 2048–4095 | 60.6 |
| 4096–8191 | 39.2 |

**Positions 128–4095 are flat to within 6%** — the band a real conversation
lives in. Deep/shallow ratio 1.81x (was 2.62x before the attention-flatness
work); with every layer windowed (`--global-every 1000` control) it is 0.98x —
flat from the first token to the 8000th — so the remaining 1.81x is entirely
the 4 global layers, which must scale with position. Other steady-state costs
per token: `record` ~0.85 ms host, `sample` ~0.08 ms, gpu ~13.4 ms shallow.
Model load: **1.90 s** cold (cache hit), 3.07 s cold first-run pack, on the
phone; 0.72 s laptop.

Prefill: ~56 tok/s (decode-shaped, one position per pass) — the known weakness.

## Baselines (same phone, same model file, md5-verified identical)

| Runtime | Backend | prefill tok/s | decode tok/s |
|---|---|---|---|
| llama.cpp (proper ARM build) | CPU | 54.79 | 29.08 |
| llama.cpp OpenCL | Adreno 840 | **638.68** | 29.55 |
| LiteRT-LM (Google, int4) | Adreno 840 | — | 48 |
| **Vulkore** | Adreno 840 | 56.1 | **70.9 shallow · ~60.6 flat · 39.2 deep** |

Honest framings:

- **Decode: ~2x llama.cpp's own Adreno OpenCL backend** on the same phone. Two
  asterisks: Vulkore reads fewer bytes/token (536.3 MiB uniform int4+bf16 scales
  vs the 762.49 MiB mixed-quant file llama.cpp streams — most of an advantage
  by itself in a memory-bound workload), and Vulkore's int4 is quantised twice
  (fidelity ~0.95 corr vs ~0.999). Byte-normalising the older 50.9 tok/s figure
  against 762.49 MiB gave ~1.35x; going native-format removes both asterisks at
  a measured ~+3.9 ms/token.
- **vs LiteRT-LM: parity with a slight edge, not a beat.** Their 48 is one
  reading of an in-app benchmark at unknown context depth; Vulkore's own runs
  spread wider than the gap depending on reply length and thermal state. The
  defensible claim: a general-purpose GPU compute layer lands in the same class
  as the vendor's tuned LLM runtime on the vendor's silicon.
- **Prefill: llama.cpp is ~11x faster** (638.68 vs 56.1) — compute-bound and
  batched vs decode-shaped. Expect the time-to-first-token question.
- **The GPU barely helps llama.cpp at decode** (29.55 vs 29.08 CPU): decode at
  batch 1 is memory-bound, so their 11x prefill win and ~0% decode win are the
  same fact from two sides.

llama.cpp method notes (they will bite again):

- `llama-bench` **falls back to CPU silently** — check the backend column every
  run; the OpenCL rows print `QUALCOMM Adreno(TM) 840` at startup.
- llama.cpp probes ARM features with `-mcpu=native`, which cannot work
  cross-compiling under qemu: `HAVE_DOTPROD - Failed` silently produces a plain
  armv8-a binary with no dotprod/i8mm (a crippled build measured 21.05 tok/s
  prefill — a floor, not a CPU result). Fix:
  `-DGGML_CPU_ARM_ARCH=armv8.7-a+dotprod+i8mm+fp16` (no `+sve`: Oryon cores
  lack SVE). The Android toolchain also confines `find_package` to its sysroot,
  so `OpenCL_INCLUDE_DIR`/`OpenCL_LIBRARY` must be passed explicitly. On
  Qualcomm, `/vendor/lib64/libOpenCL.so` IS the Adreno implementation (no ICD
  vendors dir); the ICD loader is needed only to satisfy the linker.

## The central lesson: decode is dispatch-bound, then bandwidth-saturated

1. **A submit costs ~0.31 ms** on the Adreno 840 (flat, measured with a
   do-nothing kernel at 1/130/260 dispatches). At ~130 matvecs/token,
   per-launch submission was 70% of the token. Fix: `vulkore::Batch` — one
   command buffer, one `vkQueueSubmit`. At the real 838 dispatches/token,
   batching is worth ~11x, not the 36% first measured.
2. **Inside a Batch a dispatch costs ~5.6 µs** — ~50x cheaper than a submit.
   At 130 dispatches that is negligible and "dispatch count barely matters,
   redundant work does" was the right guidance (split-k good, redundant
   one-pass norms bad). At 838 dispatches it is **4.70 ms/token = 35% of the
   GPU budget** — obtained by solving `gpu = c + bytes/BW` across two measured
   per-token byte counts, which independently returns marginal bandwidth
   **64.4 GB/s**, inside the measured pure-load ceiling. Both statements are
   scale-dependent, not contradictory.
3. **The pure-load ceiling on the phone is 62.7–70.1 GB/s** (256 MB probe,
   16-byte loads, flat across a 512x thread sweep). Weight streaming runs at
   that marginal rate — saturated — so the remaining levers are dispatch-count
   fusion (a `split3` qkv kernel ≈ 0.44 ms) and traffic reduction, not "better
   matvecs".
4. But **`--split 1` (removing 130 split-k dispatches) is 2.4 ms WORSE** —
   parallelism on tall/narrow shapes dominates the per-dispatch overhead it
   adds. Dispatch overhead is real AND split-k stays.

## Optimisation ledger (problem → change → measured effect)

Each entry is phone-measured unless marked laptop.

- **Per-launch submits → `vulkore::Batch`**: 57.8 → 42.4 ms/token at the ~130
  dispatch stage; the qualitative change mattered more — shapes stopped
  converging on a fixed ~15 ms floor and became genuinely memory-bound.
- **int4 nibble packing along k, group-interleaved (KM4)**: the baseline packed
  8 nibbles/uint along the COLUMN axis — one 4-byte load per 4 useful BITS.
  Repacking along k (32 weights per `uint4` load, wave still contiguous) is 32x
  fewer load instructions for identical traffic: matvec 11.8 → 28.5 GB/s
  best-per-shape (2.4x, 5.9x on lm_head alone). **The decisive axis is load
  instruction count, not coalescing** — coalescing was already perfect.
- **Naive k-major (per-thread contiguous)**: NEGATIVE — the worst k-major
  layout, 1.5x slower than group-interleaved at identical load widths, because
  a 64-wide wave touches 64 cache lines per instruction. **Per-WAVE contiguity
  is the goal, not per-thread.**
- **Split-k on tall/narrow shapes**: 2.02x on `down` (6912×1152 = only 1152
  threads unsplit). Neutral-to-negative on wide layers; capped by the 65535
  workgroup limit (split16 cannot run lm_head). Split factor tuning beyond 4 is
  inside run-to-run noise (a 1 ms "win" over two rounds did not survive five).
- **Register blocking (2/4 outputs/thread)**: NEGATIVE, up to 1.5x slower
  overall, 0.53x on narrow shapes — the workload is latency-bound and blocking
  spends the occupancy that hides the latency. The confidently-predicted
  optimisation, and the direct opposite of split-k.
- **Workgroup size**: 10–30% per shape but device-specific — wg32 best on
  turnip, among the worst on the 840 (wg128 wins there). A per-device knob, not
  a portable optimisation.
- **Serial argmax → 3-level tree**: 31.5 ms → 0.074–0.086 ms (~380x); took the
  whole token 56.7 → 23.8 ms (2.4x) by replacing ONE dispatch of 838. The
  serial FINALIZER was the whole problem: pass 1 hits the memory floor at 4096
  partials (0.032 ms); the 2-level optimum is a compromise between two bad
  things, and a third tree level (262144→4096→64→1) collapses it. The kernel
  had been guessed at "~1 ms class" and never timed — it was 30x that.
- **Sampling: full-vocab download + host nth_element → GPU top-4 harvest**:
  8.32 → 0.08 ms/token (~104x), token 28.7 → 19.7 ms. The cost was NOT the
  1 MiB staged readback but host work linear in candidate count (4096
  candidates still cost 0.94 ms; default is 1024 chains). Best-4-per-chain is
  exact (64.000/64 of the true top-64 on every token); best-1 (the obvious
  `argmax_partial` reuse) loses 2.1 of 64 on average — measured, not assumed.
- **Host `record` cost → `DescriptorCache` + barrier collapse + no per-dispatch
  heap allocs**: 4.46 → 0.85 ms/token (5.2x). ~60% of the old cost was
  `vkAllocateDescriptorSets`+`vkUpdateDescriptorSets` per dispatch (memoised
  away — the chain binds the same buffers every token), ~30% the six `vkCmd*`
  calls. Collapsing 1676 barriers to 839 and narrowing the per-dispatch one to
  `COMPUTE→COMPUTE` also bought **~1.2 ms of GPU time** — on Adreno a wide
  barrier is not just a host-side packet write. API details in
  `vulkore-program-launch.md`.
- **fp32 → bf16 block scales, four per `uint2` (`mv_km4_bs`)**: −10.0% weight
  traffic (595.9 → 536.3 MiB/token) AND −37.5% load instructions; gpu 14.40 →
  13.43 ms/token (5/5 paired A/B rounds, non-overlapping distributions).
  Fidelity neutral — established only by widening the probe (§traps).
- **Attention flatness (deep spans)**: `softmax_rows` two-pass **30.4x** at
  span 8288; `attention_scores_mq4` 3.07x; `attention_apply_mq4+_reduce`
  3.75x; attention total 5.66x at depth. All three are SLOWER at span 41
  (0.67x/0.77x/0.92x) — hence `attn_min_span = 128` (on the right side of the
  crossover, not tuned). Deep bucket 36.16 → 25.50 ms/token; +78
  dispatches/token at depth (+0.44 ms) buys ~17 ms. Bonus: prefill of a
  4176-token prompt 137.2 → 77.0 s (1.78x) since decode-shaped prefill pays
  the depth cost at every position. Equivalence proved twice: 200 greedy
  tokens past position 4176 byte-identical between arms on the phone, and a
  deep `--verify` agreeing with the legacy arm to every printed digit.
- **Sliding window applied (correctness fix that is also a speedup)**: windowed
  vs unwindowed at depth is 2.85–4.26x on gpu ms/token — the window makes 22 of
  26 layers stop growing with depth, so the correct implementation is also the
  cheap one. Depth cost past the fix is NOT bandwidth (deep KV adds ~3 GB/s
  against a ~65 GB/s ceiling): it was serialisation in `softmax_rows`/
  low-occupancy `attention_apply`, fixed by the flatness work above.
- **Load time: sequential pack → threaded waves + `.xpack` cache**: 14.4 →
  3.07 → **1.90 s** cold on the phone (interleaved rounds, spread <2%). The
  cache-hit path is NOT I/O bound: 0.25 s read (a cold `dd` of the file runs
  2.9 GB/s) / 0.39 s verify / **0.48 s upload** — upload is the largest term
  and is a `Buffer`/staging question, not a load-path one. Threading the
  checksum bought 10%, not the expected 4x (most tensors are one hash chunk).

## Kernel validation on the real phone, and the turnip-proxy rule

All LLM kernels are validated on the Adreno 840 under the Qualcomm proprietary
driver via `examples/llm/phone_validation.cpp` (self-contained; coverage check
fails the run if any exported kernel goes untested — 25/25 in
`llm_transformer.cl`, 5/5 in `llm.cl`, plus the windowed-attention checks).
Tolerance `|gpu−cpu| ≤ 1e-6 + 1e-4·|cpu|`; the `atol` term is load-bearing for
accumulations landing near zero, which are relatively "wrong" by 1e-3 while
absolutely fine.

**Liveness is proved, not assumed** — this repo once had a kernel (8th-power
Mandelbulb using `pow`/`acos`/`atan2`) compile clean, pass spirv-val, bind,
launch without error and silently not execute on Adreno. Two detectors:

- **Sentinel** (out-of-place): prefill output with −1e30, require every element
  gone.
- **Expected-change** (in-place — `rope`, `rope_cached`, `softmax_rows`,
  `add_residual`, `mul_scalar`, `kv_append` — where a sentinel would destroy
  the input): every element the CPU reference changes must have changed on the
  GPU.

Both detectors are themselves validated by **negative controls** (half-grid
launches flag exactly the expected counts) and the binary fails the run if a
control does not fire. Multi-dispatch chains sentinel-check the INTERMEDIATE
buffer so a dead first pass cannot hide behind a live second pass. All
transcendentals in use (`sin/cos/exp/exp2/log2/rsqrt`) execute on both drivers;
`pow` remains avoided and untried. Harness trap worth knowing: `stream_read`
writes thread 0's own strided PARTIAL, not the array sum — the obvious CPU
reference reads as a catastrophic failure against a correct kernel.

**The turnip-proxy rule:**

- **Numerics: turnip is a faithful proxy.** Every max_rel/max_abs on the phone
  matches the laptop to the printed digit across the full kernel table; two
  consecutive phone runs are byte-identical. (Not a guarantee — the Mandelbulb
  and the Mali coherency bug are counterexamples in kind — but the divergence
  risk did not materialise for these kernels.)
- **Performance: turnip is NOT a proxy.** At span 8288, turnip says attention
  is 67% `attention_scores` / 17% softmax; the Adreno 840 says 39% / 48%.
  Profiling only the laptop would have optimised the wrong kernel. Also the
  best workgroup size differs (wg32 vs wg128). **Profile on the target
  device.**

## Measurement traps (each one shipped or nearly shipped a wrong number)

- **Dead-code elimination**: a pure-load probe whose accumulator was only used
  by thread 0 let the compiler sink the loop — 419 GB/s reported on a part
  whose DRAM tops out near 135. Keep every thread's loads live via a
  data-dependent guard that can never fire.
- **robustBufferAccess**: a probe buffer over `maxStorageBufferRange` gets
  out-of-range reads squashed to zero for free — 311 GB/s from a 512 MB buffer
  on turnip while smaller sizes scaled linearly. Size probes inside the limit;
  verify linear scaling.
- **A measured rate ABOVE your ceiling means the ceiling is broken, not that
  you won.** An early pure-load reading of 39.1 GB/s was quoted repo-wide as
  "the" ceiling and produced "91% of achievable, nothing left" — while the
  decode loop itself was measuring 41.4 GB/s. Re-measurement: 62.7–70.1 GB/s.
- **Count all the bytes**: the matvec benchmarks counted quantised nibbles only
  while the kernels also read one fp32 scale per 32 weights — every GB/s figure
  was 25% low (times right, byte counts wrong). True traffic was 595.9 MiB/token
  at fp32 scales.
- **Best-of-N, not mean**: thermal drift makes a mean drift run-to-run (same
  shape 2.66 vs 2.01 ms in consecutive runs); best-of-N isolates kernel quality
  from thermal state. Report min/median/mean over every step, never the last.
- **Pin the host thread** (`taskset 80`): the host blocks on a fence all token,
  so the governor parks it on a little core and `record` swings ~2x for the
  same binary. An unpinned host-side number is not a measurement.
- **Intra-run clocks are stable to 0.2%; inter-run clocks differ up to 1.6x**
  on identical work (the device throttles under sustained load). A curve is
  internally valid; comparing shallow numbers across runs requires matched
  thermal history. Interleave A/B arms within one session.
- **Position and thermal load are perfectly confounded in one long
  generation** — the deep bucket is also the hottest. The `--global-every 1000`
  control (every layer local ⇒ position-independent work past 512) separates
  depth cost from drift; it measured drift at 0.16% and attributed the whole
  curve to genuine depth cost.
- **Page cache**: cold-load baselines vary 1.8x depending on whether the 806
  MiB GGUF is warm from a previous run. Drop caches explicitly
  (`posix_fadvise(POSIX_FADV_DONTNEED)`) before every load measurement.
- **A small greedy probe cannot judge a weight-format change**: any
  perturbation, however tiny (2^-20 on a scale), moves a 4-position correlation
  probe by ~0.004 purely by reshuffling roundings. Use `--verify-steps ~20`.
- **Constant-fill test data hides packing bugs** (every layout produces the
  same answer on `0x71717171`); use pseudo-random weights.
- **Don't difference numbers from different harnesses** and call the residual
  a mystery (single-position timings vs curve-bucket medians disagree by
  construction).
- **A quality metric printed at the wrong precision erases itself**: top-k mass
  printed as `%.6f` rendered every row 1.000000; print deficits in scientific
  notation.

## Native-format trade (the standing open decision)

| weight path | traffic/token | gpu ms | logit corr |
|---|---|---|---|
| uniform int4, fp32 scales | 595.9 MiB | 14.40 measured | 0.955 |
| **uniform int4, bf16 scales (shipped)** | **536.3 MiB** | **13.43 measured** | 0.952 |
| native formats, repack, fp32 scales | 849.1 MiB | 18.53 projected | ~0.999 |
| native formats, repack, bf16 scales | 775.4 MiB | 17.32 projected | ~0.999 |

~+3.9 ms/token (−29% GPU throughput) buys 0.95 → ~0.999 fidelity and removes
the double quantisation. Applying the bf16-scale trick to the repack is the
same change again and cuts the repack's size premium to nearly free
(`gguf.md`). A product decision, not an engineering one.
