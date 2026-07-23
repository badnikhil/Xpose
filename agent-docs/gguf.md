# GGUF: loader, tokenizer, quant formats, repack + dequant/matvec kernels

Host side: `examples/llm/model/` (`gguf.{hpp,cpp}` GGUF v3 reader,
`tokenizer.{hpp,cpp}` SentencePiece BPE, `inspect.cpp` CLI, `selftest.cpp`
synthetic-file checks) — pure CPU C++20, no Vulkan header anywhere, builds and
runs on a GPU-less machine. GPU side: `kernels/llm_quant.cl` (18 kernels) +
`examples/llm/quant_repack.{hpp,cpp}` + `examples/llm/quant_test.cpp`.

```sh
cmake --build build --target gguf_inspect        # option VULKORE_BUILD_LLM_MODEL=ON by default
./build/examples/llm/model/gguf_inspect --selftest              # no model needed
./build/examples/llm/model/gguf_inspect models/gemma-3-1b-it-Q4_K_M.gguf
gguf_inspect M.gguf --tensors --meta --encode "text" --dequant <name>
gguf_inspect A.gguf --compare B.gguf             # cross-quant agreement check
```
`models/` is gitignored (weights are hundreds of MB).

## File format as found on disk

```
magic u32 "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64
metadata_kv_count x { key:string, value_type:u32, value }
tensor_count     x { name:string, n_dims:u32, dims:u64[n_dims], type:u32, offset:u64 }
<pad to general.alignment, default 32>
tensor data blob            <- tensor.offset is relative to the START of this blob
```

Details that are easy to get wrong:

- **Strings are u64 length + bytes, NOT NUL-terminated.** Handed out as
  `std::string_view` into the mapping (copying a 262144-entry vocab into
  `std::string`s costs ~15 MB for nothing). Every view and every `Tensor::data`
  pointer dies with the `GGUFFile`; it is move-only to make that hard to violate.
- **`shape[0]` is the fastest-varying axis** — reversed relative to
  numpy/PyTorch. `blk.0.ffn_down.weight` reads `[6912, 1152]` here and
  `[1152, 6912]` in an HF checkpoint. Shapes are kept in GGUF order because the
  data is laid out in that order.
- **mmap, not read**: `MAP_PRIVATE | PROT_READ` + `MADV_SEQUENTIAL`; weights are
  consumed by pointer. Reading an ~800 MB file into the heap is not an option on
  a phone.
- **Every field read is bounds-checked** (`Reader::need`) — u64 lengths out of an
  untrusted blob would otherwise SIGBUS instead of throwing. Bad magic,
  truncated header and truncated body all throw (covered in the selftest).
- `general.alignment` is optional (default 32); every tensor offset is a
  multiple of it. Metadata is order-preserving AND hash-indexed.

## Quantisation types

Block size / bytes-per-block are ABI (from ggml's `type_traits[]`); one wrong
constant silently shifts every subsequent tensor.

| type | block | bytes | dequant | notes |
|---|---|---|---|---|
| F32  | 1   | 4   | yes | memcpy |
| F16  | 1   | 2   | yes | portable decode incl. subnormals |
| BF16 | 1   | 2   | yes | top 16 bits of an fp32 |
| Q4_0 | 32  | 18  | yes | fp16 `d`, 16 B nibbles, quants unsigned biased −8 |
| Q4_1 | 32  | 20  | yes | fp16 `d` + fp16 `m`, affine, no bias |
| Q5_0 | 32  | 22  | yes | fp16 `d`, u32 `qh` bitmask of 5th bits, 16 B nibbles |
| Q5_1 | 32  | 24  | yes | Q5_0 + fp16 `m` |
| Q8_0 | 32  | 34  | yes | fp16 `d`, 32 int8 |
| Q4_K | 256 | 144 | yes | fp16 `d`+`dmin`, 12 B packed 6-bit scales/mins, 128 B nibbles |
| Q6_K | 256 | 210 | yes | 128 B low nibbles, 64 B high 2-bit pairs, 16 int8 scales, fp16 `d` |
| Q5_K | 176 B, Q2_K/Q3_K/Q8_K/IQ* | — | — | **no** — sizes known (parsing/layout works), dequant unimplemented; `dequantize()` returns false rather than emitting garbage |

### Exact bit layouts (the error-prone ones)

Bytes little-endian; `d`/`dmin` fp16.

**Q8_0** — 32 weights in 34 B: `[0:2]` fp16 d, `[2:34]` 32×int8. `w = d*q`.

**Q5_0** — 32 weights in 22 B:
```
[0:2] fp16 d | [2:6] u32 qh | [6:22] 16 B nibbles
w[e] = d * (q[e] - 16),  q[e] = nibble | (((qh >> e) & 1) << 4)
nibble = qs[e%16] & 0xF (e<16)  |  qs[e%16] >> 4 (e>=16)
```
**The interleave is the trap**: byte `j` of `qs` holds weight `j` (low nibble)
and weight `j+16` (high nibble) — the two HALVES of the block, not adjacent
weights. ggml's `((qh >> (j+12)) & 0x10)` obscures this; stated plainly, **bit
`e` of `qh` belongs to weight `e`**. Assuming `qs[j]` holds weights `2j`/`2j+1`
yields a statistically plausible tensor that is wrong for half its values.

**Q4_K** — 256 weights in 144 B (8 sub-blocks of 32):
```
[0:2] fp16 d (scale OF the sub-block scales) | [2:4] fp16 dmin | [4:16] 12 B scales/mins | [16:144] nibbles
w = d * sc[s] * q  -  dmin * mn[s]
```
Quants are **UNSIGNED and unbiased** — no `-8` as in Q4_0; the zero point is the
per-sub-block min. Biasing by 8 is the classic Q4_K bug. The 12-byte scale
block (`get_scale_min_k4`):
```
s < 4:  sc = q[s] & 63          mn = q[s+4] & 63
s >= 4: sc = (q[s+4] & 0xF) | ((q[s-4] >> 6) << 4)
        mn = (q[s+4] >> 4)  | ((q[s]   >> 6) << 4)
```
The last four pairs take their low nibble from bytes 8–11 and borrow their high
2 bits from the top bits of bytes 0–7; note `mn` for `s>=4` borrows from `q[s]`,
not `q[s-4]` — the asymmetry is real and easy to "tidy up" into a bug. Element
`r` (0..255): sub-block `s = r/32`; nibble byte `16 + (r/64)*32 + (r%32)`, low
nibble when `(r%64) < 32` else high.

**Q6_K** — 256 weights in 210 B (16 groups of 16):
```
[0:128] ql (low 4 bits) | [128:192] qh (high 2 bits, 4/byte) | [192:208] 16 int8 scales | [208:210] fp16 d
w = d * sc[group] * (q - 32)
```
The 256 elements are two halves of 128, and inside each half the elements are
**four interleaved quarters**. For element `r`:
```
h = r/128, wq = r%128, l = wq%32, g = wq/32
ql byte = h*64 + l + (g&1)*32   (low nibble if g<2 else high)
qh byte = 128 + h*32 + l       bits (2g, 2g+1)
scale   = 192 + h*8 + l/16 + 2g
q       = (nibble | (highbits << 4)) - 32
```

### fp16 by hand

No `half` buffer type is available (and `half` is a **reserved word** in OpenCL C
even with the fp16 extension off — naming a variable `half` is a hard syntax
error), so the kernel-side decoder is pure integer bit surgery finished with
`as_float`: exponent rebias +112, mantissa `<<13`, subnormals normalised by a
shift loop, Inf/NaN passed through. **Swept over all 65536 bit patterns on
device vs host: exact**, incl. 2046 subnormals, 2046 NaNs, both zeros/infs.
Every kernel here multiplies by an fp16 scale; a decoder bug presents as a
diffuse scale error that is miserable to localise downstream.

## The Q4_K_M census surprise

Tensor tally of `gemma-3-1b-it-Q4_K_M.gguf` (806,058,240 B, GGUF v3, 38 KVs,
340 tensors):

```
F32   157 tensors   524.50 KiB ( 0.1%)   norms/biases
Q4_K   39 tensors    71.98 MiB ( 9.4%)   fast axis 1024/6912 (attn_output, ffn parts)
Q5_0  117 tensors   299.13 MiB (39.2%)   most width-1152 attention/FFN weights
Q6_K   13 tensors    80.98 MiB (10.6%)   ffn_down, output tensors
Q8_0   14 tensors   309.88 MiB (40.6%)   incl. token_embd (all 262144x1152 of it)
```

**"Q4_K_M" names the recipe, not the contents: Q4_K is 9.4% of the bytes.**
K-quants need the fast axis divisible by 256, and Gemma 3 1B's width is 1152
(= 4.5 super-blocks), so every width-1152 tensor falls back to legacy 32-block
types. Consequences: a Q4_K-only kernel cannot run this model, and the quant
name in a filename must never be trusted — run `gguf_inspect --tensors` first.

## Gemma config keys

Keys are namespaced by `general.architecture` (`gemma3.block_count` vs
`llama.block_count`); `config()` builds the prefix from the arch string, so the
loader is not Gemma-specific.

| field | key | Gemma 3 1B |
|---|---|---|
| n_layers | `<arch>.block_count` | 26 |
| n_heads / n_kv_heads | `<arch>.attention.head_count[_kv]` | 4 / 1 |
| embedding_length | `<arch>.embedding_length` | 1152 |
| feed_forward_length | `<arch>.feed_forward_length` | 6912 |
| context_length | `<arch>.context_length` | 32768 |
| **head_dim** | `<arch>.attention.key_length` | **256** |
| rope_freq_base | `<arch>.rope.freq_base` | 1e6 |
| rms_eps | `<arch>.attention.layer_norm_rms_epsilon` | 1e-6 |
| sliding window | `<arch>.attention.sliding_window` | 512 |
| vocab_size | length of `tokenizer.ggml.tokens` | 262144 |

- **head_dim is 256, NOT embedding/heads (=288)** — `attention.key_length` is
  authoritative; the division is only a fallback when the key is absent.
- vocab_size comes from the token-array length; the scalar key is often absent.
- rope_freq_base 1e6, not the 10000 default — assuming the default silently
  ruins long-context behaviour.

## Tokenizer

The GGUF carries the whole tokenizer (`tokenizer.ggml.{model,tokens,scores,
token_type,bos/eos/unknown/padding_token_id,add_bos/eos_token,add_space_prefix}`;
model `"llama"` == SentencePiece). Standard SentencePiece merge loop (same shape
as llama.cpp's `llm_tokenizer_spm`):

1. Pull CONTROL/UNKNOWN tokens out of the raw text first, longest-match-first
   (`<start_of_turn>` must become one token; the merge loop can never produce it).
2. Normalise: `' '` → U+2581 `▁`; leading `▁` only if `add_space_prefix` —
   **Gemma 3 sets it false**, the SentencePiece default is true; assuming the
   default puts a spurious token on every string.
3. Split into UTF-8 chars; repeatedly merge the adjacent pair with the highest
   vocab score via a max-heap over bigrams. Ties break toward the **leftmost**
   pair (otherwise results depend on heap internals). Stale heap entries are
   filtered on pop, not removed — omitting that check breaks the loop.
4. Byte fallback (`<0xAB>` per UTF-8 byte) for anything not in the vocab, so the
   tokenizer is total.

Deviation from llama.cpp, knowingly: **USER_DEFINED tokens are not partitioned
on** (only CONTROL/UNKNOWN). In a Gemma vocab that class is 163 whitespace runs
which the merge loop already assembles; partitioning would change whitespace
tokenization. This is the most likely source of any divergence, specifically on
whitespace runs. Duplicate spellings resolve to the lowest id (first writer
wins), matching the reference implementations.

## Host-side verification

- All 340 tensor descriptors parse; **the blob tiling is exact** — sorted by
  offset the tensors cover the 762.49 MiB blob with zero holes, zero overlaps,
  every offset 32-aligned, zero tail. Strong structural evidence for the
  block-size constants (a Q4_K at 140 B instead of 144 leaves a hole per tensor).
- **Cross-quantisation agreement is the strongest check**: the same tensors
  dequantised from the Q4_K_M file and from the Q8_0 build of the same model
  (`--compare`) — worst corr Q4_K 0.99710, Q5_0 0.99831, Q6_K 0.99979, 0
  mismatching tensors, and the ordering Q6_K > Q5_0 > Q4_K matches the formats'
  precision ordering (a coincidentally-passing bug would not reproduce that).
  This is what validates `get_scale_min_k4`: a wrong unpacking decorrelates to ~0.
- Tokenizer: `decode(encode(s)) == s` for ASCII/CJK/Cyrillic/emoji/all 255
  non-NUL bytes; `<start_of_turn>`=105, `<end_of_turn>`=106 match the documented
  Gemma 3 ids.
- `--selftest` (32 checks, synthetic byte-built GGUF, no model): exact Q4_0 and
  Q5_0 dequant values (Q4_0 is absent from the real file), config on a `llama`
  prefix, head_dim fallback, a merge path that fails if the priority queue or
  stale-entry check is wrong, malformed-input throws.

**NOT verified:** bit-exact token-id parity with llama.cpp/HF on a corpus (no
reference implementation was available offline) — if GPU output is ever subtly
bad, re-check this first. Q2_K/Q3_K/Q5_K/Q8_K/IQ* sizes are transcribed but
untested. GGUF v2 is accepted on the assumption it is layout-compatible.
Sharded GGUFs unexercised.

## GPU kernels (`kernels/llm_quant.cl`, 18 entry points)

Per format (Q8_0, Q5_0, Q4_K, Q6_K): a standalone dequant + a fused
dequant-matvec reading the **native GGUF layout** (bit-exact, no host transform,
slow), plus a **repacked fast path** (`matvec_rq*` + `_split` + shared
`mv_reduce`) fed by a load-time host repack.

### ABI constraints that shaped the code

- **No `uchar`/`ushort` buffers** — `vulkore::Buffer` static_asserts 4-byte
  element types, so quantised tensors arrive as `global const uint*` and every
  field is dug out with `u8()`/`u16()` shifts. Q4_K's 144 B block is a multiple
  of 4 so it stays word-aligned; **Q5_0 (22 B), Q8_0 (34 B), Q6_K (210 B) are
  not** — from the second block on, every field sits at an arbitrary byte phase
  and costs a whole 4-byte load per byte extracted. This is the main reason the
  native path is slow.
- **No `__local`** (clspv emits `WorkgroupVariableSize`, `Program` throws) — the
  fused matvecs give one thread a whole output row, accumulator in a register.
- **Workgroup-count limit, hit for real**: 65535 workgroups in x = 4,193,280
  threads/dispatch on turnip. The per-ELEMENT dequant kernels cannot do a full
  `ffn_down` (6912×1152 = 7.9 M elements) in one launch — `vulkore::launch`
  throws; callers must tile. The per-ROW matvecs are nowhere near the limit.
- `cols` must be a multiple of 32 (256 for K-quants) throughout.

### Native path: bit-exact, kept as the oracle

Verified on Adreno X1-85/turnip, 45/45: dequant of all four formats is
**bit-exact** (max err 0) against a from-spec CPU reference indexed by element
(so a transcription slip fails differently than ggml's pointer walk), against
`GGUFFile::dequantize`, on random-BYTE synthetic blocks (exercising patterns a
real quantiser never emits) and on real Gemma tensors. Bit-exactness is
structural, not luck: each output is a small integer times an fp16 scale, both
exact in fp32 — any nonzero error is a layout bug, hence `rtol 1e-6` here vs the
transformer harness's 1e-4. Fused-matvec residual is fp32 reduction-order noise
(≤6.6e-07 abs vs a double-precision reference).

**The harness is mutation-tested** (a first-run all-green deserves suspicion):
flipping the Q5_0 high-bit shift to `qh >> (e/2)` — the plausible-looking
misreading — turns 16 passes into 5 failures across synthetic, loader
cross-check, matvec and real-tensor tests. Liveness: every output sentinel-
prefilled with -1e30; all 18 kernels demonstrably write (this repo has had a
kernel compile, validate, bind, launch cleanly and silently not execute).

Native throughput (real tensors, laptop, best-of-N): Q8_0 12.3, Q4_K 9.9,
Q5_0 2.7, Q6_K 3.0 GB/s — 4–19% of the 65.1 GB/s measured pure-load ceiling.
Two causes: load-instruction count (see above) and thread starvation (one
thread/row; `attn_k` is 256 rows). Native timings are also **unstable
run-to-run by up to 6x** even at best-of-7; the repacked kernels are stable to a
few percent. Keep native as the correctness oracle and for one-shot dequant;
never for decode.

### Repack fast path

Decode re-reads every weight each token, so a transform paid once at load
amortises. Layout (with `g = r/64`, `t = r%64`, `kb = k/32`, `nkb = cols/32`,
`idx = (g*nkb + kb)*64 + t`):

```
quant[]  4-bit nibble plane, one uint4 per idx: uint 4*idx + j/8, shift 4*(j%8)
         Q8_0: TWO uint4 per idx:               uint 8*idx + j/4, shift 8*(j%4)
high[]   Q5_0: one uint per idx, bit j
         Q6_K: two uints per idx: uint 2*idx + j/16, shift 2*(j%16)
scale[]  fp32, one per idx  (Q6_K: two, at 2*idx + j/16 — granularity 16 not 32)
bias[]   fp32, one per idx, Q4_K ONLY
```

Pack along **k** so one load carries weights one thread consumes; interleave in
groups of 64 so a wave still reads contiguous memory (per-WAVE contiguity is
the goal — naive per-thread-contiguous k-major measured as the *worst* k-major
variant; see `llm-performance.md`). Decisions:

- **Scales pre-decoded to fp32 on the host** — removes the fp16 decode and
  6-bit unpacking from the inner loop and lets four formats share one kernel
  shape. Only Q4_K needs a bias (`w = d·sc·q − dmin·mn` has an additive term);
  the others fold their zero point into a constant subtraction (−16, −32, none).
- **Bit widths preserved via planes, not widened to int8** — int8-for-everything
  would take Q4_K from 4.5 to 10 bits/weight and roughly double the model in
  RAM, defeating quantisation. Planes never straddle a word: 32 weights in 1–3
  loads instead of ~40.
- Input vector read as `float4` (otherwise the input dominates instruction
  count once weight loads are cheap).

Correctness: `reconstruct()` pulls element (r,k) back out of the repacked
buffers and must match the from-spec dequant **bit-for-bit** — a host-only
oracle with no tolerance to argue about. Synthetic cases use rows=130
(deliberately not a multiple of 64 — a padding bug is invisible at a round row
count). Mutation-tested with 4 injected bugs (nibble parity, high-bit shift,
scale granularity, interleave); all caught.

Measured (laptop, real tensors, GB/s of quantised bytes, best-of-N):

| Format | tensor | native | repacked | +split-k | speedup |
|---|---|---|---|---|---|
| Q8_0 | `token_embd` slice | 12.3 | 23.5 | **41.0** (split8) | 3.2x |
| Q5_0 | `attn_k` | 2.7 | 3.5 | **11.8** (split16) | 4.0x |
| Q4_K | `attn_output` | 9.9 | 17.9 | **22.8** (split8) | 1.7x |
| Q6_K | `ffn_down` slice | 3.0 | 18.9 | **51.6** (split8) | **13.8x** |

Q6_K at 51.6 GB/s is ~79% of the laptop's pure-load ceiling. GB/s is not
comparable across formats (it scales with bits/weight); in weights/s the row
reads 36.4 / 14.5 / 30.4 / 51.6 G. Split-k is a **runtime argument**: narrow
tensors want 16, wide ones peak at 8 and get worse at 16.

Cost: **762.0 → 849.1 MB (+11.4%), 2.6 s single-threaded at load** (183 tensors
repacked; F32 norms, 1-D tensors and non-divisible rows skipped). Per-format
growth (+5.9/+9.1/+21.9/+33.3% for Q8_0/Q5_0/Q6_K/Q4_K) is all fp32 scale/bias
tables; storing scales as bf16 instead (the trick that shipped for the int4
path, `llm-performance.md`) cuts that to **0/0/+6.7/+11.1%**.

### Status / caveats

- These kernels are validated on turnip only (they use `uint4`/`float4` vector
  loads the phone-validated kernel set does not) — re-measure on the Qualcomm
  driver before quoting phone numbers.
- **Not wired into decode.** The decode loop still re-quantises everything to
  uniform int4 (a double quantisation, logit corr 0.95 vs ~0.999); swapping in
  these per-format kernels is the highest-value open quality task, at a measured
  throughput cost of ~+3.9 ms/token (`llm-performance.md` §native-format trade).
- No repacked standalone dequant (matvec-only — the op decode needs).
- Workgroup size untuned (phone and laptop disagree on the best value; tuning
  against turnip would mislead).
- Q4_0/Q4_1/Q5_1/Q8_1/Q2_K/Q3_K/Q5_K/IQ* not implemented; none appear in this
  model. Q5_1/Q4_1 would be ~10 lines each.
