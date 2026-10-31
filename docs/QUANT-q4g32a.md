<!-- SPDX-License-Identifier: CC-BY-4.0 -->
Local Model Access (LMA) - created by artist Josh Hughes.

# The p0pc0rn pack and its 4-bit quantisation (q4g32a)

Copyright 2026 Josh Hughes, licensed under CC BY 4.0 like [SPEC.md](../SPEC.md). This page documents `pocketflux.pack/1`, the format of the models that the p0pc0rn app runs on iPhone ([registry](../registry/models.json), first four entries), so that anyone can read them. The packs' own model cards say only p0pc0rn's engine reads them; with this page and [`lma/`](../lma), any LMA site can.

## Why the models are quantised

NVIDIA's Nemotron 3.5 Bolt E4B is published in BF16, 2 bytes per weight: about 16 GB of files. A phone browser cannot hold that. The p0pc0rn pack stores almost every weight in **4 bits, plus a 16-bit scale and a 16-bit minimum per group of 32 weights**: 5 bits per weight, 3.2 times smaller. The Bolt E4B pack is 4.68 GB. On iOS 27, the p0pc0rn app also runs that model split across the page and a second process ([SPEC.md §9](../SPEC.md#9-performance-informative)).

The weights **stay 4-bit in GPU memory**. The kernels dequantise each group as they read it, in registers, so the full-precision model never exists on the phone.

## The quantisers

Both work on a matrix `W[n, k]` of N rows (output features) and K columns (input features), in groups along K. All arithmetic is fp32; `f16()` rounds to the nearest IEEE half (ties to even), and `round` is round-half-to-even.

**q4g32a: 4-bit, groups of 32, asymmetric.** For each row and each group of 32 values `v`:

```
mn = f16(min(v))
s  = f16((max(v) - mn) / 15)          if s == 0 then s = 1
q  = clamp(round((v - mn) / s), 0, 15)
w  = q * s + mn                        the dequantised weight
```

**q8g64s: 8-bit, groups of 64, symmetric:**

```
s = f16(max|v| / 127)                  if s == 0 then s = 1
q = clamp(round(v / s), -128, 127)
w = q * s
```

`q8g64s` costs 8.25 bits per weight and is used where 4 bits lose too much.

## The recipes the tested packs use

Each manifest records its recipe in `quant_recipe` (rules matched in order against the source tensor names, first match wins), and every tensor records its `enc`.

| Pack | Recipe | What it means |
|---|---|---|
| Nemotron 3.5 Bolt E4B | `q4all` | Every matrix `q4g32a`: the layers, the embedding, which is also the output head (tied), and the per-layer embeddings (PLE). Norms are `f32`. Encodings used: `q4g32a.t64` (254 tensors), `q4g32a.rows` (1), `f32` (278). |
| Gemma 4 E2B-it | `bal_gptq4_attn8_mlp8_4_6_12_13` | MLPs `q4g32a` with GPTQ-fitted parameters (`quantiser: "GPTQ q4g32a ..."` on those tensors). Attention, gates, per-layer projections and the MLPs of layers 4, 6, 12 and 13 are `q8g64s`, as is the vision tower. The embedding and PLE are `q4g32a`; one table is `f16`; norms are `f32`. GPTQ only chooses better codes, scales and minimums: the bytes are ordinary `q4g32a`, decoded the same way. |
| The two drafters | (in `notes`) | Layers and centroids `q8g64s`; the output head `q4g32a` in cluster-major row order; `token_ordering` (`u32`) maps each head row to its token id. See [DRAFTER.md](DRAFTER.md). |

Two more transformations are recorded in each manifest's `license.modified` and in each tensor's `src`:
- **Row fusion.** `qkv_proj` is the q, k and v projections' rows concatenated in that order, and `gate_up_proj` is gate then up. One matrix-vector product then does three (or two).
- **Re-layout** into the tile records below.

## Byte layouts

All values are little-endian. A tensor occupies `bytes` bytes at `offset` in its shard file (offsets are multiples of 256 in the published packs). `N`, `K` are the manifest's `shape`.

### `q4g32a.t64`: 64-row tiles, for matrix-vector products

Requires N % 64 = 0 and K % 32 = 0. With NT = N/64 tiles and G = K/32 groups, the tensor is NT·G records of **1280 bytes**. Record (t, g) is at byte `(t·G + g)·1280`, so a tile's groups are consecutive.

| Bytes of a record | Content |
|---|---|
| 0..1023 | Codes: 64 rows × 16 bytes. Row r (output n = 64t + r) is at byte 16r; byte j holds `q[k = 32g + 2j] \| q[32g + 2j + 1] << 4` (the low nibble first) |
| 1024..1279 | 64 u32 words; word r is `bits(f16 s) \| bits(f16 mn) << 16` for row r. In WGSL, `unpack2x16float(word)` gives (s, mn) |

Size: N·K·5/8 bytes.

### `q8g64s.t64`

Requires N % 64 = 0 and K % 64 = 0. Records of **4224 bytes** at `(t·G + g)·4224`, G = K/64:
- bytes 0..4095 are the codes, 64 rows × 64 int8, with row r at 64r;
- bytes 4096..4223 are 64 f16 scales, row r at 4096 + 2r.

### `q4g32a.rows` and `q8g64s.rows`: one row at a time, for lookups

The per-layer embedding table (262,144 tokens × 10,752 values for Bolt E4B) is read a few rows at a time: the rows of the tokens in the prompt. So it is stored row by row, each row self-contained:

- `q4g32a.rows`: each row is `K/2 + (K/32)·4` bytes. First come K/2 bytes of codes, where group g's 16 bytes are at 16g, packed as above. Then K/32 pairs `(f16 s, f16 mn)`, pair g at `K/2 + 4g`. Row i is at `offset + i·rowBytes`.
- `q8g64s.rows`: each row is `K + (K/64)·2` bytes: K int8 codes, then K/64 f16 scales.

### `f16`, `f32`, `u32`

Row-major IEEE half, IEEE single, and unsigned 32-bit integers.

## A worked example

One row's group of 32 weights, `v[k] = (k - 10)/64`, except `v[5] = 0.3`:

```
v[0..5]  = -0.15625, -0.140625, -0.125, -0.109375, -0.09375, 0.3      min -0.15625, max 0.328125
mn = f16(-0.15625)                 = -0.15625          bits 0xb100
s  = f16((0.328125 + 0.15625)/15)  = 0.03228759765625  bits 0x2822   (0.032291666 before rounding to f16)
q[0..5]  = 0, 0, 1, 1, 2, 14        q[30], q[31] = 15, 15
codes    = 0x00, 0x11, 0xe2, ...    (byte j = q[2j] | q[2j+1] << 4)
word     = 0xb1002822               (s in the low half, mn in the high half)
w[0..5]  = -0.15625, -0.15625, -0.12396240234375, -0.12396240234375, -0.0916748046875, 0.2957763671875
```

Every weight comes back within s/2 of its original: here the largest error is 0.015625, against s/2 = 0.0161.

## How it runs on the GPU

A decode step multiplies a vector by every weight matrix, so it is limited by how many bytes of weights the GPU can read, not by arithmetic. A matrix-vector kernel walks K for one 64-row tile: it reads a record's 1,024 bytes of codes and 64 scale/min words, unpacks eight 4-bit codes per u32 word, computes `q·s + mn` in registers, and multiplies by the input. Nothing is written back expanded: the matrix stays at 5 bits per weight in GPU memory and is read at that size. The matrix-matrix kernels (prompt read-in) expand a tile to fp16 in workgroup memory, never to a whole tensor.

These kernels are public in [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels) (MPL-2.0): `gemv`, `gemvM` and `gemm` read `q4g32a.t64` and `q8g64s.t64` directly, and `gemvV5` reads `q4g32a.t64`, and its [docs/FORMAT.md](https://github.com/ArtistJoshHughes/p0pc0rn-kernels/blob/main/docs/FORMAT.md) is the source of the tile layouts above.

## Reading a pack with LMA

```js
import { readModel, verifyModel, Loader, pocketflux } from './lma/index.js';
const model = await readModel(files);                 // manifest.json + every file it lists, in one pick, in any order
const v = await verifyModel(files, { approved, ns: 'my-site', model, lazy: true });   // the manifest's sha256 against the approved entry
const L = new Loader(model.files, model, { verify: 'chunks' });   // each 16 MiB chunk is checked against the manifest on first read
const embed = await pocketflux.readRow(L, 'embed_tokens_per_layer.weight', tokenId);   // one row of a .rows table, dequantised
const qkv = await L.toGpu('layers.0.self_attn.qkv_proj.weight', backend);              // a q4g32a.t64 matrix, 4-bit on the GPU
```

The `lma/ext/pocketflux.js` decoders (`dequant`, `dequantTile`, `dequantRow`, `readRow`, `f16`, `rowBytes`, `tensorBytes`) are a CPU reference for checking a kernel's output. They are not the way to run a model.

## What is confirmed, and how

- **The `q4g32a` quantiser, the `q4g32a.t64` tile layout, the `q4g32a.rows` row layout, `f32`, and the q|k|v and gate|up row fusion.** Re-quantising rows of NVIDIA's BF16 source with the formulas above gives the published Bolt E4B pack's bytes exactly. This was checked on the first tile of a fused q|k|v matrix and of a fused gate|up matrix, the first four rows of the per-layer embedding, and a norm. So `q4all` is the plain quantiser above, with no refit.
- **`q8g64s` and `q8g64s.t64`.** Re-quantising Gemma 4 E2B-it source rows gives the bytes of a p0pc0rn-packed copy of the balanced E2B pack. That was a local build, not byte-checked against the published file.
- **`q8g64s.rows`.** Checked against the p0pc0rn app's decoder for this layout. No published pack uses it, so it is not checked against packed data.
- **The tied output head.** Both source configs set `tie_word_embeddings`, and the packs have no separate head tensor.
- **The drafter head's cluster-major order and `token_ordering`.** From the drafter manifests (`notes`, the `u32` encoding's description, `num_centroids` × `per_centroid` = the vocabulary) and checked against the p0pc0rn app's drafter. Clusters own consecutive blocks of `per_centroid` head rows.
- **Not reproduced here: GPTQ's choice of codes for the E2B pack's MLPs.** A decoder does not need it.
- **Test coverage.** `test-vectors/lma-test-pack/` is a tiny pack in every encoding, written by `tools/make_test_models.py` from this page. `tests/conformance.test.mjs` checks that the reference decoders reproduce it exactly.
