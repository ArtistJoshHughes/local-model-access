<!-- SPDX-License-Identifier: CC-BY-4.0 -->
Local Model Access (LMA) - created by artist Josh Hughes.

# LMA conformance

For [SPEC.md](SPEC.md) 1.0. Copyright 2026 Josh Hughes, licensed under CC BY 4.0, like the specification.

## Checklist for a site

- [ ] Reads GGUF (v2, v3) and safetensors files as published, telling them apart by their first bytes, not by name (§2).
- [ ] Accepts split GGUF parts and safetensors shards in any order, and rejects a model with a missing or repeated part (§2.1).
- [ ] Identifies every file by `sha256` and `size` against its own approved list, and ignores fields and capability values it does not know (§3, §4).
- [ ] Gets files only from a file picker the user opens with a tap, without an `accept` filter (§6).
- [ ] Never uploads a model file or sends its contents anywhere, and never copies it into origin storage (§6, §8).
- [ ] Runs the instant header check first, hashes each file once with progress, and remembers the result in its own storage (§7).
- [ ] Reads files only in bounded slices, a tensor at a time, and shows its peak memory: file bytes held at once and GPU memory (§8).
- [ ] Does not depend on the share sheet to save files (§5).
- [ ] If it supports `pocketflux.pack/1` (§2.2): approves a pack by its manifest's identity, checks every file (once, or each chunk before use) against the manifest, and rejects a pack with a file missing or short.
- [ ] Ignores `role` values it does not know, and never requires a companion model, such as a drafter, to produce the target's output (§4.1).

A reader library on its own (no site) needs the first three items and the slice rule.

## Test vectors

[test-vectors/](test-vectors) holds small real model files, and [vectors.json](test-vectors/vectors.json) describes each one: its files and identity, the header facts a reader must find (name, architecture, tensor count, parts, and the offset, size, dtype and shape of the first and last tensor in every part), and the exact matrix-vector answers stored in its metadata. The files are reproducible byte for byte with `python tools/make_test_models.py vectors DIR`.

| Vector | Files | sha256 | Tests |
|---|---|---|---|
| `small` | `lma-test-1m.gguf` (795,712 B) | `6cbae2321f17e5eb…` | a single GGUF v3 file |
| `st` | `lma-test-st.safetensors` (267,200 B) | `87ef085cb60be33e…` | a single safetensors file |
| `split` | `lma-test-split-00001-of-00002.gguf` (529,408 B), `lma-test-split-00002-of-00002.gguf` (266,496 B) | `db37feefc8a63d4f…`, `cfae3297e88f9f6c…` | a split GGUF (`split.*` keys) |
| `sharded` | `lma-test-sharded-00001-of-00002.safetensors` (263,992 B), `lma-test-sharded-00002-of-00002.safetensors` (3,216 B), plus `lma-test-sharded.safetensors.index.json` | `62f7c0a7a545a3b2…`, `434e527a3795fc25…` | sharded safetensors with an index |
| `pack` | `lma-test-pack/`: `manifest.json`, `lm.global.pfw`, `ple.rows.pfw`, `tokenizer/tokenizer_meta.json` | see `vectors.json` | a `pocketflux.pack/1` with every encoding (`q4g32a.t64`, `q8g64s.t64`, `q4g32a.rows`, `q8g64s.rows`, `f16`, `f32`, `u32`). The expected decoded values are listed, and chunks are 4 KiB so that lazy verification is exercised |
| `big` (optional, not stored here) | `lma-test-1g.gguf` (1,058,021,024 B) | `cf4b1cdd3ca47498…` | a 1 GB file: hand-over time, hashing, streaming. Make it with `python tools/make_test_models.py DIR --big` |

A conforming reader must also refuse these inputs, with the error codes the reference code uses:

| Input | Expected |
|---|---|
| 4,096 random bytes | `not-model` |
| `lma-test-1m.gguf` without its last 1,000 bytes | `truncated` |
| the first 200 bytes of `lma-test-1m.gguf` | `truncated` |
| `lma-test-1m.gguf` with bytes 4-7 set to `09 00 00 00` | `version` |
| only part 1 of the split GGUF | `missing-part` |
| shard 1 of the sharded safetensors with its index | `missing-part` |
| `lma-test-1m.gguf` with byte 400,000 changed | the header reads, but the sha256 matches no approved identity, so it is not approved |
| the pack without `ple.rows.pfw` | `missing-part` |
| the pack with `lm.global.pfw` 16 bytes short | `truncated` |
| the pack with one byte of `lm.global.pfw` changed | full verification refuses it; only that chunk fails `verifyChunk`; a chunk-verifying loader throws |

Run them against the reference code with `node tests/conformance.test.mjs`. Another implementation reads `vectors.json` and checks the same things.

## Claim

An implementation that meets the checklist and passes the vectors may say so with this line:

> LMA-conformant (Local Model Access by artist Josh Hughes)
