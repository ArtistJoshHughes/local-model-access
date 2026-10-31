<!-- SPDX-License-Identifier: CC-BY-4.0 -->
Local Model Access (LMA) - created by artist Josh Hughes.

# Drafters: faster replies with speculative decoding

Copyright 2026 Josh Hughes, licensed under CC BY 4.0 like [SPEC.md](../SPEC.md).

The models p0pc0rn runs on iPhone are set up as a **target** and a **drafter**. The target is the model whose words you get. The drafter is a small model that guesses a few words ahead, and the target checks all of the guesses in one pass. Both tested targets have a tested drafter in the [registry](../registry/models.json), marked `"role": "drafter"` with `"pairs_with"` naming its target.

## How it works

1. The drafter proposes k tokens, one after another. It is small, so this is quick.
2. The target runs **one** verify pass over its current token and the k drafts, producing k + 1 rows of predictions at once.
3. Drafts are kept from the left while the target agrees. At the first draft it rejects, the target's own token replaces it, and the rest are dropped. If every draft is kept, the target's prediction after the last one is a free extra token.

Every token of the answer is still the target's choice. The drafter only saves passes.

**Greedy mode:** a draft is kept when it equals the target's most likely token, so the output is exactly what the target alone would write.

**Sampled mode** (temperature, top-p): **speculative sampling**. A draft d is kept with probability min(1, p(d)/q(d)), where p is the target's distribution and q the drafter's. The first draft not kept is replaced by a token drawn from max(0, p − q), renormalised. The result follows the target's distribution exactly (Leviathan et al. 2023; Chen et al. 2023).

## Why it matters on a phone

Generating a token means reading every weight of the target once, and on a phone that memory traffic is the cost, not the arithmetic. A verify pass over k + 1 positions reads the weights once too, so k accepted drafts cost about one pass instead of k. The drafter's own weights are small: about 55 MB for the tested drafters, next to gigabytes for the targets.

The tested drafters are Google's Gemma 4 multi-token-prediction (MTP) drafters, in the p0pc0rn pack format:
- They read the target's hidden state and its key/value caches. The input to `pre_projection` is [state | embedding], and the `kv_source` in each manifest's `model` names the target layers.
- Their output head is clustered. A small centroid layer picks the best 32 of 2,048 clusters of the vocabulary, and only those clusters' head rows (4,096 tokens) are scored.
- Encodings: [QUANT-q4g32a.md](QUANT-q4g32a.md).

## Wiring target and drafter in an LMA site

1. **Approve the set.** Put the target and its drafter in the site's approved list; the drafter's `pairs_with` holds the target's `repo` and `revision`. A site may run the target alone if the drafter is not picked: the drafter changes speed, never the words in greedy mode.
2. **Pick.** The user picks every file of the set: in one pick (`multiple: true`), or the target first and the drafter after. Verify each model with `verifyModel` against the approved list. Verify-once, the cache and the memory rules apply to each model as to any other.
3. **Load.** Load both with their own `Loader`.
4. **Decode loop.** The drafter proposes k tokens. The target runs the verify pass over k + 1 rows. Keep the agreed prefix plus one token, roll back both models' caches past the rejected drafts, and repeat.

The verify pass, the clustered drafter head, top-k/top-p sampling and the min(1, p/q) acceptance are implemented in [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels): `gemvM` and the `llm` namespace. It documents them in [docs/DESIGN.md](https://github.com/ArtistJoshHughes/p0pc0rn-kernels/blob/main/docs/DESIGN.md), section "Language-model kernels". How p0pc0rn splits a target across two processes on iOS 27, with its drafter beside it: [Two processes, one model](https://github.com/ArtistJoshHughes/p0pc0rn/blob/main/docs/TWO-PROCESSES.md).
