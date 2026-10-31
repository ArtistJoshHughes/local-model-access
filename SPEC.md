<!-- SPDX-License-Identifier: CC-BY-4.0 -->
Local Model Access (LMA) - created by artist Josh Hughes.

# Local Model Access (LMA)

**Specification 1.0** · Copyright 2026 Josh Hughes

This specification is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) ([LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)). You may copy, adapt and implement it, commercially too, as long as you credit "Local Model Access (LMA) by artist Josh Hughes" and link to the licence. The code in this repository is under Apache-2.0 ([LICENSE](LICENSE)).

MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119. Sections 9, 11, 12 and 13 are informative; the rest is normative.

## 1. Scope

A person keeps one copy of an AI model on their device, in the system's file storage (the Files app on iPhone). Any website that can do a job with that model asks for the file with a file picker, checks it against its own list of approved models, and reads it where it is. No site downloads its own copy, and no site keeps one.

This specification covers the model files (2), their identity (3), each site's approved models (4), how a file reaches the device (5), picking it (6), verifying it (7) and the memory rules for reading it (8). It does not cover what a site does with the model: its prompts, tools, kernels and user interface are its own.

## 2. Model files

- A model file is a **GGUF** (version 2 or 3) or **safetensors** file, exactly as its publisher released it. A site MUST NOT require a repacked copy or a container of its own.
- A reader identifies the format from the first bytes, never from the file name or extension, because files get renamed on the way:
  - GGUF: bytes 0-3 are `GGUF`, then a little-endian u32 version, 2 or 3.
  - safetensors: a little-endian u64 header length *N* (2 to 100 MiB), then *N* bytes of JSON beginning with `{`.
- A reader finds tensors from the header alone, as (file, offset, size):
  - GGUF: each tensor info's offset is relative to the data start, which is the end of the header rounded up to `general.alignment` (default 32). Its size follows from its ggml type's block size.
  - safetensors: each tensor's `data_offsets` are relative to byte 8 + *N*.
- A reader MUST reject a file whose header does not parse, or whose tensors run past the end of the file (an incomplete download).

### 2.1 Models in several files

Large models are published split. A reader MUST accept the parts in any order and merge them into one model.

- **Split GGUF**: files named like `name-00001-of-00003.gguf`. Every part is a valid GGUF file whose header carries `split.no` (u16, from 0), `split.count` (u16) and `split.tensors.count` (i32, the total across all parts); these are the keys llama.cpp's gguf-split writes. The model's metadata is in part 0. A reader MUST check that every part is present exactly once and that the tensor count matches.
- **Sharded safetensors**: shard files (usually `model-00001-of-00002.safetensors`), each a complete safetensors file, plus an optional index (`model.safetensors.index.json`: `{ "metadata": { "total_size" }, "weight_map": { tensor: shard } }`). A reader MUST reject a tensor that appears in two shards and, when the index is picked, MUST check that every tensor it names is present.
- Part numbers in file names are hints. The headers, the index and the approved identity (section 4) decide whether a model is complete.

### 2.2 Extension formats

GGUF and safetensors are LMA's core formats; every LMA reader handles them. An **extension format** is a documented format for models published in another layout. A site MAY approve models in an extension format it supports, and MUST treat a format it does not support like any other file it cannot read.

**`pocketflux.pack/1`** is the format of the p0pc0rn app's quantised models, which are the models tested on iPhone ([registry](registry/models.json)). The encodings and quantisers are in [docs/QUANT-q4g32a.md](docs/QUANT-q4g32a.md).

- **Files.** A `manifest.json` plus the files it lists: shard files (`.pfw`, raw tensor data) and other files such as a tokenizer. The manifest is detected by content (`"format": "pocketflux.pack"`, `"format_version": 1`), not by name.
- **The manifest.** `format`, `format_version`, `pack_id`, `pack_version`, `model` (architecture parameters), `encodings`, `quant_recipe`, `sources` (the source model's identity), `license`, and:
  - `shards[]`: each `{ name, file, kind, bytes, sha256, chunk_bytes, chunk_sha256[] }`. `chunk_sha256[i]` is the SHA-256 of bytes `[i·chunk_bytes, (i+1)·chunk_bytes)` of the file.
  - `tensors{}`: each `{ shard, offset, bytes, shape, enc, src }`, where `enc` is the encoding.
- **Picking.** The user picks `manifest.json` and every file it lists, in one pick, in any order. A reader matches picked files to shards by file name, or by a size no other shard has. It MUST reject a pick with a shard missing or of the wrong size, and MAY ignore files the manifest does not list.
- **Verifying.** The approved identity pins the manifest (section 3), and the manifest pins every file. A site either hashes each file once and compares it with `sha256`, then caches the result (section 7), or verifies lazily: it checks every chunk against `chunk_sha256` the first time it reads it, before using any of its bytes. Lazy verification costs no extra reading.
- **Memory.** The rules of section 8 apply unchanged: tensors are read in bounded slices from the shard files.

**Companion files** (`config.json`, `tokenizer.json` and the like) are small. GGUF files embed them. For safetensors models, a site MAY ship its own copies for the approved revision, or ask the user to pick them with the weights. An approved entry MAY list them (section 4).

## 3. Model identity

One file:

```json
{ "repo": "org/name", "revision": "<40-hex commit>", "file": "name.gguf", "sha256": "<64-hex>", "size": 3349516256 }
```

Several files (section 2.1):

```json
{ "repo": "org/name", "revision": "<40-hex commit>", "files": [ { "file": "...", "sha256": "...", "size": 0 }, ... ] }
```

- `sha256` is the SHA-256 of the whole file. On Hugging Face it is the file's LFS object id, listed on the file's page and by its API. `size` is the file's length in bytes.
- Two files with the same `sha256` and `size` are the same file, whatever they are called and wherever they came from. That is what lets one copy serve every site that approves it.
- `repo`, `revision` and `file` record where it came from. For a file that is not on Hugging Face, `repo` is any `owner/name` the publisher controls and `revision` is any version label.

An extension-format model is identified by its manifest:

```json
{ "repo": "org/name", "revision": "<40-hex commit>", "format": "pocketflux.pack/1",
  "manifest": { "file": "manifest.json", "sha256": "<64-hex>", "size": 185158 }, "size": 4675838692 }
```

Here `size` is the total of the files the manifest lists.

## 4. Approved models

Each site keeps its own list of approved models, its allowlist. Which models can do a site's job is the site developer's decision. This specification names no models, tasks or skills, and limits neither which models a site may approve nor what it uses them for.

An entry is a model identity (section 3) plus optional fields:

| Field | Meaning |
|---|---|
| `format` | `"gguf"` or `"safetensors"` (recommended) |
| `name` | A display name |
| `capabilities` | An open list of what the model can do, e.g. `["text", "vision", "audio", "tools", "embeddings"]` |
| `expect` | Header values the file must carry: `name`, `architecture`, or any metadata key |
| `context`, `minRamGB` | Hints for the site |
| `license`, `licenseLink`, `card` | The model's licence and model card |
| `companions` | Small extra files (section 2.1), each `{ file, sha256, size }` |
| `role`, `pairs_with` | A companion model's part in a set, such as `"drafter"`, and the `{ repo, revision }` of the model it serves (section 4.1) |
| `quant`, `encodings`, `base_model` | How the weights were quantised, the encodings used, and the identity of the source model |
| `tested`, `note` | What was actually tested, and anything else worth saying |

- A site MUST ignore fields it does not know, and capability values it does not know, rather than reject the entry. An older site keeps working with a newer entry, and a newer site with an older one. A new capability needs only a new value in the list, not a change to this specification.
### 4.1 Model sets and companion models

Some jobs run more than one model, for example a target and the **drafter** that makes its replies faster ([docs/DRAFTER.md](docs/DRAFTER.md)). A site approves each model of the set as its own entry. A companion says what it is with `role` and names its model with `pairs_with: { repo, revision }`.
- `role` is open: a site MUST ignore a role it does not know, rather than reject the entry.
- The user picks the set's files in one pick, or one model at a time. Each model is verified and read under the same rules as any other.
- A site MUST NOT require a companion to produce the target's output. A drafter changes speed, not what the target writes.

- [registry/models.json](registry/models.json) is a shared list of identities to copy from. An entry there is not a claim that a model works on any site or device; its `tested` field says what was actually tested.

## 5. Getting the file onto the device

- The route is a **normal browser download** of the published file, for example Hugging Face's `https://huggingface.co/{repo}/resolve/{revision}/{file}?download=true`. On iPhone, Safari's own downloader saves it to Files › Downloads and keeps going in the background. This works in a Safari tab; Home Screen web apps have no downloader.
- Recommended place: **On My iPhone › Models**, a folder the user makes once. Avoid iCloud Drive folders with Optimise Storage on: iOS may remove the local copy and download it again when the file is picked.
- A model host SHOULD serve files over HTTPS with `Content-Type: application/octet-stream`, `Content-Disposition: attachment; filename="..."` and `Accept-Ranges: bytes`.
- **The share sheet is not a route for model files.** A page can download a file into its own origin storage and hand it to the share sheet (`navigator.share({ files })`) to Save to Files. That works for small files, but on iOS very large files fail this way (the page is ended), in a Safari tab and in a Home Screen web app alike. A site MUST NOT depend on this route, and a site that offers it MUST delete its own copy afterwards. Use a normal download.

## 6. Picking the file

- A site gets the files only through a file picker the user opens with a tap (`<input type="file">`, with `multiple` for a model in several files), once per visit. There is no lasting file handle on iPhone, and a site MUST NOT keep a copy to get around that.
- A site SHOULD NOT set an `accept` filter: iOS greys out file types it does not know, such as `.gguf` and `.safetensors`.
- A site MUST NOT upload the files or any part of them, and MUST NOT send their contents anywhere. It MAY send an identity (`sha256`) to its own back end, for example to choose settings, and says so in its privacy policy.

## 7. Verifying: once, then cached

1. **Instant check.** Read the headers only. Check the format, each file's size and any `expect` values against the approved list. Reject at once if no approved entry fits.
2. **Full check, once per file.** Stream each file through SHA-256 a slice at a time, off the main thread where possible, with a progress bar. Compare with the approved `sha256`. A site MUST NOT run a model until every one of its files has matched an approved identity on this device.
3. **Remember.** Store the result (the `sha256`, a few bytes) in the site's own storage, keyed by file name + size + `lastModified`, so a returning visit is instant. Store with it a fingerprint of the size and the first, middle and last MiB, and use a cached result only when the file's fingerprint still matches; otherwise hash the file again. The fingerprint MUST NOT be used as the identity or to look up a result: a file that matches no cache key is hashed in full.
4. **Trust what was checked.** Only the files that were verified are read: an engine and its `Loader` MUST read `model.files`, never other files from the same pick. A pack verified lazily MUST have every chunk checked as it is read, and the bytes used MUST be the bytes that were hashed.

The cache trusts the device: a different file given the same name, size, date and sampled bytes as a verified one would be taken from the cache. Sites that need more MAY verify with `noCache` on every visit.

## 8. Memory rules

- A site MUST NOT read a whole model file into memory: no `arrayBuffer()` of the whole file, no whole-file copy, and no copy in origin storage (OPFS, IndexedDB, Cache Storage).
- A site MUST read in bounded slices (`File.slice`; 16 MiB recommended), a tensor at a time. It uploads each slice to the GPU (or consumes it) and drops the CPU copy before reading much more, holding only a couple of slices at once.
- A site MUST measure its peak: the most file bytes held in memory at once, and the GPU memory it allocated. It MUST report both somewhere a user or developer can see them, such as an about or debug panel.

## 9. Performance (informative)

Nothing here is required. LMA works without either.

- **Kernels.** The GPU kernels that run a model are the site's choice. [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels) is a public WebGPU kernel library (MPL-2.0), tuned on phone GPUs, and is the recommended way to make an LMA site fast. Link to it; don't copy it.
- **iPhone: weights split across a second process (iOS 27).** On iOS 27, a module SharedWorker whose script is served with `Cross-Origin-Embedder-Policy: require-corp` runs in a web content process of its own, with its own memory budget and WebGPU device. A service worker can add that header to the one worker script on a host that can't set headers. So a model too big for one tab can run part of its layers (say, the first half of the stack) in that worker and the rest in the page, passing each token's activations between them. With LMA, the page can post the picked File(s) to the worker; a File is passed by reference, not copied. Each process then reads only its own tensors' slices from the File, so the memory rules above still hold in both processes. This combination has not been tested with LMA yet. The technique was found by testing, not from Safari documentation, and a future iOS may change it. A full write-up: [Two processes, one model](https://github.com/ArtistJoshHughes/p0pc0rn/blob/main/docs/TWO-PROCESSES.md).

## 10. Conformance

[CONFORMANCE.md](CONFORMANCE.md) has a checklist and a set of test vectors with their expected identities. An implementation that meets the checklist and passes the vectors may describe itself as **LMA-conformant (Local Model Access by artist Josh Hughes)**.

## 11. Status (informative)

Version 1.0, October 2026. The format readers match llama.cpp's gguf-py and the safetensors library, on the test vectors and on real models, among them a 13.6 GB GGUF, NVIDIA's two-file Nemotron 3.5 Bolt E4B and Google's Gemma 4 E2B. The `pocketflux.pack/1` reader handles the published p0pc0rn packs, and the documented quantiser reproduces their bytes from the source weights. On an iPhone with iOS 27 Safari, a 1 GB GGUF file downloaded with Safari was picked from Files by two different sites, which:
- received it without a copy;
- checked its header instantly, hashed it once and remembered the result;
- streamed it into WebGPU with exact results;
- used no site storage.

## 12. UX guidance (informative)

So that LMA feels the same on every site, a site's model control should use these states and words. The drop-in `<lma-model>` and `<lma-chat>` in `lma/ui/` do; a custom UI should match them.

| State | Title | Status line | Action |
|---|---|---|---|
| empty | Use an AI model on this device | (none) | **Choose model file**, plus a link: **Get the model** |
| checking (first visit only) | Checking the model… 42 % | This happens once. Next time it is instant. | (a progress bar) |
| ready | Ready · *model name* | (none) | **Change** |
| not approved | This site doesn't use that model | It works with: *the approved models' names* | **Choose another file** |
| error | Use an AI model on this device | One short, actionable line (below) | **Choose another file** |

Error lines, by cause:
- not a model file: "That file isn't a model. Choose a .gguf, .safetensors or pack file."
- incomplete download: "That file looks incomplete. Download it again, then choose it."
- a part missing: "This model comes in several files. Choose all of them together."
- the same file twice: "The same file was chosen twice. Choose each file once."
- files from different models: "Those files are from different models. Choose one model at a time."
- an unreadable format: "This site can't read that file." (or "...that file's format version.")
- anything else: "Something went wrong reading the file. Try again."

Every state also shows, in small text, **"Your model stays on your device."** and a quiet **"Local Model Access"** link.

- Ask for the file with one clear action, and show the checking progress only on the first visit.
- Never ask the user to upload anything. Say plainly that the model stays on the device.
- "Get the model" links to the approved model's normal download: a Hugging Face `resolve` URL for one file, or the repository for a model in several files.

A chat widget opens from a button in the lower corner, which respects the safe-area insets. On a phone it becomes a bottom sheet over a dim scrim. Escape or the close button closes it, and focus returns to the button.

## 13. The engine slot (informative)

LMA does not run models. A page that does plugs a runtime, its **engine**, into this small interface, which `<lma-chat>` uses:

```js
engine = {
  async load({ files, model, entry, verified, onProgress }),   // model.files (the verified files, in part order), readModel's model, the approved entry, verifyModel's result
  async *generate({ messages, system, tools, signal, sampling }), // yields { text } deltas and/or { tool_call: { name, arguments } }
  unload(),
}
```

`messages` are `{ role: 'user' | 'assistant' | 'tool', content }`. `tools` are tool definitions, such as the site kit's `site.tools()`. A `tool_call` is run by the page, and its result comes back as a `{ role: 'tool', name, content }` message on the next `generate`. `signal` stops generation. An engine reads weights through `Loader` from `files`, so the memory rules of section 8 and the checks of section 7 hold. A `tool_call` comes from the model, so the page treats its arguments as untrusted input. Any runtime that implements the slot works: WebGPU kernels such as [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels), WebAssembly, or a native bridge.
