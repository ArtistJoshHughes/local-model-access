<!-- SPDX-License-Identifier: Apache-2.0 -->
Local Model Access (LMA) - created by artist Josh Hughes.

# AGENTS.md: Local Model Access for AI coding agents

This file is for AI coding agents (Claude, GPT, Gemini, Copilot and others) asked to add LMA to a website or to build on it. It is self-contained: you can integrate LMA from this page without reading the source. [SPEC.md](SPEC.md) is the normative text; where this page and the spec differ, the spec wins.

## What LMA is

LMA lets many websites use one AI model file that the user keeps on their own device, for example in the iPhone's Files app. The file is a GGUF or safetensors file exactly as published on Hugging Face, identified by its `sha256` and `size`, or a model in a documented extension format: `pocketflux.pack/1`, the quantised packs the p0pc0rn app runs on iPhone. A site asks for it with a file picker, checks it against the site's own list of approved models, and reads its tensors in bounded slices straight into GPU memory. The file is never uploaded, never copied, and never read whole into memory.

**Use LMA when** a site wants an on-device model for its own job (help, search, translation, prompt editing and so on), wants the visitor's data to stay on the device, and is happy for the visitor to bring the model file once and reuse it on every site that approves it.

**Do not use LMA when** the model must run on a server, when the user cannot be asked to tap a file picker (background jobs, bots), or when the site needs a model format other than GGUF, safetensors or a documented extension format such as `pocketflux.pack/1`. LMA is also not a way to distribute a model; publishers do that with a normal download.

## Invariants: never break these

1. **Never upload or copy the user's model file.** Do not send any of its bytes over the network. Do not write it, or any part of it, into OPFS, IndexedDB, Cache Storage or localStorage. The only thing a site may store is the verification result (a `sha256`, a few bytes).
2. **Never read a whole file into memory.** No `file.arrayBuffer()`, `file.text()` or `new Response(file)` on a model file. Read with `Loader` (or `File.slice`) in bounded slices (16 MiB by default), a tensor at a time, and drop each slice once it is on the GPU.
3. **A user gesture is required for every pick.** The file comes only from an `<input type="file">` that the user taps, once per visit. There is no lasting handle; do not try to fake one.
4. **Verify against the site's allowlist before use.** Run the instant check, then the full SHA-256 (once per file, then cached). Never run a model whose files have not matched an approved identity.
5. **Ignore unknown capability values and unknown entry fields.** Never reject an approved-model entry because it has a field or a `capabilities` value you don't recognise.
6. **Keep the credit and the NOTICE.** Keep the line "Local Model Access (LMA) by artist Josh Hughes" in the site's credits or about screen. Keep [NOTICE](NOTICE) with any copy of the code.
7. **Do not paste code from the p0pc0rn app into an LMA implementation; use its published engine only under the p0pc0rn licence.**

## Integration recipe for an existing site

1. **Drop in the core.** Copy [lma/](lma) into the site, for example as `/lma/`. It is plain ES modules with no dependencies and no build step. Import from `lma/index.js`. Serve the site over HTTPS (WebGPU needs a secure context; `localhost` counts).
2. **Write the approved list.** For each model your job needs, get its identity:
   - from [registry/models.json](registry/models.json): copy the entry; or
   - from Hugging Face: run `python tools/modelfile.py hf ORG/REPO FILE [FILE ...] [--revision SHA]`. It prints an entry from Hugging Face's metadata, with `files: [...]` for split or sharded models. Nothing is downloaded.

   Add `name`, and `capabilities` from the model card (an open list, such as `["text", "vision", "tools"]`). Approve only models you have tested for the job.
3. **Write the site's own logic** (optional kit: [examples/site-kit](examples/site-kit)): the skills (tools) the model may call and the APIs they hit, the pre-prompt, and the kernel backend. With the kit, all of it lives in one `site.config.json` (see "The site kit" below).
4. **Wire the picker button.**

   ```js
   import { makePicker, inspectFiles, verifyModel, Loader, FormatError } from '/lma/index.js';
   makePicker(document.querySelector('#model-button'), { label: 'Use your model file', multiple: true, onPick: (files) => use(files) });
   ```

5. **Verify once, then cached.**

   ```js
   async function use(files) {
     if (!files || !files.length) return;                                     // null when the picker was cancelled
     let model;
     try { ({ model } = await inspectFiles(files)); }                        // instant: headers only
     catch (e) { return show(e instanceof FormatError ? e.message : 'Could not read that file'); }
     const v = await verifyModel(files, { approved, ns: 'my-site', model,      // ns: one per site (it names the cache)
       onProgress: ({ file, done, total }) => progress(file, done / total) });  // the first visit hashes; later visits are instant
     if (!v.ok) return show(v.problems.join('\n'));                           // includes "also pick X" for a missing part
     await load(model);
   }
   ```

6. **Load tensors to WebGPU, one at a time.**

   ```js
   async function load(model) {
     const L = new Loader(model.files, model, { maxSliceMB: 16 });
     const backend = myKernels;                                    // anything with alloc(bytes, label) and write(handle, data, byteOffset)
     for (const t of model.tensors) gpu[t.name] = await L.toGpu(t, backend);   // slice by slice; CPU copies released
     showMemory(L.stats.cpuPeak, backend.stats && backend.stats.gpuPeak);       // the spec requires reporting the peak
   }
   ```

7. **Kernels (optional, for speed).** [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels) is a public WebGPU kernel library (MPL-2.0) tuned on phone GPUs, and the recommended way to make an LMA site fast. Link to it; do not copy it in. To use any kernel library behind the same loader, implement the backend interface in [examples/site-kit/kernels/stub.js](examples/site-kit/kernels/stub.js): `alloc`, `write`, `ops.<op>`, `sync`, `read`, `free`, `destroy`, `supports`, `stats`. [p0pc0rn-kernels.js](examples/site-kit/kernels/p0pc0rn-kernels.js) is an adapter that imports the library at run time.
8. **iPhone, iOS 27, models too big for one tab (optional).** This is new with iOS 27 and was first shown in the p0pc0rn app, which runs a 4B model (NVIDIA Nemotron 3.5 Bolt E4B) in a web page this way. A module SharedWorker whose script is served with `Cross-Origin-Embedder-Policy: require-corp` runs in a process of its own, with its own memory budget and WebGPU device. A service worker can add that header to the one worker script on a host that can't set headers. Run part of the layers in that worker and the rest in the page, passing activations between them. Post the picked `File` objects to the worker (they are passed by reference, not copied), and let each process read only its own tensors with its own `Loader`. This combination is untested with LMA so far, and the technique could change in a future iOS. The write-up: [Two processes, one model](https://github.com/ArtistJoshHughes/p0pc0rn/blob/main/docs/TWO-PROCESSES.md).

## The tested models: p0pc0rn packs and their drafters

The first four entries of [registry/models.json](registry/models.json) are the models that have actually run on iPhone, in the p0pc0rn app, as these exact packs:

| Entry | Role | Format and quant |
|---|---|---|
| `JoshHughes/p0pc0rn-assistant-nemotron-bolt-e4b-q4` @55c29c2e | target: NVIDIA Nemotron 3.5 Bolt E4B, text | `pocketflux.pack/1`, `q4all`: everything `q4g32a` (4-bit, groups of 32) |
| `JoshHughes/p0pc0rn-assistant-drafter-gemma4-e4b` @e34437d0 | drafter for the entry above | 8-bit layers, 4-bit clustered head |
| `JoshHughes/p0pc0rn-assistant-gemma4-e2b-q4` @141f0e86 | target: Google Gemma 4 E2B-it, text + vision | balanced 4/8-bit (GPTQ 4-bit MLPs, 8-bit attention) |
| `JoshHughes/p0pc0rn-assistant-drafter-gemma4-e2b` @a75bfc34 | drafter for the entry above | 8-bit layers, 4-bit clustered head |

- **Using them.** Copy the entries you need into the approved list. The user downloads the pack's files (all of them, from the Hugging Face repo) and picks `manifest.json` together with every file it lists. `readModel` detects the pack, and `verifyModel` checks it against the manifest's identity: hashing every file once, or `lazy: true` with `new Loader(model.files, model, { verify: 'chunks' })` to check each 16 MiB chunk as it is read.
- **The weights** are 4-bit (or 8-bit) and stay that way in GPU memory. Your kernels dequantise per group as they read. The method, the exact byte layouts and a worked example: [docs/QUANT-q4g32a.md](docs/QUANT-q4g32a.md). Kernels that read these encodings: [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels). A CPU reference to check yours: `lma/ext/pocketflux.js`.
- **Drafters** make replies faster without changing them in greedy mode (speculative decoding). Approve the drafter beside its target (`role: "drafter"`, `pairs_with`), let the user pick both, and run the draft-then-verify loop: [docs/DRAFTER.md](docs/DRAFTER.md). A site can always run the target alone.
- **A new pack** (another model quantised the same way): publish `manifest.json` and its files, then add an entry `{ repo, revision, format: "pocketflux.pack/1", manifest: { file, sha256, size }, size, quant, capabilities, base_model }`. Here `sha256` is the manifest's and `size` is the total of the files it lists. Test it on a phone before marking it tested.

## Built to grow: new models, skills and pre-prompts

Everything that changes over time is data. None of it needs a change to `lma/`.

- **A new model:** run `python tools/modelfile.py hf ORG/REPO FILE...`, add the printed entry to the site's approved list, and set its `capabilities`. If the file is GGUF or safetensors, the core already reads it. Contribute it to `registry/models.json` with an honest `tested` value.
- **A new skill:** add an entry to `skills` in the site config (`name`, `description`, JSON Schema `parameters`, and a `handler` of type `local` or `http`). Bind a function for a local skill with `site.bind({ name: fn })`, or add the API under `apis` for an http skill. `site.tools()` hands the skills to a model runner as tool definitions.
- **A new pre-prompt:** edit `prompt.system`. Tie skills and prompts to what a model can do through its `capabilities`: for example, enable an image skill only when the entry includes `"vision"`.
- **New capabilities from future models** (vision, audio, longer context, tool calling, and later ones) are new strings in `capabilities`. Old sites ignore them; new sites use them.

## Public API of lma/

Everything is exported from `lma/index.js`. A "file" is a `File` or any `Blob` with `size`, `slice()` and, for an index file, `text()`; `name` and `lastModified` are used where present.

### formats.js

| Export | Signature and result |
|---|---|
| `readModel(files, opts?)` | `Promise<Model>`. Accepts one model as picked, in any order: a single file, all parts of a split GGUF, safetensors shards (with or without their `.index.json`), or a `pocketflux.pack/1` (its `manifest.json` and every file it lists). `Model = { format: 'gguf' \| 'safetensors' \| 'pocketflux.pack', name, architecture, metadata, parts: [{ name, size, ... }], files: File[] (data files in part order; the index or manifest excluded), tensors: Tensor[], index: { tensors, totalSize } \| null, headerReads, readMs }`. A pack also has `manifestFile`, `manifest` and `ignored` (picked files the manifest does not list) |
| `readHeader(file, opts?)` | `Promise<Header>` for one file: `{ format, version, metadata, tensors, dataStart, alignment, headerBytes, headerReads, headerBytesRead, name, architecture, fileSize, readMs }` |
| `sniff(file)` | `Promise<'gguf' \| 'safetensors' \| null>` from the first bytes |
| `tensorMap(model)` | `Map<name, Tensor>` |
| `FormatError` | `Error` with `code`: `not-model`, `not-gguf`, `not-safetensors`, `version`, `bad-header`, `truncated`, `mixed`, `missing-part`, `duplicate-part` |
| `GGML_TYPES`, `ST_DTYPES` | dtype tables (`{ id: [name, blockElements, blockBytes] }`, `{ ST: [name, bytes] }`) |

`Tensor = { name, dtype ('f16', 'bf16', 'q4_k', 'i8', ...; for a pack, the encoding such as 'q4g32a.t64'), shape (row-major, outermost first), offset (bytes from the start of its file), size (bytes), part (index into model.files), ggmlType?, enc?, src? }`. `opts`: `chunk` (bytes per header read, 1 MiB), `maxArray` (keep numeric GGUF arrays up to this length, 4096), `maxStrings` (64). Longer arrays come back as `{ skipped: true, of, length }`, so tokenizer vocabularies are counted, not loaded. 64-bit integers are Numbers when exact, else decimal strings.

### verify.js

| Export | Signature and result |
|---|---|
| `inspectFiles(files)` | `Promise<{ model: Model, ms, files: [{ name, size, type, lastModified }] }>`. Throws `FormatError` |
| `verifyModel(files, { approved, ns, model?, onProgress?, noCache?, lazy? })` | `Promise<{ ok, entry, how: 'cache' \| 'hashed' \| 'manifest' \| 'rejected', files: [{ name, size, sha256, how }], missing: string[], hash: { ms, bytes, readMs, hashMs, peakBytes, where } \| null, candidates, problems: string[], model, ms, lazy? }>`. `ok` only when every file matched an approved entry and none of its files is missing. `onProgress({ file, done, total, MBs })`. Without `model`, it reads the headers itself and can throw `FormatError`. For a pack, the entry's `manifest.sha256` must match. With `lazy: true`, only the manifest is hashed (`how: 'manifest'`), and the chunks are checked as they are read |
| `verifyChunk(model, part, i)` | `Promise<boolean>`: chunk i of pack file `part` against the manifest's `chunk_sha256[i]` |
| `verifyFile(file, opts)` | `verifyModel([file], opts)` |
| `candidates(model, approved)` | The entries this model could be, by format, sizes and `expect` (no hashing) |
| `partsOf(entry)` | `[{ file, sha256, size }]`, whether the entry has one file or `files` |
| `fileSha256(file, { ns, noCache, onProgress })` | `Promise<{ sha256, how, hash? }>`: cached, else hashed and remembered |
| `hashFile(file, { onProgress, chunk = 8 MiB, worker = true })` | `Promise<{ sha256, bytes, ms, readMs, hashMs, peakBytes, where: 'worker' \| 'main thread' }>` |
| `fingerprint(file)` | `Promise<hex>` over the size and the first, middle and last MiB. Stored with a cached result and checked on every cache hit; never the identity and never a lookup key |
| `cacheKey(ns, file)`, `forget(ns, file)` | The localStorage key; drop one cached result |

### loader.js

`new Loader(files, model, { maxSliceMB = 16, verify })` (the floor is 64 KiB). `files` must be `model.files` (the files the model was read and verified from), or a single File for a `readHeader` model; any other files throw. `verify: 'chunks'` (packs only) hashes every chunk a read touches the first time, checks it against the manifest, returns the bytes it hashed, and throws on a mismatch; `stats.chunksVerified` counts them. Every pack gets chunk checks unless `verifyModel` hashed all of its shards for this same model object; a pack verified with `lazy: true` always gets them.

| Member | Behaviour |
|---|---|
| `tensor(name)` | The `Tensor`; throws `Error` if absent |
| `read(start, end, part = 0)` | `Promise<ArrayBuffer>`. Throws `Error('refused: ...')` above the slice limit and `RangeError` outside the file. Call `release(ab)` after |
| `withSlice(start, end, fn, part = 0)` | Reads, calls `fn(ab)`, releases |
| `withTensor(name, fn)` | A whole tensor no larger than the slice limit, released after `fn` |
| `toGpu(nameOrTensor, backend, handle?, dstOffset = 0)` | `Promise<handle>`. Calls `backend.alloc(bytes, label)` unless a handle is given, then `backend.write(handle, ArrayBuffer, byteOffset)` once per slice. At most two slices are in memory |
| `stats` | `{ reads, bytesRead, readMs, cpuNow, cpuPeak, uploads, bytesUploaded, refused }` |

### picker.js

| Export | Signature and result |
|---|---|
| `makePicker(host, { label, multiple = false, onPick })` | Appends a `<label class="lma-pick">` that wraps a real file input (style the class). It has no `accept` filter. `onPick(fileOrFiles, info)` is called with a `File`, or `File[]` when `multiple`, or `null` if cancelled. `info = { cancelled, tapToFileMs, closedToFileMs, closedEvent }`. Returns `{ el, setLabel(text), setDisabled(bool) }` |
| `storageSnapshot()` | `Promise<{ t, usage, quota, persisted }>`. Use it to show the site's own storage did not grow |
| `describeFile(file)` | `{ name, size, type, lastModified, lastModifiedAgoS }` |

### ext/pocketflux.js (exported as `pocketflux`)

A CPU reference for the pack encodings, for checking kernels (see [docs/QUANT-q4g32a.md](docs/QUANT-q4g32a.md)):
- `dequant(enc, bytes, shape)` returns a `Float32Array` (a `Uint32Array` for `u32`);
- `dequantTile(enc, tileBytes, K)` decodes one 64-row tile of a `.t64` tensor;
- `dequantRow(enc, rowBytes, K)` decodes one row of a `.rows` tensor;
- `readRow(loader, tensor, n)` reads and decodes row n, with one tile or one row in memory;
- `rowBytes(enc, K)`, `tensorBytes(enc, shape)` and `f16(bits)` are helpers.

### sha256.js

`new Sha256().update(bytes).update(...).hex()` (or `.digest()` for a `Uint8Array(32)`) is an incremental SHA-256. `sha256hex(bytes)` hashes one buffer. `hash-worker.js` is internal: `hashFile` starts it.

## Drop-in UI (optional)

Use the components when a site wants LMA's standard look with no UI work. Stay headless (`makePicker`, `verifyModel`, `Loader`) when the site has its own design system. Both components are plain custom elements in a shadow root: import the file and use the tag.

### `<lma-model>` (lma/ui/lma-model.js)

| | |
|---|---|
| Attributes | `approved` (URL of a site config `{ models }`, a registry, or a JSON array), `ns` (the cache namespace; default `config.site.id` or the host; the cache is per origin, so sites that share an origin, such as project pages on one host, share it), `lazy` (packs: verify chunks as they are read), `get-url` (overrides the "Get the model" link) |
| Properties | `models` (the approved entries; set it directly instead of `approved`), `config`, `state` (`'empty' \| 'checking' \| 'ready' \| 'not-approved' \| 'error'`, read-only), `ready` (a Promise of the `lma-ready` detail) |
| Methods | `reset()` (back to empty, as "Change" does), `preview(state, { percent, name, message })` (render a state without a file, for design and screenshots) |
| Events (bubbling, composed) | `lma-ready` `{ files, picked, model, entry, verified }` (`files` is `model.files`, the verified files in part order; `picked` is everything the user chose), `lma-change` `{ state, previous }`, `lma-error` `{ code, message, error?, verification? }` |
| CSS variables | `--lma-accent`, `--lma-accent-ink`, `--lma-bg`, `--lma-fg`, `--lma-muted`, `--lma-border`, `--lma-ok`, `--lma-bad`, `--lma-track`, `--lma-radius`, `--lma-font` |
| Parts | `card`, `title`, `status`, `progress`, `button`, `change`, `link`, `privacy`, `mark` |

The wording of every state is fixed in [SPEC.md §12](SPEC.md#12-ux-guidance-informative). It accepts several files in one pick (split GGUF, shards, packs), and a target plus its drafter, one model at a time.

### `<lma-chat>` (lma/ui/lma-chat.js)

| | |
|---|---|
| Attributes | `config` (URL of a site config), `open` (boolean), `position` (`bottom-right`, the default, or `bottom-left`) |
| Properties | `config` (the object: `site`, `prompt.system`, `models`, `skills`, `sampling`, and `chat: { title, subtitle, starters[] }`), `engine` (the slot below), `site` (a site-kit `Site`: its skills become the tools and run tool calls), `onToolCall(call)` (used if there is no `site`), `open`, `messages` (the conversation), `model` (the inner `<lma-model>`) |
| Methods | `send(text?)`, `stop()` |
| Events | `lma-chat-open`, `lma-chat-close`, `lma-chat-ready` `{ entry, model }`, `lma-message` `{ role, text }`, `lma-tool` `{ call, result }`, plus the inner `lma-ready` / `lma-change` / `lma-error` |
| CSS variables | as above, plus `--lma-bubble`, `--lma-shadow`, `--lma-gap` |
| Parts | `launcher`, `panel`, `scrim`, `header`, `close`, `model`, `compose`, `input`, `send`, `stop`, `message`, `mark` |

Behaviour:
- The launcher is 56 px, lower right, inside the safe-area insets. The panel is 380 × 560 on a desktop and a bottom sheet over a scrim on a phone.
- First use shows `<lma-model>`. On `lma-ready` it calls `engine.load(...)` and becomes the chat.
- Enter sends and Shift+Enter makes a new line. Stop aborts. The reply streams into an `aria-live` log.
- Focus stays inside the panel while it is open. Escape closes it, and focus returns to the launcher.

### The engine slot

```js
const engine = {
  async load({ files, model, entry, verified, onProgress }) { /* new Loader(files, model): build GPU buffers with toGpu, a tensor at a time */ },
  async *generate({ messages, system, tools, signal, sampling }) { yield { text: 'Hel' }; yield { text: 'lo' }; /* or yield { tool_call: { name, arguments } } */ },
  unload() { /* free GPU memory */ },
};
document.querySelector('lma-chat').engine = engine;
```

- `messages` are `{ role: 'user' | 'assistant' | 'tool', content }`. After a `tool_call`, the widget runs the skill (`site.callTool`) and calls `generate` again with a `{ role: 'tool', name, content }` message, for up to four rounds.
- **The stand-in** (`lma/ui/engines/stand-in.js`, `createStandIn()`) runs no model. It reads the header, streams a reply that names the model, and turns `/tool NAME {json}` into a `tool_call`. Use it to build and test the page, then swap in a real engine.
- **Real engines:** any runtime that implements the slot. For fast WebGPU kernels that read GGUF-style and p0pc0rn-pack encodings, see [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels) (MPL-2.0, link to it). Do not bundle or adapt the p0pc0rn app's engine; it stays in p0pc0rn under its own licence.

### The four use cases with `<lma-chat>`

| Use case | `chat` | Skills | Approve |
|---|---|---|---|
| Help and FAQ (the natural default) | `{ title: "Help", starters: ["How do refunds work?"] }` | `search_docs` (local) | a text model |
| Prompt editor helper | `{ title: "Improve my prompt" }` | `read_prompt`, `write_prompt` (local) | a text model |
| Search | `{ title: "Search" }` | `search` (http to your search API) | a text model, and an `embeddings` one for re-ranking if you have it |
| Translation | `{ title: "Translate", starters: ["Translate this page"] }` | `page_text` (local), `glossary` (http) | a multilingual model |

Pre-prompts for each are in "Use-case patterns" below. A working help/FAQ page: [examples/chat-widget/](examples/chat-widget).

## The site kit (optional)

[examples/site-kit/site.js](examples/site-kit/site.js) reads one `site.config.json`. Start from [site.config.example.json](examples/site-kit/site.config.example.json).

- `site`: `{ id, name, job }`
- `prompt.system`: the pre-prompt
- `skills[]`: `{ name, description, parameters, handler: { type: 'local', fn } | { type: 'http', api, method, path } }`
- `apis`: `{ [name]: { baseUrl, headers, credentials } }`. `headers` are public (the config is served to every visitor): never put a secret key there. `credentials` defaults to `'omit'` (no cookies)
- `models[]`: the approved list
- `kernels`: `{ backend: 'none' | 'basic' | 'p0pc0rn-kernels' | 'custom', module, url, options }`. `module` and `url` are imported and run as code with the page's rights: point them only at code you control or have pinned
- `memory.maxSliceMB`

The API:
- `const site = await loadSite(url)`
- `site.bind(handlers)`, `site.tools()` (OpenAI-style tool definitions), `await site.callTool(name, args, ctx)`
- `await site.loadKernels()`
- properties: `site.systemPrompt`, `site.models`, `site.id`; set `site.confirm = async ({ skill, method, url, args }) => boolean` to ask the user before any skill that is not a GET

Tool calls come from the model, so a prompt in a page or a document can steer them. The kit passes on only the arguments a skill declares, keeps every http request under its API's `baseUrl` (no `..` or empty path parts, no overriding the skill's own query), and sends no cookies by default.

## Use-case patterns

Starting points written as kit config fragments. Swap in your own APIs and prompts, and approve models whose `capabilities` fit.

**1. Site help and FAQ**: answers from the site's own docs.

```json
{ "prompt": { "system": "You are the help assistant for {SITE}. Answer only from the passages search_docs returns, and name the page each answer comes from. If the passages do not answer the question, say so and point to {CONTACT_URL}." },
  "skills": [ { "name": "search_docs", "description": "Find passages in this site's help pages", "parameters": { "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] },
                "handler": { "type": "local", "fn": "search_docs" } } ] }
```

**2. Prompt editor helper**: improves a prompt in place, for example for an image or video generator.

```json
{ "prompt": { "system": "Rewrite the user's prompt for {GENERATOR}. Keep their subject and intent. Add concrete detail about subject, setting, light, camera and style, and remove contradictions. Reply with the new prompt only." },
  "skills": [ { "name": "read_prompt", "description": "The text currently in the prompt box", "parameters": { "type": "object", "properties": {} }, "handler": { "type": "local", "fn": "read_prompt" } },
              { "name": "write_prompt", "description": "Replace the prompt box text", "parameters": { "type": "object", "properties": { "text": { "type": "string" } }, "required": ["text"] }, "handler": { "type": "local", "fn": "write_prompt" } } ] }
```

**3. Search**: query rewriting and re-ranking on the device.

```json
{ "prompt": { "system": "Turn the visitor's words into up to three search queries for {SITE}'s catalogue. Then rank the results search returns by how well they answer what the visitor meant, best first, as a list of ids." },
  "skills": [ { "name": "search", "description": "The site's own search", "parameters": { "type": "object", "properties": { "q": { "type": "string" } }, "required": ["q"] },
                "handler": { "type": "http", "api": "catalogue", "method": "GET", "path": "search" } } ],
  "apis": { "catalogue": { "baseUrl": "https://{SITE}/api/" } } }
```

**4. On-device translation**: pages, user content or chat in the visitor's language.

```json
{ "prompt": { "system": "Translate the text into {LANGUAGE}. Keep names, product terms from the glossary, numbers and placeholders like {{1}} unchanged. Reply with the translation only." },
  "skills": [ { "name": "glossary", "description": "Terms this site never translates", "parameters": { "type": "object", "properties": {} }, "handler": { "type": "http", "api": "site", "method": "GET", "path": "glossary.json" } },
              { "name": "page_text", "description": "The visible text blocks of this page, numbered", "parameters": { "type": "object", "properties": {} }, "handler": { "type": "local", "fn": "page_text" } } ] }
```

## Proving conformance

- Run `node tests/conformance.test.mjs`. It runs the core against [test-vectors/](test-vectors) and ends `ALL OK: LMA-conformant (Local Model Access by artist Josh Hughes)`.
- For another implementation, read `test-vectors/vectors.json`. Check each vector's header facts and identities, and refuse each input in `refuse` with its error code.
- Then go through the checklist in [CONFORMANCE.md](CONFORMANCE.md). An implementation that meets it may use the line **LMA-conformant (Local Model Access by artist Josh Hughes)**.

## Common mistakes

- **Saving big models through the share sheet.** On iOS, very large files fail through `navigator.share({ files })`. Get model files onto the device with a normal browser download (a Hugging Face `resolve/.../file?download=true` link). It works in a Safari tab; Home Screen web apps have no downloader.
- **Picking one part of a split model.** All parts of a split GGUF must be picked together (`multiple: true`). The same goes for every shard of a sharded safetensors model; the index is optional. `readModel` throws `missing-part`, and `verifyModel` lists what is `missing`.
- **Setting `accept=".gguf"`.** iOS greys out unknown types and the user can't pick the file. Leave `accept` off; the header check identifies the file.
- **Matching by file name.** Files get renamed (for example "model (1).gguf"). Identity is `sha256` + `size`.
- **"Caching" the model in OPFS or IndexedDB.** That is a copy, which is forbidden. Cache only the verification result, as `verifyModel` already does.
- **Calling `file.arrayBuffer()` on a model.** It loads gigabytes into memory and the tab dies. Use `Loader`.
- **Opening the picker without a tap.** Browsers block `input.click()` outside a user gesture. Use `makePicker`, whose button is the input itself.
- **Pinning only a repo name.** An approved entry needs the file's `sha256` and `size`, plus the `revision`, so an update upstream can't silently change what runs.
- **Rejecting unknown capabilities or roles.** Ignore them, so newer entries keep working on older sites.
- **Picking part of a pack.** A `pocketflux.pack/1` is `manifest.json` plus every file it lists, including the tokenizer files, all picked together. Download the whole Hugging Face repository.
- **Hashing a 5 GB pack on every visit.** Verify once (the cache makes later visits instant), or use `lazy: true` with a chunk-verifying `Loader`.
- **Requiring the drafter.** It is optional: without it, the target is slower, never different in greedy mode.
- **Reporting nothing about memory.** The spec requires showing the peak: `Loader.stats.cpuPeak` and your backend's GPU bytes.

## Credit

Local Model Access was created by artist Josh Hughes. Keep this line in the credits or about screen of any site or product that uses it:

> Local Model Access (LMA) by artist Josh Hughes

Keep [NOTICE](NOTICE) with the code (Apache-2.0, section 4(d)), and credit the specification under CC BY 4.0. To cite LMA, use [CITATION.cff](CITATION.cff).

Copyright 2026 Josh Hughes. Code: Apache-2.0. Specification: CC BY 4.0.
