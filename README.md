<!-- SPDX-License-Identifier: Apache-2.0 -->
Local Model Access (LMA) - created by artist Josh Hughes.

# Local Model Access (LMA)

One AI model file on the device, used by every website that needs it.

**Building with an AI coding agent? Point it at [AGENTS.md](AGENTS.md).**

The user downloads a model once, a GGUF or safetensors file as published on Hugging Face, and keeps it in the Files app. A site that can do a job with that model asks for it with one tap on a file picker. It checks the file against its own list of approved models, then reads the weights straight from the file into the GPU, a slice at a time. Nothing is uploaded and nothing is copied: ten sites with the same model cost one download and one copy of storage.

- **Private:** the visitor's data never leaves the device.
- **No per-query server cost:** the model runs on the visitor's own hardware.
- **One download serves every site** that approves the model.

## What sites use it for

1. **Site help and FAQ:** an on-device assistant that answers from the site's own docs.
2. **Prompt editor helper:** rewrites or improves a user's prompt in place, for image and video generators, forms and editors.
3. **Search:** on-site search that understands what the visitor means, with query rewriting and re-ranking on the device.
4. **On-device translation:** pages, user content or chat in the visitor's own language. It is private, has no per-request translation fees, and works offline once the model is in Files.

LMA can also help with accessibility (alt text and summaries), form help and moderation.

## Built to grow

LMA is open-ended. It fixes how a site gets, checks and reads a model file, and nothing about what the site does with it.

- **Your own skills and pre-prompts.** Each site defines its own tools, the APIs they call, and the prompts its job needs. LMA has no list of skills to choose from.
- **Your own approved models.** The allowlist belongs to the site. A new model slots in by adding one entry (its Hugging Face identity), with no change to LMA, as long as the file is GGUF or safetensors.
- **Future capabilities.** An entry's `capabilities` is an open list. Vision, audio, longer context, tool calling, and whatever models do next, become new values there and new skills and prompts on the site. Sites ignore values they don't know, so older sites keep working with newer models, and newer sites with older ones.
- **Starting points, not limits.** The four uses above are examples. Any job a local model can do fits.

## How it works

1. **Get the file:** a normal browser download (Safari saves it to Files › Downloads).
2. **Pick it:** one tap on `<input type="file">`, once per visit. The site receives the file itself, without a copy.
3. **Check it instantly:** read the header (format, name, tensors) and the size, and match them against the site's approved list.
4. **Verify once:** SHA-256 of the whole file, streamed with a progress bar, compared with the approved identity, then remembered. The next visit is instant.
5. **Read it in slices:** one tensor at a time into GPU memory, never the whole file in RAM, with the peak reported.

The full rules are in [SPEC.md](SPEC.md).

## Use it in your site

1. Copy the [lma/](lma) folder into your site. It is plain ES modules, with no dependencies and no build step.
2. Make your approved list. For a Hugging Face file, run `python tools/modelfile.py hf ORG/REPO FILE [FILE ...]` and paste the output, or copy entries from [registry/models.json](registry/models.json).
3. Add a file picker, check the files, and load tensors:

   ```js
   import { makePicker, inspectFiles, verifyModel, Loader } from './lma/index.js';

   const approved = await (await fetch('./approved-models.json')).json();   // your site's own list
   makePicker(document.querySelector('#pick'), { multiple: true, onPick: async (files) => {
     const { model } = await inspectFiles(files);                   // instant: format, header, sizes (throws if not a model)
     const v = await verifyModel(files, { approved, ns: 'my-site', model,
       onProgress: (p) => console.log(p.file, p.done, p.total) });   // SHA-256 once per file, then remembered
     if (!v.ok) return alert(v.problems.join('\n'));
     const L = new Loader(model.files, model, { maxSliceMB: 16 });
     const w = await L.toGpu('blk.0.attn_q.weight', myKernels);     // slice by slice into GPU memory; CPU copies dropped
     console.log('peak file bytes in memory:', L.stats.cpuPeak);
   } });
   ```

4. Serve over HTTPS, open the site on the phone and pick the file.
5. Check your site against [CONFORMANCE.md](CONFORMANCE.md).

**The tested models.** NVIDIA Nemotron 3.5 Bolt E4B and Google Gemma 4 E2B-it have run on iPhone as the p0pc0rn app's 4-bit (E4B) and 4/8-bit (E2B) packs, each with a small drafter for faster replies. They are the first entries in the [registry](registry/models.json). They are in an extension format, `pocketflux.pack/1`, which the core reads. How the quantisation works and the exact byte layouts: [docs/QUANT-q4g32a.md](docs/QUANT-q4g32a.md). How drafters work: [docs/DRAFTER.md](docs/DRAFTER.md).

> **New with iOS 27: a 4B model in a web page on iPhone.** The p0pc0rn app runs NVIDIA's Nemotron 3.5 Bolt E4B in the browser, with no app to install, by splitting the model's layers between the page and a second process that iOS 27 gives a module SharedWorker. The technique was first shown in p0pc0rn. How it works: [Two processes, one model](https://github.com/ArtistJoshHughes/p0pc0rn/blob/main/docs/TWO-PROCESSES.md). Notes for LMA sites: [Performance](#performance).

`myKernels` is any object with `alloc(bytes)` and `write(handle, data, offset)`. That is all the loader needs from your GPU code. [examples/](examples) has a fuller site kit: skills, APIs, pre-prompt and kernel backends in one config file, two demo sites built from it, and a download page.

## Drop-in UI

Two optional, dependency-free web components in [lma/ui/](lma/ui). Sites that want their own UI stay headless with `makePicker` and `verifyModel`.

**`<lma-model>`** is the model control: choose the file, check it (once), say when it is ready.

```html
<script type="module" src="/lma/ui/lma-model.js"></script>
<lma-model approved="./site.config.json"></lma-model>
<script>document.querySelector('lma-model').addEventListener('lma-ready', (e) => start(e.detail));</script>
```

**`<lma-chat>`** is a lower-right chat widget that contains it. Give it an engine (the slot in [SPEC.md §13](SPEC.md#13-the-engine-slot-informative)); the stand-in engine proves the wiring without running a model.

```html
<script type="module">import '/lma/ui/lma-chat.js'; import { createStandIn } from '/lma/ui/engines/stand-in.js';
  document.querySelector('lma-chat').engine = createStandIn();</script>
<lma-chat config="./site.config.json"></lma-chat>
```

Both use the same calm wording ([SPEC.md §12](SPEC.md#12-ux-guidance-informative)). They come in light and dark themes, set through `--lma-*` CSS variables and `::part()`. They have 44 px targets, are keyboard and screen-reader friendly, respect reduced motion, and load no fonts. Try them at `examples/chat-widget/` and `reference/`.

## In this repository

| Path | What it is |
|---|---|
| [AGENTS.md](AGENTS.md) | For AI coding agents: the invariants, the integration recipe, the exact API, use-case patterns |
| [llms.txt](llms.txt) | The llmstxt.org index of this repository |
| [SPEC.md](SPEC.md) | The standard (CC BY 4.0) |
| [CONFORMANCE.md](CONFORMANCE.md) | The checklist and the test vectors |
| [lma/](lma) | The core: format readers (single, split and sharded files), verification and cache, the slice loader, the file picker |
| [reference/](reference) | A small page that runs the core on any model file |
| [test-vectors/](test-vectors) | Small real GGUF and safetensors files, with their identities and expected header facts |
| [registry/](registry) | Community list of model identities, with what has been tested. First: the p0pc0rn packs (and their drafters) tested on iPhone |
| [docs/QUANT-q4g32a.md](docs/QUANT-q4g32a.md) | The `pocketflux.pack/1` format and its 4-bit quantisation: method, byte layouts, a worked example |
| [docs/DRAFTER.md](docs/DRAFTER.md) | Drafters and speculative decoding: faster replies, the same words |
| [tools/](tools) | `modelfile.py` (headers, identities, Hugging Face lookups), `make_test_models.py` (the test vectors), `serve.mjs` (a local server with HTTP Range) |
| [tests/](tests) | `conformance.test.mjs`, the core against the vectors; `site-kit.test.mjs`, the site kit's tool calls; `ui.browser.mjs`, the drop-in UI in headless Chrome |
| [lma/ui/](lma/ui) | Optional drop-in UI: `<lma-model>`, `<lma-chat>`, and a stand-in engine |
| [examples/](examples) | Optional: the site kit, a chat-widget page, and the demo |

Try it locally: `node tools/serve.mjs . 8000`, then open `http://localhost:8000/reference/` and pick a file from `test-vectors/`.

## Performance

These are pointers, not requirements; LMA works without either.

- **Fast kernels:** [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels) is a public WebGPU kernel library (MPL-2.0), tuned on phone GPUs. It is the recommended way to make an LMA site fast. Link to it rather than copying it in. The adapter in [examples/site-kit/kernels/](examples/site-kit/kernels) imports it at run time.
- **iPhone, iOS 27: split a big model across two processes.** On iOS 27, a module SharedWorker whose script is served with `Cross-Origin-Embedder-Policy: require-corp` gets a web content process of its own, with its own memory budget and WebGPU device. A service worker can add that header on hosts that can't set headers. A model too big for one tab can then run some of its layers in that worker and the rest in the page. With LMA, the page can pass the picked File to the worker, so each process reads only its own tensors' slices from it and the memory rules still hold. This combination is untested with LMA so far. The technique was found by testing, so a future iOS may change it. The write-up: [Two processes, one model](https://github.com/ArtistJoshHughes/p0pc0rn/blob/main/docs/TWO-PROCESSES.md).

## Credit

Local Model Access was created by artist Josh Hughes. If you implement it, or your site uses it, please keep this line in your credits or about screen:

> Local Model Access (LMA) by artist Josh Hughes

About the creator: [AUTHOR.md](AUTHOR.md). To cite it, use [CITATION.cff](CITATION.cff); GitHub shows a "Cite this repository" button. Implementations that pass [CONFORMANCE.md](CONFORMANCE.md) may describe themselves as "LMA-conformant (Local Model Access by artist Josh Hughes)".

## Licence

- The code is under the Apache License 2.0 ([LICENSE](LICENSE)). Its [NOTICE](NOTICE) file must stay with copies and derivatives.
- The specification and docs ([SPEC.md](SPEC.md), [CONFORMANCE.md](CONFORMANCE.md), [AGENTS.md](AGENTS.md) and [docs/](docs)) are under CC BY 4.0 ([LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)).
- The test vectors ([test-vectors/](test-vectors)) are CC0-1.0: use them freely.
- [AUTHOR.md](AUTHOR.md) reproduces the creator's biography and is not covered by these licences.
- p0pc0rn-kernels, an optional dependency, is under MPL-2.0 in its own repository and is not included here.
- Local Model Access is a separate work by Josh Hughes, released under Apache-2.0 (code) and CC BY 4.0 (spec). It contains no code from the p0pc0rn app; p0pc0rn and its engine remain under the p0pc0rn licence.

Copyright 2026 Josh Hughes.
