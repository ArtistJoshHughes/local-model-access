<!-- SPDX-License-Identifier: Apache-2.0 -->
Local Model Access (LMA) - created by artist Josh Hughes.

# Examples (optional)

Everything here is built on the core in [../lma](../lma) and is not needed to follow the standard.

## site-kit/: a site in one config file

[site-kit/site.js](site-kit/site.js) reads a `site.config.json` holding everything one site needs. Start from [site.config.example.json](site-kit/site.config.example.json).

| Field | What it holds |
|---|---|
| `site` | `id` (it keys the site's cache), `name`, `job` |
| `prompt.system` | The pre-prompt for the site's job |
| `skills[]` | `name`, `description`, `parameters` (JSON Schema), and a `handler`: `{ "type": "local", "fn" }` for a function you bind, or `{ "type": "http", "api", "method", "path" }` for one of your APIs. `site.tools()` returns them as tool definitions for a model runner. |
| `apis` | `{ name: { baseUrl, headers, credentials } }` for the `http` skills. `headers` are public: no secret keys. No cookies are sent unless `credentials` says so |
| `models[]` | The approved models: identities from [../registry/models.json](../registry/models.json) or `tools/modelfile.py hf`, with any hints |
| `kernels` | `{ "backend": "none" \| "basic" \| "p0pc0rn-kernels" \| "custom", "module", "url", "options" }` |
| `memory.maxSliceMB` | The largest slice the loader reads at once (16 by default) |

Skills, APIs and models are all data: add a new skill or approve a new model by adding an entry, with no change to the code.

### Kernel backends

A backend module's default export is `async createBackend(options)`, which returns an object with `alloc`, `write`, `ops.<op>`, `sync`, `read`, `free`, `destroy`, `supports` and `stats`. Each method is documented in [site-kit/kernels/stub.js](site-kit/kernels/stub.js).

- [basic.js](site-kit/kernels/basic.js) is a small WebGPU reference: an int8 matrix-vector product with a per-row scale.
- [p0pc0rn-kernels.js](site-kit/kernels/p0pc0rn-kernels.js) is an adapter for [p0pc0rn-kernels](https://github.com/ArtistJoshHughes/p0pc0rn-kernels) (MPL-2.0, its own repository). It imports the library at run time and copies nothing in. Its matrix-vector kernels take weights in the library's own encodings. This adapter has not been run end to end yet.

## chat-widget/: `<lma-chat>` on a plain page

A help page with the chat widget in the lower right. It uses the stand-in engine, a help/FAQ `site.config.json` (approved: the two tested packs and the test vectors, so it runs anywhere), and one site-kit skill, `search_docs`, for the tool-call path: try `/tool search_docs {"query": "refund"}`. Serve the repository (`node tools/serve.mjs . 8000`) and open `/examples/chat-widget/`.

## demo/: two sites, one model file

- **The download page** ([get.html](demo/get.html)):
  - **B**, a normal browser download, is recommended.
  - **A** downloads into the page with resume, then uses Save to Files. It is for small files only: on iOS, very large files fail through the share sheet, so real models use B.
- **Site A** checks the file and measures how fast the device reads it.
- **Site B** streams the weights into the GPU and checks every answer against the test model's metadata.
- Each site is built from the site kit with its own `site.config.json`: skills, pre-prompt, approved models and kernel backend.

To run it:

```sh
python tools/make_test_models.py examples/demo/models          # the demo's test models (add --big for 1 GB); needs numpy
python tools/make_test_models.py approve examples/demo/models examples/demo/site-a/site.config.json examples/demo/site-b/site.config.json
node tools/serve.mjs . 8000                                    # then open http://localhost:8000/examples/demo/
```

Copyright 2026 Josh Hughes.
