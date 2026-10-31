// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// p0pc0rn-kernels.js - an adapter (interface: stub.js) for the p0pc0rn-kernels WebGPU kernels, an optional dependency with its own licence
// (MPL-2.0) and repository: https://github.com/ArtistJoshHughes/p0pc0rn-kernels. Nothing from it is copied here: the module is imported at
// run time from `url` or from a copy you host next to your site. Importing a module runs its code with your page's rights, so the default
// is pinned to one commit (it cannot change under you); to update, change the commit. For production, host a pinned copy yourself.
//   "kernels": { "backend": "p0pc0rn-kernels", "url": "/vendor/p0pc0rn-kernels/src/index.js" }
//
// Status: memory (alloc / write / read / sync / free) maps straight onto the library's GPU context. Its matrix-vector kernels take
// weights in the library's own tiled encodings (q4g32a.t64, q8g64s.t64; see its docs/FORMAT.md), which a GGUF or safetensors file does
// not hold, so supports('gemv') is true only for a tensor record that says { encoding: 'q4g32a.t64' | 'q8g64s.t64' }. A site that uses it
// re-encodes each tensor on the way to the GPU with the library's converter (packQ4 / packQ8), one tensor at a time.

export const DEFAULT_URL = 'https://cdn.jsdelivr.net/gh/ArtistJoshHughes/p0pc0rn-kernels@4e5a07d1b1cd49d2b6c357b811e09399ddaac254/src/index.js';
const ENCODINGS = ['q4g32a.t64', 'q8g64s.t64'];

export default async function createBackend(options = {}) {
  const K = await import(options.url || DEFAULT_URL);
  const g = await K.createGpu(options.gpu || {});
  const stats = { gpuNow: 0, gpuPeak: 0, dispatches: 0 };
  const live = new Map(); let nextId = 1;
  const backend = {
    name: 'p0pc0rn-kernels', lib: K, gpu: g, device: g.device, stats,
    info: { adapter: g.describe ? g.describe() : {}, limits: g.limits || {}, kmax: K.GEMV_KMAX },
    supports(op, d) {
      return op === 'gemv' && !!(d && d.weight) && ENCODINGS.includes(d.weight.encoding) && (!d.x || d.x.dtype === 'f32')
        && d.weight.shape && d.weight.shape[1] <= K.GEMV_KMAX;
    },
    alloc(bytes, label = '') {
      const buffer = g.buffer(bytes, label), h = { id: nextId++, buffer, size: Math.max(16, Math.ceil(bytes / 16) * 16), label };
      live.set(h.id, h); stats.gpuNow += h.size; if (stats.gpuNow > stats.gpuPeak) stats.gpuPeak = stats.gpuNow;
      return h;
    },
    write(h, data, byteOffset = 0) { g.runner.flush(); g.upload(data, h.label, h.buffer, byteOffset); },
    ops: {
      // y = W x for weights in a library encoding (one split: cols <= GEMV_KMAX); x and y are f32 buffers
      gemv({ weight, x, y, rows, cols, encoding }) {
        const enc = encoding || weight.encoding;
        if (!ENCODINGS.includes(enc)) throw new Error('p0pc0rn-kernels gemv: weights must be in ' + ENCODINGS.join(' or '));
        K.gemv(g, { w: { enc, buf: weight.buffer, off: 0, rows, cols }, x: { buf: x.buffer }, N: rows, K: cols, pro: 'raw', epi: 'store', y: { buf: y.buffer }, ldy: rows });
        stats.dispatches++;
      },
    },
    async sync() { await g.done(); },
    async read(h, bytes = h.size) { return g.read(h.buffer, 0, bytes); },
    free(h) { if (h && live.has(h.id)) { live.delete(h.id); stats.gpuNow -= h.size; g.free(h.buffer); } },
    destroy() { for (const h of Array.from(live.values())) backend.free(h); try { g.device.destroy(); } catch (e) { /* */ } },
  };
  return backend;
}
