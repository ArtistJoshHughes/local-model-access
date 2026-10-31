// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// stub.js - the kernel backend interface, as a template to copy for your own kernels. Point a site at your copy with
//   "kernels": { "backend": "custom", "module": "./my-kernels.js", "options": { ... } }
//
// A backend module's default export is an async factory: createBackend(options) -> backend. The kit's loader and the demo jobs only use
// what is listed here, so any kernel library fits behind it.
//
//   backend.name                      a short name for reports
//   backend.info                      what it runs on: { adapter, limits, features } (free form)
//   backend.stats                     { gpuNow, gpuPeak } bytes of GPU memory held now / at most (the memory report shows the peak)
//   backend.supports(op, desc)        can it run `op` on these tensors? desc = { weight, scale?, x? }: tensor records from formats.js
//                                     ({ name, dtype, shape, size }), so a backend can say no to a dtype it has no kernel for
//   backend.alloc(bytes, label)       -> handle: GPU memory for one tensor or buffer (zero-filled)
//   backend.write(handle, data, byteOffset = 0)
//                                     copy data (ArrayBuffer or typed array) into the handle. The caller drops its CPU copy right after,
//                                     so write must copy (queue.writeBuffer does) and must keep work already queued correct.
//   backend.ops.<op>(args)            queue a kernel. The demos use one op:
//     gemv({ weight, scale, x, acc, y, rows, cols, dtype })
//                                     y[r] = scale[r] * sum_c weight[r, c] * x[c]; acc (optional) gets the raw sums. Handles from alloc.
//   await backend.sync()              run everything queued and wait for it
//   await backend.read(handle, bytes) -> ArrayBuffer: copy GPU memory back (checks, results)
//   backend.free(handle)              release one handle
//   backend.destroy()                 release everything
//
// Add ops as your site needs them (attention, norms, sampling); keep the same alloc / write / sync / read so the loader works unchanged.

export default async function createBackend(options = {}) {
  const no = (what) => () => { throw new Error(`stub kernels: ${what} is not implemented (copy examples/site-kit/kernels/stub.js and fill it in)`); };
  return {
    name: 'stub',
    info: { note: 'template only' },
    stats: { gpuNow: 0, gpuPeak: 0 },
    supports: () => false,
    alloc: no('alloc'),
    write: no('write'),
    ops: { gemv: no('gemv') },
    sync: async () => {},
    read: no('read'),
    free: () => {},
    destroy: () => {},
  };
}
