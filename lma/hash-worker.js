// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// hash-worker.js - Local Model Access (LMA): a module worker: streams a picked File through SHA-256 a slice at a time (the file is never held whole), posting progress.
// in: { file, chunk }  out: { t: 'progress', done, total, MBs } ... { t: 'done', sha256, bytes, ms, readMs, hashMs, peakBytes } | { t: 'error', why }
import { Sha256 } from './sha256.js';

self.onmessage = async (e) => {
  const { file, chunk = 8 << 20 } = e.data || {};
  try {
    const h = new Sha256(), total = file.size, t0 = performance.now();
    let off = 0, readMs = 0, hashMs = 0, last = 0;
    const read = (o) => file.slice(o, Math.min(total, o + chunk)).arrayBuffer();
    let next = total ? read(0) : null;
    while (off < total) {
      const tr = performance.now(); const ab = await next; readMs += performance.now() - tr;
      const noff = off + ab.byteLength;
      if (!ab.byteLength) throw new Error('the file could not be read at ' + off);
      next = noff < total ? read(noff) : null;   // the next slice is read while this one is hashed: at most two slices in memory
      const th = performance.now(); h.update(new Uint8Array(ab)); hashMs += performance.now() - th;
      off = noff;
      const t = performance.now();
      if (t - last > 200) { last = t; self.postMessage({ t: 'progress', done: off, total, MBs: off / 1048576 / ((t - t0) / 1000) }); }
    }
    self.postMessage({ t: 'done', sha256: h.hex(), bytes: off, ms: performance.now() - t0, readMs, hashMs, peakBytes: Math.min(total, 2 * chunk) });
  } catch (err) {
    self.postMessage({ t: 'error', why: String((err && err.message) || err).slice(0, 300) });
  }
};
