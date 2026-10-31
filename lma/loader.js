// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// loader.js - Local Model Access (LMA): the memory rules, enforced. A model file is read only in slices of at most maxSliceMB; a tensor
// goes to the GPU a slice at a time and each CPU copy is dropped as soon as it is written; the peak bytes held are counted so a page can
// report them. There is deliberately no way to read a whole file.
//   const model = await readModel(files);                     // formats.js: one file or several parts
//   const L = new Loader(model.files, model, { maxSliceMB: 16 });
//   const h = await L.toGpu('blk.0.attn_q.weight', backend);  // a GPU handle; the CPU copies are already dropped
//   await L.withTensor('output_norm.weight', (ab) => ...);     // a small tensor on the CPU, released after the callback
import { tensorMap } from './formats.js';
import { Sha256 } from './sha256.js';

const MiB = 1 << 20;
const PIECE = 8 * MiB;   // a chunk being checked is read in pieces of at most this
const now = () => performance.now();

export class Loader {
  /** files: model.files from readModel (the files the model was read and verified from, in part order), or one File with a readHeader
   *  model. Any other files are refused, so the bytes read are always the bytes that were checked.
   *  verify: 'chunks' (a pocketflux.pack): every chunk a read touches is hashed the first time and checked against the manifest's chunk
   *  sha256, and the bytes returned are the bytes hashed; a mismatch throws, so nothing unverified reaches the GPU. Every pack gets this
   *  unless verifyModel hashed all of its shards for this same model object (a lazily verified pack always does), whatever `verify` says. */
  constructor(files, model, { maxSliceMB = 16, verify = null } = {}) {
    this.files = Array.isArray(files) ? files : [files]; this.model = model; this.max = Math.max(64 * 1024, Math.floor(maxSliceMB * MiB)); this.map = tensorMap(model);
    if (model.files && (this.files.length !== model.files.length || this.files.some((f, i) => f !== model.files[i])))
      throw new Error('Loader: pass model.files, the files the model was read and verified from, in part order');
    const chunks = verify === 'chunks' || model.needsChunkCheck === true || (!!model.manifest && model.verifiedFully !== true);
    if (verify !== null && verify !== 'chunks') throw new Error(`Loader: unknown verify option ${JSON.stringify(verify)}`);
    if (chunks && !model.manifest) throw new Error("Loader: verify: 'chunks' needs a pocketflux.pack (a manifest with chunk hashes)");
    this.verify = chunks ? this.files.map(() => new Set()) : null;
    this.stats = { reads: 0, bytesRead: 0, readMs: 0, cpuNow: 0, cpuPeak: 0, uploads: 0, bytesUploaded: 0, refused: 0, chunksVerified: 0 };
  }

  /** bytes [start, end) of pack file `part`, every chunk they touch checked: an unchecked chunk is read whole (in pieces), hashed, and
   *  the requested bytes are taken from those same pieces; a checked chunk is read only where needed */
  async readChecked(start, end, part) {
    const f = this.files[part], sh = this.model.manifest.shards[part], cb = sh.chunk_bytes, done = this.verify[part], out = new Uint8Array(end - start);
    if (!(Number.isSafeInteger(cb) && cb > 0)) throw new Error(`the pack's manifest has no valid chunk_bytes for ${f.name}`);
    const piece = async (a, b) => { const u = new Uint8Array(await f.slice(a, b).arrayBuffer()); if (u.length !== b - a) throw new Error('short read at ' + a); return u; };
    for (let c = Math.floor(start / cb); c * cb < end; c++) {
      const c0 = c * cb, c1 = Math.min(sh.bytes, c0 + cb), lo = Math.max(start, c0), hi = Math.min(end, c1);
      if (done.has(c)) { out.set(await piece(lo, hi), lo - start); continue; }
      const h = new Sha256();
      for (let o = c0; o < c1; o += PIECE) {
        const p = await piece(o, Math.min(c1, o + PIECE)), a = Math.max(lo, o), b = Math.min(hi, o + p.length);
        this.stats.cpuPeak = Math.max(this.stats.cpuPeak, this.stats.cpuNow + p.length);
        h.update(p); if (a < b) out.set(p.subarray(a - o, b - o), a - start);
      }
      if (h.hex() !== sh.chunk_sha256[c]) throw new Error(`chunk ${c} of ${f.name} does not match the pack's manifest`);
      done.add(c); this.stats.chunksVerified++;
    }
    return out.buffer;
  }

  tensor(name) {
    if (typeof name !== 'string') return name;
    const t = this.map.get(name); if (!t) throw new Error('the model has no tensor ' + name); return t;
  }

  /** bytes [start, end) of file `part` as an ArrayBuffer, at most maxSliceMB. Call release(ab) when done (withSlice does it for you) */
  async read(start, end, part = 0) {
    const f = this.files[part], n = end - start;
    if (!f) throw new RangeError('no file for part ' + part);
    if (!(start >= 0 && n >= 0 && end <= f.size)) throw new RangeError(`slice ${start}-${end} is outside the file`);
    if (n > this.max) { this.stats.refused++; throw new Error(`refused: a ${Math.round(n / MiB)} MB read (the limit is ${this.max / MiB} MB); read big tensors in pieces`); }
    const s = this.stats; s.cpuNow += n; if (s.cpuNow > s.cpuPeak) s.cpuPeak = s.cpuNow;
    const t0 = now();
    try {
      const ab = this.verify && n ? await this.readChecked(start, end, part) : await f.slice(start, end).arrayBuffer();
      s.reads++; s.bytesRead += ab.byteLength; s.readMs += now() - t0;
      if (ab.byteLength !== n) throw new Error('short read at ' + start);
      return ab;
    } catch (e) { s.cpuNow -= n; throw e; }
  }

  release(ab) { if (ab) this.stats.cpuNow = Math.max(0, this.stats.cpuNow - ab.byteLength); }

  async withSlice(start, end, fn, part = 0) { const ab = await this.read(start, end, part); try { return await fn(ab); } finally { this.release(ab); } }

  /** a tensor of at most maxSliceMB on the CPU, for the length of the callback */
  async withTensor(name, fn) { const t = this.tensor(name); return this.withSlice(t.offset, t.offset + t.size, fn, t.part || 0); }

  /** upload one tensor to the GPU through a kernel backend (alloc / write), slice by slice: the next slice is read while this one is
   *  written, so at most two slices are on the CPU. handle: an existing GPU buffer to fill (e.g. a ring), else backend.alloc makes one */
  async toGpu(name, backend, handle = null, dstOffset = 0) {
    const t = this.tensor(name), s = this.stats, part = t.part || 0;
    const h = handle || backend.alloc(Math.ceil(t.size / 4) * 4, t.name);
    const piece = (o) => this.read(t.offset + o, t.offset + Math.min(t.size, o + this.max), part);
    let off = 0, next = t.size ? piece(0) : null;
    while (off < t.size) {
      const ab = await next, n = ab.byteLength;
      next = off + n < t.size ? piece(off + n) : null;
      try { backend.write(h, ab, dstOffset + off); } finally { this.release(ab); }
      off += n; s.uploads++; s.bytesUploaded += n;
    }
    return h;
  }
}
