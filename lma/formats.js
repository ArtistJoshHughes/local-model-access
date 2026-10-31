// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// formats.js - Local Model Access (LMA): read the header of a GGUF (v2, v3) or safetensors model file straight from a File or Blob, one slice at a time, and find
// every tensor by its byte offset. Tensor data is never read here. Works in pages, workers and Node 20+ (fs.openAsBlob). No dependencies.
//   const model = await readHeader(file);   // { format, name, architecture, metadata, tensors: [{ name, dtype, shape, offset, size }] }

export const MiB = 1 << 20;

/** ggml tensor types (id -> [dtype name, elements per block, bytes per block]), as in llama.cpp's ggml.h / gguf-py */
export const GGML_TYPES = {
  0: ['f32', 1, 4], 1: ['f16', 1, 2], 2: ['q4_0', 32, 18], 3: ['q4_1', 32, 20], 6: ['q5_0', 32, 22], 7: ['q5_1', 32, 24], 8: ['q8_0', 32, 34],
  9: ['q8_1', 32, 40], 10: ['q2_k', 256, 84], 11: ['q3_k', 256, 110], 12: ['q4_k', 256, 144], 13: ['q5_k', 256, 176], 14: ['q6_k', 256, 210],
  15: ['q8_k', 256, 292], 16: ['iq2_xxs', 256, 66], 17: ['iq2_xs', 256, 74], 18: ['iq3_xxs', 256, 98], 19: ['iq1_s', 256, 50],
  20: ['iq4_nl', 32, 18], 21: ['iq3_s', 256, 110], 22: ['iq2_s', 256, 82], 23: ['iq4_xs', 256, 136], 24: ['i8', 1, 1], 25: ['i16', 1, 2],
  26: ['i32', 1, 4], 27: ['i64', 1, 8], 28: ['f64', 1, 8], 29: ['iq1_m', 256, 56], 30: ['bf16', 1, 2], 34: ['tq1_0', 256, 54],
  35: ['tq2_0', 256, 66], 39: ['mxfp4', 32, 17], 40: ['nvfp4', 64, 36], 41: ['q1_0', 128, 18],
};

/** safetensors dtypes -> [dtype name, bytes per element] */
export const ST_DTYPES = {
  F64: ['f64', 8], F32: ['f32', 4], F16: ['f16', 2], BF16: ['bf16', 2], I64: ['i64', 8], U64: ['u64', 8], I32: ['i32', 4], U32: ['u32', 4],
  I16: ['i16', 2], U16: ['u16', 2], I8: ['i8', 1], U8: ['u8', 1], BOOL: ['bool', 1], F8_E4M3: ['f8_e4m3', 1], F8_E5M2: ['f8_e5m2', 1],
  F8_E8M0: ['f8_e8m0', 1],
};

export class FormatError extends Error {
  constructor(code, message) { super(message); this.name = 'FormatError'; this.code = code; }
}

const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);
const num = (v) => (v <= MAX_SAFE && v >= -MAX_SAFE ? Number(v) : v.toString());   // 64-bit values: a Number when exact, else a decimal string
const dec = new TextDecoder('utf-8');
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** reads a Blob forward through a window of `chunk` bytes; field readers are synchronous after fill(n) */
class Reader {
  constructor(blob, chunk) { this.blob = blob; this.chunk = chunk; this.buf = new Uint8Array(0); this.dv = new DataView(this.buf.buffer); this.base = 0; this.pos = 0; this.reads = 0; this.bytes = 0; }
  has(n) { return this.pos >= this.base && this.pos + n <= this.base + this.buf.length; }
  async fill(n) {
    if (this.has(n)) return;
    if (this.pos + n > this.blob.size) throw new FormatError('truncated', 'the file ends inside its header');
    const end = Math.min(this.blob.size, this.pos + Math.max(n, this.chunk));
    const u = new Uint8Array(await this.blob.slice(this.pos, end).arrayBuffer());
    this.reads++; this.bytes += u.length; this.buf = u; this.base = this.pos; this.dv = new DataView(u.buffer, u.byteOffset, u.byteLength);
  }
  at(n) { const o = this.pos - this.base; this.pos += n; return o; }
  u8() { return this.buf[this.at(1)]; }
  i8() { return this.dv.getInt8(this.at(1)); }
  u16() { return this.dv.getUint16(this.at(2), true); }
  i16() { return this.dv.getInt16(this.at(2), true); }
  u32() { return this.dv.getUint32(this.at(4), true); }
  i32() { return this.dv.getInt32(this.at(4), true); }
  f32() { return this.dv.getFloat32(this.at(4), true); }
  f64() { return this.dv.getFloat64(this.at(8), true); }
  u64() { return num(this.dv.getBigUint64(this.at(8), true)); }
  i64() { return num(this.dv.getBigInt64(this.at(8), true)); }
  async str(limit = 64 * MiB) {
    if (!this.has(8)) await this.fill(8);
    const n = this.u64();
    if (typeof n !== 'number' || n > limit) throw new FormatError('bad-header', 'a string in the header is implausibly long');
    if (!this.has(n)) await this.fill(n);
    const o = this.at(n); return dec.decode(this.buf.subarray(o, o + n));
  }
  async skipStr() {
    if (!this.has(8)) await this.fill(8);
    const n = this.u64(); if (typeof n !== 'number' || n > 64 * MiB) throw new FormatError('bad-header', 'a string in the header is implausibly long');
    this.pos += n;
  }
}

// GGUF metadata value types: id -> [name, bytes, read]
const GV = {
  0: ['u8', 1, (r) => r.u8()], 1: ['i8', 1, (r) => r.i8()], 2: ['u16', 2, (r) => r.u16()], 3: ['i16', 2, (r) => r.i16()],
  4: ['u32', 4, (r) => r.u32()], 5: ['i32', 4, (r) => r.i32()], 6: ['f32', 4, (r) => r.f32()], 7: ['bool', 1, (r) => r.u8() !== 0],
  10: ['u64', 8, (r) => r.u64()], 11: ['i64', 8, (r) => r.i64()], 12: ['f64', 8, (r) => r.f64()],
};

// What a header may keep in memory, whatever the file says: o.left array elements and o.leftBytes string bytes in all. Past that,
// values are only counted ({ skipped: true, of, length }), so a crafted header can't grow memory far beyond its own size.
async function ggufStr(r, o, keep) {
  if (!r.has(8)) await r.fill(8);
  const n = r.u64();
  if (typeof n !== 'number' || n > 64 * MiB) throw new FormatError('bad-header', 'a string in the header is implausibly long');
  if (!keep || n > o.leftBytes) { r.pos += n; return keep ? { skipped: true, of: 'string', length: n } : null; }
  o.leftBytes -= n;
  if (!r.has(n)) await r.fill(n);
  const at = r.at(n); return dec.decode(r.buf.subarray(at, at + n));
}

async function ggufValue(r, type, o, depth = 0, keep = true) {
  if (type === 8) return ggufStr(r, o, keep);
  const s = GV[type];
  if (s) { if (!r.has(s[1])) await r.fill(s[1]); return s[2](r); }
  if (type !== 9 || depth > 2) throw new FormatError('bad-header', 'unknown GGUF metadata type ' + type);
  if (!r.has(12)) await r.fill(12);
  const et = r.u32(), n = r.u64();
  if (typeof n !== 'number' || n > 1e9) throw new FormatError('bad-header', 'a GGUF array is implausibly long');
  const e = GV[et];
  // strings are kept when the array is short (tokenizer vocabularies run to hundreds of thousands), numbers up to maxArray
  const kept = keep && n <= o.left && n <= (et === 8 ? o.maxStrings : e ? o.maxArray : Infinity);
  if (kept) o.left -= n;
  const skipped = keep ? { skipped: true, of: et === 8 ? 'string' : e ? e[0] : 'array', length: n } : null;
  if (et === 8) {
    if (!kept) { for (let i = 0; i < n; i++) await ggufStr(r, o, false); return skipped; }
    const a = []; for (let i = 0; i < n; i++) a.push(await ggufStr(r, o, true)); return a;
  }
  if (!e) { const a = kept ? [] : null; for (let i = 0; i < n; i++) { const v = await ggufValue(r, et, o, depth + 1, kept); if (a) a.push(v); } return a || skipped; }
  if (!kept) { r.pos += n * e[1]; return skipped; }
  if (!r.has(n * e[1])) await r.fill(n * e[1]);
  const a = new Array(n); for (let i = 0; i < n; i++) a[i] = e[2](r); return a;
}

/** an own property, even for a key such as "__proto__" (assignment would change the object's prototype instead) */
const own = (obj, key, value) => Object.defineProperty(obj, key, { value, enumerable: true, writable: true, configurable: true });

async function readGguf(blob, o) {
  const r = new Reader(blob, o.chunk);
  await r.fill(24);
  if (r.u32() !== 0x46554747) throw new FormatError('not-gguf', 'not a GGUF file');
  const version = r.u32();
  if (version !== 2 && version !== 3) throw new FormatError('version', `GGUF version ${version} is not supported (2 and 3 are)`);
  const nT = r.u64(), nKV = r.u64();
  if (typeof nT !== 'number' || typeof nKV !== 'number' || nT > 1e6 || nKV > 1e6) throw new FormatError('bad-header', 'implausible GGUF counts');
  const metadata = {};
  for (let i = 0; i < nKV; i++) {
    const key = await r.str(1 << 16);
    if (!r.has(4)) await r.fill(4);
    own(metadata, key, await ggufValue(r, r.u32(), o));
  }
  const raw = [];
  for (let i = 0; i < nT; i++) {
    const name = await r.str(1 << 16);
    if (!r.has(4)) await r.fill(4);
    const nd = r.u32(); if (nd < 1 || nd > 8) throw new FormatError('bad-header', `tensor ${name} has ${nd} dimensions`);
    if (!r.has(8 * nd + 12)) await r.fill(8 * nd + 12);
    const ne = []; for (let d = 0; d < nd; d++) ne.push(r.u64());
    raw.push({ name, ne, type: r.u32(), rel: r.u64() });
  }
  const alignment = metadata['general.alignment'] ?? 32;
  if (typeof alignment !== 'number' || alignment < 1 || alignment % 8) throw new FormatError('bad-header', 'general.alignment must be a multiple of 8');
  const dataStart = Math.ceil(r.pos / alignment) * alignment;
  const tensors = raw.map((t) => {
    if (t.ne.some((d) => typeof d !== 'number')) throw new FormatError('bad-header', `tensor ${t.name} is too large`);
    if (typeof t.rel !== 'number' || t.rel % alignment) throw new FormatError('bad-header', `tensor ${t.name} is not aligned`);
    const info = GGML_TYPES[t.type], n = t.ne.reduce((a, b) => a * b, 1);
    let size = null;
    if (info) { if (t.ne[0] % info[1]) throw new FormatError('bad-header', `tensor ${t.name}: a row is not a whole number of ${info[0]} blocks`); size = (n / info[1]) * info[2]; }
    return { name: t.name, dtype: info ? info[0] : 'ggml_' + t.type, ggmlType: t.type, shape: t.ne.slice().reverse(), offset: dataStart + t.rel, size };
  });
  // a type this reader does not know: its size is the gap to the next tensor (or the end of the file)
  const byOff = tensors.slice().sort((a, b) => a.offset - b.offset);
  byOff.forEach((t, i) => { if (t.size == null) { t.size = (i + 1 < byOff.length ? byOff[i + 1].offset : blob.size) - t.offset; t.sizeFromGap = true; } });
  for (const t of tensors) if (t.offset + t.size > blob.size) throw new FormatError('truncated', `tensor ${t.name} runs past the end of the file (an incomplete download?)`);
  return { format: 'gguf', version, metadata, tensors, dataStart, alignment, headerBytes: r.pos, headerReads: r.reads, headerBytesRead: r.bytes,
    name: metadata['general.name'] ?? null, architecture: metadata['general.architecture'] ?? null };
}

async function readSafetensors(blob) {
  const n = num(new DataView(await blob.slice(0, 8).arrayBuffer()).getBigUint64(0, true));
  if (typeof n !== 'number' || n < 2 || n > 100 * MiB || 8 + n > blob.size) throw new FormatError('not-safetensors', 'not a safetensors file');
  let j;
  try { j = JSON.parse(dec.decode(await blob.slice(8, 8 + n).arrayBuffer())); } catch (e) { throw new FormatError('bad-header', 'the safetensors header is not valid JSON'); }
  if (!j || typeof j !== 'object' || Array.isArray(j)) throw new FormatError('bad-header', 'the safetensors header is not a JSON object');
  const dataStart = 8 + n, dataSize = blob.size - dataStart, metadata = j.__metadata__ || {}, tensors = [];
  for (const [name, t] of Object.entries(j)) {
    if (name === '__metadata__') continue;
    const info = ST_DTYPES[t && t.dtype], off = t && t.data_offsets;
    if (!Array.isArray(off) || off.length !== 2 || !Array.isArray(t.shape)) throw new FormatError('bad-header', `tensor ${name} has no data_offsets or shape`);
    const [b, e] = off;
    if (!Number.isSafeInteger(b) || !Number.isSafeInteger(e) || b < 0 || e < b) throw new FormatError('bad-header', `tensor ${name} has bad offsets`);
    if (e > dataSize) throw new FormatError('truncated', `tensor ${name} runs past the end of the file (an incomplete download?)`);
    if (info && t.shape.reduce((a, d) => a * d, 1) * info[1] !== e - b) throw new FormatError('bad-header', `tensor ${name}: its size does not match its shape`);
    tensors.push({ name, dtype: info ? info[0] : String(t.dtype).toLowerCase(), shape: t.shape.slice(), offset: dataStart + b, size: e - b });
  }
  return { format: 'safetensors', version: null, metadata, tensors, dataStart, alignment: 1, headerBytes: dataStart, headerReads: 2, headerBytesRead: dataStart,
    name: metadata['general.name'] || metadata.name || null, architecture: metadata['general.architecture'] || metadata.architecture || null };
}

/** 'gguf' | 'safetensors' | null, from the first bytes */
export async function sniff(blob) {
  if (!blob || typeof blob.slice !== 'function' || blob.size < 9) return null;
  const u = new Uint8Array(await blob.slice(0, Math.min(blob.size, 16)).arrayBuffer());
  if (u[0] === 0x47 && u[1] === 0x47 && u[2] === 0x55 && u[3] === 0x46) return 'gguf';
  const n = num(new DataView(u.buffer).getBigUint64(0, true));
  if (typeof n === 'number' && n >= 2 && n <= 100 * MiB && 8 + n <= blob.size && u[8] === 0x7b) return 'safetensors';
  return null;
}

/** read a model file's header. opts: chunk (bytes per header read, 1 MiB), maxArray (numeric GGUF arrays kept up to this length, 4096),
 *  maxStrings (string arrays kept up to this length, 64), maxKept / maxKeptBytes (array elements / string bytes kept in all, 2^20 / 32 MiB). Throws FormatError with a code: not-model, not-gguf, version, bad-header, truncated */
export async function readHeader(blob, opts = {}) {
  const o = Object.assign({ chunk: MiB, maxArray: 4096, maxStrings: 64, maxKept: 1 << 20, maxKeptBytes: 32 * MiB }, opts), t0 = now();
  o.left = o.maxKept; o.leftBytes = o.maxKeptBytes;
  const fmt = await sniff(blob);
  if (!fmt) throw new FormatError('not-model', 'this is not a GGUF or safetensors model file');
  const m = fmt === 'gguf' ? await readGguf(blob, o) : await readSafetensors(blob);
  m.fileSize = blob.size; m.readMs = now() - t0;
  return m;
}

/** name -> tensor */
export function tensorMap(model) { return new Map(model.tensors.map((t) => [t.name, t])); }

/** a safetensors index (model.safetensors.index.json): a small JSON file naming the shard that holds each tensor */
async function readIndex(blob) {
  let j = null;
  try { j = JSON.parse(await blob.text()); } catch (e) { /* not JSON */ }
  if (!j || typeof j !== 'object' || !j.weight_map || typeof j.weight_map !== 'object') throw new FormatError('not-model', `${blob.name || 'a file'} is not a GGUF or safetensors model file`);
  return j;
}
async function isJson(blob) {   // a small file starting with '{' that is not a safetensors header: an index, a pack manifest, or a pack's JSON file
  if (blob.size > 16 * MiB || blob.size < 2) return false;
  const u = new Uint8Array(await blob.slice(0, Math.min(blob.size, 64)).arrayBuffer());
  const ws = (c) => c === 0x20 || c === 0x0a || c === 0x0d || c === 0x09 || c === 0xef || c === 0xbb || c === 0xbf;
  let i = 0; while (i < u.length && ws(u[i])) i++;
  if (u[i] !== 0x7b) return false;
  let k = i + 1; while (k < u.length && ws(u[k])) k++;
  return (u[k] === 0x22 || u[k] === 0x7d) && (await sniff(blob)) == null;   // '{' then '"' or '}': a JSON object, not binary data
}
const partNo = (name) => { const m = /-(\d{5})-of-(\d{5})\.[^.]+$/.exec(name || ''); return m ? { no: +m[1], of: +m[2] } : null; };
const base = (p) => String(p || '').split('/').pop();

/** pocketflux.pack/1 (an extension format, SPEC.md section 2.2): a manifest.json naming every shard file (with its sha256 and the sha256 of
 *  every chunk) and every tensor (shard, offset, bytes, shape, encoding). Picked files are matched to shards by name, or by a size no other
 *  shard has; files the manifest does not name are ignored. Tensor dtype is the encoding (q4g32a.t64, q8g64s.rows, f32, ...). */
function readPack(manifestFile, m, files, t0) {
  if (m.format_version !== 1) throw new FormatError('version', `pocketflux.pack version ${m.format_version} is not supported (1 is)`);
  if (!Array.isArray(m.shards) || !m.shards.length || !m.tensors || typeof m.tensors !== 'object') throw new FormatError('bad-header', 'the pack manifest has no shards or tensors');
  const shards = m.shards, byName = new Map(), left = files.slice(), picked = new Array(shards.length).fill(null);
  shards.forEach((s, i) => byName.set(base(s.file), i));
  for (const f of files) { const i = byName.get(base(f.name)); if (i != null && !picked[i]) { picked[i] = f; left.splice(left.indexOf(f), 1); } }
  for (const f of left.slice()) {   // a renamed file: accept it for the one missing shard of its exact size
    const cands = shards.map((s, i) => i).filter((i) => !picked[i] && shards[i].bytes === f.size);
    if (cands.length === 1) { picked[cands[0]] = f; left.splice(left.indexOf(f), 1); }
  }
  const missing = shards.filter((s, i) => !picked[i]).map((s) => base(s.file));
  if (missing.length) throw new FormatError('missing-part', `this pack has ${shards.length} files besides manifest.json; ${missing.length} not picked, e.g. ${missing.slice(0, 3).join(', ')}`);
  shards.forEach((s, i) => { if (picked[i].size !== s.bytes) throw new FormatError('truncated', `${base(s.file)} is ${picked[i].size} bytes; the manifest says ${s.bytes} (an incomplete download?)`); });
  const idx = new Map(shards.map((s, i) => [s.name, i])), tensors = [];
  for (const [name, t] of Object.entries(m.tensors)) {
    const part = idx.get(t.shard);
    if (part == null) throw new FormatError('bad-header', `tensor ${name} is in shard ${t.shard}, which the manifest does not list`);
    if (!Number.isSafeInteger(t.offset) || !Number.isSafeInteger(t.bytes) || t.offset < 0 || t.offset + t.bytes > shards[part].bytes) throw new FormatError('bad-header', `tensor ${name} runs past the end of its shard`);
    tensors.push({ name, dtype: t.enc, enc: t.enc, shape: (t.shape || []).slice(), offset: t.offset, size: t.bytes, part, src: t.src, quantiser: t.quantiser });
  }
  const md = m.model || {};
  return { format: 'pocketflux.pack', formatVersion: 1, name: md.id || m.pack_id || null, architecture: md.arch || null,
    metadata: { pack_id: m.pack_id, pack_version: m.pack_version, model: md, encodings: m.encodings, quant_recipe: m.quant_recipe, sources: m.sources, license: m.license, notes: m.notes },
    parts: shards.map((s, i) => ({ name: picked[i].name, size: picked[i].size, shard: s.name, kind: s.kind, chunkBytes: s.chunk_bytes })),
    files: picked, manifestFile, manifest: m, tensors, index: null, ignored: left.map((f) => f.name), headerReads: 1, readMs: now() - t0 };
}

/** read a model made of one or more files, as picked: one GGUF or safetensors file, the parts of a split GGUF (name-00001-of-0000N.gguf,
 *  with split.no / split.count in each header), or safetensors shards (with or without their .index.json). Order does not matter.
 *  -> { format, name, architecture, metadata, parts: [{ name, size }], files (the model files, in part order), tensors (each with .part),
 *       index (when an .index.json was picked), readMs }. Throws FormatError: not-model, mixed, missing-part, duplicate-part, bad-header ... */
export async function readModel(files, opts = {}) {
  const t0 = now(), list = Array.from(files || []), heads = [], jsons = []; let index = null;
  for (const f of list) if (await isJson(f)) jsons.push(f);
  for (const f of jsons) {   // a pocketflux.pack manifest among the picked files: the whole pick is that pack
    let j = null; try { j = JSON.parse(await f.text()); } catch (e) { /* not JSON */ }
    if (j && j.format === 'pocketflux.pack') return readPack(f, j, list.filter((x) => x !== f), t0);
  }
  for (const f of list) {
    if (jsons.includes(f)) { index = await readIndex(f); continue; }
    heads.push({ file: f, m: await readHeader(f, opts) });
  }
  if (!heads.length) throw new FormatError('not-model', 'none of the picked files is a GGUF or safetensors model file');
  const format = heads[0].m.format;
  if (heads.some((h) => h.m.format !== format)) throw new FormatError('mixed', 'the picked files mix GGUF and safetensors');
  let metadata;
  if (format === 'gguf') {
    const count = heads[0].m.metadata['split.count'] || 1;
    if (count > 1 || heads.length > 1) {
      for (const h of heads) {
        if ((h.m.metadata['split.count'] || 1) !== count || typeof h.m.metadata['split.no'] !== 'number') throw new FormatError('bad-header', `${h.file.name} is not a part of the same split GGUF model`);
      }
      heads.sort((a, b) => a.m.metadata['split.no'] - b.m.metadata['split.no']);
      if (heads.some((h, i) => i && h.m.metadata['split.no'] === heads[i - 1].m.metadata['split.no'])) throw new FormatError('duplicate-part', 'the same part of a split GGUF was picked twice');
      if (heads.length !== count) throw new FormatError('missing-part', `this model is split into ${count} files; ${heads.length} ${heads.length === 1 ? 'was' : 'were'} picked`);
      const want = heads[0].m.metadata['split.tensors.count'], got = heads.reduce((n, h) => n + h.m.tensors.length, 0);
      if (typeof want === 'number' && want !== got) throw new FormatError('bad-header', `the parts hold ${got} tensors; the first part says ${want}`);
    }
    metadata = heads[0].m.metadata;
  } else {
    const nos = heads.map((h) => partNo(h.file.name));
    if (nos.every(Boolean)) {   // HF shard names (model-00001-of-00002.safetensors): order by number, and check none is missing when no index says otherwise
      heads.sort((a, b) => partNo(a.file.name).no - partNo(b.file.name).no);
      if (!index && nos[0].of !== heads.length) throw new FormatError('missing-part', `this model has ${nos[0].of} shards; ${heads.length} ${heads.length === 1 ? 'was' : 'were'} picked`);
    }
    metadata = {}; for (const h of heads) for (const [k, v] of Object.entries(h.m.metadata)) own(metadata, k, v);
  }
  const tensors = [], seen = new Set();
  heads.forEach((h, part) => {
    for (const t of h.m.tensors) {
      if (seen.has(t.name)) throw new FormatError('bad-header', `tensor ${t.name} is in two of the picked files`);
      seen.add(t.name); tensors.push(Object.assign({}, t, { part }));
    }
  });
  if (index) {
    const missing = Object.keys(index.weight_map).filter((n) => !seen.has(n));
    if (missing.length) throw new FormatError('missing-part', `${missing.length} tensors named in the index are not in the picked files (a shard is missing), e.g. ${missing[0]}`);
  }
  const m0 = heads[0].m;
  return { format, name: format === 'gguf' ? m0.name : metadata['general.name'] || metadata.name || null,
    architecture: format === 'gguf' ? m0.architecture : metadata['general.architecture'] || metadata.architecture || null,
    metadata, parts: heads.map((h) => ({ name: h.file.name, size: h.file.size, headerBytes: h.m.headerBytes, tensors: h.m.tensors.length })),
    files: heads.map((h) => h.file), tensors, index: index ? { tensors: Object.keys(index.weight_map).length, totalSize: index.metadata && index.metadata.total_size } : null,
    headerReads: heads.reduce((n, h) => n + h.m.headerReads, 0), readMs: now() - t0 };
}
