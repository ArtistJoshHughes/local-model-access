// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// verify.js - Local Model Access (LMA): are the files the user picked one of the models this site approves?
//   1. instant: the format's magic bytes, the header and each file's size, against the site's approved list (no weights read);
//   2. once per file: the SHA-256 of the whole file, streamed in a worker with progress, compared with the approved sha256;
//   3. remembered: the result is cached in this site's own storage (a few bytes), so a returning visit is instant again.
// An approved entry is data: { repo, revision, format, file, sha256, size } for one file, or { ..., files: [{ file, sha256, size }] }
// for a model in several files. Any other field (name, capabilities, license, hints a newer spec adds) is carried along and ignored here.
import { readModel } from './formats.js';
import { Sha256 } from './sha256.js';

const MiB = 1 << 20;
const now = () => performance.now();
const HEX64 = /^[0-9a-f]{64}$/;

/** the weight files an approved entry names: [{ file, sha256, size }] */
export function partsOf(entry) {
  const ps = Array.isArray(entry && entry.files) ? entry.files : entry && entry.sha256 ? [{ file: entry.file, sha256: entry.sha256, size: entry.size }] : [];
  return ps.filter((p) => p && HEX64.test(p.sha256 || '') && Number.isSafeInteger(p.size));
}

/** the instant check: read the header(s) of the picked file(s) (throws FormatError for anything that is not a GGUF / safetensors model) */
export async function inspectFiles(files) {
  const t0 = now(), list = Array.isArray(files) ? files : [files], model = await readModel(list);
  return { model, ms: now() - t0, files: list.map((f) => ({ name: f.name, size: f.size, type: f.type, lastModified: f.lastModified })) };
}
export const inspectFile = (file) => inspectFiles([file]);

/** approved entries that could be this model: same format, every picked weight file has the size of one of the entry's files,
 *  and the entry's `expect` values (if any) match the header */
export function candidates(model, approved) {
  const files = model.files || [];
  if (model.format === 'pocketflux.pack') {   // a pack is approved by its manifest's identity (the manifest pins every shard)
    return (approved || []).filter((e) => e && e.manifest && HEX64.test(e.manifest.sha256 || '') && /^pocketflux\.pack(\/1)?$/.test(e.format || '')
      && e.manifest.size === model.manifestFile.size && (!e.size || e.size === files.reduce((n, f) => n + f.size, 0)));
  }
  return (approved || []).filter((e) => {
    const ps = partsOf(e);
    if (!ps.length || (e.format && e.format !== model.format)) return false;
    if (!files.every((f) => ps.some((p) => p.size === f.size))) return false;
    return !e.expect || Object.entries(e.expect).every(([k, v]) => (k === 'name' ? model.name : k === 'architecture' ? model.architecture : model.metadata[k]) === v);
  });
}

/** a quick fingerprint: SHA-256 of the size and the first, middle and last MiB (3 MiB read). Stored with a cached result and checked on
 *  every cache hit, so a different file under the same name, size and date is hashed again. Never the identity, never a cache key. */
export async function fingerprint(file) {
  const h = new Sha256(), n = file.size, size = new DataView(new ArrayBuffer(8)); size.setFloat64(0, n); h.update(size);
  for (const o of n <= 3 * MiB ? [0] : [0, Math.floor(n / 2 / MiB) * MiB, n - MiB]) h.update(await file.slice(o, Math.min(n, o + (n <= 3 * MiB ? n : MiB))).arrayBuffer());
  return h.hex();
}

/** one file's SHA-256, streamed (a worker when it can, else this thread). onProgress({ done, total, MBs }) */
export function hashFile(file, { onProgress, chunk = 8 * MiB, worker = true } = {}) {
  const viaWorker = () => new Promise((resolve, reject) => {
    let w;
    try { w = new Worker(new URL('./hash-worker.js', import.meta.url), { type: 'module' }); } catch (e) { reject(e); return; }
    w.onmessage = (e) => {
      const m = e.data;
      if (m.t === 'progress') { if (onProgress) onProgress(m); return; }
      w.terminate();
      if (m.t === 'done') resolve(Object.assign(m, { where: 'worker' })); else reject(new Error(m.why || 'hash failed'));
    };
    w.onerror = (e) => { w.terminate(); reject(new Error((e && e.message) || 'the hash worker failed to start')); };
    w.postMessage({ file, chunk });
  });
  const here = async () => {
    const h = new Sha256(), t0 = now(); let off = 0, readMs = 0, hashMs = 0, last = 0;
    while (off < file.size) {
      const tr = now(); const ab = await file.slice(off, Math.min(file.size, off + chunk)).arrayBuffer(); readMs += now() - tr;
      if (!ab.byteLength) throw new Error('the file could not be read at ' + off);
      const th = now(); h.update(new Uint8Array(ab)); hashMs += now() - th; off += ab.byteLength;
      if (onProgress && now() - last > 200) { last = now(); onProgress({ done: off, total: file.size, MBs: off / MiB / ((now() - t0) / 1000) }); }
      await new Promise((r) => setTimeout(r, 0));   // keep the page responsive
    }
    return { sha256: h.hex(), bytes: off, ms: now() - t0, readMs, hashMs, peakBytes: Math.min(file.size, chunk), where: 'main thread' };
  };
  return worker && typeof Worker !== 'undefined' ? viaWorker().catch((e) => here().then((r) => Object.assign(r, { workerError: String(e.message || e) }))) : here();
}

// ---- the cache: this site's own storage, a few bytes per file (never the file)
function store() { try { return globalThis.localStorage || null; } catch (e) { return null; } }
function get(k) { try { const s = store(); const v = s && s.getItem(k); return v ? JSON.parse(v) : null; } catch (e) { return null; } }
function put(k, v) { try { const s = store(); if (s) s.setItem(k, JSON.stringify(v)); } catch (e) { /* storage full or blocked: the next visit hashes again */ } }
export const cacheKey = (ns, f) => `${ns}:lma:${f.name}|${f.size}|${f.lastModified}`;
export function forget(ns, file) { try { const s = store(); if (s) s.removeItem(cacheKey(ns, file)); } catch (e) { /* */ } }

/** one file's sha256: from the cache, else hashed and remembered. A cache hit needs the same name, size and lastModified as the file
 *  that was hashed, and the same fingerprint (a guard against a swapped file, not the identity). -> { sha256, how, hash } */
export async function fileSha256(file, { ns = 'site', noCache = false, onProgress } = {}) {
  const fp = await fingerprint(file);
  if (!noCache) {
    const hit = get(cacheKey(ns, file));
    if (hit && HEX64.test(hit.sha256) && hit.fp === fp) return { sha256: hit.sha256, how: 'cache' };
  }
  const hash = await hashFile(file, { onProgress });
  put(cacheKey(ns, file), { sha256: hash.sha256, fp, t: Date.now() });
  return { sha256: hash.sha256, how: 'hashed', hash };
}

/** the whole check for the picked file(s). opts: approved (the site's list), ns (cache namespace, one per site), onProgress({ file, done,
 *  total, MBs }), model (from inspectFiles), noCache. -> { ok, entry, how ('cache' | 'hashed' | 'rejected'), files: [{ name,
 *  size, sha256, how }], missing (entry files not picked), hash (totals of the files hashed this time), problems, model, ms } */
export async function verifyModel(files, opts = {}) {
  const t0 = now(), list = Array.isArray(files) ? files : [files], problems = [];
  const model = opts.model || (await readModel(list));
  const weightFiles = model.files || list, cands = candidates(model, opts.approved);
  const done = (r) => Object.assign({ ok: false, entry: null, how: 'rejected', files: [], missing: [], hash: null, candidates: cands.length, problems, model, ms: now() - t0 }, r);
  if (model.format === 'pocketflux.pack') return verifyPack(model, cands, opts, done, problems);
  if (!cands.length) {
    problems.push(`no approved model is a ${model.format} model in ${weightFiles.length === 1 ? 'a file' : weightFiles.length + ' files'} of ` + weightFiles.map((f) => f.size + ' bytes').join(' + ')
      + (model.name ? ` (this one says it is "${model.name}")` : ''));
    return done({});
  }
  const per = [], hash = { ms: 0, bytes: 0, readMs: 0, hashMs: 0, peakBytes: 0, where: null };
  for (const f of weightFiles) {
    const r = await fileSha256(f, { ns: opts.ns, noCache: opts.noCache, onProgress: opts.onProgress && ((p) => opts.onProgress(Object.assign({ file: f.name }, p))) });
    per.push({ name: f.name, size: f.size, sha256: r.sha256, how: r.how });
    if (r.hash) { hash.ms += r.hash.ms; hash.bytes += r.hash.bytes; hash.readMs += r.hash.readMs; hash.hashMs += r.hash.hashMs; hash.peakBytes = Math.max(hash.peakBytes, r.hash.peakBytes); hash.where = r.hash.where; }
  }
  const shas = new Set(per.map((p) => p.sha256));
  const entry = cands.find((e) => per.every((p) => partsOf(e).some((q) => q.sha256 === p.sha256)));
  const how = per.some((p) => p.how === 'hashed') ? 'hashed' : 'cache';
  if (!entry) { problems.push('the files have the sizes of an approved model but different contents (' + per.map((p) => p.sha256.slice(0, 12) + '...').join(', ') + ')'); return done({ files: per, how: 'rejected', hash: hash.where ? hash : null }); }
  const missing = partsOf(entry).filter((q) => !shas.has(q.sha256)).map((q) => q.file);
  if (missing.length) problems.push(`this model has ${partsOf(entry).length} files; also pick ` + missing.join(', '));
  return done({ ok: !missing.length, entry, how, files: per, missing, hash: hash.where ? hash : null });
}

/** one picked file (kept for single-file sites): verifyModel([file]) */
export const verifyFile = (file, opts = {}) => verifyModel([file], opts);

// ---- pocketflux.pack/1 (SPEC.md section 2.2): the approved identity pins manifest.json; the manifest pins each shard's sha256 and the
// sha256 of every chunk (chunk_bytes, 16 MiB in published packs). opts.lazy: check only the manifest now, and every chunk as it is read
// (Loader { verify: 'chunks' } or verifyChunk) instead of hashing every shard up front.
async function verifyPack(model, cands, opts, done, problems) {
  const mf = model.manifestFile;
  if (!cands.length) { problems.push(`no approved pocketflux.pack has a ${mf.size}-byte manifest` + (model.name ? ` (this one says it is "${model.name}")` : '')); return done({}); }
  const m0 = await fileSha256(mf, { ns: opts.ns, noCache: opts.noCache });
  const entry = cands.find((e) => e.manifest.sha256 === m0.sha256);
  if (!entry) { problems.push('the manifest has the size of an approved pack but different contents (' + m0.sha256.slice(0, 12) + '...)'); return done({ files: [{ name: mf.name, size: mf.size, sha256: m0.sha256, how: m0.how }] }); }
  const per = [{ name: mf.name, size: mf.size, sha256: m0.sha256, how: m0.how }], hash = { ms: 0, bytes: 0, readMs: 0, hashMs: 0, peakBytes: 0, where: null }, bad = [];
  if (!opts.lazy) {
    for (const [i, f] of model.files.entries()) {
      const want = model.manifest.shards[i].sha256;
      const r = await fileSha256(f, { ns: opts.ns, noCache: opts.noCache, onProgress: opts.onProgress && ((p) => opts.onProgress(Object.assign({ file: f.name }, p))) });
      per.push({ name: f.name, size: f.size, sha256: r.sha256, how: r.how });
      if (r.hash) { hash.ms += r.hash.ms; hash.bytes += r.hash.bytes; hash.readMs += r.hash.readMs; hash.hashMs += r.hash.hashMs; hash.peakBytes = Math.max(hash.peakBytes, r.hash.peakBytes); hash.where = r.hash.where; }
      if (r.sha256 !== want) bad.push(f.name);
    }
  }
  if (bad.length) problems.push(`${bad.length} file(s) do not match the pack's manifest: ${bad.slice(0, 3).join(', ')}`);
  const how = opts.lazy ? 'manifest' : per.some((p) => p.how === 'hashed') ? 'hashed' : 'cache';
  // a lazily verified pack: no shard byte is checked yet, so every Loader for this model checks each chunk as it reads it (loader.js)
  if (opts.lazy) Object.defineProperty(model, 'needsChunkCheck', { value: true, enumerable: true });
  else if (!bad.length) Object.defineProperty(model, 'verifiedFully', { value: true, enumerable: true, configurable: true });   // every shard hashed: reads need no chunk checks
  return done({ ok: !bad.length, entry, how, lazy: !!opts.lazy, files: per, hash: hash.where ? hash : null });
}

/** check one chunk of a pack shard against the manifest: model from readModel (a pack), part = index into model.files, i = chunk index.
 *  Reads at most 8 MiB at a time. -> Promise<boolean> */
export async function verifyChunk(model, part, i) {
  const s = model.manifest && model.manifest.shards[part], f = model.files[part];
  if (!s || !f) throw new RangeError('no shard ' + part);
  const cb = s.chunk_bytes, start = i * cb, end = Math.min(s.bytes, start + cb);
  if (!(i >= 0 && start < s.bytes)) throw new RangeError(`no chunk ${i} in ${f.name}`);
  const h = new Sha256();
  for (let o = start; o < end; o += 8 * MiB) h.update(await f.slice(o, Math.min(end, o + 8 * MiB)).arrayBuffer());
  return h.hex() === s.chunk_sha256[i];
}
