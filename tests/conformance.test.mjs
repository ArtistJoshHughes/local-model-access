// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// conformance.test.mjs - the LMA core (lma/) against the conformance test vectors (CONFORMANCE.md), in Node 20+:
//   node tests/conformance.test.mjs [DIR]      DIR = test-vectors (default), holding vectors.json and the vector files
// Checks, per vector: the header facts (format, name, architecture, tensor count, parts, tensor offsets / sizes / dtypes / shapes);
// each file's streamed SHA-256 against its identity; verifyModel accepts the vector against its identity and refuses a decoy; a CPU
// matrix-vector product read through the Loader, slice by slice across parts, matches the answers in the metadata; the Loader refuses a
// read above its slice limit. Then every input in 'refuse' is refused with its error code.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { readModel, readHeader, FormatError } from '../lma/formats.js';
import { Sha256 } from '../lma/sha256.js';
import { Loader } from '../lma/loader.js';
import { verifyModel, partsOf, verifyChunk, fileSha256 } from '../lma/verify.js';
import * as PF from '../lma/ext/pocketflux.js';

const dir = process.argv[2] || fileURLToPath(new URL('../test-vectors', import.meta.url));
const doc = JSON.parse(fs.readFileSync(path.join(dir, 'vectors.json'), 'utf8'));
let fails = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) fails++; };
// a named, lazy file (Node's openAsBlob has no name); fs.statSync gives the true size (openAsBlob's is wrong above 4 GiB on Windows)
const open = async (name, bytes) => {
  if (bytes) { const b = new Blob([bytes]); return { name, size: b.size, lastModified: 0, slice: (a, e) => b.slice(a, e), text: () => b.text() }; }
  const p = path.join(dir, name), b = await fs.openAsBlob(p);
  return { name: path.basename(name), size: fs.statSync(p).size, lastModified: 0, slice: (a, e) => b.slice(a, e), text: () => b.text() };
};
const meta = (m, k) => (m.format === 'gguf' ? m.metadata[k] : JSON.parse(m.metadata[k]));

for (const v of doc.vectors) {
  const files = await Promise.all(v.files.map((f) => open(f.file)).concat(v.index ? [open(v.index.file)] : []));
  const m = await readModel(files.slice().reverse());   // pick order must not matter
  const H = v.header;
  check(m.format === v.format && m.name === H.name && m.architecture === H.architecture && m.tensors.length === H.tensors && m.parts.length === H.parts,
    `${v.id}: ${m.format}, "${m.name}", ${m.architecture}, ${m.tensors.length} tensors in ${m.parts.length} part(s)` + (m.index ? `, index of ${m.index.tensors}` : ''));
  const seen = H.tensorsSeen.every((e) => { const t = m.tensors.find((x) => x.name === e.name); return t && t.part === e.part && t.offset === e.offset && t.size === e.size && t.dtype === e.dtype && JSON.stringify(t.shape) === JSON.stringify(e.shape); });
  check(seen, `${v.id}: tensor offsets, sizes, dtypes and shapes as listed`);
  let shaOk = true;
  for (const [i, f] of v.files.entries()) {
    const h = new Sha256(), file = m.files[i];
    for (let o = 0; o < file.size; o += 1 << 18) h.update(await file.slice(o, Math.min(file.size, o + (1 << 18))).arrayBuffer());
    const want = crypto.createHash('sha256').update(fs.readFileSync(path.join(dir, f.file))).digest('hex');
    if (file.name !== f.file || h.hex() !== f.sha256 || want !== f.sha256 || file.size !== f.size) shaOk = false;
  }
  check(shaOk, `${v.id}: every file's streamed sha256 and size match its identity`);
  const decoy = Object.assign({}, v.identity, { name: 'decoy' }); partsOf(decoy).length === 1 ? (decoy.sha256 = '0'.repeat(64)) : (decoy.files = decoy.files.map((p) => Object.assign({}, p, { sha256: '1'.repeat(64) })));
  const good = await verifyModel(m.files, { approved: [decoy, v.identity], model: m, noCache: true });
  check(good.ok && good.entry.name === v.identity.name && good.how === 'hashed', `${v.id}: verifyModel approves it against its identity, not the decoy`);
  const bad = await verifyModel(m.files, { approved: [decoy], model: m, noCache: true });
  check(!bad.ok && /different contents/.test(bad.problems.join(' ')), `${v.id}: verifyModel refuses a list with only a same-size decoy`);
  // the answers, through the loader, slice by slice across parts (64 KiB slices, the floor: every vector's weights need several)
  const L = new Loader(m.files, m, { maxSliceMB: 0.0625 }), g = v.gemv;
  const x = await L.withTensor('input.x', (ab) => new Int32Array(ab.slice(0)));
  let exact = 0;
  for (let l = 0; l < g.layers; l++) {
    const w = L.tensor(g.weight.replace('{N}', l)), rowsPer = Math.floor(L.max / g.cols); let s = 0, ws = 0;
    for (let r0 = 0; r0 < g.rows; r0 += rowsPer) {
      const r1 = Math.min(g.rows, r0 + rowsPer);
      await L.withSlice(w.offset + r0 * g.cols, w.offset + r1 * g.cols, (ab) => {
        const W = new Int8Array(ab);
        for (let r = r0; r < r1; r++) { let a = 0; const o = (r - r0) * g.cols; for (let c = 0; c < g.cols; c++) a += W[o + c] * x[c]; s += a; ws += a * ((r % 7) + 1); }
      }, w.part);
    }
    if (s === g.sum[l] && ws === g.wsum[l] && s === meta(m, 'lma-test-gemv.check.sum')[l]) exact++;
  }
  check(exact === g.layers && L.stats.cpuNow === 0, `${v.id}: CPU matrix-vector answers ${exact}/${g.layers} exact, read in slices across parts (peak ${L.stats.cpuPeak} B)`);
  const big = m.tensors.find((t) => t.size > L.max);
  let refused = false; try { await L.withTensor(big.name, () => {}); } catch (e) { refused = /refused/.test(e.message); }
  const writes = []; await L.toGpu(big.name, { alloc: (n) => ({ n }), write: (h, ab, off) => writes.push([off, ab.byteLength]) });
  check(refused && writes.length > 1 && writes.every(([o, n]) => n <= L.max) && L.stats.cpuPeak <= 2 * L.max,
    `${v.id}: a read above the slice limit is refused; toGpu sends ${big.name} in ${writes.length} slices (peak ${L.stats.cpuPeak} B <= 2 slices)`);
}

// refusals
const raw = fs.readFileSync(path.join(dir, 'lma-test-1m.gguf'));
const wrongVersion = Buffer.from(raw); wrongVersion.writeUInt32LE(9, 4);
const changed = Buffer.from(raw); changed[400000] ^= 0xff;
const sv = doc.vectors.find((v) => v.id === 'split'), sh = doc.vectors.find((v) => v.id === 'sharded'), small = doc.vectors.find((v) => v.id === 'small');
const cases = {
  'not-a-model': [await open('random.bin', crypto.randomBytes(4096))],
  truncated: [await open('lma-test-1m.gguf', raw.subarray(0, raw.length - 1000))],
  'header-cut': [await open('lma-test-1m.gguf', raw.subarray(0, 200))],
  'bad-version': [await open('lma-test-1m.gguf', wrongVersion)],
  'missing-part': [await open(sv.files[0].file)],
  'missing-shard': [await open(sh.files[0].file), await open(sh.index.file)],
};
for (const r of doc.refuse) {
  if (r.id.startsWith('pack-')) continue;   // checked with the pack below
  if (r.id === 'changed-byte') {
    const files = [await open('lma-test-1m.gguf', changed)], m = await readModel(files);
    const res = await verifyModel(files, { approved: [small.identity], model: m, noCache: true });
    check(!res.ok, `refuses ${r.id}: the header reads, the identity does not match`);
    continue;
  }
  let got = 'accepted'; try { await readModel(cases[r.id]); } catch (e) { got = e instanceof FormatError ? e.code : 'threw ' + e.message; }
  check(got === r.code, `refuses ${r.id}: ${got}`);
}
{ // a shard renamed so its name says nothing: the reader accepts it alone, and verifyModel names the file still to pick
  const f = await open(sh.files[1].file), lone = { name: 'renamed.safetensors', size: f.size, lastModified: 0, slice: f.slice, text: f.text };
  const res = await verifyModel([lone], { approved: [sh.identity], noCache: true });
  check(!res.ok && res.entry && res.missing.length === 1 && res.missing[0] === sh.identity.files[0].file, `a lone, renamed shard: not ready, "also pick ${res.missing[0]}"`);
}

// the pocketflux.pack/1 extension (SPEC.md 2.2)
for (const pk of doc.packs || []) {
  const files = await Promise.all(pk.files.map((f) => open(f.file))), extra = await open('lma-test-1m.gguf');
  const m = await readModel(files.slice().reverse().concat([extra]));   // any order; a file the manifest does not name is ignored
  check(m.format === 'pocketflux.pack' && m.tensors.length === Object.keys(pk.tensors).length && m.files.length === pk.files.length - 1 && m.ignored.length === 1,
    `pack: manifest + ${m.files.length} files, ${m.tensors.length} tensors, encodings ${[...new Set(m.tensors.map((t) => t.enc))].join(' ')}; an unrelated file ignored`);
  const full = await verifyModel(m.files.concat([m.manifestFile]), { approved: [pk.identity], model: m, noCache: true });
  check(full.ok && full.how === 'hashed' && full.files.length === pk.files.length, `pack: verifyModel approves it by its manifest's identity and checks every file against the manifest`);
  const lazy = await verifyModel(m.files, { approved: [pk.identity], model: m, noCache: true, lazy: true });
  check(lazy.ok && lazy.how === 'manifest' && lazy.files.length === 1, 'pack: lazy verification checks only the manifest up front');
  let chunks = 0, good = 0;
  for (const [i, f] of m.files.entries()) for (let c = 0; c < pk.chunks[m.manifest.shards[i].file]; c++) { chunks++; if (await verifyChunk(m, i, c)) good++; }
  check(chunks > m.files.length && good === chunks, `pack: every chunk's sha256 matches the manifest (${good}/${chunks} chunks)`);
  const L = new Loader(m.files, m, { maxSliceMB: 0.0625, verify: 'chunks' });
  let exact = 0;
  for (const [name, e] of Object.entries(pk.tensors)) {
    const t = L.tensor(name), got = await L.withTensor(name, (ab) => PF.dequant(t.enc, ab, t.shape));
    const sum = Array.from(got).reduce((a, b) => a + b, 0), first = Array.from(got.slice(0, 8));
    if (t.enc === e.enc && first.every((x, i) => x === e.first[i]) && Math.abs(sum - e.sum) <= 1e-6 * Math.max(1, Math.abs(e.sum))) exact++;
    else console.log('       ', name, t.enc, first.slice(0, 3), e.first.slice(0, 3), sum, e.sum);
  }
  check(exact === Object.keys(pk.tensors).length && L.stats.chunksVerified > 0, `pack: the reference decoders reproduce every tensor exactly (${exact}/${Object.keys(pk.tensors).length}), reading through chunk-verified slices (${L.stats.chunksVerified} chunks)`);
  const t64 = m.tensors.find((t) => t.enc === 'q4g32a.t64'), rows = m.tensors.find((t) => t.enc === 'q4g32a.rows');
  const r65 = await PF.readRow(L, t64.name, 65), whole = await L.withTensor(t64.name, (ab) => PF.dequant(t64.enc, ab, t64.shape)), K = t64.shape[1];
  const r3 = await PF.readRow(L, rows.name, 3), wr = await L.withTensor(rows.name, (ab) => PF.dequant(rows.enc, ab, rows.shape)), K2 = rows.shape[1];
  check(r65.every((x, i) => x === whole[65 * K + i]) && r3.every((x, i) => x === wr[3 * K2 + i]), 'pack: readRow reads one row (a tile for .t64) and matches the whole-tensor decode');
  // refusals
  const byName = (n) => files.find((f) => f.name === n);
  let got = 'accepted'; try { await readModel(files.filter((f) => f.name !== 'ple.rows.pfw')); } catch (e) { got = e.code; }
  check(got === 'missing-part', `refuses pack-missing-shard: ${got}`);
  const g = byName('lm.global.pfw'), shortG = { name: g.name, size: g.size - 16, lastModified: 0, slice: (a, e) => g.slice(a, Math.min(e, g.size - 16)), text: g.text };
  got = 'accepted'; try { await readModel(files.map((f) => (f === g ? shortG : f))); } catch (e) { got = e.code; }
  check(got === 'truncated', `refuses pack-short-shard: ${got}`);
  const gb = new Uint8Array(fs.readFileSync(path.join(dir, 'lma-test-pack/lm.global.pfw'))); gb[5000] ^= 0xff;
  const bad = await open('lm.global.pfw', gb), files2 = files.map((f) => (f === g ? bad : f)), m2 = await readModel(files2);
  const v2 = await verifyModel(m2.files.concat([m2.manifestFile]), { approved: [pk.identity], model: m2, noCache: true });
  const flags = []; for (let c = 0; c < pk.chunks['lm.global.pfw']; c++) flags.push(await verifyChunk(m2, m2.files.indexOf(bad), c));
  let threw = false; try { const L2 = new Loader(m2.files, m2, { verify: 'chunks' }); await L2.withTensor('layers.0.mlp.gate_up_proj.weight', () => {}); } catch (e) { threw = /does not match/.test(e.message); }
  check(!v2.ok && flags.filter((x) => !x).length === 1 && !flags[1] && threw, `refuses pack-changed-chunk: full verify refuses; only chunk 1 fails verifyChunk; the chunk-verifying loader throws`);
}
// trust what was checked (SPEC.md 7.3-7.4), with this site's storage stubbed in memory
{
  const mem = new Map(); globalThis.localStorage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, String(v)), removeItem: (k) => mem.delete(k) };
  const MiB = 1 << 20, a = new Uint8Array(5 * MiB); for (let i = 0; i < a.length; i++) a[i] = (i * 2654435761) >>> 24;
  const sha = (u) => crypto.createHash('sha256').update(u).digest('hex'), file = (name, u, t = 1) => Object.assign(new Blob([u]), { name, lastModified: t });
  const A = file('model.gguf', a), b = a.slice(); b[1.5 * MiB] ^= 1;                  // b differs only outside the fingerprint's three sampled MiB
  const c = a.slice(); c[100] ^= 1;                                                    // c differs inside them
  const r1 = await fileSha256(A, { ns: 't' }), r2 = await fileSha256(A, { ns: 't' });
  const rB = await fileSha256(file('model (1).gguf', b, 2), { ns: 't' }), rC = await fileSha256(file('model.gguf', c), { ns: 't' });
  check(r1.how === 'hashed' && r2.how === 'cache' && r1.sha256 === sha(a), 'cache: a file is hashed once, then found by name, size and date');
  check(rB.how === 'hashed' && rB.sha256 === sha(b), 'cache: a renamed file with the same size and sampled bytes is hashed in full (the fingerprint is never a lookup)');
  check(rC.how === 'hashed' && rC.sha256 === sha(c), 'cache: a file swapped in under the same name, size and date is hashed again');
  delete globalThis.localStorage;
  const pk = (doc.packs || [])[0];
  if (pk) {
    const files = await Promise.all(pk.files.map((f) => open(f.file))), extra = await open('lma-test-1m.gguf'), m = await readModel(files.concat([extra]));
    let threw = false; try { new Loader([extra].concat(m.files.slice(1)), m); } catch (e) { threw = /model\.files/.test(e.message); }
    check(threw, 'Loader: refuses files other than model.files (an unlisted file from the same pick cannot be read in place of a verified one)');
    const gb = new Uint8Array(fs.readFileSync(path.join(dir, 'lma-test-pack/lm.global.pfw'))); gb[5000] ^= 0xff;
    const tampered = await open('lm.global.pfw', gb), m3 = await readModel(files.map((f) => (f.name === 'lm.global.pfw' ? tampered : f)));
    const lz = await verifyModel(m3.files, { approved: [pk.identity], model: m3, noCache: true, lazy: true });
    threw = false; try { await new Loader(m3.files, m3, { maxSliceMB: 16 }).withTensor('layers.0.mlp.gate_up_proj.weight', () => {}); } catch (e) { threw = /does not match/.test(e.message); }
    check(lz.ok && lz.how === 'manifest' && threw, 'Loader: a lazily verified pack is chunk-checked even when the engine does not ask, so a tampered shard throws');
    const m4 = await readModel(files.map((f) => (f.name === 'lm.global.pfw' ? tampered : f)));   // read again: never verified
    threw = false; try { await new Loader(m4.files, m4).withTensor('layers.0.mlp.gate_up_proj.weight', () => {}); } catch (e) { threw = /does not match/.test(e.message); }
    const m5 = await readModel(files), full5 = await verifyModel(m5.files.concat([m5.manifestFile]), { approved: [pk.identity], model: m5, noCache: true }), L5 = new Loader(m5.files, m5);
    await L5.withTensor('layers.0.mlp.gate_up_proj.weight', () => {});
    check(threw && full5.ok && L5.verify === null, 'Loader: a pack is chunk-checked by default, unless verifyModel hashed every shard of that same model');
  }
  // a crafted header: the "__proto__" key, and more nested arrays than the parser keeps
  const N = (1 << 20) + 1, enc = new TextEncoder(), parts = [];
  const u32 = (v) => { const x = new Uint8Array(4); new DataView(x.buffer).setUint32(0, v, true); return x; };
  const u64 = (v) => { const x = new Uint8Array(8); new DataView(x.buffer).setBigUint64(0, BigInt(v), true); return x; };
  const str = (s) => [u64(enc.encode(s).length), enc.encode(s)];
  parts.push(enc.encode('GGUF'), u32(3), u64(0), u64(2), ...str('__proto__'), u32(8), ...str('x'), ...str('k'), u32(9), u32(9), u64(N));
  const inner = new Uint8Array(12 * N); for (let i = 0; i < N; i++) inner.set(u32(0), 12 * i);   // N empty u8 arrays
  parts.push(inner, new Uint8Array(64));
  const h = await readHeader(new Blob(parts));
  check(Object.getPrototypeOf(h.metadata) === Object.prototype && h.metadata.__proto__ === 'x' && h.metadata.k && h.metadata.k.skipped === true && h.metadata.k.length === N,
    `a crafted header: "__proto__" is an ordinary key, and ${N} nested arrays are counted, not kept`);
}
const big = (doc.large || [])[0];
if (big && fs.existsSync(path.join(dir, big.files[0].file))) {
  const f = await open(big.files[0].file), m = await readModel([f]);
  check(m.tensors.length === 127, `large vector ${big.id}: header (${m.tensors.length} tensors); hash it with tools/modelfile.py identity to compare ${big.files[0].sha256.slice(0, 16)}...`);
}
console.log(fails ? `FAILED ${fails}` : 'ALL OK: LMA-conformant (Local Model Access by artist Josh Hughes)');
process.exit(fails ? 1 : 0);
