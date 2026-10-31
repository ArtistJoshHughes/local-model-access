// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// reference.js - the LMA reference page: <lma-model> (lma/ui) for choosing and checking the file, then the core (../lma/) to show what
// was found and to read the largest tensor slice by slice, with the memory peak and this site's storage before and after.
import '../lma/ui/lma-model.js';
import { storageSnapshot, Loader } from '../lma/index.js';

const $ = (s) => document.querySelector(s), MB = (b) => (b / 1048576).toFixed(b < 1048576 * 10 ? 2 : 1) + ' MB';
const rows = [], out = $('#out'), el = $('lma-model');
const show = () => { out.replaceChildren(...rows.map(([k, v]) => { const tr = document.createElement('tr'); for (const x of [k, v]) { const td = document.createElement('td'); td.textContent = x; tr.append(td); } return tr; })); };

async function defaults() {
  const list = [];
  try { list.push(...(await (await fetch('../registry/models.json')).json()).models); } catch (e) { /* no registry next to this page */ }
  try { const v = await (await fetch('../test-vectors/vectors.json')).json(); list.push(...v.vectors.concat(v.packs || [], v.large || []).map((x) => x.identity)); } catch (e) { /* no vectors */ }
  return list;
}
const parse = (t) => { const j = JSON.parse(t); return Array.isArray(j) ? j : j.models || []; };
const approved = await defaults();
el.models = approved;
$('#list').value = JSON.stringify(approved, null, 1);
$('#list').addEventListener('change', () => { try { el.models = parse($('#list').value); $('#listMsg').textContent = `${el.models.length} entries in use.`; } catch (e) { $('#listMsg').textContent = 'Not valid JSON: ' + e.message; } });
$('#count').textContent = `${approved.length} approved entries: the registry and the test vectors.`;

let before = null, t0 = 0;
el.addEventListener('lma-change', async (e) => { if (e.detail.state === 'checking' && !before) { before = await storageSnapshot(); t0 = performance.now(); } });
el.addEventListener('lma-error', (e) => { rows.length = 0; rows.push(['Refused', `${e.detail.message} (${e.detail.code})`]); show(); before = null; });
el.addEventListener('lma-ready', async (e) => {
  const { files, picked, model, entry, verified: v } = e.detail; rows.length = 0;
  rows.push(['Picked', (picked || files).map((f) => `${f.name} (${MB(f.size)})`).join(', ')]);
  rows.push(['Instant check', `${model.format} · "${model.name || '?'}" · ${model.architecture || '?'} · ${model.tensors.length} tensors in ${model.files.length} file(s)`]);
  rows.push(['Approved', `${entry.name || entry.file} (${v.how === 'hashed' ? `hashed in ${(v.hash.ms / 1000).toFixed(1)} s` : 'remembered: ' + v.how})`]);
  if (t0) rows.push(['Ready after', Math.round(performance.now() - t0) + ' ms']);
  const L = new Loader(model.files, model, { maxSliceMB: 16 }), t = model.tensors.slice().sort((a, b) => b.size - a.size)[0];
  for (let o = 0; o < t.size; o += L.max) await L.withSlice(t.offset + o, t.offset + Math.min(t.size, o + L.max), () => {}, t.part);
  rows.push(['Largest tensor', `${t.name} (${t.dtype}, ${MB(t.size)}) read in ${Math.ceil(t.size / L.max)} slices`]);
  rows.push(['Memory peak', `${MB(Math.max(L.stats.cpuPeak, (v.hash && v.hash.peakBytes) || 0))} of file data in memory at once`]);
  const after = await storageSnapshot();
  if (before && before.usage != null && after.usage != null) rows.push(["This site's storage", `${MB(before.usage)} before, ${MB(after.usage)} after`]);
  show(); before = null; t0 = 0;
});
