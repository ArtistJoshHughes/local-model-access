// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// sitepage.js - what the two demo sites share: load the site's config with the kit, show it, let the user pick the model file (one tap),
// run the instant check and the once-per-file hash through the site's own skills, then hand over to the site's job.
import { loadSite } from '../site-kit/site.js';
import { makePicker, storageSnapshot, describeFile } from '../../lma/picker.js';
import { inspectFile, verifyFile, forget } from '../../lma/verify.js';
import * as R from './report.js';
import { $, h, mb, r1, status, bar, table } from './ui.js';

/** opts: { section ('siteA'), localSkills: { fn: handler }, job(ctx, rows, show): the site's own work, run once per file per visit } */
export async function sitePage(opts) {
  const main = $('main');
  let site;
  try { site = await loadSite('./site.config.json'); }
  catch (e) { main.append(h('div', { class: 'status bad' }, 'This site could not load its config: ' + e.message)); throw e; }
  document.title = site.name;
  R.put('env', await R.env());
  const head = h('div', {}, h('p', { class: 'crumb' }, h('a', { href: '../index.html' }, 'Local Model Access demo')), h('h1', {}, site.name), h('p', { class: 'sub' }, site.job));
  const cfg = site.config, approved = site.models;
  const about = h('details', { class: 'card' }, h('summary', {}, "This site's config (from site.config.json)"),
    h('p', {}, h('b', {}, 'Approved models: '), approved.map((m) => `${m.file} (${mb(m.size)})`).join(', ')),
    h('p', {}, h('b', {}, 'Skills: '), (cfg.skills || []).map((s) => `${s.name} (${s.handler.type === 'http' ? 'API ' + s.handler.api : 'local'})`).join(', ')),
    h('p', {}, h('b', {}, 'Kernel backend: '), (cfg.kernels && cfg.kernels.backend) || 'none'),
    h('p', {}, h('b', {}, 'Pre-prompt: ')), h('pre', {}, site.systemPrompt || '(none)'),
    h('p', {}, h('b', {}, 'Tool definitions a model runner would get: ')), h('pre', {}, JSON.stringify(site.tools(), null, 1)));
  const pickCard = h('section', { class: 'card' }, h('h2', {}, 'Use the model file'),
    h('p', { class: 'sub' }, 'Pick it from the Files app (Browse > On My iPhone > Models, or Downloads). The page reads it where it is: nothing is uploaded and nothing is copied into this site.'));
  const stat = h('div', { class: 'status' }, 'Waiting for the file.'), pb = bar(pickCard), results = h('div');
  const again = h('button', { class: 'btn secondary hidden' }, 'Forget this file here (the next pick hashes it again)');
  pickCard.append(stat);
  main.append(head, pickCard, h('section', { class: 'card' }, h('h2', {}, 'Results'), results), about, h('section', { class: 'card' }, h('h2', {}, 'Report'), (opts.reportEl = h('div'))));
  const report = R.mount(opts.reportEl);
  const startStore = await storageSnapshot();
  let busy = false, lastFile = null, picks = 0;
  const ran = new Set();   // files whose job already ran on this page: picking one again measures only the instant re-check

  // the two skills every site has: the instant check and the once-per-file verification
  site.bind(Object.assign({
    check_model: async (args, ctx) => {
      const r = await inspectFile(ctx.file); ctx.model = r.model;
      return { format: r.model.format, name: r.model.name, architecture: r.model.architecture, tensors: r.model.tensors.length, ms: r.ms, reads: r.model.headerReads };
    },
    verify_model: async (args, ctx) => verifyFile(ctx.file, { approved, ns: site.id, model: ctx.model, onProgress: ctx.onProgress }),
  }, opts.localSkills || {}));

  const picker = makePicker(pickCard, { label: 'Choose the model file', onPick: (file, info) => run(file, info) });
  pickCard.append(again);
  again.addEventListener('click', () => { if (lastFile) { forget(site.id, lastFile); status(stat, 'Forgotten: pick the file again to hash it again.'); } });

  async function run(file, info) {
    if (!file) { status(stat, info && info.cancelled ? 'No file picked.' : 'No file came back.', 'warn'); return; }
    if (busy) return; busy = true; picker.setDisabled(true); picks++;
    const tPick = performance.now(), rec = { file: describeFile(file), pick: info, storageBefore: picks === 1 ? startStore : await storageSnapshot() };
    const ctx = { file, site, rec, status: (t, k) => status(stat, t, k), bar: pb, onProgress: (p) => { pb.set(p.done / p.total); status(stat, `Checking the file once (SHA-256): ${mb(p.done)} of ${mb(p.total)} · ${r1(p.MBs)} MB/s\nNext time it is instant.`); } };
    const rows = [['File', `${file.name} · ${mb(file.size)}`], ['Pick', `${info.tapToFileMs} ms from the tap` + (info.closedToFileMs != null ? ` · ${info.closedToFileMs} ms after the picker closed` : '')]];
    const show = () => table(results, rows);
    try {
      status(stat, 'Reading the header…'); show();
      const c = await site.callTool('check_model', {}, ctx);
      rec.format = c.format; rec.header = { ms: Math.round(c.ms), reads: c.reads, tensors: c.tensors, name: c.name, architecture: c.architecture };
      rows.push(['Header (instant check)', `${c.format} · "${c.name || '?'}" · ${c.architecture || '?'} · ${c.tensors} tensors · ${Math.round(c.ms)} ms`]); show();
      const v = await site.callTool('verify_model', {}, ctx); pb.hide();
      rec.verify = { ok: v.ok, how: v.how, ms: Math.round(v.ms), problems: v.problems, sha256: v.files[0] && v.files[0].sha256, entry: v.entry && (v.entry.name || v.entry.file), hash: v.hash && { ms: Math.round(v.hash.ms), bytes: v.hash.bytes, where: v.hash.where, readMs: Math.round(v.hash.readMs), hashMs: Math.round(v.hash.hashMs), peakBytes: v.hash.peakBytes, workerError: v.hash.workerError } };
      rec.readyMs = Math.round(performance.now() - tPick);
      rows.push(['Approved?', v.ok ? `yes: ${v.entry.name || v.entry.file} · ${v.how === 'hashed' ? `hashed once in ${r1(v.hash.ms / 1000)} s (${v.hash.where})` : 'remembered from an earlier visit (' + v.how + ')'}` : 'NO: ' + v.problems.join('; ')]);
      rows.push(['Ready', `${rec.readyMs} ms after the pick`]); show();
      if (!v.ok) { status(stat, 'This file is not a model this site approves.', 'bad'); return; }
      ctx.model = v.model; ctx.entry = v.entry; again.classList.remove('hidden'); lastFile = file;
      const fkey = `${file.name}|${file.size}|${file.lastModified}`;
      if (!ran.has(fkey)) {
        ran.add(fkey); status(stat, 'Approved. Running this site\'s job…', 'ok');
        await opts.job(ctx, rows, show);
      } else rows.push(['Job', 'already ran for this file on this page (this pick measures the instant re-check)']);
      rec.storageAfter = await storageSnapshot();
      rows.push(["This site's storage", `${mb(rec.storageBefore.usage)} before · ${mb(rec.storageAfter.usage)} after (the model is not copied in)`]); show();
      status(stat, 'Done. Pick the same file again to see the instant re-check, then copy the report.', 'ok');
    } catch (e) {
      rec.error = String((e && e.message) || e).slice(0, 300); status(stat, 'Error: ' + rec.error, 'bad'); rows.push(['Error', rec.error]); show();
    } finally {
      rec.skills = site.calls.splice(0).map((c) => ({ name: c.name, ok: c.ok, ms: c.ms, error: c.error }));
      R.push(opts.section, 'picks', rec); report.refresh(); busy = false; picker.setDisabled(false); picker.setLabel('Choose the model file again');
    }
  }
  return { site };
}
