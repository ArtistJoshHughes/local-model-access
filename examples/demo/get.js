// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// get.js - the model's download page (a model publisher's page, not a site that uses the model). Route A: stream the file into this page's
// private storage (OPFS) with progress and resume, then hand it to the share sheet (Save to Files) and delete the page's copy. Route B: a
// plain download link for Safari's own downloader. Test hooks: ?writer=writable|sync, ?stopAtMB=N (stops part-way, to test resuming).
import { readHeader } from '../../lma/formats.js';
import { storageSnapshot } from '../../lma/picker.js';
import * as R from './report.js';
import { $, h, mb, r1, status, bar } from './ui.js';

const MiB = 1 << 20, q = new URLSearchParams(location.search), STOP_AT = (+q.get('stopAtMB') || 0) * MiB;
const ls = { get(k) { try { return JSON.parse(localStorage.getItem(k) || 'null'); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* */ } }, del(k) { try { localStorage.removeItem(k); } catch (e) { /* */ } } };
const env = await R.env();
let root = null; try { root = await navigator.storage.getDirectory(); } catch (e) { /* no OPFS */ }
// createSyncAccessHandle is only exposed inside workers: ask the download worker (it writes in place, so a dead download resumes)
env.syncHandle = await new Promise((res) => {
  try { const w = new Worker('dl-worker.js'), t = setTimeout(() => { w.terminate(); res(false); }, 3000);
    w.onmessage = (e) => { clearTimeout(t); w.terminate(); res(!!e.data.syncHandle); }; w.onerror = () => { clearTimeout(t); res(false); }; w.postMessage({ cmd: 'probe' }); }
  catch (e) { res(false); }
});
R.put('env', env);
const writer = q.get('writer') || (root && env.syncHandle ? 'sync' : root && env.createWritable ? 'writable' : null);
const report = R.mount($('#report'));
const showStore = async () => { const s = await storageSnapshot(); $('#store').textContent = s.usage != null ? `This page's own storage: ${mb(s.usage)} used.` : ''; };
showStore();
if (env.standalone) { const w = $('#warn'); w.classList.remove('hidden'); w.textContent = 'This is a Home Screen app: use A (Download into this page), then Save to Files. Skip B here, it needs a Safari tab.'; }

const sec = () => R.all().get || {};
const putA = (key, s) => { const g = sec(); g.A = Object.assign(g.A || {}, { [key]: s }); R.put('get', g); report.refresh(); };

// a share sheet that never came back = the page was killed or reloaded while it was open
const pend = ls.get('lma-demo.pendingShare');
if (pend) {
  ls.del('lma-demo.pendingShare');
  const s = (sec().A || {})[pend.key];
  if (s) { (s.shares = s.shares || []).push({ how: pend.how, result: 'PAGE DIED or reloaded while the share sheet was open', t: pend.t }); putA(pend.key, s); }
}

let list = [];
try { list = (await (await fetch('models/models.json', { cache: 'no-store' })).json()).models; }
catch (e) { $('#models').append(h('div', { class: 'status bad' }, 'No models/models.json next to this page: run tools/make_test_models.py (see the README).')); }
for (const m of list) card(m);

function card(m) {
  const key = m.key, name = m.saveAs, url = new URL('models/' + m.file, location.href).href, skey = 'lma-demo.dl.' + key;
  const st = () => Object.assign({ resumes: 0, interrupts: 0, restarts: 0, seconds: 0, size: m.size, writer }, ls.get(skey) || {});
  const box = h('section', { class: 'card' }, h('h2', {}, m.label), h('p', { class: 'sub' }, `${name} · ${mb(m.size)} · sha256 ${m.sha256.slice(0, 16)}...`));
  const stat = h('div', { class: 'status' }, '…'), pb = bar(box);
  const dl = h('button', { class: 'btn secondary' }, 'A. Download into this page'), pause = h('button', { class: 'btn secondary hidden' }, 'Pause');
  const save = h('button', { class: 'btn hidden' }, 'Save to Files'), saveMem = h('button', { class: 'btn secondary hidden' }, 'Save to Files (memory copy)');
  const del = h('button', { class: 'btn secondary hidden' }, "Delete this page's copy");
  const link = h('a', { class: 'btn', href: 'models/' + m.file, download: m.file }, 'B. Download with the browser');
  link.addEventListener('click', () => { R.push('get', 'B', { key, t: Date.now(), standalone: env.standalone }); report.refresh(); });
  box.append(link, stat, dl, pause, save, saveMem, del);
  $('#models').append(box);
  if (!root || !writer) { dl.disabled = true; status(stat, 'Route A is not possible in this browser (no private file storage). Use B.', 'warn'); }

  let file = null, memFile = null, worker = null, running = false, wake = null, autoTries = 0, lastErr = false;
  const opfsFile = async () => { try { return await (await root.getFileHandle(name)).getFile(); } catch (e) { return null; } };
  const lock = async () => { try { if (navigator.wakeLock && !wake) { wake = await navigator.wakeLock.request('screen'); wake.addEventListener('release', () => { wake = null; }); } } catch (e) { /* */ } };
  const unlock = () => { try { if (wake) wake.release(); } catch (e) { /* */ } wake = null; };

  async function refresh() {
    if (!root || !writer) return;
    const s = st(), f = await opfsFile(), have = f ? f.size : 0;
    [save, saveMem, del].forEach((b) => b.classList.add('hidden'));
    if (s.done && have === m.size) {
      file = f; dl.classList.add('hidden'); save.classList.remove('hidden'); del.classList.remove('hidden');
      if (m.size <= 32 * MiB) { memFile = new File([await f.arrayBuffer()], name, { type: 'application/octet-stream' }); saveMem.classList.remove('hidden'); }
      status(stat, `Downloaded into this page (${mb(have)}). Tap Save to Files, then choose On My iPhone > Models. If saving fails, use B.`, 'ok'); pb.set(1);
    } else if (have > 0) {
      dl.classList.remove('hidden'); dl.textContent = 'A. Resume the download'; pb.set(have / m.size);
      status(stat, `Paused at ${mb(have)} of ${mb(m.size)}. Tap Resume.`, 'warn');
    } else { dl.classList.remove('hidden'); dl.textContent = 'A. Download into this page'; status(stat, 'Not downloaded into this page.'); pb.hide(); }
    showStore();
  }

  const shown = { t: 0, got: 0, speed: 0 };
  function progress(got, total, t0, from) {
    const t = performance.now(), dt = (t - shown.t) / 1000;
    if (dt > 1) { shown.speed = shown.t ? (got - shown.got) / MiB / dt : 0; shown.t = t; shown.got = got; }
    const avg = (got - from) / MiB / ((t - t0) / 1000), left = total ? (total - got) / MiB / Math.max(0.1, shown.speed || avg) : 0;
    pb.set(total ? got / total : 0);
    status(stat, `Downloading: ${mb(got)} of ${mb(total)} (${total ? Math.round((100 * got) / total) : '?'} %) · ${r1(shown.speed || avg)} MB/s` + (left ? ` · about ${Math.ceil(left)} s left` : '') + '\nKeep this screen open; if it stops, it resumes from here.');
  }

  function viaWorker(s, from, t0) {
    return new Promise((resolve, reject) => {
      worker = new Worker('dl-worker.js');
      worker.onmessage = (e) => {
        const d = e.data;
        if (d.t === 'progress') progress(d.got, d.total, t0, from);
        else if (d.t === 'begin') { s.status = d.status; s.etag = d.etag || s.etag; s.ranges = d.ranges; if (d.restarted) s.restarts++; ls.set(skey, s); }
        else { worker.terminate(); worker = null; if (d.t === 'error') reject(Object.assign(new Error(d.why), { got: d.got })); else resolve(d); }
      };
      worker.onerror = (e) => { if (worker) worker.terminate(); worker = null; reject(new Error((e && e.message) || 'the download worker failed')); };
      worker.postMessage({ cmd: 'start', url, name, etag: s.etag || '', stopAt: STOP_AT });
    });
  }

  let stopWritable = false;
  async function viaWritable(s, from, t0) {   // createWritable commits on close(), so it is closed every 128 MB to keep what came
    const fh = await root.getFileHandle(name, { create: true }); let have = (await fh.getFile()).size;
    const headers = have ? Object.assign({ Range: `bytes=${have}-` }, s.etag ? { 'If-Range': s.etag } : {}) : {};
    const r = await fetch(url, { headers, cache: 'no-store' });
    if (have && r.status === 200) { have = 0; s.restarts++; } else if (!r.ok) throw new Error('HTTP ' + r.status);
    const total = r.status === 206 ? +/\/(\d+)/.exec(r.headers.get('content-range'))[1] : +(r.headers.get('content-length') || 0);
    s.etag = r.headers.get('etag') || s.etag; s.status = r.status; ls.set(skey, s);
    let w = await fh.createWritable({ keepExistingData: have > 0 }); if (have) await w.seek(have);
    const rd = r.body.getReader(); let got = have, since = 0, last = 0; stopWritable = false;
    try {
      for (;;) {
        const { done, value } = await rd.read(); if (done) break;
        await w.write(value); got += value.byteLength; since += value.byteLength;
        if (since >= 128 * MiB) { await w.close(); w = await fh.createWritable({ keepExistingData: true }); await w.seek(got); since = 0; }
        if (performance.now() - last > 250) { last = performance.now(); progress(got, total, t0, from); }
        if (stopWritable || (STOP_AT && got >= STOP_AT)) { await w.close(); w = null; try { rd.cancel(); } catch (e) { /* */ } return { t: 'stopped', got }; }
      }
      await w.close(); w = null; return { t: 'done', got, total };
    } catch (e) { if (w) { try { await w.close(); } catch (e2) { /* */ } } throw e; }
  }

  async function start(auto) {
    if (running) return; running = true; lastErr = false;
    const s = st(), f0 = await opfsFile(), from = f0 ? f0.size : 0;
    if (!from && !auto) Object.assign(s, { done: false, deleted: false, header: null, resumes: 0, interrupts: 0, restarts: 0, seconds: 0, autoResumes: 0, lastError: null });   // a fresh download
    if (from > 0) s.resumes++; if (auto) s.autoResumes = (s.autoResumes || 0) + 1;
    s.writer = writer; ls.set(skey, s); putA(key, s);
    dl.classList.add('hidden'); pause.classList.remove('hidden'); await lock();
    const t0 = performance.now();
    try {
      const d = writer === 'sync' ? await viaWorker(s, from, t0) : await viaWritable(s, from, t0);
      s.seconds += (performance.now() - t0) / 1000; s.have = d.got;
      if (d.t === 'stopped') { s.interrupts++; s.lastStop = 'paused'; }
      else if (d.got !== m.size) { s.interrupts++; s.lastError = `ended at ${d.got} of ${m.size} bytes`; lastErr = true; }
      else {
        const f = await opfsFile(), hd = await readHeader(f);   // the instant check on the downloaded copy
        s.done = true; s.header = `${hd.format}, ${hd.tensors.length} tensors, "${hd.name}"`; s.finishedAt = Date.now();
      }
    } catch (e) {
      s.seconds += (performance.now() - t0) / 1000; s.interrupts++; s.lastError = String(e.message || e).slice(0, 160); s.have = e.got; lastErr = true;
    }
    ls.set(skey, s); putA(key, s); running = false; pause.classList.add('hidden'); unlock();
    await refresh();
    if (lastErr && document.visibilityState === 'visible' && autoTries < 3) { autoTries++; setTimeout(() => start(true), 1500); }
  }

  dl.addEventListener('click', () => { autoTries = 0; start(false); });
  pause.addEventListener('click', () => { if (worker) worker.postMessage({ cmd: 'stop' }); stopWritable = true; });
  document.addEventListener('visibilitychange', () => {   // back from the background: a download that died there resumes by itself
    if (document.visibilityState === 'visible') { if (running) lock(); else if (lastErr && autoTries < 5) { autoTries++; start(true); } }
  });

  const share = (f, how) => () => {
    if (!f) return;
    const s = st(), data = { files: [f] }, rec = (result) => { (s.shares = s.shares || []).push({ how, result, t: Date.now() }); ls.set(skey, s); putA(key, s); };
    if (navigator.canShare && !navigator.canShare(data)) { rec('canShare said no'); status(stat, 'This browser cannot share this file. Use route B.', 'bad'); return; }
    ls.set('lma-demo.pendingShare', { key, how, t: Date.now() });   // cleared when share() settles; still set after a reload = the page died
    navigator.share(data).then(() => { rec('resolved (an action was chosen)'); status(stat, 'Done. If you chose Save to Files, the model is in Files now: check it there, then delete this page\'s copy.', 'ok'); },
      (e) => { rec(e && e.name === 'AbortError' ? 'closed without an action' : `error ${e && e.name}: ${String(e && e.message).slice(0, 100)}`); status(stat, e && e.name === 'AbortError' ? 'The share sheet closed without saving. Tap Save to Files again.' : 'The share sheet failed: ' + (e && e.message), 'warn'); })
      .finally(() => ls.del('lma-demo.pendingShare'));
  };
  save.addEventListener('click', () => share(file, 'opfs file')());
  saveMem.addEventListener('click', () => share(memFile, 'memory copy')());
  del.addEventListener('click', async () => {
    try { await root.removeEntry(name); } catch (e) { /* already gone */ }
    const s = st(); s.deleted = true; ls.set(skey, s); putA(key, s); file = null; memFile = null; await refresh();   // the record keeps done: it finished
  });
  refresh();
}
