// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// dl-worker.js - downloads one model file into this page's private storage (OPFS) with a sync access handle, which writes in place, so a
// download that dies (page killed, screen locked) resumes from the bytes already on disk with an HTTP Range request.
// in: { cmd: 'start', url, name, etag, stopAt } | { cmd: 'stop' }
// out: begin { have, total, status, etag, restarted } · progress { got, total } · done { got, total, etag } · stopped { got } · error { why, got }
'use strict';
let stopping = false, ctl = null;

self.onmessage = async (e) => {
  const m = e.data || {};
  if (m.cmd === 'probe') {   // createSyncAccessHandle exists only inside workers, so the page asks here
    self.postMessage({ t: 'probe', syncHandle: !!(self.FileSystemFileHandle && 'createSyncAccessHandle' in FileSystemFileHandle.prototype) }); return;
  }
  if (m.cmd === 'stop') { stopping = true; if (ctl) ctl.abort(); return; }
  if (m.cmd !== 'start') return;
  stopping = false; let h = null, got = 0;
  try {
    const root = await navigator.storage.getDirectory();
    const fh = await root.getFileHandle(m.name, { create: true });
    h = await fh.createSyncAccessHandle();
    let have = await h.getSize();   // (older Safari returned promises from these methods: await is harmless either way)
    const headers = {};
    if (have > 0) { headers.Range = 'bytes=' + have + '-'; if (m.etag) headers['If-Range'] = m.etag; }
    ctl = new AbortController();
    const r = await fetch(m.url, { headers, cache: 'no-store', credentials: 'same-origin', signal: ctl.signal });
    let restarted = false;
    if (have > 0 && r.status === 200) { await h.truncate(0); have = 0; restarted = true; }   // the server ignored the range or the file changed
    else if (r.status === 206) {
      const cr = /bytes (\d+)-\d+\/(\d+)/.exec(r.headers.get('content-range') || '');
      if (!cr || +cr[1] !== have) throw new Error('the server resumed at the wrong place (' + (r.headers.get('content-range') || 'no Content-Range') + ')');
    } else if (!r.ok) throw new Error('HTTP ' + r.status);
    const total = r.status === 206 ? +/\/(\d+)/.exec(r.headers.get('content-range'))[1] : +(r.headers.get('content-length') || 0);
    const etag = r.headers.get('etag') || '';
    self.postMessage({ t: 'begin', have, total, status: r.status, etag, restarted, ranges: r.headers.get('accept-ranges') || '' });
    const rd = r.body.getReader(); got = have; let last = 0, unflushed = 0;
    for (;;) {
      const { done, value } = await rd.read();
      if (done) break;
      h.write(value, { at: got }); got += value.byteLength; unflushed += value.byteLength;
      if (unflushed >= 32 << 20) { await h.flush(); unflushed = 0; }   // what is on disk survives a killed page
      const t = performance.now();
      if (t - last > 250) { last = t; self.postMessage({ t: 'progress', got, total }); }
      if (m.stopAt && got >= m.stopAt) { stopping = true; ctl.abort(); break; }   // test hook: stop part-way to test resuming
    }
    await h.flush(); const size = await h.getSize(); await h.close(); h = null;
    if (stopping) { self.postMessage({ t: 'stopped', got: size }); return; }
    self.postMessage({ t: 'done', got: size, total, etag });
  } catch (err) {
    let size = got;
    try { if (h) { await h.flush(); size = await h.getSize(); await h.close(); } } catch (e2) { /* */ }
    self.postMessage(stopping ? { t: 'stopped', got: size } : { t: 'error', why: String((err && err.message) || err).slice(0, 200), got: size });
  }
};
