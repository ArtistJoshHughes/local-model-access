// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// report.js - the demo's report: each demo page records what happened (in this browser's localStorage, a few KB) and any page can show
// the whole report as text to copy or share. This is test tooling for the demo, not part of the standard: on separate hosts each page
// simply shows its own part.
import { h, mb, r1 } from './ui.js';

const KEY = 'lma-demo.report.v1';
export const PAGE_VERSION = 'demo 1';
const ls = () => { try { return globalThis.localStorage || null; } catch (e) { return null; } };

export function all() { try { return JSON.parse((ls() && ls().getItem(KEY)) || '{}'); } catch (e) { return {}; } }
function write(a) { try { const s = ls(); if (s) s.setItem(KEY, JSON.stringify(a)); } catch (e) { /* storage blocked: the page still shows its own results */ } }
export function put(section, data) { const a = all(); a[section] = Object.assign(a[section] || {}, data, { updated: new Date().toISOString() }); write(a); return a[section]; }
export function push(section, key, item) { const a = all(); const s = (a[section] = a[section] || {}); (s[key] = s[key] || []).push(item); s.updated = new Date().toISOString(); write(a); }
export function clear() { try { const s = ls(); if (s) s.removeItem(KEY); } catch (e) { /* */ } }

/** what this browser can do (the parts the standard depends on) */
export async function env() {
  const FH = globalThis.FileSystemFileHandle && FileSystemFileHandle.prototype;
  const e = {
    ua: navigator.userAgent, standalone: !!(navigator.standalone || (matchMedia && matchMedia('(display-mode: standalone)').matches)),
    secure: !!globalThis.isSecureContext, opfs: !!(navigator.storage && navigator.storage.getDirectory), createWritable: !!(FH && 'createWritable' in FH),
    syncHandle: !!(FH && 'createSyncAccessHandle' in FH), share: !!navigator.share, wakeLock: !!navigator.wakeLock, webgpu: !!navigator.gpu,
    moduleWorker: true, cores: navigator.hardwareConcurrency || null,
  };
  try { e.shareFiles = !!(navigator.canShare && navigator.canShare({ files: [new File(['x'], 'x.bin', { type: 'application/octet-stream' })] })); } catch (err) { e.shareFiles = false; }
  if (navigator.gpu) {
    try { const a = await navigator.gpu.requestAdapter(); const i = (a && a.info) || {}; e.gpu = a ? [i.vendor, i.architecture, i.device].filter(Boolean).join(' ') || 'adapter' : 'no adapter'; } catch (err) { e.gpu = 'error'; }
  }
  const m = /OS (\d+)[._](\d+)(?:[._](\d+))? like Mac OS X/.exec(navigator.userAgent); if (m) e.ios = m.slice(1).filter(Boolean).join('.');
  return e;
}

const yes = (b) => (b ? 'yes' : 'no');
const ms = (x) => (x == null ? '?' : Math.round(x) + ' ms');
const grow = (a, b) => (a && b && a.usage != null && b.usage != null ? (b.usage - a.usage >= 0 ? '+' : '') + mb(b.usage - a.usage) : '?');

function siteLines(tag, s) {
  const out = [];
  for (const [i, p] of (s.picks || []).entries()) {
    const f = p.file || {}, v = p.verify || {}, bits = [`${tag} pick ${i + 1}: ${f.name} ${mb(f.size)}${p.format ? ' (' + p.format + ')' : ''}`];
    if (p.pick) bits.push(`tap->file ${ms(p.pick.tapToFileMs)}` + (p.pick.closedToFileMs != null ? `, closed->file ${ms(p.pick.closedToFileMs)}` : ''));
    if (p.header) bits.push(`header ${ms(p.header.ms)} (${p.header.reads} read${p.header.reads === 1 ? '' : 's'}, ${p.header.tensors} tensors)`);
    if (p.error) bits.push('ERROR ' + p.error);
    if (v.how) bits.push(v.ok ? `approved: ${v.how}${v.hash ? ` in ${r1(v.hash.ms / 1000)} s at ${Math.round(v.hash.bytes / 1048576 / (v.hash.ms / 1000))} MB/s (${v.hash.where})` : ` in ${ms(v.ms)}`}` : `REJECTED: ${(v.problems || []).join('; ')}`);
    if (p.readyMs != null) bits.push(`ready ${ms(p.readyMs)} after the pick`);
    if (p.seq) bits.push(`read ${p.seq.sliceMB} MB slices ${p.seq.MBs} MB/s, stream ${p.seq.streamMBs} MB/s`);
    if (p.random) bits.push(`random ${p.random.sizeKB} KB reads p50 ${ms(p.random.p50)} p95 ${ms(p.random.p95)}`);
    if (p.gemv) bits.push(p.gemv.error ? 'GEMV ERROR ' + p.gemv.error : `GEMV stream ${p.gemv.layers} layers ${mb(p.gemv.bytes)} in ${r1(p.gemv.s)} s = ${p.gemv.GBs} GB/s, answers ${p.gemv.exact}/${p.gemv.layers} exact, y ${p.gemv.yOk ? 'ok' : 'BAD'}`);
    if (p.resident) bits.push(p.resident.error ? 'resident ERROR ' + p.resident.error : `resident ${p.resident.layers} layers x ${p.resident.iters} = ${p.resident.GBs} GB/s (${p.resident.exact}/${p.resident.layers} exact)`);
    if (p.memory) bits.push(`CPU peak ${mb(p.memory.cpuPeak)}` + (p.memory.gpuPeak != null ? `, GPU peak ${mb(p.memory.gpuPeak)}` : '') + (p.memory.hashPeak ? `, hash ${mb(p.memory.hashPeak)}` : ''));
    if (p.storageBefore || p.storageAfter) bits.push(`site storage ${grow(p.storageBefore, p.storageAfter)}`);
    out.push(bits.join(' · '));
  }
  return out;
}

export function text(a = all()) {
  const L = [`Local Model Access demo report · ${new Date().toISOString()} · ${PAGE_VERSION}`];
  const e = a.env;
  if (e) L.push(`ENV ${e.standalone ? 'Home Screen app' : 'browser tab'} · iOS ${e.ios || '?'} · secure ${yes(e.secure)} · OPFS ${yes(e.opfs)} · createWritable ${yes(e.createWritable)} · sync handle ${yes(e.syncHandle)} · share files ${yes(e.shareFiles)} · WebGPU ${e.webgpu ? e.gpu || 'yes' : 'no'} · wake lock ${yes(e.wakeLock)}`);
  const g = a.get || {};
  for (const [k, s] of Object.entries(g.A || {})) {
    L.push(`GET A ${k}: ` + [s.done ? `downloaded ${mb(s.size)} in ${r1(s.seconds || 0)} s (${s.seconds ? r1(s.size / 1048576 / s.seconds) : '?'} MB/s)` : `not finished (${mb(s.have || 0)})`,
      `writer ${s.writer}`, `${s.resumes || 0} resumes, ${s.interrupts || 0} interruptions, ${s.restarts || 0} restarts`, s.header ? 'header ' + s.header : null,
      s.lastError ? 'last error ' + s.lastError : null, (s.shares || []).length ? 'share: ' + s.shares.map((x) => `${x.how} ${x.result}`).join(', ') : 'share: not tried',
      s.deleted ? 'page copy deleted' : null].filter(Boolean).join(' · '));
  }
  for (const t of g.B || []) L.push(`GET B ${t.key}: Safari download link tapped ${new Date(t.t).toLocaleTimeString()} (${t.standalone ? 'Home Screen app' : 'browser tab'})`);
  if (a.siteA) L.push(...siteLines('SITE A', a.siteA));
  if (a.siteB) L.push(...siteLines('SITE B', a.siteB));
  L.push('', '--- details ---', JSON.stringify(a));
  return L.join('\n');
}

/** a report box: the text, Copy, Share, Clear */
export function mount(host) {
  const ta = h('textarea', { readonly: true, 'aria-label': 'Report' }), msg = h('p', { class: 'sub' });
  const refresh = () => { ta.value = text(); };
  const copy = h('button', { class: 'btn', onclick: async () => {
    refresh();
    try { await navigator.clipboard.writeText(ta.value); msg.textContent = 'Copied. Paste it into the chat.'; }
    catch (e) { ta.focus(); ta.select(); msg.textContent = 'Select all and copy (the clipboard was refused).'; }
  } }, 'Copy the report');
  const share = h('button', { class: 'btn secondary', onclick: async () => { refresh(); try { await navigator.share({ text: ta.value }); } catch (e) { /* closed */ } } }, 'Share the report');
  const clr = h('button', { class: 'btn secondary', onclick: () => { if (confirm('Clear the whole report?')) { clear(); refresh(); } } }, 'Clear the report');
  host.append(...[ta, copy, navigator.share ? share : null, clr, msg].filter(Boolean));
  refresh();
  return { refresh };
}
