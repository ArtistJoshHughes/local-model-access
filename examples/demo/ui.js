// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// ui.js - tiny DOM helpers for the demo pages.
export const MiB = 1 << 20;
export const $ = (sel, root = document) => root.querySelector(sel);
export const mb = (b) => (b == null ? '?' : (b / MiB).toFixed(b < 10 * MiB ? 2 : 1) + ' MB');
export const r1 = (x) => Math.round(x * 10) / 10;

/** h('div', { class: 'card', onclick: fn }, 'text', child, ...) */
export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (k === 'class') e.className = v; else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k instanceof Node ? k : String(k));
  return e;
}

export function status(el, text, kind = '') { el.textContent = text; el.className = 'status' + (kind ? ' ' + kind : ''); }

export function bar(host) {
  const i = h('i'), el = h('div', { class: 'bar hidden' }, i); host.append(el);
  return { el, set(f) { el.classList.remove('hidden'); i.style.width = Math.max(0, Math.min(100, f * 100)).toFixed(1) + '%'; }, hide() { el.classList.add('hidden'); } };
}

export function table(host, rows) {
  const t = h('table', {}, ...rows.filter(Boolean).map(([k, v]) => h('tr', {}, h('td', {}, k), h('td', {}, v)))); host.replaceChildren(t); return t;
}

export const pct = (a, b) => (b ? Math.round((100 * a) / b) + ' %' : '');
export function quantile(xs, q) { if (!xs.length) return null; const s = xs.slice().sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; }
