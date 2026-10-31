// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// lma-model.js - Local Model Access (LMA): <lma-model>, a small drop-in control that lets the visitor choose a model file on their
// device, checks it against the site's approved models (once, then remembered) and says when it is ready. Optional: the core
// (makePicker, readModel, verifyModel) stays usable headless. No dependencies; styles live in the shadow root.
//   <lma-model approved="./site.config.json"></lma-model>        or  el.models = [ ...approved entries ]
//   el.addEventListener('lma-ready', (e) => e.detail)            { files, model, entry, verified }   also: await el.ready
import { makePicker } from '../picker.js';
import { readModel, FormatError } from '../formats.js';
import { verifyModel } from '../verify.js';

export const LMA_HOME = 'https://github.com/ArtistJoshHughes/local-model-access';
export const TEXT = {
  empty: 'Use an AI model on this device', choose: 'Choose model file', get: 'Get the model', checking: 'Checking the model…',
  once: 'This happens once. Next time it is instant.', ready: 'Ready', change: 'Change', notApproved: "This site doesn't use that model",
  expects: 'It works with:', tryAgain: 'Choose another file', privacy: 'Your model stays on your device.', mark: 'Local Model Access',
  errors: { 'not-model': "That file isn't a model. Choose a .gguf, .safetensors or pack file.", truncated: 'That file looks incomplete. Download it again, then choose it.',
    'missing-part': 'This model comes in several files. Choose all of them together.', 'duplicate-part': 'The same file was chosen twice. Choose each file once.',
    mixed: 'Those files are from different models. Choose one model at a time.', version: "This site can't read that file's format version.",
    'bad-header': "This site can't read that file.", other: "Something went wrong reading the file. Try again." },
};

const CSS = `
:host { --lma-accent: #2f5fd0; --lma-accent-ink: #fff; --lma-bg: #fff; --lma-fg: #1d1f23; --lma-muted: #626872; --lma-border: #e3e3df;
  --lma-ok: #1e7a44; --lma-bad: #b3261e; --lma-track: #e9ecf3; --lma-radius: 14px; --lma-font: system-ui, -apple-system, "Segoe UI", sans-serif;
  display: block; font: 15px/1.45 var(--lma-font); color: var(--lma-fg); }
@media (prefers-color-scheme: dark) { :host { --lma-accent: #86a8ff; --lma-accent-ink: #0d1117; --lma-bg: #1c1f24; --lma-fg: #eceef1;
  --lma-muted: #a3a9b2; --lma-border: #30343b; --lma-ok: #6fd39a; --lma-bad: #ff8f86; --lma-track: #2a3140; } }
.card { background: var(--lma-bg); border: 1px solid var(--lma-border); border-radius: var(--lma-radius); padding: 16px; }
.title { font-weight: 600; font-size: 16px; margin: 0 0 4px; }
.sub, .privacy, .mark { color: var(--lma-muted); font-size: 13px; }
.status { margin: 0; } .status.ok { color: var(--lma-ok); } .status.bad { color: var(--lma-bad); }
.row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
.lma-pick, button { position: relative; overflow: hidden; display: inline-flex; align-items: center; justify-content: center; min-height: 44px; min-width: 44px;
  padding: 0 18px; border-radius: calc(var(--lma-radius) - 4px); border: 0; background: var(--lma-accent); color: var(--lma-accent-ink); font: 600 15px var(--lma-font); cursor: pointer; }
.lma-pick:focus-within, button:focus-visible, a:focus-visible { outline: 2px solid var(--lma-accent); outline-offset: 2px; }
.lma-pick.disabled { opacity: .5; pointer-events: none; }
button.quiet { background: transparent; color: var(--lma-accent); padding: 0 8px; }
a { color: var(--lma-accent); text-decoration: none; } a:hover { text-decoration: underline; }
.bar { height: 6px; border-radius: 3px; background: var(--lma-track); overflow: hidden; margin-top: 10px; }
.bar > i { display: block; height: 100%; width: 0; background: var(--lma-accent); transition: width .25s ease; }
.foot { display: flex; justify-content: space-between; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
.expects { margin: 6px 0 0; padding-left: 18px; color: var(--lma-muted); font-size: 13px; }
[hidden] { display: none !important; }
@media (prefers-reduced-motion: reduce) { .bar > i { transition: none; } }`;

const hfUrl = (e) => {
  if (!e || !e.repo || /^local\//.test(e.repo)) return null;
  const rev = e.revision || 'main', one = !e.files && !e.manifest && e.file;
  return one ? `https://huggingface.co/${e.repo}/resolve/${rev}/${encodeURIComponent(e.file)}?download=true` : `https://huggingface.co/${e.repo}/tree/${rev}`;
};
const label = (e) => (e && (e.name || e.file || e.repo)) || 'the model';

export class LmaModel extends HTMLElement {
  static get observedAttributes() { return ['approved']; }

  constructor() {
    super();
    this._models = null; this._state = 'empty'; this._busy = false; this.config = null;
    this._newReady();
    const r = this.attachShadow({ mode: 'open' });
    r.innerHTML = `<style>${CSS}</style><div class="card" part="card" role="group" aria-label="AI model">
  <p class="title" part="title"></p><p class="status" part="status" aria-live="polite"></p>
  <div class="bar" part="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" hidden><i></i></div>
  <ul class="expects" hidden></ul>
  <div class="row"><span class="pick"></span><button class="quiet change" part="change" hidden></button><a class="get" part="link" target="_blank" rel="noopener" hidden></a></div>
  <div class="foot"><span class="privacy" part="privacy"></span><a class="mark" part="mark" href="${LMA_HOME}" target="_blank" rel="noopener"></a></div></div>`;
    this.$ = (s) => r.querySelector(s);
    this.$('.privacy').textContent = TEXT.privacy; this.$('.mark').textContent = TEXT.mark; this.$('.change').textContent = TEXT.change;
    this._picker = makePicker(this.$('.pick'), { label: TEXT.choose, multiple: true, onPick: (f) => this._pick(f) });
    this._picker.el.setAttribute('part', 'button');
    this.$('.change').addEventListener('click', () => this.reset());
    this._render();
  }

  _newReady() { this.ready = new Promise((res) => { this._resolve = res; }); }

  /** the approved entries (an array), or a URL in the `approved` attribute: a site config ({ models }), a registry, or an array */
  get models() { return this._models || []; }
  set models(list) { this._models = Array.isArray(list) ? list : (list && list.models) || []; this._render(); }
  get state() { return this._state; }

  attributeChangedCallback(n, o, v) { if (n === 'approved' && v && v !== o) this._load(v); }

  async _load(url) {
    try {
      const j = await (await fetch(url, { cache: 'no-store' })).json();
      this.config = Array.isArray(j) ? null : j; this.models = Array.isArray(j) ? j : j.models || [];
    } catch (e) { this._set('error', { message: "This site's model list could not be loaded." }); this._emit('lma-error', { error: e, message: String(e.message || e) }); }
  }

  get ns() { return this.getAttribute('ns') || (this.config && this.config.site && this.config.site.id) || location.host || 'lma'; }

  /** back to the empty state (the "Change" action) */
  reset() { this._busy = false; this._newReady(); this._set('empty'); }

  /** render a state without a file, for design previews and screenshots: preview('checking', { percent: 42 }) */
  preview(state, info = {}) { this._set(state, info); }

  async _pick(files) {
    if (!files || !files.length || this._busy) return;
    this._busy = true; this._picker.setDisabled(true);
    try {
      const model = await readModel(files);
      this._set('checking', { percent: 0 });
      let done = 0; const total = (model.files || files).reduce((n, f) => n + f.size, 0) || 1, seen = new Map();
      const v = await verifyModel(files, { approved: this.models, ns: this.ns, model, lazy: this.hasAttribute('lazy'),
        onProgress: (p) => { seen.set(p.file, p.done); done = [...seen.values()].reduce((a, b) => a + b, 0); this._set('checking', { percent: Math.min(99, Math.floor((100 * done) / total)) }); } });
      if (!v.ok) {
        if (v.missing && v.missing.length) this._set('error', { message: TEXT.errors['missing-part'] + ' ' + v.missing.slice(0, 3).join(', ') });
        else this._set('not-approved', {});
        this._emit('lma-error', { code: v.missing && v.missing.length ? 'missing-part' : 'not-approved', message: v.problems.join(' '), verification: v });
        return;
      }
      // files: the verified data files in part order (what a Loader must read); picked: everything the user chose, as chosen
      const detail = { files: model.files, picked: files, model, entry: v.entry, verified: v };
      this._set('ready', { name: label(v.entry) });
      this._emit('lma-ready', detail); this._resolve(detail);
    } catch (e) {
      const code = e instanceof FormatError ? e.code : 'other';
      this._set('error', { message: TEXT.errors[code] || TEXT.errors.other }); this._emit('lma-error', { code, error: e, message: String(e.message || e) });
    } finally { this._busy = false; this._picker.setDisabled(false); }
  }

  _emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true, composed: true })); }

  _set(state, info = {}) {
    const prev = this._state; this._state = state; this._info = info; this._render();
    if (prev !== state) this._emit('lma-change', { state, previous: prev });
  }

  _render() {
    if (!this.$) return;
    const s = this._state, i = this._info || {}, st = this.$('.status'), bar = this.$('.bar'), ex = this.$('.expects'), get = this.$('.get');
    const first = this.models[0], url = this.getAttribute('get-url') || hfUrl(first);
    this.$('.title').textContent = s === 'ready' ? `${TEXT.ready} · ${i.name || ''}` : s === 'not-approved' ? TEXT.notApproved : s === 'checking' ? `${TEXT.checking} ${i.percent || 0} %` : TEXT.empty;
    st.className = 'status' + (s === 'ready' ? ' ok' : s === 'error' || s === 'not-approved' ? ' bad' : '');
    st.textContent = s === 'checking' ? TEXT.once : s === 'error' ? i.message || TEXT.errors.other : s === 'not-approved' ? TEXT.expects : '';
    st.hidden = !st.textContent;
    bar.hidden = s !== 'checking'; bar.firstElementChild.style.width = (i.percent || 0) + '%'; bar.setAttribute('aria-valuenow', String(i.percent || 0));
    ex.hidden = s !== 'not-approved'; ex.replaceChildren(...(s === 'not-approved' ? this.models.filter((e) => !e.role).slice(0, 4).map((e) => { const li = document.createElement('li'); li.textContent = label(e); return li; }) : []));
    this.$('.pick').hidden = s === 'ready' || s === 'checking';
    this._picker.setLabel(s === 'not-approved' || s === 'error' ? TEXT.tryAgain : TEXT.choose);
    this.$('.change').hidden = s !== 'ready';
    get.hidden = !url || s === 'ready' || s === 'checking'; if (url) { get.href = url; get.textContent = TEXT.get; }
  }
}

if (!customElements.get('lma-model')) customElements.define('lma-model', LmaModel);
