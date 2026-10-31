// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// lma-chat.js - Local Model Access (LMA): <lma-chat>, a lower-right chat widget. First use shows <lma-model>; once the model is
// ready, the panel becomes the chat. The model runs in an ENGINE the site supplies (the slot below); this file runs no model itself.
//   <lma-chat config="./site.config.json"></lma-chat>
//   chat.engine = myEngine;   // { async load({ files, model, entry, onProgress }), async *generate({ messages, system, tools, signal, sampling }), unload() }
// generate yields { text } deltas and/or { tool_call: { name, arguments } }. Tool calls run through chat.site.callTool (the site kit) or
// chat.onToolCall(call); the result goes back to the engine as a { role: 'tool' } message. Plain chat needs neither.
import './lma-model.js';
import { TEXT as MTEXT, LMA_HOME } from './lma-model.js';

export const CHAT_TEXT = { open: 'Open chat', close: 'Close', send: 'Send', stop: 'Stop', placeholder: 'Ask a question', loading: 'Loading the model…',
  stopped: '(stopped)', failed: 'The model could not answer. Try again.', noEngine: 'This site has no engine for the model yet.' };

const CSS = `
:host { --lma-accent: #2f5fd0; --lma-accent-ink: #fff; --lma-bg: #fff; --lma-fg: #1d1f23; --lma-muted: #626872; --lma-border: #e3e3df;
  --lma-bubble: #f1f2f5; --lma-radius: 16px; --lma-font: system-ui, -apple-system, "Segoe UI", sans-serif; --lma-shadow: 0 12px 40px rgba(0,0,0,.18);
  --lma-gap: 16px; font: 15px/1.45 var(--lma-font); color: var(--lma-fg); }
*, *::before, *::after { box-sizing: border-box; }
@media (prefers-color-scheme: dark) { :host { --lma-accent: #86a8ff; --lma-accent-ink: #0d1117; --lma-bg: #1c1f24; --lma-fg: #eceef1;
  --lma-muted: #a3a9b2; --lma-border: #30343b; --lma-bubble: #2a2e35; --lma-shadow: 0 12px 40px rgba(0,0,0,.5); } }
.launch { position: fixed; z-index: 2147483000; bottom: calc(var(--lma-gap) + env(safe-area-inset-bottom, 0px)); right: calc(var(--lma-gap) + env(safe-area-inset-right, 0px));
  width: 56px; height: 56px; border-radius: 50%; border: 0; background: var(--lma-accent); color: var(--lma-accent-ink); box-shadow: var(--lma-shadow); cursor: pointer;
  display: grid; place-items: center; }
:host([position="bottom-left"]) .launch, :host([position="bottom-left"]) .panel { right: auto; left: calc(var(--lma-gap) + env(safe-area-inset-left, 0px)); }
.panel { position: fixed; z-index: 2147483001; bottom: calc(var(--lma-gap) + 68px + env(safe-area-inset-bottom, 0px)); right: calc(var(--lma-gap) + env(safe-area-inset-right, 0px));
  width: 380px; height: 560px; max-height: calc(100vh - 110px); display: flex; flex-direction: column; background: var(--lma-bg); color: var(--lma-fg);
  border: 1px solid var(--lma-border); border-radius: var(--lma-radius); box-shadow: var(--lma-shadow); overflow: hidden; }
.scrim { position: fixed; inset: 0; z-index: 2147483000; background: rgba(0, 0, 0, .35); }
@media (min-width: 521px) { .scrim { display: none; } }
@media (max-width: 520px) { .panel { left: 0 !important; right: 0 !important; bottom: 0; width: auto; max-width: 100%; height: auto; top: calc(env(safe-area-inset-top, 0px) + 6vh); max-height: none;
  border-radius: var(--lma-radius) var(--lma-radius) 0 0; padding: 0 env(safe-area-inset-right, 0px) env(safe-area-inset-bottom, 0px) env(safe-area-inset-left, 0px); } }
.head { display: flex; align-items: center; gap: 8px; padding: 12px 12px 12px 16px; border-bottom: 1px solid var(--lma-border); }
.head .t { flex: 1; min-width: 0; } .name { font-weight: 600; } .subt { color: var(--lma-muted); font-size: 13px; }
button { min-width: 44px; min-height: 44px; border-radius: 12px; border: 0; font: 600 15px var(--lma-font); cursor: pointer; background: var(--lma-accent); color: var(--lma-accent-ink); padding: 0 14px; }
button.x { background: transparent; color: var(--lma-muted); font-size: 22px; padding: 0; }
button:focus-visible, textarea:focus-visible, a:focus-visible { outline: 2px solid var(--lma-accent); outline-offset: 2px; }
.body { flex: 1; overflow: auto; padding: 14px 16px; }
.log { display: flex; flex-direction: column; gap: 10px; }
.msg { max-width: 85%; padding: 9px 12px; border-radius: 14px; white-space: pre-wrap; word-break: break-word; }
.msg.user { align-self: flex-end; background: var(--lma-accent); color: var(--lma-accent-ink); border-bottom-right-radius: 4px; }
.msg.assistant { align-self: flex-start; background: var(--lma-bubble); border-bottom-left-radius: 4px; }
.msg.note { align-self: center; background: none; color: var(--lma-muted); font-size: 13px; }
.starters { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.starters button { background: var(--lma-bubble); color: var(--lma-fg); font-weight: 500; }
.compose { display: flex; gap: 8px; padding: 10px 12px; border-top: 1px solid var(--lma-border); }
.compose button { flex: none; }
/* 16px, not 15: iOS Safari zooms the whole page when a focused field's text is under 16px, which pushed the sheet off the right edge */
textarea { flex: 1; min-width: 0; resize: none; min-height: 44px; max-height: 120px; padding: 11px 12px; border-radius: 12px; border: 1px solid var(--lma-border); background: var(--lma-bg); color: var(--lma-fg); font: 16px/1.4 var(--lma-font); }
.foot { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 8px; padding: 0 16px 10px; color: var(--lma-muted); font-size: 12px; }
.foot a { color: var(--lma-muted); text-decoration: none; } .foot a:hover { text-decoration: underline; }
lma-model { --lma-radius: 12px; }
lma-model::part(privacy), lma-model::part(mark) { display: none; }   /* the panel's own footer carries them */
[hidden] { display: none !important; }
@media (prefers-reduced-motion: no-preference) { .panel { animation: up .18s ease-out; } @keyframes up { from { opacity: 0; transform: translateY(8px); } } }`;

const ICON = '<svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v11H9l-5 4z"/></svg>';

export class LmaChat extends HTMLElement {
  static get observedAttributes() { return ['config', 'open']; }

  constructor() {
    super();
    this.engine = null; this.site = null; this.onToolCall = null; this.messages = []; this._config = null; this._ctl = null; this._loaded = false;
    const r = this.attachShadow({ mode: 'open' });
    r.innerHTML = `<style>${CSS}</style>
<button class="launch" part="launcher" aria-haspopup="dialog" aria-expanded="false">${ICON}</button>
<div class="scrim" part="scrim" hidden></div>
<div class="panel" part="panel" role="dialog" aria-modal="true" aria-labelledby="lma-chat-name" hidden>
  <div class="head" part="header"><div class="t"><div class="name" id="lma-chat-name"></div><div class="subt"></div></div><button class="x" part="close">×</button></div>
  <div class="body"><lma-model part="model"></lma-model><div class="log" role="log" aria-live="polite" hidden></div><div class="starters" hidden></div></div>
  <div class="compose" part="compose" hidden><textarea rows="1" part="input"></textarea><button class="send" part="send"></button><button class="stop" part="stop" hidden></button></div>
  <div class="foot"><span class="privacy"></span><a part="mark" href="${LMA_HOME}" target="_blank" rel="noopener"></a></div>
</div>`;
    this.$ = (s) => r.querySelector(s);
    this.$('.foot .privacy').textContent = MTEXT.privacy; this.$('.foot a').textContent = MTEXT.mark;
    const T = CHAT_TEXT;
    this.$('.launch').setAttribute('aria-label', T.open); this.$('.x').setAttribute('aria-label', T.close);
    this.$('.send').textContent = T.send; this.$('.stop').textContent = T.stop; this.$('textarea').placeholder = T.placeholder;
    this.$('.launch').addEventListener('click', () => { this.open = !this.open; });
    this.$('.x').addEventListener('click', () => { this.open = false; });
    this.$('.scrim').addEventListener('click', () => { this.open = false; });
    this.$('.send').addEventListener('click', () => this.send());
    this.$('.stop').addEventListener('click', () => this.stop());
    this.$('textarea').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.send(); } });
    this.$('.panel').addEventListener('keydown', (e) => this._keys(e));
    this.$('lma-model').addEventListener('lma-ready', (e) => this._ready(e.detail));
  }

  get config() { return this._config; }
  set config(c) { this._config = c; this._apply(); }
  get open() { return this.hasAttribute('open'); }
  set open(v) { if (v) this.setAttribute('open', ''); else this.removeAttribute('open'); }
  get model() { return this.$('lma-model'); }

  attributeChangedCallback(n, o, v) {
    if (n === 'config' && v && v !== o) fetch(v, { cache: 'no-store' }).then((r) => r.json()).then((c) => { this.config = c; }).catch(() => this._note("This site's chat settings could not be loaded."));
    if (n === 'open') this._toggle(v !== null);
  }

  _apply() {
    const c = this._config || {}, s = c.site || {}, ch = c.chat || {};
    this.$('.name').textContent = ch.title || s.name || ''; this.$('.subt').textContent = ch.subtitle || s.job || '';
    this.$('.launch').setAttribute('aria-label', `${CHAT_TEXT.open}${s.name ? ': ' + s.name : ''}`);
    const m = this.$('lma-model'); m.config = c; if (s.id) m.setAttribute('ns', s.id); m.models = c.models || [];
  }

  _toggle(on) {
    const p = this.$('.panel'), l = this.$('.launch');
    if (on === !p.hidden) return;
    p.hidden = !on; this.$('.scrim').hidden = !on; l.setAttribute('aria-expanded', String(on));
    if (on) { this._focusFirst(); this._emit('lma-chat-open', {}); }
    else { l.focus(); this._emit('lma-chat-close', {}); }
  }

  _focusables() { return [...this.shadowRoot.querySelectorAll('.panel button:not([hidden]), .panel textarea, .panel a')].concat([...(this.model.shadowRoot ? this.model.shadowRoot.querySelectorAll('input, button:not([hidden]), a:not([hidden])') : [])])
    .filter((x) => !x.hidden && !x.disabled && x.offsetParent !== null); }

  _focusFirst() { requestAnimationFrame(() => { const t = this.$('textarea'); (this._loaded && !this.$('.compose').hidden ? t : this.$('.x')).focus(); }); }

  _keys(e) {
    if (e.key === 'Escape') { e.preventDefault(); this.open = false; return; }
    if (e.key !== 'Tab') return;
    const f = this._focusables(); if (!f.length) return;
    const active = this.shadowRoot.activeElement && this.shadowRoot.activeElement.shadowRoot ? this.shadowRoot.activeElement.shadowRoot.activeElement : this.shadowRoot.activeElement;
    const i = f.indexOf(active);
    if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); } else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
  }

  async _ready(d) {
    this.detail = d; this._note(CHAT_TEXT.loading);
    try {
      if (!this.engine) throw new Error(CHAT_TEXT.noEngine);
      await this.engine.load({ files: d.model.files, model: d.model, entry: d.entry, verified: d.verified, onProgress: () => {} });
      this._loaded = true; this.$('.log').replaceChildren(); this.$('.log').hidden = false; this.$('.compose').hidden = false; this.$('lma-model').hidden = true;
      this._starters(); this._focusFirst(); this._emit('lma-chat-ready', { entry: d.entry, model: d.model });
    } catch (e) { this._note(String(e.message || e)); this._emit('lma-error', { code: 'engine', error: e, message: String(e.message || e) }); }
  }

  _starters() {
    const box = this.$('.starters'), list = ((this._config || {}).chat || {}).starters || [];
    box.replaceChildren(...list.map((t) => { const b = document.createElement('button'); b.textContent = t; b.addEventListener('click', () => { this.$('textarea').value = t; this.send(); }); return b; }));
    box.hidden = !list.length;
  }

  _note(text) { const log = this.$('.log'); log.hidden = false; const n = document.createElement('div'); n.className = 'msg note'; n.textContent = text; log.append(n); return n; }
  _bubble(role, text) { const b = document.createElement('div'); b.className = 'msg ' + role; b.setAttribute('part', 'message ' + role); b.textContent = text; this.$('.log').append(b); this.$('.body').scrollTop = 1e9; return b; }
  _emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true, composed: true })); }

  stop() { if (this._ctl) this._ctl.abort(); }

  /** send the text in the box (or `text`) and stream the engine's reply */
  async send(text) {
    const box = this.$('textarea'), t = (text != null ? text : box.value).trim();
    if (!t || this._ctl || !this._loaded) return;
    box.value = ''; this.$('.starters').hidden = true;
    this.messages.push({ role: 'user', content: t }); this._bubble('user', t); this._emit('lma-message', { role: 'user', text: t });
    const c = this._config || {}, tools = this.site && this.site.tools ? this.site.tools() : c.tools || null;
    this._ctl = new AbortController(); this.$('.send').hidden = true; this.$('.stop').hidden = false;
    let out = null, reply = '';
    try {
      for (let round = 0; round < 4; round++) {
        let call = null; out = out || this._bubble('assistant', '');
        for await (const d of this.engine.generate({ messages: this.messages.slice(), system: (c.prompt && c.prompt.system) || '', tools, signal: this._ctl.signal, sampling: c.sampling })) {
          if (this._ctl.signal.aborted) break;
          if (d.text) { reply += d.text; out.textContent = reply; this.$('.body').scrollTop = 1e9; }
          if (d.tool_call) { call = d.tool_call; break; }
        }
        if (!call || this._ctl.signal.aborted) break;
        const result = await this._tool(call);
        this.messages.push({ role: 'assistant', content: reply, tool_call: call }, { role: 'tool', name: call.name, content: JSON.stringify(result) });
        this._emit('lma-tool', { call, result });
      }
      if (this._ctl.signal.aborted) out.textContent = (reply ? reply + ' ' : '') + CHAT_TEXT.stopped;
    } catch (e) {
      if (e && e.name !== 'AbortError') { (out || this._bubble('assistant', '')).textContent = CHAT_TEXT.failed; this._emit('lma-error', { code: 'generate', error: e, message: String(e.message || e) }); }
    } finally {
      if (reply) { this.messages.push({ role: 'assistant', content: reply }); this._emit('lma-message', { role: 'assistant', text: reply }); }
      this._ctl = null; this.$('.send').hidden = false; this.$('.stop').hidden = true; box.focus();
    }
  }

  async _tool(call) {
    if (this.site && this.site.callTool) return this.site.callTool(call.name, call.arguments || {}, { chat: this });
    if (this.onToolCall) return this.onToolCall(call);
    return { error: 'no handler for tool ' + call.name };
  }
}

if (!customElements.get('lma-chat')) customElements.define('lma-chat', LmaChat);
