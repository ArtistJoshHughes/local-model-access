// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// ui.browser.mjs - the drop-in UI (<lma-model>, <lma-chat> + the stand-in engine) in headless Chrome, one short run (Node 22+):
//   node tests/ui.browser.mjs [OUTDIR]        CHROME=/path/to/chrome to choose the browser; screenshots (PNG) go to OUTDIR
// Checks: both elements register; <lma-model> renders every state; a real pick of a test vector reaches 'ready' and fires 'lma-ready';
// the reference page refuses a split model with a part missing; <lma-chat> opens, its model becomes ready, the stand-in streams a reply,
// the tool-call path runs a site-kit skill, Escape closes and focus returns to the launcher; phone width shows a bottom sheet.
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url)), OUT = path.resolve(process.argv[2] || path.join(os.tmpdir(), 'lma-ui-shots'));
fs.mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css' };
const STATES = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;padding:20px;font:15px system-ui;background:#f7f6f3;display:grid;gap:14px;max-width:460px} @media (prefers-color-scheme: dark){body{background:#131417}}</style>
<script type="module">import '/lma/ui/lma-model.js';
const models = JSON.parse(await (await fetch('/registry/models.json')).text()).models;
for (const [s, i] of [['empty', {}], ['checking', { percent: 42 }], ['ready', { name: models[0].name }], ['not-approved', {}], ['error', { message: "That file looks incomplete. Download it again, then choose it." }]]) {
  const el = document.createElement('lma-model'); el.models = models; document.body.append(el); el.preview(s, i); }
window.__done = true;</script>`;
const server = http.createServer((q, s) => {
  const u = decodeURIComponent(q.url.split('?')[0]);
  if (u === '/__states.html') { s.writeHead(200, { 'Content-Type': 'text/html' }); return s.end(STATES); }
  let f = path.join(ROOT, u); if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, 'index.html');
  if (!f.startsWith(ROOT) || !fs.existsSync(f)) { s.writeHead(404); return s.end(); }
  s.writeHead(200, { 'Content-Type': TYPES[path.extname(f)] || 'application/octet-stream', 'Cache-Control': 'no-store' }); fs.createReadStream(f).pipe(s);
});
await new Promise((r) => server.listen(8691, '127.0.0.1', r));
const CHROME = process.env.CHROME || ['C:/Program Files/Google/Chrome/Application/chrome.exe', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].find((p) => fs.existsSync(p));
const prof = fs.mkdtempSync(path.join(os.tmpdir(), 'lma-ui-'));
const br = spawn(CHROME, ['--headless=new', `--user-data-dir=${prof}`, '--remote-debugging-port=9691', '--no-first-run', '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });
let fails = 0; const ok = (c, what) => { console.log((c ? 'ok   ' : 'FAIL ') + what); if (!c) fails++; };
try {
  let list = null; for (let i = 0; i < 100 && !list; i++) { try { list = await (await fetch('http://127.0.0.1:9691/json/list')).json(); } catch (e) { await sleep(100); } }
  const ws = new WebSocket(list.find((t) => t.type === 'page').webSocketDebuggerUrl); await new Promise((r) => { ws.onopen = r; });
  let id = 0; const pend = new Map(), chooser = [], exc = [];
  ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } else if (m.method === 'Page.fileChooserOpened') chooser.push(m.params); else if (m.method === 'Runtime.exceptionThrown') exc.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text); };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expression, gesture) => { const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: !!gesture });
    if (r.result.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text); return r.result.result.value; };
  const until = async (expr, ms = 15000) => { for (let t = 0; t < ms; t += 100) { try { if (await ev(expr)) return true; } catch (e) { /* loading */ } await sleep(100); } return false; };
  const shot = async (name) => { const r = await send('Page.captureScreenshot', { format: 'png' }); const p = path.join(OUT, name + '.png'); fs.writeFileSync(p, Buffer.from(r.result.data, 'base64')); return p; };
  const view = (w, h, mobile = false) => send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: mobile ? 2 : 1, mobile });
  const dark = (on) => send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: on ? 'dark' : 'light' }] });
  const go = async (u, ready) => { await send('Page.navigate', { url: 'http://127.0.0.1:8691' + u }); return until(ready); };
  const pick = async (inputExpr, files) => {
    chooser.length = 0; await send('Runtime.evaluate', { expression: `(${inputExpr}).click(), true`, userGesture: true });
    for (let i = 0; i < 60 && !chooser.length; i++) await sleep(50);
    if (!chooser.length) return false;
    await send('DOM.setFileInputFiles', { files: files.map((f) => path.join(ROOT, f)), backendNodeId: chooser[0].backendNodeId }); return true;
  };
  await send('Page.enable'); await send('Runtime.enable'); await send('DOM.enable'); await send('Page.setInterceptFileChooserDialog', { enabled: true });
  const shots = [];

  // 1. <lma-model>: every state
  await view(520, 900); await dark(false);
  ok(await go('/__states.html', 'window.__done && document.querySelectorAll("lma-model").length === 5'), '<lma-model> registers and renders five states');
  const titles = await ev(`[...document.querySelectorAll('lma-model')].map((e) => e.shadowRoot.querySelector('.title').textContent + ' | ' + e.shadowRoot.querySelector('.privacy').textContent)`);
  ok(/^Use an AI model on this device/.test(titles[0]) && /^Checking the model… 42 %/.test(titles[1]) && /^Ready · /.test(titles[2]) && /^This site doesn't use that model/.test(titles[3]) && titles.every((t) => /Your model stays on your device\./.test(t)), 'state wording: ' + titles.map((t) => t.split(' | ')[0]).join(' / '));
  ok(await ev(`(() => { const r = document.querySelector('lma-model').shadowRoot; const b = r.querySelector('.lma-pick').getBoundingClientRect(); return b.height >= 44 && !!r.querySelector('input[type=file]') && !r.querySelector('input[type=file]').accept && r.querySelector('.get').href.startsWith('https://huggingface.co/'); })()`), 'a 44 px real file input without an accept filter, and a "Get the model" link to Hugging Face');
  shots.push(await shot('lma-model-states-light')); await dark(true); shots.push(await shot('lma-model-states-dark')); await dark(false);

  // 2. the reference page: real picks
  await view(1100, 900);
  ok(await go('/reference/index.html', `/approved entries/.test(document.querySelector('#count').textContent)`), 'reference page loads <lma-model> with the registry and the vectors');
  await ev(`window.__ev = []; for (const t of ['lma-ready', 'lma-error', 'lma-change']) document.addEventListener(t, (e) => window.__ev.push(t + ':' + (e.detail.state || e.detail.code || (e.detail.entry && e.detail.entry.name))))`);
  const refInput = `document.querySelector('lma-model').shadowRoot.querySelector('input[type=file]')`;
  await pick(refInput, ['test-vectors/lma-test-1m.gguf']);
  ok(await until(`window.__ev.some((x) => x.startsWith('lma-ready'))`), 'a real pick of the small GGUF vector reaches ready: ' + (await ev('window.__ev.join(", ")')));
  await ev(`document.querySelector('lma-model').reset(); window.__ev = []`);
  await pick(refInput, ['test-vectors/lma-test-split-00001-of-00002.gguf']);
  ok(await until(`window.__ev.some((x) => x === 'lma-error:missing-part')`), 'one part of a split model: lma-error missing-part, "' + (await ev(`document.querySelector('lma-model').shadowRoot.querySelector('.status').textContent`)) + '"');
  await ev(`window.__ev = []`);
  await pick(refInput, ['test-vectors/lma-test-pack/manifest.json', 'test-vectors/lma-test-pack/lm.global.pfw', 'test-vectors/lma-test-pack/ple.rows.pfw', 'test-vectors/lma-test-pack/tokenizer/tokenizer_meta.json']);
  ok(await until(`window.__ev.some((x) => x.startsWith('lma-ready'))`), 'a pocketflux.pack (4 files in one pick) reaches ready');

  // 3. <lma-chat>
  await view(1280, 800);
  ok(await go('/examples/chat-widget/index.html', `customElements.get('lma-chat') && document.querySelector('lma-chat').config && document.querySelector('lma-chat').site`), '<lma-chat> registers, loads its config and the site kit');
  await ev(`window.__ev = []; for (const t of ['lma-chat-open', 'lma-chat-close', 'lma-ready', 'lma-chat-ready', 'lma-message', 'lma-tool', 'lma-error']) document.addEventListener(t, (e) => window.__ev.push(t))`);
  shots.push(await shot('chat-launcher-closed'));
  const C = `document.querySelector('lma-chat').shadowRoot`;
  await ev(`${C}.querySelector('.launch').click(), true`, true); await sleep(400);
  ok(await ev(`!${C}.querySelector('.panel').hidden && ${C}.querySelector('.launch').getAttribute('aria-expanded') === 'true'`), 'the launcher opens the panel (aria-expanded true)');
  shots.push(await shot('chat-open-use-a-model'));
  await ev(`${C}.querySelector('lma-model').preview('checking', { percent: 42 }), true`); shots.push(await shot('chat-checking'));
  await ev(`${C}.querySelector('lma-model').reset(), true`);
  await pick(`${C}.querySelector('lma-model').shadowRoot.querySelector('input[type=file]')`, ['test-vectors/lma-test-1m.gguf']);
  ok(await until(`window.__ev.includes('lma-chat-ready')`), 'the model becomes ready and the stand-in engine loads it: ' + (await ev('window.__ev.join(", ")')));
  await ev(`${C}.querySelector('textarea').value = 'How do refunds work?'; ${C}.querySelector('textarea').dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })), true`);
  ok(await until(`window.__ev.filter((x) => x === 'lma-message').length >= 2`, 20000), 'Enter sends; the stand-in streams a reply naming the model: "' + (await ev(`[...${C}.querySelectorAll('.msg.assistant')].pop().textContent.slice(0, 110)`)) + '"');
  shots.push(await shot('chat-ready-streamed-reply'));
  await ev(`document.querySelector('lma-chat').send('/tool search_docs {"query": "refund"}'), true`);
  ok(await until(`window.__ev.includes('lma-tool') && window.__ev.filter((x) => x === 'lma-message').length >= 4 && /Refunds/.test([...${C}.querySelectorAll('.msg.assistant')].pop().textContent)`, 20000), 'the tool-call path runs the site kit skill: "' + (await ev(`[...${C}.querySelectorAll('.msg.assistant')].pop().textContent.slice(0, 120)`)) + '"');
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 }); await sleep(200);
  ok(await ev(`${C}.querySelector('.panel').hidden && ${C}.activeElement === ${C}.querySelector('.launch') && window.__ev.includes('lma-chat-close')`), 'Escape closes the panel and focus returns to the launcher');
  // phone width + dark mode
  await view(390, 844, true);
  await ev(`${C}.querySelector('.launch').click(), true`, true); await sleep(400);
  ok(await ev(`(() => { const r = ${C}.querySelector('.panel').getBoundingClientRect(); return r.left === 0 && Math.round(r.right) === innerWidth && Math.round(r.bottom) === innerHeight && !${C}.querySelector('.scrim').hidden; })()`), 'phone width: the panel is a bottom sheet (full width, at the bottom) over a scrim');
  ok(await ev(`(() => { const c = ${C}; const t = c.querySelector('textarea'); return parseFloat(getComputedStyle(t).fontSize) >= 16 && [...c.querySelectorAll('.panel *')].every((e) => Math.round(e.getBoundingClientRect().right) <= innerWidth); })()`), 'phone width: the input is 16px or more (no iOS focus zoom) and nothing runs past the right edge');
  shots.push(await shot('chat-phone-bottom-sheet'));
  await dark(true); await sleep(200); shots.push(await shot('chat-phone-dark'));
  if (exc.length) { fails++; console.log('page exceptions:', exc.slice(0, 4)); }
  for (const f of ['lma/ui/lma-model.js', 'lma/ui/lma-chat.js', 'lma/ui/engines/stand-in.js']) console.log(`size  ${f}: ${(fs.statSync(path.join(ROOT, f)).size / 1024).toFixed(1)} KB`);
  console.log('screenshots:\n  ' + shots.join('\n  '));
} finally { try { br.kill(); } catch (e) { /* */ } server.close(); }
console.log(fails ? `FAILED ${fails}` : 'ALL OK'); process.exit(fails ? 1 : 0);
