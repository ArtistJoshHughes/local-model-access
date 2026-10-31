// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// stand-in.js - Local Model Access (LMA): a STAND-IN engine for <lma-chat>. It does not run the model. It proves the wiring: load()
// reads the picked model's header through the core, and generate() streams a short canned reply that names the model, so the widget,
// its events and the tool-call path can be tried without a runtime. Replace it with a real engine that implements the same slot:
//   { async load({ files, model, entry, onProgress }), async *generate({ messages, system, tools, signal, sampling }), unload() }
// Tool-call path: a message starting with "/tool NAME {json}" yields { tool_call: { name, arguments } }; the reply after it quotes the result.
import { readModel } from '../../formats.js';

const sleep = (ms, signal) => new Promise((res, rej) => {
  const t = setTimeout(res, ms);
  if (signal) signal.addEventListener('abort', () => { clearTimeout(t); rej(Object.assign(new Error('aborted'), { name: 'AbortError' })); }, { once: true });
});
const size = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : (n / 1e6).toFixed(1) + ' MB');

export function createStandIn({ delayMs = 18 } = {}) {
  let info = null;
  return {
    name: 'stand-in',
    async load({ files, model, entry, onProgress }) {
      const m = model || (await readModel(files));   // the header only: no weights are read
      const bytes = (m.files || files).reduce((n, f) => n + f.size, 0);
      info = { name: (entry && (entry.name || entry.file)) || m.name || 'the model', format: m.format, tensors: m.tensors.length, bytes, architecture: m.architecture };
      if (onProgress) onProgress({ done: 1, total: 1 });
    },
    async *generate({ messages, tools, signal }) {
      if (!info) throw new Error('stand-in: load() first');
      const last = messages[messages.length - 1] || {}, words = [];
      const m = last.role === 'user' && /^\/tool\s+([\w-]+)\s*(\{.*\})?\s*$/s.exec(last.content || '');
      if (m) {
        if (!tools || !tools.some((t) => (t.function || t).name === m[1])) { words.push(`(stand-in) This site has no skill called ${m[1]}.`); }
        else { yield { tool_call: { name: m[1], arguments: m[2] ? JSON.parse(m[2]) : {} } }; return; }
      } else if (last.role === 'tool') {
        words.push(`(stand-in) The skill ${last.name} returned: ${String(last.content).slice(0, 300)}`);
      } else {
        words.push(`This is the stand-in engine, not a real model, so it can't answer yet.`,
          `Your model file checked out: ${info.name} (${info.format}, ${info.tensors} tensors, ${size(info.bytes)}), read in place on this device.`,
          `A real engine plugged into this widget would answer: “${String(last.content || '').slice(0, 120)}”.`);
      }
      for (const w of words.join(' ').split(/(?<= )/)) { await sleep(delayMs, signal); yield { text: w }; }
    },
    unload() { info = null; },
  };
}

export default createStandIn;
