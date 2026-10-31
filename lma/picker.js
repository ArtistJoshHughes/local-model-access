// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// picker.js - Local Model Access (LMA): the user picks the model file from the Files app (one tap per visit: a site never keeps a handle or a copy).
// The button is a real <input type=file> under a label, so the tap opens the picker on every browser. No `accept` filter: iOS greys out
// file types it does not know (.gguf, .safetensors), so the page checks the header instead.
//   makePicker(hostElement, { label, onPick(file, info) })   info: { tapToFileMs, closedToFileMs, cancelled }
const now = () => performance.now();

export function makePicker(host, { label = 'Choose the model file', onPick, multiple = false } = {}) {
  const lab = document.createElement('label'); lab.className = 'lma-pick';
  const text = document.createElement('span'); text.textContent = label; lab.appendChild(text);
  const kept = [];   // used inputs stay referenced, so the File objects they handed out stay readable
  let inp = null, tTap = 0, events = [];
  const onEv = (e) => events.push({ t: now(), type: e.type + (e.type === 'visibilitychange' ? ':' + document.visibilityState : '') });
  const listen = (on) => {
    for (const [tgt, ev] of [[window, 'focus'], [window, 'pageshow'], [document, 'visibilitychange']]) (on ? tgt.addEventListener : tgt.removeEventListener).call(tgt, ev, onEv);
  };
  const fresh = () => {
    if (inp) { kept.push(inp); inp.remove(); }
    inp = document.createElement('input'); inp.type = 'file'; inp.multiple = multiple;
    Object.assign(inp.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', opacity: '0', cursor: 'pointer', fontSize: '0' });
    inp.addEventListener('click', () => { tTap = now(); events = []; listen(true); });
    inp.addEventListener('cancel', () => { listen(false); onPick && onPick(null, { cancelled: true, tapToFileMs: Math.round(now() - tTap) }); fresh(); });
    inp.addEventListener('change', () => {
      const t = now(), files = Array.from(inp.files || []); listen(false);
      const back = events.find((x) => x.t > tTap + 250);   // the first focus / visibility event after the picker opened = it closed
      const info = { cancelled: !files.length, tapToFileMs: Math.round(t - tTap), closedToFileMs: back ? Math.round(t - back.t) : null, closedEvent: back ? back.type : null };
      fresh();
      if (onPick) onPick(multiple ? files : files[0] || null, info);
    });
    lab.appendChild(inp);
  };
  fresh(); host.appendChild(lab);
  return {
    el: lab,
    setLabel(t) { text.textContent = t; },
    setDisabled(d) { lab.classList.toggle('disabled', !!d); inp.disabled = !!d; },
  };
}

/** this site's own storage use: estimate + persisted (to show the model was not copied into it) */
export async function storageSnapshot() {
  const o = { t: Date.now() };
  try { const e = await navigator.storage.estimate(); o.usage = e.usage; o.quota = e.quota; } catch (e) { o.error = 'no storage estimate'; }
  try { o.persisted = await navigator.storage.persisted(); } catch (e) { /* */ }
  return o;
}

export function describeFile(f) { return f ? { name: f.name, size: f.size, type: f.type || '', lastModified: f.lastModified, lastModifiedAgoS: Math.round((Date.now() - f.lastModified) / 1000) } : null; }
