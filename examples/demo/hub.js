// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
import * as R from './report.js';
import { $ } from './ui.js';

const report = R.mount($('#report'));
const e = await R.env(); R.put('env', e); report.refresh();
const yn = (b) => (b ? 'yes' : 'no');
$('#env').textContent = `${e.standalone ? 'Home Screen app' : 'Browser tab'} · iOS ${e.ios || '?'} · file storage ${yn(e.opfs)} · share files ${yn(e.shareFiles)} · WebGPU ${e.webgpu ? e.gpu || 'yes' : 'no'} · secure page ${yn(e.secure)}`;
if (e.standalone) {
  const w = $('#warn'); w.classList.remove('hidden');
  w.textContent = "You are in a Home Screen app: for step 1 use A, “Download into this page”, then Save to Files. Skip B (it needs a Safari tab).";
}
