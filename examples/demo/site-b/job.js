// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// Site B's job: stream the model's weight tensors from the picked file straight into GPU buffers, one tensor at a time, and run a
// matrix-vector product over every layer on the configured kernel backend; then time the same kernel over weights kept in GPU memory.
// The test model stores the exact answers in its metadata, so the page proves the GPU read every byte. Skills: site.config.json.
import { sitePage } from '../sitepage.js';
import { Loader } from '../../../lma/loader.js';
import { mb } from '../ui.js';

let backend = null;
const meta = (m, k) => (m.format === 'gguf' ? m.metadata[k] : JSON.parse(m.metadata[k]));   // safetensors metadata values are strings
const r2 = (x) => Math.round(x * 100) / 100;

function plan(model) {
  const a = model.architecture;
  if (a !== 'lma-test-gemv') throw new Error(`this job runs the LMA test vectors (lma-test-gemv), not "${a}"`);
  return { L: meta(model, a + '.block_count'), rows: meta(model, a + '.rows'), cols: meta(model, a + '.cols'), sum: meta(model, a + '.check.sum'), wsum: meta(model, a + '.check.wsum') };
}
const exact = (acc, l, g) => { let s = 0, w = 0; for (let r = 0; r < acc.length; r++) { s += acc[r]; w += acc[r] * ((r % 7) + 1); } return s === g.sum[l] && w === g.wsum[l]; };

const skills = {
  async gemv_stream({ ring = 2 }, ctx) {
    const g = plan(ctx.model), B = ctx.backend, Ld = new Loader(ctx.file, ctx.model, { maxSliceMB: ctx.site.maxSliceMB }), { rows, cols } = g;
    const desc = { weight: Ld.tensor('blk.0.weight'), scale: Ld.tensor('blk.0.scale'), x: Ld.tensor('input.x') };
    if (!B.supports('gemv', desc)) throw new Error(`the ${B.name} backend has no gemv for ${desc.weight.dtype} weights`);
    const own = [], alloc = (n, l) => { const hd = B.alloc(n, l); own.push(hd); return hd; };
    try {
      const x = await Ld.toGpu('input.x', B, alloc(cols * 4, 'x'));
      const W = Array.from({ length: ring }, (_, i) => alloc(rows * cols, 'w' + i)), S = Array.from({ length: ring }, (_, i) => alloc(rows * 4, 's' + i));
      const acc = [], y = []; for (let l = 0; l < g.L; l++) { acc.push(alloc(rows * 4, 'acc' + l)); y.push(alloc(rows * 4, 'y' + l)); }
      let bytes = 0; const t0 = performance.now();
      for (let l = 0; l < g.L; l++) {
        const k = l % ring, wt = Ld.tensor(`blk.${l}.weight`), st = Ld.tensor(`blk.${l}.scale`);
        await Ld.toGpu(wt, B, W[k]); await Ld.toGpu(st, B, S[k]); bytes += wt.size + st.size;
        B.ops.gemv({ weight: W[k], scale: S[k], x, acc: acc[l], y: y[l], rows, cols });
        if (k === ring - 1) await B.sync();   // keep the GPU at most one ring behind the reads
        ctx.bar.set((l + 1) / g.L); ctx.status(`Streaming layer ${l + 1} of ${g.L} into the GPU and multiplying…`);
      }
      await B.sync();
      const s = (performance.now() - t0) / 1000;
      let ok = 0, yOk = true;
      for (let l = 0; l < g.L; l++) {
        const a = new Int32Array(await B.read(acc[l], rows * 4)); if (exact(a, l, g)) ok++;
        if (l === 0 || l === g.L - 1) {   // the scaled output too: y = acc * scale in f32
          const yv = new Float32Array(await B.read(y[l], rows * 4));
          await Ld.withTensor(`blk.${l}.scale`, (ab) => { const sc = new Float32Array(ab); for (let r = 0; r < rows; r++) { const want = Math.fround(a[r] * sc[r]); if (Math.abs(yv[r] - want) > 1e-6 * Math.abs(want) + 1e-30) yOk = false; } });
        }
      }
      ctx.bar.hide();
      return { layers: g.L, bytes, s, GBs: r2(bytes / 1e9 / s), exact: ok, yOk, cpuPeak: Ld.stats.cpuPeak, gpuPeakDuring: B.stats.gpuPeak };
    } finally { own.forEach((hd) => B.free(hd)); }
  },

  async gemv_resident({ layers = 8, minMs = 400 }, ctx) {
    const g = plan(ctx.model), B = ctx.backend, Ld = new Loader(ctx.file, ctx.model, { maxSliceMB: ctx.site.maxSliceMB }), { rows, cols } = g, R = Math.min(layers, g.L);
    const own = [], alloc = (n, l) => { const hd = B.alloc(n, l); own.push(hd); return hd; };
    try {
      const x = await Ld.toGpu('input.x', B, alloc(cols * 4, 'x')), W = [], S = [], A = [], Y = [];
      for (let l = 0; l < R; l++) {
        W.push(await Ld.toGpu(`blk.${l}.weight`, B, alloc(rows * cols, 'w' + l))); S.push(await Ld.toGpu(`blk.${l}.scale`, B, alloc(rows * 4, 's' + l)));
        A.push(alloc(rows * 4, 'a' + l)); Y.push(alloc(rows * 4, 'y' + l));
      }
      const pass = (n) => { for (let i = 0; i < n; i++) for (let l = 0; l < R; l++) B.ops.gemv({ weight: W[l], scale: S[l], x, acc: A[l], y: Y[l], rows, cols }); };
      pass(1); await B.sync();   // warm up
      let iters = 1, ms = 0;
      for (;;) { ctx.status(`Timing the kernel on weights kept in GPU memory (${R} layers x ${iters})…`); const t0 = performance.now(); pass(iters); await B.sync(); ms = performance.now() - t0; if (ms >= minMs || iters >= 4096) break; iters *= 2; }
      let ok = 0; for (let l = 0; l < R; l++) if (exact(new Int32Array(await B.read(A[l], rows * 4)), l, g)) ok++;
      return { layers: R, iters, ms: Math.round(ms), GBs: r2((R * rows * cols * iters) / 1e9 / (ms / 1000)), exact: ok, gpuPeak: B.stats.gpuPeak };
    } finally { own.forEach((hd) => B.free(hd)); }
  },
};

await sitePage({
  section: 'siteB', localSkills: skills,
  async job(ctx, rows, show) {
    const card = await ctx.site.callTool('job_card', { model: ctx.entry.name || ctx.entry.file }, ctx);   // an API skill: the site's back end says how to run
    ctx.status('Starting the GPU…');
    backend = backend || (await ctx.site.loadKernels()); ctx.backend = backend;
    const ad = backend.info.adapter || {};
    rows.push(['Kernel backend', `${backend.name} · ${[ad.vendor, ad.architecture].filter(Boolean).join(' ') || 'GPU'}`]); show();
    try {
      const s = await ctx.site.callTool('gemv_stream', { ring: card.ring || 2 }, ctx); ctx.rec.gemv = s;
      rows.push(['Streamed into the GPU', `${s.layers} layers, ${mb(s.bytes)} in ${s.s.toFixed(2)} s = ${s.GBs} GB/s · answers ${s.exact}/${s.layers} exact · scaled output ${s.yOk ? 'ok' : 'WRONG'}`]); show();
    } catch (e) { ctx.rec.gemv = { error: String(e.message || e).slice(0, 200) }; rows.push(['Streamed into the GPU', 'ERROR ' + ctx.rec.gemv.error]); show(); }
    try {
      const r = await ctx.site.callTool('gemv_resident', { layers: card.residentLayers || 8, minMs: card.minMs || 400 }, ctx); ctx.rec.resident = r;
      rows.push(['Kernel on resident weights', `${r.layers} layers x ${r.iters} in ${r.ms} ms = ${r.GBs} GB/s · ${r.exact}/${r.layers} exact`]); show();
    } catch (e) { ctx.rec.resident = { error: String(e.message || e).slice(0, 200) }; rows.push(['Kernel on resident weights', 'ERROR ' + ctx.rec.resident.error]); show(); }
    const v = ctx.rec.verify || {};
    ctx.rec.memory = { cpuPeak: (ctx.rec.gemv && ctx.rec.gemv.cpuPeak) || 0, gpuPeak: backend.stats.gpuPeak, hashPeak: v.hash && v.hash.peakBytes };
    rows.push(['Memory peak', `${mb(ctx.rec.memory.cpuPeak)} of file data on the CPU at once · ${mb(ctx.rec.memory.gpuPeak)} of GPU memory` + (v.hash ? ` · ${mb(v.hash.peakBytes)} while hashing` : '')]); show();
  },
});
