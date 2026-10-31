// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// Site A's job: check the model file and measure how fast this device reads it (sequential slices, a stream, random tensor reads).
// Its skills are declared in site.config.json; the handlers are bound here.
import { sitePage } from '../sitepage.js';
import { Loader } from '../../../lma/loader.js';
import { mb, quantile } from '../ui.js';

const MiB = 1 << 20;

const skills = {
  async read_speed({ sliceMB = 16 }, ctx) {
    const L = new Loader(ctx.file, ctx.model, { maxSliceMB: sliceMB }), size = ctx.file.size, t0 = performance.now(); let first = null;
    for (let o = 0; o < size; o += sliceMB * MiB) {
      await L.withSlice(o, Math.min(size, o + sliceMB * MiB), () => { if (first == null) first = performance.now() - t0; });
      ctx.bar.set(o / size / 2); ctx.status(`Reading the whole file in ${sliceMB} MB slices… ${mb(o)}`);
    }
    const s1 = (performance.now() - t0) / 1000;
    const t1 = performance.now(), rd = ctx.file.stream().getReader(); let n = 0, chunks = 0;
    for (;;) { const { done, value } = await rd.read(); if (done) break; n += value.byteLength; chunks++; if (chunks % 64 === 0) { ctx.bar.set(0.5 + n / size / 2); ctx.status(`Reading it again as a stream… ${mb(n)}`); } }
    const s2 = (performance.now() - t1) / 1000; ctx.bar.hide();
    return { sliceMB, firstSliceMs: Math.round(first), MBs: Math.round(size / MiB / s1), streamMBs: Math.round(n / MiB / s2), streamChunks: chunks, cpuPeak: L.stats.cpuPeak };
  },
  async random_reads({ count = 48, sizeKB = 1024 }, ctx) {
    const L = new Loader(ctx.file, ctx.model, { maxSliceMB: 16 }), ts = ctx.model.tensors.filter((t) => t.size > 0), times = [];
    let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
    for (let i = 0; i < count; i++) {
      const t = ts[Math.floor(rnd() * ts.length)], n = Math.min(t.size, sizeKB * 1024), o = t.offset + Math.floor(rnd() * (t.size - n + 1));
      const t0 = performance.now(); await L.withSlice(o, o + n, () => {}); times.push(performance.now() - t0);
    }
    return { count, sizeKB, p50: quantile(times, 0.5), p95: quantile(times, 0.95), max: Math.max(...times), cpuPeak: L.stats.cpuPeak };
  },
};

await sitePage({
  section: 'siteA', localSkills: skills,
  async job(ctx, rows, show) {
    const seq = await ctx.site.callTool('read_speed', { sliceMB: 16 }, ctx);
    rows.push(['Read speed', `${seq.MBs} MB/s in 16 MB slices (first slice ${seq.firstSliceMs} ms) · ${seq.streamMBs} MB/s as a stream`]); show();
    ctx.status('Random reads inside tensors…');
    const rnd = await ctx.site.callTool('random_reads', { count: 48, sizeKB: 1024 }, ctx);
    rows.push(['Random 1 MB reads', `p50 ${Math.round(rnd.p50)} ms · p95 ${Math.round(rnd.p95)} ms · max ${Math.round(rnd.max)} ms`]);
    const v = ctx.rec.verify || {};
    ctx.rec.seq = seq; ctx.rec.random = rnd;
    ctx.rec.memory = { cpuPeak: Math.max(seq.cpuPeak, rnd.cpuPeak), hashPeak: v.hash && v.hash.peakBytes };
    rows.push(['Memory peak', `${mb(ctx.rec.memory.cpuPeak)} of file data on the CPU at once` + (v.hash ? ` · ${mb(v.hash.peakBytes)} while hashing` : '')]); show();
  },
});
