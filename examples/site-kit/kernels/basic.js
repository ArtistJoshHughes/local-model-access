// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// basic.js - a small built-in WebGPU backend (the interface is documented in stub.js). It has one op, an int8 matrix-vector product
// with a per-row f32 scale, which is what the demo's test model holds. It is a reference to build on, not a fast general kernel set.
//   "kernels": { "backend": "basic" }

const WGSL = /* wgsl */ `
struct P { rows: u32, cols16: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;       // int8 weights, row-major, 16 per vec4
@group(0) @binding(1) var<storage, read> S: array<f32>;             // per-row scale
@group(0) @binding(2) var<storage, read> X: array<vec4<i32>>;       // int32 input
@group(0) @binding(3) var<storage, read_write> ACC: array<i32>;     // exact sums
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;       // scaled output
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> part: array<i32, 64>;

fn dot_i8x4(w: u32, x: vec4<i32>) -> i32 {
  let s = bitcast<i32>(w);   // extractBits on an i32 sign-extends each byte
  return dot(vec4<i32>(extractBits(s, 0u, 8u), extractBits(s, 8u, 8u), extractBits(s, 16u, 8u), extractBits(s, 24u, 8u)), x);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.rows) { return; }
  let base = row * p.cols16;
  var acc = 0i;
  for (var j = li; j < p.cols16; j = j + 64u) {
    let w = W[base + j];
    let xb = j * 4u;
    acc = acc + dot_i8x4(w.x, X[xb]) + dot_i8x4(w.y, X[xb + 1u]) + dot_i8x4(w.z, X[xb + 2u]) + dot_i8x4(w.w, X[xb + 3u]);
  }
  part[li] = acc;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { part[li] = part[li] + part[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { ACC[row] = part[0]; Y[row] = f32(part[0]) * S[row]; }
}`;

export default async function createBackend(options = {}) {
  if (!globalThis.navigator || !navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: options.powerPreference || 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
  const ai = adapter.info || {};
  const info = { adapter: { vendor: ai.vendor || '', architecture: ai.architecture || '', device: ai.device || '', description: ai.description || '' },
    limits: { maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize }, features: Array.from(device.features || []), lost: null };
  device.lost.then((i) => { info.lost = (i && i.message) || 'lost'; });
  const U = GPUBufferUsage, stats = { gpuNow: 0, gpuPeak: 0, dispatches: 0, submits: 0 };
  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: 'main' } });
  let nextId = 1, pending = [];
  const live = new Map(), groups = new Map(), params = new Map();

  const alloc = (bytes, label = '') => {
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    device.pushErrorScope('out-of-memory');
    const buffer = device.createBuffer({ size, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC, label });
    device.popErrorScope().then((e) => { if (e) info.oom = (info.oom || 0) + 1; });
    const h = { id: nextId++, buffer, size, label }; live.set(h.id, h);
    stats.gpuNow += size; if (stats.gpuNow > stats.gpuPeak) stats.gpuPeak = stats.gpuNow;
    return h;
  };
  const flush = () => {
    if (!pending.length) return;
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    for (const d of pending) { pass.setBindGroup(0, d.bg); pass.dispatchWorkgroups(d.x, d.y); }
    pass.end(); device.queue.submit([enc.finish()]); stats.submits++; pending = [];
  };
  const uniform = (rows, cols16) => {
    const k = rows + ':' + cols16; let b = params.get(k);
    if (!b) { b = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array([rows, cols16, 0, 0])); params.set(k, b); }
    return b;
  };
  const backend = {
    name: 'basic', info, stats, device,
    supports(op, d) {
      return op === 'gemv' && !!d && d.weight && d.weight.dtype === 'i8' && (!d.scale || d.scale.dtype === 'f32') && (!d.x || d.x.dtype === 'i32')
        && d.weight.shape.length === 2 && d.weight.shape[1] % 16 === 0;
    },
    alloc,
    write(h, data, byteOffset = 0) {
      flush();   // queued kernels run before this write lands (writeBuffer is ordered on the queue)
      const u = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const n = u.byteLength - (u.byteLength % 4);
      if (n) device.queue.writeBuffer(h.buffer, byteOffset, u.buffer, u.byteOffset, n);
      if (n < u.byteLength) { const t = new Uint8Array(4); t.set(u.subarray(n)); device.queue.writeBuffer(h.buffer, byteOffset + n, t); }
    },
    ops: {
      gemv({ weight, scale, x, acc, y, rows, cols }) {
        if (cols % 16) throw new Error('basic gemv: cols must be a multiple of 16');
        const accH = acc || y, key = [weight.id, scale.id, x.id, accH.id, y.id, rows, cols].join(':');
        let bg = groups.get(key);
        if (!bg) {
          bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [weight, scale, x, accH, y].map((h, i) => ({ binding: i, resource: { buffer: h.buffer } }))
            .concat([{ binding: 5, resource: { buffer: uniform(rows, cols / 16) } }]) });
          groups.set(key, bg);
        }
        pending.push({ bg, x: Math.min(rows, 65535), y: Math.ceil(rows / 65535) }); stats.dispatches++;
        if (pending.length >= 64) flush();
      },
    },
    async sync() { flush(); await device.queue.onSubmittedWorkDone(); if (info.lost) throw new Error('GPU device lost: ' + info.lost); },
    async read(h, bytes = h.size) {
      flush();
      const n = Math.ceil(bytes / 4) * 4, rb = device.createBuffer({ size: n, usage: U.MAP_READ | U.COPY_DST });
      const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(h.buffer, 0, rb, 0, n); device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ); const out = rb.getMappedRange().slice(0, bytes); rb.unmap(); rb.destroy();
      return out;
    },
    free(h) {
      if (!h || !live.has(h.id)) return;
      flush(); live.delete(h.id); stats.gpuNow -= h.size; h.buffer.destroy();
      for (const k of Array.from(groups.keys())) if (k.split(':').slice(0, 5).includes(String(h.id))) groups.delete(k);
    },
    destroy() { flush(); for (const h of Array.from(live.values())) backend.free(h); for (const b of params.values()) b.destroy(); params.clear(); device.destroy(); },
  };
  return backend;
}
