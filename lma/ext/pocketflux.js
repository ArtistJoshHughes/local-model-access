// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// pocketflux.js - Local Model Access (LMA), extension: reference decoders for the tensor encodings of pocketflux.pack/1
// (docs/QUANT-q4g32a.md). Plain JavaScript on the CPU, written for clarity and for checking a GPU kernel's output, not for speed:
// real sites dequantise inside their GPU kernels, so the weights stay 4-bit or 8-bit in GPU memory.
//   q4g32a.t64   4-bit, groups of 32 along K, asymmetric (f16 scale + f16 min), 64-row tile records of 1280 bytes
//   q8g64s.t64   8-bit, groups of 64 along K, symmetric (f16 scale), 64-row tile records of 4224 bytes
//   q4g32a.rows  one row after another: K/2 bytes of codes, then K/32 x (f16 scale, f16 min)
//   q8g64s.rows  one row after another: K int8 codes, then K/64 x f16 scale
//   f16, f32, u32  little-endian, row-major
// Every decoder computes the dequantised value in fp32: w = q * s + mn (4-bit) or w = q * s (8-bit).

const F = Math.fround;

const H32 = new Float32Array(1), H32B = new Uint32Array(H32.buffer);

/** IEEE binary16 bits -> number. Widens the bits to binary32 (exponent rebiased 15 -> 127, fraction shifted up 13);
 *  subnormal halves are fraction x 2^-24 */
export function f16(h) {
  const sign = (h & 0x8000) << 16, exp = (h >>> 10) & 0x1f, frac = h & 0x3ff;
  if (exp === 0) return (sign ? -frac : frac) / 16777216;
  H32B[0] = sign | (exp === 0x1f ? 0x7f800000 : (exp + 112) << 23) | (frac << 13);
  return H32[0];
}

const u8 = (b) => (b instanceof Uint8Array ? b : ArrayBuffer.isView(b) ? new Uint8Array(b.buffer, b.byteOffset, b.byteLength) : new Uint8Array(b));
const dv = (b) => { const u = u8(b); return new DataView(u.buffer, u.byteOffset, u.byteLength); };

/** bytes one row takes in a .rows encoding */
export function rowBytes(enc, K) {
  if (enc === 'q4g32a.rows') return K / 2 + (K / 32) * 4;
  if (enc === 'q8g64s.rows') return K + (K / 64) * 2;
  throw new Error('rowBytes: not a .rows encoding: ' + enc);
}

/** bytes a whole tensor takes */
export function tensorBytes(enc, shape) {
  const [N, K] = shape.length === 1 ? [1, shape[0]] : shape;
  switch (enc) {
    case 'q4g32a.t64': return (N * K * 5) / 8;
    case 'q8g64s.t64': return (N / 64) * (K / 64) * 4224;
    case 'q4g32a.rows': case 'q8g64s.rows': return N * rowBytes(enc, K);
    case 'f16': return shape.reduce((a, b) => a * b, 1) * 2;
    case 'f32': case 'u32': return shape.reduce((a, b) => a * b, 1) * 4;
    default: throw new Error('tensorBytes: unknown encoding ' + enc);
  }
}

// The decoders below follow docs/QUANT-q4g32a.md element by element: for weight k of a row, find its group, its code and its group's
// (scale, min), then apply the formula. Clear rather than fast.
const nibble = (b, at, k) => (k % 2 ? b[at + (k - 1) / 2] >> 4 : b[at + k / 2] & 0x0f);   // code k of a packed 4-bit run starting at byte `at`
const int8 = (x) => (x > 127 ? x - 256 : x);
const q4 = (q, s, mn) => F(F(q * s) + mn);

/** one row of a .rows tensor (its rowBytes(enc, K) bytes) -> Float32Array(K) */
export function dequantRow(enc, bytes, K) {
  const b = u8(bytes), d = dv(bytes), out = new Float32Array(K);
  if (enc === 'q4g32a.rows') {   // [K/2 code bytes][K/32 x (f16 scale, f16 min)]
    const scale = [], min = [];
    for (let g = 0; g < K / 32; g++) { scale.push(f16(d.getUint16(K / 2 + 4 * g, true))); min.push(f16(d.getUint16(K / 2 + 4 * g + 2, true))); }
    for (let k = 0; k < K; k++) out[k] = q4(nibble(b, 0, k), scale[Math.floor(k / 32)], min[Math.floor(k / 32)]);
  } else if (enc === 'q8g64s.rows') {   // [K int8 codes][K/64 x f16 scale]
    for (let k = 0; k < K; k++) out[k] = F(int8(b[k]) * f16(d.getUint16(K + 2 * Math.floor(k / 64), true)));
  } else throw new Error('dequantRow: not a .rows encoding: ' + enc);
  return out;
}

/** one 64-row tile of a .t64 tensor: the tile's G consecutive records (t * G .. t * G + G - 1) -> Float32Array(64 * K), row-major */
export function dequantTile(enc, bytes, K) {
  const b = u8(bytes), d = dv(bytes), out = new Float32Array(64 * K);
  if (enc === 'q4g32a.t64') {   // record g (1280 B): row r's 16 code bytes at 16r, then row r's (scale | min << 16) word at 1024 + 4r
    for (let r = 0; r < 64; r++) {
      for (let k = 0; k < K; k++) {
        const rec = 1280 * Math.floor(k / 32), word = d.getUint32(rec + 1024 + 4 * r, true);
        out[r * K + k] = q4(nibble(b, rec + 16 * r, k % 32), f16(word & 0xffff), f16(word >>> 16));
      }
    }
  } else if (enc === 'q8g64s.t64') {   // record g (4224 B): row r's 64 int8 codes at 64r, then row r's f16 scale at 4096 + 2r
    for (let r = 0; r < 64; r++) {
      for (let k = 0; k < K; k++) {
        const rec = 4224 * Math.floor(k / 64);
        out[r * K + k] = F(int8(b[rec + 64 * r + (k % 64)]) * f16(d.getUint16(rec + 4096 + 2 * r, true)));
      }
    }
  } else throw new Error('dequantTile: not a .t64 encoding: ' + enc);
  return out;
}

/** a whole tensor's bytes -> Float32Array (u32 -> Uint32Array). For small tensors and tests: big ones go through a GPU kernel. */
export function dequant(enc, bytes, shape) {
  const [N, K] = shape.length === 1 ? [1, shape[0]] : shape, d = dv(bytes);
  if (enc === 'f32') { const o = new Float32Array(N * K); for (let i = 0; i < o.length; i++) o[i] = d.getFloat32(4 * i, true); return o; }
  if (enc === 'f16') { const o = new Float32Array(N * K); for (let i = 0; i < o.length; i++) o[i] = f16(d.getUint16(2 * i, true)); return o; }
  if (enc === 'u32') { const o = new Uint32Array(N * K); for (let i = 0; i < o.length; i++) o[i] = d.getUint32(4 * i, true); return o; }
  if (enc.endsWith('.rows')) { const rb = rowBytes(enc, K), o = new Float32Array(N * K), b = u8(bytes); for (let n = 0; n < N; n++) o.set(dequantRow(enc, b.subarray(n * rb, (n + 1) * rb), K), n * K); return o; }
  const tb = tensorBytes(enc, [64, K]), o = new Float32Array(N * K), b = u8(bytes);
  for (let t = 0; t < N / 64; t++) o.set(dequantTile(enc, b.subarray(t * tb, (t + 1) * tb), K), t * 64 * K);
  return o;
}

/** one row n of a pack tensor, read through an LMA Loader with no more than one tile (or one row) in memory -> Float32Array(K) */
export async function readRow(loader, tensor, n) {
  const t = loader.tensor(tensor), [N, K] = t.shape.length === 1 ? [1, t.shape[0]] : t.shape;
  if (!(n >= 0 && n < N)) throw new RangeError(`row ${n} of ${t.name}`);
  if (t.enc.endsWith('.rows')) { const rb = rowBytes(t.enc, K); return loader.withSlice(t.offset + n * rb, t.offset + (n + 1) * rb, (ab) => dequantRow(t.enc, ab, K), t.part); }
  if (t.enc.endsWith('.t64')) {
    const tb = tensorBytes(t.enc, [64, K]), tile = Math.floor(n / 64);
    return loader.withSlice(t.offset + tile * tb, t.offset + (tile + 1) * tb, (ab) => dequantTile(t.enc, ab, K).slice((n % 64) * K, (n % 64 + 1) * K), t.part);
  }
  const eb = t.enc === 'f16' ? 2 : 4;
  return loader.withSlice(t.offset + n * K * eb, t.offset + (n + 1) * K * eb, (ab) => dequant(t.enc, ab, [K]), t.part);
}
