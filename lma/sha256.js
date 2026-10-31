// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// sha256.js - Local Model Access (LMA): incremental SHA-256 in plain JavaScript. WebCrypto can only hash a whole buffer at once, and a model file must never be
// read into memory whole, so the file is fed through update() a slice at a time. Works in pages, workers and Node.
//   const h = new Sha256(); h.update(bytes); ...; h.hex()

const K = new Int32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);

// compress every whole 64-byte block of d[p, end) into H; returns the position after the last whole block
function blocks(H, W, d, p, end) {
  let h0 = H[0], h1 = H[1], h2 = H[2], h3 = H[3], h4 = H[4], h5 = H[5], h6 = H[6], h7 = H[7];
  while (p + 64 <= end) {
    for (let i = 0; i < 16; i++, p += 4) W[i] = (d[p] << 24) | (d[p + 1] << 16) | (d[p + 2] << 8) | d[p + 3];
    for (let i = 16; i < 64; i++) {
      const a = W[i - 15], b = W[i - 2];
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      W[i] = (W[i - 16] + s0 + W[i - 7] + s1) | 0;
    }
    let a = h0, b = h1, c = h2, e3 = h3, e = h4, f = h5, g = h6, h = h7;
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const t1 = (h + S1 + ((e & f) ^ (~e & g)) + K[i] + W[i]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const t2 = (S0 + ((a & b) ^ (a & c) ^ (b & c))) | 0;
      h = g; g = f; f = e; e = (e3 + t1) | 0; e3 = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + e3) | 0;
    h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + h) | 0;
  }
  H[0] = h0; H[1] = h1; H[2] = h2; H[3] = h3; H[4] = h4; H[5] = h5; H[6] = h6; H[7] = h7;
  return p;
}

export class Sha256 {
  constructor() {
    this.H = new Int32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    this.W = new Int32Array(64); this.tail = new Uint8Array(64); this.tailLen = 0; this.bytes = 0; this.done = null;
  }

  /** add bytes: an ArrayBuffer, a typed array or a DataView */
  update(data) {
    if (this.done) throw new Error('sha256: update after digest');
    const d = data instanceof Uint8Array ? data : ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
    let p = 0; const n = d.length; this.bytes += n;
    if (this.tailLen) {   // finish the partial block left by the last update
      const take = Math.min(64 - this.tailLen, n);
      this.tail.set(d.subarray(0, take), this.tailLen); this.tailLen += take; p = take;
      if (this.tailLen < 64) return this;
      blocks(this.H, this.W, this.tail, 0, 64); this.tailLen = 0;
    }
    p = blocks(this.H, this.W, d, p, n);
    if (p < n) { this.tail.set(d.subarray(p), 0); this.tailLen = n - p; }
    return this;
  }

  /** the 32-byte digest (the hash is finished after this) */
  digest() {
    if (this.done) return this.done.slice();
    const bits = this.bytes * 8, pad = new Uint8Array(this.tailLen < 56 ? 64 : 128);
    pad.set(this.tail.subarray(0, this.tailLen)); pad[this.tailLen] = 0x80;
    const dv = new DataView(pad.buffer);
    dv.setUint32(pad.length - 8, Math.floor(bits / 0x100000000)); dv.setUint32(pad.length - 4, bits >>> 0);
    blocks(this.H, this.W, pad, 0, pad.length);
    const out = new Uint8Array(32), ov = new DataView(out.buffer);
    for (let i = 0; i < 8; i++) ov.setInt32(i * 4, this.H[i]);
    this.done = out; return out.slice();
  }

  hex() { return Array.from(this.digest(), (b) => b.toString(16).padStart(2, '0')).join(''); }
}

/** sha256 of one in-memory buffer, as lowercase hex */
export function sha256hex(data) { return new Sha256().update(data).hex(); }
