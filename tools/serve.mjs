// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// serve.mjs - a static server for trying the demo locally, with what a model host needs: HTTP Range (resumable downloads), ETag /
// If-Range, model files as application/octet-stream + Content-Disposition: attachment. Node 18+, no dependencies.
//   node tools/serve.mjs [DIR=.] [PORT=8000] [HOST=127.0.0.1]   then open http://localhost:8000/demo/ (localhost counts as a secure page)
// It answers only this computer unless you give a HOST (0.0.0.0 for every network interface). It never serves dot-files (.git, .env).
// For a phone on your network you need HTTPS: put this behind any HTTPS reverse proxy, or use a static host.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const DIR = path.resolve(process.argv[2] || '.'), PORT = +(process.argv[3] || 8000), HOST = process.argv[4] || '127.0.0.1';
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.md': 'text/plain; charset=utf-8' };
const MODEL = /\.(gguf|safetensors|bin)$/i;

/** Content-Disposition for any file name: a plain ASCII fallback plus the exact name (RFC 6266 / 5987) */
function attachment(name) {
  const ascii = name.replace(/[^\x20-\x7e]|["\\]/g, '_');
  return `attachment; filename="${ascii}"; filename*=UTF-8''${encodeURIComponent(name)}`;
}

function send(s, f, range) {
  const r = fs.createReadStream(f, range);
  r.on('error', () => { if (!s.headersSent) s.writeHead(500); s.destroy(); });
  r.pipe(s);
}

http.createServer((q, s) => {
  let rel; try { rel = decodeURIComponent(new URL(q.url, 'http://x').pathname); } catch (e) { s.writeHead(400); return s.end(); }
  if (rel.split(/[\\/]/).some((p) => p.startsWith('.'))) { s.writeHead(404); return s.end('not found'); }
  let f = path.join(DIR, rel);
  if (!(f === DIR || f.startsWith(DIR + path.sep))) { s.writeHead(403); return s.end(); }
  try {
    if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, 'index.html');
    if (!fs.existsSync(f) || !fs.statSync(f).isFile()) { s.writeHead(404); return s.end('not found'); }
    const st = fs.statSync(f), etag = `"${st.size.toString(16)}-${Math.floor(st.mtimeMs).toString(16)}"`;
    const head = { 'Content-Type': TYPES[path.extname(f).toLowerCase()] || 'application/octet-stream', 'Accept-Ranges': 'bytes', ETag: etag, 'Cache-Control': 'no-cache' };
    if (MODEL.test(f)) head['Content-Disposition'] = attachment(path.basename(f));
    const m = /^bytes=(\d*)-(\d*)$/.exec(q.headers.range || '');
    if (m && (!q.headers['if-range'] || q.headers['if-range'] === etag) && (m[1] || m[2])) {
      let a = m[1] ? +m[1] : st.size - +m[2], b = m[1] && m[2] ? +m[2] : st.size - 1;
      if (a < 0) a = 0; if (b >= st.size) b = st.size - 1;
      if (a > b) { s.writeHead(416, { 'Content-Range': `bytes */${st.size}` }); return s.end(); }
      s.writeHead(206, { ...head, 'Content-Range': `bytes ${a}-${b}/${st.size}`, 'Content-Length': b - a + 1 });
      return q.method === 'HEAD' ? s.end() : send(s, f, { start: a, end: b });
    }
    s.writeHead(200, { ...head, 'Content-Length': st.size });
    return q.method === 'HEAD' ? s.end() : send(s, f);
  } catch (e) { if (!s.headersSent) s.writeHead(500); s.end(); }
}).listen(PORT, HOST, () => console.log(`serving ${DIR} at http://${HOST === '127.0.0.1' ? 'localhost' : HOST}:${PORT}/`));
