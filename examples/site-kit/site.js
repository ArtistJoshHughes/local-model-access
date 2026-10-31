// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// site.js - one site's config (site.config.json): its skills (tool definitions) and the APIs they call, its pre-prompt, the models it
// approves, and the kernel backend it runs them on. A site fills in the config and binds its local skills; the kit does the rest.
//   const site = await loadSite('./site.config.json');
//   site.bind({ summarise: async (args, ctx) => ... });     // skills with handler { "type": "local", "fn": "summarise" }
//   site.tools()                                          // the skills as tool definitions for a model runner (OpenAI-style)
//   await site.callTool('weather', { city: 'Leeds' })     // runs a local skill or calls its API
//   const backend = await site.loadKernels();             // the configured kernel backend (kernels/*.js here, or the site's own module)

const now = () => performance.now();
const BUILTIN = { basic: './kernels/basic.js', stub: './kernels/stub.js', 'p0pc0rn-kernels': './kernels/p0pc0rn-kernels.js' };
const BACKEND_API = ['alloc', 'write', 'read', 'sync', 'free', 'destroy', 'supports'];

export async function loadSite(url = './site.config.json') {
  const base = new URL(url, globalThis.location ? location.href : undefined);
  const r = await fetch(base, { cache: 'no-store' });
  if (!r.ok) throw new Error(`site config: HTTP ${r.status} for ${base.pathname}`);
  return new Site(await r.json(), base);
}

/** problems with a config (empty = fine) */
export function checkConfig(c) {
  const p = [], isObj = (x) => x && typeof x === 'object' && !Array.isArray(x);
  if (!isObj(c)) return ['the config is not a JSON object'];
  if (!isObj(c.site) || !c.site.name) p.push('site.name is missing');
  if (c.prompt && typeof c.prompt.system !== 'string') p.push('prompt.system must be a string');
  const names = new Set();
  for (const s of c.skills || []) {
    if (!s.name || !/^[A-Za-z0-9_-]{1,64}$/.test(s.name)) p.push(`skill name "${s.name}" must be 1-64 letters, digits, _ or -`);
    if (names.has(s.name)) p.push(`skill ${s.name} is defined twice`); names.add(s.name);
    const h = s.handler || {};
    if (h.type === 'http' && !(c.apis && c.apis[h.api])) p.push(`skill ${s.name} calls API "${h.api}", which apis does not define`);
    if (h.type !== 'http' && h.type !== 'local') p.push(`skill ${s.name}: handler.type must be "local" or "http"`);
  }
  for (const m of c.models || []) {   // one file ({ file, sha256, size }) or several ({ files: [...] }); any other field is the site's own
    const label = m.name || m.file || m.repo || '?';
    for (const f of Array.isArray(m.files) ? m.files : m.manifest ? [m.manifest] : [m]) {   // a pack is pinned by its manifest
      if (!/^[0-9a-f]{64}$/.test((f && f.sha256) || '')) p.push(`model ${label}: every file needs a sha256 of 64 lowercase hex characters`);
      if (!f || !Number.isSafeInteger(f.size) || f.size <= 0) p.push(`model ${label}: every file needs its size in bytes`);
    }
  }
  if (c.kernels && c.kernels.backend === 'custom' && !c.kernels.module) p.push('kernels.backend "custom" needs kernels.module');
  return p;
}

export class Site {
  constructor(config, base) {
    const p = checkConfig(config);
    if (p.length) throw new Error('site config: ' + p.join('; '));
    this.config = config; this.base = base; this.handlers = {}; this.calls = [];
  }

  get name() { return this.config.site.name; }
  get job() { return this.config.site.job || ''; }
  get id() { return this.config.site.id || this.config.site.name; }
  get systemPrompt() { return (this.config.prompt && this.config.prompt.system) || ''; }
  get models() { return this.config.models || []; }
  get maxSliceMB() { return (this.config.memory && this.config.memory.maxSliceMB) || 16; }

  bind(handlers) { Object.assign(this.handlers, handlers); return this; }
  skill(name) { return (this.config.skills || []).find((s) => s.name === name) || null; }

  /** the skills as tool definitions, the shape most model runners take ({ type: 'function', function: { name, description, parameters } }) */
  tools() {
    return (this.config.skills || []).map((s) => ({ type: 'function', function: { name: s.name, description: s.description || '', parameters: s.parameters || { type: 'object', properties: {} } } }));
  }

  /** run one skill: a local handler (bound with bind) or an HTTP API. ctx is passed to local handlers (the file, the model, a progress callback) */
  async callTool(name, args = {}, ctx = {}) {
    const s = this.skill(name); if (!s) throw new Error('unknown skill ' + name);
    const bad = checkArgs(s.parameters, args); if (bad) throw new Error(`skill ${name}: ${bad}`);
    args = declared(s.parameters, args);   // a model's arguments are untrusted: only the ones the skill declares go any further
    const t0 = now(), rec = { name, args, ok: true, ms: 0 }; this.calls.push(rec);
    try {
      if (s.handler.type === 'local') {
        const fn = this.handlers[s.handler.fn]; if (!fn) throw new Error(`skill ${name}: no handler bound for "${s.handler.fn}"`);
        return await fn(args, ctx);
      }
      return await this.http(s, args);
    } catch (e) { rec.ok = false; rec.error = String((e && e.message) || e).slice(0, 200); throw e; }
    finally { rec.ms = Math.round(now() - t0); }
  }

  /** an http skill. The URL always stays under the API's baseUrl: a path placeholder can't be empty, '.' or '..', and an argument can't
   *  replace a query parameter the skill's path already sets. Cookies are not sent unless the API sets "credentials". A skill that
   *  changes something (not GET) asks site.confirm({ skill, method, url, args }) first when the page sets it; false cancels the call */
  async http(s, args) {
    const api = this.config.apis[s.handler.api], rest = Object.assign({}, args);
    if (!api) throw new Error(`skill ${s.name}: no api "${s.handler.api}" in the config`);
    const tpl = s.handler.path || '', qi = tpl.includes('?') ? tpl.indexOf('?') : tpl.length;
    const fill = (t, inPath) => t.replace(/\{(\w+)\}/g, (m, k) => {
      const v = rest[k] == null ? '' : String(rest[k]); delete rest[k];
      if (inPath && /^\.{0,2}$/.test(v)) throw new Error(`skill ${s.name}: argument ${k} can't be "${v}" in a path`);
      return encodeURIComponent(v);
    });
    const path = fill(tpl.slice(0, qi), true) + fill(tpl.slice(qi), false);
    const base = new URL(api.baseUrl.endsWith('/') ? api.baseUrl : api.baseUrl + '/', this.base), url = new URL(path.replace(/^\//, ''), base);
    if (url.origin !== base.origin || !url.pathname.startsWith(base.pathname)) throw new Error(`skill ${s.name}: the request would leave ${base.href}`);
    const method = (s.handler.method || 'GET').toUpperCase(), init = { method, headers: Object.assign({}, api.headers || {}), credentials: api.credentials || 'omit' };
    if (method === 'GET') for (const [k, v] of Object.entries(rest)) {
      if (url.searchParams.has(k)) throw new Error(`skill ${s.name}: argument ${k} can't replace the skill's own query parameter`);
      url.searchParams.set(k, typeof v === 'object' ? JSON.stringify(v) : String(v));
    }
    else { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(rest); }
    if (method !== 'GET' && typeof this.confirm === 'function' && !(await this.confirm({ skill: s.name, method, url: url.href, args }))) throw new Error(`skill ${s.name}: cancelled`);
    const r = await fetch(url, init);
    const type = r.headers.get('content-type') || '', body = type.includes('json') ? await r.json() : await r.text();
    if (!r.ok) throw new Error(`HTTP ${r.status} from ${s.handler.api}`);
    return body;
  }

  /** the kernel backend named by kernels.backend ('none' -> null). Built-ins: basic, stub, p0pc0rn-kernels; 'custom' loads kernels.module */
  async loadKernels(extra = {}) {
    const k = this.config.kernels || { backend: 'none' };
    if (!k.backend || k.backend === 'none') return null;
    const href = BUILTIN[k.backend] ? new URL(BUILTIN[k.backend], import.meta.url).href : new URL(k.module, this.base).href;
    const mod = await import(href);
    const backend = await mod.default(Object.assign({}, k.options || {}, k.url ? { url: k.url } : {}, extra));
    const missing = BACKEND_API.filter((m) => typeof backend[m] !== 'function');
    if (missing.length || !backend.ops) throw new Error(`kernel backend ${k.backend} is missing ${missing.concat(backend.ops ? [] : ['ops']).join(', ')}`);
    return backend;
  }
}

/** a light check of tool arguments against a JSON-schema-like 'parameters' (required keys, primitive types) */
function checkArgs(schema, args) {
  if (!args || typeof args !== 'object' || Array.isArray(args)) return 'the arguments must be an object';
  if (!schema || schema.type !== 'object') return null;
  for (const k of schema.required || []) if (!Object.hasOwn(args, k)) return `missing argument ${k}`;
  for (const [k, v] of Object.entries(args)) {
    const t = schema.properties && schema.properties[k] && schema.properties[k].type;
    if (!t) continue;
    const ok = t === 'integer' ? Number.isInteger(v) : t === 'number' ? typeof v === 'number' : t === 'array' ? Array.isArray(v) : t === 'object' ? v && typeof v === 'object' : typeof v === t;
    if (!ok) return `argument ${k} must be ${t}`;
  }
  return null;
}

/** only the skill's declared arguments, own properties only, on an object with no prototype */
function declared(schema, args) {
  const out = Object.create(null), props = (schema && schema.properties) || {};
  for (const k of Object.keys(props)) if (Object.hasOwn(args, k)) out[k] = args[k];
  return out;
}
