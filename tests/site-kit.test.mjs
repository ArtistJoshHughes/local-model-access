// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// site-kit.test.mjs - the site kit's tool calls treat a model's arguments as untrusted (Node 20+, no network: fetch is stubbed):
//   node tests/site-kit.test.mjs
// Checks: an http skill's request never leaves its API's baseUrl ('..', '.' and empty path parts are refused); only declared arguments
// are sent (no extra query or body fields, no "__proto__" tricks); an argument can't replace the skill's own query parameter; no cookies
// unless the API asks; a skill that is not a GET asks site.confirm first.
import { Site } from '../examples/site-kit/site.js';

let fails = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) fails++; };
const sent = [];
globalThis.fetch = async (url, init) => { sent.push({ url: String(url), init }); return new Response('{"ok":true}', { headers: { 'content-type': 'application/json' } }); };
const refused = async (p) => { try { await p; return false; } catch (e) { return true; } };

const site = new Site({
  site: { id: 'test', name: 'Test' },
  apis: { shop: { baseUrl: 'https://api.example.com/v1/' } },
  skills: [
    { name: 'order', parameters: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] }, handler: { type: 'http', api: 'shop', method: 'GET', path: 'orders/{id}' } },
    { name: 'search', parameters: { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] }, handler: { type: 'http', api: 'shop', method: 'GET', path: 'search?limit=5' } },
    { name: 'note', parameters: { type: 'object', properties: { id: { type: 'string' }, text: { type: 'string' } } }, handler: { type: 'http', api: 'shop', method: 'POST', path: 'orders/{id}/notes' } },
    { name: 'find', parameters: { type: 'object', properties: { q: { type: 'string' } } }, handler: { type: 'http', api: 'shop', method: 'GET', path: 'search?q={q}' } },
    { name: 'count', parameters: { type: 'object', properties: { n: { type: 'integer' } }, required: ['n'] }, handler: { type: 'http', api: 'shop', method: 'GET', path: '{n}/invoice' } },
  ],
  models: [],
}, new URL('https://site.example/'));

await site.callTool('order', { id: 'A-17' });
check(sent.at(-1).url === 'https://api.example.com/v1/orders/A-17' && sent.at(-1).init.credentials === 'omit', 'a normal call stays under the base, with no cookies');
check(await refused(site.callTool('order', { id: '..' })) && await refused(site.callTool('order', { id: '.' })) && await refused(site.callTool('order', { id: '' })), "path parts '..', '.' and '' are refused");
const before = sent.length;
check(await refused(site.callTool('count', JSON.parse('{"__proto__": {"n": ".."}}'))) && sent.length === before, 'a "__proto__" argument cannot stand in for a declared one');
await site.callTool('order', { id: 'a/../../admin' });
check(sent.at(-1).url.startsWith('https://api.example.com/v1/orders/'), 'a slash in an argument is encoded, so it stays one path part');
await site.callTool('search', { q: 'boots', limit: 100000, admin: true });
const u = new URL(sent.at(-1).url);
check(u.searchParams.get('limit') === '5' && !u.searchParams.has('admin') && u.searchParams.get('q') === 'boots', 'undeclared arguments are dropped, and the skill keeps its own limit');
check(await refused(site.callTool('search', { q: 'x', limit: 9 })) === false && new URL(sent.at(-1).url).searchParams.get('limit') === '5', 'an undeclared argument named like the query is dropped, not used');
await site.callTool('find', { q: '' });
check(sent.at(-1).url === 'https://api.example.com/v1/search?q=', 'a placeholder in the query may be empty (only path parts are refused)');
await site.callTool('note', { id: 'A-17', text: 'hi', is_admin: true });
check(JSON.parse(sent.at(-1).init.body).is_admin === undefined, 'a POST body holds only declared arguments');
site.confirm = async () => false;
const n = sent.length;
check(await refused(site.callTool('note', { id: 'A-17', text: 'hi' })) && sent.length === n, 'site.confirm can cancel a skill that is not a GET');
site.confirm = async ({ method, url }) => method === 'POST' && url.endsWith('/notes');
await site.callTool('note', { id: 'A-17', text: 'hi' });
check(sent.length === n + 1, 'site.confirm approves it');
console.log(fails ? `FAILED ${fails}` : 'ALL OK');
process.exit(fails ? 1 : 0);
