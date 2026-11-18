// nodeFetch (packages/pikelet-wasm/complete/node-fetch.mjs), the portable
// range source's Content-Encoding and downgrade checks, the Node entry's
// default transport, the crawler, and the CLI's supported-Node and plain-http
// warnings.

import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import tls from 'node:tls';
import zlib from 'node:zlib';
import { execFileSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { nodeFetch, contentCodings } from '../packages/pikelet-wasm/complete/node-fetch.mjs';
import { httpRangeSource } from '../packages/pikelet-wasm/complete/sources.mjs';
import * as nodeEntry from '../packages/pikelet-wasm/complete/index.node.mjs';
import { isSupportedNode, nodeSupportWarning } from '../packages/pikelet/src/node-support.mjs';
import { plainHttpWarning } from '../packages/pikelet/src/common.mjs';
import { ingestUrl } from '../packages/pikelet/src/ingest.mjs';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
    if (ok) { passed++; console.log(`  ok   ${name}`); } else { failed++; console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`); }
}
async function rejects(name, fn, pattern) {
    try { await fn(); check(name, false, 'did not throw'); } catch (err) { check(name, pattern.test(String(err?.message)), String(err?.message)); }
}
const withTimeout = (promise, ms) => Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error(`hung for ${ms} ms`)), ms))]);

const payload = Buffer.from(Array.from({ length: 4096 }, (_, i) => (i * 31) & 0xff));
const seen = [];
let open = 0;
let maxOpen = 0;
const ENCODED = { '/gzip': 'gzip', '/br': 'br', '/Identity': 'Identity', '/list': 'gzip, br', '/identity-gzip': 'identity, gzip' };
const host = http.createServer((req, res) => {
    const u = new URL(req.url, 'http://x');
    seen.push({ path: u.pathname, headers: req.headers });
    if (u.pathname === '/front') { res.writeHead(302, { location: 'sub/../plain' }); res.end(); return; }
    if (u.pathname === '/loop') { res.writeHead(302, { location: '/loop' }); res.end(); return; }
    if (u.pathname === '/reset') { res.writeHead(205); res.end(); return; }
    if (u.pathname === '/truncated') {
        res.writeHead(200, { 'content-length': 1000 });
        res.write(payload.subarray(0, 100));
        setTimeout(() => res.socket.destroy(), 20);
        return;
    }
    if (u.pathname === '/slow') {
        open++; maxOpen = Math.max(maxOpen, open);
        setTimeout(() => { open--; res.writeHead(200, { 'content-length': 2 }); res.end('ok'); }, 30);
        return;
    }
    const coding = ENCODED[u.pathname];
    if (u.pathname !== '/plain' && !coding) { res.writeHead(404); res.end(); return; }
    const range = req.headers.range && /^bytes=(\d+)-(\d+)$/.exec(req.headers.range);
    const extra = coding ? { 'content-encoding': coding } : {};
    const encode = (b) => (coding === 'gzip' ? zlib.gzipSync(b) : coding === 'br' ? zlib.brotliCompressSync(b) : b);
    if (!range) {
        const body = encode(payload);
        res.writeHead(200, { 'accept-ranges': 'bytes', 'content-length': body.length, etag: '"v1"', ...extra });
        res.end(req.method === 'HEAD' ? undefined : body);
        return;
    }
    const from = Number(range[1]);
    const to = Math.min(Number(range[2]), payload.length - 1);
    const body = encode(payload.subarray(from, to + 1));
    res.writeHead(206, { 'content-range': `bytes ${from}-${to}/${payload.length}`, 'content-length': body.length, etag: '"v1"', ...extra });
    res.end(body);
});
await new Promise((resolve) => host.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${host.address().port}`;

console.log('nodeFetch');
{
    const r = await nodeFetch(`${base}/plain`, { headers: { Range: 'bytes=10-19' } });
    const body = Buffer.from(await r.arrayBuffer());
    check('a range GET returns the exact slice', r.status === 206 && Buffer.compare(body, payload.subarray(10, 20)) === 0, `status ${r.status}`);
    check('it asks for identity encoding', seen.at(-1).headers['accept-encoding'] === 'identity', seen.at(-1).headers['accept-encoding']);

    seen.length = 0;
    const followed = await nodeFetch(`${base}/front`, { headers: { Range: 'bytes=0-7', 'If-Range': '"v1"' } });
    const hop = seen.find((s) => s.path === '/plain');
    check('redirect: follow resolves a relative Location against the current URL', followed.status === 206 && !!hop);
    check('Range and If-Range ride along across the hop', hop?.headers.range === 'bytes=0-7' && hop?.headers['if-range'] === '"v1"'
        && Buffer.compare(Buffer.from(await followed.arrayBuffer()), payload.subarray(0, 8)) === 0);
    const manual = await nodeFetch(`${base}/front`, { method: 'HEAD', redirect: 'manual' });
    check('redirect: manual returns the 3xx with its Location', manual.status === 302 && manual.headers.get('location') === 'sub/../plain');
    await rejects('redirect: error refuses the hop', () => nodeFetch(`${base}/front`, { redirect: 'error' }), /redirect refused/);
    await rejects('a redirect loop is bounded at 20 hops', () => nodeFetch(`${base}/loop`), /too many redirects/);

    await rejects('Content-Encoding: gzip fails closed', () => nodeFetch(`${base}/gzip`, { headers: { Range: 'bytes=0-15' } }), /Content-Encoding: gzip/);
    await rejects('Content-Encoding: br fails closed', () => nodeFetch(`${base}/br`), /Content-Encoding: br/);
    await rejects('a coding list "gzip, br" fails closed', () => nodeFetch(`${base}/list`), /Content-Encoding: gzip, br/);
    await rejects('"identity, gzip" fails closed', () => nodeFetch(`${base}/identity-gzip`), /identity, gzip/);
    const ident = await nodeFetch(`${base}/Identity`, { headers: { Range: 'bytes=0-3' } });
    check('"Identity" (any case) is accepted', ident.status === 206 && (await ident.arrayBuffer()).byteLength === 4);
    check('contentCodings parses lists and drops identity',
        JSON.stringify(contentCodings(' GZip , identity,BR ')) === '["gzip","br"]' && contentCodings('Identity').length === 0 && contentCodings(null).length === 0);

    const inflated = await nodeFetch(`${base}/gzip`, { inflateGzip: true });
    check('inflateGzip inflates a gzip body and drops the coding headers',
        Buffer.compare(Buffer.from(await inflated.arrayBuffer()), payload) === 0 && inflated.headers.get('content-encoding') === null);
    await rejects('inflateGzip still refuses Brotli', () => nodeFetch(`${base}/br`, { inflateGzip: true }), /Content-Encoding: br/);
    await rejects('inflateGzip still refuses a "gzip, br" list', () => nodeFetch(`${base}/list`, { inflateGzip: true }), /gzip, br/);

    const reset = await nodeFetch(`${base}/reset`).then((x) => x.status, (err) => err.message);
    check('a 205 (null-body status) resolves instead of throwing', reset === 205, String(reset));
    const head = await nodeFetch(`${base}/plain`, { method: 'HEAD' });
    check('HEAD carries headers and no body', head.status === 200 && head.body === null && Number(head.headers.get('content-length')) === payload.length);

    const truncated = await nodeFetch(`${base}/truncated`);
    await rejects('a body cut off mid-stream rejects arrayBuffer() instead of hanging', () => withTimeout(truncated.arrayBuffer(), 5000), /aborted|terminated|ECONNRESET|socket|close/i);
    await rejects('an aborted request rejects', () => nodeFetch(`${base}/slow`, { signal: AbortSignal.timeout(5) }), /abort|timeout/i);

    // The aborted /slow above is still counted until its server-side timer
    // fires; start the burst from an idle server or it reads one high.
    while (open > 0) await new Promise((resolve) => setTimeout(resolve, 10));
    maxOpen = 0;
    await Promise.all(Array.from({ length: 64 }, () => nodeFetch(`${base}/slow`).then((x) => x.text())));
    check('a 64-read burst opens at most 32 connections to one host', maxOpen > 1 && maxOpen <= 32, `max concurrent ${maxOpen}`);
}

console.log('https downgrade');
{
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-tls-'));
    let tlsHost = null;
    try {
        execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
            '-keyout', path.join(dir, 'key.pem'), '-out', path.join(dir, 'cert.pem')], { stdio: 'ignore' });
        tlsHost = https.createServer({ key: fs.readFileSync(path.join(dir, 'key.pem')), cert: fs.readFileSync(path.join(dir, 'cert.pem')) }, (req, res) => {
            res.writeHead(302, { location: `${base}/plain` });
            res.end();
        });
        await new Promise((resolve) => tlsHost.listen(0, '127.0.0.1', resolve));
        const tlsBase = `https://127.0.0.1:${tlsHost.address().port}`;
        if (typeof tls.setDefaultCACertificates !== 'function') {
            // Every supported Node has it (22.19+); older runtimes skip.
            console.log('  skip: https downgrade (tls.setDefaultCACertificates needs Node 22.19+)');
        } else {
            // Trust the throwaway self-signed cert above for this check only;
            // certificate validation itself stays on.
            const priorCAs = tls.getCACertificates('default');
            tls.setDefaultCACertificates([...priorCAs, fs.readFileSync(path.join(dir, 'cert.pem'), 'utf8')]);
            try {
                await rejects('nodeFetch refuses an https -> http redirect', () => nodeFetch(`${tlsBase}/pack`), /refused redirect from https to http/);
            } finally {
                tls.setDefaultCACertificates(priorCAs);
            }
        }
    } catch (err) {
        check('https downgrade test ran (needs openssl)', false, err.message);
    } finally {
        tlsHost?.close();
        fs.rmSync(dir, { recursive: true, force: true });
    }
    // The range source's own HEAD walk reads Location itself (redirect:
    // 'manual'), so it must refuse the downgrade too, on any transport.
    const stubFetch = async (url) => (String(url).startsWith('https:')
        ? new Response(null, { status: 302, headers: { location: 'http://example.invalid/pack' } })
        : new Response(null, { status: 200, headers: { 'content-length': '10' } }));
    await rejects('the range source\'s redirect walk refuses https -> http', () => httpRangeSource('https://example.invalid/pack', { fetch: stubFetch }).init(), /refused redirect from https/);
}

console.log('httpRangeSource');
{
    const src = httpRangeSource(`${base}/front`, { fetch: nodeFetch });
    await src.init();
    const got = await src.read(100, 64);
    check('an injected fetch serves the redirect walk and reads', src.size === payload.length
        && Buffer.compare(Buffer.from(got), payload.subarray(100, 164)) === 0 && src.stats.redirects === 1, `size ${src.size}`);
    check('the source exposes its injected transport for a chain walk', src.fetch === nodeFetch && httpRangeSource(`${base}/plain`).fetch === undefined);
    // The portable check, on the runtime's own fetch: the body may have been
    // decoded already, but the header still says it was encoded.
    const portable = httpRangeSource(`${base}/gzip`);
    await portable.init();
    await rejects('globalThis.fetch: an encoded range response is refused', () => portable.read(0, 32), /Content-Encoding: gzip/);
    const listed = httpRangeSource(`${base}/list`);
    await rejects('globalThis.fetch: a coding list is refused', () => listed.read(0, 32), /Content-Encoding: gzip, br/);

    check('the Node entry exports nodeFetch', nodeEntry.nodeFetch === nodeFetch);
    check('the Node entry\'s httpRangeSource defaults to nodeFetch', nodeEntry.httpRangeSource(`${base}/plain`).fetch === nodeFetch);
    check('options.fetch still overrides the Node default', nodeEntry.httpRangeSource(`${base}/plain`, { fetch: globalThis.fetch }).fetch === globalThis.fetch);
    const resolved = import.meta.resolve('pikelet-wasm/complete');
    check('\'pikelet-wasm/complete\' resolves to the Node entry under Node', resolved.endsWith('/complete/index.node.mjs'), resolved);
    await rejects('a library mount of a Brotli-encoded pack fails with the identity error',
        () => nodeEntry.openPikeletFile(nodeEntry.httpRangeSource(`${base}/br`)), /Content-Encoding: br/);
}

console.log('CLI');
{
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-transport-'));
    fs.mkdirSync(path.join(tmp, 'docs'));
    fs.writeFileSync(path.join(tmp, 'docs', 'a.md'), '# A\n\nSome text about range reads.\n');
    // Async: a synchronous spawn would block this process's own server.
    const run = await new Promise((resolve) => {
        const child = spawn(process.execPath, [path.join(REPO, 'packages/pikelet/bin/pikelet.mjs'), 'append',
            '--parent', `${base}/br`, '--source', path.join(tmp, 'docs'), '--out', path.join(tmp, 'out.pikelet')],
        { env: { ...process.env, PIKELET_NO_NODE_WARNING: '1' } });
        let stderr = '';
        child.stderr.on('data', (d) => { stderr += d; });
        child.on('close', (status) => resolve({ status, stderr }));
    });
    check('a CLI mount of a Brotli-encoded parent fails with the identity error', run.status !== 0 && /Content-Encoding: br/.test(run.stderr),
        `exit ${run.status}: ${run.stderr.slice(-300)}`);
    fs.rmSync(tmp, { recursive: true, force: true });

    check('plain http to a remote host warns', /plain http/.test(plainHttpWarning('http://packs.example.com/a.pikelet') || ''));
    check('https does not warn', plainHttpWarning('https://packs.example.com/a.pikelet') === null);
    check('loopback http does not warn', ['http://127.0.0.1:8080/a', 'http://localhost/a', 'http://[::1]/a'].every((u) => plainHttpWarning(u) === null));
}
host.close();

console.log('crawler');
{
    const page = (title, links = '') => `<!doctype html><html><head><title>${title}</title></head><body><main><h1>${title}</h1>`
        + `<p>${title} explains how range reads fetch only the bytes a query needs from a static host.</p>${links}</main></body></html>`;
    const site = http.createServer((req, res) => {
        const html = { '/docs/': page('Index', '<a href="/docs/gz">gz</a> <a href="/docs/br">br</a>'), '/docs/gz': page('Gzipped'), '/docs/br': page('Brotli') }[req.url];
        if (!html) { res.writeHead(404); res.end(); return; }
        // A server that compresses whatever it was asked for.
        const coding = req.url.endsWith('gz') ? 'gzip' : req.url.endsWith('br') ? 'br' : null;
        const body = coding === 'gzip' ? zlib.gzipSync(html) : coding === 'br' ? zlib.brotliCompressSync(html) : Buffer.from(html);
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'content-length': body.length, ...(coding ? { 'content-encoding': coding } : {}) });
        res.end(body);
    });
    await new Promise((resolve) => site.listen(0, '127.0.0.1', resolve));
    const siteBase = `http://127.0.0.1:${site.address().port}`;
    const lines = [];
    const docs = await ingestUrl({ url: `${siteBase}/docs/`, maxPages: 10 }, (line) => lines.push(line));
    site.close();
    const titles = docs.map((d) => d.title).sort().join(',');
    check('the crawler reads a page a server gzipped despite identity', titles === 'Gzipped,Index', titles);
    check('a Brotli page is skipped with a warning naming the URL and the coding',
        lines.some((l) => l.includes(`${siteBase}/docs/br`) && /Content-Encoding: br/.test(l)), lines.join(' | '));
}
{
    // Every page repeats the same 20-link nav and adds 5 children of its own
    // (121 pages). Repeated nav links must be queued once, so a 60-page crawl
    // fills its budget instead of stalling on duplicate queue entries.
    const words = 'pages share one navigation menu and each adds pages of its own '.repeat(4);
    let served = 0;
    const site = http.createServer((req, res) => {
        // Pages are labelled from numbers, never from req.url, so nothing the
        // request carries is reflected into the HTML.
        const nav = Array.from({ length: 20 }, (_, i) => `<a href="/n${i}">n${i}</a>`).join('');
        const m = /^\/n(\d+)$/.exec(req.url);
        const section = m ? Number(m[1]) : -1;
        const kids = section >= 0 ? Array.from({ length: 5 }, (_, j) => `<a href="/n${section}/c${j}">c</a>`).join('') : '';
        const page = ++served;
        res.writeHead(200, { 'content-type': 'text/html' });
        res.end(`<html><title>page ${page}</title><body>${nav}${kids}<p>page ${page} ${words}</p></body></html>`);
    });
    await new Promise((resolve) => site.listen(0, '127.0.0.1', resolve));
    const docs = await ingestUrl({ url: `http://127.0.0.1:${site.address().port}/`, maxPages: 60 }, () => {});
    site.close();
    check('a crawl over repeated nav links fills its maxPages budget', docs.length === 60, String(docs.length));
    check('a crawl fetches each page once', new Set(docs.map((d) => d.url)).size === docs.length);
}

console.log('range response bounds and retry delays');
{
    const sourceFor = (body, contentRange = 'bytes 0-3/4') => httpRangeSource('https://range.test/pack', {
        fetch: async (_, init) => init.method === 'HEAD'
            ? new Response(null, { headers: { 'content-length': '4' } })
            : new Response(body, { status: 206, headers: { 'content-range': contentRange } }),
    });
    let pulls = 0;
    let cancelled = false;
    const oversized = sourceFor(new ReadableStream({
        pull(controller) {
            pulls++;
            controller.enqueue(new Uint8Array(4));
            if (pulls === 100) controller.close();
        },
        cancel() { cancelled = true; },
    }, { highWaterMark: 0 }));
    await rejects('oversized 206 is refused', () => oversized.read(0, 4), /body exceeds 4 bytes/);
    check('oversized 206 is cancelled at the first excess chunk', cancelled && pulls === 2, `${cancelled}, ${pulls}`);
    await rejects('short 206 is refused at EOF', () => sourceFor(new Uint8Array(3)).read(0, 4), /body is 3 bytes/);
    const valid = sourceFor(new ReadableStream({ start(c) {
        c.enqueue(new Uint8Array([1, 2])); c.enqueue(new Uint8Array([3, 4])); c.close();
    } }));
    check('valid multi-chunk 206 retains its bytes', (await valid.read(0, 4)).join(',') === '1,2,3,4');
    // A host that ignores Range: reads in flight together share one full
    // download instead of each buffering the whole file.
    {
        const SIZE = 4096;
        let fullyRead = 0;
        let gets = 0;
        const file = Uint8Array.from({ length: SIZE }, (_, i) => i & 0xff);
        const source = httpRangeSource('https://range.test/ignores-range', {
            fetch: async (_, init) => {
                if (init.method === 'HEAD') return new Response(null, { headers: { 'content-length': String(SIZE) } });
                gets++;
                let sent = 0;
                return new Response(new ReadableStream({
                    pull(c) {
                        if (sent >= SIZE) { fullyRead++; c.close(); return; }
                        c.enqueue(file.slice(sent, sent + 1024)); sent += 1024;
                    },
                }, { highWaterMark: 0 }), { status: 200, headers: { 'content-length': String(SIZE) } });
            },
        });
        await source.init();
        const warn = console.warn; console.warn = () => {};
        let slices;
        try {
            slices = await Promise.all(Array.from({ length: 16 }, (_, i) => source.read(i * 256, 256)));
        } finally { console.warn = warn; }
        check('concurrent reads on a host that ignores Range download the file once', fullyRead === 1, `${fullyRead} full bodies, ${gets} GETs`);
        check('each shared-download read gets its own slice', slices.every((s, i) => s.length === 256 && s[0] === ((i * 256) & 0xff)));
        const before = gets;
        await source.read(100, 10);
        check('a read after the fallback is served from memory', gets === before);
    }
    // 206 responses are held to the pinned ETag.
    {
        const sourceWith = (headEtag, readEtags) => {
            let n = 0;
            return httpRangeSource('https://range.test/etag', {
                fetch: async (_, init) => init.method === 'HEAD'
                    ? new Response(null, { headers: { 'content-length': '4', ...(headEtag ? { etag: headEtag } : {}) } })
                    : new Response(new Uint8Array(4), { status: 206, headers: { 'content-range': 'bytes 0-3/4', etag: readEtags[Math.min(n++, readEtags.length - 1)] } }),
            });
        };
        const changed = sourceWith('"a"', ['"b"']);
        await changed.init();
        await rejects('a 206 carrying a different ETag than HEAD pinned is refused', () => changed.read(0, 4), /artifact changed on the host/);
        const weak = sourceWith('"a"', ['W/"a"']);
        await weak.init();
        check('a 206 whose ETag matches weakly is accepted', (await weak.read(0, 4)).length === 4);
        const unpinned = sourceWith(null, ['"a"', '"b"']);
        await unpinned.init();
        await unpinned.read(0, 4);
        await rejects('with no ETag from HEAD, the first range read pins one', () => unpinned.read(0, 4), /artifact changed on the host/);
    }
    const originalTimer = globalThis.setTimeout;
    try {
        for (const [header, expected] of [
            [null, [250, 500, 1000]], ['', [250, 500, 1000]],
            ['invalid', [250, 500, 1000]], ['2', [2000, 2000, 2000]], ['0', [0, 0, 0]],
        ]) {
            const delays = [];
            globalThis.setTimeout = (fn, ms) => { delays.push(ms); return originalTimer(fn, 0); };
            let reads = 0;
            // idleTimeoutMs: 0 keeps the idle clock out of the recorded delays.
            const source = httpRangeSource('https://range.test/retry', { idleTimeoutMs: 0, fetch: async (_, init) => {
                if (init.method === 'HEAD') return new Response(null);
                if (++reads <= 3) return new Response(null, { status: 503,
                    headers: header === null ? {} : { 'retry-after': header } });
                return new Response(new Uint8Array(4), { status: 206, headers: { 'content-range': 'bytes 0-3/4' } });
            } });
            await source.read(0, 4);
            check(`Retry-After ${JSON.stringify(header)} uses the expected delays`,
                JSON.stringify(delays) === JSON.stringify(expected), JSON.stringify(delays));
        }
    } finally {
        globalThis.setTimeout = originalTimer;
    }
}

console.log('stalled hosts');
{
    // A host that accepts a request and then sends nothing must fail the
    // read, not leave it (and every query behind it) pending forever.
    const site = http.createServer((req, res) => {
        if (req.url.startsWith('/no-headers')) return;
        if (req.url.startsWith('/body-stall')) {
            res.writeHead(206, { 'content-range': 'bytes 0-99/100', 'content-length': '100' });
            res.write(Buffer.alloc(10));
            return;
        }
        if (req.url.startsWith('/slow')) {
            res.writeHead(200);
            let n = 0;
            const tick = setInterval(() => { res.write('y'); if (++n === 4) { clearInterval(tick); res.end(); } }, 150);
            return;
        }
        res.writeHead(404); res.end();
    });
    await new Promise((resolve) => site.listen(0, '127.0.0.1', resolve));
    const at = `http://127.0.0.1:${site.address().port}`;
    const timed = async (fn) => { const t0 = Date.now(); try { await fn(); return { ms: Date.now() - t0 }; } catch (err) { return { ms: Date.now() - t0, err }; } };

    let r = await timed(() => nodeFetch(`${at}/no-headers`, { idleTimeoutMs: 300 }));
    check('nodeFetch gives up on a host that never answers', r.err?.code === 'ETIMEDOUT' && r.ms < 3000, `${r.err?.message} after ${r.ms} ms`);
    r = await timed(async () => (await nodeFetch(`${at}/body-stall`, { idleTimeoutMs: 300 })).arrayBuffer());
    check('nodeFetch gives up on a body that stops arriving', /sent nothing/.test(r.err?.message) && r.ms < 3000, `${r.err?.message} after ${r.ms} ms`);
    r = await timed(async () => (await nodeFetch(`${at}/slow`, { idleTimeoutMs: 300 })).text());
    check('a body slower than the idle timeout overall, but moving, is read whole', !r.err && r.ms > 450, `${r.err?.message} after ${r.ms} ms`);
    await rejects('nodeFetch refuses a negative idleTimeoutMs', () => nodeFetch(`${at}/slow`, { idleTimeoutMs: -1 }), /idleTimeoutMs/);

    // The range source keeps its own clock, so a transport that ignores
    // signals (or a browser fetch) cannot hang a query either.
    let aborted = false;
    const silent = httpRangeSource('https://range.test/silent', {
        idleTimeoutMs: 300,
        fetch: (_, init) => {
            if (init.method === 'HEAD') return Promise.resolve(new Response(null, { headers: { 'content-length': '100' } }));
            init.signal?.addEventListener('abort', () => { aborted = true; });
            return new Promise(() => {});
        },
    });
    await silent.init();
    r = await timed(() => silent.read(0, 10));
    check('a range read on a transport that never answers fails', r.err?.code === 'ETIMEDOUT' && r.ms < 3000, `${r.err?.message} after ${r.ms} ms`);
    check('the stalled request is aborted through its signal', aborted);
    check('the error names the range and the host', /bytes 0-9 from https:\/\/range\.test\/silent/.test(r.err?.message), r.err?.message);
    const silentHead = httpRangeSource('https://range.test/silent-head', { idleTimeoutMs: 300, fetch: () => new Promise(() => {}) });
    r = await timed(() => silentHead.init());
    // Not a host without HEAD support (that falls back to plain reads): a
    // host that never answers fails init, rather than every read after it.
    check('a HEAD that never answers fails init()', r.err?.code === 'ETIMEDOUT' && r.ms < 3000, `${r.err?.message} after ${r.ms} ms`);

    const stalled = httpRangeSource(`${at}/body-stall`, { idleTimeoutMs: 300, fetch: nodeFetch, cacheKeyParam: null });
    r = await timed(() => stalled.read(0, 100));
    check('a 206 that stops mid-body fails the read', /sent nothing/.test(r.err?.message) && r.ms < 3000, `${r.err?.message} after ${r.ms} ms`);
    const neverAnswers = httpRangeSource(`${at}/no-headers`, { idleTimeoutMs: 300, fetch: nodeFetch, cacheKeyParam: null });
    r = await timed(async () => { await neverAnswers.init(); await neverAnswers.read(0, 10); });
    check('a real host that accepts and never answers fails within one timeout', /sent nothing/.test(r.err?.message) && r.ms < 1500, `${r.err?.message} after ${r.ms} ms`);
    check('httpRangeSource refuses a negative idleTimeoutMs',
        (() => { try { httpRangeSource('https://x.test/p', { idleTimeoutMs: -5 }); return false; } catch (err) { return /idleTimeoutMs/.test(err.message); } })());
    site.closeAllConnections();
    site.close();
}

console.log('supported Node versions');
for (const [v, want] of [
    ['18.20.8', false], ['20.20.2', false], ['22.23.3', false], ['22.23.4', true], ['22.30.0', true],
    ['23.11.0', false], ['24.22.0', false], ['24.22.1', true], ['25.9.0', false], ['26.12.0', false],
    ['26.12.1', true], ['27.0.0', true], ['v24.23.0', true],
]) {
    check(`${v} is ${want ? '' : 'not '}supported`, isSupportedNode(v) === want);
}
check('a supported version gets no warning', nodeSupportWarning('24.22.1') === null);
check('an unsupported version gets one line naming the range', /Node 20\.20\.2 is unsupported.*>=22\.23\.4/.test(nodeSupportWarning('20.20.2') || '') && !nodeSupportWarning('20.20.2').includes('\n'));

console.log(`\nNode transport: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
