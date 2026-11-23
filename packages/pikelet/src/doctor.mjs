// `pikelet doctor <url>` — probe an artifact hosting URL for
// the transport properties the range readers depend on, and print a verdict
// before anyone discovers them as "the demo is slow" in production:
//
//   - Range honored (206) — without it, readers fall back to a full download
//     or refuse files over the fallback cap.
//   - Range honored with a cache-key query string — the browser readers
//     append ?r=start-end to every read because Chromium serializes
//     concurrent fetches of one cacheable URL on its HTTP-cache entry lock
//     (measured 5.7-18.6 s/query without it, 0.6-1.5 s with — see
//     examples/static-wiki-pack/DEPLOY.md).
//   - Negotiated protocol — HTTP/1.1 caps browsers at ~6 connections per
//     origin, so parallel rerank reads serialize; h2/h3 multiplex.
//   - ETag — without one, If-Range cannot detect a swapped file mid-session.
//   - RTT — the multiplier on every dependent read round.

import tls from 'node:tls';
import { nodeFetch } from 'pikelet-wasm/complete/node-fetch.mjs';

const MAGICS = {
  0x31465350: 'complete artifact (PSF1)',
  0x31415350: 'sketch artifact (PSA1)',
  0x31415250: 'range artifact (PRA1)',
};

function alpnProbe(url, timeoutMs = 5000) {
  const { hostname, port, protocol } = new URL(url);
  if (protocol !== 'https:') return Promise.resolve(null);
  return new Promise((resolve) => {
    const socket = tls.connect({
      host: hostname,
      port: Number(port) || 443,
      servername: hostname,
      ALPNProtocols: ['h2', 'http/1.1'],
    }, () => {
      const alpn = socket.alpnProtocol || 'http/1.1';
      socket.end();
      resolve(alpn);
    });
    socket.setTimeout(timeoutMs, () => { socket.destroy(); resolve(null); });
    socket.on('error', () => resolve(null));
  });
}

async function timedRange(url, range, extraQuery) {
  const target = extraQuery ? `${url}${url.includes('?') ? '&' : '?'}${extraQuery}` : url;
  const t0 = performance.now();
  const response = await nodeFetch(target, { headers: { Range: `bytes=${range}` }, redirect: 'follow' });
  const body = new Uint8Array(await response.arrayBuffer());
  return { ms: performance.now() - t0, status: response.status, headers: response.headers, body };
}

export async function probeHosting(url) {
  const report = { url, checks: [], failures: 0, warnings: 0 };
  const check = (level, name, detail) => {
    report.checks.push({ level, name, detail });
    if (level === 'FAIL') report.failures++;
    if (level === 'WARN') report.warnings++;
  };

  const head = await nodeFetch(url, { method: 'HEAD', redirect: 'follow' }).catch((error) => {
    check('FAIL', 'reachable', `HEAD failed: ${error.message}`);
    return null;
  });
  if (!head) return report;
  if (!head.ok) check('FAIL', 'reachable', `HEAD returned ${head.status}`);
  const size = Number(head.headers.get('content-length'));
  check('info', 'size', Number.isFinite(size) && size > 0 ? `${(size / 1048576).toFixed(2)} MiB` : 'no Content-Length on HEAD');
  const acceptRanges = head.headers.get('accept-ranges');
  check(acceptRanges === 'bytes' ? 'ok' : 'WARN', 'Accept-Ranges header',
    acceptRanges === 'bytes' ? 'bytes' : `${acceptRanges || 'absent'} (advisory only; the GET below is the real test)`);
  const etag = head.headers.get('etag');
  check(etag ? 'ok' : 'WARN', 'ETag',
    etag || 'absent — If-Range cannot detect a swapped file mid-session; readers over 64 MiB will refuse the fallback');

  // The real range test, then the same range through a cache-key query
  // string (the form every browser read actually uses).
  const plain = await timedRange(url, '0-63').catch((error) => ({ error }));
  if (plain.error || (plain.status !== 206 && plain.status !== 200)) {
    check('FAIL', 'Range GET', plain.error ? plain.error.message : `HTTP ${plain.status}`);
    return report;
  }
  if (plain.status === 200) {
    check('FAIL', 'Range GET', 'host ignores Range and returns the full body — readers fall back to a full download or refuse');
  } else {
    check(plain.body.length === 64 ? 'ok' : 'FAIL', 'Range GET',
      plain.body.length === 64 ? `206, correct 64-byte slice` : `206 but ${plain.body.length} bytes for a 64-byte range`);
  }
  const keyed = await timedRange(url, '0-63', 'r=0-63').catch((error) => ({ error }));
  if (keyed.error || keyed.status !== 206) {
    check('WARN', 'Range GET with ?r= cache key',
      keyed.error ? keyed.error.message : `HTTP ${keyed.status} — browser readers append ?r=start-end to defeat Chromium's same-URL cache-entry lock; this host breaks that`);
  } else {
    check('ok', 'Range GET with ?r= cache key', '206 — Chromium cache-lock workaround will function');
  }

  const alpn = await alpnProbe(url);
  const altSvc = head.headers.get('alt-svc') || '';
  const h3 = /h3/.test(altSvc);
  if (alpn === 'h2' || h3) {
    check('ok', 'protocol', `${alpn || 'https'}${h3 ? ' (+h3 advertised)' : ''} — parallel range reads multiplex`);
  } else if (alpn === 'http/1.1') {
    check('WARN', 'protocol', 'HTTP/1.1 only — browsers cap ~6 connections per origin, so parallel rerank reads serialize (measured: p=6 turns a 1.5 s query into 6.9 s at 100 ms RTT)');
  } else {
    check('WARN', 'protocol',
      url.startsWith('https:') ? 'ALPN probe failed' : 'plain http — no h2 without TLS, and the 6-connection cap applies');
  }

  // RTT: median of three keyed 1-byte reads (post-warmup, so DNS/TLS are out).
  const samples = [];
  for (let i = 0; i < 3; i++) {
    const s = await timedRange(url, '0-0', `r=0-0&s=${i}`).catch(() => null);
    if (s && (s.status === 206 || s.status === 200)) samples.push(s.ms);
  }
  if (samples.length) {
    samples.sort((a, b) => a - b);
    check('info', 'RTT', `${samples[Math.floor(samples.length / 2)].toFixed(0)} ms median over ${samples.length} small reads`);
  }

  // Identify the artifact from the first 64 bytes.
  if (plain.status === 206 && plain.body.length >= 64) {
    const view = new DataView(plain.body.buffer, plain.body.byteOffset, plain.body.byteLength);
    const kind = MAGICS[view.getUint32(0, true)];
    if (kind) {
      const identity = kind.includes('PSF1')
        ? Array.from(plain.body.subarray(24, 56), (b) => b.toString(16).padStart(2, '0')).join('')
        : null;
      check('ok', 'artifact', `${kind}${identity ? `, identity ${identity.slice(0, 16)}…` : ''}`);
    } else {
      check('info', 'artifact', 'first bytes are not a Pikelet magic (fine if the URL is not an artifact)');
    }
  }
  return report;
}

export async function runDoctor(url, log = console.log) {
  if (!url || !/^https?:\/\//.test(url)) {
    throw Object.assign(new Error('usage: pikelet doctor <http(s) url of the hosted artifact>'), { exitCode: 1 });
  }
  log(`doctor: probing ${url}`);
  const report = await probeHosting(url);
  const badge = { ok: '  ok  ', WARN: ' WARN ', FAIL: ' FAIL ', info: ' info ' };
  for (const c of report.checks) log(`[${badge[c.level]}] ${c.name}: ${c.detail}`);
  if (report.failures) log(`verdict: NOT ready — ${report.failures} failing check(s); range-read serving will not work correctly`);
  else if (report.warnings) log(`verdict: usable with caveats — ${report.warnings} warning(s) above affect latency or update safety`);
  else log('verdict: ready — range serving, caching, and protocol all look right');
  return report;
}
