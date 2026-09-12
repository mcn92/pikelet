#!/usr/bin/env node
// Four failure modes a "just static files + Range" architecture must
// survive, run against a real .pikelet pack over a real HTTP server:
//
//   1. server honors Range correctly               -> works (baseline)
//   2. server ignores Range, returns the full file  -> falls back safely,
//                                                       flagged, not silent
//   3. bytes at the URL change but the hash pin doesn't -> refuses to mount
//   4. one range response is truncated/corrupted    -> fails loudly, not
//                                                       silently wrong
//
// Usage: node bench/range-proof/failure-modes.mjs [path/to/pack.pikelet]

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPikeletFile, httpRangeSource } from '../../complete/index.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const packPath = path.resolve(process.argv[2] || path.join(ROOT, 'local-packs/pride-prejudice.pikelet'));
const packBytes = fs.readFileSync(packPath);

function startServer(handler) {
  const server = http.createServer(handler);
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      resolve({ url: `http://127.0.0.1:${server.address().port}`, close: () => new Promise((r) => server.close(r)) });
    });
  });
}

function serveRangeCorrectly(bytes) {
  return (req, res) => {
    const range = req.headers.range;
    if (req.method === 'HEAD') {
      res.writeHead(200, { 'Content-Length': bytes.length, 'Accept-Ranges': 'bytes', ETag: '"v1"' });
      return res.end();
    }
    if (!range) {
      res.writeHead(200, { 'Content-Length': bytes.length, 'Accept-Ranges': 'bytes', ETag: '"v1"' });
      return res.end(bytes);
    }
    const m = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), bytes.length - 1);
    res.writeHead(206, { 'Content-Length': end - start + 1, 'Content-Range': `bytes ${start}-${end}/${bytes.length}`, 'Accept-Ranges': 'bytes', ETag: '"v1"' });
    res.end(bytes.subarray(start, end + 1));
  };
}

function serveIgnoringRange(bytes) {
  // Never returns 206, even when asked — always 200 with the whole body.
  return (req, res) => {
    if (req.method === 'HEAD') {
      res.writeHead(200, { 'Content-Length': bytes.length });
      return res.end();
    }
    res.writeHead(200, { 'Content-Length': bytes.length });
    res.end(bytes);
  };
}

function serveSwapped(swapOffset) {
  // Correct Range support, but the underlying bytes changed at swapOffset
  // onward — simulating a mutable URL (no CDN immutability, someone
  // overwrote the object) while the caller still pins the original hash.
  // Header/manifest/segment-table bytes (before swapOffset) are served
  // unswapped so the identity pin still passes at mount — the tamper is
  // only in segment content, caught by per-segment digest verification,
  // not by the mount-time identity check alone.
  return (bytes) => (req, res) => {
    const range = req.headers.range;
    const serve = (start, end) => {
      const slice = bytes.subarray(start, end + 1);
      if (end < swapOffset) return slice; // entirely before the tamper point
      const out = Buffer.from(slice);
      for (let i = Math.max(0, swapOffset - start); i < out.length; i++) out[i] ^= 0xff;
      return out;
    };
    if (req.method === 'HEAD') {
      res.writeHead(200, { 'Content-Length': bytes.length, 'Accept-Ranges': 'bytes' });
      return res.end();
    }
    if (!range) {
      res.writeHead(200, { 'Content-Length': bytes.length });
      return res.end(serve(0, bytes.length - 1));
    }
    const m = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), bytes.length - 1);
    res.writeHead(206, { 'Content-Length': end - start + 1, 'Content-Range': `bytes ${start}-${end}/${bytes.length}`, 'Accept-Ranges': 'bytes' });
    res.end(serve(start, end));
  };
}

function serveTruncatedRange(bytes) {
  // Answers 206 but the body is shorter than Content-Range promises —
  // simulating a proxy that cuts the response short.
  return (req, res) => {
    const range = req.headers.range;
    if (req.method === 'HEAD') {
      res.writeHead(200, { 'Content-Length': bytes.length, 'Accept-Ranges': 'bytes' });
      return res.end();
    }
    if (!range) {
      res.writeHead(200, { 'Content-Length': bytes.length });
      return res.end(bytes);
    }
    const m = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), bytes.length - 1);
    const full = bytes.subarray(start, end + 1);
    const truncated = full.subarray(0, Math.max(1, full.length - 16)); // lop off the tail
    res.writeHead(206, { 'Content-Length': end - start + 1, 'Content-Range': `bytes ${start}-${end}/${bytes.length}`, 'Accept-Ranges': 'bytes' });
    res.end(truncated); // fewer bytes than Content-Length claimed
  };
}

async function run(name, setup) {
  console.log('\n' + '-'.repeat(72));
  console.log(name);
  console.log('-'.repeat(72));
  const { server, url, expectedIdentity } = await setup();
  try {
    const source = httpRangeSource(url, { maxRetries: 0 });
    const remote = await openPikeletFile(source, expectedIdentity ? { expectedIdentity } : {});
    const result = await remote.query('Why did Elizabeth Bennet change her mind about Darcy?', { k: 1 });
    console.log(`OUTCOME: mounted and queried successfully (${result.results.length} result(s))`);
    console.log(`  fullFallback (server ignored Range): ${source.stats.fullFallback}`);
    if (source.stats.fullFallback) {
      console.log(`  bytes transferred: whole file (${packBytes.length} bytes) — stats.bytes only tracks range reads, not the fallback download`);
    } else {
      console.log(`  bytes transferred: ${source.stats.bytes}`);
    }
    await remote.close();
  } catch (err) {
    console.log(`OUTCOME: refused / failed loudly — ${err.message}`);
  } finally {
    await server.close();
  }
}

async function main() {
  const { openPikeletFile: local } = await import('../../complete/index.mjs');
  const localSearch = await local(packPath);
  const realIdentity = localSearch.info().identity;
  await localSearch.close();

  await run('1. Server supports Range correctly (baseline — should succeed)', async () => {
    const { url, close } = await startServer(serveRangeCorrectly(packBytes));
    return { server: { close }, url, expectedIdentity: realIdentity };
  });

  await run('2. Server ignores Range, always returns the full file (should fall back safely, not silently)', async () => {
    const { url, close } = await startServer(serveIgnoringRange(packBytes));
    return { server: { close }, url, expectedIdentity: realIdentity };
  });

  await run('3. Bytes at the URL change mid-session while the hash pin stays the original (should refuse)', async () => {
    // Serve the real header/manifest/segment-table bytes (offset < 4096,
    // comfortably past the structural region for any pack size) so the
    // identity pin still matches at mount, then flip every byte in every
    // segment from there on — simulating the underlying object being
    // overwritten (no CDN immutability) after the pin was taken. Per-segment
    // digest verification, not just the mount-time identity check, must
    // catch this.
    const { url, close } = await startServer(serveSwapped(4096)(packBytes));
    return { server: { close }, url, expectedIdentity: realIdentity };
  });

  await run('4. One range response is truncated (should fail loudly, not return wrong bytes)', async () => {
    const { url, close } = await startServer(serveTruncatedRange(packBytes));
    return { server: { close }, url, expectedIdentity: realIdentity };
  });

  console.log('\n' + '='.repeat(72));
  console.log('All four scenarios exercised. See OUTCOME lines above.');
  console.log('='.repeat(72));
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
