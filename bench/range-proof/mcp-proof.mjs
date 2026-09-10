#!/usr/bin/env node
// The MCP half of the boring proof: mount a .pikelet pack over HTTP into
// the real `pikelet mcp` server, then drive it with the repo's own tested
// MCP client (examples/06-mcp-knowledge-pack/mcp_client.mjs) over stdio
// JSON-RPC — not by importing the reader library directly. Shows the "ask
// agent question -> answer + cited records" shape end to end, plus the
// dumb server's own Range-request log as independent proof of what
// actually went over the wire.
//
// Usage: node bench/range-proof/mcp-proof.mjs [path/to/pack.pikelet]

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startDumbServer } from './dumb-server.mjs';
import { openPancakeFile } from '../../complete/index.mjs';
import { PikeletMcpClient } from '../../examples/06-mcp-knowledge-pack/mcp_client.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const packPath = path.resolve(process.argv[2] || path.join(ROOT, 'examples/05-one-file-search/pancake-wiki-inline.pancake'));
const packDir = path.dirname(packPath);
const packFile = path.basename(packPath);

const QUERIES = [
  'who was the first person on the moon',
  'how do volcanoes form',
  'what causes earthquakes',
  'how does photosynthesis work',
  'what is the capital of France',
];

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
}

async function main() {
  console.log('='.repeat(72));
  console.log('Start a deliberately dumb static file server');
  console.log('='.repeat(72));
  const server = await startDumbServer(packDir, { port: 0, logToStdout: false });
  console.log(`serving ${path.relative(ROOT, packDir)} on ${server.url}`);
  console.log('(fs.createReadStream + Range support only — no database, no index, no query logic)');
  console.log();

  const localSearch = await openPancakeFile(packPath);
  const identity = localSearch.info().identity;
  await localSearch.close();

  const url = `${server.url}/${encodeURIComponent(packFile)}#${identity}`;

  console.log('='.repeat(72));
  console.log('Start the real pikelet mcp server, mounted at that URL');
  console.log('='.repeat(72));
  console.log(`$ pikelet mcp ${url}`);
  console.log();
  const mcp = new PikeletMcpClient({ packs: [url] });
  await mcp.initialize();

  const { packs } = await mcp.call('list_packs');
  console.log('list_packs ->');
  for (const p of packs) {
    console.log(`  ${p.name}  records=${p.records}  identity=${p.identity.slice(0, 12)}...`);
  }
  console.log();

  const rows = [];
  for (let i = 0; i < QUERIES.length; i++) {
    const q = QUERIES[i];
    console.log('='.repeat(72));
    console.log(`ask agent: "${q}"`);
    console.log('='.repeat(72));
    const before = server.log.length;
    const t0 = Date.now();
    const result = await mcp.call('search', { query: q, k: 3 });
    const ms = Date.now() - t0;
    const after = server.log.length;
    const requestCount = after - before;
    const bytes = server.log.slice(before, after).reduce((sum, entry) => {
      const m = entry.range && /^bytes=(\d+)-(\d+)$/.exec(entry.range);
      return sum + (m ? Number(m[2]) - Number(m[1]) + 1 : 0);
    }, 0);
    const section = Array.isArray(result?.results) ? result : result?.sections?.[0];
    console.log(`matchQuality: ${section?.matchQuality}  confidence: ${section?.confidence?.toFixed(3)}`);
    for (const r of section?.results || []) {
      console.log(`  - record ${r.id}  (${r.title})  distance ${r.distance?.toFixed(4)}`);
      console.log(`    ${r.text.slice(0, 140).replace(/\n/g, ' ')}...`);
    }
    console.log(`network: ${requestCount} range requests, ${fmtBytes(bytes)} transferred, ${ms} ms`);
    console.log();
    rows.push({ query: q, requests: requestCount, bytes, ms });
  }

  console.log('='.repeat(72));
  console.log(`ask agent (repeat): "${QUERIES[0]}"`);
  console.log('='.repeat(72));
  const beforeRepeat = server.log.length;
  const t0 = Date.now();
  await mcp.call('search', { query: QUERIES[0], k: 3 });
  const repeatMs = Date.now() - t0;
  const repeatRequests = server.log.length - beforeRepeat;
  console.log(`network: ${repeatRequests} range requests, ${repeatMs} ms (mostly cached in-process)`);
  console.log();

  console.log('='.repeat(72));
  console.log('SUMMARY');
  console.log('='.repeat(72));
  console.log(`${'Query'.padEnd(38)}  Requests  Bytes         ms`);
  for (const r of rows) {
    console.log(`${r.query.slice(0, 38).padEnd(38)}  ${String(r.requests).padStart(8)}  ${fmtBytes(r.bytes).padStart(11)}  ${String(r.ms).padStart(4)}`);
  }
  console.log(`${'(repeat) ' + QUERIES[0]}`.slice(0, 38).padEnd(38) + `  ${String(repeatRequests).padStart(8)}  ${'-'.padStart(11)}  ${String(repeatMs).padStart(4)}`);
  console.log();
  console.log(`Total dumb-server requests across the whole session: ${server.log.length}`);
  console.log('The server that answered every one of those requests only knows how to read byte ranges from a file.');

  mcp.close();
  await server.close();
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
