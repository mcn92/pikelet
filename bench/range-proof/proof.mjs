#!/usr/bin/env node
// The boring proof: static file hosting + HTTP Range + client-side
// retrieval = no vector DB, no retrieval backend.
//
// Serves a real .pikelet pack from a deliberately dumb static HTTP server
// (fs.createReadStream + Range, nothing else), mounts it remotely through
// Pikelet's real httpRangeSource/openPancakeFile, runs several queries, and
// reports exactly how many bytes and HTTP range requests each step cost —
// next to proof that the server process knows nothing about vectors.
//
// Usage: node bench/range-proof/proof.mjs [path/to/pack.pikelet]

import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { startDumbServer } from './dumb-server.mjs';
import { openPancakeFile, httpRangeSource } from '../../complete/index.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const packPath = path.resolve(process.argv[2] || path.join(ROOT, 'examples/05-one-file-search/pancake-wiki-inline.pancake'));
const packDir = path.dirname(packPath);
const packFile = path.basename(packPath);

// Fallback query sets per known local pack — queries must match what the
// pack actually covers, or results are meaningless (a Pride and Prejudice
// question against a general-knowledge Wikipedia pack finds one summary
// article by luck and nothing else, which proves nothing about retrieval
// quality). The wiki pack's own sampleQueries (below) are preferred when
// present; these are only the fallback for packs that don't declare any.
const FALLBACK_QUERIES = {
  'pride-prejudice.pikelet': [
    'Why did Elizabeth Bennet change her mind about Darcy?',
    'What does Mr. Collins propose to Elizabeth?',
    'How does Mr. Darcy first propose to Elizabeth?',
    'What happens when Lydia runs off with Wickham?',
    'How does Mr. Bennet react to Mr. Collins as his heir?',
  ],
  default: [
    'who was the first person on the moon',
    'how do volcanoes form',
    'what causes earthquakes',
    'how does photosynthesis work',
    'what is the capital of France',
  ],
};

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
}

function snapshot(stats) {
  return { requests: stats.requests, bytes: stats.bytes };
}

function delta(before, after) {
  return { requests: after.requests - before.requests, bytes: after.bytes - before.bytes };
}

async function main() {
  console.log('='.repeat(72));
  console.log('STEP 0 — the pack, sitting on disk');
  console.log('='.repeat(72));
  const fs = await import('node:fs');
  const size = fs.statSync(packPath).size;
  console.log(`$ ls -lh ${path.relative(ROOT, packPath)}`);
  console.log(`${fmtBytes(size)}  ${path.relative(ROOT, packPath)}`);
  console.log();

  console.log('='.repeat(72));
  console.log('STEP 1 — start a deliberately dumb static file server');
  console.log('='.repeat(72));
  const server = await startDumbServer(packDir, { port: 0, logToStdout: false });
  console.log(`$ node bench/range-proof/dumb-server.mjs ${path.relative(ROOT, packDir)}`);
  console.log(`serving ${path.relative(ROOT, packDir)} on ${server.url}`);
  console.log('(fs.createReadStream + Range support only — no database process, no index, no query logic)');
  console.log();
  console.log('Proof the server is dumb — no retrieval backend is running:');
  try {
    const psOut = execSync("ps aux | grep -E 'qdrant|weaviate|postgres|pinecone|pikelet-server|elasticsearch|milvus' | grep -v grep || true").toString().trim();
    console.log('$ ps aux | grep -E \'qdrant|weaviate|postgres|pinecone|pikelet-server\'');
    console.log(psOut.length ? psOut : '(nothing)');
  } catch {
    console.log('(nothing)');
  }
  console.log();

  console.log('='.repeat(72));
  console.log('STEP 2 — mount the pack remotely, with the identity pinned');
  console.log('='.repeat(72));

  // Read the real identity from the local file once so the pin is genuine,
  // not asserted — a mismatched pin must refuse to mount. Also pulls the
  // pack's own sampleQueries when it declares any: those are guaranteed to
  // match what the pack actually covers, unlike a hardcoded guess.
  const localSearch = await openPancakeFile(packPath);
  const localInfo = localSearch.info();
  const identity = localInfo.identity;
  await localSearch.close();

  const QUERIES = localInfo.sampleQueries.length >= 3
    ? localInfo.sampleQueries
    : (FALLBACK_QUERIES[packFile] || FALLBACK_QUERIES.default);

  const url = `${server.url}/${encodeURIComponent(packFile)}`;
  console.log(`$ pikelet mcp ${url}#${identity}`);
  console.log();

  // prefetchEncoder: false isolates "mount" (structural reads: header,
  // manifest, segment table, resident sketch, lexical index) from the
  // ~25 MiB query-encoder model weights, which this pack carries inline
  // and which download lazily on the first query instead. Without this
  // flag the two costs blur together under "mount," which understates how
  // little a plain remount/re-open costs once the encoder is cached.
  const source = httpRangeSource(url);
  const beforeMount = snapshot(source.stats);
  const remote = await openPancakeFile(source, { expectedIdentity: identity, prefetchEncoder: false });
  const afterMount = snapshot(source.stats);
  const mountDelta = delta(beforeMount, afterMount);
  const info = remote.info();

  console.log('pack:', info.name || packFile);
  console.log('size:', fmtBytes(info.fileBytes));
  console.log('source: HTTP');
  console.log('identity: verified (pinned hash matched)');
  console.log('resident fetch (mount only):', fmtBytes(mountDelta.bytes));
  console.log('full artifact download: no');
  console.log();
  console.log('Note: this pack carries its own query encoder (MiniLM weights,');
  console.log('~25 MiB) inline for self-contained operation. It is NOT fetched at');
  console.log('mount — it downloads lazily on the first query and is then cached');
  console.log('for the process lifetime. That one-time cost shows up in "Query 1"');
  console.log('below, separated from ongoing per-query retrieval cost.');
  console.log();

  console.log('='.repeat(72));
  console.log(`STEP 3 — run ${QUERIES.length} queries`);
  console.log('='.repeat(72));

  const rows = [
    { operation: 'Initial mount', bytes: mountDelta.bytes, requests: mountDelta.requests },
  ];

  for (let i = 0; i < QUERIES.length; i++) {
    const q = QUERIES[i];
    const before = snapshot(source.stats);
    console.log(`\n$ pikelet query "${q}"`);
    const result = await remote.query(q, { k: 3 });
    const after = snapshot(source.stats);
    const d = delta(before, after);
    console.log('results:');
    for (const r of result.results) {
      console.log(`  record ${r.id}${r.title ? `  (${r.title})` : ''}`);
    }
    console.log(`network: ${d.requests} range requests, ${fmtBytes(d.bytes)} transferred`);
    if (i === 0) {
      console.log('  (includes the one-time ~25 MiB encoder-weights fetch, cached from here on)');
    }
    rows.push({ operation: `Query ${i + 1}${i === 0 ? ' (+ encoder load)' : ''}`, bytes: d.bytes, requests: d.requests });
  }

  console.log();
  console.log('='.repeat(72));
  console.log('STEP 4 — repeat one query (cache effect)');
  console.log('='.repeat(72));
  const repeatQuery = QUERIES[0];
  const beforeRepeat = snapshot(source.stats);
  console.log(`\n$ pikelet query "${repeatQuery}"   # same as query 1`);
  await remote.query(repeatQuery, { k: 3 });
  const afterRepeat = snapshot(source.stats);
  const repeatDelta = delta(beforeRepeat, afterRepeat);
  console.log(`network: ${repeatDelta.requests} range requests, ${fmtBytes(repeatDelta.bytes)} transferred (mostly cached in-process)`);
  rows.push({ operation: 'Repeated query 1', bytes: repeatDelta.bytes, requests: repeatDelta.requests });

  console.log();
  console.log('='.repeat(72));
  console.log('SUMMARY');
  console.log('='.repeat(72));
  const totals = snapshot(source.stats);
  console.log();
  const nameWidth = Math.max(20, ...rows.map((r) => r.operation.length));
  console.log(`${'Operation'.padEnd(nameWidth)}  Bytes transferred    Requests`);
  for (const r of rows) {
    console.log(`${r.operation.padEnd(nameWidth)}  ${fmtBytes(r.bytes).padStart(15)}    ${String(r.requests).padStart(4)}`);
  }
  console.log();
  console.log(`Total bytes transferred over the wire: ${fmtBytes(totals.bytes)} (of ${fmtBytes(info.fileBytes)} on disk)`);
  console.log(`Total HTTP range requests: ${totals.requests}`);
  console.log(`Full artifact download at any point: ${source.stats.fullFallback ? 'YES (server ignored Range!)' : 'no'}`);
  console.log();
  console.log(`The ${fmtBytes(info.fileBytes)} knowledge base is sitting on a static file server.`);
  console.log('The server did no retrieval. It served byte ranges from a file on disk.');
  console.log('All retrieval (embedding, HNSW search, reranking, abstention) ran client-side.');

  await remote.close();
  await server.close();
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
