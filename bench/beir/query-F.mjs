#!/usr/bin/env node
// Configuration F — lexical only (BM25), no vector search at all.
//
// Mirrors complete/index.mjs's retrieval: 'lexical' mode: score comes
// entirely from the BM25 index, vector distance never enters ranking (the
// mode exists in production for measurement/debugging and workloads that
// want pure lexical match). Included alongside E (hybrid) so the ladder
// covers every retrieval mode a real .pikelet artifact actually serves —
// not just the vector-only ladder A-D.
//
// Usage: node bench/beir/query-F.mjs <dataset>
// Requires: work/<dataset>/records.jsonl (convert.mjs — supplies BM25 text)
// Writes: work/<dataset>/run-F.json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { buildLexicalSegment } from '../../complete/builder.mjs';
import { openLexicalIndex } from '../../complete/lexical.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/query-F.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
const corpusIdsPath = path.join(workDir, 'vectors-corpus-float32.ids.json'); // shared row order/ids
const recordsPath = path.join(workDir, 'records.jsonl');

for (const p of [corpusIdsPath, recordsPath]) {
  if (!existsSync(p)) {
    console.error(`missing ${p} — run: node bench/beir/embed.py ${dataset} corpus first (id map only, no vectors needed)`);
    process.exit(1);
  }
}

async function readJsonl(filePath) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (t) rows.push(JSON.parse(t));
  }
  return rows;
}

const corpusIds = JSON.parse(readFileSync(corpusIdsPath, 'utf8')).ids; // row -> beirId
const records = await readJsonl(recordsPath); // pikeletRow order, matches corpusIds order

function testQueryIds() {
  const text = readFileSync(path.join(cacheDir, 'qrels', 'test.tsv'), 'utf8');
  const ids = new Set();
  for (const line of text.split('\n').slice(1)) {
    const [qid] = line.split('\t');
    if (qid) ids.add(qid);
  }
  return ids;
}
const testIds = testQueryIds();
const testQueries = [];
{
  const rl = createInterface({ input: createReadStream(path.join(cacheDir, 'queries.jsonl')), crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (!t) continue;
    const q = JSON.parse(t);
    if (testIds.has(q._id)) testQueries.push(q);
  }
}

console.error(`building lexical (BM25) index over ${records.length} records...`);
const lexicalSegment = buildLexicalSegment(records.map((r) => r.text || ''));
const lexicalIndex = openLexicalIndex(lexicalSegment.bytes);
console.error(`lexical index: ${lexicalSegment.meta.terms.toLocaleString()} terms, ${(lexicalSegment.bytes.length / 1024).toFixed(0)} KiB`);

const K = 100;
const results = {};
const latencies = [];

for (const q of testQueries) {
  const t0 = performance.now();
  const hits = lexicalIndex.search(q.text, K);
  latencies.push(performance.now() - t0);
  results[q._id] = hits.map((h) => ({ score: h.score, beirId: corpusIds[h.id] }));
}

latencies.sort((a, b) => a - b);
const median = latencies[Math.floor(latencies.length / 2)];
const p95 = latencies[Math.floor(latencies.length * 0.95)];

const out = {
  benchmark: 'BEIR',
  dataset,
  configuration: 'F',
  description: 'BM25 lexical only (retrieval: lexical)',
  system: {
    node: process.version,
    platform: process.platform,
    median_ms: median,
    p95_ms: p95,
    num_queries: latencies.length,
  },
  results,
};

writeFileSync(path.join(workDir, 'run-F.json'), JSON.stringify(out));
console.log(`${dataset} F: ${latencies.length} queries, median ${median.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms -> ${path.join(workDir, 'run-F.json')}`);
