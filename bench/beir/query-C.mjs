#!/usr/bin/env node
// Configuration C — Pikelet encoder / affine-u8 exhaustive.
//
// Corpus vectors are quantized with Pikelet's exact row-wise affine u8
// scheme (src/uint8_float_hnsw.hpp:206-234; reimplemented in quantize.mjs
// since the engine exposes no readback API for stored rows) and searched
// exhaustively — no HNSW, no approximation. This isolates what corpus
// quantization costs, holding the encoder fixed at B's (Pikelet's) output.
//
// Usage: node bench/beir/query-C.mjs <dataset>
// Requires: work/<dataset>/vectors-{corpus,queries}-pikelet.f32 already
//           built by encode-pikelet.mjs (run that first).
// Writes: work/<dataset>/run-C.json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { l2normalize, quantizeRow, dotQueryAgainstQuantized, sum } from './quantize.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/query-C.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);

const corpusVecPath = path.join(workDir, 'vectors-corpus-pikelet.f32');
const corpusIdsPath = path.join(workDir, 'vectors-corpus-float32.ids.json'); // shared row order/ids (records.jsonl order)
const queryVecPath = path.join(workDir, 'vectors-queries-pikelet.f32');
const queryIdsPath = path.join(workDir, 'vectors-queries-pikelet.ids.json');

for (const p of [corpusVecPath, corpusIdsPath, queryVecPath, queryIdsPath]) {
  if (!existsSync(p)) {
    console.error(`missing ${p} — run: node bench/beir/encode-pikelet.mjs ${dataset} first`);
    process.exit(1);
  }
}

function loadVectors(vecPath, idsPath) {
  const meta = JSON.parse(readFileSync(idsPath, 'utf8'));
  const buf = readFileSync(vecPath);
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const dim = meta.dim;
  const count = flat.length / dim;
  if (!Number.isInteger(count)) throw new Error(`${vecPath}: ${flat.length} floats not divisible by dim ${dim}`);
  return { ids: meta.ids, dim, flat, count };
}

const corpus = loadVectors(corpusVecPath, corpusIdsPath);
const queries = loadVectors(queryVecPath, queryIdsPath);

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

const dim = corpus.dim;

// Quantize corpus rows once (cosine metric: normalize, then affine-u8
// quantize — src/uint8_float_hnsw.hpp:198-234).
const quantized = new Array(corpus.count);
for (let i = 0; i < corpus.count; i++) {
  const raw = corpus.flat.subarray(i * dim, (i + 1) * dim);
  quantized[i] = quantizeRow(l2normalize(raw));
}

const K = 100;
const results = {};
const latencies = [];

for (let qi = 0; qi < queries.count; qi++) {
  const qid = queries.ids[qi];
  if (!testIds.has(qid)) continue;
  const qvec = l2normalize(queries.flat.subarray(qi * dim, (qi + 1) * dim));
  const sumQ = sum(qvec);

  const t0 = performance.now();
  const heap = [];
  for (let ci = 0; ci < corpus.count; ci++) {
    const s = dotQueryAgainstQuantized(qvec, sumQ, quantized[ci]);
    if (heap.length < K) {
      heap.push({ score: s, beirId: corpus.ids[ci] });
      if (heap.length === K) heap.sort((a, b) => a.score - b.score);
    } else if (s > heap[0].score) {
      heap[0] = { score: s, beirId: corpus.ids[ci] };
      let i = 0;
      while (i + 1 < K && heap[i].score > heap[i + 1].score) {
        [heap[i], heap[i + 1]] = [heap[i + 1], heap[i]];
        i++;
      }
    }
  }
  latencies.push(performance.now() - t0);
  heap.sort((a, b) => b.score - a.score);
  results[qid] = heap;
}

latencies.sort((a, b) => a - b);
const median = latencies[Math.floor(latencies.length / 2)];
const p95 = latencies[Math.floor(latencies.length * 0.95)];

const out = {
  benchmark: 'BEIR',
  dataset,
  configuration: 'C',
  description: 'Pikelet encoder / affine-u8 exhaustive',
  system: {
    node: process.version,
    platform: process.platform,
    median_ms: median,
    p95_ms: p95,
    num_queries: latencies.length,
  },
  results,
};

writeFileSync(path.join(workDir, 'run-C.json'), JSON.stringify(out));
console.log(`${dataset} C: ${latencies.length} queries, median ${median.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms -> ${path.join(workDir, 'run-C.json')}`);
