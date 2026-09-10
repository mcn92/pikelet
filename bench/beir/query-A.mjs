#!/usr/bin/env node
// Configuration A — upstream MiniLM float32 / exhaustive.
//
// No Pikelet involvement at all: brute-force cosine search over the plain
// float32 vectors embed.py produced from the same base MiniLM model
// (sentence-transformers/all-MiniLM-L6-v2). This is the quality ceiling
// every other configuration on the ablation ladder is compared against.
//
// Usage: node bench/beir/query-A.mjs <dataset>
// Reads:  work/<dataset>/vectors-{corpus,queries}-float32.{f32,ids.json}
//         cache/<dataset>/qrels/test.tsv (to restrict to the test split)
// Writes: work/<dataset>/run-A.json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/query-A.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);

function loadVectors(prefix) {
  const idsPath = path.join(workDir, `vectors-${prefix}-float32.ids.json`);
  const vecPath = path.join(workDir, `vectors-${prefix}-float32.f32`);
  if (!existsSync(idsPath) || !existsSync(vecPath)) {
    console.error(`missing ${idsPath} or ${vecPath} — run: python3.11 bench/beir/embed.py ${dataset} ${prefix}`);
    process.exit(1);
  }
  const meta = JSON.parse(readFileSync(idsPath, 'utf8'));
  const buf = readFileSync(vecPath);
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (flat.length !== meta.count * meta.dim) {
    throw new Error(`${vecPath}: expected ${meta.count * meta.dim} floats, got ${flat.length}`);
  }
  return { ids: meta.ids, dim: meta.dim, flat, count: meta.count };
}

function testQueryIds() {
  const qrelsPath = path.join(cacheDir, 'qrels', 'test.tsv');
  const text = readFileSync(qrelsPath, 'utf8');
  const lines = text.split('\n').slice(1); // header
  const ids = new Set();
  for (const line of lines) {
    const [qid] = line.split('\t');
    if (qid) ids.add(qid);
  }
  return ids;
}

const corpus = loadVectors('corpus');
const queries = loadVectors('queries');
const testIds = testQueryIds();

// Vectors are already L2-normalized (embed.py) so dot product == cosine.
function dot(flat, base, dim, q) {
  let s = 0;
  for (let d = 0; d < dim; d++) s += flat[base + d] * q[d];
  return s;
}

const K = 100; // enough for recall@100; nDCG@10/recall@10 are computed from a prefix
const dim = corpus.dim;
const results = {};
const latencies = [];

for (let qi = 0; qi < queries.count; qi++) {
  const qid = queries.ids[qi];
  if (!testIds.has(qid)) continue; // score only the official test split
  const qbase = qi * dim;
  const qvec = queries.flat.subarray(qbase, qbase + dim);

  const t0 = performance.now();
  // Partial top-K via a simple bounded insertion — corpus sizes here (BEIR
  // launch subset) are small enough (<1M) that this is fast and exact.
  const heap = []; // [{score, beirId}], kept sorted ascending, capped at K
  for (let ci = 0; ci < corpus.count; ci++) {
    const s = dot(corpus.flat, ci * dim, dim, qvec);
    if (heap.length < K) {
      heap.push({ score: s, beirId: corpus.ids[ci] });
      if (heap.length === K) heap.sort((a, b) => a.score - b.score);
    } else if (s > heap[0].score) {
      heap[0] = { score: s, beirId: corpus.ids[ci] };
      // re-sift the smallest to the front
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
  configuration: 'A',
  description: 'upstream MiniLM float32 / exhaustive',
  system: {
    node: process.version,
    platform: process.platform,
    median_ms: median,
    p95_ms: p95,
    num_queries: latencies.length,
  },
  results,
};

writeFileSync(path.join(workDir, 'run-A.json'), JSON.stringify(out));
console.log(`${dataset} A: ${latencies.length} queries, median ${median.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms -> ${path.join(workDir, 'run-A.json')}`);
