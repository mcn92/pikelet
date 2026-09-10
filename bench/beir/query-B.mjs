#!/usr/bin/env node
// Configuration B — Pikelet encoder / float32 exhaustive.
//
// Isolates the encoder-quantization effect in isolation from corpus
// quantization: queries AND corpus are both encoded with Pikelet's real
// inline WASM MiniLM (same vectors query-C.mjs already produced and cached),
// but searched exhaustively against full float32 corpus vectors — no u8
// quantization at all. A -> B isolates what quantizing MiniLM's weights and
// running it through the tiny WASM runtime costs, decoupled from what
// quantizing the corpus representation costs (that's B -> C).
//
// Usage: node bench/beir/query-B.mjs <dataset>
// Requires: work/<dataset>/vectors-{corpus,queries}-pikelet.f32 already built
//           by query-C.mjs (run that first — it produces and caches these).
// Writes: work/<dataset>/run-B.json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/query-B.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);

const corpusVecPath = path.join(workDir, 'vectors-corpus-pikelet.f32');
const corpusIdsPath = path.join(workDir, 'vectors-corpus-float32.ids.json'); // corpus row order/ids are shared across A/B/C (records.jsonl order)
const queryVecPath = path.join(workDir, 'vectors-queries-pikelet.f32');
const queryIdsPath = path.join(workDir, 'vectors-queries-pikelet.ids.json');

for (const p of [corpusVecPath, corpusIdsPath, queryVecPath, queryIdsPath]) {
  if (!existsSync(p)) {
    console.error(`missing ${p} — run: node bench/beir/query-C.mjs ${dataset} first (it produces and caches the Pikelet-encoder vectors)`);
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

function l2normalize(vec) {
  let s = 0;
  for (let d = 0; d < vec.length; d++) s += vec[d] * vec[d];
  const norm = Math.sqrt(s);
  if (norm < 1e-30) return vec;
  const out = new Float32Array(vec.length);
  for (let d = 0; d < vec.length; d++) out[d] = vec[d] / norm;
  return out;
}

const dim = corpus.dim;
// Pikelet-encoder vectors are already normalized by the kernel (query-C.mjs
// verified norm 1.0), but normalize defensively so cosine == dot regardless.
const corpusNormed = new Float32Array(corpus.flat.length);
for (let i = 0; i < corpus.count; i++) {
  corpusNormed.set(l2normalize(corpus.flat.subarray(i * dim, (i + 1) * dim)), i * dim);
}

const K = 100;
const results = {};
const latencies = [];

for (let qi = 0; qi < queries.count; qi++) {
  const qid = queries.ids[qi];
  if (!testIds.has(qid)) continue;
  const qvec = l2normalize(queries.flat.subarray(qi * dim, (qi + 1) * dim));

  const t0 = performance.now();
  const heap = [];
  for (let ci = 0; ci < corpus.count; ci++) {
    const base = ci * dim;
    let s = 0;
    for (let d = 0; d < dim; d++) s += corpusNormed[base + d] * qvec[d];
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
  configuration: 'B',
  description: 'Pikelet encoder / float32 exhaustive',
  system: {
    node: process.version,
    platform: process.platform,
    median_ms: median,
    p95_ms: p95,
    num_queries: latencies.length,
  },
  results,
};

writeFileSync(path.join(workDir, 'run-B.json'), JSON.stringify(out));
console.log(`${dataset} B: ${latencies.length} queries, median ${median.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms -> ${path.join(workDir, 'run-B.json')}`);
