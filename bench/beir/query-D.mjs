#!/usr/bin/env node
// Configuration D — Pikelet encoder / affine-u8 HNSW.
//
// Same encoder output as B/C (cached Pikelet-encoder vectors), same
// quantized u8 corpus representation, but searched through the REAL
// PancakeIndex (Pikelet.create({quantized:true}) + addBatch + search) —
// the actual HNSW graph, not the hand-rolled exhaustive scan query-C.mjs
// uses. C -> D isolates approximate-nearest-neighbor loss on top of
// quantization, holding the encoder and corpus representation fixed.
//
// efSearch is read from config.json (frozen before any dataset was run —
// see config.json's notes) and NOT tuned per-dataset.
//
// Usage: node bench/beir/query-D.mjs <dataset>
// Requires: work/<dataset>/vectors-{corpus,queries}-pikelet.f32 (encode-pikelet.mjs)
// Writes: work/<dataset>/run-D.json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import Pikelet from 'pikelet-wasm';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/query-D.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
const config = JSON.parse(readFileSync(path.join(__dirname, 'config.json'), 'utf8'));

const corpusVecPath = path.join(workDir, 'vectors-corpus-pikelet.f32');
const corpusIdsPath = path.join(workDir, 'vectors-corpus-float32.ids.json'); // shared row order/ids
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

console.error(`building PancakeIndex: ${corpus.count} vectors, dim ${dim}, quantized u8, efSearch ${config.efSearch}`);
const buildT0 = performance.now();
const index = await Pikelet.create({
  dim,
  maxElements: corpus.count,
  metric: 'cosine',
  quantized: true,
  efSearch: config.efSearch,
  seed: config.seed,
});

const vectors = new Array(corpus.count);
for (let i = 0; i < corpus.count; i++) {
  vectors[i] = corpus.flat.subarray(i * dim, (i + 1) * dim);
}
const insertedIds = index.addBatch(vectors); // insertion order === corpus row order === corpus.ids order
const buildMs = performance.now() - buildT0;
console.error(`index built in ${(buildMs / 1000).toFixed(1)}s (${index.memory} bytes reported)`);

// Map Pikelet's internal ids back to BEIR ids (insertion order is preserved
// by addBatch, so insertedIds[i] corresponds to corpus.ids[i]).
const pikeletIdToBeir = new Map();
for (let i = 0; i < insertedIds.length; i++) pikeletIdToBeir.set(insertedIds[i], corpus.ids[i]);

const K = 100;
const results = {};
const latencies = [];

for (let qi = 0; qi < queries.count; qi++) {
  const qid = queries.ids[qi];
  if (!testIds.has(qid)) continue;
  const qvec = queries.flat.subarray(qi * dim, (qi + 1) * dim);

  const t0 = performance.now();
  const hits = index.search(qvec, K, { efSearch: config.efSearch });
  latencies.push(performance.now() - t0);

  // PancakeIndex uses distance (lower = closer) for cosine internally per
  // pikelet.d.ts SearchResult; convert to a score where higher = better for
  // consistency with run-{A,B,C}.json's score field. Cosine distance here is
  // 1 - cosine_similarity (standard HNSW convention), so similarity = 1 - distance.
  results[qid] = hits.map((h) => ({
    score: 1 - h.distance,
    beirId: pikeletIdToBeir.get(h.id),
  }));
}

index.dispose();

latencies.sort((a, b) => a - b);
const median = latencies[Math.floor(latencies.length / 2)];
const p95 = latencies[Math.floor(latencies.length * 0.95)];

const out = {
  benchmark: 'BEIR',
  dataset,
  configuration: 'D',
  description: 'Pikelet encoder / affine-u8 HNSW',
  system: {
    node: process.version,
    platform: process.platform,
    build_ms: buildMs,
    ef_search: config.efSearch,
    median_ms: median,
    p95_ms: p95,
    num_queries: latencies.length,
  },
  results,
};

writeFileSync(path.join(workDir, 'run-D.json'), JSON.stringify(out));
console.log(`${dataset} D: ${latencies.length} queries, median ${median.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms -> ${path.join(workDir, 'run-D.json')}`);
