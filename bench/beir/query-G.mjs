#!/usr/bin/env node
// Configuration G — Arctic-XS encoder / affine-u8 sketch artifact (dense
// only), directly parallel to D (which is MiniLM / affine-u8 HNSW) so the
// encoder question is isolated on identical retrieval logic. Reads the
// arctic corpus/query vectors produced by encode-pikelet-arctic.mjs.
//
// Pass --no-prefix to score the un-prefixed query encoding
// (vectors-queries-arctic-noprefix.f32) instead — the query-prefix
// ablation this README's "prefix verified" claim is measured from.
//
// Usage: node bench/beir/query-G.mjs <dataset> [--no-prefix]
// Requires: work/<dataset>/vectors-corpus-arctic.f32,
//           work/<dataset>/vectors-queries-arctic[-noprefix].f32
//           (encode-pikelet-arctic.mjs)
// Writes: work/<dataset>/run-G[-noprefix].json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import Pikelet from 'pikelet-wasm';
import { buildSketchArtifactBytes } from '../../pikelet-artifact-sketch.js';
import { PikeletSketchArtifact } from '../../pikelet-artifact.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
const noPrefix = process.argv.includes('--no-prefix');
if (!dataset) {
  console.error('usage: node bench/beir/query-G.mjs <dataset> [--no-prefix]');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
const config = JSON.parse(readFileSync(path.join(__dirname, 'config.json'), 'utf8'));

const suffix = noPrefix ? '-noprefix' : '';
const corpusVecPath = path.join(workDir, 'vectors-corpus-arctic.f32');
const corpusIdsPath = path.join(workDir, 'vectors-corpus-float32.ids.json'); // shared row order/ids
const queryVecPath = path.join(workDir, `vectors-queries-arctic${suffix}.f32`);
const queryIdsPath = path.join(workDir, `vectors-queries-arctic${suffix}.ids.json`);

for (const p of [corpusVecPath, corpusIdsPath, queryVecPath, queryIdsPath]) {
  if (!existsSync(p)) {
    console.error(`missing ${p} — run: node bench/beir/encode-pikelet-arctic.mjs ${dataset} ${noPrefix ? '--no-prefix' : ''} first`);
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

console.error(`building PikeletIndex snapshot: ${corpus.count} vectors, dim ${dim}, quantized u8 (arctic-xs${suffix})`);
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
for (let i = 0; i < corpus.count; i++) vectors[i] = corpus.flat.subarray(i * dim, (i + 1) * dim);
index.addBatch(vectors);
const snapshotBytes = index.export();
index.dispose();

const { bytes: sketchBytes } = buildSketchArtifactBytes(snapshotBytes, {
  recommendedRerank: config.candidateCount,
});
const buildMs = performance.now() - buildT0;
console.error(`sketch artifact built in ${(buildMs / 1000).toFixed(1)}s (${sketchBytes.length} bytes)`);

function memorySource(bytes) {
  return {
    size: bytes.length,
    async read(offset, length) {
      return bytes.subarray(offset, offset + length);
    },
  };
}
const sketch = await PikeletSketchArtifact.open(memorySource(sketchBytes));

const K = 100;
const results = {};
const latencies = [];

for (let qi = 0; qi < queries.count; qi++) {
  const qid = queries.ids[qi];
  if (!testIds.has(qid)) continue;
  const qvec = queries.flat.subarray(qi * dim, (qi + 1) * dim);

  const t0 = performance.now();
  const searched = (await sketch.search(qvec, K, { rerank: config.candidateCount })).results;
  latencies.push(performance.now() - t0);

  results[qid] = searched.map((h) => ({ score: 1 - h.distance, beirId: corpus.ids[h.id] }));
}

latencies.sort((a, b) => a - b);
const median = latencies[Math.floor(latencies.length / 2)];
const p95 = latencies[Math.floor(latencies.length * 0.95)];

const configuration = `G${suffix}`;
const out = {
  benchmark: 'BEIR',
  dataset,
  configuration,
  description: `Arctic-XS encoder / affine-u8 sketch artifact (dense only)${noPrefix ? ' — NO QUERY PREFIX (ablation)' : ' — query prefix applied'}`,
  system: {
    node: process.version,
    platform: process.platform,
    build_ms: buildMs,
    rerank: config.candidateCount,
    median_ms: median,
    p95_ms: p95,
    num_queries: latencies.length,
  },
  results,
};

writeFileSync(path.join(workDir, `run-${configuration}.json`), JSON.stringify(out));
console.log(`${dataset} ${configuration}: ${latencies.length} queries, median ${median.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms -> ${path.join(workDir, `run-${configuration}.json`)}`);
