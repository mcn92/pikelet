#!/usr/bin/env node
// Configuration E — Pikelet encoder / affine-u8 sketch artifact + BM25,
// fused by reciprocal rank (hybrid RRF — the production default).
//
// The BEIR ladder (A-D) is dense-only: every prior configuration measures
// vector search alone, never what a real .pikelet artifact serves by
// default. complete/index.mjs's query() runs hybrid RRF unless a caller
// opts out (retrieval: 'vector'|'lexical'|'augmented'), and that fusion
// only exists on the sketch-artifact reader (PikeletSketchArtifact —
// extraCandidates/fullRerankOutput are not available on the raw in-memory
// PikeletIndex config D queries). This configuration builds the real
// sketch artifact from the same snapshot D would build (index.export()) and
// drives it through the actual reader class, reproducing the exact fusion
// complete/index.mjs runs: BM25 hits within LEXICAL_CUTOFF of the top
// lexical score join the exact rerank as extraCandidates (so every fused
// candidate carries a true, not synthesized, vector distance), then RRF
// (RRF_K=60) fuses the reranked vector order with the BM25 order.
//
// Usage: node bench/beir/query-E.mjs <dataset>
// Requires: work/<dataset>/vectors-{corpus,queries}-pikelet.f32 (encode-pikelet.mjs)
//           work/<dataset>/records.jsonl (convert.mjs — supplies BM25 text)
// Writes: work/<dataset>/run-E.json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import Pikelet from 'pikelet-wasm';
import { buildSketchArtifactBytes } from '../../pikelet-artifact-sketch.js';
import { PikeletSketchArtifact } from '../../pikelet-artifact.js';
import { buildLexicalSegment } from '../../complete/builder.mjs';
import { openLexicalIndex } from '../../complete/lexical.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/query-E.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
const config = JSON.parse(readFileSync(path.join(__dirname, 'config.json'), 'utf8'));

// Kept identical to complete/index.mjs's constants — a benchmark using its
// own tuned values would measure a fusion the reader never actually serves.
const RRF_K = 60;
const LEXICAL_CUTOFF = 1.5;
const LEXICAL_CANDIDATES = 24;

const corpusVecPath = path.join(workDir, 'vectors-corpus-pikelet.f32');
const corpusIdsPath = path.join(workDir, 'vectors-corpus-float32.ids.json');
const queryVecPath = path.join(workDir, 'vectors-queries-pikelet.f32');
const queryIdsPath = path.join(workDir, 'vectors-queries-pikelet.ids.json');
const recordsPath = path.join(workDir, 'records.jsonl');

for (const p of [corpusVecPath, corpusIdsPath, queryVecPath, queryIdsPath, recordsPath]) {
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

async function readJsonl(filePath) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (t) rows.push(JSON.parse(t));
  }
  return rows;
}

const corpus = loadVectors(corpusVecPath, corpusIdsPath);
const queries = loadVectors(queryVecPath, queryIdsPath);
const records = await readJsonl(recordsPath); // pikeletRow order, matches corpus.ids order

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
const queryTextById = new Map();
{
  const rl = createInterface({ input: createReadStream(path.join(cacheDir, 'queries.jsonl')), crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (!t) continue;
    const q = JSON.parse(t);
    queryTextById.set(q._id, q.text);
  }
}

const dim = corpus.dim;

console.error(`building lexical (BM25) index over ${records.length} records...`);
const lexicalSegment = buildLexicalSegment(records.map((r) => r.text || ''));
const lexicalIndex = openLexicalIndex(lexicalSegment.bytes);
console.error(`lexical index: ${lexicalSegment.meta.terms.toLocaleString()} terms, ${(lexicalSegment.bytes.length / 1024).toFixed(0)} KiB`);

console.error(`building PikeletIndex snapshot: ${corpus.count} vectors, dim ${dim}, quantized u8`);
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
index.addBatch(vectors); // insertion order === corpus row order === corpus.ids order === records order
const snapshotBytes = index.export();
index.dispose();

// The real production wrapping step (pikelet/src/complete-build.mjs): a
// PikeletIndex export snapshot -> a sketch artifact byte layout, addressed
// by plain row position (0..count-1) — no id indirection, unlike the raw
// PikeletIndex object's own insertion ids (query-D.mjs's pikeletIdToBeir
// map exists only because that script queries the raw index instead).
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
  const qtext = queryTextById.get(qid) || '';

  const t0 = performance.now();
  const lexicalRaw = lexicalIndex.search(qtext, LEXICAL_CANDIDATES);
  const lexicalHits = lexicalRaw.length ? lexicalRaw.filter((h) => h.score >= lexicalRaw[0].score / LEXICAL_CUTOFF) : [];

  // Same shape as complete/index.mjs: BM25 hits join the exact rerank as
  // extraCandidates (scored by true distance like any candidate), then RRF
  // fuses the reranked vector order (searched, distance-sorted) with the
  // BM25 order.
  const searched = (await sketch.search(qvec, K, {
    rerank: config.candidateCount,
    ...(lexicalHits.length ? { extraCandidates: lexicalHits.map((h) => h.id) } : {}),
    fullRerankOutput: true,
  })).results;

  let fused;
  if (lexicalHits.length) {
    const lexRank = new Map(lexicalHits.map((h, i) => [h.id, i]));
    fused = searched
      .map((hit, vRank) => ({
        hit,
        score: 1 / (RRF_K + vRank) + (lexRank.has(hit.id) ? 1 / (RRF_K + lexRank.get(hit.id)) : 0),
      }))
      .sort((a, b) => (b.score - a.score) || (a.hit.distance - b.hit.distance))
      .map((entry) => entry.hit);
  } else {
    fused = searched;
  }
  latencies.push(performance.now() - t0);

  results[qid] = fused.slice(0, K).map((h) => ({
    score: 1 - h.distance,
    beirId: corpus.ids[h.id],
  }));
}

latencies.sort((a, b) => a - b);
const median = latencies[Math.floor(latencies.length / 2)];
const p95 = latencies[Math.floor(latencies.length * 0.95)];

const out = {
  benchmark: 'BEIR',
  dataset,
  configuration: 'E',
  description: 'Pikelet encoder / affine-u8 sketch artifact + BM25 (hybrid RRF, production default)',
  system: {
    node: process.version,
    platform: process.platform,
    build_ms: buildMs,
    rerank: config.candidateCount,
    rrf_k: RRF_K,
    lexical_cutoff: LEXICAL_CUTOFF,
    median_ms: median,
    p95_ms: p95,
    num_queries: latencies.length,
  },
  results,
};

writeFileSync(path.join(workDir, 'run-E.json'), JSON.stringify(out));
console.log(`${dataset} E: ${latencies.length} queries, median ${median.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms -> ${path.join(workDir, 'run-E.json')}`);
