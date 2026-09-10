#!/usr/bin/env node
// Encode a BEIR dataset's corpus + test queries with Pikelet's real inline
// WASM MiniLM encoder, using the SAME code path `pikelet compile` uses in
// production — not a hand-rolled sequential loop.
//
// Corpus: embedChunksWithInlineTransformer() from pikelet/src/embed.mjs,
// which dispatches to a worker_threads pool (up to min(cores-1, 8) workers,
// one kernel+weight-blob instance each) once the corpus exceeds 32 chunks —
// exactly what a real `pikelet compile` build does for passage embedding
// (see embed.mjs:105-123 for the threshold/parallelism comment). An earlier
// version of this script ran a single-threaded loop through
// createInlineTransformerEmbedder directly, which is only Pikelet's
// *fallback* path (used when the worker pool is unavailable) — that
// understated production build throughput by roughly the worker count.
//
// Queries: single embed() calls, one at a time — this DOES match
// production, since a live reader's query() method embeds one query per
// request through the same kernel (complete/index.mjs, ensureEmbedder()).
// There is no query-side batching to reuse in production, so none is added
// here.
//
// Usage: node bench/beir/encode-pikelet.mjs <dataset>
// Reads:  work/<dataset>/records.jsonl, cache/<dataset>/queries.jsonl,
//         cache/<dataset>/qrels/test.tsv
// Writes: work/<dataset>/vectors-corpus-pikelet.f32
//         work/<dataset>/vectors-queries-pikelet.f32 (+ .ids.json)
//         work/<dataset>/encode-pikelet.json (timing + worker count)

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { embedChunksWithInlineTransformer } from '../../pikelet/src/embed.mjs';
import { createInlineTransformerEmbedder } from '../../complete/inline-transformer.mjs';
import createEncoder from '../../complete/encoder-kernels/encoder.node.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/encode-pikelet.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
const repoRoot = path.join(__dirname, '..', '..');

async function readJsonl(filePath) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (t) rows.push(JSON.parse(t));
  }
  return rows;
}

function testQueryIds() {
  const text = readFileSync(path.join(cacheDir, 'qrels', 'test.tsv'), 'utf8');
  const ids = new Set();
  for (const line of text.split('\n').slice(1)) {
    const [qid] = line.split('\t');
    if (qid) ids.add(qid);
  }
  return ids;
}

const records = await readJsonl(path.join(workDir, 'records.jsonl')); // {_id, title, text}, pikeletRow order
const queries = await readJsonl(path.join(cacheDir, 'queries.jsonl'));
const testIds = testQueryIds();
const testQueries = queries.filter((q) => testIds.has(q._id));

const encoderDir = path.join(repoRoot, 'pikelet', 'src', 'inline-encoder');
const config = {
  embedding: { dims: 384, mode: 'inline-transformer', pooling: 'mean', normalize: true },
  runtime: {
    inlineEncoder: {
      vocabPath: path.join(encoderDir, 'vocab.txt'),
      weightsPath: path.join(encoderDir, 'encoder-weights.bin'),
    },
  },
};

const cores = typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;
const expectedWorkers = records.length < 32 ? 1 : Math.max(1, Math.min(cores - 1, 8, records.length));
console.error(`encoding ${records.length} corpus records via production embedChunksWithInlineTransformer (expect ${expectedWorkers} workers)...`);

const t0 = performance.now();
const corpusChunks = records.map((r) => ({ text: r.text }));
const corpusVectors = await embedChunksWithInlineTransformer(corpusChunks, config, (msg) => console.error(msg), repoRoot);
const corpusMs = performance.now() - t0;

const dim = 384;
const corpusFlat = new Float32Array(records.length * dim);
for (let i = 0; i < records.length; i++) corpusFlat.set(corpusVectors[i], i * dim);

// Queries: single embed() calls through the same kernel, matching how a live
// reader embeds one query per request (complete/index.mjs query()) — no
// production batching exists on this path, so none is added here.
console.error(`encoding ${testQueries.length} test queries (single-embed, matches runtime query path)...`);
const vocabText = readFileSync(path.join(encoderDir, 'vocab.txt'), 'utf8');
const blob = readFileSync(path.join(encoderDir, 'encoder-weights.bin'));
const declaration = {
  kind: 'inline-transformer-v1',
  model: 'sentence-transformers/all-MiniLM-L6-v2',
  license: 'apache-2.0',
  attribution: 'sentence-transformers/all-MiniLM-L6-v2 (quantized derivative)',
  dim: 384,
  pooling: 'mean',
  normalized: true,
  maxTokens: 128,
  longInputs: 'windowed-mean-pool',
  layout: { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 6, B: 64, H: 12 },
};
const embedder = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder, verify: false });

const t1 = performance.now();
const queryFlat = new Float32Array(testQueries.length * dim);
for (let i = 0; i < testQueries.length; i++) {
  const { vector } = await embedder.embed(testQueries[i].text);
  queryFlat.set(vector, i * dim);
  if (i % 100 === 0) console.error(`queries ${i}/${testQueries.length}`);
}
const queryMs = performance.now() - t1;
embedder.dispose();

writeFileSync(path.join(workDir, 'vectors-corpus-pikelet.f32'), Buffer.from(corpusFlat.buffer, corpusFlat.byteOffset, corpusFlat.byteLength));
writeFileSync(path.join(workDir, 'vectors-queries-pikelet.f32'), Buffer.from(queryFlat.buffer, queryFlat.byteOffset, queryFlat.byteLength));
writeFileSync(path.join(workDir, 'vectors-queries-pikelet.ids.json'), JSON.stringify({ ids: testQueries.map((q) => q._id), dim: 384, count: testQueries.length }));
writeFileSync(path.join(workDir, 'encode-pikelet.json'), JSON.stringify({
  dataset,
  corpus_count: records.length,
  corpus_encode_ms: corpusMs,
  corpus_workers_expected: expectedWorkers,
  query_count: testQueries.length,
  query_encode_ms: queryMs,
  cores,
  node: process.version,
  platform: process.platform,
}, null, 2));

console.log(`${dataset}: corpus ${records.length} in ${(corpusMs / 1000).toFixed(1)}s (~${expectedWorkers} workers), queries ${testQueries.length} in ${(queryMs / 1000).toFixed(1)}s (sequential)`);
