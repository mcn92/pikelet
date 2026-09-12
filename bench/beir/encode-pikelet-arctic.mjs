#!/usr/bin/env node
// Encode a BEIR dataset's corpus + test queries with Snowflake arctic-embed-xs
// swapped in for Pikelet's bundled MiniLM, through the exact same production
// code paths encode-pikelet.mjs uses for MiniLM: embedChunksWithInlineTransformer
// (worker-pool corpus embedding) and createInlineTransformerEmbedder (query
// embedding), driven by inlineEncoderDeclaration — the same declaration
// builder `pikelet compile --encoder-model ...` uses — so this measures the
// real --encoder-model swap path, not a hand-rolled approximation of it.
//
// Arctic-XS is CLS-pooled (its native pooling; mean-pooling a BERT model
// trained for CLS pooling silently returns nonsense) and requires an
// asymmetric query prefix ("Represent this sentence for searching relevant
// passages: ") that the CLI's --encoder-query-prefix help text documents but
// this benchmark encodes explicitly so it can be tested with and without —
// see query-prefix-ablation.mjs, which the README's "query prefix verified"
// claim is measured from.
//
// Usage: node bench/beir/encode-pikelet-arctic.mjs <dataset> [--no-prefix]
// Reads:  work/<dataset>/records.jsonl, cache/<dataset>/queries.jsonl,
//         cache/<dataset>/qrels/test.tsv, arctic-xs/{vocab.txt,encoder-weights.bin}
// Writes: work/<dataset>/vectors-corpus-arctic.f32
//         work/<dataset>/vectors-queries-arctic[-noprefix].f32 (+ .ids.json)
//         work/<dataset>/encode-arctic.json

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { embedChunksWithInlineTransformer } from '../../pikelet/src/embed.mjs';
import { inlineEncoderDeclaration } from '../../pikelet/src/complete-build.mjs';
import { createInlineTransformerEmbedder } from '../../complete/inline-transformer.mjs';
import createEncoder from '../../complete/encoder-kernels/encoder.node.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
const noPrefix = process.argv.includes('--no-prefix');
if (!dataset) {
  console.error('usage: node bench/beir/encode-pikelet-arctic.mjs <dataset> [--no-prefix]');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
const arcticDir = path.join(__dirname, 'arctic-xs');
const QUERY_PREFIX = 'Represent this sentence for searching relevant passages: ';

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

// The exact declaration inlineEncoderDeclaration builds for
// `pikelet compile --encoder-model Snowflake/snowflake-arctic-embed-xs
// --encoder-pooling cls --encoder-query-prefix "..."` — driving
// embedChunksWithInlineTransformer through the same function production
// uses, not a hand-rolled config.
const config = {
  embedding: {
    dims: 384,
    mode: 'inline-transformer',
    pooling: 'cls',
    normalize: true,
    prefixPolicy: { query: noPrefix ? '' : QUERY_PREFIX, passage: '' },
  },
  runtime: {
    inlineEncoder: {
      vocabPath: path.join(arcticDir, 'vocab.txt'),
      weightsPath: path.join(arcticDir, 'encoder-weights.bin'),
      model: 'Snowflake/snowflake-arctic-embed-xs',
      pooling: 'cls',
    },
  },
};
const declaration = inlineEncoderDeclaration(config, config.runtime.inlineEncoder);

const cores = typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;
const expectedWorkers = records.length < 32 ? 1 : Math.max(1, Math.min(cores - 1, 8, records.length));
console.error(`encoding ${records.length} corpus records via production embedChunksWithInlineTransformer `
  + `(arctic-xs, cls pooling, expect ${expectedWorkers} workers)...`);

const t0 = performance.now();
const corpusChunks = records.map((r) => ({ text: r.text }));
const corpusVectors = await embedChunksWithInlineTransformer(corpusChunks, config, (msg) => console.error(msg), __dirname);
const corpusMs = performance.now() - t0;

const dim = 384;
const corpusFlat = new Float32Array(records.length * dim);
for (let i = 0; i < records.length; i++) corpusFlat.set(corpusVectors[i], i * dim);

console.error(`encoding ${testQueries.length} test queries (single-embed, matches runtime query path, `
  + `prefix ${noPrefix ? 'DISABLED (ablation)' : 'ENABLED'})...`);
const vocabText = readFileSync(path.join(arcticDir, 'vocab.txt'), 'utf8');
const blob = readFileSync(path.join(arcticDir, 'encoder-weights.bin'));
const embedder = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder, verify: false });

const queryPrefix = noPrefix ? '' : QUERY_PREFIX;
const t1 = performance.now();
const queryFlat = new Float32Array(testQueries.length * dim);
for (let i = 0; i < testQueries.length; i++) {
  const { vector } = await embedder.embed(`${queryPrefix}${testQueries[i].text}`);
  queryFlat.set(vector, i * dim);
  if (i % 100 === 0) console.error(`queries ${i}/${testQueries.length}`);
}
const queryMs = performance.now() - t1;
embedder.dispose();

const suffix = noPrefix ? '-noprefix' : '';
writeFileSync(path.join(workDir, 'vectors-corpus-arctic.f32'), Buffer.from(corpusFlat.buffer, corpusFlat.byteOffset, corpusFlat.byteLength));
writeFileSync(path.join(workDir, `vectors-queries-arctic${suffix}.f32`), Buffer.from(queryFlat.buffer, queryFlat.byteOffset, queryFlat.byteLength));
writeFileSync(path.join(workDir, `vectors-queries-arctic${suffix}.ids.json`), JSON.stringify({ ids: testQueries.map((q) => q._id), dim: 384, count: testQueries.length }));
writeFileSync(path.join(workDir, `encode-arctic${suffix}.json`), JSON.stringify({
  dataset,
  encoder: 'Snowflake/snowflake-arctic-embed-xs',
  pooling: 'cls',
  query_prefix: queryPrefix,
  corpus_count: records.length,
  corpus_encode_ms: corpusMs,
  corpus_workers_expected: expectedWorkers,
  query_count: testQueries.length,
  query_encode_ms: queryMs,
  cores,
  node: process.version,
  platform: process.platform,
}, null, 2));

console.log(`${dataset} arctic${suffix}: corpus ${records.length} in ${(corpusMs / 1000).toFixed(1)}s `
  + `(~${expectedWorkers} workers), queries ${testQueries.length} in ${(queryMs / 1000).toFixed(1)}s (sequential)`);
