#!/usr/bin/env node
// Query-prefix ablation for Arctic-XS: re-embeds ONLY the test queries
// without "Represent this sentence for searching relevant passages: ",
// reusing the corpus vectors encode-pikelet-arctic.mjs already built (corpus
// text is prefix-free in both arms — Arctic-XS's asymmetric prefix only
// ever applies to queries — so re-encoding it here would be pure waste).
//
// Usage: node bench/beir/ablate-query-prefix.mjs <dataset>
// Requires: work/<dataset>/vectors-corpus-arctic.f32 already built
//           (encode-pikelet-arctic.mjs <dataset>, run once, WITH prefix)
// Writes: work/<dataset>/vectors-queries-arctic-noprefix.f32 (+ .ids.json)

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { createInlineTransformerEmbedder } from '../../complete/inline-transformer.mjs';
import createEncoder from '../../complete/encoder-kernels/encoder.node.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataset = process.argv[2];
if (!dataset) {
  console.error('usage: node bench/beir/ablate-query-prefix.mjs <dataset>');
  process.exit(1);
}

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
const arcticDir = path.join(__dirname, 'arctic-xs');

if (!existsSync(path.join(workDir, 'vectors-corpus-arctic.f32'))) {
  console.error(`missing corpus vectors — run: node bench/beir/encode-pikelet-arctic.mjs ${dataset} first`);
  process.exit(1);
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

const declaration = {
  kind: 'inline-transformer-v1',
  model: 'Snowflake/snowflake-arctic-embed-xs',
  dim: 384,
  pooling: 'cls',
  normalized: true,
  maxTokens: 512,
  longInputs: 'first-window-cls',
  layout: { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 6, B: 64, H: 12 },
};
const vocabText = readFileSync(path.join(arcticDir, 'vocab.txt'), 'utf8');
const blob = readFileSync(path.join(arcticDir, 'encoder-weights.bin'));
const embedder = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder, verify: false });

console.error(`encoding ${testQueries.length} test queries WITHOUT the query prefix (ablation)...`);
const dim = 384;
const t0 = performance.now();
const queryFlat = new Float32Array(testQueries.length * dim);
for (let i = 0; i < testQueries.length; i++) {
  const { vector } = await embedder.embed(testQueries[i].text);
  queryFlat.set(vector, i * dim);
  if (i % 100 === 0) console.error(`queries ${i}/${testQueries.length}`);
}
const queryMs = performance.now() - t0;
embedder.dispose();

writeFileSync(path.join(workDir, 'vectors-queries-arctic-noprefix.f32'), Buffer.from(queryFlat.buffer, queryFlat.byteOffset, queryFlat.byteLength));
writeFileSync(path.join(workDir, 'vectors-queries-arctic-noprefix.ids.json'), JSON.stringify({ ids: testQueries.map((q) => q._id), dim, count: testQueries.length }));

console.log(`${dataset} arctic-noprefix: queries ${testQueries.length} in ${(queryMs / 1000).toFixed(1)}s (sequential)`);
