#!/usr/bin/env node
// Encode a BEIR dataset's corpus + test queries the way `embedding.mode:
// "workers-ai"` does — BGE-small-en-v1.5 through the CLI's own build-time
// encoder — so the CLI's DEFAULT mode gets a relevance number on the same
// ladder as the inline MiniLM (B-F) and Arctic-XS (G-H) rows.
//
// Why this is a separate script rather than a flag on encode-pikelet.mjs:
// workers-ai is a SPLIT-ENCODER mode. The corpus is embedded locally at
// build time with the inline-transformer kernel running the pinned u8 export
// of BAAI/bge-small-en-v1.5 (MODEL_MAP's weights), while queries are
// embedded at request time by Cloudflare's AI binding
// (@cf/baai/bge-small-en-v1.5, the workersAiModel). Both halves are the same
// upstream BGE-small-en-v1.5, which is the premise the whole mode rests on
// and what `assertEncoderManifest` pins by name.
//
// WHAT THIS MEASURES, AND WHAT IT DOES NOT. This encodes BOTH halves
// locally with the kernel, through embed.mjs's own embedChunks, so the
// corpus vectors are the ones a real build produces. It does NOT exercise a
// real `env.AI.run` call, so it cannot detect drift between Cloudflare's
// hosted copy and the local u8 export — spec risk R1, which
// `assertEncoderManifest` guards by NAME, not by value. Do not read these
// rows as validating the Cloudflare call path. (Before 0.12.0 this script
// encoded with transformers.js's ONNX export. The 0.12.0 CHANGELOG entry
// records the switch: kernel passage vectors scored against live Workers AI
// query vectors give SciFact nDCG@10 0.6991, against 0.6992 before.)
//
// The prefix policy is the contract: BGE v1.5 retrieval depends on an
// asymmetric query-side instruction prefix and raw passages.
// DEFAULT_CONFIG.embedding.prefixPolicy in the CLI is
// { passage: '', query: DEFAULT_PREFIX } and encoder.workers-ai.js applies
// manifest.prefixPolicy.query before calling the AI binding, so this script
// applies exactly the same split. --no-prefix ablates it, the same way
// ablate-query-prefix.mjs does for Arctic-XS.
//
// Usage: node benchmarks/beir/encode-workers-ai.mjs <dataset> [--no-prefix]
// Reads:  work/<dataset>/records.jsonl, cache/<dataset>/queries.jsonl,
//         cache/<dataset>/qrels/test.tsv
// Writes: work/<dataset>/vectors-corpus-workersai.f32
//         work/<dataset>/vectors-queries-workersai[-noprefix].f32 (+ .ids.json)
//         work/<dataset>/encode-workers-ai[-noprefix].json

import { readFileSync, writeFileSync, existsSync, createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { DEFAULT_PREFIX, MODEL_MAP } from '../../packages/pikelet/src/common.mjs';
import { embedChunks } from '../../packages/pikelet/src/embed.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const dataset = args[0];
const noPrefix = args.includes('--no-prefix');
if (!dataset || dataset.startsWith('--')) {
  console.error('usage: node benchmarks/beir/encode-workers-ai.mjs <dataset> [--no-prefix]');
  process.exit(1);
}

// The build model workers-ai mode actually uses, read from the CLI's own
// MODEL_MAP rather than hardcoded, so this tracks the allowlist.
const BUILD_MODEL = 'bge-small-en-v1.5';
const model = MODEL_MAP[BUILD_MODEL];
if (!model) {
  console.error(`MODEL_MAP has no ${BUILD_MODEL}; the CLI's allowlist changed — update this script.`);
  process.exit(1);
}
const DIM = model.dims;

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
for (const p of [path.join(workDir, 'records.jsonl'), path.join(cacheDir, 'queries.jsonl'), path.join(cacheDir, 'qrels', 'test.tsv')]) {
  if (!existsSync(p)) {
    console.error(`missing ${p} — run download.py and convert.mjs for ${dataset} first`);
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

function testQueryIds() {
  const text = readFileSync(path.join(cacheDir, 'qrels', 'test.tsv'), 'utf8');
  const ids = new Set();
  for (const line of text.split('\n').slice(1)) {
    const [qid] = line.split('\t');
    if (qid) ids.add(qid);
  }
  return ids;
}

const records = await readJsonl(path.join(workDir, 'records.jsonl')); // pikeletRow order
const queries = await readJsonl(path.join(cacheDir, 'queries.jsonl'));
const testIds = testQueryIds();
const testQueries = queries.filter((q) => testIds.has(q._id));

// embed.mjs pools/normalizes from embeddingConfig; DEFAULT_CONFIG.embedding
// sets mean + normalize true for this mode. The prefixes are applied below,
// so the declaration's own prefixPolicy stays empty.
const POOLING = 'mean';
const NORMALIZE = true;
const embedConfig = {
  embedding: {
    mode: 'workers-ai',
    buildModel: BUILD_MODEL,
    dims: DIM,
    pooling: POOLING,
    normalize: NORMALIZE,
    prefixPolicy: { passage: '', query: '' },
  },
};

async function embedTexts(texts, label) {
  const t0 = performance.now();
  const vectors = await embedChunks(texts.map((text) => ({ text })), embedConfig, (line) => console.error(`${label}: ${line}`), workDir);
  if (vectors.length !== texts.length) {
    throw new Error(`${label}: expected ${texts.length} vectors, got ${vectors.length}`);
  }
  const flat = new Float32Array(texts.length * DIM);
  for (let i = 0; i < vectors.length; i++) {
    if (vectors[i].length !== DIM) throw new Error(`${label}: vector ${i} has ${vectors[i].length} dims, expected ${DIM}`);
    flat.set(vectors[i], i * DIM);
  }
  return { flat, ms: performance.now() - t0 };
}

// Passages raw, queries prefixed — the asymmetry is the contract.
const passagePrefix = '';
const queryPrefix = noPrefix ? '' : DEFAULT_PREFIX;

console.error(`encoding ${records.length} corpus records (passage prefix ${passagePrefix ? 'ENABLED' : 'none'})...`);
const corpus = await embedTexts(records.map((r) => `${passagePrefix}${r.text}`), 'corpus');

console.error(`encoding ${testQueries.length} test queries (query prefix ${noPrefix ? 'DISABLED (ablation)' : 'ENABLED'})...`);
const q = await embedTexts(testQueries.map((t) => `${queryPrefix}${t.text}`), 'queries');

const suffix = noPrefix ? '-noprefix' : '';
writeFileSync(path.join(workDir, 'vectors-corpus-workersai.f32'), Buffer.from(corpus.flat.buffer, corpus.flat.byteOffset, corpus.flat.byteLength));
writeFileSync(path.join(workDir, `vectors-queries-workersai${suffix}.f32`), Buffer.from(q.flat.buffer, q.flat.byteOffset, q.flat.byteLength));
writeFileSync(path.join(workDir, `vectors-queries-workersai${suffix}.ids.json`), JSON.stringify({ ids: testQueries.map((t) => t._id), dim: DIM, count: testQueries.length }));
writeFileSync(path.join(workDir, `encode-workers-ai${suffix}.json`), JSON.stringify({
  dataset,
  mode: 'workers-ai',
  build_model: BUILD_MODEL,
  hf_model: model.hfModel,          // the upstream model behind the local weights
  build_weights: model.weights.file, // what encoded these vectors, locally (u8 kernel)
  build_weights_sha256: model.weights.sha256,
  workers_ai_model: model.workersAiModel, // what would encode queries in production
  measures_cloudflare_call: false,  // see the header note: model only, not the AI binding
  pooling: POOLING,
  normalize: NORMALIZE,
  passage_prefix: passagePrefix,
  query_prefix: queryPrefix,
  corpus_count: records.length,
  corpus_encode_ms: corpus.ms,
  query_count: testQueries.length,
  query_encode_ms: q.ms,
  cores: typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length,
  node: process.version,
  platform: process.platform,
}, null, 2));

console.log(`${dataset} workers-ai${suffix}: corpus ${records.length} in ${(corpus.ms / 1000).toFixed(1)}s `
  + `(${(records.length / (corpus.ms / 1000)).toFixed(1)}/s), queries ${testQueries.length} in ${(q.ms / 1000).toFixed(1)}s`);
