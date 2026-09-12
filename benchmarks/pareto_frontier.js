#!/usr/bin/env node
'use strict';

/**
 * Pareto-Frontier Benchmark Harness (QPS-Recall)
 *
 * Produces, for one configured graph (M and ef_construction are CLI-tunable):
 *   1. A QPS-recall sweep for every library that exposes a query-time knob
 *      (ef_search swept across a standard range), and
 *   2. The Pareto frontier over each library's sweep (non-dominated points), and
 *   3. Interpolated equal-recall curves: QPS for every library at a common grid
 *      of recall targets, log-linearly interpolated along each frontier.
 *
 * Configs:
 *   pikelet-wasm   u8 / fp32   (Pikelet.create, setEfSearch per ef)
 *   pikelet-native u8 / fp32   (native.pancake_*, native.pancake_set_ef per ef)
 *   usearch-native i8 / f16 / f32 (build-once-save-view per ef —
 *                                  expansion_search is fixed at construction)
 *   usearch-wasm   i8 / f32      (C ABI directly against the WASM artifact;
 *                                  f16 is opt-in via --configs — software
 *                                  fp16 makes it ~10x slower, see below)
 *   hnswlib        f32           (HierarchicalNSW.setEf per ef)
 *
 * All libraries use the same M and ef_construction, and every config is swept
 * across the configured ef_search range.
 *
 * Usage:
 *   node benchmarks/pareto_frontier.js --dataset dbpedia
 *   node benchmarks/pareto_frontier.js --dataset sift
 *   node benchmarks/pareto_frontier.js --dataset nytimes
 *   node benchmarks/pareto_frontier.js --dataset glove
 *   node benchmarks/pareto_frontier.js --dataset mnist
 *   node benchmarks/pareto_frontier.js --dataset custom --base-file base.fvecs --query-file query.fvecs --metric l2
 *   node benchmarks/pareto_frontier.js --dataset custom --hdf5-file glove-100-angular.hdf5 --metric cosine
 *   node benchmarks/pareto_frontier.js --dataset dbpedia --data-dir ./dbpedia
 *   node benchmarks/pareto_frontier.js --count 50000
 *   node benchmarks/pareto_frontier.js --ef-search-values 10,50,100,200
 *   node benchmarks/pareto_frontier.js --regenerate-gt
 *   node benchmarks/pareto_frontier.js --dataset nytimes --zero-vector-policy fail
 *   node benchmarks/pareto_frontier.js --dataset nytimes --library pikelet
 *   node benchmarks/pareto_frontier.js --dataset nytimes --configs pikelet-wasm-u8,pikelet-wasm-fp32
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Pikelet = require('../pikelet.js');
const { parseBenchmarkArgs, resolveSingleValue, resolveSweepValues } = require('./bench_args');

// --- Optional libraries (each is independently optional; missing => skipped) ---
let native;
try { native = require('../native'); }
catch (e) { console.warn('WARN: pikelet native binding not built (cd native && npm install) — skipping pikelet-native configs.'); }

let usearch;
try { usearch = require('usearch'); }
catch (e) { console.warn('WARN: usearch not installed — skipping usearch configs.'); }

const DEFAULT_USEARCH_WASM_PATHS = {
  i8: path.join(__dirname, '..', 'external', 'usearch-wasm', 'USearch', 'build_wasm_pikelet_1gb_growth', 'usearch_wasm.wasm'),
  f16: path.join(__dirname, '..', 'external', 'usearch-wasm', 'USearch', 'build_wasm_pikelet_1gb_growth', 'usearch_wasm.wasm'),
  f32: path.join(__dirname, '..', 'external', 'usearch-wasm', 'USearch', 'build_wasm_pikelet_2gb_growth', 'usearch_wasm.wasm'),
};

let HierarchicalNSW;
try { HierarchicalNSW = require('hnswlib-node').HierarchicalNSW; }
catch (e) { console.warn('WARN: hnswlib-node not installed — skipping hnswlib config.'); }

// --- Args / config ---
const parsedArgs = parseBenchmarkArgs();
const rawArgs = process.argv.slice(2);
function getIntArg(name, defaultVal) {
  const idx = rawArgs.indexOf('--' + name);
  return idx >= 0 && idx + 1 < rawArgs.length ? parseInt(rawArgs[idx + 1], 10) : defaultVal;
}
function getStrArg(name, defaultVal) {
  const idx = rawArgs.indexOf('--' + name);
  return idx >= 0 && idx + 1 < rawArgs.length ? rawArgs[idx + 1] : defaultVal;
}
function getBoolArg(name) {
  return rawArgs.includes('--' + name);
}
function hasArg(name) {
  return rawArgs.includes('--' + name);
}
function getCsvArg(name) {
  const raw = getStrArg(name, '');
  return raw.split(',').map(v => v.trim()).filter(Boolean);
}
const REGENERATE_GT = rawArgs.includes('--regenerate-gt');
const USEARCH_WASM_OVERRIDE_PATH = hasArg('usearch-wasm') ? path.resolve(getStrArg('usearch-wasm', '')) : null;
function getUsearchWasmPath(dtype) {
  return USEARCH_WASM_OVERRIDE_PATH || path.resolve(DEFAULT_USEARCH_WASM_PATHS[dtype]);
}

// --- Dataset selection ---------------------------------------------------
// Built-in datasets use fvecs/ivecs files checked into the expected local
// folders. Custom mode accepts explicit fvecs paths and optionally an ivecs
// ground-truth file; without a GT file the harness computes and caches brute
// force ground truth for the selected metric.
const DATASET = getStrArg('dataset', 'dbpedia').toLowerCase();
const DATASETS = {
  dbpedia: {
    dir: path.join(__dirname, '..', 'dbpedia'),
    baseFile: (n) => (n <= 5000 ? 'dbpedia_base_5k.fvecs' : 'dbpedia_base_100k.fvecs'),
    queryFile: 'dbpedia_query.fvecs',
    gtFile: null,
    defaultCount: 50_000,
    metric: 'l2',
  },
  // Same DBpedia OpenAI embedding files as `dbpedia`, evaluated under cosine.
  // The upstream dataset is angular (dbpedia-openai-1000k-angular); the
  // `dbpedia` entry keeps the historical L2 evaluation, this entry scores and
  // builds ground truth with cosine. Caches are keyed per dataset name and
  // metric, so the two never share ground-truth entries.
  'dbpedia-cosine': {
    dir: path.join(__dirname, '..', 'dbpedia'),
    baseFile: (n) => (n <= 5000 ? 'dbpedia_base_5k.fvecs' : 'dbpedia_base_100k.fvecs'),
    queryFile: 'dbpedia_query.fvecs',
    gtFile: null,
    defaultCount: 50_000,
    metric: 'cosine',
  },
  sift: {
    dir: path.join(__dirname, '..', 'sift'),
    baseFile: () => 'sift_base.fvecs',
    queryFile: 'sift_query.fvecs',
    gtFile: 'sift_groundtruth.ivecs',  // precomputed, read from disk
    defaultCount: 1_000_000,
    metric: 'l2',
  },
  nytimes: {
    dir: path.join(__dirname, '..', 'nytimes'),
    baseFile: () => 'nytimes_base.fvecs',
    queryFile: 'nytimes_query.fvecs',
    gtFile: 'nytimes_groundtruth.ivecs',
    defaultCount: 290_000,
    metric: 'cosine',
  },
  glove: {
    dir: path.join(__dirname, '..', 'glove'),
    hdf5File: 'glove-100-angular.hdf5',
    defaultCount: null,
    metric: 'cosine',
  },
  mnist: {
    dir: path.join(__dirname, '..'),
    imageFile: 'train-images-idx3-ubyte',
    defaultCount: 50_000,
    metric: 'l2',
  },
};
if (!DATASETS[DATASET] && DATASET !== 'custom') {
  console.error(`Unknown --dataset ${DATASET}. Use ${Object.keys(DATASETS).join(', ')} or custom.`);
  process.exit(1);
}
const CUSTOM_BASE_FILE = getStrArg('base-file', null);
const CUSTOM_QUERY_FILE = getStrArg('query-file', null);
const CUSTOM_GT_FILE = getStrArg('gt-file', null);
const CUSTOM_HDF5_FILE = getStrArg('hdf5-file', null);
const CUSTOM_METRIC = getStrArg('metric', null);
const DS = DATASET === 'custom'
  ? {
      dir: process.cwd(),
      baseFile: () => CUSTOM_BASE_FILE,
      queryFile: CUSTOM_QUERY_FILE,
      gtFile: CUSTOM_GT_FILE,
      hdf5File: CUSTOM_HDF5_FILE,
      defaultCount: null,
      metric: CUSTOM_METRIC || 'l2',
    }
  : DATASETS[DATASET];
if (DATASET === 'custom' && !CUSTOM_HDF5_FILE && (!CUSTOM_BASE_FILE || !CUSTOM_QUERY_FILE)) {
  console.error('custom dataset requires either --hdf5-file or both --base-file and --query-file');
  process.exit(1);
}
if (!['l2', 'cosine'].includes(DS.metric)) {
  console.error(`Unsupported metric '${DS.metric}'. Use l2 or cosine.`);
  process.exit(1);
}
const DATA_DIR = path.resolve(getStrArg('data-dir', DS.dir));

const COUNT_LIMIT = hasArg('count') ? getIntArg('count', null) : DS.defaultCount;
const N_BASE = COUNT_LIMIT || Number.MAX_SAFE_INTEGER;
const READ_LIMIT = COUNT_LIMIT || undefined;
const N_QUERIES = getIntArg('queries', 1_000);
const K = 10;
const METRIC = DS.metric;

// Fixed graph parameters, per request.
const M = resolveSingleValue(parsedArgs.m, 12);
const EF_CONSTRUCTION = resolveSingleValue(parsedArgs.efConstruction, 75);
// ef_search sweep — includes 100 (the requested operating point) plus the
// surrounding range needed to trace a QPS-recall frontier.
const EF_SEARCH_VALUES = resolveSweepValues(parsedArgs, [10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 800]);

const REPETITIONS = 3;
const WARMUP_QUERIES = 200;
const WRITE_PLOT = !getBoolArg('no-plot');
const DEFAULT_ZERO_VECTOR_POLICY = METRIC === 'cosine' ? 'sentinel' : 'drop';
const ZERO_VECTOR_POLICY = getStrArg('zero-vector-policy', DEFAULT_ZERO_VECTOR_POLICY).toLowerCase();
if (!['drop', 'fail', 'sentinel'].includes(ZERO_VECTOR_POLICY)) {
  console.error(`Unsupported --zero-vector-policy ${ZERO_VECTOR_POLICY}. Use drop, fail, or sentinel.`);
  process.exit(1);
}

const CONFIG_FILTER_LABELS = new Set(getCsvArg('configs'));
const CONFIG_FILTER_LIBRARIES = new Set(getCsvArg('library'));
if (CONFIG_FILTER_LABELS.size > 0 && CONFIG_FILTER_LIBRARIES.size > 0) {
  console.error('Use either --configs or --library, not both.');
  process.exit(1);
}

// --- Config table. Every config is ef-swept (no single-point libraries). ---
let CONFIGS = [];
CONFIGS.push({ label: 'pikelet-wasm-u8',   library: 'pikelet', runtime: 'wasm',   dtype: 'u8',  sweep: true });
CONFIGS.push({ label: 'pikelet-wasm-fp32',   library: 'pikelet', runtime: 'wasm',   dtype: 'f32', sweep: true });
if (native) {
  CONFIGS.push({ label: 'pikelet-native-u8', library: 'pikelet', runtime: 'native', dtype: 'u8',  sweep: true });
  CONFIGS.push({ label: 'pikelet-native-fp32', library: 'pikelet', runtime: 'native', dtype: 'f32', sweep: true });
}
if (usearch) {
  CONFIGS.push({ label: 'usearch-native-int8', library: 'usearch', runtime: 'native', dtype: 'i8',  sweep: true, aliases: ['usearch-int8'] });
  CONFIGS.push({ label: 'usearch-native-f16',  library: 'usearch', runtime: 'native', dtype: 'f16', sweep: true, aliases: ['usearch-f16'] });
  CONFIGS.push({ label: 'usearch-native-fp32', library: 'usearch', runtime: 'native', dtype: 'f32', sweep: true, aliases: ['usearch-fp32'] });
}
for (const config of [
  { label: 'usearch-wasm-int8', dtype: 'i8' },
  // f16 is opt-in, not part of the default sweep: WASM has no hardware fp16
  // path, so every distance pays a per-element software conversion. Measured
  // 2026-08-12 (dbpedia 1536D, count=2000): build 16.1s vs i8's 1.7s, and
  // 465 vs 4455 qps at ef=10 — ~10x slower across the board and dominated by
  // both i8 and fp32 on the frontier. At default counts the silent build
  // phase looks like a hang. Request it explicitly with
  // --configs usearch-wasm-f16 when the software-fp16 datapoint is wanted.
  { label: 'usearch-wasm-f16', dtype: 'f16', optIn: true },
  { label: 'usearch-wasm-fp32', dtype: 'f32' },
]) {
  if (config.optIn && !CONFIG_FILTER_LABELS.has(config.label)) continue;
  const wasmPath = getUsearchWasmPath(config.dtype);
  if (fs.existsSync(wasmPath)) {
    CONFIGS.push({ label: config.label, library: 'usearch-wasm', runtime: 'wasm', dtype: config.dtype, wasmPath, sweep: true });
  } else {
    console.warn(`WARN: usearch WASM artifact not found at ${wasmPath} — skipping ${config.label}.`);
  }
}
if (HierarchicalNSW) {
  CONFIGS.push({ label: 'hnswlib-fp32', library: 'hnswlib', dtype: 'f32', sweep: true });
}
const CONFIG_ALIAS_TO_LABEL = new Map();
for (const c of CONFIGS) {
  CONFIG_ALIAS_TO_LABEL.set(c.label, c.label);
  for (const alias of c.aliases || []) CONFIG_ALIAS_TO_LABEL.set(alias, c.label);
}
const AVAILABLE_CONFIG_LABELS = new Set(CONFIG_ALIAS_TO_LABEL.keys());
const AVAILABLE_CONFIG_LIBRARIES = new Set(CONFIGS.map(c => c.library));
if (CONFIG_FILTER_LABELS.size > 0) {
  const unknown = [...CONFIG_FILTER_LABELS].filter(label => !AVAILABLE_CONFIG_LABELS.has(label));
  if (unknown.length > 0) {
    console.error(`Unknown --configs value(s): ${unknown.join(', ')}`);
    console.error(`Available configs: ${[...AVAILABLE_CONFIG_LABELS].join(', ')}`);
    process.exit(1);
  }
  const selectedLabels = new Set([...CONFIG_FILTER_LABELS].map(label => CONFIG_ALIAS_TO_LABEL.get(label)));
  CONFIGS = CONFIGS.filter(c => selectedLabels.has(c.label));
}
if (CONFIG_FILTER_LIBRARIES.size > 0) {
  const unknown = [...CONFIG_FILTER_LIBRARIES].filter(library => !AVAILABLE_CONFIG_LIBRARIES.has(library));
  if (unknown.length > 0) {
    console.error(`Unknown --library value(s): ${unknown.join(', ')}`);
    console.error(`Available libraries: ${[...AVAILABLE_CONFIG_LIBRARIES].join(', ')}`);
    process.exit(1);
  }
  CONFIGS = CONFIGS.filter(c => CONFIG_FILTER_LIBRARIES.has(c.library));
}
if (CONFIGS.length === 0) {
  console.error('No benchmark configs selected.');
  process.exit(1);
}

// --- Output paths ---
const RESULTS_DIR = path.resolve(getStrArg('output-dir', path.join(__dirname, '..', 'benchmark_results')));
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR, { recursive: true });
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const outputStem = `pareto_${DATASET}_${timestamp}`;
const LOG_PATH      = path.join(RESULTS_DIR, `${outputStem}.log`);
const JSON_PATH     = path.join(RESULTS_DIR, `${outputStem}.json`);
const CSV_PATH      = path.join(RESULTS_DIR, `${outputStem}.csv`);
const FRONTIER_CSV  = path.join(RESULTS_DIR, `${outputStem}_frontier.csv`);
const EQRECALL_CSV  = path.join(RESULTS_DIR, `${outputStem}_equalrecall.csv`);
const PNG_PATH      = path.join(RESULTS_DIR, `${outputStem}.png`);
const logStream = fs.createWriteStream(LOG_PATH);
function log(msg = '') { console.log(msg); logStream.write(msg + '\n'); }
function resolveDatasetPath(filePath) {
  if (!filePath) return null;
  return path.isAbsolute(filePath) ? filePath : path.join(DATA_DIR, filePath);
}

// --- .fvecs reader ---
function readFvecs(filePath, maxVectors) {
  log(`  Loading ${filePath}${maxVectors ? ` (first ${maxVectors.toLocaleString()})` : ''}...`);
  const buf = fs.readFileSync(filePath);
  const vectors = [];
  let offset = 0;
  while (offset < buf.length) {
    const dim = buf.readInt32LE(offset); offset += 4;
    const vec = new Float32Array(dim);
    for (let d = 0; d < dim; d++) { vec[d] = buf.readFloatLE(offset); offset += 4; }
    vectors.push(vec);
    if (maxVectors && vectors.length >= maxVectors) break;
  }
  return { vectors, dim: vectors[0].length };
}

// --- .ivecs reader (int32 ground-truth neighbor IDs) ---
// Reads the first `maxRows` rows, keeping the first K neighbor IDs per row.
function readIvecs(filePath, maxRows) {
  log(`  Loading ${filePath} (ground truth)...`);
  const buf = fs.readFileSync(filePath);
  const rows = [];
  let offset = 0;
  while (offset < buf.length) {
    const dim = buf.readInt32LE(offset); offset += 4;
    const row = new Array(Math.min(dim, K));
    for (let d = 0; d < dim; d++) {
      const v = buf.readInt32LE(offset); offset += 4;
      if (d < K) row[d] = v;
    }
    rows.push(row);
    if (maxRows && rows.length >= maxRows) break;
  }
  return rows;
}

function readIdxImages(filePath, maxTrain, maxQueries) {
  log(`  Loading ${filePath} (IDX images)...`);
  const buf = fs.readFileSync(filePath);
  if (buf.length < 16) throw new Error(`IDX image file is truncated: ${filePath}`);
  const magic = buf.readUInt32BE(0);
  const count = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  if (magic !== 2051) throw new Error(`Unsupported IDX image magic ${magic}; expected 2051`);
  const dim = rows * cols;
  const expectedBytes = 16 + count * dim;
  if (buf.length < expectedBytes) {
    throw new Error(`IDX image file is truncated: expected ${expectedBytes} bytes, got ${buf.length}`);
  }

  const trainCount = Math.min(maxTrain || count, count);
  const queryStart = trainCount;
  const queryCount = Math.min(maxQueries || 0, Math.max(0, count - queryStart));
  if (trainCount === 0) throw new Error('IDX image dataset has no base vectors');
  if (queryCount === 0) {
    throw new Error(`IDX image dataset has no query vectors after selecting ${trainCount.toLocaleString()} base vectors`);
  }

  const readRows = (start, n) => {
    const vectors = new Array(n);
    for (let i = 0; i < n; i++) {
      const vec = new Float32Array(dim);
      const rowOffset = 16 + (start + i) * dim;
      for (let d = 0; d < dim; d++) vec[d] = buf[rowOffset + d] / 255;
      vectors[i] = vec;
    }
    return vectors;
  };

  const train = readRows(0, trainCount);
  const test = readRows(queryStart, queryCount);
  log(`  IDX images: ${count.toLocaleString()} total, ${rows}x${cols} -> ${dim}D`);
  log(`  IDX split:  ${train.length.toLocaleString()} base, ${test.length.toLocaleString()} queries`);
  return { train, test, groundTruth: null, dim };
}

function readFloatMatrix(buf, offset, rows, dim) {
  const vectors = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const vec = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      vec[d] = buf.readFloatLE(offset);
      offset += 4;
    }
    vectors[i] = vec;
  }
  return { vectors, offset };
}

function readNeighborMatrix(buf, offset, rows) {
  const neighbors = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array(K);
    for (let j = 0; j < K; j++) {
      row[j] = buf.readInt32LE(offset);
      offset += 4;
    }
    neighbors[i] = row;
  }
  return neighbors;
}

function loadHdf5Dataset(filePath, maxTrain, maxQueries) {
  log(`  Loading ${filePath} (HDF5)...`);
  const tmpPy = path.join(os.tmpdir(), `pareto_hdf5_${process.pid}_${Date.now()}.py`);
  const tmpBin = path.join(os.tmpdir(), `pareto_hdf5_${process.pid}_${Date.now()}.bin`);
  const script = `
import h5py, json, struct, sys
src, dst = sys.argv[1], sys.argv[2]
max_train, max_queries, k = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
with h5py.File(src, "r") as f:
    train_total = int(f["train"].shape[0])
    query_total = int(f["test"].shape[0])
    train_count = train_total if max_train <= 0 else min(max_train, train_total)
    query_count = query_total if max_queries <= 0 else min(max_queries, query_total)
    train = f["train"][:train_count].astype("float32")
    test = f["test"][:query_count].astype("float32")
    has_neighbors = "neighbors" in f and train_count == train_total
    neighbors = f["neighbors"][:query_count, :k].astype("int32") if has_neighbors else None
    info = {
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "dim": int(train.shape[1]),
        "train_total": train_total,
        "has_neighbors": bool(has_neighbors),
    }
with open(dst, "wb") as out:
    info_bytes = json.dumps(info).encode()
    out.write(struct.pack("<I", len(info_bytes)))
    out.write(info_bytes)
    out.write(train.tobytes())
    out.write(test.tobytes())
    if neighbors is not None:
        out.write(neighbors.tobytes())
print(json.dumps(info))
`;

  try {
    fs.writeFileSync(tmpPy, script);
    const result = spawnSync('python3', [tmpPy, filePath, tmpBin, String(maxTrain || 0), String(maxQueries || 0), String(K)], {
      encoding: 'utf8',
      maxBuffer: 1024 * 1024,
    });
    if (result.status !== 0) {
      const detail = (result.stderr || result.error || '').toString().trim();
      throw new Error(`Failed to load HDF5 dataset${detail ? `: ${detail}` : ''}`);
    }

    const info = JSON.parse(result.stdout.trim().split(/\r?\n/).filter(Boolean).pop());
    const buf = fs.readFileSync(tmpBin);
    let offset = 0;
    const infoLen = buf.readUInt32LE(offset); offset += 4;
    offset += infoLen;

    const trainData = readFloatMatrix(buf, offset, info.n_train, info.dim);
    offset = trainData.offset;
    const testData = readFloatMatrix(buf, offset, info.n_test, info.dim);
    offset = testData.offset;
    const groundTruth = info.has_neighbors ? readNeighborMatrix(buf, offset, info.n_test) : null;

    log(`  HDF5 train: ${info.n_train.toLocaleString()} vectors, ${info.dim}D`);
    log(`  HDF5 test:  ${info.n_test.toLocaleString()} queries`);
    if (groundTruth) log(`  HDF5 ground truth: ${groundTruth.length.toLocaleString()} rows, first ${K} neighbors each`);
    else log('  HDF5 ground truth: not used; computing for selected subset');

    return { train: trainData.vectors, test: testData.vectors, groundTruth, dim: info.dim };
  } finally {
    for (const tmp of [tmpPy, tmpBin]) {
      try { if (fs.existsSync(tmp)) fs.unlinkSync(tmp); } catch (_) {}
    }
  }
}

// --- Ground truth (metric-aware brute force, cached) ---
const CACHE_DIR = path.join(RESULTS_DIR, 'cache');
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
const GT_COUNT_LABEL = COUNT_LIMIT ? String(COUNT_LIMIT) : 'all';
// Built-in datasets have a fixed source, so their name identifies the data.
// A custom dataset does not: two different --base-file/--hdf5-file sources
// with the same shape would share one cache entry and silently corrupt
// recall numbers, so fingerprint the source files into the cache key.
function customGtSourceFingerprint() {
  const crypto = require('crypto');
  const hash = crypto.createHash('sha256');
  for (const file of [CUSTOM_HDF5_FILE, CUSTOM_BASE_FILE, CUSTOM_QUERY_FILE]) {
    if (!file) continue;
    const resolved = path.resolve(file);
    const stat = fs.statSync(resolved);
    hash.update(`${resolved}\0${stat.size}\0${stat.mtimeMs}\0`);
  }
  return hash.digest('hex').slice(0, 12);
}
const GT_DATASET_LABEL = DATASET === 'custom' ? `custom_${customGtSourceFingerprint()}` : DATASET;
const GT_CACHE_PATH = path.join(CACHE_DIR, `gt_${GT_DATASET_LABEL}_${METRIC}_n${GT_COUNT_LABEL}_q${N_QUERIES}_k${K}.bin`);

function saveGroundTruth(gt) {
  const buf = Buffer.alloc(8 + gt.length * K * 4);
  buf.writeUInt32LE(gt.length, 0);
  buf.writeUInt32LE(K, 4);
  let offset = 8;
  for (const row of gt) for (let j = 0; j < K; j++) { buf.writeInt32LE(row[j], offset); offset += 4; }
  fs.writeFileSync(GT_CACHE_PATH, buf);
  log(`  Cached ground truth to ${GT_CACHE_PATH}`);
}
function loadGroundTruth() {
  if (!fs.existsSync(GT_CACHE_PATH)) return null;
  const buf = fs.readFileSync(GT_CACHE_PATH);
  const nq = buf.readUInt32LE(0);
  const k = buf.readUInt32LE(4);
  if (k !== K) return null;
  const gt = new Array(nq);
  let offset = 8;
  for (let i = 0; i < nq; i++) {
    gt[i] = new Array(k);
    for (let j = 0; j < k; j++) { gt[i][j] = buf.readInt32LE(offset); offset += 4; }
  }
  return gt;
}
function distanceForMetric(a, b, dim) {
  if (METRIC === 'l2') {
    let sum = 0;
    for (let d = 0; d < dim; d++) { const diff = a[d] - b[d]; sum += diff * diff; }
    return sum;
  }

  let dot = 0, an = 0, bn = 0;
  for (let d = 0; d < dim; d++) {
    dot += a[d] * b[d];
    an += a[d] * a[d];
    bn += b[d] * b[d];
  }
  return 1 - dot / ((Math.sqrt(an) || 1) * (Math.sqrt(bn) || 1));
}

function vectorStatus(vec, dim) {
  let normSq = 0;
  for (let d = 0; d < dim; d++) {
    const value = vec[d];
    if (!Number.isFinite(value)) return { ok: false, reason: 'non-finite' };
    normSq += value * value;
  }
  if (METRIC === 'cosine' && (!(normSq > 0) || !Number.isFinite(normSq))) {
    return { ok: false, reason: 'zero-norm' };
  }
  return { ok: true, reason: null };
}

function sentinelVector(dim) {
  const v = new Float32Array(dim);
  v[0] = 1;
  return v;
}

function prepareIndexableVectors(vectors, dim, label) {
  const kept = [];
  const dropped = [];
  const replaced = [];
  for (let i = 0; i < vectors.length; i++) {
    const status = vectorStatus(vectors[i], dim);
    if (status.ok) {
      kept.push(vectors[i]);
    } else if (METRIC === 'cosine' && status.reason === 'zero-norm' && ZERO_VECTOR_POLICY === 'sentinel') {
      kept.push(sentinelVector(dim));
      replaced.push(i);
    } else if (ZERO_VECTOR_POLICY === 'fail' || ZERO_VECTOR_POLICY === 'sentinel') {
      log(`ERROR: ${label} vector ${i} is not indexable for ${METRIC} (${status.reason}).`);
      if (ZERO_VECTOR_POLICY === 'sentinel' && status.reason !== 'zero-norm') {
        log('  --zero-vector-policy sentinel only repairs zero-norm cosine vectors; non-finite vectors still fail.');
      }
      process.exit(1);
    } else {
      dropped.push(i);
    }
  }
  if (dropped.length > 0) {
    log('  Dropped ' + dropped.length.toLocaleString() + ' non-indexable ' + label + ' vector(s) for ' + METRIC + '.');
    log('  First dropped ' + label + ' row(s): ' + dropped.slice(0, 8).join(', ') + (dropped.length > 8 ? ', ...' : ''));
  }
  if (replaced.length > 0) {
    log('  Replaced ' + replaced.length.toLocaleString() + ' zero-norm ' + label + ' vector(s) with a deterministic cosine sentinel.');
    log('  First replaced ' + label + ' row(s): ' + replaced.slice(0, 8).join(', ') + (replaced.length > 8 ? ', ...' : ''));
  }
  return { vectors: kept, dropped, replaced };
}

function computeGroundTruth(train, queries, dim) {
  const nq = queries.length;
  log(`Computing brute-force ${METRIC} ground truth (${nq} x ${train.length} x ${dim}D)... this can take minutes.`);
  const t0 = performance.now();
  const gt = new Array(nq);
  for (let q = 0; q < nq; q++) {
    const query = queries[q];
    const dists = new Float32Array(train.length);
    for (let i = 0; i < train.length; i++) dists[i] = distanceForMetric(query, train[i], dim);
    const indices = new Array(train.length);
    for (let i = 0; i < train.length; i++) indices[i] = i;
    indices.sort((a, b) => dists[a] - dists[b]);
    gt[q] = indices.slice(0, K);
    if ((q + 1) % 25 === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      const eta = (elapsed / (q + 1)) * (nq - q - 1);
      process.stdout.write(`  ${q + 1}/${nq} (${elapsed.toFixed(0)}s elapsed, ~${eta.toFixed(0)}s remaining)\r`);
    }
  }
  log(`  Ground truth computed in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return gt;
}

// --- Metric helpers ---
function recall(predicted, truth) {
  const truthSet = new Set(truth);
  let hits = 0;
  for (const id of predicted) if (truthSet.has(id)) hits++;
  return hits / truth.length;
}
function percentile(sorted, p) { return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]; }
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function stddev(arr) { const m = mean(arr); return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length); }
function nativeMetricValue() { return METRIC === 'l2' ? 0 : 1; }
function usearchMetricValue() { return METRIC === 'l2' ? 'l2sq' : 'cos'; }
function hnswlibMetricValue() { return METRIC === 'l2' ? 'l2' : 'cosine'; }
function bytesToMB(bytes) { return bytes == null ? null : bytes / 1024 / 1024; }
function forceGc() { if (global.gc) global.gc(); }
function measureRssBytes() {
  forceGc();
  return process.memoryUsage().rss;
}
function rssDeltaBytes(before, after) {
  if (!Number.isFinite(before) || !Number.isFinite(after)) return null;
  return Math.max(0, after - before);
}
function formatMB(bytes) {
  const mb = bytesToMB(bytes);
  return mb == null ? 'n/a' : mb.toFixed(1) + ' MB';
}
function pikeletWasmHeapBytes(index) {
  return index.memoryUsage?.wasmHeapBytes ?? null;
}
function systemInfo() {
  const cpus = os.cpus() || [];
  return {
    platform: process.platform,
    arch: process.arch,
    node: process.version,
    cpu: cpus[0]?.model || 'unknown',
    logicalCpus: cpus.length,
    totalMemoryBytes: os.totalmem(),
    freeMemoryBytes: os.freemem(),
    hostname: os.hostname(),
  };
}
function logSystemInfo(info) {
  log('System: ' + info.platform + '/' + info.arch + ', Node ' + info.node);
  log('CPU: ' + info.cpu + ' (' + info.logicalCpus + ' logical), RAM ' + formatMB(info.totalMemoryBytes));
}
function compactCountLabel(n) {
  if (n >= 1_000_000 && n % 1_000_000 === 0) return (n / 1_000_000) + 'M';
  if (n >= 1_000 && n % 1_000 === 0) return (n / 1_000) + 'k';
  return n.toLocaleString();
}
function plotTitle({ train, test, dim }) {
  const dataset = DATASET.toUpperCase() + '-' + dim + 'D-' + METRIC.toUpperCase() + ' ' + compactCountLabel(train.length);
  const params = 'k=' + K + ', M=' + M + ', ef_construction=' + EF_CONSTRUCTION
    + ', ef_search=' + EF_SEARCH_VALUES[0] + '-' + EF_SEARCH_VALUES[EF_SEARCH_VALUES.length - 1]
    + ', queries=' + test.length.toLocaleString();
  return dataset + ' | ' + params;
}

// =====================================================================
// Per-library build + query adapters.
// Each build* returns a handle object; each query* returns {latencies, meanRecall}.
// =====================================================================

// --- Pikelet WASM ---
async function buildPikeletWasm({ train, dim, dtype }) {
  const quantized = dtype === 'u8';
  log(`  [${dtype} wasm] build (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=${METRIC})...`);
  const rssBefore = measureRssBytes();
  const index = await Pikelet.create({
    dim, maxElements: train.length, quantized, metric: METRIC,
    M, efConstruction: EF_CONSTRUCTION, efSearch: EF_SEARCH_VALUES[0],
  });
  const t0 = performance.now();
  const batchSize = 500;
  for (let start = 0; start < train.length; start += batchSize) {
    index.addBatch(train.slice(start, Math.min(start + batchSize, train.length)));
  }
  const buildMs = performance.now() - t0;
  const heapBytes = pikeletWasmHeapBytes(index);
  const rssDelta = rssDeltaBytes(rssBefore, measureRssBytes());
  log(`  [${dtype} wasm] build: ${(buildMs / 1000).toFixed(1)}s, logical index: ${formatMB(index.memory)} (wasm heap ${formatMB(heapBytes)}, rss +${formatMB(rssDelta)})`);
  return { index, buildMs, memBytes: index.memory, memorySource: 'logical_index', wasmHeapBytes: heapBytes, rssDeltaBytes: rssDelta };
}
function queryPikeletWasm(built, test, gt, ef) {
  built.index.setEfSearch(ef);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) built.index.search(test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = built.index.search(test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(results.map(r => r.id), gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- Pikelet native ---
function buildPikeletNative({ train, dim, dtype }) {
  const quantized = dtype === 'u8' ? 1 : 0;
  log(`  [${dtype} native] build (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=${METRIC})...`);
  const rssBefore = measureRssBytes();
  const h = native.pancake_init(dim, train.length, quantized, nativeMetricValue(), M, EF_CONSTRUCTION, EF_SEARCH_VALUES[0], 108);
  if (h === 0xFFFFFFFF) throw new Error('Failed to init native index');
  const t0 = performance.now();
  const flat = new Float32Array(train.length * dim);
  for (let i = 0; i < train.length; i++) flat.set(train[i], i * dim);
  const batchSize = 10000;
  for (let start = 0; start < train.length; start += batchSize) {
    const end = Math.min(start + batchSize, train.length);
    native.pancake_bulk_insert(h, flat.subarray(start * dim, end * dim), end - start);
  }
  const buildMs = performance.now() - t0;
  const memBytes = native.pancake_memory(h);
  const rssDelta = rssDeltaBytes(rssBefore, measureRssBytes());
  log(`  [${dtype} native] build: ${(buildMs / 1000).toFixed(1)}s, logical index: ${formatMB(memBytes)} (rss +${formatMB(rssDelta)})`);
  return { handle: h, buildMs, memBytes, memorySource: 'logical_index', rssDeltaBytes: rssDelta };
}
function queryPikeletNative(built, test, gt, ef) {
  native.pancake_set_ef(built.handle, ef);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) native.pancake_query(built.handle, test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const result = native.pancake_query(built.handle, test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(Array.from(result.ids), gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- USearch ---
// The JS binding only honors expansion_search at construction. We build once,
// save to disk, then view() with a fresh index per ef_search value.
function buildUsearch({ train, dim, dtype }) {
  const quantization = dtype;
  const metric = usearchMetricValue();
  log(`  [usearch-${dtype}] build (connectivity=${M}, expansion_add=${EF_CONSTRUCTION}, metric=${metric}, quantization=${quantization})...`);
  const rssBefore = measureRssBytes();
  const index = new usearch.Index({
    metric, connectivity: M, dimensions: dim,
    quantization, expansion_add: EF_CONSTRUCTION, expansion_search: EF_SEARCH_VALUES[0],
  });
  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) index.add(BigInt(i), train[i]);
  const buildMs = performance.now() - t0;
  const rssDelta = rssDeltaBytes(rssBefore, measureRssBytes());
  const savePath = path.join(RESULTS_DIR, `_usearch_${dtype}_${timestamp}.bin`);
  index.save(savePath);
  const fileBytes = fs.statSync(savePath).size;
  log(`  [usearch-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s, serialized index: ${formatMB(fileBytes)} (rss +${formatMB(rssDelta)}, saved to ${savePath})`);
  return { buildMs, memBytes: fileBytes, memorySource: 'serialized_index', rssDeltaBytes: rssDelta, savePath, quantization, dim, metric };
}
function queryUsearch(built, test, gt, ef) {
  const view = new usearch.Index({
    metric: built.metric, connectivity: M, dimensions: built.dim,
    quantization: built.quantization, expansion_search: ef,
  });
  view.view(built.savePath);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) view.search(test[i], K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const st = performance.now();
    const results = view.search(test[i], K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(Array.from(results.keys).map(Number), gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- USearch WASM ---
// Built from the official WASM target. The generated JS glue does not expose
// heap accessors, so this adapter uses the exported C ABI directly.
const USEARCH_WASM_METRIC = { cos: 1, l2sq: 3 };
const USEARCH_WASM_SCALAR = { f32: 1, f16: 3, i8: 4 };
const USEARCH_WASM_OPTIONS_SIZE = 32;
const USEARCH_WASM_MINIFIED_EXPORTS_V26 = {
  usearch_init: 's',
  usearch_free: 't',
  usearch_serialized_length: 'u',
  usearch_save_buffer: 'y',
  usearch_view_buffer: 'A',
  usearch_reserve: 'F',
  usearch_add: 'G',
  usearch_search: 'J',
  emscripten_builtin_memalign: 'N',
};
const USEARCH_WASM_MINIFIED_EXPORTS_V226 = {
  usearch_init: 'v',
  usearch_free: 'w',
  usearch_serialized_length: 'x',
  usearch_save_buffer: 'C',
  usearch_view_buffer: 'E',
  usearch_reserve: 'U',
  usearch_add: 'V',
  usearch_search: 'Y',
  emscripten_builtin_memalign: 'ea',
};

function makeUsearchWasmImports(memoryRef) {
  const errnoNosys = -52;
  const writeU32 = (ptr, value) => {
    if (!memoryRef.memory || !ptr) return;
    new DataView(memoryRef.memory.buffer).setUint32(ptr, value >>> 0, true);
  };
  const writeU64 = (ptr, value) => {
    if (!memoryRef.memory || !ptr) return;
    new DataView(memoryRef.memory.buffer).setBigUint64(ptr, BigInt(value), true);
  };
  const cxaThrow = () => { throw new Error('USearch WASM C++ exception'); };
  const fcntl64 = () => 0;
  const fstat64 = () => errnoNosys;
  const ioctl = () => errnoNosys;
  const newfstatat = () => errnoNosys;
  const openat = () => errnoNosys;
  const abortJs = () => { throw new Error('USearch WASM aborted'); };
  const mmapJs = () => 0;
  const munmapJs = () => 0;
  const heapMax = () => memoryRef.memory ? memoryRef.memory.buffer.byteLength : 0x7fffffff;
  const now = () => performance.now();
  const resizeHeap = (requestedSize) => {
    const memory = memoryRef.memory;
    if (!memory) return 0;
    const oldBytes = memory.buffer.byteLength;
    if (requestedSize <= oldBytes) return 1;
    const pageSize = 64 * 1024;
    const pages = Math.ceil((requestedSize - oldBytes) / pageSize);
    try {
      memory.grow(pages);
      return 1;
    } catch (_) {
      return 0;
    }
  };
  const fdClose = () => 0;
  const fdRead = (_fd, _iov, _iovcnt, pnum) => { writeU32(pnum, 0); return 0; };
  const fdSeek = (_fd, _offset, _whence, newOffset) => { writeU64(newOffset, 0); return 0; };
  const fdWrite = (_fd, iov, iovcnt, pnum) => {
    const memory = memoryRef.memory;
    if (!memory) { writeU32(pnum, 0); return 0; }
    const data = new DataView(memory.buffer);
    let written = 0;
    for (let i = 0; i < iovcnt; i++) {
      written += data.getUint32(iov + i * 8 + 4, true);
    }
    writeU32(pnum, written);
    return 0;
  };
  return {
    env: {
      __assert_fail: () => { throw new Error('USearch WASM assertion failed'); },
      __cxa_throw: cxaThrow,
      __syscall_openat: openat,
      __syscall_fcntl64: fcntl64,
      __syscall_ioctl: ioctl,
      __syscall_fstat64: fstat64,
      __syscall_stat64: fstat64,
      __syscall_newfstatat: newfstatat,
      __syscall_lstat64: fstat64,
      emscripten_get_now: now,
      _munmap_js: munmapJs,
      _mmap_js: mmapJs,
      _abort_js: abortJs,
      emscripten_resize_heap: resizeHeap,
      emscripten_get_heap_max: heapMax,
    },
    wasi_snapshot_preview1: {
      fd_close: fdClose,
      fd_read: fdRead,
      fd_seek: fdSeek,
      fd_write: fdWrite,
    },
    // Optimized Emscripten builds may minify imports into a single module.
    a: {
      a: cxaThrow,
      e: fcntl64,
      i: fstat64,
      k: ioctl,
      h: newfstatat,
      f: openat,
      m: abortJs,
      o: mmapJs,
      p: munmapJs,
      l: heapMax,
      b: now,
      n: resizeHeap,
      c: fdClose,
      j: fdRead,
      g: fdSeek,
      d: fdWrite,
    },
  };
}

async function loadUsearchWasmModule(wasmPath) {
  const memoryRef = { memory: null };
  const bytes = fs.readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(bytes, makeUsearchWasmImports(memoryRef));
  memoryRef.memory = instance.exports.memory || instance.exports.q;
  const stackInit = instance.exports.emscripten_stack_init;
  if (stackInit) stackInit();
  (instance.exports.__wasm_call_ctors || instance.exports.r)();
  return { exports: instance.exports, memoryRef };
}

class UsearchWasmRuntime {
  constructor(module) {
    this.exports = module.exports;
    this.memoryRef = module.memoryRef;
    this.minified = !this.exports.usearch_init;
    this.minifiedExports = this.exports.ea ? USEARCH_WASM_MINIFIED_EXPORTS_V226 : USEARCH_WASM_MINIFIED_EXPORTS_V26;
    this.useBigIntKeys = null;
  }
  fn(name, minifiedName) {
    const candidates = [name, this.minifiedExports[name], minifiedName].filter(Boolean);
    for (const candidate of candidates) {
      if (this.exports[candidate]) return this.exports[candidate];
    }
    throw new Error(`USearch WASM export not found: ${name}`);
  }
  memory() { return this.memoryRef.memory; }
  u8() { return new Uint8Array(this.memory().buffer); }
  i32() { return new Int32Array(this.memory().buffer); }
  u32() { return new Uint32Array(this.memory().buffer); }
  f32() { return new Float32Array(this.memory().buffer); }
  dataView() { return new DataView(this.memory().buffer); }
  stackSave() { return this.exports.emscripten_stack_get_current ? this.exports.emscripten_stack_get_current() : 0; }
  stackAlloc(bytes) {
    return this.exports._emscripten_stack_alloc
      ? this.exports._emscripten_stack_alloc(bytes)
      : this.alloc(bytes);
  }
  stackRestore(ptr) {
    if (ptr && this.exports._emscripten_stack_restore) this.exports._emscripten_stack_restore(ptr);
  }
  alloc(bytes, alignment = 16) {
    const memalign = this.fn('emscripten_builtin_memalign', 'N');
    const ptr = memalign(alignment, bytes);
    if (!ptr) throw new Error(`USearch WASM allocation failed (${bytes} bytes)`);
    return ptr;
  }
  readCString(ptr) {
    if (!ptr) return '';
    const heap = this.u8();
    let end = ptr;
    while (end < heap.length && heap[end] !== 0) end++;
    return Buffer.from(heap.subarray(ptr, end)).toString('utf8');
  }
  resetError(errPtr) {
    this.u32()[errPtr >> 2] = 0;
  }
  checkError(errPtr, context) {
    const err = this.u32()[errPtr >> 2];
    if (err) throw new Error(`${context}: ${this.readCString(err) || `error pointer ${err}`}`);
  }
  writeOptions(ptr, { dim, ef, quantization }) {
    this.u8().fill(0, ptr, ptr + USEARCH_WASM_OPTIONS_SIZE);
    const words = this.i32();
    words[(ptr + 0) >> 2] = USEARCH_WASM_METRIC[usearchMetricValue()];
    words[(ptr + 4) >> 2] = 0; // custom metric function pointer
    words[(ptr + 8) >> 2] = quantization;
    words[(ptr + 12) >> 2] = dim;
    words[(ptr + 16) >> 2] = M;
    words[(ptr + 20) >> 2] = EF_CONSTRUCTION;
    words[(ptr + 24) >> 2] = ef;
    words[(ptr + 28) >> 2] = 0; // multi
  }
  initIndex(dim, ef, quantization, errPtr) {
    const sp = this.stackSave();
    try {
      const optsPtr = this.stackAlloc(USEARCH_WASM_OPTIONS_SIZE);
      this.writeOptions(optsPtr, { dim, ef, quantization });
      this.resetError(errPtr);
      const handle = this.fn('usearch_init', 's')(optsPtr, errPtr);
      this.checkError(errPtr, 'usearch_init');
      if (!handle) throw new Error('usearch_init returned null');
      return handle;
    } finally {
      this.stackRestore(sp);
    }
  }
  freeIndex(handle, errPtr) {
    if (!handle) return;
    this.resetError(errPtr);
    this.fn('usearch_free', 't')(handle, errPtr);
    this.checkError(errPtr, 'usearch_free');
  }
  add(handle, key, vecPtr, vectorKind, errPtr) {
    const addFn = this.fn('usearch_add', 'G');
    if (this.useBigIntKeys === true) return addFn(handle, BigInt(key), vecPtr, vectorKind, errPtr);
    if (this.useBigIntKeys === false) return addFn(handle, key, vecPtr, vectorKind, errPtr);
    try {
      const result = addFn(handle, key, vecPtr, vectorKind, errPtr);
      this.useBigIntKeys = false;
      return result;
    } catch (e) {
      if (!(e instanceof TypeError) || !/BigInt/i.test(String(e.message))) throw e;
      const result = addFn(handle, BigInt(key), vecPtr, vectorKind, errPtr);
      this.useBigIntKeys = true;
      return result;
    }
  }
}

async function buildUsearchWasm({ train, dim, dtype, wasmPath }) {
  if (!['i8', 'f16', 'f32'].includes(dtype)) throw new Error(`Unsupported usearch-wasm dtype: ${dtype}`);
  const metric = usearchMetricValue();
  const quantization = USEARCH_WASM_SCALAR[dtype];
  log(`  [usearch-wasm-${dtype}] build (connectivity=${M}, expansion_add=${EF_CONSTRUCTION}, metric=${metric}, quantization=${dtype}, wasm=${wasmPath})...`);
  const runtime = new UsearchWasmRuntime(await loadUsearchWasmModule(wasmPath));
  const rssBefore = measureRssBytes();
  const sp = runtime.stackSave();
  let handle = 0;
  try {
    const errPtr = runtime.stackAlloc(4);
    const vecPtr = runtime.stackAlloc(dim * 4);
    handle = runtime.initIndex(dim, EF_SEARCH_VALUES[0], quantization, errPtr);
    runtime.resetError(errPtr);
    runtime.fn('usearch_reserve', 'F')(handle, train.length, errPtr);
    runtime.checkError(errPtr, 'usearch_reserve');

    const t0 = performance.now();
    for (let i = 0; i < train.length; i++) {
      runtime.f32().set(train[i], vecPtr >> 2);
      runtime.resetError(errPtr);
      runtime.add(handle, i, vecPtr, USEARCH_WASM_SCALAR.f32, errPtr);
      runtime.checkError(errPtr, `usearch_add(${i})`);
    }
    const buildMs = performance.now() - t0;

    runtime.resetError(errPtr);
    const serializedBytes = runtime.fn('usearch_serialized_length', 'u')(handle, errPtr);
    runtime.checkError(errPtr, 'usearch_serialized_length');
    const snapshotPtr = runtime.alloc(Number(serializedBytes), 16);
    runtime.resetError(errPtr);
    runtime.fn('usearch_save_buffer', 'y')(handle, snapshotPtr, serializedBytes, errPtr);
    runtime.checkError(errPtr, 'usearch_save_buffer');

    const heapBytes = runtime.memory().buffer.byteLength;
    const rssDelta = rssDeltaBytes(rssBefore, measureRssBytes());
    log(`  [usearch-wasm-${dtype}] build: ${(buildMs / 1000).toFixed(1)}s, serialized index: ${formatMB(Number(serializedBytes))} (wasm heap ${formatMB(heapBytes)}, rss +${formatMB(rssDelta)})`);
    return {
      runtime, handle, snapshotPtr, snapshotBytes: Number(serializedBytes),
      buildMs, memBytes: Number(serializedBytes), memorySource: 'serialized_index',
      wasmHeapBytes: heapBytes, rssDeltaBytes: rssDelta, dim, metric, quantization, wasmPath,
    };
  } catch (e) {
    if (handle) {
      try {
        const errPtr = runtime.stackAlloc(4);
        runtime.freeIndex(handle, errPtr);
      } catch (_) {}
    }
    throw e;
  } finally {
    runtime.stackRestore(sp);
  }
}

function queryUsearchWasm(built, test, gt, ef) {
  const runtime = built.runtime;
  const sp = runtime.stackSave();
  let viewHandle = 0;
  try {
    const errPtr = runtime.stackAlloc(4);
    const queryPtr = runtime.stackAlloc(built.dim * 4);
    const keysPtr = runtime.stackAlloc(K * 8);
    const distancesPtr = runtime.stackAlloc(K * 4);
    viewHandle = runtime.initIndex(built.dim, ef, built.quantization, errPtr);
    runtime.resetError(errPtr);
    runtime.fn('usearch_view_buffer', 'A')(viewHandle, built.snapshotPtr, built.snapshotBytes, errPtr);
    runtime.checkError(errPtr, 'usearch_view_buffer');

    const search = (queryVec) => {
      runtime.f32().set(queryVec, queryPtr >> 2);
      runtime.resetError(errPtr);
      const count = runtime.fn('usearch_search', 'J')(viewHandle, queryPtr, USEARCH_WASM_SCALAR.f32, K, keysPtr, distancesPtr, errPtr);
      runtime.checkError(errPtr, 'usearch_search');
      const data = runtime.dataView();
      const ids = new Array(Number(count));
      for (let j = 0; j < ids.length; j++) ids[j] = Number(data.getBigUint64(keysPtr + j * 8, true));
      return ids;
    };

    for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) search(test[i]);
    const latencies = new Array(test.length);
    let totalRecall = 0;
    for (let i = 0; i < test.length; i++) {
      const st = performance.now();
      const ids = search(test[i]);
      latencies[i] = performance.now() - st;
      totalRecall += recall(ids, gt[i]);
    }
    return { latencies, meanRecall: totalRecall / test.length };
  } finally {
    if (viewHandle) {
      try {
        const errPtr = runtime.stackAlloc(4);
        runtime.freeIndex(viewHandle, errPtr);
      } catch (_) {}
    }
    runtime.stackRestore(sp);
  }
}

// --- hnswlib-node ---
function buildHnswlib({ train, dim }) {
  const metric = hnswlibMetricValue();
  log(`  [hnswlib-f32] build (M=${M}, ef_c=${EF_CONSTRUCTION}, metric=${metric})...`);
  const rssBefore = measureRssBytes();
  const index = new HierarchicalNSW(metric, dim);
  index.initIndex(train.length, M, EF_CONSTRUCTION, 100);
  const t0 = performance.now();
  for (let i = 0; i < train.length; i++) index.addPoint(Array.from(train[i]), i);
  const buildMs = performance.now() - t0;
  const rssDelta = rssDeltaBytes(rssBefore, measureRssBytes());
  log(`  [hnswlib-f32] build: ${(buildMs / 1000).toFixed(1)}s, rss delta: ${formatMB(rssDelta)} (no index-size API)`);
  return { index, buildMs, memBytes: null, memorySource: null, rssDeltaBytes: rssDelta };
}
function queryHnswlib(built, test, gt, ef) {
  built.index.setEf(ef);
  for (let i = 0; i < WARMUP_QUERIES && i < test.length; i++) built.index.searchKnn(Array.from(test[i]), K);
  const latencies = new Array(test.length);
  let totalRecall = 0;
  for (let i = 0; i < test.length; i++) {
    const query = Array.from(test[i]);
    const st = performance.now();
    const results = built.index.searchKnn(query, K);
    latencies[i] = performance.now() - st;
    totalRecall += recall(results.neighbors, gt[i]);
  }
  return { latencies, meanRecall: totalRecall / test.length };
}

// --- Dispatch ---
async function build(config, dataset) {
  const { train, dim } = dataset;
  switch (config.library) {
    case 'pikelet':
      return config.runtime === 'wasm'
        ? await buildPikeletWasm({ train, dim, dtype: config.dtype })
        : buildPikeletNative({ train, dim, dtype: config.dtype });
    case 'usearch': return buildUsearch({ train, dim, dtype: config.dtype });
    case 'usearch-wasm': return await buildUsearchWasm({ train, dim, dtype: config.dtype, wasmPath: config.wasmPath });
    case 'hnswlib': return buildHnswlib({ train, dim });
  }
}
function query(config, built, dataset, ef) {
  const { test, groundTruth } = dataset;
  switch (config.library) {
    case 'pikelet':
      return config.runtime === 'wasm'
        ? queryPikeletWasm(built, test, groundTruth, ef)
        : queryPikeletNative(built, test, groundTruth, ef);
    case 'usearch': return queryUsearch(built, test, groundTruth, ef);
    case 'usearch-wasm': return queryUsearchWasm(built, test, groundTruth, ef);
    case 'hnswlib': return queryHnswlib(built, test, groundTruth, ef);
  }
}
function cleanup(config, built) {
  if (config.library === 'pikelet' && config.runtime === 'native') native.pancake_dispose(built.handle);
  else if (config.library === 'usearch-wasm' && built.runtime) {
    const sp = built.runtime.stackSave();
    try {
      const errPtr = built.runtime.stackAlloc(4);
      built.runtime.freeIndex(built.handle, errPtr);
    } finally {
      built.runtime.stackRestore(sp);
    }
  }
  else if (built.index && typeof built.index.dispose === 'function') built.index.dispose();
  if (built.savePath && fs.existsSync(built.savePath)) fs.unlinkSync(built.savePath);
}

// =====================================================================
// Sweep driver
// =====================================================================
async function sweepOne(config, dataset) {
  log(`\n${'='.repeat(70)}`);
  log(`Config: ${config.label}`);
  log('='.repeat(70));

  const built = await build(config, dataset);
  const points = [];

  for (const ef of EF_SEARCH_VALUES) {
    log(`\n  ef_search=${ef}`);
    const reps = [];
    for (let rep = 0; rep < REPETITIONS; rep++) {
      const { latencies, meanRecall } = query(config, built, dataset, ef);
      const sorted = [...latencies].sort((a, b) => a - b);
      const avgLatency = mean(latencies);
      reps.push({
        recall: meanRecall, qps: 1000 / avgLatency, meanLatencyMs: avgLatency,
        p50: percentile(sorted, 0.5), p95: percentile(sorted, 0.95), p99: percentile(sorted, 0.99),
      });
      log(`    rep ${rep + 1}: recall=${(meanRecall * 100).toFixed(2)}%  qps=${(1000 / avgLatency).toFixed(0)}  `
        + `p50=${percentile(sorted, 0.5).toFixed(3)}ms  p99=${percentile(sorted, 0.99).toFixed(3)}ms`);
    }
    const summary = {
      ef_search: ef,
      recall_mean: mean(reps.map(r => r.recall)), recall_std: stddev(reps.map(r => r.recall)),
      qps_mean: mean(reps.map(r => r.qps)),       qps_std: stddev(reps.map(r => r.qps)),
      p50_mean: mean(reps.map(r => r.p50)), p95_mean: mean(reps.map(r => r.p95)), p99_mean: mean(reps.map(r => r.p99)),
      reps,
    };
    log(`    summary: recall=${(summary.recall_mean * 100).toFixed(2)}% (+/-${(summary.recall_std * 100).toFixed(2)}) `
      + `qps=${summary.qps_mean.toFixed(0)} (+/-${summary.qps_std.toFixed(0)})`);
    points.push(summary);
  }

  cleanup(config, built);

  return {
    label: config.label, library: config.library, runtime: config.runtime || null,
    dtype: config.dtype, tunable: true,
    buildMs: built.buildMs,
    memMB: bytesToMB(built.memBytes),
    memorySource: built.memorySource || (built.memBytes != null ? 'reported' : null),
    wasmHeapMB: bytesToMB(built.wasmHeapBytes),
    rssDeltaMB: bytesToMB(built.rssDeltaBytes),
    params: { M, ef_construction: EF_CONSTRUCTION, K },
    points,
  };
}

// =====================================================================
// Analysis: Pareto frontier + interpolated equal-recall curves
// =====================================================================

// A point dominates another if it has >= recall AND >= qps (and strictly better
// on at least one). The frontier is the set of non-dominated points.
function paretoFrontier(points) {
  const pts = points.map(p => ({ ef_search: p.ef_search, recall: p.recall_mean, qps: p.qps_mean }));
  const frontier = pts.filter(a =>
    !pts.some(b => b !== a && b.recall >= a.recall && b.qps >= a.qps && (b.recall > a.recall || b.qps > a.qps)));
  // Sort along recall for a clean curve.
  return frontier.sort((x, y) => x.recall - y.recall);
}

// Log-linear interpolation of QPS at a target recall along a frontier.
// Returns null if the target falls outside the frontier's recall span.
function qpsAtRecall(frontier, targetRecall) {
  if (frontier.length === 0) return null;
  if (frontier.length === 1) {
    return Math.abs(frontier[0].recall - targetRecall) < 1e-9 ? frontier[0].qps : null;
  }
  const lo = frontier[0].recall, hi = frontier[frontier.length - 1].recall;
  if (targetRecall < lo - 1e-9 || targetRecall > hi + 1e-9) return null;
  for (let i = 1; i < frontier.length; i++) {
    if (frontier[i].recall >= targetRecall) {
      const r0 = frontier[i - 1].recall, r1 = frontier[i].recall;
      const q0 = frontier[i - 1].qps, q1 = frontier[i].qps;
      if (r1 === r0) return Math.max(q0, q1);
      const t = (targetRecall - r0) / (r1 - r0);
      return Math.exp(Math.log(q0) + t * (Math.log(q1) - Math.log(q0))); // log-linear in QPS
    }
  }
  return frontier[frontier.length - 1].qps;
}

function recallSpan(frontier) {
  if (!frontier || frontier.length === 0) return null;
  return { lo: frontier[0].recall, hi: frontier[frontier.length - 1].recall };
}

function evenlySpacedTargets(lo, hi, count = 9) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (hi < lo) return [];
  if (Math.abs(hi - lo) < 1e-9) return [hi];

  const targets = [];
  for (let i = 0; i < count; i++) targets.push(lo + ((hi - lo) * i) / (count - 1));
  return targets;
}

function uniqueSortedTargets(targets) {
  const sorted = targets
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  const unique = [];
  for (const target of sorted) {
    if (unique.length === 0 || Math.abs(unique[unique.length - 1] - target) > 1e-9) {
      unique.push(target);
    }
  }
  return unique;
}

function buildRecallTargets(frontiers, labels) {
  const spans = labels.map(label => recallSpan(frontiers[label])).filter(Boolean);
  if (spans.length === 0) return [];

  const unionLo = Math.min(...spans.map(s => s.lo));
  const unionHi = Math.max(...spans.map(s => s.hi));
  const grid = evenlySpacedTargets(unionLo, unionHi);
  const observed = labels.flatMap(label =>
    (frontiers[label] || []).map(point => point.recall)
  );

  return uniqueSortedTargets([...grid, ...observed]);
}

function buildAnalysis(allResults) {
  const sweepable = allResults.filter(r => r.tunable && r.points.length >= 2);
  const frontiers = {};
  for (const r of allResults) frontiers[r.label] = paretoFrontier(r.points);

  // Common recall grid for equal-recall curves. Use the union of frontier
  // ranges so low-ceiling configs do not pin the comparison table; libraries
  // that cannot reach a target are left blank.
  const sweepableLabels = sweepable.map(r => r.label);
  const candidateTargets = buildRecallTargets(frontiers, sweepableLabels);
  const equalRecall = candidateTargets.map(target => {
    const row = { recall: target };
    for (const r of sweepable) row[r.label] = qpsAtRecall(frontiers[r.label], target);
    return row;
  });

  return { frontiers, equalRecall, sweepableLabels };
}

// =====================================================================
// Output writers
// =====================================================================
function writeRawCsv(allResults, p) {
  const rows = [['label', 'library', 'runtime', 'dtype', 'tunable', 'ef_search',
                 'recall', 'recall_std', 'qps', 'qps_std', 'p50_ms', 'p95_ms', 'p99_ms',
                 'build_s', 'memory_mb', 'memory_source', 'wasm_heap_mb', 'rss_delta_mb']];
  for (const r of allResults) for (const pt of r.points) {
    rows.push([
      r.label, r.library, r.runtime || '', r.dtype, r.tunable, pt.ef_search,
      pt.recall_mean.toFixed(5), pt.recall_std.toFixed(5),
      pt.qps_mean.toFixed(2), pt.qps_std.toFixed(2),
      pt.p50_mean.toFixed(4), pt.p95_mean.toFixed(4), pt.p99_mean.toFixed(4),
      (r.buildMs / 1000).toFixed(3),
      r.memMB == null ? '' : r.memMB.toFixed(2),
      r.memorySource || '',
      r.wasmHeapMB == null ? '' : r.wasmHeapMB.toFixed(2),
      r.rssDeltaMB == null ? '' : r.rssDeltaMB.toFixed(2),
    ]);
  }
  fs.writeFileSync(p, rows.map(r => r.join(',')).join('\n') + '\n');
}

function writeFrontierCsv(frontiers, p) {
  const rows = [['label', 'ef_search', 'recall', 'qps']];
  for (const [label, fr] of Object.entries(frontiers))
    for (const pt of fr) rows.push([label, pt.ef_search, pt.recall.toFixed(5), pt.qps.toFixed(2)]);
  fs.writeFileSync(p, rows.map(r => r.join(',')).join('\n') + '\n');
}

function writeEqualRecallCsv(equalRecall, labels, p) {
  const rows = [['recall', ...labels]];
  for (const row of equalRecall)
    rows.push([row.recall.toFixed(3), ...labels.map(l => (row[l] == null ? '' : row[l].toFixed(2)))]);
  fs.writeFileSync(p, rows.map(r => r.join(',')).join('\n') + '\n');
}

function printFrontierTables(allResults, analysis) {
  log(`\n${'='.repeat(70)}`);
  log('Pareto frontiers (non-dominated QPS-recall points)');
  log('='.repeat(70));
  for (const r of allResults) {
    const fr = analysis.frontiers[r.label];
    log(`\n  ${r.label}`);
    log(`    ${'ef'.padStart(5)}  ${'recall'.padStart(8)}  ${'qps'.padStart(9)}`);
    for (const pt of fr)
      log(`    ${String(pt.ef_search).padStart(5)}  ${(pt.recall * 100).toFixed(2).padStart(7)}%  ${pt.qps.toFixed(0).padStart(9)}`);
  }

  log(`\n${'='.repeat(70)}`);
  log('Interpolated equal-recall QPS (log-linear along each frontier)');
  log('  Blank = recall target outside that library\'s frontier span.');
  log('='.repeat(70));
  const labels = analysis.sweepableLabels;
  log('\n  ' + 'recall'.padStart(7) + labels.map(l => l.padStart(20)).join(''));
  log('  ' + '-'.repeat(7 + labels.length * 20));
  for (const row of analysis.equalRecall) {
    let line = '  ' + `${(row.recall * 100).toFixed(1)}%`.padStart(7);
    for (const l of labels) line += (row[l] == null ? 'n/a' : row[l].toFixed(0)).padStart(20);
    log(line);
  }
}

function writePlot(title) {
  if (!WRITE_PLOT) {
    log('  plot:           skipped (--no-plot)');
    return false;
  }

  const script = path.join(__dirname, 'plot_pareto.py');
  const result = spawnSync('python3', [script, CSV_PATH, title], { encoding: 'utf8' });
  if (result.stdout) {
    for (const line of result.stdout.trim().split(/\r?\n/).filter(Boolean)) log(`  plot:           ${line}`);
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.error || '').toString().trim();
    log(`  plot:           failed${detail ? `: ${detail}` : ''}`);
    return false;
  }
  return fs.existsSync(PNG_PATH);
}

// =====================================================================
// Main
// =====================================================================
async function main() {
  const hdf5Path = resolveDatasetPath(DS.hdf5File);
  const idxImagePath = hdf5Path ? null : resolveDatasetPath(DS.imageFile);
  const basePath = hdf5Path || idxImagePath ? null : resolveDatasetPath(DS.baseFile(N_BASE));
  const queryPath = hdf5Path || idxImagePath ? null : resolveDatasetPath(DS.queryFile);
  const gtPath = hdf5Path || idxImagePath ? null : resolveDatasetPath(DS.gtFile);
  for (const f of [hdf5Path || idxImagePath || basePath, ...(queryPath ? [queryPath] : []), ...(gtPath ? [gtPath] : [])]) {
    if (!fs.existsSync(f)) { log(`Missing file: ${f}`); process.exit(1); }
  }

  const sysInfo = systemInfo();

  log('='.repeat(70));
  log(`Pareto-Frontier Benchmark (${DATASET.toUpperCase()}, ${METRIC})`);
  log('='.repeat(70));
  logSystemInfo(sysInfo);
  log(`Configs: ${CONFIGS.map(c => c.label).join(', ')}`);
  log('\nLoading dataset...');
  const loaded = hdf5Path
    ? loadHdf5Dataset(hdf5Path, READ_LIMIT, N_QUERIES)
    : idxImagePath
      ? readIdxImages(idxImagePath, READ_LIMIT, N_QUERIES)
      : (() => {
        const { vectors: train, dim } = readFvecs(basePath, READ_LIMIT);
        const { vectors: test } = readFvecs(queryPath, N_QUERIES);
        return { train, test, dim, groundTruth: null };
      })();
  let { train, test, dim } = loaded;
  let forceGroundTruthRecompute = false;
  const preparedTrain = prepareIndexableVectors(train, dim, 'base');
  const preparedTest = prepareIndexableVectors(test, dim, 'query');
  if (
    preparedTrain.dropped.length || preparedTest.dropped.length ||
    preparedTrain.replaced.length || preparedTest.replaced.length
  ) {
    train = preparedTrain.vectors;
    test = preparedTest.vectors;
    forceGroundTruthRecompute = true;
    if (train.length < K) {
      log(`ERROR: only ${train.length} indexable base vectors remain, fewer than k=${K}.`);
      process.exit(1);
    }
    if (test.length === 0) {
      log('ERROR: no indexable query vectors remain.');
      process.exit(1);
    }
    log('  Recomputing ground truth because vector preparation changed the benchmark inputs.');
  }
  log(`  Base:    ${train.length.toLocaleString()} vectors, ${dim}D`);
  log(`  Queries: ${test.length.toLocaleString()}`);
  log(`  Metric:  ${METRIC}`);
  log(`  k=${K}, M=${M}, ef_construction=${EF_CONSTRUCTION}`);
  log(`  ef_search sweep: [${EF_SEARCH_VALUES.join(', ')}]`);
  log(`  Repetitions: ${REPETITIONS}, Warmup: ${WARMUP_QUERIES}`);

  // Precomputed .ivecs/HDF5 neighbor matrices are aligned to the full base set.
  // When --count selects a subset, recompute metric-aware GT against that subset
  // so recall is not compared to neighbors that were never indexed.
  const usesFullBuiltInBase = !DS.defaultCount || train.length >= DS.defaultCount;
  const canUsePrecomputedGt = !REGENERATE_GT && !forceGroundTruthRecompute && usesFullBuiltInBase;
  let groundTruth;
  if (loaded.groundTruth && canUsePrecomputedGt) {
    groundTruth = loaded.groundTruth;
  } else if (gtPath && canUsePrecomputedGt) {
    groundTruth = readIvecs(gtPath, N_QUERIES);
    log(`  Loaded precomputed ground truth: ${groundTruth.length} rows, first ${K} neighbors each`);
  } else {
    if ((loaded.groundTruth || gtPath) && !usesFullBuiltInBase) {
      log(`  Subset run detected: indexed ${train.length.toLocaleString()} of ${DS.defaultCount.toLocaleString()} base vectors.`);
      log('  Recomputing ground truth for the selected subset.');
    } else if ((loaded.groundTruth || gtPath) && REGENERATE_GT) {
      log('  --regenerate-gt set; recomputing ground truth instead of using precomputed neighbors.');
    }
    groundTruth = (REGENERATE_GT || forceGroundTruthRecompute) ? null : loadGroundTruth();
    if (groundTruth) log(`\nLoaded cached ground truth (${groundTruth.length} queries) from ${GT_CACHE_PATH}`);
    else { groundTruth = computeGroundTruth(train, test, dim); saveGroundTruth(groundTruth); }
  }

  const dataset = { train, test, groundTruth, dim };

  const allResults = [];
  for (const config of CONFIGS) allResults.push(await sweepOne(config, dataset));

  const analysis = buildAnalysis(allResults);

  // Outputs
  fs.writeFileSync(JSON_PATH, JSON.stringify({
    benchmark: `pikelet-pareto-frontier-${DATASET}-${METRIC}`,
    timestamp: new Date().toISOString(),
    dataset: {
      name: DATASET,
      vectors: train.length,
      queries: test.length,
      dim,
      metric: METRIC,
      zeroVectorPolicy: ZERO_VECTOR_POLICY,
      source: hdf5Path || DATA_DIR,
      hdf5File: hdf5Path,
      imageFile: idxImagePath,
      baseFile: basePath,
      queryFile: queryPath,
      groundTruthFile: gtPath,
    },
    memory: {
      note: 'memory_mb is the stable index-size proxy: Pikelet reports logical index bytes, while USearch uses serialized index bytes. hnswlib-node does not expose index size, so memory_mb is null for hnswlib. wasm_heap_mb reports the full WebAssembly heap buffer when available. rss_delta_mb is process RSS growth during build and is approximate/runtime-dependent.',
    },
    system: sysInfo,
    params: {
      K,
      M,
      EF_CONSTRUCTION,
      EF_SEARCH_VALUES,
      REPETITIONS,
      WARMUP_QUERIES,
      CONFIGS: CONFIGS.map(c => c.label),
    },
    results: allResults,
    frontiers: analysis.frontiers,
    equal_recall: analysis.equalRecall,
    outputs: {
      log: LOG_PATH,
      json: JSON_PATH,
      csv: CSV_PATH,
      frontierCsv: FRONTIER_CSV,
      equalRecallCsv: EQRECALL_CSV,
      plotPng: WRITE_PLOT ? PNG_PATH : null,
    },
  }, null, 2) + '\n');
  writeRawCsv(allResults, CSV_PATH);
  writeFrontierCsv(analysis.frontiers, FRONTIER_CSV);
  writeEqualRecallCsv(analysis.equalRecall, analysis.sweepableLabels, EQRECALL_CSV);
  const plotWritten = writePlot(plotTitle({ train, test, dim }));

  printFrontierTables(allResults, analysis);

  log(`\n${'='.repeat(70)}`);
  log('Outputs:');
  log(`  log:            ${LOG_PATH}`);
  log(`  json:           ${JSON_PATH}`);
  log(`  raw sweep csv:  ${CSV_PATH}`);
  log(`  frontier csv:   ${FRONTIER_CSV}`);
  log(`  equal-recall:   ${EQRECALL_CSV}`);
  log(`  plot png:       ${plotWritten ? PNG_PATH : '(not written)'}`);
  logStream.end();
}

main().catch(e => { console.error(e); process.exit(1); });
