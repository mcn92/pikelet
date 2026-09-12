#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const Pikelet = require('../pikelet.js');

const OPERATIONAL_EF_SEARCH = 80;

function arg(name, fallback = null) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

function parseIntArg(name, fallback) {
  const value = Number.parseInt(arg(name, String(fallback)), 10);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`--${name} must be a positive integer`);
  return value;
}

function parseNumberListArg(name, fallback) {
  const raw = arg(name, null);
  if (raw == null) return fallback;
  const values = raw.split(',').map((x) => Number(x.trim())).filter((x) => Number.isFinite(x) && x >= 0);
  if (!values.length) throw new Error(`--${name} must contain at least one non-negative number`);
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function readFvecs(filePath, limit) {
  const buffer = fs.readFileSync(filePath);
  const vectors = [];
  let offset = 0;
  while (offset + 4 <= buffer.byteLength && vectors.length < limit) {
    const dim = buffer.readInt32LE(offset);
    offset += 4;
    const bytes = dim * 4;
    if (dim <= 0 || offset + bytes > buffer.byteLength) throw new Error(`Invalid fvecs record in ${filePath}`);
    const vector = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      vector[d] = buffer.readFloatLE(offset);
      offset += 4;
    }
    vectors.push(vector);
  }
  return vectors;
}

function readIvecs(filePath, limit) {
  const buffer = fs.readFileSync(filePath);
  const vectors = [];
  let offset = 0;
  while (offset + 4 <= buffer.byteLength && vectors.length < limit) {
    const dim = buffer.readInt32LE(offset);
    offset += 4;
    const bytes = dim * 4;
    if (dim <= 0 || offset + bytes > buffer.byteLength) throw new Error(`Invalid ivecs record in ${filePath}`);
    const ids = [];
    for (let d = 0; d < dim; d++) {
      ids.push(buffer.readInt32LE(offset));
      offset += 4;
    }
    vectors.push(ids);
  }
  return vectors;
}

function summarize(values) {
  if (values.length === 0) return { min: 0, mean: 0, p50: 0, p95: 0, p99: 0, max: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const pick = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
  return {
    min: sorted[0],
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    p50: pick(0.5),
    p95: pick(0.95),
    p99: pick(0.99),
    max: sorted[sorted.length - 1],
  };
}

function recallAt(predicted, truth, k) {
  const target = new Set(truth.slice(0, k));
  let hits = 0;
  for (const id of predicted.slice(0, k)) if (target.has(id)) hits++;
  return hits / k;
}

function roundLatencyMs(round, fixedMs, bandwidthMiBps, parallelism) {
  if (round.requests <= 0) return 0;
  const bandwidthBytesPerMs = bandwidthMiBps * 1048576 / 1000;
  const sizes = round.rangeBytes && round.rangeBytes.length
    ? [...round.rangeBytes].sort((a, b) => b - a)
    : Array.from({ length: round.requests }, () => round.bytes / round.requests);
  const lanes = new Array(Math.max(1, parallelism)).fill(0);
  for (const bytes of sizes) {
    let best = 0;
    for (let i = 1; i < lanes.length; i++) {
      if (lanes[i] < lanes[best]) best = i;
    }
    lanes[best] += fixedMs + bytes / bandwidthBytesPerMs;
  }
  return Math.max(...lanes);
}

function modelQueryLatency(rounds, fixedMs, bandwidthMiBps, parallelism) {
  let total = 0;
  for (const round of rounds) total += roundLatencyMs(round, fixedMs, bandwidthMiBps, parallelism);
  return total;
}

async function main() {
  const artifactPath = path.resolve(arg('artifact', 'benchmark_results/layout/pancake-sift1m-u8-metis-split.pikelet-range'));
  const dataDir = path.resolve(arg('data-dir', 'sift'));
  const queryCount = parseIntArg('queries', 1000);
  const k = parseIntArg('k', 10);
  const efSearchValues = parseNumberListArg('ef-search-values', [OPERATIONAL_EF_SEARCH]).map((v) => Math.trunc(v));
  const fixedMsValues = parseNumberListArg('fixed-ms', [0, 1, 10, 30, 80]);
  const bandwidthMiBps = Number(arg('bandwidth-mibps', '100'));
  const parallelism = parseIntArg('parallelism', 32);
  const outPath = arg('summary-out', null);

  const queries = readFvecs(path.join(dataDir, 'sift_query.fvecs'), queryCount);
  const truth = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), queryCount);
  const artifact = await Pikelet.openRangeArtifactFile(artifactPath);
  const started = performance.now();
  const byEfSearch = {};

  try {
    for (const efSearch of efSearchValues) {
      artifact.clearCache();
      artifact.resetStats();
      const recalls = [];
      const requests = [];
      const bytes = [];
      const rounds = [];
      const latenciesByFixedMs = new Map(fixedMsValues.map((fixedMs) => [fixedMs, []]));

      for (let i = 0; i < queries.length; i++) {
        const before = artifact.stats();
        const result = await artifact.search(queries[i], k, { efSearch });
        const after = artifact.stats();
        recalls.push(recallAt(result.results.map((row) => row.id), truth[i], k));
        requests.push(after.rangeRequests - before.rangeRequests);
        bytes.push(after.rangeBytes - before.rangeBytes);
        rounds.push(result.rounds.length);
        for (const fixedMs of fixedMsValues) {
          latenciesByFixedMs.get(fixedMs).push(modelQueryLatency(result.rounds, fixedMs, bandwidthMiBps, parallelism));
        }
        if ((i + 1) % 100 === 0 || i + 1 === queries.length) {
          console.error(`efSearch=${efSearch}: checked ${i + 1}/${queries.length}`);
        }
      }

      byEfSearch[String(efSearch)] = {
        recallAtK: summarize(recalls),
        requests: summarize(requests),
        bytes: summarize(bytes),
        mibMean: summarize(bytes).mean / 1048576,
        rounds: summarize(rounds),
        latencyModel: fixedMsValues.map((fixedMs) => ({
          fixedMs,
          bandwidthMiBps,
          parallelism,
          latencyMs: summarize(latenciesByFixedMs.get(fixedMs)),
        })),
        cumulative: artifact.stats(),
      };
    }
  } finally {
    await artifact.close();
  }

  const output = {
    artifact: artifactPath,
    dataset: 'sift',
    queries: queries.length,
    k,
    efSearchValues,
    latencyModel: { fixedMsValues, bandwidthMiBps, parallelism },
    byEfSearch,
    elapsedSeconds: (performance.now() - started) / 1000,
  };

  const json = `${JSON.stringify(output, null, 2)}\n`;
  if (outPath) fs.writeFileSync(outPath, json);
  console.log(json);
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
