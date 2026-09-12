#!/usr/bin/env node
'use strict';

/*
 * End-to-end test of the resident-sketch fetch-to-rerank geometry using the
 * real RangeArtifact reader (real range reads, real coalescing), instead of
 * the offline simulation in range_cluster_page_sim.js.
 *
 * Phase 1 (--mode file): recall parity against the simulation over a local
 * file source, 1000 queries.
 * Phase 2 (--mode http --delay-ms 10): measured wall-clock comparison of
 * sketch-rerank versus baseline HNSW traversal through an in-process HTTP
 * server that injects fixed per-request latency.
 *
 * The sketch sidecar (pooled u8 rows + per-row scale/offset) is built once
 * from the artifact and cached next to it.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const Pikelet = require('../pikelet.js');

function arg(name, fallback = null) {
  const idx = process.argv.indexOf('--' + name);
  if (idx === -1) return fallback;
  const next = process.argv[idx + 1];
  return next === undefined || next.startsWith('--') ? true : next;
}

function readFvecs(file, limit) {
  const buf = fs.readFileSync(file);
  const dim = buf.readInt32LE(0);
  const stride = 4 + dim * 4;
  const count = Math.min(Math.floor(buf.length / stride), limit || Infinity);
  const vectors = new Float32Array(count * dim);
  for (let i = 0; i < count; i++) {
    const off = i * stride + 4;
    for (let d = 0; d < dim; d++) vectors[i * dim + d] = buf.readFloatLE(off + d * 4);
  }
  return { vectors, count, dim };
}

function readIvecs(file, limit) {
  const buf = fs.readFileSync(file);
  const k = buf.readInt32LE(0);
  const stride = 4 + k * 4;
  const count = Math.min(Math.floor(buf.length / stride), limit || Infinity);
  const rows = new Int32Array(count * k);
  for (let i = 0; i < count; i++) {
    const off = i * stride + 4;
    for (let j = 0; j < k; j++) rows[i * k + j] = buf.readInt32LE(off + j * 4);
  }
  return { rows, count, k };
}

const SKETCH_MAGIC = 0x314B5350; // 'PSK1'

async function buildSketchSidecar(artifactPath, sidecarPath, sketchDims) {
  const probe = await Pikelet.openRangeArtifactFile(artifactPath, { loadRouter: false });
  const { count, dim, maxLevel, recordBytes, routerCount, baseCount } = probe;
  const routerOffset = probe.routerRecordsOffset;
  const baseOffset = probe.baseRecordsOffset;
  await probe.close?.();
  const fd = fs.openSync(artifactPath, 'r');
  if (dim % sketchDims !== 0) throw new Error('sketch dims must divide artifact dim');
  const pool = dim / sketchDims;
  const qdataOffset = 4 + 2 + 2 + maxLevel * 2;

  const out = Buffer.alloc(32 + count * 8 + count * sketchDims);
  out.writeUInt32LE(SKETCH_MAGIC, 0);
  out.writeUInt32LE(1, 4);
  out.writeUInt32LE(count, 8);
  out.writeUInt32LE(sketchDims, 12);
  out.writeUInt32LE(pool, 16);
  const scaleBase = 32;
  const offsetBase = 32 + count * 4;
  const sketchBase = 32 + count * 8;

  const chunkRecords = 8192;
  const chunk = Buffer.alloc(chunkRecords * recordBytes);
  for (const segment of [
    { offset: routerOffset, records: routerCount },
    { offset: baseOffset, records: baseCount },
  ]) {
    let remaining = segment.records;
    let fileOff = segment.offset;
    while (remaining > 0) {
      const n = Math.min(chunkRecords, remaining);
      fs.readSync(fd, chunk, 0, n * recordBytes, fileOff);
      for (let r = 0; r < n; r++) {
        const base = r * recordBytes;
        const id = chunk.readUInt32LE(base);
        out.writeFloatLE(chunk.readFloatLE(base + qdataOffset + dim), scaleBase + id * 4);
        out.writeFloatLE(chunk.readFloatLE(base + qdataOffset + dim + 4), offsetBase + id * 4);
        for (let sd = 0; sd < sketchDims; sd++) {
          let acc = 0;
          for (let j = 0; j < pool; j++) acc += chunk[base + qdataOffset + sd * pool + j];
          out[sketchBase + id * sketchDims + sd] = Math.round(acc / pool);
        }
      }
      remaining -= n;
      fileOff += n * recordBytes;
    }
  }
  fs.closeSync(fd);
  fs.writeFileSync(sidecarPath, out);
  return out;
}

function loadSketchSidecar(sidecarPath) {
  const buf = fs.readFileSync(sidecarPath);
  if (buf.readUInt32LE(0) !== SKETCH_MAGIC) throw new Error('bad sketch sidecar magic');
  const count = buf.readUInt32LE(8);
  const sketchDims = buf.readUInt32LE(12);
  const pool = buf.readUInt32LE(16);
  const scales = new Float32Array(buf.buffer, buf.byteOffset + 32, count);
  const offsets = new Float32Array(buf.buffer, buf.byteOffset + 32 + count * 4, count);
  const sketches = new Uint8Array(buf.buffer, buf.byteOffset + 32 + count * 8, count * sketchDims);
  return { count, sketchDims, pool, scales, offsets, sketches, bytes: buf.length };
}

// WASM-backed scanner: sketches/scales/offsets live in the engine heap; each
// query calls the SIMD pancake_sketch_scan kernel.
async function createWasmScanner(sidecar, maxC) {
  const factory = require('../dist/engine.js');
  const Module = await factory({
    wasmBinary: fs.readFileSync(path.join(__dirname, '..', 'dist', 'engine.wasm')),
  });
  const { count, sketchDims } = sidecar;
  const sketchesPtr = Module._emsc_malloc(count * sketchDims);
  const scalesPtr = Module._emsc_malloc(count * 4);
  const offsetsPtr = Module._emsc_malloc(count * 4);
  const queryPtr = Module._emsc_malloc(sketchDims * 4);
  const outIdsPtr = Module._emsc_malloc(maxC * 4);
  const outDistsPtr = Module._emsc_malloc(maxC * 4);
  // Re-acquire heap views after allocation: growth invalidates them.
  Module.HEAPU8.set(sidecar.sketches, sketchesPtr);
  Module.HEAPF32.set(sidecar.scales, scalesPtr >> 2);
  Module.HEAPF32.set(sidecar.offsets, offsetsPtr >> 2);
  return {
    scan(qSketch, C) {
      Module.HEAPF32.set(qSketch, queryPtr >> 2);
      const n = Module._pikelet_sketch_scan(
        sketchesPtr, scalesPtr, offsetsPtr, count, sketchDims, queryPtr, 0 /* l2 */, C, outIdsPtr, outDistsPtr
      );
      return Array.from(Module.HEAPU32.subarray(outIdsPtr >> 2, (outIdsPtr >> 2) + n));
    },
  };
}

// Sketch search through the real artifact reader: resident scan, then one
// coalesced parallel prefetch of the top-C records, then exact rerank from
// the reader's decoded cache.
async function sketchSearch(artifact, sidecar, query, k, opts) {
  const { sketchDims, pool, scales, offsets, sketches, count } = sidecar;
  const C = opts.rerank;
  const qSketch = new Float32Array(sketchDims);
  for (let sd = 0; sd < sketchDims; sd++) {
    let acc = 0;
    for (let j = 0; j < pool; j++) acc += query[sd * pool + j];
    qSketch[sd] = acc / pool;
  }
  const scanStart = performance.now();
  let ids;
  if (opts.scanner) {
    ids = opts.scanner.scan(qSketch, C);
  } else {
    const candDist = new Float64Array(C).fill(Infinity);
    const candId = new Int32Array(C).fill(-1);
    let candMax = Infinity;
    for (let i = 0; i < count; i++) {
      const s = scales[i];
      const o = offsets[i];
      let acc = 0;
      const rowBase = i * sketchDims;
      for (let sd = 0; sd < sketchDims; sd++) {
        const diff = qSketch[sd] - (o + s * sketches[rowBase + sd]);
        acc += diff * diff;
      }
      if (acc < candMax) {
        let worst = 0;
        for (let j = 1; j < C; j++) if (candDist[j] > candDist[worst]) worst = j;
        candDist[worst] = acc;
        candId[worst] = i;
        candMax = 0;
        for (let j = 0; j < C; j++) if (candDist[j] > candMax) candMax = candDist[j];
      }
    }
    ids = [];
    for (let j = 0; j < C; j++) if (candId[j] >= 0) ids.push(candId[j]);
  }
  if (opts.scanTimes) opts.scanTimes.push(performance.now() - scanStart);

  await artifact.prefetch(ids, { gap: opts.gap, parallelism: opts.parallelism });

  const dim = artifact.dim;
  const exact = [];
  for (const id of ids) {
    const node = await artifact.readNode(id);
    const s = node.scale;
    const o = node.offset;
    let acc = 0;
    for (let d = 0; d < dim; d++) {
      const diff = query[d] - (o + s * node.qdata[d]);
      acc += diff * diff;
    }
    exact.push([acc, id]);
  }
  exact.sort((a, b) => a[0] - b[0]);
  return exact.slice(0, k).map((e) => ({ id: e[1], distance: e[0] }));
}

function startDelayedServer(filePath, delayMs) {
  const stat = fs.statSync(filePath);
  const fd = fs.openSync(filePath, 'r');
  const server = http.createServer((req, res) => {
    const range = req.headers.range;
    const respond = () => {
      if (!range) {
        res.writeHead(200, { 'content-length': stat.size });
        fs.createReadStream(filePath).pipe(res);
        return;
      }
      const m = /bytes=(\d+)-(\d+)?/.exec(range);
      const start = Number(m[1]);
      const end = m[2] !== undefined ? Number(m[2]) : stat.size - 1;
      const len = end - start + 1;
      const buf = Buffer.alloc(len);
      fs.readSync(fd, buf, 0, len, start);
      res.writeHead(206, {
        'content-length': len,
        'content-range': `bytes ${start}-${end}/${stat.size}`,
      });
      res.end(buf);
    };
    if (delayMs > 0) setTimeout(respond, delayMs);
    else respond();
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

function createHttpRangeSource(url) {
  return {
    async read(offset, length) {
      const response = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
      if (response.status !== 206 && response.status !== 200) throw new Error(`range read failed: ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength !== length) throw new Error(`short range read: ${bytes.byteLength} != ${length}`);
      return bytes;
    },
  };
}

// Range source over an artifact stored as N fixed-size part objects
// (`<prefix>0000` ... zero-padded), presenting one virtual contiguous byte
// space. Ranges crossing a part boundary split into per-part reads.
function createHttpPartRangeSource(prefixUrl, partSize, partCount) {
  return {
    async read(offset, length) {
      const out = new Uint8Array(length);
      let written = 0;
      const pieces = [];
      while (written < length) {
        const absolute = offset + written;
        const partIndex = Math.floor(absolute / partSize);
        if (partIndex >= partCount) throw new Error(`range beyond artifact parts at offset ${absolute}`);
        const partOffset = absolute - partIndex * partSize;
        const pieceLen = Math.min(length - written, partSize - partOffset);
        const key = `${prefixUrl}${String(partIndex).padStart(4, '0')}`;
        const dest = written;
        const fetchPiece = async (attempt) => {
          try {
            const response = await fetch(key, { headers: { Range: `bytes=${partOffset}-${partOffset + pieceLen - 1}` } });
            if (response.status !== 206 && response.status !== 200) throw new Error(`part range read failed: ${response.status}`);
            const bytes = new Uint8Array(await response.arrayBuffer());
            if (bytes.byteLength !== pieceLen) throw new Error(`short part read: ${bytes.byteLength} != ${pieceLen}`);
            out.set(bytes, dest);
          } catch (err) {
            if (attempt >= 2) throw err;
            await new Promise((res) => setTimeout(res, 250 * (attempt + 1)));
            return fetchPiece(attempt + 1);
          }
        };
        pieces.push(fetchPiece(0));
        written += pieceLen;
      }
      await Promise.all(pieces);
      return out;
    },
  };
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)];
}

function summarize(label, walls, recalls, statsDelta, queries) {
  const sorted = [...walls].sort((a, b) => a - b);
  const meanRecall = recalls.reduce((a, b) => a + b, 0) / recalls.length;
  console.log(`${label}: recall@10 ${(meanRecall * 100).toFixed(2)}%  wall mean ${(walls.reduce((a, b) => a + b, 0) / walls.length).toFixed(1)} ms  p50 ${percentile(sorted, 50).toFixed(1)}  p95 ${percentile(sorted, 95).toFixed(1)}  p99 ${percentile(sorted, 99).toFixed(1)} ms`);
  console.log(`  network: ${(statsDelta.rangeRequests / queries).toFixed(1)} req/query, ${(statsDelta.rangeBytes / queries / 1024).toFixed(1)} KiB/query (cache-warming across queries included)`);
}

async function main() {
  const artifactPath = path.resolve(arg('artifact', 'benchmark_results/layout/pancake-sift1m-u8-metis-split.pikelet-range'));
  const dataDir = arg('data-dir', 'sift');
  const nQueries = Number(arg('queries', 1000));
  const K = Number(arg('k', 10));
  const C = Number(arg('rerank', 300));
  const sketchDims = Number(arg('sketch-dims', 64));
  const gap = Number(arg('gap', 65536));
  const parallelism = Number(arg('parallelism', 6));
  const mode = String(arg('mode', 'file'));
  const delayMs = Number(arg('delay-ms', 10));
  const compare = arg('compare', false) !== false;
  const efSearch = Number(arg('ef-search', 80));
  const expansionBatch = Number(arg('expansion-batch', 8));

  const sidecarPath = `${artifactPath}.sketch${sketchDims}`;
  if (!fs.existsSync(sidecarPath)) {
    console.log(`building sketch sidecar (${sketchDims}D)...`);
    await buildSketchSidecar(artifactPath, sidecarPath, sketchDims);
  }
  const sidecar = loadSketchSidecar(sidecarPath);
  console.log(`sidecar: ${(sidecar.bytes / 1048576).toFixed(1)} MiB resident, ${sidecar.sketchDims}D pooled u8`);

  const queries = readFvecs(path.join(dataDir, 'sift_query.fvecs'), nQueries);
  const gt = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), nQueries);
  const Q = Math.min(queries.count, gt.count, nQueries);
  const dim = queries.dim;

  let sketchSource;
  let traversalSource;
  let server = null;
  if (mode === 'remote') {
    // Real-world mode: artifact served as part objects from a remote host
    // (e.g. an R2 public bucket). --parts-prefix is the URL prefix up to and
    // including "part-".
    const prefixUrl = String(arg('parts-prefix'));
    const partSize = Number(arg('part-size', 67108864));
    const partCount = Number(arg('part-count', 8));
    if (!prefixUrl || prefixUrl === 'null') throw new Error('--mode remote requires --parts-prefix');
    sketchSource = createHttpPartRangeSource(prefixUrl, partSize, partCount);
    traversalSource = createHttpPartRangeSource(prefixUrl, partSize, partCount);
    console.log(`remote mode: ${partCount} parts at ${prefixUrl}*, parallelism ${parallelism}`);
  } else if (mode === 'http') {
    const started = await startDelayedServer(artifactPath, delayMs);
    server = started.server;
    const url = `http://127.0.0.1:${started.port}/artifact`;
    sketchSource = createHttpRangeSource(url);
    traversalSource = createHttpRangeSource(url);
    console.log(`http mode: injected delay ${delayMs} ms/request, parallelism ${parallelism}`);
  } else {
    sketchSource = new Pikelet.NodeFileRangeSource(artifactPath);
    traversalSource = new Pikelet.NodeFileRangeSource(artifactPath);
  }

  // Sketch reader needs no resident router.
  const sketchArtifact = await Pikelet.RangeArtifact.open(sketchSource, { loadRouter: false });
  const recallOf = (results, q) => {
    let hits = 0;
    for (let j = 0; j < K; j++) {
      const trueId = gt.rows[q * gt.k + j];
      if (results.some((r) => r.id === trueId)) hits++;
    }
    return hits / K;
  };

  {
    const scanMode = String(arg('scan', 'js'));
    const scanner = scanMode === 'wasm' ? await createWasmScanner(sidecar, C) : null;
    const walls = [];
    const recalls = [];
    const scanTimes = [];
    const before = sketchArtifact.stats();
    for (let q = 0; q < Q; q++) {
      const qv = queries.vectors.subarray(q * dim, (q + 1) * dim);
      const t0 = performance.now();
      const results = await sketchSearch(sketchArtifact, sidecar, qv, K, { rerank: C, gap, parallelism, scanner, scanTimes });
      walls.push(performance.now() - t0);
      recalls.push(recallOf(results, q));
    }
    const after = sketchArtifact.stats();
    summarize(`sketch C=${C} scan=${scanMode} (${mode}${mode === 'http' ? ` @${delayMs}ms` : ''})`, walls, recalls, {
      rangeRequests: after.rangeRequests - before.rangeRequests,
      rangeBytes: after.rangeBytes - before.rangeBytes,
    }, Q);
    const scanSorted = [...scanTimes].sort((a, b) => a - b);
    console.log(`  scan: mean ${(scanTimes.reduce((a, b) => a + b, 0) / scanTimes.length).toFixed(2)} ms, p95 ${percentile(scanSorted, 95).toFixed(2)} ms`);
  }

  if (compare) {
    const traversalArtifact = await Pikelet.RangeArtifact.open(traversalSource, { loadRouter: true });
    const walls = [];
    const recalls = [];
    const before = traversalArtifact.stats();
    for (let q = 0; q < Q; q++) {
      const qv = queries.vectors.subarray(q * dim, (q + 1) * dim);
      const t0 = performance.now();
      const result = await traversalArtifact.search(qv, K, { efSearch, gap, expansionBatch, rangeParallelism: parallelism });
      walls.push(performance.now() - t0);
      recalls.push(recallOf(result.results, q));
    }
    const after = traversalArtifact.stats();
    summarize(`traversal ef=${efSearch} batch=${expansionBatch} (${mode}${mode === 'http' ? ` @${delayMs}ms` : ''})`, walls, recalls, {
      rangeRequests: after.rangeRequests - before.rangeRequests,
      rangeBytes: after.rangeBytes - before.rangeBytes,
    }, Q);
    await traversalArtifact.close?.();
  }

  await sketchArtifact.close?.();
  if (server) server.close();
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
