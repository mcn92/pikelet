#!/usr/bin/env node
'use strict';

/*
 * Cluster-page routing simulation for range artifacts.
 *
 * Models a base-layer geometry where the artifact is laid out as one
 * contiguous page per METIS partition and a resident centroid table replaces
 * the resident HNSW router. A query:
 *
 *   1. ranks partitions by query-to-centroid distance (resident, no I/O)
 *   2. fetches the top P pages in a single parallel round
 *   3. brute-forces top-k over the fetched candidates
 *
 * Sequential round depth is 1 by construction; the experiment measures what
 * that costs in recall and bytes versus the HNSW traversal baseline
 * (gap=65536, efSearch=80, expansionBatch=8: 95.58% recall@10, 106.1 mean
 * requests, 0.535 MiB mean, ~12 sequential miss rounds).
 *
 * Usage:
 *   node benchmarks/range_cluster_page_sim.js \
 *     --artifact benchmark_results/layout/pancake-sift1m-u8-metis-split.pikelet-range \
 *     --partition benchmark_results/layout/pancake-sift1m-base.metis.part.5036 \
 *     --data-dir sift --queries 1000 --k 10 --pages 1,2,4,8,16 \
 *     --fixed-ms 1,10,30 --bandwidth-mibps 100 --parallelism 6
 */

const fs = require('fs');
const path = require('path');

function arg(name, fallback = null) {
  const idx = process.argv.indexOf('--' + name);
  if (idx === -1) return fallback;
  const next = process.argv[idx + 1];
  return next === undefined || next.startsWith('--') ? true : next;
}

function listArg(name, fallback) {
  const raw = arg(name, null);
  if (raw === null || raw === true) return fallback;
  return String(raw).split(',').map((v) => Number(v.trim())).filter((v) => Number.isFinite(v) && v > 0);
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

function readMetisPartition(file, count) {
  const lines = fs.readFileSync(file, 'utf8').trim().split(/\r?\n/);
  if (lines.length !== count) throw new Error(`partition has ${lines.length} rows; expected ${count}`);
  const partition = new Int32Array(count);
  let maxPart = -1;
  for (let i = 0; i < count; i++) {
    const p = Number(lines[i]);
    if (!Number.isInteger(p) || p < 0) throw new Error(`bad partition row ${i + 1}`);
    partition[i] = p;
    if (p > maxPart) maxPart = p;
  }
  return { partition, parts: maxPart + 1 };
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)];
}

async function main() {
  const artifactPath = arg('artifact', 'benchmark_results/layout/pancake-sift1m-u8-metis-split.pikelet-range');
  const partitionPath = arg('partition', 'benchmark_results/layout/pancake-sift1m-base.metis.part.5036');
  const dataDir = arg('data-dir', 'sift');
  const nQueries = Number(arg('queries', 1000));
  const K = Number(arg('k', 10));
  const pageCounts = listArg('pages', [1, 2, 4, 8, 16]);
  const fixedMs = listArg('fixed-ms', [1, 10, 30]);
  const bandwidthMibps = Number(arg('bandwidth-mibps', 100));
  const parallelism = Number(arg('parallelism', 6));
  const summaryOut = arg('summary-out', null);

  const Pikelet = require('../pikelet.js');
  const artifact = await Pikelet.openRangeArtifactFile(artifactPath, { loadRouter: false });
  const { dim, count, recordBytes, maxLevel } = artifact;
  const qdataOffset = 4 + 2 + 2 + maxLevel * 2;
  if (artifact.metric !== 0) throw new Error('simulation assumes L2 metric');

  console.log(`artifact: ${count.toLocaleString()} nodes, ${dim}D, recordBytes=${recordBytes}`);

  // Decode qdata + scale/offset for every node by streaming both segments.
  const qdata = new Uint8Array(count * dim);
  const scales = new Float32Array(count);
  const offsets = new Float32Array(count);
  const M0 = artifact.M0;
  const edgeOffset = 4 + 2 + 2 + maxLevel * 2 + dim + 8;
  const baseEdges = new Int32Array(count * M0).fill(-1);
  const isRouter = new Uint8Array(count);
  const routerIds = new Int32Array(artifact.routerCount);
  const byteAddress = new Float64Array(count);
  let routerFill = 0;
  const fd = fs.openSync(path.resolve(artifactPath), 'r');
  const segments = [
    { offset: artifact.routerRecordsOffset, records: artifact.routerCount, router: true },
    { offset: artifact.baseRecordsOffset, records: artifact.baseCount, router: false },
  ];
  const chunkRecords = 8192;
  const chunk = Buffer.alloc(chunkRecords * recordBytes);
  let decoded = 0;
  for (const segment of segments) {
    let remaining = segment.records;
    let fileOff = segment.offset;
    while (remaining > 0) {
      const n = Math.min(chunkRecords, remaining);
      fs.readSync(fd, chunk, 0, n * recordBytes, fileOff);
      for (let r = 0; r < n; r++) {
        const base = r * recordBytes;
        const id = chunk.readUInt32LE(base);
        byteAddress[id] = fileOff + base;
        qdata.set(chunk.subarray(base + qdataOffset, base + qdataOffset + dim), id * dim);
        scales[id] = chunk.readFloatLE(base + qdataOffset + dim);
        offsets[id] = chunk.readFloatLE(base + qdataOffset + dim + 4);
        if (segment.router) {
          isRouter[id] = 1;
          routerIds[routerFill++] = id;
        }
        for (let e = 0; e < M0; e++) {
          const target = chunk.readUInt32LE(base + edgeOffset + e * 4);
          if (target !== 0xFFFFFFFF) baseEdges[id * M0 + e] = target;
        }
      }
      decoded += n;
      remaining -= n;
      fileOff += n * recordBytes;
    }
  }
  fs.closeSync(fd);
  console.log(`decoded ${decoded.toLocaleString()} records`);

  const { partition, parts } = readMetisPartition(path.resolve(partitionPath), count);
  console.log(`partitions: ${parts.toLocaleString()}`);

  // Partition membership lists and byte sizes for the hypothetical page layout.
  const partSizes = new Uint32Array(parts);
  for (let i = 0; i < count; i++) partSizes[partition[i]]++;
  const partStarts = new Uint32Array(parts + 1);
  for (let p = 0; p < parts; p++) partStarts[p + 1] = partStarts[p] + partSizes[p];
  const members = new Uint32Array(count);
  const cursor = Uint32Array.from(partStarts.subarray(0, parts));
  for (let i = 0; i < count; i++) members[cursor[partition[i]]++] = i;

  // Resident centroid table: mean of dequantized vectors per partition.
  const centroids = new Float64Array(parts * dim);
  for (let i = 0; i < count; i++) {
    const p = partition[i];
    const s = scales[i];
    const o = offsets[i];
    for (let d = 0; d < dim; d++) centroids[p * dim + d] += o + s * qdata[i * dim + d];
  }
  for (let p = 0; p < parts; p++) {
    for (let d = 0; d < dim; d++) centroids[p * dim + d] /= Math.max(1, partSizes[p]);
  }
  // Per-partition radius: max member distance from the centroid. Enables the
  // 'bound' selector: rank by max(0, |q - centroid| - radius), a true lower
  // bound on the closest possible member distance.
  const radii = new Float64Array(parts);
  for (let i = 0; i < count; i++) {
    const p = partition[i];
    const s = scales[i];
    const o = offsets[i];
    let acc = 0;
    for (let d = 0; d < dim; d++) {
      const diff = (o + s * qdata[i * dim + d]) - centroids[p * dim + d];
      acc += diff * diff;
    }
    const dist = Math.sqrt(acc);
    if (dist > radii[p]) radii[p] = dist;
  }

  const centroidBytes = parts * dim * 4;
  console.log(`resident centroid table: ${(centroidBytes / 1048576).toFixed(2)} MiB (vs router ${(artifact.routerCount * recordBytes / 1048576).toFixed(1)} MiB)`);

  const queries = readFvecs(path.join(dataDir, 'sift_query.fvecs'), nQueries);
  const gt = readIvecs(path.join(dataDir, 'sift_groundtruth.ivecs'), nQueries);
  if (queries.dim !== dim) throw new Error(`query dim ${queries.dim} != artifact dim ${dim}`);
  const Q = Math.min(queries.count, gt.count, nQueries);
  console.log(`queries: ${Q}`);

  const maxPages = Math.max(...pageCounts);
  const partDist = new Float64Array(parts);
  const partOrder = new Int32Array(parts);
  const results = [];
  const rounds = Number(arg('rounds', 1));
  const pages1 = Number(arg('pages1', 8));
  const pages2 = Number(arg('pages2', 8));
  const expandTop = Number(arg('expand-top', 32));

  // Selection strategies:
  //   centroid — rank pages by query-to-centroid distance (default)
  //   reps     — rank pages by min distance to R resident sample members
  //   oracle   — rank pages by true GT hit count (selection upper bound)
  //   router   — rank pages by their best hit among the top R resident
  //              HNSW router nodes (the router doubles as the page selector)
  const selection = String(arg('selection', 'centroid'));
  const repsPer = Number(arg('reps', 8));
  const routerTop = Number(arg('router-top', 64));
  let repIds = null;
  if (selection === 'reps') {
    repIds = new Int32Array(parts * repsPer).fill(-1);
    for (let p = 0; p < parts; p++) {
      const size = partSizes[p];
      const take = Math.min(repsPer, size);
      for (let r = 0; r < take; r++) {
        const m = partStarts[p] + Math.floor((r * size) / take);
        repIds[p * repsPer + r] = members[m];
      }
    }
    const repBytes = parts * repsPer * (dim + 8);
    console.log(`selection=reps: ${repsPer} samples/page, resident ${(repBytes / 1048576).toFixed(2)} MiB (u8 rows + scale/offset)`);
  } else {
    console.log(`selection=${selection}`);
  }

  // Per-query: rank partitions once (to maxPages depth via full sort), then
  // evaluate every P in the sweep incrementally over the same ordering.
  const perPage = pageCounts.map(() => ({ recalls: [], bytes: [], candidates: [] }));
  const gtPartSpread = [];

  if (arg('mode', null) === 'hybrid') {
    // Two-round hybrid: round 1 fetches the sketch scan's top-C1 records
    // (range-artifact records, which carry edges); round 2 fetches the
    // unvisited graph neighbors of the best exact-scored candidates. Depth
    // is 2 by construction; the graph supplies the recall the sketch misses.
    const sketchDims = Number(arg('sketch-dims', 64));
    const sketchBits = Number(arg('sketch-bits', 4));
    const C1 = Number(arg('round1', 300));
    const expandFrom = Number(arg('expand-from', 32));
    const gapBytes = Number(arg('gap', 2048));
    if (dim % sketchDims !== 0) throw new Error('sketch-dims must divide dim');
    const pool = dim / sketchDims;
    const sketches = new Uint8Array(count * sketchDims);
    for (let i = 0; i < count; i++) {
      for (let sd = 0; sd < sketchDims; sd++) {
        let acc = 0;
        for (let j = 0; j < pool; j++) acc += qdata[i * dim + sd * pool + j];
        let value = Math.round(acc / pool);
        if (sketchBits === 4) value = Math.min(15, Math.round(value / 17)) * 17;
        sketches[i * sketchDims + sd] = value;
      }
    }
    console.log(`hybrid: sketch ${sketchDims}D u${sketchBits}, round1 C1=${C1}, expand from top ${expandFrom}, gap ${gapBytes}`);

    const coalesce = (ids) => {
      const addrs = ids.map((id) => byteAddress[id]).sort((a, b) => a - b);
      if (!addrs.length) return { requests: 0, bytes: 0 };
      let requests = 1;
      let bytes = recordBytes;
      let runEnd = addrs[0] + recordBytes;
      for (let a = 1; a < addrs.length; a++) {
        if (addrs[a] <= runEnd + gapBytes) {
          bytes += (addrs[a] + recordBytes) - runEnd;
          runEnd = addrs[a] + recordBytes;
        } else {
          requests++;
          bytes += recordBytes;
          runEnd = addrs[a] + recordBytes;
        }
      }
      return { requests, bytes };
    };

    const exactDist = (qv, id) => {
      const s = scales[id];
      const o = offsets[id];
      let acc = 0;
      const rowBase = id * dim;
      for (let d = 0; d < dim; d++) {
        const diff = qv[d] - (o + s * qdata[rowBase + d]);
        acc += diff * diff;
      }
      return acc;
    };

    const recalls1 = [];
    const recalls2 = [];
    const req1s = [];
    const req2s = [];
    const bytes1s = [];
    const bytes2s = [];
    const expandCounts = [];
    let recoveredTotal = 0;
    let missedTotal = 0;

    const candDist = new Float64Array(C1);
    const candId = new Int32Array(C1);
    const qSketch = new Float64Array(sketchDims);

    for (let q = 0; q < Q; q++) {
      const qv = queries.vectors.subarray(q * dim, (q + 1) * dim);
      for (let sd = 0; sd < sketchDims; sd++) {
        let acc = 0;
        for (let j = 0; j < pool; j++) acc += qv[sd * pool + j];
        qSketch[sd] = acc / pool;
      }
      candDist.fill(Infinity);
      candId.fill(-1);
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
          for (let j = 1; j < C1; j++) if (candDist[j] > candDist[worst]) worst = j;
          candDist[worst] = acc;
          candId[worst] = i;
          candMax = 0;
          for (let j = 0; j < C1; j++) if (candDist[j] > candMax) candMax = candDist[j];
        }
      }
      const round1Ids = [];
      for (let j = 0; j < C1; j++) if (candId[j] >= 0) round1Ids.push(candId[j]);
      const acc1 = coalesce(round1Ids);
      req1s.push(acc1.requests);
      bytes1s.push(acc1.bytes);

      const fetched = new Set(round1Ids);
      const scored = round1Ids.map((id) => [exactDist(qv, id), id]).sort((a, b) => a[0] - b[0]);

      // Round-1 recall for comparison.
      let hits1 = 0;
      const top1 = new Set(scored.slice(0, K).map((e) => e[1]));
      for (let j = 0; j < K; j++) if (top1.has(gt.rows[q * gt.k + j])) hits1++;
      recalls1.push(hits1 / K);

      // Expand one hop from the best exact candidates.
      const expandIds = new Set();
      for (let e = 0; e < Math.min(expandFrom, scored.length); e++) {
        const id = scored[e][1];
        for (let m = 0; m < M0; m++) {
          const target = baseEdges[id * M0 + m];
          if (target >= 0 && !fetched.has(target) && !expandIds.has(target)) expandIds.add(target);
        }
      }
      const round2Ids = [...expandIds];
      expandCounts.push(round2Ids.length);
      const acc2 = coalesce(round2Ids);
      req2s.push(acc2.requests);
      bytes2s.push(acc2.bytes);

      for (const id of round2Ids) scored.push([exactDist(qv, id), id]);
      scored.sort((a, b) => a[0] - b[0]);
      const top2 = new Set(scored.slice(0, K).map((e) => e[1]));
      let hits2 = 0;
      let recovered = 0;
      for (let j = 0; j < K; j++) {
        const trueId = gt.rows[q * gt.k + j];
        if (top2.has(trueId)) {
          hits2++;
          if (!top1.has(trueId) && expandIds.has(trueId)) recovered++;
        } else {
          missedTotal++;
        }
      }
      recoveredTotal += recovered;
      recalls2.push(hits2 / K);
    }

    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const p95v = (xs) => percentile([...xs].sort((a, b) => a - b), 95);
    console.log(`\nround1 recall@${K}: ${(mean(recalls1) * 100).toFixed(2)}%  ->  hybrid recall@${K}: ${(mean(recalls2) * 100).toFixed(2)}%`);
    console.log(`recovered by expansion: ${recoveredTotal} GT hits across ${Q} queries; still missed: ${missedTotal}`);
    console.log(`round1: ${mean(req1s).toFixed(1)} req / ${(mean(bytes1s) / 1024).toFixed(1)} KiB   round2: ${mean(req2s).toFixed(1)} req / ${(mean(bytes2s) / 1024).toFixed(1)} KiB (expanding ${mean(expandCounts).toFixed(0)} new rows)`);
    console.log(`total: ${(mean(req1s) + mean(req2s)).toFixed(1)} req / ${((mean(bytes1s) + mean(bytes2s)) / 1024).toFixed(1)} KiB / 2 rounds`);
    for (const ms of fixedMs) {
      const waves = Math.ceil(p95v(req1s) / parallelism) + Math.ceil(p95v(req2s) / parallelism);
      const transferMs = ((p95v(bytes1s) + p95v(bytes2s)) / (bandwidthMibps * 1048576)) * 1000;
      console.log(`  modeled p95 @ ${ms}ms, p=${parallelism}: ${(waves * ms + transferMs).toFixed(1)} ms`);
    }
    if (summaryOut) {
      fs.writeFileSync(summaryOut, JSON.stringify({
        mode: 'hybrid', sketchDims, sketchBits, C1, expandFrom, gapBytes, queries: Q, k: K,
        round1Recall: mean(recalls1), hybridRecall: mean(recalls2),
        round1Requests: mean(req1s), round2Requests: mean(req2s),
        round1Bytes: mean(bytes1s), round2Bytes: mean(bytes2s),
        expandedRows: mean(expandCounts),
      }, null, 2));
    }
    await artifact.close?.();
    return;
  }

  if (arg('mode', null) === 'sketch') {
    // Resident-sketch geometry: a pooled u8 sketch of every vector stays
    // resident; the query scans sketches locally and fetches only the top-C
    // full records for exact rerank, in one parallel round. Pooling preserves
    // the per-row affine params: mean(o + s*q_d) = o + s*mean(q_d).
    const sketchDims = Number(arg('sketch-dims', 32));
    const sketchBits = Number(arg('sketch-bits', 8));
    const rerankList = listArg('rerank', [50, 100, 200, 400]);
    if (dim % sketchDims !== 0) throw new Error('sketch-dims must divide dim');
    if (![4, 8].includes(sketchBits)) throw new Error('sketch-bits must be 4 or 8');
    const pool = dim / sketchDims;
    const sketches = new Uint8Array(count * sketchDims);
    for (let i = 0; i < count; i++) {
      for (let sd = 0; sd < sketchDims; sd++) {
        let acc = 0;
        for (let j = 0; j < pool; j++) acc += qdata[i * dim + sd * pool + j];
        let value = Math.round(acc / pool);
        if (sketchBits === 4) {
          // 16 levels; reconstruct at the level midpoint (value stays u8 in
          // the sim, but only 16 distinct values — resident cost is 4 bits).
          value = Math.min(15, Math.round(value / 17)) * 17;
        }
        sketches[i * sketchDims + sd] = value;
      }
    }
    const residentMiB = (count * sketchDims * (sketchBits / 8) + count * 8) / 1048576;
    console.log(`sketch mode: ${sketchDims}D pooled u${sketchBits}, resident ${residentMiB.toFixed(1)} MiB (sketches + per-row scale/offset)`);

    const maxC = Math.max(...rerankList);
    const perC = rerankList.map(() => ({ recalls: [] }));
    const candDist = new Float64Array(maxC);
    const candId = new Int32Array(maxC);
    const qSketch = new Float64Array(sketchDims);

    for (let q = 0; q < Q; q++) {
      const qv = queries.vectors.subarray(q * dim, (q + 1) * dim);
      for (let sd = 0; sd < sketchDims; sd++) {
        let acc = 0;
        for (let j = 0; j < pool; j++) acc += qv[sd * pool + j];
        qSketch[sd] = acc / pool;
      }
      candDist.fill(Infinity);
      candId.fill(-1);
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
          for (let j = 1; j < maxC; j++) if (candDist[j] > candDist[worst]) worst = j;
          candDist[worst] = acc;
          candId[worst] = i;
          candMax = 0;
          for (let j = 0; j < maxC; j++) if (candDist[j] > candMax) candMax = candDist[j];
        }
      }
      const bySketch = Array.from({ length: maxC }, (_, j) => j)
        .filter((j) => candId[j] >= 0)
        .sort((a, b) => candDist[a] - candDist[b])
        .map((j) => candId[j]);

      for (let ci = 0; ci < rerankList.length; ci++) {
        const C = rerankList[ci];
        const fetchedIds = bySketch.slice(0, C);
        // Coalesce the fetched records' byte ranges with the standard gap to
        // get the real request geometry under the artifact's RCM layout.
        const addrs = fetchedIds.map((id) => byteAddress[id]).sort((a, b) => a - b);
        const gapBytes = Number(arg('gap', 65536));
        let requests = 1;
        let bytes = recordBytes;
        let runEnd = addrs[0] + recordBytes;
        for (let a = 1; a < addrs.length; a++) {
          if (addrs[a] <= runEnd + gapBytes) {
            // Coalescing fetches the filler between merged records too.
            bytes += (addrs[a] + recordBytes) - runEnd;
            runEnd = addrs[a] + recordBytes;
          } else {
            requests++;
            bytes += recordBytes;
            runEnd = addrs[a] + recordBytes;
          }
        }
        perC[ci].requests = perC[ci].requests || [];
        perC[ci].coalescedBytes = perC[ci].coalescedBytes || [];
        perC[ci].requests.push(requests);
        perC[ci].coalescedBytes.push(bytes);
        // Exact rerank over fetched full records.
        const exact = fetchedIds.map((id) => {
          const s = scales[id];
          const o = offsets[id];
          let acc = 0;
          const rowBase = id * dim;
          for (let d = 0; d < dim; d++) {
            const diff = qv[d] - (o + s * qdata[rowBase + d]);
            acc += diff * diff;
          }
          return [acc, id];
        }).sort((a, b) => a[0] - b[0]).slice(0, K).map((e) => e[1]);
        let hits = 0;
        for (let j = 0; j < K; j++) if (exact.includes(gt.rows[q * gt.k + j])) hits++;
        perC[ci].recalls.push(hits / K);
      }
    }

    console.log(`\nC    recall@${K}  coalesced req (mean/p95)  bytes (mean/p95)  rounds  ${fixedMs.map((ms) => `p95ms@${ms}ms(p=${parallelism})`).join('  ')}`);
    const sketchResults = [];
    for (let ci = 0; ci < rerankList.length; ci++) {
      const C = rerankList[ci];
      const rec = perC[ci].recalls;
      const meanRecall = rec.reduce((a, b) => a + b, 0) / rec.length;
      const reqs = perC[ci].requests;
      const cBytes = perC[ci].coalescedBytes;
      const meanReq = reqs.reduce((a, b) => a + b, 0) / reqs.length;
      const reqSorted = [...reqs].sort((a, b) => a - b);
      const p95Req = percentile(reqSorted, 95);
      const meanBytes = cBytes.reduce((a, b) => a + b, 0) / cBytes.length;
      const bytesSorted = [...cBytes].sort((a, b) => a - b);
      const p95Bytes = percentile(bytesSorted, 95);
      const latencies = fixedMs.map((ms) => {
        const waves = Math.ceil(p95Req / parallelism);
        const transferMs = (p95Bytes / (bandwidthMibps * 1048576)) * 1000;
        return (waves * ms + transferMs).toFixed(1);
      });
      console.log(`${String(C).padEnd(4)} ${(meanRecall * 100).toFixed(2)}%     ${meanReq.toFixed(1)} / ${p95Req}              ${(meanBytes / 1024).toFixed(0)} / ${(p95Bytes / 1024).toFixed(0)} KiB       1  ${latencies.join('        ')}`);
      sketchResults.push({ rerank: C, meanRecall, meanRequests: meanReq, p95Requests: p95Req, meanBytes, p95Bytes });
    }
    if (summaryOut) {
      fs.writeFileSync(summaryOut, JSON.stringify({ mode: 'sketch', sketchDims, residentMiB, queries: Q, k: K, sweep: sketchResults }, null, 2));
    }
    await artifact.close?.();
    return;
  }

  if (rounds === 2) {
    // Two-round flow: centroid-ranked pages for round 1, then pages holding
    // the graph neighbors of the best round-1 candidates for round 2. Edges
    // of fetched nodes are known because their pages were fetched.
    const recalls = [];
    const bytesList = [];
    const candList = [];
    const candDist = new Float64Array(expandTop);
    const candId = new Int32Array(expandTop);
    const fetched = new Uint8Array(parts);

    const scoreMembers = (p, qv, state) => {
      for (let m = partStarts[p]; m < partStarts[p + 1]; m++) {
        const id = members[m];
        const s = scales[id];
        const o = offsets[id];
        let acc = 0;
        const rowBase = id * dim;
        for (let d = 0; d < dim; d++) {
          const diff = qv[d] - (o + s * qdata[rowBase + d]);
          acc += diff * diff;
        }
        if (acc < state.max) {
          let worst = 0;
          for (let j = 1; j < expandTop; j++) if (candDist[j] > candDist[worst]) worst = j;
          candDist[worst] = acc;
          candId[worst] = id;
          state.max = 0;
          for (let j = 0; j < expandTop; j++) if (candDist[j] > state.max) state.max = candDist[j];
        }
      }
    };

    for (let q = 0; q < Q; q++) {
      const qv = queries.vectors.subarray(q * dim, (q + 1) * dim);
      candDist.fill(Infinity);
      candId.fill(-1);
      fetched.fill(0);
      const state = { max: Infinity };

      for (let p = 0; p < parts; p++) {
        let acc = 0;
        for (let d = 0; d < dim; d++) {
          const diff = qv[d] - centroids[p * dim + d];
          acc += diff * diff;
        }
        partDist[p] = acc;
        partOrder[p] = p;
      }
      partOrder.sort((a, b) => partDist[a] - partDist[b]);

      let fetchedBytes = 0;
      let fetchedCand = 0;
      for (let r = 0; r < pages1; r++) {
        const p = partOrder[r];
        fetched[p] = 1;
        fetchedBytes += partSizes[p] * recordBytes;
        fetchedCand += partSizes[p];
        scoreMembers(p, qv, state);
      }

      // Round 2: rank unfetched pages by hits from candidate edges.
      const order = Array.from({ length: expandTop }, (_, j) => j)
        .filter((j) => candId[j] >= 0)
        .sort((a, b) => candDist[a] - candDist[b]);
      const pageHits = new Map();
      for (const j of order) {
        const id = candId[j];
        for (let e = 0; e < M0; e++) {
          const target = baseEdges[id * M0 + e];
          if (target < 0) continue;
          const p = partition[target];
          if (fetched[p]) continue;
          const entry = pageHits.get(p);
          if (entry) entry.count++;
          else pageHits.set(p, { count: 1, best: candDist[j] });
        }
      }
      const round2Pages = [...pageHits.entries()]
        .sort((a, b) => b[1].count - a[1].count || a[1].best - b[1].best)
        .slice(0, pages2)
        .map(([p]) => p);
      for (const p of round2Pages) {
        fetched[p] = 1;
        fetchedBytes += partSizes[p] * recordBytes;
        fetchedCand += partSizes[p];
        scoreMembers(p, qv, state);
      }

      const final = Array.from({ length: expandTop }, (_, j) => j)
        .filter((j) => candId[j] >= 0)
        .sort((a, b) => candDist[a] - candDist[b])
        .slice(0, K)
        .map((j) => candId[j]);
      let hits = 0;
      for (let j = 0; j < K; j++) {
        if (final.includes(gt.rows[q * gt.k + j])) hits++;
      }
      recalls.push(hits / K);
      bytesList.push(fetchedBytes);
      candList.push(fetchedCand);
    }

    const meanRecall = recalls.reduce((a, b) => a + b, 0) / recalls.length;
    const meanBytes = bytesList.reduce((a, b) => a + b, 0) / bytesList.length;
    const bytesSorted = [...bytesList].sort((a, b) => a - b);
    const p95Bytes = percentile(bytesSorted, 95);
    const meanCand = candList.reduce((a, b) => a + b, 0) / candList.length;
    console.log(`\nTwo-round (P1=${pages1} centroid, P2=${pages2} edge-guided, expandTop=${expandTop})`);
    console.log(`recall@${K}: ${(meanRecall * 100).toFixed(2)}%  bytes mean ${(meanBytes / 1048576).toFixed(3)} MiB / p95 ${(p95Bytes / 1048576).toFixed(3)} MiB  candidates ${Math.round(meanCand)}  requests ${pages1 + pages2}  rounds 2`);
    for (const ms of fixedMs) {
      const waves = Math.ceil(pages1 / parallelism) + Math.ceil(pages2 / parallelism);
      const transferMs = (p95Bytes / (bandwidthMibps * 1048576)) * 1000;
      console.log(`  modeled p95 @ ${ms}ms fixed read cost, p=${parallelism}: ${(waves * ms + transferMs).toFixed(2)} ms`);
    }
    if (summaryOut) {
      fs.writeFileSync(summaryOut, JSON.stringify({
        mode: 'two-round', pages1, pages2, expandTop, parts, queries: Q, k: K,
        meanRecall, meanBytes, p95Bytes, meanCandidates: meanCand,
      }, null, 2));
    }
    await artifact.close?.();
    return;
  }

  for (let q = 0; q < Q; q++) {
    const qv = queries.vectors.subarray(q * dim, (q + 1) * dim);

    if (selection === 'oracle') {
      const hitsPerPart = new Map();
      for (let j = 0; j < K; j++) {
        const p = partition[gt.rows[q * gt.k + j]];
        hitsPerPart.set(p, (hitsPerPart.get(p) || 0) + 1);
      }
      for (let p = 0; p < parts; p++) {
        partDist[p] = -(hitsPerPart.get(p) || 0);
        partOrder[p] = p;
      }
    } else if (selection === 'router') {
      // Brute-force the resident router segment (it is resident in the real
      // design, so this costs no I/O), then rank partitions by their best
      // router hit; unrepresented partitions rank by centroid distance after.
      const topDist = new Float64Array(routerTop).fill(Infinity);
      const topId = new Int32Array(routerTop).fill(-1);
      let topMax = Infinity;
      for (let ri = 0; ri < routerIds.length; ri++) {
        const id = routerIds[ri];
        const s = scales[id];
        const o = offsets[id];
        let acc = 0;
        const rowBase = id * dim;
        for (let d = 0; d < dim; d++) {
          const diff = qv[d] - (o + s * qdata[rowBase + d]);
          acc += diff * diff;
        }
        if (acc < topMax) {
          let worst = 0;
          for (let j = 1; j < routerTop; j++) if (topDist[j] > topDist[worst]) worst = j;
          topDist[worst] = acc;
          topId[worst] = id;
          topMax = 0;
          for (let j = 0; j < routerTop; j++) if (topDist[j] > topMax) topMax = topDist[j];
        }
      }
      for (let p = 0; p < parts; p++) {
        let acc = 0;
        for (let d = 0; d < dim; d++) {
          const diff = qv[d] - centroids[p * dim + d];
          acc += diff * diff;
        }
        partDist[p] = 1e18 + acc; // fallback tier: centroid-ranked, after all hit pages
        partOrder[p] = p;
      }
      for (let j = 0; j < routerTop; j++) {
        if (topId[j] < 0) continue;
        const p = partition[topId[j]];
        if (topDist[j] < partDist[p]) partDist[p] = topDist[j];
      }
    } else if (selection === 'reps') {
      for (let p = 0; p < parts; p++) {
        let best = Infinity;
        for (let r = 0; r < repsPer; r++) {
          const id = repIds[p * repsPer + r];
          if (id < 0) break;
          const s = scales[id];
          const o = offsets[id];
          let acc = 0;
          const rowBase = id * dim;
          for (let d = 0; d < dim; d++) {
            const diff = qv[d] - (o + s * qdata[rowBase + d]);
            acc += diff * diff;
          }
          if (acc < best) best = acc;
        }
        partDist[p] = best;
        partOrder[p] = p;
      }
    } else {
      for (let p = 0; p < parts; p++) {
        let acc = 0;
        for (let d = 0; d < dim; d++) {
          const diff = qv[d] - centroids[p * dim + d];
          acc += diff * diff;
        }
        if (selection === 'bound') {
          const lower = Math.max(0, Math.sqrt(acc) - radii[p]);
          partDist[p] = lower * lower;
        } else {
          partDist[p] = acc;
        }
        partOrder[p] = p;
      }
    }
    partOrder.sort((a, b) => partDist[a] - partDist[b]);

    // How many distinct partitions hold this query's true top-K.
    const spread = new Set();
    for (let j = 0; j < K; j++) spread.add(partition[gt.rows[q * gt.k + j]]);
    gtPartSpread.push(spread.size);

    // Incremental brute force as pages are added in ranked order.
    const heapDist = new Float64Array(K).fill(Infinity);
    const heapId = new Int32Array(K).fill(-1);
    let heapMax = Infinity;
    let fetchedBytes = 0;
    let fetchedCandidates = 0;
    let sweepIdx = 0;

    for (let rank = 0; rank < maxPages && sweepIdx < pageCounts.length; rank++) {
      const p = partOrder[rank];
      fetchedBytes += partSizes[p] * recordBytes;
      fetchedCandidates += partSizes[p];
      for (let m = partStarts[p]; m < partStarts[p + 1]; m++) {
        const id = members[m];
        const s = scales[id];
        const o = offsets[id];
        let acc = 0;
        const rowBase = id * dim;
        for (let d = 0; d < dim; d++) {
          const diff = qv[d] - (o + s * qdata[rowBase + d]);
          acc += diff * diff;
        }
        if (acc < heapMax) {
          // Replace current worst (simple K-array; K is small).
          let worst = 0;
          for (let j = 1; j < K; j++) if (heapDist[j] > heapDist[worst]) worst = j;
          heapDist[worst] = acc;
          heapId[worst] = id;
          heapMax = 0;
          for (let j = 0; j < K; j++) if (heapDist[j] > heapMax) heapMax = heapDist[j];
        }
      }
      while (sweepIdx < pageCounts.length && rank + 1 === pageCounts[sweepIdx]) {
        let hits = 0;
        for (let j = 0; j < K; j++) {
          const trueId = gt.rows[q * gt.k + j];
          for (let h = 0; h < K; h++) if (heapId[h] === trueId) { hits++; break; }
        }
        perPage[sweepIdx].recalls.push(hits / K);
        perPage[sweepIdx].bytes.push(fetchedBytes);
        perPage[sweepIdx].candidates.push(fetchedCandidates);
        sweepIdx++;
      }
    }
  }

  const spreadSorted = [...gtPartSpread].sort((a, b) => a - b);
  console.log(`\nGT top-${K} partition spread: mean ${(gtPartSpread.reduce((a, b) => a + b, 0) / Q).toFixed(2)}, p50 ${percentile(spreadSorted, 50)}, p95 ${percentile(spreadSorted, 95)}, max ${spreadSorted[spreadSorted.length - 1]}`);

  console.log(`\nP  recall@${K}  bytes/query(mean)  MiB p95  candidates  requests  rounds  ${fixedMs.map((ms) => `p95ms@${ms}ms`).join('  ')}`);
  for (let i = 0; i < pageCounts.length; i++) {
    const P = pageCounts[i];
    const rec = perPage[i].recalls;
    const bytes = perPage[i].bytes;
    const cand = perPage[i].candidates;
    const meanRecall = rec.reduce((a, b) => a + b, 0) / rec.length;
    const meanBytes = bytes.reduce((a, b) => a + b, 0) / bytes.length;
    const bytesSorted = [...bytes].sort((a, b) => a - b);
    const p95Bytes = percentile(bytesSorted, 95);
    const meanCand = cand.reduce((a, b) => a + b, 0) / cand.length;
    // One parallel round: ceil(P/parallelism) sequential waves of fixed cost,
    // plus transfer time for the fetched bytes at the modeled bandwidth.
    const latencies = fixedMs.map((ms) => {
      const waves = Math.ceil(P / parallelism);
      const transferMs = (p95Bytes / (bandwidthMibps * 1048576)) * 1000;
      return (waves * ms + transferMs).toFixed(2);
    });
    console.log(`${String(P).padEnd(2)} ${(meanRecall * 100).toFixed(2)}%     ${(meanBytes / 1048576).toFixed(3)} MiB          ${(p95Bytes / 1048576).toFixed(3)}  ${String(Math.round(meanCand)).padStart(9)}  ${String(P).padStart(8)}       1  ${latencies.join('        ')}`);
    results.push({ pages: P, meanRecall, meanBytes, p95Bytes, meanCandidates: meanCand });
  }

  if (summaryOut) {
    fs.writeFileSync(summaryOut, JSON.stringify({
      artifact: artifactPath,
      partitionFile: partitionPath,
      parts,
      queries: Q,
      k: K,
      centroidBytes,
      gtPartitionSpread: { mean: gtPartSpread.reduce((a, b) => a + b, 0) / Q, p50: percentile(spreadSorted, 50), p95: percentile(spreadSorted, 95) },
      model: { fixedMs, bandwidthMibps, parallelism },
      sweep: results,
    }, null, 2));
    console.log(`\nsummary written to ${summaryOut}`);
  }

  await artifact.close?.();
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
