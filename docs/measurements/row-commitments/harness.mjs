#!/usr/bin/env node
// POC measurement harness for a .pikelet complete artifact. Opens the file
// (local path or range-capable URL), runs a query set through the one-file
// reader, and records per-query latency, bytes requested, and range-read
// counts — the numbers the POC report needs. Two passes per query set:
// the first pass hits a fresh reader (lazy segment loads included), the
// repeat passes measure the reader warm. CDN edge-cache behavior is a
// separate axis; measure it by pointing this at a URL and flushing the
// edge cache between runs.
//
//   node harness.mjs <artifact.pikelet | http(s)://host/artifact.pikelet>
//       [--queries queries.json]   queries: ["text", ...] or [{ text, ids? }, ...];
//                                  ids (optional) are ground-truth top-k doc ids
//                                  for recall@k. Default: the artifact's own
//                                  evaluation goldens, else manifest sampleQueries.
//       [--k 10]                   results per query
//       [--rerank <n>]             sketch rerank depth passthrough
//       [--warm-passes 1]          repeat passes after the first
//       [--out results.json]       full per-query records for cost.mjs / the report
//
// Bytes are counted as requested lengths at the range-source boundary, the
// same accounting httpRangeSource.stats uses — i.e. what a CDN would bill
// as egress, before transport overhead.

import fs from 'node:fs';
import { openPikeletFile } from '../../../examples/05-one-file-search/pikelet-file-reader.mjs';
import { httpRangeSource } from '../../../examples/05-one-file-search/sources.mjs';

function parseArgs(argv) {
    const args = { positional: [], k: 10, warmPasses: 1 };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--queries') args.queries = argv[++i];
        else if (a === '--k') args.k = Number(argv[++i]);
        else if (a === '--rerank') args.rerank = Number(argv[++i]);
        else if (a === '--warm-passes') args.warmPasses = Number(argv[++i]);
        else if (a === '--out') args.out = argv[++i];
        else if (a.startsWith('--')) throw new Error(`unknown flag ${a}`);
        else args.positional.push(a);
    }
    if (args.positional.length !== 1) {
        throw new Error('usage: node harness.mjs <artifact path or URL> [--queries q.json] [--k 10] [--rerank n] [--warm-passes 1] [--out results.json]');
    }
    return args;
}

function instrumentedFileSource(filePath) {
    const fd = fs.openSync(filePath, 'r');
    const size = fs.fstatSync(fd).size;
    const stats = { requests: 0, bytes: 0 };
    return {
        stats,
        size,
        preferredParallelism: Infinity,
        preferredGapBytes: 2048,
        async read(offset, length) {
            stats.requests += 1;
            stats.bytes += length;
            const buffer = new Uint8Array(length);
            const bytesRead = fs.readSync(fd, buffer, 0, length, offset);
            return buffer.subarray(0, bytesRead);
        },
        async close() { fs.closeSync(fd); },
    };
}

const percentile = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)];

function summarize(records) {
    const ms = records.map((r) => r.ms).sort((a, b) => a - b);
    const bytes = records.map((r) => r.bytes).sort((a, b) => a - b);
    const reads = records.map((r) => r.requests).sort((a, b) => a - b);
    return {
        queries: records.length,
        latencyMs: { median: percentile(ms, 50), p95: percentile(ms, 95) },
        bytesPerQuery: { mean: bytes.reduce((s, b) => s + b, 0) / bytes.length, median: percentile(bytes, 50), p95: percentile(bytes, 95) },
        rangeReadsPerQuery: { median: percentile(reads, 50), p95: percentile(reads, 95) },
    };
}

async function loadQueries(args, search) {
    if (args.queries) {
        const parsed = JSON.parse(fs.readFileSync(args.queries, 'utf8'));
        if (!Array.isArray(parsed) || parsed.length === 0) throw new Error(`${args.queries} must be a non-empty JSON array`);
        return parsed.map((q) => (typeof q === 'string' ? { text: q } : { text: q.text, ids: q.ids }));
    }
    try {
        const evaluation = await search.evaluation();
        if (Array.isArray(evaluation.goldenQueries) && evaluation.goldenQueries.length > 0) {
            console.log(`no --queries given; using the artifact's ${evaluation.goldenQueries.length} embedded golden queries`);
            return evaluation.goldenQueries.map((q) => ({ text: q.text, expected: q.expected }));
        }
    } catch { /* no evaluation segment — fall through */ }
    const sample = search.info().sampleQueries;
    if (sample.length > 0) {
        console.log(`no --queries given; using ${sample.length} manifest sample queries`);
        return sample.map((text) => ({ text }));
    }
    throw new Error('no query set: pass --queries, or use an artifact with an evaluation segment');
}

const args = parseArgs(process.argv.slice(2));
const target = args.positional[0];
const isHttp = /^https?:\/\//.test(target);
const source = isHttp ? httpRangeSource(target) : instrumentedFileSource(target);
if (isHttp) await source.init();

const openBefore = { ...source.stats };
const openStart = performance.now();
const search = await openPikeletFile(source);
const openMs = performance.now() - openStart;
const info = search.info();
const open = {
    ms: openMs,
    bytes: source.stats.bytes - openBefore.bytes,
    requests: source.stats.requests - openBefore.requests,
};
console.log(`opened ${target}`);
console.log(`  identity ${info.identity.slice(0, 16)}..., ${info.records} records, dim ${info.dim}, `
    + `file ${(info.fileBytes / 1048576).toFixed(2)} MiB, resident ${(info.residentBytes / 1024).toFixed(1)} KiB`);
console.log(`  open: ${open.ms.toFixed(0)} ms, ${open.requests} range reads, ${(open.bytes / 1024).toFixed(1)} KiB`);

const queries = await loadQueries(args, search);
const queryOptions = { k: args.k };
if (args.rerank !== undefined) queryOptions.rerank = args.rerank;

const passes = [];
for (let pass = 0; pass <= args.warmPasses; pass++) {
    const records = [];
    for (const q of queries) {
        const before = { ...source.stats };
        const t0 = performance.now();
        const out = await search.query(q.text, queryOptions);
        const ms = performance.now() - t0;
        records.push({
            text: q.text,
            ms,
            bytes: source.stats.bytes - before.bytes,
            requests: source.stats.requests - before.requests,
            matchQuality: out.matchQuality,
            confidence: out.confidence,
            ids: out.results.map((r) => r.id),
            recall: Array.isArray(q.ids)
                ? out.results.filter((r) => q.ids.slice(0, args.k).includes(r.id)).length / Math.min(args.k, q.ids.length)
                : undefined,
            expectedQuality: q.expected,
        });
    }
    passes.push(records);
}

const label = (i) => (i === 0 ? 'first pass (cold reader)' : `repeat pass ${i} (warm reader)`);
for (let i = 0; i < passes.length; i++) {
    const s = summarize(passes[i]);
    console.log(`\n${label(i)} — ${s.queries} queries, k=${args.k}${args.rerank !== undefined ? `, rerank=${args.rerank}` : ''}`);
    console.log(`  latency ms        median ${s.latencyMs.median.toFixed(1)}   p95 ${s.latencyMs.p95.toFixed(1)}`);
    console.log(`  bytes/query       mean ${(s.bytesPerQuery.mean / 1024).toFixed(1)} KiB   median ${(s.bytesPerQuery.median / 1024).toFixed(1)} KiB   p95 ${(s.bytesPerQuery.p95 / 1024).toFixed(1)} KiB`);
    console.log(`  range reads/query median ${s.rangeReadsPerQuery.median}   p95 ${s.rangeReadsPerQuery.p95}`);
}

const first = passes[0];
const qualityCounts = {};
for (const r of first) qualityCounts[r.matchQuality] = (qualityCounts[r.matchQuality] || 0) + 1;
console.log(`\nmatch quality: ${Object.entries(qualityCounts).map(([q, n]) => `${q} ${n}`).join(', ')}`);

const withRecall = first.filter((r) => r.recall !== undefined);
if (withRecall.length > 0) {
    const mean = withRecall.reduce((s, r) => s + r.recall, 0) / withRecall.length;
    console.log(`recall@${args.k} over ${withRecall.length} queries with ground truth: ${(mean * 100).toFixed(1)}%`);
}
const withExpected = first.filter((r) => r.expectedQuality !== undefined);
if (withExpected.length > 0) {
    const hits = withExpected.filter((r) => r.matchQuality === r.expectedQuality).length;
    console.log(`golden match-quality labels reproduced: ${hits}/${withExpected.length}`);
}

if (args.out) {
    fs.writeFileSync(args.out, JSON.stringify({
        target,
        artifact: { identity: info.identity, records: info.records, fileBytes: info.fileBytes },
        k: args.k,
        rerank: args.rerank,
        open,
        passes: passes.map((records, i) => ({ label: label(i), summary: summarize(records), records })),
    }, null, 2));
    console.log(`\nwrote ${args.out}`);
}

await search.close();
await source.close?.();
