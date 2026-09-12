#!/usr/bin/env node
'use strict';

// Head-to-head of the two lazy Search Artifact profiles on the same corpus
// and the same queries:
//
//   range  (.pikelet-range)  resident router + lazy graph traversal —
//                            each hop depends on the previous read, so a
//                            query pays ~15 sequential fetch rounds cold.
//   sketch (.pikelet-sketch) resident sketch scan + exact rerank — every
//                            candidate row is known before the first fetch,
//                            so a query pays ONE parallel fetch round by
//                            construction (spec/SKETCH_PROFILE.md section 3).
//
// The sketch artifact is derived here, in-process, from the committed range
// artifact's own quantized rows — no re-embedding, no original pipeline —
// which is the derivability claim of SKETCH_PROFILE.md section 6 made
// literal.

const fs = require('fs');
const os = require('os');
const path = require('path');
const Pikelet = require('../../../pikelet.js');
const { exportSketchArtifact } = require('../../../pikelet-artifact.js');

const DEFAULT_ARTIFACT = path.join(
    __dirname,
    'static',
    'public',
    'artifacts',
    'pikelet-docs.pikelet-range'
);

function getArg(name, fallback) {
    const index = process.argv.indexOf(`--${name}`);
    return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : fallback;
}

function hasArg(name) {
    return process.argv.includes(`--${name}`);
}

function parsePositiveInt(name, fallback) {
    const raw = getArg(name, String(fallback));
    const value = Number.parseInt(raw, 10);
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`--${name} must be a positive integer`);
    }
    return value;
}

// Same generator and seed as demo.js so both demos ask the same questions.
function syntheticQueries(count, dim, seed = 1234) {
    let state = seed >>> 0;
    const next = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0xffffffff;
    };
    const queries = [];
    for (let i = 0; i < count; i++) {
        const v = new Float32Array(dim);
        for (let d = 0; d < dim; d++) v[d] = next() * 2 - 1;
        queries.push(v);
    }
    return queries;
}

function readIvecs(filePath, limit) {
    const buffer = fs.readFileSync(filePath);
    const rows = [];
    let offset = 0;
    while (offset + 4 <= buffer.byteLength && rows.length < limit) {
        const dim = buffer.readInt32LE(offset);
        offset += 4;
        if (dim <= 0 || offset + dim * 4 > buffer.byteLength) {
            throw new Error(`Invalid ivecs record in ${filePath}`);
        }
        const row = new Array(dim);
        for (let d = 0; d < dim; d++) {
            row[d] = buffer.readInt32LE(offset);
            offset += 4;
        }
        rows.push(row);
    }
    return rows;
}

function mean(values) {
    return values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
}

function formatBytes(bytes) {
    if (bytes >= 4 * 1024 * 1024) return `${(bytes / 1048576).toFixed(1)} MiB`;
    return `${(bytes / 1024).toFixed(1)} KiB`;
}

// Reassemble the quantized row set from the range artifact's records. The
// sketch builder accepts any { dim, count, metric, qdata, scales, offsets }
// source, so the committed artifact itself is a sufficient producer input.
// Batched so each prefetch fits the reader's row LRU: prefetching the whole
// id space at once would evict early records before readNode reaches them
// and degrade to one read per record at large counts.
async function deriveSketchSource(rangeArtifact) {
    const { dim, count, metric } = rangeArtifact;
    const qdata = new Uint8Array(count * dim);
    const scales = new Float32Array(count);
    const offsets = new Float32Array(count);
    const BATCH = 8192;
    for (let start = 0; start < count; start += BATCH) {
        const end = Math.min(start + BATCH, count);
        const ids = [];
        for (let id = start; id < end; id++) ids.push(id);
        await rangeArtifact.prefetch(ids, { gap: 4096, parallelism: 8 });
        for (const id of ids) {
            const node = await rangeArtifact.readNode(id);
            qdata.set(node.qdata, id * dim);
            scales[id] = node.scale;
            offsets[id] = node.offset;
        }
        if (count > 100000 && (start / BATCH) % 16 === 0) {
            process.stderr.write(`  deriving sketch rows: ${end.toLocaleString()}/${count.toLocaleString()}\r`);
        }
    }
    if (count > 100000) process.stderr.write('\n');
    return { dim, count, metric, qdata, scales, offsets };
}

function readFvecs(filePath, limit) {
    const buffer = fs.readFileSync(filePath);
    const vectors = [];
    let offset = 0;
    while (offset + 4 <= buffer.byteLength && vectors.length < limit) {
        const dim = buffer.readInt32LE(offset);
        offset += 4;
        const bytes = dim * 4;
        if (dim <= 0 || offset + bytes > buffer.byteLength) {
            throw new Error(`Invalid fvecs record in ${filePath}`);
        }
        const vector = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
            vector[d] = buffer.readFloatLE(offset);
            offset += 4;
        }
        vectors.push(vector);
    }
    return vectors;
}

async function main() {
    if (hasArg('help')) {
        console.log('Usage: node examples/legacy/range-artifact-demo/sketch_demo.js '
            + '[--artifact file.pikelet-range] [--queries 10] [--k 10] [--ef-search 10] [--rerank 64] [--compact]');
        process.exit(0);
    }

    const artifactPath = getArg('artifact', DEFAULT_ARTIFACT);
    const queryCount = parsePositiveInt('queries', 10);
    const k = parsePositiveInt('k', 10);
    const efSearch = parsePositiveInt('ef-search', 10);
    const rerank = parsePositiveInt('rerank', 96);

    if (!fs.existsSync(artifactPath)) {
        throw new Error(`Artifact not found: ${artifactPath}\nPass --artifact <file.pikelet-range>.`);
    }

    const range = await Pikelet.openRangeArtifactFile(artifactPath);
    const sketchPath = path.join(
        fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-sketch-demo-')),
        path.basename(artifactPath).replace(/\.pikelet-range$/, '') + '.pikelet-sketch'
    );
    try {
        const source = await deriveSketchSource(range);
        const manifest = exportSketchArtifact(source, sketchPath, { recommendedRerank: rerank });
        // Drop the rows cached while deriving so the comparison below starts
        // both readers cold.
        await range.clearCache();
        range.resetStats();

        const sketch = await Pikelet.openSketchArtifactFile(sketchPath);
        try {
            // Real queries when a matching-dimension fvecs file is supplied
            // (e.g. --query-file sift/sift_query.fvecs for the SIFT1M
            // artifact); synthetic otherwise so the demo stays standalone.
            let queries;
            const queryFile = getArg('query-file', null);
            if (queryFile && fs.existsSync(queryFile)) {
                const fromFile = readFvecs(queryFile, queryCount);
                if (fromFile.length > 0 && fromFile[0].length === range.dim) queries = fromFile;
            }
            if (!queries) queries = syntheticQueries(queryCount, range.dim);

            // Optional ground truth (ivecs, e.g. sift/sift_groundtruth.ivecs)
            // turns the profile-vs-profile agreement number into recall@k for
            // each profile independently — and makes the sketch line a live
            // conformance check on SKETCH_PROFILE.md section 4.3's
            // pre-registered recall-vs-C points.
            let groundTruth = null;
            const gtFile = getArg('gt-file', null);
            if (gtFile && fs.existsSync(gtFile)) {
                groundTruth = readIvecs(gtFile, queries.length);
            }
            const recallAt = (ids, gtRow) => {
                const truth = new Set(gtRow.slice(0, k));
                return ids.filter((id) => truth.has(id)).length / k;
            };

            // Every candidate range is known before the first fetch, so the
            // demo issues them all concurrently — the profile's depth-1
            // property. A deployment may throttle concurrency (HTTP/2
            // multiplexes this fine); that is a client choice, not a data
            // dependency, unlike the range profile's hop-by-hop rounds.
            const parallelism = Math.max(1, Math.trunc(getArg('parallelism', '4096')));

            // Cold per query on both sides: this is the remote-storage regime
            // the sketch profile exists for, and it makes the round counts
            // exact rather than cache-diluted. "Miss rounds" counts only
            // rounds that actually touched the source (ROADMAP terminology).
            const measureRange = async (cfg) => {
                const requests = [];
                const bytes = [];
                const rounds = [];
                const recalls = [];
                const resultIds = [];
                for (let i = 0; i < queries.length; i++) {
                    await range.clearCache();
                    const before = range.stats();
                    const result = await range.search(queries[i], k, cfg);
                    const after = range.stats();
                    requests.push(after.rangeRequests - before.rangeRequests);
                    bytes.push(after.rangeBytes - before.rangeBytes);
                    rounds.push(result.rounds.filter((round) => round.requests > 0).length);
                    const ids = result.results.map((row) => row.id);
                    resultIds.push(ids);
                    if (groundTruth) recalls.push(recallAt(ids, groundTruth[i]));
                }
                return {
                    requests: mean(requests),
                    kib: mean(bytes) / 1024,
                    missRounds: mean(rounds),
                    recall: groundTruth ? mean(recalls) : null,
                    resultIds,
                };
            };

            const measureSketch = async () => {
                const requests = [];
                const bytes = [];
                const rounds = [];
                const recalls = [];
                const resultIds = [];
                for (let i = 0; i < queries.length; i++) {
                    sketch.clearCache();
                    const before = sketch.stats();
                    const result = await sketch.search(queries[i], k, { parallelism });
                    const after = sketch.stats();
                    const sRequests = after.rangeRequests - before.rangeRequests;
                    requests.push(sRequests);
                    bytes.push(after.rangeBytes - before.rangeBytes);
                    rounds.push(Math.max(sRequests > 0 ? 1 : 0, Math.ceil(sRequests / parallelism)));
                    const ids = result.results.map((row) => row.id);
                    resultIds.push(ids);
                    if (groundTruth) recalls.push(recallAt(ids, groundTruth[i]));
                }
                return {
                    requests: mean(requests),
                    kib: mean(bytes) / 1024,
                    missRounds: mean(rounds),
                    recall: groundTruth ? mean(recalls) : null,
                    resultIds,
                };
            };

            // --range-sweep gives the graph profile its own knobs (efSearch,
            // expansionBatch, gap — the last row is the ROADMAP's settled
            // recipe) so the comparison is frontier-vs-point, not
            // point-vs-point: batching and gap shuffle traversal cost between
            // rounds, requests, and bytes, but no point on the surface
            // reaches depth 1.
            const sweep = hasArg('range-sweep')
                ? [
                    { label: `ef=${efSearch}`, efSearch },
                    { label: 'ef=80', efSearch: 80 },
                    { label: 'ef=80 batch=8', efSearch: 80, expansionBatch: 8 },
                    { label: 'ef=80 batch=8 gap=64K', efSearch: 80, expansionBatch: 8, gap: 65536, parallelism: 6 },
                ]
                : [{ label: `ef=${efSearch}`, efSearch }];

            const rangeRows = [];
            for (const cfg of sweep) {
                rangeRows.push({ label: cfg.label, ...(await measureRange(cfg)) });
            }
            const sketchRow = await measureSketch();

            // Agreement is computed against the highest-recall (last) range
            // row; with ground truth present, per-profile recall is the
            // primary quality number and agreement is context.
            const referenceRow = rangeRows[rangeRows.length - 1];
            let overlap = 0;
            for (let i = 0; i < queries.length; i++) {
                const rangeIds = new Set(referenceRow.resultIds[i]);
                overlap += sketchRow.resultIds[i].filter((id) => rangeIds.has(id)).length / k;
            }
            const stripIds = ({ resultIds, ...row }) => row;

            const output = {
                corpus: { count: range.count, dim: range.dim, artifact: path.resolve(artifactPath) },
                queries: queries.length,
                k,
                groundTruth: gtFile && groundTruth ? path.resolve(gtFile) : null,
                sketch: {
                    file: sketchPath,
                    sizeBytes: manifest.sizeBytes,
                    residentBytes: sketch.residentBytes,
                    sketchDims: sketch.sketchDims,
                    sketchBits: sketch.sketchBits,
                    rerank,
                    residentVerified: sketch.stats().residentVerified,
                },
                range: {
                    residentBytes: range.routerResident.bytes,
                    perQuery: rangeRows.map(stripIds),
                },
                sketchPerQuery: stripIds(sketchRow),
                topKAgreement: overlap / queries.length,
            };

            if (hasArg('compact')) {
                const recallText = (row) => (row.recall === null ? '' : `, recall@${k} ${(row.recall * 100).toFixed(1)}%`);
                const lines = [
                    `Sketch vs range: ${output.corpus.count.toLocaleString()} vectors, ${output.corpus.dim}D, same ${output.queries} queries, k=${k}, cold per query`,
                    `derived ${path.basename(sketchPath)} from the range artifact in-process (no re-embedding), resident hash verified: ${output.sketch.residentVerified}`,
                ];
                for (const row of output.range.perQuery) {
                    const label = output.range.perQuery.length > 1 ? ` ${row.label.padEnd(21)}` : '';
                    lines.push(`range${label}:  ${formatBytes(output.range.residentBytes)} resident | per query: ${row.requests.toFixed(1)} requests, ${row.missRounds.toFixed(1)} sequential rounds, ${row.kib.toFixed(1)} KiB${recallText(row)}`);
                }
                const s = output.sketchPerQuery;
                const sketchLabel = output.range.perQuery.length > 1 ? ` ${`(C=${rerank})`.padEnd(21)}` : ` (C=${rerank})`;
                lines.push(`sketch${sketchLabel}: ${formatBytes(output.sketch.residentBytes)} resident | per query: ${s.requests.toFixed(1)} requests, ${s.missRounds.toFixed(1)} fetch round(s), ${s.kib.toFixed(1)} KiB${recallText(s)}`);
                lines.push(`top-${k} agreement with range (${referenceRow.label}): ${(output.topKAgreement * 100).toFixed(0)}%`);
                console.log(lines.join('\n'));
            } else {
                console.log(JSON.stringify(output, null, 2));
            }
        } finally {
            await sketch.close();
        }
    } finally {
        await range.close();
        fs.rmSync(path.dirname(sketchPath), { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : error);
    process.exit(1);
});
