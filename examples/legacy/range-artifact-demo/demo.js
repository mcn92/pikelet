#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const Pikelet = require('../../../pikelet.js');

// Default to the docs artifact committed to the repo so the demo runs on a
// fresh clone with no extra data. Point --artifact at a larger .pikelet-range
// (e.g. a SIFT1M export) to exercise scale.
const DEFAULT_ARTIFACT = path.join(
    __dirname,
    'static',
    'public',
    'artifacts',
    'pikelet-docs.pikelet-range'
);
// Optional real-query source; when absent the demo synthesizes queries at the
// artifact's own dimension so it works standalone.
const DEFAULT_QUERY_FILE = path.join(__dirname, '..', '..', '..', 'sift', 'sift_query.fvecs');

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

function summarize(values) {
    if (values.length === 0) return { min: 0, mean: 0, p50: 0, p95: 0, max: 0 };
    const sorted = [...values].sort((a, b) => a - b);
    const pick = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
    return {
        min: sorted[0],
        mean: values.reduce((sum, value) => sum + value, 0) / values.length,
        p50: pick(0.5),
        p95: pick(0.95),
        max: sorted[sorted.length - 1],
    };
}

async function main() {
    if (hasArg('help')) {
        console.log('Usage: node examples/legacy/range-artifact-demo/demo.js [--artifact file] [--query-file sift_query.fvecs] [--queries 10] [--k 10] [--ef-search 10] [--compact]');
        process.exit(0);
    }

    const artifactPath = getArg('artifact', DEFAULT_ARTIFACT);
    const queryFile = getArg('query-file', DEFAULT_QUERY_FILE);
    const queryCount = parsePositiveInt('queries', 10);
    const k = parsePositiveInt('k', 10);
    const efSearch = parsePositiveInt('ef-search', 10);

    if (!fs.existsSync(artifactPath)) {
        throw new Error([
            `Artifact not found: ${artifactPath}`,
            '',
            'The default is the docs artifact committed to the repo. Pass',
            '--artifact <file.pikelet-range> to point at your own.',
        ].join('\n'));
    }

    const artifact = await Pikelet.openRangeArtifactFile(artifactPath);

    // Use the real query file when it matches the artifact's dimension;
    // otherwise synthesize queries at the artifact's dimension so the demo
    // runs on any artifact without external data.
    let queries;
    const haveQueryFile = fs.existsSync(queryFile);
    if (haveQueryFile) {
        const fromFile = readFvecs(queryFile, queryCount);
        if (fromFile.length > 0 && fromFile[0].length === artifact.dim) {
            queries = fromFile;
        }
    }
    if (!queries) {
        queries = syntheticQueries(queryCount, artifact.dim);
    }
    try {
        const perQueryRequests = [];
        const perQueryBytes = [];
        const perQueryRounds = [];
        const sampleResults = [];

        for (let i = 0; i < queries.length; i++) {
            const before = artifact.stats();
            const result = await artifact.search(queries[i], k, { efSearch });
            const after = artifact.stats();
            perQueryRequests.push(after.rangeRequests - before.rangeRequests);
            perQueryBytes.push(after.rangeBytes - before.rangeBytes);
            perQueryRounds.push(result.rounds.length);
            if (i < 3) {
                sampleResults.push({
                    query: i,
                    topK: result.results.map((row) => ({
                        id: row.id,
                        distance: Number(row.distance.toFixed(4)),
                    })),
                });
            }
        }

        const stats = artifact.stats();
        const output = {
            artifact: path.resolve(artifactPath),
            queries: queries.length,
            k,
            efSearch,
            graph: {
                count: artifact.count,
                dim: artifact.dim,
                maxLevel: artifact.maxLevel,
                recordBytes: artifact.recordBytes,
            },
            residentRouter: {
                records: artifact.routerResident.records,
                mib: artifact.routerResident.bytes / 1048576,
            },
            lazyBaseReads: {
                requests: summarize(perQueryRequests),
                bytes: summarize(perQueryBytes),
                meanMiB: summarize(perQueryBytes).mean / 1048576,
                rounds: summarize(perQueryRounds),
                cumulativeRequests: stats.rangeRequests,
                cumulativeFetchedMiB: stats.rangeBytes / 1048576,
                cachedNodes: stats.cachedNodes,
            },
            sampleResults,
        };
        if (hasArg('compact')) {
            console.log([
                `Search Artifact demo: ${output.graph.count.toLocaleString()} vectors, ${output.graph.dim}D`,
                `resident router: ${output.residentRouter.records.toLocaleString()} records (${output.residentRouter.mib.toFixed(1)} MiB)`,
                `lazy reads: ${output.lazyBaseReads.cumulativeRequests.toLocaleString()} requests, ${output.lazyBaseReads.cumulativeFetchedMiB.toFixed(2)} MiB fetched across ${output.queries} queries`,
                `per query: ${output.lazyBaseReads.requests.mean.toFixed(1)} requests, ${output.lazyBaseReads.meanMiB.toFixed(3)} MiB, ${output.lazyBaseReads.rounds.mean.toFixed(1)} rounds mean`,
                `sample top-3 ids: ${output.sampleResults[0].topK.slice(0, 3).map((row) => row.id).join(', ')}`,
            ].join('\n'));
        } else {
            console.log(JSON.stringify(output, null, 2));
        }
    } finally {
        await artifact.close();
    }
}

main().catch((error) => {
    console.error(error && error.stack ? error.stack : error);
    process.exit(1);
});
