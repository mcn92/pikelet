#!/usr/bin/env node
'use strict';

/**
 * RAW ABI ENGINE-CEILING BENCHMARK
 *
 * This is the single intentional raw-ABI benchmark kept outside examples/.
 * It bypasses pikelet-core.js to measure engine-ceiling insert/profile overhead
 * and to access C++ profiling hooks that are not part of the public API.
 *
 * Build Phase Profiler
 *
 * Uses C++-side instrumentation (BuildProfile) to measure where insert
 * time is spent: quantize, upper-layer search, base-layer search,
 * neighbor selection, connect/rewire, and operation counts.
 *
 * Prints a profile report every 10k inserts.
 *
 * Usage:
 *   ./build.sh && node benchmarks/profile_build.js
 *   node benchmarks/profile_build.js --dims 1536 --count 50000
 */

const fs = require('fs');
const path = require('path');
const Pikelet = require('../dist/engine.js');
const { parseBenchmarkArgs, resolveSingleValue } = require('./bench_args');

const args = process.argv.slice(2);
const parsedArgs = parseBenchmarkArgs(args);
function getArg(name, defaultVal) {
    const idx = args.indexOf('--' + name);
    return idx >= 0 && idx + 1 < args.length ? parseInt(args[idx + 1], 10) : defaultVal;
}

const DIMS = getArg('dims', 384);
const COUNT = getArg('count', 100000);
const REPORT_INTERVAL = getArg('interval', 10000);
const M = resolveSingleValue(parsedArgs.m, getArg('M', 12));
const EF_C = resolveSingleValue(parsedArgs.efConstruction, getArg('efc', 150));
const EF_SEARCH = resolveSingleValue(parsedArgs.efSearch, 250);
const MAX_ELEM = COUNT + 1000;

const VECTORS_PATH = path.join(__dirname, '..', 'dist', 'vectors.bin');

function generateSyntheticVec(dims) {
    const v = new Float32Array(dims);
    let norm = 0;
    for (let i = 0; i < dims; i++) {
        v[i] = Math.random() * 2 - 1;
        norm += v[i] * v[i];
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < dims; i++) v[i] /= norm;
    return v;
}

(async () => {
    console.log(`Build Phase Profiler`);
    console.log(`  dims=${DIMS} count=${COUNT} M=${M} ef_c=${EF_C} report_interval=${REPORT_INTERVAL}`);
    console.log();

    const wasmBinary = fs.readFileSync(path.join(__dirname, '..', 'dist', 'engine.wasm'));
    const engine = await Pikelet({ wasmBinary });

    // Check profile functions exist
    if (!engine._pikelet_profile_print || !engine._pikelet_profile_reset) {
        console.error('ERROR: Profile functions not found. Rebuild with: ./build.sh');
        process.exit(1);
    }

    // Load real vectors if available and dims match
    let vectors = null;
    let totalVectors = 0;
    if (DIMS === 384 && fs.existsSync(VECTORS_PATH)) {
        const buf = fs.readFileSync(VECTORS_PATH);
        vectors = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
        totalVectors = Math.floor(vectors.length / DIMS);
        console.log(`Loaded ${totalVectors.toLocaleString()} real embeddings from vectors.bin`);
    } else {
        console.log(`Using synthetic ${DIMS}D vectors`);
    }

    // quantized=1, metric=1 (cosine)
    const handle = engine._pikelet_init(DIMS, MAX_ELEM, 1, 1, M, EF_C, EF_SEARCH, 108);

    const insertPtr = engine._emsc_malloc(DIMS * 4);

    const t0 = performance.now();

    for (let i = 0; i < COUNT; i++) {
        // Get vector
        let vec;
        if (vectors && i < totalVectors) {
            const offset = i * DIMS;
            vec = vectors.subarray(offset, offset + DIMS);
        } else {
            vec = generateSyntheticVec(DIMS);
        }

        engine.HEAPF32.set(vec, insertPtr >> 2);
        engine._pikelet_add(handle, insertPtr);

        // Report every REPORT_INTERVAL inserts
        if ((i + 1) % REPORT_INTERVAL === 0) {
            const rangeStart = i + 1 - REPORT_INTERVAL;
            const rangeEnd = i + 1;
            const elapsed = (performance.now() - t0) / 1000;
            const rate = (i + 1) / elapsed;
            console.log(`\n[${(i + 1).toLocaleString()} / ${COUNT.toLocaleString()}] cumulative: ${elapsed.toFixed(1)}s (${rate.toFixed(0)} vec/s)`);
            engine._pikelet_profile_print(rangeStart, rangeEnd);
            engine._pikelet_profile_reset();
        }
    }

    const totalElapsed = (performance.now() - t0) / 1000;
    console.log(`\n=== TOTAL: ${COUNT.toLocaleString()} inserts in ${totalElapsed.toFixed(2)}s (${(COUNT / totalElapsed).toFixed(0)} vec/s) ===`);

    // Final memory stats
    const mem = engine._pikelet_memory(handle);
    const memMB = (mem / (1024 * 1024)).toFixed(1);
    console.log(`Memory: ${memMB} MB`);

    engine._emsc_free(insertPtr);
    engine._pikelet_dispose(handle);
})();
