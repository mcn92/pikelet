'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const createPikeletApi = require('../pikelet-core.js');

const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const SCALAR_BASENAME = 'engine.scalar';

function makeRng(seed) {
    let state = seed >>> 0;
    return () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

function randomFloatArray(length, rng) {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) out[i] = (rng() * 2 - 1) * 3;
    return out;
}

function randomUnitArray(length, rng) {
    const out = randomFloatArray(length, rng);
    let normSq = 0;
    for (let i = 0; i < out.length; i++) normSq += out[i] * out[i];
    const inv = normSq > 0 ? 1 / Math.sqrt(normSq) : 0;
    for (let i = 0; i < out.length; i++) out[i] *= inv;
    return out;
}

function assertSearchResultsNear(actual, expected, tolerance, label) {
    assert.strictEqual(actual.length, expected.length, `${label}: length mismatch`);
    for (let i = 0; i < actual.length; i++) {
        assert.strictEqual(actual[i].id, expected[i].id, `${label}: id mismatch at ${i}`);
        const delta = Math.abs(actual[i].distance - expected[i].distance);
        assert.ok(
            delta <= tolerance,
            `${label}: distance mismatch at ${i}, got ${actual[i].distance}, expected ${expected[i].distance}, delta ${delta}`
        );
    }
}

function newestMtimeMs(paths) {
    let newest = 0;
    for (const file of paths) {
        const stat = fs.statSync(file);
        if (stat.mtimeMs > newest) newest = stat.mtimeMs;
    }
    return newest;
}

function ensureScalarBuild() {
    const scalarJs = path.join(DIST, `${SCALAR_BASENAME}.js`);
    const scalarWasm = path.join(DIST, `${SCALAR_BASENAME}.wasm`);
    const buildInputs = [
        path.join(ROOT, 'build.sh'),
        path.join(ROOT, 'src', 'engine.cpp'),
        path.join(ROOT, 'src', 'float_hnsw.hpp'),
        path.join(ROOT, 'src', 'uint8_float_hnsw.hpp'),
    ];
    if (fs.existsSync(scalarJs) && fs.existsSync(scalarWasm)) {
        const scalarMtime = Math.min(fs.statSync(scalarJs).mtimeMs, fs.statSync(scalarWasm).mtimeMs);
        if (scalarMtime >= newestMtimeMs(buildInputs)) return;
    }

    const hasEmcc = spawnSync(process.platform === 'win32' ? 'where.exe' : 'command', process.platform === 'win32' ? ['emcc'] : ['-v', 'emcc'], {
        encoding: 'utf8',
        shell: process.platform !== 'win32',
    });
    if (hasEmcc.status !== 0) {
        console.log('SIMD parity checks skipped: dist/engine.scalar.* is older than source inputs and emcc is not available.');
        process.exit(0);
    }

    const result = spawnSync('bash', ['build.sh'], {
        cwd: ROOT,
        env: {
            ...process.env,
            OUT_BASENAME: SCALAR_BASENAME,
            WASM_SIMD: '0',
            PATCH_ENGINE_JS: '0',
            EM_CACHE: process.env.EM_CACHE || path.join(os.tmpdir(), 'pikelet-emscripten-cache'),
        },
        encoding: 'utf8',
    });

    if (result.status !== 0) {
        process.stderr.write(result.stdout || '');
        process.stderr.write(result.stderr || '');
        throw new Error('Failed to build scalar WASM artifact for SIMD parity test');
    }
}

async function loadModule(basename) {
    const factory = require(path.join(DIST, `${basename}.js`));
    const wasmBinary = fs.readFileSync(path.join(DIST, `${basename}.wasm`));
    return factory({
        wasmBinary,
        locateFile(file) {
            return path.join(DIST, file);
        },
    });
}

function makePikeletApi(basename) {
    return createPikeletApi(() => loadModule(basename));
}

async function main() {
    ensureScalarBuild();

    const PikeletSimd = makePikeletApi('engine');
    const PikeletScalar = makePikeletApi(SCALAR_BASENAME);
    const rng = makeRng(0xC0FFEE);

    const annScenarios = [
        { label: 'float-cosine', dim: 64, metric: 'cosine', quantized: false, count: 120, k: 8 },
        { label: 'float-l2', dim: 48, metric: 'l2', quantized: false, count: 100, k: 6 },
        { label: 'u8-cosine', dim: 96, metric: 'cosine', quantized: true, count: 140, k: 8 },
        { label: 'u8-l2', dim: 72, metric: 'l2', quantized: true, count: 110, k: 6 },
    ];

    for (const scenario of annScenarios) {
        const opts = {
            dim: scenario.dim,
            metric: scenario.metric,
            quantized: scenario.quantized,
            maxElements: scenario.count + 32,
            M: 16,
            efConstruction: 200,
            efSearch: 80,
        };
        const idxSimd = await PikeletSimd.create(opts);
        const idxScalar = await PikeletScalar.create(opts);
        try {
            const vectors = Array.from({ length: scenario.count }, () =>
                scenario.metric === 'cosine'
                    ? randomUnitArray(scenario.dim, rng)
                    : randomFloatArray(scenario.dim, rng)
            );

            const idsSimd = idxSimd.addBatch(vectors);
            const idsScalar = idxScalar.addBatch(vectors);
            assert.deepStrictEqual(idsSimd, idsScalar, `${scenario.label}: addBatch IDs match`);

            const deleteIds = [idsSimd[5], idsSimd[17], idsSimd[29]];
            for (const id of deleteIds) {
                idxSimd.delete(id);
                idxScalar.delete(id);
            }

            const queries = [
                vectors[0],
                vectors[Math.floor(scenario.count / 3)],
                vectors[scenario.count - 1],
                scenario.metric === 'cosine'
                    ? randomUnitArray(scenario.dim, rng)
                    : randomFloatArray(scenario.dim, rng),
            ];

            for (let i = 0; i < queries.length; i++) {
                const simdResults = idxSimd.search(queries[i], scenario.k);
                const scalarResults = idxScalar.search(queries[i], scenario.k);
                assertSearchResultsNear(simdResults, scalarResults, 1e-4, `${scenario.label}: search parity before compact q${i}`);
            }

            idxSimd.compact();
            idxScalar.compact();
            assert.strictEqual(idxSimd.count, idxScalar.count, `${scenario.label}: count parity after compact`);

            for (let i = 0; i < queries.length; i++) {
                const simdResults = idxSimd.search(queries[i], scenario.k);
                const scalarResults = idxScalar.search(queries[i], scenario.k);
                assertSearchResultsNear(simdResults, scalarResults, 1e-4, `${scenario.label}: search parity after compact q${i}`);
            }

            const nextVec = scenario.metric === 'cosine'
                ? randomUnitArray(scenario.dim, rng)
                : randomFloatArray(scenario.dim, rng);
            const nextSimdId = idxSimd.add(nextVec);
            const nextScalarId = idxScalar.add(nextVec);
            assert.strictEqual(nextSimdId, nextScalarId, `${scenario.label}: add parity after compact`);

            const exported = idxSimd.export();
            const restoredScalar = await PikeletScalar.create(opts);
            try {
                restoredScalar.import(exported);
                for (let i = 0; i < queries.length; i++) {
                    const simdResults = idxSimd.search(queries[i], scenario.k);
                    const scalarResults = restoredScalar.search(queries[i], scenario.k);
                    assertSearchResultsNear(simdResults, scalarResults, 1e-4, `${scenario.label}: simd export -> scalar import parity q${i}`);
                }
            } finally {
                restoredScalar.dispose();
            }
        } finally {
            idxSimd.dispose();
            idxScalar.dispose();
        }
    }

    console.log('SIMD parity checks passed.');
}

main().catch((err) => {
    console.error(err.stack || err);
    process.exit(1);
});
