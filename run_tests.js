/**
 * Pikelet Test Suite
 *
 * Tests core invariants of the Pikelet index.
 * Run with: npm test
 *
 * No test framework required — plain Node.js.
 */

'use strict';

const Pikelet = require('./pikelet.js');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { pathToFileURL } = require('url');
const goldenSnapshots = require('./test/fixtures/golden_snapshots.js');
const searchOracles = require('./test/fixtures/search_oracles.js');

// ─── Minimal test harness ────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
const failures = [];

function assert(condition, message) {
    if (condition) {
        passed++;
    } else {
        failed++;
        failures.push(message);
        console.error(`  ✗ FAIL: ${message}`);
    }
}

function assertNear(a, b, tolerance = 1e-4, message) {
    assert(Math.abs(a - b) <= tolerance, `${message} (got ${a}, expected ~${b})`);
}

function assertThrows(fn, message) {
    try {
        fn();
        failed++;
        failures.push(`${message} — expected throw but none occurred`);
        console.error(`  ✗ FAIL: ${message} — expected throw but none occurred`);
    } catch (e) {
        passed++;
    }
}

async function assertThrowsAsync(fn, message) {
    try {
        await fn();
        failed++;
        failures.push(`${message} — expected throw but none occurred`);
        console.error(`  ✗ FAIL: ${message} — expected throw but none occurred`);
    } catch (e) {
        passed++;
    }
}

function section(name) {
    console.log(`\n  ${name}`);
    console.log(`  ${'─'.repeat(name.length)}`);
}

// ─── Vector utilities ─────────────────────────────────────────────────────────

function randomVec(dim) {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = Math.random() * 2 - 1;
    return v;
}

function normalizedVec(dim) {
    const v = randomVec(dim);
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < dim; i++) v[i] /= norm;
    return v;
}

function zeroVec(dim) {
    return new Float32Array(dim);
}

function unitVec(dim, axis) {
    const v = new Float32Array(dim);
    v[axis] = 1.0;
    return v;
}

function cosineDist(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function l2Dist(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}

function extractRawEngineBytes(exported) {
    const bytes = exported instanceof Uint8Array ? exported : new Uint8Array(exported);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, true);
    if (magic !== 0x504E434B) return bytes;
    const version = view.getUint32(4, true);
    if (version === 3) {
        const mappingCount = view.getUint32(24, true);
        const wasmSize = view.getUint32(28, true);
        const wasmOffset = 32 + mappingCount * 8;
        return bytes.slice(wasmOffset, wasmOffset + wasmSize);
    }
    return bytes.slice(20);
}

function overwriteU32(bytes, offset, value) {
    const copy = new Uint8Array(bytes);
    new DataView(copy.buffer).setUint32(offset, value >>> 0, true);
    return copy;
}

function decodeBase64Bytes(base64) {
    return new Uint8Array(Buffer.from(base64, 'base64'));
}

function wrapV2Envelope(rawBytes, dim, quantized, metric = 1) {
    const result = new Uint8Array(20 + rawBytes.length);
    const view = new DataView(result.buffer);
    view.setUint32(0, 0x504E434B, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, dim, true);
    view.setUint32(12, metric, true);
    view.setUint32(16, quantized ? 1 : 0, true);
    result.set(rawBytes, 20);
    return result;
}

function wrapV1Envelope(rawBytes, dim, quantized, metric = 1) {
    const result = new Uint8Array(24 + rawBytes.length);
    const view = new DataView(result.buffer);
    view.setUint32(0, 0x504E434B, true);
    view.setUint32(4, 1, true);
    view.setUint32(8, dim, true);
    view.setUint32(12, dim, true); // compressed === dim: plain (non-DCT/PCA) v1 envelope
    view.setUint32(16, metric, true);
    view.setUint32(20, quantized ? 1 : 0, true);
    result.set(rawBytes, 24);
    return result;
}

function mulberry32(seed) {
    let t = seed >>> 0;
    return function next() {
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}

function normalizeInPlace(v) {
    let normSq = 0;
    for (let i = 0; i < v.length; i++) normSq += v[i] * v[i];
    const invNorm = normSq > 0 ? 1 / Math.sqrt(normSq) : 0;
    for (let i = 0; i < v.length; i++) v[i] *= invNorm;
    return v;
}

function buildClusteredCosineDataset(spec) {
    const rand = mulberry32(spec.seed);
    const centers = [];

    for (let c = 0; c < spec.clusters; c++) {
        const center = new Float32Array(spec.dim);
        for (let d = 0; d < spec.dim; d++) center[d] = rand() * 2 - 1;
        centers.push(normalizeInPlace(center));
    }

    const makeVec = (clusterId, noiseScale) => {
        const v = new Float32Array(spec.dim);
        const center = centers[clusterId];
        for (let d = 0; d < spec.dim; d++) {
            v[d] = center[d] + (rand() * 2 - 1) * noiseScale;
        }
        return normalizeInPlace(v);
    };

    const train = [];
    for (let i = 0; i < spec.trainCount; i++) train.push(makeVec(i % spec.clusters, 0.18));

    const queries = [];
    for (let i = 0; i < spec.queryCount; i++) queries.push(makeVec((i * 3) % spec.clusters, 0.11));

    return { train, queries };
}

function bruteForceTopK(vectors, query, k, metric, ids = null) {
    const distFn = metric === 'l2' ? l2Dist : cosineDist;
    return vectors
        .map((vec, idx) => ({
            id: ids ? ids[idx] : idx,
            distance: distFn(vec, query),
        }))
        .sort((a, b) => a.distance - b.distance || a.id - b.id)
        .slice(0, k);
}

function recallAtK(predictedIds, truthIds) {
    const truthSet = new Set(truthIds);
    let hits = 0;
    for (const id of predictedIds) {
        if (truthSet.has(id)) hits++;
    }
    return truthIds.length === 0 ? 1 : hits / truthIds.length;
}

function averageRecallAgainstGroundTruth(resultsByQuery, groundTruthByQuery) {
    let total = 0;
    for (let i = 0; i < resultsByQuery.length; i++) {
        total += recallAtK(
            resultsByQuery[i].map(r => r.id),
            groundTruthByQuery[i].map(r => r.id)
        );
    }
    return total / resultsByQuery.length;
}

function buildAllowedIdSet(count, filterSpec) {
    const allowedIds = [];
    for (let id = 0; id < count; id++) {
        if (id % filterSpec.modulo === filterSpec.remainder) {
            allowedIds.push(id);
        }
    }
    return allowedIds;
}

function assertSearchRowsEqualWithTolerance(actualRows, expectedRows, distanceTolerance, label) {
    assert(actualRows.length === expectedRows.length, `${label}: query count matches golden`);
    for (let qi = 0; qi < Math.min(actualRows.length, expectedRows.length); qi++) {
        const actual = actualRows[qi];
        const expected = expectedRows[qi];
        assert(actual.ids.length === expected.ids.length, `${label}: q${expected.q} result length matches golden`);
        for (let ri = 0; ri < Math.min(actual.ids.length, expected.ids.length); ri++) {
            assert(actual.ids[ri] === expected.ids[ri], `${label}: q${expected.q} id[${ri}] matches golden`);
            assertNear(actual.dists[ri], expected.dists[ri], distanceTolerance, `${label}: q${expected.q} dist[${ri}] matches golden`);
        }
    }
}

async function evaluateHeldOutOracle(config, oracleSpec) {
    const dataset = buildClusteredCosineDataset(oracleSpec);
    const idx = await Pikelet.create({
        dim: oracleSpec.dim,
        metric: 'cosine',
        maxElements: 1024,
        M: 16,
        efConstruction: 200,
        efSearch: 120,
        quantized: config.quantized,
    });
    idx.addBatch(dataset.train);

    const baselineResults = dataset.queries.map(query => idx.search(query, oracleSpec.k));
    const baselineTruth = dataset.queries.map(query => bruteForceTopK(dataset.train, query, oracleSpec.k, 'cosine'));
    const avgRecallBeforeCompact = averageRecallAgainstGroundTruth(baselineResults, baselineTruth);

    const filteredRecallBeforeCompact = {};
    for (const filterSpec of oracleSpec.filteredSpecs || []) {
        const allowedIds = buildAllowedIdSet(dataset.train.length, filterSpec);
        const allowedIdSet = new Set(allowedIds);
        const allowedVectors = allowedIds.map(id => dataset.train[id]);
        const filteredResults = dataset.queries.map(query => idx.searchFiltered(query, oracleSpec.k, allowedIdSet));
        const filteredTruth = dataset.queries.map(query => bruteForceTopK(
            allowedVectors,
            query,
            Math.min(oracleSpec.k, allowedIds.length),
            'cosine',
            allowedIds
        ));
        filteredRecallBeforeCompact[filterSpec.label] =
            averageRecallAgainstGroundTruth(filteredResults, filteredTruth);
    }

    const goldenRows = baselineResults.slice(0, 8).map((rows, q) => ({
        q,
        ids: rows.slice(0, 5).map(r => r.id),
        dists: rows.slice(0, 5).map(r => Number(r.distance.toFixed(6))),
    }));

    const exported = new Uint8Array(idx.export());

    const deletedSet = new Set(oracleSpec.deletedIdsForCompact);
    for (const id of oracleSpec.deletedIdsForCompact) idx.delete(id);
    idx.compact();

    const liveVectors = [];
    const liveIds = [];
    for (let id = 0; id < dataset.train.length; id++) {
        if (deletedSet.has(id)) continue;
        liveVectors.push(dataset.train[id]);
        liveIds.push(id);
    }

    const postCompactResults = dataset.queries.map(query => idx.search(query, oracleSpec.k));
    const postCompactTruth = dataset.queries.map(query => bruteForceTopK(liveVectors, query, oracleSpec.k, 'cosine', liveIds));
    const avgRecallAfterCompact = averageRecallAgainstGroundTruth(postCompactResults, postCompactTruth);

    idx.dispose();

    const idx2 = await Pikelet.create({
        dim: oracleSpec.dim,
        metric: 'cosine',
        maxElements: 1024,
        M: 16,
        efConstruction: 200,
        efSearch: 120,
        quantized: config.quantized,
    });
    idx2.addBatch(dataset.train);
    const exported2 = new Uint8Array(idx2.export());
    const results2 = dataset.queries.slice(0, 8).map(query => idx2.search(query, oracleSpec.k));
    idx2.dispose();

    return {
        avgRecallBeforeCompact,
        avgRecallAfterCompact,
        filteredRecallBeforeCompact,
        goldenRows,
        exported,
        exported2,
        goldenRows2: results2.slice(0, 8).map((rows, q) => ({
            q,
            ids: rows.slice(0, 5).map(r => r.id),
            dists: rows.slice(0, 5).map(r => Number(r.distance.toFixed(6))),
        })),
    };
}

// ─── Default config ───────────────────────────────────────────────────────────

const DIM = 128;

const DEFAULT_CONFIG = {
    dim:            DIM,
    metric:         'cosine',
    maxElements:    1000,
    M:              16,
    efConstruction: 200,
    efSearch:       100,
};

// ─── Test suites ──────────────────────────────────────────────────────────────

async function testCreation() {
    section('Creation');

    // Basic creation
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        assert(idx !== null && idx !== undefined, 'create() returns an index');
        assert(idx.count === 0, 'fresh index has count 0');
        assert(idx.memory > 0, 'fresh index has positive memory');
        idx.dispose();
    }

    // L2 metric
    {
        const idx = await Pikelet.create({ ...DEFAULT_CONFIG, metric: 'l2' });
        assert(idx !== null, 'create() with l2 metric succeeds');
        idx.dispose();
    }

    // Different dimensions
    for (const dim of [32, 64, 256, 384]) {
        const idx = await Pikelet.create({ ...DEFAULT_CONFIG, dim });
        assert(idx !== null, `create() with dim=${dim} succeeds`);
        idx.dispose();
    }

    // Missing required field
    await assertThrowsAsync(
        () => Pikelet.create({ metric: 'cosine', maxElements: 100 }),
        'create() without dim throws'
    );

    await assertThrowsAsync(
        () => Pikelet.create({ ...DEFAULT_CONFIG, compressed: DIM }),
        'create() with removed compressed option throws'
    );

    await assertThrowsAsync(
        () => Pikelet.create({ ...DEFAULT_CONFIG, varianceSample: [normalizedVec(DIM)] }),
        'create() with removed varianceSample option throws'
    );

    // maxElements bounds: the engine ABI takes a C int, so values above
    // 2^31-1 must be rejected in JS instead of truncating (2^32+3 used to
    // silently create a 3-element index reporting capacity 4294967299).
    for (const bad of [2 ** 31, 2 ** 32 + 3, 0, -1, 1.5]) {
        let err = null;
        try { await Pikelet.create({ ...DEFAULT_CONFIG, maxElements: bad }); } catch (e) { err = e; }
        assert(err && err.code === 'INVALID_ARGUMENT',
            `create() rejects maxElements=${bad} with INVALID_ARGUMENT`);
    }

    // Capacity guard: configurations whose eager arena allocation cannot fit
    // the wasm32 heap must be rejected with a coded error at create(), before
    // the engine is even loaded — an uncaught std::bad_alloc inside
    // pancake_init would otherwise abort the whole WASM instance.
    {
        const big = { dim: 4096, maxElements: 50_000_000, metric: 'l2', M: 12 };
        let err = null;
        try { await Pikelet.create(big); } catch (e) { err = e; }
        assert(err && err.code === 'WASM_ALLOCATION_FAILED'
            && err.details && err.details.estimatedBytes > err.details.budgetBytes
            && err.details.estimatedBytes === big.maxElements * (big.dim + 16 * big.M + 39)
            && err.details.quantized === true,
            'create() rejects an over-budget config using the quantized formula');

        // The two backends have different per-element costs (quantized rows
        // are 4x smaller but their edges are 2x larger); the guard must apply
        // the formula for the backend actually requested.
        err = null;
        try { await Pikelet.create({ ...big, quantized: false }); } catch (e) { err = e; }
        assert(err && err.code === 'WASM_ALLOCATION_FAILED'
            && err.details.estimatedBytes === big.maxElements * (4 * big.dim + 8 * big.M + 23)
            && err.details.quantized === false,
            'create() rejects the same config using the float formula');

        // A rejected create must not poison the engine for later indexes.
        const after = await Pikelet.create({ ...DEFAULT_CONFIG });
        after.add(normalizedVec(DIM));
        assert(after.search(normalizedVec(DIM), 1).length === 1,
            'engine remains usable after a capacity rejection');
        after.dispose();
    }

    // Engine init failure surfaces as a coded error. The engine returns
    // uint32_t INVALID_HANDLE (0xFFFFFFFF), which the i32 ABI delivers to JS
    // as -1; create() must detect it and free its scratch allocations.
    {
        const createPikeletApi = require('./pikelet-core.js');
        const freed = [];
        let nextPtr = 0;
        const FailingInit = createPikeletApi(async () => ({
            _emsc_malloc: () => (nextPtr += 8),
            _emsc_free: (p) => freed.push(p),
            _pikelet_init: () => -1,
        }));
        let err = null;
        try { await FailingInit.create({ dim: 4, maxElements: 10 }); } catch (e) { err = e; }
        assert(err && err.code === 'WASM_ALLOCATION_FAILED' && /init failed/i.test(err.message),
            'create() detects engine init failure (-1 handle) with a coded error');
        assert(freed.length === 3, 'create() frees scratch buffers on init failure');
    }
}


async function testAdd() {
    section('Add');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    // Add Float32Array
    const vec = normalizedVec(DIM);
    const id0 = idx.add(vec);
    assert(typeof id0 === 'number', 'add() returns a numeric id');
    assert(idx.count === 1, 'count increments after add');

    // Add plain number[]
    const vec2 = Array.from(normalizedVec(DIM));
    const id1 = idx.add(vec2);
    assert(typeof id1 === 'number', 'add() accepts plain number[]');
    assert(idx.count === 2, 'count increments after second add');
    assert(id1 !== id0, 'ids are unique');

    // Wrong dimension throws
    assertThrows(
        () => idx.add(new Float32Array(DIM + 1)),
        'add() with wrong dimension throws'
    );
    assertThrows(
        () => idx.add(new Float32Array(DIM - 1)),
        'add() with dim-1 throws'
    );

    // Non-numeric elements in a plain array must be rejected, not silently
    // coerced via Number() (e.g. '' -> 0, '1' -> 1, true -> 1, [2] -> 2).
    assertThrows(
        () => idx.add(Array.from(normalizedVec(DIM)).map(String)),
        'add() with numeric-string elements throws'
    );
    assertThrows(
        () => { const v = Array.from(normalizedVec(DIM)); v[0] = ''; return idx.add(v); },
        'add() with empty-string element throws (not coerced to 0)'
    );
    assertThrows(
        () => { const v = Array.from(normalizedVec(DIM)); v[1] = true; return idx.add(v); },
        'add() with boolean element throws'
    );
    assertThrows(
        () => { const v = Array.from(normalizedVec(DIM)); v[2] = [0.1]; return idx.add(v); },
        'add() with nested-array element throws'
    );
    assertThrows(
        () => { const v = Array.from(normalizedVec(DIM)); v[3] = null; return idx.add(v); },
        'add() with null element throws'
    );

    // Float32Array and plain number[] remain valid (no regression).
    const okF32 = idx.add(normalizedVec(DIM));
    const okArr = idx.add(Array.from(normalizedVec(DIM)));
    assert(typeof okF32 === 'number' && typeof okArr === 'number',
        'add() still accepts Float32Array and plain number[]');

    // Count reflects the 4 successful adds (2 above + these 2); the rejected
    // adds did not mutate the index.
    assert(idx.count === 4, 'count unchanged after failed (non-numeric) adds');

    idx.dispose();
}


async function testAddBatch() {
    section('AddBatch');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    assert(Array.isArray(ids), 'addBatch() returns an array');
    assert(ids.length === 10, 'addBatch() returns correct number of ids');
    assert(idx.count === 10, 'count reflects batch add');

    const unique = new Set(ids);
    assert(unique.size === 10, 'addBatch() ids are all unique');

    // Empty batch
    const emptyIds = idx.addBatch([]);
    assert(Array.isArray(emptyIds) && emptyIds.length === 0, 'addBatch([]) returns empty array');
    assert(idx.count === 10, 'count unchanged after empty batch');

    idx.dispose();
}


async function testSearch() {
    section('Search');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    // Insert known vectors and verify self-recall
    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    // Search for each inserted vector — should find itself at rank 1
    let selfRecall = 0;
    for (let i = 0; i < vecs.length; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) selfRecall++;
    }
    assert(selfRecall === vecs.length, `self-recall 100/100 (got ${selfRecall})`);

    // Result structure
    const results = idx.search(vecs[0], 5);
    assert(Array.isArray(results), 'search() returns array');
    assert(results.length === 5, 'search() returns requested k results');
    assert('id' in results[0], 'results have id field');
    assert('distance' in results[0], 'results have distance field');
    assert(typeof results[0].distance === 'number', 'distance is a number');

    // Results are sorted by distance ascending
    for (let i = 1; i < results.length; i++) {
        assert(
            results[i].distance >= results[i - 1].distance,
            `results sorted by distance (pos ${i})`
        );
    }

    // Distance to self is near zero
    assertNear(results[0].distance, 0, 0.01, 'distance to self is near zero');

    // k > count returns count results
    const bigK = idx.search(vecs[0], 1000);
    assert(bigK.length === idx.count, 'search with k>count returns count results');

    // Regression: k crosses a signed i32 boundary, but must be bounded before
    // reaching the C ABI or sizing WASM output buffers.
    const wrappedK = idx.search(vecs[0], 0x80000001);
    assert(wrappedK.length === idx.count, 'search safely bounds k above signed i32 range');
    assert(idx.search(vecs[0], 1).length === 1, 'index remains usable after oversized-k search');

    assertThrows(
        () => idx.search(vecs[0], Number.MAX_SAFE_INTEGER + 1),
        'search rejects integers that cannot be represented safely'
    );

    const overHundred = idx.search(vecs[0], 150);
    assert(overHundred.length === idx.count, 'search with k>100 is not truncated');

    // k=1
    const one = idx.search(vecs[0], 1);
    assert(one.length === 1, 'search with k=1 returns 1 result');

    // Non-numeric query elements rejected, not coerced via Number().
    assertThrows(
        () => idx.search(Array.from(vecs[0]).map(String), 1),
        'search() with numeric-string query throws'
    );
    assertThrows(
        () => { const q = Array.from(vecs[0]); q[0] = ''; return idx.search(q, 1); },
        'search() with empty-string query element throws'
    );
    // Plain number[] query still valid (no regression).
    assert(idx.search(Array.from(vecs[0]), 1).length === 1,
        'search() still accepts a plain number[] query');

    idx.dispose();

    // A result without an external-ID mapping is an internal invariant failure,
    // not a valid raw-ID fallback.
    {
        const broken = await Pikelet.create({
            ...DEFAULT_CONFIG,
            dim: 4,
            maxElements: 4,
            metric: 'l2',
            quantized: false,
        });
        broken.add(new Float32Array([1, 0, 0, 0]));
        broken._intToExt.clear();
        assertThrows(
            () => broken.search(new Float32Array([1, 0, 0, 0]), 1),
            'search fails closed when an internal result has no external-ID mapping'
        );
        broken.dispose();
    }
}


async function testSearchMetrics() {
    section('Search — metric correctness');

    const DIM2 = 4;  // Small dim for exact verification

    // Cosine: unit vectors, known angles
    {
        const idx = await Pikelet.create({
            ...DEFAULT_CONFIG,
            dim: DIM2,
            metric: 'cosine',
            maxElements: 10,
        });

        const a = new Float32Array([1, 0, 0, 0]);
        const b = new Float32Array([0, 1, 0, 0]);  // orthogonal → dist 1.0
        const c = new Float32Array([-1, 0, 0, 0]); // opposite  → dist 2.0

        idx.add(a);
        idx.add(b);
        idx.add(c);

        const results = idx.search(a, 3);
        assertNear(results[0].distance, 0.0, 0.01, 'cosine: self distance ≈ 0');
        assertNear(results[1].distance, 1.0, 0.01, 'cosine: orthogonal distance ≈ 1');
        assertNear(results[2].distance, 2.0, 0.01, 'cosine: opposite distance ≈ 2');

        idx.dispose();
    }

    // L2: known distances
    {
        const idx = await Pikelet.create({
            ...DEFAULT_CONFIG,
            dim: DIM2,
            metric: 'l2',
            maxElements: 10,
        });

        const origin = new Float32Array([0, 0, 0, 0]);
        const unit   = new Float32Array([1, 0, 0, 0]);  // l2 dist = 1
        const far    = new Float32Array([3, 4, 0, 0]);  // l2 dist = 5

        idx.add(origin);
        idx.add(unit);
        idx.add(far);

        const results = idx.search(origin, 3);
        assertNear(results[0].distance, 0.0, 0.01, 'l2: self distance ≈ 0');
        assertNear(results[1].distance, 1.0, 0.01, 'l2: unit distance ≈ 1');
        assertNear(results[2].distance, 5.0, 0.01, 'l2: (3,4) distance ≈ 5');

        idx.dispose();
    }
}


async function testDelete() {
    section('Delete');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    // Delete one vector
    idx.delete(ids[0]);
    assert(idx.count === 10, 'count unchanged after delete (soft delete)');
    assert(idx.ghostCount > 0, 'ghostCount > 0 after delete');

    // Deleted vector does not appear in search results
    const results = idx.search(vecs[0], 10);
    const deletedAppears = results.some(r => r.id === ids[0]);
    assert(!deletedAppears, 'deleted vector excluded from search results');

    // Delete remaining vectors
    for (let i = 1; i < ids.length; i++) idx.delete(ids[i]);
    assert(idx.ghostCount === 10, 'ghostCount equals total deleted');

    idx.dispose();
}


async function testGhostRatio() {
    section('Ghost ratio');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    assert(idx.ghostRatio === 0, 'ghostRatio is 0 with no deletes');

    // Delete half
    for (let i = 0; i < 50; i++) idx.delete(ids[i]);

    assertNear(idx.ghostRatio, 0.5, 0.01, 'ghostRatio ≈ 0.5 after deleting half');

    idx.dispose();
}


async function testCompact() {
    section('Compact');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    const memBefore = idx.memory;

    // Delete half and compact
    for (let i = 0; i < 50; i++) idx.delete(ids[i]);
    idx.compact();

    assert(idx.ghostCount === 0, 'ghostCount is 0 after compact');
    assert(idx.ghostRatio === 0, 'ghostRatio is 0 after compact');

    const memAfter = idx.memory;
    assert(memAfter < memBefore, `memory decreased after compact (${memBefore} → ${memAfter})`);

    // Remaining vectors still searchable after compact
    let recallAfterCompact = 0;
    for (let i = 50; i < 100; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) recallAfterCompact++;
    }
    assert(recallAfterCompact === 50, `recall intact after compact (${recallAfterCompact}/50)`);

    idx.dispose();
}


async function testMemory() {
    section('Memory');

    const idx = await Pikelet.create(DEFAULT_CONFIG);
    const memEmpty = idx.memory;

    // Memory grows with insertions
    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    idx.addBatch(vecs);

    assert(idx.memory > memEmpty, 'memory grows after insertions');

    idx.dispose();
}


async function testDispose() {
    section('Dispose');

    const idx = await Pikelet.create(DEFAULT_CONFIG);
    idx.add(normalizedVec(DIM));
    idx.dispose();

    // All operations after dispose should throw
    assertThrows(() => idx.add(normalizedVec(DIM)),   'add() after dispose throws');
    assertThrows(() => idx.search(normalizedVec(DIM), 1), 'search() after dispose throws');
    assertThrows(() => idx.delete(0),                 'delete() after dispose throws');
    assertThrows(() => idx.compact(),                 'compact() after dispose throws');
    assertThrows(() => idx.count,                     'count after dispose throws');
    assertThrows(() => idx.memory,                    'memory after dispose throws');

    // Double dispose should not throw
    try {
        idx.dispose();
        passed++;
    } catch (e) {
        failed++;
        failures.push('double dispose threw unexpectedly');
    }
}


async function testDeterminism() {
    section('Determinism');

    // Same vectors, same order → same search results
    const vecs = Array.from({ length: 50 }, () => normalizedVec(DIM));
    const query = normalizedVec(DIM);

    const runSearch = async () => {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        idx.addBatch(vecs);
        const results = idx.search(query, 5);
        idx.dispose();
        return results.map(r => r.id);
    };

    const run1 = await runSearch();
    const run2 = await runSearch();

    const deterministic = run1.every((id, i) => id === run2[i]);
    assert(deterministic, 'search results are deterministic across identical builds');
}


async function testSearchDuringMutation() {
    section('Search during mutation');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    // Build initial index
    const vecs = Array.from({ length: 100 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    // Interleave deletes and searches — search should never return deleted ids
    for (let i = 0; i < 50; i++) {
        idx.delete(ids[i]);
        const results = idx.search(vecs[i], 10);
        const contaminated = results.some(r => ids.slice(0, i + 1).includes(r.id));
        assert(!contaminated, `no deleted ids in search results after ${i + 1} deletes`);
    }

    idx.dispose();
}


async function testRuntimeEntryPoints() {
    section('Runtime entry points');

    assert(require.resolve('pikelet-wasm/engine.wasm').endsWith(path.join('dist', 'engine.wasm')),
        'engine.wasm remains a stable package asset path');
    assertThrows(
        () => require.resolve('pikelet-wasm/engine'),
        'generated Emscripten glue is not a public package export'
    );

    // CommonJS package entry
    {
        const CjsPikelet = require('./pikelet.js');
        const idx = await CjsPikelet.create({ dim: 4, maxElements: 10 });
        idx.add(new Float32Array([1, 0, 0, 0]));
        const results = idx.search(new Float32Array([1, 0, 0, 0]), 1);
        assert(results.length === 1 && results[0].id === 0, 'CJS entry works');
        idx.dispose();
    }

    // ESM package entry
    {
        const { default: EsmPikelet } = await import(pathToFileURL(path.join(process.cwd(), 'pikelet.node.mjs')).href);
        const idx = await EsmPikelet.create({ dim: 4, maxElements: 10 });
        idx.add(new Float32Array([1, 0, 0, 0]));
        const results = idx.search(new Float32Array([1, 0, 0, 0]), 1);
        assert(results.length === 1 && results[0].id === 0, 'ESM node entry works');
        idx.dispose();
    }

    // Browser-style instantiateWasm path. This validates the web runtime
    // loading contract without depending on a specific bundler or browser.
    {
        const loadEngine = require('./dist/engine.js');
        const createPikeletApi = require('./pikelet-core.js');
        async function compileWasmWithFallback() {
            const entries = [
                ['engine.wasm', 'simd'],
                ['engine.scalar.wasm', 'scalar'],
            ];
            let lastError = null;
            for (const [fileName, label] of entries) {
                try {
                    const wasmBinary = fs.readFileSync(path.join(process.cwd(), 'dist', fileName));
                    const compiled = await WebAssembly.compile(
                        wasmBinary.buffer.slice(
                            wasmBinary.byteOffset,
                            wasmBinary.byteOffset + wasmBinary.byteLength
                        )
                    );
                    return { compiled, label };
                } catch (error) {
                    lastError = error;
                }
            }
            throw lastError;
        }
        const { compiled, label } = await compileWasmWithFallback();

        const BrowserStylePikelet = createPikeletApi(() => loadEngine({
            instantiateWasm(imports, successCallback) {
                WebAssembly.instantiate(compiled, imports)
                    .then(instance => successCallback(instance))
                    .catch(err => { throw err; });
                return {};
            }
        }));

        const idx = await BrowserStylePikelet.create({ dim: 4, maxElements: 10 });
        idx.add(new Float32Array([1, 0, 0, 0]));
        const results = idx.search(new Float32Array([1, 0, 0, 0]), 1);
        assert(results.length === 1 && results[0].id === 0, `browser-style instantiateWasm path works (${label})`);
        idx.dispose();
    }
}


async function testExportImport() {
    section('Export / Import');

    const idx = await Pikelet.create(DEFAULT_CONFIG);

    const vecs = Array.from({ length: 50 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);
    idx.delete(ids[1]);
    idx.delete(ids[3]);
    idx.compact();
    const extraId = idx.add(normalizedVec(DIM));

    const query   = vecs[0];
    const before  = idx.search(query, 5).map(r => r.id);
    const exported = idx.export();

    assert(exported instanceof Uint8Array || exported instanceof ArrayBuffer,
        'export() returns binary data');
    assert(exported.byteLength > 0, 'exported data is non-empty');

    idx.dispose();

    // Import into fresh index
    const idx2 = await Pikelet.create(DEFAULT_CONFIG);
    idx2.import(exported);

    assert(idx2.count === 49, 'imported index has correct count');

    const after = idx2.search(query, 5).map(r => r.id);
    const roundTrip = before.every((id, i) => id === after[i]);
    assert(roundTrip, 'search results identical after export/import round-trip');
    assert(after.includes(ids[0]), 'stable external IDs survive round-trip');
    assert(!after.includes(ids[1]), 'deleted external IDs stay deleted after compact+export/import');

    const newId = idx2.add(normalizedVec(DIM));
    assert(newId === extraId + 1, 'new IDs continue from nextExtId after import');

    idx2.dispose();
}


async function testLargeIndex() {
    section('Scale — 1000 vectors');

    const idx = await Pikelet.create({
        ...DEFAULT_CONFIG,
        maxElements: 1000,
    });

    const vecs = Array.from({ length: 1000 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    assert(idx.count === 1000, 'count correct at 1000 vectors');

    // Self-recall on a sample
    let recall = 0;
    const sample = 50;
    for (let i = 0; i < sample; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) recall++;
    }
    // HNSW is approximate — expect high but not necessarily perfect recall
    assert(recall >= 45, `recall@1 ≥ 90% at 1000 vectors (got ${recall}/${sample})`);

    idx.dispose();
}


async function testQuantized() {
    section('Quantized mode');

    const QDIM = 128;
    const QCONFIG = {
        dim:            QDIM,
        metric:         'cosine',
        maxElements:    500,
        M:              16,
        efConstruction: 200,
        efSearch:       100,
        quantized:      true,
    };

    // Creation
    const idx = await Pikelet.create(QCONFIG);
    assert(idx !== null, 'create() with quantized: true succeeds');
    assert(idx.count === 0, 'fresh quantized index has count 0');

    // Add vectors
    const vecs = Array.from({ length: 100 }, () => normalizedVec(QDIM));
    const ids  = idx.addBatch(vecs);
    assert(idx.count === 100, 'quantized: count correct after addBatch');

    // Self-recall (allow slightly lower due to quantization)
    let selfRecall = 0;
    for (let i = 0; i < vecs.length; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) selfRecall++;
    }
    assert(selfRecall >= 90, `quantized: self-recall ≥ 90% (got ${selfRecall}/100)`);

    // Delete + compact
    for (let i = 0; i < 20; i++) idx.delete(ids[i]);
    assert(idx.ghostCount > 0, 'quantized: ghostCount > 0 after deletes');

    idx.compact();
    assert(idx.ghostCount === 0, 'quantized: ghostCount 0 after compact');

    // Surviving vectors still searchable
    let recallAfterCompact = 0;
    for (let i = 20; i < 100; i++) {
        const results = idx.search(vecs[i], 1);
        if (results.length > 0 && results[0].id === ids[i]) recallAfterCompact++;
    }
    assert(recallAfterCompact >= 70, `quantized: recall after compact ≥ 87% (got ${recallAfterCompact}/80)`);

    // Export / Import
    const exported = idx.export();
    assert(exported.byteLength > 0, 'quantized: export produces data');

    const idx2 = await Pikelet.create(QCONFIG);
    idx2.import(exported);
    assert(idx2.count === 80, 'quantized: imported count correct');

    idx2.dispose();
    idx.dispose();
}


async function testErrorPaths() {
    section('Error paths');

    // Import corrupted data
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const garbage = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7]);
        let threw = false;
        try {
            idx.import(garbage);
        } catch (e) {
            threw = true;
        }
        assert(threw, 'import with corrupted data throws');
        idx.dispose();
    }

    // Import empty data
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        let threw = false;
        try {
            idx.import(new Uint8Array(0));
        } catch (e) {
            threw = true;
        }
        assert(threw, 'import with empty data throws');
        idx.dispose();
    }

    // Import from ArrayBuffer (not Uint8Array)
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
        idx.addBatch(vecs);
        const exported = idx.export();
        idx.dispose();

        const idx2 = await Pikelet.create(DEFAULT_CONFIG);
        idx2.import(exported.buffer); // ArrayBuffer, not Uint8Array
        assert(idx2.count === 10, 'import from ArrayBuffer works');
        idx2.dispose();
    }

    // Import into non-empty index (overwrites)
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const vecs1 = Array.from({ length: 20 }, () => normalizedVec(DIM));
        idx.addBatch(vecs1);

        const idx2 = await Pikelet.create(DEFAULT_CONFIG);
        const vecs2 = Array.from({ length: 10 }, () => normalizedVec(DIM));
        idx2.addBatch(vecs2);
        const snapshot = idx2.export();
        idx2.dispose();

        idx.import(snapshot);
        assert(idx.count === 10, 'import into non-empty index overwrites (count matches source)');
        idx.dispose();
    }

    // Import rejects snapshots larger than the destination maxElements.
    {
        const src = await Pikelet.create({ ...DEFAULT_CONFIG, maxElements: 6 });
        const srcVecs = Array.from({ length: 6 }, () => normalizedVec(DIM));
        src.addBatch(srcVecs);
        const exported = src.export();
        src.dispose();

        const dst = await Pikelet.create({ ...DEFAULT_CONFIG, maxElements: 5 });
        const liveVecs = Array.from({ length: 2 }, () => normalizedVec(DIM));
        const liveIds = dst.addBatch(liveVecs);
        const beforeTop = dst.search(liveVecs[0], 1)[0];

        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('Import failed'), 'import rejects snapshot count greater than maxElements');
        assert(dst.count === 2, 'over-capacity import preserves existing destination count');
        const afterTop = dst.search(liveVecs[0], 1)[0];
        assert(afterTop.id === liveIds[0], 'over-capacity import preserves existing destination ID mapping');
        assert(afterTop.id === beforeTop.id, 'over-capacity import preserves existing destination search identity');
        dst.dispose();
    }

    // Backend-level import failure preserves an existing destination index.
    {
        const src = await Pikelet.create(DEFAULT_CONFIG);
        src.addBatch(Array.from({ length: 5 }, () => normalizedVec(DIM)));
        const exported = new Uint8Array(src.export());
        src.dispose();

        const rawSizeOffset = 28;
        const view = new DataView(exported.buffer, exported.byteOffset, exported.byteLength);
        view.setUint32(rawSizeOffset, 40, true); // valid raw header, truncated backend payload

        const dst = await Pikelet.create(DEFAULT_CONFIG);
        const liveVecs = Array.from({ length: 4 }, () => normalizedVec(DIM));
        const liveIds = dst.addBatch(liveVecs);
        dst.delete(liveIds[0]);
        dst.compact();
        const extraId = dst.add(liveVecs[0]);
        const beforeCount = dst.count;
        const beforeTop = dst.search(liveVecs[1], 1)[0];

        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('Import failed'), 'backend-level import failure throws Import failed');
        assert(dst.count === beforeCount, 'backend-level failed import preserves destination count');
        const afterTop = dst.search(liveVecs[1], 1)[0];
        assert(afterTop.id === liveIds[1], 'backend-level failed import preserves compacted ID mapping');
        assert(beforeTop.id === afterTop.id, 'backend-level failed import preserves search identity');
        assert(dst.search(liveVecs[0], 1)[0].id === extraId, 'backend-level failed import preserves later added ID mapping');
        dst.dispose();
    }

    // add() with wrong dimension after successful adds
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        idx.add(normalizedVec(DIM));
        assertThrows(
            () => idx.add(new Float32Array(DIM * 2)),
            'add() wrong dim after successful add still throws'
        );
        assert(idx.count === 1, 'count unchanged after failed add');
        idx.dispose();
    }

    // Import config mismatch: dim 64 → 128
    {
        const src = await Pikelet.create({ ...DEFAULT_CONFIG, dim: 64 });
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(64)));
        const exported = src.export();
        src.dispose();

        const dst = await Pikelet.create({ ...DEFAULT_CONFIG, dim: 128 });
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('dim mismatch'), 'import dim 64→128 throws with dim mismatch message');
        dst.dispose();
    }

    // Import config mismatch: dim 128 → 64
    {
        const src = await Pikelet.create({ ...DEFAULT_CONFIG, dim: 128 });
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(128)));
        const exported = src.export();
        src.dispose();

        const dst = await Pikelet.create({ ...DEFAULT_CONFIG, dim: 64 });
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('dim mismatch'), 'import dim 128→64 throws with dim mismatch message');
        dst.dispose();
    }

    // Import config mismatch: cosine → l2
    {
        const src = await Pikelet.create({ ...DEFAULT_CONFIG, metric: 'cosine' });
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        const exported = src.export();
        src.dispose();

        const dst = await Pikelet.create({ ...DEFAULT_CONFIG, metric: 'l2' });
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('metric mismatch'), 'import cosine→l2 throws with metric mismatch message');
        dst.dispose();
    }

    // Import of deprecated DCT/PCA envelope is rejected
    {
        const src = await Pikelet.create(DEFAULT_CONFIG);
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        const exported = src.export();
        src.dispose();

        const deprecated = new Uint8Array(exported);
        const view = new DataView(deprecated.buffer);
        view.setUint32(4, 1, true);
        view.setUint32(12, 64, true);

        const dst = await Pikelet.create(DEFAULT_CONFIG);
        let msg = '';
        try { dst.import(deprecated); } catch (e) { msg = e.message; }
        assert(msg.includes('no longer supported'), 'import rejects deprecated DCT/PCA envelopes');
        dst.dispose();
    }

    // Import of truncated v3 envelope is rejected
    {
        const src = await Pikelet.create(DEFAULT_CONFIG);
        src.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        const exported = src.export();
        src.dispose();

        const truncated = exported.subarray(0, exported.byteLength - 3);
        const dst = await Pikelet.create(DEFAULT_CONFIG);
        let msg = '';
        try { dst.import(truncated); } catch (e) { msg = e.message; }
        assert(msg.includes('truncated v3 envelope payload'), 'import rejects truncated v3 payload');
        dst.dispose();
    }

    // Import of v3 envelope with invalid nextExtId is rejected
    {
        const src = await Pikelet.create(DEFAULT_CONFIG);
        src.addBatch(Array.from({ length: 5 }, () => normalizedVec(DIM)));
        const exported = new Uint8Array(src.export());
        src.dispose();

        const view = new DataView(exported.buffer, exported.byteOffset, exported.byteLength);
        view.setUint32(20, 1, true);

        const dst = await Pikelet.create(DEFAULT_CONFIG);
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('nextExtId is invalid'), 'import rejects v3 envelope with invalid nextExtId');
        dst.dispose();
    }

    // Import of v3 envelope with mismatched mapping count is rejected
    {
        const src = await Pikelet.create(DEFAULT_CONFIG);
        src.addBatch(Array.from({ length: 5 }, () => normalizedVec(DIM)));
        const exported = new Uint8Array(src.export());
        src.dispose();

        const view = new DataView(exported.buffer, exported.byteOffset, exported.byteLength);
        view.setUint32(24, 4, true);
        const oldOffset = 32 + 5 * 8;
        const newOffset = 32 + 4 * 8;
        const wasmBytes = exported.slice(oldOffset);
        exported.set(wasmBytes, newOffset);
        view.setUint32(28, wasmBytes.byteLength, true);

        const dst = await Pikelet.create(DEFAULT_CONFIG);
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('mapping count mismatch'), 'import rejects v3 envelope with mapping-count mismatch');
        dst.dispose();
    }

    // Import of v3 envelope with duplicate external IDs is rejected
    {
        const src = await Pikelet.create(DEFAULT_CONFIG);
        src.addBatch(Array.from({ length: 5 }, () => normalizedVec(DIM)));
        const exported = new Uint8Array(src.export());
        src.dispose();

        const view = new DataView(exported.buffer, exported.byteOffset, exported.byteLength);
        const mappingOffset = 32;
        const firstExtId = view.getUint32(mappingOffset + 4, true);
        view.setUint32(mappingOffset + 12, firstExtId, true);

        const dst = await Pikelet.create(DEFAULT_CONFIG);
        let msg = '';
        try { dst.import(exported); } catch (e) { msg = e.message; }
        assert(msg.includes('mapping contains duplicates'), 'import rejects v3 envelope with duplicate external IDs');
        dst.dispose();
    }
}


async function testEdgeCases() {
    section('Edge cases');

    // Search on empty index
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const results = idx.search(normalizedVec(DIM), 5);
        assert(Array.isArray(results), 'search on empty index returns array');
        assert(results.length === 0, 'search on empty index returns 0 results');
        idx.dispose();
    }

    // k=0 search
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        idx.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        const results = idx.search(normalizedVec(DIM), 0);
        assert(results.length === 0, 'search with k=0 returns 0 results');
        idx.dispose();
    }

    // Delete non-existent ID (should not throw)
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        idx.addBatch(Array.from({ length: 5 }, () => normalizedVec(DIM)));
        let threw = false;
        try {
            idx.delete(9999);
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'delete non-existent ID does not throw');
        assert(idx.count === 5, 'count unchanged after deleting non-existent ID');
        assert(idx.ghostCount === 0, 'ghostCount unchanged after deleting non-existent ID');
        idx.dispose();
    }

    // Double-delete same ID
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 5 }, () => normalizedVec(DIM));
        const ids = idx.addBatch(vecs);
        idx.delete(ids[0]);
        let threw = false;
        try {
            idx.delete(ids[0]);
        } catch (e) {
            threw = true;
        }
        assert(!threw, 'double-delete same ID does not throw');
        idx.dispose();
    }

    // Compact with no deletions (no ghosts)
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 20 }, () => normalizedVec(DIM));
        const ids = idx.addBatch(vecs);
        idx.compact(); // no-op compact
        assert(idx.count === 20, 'compact with no ghosts preserves count');
        assert(idx.ghostCount === 0, 'ghostCount still 0 after no-op compact');

        // Vectors still searchable
        let recall = 0;
        for (let i = 0; i < vecs.length; i++) {
            const results = idx.search(vecs[i], 1);
            if (results.length > 0 && results[0].id === ids[i]) recall++;
        }
        assert(recall === 20, `recall intact after no-op compact (${recall}/20)`);
        idx.dispose();
    }

    // Engine compact_remap validates output before mutating.
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const ids = idx.addBatch(Array.from({ length: 3 }, () => normalizedVec(DIM)));
        idx.delete(ids[1]);
        assert(idx.ghostCount === 1, 'compact_remap guard setup has one ghost');

        const nullResult = idx._e._pikelet_compact_remap(idx._handle, 0, 3);
        assert(nullResult === 0, 'compact_remap with null output returns 0');
        assert(idx.count === 3, 'compact_remap with null output does not compact');
        assert(idx.ghostCount === 1, 'compact_remap with null output preserves ghost count');

        const ptr = idx._e._emsc_malloc(4);
        const zeroCapacityResult = idx._e._pikelet_compact_remap(idx._handle, ptr, 0);
        idx._e._emsc_free(ptr);
        assert(zeroCapacityResult === 0, 'compact_remap with zero capacity returns 0');
        assert(idx.count === 3, 'compact_remap with zero capacity does not compact');
        assert(idx.ghostCount === 1, 'compact_remap with zero capacity preserves ghost count');

        idx.dispose();
    }

    // Compact after deleting ALL vectors
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const vecs = Array.from({ length: 10 }, () => normalizedVec(DIM));
        const ids = idx.addBatch(vecs);
        for (const id of ids) idx.delete(id);
        idx.compact();
        assert(idx.ghostCount === 0, 'ghostCount 0 after compact of fully-deleted index');
        const results = idx.search(normalizedVec(DIM), 5);
        assert(results.length === 0, 'search returns nothing after full delete + compact');
        idx.dispose();
    }

    // Export on empty index
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const exported = idx.export();
        assert(exported.byteLength > 0, 'export on empty index produces data');

        const idx2 = await Pikelet.create(DEFAULT_CONFIG);
        idx2.import(exported);
        assert(idx2.count === 0, 'import of empty export gives count 0');
        idx2.dispose();
        idx.dispose();
    }

    // All getters after dispose
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        idx.add(normalizedVec(DIM));
        idx.dispose();

        assertThrows(() => idx.export(),             'export() after dispose throws');
        assertThrows(() => idx.import(new Uint8Array(1)), 'import() after dispose throws');
        assertThrows(() => idx.ghostCount,           'ghostCount after dispose throws');
        assertThrows(() => idx.ghostRatio,           'ghostRatio after dispose throws');
        // dim is a plain property, not guarded — it doesn't access WASM
        assert(idx.dim === DIM, 'dim still readable after dispose (no WASM call)');
    }

    // search() with plain number[] query (not Float32Array)
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const vec = normalizedVec(DIM);
        idx.add(vec);
        const query = Array.from(vec);
        const results = idx.search(query, 1);
        assert(results.length === 1, 'search with plain number[] query works');
        assertNear(results[0].distance, 0, 0.01, 'search with number[] query finds self');
        idx.dispose();
    }

    // addBatch with single element
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const ids = idx.addBatch([normalizedVec(DIM)]);
        assert(ids.length === 1, 'addBatch with single vector returns 1 id');
        assert(idx.count === 1, 'count is 1 after single-element batch');
        idx.dispose();
    }
}


// ─── New tests ───────────────────────────────────────────────────────────────

async function testMaxElementsOverflow() {
    section('maxElements overflow');

    // maxElements is a HARD limit in the C++ layer: insert() returns UINT32_MAX
    // when count_ >= max_elements_, and pikelet.js translates that to a throw.
    //
    // IMPORTANT: The WASM backend uses a single global index. Each Pikelet.create()
    // reinitializes that global, clobbering any previously created index. All
    // overflow assertions must therefore use a single index instance.

    const SMALL = 5;
    const idx = await Pikelet.create({
        ...DEFAULT_CONFIG,
        maxElements: SMALL,
    });

    // Fill to declared capacity
    const ids = [];
    for (let i = 0; i < SMALL; i++) {
        ids.push(idx.add(normalizedVec(DIM)));
    }
    assert(idx.count === SMALL, `count is ${SMALL} at capacity`);

    // One more must throw
    let threw = false;
    let errMsg = '';
    try {
        idx.add(normalizedVec(DIM));
    } catch (e) {
        threw = true;
        errMsg = e.message;
    }
    assert(threw, 'add() beyond maxElements throws');
    assert(
        errMsg.toLowerCase().includes('full') || errMsg.toLowerCase().includes('failed'),
        `overflow error message is meaningful (got: "${errMsg}")`
    );

    // Count and search still correct after the failed add
    assert(idx.count === SMALL, 'count unchanged after overflow attempt');
    const results = idx.search(normalizedVec(DIM), 3);
    assert(results.length === 3, 'search still works after overflow attempt');

    // addBatch hitting the limit also throws — free one slot first so we
    // can verify the batch fails when the *batch itself* causes overflow,
    // without needing a second create() that would clobber the global.
    idx.delete(ids[0]);
    idx.compact(); // live count_ drops to 4, max_elements_ still 5
    idx.add(normalizedVec(DIM)); // back to 5 (at capacity again)
    let batchThrew = false;
    try {
        idx.addBatch([normalizedVec(DIM), normalizedVec(DIM)]);
    } catch (e) {
        batchThrew = true;
    }
    assert(batchThrew, 'addBatch() that exceeds maxElements throws');

    idx.dispose();
}


async function testExportImportWithGhosts() {
    section('Export / Import — with ghosts');

    const idx = await Pikelet.create(DEFAULT_CONFIG);
    const vecs = Array.from({ length: 20 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    const deletedIndices = [0, 5, 10];
    for (const i of deletedIndices) idx.delete(ids[i]);
    assert(idx.ghostCount === 3, 'ghostCount is 3 before export');

    let msg = '';
    try { idx.export(); } catch (e) { msg = e.message; }
    assert(msg.includes('compact() required before export'), 'export() with ghosts throws explicit compact-required error');

    idx.compact();
    assert(idx.count === 17, 'compact() reduces count to 17 live vectors');
    const compactExported = idx.export();

    const idx2 = await Pikelet.create(DEFAULT_CONFIG);
    idx2.import(compactExported);
    assert(idx2.count === 17, 'import after compact() gives 17 vectors');

    let ghostLeak = 0;
    for (const i of deletedIndices) {
        const r = idx2.search(vecs[i], 1);
        if (r.length > 0 && r[0].distance < 0.001) ghostLeak++;
    }
    assert(ghostLeak === 0, 'compact()+export()+import() produces clean snapshot with no ghost data');

    idx2.dispose();
    idx.dispose();
}


async function testExportImportWithGhostsQuantized() {
    section('Export / Import — with ghosts (quantized)');

    const idx = await Pikelet.create({
        ...DEFAULT_CONFIG,
        quantized: true,
    });
    const vecs = Array.from({ length: 20 }, () => normalizedVec(DIM));
    const ids  = idx.addBatch(vecs);

    idx.delete(ids[2]);
    idx.delete(ids[7]);
    assert(idx.ghostCount === 2, 'quantized: ghostCount is 2 before export');

    let msg = '';
    try { idx.export(); } catch (e) { msg = e.message; }
    assert(msg.includes('compact() required before export'), 'quantized: export() with ghosts throws explicit compact-required error');

    idx.compact();
    const exported = idx.export();
    const idx2 = await Pikelet.create({
        ...DEFAULT_CONFIG,
        quantized: true,
    });
    idx2.import(exported);
    assert(idx2.count === 18, 'quantized: import after compact preserves live count');

    idx2.dispose();
    idx.dispose();
}


async function testAddBatchPartialFailure() {
    section('AddBatch — partial failure');

    // addBatch with a wrong-dimension vector mid-batch
    {
        const idx = await Pikelet.create({ ...DEFAULT_CONFIG, maxElements: 20 });
        const goodVec  = normalizedVec(DIM);
        const badVec   = new Float32Array(DIM + 10); // wrong dim
        const goodVec2 = normalizedVec(DIM);

        let threw = false;
        try {
            idx.addBatch([goodVec, badVec, goodVec2]);
        } catch (e) {
            threw = true;
        }
        assert(threw, 'addBatch with wrong-dim vector throws');

        // Index must not be corrupted — search should still work
        const countAfter = idx.count;
        assert(countAfter >= 0, 'count is non-negative after partial batch failure');

        if (countAfter > 0) {
            const results = idx.search(goodVec, 1);
            assert(Array.isArray(results), 'search still returns array after partial batch failure');
        }

        idx.dispose();
    }

    // addBatch where ALL vectors have wrong dim
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        let threw = false;
        try {
            idx.addBatch([new Float32Array(DIM + 1), new Float32Array(DIM + 1)]);
        } catch (e) {
            threw = true;
        }
        assert(threw, 'addBatch with all wrong-dim vectors throws');
        assert(idx.count === 0, 'count still 0 after all-wrong-dim batch');
        idx.dispose();
    }

    // addBatch exceeding remaining capacity must not partially insert
    {
        const idx = await Pikelet.create({ ...DEFAULT_CONFIG, maxElements: 5 });
        const existingIds = idx.addBatch(Array.from({ length: 4 }, () => normalizedVec(DIM)));
        let threw = false;
        try {
            idx.addBatch([normalizedVec(DIM), normalizedVec(DIM)]);
        } catch (e) {
            threw = true;
        }
        assert(threw, 'addBatch exceeding remaining capacity throws');
        assert(idx.count === 4, 'count unchanged after remaining-capacity overflow');
        const results = idx.search(normalizedVec(DIM), 10);
        const ids = new Set(results.map(r => r.id));
        assert(Array.from(ids).every(id => existingIds.includes(id)), 'no new IDs leak after failed remaining-capacity batch');
        idx.dispose();
    }

    // If the C bulk_insert ever reports a short insert after JS prevalidation,
    // the wrapper must not present it as a normal capacity error.
    {
        const idx = await Pikelet.create({ ...DEFAULT_CONFIG, maxElements: 5 });
        const originalBulkInsert = idx._e._pikelet_bulk_insert;
        idx._e._pikelet_bulk_insert = (handle, dataPtr, n) => originalBulkInsert(handle, dataPtr, n - 1);
        try {
            let err = null;
            try {
                idx.addBatch([normalizedVec(DIM), normalizedVec(DIM)]);
            } catch (e) {
                err = e;
            }
            assert(err instanceof Pikelet.PikeletError, 'short bulk_insert throws PikeletError');
            assert(err?.code === Pikelet.PIKELET_ERROR_CODES.INTERNAL_INVARIANT, 'short bulk_insert reports INTERNAL_INVARIANT');
            assert(err?.details?.inserted === 1 && err?.details?.requested === 2, 'short bulk_insert reports partial insert details');
            assert(idx.count === 1, 'short bulk_insert leaves the actual inserted row visible');
            assert(idx.search(normalizedVec(DIM), 1).length === 1, 'index remains searchable after short bulk_insert invariant failure');
        } finally {
            idx._e._pikelet_bulk_insert = originalBulkInsert;
            idx.dispose();
        }
    }
}


async function testIdLifecycleAcrossCompaction() {
    section('ID lifecycle across compaction');

    const idx = await Pikelet.create(DEFAULT_CONFIG);
    const initialIds = idx.addBatch(Array.from({ length: 6 }, () => normalizedVec(DIM)));

    idx.delete(initialIds[1]);
    idx.delete(initialIds[4]);
    idx.compact();

    const afterFirstCompact = idx.add(normalizedVec(DIM));
    assert(afterFirstCompact === 6, 'new ID after first compact continues monotonically');

    idx.delete(initialIds[0]);
    idx.delete(afterFirstCompact);
    idx.compact();

    const afterSecondCompact = idx.add(normalizedVec(DIM));
    assert(afterSecondCompact === 7, 'new ID after second compact continues monotonically');

    const exported = idx.export();
    const idx2 = await Pikelet.create(DEFAULT_CONFIG);
    idx2.import(exported);
    const afterImport = idx2.add(normalizedVec(DIM));
    assert(afterImport === 8, 'new ID after import still continues monotonically');

    idx2.dispose();
    idx.dispose();
}


async function testZeroAndDegenerateVectors() {
    section('Degenerate vectors');

    // Zero vector — cosine is undefined; engine should not crash or corrupt
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const zero = zeroVec(DIM);
        try {
            idx.add(zero);
        } catch (e) {
            // Throwing is acceptable
        }
        // Regardless of what happened above, index must still function
        idx.add(normalizedVec(DIM));
        const results = idx.search(normalizedVec(DIM), 1);
        assert(Array.isArray(results), 'index still functional after zero-vector add attempt');
        idx.dispose();
    }

    // NaN vector — must not silently corrupt
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const nanVec = new Float32Array(DIM).fill(NaN);
        try {
            idx.add(nanVec);
        } catch (e) {
            // Throwing is acceptable
        }
        idx.add(normalizedVec(DIM));
        const results = idx.search(normalizedVec(DIM), 1);
        assert(Array.isArray(results), 'index functional after NaN vector add attempt');
        idx.dispose();
    }

    // Infinity vector — same contract as NaN
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const infVec = new Float32Array(DIM).fill(Infinity);
        try {
            idx.add(infVec);
        } catch (e) {
            // Throwing is acceptable
        }
        idx.add(normalizedVec(DIM));
        const results = idx.search(normalizedVec(DIM), 1);
        assert(Array.isArray(results), 'index functional after Infinity vector add attempt');
        idx.dispose();
    }

    // Zero query vector — search must not crash or hang
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        idx.addBatch(Array.from({ length: 10 }, () => normalizedVec(DIM)));
        try {
            const results = idx.search(zeroVec(DIM), 5);
            assert(Array.isArray(results), 'search with zero query returns array');
        } catch (e) {
            // Throwing is also acceptable
        }
        idx.dispose();
    }
}


async function testDeterminismIds() {
    section('Determinism — ID assignment');

    // Same vectors, same order → same IDs
    const vecs = Array.from({ length: 20 }, () => normalizedVec(DIM));

    const build = async () => {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const ids = idx.addBatch(vecs);
        idx.dispose();
        return ids;
    };

    const ids1 = await build();
    const ids2 = await build();

    const sameIds = ids1.every((id, i) => id === ids2[i]);
    assert(sameIds, 'IDs assigned are deterministic across identical builds');

    // IDs are sequential starting from 0
    assert(ids1[0] === 0, 'first assigned ID is 0');
    for (let i = 1; i < ids1.length; i++) {
        assert(ids1[i] === ids1[i - 1] + 1, `IDs are sequential (pos ${i})`);
    }

    // After compact(), surviving external IDs are stable
    {
        const idx = await Pikelet.create(DEFAULT_CONFIG);
        const ids = idx.addBatch(vecs);
        const survivorId = ids[10];

        idx.delete(ids[0]);
        idx.delete(ids[1]);
        idx.compact();

        const results = idx.search(vecs[10], 1);
        assert(
            results.length > 0 && results[0].id === survivorId,
            'external IDs stable after compact (survivor found by original id)'
        );

        idx.dispose();
    }
}


async function testMetricCorrectnessQuantized() {
    section('Metric correctness — quantized uint8');

    const DIM2 = 32;
    const QCONFIG = {
        dim:            DIM2,
        metric:         'cosine',
        maxElements:    20,
        M:              8,
        efConstruction: 100,
        efSearch:       50,
        quantized:      true,
    };

    const idx = await Pikelet.create(QCONFIG);

    // a = unit vector along axis 0
    // b = close to a (normalized)
    // c = opposite of a → max cosine distance
    const a = new Float32Array(DIM2); a[0] = 1.0;
    const b = new Float32Array(DIM2); b[0] = 0.99; b[1] = 0.14;
    let normB = 0; for (let i = 0; i < DIM2; i++) normB += b[i] * b[i]; normB = Math.sqrt(normB);
    for (let i = 0; i < DIM2; i++) b[i] /= normB;
    const c = new Float32Array(DIM2); c[0] = -1.0;

    const idA = idx.add(a);
    const idB = idx.add(b);
    const idC = idx.add(c);

    const results = idx.search(a, 3);
    assert(results.length === 3, 'quantized: search returns 3 results');
    assert(results[0].id === idA, 'quantized: self is nearest');
    assert(results[1].id === idB, 'quantized: near vector is rank 2');
    assert(results[2].id === idC, 'quantized: opposite vector is rank 3');

    assert(results[0].distance <= results[1].distance, 'quantized: dist[0] ≤ dist[1]');
    assert(results[1].distance <= results[2].distance, 'quantized: dist[1] ≤ dist[2]');
    assertNear(results[0].distance, 0.0, 0.05, 'quantized: self-distance ≈ 0');
    assert(results[2].distance > 1.5, `quantized: opposite distance > 1.5 (got ${results[2].distance.toFixed(3)})`);

    idx.dispose();
}

async function testRawEngineImportValidation() {
    section('Raw engine import validation');

    const cases = [
        { label: 'float', quantized: false, magic: 0x464C4831, graphBaseOffset: (count, dim) => 40 + count * dim * 4 },
        { label: 'quantized', quantized: true, magic: 0x49384831, graphBaseOffset: (count, dim) => 40 + count * 4 + count * 4 + count * dim },
    ];

    for (const testCase of cases) {
        const cfg = { ...DEFAULT_CONFIG, quantized: testCase.quantized, dim: 16, maxElements: 64 };
        const src = await Pikelet.create(cfg);
        const vecs = Array.from({ length: 8 }, () => normalizedVec(cfg.dim));
        src.addBatch(vecs);
        const exported = src.export();
        const raw = extractRawEngineBytes(exported);
        src.dispose();

        // The envelope and embedded raw header must agree. The raw metric is
        // what the backend actually restores, so accepting an outer cosine
        // header around an inner L2 snapshot would desynchronize the wrapper's
        // distance interpretation from engine behavior.
        const rawWithL2Metric = overwriteU32(raw, 32, 0);
        const v3WithL2Metric = new Uint8Array(exported);
        const v3View = new DataView(
            v3WithL2Metric.buffer,
            v3WithL2Metric.byteOffset,
            v3WithL2Metric.byteLength
        );
        const mappingCount = v3View.getUint32(24, true);
        const rawOffset = 32 + mappingCount * 8;
        v3View.setUint32(rawOffset + 32, 0, true);

        for (const mismatch of [
            { label: 'v3', bytes: v3WithL2Metric },
            { label: 'v2', bytes: wrapV2Envelope(rawWithL2Metric, cfg.dim, testCase.quantized, 1) },
            { label: 'v1', bytes: wrapV1Envelope(rawWithL2Metric, cfg.dim, testCase.quantized, 1) },
        ]) {
            const idx = await Pikelet.create(cfg);
            assertThrows(
                () => idx.import(mismatch.bytes),
                `${testCase.label}/${mismatch.label}: rejects envelope/raw metric disagreement`
            );
            assert(idx.count === 0, `${testCase.label}/${mismatch.label}: mismatch rejection is atomic`);
            idx.dispose();
        }

        // Header: count at 12, entry point at 16, max level at 20, M at 24, M0 at 28, metric at 32
        {
            const idx = await Pikelet.create(cfg);
            const bad = overwriteU32(raw, 16, 9999);
            let msg = '';
            try { idx.import(bad); } catch (e) { msg = e.message; }
            assert(msg.includes('Import failed'), `${testCase.label}: rejects invalid entry point`);
            idx.dispose();
        }

        {
            const idx = await Pikelet.create(cfg);
            const bad = overwriteU32(raw, 32, 99);
            let msg = '';
            try { idx.import(bad); } catch (e) { msg = e.message; }
            assert(msg.includes('Import failed'), `${testCase.label}: rejects invalid metric enum`);
            idx.dispose();
        }

        {
            const idx = await Pikelet.create(cfg);
            const bad = overwriteU32(raw, 24, 1);
            let msg = '';
            try { idx.import(bad); } catch (e) { msg = e.message; }
            assert(msg.includes('Import failed'), `${testCase.label}: rejects invalid M <= 1`);
            idx.dispose();
        }

        {
            const idx = await Pikelet.create(cfg);
            const serializedM = new DataView(raw.buffer, raw.byteOffset, raw.byteLength).getUint32(24, true);
            const bad = overwriteU32(raw, 28, serializedM * 2 - 1);
            let msg = '';
            try { idx.import(bad); } catch (e) { msg = e.message; }
            assert(msg.includes('Import failed'), `${testCase.label}: rejects M0 that is not exactly 2*M`);
            idx.dispose();
        }

        {
            const idx = await Pikelet.create(cfg);
            const graphOffset = testCase.graphBaseOffset(8, cfg.dim);
            const bad = overwriteU32(raw, graphOffset + 4, 9999);
            let msg = '';
            try { idx.import(bad); } catch (e) { msg = e.message; }
            assert(msg.includes('Import failed'), `${testCase.label}: rejects invalid neighbor id`);
            idx.dispose();
        }
    }
}

async function testGhostEntryPointSearch() {
    section('Ghost entry point keeps full top-k');

    // Regression: the layer search used to push the graph's entry node into
    // the result set unconditionally, so deleting the entry point silently
    // returned k-1 results. Deleting every vector in turn guarantees the
    // entry point (whichever node the level RNG picked) is covered.
    const N = 40;
    const K = 10;
    const dim = 16;

    for (const quantized of [false, true]) {
        const label = quantized ? 'uint8' : 'float';
        const rng = mulberry32(1234);
        const vecs = Array.from({ length: N }, () => {
            const v = new Float32Array(dim);
            for (let d = 0; d < dim; d++) v[d] = rng();
            return v;
        });

        const src = await Pikelet.create({
            dim, maxElements: N, metric: 'l2', quantized,
            M: 8, efConstruction: 50, efSearch: 50,
        });
        const ids = src.addBatch(vecs);
        const snapshot = src.export();
        src.dispose();

        let fullK = 0;
        let ghostLeaks = 0;
        for (let i = 0; i < N; i++) {
            const idx = await Pikelet.create({
                dim, maxElements: N, metric: 'l2', quantized,
                M: 8, efConstruction: 50, efSearch: 50,
            });
            idx.import(snapshot);
            idx.delete(ids[i]);
            const results = idx.search(vecs[i], K);
            if (results.length === K) fullK++;
            if (results.some(r => r.id === ids[i])) ghostLeaks++;
            idx.dispose();
        }

        assert(fullK === N, `${label}: search returns full k=${K} results for every deleted vector (got ${fullK}/${N})`);
        assert(ghostLeaks === 0, `${label}: deleted vector never appears in results`);
    }
}

async function testGhostsRemainNavigable() {
    section('Ghost nodes remain navigable');

    const dim = 8;
    const N = 80;
    const liveStride = 5;
    const makeVec = (i) => {
        const v = new Float32Array(dim);
        v[0] = i / N;
        v[1] = 1 - i / N;
        for (let d = 2; d < dim; d++) v[d] = ((i * (d + 3)) % 17) / 17;
        return v;
    };

    for (const quantized of [false, true]) {
        const label = quantized ? 'uint8' : 'float';
        const idx = await Pikelet.create({
            dim,
            maxElements: N,
            metric: 'l2',
            quantized,
            M: 4,
            efConstruction: 10,
            efSearch: 50,
        });
        const vecs = Array.from({ length: N }, (_, i) => makeVec(i));
        const ids = idx.addBatch(vecs);
        const live = new Set();

        for (let i = 0; i < N; i++) {
            if (i % liveStride === 0) live.add(ids[i]);
            else idx.delete(ids[i]);
        }

        let exactTop1 = 0;
        let filteredExactTop1 = 0;
        let ghostLeaks = 0;
        for (let i = 0; i < N; i += liveStride) {
            const results = idx.search(vecs[i], 10);
            const filtered = idx.searchFiltered(vecs[i], 10, live);
            if (results[0]?.id === ids[i]) exactTop1++;
            if (filtered[0]?.id === ids[i]) filteredExactTop1++;
            if (results.some(r => !live.has(r.id)) || filtered.some(r => !live.has(r.id))) ghostLeaks++;
        }

        assert(idx.ghostCount === N - live.size, `${label}: fixture has expected ghost count`);
        assert(exactTop1 === live.size, `${label}: search navigates through ghosts to every live self-match`);
        assert(filteredExactTop1 === live.size, `${label}: filtered search navigates through ghosts to every live self-match`);
        assert(ghostLeaks === 0, `${label}: ghost nodes never appear in search results`);
        idx.dispose();
    }
}

async function testDeleteChurnRegression() {
    section('Delete churn regression');

    const scenarios = [
        { label: 'float-cosine', quantized: false, metric: 'cosine' },
        { label: 'u8-cosine', quantized: true, metric: 'cosine' },
    ];

    for (const scenario of scenarios) {
        const cfg = {
            ...DEFAULT_CONFIG,
            dim: 64,
            maxElements: 256,
            quantized: scenario.quantized,
            metric: scenario.metric,
            efSearch: 120,
        };
        const idx = await Pikelet.create(cfg);
        const corpus = Array.from({ length: 120 }, () => normalizedVec(cfg.dim));
        const ids = idx.addBatch(corpus);

        let live = new Set(ids);
        const vectorsById = new Map(ids.map((id, i) => [id, corpus[i]]));
        for (let round = 0; round < 4; round++) {
            const toDelete = [];
            for (let i = round * 10; i < round * 10 + 10; i++) toDelete.push(ids[i]);
            for (const id of toDelete) {
                idx.delete(id);
                live.delete(id);
            }
            assert(idx.ghostCount === 10, `${scenario.label}: ghost count tracks deletes in round ${round + 1}`);

            const warmQuery = corpus[119 - round];
            const beforeCompact = idx.search(warmQuery, 10);
            assert(beforeCompact.every(r => live.has(r.id)), `${scenario.label}: no deleted IDs leak before compact in round ${round + 1}`);

            idx.compact();
            assert(idx.ghostCount === 0, `${scenario.label}: compact clears ghosts in round ${round + 1}`);
            const afterCompact = idx.search(warmQuery, 10);
            assert(afterCompact.every(r => live.has(r.id)), `${scenario.label}: no deleted IDs leak after compact in round ${round + 1}`);

            const newVecs = Array.from({ length: 5 }, () => normalizedVec(cfg.dim));
            const newIds = idx.addBatch(newVecs);
            for (let i = 0; i < newIds.length; i++) {
                live.add(newIds[i]);
                vectorsById.set(newIds[i], newVecs[i]);
            }
        }

        const probeIds = Array.from(live).slice(0, 12);
        let exactHits = 0;
        for (const id of probeIds) {
            const vec = vectorsById.get(id);
            const results = idx.search(vec, 1);
            if (results.length > 0 && results[0].id === id) exactHits++;
        }
        assert(exactHits >= 8, `${scenario.label}: retains acceptable self-recall after churn (got ${exactHits}/${probeIds.length})`);
        idx.dispose();
    }
}

async function testGoldenSnapshotCompatibility() {
    section('Golden snapshot compatibility');

    const scenarios = [
        { label: 'uint8', quantized: true, base64: goldenSnapshots.quantizedBase64 },
        { label: 'float32', quantized: false, base64: goldenSnapshots.float32Base64 },
    ];

    const query = new Float32Array([1, 0, 0, 0]);

    for (const scenario of scenarios) {
        const exportedV3 = decodeBase64Bytes(scenario.base64);
        const raw = extractRawEngineBytes(exportedV3);
        const exportedV2 = wrapV2Envelope(raw, 4, scenario.quantized, 1);
        const exportedV1 = wrapV1Envelope(raw, 4, scenario.quantized, 1);

        for (const variant of [
            { label: 'v3', bytes: exportedV3 },
            { label: 'v2', bytes: exportedV2 },
            { label: 'v1', bytes: exportedV1 },
            { label: 'raw', bytes: raw },
        ]) {
            const idx = await Pikelet.create({
                dim: 4,
                maxElements: 16,
                metric: 'cosine',
                quantized: scenario.quantized,
                M: 8,
                efConstruction: 50,
                efSearch: 50,
            });

            idx.import(variant.bytes);

            assert(idx.count === 3, `${scenario.label}/${variant.label}: imports golden snapshot with expected live count`);

            const top = idx.search(query, 3);
            assert(top.length === 3, `${scenario.label}/${variant.label}: search returns expected count`);
            assert(top[0].id === 0, `${scenario.label}/${variant.label}: leading external ID preserved`);
            assert(top.every(r => Number.isFinite(r.distance)), `${scenario.label}/${variant.label}: distances are finite`);

            const reexported = idx.export();
            const idx2 = await Pikelet.create({
                dim: 4,
                maxElements: 16,
                metric: 'cosine',
                quantized: scenario.quantized,
                M: 8,
                efConstruction: 50,
                efSearch: 50,
            });
            idx2.import(reexported);

            assert(idx2.count === idx.count, `${scenario.label}/${variant.label}: export/import preserves count`);
            const roundTrip = idx2.search(query, 3);
            assert(roundTrip.length === top.length, `${scenario.label}/${variant.label}: round-trip preserves top-k length`);
            assert(roundTrip[0].id === top[0].id, `${scenario.label}/${variant.label}: round-trip preserves top-1 ID`);

            const nextId = idx2.add(new Float32Array([0, 0, 0, 1]));
            const expectedNextId = variant.label === 'v3' ? 4 : 3;
            assert(nextId === expectedNextId, `${scenario.label}/${variant.label}: next external ID follows expected ${variant.label} import semantics`);

            idx2.dispose();
            idx.dispose();
        }
    }
}


async function testNonFiniteRejection() {
    section('Non-finite vector rejection');

    const dim = 16;
    for (const quantized of [false, true]) {
        const label = quantized ? 'uint8' : 'float32';
        const idx = await Pikelet.create({ dim, maxElements: 64, metric: 'cosine', quantized });

        idx.add(normalizedVec(dim));

        const nanVec = new Float32Array(dim);
        nanVec.fill(NaN);
        const infVec = new Float32Array(dim);
        infVec.fill(Infinity);
        const negInfVec = new Float32Array(dim);
        negInfVec.fill(-Infinity);
        const partialNaN = normalizedVec(dim);
        partialNaN[7] = NaN;

        assertThrows(() => idx.add(nanVec), `${label}: add() rejects all-NaN vector`);
        assertThrows(() => idx.add(infVec), `${label}: add() rejects all-Infinity vector`);
        assertThrows(() => idx.add(negInfVec), `${label}: add() rejects all-negative-Infinity vector`);
        assertThrows(() => idx.add(partialNaN), `${label}: add() rejects vector with single NaN`);

        assertThrows(() => idx.search(nanVec, 1), `${label}: search() rejects NaN query`);
        assertThrows(() => idx.search(infVec, 1), `${label}: search() rejects Infinity query`);

        assertThrows(() => idx.addBatch([nanVec]), `${label}: addBatch() rejects NaN vector`);
        assertThrows(() => idx.addBatch([normalizedVec(dim), infVec]), `${label}: addBatch() rejects batch with Infinity at index 1`);

        assert(idx.count === 1, `${label}: count unchanged after rejected inserts`);

        idx.dispose();
    }
}

async function testCosineNormOverflowRegression() {
    section('Cosine norm overflow regression');

    const dim = 4;
    const huge = new Float32Array(dim).fill(1e19);
    const zero = new Float32Array(dim);

    for (const quantized of [false, true]) {
        const label = quantized ? 'uint8' : 'float32';
        const idx = await Pikelet.create({ dim, maxElements: 8, metric: 'cosine', quantized });

        const id = idx.add(huge);
        assert(id === 0, `${label}: huge finite vector inserts`);
        const self = idx.search(huge, 1);
        assert(self.length === 1 && self[0].id === id, `${label}: huge finite vector self-search returns itself`);
        assertNear(self[0].distance, 0, 0.01, `${label}: huge finite vector does not normalize to zero`);

        assertThrows(() => idx.add(zero), `${label}: zero cosine vector is rejected`);
        assert(idx.count === 1, `${label}: rejected zero vector does not mutate count`);
        assertThrows(() => idx.search(zero, 1), `${label}: zero cosine query is rejected`);
        assertThrows(() => idx.addBatch([normalizedVec(dim), zero]), `${label}: zero vector in batch is rejected`);
        assert(idx.count === 1, `${label}: rejected zero-vector batch does not mutate count`);

        idx.dispose();
    }
}

async function testCompactEntryPointRecovery() {
    section('Compact entry point recovery (all level 0)');

    // Small index with low M — all vectors will be level 0 with high probability.
    // Use dim=4 and M=4 to minimize the chance of any vector getting level > 0.
    // With M=4, P(level >= 1) = 1/4 = 25% per vector, so 3 vectors gives
    // ~42% chance all are level 0. We use a fixed seed via deterministic vectors.
    for (const quantized of [false, true]) {
        const label = quantized ? 'uint8' : 'float32';
        const idx = await Pikelet.create({
            dim: 4, maxElements: 32, metric: 'cosine', quantized,
            M: 16, efConstruction: 50, efSearch: 50,
        });

        // Insert vectors. ID 0 is always the entry point (first insert).
        const v0 = new Float32Array([1, 0, 0, 0]);
        const v1 = new Float32Array([0, 1, 0, 0]);
        const v2 = new Float32Array([0, 0, 1, 0]);
        const id0 = idx.add(v0);
        const id1 = idx.add(v1);
        const id2 = idx.add(v2);

        // Delete the entry point (first vector)
        idx.delete(id0);
        idx.compact();
        assert(idx.count === 2, `${label}: count is 2 after deleting entry point and compacting`);

        // Export must succeed (entry point must be valid)
        const snapshot = idx.export();
        assert(snapshot.length > 0, `${label}: export succeeds after entry point deletion + compact`);

        // Reimport must succeed
        const idx2 = await Pikelet.create({
            dim: 4, maxElements: 32, metric: 'cosine', quantized,
            M: 16, efConstruction: 50, efSearch: 50,
        });
        idx2.import(snapshot);
        assert(idx2.count === 2, `${label}: reimport preserves count`);

        // Search must work
        const results = idx2.search(v1, 2);
        assert(results.length === 2, `${label}: search returns results after reimport`);

        idx2.dispose();
        idx.dispose();
    }
}

async function testSearchFiltered() {
    section('Filtered search');

    for (const quantized of [false, true]) {
        const label = quantized ? 'uint8' : 'float32';
        const dim = 32;
        const idx = await Pikelet.create({
            dim, maxElements: 200, metric: 'cosine', quantized,
        });

        // Insert 50 vectors with deterministic values
        const vecs = [];
        const ids = [];
        for (let i = 0; i < 50; i++) {
            const v = new Float32Array(dim);
            for (let d = 0; d < dim; d++) v[d] = Math.sin(i * 17 + d * 7);
            let norm = 0;
            for (let d = 0; d < dim; d++) norm += v[d] * v[d];
            norm = Math.sqrt(norm);
            for (let d = 0; d < dim; d++) v[d] /= norm;
            vecs.push(v);
            ids.push(idx.add(v));
        }

        const query = vecs[0];

        // Unfiltered baseline
        const baseline = idx.search(query, 10);
        assert(baseline.length === 10, `${label}: unfiltered returns 10`);

        // Filter to even IDs only
        const evenIds = new Set(ids.filter(id => id % 2 === 0));
        const evenResults = idx.searchFiltered(query, 10, evenIds);
        assert(evenResults.length > 0, `${label}: even filter returns results`);
        assert(evenResults.every(r => r.id % 2 === 0), `${label}: all results have even IDs`);

        // Filter to single ID
        const singleResult = idx.searchFiltered(query, 5, new Set([ids[0]]));
        assert(singleResult.length === 1, `${label}: single-ID filter returns 1`);
        assert(singleResult[0].id === ids[0], `${label}: single-ID filter returns correct ID`);

        // Empty filter
        const emptyResult = idx.searchFiltered(query, 5, new Set());
        assert(emptyResult.length === 0, `${label}: empty filter returns 0`);

        // Allow all — should match unfiltered
        const allIds = new Set(ids);
        const allResult = idx.searchFiltered(query, 10, allIds);
        assert(allResult.length === baseline.length, `${label}: allow-all matches unfiltered count`);
        assert(allResult[0].id === baseline[0].id, `${label}: allow-all top result matches unfiltered`);

        // Filter with deleted vectors — deleted IDs should not appear
        idx.delete(ids[2]);
        idx.delete(ids[4]);
        const withDeleted = new Set(ids);
        const afterDelete = idx.searchFiltered(query, 10, withDeleted);
        assert(afterDelete.every(r => r.id !== ids[2] && r.id !== ids[4]),
            `${label}: deleted IDs excluded from filtered results`);

        // Highly selective filter (only 2 IDs) — tests iterative deepening
        const tinySet = new Set([ids[10], ids[20]]);
        const tinyResult = idx.searchFiltered(query, 2, tinySet);
        assert(tinyResult.length === 2, `${label}: selective filter finds both IDs`);
        assert(tinyResult.every(r => tinySet.has(r.id)), `${label}: selective filter returns only allowed IDs`);

        // Distances should be monotonically non-decreasing
        for (let i = 1; i < evenResults.length; i++) {
            assert(evenResults[i].distance >= evenResults[i - 1].distance,
                `${label}: filtered distances are sorted`);
        }

        idx.dispose();
    }
}

async function testHeldOutRecallOracle() {
    section('Held-out recall vs brute-force oracle');

    const oracle = searchOracles.clusteredCosine32;
    for (const scenario of [
        { label: 'float32', quantized: false },
        { label: 'uint8', quantized: true },
    ]) {
        const probe = await evaluateHeldOutOracle(scenario, oracle);
        const expected = oracle.recallBaseline[scenario.label].beforeCompact;
        assertNear(
            probe.avgRecallBeforeCompact,
            expected,
            1e-9,
            `${scenario.label}: held-out recall matches recorded brute-force baseline`
        );
    }
}

async function testHeldOutRecallAfterCompact() {
    section('Held-out recall after compact vs brute-force oracle');

    const oracle = searchOracles.clusteredCosine32;
    for (const scenario of [
        { label: 'float32', quantized: false },
        { label: 'uint8', quantized: true },
    ]) {
        const probe = await evaluateHeldOutOracle(scenario, oracle);
        const expected = oracle.recallBaseline[scenario.label].afterCompact;
        assertNear(
            probe.avgRecallAfterCompact,
            expected,
            1e-9,
            `${scenario.label}: post-compact held-out recall matches recorded brute-force baseline`
        );
    }
}

async function testFilteredHeldOutRecallOracle() {
    section('Held-out filtered recall vs brute-force oracle');

    const oracle = searchOracles.clusteredCosine32;
    for (const scenario of [
        { label: 'float32', quantized: false },
        { label: 'uint8', quantized: true },
    ]) {
        const probe = await evaluateHeldOutOracle(scenario, oracle);
        for (const filterSpec of oracle.filteredSpecs) {
            const expected = oracle.filteredRecallBaseline[scenario.label][filterSpec.label];
            assertNear(
                probe.filteredRecallBeforeCompact[filterSpec.label],
                expected,
                1e-9,
                `${scenario.label}: ${filterSpec.label} filtered held-out recall matches recorded brute-force baseline`
            );
        }
    }
}

async function testSearchOutputGoldenOracle() {
    section('Search output golden oracle');

    const oracle = searchOracles.clusteredCosine32;
    for (const scenario of [
        { label: 'float32', quantized: false },
        { label: 'uint8', quantized: true },
    ]) {
        const probe = await evaluateHeldOutOracle(scenario, oracle);
        assertSearchRowsEqualWithTolerance(
            probe.goldenRows,
            oracle.searchGolden[scenario.label],
            2e-6,
            `${scenario.label}: fixed-query search output`
        );
    }
}

async function testSearchAndSerializationDeterminismOracle() {
    section('Determinism — serialized graph and fixed queries');

    const oracle = searchOracles.clusteredCosine32;
    for (const scenario of [
        { label: 'float32', quantized: false },
        { label: 'uint8', quantized: true },
    ]) {
        const probe = await evaluateHeldOutOracle(scenario, oracle);
        assert(
            probe.exported.length === probe.exported2.length,
            `${scenario.label}: identical builds export equal-length snapshots`
        );
        const sameBytes = probe.exported.length === probe.exported2.length
            && probe.exported.every((byte, i) => byte === probe.exported2[i]);
        assert(sameBytes, `${scenario.label}: identical builds export identical snapshots`);
        assertSearchRowsEqualWithTolerance(
            probe.goldenRows2,
            probe.goldenRows,
            1e-9,
            `${scenario.label}: identical rebuilds preserve fixed-query outputs`
        );
    }
}

async function testCompactReconnectsIsolatedSurvivors() {
    section('Compact reconnects isolated survivors');

    const makeRng = (seed) => {
        let state = seed >>> 0;
        return () => {
            state = (state * 1664525 + 1013904223) >>> 0;
            return state / 0x100000000;
        };
    };

    const runCase = async (quantized) => {
        const rand = makeRng(30);
        const config = {
            dim: 4,
            metric: 'l2',
            maxElements: 90,
            M: 4,
            efConstruction: 64,
            efSearch: 200,
            quantized,
        };

        const idx = await Pikelet.create(config);
        const vecs = [];
        const ids = [];
        for (let i = 0; i < 80; i++) {
            const vec = new Float32Array([
                rand() * 10,
                rand() * 10,
                rand() * 10,
                rand() * 10,
            ]);
            vecs.push(vec);
            ids.push(idx.add(vec));
        }

        const deleted = new Set();
        for (let i = 0; i < 80; i++) {
            if (rand() < 0.65) deleted.add(i);
        }

        const survivors = [];
        for (let i = 0; i < 80; i++) {
            if (!deleted.has(i)) survivors.push(i);
        }

        for (const i of survivors) {
            const before = idx.search(vecs[i], 1);
            assert(before.length > 0 && before[0].id === ids[i], `${quantized ? 'quantized' : 'float'}: survivor ${i} finds itself before delete`);
        }

        for (const i of deleted) idx.delete(ids[i]);

        for (const i of survivors) {
            const preCompact = idx.search(vecs[i], 1);
            assert(preCompact.length > 0 && preCompact[0].id === ids[i], `${quantized ? 'quantized' : 'float'}: survivor ${i} finds itself before compact`);
        }

        idx.compact();

        assert(idx.count === survivors.length, `${quantized ? 'quantized' : 'float'}: rebuild compaction retains every survivor`);
        assert(idx.deletedCount === 0, `${quantized ? 'quantized' : 'float'}: rebuild compaction clears deleted state`);
        const survivorIds = new Set(survivors.map(i => ids[i]));

        for (const i of survivors) {
            const postCompact = idx.search(vecs[i], 1);
            assert(
                postCompact.length > 0 && postCompact[0].id === ids[i],
                `${quantized ? 'quantized' : 'float'}: survivor ${i} still finds itself after compact`
            );
        }

        const breadth = Math.min(10, survivors.length);
        const results = idx.search(vecs[survivors[0]], breadth);
        assert(results.length === breadth, `${quantized ? 'quantized' : 'float'}: rebuild compaction returns a full result set`);
        assert(results.every(result => survivorIds.has(result.id)), `${quantized ? 'quantized' : 'float'}: rebuild compaction never leaks deleted IDs`);

        const appended = new Float32Array([20, 20, 20, 20]);
        const appendedId = idx.add(appended);
        assert(appendedId === 80, `${quantized ? 'quantized' : 'float'}: external ID allocation remains monotonic after rebuild compaction`);
        const appendedResult = idx.search(appended, 1);
        assert(appendedResult.length === 1 && appendedResult[0].id === appendedId,
            `${quantized ? 'quantized' : 'float'}: index accepts searchable inserts after rebuild compaction`);

        idx.dispose();
    };

    await runCase(false);
    await runCase(true);
}

async function testConvenienceLoaders() {
    section('Convenience loaders');

    {
        const rows = [
            unitVec(4, 0),
            unitVec(4, 1),
            unitVec(4, 2),
        ];
        const { index, ids, idMap } = await Pikelet.fromVectors(rows, {
            metric: 'cosine',
            quantized: false,
        });
        assert(ids.length === 3, 'fromVectors() returns one Pikelet ID per raw vector');
        assert(index.dim === 4, 'fromVectors() infers dim from raw vectors');
        assert(index.count === 3, 'fromVectors() populates the index from raw vectors');
        assert(idMap.size === 0, 'fromVectors() leaves idMap empty for raw vectors');
        const results = index.search(unitVec(4, 0), 1);
        assert(results.length === 1 && results[0].id === ids[0], 'fromVectors() search works for raw vectors');
        index.dispose();
    }

    {
        const rows = [
            { id: 'doc-a', vector: unitVec(4, 0) },
            { id: 'doc-b', vector: unitVec(4, 1) },
        ];
        const { index, ids, idMap } = await Pikelet.fromVectors(rows, {
            metric: 'cosine',
            quantized: false,
        });
        assert(ids.length === 2, 'fromVectors() accepts { id, vector } records');
        assert(idMap.get(ids[0]) === 'doc-a', 'fromVectors() maps first record back to caller ID');
        assert(idMap.get(ids[1]) === 'doc-b', 'fromVectors() maps second record back to caller ID');
        index.dispose();
    }

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-loaders-'));
    try {
        {
            const jsonPath = path.join(tmpDir, 'vectors.json');
            fs.writeFileSync(jsonPath, JSON.stringify([
                { docId: 'alpha', embedding: Array.from(unitVec(4, 0)) },
                { docId: 'beta', embedding: Array.from(unitVec(4, 1)) },
            ]));

            const { index, ids, idMap } = await Pikelet.loadJsonFile(jsonPath, {
                metric: 'cosine',
                quantized: false,
                vectorKey: 'embedding',
                idKey: 'docId',
            });
            assert(index.count === 2, 'loadJsonFile() loads JSON arrays into an index');
            assert(idMap.get(ids[0]) === 'alpha', 'loadJsonFile() preserves custom ID field for first row');
            assert(idMap.get(ids[1]) === 'beta', 'loadJsonFile() preserves custom ID field for second row');
            index.dispose();
        }

        {
            const jsonlPath = path.join(tmpDir, 'vectors.jsonl');
            fs.writeFileSync(jsonlPath, [
                JSON.stringify({ id: 'gamma', vector: Array.from(unitVec(4, 2)) }),
                JSON.stringify({ id: 'delta', vector: Array.from(unitVec(4, 3)) }),
            ].join('\n'));

            const { index, ids, idMap } = await Pikelet.loadJsonFile(jsonlPath, {
                metric: 'cosine',
                quantized: false,
            });
            assert(index.count === 2, 'loadJsonFile() loads JSONL into an index');
            assert(idMap.get(ids[0]) === 'gamma', 'loadJsonFile() preserves JSONL ID for first row');
            assert(idMap.get(ids[1]) === 'delta', 'loadJsonFile() preserves JSONL ID for second row');
            index.dispose();
        }

        {
            const txtPath = path.join(tmpDir, 'vectors.txt');
            fs.writeFileSync(txtPath, JSON.stringify([
                { id: 'epsilon', vector: Array.from(unitVec(4, 0)) },
            ]));

            await assertThrowsAsync(
                () => Pikelet.loadJsonFile(txtPath, {
                    metric: 'cosine',
                    quantized: false,
                }),
                'loadJsonFile() rejects unsupported file extensions unless format is explicit'
            );
        }

        {
            const jsonPath = path.join(tmpDir, 'oversized.json');
            fs.writeFileSync(jsonPath, JSON.stringify([
                { id: 'zeta', vector: Array.from(unitVec(4, 0)) },
            ]));

            await assertThrowsAsync(
                () => Pikelet.loadJsonFile(jsonPath, {
                    metric: 'cosine',
                    quantized: false,
                    maxFileBytes: 8,
                }),
                'loadJsonFile() rejects files that exceed maxFileBytes'
            );
        }

        {
            const src = await Pikelet.create({
                dim: 4,
                metric: 'cosine',
                quantized: false,
                maxElements: 8,
            });
            const vec = unitVec(4, 0);
            const id = src.add(vec);
            const snapshotPath = path.join(tmpDir, 'index.pnck');
            fs.writeFileSync(snapshotPath, src.export());
            src.dispose();

            const restored = await Pikelet.loadSnapshotFile(snapshotPath, {
                dim: 4,
                metric: 'cosine',
                quantized: false,
                maxElements: 8,
            });
            const results = restored.search(vec, 1);
            assert(restored.count === 1, 'loadSnapshotFile() restores snapshot count from disk');
            assert(results.length === 1 && results[0].id === id, 'loadSnapshotFile() restores searchable IDs');
            restored.dispose();
        }

        {
            const badSnapshotPath = path.join(tmpDir, 'bad.pnck');
            fs.writeFileSync(badSnapshotPath, Buffer.from('not-a-pancake-snapshot'));

            await assertThrowsAsync(
                () => Pikelet.loadSnapshotFile(badSnapshotPath, {
                    dim: 4,
                    metric: 'cosine',
                    quantized: false,
                    maxElements: 8,
                }),
                'loadSnapshotFile() rejects unsupported snapshot magic before import'
            );
        }
    } finally {
        fs.rmSync(tmpDir, { recursive: true, force: true });
    }
}

async function testSearchFilteredInputContract() {
    section('Filtered search input contract');

    const idx = await Pikelet.create(DEFAULT_CONFIG);
    idx.addBatch([
        normalizedVec(DIM),
        normalizedVec(DIM),
        normalizedVec(DIM),
    ]);

    const query = normalizedVec(DIM);
    const ok = idx.searchFiltered(query, 2, new Set([0, 1]));
    assert(Array.isArray(ok), 'searchFiltered() accepts Set<number>');

    const oversized = idx.searchFiltered(query, 0x80000001, new Set([0, 1, 2]));
    assert(oversized.length === 3, 'searchFiltered() safely bounds k above signed i32 range');
    assert(idx.search(query, 1).length === 1, 'index remains usable after oversized filtered search');

    assertThrows(
        () => idx.searchFiltered(query, Number.MAX_SAFE_INTEGER + 1, new Set([0, 1])),
        'searchFiltered() rejects integers that cannot be represented safely'
    );

    assertThrows(
        () => idx.searchFiltered(query, 2, [0, 1]),
        'searchFiltered() rejects array allowedIds'
    );

    assertThrows(
        () => idx.searchFiltered(query, 2, '01'),
        'searchFiltered() rejects string allowedIds'
    );

    assertThrows(
        () => idx.searchFiltered(query, 2, { size: 2, values: () => [0, 1][Symbol.iterator]() }),
        'searchFiltered() rejects non-Set set-like objects'
    );

    idx.dispose();
}

async function testDemoVectorGenerator() {
    section('Deterministic clustered demo vectors');

    const generatorUrl = pathToFileURL(path.join(process.cwd(), 'scripts', 'make-demo-vectors.mjs')).href;
    const { generateClusteredVectors } = await import(generatorUrl);
    const options = { count: 64, dims: 16, clusters: 4, spread: 0.03, seed: 12345 };
    const first = generateClusteredVectors(options);
    const second = generateClusteredVectors(options);

    assert(first.length === options.count * options.dims, 'demo generator emits count * dims float values');
    assert(first.every((value, i) => value === second[i]), 'demo generator is byte-deterministic for a fixed seed');

    for (let row = 0; row < options.count; row++) {
        let normSq = 0;
        const offset = row * options.dims;
        for (let d = 0; d < options.dims; d++) normSq += first[offset + d] * first[offset + d];
        assertNear(Math.sqrt(normSq), 1, 1e-5, `demo vector ${row} is unit-normalized`);
    }

    const dot = (rowA, rowB) => {
        let value = 0;
        for (let d = 0; d < options.dims; d++) {
            value += first[rowA * options.dims + d] * first[rowB * options.dims + d];
        }
        return value;
    };
    assert(dot(0, 4) > dot(0, 1), 'same-cluster demo vectors are closer than different-cluster vectors');
}

async function testPikeletErrorContract() {
    section('PikeletError contract');

    const expectCode = (fn, code, message) => {
        try {
            fn();
            assert(false, `${message} (expected ${code})`);
        } catch (error) {
            assert(error instanceof Pikelet.PikeletError, `${message} throws PikeletError`);
            assert(error.code === code, `${message} has ${code} code`);
        }
    };
    const expectCodeAsync = async (fn, code, message) => {
        try {
            await fn();
            assert(false, `${message} (expected ${code})`);
        } catch (error) {
            assert(error instanceof Pikelet.PikeletError, `${message} throws PikeletError`);
            assert(error.code === code, `${message} has ${code} code`);
        }
    };

    assert(typeof Pikelet.PikeletError === 'function', 'CJS API exposes PikeletError');
    assert(Pikelet.PIKELET_ERROR_CODES.INDEX_DISPOSED === 'INDEX_DISPOSED',
        'CJS API exposes stable error codes');

    const esm = await import(pathToFileURL(path.join(process.cwd(), 'pikelet.node.mjs')).href);
    assert(esm.PikeletError === Pikelet.PikeletError, 'Node ESM exposes the same PikeletError class');

    await expectCodeAsync(
        () => Pikelet.create({ dim: 0 }),
        Pikelet.PIKELET_ERROR_CODES.INVALID_ARGUMENT,
        'invalid construction options'
    );

    const full = await Pikelet.create({ dim: 4, maxElements: 1, metric: 'l2', quantized: false });
    expectCode(
        () => full.add(new Float32Array(3)),
        Pikelet.PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
        'wrong-length vector'
    );
    expectCode(
        () => full.add(new Float32Array([NaN, 0, 0, 0])),
        Pikelet.PIKELET_ERROR_CODES.INVALID_VECTOR,
        'non-finite vector'
    );
    full.add(new Float32Array(4));
    expectCode(
        () => full.add(new Float32Array(4)),
        Pikelet.PIKELET_ERROR_CODES.INDEX_FULL,
        'capacity overflow'
    );
    full.delete(0);
    expectCode(
        () => full.export(),
        Pikelet.PIKELET_ERROR_CODES.COMPACTION_REQUIRED,
        'snapshot export with deleted nodes'
    );
    full.dispose();

    const source = await Pikelet.create({ dim: 4, maxElements: 2, metric: 'l2', quantized: false });
    source.addBatch([unitVec(4, 0), unitVec(4, 1)]);
    const snapshot = source.export();
    const mismatched = await Pikelet.create({ dim: 4, maxElements: 2, metric: 'cosine', quantized: false });
    expectCode(
        () => mismatched.import(snapshot),
        Pikelet.PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
        'snapshot configuration mismatch'
    );
    const undersized = await Pikelet.create({ dim: 4, maxElements: 1, metric: 'l2', quantized: false });
    expectCode(
        () => undersized.import(snapshot),
        Pikelet.PIKELET_ERROR_CODES.SNAPSHOT_CAPACITY_EXCEEDED,
        'snapshot capacity overflow'
    );
    source.dispose();
    mismatched.dispose();
    undersized.dispose();

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-errors-'));
    try {
        const invalidJson = path.join(tmpDir, 'invalid.json');
        fs.writeFileSync(invalidJson, '{');
        await expectCodeAsync(
            () => Pikelet.loadJsonFile(invalidJson),
            Pikelet.PIKELET_ERROR_CODES.PARSE_FAILED,
            'malformed JSON file'
        );
        await expectCodeAsync(
            () => Pikelet.loadJsonFile(path.join(tmpDir, 'missing.json')),
            Pikelet.PIKELET_ERROR_CODES.FILE_IO_FAILED,
            'missing JSON file'
        );
    } finally {
        fs.rmSync(tmpDir, { recursive: true, force: true });
    }

    const idx = await Pikelet.create({ dim: 4, maxElements: 4, metric: 'l2', quantized: false });
    idx.dispose();
    try {
        idx.search(new Float32Array(4), 1);
        assert(false, 'disposed index throws PikeletError');
    } catch (error) {
        assert(error instanceof Pikelet.PikeletError, 'disposed index error is a PikeletError');
        assert(error.code === Pikelet.PIKELET_ERROR_CODES.INDEX_DISPOSED,
            'disposed index error has INDEX_DISPOSED code');
    }
}

async function testPerQueryEfSearch() {
    section('Per-query efSearch');

    const idx = await Pikelet.create({
        dim: 4,
        maxElements: 8,
        metric: 'l2',
        quantized: false,
        efSearch: 100,
    });
    idx.addBatch([unitVec(4, 0), unitVec(4, 1), unitVec(4, 2)]);

    const observed = [];
    const originalQuery = idx._e._pikelet_query;
    const originalFiltered = idx._e._pikelet_query_filtered;
    idx._e._pikelet_query = (...args) => {
        observed.push(['search', args[3]]);
        return originalQuery(...args);
    };
    idx._e._pikelet_query_filtered = (...args) => {
        observed.push(['filtered', args[3]]);
        return originalFiltered(...args);
    };

    try {
        idx.search(unitVec(4, 0), 1);
        idx.search(unitVec(4, 0), 1, { efSearch: 12 });
        idx.search(unitVec(4, 0), 1);
        idx.searchFiltered(unitVec(4, 0), 1, new Set([0, 1]), { efSearch: 24 });
        idx.setEfSearch(160);
        idx.search(unitVec(4, 0), 1);

        assert(JSON.stringify(observed) === JSON.stringify([
            ['search', 100],
            ['search', 12],
            ['search', 100],
            ['filtered', 24],
            ['search', 160],
        ]), 'per-query efSearch reaches the ABI without mutating the index default');

        for (const invalid of [0, 4097, 1.5, '100']) {
            try {
                idx.search(unitVec(4, 0), 1, { efSearch: invalid });
                assert(false, `search() rejects invalid efSearch ${String(invalid)}`);
            } catch (error) {
                assert(error instanceof Pikelet.PikeletError,
                    `invalid efSearch ${String(invalid)} throws PikeletError`);
                assert(error.code === Pikelet.PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                    `invalid efSearch ${String(invalid)} has INVALID_ARGUMENT code`);
            }
        }
    } finally {
        idx._e._pikelet_query = originalQuery;
        idx._e._pikelet_query_filtered = originalFiltered;
        idx.dispose();
    }
}

async function testAdditiveIndexSurface() {
    section('0.2 index state and lifecycle surface');

    const idx = await Pikelet.create({
        dim: 4,
        maxElements: 5,
        metric: 'l2',
        quantized: false,
        M: 8,
        efConstruction: 32,
        efSearch: 40,
    });
    const ids = idx.addBatch([unitVec(4, 0), unitVec(4, 1), unitVec(4, 2)]);

    assert(idx.count === 3 && idx.liveCount === 3, 'liveCount starts at the backend count');
    assert(idx.deletedCount === 0 && idx.deletedRatio === 0, 'deleted state starts empty');
    assert(idx.capacity === 5 && idx.remainingCapacity === 2, 'capacity state reports unused insertion slots');
    assert(idx.has(ids[0]) && !idx.isDeleted(ids[0]), 'has() recognizes a live ID');
    assert(idx.delete(ids[0]) === true, 'delete() returns true for a live ID');
    assert(idx.delete(ids[0]) === false, 'delete() returns false for an already-deleted ID');
    assert(idx.delete(999999) === false, 'delete() returns false for an unknown ID');
    assert(!idx.has(ids[0]) && idx.isDeleted(ids[0]), 'has()/isDeleted() distinguish a soft-deleted ID');
    assert(idx.liveCount === 2 && idx.deletedCount === 1, 'live/deleted counts track soft deletion');
    assert(idx.remainingCapacity === 2, 'soft deletion does not claim to free insertion capacity');
    assert(idx.ghostCount === idx.deletedCount && idx.ghostRatio === idx.deletedRatio,
        'legacy ghost state remains an alias');

    const config = idx.config;
    assert(config.dim === 4 && config.maxElements === 5 && config.metric === 'l2' &&
        config.quantized === false && config.M === 8 && config.efConstruction === 32 &&
        config.efSearch === 40 && config.seed === 108,
        'config exposes fully resolved values');
    idx.setEfSearch(80);
    assert(idx.config.efSearch === 80, 'config reflects the current default efSearch policy');

    const beforeExportMemory = idx.memoryUsage;
    assert(beforeExportMemory.logicalIndexBytes === idx.memory,
        'memory remains an alias for logicalIndexBytes');
    assert(beforeExportMemory.wasmHeapBytes >= beforeExportMemory.logicalIndexBytes,
        'memoryUsage exposes the full WASM heap separately');
    assert(beforeExportMemory.snapshotBufferBytes === 0,
        'snapshotBufferBytes is zero before the first export');
    idx.compact();
    idx.export();
    assert(idx.memoryUsage.snapshotBufferBytes > 0,
        'memoryUsage tracks the backend buffer retained by export()');
    idx.dispose();

    let scopedIndex;
    const result = await Pikelet.withIndex(
        { dim: 4, maxElements: 2, metric: 'l2', quantized: false },
        async (index) => {
            scopedIndex = index;
            index.add(unitVec(4, 0));
            return index.liveCount;
        }
    );
    assert(result === 1, 'withIndex() returns the callback result');
    assertThrows(() => scopedIndex.count, 'withIndex() disposes after a successful callback');

    let failedIndex;
    await assertThrowsAsync(
        () => Pikelet.withIndex(
            { dim: 4, maxElements: 2, metric: 'l2', quantized: false },
            (index) => {
                failedIndex = index;
                throw new Error('expected callback failure');
            }
        ),
        'withIndex() propagates callback failures'
    );
    assertThrows(() => failedIndex.count, 'withIndex() disposes after a failed callback');
}

async function testSnapshotInspectionAndRestore() {
    section('Snapshot inspection and restore');

    const source = await Pikelet.create({
        dim: 4,
        maxElements: 8,
        metric: 'cosine',
        quantized: true,
        M: 8,
        efConstruction: 64,
        efSearch: 120,
    });
    const ids = source.addBatch([unitVec(4, 0), unitVec(4, 1), unitVec(4, 2)]);
    source.delete(ids[1]);
    source.compact();
    const snapshot = source.export();

    const metadata = Pikelet.inspectSnapshot(snapshot);
    assert(metadata.format === 'pikelet' && metadata.version === 3,
        'inspectSnapshot() identifies the current envelope');
    assert(metadata.dim === 4 && metadata.count === 2 && metadata.metric === 'cosine' &&
        metadata.quantized === true && metadata.M === 8 && metadata.efConstruction === 64,
        'inspectSnapshot() reports the complete construction config');
    assert(metadata.nextId === 3, 'inspectSnapshot() reports the stable next external ID');

    const exact = await Pikelet.restore(snapshot);
    assert(exact.config.dim === 4 && exact.config.metric === 'cosine' && exact.config.quantized === true &&
        exact.config.M === 8 && exact.config.efConstruction === 64 &&
        exact.config.efSearch === 100 && exact.config.seed === 108,
        'restore() infers snapshot config and resets runtime efSearch policy');
    assert(exact.capacity === 2 && exact.remainingCapacity === 0,
        'restore() defaults capacity to the restored count');
    exact.dispose();

    const grown = await Pikelet.restore(snapshot, { maxElements: 5, efSearch: 200 });
    assert(grown.capacity === 5 && grown.config.efSearch === 200,
        'restore() accepts capacity and runtime-policy overrides');
    const nextId = grown.add(unitVec(4, 3));
    assert(nextId === 3, 'restore() preserves external-ID allocation across compaction');
    grown.dispose();

    await assertThrowsAsync(
        () => Pikelet.restore(snapshot, { M: 16 }),
        'restore() rejects construction overrides that disagree with the snapshot'
    );

    const raw = extractRawEngineBytes(snapshot);
    const rawMetadata = Pikelet.inspectSnapshot(raw);
    assert(rawMetadata.format === 'raw' && rawMetadata.M === 8 && rawMetadata.efConstruction === 64,
        'inspectSnapshot() reads legacy raw headers');
    await assertThrowsAsync(
        () => Pikelet.restore(raw),
        'restore() requires explicit config for legacy raw snapshots'
    );
    const rawRestored = await Pikelet.restore(raw, {
        dim: 4,
        metric: 'cosine',
        quantized: true,
        M: 8,
        efConstruction: 64,
        maxElements: 4,
    });
    assert(rawRestored.count === 2, 'restore() supports explicitly configured legacy raw snapshots');
    rawRestored.dispose();
    source.dispose();
}

// ─── Float vs uint8 backend parity ────────────────────────────────────────────
//
// The two backends duplicate the HNSW graph algorithms (float_hnsw.hpp /
// uint8_float_hnsw.hpp) and share a byte-identical seeded level RNG, so for
// the same insert order the graph SKELETON — level sequence, entry point, max
// level — must match exactly. Neighbor SELECTION legitimately differs
// (quantized distances vs exact), so edges are compared with tolerances.
// These tests exist to catch the duplicated algorithms drifting apart as they
// evolve independently.

function seededParityVectors(count, dim, seed) {
    let state = seed >>> 0;
    const next = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0xffffffff;
    };
    const rows = [];
    for (let i = 0; i < count; i++) {
        const v = new Float32Array(dim);
        let prev = next() * 2 - 1;
        for (let d = 0; d < dim; d++) {
            prev = 0.75 * prev + 0.25 * (next() * 2 - 1);
            v[d] = prev;
        }
        rows.push(v);
    }
    return rows;
}

// Parse a raw engine snapshot's graph section into { levels, degrees }. The
// payload stride differs per backend (float: 4-byte neighbor ids; uint8:
// 8-byte {neighbor, dist} edges), but the framing is identical.
function parseGraphSkeleton(raw, quantized) {
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const dim = view.getUint32(4, true);
    const count = view.getUint32(12, true);
    const entryPoint = view.getUint32(16, true);
    const maxLevel = view.getUint32(20, true);
    let offset = quantized ? 40 + count * 8 + count * dim : 40 + count * dim * 4;
    const edgeStride = quantized ? 8 : 4;
    const levels = new Array(count);
    const degrees = new Array(count);
    for (let id = 0; id < count; id++) {
        const level = view.getUint32(offset, true); offset += 4;
        levels[id] = level;
        const perLevel = new Array(level + 1);
        for (let l = 0; l <= level; l++) {
            const size = view.getUint32(offset, true); offset += 4;
            perLevel[l] = size;
            offset += size * edgeStride;
        }
        degrees[id] = perLevel;
    }
    return { count, entryPoint, maxLevel, levels, degrees };
}

async function buildParityPair(vectors, cfg) {
    const pair = {};
    for (const quantized of [false, true]) {
        const index = await Pikelet.create({ ...cfg, quantized });
        index.addBatch(vectors);
        pair[quantized ? 'uint8' : 'float'] = index;
    }
    return pair;
}

function paritySkeletons(pair) {
    return {
        float: parseGraphSkeleton(extractRawEngineBytes(pair.float.export()), false),
        uint8: parseGraphSkeleton(extractRawEngineBytes(pair.uint8.export()), true),
    };
}

function assertSkeletonParity(skel, label) {
    assert(JSON.stringify(skel.float.levels) === JSON.stringify(skel.uint8.levels),
        `${label}: identical level sequence (shared seeded RNG)`);
    assert(skel.float.entryPoint === skel.uint8.entryPoint,
        `${label}: identical entry point`);
    assert(skel.float.maxLevel === skel.uint8.maxLevel,
        `${label}: identical max level`);
    // Neighbor counts are distance-driven, so they may differ per node — but
    // aggregate degree should stay close. A systematic gap points at the
    // dist-provenance divergence between the two prune implementations.
    let floatBase = 0;
    let uint8Base = 0;
    for (let id = 0; id < skel.float.count; id++) {
        floatBase += skel.float.degrees[id][0];
        uint8Base += skel.uint8.degrees[id][0];
    }
    const ratio = Math.min(floatBase, uint8Base) / Math.max(floatBase, uint8Base);
    assert(ratio >= 0.9,
        `${label}: aggregate base degree within 10% (float ${floatBase} vs uint8 ${uint8Base})`);
}

function parityOverlap(pair, queries, k, efSearch) {
    let top1 = 0;
    let overlap = 0;
    for (const q of queries) {
        const a = pair.float.search(q, k, { efSearch }).map((r) => r.id);
        const b = pair.uint8.search(q, k, { efSearch }).map((r) => r.id);
        if (a[0] === b[0]) top1++;
        const bSet = new Set(b);
        overlap += a.filter((id) => bSet.has(id)).length / k;
    }
    return { top1: top1 / queries.length, overlap: overlap / queries.length };
}

async function testFloatUint8GraphParity() {
    section('Float vs uint8 graph parity — build');

    const cfg = { dim: 32, maxElements: 400, metric: 'l2', M: 8, efConstruction: 60, seed: 7 };
    const vectors = seededParityVectors(400, cfg.dim, 20260812);
    const pair = await buildParityPair(vectors, cfg);

    assertSkeletonParity(paritySkeletons(pair), 'build');

    const queries = seededParityVectors(25, cfg.dim, 424242);
    const agreement = parityOverlap(pair, queries, 10, 150);
    assert(agreement.top1 >= 0.85,
        `top-1 agreement across backends >= 85% (got ${(agreement.top1 * 100).toFixed(0)}%)`);
    assert(agreement.overlap >= 0.8,
        `mean top-10 overlap across backends >= 80% (got ${(agreement.overlap * 100).toFixed(0)}%)`);

    // Recall gap against one shared exact ground truth (float brute force):
    // quantization costs a bounded amount of recall, and a widening gap means
    // one backend's graph quality regressed relative to the other.
    let floatHits = 0;
    let uint8Hits = 0;
    for (const q of queries) {
        const truth = vectors
            .map((v, id) => {
                let s = 0;
                for (let d = 0; d < cfg.dim; d++) s += (q[d] - v[d]) ** 2;
                return [s, id];
            })
            .sort((x, y) => x[0] - y[0] || x[1] - y[1])
            .slice(0, 10)
            .map((e) => e[1]);
        const truthSet = new Set(truth);
        floatHits += pair.float.search(q, 10, { efSearch: 150 }).filter((r) => truthSet.has(r.id)).length;
        uint8Hits += pair.uint8.search(q, 10, { efSearch: 150 }).filter((r) => truthSet.has(r.id)).length;
    }
    const floatRecall = floatHits / (queries.length * 10);
    const uint8Recall = uint8Hits / (queries.length * 10);
    assert(Math.abs(floatRecall - uint8Recall) <= 0.05,
        `recall gap vs shared ground truth <= 5 points (float ${(floatRecall * 100).toFixed(1)} vs uint8 ${(uint8Recall * 100).toFixed(1)})`);

    pair.float.dispose();
    pair.uint8.dispose();
}

async function testFloatUint8DeleteCompactParity() {
    section('Float vs uint8 graph parity — delete/compact/heal');

    const cfg = { dim: 32, maxElements: 300, metric: 'l2', M: 8, efConstruction: 60, seed: 11 };
    const vectors = seededParityVectors(300, cfg.dim, 77);
    const queries = seededParityVectors(20, cfg.dim, 88);

    // Heavy deletion (>= half) drives the rebuild path in compact(): a fresh
    // graph is constructed with the same seed, so the post-compact skeleton
    // must again match exactly across backends.
    {
        const pair = await buildParityPair(vectors, cfg);
        for (let id = 0; id < 300; id++) {
            if (id % 5 !== 0) {
                pair.float.delete(id);
                pair.uint8.delete(id);
            }
        }
        assert(pair.float.ghostCount === pair.uint8.ghostCount && pair.float.ghostCount === 240,
            'heavy delete: identical ghost counts');
        const ghostAgreement = parityOverlap(pair, queries, 5, 150);
        assert(ghostAgreement.overlap >= 0.8,
            `heavy delete: top-5 overlap on the ghosted graph >= 80% (got ${(ghostAgreement.overlap * 100).toFixed(0)}%)`);

        pair.float.compact();
        pair.uint8.compact();
        assert(pair.float.count === pair.uint8.count && pair.float.count === 60,
            'heavy compact: identical survivor counts');
        assertSkeletonParity(paritySkeletons(pair), 'rebuild compact');

        // Both backends must resolve every survivor to itself at rank 1.
        let floatSelf = 0;
        let uint8Self = 0;
        for (let id = 0; id < 300; id += 5) {
            if (pair.float.search(vectors[id], 1, { efSearch: 100 })[0].id === id) floatSelf++;
            if (pair.uint8.search(vectors[id], 1, { efSearch: 100 })[0].id === id) uint8Self++;
        }
        assert(floatSelf === 60 && uint8Self === 60,
            `post-compact self-recall 60/60 on both backends (float ${floatSelf}, uint8 ${uint8Self})`);
        pair.float.dispose();
        pair.uint8.dispose();
    }

    // Light deletion (< half) drives the in-place path: remap, ghost-ref
    // stripping, backfill, and the heal loop — where the two implementations
    // have genuinely different control flow. Levels are survivor-preserved so
    // the skeleton must still match; degrees get the aggregate tolerance.
    {
        const pair = await buildParityPair(vectors, cfg);
        for (let id = 0; id < 90; id++) {
            pair.float.delete(id);
            pair.uint8.delete(id);
        }
        pair.float.compact();
        pair.uint8.compact();
        assert(pair.float.count === pair.uint8.count && pair.float.count === 210,
            'in-place compact: identical survivor counts');
        assertSkeletonParity(paritySkeletons(pair), 'in-place compact');
        const agreement = parityOverlap(pair, queries, 10, 150);
        assert(agreement.overlap >= 0.8,
            `in-place compact: top-10 overlap >= 80% (got ${(agreement.overlap * 100).toFixed(0)}%)`);
        pair.float.dispose();
        pair.uint8.dispose();
    }
}

// ─── Runner ───────────────────────────────────────────────────────────────────

async function main() {
    console.log('');
    console.log('Pikelet Test Suite');
    console.log('==================');

    const suites = [
        testCreation,
        testAdd,
        testAddBatch,
        testSearch,
        testSearchMetrics,
        testDelete,
        testGhostRatio,
        testCompact,
        testCompactReconnectsIsolatedSurvivors,
        testMemory,
        testDispose,
        testDeterminism,
        testSearchDuringMutation,
        testRuntimeEntryPoints,
        testExportImport,
        testConvenienceLoaders,
        testLargeIndex,
        testQuantized,
        testErrorPaths,
        testEdgeCases,
        testMaxElementsOverflow,
        testExportImportWithGhosts,
        testExportImportWithGhostsQuantized,
        testAddBatchPartialFailure,
        testIdLifecycleAcrossCompaction,
        testZeroAndDegenerateVectors,
        testDeterminismIds,
        testMetricCorrectnessQuantized,
        testRawEngineImportValidation,
        testDeleteChurnRegression,
        testGhostEntryPointSearch,
        testGhostsRemainNavigable,
        testGoldenSnapshotCompatibility,
        testNonFiniteRejection,
        testCosineNormOverflowRegression,
        testCompactEntryPointRecovery,
        testSearchFiltered,
        testSearchFilteredInputContract,
        testDemoVectorGenerator,
        testPikeletErrorContract,
        testPerQueryEfSearch,
        testAdditiveIndexSurface,
        testSnapshotInspectionAndRestore,
        testHeldOutRecallOracle,
        testHeldOutRecallAfterCompact,
        testFilteredHeldOutRecallOracle,
        testSearchOutputGoldenOracle,
        testSearchAndSerializationDeterminismOracle,
        testFloatUint8GraphParity,
        testFloatUint8DeleteCompactParity,
    ];

    for (const suite of suites) {
        try {
            await suite();
        } catch (e) {
            failed++;
            failures.push(`${suite.name} threw unexpectedly: ${e.message}`);
            console.error(`  ✗ ${suite.name} threw: ${e.message}`);
        }
    }

    console.log('');
    console.log('─'.repeat(40));
    console.log(`  Passed: ${passed}`);
    console.log(`  Failed: ${failed}`);

    if (failures.length > 0) {
        console.log('\nFailures:');
        failures.forEach(f => console.log(`  • ${f}`));
    }

    console.log('');
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error('Test runner error:', e);
    process.exit(1);
});
