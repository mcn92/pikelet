// Artifact-layer hardening conformance (range + sketch readers, snapshot
// parser): the lifecycle and hostile-input rules that an external review on
// 2026-08-21 found missing. Hermetic — fixtures are built in-process from a
// tiny quantized index — and fast.
//
//   1. a rejected openFile() releases its file descriptor (range and sketch);
//   2. query vectors follow the core engine's contract: numeric elements,
//      finite components for every metric, right dimension;
//   3. k / rerank / efSearch are validated and bounded by the row count, so a
//      k of 1e9 over a small artifact returns count results without trying
//      to allocate 1e9 slots;
//   4. the uint8 snapshot parser refuses adjacency counts above M/M0, out of
//      range neighbor ids, and implausible graph parameters before any
//      allocation.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const Pikelet = require('../pikelet.js');
const {
    PikeletRangeArtifact, PikeletSketchArtifact, buildRangeArtifact, buildSketchArtifact, parseUint8Snapshot,
} = require('../pikelet-artifact.js');

let passed = 0, failed = 0;
function check(label, cond, detail = '') {
    if (cond) { passed++; console.log('  ok:', label); }
    else { failed++; console.log('  FAIL:', label, detail ? `— ${detail}` : ''); }
}
async function rejects(label, fn, code, pattern) {
    try {
        await fn();
        check(label, false, 'resolved instead of throwing');
    } catch (err) {
        const ok = (!code || err.code === code) && (!pattern || pattern.test(String(err.message)));
        check(label, ok, ok ? '' : `threw ${err.code}: ${String(err.message).slice(0, 140)}`);
    }
}
function openFds() {
    try { return fs.readdirSync('/proc/self/fd').length; } catch { return null; }
}

async function buildFixtures(tmp, metric) {
    const dim = 8, count = 40;
    const index = await Pikelet.create({ dim, maxElements: 100, metric, quantized: true });
    const vectors = [];
    for (let n = 0; n < count; n++) {
        const v = new Float32Array(dim);
        for (let d = 0; d < dim; d++) v[d] = Math.sin(n * 7 + d * 1.3);
        index.add(v);
        vectors.push(v);
    }
    const snapshot = index.export();
    index.dispose();
    const rangePath = path.join(tmp, `${metric}.pikelet-range`);
    const sketchPath = path.join(tmp, `${metric}.pikelet-sketch`);
    buildRangeArtifact(snapshot, rangePath, { layout: 'rcm' });
    buildSketchArtifact(snapshot, sketchPath, { recommendedRerank: 20 });
    return { dim, count, snapshot, vectors, rangePath, sketchPath };
}

async function main() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-hardening-'));

    // 1. Failed opens do not leak descriptors.
    console.log('1. openFile() releases its descriptor when the open is rejected');
    const garbage = path.join(tmp, 'garbage.bin');
    fs.writeFileSync(garbage, Buffer.alloc(4096, 0xab));
    const before = openFds();
    let rangeRejected = 0, sketchRejected = 0;
    for (let i = 0; i < 50; i++) {
        try { await PikeletRangeArtifact.openFile(garbage); } catch { rangeRejected++; }
        try { await PikeletSketchArtifact.openFile(garbage); } catch { sketchRejected++; }
    }
    const after = openFds();
    check('50 garbage range opens and 50 garbage sketch opens are all rejected', rangeRejected === 50 && sketchRejected === 50);
    if (before === null) console.log('  (skip: /proc/self/fd unavailable on this platform; descriptor count not measured)');
    else check('open descriptor count is unchanged after 100 rejected opens', after - before <= 0, `before ${before}, after ${after}`);
    await rejects('openFile() on a missing path rejects with ENOENT and leaves no descriptor',
        () => PikeletRangeArtifact.openFile(path.join(tmp, 'missing.pikelet-range')), undefined, /ENOENT/);
    if (before !== null) check('descriptor count still unchanged after the missing-file open', openFds() - before <= 0);

    for (const metric of ['l2', 'cosine']) {
        console.log(`\n2-3. query contract and bounded k on ${metric} artifacts`);
        const fx = await buildFixtures(tmp, metric);
        const range = await PikeletRangeArtifact.openFile(fx.rangePath);
        const sketch = await PikeletSketchArtifact.openFile(fx.sketchPath);
        const q = fx.vectors[3];
        const strings = Array.from(q, (v) => String(v));
        const withNaN = Float32Array.from(q); withNaN[2] = NaN;
        const withInf = Float32Array.from(q); withInf[5] = Infinity;

        const baseline = (await range.search(q, 5)).results;
        check('range search returns results for a valid query', baseline.length === 5 && Number.isFinite(baseline[0].distance));
        await rejects('range: numeric-string coordinates are rejected', () => range.search(strings, 5), 'INVALID_VECTOR', /only numbers/);
        await rejects('range: NaN component is rejected', () => range.search(withNaN, 5), 'INVALID_VECTOR', /non-finite/);
        await rejects('range: Infinity component is rejected', () => range.search(withInf, 5), 'INVALID_VECTOR', /non-finite/);
        await rejects('range: wrong dimension is rejected', () => range.search(new Float32Array(fx.dim + 1), 5), 'DIMENSION_MISMATCH');
        await rejects('range: k=0 is rejected', () => range.search(q, 0), 'INVALID_ARGUMENT', /positive integer/);
        await rejects('range: fractional k is rejected', () => range.search(q, 2.5), 'INVALID_ARGUMENT');
        await rejects('range: efSearch must be a positive integer', () => range.search(q, 5, { efSearch: -1 }), 'INVALID_ARGUMENT', /efSearch/);
        check('range: efSearch: 0 still means the default (historical `|| 100`)', (await range.search(q, 5, { efSearch: 0 })).results.length === 5);
        const hugeRange = (await range.search(q, 1e9)).results;
        check('range: k=1e9 returns at most count results', hugeRange.length <= fx.count && hugeRange.length > 0, `${hugeRange.length}`);
        check('range: k=1e9 top result equals the k=5 top result', hugeRange[0].id === baseline[0].id);

        const sBaseline = (await sketch.search(q, 5)).results;
        check('sketch search returns results for a valid query', sBaseline.length === 5 && Number.isFinite(sBaseline[0].distance));
        await rejects('sketch: numeric-string coordinates are rejected', () => sketch.search(strings, 5), 'INVALID_VECTOR', /only numbers/);
        await rejects('sketch: NaN component is rejected', () => sketch.search(withNaN, 5), 'INVALID_VECTOR', /non-finite/);
        await rejects('sketch: wrong dimension is rejected', () => sketch.search(new Float32Array(fx.dim + 1), 5), 'DIMENSION_MISMATCH');
        await rejects('sketch: k=0 is rejected', () => sketch.search(q, 0), 'INVALID_ARGUMENT', /positive integer/);
        await rejects('sketch: rerank must be a positive integer', () => sketch.search(q, 5, { rerank: 1.5 }), 'INVALID_ARGUMENT', /rerank/);
        check('sketch: rerank: 0 still means the default (historical `|| 0`)', (await sketch.search(q, 5, { rerank: 0 })).results.length === 5);
        const t0 = Date.now();
        const hugeSketch = (await sketch.search(q, 1e9, { rerank: 1e9 })).results;
        check('sketch: k=1e9 / rerank=1e9 returns at most count results without a giant allocation',
            hugeSketch.length <= fx.count && hugeSketch.length > 0 && Date.now() - t0 < 5000, `${hugeSketch.length} in ${Date.now() - t0} ms`);
        check('sketch: k=1e9 top result equals the k=5 top result', hugeSketch[0].id === sBaseline[0].id);
        await range.close();
        await sketch.close();
        await sketch.close();
        check('sketch close() is idempotent', true);
        await rejects('reads through a closed NodeFileRangeSource are refused', () => range.source.read(0, 16), 'INVALID_ARGUMENT', /closed/);
    }

    // 4. Snapshot parser bounds.
    console.log('\n4. uint8 snapshot parser refuses hostile adjacency before allocating');
    const fx = await buildFixtures(tmp, 'l2');
    // Locate the raw uint8 payload inside the export envelope by its magic
    // (parseUint8Snapshot accepts a bare payload), then tamper node 0's first
    // adjacency count.
    const magic = Buffer.from([0x31, 0x48, 0x38, 0x49]); // UINT8_HNSW_MAGIC_V1 LE
    const snap = Buffer.from(fx.snapshot);
    const rawAt = snap.indexOf(magic);
    check('raw uint8 payload located in the export', rawAt >= 0);
    const raw = Buffer.from(snap.subarray(rawAt));
    const graph = parseUint8Snapshot(raw);
    check('untampered payload parses', graph.count === fx.count && graph.M0 >= graph.M);
    const levelsAt = 40 + fx.count * 8 + fx.count * fx.dim;
    const sizeAt = levelsAt + 4; // node 0: u32 level, then u32 size for level 0
    const hugeEdges = Buffer.from(raw); hugeEdges.writeUInt32LE(0xfffffff0, sizeAt);
    const t0 = Date.now();
    await rejects('adjacency count above M0 is rejected (no allocation)', async () => parseUint8Snapshot(hugeEdges), 'SNAPSHOT_INVALID', /adjacency exceeds/);
    check('...and quickly', Date.now() - t0 < 1000);
    const overCap = Buffer.from(raw); overCap.writeUInt32LE(graph.M0 + 1, sizeAt);
    await rejects('adjacency count of M0+1 at layer 0 is rejected', async () => parseUint8Snapshot(overCap), 'SNAPSHOT_INVALID', /adjacency exceeds/);
    const badNeighbor = Buffer.from(raw); badNeighbor.writeUInt32LE(fx.count + 7, sizeAt + 4); // first neighbor id of node 0
    await rejects('neighbor id >= count is rejected', async () => parseUint8Snapshot(badNeighbor), 'SNAPSHOT_INVALID', /neighbor id/);
    const badParams = Buffer.from(raw); badParams.writeUInt32LE(0, 24); // M = 0
    await rejects('implausible graph parameters (M=0) are rejected', async () => parseUint8Snapshot(badParams), 'SNAPSHOT_INVALID', /graph parameters/);
    const truncatedEdges = Buffer.from(raw.subarray(0, sizeAt + 8)); // claims edges, bytes end
    await rejects('adjacency that runs past the buffer is rejected as truncated', async () => parseUint8Snapshot(truncatedEdges), 'SNAPSHOT_INVALID', /truncated|adjacency/);

    // Version-1 payloads store 4 bytes per edge (id only; the engine
    // recomputes distances on load). Re-encode the v2 graph as v1 and the
    // parser must produce the same adjacency.
    {
        const v1 = [];
        const head = Buffer.from(raw.subarray(0, levelsAt)); head.writeUInt32LE(1, 8); // version = 1
        v1.push(head);
        let off = levelsAt;
        const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        for (let id = 0; id < fx.count; id++) {
            const level = view.getUint32(off, true); off += 4;
            const node = [Buffer.alloc(4)]; node[0].writeUInt32LE(level, 0);
            for (let l = 0; l <= level; l++) {
                const size = view.getUint32(off, true); off += 4;
                const b = Buffer.alloc(4 + 4 * size); b.writeUInt32LE(size, 0);
                for (let e = 0; e < size; e++) { b.writeUInt32LE(view.getUint32(off, true), 4 + 4 * e); off += 8; }
                node.push(b);
            }
            v1.push(...node);
        }
        const v1graph = parseUint8Snapshot(Buffer.concat(v1));
        const same = v1graph.count === graph.count && v1graph.version === 1
            && Array.from({ length: fx.count }, (_, i) => i).every((i) => Buffer.from(v1graph.base[i].buffer, v1graph.base[i].byteOffset, v1graph.base[i].byteLength)
                .equals(Buffer.from(graph.base[i].buffer, graph.base[i].byteOffset, graph.base[i].byteLength)));
        check('a version-1 payload (4-byte edges) parses to the same layer-0 adjacency as v2', same);
    }

    // 5. A candidate pool equal to the row count (small artifact, or a large
    // k / rerank) must not pay a quadratic resident scan: every row is a
    // candidate and the exact rerank scores them all.
    console.log('\n5. sketch search with C == count stays linear');
    {
        const dim = 16, count = 6000;
        const index = await Pikelet.create({ dim, maxElements: count, metric: 'l2', quantized: true });
        for (let n = 0; n < count; n++) {
            const v = new Float32Array(dim);
            for (let d = 0; d < dim; d++) v[d] = Math.sin(n * 0.37 + d);
            index.add(v);
        }
        const big = index.export(); index.dispose();
        const bigPath = path.join(tmp, 'big.pikelet-sketch');
        buildSketchArtifact(big, bigPath, { recommendedRerank: 40 });
        const sketch = await PikeletSketchArtifact.openFile(bigPath);
        const q = new Float32Array(dim); for (let d = 0; d < dim; d++) q[d] = Math.sin(17 * 0.37 + d);
        const t0 = Date.now();
        const all = (await sketch.search(q, count)).results;
        const ms = Date.now() - t0;
        check(`search(q, count) over ${count} rows returns ${count} results`, all.length === count);
        check(`...in linear time (${ms} ms, not a 2*count^2 scan)`, ms < 3000);
        const top = (await sketch.search(q, 5)).results;
        check('k=count top-5 agree with k=5 (exact rerank over all rows)', all.slice(0, 5).every((r, i) => r.id === top[i].id));
        await sketch.close();
    }

    // 6. Stable external ids are bounded to the uint32 the snapshot stores.
    console.log('\n6. stable-id space is bounded to the uint32 snapshot representation');
    {
        const index = await Pikelet.create({ dim: 4, maxElements: 16, metric: 'l2', quantized: true });
        const v = new Float32Array(4).fill(0.5);
        index._nextExtId = 0xFFFFFFFE; // one id left in the u32 space
        const last = index.add(v);
        check('the last representable id (0xFFFFFFFE) is assigned', last === 0xFFFFFFFE);
        const snapshot = index.export();
        const restored = await Pikelet.restore(snapshot, { maxElements: 16 });
        check('a snapshot at the id boundary round-trips (nextExtId = 0xFFFFFFFF fits u32)', restored.count === 1);
        restored.dispose();
        const countBefore = index.count;
        await rejects('the next id assignment is refused before the engine mutates',
            async () => index.add(v), 'INDEX_LIMIT', /Stable id space exhausted/);
        check('the refused add did not insert', index.count === countBefore);
        await rejects('addBatch preflights the whole batch against the id bound',
            async () => { index._nextExtId = 0xFFFFFFFD; return index.addBatch([v, v, v]); }, 'INDEX_LIMIT', /run past the last representable id/);
        check('the refused batch did not insert', index.count === countBefore);
        index.dispose();
    }

    fs.rmSync(tmp, { recursive: true, force: true });
    console.log(`\nArtifact hardening: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => { console.error(err); process.exit(1); });
