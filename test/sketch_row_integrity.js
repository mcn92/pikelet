// Sketch format-2 row integrity conformance: interleaved digest blocks,
// page-hash anchoring in the resident prefix, per-read verification in
// fetchRows (full and staged tiers), version-1 compatibility, and result
// parity between the two layouts.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const Pikelet = require('../pikelet.js');
const { PikeletSketchArtifact, exportSketchArtifact } = require('../pikelet-artifact.js');

let passed = 0, failed = 0;
function check(label, cond, detail) {
    if (cond) { passed++; console.log('  ok:', label); }
    else { failed++; console.log('  FAIL:', label, detail ? `— ${detail}` : ''); }
}
async function rejects(label, fn, pattern) {
    try {
        await fn();
        check(label, false, 'resolved instead of throwing');
    } catch (err) {
        const ok = pattern.test(String(err.message));
        check(label, ok, ok ? '' : `threw: ${String(err.message).slice(0, 120)}`);
    }
}
function mulberry32(a) {
    return function () {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

function makeIndex(dim, count, seed) {
    const rand = mulberry32(seed);
    const qdata = new Uint8Array(count * dim);
    const scales = new Float32Array(count);
    const offsets = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        const v = new Float32Array(dim);
        let n = 0;
        for (let d = 0; d < dim; d++) { const x = rand() * 2 - 1; v[d] = x; n += x * x; }
        n = 1 / Math.sqrt(n);
        let mn = Infinity, mx = -Infinity;
        for (let d = 0; d < dim; d++) { v[d] *= n; if (v[d] < mn) mn = v[d]; if (v[d] > mx) mx = v[d]; }
        const s = (mx - mn) / 255 || 1e-12;
        scales[i] = s; offsets[i] = mn;
        for (let d = 0; d < dim; d++) {
            const b = Math.round((v[d] - mn) / s);
            qdata[i * dim + d] = b < 0 ? 0 : b > 255 ? 255 : b;
        }
    }
    return { dim, count, metric: 1, qdata, scales, offsets };
}

function queryFor(index, id) {
    const q = new Float32Array(index.dim);
    for (let d = 0; d < index.dim; d++) q[d] = index.offsets[id] + index.scales[id] * index.qdata[id * index.dim + d];
    return q;
}

async function main() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-rowint-'));
    const dim = 32, count = 500; // 500 rows -> partial last 16-row block
    const index = makeIndex(dim, count, 42);
    const opts = { sketchDims: 16, sketchBits: 8, recommendedRerank: 40 };
    const v1Path = path.join(tmp, 'v1.pikelet-sketch');
    const v2Path = path.join(tmp, 'v2.pikelet-sketch');
    const v1m = exportSketchArtifact(index, v1Path, { ...opts, rowIntegrity: false });
    const v2m = exportSketchArtifact(index, v2Path, opts);

    console.log('1. builder');
    check('v1 manifest reports formatVersion 1, no rowIntegrity', v1m.formatVersion === 1 && v1m.rowIntegrity === null);
    check('v2 manifest reports formatVersion 2 with 16-row/16-byte blocks',
        v2m.formatVersion === 2 && v2m.rowIntegrity.rowsPerBlock === 16 && v2m.rowIntegrity.rowDigestBytes === 16
        && v2m.rowIntegrity.blocks === Math.ceil(count / 16));
    const overhead = fs.statSync(v2Path).size - fs.statSync(v1Path).size;
    check(`v2 overhead is digest pages + page table (+${overhead} bytes)`,
        overhead > 0 && overhead <= v2m.rowIntegrity.blocks * (16 * 16 + 32) + 32);

    console.log('2. parity and per-read verification (full open)');
    const v1 = await PikeletSketchArtifact.openFile(v1Path);
    const v2 = await PikeletSketchArtifact.openFile(v2Path);
    check('v2 opens with residentVerified and per-row geometry', v2.stats().residentVerified
        && v2.formatVersion === 2 && v2.rowsPerBlock === 16 && v2.rowDigestBytes === 16);
    let same = true;
    for (let q = 0; q < 12; q++) {
        const query = queryFor(index, q * 37 % count);
        const a = (await v1.search(query, 10)).results;
        const b = (await v2.search(query, 10)).results;
        if (JSON.stringify(a) !== JSON.stringify(b)) same = false;
    }
    check('v1 and v2 return identical results for identical rows', same);
    const bulk = await v2.fetchRows(Array.from({ length: count }, (_, i) => i), {});
    check('bulk fetch of every row verifies and returns all rows (partial last block)', bulk.size === count
        && Buffer.from(bulk.get(count - 1)).equals(Buffer.from(index.qdata.subarray((count - 1) * dim, count * dim))));
    const splitFetch = await (async () => {
        v2.clearCache();
        const before = v2.stats().rangeRequests;
        await v2.fetchRows([0, 250, 499], { maxRangeBytes: 8192, gap: 0 });
        return v2.stats().rangeRequests - before;
    })();
    check('scattered blocks fetch as separate bounded runs', splitFetch === 3, `${splitFetch} runs`);
    await v1.close();
    await v2.close();

    console.log('3. tamper detection');
    const raw = fs.readFileSync(v2Path);
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const vectorsOffset = view.getUint32(44, true);
    const pageTableOffset = view.getUint32(176, true);
    const pageBytes = 16 * 16;
    const blockBytes = pageBytes + 16 * dim;
    const tamper = async (label, mutate, pattern, { search } = {}) => {
        const bytes = Buffer.from(raw);
        mutate(bytes);
        const p = path.join(tmp, 'tampered.pikelet-sketch');
        fs.writeFileSync(p, bytes);
        if (pattern === null) {
            const art = await PikeletSketchArtifact.openFile(p);
            const out = await art.search(queryFor(index, 3), 5);
            check(label, out.results.length === 5);
            await art.close();
            return;
        }
        await rejects(label, async () => {
            const art = await PikeletSketchArtifact.openFile(p, search ? {} : undefined);
            try {
                await art.search(queryFor(index, 3), 5, { rerank: count });
            } finally {
                await art.close().catch(() => {});
            }
        }, pattern);
    };
    // Row byte in block 2: open succeeds (lazy), the fetch that touches it fails.
    await tamper('a tampered row fails its digest on the read that fetches it',
        (b) => { b[vectorsOffset + 2 * blockBytes + pageBytes + 5] ^= 0xff; }, /row failed digest verification/);
    // Digest page byte: the block's page hash no longer matches the table.
    await tamper('a tampered digest page fails against the resident page table',
        (b) => { b[vectorsOffset + 2 * blockBytes + 3] ^= 0xff; }, /digest page failed hash verification/);
    // Page-table byte: resident prefix hash fails at open.
    await tamper('a tampered page table fails residentSha256 at open',
        (b) => { b[pageTableOffset + 7] ^= 0xff; }, /resident prefix failed hash verification/);
    // A consistent rewrite of row + digest page + page table defeats the
    // sketch's own chain only if the header hash is also rewritten — which
    // any container that commits to the header (the complete profile) pins.
    // Standalone, rewriting all three plus residentSha256 is the documented
    // limit of a bare sketch file, same as v1.
    // verify:false skips row verification.
    {
        const bytes = Buffer.from(raw);
        bytes[vectorsOffset + 2 * blockBytes + pageBytes + 5] ^= 0xff;
        const p = path.join(tmp, 'skipverify.pikelet-sketch');
        fs.writeFileSync(p, bytes);
        const art = await PikeletSketchArtifact.openFile(p, { verify: false });
        const out = await art.search(queryFor(index, 3), 5, { rerank: count });
        check('verify:false serves tampered rows (explicit opt-out)', out.results.length === 5);
        await art.close();
    }

    console.log('4. staged micro tier (v2)');
    const stagedPath = path.join(tmp, 'staged.pikelet-sketch');
    exportSketchArtifact(index, stagedPath, { ...opts, microDims: 8, microBits: 8 });
    const staged = await PikeletSketchArtifact.openFile(stagedPath, { staged: true });
    check('staged v2 opens in the micro tier with the page table resident', staged.tier === 'micro' && !!staged.pageTable);
    const microOut = await staged.search(queryFor(index, 7), 5);
    check('micro-tier search verifies rows through the stage-1 page table', microOut.tier === 'micro' && microOut.results.length === 5);
    await staged.fullyResident;
    const fullOut = await staged.search(queryFor(index, 7), 5);
    check('full tier after staged boot still serves', fullOut.tier === 'full' && fullOut.results.length === 5);
    await staged.close();
    // A tampered row is caught during the micro window too.
    const stagedRaw = fs.readFileSync(stagedPath);
    const sview = new DataView(stagedRaw.buffer, stagedRaw.byteOffset, stagedRaw.byteLength);
    const sVectors = sview.getUint32(44, true);
    const tamperedStaged = Buffer.from(stagedRaw);
    tamperedStaged[sVectors + pageBytes + 1] ^= 0xff; // row in block 0
    const tsPath = path.join(tmp, 'staged-tampered.pikelet-sketch');
    fs.writeFileSync(tsPath, tamperedStaged);
    await rejects('a tampered row fails during the staged micro window', async () => {
        const art = await PikeletSketchArtifact.openFile(tsPath, { staged: true });
        try {
            await art.search(queryFor(index, 0), 5, { rerank: count });
        } finally {
            await art.close().catch(() => {});
        }
    }, /row failed digest verification|digest page failed/);

    console.log('5. verifyVectors covers the interleaved region');
    const vv = await PikeletSketchArtifact.openFile(v2Path);
    check('verifyVectors passes on a clean v2 artifact', await vv.verifyVectors() === true && vv.stats().vectorsVerified);
    await vv.close();

    fs.rmSync(tmp, { recursive: true, force: true });
    console.log(`\nSketch row integrity: ${passed} passed, ${failed} failed`);
    process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => { console.error(err); process.exit(1); });
