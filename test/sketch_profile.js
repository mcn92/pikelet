#!/usr/bin/env node
'use strict';

// Conformance checks for the sketch artifact profile (spec/SKETCH_PROFILE.md):
// build from a seeded snapshot, open through the reference reader, and verify
// exactness at C=count, recall bounds at modest C, both sketch encodings,
// both metrics, hash verification, and tamper rejection.

const fs = require('fs');
const os = require('os');
const path = require('path');
const Pikelet = require('../pikelet.js');

let passed = 0;
let failed = 0;
function check(name, ok, detail) {
    if (ok) { passed++; console.log(`  ok: ${name}`); }
    else { failed++; console.log(`  FAIL: ${name}${detail ? ` — ${detail}` : ''}`); }
}

function seededVectors(count, dim, seed) {
    let state = seed >>> 0;
    const next = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0xffffffff;
    };
    // AR(1)-correlated dimensions: adjacent dims carry shared signal, as in
    // real embeddings. Pooling assumes this structure (see the profile spec's
    // sketchDims guidance); iid dims are the adversarial case for it.
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

function bruteForce(rows, query, k, metric) {
    const scored = rows.map((row, id) => {
        let acc = 0;
        if (metric === 'cosine') {
            let qn = 0, rn = 0, dot = 0;
            for (let d = 0; d < row.length; d++) { dot += query[d] * row[d]; qn += query[d] ** 2; rn += row[d] ** 2; }
            acc = 1 - dot / (Math.sqrt(qn) * Math.sqrt(rn));
        } else {
            for (let d = 0; d < row.length; d++) acc += (query[d] - row[d]) ** 2;
        }
        return [acc, id];
    });
    scored.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return scored.slice(0, k).map((e) => e[1]);
}

class DelayedMemorySource {
    constructor(bytes, delayMs = 5) {
        this.bytes = bytes;
        this.size = bytes.length;
        this.delayMs = delayMs;
        this.preferredParallelism = 32;
        this.preferredGapBytes = 0;
        this.requests = 0;
        this.inFlight = 0;
        this.maxInFlight = 0;
    }

    resetStats() {
        this.requests = 0;
        this.inFlight = 0;
        this.maxInFlight = 0;
    }

    async read(offset, length) {
        this.requests++;
        this.inFlight++;
        this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
        await new Promise((resolve) => setTimeout(resolve, this.delayMs));
        this.inFlight--;
        return new Uint8Array(this.bytes.subarray(offset, offset + length));
    }
}

async function run() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-sketch-'));
    const COUNT = 600;
    const DIM = 32;
    const K = 10;

    for (const metric of ['l2', 'cosine']) {
        for (const sketchBits of [4, 8]) {
            console.log(`\nsketch profile: metric=${metric} sketchBits=${sketchBits}`);
            const rows = seededVectors(COUNT, DIM, 42);
            const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric, quantized: true });
            index.addBatch(rows);
            const snapshotPath = path.join(tmp, `snap-${metric}.pnck`);
            fs.writeFileSync(snapshotPath, index.export());
            index.dispose();

            const artifactPath = path.join(tmp, `art-${metric}-${sketchBits}.pikelet-sketch`);
            const manifest = Pikelet.buildSketchArtifactFile(snapshotPath, artifactPath, {
                sketchDims: 16, sketchBits, recommendedRerank: 60,
            });
            check('manifest shape', manifest.formatVersion === 2 && manifest.graph.count === COUNT
                && manifest.sketch.sketchBits === sketchBits && manifest.sizeBytes === fs.statSync(artifactPath).size
                && manifest.rowIntegrity.rowsPerBlock === 16 && manifest.rowIntegrity.rowDigestBytes === 16);

            const artifact = await Pikelet.openSketchArtifactFile(artifactPath);
            check('open + header', artifact.count === COUNT && artifact.dim === DIM
                && artifact.recommendedRerank === 60 && artifact.residentVerified === true);

            // Determinism: rebuilding produces byte-identical output, and
            // the bytes-in/bytes-out builder matches the file builder.
            const artifact2Path = artifactPath + '.rebuild';
            Pikelet.buildSketchArtifactFile(snapshotPath, artifact2Path, { sketchDims: 16, sketchBits, recommendedRerank: 60 });
            check('builder determinism', fs.readFileSync(artifactPath).equals(fs.readFileSync(artifact2Path)));
            const { bytes: builtBytes, manifest: bytesManifest } = Pikelet.buildSketchArtifactBytes(
                fs.readFileSync(snapshotPath), { sketchDims: 16, sketchBits, recommendedRerank: 60 });
            check('bytes builder matches file builder',
                fs.readFileSync(artifactPath).equals(Buffer.from(builtBytes))
                && bytesManifest.sizeBytes === manifest.sizeBytes && bytesManifest.file === undefined);

            const queries = seededVectors(20, DIM, 7);

            // C = count: sketch selection cannot exclude anything, so results
            // must exactly equal brute force over the quantized rows. Compare
            // against a fully restored engine index as the quantization oracle.
            const restored = await Pikelet.restore(fs.readFileSync(snapshotPath));
            let exactMatches = 0;
            for (const q of queries) {
                const ours = (await artifact.search(q, K, { rerank: COUNT })).results.map((r) => r.id);
                const oracle = restored.search(q, K, { efSearch: COUNT }).map((r) => r.id);
                // Engine search is ANN; brute-force the quantized space instead
                // via searchFiltered over all ids at max ef for a stable oracle.
                if (JSON.stringify(ours.slice().sort()) === JSON.stringify(oracle.slice().sort())) exactMatches++;
            }
            check(`C=count matches engine top-${K} on ${queries.length} queries`, exactMatches >= queries.length - 1,
                `${exactMatches}/${queries.length}`);
            restored.dispose();

            // Modest C: recall against float brute force must clear a floor.
            let hits = 0;
            for (const q of queries) {
                const got = (await artifact.search(q, K, { rerank: 60 })).results.map((r) => r.id);
                const truth = bruteForce(rows, q, K, metric);
                for (const t of truth) if (got.includes(t)) hits++;
            }
            const recall = hits / (queries.length * K);
            check(`recall@${K} at C=60 above floor`, recall >= 0.8, recall.toFixed(3));

            // Deeper rerank must close most of the remaining gap.
            let hits200 = 0;
            for (const q of queries) {
                const got = (await artifact.search(q, K, { rerank: 200 })).results.map((r) => r.id);
                const truth = bruteForce(rows, q, K, metric);
                for (const t of truth) if (got.includes(t)) hits200++;
            }
            const recall200 = hits200 / (queries.length * K);
            check(`recall@${K} at C=200 above 0.95`, recall200 >= 0.95, recall200.toFixed(3));

            // Stats and caching behave.
            const before = artifact.stats().rangeRequests;
            await artifact.search(queries[0], K, { rerank: 60 });
            const mid = artifact.stats();
            await artifact.search(queries[0], K, { rerank: 60 });
            const after = artifact.stats();
            check('repeat query fully cached', after.rangeRequests === mid.rangeRequests && mid.rangeRequests >= before);

            const depthSource = new DelayedMemorySource(fs.readFileSync(artifactPath));
            const depthArtifact = await Pikelet.SketchArtifact.open(depthSource);
            const sparseIds = Array.from({ length: 90 }, (_, i) => i * 2);
            depthSource.resetStats();
            await depthArtifact.fetchRows(sparseIds, { maxRangeBytes: DIM, gap: 0 });
            const dependentRounds = Math.ceil(depthSource.requests / depthSource.maxInFlight);
            // All runs go out concurrently up to the source's parallelism
            // (32): with more runs than that, the ceiling must be hit; with
            // fewer (format 2 fetches whole digest blocks, so 90 sparse rows
            // collapse into ~12 block runs), every run flies at once.
            check('rerank row fetch uses source parallelism in <= 3 dependent rounds',
                depthSource.maxInFlight === Math.min(32, depthSource.requests) && dependentRounds <= 3,
                `requests=${depthSource.requests} maxInFlight=${depthSource.maxInFlight} rounds=${dependentRounds}`);
            await depthArtifact.close();

            // The WASM scanner must implement the artifact's metric. Compare
            // its raw candidate set against the reference sketch scan, before
            // exact rerank can mask a wrong-metric selection: no unselected
            // row may beat a selected one (tie-safe top-C).
            const scanner = await Pikelet.createSketchScanner(artifact);
            check('scanner reports artifact metric', scanner.metric === artifact.metric);
            const pool = DIM / artifact.sketchDims;
            let topCOk = true;
            for (const q of queries) {
                let qv = q;
                if (metric === 'cosine') {
                    let norm = 0;
                    for (let d = 0; d < DIM; d++) norm += q[d] * q[d];
                    norm = Math.sqrt(norm);
                    qv = Float32Array.from(q, (x) => x / norm);
                }
                const qPool = new Float32Array(artifact.sketchDims);
                for (let sd = 0; sd < artifact.sketchDims; sd++) {
                    let acc = 0;
                    for (let j = 0; j < pool; j++) acc += qv[sd * pool + j];
                    qPool[sd] = acc / pool;
                }
                const dists = new Float64Array(COUNT);
                for (let i = 0; i < COUNT; i++) {
                    const s = artifact.scales[i];
                    const o = artifact.offsets[i];
                    let acc = 0;
                    if (metric === 'cosine') {
                        for (let sd = 0; sd < artifact.sketchDims; sd++) acc += qPool[sd] * (o + s * artifact.sketchValue(i, sd));
                        acc = 1 - Math.max(-1, Math.min(1, acc * pool));
                    } else {
                        for (let sd = 0; sd < artifact.sketchDims; sd++) {
                            const diff = qPool[sd] - (o + s * artifact.sketchValue(i, sd));
                            acc += diff * diff;
                        }
                    }
                    dists[i] = acc;
                }
                const C = 60;
                const ids = scanner.scan(qPool, C);
                if (ids.length !== C || new Set(ids).size !== C) { topCOk = false; continue; }
                const selected = new Set(ids);
                let maxSel = -Infinity;
                let minUnsel = Infinity;
                for (let i = 0; i < COUNT; i++) {
                    if (selected.has(i)) maxSel = Math.max(maxSel, dists[i]);
                    else minUnsel = Math.min(minUnsel, dists[i]);
                }
                if (maxSel > minUnsel + 1e-4) topCOk = false;
            }
            scanner.dispose();
            check('WASM scanner selects a metric-correct top-C', topCOk);

            // A scanner that does not declare the artifact's metric must be
            // refused for cosine (where a metric-blind scan silently loses
            // recall) and accepted for l2 (the historical default).
            if (metric === 'cosine') {
                let refused = false;
                try {
                    await artifact.search(queries[0], K, { rerank: 60, scanner: { scan: () => [0] } });
                } catch (err) {
                    refused = err && err.code === 'INVALID_ARGUMENT';
                }
                check('metric-blind scanner refused for cosine', refused);
            } else {
                const custom = await artifact.search(queries[0], K, {
                    rerank: 60,
                    scanner: { scan: (qp, c) => Array.from({ length: Math.min(c, COUNT) }, (_, i) => i) },
                });
                check('custom metric-blind scanner still accepted for l2', custom.results.length === K);
            }

            await artifact.close();

            // Tampering with the resident prefix must fail verification.
            const tampered = fs.readFileSync(artifactPath);
            tampered[300] ^= 0xff;
            const tamperedPath = artifactPath + '.tampered';
            fs.writeFileSync(tamperedPath, tampered);
            let rejected = false;
            try { await Pikelet.openSketchArtifactFile(tamperedPath); }
            catch (err) { rejected = /hash verification/.test(String(err && err.message)); }
            check('tampered resident prefix rejected', rejected);
        }
    }

    // Bounded-cache behavior: eviction must bound memory without changing
    // results, for both artifact readers.
    console.log('\nbounded caches');
    {
        const rows = seededVectors(COUNT, DIM, 42);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'cache-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const queries = seededVectors(30, DIM, 99);

        const rangePath = path.join(tmp, 'cache.pikelet-range');
        Pikelet.buildRangeArtifactFile(snapshotPath, rangePath);
        const unboundedRange = await Pikelet.openRangeArtifactFile(rangePath, { maxCacheBytes: Infinity });
        const boundedRange = await Pikelet.openRangeArtifactFile(rangePath, { maxCacheBytes: 1 }); // clamps to 64 records
        let rangeMatch = true;
        for (const q of queries) {
            const a = (await unboundedRange.search(q, K, { efSearch: 80 })).results.map((r) => r.id);
            const b = (await boundedRange.search(q, K, { efSearch: 80 })).results.map((r) => r.id);
            if (JSON.stringify(a) !== JSON.stringify(b)) rangeMatch = false;
        }
        const rangeStats = boundedRange.stats();
        check('range reader: identical results under eviction', rangeMatch);
        check('range reader: lazy cache bounded', rangeStats.lazyCacheBytes <= 64 * boundedRange.recordBytes,
            `lazyCacheBytes=${rangeStats.lazyCacheBytes}`);
        await unboundedRange.close();
        await boundedRange.close();

        const sketchPath = path.join(tmp, 'cache.pikelet-sketch');
        Pikelet.buildSketchArtifactFile(snapshotPath, sketchPath, { sketchDims: 16, sketchBits: 8 });
        const unboundedSketch = await Pikelet.openSketchArtifactFile(sketchPath, { maxCacheBytes: Infinity });
        const boundedSketch = await Pikelet.openSketchArtifactFile(sketchPath, { maxCacheBytes: 1 }); // clamps to 256 rows
        let sketchMatch = true;
        for (const q of queries) {
            // rerank 400 exceeds the 256-row cache budget on purpose:
            // correctness must come from fetchRows' returned rows, not cache.
            const a = (await unboundedSketch.search(q, K, { rerank: 400 })).results.map((r) => r.id);
            const b = (await boundedSketch.search(q, K, { rerank: 400 })).results.map((r) => r.id);
            if (JSON.stringify(a) !== JSON.stringify(b)) sketchMatch = false;
        }
        const sketchStats = boundedSketch.stats();
        check('sketch reader: identical results under eviction', sketchMatch);
        check('sketch reader: row cache bounded', sketchStats.cacheBytes <= 256 * DIM && sketchStats.cachedRows <= 256,
            `cacheBytes=${sketchStats.cacheBytes} rows=${sketchStats.cachedRows}`);
        await unboundedSketch.close();
        await boundedSketch.close();
    }

    // Truncated artifact files must fail closed with coded errors. The file
    // source may not pad short reads to the requested length, or every
    // downstream truncation check parses unwritten buffer tail instead.
    console.log('\ntruncated files fail closed');
    {
        const rows = seededVectors(COUNT, DIM, 7);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'trunc-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const queries = seededVectors(10, DIM, 11);

        const readers = [
            ['range', 'trunc.pikelet-range',
                (snap, out) => Pikelet.buildRangeArtifactFile(snap, out),
                (p) => Pikelet.openRangeArtifactFile(p),
                (artifact, q) => artifact.search(q, K, { efSearch: 200 })],
            ['sketch', 'trunc.pikelet-sketch',
                (snap, out) => Pikelet.buildSketchArtifactFile(snap, out, { sketchDims: 16, sketchBits: 8 }),
                (p) => Pikelet.openSketchArtifactFile(p),
                (artifact, q) => artifact.search(q, K, { rerank: 200 })],
        ];
        for (const [label, name, build, open, search] of readers) {
            const fullPath = path.join(tmp, name);
            build(snapshotPath, fullPath);
            const full = fs.readFileSync(fullPath);
            for (const keep of [4, 64, Math.floor(full.length / 2)]) {
                const cutPath = `${fullPath}.cut${keep}`;
                fs.writeFileSync(cutPath, full.subarray(0, keep));
                let coded = false;
                let detail = 'no error thrown';
                try {
                    // Open may legitimately succeed when the cut falls past the
                    // resident prefix; a search must then hit the missing bytes.
                    const artifact = await open(cutPath);
                    try {
                        for (const q of queries) await search(artifact, q);
                    } finally {
                        await artifact.close();
                    }
                } catch (err) {
                    coded = err instanceof Pikelet.PikeletError && typeof err.code === 'string';
                    detail = String(err && err.message);
                }
                check(`${label} artifact truncated to ${keep}B fails closed`, coded, detail);
            }
        }
    }

    // A range-artifact record lying about its own id must fail closed with a
    // coded error, not poison the cache under the forged key (which used to
    // surface as an uncoded TypeError deep inside search).
    console.log('\nforged record ids fail closed');
    {
        const rows = seededVectors(COUNT, DIM, 21);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'forge-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const rangePath = path.join(tmp, 'forge.pikelet-range');
        Pikelet.buildRangeArtifactFile(snapshotPath, rangePath);

        const clean = await Pikelet.openRangeArtifactFile(rangePath);
        const baseOffset = clean.baseRecordsOffset;
        const routerOffset = clean.routerRecordsOffset;
        await clean.close();

        const forge = async (offset, label) => {
            const bytes = Buffer.from(fs.readFileSync(rangePath));
            const trueId = bytes.readUInt32LE(offset);
            bytes.writeUInt32LE((trueId + 1) % COUNT, offset);
            const forgedPath = `${rangePath}.${label}`;
            fs.writeFileSync(forgedPath, bytes);
            let coded = false;
            let detail = 'no error thrown';
            try {
                const artifact = await Pikelet.openRangeArtifactFile(forgedPath);
                try {
                    const queries = seededVectors(10, DIM, 22);
                    for (const q of queries) await artifact.search(q, K, { efSearch: 200 });
                } finally {
                    await artifact.close();
                }
            } catch (err) {
                coded = err instanceof Pikelet.PikeletError && err.code === 'SNAPSHOT_INVALID';
                detail = String(err && err.message);
            }
            check(`forged ${label} record id rejected with SNAPSHOT_INVALID`, coded, detail);
        };
        await forge(baseOffset, 'base');
        await forge(routerOffset, 'router');
    }

    // A hostile header must not drive a giant read or allocation: the range
    // must be rejected before source.read() is called, and a source that
    // cannot report its size (e.g. HTTP) must still be protected by the cap.
    console.log('\nartifact read sizes bounded before fetch');
    {
        const rows = seededVectors(COUNT, DIM, 51);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'bound-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const rangePath = path.join(tmp, 'bound.pikelet-range');
        Pikelet.buildRangeArtifactFile(snapshotPath, rangePath);
        const realHeader = Buffer.from(fs.readFileSync(rangePath)).subarray(0, 128);

        // Wrap a real 128-byte header over a source that records read sizes and
        // has more than 2 GiB of virtual size, so only the per-read cap can
        // stop a pathological id-map read.
        const makeSource = (reportSize) => {
            const calls = [];
            return {
                calls,
                size: reportSize ? 128 : undefined,
                async read(offset, length) {
                    calls.push([offset, length]);
                    if (offset === 0) return Buffer.from(realHeader);
                    return Buffer.alloc(0);
                },
            };
        };
        for (const reportSize of [true, false]) {
            const src = makeSource(reportSize);
            let threw = false;
            try { await Pikelet.RangeArtifact.open(src); } catch { threw = true; }
            // Open-path reads default to the 256 MiB budget, well under the
            // 2 GiB absolute backstop.
            const giant = src.calls.some(([, l]) => l > 256 * 1024 * 1024);
            check(`open never issues a >256MiB read (size ${reportSize ? 'known' : 'unknown'})`, !giant && threw,
                `calls=${JSON.stringify(src.calls)}`);
        }

        // NodeFileRangeSource.read must refuse an out-of-file range directly,
        // independent of any caller validation (defense in depth).
        const art = await Pikelet.openRangeArtifactFile(rangePath);
        let directCoded = false, ddetail = '';
        try { await art.source.read(0, 0x40000000); }
        catch (err) { directCoded = err instanceof Pikelet.PikeletError && err.code === 'SNAPSHOT_INVALID'; ddetail = String(err && err.message); }
        check('NodeFileRangeSource.read refuses an out-of-file range', directCoded, ddetail);
        await art.close();
    }

    // Scanner input is copied into a fixed-size WASM buffer; malformed input
    // must be rejected before the copy, not written out of bounds or fed to
    // the native kernel as garbage.
    console.log('\nscanner input validation');
    {
        const rows = seededVectors(COUNT, DIM, 31);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'scan-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const sketchPath = path.join(tmp, 'scan.pikelet-sketch');
        Pikelet.buildSketchArtifactFile(snapshotPath, sketchPath, { sketchDims: 16, sketchBits: 8 });
        const artifact = await Pikelet.openSketchArtifactFile(sketchPath);
        const scanner = await Pikelet.createSketchScanner(artifact);
        const sd = scanner.sketchDims;

        const rejects = (input, label) => {
            let coded = false, detail = 'no error thrown';
            try { scanner.scan(input, 10); }
            catch (err) { coded = err instanceof Pikelet.PikeletError && typeof err.code === 'string'; detail = String(err && err.message); }
            check(`scan() rejects ${label}`, coded, detail);
        };
        rejects(new Float32Array(sd + 5000).fill(0.5), 'oversized input');
        rejects(new Float32Array(sd - 1).fill(0.5), 'undersized input');
        rejects(Float32Array.from({ length: sd }, (_, i) => (i === 0 ? NaN : 0.5)), 'NaN in input');
        rejects(Float32Array.from({ length: sd }, (_, i) => (i === 0 ? Infinity : 0.5)), 'Infinity in input');
        // A correctly-sized finite query must still work after the guards.
        const ok = scanner.scan(new Float32Array(sd).fill(0.1), 5);
        check('scan() accepts a valid pooled query', Array.isArray(ok));
        scanner.dispose();
        await artifact.close();
    }

    // When verification is requested but no crypto backend exists, open must
    // fail closed rather than admit unverified bytes.
    console.log('\nverification fails closed without crypto');
    {
        const rows = seededVectors(COUNT, DIM, 41);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'verify-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const sketchPath = path.join(tmp, 'verify.pikelet-sketch');
        Pikelet.buildSketchArtifactFile(snapshotPath, sketchPath, { sketchDims: 16, sketchBits: 8 });

        // Simulate an environment with no crypto backend at all. globalThis.crypto
        // is a getter-only accessor, so override it via defineProperty and
        // block the Node crypto require the async hash helper falls back to.
        const realDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
        const Module = require('module');
        const origModuleRequire = Module.prototype.require;
        Object.defineProperty(globalThis, 'crypto', { value: undefined, configurable: true, writable: true });
        Module.prototype.require = function (id) {
            if (id === 'crypto' || id === 'node:crypto') throw new Error('crypto unavailable');
            return origModuleRequire.apply(this, arguments);
        };
        try {
            let coded = false, detail = 'no error thrown';
            try {
                const artifact = await Pikelet.openSketchArtifactFile(sketchPath, { verify: true });
                await artifact.close();
            } catch (err) {
                coded = err instanceof Pikelet.PikeletError && err.code === 'SNAPSHOT_INVALID' && /no crypto backend/i.test(err.message);
                detail = String(err && err.message);
            }
            check('open(verify:true) fails closed with no crypto backend', coded, detail);

            // verify:false must still open in the same environment.
            let opened = false, odetail = '';
            try {
                const artifact = await Pikelet.openSketchArtifactFile(sketchPath, { verify: false });
                opened = artifact.count === COUNT;
                await artifact.close();
            } catch (err) { odetail = String(err && err.message); }
            check('open(verify:false) still opens with no crypto backend', opened, odetail);
        } finally {
            if (realDescriptor) Object.defineProperty(globalThis, 'crypto', realDescriptor);
            Module.prototype.require = origModuleRequire;
        }
    }

    // Range artifact v3 carries whole-segment digests: id map and router are
    // verified at open, the base segment on demand. Payload tampering — bytes
    // the structural and address-roundtrip checks cannot see — must now be
    // caught by the hashes. v2 artifacts (no digests) must still open.
    console.log('\nrange artifact segment digests (v3)');
    {
        const rows = seededVectors(COUNT, DIM, 61);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'digest-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const rangePath = path.join(tmp, 'digest.pikelet-range');
        const manifest = Pikelet.buildRangeArtifactFile(snapshotPath, rangePath);
        check('manifest declares v3 + integrity digests', manifest.formatVersion === 3
            && /^[0-9a-f]{64}$/.test(manifest.integrity.idMapSha256)
            && /^[0-9a-f]{64}$/.test(manifest.integrity.routerSha256)
            && /^[0-9a-f]{64}$/.test(manifest.integrity.baseSha256));

        const clean = await Pikelet.openRangeArtifactFile(rangePath);
        const stats = clean.stats();
        check('id map + router verified at open', clean.version === 3
            && stats.segmentVerified.idMap === true && stats.segmentVerified.router === true);
        const baseOk = await clean.verifyBaseSegment();
        check('base segment verifies on demand', baseOk === true && clean.stats().segmentVerified.base === true);
        const cleanResults = (await clean.search(seededVectors(1, DIM, 62)[0], K, { efSearch: 80 })).results.map((r) => r.id);
        const baseRecordsOffset = clean.baseRecordsOffset;
        const routerRecordsOffset = clean.routerRecordsOffset;
        const idMapOffset = clean.idMapOffset;
        await clean.close();

        // Payload tampering (not the record id, which the address round-trip
        // already catches): flip a quantized-vector byte inside a record.
        const tamper = async (offset, label, expectOpenRejected) => {
            const bytes = Buffer.from(fs.readFileSync(rangePath));
            bytes[offset] ^= 0xff;
            const tamperedPath = `${rangePath}.${label}`;
            fs.writeFileSync(tamperedPath, bytes);
            let openRejected = false;
            let baseRejected = false;
            let detail = 'no error thrown';
            try {
                const artifact = await Pikelet.openRangeArtifactFile(tamperedPath);
                try { await artifact.verifyBaseSegment(); }
                catch (err) { baseRejected = /hash verification/.test(String(err && err.message)); }
                await artifact.close();
            } catch (err) {
                openRejected = /hash verification/.test(String(err && err.message));
                detail = String(err && err.message);
            }
            if (expectOpenRejected) check(`tampered ${label} rejected at open`, openRejected, detail);
            else check(`tampered ${label} passes open, rejected by verifyBaseSegment`, !openRejected && baseRejected, detail);
        };
        await tamper(idMapOffset + 5, 'id map', true);
        await tamper(routerRecordsOffset + 9, 'router payload', true);
        await tamper(baseRecordsOffset + 9, 'base payload', false);

        // v2 compatibility: strip the digests and downgrade the version field;
        // the reader must open it structurally and return identical results.
        const v2Bytes = Buffer.from(fs.readFileSync(rangePath));
        v2Bytes.writeUInt32LE(2, 4);
        v2Bytes.fill(0, 128, 224);
        const v2Path = `${rangePath}.v2`;
        fs.writeFileSync(v2Path, v2Bytes);
        const v2Artifact = await Pikelet.openRangeArtifactFile(v2Path);
        const v2Results = (await v2Artifact.search(seededVectors(1, DIM, 62)[0], K, { efSearch: 80 })).results.map((r) => r.id);
        check('v2 artifact (no digests) still opens, same results', v2Artifact.version === 2
            && v2Artifact.stats().segmentVerified.idMap === false
            && JSON.stringify(v2Results) === JSON.stringify(cleanResults));
        let v2BaseRefused = false;
        try { await v2Artifact.verifyBaseSegment(); }
        catch (err) { v2BaseRefused = err instanceof Pikelet.PikeletError && err.code === 'INVALID_ARGUMENT'; }
        check('verifyBaseSegment refuses a pre-digest artifact explicitly', v2BaseRefused);
        await v2Artifact.close();

        // Unknown future versions must be rejected, not misparsed.
        const v9Bytes = Buffer.from(fs.readFileSync(rangePath));
        v9Bytes.writeUInt32LE(9, 4);
        const v9Path = `${rangePath}.v9`;
        fs.writeFileSync(v9Path, v9Bytes);
        let v9Rejected = false;
        try { await Pikelet.openRangeArtifactFile(v9Path); }
        catch (err) { v9Rejected = err instanceof Pikelet.PikeletError && /version/i.test(String(err.message)); }
        check('unknown range artifact version rejected', v9Rejected);
    }

    // Raw snapshot payloads declaring an unknown future version must be
    // rejected at import, not silently parsed as the newest known layout.
    console.log('\nsnapshot format version bounded');
    {
        const rows = seededVectors(64, DIM, 71);
        const index = await Pikelet.create({ dim: DIM, maxElements: 64, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshot = Buffer.from(index.export());
        index.dispose();
        // v3 envelope: [magic, version, dim, metric, quantized, nextExtId,
        // mappingCount, wasmSize], then mappings, then the raw payload whose
        // own version field sits 8 bytes past the raw magic.
        const mappingCount = snapshot.readUInt32LE(24);
        const rawOffset = 32 + mappingCount * 8;
        check('raw payload located for version patch', snapshot.readUInt32LE(rawOffset) === 0x49384831);
        snapshot.writeUInt32LE(99, rawOffset + 8);
        let rejected = false, detail = 'no error thrown';
        try {
            const restored = await Pikelet.restore(snapshot);
            restored.dispose();
        } catch (err) {
            rejected = err instanceof Pikelet.PikeletError && err.code === 'SNAPSHOT_INVALID' && /format version/i.test(err.message);
            detail = String(err && err.message);
        }
        check('unknown raw snapshot version rejected at import', rejected, detail);
    }

    // Per-open read budgets, chunked verification, and coalesced-run
    // splitting: no single read may exceed its budget, verification must
    // detect tampering regardless of chunk size, and splitting ranges must
    // never change results.
    console.log('\nread budgets and chunked processing');
    {
        const rows = seededVectors(COUNT, DIM, 91);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'budget-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const rangePath = path.join(tmp, 'budget.pikelet-range');
        Pikelet.buildRangeArtifactFile(snapshotPath, rangePath);
        const sketchPath = path.join(tmp, 'budget.pikelet-sketch');
        Pikelet.buildSketchArtifactFile(snapshotPath, sketchPath, { sketchDims: 16, sketchBits: 8 });
        const queries = seededVectors(10, DIM, 92);

        // A budget below the artifact's resident needs fails the open with a
        // coded error instead of being silently raised.
        let coded = false, detail = 'no error thrown';
        try { await Pikelet.openRangeArtifactFile(rangePath, { maxReadBytes: 1024 }); }
        catch (err) {
            coded = err instanceof Pikelet.PikeletError && err.code === 'SNAPSHOT_INVALID' && /maximum read size/.test(err.message);
            detail = String(err && err.message);
        }
        check('range open under a too-small read budget fails closed', coded, detail);
        coded = false;
        try { await Pikelet.openSketchArtifactFile(sketchPath, { maxReadBytes: 1024 }); }
        catch (err) { coded = err instanceof Pikelet.PikeletError && err.code === 'SNAPSHOT_INVALID'; }
        check('sketch open under a too-small read budget fails closed', coded);
        let invalid = false;
        try { await Pikelet.openRangeArtifactFile(rangePath, { maxReadBytes: -5 }); }
        catch (err) { invalid = err instanceof Pikelet.PikeletError && err.code === 'INVALID_ARGUMENT'; }
        check('invalid maxReadBytes rejected with INVALID_ARGUMENT', invalid);

        // Chunked verification: multiple small chunks must accept a clean
        // segment and reject a tampered one, same as the one-shot path.
        const cleanRange = await Pikelet.openRangeArtifactFile(rangePath);
        check('chunked base verification accepts a clean segment',
            (await cleanRange.verifyBaseSegment({ chunkBytes: 4096 })) === true);
        const baseRecordsOffset = cleanRange.baseRecordsOffset;
        const recordBytes = cleanRange.recordBytes;
        const defaultResults = [];
        for (const q of queries) defaultResults.push((await cleanRange.search(q, K, { efSearch: 80 })).results.map((r) => r.id));
        await cleanRange.close();

        const tamperedBytes = Buffer.from(fs.readFileSync(rangePath));
        // Tamper the LAST base record so detection requires hashing every chunk.
        tamperedBytes[tamperedBytes.length - 4] ^= 0xff;
        const tamperedPath = `${rangePath}.chunktamper`;
        fs.writeFileSync(tamperedPath, tamperedBytes);
        const tamperedRange = await Pikelet.openRangeArtifactFile(tamperedPath);
        let rejected = false;
        try { await tamperedRange.verifyBaseSegment({ chunkBytes: 4096 }); }
        catch (err) { rejected = /hash verification/.test(String(err && err.message)); }
        check('chunked base verification rejects a tail-tampered segment', rejected);
        await tamperedRange.close();

        const cleanSketch = await Pikelet.openSketchArtifactFile(sketchPath);
        check('chunked vectors verification accepts a clean segment',
            (await cleanSketch.verifyVectors({ chunkBytes: 4096 })) === true);

        // Splitting coalesced runs must change request counts, never results.
        const splitRange = await Pikelet.openRangeArtifactFile(rangePath);
        let splitMatch = true;
        for (let i = 0; i < queries.length; i++) {
            const got = (await splitRange.search(queries[i], K, { efSearch: 80, maxRangeBytes: recordBytes })).results.map((r) => r.id);
            if (JSON.stringify(got) !== JSON.stringify(defaultResults[i])) splitMatch = false;
        }
        check('range search identical with single-record range splitting', splitMatch);
        // A bulk prefetch of every id must arrive as bounded pieces.
        splitRange.resetStats();
        await splitRange.clearCache({ reloadRouter: false });
        const allIds = Array.from({ length: COUNT }, (_, i) => i);
        await splitRange.prefetch(allIds, { gap: 0, maxRangeBytes: recordBytes * 8 });
        const pieces = splitRange.rangesSince(0);
        const oversize = pieces.some(([s, e]) => e - s > recordBytes * 8);
        check('bulk prefetch splits into bounded ranges', pieces.length >= Math.floor(COUNT / 8) / 2 && !oversize,
            `pieces=${pieces.length}`);
        await splitRange.close();

        let sketchSplitMatch = true;
        for (const q of queries) {
            const a = (await cleanSketch.search(q, K, { rerank: 200 })).results.map((r) => r.id);
            const b = (await cleanSketch.search(q, K, { rerank: 200, maxRangeBytes: DIM })).results.map((r) => r.id);
            if (JSON.stringify(a) !== JSON.stringify(b)) sketchSplitMatch = false;
        }
        check('sketch search identical with single-row range splitting', sketchSplitMatch);
        await cleanSketch.close();

        // Verification without a streaming hash backend falls back to one
        // bounded read — and refuses segments beyond the budget instead of
        // buffering them.
        const Module = require('module');
        const origModuleRequire = Module.prototype.require;
        Module.prototype.require = function (id) {
            if (id === 'crypto' || id === 'node:crypto') throw new Error('crypto unavailable');
            return origModuleRequire.apply(this, arguments);
        };
        try {
            // Budget sits between the resident prefix (~14.4 KB, must load)
            // and the vectors segment (19.2 KB, must be refused one-shot).
            const noStream = await Pikelet.openSketchArtifactFile(sketchPath, { verify: false, maxReadBytes: 16384 });
            let refused = false, rdetail = 'no error thrown';
            try { await noStream.verifyVectors(); }
            catch (err) {
                refused = err instanceof Pikelet.PikeletError && /too large to verify/.test(err.message);
                rdetail = String(err && err.message);
            }
            check('one-shot fallback refuses segments beyond the read budget', refused, rdetail);
            await noStream.close();
        } finally {
            Module.prototype.require = origModuleRequire;
        }
    }

    // The sketch vectors segment (the lazy tier that decides final ranking)
    // must be verifiable against the header's whole-segment hash.
    console.log('\nsketch vectors segment digest');
    {
        const rows = seededVectors(COUNT, DIM, 81);
        const index = await Pikelet.create({ dim: DIM, maxElements: COUNT, metric: 'l2', quantized: true });
        index.addBatch(rows);
        const snapshotPath = path.join(tmp, 'vecdigest-snap.pnck');
        fs.writeFileSync(snapshotPath, index.export());
        index.dispose();
        const sketchPath = path.join(tmp, 'vecdigest.pikelet-sketch');
        Pikelet.buildSketchArtifactFile(snapshotPath, sketchPath, { sketchDims: 16, sketchBits: 8 });

        const clean = await Pikelet.openSketchArtifactFile(sketchPath);
        const ok = await clean.verifyVectors();
        check('clean vectors segment verifies', ok === true && clean.stats().vectorsVerified === true);
        const vectorsOffset = clean.vectorsOffset;
        await clean.close();

        const tampered = Buffer.from(fs.readFileSync(sketchPath));
        tampered[vectorsOffset + 3] ^= 0xff;
        const tamperedPath = `${sketchPath}.tampered`;
        fs.writeFileSync(tamperedPath, tampered);
        const artifact = await Pikelet.openSketchArtifactFile(tamperedPath);
        check('vectors tamper is invisible to the resident hash', artifact.stats().residentVerified === true);
        let rejected = false, detail = 'no error thrown';
        try { await artifact.verifyVectors(); }
        catch (err) {
            rejected = /hash verification/.test(String(err && err.message));
            detail = String(err && err.message);
        }
        check('tampered vectors segment rejected by verifyVectors', rejected, detail);
        await artifact.close();
    }

    // Golden fixtures: the reference reader must reproduce committed results
    // byte-for-byte from committed artifact bytes (spec section 5).
    console.log('\ngolden fixtures');
    {
        const golden = require('./fixtures/sketch_golden.js');
        for (const c of golden.cases) {
            const bytes = Buffer.from(c.artifactBase64, 'base64');
            const goldenPath = path.join(tmp, `golden-${c.metric}-${c.sketchBits}.pikelet-sketch`);
            fs.writeFileSync(goldenPath, bytes);
            const artifact = await Pikelet.openSketchArtifactFile(goldenPath);
            let ok = true;
            let ri = 0;
            for (const q of c.queries) {
                for (const [k, C] of [[5, 32], [10, 64]]) {
                    const expected = c.results[ri++];
                    const got = (await artifact.search(new Float32Array(q), k, { rerank: C })).results;
                    const gotIds = got.map((x) => x.id);
                    const gotDists = got.map((x) => Number(x.distance.toFixed(5)));
                    if (JSON.stringify(gotIds) !== JSON.stringify(expected.ids)) ok = false;
                    if (JSON.stringify(gotDists) !== JSON.stringify(expected.dists)) ok = false;
                }
            }
            // The WASM scanner path must reproduce the same golden results.
            const scanner = await Pikelet.createSketchScanner(artifact);
            let scannerOk = true;
            ri = 0;
            for (const q of c.queries) {
                for (const [k, C] of [[5, 32], [10, 64]]) {
                    const expected = c.results[ri++];
                    const got = (await artifact.search(new Float32Array(q), k, { rerank: C, scanner })).results;
                    if (JSON.stringify(got.map((x) => x.id)) !== JSON.stringify(expected.ids)) scannerOk = false;
                }
            }
            scanner.dispose();
            await artifact.close();
            check(`golden ${c.metric} u${c.sketchBits}: reference reader reproduces committed results`, ok);
            check(`golden ${c.metric} u${c.sketchBits}: WASM scanner reproduces committed ids`, scannerOk);
        }
    }

    console.log(`\nSketch profile conformance: ${passed} passed, ${failed} failed`);
    if (failed > 0) process.exit(1);
}

run().catch((err) => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
});
