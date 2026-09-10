// Complete-profile reader conformance (spec/COMPLETE_PROFILE.md), deterministic
// and hermetic: every fixture is built in this process from seeded data or
// from the committed examples/03 assets, so the suite runs in CI with no
// downloads and no model weights.
//
//   A. synthetic kind-2 artifact, format 2 (per-record corpus integrity):
//      open/query/hydrate, host-encoder verification (contract 4.4 mode 2),
//      per-record and page-table tamper detection, structural rejection of
//      hostile headers/tables, read budgets, truncation, unknown/duplicate
//      segments, format-version bounds;
//   B. the same corpus as a format-1 file: still opens, reports the
//      transitional whole-segment integrity, and a record tamper is NOT
//      detected there (the documented v1 stance);
//   C. kind-1 (student-inline) artifact compiled from examples/03's committed
//      assets: the 10 abstention goldens reproduce their labels, hydration
//      round-trips the source corpus, and the compile is byte-deterministic.
//
// Kind 3 (inline transformer) needs the 24 MiB weight blob and is covered by
// examples/05-one-file-search/test-inline.mjs, not here.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import {
    buildCorpusSegment, buildCorpusSegmentFromBuffers, buildQueryInterpSegment,
    assemblePancakeFile, buildLexicalSegment, PROFILE_V1, PROFILE_V2, sha256, canonicalJson,
} from '../complete/builder.mjs';
import { openPancakeFile, verifyHostEncoder } from '../complete/index.mjs';
import { openLexicalIndex } from '../complete/lexical.mjs';

const require = createRequire(import.meta.url);
const Pikelet = require('../pikelet.js');
const { exportSketchArtifact } = require('../pikelet-artifact.js');
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

let passed = 0;
let failed = 0;
function check(label, cond, detail = '') {
    if (cond) { passed++; console.log(`  ok: ${label}`); }
    else { failed++; console.log(`  FAIL: ${label}${detail ? ` — ${detail}` : ''}`); }
}
async function rejects(label, fn, pattern) {
    try {
        await fn();
        check(label, false, 'resolved instead of throwing');
    } catch (err) {
        const ok = pattern.test(String(err && err.message));
        check(label, ok, ok ? '' : `threw: ${String(err && err.message).slice(0, 160)}`);
    }
}

// ---------------------------------------------------------------------------
// Deterministic helpers
// ---------------------------------------------------------------------------
function mulberry32(a) {
    return function () {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
function fnv1a(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
    return h >>> 0;
}
const DIM = 16;
const COUNT = 300;
const PAGE_RECORDS = 64; // 5 pages, last one partial
// The "host encoder": text -> deterministic unit vector. Records embed as
// hostEncode(`rec ${i}`), so querying that text finds record i.
function hostEncode(text) {
    const rand = mulberry32(fnv1a(text));
    const v = new Float32Array(DIM);
    let n = 0;
    for (let d = 0; d < DIM; d++) { const x = rand() * 2 - 1; v[d] = x; n += x * x; }
    n = 1 / Math.sqrt(n);
    for (let d = 0; d < DIM; d++) v[d] *= n;
    return v;
}

function memorySource(bytes, { withSize = true } = {}) {
    return {
        size: withSize ? bytes.length : undefined,
        preferredParallelism: Infinity,
        preferredGapBytes: 0,
        reads: 0,
        async read(offset, length) { this.reads++; return bytes.subarray(offset, offset + length); },
        async close() {},
    };
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-complete-'));
function buildSyntheticSketch() {
    const qdata = new Uint8Array(COUNT * DIM);
    const scales = new Float32Array(COUNT);
    const offsets = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
        const v = hostEncode(`rec ${i}`);
        let mn = Infinity, mx = -Infinity;
        for (let d = 0; d < DIM; d++) { if (v[d] < mn) mn = v[d]; if (v[d] > mx) mx = v[d]; }
        const s = (mx - mn) / 255 || 1e-12;
        scales[i] = s; offsets[i] = mn;
        for (let d = 0; d < DIM; d++) {
            const b = Math.round((v[d] - mn) / s);
            qdata[i * DIM + d] = b < 0 ? 0 : b > 255 ? 255 : b;
        }
    }
    const sketchPath = path.join(tmp, 'synthetic.pancake-sketch');
    exportSketchArtifact({ dim: DIM, count: COUNT, metric: 1, qdata, scales, offsets }, sketchPath,
        { sketchDims: DIM, sketchBits: 8, recommendedRerank: 40 });
    return fs.readFileSync(sketchPath);
}
const SKETCH = buildSyntheticSketch();
const RECORDS = Array.from({ length: COUNT }, (_, i) => Buffer.from(JSON.stringify({
    title: `rec ${i}`,
    text: `record number ${i} carries some text so that tampering has room to land`,
    sourcePath: `docs/${Math.floor(i / 10)}/${i}.md`,
}), 'utf8'));
const TEST_VECTOR_TEXTS = ['rec 3', 'which treaty ended the first world war', 'rec 299'];
function kind2Declaration({ withVectors = true } = {}) {
    return {
        kind: 'external-transformers-v1',
        model: 'test/deterministic-hash-encoder',
        dim: DIM,
        pooling: 'none',
        normalized: true,
        maxTokens: 64,
        ...(withVectors ? {
            testVectors: TEST_VECTOR_TEXTS.map((text) => ({
                text,
                embedding: Array.from(hostEncode(text), (v) => Number(v.toFixed(6))),
                tolerance: 1e-3,
            })),
        } : {}),
    };
}
const CALIBRATION = Buffer.from(JSON.stringify({ kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '' }), 'utf8');
const EVALUATION = Buffer.from(JSON.stringify({ goldenQueries: [{ text: 'rec 17', expectedTopId: 17 }] }), 'utf8');

function buildSynthetic({ profile = PROFILE_V2, declaration = kind2Declaration(), manifestPatch = null, extraSegments = [], name = 'synthetic' } = {}) {
    const corpus = profile === PROFILE_V2 ? buildCorpusSegment(RECORDS, { pageRecords: PAGE_RECORDS })
        : { bytes: buildCorpusSegmentFromBuffers(RECORDS), corpus: { records: COUNT } };
    const segments = [
        { kind: 'index', bytes: SKETCH },
        { kind: 'corpus', bytes: corpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(declaration), 'utf8'), CALIBRATION) },
        { kind: 'evaluation', bytes: EVALUATION },
        ...extraSegments,
    ];
    const manifest = {
        profile,
        corpus: { ...corpus.corpus, provenance: null },
        dim: DIM,
        metric: 'cosine',
        encoder: { kind: declaration.kind, model: declaration.model },
        recommendedRerank: 40,
        sampleQueries: ['rec 17'],
    };
    if (manifestPatch) manifestPatch(manifest);
    const outPath = path.join(tmp, `${name}.pikelet`);
    const built = assemblePancakeFile(manifest, segments, outPath);
    return { bytes: fs.readFileSync(outPath), path: outPath, built };
}

// Locate a region inside the built file for tampering, by re-deriving the
// layout from the file's own (verified) header and manifest.
function layoutOf(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const manifestBytes = view.getUint32(8, true);
    const segmentCount = view.getUint32(12, true);
    const manifest = JSON.parse(bytes.subarray(64, 64 + manifestBytes).toString('utf8'));
    const tableAt = 64 + manifestBytes;
    const segments = {};
    for (let i = 0; i < segmentCount; i++) {
        const entry = tableAt + 48 * i;
        segments[manifest.segments[i].kind] = {
            offset: Number(view.getBigUint64(entry + 8, true)),
            length: Number(view.getBigUint64(entry + 16, true)),
        };
    }
    return { manifest, manifestBytes, segments };
}

// ---------------------------------------------------------------------------
// A. synthetic kind-2, format 2
// ---------------------------------------------------------------------------
console.log('A. synthetic kind-2 artifact, format 2');
const A = buildSynthetic();
{
    const src = memorySource(A.bytes);
    const search = await openPancakeFile(src, { encodeQuery: hostEncode });
    const info = search.info();
    check('opens: format 2 / pancake-complete-v2', info.formatVersion === 2 && info.profile === PROFILE_V2);
    check('host encoder verified against the declaration vectors', info.encoderVerified === true);
    check('corpus integrity reported per-record', info.corpusIntegrity === 'per-record-sha256');
    check('record count and dim', info.records === COUNT && info.dim === DIM);
    check('resident bytes include offsets and page table',
        info.residentBytes >= 8 * (COUNT + 1) + 32 * Math.ceil(COUNT / PAGE_RECORDS));

    const q = await search.query('rec 17', { k: 3 });
    check('query "rec 17" ranks record 17 first', q.results.length === 3 && q.results[0].id === 17, JSON.stringify(q.results.map((r) => r.id)));
    check('hydrated record matches the source bytes', q.results[0].title === 'rec 17' && q.results[0].sourcePath === 'docs/1/17.md');
    check('calibration absent -> unscored, results still returned', q.matchQuality === 'unscored');

    const readsBefore = src.reads;
    const far = await search.query('rec 250', { k: 2 });
    check('query in another digest page hydrates (page 3)', far.results[0].id === 250 && far.results[0].title === 'rec 250');
    const readsAfterFirst = src.reads;
    await search.record(251); // same page as 250: digest page is cached
    const readsAfterSecond = src.reads;
    check('a second record from a cached digest page costs exactly one read', readsAfterSecond - readsAfterFirst === 1,
        `first ${readsAfterFirst - readsBefore}, second ${readsAfterSecond - readsAfterFirst}`);

    const rec = await search.record(0);
    check('record(id) hydrates directly', rec.title === 'rec 0');
    await rejects('record(id) rejects ids outside the corpus', () => search.record(COUNT), /outside the corpus/);
    await rejects('query with empty text rejected', () => search.query('   '), /query text is required/);

    const ev = await search.evaluation();
    check('evaluation segment loads and digest-verifies', ev && ev.goldenQueries[0].expectedTopId === 17);
    const sized = info.fileBytes;
    check('info.fileBytes matches the file', sized === A.bytes.length);
    await search.close();
}

// Source without a size (browser-style HTTP source before init()).
{
    const search = await openPancakeFile(memorySource(A.bytes, { withSize: false }), { encodeQuery: hostEncode });
    const q = await search.query('rec 42', { k: 1 });
    check('opens and queries through a source that reports no size', q.results[0].id === 42);
    await search.close();
}

// Host-encoder verification (contract 4.4 mode 2).
{
    const perturbed = (text) => { const v = hostEncode(text); v[0] += 0.01; return v; };
    await rejects('a host encoder that disagrees with the verification vectors is refused at open',
        () => openPancakeFile(memorySource(A.bytes), { encodeQuery: perturbed }), /failed the artifact's verification vector/);
    await rejects('a host encoder of the wrong dimension is refused at open',
        () => openPancakeFile(memorySource(A.bytes), { encodeQuery: () => new Float32Array(DIM + 1) }), /must return a 16-dimensional vector/);
    await rejects('a host encoder returning non-finite components is refused',
        () => openPancakeFile(memorySource(A.bytes), { encodeQuery: () => { const v = hostEncode('x'); v[3] = NaN; return v; } }), /non-finite/);
    await rejects('encodeQuery must be a function',
        () => openPancakeFile(memorySource(A.bytes), { encodeQuery: 42 }), /must be a function/);
    const noHost = await openPancakeFile(memorySource(A.bytes));
    check('without a host encoder the artifact opens and surfaces the requirement', noHost.info().encoderVerified === null);
    await rejects('...but refuses queries until one is supplied', () => noHost.query('rec 1'), /pass options.encodeQuery/);
    await noHost.close();
    const unverified = await openPancakeFile(memorySource(A.bytes), { encodeQuery: perturbed, verifyEncoder: false });
    check('verifyEncoder:false serves a wrong encoder, marked unverified', unverified.info().encoderVerified === false);
    await unverified.close();
    // Host encoders that return plain arrays are accepted and converted.
    const arrays = await openPancakeFile(memorySource(A.bytes), { encodeQuery: (t) => Array.from(hostEncode(t)) });
    check('host encoder may return a plain number[]', (await arrays.query('rec 9', { k: 1 })).results[0].id === 9);
    await arrays.close();
    // verifyHostEncoder is exported for hosts that want to check before opening.
    const direct = await verifyHostEncoder(kind2Declaration(), hostEncode, DIM);
    check('verifyHostEncoder() reports the number of vectors checked', direct.checked === TEST_VECTOR_TEXTS.length);
}

// Declaration without verification vectors.
{
    const noVec = buildSynthetic({ declaration: kind2Declaration({ withVectors: false }), name: 'novectors' });
    await rejects('kind-2 declaration without vectors is refused with a host encoder by default',
        () => openPancakeFile(memorySource(noVec.bytes), { encodeQuery: hostEncode }), /without verification vectors/);
    const allowed = await openPancakeFile(memorySource(noVec.bytes), { encodeQuery: hostEncode, allowUnverifiedEncoder: true });
    check('allowUnverifiedEncoder serves it, marked unverified', allowed.info().encoderVerified === false
        && (await allowed.query('rec 5', { k: 1 })).results[0].id === 5);
    await allowed.close();
}

// Per-record integrity: tamper inside one record's bytes.
{
    const { manifest, segments } = layoutOf(A.bytes);
    const corpus = segments.corpus;
    const view = new DataView(A.bytes.buffer, A.bytes.byteOffset, A.bytes.byteLength);
    const offsetsAt = corpus.offset + 8;
    const recStart = (id) => corpus.offset + Number(view.getBigUint64(offsetsAt + 8 * id, true));
    const tampered = Buffer.from(A.bytes);
    // Flip one letter in record 17's text ('a' -> '`' keeps the JSON valid).
    const r17 = recStart(17);
    const r17end = recStart(18);
    const textAt = tampered.subarray(r17, r17end).indexOf('carries') + r17 + 1; // the 'a' of "carries"
    tampered[textAt] ^= 0x01;
    const t = await openPancakeFile(memorySource(tampered), { encodeQuery: hostEncode });
    check('open succeeds with a tampered record (records are lazy)', t.info().corpusIntegrity === 'per-record-sha256');
    await rejects('hydrating the tampered record fails its per-record digest', () => t.record(17), /record 17 failed integrity verification/);
    check('neighboring records in the same page still hydrate', (await t.record(16)).title === 'rec 16' && (await t.record(18)).title === 'rec 18');
    await rejects('a query that lands on the tampered record fails loudly', () => t.query('rec 17', { k: 1 }), /record 17 failed integrity/);
    await t.close();
    const unverified = await openPancakeFile(memorySource(tampered), { encodeQuery: hostEncode, verifyRecords: false });
    const loose = await unverified.record(17);
    check('verifyRecords:false returns the tampered record and says so in info()',
        loose.text.includes('c`rries') && /unverified by option/.test(unverified.info().corpusIntegrity));
    await unverified.close();

    // Tamper a record digest (not the record): its whole page fails.
    const pages = manifest.corpus.pages;
    const pageTableAt = corpus.offset + 8 + 8 * (COUNT + 1);
    const recordDigestsAt = pageTableAt + 32 * pages;
    const digestTampered = Buffer.from(A.bytes);
    digestTampered[recordDigestsAt + 32 * 100 + 5] ^= 0xff; // record 100's digest, page 1
    const dt = await openPancakeFile(memorySource(digestTampered), { encodeQuery: hostEncode });
    await rejects('a tampered record digest fails its digest page against the page table', () => dt.record(100), /digest page 1 failed hash verification/);
    check('records in other pages are unaffected', (await dt.record(3)).title === 'rec 3');
    await dt.close();

    // Tamper the page table itself: refused at open (it is checked against
    // the identity-bound manifest).
    const pageTampered = Buffer.from(A.bytes);
    pageTampered[pageTableAt + 7] ^= 0xff;
    await rejects('a tampered page table is refused at open', () => openPancakeFile(memorySource(pageTampered), { encodeQuery: hostEncode }), /page table failed hash verification/);

    // Re-point record 17's offsets at record 18's bytes (still monotonic):
    // structure passes, the record digest does not.
    const swapped = Buffer.from(A.bytes);
    swapped.writeBigUInt64LE(BigInt(recStart(18) - corpus.offset), offsetsAt + 8 * 17);
    const sw = await openPancakeFile(memorySource(swapped), { encodeQuery: hostEncode });
    await rejects('an offsets table that maps a record to other bytes fails the per-record digest', () => sw.record(17), /record 17 failed integrity/);
    await sw.close();

    // Non-monotonic offsets: structural rejection before any digest work.
    const nonMono = Buffer.from(A.bytes);
    nonMono.writeBigUInt64LE(BigInt(recStart(30) - corpus.offset), offsetsAt + 8 * 5);
    await rejects('non-monotonic corpus offsets are rejected at open', () => openPancakeFile(memorySource(nonMono), { encodeQuery: hostEncode }), /offsets are inconsistent/);
    const badFirst = Buffer.from(A.bytes);
    badFirst.writeBigUInt64LE(BigInt(recStart(0) - corpus.offset + 1), offsetsAt);
    await rejects('offsets[0] must start exactly where the tables end', () => openPancakeFile(memorySource(badFirst), { encodeQuery: hostEncode }), /offsets are inconsistent/);
    const pastEnd = Buffer.from(A.bytes);
    pastEnd.writeBigUInt64LE(BigInt(corpus.length + 1), offsetsAt + 8 * COUNT);
    await rejects('an offset past the segment is rejected at open', () => openPancakeFile(memorySource(pastEnd), { encodeQuery: hostEncode }), /offsets are inconsistent/);
    const huge = Buffer.from(A.bytes);
    huge.writeBigUInt64LE(BigInt('9007199254740993'), offsetsAt + 8 * COUNT); // 2^53 + 1
    await rejects('an offset beyond the safe-integer range is rejected', () => openPancakeFile(memorySource(huge), { encodeQuery: hostEncode }), /safe integer range/);
}

// Manifest / header / segment-table hostility.
{
    const { manifestBytes, segments } = layoutOf(A.bytes);
    const m = Buffer.from(A.bytes); m[64 + 10] ^= 0x01;
    await rejects('a tampered manifest fails identity verification', () => openPancakeFile(memorySource(m), { encodeQuery: hostEncode }), /identity verification/);
    const id = Buffer.from(A.bytes); id[24 + 3] ^= 0x01;
    await rejects('a tampered identity fails identity verification', () => openPancakeFile(memorySource(id), { encodeQuery: hostEncode }), /identity verification/);
    const ver = Buffer.from(A.bytes); ver.writeUInt32LE(3, 4);
    await rejects('format version 3 is rejected explicitly', () => openPancakeFile(memorySource(ver), { encodeQuery: hostEncode }), /unsupported .pikelet format version 3/);
    const magic = Buffer.from(A.bytes); magic[0] ^= 0x01;
    await rejects('bad magic is rejected', () => openPancakeFile(memorySource(magic), { encodeQuery: hostEncode }), /bad magic/);
    const v1Header = Buffer.from(A.bytes); v1Header.writeUInt32LE(1, 4);
    await rejects('a format-1 header over a v2 manifest is rejected', () => openPancakeFile(memorySource(v1Header), { encodeQuery: hostEncode }), /unsupported profile/);
    const bigManifest = Buffer.from(A.bytes); bigManifest.writeUInt32LE(17 * 1024 * 1024, 8);
    await rejects('an implausible manifest length is rejected before reading', () => openPancakeFile(memorySource(bigManifest), { encodeQuery: hostEncode }), /implausible/);
    const manySegs = Buffer.from(A.bytes); manySegs.writeUInt32LE(65, 12);
    await rejects('an implausible segment count is rejected before reading', () => openPancakeFile(memorySource(manySegs), { encodeQuery: hostEncode }), /implausible/);
    const fileBytesUp = Buffer.from(A.bytes); fileBytesUp.writeBigUInt64LE(BigInt(A.bytes.length + 16), 16);
    await rejects('fileBytes disagreeing with the source size is rejected', () => openPancakeFile(memorySource(fileBytesUp), { encodeQuery: hostEncode }), /truncated or padded/);
    const fileBytesHuge = Buffer.from(A.bytes); fileBytesHuge.writeBigUInt64LE(BigInt('18446744073709551615'), 16);
    await rejects('fileBytes beyond the safe-integer range is rejected', () => openPancakeFile(memorySource(fileBytesHuge, { withSize: false }), { encodeQuery: hostEncode }), /safe integer range/);
    // Segment table: push the corpus segment's length past the file.
    const tableAt = 64 + manifestBytes;
    const segLen = Buffer.from(A.bytes); segLen.writeBigUInt64LE(BigInt(segments.corpus.length + 64), tableAt + 48 * 1 + 16);
    await rejects('a segment-table length disagreeing with the manifest is rejected', () => openPancakeFile(memorySource(segLen), { encodeQuery: hostEncode }), /segment table disagrees/);
    const segOff = Buffer.from(A.bytes); segOff.writeBigUInt64LE(BigInt(segments.corpus.offset + 16), tableAt + 48 * 1 + 8);
    await rejects('a segment-table offset breaking the packed layout is rejected', () => openPancakeFile(memorySource(segOff), { encodeQuery: hostEncode }), /segment table disagrees/);
    // Truncation with an unsized source: the eager reads succeed, the lazy
    // evaluation read comes back short and is refused as such.
    const truncated = A.bytes.subarray(0, A.bytes.length - 10);
    const tr = await openPancakeFile(memorySource(truncated, { withSize: false }), { encodeQuery: hostEncode });
    await rejects('a short read from a truncated source is refused (exact-length reads)', () => tr.evaluation(), /read returned .* of .* bytes/);
    await tr.close();
    await rejects('a truncated sized source is refused at open', () => openPancakeFile(memorySource(truncated), { encodeQuery: hostEncode }), /truncated or padded/);
    // Evaluation tamper.
    const ev = Buffer.from(A.bytes); ev[ev.length - 3] ^= 0xff;
    const evr = await openPancakeFile(memorySource(ev), { encodeQuery: hostEncode });
    await rejects('a tampered evaluation segment fails its digest', () => evr.evaluation(), /evaluation segment failed hash/);
    await evr.close();
    // Query-interp tamper: eager, refused at open.
    const qi = Buffer.from(A.bytes); qi[segments['query-interp'].offset + 20] ^= 0x01;
    await rejects('a tampered query-interp segment is refused at open', () => openPancakeFile(memorySource(qi), { encodeQuery: hostEncode }), /query-interp segment failed hash/);
}

// Read budgets.
{
    await rejects('maxReadBytes below the manifest size refuses the open with the budget named',
        () => openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode, maxReadBytes: 64 }), /exceeds the 64-byte read budget/);
    await rejects('maxReadBytes must be positive', () => openPancakeFile(memorySource(A.bytes), { maxReadBytes: -1 }), /must be a positive number or Infinity/);
    await rejects('maxReadBytes must be a number', () => openPancakeFile(memorySource(A.bytes), { maxReadBytes: '1mb' }), /must be a positive number or Infinity/);
    const inf = await openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode, maxReadBytes: Infinity });
    check('maxReadBytes: Infinity defers to the absolute backstop', (await inf.query('rec 2', { k: 1 })).results[0].id === 2);
    await inf.close();
    const tinyRecords = await openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode, maxRecordBytes: 8 });
    await rejects('maxRecordBytes bounds each hydration read', () => tinyRecords.record(1), /corpus record 1 read of .* exceeds the 8-byte read budget/);
    await tinyRecords.close();
}

// Result semantics and lifecycle (2026-08-21 external review items 4, 5, 7, 8).
{
    // 5. A corpus record that carries its own `id` / `distance` cannot
    // overwrite the search's values.
    const spoofRecords = RECORDS.map((b, i) => (i === 17
        ? Buffer.from(JSON.stringify({ ...JSON.parse(b.toString('utf8')), id: 999, distance: -1 }), 'utf8')
        : b));
    const spoofCorpus = buildCorpusSegment(spoofRecords, { pageRecords: PAGE_RECORDS });
    const spoofPath = path.join(tmp, 'spoof.pikelet');
    assemblePancakeFile({
        profile: PROFILE_V2, corpus: { ...spoofCorpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
        encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
    }, [
        { kind: 'index', bytes: SKETCH },
        { kind: 'corpus', bytes: spoofCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(kind2Declaration()), 'utf8'), CALIBRATION) },
        { kind: 'evaluation', bytes: EVALUATION },
    ], spoofPath);
    const spoof = await openPancakeFile(spoofPath, { encodeQuery: hostEncode });
    const hit = (await spoof.query('rec 17', { k: 1 })).results[0];
    check('a record\'s own id/distance fields do not overwrite the search id/distance', hit.id === 17 && hit.distance >= 0 && hit.title === 'rec 17', JSON.stringify(hit).slice(0, 120));
    // 8. k: absent -> default; supplied must be a positive integer.
    check('k omitted -> default 5', (await spoof.query('rec 17')).results.length === 5);
    for (const bad of [0, -1, 2.5, '3', NaN]) {
        await rejects(`k=${String(bad)} is rejected rather than silently defaulting`, () => spoof.query('rec 17', { k: bad }), /k must be a positive integer/);
    }
    check('k above the corpus size is capped', (await spoof.query('rec 17', { k: 10000 })).results.length <= COUNT);
    // 7. close() is idempotent (file-path open owns the fd) and the reader
    // refuses use after close.
    await spoof.close();
    let secondClose = 'ok';
    try { await spoof.close(); } catch (err) { secondClose = err.code || err.message; }
    check('close() twice on a file-path reader does not throw (no EBADF)', secondClose === 'ok', secondClose);
    await rejects('query() after close is refused', () => spoof.query('rec 1'), /reader is closed/);
    await rejects('record() after close is refused', () => spoof.record(1), /reader is closed/);
    // A digest-page read that fails for a transport reason must not stay
    // cached as a rejection: the next hydration retries and succeeds.
    {
        let failNext = true;
        const flaky = {
            size: A.bytes.length, preferredParallelism: Infinity, preferredGapBytes: 0,
            async read(offset, length) {
                // The digest table sits between the offsets table and the
                // records; fail the first read that lands there.
                const { segments } = layoutOf(A.bytes);
                const corpus = segments.corpus;
                const pageTableAt = corpus.offset + 8 + 8 * (COUNT + 1);
                const recordsAt = pageTableAt + 32 * Math.ceil(COUNT / PAGE_RECORDS) + 32 * COUNT;
                if (failNext && offset >= pageTableAt + 32 * Math.ceil(COUNT / PAGE_RECORDS) && offset < recordsAt) {
                    failNext = false;
                    throw new Error('simulated transport failure');
                }
                return A.bytes.subarray(offset, offset + length);
            },
            async close() {},
        };
        const fl = await openPancakeFile(flaky, { encodeQuery: hostEncode });
        await rejects('first hydration fails with the transport error', () => fl.record(100), /simulated transport failure/);
        check('the failed digest page is not cached: the retry succeeds', (await fl.record(100)).title === 'rec 100');
        await fl.close();
    }
    // httpRangeSource: a 206 whose Content-Range is not the requested slice
    // is refused instead of being handed to the reader.
    {
        const realFetch2 = globalThis.fetch;
        globalThis.fetch = async (url, init = {}) => {
            if (init.method === 'HEAD') return new Response(null, { status: 200, headers: { 'content-length': String(A.bytes.length), 'accept-ranges': 'bytes' } });
            const m = /bytes=(\d+)-(\d+)/.exec(init.headers.Range);
            const start = Number(m[1]) + 16; // off by 16: a misbehaving cache
            const body = A.bytes.subarray(start, start + (Number(m[2]) - Number(m[1]) + 1));
            return new Response(body, { status: 206, headers: { 'content-range': `bytes ${start}-${start + body.length - 1}/${A.bytes.length}` } });
        };
        try {
            const { httpRangeSource: rangeSource } = await import('../complete/sources.mjs');
            const src = rangeSource('http://host.invalid/b.pikelet');
            await src.init();
            await rejects('a 206 answering a different range than requested is refused', () => src.read(64, 64), /returned Content-Range bytes 80-143.*requested bytes 64-127/);
        } finally {
            globalThis.fetch = realFetch2;
        }
    }
    // 4. httpRangeSource: after the one-time full-download fallback, reads
    // are served from memory and issue no further requests.
    const { httpRangeSource } = await import('../complete/sources.mjs');
    const realFetch = globalThis.fetch;
    let requests = 0;
    globalThis.fetch = async (url, init = {}) => {
        requests++;
        if (init.method === 'HEAD') return new Response(null, { status: 200, headers: { 'content-length': String(A.bytes.length) } });
        // A host that ignores Range: always the full body with 200.
        return new Response(A.bytes, { status: 200, headers: { 'content-length': String(A.bytes.length) } });
    };
    try {
        const warn = console.warn; console.warn = () => {};
        const src = httpRangeSource('http://host.invalid/a.pikelet');
        await src.init();
        const reqAfterInit = requests;
        const first = await src.read(0, 64);
        const reqAfterFirst = requests;
        await src.read(64, 64);
        await src.read(128, 64);
        console.warn = warn;
        check('range-ignoring host: first read falls back to one full download', reqAfterFirst - reqAfterInit === 1 && src.stats.fullFallback && first.length === 64);
        check('subsequent reads are served from memory with no further requests', requests === reqAfterFirst, `requests after 3 reads: ${requests - reqAfterInit}`);
    } finally {
        globalThis.fetch = realFetch;
    }
}

// Index authentication: the identity must bind the embedded sketch's
// semantics, not just its bytes-as-a-lazy-segment.
{
    const { manifest, segments } = layoutOf(A.bytes);
    check('format-2 manifest commits to the index header', /^[0-9a-f]{64}$/.test(manifest.index?.headerSha256));
    // The sketch header's metric word decides search semantics. Flipping it
    // (cosine -> l2) leaves the complete identity and the sketch's own
    // resident hash untouched — the manifest's header commitment must catch it.
    const metricFlip = Buffer.from(A.bytes);
    metricFlip.writeUInt32LE(0, segments.index.offset + 12); // metric: cosine -> l2
    await rejects('format 2: a flipped index metric fails the manifest header commitment',
        () => openPancakeFile(memorySource(metricFlip), { encodeQuery: hostEncode }), /index header failed hash verification/);
    // A header field the sketch never self-verifies (recommendedRerank at
    // offset 120) is caught ONLY by the manifest commitment — this is the
    // deterministic proof of the anchor.
    const rerankFlip = Buffer.from(A.bytes);
    rerankFlip.writeUInt32LE(7, segments.index.offset + 120);
    await rejects('format 2: a tampered sketch header field the sketch never self-checks fails the manifest commitment',
        () => openPancakeFile(memorySource(rerankFlip), { encodeQuery: hostEncode }), /index header failed hash verification/);
    // Rewriting the sketch's own residentSha256 fails both the manifest
    // commitment and the sketch's resident self-check; they run in the same
    // open wave, so whichever rejects first surfaces — either way the open
    // fails closed.
    const selfHash = Buffer.from(A.bytes);
    selfHash[segments.index.offset + 60] ^= 0xff;
    await rejects('format 2: a rewritten sketch self-hash fails the open (manifest commitment or resident self-check)',
        () => openPancakeFile(memorySource(selfHash), { encodeQuery: hostEncode }),
        /index header failed hash verification|resident prefix failed hash verification/);
    // Format 1 has no header commitment; the metric cross-check against the
    // identity-verified manifest is the binding there.
    const v1 = buildSynthetic({ profile: PROFILE_V1, name: 'v1metric' });
    const v1layout = layoutOf(v1.bytes);
    const v1flip = Buffer.from(v1.bytes);
    v1flip.writeUInt32LE(0, v1layout.segments.index.offset + 12);
    await rejects('format 1: a flipped index metric fails the manifest cross-check',
        () => openPancakeFile(memorySource(v1flip), { encodeQuery: hostEncode }), /index metric l2 != manifest metric cosine/);
}

// Lazy vector rows: with a format-2 embedded sketch (the default build),
// every rerank row is verified on the read that fetches it, per-read.
{
    const { segments } = layoutOf(A.bytes);
    // Tamper a byte near the end of the index segment — inside the last
    // block's rows, far past the resident prefix.
    const rowTamper = Buffer.from(A.bytes);
    rowTamper[segments.index.offset + segments.index.length - 8] ^= 0xff;
    const lazy = await openPancakeFile(memorySource(rowTamper), { encodeQuery: hostEncode });
    check('the reader reports per-row index integrity for the format-2 sketch',
        lazy.info().indexRowIntegrity === 'per-row-sha256' && lazy.info().vectorsVerified === false);
    await rejects('a query whose rerank touches the tampered row fails its per-row digest',
        () => lazy.query('rec 299', { k: 3 }), /row failed digest verification|digest page failed/);
    await rejects('...and verifyVectors() detects it too', () => lazy.verifyVectors(), /vector|hash/i);
    await lazy.close();
    await rejects('verifyIndexVectors: true refuses the tamper at open',
        () => openPancakeFile(memorySource(rowTamper), { encodeQuery: hostEncode, verifyIndexVectors: true }), /vector|hash/i);
    const clean = await openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode, verifyIndexVectors: true });
    check('verifyIndexVectors: true on a clean artifact opens with vectorsVerified true',
        clean.info().vectorsVerified === true && (await clean.query('rec 17', { k: 1 })).results[0].id === 17);
    await clean.close();
}

// Malformed verification-vector expectations must fail verification, not
// slip through as NaN comparisons.
{
    const wrongEncoder = () => new Float32Array(DIM).fill(0.25);
    const cases = [
        ['string components', TEST_VECTOR_TEXTS.map((text) => ({ text, embedding: Array(DIM).fill('x'), tolerance: 1e-6 }))],
        ['null components', TEST_VECTOR_TEXTS.map((text) => ({ text, embedding: Array(DIM).fill(null), tolerance: 1e-6 }))],
        ['too-short embedding', TEST_VECTOR_TEXTS.map((text) => ({ text, embedding: [0.1, 0.2], tolerance: 1e-6 }))],
        ['too-long embedding', TEST_VECTOR_TEXTS.map((text) => ({ text, embedding: Array(DIM + 1).fill(0.1), tolerance: 1e-6 }))],
        ['missing embedding', TEST_VECTOR_TEXTS.map((text) => ({ text, tolerance: 1e-6 }))],
    ];
    for (const [label, testVectors] of cases) {
        const bad = buildSynthetic({ declaration: { ...kind2Declaration({ withVectors: false }), testVectors }, name: `badvec-${label.replace(/[^a-z]/g, '')}` });
        await rejects(`a declaration with ${label} fails encoder verification instead of passing vacuously`,
            () => openPancakeFile(memorySource(bad.bytes), { encodeQuery: wrongEncoder }), /expected embedding/);
    }
    // NaN / Infinity cannot ride in JSON, but the exported API can receive
    // them programmatically.
    for (const poison of [NaN, Infinity]) {
        await rejects(`verifyHostEncoder rejects a ${poison} expectation`,
            () => verifyHostEncoder({ model: 'x', testVectors: [{ text: 't', embedding: Array(DIM).fill(poison) }] }, wrongEncoder, DIM), /finite number/);
    }
    // And the validated path still catches a genuinely wrong encoder.
    const good = buildSynthetic({ name: 'goodvec' });
    await rejects('a wrong encoder still fails against well-formed vectors',
        () => openPancakeFile(memorySource(good.bytes), { encodeQuery: wrongEncoder }), /failed the artifact's verification vector/);
}

// Builders accept plain Uint8Arrays wherever the typings say Uint8Array.
{
    const u8records = RECORDS.map((b) => new Uint8Array(b)); // copies, not Buffers
    const u8corpus = buildCorpusSegment(u8records, { pageRecords: PAGE_RECORDS });
    const u8qi = buildQueryInterpSegment(2,
        new Uint8Array(Buffer.from(JSON.stringify(kind2Declaration()), 'utf8')),
        new Uint8Array(CALIBRATION));
    const u8path = path.join(tmp, 'u8.pikelet');
    assemblePancakeFile({
        profile: PROFILE_V2, corpus: { ...u8corpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
        encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
    }, [
        { kind: 'index', bytes: new Uint8Array(SKETCH) },
        { kind: 'corpus', bytes: u8corpus.bytes },
        { kind: 'query-interp', bytes: u8qi },
        { kind: 'evaluation', bytes: new Uint8Array(EVALUATION) },
    ], u8path);
    const u8 = await openPancakeFile(u8path, { encodeQuery: hostEncode });
    check('an artifact built entirely from plain Uint8Arrays opens and queries',
        (await u8.query('rec 17', { k: 1 })).results[0].id === 17);
    // set() must write the same bytes copy() did: the segments (index,
    // corpus, query-interp, evaluation) digest identically to the
    // Buffer-built artifact's. (The manifests differ in unrelated fields,
    // so whole-artifact identities are not compared.)
    const u8segments = layoutOf(fs.readFileSync(u8path)).manifest.segments;
    const bufSegments = layoutOf(A.bytes).manifest.segments;
    check('...with segments byte-identical to the Buffer-built artifact',
        u8segments.length === bufSegments.length && u8segments.every((s, i) => s.sha256 === bufSegments[i].sha256));
    await u8.close();
}

// canonicalJson follows JSON.stringify semantics for unsupported values.
{
    check('canonicalJson omits undefined object properties', canonicalJson({ a: 1, b: undefined, c: () => {} }) === '{"a":1}');
    check('canonicalJson turns unsupported array entries into null', canonicalJson([1, undefined, () => {}, 2]) === '[1,null,null,2]');
    check('canonicalJson output parses', JSON.parse(canonicalJson({ b: [undefined], a: 'x' })).b[0] === null);
}

// evaluation(): absent segment -> null; a non-object segment is refused.
{
    const noEval = (() => {
        const corpus = buildCorpusSegment(RECORDS, { pageRecords: PAGE_RECORDS });
        const outPath = path.join(tmp, 'noeval.pikelet');
        assemblePancakeFile({
            profile: PROFILE_V2, corpus: { ...corpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
            encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
        }, [
            { kind: 'index', bytes: SKETCH },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(kind2Declaration()), 'utf8'), CALIBRATION) },
        ], outPath);
        return fs.readFileSync(outPath);
    })();
    const ne = await openPancakeFile(memorySource(noEval), { encodeQuery: hostEncode });
    check('evaluation() is null when the artifact carries no evaluation segment', (await ne.evaluation()) === null);
    await ne.close();
    const arrayEval = buildSynthetic({ name: 'arrayeval' });
    const { segments: ae } = layoutOf(arrayEval.bytes);
    // Rebuild with an array evaluation segment.
    const corpus = buildCorpusSegment(RECORDS, { pageRecords: PAGE_RECORDS });
    const outPath = path.join(tmp, 'arrayeval2.pikelet');
    assemblePancakeFile({
        profile: PROFILE_V2, corpus: { ...corpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
        encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
    }, [
        { kind: 'index', bytes: SKETCH },
        { kind: 'corpus', bytes: corpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(kind2Declaration()), 'utf8'), CALIBRATION) },
        { kind: 'evaluation', bytes: Buffer.from('[1,2]', 'utf8') },
    ], outPath);
    const av = await openPancakeFile(outPath, { encodeQuery: hostEncode });
    await rejects('a non-object evaluation segment is refused', () => av.evaluation(), /must be a JSON object/);
    await av.close();
    void ae;
}

// httpRangeSource fails closed on malformed 206s and streaming overruns.
{
    const realFetch3 = globalThis.fetch;
    try {
        const { httpRangeSource: rangeSource } = await import('../complete/sources.mjs');
        // 206 without Content-Range (RFC requires it): refused.
        globalThis.fetch = async (url, init = {}) => {
            if (init.method === 'HEAD') return new Response(null, { status: 200, headers: { 'content-length': String(A.bytes.length) } });
            const m = /bytes=(\d+)-(\d+)/.exec(init.headers.Range);
            return new Response(A.bytes.subarray(Number(m[1]), Number(m[2]) + 1), { status: 206 });
        };
        const noHeader = rangeSource('http://host.invalid/c.pikelet');
        await noHeader.init();
        await rejects('a 206 without Content-Range is refused', () => noHeader.read(0, 64), /Content-Range \(missing\)/);
        // A 200 that streams more than the fallback cap is aborted on
        // received bytes, whatever the headers claimed.
        globalThis.fetch = async (url, init = {}) => {
            if (init.method === 'HEAD') return new Response(null, { status: 200, headers: { 'content-length': '1024' } });
            const chunk = new Uint8Array(1024).fill(7);
            let sent = 0;
            return new Response(new ReadableStream({
                pull(controller) {
                    if (sent >= 16) { controller.close(); return; }
                    sent++; controller.enqueue(chunk);
                },
            }), { status: 200 });
        };
        const streamy = rangeSource('http://host.invalid/d.pikelet', { maxFullFallbackBytes: 4096 });
        await streamy.init();
        await rejects('a chunked 200 streaming past the cap is aborted on received bytes',
            () => streamy.read(0, 64), /streamed over 4096 bytes/);
        // cacheKeyParam: null keeps the URL untouched (signed-URL hosts).
        const seen = [];
        globalThis.fetch = async (url, init = {}) => {
            if (init.method === 'HEAD') return new Response(null, { status: 200, headers: { 'content-length': String(A.bytes.length) } });
            seen.push(String(url));
            const m = /bytes=(\d+)-(\d+)/.exec(init.headers.Range);
            const body = A.bytes.subarray(Number(m[1]), Number(m[2]) + 1);
            return new Response(body, { status: 206, headers: { 'content-range': `bytes ${m[1]}-${m[2]}/${A.bytes.length}` } });
        };
        const unsigned = rangeSource('http://host.invalid/e.pikelet?sig=abc', { cacheKeyParam: null });
        await unsigned.init();
        await unsigned.read(0, 64);
        check('cacheKeyParam: null leaves the URL untouched for signed-URL hosts', seen[0] === 'http://host.invalid/e.pikelet?sig=abc');
    } finally {
        globalThis.fetch = realFetch3;
    }
}

// Unknown and duplicate segments; manifest integrity-block consistency.
{
    const extra = buildSynthetic({ extraSegments: [{ kind: 'vendor-extra', kindNumber: 9, bytes: Buffer.from('opaque future segment') }], name: 'extra' });
    const ex = await openPancakeFile(memorySource(extra.bytes), { encodeQuery: hostEncode });
    check('an unknown segment kind is skipped and the file still serves', (await ex.query('rec 7', { k: 1 })).results[0].id === 7);
    await ex.close();
    const dup = buildSynthetic({ extraSegments: [{ kind: 'corpus', bytes: buildCorpusSegment(RECORDS.slice(0, 2)).bytes }], name: 'dup' });
    await rejects('a duplicate corpus segment is rejected', () => openPancakeFile(memorySource(dup.bytes), { encodeQuery: hostEncode }), /more than one corpus segment/);
    const inflated = buildSynthetic({
        name: 'inflated',
        manifestPatch: (m) => {
            // A manifest claiming a billion records over this segment: the
            // integrity block is kept self-consistent so the size check is
            // what refuses it, before any allocation.
            m.corpus.records = 1e9;
            m.corpus.pages = Math.ceil(1e9 / PAGE_RECORDS);
        },
    });
    await rejects('a manifest record count the segment cannot hold is rejected before allocation',
        () => openPancakeFile(memorySource(inflated.bytes), { encodeQuery: hostEncode }), /implausible for the corpus segment/);
    const badPages = buildSynthetic({ name: 'badpages', manifestPatch: (m) => { m.corpus.pages += 1; } });
    await rejects('an inconsistent integrity block is rejected', () => openPancakeFile(memorySource(badPages.bytes), { encodeQuery: hostEncode }), /integrity block is inconsistent/);
    const badDim = buildSynthetic({ name: 'baddim', manifestPatch: (m) => { m.dim = DIM + 1; } });
    await rejects('a kind-2 declaration dim disagreeing with the manifest is rejected',
        () => openPancakeFile(memorySource(badDim.bytes), { encodeQuery: () => new Float32Array(DIM + 1) }), /disagrees with manifest dim/);
}

// ---------------------------------------------------------------------------
// B. format 1 compatibility (whole-segment integrity only)
// ---------------------------------------------------------------------------
console.log('\nB. format-1 file (pancake-complete-v1) compatibility');
{
    const B = buildSynthetic({ profile: PROFILE_V1, name: 'v1' });
    const view = new DataView(B.bytes.buffer, B.bytes.byteOffset, B.bytes.byteLength);
    check('builder writes header formatVersion 1 for the v1 profile', view.getUint32(4, true) === 1);
    const search = await openPancakeFile(memorySource(B.bytes), { encodeQuery: hostEncode });
    const info = search.info();
    check('opens: format 1 / pancake-complete-v1', info.formatVersion === 1 && info.profile === PROFILE_V1);
    check('reports the transitional whole-segment integrity', info.corpusIntegrity === 'segment-sha256');
    check('queries and hydrates', (await search.query('rec 17', { k: 1 })).results[0].title === 'rec 17');
    await search.close();
    // The documented v1 limitation: a record tamper is not detectable on
    // the lazy read. Stated, not hidden.
    const { segments } = layoutOf(B.bytes);
    const offsetsAt = segments.corpus.offset + 4;
    const r17 = segments.corpus.offset + Number(view.getBigUint64(offsetsAt + 8 * 17, true));
    const tampered = Buffer.from(B.bytes);
    const textAt = tampered.indexOf('carries', r17) + 1;
    tampered[textAt] ^= 0x01;
    const t = await openPancakeFile(memorySource(tampered), { encodeQuery: hostEncode });
    const rec = await t.record(17);
    check('format 1: a record tamper is NOT detected on hydration (transitional stance, spec 6)', rec.text.includes('c`rries'));
    await t.close();
    await rejects('a v2 manifest under a format-1 header is rejected', () => {
        const bad = buildSynthetic({ profile: PROFILE_V1, name: 'v1bad', manifestPatch: (m) => { m.corpus.layout = 'records-v2'; } });
        return openPancakeFile(memorySource(bad.bytes));
    }, /carries corpus layout v1|records-v2/);
}

// ---------------------------------------------------------------------------
// C. kind-1 artifact from the committed examples/03 assets (format 2)
// ---------------------------------------------------------------------------
console.log('\nC. kind-1 student-inline artifact compiled from examples/03 assets');
{
    const assets = path.join(ROOT, 'examples', 'legacy', '03-edge-docs-search', 'assets');
    const corpusRaw = JSON.parse(fs.readFileSync(path.join(assets, 'docs-corpus.json'), 'utf8'));
    const sourceManifest = JSON.parse(fs.readFileSync(path.join(assets, 'docs-manifest.json'), 'utf8'));
    const goldens = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples', 'legacy', '03-edge-docs-search', 'fixtures', 'abstention-golden.json'), 'utf8'));
    const count = Object.keys(corpusRaw).length;
    const records = [];
    for (let id = 0; id < count; id++) {
        const r = corpusRaw[String(id)];
        records.push(Buffer.from(JSON.stringify({ title: r.title, text: r.text, preview: r.preview, sourcePath: r.sourcePath, anchor: r.anchor }), 'utf8'));
    }
    const snapshot = fs.readFileSync(path.join(assets, 'docs-index.bin'));
    const compileDocs = (name) => {
        const { bytes: sketchBytes } = Pikelet.buildSketchArtifactBytes(snapshot, { recommendedRerank: 120 });
        const corpus = buildCorpusSegment(records);
        return assemblePancakeFile({
            profile: PROFILE_V2,
            corpus: { ...corpus.corpus, provenance: null },
            dim: sourceManifest.dim,
            metric: sourceManifest.metric,
            encoder: { kind: 'student-inline-v1', ...sourceManifest.encoder },
            recommendedRerank: 120,
            sampleQueries: sourceManifest.sampleQueries || [],
        }, [
            { kind: 'index', bytes: Buffer.from(sketchBytes) },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(1, fs.readFileSync(path.join(assets, 'docs-student.bin')), fs.readFileSync(path.join(assets, 'docs-abstention.json'))) },
            { kind: 'evaluation', bytes: Buffer.from(JSON.stringify({ goldenQueries: goldens }), 'utf8') },
        ], path.join(tmp, `${name}.pikelet`));
    };
    const first = compileDocs('docs-a');
    const second = compileDocs('docs-b');
    check('compile is byte-deterministic (same identity twice)', first.identity === second.identity
        && sha256(fs.readFileSync(first.outPath)).equals(sha256(fs.readFileSync(second.outPath))));
    const search = await openPancakeFile(first.outPath);
    const info = search.info();
    check('kind-1 opens as format 2 with per-record integrity', info.formatVersion === 2 && info.corpusIntegrity === 'per-record-sha256' && info.records === count);
    check('inline student encoder needs no host verification (encoderVerified null)', info.encoderVerified === null);
    const evaluation = await search.evaluation();
    check('evaluation carries the 10 abstention goldens', evaluation.goldenQueries.length === 10);
    let reproduced = 0;
    for (const fixture of evaluation.goldenQueries) {
        const out = await search.query(fixture.text, { k: 5 });
        if (out.matchQuality === fixture.expected && (fixture.expected !== 'none' || out.results.length === 0)) reproduced++;
        else console.log(`    golden miss [${fixture.family}] "${fixture.text}": expected ${fixture.expected}, got ${out.matchQuality} (${out.results.length} results)`);
    }
    check('all 10 abstention goldens reproduce their labels', reproduced === 10, `${reproduced}/10`);
    // Abstention is fit at a fixed top-10 window (pikelet/src/calibrate.mjs
    // K = min(10, candidates)), independent of the k a caller later passes
    // to query(). The verdict for a given retrieval must not depend on k —
    // a caller asking for k=1 should get the same matchQuality as k=20,
    // since the underlying retrieval signals (d0, margin, mean10,
    // coverage1) are windowed the same way regardless.
    let kInvariant = 0;
    for (const fixture of evaluation.goldenQueries) {
        const verdicts = new Set();
        for (const kk of [1, 2, 3, 5, 10, 20]) {
            const out = await search.query(fixture.text, { k: kk });
            verdicts.add(out.matchQuality);
        }
        if (verdicts.size === 1) kInvariant++;
        else console.log(`    k-variance [${fixture.family}] "${fixture.text}": ${[...verdicts].join(', ')}`);
    }
    check('abstention verdict is invariant across k=1..20 for all 10 goldens', kInvariant === 10, `${kInvariant}/10`);
    const abstainedFixture = evaluation.goldenQueries.find((f) => f.expected === 'none');
    const hidden = await search.query(abstainedFixture.text, { k: 3 });
    check('a none verdict withholds results by default', hidden.matchQuality === 'none' && hidden.results.length === 0);
    const shown = await search.query(abstainedFixture.text, { k: 3, showAbstained: true });
    check('showAbstained: true surfaces results under the same none verdict',
        shown.matchQuality === 'none' && shown.confidence === hidden.confidence && shown.results.length === 3);
    const probe = await search.query('how does compaction work', { k: 3 });
    check('probe query hydrates records matching the source corpus', probe.results.length === 3 && probe.results.every((r) => {
        const src = corpusRaw[String(r.id)];
        return src && src.title === r.title && src.text === r.text && src.sourcePath === r.sourcePath;
    }));
    await search.close();
}

// ---------------------------------------------------------------------------
// D. lexical segment (kind 5) and hybrid retrieval
// ---------------------------------------------------------------------------
console.log('\nD. lexical segment and hybrid retrieval');
{
    const lexical = buildLexicalSegment(RECORDS.map((b) => JSON.parse(b.toString('utf8')).text));
    const lx = openLexicalIndex(lexical.bytes);
    check('lexical index counts match the corpus', lx.docCount === COUNT && lx.termCount > 0 && lexical.meta.docCount === COUNT);
    const direct = lx.search('record number 17', 5);
    check('BM25 ranks the record holding the rare token first', direct[0]?.id === 17, JSON.stringify(direct.slice(0, 3)));
    check('BM25 finds nothing for unindexed tokens', lx.search('zzqqxxvv', 5).length === 0);

    const H = buildSynthetic({
        name: 'hybrid',
        extraSegments: [{ kind: 'lexical', bytes: lexical.bytes }],
        manifestPatch: (m) => { m.lexical = lexical.meta; },
    });
    const search = await openPancakeFile(memorySource(H.bytes), { encodeQuery: hostEncode });
    check('info reports the lexical segment', search.info().lexical?.terms === lexical.meta.terms
        && search.info().lexical?.docCount === COUNT);
    // hostEncode('record number 17') is deterministic noise unrelated to
    // hostEncode('rec 17'), so pure vector search cannot find record 17.
    // The lexical candidates bring it into the exact rerank and RRF
    // surfaces it — the mechanism that rescues known-item lookups.
    const hybrid = await search.query('record number 17', { k: 3 });
    check('hybrid query surfaces the lexical match vector search cannot find',
        hybrid.results.some((r) => r.id === 17), JSON.stringify(hybrid.results.map((r) => r.id)));
    check('hybrid results carry finite vector distances',
        hybrid.results.length === 3 && hybrid.results.every((r) => Number.isFinite(r.distance)));
    await search.close();

    const vectorOnly = await openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode });
    const control = await vectorOnly.query('record number 17', { k: 3 });
    check('vector-only artifact misses the same known-item lookup (control)',
        !control.results.some((r) => r.id === 17), JSON.stringify(control.results.map((r) => r.id)));
    check('vector-only artifact reports no lexical index', vectorOnly.info().lexical === null);
    await vectorOnly.close();

    const tampered = Buffer.from(H.bytes);
    const seg = layoutOf(tampered).segments.lexical;
    tampered[seg.offset + Math.floor(seg.length / 2)] ^= 0xff;
    await rejects('tampered lexical segment fails hash verification at open',
        () => openPancakeFile(memorySource(tampered), { encodeQuery: hostEncode }),
        /lexical segment failed hash/);

    // The lazy opener (wiki-scale path: interpolation search over remote
    // ranges) must score identically to the eager reader on the same
    // segment bytes.
    const { openLexicalIndexLazy } = await import('../complete/lexical.mjs');
    const lazyIdx = await openLexicalIndexLazy(
        async (off, len) => lexical.bytes.subarray(off, off + len), lexical.bytes.length);
    check('lazy opener reports the same counts and marks itself lazy',
        lazyIdx.docCount === COUNT && lazyIdx.termCount === lx.termCount && lazyIdx.lazy === true);
    for (const q of ['record number 17', 'carries some text', 'zzqqxxvv', 'record 250 tampering']) {
        const eager = lx.search(q, 8);
        const lazyHits = await lazyIdx.search(q, 8);
        check(`lazy search matches eager for ${JSON.stringify(q)}`,
            JSON.stringify(lazyHits) === JSON.stringify(eager),
            `${JSON.stringify(lazyHits.slice(0, 3))} vs ${JSON.stringify(eager.slice(0, 3))}`);
    }
}

{
    // --- expectedIdentity: refused one header read in ---
    // Wrong knowledge object: don't touch the knowledge object. A pinned
    // open that finds a different identity must stop after the 64-byte
    // header, before the manifest or any segment is fetched.
    let reads = 0;
    const counting = {
        size: A.bytes.length,
        preferredParallelism: Infinity,
        preferredGapBytes: 2048,
        async read(offset, length) { reads += 1; return A.bytes.subarray(offset, offset + length); },
        async close() {},
    };
    await rejects('a mismatched identity pin refuses the open',
        () => openPancakeFile(counting, { expectedIdentity: 'f'.repeat(64) }), /identity mismatch/);
    check('the mismatch is decided on the header read alone', reads === 1, `${reads} reads`);
    await rejects('a malformed identity pin is rejected as such',
        () => openPancakeFile(memorySource(A.bytes), { expectedIdentity: 'not-a-hash' }), /sha256 hex/);
    const unpinned = await openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode });
    const trueIdentity = unpinned.info().identity;
    await unpinned.close();
    const pinned = await openPancakeFile(memorySource(A.bytes), {
        encodeQuery: hostEncode,
        expectedIdentity: trueIdentity.toUpperCase(), // case-insensitive pin
    });
    check('a correct identity pin opens and serves', pinned.info().identity === trueIdentity);
    await pinned.close();
}

{
    // --- httpRangeSource redirect memoization ---
    // A redirecting host (GitHub release assets: every hit to the front
    // door 302s to a signed URL) must be charged once, not per range read;
    // the pinned signed target must never get the cache-key param
    // (signed query strings); and an expired pin (403) must re-resolve.
    const http = await import('node:http');
    const { httpRangeSource } = await import('../complete/sources.mjs');
    const payload = Buffer.from(A.bytes);
    let frontDoorHits = 0;
    let signedHits = 0;
    let paramOnSigned = 0;
    let token = 'sig-1';
    const host = http.createServer((req, res) => {
        const u = new URL(req.url, 'http://x');
        if (u.pathname === '/front/pack.pikelet') {
            frontDoorHits += 1;
            res.writeHead(302, { location: `/signed/pack.pikelet?token=${token}` });
            res.end();
            return;
        }
        if (u.pathname !== '/signed/pack.pikelet') { res.writeHead(404); res.end(); return; }
        if (u.searchParams.get('token') !== token) { res.writeHead(403); res.end(); return; }
        signedHits += 1;
        if (u.searchParams.has('r')) paramOnSigned += 1;
        const range = req.headers.range && /^bytes=(\d+)-(\d+)$/.exec(req.headers.range);
        if (!range) {
            res.writeHead(200, { 'accept-ranges': 'bytes', 'content-length': payload.length });
            res.end(req.method === 'HEAD' ? undefined : payload);
            return;
        }
        const from = Number(range[1]);
        const to = Math.min(Number(range[2]), payload.length - 1);
        res.writeHead(206, { 'content-range': `bytes ${from}-${to}/${payload.length}`, 'content-length': to - from + 1 });
        res.end(payload.subarray(from, to + 1));
    });
    await new Promise((resolve) => host.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${host.address().port}`;
    const src = httpRangeSource(`${base}/front/pack.pikelet`);
    await src.init();
    check('redirect resolution finds the size through the pinned target', src.size === payload.length,
        `size ${src.size}`);
    for (let i = 0; i < 6; i++) {
        const got = await src.read(16 * i, 16);
        check(`redirected read ${i} returns the right bytes`,
            Buffer.compare(Buffer.from(got), payload.subarray(16 * i, 16 * i + 16)) === 0);
    }
    check('the front door is charged once, not per read', frontDoorHits === 1, `hits ${frontDoorHits}`);
    check('range reads go straight to the pinned target', signedHits >= 6, `hits ${signedHits}`);
    check('the pinned signed target never gets the cache-key param', paramOnSigned === 0, `${paramOnSigned}`);
    token = 'sig-2'; // expire the pin: old token now 403s
    const afterExpiry = await src.read(0, 16);
    check('an expired pin re-resolves through the front door and the read succeeds',
        Buffer.compare(Buffer.from(afterExpiry), payload.subarray(0, 16)) === 0 && frontDoorHits === 2,
        `frontDoorHits ${frontDoorHits}`);
    check('stats count the resolution hops', src.stats.redirects >= 2, `redirects ${src.stats.redirects}`);
    host.close();
}

{
    // --- WASM sketch scanner (options.sketchScanner) ---
    // The engine's SIMD scan kernel must select the same candidates the JS
    // scan does — the exact rerank then scores both sets identically, so
    // any divergence shows up as differing [id, distance] pairs.
    const js = await openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode, sketchScanner: false });
    check('sketchScanner:false reports residentScan js', js.info().residentScan === 'js');
    const wasm = await openPancakeFile(memorySource(A.bytes), {
        encodeQuery: hostEncode,
        sketchScanner: (sk) => Pikelet.createSketchScanner(sk, { maxRerank: 64 }),
    });
    // Staging is background by design; poll until the reader reports it.
    for (let i = 0; i < 400 && wasm.info().residentScan !== 'engine'; i++) {
        await new Promise((resolve) => setTimeout(resolve, 25));
    }
    check('injected scanner factory reaches residentScan engine', wasm.info().residentScan === 'engine');
    for (const q of ['rec 17', 'rec 250', 'which treaty ended the first world war']) {
        const a = await js.query(q, { k: 5 });
        const b = await wasm.query(q, { k: 5 });
        check(`scanner parity for ${JSON.stringify(q)}`,
            JSON.stringify(a.results.map((r) => [r.id, r.distance]))
                === JSON.stringify(b.results.map((r) => [r.id, r.distance])),
            `js ${JSON.stringify(a.results.map((r) => r.id))} vs engine ${JSON.stringify(b.results.map((r) => r.id))}`);
    }
    // A rerank beyond the scanner's creation-time buffers (maxRerank 64)
    // must fall back to the JS scan for that query — same results as the
    // scanner-less reader, never a silently truncated candidate pool.
    const wideJs = await js.query('rec 17', { k: 5, rerank: 100 });
    const wideWasm = await wasm.query('rec 17', { k: 5, rerank: 100 });
    check('rerank beyond scanner maxRerank falls back to the JS scan',
        JSON.stringify(wideJs.results.map((r) => [r.id, r.distance]))
            === JSON.stringify(wideWasm.results.map((r) => [r.id, r.distance])));
    await wasm.close();
    await js.close();

    // A factory that fails must leave the reader serving the JS scan, not
    // poison queries.
    const broken = await openPancakeFile(memorySource(A.bytes), {
        encodeQuery: hostEncode,
        sketchScanner: () => { throw new Error('no engine here'); },
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    const survived = await broken.query('rec 17', { k: 3 });
    check('failed scanner factory leaves the JS scan serving queries',
        broken.info().residentScan === 'js' && survived.results.some((r) => r.id === 17));
    await broken.close();

    await rejects('a non-scanner sketchScanner option is rejected',
        () => openPancakeFile(memorySource(A.bytes), { encodeQuery: hostEncode, sketchScanner: 42 }),
        /sketchScanner must be false/);
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`\nComplete-profile reader conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
