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
// examples/one-file-search/test-inline.mjs, not here.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import {
    buildCorpusSegment, buildCorpusSegmentFromBuffers, buildQueryInterpSegment,
    assemblePikeletFile, buildLexicalSegment, PROFILE_V1, PROFILE_V2, sha256, canonicalJson,
} from 'pikelet-wasm/complete/builder';
import { openPikeletFile, verifyHostEncoder, fuseCandidates, FUSION_DEFAULTS } from 'pikelet-wasm/complete';
import { openLexicalIndex } from '../packages/pikelet-wasm/complete/lexical.mjs';

const require = createRequire(import.meta.url);
const Pikelet = require('pikelet-wasm');
const { exportSketchArtifact } = require('pikelet-wasm/artifact');
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
    const sketchPath = path.join(tmp, 'synthetic.pikelet-sketch');
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
    const built = assemblePikeletFile(manifest, segments, outPath);
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
    const search = await openPikeletFile(src, { encodeQuery: hostEncode });
    const info = search.info();
    check('opens: format 2 / pikelet-complete-v2', info.formatVersion === 2 && info.profile === PROFILE_V2);
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

    const truth = layoutOf(A.bytes);
    const layout = info.layout;
    check('info.layout starts with the header and lists every segment at its table offset',
        layout[0].kind === 'header' && layout[0].offset === 0
        && layout.length === 1 + Object.keys(truth.segments).length
        && layout.slice(1).every((s) => truth.segments[s.kind]?.offset === s.offset && truth.segments[s.kind].length === s.length),
        JSON.stringify(layout));
    check('info.layout is in file order, header ending at the first segment',
        layout[0].length === layout[1].offset && layout.every((s, i) => i === 0 || s.offset >= layout[i - 1].offset + layout[i - 1].length));
    const range = search.recordRange(17);
    const json = JSON.parse(A.bytes.subarray(range.offset, range.offset + range.length).toString('utf8'));
    check('recordRange(id) locates the record bytes inside the corpus segment',
        json.title === 'rec 17' && range.offset >= truth.segments.corpus.offset
        && range.offset + range.length <= truth.segments.corpus.offset + truth.segments.corpus.length);
    check('recordRange rejects ids outside the corpus', (() => { try { search.recordRange(COUNT); return false; } catch (err) { return /outside the corpus/.test(err.message); } })());
    await search.close();
}

// Source without a size (browser-style HTTP source before init()).
{
    const search = await openPikeletFile(memorySource(A.bytes, { withSize: false }), { encodeQuery: hostEncode });
    const q = await search.query('rec 42', { k: 1 });
    check('opens and queries through a source that reports no size', q.results[0].id === 42);
    await search.close();
}

// Host-encoder verification (contract 4.4 mode 2).
{
    const perturbed = (text) => { const v = hostEncode(text); v[0] += 0.01; return v; };
    await rejects('a host encoder that disagrees with the verification vectors is refused at open',
        () => openPikeletFile(memorySource(A.bytes), { encodeQuery: perturbed }), /failed the artifact's verification vector/);
    await rejects('a host encoder of the wrong dimension is refused at open',
        () => openPikeletFile(memorySource(A.bytes), { encodeQuery: () => new Float32Array(DIM + 1) }), /must return a 16-dimensional vector/);
    await rejects('a host encoder returning non-finite components is refused',
        () => openPikeletFile(memorySource(A.bytes), { encodeQuery: () => { const v = hostEncode('x'); v[3] = NaN; return v; } }), /non-finite/);
    await rejects('encodeQuery must be a function',
        () => openPikeletFile(memorySource(A.bytes), { encodeQuery: 42 }), /must be a function/);
    const noHost = await openPikeletFile(memorySource(A.bytes));
    check('without a host encoder the artifact opens and surfaces the requirement', noHost.info().encoderVerified === null);
    await rejects('...but refuses queries until one is supplied', () => noHost.query('rec 1'), /pass options.encodeQuery/);
    await noHost.close();
    const unverified = await openPikeletFile(memorySource(A.bytes), { encodeQuery: perturbed, verifyEncoder: false });
    check('verifyEncoder:false serves a wrong encoder, marked unverified', unverified.info().encoderVerified === false);
    await unverified.close();
    // Host encoders that return plain arrays are accepted and converted.
    const arrays = await openPikeletFile(memorySource(A.bytes), { encodeQuery: (t) => Array.from(hostEncode(t)) });
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
        () => openPikeletFile(memorySource(noVec.bytes), { encodeQuery: hostEncode }), /without verification vectors/);
    const allowed = await openPikeletFile(memorySource(noVec.bytes), { encodeQuery: hostEncode, allowUnverifiedEncoder: true });
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
    const t = await openPikeletFile(memorySource(tampered), { encodeQuery: hostEncode });
    check('open succeeds with a tampered record (records are lazy)', t.info().corpusIntegrity === 'per-record-sha256');
    await rejects('hydrating the tampered record fails its per-record digest', () => t.record(17), /record 17 failed integrity verification/);
    check('neighboring records in the same page still hydrate', (await t.record(16)).title === 'rec 16' && (await t.record(18)).title === 'rec 18');
    await rejects('a query that lands on the tampered record fails loudly', () => t.query('rec 17', { k: 1 }), /record 17 failed integrity/);
    await t.close();
    const unverified = await openPikeletFile(memorySource(tampered), { encodeQuery: hostEncode, verifyRecords: false });
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
    const dt = await openPikeletFile(memorySource(digestTampered), { encodeQuery: hostEncode });
    await rejects('a tampered record digest fails its digest page against the page table', () => dt.record(100), /digest page 1 failed hash verification/);
    check('records in other pages are unaffected', (await dt.record(3)).title === 'rec 3');
    await dt.close();

    // Tamper the page table itself: refused at open (it is checked against
    // the identity-bound manifest).
    const pageTampered = Buffer.from(A.bytes);
    pageTampered[pageTableAt + 7] ^= 0xff;
    await rejects('a tampered page table is refused at open', () => openPikeletFile(memorySource(pageTampered), { encodeQuery: hostEncode }), /page table failed hash verification/);

    // Re-point record 17's offsets at record 18's bytes (still monotonic):
    // structure passes, the record digest does not.
    const swapped = Buffer.from(A.bytes);
    swapped.writeBigUInt64LE(BigInt(recStart(18) - corpus.offset), offsetsAt + 8 * 17);
    const sw = await openPikeletFile(memorySource(swapped), { encodeQuery: hostEncode });
    await rejects('an offsets table that maps a record to other bytes fails the per-record digest', () => sw.record(17), /record 17 failed integrity/);
    await sw.close();

    // Non-monotonic offsets: structural rejection before any digest work.
    const nonMono = Buffer.from(A.bytes);
    nonMono.writeBigUInt64LE(BigInt(recStart(30) - corpus.offset), offsetsAt + 8 * 5);
    await rejects('non-monotonic corpus offsets are rejected at open', () => openPikeletFile(memorySource(nonMono), { encodeQuery: hostEncode }), /offsets are inconsistent/);
    const badFirst = Buffer.from(A.bytes);
    badFirst.writeBigUInt64LE(BigInt(recStart(0) - corpus.offset + 1), offsetsAt);
    await rejects('offsets[0] must start exactly where the tables end', () => openPikeletFile(memorySource(badFirst), { encodeQuery: hostEncode }), /offsets are inconsistent/);
    const pastEnd = Buffer.from(A.bytes);
    pastEnd.writeBigUInt64LE(BigInt(corpus.length + 1), offsetsAt + 8 * COUNT);
    await rejects('an offset past the segment is rejected at open', () => openPikeletFile(memorySource(pastEnd), { encodeQuery: hostEncode }), /offsets are inconsistent/);
    const huge = Buffer.from(A.bytes);
    huge.writeBigUInt64LE(BigInt('9007199254740993'), offsetsAt + 8 * COUNT); // 2^53 + 1
    await rejects('an offset beyond the safe-integer range is rejected', () => openPikeletFile(memorySource(huge), { encodeQuery: hostEncode }), /safe integer range/);
}

// Manifest / header / segment-table hostility.
{
    const { manifestBytes, segments } = layoutOf(A.bytes);
    const m = Buffer.from(A.bytes); m[64 + 10] ^= 0x01;
    await rejects('a tampered manifest fails identity verification', () => openPikeletFile(memorySource(m), { encodeQuery: hostEncode }), /identity verification/);
    const id = Buffer.from(A.bytes); id[24 + 3] ^= 0x01;
    await rejects('a tampered identity fails identity verification', () => openPikeletFile(memorySource(id), { encodeQuery: hostEncode }), /identity verification/);
    const ver = Buffer.from(A.bytes); ver.writeUInt32LE(3, 4);
    await rejects('format version 3 is rejected explicitly', () => openPikeletFile(memorySource(ver), { encodeQuery: hostEncode }), /unsupported .pikelet format version 3/);
    const magic = Buffer.from(A.bytes); magic[0] ^= 0x01;
    await rejects('bad magic is rejected', () => openPikeletFile(memorySource(magic), { encodeQuery: hostEncode }), /bad magic/);
    const v1Header = Buffer.from(A.bytes); v1Header.writeUInt32LE(1, 4);
    await rejects('a format-1 header over a v2 manifest is rejected', () => openPikeletFile(memorySource(v1Header), { encodeQuery: hostEncode }), /unsupported profile/);
    const bigManifest = Buffer.from(A.bytes); bigManifest.writeUInt32LE(17 * 1024 * 1024, 8);
    await rejects('an implausible manifest length is rejected before reading', () => openPikeletFile(memorySource(bigManifest), { encodeQuery: hostEncode }), /implausible/);
    const manySegs = Buffer.from(A.bytes); manySegs.writeUInt32LE(65, 12);
    await rejects('an implausible segment count is rejected before reading', () => openPikeletFile(memorySource(manySegs), { encodeQuery: hostEncode }), /implausible/);
    const fileBytesUp = Buffer.from(A.bytes); fileBytesUp.writeBigUInt64LE(BigInt(A.bytes.length + 16), 16);
    await rejects('fileBytes disagreeing with the source size is rejected', () => openPikeletFile(memorySource(fileBytesUp), { encodeQuery: hostEncode }), /truncated or padded/);
    const fileBytesHuge = Buffer.from(A.bytes); fileBytesHuge.writeBigUInt64LE(BigInt('18446744073709551615'), 16);
    await rejects('fileBytes beyond the safe-integer range is rejected', () => openPikeletFile(memorySource(fileBytesHuge, { withSize: false }), { encodeQuery: hostEncode }), /safe integer range/);
    // Segment table: push the corpus segment's length past the file.
    const tableAt = 64 + manifestBytes;
    const segLen = Buffer.from(A.bytes); segLen.writeBigUInt64LE(BigInt(segments.corpus.length + 64), tableAt + 48 * 1 + 16);
    await rejects('a segment-table length disagreeing with the manifest is rejected', () => openPikeletFile(memorySource(segLen), { encodeQuery: hostEncode }), /segment table disagrees/);
    const segOff = Buffer.from(A.bytes); segOff.writeBigUInt64LE(BigInt(segments.corpus.offset + 16), tableAt + 48 * 1 + 8);
    await rejects('a segment-table offset breaking the packed layout is rejected', () => openPikeletFile(memorySource(segOff), { encodeQuery: hostEncode }), /segment table disagrees/);
    // Truncation with an unsized source: the eager reads succeed, the lazy
    // evaluation read comes back short and is refused as such.
    const truncated = A.bytes.subarray(0, A.bytes.length - 10);
    const tr = await openPikeletFile(memorySource(truncated, { withSize: false }), { encodeQuery: hostEncode });
    await rejects('a short read from a truncated source is refused (exact-length reads)', () => tr.evaluation(), /read returned .* of .* bytes/);
    await tr.close();
    await rejects('a truncated sized source is refused at open', () => openPikeletFile(memorySource(truncated), { encodeQuery: hostEncode }), /truncated or padded/);
    // Evaluation tamper.
    const ev = Buffer.from(A.bytes); ev[ev.length - 3] ^= 0xff;
    const evr = await openPikeletFile(memorySource(ev), { encodeQuery: hostEncode });
    await rejects('a tampered evaluation segment fails its digest', () => evr.evaluation(), /evaluation segment failed hash/);
    await evr.close();
    // Query-interp tamper: eager, refused at open.
    const qi = Buffer.from(A.bytes); qi[segments['query-interp'].offset + 20] ^= 0x01;
    await rejects('a tampered query-interp segment is refused at open', () => openPikeletFile(memorySource(qi), { encodeQuery: hostEncode }), /query-interp segment failed hash/);
}

// Read budgets.
{
    await rejects('maxReadBytes below the manifest size refuses the open with the budget named',
        () => openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode, maxReadBytes: 64 }), /exceeds the 64-byte read budget/);
    await rejects('maxReadBytes must be positive', () => openPikeletFile(memorySource(A.bytes), { maxReadBytes: -1 }), /must be a positive number or Infinity/);
    await rejects('maxReadBytes must be a number', () => openPikeletFile(memorySource(A.bytes), { maxReadBytes: '1mb' }), /must be a positive number or Infinity/);
    const inf = await openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode, maxReadBytes: Infinity });
    check('maxReadBytes: Infinity defers to the absolute backstop', (await inf.query('rec 2', { k: 1 })).results[0].id === 2);
    await inf.close();
    const tinyRecords = await openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode, maxRecordBytes: 8 });
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
    assemblePikeletFile({
        profile: PROFILE_V2, corpus: { ...spoofCorpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
        encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
    }, [
        { kind: 'index', bytes: SKETCH },
        { kind: 'corpus', bytes: spoofCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(kind2Declaration()), 'utf8'), CALIBRATION) },
        { kind: 'evaluation', bytes: EVALUATION },
    ], spoofPath);
    const spoof = await openPikeletFile(spoofPath, { encodeQuery: hostEncode });
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
        const fl = await openPikeletFile(flaky, { encodeQuery: hostEncode });
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
            const { httpRangeSource: rangeSource } = await import('../packages/pikelet-wasm/complete/sources.mjs');
            const src = rangeSource('http://host.invalid/b.pikelet');
            await src.init();
            await rejects('a 206 answering a different range than requested is refused', () => src.read(64, 64), /returned Content-Range bytes 80-143.*requested bytes 64-127/);
        } finally {
            globalThis.fetch = realFetch2;
        }
    }
    // httpRangeSource: a 200 with a changed ETag means the artifact was
    // replaced, not that the host ignores Range — it must not be treated
    // as a fallback-worthy full download.
    {
        const realFetch3 = globalThis.fetch;
        globalThis.fetch = async (url, init = {}) => {
            if (init.method === 'HEAD') {
                return new Response(null, { status: 200, headers: { 'content-length': String(A.bytes.length), 'accept-ranges': 'bytes', etag: '"v1"' } });
            }
            // The host answers with 200 and a different ETag: the file
            // behind the URL changed since the HEAD pinned "v1".
            return new Response(A.bytes, { status: 200, headers: { 'content-length': String(A.bytes.length), etag: '"v2"' } });
        };
        try {
            const { httpRangeSource: rangeSource } = await import('../packages/pikelet-wasm/complete/sources.mjs');
            const src = rangeSource('http://host.invalid/c.pikelet');
            await src.init();
            await rejects('a 200 with a mismatched ETag is reported as a changed artifact, not a Range-ignoring host',
                () => src.read(0, 64), /artifact changed.*expected ETag "v1".*got "v2"/);
        } finally {
            globalThis.fetch = realFetch3;
        }
    }
    // 4. httpRangeSource: after the one-time full-download fallback, reads
    // are served from memory and issue no further requests.
    const { httpRangeSource } = await import('../packages/pikelet-wasm/complete/sources.mjs');
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
        () => openPikeletFile(memorySource(metricFlip), { encodeQuery: hostEncode }), /index header failed hash verification/);
    // A header field the sketch never self-verifies (recommendedRerank at
    // offset 120) is caught ONLY by the manifest commitment — this is the
    // deterministic proof of the anchor.
    const rerankFlip = Buffer.from(A.bytes);
    rerankFlip.writeUInt32LE(7, segments.index.offset + 120);
    await rejects('format 2: a tampered sketch header field the sketch never self-checks fails the manifest commitment',
        () => openPikeletFile(memorySource(rerankFlip), { encodeQuery: hostEncode }), /index header failed hash verification/);
    // Rewriting the sketch's own residentSha256 fails both the manifest
    // commitment and the sketch's resident self-check; they run in the same
    // open wave, so whichever rejects first surfaces — either way the open
    // fails closed.
    const selfHash = Buffer.from(A.bytes);
    selfHash[segments.index.offset + 60] ^= 0xff;
    await rejects('format 2: a rewritten sketch self-hash fails the open (manifest commitment or resident self-check)',
        () => openPikeletFile(memorySource(selfHash), { encodeQuery: hostEncode }),
        /index header failed hash verification|resident prefix failed hash verification/);
    // A host that serves different bytes for the same range: the forged
    // header (recommendedRerank rewritten — nothing in the sketch self-checks
    // it) on every read of the header except the second. When the reader
    // checked the commitment on its own separate read, the sketch parsed the
    // forged copy and the genuine one satisfied the check, so the open
    // succeeded under the pinned identity. The commitment must bind the
    // bytes the sketch actually parses.
    {
        const hs = segments.index.offset;
        let headerReads = 0;
        const inconsistent = {
            size: A.bytes.length,
            preferredParallelism: Infinity,
            preferredGapBytes: 0,
            async read(offset, length) {
                if (offset < hs + 256 && offset + length > hs) {
                    headerReads++;
                    return (headerReads === 2 ? A.bytes : rerankFlip).subarray(offset, offset + length);
                }
                return A.bytes.subarray(offset, offset + length);
            },
            async close() {},
        };
        await rejects('format 2: a host serving the genuine header to only one read cannot slip a forged header past the commitment',
            () => openPikeletFile(inconsistent, { encodeQuery: hostEncode }), /index header failed hash verification/);
    }
    // Format 1 has no header commitment; the metric cross-check against the
    // identity-verified manifest is the binding there.
    const v1 = buildSynthetic({ profile: PROFILE_V1, name: 'v1metric' });
    const v1layout = layoutOf(v1.bytes);
    const v1flip = Buffer.from(v1.bytes);
    v1flip.writeUInt32LE(0, v1layout.segments.index.offset + 12);
    await rejects('format 1: a flipped index metric fails the manifest cross-check',
        () => openPikeletFile(memorySource(v1flip), { encodeQuery: hostEncode }), /index metric l2 != manifest metric cosine/);
}

// Lazy vector rows: with a format-2 embedded sketch (the default build),
// every rerank row is verified on the read that fetches it, per-read.
{
    const { segments } = layoutOf(A.bytes);
    // Tamper a byte near the end of the index segment — inside the last
    // block's rows, far past the resident prefix.
    const rowTamper = Buffer.from(A.bytes);
    rowTamper[segments.index.offset + segments.index.length - 8] ^= 0xff;
    const lazy = await openPikeletFile(memorySource(rowTamper), { encodeQuery: hostEncode });
    check('the reader reports per-row index integrity for the format-2 sketch',
        lazy.info().indexRowIntegrity === 'per-row-sha256' && lazy.info().vectorsVerified === false);
    await rejects('a query whose rerank touches the tampered row fails its per-row digest',
        () => lazy.query('rec 299', { k: 3 }), /row failed digest verification|digest page failed/);
    await rejects('...and verifyVectors() detects it too', () => lazy.verifyVectors(), /vector|hash/i);
    await lazy.close();
    await rejects('verifyIndexVectors: true refuses the tamper at open',
        () => openPikeletFile(memorySource(rowTamper), { encodeQuery: hostEncode, verifyIndexVectors: true }), /vector|hash/i);
    const clean = await openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode, verifyIndexVectors: true });
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
            () => openPikeletFile(memorySource(bad.bytes), { encodeQuery: wrongEncoder }), /expected embedding/);
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
        () => openPikeletFile(memorySource(good.bytes), { encodeQuery: wrongEncoder }), /failed the artifact's verification vector/);
}

// Builders accept plain Uint8Arrays wherever the typings say Uint8Array.
{
    const u8records = RECORDS.map((b) => new Uint8Array(b)); // copies, not Buffers
    const u8corpus = buildCorpusSegment(u8records, { pageRecords: PAGE_RECORDS });
    const u8qi = buildQueryInterpSegment(2,
        new Uint8Array(Buffer.from(JSON.stringify(kind2Declaration()), 'utf8')),
        new Uint8Array(CALIBRATION));
    const u8path = path.join(tmp, 'u8.pikelet');
    assemblePikeletFile({
        profile: PROFILE_V2, corpus: { ...u8corpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
        encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
    }, [
        { kind: 'index', bytes: new Uint8Array(SKETCH) },
        { kind: 'corpus', bytes: u8corpus.bytes },
        { kind: 'query-interp', bytes: u8qi },
        { kind: 'evaluation', bytes: new Uint8Array(EVALUATION) },
    ], u8path);
    const u8 = await openPikeletFile(u8path, { encodeQuery: hostEncode });
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
        assemblePikeletFile({
            profile: PROFILE_V2, corpus: { ...corpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
            encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
        }, [
            { kind: 'index', bytes: SKETCH },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(kind2Declaration()), 'utf8'), CALIBRATION) },
        ], outPath);
        return fs.readFileSync(outPath);
    })();
    const ne = await openPikeletFile(memorySource(noEval), { encodeQuery: hostEncode });
    check('evaluation() is null when the artifact carries no evaluation segment', (await ne.evaluation()) === null);
    await ne.close();
    const arrayEval = buildSynthetic({ name: 'arrayeval' });
    const { segments: ae } = layoutOf(arrayEval.bytes);
    // Rebuild with an array evaluation segment.
    const corpus = buildCorpusSegment(RECORDS, { pageRecords: PAGE_RECORDS });
    const outPath = path.join(tmp, 'arrayeval2.pikelet');
    assemblePikeletFile({
        profile: PROFILE_V2, corpus: { ...corpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
        encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
    }, [
        { kind: 'index', bytes: SKETCH },
        { kind: 'corpus', bytes: corpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(kind2Declaration()), 'utf8'), CALIBRATION) },
        { kind: 'evaluation', bytes: Buffer.from('[1,2]', 'utf8') },
    ], outPath);
    const av = await openPikeletFile(outPath, { encodeQuery: hostEncode });
    await rejects('a non-object evaluation segment is refused', () => av.evaluation(), /must be a JSON object/);
    await av.close();
    void ae;
}

// httpRangeSource fails closed on malformed 206s and streaming overruns.
{
    const realFetch3 = globalThis.fetch;
    try {
        const { httpRangeSource: rangeSource } = await import('../packages/pikelet-wasm/complete/sources.mjs');
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
    const ex = await openPikeletFile(memorySource(extra.bytes), { encodeQuery: hostEncode });
    check('an unknown segment kind is skipped and the file still serves', (await ex.query('rec 7', { k: 1 })).results[0].id === 7);
    await ex.close();
    const dup = buildSynthetic({ extraSegments: [{ kind: 'corpus', bytes: buildCorpusSegment(RECORDS.slice(0, 2)).bytes }], name: 'dup' });
    await rejects('a duplicate corpus segment is rejected', () => openPikeletFile(memorySource(dup.bytes), { encodeQuery: hostEncode }), /more than one corpus segment/);
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
        () => openPikeletFile(memorySource(inflated.bytes), { encodeQuery: hostEncode }), /implausible for the corpus segment/);
    const badPages = buildSynthetic({ name: 'badpages', manifestPatch: (m) => { m.corpus.pages += 1; } });
    await rejects('an inconsistent integrity block is rejected', () => openPikeletFile(memorySource(badPages.bytes), { encodeQuery: hostEncode }), /integrity block is inconsistent/);
    const badDim = buildSynthetic({ name: 'baddim', manifestPatch: (m) => { m.dim = DIM + 1; } });
    await rejects('a kind-2 declaration dim disagreeing with the manifest is rejected',
        () => openPikeletFile(memorySource(badDim.bytes), { encodeQuery: () => new Float32Array(DIM + 1) }), /disagrees with manifest dim/);
}

// ---------------------------------------------------------------------------
// B. format 1 compatibility (whole-segment integrity only)
// ---------------------------------------------------------------------------
console.log('\nB. format-1 file (pikelet-complete-v1) compatibility');
{
    const B = buildSynthetic({ profile: PROFILE_V1, name: 'v1' });
    const view = new DataView(B.bytes.buffer, B.bytes.byteOffset, B.bytes.byteLength);
    check('builder writes header formatVersion 1 for the v1 profile', view.getUint32(4, true) === 1);
    const search = await openPikeletFile(memorySource(B.bytes), { encodeQuery: hostEncode });
    const info = search.info();
    check('opens: format 1 / pikelet-complete-v1', info.formatVersion === 1 && info.profile === PROFILE_V1);
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
    const t = await openPikeletFile(memorySource(tampered), { encodeQuery: hostEncode });
    const rec = await t.record(17);
    check('format 1: a record tamper is NOT detected on hydration (transitional stance, spec 6)', rec.text.includes('c`rries'));
    await t.close();
    await rejects('a v2 manifest under a format-1 header is rejected', () => {
        const bad = buildSynthetic({ profile: PROFILE_V1, name: 'v1bad', manifestPatch: (m) => { m.corpus.layout = 'records-v2'; } });
        return openPikeletFile(memorySource(bad.bytes));
    }, /carries corpus layout v1|records-v2/);
}

// ---------------------------------------------------------------------------
// C. kind-1 artifact from the committed encoder-conformance assets (format 2)
// ---------------------------------------------------------------------------
console.log('\nC. kind-1 student-inline artifact compiled from encoder-conformance assets');
{
    const fixtureDir = path.join(ROOT, 'test', 'fixtures', 'encoder-conformance');
    const assets = path.join(fixtureDir, 'assets');
    const corpusRaw = JSON.parse(fs.readFileSync(path.join(assets, 'docs-corpus.json'), 'utf8'));
    const sourceManifest = JSON.parse(fs.readFileSync(path.join(assets, 'docs-manifest.json'), 'utf8'));
    const goldens = JSON.parse(fs.readFileSync(path.join(fixtureDir, 'abstention-golden.json'), 'utf8'));
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
        return assemblePikeletFile({
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
    const search = await openPikeletFile(first.outPath);
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

    // options.blobCache: the resident index prefix (residentSha256) is
    // cached under its digest, so a second open reads it from the cache
    // instead of the source; a corrupt or faulting cache is only ever a miss.
    {
        const fileBytes = fs.readFileSync(first.outPath);
        const countingSource = () => {
            const src = memorySource(fileBytes);
            src.bytes = 0;
            const read = src.read.bind(src);
            src.read = async (offset, length) => { src.bytes += length; return read(offset, length); };
            return src;
        };
        const mapCache = () => {
            const map = new Map();
            const log = { gets: 0, puts: 0, deletes: 0, putKeys: [], deleteKeys: [] };
            return {
                map, log,
                get: async (key) => { log.gets++; return map.get(key) ?? null; },
                put: async (key, bytes) => { log.puts++; log.putKeys.push(key); map.set(key, Uint8Array.from(bytes)); },
                delete: async (key) => { log.deletes++; log.deleteKeys.push(key); map.delete(key); },
                putsOf: (key) => log.putKeys.filter((k) => k === key).length,
            };
        };
        const settle = () => new Promise((resolve) => setImmediate(resolve));
        const probe = evaluation.goldenQueries.find((g) => g.expected !== 'none')?.text || 'what is this';
        const idsOf = (out) => out.results.map((r) => r.id).join(',');

        const plainSrc = countingSource();
        const plain = await openPikeletFile(plainSrc);
        const openBytes = plainSrc.bytes;
        const expected = idsOf(await plain.query(probe, { k: 5 }));
        await plain.close();

        const cache = mapCache();
        const coldSrc = countingSource();
        const cold = await openPikeletFile(coldSrc, { blobCache: cache });
        const coldBytes = coldSrc.bytes;
        await settle();
        const residentKey = [...cache.map.keys()][0];
        check('blobCache: a cold open stores the resident prefix under its digest',
            cache.log.puts === 1 && /^[0-9a-f]{64}$/.test(residentKey || ''), JSON.stringify(cache.log));
        check('blobCache: a cold open reads what an uncached open reads', coldBytes === openBytes, `${coldBytes} vs ${openBytes}`);
        await cold.close();

        const warmSrc = countingSource();
        const warm = await openPikeletFile(warmSrc, { blobCache: cache });
        const residentBytes = cache.map.get(residentKey).byteLength;
        check('blobCache: a warm open skips the resident read',
            warmSrc.bytes === openBytes - residentBytes, `${warmSrc.bytes} read, ${openBytes} - ${residentBytes} expected`);
        check('blobCache: a warm open reports the resident prefix verified', warm.info().residentVerified === true);
        check('blobCache: a warm open answers identically', idsOf(await warm.query(probe, { k: 5 })) === expected);
        await settle();
        check('blobCache: a hit is not rewritten', cache.putsOf(residentKey) === 1, JSON.stringify(cache.log.putKeys.slice(0, 4)));
        await warm.close();

        // Per-query pieces: the warm query above stored the rerank rows (by
        // their digests), the corpus records it hydrated, and the digest
        // pages that verify both. A later process asking the same question
        // reads none of them from the source.
        const plainQuerySrc = countingSource();
        const plainQ = await openPikeletFile(plainQuerySrc);
        const plainOpenBytes = plainQuerySrc.bytes;
        await plainQ.query(probe, { k: 5 });
        const uncachedQueryBytes = plainQuerySrc.bytes - plainOpenBytes;
        await plainQ.close();
        const infoReader = await openPikeletFile(countingSource());
        const plainInfo = infoReader.info();
        await infoReader.close();
        check('blobCache: the fixture verifies rows and records per read (so both are cacheable)',
            plainInfo.indexRowIntegrity === 'per-row-sha256' && plainInfo.corpusIntegrity === 'per-record-sha256',
            JSON.stringify([plainInfo.indexRowIntegrity, plainInfo.corpusIntegrity]));
        const perQueryKeys = new Set(cache.log.putKeys.filter((k) => k !== residentKey));
        check('blobCache: a query stores rows, records and digest pages',
            perQueryKeys.size > 5 && [...perQueryKeys].some((k) => k.length < 64), `${perQueryKeys.size} keys`);
        const againSrc = countingSource();
        const again = await openPikeletFile(againSrc, { blobCache: cache });
        const againOpen = againSrc.bytes;
        const againIds = idsOf(await again.query(probe, { k: 5 }));
        check('blobCache: the same question in a later process reads nothing from the source',
            againSrc.bytes === againOpen && uncachedQueryBytes > 0 && againIds === expected,
            `${againSrc.bytes - againOpen} bytes (uncached query: ${uncachedQueryBytes})`);
        await again.close();
        // A corrupt row entry: a miss, deleted, refetched, same answer.
        const rowKey = [...perQueryKeys].find((k) => k.length < 64);
        cache.map.get(rowKey)[0] ^= 0xff;
        const rowSrc = countingSource();
        const rowHeal = await openPikeletFile(rowSrc, { blobCache: cache });
        const rowOpen = rowSrc.bytes;
        const rowIds = idsOf(await rowHeal.query(probe, { k: 5 }));
        check('blobCache: a corrupt row entry is a miss, deleted, and the row refetched',
            cache.log.deleteKeys.includes(rowKey) && rowSrc.bytes > rowOpen && rowIds === expected,
            `${rowSrc.bytes - rowOpen} bytes refetched`);
        await settle();
        check('blobCache: the refetched row is stored again', cache.map.has(rowKey));
        await rowHeal.close();

        // A flipped byte: the hit fails its hash, is deleted, the source
        // serves the prefix, and the verified bytes are stored again.
        cache.map.get(residentKey)[17] ^= 0xff;
        const healSrc = countingSource();
        const healed = await openPikeletFile(healSrc, { blobCache: cache });
        const healBytes = healSrc.bytes;
        await settle();
        check('blobCache: a corrupt entry is a miss, deleted and refetched',
            cache.log.deleteKeys.filter((k) => k === residentKey).length === 1 && healBytes === openBytes && cache.putsOf(residentKey) === 2,
            `${cache.log.deleteKeys.length} deletes, ${cache.putsOf(residentKey)} puts; ${healBytes} vs ${openBytes}`);
        check('blobCache: the refetched entry is the committed bytes again',
            sha256(Buffer.from(cache.map.get(residentKey))).toString('hex') === residentKey);
        check('blobCache: an open over a corrupt entry answers identically', idsOf(await healed.query(probe, { k: 5 })) === expected);
        await healed.close();

        // A truncated entry and a faulting cache are misses too.
        cache.map.set(residentKey, cache.map.get(residentKey).subarray(1));
        const short = await openPikeletFile(countingSource(), { blobCache: cache });
        check('blobCache: a wrong-length entry is a miss', idsOf(await short.query(probe, { k: 5 })) === expected);
        await short.close();
        const faulty = await openPikeletFile(countingSource(), {
            blobCache: { get: async () => { throw new Error('disk on fire'); }, put: async () => { throw new Error('disk on fire'); } },
        });
        check('blobCache: a cache whose get and put throw never fails an open', idsOf(await faulty.query(probe, { k: 5 })) === expected);
        await faulty.close();

        // The Node disk store: sharded files, atomic writes, LRU budget.
        const { diskBlobCache, defaultBlobCacheDir } = await import('../packages/pikelet-wasm/complete/blob-cache.node.mjs');
        const diskDir = path.join(tmp, 'blobs');
        const disk = diskBlobCache(diskDir);
        const diskCold = await openPikeletFile(countingSource(), { blobCache: disk });
        await settle();
        await disk.flush();
        // This fixture's resident prefix is 22 KB, under SMALL_ENTRY_BYTES,
        // so it lands in the log; the file layout is checked below with
        // 64 KiB entries.
        check('diskBlobCache: a small resident prefix is stored in the log',
            residentBytes < 64 * 1024 && fs.readdirSync(path.join(diskDir, 'small')).length === 1);
        await diskCold.close();
        const diskWarmSrc = countingSource();
        // A store that has served a log hit holds a read handle; Node 26
        // treats a FileHandle left to garbage collection as an error, so
        // every instance here is closed.
        const diskWarmCache = diskBlobCache(diskDir);
        const diskWarm = await openPikeletFile(diskWarmSrc, { blobCache: diskWarmCache });
        const diskWarmBytes = diskWarmSrc.bytes;
        check('diskBlobCache: a second process-style open reads the prefix from disk',
            diskWarmBytes === openBytes - residentBytes && idsOf(await diskWarm.query(probe, { k: 5 })) === expected,
            `${diskWarmBytes} vs ${openBytes} - ${residentBytes}`);
        await diskWarm.close();
        await diskWarmCache.close();
        check('diskBlobCache: get on a missing key is null', await disk.get('0'.repeat(64)) === null);
        let badKey = null;
        try { await disk.get('../escape'); } catch (err) { badKey = err; }
        check('diskBlobCache: a key that is not a sha256 digest is refused', /sha256/.test(badKey?.message || ''));
        await disk.close();

        const lruDir = path.join(tmp, 'blobs-lru');
        // Entries of 64 KiB and up are files, under maxBytes.
        const BIG = 64 * 1024;
        const lru = diskBlobCache(lruDir, { maxBytes: 2.5 * BIG });
        const keyOf = (n) => n.toString(16).padStart(64, '0');
        lru.put(keyOf(1), new Uint8Array(BIG)); await lru.flush();
        check('diskBlobCache: an entry of 64 KiB or more lands at <dir>/<2 hex>/<digest>, with no temp file left',
            fs.existsSync(path.join(lruDir, '00', keyOf(1)))
            && fs.readdirSync(path.join(lruDir, '00')).every((n) => !n.endsWith('.tmp')));
        lru.put(keyOf(2), new Uint8Array(BIG)); await lru.flush();
        // Touch 1 so 2 is the least recently used when 3 pushes past 2.5 entries.
        const past = new Date(Date.now() - 60000);
        fs.utimesSync(path.join(lruDir, '00', keyOf(2)), past, past);
        await lru.get(keyOf(1));
        await settle();
        lru.put(keyOf(3), new Uint8Array(BIG)); await lru.flush();
        check('diskBlobCache: past maxBytes the least recently used entry is evicted',
            await lru.get(keyOf(2)) === null && (await lru.get(keyOf(1)))?.byteLength === BIG
            && (await lru.get(keyOf(3)))?.byteLength === BIG);
        lru.put(keyOf(4), new Uint8Array(4 * BIG)); await lru.flush();
        check('diskBlobCache: an entry larger than the whole budget is not stored', await lru.get(keyOf(4)) === null);
        await lru.close();

        // Entries under 64 KiB go to the log: append-only generations with an
        // in-memory index, rebuilt by the next instance (the next process).
        const logDir = path.join(tmp, 'blobs-log');
        const small = diskBlobCache(logDir, { smallMaxBytes: 4 * 4096 });
        const smallKey = (n) => n.toString(16).padStart(32, '0');
        const filled = (n, len) => new Uint8Array(len).fill(n);
        for (let n = 1; n <= 3; n++) small.put(smallKey(n), filled(n, 1000));
        await small.flush();
        check('diskBlobCache: small entries share a log file, not a file each',
            fs.readdirSync(path.join(logDir, 'small')).length === 1 && !fs.existsSync(path.join(logDir, '00')));
        check('diskBlobCache: a small entry reads back from the log', (await small.get(smallKey(2)))?.[999] === 2);
        await small.close();
        const reopened = diskBlobCache(logDir, { smallMaxBytes: 4 * 4096 });
        check('diskBlobCache: a new instance indexes the log it finds (another process\'s entries)',
            (await reopened.get(smallKey(3)))?.[0] === 3 && (await reopened.get(smallKey(1)))?.length === 1000);
        // A torn tail: everything before it still reads.
        const genFile = path.join(logDir, 'small', fs.readdirSync(path.join(logDir, 'small'))[0]);
        fs.appendFileSync(genFile, Buffer.from([0x50, 0x4b, 0x42, 0x31, 16, 1, 2]));
        await reopened.close();
        const torn = diskBlobCache(logDir, { smallMaxBytes: 4 * 4096 });
        check('diskBlobCache: a torn record at the end of a log is ignored', (await torn.get(smallKey(3)))?.[0] === 3);
        torn.put(smallKey(4), filled(4, 1000)); await torn.flush();
        const afterTorn = diskBlobCache(logDir, { smallMaxBytes: 4 * 4096 });
        check('diskBlobCache: writes after a torn tail go to a new generation and stay readable',
            (await afterTorn.get(smallKey(4)))?.[0] === 4 && fs.readdirSync(path.join(logDir, 'small')).length === 2);
        await afterTorn.close();
        // Generations are 4096 bytes here; writing well past four of them
        // drops the oldest whole, and entry 1 was never read again.
        for (let n = 10; n < 30; n++) { torn.put(smallKey(n), filled(n, 1000)); await torn.flush(); }
        check('diskBlobCache: the log keeps four generations and drops the oldest whole',
            fs.readdirSync(path.join(logDir, 'small')).length === 4 && await torn.get(smallKey(1)) === null
            && (await torn.get(smallKey(29)))?.[0] === 29);
        await torn.close();
        let shortKey = null;
        try { await torn.get('ab'.repeat(15)); } catch (err) { shortKey = err; }
        check('diskBlobCache: a key under 128 bits is refused', /32-64 hex digits/.test(shortKey?.message || ''));
        const off = diskBlobCache(path.join(tmp, 'blobs-nolog'), { smallMaxBytes: 0 });
        await off.put(smallKey(1), filled(1, 100)); await off.flush();
        check('diskBlobCache: smallMaxBytes 0 stores no small entries', await off.get(smallKey(1)) === null);
        await off.close();
        if (process.platform !== 'win32') {
            // Entries can be record text from a private pack: owner-only,
            // whatever the umask, and a root an older store left at 0755 is
            // tightened (which also closes off the 0644 entries it holds).
            const modeOf = (p) => fs.statSync(p).mode & 0o777;
            const privDir = path.join(tmp, 'blobs-private');
            fs.mkdirSync(privDir, { mode: 0o755 });
            fs.chmodSync(privDir, 0o755);
            const oldMask = process.umask(0o022);
            try {
                const priv = diskBlobCache(privDir);
                priv.put(smallKey(1), filled(1, 100));
                priv.put(keyOf(1), new Uint8Array(BIG));
                await priv.close();
            } finally {
                process.umask(oldMask);
            }
            const genName = fs.readdirSync(path.join(privDir, 'small'))[0];
            check('diskBlobCache: directories are 0700 and entries 0600 under umask 022, and an existing root is tightened',
                modeOf(privDir) === 0o700 && modeOf(path.join(privDir, 'small')) === 0o700
                && modeOf(path.join(privDir, 'small', genName)) === 0o600
                && modeOf(path.join(privDir, '00')) === 0o700 && modeOf(path.join(privDir, '00', keyOf(1))) === 0o600,
                [privDir, path.join(privDir, 'small'), path.join(privDir, 'small', genName), path.join(privDir, '00'), path.join(privDir, '00', keyOf(1))]
                    .map((p) => modeOf(p).toString(8)).join(' '));
        }
        check('defaultBlobCacheDir honors PIKELET_CACHE_DIR, then XDG_CACHE_HOME',
            defaultBlobCacheDir({ PIKELET_CACHE_DIR: '/x' }) === path.resolve('/x', 'blobs')
            && defaultBlobCacheDir({ XDG_CACHE_HOME: '/y' }) === path.join(path.resolve('/y'), 'pikelet', 'blobs'));
    }
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
    const search = await openPikeletFile(memorySource(H.bytes), { encodeQuery: hostEncode });
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

    const vectorOnly = await openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode });
    const control = await vectorOnly.query('record number 17', { k: 3 });
    check('vector-only artifact misses the same known-item lookup (control)',
        !control.results.some((r) => r.id === 17), JSON.stringify(control.results.map((r) => r.id)));
    check('vector-only artifact reports no lexical index', vectorOnly.info().lexical === null);
    await vectorOnly.close();

    const tampered = Buffer.from(H.bytes);
    const seg = layoutOf(tampered).segments.lexical;
    tampered[seg.offset + Math.floor(seg.length / 2)] ^= 0xff;
    await rejects('tampered lexical segment fails hash verification at open',
        () => openPikeletFile(memorySource(tampered), { encodeQuery: hostEncode }),
        /lexical segment failed hash/);

    // The lazy opener (wiki-scale path: interpolation search over remote
    // ranges) must score identically to the eager reader on the same
    // segment bytes.
    const { openLexicalIndexLazy } = await import('../packages/pikelet-wasm/complete/lexical.mjs');
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

    // Lazy term entries are not individually verified, so the lazy reader
    // must hold each one to the segment before fetching its postings: a
    // forged entry must not steer a read outside the postings region or
    // request more bytes than a df-entry list can occupy.
    const lexView = new DataView(lexical.bytes.buffer, lexical.bytes.byteOffset, lexical.bytes.byteLength);
    const termCount = lexView.getUint32(8, true);
    const termTableOffset = lexView.getUint32(24, true);
    const postingsBytes = Number(lexView.getBigUint64(32, true));
    const forgeEntries = (patch) => {
        const forged = Buffer.from(lexical.bytes);
        const fv = new DataView(forged.buffer, forged.byteOffset, forged.byteLength);
        for (let i = 0; i < termCount; i++) patch(fv, termTableOffset + 24 * i);
        return forged;
    };
    const forgeries = [
        ['a postings offset past the postings region', (v, at) => v.setBigUint64(at + 8, BigInt(postingsBytes + 1024), true)],
        ['a postings length of ~4 GiB', (v, at) => v.setUint32(at + 16, 0xfffffff0, true)],
        ['a df larger than the corpus', (v, at) => v.setUint32(at + 20, COUNT + 1, true)],
        ['a postings length no df-entry list can occupy', (v, at) => v.setUint32(at + 20, 1, true)],
    ];
    for (const [label, patch] of forgeries) {
        const forged = forgeEntries(patch);
        const lazyForged = await openLexicalIndexLazy(
            async (off, len) => forged.subarray(off, off + len), forged.length);
        await rejects(`lazy lexical: ${label} is refused before its postings are read`,
            () => lazyForged.search('record number 17', 8), /lexical postings out of bounds/);
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
        () => openPikeletFile(counting, { expectedIdentity: 'f'.repeat(64) }), /identity mismatch/);
    check('the mismatch is decided on the header read alone', reads === 1, `${reads} reads`);
    await rejects('a malformed identity pin is rejected as such',
        () => openPikeletFile(memorySource(A.bytes), { expectedIdentity: 'not-a-hash' }), /sha256 hex/);
    const unpinned = await openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode });
    const trueIdentity = unpinned.info().identity;
    await unpinned.close();
    const pinned = await openPikeletFile(memorySource(A.bytes), {
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
    const { httpRangeSource } = await import('../packages/pikelet-wasm/complete/sources.mjs');
    let payload = Buffer.from(A.bytes);
    let etag = '"v1"';
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
            res.writeHead(200, { 'accept-ranges': 'bytes', 'content-length': payload.length, etag });
            res.end(req.method === 'HEAD' ? undefined : payload);
            return;
        }
        // If-Range: a validator that no longer matches gets the whole file.
        if (req.headers['if-range'] && req.headers['if-range'] !== etag) {
            res.writeHead(200, { 'accept-ranges': 'bytes', 'content-length': payload.length, etag });
            res.end(payload);
            return;
        }
        const from = Number(range[1]);
        const to = Math.min(Number(range[2]), payload.length - 1);
        res.writeHead(206, { 'content-range': `bytes ${from}-${to}/${payload.length}`, 'content-length': to - from + 1, etag });
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
    // The asset is re-uploaded under the same name (same size, new bytes and
    // ETag) and the signed URL expires: re-resolution must not adopt the new
    // ETag — that would make the If-Range check compare the replaced file
    // against itself and silently serve its bytes.
    payload = Buffer.from(payload).map((b) => b ^ 0xff);
    etag = '"v2"';
    token = 'sig-3';
    await rejects('re-resolution onto a replaced artifact is reported as a changed artifact',
        () => src.read(0, 16), /artifact changed.*expected ETag "v1".*re-resolved "v2"/);
    check('the replaced artifact\'s bytes were never served', src.stats.fullFallback === false);
    host.close();
}

{
    // --- WASM sketch scanner (options.sketchScanner) ---
    // The engine's SIMD scan kernel must select the same candidates the JS
    // scan does — the exact rerank then scores both sets identically, so
    // any divergence shows up as differing [id, distance] pairs.
    const js = await openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode, sketchScanner: false });
    check('sketchScanner:false reports residentScan js', js.info().residentScan === 'js');
    const wasm = await openPikeletFile(memorySource(A.bytes), {
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
    const broken = await openPikeletFile(memorySource(A.bytes), {
        encodeQuery: hostEncode,
        sketchScanner: () => { throw new Error('no engine here'); },
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    const survived = await broken.query('rec 17', { k: 3 });
    check('failed scanner factory leaves the JS scan serving queries',
        broken.info().residentScan === 'js' && survived.results.some((r) => r.id === 17));
    await broken.close();

    await rejects('a non-scanner sketchScanner option is rejected',
        () => openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode, sketchScanner: 42 }),
        /sketchScanner must be false/);
}

{
    // --- automatic scanner staging (no sketchScanner option) ---
    // In Node the reader stages the engine scanner by itself once the sketch
    // holds SCANNER_AUTO_CELLS (16 Mi) cells. That path imports the Node
    // engine entrypoint by a relative path the injected-factory tests above
    // never exercise, and its failures are swallowed by design, so a broken
    // import shows up only as every large pack quietly using the JS scan.
    const small = await openPikeletFile(memorySource(A.bytes), { encodeQuery: hostEncode });
    await new Promise((resolve) => setTimeout(resolve, 100));
    check('a pack below the auto threshold stays on the JS scan', small.info().residentScan === 'js');
    await small.close();

    const BIG_DIM = 256;
    const BIG_COUNT = (16 * 1024 * 1024) / BIG_DIM; // exactly SCANNER_AUTO_CELLS cells
    const bigEncode = (text) => {
        const rand = mulberry32(fnv1a(text));
        const v = new Float32Array(BIG_DIM);
        let n = 0;
        for (let d = 0; d < BIG_DIM; d++) { const x = rand() * 2 - 1; v[d] = x; n += x * x; }
        n = 1 / Math.sqrt(n);
        for (let d = 0; d < BIG_DIM; d++) v[d] *= n;
        return v;
    };
    // Rows are unit vectors like the small pack's (random bytes would leave
    // most cosine distances clamped at 0 and both scans tie-breaking by id).
    const qdata = new Uint8Array(BIG_COUNT * BIG_DIM);
    const scales = new Float32Array(BIG_COUNT);
    const offsets = new Float32Array(BIG_COUNT);
    for (let i = 0; i < BIG_COUNT; i++) {
        const v = bigEncode(`r ${i}`);
        let mn = Infinity, mx = -Infinity;
        for (let d = 0; d < BIG_DIM; d++) { if (v[d] < mn) mn = v[d]; if (v[d] > mx) mx = v[d]; }
        const s = (mx - mn) / 255 || 1e-12;
        scales[i] = s; offsets[i] = mn;
        for (let d = 0; d < BIG_DIM; d++) {
            const b = Math.round((v[d] - mn) / s);
            qdata[i * BIG_DIM + d] = b < 0 ? 0 : b > 255 ? 255 : b;
        }
    }
    const bigSketchPath = path.join(tmp, 'big.pikelet-sketch');
    exportSketchArtifact({ dim: BIG_DIM, count: BIG_COUNT, metric: 1, qdata, scales, offsets }, bigSketchPath,
        { sketchDims: BIG_DIM, sketchBits: 8, recommendedRerank: 40 });
    const bigRecords = Array.from({ length: BIG_COUNT }, (_, i) => Buffer.from(`{"title":"r${i}","text":"r${i}"}`, 'utf8'));
    const bigCorpus = buildCorpusSegment(bigRecords, { pageRecords: 1024 });
    const bigDeclaration = {
        kind: 'external-transformers-v1', model: 'test/deterministic-hash-encoder', dim: BIG_DIM, pooling: 'none', normalized: true, maxTokens: 64,
        testVectors: ['r 1', 'r 4242'].map((text) => ({ text, embedding: Array.from(bigEncode(text), (v) => Number(v.toFixed(6))), tolerance: 1e-3 })),
    };
    const bigPath = path.join(tmp, 'big.pikelet');
    assemblePikeletFile({
        profile: PROFILE_V2,
        corpus: { ...bigCorpus.corpus, provenance: null },
        dim: BIG_DIM,
        metric: 'cosine',
        encoder: { kind: bigDeclaration.kind, model: bigDeclaration.model },
        recommendedRerank: 40,
    }, [
        { kind: 'index', bytes: fs.readFileSync(bigSketchPath) },
        { kind: 'corpus', bytes: bigCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(bigDeclaration), 'utf8'), CALIBRATION) },
    ], bigPath);
    const bigBytes = fs.readFileSync(bigPath);

    const auto = await openPikeletFile(memorySource(bigBytes), { encodeQuery: bigEncode });
    check('the big pack reports the auto threshold geometry', auto.info().records === BIG_COUNT, `records ${auto.info().records}`);
    for (let i = 0; i < 400 && auto.info().residentScan !== 'engine'; i++) {
        await new Promise((resolve) => setTimeout(resolve, 25));
    }
    check('a pack at the auto threshold stages the engine scanner by itself (residentScan engine)',
        auto.info().residentScan === 'engine', `residentScan ${auto.info().residentScan}`);
    const jsBig = await openPikeletFile(memorySource(bigBytes), { encodeQuery: bigEncode, sketchScanner: false });
    const viaAuto = await auto.query('r 4242', { k: 5 });
    const viaJs = await jsBig.query('r 4242', { k: 5 });
    check('auto-staged scanner finds the record behind the query', viaAuto.results[0]?.id === 4242, `top ${viaAuto.results[0]?.id}`);
    check('auto-staged scanner matches the JS scan',
        JSON.stringify(viaAuto.results.map((r) => [r.id, r.distance])) === JSON.stringify(viaJs.results.map((r) => [r.id, r.distance])),
        `engine ${JSON.stringify(viaAuto.results.map((r) => r.id))} vs js ${JSON.stringify(viaJs.results.map((r) => r.id))}`);
    await auto.close();
    await jsBig.close();
}

{
    // --- fuseCandidates (complete/fusion.mjs): the one fusion rule the
    // reader, the calibrator and the BEIR ladder share ---
    const pool = [
        { id: 10, distance: 0.20 }, { id: 11, distance: 0.30 }, { id: 12, distance: 0.31 },
        { id: 13, distance: 0.40 }, { id: 14, distance: 0.50 },
    ];
    const ids = (hits) => hits.map((h) => h.id);
    // Plain RRF (weight 1, no guard) for the arithmetic checks; the shipped
    // defaults are checked separately below.
    const plain = { lexicalWeight: 1, guardMargin: 0 };
    check('defaults are K=60, lexical weight 0.5, guard margin 0.05', FUSION_DEFAULTS.rrfK === 60 && FUSION_DEFAULTS.lexicalWeight === 0.5 && FUSION_DEFAULTS.guardMargin === 0.05);
    check('no lexical hits: vector order, as a copy', JSON.stringify(ids(fuseCandidates(pool, []))) === JSON.stringify([10, 11, 12, 13, 14]) && fuseCandidates(pool, []) !== pool);
    // 1/(60+3) + 1/60 > 1/60: under plain RRF a record in both lists outranks a vector-only rank 1.
    check('plain RRF: a record in both lists outranks a vector-only top hit', ids(fuseCandidates(pool, [13], plain))[0] === 13);
    check('plain RRF: lexical order breaks ties among lexical hits', JSON.stringify(ids(fuseCandidates(pool, [12, 11], plain)).slice(0, 2)) === JSON.stringify([12, 11]));
    check('RRF returns the same hit objects', fuseCandidates(pool, [13], plain)[0] === pool[3]);
    check('lexicalWeight 0 restores vector order', JSON.stringify(ids(fuseCandidates(pool, [13, 14], { lexicalWeight: 0 }))) === JSON.stringify([10, 11, 12, 13, 14]));
    check('a lexical id missing from the pool is ignored', ids(fuseCandidates(pool, [99, 13], plain))[0] === 13);
    check('duplicate lexical ids keep their first rank', JSON.stringify(ids(fuseCandidates(pool, [13, 13, 11], plain)).slice(0, 2)) === JSON.stringify([13, 11]));
    // Shipped defaults on the same pool: the guard fires (top-1 leads by 50%),
    // and with weight 0.5 the lexical hit still moves up to rank 2:
    // 1/63 + 0.5/60 = 0.0242 > 1/61 = 0.0164.
    check('defaults: guard keeps the vector top-1, the lexical hit takes rank 2', JSON.stringify(ids(fuseCandidates(pool, [13])).slice(0, 3)) === JSON.stringify([10, 13, 11]));
    // Close top-2 (0.30 vs 0.31 -> 3%): the guard does not fire, so with
    // weight 0.5 a vector rank-1 still loses to a record in both lists.
    const close = pool.slice(1);
    check('defaults: below the margin a both-list record still outranks a vector-only top hit', ids(fuseCandidates(close, [13]))[0] === 13);
    // Guard: top-1 leads top-2 by (0.30-0.20)/0.20 = 50%.
    check('guard keeps the vector top-1 first when its margin clears the threshold', ids(fuseCandidates(pool, [13], { lexicalWeight: 1, guardMargin: 0.05 }))[0] === 10);
    check('guard leaves the fused order intact below it', JSON.stringify(ids(fuseCandidates(pool, [13], { lexicalWeight: 1, guardMargin: 0.05 })).slice(1, 3)) === JSON.stringify([13, 11]));
    check('guard does not fire when the margin is below the threshold', ids(fuseCandidates(pool, [13], { lexicalWeight: 1, guardMargin: 0.60 }))[0] === 13);
    check('guard does not fire on a single-hit pool', ids(fuseCandidates([pool[0]], [10], { guardMargin: 0.05 }))[0] === 10);
    // distance === 0 is the strongest possible top-1, not a degenerate case:
    // cosine distance is 1 - dot, so a query matching passage text verbatim
    // scores exactly 0. Gating the guard on `first.distance > 0` disabled it
    // for precisely that hit, so a perfect match was buried by a term match
    // while a worse one (0.001) was protected.
    const exact = [{ id: 10, distance: 0 }, { id: 11, distance: 0.9 }, { id: 12, distance: 0.95 }, { id: 13, distance: 0.97 }];
    check('guard protects a distance-0 top-1', ids(fuseCandidates(exact, [13]))[0] === 10);
    check('guard protects a near-zero top-1 the same way', ids(fuseCandidates([{ id: 10, distance: 0.001 }, ...exact.slice(1)], [13]))[0] === 10);
    // Two hits both at distance 0 are a genuine tie, not a lead: nothing to protect.
    const tied = [{ id: 10, distance: 0 }, { id: 11, distance: 0 }, { id: 12, distance: 0.9 }, { id: 13, distance: 0.95 }];
    check('guard does not fire when the top two are tied at 0', ids(fuseCandidates(tied, [13]))[0] === 13);
    let threw = false;
    try { fuseCandidates(pool, [13], { rrfK: 0 }); } catch { threw = true; }
    check('invalid fusion options are rejected', threw);
}

// ---------------------------------------------------------------------------
// maxTokens bounds in the inline-encoder declaration
// ---------------------------------------------------------------------------
console.log('inline encoder: a pack cannot declare an out-of-range maxTokens');
{
    const { parseInlineTransformerEncoder, KERNEL_LAYOUT, expectedBlobBytes } =
        await import('../packages/pikelet-wasm/complete/inline-transformer.mjs');

    // maxTokens sizes the kernel's ids/hidden heap buffers and drives
    // windowLen = maxSeq - 2. Before this was bounded, `Math.min(x, 512)`
    // clamped only the top: a pack declaring 1 made windowLen -1, so
    // interior.slice(0, -1) returned nearly the whole token sequence and the
    // forward pass wrote ~92 KB into a 1536-byte allocation. It fired at
    // mount, before any query, because the declaration's own testVectors are
    // embedded and verified after the buffers are allocated.
    const vocabText = '[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n';
    const blob = new Uint8Array(expectedBlobBytes(KERNEL_LAYOUT));
    const pack = (maxTokens) => {
        const decl = { dim: KERNEL_LAYOUT.D, layout: KERNEL_LAYOUT, pooling: 'mean' };
        if (maxTokens !== 'omit') decl.maxTokens = maxTokens;
        const declBytes = new TextEncoder().encode(JSON.stringify(decl));
        const vocabBytes = new TextEncoder().encode(vocabText);
        const out = new Uint8Array(12 + declBytes.length + vocabBytes.length + blob.length);
        const view = new DataView(out.buffer);
        view.setUint32(0, declBytes.length, true);
        view.setUint32(4, vocabBytes.length, true);
        view.setUint32(8, blob.length, true);
        out.set(declBytes, 12);
        out.set(vocabBytes, 12 + declBytes.length);
        out.set(blob, 12 + declBytes.length + vocabBytes.length);
        return out;
    };

    // The overflow range, and the shapes that dodge an integer check.
    for (const bad of [1, 2, 7, 0.5, -1, '512', 513, 1e9]) {
        await rejects(`maxTokens ${JSON.stringify(bad)} is refused at parse`,
            async () => parseInlineTransformerEncoder(pack(bad)), /maxTokens/);
    }
    // Values the toolchain actually emits, plus the absent/null fallbacks.
    for (const good of [8, 16, 512, 'omit', null]) {
        let ok = true;
        let detail = '';
        try {
            const parsed = parseInlineTransformerEncoder(pack(good === 'omit' ? 'omit' : good));
            ok = parsed.declaration.dim === KERNEL_LAYOUT.D;
        } catch (err) { ok = false; detail = String(err && err.message).slice(0, 120); }
        check(`maxTokens ${good === 'omit' ? '(absent)' : JSON.stringify(good)} still parses`, ok, detail);
    }
}

// ---------------------------------------------------------------------------
// Layer count in the inline-encoder declaration
// ---------------------------------------------------------------------------
console.log('inline encoder: the declared layer count is bounded by the kernel');
{
    const { parseInlineTransformerEncoder, createInlineTransformerEmbedder, KERNEL_LAYOUT, KERNEL_MAX_LAYERS,
        expectedBlobBytes } = await import('../packages/pikelet-wasm/complete/inline-transformer.mjs');
    const { default: createEncoderModule } =
        await import('../packages/pikelet-wasm/complete/encoder-kernels/encoder.node.mjs');

    // The kernel takes the layer count per call and holds MAXL layers; the
    // host is what keeps a declaration inside that and the blob consistent
    // with it.
    const vocabText = '[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n';
    const pack = (L, blobLayers = L) => {
        const blob = new Uint8Array(expectedBlobBytes({ ...KERNEL_LAYOUT, L: blobLayers }));
        const decl = { dim: KERNEL_LAYOUT.D, layout: { ...KERNEL_LAYOUT, L }, pooling: 'mean' };
        const declBytes = new TextEncoder().encode(JSON.stringify(decl));
        const vocabBytes = new TextEncoder().encode(vocabText);
        const out = new Uint8Array(12 + declBytes.length + vocabBytes.length + blob.length);
        const view = new DataView(out.buffer);
        view.setUint32(0, declBytes.length, true);
        view.setUint32(4, vocabBytes.length, true);
        view.setUint32(8, blob.length, true);
        out.set(declBytes, 12);
        out.set(vocabBytes, 12 + declBytes.length);
        out.set(blob, 12 + declBytes.length + vocabBytes.length);
        return out;
    };

    check('the kernel runs up to 12 layers', KERNEL_MAX_LAYERS === 12);
    for (const good of [1, 6, 12]) {
        let ok = true;
        let detail = '';
        try {
            ok = parseInlineTransformerEncoder(pack(good)).declaration.layout.L === good;
        } catch (err) { ok = false; detail = String(err && err.message).slice(0, 120); }
        check(`layout L=${good} parses`, ok, detail);
    }
    // Shapes that dodge an integer check, and counts past the kernel's
    // Layout struct. The blob is sized for 1 layer so that an unbounded L
    // never sizes an allocation here.
    for (const bad of [0, 13, -1, 6.5, '6', null, 1e9]) {
        await rejects(`layout L=${JSON.stringify(bad)} is refused at parse`,
            async () => parseInlineTransformerEncoder(pack(bad, 1)), /layout L=/);
    }
    await rejects('a 6-layer blob declared as 12 layers is refused at parse',
        async () => parseInlineTransformerEncoder(pack(12, 6)), /blob is \d+ bytes/);

    // createInlineTransformerEmbedder() is also called without the parser
    // (the BEIR ladder), so it checks the blob against the layer count too.
    const sixLayerBlob = new Uint8Array(expectedBlobBytes(KERNEL_LAYOUT));
    await rejects('the embedder refuses a blob that does not match the declared layers',
        () => createInlineTransformerEmbedder({
            declaration: { dim: KERNEL_LAYOUT.D, layout: { ...KERNEL_LAYOUT, L: 12 } },
            vocabText, blob: sixLayerBlob, createEncoder: createEncoderModule, verify: false,
        }), /12 layers of u8 weights imply/);
    await rejects('the embedder refuses an out-of-range layer count',
        () => createInlineTransformerEmbedder({
            declaration: { dim: KERNEL_LAYOUT.D, layout: { ...KERNEL_LAYOUT, L: 13 } },
            vocabText, blob: sixLayerBlob, createEncoder: createEncoderModule, verify: false,
        }), /layout L=13/);

    // The kernel's own guards, for any caller that skips the host checks.
    // Blob pointer 0: both checks run before any weight is read.
    const EM = await createEncoderModule();
    const idsPtr = EM._malloc(4 * 4);
    new Int32Array(EM.HEAP32.buffer, idsPtr, 4).set([2, 4, 4, 3]);
    const outPtr = EM._malloc(4 * KERNEL_LAYOUT.D * 4);
    for (const layers of [0, -1, 13]) {
        check(`the kernel returns -3 for ${layers} layers`, EM._encoder_forward(0, idsPtr, 4, outPtr, 0, layers, 8) === -3);
    }
    for (const bits of [0, 2, 16]) {
        check(`the kernel returns -4 for ${bits}-bit weights`, EM._encoder_forward(0, idsPtr, 4, outPtr, 0, 6, bits) === -4);
    }
}

// ---------------------------------------------------------------------------
// Weight format in the inline-encoder declaration
// ---------------------------------------------------------------------------
console.log('inline encoder: the declared weight format is one the kernel runs');
{
    const { parseInlineTransformerEncoder, createInlineTransformerEmbedder, KERNEL_LAYOUT, expectedBlobBytes } =
        await import('../packages/pikelet-wasm/complete/inline-transformer.mjs');
    const { default: createEncoderModule } =
        await import('../packages/pikelet-wasm/complete/encoder-kernels/encoder.node.mjs');

    const vocabText = '[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\n';
    const pack = (layoutFields, blobLayout = { ...KERNEL_LAYOUT, ...layoutFields }) => {
        const blob = new Uint8Array(expectedBlobBytes(blobLayout));
        const layout = { ...KERNEL_LAYOUT, ...layoutFields };
        for (const key of Object.keys(layout)) if (layout[key] === undefined) delete layout[key];
        const decl = { dim: KERNEL_LAYOUT.D, layout, pooling: 'mean' };
        const declBytes = new TextEncoder().encode(JSON.stringify(decl));
        const vocabBytes = new TextEncoder().encode(vocabText);
        const out = new Uint8Array(12 + declBytes.length + vocabBytes.length + blob.length);
        const view = new DataView(out.buffer);
        view.setUint32(0, declBytes.length, true);
        view.setUint32(4, vocabBytes.length, true);
        view.setUint32(8, blob.length, true);
        out.set(declBytes, 12);
        out.set(vocabBytes, 12 + declBytes.length);
        out.set(blob, 12 + declBytes.length + vocabBytes.length);
        return out;
    };

    // u4 halves the weight bytes; the per-block scales and offsets double
    // with the halved block size.
    const u8Bytes = expectedBlobBytes(KERNEL_LAYOUT);
    const u4Bytes = expectedBlobBytes({ ...KERNEL_LAYOUT, Q: 4, B: 32 });
    check('a u4 blob is weights/2 plus twice the block parameters of u8',
        u8Bytes === 25474368 && u4Bytes === 17023872, `${u8Bytes} ${u4Bytes}`);
    check('Q absent and Q=8 size the blob identically',
        expectedBlobBytes({ ...KERNEL_LAYOUT, Q: 8 }) === u8Bytes);

    for (const [label, fields] of [['Q absent, B=64', {}], ['Q=8, B=64', { Q: 8 }], ['Q=4, B=32', { Q: 4, B: 32 }],
        ['Q=4, B=32, L=12', { Q: 4, B: 32, L: 12 }]]) {
        let ok = true;
        let detail = '';
        try { parseInlineTransformerEncoder(pack(fields)); } catch (err) { ok = false; detail = String(err && err.message).slice(0, 120); }
        check(`layout ${label} parses`, ok, detail);
    }
    // Formats the kernel has no path for, a block size that does not match
    // the format, and shapes that dodge a numeric check. The blob is sized
    // as u8 so a bad Q never sizes an allocation here.
    for (const [label, fields] of [['Q=4, B=64', { Q: 4 }], ['Q=8, B=32', { Q: 8, B: 32 }], ['Q absent, B=32', { B: 32 }],
        ['Q=2, B=32', { Q: 2, B: 32 }], ['Q=16, B=64', { Q: 16 }], ['Q="4", B=32', { Q: '4', B: 32 }],
        ['Q=null, B=64', { Q: null }], ['Q=4.5, B=32', { Q: 4.5, B: 32 }]]) {
        await rejects(`layout ${label} is refused at parse`,
            async () => parseInlineTransformerEncoder(pack(fields, KERNEL_LAYOUT)), /layout Q=/);
    }
    await rejects('a u8 blob declared as u4 is refused at parse',
        async () => parseInlineTransformerEncoder(pack({ Q: 4, B: 32 }, KERNEL_LAYOUT)), /blob is \d+ bytes/);

    const u8Blob = new Uint8Array(u8Bytes);
    await rejects('the embedder refuses a u8 blob declared as u4',
        () => createInlineTransformerEmbedder({
            declaration: { dim: KERNEL_LAYOUT.D, layout: { ...KERNEL_LAYOUT, Q: 4, B: 32 } },
            vocabText, blob: u8Blob, createEncoder: createEncoderModule, verify: false,
        }), /6 layers of u4 weights imply/);
    await rejects('the embedder refuses a format the kernel has no path for',
        () => createInlineTransformerEmbedder({
            declaration: { dim: KERNEL_LAYOUT.D, layout: { ...KERNEL_LAYOUT, Q: 2, B: 32 } },
            vocabText, blob: u8Blob, createEncoder: createEncoderModule, verify: false,
        }), /layout Q=2/);
}

// ---------------------------------------------------------------------------
// Producer hints a pack can use to inflate per-query work
// ---------------------------------------------------------------------------

{
    // recommendedRerank is a header hint. Up to the row count it would make
    // every query fetch, hash and rerank the whole vectors region, so the
    // reader caps the value it defaults to; an explicit rerank still wins.
    const { PikeletSketchArtifact } = require('pikelet-wasm/artifact');
    const qdata = new Uint8Array(COUNT * DIM);
    const scales = new Float32Array(COUNT).fill(1 / 255);
    const offsets = new Float32Array(COUNT);
    for (let i = 0; i < qdata.length; i++) qdata[i] = (i * 37) & 0xff;
    const bigRerankPath = path.join(tmp, 'big-rerank.pikelet-sketch');
    exportSketchArtifact({ dim: DIM, count: COUNT, metric: 1, qdata, scales, offsets }, bigRerankPath,
        { sketchDims: DIM, sketchBits: 8, recommendedRerank: 1_000_000 });
    const sk = await PikeletSketchArtifact.open(memorySource(fs.readFileSync(bigRerankPath)));
    check('a header recommendedRerank of 1e6 is kept as declared but capped as the default',
        sk.declaredRecommendedRerank === 1_000_000 && sk.recommendedRerank === 4096,
        `declared ${sk.declaredRecommendedRerank}, effective ${sk.recommendedRerank}`);
    const honest = await PikeletSketchArtifact.open(memorySource(SKETCH));
    check('an honest recommendedRerank passes through unchanged', honest.recommendedRerank === 40);

    // compile's sweep used to fall back to the full row count when no rung
    // reached its recall target: a depth the reader above no longer honors
    // by default. It now stops at the same ceiling and says so.
    const artifactModule = require('pikelet-wasm/artifact');
    const { measureRecommendedRerank } = await import('../packages/pikelet-wasm/complete/builder.mjs');
    const sweepSketch = (count, tag) => {
        const q = new Uint8Array(count * DIM);
        for (let i = 0; i < q.length; i++) q[i] = (i * 2654435761) >>> 24;
        const at = path.join(tmp, `sweep-${tag}.pikelet-sketch`);
        exportSketchArtifact({ dim: DIM, count, metric: 1, qdata: q, scales: new Float32Array(count).fill(1 / 255), offsets: new Float32Array(count) },
            at, { sketchDims: DIM, sketchBits: 8 });
        const vectors = [];
        for (let r = 0; r < 32; r++) vectors.push(Float32Array.from(q.subarray(r * DIM, (r + 1) * DIM), (v) => v / 255));
        return { bytes: fs.readFileSync(at), vectors };
    };
    const big = sweepSketch(5000, 'big');
    const capped = await measureRecommendedRerank({
        artifactModule, sketchBytes: big.bytes, queryVectors: big.vectors, targetRecall: 1.01, sweep: [10, 100],
    });
    check('the rerank sweep stops at the reader ceiling, not the row count, when the target is not met',
        capped.recommendedRerank === artifactModule.MAX_RECOMMENDED_RERANK && capped.rerankCeiling === 4096
        && capped.targetReached === false && capped.curve[capped.curve.length - 1].rerank === 4096,
        JSON.stringify({ C: capped.recommendedRerank, ceiling: capped.rerankCeiling, reached: capped.targetReached, last: capped.curve.at(-1) }));
    const small = sweepSketch(300, 'small');
    const met = await measureRecommendedRerank({ artifactModule, sketchBytes: small.bytes, queryVectors: small.vectors });
    check('a sketch under the ceiling sweeps to its row count as before and reports the target met',
        met.rerankCeiling === 300 && met.targetReached === true && met.recommendedRerank <= 300,
        JSON.stringify({ C: met.recommendedRerank, ceiling: met.rerankCeiling, reached: met.targetReached }));
}

{
    // A retrieval-signals asset that cannot be scored as calibrated must
    // degrade to unscored (a null scorer), never answer everything.
    const { createAbstentionScorer, MAX_COVERAGE_TOP_K } = await import('../packages/pikelet-wasm/complete/retrieval-abstention.mjs');
    const base = {
        features: ['d0'], weights: [-1], bias: 0,
        standardize: { mean: { d0: 0.5 }, std: { d0: 0.1 } },
        thresholds: { hard: 0.2, weak: 0.5 },
        vocabBloom: { bits: 64 },
    };
    const bloom = new Uint8Array(8);
    check('a well-formed asset yields a scorer', createAbstentionScorer(base, bloom) !== null);
    const badThresholds = [
        ['missing', {}],
        ['non-numeric', { hard: '0.2', weak: '0.5' }],
        ['NaN', { hard: NaN, weak: 0.5 }],
        ['out of order', { hard: 0.6, weak: 0.5 }],
        ['outside [0,1]', { hard: -0.1, weak: 1.5 }],
    ];
    for (const [label, thresholds] of badThresholds) {
        check(`thresholds ${label} degrade to unscored`, createAbstentionScorer({ ...base, thresholds }, bloom) === null);
    }
    const coverage = { weight: 1, mean: 0.5, std: 0.2 };
    check('coverage.topK 5 (what calibrate ships) is accepted',
        createAbstentionScorer({ ...base, coverage: { ...coverage, topK: 5 } }, bloom)?.passagesNeeded === 5);
    for (const topK of [0, 1.5, MAX_COVERAGE_TOP_K + 1, 1e6, '5']) {
        check(`coverage.topK ${JSON.stringify(topK)} degrades to unscored`,
            createAbstentionScorer({ ...base, coverage: { ...coverage, topK } }, bloom) === null);
    }
}

// ---------------------------------------------------------------------------
// Rerank coalescing gap: the manifest hint is a floor, the source may widen it
// ---------------------------------------------------------------------------
console.log('rerank gap: a manifest recommendedGap is a floor under the source\'s own gap');
{
    const { httpRangeSource } = await import('../packages/pikelet-wasm/complete/sources.mjs');
    check('httpRangeSource coalesces rows within 256 KiB by default',
        httpRangeSource('https://example.invalid/x.pikelet').preferredGapBytes === 256 * 1024);
    check('an explicit preferredGapBytes still wins',
        httpRangeSource('https://example.invalid/x.pikelet', { preferredGapBytes: 4096 }).preferredGapBytes === 4096);

    // Reads one query costs, on a pack whose manifest hints `hint` (or none),
    // over a source that prefers `prefer`. A fresh open per measurement, so
    // no row is cached.
    const readsFor = async ({ hint, prefer, gap }) => {
        const pack = buildSynthetic({
            name: `gap-${hint}-${prefer}-${gap}`,
            manifestPatch: hint === undefined ? null : (m) => { m.recommendedGap = hint; },
        });
        const src = { ...memorySource(pack.bytes), preferredGapBytes: prefer };
        const search = await openPikeletFile(src, { encodeQuery: hostEncode });
        const before = src.reads;
        await search.query('rec 150', { k: 5, rerank: 40, ...(gap === undefined ? {} : { gap }) });
        const reads = src.reads - before;
        await search.close();
        return reads;
    };
    const tight = await readsFor({ hint: 0, prefer: 0 });
    const wide = await readsFor({ hint: 1 << 20, prefer: 1 << 20 });
    check('a wide gap costs fewer reads than none on this pack (the probe can tell them apart)', wide < tight, `${wide} vs ${tight}`);
    check('a source preferring a wider gap than the hint keeps its own',
        await readsFor({ hint: 0, prefer: 1 << 20 }) === wide);
    check('a hint wider than the source preference still applies',
        await readsFor({ hint: 1 << 20, prefer: 0 }) === wide);
    check('without a hint the source preference applies', await readsFor({ hint: undefined, prefer: 1 << 20 }) === wide);
    check('an explicit query gap overrides both', await readsFor({ hint: 1 << 20, prefer: 1 << 20, gap: 0 }) === tight);
}

// ---------------------------------------------------------------------------
// Row map (kind 8): several index rows per record
// ---------------------------------------------------------------------------
console.log('row map: index rows group into records');
{
    const { buildRowMapSegment } = await import('../packages/pikelet-wasm/complete/builder.mjs');
    const { parseRowMapSegment } = await import('../packages/pikelet-wasm/complete/rowmap.mjs');
    // The synthetic sketch's COUNT rows become COUNT/2 records of two rows
    // each: record r owns rows 2r and 2r+1, so row 17 belongs to record 8.
    const HALF = COUNT / 2;
    const halfCorpus = buildCorpusSegment(RECORDS.slice(0, HALF), { pageRecords: PAGE_RECORDS });
    const pairs = Array.from({ length: HALF + 1 }, (_, r) => 2 * r);
    const policy = { kind: 'sentence-windows-v1', sentences: 2, rows: COUNT };
    const windowed = (name, { rowMap = buildRowMapSegment(pairs), declare = policy } = {}) => {
        const out = path.join(tmp, `${name}.pikelet`);
        assemblePikeletFile({
            profile: PROFILE_V2, corpus: { ...halfCorpus.corpus, provenance: null }, dim: DIM, metric: 'cosine',
            encoder: { kind: 'external-transformers-v1' }, recommendedRerank: 40, sampleQueries: [],
            ...(declare ? { index: { rowMap: declare } } : {}),
        }, [
            { kind: 'index', bytes: SKETCH },
            { kind: 'corpus', bytes: halfCorpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify(kind2Declaration()), 'utf8'), CALIBRATION) },
            { kind: 'evaluation', bytes: EVALUATION },
            ...(rowMap ? [{ kind: 'rowmap', bytes: Buffer.from(rowMap) }] : []),
        ], out);
        return out;
    };
    const pack = await openPikeletFile(windowed('rowmap'), { encodeQuery: hostEncode });
    const info = pack.info();
    check('info reports records and index rows separately', info.records === HALF && info.indexRows === COUNT, `${info.records}/${info.indexRows}`);
    check('info reports the row policy', info.rowMap?.kind === 'sentence-windows-v1' && info.rowMap?.sentences === 2);
    const out = await pack.query('rec 17', { k: 10, showAbstained: true });
    check('a row hit returns the record that owns it', out.results[0]?.id === 8, String(out.results[0]?.id));
    check('each record appears once, however many of its rows matched',
        new Set(out.results.map((r) => r.id)).size === out.results.length && out.results.every((r) => r.id < HALF));
    const ev = await pack.__calibrationEvidence('rec 17', { excludeIds: new Set([8]) });
    check('excluding a record masks every row it owns', ev.vector.every((h) => h.id !== 8) && ev.fused.every((h) => h.id !== 8));
    await pack.close();

    await rejects('a declared row map with no row map segment is refused',
        () => openPikeletFile(windowed('rowmap-undeclared-segment', { rowMap: null }), { encodeQuery: hostEncode }), /declares index\.rowMap but carries no row map/);
    await rejects('a row map segment the manifest does not declare is refused',
        () => openPikeletFile(windowed('rowmap-undeclared-policy', { declare: null }), { encodeQuery: hostEncode }), /does not declare \(index\.rowMap\)/);
    // Valid on its own (the last record owns one row) but spans 299 of the index's 300 rows.
    const short = pairs.slice(0, -1).concat([COUNT - 1]);
    await rejects('a row map that does not span the index is refused',
        () => openPikeletFile(windowed('rowmap-short', { rowMap: buildRowMapSegment(short) }), { encodeQuery: hostEncode }), /row map covers \d+ rows, index holds/);
    check('a record with no rows cannot be built', (() => { try { buildRowMapSegment([0, 2, 2, 4]); return false; } catch { return true; } })());
    const good = buildRowMapSegment(pairs);
    const zeroRow = new Uint8Array(good); new DataView(zeroRow.buffer).setUint32(16 + 4 * 2, 2, true);
    check('a parsed row map rejects a record with no rows', (() => { try { parseRowMapSegment(zeroRow, HALF, COUNT); return false; } catch { return true; } })());
}

// ---------------------------------------------------------------------------
// Compile-time calibration: matchQuality is retrieval relevance (v6)
// ---------------------------------------------------------------------------
console.log('calibration: a compiled pack scores relevance, fit against off-topic queries');
{
    // The calibrator fits verified on-topic probes against off-topic ones
    // (foreign bank + gibberish); in-domain-unanswerable hard negatives are
    // only a reported diagnostic. So an off-topic query is "none", while a
    // question about a covered topic stays visible even when the corpus
    // lacks its specific fact — telling those apart is the reader's job.
    const { execFileSync } = await import('node:child_process');
    const docs = path.join(tmp, 'calibration-docs');
    fs.mkdirSync(docs);
    const topics = {
        tides: ['Tides', 'The moon pulls on the ocean, raising two bulges of water, so most coasts see two high tides and two low tides each day.', 'Spring tides happen near new and full moon, when the sun and moon line up; neap tides happen at the quarter moons.'],
        volcanoes: ['Volcanoes', 'Magma rises through cracks in the crust, collects in a chamber, and erupts as lava, ash and gas when pressure builds.', 'Shield volcanoes have gentle slopes from runny basalt lava, while stratovolcanoes are steep and explosive.'],
        bread: ['Bread', 'Yeast ferments the sugars in flour, releasing carbon dioxide that makes the dough rise before it is baked.', 'Kneading develops gluten, the elastic protein network that traps the gas and gives bread its chewy crumb.'],
        bees: ['Honeybees', 'A honeybee colony has one queen that lays eggs, thousands of female workers, and male drones that mate with queens.', 'Workers perform a waggle dance to tell nestmates the direction and distance of a good patch of flowers.'],
        glaciers: ['Glaciers', 'Glaciers form where more snow falls each winter than melts each summer, compacting over decades into dense ice.', 'Moving ice carves U-shaped valleys and leaves moraines, ridges of rock and gravel dropped at its edges.'],
        chess: ['Chess openings', 'The Italian Game begins with king pawn moves and develops the bishop toward the vulnerable f7 square.', 'The Sicilian Defence answers the king pawn with the c-pawn, creating an unbalanced fight for the centre.'],
        coffee: ['Coffee roasting', 'Roasting turns green coffee beans brown through the Maillard reaction and caramelization of their sugars.', 'Light roasts keep more acidity and origin flavor, while dark roasts taste more bitter and smoky.'],
        eclipses: ['Eclipses', 'A solar eclipse happens when the moon passes between the sun and the earth and casts its shadow on the surface.', 'A lunar eclipse happens when the earth passes between the sun and the moon, often turning the moon a coppery red.'],
    };
    for (const [file, [title, a, b]] of Object.entries(topics)) {
        fs.writeFileSync(path.join(docs, `${file}.md`), `# ${title}\n\n${a}\n\n${b}\n\n${a} ${b}\n`);
    }
    const out = path.join(tmp, 'calibrated.pikelet');
    execFileSync(process.execPath, [path.join(ROOT, 'packages', 'pikelet', 'bin', 'pikelet.mjs'),
        'compile', '--source', docs, '--out', out], { stdio: ['ignore', 'pipe', 'pipe'] });
    const pack = await openPikeletFile(out);
    const cal = (await pack.evaluation()).calibration;
    check('the evaluation segment records the relevance target',
        cal?.method === 'self-templates-v6' && cal?.target === 'relevance', JSON.stringify(cal && { method: cal.method, target: cal.target }));
    check('the cross-validated AUC against off-topic clears the gate', cal?.cvAuc >= 0.85, String(cal?.cvAuc));
    check('the in-domain-unanswerable diagnostic is reported',
        cal?.cvAucHard === null || (cal?.cvAucHard >= 0 && cal?.cvAucHard <= 1), String(cal?.cvAucHard));
    const verdict = async (q) => (await pack.query(q, { k: 5, showAbstained: true })).matchQuality;
    for (const q of ['why are there two high tides a day', 'what makes bread dough rise',
        'how do bees tell each other where flowers are']) {
        const v = await verdict(q);
        check(`an on-topic question is not "none": ${q}`, v === 'strong' || v === 'weak', v);
    }
    // Covered topic, absent fact: relevance keeps these visible.
    for (const q of ['how many eggs does a honeybee queen lay per day', 'what temperature should bread be baked at']) {
        const v = await verdict(q);
        check(`a covered topic with an absent fact stays visible: ${q}`, v === 'strong' || v === 'weak', v);
    }
    for (const q of ['how do I renew my passport', 'best mortgage refinance rates', 'nginx 502 bad gateway fix']) {
        const v = await verdict(q);
        check(`an off-topic question is "none": ${q}`, v === 'none', v);
    }
    await pack.close();

    // Calibration retrieves through __calibrationEvidence, the same code
    // query() serves from, so the evidence a fit sees is the evidence a
    // query is scored on. Checked against query() itself, over the kinds of
    // text calibration probes with: title questions, content words, and
    // off-topic queries.
    console.log('calibration evidence: the fit sees what query() scores');
    const parity = await openPikeletFile(out);
    const probes = ['what is tides', 'tell me about volcanoes', 'yeast ferments sugars flour',
        'waggle dance direction distance', 'glaciers carve valleys', 'how do I renew my passport',
        'nginx 502 bad gateway fix', 'qzvxk plorbit wendash'];
    const ids = (list) => list.map((h) => h.id).join(',');
    let vectorSame = 0, fusedSame = 0, distancesSame = 0, kStable = 0;
    for (const q of probes) {
        const ev = await parity.__calibrationEvidence(q);
        const served = await parity.query(q, { k: 10, showAbstained: true });
        const augmented = await parity.query(q, { k: 10, showAbstained: true, retrieval: 'augmented' });
        if (ids(ev.vector.slice(0, 10)) === ids(augmented.results)) vectorSame++;
        if (ev.vector.slice(0, 10).every((h, i) => h.distance === augmented.results[i]?.distance)) distancesSame++;
        if (ids(ev.fused.slice(0, 10)) === ids(served.results)) fusedSame++;
        // A caller's k up to the rerank depth changes how much is returned,
        // never the evidence a verdict is scored on.
        const atK = await Promise.all([1, 3, 5].map((k) => parity.query(q, { k, showAbstained: true })));
        if (atK.every((r) => r.matchQuality === served.matchQuality && r.confidence === served.confidence
            && JSON.stringify(r.grounding) === JSON.stringify(served.grounding)
            && ids(r.results) === ids(served.results.slice(0, r.results.length)))) kStable++;
    }
    check('calibration evidence has the vector top 10 query() ranks', vectorSame === probes.length, `${vectorSame}/${probes.length}`);
    check('calibration evidence has the exact distances query() reads', distancesSame === probes.length, `${distancesSame}/${probes.length}`);
    check('calibration evidence has the fused top 10 query() serves', fusedSame === probes.length, `${fusedSame}/${probes.length}`);
    check('verdict, confidence and ranking agree across k = 1, 3, 5, 10', kStable === probes.length, `${kStable}/${probes.length}`);
    // Ablation: excluded records leave both candidate sources before they
    // are cut to size, so they never come back through BM25, and the
    // window refills from what remains instead of shrinking.
    const ablate = 'the moon pulls on the ocean raising two bulges of water';
    const full = await parity.__calibrationEvidence(ablate);
    const excludeIds = new Set(full.vector.slice(0, 5).map((h) => h.id));
    const ablated = await parity.__calibrationEvidence(ablate, { excludeIds });
    check('an excluded record never reappears in the vector window', ablated.vector.every((h) => !excludeIds.has(h.id)));
    check('an excluded record never reappears through BM25 into the fused window', ablated.fused.every((h) => !excludeIds.has(h.id)));
    check('excluding the whole unfiltered top 5 still yields a full window',
        ablated.vector.length >= Math.min(10, full.vector.length - excludeIds.size), `${ablated.vector.length} of ${full.vector.length}`);
    await parity.close();

    // Reproducible builds: the evaluation segment's generatedAt is the only
    // wall-clock input, and SOURCE_DATE_EPOCH pins it, so two builds of the
    // same inputs are the same bytes and the same identity.
    console.log('reproducible builds: SOURCE_DATE_EPOCH pins the build time');
    const compileAt = (outPath, epoch) => execFileSync(process.execPath, [path.join(ROOT, 'packages', 'pikelet', 'bin', 'pikelet.mjs'),
        'compile', '--source', docs, '--out', outPath], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, SOURCE_DATE_EPOCH: epoch } });
    const first = path.join(tmp, 'reproducible-a.pikelet');
    const second = path.join(tmp, 'reproducible-b.pikelet');
    compileAt(first, '1700000000');
    compileAt(second, '1700000000');
    check('two builds with the same SOURCE_DATE_EPOCH are byte-identical',
        Buffer.compare(fs.readFileSync(first), fs.readFileSync(second)) === 0);
    const stamped = await openPikeletFile(first);
    check('generatedAt is the pinned time', (await stamped.evaluation()).generatedAt === '2023-11-14T22:13:20.000Z',
        String((await stamped.evaluation()).generatedAt));
    await stamped.close();
    let refused = null;
    try { compileAt(path.join(tmp, 'reproducible-bad.pikelet'), 'yesterday'); } catch (error) { refused = String(error.stderr); }
    check('a malformed SOURCE_DATE_EPOCH is refused', /SOURCE_DATE_EPOCH must be a whole number/.test(refused || ''), refused);

    // Compaction refits calibration (LAYERED_PROFILE 6.3) through the
    // compacted pack's own retrieval, so the refit runs end to end here
    // rather than only its --no-refit fallback.
    console.log('compaction: the refit fits through the compacted pack and scores');
    const more = path.join(tmp, 'calibration-more');
    fs.mkdirSync(more);
    fs.writeFileSync(path.join(more, 'comets.md'), '# Comets\n\nA comet is a ball of ice and dust that grows a glowing coma and a long tail as it nears the sun.\n\n'
        + 'The solar wind pushes the ion tail straight away from the sun, while the dust tail curves along the orbit.\n');
    const cli = (args) => execFileSync(process.execPath, [path.join(ROOT, 'packages', 'pikelet', 'bin', 'pikelet.mjs'), ...args],
        { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    const layer = path.join(tmp, 'calibrated.0001.pikelet');
    cli(['append', '--parent', out, '--source', more, '--out', layer]);
    const compacted = path.join(tmp, 'compacted.pikelet');
    const compactLog = cli(['compact', '--head', out, '--head', layer, '--out', compacted]);
    check('compaction refits rather than shipping unscored', /Calibration: refit over the compacted corpus/.test(compactLog), compactLog.slice(-300));
    check('the refit leaves no uncalibrated scratch file behind', !fs.existsSync(`${compacted}.uncalibrated`));
    const folded = await openPikeletFile(compacted);
    const foldedVerdict = async (q) => (await folded.query(q, { k: 5, showAbstained: true })).matchQuality;
    for (const q of ['why does a comet grow a tail', 'what makes bread dough rise']) {
        const v = await foldedVerdict(q);
        check(`the compacted pack scores an on-topic question: ${q}`, v === 'strong' || v === 'weak', v);
    }
    const offTopic = await foldedVerdict('best mortgage refinance rates');
    check('the compacted pack scores an off-topic question "none"', offTopic === 'none', offTopic);
    await folded.close();

    // Sentence windows: compile indexes each chunk plus its 2-sentence
    // windows, maps the rows back to chunks, and calibrates through the same
    // windowed retrieval it serves.
    console.log('sentence windows: compile, serve and calibrate over windowed rows');
    const windowedOut = path.join(tmp, 'windowed.pikelet');
    cli(['compile', '--source', docs, '--out', windowedOut, '--sentence-windows', '2']);
    const wpack = await openPikeletFile(windowedOut);
    const winfo = wpack.info();
    check('a windowed pack indexes more rows than records', winfo.indexRows > winfo.records && winfo.rowMap?.sentences === 2,
        `${winfo.indexRows} rows / ${winfo.records} records`);
    let wSame = 0;
    for (const q of probes) {
        const ev = await wpack.__calibrationEvidence(q);
        const served = await wpack.query(q, { k: 10, showAbstained: true });
        const augmented = await wpack.query(q, { k: 10, showAbstained: true, retrieval: 'augmented' });
        if (ids(ev.fused.slice(0, 10)) === ids(served.results) && ids(ev.vector.slice(0, 10)) === ids(augmented.results)
            && new Set(served.results.map((r) => r.id)).size === served.results.length) wSame++;
    }
    check('windowed calibration evidence matches what query() serves, one result per record', wSame === probes.length, `${wSame}/${probes.length}`);
    const wverdict = (await wpack.query('how do bees tell each other where flowers are', { k: 5, showAbstained: true })).matchQuality;
    check('a windowed pack is calibrated and scores an on-topic question', wverdict === 'strong' || wverdict === 'weak', wverdict);
    await wpack.close();
    let appendRefusal = '';
    try { cli(['append', '--parent', windowedOut, '--source', more, '--out', path.join(tmp, 'windowed.0001.pikelet')]); } catch (error) { appendRefusal = String(error.stderr); }
    check('append refuses a windowed pack rather than misreading its rows', /sentence windows/.test(appendRefusal), appendRefusal.slice(0, 200));
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`\nComplete-profile reader conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
