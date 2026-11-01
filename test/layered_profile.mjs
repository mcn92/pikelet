// Layered-profile conformance — LAYERED_PROFILE.md section 10.
//
// This suite grows with the implementation. Currently covered: the tombstone
// segment codec (4.4), the cumulative/superset rule and delta (3.4), and the
// supersession constraints (3.5) — including the hostile cases family (B) of
// section 10 that apply to this segment, since a publisher can sign anything
// and a digest proves only authorship.
//
// Run standalone: node test/layered_profile.mjs

import {
    buildTombstoneSegment, parseTombstoneSegment, popcountBytes, bitsetBytesFor,
    testBit, zeroExtend, firstSupersetViolation, tombstoneDelta,
    TOMBSTONE_HEADER_BYTES, MAX_RECORDS,
} from '../packages/pikelet-wasm/complete/tombstones.mjs';
import {
    validateLayerObject, validateAgainstParent, canonicalJson,
    chainIngestDeclaration, validateIngestAgainstChain, rowTotal,
    LAYER_PROFILE, MAX_DEPTH,
} from '../packages/pikelet-wasm/complete/layer-manifest.mjs';
import { KINDS, KIND_NAMES } from '../packages/pikelet-wasm/complete/format.mjs';
import {
    buildInheritedQuerySegment, parseInheritedQuerySegment, unionBloom,
    QI_HEADER_BYTES, QI_KIND_INHERITED, LAYER_VOCAB_KIND,
} from '../packages/pikelet-wasm/complete/layer-encoder.mjs';
import {
    validateLocatorShape, resolveLocatorUrl, assertConfined,
    resolveLocatorPath, resolveParentLocation,
} from '../packages/pikelet-wasm/complete/layer-locator.mjs';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import {
    buildIntervalTable, ownerOf, projectMasks, liveCount,
    buildSupersessionMap, resolveSupersession, calibrationDrift,
    calibrationStatus, globalLexicalStats, idfFor,
    buildAncestry, resolveCitation,
} from '../packages/pikelet-wasm/complete/layer-chain.mjs';

let passed = 0;
let failed = 0;
function check(label, cond, detail = '') {
    if (cond) { passed++; console.log(`  ok: ${label}`); }
    else { failed++; console.log(`  FAIL: ${label}${detail ? ` — ${detail}` : ''}`); }
}
function rejects(label, fn, pattern) {
    try {
        fn();
        check(label, false, 'returned instead of throwing');
    } catch (err) {
        const msg = String(err && err.message);
        const ok = pattern.test(msg);
        check(label, ok, ok ? '' : `threw: ${msg.slice(0, 140)}`);
    }
}

console.log('tombstones-v1: round trip');
{
    const seg = buildTombstoneSegment({ rowBase: 100, tombstonedIds: [0, 7, 8, 63, 99], records: 5 });
    const t = parseTombstoneSegment(seg, { rowBase: 100, tombstones: 5, supersessions: 0, records: 5 });
    check('rowBase survives', t.rowBase === 100);
    check('popcount matches the ids set', t.tombstoneCount === 5);
    check('every id reads back set', [0, 7, 8, 63, 99].every((id) => testBit(t.bitset, id)));
    check('an unset id reads clear', !testBit(t.bitset, 1) && !testBit(t.bitset, 98));
    check('bitset is ceil(rowBase/8) bytes', t.bitset.length === bitsetBytesFor(100) && t.bitset.length === 13);
    check('segment tiles exactly', seg.length === TOMBSTONE_HEADER_BYTES + 13);
    // LSB-first within each byte (3.4): id 0 is bit 0 of byte 0, id 7 is bit 7.
    check('bit order is LSB-first', t.bitset[0] === 0b10000001, `byte0=${t.bitset[0].toString(2)}`);
}

console.log('tombstones-v1: empty and boundary shapes');
{
    const empty = buildTombstoneSegment({ rowBase: 0, tombstonedIds: [] });
    const t0 = parseTombstoneSegment(empty, { rowBase: 0, tombstones: 0, supersessions: 0 });
    check('rowBase 0 yields a header-only segment', empty.length === TOMBSTONE_HEADER_BYTES && t0.tombstoneCount === 0);

    // rowBase 8 fills exactly one byte, so there are no pad bits at all.
    const exact = buildTombstoneSegment({ rowBase: 8, tombstonedIds: [7] });
    check('rowBase on a byte boundary has no pad bits', parseTombstoneSegment(exact).tombstoneCount === 1);

    const one = buildTombstoneSegment({ rowBase: 1, tombstonedIds: [0] });
    check('rowBase 1 uses one byte with 7 pad bits', one.length === TOMBSTONE_HEADER_BYTES + 1);

    const none = buildTombstoneSegment({ rowBase: 500, tombstonedIds: [] });
    check('a layer may tombstone nothing', parseTombstoneSegment(none).tombstoneCount === 0);
}

console.log('tombstones-v1: supersessions (3.5)');
{
    const seg = buildTombstoneSegment({
        rowBase: 100, tombstonedIds: [10, 20, 30], records: 5,
        // Deliberately unsorted on input: the builder sorts by oldId.
        supersessions: [[30, 102], [10, 100]],
    });
    const t = parseTombstoneSegment(seg, { rowBase: 100, tombstones: 3, supersessions: 2, records: 5 });
    check('supersessions are stored sorted by oldId', JSON.stringify(t.supersessions) === JSON.stringify([[10, 100], [30, 102]]));
    check('segment length accounts for 8 bytes per edge',
        seg.length === TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100) + 16);

    rejects('an oldId that is not tombstoned is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[11, 100]] }),
        /not tombstoned/);
    rejects('a newId outside this layer\'s range is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[10, 150]] }),
        /outside this layer/);
    rejects('a newId below rowBase is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[10, 99]] }),
        /outside this layer/);
    rejects('a duplicate oldId within one layer is refused at build',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10], records: 5, supersessions: [[10, 100], [10, 101]] }),
        /more than one supersession/);
}

console.log('tombstones-v1: a layer cannot retract its own records (3.4)');
{
    rejects('an id at rowBase is out of range',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [100] }), /outside \[0, 100\)/);
    rejects('an id above rowBase is out of range',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [500] }), /outside \[0, 100\)/);
    rejects('a negative id is refused',
        () => buildTombstoneSegment({ rowBase: 100, tombstonedIds: [-1] }), /outside \[0, 100\)/);
}

console.log('tombstones-v1: hostile segments are refused (section 10 family B)');
{
    const good = buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10, 20], records: 4, supersessions: [[10, 100]] });
    const tamper = (fn) => { const c = good.slice(); fn(new DataView(c.buffer), c); return c; };

    rejects('a wrong version', () => parseTombstoneSegment(tamper((v) => v.setUint32(0, 2, true))), /version 2/);
    rejects('non-zero flags', () => parseTombstoneSegment(tamper((v) => v.setUint32(4, 1, true))), /flags must be 0/);
    rejects('non-zero reserved bytes', () => parseTombstoneSegment(tamper((v) => v.setUint32(28, 7, true))), /reserved bytes must be zero/);
    rejects('a tombstoneCount that disagrees with the popcount',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(16, 99n, true))), /popcount is 2/);
    rejects('a tombstoneCount above rowBase',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(16, 200n, true))), /exceeds rowBase/);
    rejects('a rowBase whose bitset no longer fits',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(8, 100000n, true))), /needs \d+ for the bitset/);
    rejects('a supersessionCount that does not tile the segment',
        () => parseTombstoneSegment(tamper((v) => v.setUint32(24, 9, true))), /needs 72/);
    rejects('a truncated segment', () => parseTombstoneSegment(good.subarray(0, 20)), /shorter than the 32-byte header/);
    rejects('a u64 above MAX_SAFE_INTEGER',
        () => parseTombstoneSegment(tamper((v) => v.setBigUint64(8, 2n ** 60n, true))), /MAX_SAFE_INTEGER/);

    // Pad bits: rowBase 100 leaves 4 pad bits in byte 12. Setting one must fail,
    // otherwise two byte-different segments would carry the same meaning and the
    // popcount check could be satisfied with a lie.
    rejects('a non-zero pad bit above rowBase', () => parseTombstoneSegment(tamper((v, c) => {
        c[TOMBSTONE_HEADER_BYTES + 12] |= 0b10000000;
    })), /pad bit/);

    // Manifest disagreement: the segment is well-formed but contradicts the
    // manifest that commits to it.
    rejects('a segment whose rowBase disagrees with the manifest',
        () => parseTombstoneSegment(good, { rowBase: 200, tombstones: 2, supersessions: 1 }),
        /disagrees with manifest layer.rowBase/);
    rejects('a segment whose tombstone count disagrees with the manifest',
        () => parseTombstoneSegment(good, { rowBase: 100, tombstones: 5, supersessions: 1 }),
        /disagrees with manifest layer.tombstones/);
    rejects('a segment whose supersession count disagrees with the manifest',
        () => parseTombstoneSegment(good, { rowBase: 100, tombstones: 2, supersessions: 4 }),
        /needs 32|disagrees with manifest layer.supersessions/);

    // Unsorted / duplicate supersessions, written past the builder's checks.
    const twoEdges = buildTombstoneSegment({ rowBase: 100, tombstonedIds: [10, 20], records: 4, supersessions: [[10, 100], [20, 101]] });
    rejects('supersessions out of order', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at, 20, true); v.setUint32(at + 8, 10, true);
    })), /sorted by oldId/);
    rejects('a duplicate oldId', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at + 8, 10, true);
    })), /sorted by oldId/);
    rejects('an oldId that is not tombstoned in the bitset', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at, 11, true);
    })), /not tombstoned/);
    rejects('an oldId at or above rowBase', () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
        const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
        v.setUint32(at, 100, true);
    })), /not below rowBase/);
    rejects('a newId below rowBase, with no manifest record count to bound it',
        () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
            const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
            v.setUint32(at + 4, 50, true);
        })), /below rowBase/);
    rejects('a newId outside the layer range stated by the manifest',
        () => parseTombstoneSegment(tamper2(twoEdges, (v) => {
            const at = TOMBSTONE_HEADER_BYTES + bitsetBytesFor(100);
            v.setUint32(at + 4, 900, true);
        }), { rowBase: 100, tombstones: 2, supersessions: 2, records: 4 }), /outside this layer/);
}
function tamper2(seg, fn) { const c = seg.slice(); fn(new DataView(c.buffer), c); return c; }

console.log('zeroExtend and the cumulative rule (3.4)');
{
    const parent = buildTombstoneSegment({ rowBase: 64, tombstonedIds: [1, 5] });
    const parentMask = parseTombstoneSegment(parent).bitset;

    const ext = zeroExtend(parentMask, 100);
    check('zeroExtend widens to the child\'s byte length', ext.length === bitsetBytesFor(100));
    check('zeroExtend preserves the parent\'s bits', testBit(ext, 1) && testBit(ext, 5));
    check('zeroExtend leaves the newly addressable range clear',
        popcountBytes(ext) === 2 && !testBit(ext, 64) && !testBit(ext, 99));
    check('zeroExtend of an empty parent is all zeros', popcountBytes(zeroExtend(new Uint8Array(0), 100)) === 0);

    // A conforming child: keeps both inherited bits and adds two of its own.
    const childMask = parseTombstoneSegment(
        buildTombstoneSegment({ rowBase: 100, tombstonedIds: [1, 5, 70, 80] }),
    ).bitset;
    check('a superset child passes', firstSupersetViolation(childMask, parentMask, 100) === -1);
    check('the delta is exactly the child\'s own deletions',
        JSON.stringify(tombstoneDelta(childMask, parentMask, 100)) === JSON.stringify([70, 80]));

    // A child that clears an inherited bit: the rule 3.4 exists to catch, and
    // 6.2's rebase correctness depends on it.
    const cleared = parseTombstoneSegment(
        buildTombstoneSegment({ rowBase: 100, tombstonedIds: [1, 70] }),
    ).bitset;
    check('a child clearing an inherited bit is caught, naming the id',
        firstSupersetViolation(cleared, parentMask, 100) === 5);

    check('an identical child has an empty delta',
        tombstoneDelta(parentMask, parentMask, 64).length === 0);
}

console.log('scaling (3.4)');
{
    // The documented sizes: the cumulative bitset costs ceil(rowBase/8) bytes
    // in every layer.
    check('456,153 records cost 57,020 bytes', bitsetBytesFor(456153) === 57020);
    check('1,000,000 records cost 125,000 bytes', bitsetBytesFor(1000000) === 125000);
    check('100,000,000 records cost 12.5 MB', bitsetBytesFor(100000000) === 12500000);
    rejects('a rowBase above the id ceiling is refused',
        () => buildTombstoneSegment({ rowBase: MAX_RECORDS + 1, tombstonedIds: [] }), /rowBase must be an integer/);
    rejects('rowBase + records above the id ceiling is refused',
        () => buildTombstoneSegment({ rowBase: MAX_RECORDS - 1, tombstonedIds: [], records: 10 }), /must not exceed/);
}

console.log('segment kinds (4.2)');
{
    check('tombstones is kind 6', KINDS.tombstones === 6 && KIND_NAMES[6] === 'tombstones');
    check('lineage is kind 7', KINDS.lineage === 7 && KIND_NAMES[7] === 'lineage');
    check('the complete profile\'s kinds are unchanged',
        KINDS.index === 1 && KINDS.corpus === 2 && KINDS['query-interp'] === 3
        && KINDS.evaluation === 4 && KINDS.lexical === 5);
}

// ---------------------------------------------------------------------------
// Layer manifest (4.1) and row arithmetic (3.3)
// ---------------------------------------------------------------------------
const INGEST = { chunker: 'v1', targetTokens: 256, overlapPercent: 15 };
const ENCODER = { kind: 'inline-transformer-v1', model: 'all-MiniLM-L6-v2' };
const BASE_ID = 'a'.repeat(64);
const L1_ID = 'b'.repeat(64);

function baseManifest(over = {}) {
    return {
        profile: 'pikelet-complete-v2', dim: 384, metric: 'cosine', encoder: ENCODER,
        corpus: { records: 1000, layout: 'records-v2', ingest: INGEST },
        index: { headerSha256: 'c'.repeat(64) },
        segments: [], ...over,
    };
}
function layerManifest(over = {}) {
    const { layer: layerOver, corpus: corpusOver, ...rest } = over;
    return {
        profile: LAYER_PROFILE, dim: 384, metric: 'cosine', encoder: ENCODER,
        layer: {
            parent: { identity: BASE_ID }, baseIdentity: BASE_ID, depth: 1,
            rowBase: 1000, records: 10, tombstones: 0, supersessions: 0, ingest: INGEST,
            ...layerOver,
        },
        corpus: { records: 10, layout: 'records-v2', ...corpusOver },
        index: { headerSha256: 'd'.repeat(64) },
        segments: [], ...rest,
    };
}

console.log('layer manifest: a well-formed layer');
{
    const m = layerManifest();
    const v = validateLayerObject(m);
    check('depth, rowBase and records survive', v.depth === 1 && v.rowBase === 1000 && v.records === 10);
    check('a layer with records is not tombstone-only', v.tombstoneOnly === false);
    const rel = validateAgainstParent(m, { manifest: baseManifest(), identity: BASE_ID, isBase: true });
    check('it validates against its base', rel.rowBase === 1000);
    check('rowTotal is rowBase + records', rowTotal(m) === 1010 && rowTotal(baseManifest()) === 1000);
}

console.log('layer manifest: tombstone-only layers (4.1, 4.3)');
{
    const t = layerManifest({
        layer: { records: 0, tombstones: 3 },
        corpus: { records: 0, layout: undefined },
        index: undefined,
    });
    const v = validateLayerObject(t);
    check('a tombstone-only layer is accepted', v.tombstoneOnly === true && v.records === 0);

    rejects('a tombstone-only layer carrying corpus.layout',
        () => validateLayerObject(layerManifest({ layer: { records: 0 }, corpus: { records: 0, layout: 'records-v2' }, index: undefined })),
        /must omit corpus.layout/);
    rejects('a tombstone-only layer carrying an index object',
        () => validateLayerObject(layerManifest({ layer: { records: 0 }, corpus: { records: 0, layout: undefined } })),
        /must omit the index object/);
    rejects('a tombstone-only layer recording a supersession (3.5)',
        () => validateLayerObject(layerManifest({ layer: { records: 0, tombstones: 2, supersessions: 1 }, corpus: { records: 0, layout: undefined }, index: undefined })),
        /tombstone-only layer cannot record a supersession/);
    rejects('a layer with records but no corpus.layout',
        () => validateLayerObject(layerManifest({ corpus: { records: 10, layout: undefined } })),
        /must declare corpus.layout/);
    rejects('a layer with records but no index object',
        () => validateLayerObject(layerManifest({ index: undefined })),
        /must carry an index object/);
}

console.log('layer manifest: hostile shapes (section 10 family B)');
{
    rejects('a complete-profile string', () => validateLayerObject(layerManifest({ profile: 'pikelet-complete-v2' })), /profile is/);
    rejects('no layer object', () => validateLayerObject({ profile: LAYER_PROFILE, corpus: { records: 0 } }), /no layer object/);
    rejects('a missing parent', () => validateLayerObject(layerManifest({ layer: { parent: undefined } })), /layer.parent is missing/);
    rejects('a short parent identity', () => validateLayerObject(layerManifest({ layer: { parent: { identity: 'abc' } } })), /64 lowercase hex/);
    rejects('an uppercase parent identity', () => validateLayerObject(layerManifest({ layer: { parent: { identity: 'A'.repeat(64) } } })), /64 lowercase hex/);
    rejects('a non-string locator', () => validateLayerObject(layerManifest({ layer: { parent: { identity: BASE_ID, locator: 7 } } })), /locator, when present/);
    rejects('depth 0', () => validateLayerObject(layerManifest({ layer: { depth: 0 } })), /layer.depth must be an integer/);
    rejects(`depth ${MAX_DEPTH + 1}`, () => validateLayerObject(layerManifest({ layer: { depth: MAX_DEPTH + 1 } })), /layer.depth must be an integer/);
    rejects('a negative rowBase', () => validateLayerObject(layerManifest({ layer: { rowBase: -1 } })), /layer.rowBase must be an integer/);
    rejects('rowBase + records past the id ceiling',
        () => validateLayerObject(layerManifest({ layer: { rowBase: MAX_RECORDS - 1, records: 10 }, corpus: { records: 10, layout: 'records-v2' } })),
        /exceeds 2147483647/);
    rejects('more tombstones than rowBase (a layer cannot tombstone its own rows)',
        () => validateLayerObject(layerManifest({ layer: { rowBase: 10, tombstones: 11 } })), /layer.tombstones must be an integer/);
    rejects('more supersessions than tombstones (3.5)',
        () => validateLayerObject(layerManifest({ layer: { tombstones: 1, supersessions: 2 } })), /exceeds layer.tombstones/);
    rejects('a missing ingestion declaration',
        () => validateLayerObject(layerManifest({ layer: { ingest: undefined } })), /layer.ingest is missing/);
    rejects('ingestAsserted smuggled inside the declaration (4.1)',
        () => validateLayerObject(layerManifest({ layer: { ingest: { ...INGEST, ingestAsserted: true } } })),
        /must not contain ingestAsserted/);
    rejects('a non-boolean ingestAsserted',
        () => validateLayerObject(layerManifest({ layer: { ingestAsserted: 'yes' } })), /must be a boolean/);
    rejects('corpus.records disagreeing with layer.records',
        () => validateLayerObject(layerManifest({ corpus: { records: 9, layout: 'records-v2' } })), /must equal layer.records/);
}

console.log('layer manifest: relational checks against the opened parent (3.3)');
{
    const base = baseManifest();
    const good = { manifest: base, identity: BASE_ID, isBase: true };

    rejects('a layer committing to a different parent identity',
        () => validateAgainstParent(layerManifest(), { manifest: base, identity: 'e'.repeat(64), isBase: true }),
        /but the opened parent is/);
    rejects('a wrong rowBase for the parent actually opened',
        () => validateAgainstParent(layerManifest({ layer: { rowBase: 999 } }), good),
        /but the opened parent implies 1000/);
    rejects('a wrong depth for the parent actually opened',
        () => validateAgainstParent(layerManifest({ layer: { depth: 2 } }), good),
        /but the opened parent is at depth 0/);
    rejects('a baseIdentity that is not the base\'s identity',
        () => validateAgainstParent(layerManifest({ layer: { baseIdentity: 'f'.repeat(64) } }), good),
        /but the chain's base is/);
    rejects('a layer that changes the encoder (4.1)',
        () => validateAgainstParent(layerManifest({ encoder: { kind: 'inline-transformer-v1', model: 'other' } }), good),
        /may not change the encoder/);
    rejects('a layer that changes dim',
        () => validateAgainstParent(layerManifest({ dim: 768 }), good), /differs from its parent's 384/);
    rejects('a layer that changes the metric',
        () => validateAgainstParent(layerManifest({ metric: 'l2' }), good), /differs from its parent's/);

    // Depth 2 onto a depth-1 layer: rowBase accumulates through the parent's
    // own rowBase, and a tombstone-only parent leaves rowBase unchanged (3.3).
    const l1 = layerManifest();
    const l2 = validateAgainstParent(
        layerManifest({ layer: { parent: { identity: L1_ID }, depth: 2, rowBase: 1010, records: 5 }, corpus: { records: 5, layout: 'records-v2' } }),
        { manifest: l1, identity: L1_ID, isBase: false },
    );
    check('depth 2 accumulates rowBase through its parent', l2.rowBase === 1010 && l2.depth === 2);

    const tombOnly = layerManifest({ layer: { records: 0, tombstones: 4 }, corpus: { records: 0, layout: undefined }, index: undefined });
    const after = validateAgainstParent(
        layerManifest({ layer: { parent: { identity: L1_ID }, depth: 2, rowBase: 1000, records: 3 }, corpus: { records: 3, layout: 'records-v2' } }),
        { manifest: tombOnly, identity: L1_ID, isBase: false },
    );
    check('rowBase is not unique along a chain: a tombstone-only parent leaves it unchanged (3.3)', after.rowBase === 1000);
}

console.log('chain ingestion declaration (4.1, 6.1)');
{
    const withIngest = baseManifest();
    const decl = chainIngestDeclaration(withIngest, null);
    check('a base carrying corpus.ingest defines the declaration', decl.source === 'base' && decl.asserted === false);
    check('a layer matching it passes', validateIngestAgainstChain(layerManifest(), decl) === true);
    rejects('a layer whose declaration differs',
        () => validateIngestAgainstChain(layerManifest({ layer: { ingest: { chunker: 'v2' } } }), decl),
        /differs from the chain's/);
    rejects('a layer asserting on a chain whose base declares',
        () => validateIngestAgainstChain(layerManifest({ layer: { ingestAsserted: true } }), decl),
        /but the chain's base declares one/);

    // A base with no corpus.ingest: only the depth-1 layer can assert one.
    const noIngest = baseManifest({ corpus: { records: 1000, layout: 'records-v2' } });
    rejects('a base with no declaration and no depth-1 layer',
        () => chainIngestDeclaration(noIngest, null), /no ingestion declaration/);
    rejects('a depth-1 layer that supplies one without asserting it',
        () => chainIngestDeclaration(noIngest, layerManifest()), /must set layer.ingestAsserted true/);
    const asserted = chainIngestDeclaration(noIngest, layerManifest({ layer: { ingestAsserted: true } }));
    check('a depth-1 layer may assert the declaration', asserted.asserted === true && asserted.source === 'depth-1');
    rejects('a later layer dropping the assertion',
        () => validateIngestAgainstChain(layerManifest({ layer: { depth: 2 } }), asserted),
        /must carry ingestAsserted true/);
    check('a later layer keeping it passes',
        validateIngestAgainstChain(layerManifest({ layer: { depth: 2, ingestAsserted: true } }), asserted) === true);
}

console.log('canonical JSON (4.1)');
{
    check('key order does not matter', canonicalJson({ b: 1, a: 2 }) === canonicalJson({ a: 2, b: 1 }));
    check('nested key order does not matter',
        canonicalJson({ x: { q: 1, p: 2 } }) === canonicalJson({ x: { p: 2, q: 1 } }));
    check('array order does matter', canonicalJson([1, 2]) !== canonicalJson([2, 1]));
    check('undefined members are dropped', canonicalJson({ a: 1, b: undefined }) === canonicalJson({ a: 1 }));
    // The property 4.1 depends on: asserting a declaration must not change its
    // canonical form, which is why ingestAsserted is a sibling.
    check('the declaration\'s canonical form is independent of assertion',
        canonicalJson(INGEST) === canonicalJson({ ...INGEST }));
}

// ---------------------------------------------------------------------------
// Query-interpretation kind 4, inherited-v1 (4.5)
// ---------------------------------------------------------------------------
const QI_SHA = '1'.repeat(64);
const BLOOM_GEOM = { bits: 2048, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'] };
const bloomOf = (setBits) => { const b = new Uint8Array(BLOOM_GEOM.bits / 8); for (const i of setBits) b[i >> 3] |= 1 << (i & 7); return b; };

console.log('inherited-v1: round trip');
{
    const seg = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: BLOOM_GEOM, vocabBloomBytes: bloomOf([1, 100, 2047]),
    });
    const p = parseInheritedQuerySegment(seg, {
        layerBaseIdentity: BASE_ID, baseQueryInterpSha256: QI_SHA, baseBloom: BLOOM_GEOM,
    });
    check('kind is 4', p.kind === QI_KIND_INHERITED);
    check('inheritFrom round-trips', p.inheritFrom === BASE_ID);
    check('queryInterpSha256 round-trips', p.queryInterpSha256 === QI_SHA);
    check('the layer bloom decodes to the base geometry', p.bloomBytes.length * 8 === BLOOM_GEOM.bits);
    check('calibration kind is layer-vocab-v1', p.calibrationKind === LAYER_VOCAB_KIND);
    // The point of kind 4 is that the encoder region is a COMMITMENT, not
    // weights: a ~24 MiB MiniLM blob would be two orders of magnitude larger.
    // (The segment's bulk is the layer's own base64 bloom, not the encoder.)
    const encRegion = new DataView(seg.buffer).getUint32(8, true);
    check('the encoder region is a commitment, not encoder weights',
        encRegion < 256, `encoder region is ${encRegion} bytes`);

    const none = buildInheritedQuerySegment({ baseIdentity: BASE_ID, queryInterpSha256: QI_SHA });
    const pn = parseInheritedQuerySegment(none, { layerBaseIdentity: BASE_ID, baseQueryInterpSha256: QI_SHA });
    check('a layer may ship kind "none"', pn.calibrationKind === 'none' && pn.bloomBytes === null);
}

console.log('inherited-v1: the encoder invariant (decision 5)');
{
    const seg = buildInheritedQuerySegment({ baseIdentity: BASE_ID, queryInterpSha256: QI_SHA });
    rejects('inheritFrom that is not the manifest\'s baseIdentity',
        () => parseInheritedQuerySegment(seg, { layerBaseIdentity: 'a9'.repeat(32), baseQueryInterpSha256: QI_SHA }),
        /does not equal the manifest's layer.baseIdentity/);
    rejects('a queryInterpSha256 that is not the base\'s query-interp digest',
        () => parseInheritedQuerySegment(seg, { layerBaseIdentity: BASE_ID, baseQueryInterpSha256: '2'.repeat(64) }),
        /does not match the base's query-interp digest/);
    rejects('a non-hex inheritFrom is refused at build',
        () => buildInheritedQuerySegment({ baseIdentity: 'nope', queryInterpSha256: QI_SHA }), /baseIdentity must be 64/);
    rejects('a non-hex queryInterpSha256 is refused at build',
        () => buildInheritedQuerySegment({ baseIdentity: BASE_ID, queryInterpSha256: 'nope' }), /queryInterpSha256 must be 64/);
}

console.log('inherited-v1: hostile segments (section 10 family B)');
{
    const seg = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: BLOOM_GEOM, vocabBloomBytes: bloomOf([5]),
    });
    const exp = { layerBaseIdentity: BASE_ID, baseQueryInterpSha256: QI_SHA, baseBloom: BLOOM_GEOM };
    const tamp = (fn) => { const c = seg.slice(); fn(new DataView(c.buffer), c); return c; };

    rejects('a wrong version', () => parseInheritedQuerySegment(tamp((v) => v.setUint32(0, 2, true)), exp), /unsupported version 2/);
    rejects('a kind that is not 4', () => parseInheritedQuerySegment(tamp((v) => v.setUint32(4, 3, true)), exp), /segment kind is 3/);
    rejects('regions that do not tile the segment',
        () => parseInheritedQuerySegment(tamp((v) => v.setUint32(8, 4, true)), exp), /layout is inconsistent/);
    rejects('a truncated segment', () => parseInheritedQuerySegment(seg.subarray(0, 8), exp), /shorter than its 16-byte header/);
    rejects('no expectation supplied', () => parseInheritedQuerySegment(seg, null), /no expectation supplied/);

    // A layer bloom whose geometry differs from the base's: the union would no
    // longer be an exact bitwise OR (4.5).
    const wrongBits = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: { bits: 4096, hashes: BLOOM_GEOM.hashes }, vocabBloomBytes: new Uint8Array(512),
    });
    rejects('a layer bloom with the wrong bit count',
        () => parseInheritedQuerySegment(wrongBits, exp), /the union would not be exact/);
    const wrongHashes = buildInheritedQuerySegment({
        baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
        vocabBloom: { bits: BLOOM_GEOM.bits, hashes: ['fnv1a:0'] }, vocabBloomBytes: bloomOf([5]),
    });
    rejects('a layer bloom with different hashes',
        () => parseInheritedQuerySegment(wrongHashes, exp), /differ from the base's/);
    rejects('a bloom whose base64 length disagrees with its declared bits',
        () => buildInheritedQuerySegment({
            baseIdentity: BASE_ID, queryInterpSha256: QI_SHA,
            vocabBloom: BLOOM_GEOM, vocabBloomBytes: new Uint8Array(4),
        }), /but vocabBloom.bits says/);
    rejects('an unknown calibration kind', () => {
        // Hand-build a segment whose calibration names a kind this reader does
        // not implement; it must fail closed rather than serve unscored.
        const enc = new TextEncoder();
        const e = enc.encode(JSON.stringify({ inheritFrom: BASE_ID, queryInterpSha256: QI_SHA }));
        const c = enc.encode(JSON.stringify({ kind: 'layer-vocab-v2' }));
        const out = new Uint8Array(QI_HEADER_BYTES + e.length + c.length);
        const v = new DataView(out.buffer);
        v.setUint32(0, 1, true); v.setUint32(4, 4, true); v.setUint32(8, e.length, true); v.setUint32(12, c.length, true);
        out.set(e, QI_HEADER_BYTES); out.set(c, QI_HEADER_BYTES + e.length);
        return parseInheritedQuerySegment(out, exp);
    }, /calibration kind must be/);
}

console.log('the union bloom (4.5)');
{
    const base = bloomOf([1, 2, 3]);
    const l1 = bloomOf([3, 10]);
    const l2 = bloomOf([200]);
    const u = unionBloom([base, l1, l2]);
    const isSet = (b, i) => ((b[i >> 3] >> (i & 7)) & 1) === 1;
    check('the union carries every contributor\'s bits',
        [1, 2, 3, 10, 200].every((i) => isSet(u, i)));
    check('the union sets nothing else', popcountBytes(u) === 5);
    check('overlapping bits are not double-counted', isSet(u, 3) && popcountBytes(unionBloom([bloomOf([3]), bloomOf([3])])) === 1);
    check('the union is exact, not approximate: OR of identical geometry',
        popcountBytes(unionBloom([base])) === 3);
    check('a null contributor is ignored', popcountBytes(unionBloom([base, null])) === 3);
    check('an all-null list yields null', unionBloom([null, null]) === null);
    rejects('a geometry mismatch is refused rather than silently truncated',
        () => unionBloom([base, new Uint8Array(8)]), /geometry mismatch/);
    // The union is monotone: it can only gain bits. This is why pollution is
    // one-sided (4.5) and cannot be undone without compaction.
    check('the union is a superset of every contributor',
        [base, l1, l2].every((b) => b.every((byte, i) => (byte & ~u[i]) === 0)));
}

// ---------------------------------------------------------------------------
// Parent resolution (5.1.1) — the one place a hostile manifest steers fetches
// ---------------------------------------------------------------------------
const CHILD_URL = 'https://cdn.example.com/packs/docs/docs.0007.9f3c.pikelet';

async function rejectsAsync(label, fn, pattern) {
    try {
        await fn();
        check(label, false, 'resolved instead of throwing');
    } catch (err) {
        const msg = String(err && err.message);
        const ok = pattern.test(msg);
        check(label, ok, ok ? '' : `threw: ${msg.slice(0, 140)}`);
    }
}

console.log('locator shape: only a relative reference under the child (5.1.1)');
{
    check('a plain sibling filename is accepted', validateLocatorShape('docs.0006.8e2b.pikelet') === true);
    check('a subdirectory is accepted', validateLocatorShape('layers/docs.0006.pikelet') === true);

    rejects('a scheme', () => validateLocatorShape('https://evil.example/x.pikelet'), /no scheme/);
    rejects('an authority', () => validateLocatorShape('//evil.example/x.pikelet'), /no authority/);
    rejects('a root-absolute path', () => validateLocatorShape('/admin/secret'), /not root-absolute/);
    rejects('a parent dot segment', () => validateLocatorShape('../../etc/passwd'), /dot segment/);
    rejects('a current dot segment', () => validateLocatorShape('./x.pikelet'), /dot segment/);
    // Every form on which a validator and a server could disagree is refused
    // rather than normalized — the governing rule of 5.1.1.
    rejects('an encoded path separator', () => validateLocatorShape('a%2fb.pikelet'), /encoded path separator/);
    rejects('an uppercase encoded separator', () => validateLocatorShape('a%2Fb.pikelet'), /encoded path separator/);
    rejects('an encoded backslash', () => validateLocatorShape('a%5cb.pikelet'), /encoded backslash/);
    rejects('a literal backslash', () => validateLocatorShape('a\\b.pikelet'), /literal backslash/);
    rejects('an encoded dot segment', () => validateLocatorShape('%2e%2e/x.pikelet'), /encoded dot segment/);
    rejects('a doubled slash', () => validateLocatorShape('a//b.pikelet'), /doubled slash/);
    rejects('an encoded NUL', () => validateLocatorShape('a%00b.pikelet'), /encoded NUL/);
    rejects('a control character', () => validateLocatorShape('a\u0001b.pikelet'), /control character/);
    rejects('a query string', () => validateLocatorShape('x.pikelet?a=1'), /no query or fragment/);
    rejects('a fragment', () => validateLocatorShape('x.pikelet#abc'), /no query or fragment/);
    rejects('a directory', () => validateLocatorShape('layers/'), /name a file/);
    rejects('an empty locator', () => validateLocatorShape(''), /non-empty string/);
    rejects('a non-string locator', () => validateLocatorShape(7), /non-empty string/);
}

console.log('locator resolution against the child\'s own URL (5.1.1)');
{
    check('a sibling resolves within the child\'s directory',
        resolveLocatorUrl('docs.0006.8e2b.pikelet', CHILD_URL)
            === 'https://cdn.example.com/packs/docs/docs.0006.8e2b.pikelet');
    check('a subdirectory resolves',
        resolveLocatorUrl('old/docs.0005.pikelet', CHILD_URL)
            === 'https://cdn.example.com/packs/docs/old/docs.0005.pikelet');
    rejects('a non-http child location', () => resolveLocatorUrl('x.pikelet', 'ftp://h/a/b'), /http and https only/);
}

console.log('redirect confinement, re-checked at every hop (5.1.1)');
{
    const child = new URL(CHILD_URL);
    check('a same-directory target is confined',
        assertConfined('https://cdn.example.com/packs/docs/other.pikelet', child) === true);
    check('a deeper path is confined',
        assertConfined('https://cdn.example.com/packs/docs/old/x.pikelet', child) === true);

    rejects('a cross-origin redirect',
        () => assertConfined('https://evil.example/packs/docs/x.pikelet', child), /leaves the child's origin/);
    rejects('a port change',
        () => assertConfined('https://cdn.example.com:8443/packs/docs/x.pikelet', child), /leaves the child's origin/);
    rejects('a scheme downgrade',
        () => assertConfined('http://cdn.example.com/packs/docs/x.pikelet', child), /changes scheme/);
    // The case 5.1.1 calls out by name: same origin, outside the directory.
    rejects('a same-origin redirect outside the directory',
        () => assertConfined('https://cdn.example.com/admin/private-object', child), /not under the child's directory/);
    rejects('a same-origin redirect to a sibling directory',
        () => assertConfined('https://cdn.example.com/packs/other/x.pikelet', child), /not under the child's directory/);
    rejects('a redirect carrying credentials',
        () => assertConfined('https://u:p@cdn.example.com/packs/docs/x.pikelet', child), /carries credentials/);
    rejects('a redirect naming the directory itself',
        () => assertConfined('https://cdn.example.com/packs/docs/', child), /names the directory itself/);
    // A redirect's Location is a fresh string off the network, so the raw-form
    // refusals apply again rather than only to the validated locator.
    rejects('a redirect target with an encoded backslash',
        () => assertConfined('https://cdn.example.com/packs/docs/a%5cb', child), /encoded backslash/);
}

console.log('file-source resolution follows symlinks (5.1.1)');
{
    // A fake fs where `link.pikelet` in the child's directory really lives
    // outside it: the textual check passes, the realpath check must not.
    const real = {
        '/packs/docs': '/packs/docs',
        '/packs/docs/parent.pikelet': '/packs/docs/parent.pikelet',
        '/packs/docs/link.pikelet': '/etc/secret.pikelet',
    };
    const fsops = {
        realpath: async (p) => { if (!(p in real)) throw new Error(`ENOENT ${p}`); return real[p]; },
        dirname: (p) => p.slice(0, p.lastIndexOf('/')) || '/',
        join: (...parts) => parts.join('/').replace(/\/+/g, '/'),
        sep: '/',
    };
    const childPath = '/packs/docs/child.pikelet';

    await (async () => {
        const got = await resolveLocatorPath('parent.pikelet', childPath, fsops);
        check('a real sibling resolves', got === '/packs/docs/parent.pikelet');
    })();
    await rejectsAsync('a symlink escaping the directory is refused on its real path',
        () => resolveLocatorPath('link.pikelet', childPath, fsops), /outside the child's real directory/);
    await rejectsAsync('a locator naming nothing is an availability failure',
        () => resolveLocatorPath('missing.pikelet', childPath, fsops), /cannot be located/);
}

console.log('resolution order: lineage, then locator, then host (5.1.1)');
{
    const want = { identity: BASE_ID, locator: 'parent.pikelet' };
    const lineage = new Map([[BASE_ID, 'https://ops.example.com/mirror/base.pikelet']]);

    await (async () => {
        // A lineage listing is operator-supplied and MAY name any origin the
        // operator trusts — deliberately not confined like a locator.
        const r = await resolveParentLocation(want, { lineage, childLocation: CHILD_URL });
        check('a lineage entry wins and may be cross-origin',
            r.via === 'lineage' && r.location === 'https://ops.example.com/mirror/base.pikelet', r.location);

        const r2 = await resolveParentLocation(want, { childLocation: CHILD_URL });
        check('the locator is used when no lineage entry exists',
            r2.via === 'locator' && r2.location === 'https://cdn.example.com/packs/docs/parent.pikelet');

        const r3 = await resolveParentLocation({ identity: BASE_ID, locator: null }, {
            childLocation: CHILD_URL, hostResolver: (id) => `https://host.example/${id}.pikelet`,
        });
        check('the host resolver is the last resort', r3.via === 'host');
    })();

    await rejectsAsync('resolveParents false refuses a locator',
        () => resolveParentLocation(want, { childLocation: CHILD_URL, resolveParents: false }),
        /resolution is disabled/);
    // 3.2: never serve the layers that were reached as a partial corpus.
    await rejectsAsync('an unlocatable ancestor fails the mount explicitly',
        () => resolveParentLocation({ identity: BASE_ID, locator: null }, { childLocation: CHILD_URL }),
        /cannot locate chain member/);
}

// ---------------------------------------------------------------------------
// Chain state: interval table, id ownership, masks (5.1 step 4, 5.2, 5.3)
// ---------------------------------------------------------------------------
const id64 = (n) => String(n).padStart(2, '0').repeat(32);

console.log('interval table and search tiers (3.1, 5.2)');
{
    // base 100 records, L1 +10, L2 tombstone-only, L3 +5.
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
        { depth: 1, identity: id64(2), rowBase: 100, records: 10 },
        { depth: 2, identity: id64(3), rowBase: 110, records: 0 },
        { depth: 3, identity: id64(4), rowBase: 110, records: 5 },
    ];
    const table = buildIntervalTable(members);
    check('the base is search tier 0', table.tiers[0].depth === 0 && table.tiers[0].rowBase === 0);
    check('a tombstone-only layer is not a search tier', table.tiers.length === 3);
    check('rowTotal counts every member\'s records', table.rowTotal === 115);
    check('tier intervals are contiguous',
        table.tiers[1].rowBase === 100 && table.tiers[1].end === 110
        && table.tiers[2].rowBase === 110 && table.tiers[2].end === 115);

    check('owner of a base id', ownerOf(table, 0).tier.depth === 0 && ownerOf(table, 0).localId === 0);
    check('owner of the last base id', ownerOf(table, 99).tier.depth === 0 && ownerOf(table, 99).localId === 99);
    check('owner of the first L1 id', ownerOf(table, 100).tier.depth === 1 && ownerOf(table, 100).localId === 0);
    // The trap 5.2 names: a tombstone-only layer shares its successor's
    // rowBase, so "largest rowBase <= id" would land on a member owning no ids.
    check('owner of an id after a tombstone-only layer is the layer that owns it',
        ownerOf(table, 110).tier.depth === 3 && ownerOf(table, 110).localId === 0);
    check('owner of the last id', ownerOf(table, 114).tier.depth === 3 && ownerOf(table, 114).localId === 4);
    check('an id past the end is rejected', ownerOf(table, 115) === null);
    check('a negative id is rejected', ownerOf(table, -1) === null);
    check('a non-integer id is rejected', ownerOf(table, 1.5) === null);

    rejects('a chain whose first member is not the base',
        () => buildIntervalTable([{ depth: 1, identity: id64(2), rowBase: 0, records: 5 }]), /must be the base/);
    rejects('a chain with a rowBase gap',
        () => buildIntervalTable([
            { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
            { depth: 1, identity: id64(2), rowBase: 105, records: 10 },
        ]), /expected 100/);
    rejects('a chain with no search tier',
        () => buildIntervalTable([{ depth: 0, identity: id64(1), rowBase: 0, records: 0 }]), /at least one search tier/);
}

console.log('mask projection onto each tier (5.3)');
{
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
        { depth: 1, identity: id64(2), rowBase: 100, records: 10 },
    ];
    const table = buildIntervalTable(members);
    // Head tombstones global ids 5, 99 (base) and 100, 109 (L1).
    const head = parseTombstoneSegment(
        buildTombstoneSegment({ rowBase: 110, tombstonedIds: [5, 99, 100, 109] }),
    ).bitset;
    const masks = projectMasks(table, head);
    check('one mask per search tier', masks.length === 2);
    check('base mask is ceil(records/8) bytes', masks[0].length === 13);
    check('a base tombstone projects to its local row', testBit(masks[0], 5) && testBit(masks[0], 99));
    check('an untombstoned base row is clear', !testBit(masks[0], 0) && !testBit(masks[0], 50));
    // The projection is the reason a tier's scan needs no global arithmetic:
    // global 100 is L1's local 0.
    check('an L1 tombstone projects to local 0 and 9', testBit(masks[1], 0) && testBit(masks[1], 9));
    check('L1 rows in between are clear', !testBit(masks[1], 5));
    check('popcounts sum to the head total', popcountBytes(masks[0]) + popcountBytes(masks[1]) === 4);
    check('live count is rowTotal minus tombstones', liveCount(table, 4) === 106);

    const noneMasked = projectMasks(table, new Uint8Array(0));
    check('an empty head bitset masks nothing',
        popcountBytes(noneMasked[0]) === 0 && popcountBytes(noneMasked[1]) === 0);
}

console.log('supersession views (3.5)');
{
    // A -> B recorded at depth 1, B -> C at depth 3, and a second edge for A
    // at depth 2 (the keep-both rebase case).
    const map = buildSupersessionMap([
        { depth: 1, supersessions: [[10, 100]] },
        { depth: 2, supersessions: [[10, 150]] },
        { depth: 3, supersessions: [[100, 200]] },
    ]);
    const a = resolveSupersession(map, 10);
    check('immediate is the least-depth edge', a.immediate === 100);
    check('all edges are ascending by depth', JSON.stringify(a.all) === JSON.stringify([100, 150]));
    // current: start at the greatest-depth edge (150), which has no further
    // edge, so it is the answer. 100's chain is not followed, because 150 is
    // the latest thing recorded for id 10.
    check('current starts from the greatest-depth edge', a.current === 150);

    const twoHop = buildSupersessionMap([
        { depth: 1, supersessions: [[10, 100]] },
        { depth: 2, supersessions: [[100, 200]] },
    ]);
    const b = resolveSupersession(twoHop, 10);
    check('current follows a two-hop chain A -> B -> C', b.current === 200);
    check('an id with no edges resolves to null', resolveSupersession(twoHop, 42).current === null);

    rejects('two edges for one oldId at the same depth',
        () => buildSupersessionMap([{ depth: 1, supersessions: [[10, 100], [10, 101]] }]),
        /at most one edge/);
}

console.log('calibration drift and status (3.1, 5.5)');
{
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 1000 },
        { depth: 1, identity: id64(2), rowBase: 1000, records: 50 },
        { depth: 2, identity: id64(3), rowBase: 1050, records: 30 },
    ];
    // Drift counts history: 80 appended + 20 tombstoned over a 1000-record base.
    check('drift is (appended + tombstoned) / base records',
        Math.abs(calibrationDrift(members, 20) - 0.1) < 1e-12);
    check('a record appended then tombstoned counts twice',
        Math.abs(calibrationDrift(members, 80) - 0.16) < 1e-12);

    const st = (drift, envelope, reader, opts = {}) => calibrationStatus({
        drift, producerEnvelope: envelope, readerLimit: reader,
        baseHasFit: true, everyLayerShipsBloom: true, ...opts,
    });
    check('within both limits is inherited', st(0.1, 0.5, 0.2).status === 'inherited');
    check('past the reader limit is drift-exceeded', st(0.3, 0.5, 0.2).status === 'drift-exceeded');
    check('past the producer envelope is drift-exceeded', st(0.15, 0.1, 0.2).status === 'drift-exceeded');
    // The precedence 5.5 fixes: a host can always be stricter; an artifact can
    // never make a host less strict.
    check('a looser manifest cannot raise the host\'s limit', st(0.5, 0.9, 0.2).effectiveLimit === 0.2);
    check('a stricter manifest lowers it', st(0.05, 0.1, 0.2).effectiveLimit === 0.1);
    check('an absent envelope is no constraint', st(0.15, undefined, 0.2).effectiveLimit === 0.2);
    check('no base fit means status none', st(0.01, 0.5, 0.2, { baseHasFit: false }).status === 'none');
    check('a layer shipping no bloom degrades the chain',
        st(0.01, 0.5, 0.2, { everyLayerShipsBloom: false }).status === 'drift-exceeded');
    rejects('an infinite reader limit is refused',
        () => st(0.1, 0.5, Infinity), /MUST be finite/);
}

console.log('global lexical statistics (5.4)');
{
    const stats = globalLexicalStats([
        { docCount: 1000, totalTokens: 150000 },
        { docCount: 50, totalTokens: 6000 },
    ]);
    check('N sums every search tier', stats.N === 1050);
    check('avgdl is total tokens over N', Math.abs(stats.avgdl - 156000 / 1050) < 1e-9);
    check('the reader must report that stats include tombstoned records',
        stats.statsIncludeTombstoned === true);
    // idf uses the chain-wide N and summed df.
    check('idf falls as df rises', idfFor(1050, 500) < idfFor(1050, 5));
    check('a single tier degenerates to the single-file case',
        globalLexicalStats([{ docCount: 10, totalTokens: 100 }]).avgdl === 10);
}

console.log('ancestry and citation scoping (5.6)');
{
    const members = [
        { depth: 0, identity: id64(1), rowBase: 0, records: 100 },
        { depth: 1, identity: id64(2), rowBase: 100, records: 10 },
        { depth: 2, identity: id64(3), rowBase: 110, records: 5 },
    ];
    const anc = buildAncestry(members);
    check('every head on the ancestry is listed', anc.size === 3);
    check('rowTotal is the ids that existed when that head was current',
        anc.get(id64(1)).rowTotal === 100 && anc.get(id64(2)).rowTotal === 110 && anc.get(id64(3)).rowTotal === 115);

    check('a citation against the base resolves', resolveCitation(anc, id64(1), 50).ok === true);
    check('a citation against a later head resolves', resolveCitation(anc, id64(3), 114).ok === true);
    // The case 5.6 calls out: (H3, id) with id >= rowTotal(H1) is invalid under
    // H1's identity, even though the id is valid in the current head.
    check('an id that postdates the cited head is invalid',
        resolveCitation(anc, id64(1), 105).reason === 'id-postdates-that-head');
    check('a fork not on this ancestry is answered as such',
        resolveCitation(anc, id64(9), 5).reason === 'not-on-this-history');
    check('a negative id is invalid', resolveCitation(anc, id64(1), -1).reason === 'invalid-id');
}

// ---------------------------------------------------------------------------
// Masked sketch search (5.3) — against a real PikeletSketchArtifact
// ---------------------------------------------------------------------------
// `pikelet-wasm/complete` is ESM. require() of an ES module works on Node 20+
// (require(esm) was backported there) but throws ERR_REQUIRE_ESM on Node 18,
// which the CI matrix still covers, so it is imported rather than required.
// Loaded here, not beside its one caller: a `const` does not hoist the way a
// function declaration does, and the caller runs earlier in the file.
const { openLexicalIndex: openLexicalIndexImpl } = await import('../packages/pikelet-wasm/complete/index.mjs');
const require_ = createRequire(import.meta.url);
const Pikelet = require_('pikelet-wasm');
const { exportSketchArtifact } = require_('pikelet-wasm/artifact');
const maskTmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-layered-'));

// A deterministic 1-D-ish geometry: record i sits at distance ~i from the
// query, so the nearest rows are 0, 1, 2, ... That makes "the C nearest are
// all masked" constructible exactly.
const MDIM = 16;
const MCOUNT = 64;
function ladderVector(i) {
    const v = new Float32Array(MDIM);
    v[0] = 1;
    v[1] = i / MCOUNT;          // grows with i, so cosine distance grows with i
    let n = 0;
    for (let d = 0; d < MDIM; d++) n += v[d] * v[d];
    n = Math.sqrt(n);
    for (let d = 0; d < MDIM; d++) v[d] /= n;
    return v;
}
function buildLadderSketch(recommendedRerank) {
    const qdata = new Uint8Array(MCOUNT * MDIM);
    const scales = new Float32Array(MCOUNT);
    const offsets = new Float32Array(MCOUNT);
    for (let i = 0; i < MCOUNT; i++) {
        const v = ladderVector(i);
        let mn = Infinity; let mx = -Infinity;
        for (let d = 0; d < MDIM; d++) { if (v[d] < mn) mn = v[d]; if (v[d] > mx) mx = v[d]; }
        const sc = (mx - mn) / 255 || 1e-12;
        scales[i] = sc; offsets[i] = mn;
        for (let d = 0; d < MDIM; d++) {
            const b = Math.round((v[d] - mn) / sc);
            qdata[i * MDIM + d] = b < 0 ? 0 : b > 255 ? 255 : b;
        }
    }
    const p = path.join(maskTmp, `ladder-${recommendedRerank}.pikelet-sketch`);
    exportSketchArtifact({ dim: MDIM, count: MCOUNT, metric: 1, qdata, scales, offsets }, p,
        { sketchDims: MDIM, sketchBits: 8, recommendedRerank });
    return p;
}
const maskOf = (rows) => { const b = new Uint8Array(Math.ceil(MCOUNT / 8)); for (const r of rows) b[r >> 3] |= 1 << (r & 7); return b; };

console.log('masked search: a masked row never enters the result (5.3)');
{
    const art = await Pikelet.openSketchArtifactFile(buildLadderSketch(40));
    const q = ladderVector(0);

    const plain = await art.search(q, 5);
    const order = plain.results.map((r) => r.id);
    check('unmasked search returns the nearest rows in order',
        order[0] === 0 && order[1] === 1, JSON.stringify(order));

    const masked = await art.search(q, 5, { exclude: maskOf([0, 1, 2]) });
    const mids = masked.results.map((r) => r.id);
    check('masked rows are absent from the results', !mids.some((id) => [0, 1, 2].includes(id)), JSON.stringify(mids));
    check('the next live rows take their place', mids[0] === 3, JSON.stringify(mids));
    check('k results are still returned', mids.length === 5);

    // THE masking case of section 10 fixture (A): the C nearest rows are all
    // masked and the nearest live row is the C+1-th. A post-hoc mask would
    // return nothing here; masking before truncation returns the live row.
    const small = await Pikelet.openSketchArtifactFile(buildLadderSketch(4));
    const allNearestMasked = await small.search(q, 1, { rerank: 4, exclude: maskOf([0, 1, 2, 3]) });
    check('with the C nearest rows all masked, the C+1-th live row is still found',
        allNearestMasked.results.length === 1 && allNearestMasked.results[0].id === 4,
        JSON.stringify(allNearestMasked.results.map((r) => r.id)));

    const everything = await art.search(q, 5, { exclude: maskOf(Array.from({ length: MCOUNT }, (_, i) => i)) });
    check('a fully masked tier returns no results, not an error', everything.results.length === 0);

    const noneMasked = await art.search(q, 5, { exclude: new Uint8Array(Math.ceil(MCOUNT / 8)) });
    check('an all-zero mask is identical to no mask',
        JSON.stringify(noneMasked.results.map((r) => r.id)) === JSON.stringify(order));

    // C counts unmasked rows, so a heavily masked tier still offers a full pool.
    const heavy = await art.search(q, 3, { rerank: 8, exclude: maskOf(Array.from({ length: 50 }, (_, i) => i)) });
    check('C counts unmasked rows: a heavily masked tier still fills k',
        heavy.results.length === 3 && heavy.results[0].id === 50,
        JSON.stringify(heavy.results.map((r) => r.id)));

    // search() is async, so its validation surfaces as a rejected promise.
    await rejectsAsync('a non-Uint8Array mask is refused',
        () => art.search(q, 5, { exclude: [1, 2, 3] }), /must be a Uint8Array/);
    await rejectsAsync('a mask too short for the artifact is refused',
        () => art.search(q, 5, { exclude: new Uint8Array(2) }), /too short/);

    await art.close();
    await small.close();
}

console.log('masked search with a WASM scanner: over-fetch, never post-truncate (5.3)');
{
    // A scan kernel knows nothing about a mask, so masking its top-C output
    // would be the post-truncation defect 5.3 forbids. The kernel is used
    // safely by asking for C + maskedCount candidates and dropping the masked
    // ones: at least C live rows survive by construction.
    const art = await Pikelet.openSketchArtifactFile(buildLadderSketch(40));
    const q = ladderVector(0);
    let lastC = null;
    const scanner = {
        metric: 1, sketchDims: MDIM, maxRerank: 1024,
        scan(_q, c) {
            lastC = c;
            // A real kernel returns the c nearest; the ladder makes that 0..c-1.
            return Array.from({ length: Math.min(c, MCOUNT) }, (_, i) => i);
        },
    };
    await art.search(q, 5, { scanner, rerank: 10 });
    check('an unmasked query asks the scanner for exactly C', lastC === 10);

    const masked = await art.search(q, 5, { scanner, rerank: 10, exclude: maskOf([0, 1, 2]) });
    check('a masked query over-fetches by the masked count', lastC === 13, `asked for ${lastC}`);
    const mids = masked.results.map((r) => r.id);
    check('masked rows are absent from the result', !mids.some((id) => [0, 1, 2].includes(id)), JSON.stringify(mids));
    check('and the nearest live row leads', mids[0] === 3, JSON.stringify(mids));

    // The case that made the naive post-filter wrong: EVERY row the kernel
    // would have returned at C is masked. Over-fetching still finds live rows.
    const allTopMasked = await art.search(q, 2, { scanner, rerank: 4, exclude: maskOf([0, 1, 2, 3]) });
    const atm = allTopMasked.results.map((r) => r.id);
    check('with the C nearest all masked, over-fetch still returns live rows',
        atm.length === 2 && atm[0] === 4, JSON.stringify(atm));

    // A scanner whose buffers cannot hold the over-fetch must fall back to the
    // masked JS scan rather than silently return a short pool.
    let smallCalls = 0;
    const smallScanner = { metric: 1, sketchDims: MDIM, maxRerank: 6, scan() { smallCalls++; return [0, 1, 2, 3, 4, 5]; } };
    const fellBack = await art.search(q, 3, { scanner: smallScanner, rerank: 5, exclude: maskOf(Array.from({ length: 20 }, (_, i) => i)) });
    check('an over-fetch past the scanner\'s buffers falls back to the JS scan', smallCalls === 0);
    check('and the fallback is still correct',
        !fellBack.results.map((r) => r.id).some((id) => id < 20), JSON.stringify(fellBack.results.map((r) => r.id)));
    await art.close();
}

// ---------------------------------------------------------------------------
// Real chain files: a base plus two layers, assembled and re-read (4.1-4.5)
// ---------------------------------------------------------------------------
const {
    assemblePikeletFile, buildCorpusSegment, buildQueryInterpSegment,
    buildLexicalSegment: buildLexSeg, PROFILE_LAYER, PROFILE_V2,
} = await import('../packages/pikelet-wasm/complete/builder.mjs');
const crypto = await import('node:crypto');
const sha256hexOf = (b) => crypto.createHash('sha256').update(b).digest('hex');

console.log('chain files: a base and two layers assemble and re-read');
{
    const CDIM = 16;
    const recBytes = (i) => Buffer.from(JSON.stringify({ title: `rec ${i}`, text: `record ${i} body text` }));
    const sketchFor = (ids) => {
        const n = ids.length;
        const qdata = new Uint8Array(n * CDIM);
        const scales = new Float32Array(n);
        const offsets = new Float32Array(n);
        for (let r = 0; r < n; r++) {
            const v = ladderVector(ids[r] % MCOUNT);
            let mn = Infinity; let mx = -Infinity;
            for (let d = 0; d < CDIM; d++) { if (v[d] < mn) mn = v[d]; if (v[d] > mx) mx = v[d]; }
            const sc = (mx - mn) / 255 || 1e-12;
            scales[r] = sc; offsets[r] = mn;
            for (let d = 0; d < CDIM; d++) {
                const b = Math.round((v[d] - mn) / sc);
                qdata[r * CDIM + d] = b < 0 ? 0 : b > 255 ? 255 : b;
            }
        }
        const sp = path.join(chainTmp, `sk-${ids[0]}-${n}.pikelet-sketch`);
        exportSketchArtifact({ dim: CDIM, count: n, metric: 1, qdata, scales, offsets }, sp,
            { sketchDims: CDIM, sketchBits: 8, recommendedRerank: 20 });
        return fs.readFileSync(sp);
    };
    const chainTmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-chain-'));
    const INGEST2 = { chunker: 'v1', targetTokens: 256 };
    const ENC2 = { kind: 'host-encoder-v1', model: 'test' };

    // --- base: 20 records ---
    const baseIds = Array.from({ length: 20 }, (_, i) => i);
    const baseCorpus = buildCorpusSegment(baseIds.map(recBytes));
    const baseQi = buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' })));
    const baseLex = buildLexSeg(baseIds.map((i) => `record ${i} body text`));
    const basePath = path.join(chainTmp, 'base.pikelet');
    const baseBuilt = assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...baseCorpus.corpus, ingest: INGEST2, provenance: { name: 'ladder-docs' } }, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(baseIds) },
        { kind: 'corpus', bytes: baseCorpus.bytes },
        { kind: 'query-interp', bytes: baseQi },
        { kind: 'lexical', bytes: baseLex.bytes },
    ], basePath);
    const baseIdentity = baseBuilt.identity;
        check('the base assembles and yields an identity', /^[0-9a-f]{64}$/.test(baseIdentity));

    const baseQiSha = baseBuilt.manifest.segments.find((x) => x.kind === 'query-interp').sha256;
    check('the base manifest carries a query-interp digest to inherit', /^[0-9a-f]{64}$/.test(baseQiSha));

    // --- layer 1: +5 records, tombstones base ids 3 and 4 ---
    const l1Ids = [20, 21, 22, 23, 24];
    const l1Corpus = buildCorpusSegment(l1Ids.map(recBytes));
    const l1Tomb = buildTombstoneSegment({ rowBase: 20, tombstonedIds: [3, 4], records: 5, supersessions: [[3, 20]] });
    const l1Qi = buildInheritedQuerySegment({ baseIdentity: baseIdentity, queryInterpSha256: baseQiSha });
    const l1Lex = buildLexSeg(l1Ids.map((i) => `record ${i} body text`));
    const l1Path = path.join(chainTmp, 'l1.pikelet');
    const l1Built = assemblePikeletFile({
        profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
        layer: {
            parent: { identity: baseIdentity, locator: 'base.pikelet' },
            baseIdentity: baseIdentity, depth: 1, rowBase: 20, records: 5,
            tombstones: 2, supersessions: 1, ingest: INGEST2,
        },
        corpus: l1Corpus.corpus, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(l1Ids) },
        { kind: 'corpus', bytes: l1Corpus.bytes },
        { kind: 'query-interp', bytes: l1Qi },
        { kind: 'lexical', bytes: l1Lex.bytes },
        { kind: 'tombstones', bytes: l1Tomb },
    ], l1Path);
    const l1Identity = l1Built.identity;
        check('a layer assembles with a tombstones segment', /^[0-9a-f]{64}$/.test(l1Identity));

    // --- layer 2: tombstone-only, hides base id 7 and keeps l1's bits ---
    const l2Tomb = buildTombstoneSegment({ rowBase: 25, tombstonedIds: [3, 4, 7], records: 0 });
    const l2Qi = buildInheritedQuerySegment({ baseIdentity: baseIdentity, queryInterpSha256: baseQiSha });
    const l2Path = path.join(chainTmp, 'l2.pikelet');
    const l2Built = assemblePikeletFile({
        profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
        layer: {
            parent: { identity: l1Identity, locator: 'l1.pikelet' },
            baseIdentity: baseIdentity, depth: 2, rowBase: 25, records: 0,
            tombstones: 3, supersessions: 0, ingest: INGEST2,
        },
        corpus: { records: 0 },
    }, [
        { kind: 'query-interp', bytes: l2Qi },
        { kind: 'tombstones', bytes: l2Tomb },
    ], l2Path);
    const l2Identity = l2Built.identity;
        check('a tombstone-only layer assembles with no index or corpus segment',
        /^[0-9a-f]{64}$/.test(l2Identity));

    // --- read the chain back and validate every relation ---
    const readManifest = (p) => {
        const buf = fs.readFileSync(p);
        const mlen = new DataView(buf.buffer, buf.byteOffset, 64).getUint32(8, true);
        return JSON.parse(buf.subarray(64, 64 + mlen).toString('utf8'));
    };
    const bm = readManifest(basePath);
    const l1m = readManifest(l1Path);
    const l2m = readManifest(l2Path);

    check('the layer profile string is what a pre-layered reader would refuse',
        l1m.profile === 'pikelet-layer-v1');
    const v1 = validateAgainstParent(l1m, { manifest: bm, identity: baseIdentity, isBase: true });
    check('layer 1 validates against the real base', v1.rowBase === 20 && v1.depth === 1);
    const v2 = validateAgainstParent(l2m, { manifest: l1m, identity: l1Identity, isBase: false });
    check('layer 2 validates against the real layer 1', v2.rowBase === 25 && v2.tombstoneOnly === true);

    const decl = chainIngestDeclaration(bm, l1m);
    check('the chain ingestion declaration comes from the base', decl.source === 'base');
    check('both layers match it',
        validateIngestAgainstChain(l1m, decl) && validateIngestAgainstChain(l2m, decl));

    // The kind-4 commitment, checked against the base actually read.
    const l1QiSeg = (() => {
        const buf = fs.readFileSync(l1Path);
        const idx = l1m.segments.findIndex((x) => x.kind === 'query-interp');
        const mlen = new DataView(buf.buffer, buf.byteOffset, 64).getUint32(8, true);
        const tableAt = 64 + mlen;
        const off = Number(new DataView(buf.buffer, buf.byteOffset).getBigUint64(tableAt + idx * 48 + 8, true));
        const len = Number(new DataView(buf.buffer, buf.byteOffset).getBigUint64(tableAt + idx * 48 + 16, true));
        return new Uint8Array(buf.subarray(off, off + len));
    })();
    const inh = parseInheritedQuerySegment(l1QiSeg, {
        layerBaseIdentity: baseIdentity, baseQueryInterpSha256: baseQiSha,
    });
    check('the layer inherits the base\'s encoder by digest', inh.inheritFrom === baseIdentity);
    check('every segment digest in the manifest matches its bytes', (() => {
        for (const p of [basePath, l1Path, l2Path]) {
            const buf = fs.readFileSync(p);
            const m = readManifest(p);
            const mlen = new DataView(buf.buffer, buf.byteOffset, 64).getUint32(8, true);
            const tableAt = 64 + mlen;
            for (let i = 0; i < m.segments.length; i++) {
                const dv = new DataView(buf.buffer, buf.byteOffset);
                const off = Number(dv.getBigUint64(tableAt + i * 48 + 8, true));
                const len = Number(dv.getBigUint64(tableAt + i * 48 + 16, true));
                if (sha256hexOf(buf.subarray(off, off + len)) !== m.segments[i].sha256) return false;
            }
        }
        return true;
    })());

    // Chain state over the real members.
    const members = [
        { depth: 0, identity: baseIdentity, rowBase: 0, records: 20 },
        { depth: 1, identity: l1Identity, rowBase: 20, records: 5 },
        { depth: 2, identity: l2Identity, rowBase: 25, records: 0 },
    ];
    const table = buildIntervalTable(members);
    const headTomb = parseTombstoneSegment(l2Tomb, { rowBase: 25, tombstones: 3, supersessions: 0 });
    check('the chain has two search tiers and 25 ids', table.tiers.length === 2 && table.rowTotal === 25);
    check('live count is 22 after three tombstones', liveCount(table, headTomb.tombstoneCount) === 22);
    const masks = projectMasks(table, headTomb.bitset);
    check('the head mask projects onto the base tier',
        testBit(masks[0], 3) && testBit(masks[0], 4) && testBit(masks[0], 7) && popcountBytes(masks[0]) === 3);
    check('layer 1\'s tier is unmasked', popcountBytes(masks[1]) === 0);

    // The cumulative rule across the real segments.
    const l1Mask = parseTombstoneSegment(l1Tomb).bitset;
    check('layer 2\'s mask is a superset of layer 1\'s',
        firstSupersetViolation(headTomb.bitset, l1Mask, 25) === -1);
    check('layer 2\'s own delta is just base id 7',
        JSON.stringify(tombstoneDelta(headTomb.bitset, l1Mask, 25)) === JSON.stringify([7]));

    // A masked search over the base tier's real sketch: the tombstoned rows
    // must not come back.
    const baseSketchBytes = sketchFor(baseIds);
    const bsPath = path.join(chainTmp, 'basesketch.pikelet-sketch');
    fs.writeFileSync(bsPath, baseSketchBytes);
    const baseArt = await Pikelet.openSketchArtifactFile(bsPath);
    const res = await baseArt.search(ladderVector(3), 5, { exclude: masks[0] });
    const got = res.results.map((r) => r.id);
    check('a masked search over the real base tier omits every tombstoned row',
        !got.some((id) => [3, 4, 7].includes(id)), JSON.stringify(got));
    await baseArt.close();

    // Supersession over the real segments: base id 3 was replaced by id 20.
    const supMap = buildSupersessionMap([
        { depth: 1, supersessions: parseTombstoneSegment(l1Tomb).supersessions },
        { depth: 2, supersessions: headTomb.supersessions },
    ]);
    check('the union supersession map resolves the recorded edge',
        resolveSupersession(supMap, 3).current === 20);

    // Global lexical statistics over both search tiers.
    const stats = globalLexicalStats([
        { docCount: baseLex.meta.docCount, totalTokens: openLexicalIndexFor(baseLex.bytes).totalTokens },
        { docCount: l1Lex.meta.docCount, totalTokens: openLexicalIndexFor(l1Lex.bytes).totalTokens },
    ]);
    check('N is the sum over search tiers, tombstoned records included', stats.N === 25);
    check('the reader reports that stats include tombstoned records', stats.statsIncludeTombstoned === true);

    // ------------------------------------------------------------------
    // Mount the chain and serve it (5.1, 5.3, 5.6)
    // ------------------------------------------------------------------
    const { openPikeletChain } = await import('../packages/pikelet-wasm/complete/layer-reader.mjs');
    const chain = await openPikeletChain([basePath, l1Path, l2Path]);
    const ci = chain.info();
    check('the chain mounts every member', ci.layers === 3 && ci.searchTiers === 2);
    check('it reports the head identity, not a member\'s', ci.identity === l2Identity);
    check('and the base identity separately', ci.baseIdentity === baseIdentity);
    check('the chain carries the base\'s name, so a host need not name it after a layer file', ci.name === 'ladder-docs', ci.name);
    check('record counts are chain-wide', ci.records === 25 && ci.liveRecords === 22 && ci.tombstones === 3);
    check('global lexical stats count tombstoned records (5.4)',
        ci.lexical.N === 25 && ci.lexical.statsIncludeTombstoned === true);

    const qres = await chain.query(ladderVector(3), 5, { text: 'record 3 body text' });
    const qids = qres.results.map((r) => r.id);
    check('a query returns no tombstoned id', !qids.some((id) => [3, 4, 7].includes(id)), JSON.stringify(qids));
    check('results carry the owning layer depth', qres.results.every((r) => Number.isInteger(r.layer)));
    check('the response carries the head identity for citation (5.3)', qres.identity === l2Identity);

    // A query whose nearest rows are all tombstoned still finds live ones.
    const nearDeleted = await chain.query(ladderVector(4), 3, {});
    check('a query nearest a deleted record returns live neighbours',
        !nearDeleted.results.map((r) => r.id).some((id) => [3, 4, 7].includes(id)));

    // record() over global ids, across the tier boundary.
    const rBase = await chain.record(2);
    check('record() resolves a base id', rBase.id === 2 && rBase.layer === 0 && rBase.tombstoned === false);
    const rLayer = await chain.record(22);
    check('record() resolves an id owned by a layer', rLayer.id === 22 && rLayer.layer === 1);
    const rTomb = await chain.record(3);
    check('record() reports a tombstoned record with its supersession (5.6)',
        rTomb.tombstoned === true && rTomb.supersededBy === 20 && rTomb.currentSuccessor === 20);
    await rejectsAsync('an id past the chain is refused', () => chain.record(25), /outside every search tier/);

    // Citations resolve only along the mounted ancestry (5.6).
    check('a citation against the base resolves', chain.citation(baseIdentity, 5).ok === true);
    check('a citation against the head resolves', chain.citation(l2Identity, 24).ok === true);
    check('an id that postdates the cited head is invalid',
        chain.citation(baseIdentity, 22).reason === 'id-postdates-that-head');
    check('a fork not on this ancestry is answered as such',
        chain.citation('9'.repeat(64), 1).reason === 'not-on-this-history');

    // 5.5: this base carries no fit, so the chain is `none`, and drift is
    // still computed and reported.
    // 5.3 step 3: a tier's lexical hits join its exact rerank. A query vector
    // at ladder 24 puts base record 10 fourteen rungs away; each tier used to
    // return only its min(4k, records) nearest with no lexical extras, so at
    // k=2 the base tier offered 12..19 and the exact-match record 10 was not a
    // vector candidate. Fusion only reorders vector candidates, so it was
    // dropped. (The fusion guard keeps the vector top hit first; the lexical
    // match lands second.)
    const known = await chain.query(ladderVector(24), 2, { text: '10' });
    check('a lexical-only hit outside the vector window is still returned',
        known.results.some((r) => r.id === 10 && r.lexicalRank === 1),
        JSON.stringify(known.results.map((r) => [r.id, r.lexicalRank, r.vectorRank])));

    // 5.4: df is summed across tiers. "record" and "body" occur in every
    // record (base 20, layer 5) with identical tf and length, so under one
    // global idf every live record ties and the tie breaks by ascending id:
    // the base's records lead. Per-tier df gave the layer's 5 records a far
    // larger idf and they took the top of the lexical list.
    const flat = await chain.query(ladderVector(0), 22, { text: 'record body' });
    const lexFirst = flat.results.find((r) => r.lexicalRank === 1);
    check('global df: the lexical list is led by the base, not inflated layer scores',
        lexFirst?.id === 0 && lexFirst?.layer === 0, JSON.stringify(lexFirst && [lexFirst.id, lexFirst.layer]));
    const lexOnly = await chain.query(ladderVector(0), 3, { text: 'record body', retrieval: 'lexical' });
    check('retrieval lexical returns lexical order',
        JSON.stringify(lexOnly.results.map((r) => r.id)) === JSON.stringify([0, 1, 2]),
        JSON.stringify(lexOnly.results.map((r) => r.id)));
    const baseLexIdx = openLexicalIndexFor(baseLex.bytes);
    const layerLexIdx = openLexicalIndexFor(l1Lex.bytes);
    const gdf = new Map([['record', 25]]);
    const gstats = { docCount: 25, avgdl: stats.avgdl };
    check('documentFrequencies reports a tier\'s own df per query term',
        baseLexIdx.documentFrequencies('record zzz').get('record') === 20
        && !baseLexIdx.documentFrequencies('record zzz').has('zzz'));
    // Records 10..24 all have four tokens (the tokenizer drops one-character
    // ones, so 0..9 have three); compare base 10 with layer 20.
    const scoreOf = (idx, id) => idx.search('record', 25, { stats: gstats, df: gdf }).find((h) => h.id === id)?.score;
    check('with options.df both tiers score an identical record identically',
        scoreOf(baseLexIdx, 10) !== undefined && scoreOf(baseLexIdx, 10) === scoreOf(layerLexIdx, 0));
    rejects('a global df below the tier\'s own is refused',
        () => baseLexIdx.search('record', 1, { df: new Map([['record', 3]]) }), /below this index's own df/);
    await rejectsAsync('an unknown retrieval mode is refused, not treated as hybrid',
        () => chain.query(ladderVector(0), 1, { text: 'x', retrieval: 'bogus' }), /retrieval must be/);

    check('a base with no fit yields calibrationStatus none', ci.calibrationStatus === 'none');
    check('drift is reported even when unscored', Math.abs(ci.calibrationDrift - 0.4) < 1e-9);
    check('the effective limit is the reader\'s finite default', ci.driftLimit === 0.2);
    await chain.close();

    // 4.1: a layer opened alone must still be refused — it would serve a
    // fraction of a corpus.
    const { openPikeletFile: openOne } = await import('../packages/pikelet-wasm/complete/index.mjs');
    await rejectsAsync('a layer opened alone is refused',
        () => openOne(l1Path), /unsupported profile pikelet-layer-v1/);
    await rejectsAsync('a chain whose first member is a layer is refused',
        () => openPikeletChain([l1Path, l2Path]), /unsupported profile|first chain member is a layer/);

    // 3.2: a chain member that does not link is refused at mount.
    const strayPath = path.join(chainTmp, 'stray.pikelet');
    const strayCorpus = buildCorpusSegment([recBytes(99)]);
    assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...strayCorpus.corpus, ingest: INGEST2 }, index: {},
    }, [
        { kind: 'index', bytes: sketchFor([99]) },
        { kind: 'corpus', bytes: strayCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
    ], strayPath);
    await rejectsAsync('a layer mounted on a base it does not commit to is refused',
        () => openPikeletChain([strayPath, l1Path]), /but the opened parent is/);

    fs.rmSync(chainTmp, { recursive: true, force: true });
}
function openLexicalIndexFor(bytes) {
    return openLexicalIndexImpl(bytes);
}

fs.rmSync(maskTmp, { recursive: true, force: true });

// ---------------------------------------------------------------------------
// Calibration status with a real fit in the base (5.5)
// ---------------------------------------------------------------------------
console.log('chain calibration: inherited and drift-exceeded with a real fit');
{
    const CDIM = 16;
    const calTmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-cal-'));
    const recBytes = (i) => Buffer.from(JSON.stringify({ title: `rec ${i}`, text: `record ${i} body text` }));
    const sketchFor = (ids, tag) => {
        const n = ids.length;
        const qdata = new Uint8Array(n * CDIM);
        const scales = new Float32Array(n);
        const offsets = new Float32Array(n);
        for (let r = 0; r < n; r++) {
            const v = ladderVector(ids[r] % MCOUNT);
            let mn = Infinity; let mx = -Infinity;
            for (let d = 0; d < CDIM; d++) { if (v[d] < mn) mn = v[d]; if (v[d] > mx) mx = v[d]; }
            const sc = (mx - mn) / 255 || 1e-12;
            scales[r] = sc; offsets[r] = mn;
            for (let d = 0; d < CDIM; d++) {
                const b = Math.round((v[d] - mn) / sc);
                qdata[r * CDIM + d] = b < 0 ? 0 : b > 255 ? 255 : b;
            }
        }
        const sp = path.join(calTmp, `sk-${tag}.pikelet-sketch`);
        exportSketchArtifact({ dim: CDIM, count: n, metric: 1, qdata, scales, offsets }, sp,
            { sketchDims: CDIM, sketchBits: 8, recommendedRerank: 20 });
        return fs.readFileSync(sp);
    };
    const ING = { chunker: 'v1' };
    const ENC = { kind: 'host-encoder-v1' };
    const BLOOM_BITS = 2048;
    const bloomB64 = (setBits) => {
        const b = new Uint8Array(BLOOM_BITS / 8);
        for (const i of setBits) b[i >> 3] |= 1 << (i & 7);
        return Buffer.from(b).toString('base64');
    };

    // A base whose calibration region carries a real `asset` — that is what
    // 5.5 means by "the base carries a fit". driftLimit on the asset is the
    // producer's declared validity envelope.
    function buildBase(driftLimit) {
        const ids = Array.from({ length: 100 }, (_, i) => i);
        const corpus = buildCorpusSegment(ids.map(recBytes));
        const cal = {
            kind: 'retrieval-signals-v1',
            asset: {
                features: ['d0'], weights: [1], bias: 0,
                standardize: { mean: { d0: 0.5 }, std: { d0: 0.1 } },
                thresholds: { hard: 0.4, weak: 0.6 },
                // The scorer reads the bloom geometry from asset.vocabBloom
                // (retrieval-abstention.mjs), not from the region root.
                vocabBloom: { bits: BLOOM_BITS, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'] },
                ...(driftLimit === undefined ? {} : { driftLimit }),
            },
            vocabBloomBase64: bloomB64([1, 2, 3]),
        };
        const p = path.join(calTmp, `base-${driftLimit ?? 'none'}.pikelet`);
        return assemblePikeletFile({
            profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC,
            corpus: { ...corpus.corpus, ingest: ING }, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, `b${driftLimit ?? 'n'}`) },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify(cal))) },
        ], p);
    }

    // A layer appending `n` records on top, shipping its own bloom at the
    // base's geometry so the union of 4.5 stays exact.
    function buildLayer(built, n, tag, withBloom = true) {
        const qiSha = built.manifest.segments.find((x) => x.kind === 'query-interp').sha256;
        const ids = Array.from({ length: n }, (_, i) => 100 + i);
        const corpus = buildCorpusSegment(ids.map(recBytes));
        const p = path.join(calTmp, `layer-${tag}.pikelet`);
        return assemblePikeletFile({
            profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC,
            layer: {
                parent: { identity: built.identity }, baseIdentity: built.identity,
                depth: 1, rowBase: 100, records: n, tombstones: 0, supersessions: 0, ingest: ING,
            },
            corpus: corpus.corpus, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, `l${tag}`) },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildInheritedQuerySegment({
                baseIdentity: built.identity, queryInterpSha256: qiSha,
                ...(withBloom ? {
                    vocabBloom: { bits: BLOOM_BITS, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'] },
                    vocabBloomBytes: Uint8Array.from(Buffer.from(bloomB64([4, 5]), 'base64')),
                } : {}),
            }) },
            { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: 100, tombstonedIds: [], records: n }) },
        ], p);
    }

    const { openPikeletChain } = await import('../packages/pikelet-wasm/complete/layer-reader.mjs');

    // 10 appended over a 100-record base = 0.10 drift, inside the 0.20 default.
    const b1 = buildBase(undefined);
    const l1 = buildLayer(b1, 10, 'small');
    const c1 = await openPikeletChain([b1.outPath, l1.outPath]);
    const i1 = c1.info();
    check('a base with a real fit and drift inside the limit is inherited',
        i1.calibrationStatus === 'inherited', `status ${i1.calibrationStatus}, drift ${i1.calibrationDrift}`);
    check('drift is (appended + tombstoned) / base records', Math.abs(i1.calibrationDrift - 0.1) < 1e-9);
    check('an absent producer envelope leaves the reader limit in force',
        i1.driftLimit === 0.2 && i1.producerEnvelope === Infinity);
    // 4.5: a chain's encoder is the base's. This was omitted from the chain's
    // info(), so every consumer reading info().encoder saw undefined -- the MCP
    // list_packs tool advertised "encoder": null for a pack whose base is
    // MiniLM, which is what a model reads to know what embedded the corpus.
    // Shape varies by query-interp kind (a kind-2 base's encoderInfo is shaped
    // by the host declaration, a kind-3's carries model/pooling), so this
    // asserts only that SOMETHING is reported and that it is the base's own.
    check('a chain reports the base\'s encoder, not nothing',
        !!i1.encoder && typeof i1.encoder === 'object'
        && Object.keys(i1.encoder).length > 0,
        JSON.stringify(i1.encoder));

    check('the union bloom carries base and layer bits', (() => {
        const u = c1.chainBloom();
        const set = (i) => ((u[i >> 3] >> (i & 7)) & 1) === 1;
        return u && set(1) && set(2) && set(3) && set(4) && set(5);
    })());
    await c1.close();

    // 5.5 k-mismatch: the fit reads mean10 over the top-10 window whatever
    // the caller's k. Each tier searched only min(4k, records) rows, so at
    // k=1 the scorer saw a handful of hits and a different mean10 than at
    // k=10: the same query got a different confidence depending on k.
    {
        const ids = Array.from({ length: 100 }, (_, i) => i);
        const corpus = buildCorpusSegment(ids.map(recBytes));
        const cal = {
            kind: 'retrieval-signals-v1',
            asset: {
                features: ['mean10'], weights: [-5], bias: 0,
                standardize: { mean: { mean10: 0.01 }, std: { mean10: 0.01 } },
                thresholds: { hard: 0.01, weak: 0.02 },
                vocabBloom: { bits: BLOOM_BITS, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'] },
            },
            vocabBloomBase64: bloomB64([1, 2, 3]),
        };
        const mb = assemblePikeletFile({
            profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC,
            corpus: { ...corpus.corpus, ingest: ING }, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, 'mean10') },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify(cal))) },
        ], path.join(calTmp, 'base-mean10.pikelet'));
        const ml = buildLayer(mb, 5, 'mean10');
        const mc = await openPikeletChain([mb.outPath, ml.outPath]);
        const at1 = await mc.query(ladderVector(50), 1, { showAbstained: true });
        const at10 = await mc.query(ladderVector(50), 10, { showAbstained: true });
        check('the abstention verdict does not depend on the caller\'s k',
            Number.isFinite(at1.confidence) && at1.confidence === at10.confidence
            && at1.matchQuality === at10.matchQuality,
            `k=1 ${at1.matchQuality}/${at1.confidence}, k=10 ${at10.matchQuality}/${at10.confidence}`);
        await mc.close();
    }

    // A chain returns the base scorer's grounding, as a single file does:
    // which passage grounded the verdict and which query words no passage
    // contains. The chain kept only matchQuality and confidence, so a layered
    // pack served over MCP lost grounding.uncovered, the fact check a
    // relevance verdict leaves to the reader.
    {
        const ids = Array.from({ length: 100 }, (_, i) => i);
        const corpus = buildCorpusSegment(ids.map(recBytes));
        const cal = {
            kind: 'retrieval-signals-v1',
            asset: {
                features: ['d0'], weights: [-1], bias: 0,
                standardize: { mean: { d0: 0.5 }, std: { d0: 0.1 } },
                thresholds: { hard: 0.01, weak: 0.02 },
                vocabBloom: { bits: BLOOM_BITS, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'] },
                coverage: { weight: 1, mean: 0.5, std: 0.2, topK: 5 },
            },
            vocabBloomBase64: bloomB64([1, 2, 3]),
        };
        const gb = assemblePikeletFile({
            profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC,
            corpus: { ...corpus.corpus, ingest: ING }, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, 'ground') },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify(cal))) },
        ], path.join(calTmp, 'base-grounding.pikelet'));
        const gl = buildLayer(gb, 5, 'grounding');
        const gc = await openPikeletChain([gb.outPath, gl.outPath]);
        check('the grounding chain inherits the fit', gc.info().calibrationStatus === 'inherited', gc.info().calibrationStatus);
        const out = await gc.query(ladderVector(50), 5, { text: 'record body zeppelin', showAbstained: true });
        const g = out.grounding;
        check('a chain query returns grounding from the inherited scorer',
            !!g && Array.isArray(g.uncovered) && Array.isArray(g.covered), JSON.stringify(g));
        check('grounding names the query words no passage contains',
            !!g && g.uncovered.includes('zeppelin') && g.covered.includes('record') && g.covered.includes('body'),
            JSON.stringify(g));
        check('grounding recordId is a chain-global id inside the chain',
            !!g && Number.isSafeInteger(g.recordId) && g.recordId >= 0 && g.recordId < 105, JSON.stringify(g));
        await gc.close();
    }

    // A chain applies the single-file reader's fetch tuning to every tier:
    // gap, parallelism and maxRangeBytes from the query or the mount, else
    // each tier's own default gap (its manifest hint as a floor under its
    // source's preference). The chain forwarded rerank and masks but dropped
    // these, so a layered pack ignored them.
    {
        const ids = Array.from({ length: 100 }, (_, i) => i);
        const corpus = buildCorpusSegment(ids.map(recBytes));
        const buildGapBase = (hint) => assemblePikeletFile({
            profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC,
            corpus: { ...corpus.corpus, ingest: ING }, index: {},
            ...(hint === undefined ? {} : { recommendedGap: hint }),
        }, [
            { kind: 'index', bytes: sketchFor(ids, 'gap') },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '' }))) },
        ], path.join(calTmp, `base-gap-${hint ?? 'none'}.pikelet`));
        // Reads one chain query costs over sources that prefer `prefer`, on a
        // base hinting `hint`; a fresh mount per measurement, so nothing is
        // cached.
        const readsFor = async ({ hint, prefer, queryOptions = {}, mountOptions = {} }) => {
            const base = buildGapBase(hint);
            // No bloom: this base carries no fit, and a layer bloom needs one
            // whose geometry it matches (4.5).
            const layer = buildLayer(base, 5, `gap-${hint ?? 'none'}`, false);
            let reads = 0;
            const src = (file) => {
                const bytes = fs.readFileSync(file);
                return {
                    size: bytes.length, preferredParallelism: Infinity, preferredGapBytes: prefer,
                    async read(offset, length) { reads++; return bytes.subarray(offset, offset + length); },
                    async close() {},
                };
            };
            const chain = await openPikeletChain([src(base.outPath), src(layer.outPath)], mountOptions);
            const before = reads;
            await chain.query(ladderVector(50), 5, { rerank: 60, showAbstained: true, ...queryOptions });
            const n = reads - before;
            await chain.close();
            return n;
        };
        const tight = await readsFor({ prefer: 0 });
        const wide = await readsFor({ prefer: 1 << 20 });
        check('on this chain a wide gap costs fewer reads than none (the probe can tell them apart)', wide < tight, `${wide} vs ${tight}`);
        check('a chain query\'s gap reaches the tiers', await readsFor({ prefer: 0, queryOptions: { gap: 1 << 20 } }) === wide);
        check('a mount-level rerankGap reaches the tiers', await readsFor({ prefer: 0, mountOptions: { rerankGap: 1 << 20 } }) === wide);
        check('a tier\'s manifest recommendedGap applies as its default', await readsFor({ hint: 1 << 20, prefer: 0 }) === wide);
        check('a query gap overrides the tier default', await readsFor({ hint: 1 << 20, prefer: 0, queryOptions: { gap: 0 } }) === tight);
        // maxRangeBytes splits a coalesced run: one row-block per request
        // costs more reads than the same wide gap uncapped.
        check('a chain query\'s maxRangeBytes reaches the tiers',
            await readsFor({ prefer: 1 << 20, queryOptions: { maxRangeBytes: 1 } }) > wide);
    }

    // The lazy lexical reader shares pending reads and caches postings: an
    // identical second query re-read both postings lists, and two query
    // terms in one uncached term-table window read it twice.
    {
        const { openLexicalIndexLazy } = await import('../packages/pikelet-wasm/complete/lexical.mjs');
        const texts = Array.from({ length: 40 }, (_, i) => `alpha bravo item${i} ${i % 2 ? 'charlie' : 'delta'}`);
        const built = buildLexSeg(texts);
        const seg = built.bytes ?? built;
        const log = [];
        const read = async (offset, length) => { log.push([offset, length]); return seg.subarray(offset, offset + length); };
        const lex = await openLexicalIndexLazy(read, seg.length);
        log.length = 0;
        const first = await lex.search('alpha bravo charlie', 5);
        const firstReads = log.length;
        log.length = 0;
        const second = await lex.search('alpha bravo charlie', 5);
        check('a repeated lazy lexical query reads nothing again', log.length === 0, `${log.length} reads (first query ${firstReads})`);
        check('the cached query scores identically',
            JSON.stringify(first) === JSON.stringify(second));
        // Fresh index: three terms whose entries share the one term-table
        // window must cost one window read, not three.
        const lex2 = await openLexicalIndexLazy(read, seg.length);
        log.length = 0;
        await lex2.search('alpha bravo charlie', 5);
        const v = new DataView(seg.buffer, seg.byteOffset, seg.byteLength);
        const termTableOffset = v.getUint32(24, true);
        const postingsOffset = v.getUint32(28, true);
        const windowReads = log.filter(([o]) => o >= termTableOffset && o < postingsOffset).length;
        check('concurrent terms share one pending term-table window read', windowReads === 1, `${windowReads} window reads`);
        // A failed read is not cached: the next query retries it.
        let fail = true;
        const flaky = async (offset, length) => {
            if (fail && offset >= postingsOffset) throw new Error('transient');
            return seg.subarray(offset, offset + length);
        };
        const lex3 = await openLexicalIndexLazy(flaky, seg.length);
        let threw = false;
        try { await lex3.search('alpha', 5); } catch { threw = true; }
        fail = false;
        const retried = await lex3.search('alpha', 5);
        check('a failed postings read is retried, not cached', threw && retried.length > 0);
    }

    // 5.3/5.4: the tombstone mask must reach the lexical index BEFORE its cap,
    // and the chain-global stats must actually drive scoring. Both were
    // computed and then dropped: hits were capped at 24 per tier and filtered
    // afterwards, so a tier whose top-24 were all tombstoned contributed
    // nothing, and info().lexical reported stats no query used.
    {
        const { openLexicalIndex } = await import('../packages/pikelet-wasm/complete/lexical.mjs');
        // 30 docs all matching 'widget'; the first five score highest.
        const texts = [];
        for (let i = 0; i < 30; i++) {
            texts.push(i < 5 ? 'widget widget widget widget' : 'widget filler filler filler filler filler');
        }
        const built = buildLexSeg(texts);
        const lex = openLexicalIndex(built.bytes ?? built);
        const top5 = lex.search('widget', 5).map((h) => h.id);
        const dead = new Set(top5);
        const refilled = lex.search('widget', 5, { exclude: (id) => dead.has(id) });
        check('excluding the top hits refills the cap from live rows',
            refilled.length === 5 && refilled.every((h) => !dead.has(h.id)),
            `got ${refilled.length} hits: ${refilled.map((h) => h.id).join(',')} (capping before the mask returns 0)`);
        const own = lex.search('widget', 3);
        const global = lex.search('widget', 3, { stats: { docCount: 100000, avgdl: 50 } });
        check('chain-global docCount/avgdl change the BM25 score',
            own[0].score !== global[0].score,
            `own ${own[0].score}, global ${global[0].score} — equal means stats were ignored`);
    }

    // 4.5's normative MUST: the union has to reach the SCORER, not merely be
    // computed. This asserted only that chainBloom() held the right bits,
    // which stayed true for the whole time the reader built the union and then
    // scored against the base's bloom alone. Assert the seam itself: an
    // override changes known_frac, and a wrong-sized one is refused rather
    // than silently probing the wrong bit positions.
    {
        const { createAbstentionScorer } = await import('../packages/pikelet-wasm/complete/retrieval-abstention.mjs');
        const BITS = 1024;
        const fnv = (str, seed) => {
            let h = 0x811c9dc5 ^ seed;
            for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
            return (h >>> 0) % BITS;
        };
        const bloomFor = (words) => {
            const b = new Uint8Array(BITS / 8);
            for (const w of words) for (const seed of [0, 0x9e3779b9]) { const bit = fnv(w, seed); b[bit >> 3] |= 1 << (bit & 7); }
            return b;
        };
        // known_frac only, with the shipped model's NEGATIVE sign, so a
        // vocabulary the chain knows pushes toward abstention.
        const asset = {
            bias: 0, features: ['d0', 'margin', 'mean10', 'known_frac'], weights: [0, 0, 0, -2],
            standardize: {
                mean: { d0: 0, margin: 0, mean10: 0, known_frac: 0 },
                std: { d0: 1, margin: 1, mean10: 1, known_frac: 1 },
            },
            thresholds: { hard: 0.1, weak: 0.4 }, vocabBloom: { bits: BITS },
        };
        const baseBloom = bloomFor(['alpha', 'beta']);
        const layerBloom = bloomFor(['gamma', 'delta']);
        const union = new Uint8Array(baseBloom);
        for (let i = 0; i < union.length; i++) union[i] |= layerBloom[i];

        const scorer = createAbstentionScorer(asset, baseBloom);
        const hits = [{ distance: 0.2 }, { distance: 0.3 }, { distance: 0.4 }];
        // A query whose words live only in the LAYER's vocabulary.
        const baseOnly = await scorer.score('gamma delta', hits, []);
        const withUnion = await scorer.score('gamma delta', hits, [], { vocabBloom: union });
        check('the union bloom changes known_frac when passed to the scorer',
            baseOnly.p !== withUnion.p,
            `base-only p ${baseOnly.p}, union p ${withUnion.p} — equal means the override was ignored`);
        check('scoring against the base bloom alone over-answers a layer-vocabulary query',
            baseOnly.p > withUnion.p,
            `base-only p ${baseOnly.p} should exceed union p ${withUnion.p} at a negative known_frac weight`);
        let refused = null;
        try { await scorer.score('gamma delta', hits, [], { vocabBloom: new Uint8Array(union.length + 1) }); }
        catch (err) { refused = err.message; }
        check('a bloom override of the wrong geometry is refused',
            refused !== null && /probe the wrong bits/.test(refused), String(refused));
    }

    // 30 appended = 0.30 drift, past the reader's 0.20 default.
    const l2 = buildLayer(b1, 30, 'big');
    const c2 = await openPikeletChain([b1.outPath, l2.outPath]);
    check('drift past the reader limit is drift-exceeded',
        c2.info().calibrationStatus === 'drift-exceeded');
    await c2.close();

    // A host may be stricter than its own default.
    const c3 = await openPikeletChain([b1.outPath, l1.outPath], { readerDriftLimit: 0.05 });
    check('a stricter host limit lowers the effective limit',
        c3.info().driftLimit === 0.05 && c3.info().calibrationStatus === 'drift-exceeded');
    await c3.close();

    // A producer envelope stricter than the host's limit wins; a looser one
    // cannot raise it (5.5's fixed precedence).
    const bStrict = buildBase(0.05);
    const lStrict = buildLayer(bStrict, 10, 'strict');
    const c4 = await openPikeletChain([bStrict.outPath, lStrict.outPath]);
    check('a stricter producer envelope lowers the effective limit',
        c4.info().driftLimit === 0.05 && c4.info().calibrationStatus === 'drift-exceeded');
    await c4.close();

    const bLoose = buildBase(0.9);
    const lLoose = buildLayer(bLoose, 30, 'loose');
    const c5 = await openPikeletChain([bLoose.outPath, lLoose.outPath]);
    check('a looser producer envelope cannot raise the host\'s limit',
        c5.info().driftLimit === 0.2 && c5.info().calibrationStatus === 'drift-exceeded');
    await c5.close();

    // 4.5/5.5: a layer shipping no bloom while the base carries a fit degrades
    // the chain, even though drift is well inside the limit.
    const lNoBloom = buildLayer(b1, 10, 'nobloom', false);
    const c6 = await openPikeletChain([b1.outPath, lNoBloom.outPath]);
    check('a layer shipping no bloom degrades a scored chain',
        c6.info().calibrationStatus === 'drift-exceeded', `status ${c6.info().calibrationStatus}`);
    await c6.close();

    // --- Read-side MUSTs a producer used to be the only one to check -------
    // A layer over `built` with full control of its bloom, lexical segment
    // and tombstones, so each rule can be broken on its own.
    const GEOM = { bits: BLOOM_BITS, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'] };
    function craftLayer(built, tag, { n = 10, bloom = GEOM, lexical = false, tombstonedIds = [], parent = built, depth = 1, rowBase = 100 } = {}) {
        const qiSha = built.manifest.segments.find((x) => x.kind === 'query-interp').sha256;
        const ids = Array.from({ length: n }, (_, i) => rowBase + i);
        const segs = [];
        if (n) {
            const corpus = buildCorpusSegment(ids.map(recBytes));
            segs.push({ kind: 'index', bytes: sketchFor(ids, `x${tag}`) }, { kind: 'corpus', bytes: corpus.bytes });
            if (lexical) segs.push({ kind: 'lexical', bytes: buildLexSeg(ids.map((i) => `record ${i} body text`)).bytes });
            segs.corpusMeta = corpus.corpus;
        }
        segs.push({ kind: 'query-interp', bytes: buildInheritedQuerySegment({
            baseIdentity: built.identity, queryInterpSha256: qiSha,
            ...(bloom && n ? {
                vocabBloom: bloom,
                vocabBloomBytes: Uint8Array.from(Buffer.from(bloomB64([7]), 'base64')),
            } : {}),
        }) });
        segs.push({ kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase, tombstonedIds, records: n }) });
        const order = ['index', 'corpus', 'query-interp', 'lexical', 'tombstones'];
        return assemblePikeletFile({
            profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC,
            layer: {
                parent: { identity: parent.identity }, baseIdentity: built.identity,
                depth, rowBase, records: n, tombstones: tombstonedIds.length, supersessions: 0, ingest: ING,
            },
            corpus: n ? segs.corpusMeta : { records: 0 }, ...(n ? { index: {} } : {}),
        }, segs.slice().sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind)), path.join(calTmp, `craft-${tag}.pikelet`));
    }

    // 4.5: the bloom geometry is checked against the base's fit. It came only
    // from options.baseBloom, which no caller set, so a layer with the right
    // size but foreign hashes was OR'd into the union as if exact.
    const okBloom = craftLayer(b1, 'okbloom');
    const cOk = await openPikeletChain([b1.outPath, okBloom.outPath]);
    check('a layer bloom at the base\'s geometry mounts', cOk.info().calibrationStatus === 'inherited');
    await cOk.close();
    const evil = craftLayer(b1, 'evilhash', { bloom: { bits: BLOOM_BITS, hashes: ['evil'] } });
    await rejectsAsync('a layer bloom with foreign hashes is refused, not OR\'d in',
        () => openPikeletChain([b1.outPath, evil.outPath]), /bloom hashes .* differ from the base's/);
    const noFitBase = (() => {
        const ids = Array.from({ length: 100 }, (_, i) => i);
        const corpus = buildCorpusSegment(ids.map(recBytes));
        return assemblePikeletFile({
            profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC,
            corpus: { ...corpus.corpus, ingest: ING }, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, 'nofit') },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
        ], path.join(calTmp, 'base-nofit.pikelet'));
    })();
    const orphanBloom = craftLayer(noFitBase, 'orphanbloom');
    await rejectsAsync('a layer bloom over a base with no bloom to match is refused',
        () => openPikeletChain([noFitBase.outPath, orphanBloom.outPath]), /base carries none whose geometry it could match/);

    // 4.3: a layer must neither drop nor introduce lexical retrieval.
    const lexBase = (() => {
        const ids = Array.from({ length: 100 }, (_, i) => i);
        const corpus = buildCorpusSegment(ids.map(recBytes));
        return assemblePikeletFile({
            profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC,
            corpus: { ...corpus.corpus, ingest: ING }, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, 'lexbase') },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
            { kind: 'lexical', bytes: buildLexSeg(ids.map((i) => `record ${i} body text`)).bytes },
        ], path.join(calTmp, 'base-lex.pikelet'));
    })();
    const lexKept = craftLayer(lexBase, 'lexkept', { bloom: null, lexical: true });
    const cLex = await openPikeletChain([lexBase.outPath, lexKept.outPath]);
    check('a layer carrying lexical over a hybrid base mounts', cLex.info().layers === 2);
    await cLex.close();
    const lexDropped = craftLayer(lexBase, 'lexdropped', { bloom: null, lexical: false });
    await rejectsAsync('a layer that drops lexical retrieval from a hybrid base is refused',
        () => openPikeletChain([lexBase.outPath, lexDropped.outPath]), /must not drop lexical retrieval/);
    const lexAdded = craftLayer(noFitBase, 'lexadded', { bloom: null, lexical: true });
    await rejectsAsync('a layer that introduces lexical retrieval is refused',
        () => openPikeletChain([noFitBase.outPath, lexAdded.outPath]), /must not introduce lexical retrieval/);

    // 4.3: a tombstone-only layer must delete something its parent had not.
    const withDel = craftLayer(b1, 'withdel', { tombstonedIds: [5] });
    const noop = craftLayer(b1, 'noop', { n: 0, bloom: null, tombstonedIds: [5], parent: withDel, depth: 2, rowBase: 110 });
    await rejectsAsync('a tombstone-only layer that deletes nothing new is refused',
        () => openPikeletChain([b1.outPath, withDel.outPath, noop.outPath]), /deletes nothing its parent had not already deleted/);
    const realDel = craftLayer(b1, 'realdel', { n: 0, bloom: null, tombstonedIds: [5, 6], parent: withDel, depth: 2, rowBase: 110 });
    const cDel = await openPikeletChain([b1.outPath, withDel.outPath, realDel.outPath]);
    check('a tombstone-only layer that adds a deletion mounts', cDel.info().tombstones === 2);
    // 4.5/5.5: a tombstone-only layer has no vocabulary, so its `none` region
    // is the empty bloom, not a gap. Counting it as one unscored a real chain
    // that deleted 3 of 3,255 records at drift 0.0009.
    check('a tombstone-only layer shipping no bloom keeps the chain scored',
        cDel.info().calibrationStatus === 'inherited', `status ${cDel.info().calibrationStatus}`);
    await cDel.close();
    const tombFirst = craftLayer(b1, 'tombfirst', { n: 0, bloom: null, tombstonedIds: [7] });
    const cTomb = await openPikeletChain([b1.outPath, tombFirst.outPath]);
    check('a lone tombstone-only layer over a scored base stays inherited',
        cTomb.info().calibrationStatus === 'inherited', `status ${cTomb.info().calibrationStatus}`);
    await cTomb.close();

    fs.rmSync(calTmp, { recursive: true, force: true });
}

// ---------------------------------------------------------------------------
// append (6.1)
// ---------------------------------------------------------------------------
const {
    planAppendability, planLayer, buildLayerSegments, checkAppendDrift, idsSetIn,
    PRODUCER_MAX_DEPTH, FORMAT_MAX_DEPTH,
} = await import('../packages/pikelet-wasm/complete/layer-append.mjs');

console.log('append: appendability and the encoder source (6.1)');
{
    const ING = { chunker: 'v1', targetTokens: 256 };
    const k3 = planAppendability({ qiKind: 3, corpusIngest: ING });
    check('a kind-3 base is appendable from its own pack', k3.encoderSource === 'pack' && k3.appendable === true);
    const k2 = planAppendability({ qiKind: 2, corpusIngest: ING });
    check('a kind-2 base appends with the host\'s encoder', k2.encoderSource === 'host');

    // The rule with a normative answer for every base ever released.
    rejects('a kind-1 base is readable but not appendable',
        () => planAppendability({ qiKind: 1, corpusIngest: ING }), /readable, but not appendable/);
    rejects('a layer is not a base to append to',
        () => planAppendability({ qiKind: 4, corpusIngest: ING }), /a layer is not a base/);

    // The ingestion declaration: granularity is part of what calibration saw.
    rejects('a base with no corpus.ingest refuses without an assertion',
        () => planAppendability({ qiKind: 3, corpusIngest: null }), /supply --assert-ingest/);
    const asserted = planAppendability({ qiKind: 3, corpusIngest: null, assertIngest: ING });
    check('an operator may assert the declaration',
        asserted.ingestAsserted === true && canonicalJson(asserted.ingest) === canonicalJson(ING));
    rejects('an assertion cannot override an artifact\'s own declaration',
        () => planAppendability({ qiKind: 3, corpusIngest: ING, assertIngest: ING }), /would override an artifact/);
    rejects('the asserted declaration must not smuggle ingestAsserted inside it',
        () => planAppendability({ qiKind: 3, corpusIngest: null, assertIngest: { ...ING, ingestAsserted: true } }),
        /must not contain ingestAsserted/);
}

console.log('append: id assignment, mask and supersessions (6.1 steps 4-5)');
{
    const parentBitset = parseTombstoneSegment(
        buildTombstoneSegment({ rowBase: 100, tombstonedIds: [5, 6] }),
    ).bitset;
    const common = { parentRowBase: 0, parentRecords: 100, parentBitset, parentDepth: 0 };

    const p = planLayer({ ...common, newRecordCount: 10, remove: [7, 8] });
    check('rowBase is parent rowBase + parent records', p.rowBase === 100);
    check('depth is parent depth + 1', p.depth === 1);
    check('the mask inherits the parent\'s bits', testBit(p.bitset, 5) && testBit(p.bitset, 6));
    check('and adds the layer\'s own removals', testBit(p.bitset, 7) && testBit(p.bitset, 8));
    check('tombstones is the cumulative popcount', p.tombstones === 4);
    check('the delta is only what this layer introduced',
        JSON.stringify(p.delta) === JSON.stringify([7, 8]));

    // A --remove of an already-deleted id is a no-op, not an error.
    const noop = planLayer({ ...common, newRecordCount: 1, remove: [5] });
    check('removing an already-deleted id is a no-op', noop.delta.length === 0 && noop.tombstones === 2);

    // Supersessions take ids from this layer's own range, in order.
    const sup = planLayer({ ...common, newRecordCount: 3, supersede: [[10, 'a.md'], [20, 'b.md']] });
    check('supersessions assign newIds from this layer\'s range',
        JSON.stringify(sup.supersessions) === JSON.stringify([[10, 100], [20, 101]]));
    check('a superseded oldId is tombstoned by the same step',
        testBit(sup.bitset, 10) && testBit(sup.bitset, 20));

    // A resolved numeric target binds oldId to a SPECIFIC new record. Before
    // this, every pair was assigned positionally and the second element of the
    // pair -- the --supersede path -- was destructured away, so which file
    // replaced which record was decided by argument order.
    const supResolved = planLayer({ ...common, newRecordCount: 3, supersede: [[10, 2], [20, 0]] });
    check('a resolved supersede target binds oldId to that new record',
        JSON.stringify(supResolved.supersessions) === JSON.stringify([[10, 102], [20, 100]]),
        JSON.stringify(supResolved.supersessions));
    const supMixed = planLayer({ ...common, newRecordCount: 3, supersede: [[10, 1], [20, 'b.md']] });
    check('an unresolved target falls back to a slot the resolved ones did not claim',
        JSON.stringify(supMixed.supersessions) === JSON.stringify([[10, 101], [20, 100]]),
        JSON.stringify(supMixed.supersessions));
    rejects('a resolved target outside the layer\'s own records is refused',
        () => planLayer({ ...common, newRecordCount: 2, supersede: [[10, 5]] }),
        /outside this layer's 2 record\(s\)/);
    rejects('two old ids resolving onto one new record is refused',
        () => planLayer({ ...common, newRecordCount: 3, supersede: [[10, 1], [20, 1]] }),
        /each new record supersedes at most one predecessor/);

    // --supersede path resolution, through the real resolver. The checks below
    // drive planLayer directly with already-resolved indices, which is how a
    // resolver bug survived them: a chunk's sourcePath is RELATIVE to its
    // ingest root, and comparing it against a cwd-resolved target matched
    // nothing, so `--supersede 2=/abs/dir/page.md` failed on a path that
    // plainly existed. Exercised here against the shapes a caller actually
    // types.
    {
        const { resolveSupersedeTargets } = await import('../packages/pikelet/src/append.mjs');
        const root = path.join(os.tmpdir(), 'pikelet-sup-root');
        const chunks = [
            { sourcePath: 'install.md' },
            { sourcePath: 'pricing.md' },
            { sourcePath: 'guide/setup.md' },
        ];
        const sources = [root];
        // Caught, not thrown: a resolver that cannot match one of these forms
        // raises rather than returning, and a bare call would abort the run
        // instead of reporting which form it failed on.
        const one = (target) => {
            try { return resolveSupersedeTargets([[2, target]], chunks, sources)[0][1]; }
            catch (err) { return `threw: ${err.message.slice(0, 60)}`; }
        };
        check('a bare relative path resolves', one('pricing.md') === 1, String(one('pricing.md')));
        check('an absolute path under a --source root resolves',
            one(path.join(root, 'pricing.md')) === 1, String(one(path.join(root, 'pricing.md'))));
        check('a nested relative path resolves', one('guide/setup.md') === 2, String(one('guide/setup.md')));
        check('a nested absolute path resolves',
            one(path.join(root, 'guide', 'setup.md')) === 2, String(one(path.join(root, 'guide', 'setup.md'))));
        rejects('a path no source ingested is refused',
            () => resolveSupersedeTargets([[2, 'nosuch.md']], chunks, sources),
            /matched no ingested record/);
        rejects('--supersede without a --source is refused',
            () => resolveSupersedeTargets([[2, 'pricing.md']], chunks, []),
            /needs a --source/);
        // Two roots each holding the same filename: basename matching would
        // silently pick one, which is the wrong edge this refuses to guess.
        const rootB = path.join(os.tmpdir(), 'pikelet-sup-root-b');
        rejects('an ambiguous path across two source roots is refused, not guessed',
            () => resolveSupersedeTargets([[2, 'pricing.md']],
                [{ sourcePath: 'pricing.md' }, { sourcePath: 'pricing.md' }], [root, rootB]),
            /matched 2 records/);
    }

    rejects('removing an id at or above rowBase is refused',
        () => planLayer({ ...common, newRecordCount: 5, remove: [100] }),
        /cannot retract its own records/);
    rejects('more supersessions than new records is refused',
        () => planLayer({ ...common, newRecordCount: 1, supersede: [[10, 'a'], [20, 'b']] }),
        /more --supersede pairs than new records/);
    rejects('the same oldId twice in one layer is refused',
        () => planLayer({ ...common, newRecordCount: 2, supersede: [[10, 'a'], [10, 'b']] }),
        /more than once/);

    // "A layer must do something" (6.1).
    rejects('a layer introducing neither a record nor a new tombstone is refused',
        () => planLayer({ ...common, newRecordCount: 0, remove: [5, 6] }),
        /neither a record nor a new tombstone/);
    const tombOnly = planLayer({ ...common, newRecordCount: 0, remove: [9] });
    check('a tombstone-only layer that does delete something is allowed',
        tombOnly.tombstoneOnly === true && tombOnly.delta.length === 1);

    // Depth limits: the producer default and the format ceiling.
    rejects('appending past the producer default is refused',
        () => planLayer({ ...common, parentDepth: PRODUCER_MAX_DEPTH, newRecordCount: 1 }),
        new RegExp(`past --max-depth ${PRODUCER_MAX_DEPTH}`));
    check('a higher --max-depth allows it',
        planLayer({ ...common, parentDepth: PRODUCER_MAX_DEPTH, newRecordCount: 1, maxDepth: 12 }).depth === 9);
    rejects('the format ceiling is enforced regardless of --max-depth',
        () => planLayer({ ...common, parentDepth: FORMAT_MAX_DEPTH, newRecordCount: 1, maxDepth: 99 }),
        /past the format ceiling of 16/);

    // 3.3: the id space is finite and only compaction reclaims it.
    rejects('an append that would exhaust the id space is refused',
        () => planLayer({
            parentRowBase: MAX_RECORDS - 5, parentRecords: 5, parentBitset: new Uint8Array(0),
            parentDepth: 0, newRecordCount: 10,
        }), /would exhaust the id space/);
}

console.log('append: the layer\'s segments and manifest (6.1 step 6)');
{
    const parentBitset = parseTombstoneSegment(buildTombstoneSegment({ rowBase: 50, tombstonedIds: [1] })).bitset;
    const plan = planLayer({
        parentRowBase: 0, parentRecords: 50, parentBitset, parentDepth: 0,
        newRecordCount: 4, remove: [2], supersede: [[3, 'x.md']],
    });
    const ING = { chunker: 'v1' };
    const built = buildLayerSegments({
        plan, baseIdentity: BASE_ID, parentIdentity: BASE_ID,
        baseQueryInterpSha256: '7'.repeat(64), parentLocator: 'base.pikelet',
        ingest: ING, ingestAsserted: false, dim: 384, metric: 'cosine',
        encoder: { kind: 'inline-transformer-v1' },
        corpus: { records: 4, layout: 'records-v2' },
    });
    const v = validateLayerObject(built.manifestFields);
    check('the assembled manifest passes layer validation', v.depth === 1 && v.rowBase === 50);
    check('the tombstone segment round-trips with the manifest counts', (() => {
        const t = parseTombstoneSegment(built.tombstones, {
            rowBase: 50, tombstones: built.manifestFields.layer.tombstones,
            supersessions: 1, records: 4,
        });
        return t.tombstoneCount === 3 && JSON.stringify(t.supersessions) === JSON.stringify([[3, 50]]);
    })());
    check('the locator is recorded when supplied', built.manifestFields.layer.parent.locator === 'base.pikelet');
    check('ingestAsserted is absent when the base declared the ingest',
        built.manifestFields.layer.ingestAsserted === undefined);

    // An asserted chain records the sibling flag, and the declaration's
    // canonical form is unchanged by the assertion (4.1).
    const assertedBuilt = buildLayerSegments({
        plan, baseIdentity: BASE_ID, parentIdentity: BASE_ID,
        baseQueryInterpSha256: '7'.repeat(64), ingest: ING, ingestAsserted: true,
        dim: 384, metric: 'cosine', encoder: { kind: 'inline-transformer-v1' },
        corpus: { records: 4, layout: 'records-v2' },
    });
    check('an asserted chain records the sibling flag',
        assertedBuilt.manifestFields.layer.ingestAsserted === true);
    check('the declaration\'s canonical form is identical either way',
        canonicalJson(built.manifestFields.layer.ingest) === canonicalJson(assertedBuilt.manifestFields.layer.ingest));

    // A tombstone-only layer's manifest omits the layout fields.
    const tPlan = planLayer({ parentRowBase: 0, parentRecords: 50, parentBitset, parentDepth: 0, newRecordCount: 0, remove: [9] });
    const tBuilt = buildLayerSegments({
        plan: tPlan, baseIdentity: BASE_ID, parentIdentity: BASE_ID,
        baseQueryInterpSha256: '7'.repeat(64), ingest: ING, ingestAsserted: false,
        dim: 384, metric: 'cosine', encoder: { kind: 'inline-transformer-v1' },
        corpus: { records: 4, layout: 'records-v2' },
    });
    check('a tombstone-only layer omits the corpus layout fields',
        tBuilt.manifestFields.corpus.records === 0 && tBuilt.manifestFields.corpus.layout === undefined);
    check('and validates as tombstone-only', validateLayerObject(tBuilt.manifestFields).tombstoneOnly === true);
}

console.log('append: drift refusal (6.1, 5.5)');
{
    const base = { baseRecords: 1000, appendedBefore: 100, headTombstones: 50, effectiveLimit: 0.2 };
    const ok = checkAppendDrift({ ...base, newRecords: 10 });
    check('an append inside the limit is allowed', ok.exceedsLimit === false && Math.abs(ok.drift - 0.16) < 1e-9);
    rejects('an append past the limit is refused',
        () => checkAppendDrift({ ...base, newRecords: 100 }), /past the chain's effective limit/);
    const forced = checkAppendDrift({ ...base, newRecords: 100, allowDrift: true });
    check('--allow-drift emits it anyway and reports that it exceeds',
        forced.exceedsLimit === true && Math.abs(forced.drift - 0.25) < 1e-9);
}

// ---------------------------------------------------------------------------
// rebase (6.2) — section 10 fixture family (D)
// ---------------------------------------------------------------------------
const {
    forkPoint, checkRebasePreconditions, planRebase, ON_CONFLICT, ON_FOREIGN,
} = await import('../packages/pikelet-wasm/complete/layer-rebase.mjs');

const maskFor = (rowBase, ids) => parseTombstoneSegment(
    buildTombstoneSegment({ rowBase, tombstonedIds: ids }),
).bitset;

console.log('rebase: THE DELTA CASE — a rebase replays what the layer did (6.2)');
{
    // Section 10 (D): on branch A, A1 tombstones base record 5 (once plainly,
    // once with a supersession); then A2 MERELY APPENDS. Rebasing A2 onto
    // branch B must NOT set bit 5 and must NOT carry A1's edge: bit 5 is in
    // A2's CUMULATIVE bitset only because it inherited it, and replaying the
    // cumulative set would import an ancestor's operation into a branch that
    // never performed it. Draft 2 replayed the cumulative set; this is the
    // correction.
    const base = { identity: id64(1), records: 100 };
    const a1 = { identity: id64(2), records: 10 };
    const b1 = { identity: id64(3), records: 7 };

    // A1 tombstoned base id 5. A2 inherits that and appends 4 records.
    const a1Bitset = maskFor(100, [5]);
    const a2Bitset = maskFor(110, [5]);        // cumulative: inherited only
    const headBitset = maskFor(107, []);       // branch B deleted nothing

    const { forkRowBase } = forkPoint([base, a1], [base, b1]);
    check('the fork point is the shared base only', forkRowBase === 100);

    const plan = planRebase({
        layerBitset: a2Bitset, oldParentBitset: a1Bitset, oldRowBase: 110,
        layerRecords: 4, layerSupersessions: [],
        newRowBase: 107, headBitset, forkRowBase,
    });
    check('the rebase emits (it owns records)', plan.emit === true);
    check('its delta over its own parent is empty', plan.delta.length === 0);
    check('IT DOES NOT SET BIT 5: that was its ancestor\'s deletion, not its own',
        !testBit(plan.bitset, 5), 'bit 5 leaked from the cumulative set');
    check('it carries no supersession it did not record', plan.supersessions.length === 0);
    check('its rowBase is the new head\'s row total', plan.newRowBase === 107);

    // The same layer whose OWN segment recorded the deletion does carry it.
    const a2Own = planRebase({
        layerBitset: maskFor(110, [5, 9]), oldParentBitset: a1Bitset, oldRowBase: 110,
        layerRecords: 4, layerSupersessions: [], newRowBase: 107, headBitset, forkRowBase,
    });
    check('a deletion the layer itself introduced IS replayed',
        testBit(a2Own.bitset, 9) && a2Own.delta.length === 1 && a2Own.delta[0] === 9);
    check('and the inherited bit still is not', !testBit(a2Own.bitset, 5));

    rejects('a rebase with no access to the original parent refuses rather than guessing',
        () => planRebase({
            layerBitset: a2Bitset, oldParentBitset: null, oldRowBase: 110,
            layerRecords: 4, newRowBase: 107, headBitset, forkRowBase,
        }), /delta over that parent is unknowable/);
}

console.log('rebase: two kinds of id (6.2)');
{
    const base = { identity: id64(1), records: 100 };
    const a1 = { identity: id64(2), records: 10 };
    const { forkRowBase } = forkPoint([base, a1], [base, { identity: id64(3), records: 5 }]);

    // A layer owning ids 110..113 with an edge (7 -> 111): the y side is its
    // own and must be translated; the x side is an ancestor id and is not.
    const plan = planRebase({
        layerBitset: maskFor(110, [7]), oldParentBitset: maskFor(110, []), oldRowBase: 110,
        layerRecords: 4, layerSupersessions: [[7, 111]],
        newRowBase: 105, headBitset: maskFor(105, []), forkRowBase,
    });
    check('the y side is translated into the new range',
        JSON.stringify(plan.supersessions) === JSON.stringify([[7, 106]]),
        JSON.stringify(plan.supersessions));
    check('the x side is an ancestor id and is unchanged', plan.supersessions[0][0] === 7);
    check('the superseded id is tombstoned in the rebased bitset', testBit(plan.bitset, 7));

    rejects('a translated newId outside the new range is refused',
        () => planRebase({
            layerBitset: maskFor(110, [7]), oldParentBitset: maskFor(110, []), oldRowBase: 110,
            layerRecords: 1, layerSupersessions: [[7, 118]],
            newRowBase: 105, headBitset: maskFor(105, []), forkRowBase,
        }), /falls outside the layer's new range/);
}

console.log('rebase: foreign ids above the fork point (6.2)');
{
    // Two depth-1 forks of the same base assign OVERLAPPING id ranges to
    // different records, so an id at or above the fork point names a different
    // record in the two histories.
    const base = { identity: id64(1), records: 100 };
    const a1 = { identity: id64(2), records: 10 };
    const a2 = { identity: id64(4), records: 5 };
    const b1 = { identity: id64(3), records: 8 };
    const { forkRowBase } = forkPoint([base, a1, a2], [base, b1]);
    check('the fork point excludes both branches\' own layers', forkRowBase === 100);

    // A3 deleted id 105 — an id A1 owned, which B's history never had.
    const spec = {
        layerBitset: maskFor(115, [105]), oldParentBitset: maskFor(115, []), oldRowBase: 115,
        layerRecords: 2, layerSupersessions: [], newRowBase: 108,
        headBitset: maskFor(108, []), forkRowBase,
    };
    rejects('a foreign deletion refuses by default, naming the id',
        () => planRebase(spec), /foreign id\(s\) at or above the fork point 100.*105/);
    const dropped = planRebase({ ...spec, onForeign: ON_FOREIGN.DROP });
    check('--on-foreign drop discards it', dropped.emit === true && dropped.droppedForeign.deletions[0] === 105);
    check('and sets no bit for it', !testBit(dropped.bitset, 105));

    // The common case: H merely grew from L's own parent, so there are no
    // foreign ids at all.
    const grown = forkPoint([base, a1], [base, a1, { identity: id64(9), records: 3 }]);
    check('a head that merely grew has forkRowBase = the layer\'s old rowBase',
        grown.forkRowBase === 110);
}

console.log('rebase: deletion and supersession conflicts (6.2)');
{
    const base = { identity: id64(1), records: 100 };
    const shared = [base];
    const { forkRowBase } = forkPoint(shared, shared);
    const common = {
        oldParentBitset: maskFor(100, []), oldRowBase: 100,
        layerRecords: 3, newRowBase: 100, forkRowBase,
    };

    // Deletion already performed by H: a no-op, not a conflict. Idempotent.
    const idem = planRebase({
        ...common, layerBitset: maskFor(100, [5]), headBitset: maskFor(100, [5]),
    });
    check('a deletion H already performed is a no-op, not a conflict',
        idem.emit === true && idem.newlySet.length === 0 && testBit(idem.bitset, 5));

    // (x -> y) where H tombstoned x but recorded no edge: L adds information.
    const addsInfo = planRebase({
        ...common, layerBitset: maskFor(100, [5]), layerSupersessions: [[5, 100]],
        headBitset: maskFor(100, [5]), headSupersessions: new Map(),
    });
    check('an edge for an id H tombstoned without one is recorded',
        JSON.stringify(addsInfo.supersessions) === JSON.stringify([[5, 100]]));

    // H records a DIFFERENT successor: a conflict.
    const headSup = new Map([[5, [{ depth: 1, newId: 42 }]]]);
    const conflictSpec = {
        ...common, layerBitset: maskFor(100, [5]), layerSupersessions: [[5, 100]],
        headBitset: maskFor(100, [5]), headSupersessions: headSup,
    };
    rejects('a different successor in H is a conflict, refused by default',
        () => planRebase(conflictSpec), /supersession conflict\(s\).*5/);
    const skipped = planRebase({ ...conflictSpec, onConflict: ON_CONFLICT.SKIP });
    check('--on-conflict skip drops L\'s edge and keeps its record',
        skipped.supersessions.length === 0 && skipped.skippedEdges[0] === 5 && skipped.records === 3);
    const both = planRebase({ ...conflictSpec, onConflict: ON_CONFLICT.KEEP_BOTH });
    check('--on-conflict keep-both records L\'s edge as an additional one',
        JSON.stringify(both.supersessions) === JSON.stringify([[5, 100]]));
}

console.log('rebase: the terminal empty rule (6.2)');
{
    const base = { identity: id64(1), records: 100 };
    const { forkRowBase } = forkPoint([base], [base]);
    // A tombstone-only layer whose every deletion H has independently performed.
    const empty = planRebase({
        layerBitset: maskFor(100, [5, 6]), oldParentBitset: maskFor(100, []), oldRowBase: 100,
        layerRecords: 0, newRowBase: 100, headBitset: maskFor(100, [5, 6]), forkRowBase,
    });
    check('an empty rebase emits nothing', empty.emit === false);
    check('and says why', /already represents/.test(empty.reason));

    // One deletion H lacks: it emits.
    const notEmpty = planRebase({
        layerBitset: maskFor(100, [5, 6]), oldParentBitset: maskFor(100, []), oldRowBase: 100,
        layerRecords: 0, newRowBase: 100, headBitset: maskFor(100, [5]), forkRowBase,
    });
    check('a tombstone-only rebase that adds a deletion does emit',
        notEmpty.emit === true && notEmpty.newlySet.length === 1 && notEmpty.newlySet[0] === 6);

    // A layer with records always emits, whatever became of its deletions.
    const withRecords = planRebase({
        layerBitset: maskFor(100, [5]), oldParentBitset: maskFor(100, []), oldRowBase: 100,
        layerRecords: 2, newRowBase: 100, headBitset: maskFor(100, [5]), forkRowBase,
    });
    check('a rebase with records always emits', withRecords.emit === true);
}

console.log('rebase: preconditions and evaluation material (6.2)');
{
    const ING = { chunker: 'v1' };
    const lm = { layer: { ingest: ING, depth: 1 } };
    const same = { canonical: canonicalJson(ING), asserted: false };
    const ok = checkRebasePreconditions({
        layerManifest: lm, headChainDecl: same, layerChainDecl: same,
        headBaseIdentity: BASE_ID, layerBaseIdentity: BASE_ID,
    });
    check('matching declarations and bases pass', canonicalJson(ok.ingest) === canonicalJson(ING));

    rejects('a different base refuses',
        () => checkRebasePreconditions({
            layerManifest: lm, headChainDecl: same, layerChainDecl: same,
            headBaseIdentity: BASE_ID, layerBaseIdentity: 'c'.repeat(64),
        }), /is not the layer's base/);
    // The case 6.2 spells out: two depth-1 forks of a legacy base each asserted
    // a DIFFERENT declaration. Same base is not sufficient.
    rejects('two histories that disagree on the declaration refuse',
        () => checkRebasePreconditions({
            layerManifest: lm,
            headChainDecl: { canonical: canonicalJson({ chunker: 'v2' }), asserted: true },
            layerChainDecl: { canonical: canonicalJson(ING), asserted: true },
            headBaseIdentity: BASE_ID, layerBaseIdentity: BASE_ID,
        }), /disagree on the chain ingestion declaration/);
    const legacy = checkRebasePreconditions({
        layerManifest: lm, headIsLegacyBase: true,
        layerChainDecl: { canonical: canonicalJson(ING), asserted: true },
        headBaseIdentity: BASE_ID, layerBaseIdentity: BASE_ID,
    });
    check('onto a legacy base, an asserted declaration may be introduced',
        legacy.ingestAsserted === true);

    // Chain-level golden queries carry global ids that are wrong after a rebase.
    const plan = planRebase({
        layerBitset: maskFor(100, [5]), oldParentBitset: maskFor(100, []), oldRowBase: 100,
        layerRecords: 1, newRowBase: 100, headBitset: maskFor(100, []), forkRowBase: 100,
    });
    check('a rebase drops chain-level golden queries rather than copying them',
        plan.goldenQueries === null && plan.goldenQueriesDropped === 'rebase');
}

// ---------------------------------------------------------------------------
// compact and the lineage segment (6.3)
// ---------------------------------------------------------------------------
const {
    planCompaction, buildLineageSegment, parseLineageSegment, translateCitation,
    LINEAGE_HEADER_BYTES, LINEAGE_HEAD_ENTRY_BYTES, MAX_HEADS,
} = await import('../packages/pikelet-wasm/complete/layer-compact.mjs');

console.log('compact: live enumeration and dense renumbering (6.3 steps 1-2)');
{
    // rowTotal 20, ids 3, 7, 8 tombstoned.
    const head = maskFor(20, [3, 7, 8]);
    const plan = planCompaction({ rowTotal: 20, headBitset: head });
    check('live ids are enumerated in ascending global order',
        JSON.stringify(plan.liveIds.slice(0, 5)) === JSON.stringify([0, 1, 2, 4, 5]));
    check('the live count excludes every tombstone', plan.liveRecords === 17 && plan.tombstones === 3);
    // Survivors renumbered densely IN THE SAME ORDER — the translation is
    // positional, which is what makes compaction byte-deterministic.
    check('renumbering is dense and order-preserving',
        plan.newIdOf(0) === 0 && plan.newIdOf(2) === 2 && plan.newIdOf(4) === 3 && plan.newIdOf(19) === 16);
    check('a tombstoned id maps to nothing', plan.newIdOf(3) === null && plan.newIdOf(7) === null);
    rejects('a chain with no live records cannot be compacted',
        () => planCompaction({ rowTotal: 4, headBitset: maskFor(4, [0, 1, 2, 3]) }), /no live records/);
}

console.log('lineage-v1: round trip and the head table (6.3)');
{
    // A three-head chain: base(10) -> L1(+5) -> L2(+3). L1 superseded id 2
    // with id 10; L2 deleted id 4.
    const heads = [
        { identity: id64(1), rowTotal: 10, depth: 0 },
        { identity: id64(2), rowTotal: 15, depth: 1 },
        { identity: id64(3), rowTotal: 18, depth: 2 },
    ];
    const head = maskFor(18, [2, 4]);
    const seg = buildLineageSegment({
        heads, headBitset: head, oldRowTotal: 18,
        supersessions: [[2, 10, 1]],
    });
    const lin = parseLineageSegment(seg, { identity: id64(3), liveRecords: 16, tombstones: 2 });
    check('the head table carries every artifact of the compacted chain', lin.headCount === 3);
    check('depths are 0..headCount-1 in order', lin.heads.map((h) => h.depth).join() === '0,1,2');
    check('rowTotal values are non-decreasing, the last equal to oldRowTotal',
        lin.heads[2].rowTotal === 18 && lin.oldRowTotal === 18);
    check('the bitset and its popcount survive', lin.tombstoneCount === 2 && testBit(lin.bitset, 2) && testBit(lin.bitset, 4));
    check('each edge carries its own depth, so nothing is inferred from order (3.5)',
        JSON.stringify(lin.supersessions) === JSON.stringify([[2, 10, 1]]));
    check('the segment tiles exactly',
        seg.length === LINEAGE_HEADER_BYTES + LINEAGE_HEAD_ENTRY_BYTES * 3 + bitsetBytesFor(18) + 12);
    // Draft 2's segment named only the old head; the table is what makes a
    // citation against an INTERMEDIATE head resolvable.
    check('the table is at most 17 x 48 bytes', MAX_HEADS === 17 && LINEAGE_HEAD_ENTRY_BYTES === 48);
}

console.log('lineage-v1: validation of untrusted input (6.3)');
{
    const heads = [
        { identity: id64(1), rowTotal: 10, depth: 0 },
        { identity: id64(2), rowTotal: 15, depth: 1 },
    ];
    const good = buildLineageSegment({ heads, headBitset: maskFor(15, [2]), oldRowTotal: 15, supersessions: [[2, 10, 1]] });
    const tamp = (fn) => { const c = good.slice(); fn(new DataView(c.buffer), c); return c; };

    rejects('a wrong version', () => parseLineageSegment(tamp((v) => v.setUint32(0, 2, true))), /unsupported version 2/);
    rejects('non-zero flags', () => parseLineageSegment(tamp((v) => v.setUint32(4, 1, true))), /flags must be 0/);
    rejects('non-zero reserved header bytes', () => parseLineageSegment(tamp((v) => v.setUint32(32, 9, true))), /reserved header bytes/);
    rejects('a headCount past the ceiling', () => parseLineageSegment(tamp((v) => v.setUint32(8, 18, true))), /outside \[1, 17\]/);
    rejects('a headCount that does not tile', () => parseLineageSegment(tamp((v) => v.setUint32(8, 1, true))), /do not tile/);
    rejects('a tombstoneCount that disagrees with the popcount',
        () => parseLineageSegment(tamp((v) => v.setBigUint64(24, 9n, true))), /popcount is 1/);
    rejects('a truncated segment', () => parseLineageSegment(good.subarray(0, 40)), /shorter than its 64-byte header/);

    // Out-of-order depths, duplicate identities, decreasing rowTotals.
    rejects('head depths out of order', () => parseLineageSegment(tamp((v) => v.setUint32(LINEAGE_HEADER_BYTES + 40, 5, true))), /depths must be exactly/);
    rejects('a decreasing rowTotal', () => parseLineageSegment(tamp((v) => v.setBigUint64(LINEAGE_HEADER_BYTES + LINEAGE_HEAD_ENTRY_BYTES + 32, 5n, true))), /non-decreasing|must equal oldRowTotal/);
    rejects('duplicate head identities', () => parseLineageSegment(tamp((v, c) => {
        c.set(c.subarray(LINEAGE_HEADER_BYTES, LINEAGE_HEADER_BYTES + 32), LINEAGE_HEADER_BYTES + LINEAGE_HEAD_ENTRY_BYTES);
    })), /pairwise distinct|must equal oldRowTotal/);
    rejects('the last head not matching compactedFrom.identity',
        () => parseLineageSegment(good, { identity: id64(9) }), /must equal the manifest's compactedFrom.identity/);
    rejects('compactedFrom counts that disagree',
        () => parseLineageSegment(good, { identity: id64(2), tombstones: 5 }), /disagrees with manifest compactedFrom.tombstones/);

    // Edge constraints.
    rejects('an edge whose newId is outside the depth\'s owned range',
        () => buildLineageSegment({ heads, headBitset: maskFor(15, [2]), oldRowTotal: 15, supersessions: [[2, 14, 2]] }),
        /depth 2 must be in \[1, 1\]|outside the range/);
    rejects('an edge with oldId >= newId',
        () => buildLineageSegment({ heads, headBitset: maskFor(15, [11]), oldRowTotal: 15, supersessions: [[11, 10, 1]] }),
        /oldId < newId/);
    rejects('an edge whose oldId is not tombstoned',
        () => buildLineageSegment({ heads, headBitset: maskFor(15, [3]), oldRowTotal: 15, supersessions: [[2, 10, 1]] }),
        /not tombstoned/);
    rejects('duplicate (oldId, depth) pairs',
        () => buildLineageSegment({ heads, headBitset: maskFor(15, [2]), oldRowTotal: 15, supersessions: [[2, 10, 1], [2, 11, 1]] }),
        /appears twice/);
}

console.log('forward translation of a citation by a compacted base (6.3)');
{
    // base(10) -> L1(+5, supersedes 2 -> 10) -> L2(+3, deletes 4).
    const heads = [
        { identity: id64(1), rowTotal: 10, depth: 0 },
        { identity: id64(2), rowTotal: 15, depth: 1 },
        { identity: id64(3), rowTotal: 18, depth: 2 },
    ];
    const head = maskFor(18, [2, 4]);
    const lin = parseLineageSegment(buildLineageSegment({
        heads, headBitset: head, oldRowTotal: 18, supersessions: [[2, 10, 1]],
    }));
    const plan = planCompaction({ rowTotal: 18, headBitset: head });

    // A live record cited at the old base.
    const live = translateCitation(lin, plan.forward, id64(1), 0);
    check('a live record cited at the base translates to its new dense id',
        live.ok === true && live.newId === 0 && live.via === 'live');
    const live9 = translateCitation(lin, plan.forward, id64(1), 9);
    check('a later live id shifts down past the tombstones',
        live9.ok === true && live9.newId === 7);

    // THE reason the head table exists: a citation against an INTERMEDIATE
    // head, made while H1 was published, still resolves.
    const mid = translateCitation(lin, plan.forward, id64(2), 12);
    check('a citation against an intermediate head resolves', mid.ok === true && mid.citedAtDepth === 1);

    // A superseded record: forward translation follows the edge set.
    const sup = translateCitation(lin, plan.forward, id64(1), 2);
    check('a superseded record translates to its successor',
        sup.ok === true && sup.via === 'superseded' && sup.newId === plan.newIdOf(10));

    // A plainly deleted record has no successor.
    const gone = translateCitation(lin, plan.forward, id64(3), 4);
    check('a deleted record with no successor answers "deleted"', gone.reason === 'deleted');

    // 5.6's scoping rules carry over.
    check('an id that postdates the cited head is invalid',
        translateCitation(lin, plan.forward, id64(1), 12).reason === 'id-postdates-that-head');
    check('a fork not on this history is answered as such',
        translateCitation(lin, plan.forward, id64(9), 1).reason === 'not-on-this-history');
}

// ---------------------------------------------------------------------------
// The lineage segment through a REAL reader (6.3)
// ---------------------------------------------------------------------------
// Everything above tests buildLineageSegment/parseLineageSegment/
// translateCitation as functions. Nothing read the segment off a pack: the
// writer emitted it, the manifest carried its digest, and openPikeletFile had
// no citation(), no lineage() and no compactedFrom on info(). So a real
// compacted pack could not translate a citation at all, which is the entire
// purpose of the segment.
// ---------------------------------------------------------------------------
// Locator following through openPikeletChain (5.1.1)
// ---------------------------------------------------------------------------
// validateLocatorShape/resolveLocatorUrl/assertConfined/resolveLocatorPath/
// resolveParentLocation are all covered above, as functions. Nothing called
// them: layer-reader.mjs did not import layer-locator.mjs at all, while its
// header comment claimed "locator following ... is wired". So every ancestor
// had to be named by the caller and a published chain could not be mounted
// from its head, which is what 5.1.1 exists for.
console.log('locator following: a chain mounts from its head alone (5.1.1)');
{
    const wt = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-walk-'));
    const { openPikeletChain: openChain } = await import('../packages/pikelet-wasm/complete/layer-reader.mjs');
    const CDIM = 16;
    const ENC2 = { kind: 'host-encoder-v1', model: 'test' };
    const INGEST2 = { chunker: 'v1', targetTokens: 256 };
    const recBytes = (i) => Buffer.from(JSON.stringify({ title: `rec ${i}`, text: `record ${i} body text` }));
    const sketchFor = (ids, tag) => {
        const n = ids.length;
        const qdata = new Uint8Array(n * CDIM);
        const scales = new Float32Array(n);
        const offsets = new Float32Array(n);
        for (let r = 0; r < n; r++) {
            for (let d = 0; d < CDIM; d++) qdata[r * CDIM + d] = (ids[r] * 7 + d) % 256;
            scales[r] = 1 / 255; offsets[r] = 0;
        }
        const sp = path.join(wt, `sk-${tag}.pikelet-sketch`);
        exportSketchArtifact({ dim: CDIM, count: n, metric: 1, qdata, scales, offsets }, sp,
            { sketchDims: CDIM, sketchBits: 8, recommendedRerank: 20 });
        return fs.readFileSync(sp);
    };
    const mkBase = (dir, name, ids = [0, 1, 2, 3]) => {
        const corpus = buildCorpusSegment(ids.map(recBytes));
        const qi = buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' })));
        const at = path.join(dir, name);
        const built = assemblePikeletFile({
            profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
            corpus: { ...corpus.corpus, ingest: INGEST2 }, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, `b-${name}`) },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: qi },
        ], at);
        return { at, identity: built.identity, qiSha: built.manifest.segments.find((x) => x.kind === 'query-interp').sha256 };
    };
    const mkLayer = (dir, name, base, locator) => {
        const ids = [4, 5];
        const corpus = buildCorpusSegment(ids.map(recBytes));
        const at = path.join(dir, name);
        assemblePikeletFile({
            profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
            layer: {
                depth: 1, rowBase: 4, records: 2, baseIdentity: base.identity,
                tombstones: 1, supersessions: 0, ingest: INGEST2,
                parent: { identity: base.identity, ...(locator ? { locator } : {}) },
            },
            corpus: { ...corpus.corpus, ingest: INGEST2 }, index: {},
        }, [
            { kind: 'index', bytes: sketchFor(ids, `l-${name}`) },
            { kind: 'corpus', bytes: corpus.bytes },
            { kind: 'query-interp', bytes: buildInheritedQuerySegment({ baseIdentity: base.identity, queryInterpSha256: base.qiSha }) },
            { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: 4, tombstonedIds: [1], records: 2 }) },
        ], at);
        return at;
    };

    const base = mkBase(wt, 'base.pikelet');
    const headWith = mkLayer(wt, 'head.pikelet', base, 'base.pikelet');
    const headWithout = mkLayer(wt, 'nolocator.pikelet', base, null);

    // The whole point: one member in, a full chain mounted. Caught rather than
    // awaited bare, because a reader that does not walk throws here ("a layer
    // opened alone") and would abort the run instead of reporting a FAIL.
    let walked = null;
    let walkErr = null;
    try { walked = await openChain([headWith]); } catch (err) { walkErr = err; }
    check('a head alone mounts its whole chain through the locator',
        !!walked && walked.info().layers === 2 && walked.info().records === 6 && walked.info().tombstones === 1,
        walkErr ? `threw: ${walkErr.message.slice(0, 80)}` : JSON.stringify({ layers: walked.info().layers, records: walked.info().records }));
    const explicit = await openChain([base.at, headWith]);
    check('the walked chain is identical to the explicitly named one',
        !!walked && walked.info().identity === explicit.info().identity
        && JSON.stringify(walked.info().members) === JSON.stringify(explicit.info().members),
        walkErr ? 'the walk did not mount' : 'members differ');
    if (walked) await walked.close();
    await explicit.close();

    // A lineage listing is operator-supplied and takes precedence (5.1.1).
    const elsewhere = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-walk-far-'));
    fs.copyFileSync(base.at, path.join(elsewhere, 'base.pikelet'));
    const viaLineage = await openChain([headWithout], {
        lineage: new Map([[base.identity, path.join(elsewhere, 'base.pikelet')]]),
    });
    check('a lineage listing locates a parent the manifest names no locator for',
        viaLineage.info().layers === 2);
    await viaLineage.close();

    await rejectsAsync('a head with no locator and no lineage cannot be walked',
        () => openChain([headWithout]), /no lineage entry, no locator and no host resolver/);
    await rejectsAsync('resolveParents:false refuses to follow a locator',
        () => openChain([headWith], { resolveParents: false }), /locator resolution is disabled/);

    // Confinement: the locator names a sibling, but the real file it reaches is
    // outside the child's real directory (5.1.1's realpath wording).
    const sub = path.join(wt, 'sub');
    fs.mkdirSync(sub);
    const subHead = mkLayer(sub, 'head.pikelet', base, 'base.pikelet');
    await rejectsAsync('a locator naming a file that is not there is an availability failure',
        () => openChain([subHead]), /cannot be located/);
    fs.symlinkSync(base.at, path.join(sub, 'base.pikelet'));
    await rejectsAsync('a symlink escaping the child\'s real directory is refused',
        () => openChain([subHead]), /outside the child's real directory/);

    // A locator that resolves to a real pack which is NOT the parent the child
    // committed to must fail on identity, before the pairwise checks.
    fs.rmSync(path.join(sub, 'base.pikelet'));
    // Different records, so this is genuinely a different artifact: mkBase is
    // deterministic and a second call with the same ids reproduces the identity.
    const other = mkBase(sub, 'base.pikelet', [10, 11, 12, 13]);
    check('the decoy base really is a different artifact', other.identity !== base.identity);
    await rejectsAsync('a locator resolving to the wrong artifact is refused on identity',
        () => openChain([subHead]), /is not the parent the chain commits to/);

    fs.rmSync(wt, { recursive: true, force: true });
    fs.rmSync(elsewhere, { recursive: true, force: true });
}

// ---------------------------------------------------------------------------
// rebase, through the CLI (6.2)
// ---------------------------------------------------------------------------
// planRebase's rules are covered above as a function. Nothing exercised the
// COMMAND: it threw "not wired as a command yet" until now, and the first
// working version still had a bug the function tests could not see (it
// destructured plan.skippedEdges as [x, y] pairs when planRebase pushes bare
// ids, so --on-conflict skip crashed). These drive the real binary.
console.log('cli: repeated --source (6.1)');
{
    const { parseArgs } = await import('../packages/pikelet/src/cli.mjs');
    const a = parseArgs(['append', '--source', 'a', '--source', 'b']);
    check('append keeps every --source, not only the last',
        Array.isArray(a.flags.source) && a.flags.source.join() === 'a,b', JSON.stringify(a.flags.source));
    const c = parseArgs(['compile', '--source', 'docs']);
    check('compile still reads a single --source as a string', c.flags.source === 'docs');
    rejects('compile refuses a second --source rather than keeping the last',
        () => parseArgs(['compile', '--source', 'a', '--source', 'b']), /--source may be given once/);
}

console.log('cli: --remove and --supersede ids parse strictly (6.1)');
{
    const { appendLayer } = await import('../packages/pikelet/src/append.mjs');
    const out = path.join(os.tmpdir(), `pikelet-remove-parse-${process.pid}.pikelet`);
    // parseInt read '1,2,3' as 1 and '12abc' as 12: a different tombstone set
    // than the one named, and no error. Each must now refuse before any mount.
    for (const bad of ['1,2,3', '12abc', '1.5', '-1', '', '1e3']) {
        await rejectsAsync(`--remove ${JSON.stringify(bad)} is refused`,
            () => appendLayer({ parent: 'nosuch.pikelet', out, remove: [bad] }), /--remove takes one non-negative record id/);
    }
    await rejectsAsync('--supersede with a non-integer old id is refused',
        () => appendLayer({ parent: 'nosuch.pikelet', out, supersede: ['2x=a.md'] }), /--supersede oldId must be a non-negative id/);
    await rejectsAsync('a well-formed --remove gets past parsing (fails on the parent instead)',
        () => appendLayer({ parent: 'nosuch.pikelet', out, remove: [' 7 '] }), /^(?!.*--remove takes)/);
}

console.log('rebase: the command, end to end (6.2)');
{
    const { execFileSync } = await import('node:child_process');
    const BIN = path.resolve('packages/pikelet/bin/pikelet.mjs');
    const rt = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-rebase-cli-'));
    const run = (args) => {
        try {
            return { ok: true, out: execFileSync(process.execPath, [BIN, ...args], { encoding: 'utf8', stdio: ['ignore','pipe','pipe'] }) };
        } catch (err) {
            return { ok: false, out: String(err.stdout || '') + String(err.stderr || '') };
        }
    };

    // Own fixtures: the chain-file block above scopes its tmpdir to itself.
    const CDIM = 16;
    const ENC2 = { kind: 'host-encoder-v1', model: 'test' };
    const INGEST2 = { chunker: 'v1', targetTokens: 256 };
    const recBytes = (i) => Buffer.from(JSON.stringify({ title: `rec ${i}`, text: `record ${i} body text` }));
    const sketchFor = (ids, tag) => {
        const n = ids.length;
        const qdata = new Uint8Array(n * CDIM);
        const scales = new Float32Array(n);
        const offsets = new Float32Array(n);
        for (let r = 0; r < n; r++) {
            for (let d = 0; d < CDIM; d++) qdata[r * CDIM + d] = (ids[r] * 7 + d) % 256;
            scales[r] = 1 / 255; offsets[r] = 0;
        }
        const sp = path.join(rt, `sk-${tag}.pikelet-sketch`);
        exportSketchArtifact({ dim: CDIM, count: n, metric: 1, qdata, scales, offsets }, sp,
            { sketchDims: CDIM, sketchBits: 8, recommendedRerank: 20 });
        return fs.readFileSync(sp);
    };
    const baseIds = [0, 1, 2, 3];
    const baseCorpus = buildCorpusSegment(baseIds.map(recBytes));
    const baseFile = path.join(rt, 'base.pikelet');
    const baseBuilt = assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...baseCorpus.corpus, ingest: INGEST2 }, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(baseIds, 'base') },
        { kind: 'corpus', bytes: baseCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
    ], baseFile);
    const l1Ids = [4, 5];
    const l1Corpus = buildCorpusSegment(l1Ids.map(recBytes));
    const l1File = path.join(rt, 'l1.pikelet');
    const l1Built = assemblePikeletFile({
        profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
        layer: {
            parent: { identity: baseBuilt.identity, locator: 'base.pikelet' },
            baseIdentity: baseBuilt.identity, depth: 1, rowBase: 4, records: 2,
            tombstones: 1, supersessions: 0, ingest: INGEST2,
        },
        corpus: l1Corpus.corpus, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(l1Ids, 'l1') },
        { kind: 'corpus', bytes: l1Corpus.bytes },
        { kind: 'query-interp', bytes: buildInheritedQuerySegment({
            baseIdentity: baseBuilt.identity,
            queryInterpSha256: baseBuilt.manifest.segments.find((x) => x.kind === 'query-interp').sha256,
        }) },
        { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: 4, tombstonedIds: [1], records: 2 }) },
    ], l1File);
    check('rebase fixtures assemble', fs.existsSync(baseFile) && fs.existsSync(l1File));

    // Argument validation happens before any mount, so it needs no fixtures.
    const noLayer = run(['rebase', '--onto', baseFile, '--out', path.join(rt, 'a.pikelet')]);
    check('rebase without --layer is refused',
        !noLayer.ok && /requires --layer/.test(noLayer.out), noLayer.out.slice(0, 90));
    const noOnto = run(['rebase', '--layer', l1File, '--out', path.join(rt, 'a.pikelet')]);
    check('rebase without --onto is refused',
        !noOnto.ok && /requires --onto/.test(noOnto.out), noOnto.out.slice(0, 90));
    const noOut = run(['rebase', '--layer', l1File, '--onto', baseFile]);
    check('rebase without --out is refused',
        !noOut.ok && /requires --out/.test(noOut.out), noOut.out.slice(0, 90));
    const badConflict = run(['rebase', '--layer', l1File, '--onto', baseFile,
        '--on-conflict', 'nonsense', '--out', path.join(rt, 'a.pikelet')]);
    check('an unknown --on-conflict is refused, naming the allowed values',
        !badConflict.ok && /--on-conflict must be one of refuse, skip, keep-both/.test(badConflict.out),
        badConflict.out.slice(0, 90));
    const badForeign = run(['rebase', '--layer', l1File, '--onto', baseFile,
        '--on-foreign', 'nonsense', '--out', path.join(rt, 'a.pikelet')]);
    check('an unknown --on-foreign is refused',
        !badForeign.ok && /--on-foreign must be one of refuse, drop/.test(badForeign.out),
        badForeign.out.slice(0, 90));

    // --layer must BE a layer.
    const notALayer = run(['rebase', '--layer', baseFile, '--onto', baseFile, '--out', path.join(rt, 'a.pikelet')]);
    check('rebase refuses a --layer that is not a layer',
        !notALayer.ok && /is not a layer/.test(notALayer.out), notALayer.out.slice(0, 100));

    // 6.2: the original parent is REQUIRED to compute what the layer itself
    // deleted. A copy of the layer with no reachable parent must refuse, and
    // must explain why rather than leaking an ENOENT.
    const orphanDir = path.join(rt, 'orphan');
    fs.mkdirSync(orphanDir);
    fs.copyFileSync(l1File, path.join(orphanDir, 'l1.pikelet'));
    const orphan = run(['rebase', '--layer', path.join(orphanDir, 'l1.pikelet'),
        '--onto', baseFile, '--out', path.join(rt, 'a.pikelet')]);
    check('a layer whose original parent cannot be opened is refused, with the reason',
        !orphan.ok && /delta is unknowable|difference is unknowable/.test(orphan.out),
        orphan.out.slice(0, 120));

    // --old-parent pointing at a real but WRONG artifact must fail on identity,
    // not later. A second base with different records is openable and has a
    // different identity, which is the case that matters: a locator or operator
    // path that resolves to something valid but unrelated.
    const otherIds = [10, 11, 12, 13];
    const otherCorpus = buildCorpusSegment(otherIds.map(recBytes));
    const otherBase = path.join(rt, 'otherbase.pikelet');
    assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...otherCorpus.corpus, ingest: INGEST2 }, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(otherIds, 'other') },
        { kind: 'corpus', bytes: otherCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
    ], otherBase);
    const wrongParent = run(['rebase', '--layer', l1File, '--onto', baseFile,
        '--old-parent', otherBase, '--out', path.join(rt, 'a.pikelet')]);
    check('an --old-parent that is not the committed parent is refused on identity',
        !wrongParent.ok && /not the parent this layer was cut against/.test(wrongParent.out),
        wrongParent.out.slice(-140));

    // A SUCCESSFUL rebase: fork the base twice, then move one layer onto the
    // other. Both forks claim rowBase 4; the rebased one must move to 6.
    const forkIds = [4, 5];
    const forkCorpus = buildCorpusSegment(forkIds.map(recBytes));
    const forkFile = path.join(rt, 'fork.pikelet');
    const forkBuilt = assemblePikeletFile({
        profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
        layer: {
            parent: { identity: baseBuilt.identity, locator: 'base.pikelet' },
            baseIdentity: baseBuilt.identity, depth: 1, rowBase: 4, records: 2,
            tombstones: 0, supersessions: 0, ingest: INGEST2,
        },
        corpus: forkCorpus.corpus, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(forkIds, 'fork') },
        { kind: 'corpus', bytes: forkCorpus.bytes },
        { kind: 'query-interp', bytes: buildInheritedQuerySegment({
            baseIdentity: baseBuilt.identity,
            queryInterpSha256: baseBuilt.manifest.segments.find((x) => x.kind === 'query-interp').sha256,
        }) },
        { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: 4, tombstonedIds: [], records: 2 }) },
    ], forkFile);
    check('the two forks are different artifacts claiming the same rowBase',
        forkBuilt.identity !== baseBuilt.identity);

    const outFile = path.join(rt, 'rebased.pikelet');
    const ok = run(['rebase', '--layer', l1File, '--onto', baseFile, '--onto', forkFile, '--out', outFile]);
    check('a rebase onto a fork sibling succeeds', ok.ok && fs.existsSync(outFile), ok.out.slice(-160));
    check('the rebase reports the new rowBase and a verbatim copy',
        /rowBase 4 -> 6/.test(ok.out) && /copied verbatim/.test(ok.out), ok.out.slice(-200));
    check('chain-level goldens are dropped, never copied (6.2)',
        /goldenQueriesDropped/.test(ok.out));

    // The rebased layer must mount as depth 2 on the new history and serve both
    // forks' records.
    const { openPikeletChain: openChain2 } = await import('../packages/pikelet-wasm/complete/layer-reader.mjs');
    const rebasedChain = await openChain2([baseFile, forkFile, outFile]);
    const ri = rebasedChain.info();
    check('the rebased chain mounts as a valid three-member chain',
        ri.layers === 3 && ri.records === 8, JSON.stringify({ layers: ri.layers, records: ri.records }));
    check('the rebased layer sits at depth 2 with its own range',
        ri.members[2].identity !== l1Built.identity && ri.members[2].depth === 2 && ri.members[2].records === 2,
        JSON.stringify(ri.members.map((m) => m.depth)));
    // The layer tombstoned base id 1; that deletion must survive the move.
    const moved = await rebasedChain.record(1);
    check('a deletion the layer introduced survives the rebase', moved.tombstoned === true, JSON.stringify(moved.tombstoned));
    await rebasedChain.close();

    // 5.1.1: rebase wrote no parent.locator unless told to, so the rebased
    // layer could not be walked from its head and its next rebase needed
    // --old-parent. A local parent under the output directory now gets one.
    const { readMemberManifest: peekManifest } = await import('../packages/pikelet-wasm/complete/index.mjs');
    const rebasedPeek = await peekManifest(outFile);
    check('the rebased layer records its new parent\'s locator',
        rebasedPeek.manifest.layer.parent.locator === 'fork.pikelet',
        JSON.stringify(rebasedPeek.manifest.layer.parent));
    const walked = await openChain2([outFile]);
    check('the rebased layer mounts alone, walking up by locator',
        walked.info().layers === 3 && walked.info().records === 8, JSON.stringify(walked.info().layers));
    await walked.close();

    // `#<sha256>` pins on chain members. They were advertised in every usage
    // line but never parsed: a local path failed as a missing file and a URL
    // dropped its fragment, so the pin was silently ignored.
    const pinnedOut = path.join(rt, 'rebased-pinned.pikelet');
    const pinned = run(['rebase', '--layer', `${l1File}#${l1Built.identity}`, '--onto', baseFile,
        '--onto', `${forkFile}#${forkBuilt.identity}`, '--out', pinnedOut]);
    check('correct #identity pins are accepted', pinned.ok && fs.existsSync(pinnedOut), pinned.out.slice(-160));
    const misPinned = run(['rebase', '--layer', l1File, '--onto', baseFile,
        '--onto', `${forkFile}#${baseBuilt.identity}`, '--out', path.join(rt, 'mis.pikelet')]);
    check('a pin naming a different artifact is refused on identity',
        !misPinned.ok && /identity mismatch/.test(misPinned.out) && !fs.existsSync(path.join(rt, 'mis.pikelet')),
        misPinned.out.slice(-160));
    const shortPin = run(['rebase', '--layer', l1File, '--onto', baseFile,
        '--onto', `${forkFile}#${forkBuilt.identity.slice(0, 12)}`, '--out', path.join(rt, 'short.pikelet')]);
    check('a truncated pin is refused, not read as a filename',
        !shortPin.ok && /full 64-hex/.test(shortPin.out), shortPin.out.slice(-160));
    const urlPin = run(['append', '--parent', 'https://example.invalid/x.pikelet#abc', '--remove', '1',
        '--out', path.join(rt, 'url.pikelet')]);
    check('a malformed pin on a URL is refused before any request',
        !urlPin.ok && /full 64-hex/.test(urlPin.out), urlPin.out.slice(-160));
    const badLocator = run(['rebase', '--layer', l1File, '--onto', baseFile, '--onto', forkFile,
        '--parent-locator', '../fork.pikelet', '--out', path.join(rt, 'badloc.pikelet')]);
    check('an explicit --parent-locator that escapes the directory is refused',
        !badLocator.ok && /--parent-locator .*dot segment/.test(badLocator.out), badLocator.out.slice(-160));

    // 5.1.1 over HTTP: the walk up from a URL head. The first hop had no child
    // location (a range source carried no url), and a parent URL string was
    // then read as a file path, so no URL chain could be mounted from its head.
    // Every redirect on the way to a locator-named parent is confined too.
    {
        const http = await import('node:http');
        // The Node entry, as 'pikelet-wasm/complete' resolves under Node: its
        // httpRangeSource defaults to nodeFetch, and the walk must open every
        // parent with the head's transport (npm run test:fetch-tripwire).
        const { httpRangeSource } = await import('../packages/pikelet-wasm/complete/index.node.mjs');
        let redirectFork = null; // null | where /packs/fork.pikelet redirects to
        const host = http.createServer((req, res) => {
            const u = new URL(req.url, 'http://x');
            if (u.pathname === '/packs/fork.pikelet' && redirectFork) {
                res.writeHead(302, { location: redirectFork });
                res.end();
                return;
            }
            const m = /^\/(?:packs|packs\/v2|elsewhere)\/([a-z-]+\.pikelet)$/.exec(u.pathname);
            const file = m && path.join(rt, m[1]);
            if (!file || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
            const payload = fs.readFileSync(file);
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
        const origin = `http://127.0.0.1:${host.address().port}`;
        const headAt = async () => {
            const src = httpRangeSource(`${origin}/packs/rebased.pikelet`);
            await src.init();
            return src;
        };
        try {
            try {
                const urlChain = await openChain2([await headAt()]);
                const ui = urlChain.info();
                check('a URL head mounts by walking its locators over HTTP',
                    ui.layers === 3 && ui.records === 8, JSON.stringify({ layers: ui.layers, records: ui.records }));
                await urlChain.close();
            } catch (err) {
                check('a URL head mounts by walking its locators over HTTP', false, err.message.slice(0, 160));
            }

            redirectFork = '/packs/v2/fork.pikelet';
            try {
                const inDir = await openChain2([await headAt()]);
                check('a redirect that stays under the child\'s directory is followed', inDir.info().layers === 3);
                await inDir.close();
            } catch (err) {
                check('a redirect that stays under the child\'s directory is followed', false, err.message.slice(0, 160));
            }

            redirectFork = '/elsewhere/fork.pikelet';
            await rejectsAsync('a redirect out of the child\'s directory is refused before it is followed',
                async () => { const c = await openChain2([await headAt()]); await c.close(); },
                /refused redirect .*not under the child's directory/);
        } finally {
            host.close();
        }
    }

    // Terminal rule: a tombstone-only layer whose every deletion the target
    // history already performed emits NOTHING (6.2).
    const tombOnly = (name) => {
        const at = path.join(rt, name);
        assemblePikeletFile({
            profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
            layer: {
                parent: { identity: baseBuilt.identity, locator: 'base.pikelet' },
                baseIdentity: baseBuilt.identity, depth: 1, rowBase: 4, records: 0,
                tombstones: 1, supersessions: 0, ingest: INGEST2,
            },
            // 4.1: a tombstone-only layer has no index segment, so it must omit
            // the index object entirely rather than carry an empty one.
            corpus: { records: 0 },
        }, [
            { kind: 'query-interp', bytes: buildInheritedQuerySegment({
                baseIdentity: baseBuilt.identity,
                queryInterpSha256: baseBuilt.manifest.segments.find((x) => x.kind === 'query-interp').sha256,
            }) },
            { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: 4, tombstonedIds: [2], records: 0 }) },
        ], at);
        return at;
    };
    const tombA = tombOnly('tomb-a.pikelet');
    const tombB = tombOnly('tomb-b.pikelet');
    const emptyOut = path.join(rt, 'empty.pikelet');
    const empty = run(['rebase', '--layer', tombA, '--onto', baseFile, '--onto', tombB, '--out', emptyOut]);
    check('a rebase whose operation the target already represents emits nothing',
        empty.ok && /Nothing to emit/.test(empty.out) && !fs.existsSync(emptyOut),
        `exit ${empty.ok}, file ${fs.existsSync(emptyOut)}: ${empty.out.slice(-160)}`);

    // A head that is a lone base DECLARING ingest is not a legacy base. The
    // precondition read a field __head never carried and treated every
    // one-member head as legacy, so this rebase was always refused.
    const bareOut = path.join(rt, 'onto-bare.pikelet');
    const bare = run(['rebase', '--layer', l1File, '--onto', baseFile, '--old-parent', baseFile, '--out', bareOut]);
    check('a rebase onto a lone base that declares ingest succeeds',
        bare.ok && fs.existsSync(bareOut), bare.out.slice(-200));
    if (fs.existsSync(bareOut)) {
        const bc = await openChain2([baseFile, bareOut]);
        check('the layer rebased onto a bare base mounts', bc.info().layers === 2);
        await bc.close();
    }

    // Two depth-1 forks of a LEGACY base (no corpus.ingest) that asserted
    // DIFFERENT declarations. The precondition compared `.canonical` on
    // objects that had none, so undefined === undefined let this through and
    // wrote a layer the reader then rejected (6.1, 6.2).
    const legacyCorpus = buildCorpusSegment(baseIds.map(recBytes));
    const legacyFile = path.join(rt, 'legacy.pikelet');
    const legacyBuilt = assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...legacyCorpus.corpus }, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(baseIds, 'legacy') },
        { kind: 'corpus', bytes: legacyCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
    ], legacyFile);

    // 6.1 through the CLI: a legacy base's refusal says what to assert, and a
    // declared chunker this pikelet does not implement is refused before any
    // source is chunked under the wrong rules.
    // The literal, not ingestDeclaration(): every compiled pack carries these
    // bytes, so a change to them must fail here rather than pass silently.
    const DECL = '{"chunker":"pikelet-section-v1","targetTokens":256,"overlapPercent":15,"minTokens":25}';
    const legacyAppend = run(['append', '--parent', legacyFile, '--remove', '1', '--out', path.join(rt, 'la.pikelet')]);
    check('append on a base without corpus.ingest prints the declaration to assert',
        !legacyAppend.ok && legacyAppend.out.includes(DECL),
        legacyAppend.out.slice(-200));
    const foreign = run(['append', '--parent', baseFile, '--remove', '1', '--out', path.join(rt, 'fc.pikelet')]);
    check('append refuses a chain declared with a chunker this pikelet does not implement',
        !foreign.ok && /made by chunker "v1", but this pikelet chunks with "pikelet-section-v1"/.test(foreign.out)
        && !fs.existsSync(path.join(rt, 'fc.pikelet')),
        foreign.out.slice(-200));
    const assertedFork = (name, ingest) => {
        const at = path.join(rt, name);
        const c = buildCorpusSegment([4, 5].map(recBytes));
        assemblePikeletFile({
            profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
            layer: {
                parent: { identity: legacyBuilt.identity, locator: 'legacy.pikelet' },
                baseIdentity: legacyBuilt.identity, depth: 1, rowBase: 4, records: 2,
                tombstones: 0, supersessions: 0, ingest, ingestAsserted: true,
            },
            corpus: c.corpus, index: {},
        }, [
            { kind: 'index', bytes: sketchFor([4, 5], name) },
            { kind: 'corpus', bytes: c.bytes },
            { kind: 'query-interp', bytes: buildInheritedQuerySegment({
                baseIdentity: legacyBuilt.identity,
                queryInterpSha256: legacyBuilt.manifest.segments.find((x) => x.kind === 'query-interp').sha256,
            }) },
            { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: 4, tombstonedIds: [], records: 2 }) },
        ], at);
        return at;
    };
    const forkA = assertedFork('fork-a.pikelet', { chunker: 'A' });
    const forkB = assertedFork('fork-b.pikelet', { chunker: 'B' });
    const mismatchOut = path.join(rt, 'mismatch.pikelet');
    const mismatch = run(['rebase', '--layer', forkB, '--onto', legacyFile, '--onto', forkA, '--out', mismatchOut]);
    check('a rebase between histories that asserted different declarations is refused',
        !mismatch.ok && /disagree on the chain ingestion declaration/.test(mismatch.out) && !fs.existsSync(mismatchOut),
        mismatch.out.slice(-200));
    // The same fork moved onto the bare legacy base may introduce its
    // assertion, and the output must carry it as the new depth-1 member.
    const legacyOut = path.join(rt, 'onto-legacy.pikelet');
    const ontoLegacy = run(['rebase', '--layer', forkB, '--onto', legacyFile, '--out', legacyOut]);
    check('an asserted layer rebased onto a bare legacy base succeeds',
        ontoLegacy.ok && fs.existsSync(legacyOut), ontoLegacy.out.slice(-200));
    if (fs.existsSync(legacyOut)) {
        const lc = await openChain2([legacyFile, legacyOut]);
        check('it mounts as an asserted depth-1 chain', lc.__head.chainIngest?.asserted === true,
            JSON.stringify(lc.__head.chainIngest));
        await lc.close();
    }

    // A segment that fails its digest must stop the copy. copyLayerSegments
    // caught the error as "absent", so a corrupt corpus segment was silently
    // left out and the command wrote an unmountable layer with exit 0.
    const corruptFile = path.join(rt, 'corrupt.pikelet');
    const cbytes = fs.readFileSync(l1File);
    {
        const man = cbytes.readUInt32LE(8);
        const nseg = cbytes.readUInt32LE(12);
        for (let i = 0; i < nseg; i++) {
            const at = 64 + man + i * 48;
            if (cbytes.readUInt32LE(at) !== 2) continue;   // corpus
            cbytes[Number(cbytes.readBigUInt64LE(at + 8)) + 5] ^= 0xff;
        }
    }
    fs.writeFileSync(corruptFile, cbytes);
    const corruptOut = path.join(rt, 'from-corrupt.pikelet');
    const corrupt = run(['rebase', '--layer', corruptFile, '--onto', baseFile, '--onto', forkFile,
        '--old-parent', baseFile, '--out', corruptOut]);
    check('a layer whose corpus segment fails its digest is refused, not copied without it',
        !corrupt.ok && /corpus segment fails its manifest digest/.test(corrupt.out) && !fs.existsSync(corruptOut),
        corrupt.out.slice(-200));

    fs.rmSync(rt, { recursive: true, force: true });
}

console.log('tombstone-only layers: the segment table must agree with the manifest (4.3)');
{
    // The segment table sits outside the identity. readChainMemberShell, the
    // path a tombstone-only layer is read through, used to keep only length
    // and digest shape per entry: it never compared the table's kind with the
    // manifest's, and a later entry of the same kind overwrote an earlier one.
    // Relabelling a committed evaluation segment as kind 6 (tombstones) in
    // the table therefore swapped which bytes served as the deletion mask,
    // and the identity still verified.
    const { readChainMemberShell } = await import('../packages/pikelet-wasm/complete/index.mjs');
    const tt = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-shell-'));
    const BASE = 'c'.repeat(64);
    const QISHA = 'd'.repeat(64);
    const qi = buildInheritedQuerySegment({ baseIdentity: BASE, queryInterpSha256: QISHA });
    const layerFields = {
        profile: PROFILE_LAYER, dim: 16, metric: 'cosine', encoder: { kind: 'host-encoder-v1', model: 'test' },
        layer: {
            parent: { identity: BASE }, baseIdentity: BASE, depth: 1, rowBase: 10, records: 0,
            tombstones: 1, supersessions: 0, ingest: { chunker: 'v1' },
        },
        corpus: { records: 0 },
    };
    const mask = buildTombstoneSegment({ rowBase: 10, tombstonedIds: [3], records: 0 });
    // An evaluation segment that happens to parse as a DIFFERENT mask.
    const decoy = buildTombstoneSegment({ rowBase: 10, tombstonedIds: [1, 2, 4], records: 0 });
    const good = path.join(tt, 'good.pikelet');
    assemblePikeletFile(layerFields, [
        { kind: 'query-interp', bytes: qi },
        { kind: 'tombstones', bytes: mask },
        { kind: 'evaluation', bytes: decoy },
    ], good);
    const shell = await readChainMemberShell(good);
    check('an untampered tombstone-only layer reads through the shell', shell.tombstoneOnly === true);
    await shell.close();

    // Rewrite table entry `i`'s kind number on a copy; the manifest, and so
    // the identity, are untouched.
    const relabel = (name, i, kindNumber) => {
        const buf = fs.readFileSync(good);
        buf.writeUInt32LE(kindNumber, 64 + buf.readUInt32LE(8) + i * 48);
        const at = path.join(tt, name);
        fs.writeFileSync(at, buf);
        return at;
    };
    await rejectsAsync('a table that relabels the evaluation segment as tombstones is refused',
        () => readChainMemberShell(relabel('swap.pikelet', 2, 6)), /segment table disagrees with manifest at entry 2/);
    await rejectsAsync('a table kind the manifest does not name is refused',
        () => readChainMemberShell(relabel('mislabel.pikelet', 1, 4)), /segment table disagrees with manifest at entry 1/);

    // 4.3: a tombstone-only layer MUST NOT carry index, corpus or lexical.
    const withLex = path.join(tt, 'withlex.pikelet');
    assemblePikeletFile(layerFields, [
        { kind: 'query-interp', bytes: qi },
        { kind: 'tombstones', bytes: mask },
        { kind: 'lexical', bytes: buildLexSeg(['stray text']).bytes },
    ], withLex);
    await rejectsAsync('a tombstone-only layer carrying a lexical segment is refused',
        () => readChainMemberShell(withLex), /must not carry a lexical segment/);

    fs.rmSync(tt, { recursive: true, force: true });
}

console.log('compact: live records are copied byte for byte (6.3)');
{
    // compact read each record through chain.record(), which lays chain
    // fields over the parsed record -- including a global `id` that replaces
    // the record's own -- then stripped those fields and re-serialized. Every
    // compacted record lost its own id, and key order and number formatting
    // were whatever JSON.stringify produced. 6.3 says the compacted base's
    // records are the old head's live records, bytes unchanged.
    const { execFileSync } = await import('node:child_process');
    const { openPikeletFile } = await import('../packages/pikelet-wasm/complete/index.mjs');
    const ct = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-compact-bytes-'));
    const CDIM = 16;
    const ENC2 = { kind: 'host-encoder-v1', model: 'test' };
    const INGEST2 = { chunker: 'v1', targetTokens: 256 };
    // Deliberately not what JSON.stringify would write: the record's own id
    // first, spaces after separators, and a number with a trailing zero.
    const raw = (i) => Buffer.from(`{"id": "chunk-${i}", "weight": 1.50, "title": "rec ${i}", "text": "record ${i} body text"}`);
    const sketchFor = (ids, tag) => {
        const qdata = new Uint8Array(ids.length * CDIM);
        for (let r = 0; r < ids.length; r++) for (let d = 0; d < CDIM; d++) qdata[r * CDIM + d] = (ids[r] * 7 + d) % 256;
        const sp = path.join(ct, `sk-${tag}.pikelet-sketch`);
        exportSketchArtifact({ dim: CDIM, count: ids.length, metric: 1, qdata,
            scales: new Float32Array(ids.length).fill(1 / 255), offsets: new Float32Array(ids.length) }, sp,
        { sketchDims: CDIM, sketchBits: 8, recommendedRerank: 20 });
        return fs.readFileSync(sp);
    };
    const baseIds = [0, 1, 2, 3];
    const baseCorpus = buildCorpusSegment(baseIds.map(raw));
    const basePath = path.join(ct, 'base.pikelet');
    const baseBuilt = assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...baseCorpus.corpus, ingest: INGEST2 }, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(baseIds, 'base') },
        { kind: 'corpus', bytes: baseCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
    ], basePath);
    const layerIds = [4, 5];
    const layerCorpus = buildCorpusSegment(layerIds.map(raw));
    const layerPath = path.join(ct, 'l1.pikelet');
    assemblePikeletFile({
        profile: PROFILE_LAYER, dim: CDIM, metric: 'cosine', encoder: ENC2,
        layer: {
            parent: { identity: baseBuilt.identity }, baseIdentity: baseBuilt.identity,
            depth: 1, rowBase: 4, records: 2, tombstones: 1, supersessions: 0, ingest: INGEST2,
        },
        corpus: layerCorpus.corpus, index: {},
    }, [
        { kind: 'index', bytes: sketchFor(layerIds, 'l1') },
        { kind: 'corpus', bytes: layerCorpus.bytes },
        { kind: 'query-interp', bytes: buildInheritedQuerySegment({
            baseIdentity: baseBuilt.identity,
            queryInterpSha256: baseBuilt.manifest.segments.find((x) => x.kind === 'query-interp').sha256,
        }) },
        { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: 4, tombstonedIds: [1], records: 2 }) },
    ], layerPath);

    const outPath = path.join(ct, 'compacted.pikelet');
    let out = '';
    try {
        out = execFileSync(process.execPath, [path.resolve('packages/pikelet/bin/pikelet.mjs'), 'compact',
            '--head', basePath, '--head', layerPath, '--out', outPath, '--no-refit'],
        { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (err) { out = String(err.stdout || '') + String(err.stderr || ''); }
    check('compact over a chain whose records carry their own ids succeeds', fs.existsSync(outPath), out.slice(-200));
    if (fs.existsSync(outPath)) {
        const compacted = await openPikeletFile(outPath);
        const liveOld = [0, 2, 3, 4, 5];
        const got = await Promise.all(liveOld.map((_, i) => compacted.__chainTier.readRecordBytes(i)));
        const same = liveOld.every((old, i) => Buffer.compare(Buffer.from(got[i]), raw(old)) === 0);
        check('every live record is byte-identical to the one it was compacted from', same,
            same ? '' : Buffer.from(got[0]).toString('utf8'));
        const first = await compacted.record(0);
        check('a compacted record keeps its own id field', first.id === 'chunk-0', JSON.stringify(first));
        await compacted.close();
    }
    fs.rmSync(ct, { recursive: true, force: true });
}

console.log('lineage: a compacted base translates citations through the reader');
{
    const linTmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-lineage-'));
    const { openPikeletFile } = await import('../packages/pikelet-wasm/complete/index.mjs');
    // Local copies: the chain-file block above scopes these to itself.
    const CDIM = 16;
    const ENC2 = { kind: 'host-encoder-v1', model: 'test' };
    const INGEST2 = { chunker: 'v1', targetTokens: 256 };
    const recBytes = (i) => Buffer.from(JSON.stringify({ title: `rec ${i}`, text: `record ${i} body text` }));
    const sketchFor = (ids) => {
        const n = ids.length;
        const qdata = new Uint8Array(n * CDIM);
        const scales = new Float32Array(n);
        const offsets = new Float32Array(n);
        for (let r = 0; r < n; r++) {
            for (let d = 0; d < CDIM; d++) qdata[r * CDIM + d] = (ids[r] * 7 + d) % 256;
            scales[r] = 1 / 255; offsets[r] = 0;
        }
        const sp = path.join(linTmp, `sk-${n}.pikelet-sketch`);
        exportSketchArtifact({ dim: CDIM, count: n, metric: 1, qdata, scales, offsets }, sp,
            { sketchDims: CDIM, sketchBits: 8, recommendedRerank: 20 });
        return fs.readFileSync(sp);
    };

    // An old chain: base of 6 (depth 0), layer of 2 (depth 1) that tombstones
    // old ids 1 and 2 and supersedes 2 -> 6. Live after: 0, 3, 4, 5, 6, 7.
    const OLD_BASE = id64(41);
    const OLD_HEAD = id64(42);
    const oldRowTotal = 8;
    const headBitset = new Uint8Array(1);
    headBitset[0] |= (1 << 1) | (1 << 2);
    const lineageBytes = buildLineageSegment({
        heads: [
            { identity: OLD_BASE, rowTotal: 6, depth: 0 },
            { identity: OLD_HEAD, rowTotal: 8, depth: 1 },
        ],
        headBitset,
        oldRowTotal,
        supersessions: [[2, 6, 1]],
    });

    // The compacted base: 6 live records, dense ids 0..5.
    const liveOld = [0, 3, 4, 5, 6, 7];
    const cIds = liveOld.map((_, i) => i);
    const cCorpus = buildCorpusSegment(cIds.map(recBytes));
    const cQi = buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' })));
    const cPath = path.join(linTmp, 'compacted.pikelet');
    assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...cCorpus.corpus, ingest: INGEST2 }, index: {},
        compactedFrom: { identity: OLD_HEAD, depth: 1, liveRecords: 6, tombstones: 2 },
    }, [
        { kind: 'index', bytes: sketchFor(cIds) },
        { kind: 'corpus', bytes: cCorpus.bytes },
        { kind: 'query-interp', bytes: cQi },
        { kind: 'lineage', bytes: lineageBytes },
    ], cPath);

    const reader = await openPikeletFile(cPath);
    check('info() reports compactedFrom on a compacted base',
        reader.info().compactedFrom?.identity === OLD_HEAD,
        JSON.stringify(reader.info().compactedFrom));
    const surfaced = await reader.lineage();
    check('lineage() surfaces both heads in depth order',
        !!surfaced && surfaced.heads.length === 2 && surfaced.heads[0].identity === OLD_BASE
        && surfaced.heads[1].identity === OLD_HEAD && surfaced.oldRowTotal === 8,
        JSON.stringify(surfaced && surfaced.heads.map((h) => h.depth)));
    // A reader that does not read the segment returns null from citation() for
    // a pack that HAS one, so every check below would throw on null rather
    // than report. `?? {}` keeps the failure legible as a FAIL line.
    const cite = async (ident, id) => (await reader.citation(ident, id)) ?? {};

    // A live id loses one position per tombstone below it (bits 1 and 2 set).
    const t0 = await cite(OLD_BASE, 0);
    check('a live id below every tombstone translates unshifted',
        t0.ok === true && t0.newId === 0, JSON.stringify(t0));
    const t3 = await cite(OLD_BASE, 3);
    check('a live id above two tombstones shifts down by two',
        t3.ok === true && t3.newId === 1, JSON.stringify(t3));
    const t7 = await cite(OLD_HEAD, 7);
    check('a layer-owned id translates against the depth-1 head',
        t7.ok === true && t7.newId === 5, JSON.stringify(t7));
    // The head table is what makes these two differ for the same id.
    const at6Base = await cite(OLD_BASE, 6);
    const at6Head = await cite(OLD_HEAD, 6);
    check('an id that postdates the cited head is refused at that head but not at a later one',
        at6Base.ok === false && at6Base.reason === 'id-postdates-that-head' && at6Head.ok === true,
        `${JSON.stringify(at6Base)} vs ${JSON.stringify(at6Head)}`);
    const tomb = await cite(OLD_HEAD, 1);
    check('a tombstoned id with no successor answers deleted',
        tomb.ok === false && tomb.reason === 'deleted', JSON.stringify(tomb));
    // Old id 2 was superseded by old id 6, which survives as dense id 4. The
    // walk and the dense translation both have to happen, so this pins the
    // whole shape rather than just "answered something".
    const sup = await cite(OLD_HEAD, 2);
    check('a superseded id translates through its successor to a dense id',
        sup.ok === true && sup.newId === 4 && sup.via === 'superseded'
        && JSON.stringify(sup.supersededChain) === '[6]',
        JSON.stringify(sup));
    const foreign = await cite(id64(77), 0);
    check('an identity absent from the head table is not on this history',
        foreign.ok === false && foreign.reason === 'not-on-this-history', JSON.stringify(foreign));
    await reader.close();

    // A pack with no lineage segment must answer null, not throw.
    const plainPath = path.join(linTmp, 'plain.pikelet');
    const pCorpus = buildCorpusSegment([0, 1, 2].map(recBytes));
    assemblePikeletFile({
        profile: PROFILE_V2, dim: CDIM, metric: 'cosine', encoder: ENC2,
        corpus: { ...pCorpus.corpus, ingest: INGEST2 }, index: {},
    }, [
        { kind: 'index', bytes: sketchFor([0, 1, 2]) },
        { kind: 'corpus', bytes: pCorpus.bytes },
        { kind: 'query-interp', bytes: buildQueryInterpSegment(2, Buffer.from(JSON.stringify({ dim: CDIM })), Buffer.from(JSON.stringify({ kind: 'none' }))) },
    ], plainPath);
    const plain = await openPikeletFile(plainPath);
    check('a pack that was never compacted answers null rather than throwing',
        (await plain.citation(OLD_BASE, 0)) === null && (await plain.lineage()) === null);
    check('info().compactedFrom is null on a pack that was never compacted',
        plain.info().compactedFrom === null);
    await plain.close();
    fs.rmSync(linTmp, { recursive: true, force: true });
}

console.log('mcp: verify_pack on a chain runs the base\'s tests and says so');
{
    const { callVerifyPack } = await import('../packages/pikelet/src/mcp.mjs');
    const baseId = 'b'.repeat(64);
    const packs = new Map([['docs', {
        chain: true, identity: 'c'.repeat(64),
        search: {
            evaluation: async () => ({
                evaluationScope: 'base', evaluationIdentity: baseId,
                goldenQueries: [{ text: 'kept', expectId: 1 }, { text: 'gone', expectId: 2 }],
                calibration: { cvAuc: 0.9, cvAucHard: 0.8, fitAuc: 0.95 },
            }),
            // A layer tombstoned record 2.
            record: async (id) => ({ id, tombstoned: id === 2 }),
            query: async () => ({ matchQuality: 'strong', results: [{ id: 1 }] }),
            info: () => ({
                encoderVerified: true, corpusIntegrity: 'per-record-sha256', indexRowIntegrity: 'per-row-sha256',
                members: [{ depth: 0 }, { depth: 1, tombstoneOnly: true }],
                calibrationStatus: 'inherited', calibrationDrift: 0.01, driftLimit: 0.2,
            }),
        },
    }]]);
    const out = typeof callVerifyPack === 'function' ? await callVerifyPack(packs, {}) : { goldenQueries: {} };
    check('a golden whose expected record a layer removed is skipped, not failed',
        out.verdict === 'pass' && out.goldenQueries.total === 1 && out.goldenQueries.passed === 1
        && out.goldenQueries.skipped?.length === 1 && /record 2 was removed or superseded/.test(out.goldenQueries.skipped[0].reason),
        JSON.stringify(out.goldenQueries));
    check('verify_pack on a chain names the tests as the base\'s and carries its calibration figures',
        out.chain?.testsFrom?.scope === 'base' && out.chain.testsFrom.identity === baseId
        && out.chain.calibrationStatus === 'inherited' && out.calibration?.cvAucHard === 0.8
        && /base pack's tests run against the whole chain/.test(out.note || ''),
        JSON.stringify({ chain: out.chain, note: out.note }));
}

console.log('mcp: a mounted chain embeds queries with the query prefix');
{
    // A real kind-3 base from the packaged inline MiniLM, compiled with an
    // asymmetric prefix. The MCP chain adapter embedded queries through the
    // passage side (embed), which on e5/arctic/bge puts every query in the
    // wrong space; with a prefix set the two sides give different vectors, so
    // the served distance says which one was used.
    const { execFileSync } = await import('node:child_process');
    const { PassThrough } = await import('node:stream');
    const { runMcpServer } = await import('../packages/pikelet/src/mcp.mjs');
    const { openPikeletFile } = await import('../packages/pikelet-wasm/complete/index.mjs');
    const { openPikeletChain } = await import('../packages/pikelet-wasm/complete/layer-reader.mjs');
    const mt = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-mcp-chain-'));
    const docs = path.join(mt, 'docs');
    fs.mkdirSync(docs);
    // Chunking drops documents under 25 whitespace tokens.
    const body = (title, line) => `# ${title}\n\n${Array(4).fill(line).join('\n')}\n`;
    fs.writeFileSync(path.join(docs, 'tides.md'), body('Tides', 'The moon pulls the ocean and makes tides twice a day.'));
    fs.writeFileSync(path.join(docs, 'volcanoes.md'), body('Volcanoes', 'Magma rises through the crust and erupts as lava.'));
    fs.writeFileSync(path.join(docs, 'bread.md'), body('Bread', 'Yeast ferments sugar so dough rises before baking.'));
    const basePath = path.join(mt, 'base.pikelet');
    execFileSync(process.execPath, [path.resolve('packages/pikelet/bin/pikelet.mjs'), 'compile',
        '--source', docs, '--out', basePath, '--skip-calibration', '--encoder-query-prefix', 'query: '],
    { stdio: ['ignore', 'pipe', 'pipe'] });
    const readManifestOf = (p) => {
        const buf = fs.readFileSync(p);
        return JSON.parse(buf.subarray(64, 64 + buf.readUInt32LE(8)).toString('utf8'));
    };
    const bm = readManifestOf(basePath);
    check('compile records its ingestion declaration as corpus.ingest (6.1)',
        canonicalJson(bm.corpus.ingest ?? null)
            === canonicalJson({ chunker: 'pikelet-section-v1', targetTokens: 256, overlapPercent: 15, minTokens: 25 }),
        JSON.stringify(bm.corpus.ingest));
    const baseRef = await openPikeletFile(basePath);
    const baseId = baseRef.info().identity;
    await baseRef.close();
    // A tombstone-only layer: the chain needs a member below the head for
    // the shelf lineage, and this one needs no encoder to build.
    const layerPath = path.join(mt, 'l1.pikelet');
    const layerBuilt = assemblePikeletFile({
        profile: PROFILE_LAYER, dim: bm.dim, metric: bm.metric, encoder: bm.encoder,
        layer: {
            parent: { identity: baseId, locator: 'base.pikelet' },
            baseIdentity: baseId, depth: 1, rowBase: bm.corpus.records, records: 0,
            tombstones: 1, supersessions: 0,
            // compile records no corpus.ingest, so depth 1 asserts one (6.1).
            ...(bm.corpus.ingest ? { ingest: bm.corpus.ingest } : { ingest: { chunker: 'test' }, ingestAsserted: true }),
        },
        corpus: { records: 0 },
    }, [
        { kind: 'query-interp', bytes: buildInheritedQuerySegment({
            baseIdentity: baseId,
            queryInterpSha256: bm.segments.find((x) => x.kind === 'query-interp').sha256,
        }) },
        { kind: 'tombstones', bytes: buildTombstoneSegment({ rowBase: bm.corpus.records, tombstonedIds: [2], records: 0 }) },
    ], layerPath);

    const QUERY = 'why does the sea rise and fall';
    const direct = await openPikeletChain([basePath, layerPath]);
    const enc = await direct.__head.passageEmbedder();
    const asQuery = await direct.query(await enc.embedQuery(QUERY), 1, { text: QUERY });
    const asPassage = await direct.query(await enc.embed(QUERY), 1, { text: QUERY });
    await direct.close();
    check('the fixture separates the two sides of the prefix contract',
        asQuery.results[0].distance !== asPassage.results[0].distance,
        `${asQuery.results[0].distance} vs ${asPassage.results[0].distance}`);

    const stdin = new PassThrough();
    const stdout = new PassThrough();
    const replies = [];
    let buffer = '';
    stdout.on('data', (d) => {
        buffer += d;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
            replies.push(JSON.parse(buffer.slice(0, idx)));
            buffer = buffer.slice(idx + 1);
        }
    });
    const done = runMcpServer({
        packPaths: [{ location: layerPath, identity: layerBuilt.identity, lineage: [{ location: basePath, identity: baseId }] }],
        openPikeletFile,
        httpRangeSource: () => { throw new Error('no network'); },
        stdin, stdout, log: () => {},
    });
    const waitFor = (id) => new Promise((resolve, reject) => {
        const deadline = Date.now() + 60000;
        const timer = setInterval(() => {
            const hit = replies.find((r) => r.id === id);
            if (hit) { clearInterval(timer); resolve(hit); }
            else if (Date.now() > deadline) { clearInterval(timer); reject(new Error(`no reply for ${id}`)); }
        }, 5);
    });
    try {
        stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'search', arguments: { query: QUERY, k: 1 } } })}\n`);
        const reply = (await waitFor(1)).result;
        const served = JSON.parse(reply.content[0].text).sections[0].results[0];
        check('MCP search over a chain serves the query-prefixed distance',
            served?.distance === asQuery.results[0].distance,
            `served ${served?.distance}, query-side ${asQuery.results[0].distance}, passage-side ${asPassage.results[0].distance}`);
    } finally {
        stdin.end();
        await done;
    }

    // 7: "Entries without lineage are resolved by locators." `--pack <layer>`
    // was refused as an unsupported profile; the head alone now mounts the
    // chain by walking its parent locator, pinned or not.
    const serveOnce = async (packPaths, args) => {
        const sin = new PassThrough();
        const sout = new PassThrough();
        let buf = '';
        const got = [];
        sout.on('data', (d) => {
            buf += d;
            let at;
            while ((at = buf.indexOf('\n')) >= 0) { got.push(JSON.parse(buf.slice(0, at))); buf = buf.slice(at + 1); }
        });
        const run = runMcpServer({
            packPaths, openPikeletFile,
            httpRangeSource: () => { throw new Error('no network'); },
            stdin: sin, stdout: sout, log: () => {},
        });
        const failed = run.then(() => null, (err) => err);
        sin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'search', arguments: args } })}\n`);
        sin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'list_packs', arguments: {} } })}\n`);
        sin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'verify_pack', arguments: {} } })}\n`);
        const deadline = Date.now() + 60000;
        while (!(got.some((r) => r.id === 1) && got.some((r) => r.id === 2) && got.some((r) => r.id === 3)) && Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 5));
        }
        sin.end();
        const err = await failed;
        return { err, search: got.find((r) => r.id === 1)?.result, packs: got.find((r) => r.id === 2)?.result, verify: got.find((r) => r.id === 3)?.result };
    };
    try {
        for (const [label, loc] of [['bare', layerPath], ['pinned', `${layerPath}#${layerBuilt.identity}`]]) {
            const out = await serveOnce([loc], { query: QUERY, k: 1 });
            // A failed mount answers tool calls with plain error text, not JSON.
            const parse = (r) => { try { return JSON.parse(r.content[0].text); } catch { return { text: r?.content?.[0]?.text }; } };
            const sp = out.search ? parse(out.search) : null;
            const lp = out.packs ? parse(out.packs) : null;
            const sec = sp?.sections?.[0];
            const listed = lp?.packs?.[0];
            check(`mcp --pack <layer> (${label}) mounts the chain by walking its locator`,
                !out.err && sec?.results?.[0]?.distance === asQuery.results[0].distance
                && listed?.identity === layerBuilt.identity,
                out.err ? out.err.message.slice(0, 140)
                    : sp?.text ? String(sp.text).slice(0, 140)
                        : JSON.stringify({ d: sec?.results?.[0]?.distance, id: listed?.identity?.slice(0, 12) }));
            if (label === 'bare') {
                // The base was compiled --skip-calibration, so it carries no
                // tests; the chain must still report its members, encoder and
                // integrity state, and must not blame an "older build".
                const vp = out.verify ? parse(out.verify) : null;
                check('verify_pack on a chain reports its members, encoder and integrity state',
                    vp?.chain?.members?.length === 2 && vp.chain.members[1].tombstoneOnly === true
                    && vp.encoderVerified === true && typeof vp.corpusIntegrity === 'string'
                    && typeof vp.indexRowIntegrity === 'string',
                    JSON.stringify(vp && { chain: vp.chain, encoderVerified: vp.encoderVerified, ci: vp.corpusIntegrity }).slice(0, 200));
                check('verify_pack on an untested chain says so without blaming an older build',
                    vp?.verdict === 'no-tests' && /chain-level tests are not assembled yet/.test(vp.note || '')
                    && !/older build/.test(vp.note || ''),
                    String(vp?.note || vp?.text).slice(0, 160));
            }
        }
    } finally {
        fs.rmSync(mt, { recursive: true, force: true });
    }
}

console.log(`\nLayered profile conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
