// `append` — LAYERED_PROFILE.md 6.1.
//
// Compiles one layer on top of a mounted chain: new records, a cumulative
// tombstone bitset, and a commitment to the parent's identity. The hard parts
// are not the bytes but the preconditions, so they are checked first and
// separately, and each refusal names the rule it enforces.
//
// Two things the complete profile never promised are required to append:
//
//   1. An encoder that can embed PASSAGES. A kind-3 base carries the teacher,
//      so a pack is sufficient to compile its own successors. A kind-2 base
//      names its encoder and the host supplies it. A KIND-1 base carries a
//      corpus-distilled QUERY encoder whose passages were embedded by a teacher
//      the artifact neither carries nor verifies — so a kind-1 base is readable
//      but NOT appendable, and append MUST refuse it.
//   2. An ingestion declaration, because record granularity is part of what the
//      calibration saw and MUST NOT change mid-chain. A base compiled after
//      this profile carries `corpus.ingest`; an older one does not, and append
//      MUST refuse unless the operator asserts one.
//
// This module does the planning and assembly that is testable in-process. It
// takes already-embedded vectors rather than embedding itself, so it stays
// environment-neutral and so a test can drive it without a 24 MiB encoder.

import { buildTombstoneSegment, zeroExtend, bitsetBytesFor, testBit, popcountBytes, MAX_RECORDS } from './tombstones.mjs';
import { buildInheritedQuerySegment } from './layer-encoder.mjs';
import { canonicalJson } from './layer-manifest.mjs';

// 6.1: the producer default is 8 — "an operational recommendation until the
// range-proof measurements of section 13 say otherwise; the format ceiling of
// 16 is enforced regardless". 13.5 still lists it as unmeasured.
export const PRODUCER_MAX_DEPTH = 8;
export const FORMAT_MAX_DEPTH = 16;

/**
 * Decide whether a base can be appended to, and how its encoder is obtained.
 * Separated from the assembly because it is the part with a normative answer
 * for every base ever released.
 *
 * @param {{qiKind: number, corpusIngest: object|null, assertIngest: object|null}} spec
 * @returns {{appendable: true, encoderSource: 'pack'|'host', ingest: object, ingestAsserted: boolean}}
 */
export function planAppendability({ qiKind, corpusIngest = null, assertIngest = null }) {
    if (qiKind === 1) {
        throw new Error('a kind-1 base carries a corpus-distilled query encoder whose passages were embedded by a teacher the artifact does not carry: readable, but not appendable (6.1)');
    }
    if (qiKind === 4) {
        throw new Error('a layer is not a base: append to the chain\'s head, whose base supplies the encoder (4.5)');
    }
    let encoderSource;
    if (qiKind === 3) encoderSource = 'pack';       // the pack carries its teacher
    else if (qiKind === 2) encoderSource = 'host';  // the host supplies the named encoder
    else throw new Error(`unsupported query-interpretation kind ${qiKind} for append`);

    if (corpusIngest !== null && corpusIngest !== undefined) {
        if (assertIngest) {
            throw new Error('the base declares corpus.ingest; --assert-ingest would override an artifact\'s own declaration (6.1)');
        }
        return { appendable: true, encoderSource, ingest: corpusIngest, ingestAsserted: false };
    }
    if (!assertIngest) {
        throw new Error('the base carries no corpus.ingest, so record granularity cannot be enforced; supply --assert-ingest to state it (6.1). The assertion is the operator\'s statement, not the artifact\'s.');
    }
    if (typeof assertIngest !== 'object' || Array.isArray(assertIngest)) {
        throw new Error('--assert-ingest must be a JSON object');
    }
    if ('ingestAsserted' in assertIngest) {
        throw new Error('the asserted declaration must not contain ingestAsserted: it is a sibling field, so asserting must not change the declaration\'s canonical form (4.1)');
    }
    return { appendable: true, encoderSource, ingest: assertIngest, ingestAsserted: true };
}

/**
 * Plan a layer's ids, mask and supersessions — step 4 and 5 of 6.1.
 *
 * @param {{parentRowBase: number, parentRecords: number, parentBitset: Uint8Array,
 *          newRecordCount: number, remove?: number[], supersede?: Array<[number, string|number]>,
 *          maxDepth?: number, parentDepth: number}} spec
 */
export function planLayer({
    parentRowBase, parentRecords, parentBitset,
    newRecordCount, remove = [], supersede = [],
    parentDepth, maxDepth = PRODUCER_MAX_DEPTH,
}) {
    const depth = parentDepth + 1;
    if (depth > FORMAT_MAX_DEPTH) {
        throw new Error(`appending would reach depth ${depth}, past the format ceiling of ${FORMAT_MAX_DEPTH} (2.7)`);
    }
    if (depth > maxDepth) {
        throw new Error(`appending would reach depth ${depth}, past --max-depth ${maxDepth}; compact the chain instead (6.1)`);
    }
    const rowBase = parentRowBase + parentRecords;
    if (!Number.isSafeInteger(rowBase) || rowBase < 0) throw new Error('parent row arithmetic is not a safe integer');
    // 3.3: a producer MUST refuse an append whose rowBase + records would
    // exceed the id ceiling. There is no other mechanism.
    if (rowBase + newRecordCount > MAX_RECORDS) {
        throw new Error(`appending ${newRecordCount} records at rowBase ${rowBase} would exhaust the id space (${MAX_RECORDS}); compact the chain (3.3)`);
    }

    // Step 5: the cumulative bitset is zeroExtend(T_parent, rowBase), then
    // every --remove id and every --supersede old id.
    const inherited = zeroExtend(parentBitset, rowBase);
    const bitset = new Uint8Array(bitsetBytesFor(rowBase));
    bitset.set(inherited.subarray(0, bitset.length));
    const introduced = [];
    const setOne = (id, what) => {
        if (!Number.isSafeInteger(id) || id < 0) throw new Error(`${what} id ${id} is not a valid id`);
        // A layer cannot retract its own records: ids at or above rowBase are
        // out of range by construction (3.4).
        if (id >= rowBase) {
            throw new Error(`${what} id ${id} is at or above this layer's rowBase ${rowBase}: a layer cannot retract its own records (3.4)`);
        }
        if (testBit(bitset, id)) return false;   // already deleted is a no-op
        bitset[id >> 3] |= 1 << (id & 7);
        introduced.push(id);
        return true;
    };
    for (const id of remove) setOne(id, '--remove');

    // Each pair is [oldId, target]. A numeric target is a local index into
    // this layer's records, resolved by the producer from the chunks the
    // --supersede path produced. A non-numeric target (a path the producer
    // could not resolve) falls back to positional assignment.
    const supersessions = [];
    let nextNewId = rowBase;
    const claimed = new Set();
    for (const [oldId, target] of supersede) {
        let newId;
        if (Number.isSafeInteger(target)) {
            if (target < 0 || target >= newRecordCount) {
                throw new Error(`--supersede resolved to local record ${target}, outside this layer's ${newRecordCount} record(s) (3.5)`);
            }
            newId = rowBase + target;
            if (claimed.has(newId)) {
                throw new Error(`--supersede maps two old ids onto new record ${newId}: each new record supersedes at most one predecessor (3.5)`);
            }
        } else {
            while (claimed.has(nextNewId)) nextNewId += 1;
            if (nextNewId >= rowBase + newRecordCount) {
                throw new Error('more --supersede pairs than new records: every newId must lie in this layer\'s own id range (3.5)');
            }
            newId = nextNewId;
            nextNewId += 1;
        }
        setOne(oldId, '--supersede');
        supersessions.push([oldId, newId]);
        claimed.add(newId);
    }
    // 3.5: within one layer each oldId appears at most once.
    const seen = new Set();
    for (const [oldId] of supersessions) {
        if (seen.has(oldId)) throw new Error(`--supersede names id ${oldId} more than once: an oldId has at most one edge per layer (3.5)`);
        seen.add(oldId);
    }

    // 6.1: "A layer must do something." A layer that introduces neither a
    // record nor a new tombstone would still lengthen every chain carrying it,
    // cost a mount round trip and count against the depth limits, for no
    // corpus meaning. This is also why 4.3 requires a tombstone-only layer's
    // tombstoneCount to EXCEED its parent's.
    if (newRecordCount === 0 && introduced.length === 0) {
        throw new Error('refusing to emit a layer that introduces neither a record nor a new tombstone (6.1): --remove of already-deleted ids alone produces nothing');
    }

    return {
        depth, rowBase, records: newRecordCount,
        bitset, tombstones: popcountBytes(bitset),
        delta: introduced.sort((a, b) => a - b),
        supersessions,
        tombstoneOnly: newRecordCount === 0,
    };
}

/**
 * Assemble the layer's manifest fields and its tombstone + kind-4 segments.
 * The caller supplies the index/corpus/lexical segment bytes, since building
 * those is the existing builder's job.
 */
export function buildLayerSegments({ plan, baseIdentity, parentIdentity, baseQueryInterpSha256, parentLocator = null, ingest, ingestAsserted, dim, metric, encoder, corpus, layerBloom = null }) {
    const tombstones = buildTombstoneSegment({
        rowBase: plan.rowBase,
        tombstonedIds: idsSetIn(plan.bitset, plan.rowBase),
        supersessions: plan.supersessions,
        records: plan.records,
    });
    const queryInterp = buildInheritedQuerySegment({
        baseIdentity, queryInterpSha256: baseQueryInterpSha256,
        vocabBloom: layerBloom ? layerBloom.geometry : null,
        vocabBloomBytes: layerBloom ? layerBloom.bytes : null,
    });
    const manifestFields = {
        profile: 'pikelet-layer-v1',
        dim, metric, encoder,
        layer: {
            parent: { identity: parentIdentity, ...(parentLocator ? { locator: parentLocator } : {}) },
            baseIdentity,
            depth: plan.depth,
            rowBase: plan.rowBase,
            records: plan.records,
            tombstones: plan.tombstones,
            supersessions: plan.supersessions.length,
            ingest,
            // 4.1: a SIBLING of layer.ingest, never a member — asserting must
            // not change the declaration's canonical form.
            ...(ingestAsserted ? { ingestAsserted: true } : {}),
        },
        // A tombstone-only layer still carries a corpus object, with records 0,
        // and omits the layout fields (4.1).
        corpus: plan.tombstoneOnly ? { records: 0, ...(corpus?.provenance ? { provenance: corpus.provenance } : {}) } : corpus,
        // A layer with records carries an index object; assemblePikeletFile
        // fills in headerSha256 from the sketch bytes, but the object must be
        // present for the manifest to be a valid layer manifest (4.1). A
        // tombstone-only layer omits it, having no index segment (4.3).
        ...(plan.tombstoneOnly ? {} : { index: {} }),
    };
    return { manifestFields, tombstones, queryInterp };
}

/**
 * The drift an append would produce, and whether the producer may emit it.
 * 6.1: append MUST refuse a layer whose drift would exceed the chain's
 * effective limit unless --allow-drift is given.
 */
export function checkAppendDrift({ baseRecords, appendedBefore, newRecords, headTombstones, effectiveLimit, allowDrift = false }) {
    if (!Number.isSafeInteger(baseRecords) || baseRecords <= 0) {
        throw new Error('drift needs a base with at least one record');
    }
    const drift = (appendedBefore + newRecords + headTombstones) / baseRecords;
    if (drift > effectiveLimit && !allowDrift) {
        throw new Error(`this append would take calibrationDrift to ${drift.toFixed(4)}, past the chain's effective limit ${effectiveLimit}; pass --allow-drift to emit it anyway (the chain then serves unscored) (6.1)`);
    }
    return { drift, exceedsLimit: drift > effectiveLimit };
}

/** Every id set in a bitset, ascending. */
export function idsSetIn(bitset, rowBase) {
    const out = [];
    for (let id = 0; id < rowBase; id++) if (testBit(bitset, id)) out.push(id);
    return out;
}

export { canonicalJson };
