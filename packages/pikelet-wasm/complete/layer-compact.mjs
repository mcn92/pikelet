// `compact` and the lineage segment — LAYERED_PROFILE.md 6.3.
//
// Compaction is SEMANTIC maintenance, not only physical consolidation. It is the
// step that restores every exact property the layers approximated: exact lexical
// statistics over the live set (5.4), a vocabulary bloom over the live
// vocabulary only (4.5), a calibration fit against the corpus actually served
// (5.5), no tombstones, no supersession walks, depth zero, and reclaimed id
// space (3.3). The drift and depth limits exist to force it to happen.
//
// What it preserves, and what it deliberately does not:
//
//   Corpus meaning and provenance are preserved exactly — the compacted base's
//   records are the old head's live records, in ascending old-id order, bytes
//   unchanged. RANKINGS ARE NOT, and are not meant to be: layered BM25 scored
//   with statistics that counted tombstoned documents, compaction scores with
//   exact live statistics. "If the layered approximations had any observable
//   effect, a compaction that reproduced the chain's rankings would have failed
//   at its job." Conformance therefore tests corpus preservation and retrieval
//   correctness separately, and records rankings before and after as a
//   comparison, never an equality.

import { testBit, popcountBytes, bitsetBytesFor } from './tombstones.mjs';

export const LINEAGE_HEADER_BYTES = 64;
export const LINEAGE_HEAD_ENTRY_BYTES = 48;
export const LINEAGE_VERSION = 1;
// Depth is at most 16, so the head table holds at most 17 entries — the old
// base at depth 0 through the old head.
export const MAX_HEADS = 17;

const isHex64 = (v) => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v);
const hexToBytes = (hex) => {
    const out = new Uint8Array(32);
    for (let i = 0; i < 32; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    return out;
};
const bytesToHex = (bytes) => [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');

/**
 * Plan a compaction: which ids survive, and the dense renumbering.
 *
 * Live ids in ascending global order; survivors renumbered densely in the same
 * order (step 1-2 of 6.3). The translation is positional, which is what makes
 * compaction byte-deterministic given the chain.
 */
export function planCompaction({ rowTotal, headBitset }) {
    if (!Number.isSafeInteger(rowTotal) || rowTotal < 0) throw new Error('compact: rowTotal must be a non-negative safe integer');
    const liveIds = [];
    for (let id = 0; id < rowTotal; id++) if (!testBit(headBitset, id)) liveIds.push(id);
    if (!liveIds.length) throw new Error('compact: the chain has no live records');
    // oldId -> newId, dense, order-preserving.
    const forward = new Map();
    liveIds.forEach((oldId, newId) => forward.set(oldId, newId));
    return {
        liveIds,
        liveRecords: liveIds.length,
        tombstones: rowTotal - liveIds.length,
        newIdOf: (oldId) => (forward.has(oldId) ? forward.get(oldId) : null),
        forward,
    };
}

/**
 * Build a `lineage-v1` segment (kind 7), REQUIRED for a base compacted from a
 * chain.
 *
 * The HEAD TABLE lets a citation against any head of the compacted chain
 * translate, not only the last one. The previous compacted base is its
 * depth-0 entry, so retaining intermediate compacted bases gives multi-hop
 * translation with nothing further.
 *
 * @param {{heads: Array<{identity: string, rowTotal: number, depth: number}>,
 *          headBitset: Uint8Array, oldRowTotal: number,
 *          supersessions: Array<[number, number, number]>}} spec
 *   supersessions are (oldId, newId, depth) triples — every edge of the old
 *   chain. Each carries its depth, so the immediate/all/current views of 3.5 are
 *   computed exactly as on a live chain, with nothing inferred from ordering.
 */
export function buildLineageSegment({ heads, headBitset, oldRowTotal, supersessions = [] }) {
    if (!Array.isArray(heads) || heads.length < 1 || heads.length > MAX_HEADS) {
        throw new Error(`lineage: headCount must be in [1, ${MAX_HEADS}], got ${heads?.length}`);
    }
    heads.forEach((h, i) => {
        if (!isHex64(h.identity)) throw new Error(`lineage: head ${i} identity must be 64 lowercase hex characters`);
        if (h.depth !== i) throw new Error(`lineage: head table depths must be exactly 0..${heads.length - 1} in order; entry ${i} says ${h.depth}`);
        if (!Number.isSafeInteger(h.rowTotal) || h.rowTotal < 0) throw new Error(`lineage: head ${i} rowTotal is not a safe integer`);
        if (i > 0 && h.rowTotal < heads[i - 1].rowTotal) throw new Error('lineage: head table rowTotal values must be non-decreasing');
    });
    if (heads[heads.length - 1].rowTotal !== oldRowTotal) {
        throw new Error(`lineage: the last head's rowTotal ${heads[heads.length - 1].rowTotal} must equal oldRowTotal ${oldRowTotal}`);
    }
    if (new Set(heads.map((h) => h.identity)).size !== heads.length) {
        throw new Error('lineage: head table identities must be pairwise distinct');
    }

    const bitsetBytes = bitsetBytesFor(oldRowTotal);
    const bitset = new Uint8Array(bitsetBytes);
    bitset.set(headBitset.subarray(0, Math.min(headBitset.length, bitsetBytes)));
    const tombstoneCount = popcountBytes(bitset);

    const edges = [...supersessions].map(([oldId, newId, depth]) => {
        if (!(oldId < newId && newId < oldRowTotal)) {
            throw new Error(`lineage: edge (${oldId} -> ${newId}) must satisfy oldId < newId < oldRowTotal ${oldRowTotal}`);
        }
        if (!testBit(bitset, oldId)) throw new Error(`lineage: edge oldId ${oldId} is not tombstoned`);
        if (!Number.isInteger(depth) || depth < 1 || depth > heads.length - 1) {
            throw new Error(`lineage: edge depth ${depth} must be in [1, ${heads.length - 1}]`);
        }
        // newId must lie in the range the layer at that depth owned.
        const lo = heads[depth - 1].rowTotal;
        const hi = heads[depth].rowTotal;
        if (newId < lo || newId >= hi) {
            throw new Error(`lineage: edge newId ${newId} is outside the range [${lo}, ${hi}) the layer at depth ${depth} owned`);
        }
        return [oldId, newId, depth];
    });
    edges.sort((a, b) => (a[0] - b[0]) || (a[2] - b[2]));
    for (let i = 1; i < edges.length; i++) {
        if (edges[i][0] === edges[i - 1][0] && edges[i][2] === edges[i - 1][2]) {
            throw new Error(`lineage: (oldId ${edges[i][0]}, depth ${edges[i][2]}) appears twice`);
        }
    }

    const total = LINEAGE_HEADER_BYTES + LINEAGE_HEAD_ENTRY_BYTES * heads.length + bitsetBytes + edges.length * 12;
    const out = new Uint8Array(total);
    const view = new DataView(out.buffer);
    view.setUint32(0, LINEAGE_VERSION, true);
    view.setUint32(4, 0, true);
    view.setUint32(8, heads.length, true);
    view.setUint32(12, edges.length, true);
    view.setBigUint64(16, BigInt(oldRowTotal), true);
    view.setBigUint64(24, BigInt(tombstoneCount), true);
    let at = LINEAGE_HEADER_BYTES;
    for (const h of heads) {
        out.set(hexToBytes(h.identity), at);
        view.setBigUint64(at + 32, BigInt(h.rowTotal), true);
        view.setUint32(at + 40, h.depth, true);
        view.setUint32(at + 44, 0, true);
        at += LINEAGE_HEAD_ENTRY_BYTES;
    }
    out.set(bitset, at);
    at += bitsetBytes;
    for (const [oldId, newId, depth] of edges) {
        view.setUint32(at, oldId, true);
        view.setUint32(at + 4, newId, true);
        view.setUint32(at + 8, depth, true);
        at += 12;
    }
    return out;
}

/**
 * Parse and fully validate a lineage segment. A digest-valid lineage segment is
 * still untrusted input (section 9), so every rule of 6.3's validation list is
 * enforced here rather than assumed.
 *
 * @param {Uint8Array} bytes
 * @param {{identity: string, liveRecords: number, tombstones: number}} [compactedFrom]
 *   The manifest's compactedFrom block. When given, the segment must agree.
 */
export function parseLineageSegment(bytes, compactedFrom = null) {
    if (!(bytes instanceof Uint8Array)) throw new Error('lineage: segment must be a Uint8Array');
    if (bytes.length < LINEAGE_HEADER_BYTES) throw new Error(`lineage: segment is shorter than its ${LINEAGE_HEADER_BYTES}-byte header`);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(0, true);
    if (version !== LINEAGE_VERSION) throw new Error(`lineage: unsupported version ${version}`);
    const flags = view.getUint32(4, true);
    if (flags !== 0) throw new Error(`lineage: flags must be 0, got ${flags}`);
    const headCount = view.getUint32(8, true);
    if (headCount < 1 || headCount > MAX_HEADS) throw new Error(`lineage: headCount ${headCount} is outside [1, ${MAX_HEADS}]`);
    const supersessionCount = view.getUint32(12, true);
    const oldRowTotalBig = view.getBigUint64(16, true);
    const tombBig = view.getBigUint64(24, true);
    if (oldRowTotalBig > BigInt(Number.MAX_SAFE_INTEGER) || tombBig > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error('lineage: a u64 field exceeds MAX_SAFE_INTEGER');
    }
    const oldRowTotal = Number(oldRowTotalBig);
    const tombstoneCount = Number(tombBig);
    for (let off = 32; off < LINEAGE_HEADER_BYTES; off += 4) {
        if (view.getUint32(off, true) !== 0) throw new Error('lineage: reserved header bytes must be zero');
    }

    // The regions MUST tile the segment exactly, computed with subtraction.
    const bitsetBytes = bitsetBytesFor(oldRowTotal);
    const headBytes = LINEAGE_HEAD_ENTRY_BYTES * headCount;
    const need = headBytes + bitsetBytes + supersessionCount * 12;
    if (bytes.length - LINEAGE_HEADER_BYTES !== need) {
        throw new Error(`lineage: regions do not tile the segment (${bytes.length - LINEAGE_HEADER_BYTES} bytes after the header, need ${need})`);
    }

    const heads = [];
    let at = LINEAGE_HEADER_BYTES;
    for (let i = 0; i < headCount; i++) {
        const identity = bytesToHex(bytes.subarray(at, at + 32));
        const rowTotalBig = view.getBigUint64(at + 32, true);
        if (rowTotalBig > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('lineage: a head rowTotal exceeds MAX_SAFE_INTEGER');
        const rowTotal = Number(rowTotalBig);
        const depth = view.getUint32(at + 40, true);
        if (view.getUint32(at + 44, true) !== 0) throw new Error(`lineage: head ${i} reserved word must be zero`);
        if (depth !== i) throw new Error(`lineage: head table depths must be exactly 0..${headCount - 1} in order; entry ${i} says ${depth}`);
        if (i > 0 && rowTotal < heads[i - 1].rowTotal) throw new Error('lineage: head table rowTotal values must be non-decreasing');
        heads.push({ identity, rowTotal, depth });
        at += LINEAGE_HEAD_ENTRY_BYTES;
    }
    if (heads[headCount - 1].rowTotal !== oldRowTotal) {
        throw new Error(`lineage: the last head's rowTotal ${heads[headCount - 1].rowTotal} must equal oldRowTotal ${oldRowTotal}`);
    }
    if (new Set(heads.map((h) => h.identity)).size !== headCount) {
        throw new Error('lineage: head table identities must be pairwise distinct');
    }

    const bitset = bytes.subarray(at, at + bitsetBytes);
    at += bitsetBytes;
    const padBits = bitsetBytes * 8 - oldRowTotal;
    if (padBits > 0 && bitsetBytes > 0) {
        const validMask = (1 << (8 - padBits)) - 1;
        if ((bitset[bitsetBytes - 1] & ~validMask & 0xff) !== 0) {
            throw new Error(`lineage: ${padBits} pad bit(s) above oldRowTotal are non-zero`);
        }
    }
    const actualTomb = popcountBytes(bitset);
    if (actualTomb !== tombstoneCount) {
        throw new Error(`lineage: header says ${tombstoneCount} tombstones, bitset popcount is ${actualTomb}`);
    }

    const supersessions = [];
    let prev = null;
    for (let i = 0; i < supersessionCount; i++) {
        const oldId = view.getUint32(at, true);
        const newId = view.getUint32(at + 4, true);
        const depth = view.getUint32(at + 8, true);
        at += 12;
        if (!(oldId < newId && newId < oldRowTotal)) {
            throw new Error(`lineage: edge (${oldId} -> ${newId}) must satisfy oldId < newId < oldRowTotal ${oldRowTotal}`);
        }
        if (!testBit(bitset, oldId)) throw new Error(`lineage: edge oldId ${oldId} is not tombstoned`);
        if (depth < 1 || depth > headCount - 1) throw new Error(`lineage: edge depth ${depth} must be in [1, ${headCount - 1}]`);
        const lo = heads[depth - 1].rowTotal;
        const hi = heads[depth].rowTotal;
        if (newId < lo || newId >= hi) {
            throw new Error(`lineage: edge newId ${newId} is outside the range [${lo}, ${hi}) the layer at depth ${depth} owned`);
        }
        if (prev && (oldId < prev[0] || (oldId === prev[0] && depth <= prev[2]))) {
            throw new Error('lineage: edges must be sorted by (oldId, depth) and unique');
        }
        prev = [oldId, newId, depth];
        supersessions.push([oldId, newId, depth]);
    }

    if (compactedFrom) {
        if (heads[headCount - 1].identity !== compactedFrom.identity) {
            throw new Error(`lineage: the last head ${heads[headCount - 1].identity.slice(0, 12)}… must equal the manifest's compactedFrom.identity ${String(compactedFrom.identity).slice(0, 12)}…`);
        }
        if (Number.isInteger(compactedFrom.tombstones) && compactedFrom.tombstones !== tombstoneCount) {
            throw new Error(`lineage: segment tombstoneCount ${tombstoneCount} disagrees with manifest compactedFrom.tombstones ${compactedFrom.tombstones}`);
        }
        if (Number.isInteger(compactedFrom.liveRecords) && compactedFrom.liveRecords !== oldRowTotal - tombstoneCount) {
            throw new Error(`lineage: segment implies ${oldRowTotal - tombstoneCount} live records, manifest compactedFrom says ${compactedFrom.liveRecords}`);
        }
    }

    return { version, headCount, heads, oldRowTotal, tombstoneCount, bitset, supersessions };
}

/**
 * `translate(identity, id)` by a reader of a compacted base — 6.3.
 *
 * This is FORWARD translation: it answers "what became of the record cited at
 * H3 by the time H8 was compacted", which is the question a citation asks of a
 * compacted base. It is computed from the FINAL pre-compaction state — the old
 * head's mask and the full edge set — regardless of which head was cited. It is
 * deliberately not historical state reconstruction: the head table does not make
 * it able to answer "what did H3 serve".
 *
 * @param {object} lineage parsed lineage segment
 * @param {Map<number, number>|((oldId:number)=>number|null)} compactMap oldId -> new dense id
 */
export function translateCitation(lineage, compactMap, identity, id) {
    const entry = lineage.heads.find((h) => h.identity === identity);
    if (!entry) return { ok: false, reason: 'not-on-this-history' };
    if (!Number.isSafeInteger(id) || id < 0) return { ok: false, reason: 'invalid-id' };
    // The id did not exist when that artifact was the head.
    if (id >= entry.rowTotal) return { ok: false, reason: 'id-postdates-that-head' };

    const lookup = typeof compactMap === 'function' ? compactMap : (oldId) => (compactMap.has(oldId) ? compactMap.get(oldId) : null);
    const live = lookup(id);
    if (live !== null && live !== undefined) {
        return { ok: true, newId: live, via: 'live', citedAtDepth: entry.depth };
    }

    // The cited record was tombstoned. Follow the edge set to the current
    // successor, by DEPTH, exactly as 3.5 prescribes: start from the
    // greatest-depth edge and keep following each reached id's greatest-depth
    // edge until one has none.
    const byOld = new Map();
    for (const [oldId, newId, depth] of lineage.supersessions) {
        if (!byOld.has(oldId)) byOld.set(oldId, []);
        byOld.get(oldId).push({ newId, depth });
    }
    for (const edges of byOld.values()) edges.sort((a, b) => a.depth - b.depth);

    let cursor = id;
    const seen = new Set([id]);
    for (let hops = 0; hops < MAX_HEADS + 1; hops++) {
        const edges = byOld.get(cursor);
        if (!edges || !edges.length) break;
        const next = edges[edges.length - 1].newId;
        if (seen.has(next)) break;
        seen.add(next);
        cursor = next;
        const mapped = lookup(cursor);
        if (mapped !== null && mapped !== undefined) {
            return { ok: true, newId: mapped, via: 'superseded', supersededChain: [...seen].slice(1), citedAtDepth: entry.depth };
        }
    }
    return { ok: false, reason: 'deleted', citedAtDepth: entry.depth };
}
