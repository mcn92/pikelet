// `rebase` — LAYERED_PROFILE.md 6.2.
//
// Two producers appending to the same parent create a fork. Both chains are
// valid; the ref names one head. The other layer is re-based onto it: its rows,
// sketches, records, postings and bloom are copied verbatim (nothing is
// re-embedded, local ids do not change), and its rowBase, bitset, supersessions
// and evaluation material are recomputed.
//
// The correctness rule:
//
//   A rebase replays what the layer ITSELF did -- its delta over its old
//   parent, and the edges physically in its own segment -- and NEVER its
//   cumulative bitset. T_L contains every deletion any ancestor on the old
//   branch performed; replaying it would import those ancestors' operations
//   into a branch that never performed them, and worse, import an ancestor's
//   tombstone without that ancestor's supersession edge: half of one operation.
//
// To compute the delta, the rebase MUST open L's ORIGINAL parent. If that
// artifact is unavailable the delta is unknowable and the rebase must refuse --
// it cannot be reconstructed from L alone.

import { zeroExtend, bitsetBytesFor, testBit, popcountBytes } from './tombstones.mjs';
import { canonicalJson } from './layer-manifest.mjs';

export const ON_CONFLICT = Object.freeze({ REFUSE: 'refuse', SKIP: 'skip', KEEP_BOTH: 'keep-both' });
export const ON_FOREIGN = Object.freeze({ REFUSE: 'refuse', DROP: 'drop' });

/**
 * The fork point: the longest common prefix of L's original parent chain and
 * H's chain, compared layer by layer from the base BY IDENTITY, and the total
 * row count of that prefix.
 *
 * Every id below forkRowBase names the same record in both histories. Every id
 * at or above it belongs to a layer the two histories do not share — fork
 * siblings assign overlapping id ranges to different records, so such an id is
 * foreign.
 *
 * @param {Array<{identity: string, records: number}>} oldChain base-first, L's ancestors
 * @param {Array<{identity: string, records: number}>} newChain base-first, H's chain
 */
export function forkPoint(oldChain, newChain) {
    let i = 0;
    let forkRowBase = 0;
    while (i < oldChain.length && i < newChain.length
           && oldChain[i].identity === newChain[i].identity) {
        forkRowBase += oldChain[i].records;
        i += 1;
    }
    if (i === 0) {
        throw new Error('the two histories share no base: a rebase requires H.baseIdentity == L.baseIdentity (6.2)');
    }
    return { sharedPrefix: i, forkRowBase };
}

/**
 * Preconditions a rebase must satisfy before it constructs any bytes (6.2).
 * They are producer preconditions precisely so a rebase never emits a layer a
 * reader would refuse.
 */
export function checkRebasePreconditions({ layerManifest, headChainDecl, layerChainDecl, headIsLegacyBase = false, headBaseIdentity, layerBaseIdentity }) {
    if (headBaseIdentity !== layerBaseIdentity) {
        throw new Error(`rebase refuses: the head's base ${String(headBaseIdentity).slice(0, 12)}… is not the layer's base ${String(layerBaseIdentity).slice(0, 12)}… (6.2)`);
    }
    // Same base is NOT enough. Two depth-1 forks of a legacy base can each have
    // asserted a DIFFERENT declaration, and copying L's layer.ingest onto H's
    // history would emit a layer a reader must reject.
    if (headIsLegacyBase) {
        // H is the legacy depth-0 base itself and has no declaration yet, so a
        // rebased depth-1 layer MAY introduce L's asserted declaration.
        if (layerChainDecl && !layerChainDecl.asserted) {
            throw new Error('rebase refuses: the head is a legacy base with no declaration, but the layer\'s declaration is not asserted (6.2)');
        }
        return { ingest: layerManifest.layer.ingest, ingestAsserted: true };
    }
    if (!headChainDecl || !layerChainDecl) {
        throw new Error('rebase refuses: an ingestion declaration is missing on one of the two histories (6.1, 6.2)');
    }
    if (headChainDecl.canonical !== layerChainDecl.canonical) {
        throw new Error('rebase refuses: the two histories disagree on the chain ingestion declaration; a copied layer.ingest would be rejected by a reader (6.2)');
    }
    return { ingest: layerManifest.layer.ingest, ingestAsserted: headChainDecl.asserted };
}

/**
 * Plan a rebase.
 *
 * @param {object} spec
 * @param {Uint8Array} spec.layerBitset        T_L, the layer's own cumulative bitset
 * @param {Uint8Array} spec.oldParentBitset    T_oldParent — REQUIRED; the delta is
 *   unknowable without it, and a rebase must refuse rather than guess.
 * @param {number} spec.oldRowBase             L.layer.rowBase
 * @param {number} spec.layerRecords           R_L
 * @param {Array<[number, number]>} spec.layerSupersessions  S_L, physically recorded in L
 * @param {number} spec.newRowBase             H.rowBase + H.records
 * @param {Uint8Array} spec.headBitset         T_H
 * @param {Map<number, Array<{depth:number,newId:number}>>} spec.headSupersessions  S_H
 * @param {number} spec.forkRowBase
 * @param {'refuse'|'drop'} [spec.onForeign]
 * @param {'refuse'|'skip'|'keep-both'} [spec.onConflict]
 */
export function planRebase({
    layerBitset, oldParentBitset, oldRowBase, layerRecords, layerSupersessions = [],
    newRowBase, headBitset, headSupersessions = new Map(), forkRowBase,
    onForeign = ON_FOREIGN.REFUSE, onConflict = ON_CONFLICT.REFUSE,
}) {
    if (!(oldParentBitset instanceof Uint8Array)) {
        throw new Error('rebase refuses: the layer\'s original parent is unavailable, so its delta over that parent is unknowable (6.2)');
    }
    if (!Object.values(ON_FOREIGN).includes(onForeign)) throw new Error(`--on-foreign must be one of ${Object.values(ON_FOREIGN).join(', ')}`);
    if (!Object.values(ON_CONFLICT).includes(onConflict)) throw new Error(`--on-conflict must be one of ${Object.values(ON_CONFLICT).join(', ')}`);

    // ΔT_L = T_L \ zeroExtend(T_oldParent, L.rowBase) — never T_L as a whole.
    const inheritedOld = zeroExtend(oldParentBitset, oldRowBase);
    const delta = [];
    for (let id = 0; id < oldRowBase; id++) {
        if (testBit(layerBitset, id) && !testBit(inheritedOld, id)) delta.push(id);
    }

    // Two kinds of id, treated oppositely (6.2). Ids L OWNS — its records and
    // the y side of every edge — are offsets into its own range and must be
    // translated. Ids L REFERENCES — ΔT_L and the x side — are ancestor ids,
    // stable only below the fork point.
    const translateY = (oldY) => newRowBase + (oldY - oldRowBase);

    // Foreign ids: at or above forkRowBase they belong to a layer the two
    // histories do not share. The test applies to ΔT_L, not to T_L: bits L
    // inherited from its old branch are not L's to carry anywhere.
    const foreignDeletions = delta.filter((x) => x >= forkRowBase);
    const foreignEdges = layerSupersessions.filter(([x]) => x >= forkRowBase);
    if ((foreignDeletions.length || foreignEdges.length) && onForeign === ON_FOREIGN.REFUSE) {
        const ids = [...new Set([...foreignDeletions, ...foreignEdges.map(([x]) => x)])].sort((a, b) => a - b);
        throw new Error(`rebase refuses: ${ids.length} foreign id(s) at or above the fork point ${forkRowBase} name different records in the two histories: ${ids.join(', ')} (pass --on-foreign drop to discard them) (6.2)`);
    }
    const survivingDeletions = delta.filter((x) => x < forkRowBase);
    const survivingEdges = layerSupersessions.filter(([x]) => x < forkRowBase);
    const droppedForeign = { deletions: foreignDeletions, edges: foreignEdges.map(([x]) => x) };

    // Conflicts, deterministically (6.2).
    const conflicts = [];
    const recordedEdges = [];
    const skippedEdges = [];
    for (const [x, oldY] of survivingEdges) {
        const newY = translateY(oldY);
        // Every newId a rebased layer writes MUST lie in its new range.
        if (newY < newRowBase || newY >= newRowBase + layerRecords) {
            throw new Error(`rebase refuses: translated newId ${newY} for oldId ${x} falls outside the layer's new range [${newRowBase}, ${newRowBase + layerRecords}) (6.2)`);
        }
        const headEdges = headSupersessions.get(x);
        const headY = headEdges && headEdges.length ? headEdges[headEdges.length - 1].newId : null;
        if (headY !== null && headY !== newY) {
            // H already records a DIFFERENT successor for x.
            if (onConflict === ON_CONFLICT.REFUSE) { conflicts.push(x); continue; }
            if (onConflict === ON_CONFLICT.SKIP) { skippedEdges.push(x); continue; }
            // keep-both: L's edge is recorded as an additional edge, which makes
            // 3.5's "all edges" view non-trivial — H's edge is the immediate
            // successor and L's the current one (greatest depth wins).
            recordedEdges.push([x, newY]);
            continue;
        }
        // x not tombstoned in H, or tombstoned with no edge: recorded either
        // way — L adds information H lacked.
        recordedEdges.push([x, newY]);
    }
    if (conflicts.length) {
        throw new Error(`rebase refuses: ${conflicts.length} supersession conflict(s) — the head already records a different successor for id(s) ${conflicts.join(', ')} (pass --on-conflict skip or keep-both) (6.2)`);
    }

    // The rebased bitset is zeroExtend(T_H, newRowBase) with the surviving
    // ΔT_L bits set. A deletion already in T_H is a no-op, not a conflict:
    // deletion is idempotent.
    const bitset = new Uint8Array(bitsetBytesFor(newRowBase));
    bitset.set(zeroExtend(headBitset, newRowBase).subarray(0, bitset.length));
    const newlySet = [];
    const setBit = (id) => {
        if (id >= newRowBase) {
            throw new Error(`rebase refuses: id ${id} is at or above the new rowBase ${newRowBase}`);
        }
        if (testBit(bitset, id)) return false;
        bitset[id >> 3] |= 1 << (id & 7);
        newlySet.push(id);
        return true;
    };
    for (const x of survivingDeletions) setBit(x);
    for (const [x] of recordedEdges) setBit(x);

    // Terminal rule: an empty rebase emits NOTHING (6.2). If the layer owns no
    // records and its surviving deletions set no bit the head does not already
    // have, emitting would manufacture exactly the artifact 6.1 forbids and 4.3
    // rejects — records 0 with no increase in tombstoneCount.
    if (layerRecords === 0 && newlySet.length === 0) {
        return {
            emit: false,
            reason: 'the rebased operation is empty: the target history already represents every deletion this layer performed (6.2)',
            droppedForeign, skippedEdges,
        };
    }

    return {
        emit: true,
        newRowBase,
        records: layerRecords,
        bitset,
        tombstones: popcountBytes(bitset),
        delta: survivingDeletions,
        newlySet: newlySet.sort((a, b) => a - b),
        supersessions: recordedEdges.sort((a, b) => a[0] - b[0]),
        droppedForeign,
        skippedEdges,
        // Chain-level golden queries carry expected GLOBAL ids, which are wrong
        // after a rebase: L's own ids moved and the surrounding chain changed.
        // A rebase MUST either regenerate them against the rebased chain or drop
        // them explicitly; it MUST NOT copy them (6.2).
        goldenQueries: null,
        goldenQueriesDropped: 'rebase',
    };
}

export { canonicalJson };
