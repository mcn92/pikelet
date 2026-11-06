// Chain state — LAYERED_PROFILE.md 5.1 (steps 4-5), 5.2, 5.3, 5.5, 5.6.
//
// Everything a mounted chain needs that is not bytes: the ordered interval
// table of search tiers, id ownership, the head mask projected onto each tier's
// local range, the union supersession map, calibration drift, and the ancestry
// used to scope citations.
//
// Pure: the caller supplies already-verified per-member facts. That keeps the
// whole of 5.2/5.3/5.6 testable without building files, and keeps the sharp
// edges in one place.

import { testBit, bitsetBytesFor } from './tombstones.mjs';

/**
 * A **search tier** (3.1) is the base, or any layer with records >= 1: a chain
 * member that owns ids. Tombstone-only layers are chain members but not search
 * tiers, and take no part in id resolution.
 *
 * Every per-tier construction in section 5 ranges over search tiers, so the
 * base is always included ("for every layer" would exclude it).
 */
export function buildIntervalTable(members) {
    if (!Array.isArray(members) || members.length === 0) {
        throw new Error('chain has no members');
    }
    if (!members[0] || members[0].depth !== 0) {
        throw new Error('the first chain member must be the base (depth 0)');
    }
    const tiers = [];
    let expectedRowBase = 0;
    let expectedDepth = 0;
    for (const m of members) {
        if (m.depth !== expectedDepth) {
            throw new Error(`chain member at index ${expectedDepth} declares depth ${m.depth}`);
        }
        const rowBase = m.depth === 0 ? 0 : m.rowBase;
        if (rowBase !== expectedRowBase) {
            throw new Error(`chain member at depth ${m.depth} has rowBase ${rowBase}, expected ${expectedRowBase}`);
        }
        if (!Number.isSafeInteger(m.records) || m.records < 0) {
            throw new Error(`chain member at depth ${m.depth} has a bad record count`);
        }
        if (m.records > 0) {
            tiers.push({
                tier: tiers.length,
                depth: m.depth,
                identity: m.identity,
                rowBase,
                records: m.records,
                end: rowBase + m.records,
            });
        }
        expectedRowBase = rowBase + m.records;
        expectedDepth += 1;
    }
    if (!tiers.length) throw new Error('a chain must have at least one search tier');
    // Search tiers' rowBase values are strictly increasing, since each owns at
    // least one id. This is the property that makes binary search valid over
    // THIS table and invalid over all members (5.2).
    for (let i = 1; i < tiers.length; i++) {
        if (tiers[i].rowBase <= tiers[i - 1].rowBase) {
            throw new Error('search tier rowBase values must be strictly increasing');
        }
    }
    return { tiers, rowTotal: expectedRowBase };
}

/**
 * `owner(id)` — 5.2. The unique search tier whose interval contains `id`.
 *
 * Binary search over the SEARCH TIERS only. Searching "the largest rowBase <=
 * id" over all chain members is wrong: consecutive tombstone-only layers give
 * the next layer the same rowBase (3.3), so rowBase is not a unique key over
 * members and such a search can land on a member that owns no ids at all.
 */
export function ownerOf(table, id) {
    if (!Number.isSafeInteger(id) || id < 0) return null;
    const { tiers } = table;
    let lo = 0;
    let hi = tiers.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const t = tiers[mid];
        if (id < t.rowBase) hi = mid - 1;
        else if (id >= t.end) lo = mid + 1;
        else return { tier: t, localId: id - t.rowBase };
    }
    // Ids outside every interval are invalid and MUST be rejected (5.2).
    return null;
}

/**
 * Project the head's bitset onto each search tier's local range (5.3):
 * `mask_j[i] = T_head[rowBase_j + i]`, computed once at mount.
 *
 * Returned as one Uint8Array per tier, LSB-first, so a tier's scan can test a
 * local row with one bit test and no arithmetic against the global id space.
 */
export function projectMasks(table, headBitset) {
    const masks = [];
    for (const t of table.tiers) {
        const bytes = bitsetBytesFor(t.records);
        const mask = new Uint8Array(bytes);
        if (headBitset && headBitset.length) {
            for (let i = 0; i < t.records; i++) {
                if (testBit(headBitset, t.rowBase + i)) mask[i >> 3] |= 1 << (i & 7);
            }
        }
        masks.push(mask);
    }
    return masks;
}

/** Live record count: rowTotal minus the head's tombstone popcount. */
export function liveCount(table, headTombstoneCount) {
    return table.rowTotal - (headTombstoneCount || 0);
}

/**
 * The union supersession map (5.1 step 4, 3.5). Each layer's segment carries
 * only the edges that layer recorded; the reader forms the union at mount.
 *
 * Edges are ordered by the DEPTH of the layer that recorded them — never by
 * segment order, file order or id — so "earliest" and "latest" are always
 * unique (within one layer an oldId has at most one edge).
 */
export function buildSupersessionMap(perLayer) {
    const map = new Map();
    for (const { depth, supersessions } of perLayer) {
        for (const [oldId, newId] of supersessions || []) {
            if (!map.has(oldId)) map.set(oldId, []);
            map.get(oldId).push({ depth, newId });
        }
    }
    for (const edges of map.values()) {
        edges.sort((a, b) => a.depth - b.depth);
        for (let i = 1; i < edges.length; i++) {
            if (edges[i].depth === edges[i - 1].depth) {
                throw new Error(`two supersessions for the same oldId at depth ${edges[i].depth}: within one layer an oldId has at most one edge`);
            }
        }
    }
    return map;
}

/**
 * The three views of 3.5.
 *  - immediate: the edge recorded at the LEAST depth
 *  - all:       every edge, ascending by depth
 *  - current:   start from the GREATEST-depth edge and repeatedly follow each
 *               reached id's greatest-depth edge until an id with none
 *
 * Cycles are structurally impossible (newId >= rowBase_j > oldId, so edges
 * always point to higher ids), but the walk is bounded anyway: a corrupted map
 * must not hang a reader.
 */
export function resolveSupersession(map, id) {
    const edges = map.get(id);
    if (!edges || !edges.length) return { immediate: null, all: [], current: null };
    const immediate = edges[0].newId;
    const all = edges.map((e) => e.newId);
    let current = edges[edges.length - 1].newId;
    const seen = new Set([id, current]);
    for (let hops = 0; hops < 64; hops++) {
        const next = map.get(current);
        if (!next || !next.length) break;
        const candidate = next[next.length - 1].newId;
        if (seen.has(candidate)) break;
        seen.add(candidate);
        current = candidate;
    }
    return { immediate, all, current };
}

/**
 * Calibration drift (3.1):
 *   (sum of layer records for depth >= 1 + head tombstones) / base records
 *
 * It counts HISTORY, not current divergence: a record appended and later
 * tombstoned counts twice, and a supersession counts as one append plus one
 * tombstone. Intentionally conservative — the fit's population lost a member it
 * had seen and gained one it had not.
 */
export function calibrationDrift(members, headTombstoneCount) {
    const base = members[0];
    if (!base || base.depth !== 0) throw new Error('calibrationDrift needs the base as the first member');
    if (!Number.isSafeInteger(base.records) || base.records <= 0) {
        throw new Error('calibrationDrift needs a base with at least one record');
    }
    let appended = 0;
    for (const m of members) if (m.depth >= 1) appended += m.records;
    return (appended + (headTombstoneCount || 0)) / base.records;
}

/**
 * Calibration status (5.5). The effective limit is
 * `min(producerEnvelope, readerLimit)`: a host can always be stricter than the
 * artifact, and an artifact can never make a host less strict.
 *
 * An absent producer envelope is +Infinity (no claim is no constraint), and the
 * reader's limit MUST be finite, so an artifact that declares nothing is still
 * gated.
 */
export function calibrationStatus({ drift, producerEnvelope, readerLimit, baseHasFit, everyLayerShipsBloom }) {
    if (!Number.isFinite(readerLimit)) {
        throw new Error('readerLimit MUST be finite (5.5)');
    }
    const envelope = Number.isFinite(producerEnvelope) ? producerEnvelope : Infinity;
    const effectiveLimit = Math.min(envelope, readerLimit);
    if (!baseHasFit) {
        return { status: 'none', drift, effectiveLimit, producerEnvelope: envelope, readerLimit };
    }
    // A layer WITH RECORDS whose calibration region is `none` while the base
    // carries a fit degrades the chain to unscored (4.5, 5.5). The caller
    // leaves tombstone-only layers out of everyLayerShipsBloom: they have no
    // vocabulary for a bloom to carry.
    const status = (drift <= effectiveLimit && everyLayerShipsBloom) ? 'inherited' : 'drift-exceeded';
    return { status, drift, effectiveLimit, producerEnvelope: envelope, readerLimit };
}

/**
 * Global lexical statistics over search tiers (5.4). Tombstoned records are
 * counted in N, avgdl and df: removing them would require reading every posting
 * list at mount.
 *
 * The resulting scores differ from a from-scratch build over the live set. The
 * drift gate of 5.5 is a change-volume gate and implies no bound on that
 * difference; 13.3 measured it at under 0.005 nDCG@10 up to 25% drift on two
 * BEIR datasets, which is why no correction table ships. A reader MUST report
 * `statsIncludeTombstoned: true`.
 */
export function globalLexicalStats(perTier) {
    let N = 0;
    let totalTokens = 0;
    for (const t of perTier) {
        if (!Number.isSafeInteger(t.docCount) || t.docCount < 0) throw new Error('a tier has a bad docCount');
        N += t.docCount;
        totalTokens += t.totalTokens;
    }
    return {
        N,
        totalTokens,
        avgdl: N > 0 ? Math.max(1, totalTokens / N) : 1,
        statsIncludeTombstoned: true,
    };
}

/** df(t) summed across tiers, and the idf the reader scores with (5.4). */
export function idfFor(N, df) {
    return Math.log(1 + (N - df + 0.5) / (df + 0.5));
}

/**
 * Ancestry for citation scoping (5.6): identity -> { depth, rowTotal }.
 *
 * A bare id is an id in the mounted head; a citation is (identity, id), and it
 * resolves only along the mounted ancestry. A citation against a head that is
 * on this ancestry is valid when its id existed when that head was current —
 * that is, `id < rowTotal(thatHead)`.
 */
export function buildAncestry(members) {
    const ancestry = new Map();
    let rowTotal = 0;
    for (const m of members) {
        rowTotal += m.records;
        ancestry.set(m.identity, { depth: m.depth, rowTotal });
    }
    return ancestry;
}

/**
 * Resolve a citation (identity, id) against the mounted ancestry (5.6).
 * Returns a reason rather than throwing, because "not on this history" is a
 * legitimate answer a consumer needs to be able to act on.
 */
export function resolveCitation(ancestry, identity, id) {
    const entry = ancestry.get(identity);
    if (!entry) return { ok: false, reason: 'not-on-this-history' };
    if (!Number.isSafeInteger(id) || id < 0) return { ok: false, reason: 'invalid-id' };
    // An id at or above that head's rowTotal did not exist when it was current,
    // so a citation naming it is invalid under that identity even though the id
    // is valid in the current head.
    if (id >= entry.rowTotal) return { ok: false, reason: 'id-postdates-that-head' };
    return { ok: true, depth: entry.depth, id };
}
