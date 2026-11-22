// Layer manifest validation — LAYERED_PROFILE.md 4.1, 4.2, 3.3.
//
// A layer is a complete-profile container with a `layer` object and profile
// string `pikelet-layer-v1`. This module validates that object's shape
// standalone, and then validates it against the parent a reader ACTUALLY
// OPENED (3.3): locators are hints, identities are the truth, and every
// arithmetic relation is re-derived rather than trusted.
//
// Nothing here reads bytes. The caller supplies a parsed manifest and, for the
// relational checks, the parent's parsed manifest plus the identity the parent
// was actually verified under. Keeping it pure makes the whole hostile-chain
// family of section 10 testable without building files.

export const LAYER_PROFILE = 'pikelet-layer-v1';
// 2.7: 1 <= depth <= 16. The producer default is lower (8, still unmeasured —
// see 13.5); this is the format ceiling a reader must refuse past.
export const MAX_DEPTH = 16;
// 3.3: a chain holds at most 2^31 - 1 records, ids 0 .. 2^31 - 2.
export const MAX_RECORDS = 2 ** 31 - 1;
// 3.3: warn once a chain's row total passes 2^30 so compaction can be
// scheduled before appends start failing.
export const ROW_TOTAL_WARN = 2 ** 30;

const isHex64 = (v) => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v);
const isCount = (v, max = MAX_RECORDS) => Number.isSafeInteger(v) && v >= 0 && v <= max;

/**
 * Canonical JSON, for the equality comparisons 4.1 requires on `encoder` and
 * `layer.ingest`. Object keys sorted, no insignificant whitespace. Must match
 * the builder's canonicalization or the equality checks are meaningless.
 */
export function canonicalJson(value) {
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
    // Code-unit order, not localeCompare: key order is part of the pack identity.
    const keys = Object.keys(value).filter((k) => value[k] !== undefined).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(',')}}`;
}

/**
 * Validate a layer manifest's own `layer` object — everything checkable
 * without the parent. Throws on the first violation; the message names the
 * field so a hostile chain's refusal says which rule it broke.
 *
 * @returns {{identityFields: {parentIdentity: string, baseIdentity: string},
 *            depth: number, rowBase: number, records: number,
 *            tombstones: number, supersessions: number,
 *            tombstoneOnly: boolean, locator: string|null}}
 */
export function validateLayerObject(manifest) {
    if (!manifest || typeof manifest !== 'object') throw new Error('layer manifest is not an object');
    if (manifest.profile !== LAYER_PROFILE) {
        throw new Error(`layer manifest profile is ${JSON.stringify(manifest.profile)}, expected ${LAYER_PROFILE}`);
    }
    const layer = manifest.layer;
    if (!layer || typeof layer !== 'object' || Array.isArray(layer)) {
        throw new Error('layer manifest has no layer object');
    }
    const parent = layer.parent;
    if (!parent || typeof parent !== 'object' || Array.isArray(parent)) {
        throw new Error('layer.parent is missing');
    }
    if (!isHex64(parent.identity)) {
        throw new Error('layer.parent.identity must be 64 lowercase hex characters');
    }
    if (parent.locator !== undefined && typeof parent.locator !== 'string') {
        throw new Error('layer.parent.locator, when present, must be a string');
    }
    if (!isHex64(layer.baseIdentity)) {
        throw new Error('layer.baseIdentity must be 64 lowercase hex characters');
    }
    if (!Number.isSafeInteger(layer.depth) || layer.depth < 1 || layer.depth > MAX_DEPTH) {
        throw new Error(`layer.depth must be an integer in [1, ${MAX_DEPTH}], got ${layer.depth}`);
    }
    if (!isCount(layer.rowBase)) {
        throw new Error(`layer.rowBase must be an integer in [0, ${MAX_RECORDS}], got ${layer.rowBase}`);
    }
    if (!isCount(layer.records)) {
        throw new Error(`layer.records must be an integer in [0, ${MAX_RECORDS}], got ${layer.records}`);
    }
    // 3.3: every rowBase + records sum is bounded by the id ceiling.
    if (layer.rowBase + layer.records > MAX_RECORDS) {
        throw new Error(`layer.rowBase + layer.records (${layer.rowBase + layer.records}) exceeds ${MAX_RECORDS}`);
    }
    // 3.4: the cumulative bitset covers [0, rowBase), so it cannot hold more
    // bits than that, and a layer cannot tombstone its own rows.
    if (!isCount(layer.tombstones) || layer.tombstones > layer.rowBase) {
        throw new Error(`layer.tombstones must be an integer in [0, rowBase=${layer.rowBase}], got ${layer.tombstones}`);
    }
    if (!isCount(layer.supersessions, layer.rowBase)) {
        throw new Error(`layer.supersessions must be an integer in [0, rowBase=${layer.rowBase}], got ${layer.supersessions}`);
    }
    // 3.5: each supersession's oldId must be tombstoned in this layer's
    // bitset, and oldId is unique within a layer, so there cannot be more
    // edges than tombstones.
    if (layer.supersessions > layer.tombstones) {
        throw new Error(`layer.supersessions (${layer.supersessions}) exceeds layer.tombstones (${layer.tombstones}); every oldId must be tombstoned and is unique per layer`);
    }
    // 3.5: newId lies in this layer's own id range, so a layer with no
    // records of its own can record no supersession.
    if (layer.records === 0 && layer.supersessions > 0) {
        throw new Error('a tombstone-only layer cannot record a supersession: newId must lie in its own id range');
    }
    if (layer.ingest === undefined || layer.ingest === null || typeof layer.ingest !== 'object' || Array.isArray(layer.ingest)) {
        throw new Error('layer.ingest is missing; appendability requires an ingestion declaration (6.1)');
    }
    // 4.1: ingestAsserted is a SIBLING of layer.ingest, never a member of it.
    // If it were a member, asserting a declaration would change its canonical
    // form and a compacted base's corpus.ingest would not be byte-identical to
    // the declaration the chain carried (6.3).
    if ('ingestAsserted' in layer.ingest) {
        throw new Error('layer.ingest must not contain ingestAsserted; it is a sibling field inside layer, not part of the declaration');
    }
    if (layer.ingestAsserted !== undefined && typeof layer.ingestAsserted !== 'boolean') {
        throw new Error('layer.ingestAsserted, when present, must be a boolean');
    }

    // 4.1: corpus.records MUST equal layer.records. A tombstone-only layer
    // still carries a corpus object, with records 0 and provenance, and MUST
    // omit the layout fields and the index object.
    const corpus = manifest.corpus;
    if (!corpus || typeof corpus !== 'object' || Array.isArray(corpus)) {
        throw new Error('layer manifest has no corpus block');
    }
    if (corpus.records !== layer.records) {
        throw new Error(`manifest corpus.records (${corpus.records}) must equal layer.records (${layer.records})`);
    }
    const tombstoneOnly = layer.records === 0;
    const layoutFields = ['layout', 'pageRecords', 'pages', 'recordDigest', 'pageTableSha256'];
    if (tombstoneOnly) {
        for (const f of layoutFields) {
            if (corpus[f] !== undefined) {
                throw new Error(`a tombstone-only layer must omit corpus.${f}: it has no corpus segment to describe`);
            }
        }
        if (manifest.index !== undefined) {
            throw new Error('a tombstone-only layer must omit the index object: it has no index segment');
        }
    } else {
        if (typeof corpus.layout !== 'string') {
            throw new Error('a layer with records must declare corpus.layout');
        }
        if (!manifest.index || typeof manifest.index !== 'object') {
            throw new Error('a layer with records must carry an index object');
        }
    }

    return {
        identityFields: { parentIdentity: parent.identity, baseIdentity: layer.baseIdentity },
        depth: layer.depth,
        rowBase: layer.rowBase,
        records: layer.records,
        tombstones: layer.tombstones,
        supersessions: layer.supersessions,
        tombstoneOnly,
        locator: typeof parent.locator === 'string' ? parent.locator : null,
    };
}

/**
 * The relational checks of 3.3 and 4.1, against the parent a reader actually
 * opened and verified. `parentIdentity` is the identity the parent's own bytes
 * hashed to — never a value read out of the child.
 *
 * @param {object} childManifest
 * @param {{manifest: object, identity: string, isBase: boolean}} parent
 */
export function validateAgainstParent(childManifest, parent) {
    const child = validateLayerObject(childManifest);
    if (!parent || typeof parent !== 'object') throw new Error('no parent supplied');
    const { manifest: pm, identity: parentIdentity, isBase } = parent;
    if (!isHex64(parentIdentity)) throw new Error('parent identity must be 64 lowercase hex characters');

    // 3.2: refuse a chain in which any opened parent's identity differs from
    // the identity its child committed to. This is the hash-chain link.
    if (child.identityFields.parentIdentity !== parentIdentity) {
        throw new Error(`layer commits to parent identity ${child.identityFields.parentIdentity.slice(0, 12)}… but the opened parent is ${parentIdentity.slice(0, 12)}…`);
    }

    const parentDepth = isBase ? 0 : pm?.layer?.depth;
    const parentRowBase = isBase ? 0 : pm?.layer?.rowBase;
    const parentRecords = pm?.corpus?.records;
    if (!isBase) {
        if (!Number.isSafeInteger(parentDepth)) throw new Error('parent layer.depth is not an integer');
        if (!Number.isSafeInteger(parentRowBase)) throw new Error('parent layer.rowBase is not an integer');
    }
    if (!Number.isSafeInteger(parentRecords)) throw new Error('parent corpus.records is not an integer');

    // 3.3, verified against the parent actually opened:
    //   layer.rowBase == parent.layer.rowBase + parent.corpus.records  (parent a layer)
    //   layer.rowBase == parent.corpus.records                         (parent the base)
    const expectedRowBase = isBase ? parentRecords : parentRowBase + parentRecords;
    if (child.rowBase !== expectedRowBase) {
        throw new Error(`layer.rowBase is ${child.rowBase} but the opened parent implies ${expectedRowBase}`);
    }
    if (child.depth !== parentDepth + 1) {
        throw new Error(`layer.depth is ${child.depth} but the opened parent is at depth ${parentDepth}`);
    }
    // layer.baseIdentity == parent.layer.baseIdentity, or the parent's own
    // identity when the parent is the base.
    const expectedBase = isBase ? parentIdentity : pm?.layer?.baseIdentity;
    if (child.identityFields.baseIdentity !== expectedBase) {
        throw new Error(`layer.baseIdentity is ${String(child.identityFields.baseIdentity).slice(0, 12)}… but the chain's base is ${String(expectedBase).slice(0, 12)}…`);
    }

    // 4.1: encoder, dim and metric MUST equal the base's. Checked against the
    // parent here, which gives the same result transitively along the chain and
    // catches the difference at the layer that introduced it.
    if (canonicalJson(childManifest.encoder) !== canonicalJson(pm?.encoder)) {
        throw new Error('layer encoder object differs from its parent\'s; a chain may not change the encoder');
    }
    if (childManifest.dim !== pm?.dim) {
        throw new Error(`layer dim ${childManifest.dim} differs from its parent's ${pm?.dim}`);
    }
    if (childManifest.metric !== pm?.metric) {
        throw new Error(`layer metric ${JSON.stringify(childManifest.metric)} differs from its parent's ${JSON.stringify(pm?.metric)}`);
    }

    return child;
}

/**
 * The chain ingestion declaration of 4.1/6.1: the base's `corpus.ingest` when
 * the base carries one; otherwise the depth-1 layer's `layer.ingest`, which
 * must then be marked `ingestAsserted`. Returns the canonical form to compare
 * every layer against, plus whether the chain is asserted.
 */
export function chainIngestDeclaration(baseManifest, depth1Manifest) {
    const baseIngest = baseManifest?.corpus?.ingest;
    if (baseIngest !== undefined && baseIngest !== null) {
        return { canonical: canonicalJson(baseIngest), asserted: false, source: 'base' };
    }
    if (!depth1Manifest) {
        throw new Error('the base carries no corpus.ingest and no depth-1 layer was supplied: the chain has no ingestion declaration (6.1)');
    }
    const layer = depth1Manifest.layer || {};
    if (layer.ingestAsserted !== true) {
        throw new Error('the base carries no corpus.ingest, so the depth-1 layer must set layer.ingestAsserted true to introduce one (6.1)');
    }
    return { canonical: canonicalJson(layer.ingest), asserted: true, source: 'depth-1' };
}

/**
 * Check one layer's declaration against the chain's. Only the depth-1 layer of
 * an assertion-bearing chain can introduce the assertion; no later layer may
 * introduce, drop or change one.
 */
export function validateIngestAgainstChain(layerManifest, chainDecl) {
    const layer = layerManifest?.layer || {};
    if (canonicalJson(layer.ingest) !== chainDecl.canonical) {
        throw new Error(`layer at depth ${layer.depth} carries an ingestion declaration that differs from the chain's (6.1)`);
    }
    const asserted = layer.ingestAsserted === true;
    if (chainDecl.asserted && !asserted) {
        throw new Error(`layer at depth ${layer.depth} must carry ingestAsserted true on an asserted chain`);
    }
    if (!chainDecl.asserted && asserted) {
        throw new Error(`layer at depth ${layer.depth} asserts an ingestion declaration, but the chain's base declares one`);
    }
    return true;
}

/**
 * `rowTotal(A) = A.rowBase + A.records` (3.1) — the number of ids that existed
 * when A was the head. For a base it is just its record count.
 */
export function rowTotal(manifest) {
    const rb = manifest?.layer?.rowBase;
    const records = manifest?.corpus?.records;
    if (!Number.isSafeInteger(records)) throw new Error('manifest corpus.records is not an integer');
    return (Number.isSafeInteger(rb) ? rb : 0) + records;
}
