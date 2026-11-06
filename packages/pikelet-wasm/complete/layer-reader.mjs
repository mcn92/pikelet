// The chain reader — LAYERED_PROFILE.md 5.1, 5.2, 5.3, 5.6.
//
// openPikeletChain() mounts a base plus zero or more layers and serves them as
// one corpus. It does NOT reimplement the container: every member is opened
// through openPikeletFile(), which is decision 1 — "a reader that implements the
// complete profile implements this one by adding the chain logic of section 5,
// not a second container parser."
//
// What this file adds on top of those opens:
//   - chain resolution and ancestry verification (5.1 steps 1-2, 3.2, 3.3)
//   - the interval table, per-tier masks and chain state (5.1 step 4)
//   - a query() that generates candidates per search tier with that tier's
//     mask supplied to the scan, merges by distance, and fuses once (5.3)
//   - record()/citation resolution over global ids (5.6)
//
// The caller may supply every member base-first, or just the head and let the
// reader walk up to the base through 5.1.1's parent resolution: a caller-
// supplied lineage map (section 7), then each layer's confined
// `parent.locator`, then a host resolver. The lineage map is the cheaper path
// when available, since every location is known up front and members open in
// one wave; the locator walk is sequential, because each hop's location is
// only known once its child's manifest has been read.

import { openPikeletFile, readChainMemberShell, readMemberManifest, base64Bytes, LEXICAL_CANDIDATES, LEXICAL_CUTOFF } from './index.mjs';
import { resolveParentLocation, assertConfined } from './layer-locator.mjs';
import { httpRangeSource } from './sources.mjs';
import { fuseCandidates, FUSION_DEFAULTS } from './fusion.mjs';
import { parseTombstoneSegment, firstSupersetViolation } from './tombstones.mjs';
import { validateAgainstParent, chainIngestDeclaration, validateIngestAgainstChain, LAYER_PROFILE, MAX_DEPTH } from './layer-manifest.mjs';
import { parseInheritedQuerySegment, unionBloom } from './layer-encoder.mjs';
import {
    buildIntervalTable, ownerOf, projectMasks, liveCount, buildSupersessionMap,
    resolveSupersession, calibrationDrift, calibrationStatus, globalLexicalStats,
    buildAncestry, resolveCitation,
} from './layer-chain.mjs';

// 5.9's budgets, provisional: "chosen for round-trip count and plausibility,
// not measured" (13.5). Named here so a host can lower them and so a refusal
// can name the budget it hit.
export const CHAIN_DEFAULTS = Object.freeze({
    maxDepth: MAX_DEPTH,
    // 5.5: the reader's limit MUST be finite and declared. 0.20 is
    // conservative (13.1 measured no degradation out to 40% drift), and it is
    // a reader policy, not a format constant.
    readerDriftLimit: 0.20,
});

/**
 * @param {Array<string|object>} members  head-last or base-first list of
 *   locations/sources. Base first is the natural order and what this accepts:
 *   [base, layer1, layer2, ...]. Every member is opened through
 *   openPikeletFile(), so a string is a path and an object is a range source.
 * @param {object} [options] forwarded to each member's open, plus:
 *   readerDriftLimit, maxDepth, and fetch (the transport for any parent the
 *   walk reaches over HTTP; default: the child source's own, else
 *   globalThis.fetch).
 */
/**
 * 5.1.1: walk from a head to its base, resolving each parent by lineage
 * listing, then confined locator, then host resolver. Returns the members
 * base-first, which is what openPikeletChain consumes.
 *
 * Sequential by necessity: a hop's location comes from its child's manifest,
 * so nothing below the head is known until the head is read. Bounded by
 * maxDepth so a manifest cycle or a hostile chain of locators cannot walk
 * forever, and every hop's identity is checked against what its child named
 * before that member is accepted -- a locator that resolves to the wrong
 * artifact fails here rather than at the pairwise check later.
 */
async function walkToBase(head, options) {
    const maxDepth = options.maxDepth ?? CHAIN_DEFAULTS.maxDepth;
    const lineage = options.lineage instanceof Map ? options.lineage : null;
    const chain = [head];
    let cursor = head;
    let expectIdentity = null;
    let childFetch = null;
    for (let hop = 0; hop <= maxDepth; hop++) {
        const peek = await readMemberManifest(cursor, options);
        if (expectIdentity !== null && peek.identity !== expectIdentity) {
            throw new Error(`chain member at ${describeLocation(cursor)} has identity ${peek.identity.slice(0, 12)}… `
                + `but its child named ${expectIdentity.slice(0, 12)}…: the located artifact is not the parent the chain commits to (3.2)`);
        }
        if (!peek.isLayer) return chain;          // reached the base
        if (!peek.parent) throw new Error('a layer manifest carries no layer.parent (3.2)');
        // A child is a URL whether it arrived as a range source or, one hop
        // up, as the URL string a locator resolved to.
        const childLocation = typeof cursor === 'string' ? cursor : (cursor.url ?? cursor.location ?? null);
        const childIsFile = typeof cursor === 'string' && !isHttpUrl(cursor);
        if (typeof cursor === 'object' && typeof cursor?.fetch === 'function') childFetch = cursor.fetch;
        const located = await resolveParentLocation(peek.parent, {
            lineage,
            childLocation,
            kind: childIsFile ? 'file' : 'url',
            resolveParents: options.resolveParents !== false,
            hostResolver: options.hostResolver || null,
            // A file-path locator needs realpath/dirname/join to confine
            // against the child's REAL directory (symlink escapes are the
            // point of 5.1.1's "real path" wording). The locator module stays
            // runtime-agnostic, so the Node implementation is supplied here
            // and only when a file path is actually being resolved -- a
            // browser or Worker mount is URL-only and never loads node:fs.
            fsops: options.fsops || (childIsFile ? await nodeFsops() : null),
        });
        expectIdentity = peek.parent.identity;
        cursor = located.location;
        if (typeof cursor === 'string' && isHttpUrl(cursor)) {
            // A publisher's locator is confined to the child's origin and
            // directory, and so is every redirect on the way to it (5.1.1).
            // A lineage listing or host resolver is the operator's choice and
            // may name any origin, so its redirects are not confined.
            const confineTo = located.via === 'locator' ? childLocation : null;
            cursor = httpRangeSource(cursor, {
                // The walk's transport: the caller's, else the one the
                // child source was opened with.
                ...((options.fetch || childFetch) ? { fetch: options.fetch || childFetch } : {}),
                ...(confineTo ? { redirectGuard: (target) => assertConfined(target, confineTo) } : {}),
            });
            await cursor.init();
        }
        chain.unshift(cursor);
    }
    throw new Error(`walking to the base exceeded the depth limit ${maxDepth} (2.7): a layer.parent cycle or an over-deep chain`);
}

function memberIntegritySummary(opened) {
    const infos = opened.filter((r) => typeof r?.info === 'function').map((r) => r.info());
    const one = (key) => {
        const values = [...new Set(infos.map((i) => i[key]))];
        return values.length === 1 ? values[0] : 'mixed';
    };
    return { corpusIntegrity: one('corpusIntegrity'), indexRowIntegrity: one('indexRowIntegrity') };
}

const isHttpUrl = (loc) => /^https?:\/\//i.test(loc);
const describeLocation = (loc) => (typeof loc === 'string' ? loc : (loc?.url ?? loc?.location ?? '<source>'));

// Lazily imported so a URL-only mount in a runtime without node:fs never
// touches it. Cached: the walk resolves one hop at a time and would otherwise
// re-import per hop.
let fsopsPromise = null;
const nodeFsops = () => {
    fsopsPromise ??= (async () => {
        const [fsp, path] = await Promise.all([import('node:fs/promises'), import('node:path')]);
        return {
            realpath: (p) => fsp.realpath(p),
            dirname: (p) => path.dirname(p),
            join: (a, b) => path.join(a, b),
            sep: path.sep,
        };
    })();
    return fsopsPromise;
};

export async function openPikeletChain(members, options = {}) {
    if (!Array.isArray(members) || members.length === 0) {
        throw new Error('openPikeletChain() needs at least a base');
    }
    // A single member that turns out to be a layer is a head to walk up from,
    // not an error: 5.1.1 exists so a published chain can be mounted by naming
    // its head. A single base is just a base, and the walk returns it as-is.
    if (members.length === 1 && options.followParents !== false) {
        const peek = await readMemberManifest(members[0], options);
        if (peek.isLayer) members = await walkToBase(members[0], options);
    }
    const maxDepth = options.maxDepth ?? CHAIN_DEFAULTS.maxDepth;
    if (members.length - 1 > maxDepth) {
        throw new Error(`chain depth ${members.length - 1} exceeds the limit ${maxDepth} (2.7)`);
    }
    const readerDriftLimit = options.readerDriftLimit ?? CHAIN_DEFAULTS.readerDriftLimit;

    const opened = [];
    const closeAll = async () => {
        for (const r of opened) { try { await r.close(); } catch { /* best effort */ } }
    };

    try {
        // --- 5.1 steps 1-2: open every member, verify the chain links ---
        // asChainMember lets openPikeletFile accept a `pikelet-layer-v1`
        // manifest. Only this function passes it, and only because it goes on
        // to open every ancestor and verify the hash chain — a layer opened
        // alone still fails, which is what 4.1 requires.
        //
        // A tombstone-only layer has no index, corpus or lexical segment, so it
        // is read as a "shell": manifest plus its two segments, verified the
        // same way, with none of the corpus/sketch machinery a search tier
        // needs. It is a chain member but not a search tier (3.1).
        const tiersRaw = [];
        for (let i = 0; i < members.length; i++) {
            const shell = i > 0 ? await readChainMemberShell(members[i], options) : null;
            if (shell && shell.tombstoneOnly) {
                opened.push(shell);
                tiersRaw.push(shell.tier);
                continue;
            }
            if (shell) await shell.close();
            const reader = await openPikeletFile(members[i], { ...options, asChainMember: i > 0 });
            opened.push(reader);
            tiersRaw.push(reader.__chainTier);
        }
        const base = tiersRaw[0];
        if (base.manifest.profile === LAYER_PROFILE) {
            throw new Error('the first chain member is a layer, not a base: a layer opened alone would serve a fraction of a corpus (4.1)');
        }
        // A chain maps each member's sketch rows to records positionally
        // (rowBase + local id). A member whose index holds several rows per
        // record (row map, COMPLETE_PROFILE.md 3.9) would be misread, so it is
        // refused until the layered profile defines rows.
        if (tiersRaw.some((t) => t && t.rowMap)) {
            throw new Error('a pack whose index holds several rows per record (sentence windows) cannot be part of a layered chain yet: '
                + 'mount it on its own, or rebuild it without --sentence-windows to append to it');
        }

        const chainMembers = [{
            depth: 0, identity: base.identity, rowBase: 0, records: base.recordCount,
        }];
        // Ingestion declaration: defined by the base, or asserted by depth 1.
        const depth1Manifest = tiersRaw.length > 1 ? tiersRaw[1].manifest : null;
        const ingestDecl = tiersRaw.length > 1 ? chainIngestDeclaration(base.manifest, depth1Manifest) : null;

        const perLayerTombstones = [];
        let headBitset = new Uint8Array(0);
        const layerBlooms = [];
        let everyLayerShipsBloom = true;
        // 4.5: every layer's bloom MUST use the base asset's geometry, so the
        // union is an exact OR. The geometry comes from the base's fit.
        const baseVocabBloom = base.calibrationJson?.asset?.vocabBloom;
        const baseBloomGeometry = options.baseBloom
            || (baseVocabBloom ? { bits: baseVocabBloom.bits, hashes: baseVocabBloom.hashes } : null);
        // 4.3: lexical is REQUIRED on a layer with records if the base carries
        // one, else MUST be absent. A layer that drops it leaves its records
        // invisible to BM25 and skews the global statistics; one that adds it
        // introduces lexical retrieval the base's fit never saw.
        const baseHasLexical = base.segments.has('lexical');

        for (let i = 1; i < tiersRaw.length; i++) {
            const tier = tiersRaw[i];
            const parent = tiersRaw[i - 1];
            // 3.2/3.3/4.1: verified against the parent ACTUALLY OPENED.
            const rel = validateAgainstParent(tier.manifest, {
                manifest: parent.manifest, identity: parent.identity, isBase: i === 1,
            });
            validateIngestAgainstChain(tier.manifest, ingestDecl);

            // 4.5: the encoder is resolved from the base, and the layer's
            // commitment must match the base's query-interp digest.
            const baseQiSha = base.manifest.segments.find((sg) => sg.kind === 'query-interp')?.sha256;
            if (!baseQiSha) throw new Error('the base carries no query-interp segment digest to inherit');
            const qiSeg = tier.segments.get('query-interp');
            if (!qiSeg) throw new Error(`layer at depth ${rel.depth} carries no query-interp segment`);
            const inherited = parseInheritedQuerySegment(
                await readSegment(opened[i], qiSeg),
                {
                    layerBaseIdentity: rel.identityFields.baseIdentity,
                    baseQueryInterpSha256: baseQiSha,
                    baseBloom: baseBloomGeometry,
                },
            );
            if (inherited.bloomBytes && !baseBloomGeometry) {
                throw new Error(`layer at depth ${rel.depth} ships a vocabulary bloom but the base carries none whose geometry it could match (4.5)`);
            }
            // A missing bloom degrades the chain because it hides a layer's
            // vocabulary from known_frac (4.5). A tombstone-only layer has no
            // records and so no vocabulary to hide: its `none` region is the
            // empty bloom, not a gap.
            if (inherited.bloomBytes) layerBlooms.push(inherited.bloomBytes);
            else if (!rel.tombstoneOnly) everyLayerShipsBloom = false;

            if (!rel.tombstoneOnly && tier.segments.has('lexical') !== baseHasLexical) {
                throw new Error(baseHasLexical
                    ? `layer at depth ${rel.depth} carries no lexical segment, but the base does: a layer must not drop lexical retrieval (4.3)`
                    : `layer at depth ${rel.depth} carries a lexical segment, but the base does not: a layer must not introduce lexical retrieval (4.3)`);
            }

            // 4.4: the tombstone segment is eager and digest-verified by the
            // member's own open; here it is parsed and structurally validated.
            const tombSeg = tier.segments.get('tombstones');
            if (!tombSeg) throw new Error(`layer at depth ${rel.depth} carries no tombstones segment (4.3)`);
            const tomb = parseTombstoneSegment(await readSegment(opened[i], tombSeg), {
                rowBase: rel.rowBase, tombstones: rel.tombstones,
                supersessions: rel.supersessions, records: rel.records,
            });
            // 3.4: the cumulative superset rule, against the parent's mask
            // zero-extended over the ids the parent did not yet address.
            const violation = firstSupersetViolation(tomb.bitset, headBitset, rel.rowBase);
            if (violation !== -1) {
                throw new Error(`layer at depth ${rel.depth} clears inherited tombstone for id ${violation}: a child's mask must be a superset of its parent's (3.4)`);
            }
            // 4.3: a tombstone-only layer exists only to delete, so its
            // cumulative count MUST exceed its parent's. Checked here too, not
            // only by the producer.
            if (rel.tombstoneOnly && countBits(tomb.bitset) <= countBits(headBitset)) {
                throw new Error(`tombstone-only layer at depth ${rel.depth} deletes nothing its parent had not already deleted (4.3)`);
            }
            headBitset = tomb.bitset;
            perLayerTombstones.push({ depth: rel.depth, supersessions: tomb.supersessions });
            chainMembers.push({
                depth: rel.depth, identity: tier.identity,
                rowBase: rel.rowBase, records: rel.records,
                // Retained per member, not just the head's: `rebase` needs a
                // layer's OWN bitset and its parent's to compute the delta it
                // replays, and 6.2 refuses rather than guessing without them.
                bitset: tomb.bitset,
                supersessions: tomb.supersessions,
            });
        }

        // --- 5.1 step 4: chain state ---
        const table = buildIntervalTable(chainMembers);
        const byIdentity = new Map(tiersRaw.map((tr) => [tr.identity, tr]));
        const searchTiers = table.tiers.map((t) => ({ ...t, reader: byIdentity.get(t.identity) }));
        for (const t of searchTiers) {
            if (!t.reader || !t.reader.sketch) {
                throw new Error(`search tier at depth ${t.depth} has no index: a tier that owns ids must carry one (4.3)`);
            }
        }
        const masks = projectMasks(table, headBitset);
        // Per-tier popcount, so a query can tell an empty mask (skip it, keep
        // the WASM scan) from one that actually excludes something.
        const maskCounts = masks.map((m) => countBits(m));
        const supersessions = buildSupersessionMap(perLayerTombstones);
        const ancestry = buildAncestry(chainMembers);
        const headTombstoneCount = countBits(headBitset);
        const drift = chainMembers.length > 1
            ? calibrationDrift(chainMembers, headTombstoneCount) : 0;
        // 5.5: "the base carries no fit" is status `none`. The fit lives in the
        // base's query-interp calibration region (`asset`), never in the
        // manifest.
        const baseCal = base.calibrationJson || null;
        const baseHasFit = !!(baseCal && baseCal.asset);
        const calibration = calibrationStatus({
            drift,
            // producerEnvelope is the base producer's declared validity
            // envelope. No claim is no constraint (+Infinity), which
            // calibrationStatus() handles; the reader's finite limit still
            // gates it.
            producerEnvelope: baseCal?.asset?.driftLimit,
            readerLimit: readerDriftLimit,
            baseHasFit,
            everyLayerShipsBloom: chainMembers.length === 1 ? true : everyLayerShipsBloom,
        });
        const lexStats = globalLexicalStats(searchTiers.map((t) => ({
            docCount: t.reader.lexicalIndex ? t.reader.lexicalIndex.docCount : 0,
            totalTokens: t.reader.lexicalIndex ? t.reader.lexicalIndex.totalTokens : 0,
        })));
        // The union bloom (4.5) starts from the base's own vocabulary bloom,
        // which ships in its calibration region beside the fit.
        const baseBloomBytes = baseCal?.vocabBloomBase64
            ? base64Bytes(baseCal.vocabBloomBase64)
            : null;
        const chainBloom = unionBloom([baseBloomBytes, ...layerBlooms]);

        let closed = false;
        const assertOpen = () => { if (closed) throw new Error('chain is closed'); };

        return {
            info() {
                assertOpen();
                return {
                    profile: LAYER_PROFILE,
                    identity: chainMembers[chainMembers.length - 1].identity,
                    baseIdentity: base.identity,
                    // The pack's name is the base's: layers add records, not a
                    // new name.
                    name: opened[0].info().name ?? null,
                    layers: chainMembers.length,
                    searchTiers: searchTiers.length,
                    records: table.rowTotal,
                    liveRecords: liveCount(table, headTombstoneCount),
                    tombstones: headTombstoneCount,
                    supersessions: supersessions.size,
                    dim: base.dim,
                    metric: base.manifest.metric,
                    // 4.5: a chain's encoder IS the base's — a layer carries no
                    // encoder bytes and inherits by commitment.
                    encoder: base.encoderInfo ?? null,
                    calibrationDrift: calibration.drift,
                    calibrationStatus: calibration.status,
                    driftLimit: calibration.effectiveLimit,
                    producerEnvelope: calibration.producerEnvelope,
                    readerLimit: calibration.readerLimit,
                    lexical: { ...lexStats },
                    // 4.1: sampleQueries may appear on any member and applies
                    // to that member. A chain surfaces the base's, which are
                    // the ones fit against the corpus the encoder saw.
                    sampleQueries: Array.isArray(base.manifest.sampleQueries) ? base.manifest.sampleQueries : [],
                    // Each member's own integrity, from its reader. A
                    // tombstone-only layer is a verified shell (manifest plus
                    // tombstones), with no corpus or index to report on.
                    members: chainMembers.map((m, i) => {
                        const ri = typeof opened[i]?.info === 'function' ? opened[i].info() : null;
                        return {
                            depth: m.depth, identity: m.identity, records: m.records,
                            ...(ri ? { corpusIntegrity: ri.corpusIntegrity, indexRowIntegrity: ri.indexRowIntegrity }
                                : { tombstoneOnly: true }),
                        };
                    }),
                    // The chain encodes with the base's encoder (4.5), so the
                    // base's verification is the chain's. Chain-level integrity
                    // is the members' when they agree, else 'mixed'.
                    encoderVerified: opened[0].info().encoderVerified,
                    ...memberIntegritySummary(opened),
                };
            },

            /**
             * 5.3. One encode, per-tier masked candidate generation, merge by
             * distance, fuse once. Equal distances break by ascending global id
             * — the reference reader's declared tie rule for chains.
             */
            async query(vector, k = 5, queryOptions = {}) {
                assertOpen();
                if (!Number.isSafeInteger(k) || k < 1) throw new Error('chain query() k must be a positive integer');
                const retrieval = queryOptions.retrieval ?? 'hybrid';
                if (!['hybrid', 'vector', 'lexical', 'augmented'].includes(retrieval)) {
                    throw new Error(`query() retrieval must be hybrid, vector, lexical, or augmented, got ${retrieval}`);
                }

                // --- 2. Lexical candidates (5.3 step 2, 5.4) -------------------
                // df is summed across tiers per query term, so a term scores
                // with one idf everywhere (per-tier df would let layer records
                // outrank base records on the same term). With comparable
                // scores, each tier's own top LEXICAL_CANDIDATES contains its
                // share of the merged top LEXICAL_CANDIDATES, so the cutoff
                // and cap below are exact over the merged, live-only list.
                let lexicalHits = [];
                const lexTiers = searchTiers.map((t, j) => ({ lex: t.reader.lexicalIndex, j }))
                    .filter((x) => x.lex);
                if (retrieval !== 'vector' && queryOptions.text && lexTiers.length) {
                    const perTierDf = await Promise.all(lexTiers.map(({ lex }) => lex.documentFrequencies(queryOptions.text)));
                    const df = new Map();
                    for (const m of perTierDf) for (const [term, n] of m) df.set(term, (df.get(term) || 0) + n);
                    const merged = [];
                    for (const { lex, j } of lexTiers) {
                        const tierMask = masks[j];
                        // The mask goes into search() so a tombstoned row never
                        // takes a slot before the cap (5.3); lazy openers
                        // return a promise, which await covers.
                        const hits = await lex.search(queryOptions.text, LEXICAL_CANDIDATES, {
                            exclude: maskCounts[j] > 0 ? (id) => bitAt(tierMask, id) : undefined,
                            // globalLexicalStats names the corpus size `N`;
                            // search() takes `docCount`.
                            stats: { docCount: lexStats.N, avgdl: lexStats.avgdl },
                            df,
                        });
                        for (const h of hits) {
                            merged.push({ id: searchTiers[j].rowBase + h.id, localId: h.id, tier: j, score: h.score });
                        }
                    }
                    merged.sort((a, b) => (b.score - a.score) || (a.id - b.id));
                    lexicalHits = merged.length
                        ? merged.filter((h) => h.score >= merged[0].score / LEXICAL_CUTOFF).slice(0, LEXICAL_CANDIDATES)
                        : [];
                }

                // --- 3. Vector candidates (5.3 step 3) -------------------------
                const perTier = await Promise.all(searchTiers.map(async (t, j) => {
                    // The lexical hits this tier owns join its exact rerank, so
                    // a known-item match the sketch scan's top-C missed is still
                    // scored by true distance (the single-file reader's rule).
                    // They were masked while postings scored; the check here is
                    // belt and braces, since extraCandidates bypass the mask.
                    const extra = lexicalHits.filter((h) => h.tier === j && !(maskCounts[j] > 0 && bitAt(masks[j], h.localId)))
                        .map((h) => h.localId);
                    // An all-zero mask is equivalent to no mask, and passing one
                    // makes the sketch reader over-fetch; it is omitted when the
                    // tier has no live deletion (maskCounts, computed at mount).
                    const out = await t.reader.sketch.search(vector, Math.min(k, t.records), {
                        rerank: queryOptions.rerank,
                        // The single-file reader's fetch tuning, with the same
                        // precedence: the query's own value, then the mount's,
                        // then this tier's default gap (its manifest hint as a
                        // floor under its source's preference).
                        parallelism: queryOptions.parallelism ?? queryOptions.rerankParallelism ?? options.rerankParallelism,
                        gap: queryOptions.gap ?? queryOptions.rerankGap ?? options.rerankGap ?? t.reader.rerankGap,
                        maxRangeBytes: queryOptions.maxRangeBytes ?? options.rerankMaxRangeBytes,
                        ...(maskCounts[j] > 0 ? { exclude: masks[j] } : {}),
                        // The tier's own staged scan kernel. Read per query, not
                        // captured at mount: staging finishes in the background
                        // after the member's open returns.
                        ...(t.reader.scanner ? { scanner: t.reader.scanner } : {}),
                        ...(extra.length ? { extraCandidates: extra } : {}),
                        // Every reranked candidate, not the top k: abstention is
                        // fit at a fixed top-10 window whatever the caller's k,
                        // as in the single-file reader.
                        fullRerankOutput: true,
                    });
                    return out.results.map((r) => ({
                        id: t.rowBase + r.id, localId: r.id, tier: j,
                        depth: t.depth, distance: r.distance,
                    }));
                }));
                const searched = perTier.flat()
                    .sort((a, b) => (a.distance - b.distance) || (a.id - b.id));

                // --- 4. Fuse (5.3 step 4) --------------------------------------
                // fusedFull is the full-window order (not k-truncated): coverage
                // grounds its verdict in what fusion ranks first, exactly as the
                // calibrator fit it. Null where the mode keeps distance order.
                let fusedFull = null;
                if (retrieval === 'lexical') {
                    const byId = new Map(searched.map((hit) => [hit.id, hit]));
                    fusedFull = lexicalHits.map((h) => byId.get(h.id)).filter(Boolean);
                } else if (retrieval === 'hybrid' && lexicalHits.length) {
                    const lexRank = new Map(lexicalHits.map((h, i) => [h.id, i]));
                    fusedFull = fuseCandidates(searched, lexicalHits.map((h) => h.id), queryOptions.fusion);
                    const vecRank = new Map(searched.map((h, i) => [h.id, i]));
                    fusedFull = fusedFull.map((hit, i) => ({
                        ...hit,
                        fusedRank: i + 1,
                        lexicalRank: lexRank.has(hit.id) ? lexRank.get(hit.id) + 1 : null,
                        vectorRank: vecRank.get(hit.id) + 1,
                    }));
                }
                const top = (fusedFull ?? searched).slice(0, k);
                const results = await Promise.all(top.map(async (hit) => {
                    // ownerOf searches the interval table, whose entries carry
                    // no reader; searchTiers[] is the parallel array that does.
                    const owner = ownerOf(table, hit.id);
                    if (!owner) throw new Error(`fused result id ${hit.id} owns no search tier (5.2)`);
                    const tier = searchTiers[owner.tier.tier];
                    const record = await tier.reader.hydrate(owner.localId);
                    return {
                        ...record, id: hit.id, distance: hit.distance, layer: tier.depth,
                        ...(hit.fusedRank ? { fusedRank: hit.fusedRank, lexicalRank: hit.lexicalRank, vectorRank: hit.vectorRank } : {}),
                    };
                }));
                // 5.5: under `inherited` the base's scorer runs unchanged over
                // the merged window. Under `drift-exceeded` matchQuality is
                // `unscored`, confidence is omitted, results still ship, and the
                // response says which input bound it, so a consumer can tell
                // "no fit" from "fit outgrown". The verdict is never applied
                // silently or omitted.
                let matchQuality = 'unscored';
                let confidence;
                let grounding = null;
                if (calibration.status === 'inherited' && base.scoreQuality) {
                    // Coverage reads the fit's passagesNeeded from the FULL
                    // fused window, not the k-truncated results.
                    const passagesNeeded = baseCal?.asset?.coverage?.topK || 5;
                    const passageOrder = fusedFull?.length ? fusedFull : searched;
                    const passages = await Promise.all(passageOrder.slice(0, passagesNeeded).map(async (hit) => {
                        const owner = ownerOf(table, hit.id);
                        const rec = await searchTiers[owner.tier.tier].reader.hydrate(owner.localId);
                        return rec?.text ?? '';
                    }));
                        // 4.5: "A reader MUST compute the union before
                    // scoring", so known_frac counts every layer's
                    // vocabulary, not only the base's.
                    const scored = await base.scoreQuality(
                        searched,
                        { text: queryOptions.text ?? '', vector, passages },
                        fusedFull,
                        { vocabBloom: chainBloom },
                    );
                    if (scored && scored.match_quality) {
                        matchQuality = scored.match_quality;
                        if (Number.isFinite(scored.confidence)) confidence = scored.confidence;
                        // Which passage grounded the verdict and which query
                        // words no passage contains, as the single-file
                        // reader returns it; recordId is the chain-global id
                        // of the passage coverage scored.
                        if (scored.grounding) {
                            grounding = { ...scored.grounding, recordId: passageOrder[scored.grounding.passageIndex]?.id ?? null };
                        }
                    }
                } else if (calibration.status === 'none') {
                    matchQuality = 'unscored';
                }
                // A 'none' verdict withholds results, exactly as the
                // single-file reader does: the calibrated abstention signal is
                // doing its job, and a chain must not be a way to get results
                // the base would have refused. showAbstained is the same
                // explicit opt-out, and never changes matchQuality.
                const withheld = matchQuality === 'none' && !queryOptions.showAbstained;
                return {
                    matchQuality,
                    ...(confidence === undefined ? {} : { confidence }),
                    results: withheld ? [] : results,
                    ...(grounding ? { grounding } : {}),
                    identity: chainMembers[chainMembers.length - 1].identity,
                    calibration: {
                        status: calibration.status,
                        drift: calibration.drift,
                        effectiveLimit: calibration.effectiveLimit,
                        producerEnvelope: calibration.producerEnvelope,
                        readerLimit: calibration.readerLimit,
                    },
                };
            },

            /** 5.6: a bare id is an id in the mounted head. */
            async record(id) {
                assertOpen();
                const owner = ownerOf(table, id);
                if (!owner) throw new Error(`id ${id} is outside every search tier's interval (5.2)`);
                const tier = searchTiers[owner.tier.tier];
                const record = await tier.reader.hydrate(owner.localId);
                const tombstoned = bitAt(headBitset, id);
                const sup = resolveSupersession(supersessions, id);
                return {
                    ...record, id, layer: tier.depth, tombstoned,
                    supersededBy: sup.immediate, successors: sup.all, currentSuccessor: sup.current,
                };
            },

            /** 5.6: a citation is (identity, id), resolved along the mounted ancestry. */
            citation(identity, id) {
                assertOpen();
                return resolveCitation(ancestry, identity, id);
            },

            /**
             * 6.2: a chain has no evaluation segment of its own, so this
             * surfaces the BASE's, tagged with where it came from. Callers use
             * it to run goldens against the chain; a golden written for the
             * base may legitimately fail here, because a layer can tombstone
             * or supersede the record it expects, so `evaluationScope` says
             * the material is the base's and not the chain's.
             */
            async evaluation() {
                assertOpen();
                if (typeof base.evaluation !== 'function') return null;
                const ev = await base.evaluation();
                if (!ev) return null;
                return { ...ev, evaluationScope: 'base', evaluationIdentity: base.identity };
            },

            chainBloom() { assertOpen(); return chainBloom; },

            /**
             * The stored bytes of each id's record, from its owning tier's
             * corpus segment and verified per record, in the order ids were
             * given. `compact` (6.3) copies these verbatim: record() returns
             * the parsed record with chain fields laid over it, and
             * re-serializing that is not the original bytes.
             */
            async __recordBytes(ids) {
                assertOpen();
                return Promise.all(ids.map(async (id) => {
                    const owner = ownerOf(table, id);
                    if (!owner) throw new Error(`id ${id} is outside every search tier's interval (5.2)`);
                    return searchTiers[owner.tier.tier].reader.readRecordBytes(owner.localId);
                }));
            },

            /**
             * 6.3's verbatim row copy: the quantized bytes and per-row
             * scale/offset for a set of chain ids, read back from whichever
             * tier owns each one, via the sketch reader's fetchRows() (format 2
             * verifies each row's digest) and the resident scales/offsets.
             * Producer seam, like __head. Copying, rather than re-embedding,
             * keeps the bytes without assuming a bit-identical encoder.
             *
             * Returns rows in the order ids were given. Throws if an id is
             * outside every tier, which callers should read as a planning bug
             * rather than a missing row.
             */
            async __fetchRows(ids) {
                assertOpen();
                // Group by owning tier so each tier's fetchRows sees one
                // batched call: the reader coalesces adjacent rows into single
                // range reads, which per-id calls would defeat on a URL mount.
                const byTier = new Map();
                const placed = ids.map((id, at) => {
                    const owner = ownerOf(table, id);
                    if (!owner) throw new Error(`id ${id} is outside every search tier's interval (5.2)`);
                    const t = owner.tier.tier;
                    if (!byTier.has(t)) byTier.set(t, []);
                    byTier.get(t).push({ at, localId: owner.localId });
                    return { tier: t, localId: owner.localId };
                });
                const out = new Array(ids.length);
                for (const [tierIndex, wanted] of byTier) {
                    const { reader } = searchTiers[tierIndex];
                    const fetched = await reader.sketch.fetchRows(wanted.map((w) => w.localId));
                    for (const w of wanted) {
                        const row = fetched.get(w.localId);
                        if (!row) throw new Error(`tier ${tierIndex} returned no row for local id ${w.localId}`);
                        out[w.at] = {
                            row,
                            scale: reader.sketch.scales[w.localId],
                            offset: reader.sketch.offsets[w.localId],
                        };
                    }
                }
                for (let i = 0; i < out.length; i++) {
                    if (!out[i]) throw new Error(`no row read back for id ${ids[i]} (tier ${placed[i].tier})`);
                }
                return out;
            },

            /**
             * The chain's members with each layer's own bitset and edges, which
             * is what `rebase` replays (6.2) and what `compact` reads to
             * enumerate live ids and rebuild the lineage segment (6.3).
             */
            get __members() {
                assertOpen();
                return chainMembers.map((m) => ({ ...m }));
            },

            /**
             * Everything `append` needs about the head to plan a new layer
             * (6.1). Internal, like __chainTier: a producer is part of this
             * package, and the names carry no compatibility promise.
             */
            get __head() {
                assertOpen();
                const last = chainMembers[chainMembers.length - 1];
                const baseMember = chainMembers[0];
                return {
                    identity: last.identity,
                    depth: last.depth,
                    // The head's own rowBase and record count, which together
                    // give the next layer's rowBase (3.3).
                    rowBase: last.depth === 0 ? 0 : last.rowBase,
                    records: last.records,
                    bitset: headBitset,
                    baseIdentity: base.identity,
                    baseRecords: baseMember.records,
                    // The base sketch's numeric metric code, which
                    // exportSketchArtifact writes verbatim. manifest.metric is
                    // the string form and is not interchangeable with it.
                    metricCode: base.sketch.metric,
                    // Sum of every layer's records, for the drift numerator.
                    appendedBefore: chainMembers.slice(1).reduce((n, m) => n + m.records, 0),
                    // 6.1 needs the base's kind to decide appendability, its
                    // ingestion declaration to enforce granularity, and its
                    // chunking so new records are cut the same way. The kind
                    // comes from the segment header verbatim, not from
                    // encoderInfo: kind 2's encoderInfo is shaped by the host
                    // declaration and carries no fixed name to match on.
                    qiKind: base.qiKind ?? null,
                    // The base's own encoder, resolved by the base's open.
                    passageEmbedder: () => base.passageEmbedder(),
                    // The base's query-interp segment bytes. `compact` copies
                    // them verbatim: compaction MUST NOT change the encoder,
                    // dim, metric, ingestion or tokenization (6.3), and the
                    // surest way not to is to carry the same bytes.
                    // The base's bloom geometry, which every layer's bloom
                    // MUST match so the chain's union is an exact bitwise OR
                    // (4.5). Null when the base carries no fit.
                    baseBloomGeometry: baseCal?.asset?.vocabBloom
                        ? {
                            bits: baseCal.asset.vocabBloom.bits,
                            hashes: baseCal.asset.vocabBloom.hashes,
                            minCount: baseCal.asset.vocabBloom.minCount ?? null,
                        }
                        : null,
                    readBaseQueryInterp: () => {
                        const seg = base.segments.get('query-interp');
                        if (!seg) return null;
                        return base.readSegmentBytes(seg);
                    },
                    corpusIngest: base.manifest.corpus?.ingest ?? null,
                    // The chain's ingestion declaration as validated at mount
                    // ({canonical, asserted, source}); null for a lone base,
                    // whose declaration is corpusIngest alone. rebase (6.2)
                    // compares two histories' declarations through this.
                    chainIngest: ingestDecl,
                    chunking: base.manifest.corpus?.ingest?.chunking
                        ?? base.manifest.corpus?.ingest
                        ?? null,
                    baseQueryInterpSha256: base.manifest.segments.find((sg) => sg.kind === 'query-interp')?.sha256 ?? null,
                    dim: base.dim,
                    metric: base.manifest.metric,
                    encoder: base.manifest.encoder,
                };
            },

            async close() {
                if (closed) return;
                closed = true;
                await closeAll();
            },
        };
    } catch (err) {
        await closeAll();
        throw err;
    }
}

function countBits(bytes) {
    let n = 0;
    for (let i = 0; i < bytes.length; i++) {
        let x = bytes[i];
        while (x) { n += x & 1; x >>= 1; }
    }
    return n;
}

function bitAt(bitset, id) {
    const byte = id >> 3;
    if (!bitset || byte >= bitset.length) return false;
    return ((bitset[byte] >> (id & 7)) & 1) === 1;
}

/**
 * Read one segment's bytes through the member's own reader. The member's open
 * already verified the segment digest where the profile requires it eagerly;
 * this is the bytes for structural parsing.
 */
async function readSegment(member, seg) {
    const tier = member.__chainTier || member.tier;
    if (tier && typeof tier.readSegmentBytes === 'function') return tier.readSegmentBytes(seg);
    throw new Error('this reader build does not expose segment bytes for chain parsing');
}

export { FUSION_DEFAULTS };
