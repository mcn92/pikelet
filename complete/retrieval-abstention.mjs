// Client-side abstention scorer for the wiki pack. Mirrors the math in
// ../../calibrate_abstention.mjs exactly: retrieval signals (d0, margin,
// mean10) plus the corpus-vocabulary known-token fraction from the shipped
// bloom filter, standardized and passed through the fitted logistic model.
// Verdict semantics: 'answer' (strong match), 'weak' (closest match is
// distant — shown with a caveat), 'abstain' (nothing useful in the pack).
//
// Assets calibrated by pikelet's self-templates-v2+ may carry
// an additional grounding term (asset.coverage): the fraction of the query's
// content words that appear in the top retrieved passage's text. Every base
// feature measures topic similarity, so "the corpus discusses this area" and
// "this passage answers this question" are indistinguishable without it. The
// term is serialized outside features[]/weights[] deliberately: a reader
// that predates it scores the topic-only model against the same thresholds
// (a conservative degradation) instead of hitting an unknown feature name
// and computing NaN. Word rules (min length, stopwords) ship in the asset so
// builder and reader cannot drift.

export function createAbstentionScorer(asset, bloomBytes) {
    if (!asset || !Array.isArray(asset.weights) || !asset.thresholds) return null;
    const coverageCfg = asset.coverage && typeof asset.coverage.weight === 'number'
        && Number.isFinite(asset.coverage.mean) && Number.isFinite(asset.coverage.std)
        ? asset.coverage : null;
    const coverageStopwords = coverageCfg ? new Set(coverageCfg.stopwords || []) : null;
    const coverageMinLen = coverageCfg ? (coverageCfg.minWordLen || 3) : 3;
    // Words the corpus uses everywhere ground any query that mentions them,
    // so they are excluded from coverage; the builder ships them as a bloom.
    const commonBloom = coverageCfg?.commonBloom?.base64
        ? (typeof Buffer !== 'undefined'
            ? new Uint8Array(Buffer.from(coverageCfg.commonBloom.base64, 'base64'))
            : Uint8Array.from(atob(coverageCfg.commonBloom.base64), (c) => c.charCodeAt(0)))
        : null;
    const commonBits = coverageCfg?.commonBloom?.bits;
    const isCommon = commonBloom ? (w) => SEEDS.every((seed) => {
        let h = 0x811c9dc5 ^ seed;
        for (let i = 0; i < w.length; i++) { h ^= w.charCodeAt(i); h = Math.imul(h, 0x01000193); }
        const bit = (h >>> 0) % commonBits;
        return (commonBloom[bit >> 3] >> (bit & 7)) & 1;
    }) : null;

    // Max over the scored passages, with corpus-common words counted at
    // reduced weight — mirrors pikelet/src/calibrate.mjs
    // coverageFrac exactly (the weight ships in the asset).
    const commonWordWeight = coverageCfg?.commonWordWeight ?? 1 / 3;
    function coverageFrac(text, passageTexts) {
        const content = (String(text).toLowerCase().match(/[a-z0-9']+/g) || [])
            .filter((w) => w.length >= coverageMinLen && !coverageStopwords.has(w));
        if (!content.length) return 0;
        const weights = content.map((w) => (isCommon && isCommon(w) ? commonWordWeight : 1));
        const weightSum = weights.reduce((a, c) => a + c, 0);
        let best = 0;
        for (const passageText of passageTexts || []) {
            const passage = new Set(String(passageText || '').toLowerCase().match(/[a-z0-9']+/g) || []);
            const present = (w) => passage.has(w) || passage.has(`${w}s`) || passage.has(`${w}es`)
                || (w.endsWith('s') && passage.has(w.slice(0, -1)));
            const grounded = content.reduce((sum, w, i) => sum + (present(w) ? weights[i] : 0), 0);
            best = Math.max(best, grounded / weightSum);
        }
        return best;
    }
    const bloom = new Uint8Array(bloomBytes);
    const bits = asset.vocabBloom.bits;

    function fnv1a(str, seed) {
        let h = 0x811c9dc5 ^ seed;
        for (let i = 0; i < str.length; i++) {
            h ^= str.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        return (h >>> 0) % bits;
    }
    const SEEDS = [0, 0x9e3779b9];

    function knownFrac(text) {
        const words = text.toLowerCase().match(/[a-z0-9']+/g) || [];
        if (!words.length) return 0;
        let known = 0;
        for (const w of words) {
            const hit = SEEDS.every((seed) => {
                const bit = fnv1a(w, seed);
                return (bloom[bit >> 3] >> (bit & 7)) & 1;
            });
            if (hit) known++;
        }
        return known / words.length;
    }

    return {
        // The caller must hydrate the top passagesNeeded results' text
        // before scoring when usesPassage is true; the coverage term reads
        // them (older assets scored one passage; topK now defaults to it).
        usesPassage: !!coverageCfg,
        passagesNeeded: coverageCfg ? (coverageCfg.topK || 1) : 0,
        score(queryText, results, passageTexts) {
            // Fit at build time (pikelet/src/calibrate.mjs) always scores a
            // fixed top-10 window (K = min(10, candidates)), independent of
            // whatever k a caller later passes to query(). Slicing to the
            // caller's k here before windowing would shrink margin/mean10/
            // coverage's inputs at small k and change the verdict for an
            // identical retrieval — the window, not the caller's k, is what
            // must match the fit.
            const top = results.slice(0, Math.min(10, results.length));
            const d0 = top.length ? top[0].distance : 1;
            const margin = top.length > 1
                ? top[Math.min(4, top.length - 1)].distance - d0 : 0;
            const mean10 = top.length
                ? top.reduce((s, r) => s + r.distance, 0) / top.length : 1;
            const signals = { d0, margin, mean10, known_frac: knownFrac(queryText) };
            let z = asset.bias;
            asset.features.forEach((f, j) => {
                z += ((signals[f] - asset.standardize.mean[f]) / asset.standardize.std[f]) * asset.weights[j];
            });
            if (coverageCfg) {
                // passageTexts is hydrated by the caller for the fixed
                // top-COVERAGE_TOP_PASSAGES window (see passagesNeeded
                // below), independent of the caller's k, matching
                // calibrate.mjs's top.slice(0, COVERAGE_TOP_PASSAGES).
                signals.coverage1 = coverageFrac(queryText, Array.isArray(passageTexts) ? passageTexts : [passageTexts]);
                z += ((signals.coverage1 - coverageCfg.mean) / (coverageCfg.std || 1)) * coverageCfg.weight;
            }
            // A malformed asset (unknown feature name, non-numeric term) must
            // degrade to unscored, never to a NaN that compares false against
            // both thresholds and answers everything.
            if (!Number.isFinite(z)) return { p: null, verdict: 'unscored', signals };
            const p = 1 / (1 + Math.exp(-z));
            const verdict = p < asset.thresholds.hard ? 'abstain'
                : p < asset.thresholds.weak ? 'weak' : 'answer';
            return { p, verdict, signals };
        },
    };
}
