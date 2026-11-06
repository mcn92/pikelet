// Reader-side matchQuality scorer. Mirrors the fit in pikelet/src/calibrate.mjs
// (and examples/static-wiki-pack/calibrate_abstention.mjs): retrieval signals
// (d0, margin, mean10) plus the known-token fraction from the shipped bloom,
// standardized and passed through the fitted logistic model.
//
// Verdicts: 'answer' (strong), 'weak' (shown with a caveat), 'abstain'.
// For self-templates-v6 assets (calibration.target 'relevance'), 'abstain'
// means the pack does not cover the query's topic, not that no passage
// states the fact. Older assets used a different target; the math is the same.
//
// asset.coverage, when present, adds coverage1: the share of the query's
// content words found in the top passages. It lives outside
// features[]/weights[] so a reader that predates it scores the topic-only
// model instead of NaN. Word rules ship in the asset so builder and reader
// cannot drift. There is no semantic (maxSim) grounding path; see
// docs/calibration.md for why it was removed.

// Upper bound on asset.coverage.topK. calibrate.mjs ships 5; every passage
// counted here is a record hydrated on every query, so a pack must not be
// able to turn one query into a hydration of its whole rerank window.
export const MAX_COVERAGE_TOP_K = 10;

export function createAbstentionScorer(asset, bloomBytes) {
    if (!asset || !Array.isArray(asset.weights) || !asset.thresholds) return null;
    // Thresholds that are missing, non-numeric or out of order compare false
    // against every p and would answer everything; a malformed asset must
    // degrade to unscored instead (callers map a null scorer to it).
    const { hard, weak } = asset.thresholds;
    if (!Number.isFinite(hard) || !Number.isFinite(weak) || hard < 0 || hard > weak || weak > 1) return null;
    const coverageCfg = asset.coverage && typeof asset.coverage.weight === 'number'
        && Number.isFinite(asset.coverage.mean) && Number.isFinite(asset.coverage.std)
        ? asset.coverage : null;
    if (coverageCfg && coverageCfg.topK !== undefined
        && !(Number.isInteger(coverageCfg.topK) && coverageCfg.topK >= 1 && coverageCfg.topK <= MAX_COVERAGE_TOP_K)) {
        return null;
    }
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
    // Light suffix stripping ("configure" grounds "configuration"). Must stay
    // byte-identical to calibrate.mjs's stem() (same suffix list and order).
    const STEM_MIN_LEN = 4;
    const STEM_SUFFIXES = ['ization', 'isation', 'ication', 'ation', 'ition', 'tion', 'ing', 'ed', 'ate', 'ize', 'ise'];
    function stem(w) {
        for (const suf of STEM_SUFFIXES) {
            if (w.length - suf.length >= STEM_MIN_LEN && w.endsWith(suf)) return w.slice(0, -suf.length);
        }
        if (w.length - 1 >= STEM_MIN_LEN && w.endsWith('e')) return w.slice(0, -1);
        return w;
    }
    // Byte-identical to calibrate.mjs's stripPossessive: the tokenizer keeps
    // apostrophes, so "darcy's" would never match "darcy".
    function stripPossessive(w) {
        if (w.endsWith("'s")) return w.slice(0, -2);
        if (w.endsWith("s'")) return w.slice(0, -1);
        if (w.endsWith("'")) return w.slice(0, -1);
        return w;
    }
    // Byte-identical to calibrate.mjs's isNumeral: numerals skip the
    // coverageMinLen floor ("Chamber 4" vs "Chamber 5").
    const isNumeral = (w) => /^\d+$/.test(w);
    // A heading-only match grounds at reduced credit, byte-identical to
    // calibrate.mjs's HEADING_COVERAGE_WEIGHT.
    const HEADING_COVERAGE_WEIGHT = 0.5;
    // Returns { value, grounding } instead of writing to shared state, so two
    // concurrent queries on the same open pack cannot read each other's
    // grounding.
    function coverageFrac(text, passages) {
        const content = (String(text).toLowerCase().match(/[a-z0-9']+/g) || [])
            .map(stripPossessive)
            .filter((w) => (w.length >= coverageMinLen || isNumeral(w)) && !coverageStopwords.has(w));
        if (!content.length) return { value: 0, grounding: null };
        const weights = content.map((w) => (isCommon && isCommon(w) ? commonWordWeight : 1));
        const weightSum = weights.reduce((a, c) => a + c, 0);
        let best = 0; let bestGrounding = null; let passageIndex = 0;
        for (const { heading, body } of passages || []) {
            const bodyWords = (String(body || '').toLowerCase().match(/[a-z0-9']+/g) || []).map(stripPossessive);
            const bodySet = new Set(bodyWords);
            const bodyStems = new Set(bodyWords.map(stem));
            const headingSet = new Set((String(heading || '').toLowerCase().match(/[a-z0-9']+/g) || []).map(stripPossessive));
            const presentIn = (set, stems, w) => set.has(w) || set.has(`${w}s`) || set.has(`${w}es`)
                || (w.endsWith('s') && set.has(w.slice(0, -1)))
                || (stems && stems.has(stem(w)));
            const creditFor = (w) => {
                if (presentIn(bodySet, bodyStems, w)) return 1;
                if (presentIn(headingSet, null, w)) return HEADING_COVERAGE_WEIGHT;
                return 0;
            };
            const grounded = content.reduce((sum, w, i) => sum + creditFor(w) * weights[i], 0);
            const frac = grounded / weightSum;
            if (frac > best) {
                best = frac;
                bestGrounding = { passageIndex, coverage: +frac.toFixed(3), covered: content.filter((w) => creditFor(w) > 0), uncovered: content.filter((w) => creditFor(w) === 0) };
            }
            passageIndex++;
        }
        return { value: best, grounding: bestGrounding };
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

    // `override` is the layered profile's union bloom (LAYERED_PROFILE.md
    // 4.5): the bitwise OR of the base's bloom and every mounted layer's,
    // which a chain MUST score against rather than the base's alone.
    // Identical geometry is enforced where the union is built, and re-checked
    // here because reading a different-sized bloom with these `bits` would
    // silently probe the wrong positions rather than fail.
    function knownFrac(text, override) {
        const table = override || bloom;
        if (table !== bloom && table.length !== bloom.length) {
            throw new Error(`abstention: bloom override is ${table.length} bytes but the fit's is ${bloom.length}; `
                + 'known_frac would probe the wrong bits');
        }
        const words = text.toLowerCase().match(/[a-z0-9']+/g) || [];
        if (!words.length) return 0;
        let known = 0;
        for (const w of words) {
            const hit = SEEDS.every((seed) => {
                const bit = fnv1a(w, seed);
                return (table[bit >> 3] >> (bit & 7)) & 1;
            });
            if (hit) known++;
        }
        return known / words.length;
    }

    return {
        // When usesPassage is true the caller must hydrate the top passagesNeeded
        // results' text before scoring; coverage reads them.
        usesPassage: !!coverageCfg,
        passagesNeeded: coverageCfg ? (coverageCfg.topK || 1) : 0,
        async score(queryText, results, passageTexts, options = {}) {
            // The fit always scores a fixed top-10 window (K = min(10, candidates)),
            // whatever k the caller passes. Slicing to k first would change the
            // verdict for an identical retrieval.
            const top = results.slice(0, Math.min(10, results.length));
            const d0 = top.length ? top[0].distance : 1;
            const margin = top.length > 1
                ? top[Math.min(4, top.length - 1)].distance - d0 : 0;
            const mean10 = top.length
                ? top.reduce((s, r) => s + r.distance, 0) / top.length : 1;
            const signals = { d0, margin, mean10, known_frac: knownFrac(queryText, options.vocabBloom) };
            let z = asset.bias;
            asset.features.forEach((f, j) => {
                z += ((signals[f] - asset.standardize.mean[f]) / asset.standardize.std[f]) * asset.weights[j];
            });
            // Local to this call so concurrent queries never share grounding detail.
            let grounding = null;
            if (coverageCfg) {
                // passageTexts covers the fixed top-COVERAGE_TOP_PASSAGES window, matching
                // calibrate.mjs's top.slice(0, COVERAGE_TOP_PASSAGES).
                const passages = Array.isArray(passageTexts) ? passageTexts : [passageTexts];
                const coverage = coverageFrac(queryText, passages);
                signals.coverage1 = coverage.value;
                grounding = coverage.grounding;
                // grounding1 is coverage1 alone (see docs/calibration.md).
                z += ((signals.coverage1 - coverageCfg.mean) / (coverageCfg.std || 1)) * coverageCfg.weight;
            }
            // A malformed asset (unknown feature name, non-numeric term) must
            // degrade to unscored, never to a NaN that compares false against
            // both thresholds and answers everything.
            if (!Number.isFinite(z)) return { p: null, verdict: 'unscored', signals };
            const p = 1 / (1 + Math.exp(-z));
            const verdict = p < asset.thresholds.hard ? 'abstain'
                : p < asset.thresholds.weak ? 'weak' : 'answer';
            return { p, verdict, signals, grounding };
        },
    };
}
