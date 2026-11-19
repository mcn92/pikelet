// Grounding: which of the query's content words the best of the top
// passages contains (covered) and which none of them does (uncovered). It
// needs nothing from the pack but the passages' text, so every pack gets
// it, calibrated or not; an agent reads `uncovered` to tell "the passage is
// about this" from "the passage states this".
//
// The word rules follow the calibrated coverage term
// (retrieval-abstention.mjs) — possessives stripped, light suffix stemming,
// plural forms, heading-only matches at half credit — with two
// differences: a token holding a digit counts whatever its length ("v8",
// "3d", "4"), and every word weighs the same (the calibrated term discounts
// corpus-common words through a bloom only a calibration asset carries).
// The calibrated verdict keeps its own coverage term unchanged.

// How many passages grounding reads, in fused order.
export const GROUNDING_PASSAGES = 5;

const MIN_WORD_LEN = 3;
const STOPWORDS = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'into',
    'through', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these', 'those',
    'as', 'their', 'they', 'them', 'also', 'can', 'will', 'which', 'when', 'where', 'while', 'his', 'her', 'has',
    'have', 'had', 'not', 'often', 'usually', 'without',
    // Question scaffolding, so "how does X work" is grounded by X.
    'what', 'how', 'who', 'why', 'do', 'does', 'did', 'me', 'tell', 'about', 'explain', 'explained', 'facts',
    'information', 'overview', 'history', 'definition', 'important', 'work', 'works', 'known', 'anything', 'said',
    'much', 'many', 'each', 'would', 'could', 'some', 'such', 'very', 'more', 'after', 'before',
]);
const HEADING_WEIGHT = 0.5;
const STEM_MIN_LEN = 4;
const STEM_SUFFIXES = ['ization', 'isation', 'ication', 'ation', 'ition', 'tion', 'ing', 'ed', 'ate', 'ize', 'ise'];

const tokenize = (text) => String(text || '').toLowerCase().match(/[a-z0-9']+/g) || [];
function stem(w) {
    for (const suf of STEM_SUFFIXES) {
        if (w.length - suf.length >= STEM_MIN_LEN && w.endsWith(suf)) return w.slice(0, -suf.length);
    }
    if (w.length - 1 >= STEM_MIN_LEN && w.endsWith('e')) return w.slice(0, -1);
    return w;
}
function stripPossessive(w) {
    if (w.endsWith("'s")) return w.slice(0, -2);
    if (w.endsWith("s'")) return w.slice(0, -1);
    if (w.endsWith("'")) return w.slice(0, -1);
    return w;
}

// The query words grounding grades, in query order, without repeats.
export function groundingWords(query) {
    const words = tokenize(query).map(stripPossessive)
        .filter((w) => w && (w.length >= MIN_WORD_LEN || /\d/.test(w)) && !STOPWORDS.has(w));
    return [...new Set(words)];
}

/**
 * Grade the query's content words against passage texts (a record's first
 * line is its heading, as ingest writes it). Returns the best passage's
 * { passageIndex, coverage, covered, uncovered }, or null when the query has
 * no gradable word or there are no passages.
 */
export function groundQuery(query, passageTexts) {
    const content = groundingWords(query);
    if (!content.length || !passageTexts?.length) return null;
    let best = null;
    let bestRaw = -1;
    passageTexts.forEach((text, passageIndex) => {
        const s = String(text || '');
        const nl = s.indexOf('\n');
        const heading = nl === -1 ? s : s.slice(0, nl);
        const body = nl === -1 ? '' : s.slice(nl + 1);
        const bodyWords = tokenize(body).map(stripPossessive);
        const bodySet = new Set(bodyWords);
        const bodyStems = new Set(bodyWords.map(stem));
        const headingSet = new Set(tokenize(heading).map(stripPossessive));
        const presentIn = (set, stems, w) => set.has(w) || set.has(`${w}s`) || set.has(`${w}es`)
            || (w.endsWith('s') && set.has(w.slice(0, -1)))
            || (stems !== null && stems.has(stem(w)));
        const credit = (w) => (presentIn(bodySet, bodyStems, w) ? 1 : presentIn(headingSet, null, w) ? HEADING_WEIGHT : 0);
        const credits = content.map(credit);
        const coverage = credits.reduce((a, c) => a + c, 0) / content.length;
        if (coverage > bestRaw) {
            bestRaw = coverage;
            best = {
                passageIndex,
                coverage: +coverage.toFixed(3),
                covered: content.filter((_, i) => credits[i] > 0),
                uncovered: content.filter((_, i) => credits[i] === 0),
            };
        }
    });
    return best;
}
