import fs from 'node:fs/promises';
import path from 'node:path';

// Build-time matchQuality calibration for complete kind-3 artifacts.
//
// Fits a logistic model on the reader's own retrieval signals (d0, margin,
// mean10, known_frac) plus coverage1, so matchQuality measures retrieval
// relevance: positives are retrieval-verified template queries about the
// corpus, negatives are off-topic queries (a foreign bank plus gibberish).
// The hard threshold sits at a high percentile of off-topic probability.
// Hard negatives (ablation, entity-swap, held-out-document) are generated
// only to report cvAucHard; they do not shape the fit. Human-written
// questions, when the corpus ships them, are held out as validation.
//
// The scoring math mirrors the reader (complete/retrieval-abstention.mjs)
// and the wiki pack calibrator (examples/static-wiki-pack/
// calibrate_abstention.mjs). Why the probe classes are built this way, and
// the approaches tried and rejected, are in docs/calibration.md.
//
// Returns { calibrationJson, summary }, or null (with a logged reason) when
// the corpus cannot support a trustworthy fit; the caller then ships the
// unscored placeholder.

const SEED = 424242;
// The four base features measure topic similarity only. coverage1 (the
// share of the query's content words found in a top passage) joins the fit
// as a fifth feature, but is serialized as asset.coverage rather than a
// features[] entry, so older readers score the topic-only model instead of
// an unknown feature name.
const BASE_FEATS = ['d0', 'margin', 'mean10', 'known_frac'];
const COVERAGE_FEAT = 'coverage1';
const MAXSIM_FEAT = 'maxSim1';
// grounding1 is plain coverage1. maxSim1 is computed for comparison
// telemetry only; blending it in made real-query behavior worse (see
// docs/calibration.md, "Approaches tried and rejected").
const GROUNDING_FEAT = COVERAGE_FEAT;
const FEATS = [...BASE_FEATS, GROUNDING_FEAT];
const COVERAGE_MIN_WORD_LEN = 3;
const COVERAGE_TOP_PASSAGES = 5;
// Held-out-document hard negatives: how many title-template questions to
// ask, and what share of the corpus's chunks the sampled documents cover.
const HELDOUT_QUERIES = 36;
const HELDOUT_CHUNK_RATE = 0.12;
const BLOOM_SEEDS = [0, 0x9e3779b9];
const MAX_POSITIVES = 96;
const GIBBERISH_QUERIES = 24;
const MIN_VERIFIED_POSITIVES = 4;
const MIN_NEGATIVES = 8;
const MIN_AUC = 0.85;
// A foreign-bank query whose best passages share at least this fraction of
// its content words is not off-topic here (a docs corpus can discuss "docker
// compose port binding"); it is kept out of the fit and reported as
// eval-only overlap.
const FOREIGN_OVERLAP_COVERAGE = 0.5;
// Hard threshold: this percentile of off-topic probability.
const OFFTOPIC_HARD_QUANTILE = 0.95;
// Encoder-guided substitution positives: sample up to SUBSTITUTION_BUDGET
// base positives, swap from the SUBSTITUTION_VOCAB_TOP most frequent corpus
// words, and keep a swap only above SUBSTITUTION_MIN_COS word cosine (0.55
// measured as where swaps stay related, not incoherent).
const SUBSTITUTION_BUDGET = 32;
const SUBSTITUTION_VOCAB_TOP = 800;
const SUBSTITUTION_MIN_COS = 0.55;
// Real-query gate: a fit that abstains on more than a third of held-out
// human-written (or substituted) queries does not ship, whatever its AUC.
const MAX_PARAPHRASE_ABSTENTION_RATE = 1 / 3;
// Below this many rows a rate is one row's outcome, so the gate above is
// skipped (not loosened); the cross-validated AUC gate still applies.
const MIN_PARAPHRASE_STRESS_SAMPLE = 8;
// Threshold rails. Perfectly separable probes saturate the logistic and
// push percentile thresholds too high: the ceiling stops a hard bar above
// 0.5, and the band floor keeps a real "weak" region when no weak probes
// exist.
const HARD_CEILING = 0.5;
const MIN_WEAK_BAND = 0.08;

// Off-domain but real-English queries, spanning consumer, developer, finance,
// and household domains. Same role as the wiki calibrator's foreign title
// bank: teach the scorer what a well-formed query with no corpus support
// looks like, beyond what gibberish covers.
const FOREIGN_TITLE_BANK = [
  '1040 tax form earned income credit', 'mortgage escrow shortage statement',
  'kubernetes crashloopbackoff', 'react usestate batching',
  'postgres vacuum analyze', 'aws iam role trust policy',
  'stripe webhook signature verification', 'chase sapphire annual fee waiver',
  'tsa precheck renewal appointment', 'iphone battery health service message',
  'netgear router admin password reset', 'excel vlookup not available error',
  'docker compose port binding', 'github actions cache miss',
  'python nonetype subscriptable error', 'medicare part d formulary exception',
  'irs quarterly estimated tax payment', 'student loan income driven repayment',
  'car alternator belt squeal', 'ev charger nema 14-50 permit',
  'tenant security deposit demand letter', 'small claims filing fee',
  'hsa eligible expense receipt', 'merchant chargeback reason code',
  'oauth redirect uri mismatch', 'terraform state lock',
  'nginx reverse proxy websocket upgrade', 'redis eviction policy',
  'pandas dataframe groupby transform', 'homeowners insurance deductible claim',
  'credit card balance transfer fee', 'passport expedited renewal appointment',
  'property tax homestead exemption', 'medical prior authorization denial',
  'printer offline windows settings', 'wifi mesh backhaul channel',
  'airbnb host cancellation refund', 'uber driver cancelled ride refund',
  'shopify abandoned cart email', 'quickbooks payroll tax deposit',
  'salesforce validation rule formula', 'figma component variant property',
  'blender cycles render noise', 'unity rigidbody collision layer',
  'rust borrow checker lifetime error', 'go module replace directive',
  'java maven dependency conflict', 'android gradle signing config',
  'ios provisioning profile expired', 'linux systemd service restart loop',
  'windows bitlocker recovery key', 'router dns over https setting',
  'zelle payment pending review', 'venmo instant transfer fee',
  'mortgage refinance closing disclosure', 'california dmv real id appointment',
  'new york parking ticket dispute', 'texas franchise tax no tax due report',
  'college fafsa dependency override', 'w2 corrected form box 12 code',
  'health insurance out of network appeal', 'clinic cpt code billing modifier',
  'jira workflow transition condition', 'slack app manifest oauth scope',
  'zoom webinar registration limit', 'mailchimp dkim authentication',
  'cloudflare cname flattening', 'dns txt spf include limit',
  'elasticsearch shard relocation', 'snowflake warehouse auto suspend',
  'datadog log exclusion filter', 'prometheus alertmanager silence',
  'kafka consumer group lag', 's3 lifecycle transition rule',
  'azure managed identity role assignment', 'gcp service account key rotation',
  'kubernetes ingress tls secret', 'nextjs server action form data',
  'tailwind container query plugin', 'vite dependency prebundle cache',
  'playwright trace viewer',
];

const TEMPLATES = [
  (t) => `what is ${t}`,
  (t) => `tell me about ${t}`,
  (t) => `${t} explained`,
  (t) => `facts about ${t}`,
  (t) => `information about ${t}`,
  (t) => `overview of ${t}`,
  (t) => `history of ${t}`,
  (t) => `definition of ${t}`,
  (t) => `who is ${t}`,
  (t) => `where is ${t}`,
  (t) => `why is ${t} important`,
  (t) => `how does ${t} work`,
];

function rng(seed) {
  let state = seed >>> 0;
  return () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 0xffffffff);
}

function sample(items, n, seed) {
  const next = rng(seed);
  const picked = [];
  const used = new Set();
  while (picked.length < n && used.size < items.length) {
    const i = Math.floor(next() * items.length);
    if (used.has(i)) continue;
    used.add(i);
    picked.push(items[i]);
  }
  return picked;
}

// Pick whole documents (grouped by title) for held-out-document negatives:
// greedy over shuffled titles until ~HELDOUT_CHUNK_RATE of chunks are
// covered, at least two and at most 12 titles, always leaving two untouched.
// Corpora under three titles get none.
function chooseHeldOutTitles(chunks, titles, seed) {
  if (titles.length < 3) return { titles: new Set(), chunks: 0 };
  const byTitle = new Map();
  for (const c of chunks) {
    const t = (c.title || '').trim();
    if (t) byTitle.set(t, (byTitle.get(t) || 0) + 1);
  }
  const targetChunks = Math.max(1, Math.round(chunks.length * HELDOUT_CHUNK_RATE));
  const heldOut = new Set();
  let count = 0;
  for (const title of sample(titles, titles.length, seed)) {
    if (titles.length - heldOut.size <= 2 || heldOut.size >= 12) break;
    heldOut.add(title);
    count += byTitle.get(title) || 0;
    if (count >= targetChunks && heldOut.size >= 2) break;
  }
  return { titles: heldOut, chunks: count };
}

function titleQuestions(titles, n, seed) {
  const per = Math.min(TEMPLATES.length, Math.max(1, Math.floor(n / titles.length)));
  const seen = new Set();
  return sample(titles, Math.min(n, titles.length * TEMPLATES.length), seed)
    .flatMap((t, i) => Array.from({ length: per }, (_, j) => ({
      text: TEMPLATES[(i + j) % TEMPLATES.length](t.toLowerCase()),
      sourceTitle: t,
    })))
    .filter(({ text }) => !seen.has(text) && seen.add(text))
    .slice(0, n);
}

const STOPWORDS = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'into', 'through', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these', 'those', 'as', 'their', 'they', 'them', 'also', 'can', 'will', 'which', 'when', 'where', 'while', 'his', 'her', 'has', 'have', 'had', 'not', 'often', 'usually', 'without']);

// Content-word positives, rotated through question frames. Title templates
// retrieve too easily and bare keyword runs are too narrow a phrasing
// distribution; both let the hard threshold creep up until real questions
// abstain.
const CONTENT_FRAMES = [
  (w) => w,
  (w) => `what is known about ${w}`,
  (w) => `which ${w}`,
  (w) => `is anything said about ${w}`,
];

function contentWordQuestions(chunks, n, seed, eligiblePos = null) {
  const next = rng(seed);
  const pool = (eligiblePos ?? chunks.map((_, pos) => pos)).map((pos) => ({ chunk: chunks[pos], pos }));
  const picked = sample(pool, Math.min(n, pool.length * 2), seed ^ 0x77aa11);
  const seen = new Set();
  const out = [];
  let frame = 0;
  for (const { chunk, pos } of picked) {
    const words = [...new Set(tokenize(chunk.text).filter((w) => w.length >= 4 && !STOPWORDS.has(w)))];
    if (words.length < 3) continue;
    const count = Math.min(3 + Math.floor(next() * 3), words.length);
    const start = Math.floor(next() * Math.max(1, words.length - count));
    const text = CONTENT_FRAMES[frame++ % CONTENT_FRAMES.length](words.slice(start, start + count).join(' '));
    if (seen.has(text)) continue;
    seen.add(text);
    out.push({ text, sourceId: pos });
    if (out.length >= n) break;
  }
  return out;
}

const tokenize = (text) => (text.toLowerCase().match(/[a-z0-9']+/g) || []);

// Question/template scaffolding is stopped on top of the base stopwords, so
// "how does X work" is grounded by X. Words that are also ordinary content
// words in technical prose ("make", "like", "other", "only") are left in.
// The list ships in the asset so the reader's copy cannot drift.
const COVERAGE_STOPWORDS = new Set([...STOPWORDS,
  'what', 'how', 'who', 'why', 'where', 'when', 'do', 'does', 'did', 'me',
  'tell', 'about', 'explain', 'explained', 'facts', 'information',
  'overview', 'history', 'definition', 'important', 'work', 'works',
  'known', 'anything', 'said',
  'much', 'many', 'each', 'would', 'could', 'some', 'such', 'very',
  'more', 'after', 'before']);

// Coverage weights words by informativeness: corpus-common words (chunk
// document frequency over ~5%, shipped as a bloom) count at reduced weight,
// not zero. The score is the MAX over the top COVERAGE_TOP_PASSAGES passages,
// not their union, so weak partial matches cannot add up.
const COVERAGE_COMMON_WORD_WEIGHT = 1 / 3;
// Light suffix stripping so "configure" grounds against "configuration".
// Not a general stemmer: only suffix families measured to matter, with
// MIN_STEM_LEN guarding short words. Must stay identical (rules and order)
// to complete/retrieval-abstention.mjs, which scores with the same function.
const STEM_MIN_LEN = 4;
const STEM_SUFFIXES = ['ization', 'isation', 'ication', 'ation', 'ition', 'tion', 'ing', 'ed', 'ate', 'ize', 'ise'];
function stem(w) {
  for (const suf of STEM_SUFFIXES) {
    if (w.length - suf.length >= STEM_MIN_LEN && w.endsWith(suf)) return w.slice(0, -suf.length);
  }
  if (w.length - 1 >= STEM_MIN_LEN && w.endsWith('e')) return w.slice(0, -1);
  return w;
}
// tokenize() keeps apostrophes, so "darcy's" would never match "darcy".
// Strip a trailing 's or ' from both query and passage words.
function stripPossessive(w) {
  if (w.endsWith("'s")) return w.slice(0, -2);
  if (w.endsWith("s'")) return w.slice(0, -1);
  if (w.endsWith("'")) return w.slice(0, -1);
  return w;
}
// A heading-only match grounds at reduced credit: a heading names its
// section's topic, but full credit would hand title-templated queries free
// coverage.
const HEADING_COVERAGE_WEIGHT = 0.5;
// Numerals are exempt from COVERAGE_MIN_WORD_LEN: "Chamber 4" vs "Chamber 5"
// differ only in a one-character token, and it is the most identifying
// word in the query.
const isNumeral = (w) => /^\d+$/.test(w);
function coverageFrac(text, passages, isCommon) {
  const content = tokenize(text).map(stripPossessive)
    .filter((w) => (w.length >= COVERAGE_MIN_WORD_LEN || isNumeral(w)) && !COVERAGE_STOPWORDS.has(w));
  if (!content.length) return 0;
  const weights = content.map((w) => (isCommon && isCommon(w) ? COVERAGE_COMMON_WORD_WEIGHT : 1));
  const weightSum = weights.reduce((a, c) => a + c, 0);
  let best = 0;
  for (const { heading, body } of passages) {
    const bodyWords = tokenize(body || '').map(stripPossessive);
    const bodySet = new Set(bodyWords);
    const bodyStems = new Set(bodyWords.map(stem));
    const headingSet = new Set(tokenize(heading || '').map(stripPossessive));
    const presentIn = (set, stems, w) => set.has(w) || set.has(`${w}s`) || set.has(`${w}es`)
      || (w.endsWith('s') && set.has(w.slice(0, -1)))
      || (stems && stems.has(stem(w)));
    const creditFor = (w) => {
      if (presentIn(bodySet, bodyStems, w)) return 1;
      if (presentIn(headingSet, null, w)) return HEADING_COVERAGE_WEIGHT;
      return 0;
    };
    const grounded = content.reduce((sum, w, i) => sum + creditFor(w) * weights[i], 0);
    best = Math.max(best, grounded / weightSum);
  }
  return best;
}

// MaxSim grounding (comparison telemetry only, not in the fit): per query
// word, the best embedding cosine against the passage's content words,
// weighted like coverageFrac. Raw cosine has no real zero (unrelated corpus
// words score ~0.45-0.55), so estimateMaxSimBaseline measures this corpus's
// chance similarity and maxSimFrac rescales it to 0, with MAXSIM_GAMMA
// damping the marginal range above it.
const MAXSIM_BASELINE_PAIRS = 400;
const MAXSIM_BASELINE_PERCENTILE = 0.95;
const MAXSIM_GAMMA = 3;
async function estimateMaxSimBaseline(vocabWords, embedWordVec, seed) {
  if (vocabWords.length < 2) return 0;
  const cosine = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d; };
  const next = rng(seed);
  const sims = [];
  const n = Math.min(MAXSIM_BASELINE_PAIRS, Math.floor((vocabWords.length * (vocabWords.length - 1)) / 2));
  let attempts = 0;
  while (sims.length < n && attempts < n * 20) {
    attempts++;
    const i = Math.floor(next() * vocabWords.length);
    let j = Math.floor(next() * vocabWords.length);
    if (j === i) j = (j + 1) % vocabWords.length;
    const a = vocabWords[i], b = vocabWords[j];
    if (a === b) continue;
    // Suffix-sharing pairs ("configure"/"configuration") are genuinely
    // related, not a noise sample — excluded so the baseline measures
    // unrelated-word similarity, not near-duplicate-word similarity.
    if (a.startsWith(b) || b.startsWith(a)) continue;
    sims.push(cosine(await embedWordVec(a), await embedWordVec(b)));
  }
  if (!sims.length) return 0;
  sims.sort((x, y) => x - y);
  return sims[Math.min(sims.length - 1, Math.floor(sims.length * MAXSIM_BASELINE_PERCENTILE))];
}

async function maxSimFrac(text, passages, isCommon, embedWordVec, baseline = 0) {
  const content = tokenize(text).filter((w) => w.length >= COVERAGE_MIN_WORD_LEN && !COVERAGE_STOPWORDS.has(w));
  if (!content.length) return 0;
  const weights = content.map((w) => (isCommon && isCommon(w) ? COVERAGE_COMMON_WORD_WEIGHT : 1));
  const weightSum = weights.reduce((a, c) => a + c, 0);
  const queryVecs = await Promise.all(content.map((w) => embedWordVec(w)));
  const cosine = (a, b) => {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot; // both sides are already L2-normalized by embedWords()
  };
  // Map the corpus baseline to 0 and cosine 1 to 1, then damp with
  // MAXSIM_GAMMA. At or below baseline is chance similarity: 0.
  const rescale = (sim) => (baseline >= 1 ? 0 : Math.max(0, (sim - baseline) / (1 - baseline)) ** MAXSIM_GAMMA);
  let best = 0;
  for (const { heading, body } of passages) {
    const bodyWords = [...new Set(tokenize(body || '').filter((w) => w.length >= COVERAGE_MIN_WORD_LEN))];
    const headingWords = [...new Set(tokenize(heading || '').filter((w) => w.length >= COVERAGE_MIN_WORD_LEN))];
    if (!bodyWords.length && !headingWords.length) continue;
    const bodyVecs = await Promise.all(bodyWords.map((w) => embedWordVec(w)));
    const headingVecs = await Promise.all(headingWords.map((w) => embedWordVec(w)));
    let simSum = 0;
    for (let i = 0; i < content.length; i++) {
      let simBest = 0;
      for (const bv of bodyVecs) simBest = Math.max(simBest, cosine(queryVecs[i], bv));
      let simHeading = 0;
      for (const hv of headingVecs) simHeading = Math.max(simHeading, cosine(queryVecs[i], hv));
      simBest = Math.max(simBest, simHeading * HEADING_COVERAGE_WEIGHT);
      simSum += rescale(simBest) * weights[i];
    }
    best = Math.max(best, simSum / weightSum);
  }
  return best;
}

function buildCommonWordsBloom(chunks) {
  const df = new Map();
  for (const chunk of chunks) {
    for (const w of new Set(tokenize(chunk.text))) df.set(w, (df.get(w) || 0) + 1);
  }
  const dfCap = Math.max(4, Math.round(chunks.length * 0.05));
  const common = [...df.entries()]
    .filter(([w, c]) => c > dfCap && w.length >= COVERAGE_MIN_WORD_LEN && !COVERAGE_STOPWORDS.has(w))
    .map(([w]) => w);
  let bits = 1 << 12;
  while (bits < common.length * 32 && bits < (1 << 18)) bits <<= 1;
  const bloom = new Uint8Array(bits / 8);
  for (const w of common) {
    for (const seed of BLOOM_SEEDS) {
      const bit = fnv1a(w, seed, bits);
      bloom[bit >> 3] |= 1 << (bit & 7);
    }
  }
  return { bloom, bits, dfCap, commonWords: common.length };
}

function fnv1a(str, seed, bits) {
  let h = 0x811c9dc5 ^ seed;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) % bits;
}

/**
 * Build a vocabulary bloom over a LAYER's records with a base's geometry —
 * LAYERED_PROFILE.md 4.5, 6.1 step 4.
 *
 * A layer's bloom must use the base's `bits`, hashes and `minCount` so the
 * chain's union is an exact bitwise OR rather than an approximation. Exported
 * (rather than reimplemented in the append path) so the two can never drift:
 * the same fnv1a, the same seeds, the same tokenizer.
 */
export function buildLayerVocabBloom(chunks, { bits, minCount }) {
  if (!Number.isInteger(bits) || bits <= 0 || bits % 8 !== 0) {
    throw new Error(`buildLayerVocabBloom: bits must be a positive multiple of 8, got ${bits}`);
  }
  const counts = new Map();
  for (const chunk of chunks) {
    for (const w of tokenize(chunk.text)) counts.set(w, (counts.get(w) || 0) + 1);
  }
  const floor = Number.isInteger(minCount) && minCount > 0 ? minCount : 1;
  const bloom = new Uint8Array(bits / 8);
  let kept = 0;
  for (const [w, c] of counts) {
    if (c < floor) continue;
    kept += 1;
    for (const seed of BLOOM_SEEDS) {
      const bit = fnv1a(w, seed, bits);
      bloom[bit >> 3] |= 1 << (bit & 7);
    }
  }
  return { bloom, bits, minCount: floor, keptWords: kept, uniqueWords: counts.size };
}

/** The hash names a layer's bloom declares (4.5), matching BLOOM_SEEDS. */
export const BLOOM_HASH_NAMES = Object.freeze(['fnv1a:0', 'fnv1a:0x9e3779b9']);

function buildVocabBloom(chunks) {
  const counts = new Map();
  for (const chunk of chunks) {
    for (const w of tokenize(chunk.text)) counts.set(w, (counts.get(w) || 0) + 1);
  }
  // Small corpora cannot afford the wiki pack's appears-3-times floor: with a
  // handful of documents most content words appear once or twice and the
  // known-token signal would read every real query as out-of-vocabulary.
  const minCount = chunks.length >= 200 ? 3 : 1;
  const kept = [...counts.entries()].filter(([, c]) => c >= minCount).map(([w]) => w);
  let bits = 1 << 14;
  while (bits < kept.length * 32 && bits < (1 << 21)) bits <<= 1;
  const bloom = new Uint8Array(bits / 8);
  for (const w of kept) {
    for (const seed of BLOOM_SEEDS) {
      const bit = fnv1a(w, seed, bits);
      bloom[bit >> 3] |= 1 << (bit & 7);
    }
  }
  return { bloom, bits, minCount, keptWords: kept.length, uniqueWords: counts.size };
}

// Proper-noun detection without a tagger: a word capitalized somewhere but
// never lowercase anywhere in the corpus is very likely a name. It misses
// single-mention names and lets roman numerals and month names through,
// which only yields an odd but still retrieval-verified negative.
const ENTITY_MIN_LEN = 3;
const ROMAN_NUMERAL = /^[ivxlcdm]+$/;
function buildEntityIndex(chunks) {
  const capitalized = new Map(); // lowercase -> original-case display form
  const lowercaseSeen = new Set();
  const perChunkCapWords = chunks.map((chunk) => {
    const words = String(chunk.text || '').match(/[A-Za-z']+/g) || [];
    const caps = [];
    for (const w of words) {
      if (w.length < ENTITY_MIN_LEN) continue;
      const lower = w.toLowerCase();
      if (/^[A-Z]/.test(w)) {
        caps.push(lower);
        if (!capitalized.has(lower)) capitalized.set(lower, w);
      } else lowercaseSeen.add(lower);
    }
    return caps;
  });
  const df = new Map();
  perChunkCapWords.forEach((caps) => {
    for (const w of new Set(caps)) {
      if (lowercaseSeen.has(w) || STOPWORDS.has(w) || ROMAN_NUMERAL.test(w)) continue;
      df.set(w, (df.get(w) || 0) + 1);
    }
  });
  // Skip names that appear in most chunks (a protagonist): swapping them in
  // is not a real mismatch.
  const maxDf = Math.max(2, Math.ceil(chunks.length * 0.4));
  const entitySet = new Set([...df.entries()].filter(([, c]) => c >= 1 && c <= maxDf).map(([w]) => w));
  return {
    entities: [...entitySet],
    display: (w) => capitalized.get(w) || w,
    // Which of this corpus's entities occur in a text: tokenize + set lookup,
    // not a regex per entity (too slow with hundreds of names).
    presentIn: (text) => tokenize(text).filter((w) => entitySet.has(w)),
  };
}

function knownFrac(text, bloom, bits) {
  const words = tokenize(text);
  if (!words.length) return 0;
  let known = 0;
  for (const w of words) {
    const hit = BLOOM_SEEDS.every((seed) => {
      const bit = fnv1a(w, seed, bits);
      return (bloom[bit >> 3] >> (bit & 7)) & 1;
    });
    if (hit) known++;
  }
  return known / words.length;
}

function syntheticGibberish(n, seed, bloom, bits) {
  const next = rng(seed);
  const alphabet = 'abcdefghijklmnopqrstuvwxyz';
  const queries = [];
  const used = new Set();
  function token() {
    const len = 5 + Math.floor(next() * 6);
    let s = '';
    for (let i = 0; i < len; i++) s += alphabet[Math.floor(next() * alphabet.length)];
    return s;
  }
  while (queries.length < n && used.size < n * 100) {
    const words = [];
    const wordCount = 3 + Math.floor(next() * 4);
    for (let i = 0; i < wordCount; i++) words.push(token());
    const text = words.join(' ');
    used.add(text);
    if (knownFrac(text, bloom, bits) === 0) queries.push(text);
  }
  return queries;
}


function aucFor(posRows, negRows) {
  if (!posRows.length || !negRows.length) return null;
  let score = 0;
  for (const a of posRows) for (const c of negRows) score += a.p > c.p ? 1 : a.p === c.p ? 0.5 : 0;
  return score / (posRows.length * negRows.length);
}

// Human-written calibration queries: JSON lines of {text, expectId} or
// {text, expectTitle}, from runtime.calibrationQueries in pikelet.config.json
// (resolved relative to the project directory, like encoder.calibrationPath).
// Optional; they are held out of the fit as validation.
export async function loadCalibrationQueries(config, projectDir, log) {
  const rel = config?.runtime?.calibrationQueries;
  if (!rel || !projectDir) return [];
  const file = path.resolve(projectDir, rel);
  let text;
  try {
    text = await fs.readFile(file, 'utf8');
  } catch (err) {
    // The path is set only when someone asked for it, so a missing file is an
    // error, not a corpus without human questions.
    if (err.code === 'ENOENT') throw new Error(`calibration queries file not found: ${file}`);
    throw err;
  }
  const out = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let row;
    try { row = JSON.parse(trimmed); } catch {
      log(`Calibration: skipping malformed line in ${rel}: ${trimmed.slice(0, 80)}`);
      continue;
    }
    if (typeof row?.text !== 'string' || !row.text.trim()) continue;
    if (row.expectId === undefined && typeof row.expectTitle !== 'string') continue;
    out.push({ text: row.text, sourceId: row.expectId, sourceTitle: row.expectTitle });
  }
  return out;
}

// Why a human-written question names no record, or null when it names one.
// Such a question can never verify, so it is reported by name instead of
// being counted with the unretrievable ones. The usual cause is a title
// written with the Markdown formatting ingest strips ("How a `.pikelet`
// query runs" for the record "How a .pikelet query runs").
export function calibrationQuestionUnmatched({ sourceTitle, sourceId }, chunks) {
  if (sourceId !== undefined) {
    return Number.isSafeInteger(sourceId) && sourceId >= 0 && sourceId < chunks.length
      ? null
      : `expectId ${JSON.stringify(sourceId)} is not a record id (0-${chunks.length - 1})`;
  }
  if (chunks.some((c) => (c.title || '').trim() === sourceTitle)) return null;
  const plain = String(sourceTitle).replace(/[`*_]/g, '').trim();
  const near = plain !== sourceTitle && chunks.some((c) => (c.title || '').trim() === plain);
  return `no record is titled ${JSON.stringify(sourceTitle)}`
    + (near ? `; did you mean ${JSON.stringify(plain)}? Titles are matched after Markdown formatting is stripped` : '');
}

// evidence(text, { excludeIds }) is the artifact reader's own retrieval
// (complete/index.mjs __calibrationEvidence over the uncalibrated build):
// { vector, fused }, the windows query() scores. Every probe below is
// retrieved through it, so the fit sees the evidence the reader will score
// at serve time — not a second retrieval implementation's approximation.
// embedQuery is still needed, for the word vectors behind substitution.
export async function calibrateRetrievalAbstention({ chunks, config, embedQuery, embedWordVecs = null, evidence, log = () => {}, projectDir = null }) {
  const skip = (reason) => {
    log(`Abstention calibration skipped: ${reason}; the artifact will report match_quality "unscored"`);
    return null;
  };
  const titles = [...new Set(chunks.map((c) => (c.title || '').trim()).filter(Boolean))];
  if (titles.length < 2) return skip('needs at least 2 distinct chunk titles for verified positives');

  // The bloom covers the FULL corpus — it ships with the artifact, and the
  // held-out documents are present at serve time. Held-out questions scoring
  // known_frac high against it is the point: that is the profile of a real
  // in-domain unanswerable query.
  const { bloom, bits, minCount, keptWords, uniqueWords } = buildVocabBloom(chunks);
  const commonWords = buildCommonWordsBloom(chunks);
  const isCommon = (w) => BLOOM_SEEDS.every((seed) => {
    const bit = fnv1a(w, seed, commonWords.bits);
    return (commonWords.bloom[bit >> 3] >> (bit & 7)) & 1;
  });
  // Single-word vector cache for maxSimFrac: corpus vocabulary recurs across
  // probes.
  const wordVecCache = new Map();
  const embedWordVec = embedWordVecs ? async (w) => {
    if (!wordVecCache.has(w)) {
      const vecs = await embedWordVecs(w);
      wordVecCache.set(w, vecs.get(w) || new Float32Array(config.embedding.dims));
    }
    return wordVecCache.get(w);
  } : null;
  // maxSim1's chance-similarity floor for this corpus and encoder.
  const maxSimVocab = embedWordVec
    ? [...new Set(chunks.flatMap((c) => tokenize(c.text || '').filter((w) => w.length >= COVERAGE_MIN_WORD_LEN && !COVERAGE_STOPWORDS.has(w))))]
    : [];
  const maxSimBaseline = embedWordVec ? await estimateMaxSimBaseline(maxSimVocab, embedWordVec, SEED ^ 0xba5e11e) : 0;

  // Nothing is held out at the corpus level; the held-out-document class
  // builds its own exclusions per query.
  const retainedPos = chunks.map((_, pos) => pos);
  const retainedSet = new Set(retainedPos);
  const K = Math.min(10, retainedPos.length);
  // A chunk's text starts with its heading (ingest.mjs echoes it). Split it
  // off so coverageFrac can give a heading match partial credit
  // (HEADING_COVERAGE_WEIGHT) rather than full or none.
  const splitHeadingBody = (text) => {
    const nl = text.indexOf('\n');
    return nl === -1 ? { heading: text, body: '' } : { heading: text.slice(0, nl), body: text.slice(nl + 1) };
  };
  // Signals are computed from the reader's own evidence the way its scorer
  // computes them (complete/retrieval-abstention.mjs): d0, the rank-4
  // margin and the mean over the vector window, coverage over the top fused
  // passages. An ablation's evidence already excludes the ablated records
  // from both the vector and the BM25 side.
  const signalsFor = async (text, ev) => {
    const top = ev.vector.slice(0, K);
    const d0 = top.length ? top[0].distance : 1;
    const margin = top.length > 1 ? top[Math.min(4, top.length - 1)].distance - d0 : 0;
    const mean10 = top.length ? top.reduce((s, r) => s + r.distance, 0) / top.length : 1;
    const passages = ev.fused.slice(0, COVERAGE_TOP_PASSAGES).map((h) => splitHeadingBody(chunks[h.id]?.text || ''));
    const coverage1 = coverageFrac(text, passages, isCommon);
    // null when no embedWordVecs was provided (e.g. a kind-2 external-encoder
    // build, where per-word embedding would mean a host call per word).
    const maxSim1 = embedWordVec ? await maxSimFrac(text, passages, isCommon, embedWordVec, maxSimBaseline) : null;
    return {
      d0,
      margin,
      mean10,
      known_frac: knownFrac(text, bloom, bits),
      // GROUNDING_FEAT is coverage1; maxSim1 is reported for comparison only.
      [COVERAGE_FEAT]: coverage1,
      [MAXSIM_FEAT]: maxSim1,
    };
  };

  const search = (text) => evidence(text);
  // Ablation search: the same retrieval minus the positive's answer.
  // excludeSet is every chunk the label depends on: the sampled chunk for a
  // content-word positive, every chunk of the document for a title positive
  // (sibling chunks would still answer it).
  const searchExcluding = (text, excludeSet) => evidence(text, { excludeIds: excludeSet });
  const ablationTargets = ({ sourceId, sourceTitle }) => new Set(sourceId !== undefined
    ? [sourceId]
    : retainedPos.filter((p) => (chunks[p].title || '').trim() === sourceTitle));

  const rows = [];
  let droppedPositives = 0;
  const verify = (hits, sourceTitle, sourceId) => {
    // A positive counts only when retrieval lands on its source within the base
    // K window of the vector order (any chunk of the document for title
    // queries, the sampled chunk for content-word queries).
    const topK = hits.vector.slice(0, K);
    return sourceId !== undefined
      ? topK.some((h) => h.id === sourceId)
      : topK.some((h) => (chunks[h.id]?.title || '').trim() === sourceTitle);
  };
  // Positives: title/content-word templates, verified by retrieval. They copy
  // the target passage's words, so coverage1 is ~1.0 across the class; the
  // substitution class below supplies answerable queries with partial
  // overlap.
  const baseTemplates = [
    ...titleQuestions(titles, Math.ceil(MAX_POSITIVES / 2), SEED ^ 0x51f15e),
    ...contentWordQuestions(chunks, Math.floor(MAX_POSITIVES / 2), SEED ^ 0xc0ffee, retainedPos),
  ];
  const basePositives = [];
  for (const { text, sourceTitle, sourceId } of baseTemplates) {
    const hits = await search(text);
    if (verify(hits, sourceTitle, sourceId)) {
      const row = { text, label: 1, sourceTitle, sourceId, ...(await signalsFor(text, hits)) };
      rows.push(row);
      basePositives.push(row);
    } else droppedPositives++;
  }
  if (basePositives.length < MIN_VERIFIED_POSITIVES) {
    return skip(`only ${basePositives.length} of ${baseTemplates.length} templated queries verified by retrieval (need ${MIN_VERIFIED_POSITIVES})`);
  }
  const basePositiveD0 = basePositives.map((r) => r.d0).sort((a, b) => a - b);
  const positiveMedianD0 = basePositiveD0[Math.floor(basePositiveD0.length / 2)];
  // Encoder-guided substitution: swap some content words for nearby corpus
  // vocabulary the source passage does not contain, then re-verify by
  // retrieval; a swap that no longer retrieves its source is dropped, not
  // mislabeled. See SUBSTITUTION_* for the vocabulary cap and coherence gate.
  const substWords = (t) => (String(t).toLowerCase().match(/[a-z0-9']+/g) || []);
  const isSubstContent = (w) => w.length >= 4 && !STOPWORDS.has(w) && !/^\d+$/.test(w);
  const substDf = new Map();
  for (const c of chunks) for (const w of new Set(substWords(c.text).filter(isSubstContent))) substDf.set(w, (substDf.get(w) || 0) + 1);
  const substVocab = [...substDf.entries()].sort((a, b) => b[1] - a[1]).slice(0, SUBSTITUTION_VOCAB_TOP).map(([w]) => w);
  const substWordVec = new Map();
  const substVecOf = async (w) => {
    if (!substWordVec.has(w)) {
      const v = Float32Array.from(await embedQuery(w));
      let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1;
      substWordVec.set(w, v.map((x) => x / n));
    }
    return substWordVec.get(w);
  };
  for (const w of substVocab) await substVecOf(w);
  const substCos = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d; };
  const substPassageWords = ({ sourceId, sourceTitle }) => {
    const ids = sourceId !== undefined ? [sourceId]
      : retainedPos.filter((p) => (chunks[p].title || '').trim() === sourceTitle);
    return new Set(ids.flatMap((id) => substWords(chunks[id]?.text || '')));
  };
  const substRng = rng(SEED ^ 0x5ab571);
  let substitutedDropped = 0, substitutedKept = 0;
  for (const pos of sample(basePositives, Math.min(SUBSTITUTION_BUDGET, basePositives.length), SEED ^ 0x5ab57)) {
    const inPassage = substPassageWords(pos);
    const toks = substWords(pos.text);
    const out = [...toks];
    let swapped = 0;
    for (let i = 0; i < toks.length; i++) {
      const w = toks[i];
      if (!isSubstContent(w) || substRng() < 0.5) continue;
      const wv = await substVecOf(w);
      let best = null, bestCos = SUBSTITUTION_MIN_COS;
      for (const v of substVocab) {
        if (v === w || inPassage.has(v) || inPassage.has(v.replace(/s$/, '')) || v === `${w}s` || w === `${v}s`) continue;
        const c = substCos(wv, substWordVec.get(v));
        if (c > bestCos) { bestCos = c; best = v; }
      }
      if (best) { out[i] = best; swapped++; }
    }
    if (!swapped) { substitutedDropped++; continue; }
    const text = out.join(' ');
    const hits = await search(text);
    if (verify(hits, pos.sourceTitle, pos.sourceId)) {
      rows.push({ text, label: 1, sourceTitle: pos.sourceTitle, sourceId: pos.sourceId, genKind: 'substituted', ...(await signalsFor(text, hits)) });
      substitutedKept++;
    } else substitutedDropped++;
  }
  const positives = rows.filter((r) => r.label === 1);
  if (positives.length < MIN_VERIFIED_POSITIVES) {
    return skip(`only ${positives.length} verified positives (need ${MIN_VERIFIED_POSITIVES})`);
  }

  // Human-written queries are held out of the fit entirely and scored after
  // it as the headline validation: they are the only rows the generator did
  // not make.
  const humanTemplates = await loadCalibrationQueries(config, projectDir, log);
  const humanRows = [];
  let humanDropped = 0;
  let humanUnmatched = 0;
  for (const { text, sourceTitle, sourceId } of humanTemplates) {
    const unmatched = calibrationQuestionUnmatched({ sourceTitle, sourceId }, chunks);
    if (unmatched) {
      humanUnmatched++;
      log(`Calibration: skipping question ${JSON.stringify(text.slice(0, 80))}: ${unmatched}`);
      continue;
    }
    const hits = await search(text);
    if (verify(hits, sourceTitle, sourceId)) {
      humanRows.push({ text, label: 1, sourceTitle, sourceId, ...(await signalsFor(text, hits)) });
    } else humanDropped++;
  }

  let droppedForeign = 0;
  for (const { text } of titleQuestions(FOREIGN_TITLE_BANK, FOREIGN_TITLE_BANK.length, SEED ^ 0xf03e16)) {
    const sig = await signalsFor(text, await search(text));
    // An off-domain query that retrieves as strongly as a median positive,
    // or whose best passages share most of its words (see
    // FOREIGN_OVERLAP_COVERAGE), overlaps the corpus domain; its label is
    // untrustworthy, so it stays out of the fit — but it is scored and
    // reported (eval-only) rather than discarded, so the summary states
    // how the model treats it.
    if (sig.d0 <= positiveMedianD0 || sig[GROUNDING_FEAT] >= FOREIGN_OVERLAP_COVERAGE) {
      droppedForeign++;
      rows.push({ text, label: 0, evalOnly: true, negativeKind: 'foreign-overlap', ...sig });
    } else rows.push({ text, label: 0, negClass: 'easy', negativeKind: 'foreign-bank', ...sig });
  }
  for (const text of syntheticGibberish(GIBBERISH_QUERIES, SEED ^ 0x9166e11, bloom, bits)) {
    rows.push({ text, label: 0, negClass: 'easy', negativeKind: 'synthetic-gibberish', ...(await signalsFor(text, await search(text))) });
  }

  // Ablation negatives (diagnostic since v6): a verified positive's exact
  // text, scored with its source excluded, so only what retrieval returns
  // differs from the paired positive.
  //
  // Dropped as "still grounded" when distance stays close to the paired
  // positive's own d0, or coverage stays above an absolute floor: either
  // means another chunk still answers it (a duplicate or cross-reference).
  // The relative-coverage check applies only when the positive itself was
  // well grounded; otherwise it is nearly always true.
  const ABLATION_RELATIVE_COVERAGE_FLOOR = 0.5;
  let ablationDropped = 0;
  for (const positiveRow of positives) {
    const { text } = positiveRow;
    const targets = ablationTargets(positiveRow);
    // signalsFor handles a window smaller than K, so only require that the
    // ablation leaves something to search; requiring K would skip every
    // positive on corpora where K was capped to the chunk count.
    if (targets.size === 0 || retainedSet.size - targets.size < 1) continue;
    const hits = await searchExcluding(text, targets);
    const sig = await signalsFor(text, hits);
    const stillClose = sig.d0 <= positiveRow.d0 + 0.05;
    const stillGrounded = sig[GROUNDING_FEAT] >= 0.75
      || (positiveRow[GROUNDING_FEAT] >= ABLATION_RELATIVE_COVERAGE_FLOOR
        && sig[GROUNDING_FEAT] >= positiveRow[GROUNDING_FEAT] * 0.75);
    if (stillClose || stillGrounded) ablationDropped++;
    else rows.push({ text, label: 0, negClass: 'hard', negativeKind: 'ablation', sourceId: positiveRow.sourceId, ...sig });
  }

  // Entity-swap negatives: ablation fails on thematically uniform corpora
  // (a novel's chapters answer each other's questions). Swap a positive's
  // proper noun for one that never co-occurs with it, and keep the query only
  // if it no longer retrieves the source.
  const entityIndex = buildEntityIndex(chunks);
  let entitySwapDropped = 0;
  if (entityIndex.entities.length >= 2) {
    const entityRng = rng(SEED ^ 0xe57171);
    // Entities in the source chunk(s), excluded as replacements; indexed from
    // the positive's own chunks rather than every entity's chunk list.
    for (const positiveRow of positives) {
      const { text, sourceId, sourceTitle } = positiveRow;
      const sourceChunks = ablationTargets(positiveRow);
      const present = entityIndex.presentIn(text);
      if (!present.length) continue;
      const from = present[Math.floor(entityRng() * present.length)];
      const inSourceChunks = new Set();
      for (const id of sourceChunks) for (const w of entityIndex.presentIn(chunks[id]?.text || '')) inSourceChunks.add(w);
      // Prefer a replacement that never appears in the same chunk(s) as
      // the source — a swap between two entities that already coexist
      // there could still describe something the passage actually says.
      const candidates = entityIndex.entities.filter((w) => w !== from && !inSourceChunks.has(w));
      const pool = candidates.length ? candidates : entityIndex.entities.filter((w) => w !== from);
      if (!pool.length) continue;
      const to = pool[Math.floor(entityRng() * pool.length)];
      const swapped = text.replace(new RegExp(`\\b${from}\\b`, 'i'), entityIndex.display(to));
      if (swapped === text) continue;
      const hits = await search(swapped);
      if (verify(hits, sourceTitle, sourceId)) { entitySwapDropped++; continue; }
      rows.push({
        text: swapped, label: 0, negClass: 'hard', negativeKind: 'entity-swap', sourceId, ...(await signalsFor(swapped, hits)),
      });
    }
  }

  // Held-out-document negatives: title-template questions about sampled
  // documents, scored with the whole document excluded, kept when the best
  // remaining hit is farther than the median positive d0. They cover the
  // "entity named, fact absent" case that ablation's drop rule discards on
  // entity-dense corpora.
  const heldOut = chooseHeldOutTitles(chunks, titles, SEED ^ 0x8e1d00);
  let heldOutDroppedCovered = 0;
  if (heldOut.titles.size) {
    const chunksOfTitle = new Map();
    chunks.forEach((c, pos) => {
      const t = (c.title || '').trim();
      if (!chunksOfTitle.has(t)) chunksOfTitle.set(t, new Set());
      chunksOfTitle.get(t).add(pos);
    });
    for (const { text, sourceTitle } of titleQuestions([...heldOut.titles], HELDOUT_QUERIES, SEED ^ 0xab5e27)) {
      const exclude = chunksOfTitle.get(sourceTitle) ?? new Set();
      if (retainedSet.size - exclude.size < 1) continue;
      const hits = await searchExcluding(text, exclude);
      const sig = await signalsFor(text, hits);
      if (sig.d0 <= positiveMedianD0) heldOutDroppedCovered++;
      else rows.push({ text, label: 0, negClass: 'hard', negativeKind: 'held-out-doc', sourceTitle, ...sig });
    }
  }

  const negatives = rows.filter((r) => r.label === 0 && !r.evalOnly);
  // v6: the fit's negatives are the off-topic ones; hard negatives are
  // diagnostic only.
  const offTopicNegatives = negatives.filter((r) => r.negClass === 'easy');
  const hardNegatives = negatives.filter((r) => r.negClass === 'hard');
  if (offTopicNegatives.length < MIN_NEGATIVES) {
    return skip(`only ${offTopicNegatives.length} off-topic negatives survived the overlap drop (need ${MIN_NEGATIVES})`);
  }

  // Weak band: retained-title questions whose source lands at rank 5..K on
  // corpora deep enough to have one. They keep the weak threshold honest so
  // adjacent content is shown with a caveat instead of hidden. search()
  // returns the reader's whole rerank window; rank must still be checked
  // against the base K window — a source landing at, say, rank 40 is well
  // outside "adjacent content", not a weak match.
  if (chunks.length >= 25) {
    const weakTemplates = titleQuestions(titles, MAX_POSITIVES, SEED ^ 0x0ddba11);
    for (const { text, sourceTitle } of weakTemplates) {
      if (rows.filter((r) => r.label === -1).length >= 24) break;
      const hits = await search(text);
      const rank = hits.vector.findIndex((h) => (chunks[h.id]?.title || '').trim() === sourceTitle) + 1;
      if (rank >= 5 && rank <= K) rows.push({ text, label: -1, ...(await signalsFor(text, hits)) });
    }
  }

  // Fit: logistic regression on standardized signals by gradient descent
  // (the wiki calibrator's optimizer, so the assets stay comparable):
  // positives vs. off-topic negatives, class-balanced. Hard negatives are
  // scored, never fit.
  const genericScorerFor = (feats, fitRows, weightOf = () => 1) => {
    const weights = fitRows.map(weightOf);
    const weightSum = weights.reduce((a, c) => a + c, 0);
    const mean = {}, std = {};
    for (const f of feats) {
      mean[f] = fitRows.reduce((sum, r, i) => sum + r[f] * weights[i], 0) / weightSum;
      std[f] = Math.sqrt(fitRows.reduce((sum, r, i) => sum + ((r[f] - mean[f]) ** 2) * weights[i], 0) / weightSum) || 1;
    }
    const xs = fitRows.map((r) => feats.map((f) => (r[f] - mean[f]) / std[f]));
    const ys = fitRows.map((r) => r.label);
    let w = feats.map(() => 0);
    let b = 0;
    for (let epoch = 0; epoch < 4000; epoch++) {
      const gw = feats.map(() => 0);
      let gb = 0;
      for (let i = 0; i < xs.length; i++) {
        const z = xs[i].reduce((s, v, j) => s + v * w[j], b);
        const p = 1 / (1 + Math.exp(-z));
        const err = (p - ys[i]) * weights[i];
        xs[i].forEach((v, j) => { gw[j] += err * v; });
        gb += err;
      }
      w = w.map((wj, j) => wj - 0.1 * (gw[j] / weightSum + 1e-3 * wj));
      b -= 0.1 * (gb / weightSum);
    }
    return { mean, std, w, b, prob: (r) => 1 / (1 + Math.exp(-(feats.reduce((s, f, j) => s + ((r[f] - mean[f]) / std[f]) * w[j], b)))) };
  };
  // Class-balanced: negative rows are reweighted to the total positive
  // weight. Human-written queries are held out (see humanRows).
  const fitPositives = positives;
  const fitNegatives = offTopicNegatives;
  const balanceFactor = fitPositives.length > 0 && fitNegatives.length > 0
    ? fitPositives.length / fitNegatives.length : 1;
  const fit = [...fitPositives, ...fitNegatives];
  const fitWeightOf = (r) => (r.label === 0 ? balanceFactor : 1);
  const scorerFor = (fitRows) => genericScorerFor(FEATS, fitRows, fitWeightOf);
  const model = scorerFor(fit);
  const { mean, std, w, b } = model;
  for (const r of rows) r.p = model.prob(r);

  // The fit AUC is computed on the same rows the regression was fit on —
  // in-sample, and reported as such. The gate uses a deterministic 5-fold
  // cross-validation instead: refit on 4/5 of the rows, score the held-out
  // fold, pool the held-out probabilities into one AUC. The embeds and
  // searches are already done, so the extra fits cost milliseconds.
  const fitAuc = aucFor(fitPositives, fitNegatives);
  const shuffled = sample(fit, fit.length, SEED ^ 0xcf01d);
  const FOLDS = 5;
  const heldout = [];
  for (let fold = 0; fold < FOLDS; fold++) {
    const test = shuffled.filter((_, i) => i % FOLDS === fold);
    const train = shuffled.filter((_, i) => i % FOLDS !== fold);
    if (!train.some((r) => r.label === 1) || !train.some((r) => r.label === 0)) continue;
    const foldModel = scorerFor(train);
    for (const r of test) heldout.push({ label: r.label, negClass: r.negClass, negativeKind: r.negativeKind, genKind: r.genKind, p: foldModel.prob(r) });
  }
  const heldoutPos = heldout.filter((r) => r.label === 1);
  const cvAuc = aucFor(heldoutPos, heldout.filter((r) => r.label === 0));
  // cvAucHard: positives vs. hard negatives, a diagnostic, not a gate. Hard
  // negatives are never fit, so their probabilities are already held out;
  // positives use their cross-validated ones. Expected well below cvAuc.
  const cvAucHard = aucFor(heldoutPos, hardNegatives.map((r) => ({ p: model.prob(r) })));
  // Per-generator-class held-out AUC: the base template class separates
  // easily and can hide a poorly scoring substituted class.
  const cvAucBase = aucFor(heldoutPos.filter((r) => r.genKind === undefined), heldout.filter((r) => r.label === 0));
  const cvAucSubstituted = aucFor(heldoutPos.filter((r) => r.genKind === 'substituted'), heldout.filter((r) => r.label === 0));
  const gateAuc = cvAuc ?? fitAuc;
  if (gateAuc === null || gateAuc < MIN_AUC) {
    return skip(`${cvAuc === null ? 'fit' : 'cross-validated'} AUC ${gateAuc === null ? 'n/a' : gateAuc.toFixed(3)} separating on-topic from off-topic queries < ${MIN_AUC}`);
  }
  // A positive d0 weight (worse distance = more relevant) means the negative
  // pool's retrieval barely differs from the positives' and the fit is
  // scoring on coverage alone. The AUC gates do not catch it.
  const d0Weight = w[FEATS.indexOf('d0')];
  if (!(d0Weight <= 0)) {
    return skip(`fitted d0 weight ${d0Weight.toFixed(3)} is not negative: distance carries no signal in this fit `
      + '(the negatives are indistinguishable from positives by retrieval), so the verdict would reduce to a coverage threshold');
  }

  // Hard threshold from the negative side: the OFFTOPIC_HARD_QUANTILE
  // percentile of off-topic probability, not a fraction of the positive floor
  // (near-1 positive scores would push the bar too high).
  const quantile = (values, q) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  };
  const pos = positives.map((r) => r.p);
  const posFloor = quantile(pos, 0.05);
  const hardCeil = quantile(fitNegatives.map((r) => r.p), OFFTOPIC_HARD_QUANTILE);
  // Weak rows scoring under 5% relevant are negatives in all but name;
  // they must not shape the weak threshold.
  const allWeakP = rows.filter((r) => r.label === -1).map((r) => r.p);
  const weakP = allWeakP.filter((p) => p > Math.max(hardCeil, 0.05));
  const hardOverlap = hardCeil >= posFloor;
  const hardRaw = hardCeil;
  // Saturation rail (HARD_CEILING). Log when it engages: real phrasings unlike
  // the probes are what a too-high bar hides.
  const hard = Math.min(hardRaw, HARD_CEILING);
  if (hardRaw > HARD_CEILING) {
    log(`Calibration: hard threshold computed at ${hardRaw.toFixed(3)} (saturated fit); clamped to ${HARD_CEILING}. `
      + 'Probe classes separate almost perfectly — treat abstention boundaries on this corpus with caution.');
  }
  const weakCeil = weakP.length ? quantile(weakP, 0.9) : 0;
  // "Strong" must clear the off-topic mass. If off-topic overlaps the weakest
  // positives, the hard threshold stays protective (a false "none" hides
  // results) and the weak threshold rises to the off-topic ceiling instead,
  // capped at the positive median.
  const weakRaw = Math.min(
    Math.max(
      weakCeil > hard && weakCeil < posFloor ? Math.sqrt(weakCeil * posFloor) : posFloor * 0.9,
      hardCeil,
      hard,
    ),
    Math.max(quantile(pos, 0.5), hard),
  );
  // Band floor: widening only demotes strong to weak (still shown, with a
  // caveat), so it is safe in the direction that matters.
  const weak = Math.min(0.95, Math.max(weakRaw, hard + MIN_WEAK_BAND));

  // Validate against something the generator did not make: held-out
  // human-written queries. There is no synthetic fallback; corpora without
  // runtime.calibrationQueries are validated by the cross-validated AUC gate
  // only.
  for (const r of humanRows) r.p = model.prob(r);
  const realQueryAbstained = humanRows.filter((r) => r.p < hard).length;
  const realQueryAuc = humanRows.length
    ? aucFor(humanRows, fitNegatives) : null;
  if (humanRows.length >= MIN_PARAPHRASE_STRESS_SAMPLE
    && realQueryAbstained / humanRows.length > MAX_PARAPHRASE_ABSTENTION_RATE) {
    return skip(`${realQueryAbstained}/${humanRows.length} human-written calibration queries (runtime.calibrationQueries) `
      + `scored below the hard threshold (> ${(MAX_PARAPHRASE_ABSTENTION_RATE * 100).toFixed(0)}%): the fit abstains `
      + 'on too many of the corpus author\'s own queries');
  }

  // Eval-only rows (foreign-bank queries that overlap the corpus) are
  // scored by the final model and reported: how many the shipped
  // thresholds would answer is the asset's stated hard-negative exposure.
  const evalOnlyRows = rows.filter((r) => r.evalOnly);
  // Golden queries the artifact embeds for verify_pack: human-written
  // questions first (never displaced by the cap), then an even spread of
  // verified positives across both templates.
  const goldenSample = (() => {
    const cap = 24;
    const rest = Math.max(0, cap - humanRows.length);
    if (positives.length <= rest) return [...humanRows, ...positives];
    const step = positives.length / rest;
    return [...humanRows, ...Array.from({ length: rest }, (_, i) => positives[Math.floor(i * step)])];
  })();
  const goldenQueries = goldenSample.map((r) => ({
    text: r.text,
    ...(r.sourceId !== undefined ? { expectId: r.sourceId } : { expectTitle: r.sourceTitle }),
  }));
  // maxSim1 vs coverage1, scored like aucFor so the numbers compare. null
  // without embedWordVecs.
  const maxSimRows = hardNegatives.length && positives.length && positives[0][MAXSIM_FEAT] !== null
    ? { pos: positives.map((r) => ({ p: r[MAXSIM_FEAT] })), neg: hardNegatives.map((r) => ({ p: r[MAXSIM_FEAT] })) }
    : null;
  const coverageRows = hardNegatives.length
    ? { pos: positives.map((r) => ({ p: r[COVERAGE_FEAT] })), neg: hardNegatives.map((r) => ({ p: r[COVERAGE_FEAT] })) }
    : null;
  // Paraphrase robustness: substituted positives have the same answer but
  // less lexical overlap with their source. A paraphrase-robust feature
  // should dip less on them than on base positives.
  const basePos = positives.filter((r) => r.genKind === undefined);
  const substPos = positives.filter((r) => r.genKind === 'substituted');
  const paraphraseRows = (feat) => (basePos.length && substPos.length && basePos[0][feat] !== null
    ? {
      meanBase: +(basePos.reduce((s, r) => s + r[feat], 0) / basePos.length).toFixed(4),
      meanSubstituted: +(substPos.reduce((s, r) => s + r[feat], 0) / substPos.length).toFixed(4),
      separationAuc: aucFor(basePos.map((r) => ({ p: r[feat] })), substPos.map((r) => ({ p: r[feat] }))),
    }
    : null);
  const maxSimParaphrase = paraphraseRows(MAXSIM_FEAT);
  const coverageParaphrase = paraphraseRows(COVERAGE_FEAT);
  const maxSimVsCoverage = {
    // Chance similarity between unrelated corpus words; maxSim1 is already
    // rescaled against it.
    maxSimBaseline: +maxSimBaseline.toFixed(4),
    maxSimSeparationAuc: maxSimRows ? aucFor(maxSimRows.pos, maxSimRows.neg) : null,
    coverageSeparationAuc: coverageRows ? aucFor(coverageRows.pos, coverageRows.neg) : null,
    meanMaxSimPositive: maxSimRows ? +(maxSimRows.pos.reduce((s, r) => s + r.p, 0) / maxSimRows.pos.length).toFixed(4) : null,
    meanMaxSimHardNegative: maxSimRows ? +(maxSimRows.neg.reduce((s, r) => s + r.p, 0) / maxSimRows.neg.length).toFixed(4) : null,
    meanCoveragePositive: coverageRows ? +(coverageRows.pos.reduce((s, r) => s + r.p, 0) / coverageRows.pos.length).toFixed(4) : null,
    meanCoverageHardNegative: coverageRows ? +(coverageRows.neg.reduce((s, r) => s + r.p, 0) / coverageRows.neg.length).toFixed(4) : null,
    // separationAuc here means "distinguishes base from substituted" —
    // for a paraphrase-robust feature, LOWER is better (0.5 = doesn't
    // notice a paraphrase happened at all; both are positives).
    paraphrase: { maxSim1: maxSimParaphrase, coverage1: coverageParaphrase },
  };
  const summary = {
    method: 'self-templates-v6',
    // What matchQuality measures: retrieval relevance, fit against
    // off-topic queries.
    target: 'relevance',
    seed: SEED,
    searchConfig: { k: K },
    verifiedPositiveQueries: positives.length,
    positivesByGenKind: {
      base: positives.filter((r) => r.genKind === undefined).length,
      substituted: positives.filter((r) => r.genKind === 'substituted').length,
    },
    substitutedDropped,
    positivesDroppedAsUnretrievable: droppedPositives,
    // Held-out human-written queries. null fields mean the corpus shipped none.
    humanCalibrationQueries: humanRows.length,
    humanQueriesDroppedAsUnretrievable: humanDropped,
    // Present only when some question named no record, so packs built
    // without such questions keep their bytes (and identities).
    ...(humanUnmatched ? { humanQueriesNamingNoRecord: humanUnmatched } : {}),
    realQueryAuc: realQueryAuc === null ? null : +realQueryAuc.toFixed(6),
    realQueryAbstentionRate: humanRows.length ? +(realQueryAbstained / humanRows.length).toFixed(6) : null,
    foreignNegativeQueries: negatives.filter((r) => r.negativeKind === 'foreign-bank').length,
    foreignKeptEvalOnlyAsSemanticOverlap: droppedForeign,
    foreignDropD0Threshold: +positiveMedianD0.toFixed(6),
    evalOnlyWouldAnswerAtHard: evalOnlyRows.filter((r) => r.p >= hard).length,
    syntheticGibberishQueries: negatives.filter((r) => r.negativeKind === 'synthetic-gibberish').length,
    // Hard negatives feed only the cvAucHard diagnostic.
    ablationNegativeQueries: negatives.filter((r) => r.negativeKind === 'ablation').length,
    ablationDroppedAsSuspectedAnswerable: ablationDropped,
    entitySwapNegativeQueries: negatives.filter((r) => r.negativeKind === 'entity-swap').length,
    entitySwapDroppedAsStillAnswerable: entitySwapDropped,
    corpusEntitiesDetected: entityIndex.entities.length,
    heldOutNegativeQueries: negatives.filter((r) => r.negativeKind === 'held-out-doc').length,
    heldOutDroppedAsCoveredElsewhere: heldOutDroppedCovered,
    heldOutTitles: heldOut.titles.size,
    weakQueries: weakP.length,
    weakDroppedAsIndistinguishableFromNegatives: allWeakP.length - weakP.length,
    // fitAuc (in-sample) and cvAuc (5-fold, gated) are positives vs. off-topic.
    // cvAucHard is positives vs. hard negatives, never fit or gated: how far
    // "strong" can be read as "answered". null when no hard negative exists.
    fitAuc: fitAuc === null ? null : +fitAuc.toFixed(6),
    cvAuc: cvAuc === null ? null : +cvAuc.toFixed(6),
    cvAucHard: cvAucHard === null ? null : +cvAucHard.toFixed(6),
    // Per-generator-class held-out AUC. null when too few held-out rows of a
    // class landed in any fold.
    cvAucByGenKind: {
      base: cvAucBase === null ? null : +cvAucBase.toFixed(6),
      substituted: cvAucSubstituted === null ? null : +cvAucSubstituted.toFixed(6),
    },
    hardThresholdOverlap: hardOverlap,
    hardThresholdClampedFrom: hardRaw > HARD_CEILING ? Number(hardRaw.toFixed(6)) : null,
    vocab: { uniqueWords, keptWords, minCount },
    coverage: { topPassages: COVERAGE_TOP_PASSAGES, commonWords: commonWords.commonWords, commonDfCap: commonWords.dfCap },
    // Comparison only: maxSim1 is not in the fit and no reader reads this
    // field.
    maxSimVsCoverage,
    maxSimInFit: false,
  };
  // features[]/weights[] carry only the topic features; coverage1 rides in
  // asset.coverage so an older reader scores the topic-only model instead of
  // feeding NaN for an unknown feature. useMaxSim is never set: readers
  // compute plain coverageFrac.
  const asset = {
    version: 1,
    corpus: config.name,
    searchConfig: { k: K },
    calibration: summary,
    features: BASE_FEATS,
    standardize: { mean, std },
    weights: w.slice(0, BASE_FEATS.length),
    bias: b,
    coverage: {
      weight: w[BASE_FEATS.length],
      mean: mean[GROUNDING_FEAT],
      std: std[GROUNDING_FEAT],
      minWordLen: COVERAGE_MIN_WORD_LEN,
      stopwords: [...COVERAGE_STOPWORDS],
      topK: COVERAGE_TOP_PASSAGES,
      commonWordWeight: COVERAGE_COMMON_WORD_WEIGHT,
      commonBloom: {
        bits: commonWords.bits,
        hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'],
        dfCap: commonWords.dfCap,
        base64: Buffer.from(commonWords.bloom).toString('base64'),
      },
    },
    thresholds: { hard: +hard.toFixed(6), weak: +weak.toFixed(6) },
    vocabBloom: { bits, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'], minCount },
  };
  return {
    calibrationJson: {
      kind: 'retrieval-signals-v1',
      asset,
      vocabBloomBase64: Buffer.from(bloom).toString('base64'),
    },
    summary,
    goldenQueries,
  };
}
