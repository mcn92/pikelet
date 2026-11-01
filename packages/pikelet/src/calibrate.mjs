import fs from 'node:fs/promises';
import path from 'node:path';

// Build-time abstention calibration for complete kind-3 artifacts. Mirrors
// the signal and scoring math of the wiki pack calibrator
// (examples/static-wiki-pack/calibrate_abstention.mjs) and of the reader's
// scorer (complete/retrieval-abstention.mjs) — retrieval signals (d0, margin,
// mean10) plus the corpus-vocabulary known-token fraction, standardized and
// passed through a fitted logistic model — but is corpus-generic.
//
// Design principle (v5): the model must never be able to tell positives
// from negatives by looking at the query text alone — only the retrieval
// response may carry the label. Every earlier version of this file broke
// that rule, in increasingly subtle ways, and each time the symptom was
// the same: aggregate AUC numbers looked fine while real questions still
// got wrongly abstained on. Traced (by inspecting real fitted weights and
// real query outcomes, not just summary metrics) to five compounding
// causes, all structural:
//   1. Every positive was assembled entirely from the target passage's
//      own words (a title, or a run of the chunk's own tokens), so
//      coverage1 was ~1.0 for every positive by construction — the fit
//      handed coverage1 the largest weight because it was the cleanest
//      separator available, not because high coverage is what makes a
//      query answerable.
//   2. Hard negatives (first cross-chunk recombination, later even the
//      ablation-negative attempts that only paired SOME of the positive
//      classes) landed at a different, generator-specific coverage band
//      than positives — recombination sat near 0.5 by how it combined two
//      chunks' words. The fit learned "coverage in this band =>
//      unanswerable", which is exactly the regime a real paraphrase of a
//      real answer lands in.
//   3. Positives and hard negatives came from different generator
//      functions (a "low-coverage positive" class built by swapping words
//      in an existing positive still inherited half its words verbatim;
//      a "held-out-document" negative used the same title-question
//      template as positives, but drawn from a different, excluded title
//      pool). Either way, query-side signals — not just the retrieval
//      response — could carry the label, so the fit could partially learn
//      "which generator produced this query" instead of answerability.
//   4. The one validation metric meant to catch this ("paraphrase-stress
//      abstention rate") was measured on positives already inside the
//      fit — training accuracy wearing a different name, incapable of
//      detecting exactly the failure it existed to catch.
//   5. Threshold placement, even once nominally anchored to the negative
//      side, was still built on a feature space and fit the four points
//      above had already contaminated.
//
// The fix addresses all five at once by collapsing to ONE generator
// family for both labels: positives are title/content-word templates,
// verified by retrieval (same as always). Every verified positive gets
// exactly one paired ablation negative — the identical query text, scored
// with the answer excluded from the index (searchExcluding). Because a
// paired row's positive and negative share byte-identical text, known_frac,
// query length, and lexical shape, the only thing that can differ between
// them is d0/margin/mean10/coverage1 as measured against what retrieval
// actually returns — which is the one thing that is supposed to carry the
// label. There is no second "low-coverage positive" class, no held-out-
// document negative class, and no synthetic "paraphrase-stress" proxy:
// human-written queries (runtime.calibrationQueries), held out of the fit
// entirely, are the only real validation this design can have, because
// they are the only rows not produced by the training generator itself.
//
// The fit is still split in two stages (point 3 in the design, distinct
// from point 3 above): stage 1 gates off-domain/gibberish negatives on
// known_frac + d0 alone — signals that already separate them almost
// trivially — so those rows never dilute stage 2's loss. Stage 2, the
// model this asset ships, is fit only on positives vs. paired ablation
// negatives, class-balanced. The hard threshold is set from the negative
// side (point 4): the 90th percentile of hard-negative probability, not a
// fraction of the positive floor — abstain when a score looks like where
// unanswerable queries actually live, not merely where it falls short of
// a suspiciously clean positive score. That placement is now honest,
// because the feature space it is placed on is no longer separating two
// different generators.
//
// v6 changes what the score means. Measured against real questions — 40
// hand-written ones on the docs pack (answerable, missing-detail,
// near-miss, false-premise) and the Veyra registry's labeled set — the v5
// answerability fit separated answerable from in-domain-unanswerable at
// AUC 0.61, and its hard threshold, placed among the hard negatives,
// withheld about half of the genuinely answerable questions. What the
// five retrieval signals can actually tell apart is whether a query is
// about something this corpus covers; whether the covering passage holds
// the specific fact is a reading task the signals cannot do, and the
// agent reading the passages can. So the shipped model is now fit on
// positives vs. off-topic queries (the foreign bank plus gibberish), its
// hard threshold sits at the 95th percentile of off-topic probability,
// and matchQuality means retrieval relevance: "none" = this pack does not
// cover the topic, "strong" = it does. The hard-negative classes below
// (ablation, entity-swap, held-out-doc) are still generated, but only to
// report how well the score separates in-domain-unanswerable questions
// (cvAucHard) — they no longer shape the fit, the threshold, or the gate.
// The v5 note above stays as the record of why the generators are built
// the way they are; its stage-2 and threshold paragraphs describe v5.
//
// Returns { calibrationJson, summary } on success, or null (with a logged
// reason) when the corpus cannot support a trustworthy fit — the caller then
// ships the unscored placeholder, which is strictly safer than a bad model.

const SEED = 424242;
// The four base features all measure topic similarity — distances plus
// membership in a corpus-wide bloom — so "the corpus discusses this area"
// and "this passage answers this question" are indistinguishable by them.
// coverage1 is the grounding feature that separates the two: the fraction
// of the query's content words present in the top retrieved passage's text
// (known_frac's question, asked of the passage instead of the corpus). It
// joins the logistic fit as a fifth feature but is serialized as a
// supplemental asset term (asset.coverage), not a features[] entry, so
// readers that predate it score the topic-only model instead of an unknown
// feature name (see complete/retrieval-abstention.mjs).
const BASE_FEATS = ['d0', 'margin', 'mean10', 'known_frac'];
const COVERAGE_FEAT = 'coverage1';
const MAXSIM_FEAT = 'maxSim1';
// grounding1 = max(coverage1, maxSim1) was tried twice and reverted both
// times:
//   1. Raw cosine similarity between encoder word vectors has no
//      meaningful zero (unrelated words still cluster somewhat), so
//      max()-ing the raw value in dragged every row's floor up toward
//      maxSim1's baseline and pushed the fitted threshold up sharply —
//      confirmed causing widespread false abstention in production.
//   2. Rescaling maxSim1 against a corpus-specific "how similar do two
//      unrelated words look by chance" baseline (estimateMaxSimBaseline)
//      fixed the floor-inflation, but even correctly rescaled and further
//      damped (a MAXSIM_GAMMA power curve, since a linear rescale alone
//      was still enough boundary noise to tip Pride and Prejudice's
//      held-out AUC across the MIN_AUC gate), the ACTUAL behavior on real
//      probe questions got worse, not better: more false abstention on
//      genuinely answerable questions, and more false-confidence
//      strong/weak verdicts on genuinely unsupported ones, than plain
//      coverage1 alone. Damping the blend numerically recovered the AUC
//      number but not the thing the AUC number is supposed to predict.
// grounding1 is coverage1 again. maxSim1 stays computed and reported
// (maxSimVsCoverage, comparison only) for whoever next attempts this —
// the standalone paraphrase-robustness numbers are real (see
// maxSimVsCoverage.paraphrase), but two attempts at folding it into the
// fit via max() have now made real-query behavior worse in practice, so
// it needs a different mechanism entirely, not another constant.
const GROUNDING_FEAT = COVERAGE_FEAT;
const FEATS = [...BASE_FEATS, GROUNDING_FEAT];
const COVERAGE_MIN_WORD_LEN = 3;
const COVERAGE_TOP_PASSAGES = 5;
// Held-out-document hard negatives (see the block above the negatives
// gate): how many title-template questions to ask about the sampled
// documents, and what fraction of the corpus's chunks those documents
// should add up to.
const HELDOUT_QUERIES = 36;
const HELDOUT_CHUNK_RATE = 0.12;
const BLOOM_SEEDS = [0, 0x9e3779b9];
const MAX_POSITIVES = 96;
const GIBBERISH_QUERIES = 24;
const MIN_VERIFIED_POSITIVES = 4;
const MIN_NEGATIVES = 8;
const MIN_AUC = 0.85;
// A foreign-bank query whose best passages share at least this fraction of
// its content words is not off-topic for this corpus (a developer-docs
// corpus can legitimately discuss "docker compose port binding"); like a
// foreign query retrieving closer than the median positive, it is kept
// out of the fit and reported as eval-only overlap.
const FOREIGN_OVERLAP_COVERAGE = 0.5;
// Hard threshold: this percentile of off-topic probability. 0.95 rather
// than v5's 0.9 because the off-topic pool is larger and cleaner than the
// hard-negative pool was, and a "none" verdict hides results.
const OFFTOPIC_HARD_QUANTILE = 0.95;
// Encoder-guided substitution positives (see the comment above the loop
// that builds them): sampled over up to this many base positives, from a
// vocabulary capped to the SUBSTITUTION_VOCAB_TOP most frequent corpus
// words, keeping a swap only above SUBSTITUTION_MIN_COS word-level cosine
// similarity. 0.55 measured as the point past which swaps stayed
// recognizably related rather than incoherent on a real corpus (0.35 let
// through nonsense like "fourth location sits during layout").
const SUBSTITUTION_BUDGET = 32;
const SUBSTITUTION_VOCAB_TOP = 800;
const SUBSTITUTION_MIN_COS = 0.55;
// Paraphrase-agreement was tried and abandoned (comparison telemetry
// only): every grounding feature above measures resemblance between a
// query and a passage's text, and each gave a topically-adjacent-but-
// unanswered passage the same high score a truly answered one gets, since
// resemblance is exactly what a topically-close passage has plenty of.
// paraphraseAgreement1 asked a structurally different question instead —
// not "does this look like a match" but "does retrieval agree with
// itself under rewording": several independent paraphrases of the SAME
// question (via the encoder-guided substitution mechanism below, run
// with different seeds) were searched independently, scored by the
// fraction whose rank-1 hit landed back on the original source. It
// measured a good base rate on genuinely answerable questions (0.83 mean
// agreement on internal-docs), but tested directly against real Pride and
// Prejudice questions with hand-varied wording, it did not separate
// answerable from unanswerable at all — both classes drifted to
// different top-1 passages under rewording at similar, high rates (~65-
// 75% either way). Retrieval instability under paraphrasing turned out to
// be common regardless of whether the question is actually answered,
// likely because many topically-similar chapters compete for rank 1 in a
// novel — so retrieval stability is not, on this evidence, a reliable
// proxy for fact-presence either.
// Point 5's real-query/paraphrase-stress gate: a calibrator that abstains
// on more than a third of genuinely paraphrased (or human-written) queries
// is failing the thing this whole redesign exists to catch, regardless of
// what the cross-validated AUC reports — measured on dutch-grammar-src, a fit that
// cleared every other gate still abstained on 50% of its held-out
// substitution positives, which is not a fit that should ship.
const MAX_PARAPHRASE_ABSTENTION_RATE = 1 / 3;
// Below this many rows, an abstention *rate* is not a measurement — it is
// one row's outcome wearing a percentage. Measured directly: a pack with
// independently-verified 0% real-world false abstention (35 hand-written
// probe questions, none abstained) failed this gate on a single bad row
// out of two substitution positives, because n=2 can only ever read 0%,
// 50%, or 100%. Below the floor the gate is skipped, not loosened — the
// fit still has to clear the cross-validated AUC gate.
const MIN_PARAPHRASE_STRESS_SAMPLE = 8;
// Threshold sanity rails. When the probe classes separate perfectly (small
// or template-uniform corpora), the logistic saturates: positive
// probabilities pile near 1 and percentile thresholds land absurdly high —
// a hard bar above 0.5 abstains on queries the model itself scores as
// more-likely-answerable-than-not, which is never the intended asymmetry.
// The ceiling caps that failure; the band floor guarantees a real "weak"
// caveat region even when no weak probes were generated (rank-5..K probe
// sources don't exist on corpora where every probe retrieves at rank 1).
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

// Pick whole documents (grouped by title) to ask held-out-document negative
// questions about: greedy over a shuffled title order until ~HELDOUT_CHUNK_RATE
// of chunks are covered, at least two titles so the negatives are not all one
// topic, never more than 12, and always leaving at least two titles untouched.
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

// Keyword-style positives sampled from chunk content words. Title-templated
// questions retrieve too easily — every fit positive lands near the corpus
// and the hard threshold creeps up until honest paraphrases abstain. These
// sit closer to how a real query scores, dragging the answerable floor down
// to where it belongs.
// Real queries are interrogative sentences, not keyword runs; a fit whose
// positives are all bare keywords (or title templates) learns a phrasing
// distribution narrower than what users type, and on uniform corpora the
// thresholds overfit to it — measured on a 94-passage synthetic registry,
// where naturally-phrased supported questions landed below a hard threshold
// of 0.68. Rotating the same content words through question frames keeps
// the retrieval target identical while spreading the phrasing distribution.
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

// Coverage ignores question/template scaffolding on top of the base
// stopwords: "how does X work" should be grounded by X's words appearing in
// the passage, not by "how"/"does"/"work". The list ships in the asset so
// the reader's mirror cannot drift.
// Quantifiers and modal/hedging words added here are unambiguous
// scaffolding ("how MUCH", "how MANY", "EACH year", "WOULD/COULD X") —
// deliberately not every word a review of false-positive coverage might
// flag: "make", "like", "other", and "only" stay OUT, since they are also
// ordinary content words in technical prose ("make an HTTP request",
// "looks like", "other endpoints", "only admins can..."), and stopping
// them here would zero out coverage for a query that is genuinely asking
// about that word.
const COVERAGE_STOPWORDS = new Set([...STOPWORDS,
  'what', 'how', 'who', 'why', 'where', 'when', 'do', 'does', 'did', 'me',
  'tell', 'about', 'explain', 'explained', 'facts', 'information',
  'overview', 'history', 'definition', 'important', 'work', 'works',
  'known', 'anything', 'said',
  'much', 'many', 'each', 'would', 'could', 'some', 'such', 'very',
  'more', 'after', 'before']);

// Coverage weights words by informativeness: a word the corpus uses
// everywhere ("templates", "support" in a docs corpus) grounds any query
// that mentions it, so corpus-common words (chunk-level document frequency
// over ~5% of the corpus, shipped as a bloom) count at reduced weight
// rather than full — and rather than zero, which measured as punishing
// honest paraphrases whose only grounded words are common ones. A query of
// only common words degrades to plain coverage by construction (uniform
// weights cancel).
//
// The score is the MAX over the top COVERAGE_TOP_PASSAGES passages, not the
// union: a paraphrased query gets several chances to find the passage that
// shares its vocabulary, while an off-topic passage should not have its
// coverage padded out by unioning several weak partial matches.
const COVERAGE_COMMON_WORD_WEIGHT = 1 / 3;
// Light suffix stripping so "quantize" grounds against "quantization" and
// "configure" against "configuration" — present() below otherwise only
// forgave plurals, so any word-form mismatch at all (a verb the corpus uses
// as a noun, -ing vs -ed) scored zero coverage for a passage that plainly
// answers the query. Deliberately conservative: MIN_STEM_LEN guards against
// over-stripping short words ("king" stays "king", not "k"), and only the
// suffix families measured to matter on real technical-doc word pairs
// (quantize/quantization, authenticate/authentication, configure/
// configuration, validate/validation) are covered — this is not a general
// stemmer, just enough to stop coverage1 punishing common noun/verb
// word-form drift between a query and its answer passage. Kept in
// complete/retrieval-abstention.mjs in lockstep (identical rule list and
// order) since the reader scores queries against shipped passages with
// this exact function, not a copy the builder only used at fit time.
const STEM_MIN_LEN = 4;
const STEM_SUFFIXES = ['ization', 'isation', 'ication', 'ation', 'ition', 'tion', 'ing', 'ed', 'ate', 'ize', 'ise'];
function stem(w) {
  for (const suf of STEM_SUFFIXES) {
    if (w.length - suf.length >= STEM_MIN_LEN && w.endsWith(suf)) return w.slice(0, -suf.length);
  }
  if (w.length - 1 >= STEM_MIN_LEN && w.endsWith('e')) return w.slice(0, -1);
  return w;
}
// tokenize() keeps the apostrophe ([a-z0-9']+), so a possessive is one
// token ("darcy's") that a plain-name mention in the passage ("darcy")
// never matches under presentIn's plural/stem rules — "What is X's Y?" is
// among the most common question shapes there is, and this was silently
// losing X from coverage every time. Strips a trailing 's or bare
// trailing ' (own's', dogs') from BOTH the query's content words and a
// passage's body/heading words, so the match works in either direction
// (a possessive in the query against a plain mention in the passage, or
// the reverse).
function stripPossessive(w) {
  if (w.endsWith("'s")) return w.slice(0, -2);
  if (w.endsWith("s'")) return w.slice(0, -1);
  if (w.endsWith("'")) return w.slice(0, -1);
  return w;
}
// A word present only in a passage's heading, not its body, still grounds
// the query — a heading names its section's exact topic — but at reduced
// credit: a heading is a handful of words repeated verbatim by every
// title-templated query about that passage, so full credit there is the
// same free-coverage bug body-only scoring was built to fix. Reduced,
// not zero, credit avoids overcorrecting into the opposite failure (a
// real rank-1 hit whose match is in the heading scoring no coverage at
// all — see splitHeadingBody above).
const HEADING_COVERAGE_WEIGHT = 0.5;
// COVERAGE_MIN_WORD_LEN exists to drop short function words ("is", "to",
// "on") that carry no topical content on their own. A short NUMERAL is
// the opposite case: "Chamber 4" vs "Chamber 5" differ only in a
// one-character token, and it is the single most identity-bearing word
// in the query — dropping it left coverage scoring "chamber"/"cooling"/
// "use" as a perfect match against ANY chamber's cooling record, with no
// way to notice the passage names the wrong chamber. Numerals are
// unambiguous (unlike a bare single letter, which could be an article or
// an identifier depending on case the lowercased tokenizer has already
// discarded — not fixed here), so they are exempted from the length
// floor entirely rather than lowering the floor for every short token.
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

// Proximity grounding was tried and abandoned (comparison telemetry only,
// never shipped): coverageFrac and maxSim1 both score each query word
// independently against the whole passage, so a passage that happens to
// mention every query word — without ever stating the relation connecting
// them — scores as if it answered the question ("what colour were Jane's
// eyes" against a passage that separately discusses Jane's manner and
// someone else's eyes). proximityFrac required a query's own content
// words to co-occur within a token-distance window (12 tokens) as a
// cheap, dependency-free proxy for "this passage asserts something about
// these things together," not just "these things are each mentioned
// somewhere." Two negative results killed it: (1) it was maximally fooled
// by paraphrasing — a word swapped for a synonym is no longer at the
// "expected" position, so requiring literal co-location made the existing
// paraphrase-robustness problem worse, not better (paraphrase separation
// AUC 1.0, the worst possible score); (2) tested directly against real
// Pride and Prejudice chunks, it did not separate answerable from
// genuinely-unanswered questions at all — continuous narrative prose puts
// unrelated words within 12 tokens of each other constantly by pure
// happenstance, so proximity in flowing text is not a reliable proxy for
// "these words are asserting a relationship," only for "these words are
// in the same paragraph." A token window is the wrong tool for this;
// fixing it would need actual relation/predicate identification, which
// this codebase deliberately avoids (see buildEntityIndex's no-NLP-tagger
// approach) rather than a distance heuristic.

// MaxSim grounding (comparison feature — see the design note near
// MAXSIM_FEAT below): coverageFrac requires exact or stemmed word
// identity, so a query and its answer passage that use different words
// for the same idea ("login" vs "authentication", "big" vs "huge") ground
// each other at zero — a real, structural blind spot on a system built
// around a semantic, paraphrase-tolerant retriever. maxSimFrac replaces
// exact match with per-word embedding cosine similarity (ColBERT-style
// late interaction, computed cheaply here: the inline encoder's per-token
// hidden states are already produced by the same forward pass embed()
// uses for the sentence vector — see embedWords() in
// complete/inline-transformer.mjs — so this costs one extra encoder call
// per scored passage, not one per word). For each query content word, the
// score is the best cosine similarity against any content word in the
// passage (heading words at reduced credit, same asymmetry as
// coverageFrac and for the same reason), averaged across query words with
// the same common-word downweighting coverageFrac uses.
//
// Raw cosine similarity between two encoder word vectors does not have a
// meaningful zero: unrelated words in the same embedding space still
// cluster somewhat, so even a passage that shares nothing with the query
// scores well above 0 — measured on real corpora, ~0.45-0.55 for pairs of
// genuinely unrelated corpus words, depending on the corpus's own
// vocabulary and encoder. Using the raw value as-is (as an earlier
// version of this file did, blended via max() with coverage1 into the
// fit) inflated the grounding signal's floor on every row, not just
// grounded ones, and pushed the fitted abstention threshold up sharply —
// confirmed causing widespread false abstention in production on a real
// corpus. estimateMaxSimBaseline fixes this the same way a z-score fixes
// an uncentered measurement: sample random pairs of distinct corpus
// words, take a high percentile of their cosine similarities as "how
// similar do two unrelated words look by chance in THIS corpus, under
// THIS encoder," and rescale every maxSimFrac call so that baseline maps
// to 0 and a perfect match still maps to 1. A word pair that's no more
// similar than corpus-typical unrelated words now correctly scores near
// 0, the same way coverageFrac would.
//
// A linear rescale alone still lets marginal, barely-above-baseline
// similarity through at meaningful (if small) credit, which is enough
// noise on a small hard-negative pool to tip a corpus's held-out AUC
// across the MIN_AUC gate — measured directly: Pride and Prejudice went
// from clearing the gate comfortably on coverage1 alone (~0.86) to
// missing it (0.849 < 0.85) once maxSim1 (correctly rescaled) joined the
// blend, reproducibly across identical reruns. MAXSIM_GAMMA compresses
// that low end further (a power curve, not just linear) — matched
// similarity credit above baseline still approaches 1, but the marginal
// range right above baseline now contributes closer to 0, so maxSim1 can
// still rescue a genuine, strongly-similar paraphrase without adding
// boundary jitter from near-chance similarity everywhere else.
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
  // Rescale so the corpus's own unrelated-word baseline maps to 0 and a
  // perfect match (cosine 1) still maps to 1, then apply MAXSIM_GAMMA to
  // suppress marginal, barely-above-baseline credit further — see this
  // function's and MAXSIM_GAMMA's header comments. A raw similarity at or
  // below baseline is exactly the "no better than chance" case
  // coverageFrac would score 0.
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

// Passage-direction grounding was tried and abandoned (not reverted from a
// shipped state — this never got past comparison telemetry): instead of
// comparing the query against individual passage words (maxSim1's
// approach), find each passage's own dominant semantic direction via
// power iteration over its content words' embeddings (the top principal
// component — "what this passage is about"), then score cosine(query
// embedding, that direction). Measured on internal-docs: the top
// component captured only 8-13% of a passage's total variance (26-120
// content words per chunk) — nowhere near dominant, so the resulting
// "direction" tracked closer to embedding-space noise (the same common
// direction nearly all transformer word vectors share, which is also why
// raw maxSim1 cosine similarity has no real zero) than any genuine topic
// axis. Standalone hard-negative separation AUC came in at 0.61 versus
// coverage1's 0.98, confirming it. Short chunks (the common case here)
// are the wrong scale for small-sample PCA to find a stable direction;
// might behave differently on much longer passages, untested.

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

// Proper-noun detection without a tagger: a word that is capitalized
// somewhere in a chunk's raw text but never appears in lowercase anywhere
// in the corpus is very likely a name (character, place) rather than a
// sentence-initial common word — "Elizabeth" never occurs as "elizabeth"
// in running prose, but "Dogs" (sentence-initial "Dogs bark") usually also
// occurs as plain "dogs" elsewhere. This needs no NLP dependency and is
// cheap over a whole corpus; it undercounts (single-mention names, or ones
// that coincide with a common word, are missed), and roman numerals /
// day-and-month names slip through as false positives (harmless — a
// "swap" landing on one just produces an odd-but-still-unanswerable
// negative, verified by retrieval like every other class here).
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
  // A name appearing in nearly every chunk (a protagonist) is a weak
  // negative candidate — swapping it in barely changes anything a reader
  // (or the retriever) would find implausible. Keep entities confined to a
  // minority of the corpus, so a swap is a genuine mismatch.
  const maxDf = Math.max(2, Math.ceil(chunks.length * 0.4));
  const entitySet = new Set([...df.entries()].filter(([, c]) => c >= 1 && c <= maxDf).map(([w]) => w));
  return {
    entities: [...entitySet],
    display: (w) => capitalized.get(w) || w,
    // Which of this corpus's entities occur in a piece of text — O(words
    // in text) via tokenize + set membership, not a regex scan per entity
    // per call (that was O(positives x entities) regex compilations,
    // measured pathologically slow on a corpus with hundreds of detected
    // names: Pride and Prejudice alone has 200+).
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

// Human-written calibration queries (point 2a — the highest-leverage
// positive source, since these are the only rows that are not a
// template's artifact): a JSON-lines file, one {text, expectId} or
// {text, expectTitle} object per line, path given by
// runtime.calibrationQueries in pikelet.config.json (resolved relative to
// the project directory, same as encoder.calibrationPath). Optional — a
// corpus without one just has no human-written source and relies on the
// synthetic classes plus substitution/morphological positives.
export async function loadCalibrationQueries(config, projectDir, log) {
  const rel = config?.runtime?.calibrationQueries;
  if (!rel || !projectDir) return [];
  const file = path.resolve(projectDir, rel);
  let text;
  try {
    text = await fs.readFile(file, 'utf8');
  } catch (err) {
    // The path is set only when someone asked for it (--calibration-queries
    // or the project config), so a missing file is a mistake to report, not
    // a corpus without human questions: it used to calibrate without them
    // and say nothing.
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
  // Single-word vector cache for maxSimFrac (comparison feature): the same
  // content word recurs across many probe rows (a corpus's own vocabulary
  // is finite), so caching avoids re-embedding it every time it appears
  // in a query or a passage.
  const wordVecCache = new Map();
  const embedWordVec = embedWordVecs ? async (w) => {
    if (!wordVecCache.has(w)) {
      const vecs = await embedWordVecs(w);
      wordVecCache.set(w, vecs.get(w) || new Float32Array(config.embedding.dims));
    }
    return wordVecCache.get(w);
  } : null;
  // maxSim1's noise floor for THIS corpus under THIS encoder — see
  // estimateMaxSimBaseline's header comment. Computed once, up front,
  // from the same content-word vocabulary maxSimFrac itself draws from.
  const maxSimVocab = embedWordVec
    ? [...new Set(chunks.flatMap((c) => tokenize(c.text || '').filter((w) => w.length >= COVERAGE_MIN_WORD_LEN && !COVERAGE_STOPWORDS.has(w))))]
    : [];
  const maxSimBaseline = embedWordVec ? await estimateMaxSimBaseline(maxSimVocab, embedWordVec, SEED ^ 0xba5e11e) : 0;

  // Nothing is held out at the corpus level: the earlier held-out-document
  // hard-negative class asked a titleQuestions template about an excluded
  // title, which is a DIFFERENT generator input (a title never used for a
  // positive) from the positives it was fit against — the model could
  // learn "this title-shape belongs to the held-out set" instead of
  // answerability. Ablation (below) is now the only hard-negative source,
  // and it is generator-symmetric with positives by construction (same
  // text, answer excluded), so no title needs to be withheld up front.
  const retainedPos = chunks.map((_, pos) => pos);
  const retainedSet = new Set(retainedPos);
  const K = Math.min(10, retainedPos.length);
  // A chunk's text starts with its own heading, echoed as the first line
  // (ingest.mjs's section-to-chunk join). Excluding it entirely (the first
  // attempt at this) was too strong a correction: a query that names the
  // exact concept a heading names — "export buffer ownership" against a
  // section titled exactly that — retrieves the right passage at rank 1,
  // but with the heading discounted to zero, coverage1 sees no grounding
  // at all and the fit, which weighs coverage several times heavier than
  // d0 (see the design note below), can score a perfect hit as "none".
  // Splitting heading from body lets coverageFrac count a heading match as
  // real but partial evidence — see HEADING_COVERAGE_WEIGHT — rather than
  // full credit (the original bug: a title-templated positive's own words
  // ARE the heading, so full credit there is free, unearned coverage) or
  // zero credit (this bug).
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
    // null when the caller didn't provide embedWordVecs (e.g. a kind-2
    // external-encoder build, where word-level embedding would mean a live
    // call to a host encoder per word — a cost this feature isn't asking
    // anyone to pay yet).
    const maxSim1 = embedWordVec ? await maxSimFrac(text, passages, isCommon, embedWordVec, maxSimBaseline) : null;
    return {
      d0,
      margin,
      mean10,
      known_frac: knownFrac(text, bloom, bits),
      // GROUNDING_FEAT (the actual fit feature) is coverage1 itself — see
      // its design note above for why maxSim1 doesn't safely fold in via
      // max(), even rescaled and damped. maxSim1 is still computed and
      // reported for comparison only.
      [COVERAGE_FEAT]: coverage1,
      [MAXSIM_FEAT]: maxSim1,
    };
  };

  const search = (text) => evidence(text);
  // Ablation search (point 1): the same retrieval, minus the positive's
  // answer. Used to build a paired hard negative from a verified
  // positive's own text — see the negatives section below. excludeSet is
  // every chunk id the query's label depends on: one chunk for a
  // content-word positive, every chunk of the source document for a
  // title positive (excluding only its one sampled chunk would leave
  // sibling chunks of the same document still answering the query,
  // which is not a real ablation of "the source").
  const searchExcluding = (text, excludeSet) => evidence(text, { excludeIds: excludeSet });
  const ablationTargets = ({ sourceId, sourceTitle }) => new Set(sourceId !== undefined
    ? [sourceId]
    : retainedPos.filter((p) => (chunks[p].title || '').trim() === sourceTitle));

  const rows = [];
  let droppedPositives = 0;
  const verify = (hits, sourceTitle, sourceId) => {
    // A positive counts only when retrieval verifiably lands on the source
    // document — otherwise the query is unanswerable in practice and would
    // drag the no-false-abstain floor toward zero. Title queries verify by
    // title (any chunk of the source document counts); content-word queries
    // verify by the chunk they were sampled from. Checked against the base
    // K window of the vector order, not the whole rerank window — a
    // source landing at, say, rank 40 is not "retrieval verifiably lands
    // on the source" in any sense the reader's own top-K would agree with.
    const topK = hits.vector.slice(0, K);
    return sourceId !== undefined
      ? topK.some((h) => h.id === sourceId)
      : topK.some((h) => (chunks[h.id]?.title || '').trim() === sourceTitle);
  };
  // Positives: title/content-word templates, verified by retrieval.
  // Every one of these copies its words from the target passage, so
  // coverage1 is ~1.0 for the whole class by construction — not a
  // generator-identity problem (every negative below is retrieval-
  // verified the same way and paired 1:1 with a positive via ablation,
  // so the fit cannot tell positive from negative by query shape), but
  // a genuine blind spot: the fit never sees an answerable query with
  // partial lexical overlap, so it has no evidence that low coverage
  // can still mean "answered" — and leans on coverage harder than
  // anything else as a result (measured: coverage weighted ~4x d0 in a
  // fit trained on this class alone).
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
  // Encoder-guided substitution: swap some of a verified positive's
  // content words for a same-corpus vocabulary word the encoder places
  // close by cosine similarity, excluding words the source passage
  // already contains, then re-verify by retrieval — a substituted query
  // that no longer retrieves its source is dropped, not mislabeled.
  // Retrieval verification is what makes this safe to mix with the base
  // class: the label is earned the same way every other positive earns
  // it, so its presence doesn't let the fit cheat on query shape — it
  // only teaches the fit that partial lexical overlap can still be
  // answerable, which the base class alone cannot. Vocabulary capped to
  // the most frequent SUBSTITUTION_VOCAB_TOP corpus words (excludes
  // hapax/rare words — typos, ids, proper nouns, corpus noise) and
  // SUBSTITUTION_MIN_COS gates coherence: below it, a "nearest" neighbor
  // in a sentence-trained encoder's word-level cosine is not actually
  // related, just the least-far of a bad candidate pool — measured
  // producing incoherent queries ("fourth location sits during layout")
  // that still happened to retrieve their source on a small corpus.
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

  // Human-written queries (runtime.calibrationQueries), when the corpus
  // ships them, are held out of the fit entirely — never trained on,
  // scored below after the model exists, and reported as the headline
  // validation number. They are the only rows in this whole pipeline
  // that are not a template's artifact, so they are the one number that
  // actually answers "does this generalize past what the generator
  // makes" rather than measuring the fit against more of its own output.
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

  // Ablation negatives (point 1) replace recombination as the load-bearing
  // hard class. Recombination's word-salad construction always landed
  // coverage1 near 0.5 by how it combines two chunks' words, so the fit
  // learned "roughly half the words present -> unanswerable" — exactly the
  // regime a real paraphrase of a real answer lands in (measured: the
  // same coverage band). An ablation negative reuses a verified positive
  // query's exact text, scored with only its source chunk excluded from
  // the index (searchExcluding) — same known_frac, same query length, same
  // lexical shape as the positive it is paired with, differing only in
  // what the top passages actually contain. The model is forced onto
  // d0/margin/mean10/coverage1 as measured against the retrieved content,
  // not onto a query-shape tell, because the query is byte-identical to a
  // positive.
  //
  // The drop check compares against the PAIRED positive's own d0/coverage,
  // not a corpus-wide median: measured on a real corpus, positiveMedianD0
  // (spanning weak title-template phrasings as well as tight ones) was
  // loose enough that half of all ablation negatives slid under it while
  // still scoring coverage1 near 1.0 — a different chunk lexically
  // supporting the same query almost as well as the excluded source, which
  // is a real duplicate/cross-reference, not evidence the label is fine.
  // A negative is dropped when EITHER signal alone shows the answer still
  // exists elsewhere: distance staying close to the paired positive's own
  // d0 (a near-duplicate chunk ranks almost as well), OR coverage staying
  // high in absolute terms (some passage still quotes most of the query's
  // words) — a close-but-different-words match and a distant-but-still-
  // quoting match are both real duplicates/cross-references, not evidence
  // the ablated query is actually unanswerable. Coverage is checked
  // against an absolute floor, not only relative to the paired positive:
  // a positive that already had coverage 1.0 leaves no room for a
  // relative drop to ever fire, so a negative sitting at the same 1.0
  // would otherwise never be caught.
  // Paired from every verified positive: each one gets exactly one
  // ablation negative built from its own text, so positives and
  // negatives are the same generation process throughout — the only
  // thing that can differ between a paired row's two labels is what
  // retrieval returns.
  // The relative-coverage branch only means something when the paired
  // positive itself was well-grounded: for a positive whose own coverage
  // was already low (0.2), "did coverage stay above 75% of that" is
  // satisfied by almost any coverage value >= 0.15 — nearly always true,
  // and not evidence of anything, since 0.2 was never meaningfully
  // grounded to begin with. Measured on a real corpus: this false-
  // positive pattern alone caused every single ablation negative to be
  // dropped as "still grounded". Gated on the positive clearing an
  // absolute floor first.
  const ABLATION_RELATIVE_COVERAGE_FLOOR = 0.5;
  let ablationDropped = 0;
  for (const positiveRow of positives) {
    const { text } = positiveRow;
    const targets = ablationTargets(positiveRow);
    // signalsFor already degrades gracefully below the nominal K window
    // (it uses whatever top.length actually is for margin/mean10) — the
    // only real requirement is that ablation leaves something to search
    // at all. Requiring the post-ablation pool stay >= K (the ORIGINAL,
    // pre-ablation window size) is impossible on any corpus small enough
    // that K was itself capped down to retainedPos.length: excluding
    // even one chunk then always leaves fewer than K, so every single
    // positive got skipped before ever reaching a search — measured on
    // an 8-chunk fixture, where this alone produced 0 ablation negatives
    // out of 56 verified positives.
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

  // Entity-swap negatives: ablation assumes excluding a passage's source
  // document leaves the query genuinely unanswerable elsewhere in the
  // corpus — true for topically distinct sections (docs, reference
  // material), false for thematically uniform or narrative content
  // (a novel's chapters share characters, settings, and themes, so
  // excluding one chapter often leaves the query answerable by another).
  // Measured on Pride and Prejudice: only 3 of 101 verified positives
  // survived ablation as genuine hard negatives, well under
  // the v5 hard-negative minimum, and the corpus shipped unscored. A swapped-entity
  // negative sidesteps the topical-uniqueness assumption entirely: take a
  // verified positive that names a proper noun (character, place —
  // detected corpus-wide, see buildEntityIndex) and substitute a
  // different entity that never co-occurs with it, so the query becomes
  // an entity combination that (as far as the corpus is concerned) never
  // happened — genuinely unanswerable regardless of how uniform the
  // corpus's themes are. Verified by retrieval like every other class
  // here: kept only when it does NOT retrieve the original source.
  const entityIndex = buildEntityIndex(chunks);
  let entitySwapDropped = 0;
  if (entityIndex.entities.length >= 2) {
    const entityRng = rng(SEED ^ 0xe57171);
    // Entities that occur anywhere in the source chunk(s) — excluded as
    // replacement candidates (see below) — indexed once per positive via
    // its own (small) chunk set rather than scanning every corpus
    // entity's (potentially large) chunksOf per positive.
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

  // Held-out-document negatives: title-template questions about a sampled
  // set of documents, each scored with every chunk of its document
  // excluded (searchExcluding), kept only when the best remaining hit is
  // farther than the positive median d0. This is the class ablation
  // replaced, reinstated because ablation's drop rule discards any
  // negative whose coverage stays high — and on entity-dense corpora that
  // is exactly the "entity named, fact absent" negative the calibrator
  // otherwise never sees. Measured on the 94-record Veyra registry: 75 of
  // 94 ablations dropped as "still grounded", the fitted d0 weight came
  // out positive, and unsupported questions naming a real entity scored
  // strong (harness false answers 8 -> 19 against the v3 calibrator that
  // still had this class). These rows have no positive twin — the
  // objection that retired them — but the model never sees query text,
  // only retrieval signals, and the effect on real questions decides it.
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
  // diagnostic only (see the v6 note at the top of this file).
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

  // Fit: logistic regression on standardized signals, gradient descent —
  // the same optimizer as the wiki calibrator so the two assets stay
  // comparable. v6 fits positives vs. off-topic negatives (foreign bank
  // plus gibberish), class-balanced so neither side dominates; hard
  // negatives are scored by the result but never fit (see the v6 note at
  // the top of this file). v5 split this into a stage-1 off-topic gate
  // and a stage-2 positives-vs-hard-negatives model; with off-topic as
  // the target the two stages are the same fit, so there is one.
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
  // Class-balanced: negative rows are reweighted so total negative weight
  // equals total positive weight. Human-written queries are held out
  // entirely (point 5, see humanRows below).
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
  // cvAucHard: how well the score separates in-domain-unanswerable
  // questions (the hard-negative classes) from answerable ones — a
  // diagnostic, not a gate. Hard negatives are never in the fit, so the
  // full model's probabilities for them are already held out; positives
  // use their cross-validated probabilities. Expected well below cvAuc:
  // relevance is what the score measures.
  const cvAucHard = aucFor(heldoutPos, hardNegatives.map((r) => ({ p: model.prob(r) })));
  // Per-generator-class held-out AUC: the base template class alone
  // tends to separate cleanly (its positives are all coverage ~1.0,
  // which the fit leans on hard), which can mask the substituted class
  // — the one actually testing "does partial lexical overlap still
  // score as answerable" — scoring poorly. A regression here is exactly
  // the failure mode a pooled number hides.
  const cvAucBase = aucFor(heldoutPos.filter((r) => r.genKind === undefined), heldout.filter((r) => r.label === 0));
  const cvAucSubstituted = aucFor(heldoutPos.filter((r) => r.genKind === 'substituted'), heldout.filter((r) => r.label === 0));
  const gateAuc = cvAuc ?? fitAuc;
  if (gateAuc === null || gateAuc < MIN_AUC) {
    return skip(`${cvAuc === null ? 'fit' : 'cross-validated'} AUC ${gateAuc === null ? 'n/a' : gateAuc.toFixed(3)} separating on-topic from off-topic queries < ${MIN_AUC}`);
  }
  // A d0 weight above zero says a worse best-distance is more relevant —
  // impossible for a real relevance signal. Measured symptom of a
  // negative pool whose retrieval barely differs from its positives': the
  // fit then scores on coverage alone and answers anything that names a
  // corpus entity. AUC gates do not catch it (the coverage-only fit still
  // separates the synthetic rows), so check the sign directly.
  const d0Weight = w[FEATS.indexOf('d0')];
  if (!(d0Weight <= 0)) {
    return skip(`fitted d0 weight ${d0Weight.toFixed(3)} is not negative: distance carries no signal in this fit `
      + '(the negatives are indistinguishable from positives by retrieval), so the verdict would reduce to a coverage threshold');
  }

  // Threshold placement (point 4): hard is set from the negative side —
  // the OFFTOPIC_HARD_QUANTILE percentile of off-topic probability — not
  // from where positives happen to land. Placing it from the positive
  // side (posFloor * 0.5, or a blend toward the positive floor) let an
  // easy fit's positive distribution set the bar: generated positives at
  // p~0.99 pushed hard high enough to abstain on queries the model itself
  // scored as more-likely-relevant-than-not. Percentiles still guard the
  // tails (a single outlier negative should not set the whole bar), but
  // the reference point is where off-topic queries actually live: "none"
  // when the score looks like a query this pack does not cover.
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
  // Saturation rail: see HARD_CEILING. A clamp engaging means the fit
  // separated its own probes too cleanly for percentile placement to be
  // meaningful — log it, because real phrasings unlike the probes are the
  // queries a too-high bar silently hides.
  const hard = Math.min(hardRaw, HARD_CEILING);
  if (hardRaw > HARD_CEILING) {
    log(`Calibration: hard threshold computed at ${hardRaw.toFixed(3)} (saturated fit); clamped to ${HARD_CEILING}. `
      + 'Probe classes separate almost perfectly — treat abstention boundaries on this corpus with caution.');
  }
  const weakCeil = weakP.length ? quantile(weakP, 0.9) : 0;
  // "Strong" must clear the off-topic mass. When off-topic negatives
  // overlap the weakest positives (hardThresholdOverlap), the hard
  // threshold stays protective of positives — a false abstain hides
  // results — but the weak threshold rises to the off-topic ceiling, so
  // overlapped queries are shown with a caveat instead of full
  // confidence. Capped at the positive median: past that the fit is too
  // entangled for the ceiling to be meaningful, and the cross-validated
  // AUC gate is the real protection.
  const weakRaw = Math.min(
    Math.max(
      weakCeil > hard && weakCeil < posFloor ? Math.sqrt(weakCeil * posFloor) : posFloor * 0.9,
      hardCeil,
      hard,
    ),
    Math.max(quantile(pos, 0.5), hard),
  );
  // Band floor: without weak probes the sandwich above can collapse the
  // weak band to nothing (measured 0.015 wide on a synthetic corpus),
  // making every uncertain query binary strong/none. Widening only ever
  // demotes strong to weak — results still shown, with a caveat — so the
  // floor is safe in the direction that matters.
  const weak = Math.min(0.95, Math.max(weakRaw, hard + MIN_WEAK_BAND));

  // Point 5: validate against something the generator didn't make. Every
  // other row in this fit is templated — the model could still be
  // fitting to properties specific to this one template family rather
  // than relevance. Human-written queries (held out above, never fit) are
  // the only rows in this pipeline that are not that family's artifact,
  // so they are the one real validation this asset can have.
  // There is no synthetic fallback for this check: a proxy built from
  // the same generator that made the training positives (the earlier
  // "paraphrase-stress" metric, scored on positives already IN the fit)
  // is not validation, it is training accuracy wearing a different name
  // — it cannot detect the fit learning the generator instead of
  // relevance, because it IS the generator. Corpora without
  // runtime.calibrationQueries are validated only by the cross-validated
  // AUC gate above.
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
  // A deterministic sample of the verified positives, exported as golden
  // queries the artifact can embed: each was checked above to actually
  // retrieve its source, so a reader can re-run them later (verify_pack,
  // acceptance tests) as tests stored inside the file. Spread evenly so
  // both title and content-word templates are represented.
  // Human-written queries are prepended: real, corpus-author-written
  // questions are the most valuable thing verify_pack could re-run, so
  // they are never displaced by the cap below.
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
  // maxSim1 comparison (see MAXSIM_FEAT): how would this feature alone
  // separate positives from hard negatives, versus coverage1 doing the
  // same job — computed the same way aucFor scores the real fit, so the
  // two numbers are directly comparable. null when embedWordVecs wasn't
  // provided (maxSim1 is null on every row in that case).
  const maxSimRows = hardNegatives.length && positives.length && positives[0][MAXSIM_FEAT] !== null
    ? { pos: positives.map((r) => ({ p: r[MAXSIM_FEAT] })), neg: hardNegatives.map((r) => ({ p: r[MAXSIM_FEAT] })) }
    : null;
  const coverageRows = hardNegatives.length
    ? { pos: positives.map((r) => ({ p: r[COVERAGE_FEAT] })), neg: hardNegatives.map((r) => ({ p: r[COVERAGE_FEAT] })) }
    : null;
  // The hard-negative comparison above tests "did we hide the answer" —
  // ablation negatives share the positive's own query text, so they stay
  // lexically close to it and can't show whether a feature survives a
  // genuine reworded question. The substituted class (retrieval-verified,
  // built by swapping words for corpus vocabulary NOT in the source
  // passage — see the loop above) is a real paraphrase: same answer,
  // deliberately reduced lexical overlap with its own source. A grounding
  // feature that fights the semantic retriever's paraphrase tolerance
  // should score substituted positives lower than base positives even
  // though both are equally answerable; maxSim1's whole premise is that
  // it shouldn't dip nearly as much as coverage1 does here.
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
    // "How similar do two unrelated words in this corpus look by
    // chance" — see estimateMaxSimBaseline. maxSim1 is already rescaled
    // against this before every other number below is computed.
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
    // off-topic queries (see the v6 note at the top of this file).
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
    // Point 5: human-written queries (runtime.calibrationQueries) are
    // held out of the fit entirely and validated here instead — the only
    // real validation this asset can have (see the comment above this
    // block in the fit). null fields mean the corpus shipped none; a
    // corpus without them is validated only by cvAuc.
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
    // Hard negatives (ablation, entity-swap, held-out-doc) feed only the
    // cvAucHard diagnostic since v6. Ablation: same query text as a
    // verified positive, scored with the answer excluded — see the
    // comment above the ablation loop. Entity-swap tops it up on corpora
    // where ablation alone starves (thematically uniform / narrative
    // content — see the comment above that loop).
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
    // fitAuc/cvAuc are positives vs. off-topic negatives, the fit's own
    // target: fitAuc in-sample, cvAuc the pooled held-out AUC from the
    // deterministic 5-fold cross-validation, which the gate checks.
    // cvAucHard is positives vs. the hard-negative classes (in-domain-
    // unanswerable), never fit, never gated: how far "strong" can be
    // read as "answered" on this pack. null when no hard negative was
    // generated.
    fitAuc: fitAuc === null ? null : +fitAuc.toFixed(6),
    cvAuc: cvAuc === null ? null : +cvAuc.toFixed(6),
    cvAucHard: cvAucHard === null ? null : +cvAucHard.toFixed(6),
    // Per-generator-class held-out AUC: a regression specific to the
    // substituted (low-coverage) class is exactly the failure a pooled
    // number can hide, since the base class alone tends to separate
    // cleanly. null means too few held-out rows of that class landed in
    // any fold to compute it.
    cvAucByGenKind: {
      base: cvAucBase === null ? null : +cvAucBase.toFixed(6),
      substituted: cvAucSubstituted === null ? null : +cvAucSubstituted.toFixed(6),
    },
    hardThresholdOverlap: hardOverlap,
    hardThresholdClampedFrom: hardRaw > HARD_CEILING ? Number(hardRaw.toFixed(6)) : null,
    vocab: { uniqueWords, keptWords, minCount },
    coverage: { topPassages: COVERAGE_TOP_PASSAGES, commonWords: commonWords.commonWords, commonDfCap: commonWords.dfCap },
    // Standalone separation power of coverage1 and maxSim1 (see
    // GROUNDING_FEAT's design note) — comparison only. maxSim1 does NOT
    // feed the fit (grounding1 is plain coverage1), and the reader's
    // matching useMaxSim/maxSimFrac code path has been removed entirely
    // (not just left unset here) — it cost 200-400 individual per-word
    // encoder calls per query, turning a 30-80ms search into 1.5-4.6s
    // whenever some calibration asset (however it came to exist) set
    // coverage.useMaxSim: true. This field name is kept for the
    // telemetry's own history; it is not read by anything anymore.
    maxSimVsCoverage,
    maxSimInFit: false,
  };
  // features[]/weights[] carry only the base topic features; grounding1
  // (== coverage1) rides in asset.coverage so a reader that predates it
  // scores the topic-only model against the same thresholds
  // (conservative — it lacks a term that is positive for answerable
  // queries) instead of feeding an unknown feature name NaN into the
  // logistic. useMaxSim is deliberately omitted: see GROUNDING_FEAT's
  // design note for why blending maxSim1 in at serve time is unsafe
  // here — a reader must compute plain coverageFrac.
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
