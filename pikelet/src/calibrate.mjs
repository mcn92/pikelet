// Build-time abstention calibration for complete kind-3 artifacts. Mirrors
// the signal and scoring math of the wiki pack calibrator
// (examples/04-static-wiki-pack/calibrate_abstention.mjs) and of the reader's
// scorer (complete/retrieval-abstention.mjs) — retrieval signals (d0, margin,
// mean10) plus the corpus-vocabulary known-token fraction, standardized and
// passed through a fitted logistic model — but is corpus-generic: positives
// are templated questions from chunk titles verified by retrieval.
//
// Negatives come in two classes, and the hard class is the load-bearing one.
// Easy negatives (off-domain query bank, synthetic gibberish) have low
// known-token fractions and distant retrieval; a fit trained on them alone
// hands known_frac a dominant weight and answers any in-domain query no
// matter how unsupported (measured: 6 of 8 unanswerable in-domain probes
// scored "strong" at p 0.75–0.94). Hard negatives populate the missing
// region — high known_frac, mid-range distance:
//   - held-out documents: whole documents (by title) are excluded from the
//     calibration searches (searchFiltered over the retained ids); templated
//     questions about them are in-domain by vocabulary and unanswerable by
//     construction — the wiki calibrator's held-out-shard trick, done
//     corpus-generically;
//   - cross-chunk recombinations: corpus content words drawn from two
//     unrelated chunks, with a lexical guarantee that no single chunk
//     contains most of the words.
// Both classes verify their labels by retrieval: a hard negative that
// retrieves as strongly as a median positive is suspected answerable
// (near-duplicate coverage, accidental paraphrase) and dropped. Off-domain
// queries that overlap the corpus are kept as eval-only rows — excluded from
// the fit but scored and reported, so the asset states its hard-negative
// operating point instead of silently discarding it.
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
const FEATS = [...BASE_FEATS, COVERAGE_FEAT];
const COVERAGE_MIN_WORD_LEN = 3;
const COVERAGE_TOP_PASSAGES = 5;
const BLOOM_SEEDS = [0, 0x9e3779b9];
const MAX_POSITIVES = 96;
const GIBBERISH_QUERIES = 24;
const HELDOUT_QUERIES = 36;
const RECOMBINATION_QUERIES = 16;
const HELDOUT_CHUNK_RATE = 0.12;
const MIN_VERIFIED_POSITIVES = 4;
const MIN_NEGATIVES = 8;
const MIN_HARD_NEGATIVES = 6;
const MIN_AUC = 0.85;
// The hard-negative bar is lower than the pooled bar: these queries sit near
// the decision boundary by design, and demanding easy-class separation from
// them would fail honest fits. Below this, the model cannot tell answerable
// from in-domain-unanswerable and must not ship.
const MIN_HARD_AUC = 0.75;
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
const COVERAGE_STOPWORDS = new Set([...STOPWORDS,
  'what', 'how', 'who', 'why', 'where', 'when', 'do', 'does', 'did', 'me',
  'tell', 'about', 'explain', 'explained', 'facts', 'information',
  'overview', 'history', 'definition', 'important', 'work', 'works',
  'known', 'anything', 'said']);

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
// shares its vocabulary, while a recombination negative's two source chunks
// each ground only their own half — a union would merge them to full
// coverage and erase the class.
const COVERAGE_COMMON_WORD_WEIGHT = 1 / 3;
function coverageFrac(text, passageTexts, isCommon) {
  const content = tokenize(text).filter((w) => w.length >= COVERAGE_MIN_WORD_LEN && !COVERAGE_STOPWORDS.has(w));
  if (!content.length) return 0;
  const weights = content.map((w) => (isCommon && isCommon(w) ? COVERAGE_COMMON_WORD_WEIGHT : 1));
  const weightSum = weights.reduce((a, c) => a + c, 0);
  let best = 0;
  for (const passageText of passageTexts) {
    const passage = new Set(tokenize(passageText || ''));
    const present = (w) => passage.has(w) || passage.has(`${w}s`) || passage.has(`${w}es`)
      || (w.endsWith('s') && passage.has(w.slice(0, -1)));
    const grounded = content.reduce((sum, w, i) => sum + (present(w) ? weights[i] : 0), 0);
    best = Math.max(best, grounded / weightSum);
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

// Hold out whole documents (grouped by title) from the calibration searches.
// Greedy over a shuffled title order until ~HELDOUT_CHUNK_RATE of chunks are
// excluded, preferring at least two held-out titles so the hard negatives are
// not all about one topic, and always retaining at least two titles for the
// positives. Corpora under three titles get no hold-out and rely on
// recombination negatives alone.
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

// In-domain word-salad negatives: two rare content words from each of two
// chunks with different titles, interleaved. Every word is in the shipped
// vocabulary bloom (high known_frac by construction), and a posting-list
// tally guarantees no single chunk in the FULL corpus contains three or more
// of the four words — so no document lexically supports the query. Words are
// drawn from retained chunks only; co-occurrence is checked corpus-wide.
function recombinationQueries(chunks, eligiblePos, n, seed, bloom, bits) {
  const next = rng(seed);
  const posting = new Map();
  chunks.forEach((chunk, pos) => {
    for (const w of new Set(tokenize(chunk.text))) {
      if (w.length < 4 || STOPWORDS.has(w)) continue;
      if (!posting.has(w)) posting.set(w, []);
      posting.get(w).push(pos);
    }
  });
  const rarityCap = Math.max(2, Math.round(chunks.length * 0.05));
  const wordsByPos = new Map();
  for (const pos of eligiblePos) {
    const ws = [...new Set(tokenize(chunks[pos].text))].filter((w) =>
      w.length >= 4 && !STOPWORDS.has(w) && posting.get(w).length <= rarityCap && knownFrac(w, bloom, bits) === 1);
    if (ws.length >= 2) wordsByPos.set(pos, ws);
  }
  const candidates = [...wordsByPos.keys()];
  const out = [];
  const seen = new Set();
  const pickTwo = (ws) => {
    const i = Math.floor(next() * ws.length);
    const j = (i + 1 + Math.floor(next() * (ws.length - 1))) % ws.length;
    return [ws[i], ws[j]];
  };
  let attempts = 0;
  while (out.length < n && attempts++ < n * 40 && candidates.length >= 2) {
    const a = candidates[Math.floor(next() * candidates.length)];
    const b = candidates[Math.floor(next() * candidates.length)];
    if (a === b || (chunks[a].title || '').trim() === (chunks[b].title || '').trim()) continue;
    const words = [...pickTwo(wordsByPos.get(a)), ...pickTwo(wordsByPos.get(b))];
    if (new Set(words).size < 4) continue;
    const tally = new Map();
    for (const w of new Set(words)) for (const pos of posting.get(w)) tally.set(pos, (tally.get(pos) || 0) + 1);
    if ([...tally.values()].some((c) => c >= 3)) continue;
    const text = [words[0], words[2], words[1], words[3]].join(' ');
    if (seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

function aucFor(posRows, negRows) {
  if (!posRows.length || !negRows.length) return null;
  let score = 0;
  for (const a of posRows) for (const c of negRows) score += a.p > c.p ? 1 : a.p === c.p ? 0.5 : 0;
  return score / (posRows.length * negRows.length);
}

// Reciprocal-rank-fusion constant, kept identical to complete/index.mjs's
// RRF_K so a fit-time fused ranking matches what the reader actually
// serves. A build-time-only constant drifting from the reader's would
// make coverage grade a passage the reader would never actually rank
// first.
const RRF_K = 60;
// The lexical candidate cutoff: only BM25 hits within this fraction of
// the top score join fusion. Kept in sync with complete/index.mjs's own
// cutoff for the same reason as RRF_K. Tighter than the reader's original
// /3 (tried first, then measured too permissive — see the LEX_AGREE_FEAT
// history this replaced): idf collapses common query terms into a flat,
// near-tied score mass, and a wide cutoff let that mass's members ride
// into fusion on a technicality rather than a real lexical match.
const LEXICAL_CUTOFF = 1.5;

export async function calibrateRetrievalAbstention({ Pikelet, chunks, vectors, config, embedQuery, lexicalIndex = null, log = () => {} }) {
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

  // All calibration searches run against the retained chunks only
  // (searchFiltered over one full index — ids stay chunk positions). The
  // signals shift slightly from serve time, where nothing is held out, but
  // every fit row is scored consistently, which is what the standardization
  // and thresholds need.
  const heldOut = chooseHeldOutTitles(chunks, titles, SEED ^ 0x8e1d00);
  const retainedTitles = titles.filter((t) => !heldOut.titles.has(t));
  const retainedPos = chunks.map((_, pos) => pos)
    .filter((pos) => !heldOut.titles.has((chunks[pos].title || '').trim()));
  const retainedSet = new Set(retainedPos);
  const K = Math.min(10, retainedPos.length);
  // Fused ranking for coverage's passage selection: reciprocal-rank
  // fusion of the vector top-K with the lexical (BM25) hits, same math as
  // complete/index.mjs's hybrid retrieval. coverage1 grounds the verdict
  // in "does the passage actually contain the query's words" — scoring it
  // against the vector-only top passages (the original implementation)
  // left it blind to a passage hybrid fusion already ranks first on
  // lexical strength alone, which is exactly the shape of every observed
  // false abstention: a chunk BM25 finds immediately but the embedding
  // ranks far down. Restricted to the retained set so a held-out
  // document's own title can't leak in through BM25 and inflate coverage
  // for a query calibration expects unanswerable.
  const fusedTop = (text, vectorHits) => {
    if (!lexicalIndex) return vectorHits;
    const lexHits = lexicalIndex.search(text, 5).filter((h) => retainedSet.has(h.id));
    const cut = lexHits.length ? lexHits[0].score / LEXICAL_CUTOFF : Infinity;
    const lexRank = new Map(lexHits.filter((h) => h.score >= cut).map((h, i) => [h.id, i]));
    if (lexRank.size === 0) return vectorHits;
    const byId = new Map(vectorHits.map((h) => [h.id, h]));
    for (const id of lexRank.keys()) if (!byId.has(id)) byId.set(id, { id, distance: 1 });
    return [...byId.values()]
      .map((hit, vRank) => ({
        hit,
        score: 1 / (RRF_K + vRank) + (lexRank.has(hit.id) ? 1 / (RRF_K + lexRank.get(hit.id)) : 0),
      }))
      .sort((a, b) => (b.score - a.score) || (a.hit.distance - b.hit.distance))
      .map((entry) => entry.hit);
  };
  // Signals must be computed the way the reader computes them from its own
  // search hits (complete/retrieval-abstention.mjs): d0, the rank-4 margin,
  // and the mean over the returned list.
  const signalsFor = (text, hits) => {
    const top = hits.slice(0, K);
    const d0 = top.length ? top[0].distance : 1;
    const margin = top.length > 1 ? top[Math.min(4, top.length - 1)].distance - d0 : 0;
    const mean10 = top.length ? top.reduce((s, r) => s + r.distance, 0) / top.length : 1;
    // fusedTop gets the full widened pool, not the K-sliced top: a lexical
    // top hit ranked, say, 40th by vector distance still needs its real
    // distance available to fuse correctly, which only the wider search()
    // pool (not top) carries. d0/margin/mean10 above stay on top/K, matching
    // the reader's own base-signal window exactly.
    const fused = fusedTop(text, hits);
    return {
      d0,
      margin,
      mean10,
      known_frac: knownFrac(text, bloom, bits),
      [COVERAGE_FEAT]: coverageFrac(
        text,
        fused.slice(0, COVERAGE_TOP_PASSAGES).map((h) => chunks[h.id]?.text || ''),
        isCommon,
      ),
    };
  };

  const index = await Pikelet.create({
    dim: config.embedding.dims,
    maxElements: Math.max(chunks.length, Math.ceil(chunks.length * 1.25)),
    metric: config.index.metric,
    quantized: config.index.quantized !== false,
  });
  try {
    index.addBatch(vectors);
    // Widened beyond K: fusedTop below needs a real (not synthesized)
    // vector distance for any lexical top hit to fuse it correctly, the
    // same guarantee complete/index.mjs gets by feeding lexical ids into
    // the sketch's exact rerank as extraCandidates before RRF ever runs.
    // d0/margin/mean10 still slice back down to K — this only widens the
    // pool fusedTop can search within, not the base signal window.
    const FUSION_POOL = Math.min(retainedPos.length, 200);
    const search = async (text) => index.searchFiltered(await embedQuery(text), FUSION_POOL, retainedSet);

    const rows = [];
    let droppedPositives = 0;
    const positiveTemplates = [
      ...titleQuestions(retainedTitles, Math.ceil(MAX_POSITIVES / 2), SEED ^ 0x51f15e),
      ...contentWordQuestions(chunks, Math.floor(MAX_POSITIVES / 2), SEED ^ 0xc0ffee, retainedPos),
    ];
    for (const { text, sourceTitle, sourceId } of positiveTemplates) {
      const hits = await search(text);
      // A positive counts only when retrieval verifiably lands on the source
      // document — otherwise the query is unanswerable in practice and would
      // drag the no-false-abstain floor toward zero. Title queries verify by
      // title (any chunk of the source document counts); content-word queries
      // verify by the chunk they were sampled from. Checked against the base
      // K window, not the widened FUSION_POOL search() now returns — a
      // source landing at, say, rank 80 is not "retrieval verifiably lands
      // on the source" in any sense the reader's own top-K would agree with.
      const topK = hits.slice(0, K);
      const found = sourceId !== undefined
        ? topK.some((h) => h.id === sourceId)
        : topK.some((h) => (chunks[h.id]?.title || '').trim() === sourceTitle);
      if (found) rows.push({ text, label: 1, sourceTitle, sourceId, ...signalsFor(text, hits) });
      else droppedPositives++;
    }
    const positives = rows.filter((r) => r.label === 1);
    if (positives.length < MIN_VERIFIED_POSITIVES) {
      return skip(`only ${positives.length} of ${positiveTemplates.length} templated queries verified by retrieval (need ${MIN_VERIFIED_POSITIVES})`);
    }
    const positiveD0 = positives.map((r) => r.d0).sort((a, b) => a - b);
    const positiveMedianD0 = positiveD0[Math.floor(positiveD0.length / 2)];

    let droppedForeign = 0;
    for (const { text } of titleQuestions(FOREIGN_TITLE_BANK, FOREIGN_TITLE_BANK.length, SEED ^ 0xf03e16)) {
      const sig = signalsFor(text, await search(text));
      // An off-domain query that retrieves as strongly as a median positive
      // overlaps the corpus domain; its label is untrustworthy, so it stays
      // out of the fit — but it is scored and reported (eval-only) rather
      // than discarded, so the summary states how the model treats it.
      if (sig.d0 <= positiveMedianD0) {
        droppedForeign++;
        rows.push({ text, label: 0, evalOnly: true, negativeKind: 'foreign-overlap', ...sig });
      } else rows.push({ text, label: 0, negClass: 'easy', negativeKind: 'foreign-bank', ...sig });
    }
    for (const text of syntheticGibberish(GIBBERISH_QUERIES, SEED ^ 0x9166e11, bloom, bits)) {
      rows.push({ text, label: 0, negClass: 'easy', negativeKind: 'synthetic-gibberish', ...signalsFor(text, await search(text)) });
    }

    // Hard negatives. Held-out-document questions are in-domain by
    // vocabulary and unanswerable against the retained corpus by
    // construction; recombinations are corpus words no document supports.
    // Either kind retrieving at or under the positive median d0 means the
    // topic survives elsewhere (near-duplicate trees, accidental paraphrase)
    // — the label is suspect and the row is dropped, not fit.
    let heldOutDroppedCovered = 0;
    if (heldOut.titles.size) {
      for (const { text } of titleQuestions([...heldOut.titles], HELDOUT_QUERIES, SEED ^ 0xab5e27)) {
        const sig = signalsFor(text, await search(text));
        if (sig.d0 <= positiveMedianD0) heldOutDroppedCovered++;
        else rows.push({ text, label: 0, negClass: 'hard', negativeKind: 'held-out-doc', ...sig });
      }
    }
    let recombinationDropped = 0;
    for (const text of recombinationQueries(chunks, retainedPos, RECOMBINATION_QUERIES, SEED ^ 0x5a1ad5, bloom, bits)) {
      const sig = signalsFor(text, await search(text));
      if (sig.d0 <= positiveMedianD0) recombinationDropped++;
      else rows.push({ text, label: 0, negClass: 'hard', negativeKind: 'recombination', ...sig });
    }

    const negatives = rows.filter((r) => r.label === 0 && !r.evalOnly);
    if (negatives.length < MIN_NEGATIVES) return skip(`only ${negatives.length} negatives survived the overlap drop (need ${MIN_NEGATIVES})`);
    const hardNegatives = negatives.filter((r) => r.negClass === 'hard');
    if (hardNegatives.length < MIN_HARD_NEGATIVES) {
      return skip(`only ${hardNegatives.length} hard negatives (held-out + recombination) survived verification (need ${MIN_HARD_NEGATIVES}); without them the fit cannot separate answerable from in-domain-unanswerable`);
    }

    // Weak band: retained-title questions whose source lands at rank 5..K on
    // corpora deep enough to have one. They keep the weak threshold honest so
    // adjacent content is shown with a caveat instead of hidden. search()
    // now returns a widened FUSION_POOL-sized pool (for fusedTop's benefit,
    // above); rank must still be checked against the base K window — a
    // source landing at, say, rank 40 of 200 is well outside "adjacent
    // content", not a weak match.
    if (chunks.length >= 25) {
      const weakTemplates = titleQuestions(retainedTitles, MAX_POSITIVES, SEED ^ 0x0ddba11);
      for (const { text, sourceTitle } of weakTemplates) {
        if (rows.filter((r) => r.label === -1).length >= 24) break;
        const hits = await search(text);
        const rank = hits.findIndex((h) => (chunks[h.id]?.title || '').trim() === sourceTitle) + 1;
        if (rank >= 5 && rank <= K) rows.push({ text, label: -1, ...signalsFor(text, hits) });
      }
    }

    // Fit: logistic regression on standardized signals, gradient descent —
    // the same optimizer, weighting, and regularization as the wiki
    // calibrator so the two assets stay comparable.
    const fit = rows.filter((r) => r.label >= 0 && !r.evalOnly);
    const scorerFor = (fitRows) => {
      const weights = fitRows.map((r) => (r.negativeKind === 'synthetic-gibberish' ? 0.25 : 1));
      const weightSum = weights.reduce((a, c) => a + c, 0);
      const mean = {}, std = {};
      for (const f of FEATS) {
        mean[f] = fitRows.reduce((sum, r, i) => sum + r[f] * weights[i], 0) / weightSum;
        std[f] = Math.sqrt(fitRows.reduce((sum, r, i) => sum + ((r[f] - mean[f]) ** 2) * weights[i], 0) / weightSum) || 1;
      }
      const xs = fitRows.map((r) => FEATS.map((f) => (r[f] - mean[f]) / std[f]));
      const ys = fitRows.map((r) => r.label);
      let w = FEATS.map(() => 0);
      let b = 0;
      for (let epoch = 0; epoch < 4000; epoch++) {
        const gw = FEATS.map(() => 0);
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
      return { mean, std, w, b, prob: (r) => 1 / (1 + Math.exp(-(FEATS.reduce((s, f, j) => s + ((r[f] - mean[f]) / std[f]) * w[j], b)))) };
    };
    const model = scorerFor(fit);
    const { mean, std, w, b } = model;
    for (const r of rows) r.p = model.prob(r);

    // The fit AUC is computed on the same rows the regression was fit on —
    // in-sample, and reported as such. The gate uses a deterministic 5-fold
    // cross-validation instead: refit on 4/5 of the rows, score the held-out
    // fold, pool the held-out probabilities into one AUC. The embeds and
    // searches are already done, so the extra fits cost milliseconds.
    const fitAuc = aucFor(rows.filter((r) => r.label === 1), negatives);
    const shuffled = sample(fit, fit.length, SEED ^ 0xcf01d);
    const FOLDS = 5;
    const heldout = [];
    for (let fold = 0; fold < FOLDS; fold++) {
      const test = shuffled.filter((_, i) => i % FOLDS === fold);
      const train = shuffled.filter((_, i) => i % FOLDS !== fold);
      if (!train.some((r) => r.label === 1) || !train.some((r) => r.label === 0)) continue;
      const foldModel = scorerFor(train);
      for (const r of test) heldout.push({ label: r.label, negClass: r.negClass, negativeKind: r.negativeKind, p: foldModel.prob(r) });
    }
    const heldoutPos = heldout.filter((r) => r.label === 1);
    const cvAuc = aucFor(heldoutPos, heldout.filter((r) => r.label === 0));
    // Per-class held-out AUCs. The pooled number is dominated by the easy
    // classes and stays near 1 even when the model cannot tell answerable
    // from in-domain-unanswerable (measured 0.998 pooled on a fit with that
    // exact blindness), so the gate checks the hard class on its own.
    const cvAucEasy = aucFor(heldoutPos, heldout.filter((r) => r.negClass === 'easy'));
    const cvAucHard = aucFor(heldoutPos, heldout.filter((r) => r.negClass === 'hard'));
    // Per-kind hard AUCs diagnose the two hard classes separately: the
    // grounding feature could plausibly separate held-out-doc questions yet
    // fail recombinations (retrieval can surface a recombination's source
    // chunk, granting partial coverage), and a pooled hard AUC would hide
    // which class regressed.
    const cvAucHeldOutDoc = aucFor(heldoutPos, heldout.filter((r) => r.negativeKind === 'held-out-doc'));
    const cvAucRecombination = aucFor(heldoutPos, heldout.filter((r) => r.negativeKind === 'recombination'));
    const gateAuc = cvAuc ?? fitAuc;
    if (gateAuc === null || gateAuc < MIN_AUC) {
      return skip(`${cvAuc === null ? 'fit' : 'cross-validated'} AUC ${gateAuc === null ? 'n/a' : gateAuc.toFixed(3)} < ${MIN_AUC} separating answerable from off-domain`);
    }
    if (cvAucHard !== null && cvAucHard < MIN_HARD_AUC) {
      return skip(`cross-validated hard-negative AUC ${cvAucHard.toFixed(3)} < ${MIN_HARD_AUC}: the fit cannot separate answerable from in-domain-unanswerable queries`);
    }

    // Threshold placement diverges from the wiki calibrator, which places
    // hard between the raw negative ceiling and the raw answerable/weak
    // floor. Both extremes are fragile here: the queries are auto-generated,
    // so a single junk positive that happens to verify (or a weak row that
    // retrieves with near-zero probability) drags a raw min/max to an
    // extreme, and the logistic saturates on clean separation so geometric
    // means of near-zero values collapse the thresholds. Percentiles make
    // placement robust to the tails: 5% of generated positives may fall
    // below hard and 5% of negatives above it, which measured far better
    // than protecting every outlier. Between the two bounds, hard sits a
    // quarter of the way up — the costs are asymmetric (a false abstain
    // hides results, a false weak shows them with a caveat), so the weak
    // verdict owns most of the uncertain band.
    const quantile = (values, q) => {
      const sorted = [...values].sort((a, b) => a - b);
      return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
    };
    const pos = positives.map((r) => r.p);
    const posFloor = quantile(pos, 0.05);
    // The negative ceiling is taken per class and the maximum wins: hard
    // negatives score strictly higher than the easy classes, and a pooled
    // percentile over the (larger) easy classes would put the hard threshold
    // below where in-domain-unanswerable queries actually land. The hard
    // class uses the 90th percentile because it has fewer rows.
    const easyCeil = quantile(negatives.filter((r) => r.negClass === 'easy').map((r) => r.p), 0.95);
    const hardCeil = quantile(hardNegatives.map((r) => r.p), 0.9);
    const negCeil = Math.max(easyCeil, hardCeil);
    // Weak rows scoring under 5% answerable are negatives in all but name;
    // they must not shape the weak threshold.
    const allWeakP = rows.filter((r) => r.label === -1).map((r) => r.p);
    const weakP = allWeakP.filter((p) => p > Math.max(negCeil, 0.05));
    const hardOverlap = negCeil >= posFloor;
    const hardRaw = hardOverlap ? posFloor * 0.5 : negCeil + 0.25 * (posFloor - negCeil);
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
    // "Strong" must clear the hard-negative mass. When hard negatives
    // overlap the weakest positives (hardThresholdOverlap), the hard
    // threshold stays protective of positives — a false abstain hides
    // results — but the weak threshold rises to the hard-negative ceiling,
    // so overlapped queries are shown with a caveat instead of full
    // confidence. Capped at the positive median: past that the fit is too
    // entangled for the ceiling to be meaningful, and the cvAucHard gate is
    // the real protection.
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

    // Eval-only rows (foreign-bank queries that overlap the corpus) are
    // scored by the final model and reported: how many the shipped
    // thresholds would answer is the asset's stated hard-negative exposure.
    const evalOnlyRows = rows.filter((r) => r.evalOnly);
    // A deterministic sample of the verified positives, exported as golden
    // queries the artifact can embed: each was checked above to actually
    // retrieve its source, so a reader can re-run them later (verify_pack,
    // acceptance tests) as tests stored inside the file. Spread evenly so
    // both title and content-word templates are represented.
    const goldenSample = (() => {
      const cap = 24;
      if (positives.length <= cap) return positives;
      const step = positives.length / cap;
      return Array.from({ length: cap }, (_, i) => positives[Math.floor(i * step)]);
    })();
    const goldenQueries = goldenSample.map((r) => ({
      text: r.text,
      ...(r.sourceId !== undefined ? { expectId: r.sourceId } : { expectTitle: r.sourceTitle }),
    }));
    const summary = {
      method: 'self-templates-v3',
      seed: SEED,
      searchConfig: { k: K },
      heldOut: { titles: heldOut.titles.size, chunks: heldOut.chunks },
      verifiedPositiveQueries: positives.length,
      positivesDroppedAsUnretrievable: droppedPositives,
      foreignNegativeQueries: negatives.filter((r) => r.negativeKind === 'foreign-bank').length,
      foreignKeptEvalOnlyAsSemanticOverlap: droppedForeign,
      foreignDropD0Threshold: +positiveMedianD0.toFixed(6),
      evalOnlyWouldAnswerAtHard: evalOnlyRows.filter((r) => r.p >= hard).length,
      syntheticGibberishQueries: negatives.filter((r) => r.negativeKind === 'synthetic-gibberish').length,
      syntheticGibberishFitWeight: 0.25,
      heldOutNegativeQueries: negatives.filter((r) => r.negativeKind === 'held-out-doc').length,
      heldOutDroppedAsCoveredElsewhere: heldOutDroppedCovered,
      recombinationNegativeQueries: negatives.filter((r) => r.negativeKind === 'recombination').length,
      recombinationDroppedAsSuspectedAnswerable: recombinationDropped,
      weakQueries: weakP.length,
      weakDroppedAsIndistinguishableFromNegatives: allWeakP.length - weakP.length,
      // fitAuc is in-sample (scored on the rows the regression was fit on);
      // cvAuc is the pooled held-out AUC from the deterministic 5-fold
      // cross-validation. The gates use cvAuc and cvAucHard; cvAucHard is
      // the number that detects the answers-anything-in-domain failure the
      // pooled AUC cannot see.
      fitAuc: fitAuc === null ? null : +fitAuc.toFixed(6),
      cvAuc: cvAuc === null ? null : +cvAuc.toFixed(6),
      cvAucEasy: cvAucEasy === null ? null : +cvAucEasy.toFixed(6),
      cvAucHard: cvAucHard === null ? null : +cvAucHard.toFixed(6),
      cvAucHardByKind: {
        heldOutDoc: cvAucHeldOutDoc === null ? null : +cvAucHeldOutDoc.toFixed(6),
        recombination: cvAucRecombination === null ? null : +cvAucRecombination.toFixed(6),
      },
      hardThresholdOverlap: hardOverlap,
      hardThresholdClampedFrom: hardRaw > HARD_CEILING ? Number(hardRaw.toFixed(6)) : null,
      vocab: { uniqueWords, keptWords, minCount },
      coverage: { topPassages: COVERAGE_TOP_PASSAGES, commonWords: commonWords.commonWords, commonDfCap: commonWords.dfCap },
    };
    // features[]/weights[] carry only the base topic features; the coverage
    // term rides in asset.coverage so a reader that predates it scores the
    // topic-only model against the same thresholds (conservative — it lacks
    // a term that is positive for answerable queries) instead of feeding an
    // unknown feature name NaN into the logistic.
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
        mean: mean[COVERAGE_FEAT],
        std: std[COVERAGE_FEAT],
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
  } finally {
    index.dispose();
  }
}
