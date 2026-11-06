# Calibration: design notes

`packages/pikelet/src/calibrate.mjs` fits the model behind a pack's
`matchQuality` verdicts. This page holds the reasoning and measurements
behind that design, including the approaches that were tried and dropped,
so the source comments can stay focused on what the code does now. Release-
visible changes are recorded in the [CHANGELOG](../CHANGELOG.md).

## What is fit today (v6, `self-templates-v6`)

- **Signals.** Four topic features (`d0`, `margin`, `mean10`, `known_frac`)
  plus `coverage1`, the weighted fraction of the query's content words found
  in the best of the top retrieved passages. All are computed from the
  reader's own retrieval (`__calibrationEvidence` in
  `packages/pikelet-wasm/complete/index.mjs`), so the fit sees exactly the
  evidence the reader scores at serve time.
- **Target: relevance, not answerability.** Positives are verified on-topic
  template queries. Negatives are off-topic queries: the foreign bank plus
  gibberish. "none" means this pack does not cover the topic; "strong" means
  it does.
- **Hard threshold:** the 95th percentile of off-topic probability.
- **Hard negatives** (ablation, entity-swap, held-out-document) are still
  generated, but only to report `cvAucHard`, i.e. how far "strong" can be
  read as "answered". They do not shape the fit, the thresholds, or the gates.
- **Validation:** a deterministic 5-fold cross-validated AUC gate, a check
  that the `d0` weight's sign makes sense, and, when the corpus ships
  human-written questions (`runtime.calibrationQueries`), a held-out
  real-query abstention check.

## Why v6 changed the target to relevance

v6 was measured against real questions: 40 hand-written ones on the docs
pack (answerable, missing-detail, near-miss, false-premise) and the Veyra
registry's labelled set. On those, the v5 answerability fit separated
answerable from in-domain-unanswerable at AUC 0.61. Its hard threshold sat
among the hard negatives and withheld about half of the genuinely
answerable questions.

The five retrieval signals can tell whether a query is about something the
corpus covers. They cannot tell whether the covering passage holds the
specific fact. That is a reading task, and the agent reading the passages
can do it. So v6 fits relevance and leaves fact-checking to the reader.

## The v5 principle: one generator family for both labels

The model must never be able to tell positives from negatives by looking at
the query text alone; only the retrieval response may carry the label.
Every version before v5 broke that rule, and the symptom was always the
same: aggregate AUC looked fine while real questions were wrongly abstained
on. Inspecting real fitted weights and real query outcomes traced it to
five compounding, structural causes:

1. **Positives were built from the target passage's own words** (a title,
   or a run of the chunk's tokens), so `coverage1` was ~1.0 for every
   positive by construction. The fit gave `coverage1` the largest weight
   because it was the cleanest separator, not because high coverage makes a
   query answerable.
2. **Hard negatives landed in a generator-specific coverage band.**
   Cross-chunk recombination sat near 0.5 because of how it combined two
   chunks' words, and later ablation attempts paired only some positive
   classes. The fit learned "coverage in this band means unanswerable",
   which is exactly where a real paraphrase of a real answer lands.
3. **Positives and hard negatives came from different generators.** A
   "low-coverage positive" made by swapping words still inherited half its
   words verbatim. A "held-out-document" negative used the positives'
   title-question template on a different, excluded title pool. Either way,
   query-side signals could carry the label, so the fit could partly learn
   which generator produced a query.
4. **The validation metric could not catch it.** The "paraphrase-stress
   abstention rate" was measured on positives already inside the fit:
   training accuracy under a different name.
5. **Threshold placement was built on the contaminated feature space**, even
   once nominally anchored to the negative side.

v5's fix was one generator family for both labels. Positives are verified
title or content-word templates. Every verified positive gets exactly one
paired ablation negative: the identical query text, scored with the answer
excluded from the index. A paired positive and negative share
byte-identical text, `known_frac`, length and lexical shape, so only what
retrieval returns can differ between them. Human-written queries, held out
of the fit entirely, became the only real validation, because they are the
only rows the training generator did not produce.

v5 fit in two stages. Stage 1 gated off-domain and gibberish negatives on
`known_frac` + `d0`; stage 2 fit positives against the paired ablation
negatives. The hard threshold was the 90th percentile of hard-negative
probability. Once v6 made off-topic queries the target, the two stages
became the same fit, so there is now one.

## Probe classes

- **Content-word positives in question frames.** Title-templated questions
  retrieve too easily: every positive lands near the corpus, and the hard
  threshold creeps up until honest paraphrases abstain. Bare keyword runs
  are too narrow a phrasing distribution. On a 94-passage synthetic
  registry, naturally phrased supported questions landed below a hard
  threshold of 0.68. Rotating the same content words through question frames
  keeps the retrieval target identical while widening the phrasing.
- **Encoder-guided substitution positives.** Every template positive copies
  its words from the target passage, so the base class never shows the fit
  an answerable query with partial lexical overlap. Trained on that class
  alone, the fit weighted coverage about 4× `d0`. Substitution swaps some
  content words for nearby corpus vocabulary that is absent from the source
  passage, then re-verifies by retrieval. `SUBSTITUTION_MIN_COS = 0.55` was
  the point past which swaps stayed recognisably related. At 0.35, nonsense
  like "fourth location sits during layout" got through and still retrieved
  its source on a small corpus.
- **Ablation negatives** replaced cross-chunk recombination (see v5 cause 2).
  - **Drop rule.** A negative is dropped as "still grounded" when its
    distance stays close to its paired positive's own `d0`, or its coverage
    stays above an absolute floor. A corpus-wide median `d0` was too loose:
    on a real corpus, half of all ablation negatives slid under it while
    still scoring coverage near 1.0, i.e. real duplicates or
    cross-references.
  - **Relative-coverage branch.** It applies only when the paired positive
    was itself well grounded. Without that gate, a positive at coverage 0.2
    made "stayed above 75% of it" nearly always true. On a real corpus that
    alone dropped every ablation negative.
  - **Window size.** The post-ablation pool need not stay at the original K.
    On corpora small enough that K was capped to the chunk count, requiring
    it skipped every positive: an 8-chunk fixture produced 0 ablation
    negatives out of 56 verified positives.
- **Entity-swap negatives.** Ablation assumes that removing the source
  leaves the query unanswerable. That fails on thematically uniform content
  such as a novel's chapters. On Pride and Prejudice only 3 of 101 verified
  positives survived ablation, under v5's hard-negative minimum, and the pack
  shipped unscored. Swapping in a proper noun that never co-occurs with the
  original avoids the uniqueness assumption. Proper nouns are detected
  without a tagger: capitalised somewhere, never lowercase anywhere. The
  entity lookup is a token-set membership test. A regex per entity per call
  was pathologically slow with Pride and Prejudice's 200+ names.
- **Held-out-document negatives.** Ablation replaced this class, and it was
  later reinstated. Ablation's drop rule discards any negative whose coverage
  stays high, and on entity-dense corpora that is exactly the "entity named,
  fact absent" negative. On the 94-record Veyra registry, 75 of 94 ablations
  were dropped, the fitted `d0` weight came out positive, and harness false
  answers rose from 8 to 19 against the v3 calibrator that still had this
  class. These rows have no positive twin, but the model sees only retrieval
  signals, never query text.

## Coverage details

- **Heading words count at reduced credit.** Full credit is free coverage:
  a title-templated positive's words *are* the heading. Zero credit (the
  first attempt) scored a perfect rank-1 heading match as "none", because
  the fit weighs coverage several times more than `d0`.
- **Common words are down-weighted rather than zeroed.** Zeroing words with
  chunk-level document frequency above ~5% punished honest paraphrases
  whose only grounded words were common ones.
- **Scaffolding stopwords.** "make", "like", "other" and "only" are
  deliberately *not* stopped: they are ordinary content words in technical
  prose.
- **Suffix stripping.** Only the suffix families measured to matter on real
  technical-doc word pairs are stripped (quantize/quantization,
  authenticate/authentication, configure/configuration,
  validate/validation). Before that, only plurals were forgiven.
- **Possessives.** "What is X's Y?" lost X from coverage until a trailing
  `'s` or `'` was stripped from both query and passage words.
- **Numerals** are exempt from the minimum word length. Without that,
  "Chamber 4" matched any chamber's record.

## Thresholds and gates

- **Real-query abstention gate (`MAX_PARAPHRASE_ABSTENTION_RATE = 1/3`).**
  On dutch-grammar-src, a fit that cleared every other gate still abstained
  on 50% of its held-out substitution positives.
- **Minimum sample (`MIN_PARAPHRASE_STRESS_SAMPLE = 8`).** A pack with 0%
  real-world false abstention (35 hand-written questions) failed the gate on
  one bad row out of two, because n=2 can only read 0%, 50% or 100%.
- **`OFFTOPIC_HARD_QUANTILE = 0.95`**, up from v5's 0.9. The off-topic pool
  is larger and cleaner than the hard-negative pool was, and a "none" verdict
  hides results.
- **Hard threshold set from the negative side.** Placing it from the
  positive side (`posFloor * 0.5`, or a blend toward the positive floor) let
  generated positives at p≈0.99 push the bar high enough to abstain on
  queries the model scored as more likely relevant than not.
- **`HARD_CEILING` and `MIN_WEAK_BAND`.** Perfectly separable probes saturate
  the logistic. Without the band floor, the weak band measured 0.015 wide on
  a synthetic corpus.
- **`d0` weight sign check.** A positive `d0` weight was the measured
  symptom of a negative pool whose retrieval barely differed from its
  positives': the fit then answered anything that named a corpus entity, and
  the AUC gates did not catch it.

## Approaches tried and rejected

These are kept so they are not re-attempted without a new idea.

- **`grounding1 = max(coverage1, maxSim1)`**, tried twice and reverted both
  times.
  1. Raw word-vector cosine similarity has no meaningful zero: unrelated
     corpus words score ~0.45–0.55. Blending it in raised every row's floor
     and pushed the fitted threshold up sharply, causing widespread false
     abstention in production.
  2. Rescaling against a per-corpus unrelated-word baseline
     (`estimateMaxSimBaseline`) and damping with a power curve
     (`MAXSIM_GAMMA`) fixed the floor. Even so, real probe questions got
     worse: more false abstention on answerable questions and more
     false-confidence verdicts on unsupported ones than plain `coverage1`.
     Without the damping, Pride and Prejudice's held-out AUC fell from ~0.86
     to 0.849, under the 0.85 gate, reproducibly.

  `maxSim1` is still computed and reported (`maxSimVsCoverage`) for
  comparison. The reader's `useMaxSim` path was removed entirely: it cost
  200–400 per-word encoder calls per query, turning a 30–80 ms search into
  1.5–4.6 s, and nothing in the response said why. It was removed rather
  than left behind a flag: an external asset could silently reactivate
  dead code, and both attempts failed for structural reasons, not tuning.
- **Paraphrase agreement.** Search several independent paraphrases of a
  question and score how often rank 1 returns to the source. The base rate
  on answerable questions was good (0.83 on internal-docs). On real Pride
  and Prejudice questions with hand-varied wording, though, answerable and
  unanswerable questions both drifted at ~65–75%, probably because many
  similar chapters compete for rank 1.
- **Proximity grounding.** Require the query's content words to co-occur
  within a 12-token window. Paraphrasing defeated it (paraphrase separation
  AUC 1.0, the worst possible). On Pride and Prejudice it did not separate
  answerable from unanswerable, because flowing prose puts unrelated words
  within 12 tokens constantly. Fixing it would need relation extraction,
  which this codebase deliberately avoids.
- **Passage-direction grounding.** Compare the query embedding to each
  passage's top principal component. On internal-docs that component
  explained only 8–13% of variance, and standalone hard-negative separation
  AUC was 0.61 against `coverage1`'s 0.98. It might behave differently on
  much longer passages; that is untested.
- **The "paraphrase-stress" metric.** It was scored on positives already in
  the fit, so it could not detect the fit learning the generator. There is
  no synthetic stand-in for human-written validation questions.
