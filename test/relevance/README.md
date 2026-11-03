# Relevance regression suite

Labeled query sets with expected sections, run against compiled artifacts
with `scripts/bakeoff-retrieval.mjs`. This is the permanent record of what
"retrieval works" means: when ranking, chunking, calibration, or the
encoder changes, rerun the suite and compare against the baselines
committed beside each query set.

## Query sets

- `nodeapi-queries.json` — nodejs.org/api, crawled with
  `compile --source https://nodejs.org/api/ --max-pages 30`. A real,
  substantial docsite: ~2,000 sections, dense API identifiers, authored
  anchor ids. Mixes natural-language questions, API identifiers
  (`fs.readFileSync`), error codes, CLI flags, and HTTP-ish exact
  lookups.
- `nodeapi-baseline.json` — the metrics recorded when the set was
  authored, with the artifact identity they were measured against.

## Abstention sets

Labeled question sets for `scripts/calibration-harness.mjs`, which reports
false-abstention and false-answer rates per question class. Each row is
answerable unless its class is unsupported/offdomain/unanswerable/nearmiss
or its answer is null; answerable rows carry near-verbatim evidence quotes
the harness checks the returned text against.

- `pride-prejudice-queries.json` — 174 questions over the 61-chapter novel
  (corpus: `local-packs/pride-prejudice-src/`, not tracked); 157 direct,
  12 unanswerable, 5 near-miss. `pride-prejudice-questions.txt` is the bare
  source list the labeled set was built from.
- `internal-docs-queries.json` — 48 questions over a 4-file, 45-record docs
  corpus (`local-packs/internal-docs/corpus/`, not tracked): 24 paraphrase
  (reworded to avoid the section's own terms), 6 direct, 12 unsupported,
  6 near-miss.
- `veyra-baseline.json` — harness result for the Station Veyra synthetic
  registry. Its 90-question set is generated, not tracked: `node
  examples/one-file-search/web/veyra-corpus/gen.mjs` writes
  `questions.json` beside the corpus it generates (deterministic; the
  generator has no random source).

```bash
node scripts/calibration-harness.mjs <pack.pikelet> test/relevance/internal-docs-queries.json
node scripts/calibration-harness.mjs <pack.pikelet> \
  examples/one-file-search/web/veyra-corpus/questions.json \
  --baseline test/relevance/veyra-baseline.json
```

Not part of `npm test` (needs the local corpora above). Rerun on Veyra and
one real corpus before and after any change to `packages/pikelet/src/calibrate.mjs`
or `packages/pikelet-wasm/complete/retrieval-abstention.mjs`.

### Retrieval-side reports

The same sets measure retrieval, not just the verdict. Question loading and
evidence matching live in `scripts/lib/relevance-sets.mjs` so every tool
agrees on what "the evidence is in this result" means: `{ quote }` evidence
is checked against result text, string evidence (Veyra's fact-file ids)
against the record's `sourcePath`/`url`/`anchor` stem.

```bash
# evidence rank per retrieval mode, misses bucketed by what could fix them
node scripts/calibration-harness.mjs <pack.pikelet> <queries.json> \
  --rank-depth 50 [--rank-modes hybrid,vector,lexical] [--rank-report out.json]

# replay alternative vector/lexical fusion rules offline, judged on evidence rank
node scripts/fusion-sim.mjs <pack.pikelet> <queries.json> [--depth 50] [--report out.json]
```

`--rank-depth` reports, per class and mode, how many answerable questions
have their evidence at @1, @3 and @depth, then buckets the hybrid misses:
`gap` (no mode reaches the evidence within the depth — only the index
contents or the encoder can move it), `range` (reached, but deeper than 3 —
a reranker's territory), `demotion` (vector or lexical has it in the top 3
and fusion pushed it out — a fusion-logic fix). The top-3 decision uses the
serving path (k=3) so a deeper search, which widens the sketch's candidate
pool, cannot turn a served rank 3 into a reported 4. Use a depth smaller
than the corpus, or `gap` cannot occur.

`fusion-sim.mjs` fetches the reader's vector, lexical and hybrid lists once
per question and scores each fusion rule as a pure function of the first
two; its `defaults` rule (the reader's `FUSION_DEFAULTS`) must reproduce
the shipped hybrid ranks (it reports
how many it matched — the residue is phrase pinning, which it omits). Add
a candidate rule there and read its @1/@3 against `shipped` and `oracle`
(the better of vector and lexical per question) before touching the
reader.

## Running

```bash
# recompile the corpus (network; page content drifts over time)
npx pikelet compile --source https://nodejs.org/api/ \
  --max-pages 30 --out nodeapi.pikelet

# measure all three retrieval modes
node scripts/bakeoff-retrieval.mjs nodeapi.pikelet \
  test/relevance/nodeapi-queries.json
```

Not part of `npm test`: it needs the network and ~20 minutes of local
embedding. Treat a drop against the committed baseline as a regression to
explain, not noise to re-record — the corpus can drift when the site
publishes new docs, so re-author expectations only when a target section
verifiably moved or was renamed upstream.
