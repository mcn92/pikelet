# BEIR retrieval benchmark

Compact, reproducible BEIR subset for measuring how much retrieval quality
Pikelet loses (if any) moving from ordinary float MiniLM retrieval to
Pikelet's compact, staged, self-contained artifact — and, as of the E-H
configurations below, for measuring what a real `.pikelet` artifact actually
serves. **A-D are dense-only** (vector search alone, never how a served
query is actually answered); **E is hybrid RRF, the production default**
(`complete/index.mjs`'s `query()` fuses BM25 with the vector rerank unless a
caller opts out), and **F is lexical-only** (`retrieval: 'lexical'`). A
benchmark that only ever measured A-D was reporting a mode the reader almost
never runs in practice.

The ladder now covers eight points on SciFact, NFCorpus, and ArguAna (FiQA
has only config A so far; TREC-COVID has not been run):

- **A** upstream MiniLM float exhaustive (reference baseline, not Pikelet)
- **B** Pikelet encoder + float32 exhaustive
- **C** Pikelet encoder + affine-u8 exhaustive
- **D** Pikelet encoder + affine-u8 sketch artifact (dense only — the real
  `PikeletSketchArtifact`/HNSW-equivalent search, but vector-only)
- **E** Pikelet encoder + affine-u8 sketch artifact + BM25, hybrid RRF — **the
  production default**, same fusion math as `complete/index.mjs` exactly
  (RRF_K=60, BM25 candidates within a factor of 1.5 of the top lexical score
  join the exact rerank as `extraCandidates` before RRF runs)
- **F** BM25 lexical only (`retrieval: 'lexical'`)
- **G** Arctic-XS encoder (Snowflake/snowflake-arctic-embed-xs, CLS pooling)
  + affine-u8 sketch artifact, dense only — directly parallel to D, isolating
  the encoder question on identical retrieval logic (`G-noprefix` is the
  same run without the query prefix — see the ablation below)
- **H** Arctic-XS encoder + affine-u8 sketch artifact + BM25, hybrid RRF —
  parallel to E, so an encoder decision is checked against what actually
  ships (hybrid), not dense-only in isolation
- **I** BGE-small-en-v1.5 — the encoder `embedding.mode: "workers-ai"` uses
  at build time, which is the CLI's **default** mode. Scored through
  query-B/C/D/E with `--vectors workersai`, so the retrieval logic is
  identical to B-E and only the encoder changes. **This measures the model,
  not the Workers AI call**; see the note below. The rows below were encoded
  with transformers.js's `Xenova/bge-small-en-v1.5` ONNX export, which the
  CLI used before 0.12.0; `encode-workers-ai.mjs` now encodes with the CLI's
  own u8 BGE kernel, whose passage vectors are within 0.997-0.999 cosine of
  that export's (SciFact nDCG@10 0.6991 against 0.6992, per the 0.12.0
  CHANGELOG entry).

See `config.json` for the frozen dataset list committed ahead of any
results.

## Layout

```
benchmarks/beir/
  README.md
  config.json          frozen params (datasets, k, efSearch, seed, ...) — committed before tuning
  download.py           fetch official BEIR corpus/queries/qrels -> cache/<dataset>/
  convert.mjs            BEIR corpus.jsonl -> Pikelet source records + id map
  embed.py                float32 MiniLM encode (config A's encoder) -> vectors-{corpus,queries}-float32.f32
  encode-pikelet.mjs       Pikelet's real inline WASM MiniLM, corpus via the production worker-pool
                           path (embedChunksWithInlineTransformer), queries via single embed() calls
                           (matches the runtime query path) -> vectors-{corpus,queries}-pikelet.f32
  quantize.mjs             Pikelet's row-wise affine-u8 quantization, reimplemented (no engine readback API — see below)
  query-A.mjs              config A: brute-force cosine over float32 vectors
  query-B.mjs              config B: brute-force cosine over Pikelet-encoder float32 vectors
  query-C.mjs              config C: Pikelet-encoder vectors + exhaustive affine-u8 search
  query-D.mjs              config D: Pikelet-encoder vectors + real sketch artifact search (dense only)
  query-E.mjs              config E: sketch artifact + BM25, hybrid RRF (production default) — builds an
                           in-memory lexical index (complete/builder.mjs buildLexicalSegment) and a real
                           PikeletSketchArtifact from index.export(), fuses exactly as complete/index.mjs does
  query-F.mjs              config F: BM25 lexical only (retrieval: 'lexical')
  encode-workers-ai.mjs    config I: BGE-small-en-v1.5 through the CLI's u8 kernel — the encoder
                           `embedding.mode: "workers-ai"` (the CLI's DEFAULT mode) uses at build
                           time -> vectors-{corpus,queries}-workersai.f32. Queries carry the
                           asymmetric BGE prefix, passages do not. Measures the MODEL only, never
                           a live Workers AI call (see the config I note below)
  encode-pikelet-arctic.mjs   Arctic-XS (Snowflake/snowflake-arctic-embed-xs) through the same production
                              embedChunksWithInlineTransformer/createInlineTransformerEmbedder paths as
                              encode-pikelet.mjs, driven by inlineEncoderDeclaration (cls pooling, query
                              prefix) -> vectors-{corpus,queries}-arctic.f32
  ablate-query-prefix.mjs  re-embeds ONLY the test queries without the query prefix, reusing the
                           already-encoded corpus vectors -> vectors-queries-arctic-noprefix.f32
  query-G.mjs              config G (or G-noprefix with --no-prefix): Arctic-XS + sketch artifact, dense only
  query-H.mjs              config H: Arctic-XS + sketch artifact + BM25, hybrid RRF
  arctic-xs/               exported Arctic-XS weight blob + vocab (export_encoder_blob.py --model
                           Snowflake/snowflake-arctic-embed-xs --pooling cls); kept separate from
                           examples/one-file-search/encoder-spike/real/, which stays the shared MiniLM blob
  score.py                 score a run against official qrels (nDCG@10, Recall@10/100) via pytrec_eval
  conformance/             golden-vector test proving quantize.mjs matches the real engine exactly (see below)
  cache/                  downloaded datasets (gitignored)
  work/                   converted records, id maps, embedded vectors, raw runs (gitignored)
  results/                scored JSON output, one file per dataset+configuration (gitignored)
```

## Why the official host, not the `beir` pip package's default

The `beir` package's bundled `util.py` defaults to a mirror
(`public.ucllm.nrp-nautilus.io`) that currently fails TLS hostname
verification. `download.py` pulls from the original
`public.ukp.informatik.tu-darmstadt.de` release host directly instead —
same files, same official BEIR release.

## Usage (per dataset)

```bash
python3.11 benchmarks/beir/download.py <dataset>
node benchmarks/beir/convert.mjs <dataset>

python3.11 benchmarks/beir/embed.py <dataset> corpus
python3.11 benchmarks/beir/embed.py <dataset> queries
node benchmarks/beir/query-A.mjs <dataset>
python3.11 benchmarks/beir/score.py <dataset> A

node benchmarks/beir/encode-pikelet.mjs <dataset>   # production worker-pool path for
                                                # corpus (~7 workers on 8 cores),
                                                # single embed() calls for queries
node benchmarks/beir/query-B.mjs <dataset>
python3.11 benchmarks/beir/score.py <dataset> B
node benchmarks/beir/query-C.mjs <dataset>
python3.11 benchmarks/beir/score.py <dataset> C
node benchmarks/beir/query-D.mjs <dataset>
python3.11 benchmarks/beir/score.py <dataset> D

# E, F: what a real .pikelet artifact actually serves (hybrid default,
# lexical-only). Both reuse the vectors-*-pikelet.f32 encode-pikelet.mjs
# already produced above, plus records.jsonl for BM25 text — no re-encoding.
node benchmarks/beir/query-E.mjs <dataset>
python3.11 benchmarks/beir/score.py <dataset> E
node benchmarks/beir/query-E.mjs <dataset> --fusion '{"lexicalWeight":1,"guardMargin":0}' --label oldrule   # fusion ablation -> run-E-oldrule.json
python3.11 benchmarks/beir/score.py <dataset> E-oldrule
node benchmarks/beir/query-F.mjs <dataset>
python3.11 benchmarks/beir/score.py <dataset> F

# I: the CLI's default embedding.mode (workers-ai / BGE-small), on the same
# retrieval logic as B-D. --vectors workersai writes run-<cfg>-workersai.json,
# so the MiniLM baseline rows are never overwritten.
node benchmarks/beir/encode-workers-ai.mjs <dataset>
node benchmarks/beir/encode-workers-ai.mjs <dataset> --no-prefix   # query-prefix ablation
node benchmarks/beir/query-B.mjs <dataset> --vectors workersai
node benchmarks/beir/query-C.mjs <dataset> --vectors workersai
node benchmarks/beir/query-D.mjs <dataset> --vectors workersai
python3.11 benchmarks/beir/score.py <dataset> B-workersai
python3.11 benchmarks/beir/score.py <dataset> C-workersai
python3.11 benchmarks/beir/score.py <dataset> D-workersai

# G, H: the encoder question, on identical retrieval logic to D/E.
node benchmarks/beir/encode-pikelet-arctic.mjs <dataset>              # corpus + queries, WITH the query prefix
node benchmarks/beir/ablate-query-prefix.mjs <dataset>                # queries only, WITHOUT the prefix (ablation)
node benchmarks/beir/query-G.mjs <dataset>                            # Arctic-XS, dense only
node benchmarks/beir/query-G.mjs <dataset> --no-prefix                # same, no query prefix
node benchmarks/beir/query-H.mjs <dataset>                            # Arctic-XS, hybrid RRF
python3.11 benchmarks/beir/score.py <dataset> G
python3.11 benchmarks/beir/score.py <dataset> G-noprefix
python3.11 benchmarks/beir/score.py <dataset> H

node benchmarks/beir/conformance/test-quantization.mjs   # golden-vector proof quantize.mjs == the real engine, run once
```

## Results so far

| Dataset | Config | nDCG@10 | Recall@10 | Recall@100 | median ms/query |
|---|---|---|---|---|---|
| SciFact (300q) | A upstream float exhaustive | 0.6451 | 0.7833 | 0.9250 | 3.6 |
| SciFact | B Pikelet encoder / float exhaustive | 0.6554 | 0.8086 | 0.9317 | 3.7 |
| SciFact | C Pikelet encoder / affine-u8 exhaustive | 0.6560 | 0.8086 | 0.9317 | 3.7 |
| SciFact | D Pikelet encoder / affine-u8, dense only | 0.6562 | 0.8086 | 0.9317 | 0.2 |
| SciFact | **E MiniLM / hybrid (production default)** | **0.7011** | **0.8466** | 0.9400 | 7.6 |
| SciFact | F BM25 lexical only | 0.6591 | 0.7816 | 0.8726 | 0.7 |
| SciFact | G Arctic-XS / dense only | 0.6427 | 0.7700 | 0.8620 | 7.8 |
| SciFact | G-noprefix (Arctic-XS, no query prefix) | 0.4791 | 0.6154 | 0.7363 | 7.7 |
| SciFact | H Arctic-XS / hybrid | 0.6879 | 0.8077 | 0.9103 | 8.4 |
| SciFact | I BGE-small / float exhaustive | 0.7040 | 0.8229 | 0.9467 | 3.8 |
| SciFact | I BGE-small / affine-u8 exhaustive | 0.7051 | 0.8262 | 0.9467 | 3.7 |
| SciFact | I BGE-small / affine-u8, dense only | 0.7044 | 0.8246 | 0.9400 | 0.2 |
| SciFact | **I BGE-small / hybrid** | **0.7298** | **0.8496** | 0.9417 | 8.2 |
| NFCorpus (323q) | A upstream float exhaustive | 0.3159 | 0.1550 | 0.3115 | 2.6 |
| NFCorpus | B Pikelet encoder / float exhaustive | 0.3173 | 0.1562 | 0.3088 | 2.8 |
| NFCorpus | C Pikelet encoder / affine-u8 exhaustive | 0.3174 | 0.1563 | 0.3089 | 2.8 |
| NFCorpus | D Pikelet encoder / affine-u8, dense only | 0.3164 | 0.1572 | 0.3080 | 0.2 |
| NFCorpus | **E MiniLM / hybrid (production default)** | **0.3391** | 0.1655 | 0.3100 | 4.9 |
| NFCorpus | F BM25 lexical only | 0.3063 | 0.1435 | 0.2337 | 0.1 |
| NFCorpus | G Arctic-XS / dense only | 0.3085 | 0.1462 | 0.2607 | 5.9 |
| NFCorpus | G-noprefix (Arctic-XS, no query prefix) | 0.1546 | 0.0777 | 0.1818 | 5.5 |
| NFCorpus | H Arctic-XS / hybrid | 0.3283 | 0.1558 | 0.2729 | 5.6 |
| ArguAna (1406q) | A upstream float exhaustive | 0.5017 | 0.7909 | 0.9772 | 7.8 |
| ArguAna | B Pikelet encoder / float exhaustive | 0.4875 | 0.7824 | 0.9772 | 6.1 |
| ArguAna | C Pikelet encoder / affine-u8 exhaustive | 0.4889 | 0.7831 | 0.9772 | 6.3 |
| ArguAna | D Pikelet encoder / affine-u8, dense only | 0.4889 | 0.7831 | 0.9772 | 0.1 |
| ArguAna | **E MiniLM / hybrid (production default)** | **0.4907** | 0.7809 | 0.9772 | 16.7 |
| ArguAna | F BM25 lexical only | 0.4351 | 0.7098 | 0.9324 | 6.0 |
| ArguAna | G Arctic-XS / dense only | 0.5166 | 0.7845 | 0.9566 | 10.5 |
| ArguAna | G-noprefix (Arctic-XS, no query prefix) | 0.5217 | 0.7817 | 0.9381 | 10.4 |
| ArguAna | **H Arctic-XS / hybrid** | **0.5188** | 0.7838 | 0.9566 | 17.5 |
| FiQA (648q) | A upstream float exhaustive | 0.3687 | 0.4413 | 0.7061 | 55.1 |
| FiQA | B Pikelet encoder / float exhaustive | 0.3608 | 0.4313 | 0.7067 | 40.0 |
| FiQA | C Pikelet encoder / affine-u8 exhaustive | 0.3618 | 0.4329 | 0.7073 | 40.1 |
| FiQA | D Pikelet encoder / affine-u8, dense only | 0.3569 | 0.4268 | 0.6993 | 0.4 |
| FiQA | **E MiniLM / hybrid (production default)** | **0.3737** | **0.4482** | 0.7105 | 88.9 |
| FiQA | E-oldrule (equal-weight RRF, no guard — the rule before 2026-09-23) | 0.3102 | 0.3554 | 0.7105 | 79.3 |
| FiQA | F BM25 lexical only | 0.2319 | 0.2905 | 0.5201 | 9.5 |
| FiQA | G / H Arctic-XS | — | not encoded | | |
| TREC-COVID | not run | — | — | — | — |

**The ArguAna rows were rescored on 2026-10-09** with `score.py` dropping a
hit whose id is the query's own, as BEIR's official evaluation does
(`EvaluateRetrieval.evaluate`, `ignore_identical_ids=True` by default).
ArguAna's queries are arguments that are also corpus documents, and 1298 of
the 1406 queries retrieved themselves, which the old scoring counted as a
miss in the first position: every ArguAna nDCG@10 above was about 0.13 low
(A read 0.3698; the published all-MiniLM-L6-v2 figure is about 0.50, which
A now matches at 0.5017). Recall@100 is unchanged, since each run holds 100
hits. No SciFact, NFCorpus or FiQA run retrieves a document with its query's
id, so those rows are identical under either rule. ArguAna figures quoted
below from before that date (the previous-RRF numbers and the 128 -> 512
delta) were scored the old way.

Until 2026-09-23 the E and H rows were scored in vector-distance order:
`score.py` ranks by the run's `score` field, and `query-E.mjs`/`query-H.mjs`
wrote `1 - distance` like the dense-only configurations, so their rows
equalled D/G within the top-100 set. Both scripts now write the fused rank
as the score. The E rows above are re-measured that way with the current
fusion defaults (`FUSION_DEFAULTS` in `pikelet-wasm/complete`: lexical rank
weight 0.5, vector-margin guard 0.05); under the previous equal-weight RRF
the fused-rank numbers were SciFact 0.6941 / 0.8267, NFCorpus 0.3345 /
0.1565, ArguAna 0.3494 / 0.7368. The H rows are re-measured the same way.

`encode-pikelet.mjs` declared `maxTokens: 128` until 2026-09-24, so every
B-F row measured before then was encoded with 128-token windows and
windowed mean pooling, while the CLI has shipped a 512-token window since
2026-09-11. The SciFact, NFCorpus and ArguAna B-F rows above are re-encoded
at 512 (the H rows and the FiQA rows are still the 128-window vectors).
**The NFCorpus B-E rows were left at their 128-window values in this table
until 2026-09-27**, although the re-encode had rewritten
`results/nfcorpus-{B..F}.json` on 2026-09-23: `e4b826b` refreshed only the
SciFact and ArguAna rows. They now match the files on disk. The correction
raised the published figures (D 0.3049 -> 0.3164, E 0.3328 -> 0.3391), so the
512-window effect on NFCorpus is smaller than the -0.005 stated below, not
larger.
128 -> 512 moved nDCG@10 by +0.008 on ArguAna and +0.002..+0.008 on SciFact
(long passages, previously 2-3 windows each). The NFCorpus figure was reported
as -0.005 from the stale rows; against the corrected ones the 512 encode is
slightly BETTER there too, so the claim that NFCorpus regressed does not
survive the correction. Recall@100 dipped
0.6-1.0 points on the two long-document sets while recall@10 rose. Corpus
encoding costs 5-15% more (one long pass instead of several short ones).
FiQA was encoded after the change and is the one dataset the rule was never
tuned or re-scored against: `E-oldrule` replays the same vectors and BM25
hits under the previous rule (`node benchmarks/beir/query-E.mjs fiqa
--fusion '{"lexicalWeight":1,"guardMargin":0}' --label oldrule`), where
hybrid scored below dense-only D; the shipped rule is 6.3 nDCG points above
it and 1.7 above D. FiQA's E latency is the raw sketch reader's JS scan over
57,638 rows plus BM25 over 80,910 terms; the complete reader stages the WASM
scanner for sketches this size, which the ladder does not exercise.

All three completed datasets independently validate the pipeline against
published literature: SciFact nDCG@10 ~0.64-0.65, NFCorpus ~0.31-0.32,
ArguAna ~0.50 are the expected values for MiniLM-L6-v2-class encoders
on these tasks — this is not a benchmark that happens to favor Pikelet, it
reproduces known-hard and known-easy cases correctly.

**Per-dataset A→B (encoder-quantization) deltas differ meaningfully:**
SciFact **+0.0103** (net positive), NFCorpus **+0.0014** (net positive),
ArguAna **-0.0142** (still the only negative here, and outside noise at
n=1406). All three were restated on 2026-09-27 against the scored runs on
disk; they had read +0.0061, -0.0005 and -0.0192, computed from B rows the
512-token re-encode had already superseded, and ArguAna's read -0.0087 until
the 2026-10-09 rescoring. The shape of the finding survives — ArguAna is the
one dataset where quantizing the encoder costs anything — and NFCorpus moved
from straddling zero to net positive. B→C (corpus quantization) and C→D
(dense-only sketch artifact search) stay small and roughly flat relative to
B on all three datasets — the loss so far, where it exists, concentrates in
the encoder step, not the storage or search-approximation steps. Do not
generalize the "quantization is free" framing past what's actually
measured; ArguAna is evidence against an unqualified version of that claim.

Latency: exhaustive search (A/B/C) runs a few ms/query at these corpus
sizes; the dense-only sketch artifact (D) runs at 0.1-0.2ms/query, 20-70x
faster, at equal or near-equal measured quality on every dataset run so
far. Hybrid (E/H) costs a few ms more than dense-only per query — the BM25
lookup plus a wider exact-rerank candidate set — still well under exhaustive
search's cost and nowhere near a latency-sensitive regime at these corpus
sizes.

### Hybrid RRF vs. dense only (E vs. D): a real gain since the 2026-09-23 fusion change

**This section's original premise no longer holds.** It was written when E
reproduced D almost exactly, and that is what the old heading claimed.
Against the runs on disk, E now *beats* D by **+0.0449** on SciFact and
**+0.0227** on NFCorpus, and gains +0.0018 on ArguAna. Hybrid fusion is
not a no-op on this suite; it is the largest single gain on the ladder
outside the encoder step.

What changed is the fusion rule, not the data. These E rows were scored with
`rrfK: 60, lexicalWeight: 0.5, guardMargin: 0.05` (recorded in each run's
`system.fusion`), the weighted fusion that became the default on 2026-09-23.
The text below described the older unweighted RRF, where an
`LEXICAL_CUTOFF=1.5` gate admitted few enough BM25 hits that the vector
ranking usually survived untouched. Under the weighted rule the lexical
ranking carries real weight and the guard margin, not the cutoff, is what
protects a strong vector hit.

ArguAna's near-tie is still explained the old way: its "queries" are full
argument passages, so BM25 and the vector ranking largely agree and there is
little for fusion to reorder. F (lexical-only) vs. D is no longer uniformly
negative either — F beats D by +0.0029 on SciFact, and trails it by 0.0101
on NFCorpus and 0.0538 on ArguAna — so "BM25 alone is a weak signal on this
suite" holds on two of three datasets, not all three.

This ladder does not yet include a
dataset where the two rankings meaningfully disagree at the top (that would
be the interesting case to add next — a corpus with real known-item/keyword
lookups, closer to what surfaced the calibration-side lexical blind spot;
see `packages/pikelet/src/calibrate.mjs`'s `fusedTop`/`LEXICAL_CUTOFF`).

**`config.json`'s `hybridWeight: 0.35` field is still stale/unused dead
config**, but the reason has changed: production's fusion does now carry a
weight term, just not that one. `fuseCandidates()` takes `lexicalWeight`
(0.5 by default) and a `guardMargin`, which is what these E/H rows were
scored with. `hybridWeight` predates that rule and nothing in the shipped
reader consults it; do not read it as documenting the knob that exists.

### Query-prefix ablation: Arctic-XS's asymmetric prefix is real and large — where queries are short

The CLI's `--encoder-query-prefix` help text documents that Snowflake
arctic-embed models require `"Represent this sentence for searching
relevant passages: "` on queries (and nothing on passages) "or they
silently lose the retrieval quality they were trained for." G-noprefix
measures exactly that claim instead of trusting it:

| Dataset | G (prefix) | G-noprefix | Δ nDCG@10 |
|---|---|---|---|
| SciFact | 0.6427 | 0.4791 | **-0.1636** |
| NFCorpus | 0.3085 | 0.1546 | **-0.1539** |
| ArguAna | 0.5166 | 0.5217 | +0.0052 |

Confirmed and large on SciFact and NFCorpus — both have short, question-shaped
queries, exactly the asymmetric case the prefix exists for. ArguAna is the
one dataset in this suite where the prefix does not help (dropping it scores
slightly higher),
because ArguAna's "queries" are full argument passages, not questions — the
query/passage asymmetry the prefix encodes barely applies when both sides
look like documents. The claim in the CLI help text is verified, with the
caveat that its size depends entirely on how question-like the query
workload is; a corpus queried with long passage-shaped input should not
expect the same effect size.

### The encoder question: does Arctic-XS win, and does it warrant recalibration?

**Mixed, dataset-dependent — not a clean win.** Comparing G/H (Arctic-XS)
against D/E (MiniLM), the same encoder swap that helped at ~2800-record real
Wikipedia scale (there Arctic-XS "rescued" 16/18 false-abstention cases, but by sidestepping
a calibration blind spot, not by being a better encoder in general) does not
reproduce as a general win here:

Every MiniLM figure in this comparison was restated on 2026-09-27 against the
scored runs on disk. The G/H rows were written before the E rows were rescored
in vector order (see the note under the table) and before the 512-token
re-encode, so this section had been comparing Arctic-XS against MiniLM numbers
that no longer appeared anywhere in the table — SciFact hybrid was cited as
0.6500 where the run scores 0.7011. The verdict below is unchanged; the margins
are not.

- **SciFact**: Arctic-XS trails MiniLM on every metric, dense (0.6427 vs.
  0.6562, -0.0135) and hybrid (0.6879 vs. 0.7011, -0.0132).
- **NFCorpus**: Arctic-XS trails MiniLM on every metric, dense (0.3085 vs.
  0.3164, -0.0079) and hybrid (0.3283 vs. 0.3391, -0.0108) — though the two
  are close enough here that the gap could plausibly narrow or flip with a
  larger query sample.
- **ArguAna**: Arctic-XS **beats** MiniLM, dense (0.5166 vs. 0.4889,
  +0.0277) and hybrid (0.5188 vs. 0.4907, +0.0281 — close to dense here,
  since hybrid is a near-no-op on ArguAna for either encoder). ArguAna is
  also the one dataset in the existing A→B row where MiniLM's own
  quantization shows a real, measured cost (-0.0142) — the dataset
  chosen specifically to stress semantic discrimination is the one place a
  different encoder pulls ahead.

**Verdict: does not warrant recalibrating the shipped default.** The
instruction was to recalibrate if Arctic-XS wins; it wins on one of three
datasets and loses on the other two, including the two datasets where BEIR
queries most resemble Pikelet's actual target workload (short natural-
language questions against a document corpus — SciFact and NFCorpus are
much closer to that than ArguAna's passage-vs-passage argument retrieval).
Swapping the shipped default encoder on the strength of a single favorable
dataset, when the other two show a real regression, would trade a measured
loss on the more representative workload for a measured gain on the least
representative one. This matches the earlier Wikipedia-scale finding
(2026-09-11): Arctic-XS's apparent "improvement" there was traced to a
blind spot in the calibration fit, which scored vector evidence only and
could not see BM25, and the encoder swap happened to route around it. It
was not a general retrieval-quality edge — this BEIR
run is independent confirmation that the encoder itself is not the fix.
`--encoder-model`/`--encoder-weights`/`--encoder-pooling`/
`--encoder-query-prefix` remain available as a CLI-level opt-in for a
corpus/workload where an operator has measured (as here) that it actually
helps — ArguAna-shaped corpora, in particular — but MiniLM stays the
default, and no recalibration was performed.

Independently verified the encoder is doing real, distinct inference (not a
fallback): mean cosine similarity between Pikelet-encoder and
float-reference query embeddings for identical SciFact query text is 0.9998
(min 0.9998, max 0.9999).

**Windowed mean-pooling rate (documents exceeding the inline encoder's
128-token window) varies a lot by dataset**: SciFact 5073/5183 (97.9%),
ArguAna 6595/8674 (76.0%). Despite both being mostly windowed, SciFact held
quality; ArguAna is also where the real encoder-quantity drop showed up —
worth investigating whether these two facts are related (windowing cost
compounding with encoder-quantization cost) once more datasets are in.

## Production vs. benchmark-harness encoding cost

An earlier version of this benchmark encoded the corpus with a hand-rolled
sequential loop through the raw kernel (one `embed()` call at a time,
no parallelism) — the wrong comparison, since it measured Pikelet's
*fallback* single-threaded path, not what `pikelet compile` actually does.
`encode-pikelet.mjs` now calls `embedChunksWithInlineTransformer` from
`packages/pikelet/src/embed.mjs` directly — the same function the CLI uses — which
dispatches to a `worker_threads` pool once a corpus exceeds 32 chunks.

Measured on this 8-core machine (`work/<dataset>/encode-pikelet.json`):

| Dataset | Corpus docs | Corpus encode (7 workers) | Test queries | Query encode (sequential) |
|---|---|---|---|---|
| SciFact | 5183 | 933s (~15.6 min) | 300 | 15.6s |
| NFCorpus | 3633 | 719s (~12.0 min) | 323 | 5.3s |
| ArguAna | 8674 | 1036s (~17.3 min) | 1406 | 789s (~13.1 min) |

Query encoding is sequential because that matches the real runtime query
path (a live reader embeds one query per request — no production batching
exists there to benchmark against). ArguAna's query-encode time is much
higher than the others because ArguAna queries are full argument passages,
not short questions — same per-item cost as corpus docs, just a lot more of
them (1406 vs. 300/323).

The worker-pool path produced byte-identical embeddings to an earlier
sequential-loop version of this script on SciFact (same nDCG@10/Recall
scores to full precision) — confirms determinism, not just speed.

## No readback API for stored quantized vectors

`Pikelet.create()`'s `export()` is an opaque whole-index snapshot; there is
no JS-callable method to read back a stored row's quantized bytes or
(scale, offset). `quantize.mjs` reimplements the engine's exact per-row
affine scheme (`src/uint8_float_hnsw.hpp:206-234`: min/max range, `scale =
range/255`, round-half-up, cosine metric normalizes before quantizing) so
config C measures the real algorithm, not an approximation of it.
`conformance/test-quantization.mjs` proves this with golden vectors
generated from the C++ logic copied verbatim into a standalone program —
not just an assertion in a README, a passing test.

## ID mapping

BEIR qrels reference BEIR corpus IDs (e.g. `"4983"`). Pikelet rows are
addressed by internal row index. `convert.mjs` writes
`work/<dataset>/id-map.json` mapping `pikeletRow -> beirId` so query results
can be translated back to BEIR IDs before scoring. No model involvement in
that mapping — it's positional, fixed at conversion time.

## Determinism

Corpus text is not rechunked: one BEIR document -> one Pikelet record, text
built deterministically as `<title>\n\n<body>`. Same corpus, same queries,
same qrels for every configuration on a given dataset.
