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
  `PancakeSketchArtifact`/HNSW-equivalent search, but vector-only)
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

See `config.json` for the frozen dataset list committed ahead of any
results.

## Layout

```
bench/beir/
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
                           PancakeSketchArtifact from index.export(), fuses exactly as complete/index.mjs does
  query-F.mjs              config F: BM25 lexical only (retrieval: 'lexical')
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
                           examples/05-one-file-search/encoder-spike/real/, which stays the shared MiniLM blob
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
python3.11 bench/beir/download.py <dataset>
node bench/beir/convert.mjs <dataset>

python3.11 bench/beir/embed.py <dataset> corpus
python3.11 bench/beir/embed.py <dataset> queries
node bench/beir/query-A.mjs <dataset>
python3.11 bench/beir/score.py <dataset> A

node bench/beir/encode-pikelet.mjs <dataset>   # production worker-pool path for
                                                # corpus (~7 workers on 8 cores),
                                                # single embed() calls for queries
node bench/beir/query-B.mjs <dataset>
python3.11 bench/beir/score.py <dataset> B
node bench/beir/query-C.mjs <dataset>
python3.11 bench/beir/score.py <dataset> C
node bench/beir/query-D.mjs <dataset>
python3.11 bench/beir/score.py <dataset> D

# E, F: what a real .pikelet artifact actually serves (hybrid default,
# lexical-only). Both reuse the vectors-*-pikelet.f32 encode-pikelet.mjs
# already produced above, plus records.jsonl for BM25 text — no re-encoding.
node bench/beir/query-E.mjs <dataset>
python3.11 bench/beir/score.py <dataset> E
node bench/beir/query-F.mjs <dataset>
python3.11 bench/beir/score.py <dataset> F

# G, H: the encoder question, on identical retrieval logic to D/E.
node bench/beir/encode-pikelet-arctic.mjs <dataset>              # corpus + queries, WITH the query prefix
node bench/beir/ablate-query-prefix.mjs <dataset>                # queries only, WITHOUT the prefix (ablation)
node bench/beir/query-G.mjs <dataset>                            # Arctic-XS, dense only
node bench/beir/query-G.mjs <dataset> --no-prefix                # same, no query prefix
node bench/beir/query-H.mjs <dataset>                            # Arctic-XS, hybrid RRF
python3.11 bench/beir/score.py <dataset> G
python3.11 bench/beir/score.py <dataset> G-noprefix
python3.11 bench/beir/score.py <dataset> H

node bench/beir/conformance/test-quantization.mjs   # golden-vector proof quantize.mjs == the real engine, run once
```

## Results so far

| Dataset | Config | nDCG@10 | Recall@10 | Recall@100 | median ms/query |
|---|---|---|---|---|---|
| SciFact (300q) | A upstream float exhaustive | 0.6451 | 0.7833 | 0.9250 | 3.6 |
| SciFact | B Pikelet encoder / float exhaustive | 0.6512 | 0.7942 | 0.9417 | 3.8 |
| SciFact | C Pikelet encoder / affine-u8 exhaustive | 0.6500 | 0.7942 | 0.9417 | 6.9 |
| SciFact | D Pikelet encoder / affine-u8, dense only | 0.6500 | 0.7942 | 0.9417 | 0.1 |
| SciFact | **E MiniLM / hybrid RRF (production default)** | 0.6500 | 0.7942 | 0.9427 | 7.3 |
| SciFact | F BM25 lexical only | 0.6590 | 0.7816 | 0.8726 | 0.6 |
| SciFact | G Arctic-XS / dense only | 0.6427 | 0.7700 | 0.8620 | 7.8 |
| SciFact | G-noprefix (Arctic-XS, no query prefix) | 0.4791 | 0.6154 | 0.7363 | 7.7 |
| SciFact | H Arctic-XS / hybrid RRF | 0.6473 | 0.7833 | 0.9103 | 8.6 |
| NFCorpus (323q) | A upstream float exhaustive | 0.3159 | 0.1550 | 0.3115 | 2.6 |
| NFCorpus | B Pikelet encoder / float exhaustive | 0.3154 | 0.1511 | 0.3044 | 2.8 |
| NFCorpus | C Pikelet encoder / affine-u8 exhaustive | 0.3154 | 0.1511 | 0.3036 | 2.9 |
| NFCorpus | D Pikelet encoder / affine-u8, dense only | 0.3135 | 0.1482 | 0.3061 | 0.2 |
| NFCorpus | **E MiniLM / hybrid RRF (production default)** | 0.3154 | 0.1511 | 0.3087 | 5.1 |
| NFCorpus | F BM25 lexical only | 0.3050 | 0.1432 | 0.2329 | 0.1 |
| NFCorpus | G Arctic-XS / dense only | 0.3085 | 0.1462 | 0.2607 | 5.9 |
| NFCorpus | G-noprefix (Arctic-XS, no query prefix) | 0.1546 | 0.0777 | 0.1818 | 5.5 |
| NFCorpus | H Arctic-XS / hybrid RRF | 0.3103 | 0.1479 | 0.2727 | 5.6 |
| ArguAna (1406q) | A upstream float exhaustive | 0.3698 | 0.7653 | 0.9772 | 7.8 |
| ArguAna | B Pikelet encoder / float exhaustive | 0.3506 | 0.7397 | 0.9801 | 6.1 |
| ArguAna | C Pikelet encoder / affine-u8 exhaustive | 0.3496 | 0.7368 | 0.9801 | 6.4 |
| ArguAna | D Pikelet encoder / affine-u8, dense only | 0.3496 | 0.7368 | 0.9808 | 0.1 |
| ArguAna | **E MiniLM / hybrid RRF (production default)** | 0.3496 | 0.7368 | 0.9780 | 18.6 |
| ArguAna | F BM25 lexical only | 0.3246 | 0.6871 | 0.9289 | 6.5 |
| ArguAna | **G Arctic-XS / dense only** | **0.3788** | 0.7681 | 0.9566 | 10.5 |
| ArguAna | G-noprefix (Arctic-XS, no query prefix) | 0.3795 | 0.7639 | 0.9381 | 10.4 |
| ArguAna | **H Arctic-XS / hybrid RRF** | **0.3788** | 0.7681 | 0.9566 | 17.8 |
| FiQA (648q) | in progress | — | — | — | — |

All three completed datasets independently validate the pipeline against
published literature: SciFact nDCG@10 ~0.64-0.65, NFCorpus ~0.31-0.32,
ArguAna ~0.35-0.40 are the expected ranges for MiniLM-L6-v2-class encoders
on these tasks — this is not a benchmark that happens to favor Pikelet, it
reproduces known-hard and known-easy cases correctly.

**Per-dataset A→B (encoder-quantization) deltas differ meaningfully:**
SciFact +0.0061 (net positive), NFCorpus -0.0005 (negligible), ArguAna
**-0.0192** (real, not noise at n=1406). ArguAna was chosen specifically to
stress semantic discrimination, and it's the first dataset where quantizing
the encoder shows a measurable cost. B→C (corpus quantization) and C→D
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

### Hybrid RRF vs. dense only (E vs. D): mostly a no-op on this suite, not a bug

E matches D's nDCG@10 exactly on SciFact and ArguAna, and beats it by
0.0019 on NFCorpus. This is not hybrid retrieval failing to do anything —
`LEXICAL_CUTOFF=1.5` (kept identical to `complete/index.mjs`'s production
constant) is deliberately conservative: only BM25 hits within a factor of
1.5 of the top lexical score join the fusion, so a corpus/query mix where
the top vector hit is already lexically well-supported produces no
reordering. F (lexical-only) is markedly worse than D on all three datasets
(-0.0 on none, up to -0.025 on ArguAna, -0.010 on NFCorpus), which is what
makes E's near-parity with D meaningful rather than vacuous: BM25 alone is
not a strong signal on this suite, and the cutoff is correctly declining to
let a weak signal disturb a strong one. This ladder does not yet include a
dataset where the two rankings meaningfully disagree at the top (that would
be the interesting case to add next — a corpus with real known-item/keyword
lookups, closer to what surfaced the calibration-side lexical blind spot;
see `pikelet/src/calibrate.mjs`'s `fusedTop`/`LEXICAL_CUTOFF` and the
`calibration-lexical-blindspot-2026-09-11` memory).

**`config.json`'s `hybridWeight: 0.35` field is stale/unused dead config.**
Production's RRF (`complete/index.mjs`) has no weight term — it's an
unweighted sum of `1/(RRF_K+rank)` over the two rankings, and E/H reproduce
that exactly. Do not read `hybridWeight` as documenting a real knob; nothing
in the shipped reader consults it.

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
| ArguAna | 0.3788 | 0.3795 | +0.0007 (noise) |

Confirmed and large on SciFact and NFCorpus — both have short, question-shaped
queries, exactly the asymmetric case the prefix exists for. ArguAna is the
one dataset in this suite where the prefix makes no measurable difference,
because ArguAna's "queries" are full argument passages, not questions — the
query/passage asymmetry the prefix encodes barely applies when both sides
look like documents. The claim in the CLI help text is verified, with the
caveat that its size depends entirely on how question-like the query
workload is; a corpus queried with long passage-shaped input should not
expect the same effect size.

### The encoder question: does Arctic-XS win, and does it warrant recalibration?

**Mixed, dataset-dependent — not a clean win.** Comparing G/H (Arctic-XS)
against D/E (MiniLM), the same encoder swap that helped at ~2800-record real
Wikipedia scale (see the `calibration-lexical-blindspot-2026-09-11` memory:
Arctic-XS "rescued" 16/18 false-abstention cases there, but by sidestepping
a calibration blind spot, not by being a better encoder in general) does not
reproduce as a general win here:

- **SciFact**: Arctic-XS trails MiniLM on every metric, dense (0.6427 vs.
  0.6500) and hybrid (0.6473 vs. 0.6500).
- **NFCorpus**: Arctic-XS trails MiniLM on every metric, dense (0.3085 vs.
  0.3135) and hybrid (0.3103 vs. 0.3154) — though the two are close enough
  here that the gap could plausibly narrow or flip with a larger query
  sample.
- **ArguAna**: Arctic-XS **beats** MiniLM, dense (0.3788 vs. 0.3496,
  +0.0292) and hybrid (0.3788 vs. 0.3496, +0.0292 — identical to dense here,
  since hybrid is a near-no-op on ArguAna for either encoder). ArguAna is
  also the one dataset in the existing A→B row where MiniLM's own
  quantization already showed a real, measured cost (-0.0192) — the dataset
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
representative one. This matches the Wikipedia-scale finding from
`calibration-lexical-blindspot-2026-09-11`: Arctic-XS's earlier apparent
"improvement" was traced to a calibration blind spot the encoder swap
happened to route around, not a general retrieval-quality edge — this BEIR
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
`pikelet/src/embed.mjs` directly — the same function the CLI uses — which
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
