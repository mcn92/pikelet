# BEIR retrieval benchmark

Compact, reproducible BEIR subset for measuring how much retrieval quality
Pikelet loses (if any) moving from ordinary float MiniLM retrieval to
Pikelet's compact, staged, self-contained artifact.

This is Day 1/2 infrastructure: the adapter pipeline (download → convert →
embed → query → score) proven end-to-end across four points on the ablation
ladder — **A** (upstream MiniLM float exhaustive), **B** (Pikelet encoder +
float32 exhaustive), **C** (Pikelet encoder + affine-u8 exhaustive), **D**
(Pikelet encoder + affine-u8 HNSW, the real `PancakeIndex`) — on SciFact,
NFCorpus, and ArguAna, with FiQA in progress. **E**, **F** are not built
yet; see `config.json` for the frozen dataset list committed ahead of any
results. TREC-COVID (the fifth listed dataset) has not been run.

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
  query-D.mjs              config D: Pikelet-encoder vectors + real PancakeIndex HNSW (Pikelet.create/addBatch/search)
  score.py                 score a run against official qrels (nDCG@10, Recall@10/100) via pytrec_eval
  conformance/             golden-vector test proving quantize.mjs matches the real engine exactly (see below)
  cache/                  downloaded datasets (gitignored)
  work/                   converted records, id maps, embedded vectors, raw runs (gitignored)
  results/                scored JSON output, one file per dataset+configuration (gitignored)
```

`build.mjs` / `query.mjs` / `run-all.sh` (configs E, F, and a
single-command driver) don't exist yet — that's the next slice, not this one.

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

node bench/beir/conformance/test-quantization.mjs   # golden-vector proof quantize.mjs == the real engine, run once
```

## Results so far

| Dataset | Config | nDCG@10 | Recall@10 | Recall@100 | median ms/query |
|---|---|---|---|---|---|
| SciFact (300q) | A upstream float exhaustive | 0.6451 | 0.7833 | 0.9250 | 3.6 |
| SciFact | B Pikelet encoder / float exhaustive | 0.6512 | 0.7942 | 0.9417 | 3.8 |
| SciFact | C Pikelet encoder / affine-u8 exhaustive | 0.6500 | 0.7942 | 0.9417 | 6.9 |
| SciFact | D Pikelet encoder / affine-u8 HNSW | 0.6500 | 0.7942 | 0.9417 | 0.1 |
| NFCorpus (323q) | A upstream float exhaustive | 0.3159 | 0.1550 | 0.3115 | 2.6 |
| NFCorpus | B Pikelet encoder / float exhaustive | 0.3154 | 0.1511 | 0.3044 | 2.8 |
| NFCorpus | C Pikelet encoder / affine-u8 exhaustive | 0.3154 | 0.1511 | 0.3036 | 2.9 |
| NFCorpus | D Pikelet encoder / affine-u8 HNSW | 0.3135 | 0.1482 | 0.3061 | 0.2 |
| ArguAna (1406q) | A upstream float exhaustive | 0.3698 | 0.7653 | 0.9772 | 7.8 |
| ArguAna | B Pikelet encoder / float exhaustive | 0.3506 | 0.7397 | 0.9801 | 6.1 |
| ArguAna | C Pikelet encoder / affine-u8 exhaustive | 0.3496 | 0.7368 | 0.9801 | 6.4 |
| ArguAna | D Pikelet encoder / affine-u8 HNSW | 0.3496 | 0.7368 | 0.9808 | 0.1 |
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
(HNSW) stay small and roughly flat relative to B on all three datasets —
the loss so far, where it exists, concentrates in the encoder step, not the
storage or search-approximation steps. Do not generalize the "quantization
is free" framing past what's actually measured; ArguAna is evidence against
an unqualified version of that claim.

Latency: exhaustive search (A/B/C) runs a few ms/query at these corpus
sizes; the real HNSW index (D) runs at 0.1-0.2ms/query, 20-70x faster,
at equal or near-equal measured quality on every dataset run so far.

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
