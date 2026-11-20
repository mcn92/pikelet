# The `.pikelet` format

A `.pikelet` file is a searchable corpus in one file. It holds the source
text, a vector index, an optional keyword (BM25) index, the neural encoder
that turns a question into a vector, and the tests the pack was built to
pass. A
single SHA-256 digest, the file's **identity**, commits to all of it.

The layout is designed for byte-range reads. A reader fetches a small
prefix to open the file, then only the rows and records a query touches,
so the same file serves from a local disk, an object store or any static
HTTP host that honours `Range`.

This document describes the format as it exists in `pikelet-wasm` and is
written by `pikelet compile` today. The normative texts are the
specifications in [`spec/`](spec/) and
[`LAYERED_PROFILE.md`](LAYERED_PROFILE.md); the reference reader is
[`packages/pikelet-wasm/complete/`](packages/pikelet-wasm/complete/).
Where this page simplifies, those win.

**Status: not frozen.** The format is versioned and every released version
stays readable (see [Compatibility](#compatibility)), but new versions may
still be added before 1.0.

## Contents

- [What is in a pack](#what-is-in-a-pack)
- [Binary structure](#binary-structure)
- [Identity and integrity](#identity-and-integrity)
- [The embedded encoder](#the-embedded-encoder)
- [Indexes](#indexes)
- [Layers: updating a pack](#layers-updating-a-pack)
- [Versioning](#versioning)
- [Compatibility](#compatibility)
- [Operational limitations](#operational-limitations)
- [Specifications and code](#specifications-and-code)

## What is in a pack

Two packs built by the current compiler, for scale:

| Segment | 3-record folder | Simple English Wikipedia (456,153 records) |
| --- | ---: | ---: |
| Query interpretation (encoder + calibration) | 24.5 MiB | 24.9 MiB |
| Index (vector sketch + rows) | 2.0 KiB | 220.1 MiB |
| Corpus (records + digests) | 2.4 KiB | 334.4 MiB |
| Lexical (BM25) | 1.1 KiB | 69.2 MiB |
| Evaluation | 2.2 KiB | under 0.1 MiB |
| **File** | **24.5 MiB** | **648.5 MiB** |

The encoder is a fixed cost: every pack carries it, so the smallest pack
is about 25 MiB. Above that, the size grows with the corpus.

A reader never needs the whole file. Opening the Wikipedia pack reads the
header, manifest, segment tables and the index's 46.1 MiB resident prefix;
the encoder (~25 MiB) loads with the first query; each later query reads
roughly 0.5–1.1 MiB of rows and records
([measured](docs/wikipedia-demonstration.md)).

## Binary structure

All integers are little-endian and all offsets are absolute.

```
[0, 64)        header
[64, ...)      manifest          canonical JSON, manifestBytes long
[...]          segment table     segmentCount entries of 48 bytes
[...]          segments          each at a 16-byte-aligned offset; padding is zero
```

### Header (64 bytes)

| Offset | Type | Field | Meaning |
| ---: | --- | --- | --- |
| 0 | u32 | magic | `0x31465350`, the bytes `PSF1` |
| 4 | u32 | formatVersion | `1` or `2`; `pikelet compile` writes `2` |
| 8 | u32 | manifestBytes | length of the manifest; at most 16 MiB |
| 12 | u32 | segmentCount | 1 to 64 |
| 16 | u64 | fileBytes | total file size; must equal the actual size |
| 24 | 32 bytes | manifestSha256 | SHA-256 of the manifest bytes: **the identity** |
| 56 | 8 bytes | reserved | zero |

### Manifest

UTF-8 JSON with sorted keys and no insignificant whitespace, so it has one
byte serialization and one digest. Abridged from a pack compiled today:

```jsonc
{
  "profile": "pikelet-complete-v2",
  "corpus": {
    "records": 3,
    "provenance": { "source": "folder", "name": "src" },   // plus "license" when compiled with --license
    "ingest": { "chunker": "pikelet-section-v1", "targetTokens": 256, "overlapPercent": 15, "minTokens": 25 },
    "layout": "records-v2", "pageRecords": 256, "pages": 1,
    "recordDigest": "sha256", "pageTableSha256": "0c25…"
  },
  "dim": 384,
  "metric": "cosine",
  "encoder": { "kind": "inline-transformer-v1", "model": "sentence-transformers/all-MiniLM-L6-v2",
               "pooling": "mean", "normalized": true, "maxTokens": 512 },
  "index": { "headerSha256": "7c7f…" },
  "lexical": { "layout": "bm25-v1", "docCount": 3, "terms": 36, "totalTokens": 120 },
  "recommendedRerank": 3,
  "sampleQueries": [],
  "segments": [
    { "kind": "index",        "bytes": 2016,     "sha256": "7283…" },
    { "kind": "corpus",       "bytes": 2478,     "sha256": "8949…" },
    { "kind": "query-interp", "bytes": 25724590, "sha256": "6c28…" },
    { "kind": "evaluation",   "bytes": 2265,     "sha256": "dbd2…" },
    { "kind": "lexical",      "bytes": 1156,     "sha256": "f509…" }
  ]
}
```

`segments` lists every segment's length and SHA-256 in table order.
Readers ignore manifest fields they do not know; those fields are still
covered by the identity.

### Segment table (48 bytes per entry)

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | u32 | kind |
| 4 | u32 | reserved, zero |
| 8 | u64 | offset, 16-byte aligned |
| 16 | u64 | bytes |
| 24 | 24 bytes | reserved, zero |

### Segment kinds

| Kind | Name | In a complete pack | Read |
| ---: | --- | --- | --- |
| 1 | index | required | resident prefix at open, rows per query |
| 2 | corpus | required | tables at open, one record per hydrated result |
| 3 | query-interp | required | calibration at open; encoder before the first query |
| 4 | evaluation | required for conformance | only by `verify_pack` and tests |
| 5 | lexical | optional; written by default | whole at open, or lazily above 8 MiB |
| 6 | tombstones | layers only | whole at mount |
| 7 | lineage | compacted bases only | whole at mount |
| 8 | row map | optional (`--sentence-windows`) | whole at open |

A reader skips kinds it does not know; their bytes are still committed by
the manifest. At most one segment of each known kind may be present.

### Corpus segment (kind 2), layout `records-v2`

```
u32 count
u32 pageRecords                  P, 256 by default
u64 offsets[count + 1]           record i is bytes [offsets[i], offsets[i+1])
32-byte pageSha256[pages]        SHA-256 of the record digests in page p
32-byte recordSha256[count]      SHA-256 of each record's bytes
records                          UTF-8 JSON objects, one per record
```

A record carries at least `title`, `text` and `sourcePath`, and optionally
`preview`, `anchor`, `url` and source-specific metadata (a commit, a
conversation turn, a journal note). Records are stored uncompressed.
Record `i` is the vector index's row `i` (or, with a
[row map](#row-map-kind-8), owns a run of rows): the compiler numbers
them together, so the file has no id map. Format-1 files use layout v1: the same
count and offsets, with no page table or record digests.

### Query-interpretation segment (kind 3)

```
u32 version            1
u32 kind               1 student-inline-v1, 2 external-transformers-v1,
                       3 inline-transformer-v1, 4 inherited-v1 (layers only)
u32 encoderBytes
u32 calibrationBytes
encoder                per kind (see The embedded encoder)
calibration            UTF-8 JSON
```

The encoder and the calibration share a segment and a version because
the calibration is fit to that encoder's outputs; neither can change
without the other.

Packs compiled after 0.14.0 carry an empty calibration
(`{"kind":"retrieval-signals-v1","asset":null,"vocabBloomBase64":""}`),
and the reference reader reports their `matchQuality` as `unscored`. A
query on any pack also returns `grounding`, the question's words the best
of the top five passages contains and lacks, computed from the passages'
text alone ([how a query runs](docs/how-a-query-runs.md#grounding-which-question-words-the-passages-contain)).

Packs compiled by 0.14.0 and earlier may carry a fitted calibration
(`retrieval-signals-v1`): a small model plus a base64 bloom filter of the
corpus vocabulary. At query time it scores the search's distance signals
and the query's word coverage into a `matchQuality` verdict: `strong`,
`weak` or `none`. Packs from the last of those releases also carry a
second, distance-only model (`gate`) that alone decides `none`; word
coverage then only separates `strong` from `weak`. Readers that predate
the gate ignore it. The verdict measures whether the corpus covers the
question's topic, not whether a passage states the answer.
[docs/calibration.md](docs/calibration.md) says why `compile` stopped
fitting it.

### Evaluation segment (kind 4)

UTF-8 JSON: the golden queries the pack was built with (each a question
and the record or title it must retrieve in the top 10) and the
recall-versus-rerank-depth sweep that chose `recommendedRerank`; packs
compiled by 0.14.0 and earlier also hold expected verdicts and the
calibration's fit metrics. `compile` keeps only goldens the assembled
pack passes, and `verify_pack` (MCP) replays them on demand. A reader can serve queries
without ever reading this segment.

## Identity and integrity

**Identity** is `manifestSha256`. The manifest commits to each segment's
digest, and some segments commit further to their own parts, so the
identity transitively commits to every byte of content. Two files with the
same identity hold the same corpus, indexes, encoder and calibration.

**Pinning.** A reader can be told the identity it expects:
`openPikeletFile(url, { expectedIdentity })`, `pikelet mcp --pack
<location>#<sha256>`, or a pin in a shelf file. The reader checks the
64-byte header against the pin before downloading anything else, and
refuses a mismatch.

**What is checked, and when.** Integrity is enforced per read, so a reader
never has to download a whole large segment to trust the part it uses:

| Bytes | Committed by | Checked |
| --- | --- | --- |
| Manifest | the identity | at open |
| Query interpretation | manifest digest | at open; a segment over 4 MiB is fetched in the background and checked whole before the first query result, with the slices used at open compared against it |
| Corpus tables and page table | manifest (`pageTableSha256`) | at open |
| Each record | its digest, in a digest page anchored by the page table | on every read, before the record is returned; a failure fails the query |
| Index header (256 bytes) | manifest (`index.headerSha256`) | at open |
| Index resident prefix | the header's `residentSha256` | at open |
| Index rows | per-row digests in 16-row blocks, anchored by a page-hash table in the resident prefix | on every read that fetches them |
| Lexical segment up to 8 MiB | manifest digest | whole, at open |
| Lexical segment over 8 MiB | manifest digest | **not per read**; only by a full pass (`verifyLexical()`, or `verify_pack` with `full: true`) |
| Evaluation, row map | manifest digest | whole, when read |

`info()` reports the stance of each part (`corpusIntegrity`,
`indexRowIntegrity`, `lexical.integrity`, `lexical.verified`,
`encoderVerified`) so a host can tell what has been checked.

**What integrity does not mean.** A verified pack is exactly the pack its
identity names; it says nothing about whether its text is true or safe.
There are no signatures: an identity proves which bytes you have, not who
produced them. Trust in a publisher has to come from where you got the
identity.

**Hostile input.** Every offset, length and count is validated before it
is used: reads must lie inside the file and the segment they belong to,
u64 fields above 2^53 − 1 are refused, short reads fail, and reads are
capped (256 MiB per open-path read by default, 16 MiB per record, 2 GiB
absolute). A reader never executes code from a pack; the encoder is data
run by kernels that ship with the reader.

## The embedded encoder

The query encoder is inside the file, so a query needs no model download,
no embedding API and no network beyond the file itself. Three
query-interpretation kinds exist, each giving up one of self-contained,
small and full-quality:

| Kind | Name | Encoder bytes | Trade-off |
| ---: | --- | --- | --- |
| 3 | `inline-transformer-v1` | the quantized transformer itself, ~24 MiB | the default; self-contained and full quality, but large |
| 1 | `student-inline-v1` | a distilled student model, small | self-contained and small, lower quality; not appendable |
| 2 | `external-transformers-v1` | a JSON declaration naming the model, plus test vectors | small and full quality, but the host must supply the encoder |

**Kind 3**, which `pikelet compile` writes, holds three length-prefixed
regions:

```
u32 declBytes, u32 vocabBytes, u32 blobBytes
declaration     JSON: model and revision, license and attribution, pooling,
                normalization, maxTokens, query/passage prefixes, test
                vectors, and the weight layout (V, P, T, D, F, L, H, B, Q)
vocab           WordPiece vocabulary, one token per line
weight blob     block-quantized matrices and f32 norm parameters
```

The default encoder is `sentence-transformers/all-MiniLM-L6-v2`
(Apache-2.0): 6 layers, 384 dimensions, mean pooling, normalized, with
weights quantized to u8 in blocks of 64 (a 24.3 MiB blob and a
30,522-token vocabulary). Text longer than the 512-token window is embedded in windows
whose vectors are mean-pooled (`longInputs: "windowed-mean-pool"`).
Packs compiled before that declared `maxTokens: 128`.
The vocabulary is bert-base-uncased's (Apache-2.0, Google), unmodified.
A pack therefore redistributes both under Apache-2.0; the repository's
NOTICE file credits them.

The declaration carries **test vectors**: query texts with the embeddings
the build produced. When the reader loads the encoder it embeds each one
and refuses to serve if any component differs beyond the declared
tolerance, so a kernel that computes differently from the one that built
the index is caught before it answers (`info().encoderVerified`).

The reader's kernels (C++ compiled to WebAssembly, integer GEMM) run a
BERT-shaped encoder with a fixed shape: vocabulary 30,522, 512 positions,
hidden size 384, 12 heads, feed-forward 1,536, and 1 to 12 layers, with
u8 weights in blocks of 64 or u4 weights in blocks of 32. A pack
declaring any other shape is refused as unsupported. The kernels are
bit-identical across Node and the three major browser engines, which is
what lets test vectors use a tight tolerance.

**Kind 2** is for hosts that already run an encoder. Its declaration pins
the model and carries test vectors; a reader given a host encoder must
pass every vector before the first query, and a reader without one
refuses to serve the pack rather than falling back.

## Indexes

### Vector index (kind 1)

The index segment is a complete `.pikelet-sketch` file embedded verbatim
([`spec/SKETCH_PROFILE.md`](spec/SKETCH_PROFILE.md)), magic `PSA1`. It is
built for range reads: one round of parallel fetches per query, instead
of the sequential hops a graph index needs.

```
[0, 256)           sketch header: metric, dim, count, sketch geometry,
                   residentSha256, vectorsSha256, recommendedRerank,
                   per-row digest geometry
resident prefix    per-row scales and offsets, then one compressed sketch per row
                   (192 dimensions at 4 bits by default), then the page-hash table
vectors            every row as dim u8 values (affine-quantized), in blocks of
                   16 rows each preceded by its 16 per-row digests
```

A query embeds the question, scans every row's sketch in memory (in
WebAssembly where available), takes the best `C` candidates
(`recommendedRerank`, chosen at build time from the evaluation sweep:
200 on the Wikipedia pack), fetches those rows by range in one coalesced
round, verifies them, and reranks by exact distance. Metric is cosine (or
L2); vectors are normalized before quantization.

### Keyword index (kind 5, `bm25-v1`)

A static BM25 inverted index over the records:

```
64-byte header      version, docCount, termCount, totalTokens, region offsets
doclens             u32 per record
term table          24 bytes per term: 64-bit term hash, postings offset,
                    postings length, document frequency; sorted by hash
postings            per term: varint doc-id deltas and term frequencies
```

Terms are stored only as hashes (two FNV-1a-32 seeds), and the sorted,
uniform hashes let a reader find a term by interpolation without loading
the table. Up to 8 MiB the segment is read whole; above that only the
header and doclens are read at open, and term windows and postings are
fetched per query (see [integrity](#identity-and-integrity) for the cost).

The tokenizer keeps lowercase `[a-z0-9']` runs of 2–32 characters. The
same rule drives a fitted calibration's word coverage; `grounding` grades
words of three letters or more, or any word holding a digit. BM25 parameters are
k1 = 1.2, b = 0.75.

### How the indexes combine

BM25's top candidates join the vector rerank, so they are scored by true
vector distance like any other candidate; the final order fuses the vector
ranking and the BM25 ranking by reciprocal rank. The fusion constants,
BM25 parameters and cut-offs belong to the reader version, not the file:
they are the same for every pack a given reader opens.

### Row map (kind 8)

With `compile --sentence-windows <n>`, the index holds each record's own
vector plus one vector per run of `n` sentences, so a single answering
sentence is not averaged away by the rest of its chunk. The row map
(`u32 offsets[records + 1]`) says which rows belong to which record;
results, citations and the corpus stay per record. The manifest declares
the policy as `index.rowMap`.

## Layers: updating a pack

A pack is immutable. Changes ship as **layers**: small `.pikelet` files
(profile `pikelet-layer-v1`, format version 2) that each name their
parent's identity and add records, delete records, or supersede one record
with another. `pikelet append` writes a layer, and a reader mounts the
chain from its newest member, the **head**, whose identity commits to every
ancestor.

- A layer's new records take the next global ids, so ids never change
  and are never reused; citations made against an earlier head still
  resolve.
- Deletions are a cumulative tombstone bitset (kind 6); supersessions are
  `(oldId, newId)` pairs in the same segment.
- A layer carries no encoder. Its query-interpretation segment is kind 4
  (`inherited-v1`), which names the base's identity and the digest of the
  base's encoder segment; when the base is calibrated, its calibration
  region holds a bloom filter of the layer's own vocabulary.
- A chain is at most 16 layers deep; by default `append` stops at 8.
- A base calibrated by 0.14.0 or earlier keeps its calibration until
  accumulated change (records appended plus records deleted, over the
  base's record count) passes a limit of 0.20; past it the chain is
  served `unscored` and `append` refuses unless given `--allow-drift`. A
  base with no calibration has no such limit.
- `pikelet compact` folds a chain into a new base whose lineage segment
  (kind 7) translates every earlier head's ids forward.

The full rules are in [`LAYERED_PROFILE.md`](LAYERED_PROFILE.md).

## Versioning

Versions sit at several levels, and each is checked by the reader:

| Level | Where | Values |
| --- | --- | --- |
| Container | header `formatVersion` + manifest `profile` | `1` = `pikelet-complete-v1`; `2` = `pikelet-complete-v2` or `pikelet-layer-v1` |
| Corpus layout | manifest `corpus.layout` | absent (v1, format 1 only) or `records-v2` (format 2 only) |
| Vector index | sketch header `formatVersion` | `1` (whole-segment digests) or `2` (per-row digests) |
| Encoder and calibration | query-interp `version` and `kind` | version 1; kinds 1–4 |
| Keyword index | manifest `lexical.layout` + segment version | `bm25-v1` |
| Other segments | their own version words | tombstones, lineage and row map: 1 |
| Ingestion | manifest `corpus.ingest.chunker` | `pikelet-section-v1` |

A reader refuses a container or segment version it does not know, and a
manifest whose profile or corpus layout disagrees with the header. A new
segment *kind* does not need a new container version: readers skip kinds
they do not know.

Format 2 with a version-2 sketch has been what the compiler writes since
0.5.0 (2026-08-26). Keyword indexes arrived in 0.6.0, layers in 0.9.0,
the ingestion declaration in 0.9.2 and row maps in 0.11.0.

## Compatibility

- **Old files open.** Readers open format 1 and 2, sketch format 1 and 2,
  and every query-interpretation kind. Each part reports the integrity
  stance it actually has instead of claiming the strongest.
- **The Pancake era.** The project was called Pancake until September
  2026. Readers accept the old profile strings `pancake-complete-v1` and
  `-v2`, and dispatch on magic bytes, not file extensions, so `.pancake`
  files open unchanged ([history](docs/history.md)).
- **Older readers and newer files.** A reader that predates the keyword
  index skips it and serves vector-only results from the same file. A
  reader that predates row maps refuses a file that carries one (its
  row count no longer matches the record count), rather than misreading
  it. A reader that predates layers refuses a layer.
- **Identity is stable under reading, not rebuilding.** Recompiling the
  same inputs with the same compiler version gives the same identity
  (releases up to 0.14.0 need `SOURCE_DATE_EPOCH` set for this). A
  different compiler version can change chunking, calibration or
  defaults, and with them the identity.
- **Transport.** Over HTTP the host must answer `Range` requests with
  `206 Partial Content`. If the file's validator (`ETag`) changes while it
  is open, reads fail with `ARTIFACT_CHANGED` instead of mixing bytes from
  two files.

## Operational limitations

- **About 25 MiB minimum.** The encoder is in every pack, and the first
  query on a remote pack downloads it.
- **English-shaped text processing.** The keyword index and the word
  coverage behind `grounding` keep only `[a-z0-9']` runs, so accented words are
  split (`café` is indexed as `caf`) and Greek, Cyrillic and CJK text
  produce no keywords. The default encoder was trained on English.
  Chunking needs whitespace-separated words and at least 25 of them, so
  unsegmented scripts produce no chunks. See the
  [Limitations](packages/pikelet/README.md#limitations) section of the
  CLI reference.
- **One encoder shape.** The kernels run only the 384-dimensional
  BERT-shaped encoder described above. A different model needs kind 2 (a
  host encoder) or new kernels.
- **Large keyword indexes are not checked per read.** Above 8 MiB, BM25
  reads are confined to their segment but not verified one by one; until a
  full pass, altered postings could change which authentic records rank
  highly (record text and vector rows are still verified). Format-1
  files have the same gap for records and rows.
- **Retrieval behaviour is partly in the reader.** The file fixes the
  encoder, the indexes and the calibration. The fusion rule, BM25
  parameters and cut-offs come from the reader, so two reader versions
  may rank the same pack differently.
- **No metadata filters, no compression.** There is no filter or facet
  index, and records are stored as raw JSON.
- **Size limits.** At most 2^31 − 1 records per pack or chain; a manifest
  of at most 16 MiB and at most 64 segments; reads capped as above. The
  kind-3 encoder is loaded whole (about 25 MiB of memory per open pack).
- **Updates are append-only.** Layers grow a chain up to 16 deep, and
  each adds a file to fetch and an index to search until a compaction
  folds the chain into one base.
  A pack with a row map cannot be layered.
- **No signatures.** Identity establishes which bytes you have, not who
  made them.
- **One implementation.** The reference reader and builder are the only
  implementation. The specifications are written to allow others, but
  none has been tested against them.

## Specifications and code

| Document | Covers |
| --- | --- |
| [`spec/SEARCH_ARTIFACT_CONTRACT.md`](spec/SEARCH_ARTIFACT_CONTRACT.md) | the behavioural contract: identity, tolerance, evaluation, calibration |
| [`spec/COMPLETE_PROFILE.md`](spec/COMPLETE_PROFILE.md) | the `.pikelet` container and its segments |
| [`spec/SKETCH_PROFILE.md`](spec/SKETCH_PROFILE.md) | the embedded vector index |
| [`LAYERED_PROFILE.md`](LAYERED_PROFILE.md) | layers, chains, compaction and rebase |
| [`packages/pikelet-wasm/complete/`](packages/pikelet-wasm/complete/) | the reference reader and builder (`format.mjs` holds the shared constants) |
| [`test/complete_profile.mjs`](test/complete_profile.mjs), [`test/layered_profile.mjs`](test/layered_profile.mjs) | conformance suites, including tampered and hostile files |
