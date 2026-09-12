# Pikelet Complete Search Artifact Profile

Copyright 2026 Matthew Noonan. This specification is part of the Pikelet
project and is licensed under the Apache License 2.0 (see `LICENSE`).

**Status:** Draft 2 (2026-08-21) — for review, not frozen. Draft 2 adds
format version 2: per-record corpus integrity (section 3.5), the host-encoder
verification obligation for kind 2 (section 3.6), the reader's bounded-read
rules (section 4), and the CI conformance suite (section 5). Format-1 files
remain readable; readers report which integrity stance a file carries.
**Profile of:** the Search Artifact Contract (`SEARCH_ARTIFACT_CONTRACT.md`, section 9.4)
**File extension:** `.pikelet`
**Magic:** `PSF1` (`0x31465350`, little-endian u32)

## 1. Purpose

One file that is a search engine for one corpus. A complete artifact carries
all five contract components — corpus, index, encoder, evaluation,
calibration — under a single content-addressed identity, so that a reader
can answer natural-language queries with hydrated, confidence-scored results
using nothing but the file and byte-range reads against it.

The profile composes formats this repository has already frozen rather than
inventing new ones: the index segment is a complete `.pikelet-sketch`
artifact embedded verbatim, and the query-interpretation segment carries the
existing student-encoder and calibration formats. The container contributes
identity, addressing, and the corpus layout — the three things the
2026-08-13 composition spike (`examples/05-one-file-search/`) identified as
missing between the components.

## 2. Design decisions (resolved 2026-08-13)

1. **Compile-time identity mapping.** Index row `i` IS corpus record `i`.
   The compiler renumbers when assembling the file; the container carries no
   id map and readers perform no id translation. A producer packing a
   snapshot with a non-identity internal/external id map MUST renumber the
   corpus to match the index's internal order (or rebuild) at compile time.
2. **Embedded sketch artifact as the index segment.** The index segment's
   bytes are a valid `.pikelet-sketch` file (SKETCH_PROFILE.md), opened by
   the existing sketch reader at `indexOffset`. Depth-1 execution, staged
   boot, and resident-hash verification are inherited, not re-specified.
3. **One query-interpretation unit.** Encoder and calibration share a
   segment and a version, because calibration may consume the encoder's
   feature stream and hidden state, not only its output vector. They cannot
   version independently.
4. **Three query-interpretation kinds.** Kinds 1 and 2 were added
   2026-08-14 for the wiki-scale compile: inline student encoders (kind 1)
   and pinned external encoders with verification vectors (kind 2,
   contract section 4.4 mode 2). Kind 3 (inline transformer, section 3.6)
   followed and is the product default — `pikelet compile`
   emits it. The three are not generations of one idea but corners of a
   trade-off: each picks two of self-contained, small, and teacher-quality
   (kind 3 gives up small — its encoder bytes are ~24 MB; kind 1 gives up
   teacher quality; kind 2 gives up self-contained). A conforming reader
   supports all three.

## 3. File layout

All integers little-endian. All offsets absolute. Segments begin at
16-byte-aligned offsets; inter-segment padding MUST be zero bytes.

```
[0, 64)       header
[64, ...)     manifest        canonical JSON, manifestBytes long
[...]         segment table   segmentCount x 48-byte entries
[...]         segments        in table order, 16-byte aligned
```

### 3.1 Header

| Offset | Type | Field | Notes |
| ---: | --- | --- | --- |
| 0 | u32 | magic | `0x31465350` (`PSF1`) |
| 4 | u32 | formatVersion | `1` (corpus layout v1, profile `pikelet-complete-v1`) or `2` (corpus layout v2, profile `pikelet-complete-v2`); readers MUST reject other values |
| 8 | u32 | manifestBytes | length of the canonical manifest JSON (readers MUST reject > 16 MiB) |
| 12 | u32 | segmentCount | number of segment-table entries |
| 16 | u64 | fileBytes | total file size; MUST match |
| 24 | 32 bytes | manifestSha256 | digest of the manifest bytes |
| 56 | 8 bytes | reserved | MUST be zero in v1; readers MUST ignore |

The **artifact identity** (contract section 4.1) is `manifestSha256`. The
manifest commits to every segment's digest, so the identity transitively
commits to all content. `fileBytes` and the header itself are addressing
convenience, not identity: two files with identical manifests and segments
but different segment order have different bytes and the same identity.

### 3.2 Manifest

Canonical JSON (UTF-8, sorted keys, no insignificant whitespace — the byte
serialization the digest is computed over is the one in the file). Required
fields:

```jsonc
{
  "profile": "pikelet-complete-v2",                    // "pikelet-complete-v1" for format-1 files
  "corpus": {
    "records": 208,
    "provenance": null,                                 // reserved per contract 4.3
    // format 2 only (layout v2, section 3.5); absent on format 1:
    "layout": "records-v2",
    "pageRecords": 256,
    "pages": 1,
    "recordDigest": "sha256",
    "pageTableSha256": "..."                            // digest of the page table bytes
  },
  "dim": 384,
  "metric": "cosine",
  // format 2 only: SHA-256 of the index segment's first 256 bytes (the
  // embedded sketch's header — section 3.4). REQUIRED on format 2.
  "index": { "headerSha256": "..." },
  "encoder": { /* identity, preprocessing, dims, normalization — contract 4.4 list */ },
  "segments": [
    { "kind": "index",       "sha256": "...", "bytes": 126400 },
    { "kind": "corpus",      "sha256": "...", "bytes": 167994 },
    { "kind": "query-interp","sha256": "...", "bytes": 1208893 },
    { "kind": "evaluation",  "sha256": "...", "bytes": 190192 }
  ],
  "sampleQueries": [ "..." ]
}
```

Segment order in `segments` MUST match the segment table. Unknown manifest
fields MUST be ignored by readers (and are covered by the identity digest).

### 3.3 Segment table entry (48 bytes)

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | u32 | kind (`1` index, `2` corpus, `3` query-interp, `4` evaluation, `5` lexical) |
| 4 | u32 | reserved (zero) |
| 8 | u64 | offset (16-byte aligned) |
| 16 | u64 | bytes |
| 24 | 24 bytes | reserved (zero) |

Exactly one segment of each kind 1–3 is REQUIRED; evaluation (kind 4) is
REQUIRED for conformance but a reader MAY serve queries without reading it.
At most one lexical segment (kind 5, section 3.8) MAY be present.
Unknown kinds MUST be skipped (they are still committed via the manifest).

### 3.4 Index segment (kind 1)

A byte-for-byte valid `.pikelet-sketch` artifact (magic `PSA1`), row ids
`[0, count)` binding positionally to corpus records. Readers open it with
the sketch reader against a range source offset by the segment's `offset`;
`staged` open is RECOMMENDED for interactive hosts.

The segment is deliberately lazy, so the container's whole-segment digest
is not checked at open — which means the sketch's *self*-checks
(`residentSha256`/`vectorsSha256`, held in its own header) are not by
themselves authentication: an attacker who rewrites the segment also
rewrites those fields. Format 2 therefore commits to the sketch's 256-byte
header in the manifest (`index.headerSha256`, REQUIRED): a reader MUST
verify the header against it at open, which anchors the header's metric /
dim / count and its `residentSha256` / `vectorsSha256` — and through them
the resident prefix and the lazy rows — to the artifact identity without a
whole-segment read. On every format a reader MUST also cross-check the
sketch header's metric, dim, and count against the identity-verified
manifest; on format 1 (no header commitment) that cross-check is the only
binding, and the residual gap is stated in section 6.

### 3.5 Corpus segment (kind 2)

The contract's section 4.8 id-to-byte-range requirement, in the simplest
layout that satisfies it:

```
[0, 4)                u32 count
[4, 4 + 8*(count+1))  u64 offsets[count+1]   record i occupies
                                             [offsets[i], offsets[i+1])
                                             relative to segment start
[...]                 records                UTF-8 JSON, one object each
```

Record objects carry at minimum `title`, `text`, `sourcePath`; `preview`,
`anchor`, and `url` are OPTIONAL. Hydrating result id `i` is one range read
of `[offsets[i], offsets[i+1])` — resident cost is the offsets array
(`8*(count+1)` bytes, ~1.6 MB at 200k records). Readers MAY cache records;
cache state MUST NOT change results.

That is **layout v1**, carried by format-1 files: record reads are covered
only by the whole-segment digest, so a lazily read record is not
independently verifiable. **Layout v2** (format 2, manifest
`corpus.layout: "records-v2"`) makes each record read range-verifiable per
contract sections 4.1 and 7:

```
[0, 4)                 u32 count
[4, 8)                 u32 pageRecords (P)        1 <= P <= 65536; 256 by default
[8, 8 + 8*(count+1))   u64 offsets[count+1]       record i = [offsets[i], offsets[i+1])
[A, A + 32*pages)      pageSha256[pages]          pages = ceil(count / P); entry p is the
                                                  SHA-256 of recordSha256[p*P, min(count,(p+1)*P))
[B, B + 32*count)      recordSha256[count]        SHA-256 of each record's bytes
[C, ...)               records                    offsets[0] MUST equal C
```

The manifest commits to the page table (`corpus.pageTableSha256`, together
with `pageRecords`, `pages`, and `recordDigest: "sha256"`), so the
verification chain is: identity → page table (read at open, 32 bytes per
`P` records — 57 KB at 456k records) → one page of record digests (one read
of `32*P` bytes the first time a record in that page is hydrated; readers
SHOULD cache verified pages) → the record. A reader MUST verify the page
table against the manifest at open, and each hydrated record against its
digest (after verifying that digest's page) before returning it; a mismatch
MUST fail the hydration, not degrade it. Per-record reads stay one range
read plus, per page touched, one small read.

Format-2 files MUST use layout v2 and header `formatVersion` 2; format-1
files MUST use layout v1 (no `corpus.layout`) and `formatVersion` 1. A
reader MUST reject a manifest whose layout does not match its header
version.

### 3.6 Query-interpretation segment (kind 3)

```
[0, 4)    u32 version          shared encoder+calibration version
[4, 8)    u32 kind             1 = student-inline-v1, 2 = external-transformers-v1,
                               3 = inline-transformer-v1
[8, 12)   u32 encoderBytes
[12, 16)  u32 calibrationBytes
[16, ...) encoder              per kind, below
[...]     calibration          UTF-8 JSON, per kind
```

**kind 1 — student-inline-v1:** encoder bytes are a distilled student model
(loadStudentModel format); calibration is the student abstention model,
which consumes the encoder's feature stream. Fully self-contained, pure-JS
execution.

**kind 2 — external-transformers-v1:** the contract's section 4.4 mode 2
(pinned external encoder). The encoder bytes are a UTF-8 JSON *declaration*
pinning the model (id, pooling, normalization policy, max tokens) and
carrying verification test vectors — query texts with their expected
embeddings at a declared tolerance — so a host-supplied encoder can be
checked against the artifact before serving. Calibration is a
retrieval-signal abstention model (distance signals plus a corpus-vocabulary
bloom filter, base64-embedded) that scores query text and hits without
touching the encoder internals. A reader without a host encoder for the
declared model MUST surface the artifact as requiring one, not fall back
silently. A reader given a host encoder MUST run it against every
verification vector before serving the first query — dimension equal to
the manifest's, every component within the vector's declared tolerance —
and MUST refuse the open on disagreement (encoder/index skew is a contract
violation, section 4.4). A verification vector's expected embedding MUST
itself be validated before comparison — exactly `dim` finite numbers — and
a malformed expectation MUST fail the verification rather than skip it
(NaN comparisons are never "within tolerance"). A kind-2 declaration
without verification vectors
is incomplete: a reader MUST refuse to serve it as verified and MAY serve
it only when the host explicitly accepts an unverified encoder, reporting
that state (the reference reader: `allowUnverifiedEncoder`,
`info().encoderVerified === false`).

**kind 3 — inline-transformer-v1:** the pinned teacher compiled into the
artifact as data. The encoder bytes are three length-prefixed regions:

```
[0, 4)   u32 declBytes      [4, 8)  u32 vocabBytes   [8, 12) u32 blobBytes
[12, ..) declaration        UTF-8 JSON: model identity, revision, license
                            and attribution (the corpus-provenance rule of
                            contract 4.3 applied to weights), pooling,
                            normalization, max tokens, an optional prefix
                            policy ({passage, query} strings; readers MUST
                            prepend the query prefix before embedding, so
                            queries land in the same space as prefixed
                            passages; absent means empty), and the blob
                            layout constants (V, P, D, F, L, B, H)
[...]    vocab              UTF-8, one WordPiece token per line
[...]    weight blob        block-affine u8 matrices + f32 norm params in
                            the declared deterministic layout
```

Weights are data; the kernels that execute them ship with the READER, never
inside the artifact (a reader MUST NOT execute code from artifact bytes —
contract section 7). Calibration is the same retrieval-signal model as
kind 2. A reader without transformer kernels for the declared layout MUST
surface the artifact as unsupported. The token-embedding table occupies a
contiguous region of the blob; a future revision of this kind will declare
its offsets so readers can leave it lazy (range-read per token id,
sketch-style) instead of resident — v1 readers load the blob whole.

In all kinds, encoder and calibration are loaded together and verified
together under the shared version. A reader that cannot evaluate the
calibration MUST report results as uncalibrated rather than inventing
confidence (contract section 4.7).

### 3.7 Evaluation segment (kind 4)

UTF-8 JSON: golden queries with expected top-k ids and match-quality
labels, recall-vs-C measurements for the embedded sketch geometry, and
teacher/student fidelity metrics. The golden queries double as the
conformance fixtures for readers (section 5).

### 3.8 Lexical index segment (kind 5, OPTIONAL)

A static BM25 inverted index over the corpus records, for hybrid
retrieval. At most one lexical segment MAY be present; readers that
predate the kind skip it (section 3.3) and serve vector-only queries from
the same file. Layout `bm25-v1`:

```
[0,4)   u32 version = 1
[4,8)   u32 docCount        MUST equal manifest corpus.records
[8,12)  u32 termCount
[12,20) u64 totalTokens
[20,24) u32 doclenOffset    absolute within the segment (= 64)
[24,28) u32 termTableOffset
[28,32) u32 postingsOffset
[32,40) u64 postingsBytes
[40,64) reserved, zero
doclens    u32[docCount]     indexed tokens per record
termTable  termCount x 24 B  u32 hashLo, u32 hashHi, u64 postingsOffset
                             (relative), u32 postingsBytes, u32 df —
                             sorted by (hashHi, hashLo)
postings   per term          varint docId (first absolute, then deltas),
                             varint tf, repeated df times
```

Terms are addressed by hash only — fnv1a-32 of the lowercase token with
seeds `0` and `0x9e3779b9` — so a reader searches the fixed-width table
without materializing term strings. The sorted uniform hashes make the
table an implicit interpolation index, which is how the reference reader
serves segments above 8 MiB lazily: header and doclens resident, term
windows and postings fetched per query. Lazily read regions carry the
transitional integrity stance (committed via the manifest segment digest,
not individually verified); the eager opener for smaller segments verifies
the whole-segment digest at open.
Hash collisions merge two terms' postings; over realistic vocabularies the
probability is negligible and the failure mode is a slightly-wrong lexical
score, never corruption. Builders tokenize on lowercase `[a-z0-9']+`,
keep tokens of 2–32 characters, and MAY skip function words; readers need
no knowledge of the builder's stopword list, because an unindexed query
token simply finds no postings. BM25 parameters are fixed: k1 = 1.2,
b = 0.75, idf = ln(1 + (N − df + 0.5)/(df + 0.5)).

The regions MUST tile the segment exactly in the order above; a reader
MUST reject a segment whose offsets disagree with its counts, whose
postings run past `postingsBytes`, or whose doc ids reach `docCount`.

## 4. Execution semantics

`open(source)`:

1. Read header (64 bytes); check magic, version `1` or `2`, sane counts
   (`manifestBytes` ≤ 16 MiB, 1 ≤ `segmentCount` ≤ 64, `fileBytes` a safe
   integer equal to the source size when the source reports one).
2. Read manifest + segment table; verify `manifestSha256`; parse; verify
   table/manifest agreement (kinds, packed 16-byte-aligned offsets within
   file, byte lengths, one segment per known kind, unknown kinds skipped),
   and the profile/layout against the header version.
3. Read the query-interpretation segment and verify its digest. A reader
   MAY defer a large kind-3 encoder region instead of reading it at open
   (the reference reader defers above 4 MiB): it reads the 16-byte
   segment header and the calibration region eagerly, fetches the full
   segment before the first query needs the encoder (prefetching in the
   background by default), verifies the whole-segment digest then, and
   MUST byte-compare the header and calibration slices it used at open
   against the verified segment before any query result is returned — a
   mismatch fails every query, never degrades one. Until the encoder
   arrives, `info()` serves the identity-verified manifest declaration
   and reports encoder verification as unknown. Eager readers read the
   segment in one request and verify
   its digest; check its version word; load encoder + calibration; for
   kind 2 with a host encoder, verify the host encoder (section 3.6).
4. Open the index segment with the sketch reader (staged or full); read the
   corpus tables (count + offsets, plus the page table on layout v2) in one
   read; check the count words against the manifest, the offsets
   (monotonic, starting exactly where the tables end, ending inside the
   segment), and on layout v2 the page table's digest.
5. The artifact is now serving. Total cold-open transfer at the reference
   corpus: header + manifest + query-interp + sketch stage-1 — under 2 MB.

Every read in steps 1–4 (and every lazy read below) is issued only after
its `(offset, length)` has been validated — safe non-negative integers, no
overflow, within `fileBytes`, within the source's size when known, and
within the reader's per-read budget (the reference reader: 256 MiB per
open-path read, configurable; 16 MiB per corpus record; 2 GiB absolute) —
and MUST return exactly the bytes requested. A short read is a failure, not
a partial success. u64 fields above `2^53 - 1` are rejected.

`query(text, k)`:

1. Encode: text → vector + feature stream (pre-search abstention MAY answer
   here without touching the index).
2. Search: sketch scan + one parallel rerank fetch round (SKETCH_PROFILE.md
   section 3). The manifest MAY carry fetch hints (`recommendedRerank`,
   `recommendedGap`) that readers SHOULD apply when the caller does not
   override: a producer that laid rows out in cluster order chooses the
   coalescing gap that matches its cluster geometry, and a reader using an
   unmatched gap forfeits the layout's request savings. When the artifact
   carries a lexical segment (3.8), the BM25
   top matches join the rerank as extra candidates — scored by true vector
   distance like any candidate, which is what recovers a known-item lookup
   the sketch scan's top-C missed — and the final result order fuses the
   distance ranking with the BM25 ranking by reciprocal rank (RRF,
   constant 60). Lexical hits scoring under a third of the best lexical
   score are dropped before fusion: idf collapses common query terms to
   near-tied scores that carry no ranking information. Readers SHOULD
   also offer an augmented mode — lexical candidates join the rerank but
   results keep pure distance order — for workloads scored against exact
   nearest neighbors, where rank fusion's reordering registers as loss by
   construction.
3. Hydrate: one range read per result id via the corpus offsets; on layout
   v2, verify the record against its digest (fetching and verifying that
   digest's page on first use). A record that fails verification fails the
   query rather than being returned.
4. Calibrate: score match quality from hits + feature stream, where hits
   are the distance-sorted top-k (fusion affects the returned order, not
   the calibrated signals); a `none` verdict returns zero results with the
   score.

Result shape: `{ matchQuality, confidence, results: [{ id, distance,
title, text, sourcePath, ... }] }` — ids, distances, and hydrated records
in one response; returning bare ids does not satisfy this profile. `id`
and `distance` are reserved result names: they MUST come from the search,
and a corpus record carrying fields of those names MUST NOT replace them
(records are application data; the binding between a record and the row
that matched is the reader's to assert). A reader MUST validate `k`
(positive integer, capped to the corpus size) and MUST refuse queries
after `close()`; `close()` MUST be idempotent.

## 5. Conformance

- **Fixtures:** `test/complete_profile.mjs` (run by `npm test`, so by CI)
  builds every fixture deterministically in-process — no downloads, no
  model weights — and is the reference reader's conformance suite:
  (A) a seeded kind-2 format-2 artifact with a deterministic host encoder,
  covering open/query/hydrate, host-encoder verification, per-record and
  page-table tamper detection, structural rejection of hostile headers,
  tables, and offsets, read budgets, truncation and short reads, unknown
  and duplicate segments, and format-version bounds; (B) the same corpus as
  a format-1 file, which MUST still open and report the transitional
  integrity stance (a record tamper is documented as undetectable there);
  (C) a kind-1 artifact compiled from the committed
  `examples/legacy/03-edge-docs-search` assets, whose 10 abstention goldens MUST
  reproduce their labels, whose hydration round-trips the source corpus,
  and whose compile MUST be byte-deterministic. Kind 3 is covered by
  `examples/05-one-file-search/test-inline.mjs` against the released
  wiki-inline artifact (its weight blob is not a CI fixture).
- **Producer:** emits structurally valid files whose manifest digests
  verify, whose index segment passes sketch-profile conformance, whose
  corpus round-trips every record, and whose evaluation bounds hold under
  the reference reader.
- **Reader:** verifies identity before serving (manifest digest; segment
  digests for eagerly-read segments; sketch resident hash), rejects
  unsupported versions/encoders, executes section 4, and passes the golden
  fixtures.
- All artifact bytes are untrusted input: every offset/length/count is
  validated before allocation per the contract's section 7 rules, and reads
  respect the layered budgets the readers already enforce.

## 6. Integrity stance

Format 2 commits to every corpus record individually (section 3.5): a
hydrated record is verified on its own range read through the page table
the manifest commits to, which is contract section 4.1's range-verifiable
chunk commitment sized to the corpus's real read shape (one record). Eager
segments (manifest, query-interp, evaluation, corpus tables) verify whole.
The index segment is anchored through the manifest's header commitment
(section 3.4): the sketch header is verified at open, so its
`residentSha256` authenticates the resident prefix and its `vectorsSha256`
authenticates the lazy rows, both under the artifact identity.

With a **format-2 embedded sketch** (SKETCH_PROFILE.md section 2.4 —
interleaved per-row digest blocks anchored by a page-hash table in the
resident prefix, which `residentSha256` and therefore `index.headerSha256`
and the identity cover), every rerank row is verified on the read that
fetches it: nothing that influences a query goes unauthenticated, per
read. `info().indexRowIntegrity` reports `'per-row-sha256'`.

What remains transitional applies only to artifacts whose embedded sketch
is **format 1**. Format 1 was every released artifact through 0.4.0
(2026-08-25); format 2 has been the builder's default, and every artifact
actually released, since 0.5.0 (2026-08-26) — format 1 is no longer
produced by any in-repo build path, but a reader MUST still open one
correctly, since "released" only constrains what this project ships, not
what every artifact a host might serve was built with. Format-1 lazy
rerank rows are covered by the identity-anchored whole-segment
`vectorsSha256` but are NOT verified on the reads that feed reranking, so
between open and a full vectors pass, modified vector bytes can alter
results under an unchanged identity. Hosts serving format-1-sketch
artifacts that need every byte authenticated MUST run the full pass —
`verifyIndexVectors: true` at open or `verifyVectors()` afterwards
(`info().vectorsVerified` reports the state).

Format-1 files keep Draft 1's stance — whole-segment digests only, lazy
record reads not independently verifiable — and readers MUST report which
stance a file carries (`info().corpusIntegrity`) rather than presenting
both as equivalent.

## 7. Relationship to existing tooling

- **Compiler:** `pikelet`'s pipeline (ingest → chunk → embed
  → index → distill → calibrate) becomes the frontend; a `compile` step
  assembles its outputs into one `.pikelet`. Requires bytes-in/bytes-out
  builder variants (spike lesson) and compile-time renumbering.
- **Hosts:** a static page with the browser reader, a Worker, and Node all
  open the same file; the Worker example becomes one host among three.
- **Existing profiles:** `.pnck`, `.pikelet-range`, and `.pikelet-sketch`
  remain valid standalone profiles; this container embeds the sketch
  profile and does not deprecate anything.

## 8. Open questions for Draft 2

1. Corpus compression (per-record or segment-level) — record-granular
   range reads argue for per-record; measure before deciding.
2. Whether `sampleQueries` belongs in the manifest or the evaluation
   segment (identity implications of moving it).
3. Signatures over the identity (contract section 10, question 7).
4. A `filters` or metadata-index segment — out of contract today
   (section 8), revisit only with a concrete host need.
5. Browser reader packaging (the encoder runs in plain JS today; confirm
   no Node-only dependencies before freezing the host story).
6. ~~Per-row vector commitments (closing section 6's transitional gap per
   read).~~ **Resolved and shipped as SKETCH_PROFILE.md's format 2**,
   default since 0.5.0 (2026-08-26) — kept here as the design record.
   Measured 2026-08-25 on the wiki workload — byte/run model,
   adversarial dispersion, and real-HTTP wall-clock replay
   (`docs/measurements/row-commitments/ROW_COMMITMENT_MEASUREMENT.md`).
   Outcome, for SKETCH_PROFILE.md's next revision: **interleaved digest
   blocks** — each 16-row group stores a 16-byte-per-row truncated
   SHA-256 digest page inline ahead of its rows, with 32-byte page
   hashes in the resident prefix (891 KiB at wiki scale,
   identity-covered via `residentSha256`). Measured: ~0% query-latency
   overhead on HTTP/1.1 and HTTP/2 at 20/80 ms RTT (digests ride inside
   the reader's existing coalesced row runs — zero extra requests),
   ~3× byte inflation on the rerank's cold reads (~230 ms at 50 Mbps
   when bandwidth, not RTT, dominates), +4.2% file size,
   adversarial-safe (degrades linearly). Rejected by measurement:
   corpus-v2's 256-row/32-byte geometry (4–6× byte amplification),
   lazily fetched separate-region tables (~2× wall-clock on *both*
   protocols — request count dominates even under h2 multiplexing), and
   Merkle-per-run proofs (perpetual ≥0.75× byte cost, no cache
   convergence). Verify-everything hosts warm the ~7 MB digest region
   once instead of today's 167 MB full-vectors pass (~24× less
   transfer).
