# Pikelet Sketch Artifact Profile

Copyright 2026 Matthew Noonan. This specification is part of the Pikelet
project and is licensed under the Apache License 2.0 (see `LICENSE`).

**Status:** Draft 1
**Profile of:** the Search Artifact Contract (`SEARCH_ARTIFACT_CONTRACT.md`)
**File extension:** `.pikelet-sketch`
**Magic:** `PSA1` (`0x31415350`, little-endian u32)

## 1. Purpose

A sketch artifact executes approximate nearest-neighbor search over
immutable, range-readable storage with **one parallel fetch round per
query**. It replaces graph traversal (the range artifact profile's
execution model) with a two-tier design:

1. a **resident tier** — a compressed sketch of every vector, loaded once
   and kept in memory;
2. a **lazy tier** — the full quantized vectors, fetched by byte range only
   for the candidates the resident scan selects.

The profile exists because sequential dependency depth, not byte volume,
dominates remote-read latency. Measured on SIFT1M against object storage
over the public internet, this geometry answered queries 5x faster than
graph traversal at equal recall (Track A measurement, 2026-08-01; project
roadmap notes now kept internally).

Under the contract, this is a distinct profile, not an amendment to the
range artifact profile: the two share the contract layers but declare
different execution semantics.

## 2. File layout

All integers are little-endian. All offsets are absolute byte positions.

```
[0,   256)                      header
[256, scalesEnd)                scales    count x f32
[..., offsetsEnd)               offsets   count x f32
[..., sketchesEnd)              sketches  count x sketchDims x sketchBits/8 bytes
[..., vectorsOffset)            padding to 16-byte alignment (zero bytes)
[vectorsOffset, end)            vectors   count x dim bytes (u8 rows)
```

Everything before `vectorsOffset` is the **resident prefix**. A reader
opens the artifact by fetching `[0, vectorsOffset)` in one read (or a few
large reads) and holding it in memory. The vectors segment is never read
eagerly.

### 2.1 Header

| Offset | Type | Field | Notes |
| ---: | --- | --- | --- |
| 0 | u32 | magic | `0x31415350` (`PSA1`) |
| 4 | u32 | formatVersion | `1`, or `2` with per-row integrity (section 2.4); readers MUST reject other values |
| 8 | u32 | kind | `1` = u8 affine rows (the only defined kind) |
| 12 | u32 | metric | `0` = L2, `1` = cosine |
| 16 | u32 | dim | full vector dimensionality |
| 20 | u32 | count | number of vectors; row id range is `[0, count)` |
| 24 | u32 | sketchDims | pooled dimensionality; MUST divide `dim` |
| 28 | u32 | sketchBits | `4` or `8` |
| 32 | u32 | scalesOffset | always `256` in v1 |
| 36 | u32 | offsetsOffset | |
| 40 | u32 | sketchesOffset | |
| 44 | u32 | vectorsOffset | 16-byte aligned |
| 48 | u64 | fileBytes | total file size; MUST equal `vectorsOffset + count*dim` |
| 56 | 32 bytes | residentSha256 | SHA-256 of `[256, vectorsOffset)` |
| 88 | 32 bytes | vectorsSha256 | SHA-256 of the vectors segment |
| 120 | u32 | recommendedRerank | producer's suggested top-C (see 4.3); `0` = unset |
| 124..168 | — | staged residency extension | micro-tier fields and stage-1 hash (section 8); all zero when the extension is absent |
| 168 | u32 | rowsPerBlock | v2 only (section 2.4); rows per digest block, `[1, 4096]` |
| 172 | u32 | rowDigestBytes | v2 only; truncated per-row digest size, `[8, 32]` |
| 176 | u32 | pageTableOffset | v2 only; start of the page-hash table, exactly at the end of the sketch tiers |
| 180..256 | — | reserved | MUST be zero; readers MUST ignore |

### 2.4 Per-row integrity (format version 2)

Version 2 makes every lazily fetched row verifiable on the read that
fetches it. The vectors region becomes a sequence of blocks —

```
[ digest page: rowsPerBlock x rowDigestBytes ][ rowsPerBlock rows x dim ]
```

— where slot `j` of a block's digest page is the first `rowDigestBytes`
bytes of the SHA-256 of row `blockIndex*rowsPerBlock + j` (unused slots in
the final block are zero; the final block carries only the remaining
rows). A **page-hash table** of one full 32-byte SHA-256 per block sits at
`pageTableOffset`, at the end of the resident prefix, so `residentSha256`
— and any container identity that commits to this header — anchors it.
`fileBytes` equals `vectorsOffset` plus the interleaved region;
`vectorsSha256` covers the whole interleaved region. Row `id` lives at
`vectorsOffset + floor(id/P)*(P*D + P*dim) + P*D + (id mod P)*dim`.

A v2 reader MUST fetch a needed block from its digest page through the
last needed row (one span; the page rides inside the read the reader
already makes), verify the page against the page-hash table and each
returned row against its digest slot before returning or caching it, and
fail the fetch on any mismatch. `verify: false` opts the whole open out,
reported as unverified. Defaults (16 rows, 16-byte digests) were chosen by
the 2026-08-25 access-pattern measurement
(`docs/measurements/row-commitments/ROW_COMMITMENT_MEASUREMENT.md`):
digests ride inside existing coalesced runs at ~0% latency overhead and
+4.2% file size. Version-1 files remain valid; readers report which
stance a file carries.

Integrity stance: the two whole-segment hashes let a reader verify the
resident prefix after loading it and the vectors segment after a full
download. The reference reader verifies `residentSha256` at open (default
on, fail-closed when verification is requested but no crypto backend
exists) and exposes `verifyVectors()` to check the vectors segment on
demand, streamed in bounded chunks where the runtime provides a streaming
hash. On **format 1** the hashes do not make individual range reads
verifiable — the transitional stance shared with the snapshot and range
profiles. **Format 2** (section 2.4) closes that per-read: every fetched
row is verified against its digest page, and the page against the
resident page-hash table, before it can influence a result. A bare v2
file's chain still terminates in its own header, so an attacker who can
rewrite the whole file (rows, pages, table, and both header hashes)
defeats it; anchoring the header externally — the complete profile's
`index.headerSha256` manifest commitment — extends the chain to a
content-addressed identity.

### 2.2 Row encoding

Vectors are stored as row-wise affine u8, identical to the engine's
quantized backend: the real value of dimension `d` of row `i` is

```
value(i, d) = offsets[i] + scales[i] * vectors[i*dim + d]
```

For `metric = cosine`, the producer MUST normalize each vector to unit L2
norm **before** quantization, and readers normalize queries before use.

### 2.3 Sketch encoding

The sketch of row `i` is produced by mean-pooling adjacent groups of
`pool = dim / sketchDims` quantized bytes:

```
pooled(i, sd) = round( mean( vectors[i*dim + sd*pool .. i*dim + (sd+1)*pool - 1] ) )
```

Pooling commutes with the affine dequantization — the mean of
`offset + scale*b` over a group equals `offset + scale*mean(b)` — so the
sketch reuses each row's `scales[i]` / `offsets[i]` unchanged. No
codebooks, no training step.

- `sketchBits = 8`: store `pooled` directly, one byte per sketch dimension.
- `sketchBits = 4`: store `q = min(15, round(pooled / 17))` in a nibble;
  reconstruct as `q * 17`. Two sketch dimensions per byte, **low nibble
  first** (dimension `2k` in bits 0–3, `2k+1` in bits 4–7). Rows are
  nibble-aligned: `sketchDims` MUST be even when `sketchBits = 4`.

Producers SHOULD choose `sketchDims` so that pooling preserves ranking
quality on their data; 2:1 pooling is the measured reference point for
SIFT-class vectors (4:1 measurably degrades recall, per project
measurement notes kept internally).

## 3. Execution semantics

A conforming reader executes `search(query, k, C)` as:

1. **Query preparation.** For cosine, L2-normalize the query. Compute the
   pooled query `qpool[sd] = mean(query[sd*pool .. (sd+1)*pool - 1])` in
   float; the query side is never quantized (asymmetric distance).
2. **Resident scan.** For every row `i`, compute the sketch distance using
   dequantized sketch values `offsets[i] + scales[i] * recon(i, sd)`:
   L2 = sum of squared differences against `qpool`; cosine = `1 - dot`,
   with the dot clamped to `[-1, 1]`. Select the `C` rows with the
   smallest sketch distance. Ties MAY break arbitrarily.
3. **One fetch round.** Fetch the byte ranges
   `[vectorsOffset + i*dim, vectorsOffset + (i+1)*dim)` for the selected
   rows. Readers SHOULD coalesce adjacent/near ranges and issue the
   resulting requests concurrently; there is no ordering dependency
   between any of them. Coalescing gaps SHOULD be small (0–4096 bytes):
   candidates are scattered, and large gaps fetch filler without reducing
   round count (measured: gap 65536 costs ~17x the bytes of gap 0 at
   C=300 on SIFT1M).
4. **Exact rerank.** Compute the full asymmetric distance for each fetched
   row using the resident `scales`/`offsets`. Return the top `k` ascending
   by distance. Reported distances follow the engine contract: L2 readers
   report Euclidean distance (or squared, if declared); cosine readers
   report `1 - cos`.

Recall is governed by `C` (and by the sketch geometry the producer chose);
round depth is 1 by construction. `k` MUST be `<= C`; a reader MAY satisfy
this by raising an under-sized requested `C` to `k` (the reference reader
clamps rather than rejects).

A reader MAY cache fetched rows across queries; cache state MUST NOT
change result semantics.

## 4. Parameters and guidance

### 4.1 Resident cost

`residentBytes = 256 + count*8 + count*sketchDims*sketchBits/8` (plus
alignment). Reference points at 1M rows, 128D: 68.7 MiB at 64D/8-bit,
38.1 MiB at 64D/4-bit — the latter matching the range profile's resident
router for the same dataset, with recall preserved (96.00% vs 96.11%
recall@10 at C=300 on SIFT1M).

### 4.2 Fetch cost

`C * dim` bytes before coalescing overhead — 300 x 128 = 37.5 KiB per
query at the SIFT1M reference point. This profile stores no edges, so the
per-candidate fetch is the vector alone.

### 4.3 Choosing C

`recommendedRerank` records the producer's measured operating point (the C
at which held-out recall@10 crossed the producer's target). Readers SHOULD
default to it when set. On SIFT1M with 64D sketches: C=200 → ~94%,
C=300 → ~96%, C=400 → ~97% recall@10.

### 4.4 Scan cost

The resident scan is O(count x sketchDims). Reference: 1M x 64D in ~17 ms
through the engine's `pancake_sketch_scan` SIMD kernel, ~85 ms in plain
JS. At corpus sizes where the linear scan dominates, an in-memory index
over the sketch tier is the intended escalation path; it does not change
this profile's on-disk format.

## 5. Contract conformance

Mapping to the Search Artifact Contract's layers:

- **Index semantics (4.5):** the header declares dim, metric, count,
  quantization, and format version. Row ids are corpus-binding ids: id `i`
  refers to the producer's `i`-th record. The "traversal" declaration for
  this profile is section 3 of this document — scan, one fetch round,
  exact rerank.
- **Execution (4.8):** the resident prefix is the declared resident
  segment; the vectors segment is the declared lazy segment. Result
  hydration is out of scope for v1 (corpus is an adjacent asset, as in the
  other transitional profiles).
- **Identity (4.1):** transitional — whole-segment hashes only, as
  described in 2.1.
- **Evaluation (4.6):** producers SHOULD ship recall-vs-C measurements as
  an adjacent asset and set `recommendedRerank`; this becomes a mandatory
  segment in the complete profile.

Conformance fixtures (contract 5.4): committed in
`test/fixtures/sketch_golden.js` and checked by `test/sketch_profile.js`
(part of `npm test`). Each of the four cases (l2/cosine × 4-bit/8-bit)
carries the base64 `.pikelet-sketch` bytes, fixed queries, and the reference
reader's exact ids and distances at `(k, C)` of `(5, 32)` and `(10, 64)`. A
conforming reader must reproduce those results from the committed bytes; the
WASM-scanner path is held to the same ids. Regenerate with
`node scripts/make-sketch-fixture.mjs`. The suite additionally checks
C=count exactness against the restored engine and recall floors against
float brute force.

## 6. Relationship to other profiles

| | Snapshot (`.pnck`) | Range (`.pikelet-range`, deprecated) | Sketch (`.pikelet-sketch`) |
| --- | --- | --- | --- |
| Execution | full restore, in-memory HNSW | resident router + lazy graph traversal | resident scan + one-round rerank |
| Sequential fetch rounds | n/a | ~24 cold (measured) | 1 |
| Stores graph | yes | yes | no |
| Best regime | corpus fits in memory | warm caches, small corpora | cold/remote, large corpora |

The range profile is deprecated (contract section 9.2): the sketch's depth-1
geometry beat traversal in every measured regime, so readers stay supported
but no new range artifacts should be built.

All three are produced from the same quantized index build; the sketch
artifact is derivable from a snapshot (or from a range artifact) without
re-embedding or re-training anything.

## 7. Open questions for v2

1. ~~Per-chunk integrity commitments~~ — resolved in format version 2
   (section 2.4): interleaved per-row digest blocks anchored by a resident
   page-hash table, geometry chosen by the 2026-08-25 access-pattern
   measurement.
2. An optional in-file evaluation segment (recall-vs-C table) ahead of the
   complete profile.
3. Sub-4-bit or trained (PQ) sketch encodings, which would introduce
   codebook segments.
4. An optional resident index over the sketch tier for very large counts.


## 8. Staged residency extension (optional, v1-compatible)

A producer MAY append a **micro tier**: a coarser pooling of the same
quantized rows, written after the full sketches and before `vectorsOffset`.
Because pooling commutes with the per-row affine map, the micro tier reuses
`scales`/`offsets` unchanged. Header fields (all zero when absent):
`microDims` u32 @124, `microBits` u32 @128 (4 or 8), `microOffset` u32 @132,
and a **stage-1 hash** @136..167: SHA-256 over the concatenation of the
scales+offsets segment and the micro segment, in that order.

Constraints: `microDims` divides `sketchDims`, `microDims < sketchDims`,
`microOffset == sketchesOffset + count*sketchRowBytes`, and the micro segment
ends at or before `vectorsOffset`. v1 readers that ignore these fields remain
correct: the micro segment is resident-tail bytes already covered by the
resident hash, and all existing layout checks pass.

Geometry guidance (measured, 456k-chunk 384-D corpus, 2026-08-04): spend the
micro byte budget on dims, never bits. At identical stage-1 bytes, 96d/4-bit
captured 87.8% of the exact top-10 at C=800 versus 48d/8-bit's 59.4%, and
4-bit equaled 8-bit at every width tested. Producers SHOULD set
`microDims = sketchDims / 2` with `microBits = 4` (the builder default), and
MUST NOT pool deeper than 4:1 relative to `dim` without measuring — the same
pooling cliff governs both tiers.

**Staged open** (`open(source, { staged: true })`): the reader fetches the
header, then scales+offsets and the micro segment in one parallel wave,
verifies the stage-1 hash, and serves queries from the micro tier while the
remainder of the resident prefix streams in the background. On completion the
full resident hash is verified before the tier swap; a failed verification
rejects `fullyResident` and the reader keeps serving the verified micro tier.
Tier state is explicit: every search result carries `tier: 'micro' | 'full'`,
and `fullyResident` resolves at convergence, after which results are
byte-identical to a non-staged open. While serving from the micro tier the
default candidate pool is `recommendedRerank * microBoost` (default 4).
Caching still MUST NOT change result semantics within a tier; the tier
transition is the one sanctioned, labeled semantic change.
