# Pikelet Layered Search Artifact Profile

Copyright 2026 Matthew Noonan. This specification is part of the Pikelet
project and is licensed under the Apache License 2.0 (see `LICENSE`).

**Status:** Draft 4 (2026-09-26) — the last conceptual-review draft; not
frozen. Implemented and shipping since pikelet-wasm 0.9.0 / pikelet 0.9.0
(2026-09-29): `pikelet append`, `compact` and `rebase`, chain mounting in
`pikelet mcp` and `openPikeletChain` in `pikelet-wasm`. Section 11 records
where the implementation differs from this text.

**Measurement pass, 2026-09-26.** Open questions 1, 2 and 3 have been run
against the shipped code on BEIR corpora; 13's preamble states the method
and the scope limits, and the findings are recorded in 13.1-13.3 with the
affected normative sections updated (4.5, 5.4, 5.5). In summary: the
inherited fit does not degrade out to 40% drift, so 5.5's 0.20 limit is
conservative; bloom pollution is real but never changed a verdict, and
4.5's claim about *which direction* it pushes was backwards; stale lexical
statistics cost under 0.005 nDCG@10, so 13.3's proposed per-layer `df`
correction table is dropped. None of this is wiki-pack scale, and questions
4-9 remain untouched, so the implementation gate of 13 had moved but not
opened (the implementation went ahead after this pass; 4-9 are still
open). Draft 4 revises Draft 3 (same day) after a
fourth review that found no remaining architectural defect: the
compaction conformance oracle no longer demands ranking invariance from
an operation designed to change retrieval statistics, and instead asserts
corpus preservation plus independently checked retrieval correctness
(6.3, 10 E); redirect targets during parent resolution are confined to
the child's canonical directory at every hop, and encoded separators are
rejected (5.1.1); the chain's ingestion declaration is defined by
inheritance, with only a depth-1 layer able to introduce an assertion
(6.1); an absent producer envelope is `+∞` and a reader's limit MUST be
finite (5.5); a layer that introduces neither a record nor a new
tombstone is prohibited (6.1); `C` is defined as counting unmasked rows
at query time, so `recommendedRerank` stays a property of the layer's
sketch (4.6, 5.3); and lineage translation is named *forward*
translation, distinct from historical state (6.3). A post-review pass
the same day added the term **search tier** (3.1) so that masking,
candidate generation and lexical statistics unambiguously include the
base, which "for every layer" had literally excluded (5.1–5.4, 8); a
terminal rule that an empty rebase emits no layer (6.2); an
ingestion-declaration compatibility precondition for rebase (6.2); and
the schema rule that `ingestAsserted` is a sibling of `layer.ingest`,
never a member of it (4.1). Draft 5 should be
driven by `layered_profile.mjs`, the range-proof harness, R2 and real
queries rather than by another reading of this text. Draft 3 revised
Draft 2 after a third review, and fixed two correctness defects: rebase now replays only the tombstones a
layer itself introduced (its delta over `zeroExtend` of its parent's mask,
3.4) and the supersession edges it physically recorded, never its
inherited cumulative state (6.2); and tombstone masks are supplied to
candidate generation in every layer so dead rows never occupy a candidate
slot or rerank budget (5.3). It also repairs the provenance story: a bare
id is an id in the mounted head, a citation is `(identity, id)`, and it
resolves only along the mounted ancestry (5.6); the lineage segment now
carries a table of every head in the compacted chain and per-edge depth,
with normative validation, so a citation against any retired head of that
chain translates (6.3). Further: the drift limit is
`min(producerEnvelope, readerLimit)` so an artifact cannot loosen a host
(5.5); "bounded by drift" is replaced everywhere by the truthful
statement that drift is a change-volume gate implying no error bound
(4.5, 5.4); `zeroExtend` is defined (3.4); rebase regenerates chain-level
evaluation material (6.2); kind-1 bases are readable but not appendable,
and appendability requires an ingestion declaration (6.1); locator checks
run after canonicalization (5.1.1); the candidate budget counts every row
fetched (5.9); and the chain-only 32-term truncation is withdrawn in
favour of an explicit failure at the lexical ceiling (5.4, 5.9). Draft 2
fixed rebase id translation, interval-based ownership, preflight budgets
and translation-versus-dereference; Draft 1 fixed bloom semantics,
calibration drift, the layer's own segment requirements, depth limits,
supersession chaining, rebase conflicts, parent resolution, chain budgets
and identity-versus-availability. Nothing in this document changes the
bytes of any existing `.pikelet`; every released artifact is a valid
**base** of a chain as it stands (whether it is *appendable* is 6.1's
question).
**Profile of:** the Search Artifact Contract (`SEARCH_ARTIFACT_CONTRACT.md`)
and the Complete Search Artifact Profile (`COMPLETE_PROFILE.md`, Draft 2),
whose container and segment layouts this profile reuses without amending.
**File extension:** `.pikelet` (a layer is a `.pikelet` file)
**Magic:** `PSF1` (`0x31465350`), unchanged
**Manifest profile string:** `pikelet-layer-v1`

## 1. Purpose

This is not a mutable `.pikelet`. It is **mutable corpus state represented
by immutable `.pikelet` artifacts**: every file keeps the contract's
immutability and identity rules exactly, and the thing that changes is
which files a reader mounts. That boundary is the reason the design works
and every rule below follows from it.

The contract's immutability rule (section 4.2) is not "no writes"; it is
that an artifact is immutable after publication and a superseding artifact
has a new identity. This profile takes that literally. A write compiles a
small **layer**: a range-readable artifact carrying only the records it
adds, a tombstone set over the records it hides, and a commitment to the
identity of the artifact it extends. A **chain** is a base `.pikelet`
followed by zero or more layers, each committing to its parent. The head
of the chain has an identity like any artifact, and because every layer's
manifest commits to its parent's identity, the head's identity commits to
the whole corpus a reader will serve. `docs.pikelet#<head>` still answers
"what body of knowledge did this agent query?" — as a commitment; serving
it additionally requires every ancestor's bytes to be reachable (3.2).

Everything a reader relies on today survives: static hosting (on a host
with a mutable pointer — a shelf file or a redirect — a write is one new
object and a pointer update; on a host with no mutable pointer, naming the
new head is the caller's job, as it is for any new artifact today), range
reads (each layer is a normal range-readable file), lazy integrity (each
layer is verified as a complete container is), and the identity-mapping
rule (index row `i` is corpus record `i`, per file). What changes is that a
reader mounts a list of files instead of one, scans a list of resident
tiers instead of one, and masks a bitset.

This is the log-structured design every mutable index eventually arrives
at — Lucene segments, LSM trees, git — kept deliberately small: no
in-place mutation, no server, no id map, a hard cap on depth, and a
compaction step that turns a chain back into one file and restores every
exact property the layers approximated.

## 2. Design decisions (proposed)

1. **A layer reuses the complete container and every segment layout, and
   is a distinct profile.** The additions are a `layer` object in the
   manifest, a `tombstones` segment (kind 6), and a query-interpretation
   kind that inherits the base's encoder (kind 4). A layer has its own
   required-segment table (4.3) and is never served on its own; a reader
   that implements the complete profile implements this one by adding the
   chain logic of section 5, not a second container parser.
2. **Identity is a hash chain.** A layer's manifest carries
   `layer.parent.identity`. Contract 4.1 requires that an artifact
   referencing external segments commit to their identities, not their
   URLs; a parent is an external segment in that sense. The head identity
   therefore transitively commits to every byte of every layer and of the
   base. Any existing `pikelet-complete-v1` or `-v2` artifact can be the
   base of a chain with no change to its bytes. Commitment is not
   availability: a chain whose ancestor is missing does not mount (3.2).
3. **Global record ids are offsets, and are scoped to a history.** Layer
   `j` owns the contiguous id range `[rowBase_j, rowBase_j + records_j)`,
   where `rowBase_0 = 0` for the base and `rowBase_j = rowBase_{j-1} +
   records_{j-1}`. Within a layer, index row `i` IS corpus record `i` IS
   global id `rowBase_j + i`. No id map exists anywhere (complete profile
   decision 1, preserved). Ids are never reused within a chain; only
   compaction (6.3) reclaims them, by renumbering into a new chain. An id
   names one record only along one ancestry: two children of the same
   parent assign the same id ranges to different records, so a **citation
   is `(identity, id)`**, and a bare id means "in the mounted head" (5.6).
4. **Deletion is a cumulative bitset, never a removal.** A layer's
   tombstone set covers `[0, rowBase_j)` and includes every ancestor's
   tombstones, under the `zeroExtend` superset rule of 3.4. This is
   deliberately `O(rowBase)` bytes per layer in exchange for constant-time
   lookup, a head-only mask, and a superset check that is one pass over
   two byte arrays (3.4 states the scaling and where it stops being free).
   The mask is applied *before* candidate selection in every layer (5.3),
   never after truncation. A tombstoned record is hidden from search but
   remains hydratable by id, because an agent may have cited it: deletes
   hide provenance, they do not destroy it. Nothing can be un-deleted;
   restoring a record means appending it again under a new id.
5. **One encoder per chain, committed by digest.** A layer MUST be
   embedded by the base's encoder. The invariant a reader verifies is kind
   4's commitment to the base's query-interpretation segment digest (4.5);
   the layer's manifest also copies the base's `encoder` declaration
   verbatim so the layer is self-describing under contract 4.4, and that
   copy is checked for consistency (4.1). Layers do not carry the ~24 MB
   inline transformer again.
6. **Calibration is inherited while a change-volume gate allows it.** The
   base's abstention model is applied to the chain while the accumulated
   change since its fit — appended records plus tombstones, over base
   records — is under the effective limit; beyond it the reader reports
   `unscored` with a reason, per contract 4.7. The gate bounds *how much
   changed*, not the error that change causes (4.5, 5.4). The effective
   limit is the stricter of the producer's declared envelope and the
   host's policy (5.5). Compaction refits.
7. **Two depth limits, of different kinds.** The format's interoperability
   ceiling is 16 layers above the base: a reader MUST reject deeper
   chains, and changing this number is a format change. The producer's
   default refusal at depth 8 is an operational recommendation pending
   measurement (`append --max-depth`, 6.1), and tuning it is not a format
   change. Both exist because the profile serves release-cadence change,
   not online writes (contract section 8), and compaction is part of the
   workflow.
8. **Compaction is semantic maintenance, and it renumbers.** A compacted
   base drops tombstoned records, renumbers survivors, restores exact
   lexical statistics, clears the historical vocabulary from the bloom,
   resets depth to zero and refits calibration (6.3). It carries a
   lineage segment naming **every head of the compacted chain**, its
   final tombstone state and every supersession edge with its depth, so a
   citation against any of those heads translates, and the new identity
   commits to that translation.
9. **Records have identity within a history, not content identity.**
   Appending a record whose bytes equal an existing one creates a new id.
   The profile does not deduplicate; a producer MAY, as policy, using the
   corpus layout's per-record digests (6.1).

## 3. Chain model

### 3.1 Definitions

**Base** — an artifact whose manifest has no `layer` object: any
`pikelet-complete-v1` or `pikelet-complete-v2` file. Depth 0.

**Layer** — an artifact with manifest profile `pikelet-layer-v1` and a
`layer` object. Depth = parent depth + 1.

**Chain** — the sequence base, layer₁, …, layer_k reachable from a head by
following `layer.parent.identity`. The **head** is the last layer (or the
base, for a chain of depth 0). **Ancestry** of a head — the set of
identities in its chain, base included.

**Row total** of an artifact `A` in a chain — `rowTotal(A) = A.rowBase +
A.records` (`A.records` for a base): the number of ids that existed when
`A` was the head.

**Search tier** — the base, or any layer with `records ≥ 1`: a chain
member that owns ids and carries an index (and corpus, and lexical
segment when the chain is hybrid). Tombstone-only layers are chain
members but not search tiers. Search tiers are indexed `j = 0 … m` in
chain order with the base as `j = 0` and `rowBase_0 = 0`; every
per-tier construction in section 5 — masks, candidate generation,
lexical statistics, id ownership — ranges over search tiers, so that the
base is always included and tombstone-only layers never are.

**Live record** — a global id not set in the head's tombstone bitset.

**Calibration drift** — `(Σ_{j≥1} records_j + tombstones_head) / records_base`:
the change accumulated since the base's calibration was fit. It counts
history, not current divergence: a record appended and later tombstoned
counts twice, and a supersession counts as one append plus one tombstone.
This is intentionally conservative — the fit's population lost one member
it had seen and gained one it had not. It is a measure of change volume
and nothing more; section 5.5 says what it gates and 4.5/5.4 say what it
does not bound.

### 3.2 Identity and availability

The identity of a layer is its `manifestSha256`, exactly as for a complete
artifact. The **chain identity** is the head's identity. A reader that
pins a chain pins the head; `openPikeletFile(url, { expectedIdentity })`
and the `url#<sha256>` fragment keep their meaning unchanged.

A layer's manifest MUST carry `layer.parent.identity` and
`layer.baseIdentity`. A reader MUST refuse to serve a chain in which any
opened parent's identity differs from the identity its child committed
to. Locators (URLs, paths) in a manifest or a shelf are hints for finding
bytes; they are never trusted for what the bytes are.

Identity completeness and storage availability are different properties.
The head cryptographically commits to every ancestor; it does not make
them reachable. A reader that cannot fetch an ancestor MUST fail the
mount explicitly — never serve the layers it did reach as a partial
corpus, and never substitute another head it has cached. A host that
deletes a layer breaks every head above it; section 6.4 states the
retention rule.

### 3.3 Row arithmetic and the id space

For every layer a reader MUST verify, against the parent it actually
opened:

```
layer.rowBase == parent.layer.rowBase + parent.corpus.records     (parent a layer)
layer.rowBase == parent.corpus.records                            (parent the base)
layer.depth   == parent.layer.depth + 1                            (base depth = 0)
layer.baseIdentity == parent.layer.baseIdentity                    (or parent identity, for the base)
```

and MUST refuse the chain on any disagreement. `rowBase` is not unique
along a chain: a tombstone-only layer has `records == 0`, so the layer
after it has the same `rowBase`. Id ownership is therefore defined by
containing interval, not by `rowBase` (5.2).

A chain holds **at most `2^31 − 1` records** (the reader's `MAX_RECORDS`),
with global ids `0 … 2^31 − 2`; every `rowBase`, `records` sum and
`rowTotal` is bounded by `2^31 − 1`, and every id by `2^31 − 2`. Ids are
never reused within a chain, so a chain consumes id space monotonically
and only compaction reclaims it. A producer MUST refuse an append whose
`rowBase + records` would exceed `2^31 − 1`, and SHOULD warn once a
chain's row total passes `2^30` so that compaction can be scheduled before
appends start failing. There is no other mechanism.

### 3.4 Tombstones

The bitset in layer `j`, `T_j`, covers `[0, rowBase_j)`, one bit per
global id, LSB-first within each byte (bit `r & 7` of byte `r >> 3`).

**`zeroExtend`.** Parent and child masks have different lengths: the
parent's own records become addressable by the child's mask. Define
`zeroExtend(T_parent, rowBase_child)` as `T_parent` followed by zero bits
over `[rowBase_parent, rowBase_child)`; for a base parent,
`zeroExtend(∅, rowBase_child)` is all zeros. Then:

- **Superset (cumulative) rule:** `T_child ⊇ zeroExtend(T_parent,
  rowBase_child)` — every bit set in the extended parent mask is set in
  the child. The newly addressable range starts clear and may acquire
  deletions.
- **Delta:** `ΔT_child = T_child ∖ zeroExtend(T_parent, rowBase_child)`,
  the deletions the child itself introduced. `ΔT` is what a rebase
  replays (6.2) and what a layer's `--remove` and `--supersede` produce
  (6.1); it is derived, not stored.

A reader SHOULD check the superset rule at mount when it has read both
bitsets and MUST check it in a full verification pass (5.7). A layer MUST
NOT set a bit for a row it owns (`r ≥ rowBase_j` is out of range by
construction) — a layer cannot retract its own records; the next layer
can.

**Scaling.** The cumulative bitset costs `⌈rowBase/8⌉` bytes in every
layer: 57,020 bytes at the Wikipedia pack's 456,153 records, 125,000 at
one million, 12.5 MB at one hundred million. The choice is deliberate —
lookup is a bit test, only the head's bitset is needed to serve, and the
superset check is a linear pass — and compaction bounds how many copies
exist. Where the per-layer cost stops being negligible, a compressed
encoding is `tombstones-v2`, a layout version of the segment, not a
change to this profile; 5.9's budgets bound the reader either way.

A layer MAY be **tombstone-only** (`layer.records == 0`): a pure deletion.
Section 4.3 gives its segment requirements.

### 3.5 Supersession

A layer MAY record that a new record replaces a tombstoned one, as
`(oldId, newId)` pairs in the tombstone segment. `oldId` MUST be set in
this layer's bitset (by this layer or an ancestor); `newId` MUST lie in
this layer's own id range; within one layer each `oldId` appears at most
once. The supersession list is **per layer**, never cumulative: a layer's
segment carries only the edges that layer recorded, and the reader forms
the union at mount. The relation is informational — it changes no search
result — and is surfaced by `record(id)` (5.6) so a consumer holding an
old citation can follow it.

**Chaining.** Because `newId ≥ rowBase_j > oldId`, edges always point to
higher ids and cannot form cycles. An id may acquire outgoing edges in
more than one layer (a rebase with `keep-both`, 6.2, is the only producer
path that creates this). Edges are ordered by the **depth** of the layer
that recorded them — never by segment order, file order or id — and
within one layer an `oldId` has at most one edge, so "earliest" and
"latest" are always unique. The reader exposes three views: the
**immediate** successor — the edge recorded at the least depth; **all**
edges, in ascending depth order; and the **current** successor — the id
reached by starting from the edge recorded at the greatest depth and
repeatedly following each reached id's greatest-depth edge until an id
with no outgoing edge is reached. Current-successor resolution is a walk
over the union supersession map, which is small and resident, and is
fully deterministic given the chain.

## 4. Layer file layout

A layer is a complete-profile container (`COMPLETE_PROFILE.md` section 3):
64-byte header, canonical-JSON manifest, 48-byte segment-table entries,
16-byte-aligned segments. The header's `formatVersion` MUST be `2`; a
layer MUST use corpus layout v2 (per-record digests). This section lists
only what a layer adds or constrains.

### 4.1 Manifest

```jsonc
{
  "profile": "pikelet-layer-v1",
  "layer": {
    "parent": {
      "identity": "9f3c…",                 // 64 hex; REQUIRED
      "locator": "docs.0007.9f3c1a2b.pikelet" // OPTIONAL relative reference (5.1.1); never identity
    },
    "baseIdentity": "8d73…",               // identity of the depth-0 artifact
    "depth": 8,                             // parent.depth + 1; 1 <= depth <= 16 (2.7)
    "rowBase": 456153,                      // first global id owned by this layer (3.3)
    "records": 87,                          // == corpus.records; 0 for a tombstone-only layer
    "tombstones": 12,                       // popcount of the cumulative bitset
    "supersessions": 3,                     // number of (oldId, newId) pairs recorded by THIS layer
    "ingest": { … }                         // the chain's ingestion declaration (6.1); copied from the base or asserted
  },
  "corpus": { "records": 87, "layout": "records-v2", "pageRecords": 256, "pages": 1,
              "recordDigest": "sha256", "pageTableSha256": "…", "provenance": { … } },
  "dim": 384,
  "metric": "cosine",
  "index":   { "headerSha256": "…" },      // as complete-v2; absent on a tombstone-only layer
  "encoder": { … },                        // verbatim copy of the base's `encoder` (decision 5)
  "segments": [
    { "kind": "index",        "sha256": "…", "bytes": 41984 },
    { "kind": "corpus",       "sha256": "…", "bytes": 83120 },
    { "kind": "query-interp", "sha256": "…", "bytes": 4210 },
    { "kind": "evaluation",   "sha256": "…", "bytes": 1188 },
    { "kind": "lexical",      "sha256": "…", "bytes": 15360 },
    { "kind": "tombstones",   "sha256": "…", "bytes": 57076 }
  ]
}
```

Rules:

- `encoder` MUST equal the base's `encoder` object under canonical JSON
  serialization, and `dim` and `metric` MUST equal the base's. A reader
  MUST check these at mount and refuse on difference. These checks are
  consistency checks on a self-describing manifest; the authoritative
  encoder invariant is the digest commitment of 4.5. If the two ever
  disagree, the manifest is the defect and the reader refuses.
- `corpus.records` MUST equal `layer.records`. A **tombstone-only layer
  still carries a `corpus` object**, with `records: 0` and `provenance`,
  and MUST omit the layout fields (`layout`, `pageRecords`, `pages`,
  `recordDigest`, `pageTableSha256`) and the `index` object, because it
  has no corpus or index segment to describe (4.3). Keeping the object
  means the manifest parser has one shape for every layer and the
  existing "manifest has no corpus block" check stays as it is.
- `layer.ingest` MUST be present and MUST equal, under canonical JSON,
  the **chain ingestion declaration**, defined as: the base's
  `corpus.ingest` when the base carries one; otherwise the depth-1
  layer's `layer.ingest`, and that layer MUST then carry the sibling
  field `layer.ingestAsserted: true`. Every layer at depth ≥ 2 MUST carry
  a canonical-JSON-identical `layer.ingest` and, on an asserted chain,
  `layer.ingestAsserted: true`. Only the depth-1 layer of a chain whose
  base lacks `corpus.ingest` can introduce an assertion; a later layer
  MUST NOT introduce, drop or change one, and a reader MUST refuse a
  chain in which the declaration differs anywhere along it (6.1). This is
  what lets a reader tell from any layer how the chain's records were
  made, and lets mount validation be one equality per layer.
- **Schema of the two fields.** `layer.ingest` is the declaration itself
  — the object that canonical equality is computed over — and contains
  only ingestion parameters. `layer.ingestAsserted` is a boolean
  **sibling** of `layer.ingest` inside the `layer` object, recording
  provenance (an operator asserted the declaration rather than the base
  declaring it). It is never a member of the declaration, so asserting a
  declaration does not change its canonical form and a compacted base's
  `corpus.ingest` (6.3) is byte-identical to the declaration the chain
  carried. `layer.ingestAsserted` is `false` or absent on a chain whose
  base carries `corpus.ingest`.
- `sampleQueries`, `recommendedRerank` and `recommendedGap` MAY appear and
  apply to this layer's own index segment only.
- Unknown fields are ignored and identity-covered, as in the complete
  profile.

A reader that predates this profile sees `profile: "pikelet-layer-v1"`
under `formatVersion` 2 and refuses with its existing
"unsupported profile" error. That is the intended behavior: a layer opened
alone would silently serve a fraction of a corpus, so the rejection MUST
be explicit (contract section 6). Readers implementing this profile accept
two profile strings under `formatVersion` 2.

### 4.2 Segment table

The complete profile's kinds are unchanged: `1` index, `2` corpus,
`3` query-interp, `4` evaluation, `5` lexical. This profile registers:

| kind | name | where |
| ---: | --- | --- |
| 6 | tombstones | a layer: REQUIRED, exactly one (4.4) |
| 7 | lineage | a compacted base: REQUIRED when compacted from a chain, at most one (6.3) |

Unknown kinds remain skippable under the complete profile's rule (a
manifest entry for an unknown kind number must not claim a known kind's
name). `segmentCount` stays bounded by 64.

### 4.3 Segment requirements (normative for this profile)

A layer is not a complete artifact and is not held to the complete
profile's "exactly one of kinds 1–3" rule. Its requirements are:

| segment | layer with records ≥ 1 | tombstone-only layer (records = 0) |
| --- | --- | --- |
| index (1) | REQUIRED; `count == layer.records` | MUST be absent |
| corpus (2) | REQUIRED; layout v2 | MUST be absent |
| query-interp (3) | REQUIRED; kind 4 (4.5) | REQUIRED; kind 4 |
| evaluation (4) | REQUIRED for conformance | REQUIRED for conformance; `{"layer":true,"records":0}` is sufficient |
| lexical (5) | REQUIRED if the base carries one, else MUST be absent | MUST be absent |
| tombstones (6) | REQUIRED | REQUIRED; `tombstoneCount` MUST exceed the parent's |

A tombstone-only layer omits segments rather than carrying empty ones
because the sketch reader rejects `count < 1` and an empty corpus segment
has nothing to commit to; this is a rule of this profile, not an
exception to the complete profile, which never sees a layer. A reader
MUST reject a records-≥-1 layer missing a required segment and a
tombstone-only layer carrying a forbidden one.

A chain is hybrid (BM25 + vector) if and only if its base is; a layer MUST
NOT introduce or drop lexical retrieval, because the fused ranking and the
calibration were fit under one retrieval mode.

### 4.4 Tombstone segment (kind 6), layout `tombstones-v1`

```
[0,4)    u32 version = 1
[4,8)    u32 flags (0; readers MUST reject non-zero)
[8,16)   u64 rowBase             MUST equal manifest layer.rowBase
[16,24)  u64 tombstoneCount      popcount of the bitset; MUST equal manifest layer.tombstones
[24,28)  u32 supersessionCount   MUST equal manifest layer.supersessions
[28,32)  reserved, zero
[32, 32 + ceil(rowBase/8))       bitset, LSB-first (3.4); pad bits above rowBase MUST be zero
[...]    supersessions           supersessionCount x (u32 oldId, u32 newId), sorted by oldId ascending, oldId unique within the segment
```

The regions MUST tile the segment exactly. A reader MUST reject a segment
whose bitset length disagrees with `rowBase`, whose popcount disagrees
with `tombstoneCount`, whose pad bits are non-zero, or whose supersessions
violate 3.5 (`oldId < rowBase` with its bit set; `newId` within this
layer's own range; sorted; unique). The segment is read eagerly at mount
and verified against its manifest digest; its size is stated in 3.4.

### 4.5 Query-interpretation kind 4 — `inherited-v1`

The segment header is the complete profile's (`version`, `kind = 4`,
`encoderBytes`, `calibrationBytes`). The encoder region is UTF-8 JSON:

```jsonc
{
  "inheritFrom": "8d73…",          // MUST equal manifest layer.baseIdentity
  "queryInterpSha256": "…"         // MUST equal the base manifest's segments[kind=query-interp].sha256
}
```

`queryInterpSha256` is the encoder invariant of decision 5: it commits to
the base's encoder bytes and calibration together, under the base's
identity, which the layer's `baseIdentity` in turn commits to. A reader
MUST resolve the encoder from the base it actually opened, MUST check
`queryInterpSha256` against the base's manifest entry, and MUST NOT load
encoder bytes from a layer. The base's own kind (1, 2 or 3) governs
verification obligations for the chain: a kind-2 base still requires a
host encoder verified against the base's test vectors before the first
query. (A kind-1 base can be *read* as the base of a chain but not
appended to — 6.1.)

The calibration region is UTF-8 JSON, one of:

```jsonc
{ "kind": "layer-vocab-v1",
  "vocabBloom": { "bits": 2097152, "hashes": ["fnv1a:0", "fnv1a:0x9e3779b9"] },   // MUST match the base asset's vocabBloom geometry
  "vocabBloomBase64": "…" }                                                          // this layer's records' vocabulary
```

or `{ "kind": "none" }`. A layer's bloom is over its own records'
vocabulary, built with the base's `vocabBloom.bits`, hashes and `minCount`
(from `retrieval-signals-v1`'s asset). A tombstone-only layer has no
records and therefore no vocabulary; its `none` region is equivalent to an
empty bloom, and a producer SHOULD ship `none` there rather than an
all-zero bloom the size of the base's.

**What the union means.** The chain's vocabulary bloom is the bitwise OR
of the base's and every layer's bloom, which the identical geometry makes
exact. That union is a bloom over the vocabulary of **every record ever
appended to the chain**, live or tombstoned — the known-token signal
answers "has this chain ever indexed this word", not "does the live corpus
contain it". It is not, and cannot be made, a bloom over the live
vocabulary: nothing can be removed from a bloom. The error is one-sided:
words that survive only in tombstoned records still read as known, so the
scorer's known-token fraction over-reads for a query about deleted content;
it never reads a live word as unknown, because a bloom can only gain bits.

**Which way that over-read pushes depends on the sign of `known_frac`'s
weight in the base's fit, and in the shipped model it is fail-safe.** Drafts
1-4 asserted the over-read makes abstention *less* likely. Measured against
the shipped `retrieval-signals-v1` stage-2 model that is backwards: the
weight is negative (see 13.2 for the numbers), by design — a query whose
vocabulary the corpus knows well but whose answer retrieval cannot find is
the profile of an in-domain unanswerable question, so high `known_frac`
pushes *toward* abstention. Since pollution can only raise `known_frac`, it
can only push further in that direction. A producer or reader MUST NOT rely
on the *outcome* here without checking the sign: a future refit that gives
`known_frac` a positive weight restores the lenient reading, and the
profile makes no guarantee either way. What it does guarantee is that the
error is one-sided in `known_frac` itself.

The retrieval signals (`d0`, margin, `mean10`) are computed over live
candidates only, since tombstoned rows never enter candidate generation
(5.3), so such a query still carries poor distance evidence — which is why,
in the measurement of 13.2, the pollution changed no verdict in either
direction on the great majority of probes: the distance evidence, not the
bloom, decides a query about deleted content.

**What the drift gate does and does not say about it.** The pollution is
*permitted* only while `calibrationStatus` is `inherited` (5.5); a chain
appended past the effective limit with `--allow-drift` (6.1) is
`drift-exceeded`, where the scorer no longer runs, and the union keeps
accumulating until compaction rebuilds it from live records alone. The
gate is a change-volume gate: **no analytical bound on calibration error
is implied by it.** A single record with a large, unusual vocabulary can
set a substantial fraction of the bloom's bits while record-count drift
is negligible. What the drift limit buys is a policy under which the
inherited fit is *allowed* to be served; how the fit actually degrades is
section 13's first measurement, not a theorem of this profile.

A reader MUST compute the union before scoring: without it the signal
under-reads every query about a layer's content, and the scorer abstains
on exactly the records the layer was published to add. A layer **with
records** whose calibration region is `none` while the base carries a fit
degrades the chain to `unscored` (5.5), so a producer SHOULD always ship
the bloom on such a layer. A tombstone-only layer's `none` does not
degrade the chain: it hides no vocabulary. (Drafts 1-4 applied the rule to
every layer, so a single deletion unscored the whole chain.)

### 4.6 Index, corpus and lexical segments

Unchanged layouts. Constraints:

- **Index:** a `.pikelet-sketch` (format 2 REQUIRED) over the layer's
  own rows, `dim`/`metric` equal to the base's. Sketch geometry
  (`sketchDims`, `sketchBits`, staged tier, `rowsPerBlock`) SHOULD equal
  the base's and MAY differ; each search tier is scanned by its own geometry.
  `recommendedRerank` is measured for this layer alone and is a property
  of the layer's sketch, not of any head: at publication every row a
  layer owns is necessarily live, since a layer cannot tombstone its own
  range, and a layer cannot know what a descendant will delete. At query
  time `C` counts **unmasked** rows (5.3): the exclusion mask changes
  which rows may consume the `C` slots, not the recommendation. A layer
  with `records <= 1024` SHOULD set it to `records`, making the layer's
  rerank exact and its evaluation trivial.
- **Corpus:** layout v2 over the layer's own records; local ids.
- **Lexical:** layout `bm25-v1` over the layer's own records with local
  doc ids; `docCount == layer.records`. Tokenization MUST equal the
  base's. Global statistics are the reader's job (5.4); a layer's header
  carries only its own `docCount` and `totalTokens`.

### 4.7 Evaluation segment

UTF-8 JSON. REQUIRED fields: `layer: true`, `records`, and — when
`records ≥ 1` — either `exact: true` (rerank covers every row) or
`recallVsC` for the layer's own sketch geometry, measured against brute
force over the layer's rows. These are **local** measurements: they
depend only on the layer's own rows and survive a rebase unchanged. The
segment MAY additionally carry **chain-level** golden queries (queries
with expected global ids as of this layer), which become conformance
fixtures for the chain and which a rebase MUST regenerate or drop (6.2),
never copy.

## 5. Execution semantics

### 5.1 Mount

`open(head, options)`:

1. Open `head` as a complete container (complete profile section 4,
   steps 1–2): header, manifest, identity, `expectedIdentity` if given.
   If the manifest has no `layer` object, this is a base; serve it as
   today. Otherwise it is a layer; continue.
2. Resolve the parent under 5.1.1, open it the same way, and verify its
   identity equals `layer.parent.identity`. Repeat until a base is
   reached. Refuse if depth would exceed 16, if a parent cannot be
   located or fetched (3.2), or if any identity, row-arithmetic, segment
   or encoder check of sections 3.3–3.4, 4.1, 4.3 and 4.5 fails.
   With a lineage listing, every layer's identity and location is known
   up front and layers open in one parallel wave; following locators
   costs one dependent round trip per layer, which is one reason the
   depth limits are small.
3. For each chain member, complete the complete-profile open as its
   segments require (steps 3–4: encoder or kind-4 inheritance; for each
   search tier the sketch resident prefix, corpus tables, lexical header
   and doclens), and for each layer read and verify the tombstone
   segment.
   Preflight the chain budgets of 5.9 from the declared sizes before the
   first segment read, then charge every read as it is issued.
4. Build the chain state: the ordered interval table of search tiers
   (5.2), the ancestry (identity → `rowTotal`, depth) for citation scoping
   (5.6), the head's bitset and its projection onto every search tier's
   local range (5.3), the
   union supersession map, the union vocabulary bloom, global lexical
   statistics (5.4), and the calibration status (5.5).
5. The chain is serving. `info()` reports the head identity, the base
   identity, `layers` (depth + 1), live and total record counts,
   `tombstones`, `calibrationDrift`, `driftLimit` (the effective one, and
   both inputs), `calibrationStatus`, and per-layer identities.

Every read obeys the complete profile's bounded-read rules; nothing here
relaxes a per-read budget, and 5.9 adds aggregate ones.

#### 5.1.1 Parent resolution

Identity verification prevents substitution of bytes; it does not
prevent a hostile manifest from directing a reader's fetches. Resolution
is therefore constrained, and every check below is applied to the
**canonical** form of a location — for URLs, after percent-decoding of
unreserved characters and removal of dot segments per RFC 3986 §5.2.4,
with a lowercased scheme and host; for file paths, after resolving to a
real path with every symbolic link followed — never to the string as
written, so that `%2e%2e`, a doubled slash, a case-varied host, or a
symlink escaping the directory cannot pass a textual check:

- `layer.parent.locator`, when present, MUST be a relative reference
  (RFC 3986 `relative-ref`) with no scheme and no authority, and after
  canonicalization MUST resolve to a location under the child's own
  directory. A reader MUST reject a locator that violates this, at the
  manifest, before any fetch.
- A locator resolves against the *child's own location*: for a URL
  source, the canonical result MUST have the child's origin (scheme,
  host, port) and a path under the child's directory; for a file source,
  a real path under the child's real directory. A reader MUST NOT fetch a
  parent from any other origin or directory on the strength of a locator.
- Redirects encountered while fetching a parent are subject to the *same*
  confinement as the locator, not a weaker one: every redirect target
  MUST, after canonicalization, retain the child's origin and scheme and
  resolve beneath the child's canonical directory, and the check is
  repeated at every hop before the redirected request is issued. A
  same-origin redirect to a path outside the directory
  (`/packs/foo/head.pikelet` → `/admin/private-object`) fails the mount
  exactly as a cross-origin one does.
- A locator or redirect target MUST be rejected if, before or after
  decoding, its path contains an encoded path separator (`%2f`, `%2F`),
  an encoded or literal backslash (`%5c`, `%5C`, `\`), or an encoded dot
  segment. Normalization defects live in the disagreements between a
  validator and the server that finally interprets the path, so the
  reader refuses every form on which the two could disagree rather than
  picking one reading.
- A host MAY disable locator resolution (`resolveParents: false`), in
  which case parents are located only through a caller-supplied lineage
  listing (section 7) or a host resolver. Lineage listings are
  operator-supplied input in the same sense as a shelf's existing `url`
  entries — the operator chose the shelf — and MAY name any origin the
  operator trusts; manifest locators are publisher-supplied and get the
  constraints above.
- Resolution order is: lineage listing, then locator, then host
  resolver. Whatever located the bytes, the identity check of 3.2 is what
  admits them.

### 5.2 Id resolution

`owner(id)`: the unique search tier `j` (3.1; the base is `j = 0`)
whose interval `[rowBase_j, rowBase_j + records_j)` contains `id`; local
id `id − rowBase_j`. Tombstone-only layers own no ids and take no part in
resolution. Because consecutive tombstone-only layers give the next layer
the same `rowBase` (3.3), `rowBase` is not a unique key over chain
members: an implementation that binary-searches MUST search the intervals
of the search tiers only (their `rowBase` values are strictly increasing,
since each owns at least one id), never "the largest `rowBase ≤ id`" over
all members. Ids outside every interval are invalid and MUST be rejected.
The table has at most 17 entries.

### 5.3 Query

The mask comes first. For every **search tier** `j` (3.1) — the base,
`j = 0`, included, which is where most tombstones point — the reader
holds the head's bitset projected onto `j`'s local range,
`mask_j[i] = T_head[rowBase_j + i]`, computed once at mount. A
tombstoned row MUST NOT enter a candidate heap, MUST NOT count toward a
tier's rerank `C`, MUST NOT be fetched for reranking, and MUST NOT occupy
a slot in the lexical candidate list before its cap. Masking *after*
truncation is a defect, not a simplification: if the `C` nearest rows of
a tier are all tombstoned and the nearest live row is the `C+1`-th, a
post-hoc mask returns nothing for that tier while the correct answer
exists, and the drift needed to arrange that is one deletion per
candidate slot. Conformance fixture (A) contains exactly this case, on
the base, for both the vector and the lexical path.

`query(text, k, options)` executes the complete profile's steps with these
substitutions:

1. **Encode** once with the base's encoder.
2. **Lexical candidates** (hybrid chains): for each query term, look the
   term up in every search tier's term table concurrently; sum `df`
   across tiers; compute `idf` and `avgdl` from the global statistics of
   5.4; score each tier's postings with those global values, **skipping
   tombstoned docs as postings are scored**; map local doc ids to global
   ids; merge; then apply the reader's existing relative cutoff
   (`LEXICAL_CUTOFF`) and candidate cap (`LEXICAL_CANDIDATES`) to the
   merged, live-only list.
3. **Vector candidates**: run each search tier's sketch search
   concurrently with `fullRerankOutput`, that tier's `recommendedRerank`
   (or the caller's `rerank`, subject to 5.9's candidate budget), **the
   tier's exclusion mask supplied to candidate generation** (the sketch
   reader's scan skips masked rows, so `C` counts unmasked rows only), and
   as `extraCandidates` the lexical hits that tier owns, translated to
   local ids. Map results to global ids, merge ascending by distance;
   equal distances break by ascending global id (the reference reader's
   declared tie rule for chains). This merged list is `searched`, and it
   contains no tombstoned id by construction.
4. **Fuse** with `fuseCandidates` exactly as today over `searched` and the
   merged lexical order. `fusedRank`, `lexicalRank`, `vectorRank` refer to
   the merged lists.
5. **Calibrate** (5.5) over `searched` and the fused window, hydrating
   coverage passages from their owning tiers.
6. **Hydrate** each returned id from its owning tier's corpus segment
   with per-record verification.

Result shape is unchanged. Each result additionally carries `layer`
(depth of the owning tier; `0` for the base) so a consumer can see which release a record
came from, and the response carries `identity` (the head) so a consumer
can form the citation of 5.6 without a second call.

### 5.4 Global lexical statistics

For a hybrid chain the reader computes, once at mount:

```
N        = Σ_j docCount_j           over search tiers j = 0 … m (base included)
avgdl    = Σ_j totalTokens_j / N
df(t)    = Σ_j df_j(t)              (per query term, at query time)
idf(t)   = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
```

Tombstoned records are counted in `N`, `avgdl` and `df`: removing them
would require reading every posting list at mount. The resulting scores
differ from a from-scratch build over the live set. **This approximation
is permitted while the drift gate of 5.5 is within policy; no analytical
bound on ranking error is implied by the gate.** Deleting one document
that holds a rare term changes that term's `df` drastically at negligible
record-count drift, so "small drift" does not mean "small BM25 error"; it
means "little has changed by volume". Compaction restores the exact
statistics. **Measured (13.3): across 24 configurations on two datasets, at
up to 25% drift and including a deletion policy built to maximise `df`
distortion, the aggregate nDCG@10 difference against a from-scratch live-set
build never exceeded 0.005.** The reasoning above still holds — no bound is
implied, and single queries do move — but the effect did not appear at that
scale, and the per-layer `df` correction table 13.3 floated is not part of
this design. A reader MUST report `info().lexical.statsIncludeTombstoned =
true`.

Per query term the cost is one term-table window read per search tier
(12 KiB windows in the reference lazy opener), issued in parallel, plus
one postings read per tier that holds the term. The reader uses every
unique query token, exactly as the single-file reader does — a chain
MUST NOT interpret a query differently from the base it extends; a query
whose reads would exceed 5.9's lexical ceiling fails explicitly, naming
the budget, rather than silently dropping terms.

### 5.5 Calibration drift and status

```
calibrationDrift = (Σ_{j≥1} records_j + tombstones_head) / records_base     (3.1)

effectiveLimit   = min(producerEnvelope, readerLimit)

calibrationStatus =
  'none'            the base carries no fit (unscored today; unchanged)
  'inherited'       calibrationDrift <= effectiveLimit and every layer with records
                    ships layer-vocab-v1 (a tombstone-only layer's `none` is the
                    empty bloom, 4.5)
  'drift-exceeded'  otherwise
```

Two numbers, with a fixed precedence and no exceptional cases:

```
producerEnvelope = base.calibration.driftLimit ?? +∞     (no claim = no constraint)
readerLimit      = the host's configured value, else the reader's declared default
                   — MUST be finite; a conforming reader MUST declare its default
                   and report it in info()
```

`producerEnvelope` is the base producer's declared validity envelope —
its claim of how much change its fit tolerates. `readerLimit` is the
host's policy. The effective limit is the stricter of the two: **a host
can always be stricter than the artifact, and an artifact can never make
a host less strict.** Because `readerLimit` is finite, an artifact that
declares nothing is still gated. (Draft 1 had the precedence the other
way, letting a manifest loosen a host's policy.)

The reference reader's `readerLimit` default is **0.20, non-normative**. It
was chosen before any measurement existed, on the suspicion that content or
domain shift might matter far more than the fraction of records changed, in
which case the gate's *input* would change (a drift measure that sees what
changed, not how much) without any change to the format, because nothing
about the gate lives in a layer's bytes. **13.1 has since measured the
inherited fit out to 40% drift, twice this limit, under both random and
topically concentrated change, and found no degradation** — so 0.20 stands
as a conservative default rather than a guess, and the shift-aware
alternative is not currently motivated. That measurement is one dataset at
one seed on a ~1,800-record base, so it retires the suspicion at that scale
and not at wiki-pack scale; the limit remains a reader policy, not a format
constant, precisely so it can be revisited without a format change.
Conformance tests exercise the gate at whatever limit the reader under
test declares, never at the number 0.20.

Under `inherited`, the base's `retrieval-signals-v1` scorer runs
unchanged over the merged `searched` window and fused passages, with the
union bloom of 4.5. Under `drift-exceeded`, `matchQuality` is `unscored`,
`confidence` is omitted, results still ship, and the response carries
`calibration: { status: 'drift-exceeded', drift, effectiveLimit,
producerEnvelope, readerLimit }` so a consumer can tell "no fit" from
"fit outgrown" and see which input bound it. A reader MUST NOT apply an
outgrown fit silently.

### 5.6 `record(id)` and citations

A **bare id is an id in the mounted head**, nothing else. `record(id)`
hydrates it from the owning layer with verification, for any valid id,
live or not, and adds `tombstoned: boolean` and, when tombstoned:

```
supersededBy:      immediate successor (3.5) or null
successors:        every recorded edge from this id, in ascending depth order
currentSuccessor:  the current successor (3.5), or null
```

A **citation is `(identity, id)`**, where `identity` is the head the
result was served from — every search response carries it (5.3), and the
MCP surface already emits `packIdentity` with every result. An id alone is
not a citation, because an id names one record only along one ancestry:
two children of one parent assign the same id range to different text.
`record({ identity, id })` therefore resolves by this rule:

1. `identity` is in the mounted head's **ancestry** (3.1): the cited
   record is on this history. Validate `id < rowTotal(identity)` — the id
   existed when that artifact was the head — then resolve `id` exactly as
   a bare id; its current tombstone and supersession state is reported
   as above. A citation made against `H3` still resolves under `H8`.
2. `identity` names a head that a **compaction retired**, and the mounted
   base carries a lineage segment listing it: translate per 6.3.
3. Otherwise — a fork not on this history, or an identity the reader has
   never seen — the reader MUST answer "not on this history" and MUST NOT
   resolve the bare id as if it were current. Guessing here is exactly
   the provenance failure the pair exists to prevent.

The MCP `get_record` tool takes an optional `identity` and applies the
same rule; without it, `id` means the mounted head. Immediate historical
edges are never rewritten; the current successor is derived.

### 5.7 `verify()`

A chain-level verification pass MUST: verify each layer's identity and
its parent commitment; the row arithmetic of 3.3; the `zeroExtend`
superset rule of 3.4 for every layer over its parent; the segment
requirements of 4.3; the encoder consistency of 4.1 and the digest
commitment of 4.5; the `ingest` equality of 4.1; and, per layer,
everything the complete profile's verification does (manifest, header
commitments, per-row and per-record digests on request, `verifyVectors`
on demand). On a compacted base it MUST additionally validate the lineage
segment per 6.3. The MCP `verify_pack` tool runs this pass and reports
per layer.

### 5.8 Caching

A reader MAY cache per-layer state across queries and across mounts of
different heads that share layers (a new head shares every layer but the
last with its predecessor; a reader that keeps opened layers keyed by
identity mounts an update by opening one small file). Projected masks are
per head, cover every search tier including the base, and MUST be
recomputed when the head changes. Cache state MUST
NOT change results.

### 5.9 Chain resource budgets

Sixteen individually well-formed layers can multiply an acceptable
per-file cost into an unacceptable one. The complete profile's per-read
limits (256 MiB per open-path read, 16 MiB per record, 2 GiB absolute)
remain per read; this profile adds aggregate limits that a reader MUST
enforce across the whole chain and MUST refuse the mount or query on,
naming the budget. The reference reader's defaults are provisional:

| budget | scope | kind | reference default |
| --- | --- | --- | --- |
| `maxChainDepth` | layers above the base | format ceiling | 16 (not configurable upward) |
| `maxChainSegments` | Σ segment-table entries | structural ceiling | 64 × 17 = 1,088 (what the container can express) |
| `maxChainSegmentsPolicy` | Σ segment-table entries | operational default | 8 × (depth + 1); a layer carries 3–6 |
| `maxChainResidentBytes` | Σ resident prefixes + corpus tables + lexical headers/doclens + bitsets + projected masks | operational | 768 MiB |
| `maxChainMountBytes` | Σ bytes read during mount | operational | 1 GiB |
| `maxChainTombstoneBytes` | Σ tombstone segments read | operational | 64 MiB |
| `maxChainCandidates` | rows fetched for one query's rerank: Σ_j (live `C_j` + lexical `extraCandidates_j` not already among them) | operational | 8,192 (each layer also under `SCANNER_MAX_RERANK`) |
| `maxChainRowFetchBytes` | Σ bytes fetched by one query's rerank round | operational | 64 MiB |
| `maxChainLexicalReads` | term-table windows + postings reads for one query | calculated allowance **and** absolute ceiling | allowance `(depth + 1) × 2 × uniqueTerms`; ceiling 2,048 reads — a query that would exceed it fails explicitly (5.4) |

Ceilings are format facts: the structural segment ceiling is simply what
64 entries × 17 files can express, and it hardens nothing by itself. The
operational defaults are what the reference reader refuses on, and hosts
serving public heads SHOULD lower them to what their corpora need. Where a
budget is a calculated allowance, its inputs (`depth`, the query's unique
terms) are what they are, and the absolute ceiling applies on top; at the
depth ceiling the lexical ceiling admits queries of up to 60 unique terms,
and at the producer's default depth of 8, 113 — a failure there is a
resource failure reported as such, not a changed interpretation of the
query.

Enforcement is in two stages, and both are REQUIRED. **Preflight:**
everything a declared size makes knowable is checked before the read that
would consume it — the header's `fileBytes` and a source's reported size,
every manifest `segments[].bytes`, `corpus.records`, `layer.rowBase`
(hence bitset length), the sketch header's `count` and geometry (hence
resident prefix), and `recommendedRerank` — and a chain whose declared
sizes already exceed a budget is refused before those bytes are
requested. **Metering:** every read actually issued is charged, so a
chain that declares small sizes and serves large ones fails at the read
that crosses the line; a short or over-long read is a failure, not a
partial success (complete profile section 4). Preflight bounds what an
honest declaration can cost; metering bounds what a dishonest one can.

## 6. Producer semantics

### 6.1 `append`

```
pikelet append --parent <file|url>[#identity] [--lineage <shelf>] \
               [--source <path|url> ...] [--remove <id|selector> ...] [--supersede <oldId>=<sourcePath> ...] \
               [--max-depth <n>] [--allow-drift] [--assert-ingest <file>] --out <file>
```

**Readable versus appendable.** Every complete artifact is a valid base
to *read* under section 5. Appending to one requires two things the
complete profile never promised:

- an encoder that can embed **passages**. A kind-3 base carries the
  teacher, and `append` loads it from the base's own query-interp segment
  — no model download, no external dependency: a pack is sufficient to
  compile its own successors. A kind-2 base names its encoder; `append`
  uses the host's copy, verified against the base's test vectors first,
  as at mount. A **kind-1 base carries a corpus-distilled *query*
  encoder** whose passages were embedded by a teacher the artifact does
  not carry or verify, so a kind-1 base is **readable but not
  appendable** under this profile; `append` MUST refuse it.
- an **ingestion declaration**: how the base's records were made
  (chunker identity and version, target and minimum sizes, section and
  heading handling, title prefixing, exclusions — whatever the compiler
  version's ingestion is parameterized by), because record granularity
  is part of what the calibration saw and MUST NOT change mid-chain.
  A base compiled after this profile carries it as `corpus.ingest`.
  Bases released before it record only `source`, `name` and `license`
  (`complete-build.mjs`), which is not enough to enforce the rule, so for
  such a base `append` MUST refuse unless the operator supplies
  `--assert-ingest`, whose contents are recorded as `layer.ingest` with
  the sibling `layer.ingestAsserted: true` in every layer of the chain
  (4.1 gives the inheritance rule and the schema), and a reader reports
  `info().ingestAsserted`. The assertion is the operator's
  statement, not the artifact's; a chain built on a wrong assertion is
  a conforming chain with an unmeasured calibration, which is the
  operator's to answer for.

1. Mount the parent chain (section 5). Refuse if `--parent` carried an
   identity that does not match; if the base is not appendable as above;
   if the parent's depth is already `--max-depth` (default 8 — an
   operational recommendation until the range-proof measurements of
   section 13 say otherwise; the format ceiling of 16 is enforced
   regardless); or if the append would exhaust the id space (3.3).
2. Ingest and chunk `--source` inputs under the chain's ingestion
   declaration. Records are not deduplicated (decision 9): a producer
   that wants "append only what is new" compares candidate records'
   digests against the chain's live `recordSha256` values as its own
   policy, and the profile neither requires nor forbids that.
3. Embed the new records with the chain's encoder.
4. Assign ids from `rowBase = parent.rowBase + parent.records`
   (`parent.records` for a base). Build the sketch over the new rows
   (base geometry by default), measure `recommendedRerank` against brute
   force over the layer's rows or set it to `records` when small, build
   the corpus (layout v2) and lexical segments with the base's tokenizer,
   build the vocabulary bloom with the base's geometry, and write the
   evaluation segment.
5. Compute the cumulative bitset: `zeroExtend(T_parent, rowBase)`, then
   set every `--remove` id (refuse an id `>= rowBase`; an id already set
   is a no-op) and the old id of every `--supersede`. The bits this step
   sets are the layer's `ΔT` (3.4). Record supersessions.
6. Write the manifest with the `layer` object; canonicalize; write the
   file; print the new identity, the resulting `calibrationDrift` and
   `calibrationStatus`, and, with `--lineage`, an updated shelf entry
   (section 7).

`append` MUST be byte-deterministic for identical inputs and parent, as
`compile` is. It MUST refuse to produce a layer whose drift would exceed
the chain's effective limit (5.5, computed with the producer's own
`readerLimit` policy) unless `--allow-drift` is given.

**A layer must do something.** `append` MUST refuse to emit a layer that
introduces neither a record nor a new tombstone — `records == 0` and
`ΔT = ∅`, which is what `--remove` of already-deleted ids alone
produces. This is the reason 4.3 requires a tombstone-only layer's
`tombstoneCount` to *exceed* its parent's: an identity that changes
nothing would still lengthen every chain that carried it, cost a mount
round trip, and count against the depth limits, for no corpus meaning.

### 6.2 `rebase`

Two producers appending to the same parent create two children — a fork.
Both are valid chains; the ref (section 7) names one as the head. The
other is re-based: `pikelet rebase --layer <file> --onto <newHead>
[--on-conflict refuse|skip|keep-both] [--on-foreign refuse|drop]`
recompiles the layer onto a new parent, copying its u8 rows, sketches,
records, postings and vocabulary bloom verbatim (nothing is re-embedded;
local ids do not change), recomputing `rowBase`, the bitset, the
supersessions, the evaluation segment's chain-level material and the
manifest. The result has a new identity. Ids the old layer had assigned
change; consumers holding `(oldIdentity, id)` citations still resolve
them against the old files, which remain immutable and hostable.

**What a layer *did*, not what it inherited.** A rebase replays the
operations the layer itself performed, which are exactly:

```
ΔT_L  = T_L ∖ zeroExtend(T_oldParent, L.rowBase)      the deletions L introduced (3.4)
S_L   = the supersession edges physically recorded in L's own segment (3.5)
R_L   = L's own records
```

and **never** `T_L` as a whole. `T_L` is cumulative: it contains every
deletion any ancestor on the *old* branch performed, and replaying it
would import those ancestors' operations into a branch that never
performed them — worse, it would import an ancestor's tombstone without
that ancestor's supersession edge, half of one operation. To compute
`ΔT_L` the rebase MUST open `L`'s original parent by
`L.layer.parent.identity`; if that artifact is unavailable, the delta is
unknowable and the rebase MUST refuse. Draft 2 replayed the cumulative
set; this is the correction.

**Two kinds of id.** `L` refers to ids of two kinds, and a rebase treats
them oppositely. Ids `L` **owns** — `R_L`, and the `y` side of every edge
in `S_L` — are offsets into its own range and MUST be translated:

```
newRowBase = H.rowBase + H.records          (H.records for a base)
newY       = newRowBase + (oldY − oldRowBase)
```

Every `newId` a rebased layer writes MUST lie in its new range, and a
rebase MUST refuse to write one that does not. Ids `L` **references** —
`ΔT_L` and the `x` side of `S_L` — are ancestor ids, and they are stable
only where `L`'s old history and `H`'s history agree.

**Ancestry and ingestion preconditions.** A rebase MUST refuse unless
`H.baseIdentity == L.baseIdentity`. It MUST also refuse unless the two
histories agree on the chain ingestion declaration (4.1): if both `H`'s
chain and `L`'s original chain already have one, they MUST be
canonical-JSON-identical; if `H` is the legacy depth-0 base itself and
therefore has no declaration yet, a rebased depth-1 layer MAY introduce
`L`'s existing asserted declaration, carrying `ingestAsserted: true`.
Same base is not enough on its own: two depth-1 forks of a legacy base
can each have asserted a *different* declaration — `B → A1 (assert X)`
and `B → B1 (assert Y)` — and a rebase from `A` onto `B1` would otherwise
proceed through every rule below only to emit a layer whose copied
`layer.ingest` a reader must reject. The check is a producer
precondition precisely so that rebase never constructs bytes the reader
refuses.

**The fork point.** Same base is necessary, not sufficient for id
stability either: let the **fork point** be the longest common prefix of `L`'s
original parent chain and `H`'s chain, compared layer by layer from the
base by identity, and `forkRowBase` the total row count of that prefix.
Every id below `forkRowBase` names the same record in both histories;
every id at or above it belongs to a layer the two histories do not share
(fork siblings assign overlapping id ranges to different records). An
`x ∈ ΔT_L` or `(x → y) ∈ S_L` with `x ≥ forkRowBase` is **foreign**:
`refuse` (the default) fails the rebase listing every foreign id; `drop`
discards that deletion or edge (the appended record `y` stays, as an
ordinary record). The foreign test applies to `ΔT_L`, not to `T_L`: bits
`L` inherited from its old branch are not `L`'s to carry anywhere. A
rebase onto a head that descends from `L`'s own original parent — the
common case, `H` merely having grown — has `forkRowBase = L.oldRowBase`
and no foreign ids.

**Deletion and supersession conflicts.** With foreign ids resolved and
`y` translated, let `H` have cumulative bitset `T_H` and union
supersession map `S_H`. Then, deterministically:

- `x ∈ ΔT_L` with `x ∈ T_H` (already tombstoned in `H`, with or without an
  edge): a no-op, not a conflict. Deletion is idempotent.
- `(x → newY)` with `x ∉ T_H`: recorded; the rebase sets `x`'s bit.
- `(x → newY)` with `x ∈ T_H` and no edge from `x` in `S_H`: recorded —
  `L` adds information `H` lacked.
- `(x → newY)` with an edge `(x → z) ∈ S_H`, `z ≠ newY`: a **conflict**.
  `refuse` (the default) fails the rebase listing every conflicting `x`;
  `skip` drops `L`'s edge for `x` and keeps `newY` as an ordinary appended
  record; `keep-both` records `L`'s edge as an additional edge, which is
  what makes 3.5's "all edges" view non-trivial and makes `H`'s edge the
  immediate successor and `L`'s the current one (3.5: greatest depth).
- An id `L` owned that a later layer of the *old* chain had tombstoned or
  superseded is not part of `L` and is unaffected.

The rebased layer's bitset is `zeroExtend(T_H, newRowBase)` with the
surviving `ΔT_L` bits set.

**Terminal rule: an empty rebase emits nothing.** After foreign-id
handling and conflict resolution, if `R_L = ∅` and applying the surviving
`ΔT_L` would set no bit not already set in `zeroExtend(T_H, newRowBase)`
— a tombstone-only `L` whose every deletion `H` has independently
performed — the rebase MUST NOT emit a layer, and MUST report that the
rebased operation is empty because the target history already represents
it. Emitting would manufacture exactly the artifact 6.1 forbids and 4.3
rejects (`records == 0` with no increase in `tombstoneCount`); this is
the rebase analogue of `append`'s no-op rule. A rebase with `R_L ≠ ∅`
always emits, whatever became of its deletions.

**Evaluation material.** The layer's local measurements (`exact`,
`recallVsC`) depend only on its own rows and are copied. Chain-level
golden queries (4.7) carry expected global ids that are wrong after a
rebase — `L`'s own ids moved and the surrounding chain changed — so a
rebase MUST either regenerate them by running the queries against the
rebased chain and recording the results, or drop them with
`goldenQueries: null, goldenQueriesDropped: "rebase"`; it MUST NOT copy
them.

A rebase never rewrites `H`. The rebased layer's `records`, `rowBase`,
bitset, supersessions and evaluation are recomputed from these rules
alone, so two producers rebasing the same layer onto the same head with
the same options produce identical bytes.

### 6.3 `compact`

```
pikelet compact --head <file|url>[#identity] [--lineage <shelf>] --out <file>
```

Compaction is **semantic maintenance**, not only physical consolidation.
It is the step that restores every exact property the layers
approximated: exact lexical statistics over the live set (5.4), a
vocabulary bloom over the live vocabulary only (4.5), a calibration fit
against the corpus actually served (5.5), no tombstones, no supersession
walks, depth zero, and reclaimed id space (3.3). The drift limit and the
depth limits exist to force it to happen. It produces a
`pikelet-complete-v2` base from a chain:

1. Mount the chain; enumerate live ids in ascending global order.
2. Copy live rows' quantized bytes and per-row `scale`/`offset` verbatim
   (same encoder, same quantizer — no re-embedding); recompute sketches by
   pooling and per-row digests; renumber survivors densely in the same
   order.
3. Rebuild the corpus segment over live records and the lexical index
   over live records with the base's tokenizer; statistics are now exact.
4. Refit calibration with the chain's encoder (the fit needs the encoder
   to embed its generated queries; a kind-3 base carries it), building
   the vocabulary bloom from live records only. If the producer cannot
   refit, it MUST ship `unscored`, never the inherited fit.
5. Record `compactedFrom: { identity: <old head>, depth, liveRecords,
   tombstones }` in the manifest and write the lineage segment (kind 7,
   below), which is REQUIRED for a base compacted from a chain.

Compaction MUST be byte-deterministic given the chain. The output has a
new identity and starts a new chain at depth 0. Compaction MUST NOT change
`encoder`, `dim`, `metric`, ingestion or tokenization; those changes are a
new compile, not a compaction. The compacted base carries `corpus.ingest`
copied from the chain (6.1) so it is appendable on its own terms.

**What compaction preserves — and does not.** Compaction preserves
**corpus meaning and provenance**: the compacted base's records are
exactly the old head's live records, in ascending old-id order, under the
translation the lineage segment prescribes, with every record's bytes
unchanged. It does **not** preserve rankings, and it is not meant to:
layered BM25 scored with statistics that included tombstoned documents,
compaction scores with exact live statistics; layered vector retrieval
generated candidates per layer and merged, compaction scans one sketch
over the unified corpus; the calibration is refit and the bloom is
live-only. If the layered approximations had any observable effect, a
compaction that reproduced the chain's rankings would have failed at its
job. Conformance therefore tests corpus preservation and retrieval
correctness separately (section 10, E) and records golden-query rankings
before and after as a behavioral comparison, never as an equality.

**Lineage segment (kind 7), layout `lineage-v1`:**

```
[0,4)    u32 version = 1
[4,8)    u32 flags (0; readers MUST reject non-zero)
[8,12)   u32 headCount            depth of the old head + 1: every artifact of the compacted chain
[12,16)  u32 supersessionCount
[16,24)  u64 oldRowTotal          rowTotal(old head)
[24,32)  u64 tombstoneCount       popcount of the bitset below
[32,64)  reserved, zero
[64, 64 + 48·headCount)           head table: headCount x (32-byte identity, u64 rowTotal, u32 depth, u32 zero),
                                  in ascending depth from the old base (depth 0) to the old head
[...]    bitset                   the old head's cumulative mask, zero-extended to oldRowTotal (3.4)
[...]    supersessions            every edge of the old chain: supersessionCount x (u32 oldId, u32 newId, u32 depth),
                                  sorted by (oldId, depth)
```

The **head table** is what lets a citation against any head of the
compacted chain — not only the last one — translate. Draft 2's segment
named only the old head, so once layers were retired a citation made
while `H3` or `H7` was the published head was unresolvable, and a chain
compacted twice could not recognize the intermediate base. Because depth
is at most 16, the table is at most 17 × 48 bytes; the previous compacted
base is its depth-0 entry, so retaining intermediate compacted bases
gives multi-hop translation with nothing further.

**Validation** (a digest-valid lineage segment is still untrusted
input). A reader MUST reject the segment unless all of the following
hold: the regions tile the segment exactly; `flags` is zero; `headCount`
is in `[1, 17]`; the head table's depths are exactly `0 … headCount − 1`
in order, its identities are pairwise distinct, its `rowTotal` values are
non-decreasing with the last equal to `oldRowTotal`, and the last
identity equals the manifest's `compactedFrom.identity`; the bitset's pad
bits are zero and its popcount equals `tombstoneCount`; every edge has
`oldId < newId < oldRowTotal`, `oldId`'s bit set, `depth` in
`[1, headCount − 1]`, `newId` within the range the layer at that depth
owned (`rowTotal(depth − 1) ≤ newId < rowTotal(depth)`), and `(oldId,
depth)` pairs sorted and unique; and the manifest's `compactedFrom`
counts agree with the segment's. Nothing in this list is inferred from
ordering that 3.5 says not to infer: each edge carries its depth, so the
immediate, all and current views are computed exactly as on a live chain.

**Translation** of `(identity, id)` by a reader of the compacted base,
`translate(identity, id)`:

1. `identity` not in the head table → "not on this history" (5.6 rule 3).
2. `id ≥ rowTotal(identity)` → invalid: the id did not exist when that
   artifact was the head.
3. Otherwise the id is translated according to the **final
   pre-compaction state** — the old head's mask and the full edge set —
   regardless of which head was cited. This is **forward translation**:
   it answers "what became of the record cited at `H3` by the time `H8`
   was compacted", which is the question a citation asks of a compacted
   base. It does not, and the head table does not make it able to,
   reconstruct **historical state** — "what was that record's tombstone
   or successor status *as of* `H3`". Historical state, like historical
   content, lives only in the old artifacts (see *Translation is not
   dereference*, below). Concretely: bit clear →
   `{ id: id − popcount(bitset[0, id)) }`; bit set with no outgoing edge →
   `{ deleted: true, supersededBy: null, currentSuccessor: null }`; bit
   set with edges → `supersededBy` is the immediate successor per 3.5 (an
   old-chain id, reported as such), and `currentSuccessor` is the
   current-successor walk of 3.5 over the segment's edges, then: if the
   final id is live in the old head, translated by the bit-clear rule; if
   it is itself tombstoned with no edge, `null`.

**What the lineage commits to.** The lineage segment is a segment of the
compacted base like any other: its digest is in the manifest, and the
manifest's digest is the new identity. So the new identity commits not
merely to *which* chain it was compacted from but to the exact
translation — the head table, the bitset and every edge with its depth —
and two readers of the same compacted base translate every old citation
identically, while a lineage segment altered after publication fails
verification like any other segment. This is what lets a consumer treat a
translated citation as evidence rather than as a hint.

**Translation is not dereference.** The lineage carries enough to say
what an old id *became*; it carries none of the old record's bytes. After
the old layers are retired (6.4), `(oldHead, id)` remains
**translatable** — a reader can say "this became new id 4127", "this was
deleted with no successor" or "this was superseded and its current
successor is new id 9910" — but is no longer **dereferenceable** in its
original corpus state: the text the agent actually read under `oldHead`
can only be produced by the old artifacts themselves. A reader answering
`record({ identity, id })` from a compacted base MUST report which of the
two it is doing (`historicalContent: 'unavailable'` when the old artifact
is not mounted) and MUST NOT present the successor's text as if it were
the cited text. This is a provenance boundary the operator chooses when
retiring layers, and section 6.4 states it in those terms.

### 6.4 Publishing order and retention

Objects before refs: a producer MUST make a layer's bytes fetchable at its
locator before publishing any ref or lineage listing that names it, and
MUST NOT overwrite an object at a locator that another artifact's
`locator` hint or a shelf entry names. Naming layers by depth and identity
prefix (`docs.0008.c41e7d09.pikelet`) makes overwriting structurally
unlikely. A reader that finds a listed layer missing MUST fail the mount
explicitly (3.2).

Retention follows from 3.2: every ancestor of a published head MUST stay
hosted for as long as that head is published or pinned by any consumer
the operator cares about. Layers become *eligible* for retirement only
when a compaction has superseded the chain and every ref has moved to the
compacted base. Retiring them is a provenance decision, not housekeeping:
the compacted base's lineage segment keeps every citation against any
head **of that chain** translatable (6.3) — citations against forks not
on the compacted ancestry were never on this history and are outside the
guarantee — and nothing keeps any citation **dereferenceable** except the
old artifacts. An operator who needs "what exactly did the agent read
under `oldHead`" to stay answerable keeps the old chain hosted, pinned by
identity, beside the compacted base; one who needs only "what is that
citation now" may retire it. Both are conforming; a host SHOULD document
which it does.

For the 648.5 MiB Wikipedia pack, publishing an update of 100 records is
one ~250 KB PUT and a shelf write, instead of a new 648.5 MiB object.

## 7. The ref: lineage in a shelf

A shelf is already "a registry that is also just a file" (`mcp.mjs`,
`loadShelf`). This profile adds an OPTIONAL `lineage` array to an entry:

```jsonc
{
  "packs": [
    {
      "name": "docs",
      "url": "docs.0008.c41e7d09.pikelet",       // the head (depth 8)
      "identity": "c41e…",                        // the head identity — pins the chain
      "lineage": [                                // base first; every layer below the head, in order
        { "identity": "8d73…", "url": "docs.pikelet" },
        { "identity": "41aa…", "url": "docs.0001.41aa0c77.pikelet" },
        // …
        { "identity": "9f3c…", "url": "docs.0007.9f3c1a2b.pikelet" }
      ]
    }
  ]
}
```

A reader given a shelf MUST verify that the chain reached by following
parent commitments from the head is exactly the listed lineage (same
identities, same order) and refuse otherwise; the listing accelerates
resolution, it does not define the chain. Entries without `lineage` are
resolved by locators under 5.1.1. Readers that predate this profile
ignore `lineage`, try to open `url`, and refuse the head's profile
explicitly.

The shelf is the one mutable object in the system. It has no identity and
never had one; updating it is the write. A host that wants "latest"
without a shelf can serve an HTTP redirect from a stable name to the
current head; a pinned reader ignores both. A host with neither — a
strictly immutable static store — has no in-band way to name the new
head, and the caller supplies it, exactly as they supply the URL of any
new artifact today.

## 8. Costs

Per layer with `n` records, `dim = 384`, 192-dim 4-bit sketches, format-2
rows (16-byte digests per 16-row block):

| region | bytes | resident at mount |
| --- | ---: | --- |
| sketch resident prefix | `256 + 8n + 96n` | yes |
| rows + digest blocks | `≈ 1.042 × 384n` | no (range-read per candidate) |
| corpus offsets + page table + digests | `8(n+1) + 32⌈n/256⌉ + 32n` | offsets and page table |
| corpus records | source text | no |
| lexical header + doclens | `64 + 4n` | yes |
| tombstone segment | `32 + ⌈rowBase/8⌉ + 8·supersessions` | yes |
| projected mask (reader state, not a segment) | `⌈n/8⌉` | yes |

A 100-record layer on the Wikipedia base is about 200 KB of content plus
a 57 KB bitset; its resident cost is under 70 KB. Mounting a depth-8 chain
on that base adds roughly 0.6 MB and, with a lineage listing, one parallel
wave of small reads to the base's 51.7 MiB mount; at the format ceiling of
16 the additions double. Query cost adds, per search tier beyond the
base, one masked sketch scan over `n` rows, one rerank fetch round issued
in the same parallel wave as the base's, and one term-table window read
per query term; the base's own scan is masked too, at no extra read. At one
hundred million base records the bitset alone is 12.5 MB per layer (3.4).
These are estimates from the layouts; the `range-proof` harness is where
they get measured before any default in 5.9 or 6.1 is frozen.

## 9. Integrity stance

Nothing in a chain is authenticated more weakly than in a single complete
artifact, and two things are authenticated additionally:

- Every layer is a format-2 complete container: manifest identity, header
  commitment for its sketch, per-row digests on rerank reads, per-record
  digests on hydration, whole-segment digests for eager segments. The
  tombstone segment is eager and digest-verified; so is a lineage segment.
- The head commits to its parent's identity and so, transitively, to every
  byte a query can touch in any layer. A host that rewrites an old layer
  changes bytes that a verified identity commits to, and the reader
  refuses at mount.

What identity does not provide is availability (3.2) or protection
against a manifest steering fetches (5.1.1); those are addressed as
mount rules, not commitments. Digest validity does not make a segment's
*contents* trustworthy either: the tombstone and lineage layouts carry
normative structural validation (4.4, 6.3) precisely because a publisher
can sign anything. The residual stances carry over unchanged: a lexical
segment above the lazy-open threshold is committed by segment digest, not
per read; a format-1 base's rerank rows are committed but not verified
per read until a full vectors pass. A chain's `info()` reports the
weakest stance among its layers for each field (`corpusIntegrity`,
`indexRowIntegrity`).

The shelf is untrusted input like everything else: a reader validates its
shape and identities and treats a listing that disagrees with the chain's
own commitments as an attack, not a hint.

## 10. Conformance

- **Fixtures:** `test/layered_profile.mjs` (run by `npm test`)
  builds in-process, deterministically, from the complete-profile suite's
  seeded kind-2 host-encoder corpus:
  (A) a base plus three layers — additions, deletions, a supersession, and
  one tombstone-only layer — asserting: no tombstoned id in any `query`
  result under every `retrieval` mode; **the masking case**: with the
  base's `C` nearest rows to a query all tombstoned and the nearest live
  row at position `C + 1`, that live row is returned, and the lexical
  analogue with the top `LEXICAL_CANDIDATES` BM25 hits all tombstoned;
  `record()` on a tombstoned id returns the record with `tombstoned`,
  `supersededBy`, `successors` and `currentSuccessor`, including a
  two-hop chain `A → B → C`; merged vector results at `rerank = records`
  per layer equal brute force over live rows; global ids are stable
  across heads (a query against head₂ and head₃ returns the same ids for
  records both contain); lexical scores equal a single-file build over
  the same records when no tombstones exist, and differ — with the
  difference recorded, not bounded — when they do; the union bloom equals
  a bloom built over **every record ever appended**, live and tombstoned,
  with the base's geometry — and is asserted *not* to equal a live-only
  bloom once a deletion has removed a word; the calibration status flips
  to `drift-exceeded` exactly at the reader's declared effective limit,
  with a supersession counted as two units, and `min(producerEnvelope,
  readerLimit)` is asserted in both directions (a looser manifest cannot
  raise a host's limit; a stricter one lowers it); appending
  byte-identical records yields distinct ids; a citation `(H1, id)`
  resolves under `H3`, `(H3, id)` with `id ≥ rowTotal(H1)` is invalid
  under `H1`'s identity, and `(fork, id)` is answered "not on this
  history".
  (B) hostile chains — wrong `parent.identity`, wrong `rowBase`, a bitset
  that clears a bit of `zeroExtend(parent)`, a bitset with pad bits set, a
  layer with a different `encoder` object, a kind-4 digest that does not
  match the base, a records-≥-1 layer missing a required segment, a
  tombstone-only layer carrying an index segment, a layer whose `ingest`
  differs from the chain's, depth 17, a shelf lineage that disagrees with
  the commitments, a listed layer missing, a locator with a scheme, an
  authority, a `..` segment, a percent-encoded `%2e%2e`, an encoded
  separator (`%2f`, `%5c`) or literal backslash, a doubled slash or (for
  file sources) a symlink escaping the directory, a parent fetch that
  redirects cross-origin, and one that redirects same-origin to a path
  outside the child's directory — every one refused at mount with a
  distinct error.
  (C) **amplification**, in two halves matching 5.9's two stages: a chain
  whose *declared* sizes exceed a budget — bitsets declared at the
  id-space maximum, segment counts at the structural ceiling,
  `recommendedRerank` at the per-layer cap on all 17 files — MUST be
  refused at preflight, before any read that would consume the declared
  bytes is issued, with the budget named; and a chain whose declarations
  are modest but whose source serves more bytes than declared, or a query
  whose unique terms times `2 × (depth + 1)` exceed the lexical ceiling,
  MUST fail at the read that crosses the budget, or explicitly before it
  for the lexical ceiling, with the budget named. In neither half may
  reads or allocations exceed the budgets before refusal.
  (D) `rebase`: **the delta case** — on branch A, `A1` tombstones base
  record 5 (once plainly, once with a supersession), then `A2` merely
  appends; rebasing `A2` onto branch B MUST NOT set bit 5 and MUST NOT
  record `A1`'s edge, and rebasing `A1` MUST carry both the bit and the
  edge; onto a head that descends from the layer's own original parent
  (no foreign ids) with each of 6.2's conflict cases, asserting the
  no-op, recorded, conflict-refused, conflict-skipped and conflict-kept
  outcomes, that every supersession target is translated by
  `newRowBase + (oldY − oldRowBase)` and lands in the rebased layer's own
  range, that local evaluation is copied and chain-level goldens are
  regenerated or dropped, never copied, and byte-identical output across
  two runs; onto a fork sibling, asserting the fork point is computed
  from the common prefix, that a deletion or edge in `ΔT_L`/`S_L`
  referencing an id at or above it is refused by default and dropped
  under `--on-foreign drop`, and that an inherited bit above the fork
  point is neither foreign nor replayed; a tombstone-only layer whose only
  deletion the target head has already performed, asserting no layer is
  emitted and the empty result is reported; two depth-1 forks of a legacy
  base with different asserted declarations, asserting the rebase between
  them is refused before any bytes are built, and that a depth-1 layer
  rebased onto the bare legacy base carries its assertion across; onto a
  chain with a different `baseIdentity`, refused; with the original parent
  unavailable, refused.
  (E) `compact` over (A)'s chain, asserting three separate things.
  **Corpus preservation:** the compacted base's records are exactly the
  old head's live records, in ascending old-id order, byte-identical,
  and `translate` maps every live old id to the position holding the
  same bytes. **Retrieval correctness, each checked on its own terms:**
  exact vector retrieval over all live rows of the compacted base
  (`rerank = records`) equals brute force over the old chain's live rows
  after id translation; lexical retrieval equals a from-scratch
  `bm25-v1` build over exactly the live records (this is the assertion
  that the chain's tombstone-inclusive statistics are *gone*); and the
  compacted sketch meets the complete profile's ordinary evaluation
  requirements at its `recommendedRerank`. **Behavioral comparison, not
  equality:** the golden queries' rankings on the chain and on the
  compacted base are both recorded in the fixture output for inspection,
  and a difference between them is expected wherever the layered
  approximations had an effect — the suite MUST NOT assert they are
  equal. Then, on the same compacted base: the lineage head table lists
  every head with the right `rowTotal`; `translate` resolves a live id, a
  deleted id without successor, a superseded id and a two-hop
  supersession **cited against an intermediate head** as well as the last
  one, and answers forward translation (the state as of the old head, not
  as of the cited head — a fixture where a record was live at `H3` and
  deleted by `H8` MUST report it deleted when cited against `H3`);
  rejects an id at or above the cited head's `rowTotal` and an identity
  not in the table; `record({identity, id})` reports `historicalContent:
  'unavailable'` when the old chain is not mounted and never returns a
  successor's text as the cited text; lineage validation rejects each of
  6.3's malformed-but-digest-valid cases (an edge without its bit,
  `newId` outside its depth's range, a duplicate `(oldId, depth)`, a head
  table out of depth order or with a repeated identity, a last identity
  not equal to `compactedFrom.identity`, a non-zero flag); and the bytes
  are identical across two runs. Then a **second compaction** of a chain
  built on that base, asserting a citation against the first compacted
  base's identity translates through the second's head table.
  (F) a reader built from the current `SUPPORTED_PROFILES` refuses a layer
  file with the unsupported-profile error.
  (G) **consecutive tombstone-only layers**: a base, two tombstone-only
  layers in a row, then a layer with records, asserting that the last
  layer's `rowBase` equals the base's record count, that `owner(id)`
  resolves every id of the base and of the last layer to the right file
  and local id, that ids in no interval are rejected, that
  `record(id)` and `query` behave identically to the same chain built
  with one tombstone-only layer carrying both deletions, and — the reason
  the fixture exists — that a reader keyed on "largest `rowBase ≤ id`"
  over all layers would have resolved wrongly, so the suite MUST include
  the id at exactly `rowBase` of the shared value.
  (H) **appendability and inheritance**: `append` refuses a kind-1 base;
  refuses a released-style base lacking `corpus.ingest` without
  `--assert-ingest` and records `ingestAsserted` with it; a base compiled
  with `corpus.ingest` appends without assertion; on an asserted chain,
  depth-2 and deeper layers inherit the depth-1 declaration, and a layer
  that changes, drops or re-asserts it is refused at mount; and `append`
  refuses to emit a layer with no records whose `--remove` ids are all
  already tombstoned (`ΔT = ∅`).
- **Producer:** emits layers that verify, whose row arithmetic holds
  against their parents, whose bitsets satisfy the `zeroExtend` superset
  rule, whose segment sets match 4.3, whose `ingest` matches the chain's,
  whose evaluation bounds hold, and that refuses depth, drift, id-space
  and appendability violations.
- **Reader:** executes section 5 with masks supplied to candidate
  generation, enforces 5.1.1 and 5.9, passes (A)–(H), and reports the
  fields of 5.1 step 5 and 5.4–5.6.

## 11. Relationship to existing documents and code

This section was written as the plan for implementing the profile. It now
records what shipped, and where the implementation differs from the plan.

- **Contract §8 (non-goals).** Still true: no mutable indexes, no online
  updates. A chain is a sequence of immutable artifacts; the mutable thing
  is a ref, which the contract never covered. Contract §4.2 notes that
  supersession may be incremental (a layer) as well as whole (a new base).
- **`COMPLETE_PROFILE.md`.** Segment kinds 6 (`tombstones`) and 7
  (`lineage`) are registered in §3.3 and query-interpretation kind 4
  (`inherited-v1`) in §3.6. The compiler writes `corpus.ingest` (§3.2), so
  bases compiled since are appendable without `--assert-ingest`. No layout
  in that document changed.
- **`SKETCH_PROFILE.md`.** Unchanged on disk. The sketch reader's search
  takes an `exclude` option (an LSB-first bitset over local ids), and
  masked rows never enter the candidate heap, so `C` counts live rows.
  *Differs from the plan:* the engine's `_pikelet_sketch_scan` kernel takes
  no mask. When a mask is in play the reader asks the kernel for `C` plus
  the masked count and drops masked ids, which yields the exact top-`C`
  live rows; if that over-fetch exceeds the kernel's buffers it falls back
  to the masked JS scan.
- **Lexical reader (`complete/lexical.mjs`).** Its scoring takes an
  `exclude(id)` predicate applied while postings are scored, before the
  top-`n` cut, and `stats` supplying chain-global `docCount`/`avgdl`.
- **Reader.** The chain reader is `openPikeletChain`
  (`complete/layer-reader.mjs`): it owns the interval and ancestry tables,
  projected masks, the union supersession map, merged lexical statistics,
  merged search, the resolution rules of 5.1.1 and the budgets of 5.9.
  `fuseCandidates` and `createAbstentionScorer` are unchanged. *Differs:*
  `SUPPORTED_PROFILES` in `complete/index.mjs` still lists only the
  complete profiles, because `openPikeletFile` opens single files; the
  layer profile is recognized by the chain reader. A chain's `query()`
  takes a query vector; text queries go through `pikelet mcp`.
- **CLI (`packages/pikelet`).** `append`, `rebase` and `compact` ship;
  `compile` records `corpus.ingest`; `append` takes
  `--reader-drift-limit`; `mcp` mounts `lineage` entries from a shelf.
  Parent-locator resolution and the reader drift limit are reader options
  (`openPikeletChain`'s `resolveParents` and `readerDriftLimit`), not `mcp`
  flags. *Not implemented:* `doctor` reports nothing chain-specific (depth,
  drift, limits, `ingestAsserted`, budgets).
- **MCP.** `search` results carry `layer`, and each section carries the
  head identity as `packIdentity`. `get_record` adds `tombstoned`,
  `supersededBy`, `successors` and `currentSuccessor`. `list_packs` adds
  `layers`, `tombstones`, `calibrationDrift` and `calibrationStatus`.
  *Differs:* `get_record` takes no `identity` argument (citation
  translation is the library's `citation(identity, id)`), and there is no
  `historicalContent` on a compacted base. `verify_pack` on a chain runs
  the base's tests against the whole chain, skipping goldens whose
  expected record a layer removed, and reports them as base-scoped
  (`testsFrom`), rather than reporting per layer. A compacted base's
  lineage segment is hash-checked and parsed when a citation first needs
  it, not by `verify_pack`.

## 12. Non-goals

- Writes at query time, by readers, from any host. Readers never write.
- In-place modification of any artifact, including appends to an existing
  file: it breaks header-first addressing, cached headers, PUT-semantics
  object stores, and the identity of bytes already served.
- Un-deletion; transactions; multi-writer coordination beyond the fork
  rule of 6.2.
- Content deduplication (decision 9).
- Resolving a bare id across histories, or translating a citation against
  a fork not on the mounted ancestry (5.6).
- Multi-hop citation translation across several compactions as a format
  guarantee when intermediate compacted bases are not retained (6.3).
- An error bound, on ranking or on calibration, derived from the drift
  gate (4.5, 5.4).
- Appending to a kind-1 base (6.1).
- Changing the encoder, metric, ingestion or tokenization within a chain.
- Chains at the depth ceiling as a steady state. Compaction is part of the
  workflow, not an emergency procedure.

## 13. Open questions — for measurement and implementation, not another reading

**Measured, 2026-09-26 — questions 1, 2 and 3 below.** All three were run on
BEIR NFCorpus (1,817-record base, half the corpus withheld for appends,
SciFact as the off-domain bank) and, for question 3, on SciFact as well.
Harnesses: `benchmarks/beir/measure-inherited-fit.mjs`,
`measure-bloom-pollution.mjs`, `measure-stale-idf.mjs`. Each drives the
shipped code — `calibrateRetrievalAbstention`, `createAbstentionScorer`,
`buildLexicalSegment`, the inline encoder — not a reimplementation. Every
result below is one dataset pair at one seed on a ~1,800-record base; the
wiki-pack measurement question 1 originally asked for has NOT been run, and
a corpus that size has a much heavier-tailed `df` distribution. Treat these
as "the feared effect did not appear at this scale", not as a bound.

1. **How the inherited fit actually degrades.** **Measured: it does not, up
   to 40% drift.** A fit made on the base was applied unchanged to corpora
   drifted by appends plus tombstones at 5/10/20/40%, under random and
   topically concentrated policies, and scored on three probe families
   (the fit's own verified positives; queries about appended records the
   fit never saw; off-domain queries). Separation between appended-record
   queries and off-domain ones spans **0.9833–0.9947 AUC across all eight
   configurations — a 0.0114 spread with no trend in either policy** —
   with appended-record queries answered at p ≈ 0.95 and off-domain ones
   held at p ≈ 0.10 throughout. So `driftLimit` as a fraction is not the
   liability this question assumed, and the provisional 0.20 `readerLimit`
   of 5.5 is **conservative rather than risky**; nothing here argues for a
   shift-aware measure or for a per-layer refit. Two caveats. Separation
   between the fit's *own positives* and off-domain queries does fall
   (0.962 → 0.917 random), but that is an artifact of the probe design, not
   the fit: those positives are derived from base records, and drift
   deletes some of them, so the fall conflates "the fit degraded" with "the
   answer was removed". And because the two policies append different
   records, each policy is comparable against its own drift trend, not
   head-to-head at equal volume — an earlier run that compared them
   directly produced a spurious 0.22 AUC gap that was entirely probe/corpus
   overlap.
2. **Bloom pollution in practice.** **Measured: real, but it never caused
   the harm 4.5 describes — and the direction 4.5 states is wrong.** Two
   blooms were built at the shipped geometry over the same drifted corpus:
   the union a chain serves (every record ever present) and the live-only
   bloom compaction restores. Only the bloom differed between two scorings
   — same fit, same asset, same retrieval, same probe texts. Pollution is
   measurable: on queries about tombstoned content, `known_frac` rises
   **+0.045 to +0.057** (0.926→0.983 at 5% drift, 0.941→0.986 at 40%), and
   the union carries ~5% more set bits. Verdict flips in the harmful
   direction — the live-only bloom would have abstained, the union answers
   — were **0 out of 225 probes across all four drift levels**. Controls
   behaved: the live-record family's `known_frac` delta was exactly 0.000
   in every row, and off-domain probes stayed at 0.92–0.95 abstention.
   **The correction 4.5 needs** is that pollution pushes *toward*
   abstention, not away from it: `known_frac`'s fitted weight in the
   shipped stage-2 model is **negative** (measured −0.722, against d0
   −0.914, margin +1.514, mean10 +0.385), which is deliberate — see
   `calibrate.mjs`'s note that a held-out question scoring `known_frac`
   high "is the profile of a real in-domain unanswerable query". A bloom
   can only gain bits, so pollution can only raise `known_frac`, which
   lowers the probability and makes abstention *more* likely. The observed
   drift was in that fail-safe direction (2 extra abstentions at 5% drift,
   1 at 10%, 0 above). This conclusion depends on a weight *sign* that a
   future refit could flip, so 4.5 should state the dependency rather than
   the outcome alone.
3. **`idf` over tombstoned records** (5.4). **Measured: negligible; the
   proposed correction table is not needed.** Two lexical segments were
   built with the shipped builder over the same live document set, one
   scoring with `N`/`avgdl`/`df` from the full pre-deletion corpus (what a
   chain serves) and one from a scratch build over the live set (what
   compaction restores), then scored against graded qrels. Across 24
   configurations — SciFact and NFCorpus, at 2/5/10/25% drift, under
   uniform-random, rare-term-concentrated and single-topic deletion
   policies — **the aggregate nDCG@10 difference never exceeded 0.005**,
   and it was positive (stale statistics slightly favouring the retained
   documents) in 15 of 18 sweep rows. The rare-term policy this question
   names as the worst case was **not** the worst case: targeting documents
   holding terms with df ≤ 3 produced smaller deltas than random deletion,
   because destroying a rare term's `df` only matters when a query uses
   that term, and rare terms are rare in queries too. Individual queries do
   move (largest single-query nDCG@10 drop 0.29–0.37 on SciFact under
   topical deletion); it averages out over 300 graded queries. So the
   per-layer `df` correction table should be **dropped from the design**:
   it is new segment machinery correcting an effect below noise. What is
   not covered: this is lexical-only ranking, not hybrid fusion, and the
   guard of the shipped fusion rule interacts with lexical rank.
4. **Sketch geometry per layer.** Allowed as drafted; forbidding it would
   simplify a future resident index over the union of sketch tiers.
5. **The producer default of 8 and the budgets of 5.9.** Chosen for
   round-trip count and plausibility, not measured. The range-proof
   harness over R2 decides both.
6. **A shared query-term limit.** Draft 3 keeps the single-file and chain
   paths identical and fails a chain query only at the lexical ceiling.
   If very long queries on deep chains turn out to matter, the
   alternative is one term limit for *both* paths, which is an amendment
   to the complete profile's reader behavior rather than a rule here.
7. **`tombstones-v2`.** A compressed bitset encoding for the 100M-record
   regime (3.4); decide when a corpus that size exists.
8. **Signatures over the head identity** (contract §10, question 7).
   A mutable ref makes this more urgent than it was for a single file:
   the question "who moved the head?" has no answer inside the format.
9. **Merged ranking across unrelated packs on a shelf.** This profile
   merges layers of one chain. The MCP server's per-pack sections for
   unrelated packs are a separate question this draft does not touch.
