# Search Artifact Contract

Copyright 2026 Matthew Noonan. This specification is part of the Pikelet
project and is licensed under the Apache License 2.0 (see `LICENSE`).

**Status:** Draft 1  
**Scope:** Behavioral contract for Pikelet Search Artifacts. This document is
not a byte-layout specification.

Draft 1 resolves three previously open decisions — the identity scheme, the
tolerance regime, and graph traversal semantics — adds result hydration to the
execution layer, and grounds conformance in the fixture assets that exist in
this repository today.

## 1. Thesis

Search is something you compile.

A Search Artifact is an immutable, content-addressed package that contains
enough information to execute and evaluate search for a specific corpus without
requiring the original indexing pipeline at query time.

The artifact contract elevates the search index from an implementation detail
to a portable deployment unit. The byte format is one expression of the
contract; the contract itself is the observable behavior a reader, producer, and
host can rely on.

## 2. Mechanism

A complete Search Artifact carries, references, or commits to these components,
in this order:

1. **Corpus**: the records, chunks, identifiers, and metadata search results
   refer to.
2. **Index**: the vector/search structure used to retrieve candidates for that
   corpus.
3. **Encoder**: the mechanism that interprets queries for this corpus.
4. **Evaluation**: evidence that the encoder and index behave as declared.
5. **Calibration**: thresholds and policies for confidence, abstention, and
   known limits.

Only after these mechanisms are present is it fair to say that the artifact
carries its own comprehension. The artifact does not contain an LLM by default;
it contains the compiled representation needed to interpret queries in the
context of one corpus.

## 3. Definitions

**Artifact**  
An immutable package of bytes and metadata that declares a search contract.
Current Pikelet artifacts are the complete `.pikelet` profile and the
`.pikelet-sketch` index profile it embeds. `.pnck` snapshots (engine
serialization) and the deprecated `.pikelet-range` profile predate the
contract and cover fewer layers; see section 9.

**Reader**  
Software that opens an artifact and executes the behavior declared by the
artifact.

**Producer**  
Software that emits an artifact. A producer does not need to use Pikelet's
index builder or algorithms.

**Host**  
The runtime environment that supplies storage, range reads, memory, execution,
and optionally query embedding services.

**Corpus record**  
A stable unit of returned content, usually a document chunk with an id, text,
title, source path or URL, and optional anchor.

**Query interpretation**  
The full process that turns a user query into the vector or representation used
by the index. This includes model identity, preprocessing, prefix policy,
normalization, dimensionality, quantization compatibility, and abstention
policy.

## 4. Contract Layers

### 4.1 Identity

An artifact has a stable identity derived from content.

The identity of an artifact is the cryptographic hash of a canonical manifest.
The manifest commits to the identity of every segment the artifact carries or
references. A flat hash over the full artifact bytes MAY be recorded as an
additional convenience digest, but it is not the artifact identity, because a
flat hash cannot verify a partial read.

Each segment commitment MUST be range-verifiable: the manifest commits to a
segment either as a list of fixed-size chunk hashes or as a Merkle root with a
declared chunk size. A reader that fetches a byte range MUST be able to verify
the fetched bytes against the manifest without fetching the rest of the
segment. Chunk sizes are declared per segment by the byte-layout specification
for each profile.

The identity MUST be independent of transport location. Moving an artifact
from a local file to object storage, a CDN, or another mirror must not change
what the artifact is.

If an artifact references external segments, the artifact identity MUST commit
to those segment identities, not only their URLs.

### 4.2 Immutability

An artifact is immutable after publication.

Readers MUST NOT mutate artifact bytes as part of query execution. Producers
MAY create a new artifact that supersedes an old one, but that new artifact has
a new identity.

Mutable services may cache, prefetch, or materialize artifact ranges, but those
runtime caches are not part of the artifact.

### 4.3 Corpus Binding

Search results are bound to a fixed corpus.

A conforming artifact MUST define stable corpus record ids. A result id returned
by a reader MUST refer to the same corpus record for every conforming reader of
that artifact.

The artifact MUST either carry the corpus records or commit to content-addressed
external corpus segments. A location URL alone is not enough to define corpus
identity.

If corpus records include source URLs, those URLs are metadata. They do not
define artifact identity unless explicitly included in the canonical corpus
segment.

A corpus-bearing artifact redistributes content. Corpus segments SHOULD carry
provenance and licensing metadata for that content, and the manifest reserves
a corpus provenance field so this metadata has a stable place even when it is
empty.

### 4.4 Query Interpretation

An artifact MUST declare how queries are interpreted.

The declaration MUST include one of these modes:

1. **Inline encoder**: the artifact carries an encoder, preprocessing rules, and
   calibration data.
2. **Pinned external encoder**: the artifact names an external encoder by
   content hash or equivalent immutable identity and carries verification test
   vectors.
3. **Host-provided compatible encoder**: the host supplies query vectors that
   match the artifact's declared dimensionality, metric, preprocessing, and
   normalization requirements.

An artifact that carries or references an encoder MUST declare:

- encoder identity
- input normalization rules
- query prefix or instruction policy
- output dimensionality
- vector normalization policy
- numeric representation and quantization compatibility
- known limits and abstention policy, if any

Encoder/index skew is a contract violation. A reader MUST NOT silently use an
encoder that does not match the artifact's declaration.

Every mode carries a verification obligation. Modes 1 and 2 carry encoder test
vectors. Mode 3 MUST carry golden end-to-end fixtures: query texts with
expected top-k corpus record ids under the profile's declared tolerances, so a
host can verify its encoder against the artifact without access to the
producer's encoder. A mode 3 artifact without golden fixtures is incomplete:
its query interpretation cannot be checked, so a reader MUST surface it as
unverified rather than treating compatibility as established.

### 4.5 Index Semantics

An artifact MUST declare the semantics required to interpret index distances
and results.

At minimum, the declaration MUST include:

- vector dimensionality
- distance metric
- numeric representation
- quantization scheme, if any
- record count
- id mapping between index nodes and corpus records
- format version and compatibility profile

For graph indexes, the artifact SHOULD declare graph parameters such as entry
point, max level, degree limits, and construction metadata when those values are
needed for validation or compatibility.

For a graph profile, the declared search behavior includes the traversal
procedure itself: entry selection, upper-layer descent, beam semantics and
their parameters, termination, and tie behavior. The byte-layout specification
for each graph profile MUST define that traversal. Construction remains out of
contract: a conforming reader does not need to reconstruct the producer's
build algorithm, and a producer does not need to use Pikelet's builder — but a
reader cannot claim conformance for a graph profile without implementing that
profile's traversal semantics.

### 4.6 Evaluation

An artifact SHOULD carry or reference an evaluation segment.

An artifact that carries or references an encoder MUST carry or reference an
evaluation segment. Encoder-bearing artifacts without evaluation data are
incomplete because their query interpretation cannot be judged independently.

Evaluation data MAY include:

- golden queries
- expected nearest records
- teacher model identity
- teacher/student fidelity metrics
- recall measurements
- abstention thresholds
- calibration curves
- dataset and sampling notes

Evaluation is part of the artifact contract, not marketing metadata. It tells a
reader, producer, host, and user what behavior the artifact claims and what
limits it admits.

### 4.7 Calibration and Abstention

If an artifact declares confidence, no-result, or abstention behavior, it MUST
declare the thresholds and data used to calibrate that behavior.

A reader implementing abstention MUST follow the artifact's declared policy. A
reader that does not implement abstention MUST report that limitation instead of
returning confident-looking uncalibrated results.

### 4.8 Execution

A conforming reader MUST be able to execute top-k search according to the
artifact's declared profile.

For a range-readable artifact, a reader SHOULD be able to execute search without
loading the entire index into memory. The artifact MAY declare resident segments
and lazy segments. The reader MAY cache lazy segments, but cache state must not
change result semantics.

Execution includes result hydration. Returning corpus record ids alone does
not complete a search; a reader MUST be able to resolve result ids to the
corpus records they are bound to. For a range-readable artifact, the corpus
segment MUST therefore be addressable with the same resident/lazy/range
mechanics and the same integrity rules as the index segment, and the profile
MUST declare how record ids map to corpus byte ranges.

Execution statistics such as range requests, bytes read, resident bytes, and
cache hit behavior are not required for semantic conformance, but they SHOULD be
reported by diagnostic readers because they determine deployability.

## 5. Conformance

The contract defines observable behavior, not construction algorithms.

### 5.1 Reader Conformance

A reader is conforming for a profile if it can:

- parse and validate artifact identity and version metadata
- resolve all required segments by content identity
- reject unsupported or incompatible query interpretation declarations
- execute search for the declared metric and numeric representation
- return corpus-bound ids and distances in the declared format
- pass the conformance suite for that profile

Reader conformance may be profile-specific. For example, a reader may support
snapshot artifacts but not range-readable artifacts, or host-provided vectors
but not inline encoders. Such a reader is conforming only for the profiles it
declares.

### 5.2 Producer Conformance

A producer is conforming for a profile if it emits artifacts that:

- validate structurally
- declare all required contract layers
- pass the conformance suite
- meet their own declared evaluation bounds

A producer does not need to replicate Pikelet's HNSW construction, quantizer, or
layout algorithm. It must emit a valid artifact whose observable behavior
matches the contract it declares.

This distinction is essential: conformance is about outcomes and invariants, not
about cloning one implementation.

### 5.3 Behavioral Tolerances

Exact byte-for-byte equality is not required between producers.

Rank-identical output is a property of one reference reader on one build
target, not of the contract. Pikelet's own engine produces different distance
reduction orders across WASM SIMD, scalar WASM, and native SIMD builds, so a
tolerance regime that demands rank-identical results from independent readers
cannot be satisfied honestly.

The tolerance regime is therefore:

- **Structural conformance**: required fields, segment hashes, bounds, and
  version compatibility are exact.
- **Reference-reader conformance**: each profile designates one reference
  reader and build target. That reader, on a fixed artifact and fixed queries,
  produces rank-identical results with declared tie behavior. Its results are
  recorded as golden fixtures.
- **Independent-reader conformance**: any other reader meets declared recall
  and distance-error bounds against the golden fixtures. It does not need to
  be rank-identical to the reference reader.
- **Producer conformance**: a producer's artifact meets its own declared
  recall, distance, or fidelity bounds on golden queries under the reference
  reader.

Every bound is explicit. Silent tolerance is not allowed.

### 5.4 Conformance Suite Assets

The conformance suite is assembled from fixture assets that already exist in
this repository, versioned alongside the profiles they exercise. Each asset
lists the gate that actually executes it:

- golden snapshot fixtures and fixed-query golden results
  (`test/fixtures/golden_snapshots.js`; run by `npm test`)
- recorded brute-force baselines locking recall and search output
  (`test/fixtures/search_oracles.js`; the brute-force computation itself
  lives in `run_tests.js` — the fixture freezes its outputs; run by
  `npm test`)
- sketch profile golden fixtures: committed artifact bytes with exact
  reference-reader results (`test/fixtures/sketch_golden.js`, checked by
  `test/sketch_profile.js`; run by `npm test`)
- abstention golden fixtures with exact expected labels
  (`examples/legacy/03-edge-docs-search/fixtures/abstention-golden.json`, driven
  under Miniflare by `npm run test:worker-example`; run in CI)
- encoder parity verification between independent implementations
  (`examples/legacy/03-edge-docs-search/verify_student.mjs`). This is a
  producer-side gate: it requires the trained student artifacts, which are
  not committed, so it runs when a student encoder is (re)trained — not in
  the automated suite. An artifact shipping a distilled encoder without this
  check passing is a non-conforming producer, not a missing fixture.

Until these are packaged as a standalone kit, "pass the conformance suite"
means passing the checks these assets drive. A contract requirement with no
fixture behind it is a defect in this specification, not a lesser kind of
requirement.

## 6. Versioning and Compatibility

Artifact formats MUST be versioned.

A reader MUST reject unsupported major versions. A reader MAY accept older minor
versions if the compatibility rules for that format permit it.

New fields MAY be added only when old readers can safely ignore them or when the
format version changes to make rejection explicit.

Compatibility rules belong in the byte-layout specification for each artifact
format, but those rules MUST preserve the contract layers defined here.

## 7. Security and Integrity

A reader MUST treat artifact bytes as untrusted input.

At minimum, a reader MUST:

- validate all sizes, offsets, counts, and arithmetic before allocation
- reject out-of-bounds segment references
- reject malformed strings and metadata that cannot be safely represented
- avoid executing code from artifact metadata
- verify referenced segment hashes before use
- verify lazily fetched byte ranges against the manifest's chunk commitments,
  not only whole segments

URLs inside corpus metadata are not executable authority. User interfaces that
render artifact metadata MUST sanitize URLs and text before display.

## 8. Non-Goals

The Search Artifact contract does not require:

- mutable indexes
- online updates
- a database server
- a specific HNSW implementation
- a specific embedding model
- a specific host such as Cloudflare Workers
- universal best performance across all ANN workloads
- filtered or metadata-constrained search; filtering is host and application
  behavior in the current profiles and stays out of contract until a profile
  declares it

Immutability is a feature of the contract. Mutable search systems can exist
beside it, but they are not the artifact model.

## 9. Current Pikelet Profiles

Current Pikelet implementations are moving toward this contract through these
profiles:

### 9.1 Snapshot Profile (`.pnck`)

The snapshot profile carries a Pikelet index snapshot suitable for full restore
into memory. It is useful for Node, browser, and Worker deployments where the
entire index fits within memory and bundle limits.

Current snapshots primarily cover the index layer. Corpus and query
interpretation are supplied by surrounding application assets.

Snapshots carry no integrity digests: readers validate them structurally
(sizes, offsets, counts, version bounds) but cannot verify bytes against a
committed hash. Identity as defined in 4.1 is not yet implemented for this
profile; hosts that need integrity must hash the snapshot externally.

### 9.2 Range Artifact Profile (`.pikelet-range`) — deprecated

**Status: deprecated (2026-08-28).** The sketch profile (9.3) supersedes this
profile for every measured regime: depth-1 execution beat graph traversal
~5x end-to-end over real networks at roughly a third of the artifact size
(ROADMAP Track A, SIFT1M). Readers remain supported so existing
`.pikelet-range` artifacts stay openable; producers should not emit new
ones, and no further format revisions are planned.

The range artifact profile carries a range-readable Pikelet index that separates
resident routing data from lazily materialized node records.

It is intended for bounded-memory execution over immutable storage. Current
range artifacts primarily cover the index and execution layers. Corpus, encoder,
evaluation, and calibration are currently adjacent assets or future segments.

Integrity stance: format v3 stamps whole-segment SHA-256 digests for the id
map, router segment, and base segment into the header. The reference reader
verifies the id map and resident router at open (default on, fail-closed
when verification is requested but no crypto backend exists) and exposes
`verifyBaseSegment()` to check the lazy base segment on demand — streamed
in bounded chunks where the runtime provides a streaming hash, one bounded
read otherwise. Individual lazy range reads remain unverifiable until
per-chunk commitments arrive with the complete profile's manifest — the same
transitional stance as the sketch profile. Pre-v3 range artifacts carry no
digests at all and open structurally validated but unverified.

The current range profile is an HNSW split layout: a resident router segment
over lazily materialized base records. Base-layer geometry changes — for
example inline-neighbor records or cluster-page layouts that reduce sequential
round depth — are new profiles under this same contract, not amendments to
this one. The contract layers are what carry over; the traversal and layout
semantics are what each profile defines.

### 9.3 Sketch Artifact Profile (`.pikelet-sketch`)

The sketch artifact profile carries a two-tier index: a resident compressed
sketch of every vector and a lazily range-read tier of full quantized rows.
Its execution model is a resident scan followed by a single parallel fetch
round for exact rerank — sequential fetch depth 1 by construction.

Byte layout and execution semantics are specified in `SKETCH_PROFILE.md`.
Like the other current profiles it is transitional on identity (whole-segment
hashes, no per-chunk commitments) and carries corpus, encoder, evaluation,
and calibration as adjacent assets.

### 9.4 Complete Search Artifact Profile

The complete profile is the target contract: corpus, index, encoder,
evaluation, and calibration are all carried by or content-addressed from one
artifact identity.

This profile is implemented and shipping: byte layout and reader obligations
are specified in `COMPLETE_PROFILE.md` (Draft 2), the reader and builder
ship in `pikelet-wasm/complete`, and `pikelet compile` emits
it. It is the profile that turns Pikelet from a vector-search library into
an artifact compiler and reader.

## 10. Open Decisions

These decisions must be resolved before freezing a byte-layout specification:

1. Which corpus fields are mandatory, and which are host/application metadata?
2. Are inline encoders mandatory for the complete profile, or can a pinned
   external teacher be complete?
3. What is the minimum evaluation segment for small artifacts?
4. How should abstention be represented when a host does not implement it?
5. What compatibility promise is attached to each major format version?
6. Which range-read diagnostics are required versus optional?
7. How are signatures and provenance layered on top of content identity?
8. What chunk sizes balance range-verification overhead against the real
   range-read shapes of each segment type?
9. What numeric distance-error bounds apply per metric and quantization scheme
   for independent-reader conformance?

Resolved in Draft 1: artifact identity is the hash of a canonical manifest
with range-verifiable segment commitments (4.1); rank-exactness is scoped to a
designated reference reader, with recall and distance bounds for all other
readers (5.3); graph traversal semantics are part of each graph profile's
declaration (4.5).

## 11. One-Sentence Contract

A Search Artifact is a content-addressed, immutable search package whose corpus,
index, query interpretation, evaluation, and calibration define a portable
behavioral contract that any conforming reader or producer can implement from
the profile specifications alone, without reproducing Pikelet's construction
algorithms.
