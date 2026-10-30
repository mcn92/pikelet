# Changelog

Release notes for the two published packages in this repository,
`pikelet-wasm` (engine + artifact readers/builders) and `pikelet` (the
CLI: compile, mcp, doctor, scaffold; plus the Docusaurus plugin).
Through 0.6.0 these were published as `pancake-wasm` and
`create-pancake-search`; both old names are deprecated pointers. Both
packages are pre-1.0: a minor bump may carry breaking changes, and each entry lists them
first.

## pikelet-wasm 0.12.0 / pikelet 0.12.0 — 2026-10-30

Query embedding is about 1.65x faster, because the inline encoder now
computes in integers, and `pikelet` no longer installs
`@xenova/transformers`, which took a plain install from 82 packages and a
critical audit advisory down to two packages. Upgrade if you embed queries
in the browser or Node, or build `workers-ai` projects with `create`.
Embeddings shift slightly (see Compatibility); packs compiled before this
release still verify and serve. Both packages move to 0.12.0 together, and
`pikelet` requires `pikelet-wasm` ^0.12.0.

### Compatibility

- **The inline encoder computes in integers, so its embeddings shift
  slightly.** Every projection is now an integer GEMM: activations are
  quantized to int16 per token and multiplied against the u8/u4 weights
  with `i32x4.dot_i16x8_s`, and GELU is vectorized. Embeddings move by at
  most about 3e-4 per component (MiniLM; 9e-5 for BGE-small), inside the
  1e-3 tolerance packs declare for their test vectors, so every pack
  compiled before this still verifies and serves: on every pack tested,
  test vectors, golden queries and abstention probes gave the same results
  under the old and new kernel. A pack rebuilt from the same inputs gets
  slightly different vectors and so a new identity. The kernel uses only
  baseline WASM SIMD, as before, and gives bit-identical embeddings in
  Node, Chromium, Firefox and WebKit.

- **`pikelet` no longer depends on `@xenova/transformers`.** `create`
  (embedding mode `workers-ai`) now embeds passages at build time with the
  inline-transformer kernel, running a u8 export of `BAAI/bge-small-en-v1.5`
  rather than transformers.js's ONNX copy. The 36 MiB weight blob downloads
  on the first build, pinned by sha256, into `~/.cache/pikelet/models`
  (`PIKELET_MODEL_CACHE` moves it, `PIKELET_BGE_WEIGHTS_URL` names a
  mirror); a cached blob that fails the pin is fetched again. Passage
  vectors change slightly, so `pikelet rebuild` gives existing projects a
  new index: they are within 0.9999 cosine of fp32 BGE-small where the old
  path's were 0.996 to 0.998, and on BEIR SciFact, scored against live
  Workers AI query vectors, nDCG@10 is 0.6991 against 0.6992 before.
  Passages longer than 512 tokens are mean-pooled across windows instead of
  truncated. `--omit=optional` is no longer needed: a plain install is two
  packages, where it was 82 packages and six audit advisories (one critical)
  inside that dependency.

### Changed

- **Query embedding is about 1.65x faster.** Single-query p50 on a Ryzen 9
  4900HS: MiniLM 27.5 to 16.4 ms in Node (Chromium 26.4 to 15.8, WebKit
  28.0 to 18.0), BGE-small u8 76.0 to 47.1 ms; long documents about 1.4x.
  u4 weights gain little (about 1.06x): their 32-wide blocks each need a
  conversion and scale. On BEIR SciFact with BGE-small, nDCG@10 is
  unchanged (0.7071) and 298 of 300 top-10 lists are identical.

## pikelet-wasm 0.11.0 / pikelet 0.11.0 — 2026-10-24

Pack reads on Node no longer go through the runtime's fetch, whose
response decoder is vulnerable on unpatched Node releases, and the CLI now
requires a patched Node 22, 24 or 26. Upgrade if you mount packs over HTTP
from Node. This release also fixes an out-of-bounds read on crafted engine
snapshots, a signed-URL refresh that could mix bytes from two artifacts,
and an MCP server that stopped serving every pack when one failed to
mount. It adds a writable agent journal, packs compiled from git history
and past conversations, and sentence windows. Both packages move to 0.11.0
together, and `pikelet` requires `pikelet-wasm` ^0.11.0.

### Compatibility

- **`pikelet` requires Node 22.23.4+, 24.22.1+ or 26.12.1+** (`engines`
  was `>=20`). Earlier releases ship a fetch whose response decoder runs
  on any body a server labels with a `Content-Encoding`; 18.x, 20.x, 23.x
  and 25.x are end-of-life and will not get the fix. npm only warns on an
  engines mismatch and an existing install never re-checks it, so the CLI
  now prints a warning to stderr on every start under an unsupported
  version (`PIKELET_NO_NODE_WARNING=1` silences it). CI tests Node 22, 24
  and 26 instead of 18, 20 and 22. `pikelet-wasm`'s own range is unchanged.
- **Under Node, `pikelet-wasm/complete` resolves to a Node entry** whose
  `httpRangeSource` reads over `nodeFetch` (see Fixed) instead of
  `globalThis.fetch`. Pass `options.fetch` to choose otherwise. Browsers
  and Workers still get the portable module.
- **The CLI refuses unknown flags** with `Unknown flag --x`. An
  unregistered flag used to be read as taking a value and swallowed the
  next argument (`pikelet mcp --no-resolve-parents pack.pikelet` mounted
  nothing), and a typo was ignored. A script passing a flag pikelet never
  read now exits with an error.
- **Calibration is fit on the reader's own retrieval.** `compile` and
  `compact` now assemble the pack unscored, open it with the reader, and
  run every calibration probe through the same retrieval `query()` serves
  from (sketch scan, BM25 candidates, exact rerank, fusion, phrase pin),
  then write the fitted pack. Before, calibration searched a separate
  in-memory HNSW index and approximated fusion with 5 BM25 candidates
  instead of 24 and a placeholder distance for lexical hits outside its
  pool. Measured on a 1,608-record docs pack, the passages coverage was fit
  on matched the ones the reader scores for only 27% of calibration
  queries; they now match for all of them. Packs are unaffected until
  rebuilt; a rebuilt pack ranks exactly as before, and on that pack 4 of
  331 calibration queries changed `matchQuality` band, in both directions.
- **`compact` no longer re-embeds every record to refit calibration**: the
  refit runs on the compacted pack's own rows.

### Fixed

- **Pack bytes on Node no longer pass through the runtime's response
  decoder.** `pikelet-wasm` ships `nodeFetch`
  (`pikelet-wasm/complete/node-fetch.mjs`), a fetch over
  `node:http`/`node:https` that never decompresses, sends
  `Accept-Encoding: identity`, and refuses any other `Content-Encoding`
  (parsed as a list, any case). It refuses an `https` -> `http` redirect,
  bounds itself to 32 keep-alive sockets per host (the range source's
  default parallelism), and rejects `arrayBuffer()`/`text()` when a body is
  cut off. Under Node, `pikelet-wasm/complete` now resolves to a Node entry
  (export condition `node`; `workerd` and `browser` still get the portable
  module) whose `httpRangeSource` defaults to it and which exports it.
  `httpRangeSource` takes `options.fetch` to choose another transport, and
  exposes it as `source.fetch`; a chain walk opens every parent it reaches
  over HTTP with the head's transport (or `openPikeletChain`'s `fetch`
  option). Every network read the CLI makes uses it: URL packs and chain
  parents under `mcp`, `append` and `compact`, `--shelf`, `doctor`, the
  encoder-weights download, and the URL crawler. `pikelet` therefore needs
  `pikelet-wasm` 0.11.0 or later.
- **`httpRangeSource` refuses an encoded response on any transport.** A
  200 or 206 with a `Content-Encoding` other than `identity` fails closed,
  next to the existing `Content-Range` checks: a pack is content-addressed
  bytes, and an encoded range cannot be sliced. In browsers the header is
  only visible on same-origin responses or when the host exposes it. Its
  redirect walk also refuses `https` -> `http`.
- **The URL crawler still reads sites that compress regardless.** A page
  served with `gzip` or `deflate` despite `Accept-Encoding: identity` is
  inflated with `node:zlib` (crawled HTML only; packs stay identity-only).
  A page in any other coding is skipped with a warning naming the URL and
  the header.
- **Engine snapshots with inconsistent graph levels are refused.** An
  entry point below the snapshot's max level, or a level-l edge into a node
  whose top level is below l, made traversal read past a node's edge lists
  (a segfault in the native build, an out-of-bounds read in WASM). Such a
  snapshot now fails `import()` like any other malformed one. Both `.wasm`
  builds are rebuilt.
- **A signed URL re-resolved onto a replaced artifact fails.** When a
  signed URL expired and was resolved again, the reader adopted the new
  HEAD's ETag, so a pack re-uploaded under the same name was compared
  against itself and its bytes served alongside the old ones. The first
  ETag and size now stay pinned and a mismatch throws `ARTIFACT_CHANGED`.
  `If-Range` is sent only with a strong ETag.
- **One pack failing to mount no longer takes down `pikelet mcp`.** Each
  pack mounts on its own; the rest keep serving, `list_packs` lists the
  failures under `unavailable`, and the exit status is still non-zero.
  Before, the first failure made every tool call, `tools/list` included,
  return that error.
- **`create --force` replaces a project only after the new one builds**,
  and works when the temp directory is on another filesystem (WSL's `/tmp`
  vs `/mnt/c` used to fail with `EXDEV`). It refuses to replace the
  current directory, a parent of it, or a directory overlapping
  `--source`.

### Added

- **An agent journal** (`pikelet journal init|list|compact`, `mcp
  --journal`): a private pack an agent writes to over MCP with
  `write_note` and reads back with `latest`, `search` and `get_record`.
  Each note is one immutable layer; the chain compacts itself at the depth
  limit, writes are serialized across processes, and several sessions can
  share one journal. Notes are labelled agent-written, and their author is
  the MCP client, never the model's arguments.
- **Packs from git history and past conversations.** `compile --git
  <repo> [--git-rev <rev|range>]` makes one record per non-merge commit,
  cited by commit URL or sha, with git run without a shell and with the
  repository's program-running config neutralized. `compile
  --conversations <file|folder>` makes one record per exchange from Claude
  Code transcripts, Claude.ai exports and ChatGPT exports, with tool output,
  credential shapes and email addresses redacted (best effort: keep such a
  pack private).
- **`info().layout` and `recordRange(id)`** report where each part of a
  `.pikelet` file and each record's bytes live, without a read. The
  playground draws its range reads on a map of the file with them.
- **`appendLayer` and `compactChain` run programmatically**, with a `log`
  option, ready-made `chunks` for `append`, and the written file's
  identity and records returned.
- **`pikelet` warns on a plain `http://` pack or shelf URL** (loopback
  excepted). The digests still refuse altered bytes, but anyone on the
  path can see what is read and stall or block it.
- **Sentence windows (`compile --sentence-windows <n>`, opt-in).** The
  index also holds a vector for every run of `n` consecutive sentences of
  each chunk, mapped back to the chunk by a new row-map segment (kind 8,
  `COMPLETE_PROFILE.md` 3.9); results, citations and the corpus stay whole
  chunks. A chunk's single vector averages everything it says, so a
  one-sentence answer could lose to a chunk merely about the same topic.
  On a 3,742-chunk docs corpus with 51 judged questions, `n = 2` raised
  vector-only nDCG@10 from 0.551 to 0.595 (hybrid 0.592 to 0.600), at
  9.4 rows per chunk: file 31.9 to 47.2 MiB, HTTP query 49 to 111 ms.
  Readers without kind-8 support refuse such a pack (row-count check)
  rather than misread it; `append`, `compact` and chain mounts refuse it
  until the layered profile defines rows. `info()` gains `indexRows` and
  `rowMap`.
- **Reproducible builds.** `compile` honours `SOURCE_DATE_EPOCH` for the
  build time it records in the evaluation segment (and `create`/`rebuild`
  in the project manifest). The build time was the only nondeterministic
  input, and it is part of a pack's identity, so every rebuild of the same
  inputs used to get a new identity; with the variable set, the same
  inputs give byte-identical packs.

## pikelet-wasm 0.10.1 / pikelet 0.10.1 — 2026-10-11

Layered packs now serve the same grounding and fetch tuning as single
files, HTTP readers make fewer, larger range requests, and lazily read
keyword indexes cache what they read. Upgrade if you serve layered packs
over `pikelet mcp`: under 0.10.0 they returned no `grounding`, the fact
check a relevance verdict leaves to the reader. Both packages move to
0.10.1 together, and `pikelet` requires `pikelet-wasm` ^0.10.1.

### Compatibility

- **HTTP range sources coalesce rerank rows within 256 KiB by default**
  (was 16 KiB), and a pack's `recommendedGap` is a floor under the
  source's own gap rather than a replacement for it. On the 456,153-record
  Wikipedia pack a query makes about 38 range requests instead of 61 and
  transfers about 2.3 MiB instead of 0.9 MiB; results are unchanged. Pass
  `preferredGapBytes` to `httpRangeSource`, or `gap` to `query()`, to
  choose otherwise.

### Fixed

- **A layered pack returns `grounding`.** The chain reader kept only the
  inherited scorer's verdict and confidence, so `grounding.covered` and
  `grounding.uncovered` were missing from chain queries and from MCP
  `search` sections as soon as a pack gained a layer. `recordId` is the
  chain-global id of the passage the coverage was taken from.
- **A layered pack honors fetch tuning.** `gap`, `parallelism` and
  `maxRangeBytes` (and the mount-level `rerankGap`, `rerankParallelism`
  and `rerankMaxRangeBytes`) did not reach the tiers, and a tier's
  manifest `recommendedGap` never applied. Each tier now fetches as its
  own file would.
- **Lazily read keyword indexes cache postings and share reads.** A
  repeated query re-read every postings list, and query terms whose
  entries share an uncached term-table window each read it. Postings are
  now cached (4 MiB, by bytes), pending reads are shared, and a failed
  read is retried by the next query rather than cached.
- **`CompleteQueryResult` declares `grounding`** in the TypeScript
  types; `query()` already returned it.

## pikelet-wasm 0.10.0 / pikelet 0.10.0 — 2026-10-09

`matchQuality` changes meaning for packs compiled with this release: it
now measures retrieval relevance — whether the pack covers the
question's topic — instead of trying to say whether the answer is there.
Packs already built keep their behavior, since the calibration ships
inside each file. The inline query encoder runs about 1.65× faster in
Node.js and Chromium and about 2.2× in WebKit with identical embeddings,
and readers can now run 12-layer and 4-bit encoders. Both packages move
to 0.10.0 together, and `pikelet` requires `pikelet-wasm` ^0.10.0.

### Compatibility

- **Newly compiled packs score relevance.** `none` means the pack does
  not cover the question's topic; a question about a covered topic whose
  specific fact the pack lacks now comes back `strong` or `weak` instead
  of `none`. Do not read `strong` as "answered": an asked-for value in a
  result's `grounding.uncovered` is one no passage states. On the
  repository's labeled questions, answerable questions called `none`
  went from 12 of 44 to 0 (docs) and 26 of 50 to 0 (Veyra registry).
  Rebuild a pack to pick this up; packs compiled by 0.9.x keep their
  calibration.
- **The calibration summary changes shape.** The evaluation segment's
  `calibration` block gains `method` (`self-templates-v6`) and `target`
  (`relevance`); `cvAuc` is now measured against off-topic queries;
  `cvAucHard`, answerable vs. in-domain-unanswerable, is reported but no
  longer gates the fit; `stage1FloorAuc` is gone. Corpora that shipped
  `unscored` for lack of hard negatives or a low hard-negative AUC now
  calibrate.
- **MCP wording changes.** The `search` description, the note on a
  `none` result, the server instructions and `verify_pack`'s calibration
  note describe relevance; `verify_pack`'s calibration block gains
  `method` and `target` (null on older packs). Tool names and result
  shapes are unchanged.
- **Packs that declare a layer count other than 6, or 4-bit weights,
  open only in 0.10.0 readers**; 0.9.x readers refuse them at parse. No
  published pack declares either.

### Added

- **Inline encoders with 1 to 12 layers and u4 weights.** A kind-3
  declaration may give `layout.L` from 1 to 12 and `layout.Q: 4` with
  `layout.B: 32` (u4 weights, two per byte, an f32 scale and offset per
  block of 32); `Q` absent means u8 in blocks of 64, as before. Exported
  as `KERNEL_MAX_LAYERS`. The parser and the embedder check the layer
  count, the format and the blob length before the kernel runs. A u4
  BGE-small-en-v1.5 blob is 23.9 MiB against MiniLM's 24.3 and matches
  PyTorch at cosine 0.985 (1.000 against the same weights fake-quantized).
  `compile` has no flag for these layouts yet.

### Changed

- **The inline encoder runs its projections over tiles of four tokens**
  instead of one token at a time, so each layer's weights stream through
  memory once per tile. Median query encoding, before → after: about
  23 → 14 ms for a 10-word question and 213 → 125 ms for 100 words in
  Chromium; 32 → 14 and 285 → 134 ms in WebKit. Embeddings are
  bit-identical, so existing packs and their verification vectors are
  unaffected.
- **Calibration** (`compile`, `compact` refits) fits verified on-topic
  probes against off-topic ones — the built-in foreign bank, minus
  entries the corpus turns out to cover, plus gibberish — and places
  `none` at the 95th percentile of the off-topic scores. The
  in-domain-unanswerable probe classes are still generated, for the
  `cvAucHard` diagnostic only.
- **Documentation** (both READMEs, `docs/how-a-query-runs.md`,
  `docs/veyra-ablation.md`) describes `matchQuality` as relevance.

## pikelet-wasm 0.9.3 / pikelet 0.9.3 — 2026-10-06

Hardening for readers that mount packs from other publishers or hosts:
the index header commitment now covers the bytes actually parsed, and a
pack can no longer steer lazy reads, demand an exhaustive rerank, or turn
a malformed abstention asset into confident answers. Upgrade if you mount
packs you did not build, including over `pikelet mcp`. Both packages move
to 0.9.3 together, and `pikelet` requires `pikelet-wasm` ^0.9.3 so that
`pikelet mcp` serves through the hardened reader.

### Compatibility

- **A pack that declares a rerank depth over 4096 is served at 4096 by
  default**, unless the caller passes `rerank`. No published pack is
  affected.
- **A pack whose abstention asset has malformed `thresholds` or
  `coverage.topK` now reports `unscored`** instead of answering. Packs
  built by `compile` are not affected.
- **`compile` declares at most 4096** for a pack over 4096 records whose
  sweep does not meet its recall target, and warns. Other packs compile
  as under 0.9.2.

### Fixed

- **The index header commitment binds the bytes the reader parses.** The
  complete reader checked `manifest.index.headerSha256` on its own read of
  the sketch header, while the sketch parsed a separate read of the same
  range. A host answering the two reads differently could pass the check
  with the genuine header and have a forged one parsed, together with the
  integrity anchors it carries, replacing the index under a pinned
  identity. `PikeletSketchArtifact.open()` now takes
  `expectedHeaderSha256` and checks the header it parses.
- **Lazily read keyword segments cannot steer reads or inflate them.**
  Lexical segments over 8 MiB are read on demand without per-read
  verification. Reads are now confined to the segment's window, and each
  term entry is checked before its postings are fetched: inside the
  postings region, `df` no larger than the corpus, and a length a postings
  list of that `df` can occupy. `info().lexical.integrity` reports
  `'unverified-lazy-reads'` for such segments and `'segment-sha256'` for
  eagerly verified ones.
- **A pack's declared rerank depth is capped at 4096 by default.** The
  sketch header's `recommendedRerank` is a producer hint; a value near the
  row count made every query fetch, hash and rerank the whole vectors
  region. Readers default to at most 4096, keep the declared value as
  `declaredRecommendedRerank`, and still honor an explicit `rerank`. The
  published wiki, astro-docs and rust-book packs declare 200, 40 and 30.
- **`compile` and `append` no longer declare a rerank depth readers will not
  use.** `compile`'s rerank sweep fell back to the full row count when no
  depth up to 640 met its recall target; it now stops at the same 4096
  ceiling, records the recall measured there, and logs a warning when the
  target was not met. `append` caps a layer's exhaustive hint the same way.
  `measureRecommendedRerank` returns `targetReached` and `rerankCeiling`,
  and `pikelet-wasm/artifact` exports `MAX_RECOMMENDED_RERANK`.
- **A malformed abstention asset serves `unscored`.** Missing or
  non-numeric `thresholds` compared false against every score, so the pack
  answered every query as if calibrated; out-of-order or out-of-range ones
  gave verdicts the fit never defined. A `coverage.topK` outside 1–10 is
  refused too, since each counted passage is a record read on every query.
  The scorer now rejects such an asset and the pack reports `unscored`.
  Assets written by `compile` are unaffected.

## pikelet-wasm 0.9.2 / pikelet 0.9.2 — 2026-10-01

Makes the layered update path work as documented: `append` on a freshly
compiled pack needs no flag, `pikelet mcp --pack` takes the newest layer,
and `verify_pack` reports honestly on a chain. Both packages move to 0.9.2
together, and `pikelet` requires `pikelet-wasm` ^0.9.2, whose chain reader
supplies the integrity fields `verify_pack` reports.

### Compatibility

- **A pack compiled from the same source gets a different identity than
  under 0.9.1**, because its manifest gains `corpus.ingest`. Pins and
  citations to packs already published are unaffected; only new compiles
  change.

### Changed

- **`compile` records how a pack's records were made, so `append` works on
  it with no flag.** New packs carry `corpus.ingest` (the chunker and its
  parameters). Packs compiled before it still need `--assert-ingest`, and
  `append`'s error now prints the exact JSON to assert. `append` refuses a
  chain whose declaration names a chunker this `pikelet` does not
  implement. The new manifest field means a pack compiled from the same
  source gets a different identity than under 0.9.1.

### Fixed

- **`verify_pack` on a chain reports what it actually checked.** It returned
  "no tests" with no encoder, integrity or calibration figures, and a note
  blaming an "older build", so an agent concluded no verdict could be
  trusted. It now runs the base's golden queries and probes against the
  chain, labelled as the base's, skips a golden whose expected record a
  layer removed, and reports the base's calibration figures, the chain's
  calibration status and drift, and each member's integrity. A chain's
  `info()` gains `encoderVerified`, `corpusIntegrity` and
  `indexRowIntegrity`, and per-member integrity under `members`.
- **`pikelet mcp --pack` accepts the newest layer of an updated pack.** A
  layer passed to `--pack`, or named by a shelf entry without a `lineage`,
  failed to mount with "unsupported profile pikelet-layer-v1". It now mounts
  the chain by walking each layer's parent locator, as LAYERED_PROFILE.md 7
  specifies, locally or over HTTP; `#<sha256>` pins the head. Shelf entries
  with a `lineage` mount as before.

### Documentation

- **Both READMEs describe layered updates** (`append`, `compact`, `rebase`,
  mounting the newest layer, pins, the drift limit) and citations that
  survive updates. The root README's opening states that a pack fixes how
  its corpus is queried; the long Wikipedia demonstration moved to
  `docs/wikipedia-demonstration.md`, and the abstention signals to
  `docs/how-a-query-runs.md`. Stale claims were corrected: a "retrieval
  agreement" signal that does not exist, MCP returning zero results under
  `none`, and the old `PIKELET_SEARCH_EMBED_WORKERS` name.

## pikelet-wasm 0.9.1 / pikelet 0.9.1 — 2026-09-30

Fixes to the layered profile shipped in 0.9.0, headlined by a chain that
lost its abstention verdicts as soon as a layer deleted anything. Both
packages move to 0.9.1 together, and `pikelet` requires `pikelet-wasm`
^0.9.1 so that `pikelet mcp` serves chains with the reader fixes.

### Compatibility

- **Layers from `append` and `rebase` now record a parent locator** when
  the parent is a local file under the output's directory, so the same
  inputs produce a layer with a different identity than under 0.9.0.
- **`--remove` and `--supersede` refuse ids they used to truncate**
  (`1,2,3`, `12abc`, `1.5`, `1e3`); give one id per flag.
- **A chain-member location ending in `#` plus hex is read as a pin.** A
  full 64-hex suffix is verified against the member's identity, and a
  shorter one is refused; it is no longer treated as part of a filename.

### Fixed

- **`pikelet-wasm` now exports `./package.json`.** `require.resolve('pikelet-wasm/package.json')`
  and JSON imports of it threw `ERR_PACKAGE_PATH_NOT_EXPORTED`, which
  tooling that reads the installed version relies on.
- **`pikelet append` parses `--remove` and `--supersede` ids strictly.**
  `--remove 1,2,3` used to tombstone record 1 alone, and `--remove 12abc`
  record 12, without an error; each flag now takes exactly one decimal id.
- **`#<sha256>` pins on chain members are enforced.** Every layered
  producer's usage line advertised `--parent <file|url>[#identity]`, but
  nothing parsed the pin: a local path failed as a missing file, and a URL
  dropped the fragment and built on whatever the server returned. `--parent`,
  `--head`, `--onto`, `--old-parent` and `--layer` now verify each pinned
  member's manifest identity before opening anything, and refuse a
  truncated pin.
- **`append` and `rebase` record a parent locator by default.** They wrote
  one only when given `--parent-locator`, so a rebased layer could not be
  mounted from its head alone and its next rebase needed `--old-parent`.
  When the parent is a local file under the output's directory, its relative
  path is now recorded. An explicit `--parent-locator` is validated up front
  instead of being written as-is and refused by every reader.
- **A URL chain mounts from its head alone, and locator redirects are
  confined.** Walking up from a URL head failed at the first hop ("a locator
  needs the child's own location") because a range source carried no URL,
  and a parent's URL would then have been read as a file path. The walk now
  resolves each parent over HTTP. A parent named by a publisher's locator is
  confined to the child's origin and directory at every redirect hop, as
  5.1.1 requires; lineage and host-resolver locations are the operator's
  choice and are not confined. `httpRangeSource` gains `url` and a
  `redirectGuard` option.
- **A tombstone-only layer no longer unscores its chain.** A layer that only
  deletes records carries no vocabulary bloom, and the reader counted that
  as a layer hiding its vocabulary, marking the whole chain
  `drift-exceeded`: on the astro pack, deleting 3 of 3,255 records (drift
  0.0009 against a 0.20 limit) turned every query's verdict to `unscored`.
  The rule now applies only to layers with records, and LAYERED_PROFILE.md
  4.5 and 5.5 say so. Tombstone-only layers already written with 0.9.0
  score again with no rebuild.
- **`pikelet --help` documents `append`, `compact` and `rebase`**, including
  `--supersede`, `--allow-drift`, `--max-depth` and pins.

## pikelet-wasm 0.9.0 / pikelet 0.9.0 — 2026-09-29

Everything merged to `main` since 0.8.0 that the 0.8.1 and 0.8.2 point
releases below did not carry, headlined by the layered profile. Both
packages move to 0.9.0 together. Pre-1.0, this minor carries breaking
changes; they are listed first.

### Breaking / compatibility

- **`pikelet` 0.9.0 requires `pikelet-wasm` ^0.9.0.** Its new `append`,
  `compact` and `rebase` commands and its chain mounts import
  `pikelet-wasm/complete/layer-*.mjs` and `complete/format.mjs`, which no
  0.8.x exports; upgrade both together. A layer's profile string,
  `pikelet-layer-v1`, is refused by every earlier reader by design: a layer
  opened alone would serve a fraction of a corpus.

- **The `PANCAKE_ENCODER_WEIGHTS_URL`, `PANCAKE_SEARCH_EMBED_WORKERS`,
  `PANCAKE_SEARCH_STUB_EMBEDDINGS`, and `PANCAKE_SEARCH_PYTHON`
  environment variable fallbacks are removed.** 0.7.0 introduced
  `PIKELET_ENCODER_WEIGHTS_URL` / `PIKELET_EMBED_WORKERS` while still
  honoring the old names; those old names no longer work. Use
  `PIKELET_ENCODER_WEIGHTS_URL`, `PIKELET_EMBED_WORKERS`,
  `PIKELET_SEARCH_EMBED_WORKERS`, `PIKELET_SEARCH_STUB_EMBEDDINGS`, and
  `PIKELET_SEARCH_PYTHON`. Pre-1.0; scripts/CI referencing the old
  names need updating.
- **`pikelet-wasm` declares `engines.node >=18`** (was `>=16`).
  `complete/sources.mjs` uses the global `fetch`, which Node 16 does not
  have; CI tests 18, 20 and 22. The `pikelet` CLI is unchanged at
  `>=20`.

### Added

- **Layered search artifacts** (`LAYERED_PROFILE.md`, Draft 4). A pack can
  be updated by appending a small `pikelet-layer-v1` file instead of
  recompiling: a layer carries only its new records plus a cumulative
  tombstone mask and supersession edges, and commits to its parent's
  identity, so a chain's head identity pins every byte a query can touch.
  `openPikeletChain` (`pikelet-wasm/complete/layer-reader.mjs`) mounts base
  and layers, verifies every parent commitment and spec rule at mount, and
  queries them as one corpus: masked candidate generation per tier,
  chain-global BM25 statistics, one fusion, and the base's abstention fit
  inherited until calibration drift passes the reader's limit (0.20 by
  default), after which results ship `unscored`. Records carry the layer
  that owns them; `record()` reports tombstoned records and their
  successors; `citation()` resolves ids against any head on the mounted
  history. A chain of local files mounts from its head alone through parent
  locators; URL chains mount from a shelf `lineage`.
- **`pikelet append`** writes a layer over a base or chain: new documents
  from `--source` (repeatable), deletions with `--remove`, and replacements
  with `--supersede <id>=<path>`, reusing the base's own encoder, chunking
  and vocabulary-bloom geometry. It refuses a layer that would push
  calibration drift past the limit unless `--allow-drift` is given.
- **`pikelet compact`** collapses a chain into a fresh depth-0 base: live
  records and quantized rows copied byte for byte, ids renumbered densely,
  exact lexical statistics, and a refit abstention calibration (`--no-refit`
  ships unscored). A lineage segment lets the compacted base translate
  citations made against the chain.
- **`pikelet rebase`** moves a layer onto a different head (for instance a
  fork sibling), replaying only what the layer itself did, with
  `--on-conflict refuse|skip|keep-both` and `--on-foreign refuse|drop`.
- **MCP serves chains.** A shelf entry with a `lineage` mounts a chain,
  pinned by its head identity; `list_packs` reports `chain`, `layers`,
  `tombstones` and `calibrationStatus`, and counts live records; `get_record`
  and search results carry `tombstoned`, `layer`, `supersededBy` and
  `currentSuccessor`.

### Changed

- **The repository is an npm workspace with the two packages under
  `packages/`.** `packages/pikelet-wasm/` holds the published library —
  entrypoints and readers under `src/` (`src/core`, `src/artifact`,
  `src/errors`, `src/loader`), the C++ engine under `engine/`
  (formerly the root `src/`), `complete/`, `native/` and the four tracked
  `dist/` build outputs; `packages/pikelet/` holds the CLI (formerly
  `pikelet/`). The root `package.json` is a private workspace root
  carrying the test ladder and dev tooling; `npm install` links
  `node_modules/pikelet-wasm` and `node_modules/pikelet` to the checkouts,
  so tests, benchmarks and examples import `pikelet-wasm` exactly as
  consumers do, and the CLI's and Docusaurus plugin's "repo root is
  pikelet-wasm" resolution fallbacks are gone. The published tarball's
  file list is unchanged: `main`/`types`/`exports` point into `src/`, and
  `CHANGELOG.md`, `LICENSE`, `NOTICE` and `README.md` are staged into the
  package by an npm `prepack` script instead of being duplicated in the
  tree. `run_tests.js` lives at `test/run_tests.js`; the engine benchmark
  scripts, whose relative requires had been left pointing one directory
  too high by the benchmarks move, resolve the package again.
- **Examples renamed, not renumbered**: `examples/04-static-wiki-pack` →
  `examples/static-wiki-pack`, `examples/05-one-file-search` →
  `examples/one-file-search`, `examples/06-mcp-knowledge-pack` →
  `examples/mcp-knowledge-pack`; `npm run demo:veyra` and the committed
  `veyra.mcp.json` follow. The raw browser demo page moves from `dist/`
  to `examples/technical-demo.html` (loads `../dist/engine.js`), and
  `docs/SYSTEM_DESIGN.md` is `docs/architecture.md`. The wiki playground's
  built bundle and deployment are unchanged.
- **README trimmed to the pitch, quickstart, live demo, numbers and
  links** (577 → ~340 lines). Its long-form sections moved verbatim to
  `docs/how-a-query-runs.md` (what is inside the file, compression cost,
  remote execution, integrity), `docs/veyra-ablation.md` and
  `docs/why-a-file.md`, linked from a new "Deeper dives" section.
- **Benchmarks consolidated under `benchmarks/`**: `bench/beir` →
  `benchmarks/beir`, `bench/range-proof` → `benchmarks/range-proof`, the
  engine suite (formerly `benchmarks/`) → `benchmarks/engine`, and the
  committed release runs (formerly `benchmark_results/release`) →
  `benchmarks/results`, with raw runs under the ignored
  `benchmarks/results/raw`. `npm run bench` runs `benchmarks/engine/run.js`.
  Nothing published changes.
- **Abstention calibration is fit against a different negative pool.**
  Hard negatives are now ablation pairs (a verified positive's own text,
  scored with its source excluded), entity swaps (a proper noun replaced
  by a corpus entity that never co-occurs with it), and held-out-document
  questions (title questions scored with that document excluded). A fit
  whose `d0` weight comes out non-negative ships `unscored` instead. The
  calibration summary (`fitAuc`, `cvAuc`, `cvAucHard`, per-class counts)
  is persisted in the evaluation segment and reported by `verify_pack`;
  golden queries are sampled from retrieval-verified positives. Packs
  rebuilt with this calibrator get a different fit and therefore a
  different identity; existing packs are unaffected.
- **`query()` returns `grounding`** — `{ recordId, coverage, covered,
  uncovered }`: which of the query's content words the best passage
  restates and which no passage contains — and each result carries
  `vectorRank`, `lexicalRank` and `fusedRank`. A verbatim match of three
  or more consecutive query words is pinned to rank 1 ahead of semantic
  near-misses. Phrase pinning and the coverage term now hydrate their
  candidates in one batch.
- **MCP `search` ships results under a `"none"` verdict by default**,
  with a note, instead of withholding them (`showAbstained: false`
  restores withholding); packs without a fitted calibrator get an
  `unscoredWarning`. Each section carries `grounding`. The tool
  description and `server/discover` instructions describe `matchQuality`
  as how directly the text restates the question, not whether the answer
  is present, and tell the caller to treat a queried-for term in
  `grounding.uncovered` as unestablished and to search again for the
  second hop of a chained question.
- **The reader no longer reads `coverage.useMaxSim`**; the word-level
  semantic grounding path is removed rather than left dormant (a pack
  with the flag set was measured at 1.5–4.6 s per query).
- `scripts/calibration-harness.mjs` reports false-abstention and
  false-answer rates on labeled question sets; the sets live under
  `test/relevance/`.
- Public browser playground for the Wikipedia pack, with an ablation
  tab.

- **`pikelet-wasm/complete` exports `fuseCandidates()` and
  `FUSION_DEFAULTS`.** The hybrid rank-fusion rule (reciprocal-rank fusion
  of the exact-reranked vector order with the BM25 order) is one function
  the reader, the calibrator (`calibrate.mjs`, which fits coverage on the
  fused top passage) and the BEIR ladder (`query-E.mjs`) all call, instead
  of three copies of the arithmetic. It takes `{ rrfK, lexicalWeight,
  guardMargin }`.
- **Hybrid fusion weights the lexical rank at 0.5 and keeps a confident
  vector top-1 in place.** `FUSION_DEFAULTS` is now `{ rrfK: 60,
  lexicalWeight: 0.5, guardMargin: 0.05 }`: the BM25 rank term counts half
  as much as the vector rank term, and when the vector top-1 leads the
  top-2 by at least 5% of its own distance it keeps rank 1 regardless of
  fusion. Under equal weights any record present in both lists outranked a
  vector-only rank 1 (`1/(60+v) + 1/(60+l) > 1/60` for every finite `v`),
  so a query whose terms match many chunks — a character's name in a
  novel — filled the top ranks with term matches and buried the passage
  the encoder placed first. Measured with the evidence-rank report and the
  fusion simulator (`test/relevance/README.md`): evidence in the top 3 on
  Pride & Prejudice 36 → 42 of 157 direct questions, internal-docs
  paraphrases 19 → 21 of 24, Veyra direct 19 → 20 of 20 and multihop
  17 → 18 of 20; BEIR E by fused rank scifact nDCG@10 0.694 → 0.699
  (recall@10 0.827 → 0.853), nfcorpus 0.3345 → 0.3359, arguana unchanged.
  Verdicts on packs calibrated under the old rule move the same way (false
  answers: internal-docs 10 → 9, Pride & Prejudice 74 → 66, Veyra 13 → 13);
  a Veyra pack recalibrated under the new rule fits at cvAUC 0.890 (was
  0.870). Packs need no rebuild to get the new ranking.
### Fixed

- **Engine constructors validate their configuration.** `pikelet_init`
  (WASM and the native addon) passed caller-supplied `dims`, `M` and
  `max_elements` straight into the index constructors. `M = 1` made the
  level multiplier `1/log(1) = +inf`, whose conversion to an integer on
  the first insert is undefined behaviour, and `max_elements * dims` or
  `max_elements * M0` could wrap a 32-bit `size_t` on wasm32 into an
  undersized arena. Both backends now reject `dims` outside 1..65536, `M`
  outside 2..128 (0 still selects the default), `max_elements = 0`, and
  products that do not fit `size_t`; WASM returns `INVALID_HANDLE`, the
  native addon throws a `RangeError` (a C++ exception used to escape the
  N-API boundary and abort the process). `Pikelet.create()` mirrors the
  `dim` ceiling with an `INVALID_ARGUMENT` error, and the native
  `pikelet_import` rejects a non-buffer argument with a `TypeError`
  instead of continuing with a null view while a JS exception is pending.
- **Artifacts with a NaN or infinite scale/offset are rejected.** A
  producer signs whatever bytes it writes, so a non-finite affine pair in
  a `.pikelet-range` record or a sketch artifact's affine table passed
  every digest check and then made that row's distance non-finite, which
  the result heap never evicts. The range reader rejects the record at
  decode and the sketch reader rejects the table wherever it adopts it,
  both with `SNAPSHOT_INVALID`.
- **`stripMarkdown` no longer backtracks quadratically on backtick runs.**
  The fenced-block and inline-code regexes took seconds to minutes on a
  line of 40k backticks (each candidate fence length re-scanned the rest
  of the document; each suffix of an inline run was retried). Both are
  replaced by linear scanners with CommonMark's rules: a fence closes on
  the first later fence line of the same character at least as long, an
  unclosed fence runs to the end of the document, and an inline span of N
  backticks closes on the next run of exactly N on the same line, may
  contain runs of other lengths (`` ``a ` b`` ``), and drops one padding
  space from each end. Under the old regexes a shorter closing fence or
  an unclosed one left the block as prose.
- **Folder ingestion reports what it skips and caps file size.** The walk
  skipped symlinked files and directories silently (Dirents carry lstat
  semantics); it now logs `warn: skipped symlink <path>` for each. Files
  over 16 MiB are skipped with a warning naming the path and the limit
  instead of being read whole into the build, matching the crawl path's
  body cap.
- **`pikelet rebuild --verbose` parses.** `--verbose` was read as a
  boolean (`verbose: !!flags.verbose`) but was not registered as one, so
  as the last argument it failed with "Missing value for --verbose" and
  before another flag it swallowed that flag as its value.
- **`verify_pack` surfaces a corrupt evaluation segment.** The MCP tool
  turned any `evaluation()` failure into "no evaluation", reporting 0
  golden queries and 0 probes, all passed, for a pack whose evaluation
  segment failed its hash or did not parse. Those errors now propagate
  as tool errors; a pack with no evaluation segment still reports zero.
- **Generated Workers no longer echo internal error messages on 5xx.**
  The `pikelet` scaffold's `worker.js` / `worker.artifact.js` templates
  returned `error.message` for every failure, including unexpected ones,
  which can name bundled asset paths and load state. A 5xx now carries
  `{ error: "Internal server error", code }` and the original error goes
  to the Worker log (`wrangler tail`); 4xx and 503 (`EMBED_UNAVAILABLE`)
  bodies still carry the message. The legacy reference Worker's
  "Invalid JSON body" error no longer appends the parser's message.
  Projects generated by earlier versions keep their old template.
- **References the 0.8.0 rename left broken are repaired.** The engine
  benchmark scripts called `native.pancake_*` (the N-API addon exports
  `pikelet_*`); the ANN-Benchmarks adapter imported `pancake_py` (the
  pybind module is `pikelet_py`) and, after the benchmarks move, resolved
  the repo root one level short; `dist/technical-demo.html` and
  `test/ios-smoke.html` called `_pancake_*` WASM exports (the iOS page also
  inlined a pre-rename engine build and used the pre-`ef_search` query
  ABI — regenerated from the current `dist/`); the Docusaurus plugin's
  webpack rule matched `pancake-(artifact|errors).js`, which no longer
  exists, so it never applied; READMEs showed `openPancakeFile`,
  `PancakeRangeArtifact`, `PancakeSketchArtifact` and `PancakeIndex`. The
  scaffold template's UI title and the wiki example page's wordmark and
  GitHub links now say Pikelet. `docs/history.md` records the rename and
  lists what intentionally keeps the old name.
- **`compact()`'s rebuild path no longer leaves the index half-cleared**
  when a reinsertion fails: the old graph is released only after the
  rebuilt one is complete, so a failed rebuild returns with the index
  unchanged. Previously the next `search`/`insert` read out of bounds.
- Coverage word rules: possessives (`darcy's`) match plain mentions,
  eleven quantifier/hedge words are stopwords, and numerals are exempt
  from the minimum word length (`Chamber 4` vs `Chamber 5`).
- Ingest: `**/dir/**` globs no longer substring-match sibling names,
  minified HTML with unquoted attributes chunks correctly, a leading BOM
  no longer breaks frontmatter, numeric entities decode, and
  `stripMarkdown` no longer deletes hyphens, underscores and minus
  signs; merging a section with a child heading keeps its provenance;
  ingestion conformance runs on Windows.
- Browser verification without a WebCrypto backend fails with a clear
  error instead of an obscure one.

## 0.8.2 — `pikelet`

### Fixed

- **`pikelet mcp` answers `initialize` without waiting for its packs to
  mount.** The server mounted every pack before replying to the MCP
  handshake, and a URL mount fetches the pack's resident prefix first: about
  5.6 s for the rust-book pack and 9.2 s for the 649 MiB Simple English
  Wikipedia one. The published three-pack shelf took **10.2 s** to answer
  `initialize`, past the point most MCP clients give up — so the documented
  `mcp --shelf` command timed out for a new user.

  Mounting now runs concurrently with the protocol loop. `initialize` returns
  in about **0.6 s**; `tools/call` and `tools/list` await the mounts, so a
  client gets a fast handshake and pays the mount cost on its first real call.
  `tools/list` waits deliberately: it advertises the mounted pack names, and
  answering early would name fewer packs than the server goes on to serve.

### Documentation

- The install instructions lead with `npm install -g pikelet --omit=optional`.
  Measured on a clean project, that is 2 packages / 1.8 MB / "found 0
  vulnerabilities" against 82 packages / 257 MB / "6 vulnerabilities (5 high,
  1 critical)" — every advisory is inside `@xenova/transformers`, which only
  the `create` scaffold path loads. The previous README put the flag in a
  mid-paragraph aside and estimated its cost at "~140 MB".
- README carries an upgrade note for the `pikelet-wasm@0.8.1` heap-overflow
  fix, which shipped without one.

## 0.8.1 — `pikelet-wasm`

Security release, cut from `main` ahead of the rest of the backlog, which
shipped in 0.9.0. `pikelet` is unchanged at 0.8.1 and picks this up through its
`^0.8.0` dependency range.

### Security

- **A `.pikelet` can no longer overflow the encoder's WASM heap through an
  out-of-range `maxTokens`.** `complete/inline-transformer.mjs` sized the
  kernel's ids and hidden-state buffers from the artifact's own encoder
  declaration, clamped with `Math.min(declaration.maxTokens || 512, 512)` —
  an upper bound only. A pack declaring `maxTokens: 1` made the windowing
  arithmetic (`windowLen = maxSeq - 2`) negative, so `interior.slice(w *
  windowLen, (w + 1) * windowLen)` returned nearly the whole token sequence
  and the forward pass wrote past both allocations: a 60-token query wrote
  about 92 KB into a 1,536-byte buffer. `maxTokens` is now validated as an
  integer in `[8, 512]` at parse, before anything is allocated, `maxSeq` is
  clamped on both sides, and both call sites assert the window fits.

  This fired at **mount**, before any query: the reader awaits the encoder
  for a non-deferred query-interp segment (and prefetches it otherwise), and
  the declaration's own `testVectors` are verified after the buffers exist,
  so a long test vector was itself a trigger. Every integrity check passed,
  because the attacker authored the file and its digests bind their bytes.

  Affects anyone who mounts a `.pikelet` they did not compile — the format's
  intended use, since packs are distributed and range-read from URLs. The
  write is confined to the encoder's WASM linear memory and does not reach
  host JavaScript objects, which is why it is rated high rather than
  critical. Upgrade to get the bound; there is no workaround short of not
  mounting third-party packs.

## pikelet 0.8.1 — 2026-09-18

- `pikelet` records `mcpName: "io.github.mcn92/pikelet"` and ships
  `server.json`, the manifest for the official MCP registry entry
  (`pikelet mcp <pack>`; one required positional argument, any
  `.pikelet` file). No code change.

## pikelet-wasm 0.8.0 / pikelet 0.8.0 — 2026-09-14

### Breaking / compatibility

- **Manifest profile strings are renamed** `pancake-complete-v1/v2` →
  `pikelet-complete-v1/v2`. New builds write the `pikelet-` string;
  readers accept either for the same format version, so every pack
  published before this release keeps opening with its identity
  unchanged. Because the profile string is part of the manifest that
  the identity hashes, **a 0.8.0 compile of the same inputs produces a
  different identity than 0.7.0 did**. This reverses the 0.7.0 note that
  the wire format would keep its pancake-era names.
- **Published API names are renamed** to their Pikelet forms:
  `PancakeError`/`PANCAKE_ERROR_CODES` → `PikeletError`/
  `PIKELET_ERROR_CODES`, `PancakeIndex` → `PikeletIndex`,
  `PancakeRangeArtifact`/`PancakeSketchArtifact` → `PikeletRangeArtifact`/
  `PikeletSketchArtifact`, `PancakeApi`/`NodePancakeApi` → `PikeletApi`/
  `NodePikeletApi`, `openPancakeFile`/`assemblePancakeFile` →
  `openPikeletFile`/`assemblePikeletFile`. The compatibility aliases are
  dropped. The C ABI's `pancake_*` symbols are `pikelet_*`; the native
  addon source is `native/pikelet_napi.cpp`. The `.pancake-range` and
  `.pancake-sketch` extension conventions are `.pikelet-range` and
  `.pikelet-sketch` (readers dispatch on magic bytes, not extension).
- **`export()` refuses while deleted vectors are resident.** The snapshot
  format has no deleted flag, so a snapshot taken with ghosts present
  resurrected them on import. C++ `serialize()` throws, the WASM
  `pikelet_export` returns null, the native binding throws; call
  `compact()` first. The JS wrapper already enforced this; the native
  path did not.
- **`@xenova/transformers` is an optional dependency of `pikelet`.** It
  is needed only by the scaffold (`create`) path; `npm install -g pikelet
  --omit=optional` skips its ~140 MB for `compile`/`mcp`-only use.
- `pikelet` depends on `pikelet-wasm ^0.8.0`.

### Added

- **Swappable inline encoder.** `compile --encoder-model
  --encoder-weights --encoder-vocab --encoder-pooling mean|cls
  --encoder-query-prefix --encoder-passage-prefix` records a different
  BERT-shaped model (same hidden size, layers, heads and vocab size as
  MiniLM-L6, e.g. Snowflake `arctic-embed-xs`) as artifact data; CLS
  pooling is honored end to end, and `export_encoder_blob.py` checks the
  architecture before exporting.
- **The inline-transformer kernel's window is 512 tokens** (was 128).
  Chunks longer than 128 tokens are no longer silently truncated at
  embed time; long chunks cost proportionally more to embed. Existing
  artifacts declaring `maxTokens: 128` open and verify unchanged.
- **`query({ showAbstained: true })`** returns the retrieval under a
  `"none"` verdict alongside the unchanged `matchQuality`/`confidence`.
- `examples/06-mcp-knowledge-pack`: compile a corpus, mount it over MCP,
  answer with citations. Engine-reference examples move to
  `examples/legacy/`.
- `bench/beir/` retrieval-quality ladder (dense, hybrid RRF, lexical-only,
  Arctic-XS configs) and `bench/range-proof/` (dumb static server, MCP
  and headless-LLM proofs of the static-hosting claim).
- A committed MCP config makes the Veyra demo packs queryable
  interactively; the packs themselves are GitHub release assets fetched
  by `web/veyra-corpus/fetch-veyra.mjs` rather than tracked files.
- Encoder-conformance fixtures live under `test/fixtures/` instead of
  inside an example.

### Fixed

- **The abstention verdict no longer depends on the caller's `k`.** The
  scorer windows its inputs to the fixed top-10 the calibrator was fit
  on; at `k=3`, `margin`, `mean10` and coverage were computed over
  fewer candidates than at fit time and flipped verdicts between
  `strong` and `weak` for identical retrieval.
- **Coverage is scored against the fused (vector + BM25) ranking**
  in both the calibrator and the reader, with the lexical cutoff shared
  between them. Calibration was fit on vector-only signals while the
  reader served hybrid results; on a 2,841-record Wikipedia sample every
  measured false abstention had the correct record at hybrid rank 1–6.
- `httpRangeSource` treats a `200` whose ETag differs from the pinned
  one as `ARTIFACT_CHANGED` instead of a Range-unsupported host; the
  sketch row-digest floor is 16 bytes (readers still accept 8–32);
  `deserialize()` validates every header field before assigning it; the
  L2 insert/search path rejects non-finite vectors (Cosine already did).
- `chunkDocs` no longer emits title-only stub records from leading
  undersized sections or attributes a merged chunk to the wrong section.
- Calibration probes: phrasing diversity across question frames, a
  saturation rail on the hard threshold, and a minimum weak band.
- `spec/COMPLETE_PROFILE.md` Draft 2 resolves the corpus-compression,
  `sampleQueries` placement and browser-packaging questions.

## pikelet-wasm 0.7.0 / pikelet 0.7.0 — 2026-09-02

### Breaking / compatibility

- **The project is renamed to Pikelet.** Packages: `pancake-wasm` →
  `pikelet-wasm`, `create-pancake-search` → `pikelet` (one CLI:
  `npx pikelet create|compile|doctor|mcp`; the old `npm create
  pancake-search` flow becomes `npx pikelet create`). New compiles
  default to a `.pikelet` extension. What deliberately does NOT change:
  the wire format — magic bytes, manifest profile strings
  (`pancake-complete-v2` and family), segment kinds, and reader error
  text keep their pancake-era names, so every existing artifact
  identity (the published wiki pack included) stays valid, and
  `.pancake` files remain fully readable (readers dispatch on magic
  bytes, never extension). Library compatibility: `openPancakeFile` and
  the other exported API names are unchanged; `openPikeletFile` is the
  documented alias. Env vars gain `PIKELET_ENCODER_WEIGHTS_URL` /
  `PIKELET_EMBED_WORKERS` (old names still honored). The Docusaurus
  plugin's surface renames with the package (default asset dir
  `pikelet-search/`, mount attribute `data-pikelet-search`, global
  `window.PikeletDocusaurusSearch`, debug attribute
  `data-pikelet-debug`); sites upgrading from the pancake-named plugin
  update their config and any custom CSS hooks. Release-asset URLs in
  code and docs keep their original hosts/paths — GitHub redirects them
  after the repository rename, and the reader resolves and pins
  redirects. Benchmark backend labels in committed result sets
  (`pancake-wasm-int8` and friends) are measurement lineage and stay.

- **The Docusaurus plugin's default output is now the complete kind-3
  `.pancake`**, served by the widget over genuine HTTP range reads — the
  sketch profile's execution model, verified in a real browser (zero
  search bytes until the panel opens, then 206 range responses with the
  kind-3 encoder prefetched in the background). Previously the default
  was the deprecated `.pancake-range` plus a Python-trained student
  encoder; the flagship integration shipped the geometry the project's
  own measurements had retired, and the complete-profile mode downloaded
  the whole artifact into memory. Zero-config installs now stage the
  packaged encoder assets (vocab shipped, weights digest-pinned and
  fetched once) — no Python in the toolchain. Opting out: `mode:
  'student'` (or any student option) keeps the small-footprint student
  path, `mode: 'artifact'` keeps the deprecated range output, and both
  log what they are. The widget defers all loading to the first panel
  open, renders section heading-path breadcrumbs on results, and hides
  raw distances unless `data-pancake-debug="1"` is set. The in-repo
  docs-site now exercises the zero-config default.

### Added

- **`create-pancake-search mcp` serves `.pancake` knowledge packs over
  the Model Context Protocol** — the first concrete piece of the
  knowledge-pack direction: an LLM client (Claude Code, Claude Desktop,
  agent frameworks) attaches compiled packs as a retrieval tool with one
  config line and no vector database, embedding service, or retrieval
  backend. `mcp --pack a.pancake --pack b.pancake` mounts any number of
  packs (named from the manifest, collision-suffixed) and exposes three
  tools over stdio: `search` (per-pack result sections — distances are
  only comparable within one pack — each result carrying pack name,
  immutable manifest identity, title, heading path, anchor, and source,
  so citations can pin the exact knowledge state an answer used),
  `list_packs`, and `get_record` (full integrity-verified chunk by id).
  Calibrated abstention crosses the protocol intact: a pack that cannot
  answer reports `matchQuality: "none"` with zero results and the tool
  result says to admit it rather than guess. The protocol layer is
  hand-rolled newline-delimited JSON-RPC (four methods; no new
  dependency); tool failures return correctable `isError` results, not
  protocol errors. Kind-2 packs (host-encoder) are refused at mount with
  an explanation; kind-1/3 packs are self-contained. The complete
  reader's `info()` now exposes the manifest's pack `name`. Verified
  live against real packs (docs.astro.build + nodejs.org mounted
  together: cross-pack grouping, off-domain abstention, honest
  `unscored` from a pack whose calibration failed its gate) and covered
  by scaffold-e2e checks driving the compiled fixture pack through the
  actual protocol.
- **`--pack` takes a URL** — the format's native habitat, wired into the
  mount: a URL pack is range-read off dumb HTTP, never downloaded whole.
  One line attaches the 456k-passage Wikipedia pack; mounting transfers
  the ~52 MiB resident slice (measured 6–8 s over the real internet,
  identity-pinned) and each question costs ~127 range requests against
  the 649 MiB file. Both URL and file mounts take `#<sha256>` to pin the
  manifest identity — a mount that finds different bytes refuses to
  serve. `--shelf <file-or-url>` mounts every pack on a static
  `packs.json` listing (`packs/` in this repo — a registry that is also
  just a file: name, location, identity, license per entry, no service,
  no accounts). All packs warm in the background at mount (encoder,
  kernels, first rerank round trips) so the first tool call pays for
  retrieval, not staging. `httpRangeSource` learned two things the wiki
  mount taught the hard way. First, redirect memoization: github.com
  302s every release-asset hit to a signed CDN URL, so ~130 range reads
  per query were charging GitHub's rate-limited web front door ~130
  times — the actual source of the 429 storms, while the blob CDN
  behind the redirect served parallel 206s happily. The reader now
  resolves the redirect chain once, pins the final URL (never appending
  the cache-key param to a signed query string), and re-resolves
  automatically when the signed URL expires (401/403); the front door
  is charged per mount, not per read, and the live wiki demo went from
  unable to finish one query to 6 s mount + ~1 s warm queries on the
  same rate-limited IP. Second, transient statuses (429/502/503/504)
  retry with capped backoff honoring `Retry-After`; `stats.retries` and
  `stats.redirects` count both mechanisms. New `verify_pack` tool runs the
  tests a pack carries inside itself: `compile` now embeds a sample of
  the calibration's retrieval-verified positives as golden queries
  (`evaluation.goldenQueries`), and the wiki pack's abstention probes
  run the same way — an agent audits a newly attached knowledge source
  with tests stored in the file it is auditing (24/24 on the fixture
  pack, over HTTP). `mcp install --client claude-code|claude-desktop`
  writes the client config (merge-safe, `--force` to replace) instead of
  documenting it. `compile --license <SPDX-id>` records a content
  license in the pack manifest (`corpus.provenance.license`), surfaced
  by `list_packs` and `info().license` — packs meant for redistribution
  should carry one. Scaffold e2e grows to 46 checks (verify_pack
  goldens, URL mount over a local range server, wrong-pin refusal,
  install round-trip).
- **The MCP server is dual-era: current protocol revision 2026-07-28
  served statelessly alongside the legacy initialize handshake.** Modern
  requests carry their protocol version in `_meta` per request; the
  server answers the mandatory `server/discover` (supported versions,
  capabilities, pack-usage instructions, cache hints), decorates modern
  results with `resultType` and server identity, returns
  `UnsupportedProtocolVersionError` (-32022, with the supported list)
  for unknown versions, and marks `tools/list` as a CacheableResult with
  deterministic tool order. Legacy clients (2025-11-25 and earlier) get
  the exact pre-modern shapes, initialize handshake included — era is
  selected per request, as the spec's dual-era rules describe. A
  protocol-only conformance suite (16 checks, stubbed pack, no compile
  or network) runs in `npm test`, so interoperability is verified in CI,
  both eras. Also: an identity pin (`--pack …#sha256`, shelf entries) is
  now enforced by the reader itself via `openPancakeFile`'s new
  `expectedIdentity` option — a mismatched artifact is refused after the
  64-byte header read, before the manifest or any segment is touched
  (conformance asserts exactly one read happens); `mcp install` pins the
  runtime version it writes (`create-pancake-search@<version>`) so
  content-addressed packs are not served by whatever npm publishes next
  month; and `verify_pack` describes itself as self-verification —
  proving the artifact is intact and behaves as built, not that its
  content is true or its publisher trustworthy.
- **The complete reader's resident scan accelerates through the engine's
  SIMD scan kernel — warm-query p50 at the 456k wiki scale drops from
  ~540 ms to ~129 ms** (measured A/B on a local file source; ~110 ms/query
  end to end in the kind-3 acceptance run). This closes the honest
  regression the 0.6.0 wiki-recovery entry named: the correct 192-dim
  sketch geometry made the pure-JS `O(count × sketchDims)` scan the warm
  bottleneck. `openPancakeFile` gains `sketchScanner`: undefined (the
  default) auto-stages `createSketchScanner` in the background in Node
  when the corpus crosses ~16M sketch cells — docs-scale artifacts scan
  in single-digit milliseconds and never load the engine; `false` forces
  the pure-JS scan; a scanner object or async factory
  `(sketchArtifact) => scanner` is the injection point for browsers,
  where the reader cannot import the Node engine entrypoint (the wiki
  web demo injects one — Vite splits the engine glue and wasm into lazy
  assets fetched only when the factory runs, verified staging in a real
  browser). Queries never wait on staging: the JS scan serves until the
  kernel is ready, and permanently if loading fails — acceleration only
  ever changes speed, never results (conformance asserts identical
  `[id, distance]` output, and the wiki recall gates reproduce 96.0% /
  95.2% exactly with the scanner active). `info().residentScan` reports
  `'engine'` or `'js'`. Hardening in the sketch reader: a query whose
  candidate pool C exceeds a scanner's creation-time buffers
  (`maxRerank`) falls back to the JS scan for that query instead of
  silently truncating the pool.

### Fixed

- **Both benchmark-reproduction defects a fresh clone would hit are
  fixed.** `test-inline.mjs` still pointed its download URL at the
  never-uploaded `artifact-wiki-inline-v3` release asset (its identity
  and size pins were already v4) — a fresh clone's acceptance run
  deleted nothing but 404'd; the URL now matches the pins. And the
  committed churn decay curve (`benchmark_results/churn_scale_100k.json`,
  measured 2026-07-11) still showed recall collapsing 96% → 7.2% over
  five population turnovers — behavior the churn-recovery work of
  2026-08-07 eliminated. Re-measured on the current engine: recall holds
  92.0–96.8% through all five turnovers with a full top-10 at every
  round (compaction 98.8%, fresh-build reference 96.8%); the committed
  result set and its README paragraph now describe the engine as it is.

## pikelet-wasm 0.6.0 / create-pancake-search 0.6.0 — 2026-08-31

### Breaking / compatibility

- **`create-pancake-search` rejects `--runtime complete`** on create and
  rebuild instead of silently building a snapshot project (the generated
  Worker templates only serve snapshot and artifact runtimes; `complete`
  passed validation but was dropped by the config builder). The error
  points at the new command below.
- **`--include`/`--exclude` against a URL source now error** instead of
  being silently ignored (they never applied to crawls). URL sources get
  their own filters: `--include-url`/`--exclude-url`, matched against the
  URL path with `*` as the only wildcard, and aggregate pages such as
  mdBook's `print.html` are excluded from crawls by default (they
  duplicate a whole site's content past exact-text chunk dedupe).
  `--include-url`/`--exclude-url` against a folder source error
  symmetrically.
- **The `.pancake-range` profile is deprecated** (contract §9.2). The
  sketch profile superseded it in every measured regime — depth-1
  execution ~5x faster end-to-end over real networks at roughly a third
  of the artifact size — and the complete `.pancake` profile is the
  product surface. Nothing breaks in this release: readers
  (`PancakeRangeArtifact`, `openRangeArtifactFile`) stay supported for
  existing artifacts, and the scaffold's `--runtime artifact` still
  works. The builders (`buildRangeArtifact[File]`) are marked
  `@deprecated` in the type declarations, the scaffold logs a deprecation
  warning when the artifact runtime builds, and no further range format
  revisions are planned. The Docusaurus plugin's default output still
  uses the profile; its `completeProfile` option is the recommended
  configuration.
- **Crawler fixes from a five-site independent-docs sweep** (Sphinx,
  MkDocs Material, Docusaurus, VitePress, Jekyll): URL filters now gate
  the queue instead of the dequeue, so filtered links no longer consume
  the `--max-pages` budget (a large excluded section — e.g. a versioned
  doc tree — used to starve discovery: docusaurus.io yielded 30 pages of
  a 70-page budget, 69 after the fix); self-recall validation tolerates
  near-duplicate content (original within top 5 passes as degraded with a
  logged hint toward `--exclude-url`, instead of hard-failing the build —
  versioned doc trees put several near-identical chunks at the same point
  in embedding space, where rank 1 among them is arbitrary); crawls that
  discover fewer than 5 pages warn that client-side-rendered navigation
  cannot be followed; the filter-skip log names which filters did the
  skipping.

### Added

- **The wiki pack's layout and sketch geometry are recovered — recall@10
  rises from 82.8% to 95.2% and requests/query drop from ~458 to ~127.**
  The pack pipeline's blessed build directory (`data-perm`: k-means
  cluster-ordered rows, 192-dim 2:1-pooled sketch) had been deleted from
  disk, and `compile-wiki.mjs` was written against `data-full` — whose
  sketch was 96-dim 4:1 pooling, the geometry the pack's own measurements
  had rejected (−11pt candidate capture). Every complete wiki artifact
  since 2026-08-14 embedded the degraded sketch in unpermuted row order;
  the range-storage ledger's request-count column exposed it. Recovered
  by regenerating `data-perm` from the surviving `layout-perm.npy`
  (ground truth mapped through the inverse permutation and
  lockstep-verified), rebuilding the pack (format-2 sketch, measured
  `recommendedRerank: 200`), and rebuilding the inline artifact. Three
  causes, three fixes:
  1. `compile-wiki.mjs` now prefers `data-perm` and warns loudly when
     building from the unpermuted source;
  2. the correct sketch geometry reaches target recall at C=200 instead
     of C=600 — the same root cause was suppressing recall and inflating
     request counts;
  3. cluster-ordered rows only coalesce at a gap matched to the cluster
     geometry, which nothing ever passed (even the original demo ran at
     the 2 KiB default): the manifest MAY now carry `recommendedGap`
     (spec §4 fetch hints) and the reader honors it unless the caller
     overrides — the wiki artifact ships 16 KiB.
  Measured on the rebuilt artifact (identity `1b180adf…`): augmented
  recall@10 95.2% at C=200 / 97.0% at C=600 (titles 94.8%, hand-written
  98.5%) with the artifact's own inline kernel — the true fp32 reference
  is 96.0% (kind-2, precomputed vectors; `options.encodeQuery` is
  ignored on kind-3 artifacts, so kernel-vs-fp32 comparisons need the
  kind-2 build), a ~0.8pt quantization tax; ~127 range
  requests/query over HTTP (≈ 88 sketch + 24 lexical + scoring), cutting
  R2 cost from ~$165 to ~$46 per million queries. Verified along the
  way: a reverse-Cuthill-McKee ordering extracted from the real graph is
  equal-or-worse than plain corpus order for the rerank access pattern
  in every (C, gap) cell — RCM served only the deprecated traversal
  profile and is not worth porting forward. Honest regressions of the
  correct geometry: open transfer grows to ~52 MiB (the true resident
  sketch) and warm-query p50 roughly doubles to ~500 ms (192-dim
  pure-JS scan — the WASM-scanner integration is now the top follow-up).
  The augmented-vs-vector recall gap shrinks to +0.3pt: correct
  candidate geometry absorbed most of what lexical rescue had been
  compensating for.
- **Search ledger benchmark** (`scripts/benchmark-search-ledger.mjs`):
  renders artifact performance on the axes a general audience can price —
  time to first answer from nothing, per-query requests/KiB and dollars
  per million queries on parameterized object-store pricing (R2 and S3
  defaults), queries per GB of egress, storage cost, warm-session
  latency, and an infrastructure column that reads zero. Composes
  benchmark-range-storage across named distance presets
  (city/continent/world as injected per-response delays, labeled as a
  model) and emits a Markdown report with the reproduction commands
  embedded. Steady-state economics exclude the first query's one-time
  costs (the deferred encoder download) — charging those to every query
  overstated cost ~50× in the first draft. Representative output at
  docs scale: 42 KiB and 6 requests per query — $2.19 per million
  queries on R2, 23,375 queries per GB, $0.0004/month to store.
- **Wiki-scale hybrid search: the lexical path now works at 456k chunks,
  with an `augmented` retrieval mode that lifts the pack's headline
  recall.** Three pieces:
  - `buildLexicalSegment` is rebuilt as a counting-sort over typed arrays
    (terms interned to dense indexes, entry arrays preallocated from df
    prefix sums, one scatter pass) — the Map-of-Maps build died in the
    tens of millions of posting entries; the wiki corpus (456,153
    records, ~30M postings, 784,693 terms, 71.2 MiB segment) now builds
    in one pass without heap pressure. Output bytes are unchanged.
  - Lexical segments above 8 MiB open lazily (`openLexicalIndexLazy`):
    only the 64-byte header and the doclen array (4 bytes/record) become
    resident, and term lookup exploits the format's own structure — fnv
    hashes are uniform and the table is sorted by them, so it is an
    implicit interpolation index: predict the entry position from the
    hash, read one ~512-entry window, re-aim for the rare tail. A query
    term costs ~1 window read plus 1 postings read, parallel across
    terms. Verified equivalent to the eager reader entry-for-entry in
    conformance. Lazy reads carry the transitional integrity stance
    (committed via the manifest digest, not individually verified — the
    format-1 sketch-row stance); `info().lexical.lazy` reports it.
  - `retrieval: 'augmented'` joins the mode set: BM25 candidates enter
    the exact rerank but results keep pure distance order. Measured on
    the wiki pack's 200-query pre-registered eval (456k records, exact
    fp32 ground truth): vector 82.8% recall@10 (titles 80.8%), augmented
    **85.3%** (titles 83.6%) — a new best on that eval at identical
    latency — while RRF-ordered hybrid scores 45.9% on this metric, by
    definition: RRF deliberately promotes lexical matches over exact
    nearest neighbors (reciprocal ranks make lexical-rank-8 outscore
    vector-rank-9), which labeled-relevance bakeoffs reward and
    exact-NN-overlap metrics punish. On the docs-scale bakeoffs
    augmented is identical to vector (the sketch's top-C misses nothing
    at 2k chunks), confirming the decomposition: candidate augmentation
    is what pays at scale, rank fusion is what pays on
    keyword-vs-semantic relevance. `hybrid` remains the default;
    `compile-wiki.mjs` now builds the lexical segment into wiki
    artifacts. Range-storage at 456k over loopback HTTP with 25 ms
    injected delay: the 597 MiB artifact opens on 30.0 MiB in 10 range
    GETs (lazy lexical keeps its 71 MiB out of open); first-pass hybrid
    queries fetch ~680 KiB in ~436 ranges (p50 746 ms at parallelism
    64), repeat queries 62 KiB / 5.9 ranges with p50 ~320 ms dominated
    by the pure-JS sketch scan — wiring the WASM sketch scanner into the
    complete reader is the identified next lever.
- **Kind-3 artifacts open lazily: the inline encoder no longer blocks (or
  bloats) open.** The first range-storage measurements showed cold open
  transferring ~25 MiB in every kind-3 artifact regardless of corpus size
  — the eagerly-read inline-encoder region — which at docs scale meant
  open was effectively a full download. The reader now defers a
  query-interpretation segment above 4 MiB: the 16-byte header and the
  small calibration region are read at open, the encoder region
  downloads in the background from the moment open resolves
  (`prefetchEncoder: false` defers it to the first query), and the first
  query awaits it. Integrity stance is unchanged in substance: the
  whole-segment digest is verified when the bytes arrive, and the
  header/calibration slices used at open are byte-compared against the
  verified segment before any query can return — a mismatch fails every
  query. Until then `info().encoder` serves the identity-verified
  manifest declaration (marked `deferred`) and `encoderVerified` is null
  (unknown, not unverified). Kinds 1/2 and small segments keep the
  eager one-read path byte for byte; the spec's open semantics (§4)
  document the deferral. Measured over loopback HTTP with 25 ms
  injected delay: open drops from 25.15 MiB / ~860 ms to 0.63 MiB /
  ~200 ms, with the encoder's ~25 MiB paid by the first query (~900 ms)
  or overlapped with user think-time under the default prefetch.
  Retroactive: every published kind-3 artifact benefits without
  recompiling. The bench gains `--no-prefetch-encoder` and a first-query
  line so the open/first-query split is visible.
- **HTTP range-storage benchmark** (`npm run bench:range-storage`):
  serves an artifact from a loopback server implementing HEAD + byte
  ranges and opens it through `httpRangeSource()` — the same 206 path a
  CDN or object store serves — measuring what the storage model actually
  costs, separate from retrieval relevance. Reports per artifact: open
  transfer (bytes/ranges/time before the first query), first-pass
  per-query bytes, range counts, and latency percentiles, a repeat pass
  isolating cache effects, and RSS/heap deltas; each artifact runs in a
  fresh child process so memory numbers do not accumulate across corpus
  sizes, `--server-delay-ms` injects a fixed per-response delay as a
  rough RTT model, and the server cross-checks that bytes requested by
  the range source equal bytes served. `--json`/`--csv` emit
  machine-readable results. First measurements (kind-3 docs artifacts,
  20 queries, hybrid): open transfer is ~25 MiB in 9 range GETs
  regardless of corpus size — the eager inline-encoder segment dominates
  cold start; first-pass queries cost 40–127 KiB in 5–16 parallel
  ranges (p50 78–100 ms with 25 ms injected delay, so roughly 2–3
  sequential request waves); repeat queries transfer zero bytes — the
  row/record caches absorb everything, p50 ~20 ms of pure compute.
- **Ingestion/anchor conformance suite** (`test/ingestion/`, in npm
  test): fixture corpora covering nested and duplicate headings, explicit
  `{#custom-id}` anchors, headings inside fenced code, Unicode and setext
  headings, inline code/links in headings, snake_case identifiers,
  frontmatter titles/slugs, MDX components around headings, oversized
  section splitting, and HTML pages with authored `id` attributes — each
  asserting the full chain a search result depends on: document URL,
  headingPath, anchor, final href, chunk text. Anchor generation is
  parity-tested against the real github-slugger (the library Docusaurus
  renders with) via a recorded 26-heading battery plus a live comparison
  when the library is installed; two divergences are documented as
  deliberate (multi-space heading runs, JSX tags stripped by MDX).
  Building the suite surfaced and fixed four real defects: the slugger
  was ASCII-only (github-slugger keeps Unicode letters — "Über uns" must
  slug to "über-uns"), dash runs were wrongly collapsed ("C++ & C#" slugs
  to "c--c"), heading underscores were stripped (destroying snake_case
  identifiers like ACTION_QUERY_PARAMS), and setext headings
  (`Title\n===`) were not recognized at all. Compiling nodejs.org's API
  docs surfaced a fifth: minified HTML with unquoted attributes
  (`href=fs.html`, `id=fsreadfilesync…`) was invisible to the link and
  anchor extractors — the crawl found 2 pages of 30.
- **The v1 spec no longer lists hybrid search as a non-goal**:
  docs/CREATE_PANCAKE_SEARCH_SPEC.md's non-goals entry for "Hybrid BM25 +
  vector scoring" is struck through with a pointer to the shipped lexical
  segment (spec §3.8) and the `query({ retrieval })` modes. (That planning
  document was moved out of the repository on 2026-09-05.)
- **Section-aware ingestion and chunking.** Markdown/MDX documents are
  parsed into their heading structure before chunking — heading lines are
  recognized only outside code fences, explicit `{#custom-id}` heading ids
  win over derived slugs, and duplicate headings get GitHub-style `-1`
  suffixes — and HTML crawls section on `h1`–`h6` tags, preserving the
  page's own `id` anchors. Chunking keeps a section together when it fits
  (up to 1.6x the target size), subdivides at paragraph boundaries when it
  does not, and merges sub-25-token sections into their neighbor. Every
  chunk now carries the fields the corpus schema always reserved but
  nothing populated: `headingPath` (ancestor headings ending in the
  section's own) and `anchor`, so results deep-link to the exact section
  (`Routing > Dynamic routes > Rest parameters`, `#rest-parameters`)
  instead of pointing at a page. A leading h1 that restates the document
  title stays out of heading paths. Plain-text files keep the flat token
  windows.
- **Retrieval modes and a retrieval-quality bakeoff.** `query()` accepts
  `retrieval: 'hybrid' | 'vector' | 'lexical'` (hybrid remains the
  default; the single-signal modes exist for measurement and debugging —
  abstention scores identically in every mode), and
  `scripts/bakeoff-retrieval.mjs` runs a query set with expected sections
  through all three modes, reporting success@1/@3 and MRR@5 split into
  semantic questions and exact lookups. Measured on a 120-page Astro docs
  crawl (2,123 section chunks, 24 queries): exact lookups score
  success@3 0.67 vector-only vs 1.00 BM25-only vs 0.92 hybrid; semantic
  questions 0.75 / 0.67 / 0.83 — hybrid is the only mode competitive on
  both classes. A 9-doc repo-docs corpus agrees on exact lookups (hybrid
  success@1 0.71 vs 0.57 for either single mode) with semantic within
  noise at that sample size.
- **Hybrid search: complete artifacts carry a BM25 lexical index**
  (segment kind 5, layout `bm25-v1`, spec §3.8) and the reader fuses
  lexical and vector retrieval. The builder writes a static inverted
  index over the corpus records (hash-addressed fixed-width term table,
  varint postings, ~113 KiB for a 132-record docs corpus); `compile`
  includes it by default. At query time the BM25 top matches join the
  sketch's exact rerank as extra candidates — scored by true vector
  distance, which is what rescues known-item lookups the sketch scan's
  top-C misses — and the final result order fuses the distance and BM25
  rankings by reciprocal rank (RRF, constant 60, untuned). Lexical hits
  under a third of the best lexical score are dropped before fusion: idf
  collapses common query terms into a near-tied mass that carries no
  ranking information. Abstention is unaffected — it scores the
  distance-sorted top-k exactly as calibrated; fusion changes result
  order only. Readers that predate the segment kind skip it (spec §3.3)
  and serve vector-only queries from the same file; artifacts without
  the segment behave exactly as before (`info().lexical` reports which).
  `PancakeSketchArtifact.search` gains `extraCandidates` (external row
  ids joining the exact rerank) and `fullRerankOutput` (return every
  reranked candidate, not the top k) to support this. The format is
  lazy-friendly by construction (absolute offsets, hash-sorted
  fixed-width term table) so a range-reading opener for wiki-scale packs
  needs no format revision; the phase-1 reader loads the segment eagerly
  and verifies its digest at open. Covered by 9 new conformance checks
  including a deterministic rescue case (a known-item query whose
  encoder vector is pure noise still surfaces the right record) and
  lexical-segment tamper rejection.
- **The crawler follows the seed's own redirects** — HTTP and meta-refresh,
  bounded at 5 hops, each hop logged — and the final URL defines the crawl
  origin. Sites routinely send their root to a canonical host or localized
  landing page (docs.astro.build's root is an HTTP 200 whose only content
  is `<meta http-equiv="refresh" ...>`); the crawl previously discovered
  one empty page and failed with an error blaming the content. Every other
  fetch keeps the strict redirect-skip, and a contentless meta-refresh
  page found mid-crawl is now followed through the normal frontier filters
  instead of consuming page budget as an empty document. Exercised by a
  new scaffold-e2e check against a local server chaining a 301 into a
  meta refresh.
- **Grounding coverage weighs word informativeness and scores more
  passages**: corpus-common words (chunk document frequency above ~5%,
  shipped as a bloom in `asset.coverage.commonBloom`) count at 1/3 weight
  — a docs corpus's "templates"/"support" no longer ground a query on
  their own, which was letting roadmap-style questions score strong — and
  coverage is the max over the top 5 passages instead of the top 1, so a
  paraphrased query gets more chances to find the passage sharing its
  vocabulary while recombination negatives' source chunks still each
  ground only their own half. Hard-negative CV AUC across the three test
  corpora: docs repro 0.808 → 0.831, wiki 0.918 → 0.924, Astro docs
  crawl 0.877 → 0.887, with recombination-kind AUCs at 0.91–0.98.
- **The abstention model gains a grounding feature** (`self-templates-v3`):
  the fraction of the query's content words present in the top retrieved
  passage's text, fit as a fifth term in the same logistic model. Every
  prior feature measures topic similarity (three distances plus membership
  in a corpus-wide bloom), so "the corpus discusses this area" and "this
  passage answers this question" were indistinguishable by construction —
  a query can retrieve topically-adjacent content at a closer distance
  than genuine controls while none of its terms appear in the returned
  passage. The term is serialized as a supplemental `asset.coverage` field
  rather than a `features[]` entry, so readers that predate it score the
  topic-only model against the same thresholds (a conservative
  degradation) instead of feeding an unknown feature name NaN into the
  logistic — and the reader now degrades any non-finite score to
  `unscored` rather than letting NaN compare false against both
  thresholds and answer everything. Scoring an artifact that carries the
  term hydrates the top-1 record before the verdict (the page is hot for
  result hydration when answered; an abstained query costs one extra
  read); word rules ship in the asset so builder and reader cannot drift.
  Measured on the hard-negative repro corpus: hard-negative CV AUC 0.772
  → 0.808, recombination-kind AUC 0.891, all five same-field near-domain
  probes now abstain outright, and the calibration summary adds
  `cvAucHardByKind`. A/B-verified on a second, structurally opposite
  corpus (250 distinct-topic wiki articles, low vocabulary overlap):
  hard-negative CV AUC 0.853 without the term → 0.907 with it, pooled
  0.903 → 0.937, with paraphrased no-title-word controls staying strong
  under both. Known residual limits: a single-content-word query can be
  spuriously grounded by a topically adjacent passage containing that
  word, and attribute questions about present entities (entity words
  grounded, attribute absent) still score strong under both models.
- **compile's self-calibration now trains against hard negatives**
  (`self-templates-v2`), fixing artifacts that confidently answered
  unanswerable in-domain questions. The v1 calibrator's negatives were all
  easy — off-domain bank and gibberish, low known-token fraction, distant
  retrieval — so the fitted model handed the vocabulary feature a dominant
  weight (+4.1 standardized) and scored any in-domain query answerable:
  measured on a docs corpus, 6 of 8 unanswerable in-domain probes returned
  `strong` at p 0.75–0.94 while the pooled CV AUC read 0.998. v2 adds two
  hard-negative classes with high known-token fractions: held-out
  documents (excluded from calibration searches via `searchFiltered`, then
  asked about — the wiki calibrator's held-out-shard trick made
  corpus-generic) and cross-chunk recombinations (corpus words no single
  document lexically supports). Hard-negative labels are verified by
  retrieval (a held-out topic covered by a near-duplicate document is
  dropped, not mislearned); overlapping off-domain queries are kept as
  eval-only rows and reported instead of silently discarded. The gate now
  also requires hard-negative CV AUC ≥ 0.75 — the pooled number is
  dominated by the easy classes and cannot see this failure — and the
  `weak` threshold rises to the hard-negative ceiling, so overlapped
  queries surface with a caveat instead of full confidence. Same probes
  after the change: 0 of 8 unanswerable queries score `strong` (6 weak,
  2 none), answerable controls unchanged at `strong`. The calibration
  summary adds per-class counts and `cvAucEasy`/`cvAucHard`.
- **`create-pancake-search compile`** builds a complete kind-3 `.pancake`
  artifact from `--source <path|url>` and stops — no project, no Worker,
  no Cloudflare. It chunks and embeds the corpus with the packaged inline
  MiniLM encoder (auto-fetched, digest-pinned, when not present), measures
  the rerank operating point, and writes one file (`--out`, default
  `search.pancake`) openable with `pikelet-wasm/complete` on any runtime.
  Takes `--name`, `--max-pages`, `--include`/`--exclude`, `--force`.
  Exercised by the scaffold e2e (compile → open → query, plus the
  `--runtime complete` rejection).
- **compile self-calibrates abstention** (`src/calibrate.mjs`), so compiled
  artifacts answer `matchQuality` strong/weak/none out of the box instead
  of "unscored". Positives are templated title questions plus content-word
  queries, each verified by retrieval before it counts; negatives are a
  built-in off-domain bank (overlap-dropped against the positive median
  d0) and bloom-checked gibberish; the fit is the wiki calibrator's
  logistic model over (d0, margin, mean10, known_frac). Thresholds are
  placed on percentiles (5th of positives, 95th of negatives) rather than
  the wiki calibrator's raw extremes — auto-generated queries put junk in
  both tails and the saturated fit collapses geometric means of them —
  with hard a quarter of the way up the gap: a false abstain hides
  results, a false weak shows them with a caveat, so the weak verdict
  owns the uncertain band. Skips with a logged reason (and ships unscored)
  when positives or negatives run short or the cross-validated AUC misses
  the 0.85 gate — the gate uses a deterministic 5-fold cross-validation
  over the generated queries, since the fit AUC is in-sample; the embedded
  asset records method, query counts, and both numbers (`fitAuc`,
  `cvAuc`). New
  flags: `--calibration <file>` embeds a prebuilt retrieval-signals-v1
  asset, `--skip-calibration` opts out. The Docusaurus plugin default is
  unchanged (auto-calibration is opt-in via `runtime.calibration: 'auto'`
  outside compile).
- **Inline-transformer passage embedding runs on a worker pool** past 32
  chunks (one WASM kernel + weight blob per worker, chunks dealt by index
  so builds stay deterministic — pool and sequential builds return
  bit-identical distances). 571 chunks: 13 min → 3 min on 8 cores.
  `PANCAKE_SEARCH_EMBED_WORKERS` overrides the worker count (0/1 forces
  sequential); environments without worker support fall back to the
  sequential path with a logged reason.

## pikelet-wasm 0.5.0 / create-pancake-search 0.5.0 — 2026-08-26

### Breaking / compatibility

- **Sketch artifact format 2 (per-row integrity) is now what the builders
  write by default.** The vectors region is laid out as interleaved blocks
  — a digest page of truncated per-row SHA-256s (16 rows × 16 bytes by
  default) ahead of each block's rows — anchored by a 32-byte-per-block
  page-hash table at the end of the resident prefix, so `residentSha256`
  (and, in a complete artifact, `index.headerSha256` and the identity)
  covers it. `fetchRows` verifies every page against the table and every
  row against its digest slot **on the read that fetches it**, in both the
  full and staged micro tiers; tampered rows fail the query instead of
  silently altering results. Geometry per the 2026-08-25 measurement:
  ~0% query-latency overhead, +4.2% file size. Readers ≤ 0.4.0 reject
  format-2 sketches ("Unsupported sketch artifact version"); the new
  reader opens format 1 and 2 (`verify: false` opts out, reported).
  `exportSketchArtifact(..., { rowIntegrity: false })` still emits
  format 1; `rowsPerBlock` / `rowDigestBytes` tune the geometry.
- Complete artifacts embedding a format-2 sketch report
  `info().indexRowIntegrity: 'per-row-sha256'`; with that, every byte that
  influences a query is authenticated per read and the spec's transitional
  integrity stance applies only to format-1-sketch artifacts.
- New `test/sketch_row_integrity.js` in `npm test`.
- `create-pancake-search` has no behavior changes of its own; generated
  projects now depend on `pikelet-wasm@^0.5.0` so the artifacts the
  scaffolder builds (format-2 sketches included) open on the installed
  reader.

## pikelet-wasm 0.4.0 / create-pancake-search 0.4.0 — 2026-08-25

### Breaking / compatibility

- **Complete artifact format 2** (`pancake-complete-v2`, header
  `formatVersion` 2) is now what the builder writes by default
  (`buildCorpusSegment`, `PROFILE_V2`): the corpus segment carries one
  SHA-256 per record behind a page table the manifest commits to, so every
  hydrated record is verified on its own range read. Readers ≤ 0.3.0 reject
  format-2 files explicitly ("unsupported .pancake format version 2"); the
  0.4.0 reader opens format 1 and 2 and reports the stance in
  `info().corpusIntegrity`. `create-pancake-search` complete-mode builds
  (`search.pancake`) and `examples/05` compiles now produce format 2;
  released wiki artifacts stay format 1 and remain readable.
- `openPancakeFile` on kind-2 artifacts now runs `options.encodeQuery`
  against the declaration's verification vectors **at open** and refuses
  the open on disagreement (dimension or tolerance); declarations without
  vectors are refused unless `allowUnverifiedEncoder: true`. Previously the
  host encoder was trusted.
- Reader hardening: every artifact-derived read is validated (safe
  integers, overflow, `fileBytes`, source size, per-open `maxReadBytes`
  budget — default 256 MiB, `Infinity` → 2 GiB backstop — and
  `maxRecordBytes` per record, default 16 MiB) and must return exactly the
  bytes requested; u64 fields above 2^53−1, unknown query-interp versions,
  duplicate segment kinds, and inconsistent offsets tables are rejected.
  Unknown segment kinds are skipped (spec 3.3). Files that previously
  opened only because a check was missing may now be refused with a
  specific error.

### Added

- `info().formatVersion`, `profile`, `encoderVerified`, `corpusIntegrity`;
  `record(id)`; `verifyHostEncoder()`, `SUPPORTED_PROFILES`,
  `CORPUS_LAYOUT_V2` exports; `assemblePancakeFile` accepts unknown kinds
  via `kindNumber` and returns `formatVersion`.
- `test/complete_profile.mjs` — deterministic complete-profile reader
  conformance (77 checks: hostile input, verification, integrity, format-1
  compatibility, kind-1 goldens from the committed 03 assets), run by
  `npm test` and therefore CI.
- spec/COMPLETE_PROFILE.md Draft 2.
- `create-pancake-search --help` / `-h` print usage (previously "Missing
  value for --help").

### Fixed (artifact layer hardening, from the 2026-08-21 external review)

- `PancakeRangeArtifact.openFile` / `PancakeSketchArtifact.openFile` close
  their file descriptor when the open is rejected (previously every corrupt
  artifact leaked one fd); `NodeFileRangeSource` releases the fd if `fstat`
  fails after `open`, and refuses reads after `close()`.
- Range and sketch `search()` apply the core engine's query contract:
  plain-array coordinates must be numbers (no `'1'` coercion), every
  component must be finite for every metric (L2 no longer returns
  `distance: NaN`), dimension checked; `k`, `rerank`, `efSearch` must be
  positive integers and `k` / the sketch candidate pool are capped to the
  row count before sizing any allocation (`search(q, 1e9)` over 300 rows no
  longer attempts a 1e9-slot allocation).
- `parseUint8Snapshot` validates `M`/`M0`, bounds each level's adjacency
  count by `M0`/`M`, checks the bytes it claims are present, and range-checks
  neighbor ids — all before allocating that level's edge array.
- `httpRangeSource`: after the one-time full-download fallback, reads are
  served from memory with no further requests (previously every read still
  issued a fetch).
- Complete reader: `id` and `distance` are reserved result names and always
  come from the search (a record's own fields no longer overwrite them);
  `close()` is idempotent (no `EBADF` on a second call) and queries after
  close are refused; `query({ k })` rejects 0 / negative / fractional `k`
  instead of silently defaulting to 5.
- `create-pancake-search` loaders prefer the in-repo `pikelet-wasm` when
  running inside the monorepo (a stale installed copy higher in the tree
  no longer shadows the checkout); npm consumers are unaffected.
- Packaging: the native benchmark baselines (`faiss-node`, `hnswlib-node`,
  `usearch`) and the `@emnapi/*` pins are no longer `optionalDependencies`
  of `pikelet-wasm` — consumers no longer attempt those native builds; they
  live in `benchmarks/package.json` (`cd benchmarks && npm install`). The
  unused self devDependency on `pikelet-wasm` is removed.
- Sketch `search()` with a candidate pool equal to the row count (small
  artifact, or `k`/`rerank` ≥ count) no longer runs the resident selection
  scan — every row is a candidate for the exact rerank — instead of the
  2·count² slot operations the selection loop degenerated to.
- `parseUint8Snapshot` reads version-1 uint8 payloads with their 4-byte
  edges (id only), matching the engine's deserializer; v1 payloads were
  previously misparsed as the v2 {id, distance} layout.
- `rerank: 0` / `efSearch: 0` keep their historical "use the default"
  meaning (`undefined`, `null`, `0` → default; anything else must be a
  positive integer).
- `parseUint8Snapshot` bounds `M`/`M0` to the engine's own envelope
  (`M ∈ [2,128]`, `M ≤ M0 ≤ 256`).
- Complete reader: a kind-3 encoder's WASM buffers are released when the
  open fails after the encoder was created; a digest-page read that fails
  for a transport reason is not cached as a rejection (the next hydration
  retries). `httpRangeSource` refuses a 206 whose `Content-Range` is not the
  requested slice.
- **Format 2 manifests now commit to the embedded sketch's 256-byte header
  (`index.headerSha256`, required)**, and the reader verifies it at open:
  the sketch's metric/dim/count and its `residentSha256`/`vectorsSha256`
  are anchored to the artifact identity instead of living only in the
  segment's own (attacker-rewritable) header. Previously a flipped metric
  word changed search semantics under an unchanged identity. On every
  format the reader also cross-checks the sketch metric against the
  manifest. Format-2 identities change (format 2 was never released).
- `httpRangeSource`: a 206 without a valid `Content-Range` naming the
  requested slice is refused (RFC 9110 requires the header) and the body
  length must match it; the full-download fallback cap is enforced on
  bytes actually received (a chunked 200 aborts as soon as it streams past
  the cap) instead of trusting `Content-Length`/HEAD; new `cacheKeyParam`
  option (`null` for signed-URL hosts keeps the URL untouched, default
  `'r'` keeps the Chromium cache-lock workaround).
- Stable external ids are bounded to the uint32 the v3 snapshot stores:
  `add()` refuses (INDEX_LIMIT) before the engine mutates and `addBatch()`
  preflights the whole batch, instead of assigning ids the snapshot would
  truncate into an unrestorable export.
- `canonicalJson()` follows JSON semantics for unsupported values
  (undefined/function object properties omitted, array entries become
  null) instead of emitting invalid JSON.
- `evaluation()` is typed `Promise<Record<string, unknown> | null>` (it
  returns null with no evaluation segment) and refuses a non-object
  segment at runtime.
- Encoder verification validates the *expected* embeddings too: a
  verification vector whose expectation is malformed (strings, null, NaN,
  wrong length, missing) now fails verification instead of passing
  vacuously — `x - "x"` is NaN and `NaN > maxDiff` is false, so such
  vectors previously compared nothing while still reporting
  `encoderVerified: true`. Applies to kind-2 host verification and kind-3
  inline verification (shared `validateExpectedEmbedding`, exported).
- Lazy rerank rows: new `verifyIndexVectors: true` open option and
  `verifyVectors()` method on the complete reader run the full pass
  against the identity-anchored `vectorsSha256`; `info().vectorsVerified`
  reports the state. Documented prominently (README, spec section 6):
  without the pass, rows that feed reranking are committed but not
  verified per read; per-row commitments are spec section 8 question 6.
- Builder inputs may be plain `Uint8Array`s, as the typings always said:
  `buildCorpusSegment(FromBuffers)`, `buildQueryInterpSegment`, and
  `buildInlineTransformerEncoderSegment` no longer call Buffer-only
  `.copy()` on caller-supplied bytes.
- New `test/artifact_hardening.js` (66 checks) in `npm test`; 41 more checks
  in `test/complete_profile.mjs`.

### Internal

- `pancake-artifact.js` is now a thin entry over `pancake-artifact-common.js`,
  `pancake-artifact-range.js`, `pancake-artifact-sketch.js` (same exports,
  same typings; the three files ship in the tarball).
- `create-pancake-search/src/cli.mjs` is split into `common`, `ingest`,
  `embed`, `complete-build`, `scaffold`, `build` modules; `cli.mjs` keeps
  the commands and re-exports `buildSearchAssets`, `fetchInlineEncoderWeights`,
  `CliError` for the Docusaurus plugin. See the README's "Package layout".

## pikelet-wasm 0.3.0 — 2026-08-20

### Breaking / compatibility

- `buildRangeArtifact` now emits `.pancake-range` **format v3** (whole-segment
  SHA-256 digests for the id map, router, and base segments stamped in the
  header). Readers in 0.2.1 and earlier only accept v1/v2 and reject v3 files
  with `SNAPSHOT_INVALID` ("Unsupported Pancake range artifact version").
  The 0.3.0 reader still opens v1/v2 files (structurally validated,
  unverified — they carry no digests).
- Snapshots whose payload declares an unknown future format version are
  rejected at import instead of being parsed as the newest known layout
  (JS and engine). Every version a released writer has emitted still
  imports.
- `Pancake.create()` rejects configurations whose eager arena estimate
  exceeds the 1.5 GiB wasm32 budget with `WASM_ALLOCATION_FAILED` (estimate,
  budget, and backend formula in `details`) instead of aborting the WASM
  instance from inside the engine.
- Types: `SketchArtifactBuildManifest.file` is now optional (absent on the
  bytes-in/bytes-out builder).

### Added

- **`pikelet-wasm/complete`** — the one-file `.pancake` reader
  (`openPancakeFile`, `httpRangeSource`, the inline-transformer host and its
  verification-vector API), and **`pikelet-wasm/complete/builder`** — Node
  assembly (`assemblePancakeFile`, segment builders,
  `measureRecommendedRerank`, `loadInlineEncoderKernel`). Typed via
  `complete/index.d.ts` / `complete/builder.d.ts`. The PSF1 wire format has one
  definition (`complete/format.mjs`) shared by reader and builder; the encoder
  kernels (`complete/encoder-kernels/`) ship here.
- Kind-3 (inline transformer) declarations are validated against the
  compiled kernel before the first forward (layout, dim, exact blob length),
  and their verification vectors (contract §4.4 mode 1, including a
  >128-token windowed-pool probe) run automatically on open; `verify: false`
  opts out.
- Range artifacts: `integrity` digests, `segmentVerified`, `verify` open
  option, `verifyBaseSegment()`; sketch artifacts: `vectorsVerified`,
  `verifyVectors()`, `clearCache()`, `buildSketchArtifactBytes()` (no
  filesystem).
- Layered read budgets: `maxReadBytes` (open-path reads, default 256 MiB,
  strict) and `maxRangeBytes` (coalesced query-path reads split at 16 MiB
  record/row-aligned boundaries; also accepted by `prefetch()`). The 2 GiB
  backstop remains. Segment verification streams through Node's incremental
  SHA-256 in bounded chunks.

### Changed / fixed

- Complete-artifact `open()` fetches its post-segment-table extents as one
  wave: 8 dependent range rounds down to 5.
- `manifest.corpus.records` is validated before it sizes a read; the
  full-download fallback refuses when the content length is unknown
  (previously an unbounded read on Range-ignoring hosts).
- AVX-512 native builds no longer drop `uint8_dot` to scalar (the kernel's
  AVX2 branch is now taken on AVX-512 targets); the native addon and an ISA
  compile sweep now run in CI.
- Both engine variants rebuilt (`dist/`).

## create-pancake-search 0.3.1 — 2026-08-21

### Fixed

- Generated projects depended on `pikelet-wasm: ^0.2.0`, which `npm install`
  resolves to 0.2.1 — a reader that rejects the format-v3 `.pancake-range`
  this version's scaffolder builds. Every `--runtime artifact` project
  generated by 0.3.0 therefore answered `/search` with `SNAPSHOT_INVALID`
  ("Unsupported Pancake range artifact version") until its dependency was
  bumped by hand. The generated `package.json` now inherits this package's
  own `pikelet-wasm` range (`^0.3.0`) instead of carrying a second hardcoded
  one. Projects already generated by 0.3.0: change the dependency to
  `^0.3.0` and `npm install`.

## create-pancake-search 0.3.0 — 2026-08-20

Requires `pikelet-wasm >= 0.3.0` (uses `pikelet-wasm/complete`).

### Breaking / compatibility

- The 24.3 MiB MiniLM encoder weights are no longer in the tarball
  (package size ~170 kB, was ~25.7 MB). The CLI and the Docusaurus plugin
  download them once on first kind-3 use from the `inline-encoder-v1`
  release asset, verify the pinned SHA-256, and write them to the configured
  `weightsPath`; `PANCAKE_ENCODER_WEIGHTS_URL` points at a mirror.
  Custom-named weights are never fetched.
- Kind-3 declarations now state `maxTokens: 128` (the embedded kernel's
  real limit; earlier builds declared the teacher's 256) and
  `longInputs: 'windowed-mean-pool'`, and carry verification vectors.
  Artifacts built by 0.2.0 remain readable; rebuild to get the corrected
  declaration.

### Added

- `create-pancake-search doctor <url>` — probes an artifact hosting URL
  (Range/206, Range with a cache-key query string, negotiated protocol
  h1/h2/h3, ETag, median RTT, artifact magic/identity) and exits 1 on any
  failing check.
- Complete profile (kind 3) output: `runtime.mode: "complete"` /
  `profile: "kind3"` with `embedding.mode: "inline-transformer"` emits a
  single `search.pancake`. Docusaurus plugin option `completeProfile`
  (`enabled`, optional `vectors`, `vocab`, `weights`, `calibration`,
  `model`, `maxTokens`) writes `build/pancake-search/search.pancake` and
  serves it with a bundled kind-3 browser reader; without precomputed
  vectors the build embeds chunks through the packaged encoder.
- Docusaurus plugin `sourcePath` / `sourceRouteBase`: index the site's
  markdown/MDX sources instead of rendered HTML, with front-matter `slug`
  honored and `NN-` sidebar prefixes stripped in result routes.
- Generated Worker README and config schema updated for the new modes.

### Fixed

- `pikelet-wasm/artifact` and `pikelet-wasm/complete` resolve from the site,
  the plugin, or the monorepo sibling, with an actionable error otherwise
  (npm consumers of the Docusaurus plugin previously failed at bundle time).

## Earlier releases (summary)

- **pikelet-wasm 0.2.1 / create-pancake-search 0.2.0 — 2026-08-12.**
  pancake-wasm: hostile-input hardening on deserialize, `packVersion`
  stamping. create-pancake-search: first publish of the Docusaurus plugin and
  `--mode student` (corpus-specific PSTU query encoder bundled in the
  Worker, no Workers AI binding).
- **pikelet-wasm 0.2.0 / create-pancake-search 0.1.0 — 2026-07-30.**
  Pancake 0.2 contract: `restore`, `inspectSnapshot`, per-query `efSearch`,
  `PancakeError` / `PANCAKE_ERROR_CODES`; mutable-ef and matrix helpers
  removed; `pikelet-wasm/artifact` search-artifact readers/builders.
  create-pancake-search: initial scaffolder (snapshot and artifact
  runtimes).
- **pikelet-wasm 0.1.0 — 2026-06-27.** Initial release: HNSW engine in WASM
  (float and uint8 backends) for Node, browsers, and Cloudflare Workers.
