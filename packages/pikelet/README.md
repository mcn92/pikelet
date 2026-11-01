# pikelet

The Pikelet CLI: compile a corpus into one `.pikelet` knowledge pack,
serve packs to LLMs over MCP, scaffold a search app, or certify hosting.
(Formerly `pancake`; artifacts and profile strings from before the rename
remain readable — see docs/history.md.)

Turn a documentation site into search: either one complete `.pikelet` file
you can query from any JavaScript runtime, or a deployable Worker + UI app.

The 15-second version — compile a folder or a live site into one file:

```bash
npx pikelet compile --source ./docs --out search.pikelet
# or point it at a site:
npx pikelet compile --source https://docs.helix-editor.com --out search.pikelet
```

```js
import { openPikeletFile } from 'pikelet-wasm/complete';
const search = await openPikeletFile('search.pikelet');
const out = await search.query('how do I remap keys', { k: 5 });
// out.results, out.matchQuality ('strong' | 'weak' | 'none'), out.confidence
```

The file carries the corpus records, sketch index, inline MiniLM query
encoder, calibrated abstention, and evaluation data — no service, no model
host, no Cloudflare. Off-domain queries return `matchQuality: "none"` with
zero results instead of confidently wrong ones. See
[Compiling a complete `.pikelet` artifact](#compiling-a-complete-pikelet-artifact)
for the details.

## Installing

`npx pikelet` needs no install. To install it, use `npm install -g pikelet`:
the CLI and `pikelet-wasm` are its only two packages, with no native code and
no transitive dependencies. The CLI requires Node 22.23.4+, 24.22.1+ or 26.12.1+: earlier releases ship a
fetch that decompresses any response a server labels with a
`Content-Encoding`, and 18.x, 20.x, 23.x and 25.x are end-of-life with no
fix. The CLI warns on start under any other version. `npx pikelet --help` lists every command and flag; a
flag the CLI does not know is refused (`Unknown flag --x`) rather than
silently ignored.

## Scaffolding a search app

When you want a deployed app instead of a file, the scaffold path ingests
your docs, builds the search assets offline, and ships them with a
Worker/UI shell serving retrieval at the edge:

```bash
npx pikelet create --name my-docs-search --source ./docs --no-deploy --yes
cd my-docs-search
npm run dev
```

An existing directory is refused unless you pass `--force`, which replaces
it only once the new project has built: a failed build leaves the old one
as it was. `--force` will not replace the current directory, a directory
containing it, or one inside or containing the `--source`.

The generated project contains a bundled Pikelet snapshot, corpus metadata, a
Workers AI search worker, and a static UI. A second runtime,
`--runtime artifact`, serves the deprecated `.pikelet-range` profile; it
still works, but new projects should use the default snapshot runtime, or
`compile` (below) when the deliverable is a complete `.pikelet` file.

In both runtimes, the story is the same: the expensive work happens at build
time; query-time code embeds the query, searches Pikelet, and hydrates result
metadata. Passages are embedded at build time by the same WASM kernel that
runs inline encoders, using a u8 export of `BAAI/bge-small-en-v1.5`
(downloaded once, pinned by sha256, cached in `~/.cache/pikelet/models`;
`PIKELET_MODEL_CACHE` moves it). Its vectors are within 0.9999 cosine of the
fp32 model's, the same model Workers AI serves. Query embedding comes from Workers AI by default, or from a bundled
corpus-distilled encoder with `--mode student` (see below), which removes the
Cloudflare AI dependency entirely. For workers-ai projects, `LOCAL_STUB_AI=1`
can exercise the endpoint mechanics locally without Workers AI.

In the deprecated artifact runtime, a prebuilt `.pikelet-range` file can be
supplied with `--artifact`; external artifacts must have dimension, count,
and IDs that match the generated corpus.

Compiled `.pikelet` artifacts also carry a BM25 lexical index (a few
hundred KiB at docs scale), and queries run hybrid: the lexical matches
join the vector rerank as candidates — so exact identifiers, config
options, and title lookups land even when embedding similarity alone would
miss them — and the final ordering fuses the two rankings by reciprocal
rank. Readers that predate the segment serve vector-only from the same
file. `query()` also takes `retrieval: 'vector' | 'lexical'` for
measurement (`scripts/bakeoff-retrieval.mjs` in the main repo compares
all three modes over a labeled query set).

Ingestion is section-aware: Markdown/MDX parses into its heading
structure (code fences respected, `{#custom-id}` heading ids honored) and
HTML crawls section on `h1`–`h6` with the page's own `id` anchors, so a
section stays one chunk when it fits and every result carries
`headingPath` and `anchor` — deep links to the exact section, not the
page.

## Compiling a complete `.pikelet` artifact

When you want the search *file* rather than a search *app*, `compile` builds
a complete kind-3 artifact and stops — no project, no Worker, no Cloudflare:

```bash
npx pikelet compile --source ./docs --out search.pikelet
```

The output is one self-contained file carrying the corpus records, sketch
index, inline MiniLM query encoder, and evaluation data. Open it from any
runtime with the complete reader:

```js
import { openPikeletFile } from 'pikelet-wasm/complete';
const search = await openPikeletFile('search.pikelet');
const out = await search.query('how do I configure auth', { k: 5 });
```

Passage embedding runs locally through the same inline encoder the artifact
carries (the ~24 MiB weight blob is fetched once, digest-pinned, when the
package copy is absent — registry installs ship without it), on a worker
pool sized to your cores — roughly 3 minutes for a ~570-chunk docs site on
8 cores. Each worker holds its own kernel and weight copy, so the pool
trades a few hundred MB of build-time memory for the near-linear speedup;
`PIKELET_EMBED_WORKERS` overrides the pool size (0 forces
sequential). `compile` accepts `--source` (folder or URL), `--out`,
`--name` (corpus name recorded in the artifact), and `--force` to
overwrite the output file. Folder sources take `--include`/`--exclude`
filesystem globs; URL sources take `--max-pages` and
`--include-url`/`--exclude-url` (URL-path patterns with `*` as the
wildcard — mixing the two families is an error, not a silent no-op), and
aggregate pages like mdBook's `print.html` are excluded by default so a
book's content is not crawled twice. The scaffold-only flags (`--mode`,
`--runtime`, `--artifact`, deploy and student options) are rejected:
compile always builds the complete kind-3 profile.

`--sentence-windows 2` also indexes every pair of consecutive sentences in
each chunk, as extra vectors that point back to that chunk. A chunk's single
vector averages everything it says, so a question answered by one sentence
can lose to a chunk that is merely about the same thing; a window keeps that
sentence close to the question. Results, citations and the corpus are still
whole chunks — only the index has more rows (spec `COMPLETE_PROFILE.md`
3.9). Measured on a 3,742-chunk docs corpus (a public project's docs, 51 judged
questions): vector-only nDCG@10 rose from 0.551 to 0.595 and hybrid from
0.592 to 0.600, at about 9.4 index rows per chunk — the file grew from 31.9
to 47.2 MiB, an HTTP mount from 1.3 to 4.5 MiB, and a query from about
1.1 to 1.4 MiB, 49 to 111 ms (32 to 80 ms on a local file). Build time
grows with the row count. A windowed pack cannot be extended with
`append` yet.

A build is deterministic apart from the time it records, and that time
is part of the pack's identity, so two builds of the same inputs differ
only by it. Set `SOURCE_DATE_EPOCH` (whole seconds since 1970, the
[reproducible-builds](https://reproducible-builds.org/docs/source-date-epoch/)
convention) to pin it: the same inputs, encoder and CLI version then give
the same bytes and the same identity.

Match quality is calibrated from the corpus at build time, so queries
the artifact does not cover return `matchQuality: "none"` (or `"weak"`,
shown with a caveat) instead of confidently wrong results. The calibrator
generates on-topic queries from chunk titles and content words (each
verified by retrieval before it counts) and fits the same
retrieval-signals model the wiki pack ships against off-topic ones: a
built-in off-domain bank, minus any entry the corpus turns out to cover,
and out-of-vocabulary gibberish. Alongside the distance signals the model
fits a grounding feature: the fraction of the query's content words that
appear in the top retrieved passages' text. The verdict measures
retrieval relevance — whether the pack covers the question's topic — not
whether a passage states the specific fact asked for; a query returns
the per-word detail (`grounding.covered` / `grounding.uncovered`) for
that check. The calibrator also generates in-domain questions the corpus
does not answer (an ablated source passage, a swapped entity, a held-out
document) and reports how well the verdict separates them as a
diagnostic (`cvAucHard`); they do not shape the fit. The asset records
its method, target, per-class query counts, in-sample fit AUC, and
cross-validated AUCs for inspection. The acceptance gates use the
cross-validated numbers (a deterministic 5-fold split): an AUC against
off-topic queries under 0.85, too few verified positives or off-topic
negatives, or a distance weight with the wrong sign all log why and ship
unscored rather than miscalibrated. `--calibration <file>` embeds a prebuilt
retrieval-signals-v1 asset instead; `--skip-calibration` ships unscored
deliberately.

## Attaching packs to an LLM (MCP)

A compiled `.pikelet` is a portable knowledge pack: everything needed to
query the knowledge it represents — corpus, semantic and lexical indexes,
query encoder, calibrated abstention, integrity commitments — in one
immutable file. `mcp` serves packs over the Model Context Protocol on
stdio, so any MCP client (Claude Code, Claude Desktop, agent frameworks)
can attach them as a retrieval tool with no vector database, embedding
service, or retrieval backend:

```bash
npx pikelet compile --source https://docs.astro.build --out astro.pikelet
npx pikelet mcp --pack astro.pikelet --pack team-handbook.pikelet
```

`--pack` also takes a URL — packs are range-read off dumb HTTP, never
downloaded whole. One line attaches 456k passages of Simple English
Wikipedia; mounting transfers a ~52 MiB resident slice and each question
costs ~127 range requests against the 649 MiB file:

```bash
npx pikelet mcp --pack https://github.com/mcn92/pikelet/releases/download/artifact-wiki-inline-v4/pancake-wiki-inline.pancake
```

Most of a URL mount is two regions the pack commits to by sha256: the
resident index prefix and the query encoder (about 71 of 77 MiB on that
Wikipedia pack). `mcp` caches them on disk under their digests, in
`~/.cache/pikelet/blobs` (`$PIKELET_CACHE_DIR` or `$XDG_CACHE_HOME` moves
it; 2 GiB, least recently used evicted first), so the next process to
mount the same pack reads them locally: measured against the R2-hosted
copy, a fresh `mcp` process went from 80.1 MiB and 11.5 s to its first
answer down to 9.1 MiB and 6.9 s. Queries add to it: the rerank rows,
records, and digest pages a question reads are stored the same way (in
append-only logs capped at 64 MiB, oldest dropped first), so asking a
question an earlier process already asked transferred 178 KiB instead of
1.3 MiB and answered in 0.65 s instead of 1.45 s. The keyword index's lazy
reads are never cached: nothing verifies them per read. Every hit is hashed
against its digest before use, so a corrupt entry is just a refetch.
Because the keys are content digests, an entry never goes stale.
`--no-cache` turns it off; local files never use it. Library callers pass `diskBlobCache()` (from
`pikelet-wasm/complete` under Node) as `openPikeletFile`'s `blobCache`.

Either form takes `#<sha256>` to pin the pack's manifest identity — a
mount that finds different bytes at that location refuses to serve. And
`--shelf <file-or-url>` mounts every pack on a static `packs.json`
listing (see `packs/README.md` in the main repo): a registry that is also
just a file. Redirecting hosts are handled the cheap way: the reader
resolves the redirect once and pins the signed target, so GitHub's
rate-limited front door is charged per mount rather than per range read
(and expired signed URLs re-resolve automatically — but if the file behind
them was replaced in the meantime, reads fail with `ARTIFACT_CHANGED`
rather than mixing bytes from two artifacts); transient CDN pressure
(429/502/503/504) is retried with backoff.

Each pack mounts on its own. One that cannot mount — an unreachable host, a
refused pin, an unsupported pack — does not take the others down: they
serve, `list_packs` reports the missing one under `unavailable` with the
reason, and the server exits non-zero when it stops.

Instead of running the server by hand, write your MCP client's config:

```bash
npx pikelet mcp install --pack astro.pikelet --client claude-code
```

(`claude-code` writes `./.mcp.json`, `claude-desktop` the platform's
Claude Desktop config; `--server-name` names the entry, `--force`
replaces an existing one.) The equivalent JSON, if you'd rather write it
yourself:

```json
{
  "mcpServers": {
    "knowledge-packs": {
      "command": "npx",
      "args": ["-y", "pikelet", "mcp", "--pack", "astro.pikelet", "--pack", "team-handbook.pikelet"]
    }
  }
}
```

The model gets four tools (six with a journal, below). `search` queries
one pack or all of them; results stay grouped per pack (distances are
only comparable within one pack's encoder and corpus) and every result carries its provenance —
pack name, the pack's immutable manifest identity (sha256), title,
heading path, anchor, and source — so an answer can cite the exact
knowledge state it was derived from, and a pinned identity means the
citation survives pack rebuilds detectably. Calibrated match quality
crosses the protocol intact: a pack that does not cover the question's
topic says `matchQuality: "none"`, and the tool result tells the model
the support is at best indirect; on any verdict, an asked-for value in
`grounding.uncovered` is one no passage states. The results still ship
under `none` by default, because the calibrator can misjudge a
paraphrase; pass
`showAbstained: false` to withhold them. (The library's `query()` does the
reverse: it withholds under `none` unless asked.) `list_packs` reports names, identities, record counts, licenses,
and each pack's sample queries; `get_record` hydrates one full chunk
(integrity-verified from the pack) by the id a search result reported.
`verify_pack` runs the tests the pack carries inside itself — golden
queries, each verified at build time to retrieve its source (compile
embeds a sample of the calibration's retrieval-verified positives), and
abstention probes the pack must answer or refuse — so an agent can audit
a newly attached knowledge source with tests stored in the file it is
auditing.

Because packs are just files, they share like files: copy one to a
laptop, publish it on GitHub Releases or object storage, version it,
pin it by identity. The recipient attaches the finished object — not
30,000 documents and instructions for rebuilding your retrieval stack.
Packs compiled with the inline encoder (the `compile` default) are fully
self-contained; kind-2 packs, which need a host encoder, are refused at
mount with an explanation. Set `compile --license <SPDX-id>` on anything
meant for redistribution — it is recorded in the pack manifest, surfaced
by `list_packs`, and result provenance carries attribution through to
answers.

### A journal the agent can write (experimental)

Every pack above is read-only to the model. A **journal** is the one
exception: a private pack an agent appends notes to while it works —
handoffs, decisions, dead ends, measurements — so that a different agent,
from any vendor, can later mount it and pick up where the work stands.

```bash
npx pikelet journal init myproject          # ~/.pikelet/journals/myproject/
npx pikelet mcp install --client claude-code --pack ./docs.pikelet --journal myproject
npx pikelet journal list myproject          # read what the agents wrote
```

With `--journal`, the server adds two tools. `write_note(kind, title,
text, tags?, repo?, sources?)` appends one note and returns its citation;
the server, not the model, records the time, the client and the session.
`latest(kind?, tag?)` lists notes newest first, for orienting at the start
of a session. Journal notes also come back from `search` and `get_record`,
always marked `agentWritten` and unverified.

Several sessions can share one journal. Writes queue on a lock file (a
writer waits up to 30 s for another to finish), and a running server picks
up notes and compactions from other sessions on its next tool call, without
a restart. Cite a note as `(pack, id)`: ids survive compaction. Who wrote a
note is the MCP client's own name from the protocol, never something the
model claims.

A journal lives outside your project by default and is never published
unless you move it. It is unscored by design (no calibration), each note is
an immutable layer, and it folds itself back into one file every 8 notes.
See [`docs/agent-journal.md`](https://github.com/mcn92/pikelet/blob/main/docs/agent-journal.md) for the design and
what is still to come.

## Updating a pack: `append`, `compact`, `rebase`

A pack does not have to be recompiled when a few documents change.
`append` writes a small layer on top of it, and the base plus its layers
(a chain) is searched as one pack:

```bash
npx pikelet append --parent docs.pikelet --source ./changed-docs --out docs.0001.pikelet
npx pikelet append --parent docs.0001.pikelet --remove 412 --out docs.0002.pikelet
npx pikelet append --parent docs.0002.pikelet --source ./auth-v2 --supersede 87=./auth-v2/auth.md --out docs.0003.pikelet
npx pikelet mcp --pack docs.0003.pikelet
```

A layer adds records (`--source`, ingested and chunked exactly as the base
was), deletes them (`--remove <id>`, one id per flag), or replaces one
(`--supersede <oldId>=<path>` retires `oldId` in favour of the record
ingested from that file, which must come from a `--source` of the same
command and produce exactly one record). Ids are the ones search results and
`get_record` report. `--parent` names the newest member; a layer records
its parent's identity and, when the parent is a local file in the same
directory or below, its relative path (`--parent-locator` sets it
explicitly), so opening or mounting the newest layer walks down to the base,
locally or over HTTP, verifying each link. `#<sha256>` on any member flag
pins it, and pinning the newest layer pins the whole chain.

**Deleting is not redacting.** `--remove` and `--supersede` take a record
out of search, but every earlier file stays byte-for-byte as it was, so the
record's bytes are still in the chain. `get_record` serves it by id, marked
`tombstoned: true`, so that citations made before the deletion still
resolve. To actually drop the bytes, `compact` the chain into a new base and
stop publishing the old members; anyone who already has a copy of them
keeps the record.

The base's encoder and calibration carry over. The chain keeps the base's
abstention fit until the records added plus deleted exceed the drift limit
(20% of the base's records by default); `append` refuses to cross it unless
`--allow-drift`, after which the chain serves `unscored`. `compact
--head <base> --head <layer> ... --out new.pikelet` folds a chain into one
fresh base with live records copied byte for byte and the calibration refit
(`--no-refit` skips the refit). `rebase --layer <file> --onto <chain> ...`
moves a layer onto a different parent, replaying only what that layer did.

`append` needs to know how the base's records were made, so new records
match them. Packs from this version of `compile` record it; for a pack
compiled earlier, `append` stops and prints the declaration to pass with
`--assert-ingest`. Shelves can list a chain's members as a `lineage` so a
mount does not have to walk locators. The format is specified in
`LAYERED_PROFILE.md` in the main repo.

## Where this sits

This package is the **product layer** of the Pikelet stack. It consumes the
two layers below it — the `pikelet-wasm` ANN engine and the Search Artifact
readers/builders (`spec/SEARCH_ARTIFACT_CONTRACT.md` in the main repo) — and
emits a project that is *yours*: the generated Worker, UI, and config are
application code with `pikelet-wasm` as a dependency, not part of this
package. Engine and artifact behavior are documented in the main repo;
this README covers only scaffolding, generation options, and the generated
project's layout.

### URL ingestion trust boundary

`--source <url>` crawls a website from your machine at build time. The
crawler runs locally under your account and follows the URL *you* typed —
including through the seed's own redirects (HTTP and meta-refresh, bounded
at 5 hops), since sites routinely send their root to a canonical host or a
localized landing page. The final seed URL defines the crawl origin; every
other fetch skips redirects (contentless meta-refresh pages are followed
through the normal frontier filters instead of wasting page budget). It
keeps the crawl frontier on that origin, enforces timeouts and
per-page/body caps, and never runs at query time — the deployed Worker
makes no outbound fetches at all. It deliberately does not block
private-network addresses: it is a local developer tool, and pointing it at
your own intranet docs is a supported use. Do not lift the crawl code into a
deployed service without adding SSRF protections (scheme allowlist,
private/link-local IP rejection, redirect pinning).

## Self-contained query embedding (`--mode student`)

`--mode student` removes the Workers AI dependency entirely. At build time the
CLI distills a corpus-specific teacher-student (PSTU) query encoder — the same
one the Docusaurus plugin and the edge docs-search demo use — and bundles it
into the Worker (~1.1 MiB). Queries embed in-process in single-digit
milliseconds, the generated `wrangler.toml` has no `[ai]` binding, and
`wrangler dev` runs fully local with no Cloudflare account:

```bash
npx pikelet create --name my-docs-search --source ./docs --mode student --no-deploy --yes
```

Training requires a Python 3 environment with `torch` and `transformers`
(`PIKELET_SEARCH_PYTHON` selects the interpreter). The trainer also calibrates
the abstention scorer and enforces acceptance gates; on small or noisy corpora
those gates can fail, in which case pass `--skip-abstention` to ship the
encoder without a match-quality scorer (responses report
`match_quality: "unscored"`), or improve the source corpus. When calibration
succeeds, `/search` reports `match_quality` and returns no results for
out-of-domain queries.

To reuse a previously trained encoder instead of retraining, pass
`--student-model <model.bin> --student-vectors <docs-vectors.f32>`
(optionally `--student-abstention <scorer.json>`). The teacher document
vectors must come from the same training run so the index geometry matches
the query encoder.

For local endpoint testing without Cloudflare Workers AI, generated Workers
support `LOCAL_STUB_AI=1`. It uses deterministic hash embeddings and is meant
only for testing the Worker/search path. If you build with
`PIKELET_SEARCH_STUB_EMBEDDINGS=1`, rebuild with real Workers AI embeddings
before deploy; stub-built indexes contain hash embeddings, not semantic
embeddings.

In Search Artifact mode, `/search` responses include per-query and cumulative
range-read stats so cold-load and warm-cache behavior are visible directly.

URL ingestion is intentionally conservative: crawls stay on the seed origin,
skip redirects, and cap HTML response bodies before parsing.

## Package layout

`bin/pikelet.mjs` calls `main()` in `src/cli.mjs`, which owns
argument parsing, dispatches every command, and holds the config a
scaffold is generated from. The work lives beside it:

| module | responsibility |
| --- | --- |
| `src/common.mjs` | package paths and version, config defaults, the model table, `CliError`, loaders that resolve `pikelet-wasm` (engine, `/artifact`, `/complete`) from npm or the monorepo |
| `src/ingest.mjs` | folder walk and URL crawl, HTML/Markdown/MDX extraction, chunking, dedupe, Docusaurus route mapping, the public chunk shape |
| `src/embed.mjs` | build-time embeddings: the inline-transformer kernel (workers-ai builds and complete artifacts), the student trainer, precomputed vectors, the deterministic stub, self-recall |
| `src/embed-worker.mjs` | the worker thread `embed.mjs` fans passage embedding out to |
| `src/student-embedder.mjs` | the corpus-distilled student query encoder used by `--mode student` |
| `src/complete-build.mjs` | kind-3 complete artifact assembly, the inline-encoder declaration, the pinned weights download |
| `src/calibrate.mjs` | build-time abstention calibration: probe generation, the retrieval-signals fit, its acceptance gates |
| `src/mcp.mjs` | the MCP server (`search`, `list_packs`, `get_record`, `verify_pack`), pack and chain mounting, shelves, `mcp install` |
| `src/append.mjs` | `append`: a layer of new, deleted or superseded records over a chain |
| `src/compact.mjs` | `compact` and `rebase` |
| `src/scaffold.mjs` | generated-project files: runtime modules, templates, `wrangler.toml` / `package.json`, student input staging, deploy |
| `src/build.mjs` | `buildAssets` (ingest → chunk → embed → index → artifact), config validation, `manifest.json`, student asset publishing, bundle sizing |
| `src/doctor.mjs` | the `doctor <url>` hosting probe |
| `docusaurus/` | the Docusaurus plugin and its browser client, built on `buildSearchAssets` |

## Checking a host: `doctor`

Range-read artifacts depend on transport properties that hosts get wrong
silently, and the symptom is "the demo is slow", not an error. Before (or
after) deploying a `.pikelet`, `.pikelet-sketch`, or `.pikelet-range` file,
probe the URL it is served from:

```bash
npx pikelet doctor https://example.com/search/search.pikelet
```

It prints a pass/warn/fail line per check — `HEAD` (size, `Accept-Ranges`,
`ETag`), a real 64-byte `Range` GET (206 vs full-body 200), the same range
with a `?r=start-end` cache-key query (the form every browser read uses, to
defeat Chromium's same-URL cache-entry lock), the negotiated protocol
(HTTP/1.1 serializes parallel rerank reads at ~6 connections; h2/h3
multiplex), median RTT over three small reads, and the artifact's magic and
identity from its first 64 bytes — and exits 1 if any check fails.

## Docusaurus

Docusaurus sites can build a static Pikelet Search Artifact through the package
subpath plugin:

```js
// docusaurus.config.js
import pikeletSearch from 'pikelet/docusaurus';

export default {
  plugins: [
    [
      pikeletSearch,
      {
        assetBase: 'pikelet-search',
        name: 'my-docs-search',
      },
    ],
  ],
};
```

On `docusaurus build`, the plugin indexes the rendered HTML in the build
output directory and, by default, compiles it into a complete kind-3
`search.pikelet` in `build/pikelet-search/` — hybrid retrieval, calibrated
abstention, section anchors, one file. The widget serves it over HTTP
range reads (opening on a fraction of the file, prefetching the inline
encoder in the background from the moment the search panel first opens)
and degrades to a bounded download-once on hosts that ignore `Range`.
Zero configuration: the packaged encoder assets stage automatically
(weights digest-pinned, fetched once). Opt-outs: `mode: 'student'` keeps
the small Python-trained student encoder with the range-artifact runtime;
`mode: 'artifact'` keeps the deprecated `.pikelet-range` output; both log
a deprecation note for the range format.

That means docs, blog posts, pages, and rendered MDX all flow through the
same folder ingestion, section-aware chunking, and complete-artifact builder
as the CLI's `compile`, without generating or deploying a Worker. The output
is `build/pikelet-search/search.pikelet` (plus `corpus.json` and
`manifest.json`). The widget defers all loading until the first time the
panel opens; results carry section heading-path breadcrumbs, and raw
distances only render when the mount element sets
`data-pikelet-debug="1"`. Because the artifact is range-read, the host must
honor `Range` — check with `pikelet doctor <url>` (the widget
degrades to a bounded one-time download when a host ignores it).

By default, the plugin injects a floating, draggable search panel into the page
and exposes `window.PikeletDocusaurusSearch` for custom UI code. The panel's JS
and CSS are bundled through Docusaurus; the generated static directory only
contains the search artifact assets. To ship only the assets and mount your own
UI, disable the default mount:

```js
[pikeletSearch, { assetBase: 'pikelet-search', mount: false }]
```

The `completeProfile` block tunes the default output when the packaged
assets are not what you want:

```js
[
  pikeletSearch,
  {
    assetBase: 'pikelet-search',
    sourcePath: 'docs',              // index markdown/MDX sources instead of built HTML
    sourceRouteBase: 'docs',
    completeProfile: {
      vocab: './my-vocab.txt',                     // default: packaged vocab
      weights: './pikelet-search/encoder-weights.bin', // default: fetched, digest-pinned
      maxTokens: 128,
      // vectors: './docs-vectors.f32',       // optional precomputed document vectors
      // calibration: './calibration.json',   // optional abstention calibration
    },
  },
]
```

Paths resolve against the site directory. `vocab.txt` ships in this package
(`src/inline-encoder/vocab.txt`) and is used when no `vocab` is configured.
The 24.3 MiB `encoder-weights.bin` does not ship: when it is absent (or a
configured path with that basename is missing), the plugin (and the CLI's
`runtime.mode: "complete"` path) downloads it once from the
`inline-encoder-v1` GitHub release, verifies the pinned SHA-256, and caches
it for reuse. Set `PIKELET_ENCODER_WEIGHTS_URL` to fetch from a mirror;
custom-named weights are never fetched. Without `vectors`, the build embeds
every chunk through the packaged encoder at build time (inputs longer than
`maxTokens` are windowed and mean-pooled, and the build logs how many).

### Student mode (deprecated range profile)

`mode: 'student'` (or configuring any `studentModel`/`trainStudent` option)
keeps the previous default: a `.pikelet-range` artifact plus a
corpus-distilled student encoder trained at build time. It needs a Python
environment with `torch` and `transformers`; set `trainStudent.python` if
Docusaurus should call a specific interpreter:

```js
[pikeletSearch, { mode: 'student', trainStudent: { python: '.venv/bin/python', epochs: 60 } }]
```

Advanced users can provide pre-trained assets, but the model has to travel with
the matching teacher document vectors for the rendered corpus:

```js
[
  pikeletSearch,
  {
    studentModel: './pikelet-student.bin',
    studentVectors: './docs-vectors.f32',
    studentAbstention: './student-abstention.json',
  },
]
```

`studentModel` is query-side only. The plugin refuses to build passages from a
bare student model because that silently changes the index geometry. A
Wikipedia-trained student is only useful for smoke testing the mechanics; it is
not a general-purpose docs encoder. `mode: 'artifact'` keeps the range
artifact without any student training (queries need an external embedding
path). Both modes log a deprecation note for the `.pikelet-range` format.

## Limitations

The bundled student encoder featurizes `[a-z0-9']` tokens only, and chunking
counts whitespace-separated tokens. English and other Latin-script,
whitespace-delimited content works; unsegmented scripts (CJK and similar) do
not — the build fails with an explicit 0-chunks error, and queries with no
recognized terms return a graceful no-match instead of results.
