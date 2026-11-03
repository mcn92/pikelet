# Pikelet Quick Start

This guide walks a folder of documents all the way through: compile it into
one `.pikelet` file, check what you built, query it from code, give it to an
LLM over MCP, publish it on a static host, and update it without rebuilding.
The last section covers the engine layer underneath, for when you already
have vectors and want the index alone.

The outputs below come from a real run over eight short Markdown pages
documenting a file-sync tool. Your identities, sizes and calibration numbers
will differ.

## 1. Install

```bash
npm install -g pikelet       # the CLI: compile, mcp, append, doctor, create
npm install pikelet-wasm     # the library: readers and the engine
```

The CLI installs two packages, itself and `pikelet-wasm`, with no native code.
`pikelet create` embeds its corpus locally with the same WASM kernel and
serves queries from Cloudflare Workers AI; it downloads its 36 MiB encoder
weights on first use.

The CLI needs Node 22.23.4+, 24.22.1+ or 26.12.1+. `pikelet-wasm` runs on
Node 18+, browsers, and Cloudflare Workers (CI tests Node 22, 24 and 26); this guide assumes both packages at
0.13.0 or later, the first release with the on-disk cache in step 5. The
CHANGELOG lists what changed between releases.

## 2. Compile a pack

Point `compile` at a folder (Markdown, MDX, HTML or text) or a live site:

```bash
pikelet compile --source ./docs --out docs.pikelet \
  --name ferry-docs --license MIT \
  --calibration-queries questions.jsonl
# or crawl a site:
pikelet compile --source https://docs.example.com --out docs.pikelet
```

```text
Ingested 8 docs -> 8 chunks
Embedded 8/8 chunks with inline transformer
Validated self-recall@1 on 8 sampled chunks
Built lexical index: 199 terms over 8 records (5 KiB)
Embedded 3 encoder verification vectors in the declaration
Calibrated relevance: 67 on-topic / 81 off-domain + 24 gibberish off-topic / 0 weak queries,
  5-fold CV AUC 0.999858 vs off-topic (fit AUC 1, in-sample), validation: 5 human queries:
  AUC 1, abstention rate 0%
  ...
Measured rerank operating point: C=8 (recall@8 1 over 8 corpus-embeddings queries)
Golden replay against the assembled artifact: 24/24 reproduce
Built complete .pikelet artifact with 24.55 MB
Compiled docs.pikelet
  24.55 MB, 8 records, identity cbec6b68845d34d9e48bf43f4192110eb0f90219124342129dd051fe2143eece
```

The file carries the corpus records, the semantic index, a BM25 lexical
index, the inline MiniLM query encoder, integrity commitments, the
calibrated abstention threshold, and the evaluation fixtures. There is no
service, no model host and no vector database behind it. The `identity` on
the last line is the sha256 that names this exact pack; keep it, because
you pin it when you publish (step 5).

Every pack is at least ~24 MB regardless of corpus size, because the query
encoder ships inside it. The first `compile` fetches that encoder (~25 MiB)
from a GitHub release and caches it; later runs are offline.

The flags in the example are optional but worth setting:

- `--name` is the pack name an LLM sees in `list_packs` and in every search
  result (default: the output file name).
- `--license` is recorded in the manifest and surfaced by `list_packs`; set it
  on anything you will redistribute.
- `--calibration-queries` takes questions a reader would really ask, one JSON
  object per line, each naming the page that answers it:

  ```json
  {"text": "how do I roll the bucket back to an earlier state", "expectTitle": "Snapshots and restore"}
  {"text": "what does a 403 error mean", "expectTitle": "Troubleshooting"}
  ```

  They validate the abstention fit (they are never training data for it) and
  become golden queries stored in the pack, which `verify_pack` replays.

Other flags: `--force` overwrites the output, `--max-pages` caps a crawl, and
`--include`/`--exclude` filter a folder. `pikelet compile --help` lists the
rest, including the `--encoder-*` flags for swapping the packaged MiniLM.
Documents too short to index are dropped, and `compile` names each one it
drops.

Two other sources replace `--source`:

```bash
pikelet compile --git ./repo --git-rev v0.9.0..HEAD --out history.pikelet
pikelet compile --conversations ~/.claude/projects/my-project --out conversations.pikelet
```

`--git` builds one record per commit (subject, message, date, author, files
changed; merges excluded), cited by the commit's GitHub URL when `origin` is
on GitHub and by its sha otherwise. `--git-rev` takes a revision or a range
and defaults to `HEAD`, the whole history. `--conversations` builds one
record per exchange from Claude Code session transcripts or a Claude.ai or
ChatGPT data export, keeping only what was said and redacting common
credential shapes and email addresses. Redaction is pattern-based and best
effort, so keep a conversation pack private.

## 3. Query it from code

```js
import { openPikeletFile } from 'pikelet-wasm/complete';

const pack = await openPikeletFile('docs.pikelet');

const out = await pack.query('how do I roll back to an earlier snapshot', { k: 3 });
console.log(out.matchQuality, out.results[0].title, out.results[0].id);
// strong Snapshots and restore 5

const miss = await pack.query('what is the best pizza in Chicago', { k: 3 });
console.log(miss.matchQuality, miss.results.length);
// none 0

const record = await pack.record(out.results[0].id);  // full text and provenance
console.log(record.title, record.sourcePath);
// Snapshots and restore snapshots.md

await pack.close();
```

`query()` returns `matchQuality` — `'strong' | 'weak' | 'none' |
'unscored'` — alongside the results. An off-domain query abstains instead of
answering, and a `none` verdict withholds `results` unless you pass
`showAbstained: true`. The threshold was fit from the corpus at build time
and stored in the file, so the refusal behavior travels with the pack rather
than living in your application code.

`matchQuality` measures whether the pack covers the question's topic, not
whether a passage states the exact fact asked for. Each result's
`grounding.uncovered` lists the question's words no top passage contains;
step 4 shows it.

## 4. Give it to an LLM

`mcp` exposes packs over the Model Context Protocol on stdio. Write the
config for your client:

```bash
pikelet mcp install --client claude-code --pack ./docs.pikelet
```

```text
Wrote MCP server "knowledge-packs" to ./.mcp.json
Claude Code picks it up on the next session in this project.
```

`--client claude-desktop` writes Claude Desktop's config instead. Any other
MCP client can run `pikelet mcp --pack ./docs.pikelet` directly, with no
install step. The server offers four tools:

- `search`: per-pack results with provenance and calibrated abstention
- `list_packs`: names, licenses and immutable identities, for citation pinning
- `get_record`: one record's full text
- `verify_pack`: replays the golden queries and abstention probes stored
  inside the pack, and reports encoder and integrity checks

Asked to `verify_pack` the pack above, the server answers with
`"encoderVerified": true`, `"corpusIntegrity": "per-record-sha256"` and
`"goldenQueries": { "total": 24, "passed": 24, ... }`. A `search` for "how do
I keep snapshots forever" returns, trimmed:

```json
{
  "pack": "ferry-docs",
  "packIdentity": "cbec6b68845d34d9e48bf43f4192110eb0f90219124342129dd051fe2143eece",
  "matchQuality": "strong",
  "grounding": { "recordId": 5, "covered": ["snapshots"], "uncovered": ["keep", "forever"] },
  "results": [
    { "id": 5, "title": "Snapshots and restore", "source": "snapshots.md", "text": "..." }
  ]
}
```

The pack is on topic (`strong`), but "keep" and "forever" are in
`uncovered`: no passage says how to keep snapshots forever, and the model is
told not to claim one does. Step 6 adds the missing page.

## 5. Publish it

A `.pikelet` needs no server of its own. Upload it to anything that answers
HTTP Range requests with `206 Partial Content`: R2, S3, GitHub release
assets, Cloudflare Pages, nginx, a CDN. Readers fetch byte ranges and never
download the whole file.

Check the host before you rely on it:

```bash
pikelet doctor https://example.com/docs.pikelet
```

```text
[ info ] size: 24.55 MiB
[  ok  ] Accept-Ranges header: bytes
[  ok  ] ETag: W/"245770-25745192-2026-10-14T20:37:06.185Z"
[  ok  ] Range GET: 206, correct 64-byte slice
[  ok  ] Range GET with ?r= cache key: 206 — Chromium cache-lock workaround will function
[  ok  ] artifact: complete artifact (PSF1), identity cbec6b68845d34d9…
```

It also reports the HTTP protocol and the round-trip time, and exits 1 if any
check fails. (These checks come from a local plain-HTTP server, which also
drew a warning that HTTP/1.1 caps parallel reads; HTTPS hosts with h2 avoid
it.)

Mount it by URL from code with `httpRangeSource`. `openPikeletFile` treats a
bare string as a local path, so wrap the URL:

```js
import { openPikeletFile, httpRangeSource } from 'pikelet-wasm/complete';

const pack = await openPikeletFile(httpRangeSource('https://example.com/docs.pikelet'), {
  expectedIdentity: 'cbec6b68845d34d9e48bf43f4192110eb0f90219124342129dd051fe2143eece',
});
```

Under Node, `pikelet-wasm/complete` reads over `nodeFetch`, a `node:http`
transport that never decompresses and refuses any response a server
encoded; browsers and Workers use their own `fetch`. Pass
`httpRangeSource(url, { fetch })` to choose another transport.

`expectedIdentity` pins the pack: a file whose identity differs is refused
one 64-byte read in, before anything else is fetched:

```text
.pikelet identity mismatch: expected 0000…, found cbec6b68845d…
```

Give the same URL to an LLM, pinned with `#<identity>`:

```bash
pikelet mcp install --client claude-code \
  --pack 'https://example.com/docs.pikelet#cbec6b68845d34d9e48bf43f4192110eb0f90219124342129dd051fe2143eece'
```

`--pack` repeats for several packs; if one fails to mount (an unreachable
host, a refused pin), the others still serve and `list_packs` reports the
missing one.

`mcp` caches what a URL mount reads on disk, under `~/.cache/pikelet/blobs`
(`$PIKELET_CACHE_DIR` moves it). Most of a mount is two regions the pack
commits to by sha256, the resident index and the query encoder, so the next
process to mount the same pack reads them locally instead of over the
network. From the release after 0.13.0, the verified pieces a query reads
(rerank rows, records, digest pages) are kept the same way. The keys
are content digests, so an entry never goes stale, and every hit is
re-hashed before use. `--no-cache` turns it off. From code, pass
`blobCache: diskBlobCache()` (exported by `pikelet-wasm/complete` under Node)
to `openPikeletFile`.

To publish several packs together, list them in a static
`packs.json` shelf and mount it with `--shelf <url>`; see
[packs/README.md](packs/README.md) for the format.

## 6. Update it

When a few documents change, `append` writes a small layer on top of the
pack instead of rebuilding it. The parent is never modified:

```bash
pikelet append --parent docs.pikelet --source ./changed --out docs.0001.pikelet
```

```text
Planned layer: depth 1, rowBase 8, +1 record(s), 0 new tombstone(s), 0 supersession(s)
  calibrationDrift would be 0.1250 against limit 0.2
...
Wrote docs.0001.pikelet
  7.0 KiB, depth 1, +1 record(s), identity 5b4a937595bed28b226aece374f6fd50b69ff31a6b8e068e9471c834d89607d7
```

A layer can also delete records (`--remove <id>`, one id per flag, using the
ids that search results report) or replace one (`--supersede
<oldId>=<path>`). Mount the newest layer and the reader walks down to the
base, verifying every link; pinning the newest layer pins the whole chain:

```bash
pikelet mcp --pack ./docs.0001.pikelet
```

The search from step 4 now finds the new "Snapshot retention" page, with
nothing left in `uncovered`. Publish the layer next to its parent, so a
reader mounting it by URL finds the base the same way.

A chain keeps the base's calibration only while records added plus deleted
stay within 20% of the base's records. Past that, `append` refuses:

```text
this append would take calibrationDrift to 0.2500, past the chain's effective
limit 0.2; pass --allow-drift to emit it anyway (the chain then serves unscored)
```

That is the point to fold the chain into one fresh base, with the
calibration refit:

```bash
pikelet compact --head docs.pikelet --head docs.0001.pikelet --out docs-v2.pikelet
```

```text
Compacting to 9 live record(s), reclaiming 0 id(s)
Refitting calibration over 9 live record(s) with the base's own encoder ...
Wrote docs-v2.pikelet
  24.55 MB, 9 records, depth 0, identity 363c71e6b1f48066025f705f2ef366e8df64bce2ca61ad2473a981f50f938846
```

The compacted file is an ordinary pack: open it with `openPikeletFile`, mount
it, or append to it. Citations made against the old chain still translate to
it. From code, use `openPikeletFile` on single-file packs; chains are served
through `pikelet mcp`, or compacted first. `rebase` moves a layer onto a
different parent. [README.md](README.md#updating-a-pack)
and [the CLI README](packages/pikelet/README.md#updating-a-pack-append-compact-rebase)
cover the details.

## 7. Optional: a journal the agent can write

Every pack is read-only to the model, with one opt-in exception. A journal is
a private pack the agent appends notes to (decisions, dead ends,
measurements, handoffs), so a later session can pick up where the work
stands:

```bash
pikelet journal init my-project
pikelet mcp install --client claude-code --pack ./docs.pikelet --journal my-project
```

This adds two tools, `write_note` and `latest`. A note comes back labelled
agent-written and unverified, with the name of the MCP client that wrote it
(`e2e` below is the test client this run used):

```bash
pikelet journal list my-project
```

```text
#1  2026-10-14T20:39:58.198Z  decision  Retention default
    by e2e  tags: snapshots
    Keep the 30-day snapshot retention default; retention_days = 0 keeps snapshots forever.
```

A bare name lives in `~/.pikelet/journals/`; pass a directory path to keep it
elsewhere. The feature is experimental; see
[the CLI README](packages/pikelet/README.md#a-journal-the-agent-can-write-experimental)
and [docs/agent-journal.md](docs/agent-journal.md).

## The engine layer

Everything above is the artifact layer. Underneath it is the HNSW engine,
which you can use directly when you already have vectors and want an index
rather than a queryable pack. At this layer you supply the embedding model,
the query vectors, and the metadata hydration yourself; `matchQuality`,
abstention, the lexical index and the corpus records are all artifact-layer
features that do not exist here.

Reach for it when you are embedding with your own model and only need
approximate nearest neighbors. If you have documents and want search, use
`compile` above instead.

`loadJsonFile()` and `loadSnapshotFile()` are Node-only; the rest of the
engine API also runs in browsers and Workers.

### From a repository checkout

```bash
git clone https://github.com/mcn92/pikelet.git
cd pikelet
npm install          # dev deps for the demos, benchmarks, and tests
```

No build step is needed: the checkout ships the prebuilt WASM engine in
`packages/pikelet-wasm/dist/` (`engine.{js,wasm}` plus the scalar fallback `engine.scalar.{js,wasm}`),
and the root entry points use it directly.

Rebuild the engine only if you are changing the C++ under `packages/pikelet-wasm/engine/`. That
requires an Emscripten toolchain (see the README's "Building from source"):

```bash
npm run build:all    # rebuilds packages/pikelet-wasm/dist/engine.* — engine development only
```

(`./build.sh` builds only the SIMD pair, `packages/pikelet-wasm/dist/engine.{js,wasm}`; use `build:all` when you also need the scalar fallback.)

### Pick an ingest path

Use the path that matches what you already have:

- Vectors already in memory: `Pikelet.fromVectors(...)`
- Vectors saved as JSON / JSONL: `Pikelet.loadJsonFile(...)` on the Node entrypoints
- Existing Pikelet snapshot on disk: `Pikelet.loadSnapshotFile(...)` on the Node entrypoints

If you are working from a repo checkout, `import Pikelet from 'pikelet-wasm'` already resolves to the in-tree package: the repository is an npm workspace, and `npm install` at the root links `node_modules/pikelet-wasm` to `packages/pikelet-wasm`. The entrypoints themselves live at `packages/pikelet-wasm/src/`.

### Local Node workflow

#### 1. Build an index from in-memory vectors

```js
import Pikelet from 'pikelet-wasm';

const rows = [
  { id: 'doc-1', vector: [1, 0, 0, 0] },
  { id: 'doc-2', vector: [0, 1, 0, 0] },
  { id: 'doc-3', vector: [0, 0, 1, 0] },
];

const { index, ids, idMap } = await Pikelet.fromVectors(rows, {
  metric: 'cosine',
  quantized: true,
});

const results = index.search(new Float32Array([1, 0, 0, 0]), 2);
console.log(results);
console.log(idMap.get(results[0].id)); // -> 'doc-1'
```

Use this path when your embedder already returns arrays or `Float32Array`s in the current process.

#### 2. Build an index from JSON or JSONL

On the Node.js entrypoints, Pikelet can load vectors directly from disk:

```js
import Pikelet from 'pikelet-wasm';

const { index, ids, idMap } = await Pikelet.loadJsonFile('vectors.jsonl', {
  metric: 'cosine',
  quantized: true,
  vectorKey: 'embedding', // default: 'vector'
  idKey: 'docId',         // default: 'id'
  maxFileBytes: 64 * 1024 * 1024,
});

const results = index.search(new Float32Array([1, 0, 0, 0]), 5);
console.log(results);
console.log(idMap.get(results[0].id));
```

Supported file types:

- `.json`
- `.jsonl`
- `.ndjson`

Accepted row shapes:

- `[1, 2, 3, ...]`
- `{ "id": "doc-1", "vector": [1, 2, 3, ...] }`
- `{ "docId": "doc-1", "embedding": [1, 2, 3, ...] }` with `idKey` / `vectorKey`

If your embedding pipeline already writes JSONL, this is the simplest file-based path.

#### 3. Export and restore a snapshot

If you want to reuse a built index later, export a snapshot:

```js
import fs from 'node:fs';
import Pikelet from 'pikelet-wasm';

const { index } = await Pikelet.fromVectors([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
], {
  metric: 'cosine',
  quantized: true,
});

// If you have deleted anything, compact() before export().
const snapshot = index.export();
fs.writeFileSync('index.pnck', snapshot);

const restored = await Pikelet.loadSnapshotFile('index.pnck', {
  dim: 4,
  maxElements: 2,
  metric: 'cosine',
  quantized: true,
});

console.log(restored.search(new Float32Array([1, 0, 0, 0]), 1));
```

Snapshot notes:

- `export()` throws if `ghostCount > 0`; call `compact()` first after deletions
- `loadSnapshotFile()` is Node-only
- `loadSnapshotFile()` restores Pikelet snapshots from disk, not arbitrary ANN binary formats

### Embedding your own documents

Pikelet does not care which embedder you use, as long as you end up with vectors.

Typical workflow:

1. Read your source documents.
2. Generate embeddings with your model or API of choice.
3. Store them either:
   - directly in memory and call `fromVectors()`, or
   - in JSON / JSONL and call `loadJsonFile()`
4. Query the built index with new query embeddings produced by the same model family.

If you already have parquet, numpy, or another upstream format, convert it into JSONL or feed the vectors into `fromVectors()` directly from your application code.

### Reference Worker example

This example predates the `.pikelet` artifact and lives under
`examples/legacy/`; a pack mounted over HTTP Range needs no Worker at all. It
is still the reference for exposing the raw engine over HTTP, and the
`/export`/`/import` contract below is still current, so it is documented here
rather than removed.

It is repo-based, loads the checked-in WASM engine artifacts, and exposes
Pikelet over HTTP. Treat it as a deployment pattern you run in your own
Cloudflare account, not as a centrally hosted Pikelet service.

When you run or deploy this example, it runs in your own Cloudflare environment:

- `npx wrangler dev` starts a local dev instance on your machine
- `wrangler deploy` publishes the Worker into the Cloudflare account authenticated in your local Wrangler setup
- any R2 bucket, auth settings, and rate limits belong to your own deployment, not to this repository

#### Run the Worker locally

```bash
cd examples/legacy/reference-worker
npx wrangler dev --port 8787 --var ALLOW_INSECURE_ADMIN:1
```

`ALLOW_INSECURE_ADMIN=1` is a local-only opt-in: without it (or an `API_KEY`), admin routes such as `/init`, `/add`, and `/import` return 403, so the curl examples below would be rejected.

#### Deploy the Worker to your own Cloudflare account

```bash
cd examples/legacy/reference-worker
wrangler r2 bucket create pikelet-indexes
wrangler deploy
```

#### Initialize a small index over HTTP

In another terminal:

```bash
curl -X POST http://localhost:8787/init \
  -H 'Content-Type: application/json' \
  -d '{
    "dims": 4,
    "maxElements": 16,
    "vectors": [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0]
    ]
  }'
```

Then query it:

```bash
curl -X POST http://localhost:8787/search \
  -H 'Content-Type: application/json' \
  -d '{"query": [1, 0, 0, 0], "k": 2}'
```

Useful local checks:

```bash
curl http://localhost:8787/health
curl http://localhost:8787/stats
```

#### Worker import / export contract

The Worker has its own binary envelope for `/export` and `/import`.

- `GET /export` returns a Worker snapshot blob
- `POST /import` expects a previous blob from that Worker `/export` path

Do not assume that a Node.js `index.export()` snapshot is interchangeable with the Worker `/import` format. The Worker wraps the engine snapshot with additional metadata for its own restore path.

#### Worker integration test

```bash
node test/test_worker_features.js
```

Starts a local `wrangler dev` instance against `examples/legacy/reference-worker` and runs a synthetic 1536D Worker/API integration test against it.

### Sizing and tuning

| Parameter | Typical range | Effect |
|-----------|---------------|--------|
| `M` | 8–32 | Graph connectivity. Higher improves recall, increases memory and build time. |
| `efConstruction` | 50–400 | Build beam width. Higher improves graph quality, slows build. |
| `efSearch` / `ef` | 50–500 | Query beam width. Higher improves recall, slows search. |
| `quantized` | `true` / `false` | uint8 quantization cuts memory significantly with some recall tradeoff. |
| `metric` | `'cosine'` / `'l2'` | Use cosine for normalized embeddings and L2 for unnormalized vectors. |

For memory-constrained Worker deployments, the quantized path is usually the right default.

## Troubleshooting

### `openPikeletFile()` fails with `ENOENT` on a URL

A string argument is a local path. Wrap a URL: `openPikeletFile(httpRangeSource(url))`.

### `append` refuses with `calibrationDrift`

The chain has changed more than 20% of the base's records since it was
compiled. Run `compact` over the chain (step 6) and append to the result, or
pass `--allow-drift` and accept that the chain serves `unscored`.

### Every query comes back `unscored`

The pack has no calibration: the corpus was too small for a trustworthy fit
(`compile` logs the reason), it was compiled with `--skip-calibration`, or a
chain passed its drift limit. Results still come back; they carry no
confidence verdict.

### `loadJsonFile()` rejects my file

Check:

- the extension is `.json`, `.jsonl`, or `.ndjson`
- the rows contain vectors under the expected field name
- `vectorKey` / `idKey` match your actual JSON schema

### I already have vectors in another format

Pikelet does not currently load formats like `.fvecs`, `.npy`, or `.parquet` directly.

Use one of these instead:

- convert the vectors to JSONL and call `loadJsonFile()`
- load them in your own code and pass them to `fromVectors()`

### Worker `/import` does not accept my local snapshot

The Worker `/import` route expects the Worker export format, not a raw local package snapshot. Use the Worker's own `/export` output when testing `/import`.

## Next steps

- [README.md](README.md) for what a `.pikelet` is and the engine API surface
- [packages/pikelet/README.md](packages/pikelet/README.md) for the full CLI reference: `compile`, `mcp`, `append`, `compact`, `rebase`, `journal`, `create`, `doctor`
- [spec/COMPLETE_PROFILE.md](spec/COMPLETE_PROFILE.md) for the artifact format, and [LAYERED_PROFILE.md](LAYERED_PROFILE.md) for layers and chains
- [examples/legacy/reference-worker/README.md](examples/legacy/reference-worker/README.md) for the reference Worker deployment model
- [docs/architecture.md](docs/architecture.md) for the system design document
