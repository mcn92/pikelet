# Pikelet Quick Start

This guide walks a folder of documents all the way through: compile it into
one `.pikelet` file, check what you built, query it from code, give it to an
LLM over MCP, publish it on a static host, and update it without rebuilding.
The last section covers the engine layer underneath, for when you already
have vectors and want the index alone.

The outputs below come from a real run over eight short Markdown pages
documenting a file-sync tool. Your identities and sizes will differ.

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

Point `compile` at a folder (Markdown, MDX, HTML or text), a single file, or
a live site:

```bash
pikelet compile ./docs --out docs.pikelet \
  --name ferry-docs --license MIT \
  --golden-queries questions.jsonl
# or crawl a site's docs:
pikelet compile https://docs.example.com/guide/ --out docs.pikelet
```

A crawl stays under the path you give it (`/guide/` and below; start at the
site root to crawl everything), follows the site's robots.txt, and waits
250 ms between requests, longer if robots.txt asks. It stops at 500 pages
unless `--max-pages` says otherwise.

```text
Ingested 8 docs -> 8 chunks
Embedded 8/8 chunks with inline transformer
Validated self-recall@1 on 8 sampled chunks
Embedded 24 golden queries, each verified to retrieve its source (5 of your 5 questions)
Built complete .pikelet artifact with 24.55 MiB
Compiled docs.pikelet
  24.55 MiB, 8 records, identity a8df9bb053b8369c06a01b62666b5a565690cd4b9f531f989053700fe734e17b
Next:
  pikelet query docs.pikelet "a question"   search it here
  pikelet mcp install --pack docs.pikelet   give it to Claude Code
```

This took 3 seconds. Compiling is mostly embedding, so time grows with the
corpus: about one chunk a second per CPU core, so 828 chunks took 110
seconds on 8 cores. `--verbose` also prints the
build's internals, such as index sizes and the rerank sweep.

The file carries the corpus records, the semantic index, a BM25 lexical
index, the inline MiniLM query encoder, integrity commitments, and golden
queries the finished pack was checked against. There is no
service, no model host and no vector database behind it. The `identity` on
the last line is the sha256 that names this exact pack; keep it, because
you pin it when you publish (step 5).

Every pack is at least ~25 MiB regardless of corpus size, because the query
encoder ships inside it. The first `compile` fetches that encoder (~25 MiB)
from a GitHub release and caches it; later runs are offline.

The flags in the example are optional but worth setting:

- `--name` is the pack name an LLM sees in `list_packs` and in every search
  result (default: the source folder's name, or the site's host name).
- `--license` is recorded in the manifest and surfaced by `list_packs`; set it
  on anything you will redistribute.
- `--golden-queries` takes questions a reader would really ask, one JSON
  object per line, each naming the page that answers it:

  ```json
  {"text": "how do I roll the bucket back to an earlier state", "expectTitle": "Snapshots and restore"}
  {"text": "what does a 403 error mean", "expectTitle": "Troubleshooting"}
  ```

  Each one is kept only if the finished pack returns the named page in its
  top 10, and is stored in the pack as a golden query that `verify_pack`
  replays. `compile` names any it drops. Questions it generates from the
  corpus fill the rest, up to 24.

Try it from the terminal before giving it to anything else:

```bash
pikelet query docs.pikelet "how do I restore an earlier snapshot" --limit 2
```

```text
ferry-docs: 2 results

1. Snapshots and restore
   snapshots.md#snapshots-and-restore
   Every write creates a new snapshot. To return the bucket to an earlier state, list the snapshots with ferry snapshots, pick the one you want by its timestamp, and run ferry restore <snapshot-id>. A restore writes the old files back as a new snapshot, so the restore itself can be…

2. Command reference
   cli.md#command-reference
   ferry init <bucket-url> sets up a folder to sync with a bucket. ferry status shows pending uploads, downloads and conflicts. ferry pause and ferry resume stop and restart syncing without quitting the daemon. ferry snapshots lists the bucket's snapshots, newest first. ferry resto…
```

`query` prints what the MCP `search` tool would return for the same question
(`--json` prints that response unchanged, `--limit` sets how many results).
A smaller distance is a closer match. When the passage that matches the
most of the question's words still lacks some of them, a last line lists
them:

```text
Question words the best passage lacks: best, pizza, chicago
```

That line is the useful signal. A passage can be the nearest one in the pack
without saying what was asked.

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

const out = await pack.query('how do I restore an earlier snapshot', { k: 3 });
console.log(out.results[0].title, out.results[0].id, out.results[0].distance.toFixed(3));
// Snapshots and restore 6 0.388

const miss = await pack.query('what is the best pizza in Chicago', { k: 3 });
console.log(miss.results[0].distance.toFixed(3), miss.grounding.uncovered);
// 1.020 [ 'best', 'pizza', 'chicago' ]

const record = await pack.record(out.results[0].id);  // full text and provenance
console.log(record.title, record.sourcePath);
// Snapshots and restore snapshots.md

await pack.close();
```

`query()` returns the nearest passages, each with its `distance`, and does not
judge whether they answer the question. `grounding.uncovered` lists the
question's words missing from whichever of the top five passages contains
the most of them; whoever reads
the passages, a person or a model, decides from there. The off-topic
question above still returns its nearest passages, at a much larger
distance, with every content word uncovered.

The response also carries `matchQuality`. Packs compiled after 0.14.0
report `'unscored'`. Packs calibrated by earlier releases keep their verdict
(`'strong' | 'weak' | 'none'`), and a `none` withholds `results` unless you
pass `showAbstained: true`. [docs/calibration.md](docs/calibration.md)
explains why the verdict was dropped.

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

The pack must exist; `install` refuses a path with no file. A pack inside
the project is written relative to it, so a committed `.mcp.json` works on
a teammate's checkout; the server resolves it against `CLAUDE_PROJECT_DIR`,
which Claude Code sets to the project root, so it works whichever directory
the session started in; one outside the project is written as an absolute path.
`--client claude-desktop` writes Claude Desktop's config instead, always
with absolute paths. Any other
MCP client can run `pikelet mcp --pack ./docs.pikelet` directly, with no
install step. The server offers four tools:

- `search`: per-pack results with provenance, distances, and the question
  words the best-matching passage lacks
- `list_packs`: names, licenses and immutable identities, for citation pinning
- `get_record`: one record's full text
- `verify_pack`: replays the golden queries stored inside the pack, and
  reports encoder and integrity checks

Asked to `verify_pack` the pack above, the server answers with
`"encoderVerified": true`, `"corpusIntegrity": "per-record-sha256"` and
`"goldenQueries": { "total": 24, "passed": 24, ... }`. A `search` for "how do
I keep snapshots forever" returns, trimmed:

```json
{
  "pack": "ferry-docs",
  "packIdentity": "a8df9bb053b8369c06a01b62666b5a565690cd4b9f531f989053700fe734e17b",
  "grounding": { "recordId": 6, "coverage": 0.333, "covered": ["snapshots"], "uncovered": ["keep", "forever"] },
  "results": [
    { "id": 6, "title": "Snapshots and restore", "source": "snapshots.md", "distance": 0.461, "text": "..." }
  ]
}
```

The passage is about snapshots, but "keep" and "forever" are in
`uncovered`: no passage says how to keep snapshots forever, and the model is
told not to claim one does. There is no relevance verdict: the model reads
the passages and decides, which measured as well as a verdict did (see
`docs/calibration.md`). Step 6 adds the missing page.

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
[  ok  ] ETag: W/"649833-25739378-2026-11-19T23:02:44.402Z"
[  ok  ] Range GET: 206, correct 64-byte slice
[  ok  ] Range GET with ?r= cache key: 206 — Chromium cache-lock workaround will function
[  ok  ] artifact: complete artifact (PSF1), identity a8df9bb053b8369c…
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
  expectedIdentity: 'a8df9bb053b8369c06a01b62666b5a565690cd4b9f531f989053700fe734e17b',
});
```

Under Node, `pikelet-wasm/complete` reads over `nodeFetch`, a `node:http`
transport that never decompresses and refuses any response a server
encoded; browsers and Workers use their own `fetch`. Pass
`httpRangeSource(url, { fetch })` to choose another transport.

`expectedIdentity` pins the pack: a file whose identity differs is refused
one 64-byte read in, before anything else is fetched:

```text
.pikelet identity mismatch: expected 0000…, found a8df9bb053b8…
```

Give the same URL to an LLM, pinned with `#<identity>`:

```bash
pikelet mcp install --client claude-code \
  --pack 'https://example.com/docs.pikelet#a8df9bb053b8369c06a01b62666b5a565690cd4b9f531f989053700fe734e17b'
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
Embedding 1 chunk(s) with the base's own encoder (sentence-transformers/all-MiniLM-L6-v2)
...
Wrote docs.0001.pikelet
  4.3 KiB, depth 1, +1 record(s), identity 3d408e74cb129144fda6ed536ec965d66a0fbc5cbe520eb623092d45c3254495
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

Each layer is one more file to fetch and verify, and one more index to
search. Once a chain has grown long, fold it into one fresh base:

```bash
pikelet compact --head docs.0001.pikelet --out docs-v2.pikelet
```

```text
Mounted chain: 2 member(s), 9 records (9 live), 0 tombstone(s), depth 1
Compacting to 9 live record(s), reclaiming 0 id(s)
...
Wrote docs-v2.pikelet
  24.55 MiB, 9 records, depth 0, identity e1f24cd6aaa2b290f1e96c78a59a7e4f94f15a686664dc363c5758497c040fe0
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
the query vectors, and the metadata hydration yourself; grounding, the
lexical index, golden queries and the corpus records are all artifact-layer
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

The base was compiled, and calibrated, by 0.14.0 or earlier, and the
chain has changed more than 20% of the base's records since. Run `compact`
over the chain (step 6) and append to the result, or pass `--allow-drift`
and accept that the chain serves `unscored` in the library. Bases compiled
since carry no calibration and are never refused for drift.

### `query()` always reports `matchQuality: 'unscored'`

That is expected for a pack compiled after 0.14.0: `compile` no longer fits
a relevance verdict (step 3). Results still come back; use each result's
`distance` and the response's `grounding.uncovered`.

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
