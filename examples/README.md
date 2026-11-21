# Pikelet Examples

## Start here

### [`mcp-knowledge-pack/`](mcp-knowledge-pack/) — compile, mount, and query a pack over MCP

The current product, end to end, in the smallest form: compile a short
public-domain text into a `.pikelet`, mount it with `pikelet mcp`, and run
two queries — one straightforward, one whose premise is false — to show
what distance and `grounding.uncovered` do and don't tell an agent. If you
have five minutes, run this one.

### [`one-file-search/`](one-file-search/) — the file itself

Compiles the Search Artifact components (corpus, index, encoder,
evaluation) into one content-addressed `.pikelet` and serves
queries from it in Node and in the browser over HTTP range requests, with
no backend. This is what `mcp-knowledge-pack` mounts over MCP and what
`pikelet compile` builds.

### [`static-wiki-pack/`](static-wiki-pack/) — the same idea at wiki scale

The offline pipeline that builds a 456k-passage Wikipedia pack: embed,
compile, host on static/R2 storage, query entirely client-side. Also the
data-prep step `one-file-search`'s wiki-scale scripts consume directly.

If you have 15 seconds instead, search a published pack where it sits:

```bash
npx pikelet query "https://github.com/mcn92/pikelet/releases/download/knowledge-packs-v1/rust-book.pikelet#21f86277770e02b574ec18ceaa708c0526a93843fece63345cb8ba67551c9e03" "how do I borrow a value mutably"
```

The CLI range-reads the Rust book pack from a GitHub release, checks it
against the pinned identity after `#`, runs the query encoder from inside
the file, and prints the top passages; it never downloads the whole pack.
To build a `.pikelet` from your own folder, or scaffold a
deployable search app instead:

```bash
npx pikelet compile --source ./docs --out docs.pikelet
npx pikelet create --name my-docs-search --source ./docs --no-deploy --yes
```

## Legacy: the engine, directly

Everything below predates the single-file `.pikelet` artifact and the MCP
story above. These exercise `pikelet-wasm`'s raw HNSW engine or a live
server managing its own vectors and embeddings — the model this project
has moved away from as its product surface. They're kept because they're
still exercised by real tests or npm scripts and remain useful reference
for the low-level engine API, not because they demonstrate the current
direction.

- [`legacy/01-hello-pack/`](legacy/01-hello-pack/) — smallest bundled
  browser fixture for `pikelet-wasm/web`; raw `.add()`/`.search()` on
  hand-fed vectors. Backs `npm run test:browser`.
- [`legacy/02-catalog-hydration/`](legacy/02-catalog-hydration/) — search
  or a live server owns vectors; a catalog API hydrates live price and
  inventory by id. Demonstrates the "search returns ids, app hydrates
  state" pattern, pre-artifact.
- [`legacy/03-edge-docs-search/`](legacy/03-edge-docs-search/) — a
  Cloudflare Worker that restores a raw snapshot and embeds queries
  locally with a bundled student encoder: the "six separate asset files
  in a live server" shape `one-file-search`'s single artifact replaced.
  Its assets and abstention goldens now live in
  `test/fixtures/encoder-conformance/`, which is what `spec/` cites.
- [`legacy/reference-worker/`](legacy/reference-worker/) — full
  Cloudflare Worker reference: raw-vector `/search`/`/add` HTTP API, R2
  snapshot restore, auth, rate limiting, admin routes. Covered by
  `test/test_worker_features.js`.
- [`legacy/range-artifact-demo/`](legacy/range-artifact-demo/) — earlier
  `.pikelet-range` artifact demo and SIFT-shaped smoke harness. Wired into
  `npm run demo:artifact`/`demo:sketch`.

There is also a raw, no-bundler browser page at
[`technical-demo.html`](technical-demo.html).
