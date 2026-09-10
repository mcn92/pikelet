# Range-read proof

The boring proof: static file hosting + HTTP Range + client-side retrieval =
no vector DB, no retrieval backend.

Serves a real `.pikelet` pack from a deliberately dumb static HTTP server
(`fs.createReadStream` + `Range` support, nothing else — no index, no query
parsing, no knowledge of what a `.pikelet` is), mounts it remotely through
Pikelet's real `httpRangeSource`/`openPancakeFile`, runs several queries,
and reports exactly how many bytes and HTTP range requests each step cost.

## Run it

```
node bench/range-proof/proof.mjs [path/to/pack.pikelet]
```

Defaults to `examples/05-one-file-search/pancake-wiki-inline.pancake` (the
648.5 MiB Simple English Wikipedia pack; build it first with `node
examples/05-one-file-search/compile-wiki.mjs --inline-encoder` if it isn't
present — needs `examples/04-static-wiki-pack/data-perm/` on disk). Prints,
in order: the pack's on-disk size, the dumb server starting (plus a `ps
aux` check showing no retrieval-backend process is running), the remote
mount with its hash pin verified, five queries with per-query
bytes/requests, a repeated query showing the cache effect, and a summary
table.

Queries are picked per pack (`FALLBACK_QUERIES` in the script) — a query
set only makes sense for the corpus it was written against; passing a
different pack without a matching entry falls back to the wiki pack's
general-knowledge query set, which will read as off-topic noise against a
narrow-domain pack.

The mount is opened with `prefetchEncoder: false` so "mount cost"
(structural reads: header, manifest, segment table, resident sketch,
lexical index) is reported separately from the pack's inline query-encoder
model weights (~25 MiB for the bundled MiniLM), which download lazily on
the first query and are then cached for the process lifetime. Folding the
two together would make "mount" look like it pulls most of the file, which
isn't what mounting costs — it's what *becoming ready to answer at all*
costs, once, for a self-contained pack that carries its own encoder.

## Failure modes

```
node bench/range-proof/failure-modes.mjs [path/to/pack.pikelet]
```

Exercises four scenarios against a real reader/pack pair:

1. Server honors Range correctly — baseline, succeeds.
2. Server ignores Range and always returns the full file — Pikelet falls
   back to a one-time full download, loudly logged and flagged in
   `source.stats.fullFallback`, not silently.
3. The bytes at the URL change mid-session while the caller's hash pin
   stays the original — refused with a hash-verification error, not
   served.
4. One range response is truncated — fails loudly (a body-length
   mismatch aborts the read), not silently wrong.

## MCP proof

```
node bench/range-proof/mcp-proof.mjs [path/to/pack.pikelet]
```

The other half of the demo: mounts the pack over HTTP into the *real*
`pikelet mcp` server (`pikelet/bin/pikelet.mjs mcp <url>#<sha256>`) as a
subprocess, then drives it as an actual MCP client would — JSON-RPC 2.0
over stdio (`initialize`, `tools/list`, `tools/call` with `list_packs` and
`search`) — exactly what an agent framework's LLM tool caller does, not a
direct call into the reader library. Reports each query's MCP response
(matchQuality, confidence, cited records with snippets) next to the dumb
server's own range-request count and bytes for that call, pulled from the
server's independent request log — so the "answer + citations" side and
the "what actually went over the wire" side are shown from two different
vantage points that have to agree.

## What this is not

Not a benchmark of retrieval quality or latency under load — see
`bench/beir/` for retrieval-quality measurement. This only demonstrates
the transport/architecture claim: a `.pikelet` pack on any static file
host (`python -m http.server`, nginx, S3/R2, GitHub Releases, ...) needs
no application server, index service, or database process to serve
retrieval — only `Range` support.
