# Browser Search Artifact Demo

This is a static browser demo for Pikelet range-readable artifacts. It is a
fully client-side semantic search over the Pikelet documentation: a
`.pikelet-range` index opened through HTTP range requests, a 1.08 MiB distilled
student encoder that embeds queries locally, and the docs corpus for result
display. No backend, no outbound API calls — a static host only distributes
bytes, and the browser accumulates cache state as it searches.

On open, the v2 router segment (15 records, ~9.5 KiB) becomes resident and
base-layer records are lazily materialized through coalesced range reads. The
metrics panel shows per-query wall time, embed time, range requests/bytes,
sequential miss rounds, and cached node count, so cold-versus-warm behavior is
directly visible.

From the repository root:

```bash
npm run demo:artifact:browser:build
npm run demo:artifact:browser
```

Then open the printed local URL. Use the raw Vite commands
(`npx vite build|preview examples/legacy/range-artifact-demo/static`) if you prefer;
the build/preview flow is required — raw `vite dev` serves repo CommonJS files
too literally.

Bundled assets under `public/`:

```text
artifacts/pikelet-docs.pikelet-range   docs index (208 chunks, 384D, 135 KiB)
artifacts/pikelet-smoke-split.pikelet-range   tiny SIFT-shaped smoke artifact
models/docs-student.bin                distilled query encoder (1.08 MiB)
corpus/docs-corpus.json                doc chunks for result display
```

The docs artifact is built from the worker-semantic-search snapshot:

```bash
node -e "require('./pikelet.js').buildRangeArtifactFile(
  'examples/legacy/03-edge-docs-search/assets/docs-index.bin',
  'examples/legacy/range-artifact-demo/static/public/artifacts/pikelet-docs.pikelet-range',
  { layout: 'rcm' })"
```

Corpus chunk ids map one-to-one onto artifact node ids (the snapshot was built
in insertion order with no deletions), so results hydrate by direct lookup. If
you rebuild the index, rebuild the artifact and re-copy
`docs-corpus.json` and `docs-student.bin` from
`examples/legacy/03-edge-docs-search/assets/` together — the encoder, index, and
corpus are one matched set.

## Measured warm-cache behavior

Measured 2026-07-31 with headless Chromium against an instrumented local Range
server, driving 100 held-out doc queries (server-side request counts, so the
numbers are what actually crossed the network):

| Phase | Artifact requests | Artifact bytes |
| --- | ---: | ---: |
| Page load (open, router resident) | 3 | 10.4 KiB |
| Query 1 (cold) | 8 | 121.8 KiB |
| Queries 2–10 | 2 | 1.3 KiB |
| Queries 11–100 | 0 | 0 B |

Cold first-query wall time was ~132 ms; warm queries ran at p50 0.4 ms /
p95 0.9 ms with zero network. Total artifact transfer across 100 queries was
136.6 KiB — essentially the artifact size, with no redundant refetching.

On a page reload, HTTP cache headers decide the re-warm cost:

| Reload cost | No cache headers | `max-age=31536000, immutable` |
| --- | ---: | ---: |
| Page assets (JS/WASM/model/corpus) | 1.37 MiB refetched | 2.4 KiB (index.html only) |
| Artifact ranges (open + first query) | ~24.9 KiB | ~10.4 KiB |
| First-query wall after reload | 114.7 ms | 84.1 ms |

Chromium partially reuses cached 206 range responses either way, but the
1.37 MiB of static assets only stays local when the host sends long-lived
cache headers — configure them on whatever host serves this demo.

The build writes static files to:

```text
examples/legacy/range-artifact-demo/static/dist
```

For deployment, upload the contents of `dist` to any static host. Hosts that
preserve `Range` requests get true lazy reads; hosts that ignore the `Range`
header are detected at the first read and handled by a one-time full download
served as local slices (the header line reports
`host ignores Range: full artifact fetched once`).

## Static host findings (2026-07-31)

Live deployment: https://pikelet-artifact-demo.pages.dev

| Host | Range behavior | Notes |
| --- | --- | --- |
| Cloudflare Pages (static pipeline) | **Ignored** — returns `200` with the full file | `_headers` cache policy is honored (immutable assets, 86400 artifact). Without the Function below, the demo works through the full-download fallback. |
| Cloudflare Pages + bundled Function | **Honored** — `206` via `functions/artifacts/[[path]].js` | The Function slices the edge asset per ranged request. Verified live: mid-file slices byte-identical, cold query 3 requests / 121.1 KiB at 224 ms real CDN latency, warm 1.3 ms / 0 requests. The Function reads the full asset internally per request, so this is for demo-scale artifacts; large artifacts belong on natively range-capable storage. |
| jsDelivr (`cdn.jsdelivr.net/gh/...`) | **Honored** — `206` with correct bytes at any offset | Full download is byte-identical to the git object. Caveat: the `content-range` total field can report a bogus size (their compressed storage size); harmless because the reader derives all offsets from the artifact header. |
| GitHub Pages | untested | |
| S3/CloudFront | untested | |

For large artifacts where lazy reads actually matter, pick a Range-honoring
host; Cloudflare R2 with a public bucket also serves real ranges (the Pages
asset pipeline is the limitation, not Cloudflare storage).

To point the page at a different artifact, change `ARTIFACT_URL` in
`src/main.js` (and supply a matching encoder/corpus if it is not the docs set).
