# One-file search

A search engine as a single static file. This example compiles the five
Search Artifact components — corpus, index, encoder, evaluation,
calibration — into one content-addressed `.pikelet`
(`spec/COMPLETE_PROFILE.md`), and serves natural-language queries from it
in Node and in the browser over HTTP range requests, with no backend and
no separate search service.

```bash
node compile.mjs                      # 03's assets -> pikelet-docs.pikelet (1.4 MiB)
node compile.mjs --inspect pikelet-docs.pikelet
node test-file.mjs                    # the file proves itself (goldens live inside it)
npx vite build web                    # build the browser page
node serve.mjs                        # http://127.0.0.1:8790 — search in the browser
node test-browser.mjs                 # Chromium acceptance (needs playwright chromium)
```

```js
import { openPancakeFile } from 'pikelet-wasm/complete';
const search = await openPancakeFile('pikelet-docs.pikelet');   // or a range source
const out = await search.query('how do workers restore snapshots');
// { matchQuality: 'strong', confidence: 0.94, results: [{ title, text, sourcePath, ... }] }
```

## Wiki scale

The same container compiles the Simple English Wikipedia pack (456,153
chunks, `examples/04-static-wiki-pack/data-full`) into a 512 MiB
`pancake-wiki.pancake` — the profile's first kind-2 artifact: the MiniLM
encoder is *declared* with verification vectors (contract section 4.4
mode 2) rather than embedded, and calibration is the pack's
retrieval-signal abstention model, bloom vocabulary included.

```bash
node compile-wiki.mjs                 # data-perm -> pancake-wiki.pancake
node compile.mjs --inspect pancake-wiki.pancake
node test-wiki.mjs                    # 200 eval queries vs exact ground truth
```

compile-wiki builds from `data-perm` — the pack's canonical layout
(k-means cluster-ordered rows, 192-dim 2:1-pooled sketch) — and warns
loudly if only the unpermuted `data-full` source is present: building
from it forfeits the layout and, historically, embedded a rejected
96-dim sketch geometry that cost ~12 points of recall (the 82.8% era).
At this scale the reader auto-stages the engine's SIMD scan kernel in the
background (`info().residentScan` flips to `'engine'`), cutting mean query
time from ~540 ms on the pure-JS scan to ~110 ms; results are identical
either way, and the JS scan keeps serving if staging fails.

## Inline wiki artifact

The kind-3 inline transformer artifact is published as a GitHub release asset,
not committed to git. Run one command; if the 649 MiB file is missing locally,
the test downloads it into this ignored directory, verifies the manifest
identity, and runs the self-contained query path:

```bash
node test-inline.mjs
```

Expected manifest identity:

```text
1b180adf4c6cebb2dcd5615256df6a25dac5fda8738dbbc11d60af86046f97f3
```

If the local `data-perm` eval query files are present (permuted id space,
matching the artifact), the same command also runs the full 200-query
augmented-recall sweep (gate: >= 94%). Otherwise it runs the release-artifact
smoke, provenance, identity, abstention, and embedded-evaluation checks.

## Files

- `compile.mjs` — assembles a `.pikelet` per the spec: 64 B header,
  canonical-JSON manifest (its SHA-256 is the artifact identity), segment
  table, and four segments — the index is an embedded `.pikelet-sketch`,
  the corpus is an offsets table + JSON records (one range read per
  hydration), encoder+calibration share one query-interpretation segment,
  and the evaluation segment carries the golden queries. `--inspect`
  verifies every digest.
- `pikelet-file-reader.mjs` — now a shim over the published reader at
  `pikelet-wasm/complete` (`complete/index.mjs` in this repo).
  Environment-neutral: Node opens a path, the browser passes an HTTP range
  source. Verifies the manifest identity and eager segments at open; the
  sketch tier and corpus records stay lazy. Kind-1 files use a pure-JS
  query path; kind-3 files load the reader-owned inline-transformer module
  and WASM kernels from `complete/encoder-kernels/` (rebuilt and synced by
  `encoder-spike/build-encoder.sh`).
- `web/` — the browser host: an input box over the reader, showing per-query
  range requests and bytes. `serve.mjs` is the entire hosting requirement:
  static files + `Range` support.
- `search-reader.mjs`, `demo.mjs`, `test.mjs` — the original composition
  spike over 03's six separate asset files, kept as the reference the
  one-file reader is tested against.
- `abstention.mjs` — calibration scoring shared by both readers (extracted
  from 03's worker.js).
- `../../complete/builder.mjs` (`pikelet-wasm/complete/builder`) — shared
  container assembly (canonical JSON, header + table layout, streamed
  writes, the recall-vs-C rerank sweep) used by both compilers.
- `compile-wiki.mjs` / `test-wiki.mjs` — the wiki-scale compiler and its
  acceptance gate (see "Wiki scale" above).

- `test-inline.mjs` - downloads the published kind-3 inline wiki artifact
  when needed and verifies the zero-option natural-language query path.

## What the acceptance tests establish

- `test-file.mjs` (15 checks): the golden queries come from the file's own
  evaluation segment — the artifact carries its conformance fixtures — and
  all 10 reproduce their labels; hydrated results byte-match the source
  corpus; the one-file reader returns identical results to the six-asset
  spike; a tampered segment fails its digest on read.
- `test-browser.mjs` (6 checks, real Chromium): the page opens the file
  over HTTP ranges, verifies the resident hash, answers a docs question
  with hydrated results and calibrated confidence, abstains on an
  out-of-domain query, and provably fetches by range rather than
  downloading the file.

## Numbers (docs corpus: 208 chunks, 384D)

- File: 1.40 MiB total — encoder 1.15 MiB, corpus 147 KiB, index 99 KiB,
  evaluation 4.5 KiB.
- Open: ~23 KiB resident (sketch tier + corpus offsets) plus the encoder;
  browser cold open is a handful of range requests.
- Query: sketch scan + source-tuned parallel rerank row reads + parallel
  hydration; single-digit-millisecond embeds for kind 1, encoder-dominated
  latency for kind 3.

## Still out of scope

Promotion into the published `pikelet-wasm` package, the
`pikelet` compile frontend, per-chunk range verification
(spec section 6), and large-corpus compilation (the wiki pack is the
intended flagship).
