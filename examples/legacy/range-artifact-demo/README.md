# Search Artifact Demo

This is the smallest demo of Pikelet's range-readable Search Artifact path.
It opens a compiled `.pikelet-range` artifact, keeps the v2 router segment
resident, and lazily materializes base-layer records through byte-range reads.

From the repository root:

```bash
npm run demo:artifact
```

A companion demo compares this profile against the sketch profile
(`.pikelet-sketch`) on the same corpus and queries, cold per query — same
byte volume, but sequential fetch depth drops from ~11 miss rounds to
exactly 1. The sketch artifact is derived from the committed range artifact
in-process, demonstrating `SKETCH_PROFILE.md`'s derivability claim:

```bash
npm run demo:sketch
```

At SIFT1M scale (same artifact as the full-JSON example above) the gap is
starker, because range-profile quality is bought with traversal depth while
sketch quality is bought with rerank width inside the same single round.
`--range-sweep` gives the graph profile its own knobs (`efSearch`,
`expansionBatch`, `gap` — the last row is the ROADMAP's settled recipe) so
the comparison is the graph's frontier against the sketch's single point:

```bash
node examples/legacy/range-artifact-demo/sketch_demo.js --compact \
  --artifact benchmark_results/layout/pancake-sift1m-u8-metis-split.pikelet-range \
  --query-file sift/sift_query.fvecs --gt-file sift/sift_groundtruth.ivecs \
  --rerank 300 --ef-search 60 --range-sweep --queries 100
```

Measured 2026-08-13, cold per query, 100 real SIFT queries scored against
the published ground truth (resident: `38.9 MiB` router vs `38.1 MiB`
sketch tier — the parity point SKETCH_PROFILE.md section 4.1 records):

| config | requests/query | sequential rounds | bytes/query | recall@10 |
| --- | ---: | ---: | ---: | ---: |
| range ef=60 | 859.6 | 61.2 | 424.2 KiB | 94.5% |
| range ef=80 | 1097.0 | 81.0 | 540.6 KiB | 95.6% |
| range ef=80 batch=8 | 1173.0 | 12.7 | 586.2 KiB | 95.8% |
| range ef=80 batch=8 gap=64K | 477.3 | 12.7 | 5171.2 KiB | 95.8% |
| sketch C=300 | 288.6 | **1.0** | 47.5 KiB | 95.9% |

The graph's knobs genuinely trade cost between axes — batching collapses 81
rounds to 12.7 by widening each round, gap trades requests for ~10x bytes —
but the whole surface bottoms out at ~13 sequential rounds, while the
sketch's single point sits at 1 round and a tenth of the bytes at equal
recall. The sketch's 95.9% at C=300 independently reproduces the
recall-vs-C point SKETCH_PROFILE.md section 4.3 pre-registers for exactly
this corpus (~96%).

Why the byte gap: each graph record here is 490 B — 18 B of framing (id,
level, per-level counts), the 128 B quantized vector, 8 B scale/offset, and
a fixed-capacity adjacency allocation of 96 B base-layer slots (`M0=24`)
plus 240 B upper-layer slots (`maxLevel=5 × M=12 × 4 B`) carried by every
record whether or not the node exists at those levels. That 362 B of
non-vector overhead is structural to the record format and independent of
`dim`: about 2.8x the vector on 128-D SIFT, but only ~24% of a 1536-D u8
row — so the bytes multiple compresses on high-dimensional corpora. The
rounds collapse does not, because dependency depth is a property of the
traversal, not the record layout.

The derive step reads every record once (a few minutes from local disk), so
this variant is a measurement recipe, not part of the quickstart.

For full JSON output:

```bash
node examples/legacy/range-artifact-demo/demo.js \
  --artifact benchmark_results/layout/pancake-sift1m-u8-metis-split.pikelet-range \
  --query-file sift/sift_query.fvecs \
  --queries 10 \
  --k 10 \
  --ef-search 10
```

Expected shape on the current SIFT1M artifact:

- artifact graph: `1,000,000` nodes, `128` dimensions
- resident router: `83,254` records, about `38.9 MiB`
- first 10-query warm stream: `2,689` cumulative lazy range requests
- first 10-query warm stream: about `1.30 MiB` cumulative lazy bytes

The demo intentionally does not load the base index into memory. It uses the
same public API a Worker or HTTP/R2 deployment would use:

```js
const artifact = await Pikelet.openRangeArtifactFile('index.pikelet-range');
const result = await artifact.search(query, 10, { efSearch: 10 });
```

For HTTP or R2, provide a range source instead of a local file. See
[`range_sources.js`](range_sources.js) for minimal adapters.

HTTP range source:

```js
const { PancakeRangeArtifact } = require('pikelet-wasm/artifact');
const { createHttpRangeSource } = require('./range_sources.js');

const source = createHttpRangeSource('https://example.com/index.pikelet-range');
const artifact = await PancakeRangeArtifact.open(source);
```

R2 range source:

```js
const source = {
  async read(offset, length) {
    const object = await env.INDEX_BUCKET.get('index.pikelet-range', {
      range: { offset, length },
    });
    if (!object) throw new Error('artifact missing');
    return new Uint8Array(await object.arrayBuffer());
  },
};

const artifact = await Pikelet.RangeArtifact.open(source);
```
