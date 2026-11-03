# Search Artifact Walkthrough

## One-line Demo

```bash
npm run demo:artifact
```

It opens the committed docs artifact (`pikelet-docs.pikelet-range`, so it
runs from a fresh clone). Expected output shape:

```txt
Search Artifact demo: 208 vectors, 384D
resident router: 15 records (0.0 MiB)
lazy reads: 161 requests, 0.12 MiB fetched across 10 queries
per query: 16.1 requests, 0.012 MiB, 15.3 rounds mean
sample top-3 ids: 90, 204, 181
```

The rest of this walkthrough describes the SIFT1M artifact, which is not
committed; pass `--artifact <file.pikelet-range>` to run the demo against it.
At that scale the same demo printed 1,000,000 vectors (128D), an 83,254-record
(38.9 MiB) resident router, and 268.9 requests / 0.130 MiB / 26.6 rounds per
query.

## What It Demonstrates

Pikelet Search Artifacts are compiled, immutable vector-search artifacts that
can be searched without loading the full index into memory.

The current v2 artifact separates the graph into:

- a resident router segment: all upper-layer HNSW nodes
- a lazy base segment: range-addressable level-0 records

On SIFT1M the demo keeps only the router resident, and fetches
base-layer records on demand through the same `read(offset, length)` interface
that an HTTP Range or Cloudflare R2 deployment would use.

## Correctness

The split artifact has already been checked against the original Pikelet
snapshot traversal:

- dataset: SIFT1M
- queries: `1000`
- `k`: `10`
- `efSearch`: `10`
- exact top-10 matches: `1000 / 1000`

This is exact traversal parity against the source snapshot, not a separate ANN
quality estimate.

## Range-read Shape

On the 1000-query warm stream:

- resident router: `83,254` records, about `38.9 MiB`
- lazy base reads: mean `189.9` requests/query
- lazy bytes: mean `0.0917 MiB`/query
- lazy working set after 1000 queries: `91.65 MiB`
- combined router plus lazy base working set: about `130.6 MiB`

Latency model at 100 MiB/s, 32-way request parallelism:

- 2ms fixed request cost: p95 `38.1ms`, p99 `44.2ms`
- 5ms fixed request cost: p95 `95.1ms`, p99 `110.2ms`
- 10ms fixed request cost: p95 `190.1ms`, p99 `220.2ms`

## Deployment Boundary

The runtime does not require a local file. It requires only:

```js
{
  async read(offset, length) {
    return bytes;
  }
}
```

Minimal HTTP and R2 adapters are in [`range_sources.js`](range_sources.js).

That is the important architectural point: search deployment is decoupled from
full in-memory index residency. The resident part is a small compiled router;
the large base layer is immutable and range-addressable.
