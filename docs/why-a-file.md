# Why make knowledge a file

The reasoning behind the single-artifact design. Moved verbatim from the README on 2026-09-23.

## Why make knowledge a file?

The conventional retrieval deployment looks something like:

```text
documents → chunking → embedding service → vector database
                                          → keyword database
                                          → retrieval service → application API → agent
```

That is the correct architecture for many workloads. But it is a lot of machinery when the corpus is fundamentally a release artifact: product documentation, a manual, a source tree, a legal code, a standards corpus, a research collection, a book, a knowledge snapshot, an offline reference set. These datasets often change daily, weekly, monthly, or with releases — not hundreds of times per second.

For those workloads, Pikelet asks a different question:

> **What if retrieval could be compiled ahead of time and distributed with the corpus?**

Then deployment becomes: copy the file, cache it, pin it, put it behind a CDN, mount it by URL. The storage layer does not need to know that the file contains vectors.

It is not a replacement for a mutable vector database. It is an attempt to make **static and slowly changing knowledge deploy like any other artifact**.

---

## One decision

Most of the implementation follows from how a vector is stored. Each row is 8-bit integers plus two floats:

```text
x[d] ≈ offset + scale · q[d]        q[d] ∈ 0..255
```

That's a per-row affine map. It costs 4× less memory than float32, which is the ordinary reason to do it. The reason it runs through the whole project is that every operation search needs — dot products, sums, averages — is linear, and affine terms factor out of linear operations.

**Query against a stored row.** The query stays float32 and is never quantized.

```text
y·x = offset·Σy + scale·(y·q)
```

The inner loop multiplies floats by bytes. The two constants come in once at the end. No decompressed copy of the row ever exists.

**Stored row against stored row.** This is what graph construction needs, thousands of times per insert.

```text
x_i·x_j = D·o_i·o_j + o_i·s_j·Σq_j + o_j·s_i·Σq_i + s_i·s_j·(q_i·q_j)
```

`D` is the vector's dimensionality (384 for the bundled encoder). `q_i·q_j` is an integer dot over two byte arrays, which maps efficiently onto SIMD. `Σq` is stored per row. The HNSW graph is built and repaired entirely on compressed data.

**Pooling.** Average adjacent groups of `p` bytes and you get a shorter row. Because the mean of `offset + scale·q` over a group equals `offset + scale·mean(q)`, the shorter row keeps the *same two constants*. That's the resident sketch tier; a micro tier is the same thing done twice.

**Weights.** The bundled query encoder is a 6-layer MiniLM whose matrices are stored the same way, one (scale, offset) per 64-column block. Activations stay float32, weights stay bytes, dequantization happens inside the matmul — the same widen-and-multiply trick, a separate kernel, that scores vectors.

The consequence for the file format: a row decodes from its own bytes and its own two floats and nothing else. No codebook, no global statistics. Every row is a fixed-size byte range at a computable offset — locatable by arithmetic, fetchable on its own, hashable on its own, verifiable on the read that fetches it.

**Why not product quantization.** Product quantization would compress harder. It would also need a codebook, a training pass, table lookups in the distance kernel, and rows that mean nothing without the codebook. Four-to-one with no shared state was the better trade for a file meant to be read in pieces.

---

## Build once, publish anywhere

```bash
npx pikelet compile --source ./docs --out docs.pikelet
```

```js
import { openPikeletFile, httpRangeSource } from 'pikelet-wasm/complete';

const pack = await openPikeletFile(httpRangeSource('https://example.com/docs.pikelet'), {
  expectedIdentity: '<sha256>',
});
const result = await pack.query('how do workers restore snapshots', { k: 5 });

console.log(result.grounding?.uncovered);
for (const hit of result.results) console.log(hit.title, hit.sourcePath, hit.distance);

await pack.close();
```

The same artifact can also be opened from a local path or any custom source implementing `{ size, read(offset, length) }`.

**Or use the search engine directly**, if you already have vectors and don't need the artifact layer:

```js
import Pikelet from 'pikelet-wasm';

const index = await Pikelet.create({ dim: 384, maxElements: 100000, metric: 'cosine', quantized: true });
index.add(vector);
const results = index.search(query, 10);
const snapshot = index.export();
```

float32 HNSW, affine-u8 HNSW, insert/delete, compaction, import/export, deterministic snapshots, WASM with no native addon dependency. Pikelet is usable as a vector engine — that's just no longer the main reason the project exists.

---

## Why the file format matters

A retrieval service is identified by an endpoint. A Pikelet is identified by its contents.

- **Reproducibility** — pack hash + record ID identifies the exact evidence a model retrieved.
- **Versioning** — different releases are different artifacts (`docs-v1.pikelet`, `docs-v2.pikelet`); no silent mutation.
- **Distribution** — a pack can be mirrored by infrastructure that knows nothing about semantic search.
- **Offline use** — once local, no network or embedding service needed.
- **Customer-controlled knowledge** — hand someone the artifact instead of granting access to internal retrieval infrastructure.
- **Agent knowledge environments** — run a task against a frozen snapshot, reproduce it later against the same bytes.

> **Can knowledge become a first-class software artifact rather than something that always has to live behind a service?**

---

## The experiment

> **For static and slowly changing corpora, useful semantic retrieval can be compiled into the artifact being distributed instead of operated as a separate service.**

The implementation currently demonstrates: one file, 456,153 records, 648.5 MiB, embedded query encoder, semantic + lexical retrieval, content identity, per-record integrity, MCP mounting, static HTTP hosting, ~0.5–1.1 MiB fresh-query range traffic after warmup, real multi-record LLM synthesis, no retrieval backend.

There are many reasons a database remains the right answer. Pikelet exists for the cases where the knowledge itself should be something you can **build, hash, copy, cache, publish, mount, query, and keep.**
