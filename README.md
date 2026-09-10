# Pikelet

**Knowledge that ships as a file.**

Pikelet compiles a corpus into one self-contained, queryable artifact.

A model can interrogate a 456,153-record knowledge base whose backend is a static file.

A `.pikelet` can carry the source text, semantic index, keyword index, query encoder, integrity commitments, retrieval calibration, and evaluation fixtures needed to interrogate that corpus. Put the file on disk, S3, R2, a CDN, or any static HTTP host. A reader can mount it locally or over HTTP Range and search it without a vector database, embedding API, or retrieval server.

```bash
npx pikelet compile --source ./docs --out docs.pikelet
```

Then hand the resulting file to an application—or mount it directly into an LLM through MCP.

```text
documents
    │
    ▼
┌──────────────────┐
│   docs.pikelet   │
│                  │
│ corpus           │
│ semantic index   │
│ lexical index    │
│ query encoder    │
│ integrity        │
│ calibration      │
│ evaluation       │
└──────────────────┘
    │
    ├── local file
    └── static HTTP / object storage
              │
              ▼
        browser / Node / edge / agent
```

Search is the interface. **The file is the knowledge deployment unit.**

---

## The shortest demonstration

A real Pikelet pack contains **456,153 Simple English Wikipedia records** in a **648.5 MiB** artifact.

That file was served by a deliberately dumb HTTP server whose only relevant capability was:

```text
fs.createReadStream + HTTP Range
```

There was no vector database, no search service, no embedding service, no application server that understood the pack, no index process on the server.

The pack was mounted by URL and exposed to a real headless Claude process through MCP. Each invocation was restricted with `--strict-mcp-config` to only:

```text
mcp__wiki-pack__search
mcp__wiki-pack__list_packs
mcp__wiki-pack__get_record
```

No web tool. No filesystem tool. No alternate retrieval tool.

One of the actual responses:

> **Q: What causes earthquakes?**
>
> Earthquakes are caused by the movement of tectonic plates in the Earth's crust. Stress builds up along fault boundaries where plates grind against or ride over one another; when that stress exceeds the friction holding the locked fault in place, it suddenly releases as shock waves (seismic waves) that shake the ground [record 429637].
>
> There are three main fault types that can produce quakes — normal, reverse (thrust), and strike-slip — and some quakes have less clear-cut causes, like deep shifts in ancient rock or mine-collapse "tremors" [record 429638].
>
> Two related mechanisms compound this:
>
> * Subduction, where one plate is forced under another into the mantle, produces the most powerful earthquakes and can trigger tsunamis [record 132833].
> * More broadly, the slow convection-driven movement of crustal plates over the hot mantle is the underlying driver of quakes, volcanoes, and mountain formation [record 132267].
>
> Volcanic eruptions can also cause earthquakes, though these are localized to areas near active volcanoes [record 429637].

The model did not just repeat the first search hit. It retrieved multiple records and synthesized them into a sourced answer.

Other runs produced:

```text
Who was the first person on the moon?          → Neil Armstrong          (records 408686, 408684)
How do volcanoes form?                          → plates + subduction + hotspots
                                                                          (records 132770, 132765, 132852, 132853)
How does photosynthesis work?                   → two-phase reaction, cited (5 records: 114219, 114220, 114222, 114223, 269279)
What is the capital of France?                  → Paris                  (records 454641, 302684)
```

That is the product in one demonstration:

> **A model is interrogating a 456,153-record knowledge base whose backend is a static file.**

The model may already contain some of these facts in its pretrained parameters. This test demonstrates the retrieval, synthesis, citation, and deployment path; the Veyra ablation below tests whether support changes when evidence is removed from the pack.

**Network cost.** In a single persistent session (one mount, five queries, one repeat):

| Operation                       | Bytes transferred | Requests |
| ------------------------------- | ----------------: | -------: |
| Initial mount                   |          51.7 MiB |       11 |
| Query 1 + one-time encoder load |          25.9 MiB |       77 |
| Query 2                         |         504.4 KiB |       37 |
| Query 3                         |         499.4 KiB |       46 |
| Query 4                         |         968.9 KiB |       75 |
| Query 5                         |           1.1 MiB |      116 |
| Repeated query 1                |         238.4 KiB |        4 |
| **Total**                       |      **80.8 MiB** |  **366** |

Roughly **12.5% of the 648.5 MiB artifact** crossed the wire across that whole session. Excluding the one-time ~25 MiB encoder load, fresh-query traffic ran **~0.5–1.1 MiB per query**. The complete artifact was never downloaded.

The headless-Claude test above used a fresh process per question, so each invocation repaid the ~52 MiB mount and ~25 MiB encoder cost — about 78 MiB per cold query. That's a real operational distinction: **persistent sessions amortize mount and encoder cost; independent cold processes do not.**

*(The large benchmark fixture still carries its historical `.pancake` filename from before the Pikelet rename; current artifacts use `.pikelet`.)*

---

## What happens if the answer is removed from the file?

This is different from asking whether search returns sensible documents. To test whether Pikelet's retrieval-quality signal could form a useful evidence boundary, a synthetic corpus called **Station Veyra Registry** was built. One version contained the fact:

```text
The Tovash project is housed in Chamber 17.
```

A second pack was byte-for-byte identical except that the record containing that fact was removed.

```text
full pack:     matchQuality: strong    confidence: 0.908   → Chamber 17
ablated pack:  matchQuality: none      confidence: 0.136   → unsupported
```

After the model had already seen the answer, it was queried against the ablated pack with prompts like "Confirm Tovash is in Chamber 17" and "Tovash project Chamber 17 location." The retrieval result stayed unsupported and the model declined to confirm the location from the pack. A fresh isolated agent was then tested with neutral prompts, leading prompts, authority pressure, invitations to use general knowledge, cross-record distractors, and repeated pressure — six adversarial probes total — and continued distinguishing supported facts from the removed one in every case.

This does **not** mean Pikelet can prevent an LLM from hallucinating. It means the artifact can expose an explicit evidence boundary that a consuming model can choose to respect — and removing evidence from the artifact changed what that model was able to support from the mounted source.

The test is synthetic and intentionally narrow. It is in the repo (`local-packs/synthbench/`) so it can be reproduced rather than taken on faith.

---

## What does the compression cost?

The artifact architecture is only useful if the compact representation does not quietly destroy retrieval. The retrieval path was decomposed on BEIR rather than evaluated only end-to-end, across four configurations that isolate each transformation:

```text
A  upstream all-MiniLM-L6-v2, float32 corpus, exhaustive search
B  Pikelet embedded encoder,  float32 corpus, exhaustive search
C  Pikelet embedded encoder,  affine-u8 corpus, exhaustive search
D  Pikelet embedded encoder,  affine-u8 corpus, Pikelet HNSW
```

Corpus mapping and evaluation config were frozen before the runs. Official BEIR qrels are used; no LLM judges relevance.

| Dataset  | Config                                     |    nDCG@10 |  Recall@10 | Recall@100 | median ms/q | p95 ms/q |
| -------- | ------------------------------------------ | ---------: | ---------: | ---------: | ----------: | -------: |
| SciFact  | A — upstream float exhaustive              |     0.6451 |     0.7833 |     0.9250 |         3.6 |      4.0 |
| SciFact  | B — Pikelet encoder / float exhaustive     |     0.6512 |     0.7942 |     0.9417 |         3.8 |      4.4 |
| SciFact  | C — Pikelet encoder / affine-u8 exhaustive |     0.6500 |     0.7942 |     0.9417 |         6.9 |      8.7 |
| SciFact  | D — Pikelet encoder / affine-u8 HNSW       | **0.6500** | **0.7942** | **0.9417** |    **0.15** | **0.24** |
| NFCorpus | A — upstream float exhaustive              |     0.3159 |     0.1550 |     0.3115 |         2.6 |      3.1 |
| NFCorpus | B — Pikelet encoder / float exhaustive     |     0.3154 |     0.1511 |     0.3044 |         2.8 |      3.3 |
| NFCorpus | C — Pikelet encoder / affine-u8 exhaustive |     0.3154 |     0.1511 |     0.3036 |         2.9 |      3.6 |
| NFCorpus | D — Pikelet encoder / affine-u8 HNSW       | **0.3135** | **0.1482** | **0.3061** |    **0.16** | **0.25** |
| ArguAna  | A — upstream float exhaustive              |     0.3698 |     0.7653 |     0.9772 |         7.8 |     10.8 |
| ArguAna  | B — Pikelet encoder / float exhaustive     |     0.3506 |     0.7397 |     0.9801 |         6.1 |      7.3 |
| ArguAna  | C — Pikelet encoder / affine-u8 exhaustive |     0.3496 |     0.7368 |     0.9801 |         6.4 |      7.5 |
| ArguAna  | D — Pikelet encoder / affine-u8 HNSW       | **0.3496** | **0.7368** | **0.9808** |    **0.12** | **0.20** |

These results are more useful because they are not uniformly flattering.

**Affine-u8 storage is not the main quality cost** (B→C nDCG@10: SciFact 0.6512→0.6500, NFCorpus 0.3154→0.3154, ArguAna 0.3506→0.3496). **HNSW is similarly close to exhaustive search** at these settings (C→D: SciFact and ArguAna unchanged at reported precision; NFCorpus exposes a small approximation loss, 0.3154→0.3135).

**The largest observed loss is the embedded query encoder on ArguAna**: A→B moves nDCG@10 from 0.3698 to 0.3506, about a 5% relative reduction. Recall@100 actually improves slightly (0.9772→0.9801) — the relevant document is generally still in the candidate set; the degradation is in fine ordering near the top. That limitation isn't hidden: MiniLM is small *because* the goal is to fit the query encoder inside the artifact. It is not state of the art, and the compact implementation is not behaviorally identical to an upstream sentence-transformers runtime on every task.

The benchmark harness, quantization-conformance test, frozen configuration, and raw runs live under `bench/beir/`.

---

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

## Attach a pack to an LLM

`pikelet mcp` exposes one or more packs through the Model Context Protocol:

```bash
npx pikelet mcp install \
  --client claude-code \
  --pack https://example.com/docs.pikelet#<sha256>
```

The agent gets `search`, `get_record`, `list_packs`, `verify_pack`. A search result carries the identity of the pack and the location of the source record, so an agent can work against `product-docs.pikelet`, `rust-reference.pikelet`, `policy-2026-09.pikelet`, `customer-manual-v4.pikelet` without each publisher operating a retrieval API. The artifact can be local, private, public, behind authenticated object storage, or distributed like any other static asset.

A pack mounted with a content hash has a stable identity — `https://example.com/docs.pikelet#8d731...` — so "what body of knowledge did this agent query?" has a reproducible answer.

---

## How a remote query runs

Over a network, what makes search slow is not bytes; it's sequential round trips. Graph traversal is a chain of dependent reads — fetch node, inspect neighbors, fetch the next node, repeat — and across object storage those dependent round trips dominate. Measured against SIFT1M over real network storage, a resident scan plus one batched candidate-fetch phase beat graph traversal by about 5× at equal recall. So the remote artifact path carries no graph. The HNSW graph still exists — it's the in-memory engine, used when the whole index is local and round trips are free; that's the path BEIR Config D measures.

For the remote path:

1. **Resident tier** — a pooled sketch of every semantic row loads once when the artifact opens and stays in memory. For the 456,153-record Wikipedia pack: 648.5 MiB artifact, 51.7 MiB mount fetch. A SIMD kernel scans all of it in milliseconds.
2. **Candidate selection** — the scan picks the top `C` candidates against the query vector; the lexical BM25 index can contribute additional candidates into the same set. `C` is the only real knob: the compiler measures recall against brute force on held-out queries at build time and writes the operating point into the file, so readers don't have to guess it.
3. **Batched byte-range fetch** — full affine-u8 rows for the candidate set are fetched in one parallel round, nearby ranges coalesced, rather than a graph-dependent network walk.
4. **Full rerank** — fetched rows are scored against the float query and verified against their digest.
5. **Corpus hydration** — only the records needed for returned results are fetched, each carrying record ID, title, section, source, and artifact identity. The reader returns evidence, not anonymous vector IDs.

That architecture is why a 648.5 MiB pack can answer a fresh query while fetching about 0.5–1.1 MiB after warmup.

---

## Integrity is part of the read path

A range-readable artifact cannot hash the entire file on every open without defeating the point of range reads, so integrity is layered: the resident structural portion is verified during open; lazily fetched index rows and corpus records carry independent commitments and are checked when read. Bytes a query never touches don't have to cross the network merely to prove the bytes it *did* use were correct.

The failure-mode suite (`bench/range-proof/failure-modes.mjs`) exercises the important cases:

```text
correct Range server         → mounts, queries successfully
server ignores Range         → small files fall back to a bounded download;
                                large files refuse ("host ignores Range and the
                                file is 680029254 bytes; refusing full download" —
                                the cap is currently 64 MiB)
bytes change under a pinned identity → refused; tampered data is not silently served
truncated Range response     → fetch fails; no partial result is silently
                                interpreted as valid data
```

The point is not that static HTTP is magically reliable. The point is that a static artifact can fail in explicit, testable ways.

---

## Match quality and abstention

A nearest neighbour is not automatically evidence that a corpus answers a question. Pikelet can calibrate retrieval signals at build time (best semantic distance, distance margin, lexical coverage, retrieval agreement). When the corpus supports a reliable classifier, results carry `matchQuality: strong | weak | none`. When calibration can't separate supported from unsupported reliably — a single novel may be semantically homogeneous enough that the fit isn't trustworthy — Pikelet reports `matchQuality: unscored` and records why calibration was skipped, rather than manufacturing confidence. Raw retrieval scores remain available either way.

`matchQuality` is evidence about retrieval support. It is **not** a guarantee that an LLM will never hallucinate.

---

## What's inside a `.pikelet`

```text
manifest        format versions, segment offsets, artifact identity
semantic index  resident sketch rows, full affine-u8 rows, row commitments
corpus          source records, offsets, per-record commitments
query encoder   WordPiece vocabulary, quantized MiniLM weights, encoder declaration
lexical index   BM25 postings
calibration     supported / unsupported retrieval model
evaluation      golden queries / expected behavior
```

The artifact is immutable. Publish a new corpus by publishing a new artifact; old artifacts retain their identity.

**Encoder profiles.** Self-containment has a cost — the default profile carries ~25 MiB of MiniLM data regardless of corpus size. Three arrangements: **inline** (tokenizer + weights + runtime in the artifact — largest file, no external dependency, used by the Wikipedia demo above), **distilled/compact** (smaller corpus-specific representation, lower footprint, potentially lower quality), **host-supplied** (the artifact declares expected encoder behavior and verifies the host implementation against embedded test vectors — smallest artifact, no longer fully self-contained). The format treats the encoder as a capability, not Pikelet's identity — MiniLM is the current choice, not a permanent requirement.

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
import { openPikeletFile } from 'pikelet-wasm/complete';

const pack = await openPikeletFile('https://example.com/docs.pikelet#<sha256>');
const result = await pack.query('how do workers restore snapshots', { k: 5 });

console.log(result.matchQuality);
for (const hit of result.results) console.log(hit.title, hit.section);

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

## What this is not

**Not a claim that vector databases are obsolete.** If your corpus changes continuously, needs transactional updates, serves many tenants, or already lives comfortably in a database, use a database. Pikelet targets `build → publish → query many times → replace on release`.

**Not state-of-the-art embedding research.** The bundled model is MiniLM-L6, chosen for being small enough to live inside the artifact. ArguAna demonstrates a real quality cost from the compact encoder.

**Not "LLMs can no longer hallucinate."** The artifact exposes retrieved evidence, provenance, integrity, and an explicit support signal. Whether an agent obeys those signals is an agent behavior question.

**Not a new HNSW algorithm.** HNSW is HNSW. HTTP Range is HTTP Range. BM25 is BM25. MiniLM is MiniLM. Affine quantization is not new either. The project is about what becomes possible when those pieces are arranged around one constraint: **the knowledge base itself must be distributable as a file and remain useful without dedicated retrieval infrastructure.**

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

## When Pikelet is a good fit — and when it isn't

**Use it when:**

- your corpus is static or changes on a release cycle
- you want semantic search without operating search infrastructure
- you want search inside a browser, Node process, edge worker, Electron app, or offline tool
- you want to distribute a searchable corpus to someone else
- you want an LLM agent to query a frozen, versioned body of knowledge
- static/object storage is easier to deploy than a database
- provenance and reproducibility matter
- a content-addressed knowledge snapshot is useful

Examples: documentation, SDK/API references, technical manuals, research corpora, standards, legal texts, books, product knowledge, code/documentation snapshots, evaluation corpora, agent reference packs.

**Don't use it if:**

- your corpus has heavy online writes
- you need transactional mutation
- you need a multi-tenant authoritative search service
- you need exact nearest-neighbour search at very large scale
- your corpus is so large that a linear resident sketch scan is inappropriate
- you already operate a vector database happily and portability buys you nothing

The resident remote scan is linear in row count. The current architecture targets corpora through the low millions of records, not arbitrary web scale.

---

## Repository map

```text
src/                              C++/WASM vector engine: HNSW, float32 and
                                   affine-u8 backends, mutation, compaction,
                                   snapshot import/export
complete/, pikelet-artifact.js    Readers and builders for the complete
                                   range-readable artifact
pikelet/                          CLI, compiler, MCP server, encoder
                                   integration, higher-level tooling
spec/                             Byte-level artifact contracts
bench/beir/                       Frozen BEIR ablation harness (encoder,
                                   quantization, HNSW quality)
bench/range-proof/                The deliberately boring static-HTTP proof:
                                   dumb-server.mjs, proof.mjs, mcp-proof.mjs,
                                   llm-proof.mjs, failure-modes.mjs
examples/05-one-file-search/      Large single-artifact search and embedded
                                   encoder work
examples/06-mcp-knowledge-pack/   Compile, mount, search, and hydrate
                                   records through MCP
packs/                            Pack hosting and distribution examples
```

---

## Reproduce the proofs

**Range proof** (`bench/range-proof/`) is built around a server that does not understand Pikelet. It checks: static HTTP + Range + Pikelet client = remote semantic retrieval, without a full download. The main script reports artifact size, records, mount bytes/requests, per-query bytes/requests, cache behavior. `failure-modes.mjs` independently exercises Range supported / ignored / tampered bytes / truncated response. `mcp-proof.mjs` and `llm-proof.mjs` use a real MCP client and a real headless Claude process rather than a mocked model response.

**BEIR evaluation** (`bench/beir/`) keeps stages separate so a quality change can be attributed to the component that caused it, plus a standalone conformance fixture checking the JS benchmark path against the actual C++ affine-u8 representation. Don't read the latency columns as a universal native-performance comparison — they measure the benchmark paths under the stated harness. The quality deltas are the important part.

---

## Status

Pikelet is early. The implementation is real; the format is not frozen. One primary author. Draft 2 artifact format.

- prebuilt WASM included
- local and HTTP readers
- affine-u8 and float HNSW backends
- embedded query encoder
- BM25 hybrid retrieval
- content identity and lazy-read integrity verification
- calibration/abstention support
- MCP mounting
- BEIR retrieval evaluation
- range-read and failure-mode tests
- a large 456k-record example artifact

`npm test` exercises the engine, artifact profiles, MCP, ingestion, format hardening, and related conformance suites.

The project was renamed from **Pancake** to **Pikelet** in September 2026; some old fixture filenames and compatibility identifiers still use the previous name. `pancake-wasm` and `create-pancake-search` are deprecated compatibility pointers to the Pikelet packages. Pikelet is unrelated to the pre-existing Pikelet programming language.

---

## The experiment

> **For static and slowly changing corpora, useful semantic retrieval can be compiled into the artifact being distributed instead of operated as a separate service.**

The implementation currently demonstrates: one file, 456,153 records, 648.5 MiB, embedded query encoder, semantic + lexical retrieval, content identity, per-record integrity, MCP mounting, static HTTP hosting, ~0.5–1.1 MiB fresh-query range traffic after warmup, real multi-record LLM synthesis, no retrieval backend.

There are many reasons a database remains the right answer. Pikelet exists for the cases where the knowledge itself should be something you can **build, hash, copy, cache, publish, mount, query, and keep.**

## License

Apache-2.0. See [LICENSE](LICENSE).
