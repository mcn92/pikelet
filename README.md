# Pikelet

**Knowledge that ships as a file.**

A `.pikelet` file is a searchable corpus in a single file. It holds the source text, a vector index and a keyword (BM25) index, the neural encoder that turns a question into a vector, a calibration model that judges how well the corpus covers a question, and the tests the pack was built to pass. Its identity is a SHA-256 digest that commits to all of it, so the file fixes not just what the corpus says but how it is queried: the model that encodes a question, the indexes that find candidates, and when the honest answer is "this corpus doesn't say." **The format makes retrieval semantics part of the versioned knowledge artifact.**

The layout is built for byte-range reads. A reader opens the file from a small prefix and then fetches only the index rows and records a query needs, verifying each against the identity, so a pack can sit on disk, S3, R2, a CDN or any static HTTP host, with no vector database, embedding API or retrieval server behind it. [FORMAT.md](FORMAT.md) describes the binary structure, the embedded encoder, the indexes, the integrity commitments, versioning, compatibility and the format's limits.

**Pikelet** is the open-source toolchain for the format: a compiler (`pikelet compile`) that builds packs from folders, websites, git history or exported chats; a reader library (`pikelet-wasm`) for browsers, Node and edge workers; and an MCP server (`pikelet mcp`) that mounts packs for LLM agents.

It is built for corpora that are static or change on a release cycle: documentation, API references, manuals, standards, legal and policy texts, research corpora, and reference packs for agents.

## Quick start

The `pikelet` CLI requires Node 22.23.4+, 24.22.1+ or 26.12.1+ (earlier releases' fetch decompresses whatever a server labels as encoded; 18.x and 20.x are end-of-life and cannot be fixed); the `pikelet-wasm` library runs on Node 18+, browsers, and Cloudflare Workers.

```bash
npx pikelet compile ./docs --out docs.pikelet
```

```text
Ingested 3 docs -> 27 chunks
Embedded 27/27 chunks with inline transformer
4/27 chunks exceeded the 512-token encoder window and were mean-pooled across windows
Validated self-recall@1 on 27 sampled chunks
Calibrated matchQuality: cross-validated AUC 1.000 separating 72 on-topic from 104 off-topic questions
Built complete .pikelet artifact with 24.64 MiB
Compiled docs.pikelet
  24.64 MiB, 27 records, identity dbe8be15c79e...
Next:
  pikelet query docs.pikelet "a question"       search it here
  pikelet mcp install --pack docs.pikelet       give it to Claude Code
```

Every `.pikelet` is at least ~25 MiB, because the query encoder ships inside it: a 5-file folder and a 500-file folder both start around that size. The first run downloads the encoder from a GitHub release and caches it.

Compiling is mostly embedding, at about two chunks a second per CPU core, plus roughly 20 to 30 seconds of calibration: 655 chunks took 83 seconds on 8 cores, so a few thousand chunks take several minutes, and longer on a 2-core machine.

Search it from the terminal:

```bash
npx pikelet query docs.pikelet "why is a pack one file"
```

```text
docs: strong: the pack covers this topic (confidence 1.00)

1. Why make knowledge a file
   why-a-file.md#why-make-knowledge-a-file
   The reasoning behind the single-artifact design. ...
```

Give it to an LLM — Claude Code, Claude Desktop, or any MCP client:

```bash
npx pikelet mcp install --client claude-code --pack ./docs.pikelet
# or --client claude-desktop; any other MCP client can run
# `npx pikelet mcp --pack ./docs.pikelet` directly
```

Or query it from code:

```js
import { openPikeletFile } from 'pikelet-wasm/complete';
const pack = await openPikeletFile('docs.pikelet');
const out = await pack.query('how do workers restore snapshots', { k: 5 });
console.log(out.matchQuality, out.results[0]?.title);
// 'strong' 'Snapshot restore'
```

`compile` also takes a live URL (`--source https://docs.example.com`), a repository's commit history (`--git ./repo`), or exported chats (`--conversations <file|folder>`). To install instead of using `npx`, run `npm install -g pikelet`. `create` builds a deployed search app (a Worker and UI) rather than a file. [QUICKSTART.md](QUICKSTART.md) walks the whole lifecycle; [`packages/pikelet/README.md`](packages/pikelet/README.md) is the full CLI reference.

## The shortest demonstration

A real pack holds **456,153 Simple English Wikipedia records** in a **648.5 MiB** file. The [live browser demo](https://pikelet-wiki-playground.pages.dev/) is static HTML on Cloudflare Pages that mounts it straight from an R2 bucket — there is no `/search`, `/embed`, or `/api` behind it. The browser range-reads the file, verifies it, loads the query encoder from inside it, and shows every byte range it fetches.

For the agent test, the same file was served by a server whose only relevant capability was `fs.createReadStream` plus HTTP Range, and mounted by URL into a headless Claude restricted to the pack's MCP tools (no web, no filesystem). The model answered questions such as "What causes earthquakes?" by searching, reading several records, and citing them.

| In one persistent session                  | Transferred                 |
| ------------------------------------------ | --------------------------: |
| Mount (once)                               | 51.7 MiB in 11 requests     |
| Query encoder (once, with the first query) | ~25 MiB                     |
| Each fresh query after that                | ~0.5–1.1 MiB                |
| Whole session: 5 queries and a repeat      | 80.8 MiB, 12.5% of the file |

The complete artifact is never downloaded. The transcript and per-query table are in [the Wikipedia demonstration](docs/wikipedia-demonstration.md); [`benchmarks/range-proof/`](benchmarks/range-proof/) reproduces it.

## Packs for agents

`pikelet mcp` serves one or more packs over the Model Context Protocol, with four tools: `search`, `get_record`, `list_packs`, and `verify_pack`, which replays the tests stored inside the pack. If one pack fails to mount (an unreachable host, a refused pin), the others still serve and `list_packs` names the missing one and why; one on a slow host does not hold the others up past a 2 s grace.

A pack mounted with a content hash — `https://example.com/docs.pikelet#<sha256>` — has a stable identity, so "what body of knowledge did this agent query?" has a reproducible answer. A citation is `(pack identity, record id)`, and it survives updates: a chain of layers resolves citations made against its earlier versions, and a compacted pack translates them forward or reports that the record was deleted. Citation translation is a library call today (`citation(identity, id)`); the MCP tools do not expose it yet.

**A mounted pack's content reaches the model as tool output.** `verify_pack` reports which of the pack's bytes have been checked against its identity (with `full: true` it also checks a large pack's lexical index whole) and replays the pack's own tests; it does not prove the corpus is trustworthy. Mounting a pack from a source you don't control is the same trust decision as giving an agent any other untrusted-content tool — treat pack text like search results or fetched web pages, not as instructions.

**A journal the agent can write (experimental).** Packs are read-only to the model, with one opt-in exception: `--journal <name>` mounts a private pack the agent appends notes to — handoffs, decisions, dead ends, measurements — through `write_note` and `latest`, so a later session, from any vendor, can pick up where the work stands. Notes come back labelled agent-written and unverified. See [the CLI README](packages/pikelet/README.md#a-journal-the-agent-can-write-experimental) and [`docs/agent-journal.md`](docs/agent-journal.md).

**Try it.** The three packs from the [Veyra ablation](docs/veyra-ablation.md) are pinned GitHub release assets. They derive from one small synthetic corpus with controlled differences in a single fact: ask where the Tovash project is housed, and the full pack supports Chamber 17, the modified pack Chamber 43, and the ablated pack no chamber at all.

```bash
npm run demo:veyra   # fetch and verify the packs
claude --mcp-config examples/one-file-search/web/public/veyra.mcp.json
```

## Updating a pack

A pack doesn't need recompiling when a few documents change. `append` writes a small new `.pikelet` — a **layer** — holding new records, deletions, or replacements, and never touches its parent. The base plus its layers is searched as one pack, and every published file stays byte-for-byte immutable.

```bash
npx pikelet append --parent docs.pikelet --source ./changed-docs --out docs.0001.pikelet
npx pikelet append --parent docs.0001.pikelet --remove 412 --out docs.0002.pikelet
npx pikelet mcp --pack docs.0002.pikelet
```

On a 20-record test pack with a 24.6 MiB base, deleting a record took a 1.2 KiB layer and replacing one took 6.9 KiB. Deleting hides a record from search but does not erase it: the older files still hold its bytes, and `get_record` still serves it by id, marked `tombstoned`, so old citations resolve. `compact` writes a new base without it. Mounting the newest layer finds and verifies the rest of the chain, locally or over HTTP, and `#<sha256>` on it pins the whole history. Layers inherit the base's calibration until the chain drifts past 20% of the base's records; then it serves `unscored` until `compact` folds it into a fresh base. `append` refuses to write a layer that would cross that line unless given `--allow-drift`. The format is specified in [`LAYERED_PROFILE.md`](LAYERED_PROFILE.md).

## Match quality

A nearest neighbour is not automatically evidence that a corpus covers a question. At build time Pikelet fits a classifier over retrieval signals against queries the corpus does not cover, and every result carries `matchQuality: strong | weak | none`. It measures topic relevance: `strong` means the pack covers the question's topic, not that a passage states the fact asked for. For that, each result lists the question's words no top passage contains (`grounding.uncovered`). When calibration can't separate on-topic from off-topic queries, the pack reports `unscored` rather than manufacturing confidence. [How abstention is decided](docs/how-a-query-runs.md#how-abstention-is-decided) covers the details, including how the library and MCP differ in what they return under `none`.

`matchQuality` is evidence about retrieval support. It is **not** a guarantee that an LLM will never hallucinate.

## When it fits, and when it doesn't

Pikelet targets `build → publish → query many times → replace on release`, with small layered updates in between. Use it when you want search without operating search infrastructure, search inside a browser, edge worker, or offline tool, a corpus you can hand to someone else, or an agent working against a frozen, versioned body of knowledge.

Use a database instead if your corpus has heavy online writes, needs transactional mutation, serves many tenants, or already lives comfortably in one. The remote scan is linear in row count, so the design targets corpora through the low millions of records, not web scale. Text handling is English-first: keyword search keeps only ASCII letters and digits, so accented words are split into fragments and non-Latin words are dropped (see [Limitations](packages/pikelet/README.md#limitations)).

It is also not new retrieval research. HNSW is HNSW, BM25 is BM25, HTTP Range is HTTP Range, and the bundled MiniLM-L6 encoder was chosen for being small enough to live inside the file — ArguAna shows a real quality cost from that. The project is about arranging those pieces around one constraint: **the knowledge base itself must be distributable as a file and stay useful without retrieval infrastructure.**

## Deeper dives

- [How a `.pikelet` query runs](docs/how-a-query-runs.md) — what is inside the file, what a query costs on the wire, integrity checks on the read path, and abstention.
- [The Wikipedia demonstration](docs/wikipedia-demonstration.md) — the agent transcript, the network table, and what cold processes cost.
- [The Veyra ablation](docs/veyra-ablation.md) — what happens to answers when the supporting record is edited or removed.
- [Why make knowledge a file](docs/why-a-file.md) — the reasoning behind the single-artifact design.
- [Architecture](docs/architecture.md) — the engine, the C ABI, and the JavaScript wrapper; byte-level formats are in [`spec/`](spec/).

## Repository map

```text
packages/pikelet-wasm/   The pikelet-wasm library: C++ HNSW engine compiled to WASM
                         (used at compile time), readers and builders for .pikelet
packages/pikelet/        The pikelet CLI: compiler, MCP server, Docusaurus plugin
docs/                    Deeper dives, design notes, measurement reports
spec/                    Byte-level artifact contracts
LAYERED_PROFILE.md       The layered profile: append, compact, rebase
benchmarks/beir/         BEIR ablation harness (encoder, quantization, HNSW quality)
benchmarks/range-proof/  Static-HTTP proof against a server that knows nothing of Pikelet
examples/                Runnable examples: MCP packs, one-file search, static wiki pack
packs/                   Pack hosting and distribution examples
test/                    Conformance suites (`npm test`)
```

## Reproducing the proofs

The **range proof** uses a server that does not understand Pikelet and reports mount and per-query bytes and requests; `failure-modes.mjs` exercises Range ignored, tampered bytes, and truncated responses, and `mcp-proof.mjs` and `llm-proof.mjs` use a real MCP client and a real headless Claude. The **BEIR evaluation** keeps encoder, quantization, and index stages separate so a quality change can be attributed to the component that caused it. Read its quality deltas, not its latency columns, which measure the benchmark paths under that harness.

## Status

Pikelet is early: the implementation is real, the format is not frozen, and there is one primary author. Prebuilt WASM, local and HTTP readers, the embedded encoder, BM25 hybrid retrieval, content identity with lazy-read verification, calibrated abstention, MCP mounting, layered updates, and the experimental journal all ship today. `npm test` runs the engine, artifact-profile, MCP, ingestion, and format-hardening conformance suites.

Previously known as **Pancake** (renamed September 2026); files from before the rename remain readable — see [`docs/history.md`](docs/history.md). Pikelet is unrelated to the Pikelet programming language.

## License

Apache-2.0. See [LICENSE](LICENSE).

Every `.pikelet` carries its query encoder, so a pack redistributes the all-MiniLM-L6-v2 weights (quantized) and the BERT WordPiece vocabulary, both Apache-2.0. [NOTICE](NOTICE) credits them; keep it with packs you publish.
