# The Wikipedia demonstration

The full record of the demonstration summarized in the root README: a
456,153-record pack served by a server that only knows how to answer HTTP
Range requests, queried by a model through MCP. The reproduction scripts are
in [`benchmarks/range-proof/`](../benchmarks/range-proof/).

A real Pikelet pack contains **456,153 Simple English Wikipedia records** in a **648.5 MiB** artifact.

Try the live browser demo:

```text
https://pikelet-wiki-playground.pages.dev/
```

The page is static HTML/JS on Cloudflare Pages. It mounts the pack from R2:

```text
https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev/wikipedia.pikelet
```

There is no `/search`, `/embed`, `/query`, or `/api` backend behind the demo.
The browser opens the `.pikelet` file over HTTP Range, verifies the artifact,
loads the embedded query encoder, and shows the byte ranges it reads.

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

The model may already contain some of these facts in its pretrained parameters. This test demonstrates the retrieval, synthesis, citation, and deployment path; the [Veyra ablation](veyra-ablation.md) tests whether support changes when evidence is removed from the pack.

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

The headless-Claude test above used a fresh process per question, so each invocation repaid the ~52 MiB mount and ~25 MiB encoder cost — about 78 MiB per cold query. When this was measured that was a real operational distinction: persistent sessions amortized mount and encoder cost, and independent cold processes did not. Since 0.13.0, `pikelet mcp` caches those two regions on disk under their sha256 digests, so a later process mounting the same pack reads them locally: measured against the R2-hosted copy, a fresh process went from 80.1 MiB to 9.1 MiB before its first answer. `--no-cache` reproduces the cold behavior above.

*(The large benchmark fixture still carries its historical `.pancake` filename from before the Pikelet rename; current artifacts use `.pikelet`.)*
