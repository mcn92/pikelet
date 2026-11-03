# MCP knowledge pack

The smallest full loop from source text to an LLM agent citing it: compile a
`.pikelet`, mount it over MCP, and query it — including a query whose premise
is false, to show what `matchQuality` does and does not tell you.

The source is Jonathan Swift's *A Modest Proposal* (1729, public domain),
split into five titled sections under `sources/modest-proposal/`. It's a
useful demo text on purpose: it's satire, so a query that takes its literal
proposal at face value is a built-in false-premise test.

## Build

```bash
./build.sh
```

Runs `pikelet compile` and writes `modest-proposal.pikelet` (~25 MB — almost
entirely the bundled query encoder; the 21 text records are a few KB).
`compile` fits the pack's relevance calibration from the corpus itself, so
even a pack this small reports `matchQuality` as strong/weak/none, and it
stores 24 golden queries that `verify_pack` replays. The manifest identity
changes on every rebuild (nothing pins it here); for a real pack you'd
publish once and cite `#<sha256>`.

## Query it over MCP

```bash
node query.mjs
```

`mcp_client.mjs` is a ~70-line MCP client (JSON-RPC over the server's
stdio) that spawns `pikelet mcp --pack modest-proposal.pikelet` directly —
the same protocol any real MCP client (Claude Code, Claude Desktop) speaks.
`query.mjs` calls `list_packs`, runs two searches, then `verify_pack`.

The two searches:

- **Straightforward** — "What does the author propose doing with poor
  children?" Retrieval surfaces the proposal itself, `strong`.
- **False premise** — "What farming techniques does Swift recommend for
  raising healthier children to eat?" This assumes the essay is a sincere
  farming manual. It isn't — Swift's point is the opposite of the literal
  proposal. Retrieval still returns passages that describe the proposal in
  concrete, confident-sounding detail, because that content is genuinely
  in the corpus, and `matchQuality` comes back `weak`, not `none`: the
  pack does cover the topic. **Distance and matchQuality tell you the pack has
  relevant-looking text. They do not tell you the premise is true.** Only
  reading the passage does. That's the thing an agent citing a pack has to
  do — inspect the evidence, not just trust a high-ranked result.

## Attaching it the way an agent would

```bash
npx pikelet mcp install --client claude-code --pack ./modest-proposal.pikelet
```

writes `.mcp.json` so any MCP client gets the same four tools `query.mjs`
calls directly: `search`, `list_packs`, `get_record`, `verify_pack`.

## What this is not

This is a two-query smoke test, not a benchmark or an evaluation suite. It
doesn't do multi-turn follow-up search, cross-pack synthesis, or a staged
adversarial query set — if you need that scale of methodology, see
`packages/pikelet/README.md`'s MCP section for the full tool reference and design.
