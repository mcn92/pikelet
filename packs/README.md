# The pack shelf

`packs.json` is a shelf: a static listing of `.pikelet` knowledge packs.
There is no registry service and no accounts — the registry is also just a
file. Mount everything on a shelf in one line:

```bash
npx pikelet mcp --shelf https://raw.githubusercontent.com/mcn92/pikelet/main/packs/packs.json
```

or write the config for your MCP client instead of running it by hand:

```bash
npx pikelet mcp install --shelf <shelf-url> --client claude-code
```

Each entry names a pack, where it lives (`url`, or `path` relative to the
shelf file), and optionally its immutable manifest `identity` — the sha256
every mount verifies before serving, so a shelf pins exact knowledge
states, not just locations. Packs are range-read off dumb HTTP: mounting
the 649 MiB Wikipedia pack transfers ~52 MiB and each question costs ~127
range requests; the file is never downloaded whole. `pikelet mcp` caches
the mount's digest-committed regions on disk, so a later mount of the same
pack reads most of that locally.

## Adding a pack

Compile one and host it anywhere that serves HTTP ranges — GitHub
release assets, R2, S3, any CDN; `pikelet doctor <url>`
certifies a host. Redirecting hosts are fine: the reader resolves the
redirect once and pins the target, so GitHub's rate-limited front door
is charged once per mount while range reads go straight to its CDN
(measured on the Wikipedia pack: 6 s mount, ~1 s warm queries), and
transient 429/5xx pressure is retried with backoff:

```bash
npx pikelet compile --source https://docs.example.com \
  --out example-docs.pikelet --license MIT
```

The compile output prints the manifest identity; put it in the shelf entry.
Set `--license` on anything meant for redistribution — it is recorded in
the pack manifest and surfaced by `list_packs`, and result provenance
(title, heading path, source URL) carries attribution through to answers.
