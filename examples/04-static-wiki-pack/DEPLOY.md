# Deploying the wiki pack (Cloudflare Pages + R2)

From empty account to live site. Requires: a built pack in `data-perm/`
(see README.md), `npx wrangler login` completed, and the JS deps installed
(`ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm install`).

## 1. Bucket and assets

```bash
npx wrangler r2 bucket create pancake-wiki-pack
bash upload_r2.sh
```

The script splits `corpus.bin` into two 200 MiB-boundary parts
(`corpus.bin.p0`/`.p1`) because wrangler caps single uploads at 300 MiB;
the Pages Function stitches ranges across the split. If you change the
part size or the corpus changes size, update `SPLIT_OBJECTS` in
`web/functions/_serve-r2.js` to match (`partSize`, `total`, `parts`) —
the handler reads only the part objects, never a whole `corpus.bin`.

## 2. Pages project and deployments

```bash
npx vite build web
cd web
npx wrangler pages project create pancake-wiki-pack-demo --production-branch main
npx wrangler pages deploy
```

R2 binding and project config live in `web/wrangler.toml`.

One transport lesson is load-bearing in the page code and must not be
"simplified" away: every range read carries its range in the query string
(`?r=start-end`) in addition to the `Range` header. Chromium serializes
concurrent fetches of a single cacheable URL on its HTTP-cache entry
lock, so same-URL range reads execute one at a time regardless of
requested parallelism — measured live at 5.7–18.6 s per query without
the query-string ranges, 0.6–1.5 s with them. (A Node client shows no
such collapse — undici has no HTTP cache — which is how the cause was
isolated.) The Functions also send permissive CORS headers, which the
demo itself no longer needs but which let other sites query the pack
directly.

## 3. Pack versioning (cache correctness)

Every pack asset except `pack-manifest.json` is served with
`Cache-Control: immutable` under a URL containing `packVersion` — a
content hash of the artifact that `build_pack`'s manifest step stamps and
the page threads into its URLs at boot. The manifest itself is the
version pointer and gets `max-age=300`. Consequence: **rebuilding the
pack requires re-running the manifest-stamp (new hash), re-uploading the
changed assets and the manifest, and nothing else** — no cache purge;
clients converge within five minutes. Never overwrite pack objects with
different bytes without bumping `packVersion`.

## 4. Verify

```bash
BASE=https://pancake-wiki-pack-demo.pages.dev
VER=$(curl -s $BASE/pack/pack-manifest.json | grep -o '"packVersion": "[^"]*"' | cut -d'"' -f4)
curl -s -H "Range: bytes=0-127" -o /dev/null -w "artifact: %{http_code}\n" $BASE/pack/$VER/wiki.pikelet-sketch
# boundary read across the corpus split (expects the full 201 bytes)
curl -s -H "Range: bytes=209715100-209715300" -o /dev/null -w "boundary: %{http_code} %{size_download}B\n" $BASE/pack/$VER/corpus.bin
curl -s -H "Range: bytes=0-127" -o /dev/null -w "shard1: %{http_code}\n" https://shard1.pancake-wiki-pack-demo.pages.dev/pack/$VER/wiki.pikelet-sketch
```

Then load the site and run the golden probes in the console:
`await __probes()` should report zero failures. For the full measurement
methodology (cold/warm contexts, per-query timings), see the Playwright
snippets in the repo history or drive `window.__bench([...])` directly.
