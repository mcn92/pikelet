# POC measurement kit

Everything the docs-search POC needs beyond what the example already ships:
an instrumented query harness, a cost model, and a report template. The
build/host/reader pieces are in `examples/one-file-search/` (`compile.mjs`,
`serve.mjs`, `pikelet-file-reader.mjs`); building from your own docs is
`npx pikelet compile --source <dir> --out <name>.pikelet` (or the
Docusaurus plugin in its default `complete` mode — see `docs-site/` in the
repo root for a working config).

Run from the repository root:

```bash
# smoke it on the shipped example artifact (goldens come from the file itself)
(cd examples/one-file-search && node compile.mjs)
node docs/measurements/row-commitments/harness.mjs examples/one-file-search/pikelet-docs.pikelet --k 5 --out results.json

# over HTTP range requests (real egress accounting)
(cd examples/one-file-search && node serve.mjs &)
node docs/measurements/row-commitments/harness.mjs http://127.0.0.1:8790/pikelet-docs.pikelet --k 5

# your artifact, your queries, with ground truth for recall@10
node docs/measurements/row-commitments/harness.mjs /path/to/your.pikelet --queries queries.json --k 10 --out results.json

# cost estimate from the measurements
node docs/measurements/row-commitments/cost.mjs --results results.json --queries-per-month 1000000 \
    --current-monthly 500 --integration-cost 8000
```

`queries.json` is either `["query text", ...]` or
`[{ "text": "...", "ids": [12, 40, 7] }, ...]` where `ids` are ground-truth
top-k record ids (build them brute-force over the same corpus — see
`examples/one-file-search/test-wiki.mjs` for the pattern). With no `--queries`, the harness uses
the artifact's embedded evaluation goldens and checks their match-quality
labels reproduce.

Kind-2 artifacts (declared external encoder) need `options.encodeQuery` and
are out of the harness's scope; use kind-1 (student) or kind-3 (inline
transformer) artifacts, which query self-contained.

Write up the numbers in `REPORT_TEMPLATE.md`.
