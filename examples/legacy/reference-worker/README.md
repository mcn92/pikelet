# Pikelet Worker Reference Architecture

This directory contains a reference Cloudflare Worker deployment built on top of `pikelet-wasm`.

> **Note:** This example runs from the repository checkout and imports the same
> public Worker entrypoint that the package exports as `pikelet-wasm/web`. To
> use the published package instead, install `pikelet-wasm` (any published
> version: the first, 0.7.0, already carried the API this example is written
> against). See the [root README](../../../README.md#quick-start).

The Worker keeps a Pikelet index in-process when hot and persists snapshots to
Cloudflare R2. Treat it as a **snapshot-first ANN serving layer** rather than
as a durable mutable vector database.

Best fit:

- read-heavy search workloads
- modest-sized indexes that can be restored into Worker memory
- explicit import/export flows
- periodic snapshot rebuilds published to R2

Less suitable:

- high-write authoritative online mutation
- strict cross-isolate read-after-write guarantees
- relying on in-memory Worker state as the only source of truth

## Recommended deployment shape

1. build or update the index outside the Worker
2. publish a clean snapshot to R2
3. deploy the Worker in `READ_ONLY=1`
4. expose `/search` and `/health` publicly
5. keep mutation routes for admin-only workflows or local development

This runs the Worker as a snapshot-backed search endpoint at the edge.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /search` | k-NN search (`{ query: float[], k?, efSearch?, allowedIds? }`) |
| `GET /health` | Health check (public, no auth required) |
| `GET /readiness` | Readiness check with snapshot visibility (bearer auth once `API_KEY` is set) |
| `GET /stats` | Live/deleted counts, capacity, and structured memory |
| `POST /export` | Compact, then serialize index to binary blob |
| `POST /init` | Create an index (`{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }`) |
| `POST /import` | Restore from a Pikelet snapshot (`?dims=N` only needed for legacy raw snapshots) |
| `POST /add` | Insert a vector (`{ vector: float[] }`) |
| `POST /add_batch` | Insert multiple vectors (`{ vectors: float[][] }`) |
| `POST /delete` | Soft-delete by ID (`{ id: number }`) |
| `POST /compact` | Rebuild graph without deleted entries |
| `POST /reset_cache` | Drop the warm in-memory index so the next query restores from R2 |

`/health` and `/search` are public. `API_KEY`, when configured, gates
`/readiness`, `/stats`, and the admin routes. `/health` stays cheap and public:
it reports whether the current isolate has an index loaded, but it does not
trigger a restore. `/readiness` checks whether the Worker is loaded now and
uses R2 object metadata or a small range read to inspect snapshot availability
without downloading the full snapshot.

## Running locally

```bash
cd examples/legacy/reference-worker
npx wrangler dev --port 8787 --var ALLOW_INSECURE_ADMIN:1
```

`ALLOW_INSECURE_ADMIN=1` is a local-only opt-in; without it (or an `API_KEY`)
the admin routes (`/init`, `/add`, `/import`, ...) fail closed with `403`.

## Deploying

Build a snapshot outside the Worker first — see
[`build_and_export_index.js`](build_and_export_index.js) — then:

```bash
cd examples/legacy/reference-worker
wrangler r2 bucket create pikelet-indexes
# Uncomment the [[r2_buckets]] block in wrangler.toml so the Worker can
# persist and restore snapshots from the bucket.
wrangler deploy
```

See [`wrangler.toml`](wrangler.toml) for the full configuration. Key env vars:

- `API_KEY` for bearer auth
- `ALLOWED_ORIGIN` for browser CORS access
- `RATE_LIMIT_RPM` for per-IP rate limiting
- `MAX_JSON_BYTES` to cap JSON request bodies
- `MAX_ELEMENTS_LIMIT` to cap declared index capacity
- `READ_ONLY=1` to reject mutation/admin routes
- `ALLOW_INSECURE_ADMIN=1` only for local/demo deployments where you explicitly want unauthenticated admin routes

For a safer internet-facing deployment, set at least:

- `API_KEY`
- `ALLOWED_ORIGIN`
- `READ_ONLY=1`

By default, admin routes now fail closed unless `API_KEY` is set or
`ALLOW_INSECURE_ADMIN=1` is explicitly enabled. CORS also fails closed by
default: if `ALLOWED_ORIGIN` is unset, the Worker omits `Access-Control-Allow-Origin`
instead of returning `*`.

## Deployment notes

**Memory.** Workers have a 128 MB memory ceiling per isolate. Rough formula for quantized index memory:

```txt
~(dim + 8 + 7 * M) * num_vectors bytes
```

At the default `M=12` this is `(dim + 92)` bytes per vector. Examples: `30k x 256D = 10 MB`, `200k x 384D = 93 MB`. The fp32 backend uses `(4*dim + 8 + 7*M)` bytes per vector, roughly 4x more. Keep `MAX_ELEMENTS_LIMIT` aligned with this formula and your configured dimensions; the shipped `wrangler.toml` default is intentionally conservative for a 128 MB isolate.

**CPU time.** Workers paid plan allows 50ms CPU per request (free tier: 10ms). Search comfortably fits within both tiers. Heavy operations such as `/import`, `/compact`, and large `/add_batch` requests can exceed free-tier limits on larger indexes.

**Cold starts and R2 restore.** Worker isolates are not persistent. On cold start, the Worker fetches the index from R2 and deserializes it lazily on the first request. For a large index, the first request after idle will be slow.

**Persistence.** The current Worker example writes standard Pikelet snapshots to R2 and restores
from R2 on cold start. This is the durable boundary. In-memory state is only a
warm cache for the current isolate. Snapshot writes are append-only and restore
loads the latest saved snapshot, which avoids older async writes overwriting a
newer one under the same R2 key.

Configure an R2 lifecycle rule or delete superseded keys when adapting the
example for production; the reference implementation intentionally leaves
snapshot retention to the application.

Mutation routes exist for demos, local validation, and administrative flows,
but they should not be treated as the primary production write path for a
stateful vector database. Deletes are soft-deletes in the warm isolate and are
not persisted until an explicit `/compact` or `/export` compacts the index and
writes a fresh snapshot. If you need stronger mutation semantics, put an
authoritative layer elsewhere and use the Worker as the search-serving
frontend.

**Read-only mode.** When `READ_ONLY=1`, the Worker rejects `/init`, `/import`,
`/add`, `/add_batch`, `/delete`, `/compact`, `/export`, `/reset_cache`, and
`/search_debug` with `403`. This is the recommended production setting for a
public search endpoint backed by snapshots.

**Rate limiting.** Rate limiting is in-memory per isolate. Each isolate tracks its own sliding-window counter, so the effective limit across multiple isolates is approximately `limit * number_of_isolates`.
