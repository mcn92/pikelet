# Pikelet — System Design

**Scope:** The full Pikelet vector-search system as built today — the C++ HNSW
backends, the WASM C ABI, the JavaScript wrapper, the native benchmarking addon,
serialization, and the Cloudflare Worker reference deployments.
**Last updated:** 2026-07-19
**Status:** Reflects the current source tree (`src/`, `pikelet-core.js`,
`native/`, `examples/legacy/reference-worker*`). This document was written from a ground-up
re-read of the code.

> Verification note: every mechanism below was read out of the source. Where a
> behavior is subtle (default divergence between layers, deletion state on
> import, ID remap on compaction), the exact code path is named so it can be
> re-checked.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Layered Architecture](#2-layered-architecture)
3. [The Two HNSW Backends](#3-the-two-hnsw-backends)
4. [Quantization and Asymmetric Distance](#4-quantization-and-asymmetric-distance)
5. [SIMD and Distance Kernels](#5-simd-and-distance-kernels)
6. [The C ABI and Handle Table](#6-the-c-abi-and-handle-table)
7. [The JavaScript Wrapper](#7-the-javascript-wrapper)
8. [ID Translation and Compaction](#8-id-translation-and-compaction)
9. [Serialization Formats](#9-serialization-formats)
10. [Build Pipeline (WASM and Native)](#10-build-pipeline-wasm-and-native)
11. [Cloudflare Worker Deployment](#11-cloudflare-worker-deployment)
12. [Configuration Reference](#12-configuration-reference)
13. [Invariants](#13-invariants)
14. [Limitations and Non-Goals](#14-limitations-and-non-goals)
15. [Appendix A: WASM Export Inventory](#appendix-a-wasm-export-inventory)
16. [Appendix B: Serialization Byte Layouts](#appendix-b-serialization-byte-layouts)

---

## 1. Overview

Pikelet is an HNSW (Hierarchical Navigable Small World) approximate
nearest-neighbor index compiled from C++ to WebAssembly. The primary artifact is
a single portable WASM module (`dist/engine.wasm`, ~141 KB raw / ~50 KB gzipped,
plus a ~17 KB `engine.js` loader / ~5 KB gzipped) that runs unchanged in Node.js,
browsers, and Cloudflare Workers — no native dependency on the default path.

The engine provides two interchangeable HNSW backends behind one API:

- **`FloatHNSW`** — full-precision float32 storage and distances.
- **`Uint8FloatHNSW`** — row-wise uint8 quantized storage with *asymmetric* search
  (float32 query against uint8 database), roughly 3.5× smaller than float32 at
  1536D with a modest recall-ceiling cost.

A separate native Node N-API addon (`native/`) compiles the *same* two C++
backend classes with AVX-512/AVX2/SSE2 SIMD. It is a benchmarking tool — it is not part
of the shipped package — and exists to separate runtime overhead (WASM vs native)
from graph quality (which is identical because the backend code is shared).

Pikelet is an index, not a database and not an embedding model. It stores
vectors and graph edges only; metadata, persistence policy, and embedding are
the caller's responsibility.

---

## 2. Layered Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Application                                                       │
│   Node.js app   │   Cloudflare Worker   │   Browser (ESM import)   │
├─────────┬───────┴───────────┬───────────┴──────────┬──────────────┤
│  JS Wrapper Layer           │                      │              │
│   pikelet.js / pikelet.node.mjs / pikelet.web.mjs / pikelet.workerd.mjs │
│            └──────────── pikelet-core.js ──────────┘              │
│            PancakeIndex: marshalling, ID translation,             │
│            export envelope, buffer management                     │
├──────────────────────────────┬────────────────────────────────────┤
│  C ABI (WASM exports)         │                                    │
│   src/engine.cpp — handle table, IndexWrapper dispatch            │
│   _pancake_init / _add / _query / _query_filtered / _export / ... │
├──────────────────────────────┬────────────────────────────────────┤
│  Backend Layer (C++ templates-free, runtime dimension)            │
│   ┌──────────────────┐   ┌────────────────────────────────────┐   │
│   │ FloatHNSW        │   │ Uint8FloatHNSW                      │   │
│   │ float_hnsw.hpp   │   │ uint8_float_hnsw.hpp               │   │
│   └──────────────────┘   └────────────────────────────────────┘   │
│   Distance kernels: WASM SIMD128 / AVX-512 / AVX2 / SSE2 / scalar │
└──────────────────────────────────────────────────────────────────┘
```

One sibling of `pikelet-core.js` is deliberately absent from the diagram
above: `pikelet-artifact.js`, the Search Artifact layer. It implements the
sketch (`.pikelet-sketch`) and deprecated range-readable (`.pikelet-range`)
readers and builders in pure JS on top of a `read(offset, length)` source
abstraction —
it uses the WASM engine only optionally (the sketch scan kernel). Its
behavioral contract is `spec/SEARCH_ARTIFACT_CONTRACT.md`, its sketch byte
layout `spec/SKETCH_PROFILE.md`, and every entrypoint bundles it
(`Pikelet.RangeArtifact` / `Pikelet.SketchArtifact`, backed by the
frozen `PancakeRangeArtifact` / `PancakeSketchArtifact` classes). This document covers
the engine below that line; the artifact layer is specified in `spec/`.

The native addon (`native/pancake_napi.cpp`) replaces the *C ABI* layer with an
N-API binding but reuses the identical backend layer — it `#include`s
`float_hnsw.hpp` and `uint8_float_hnsw.hpp` directly.

The same `IndexWrapper` abstraction (`src/engine.cpp:38`) is used by both the
WASM C ABI and the native addon (the native addon defines an equivalent wrapper
pair in `pancake_napi.cpp`). Backend choice is made once, at construction, from
the `quantized` flag, and dispatched through virtual calls thereafter — there is
no per-call branching on a backend type.

---

## 3. The Two HNSW Backends

Both backends are runtime-dimension (no compile-time `DIMS` template
specialization) and implement the same HNSW algorithm with backend-specific
storage and distance kernels. The graph algorithm code (level assignment, greedy
descent, layer beam search, neighbor selection, compaction) is **largely
duplicated** between `float_hnsw.hpp` (~1190 lines) and `uint8_float_hnsw.hpp`
(~1840 lines) rather than shared through a common base — a deliberate trade of
DRY-ness for keeping each backend's hot path self-contained and independently
tunable.

### 3.1 Graph parameters

| Parameter | Meaning | Notes |
|-----------|---------|-------|
| `M` | Max neighbors per node on upper layers | |
| `M0` | Max neighbors at layer 0 | Hard-coded `M0 = 2 * M`; not separately configurable. Layer 0 is denser because all searches terminate there. |
| `ef_construction` | Beam width during insert | |
| `ef_search` | Beam width during query | Passed per query; the JS wrapper owns the mutable default. |

### 3.2 Level assignment

Identical in both backends: an exponential distribution
`level = floor(-ln(U) * level_mult)` where `level_mult = 1 / ln(M)` and `U` is
uniform (0,1) from a seeded `std::mt19937`. (`float_hnsw.hpp:115`,
`uint8_float_hnsw.hpp:244`.)

### 3.3 Insert

1. Capacity check — returns `UINT32_MAX` if `count == max_elements` (no
   resize). (`float_hnsw.hpp:120`, `uint8_float_hnsw.hpp:194`.)
2. Assign sequential internal id `count_++`.
3. Store the vector (float32 stored raw, or normalized-then-quantized for the
   uint8 cosine path — see §4).
4. Greedy descent from the entry point through the upper layers to the new
   node's top level (exhaustive per-layer neighbor scan, no beam).
5. For each layer from the node's level down to 0: a beam search of width
   `ef_construction` collects candidates; the neighbor-selection heuristic picks
   up to `M` (or `M0` at layer 0); reciprocal edges are added to neighbors with
   pruning.
6. Update the entry point if the new node's level exceeds the current max.

### 3.4 Neighbor-selection heuristic

Both backends default to `use_heuristic = true` and implement the HNSW
diversity heuristic with **backfill** (the paper's "keep pruned connections"):
a candidate is kept only if it is not closer to an already-selected neighbor than
to the inserting node; if fewer than `M` survive, the closest rejected
candidates are added back until `M` slots are filled. (`float_hnsw.hpp:1055`,
`uint8_float_hnsw.hpp:1716`.) The float backend caches pairwise candidate
distances during selection to avoid recomputing them; the uint8 backend leans on
its closed-form symmetric distance (§4.3) instead.

**Per-node slot capacity is hard.** Layer-0 neighbor slots are pre-allocated at
exactly `M0` entries per node with no headroom between adjacent slots, so adding
a reciprocal edge to a node already holding `M0` neighbors must not append past
`M0`. When a node is saturated, `append_neighbor` runs the diversity heuristic
over the existing neighbors plus the new candidate and re-selects within `M0`
rather than writing a transient `M0+1`th entry (which would corrupt the next
node's slot, or overrun the buffer on the last node). The uint8 backend's
`append_edge_with_prune` follows the same rule. (`float_hnsw.hpp:853`,
`uint8_float_hnsw.hpp:1101`.)

### 3.5 Delete and compaction

- **Soft delete** sets a per-node flag in a `std::vector<uint8_t> deleted_`
  (one byte per node, not a packed bitset — a deliberate trade of memory for a
  branch-free check). The node stays in the graph and is skipped during
  traversal. (`float_hnsw.hpp:376`, `uint8_float_hnsw.hpp:478`.)
- **Compaction** has two topology strategies behind one stable-remap contract.
  Below 50% deletions it physically moves survivors into compacted positions,
  remaps every neighbor id, and backfills under-connected nodes through
  neighbors-of-neighbors. At 50% or more deletions, it rebuilds the HNSW graph
  from the live vectors in old-ID order; repairing a mostly hollow skeleton did
  not recover recall reliably at scale. The uint8 rebuild compacts its quantized
  rows first and releases the old topology before reinsertion so the WASM
  allocator can reuse those blocks without increasing the retained heap.
  Deletion state is reset and the `old_id → new_id` map is returned through an
  `out_map` out-parameter. (`float_hnsw.hpp:390`,
  `uint8_float_hnsw.hpp:503`.)

The C ABI exposes both a void `compact()` and a `compact_remap()` that surfaces
the `out_map`; the JS layer consumes `compact_remap()`'s out-map to rebuild its
stable external-id maps (§8).

### 3.6 Filtered search

Both backends implement `search_filtered(query, k, filter_bitset, bitset_len, ef_search)`.
The filter is a bitset over **internal** ids: bit `(id & 7)` of byte `(id >> 3)`.
Filtering happens *inside* the layer-0 traversal, not as a post-filter:
non-matching nodes still navigate the graph (they remain in the candidate queue),
but only matching nodes enter the result heap. The search starts with
`ef = max(ef_search, k*2)` and **dynamically widens** ef (up to ~4× initial) when
too few filtered results have been found, so that restrictive filters still
return `k` results where the graph allows. (`float_hnsw.hpp:237`,
`uint8_float_hnsw.hpp:362`.) Effectiveness degrades below ~1% selectivity, where
the graph may lack navigable paths to the target set.

### 3.7 The visited-set generation trick

Rather than clearing a visited bitmap before each search, both backends keep a
`std::vector<uint32_t> visited_list_` and a monotonically increasing
`visited_curr_` counter; a node is "visited this search" iff
`visited_list_[id] == visited_curr_`. Each search just increments the counter;
the array is only zeroed on the rare `uint32_t` wraparound. This avoids an
`O(max_elements)` clear per query. (`float_hnsw.hpp:116`,
`uint8_float_hnsw.hpp:189`.)

---

## 4. Quantization and Asymmetric Distance

### 4.1 Row-wise affine quantization (uint8 backend)

Each vector is quantized independently to **uint8** using its own min/max
(`uint8_float_hnsw.hpp:213`):

```
vmin = min(v),  vmax = max(v)
range = vmax - vmin
if (range < 1e-30) range = 1.0      // degenerate-vector guard (constant vector)
scale = range / 255
q[d]  = clamp(round((v[d] - vmin) / scale), 0, 255)   // round via +0.5
// stored per vector: scale (f32), offset = vmin (f32), q[0..D-1] (uint8)
```

Dequantization is `v[d] ≈ offset + scale * q[d]` (`uint8_float_hnsw.hpp:1195`).

**Per-vector storage:** `D` bytes (quantized data) + 4 (scale) + 4 (offset) +
4 (`sum_q`) + 4 (`sum_q2`) = **D + 16 bytes**, versus `4D` for float32. The two
extra `uint32` sums are precomputed statistics used by the symmetric distance
(§4.3). (Fields: `uint8_float_hnsw.hpp:1815–1819`.)

### 4.2 Asymmetric search distance

At query time the **query stays float32**; the database vector is dequantized on
the fly during the distance computation. This avoids quantizing the query (which
would compound error and require knowing the query's scale before the distance is
computed) and preserves query-side precision. For cosine, both the stored vectors
(at insert) and the query (at search) are L2-normalized first; distance is
`1 − clamp(dot, −1, 1)`. (`uint8_float_hnsw.hpp:199`, `:1204`.)

### 4.3 Closed-form symmetric distance

Graph maintenance (neighbor selection, edge-distance recomputation) needs
node-to-node distances between two *stored* uint8 vectors. Instead of
dequantizing both, the uint8 backend computes the distance algebraically from the
quantized bytes and the cached `sum_q` / `sum_q2` statistics plus an integer
`uint8_dot` (`uint8_float_hnsw.hpp:1525`):

```
L2(a,b) = D·(oa−ob)² + 2(oa−ob)(sa·sum_q[a] − sb·sum_q[b])
        + sa²·sum_q2[a] + sb²·sum_q2[b] − 2·sa·sb·dot(a,b)
```

This makes graph construction distance evaluations cheap and is the main reason
the uint8 build path stays competitive with float32.

The float backend has no analog — it computes node-to-node distance directly on
the stored float vectors.

---

## 5. SIMD and Distance Kernels

Both backends select a kernel at compile time:

```cpp
#if defined(__wasm_simd128__)                                      // WASM SIMD128 (the shipped path)
#elif defined(PIKELET_ENABLE_AVX512_SIMD) && defined(__AVX512F__)  // native, 512-bit
#elif defined(PIKELET_ENABLE_AVX2_SIMD)                            // native, 256-bit
#elif defined(PIKELET_ENABLE_SSE2_SIMD)                            // native, 128-bit
#else                                                              // scalar fallback
#endif
```

Priority is WASM SIMD128 → AVX-512 → AVX2 → SSE2 → scalar. The uint8 backend's
AVX-512 gate additionally requires `__AVX512BW__` for its byte-wide widening.
Each kernel processes the dimension in SIMD-width chunks with a scalar tail for
the remainder.

- **Float L2:** load 4/8/16 floats, subtract, square, accumulate
  (`float_hnsw.hpp:883`; `_mm512` variant at `:895`). **Float cosine:**
  dot-product accumulate, then `1 − clamp(dot)` (`float_hnsw.hpp:930`).
- **uint8 asymmetric:** load 16 uint8, widen u8→u16→f32, dequantize in-register
  (`offset + scale·q`) via FMA, subtract the float query, square/dot, accumulate.
  Multiple independent accumulators (`acc0..acc3`) hide FMA latency; the AVX-512
  path uses four independent 512-bit accumulators.
  (`uint8_float_hnsw.hpp:1204`, `:1359`.)
- **Relaxed SIMD:** an opt-in WASM build path uses `wasm_f32x4_relaxed_madd`
  (fused multiply-add) where available, falling back to separate mul+add
  otherwise (`uint8_float_hnsw.hpp:58`). Relaxed-SIMD reductions are
  non-deterministic across runtimes; this is why it is opt-in (see §10).

The native build compiles the widest kernel the host compiler exposes — AVX-512
on capable hosts (not Windows; see §10.2), otherwise AVX2/SSE2 — and is the
fastest distance path; the shipped WASM SIMD128 build is the portable one.
Recall is identical across all of them because they compute the same distances —
only throughput differs.

---

## 6. The C ABI and Handle Table

`src/engine.cpp` (361 lines) is the boundary between WASM and the C++ backends.

### 6.1 Handle table

```cpp
constexpr uint32_t MAX_HANDLES = 64;
constexpr uint32_t INVALID_HANDLE = 0xFFFFFFFF;
struct HandleSlot { IndexWrapper* index = nullptr; size_t dim = 0; };
static HandleSlot g_handles[MAX_HANDLES];
static std::vector<uint8_t> g_export_bufs[MAX_HANDLES];
```

A fixed 64-slot static array. `alloc_handle()` linear-scans for the first free
slot (or returns `INVALID_HANDLE`); `free_handle()` deletes the wrapper and
clears the slot. Multiple independent indexes can coexist in one WASM instance —
e.g., one per tenant — up to 64. State is plain mutable globals, which is safe
under WASM's single-threaded execution model. (`engine.cpp:140–165`.)

### 6.2 Backend dispatch

`IndexWrapper` (`engine.cpp:38`) is an abstract base with virtual `insert`,
`bulk_insert`, `search`, `search_filtered`, `mark_delete`, `compact` (both void
and `out_map` forms), `count`, `ghost_count`, `ghost_ratio`, `memory_bytes`,
`serialize`, `deserialize`, `dimension`. Query beam width is an explicit
argument to `search` and `search_filtered`. `pancake_init` picks
the concrete wrapper from the `quantized` flag and the metric (`metric == 1` →
cosine, else L2):

```cpp
if (quantized) g_handles[h].index = new Uint8FloatHNSWWrapper(dim, u8cfg);
else           g_handles[h].index = new FloatHNSWWrapper(dim, cfg);
```

(`src/engine.cpp` backend dispatch.)

> **Defaults.** The library defaults are `M=12`, `efConstruction=75`,
> `efSearch=100`, `maxElements=100000`, and `seed=108`. The JavaScript wrapper
> passes them explicitly. The C ABI
> only falls back to its internal defaults when a direct caller passes
> non-positive values to `pancake_init`; normal JS callers use the canonical
> library defaults.

### 6.3 Result marshalling

`pancake_query` / `pancake_query_filtered` take caller-allocated `uint64_t* ids`
and `float* dists` buffers, run the search, and copy results in — widening the
backend's internal `uint32_t` ids to `uint64_t` on the way out (the BigInt-wide
ABI is why `WASM_BIGINT=1` is set at build time). They return the result count.
(`engine.cpp:220`, `:232`.)

### 6.4 Export buffer ownership

`pancake_export` serializes into the per-handle static `g_export_bufs[h]` and
returns a pointer + size; the pointer is valid only until the next export on that
handle or `pancake_dispose`. The caller must copy promptly. `pancake_import`
returns `0` / `-1`; on failure the existing index is left intact. The wrapper's
`deserialize` builds a fresh backend into a `unique_ptr` and only swaps it in on
success, and `pancake_import` wraps the call in a `try/catch` so a hostile
snapshot that still slips an oversized allocation through the bounds checks
(below) returns `-1` instead of aborting the WASM instance. (`engine.cpp:294`,
`:301`.)

### 6.5 Utilities and lifecycle

Beyond the index API, the ABI exports `emsc_malloc`/`emsc_free` (heap for
marshalling), `pancake_profile_print`/`pancake_profile_reset` (no-ops unless built with
`PIKELET_UINT8_HNSW_BUILD_PROFILE`), and `pancake_shutdown_all` (frees all
handles). The WASM ABI no longer exposes the earlier experimental `emb_*`
embedding-model or matrix-helper paths; the public surface is the index API plus
allocation, profiling, and lifecycle helpers.

The complete export list is in [Appendix A](#appendix-a-wasm-export-inventory).

---

## 7. The JavaScript Wrapper

### 7.1 Entry points and engine loading

| Entry | File | WASM resolution |
|-------|------|-----------------|
| CJS (Node) | `pikelet.js` | reads each selected asset once, compiles once, instantiates per index |
| ESM (Node) | `pikelet.node.mjs` | same, via `node:fs` + `fileURLToPath` |
| Web | `pikelet.web.mjs` | fetches each selected asset once, compiles once, instantiates per index |
| workerd | `pikelet.workerd.mjs` | consumes the bundled precompiled module and instantiates per index |

All entrypoints share `pikelet-loader.js`, which memoizes source and compiled
module promises per SIMD/scalar variant. Concurrent `create()` calls therefore
share fetch/read/compile work but each call invokes the Emscripten factory with
a fresh `WebAssembly.Instance`. Heaps, handle tables, growth, failure, and
disposal remain isolated per index. `Pikelet.create()` then allocates per-index
scratch buffers, calls `_pancake_init`, and returns the wrapper.

### 7.2 PancakeIndex API

| Method | Marshalling |
|--------|-------------|
| `add(vec)` | validates element type (plain-array elements must be numbers, not coerced) + dim + finiteness, `HEAPF32.set` into the query buffer, `_pancake_add`, assigns an external id |
| `addBatch(vecs)` | one `emsc_malloc` for the whole batch, `_pancake_bulk_insert`, records a contiguous id range |
| `search(q,k,options?)` | resolves the per-call/default `efSearch`, clamps `k` to `count` (prevents 32-bit size wrap), `_ensureSearchCapacity(boundedK)`, marshals query, `_pancake_query`, translates ids + (for L2) `sqrt` the squared distance |
| `searchFiltered(q,k,allowedIds,options?)` | resolves `efSearch`, builds an internal-id bitset from the allowed external-id `Set`, `_pancake_query_filtered` |
| `setEfSearch(ef)` | validates and changes the JS-owned default for future queries; it does not mutate WASM engine state |
| `delete(id)` | external→internal, `_pancake_delete`, record in `_deletedExt`; returns whether a live ID changed state |
| `has(id)` / `isDeleted(id)` | inspect the stable external-ID maps without entering WASM |
| `compact()` | rebuild id maps from survivors (§8) |
| `export()` | guard `ghostCount===0`, prepend v3 envelope (§9.3) |
| `import(data)` | parse + validate the envelope **and** the embedded raw backend header (dim/metric/quantized/M/efConstruction must match; `count ≤ maxElements` or `SNAPSHOT_CAPACITY_EXCEEDED`), load WASM state, restore id maps |
| `dispose()` / `Symbol.dispose` | free handle + scratch buffers, idempotent |

The preferred state properties are `liveCount`, `deletedCount`, `deletedRatio`,
`capacity`, `remainingCapacity`, resolved `config`, and structured `memoryUsage`.
`ghostCount`, `ghostRatio`, and `memory` remain aliases. `dim`, capacity, graph
configuration, and the default `efSearch` are cached; backend counts and logical
memory proxy to WASM.

**Module-level API.** Beyond `create()`, `pikelet-core.js` exports
`fromVectors()` (bulk ingest returning `{ index, ids, idMap }`), `restore()`
(config-inferring snapshot restore), `inspectSnapshot()` (header validation
without creating a WASM instance), and `withIndex()` (scoped create/use/dispose)
(`pikelet-core.js:892–1035`). The Node entrypoints add `loadJsonFile()` and
`loadSnapshotFile()`; the web/workerd entrypoints stub the file helpers to throw
`INVALID_ARGUMENT`.

### 7.3 Buffer management

Search result buffers (`_idPtr`, `_distPtr`) are reused across queries and grown
on demand by `_ensureSearchCapacity` when `k` exceeds the current capacity
(`pikelet-core.js:639`). Result ids are read back as a `uint64` (two `uint32`
halves recombined) from the heap and translated to external ids; for L2 the
stored squared distance is `Math.sqrt`-ed before being returned
(`pikelet-core.js:773`). `dispose()` frees all three scratch pointers even if the
handle dispose throws, then sets a disposed flag that every method checks.

---

## 8. ID Translation and Compaction

The WASM backend assigns sequential internal ids (0, 1, 2, …) and **reassigns
them on compaction** (gaps from deletes are closed). To give callers stable ids,
`pikelet-core.js` keeps a bidirectional map:

- `_extToInt: Map<extId, intId>`
- `_intToExt: Map<intId, extId>`
- `_deletedExt: Set<extId>`
- `_nextExtId`: monotonic external-id counter, never reused

External ids are assigned at insert and never change. On `compact()`
(`pikelet-core.js:285`):

1. Allocate a `countBefore × u32` map buffer and call
   `_pancake_compact_remap(handle, mapPtr, countBefore)`.
2. Read back the engine's `old → new` internal-id map (`0xFFFFFFFF` marks a
   removed node).
3. Rebuild `_extToInt` / `_intToExt` from that map and clear `_deletedExt`.
   (An empty index short-circuits to plain `_pancake_compact`.)
4. Cross-check: the rebuilt mapping count must equal the engine's post-compact
   `_pancake_count`; a mismatch throws `INTERNAL_INVARIANT` and clears all
   mappings rather than serving misattributed ids.

The correctness hinge is that cross-check: the wrapper consumes the engine's
`compact_remap` out-map rather than assuming any particular renumbering rule,
then verifies the rebuilt map's size against the engine before trusting it.

---

## 9. Serialization Formats

There are two nested layers: the backend blob and the JavaScript export
envelope. The Worker persists the same package snapshot format.

### 9.1 Backend blobs

Both backends emit little-endian blobs with a magic, a version, a fixed header,
the vector data, then the per-node graph (level, base edges, upper-level edges).
Byte layouts are in [Appendix B](#appendix-b-serialization-byte-layouts).

- **FloatHNSW:** magic `0x464C4831` ("FLH1"), current version `1`; also reads a
  legacy magic. Stores raw (or normalized) float vectors + graph.
  (`float_hnsw.hpp:513`.)
- **Uint8FloatHNSW:** magic `0x49384831` ("I8H1"), current version `2`; reads a
  legacy magic and older versions. Stores scales, offsets, quantized bytes, then
  graph. Version 2 stores edge distances inline; importing an older version
  recomputes them. (`uint8_float_hnsw.hpp:660`.)

> **Deletion state does not survive a round-trip.** Neither backend serializes
> the `deleted_` flags — ghosts are written as ordinary vectors and come back
> *live* on import, and `num_deleted_` resets to 0. The contract is: **compact
> before export** if deletes must persist. The JS `export()` enforces this by
> throwing when `ghostCount > 0`; the Worker's persist path compacts first.
> (`float_hnsw.hpp:513`, `uint8_float_hnsw.hpp:660`.)

### 9.2 Statistics reconstruction (uint8)

The uint8 deserializer recomputes `sum_q` / `sum_q2` from the loaded quantized
bytes rather than storing them, and (for pre-v2 blobs) recomputes all edge
distances. So the on-disk format is slightly smaller than the in-memory
footprint.

### 9.3 JS export envelope (v3)

`PancakeIndex.export()` wraps the backend blob with a 32-byte header **plus an
embedded id-mapping table** so external ids survive an export/import cycle. This
is new in v3 — earlier envelopes (v1/v2, 20-byte header, still accepted on
import) carried no mapping. (`pikelet-core.js:10`, `:334`.)

```
Offset  Size           Field
0       4              Magic 0x504E434B ("PNCK")
4       4              Envelope version (3)
8       4              Dimension
12      4              Metric (0=L2, 1=Cosine)
16      4              Quantized (0/1)
20      4              nextExtId
24      4              Mapping entry count
28      4              WASM blob byte length
32      8 × count      Mapping table: [intId u32, extId u32] per entry
...     blobLen        Backend blob (PNCK-less, raw pancake_export bytes)
```

`import()` validates magic, version (1/2/3 accepted, ≥4 rejected), and that the
embedded **dim / metric / quantized** match the target index — mismatches throw
before the WASM import runs, preventing silent memory corruption. It also parses
the embedded raw backend header (for enveloped and bare blobs alike) and
validates its construction fields against the target, including
`count ≤ maxElements`. For v3 it validates the mapping table pre-import — the
entry count must equal the vector count declared in the backend header, internal
ids must fall in `[0, count)`, external ids must be unique and non-negative, and
`nextExtId` must exceed every mapped id — then commits the restored maps
(`pikelet-core.js:686–718`). A bare backend blob with no envelope is imported
with identity id mappings.

### 9.4 Untrusted-snapshot hardening (backend deserialize)

The JS envelope checks in §9.3 are the *outer* gate. The backend
`deserialize()` in each `*.hpp` parses an attacker-controlled byte buffer
directly (anything reaching `index.import()` or the raw-blob path), so it is
hardened to fail closed rather than corrupt memory or abort the instance:

- **Bounds checks use subtraction, not addition.** `size_t` is 32-bit under
  Emscripten (wasm32), so an `offset + len > data_size` test could wrap and pass
  while the following `memcpy` ran out of bounds. Every block-size check is
  written as `offset > data_size || data_size - offset < len`, and the float
  backend adds an explicit `count_*dims_` multiply-overflow guard before sizing
  the vector store. (`float_hnsw.hpp:638`, `uint8_float_hnsw.hpp:791`.)
- **The HNSW level count is capped** at `MAX_DESERIALIZE_LEVEL` (64). An
  unbounded `max_level` read from a snapshot would otherwise drive a
  multi-gigabyte per-node `upper_[i].resize()`, throwing `length_error` /
  `bad_alloc`. 64 levels covers any realistic element count.
  (`uint8_float_hnsw.hpp:158`.)
- **Quantization scales are validated** `> 0` (and finite, and `≤ 1e20`). A
  zero/negative stored scale would collapse a vector to a constant and poison
  every distance to it; `insert()` never produces one, so a snapshot may not
  carry one either. (`uint8_float_hnsw.hpp:215` for the insert-side guard the
  importer mirrors.)
- **Exceptions are contained** at the `pancake_import` boundary (§6.4): any
  remaining oversized allocation returns `-1` rather than unwinding out of the
  WASM module.

Per-edge neighbor ids are also validated `< count_`, and vector/scale/offset
components are rejected if non-finite (bit-level exponent check, since
`std::isfinite` is unreliable under `-ffast-math`).

## 10. Build Pipeline (WASM and Native)

### 10.1 WASM (`scripts/build-engine.mjs`)

Single translation unit (`src/engine.cpp`) compiled with Emscripten.
`./build.sh` is a thin shim that execs `node scripts/build-engine.mjs`; the
flags, export list, and env-var handling all live in the script. Key flags:

- **Optimization:** `-O3` (release) / `-O2 -gsource-map` (debug). Speed, not size
  — the binary is small enough that the speed trade wins.
- **SIMD:** `-msimd128` by default. `WASM_RELAXED_SIMD=1` adds `-mrelaxed-simd`
  (opt-in; the default build stays on plain SIMD so the checked-in artifact is
  broadly compatible and bit-reproducible). Also `-mnontrapping-fptoint`,
  `-msign-ext`, `-mbulk-memory`.
- **Runtime:** `MODULARIZE=1 EXPORT_NAME="P"`, `ENVIRONMENT='web,node'`,
  `ALLOW_MEMORY_GROWTH=1`, `INITIAL_MEMORY=16MB`, `STACK_SIZE=64KB`,
  `MALLOC=emmalloc`, `WASM_BIGINT=1` (64-bit ids across the boundary),
  `FILESYSTEM=0`, `DYNAMIC_EXECUTION=0`, `ASSERTIONS=0`,
  `DISABLE_EXCEPTION_CATCHING=0` (C++ exceptions on).
- **Exports:** the 22-function list in Appendix A; runtime methods
  `ccall, cwrap, HEAPF32, HEAPU8, HEAPU32, HEAP32`.
- **Output:** `dist/engine.js` (~17 KB) + `dist/engine.wasm` (~141 KB;
  ~50 KB gzipped).

A post-build `scripts/patch_engine.py` rewrites the Emscripten-generated environment
detection, forcing `ENVIRONMENT_IS_NODE = false` in `engine.js` so the modular
factory loads cleanly across Node, browser, and Workers rather than taking
Emscripten's CommonJS auto-path. (`scripts/patch_engine.py`.)

**Scalar fallback.** `npm run build:all` (`scripts/build-all.mjs`) builds the
SIMD engine above and then re-runs the same compile with `WASM_SIMD=0` to emit
`dist/engine.scalar.{js,wasm}`, a non-SIMD engine for runtimes without WASM
SIMD. The JS loaders probe `WebAssembly.validate` for SIMD support and pick the
scalar artifact when it is absent. Both engines compile from the same source, so
`prepublishOnly` runs `build:all` first to keep the shipped SIMD and scalar
artifacts in lockstep with the current `src/`.

### 10.2 Native (`native/binding.gyp`)

node-gyp compiles `pancake_napi.cpp` (which includes the same backend headers
from `../src`) with `-O3 -std=c++17 -ffast-math -ftree-vectorize -march=native
-mavx2 -msse2 -fno-rtti` and `-DPIKELET_ENABLE_AVX512_SIMD
-DPIKELET_ENABLE_AVX2_SIMD -DPIKELET_ENABLE_SSE2_SIMD`. AVX-512 instructions
come via `-march=native` and the kernels are compile-time gated on
`__AVX512F__` (plus `__AVX512BW__` for uint8), so non-AVX-512 hosts fall back to
AVX2 automatically. The macOS block defines the same macros; the Windows block
does not define `PIKELET_ENABLE_AVX512_SIMD` and stays on AVX2/SSE2. Output:
`native/build/Release/pikelet_native.node`.

### 10.3 Native vs WASM divergence

| | WASM | Native |
|---|---|---|
| Compiler | Emscripten | system clang/gcc/MSVC |
| SIMD | SIMD128 (opt. relaxed) | AVX-512 + AVX2 + SSE2 (AVX-512 on capable non-Windows hosts) |
| Boundary | C ABI + Emscripten heap | N-API |
| Query result | caller buffers (`uint64` ids) | JS object `{ ids: Uint32Array, distances: Float32Array, count }` |
| Exported surface | 22 C functions | ~15 N-API functions |
| Backends | identical (`src/*.hpp`) | identical (`src/*.hpp`) |

Because the backend code is shared, recall and graph structure are identical;
the native build exists only to measure the runtime-overhead delta.

---

## 11. Cloudflare Worker Deployment

Two reference Workers live under `examples/`. The first (`examples/legacy/reference-worker/`) is a
full read/write reference; the second (`examples/legacy/03-edge-docs-search/`) is a
hardened snapshot-first demo. The mental model for both is **snapshot search at
the edge**, not a durable mutable database inside one isolate.

**Ingress trust boundary (SSRF by construction):** deployed Workers never
fetch caller-supplied URLs. Snapshot import accepts size-capped *bytes* in
the request body; artifact and index data arrive as build-time bundled
assets or Cloudflare bindings (`env.ASSETS`, R2) whose targets are fixed at
deploy time. URL ingestion exists only in local, operator-driven tooling
(the `pikelet --source <url>` crawler, benchmarks) where the
operator controls both the URL and the machine. Keep it that way: a future
"restore from URL" route on a deployed Worker is the moment scheme
allowlisting, private/link-local IP rejection, redirect pinning, and
response size caps become mandatory — the crawler in
`pikelet/src/ingest.mjs` already implements most of that list
and is the model to copy.

### 11.1 Request lifecycle

Each isolate holds the active public `PancakeIndex` and a memoized restore
promise. The index stays warm across requests within an isolate. On any
non-trivial route, if `index` is null
and a bucket is bound, the Worker lazily **restores from R2** before serving
(`/health`, `/readiness`, `/reset_cache`, `/init`, `/import` skip auto-restore).
(`examples/legacy/reference-worker/worker.js`, `restoreIndex()`.)

### 11.2 Endpoints

| Endpoint | Method | Body → Response | Admin? |
|----------|--------|-----------------|--------|
| `/health` | GET | — → status, dims, count, restore timings, read_only | no (public) |
| `/readiness` | GET | — → loaded state + latest snapshot metadata/range inspection (no restore) | no (bearer once `API_KEY` set) |
| `/search` | POST | `{query,k?,efSearch?,allowedIds?}` → `{neighbors,search_ms}` | no (public) |
| `/stats` | GET | — → live/deleted counts, capacity, and structured memory | no (bearer once `API_KEY` set) |
| `/init` | POST | `{dims,maxElements,M?,efConstruction?,efSearch?,vectors?}` → init result | yes |
| `/add` | POST | `{vector}` → `{id,count}` | yes |
| `/add_batch` | POST | `{vectors}` → `{inserted,ids,count}` | yes |
| `/delete` | POST | `{id}` → boolean deletion result + deleted count | yes |
| `/compact` | POST | — → compaction result (awaits persist) | yes |
| `/export` | POST | — → compacted standard Pikelet snapshot | yes |
| `/import` | POST | Pikelet snapshot → config-inferred restore | yes |
| `/reset_cache` | POST | — → drops warm index, forces cold restore | yes |
| `/search_debug` | POST | same body and response as `/search` (admin-gated legacy alias) | yes |

Cross-cutting: `/health` and `/search` stay public, but admin routes now fail
closed unless `API_KEY` is set or `ALLOW_INSECURE_ADMIN=1` is explicitly enabled
for a local/demo deployment. Once `API_KEY` is set, `/stats`, `/readiness`, and
admin routes require the bearer token. Also: optional per-IP sliding-window rate limiting
(`RATE_LIMIT_RPM`, 60 s window, per-isolate so the effective global limit is
`limit × isolates`); request-body caps for JSON (`MAX_JSON_BYTES`) and binary
snapshot import (`MAX_SNAPSHOT_BYTES`); and opt-in CORS via `ALLOWED_ORIGIN`
(unset => no `Access-Control-Allow-Origin` header).

### 11.3 R2 persistence

Snapshots are written under **timestamped, append-only keys**
(`pancake-index-<13-digit-ms>-<6-digit-seq>.pnck`); restore lists the prefix and
picks the lexicographically greatest key across all R2 list pages (zero-padding
makes string order match time order), with a fallback to a legacy fixed key.
`/init`, `/import`, `/add`, and `/add_batch` persist snapshots when the index has
no ghosts. `/delete` is intentionally cheap and leaves the soft-delete in warm
memory; deletes become durable when an explicit `/compact` or `/export` compacts
the index and writes a fresh snapshot. Without an R2 binding, the persist path
still keeps the latest exported snapshot in isolate memory (`localSnapshot`), so
`/reset_cache` → restore works bucket-less within an isolate after a persisted
mutation. A production deployment should add an R2 lifecycle rule or delete
superseded keys; retention is deliberately left to the application.

The append-only scheme means a slow/late async write cannot clobber a newer
snapshot under a shared key — restore always loads the newest. The durability
boundary is R2; in-memory isolate state is a warm cache only.

The stored object is the standard Pikelet v3 envelope. `inspectSnapshot()` reads
its construction fields for readiness and `restore()` reconstructs the index on
cold start. R2 custom metadata stores the full config, but for a v3 envelope
snapshot only capacity and runtime `efSearch` policy are taken from it — the
fixed construction fields come from the envelope itself. Only a legacy raw
snapshot takes its full config from metadata, validated against the Worker
limits.

### 11.4 READ_ONLY mode

`READ_ONLY` (`1/true/yes/on`) makes every admin route return `403`. The check is
a single guard that runs **before** auth and rate limiting, immediately after
the CORS preflight: `if (ADMIN_ROUTES.has(path) && isReadOnly(env)) return 403`.
`/search`, `/stats`, `/health`, `/readiness` remain available. This is the
recommended posture for a public, snapshot-backed search endpoint: publish the
index out-of-band, deploy read-only, expose only search.
(`examples/legacy/reference-worker/worker.js`, `ADMIN_ROUTES` / `isReadOnly()`.)

### 11.5 ID mapping

The Worker uses the public API and does not maintain a second mapping layer.
Stable external IDs live in `pikelet-core.js` and are serialized inside the v3
snapshot envelope. Cold restore, compaction, and subsequent insertion therefore
exercise the same mapping contract as Node and browser consumers.

### 11.6 Semantic-search demo differences

`examples/legacy/03-edge-docs-search/` is snapshot-first and read-oriented: it
builds the index offline and **bundles four assets into the Worker script** as
ES-module imports (`assets/docs-index.bin` snapshot, `assets/docs-student.bin`
distilled encoder weights, `assets/docs-corpus.json`, `assets/docs-manifest.json`)
— no R2 binding, no model API, no outbound requests. Query embedding runs
locally through a 1.08 MB int8 distilled student encoder, which is validated
against the manifest (dim, byte length, SHA-256) before the snapshot is restored
with the manifest's `maxElements`/`efSearch`. It serves `/search` (GET
`?q=&k=&ef=&source=` or POST, with source-filtered search via precomputed
per-source ID sets fed to `searchFiltered`), plus `/health` (public) and a
minimal UI; `/readiness` and `/reset_cache` are its admin routes, requiring
`API_KEY` or `ALLOW_INSECURE_ADMIN`. Public search parameters are clamped
(`k ≤ 8`, `ef` clamped to 10–400), it deploys with `READ_ONLY=1` by default,
and it has no write endpoints. It uses the high-level `pikelet-core.js` API
rather than raw WASM exports.

---

## 12. Configuration Reference

### 12.1 `Pikelet.create(opts)` — JS defaults (the effective ones)

| Option | Default | Notes |
|--------|---------|-------|
| `dim` | required | positive integer |
| `maxElements` | `100000` | capacity, pre-allocated |
| `metric` | `'cosine'` | `'cosine'` or `'l2'` |
| `quantized` | `true` | uint8 backend when true |
| `M` | `12` | |
| `efConstruction` | `75` | |
| `efSearch` | `100` | valid range 1–4096; mutable via `setEfSearch()` or overridden per query |
| `seed` | `108` | deterministic HNSW level-assignment seed |

Removed options (`compressed`, `varianceSample`) throw if passed.

### 12.2 C ABI fallback defaults (direct-ABI callers only)

`M=12`, `ef_construction=75`, `ef_search=100`, `max_elements=100000`, and
`seed=108`, applied only when a non-positive value reaches `pancake_init`.
Normal JS callers already pass these explicitly.

### 12.3 Worker limits and env

| Constant | Value | | Env var | Default | Purpose |
|----------|-------|---|---------|---------|---------|
| `MAX_RESULTS` | 100 | | `API_KEY` | required for admin routes unless `ALLOW_INSECURE_ADMIN=1` | bearer token |
| `MAX_DIMS` | 4096 | | `ALLOWED_ORIGIN` | unset | opt-in CORS |
| `MAX_EF` | 4096 | | `RATE_LIMIT_RPM` | 0 (off) | per-IP/min |
| `MAX_M` | 128 | | `READ_ONLY` | off | reject admin routes |
| `DEFAULT_MAX_ELEMENTS` | 5000 | | `MAX_ELEMENTS_LIMIT` | 5,000 (code fallback; example wrangler.toml sets 200,000) | capacity ceiling |
| `DEFAULT_MAX_JSON_BYTES` | 1 MiB | | `MAX_JSON_BYTES` | 1 MiB | JSON body cap |
| `DEFAULT_MAX_SNAPSHOT_BYTES` | 64 MiB | | `MAX_SNAPSHOT_BYTES` | 64 MiB | snapshot import/restore cap |
| `RATE_LIMIT_WINDOW_MS` | 60000 | | | | |
| | | | `ALLOW_INSECURE_ADMIN` | off | local/demo opt-in for unauthenticated admin routes |

### 12.4 Build constants

`MAX_HANDLES=64`, `INVALID_HANDLE=0xFFFFFFFF` (engine.cpp);
`INITIAL_MEMORY=16MB`, `STACK_SIZE=64KB` (scripts/build-engine.mjs).

---

## 13. Invariants

- **I1 — Handle validity.** A handle is valid from `pancake_init` until
  `pancake_dispose`; range and null checks guard every ABI call. Max 64 live.
- **I2 — Single-threaded.** All access to a WASM instance must be from one
  thread; enforced by the WASM model, not the code. Each Worker isolate /
  worker_thread needs its own instance.
- **I3 — Insert ids are sequential and unique** within a handle (`count_++`),
  until compaction renumbers survivors (transparent through the JS id map).
- **I4 — Capacity is hard.** Insert at `count == max_elements` returns
  `UINT32_MAX` and mutates nothing.
- **I5 — Delete is immediate and permanent** (no undelete); the node is skipped
  from the next search on.
- **I6 — Compact preserves the live set**, rebuilds the graph (quality may shift),
  and renumbers survivors in old-id order. (The JS layer does not rely on the
  renumbering rule — it consumes the `compact_remap` out-map.)
- **I7 — Deletion state is not serialized.** Import returns all vectors live;
  compact before export to persist deletes. JS `export()` enforces
  `ghostCount===0`.
- **I8 — Envelope validation on import.** dim/metric/quantized mismatches (and,
  for v3, mapping-count / nextExtId inconsistencies) throw before the WASM import.
- **I9 — Export pointer lifetime.** The `pancake_export` pointer is valid only
  until the next export on that handle or dispose; copy immediately.

---

## 14. Limitations and Non-Goals

- **No metadata storage.** The index stores vectors + graph only. *Filtered*
  search exists (`searchFiltered`), but the caller supplies the allowed-id set
  and owns all id→metadata mapping.
- **No server-side persistence in Node.** In-memory only; persistence is
  `export()`/`import()` that the application drives. (The Worker adds R2.)
- **No concurrency within an instance.** Single-threaded by design.
- **No cross-handle / multi-index query merge.** Each handle is independent.
- **No incremental serialization.** `export` writes the whole index.
- **No automatic compaction.** The host (or Worker `/compact`) triggers it.
- **No dimensionality reduction.** Vectors arrive at their final dimension.
- **Worker isolates are independent.** No cross-isolate read-after-write
  consistency; R2 is the only durability boundary and uses last-write-wins by
  newest key.
- **Backend code is duplicated, not shared.** float and uint8 backends repeat the
  graph algorithm; a change to HNSW logic must be made in both.
- **Relaxed SIMD is non-deterministic** and therefore opt-in; the default build
  is the reproducible one.

---

## Appendix A: WASM Export Inventory

22 functions exported by `scripts/build-engine.mjs` (`-s EXPORTED_FUNCTIONS`):

**Index API:** `_pancake_init`, `_pancake_add`, `_pancake_bulk_insert`,
`_pancake_query`, `_pancake_query_filtered`, `_pancake_delete`,
`_pancake_compact`, `_pancake_compact_remap`, `_pancake_count`,
`_pancake_memory`, `_pancake_ghost_count`, `_pancake_ghost_ratio`,
`_pancake_export`, `_pancake_import`, `_pancake_dispose`,
`_pancake_dimension`, `_pancake_shutdown_all`, `_shutdown_all`.

**Utilities:** `_emsc_malloc`, `_emsc_free`, `_pancake_profile_print`,
`_pancake_profile_reset`.

**Runtime methods:** `ccall`, `cwrap`, `HEAPF32`, `HEAPU8`, `HEAPU32`, `HEAP32`.

Selected signatures:

| Export | Signature → returns |
|--------|---------------------|
| `_pancake_init` | `(dim, max_elem, quantized, metric, M, ef_c, ef_s)` → handle / `0xFFFFFFFF` |
| `_pancake_add` | `(handle, vec_ptr)` → id / `0xFFFFFFFF` |
| `_pancake_bulk_insert` | `(handle, vecs_ptr, n)` → inserted count |
| `_pancake_query` | `(handle, q_ptr, k, ef_search, ids_ptr, dists_ptr)` → count |
| `_pancake_query_filtered` | `(handle, q_ptr, k, ef_search, ids_ptr, dists_ptr, bitset_ptr, bitset_len)` → count |
| `_pancake_compact_remap` | `(handle, out_buf, out_capacity)` → pre-compaction count |
| `_pancake_export` | `(handle, out_size_ptr)` → data_ptr / null |
| `_pancake_import` | `(handle, data_ptr, size)` → `0` / `-1` |

The native N-API addon exposes an equivalent (smaller) surface; its
`pancake_query` returns `{ ids: Uint32Array, distances: Float32Array, count }`
rather than writing into caller buffers.

---

## Appendix B: Serialization Byte Layouts

### FloatHNSW blob — magic `0x464C4831` ("FLH1"), version 1

```
0   u32   magic 0x464C4831
4   u32   dims
8   u32   version (1)
12  u32   count
16  u32   entry_point
20  u32   max_level
24  u32   M
28  u32   M0
32  u32   metric (0=L2, 1=Cosine)
36  u32   ef_construction
40  count×dims×f32   vectors (raw or normalized)
... per node: u32 level, then per level: u32 edge_count, edge_count×u32 neighbor ids
```

Deletion state not serialized.

### Uint8FloatHNSW blob — magic `0x49384831` ("I8H1"), version 2

```
0   u32   magic 0x49384831
4   u32   dims
8   u32   version (2)
12  u32   count
16  u32   entry_point
20  u32   max_level
24  u32   M
28  u32   M0
32  u32   metric
36  u32   ef_construction
40  count×f32   scales
... count×f32   offsets
... count×dims×u8   quantized data
... per node: u32 level, u32 base_edge_count, base edges (u32 neighbor + f32 dist),
              then per upper level: u32 edge_count, edges (u32 neighbor + f32 dist)
```

`sum_q`/`sum_q2` are recomputed on load, not stored. Pre-v2 blobs omit edge
distances and have them recomputed on import. Deletion state not serialized.

### JS export envelope (v3) — magic `0x504E434B` ("PNCK")

See §9.3. 32-byte header + `[intId,extId]` mapping table + backend blob.

---

*End of document.*
