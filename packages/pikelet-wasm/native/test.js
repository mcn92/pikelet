'use strict';

const assert = require('node:assert/strict');
const native = require('./index.js');

const handle = native.pikelet_init(4, 8, 0, 0, 4, 16, 16);
assert.notEqual(handle, 0xFFFFFFFF);

try {
  assert.throws(
    () => native.pikelet_add(handle, new Float32Array(3)),
    /shorter than the index dimension/
  );
  assert.throws(
    () => native.pikelet_add(handle, new Uint8Array(4)),
    /must be a Float32Array/
  );
  assert.throws(
    () => native.pikelet_bulk_insert(handle, new Float32Array(7), 2),
    /shorter than n \* dimension/
  );
  assert.throws(
    () => native.pikelet_bulk_insert(handle, new Float32Array(8), -1),
    /n must be non-negative/
  );
  assert.throws(
    () => native.pikelet_query(handle, new Float32Array(3), 1),
    /shorter than the index dimension/
  );
  assert.throws(
    () => native.pikelet_query(handle, new Uint8Array(4), 1),
    /must be a Float32Array/
  );

  assert.equal(native.pikelet_add(handle, new Float32Array([0, 0, 0, 0])), 0);
  const result = native.pikelet_query(handle, new Float32Array([0, 0, 0, 0]), 99);
  assert.equal(result.count, 1);
  assert.throws(
    () => native.pikelet_import(handle, 'not a buffer'),
    /must be a Buffer or Uint8Array/
  );
  assert.throws(
    () => native.pikelet_import(handle, new Float32Array(4)),
    /must be a Buffer or Uint8Array/
  );
} finally {
  native.pikelet_dispose(handle);
}

// Constructor limits reach the binding as JS RangeErrors instead of UB
// (M = 1 made level_mult_ infinite) or a process abort (a C++ exception
// crossing N-API). M = 0 still means "default".
for (const quantized of [0, 1]) {
  assert.throws(() => native.pikelet_init(4, 8, quantized, 0, 1, 16, 16), /M must be between 2 and 128/);
  assert.throws(() => native.pikelet_init(4, 8, quantized, 0, 129, 16, 16), /M must be between 2 and 128/);
  assert.throws(() => native.pikelet_init(65537, 8, quantized, 0, 4, 16, 16), /dims must be between 1 and 65536/);
  const dflt = native.pikelet_init(4, 8, quantized, 0, 0, 16, 16);
  assert.notEqual(dflt, 0xFFFFFFFF);
  native.pikelet_dispose(dflt);
}

// serialize() must refuse while ghosts are resident: the wire format has no
// deleted flag, so an export taken with a ghost present would resurrect it
// as live, searchable data on the next import with no signal anything was
// wrong. This binding had no such guard before (unlike the WASM path, which
// checks ghostCount in JS before ever reaching serialize()) — this is the
// one path where a ghost could actually reach a snapshot, so it is the one
// that most needs direct coverage.
for (const quantized of [0, 1]) {
  const gh = native.pikelet_init(4, 8, quantized, 0, 4, 16, 16);
  try {
    native.pikelet_add(gh, new Float32Array([1, 0, 0, 0]));
    native.pikelet_add(gh, new Float32Array([0, 1, 0, 0]));
    native.pikelet_delete(gh, 0);
    assert.equal(native.pikelet_ghost_count(gh), 1);
    assert.throws(
      () => native.pikelet_export(gh),
      /ghost entries/
    );
    // compact() strips the ghost; export must then succeed normally.
    native.pikelet_compact(gh);
    assert.equal(native.pikelet_ghost_count(gh), 0);
    const buf = native.pikelet_export(gh);
    assert.ok(buf.length > 0);
  } finally {
    native.pikelet_dispose(gh);
  }
}

// L2 insert/search must reject non-finite vectors the same way the Cosine
// path already does via normalize_cosine_vector: the L2 path stores raw
// floats with no check of its own, so a NaN component used to make every
// distance comparison false (arbitrary neighbours, NaN keys breaking the
// priority queue's ordering invariant) and, in the quantized engine, drove
// vmin/vmax/scale to NaN and reached static_cast<uint8_t>(NaN) past its
// clamp guards — undefined behavior. Only the JS-side isFinite check in
// add() shielded this; native bindings had no guard of their own.
for (const quantized of [0, 1]) {
  const nh = native.pikelet_init(4, 8, quantized, 0 /* L2 */, 4, 16, 16);
  try {
    assert.equal(native.pikelet_add(nh, new Float32Array([1, 0, 0, 0])), 0);
    assert.equal(
      native.pikelet_add(nh, new Float32Array([Number.NaN, 0, 0, 0])),
      0xFFFFFFFF,
      `quantized=${quantized}: insert must refuse a NaN component`
    );
    assert.equal(
      native.pikelet_add(nh, new Float32Array([Infinity, 0, 0, 0])),
      0xFFFFFFFF,
      `quantized=${quantized}: insert must refuse an infinite component`
    );
    const result = native.pikelet_query(nh, new Float32Array([Number.NaN, 0, 0, 0]), 1);
    assert.equal(result.count, 0, `quantized=${quantized}: search must refuse a NaN query`);
  } finally {
    native.pikelet_dispose(nh);
  }
}

// The symmetric u8 kernel (uint8_dot) runs only during graph maintenance —
// insert-time edge recompute, the diversity heuristic, pruning — so a real
// quantized build is the only thing that exercises its ISA branch. 400
// deterministic unit vectors at 128D: every vector must find itself first
// at near-zero distance, and the graph's top-1 for held-out probes must
// match float brute force. A garbage or silently-scalar SIMD branch
// corrupts edge selection and this collapses.
{
  const DIM = 128;
  const N = 400;
  let seed = 42 >>> 0;
  const rand = () => {
    seed ^= (seed << 13) >>> 0; seed ^= seed >>> 17; seed ^= (seed << 5) >>> 0;
    return (seed >>> 0) / 0xffffffff;
  };
  const vectors = [];
  for (let i = 0; i < N; i++) {
    const v = new Float32Array(DIM);
    let norm = 0;
    for (let d = 0; d < DIM; d++) { v[d] = rand() - 0.5; norm += v[d] * v[d]; }
    norm = Math.sqrt(norm);
    for (let d = 0; d < DIM; d++) v[d] /= norm;
    vectors.push(v);
  }
  const qh = native.pikelet_init(DIM, N, 1, 1, 16, 100, 60);
  assert.notEqual(qh, 0xFFFFFFFF);
  try {
    for (const v of vectors) native.pikelet_add(qh, v);
    let selfHits = 0;
    for (let i = 0; i < N; i++) {
      const res = native.pikelet_query(qh, vectors[i], 1);
      if (res.count === 1 && res.ids[0] === i) {
        selfHits++;
        assert.ok(res.distances[0] < 0.05, `self-distance ${res.distances[0]} at id ${i}`);
      }
    }
    assert.ok(selfHits >= Math.floor(N * 0.99), `quantized self-recall@1 ${selfHits}/${N}`);

    // Held-out probes: perturb a stored vector slightly; float brute force
    // and the quantized graph must agree on the nearest neighbor.
    for (let p = 0; p < 20; p++) {
      const base = Math.floor(rand() * N);
      const q = new Float32Array(DIM);
      let norm = 0;
      for (let d = 0; d < DIM; d++) { q[d] = vectors[base][d] + (rand() - 0.5) * 0.02; norm += q[d] * q[d]; }
      norm = Math.sqrt(norm);
      for (let d = 0; d < DIM; d++) q[d] /= norm;
      let best = -1;
      let bestDot = -Infinity;
      for (let i = 0; i < N; i++) {
        let dot = 0;
        for (let d = 0; d < DIM; d++) dot += q[d] * vectors[i][d];
        if (dot > bestDot) { bestDot = dot; best = i; }
      }
      const res = native.pikelet_query(qh, q, 1);
      assert.equal(res.ids[0], best, `probe ${p}: graph ${res.ids[0]} vs brute force ${best}`);
    }
  } finally {
    native.pikelet_dispose(qh);
  }
}

console.log('Native validation checks passed.');
