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
} finally {
  native.pikelet_dispose(handle);
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
