// Row-wise affine u8 quantization, matching Pikelet's engine exactly
// (src/uint8_float_hnsw.hpp:206-234 — insert(), cosine-normalized-then-quantized).
// Reimplemented here because the WASM index exposes no readback API for
// stored quantized bytes (Pikelet.create()'s export() is an opaque
// whole-index snapshot) — see bench/beir/README.md quantization note.

// L2-normalize (cosine metric normalizes before quantizing: uint8_float_hnsw.hpp:198-200).
export function l2normalize(vec) {
  let s = 0;
  for (let d = 0; d < vec.length; d++) s += vec[d] * vec[d];
  const norm = Math.sqrt(s);
  if (norm < 1e-30) return vec.slice();
  const out = new Float32Array(vec.length);
  for (let d = 0; d < vec.length; d++) out[d] = vec[d] / norm;
  return out;
}

// Per-row affine u8 quantize. Returns { q: Uint8Array, scale, offset, sumQ }.
export function quantizeRow(vec) {
  const dim = vec.length;
  let vmin = vec[0];
  let vmax = vec[0];
  for (let d = 1; d < dim; d++) {
    if (vec[d] < vmin) vmin = vec[d];
    if (vec[d] > vmax) vmax = vec[d];
  }
  let range = vmax - vmin;
  if (range < 1e-30) range = 1.0;
  const scale = range / 255.0;
  const invScale = 255.0 / range;
  const q = new Uint8Array(dim);
  let sumQ = 0;
  for (let d = 0; d < dim; d++) {
    let v = (vec[d] - vmin) * invScale + 0.5;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    const qv = v | 0;
    q[d] = qv;
    sumQ += qv;
  }
  return { q, scale, offset: vmin, sumQ };
}

// Exact dot product of a float query against a quantized stored row, via the
// affine identity (README "One decision"): y·x = offset·Σy + scale·(y·q)
// No decompressed copy of the row is materialized.
export function dotQueryAgainstQuantized(queryVec, sumQuery, row) {
  const { q, scale, offset } = row;
  let acc = 0;
  for (let d = 0; d < queryVec.length; d++) acc += queryVec[d] * q[d];
  return offset * sumQuery + scale * acc;
}

export function sum(vec) {
  let s = 0;
  for (let d = 0; d < vec.length; d++) s += vec[d];
  return s;
}
