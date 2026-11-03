# Quantization conformance test

`benchmarks/beir/quantize.mjs` reimplements Pikelet's row-wise affine u8
quantization in JS because `PikeletIndex` has no readback API for stored
vectors (see `benchmarks/beir/README.md`). This directory proves that
reimplementation is exact, not just plausible.

- `gen-golden-quantization.cpp` — the quantization logic copied **verbatim**
  from `packages/pikelet-wasm/engine/uint8_float_hnsw.hpp` (the quantize
  block of the real engine's `insert()`, cosine path), compiled standalone with plain `g++` (no Emscripten
  needed — this only exercises the scalar math, not the WASM bindings).
  Emits 5 known rows (input, normalized, scale, offset, quantized bytes,
  a query vector, and the exact float dot product) as
  `golden-quantization.json`.
- `test-quantization.mjs` — asserts `quantize.mjs` produces byte-identical
  `scale`/`offset`/`q[]` for each golden row, and that the affine
  dot-product identity (`offset·Σy + scale·(y·q)`) reproduces the true
  float dot product within 8-bit quantization tolerance.

Regenerate the golden file only if `uint8_float_hnsw.hpp`'s quantization
logic changes:

```bash
g++ -O2 -std=c++17 benchmarks/beir/conformance/gen-golden-quantization.cpp -o /tmp/gen
/tmp/gen > benchmarks/beir/conformance/golden-quantization.json
```

Run the test:

```bash
node benchmarks/beir/conformance/test-quantization.mjs
```
