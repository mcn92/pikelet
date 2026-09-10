# Quantization conformance test

`bench/beir/quantize.mjs` reimplements Pikelet's row-wise affine u8
quantization in JS because `PancakeIndex` has no readback API for stored
vectors (see `bench/beir/README.md`). This directory proves that
reimplementation is exact, not just plausible.

- `gen-golden-quantization.cpp` — the quantization logic copied **verbatim**
  from `src/uint8_float_hnsw.hpp:198-234` (the real engine's `insert()`
  cosine path), compiled standalone with plain `g++` (no Emscripten
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
g++ -O2 -std=c++17 bench/beir/conformance/gen-golden-quantization.cpp -o /tmp/gen
/tmp/gen > bench/beir/conformance/golden-quantization.json
```

Run the test:

```bash
node bench/beir/conformance/test-quantization.mjs
```
