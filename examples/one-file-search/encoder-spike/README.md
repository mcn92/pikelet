# Inline-transformer encoder spike (query-interp kind 3)

Feasibility spike for compiling the wiki pack's pinned teacher
(MiniLM-L6-v2) INTO the `.pikelet` as fused-quantized WASM — the engine's
asymmetric-distance move applied to transformer weights: block-affine u8
weights, f32 activations, dequantization fused into the SIMD dot product so
float weight tensors are never materialized
(`dot(dequant(block), x) == scale * dot(u8, x) + offset * sum(x)`).

```bash
bash build.sh            # standalone WASM (not part of dist/)
node bench.mjs           # throughput + parity, synthetic weights
python3 export_weights.py  # real MiniLM layer-0 weights + true activations
node bench-real.mjs      # quantization accuracy on real distributions
```

## Measured (2026-08-14, Ryzen 9 4900HS, single thread, WASM SIMD128)

Throughput at MiniLM's GEMV shapes, block=64:

| shape | fused-u8 | scalar-u8 | f32 |
| --- | ---: | ---: | ---: |
| attn 384x384 | 11.58 GFLOPs | 2.89 | 11.37 |
| ffn up 1536x384 | 11.89 GFLOPs | 2.86 | 11.59 |
| ffn down 384x1536 | 11.91 GFLOPs | 2.84 | 10.59 |

The headline reproduces the engine's result at transformer scale: fused
dequant runs at full f32 speed with one quarter of the weight bytes — the
conversion cost is absorbed entirely by reduced memory traffic.

Projected full-forward GEMV latency at seq=32, single thread: MiniLM-L6
(679 MFLOPs) ~58 ms; a tiny-2L design point ~4.8 ms. Untouched
accelerators: threads (~/4) and the relaxed-SIMD int8 dot product (the
build system already carries the WASM_RELAXED_SIMD flag).

Quantization accuracy on REAL layer-0 weights with the true activations
that feed them (fused-u8 output vs exact f32, per GEMV):

| matrix | block 32 | block 64 | block 128 |
| --- | ---: | ---: | ---: |
| attn q | 0.9999909 | 0.9999877 | 0.9999838 |
| attn o | 0.9999920 | 0.9999878 | 0.9999841 |
| ffn up | 0.9999945 | 0.9999927 | 0.9999908 |
| ffn down | 0.9999701 | 0.9999586 | 0.9999465 |

(mean cosine; worst single row 0.99992.) ffn_down — post-GELU activations,
widest input — is the least accurate matrix and still ≥ 0.9999. Per-GEMV
relRMSE 0.3–1%, consistent with 8-bit block quantization being
near-lossless end to end in comparable systems.

## Verdict

Compute-feasible and numerically promising: a self-contained ~11–23 MB
in-file teacher at ~58 ms/query single-threaded (the wiki JS sketch scan
is 273 ms today, so the encoder would not be the bottleneck), with no host
ML runtime. Remaining before this becomes a format kind:

1. Full BERT-shaped layer with the fused pipeline (LayerNorm in fp32,
   fused QKV, streamed GELU -> down-projection accumulators that never
   materialize the FFN intermediate; column-major quantized blocks for the
   down matrix), parity-checked against fp32 on the real weights.
2. Six-layer forward + WordPiece tokenizer, verified against the wiki
   artifact's committed encoder test vectors (the kind-2 verification
   machinery is the acceptance harness).
3. Weight-segment layout: the 11.5 MB token-embedding table (half the
   model) as a LAZY range-read segment — rows fetched per token id,
   LRU-cached — the sketch profile's resident/lazy split applied inside
   the encoder. Resident cost drops to the ~11 MB of layer weights.

The lexicon-student findings that motivated this (generalist 14.2%,
category specialist 60.8% vs teacher 93.8% on wiki) are recorded in
../README.md history and the training scripts alongside this directory.

## Update 2026-08-15: full six-layer forward, verified

The remaining risk items are closed. The complete inline encoder now runs:
JS WordPiece (wordpiece.mjs, 200/200 id parity with the HF tokenizer on the
wiki eval queries) -> six-layer fused-u8 WASM forward (encoder.cpp over the
24.3 MiB blob from export_encoder_blob.py) -> mean-pool + normalize.

- Stage parity (test-encoder-parity.mjs): cosine 1.00000000 at every stage
  (embeddings+LN and all six layers) vs the numpy reference computed from
  the same fake-quantized weights; max abs diff ~2e-6 (f32 accumulation
  noise). Forward pass 26 ms at seq=12, single thread.
- Error compounding (sim_quant_forward.py): block-64 u8 on every Linear
  and embedding table costs one nine over 42 GEMVs — final-embedding
  cosine mean 0.999867 / min 0.999392 vs fp32.
- End to end (run-encoder-recall.mjs): recall@10 = 82.4% through
  pancake-wiki.pancake vs exact ground truth — identical to the torch
  fake-quant simulation, 0.4 points under the fp32 teacher (82.8%).
  Mean encode 12.5 ms/query single thread, ~20x faster than the current
  JS sketch scan (273 ms).

Remaining for a kind-3 format entry is packaging, not correctness: the
encoder segment layout (blob + vocab as segments, embedding table as a
lazy range-read region), the reader integration, and the streamed-FFN /
threads / relaxed-SIMD optimizations as wanted.

(Since shipped: the packaging and reader integration are kind 3,
`inline-transformer-v1`, in `spec/COMPLETE_PROFILE.md` 3.6, and the
kernels build from `packages/pikelet-wasm/complete/encoder-kernels/`. This
log is the spike's record as of its writing.)
