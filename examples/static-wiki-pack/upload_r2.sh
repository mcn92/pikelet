#!/bin/bash
# Upload the wiki pack to R2. See DEPLOY.md for the full runbook.
#
# corpus.bin exceeds wrangler's 300 MiB single-upload ceiling, so it is
# uploaded as two fixed-size parts; the Pages Function's SPLIT_OBJECTS map
# (web/functions/_serve-r2.js) must agree with PART_SIZE and the file's
# total size. Everything else uploads whole.
set -euo pipefail
cd "$(dirname "$0")"

BUCKET=pancake-wiki-pack
DATA=data-perm
M=node_modules/@huggingface/transformers/.cache/Xenova/all-MiniLM-L6-v2
O=node_modules/onnxruntime-web/dist
PART_SIZE=209715200   # 200 MiB; keep in sync with SPLIT_OBJECTS

put() { npx wrangler r2 object put "$BUCKET/$1" --file "$2" --content-type "$3" --remote; }

# --- pack: corpus split ---
split -b "$PART_SIZE" "$DATA/corpus.bin" "$DATA/corpus.split."
put pack/corpus.bin.p0 "$DATA/corpus.split.aa" application/octet-stream
put pack/corpus.bin.p1 "$DATA/corpus.split.ab" application/octet-stream
rm "$DATA"/corpus.split.*

# --- pack: everything else ---
put pack/wiki.pikelet-sketch      "$DATA/wiki.pikelet-sketch"      application/octet-stream
put pack/corpus-offsets.u32       "$DATA/corpus-offsets.u32"       application/octet-stream
put pack/wiki-vocab.bloom         "$DATA/wiki-vocab.bloom"         application/octet-stream
put pack/wiki-abstention.json     "$DATA/wiki-abstention.json"     application/json
put pack/wiki-abstention-probes.json "$DATA/wiki-abstention-probes.json" application/json
put pack/pack-manifest.json       "$DATA/pack-manifest.json"       application/json

# --- encoder (self-hosted MiniLM) ---
put models/Xenova/all-MiniLM-L6-v2/config.json           "$M/config.json"           application/json
put models/Xenova/all-MiniLM-L6-v2/tokenizer.json        "$M/tokenizer.json"        application/json
put models/Xenova/all-MiniLM-L6-v2/tokenizer_config.json "$M/tokenizer_config.json" application/json
put models/Xenova/all-MiniLM-L6-v2/onnx/model_fp16.onnx  "$M/onnx/model_fp16.onnx"  application/octet-stream

# --- ONNX runtime wasm ---
put ort/ort-wasm-simd-threaded.jsep.wasm "$O/ort-wasm-simd-threaded.jsep.wasm" application/wasm
put ort/ort-wasm-simd-threaded.wasm      "$O/ort-wasm-simd-threaded.wasm"      application/wasm
put ort/ort-wasm-simd-threaded.jsep.mjs  "$O/ort-wasm-simd-threaded.jsep.mjs"  text/javascript
put ort/ort-wasm-simd-threaded.mjs      "$O/ort-wasm-simd-threaded.mjs"      text/javascript

echo "ALL_UPLOADS_DONE"
