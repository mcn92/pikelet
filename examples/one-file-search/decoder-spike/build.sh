#!/bin/bash
# Build the decoder kernel to standalone WASM modules (not part of any
# package): decoder.mjs for Node, decoder.web.mjs for browsers
# (browser_check.mjs). Same source and flags.
set -e
cd "$(dirname "$0")"
source ~/emsdk/emsdk_env.sh >/dev/null 2>&1
emcc decoder.cpp -std=c++20 -O3 -msimd128 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node \
  -sEXPORTED_FUNCTIONS=_decoder_forward,_decoder_expand_u5,_decoder_set_shortlist,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32 \
  -sALLOW_MEMORY_GROWTH=1 \
  -o decoder.mjs
emcc decoder.cpp -std=c++20 -O3 -msimd128 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker \
  -sEXPORTED_FUNCTIONS=_decoder_forward,_decoder_expand_u5,_decoder_set_shortlist,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32 \
  -sALLOW_MEMORY_GROWTH=1 \
  -o decoder.web.mjs
# The Llama-architecture kernel (llama_kernel.cpp): its blobs and KV cache run to
# hundreds of MB, so memory may grow to the wasm32 limit.
for env in node web,worker; do
  out=llama.mjs; [ "$env" = node ] || out=llama.web.mjs
  emcc llama_kernel.cpp -std=c++20 -O3 -msimd128 \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=$env \
    -sEXPORTED_FUNCTIONS=_llama_set_threads,_llama_init,_llama_forward,_llama_expand_u5,_llama_dim,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32 \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB \
    -o $out
done
# The same source with -pthread: llama-mt.mjs (Node) and llama-mt.web.mjs
# (a browser worker; needs a cross-origin-isolated page). 15 workers are
# started with the module, so llama_set_threads can use up to 16 threads
# without returning to the event loop first.
for env in node web,worker; do
  out=llama-mt.mjs; [ "$env" = node ] || out=llama-mt.web.mjs
  emcc llama_kernel.cpp -std=c++20 -O3 -msimd128 -pthread -sPTHREAD_POOL_SIZE=15 \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=$env \
    -sEXPORTED_FUNCTIONS=_llama_set_threads,_llama_init,_llama_forward,_llama_expand_u5,_llama_dim,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32,HEAP32 \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB \
    -o $out
done
echo "built decoder.mjs + decoder.wasm, decoder.web.mjs + decoder.web.wasm, llama.mjs + llama.wasm, llama.web.mjs + llama.web.wasm, llama-mt.mjs + llama-mt.wasm, llama-mt.web.mjs + llama-mt.web.wasm"
