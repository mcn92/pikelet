#!/usr/bin/env node
// Full-stack acceptance for the inline encoder: WordPiece (JS) -> six-layer
// fused-u8 forward (WASM) -> mean-pool + normalize -> recall@10 through
// pancake-wiki.pancake against exact ground truth. The fake-quant torch
// simulation measured 82.4% for this exact weight encoding; matching it
// means the WASM path is the simulation, realized.

import fs from 'node:fs';
import path from 'node:path';
import createModule from './encoder.node.mjs';
import { createWordPiece } from './wordpiece.mjs';
import { openPikeletFile } from '../pikelet-file-reader.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const REAL = path.join(here, 'real');
const DATA = path.join(here, '..', '..', '04-static-wiki-pack', 'data-full');
const D = 384;

const tokenizer = createWordPiece(fs.readFileSync(path.join(REAL, 'vocab.txt'), 'utf8'));
const blob = fs.readFileSync(path.join(REAL, 'encoder-weights.bin'));
const M = await createModule();
const blobPtr = M._malloc(blob.length);
M.HEAPU8.set(blob, blobPtr);
const idsPtr = M._malloc(128 * 4);
const outPtr = M._malloc(128 * D * 4);

let encodeMs = 0;
function encode(text) {
    const ids = tokenizer.encode(text);
    new Int32Array(M.HEAP32.buffer, idsPtr, ids.length).set(ids);
    const t0 = performance.now();
    const rc = M._encoder_forward(blobPtr, idsPtr, ids.length, outPtr, 0);
    encodeMs += performance.now() - t0;
    if (rc !== ids.length) throw new Error(`encoder_forward failed: ${rc}`);
    const hidden = new Float32Array(M.HEAPF32.buffer, outPtr, ids.length * D);
    const pooled = new Float32Array(D);
    for (let t = 0; t < ids.length; t++) for (let d = 0; d < D; d++) pooled[d] += hidden[t * D + d];
    let norm = 0;
    for (let d = 0; d < D; d++) { pooled[d] /= ids.length; norm += pooled[d] ** 2; }
    norm = Math.sqrt(norm);
    for (let d = 0; d < D; d++) pooled[d] /= norm;
    return pooled;
}

const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
const groundTruth = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8'));

const search = await openPikeletFile(path.join(here, '..', 'pancake-wiki.pancake'), {
    encodeQuery: async (text) => encode(text),
});
let hits = 0;
for (let i = 0; i < evalQueries.length; i++) {
    const out = await search.query(evalQueries[i].text, { k: 10 });
    const truth = new Set(groundTruth[i]);
    hits += out.results.filter((r) => truth.has(r.id)).length;
}
await search.close();

const recall = hits / (evalQueries.length * 10);
console.log(`WASM inline encoder end to end (tokenize -> fused-u8 forward -> container):`);
console.log(`  recall@10 = ${(recall * 100).toFixed(1)}%   (fake-quant sim 82.4%, fp32 teacher 82.8%)`);
console.log(`  mean encode ${(encodeMs / evalQueries.length).toFixed(1)} ms/query, single thread`);
