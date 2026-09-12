#!/usr/bin/env node
// Retrieval-level verdict for the inline-transformer simulation: run the
// 200 wiki eval queries through pancake-wiki.pancake three ways — the
// pack's shipped teacher embeddings, the fp32 re-encode, and the
// block-u8 fake-quantized six-layer forward — and compare recall@10
// against exact ground truth.

import fs from 'node:fs';
import path from 'node:path';
import { openPikeletFile } from '../pikelet-file-reader.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const DATA = path.join(here, '..', '..', '04-static-wiki-pack', 'data-full');
const dim = 384;

const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
const groundTruth = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8'));

function loadVectors(file) {
    const raw = fs.readFileSync(file);
    return (i) => new Float32Array(raw.buffer.slice(raw.byteOffset + i * dim * 4, raw.byteOffset + (i + 1) * dim * 4));
}

const sources = [
    ['pack teacher ', (i) => loadVectors(path.join(DATA, 'eval-queries.f32'))(i)],
    ['fp32 re-enc  ', loadVectors(path.join(here, 'real', 'fp32-queries.f32'))],
    ['quantized u8 ', loadVectors(path.join(here, 'real', 'quant-queries.f32'))],
];

for (const [label, vecFor] of sources) {
    const search = await openPikeletFile(path.join(here, '..', 'pancake-wiki.pancake'), {
        encodeQuery: async (text) => vecFor(evalQueries.findIndex((q) => q.text === text)),
    });
    let hits = 0;
    for (let i = 0; i < evalQueries.length; i++) {
        const out = await search.query(evalQueries[i].text, { k: 10 });
        const truth = new Set(groundTruth[i]);
        hits += out.results.filter((r) => truth.has(r.id)).length;
    }
    await search.close();
    console.log(`${label} recall@10 = ${((hits / (evalQueries.length * 10)) * 100).toFixed(1)}%`);
}
