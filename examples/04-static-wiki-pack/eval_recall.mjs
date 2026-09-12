#!/usr/bin/env node
// Recall shootout: search the full pack with queries encoded at each dtype
// and score recall@10 against exact fp32 float brute force (eval-gt.json).
// The fp32 row is the pack's ceiling (u8 storage + sketch + rerank losses);
// fp16/q8 deltas below it isolate the encoder-compression cost.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from '@huggingface/transformers';
import Pikelet from '../../pikelet.node.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(here, process.argv[2] || 'data-full');
const K = 10;
const RERANK = Number(process.argv[3] || 300);

const queries = JSON.parse(fs.readFileSync(path.join(dataDir, 'eval-queries.json'), 'utf8'));
const gt = JSON.parse(fs.readFileSync(path.join(dataDir, 'eval-gt.json'), 'utf8'));
const EVAL_K = gt[0]?.length || K;

const artifact = await Pikelet.openSketchArtifactFile(path.join(dataDir, 'wiki.pikelet-sketch'));
const scanner = await Pikelet.createSketchScanner(artifact);

async function evalDtype(dtype) {
    const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype });
    let hits = 0;
    let handHits = 0;
    let handTotal = 0;
    const t0 = performance.now();
    for (let i = 0; i < queries.length; i++) {
        const out = await embed(queries[i].text, { pooling: 'mean', normalize: true });
        const { results } = await artifact.search(Float32Array.from(out.data), EVAL_K, { rerank: RERANK, scanner });
        const got = new Set(results.map((r) => r.id));
        const q = gt[i].filter((id) => got.has(id)).length;
        hits += q;
        if (queries[i].source === 'hand') { handHits += q; handTotal += EVAL_K; }
    }
    const recall = hits / (queries.length * EVAL_K);
    const handPart = handTotal ? `  (hand-written only: ${((handHits / handTotal) * 100).toFixed(2)}%)` : '';
    console.log(`${dtype.padEnd(5)} recall@${EVAL_K}: ${(recall * 100).toFixed(2)}%${handPart}  [${((performance.now() - t0) / queries.length).toFixed(0)} ms/q total]`);
    return recall;
}

console.log(`pack: ${artifact.count} chunks, rerank C=${RERANK}, ${queries.length} queries vs exact fp32 float top-${EVAL_K}`);
await evalDtype('fp32');
await evalDtype('fp16');
await evalDtype('q8');
scanner.dispose();
await artifact.close();
