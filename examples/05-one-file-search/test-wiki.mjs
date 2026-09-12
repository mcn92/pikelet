#!/usr/bin/env node
// Scale acceptance for the wiki .pikelet (456k chunks, kind-2 external
// encoder). The 200 eval queries' precomputed MiniLM embeddings act as the
// host encoder, so the whole container path — manifest verify, embedded
// sketch at offset, corpus hydration, retrieval-signal abstention — is
// exercised without loading a model. Pass bar: reproduce the pack
// fp32 augmented recall@10 at the recommended rerank (0.960).

import fs from 'node:fs';
import path from 'node:path';
import { openPikeletFile } from './pikelet-file-reader.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
// data-perm: the pack's canonical cluster-ordered layout; its eval ground
// truth is in permuted id space, matching artifacts compiled from it.
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-perm');
const pikeletPath = path.join(here, 'pancake-wiki.pancake');

let passed = 0;
let failed = 0;
const check = (name, ok, detail) => {
    if (ok) { passed++; console.log(`  ok: ${name}`); }
    else { failed++; console.log(`  FAIL: ${name}${detail ? ` — ${detail}` : ''}`); }
};

// Host encoder from precomputed eval embeddings (text -> vector lookup).
const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
const vectorsRaw = fs.readFileSync(path.join(DATA, 'eval-queries.f32'));
const groundTruth = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8'));
const dim = 384;
const vectorFor = new Map(evalQueries.map((q, i) => [q.text,
    new Float32Array(vectorsRaw.buffer.slice(vectorsRaw.byteOffset + i * dim * 4,
        vectorsRaw.byteOffset + (i + 1) * dim * 4))]));

const openStart = performance.now();
const search = await openPikeletFile(pikeletPath, {
    encodeQuery: async (text) => {
        const vector = vectorFor.get(text);
        if (!vector) throw new Error(`no precomputed embedding for "${text}"`);
        return vector;
    },
});
const openMs = performance.now() - openStart;
const info = search.info();
console.log(`opened ${path.basename(pikeletPath)} in ${(openMs / 1000).toFixed(1)}s: `
    + `${info.records.toLocaleString()} records, ${(info.fileBytes / 1048576).toFixed(0)} MiB file, `
    + `resident ${(info.residentBytes / 1048576).toFixed(1)} MiB, sketch hash verified: ${info.residentVerified}`);

check('identity + manifest verified at open', info.residentVerified === true
    && info.encoder.kind === 'external-transformers-v1' && info.records === 456153);

// Encoder declaration carries usable verification vectors.
check('encoder declaration pins model + test vectors',
    info.encoder.model === 'sentence-transformers/all-MiniLM-L6-v2'
    && Array.isArray(info.encoder.testVectors) && info.encoder.testVectors.length === 3
    && info.encoder.testVectors.every((v) => v.embedding.length === dim));

// Recall@10 at the recommended rerank across all 200 eval queries, scored
// against the pack's exact-brute-force ground truth.
const t0 = performance.now();
let hits10 = 0;
let answered = 0;
for (let i = 0; i < evalQueries.length; i++) {
    // augmented: lexical candidates join the exact rerank, distance order —
    // the mode this exact-NN metric measures (RRF-ordered hybrid trades
    // exact-NN overlap for keyword relevance by construction).
    const out = await search.query(evalQueries[i].text, { k: 10, retrieval: 'augmented' });
    const truth = new Set(groundTruth[i]);
    for (const r of out.results) if (truth.has(r.id)) hits10++;
    if (out.matchQuality !== 'none') answered++;
}
const queryMs = (performance.now() - t0) / evalQueries.length;
const recall = hits10 / (evalQueries.length * 10);
console.log(`  recall@10 over ${evalQueries.length} queries: ${(recall * 100).toFixed(1)}% `
    + `(fp32 reference 96.0% augmented at the recommended C=200); mean ${queryMs.toFixed(0)} ms/query`);
check('recall@10 reproduces the pack evaluation (>= 0.95)', recall >= 0.95, recall.toFixed(4));
check('in-domain queries are answered, not abstained', answered >= evalQueries.length * 0.95,
    `${answered}/${evalQueries.length}`);

// Hydration sanity: top hit for a title query resolves to a real record.
const sample = await search.query(evalQueries[0].text, { k: 3 });
check('hydrated records carry title/url/text', sample.results.length === 3
    && sample.results.every((r) => r.title && r.url && r.text));

// Abstention: nonsense embedding routed through the retrieval-signal scorer.
const noise = new Float32Array(dim);
let state = 99;
for (let d = 0; d < dim; d++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    noise[d] = (state / 0xffffffff) * 2 - 1;
}
let norm = 0;
for (let d = 0; d < dim; d++) norm += noise[d] * noise[d];
for (let d = 0; d < dim; d++) noise[d] /= Math.sqrt(norm);
vectorFor.set('zzqx blorptt nym vex', noise);
const nonsense = await search.query('zzqx blorptt nym vex', { k: 10 });
check('nonsense query abstains or downgrades', nonsense.matchQuality !== 'strong',
    `got ${nonsense.matchQuality} (${nonsense.confidence?.toFixed(3)})`);

await search.close();
console.log(`\nWiki .pikelet acceptance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
