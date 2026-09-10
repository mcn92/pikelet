#!/usr/bin/env node
// Wiki student pilot, step 2: score the distilled encoder against the
// teacher on the pack's 200 eval queries, two ways:
//   1. embedding fidelity — cosine similarity student vs teacher (the
//      teacher embeddings ship with the pack, so no model is loaded);
//   2. end-to-end recall@10 through pancake-wiki.pancake with the student
//      as the host encoder, vs exact brute-force ground truth — directly
//      comparable to the teacher's 82.9% through the same container.
// Splits by query source (sampled titles vs hand-written) because the
// expected open-domain failure mode is paraphrase degradation.

import fs from 'node:fs';
import path from 'node:path';
import { loadStudentModel, embedTextWithStudent } from '../legacy/03-edge-docs-search/student-embedder.mjs';
import { openPancakeFile } from './pikelet-file-reader.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-full');
const PILOT = path.join(here, 'student-pilot');

const student = loadStudentModel(fs.readFileSync(path.join(PILOT, 'student-model.bin')));
const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
const groundTruth = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8'));
const teacherRaw = fs.readFileSync(path.join(DATA, 'eval-queries.f32'));
const dim = 384;

const cosine = (a, b) => {
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += a[d] * b[d];
    return dot; // both sides unit-normalized
};

// 1. Embedding fidelity vs teacher.
const bySource = new Map();
for (let i = 0; i < evalQueries.length; i++) {
    const teacher = new Float32Array(teacherRaw.buffer.slice(
        teacherRaw.byteOffset + i * dim * 4, teacherRaw.byteOffset + (i + 1) * dim * 4));
    const sim = cosine(embedTextWithStudent(evalQueries[i].text, student).vector, teacher);
    const source = evalQueries[i].source || 'unknown';
    if (!bySource.has(source)) bySource.set(source, []);
    bySource.get(source).push(sim);
}
console.log('student-vs-teacher embedding cosine (200 eval queries):');
for (const [source, sims] of bySource) {
    const mean = sims.reduce((s, v) => s + v, 0) / sims.length;
    const sorted = [...sims].sort((a, b) => a - b);
    console.log(`  ${source.padEnd(8)} n=${String(sims.length).padStart(3)}  `
        + `mean ${mean.toFixed(4)}  p10 ${sorted[Math.floor(sims.length * 0.1)].toFixed(4)}  `
        + `min ${sorted[0].toFixed(4)}`);
}

// 2. End-to-end recall@10 through the container, student as host encoder.
const search = await openPancakeFile(path.join(here, 'pancake-wiki.pancake'), {
    encodeQuery: async (text) => embedTextWithStudent(text, student).vector,
});
const recallBySource = new Map();
for (let i = 0; i < evalQueries.length; i++) {
    const out = await search.query(evalQueries[i].text, { k: 10 });
    const truth = new Set(groundTruth[i]);
    const recall = out.results.filter((r) => truth.has(r.id)).length / 10;
    const source = evalQueries[i].source || 'unknown';
    if (!recallBySource.has(source)) recallBySource.set(source, []);
    recallBySource.get(source).push(recall);
}
await search.close();

console.log('\nrecall@10 through pancake-wiki.pancake (teacher baseline 82.9% overall):');
let totalHits = 0;
let totalQueries = 0;
for (const [source, recalls] of recallBySource) {
    const mean = recalls.reduce((s, v) => s + v, 0) / recalls.length;
    totalHits += recalls.reduce((s, v) => s + v, 0);
    totalQueries += recalls.length;
    console.log(`  ${source.padEnd(8)} n=${String(recalls.length).padStart(3)}  recall@10 ${(mean * 100).toFixed(1)}%`);
}
console.log(`  ${'overall'.padEnd(8)} n=${String(totalQueries).padStart(3)}  recall@10 ${((totalHits / totalQueries) * 100).toFixed(1)}%`);
