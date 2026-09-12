#!/usr/bin/env node
// Phase 1, step 3: score the category specialist on ITS OWN eval queries
// (>= 8/10 ground truth inside the category), searching the FULL 456k-chunk
// index through the container — the honest mixture-of-experts setting:
// global index, specialist encoder. Teacher recall on the same subset is
// recomputed for an apples-to-apples baseline.

import fs from 'node:fs';
import path from 'node:path';
import { loadStudentModel, embedTextWithStudent } from '../legacy/03-edge-docs-search/student-embedder.mjs';
import { openPikeletFile } from './pikelet-file-reader.mjs';

const here = path.dirname(new URL(import.meta.url).pathname);
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-full');
const PILOT = path.join(here, 'student-pilot');

const meta = JSON.parse(fs.readFileSync(path.join(PILOT, 'category-meta.json'), 'utf8'));
const student = loadStudentModel(fs.readFileSync(path.join(PILOT, 'category', 'student-model.bin')));
const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
const groundTruth = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8'));
const teacherRaw = fs.readFileSync(path.join(DATA, 'eval-queries.f32'));
const dim = 384;

const teacherVec = (i) => new Float32Array(teacherRaw.buffer.slice(
    teacherRaw.byteOffset + i * dim * 4, teacherRaw.byteOffset + (i + 1) * dim * 4));

const subset = meta.evalQueryIndices;
console.log(`category ${meta.cluster}: ${meta.chunks.toLocaleString()} chunks, ${subset.length} owned eval queries`);

// Embedding fidelity on the subset.
let cosSum = 0;
for (const q of subset) {
    const s = embedTextWithStudent(evalQueries[q].text, student).vector;
    const t = teacherVec(q);
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += s[d] * t[d];
    cosSum += dot;
}
console.log(`student-vs-teacher cosine on owned queries: mean ${(cosSum / subset.length).toFixed(4)}`);

// Recall@10 through the container, both encoders, same subset.
async function recallWith(encodeQuery, label) {
    const search = await openPikeletFile(path.join(here, 'pancake-wiki.pancake'), { encodeQuery });
    let hits = 0;
    for (const q of subset) {
        const out = await search.query(evalQueries[q].text, { k: 10 });
        const truth = new Set(groundTruth[q]);
        hits += out.results.filter((r) => truth.has(r.id)).length;
    }
    await search.close();
    const recall = hits / (subset.length * 10);
    console.log(`${label} recall@10 on owned queries: ${(recall * 100).toFixed(1)}%`);
    return recall;
}

const teacher = await recallWith(async (text) => {
    const i = subset.find((q) => evalQueries[q].text === text) ?? evalQueries.findIndex((e) => e.text === text);
    return teacherVec(i);
}, 'teacher ');
const studentRecall = await recallWith(async (text) => embedTextWithStudent(text, student).vector, 'student ');
console.log(`\ngap: ${((teacher - studentRecall) * 100).toFixed(1)} points `
    + `(pilot's full-corpus student gapped ~68 points; docs-narrow parity is the target)`);
