#!/usr/bin/env node
// Build the wiki knowledge pack from embed_corpus.py output:
//
//   data/vectors.f32 + data/corpus.jsonl
//     -> data/wiki.pnck            (engine snapshot, cosine, u8 quantized)
//     -> data/wiki.pikelet-sketch  (resident-sketch artifact, the pack's index)
//     -> data/corpus.bin           (corpus chunks, one JSON row per line, as bytes)
//     -> data/corpus-offsets.u32   (chunk id -> [byteStart, byteEnd) into corpus.bin)
//
// The offsets file is what makes result hydration a range read: the demo
// fetches corpus.bin slices for the top-k ids only, never the whole corpus.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Pikelet from '../../pikelet.node.mjs';
import { stampPackVersion } from './stamp_pack_version.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(here, process.argv[2] || 'data');
const DIM = 384;
// From this corpus's own recall sweep (eval_recall.mjs, 2026-08-03):
// C=200 gives 95.65% recall@10 vs exact float truth (100% on hand-written
// natural questions) at ~66 requests / ~390 KiB per browser query; C=300
// adds 1.3 points for ~35% more requests. The abstention calibration is
// fitted at this depth and must move with it.
const RECOMMENDED_RERANK = 200;

const manifest = JSON.parse(fs.readFileSync(path.join(dataDir, 'corpus-manifest.json'), 'utf8'));
const vectorsBytes = fs.readFileSync(path.join(dataDir, 'vectors.f32'));
const count = vectorsBytes.byteLength / (DIM * 4);
if (!Number.isInteger(count) || count !== manifest.chunks) {
    throw new Error(`vectors.f32 holds ${count} rows but manifest says ${manifest.chunks}`);
}
const vectors = new Float32Array(vectorsBytes.buffer, vectorsBytes.byteOffset, count * DIM);

console.log(`building index: ${count} chunks, dim ${DIM}, cosine u8`);
const t0 = Date.now();
const index = await Pikelet.create({
    dim: DIM,
    maxElements: count,
    metric: 'cosine',
    quantized: true,
    M: 16,
    efConstruction: 200,
});
const batch = [];
for (let i = 0; i < count; i++) {
    batch.push(vectors.subarray(i * DIM, (i + 1) * DIM));
    if (batch.length === 10000) {
        index.addBatch(batch);
        batch.length = 0;
        process.stdout.write(`  ${i + 1}/${count}\r`);
    }
}
if (batch.length) index.addBatch(batch);
console.log(`\nindexed in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

const snapshotPath = path.join(dataDir, 'wiki.pnck');
fs.writeFileSync(snapshotPath, index.export());
index.dispose();

const sketchPath = path.join(dataDir, 'wiki.pikelet-sketch');
const sketchManifest = Pikelet.buildSketchArtifactFile(snapshotPath, sketchPath, {
    sketchDims: 192,    // 2:1 pooling of 384-D (4:1 measured -11pt candidate capture)
    sketchBits: 4,
    recommendedRerank: RECOMMENDED_RERANK,
});
console.log('sketch artifact:', JSON.stringify({
    sizeMB: +(sketchManifest.sizeBytes / 1e6).toFixed(1),
    residentMB: +(sketchManifest.sketch.residentBytes / 1e6).toFixed(1),
    recommendedRerank: sketchManifest.recommendedRerank,
}));

// Corpus hydration asset: corpus.bin is the jsonl bytes verbatim; the
// offsets table lets a reader turn chunk id -> one range read.
const corpusSrc = fs.readFileSync(path.join(dataDir, 'corpus.jsonl'));
fs.copyFileSync(path.join(dataDir, 'corpus.jsonl'), path.join(dataDir, 'corpus.bin'));
const offsets = new Uint32Array((count + 1));
let pos = 0;
let row = 0;
for (let i = 0; i < corpusSrc.length; i++) {
    if (corpusSrc[i] === 0x0a) {
        offsets[row + 1] = i + 1;
        row++;
    }
}
if (row !== count) throw new Error(`corpus.jsonl has ${row} rows, expected ${count}`);
fs.writeFileSync(path.join(dataDir, 'corpus-offsets.u32'), Buffer.from(offsets.buffer));

const packManifest = {
    ...manifest,
    metric: 'cosine',
    files: {
        index: 'wiki.pikelet-sketch',
        corpus: 'corpus.bin',
        corpusOffsets: 'corpus-offsets.u32',
    },
    sketch: sketchManifest.sketch,
    recommendedRerank: sketchManifest.recommendedRerank,
};
fs.writeFileSync(path.join(dataDir, 'pack-manifest.json'), JSON.stringify(packManifest, null, 2) + '\n');
const { packVersion } = stampPackVersion(dataDir);
console.log(`pack manifest written (packVersion ${packVersion})`);
