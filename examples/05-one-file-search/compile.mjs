#!/usr/bin/env node
// Compile the five Search Artifact components into one .pikelet file per
// spec/COMPLETE_PROFILE.md (Draft 1), and inspect/verify the result:
//
//   node compile.mjs                 # compile 03's assets -> pikelet-docs.pikelet
//   node compile.mjs --inspect pikelet-docs.pikelet
//
// This is the kind-1 (student-inline) compiler over the docs corpus; see
// compile-wiki.mjs for the kind-2 (pinned-external encoder) wiki-scale
// compiler. Shared layout logic lives in pikelet/src/complete-profile.mjs.

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { assertIdentityMapping, docsAssetPaths } from './search-reader.mjs';
import {
    MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KIND_NAMES,
    sha256, buildQueryInterpSegment, buildCorpusSegment, assemblePikeletFile,
    measureRecommendedRerank, PROFILE_V2, FORMAT_VERSIONS,
} from '../../complete/builder.mjs';

const require = createRequire(import.meta.url);
const Pikelet = require('../../pikelet.js');
const Artifact = require('../../pikelet-artifact.js');
const here = path.dirname(fileURLToPath(import.meta.url));

// The full student evaluation is per-row and large; the evaluation segment
// carries the golden queries plus the evaluation's scalar/summary fields
// (arrays over 100 entries dropped, noted by key).
function buildEvaluationSegment(goldenQueries, studentEvaluation, rerankSweep) {
    const summary = {};
    const dropped = [];
    for (const [key, value] of Object.entries(studentEvaluation)) {
        if (Array.isArray(value) && value.length > 100) dropped.push(key);
        else summary[key] = value;
    }
    return Buffer.from(JSON.stringify({
        goldenQueries,
        student: summary,
        studentFieldsOmitted: dropped,
        // Spec 5.4: the recall-vs-C measurements behind recommendedRerank.
        rerankSweep,
    }), 'utf8');
}

async function compile(paths, outPath) {
    const sourceManifest = JSON.parse(fs.readFileSync(paths.manifestPath, 'utf8'));
    const corpusRaw = JSON.parse(fs.readFileSync(paths.corpusPath, 'utf8'));
    const count = Object.keys(corpusRaw).length;
    const records = [];
    for (let id = 0; id < count; id++) {
        const r = corpusRaw[String(id)];
        if (!r) throw new Error(`corpus record ${id} missing — ids must be dense [0, count)`);
        records.push(Buffer.from(JSON.stringify({
            title: r.title, text: r.text, preview: r.preview,
            sourcePath: r.sourcePath, anchor: r.anchor,
        }), 'utf8'));
    }
    const snapshotBytes = fs.readFileSync(paths.indexPath);
    assertIdentityMapping(snapshotBytes);

    // Spec section 5: recommendedRerank is a measured operating point, not a
    // copy of efSearch. Sweep recall-vs-C on the artifact's own vectors
    // (03 ships no float query set, so queries are dequantized rows) and
    // bake the smallest C that reaches the target into the sketch header.
    const { bytes: provisionalSketch } = Pikelet.buildSketchArtifactBytes(snapshotBytes, {});
    const rerankSweep = await measureRecommendedRerank({
        artifactModule: Artifact,
        sketchBytes: provisionalSketch,
        snapshotBytes,
    });
    console.log(`measured rerank operating point: C=${rerankSweep.recommendedRerank} `
        + `(recall@${rerankSweep.k} ${rerankSweep.recall} over ${rerankSweep.queries} ${rerankSweep.querySource} queries)`);
    const { bytes: sketchBytes } = Pikelet.buildSketchArtifactBytes(snapshotBytes, {
        recommendedRerank: rerankSweep.recommendedRerank,
    });
    const goldenQueries = JSON.parse(fs.readFileSync(path.join(
        path.dirname(paths.manifestPath), '..', 'fixtures', 'abstention-golden.json'), 'utf8'));

    // Corpus layout v2: per-record digests behind a page table, so each
    // hydrated record verifies on its own range read (spec 3.5, Draft 2).
    const corpusSegment = buildCorpusSegment(records);
    const segments = [
        { kind: 'index', bytes: Buffer.from(sketchBytes) },
        { kind: 'corpus', bytes: corpusSegment.bytes },
        {
            kind: 'query-interp',
            bytes: buildQueryInterpSegment(1,
                fs.readFileSync(paths.encoderPath),
                fs.readFileSync(paths.calibrationPath)),
        },
        {
            kind: 'evaluation',
            bytes: buildEvaluationSegment(goldenQueries,
                JSON.parse(fs.readFileSync(paths.manifestPath.replace('docs-manifest', 'docs-student-evaluation'), 'utf8')),
                rerankSweep),
        },
    ];

    return assemblePikeletFile({
        profile: PROFILE_V2,
        corpus: { ...corpusSegment.corpus, provenance: null },
        dim: sourceManifest.dim,
        metric: sourceManifest.metric,
        encoder: { kind: 'student-inline-v1', ...sourceManifest.encoder },
        recommendedRerank: rerankSweep.recommendedRerank,
        sampleQueries: sourceManifest.sampleQueries || [],
    }, segments, outPath);
}

export function inspect(filePath) {
    const bytes = fs.readFileSync(filePath);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (view.getUint32(0, true) !== MAGIC) throw new Error('not a .pikelet file (bad magic)');
    const formatVersion = view.getUint32(4, true);
    if (!Object.values(FORMAT_VERSIONS).includes(formatVersion)) throw new Error(`unsupported format version ${formatVersion}`);
    const manifestBytes = view.getUint32(8, true);
    const segmentCount = view.getUint32(12, true);
    const fileBytes = Number(view.getBigUint64(16, true));
    if (fileBytes !== bytes.length) throw new Error(`fileBytes ${fileBytes} != actual ${bytes.length}`);

    const manifestBuf = bytes.subarray(HEADER_BYTES, HEADER_BYTES + manifestBytes);
    const manifestOk = sha256(manifestBuf).equals(bytes.subarray(24, 56));
    const manifest = JSON.parse(manifestBuf.toString('utf8'));

    console.log(`${path.basename(filePath)}: ${(fileBytes / 1048576).toFixed(2)} MiB, profile ${manifest.profile} (format ${formatVersion})`);
    console.log(`identity (manifest sha256): ${Buffer.from(bytes.subarray(24, 56)).toString('hex')}`);
    console.log(`manifest digest verifies: ${manifestOk}`);
    console.log(`corpus: ${manifest.corpus.records} records, ${manifest.dim}D ${manifest.metric}, encoder ${manifest.encoder?.kind || 'unknown'}`);
    console.log(`corpus integrity: ${manifest.corpus.layout === 'records-v2'
        ? `per-record sha256, ${manifest.corpus.pages} page(s) of ${manifest.corpus.pageRecords}, page table ${manifest.corpus.pageTableSha256.slice(0, 12)}...`
        : 'whole-segment digest only (layout v1)'}`);
    console.log('segments:');
    const tableOffset = HEADER_BYTES + manifestBytes;
    for (let i = 0; i < segmentCount; i++) {
        const entry = tableOffset + i * TABLE_ENTRY_BYTES;
        const kind = view.getUint32(entry, true);
        const offset = Number(view.getBigUint64(entry + 8, true));
        const length = Number(view.getBigUint64(entry + 16, true));
        const declared = manifest.segments[i];
        const digestOk = sha256(bytes.subarray(offset, offset + length)).toString('hex') === declared.sha256;
        const aligned = offset % 16 === 0;
        console.log(`  ${String(KIND_NAMES[kind] || kind).padEnd(13)} offset ${String(offset).padStart(10)}  `
            + `${(length / 1048576).toFixed(2).padStart(8)} MiB  digest ${digestOk ? 'ok' : 'MISMATCH'}  aligned ${aligned}`);
        if (!digestOk || !aligned || declared.kind !== KIND_NAMES[kind]) process.exitCode = 1;
    }
    if (!manifestOk) process.exitCode = 1;
}

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
if (invokedDirectly) {
    const args = process.argv.slice(2);
    if (args[0] === '--inspect') {
        inspect(args[1] || path.join(here, 'pikelet-docs.pikelet'));
    } else {
        const outPath = args[0] || path.join(here, 'pikelet-docs.pikelet');
        const result = await compile(docsAssetPaths(), outPath);
        console.log(`compiled ${result.outPath}`);
        console.log(`  ${(result.fileBytes / 1048576).toFixed(2)} MiB, identity ${result.identity.slice(0, 16)}...`);
        for (const s of result.manifest.segments) {
            console.log(`  ${s.kind.padEnd(13)} ${(s.bytes / 1024).toFixed(1).padStart(10)} KiB  ${s.sha256.slice(0, 12)}...`);
        }
    }
}
