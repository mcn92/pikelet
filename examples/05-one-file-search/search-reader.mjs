// Spike: the composed Search Artifact reader — one call from query text to
// hydrated, calibrated results, over the five components that
// examples/legacy/03-edge-docs-search ships as separate Worker assets:
//
//   corpus      docs-corpus.json       (records: title/text/preview/anchor/path)
//   index       docs-index.bin         (.pnck snapshot -> sketch tier at open)
//   encoder     docs-student.bin       (distilled int8 student)
//   calibration docs-abstention.json   (feature weights + thresholds)
//   manifest    docs-manifest.json     (component naming + encoder config)
//
// This file exists to answer the composition questions ahead of the complete
// artifact profile (SEARCH_ARTIFACT_CONTRACT.md section 9.4) — its API is a
// draft of the future one-file reader, not a published surface. The
// abstention helpers are extracted verbatim from 03's worker.js so the
// committed golden fixtures transfer as acceptance tests.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import {
    loadStudentModel,
    embedTextWithStudent,
} from '../legacy/03-edge-docs-search/student-embedder.mjs';

const require = createRequire(import.meta.url);
const Pikelet = require('../../pikelet.js');
const { buildSketchArtifact } = require('../../pikelet-artifact.js');

// Abstention scoring lives in abstention.mjs (shared with the one-file
// reader); re-exported here for compatibility.
import { computeMatchQuality, computePreSearchAbstention } from './abstention.mjs';
export { computeMatchQuality, computePreSearchAbstention };

// --- The v3 snapshot envelope maps internal to external ids; the sketch
// profile binds ids positionally, so this reader requires the identity
// mapping (true for any freshly built, never-compacted snapshot).
// CONTAINER LESSON: the complete profile must either carry the id map as a
// segment or require identity mapping of its index segment.
export function assertIdentityMapping(snapshotBytes) {
    const view = new DataView(snapshotBytes.buffer, snapshotBytes.byteOffset, snapshotBytes.byteLength);
    if (view.getUint32(0, true) !== 0x504e434b || view.getUint32(4, true) !== 3) return;
    const mappingCount = view.getUint32(24, true);
    for (let i = 0; i < mappingCount; i++) {
        if (view.getUint32(32 + i * 8, true) !== view.getUint32(32 + i * 8 + 4, true)) {
            throw new Error('openDocsSearch: snapshot has a non-identity id mapping; '
                + 'the sketch tier binds ids positionally, so corpus hydration would be wrong');
        }
    }
}

/**
 * Open a composed docs search over the five component files.
 * Returns { query(text, {k}), close(), info() }.
 */
export async function openDocsSearch({ manifestPath, indexPath, encoderPath, calibrationPath, corpusPath }) {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const abstention = JSON.parse(fs.readFileSync(calibrationPath, 'utf8'));
    const corpusRaw = JSON.parse(fs.readFileSync(corpusPath, 'utf8'));
    const corpusById = new Map(Object.entries(corpusRaw).map(([id, record]) => [Number(id), record]));
    const student = loadStudentModel(fs.readFileSync(encoderPath));

    const snapshotBytes = fs.readFileSync(indexPath);
    assertIdentityMapping(snapshotBytes);

    // Derive the sketch tier from the snapshot at open. Written through a
    // temp file because the builder is currently path-based — CONTAINER
    // LESSON: buildSketchArtifact needs a bytes-in/bytes-out variant before
    // the compiler can assemble segments without touching disk.
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pancake-one-file-'));
    const sketchPath = path.join(tmpDir, 'index.pancake-sketch');
    buildSketchArtifact(snapshotBytes, sketchPath, {
        recommendedRerank: manifest.efSearch || 120,
    });
    const sketch = await Pikelet.openSketchArtifactFile(sketchPath);
    if (sketch.count !== corpusById.size) {
        throw new Error(`openDocsSearch: index count ${sketch.count} != corpus records ${corpusById.size}`);
    }

    return {
        info() {
            return {
                chunks: sketch.count,
                dim: sketch.dim,
                encoder: manifest.encoder || null,
                residentBytes: sketch.residentBytes,
                residentVerified: sketch.stats().residentVerified,
                sampleQueries: manifest.sampleQueries || [],
            };
        },

        async query(text, options = {}) {
            const k = Number.isInteger(options.k) && options.k > 0 ? options.k : 5;
            const trimmed = String(text || '').trim();
            if (!trimmed) throw new Error('query text is required');

            const embedded = embedTextWithStudent(trimmed, student);
            const pre = computePreSearchAbstention(embedded, abstention);
            let hits = [];
            if (!pre) {
                hits = (await sketch.search(embedded.vector, k)).results;
            }
            const quality = pre || computeMatchQuality(hits, embedded, abstention);
            const returned = quality.match_quality === 'none' ? [] : hits;

            return {
                matchQuality: quality.match_quality,
                confidence: quality.confidence,
                results: returned.map((hit) => {
                    const chunk = corpusById.get(hit.id);
                    return {
                        id: hit.id,
                        distance: hit.distance,
                        title: chunk?.title || `Chunk ${hit.id}`,
                        preview: chunk?.preview || '',
                        text: chunk?.text || '',
                        sourcePath: chunk?.sourcePath || '',
                        anchor: chunk?.anchor || '',
                    };
                }),
            };
        },

        async close() {
            await sketch.close();
            fs.rmSync(tmpDir, { recursive: true, force: true });
        },
    };
}

/** Convenience: open directly over 03's committed assets. */
export function docsAssetPaths() {
    const assets = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'legacy', '03-edge-docs-search', 'assets');
    return {
        manifestPath: path.join(assets, 'docs-manifest.json'),
        indexPath: path.join(assets, 'docs-index.bin'),
        encoderPath: path.join(assets, 'docs-student.bin'),
        calibrationPath: path.join(assets, 'docs-abstention.json'),
        corpusPath: path.join(assets, 'docs-corpus.json'),
    };
}
