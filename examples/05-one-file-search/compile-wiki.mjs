#!/usr/bin/env node
// Compile the Simple English Wikipedia pack (examples/04-static-wiki-pack,
// 456k chunks) into one .pikelet — the complete profile's scale test, and
// its first kind-2 (pinned-external encoder) artifact: the MiniLM encoder
// is declared and verifiable, not embedded (contract section 4.4 mode 2);
// the host executes it, exactly as the deployed wiki demo already does.
//
//   node compile-wiki.mjs                      # -> pancake-wiki.pancake (~535 MB)
//   node compile.mjs --inspect pancake-wiki.pancake

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildQueryInterpSegment, buildCorpusSegment, buildLexicalSegment, assemblePikeletFile, PROFILE_V2 } from '../../complete/builder.mjs';
import { inspect } from './compile.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
// The canonical pack layout is data-perm (k-means cluster order — a
// query's rerank candidates land physically adjacent; see the pack README's
// layout section). data-full is the unpermuted source; building from it
// silently forfeits the layout, which is exactly what happened to every
// complete artifact before 2026-08-31 (~440 scattered requests/query
// instead of ~200 at the recommended gap).
const PERM = path.join(here, '..', '04-static-wiki-pack', 'data-perm');
const FULL = path.join(here, '..', '04-static-wiki-pack', 'data-full');
const DATA = fs.existsSync(path.join(PERM, 'wiki.pikelet-sketch')) ? PERM : FULL;
if (DATA === FULL) {
    console.warn('WARNING: building from data-full (unpermuted layout) — rerank candidates will be physically scattered; build the pack in data-perm first (pack README steps 3-4)');
} else {
    console.log('building from data-perm (cluster-ordered layout)');
}

function buildWikiCorpusSegment() {
    // The pack already stores the corpus as JSONL bytes plus a u32
    // count+1 offsets index; the container form is the same records behind
    // u64 offsets plus per-record digests (spec 3.5, layout records-v2).
    // Records pass through untouched — sliced as views, no copy until the
    // segment is assembled.
    const corpusBytes = fs.readFileSync(path.join(DATA, 'corpus.bin'));
    const u32 = fs.readFileSync(path.join(DATA, 'corpus-offsets.u32'));
    const packOffsets = new Uint32Array(u32.buffer, u32.byteOffset, u32.byteLength / 4);
    const count = packOffsets.length - 1;
    if (packOffsets[count] !== corpusBytes.length) {
        throw new Error(`corpus offsets end ${packOffsets[count]} != corpus bytes ${corpusBytes.length}`);
    }
    const records = new Array(count);
    for (let i = 0; i < count; i++) {
        records[i] = corpusBytes.subarray(packOffsets[i], packOffsets[i + 1]);
    }
    const built = buildCorpusSegment(records);
    return { segment: built.bytes, corpus: built.corpus, count, records };
}

// kind 3: the teacher compiled in — declaration + vocab + weight blob as
// segment data (kernels live in the reader, per spec 3.6).
async function buildInlineQueryInterp(packManifest) {
    const SPIKE = path.join(here, 'encoder-spike', 'real');
    const blobPath = path.join(SPIKE, 'encoder-weights.bin');
    const vocabPath = path.join(SPIKE, 'vocab.txt');
    if (!fs.existsSync(blobPath) || !fs.existsSync(vocabPath)) {
        throw new Error('inline encoder assets missing — run encoder-spike/export_encoder_blob.py first');
    }
    const vocab = fs.readFileSync(vocabPath);
    const blob = fs.readFileSync(blobPath);
    const declarationFields = {
        kind: 'inline-transformer-v1',
        model: packManifest.model,
        license: 'apache-2.0',
        attribution: 'sentence-transformers/all-MiniLM-L6-v2 (quantized derivative)',
        dim: packManifest.dim,
        pooling: packManifest.pooling,
        normalized: packManifest.normalized,
        // NOT packManifest.maxTokens: the pack's 256 describes the teacher
        // that embedded the passages. This kernel's compiled window is 128
        // (encoder.cpp MAXSEQ) and longer inputs are mean-pooled across
        // [CLS]…[SEP] windows — the declaration states what the embedded
        // encoder does.
        maxTokens: 128,
        longInputs: 'windowed-mean-pool',
        layout: { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 6, B: 64, H: 12 },
    };
    // Contract section 4.4 mode 1: embed verification vectors produced by
    // this very kernel+blob, so readers can prove theirs matches.
    const { createInlineTransformerEmbedder, buildInlineTestVectors } =
        await import('../../complete/inline-transformer.mjs');
    const createEncoder = (await import('../../complete/encoder-kernels/encoder.node.mjs')).default;
    const embedder = await createInlineTransformerEmbedder({
        declaration: declarationFields,
        vocabText: vocab.toString('utf8'),
        blob,
        createEncoder,
    });
    try {
        declarationFields.testVectors = await buildInlineTestVectors(embedder);
    } finally {
        embedder.dispose();
    }
    const declaration = Buffer.from(JSON.stringify(declarationFields), 'utf8');
    const encoder = Buffer.alloc(12 + declaration.length + vocab.length + blob.length);
    encoder.writeUInt32LE(declaration.length, 0);
    encoder.writeUInt32LE(vocab.length, 4);
    encoder.writeUInt32LE(blob.length, 8);
    declaration.copy(encoder, 12);
    vocab.copy(encoder, 12 + declaration.length);
    blob.copy(encoder, 12 + declaration.length + vocab.length);

    const calibration = Buffer.from(JSON.stringify({
        kind: 'retrieval-signals-v1',
        asset: JSON.parse(fs.readFileSync(path.join(DATA, 'wiki-abstention.json'), 'utf8')),
        vocabBloomBase64: fs.readFileSync(path.join(DATA, 'wiki-vocab.bloom')).toString('base64'),
    }), 'utf8');
    return buildQueryInterpSegment(3, encoder, calibration);
}

function buildWikiQueryInterp(packManifest) {
    // Encoder declaration (kind 2): pin the external model and carry
    // verification vectors — the first hand-written eval queries with their
    // exact MiniLM embeddings, so a host encoder is checkable before serving.
    const evalQueries = JSON.parse(fs.readFileSync(path.join(DATA, 'eval-queries.json'), 'utf8'));
    const vectors = fs.readFileSync(path.join(DATA, 'eval-queries.f32'));
    const dim = packManifest.dim;
    const queryTexts = evalQueries.queries || evalQueries;
    const handWritten = queryTexts
        .map((q, i) => ({ q, i }))
        .filter(({ q }) => (typeof q === 'object' ? q.kind === 'hand' || q.source === 'hand' : false));
    const testPool = handWritten.length >= 3 ? handWritten : queryTexts.map((q, i) => ({ q, i })).slice(0, 3);
    const testVectors = testPool.slice(0, 3).map(({ q, i }) => ({
        text: typeof q === 'object' ? q.text : q,
        embedding: Array.from(new Float32Array(vectors.buffer, vectors.byteOffset + i * dim * 4, dim))
            .map((v) => Number(v.toFixed(6))),
        tolerance: 1e-3,
    }));
    const encoderDeclaration = Buffer.from(JSON.stringify({
        kind: 'external-transformers-v1',
        model: packManifest.model,
        dim,
        pooling: packManifest.pooling,
        normalized: packManifest.normalized,
        maxTokens: packManifest.maxTokens,
        testVectors,
    }), 'utf8');

    const calibration = Buffer.from(JSON.stringify({
        kind: 'retrieval-signals-v1',
        asset: JSON.parse(fs.readFileSync(path.join(DATA, 'wiki-abstention.json'), 'utf8')),
        vocabBloomBase64: fs.readFileSync(path.join(DATA, 'wiki-vocab.bloom')).toString('base64'),
    }), 'utf8');

    return buildQueryInterpSegment(2, encoderDeclaration, calibration);
}

const packManifest = JSON.parse(fs.readFileSync(path.join(DATA, 'pack-manifest.json'), 'utf8'));
const { segment: corpusSegment, corpus: corpusFields, count, records: corpusRecords } = buildWikiCorpusSegment();
if (count !== packManifest.chunks) {
    throw new Error(`corpus count ${count} != pack manifest chunks ${packManifest.chunks}`);
}

const evaluationSegment = Buffer.from(JSON.stringify({
    packEvaluation: packManifest.evaluation,
    groundTruth: JSON.parse(fs.readFileSync(path.join(DATA, 'eval-gt.json'), 'utf8')),
    abstentionProbes: JSON.parse(fs.readFileSync(path.join(DATA, 'wiki-abstention-probes.json'), 'utf8')),
}), 'utf8');

const inline = process.argv.includes('--inline-encoder');
const args = process.argv.slice(2).filter((a) => a !== '--inline-encoder');

// Lexical index for hybrid retrieval (kind 5): BM25 candidates join the
// sketch rerank so known-item title lookups survive the scan's top-C
// cutoff — the measured recall gap at this scale. Built from the record
// texts; the wiki-scale reader opens it lazily.
console.log('building lexical index...');
const lexicalTexts = corpusRecords.map((r) => {
    const row = JSON.parse(r.toString('utf8'));
    return `${row.title || ''}\n${row.text || ''}`;
});
const lexical = buildLexicalSegment(lexicalTexts);
console.log(`  ${lexical.meta.terms.toLocaleString()} terms over ${lexical.meta.docCount.toLocaleString()} records (${(lexical.bytes.length / 1048576).toFixed(1)} MiB)`);

const segments = [
    { kind: 'index', bytes: fs.readFileSync(path.join(DATA, 'wiki.pikelet-sketch')) },
    { kind: 'corpus', bytes: corpusSegment },
    { kind: 'lexical', bytes: lexical.bytes },
    {
        kind: 'query-interp',
        bytes: inline ? await buildInlineQueryInterp(packManifest) : buildWikiQueryInterp(packManifest),
    },
    { kind: 'evaluation', bytes: evaluationSegment },
];

const outPath = args[0] || path.join(here, inline ? 'pancake-wiki-inline.pancake' : 'pancake-wiki.pancake');
const result = assemblePikeletFile({
    profile: PROFILE_V2,
    corpus: {
        ...corpusFields,
        provenance: { dataset: packManifest.dataset, articles: packManifest.articles },
    },
    dim: packManifest.dim,
    metric: packManifest.metric,
    encoder: {
        kind: inline ? 'inline-transformer-v1' : 'external-transformers-v1',
        model: packManifest.model,
        pooling: packManifest.pooling,
        normalized: packManifest.normalized,
        // Inline: the embedded kernel's true window, not the teacher's.
        maxTokens: inline ? 128 : packManifest.maxTokens,
    },
    recommendedRerank: packManifest.recommendedRerank,
    // Cluster-ordered rows only coalesce at a gap matched to the cluster
    // geometry; readers honor this hint unless the caller overrides.
    recommendedGap: 16384,
    lexical: lexical.meta,
    sampleQueries: ['who was the first person on the moon', 'how do volcanoes form'],
}, segments, outPath);

console.log(`compiled ${result.outPath}`);
console.log(`  ${(result.fileBytes / 1048576).toFixed(2)} MiB, identity ${result.identity.slice(0, 16)}...`);
for (const s of result.manifest.segments) {
    console.log(`  ${s.kind.padEnd(13)} ${(s.bytes / 1048576).toFixed(2).padStart(8)} MiB  ${s.sha256.slice(0, 12)}...`);
}
