#!/usr/bin/env node
// Calibrate abstention for a static knowledge pack. Signals are retrieval-only
// (d0, margin, mean10) plus corpus vocabulary coverage, so the client can score
// any query from the results it already has.
//
// Labels:
//   answerable  — templated questions from retained article titles, verified by
//                 source-title retrieval
//   unanswerable— templated questions from held-out titles, scored against a
//                 reduced pack that excludes those titles by construction,
//                 foreign-bank titles that are absent from the pack, plus
//                 synthetic out-of-vocabulary gibberish
//   weak        — templated retained-title questions whose source title is
//                 retrieved at rank 5-50: adjacent content exists, but the
//                 top answer is not strong enough for an unqualified answer
//
// Fit: logistic regression P(answerable) on standardized signals. Thresholds:
//   hard — placed between the negative ceiling and answerable/weak floor
//   weak — placed between the weak ceiling and answerable floor when possible
//
// Negative labels must be verified against this pack. Title identity is only a
// proxy for answerability; foreign-bank negatives are dropped by retrieval
// strength so overlapping domains do not teach the scorer to abstain on useful
// matches.
//
// Output: data-perm/wiki-abstention.json + wiki-abstention-probes.json
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from '@huggingface/transformers';
import Pikelet from '../../pikelet.node.mjs';
import { stampPackVersion } from './stamp_pack_version.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = parseArgs(process.argv.slice(2));
const dataDir = path.resolve(here, args.dataDir);
const sourceManifest = JSON.parse(fs.readFileSync(path.join(dataDir, 'corpus-manifest.json'), 'utf8'));
const K = 10;
const RERANK = 200; // must match the demo's search config
const DIM = sourceManifest.dim || 384;
const HOLDOUT_RATE = Number(process.env.HOLDOUT_RATE || 0.02);
const FIT_QUERIES = Number(process.env.FIT_QUERIES || 160);
const HOLDOUT_QUERIES = Number(process.env.HOLDOUT_QUERIES || FIT_QUERIES);
const GIBBERISH_QUERIES = Number(process.env.GIBBERISH_QUERIES || 80);
const FOREIGN_QUERIES = Number(process.env.FOREIGN_QUERIES || 80);
const WEAK_QUERIES = Number(process.env.WEAK_QUERIES || 80);
const WEAK_SCAN_K = Number(process.env.WEAK_SCAN_K || 50);
const SEED = Number(process.env.ABSTENTION_SEED || 424242);
const CALIB_DIR = path.join(dataDir, '.abstention-calibration');

const FOREIGN_TITLE_BANK = [
    '1040 tax form earned income credit',
    'mortgage escrow shortage statement',
    'kubernetes crashloopbackoff',
    'react usestate batching',
    'postgres vacuum analyze',
    'aws iam role trust policy',
    'stripe webhook signature verification',
    'chase sapphire annual fee waiver',
    'tsa precheck renewal appointment',
    'iphone battery health service message',
    'netgear router admin password reset',
    'excel vlookup not available error',
    'docker compose port binding',
    'github actions cache miss',
    'python nonetype subscriptable error',
    'medicare part d formulary exception',
    'irs quarterly estimated tax payment',
    'student loan income driven repayment',
    'car alternator belt squeal',
    'ev charger nema 14-50 permit',
    'tenant security deposit demand letter',
    'small claims filing fee',
    'hsa eligible expense receipt',
    'merchant chargeback reason code',
    'oauth redirect uri mismatch',
    'terraform state lock',
    'nginx reverse proxy websocket upgrade',
    'redis eviction policy',
    'pandas dataframe groupby transform',
    'homeowners insurance deductible claim',
    'credit card balance transfer fee',
    'passport expedited renewal appointment',
    'property tax homestead exemption',
    'medical prior authorization denial',
    'printer offline windows settings',
    'wifi mesh backhaul channel',
    'airbnb host cancellation refund',
    'uber driver cancelled ride refund',
    'shopify abandoned cart email',
    'quickbooks payroll tax deposit',
    'salesforce validation rule formula',
    'figma component variant property',
    'blender cycles render noise',
    'unity rigidbody collision layer',
    'rust borrow checker lifetime error',
    'go module replace directive',
    'java maven dependency conflict',
    'android gradle signing config',
    'ios provisioning profile expired',
    'linux systemd service restart loop',
    'windows bitlocker recovery key',
    'router dns over https setting',
    'zelle payment pending review',
    'venmo instant transfer fee',
    'mortgage refinance closing disclosure',
    'california dmv real id appointment',
    'new york parking ticket dispute',
    'texas franchise tax no tax due report',
    'college fafsa dependency override',
    'w2 corrected form box 12 code',
    'health insurance out of network appeal',
    'clinic cpt code billing modifier',
    'jira workflow transition condition',
    'slack app manifest oauth scope',
    'zoom webinar registration limit',
    'mailchimp dkim authentication',
    'cloudflare cname flattening',
    'dns txt spf include limit',
    'elasticsearch shard relocation',
    'snowflake warehouse auto suspend',
    'datadog log exclusion filter',
    'prometheus alertmanager silence',
    'kafka consumer group lag',
    's3 lifecycle transition rule',
    'azure managed identity role assignment',
    'gcp service account key rotation',
    'kubernetes ingress tls secret',
    'nextjs server action form data',
    'tailwind container query plugin',
    'vite dependency prebundle cache',
    'playwright trace viewer',
];

function parseArgs(argv) {
    const out = { dataDir: 'data-perm', probes: [], foreignTitles: null };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === '--probes') {
            if (!argv[i + 1]) throw new Error('--probes requires a path');
            out.probes.push(argv[++i]);
        } else if (arg.startsWith('--probes=')) {
            out.probes.push(arg.slice('--probes='.length));
        } else if (arg === '--foreign-titles') {
            if (!argv[i + 1]) throw new Error('--foreign-titles requires a path');
            out.foreignTitles = argv[++i];
        } else if (arg.startsWith('--foreign-titles=')) {
            out.foreignTitles = arg.slice('--foreign-titles='.length);
        } else if (!arg.startsWith('--')) {
            out.dataDir = arg;
        } else {
            throw new Error(`unknown argument: ${arg}`);
        }
    }
    return out;
}

function loadForeignTitles(file) {
    if (!file) return { titles: FOREIGN_TITLE_BANK, source: 'built-in' };
    const foreignPath = path.resolve(file);
    const text = fs.readFileSync(foreignPath, 'utf8');
    const parsed = foreignPath.endsWith('.json') ? JSON.parse(text) : text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!Array.isArray(parsed) || parsed.some((title) => typeof title !== 'string')) {
        throw new Error(`${file}: expected a JSON array of title strings or a newline-delimited title file`);
    }
    return { titles: parsed, source: file };
}

function loadProbeFiles(files) {
    const probes = [];
    for (const file of files) {
        const probePath = path.resolve(file);
        const parsed = JSON.parse(fs.readFileSync(probePath, 'utf8'));
        if (!Array.isArray(parsed)) throw new Error(`${file}: expected an array of probes`);
        for (const probe of parsed) {
            if (!probe || typeof probe.text !== 'string' || probe.expect === undefined) {
                throw new Error(`${file}: each probe needs { "text": string, "expect": string|string[] }`);
            }
            probes.push({ ...probe, source: file });
        }
    }
    return probes;
}

function titleQuestions(titles, n, seed) {
    // Deterministic sample of article titles -> templated questions. Positives
    // keep sourceTitle so the label can be verified by retrieval; held-out
    // negatives are absent by construction from the reduced calibration pack.
    const picked = sample(titles, Math.min(n, titles.length), seed);
    const templates = [
        (t) => `what is ${t}`,
        (t) => `tell me about ${t}`,
        (t) => `${t} explained`,
        (t) => `facts about ${t}`,
        (t) => `information about ${t}`,
        (t) => `overview of ${t}`,
        (t) => `history of ${t}`,
        (t) => `definition of ${t}`,
        (t) => `who is ${t}`,
        (t) => `where is ${t}`,
        (t) => `why is ${t} important`,
        (t) => `how does ${t} work`,
    ];
    return picked.map((t, i) => ({ text: templates[i % templates.length](t.toLowerCase()), sourceTitle: t }));
}

function rng(seed) {
    let state = seed >>> 0;
    return () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 0xffffffff);
}

function sample(items, n, seed) {
    const next = rng(seed);
    const picked = [];
    const used = new Set();
    while (picked.length < n && used.size < items.length) {
        const i = Math.floor(next() * items.length);
        if (used.has(i)) continue;
        used.add(i);
        picked.push(items[i]);
    }
    return picked;
}

function readCorpusRows(corpusPath) {
    const rows = [];
    const seen = new Set();
    for (const line of fs.readFileSync(corpusPath, 'utf8').split('\n')) {
        if (!line) continue;
        const row = JSON.parse(line);
        rows.push(row);
        seen.add(row.title);
    }
    return { rows, titles: [...seen] };
}

function chooseHeldoutTitles(rows, rate, seed) {
    const byTitle = new Map();
    for (const row of rows) byTitle.set(row.title, (byTitle.get(row.title) || 0) + 1);
    const target = Math.max(1, Math.round(rows.length * rate));
    const shuffled = sample([...byTitle.keys()], byTitle.size, seed);
    const heldout = new Set();
    let chunks = 0;
    for (const title of shuffled) {
        heldout.add(title);
        chunks += byTitle.get(title);
        if (chunks >= target) break;
    }
    return { heldout, chunks, target };
}

function writeReducedCorpus(fullRows, heldoutTitles, dstDir) {
    fs.rmSync(dstDir, { recursive: true, force: true });
    fs.mkdirSync(dstDir, { recursive: true });
    const vecBytes = fs.readFileSync(path.join(dataDir, 'vectors.f32'));
    const fullVectors = new Float32Array(vecBytes.buffer, vecBytes.byteOffset, vecBytes.byteLength / 4);
    const keptRows = [];
    const vectorsFd = fs.openSync(path.join(dstDir, 'vectors.f32'), 'w');
    const corpusFd = fs.openSync(path.join(dstDir, 'corpus.jsonl'), 'w');
    try {
        for (let pos = 0; pos < fullRows.length; pos++) {
            const row = fullRows[pos];
            if (heldoutTitles.has(row.title)) continue;
            const id = keptRows.length;
            const nextRow = { ...row, id };
            keptRows.push(nextRow);
            const start = pos * DIM;
            const vectorBytes = Buffer.from(fullVectors.buffer, fullVectors.byteOffset + start * 4, DIM * 4);
            fs.writeSync(vectorsFd, vectorBytes);
            fs.writeSync(corpusFd, JSON.stringify(nextRow, null, 0) + '\n');
        }
    } finally {
        fs.closeSync(vectorsFd);
        fs.closeSync(corpusFd);
    }
    fs.writeFileSync(path.join(dstDir, 'corpus-manifest.json'), JSON.stringify({
        ...sourceManifest,
        chunks: keptRows.length,
        abstentionCalibration: {
            sourceChunks: fullRows.length,
            heldoutTitles: heldoutTitles.size,
            heldoutChunks: fullRows.length - keptRows.length,
        },
    }, null, 2) + '\n');
    return keptRows;
}

async function buildPack(dataPath) {
    const manifest = JSON.parse(fs.readFileSync(path.join(dataPath, 'corpus-manifest.json'), 'utf8'));
    const vectorsBytes = fs.readFileSync(path.join(dataPath, 'vectors.f32'));
    const count = vectorsBytes.byteLength / (DIM * 4);
    if (!Number.isInteger(count) || count !== manifest.chunks) {
        throw new Error(`${dataPath}: vectors.f32 holds ${count} rows but manifest says ${manifest.chunks}`);
    }
    const vectors = new Float32Array(vectorsBytes.buffer, vectorsBytes.byteOffset, count * DIM);
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
        }
    }
    if (batch.length) index.addBatch(batch);
    const snapshotPath = path.join(dataPath, 'wiki.pnck');
    fs.writeFileSync(snapshotPath, index.export());
    index.dispose();
    Pikelet.buildSketchArtifactFile(snapshotPath, path.join(dataPath, 'wiki.pikelet-sketch'), {
        sketchDims: 192,
        sketchBits: 4,
        recommendedRerank: RERANK,
    });
    const corpusSrc = fs.readFileSync(path.join(dataPath, 'corpus.jsonl'));
    fs.copyFileSync(path.join(dataPath, 'corpus.jsonl'), path.join(dataPath, 'corpus.bin'));
    const offsets = new Uint32Array(count + 1);
    let row = 0;
    for (let i = 0; i < corpusSrc.length; i++) {
        if (corpusSrc[i] === 0x0a) offsets[++row] = i + 1;
    }
    if (row !== count) throw new Error(`${dataPath}: corpus.jsonl has ${row} rows, expected ${count}`);
    fs.writeFileSync(path.join(dataPath, 'corpus-offsets.u32'), Buffer.from(offsets.buffer));
}

// --- Corpus vocabulary bloom: the known-token fraction separates gibberish
// (d0 ~0.49, inside the answerable band — distance alone cannot catch it)
// from real queries. FNV-1a with two seeds over lowercase word tokens;
// the client computes the identical hash from the shipped bitset.
const BLOOM_BITS = 1 << 21;
function fnv1a(str, seed) {
    let h = 0x811c9dc5 ^ seed;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0) % BLOOM_BITS;
}
const tokenize = (text) => (text.toLowerCase().match(/[a-z0-9']+/g) || []);

function buildVocabBloom(corpusPath) {
    const counts = new Map();
    for (const line of fs.readFileSync(corpusPath, 'utf8').split('\n')) {
        if (!line) continue;
        for (const w of tokenize(JSON.parse(line).text)) {
            counts.set(w, (counts.get(w) || 0) + 1);
        }
    }
    const bloom = new Uint8Array(BLOOM_BITS / 8);
    let kept = 0;
    for (const [w, c] of counts) {
        if (c < 3) continue;
        kept++;
        for (const seed of [0, 0x9e3779b9]) {
            const bit = fnv1a(w, seed);
            bloom[bit >> 3] |= 1 << (bit & 7);
        }
    }
    console.log(`vocab bloom: ${kept} words (of ${counts.size} unique) -> ${bloom.length / 1024} KiB`);
    return bloom;
}

function loadOrBuildBloom(dir, write) {
    const bloomPath = path.join(dir, 'wiki-vocab.bloom');
    if (fs.existsSync(bloomPath)) {
        // Copy out of the read buffer: small files share Node's buffer pool, so
        // aliasing .buffer directly would read unrelated bytes.
        const bloom = Uint8Array.from(fs.readFileSync(bloomPath));
        console.log(`vocab bloom: loaded existing from ${path.relative(here, bloomPath)}`);
        return bloom;
    }
    const bloom = buildVocabBloom(path.join(dir, 'corpus.jsonl'));
    if (write) fs.writeFileSync(bloomPath, Buffer.from(bloom));
    return bloom;
}

function knownFrac(text, bloom) {
    const words = tokenize(text);
    if (!words.length) return 0;
    let known = 0;
    for (const w of words) {
        const hit = [0, 0x9e3779b9].every((seed) => {
            const bit = fnv1a(w, seed);
            return (bloom[bit >> 3] >> (bit & 7)) & 1;
        });
        if (hit) known++;
    }
    return known / words.length;
}

function syntheticGibberish(n, seed, bloom) {
    const next = rng(seed);
    const alphabet = 'abcdefghijklmnopqrstuvwxyz';
    const queries = [];
    const used = new Set();
    function token() {
        const len = 5 + Math.floor(next() * 6);
        let s = '';
        for (let i = 0; i < len; i++) s += alphabet[Math.floor(next() * alphabet.length)];
        return s;
    }
    while (queries.length < n && used.size < n * 100) {
        const words = [];
        const wordCount = 3 + Math.floor(next() * 4);
        for (let i = 0; i < wordCount; i++) words.push(token());
        const text = words.join(' ');
        used.add(text);
        if (knownFrac(text, bloom) === 0) queries.push(text);
    }
    return queries;
}

const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'fp16' });

async function openPack(dir, bloom) {
    const artifact = await Pikelet.openSketchArtifactFile(path.join(dir, 'wiki.pikelet-sketch'));
    const scanner = await Pikelet.createSketchScanner(artifact);
    const offsetsBuf = fs.readFileSync(path.join(dir, 'corpus-offsets.u32'));
    const offsets = new Uint32Array(offsetsBuf.buffer, offsetsBuf.byteOffset, offsetsBuf.byteLength / 4);
    const corpusFd = fs.openSync(path.join(dir, 'corpus.bin'), 'r');
    return {
        dir,
        artifact,
        scanner,
        chunkTitle(id) {
            const buf = Buffer.alloc(offsets[id + 1] - offsets[id]);
            fs.readSync(corpusFd, buf, 0, buf.length, offsets[id]);
            return JSON.parse(buf.toString('utf8')).title;
        },
        signalsFromResults(text, results) {
            const top = results.slice(0, K);
            const d0 = top[0]?.distance ?? 1;
            const margin = top.length > 1 ? top[Math.min(4, top.length - 1)].distance - d0 : 0;
            const mean10 = top.length ? top.reduce((s, r) => s + r.distance, 0) / top.length : 1;
            return { d0, margin, mean10, known_frac: knownFrac(text, bloom), topIds: top.map((r) => r.id) };
        },
        async search(text, k = K) {
            const out = await embedder(text, { pooling: 'mean', normalize: true });
            const { results } = await artifact.search(Float32Array.from(out.data), k, { rerank: Math.max(RERANK, k), scanner });
            return results;
        },
        async signalsFor(text) {
            return this.signalsFromResults(text, await this.search(text, K));
        },
        async close() {
            scanner.dispose();
            await artifact.close();
            fs.closeSync(corpusFd);
        },
    };
}

const { rows: fullRows, titles: fullTitles } = readCorpusRows(path.join(dataDir, 'corpus.jsonl'));
const { heldout, chunks: heldoutChunks, target } = chooseHeldoutTitles(fullRows, HOLDOUT_RATE, SEED);
const retainedTitles = fullTitles.filter((title) => !heldout.has(title));
console.log(`held-out title shard: ${heldout.size} titles, ${heldoutChunks}/${fullRows.length} chunks (${((heldoutChunks / fullRows.length) * 100).toFixed(2)}%, target ${target})`);
writeReducedCorpus(fullRows, heldout, CALIB_DIR);
console.log(`building reduced calibration pack in ${path.relative(here, CALIB_DIR)}`);
await buildPack(CALIB_DIR);

const calibBloom = loadOrBuildBloom(CALIB_DIR, false);
const fullBloom = loadOrBuildBloom(dataDir, true);
const calibPack = await openPack(CALIB_DIR, calibBloom);
const fullPack = await openPack(dataDir, fullBloom);

const foreignTitleBank = loadForeignTitles(args.foreignTitles);
const positiveTemplates = titleQuestions(retainedTitles, FIT_QUERIES, SEED ^ 0x51f15e);
const negativeTemplates = titleQuestions([...heldout], HOLDOUT_QUERIES, SEED ^ 0xdeadbeef);
const foreignTemplates = titleQuestions(foreignTitleBank.titles, FOREIGN_QUERIES, SEED ^ 0xf03e16);
const weakTemplates = titleQuestions(retainedTitles, Math.max(WEAK_QUERIES * 12, WEAK_QUERIES), SEED ^ 0x0ddba11);
const rows = [];

// Template positives are kept only when retrieval verifiably returns the
// source article — otherwise they are silently unanswerable-in-practice and
// would drag the no-false-abstain floor toward zero.
let dropped = 0;
for (const { text, sourceTitle } of positiveTemplates) {
    const sig = await calibPack.signalsFor(text);
    const found = sig.topIds.some((id) => calibPack.chunkTitle(id) === sourceTitle);
    if (found) rows.push({ text, label: 1, ...sig });
    else dropped++;
}
console.log(`template positives: ${positiveTemplates.length - dropped} verified, ${dropped} dropped (source article not retrieved)`);
const positiveD0 = rows.filter((r) => r.label === 1).map((r) => r.d0).sort((a, b) => a - b);
if (!positiveD0.length) throw new Error('abstention calibration needs at least one verified positive');
const positiveMedianD0 = positiveD0[Math.floor(positiveD0.length / 2)];
let heldoutDropped = 0;
const heldoutDroppedTitles = [];
for (const { text, sourceTitle } of negativeTemplates) {
    const sig = await calibPack.signalsFor(text);
    if (sig.d0 <= positiveMedianD0) {
        heldoutDropped++;
        heldoutDroppedTitles.push({ title: sourceTitle, text, d0: +sig.d0.toFixed(6), topTitle: calibPack.chunkTitle(sig.topIds[0]) });
    } else {
        rows.push({ text, sourceTitle, label: 0, negativeKind: 'heldout-title', ...sig });
    }
}

let foreignDropped = 0;
const foreignDroppedTitles = [];
for (const { text, sourceTitle } of foreignTemplates) {
    const sig = await calibPack.signalsFor(text);
    if (sig.d0 <= positiveMedianD0) {
        foreignDropped++;
        foreignDroppedTitles.push({ title: sourceTitle, text, d0: +sig.d0.toFixed(6), topTitle: calibPack.chunkTitle(sig.topIds[0]) });
    }
    else rows.push({ text, sourceTitle, label: 0, negativeKind: 'foreign-title', ...sig });
}

const gibberish = syntheticGibberish(GIBBERISH_QUERIES, SEED ^ 0x9166e11, calibBloom);
for (const text of gibberish) {
    rows.push({ text, label: 0, negativeKind: 'synthetic-gibberish', ...await calibPack.signalsFor(text) });
}

let weakScanned = 0;
for (const { text, sourceTitle } of weakTemplates) {
    if (rows.filter((r) => r.label === -1).length >= WEAK_QUERIES) break;
    weakScanned++;
    const results = await calibPack.search(text, WEAK_SCAN_K);
    const rank = results.findIndex((r) => calibPack.chunkTitle(r.id) === sourceTitle) + 1;
    if (rank >= 5 && rank <= WEAK_SCAN_K) {
        rows.push({ text, sourceTitle, sourceRank: rank, label: -1, ...calibPack.signalsFromResults(text, results) });
    }
}

console.log(`held-out negatives: ${negativeTemplates.length - heldoutDropped} kept, ${heldoutDropped} dropped (d0 <= positive median ${positiveMedianD0.toFixed(4)})`);
if (heldoutDroppedTitles.length) {
    for (const droppedTitle of heldoutDroppedTitles.slice(0, 20)) {
        console.log(`  DROP held-out overlap: "${droppedTitle.text}" d0=${droppedTitle.d0.toFixed(4)} top="${droppedTitle.topTitle}"`);
    }
    if (heldoutDroppedTitles.length > 20) console.log(`  ... ${heldoutDroppedTitles.length - 20} more held-out overlaps dropped`);
}
console.log(`foreign negatives (${foreignTitleBank.source}): ${foreignTemplates.length - foreignDropped} kept, ${foreignDropped} dropped (d0 <= positive median ${positiveMedianD0.toFixed(4)})`);
if (foreignDroppedTitles.length) {
    for (const droppedTitle of foreignDroppedTitles.slice(0, 20)) {
        console.log(`  DROP foreign overlap: "${droppedTitle.text}" d0=${droppedTitle.d0.toFixed(4)} top="${droppedTitle.topTitle}"`);
    }
    if (foreignDroppedTitles.length > 20) console.log(`  ... ${foreignDroppedTitles.length - 20} more foreign overlaps dropped`);
}
console.log(`scoring ${rows.filter((r) => r.label === 1).length} answerable, ${negativeTemplates.length - heldoutDropped} held-out negatives, ${foreignTemplates.length - foreignDropped} foreign negatives, ${gibberish.length} gibberish negatives, ${rows.filter((r) => r.label === -1).length} weak (${weakScanned} candidates scanned)`);
if (!rows.some((r) => r.label === 1) || !rows.some((r) => r.label === 0)) {
    throw new Error('abstention calibration needs at least one verified positive and one held-out negative');
}

// Flag remaining suspicious labels for manual review.
for (const r of rows) {
    if (r.label === 0 && r.d0 < 0.30) console.log(`  REVIEW ${r.negativeKind} negative with strong hit: "${r.text}" d0=${r.d0.toFixed(3)} top="${calibPack.chunkTitle(r.topIds[0])}"`);
}

// Standardize features, fit logistic regression by gradient descent.
const FEATS = ['d0', 'margin', 'mean10', 'known_frac'];
const fit = rows.filter((r) => r.label >= 0);
const fitWeights = fit.map((r) => r.negativeKind === 'synthetic-gibberish' ? 0.25 : 1);
const weightSum = fitWeights.reduce((a, c) => a + c, 0);
const mean = {}, std = {};
for (const f of FEATS) {
    mean[f] = fit.reduce((sum, r, i) => sum + r[f] * fitWeights[i], 0) / weightSum;
    std[f] = Math.sqrt(fit.reduce((sum, r, i) => sum + ((r[f] - mean[f]) ** 2) * fitWeights[i], 0) / weightSum) || 1;
}
const xs = fit.map((r) => FEATS.map((f) => (r[f] - mean[f]) / std[f]));
const ys = fit.map((r) => r.label);
let w = FEATS.map(() => 0);
let b = 0;
for (let epoch = 0; epoch < 4000; epoch++) {
    const gw = FEATS.map(() => 0);
    let gb = 0;
    for (let i = 0; i < xs.length; i++) {
        const z = xs[i].reduce((s, v, j) => s + v * w[j], b);
        const p = 1 / (1 + Math.exp(-z));
        const err = (p - ys[i]) * fitWeights[i];
        xs[i].forEach((v, j) => { gw[j] += err * v; });
        gb += err;
    }
    w = w.map((wj, j) => wj - 0.1 * (gw[j] / weightSum + 1e-3 * wj));
    b -= 0.1 * (gb / weightSum);
}
const prob = (r) => 1 / (1 + Math.exp(-(FEATS.reduce((s, f, j) => s + ((r[f] - mean[f]) / std[f]) * w[j], b))));
for (const r of rows) r.p = prob(r);

function aucFor(posRows, negRows) {
    if (!posRows.length || !negRows.length) return null;
    let score = 0;
    for (const a of posRows) for (const c of negRows) score += a.p > c.p ? 1 : a.p === c.p ? 0.5 : 0;
    return score / (posRows.length * negRows.length);
}

// AUC as sanity, then thresholds.
const pos = rows.filter((r) => r.label === 1).map((r) => r.p);
const neg = rows.filter((r) => r.label === 0).map((r) => r.p);
const aucByKind = {
    all: aucFor(rows.filter((r) => r.label === 1), rows.filter((r) => r.label === 0)),
    excludingGibberish: aucFor(rows.filter((r) => r.label === 1), rows.filter((r) => r.label === 0 && r.negativeKind !== 'synthetic-gibberish')),
    heldout: aucFor(rows.filter((r) => r.label === 1), rows.filter((r) => r.negativeKind === 'heldout-title')),
    foreignTitle: aucFor(rows.filter((r) => r.label === 1), rows.filter((r) => r.negativeKind === 'foreign-title')),
    syntheticGibberish: aucFor(rows.filter((r) => r.label === 1), rows.filter((r) => r.negativeKind === 'synthetic-gibberish')),
};

// hard: place it between the negative ceiling and the answerable/weak floor
// when labels separate cleanly. Weak rows represent adjacent content, so they
// should not be hidden. weak: place it between the weak ceiling and answerable
// floor when possible, otherwise stay just under the weakest answerable query.
const minPos = Math.min(...pos);
const maxNeg = Math.max(...neg);
const weakP = rows.filter((r) => r.label === -1).map((r) => r.p);
const minWeak = weakP.length ? Math.min(...weakP) : minPos;
const maxWeak = weakP.length ? Math.max(...weakP) : 0;
const floor = Math.min(minPos, minWeak);
const hardOverlap = maxNeg >= floor;
let hard = hardOverlap ? floor * 0.5 : Math.sqrt(maxNeg * floor);
const weak = weakP.length && maxWeak > 0 && maxWeak < minPos ? Math.sqrt(maxWeak * minPos) : minPos * 0.9;
const legacyHardClampWouldBind = hard > weak * 0.6;

const verdict = (p) => (p < hard ? 'abstain' : p < weak ? 'weak' : 'answer');
if (process.env.DUMP) {
    for (const r of [...rows].sort((a, b) => a.p - b.p)) {
        const lab = r.label === 1 ? 'POS' : r.label === 0 ? 'NEG' : 'WEAK';
        console.log(`${lab} p=${r.p.toFixed(3)} d0=${r.d0.toFixed(3)} margin=${r.margin.toFixed(3)} mean=${r.mean10.toFixed(3)} top="${calibPack.chunkTitle(r.topIds[0])}"  ${r.text.slice(0, 48)}`);
    }
}
const confusion = {};
for (const r of rows) {
    const label = r.label === 1 ? 'answerable'
        : r.label === -1 ? 'weak'
            : r.negativeKind || 'negative';
    const key = `${label}:${verdict(r.p)}`;
    confusion[key] = (confusion[key] || 0) + 1;
}
console.log(`\nAUC ${aucByKind.all.toFixed(4)}  AUC excluding gibberish ${aucByKind.excludingGibberish?.toFixed(4) ?? 'n/a'}  minPos ${minPos.toFixed(4)}  maxNeg ${Math.max(...neg).toFixed(4)}`);
console.log('AUC by kind:', Object.fromEntries(Object.entries(aucByKind).map(([k, v]) => [k, v == null ? null : +v.toFixed(4)])));
console.log('thresholds:', { hard: +hard.toFixed(4), weak: +weak.toFixed(4) });
if (hardOverlap) console.log(`hard threshold overlap: maxNeg ${maxNeg.toFixed(4)} >= protected floor ${floor.toFixed(4)}; preserving positives/weak rows`);
if (legacyHardClampWouldBind) console.log(`legacy hard clamp would have lowered hard to ${(weak * 0.6).toFixed(4)}`);
console.log('confusion:', JSON.stringify(confusion, null, 1));

const scoreSignals = (r) => 1 / (1 + Math.exp(-(FEATS.reduce((s, f, j) => s + ((r[f] - mean[f]) / std[f]) * w[j], b))));
const fullShiftRows = [];
let recoveredHeldout = 0;
for (const row of rows) {
    const fullSig = await fullPack.signalsFor(row.text);
    const fullP = scoreSignals(fullSig);
    const fullVerdict = verdict(fullP);
    if (row.negativeKind === 'heldout-title' && fullSig.topIds.some((id) => fullPack.chunkTitle(id) === row.sourceTitle)) recoveredHeldout++;
    fullShiftRows.push({ label: row.label, negativeKind: row.negativeKind, sourceTitle: row.sourceTitle, reducedP: row.p, fullP, fullVerdict });
}
const fullPos = fullShiftRows.filter((r) => r.label === 1);
const fullHeldoutNeg = fullShiftRows.filter((r) => r.negativeKind === 'heldout-title');
const fullForeignNeg = fullShiftRows.filter((r) => r.negativeKind === 'foreign-title');
const fullGibberishNeg = fullShiftRows.filter((r) => r.negativeKind === 'synthetic-gibberish');
const fullWeak = fullShiftRows.filter((r) => r.label === -1);
const avgAbsDelta = fullShiftRows.reduce((s, r) => s + Math.abs(r.fullP - r.reducedP), 0) / fullShiftRows.length;
const fullMinPos = Math.min(...fullPos.map((r) => r.fullP));
const shift = {
    avgAbsProbabilityDelta: +avgAbsDelta.toFixed(6),
    fullMinPositiveProbability: +fullMinPos.toFixed(6),
    heldoutTitlesRecoveredInFullTopK: recoveredHeldout,
    heldoutQueriesScoredAnswerInFull: fullHeldoutNeg.filter((r) => r.fullVerdict === 'answer').length,
    foreignQueriesScoredAnswerInFull: fullForeignNeg.filter((r) => r.fullVerdict === 'answer').length,
    gibberishQueriesScoredAnswerInFull: fullGibberishNeg.filter((r) => r.fullVerdict === 'answer').length,
    weakQueriesScoredAnswerInFull: fullWeak.filter((r) => r.fullVerdict === 'answer').length,
    generatedQueries: fullShiftRows.length,
};
console.log('reduced-vs-full shift:', shift);

const asset = {
    version: 1,
    corpus: sourceManifest.dataset || path.basename(dataDir),
    searchConfig: { rerank: RERANK, k: K },
    calibration: {
        method: 'heldout-title-shard',
        seed: SEED,
        holdoutRate: HOLDOUT_RATE,
        heldoutTitles: heldout.size,
        heldoutChunks,
        sourceChunks: fullRows.length,
        fitQueries: rows.length,
        verifiedPositiveQueries: rows.filter((r) => r.label === 1).length,
        heldoutNegativeQueries: rows.filter((r) => r.negativeKind === 'heldout-title').length,
        heldoutDroppedAsSemanticOverlap: heldoutDropped,
        heldoutDropD0Threshold: +positiveMedianD0.toFixed(6),
        foreignNegativeQueries: rows.filter((r) => r.negativeKind === 'foreign-title').length,
        foreignTitleSource: foreignTitleBank.source,
        foreignDroppedAsSemanticOverlap: foreignDropped,
        foreignDropD0Threshold: +positiveMedianD0.toFixed(6),
        syntheticGibberishQueries: rows.filter((r) => r.negativeKind === 'synthetic-gibberish').length,
        syntheticGibberishFitWeight: 0.25,
        weakQueries: rows.filter((r) => r.label === -1).length,
        aucByKind: Object.fromEntries(Object.entries(aucByKind).map(([k, v]) => [k, v == null ? null : +v.toFixed(6)])),
        hardThresholdOverlap: hardOverlap,
        legacyHardClampWouldBind,
        reducedVsFullShift: shift,
    },
    features: FEATS,
    standardize: { mean, std },
    weights: w,
    bias: b,
    thresholds: { hard: +hard.toFixed(6), weak: +weak.toFixed(6) },
    vocabBloom: { file: 'wiki-vocab.bloom', bits: BLOOM_BITS, hashes: ['fnv1a:0', 'fnv1a:0x9e3779b9'], minCount: 3 },
};
fs.writeFileSync(path.join(dataDir, 'wiki-abstention.json'), JSON.stringify(asset, null, 2) + '\n');

// Golden probes: generated probes assert labels, and optional owner probes add
// corpus-specific expectations without making them a requirement for default
// calibration.
const probes = [
    ...rows.filter((r) => r.label === 1).slice(0, 8).map((r) => ({ text: r.text, expect: 'answer' })),
    ...rows.filter((r) => r.negativeKind === 'synthetic-gibberish').slice(0, 6).map((r) => ({ text: r.text, expect: 'abstain' })),
    ...rows.filter((r) => r.label === -1).slice(0, 6).map((r) => ({ text: r.text, expect: ['weak', 'answer'] })),
    ...loadProbeFiles(args.probes),
];
let probeFails = 0;
for (const probe of probes) {
    const r = await fullPack.signalsFor(probe.text);
    const v = verdict(scoreSignals(r));
    const expected = Array.isArray(probe.expect) ? probe.expect : [probe.expect];
    if (!expected.includes(v)) { probeFails++; console.log(`  PROBE FAIL: "${probe.text}" expected ${expected.join('|')} got ${v}`); }
}
fs.writeFileSync(path.join(dataDir, 'wiki-abstention-probes.json'), JSON.stringify(probes, null, 2) + '\n');
console.log(`golden probes: ${probes.length - probeFails}/${probes.length} pass`);
console.log('wrote wiki-abstention.json + wiki-abstention-probes.json');
// The abstention asset and bloom are served under the versioned pack URL, so
// their bytes are part of the packVersion hash: re-stamp the manifest.
const { packVersion } = stampPackVersion(dataDir);
console.log(`pack manifest re-stamped (packVersion ${packVersion})`);

await calibPack.close();
await fullPack.close();
if (probeFails) throw new Error(`${probeFails}/${probes.length} golden probes failed`);
