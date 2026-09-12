#!/usr/bin/env node
// Acceptance for the kind-3 artifact: open pancake-wiki-inline.pancake with
// no host encoder and no ML runtime. If the large artifact is not present in
// this ignored example directory, download the release asset first.

import fs from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPikeletFile } from './pikelet-file-reader.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
// data-perm is the canonical pack layout (cluster-ordered rows, 192-dim
// sketch); its eval ground truth is in permuted id space, matching the
// artifact. data-full is the unpermuted source and its GT does not match.
const DATA = path.join(here, '..', '04-static-wiki-pack', 'data-perm');
const pikeletPath = path.join(here, 'pancake-wiki-inline.pancake');
const artifactUrl = 'https://github.com/mcn92/pikelet/releases/download/artifact-wiki-inline-v4/pancake-wiki-inline.pancake';
const expectedIdentity = '1b180adf4c6cebb2dcd5615256df6a25dac5fda8738dbbc11d60af86046f97f3';
const expectedBytes = 680029254;
const HEADER_BYTES = 64;

let passed = 0;
let failed = 0;
const check = (name, ok, detail) => {
    if (ok) { passed++; console.log(`  ok: ${name}`); }
    else { failed++; console.log(`  FAIL: ${name}${detail ? ` - ${detail}` : ''}`); }
};

async function download(url, outPath, redirects = 0) {
    const client = url.startsWith('https:') ? https : http;
    await fs.promises.mkdir(path.dirname(outPath), { recursive: true });
    const tmpPath = `${outPath}.download`;

    await new Promise((resolve, reject) => {
        const req = client.get(url, (res) => {
            if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                res.resume();
                if (redirects >= 5) {
                    reject(new Error('too many redirects while downloading artifact'));
                    return;
                }
                download(new URL(res.headers.location, url).toString(), outPath, redirects + 1)
                    .then(resolve, reject);
                return;
            }

            if (res.statusCode !== 200) {
                res.resume();
                reject(new Error(`download failed: HTTP ${res.statusCode}`));
                return;
            }

            const total = Number(res.headers['content-length'] || 0);
            let received = 0;
            let nextLog = 64 * 1024 * 1024;
            const file = fs.createWriteStream(tmpPath);

            res.on('data', (chunk) => {
                received += chunk.length;
                if (received >= nextLog) {
                    const suffix = total ? ` / ${(total / 1048576).toFixed(0)} MiB` : '';
                    console.log(`  downloaded ${(received / 1048576).toFixed(0)} MiB${suffix}`);
                    nextLog += 64 * 1024 * 1024;
                }
            });

            res.pipe(file);
            file.on('finish', () => file.close(resolve));
            file.on('error', reject);
            res.on('error', reject);
        });
        req.on('error', reject);
    });

    await fs.promises.rename(tmpPath, outPath);
}

async function ensureArtifact() {
    if (fs.existsSync(pikeletPath)) {
        const stat = fs.statSync(pikeletPath);
        if (stat.size === expectedBytes) return;
        console.log(`local ${path.basename(pikeletPath)} has unexpected size ${stat.size}; re-downloading`);
        fs.rmSync(pikeletPath, { force: true });
    }

    console.log(`downloading ${path.basename(pikeletPath)} from release asset`);
    console.log(`  ${artifactUrl}`);
    try {
        await download(artifactUrl, pikeletPath);
    } catch (err) {
        fs.rmSync(`${pikeletPath}.download`, { force: true });
        throw err;
    }
}

await ensureArtifact();

function readIdentity(filePath) {
    const fd = fs.openSync(filePath, 'r');
    try {
        const header = Buffer.alloc(HEADER_BYTES);
        const bytesRead = fs.readSync(fd, header, 0, HEADER_BYTES, 0);
        if (bytesRead !== HEADER_BYTES || header.readUInt32LE(0) !== 0x31465350) {
            throw new Error('not a .pikelet file (bad magic)');
        }
        return header.subarray(24, 56).toString('hex');
    } finally {
        fs.closeSync(fd);
    }
}

const identity = readIdentity(pikeletPath);
check('manifest identity matches release notes', identity === expectedIdentity, identity);

const openStart = performance.now();
const search = await openPikeletFile(pikeletPath);
const openMs = performance.now() - openStart;
const info = search.info();
console.log(`opened ${path.basename(pikeletPath)} in ${(openMs / 1000).toFixed(1)}s: `
    + `${(info.fileBytes / 1048576).toFixed(0)} MiB, encoder ${info.encoder.kind}`);

check('reader reports the same identity', info.identity === expectedIdentity, info.identity);
check('opens with zero options (self-contained)', info.encoder.kind === 'inline-transformer-v1');
const probe = await search.query('how do volcanoes form', { k: 5 });
// The encoder opens lazily: until the first query forces (or the prefetch
// finishes) its fetch, info().encoder serves the identity-verified manifest
// declaration. After a query, the full declaration is in place — provenance
// checks belong here.
const loadedInfo = search.info();
check('declaration carries provenance once the encoder is loaded',
    loadedInfo.encoder.model === 'sentence-transformers/all-MiniLM-L6-v2'
    && loadedInfo.encoder.license === 'apache-2.0' && !!loadedInfo.encoder.attribution);
check('inline encoder verified against its test vectors', loadedInfo.encoderVerified === true);
check('natural-language query answers with hydrated results',
    probe.matchQuality === 'strong' && probe.results.length === 5
    && probe.results.every((r) => r.title && r.text && r.url),
    `${probe.matchQuality}, ${probe.results.length} results`);

const nonsense = await search.query('qzxv blorpt nym vex', { k: 5 });
check('nonsense abstains or downgrades', nonsense.matchQuality !== 'strong',
    `${nonsense.matchQuality} (${nonsense.confidence?.toFixed(3)})`);

const evaluation = await search.evaluation();
check('artifact carries evaluation ground truth',
    evaluation && Array.isArray(evaluation.groundTruth) && evaluation.groundTruth.length === 200);

const evalQueryPath = path.join(DATA, 'eval-queries.json');
const evalGtPath = path.join(DATA, 'eval-gt.json');
if (fs.existsSync(evalQueryPath) && fs.existsSync(evalGtPath)) {
    const evalQueries = JSON.parse(fs.readFileSync(evalQueryPath, 'utf8'));
    const groundTruth = JSON.parse(fs.readFileSync(evalGtPath, 'utf8'));
    const t0 = performance.now();
    let hits = 0;
    for (let i = 0; i < evalQueries.length; i++) {
        // augmented mode: lexical candidates join the exact rerank, results
        // stay in distance order — the mode this exact-NN metric measures.
        // Default hybrid (RRF ordering) deliberately trades exact-NN overlap
        // for keyword relevance and scores ~46% here by construction.
        const out = await search.query(evalQueries[i].text, { k: 10, retrieval: 'augmented' });
        const truth = new Set(groundTruth[i]);
        hits += out.results.filter((r) => truth.has(r.id)).length;
    }
    const perQuery = (performance.now() - t0) / evalQueries.length;
    const recall = hits / (evalQueries.length * 10);
    console.log(`  augmented recall@10 over ${evalQueries.length} queries: ${(recall * 100).toFixed(1)}% `
        + `(verified 95.2% inline kernel; fp32 reference 96.0% — ~0.8pt encoder quantization tax; the correct 192-dim 2:1 sketch geometry — `
        + `the pre-v4 82.8% era embedded a 96-dim 4:1 sketch the pack's own measurements had rejected); `
        + `${perQuery.toFixed(0)} ms/query end to end`);
    check('augmented recall matches the verified inline-encoder number (>= 94%)', recall >= 0.94, recall.toFixed(4));
} else {
    console.log('  recall sweep skipped: local data-perm eval query files are not present');
    console.log('  smoke/provenance/identity checks above only need the downloaded release artifact');
}

await search.close();
console.log(`\nkind-3 artifact acceptance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
