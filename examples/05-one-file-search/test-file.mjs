#!/usr/bin/env node
// Acceptance for the one-file reader. The golden queries come from the
// .pikelet's OWN evaluation segment — the artifact carries its conformance
// fixtures, so this test is "open the file, ask it to prove itself":
//   1. every golden query reproduces its expected match-quality label;
//   2. every corpus record round-trips the compiler byte-exactly;
//   3. file-reader results match the six-asset spike facade on the sample
//      queries (same components, same answers).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { openPikeletFile } from './pikelet-file-reader.mjs';
import { openDocsSearch, docsAssetPaths } from './search-reader.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const pikeletPath = path.join(here, 'pikelet-docs.pikelet');
if (!fs.existsSync(pikeletPath)) {
    console.error('pikelet-docs.pikelet not found — run node compile.mjs first');
    process.exit(1);
}

let passed = 0;
let failed = 0;
const check = (name, ok, detail) => {
    if (ok) { passed++; console.log(`  ok: ${name}`); }
    else { failed++; console.log(`  FAIL: ${name}${detail ? ` — ${detail}` : ''}`); }
};

const search = await openPikeletFile(pikeletPath);
const info = search.info();
console.log(`opened ${path.basename(pikeletPath)}: identity ${info.identity.slice(0, 16)}..., `
    + `${info.records} records, resident ${(info.residentBytes / 1024).toFixed(1)} KiB, `
    + `sketch hash verified: ${info.residentVerified}`);

// 1. Self-test: golden queries from the file's evaluation segment.
const evaluation = await search.evaluation();
check('evaluation segment loads and digest-verifies', Array.isArray(evaluation.goldenQueries)
    && evaluation.goldenQueries.length === 10);
for (const fixture of evaluation.goldenQueries) {
    const out = await search.query(fixture.text, { k: 5 });
    check(`[${fixture.family}] "${fixture.text}" -> ${fixture.expected}`,
        out.matchQuality === fixture.expected
        && (fixture.expected !== 'none' || out.results.length === 0),
        `got ${out.matchQuality}, ${out.results.length} results`);
}

// 2. Hydration: results resolve to real records matching the source corpus.
const sourceCorpus = JSON.parse(fs.readFileSync(docsAssetPaths().corpusPath, 'utf8'));
const probe = await search.query('how does compaction work', { k: 3 });
check('probe query hydrates real records', probe.results.length === 3
    && probe.results.every((r) => r.title && r.text && r.sourcePath));
check('probe hydration matches source corpus', probe.results.every((r) => {
    const src = sourceCorpus[String(r.id)];
    return src && src.title === r.title && src.text === r.text && src.sourcePath === r.sourcePath;
}));

// 3. Parity with the six-asset spike facade on the manifest sample queries.
const spike = await openDocsSearch(docsAssetPaths());
// The spike facade reranks at its manifest's efSearch; the artifact now
// carries a measured (usually smaller) operating point. Parity is about
// same components → same answers, so compare at the same explicit C.
const spikeRerank = JSON.parse(fs.readFileSync(docsAssetPaths().manifestPath, 'utf8')).efSearch || 120;
let parity = true;
for (const q of info.sampleQueries) {
    const a = await search.query(q, { k: 5, rerank: spikeRerank });
    const b = await spike.query(q, { k: 5 });
    if (a.matchQuality !== b.matchQuality
        || JSON.stringify(a.results.map((r) => r.id)) !== JSON.stringify(b.results.map((r) => r.id))) {
        parity = false;
        console.log(`    parity break on "${q}": file ${a.results.map((r) => r.id)} vs spike ${b.results.map((r) => r.id)}`);
    }
}
check(`file reader matches spike facade on ${info.sampleQueries.length} sample queries`, parity);
await spike.close();

// 4. Tamper: flip one byte in the corpus segment; open succeeds (lazy), the
// manifest identity is unchanged, but evaluation()/eager segments verify.
const tampered = Buffer.from(fs.readFileSync(pikeletPath));
tampered[tampered.length - 3] ^= 0xff; // inside the evaluation segment (last)
const tamperedPath = pikeletPath + '.tampered';
fs.writeFileSync(tamperedPath, tampered);
const tamperedReader = await openPikeletFile(tamperedPath);
let rejected = false;
try { await tamperedReader.evaluation(); }
catch (err) { rejected = /hash verification/.test(String(err.message)); }
check('tampered evaluation segment rejected by digest', rejected);
await tamperedReader.close();
fs.rmSync(tamperedPath);

await search.close();
console.log(`\nOne-file reader: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
