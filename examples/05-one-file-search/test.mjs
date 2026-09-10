#!/usr/bin/env node
// Acceptance gate for the composed reader: every abstention golden fixture
// (calibrated and committed against 03's Worker, which searches a restored
// snapshot) must reproduce its expected match-quality label through this
// reader's sketch-tier path. A flip here is a real finding about
// snapshot-vs-sketch candidate drift, so mismatches print full signals.

import fs from 'node:fs';
import path from 'node:path';
import { openDocsSearch, docsAssetPaths } from './search-reader.mjs';

const fixturesPath = path.join(
    path.dirname(new URL(import.meta.url).pathname),
    '..', 'legacy', '03-edge-docs-search', 'fixtures', 'abstention-golden.json'
);
const fixtures = JSON.parse(fs.readFileSync(fixturesPath, 'utf8'));

const search = await openDocsSearch(docsAssetPaths());
let passed = 0;
let failed = 0;

for (const fixture of fixtures) {
    const out = await search.query(fixture.text, { k: 5 });
    const ok = out.matchQuality === fixture.expected
        && (fixture.expected !== 'none' || out.results.length === 0);
    if (ok) {
        passed++;
        console.log(`  ok: [${fixture.family}] "${fixture.text}" -> ${out.matchQuality}`
            + (out.confidence !== undefined ? ` (${out.confidence.toFixed(4)})` : ''));
    } else {
        failed++;
        console.log(`  FAIL: [${fixture.family}] "${fixture.text}" expected ${fixture.expected}, `
            + `got ${out.matchQuality} (confidence ${out.confidence}, results ${out.results.length})`);
    }
}

await search.close();
console.log(`\nComposed reader vs abstention goldens: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
