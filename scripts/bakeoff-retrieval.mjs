#!/usr/bin/env node
// Retrieval-quality bakeoff over a compiled .pikelet artifact: run the same
// query set through vector-only, BM25-only, and hybrid retrieval (the
// reader's query({ retrieval }) modes) and score each against expected
// sections.
//
//   node scripts/bakeoff-retrieval.mjs <artifact.pikelet> <queries.json>
//
// queries.json: [{ text, kind: 'semantic' | 'exact', expect: {...} }, ...]
// expect matches a returned record when every given field matches:
//   anchor  — exact match on the record's anchor
//   title   — case-insensitive substring of the record's title
//   path    — case-insensitive substring of headingPath joined with ' > '
//   url     — case-insensitive substring of the record's url
//
// Reported per mode and kind: success@1, success@3, MRR@5, and how many
// queries the abstention layer answered with no results (counted as
// misses — the bakeoff measures what a user of the artifact experiences).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { openPikeletFile } = await import(path.join(ROOT, 'complete', 'index.mjs'));

const [artifactPath, queriesPath] = process.argv.slice(2);
if (!artifactPath || !queriesPath) {
  console.error('usage: node scripts/bakeoff-retrieval.mjs <artifact.pikelet> <queries.json>');
  process.exit(1);
}
const queries = JSON.parse(fs.readFileSync(queriesPath, 'utf8'));
const MODES = ['vector', 'lexical', 'augmented', 'hybrid'];
const K = 5;

function matches(record, expect) {
  if (!record) return false;
  const contains = (hay, needle) => String(hay || '').toLowerCase().includes(String(needle).toLowerCase());
  if (expect.anchor !== undefined && record.anchor !== expect.anchor) return false;
  if (expect.title !== undefined && !contains(record.title, expect.title)) return false;
  if (expect.path !== undefined && !contains((record.headingPath || []).join(' > '), expect.path)) return false;
  if (expect.url !== undefined && !contains(record.url, expect.url)) return false;
  return true;
}

const search = await openPikeletFile(artifactPath);
console.log(`artifact: ${path.basename(artifactPath)} — ${search.info().records} records, `
  + `lexical: ${search.info().lexical ? `${search.info().lexical.terms} terms` : 'absent'}`);
console.log(`queries: ${queries.length} (${queries.filter((q) => q.kind === 'exact').length} exact, `
  + `${queries.filter((q) => q.kind === 'semantic').length} semantic)\n`);

const table = [];
for (const mode of MODES) {
  const perKind = new Map();
  const misses = [];
  for (const q of queries) {
    const out = await search.query(q.text, { k: K, retrieval: mode });
    const rank = out.results.findIndex((r) => matches(r, q.expect));
    const stats = perKind.get(q.kind) || { n: 0, s1: 0, s3: 0, mrr: 0, abstained: 0 };
    stats.n++;
    if (out.results.length === 0) stats.abstained++;
    if (rank === 0) stats.s1++;
    if (rank >= 0 && rank < 3) stats.s3++;
    if (rank >= 0) stats.mrr += 1 / (rank + 1);
    else misses.push(q.text);
    perKind.set(q.kind, stats);
  }
  for (const [kind, s] of perKind) {
    table.push({
      mode,
      kind,
      'success@1': (s.s1 / s.n).toFixed(2),
      'success@3': (s.s3 / s.n).toFixed(2),
      'MRR@5': (s.mrr / s.n).toFixed(3),
      abstained: s.abstained,
      n: s.n,
    });
  }
  if (process.env.BAKEOFF_VERBOSE && misses.length) {
    console.log(`${mode} misses: ${misses.join(' | ')}`);
  }
}
console.table(table);
await search.close();
