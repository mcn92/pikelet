#!/usr/bin/env node
// Fusion simulator: replays alternative vector/lexical fusion rules offline
// against a compiled .pikelet artifact and a labeled question set, so a
// candidate rule can be judged on evidence rank before the reader changes.
//
//   node scripts/fusion-sim.mjs <artifact.pikelet> <queries.json> [--depth 50] [--report <out.json>]
//
// Per answerable question it fetches, once each, the reader's own lists:
// vector order (ids + exact distances), lexical order (BM25 hits after the
// reader's cutoff, each with its exact distance), and the shipped hybrid
// order (to validate the simulation: the 'defaults' rule, the reader's
// FUSION_DEFAULTS, must reproduce it, give or take phrase pinning). Every rule is then a pure function of
// those lists. Ranks are the position of the first result carrying the
// evidence (scripts/lib/relevance-sets.mjs: source-record ids when the
// set names them, near-verbatim quotes otherwise).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { openPikeletFile } = await import(path.join(ROOT, 'packages', 'pikelet-wasm', 'complete', 'index.mjs'));
const { loadQuestions, isAnswerable, hasEvidence } = await import(path.join(ROOT, 'scripts', 'lib', 'relevance-sets.mjs'));

function extractFlag(args, name) {
  const idx = args.indexOf(name);
  if (idx === -1) return [args, null];
  return [args.filter((_, i) => i !== idx && i !== idx + 1), args[idx + 1]];
}
let args = process.argv.slice(2);
let depthArg, reportPath;
[args, depthArg] = extractFlag(args, '--depth');
[args, reportPath] = extractFlag(args, '--report');
const [artifactPath, queriesPath] = args;
if (!artifactPath || !queriesPath) {
  console.error('usage: node scripts/fusion-sim.mjs <artifact.pikelet> <queries.json> [--depth 50] [--report <out.json>]');
  process.exit(1);
}
const depth = depthArg ? Number(depthArg) : 50;
if (!Number.isInteger(depth) || depth < 3) { console.error(`--depth must be an integer >= 3, got ${depthArg}`); process.exit(1); }

// ---- fusion rules: (lists) -> ordered array of ids -------------------------
// lists = { vector: [{id, distance}], lexical: [{id, distance}] }
// vRank: position in vector order; a lexical hit outside the vector
// top-`depth` is placed by its exact distance among the vector distances
// (the reader reranks every lexical candidate exactly, so this is what its
// fused list sees, up to ties beyond the depth).
function ranked(lists) {
  const vDist = lists.vector.map((h) => h.distance);
  const rows = new Map();
  lists.vector.forEach((h, i) => rows.set(h.id, { id: h.id, distance: h.distance, v: i + 1, l: null }));
  let extra = 0;
  for (const [i, h] of lists.lexical.entries()) {
    const row = rows.get(h.id);
    if (row) { row.l = i + 1; continue; }
    const better = vDist.filter((d) => d < h.distance).length;
    const v = better < vDist.length ? better + 1 : vDist.length + 1 + extra++;
    rows.set(h.id, { id: h.id, distance: h.distance, v, l: i + 1 });
  }
  return [...rows.values()];
}
const byScore = (rows, score) => rows
  .map((r) => ({ r, s: score(r) }))
  .sort((a, b) => (b.s - a.s) || (a.r.distance - b.r.distance))
  .map((x) => x.r.id);

// RRF and the vector-margin guard are the reader's own fuseCandidates
// (complete/fusion.mjs), fed the pool in exact-distance order the way the
// reader's rerank output is; a rule that wins here is the rule that ships.
const { fuseCandidates, FUSION_DEFAULTS } = await import(path.join(ROOT, 'packages', 'pikelet-wasm', 'complete', 'fusion.mjs'));
const poolByDistance = (lists) => ranked(lists).sort((a, b) => a.distance - b.distance).map((r) => ({ id: r.id, distance: r.distance }));
const rrf = (K, w) => (lists) => fuseCandidates(poolByDistance(lists), lists.lexical.map((h) => h.id), { rrfK: K, lexicalWeight: w }).map((h) => h.id);
const guard = (tau, K = 60, w = 1) => (lists) => fuseCandidates(poolByDistance(lists), lists.lexical.map((h) => h.id), { rrfK: K, lexicalWeight: w, guardMargin: tau }).map((h) => h.id);
// Agreement-limited: lexical rank only counts for records already in the
// vector top-m (it reorders the vector head instead of importing a pile of
// term matches), unless the lexical list is short (<= s hits survive the
// reader's cutoff), which is the known-item shape: then full RRF.
const agree = (m, s, K = 60, w = 1) => (lists) => {
  if (lists.lexical.length <= s) return rrf(K, w)(lists);
  const lexRank = new Map(lists.lexical.map((h, i) => [h.id, i + 1]));
  const rows = poolByDistance(lists).map((h, i) => ({ ...h, v: i + 1, l: lexRank.get(h.id) || null }));
  return byScore(rows, (r) => 1 / (K + r.v) + (r.l && r.v <= m ? w / (K + r.l) : 0));
};
const augmented = (lists) => poolByDistance(lists).map((r) => r.id);
const vectorOnly = (lists) => lists.vector.map((h) => h.id);
const lexicalFirst = (lists) => [...lists.lexical.map((h) => h.id), ...lists.vector.map((h) => h.id).filter((id) => !lists.lexical.some((h) => h.id === id))];

const RULES = {
  vector: vectorOnly,
  lexicalFirst,
  augmented,
  // the shipped rule (minus phrase pinning), read from the reader itself
  defaults: (lists) => fuseCandidates(poolByDistance(lists), lists.lexical.map((h) => h.id), FUSION_DEFAULTS).map((h) => h.id),
  rrf60: rrf(60, 1),          // equal-weight RRF, the shipped rule before 2026-09-23
  'rrf60 w.5': rrf(60, 0.5),
  rrf10: rrf(10, 1),
  'rrf10 w.5': rrf(10, 0.5),
  'guard.05': guard(0.05),
  'guard.10': guard(0.10),
  'guard.20': guard(0.20),
  'agree10/3': agree(10, 3),
  'agree5/3': agree(5, 3),
  'agree10/1': agree(10, 1),
  // combinations: the K=10 / w=.5 base with a guard on top, and finer tau
  'k10w.5+g.03': guard(0.03, 10, 0.5),
  'k10w.5+g.05': guard(0.05, 10, 0.5),
  'k10w.5+g.08': guard(0.08, 10, 0.5),
  'k60w.5+g.05': guard(0.05, 60, 0.5),
  'k10w.3': rrf(10, 0.3),
  'k10w.3+g.05': guard(0.05, 10, 0.3),
  'agree10/3 k10w.5': agree(10, 3, 10, 0.5),
};

// ---- collect --------------------------------------------------------------
const questions = loadQuestions(JSON.parse(fs.readFileSync(queriesPath, 'utf8'))).filter(isAnswerable);
const search = await openPikeletFile(artifactPath);
const info = search.info();
console.log(`artifact: ${path.basename(artifactPath)} — ${info.records} records, identity ${info.identity?.slice(0, 16)}…`);
console.log(`answerable questions: ${questions.length}  depth: ${depth}\n`);

const rows = [];
let simMatches = 0, simWithin1 = 0, simCompared = 0;
for (const q of questions) {
  const [vec, lex, hyb] = await Promise.all(['vector', 'lexical', 'hybrid'].map((retrieval) =>
    search.query(q.text, { k: depth, retrieval, showAbstained: true })));
  const evidenceIds = new Set();
  for (const r of [...vec.results, ...lex.results, ...hyb.results]) if (hasEvidence(r, q)) evidenceIds.add(r.id);
  const lists = {
    vector: vec.results.map((r) => ({ id: r.id, distance: r.distance })),
    lexical: lex.results.map((r) => ({ id: r.id, distance: r.distance })),
  };
  const rankOf = (ids) => { const i = ids.findIndex((id) => evidenceIds.has(id)); return i === -1 ? null : i + 1; };
  const ranks = {};
  for (const [name, rule] of Object.entries(RULES)) ranks[name] = rankOf(rule(lists));
  const shipped = rankOf(hyb.results.map((r) => r.id));
  ranks.shipped = shipped;
  const oracle = [ranks.vector, rankOf(lists.lexical.map((h) => h.id))].filter((r) => r !== null);
  ranks.oracle = oracle.length ? Math.min(...oracle) : null;
  if (shipped !== null || ranks.defaults !== null) {
    simCompared++;
    if (shipped === ranks.defaults) simMatches++;
    if (shipped !== null && ranks.defaults !== null && Math.abs(shipped - ranks.defaults) <= 1) simWithin1++;
  }
  rows.push({ id: q.id, cls: q.cls, q: q.text, lexicalHits: lists.lexical.length, ranks });
}
await search.close();

// ---- report ---------------------------------------------------------------
console.log(`simulation check: defaults (FUSION_DEFAULTS) reproduces the shipped hybrid rank on ${simMatches}/${simCompared} questions (within ±1: ${simWithin1}); differences come from phrase pinning and pool ties.\n`);
const classes = [...new Set(rows.map((r) => r.cls))];
const names = [...Object.keys(RULES), 'shipped', 'oracle'];
const at = (rs, name, n) => rs.filter((r) => r.ranks[name] !== null && r.ranks[name] <= n).length;
for (const cls of classes) {
  const rs = rows.filter((r) => r.cls === cls);
  console.log(`class ${cls} (n=${rs.length}; median lexical hits after cutoff ${[...rs.map((r) => r.lexicalHits)].sort((a, b) => a - b)[Math.floor(rs.length / 2)]}):`);
  console.table(names.map((name) => ({ rule: name, '@1': at(rs, name, 1), '@3': at(rs, name, 3), '@10': at(rs, name, 10), [`@${depth}`]: at(rs, name, depth) })));
}
if (reportPath) {
  fs.writeFileSync(reportPath, JSON.stringify({ artifact: path.basename(artifactPath), identity: info.identity, depth, rules: names, rows }, null, 2));
  console.log(`report written to ${reportPath}`);
}
