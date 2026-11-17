#!/usr/bin/env node
// Measure what LAYERED_PROFILE.md section 5.4 leaves unbounded: how much BM25
// ranking moves when N, avgdl and df still count deleted records.
//
// A layered chain cannot cheaply recompute lexical statistics over the live
// set -- that needs every posting list at mount -- so the profile counts
// tombstoned records in all three and gates the error on record-count drift
// alone. The profile is explicit that this is a change-volume gate implying no
// ranking bound, and open question 3 asks for the size of the effect, with
// rare-term deletions specifically. This measures it.
//
// Method: build two lexical segments over the same corpus with the shipped
// builder, then score the same queries against each.
//   STALE -- postings over live docs only (a chain skips tombstoned docs as
//            postings are scored, 5.3) but N/avgdl/df from the FULL corpus.
//   EXACT -- a from-scratch build over the live set, which is what compaction
//            restores.
// The delta is attributable to the statistics alone: same texts, same
// tokenizer, same BM25 constants, same live document set.
//
// Deletion policies, because the profile's worry is concentration, not volume:
//   random     -- uniform over the corpus, the optimistic case
//   rare-term  -- every doc holding a term whose df is in [1, RARE_DF_MAX],
//                 chosen to maximise df distortion at low drift
//   topical    -- all docs matching a seed term, the realistic "a section of
//                 the docs was removed" case
//
// Usage:
//   node benchmarks/beir/measure-stale-idf.mjs <dataset> [--policy random|rare-term|topical]
//     [--drift 0.02] [--seed 1] [--k 10] [--json <out>]
//
// Reads:  work/<dataset>/records.jsonl, cache/<dataset>/queries.jsonl,
//         cache/<dataset>/qrels/test.tsv
// Writes: nothing unless --json.

import { readFileSync, writeFileSync, existsSync, createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLexicalSegment } from '../../packages/pikelet-wasm/complete/builder.mjs';
import { openLexicalIndex } from '../../packages/pikelet-wasm/complete/lexical.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const dataset = args[0];
if (!dataset || dataset.startsWith('--')) {
  console.error('usage: node benchmarks/beir/measure-stale-idf.mjs <dataset> [--policy ...] [--drift 0.02] [--seed 1] [--k 10] [--json <out>]');
  process.exit(1);
}
const argAfter = (f, d) => { const i = args.indexOf(f); return i === -1 ? d : args[i + 1]; };
const POLICY = argAfter('--policy', 'rare-term');
if (!['random', 'rare-term', 'topical'].includes(POLICY)) {
  console.error(`--policy must be random, rare-term or topical (got ${POLICY})`);
  process.exit(1);
}
const DRIFT = Number.parseFloat(argAfter('--drift', '0.02'));
if (!(DRIFT > 0 && DRIFT < 1)) { console.error('--drift must be in (0,1)'); process.exit(1); }
const SEED = Number.parseInt(argAfter('--seed', '1'), 10);
const K = Number.parseInt(argAfter('--k', '10'), 10);
const RARE_DF_MAX = Number.parseInt(argAfter('--rare-df-max', '3'), 10);
const jsonOut = argAfter('--json', null);

const workDir = path.join(__dirname, 'work', dataset);
const cacheDir = path.join(__dirname, 'cache', dataset);
for (const p of [path.join(workDir, 'records.jsonl'), path.join(cacheDir, 'queries.jsonl'), path.join(cacheDir, 'qrels', 'test.tsv')]) {
  if (!existsSync(p)) { console.error(`missing ${p} — run download.py and convert.mjs for ${dataset} first`); process.exit(1); }
}

function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
async function readJsonl(filePath) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(filePath), crlfDelay: Infinity });
  for await (const line of rl) { const t = line.trim(); if (t) rows.push(JSON.parse(t)); }
  return rows;
}

const records = await readJsonl(path.join(workDir, 'records.jsonl'));
const texts = records.map((r) => r.text);
const beirIds = records.map((r) => String(r._id));

// qrels: query id -> { beirId: grade }
const qrels = new Map();
for (const line of readFileSync(path.join(cacheDir, 'qrels', 'test.tsv'), 'utf8').split('\n').slice(1)) {
  const [qid, cid, score] = line.split('\t');
  if (!qid || !cid) continue;
  const g = Number.parseInt(score, 10);
  if (!Number.isFinite(g) || g <= 0) continue;
  if (!qrels.has(qid)) qrels.set(qid, new Map());
  qrels.get(qid).set(String(cid), g);
}
const allQueries = await readJsonl(path.join(cacheDir, 'queries.jsonl'));
const queries = allQueries.filter((q) => qrels.has(String(q._id)));

// ---------------------------------------------------------------------------
// Pick the deleted set.
// ---------------------------------------------------------------------------
// The builder's own tokenizer is not exported; for SELECTING documents this
// approximation only has to be consistent, since the actual scoring on both
// sides goes through buildLexicalSegment/openLexicalIndex.
const tokenize = (s) => String(s || '').toLowerCase().match(/[a-z0-9]+/g) || [];
const target = Math.max(1, Math.round(records.length * DRIFT));
const rng = mulberry32(SEED);
const deleted = new Set();
let policyNote = '';

if (POLICY === 'random') {
  while (deleted.size < target) deleted.add(Math.floor(rng() * records.length));
  policyNote = 'uniform random';
} else if (POLICY === 'rare-term') {
  // Documents holding the rarest terms: maximises df distortion per deletion,
  // which is exactly the case 5.4 warns about.
  const df = new Map();
  const docTerms = texts.map((t) => new Set(tokenize(t)));
  for (const terms of docTerms) for (const t of terms) df.set(t, (df.get(t) || 0) + 1);
  const rare = [...df.entries()].filter(([, d]) => d >= 1 && d <= RARE_DF_MAX).map(([t]) => t);
  // Deterministic order, then shuffle with the seeded rng.
  rare.sort();
  for (let i = rare.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [rare[i], rare[j]] = [rare[j], rare[i]]; }
  const holders = new Map(); // term -> doc indices
  for (let d = 0; d < docTerms.length; d++) for (const t of docTerms[d]) if (df.get(t) <= RARE_DF_MAX) {
    if (!holders.has(t)) holders.set(t, []); holders.get(t).push(d);
  }
  for (const t of rare) { if (deleted.size >= target) break; for (const d of holders.get(t) || []) { deleted.add(d); if (deleted.size >= target) break; } }
  policyNote = `docs holding terms with df <= ${RARE_DF_MAX}`;
} else {
  // Topical: the largest single-term cluster at or under the target size,
  // so one coherent subject area disappears.
  const df = new Map();
  const docTerms = texts.map((t) => new Set(tokenize(t)));
  for (const terms of docTerms) for (const t of terms) df.set(t, (df.get(t) || 0) + 1);
  let best = null;
  for (const [t, d] of df) if (d <= target && d >= Math.max(2, target * 0.4) && (!best || d > best[1])) best = [t, d];
  if (!best) { console.error('no single term forms a cluster near the target size; try a larger --drift'); process.exit(1); }
  for (let d = 0; d < docTerms.length; d++) if (docTerms[d].has(best[0])) deleted.add(d);
  policyNote = `docs containing "${best[0]}" (df ${best[1]})`;
}

const liveIdx = [];
for (let i = 0; i < records.length; i++) if (!deleted.has(i)) liveIdx.push(i);
const actualDrift = deleted.size / records.length;

// ---------------------------------------------------------------------------
// Build the two indexes.
// ---------------------------------------------------------------------------
// EXACT: a from-scratch build over live docs only -- what compaction restores.
const exactSeg = buildLexicalSegment(liveIdx.map((i) => texts[i]));
const exactIdx = openLexicalIndex(exactSeg.bytes);

// STALE: the full corpus, so N/avgdl/df are the pre-deletion values. A chain
// skips tombstoned docs while scoring postings (5.3), which we reproduce by
// dropping deleted docs from the result list rather than from the statistics.
const fullSeg = buildLexicalSegment(texts);
const fullIdx = openLexicalIndex(fullSeg.bytes);

console.error(`${dataset}: ${records.length} records, ${deleted.size} deleted (${(actualDrift * 100).toFixed(2)}% drift, ${POLICY}: ${policyNote})`);
console.error(`stats: N ${fullIdx.docCount} -> ${exactIdx.docCount}, avgdl ${(fullIdx.totalTokens / fullIdx.docCount).toFixed(1)} -> ${(exactIdx.totalTokens / exactIdx.docCount).toFixed(1)}`);

// ---------------------------------------------------------------------------
// Score.
// ---------------------------------------------------------------------------
const exactLocalToBeir = liveIdx.map((i) => beirIds[i]);
const dcg = (gains) => gains.reduce((s, g, i) => s + g / Math.log2(i + 2), 0);
function ndcgAt(rankedBeirIds, rel, k) {
  const gains = rankedBeirIds.slice(0, k).map((id) => rel.get(id) || 0);
  const ideal = [...rel.values()].sort((a, b) => b - a).slice(0, k);
  const idcg = dcg(ideal);
  return idcg === 0 ? 0 : dcg(gains) / idcg;
}

let sumStale = 0, sumExact = 0, scored = 0;
let top1Changed = 0, topKOverlapSum = 0, maxNdcgDrop = 0, worstQuery = null;
const perQuery = [];

for (const q of queries) {
  const rel = qrels.get(String(q._id));
  // STALE: full-corpus statistics, then drop tombstoned docs from the ranking.
  const staleRaw = fullIdx.search(q.text, K * 8);
  const staleIds = staleRaw.filter((h) => !deleted.has(h.id)).slice(0, K).map((h) => beirIds[h.id]);
  // EXACT: from-scratch over the live set.
  const exactIds = exactIdx.search(q.text, K).map((h) => exactLocalToBeir[h.id]);
  if (!staleIds.length && !exactIds.length) continue;

  const nStale = ndcgAt(staleIds, rel, K);
  const nExact = ndcgAt(exactIds, rel, K);
  sumStale += nStale; sumExact += nExact; scored++;

  if ((staleIds[0] || null) !== (exactIds[0] || null)) top1Changed++;
  const inter = staleIds.filter((id) => exactIds.includes(id)).length;
  topKOverlapSum += exactIds.length ? inter / Math.max(staleIds.length, exactIds.length) : 1;
  const drop = nExact - nStale;
  if (drop > maxNdcgDrop) { maxNdcgDrop = drop; worstQuery = { id: String(q._id), text: String(q.text).slice(0, 70), nStale, nExact }; }
  perQuery.push({ id: String(q._id), nStale, nExact });
}

const out = {
  dataset, policy: POLICY, policy_note: policyNote, seed: SEED, k: K,
  records: records.length, deleted: deleted.size, drift: actualDrift,
  rare_df_max: POLICY === 'rare-term' ? RARE_DF_MAX : null,
  stats: {
    N_stale: fullIdx.docCount, N_exact: exactIdx.docCount,
    avgdl_stale: fullIdx.totalTokens / fullIdx.docCount,
    avgdl_exact: exactIdx.totalTokens / exactIdx.docCount,
  },
  queries_scored: scored,
  ndcg_at_k_stale: sumStale / scored,
  ndcg_at_k_exact: sumExact / scored,
  ndcg_delta: (sumStale - sumExact) / scored,
  top1_changed_frac: top1Changed / scored,
  mean_topk_overlap: topKOverlapSum / scored,
  max_single_query_ndcg_drop: maxNdcgDrop,
  worst_query: worstQuery,
};

console.log(JSON.stringify(out, null, 2));
if (jsonOut) { writeFileSync(jsonOut, `${JSON.stringify(out, null, 2)}\n`); console.error(`-> ${jsonOut}`); }
