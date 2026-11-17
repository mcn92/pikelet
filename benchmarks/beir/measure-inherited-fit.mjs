#!/usr/bin/env node
// LAYERED_PROFILE.md open question 1: how an inherited abstention fit
// degrades as a chain drifts away from the corpus its base was fit on.
//
// Under `inherited` (5.5) the base's retrieval-signals-v1 scorer runs
// unchanged over the merged window of a whole chain. The gate is
// calibrationDrift = (appended + tombstoned) / records_base against a
// provisional, non-normative readerLimit of 0.20. The question the profile
// asks is not only "how big is the error at 0.20" but whether a
// fraction-of-records gate is the right INPUT at all -- i.e. whether the
// same drift fraction hurts equally when the change is random versus
// topically concentrated.
//
// Method. Split a corpus into a BASE half and a WITHHELD half.
//   1. Fit the real calibrator on BASE (calibrateRetrievalAbstention).
//   2. For each drift level and policy, form a drifted corpus by appending
//      records from WITHHELD and/or tombstoning BASE records.
//   3. Score two probe families through the BASE fit, unchanged:
//        answerable  -- questions built from records LIVE in the drifted
//                       corpus, which SHOULD score >= threshold
//        off-domain  -- questions from a foreign corpus, which SHOULD abstain
//      plus a third that only exists because of drift:
//        appended    -- questions built from records the fit never saw
//   4. Report separation (AUC), the abstain rate on each family, and how
//      many verdicts flip relative to the same probes at drift 0.
//
// What this measures is the FIT's behavior, not BM25: the retrieval side is
// held fixed by re-indexing the drifted corpus each time, so a verdict flip
// is attributable to the scorer seeing a corpus it was not fit on.
//
// Usage:
//   node benchmarks/beir/measure-inherited-fit.mjs <dataset> [--foreign <dataset>]
//     [--base-frac 0.5] [--drifts 0,0.05,0.1,0.2,0.4] [--policy random|topical|both]
//     [--probes 120] [--seed 1] [--json <out>]
//
// Needs the inline encoder (same weights `pikelet compile` uses); it embeds
// the base corpus once and each appended slice once, so runtime is dominated
// by embedding.

import { readFileSync, writeFileSync, existsSync, createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { calibrateRetrievalAbstention } from '../../packages/pikelet/src/calibrate.mjs';
import { createAbstentionScorer } from '../../packages/pikelet-wasm/complete/retrieval-abstention.mjs';
import { buildLexicalSegment } from '../../packages/pikelet-wasm/complete/builder.mjs';
import { openLexicalIndex } from '../../packages/pikelet-wasm/complete/lexical.mjs';
import { inlineEncoderDeclaration } from '../../packages/pikelet/src/complete-build.mjs';
import { embedChunksWithInlineTransformer } from '../../packages/pikelet/src/embed.mjs';
import { createInlineTransformerEmbedder } from '../../packages/pikelet-wasm/complete/inline-transformer.mjs';
import createEncoder from '../../packages/pikelet-wasm/complete/encoder-kernels/encoder.node.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const dataset = args[0];
if (!dataset || dataset.startsWith('--')) {
  console.error('usage: node benchmarks/beir/measure-inherited-fit.mjs <dataset> [--foreign <ds>] [--base-frac 0.5] [--drifts 0,0.05,0.1,0.2,0.4] [--policy both] [--probes 120] [--seed 1] [--json <f>]');
  process.exit(1);
}
const argAfter = (f, d) => { const i = args.indexOf(f); return i === -1 ? d : args[i + 1]; };
const FOREIGN = argAfter('--foreign', dataset === 'scifact' ? 'nfcorpus' : 'scifact');
const BASE_FRAC = Number.parseFloat(argAfter('--base-frac', '0.5'));
const DRIFTS = argAfter('--drifts', '0,0.05,0.1,0.2,0.4').split(',').map(Number).filter((x) => Number.isFinite(x) && x >= 0);
const POLICY = argAfter('--policy', 'both');
const PROBES = Number.parseInt(argAfter('--probes', '120'), 10);
const SEED = Number.parseInt(argAfter('--seed', '1'), 10);
const jsonOut = argAfter('--json', null);
const policies = POLICY === 'both' ? ['random', 'topical'] : [POLICY];

const encoderDir = path.resolve(__dirname, '..', '..', 'examples', 'one-file-search', 'encoder-spike', 'real');
for (const p of [path.join(__dirname, 'work', dataset, 'records.jsonl'), path.join(__dirname, 'work', FOREIGN, 'records.jsonl'),
  path.join(encoderDir, 'vocab.txt'), path.join(encoderDir, 'encoder-weights.bin')]) {
  if (!existsSync(p)) { console.error(`missing ${p}`); process.exit(1); }
}

function mulberry32(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
async function readJsonl(f) {
  const rows = [];
  const rl = createInterface({ input: createReadStream(f), crlfDelay: Infinity });
  for await (const line of rl) { const t = line.trim(); if (t) rows.push(JSON.parse(t)); }
  return rows;
}
const tokenize = (s) => String(s || '').toLowerCase().match(/[a-z0-9]+/g) || [];

const allRecords = await readJsonl(path.join(__dirname, 'work', dataset, 'records.jsonl'));
const foreignRecords = await readJsonl(path.join(__dirname, 'work', FOREIGN, 'records.jsonl'));
const rng = mulberry32(SEED);

// Deterministic shuffle, then split base / withheld.
const order = allRecords.map((_, i) => i);
for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
const baseCount = Math.round(order.length * BASE_FRAC);
const baseIdx = order.slice(0, baseCount);
const withheldIdx = order.slice(baseCount);

const chunkOf = (recIdx, id) => ({
  id, title: (allRecords[recIdx].title || `record ${recIdx}`).trim() || `record ${recIdx}`,
  text: allRecords[recIdx].text, sourcePath: `r${recIdx}.md`,
});
const baseChunks = baseIdx.map((r, i) => chunkOf(r, i));

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------
const config = {
  embedding: { dims: 384, mode: 'inline-transformer', pooling: 'mean', normalize: true, prefixPolicy: { query: '', passage: '' } },
  index: { metric: 'cosine', quantized: true, M: 16, efConstruction: 200, efSearch: 120 },
  runtime: { inlineEncoder: { vocabPath: path.join(encoderDir, 'vocab.txt'), weightsPath: path.join(encoderDir, 'encoder-weights.bin') } },
};
const declaration = inlineEncoderDeclaration(config, config.runtime.inlineEncoder);
const embedder = await createInlineTransformerEmbedder({
  declaration,
  vocabText: readFileSync(path.join(encoderDir, 'vocab.txt'), 'utf8'),
  blob: readFileSync(path.join(encoderDir, 'encoder-weights.bin')),
  createEncoder, verify: false,
});
const embedQuery = async (text) => (await embedder.embed(text)).vector;
const embedWordVecs = async (text) => embedder.embedWords(text);

// Corpus embedding goes through the production worker pool (the same
// function `pikelet compile` calls); the sequential embedder above is kept
// for queries, which is what the runtime query path does.
async function embedAll(texts, label) {
  const t0 = performance.now();
  const vectors = await embedChunksWithInlineTransformer(
    texts.map((text) => ({ text })), config, () => {}, __dirname,
  );
  console.error(`  ${label}: ${texts.length} in ${((performance.now() - t0) / 1000).toFixed(1)}s `
    + `(${(texts.length / ((performance.now() - t0) / 1000)).toFixed(1)}/s)`);
  return vectors;
}

console.error(`base corpus: ${baseChunks.length} records; withheld: ${withheldIdx.length}; foreign: ${foreignRecords.length}`);
console.error('embedding base corpus...');
const baseVectors = await embedAll(baseChunks.map((c) => c.text), 'base');

// ---------------------------------------------------------------------------
// Fit the calibrator ONCE on the base. This is the inherited fit.
// ---------------------------------------------------------------------------
console.error('fitting the base calibrator (this is the fit a chain inherits)...');
// The fit retrieves exactly the way this script scores its probes: exact
// cosine over the float vectors, with coverage read in vector order. Fit and
// probes then see the same evidence, so a measured verdict flip is the
// scorer's, not a retrieval difference between the two (calibrate.mjs takes
// its retrieval as `evidence`; compile passes the reader's own).
const exactEvidence = (vectors) => async (text, { excludeIds = null } = {}) => {
  const qv = await embedQuery(text);
  const scored = [];
  for (let i = 0; i < vectors.length; i++) {
    if (excludeIds && excludeIds.has(i)) continue;
    let dot = 0; const v = vectors[i];
    for (let d = 0; d < qv.length; d++) dot += qv[d] * v[d];
    scored.push({ id: i, distance: 1 - dot });
  }
  scored.sort((a, b) => a.distance - b.distance || a.id - b.id);
  const vector = scored.slice(0, 60);
  return { vector, fused: vector };
};
const fit = await calibrateRetrievalAbstention({
  chunks: baseChunks, config,
  embedQuery, embedWordVecs, evidence: exactEvidence(baseVectors),
  log: (m) => console.error(`  [fit] ${m}`), projectDir: __dirname,
});
if (!fit) { console.error('calibration returned null (corpus cannot support a fit) — pick a larger --base-frac'); process.exit(1); }
const asset = fit.calibrationJson.asset;
const bloomBytes = Buffer.from(fit.calibrationJson.vocabBloomBase64, 'base64');
const scorer = createAbstentionScorer(asset, bloomBytes);
console.error(`fit ready: thresholds ${JSON.stringify(asset.thresholds)}, features ${(asset.features || []).join(',')}`);

// ---------------------------------------------------------------------------
// Probes. Built from titles so they read like real questions, and held
// constant across drift levels so verdict flips are comparable.
// ---------------------------------------------------------------------------
// The answerable family is the fit's OWN retrieval-verified positives
// (calibrate.mjs's goldenQueries): queries it confirmed retrieve their
// source record at build time. Synthetic "what is <title>" strings score
// near zero against this fit -- every family abstained on a first pass --
// so they cannot show a verdict flip and would make the abstain columns
// vacuous.
const goldenTexts = (fit.goldenQueries || []).map((g) => g.text).filter(Boolean);
if (goldenTexts.length < 10) { console.error(`fit produced only ${goldenTexts.length} golden queries; raise --base-frac`); process.exit(1); }
// Off-domain: the same construction applied to the FOREIGN corpus, which is
// how the fit's own foreign negatives are made -- a passage-derived query
// about a subject the corpus does not cover.
const qFromRecord = (r) => String(r.title || r.text || '').replace(/\s+/g, ' ').trim().split(' ').slice(0, 14).join(' ');
// The appended family CANNOT be a fixed global list. Each policy appends a
// different set of records (random takes the head of withheldIdx; topical
// takes a term-matched slice), so a fixed list drawn from one of them leaves
// most probes asking about records the other policy never added -- a first
// sweep had 1/60 of them live under topical and 60/60 under random, and the
// resulting "topical degradation" was that overlap, not the fit. Each
// configuration therefore builds its own appended family from the records it
// actually appends, and the comparison is each policy against its OWN drift
// trend, never cross-policy at one drift level.
const probes = {
  answerable: goldenTexts.slice(0, PROBES).map((t) => ({ text: t, rec: -1 })),
  offDomain: foreignRecords.slice(0, PROBES).map((r) => ({ text: qFromRecord(r), rec: -1 })),
};
console.error('embedding the fixed probe families...');
for (const fam of Object.keys(probes)) {
  for (const p of probes[fam]) p.vector = await embedQuery(p.text);
}
// Per-config appended probes are embedded on demand and cached by record.
const appendedProbeCache = new Map();
async function appendedProbesFor(appendRecs) {
  const picked = appendRecs.slice(0, PROBES);
  const out = [];
  for (const r of picked) {
    if (!appendedProbeCache.has(r)) {
      const text = qFromRecord(allRecords[r]);
      appendedProbeCache.set(r, { text, rec: r, vector: await embedQuery(text) });
    }
    out.push(appendedProbeCache.get(r));
  }
  return out;
}

// ---------------------------------------------------------------------------
// One drift configuration: build the drifted corpus, index it, score probes.
// ---------------------------------------------------------------------------
// Every withheld record any drift level might append, embedded once on the
// pool rather than one at a time inside the sweep.
const maxAppend = Math.min(withheldIdx.length, Math.ceil(baseChunks.length * Math.max(...DRIFTS)));
const withheldVectorCache = new Map();
if (maxAppend > 0) {
  const slice = withheldIdx.slice(0, maxAppend);
  const vecs = await embedAll(slice.map((r) => allRecords[r].text), 'withheld');
  slice.forEach((r, i) => withheldVectorCache.set(r, vecs[i]));
}
async function vectorForWithheld(r) {
  if (!withheldVectorCache.has(r)) withheldVectorCache.set(r, (await embedder.embed(allRecords[r].text)).vector);
  return withheldVectorCache.get(r);
}

function pickTopicalBase(count, seed) {
  // Base records sharing the most common mid-frequency term: a coherent slice.
  const df = new Map();
  const terms = baseIdx.map((r) => new Set(tokenize(allRecords[r].text)));
  for (const s of terms) for (const t of s) df.set(t, (df.get(t) || 0) + 1);
  let best = null;
  for (const [t, d] of df) if (d >= count && (!best || d < best[1])) best = [t, d];
  if (!best) return baseIdx.slice(0, count);
  const hits = [];
  for (let i = 0; i < terms.length; i++) if (terms[i].has(best[0])) hits.push(i);
  return { localIdx: hits.slice(0, count), term: best[0] };
}

async function runConfig(drift, policy) {
  const change = Math.round(baseChunks.length * drift);
  const nAppend = Math.ceil(change / 2);
  const nTombstone = change - nAppend;

  let tombstonedLocal = new Set();
  let appendRecs = [];
  let note = 'no change';
  if (change > 0) {
    if (policy === 'random') {
      const r2 = mulberry32(SEED + 17);
      while (tombstonedLocal.size < Math.min(nTombstone, baseChunks.length - 2)) tombstonedLocal.add(Math.floor(r2() * baseChunks.length));
      appendRecs = withheldIdx.slice(0, Math.min(nAppend, withheldIdx.length));
      note = `random: +${appendRecs.length} / -${tombstonedLocal.size}`;
    } else {
      const picked = pickTopicalBase(Math.min(nTombstone, baseChunks.length - 2), SEED);
      const localIdx = Array.isArray(picked) ? picked : picked.localIdx;
      tombstonedLocal = new Set(localIdx);
      // Appends drawn from one topical slice of the withheld half.
      const wTerms = withheldIdx.map((r) => new Set(tokenize(allRecords[r].text)));
      const dfw = new Map();
      for (const s of wTerms) for (const t of s) dfw.set(t, (dfw.get(t) || 0) + 1);
      let bestW = null;
      for (const [t, d] of dfw) if (d >= nAppend && (!bestW || d < bestW[1])) bestW = [t, d];
      const wHits = [];
      if (bestW) for (let i = 0; i < wTerms.length; i++) if (wTerms[i].has(bestW[0])) wHits.push(withheldIdx[i]);
      appendRecs = (wHits.length ? wHits : withheldIdx).slice(0, Math.min(nAppend, withheldIdx.length));
      note = `topical: +${appendRecs.length} ("${bestW ? bestW[0] : 'n/a'}") / -${tombstonedLocal.size}${Array.isArray(picked) ? '' : ` ("${picked.term}")`}`;
    }
  }

  // The drifted live corpus: surviving base records, then appended ones.
  const liveChunks = [];
  const liveVectors = [];
  for (let i = 0; i < baseChunks.length; i++) if (!tombstonedLocal.has(i)) { liveChunks.push(baseChunks[i]); liveVectors.push(baseVectors[i]); }
  for (const r of appendRecs) { liveChunks.push(chunkOf(r, liveChunks.length)); liveVectors.push(await vectorForWithheld(r)); }

  // Retrieval over the drifted corpus, exact (no ANN approximation), so a
  // verdict change is the scorer's, not the index's.
  // Built so the drifted corpus is indexed the way a real mount would be;
  // the scorer's inputs are the vector window and hydrated passages.
  openLexicalIndex(buildLexicalSegment(liveChunks.map((c) => c.text)).bytes);
  const dim = 384;
  function topK(qv, k) {
    const scored = [];
    for (let i = 0; i < liveVectors.length; i++) {
      let dot = 0; const v = liveVectors[i];
      for (let d = 0; d < dim; d++) dot += qv[d] * v[d];
      scored.push({ i, distance: 1 - dot });
    }
    scored.sort((a, b) => a.distance - b.distance || a.i - b.i);
    return scored.slice(0, k);
  }

  // Every appended probe's source record is live in THIS configuration by
  // construction, so P_app/abstain_app measure the fit, not availability.
  const appendedProbes = await appendedProbesFor(appendRecs);
  const families = { answerable: probes.answerable, appended: appendedProbes, offDomain: probes.offDomain };
  const verdicts = {};
  for (const fam of ['answerable', 'appended', 'offDomain']) {
    const rows = [];
    for (const p of families[fam]) {
      const hits = topK(p.vector, 10);
      const passages = hits.map((h) => liveChunks[h.i].text);
      // scorer.score(queryText, results, passageTexts) -> { p, verdict, signals }.
      // coverageFrac destructures each passage as { heading, body } (see
      // retrieval-abstention.mjs), so raw strings make coverage 0 on every
      // row and drive every probability to the floor -- which is exactly what
      // a first pass here produced. index.mjs splits the record at its first
      // newline; this mirrors that.
      const splitHeadingBody = (text) => {
        const t = String(text || '');
        const nl = t.indexOf('\n');
        return nl === -1 ? { heading: t, body: '' } : { heading: t.slice(0, nl), body: t.slice(nl + 1) };
      };
      const need = scorer.usesPassage ? (scorer.passagesNeeded || 1) : 0;
      const out = await scorer.score(
        p.text,
        hits.map((h) => ({ id: h.i, distance: h.distance })),
        passages.slice(0, need).map(splitHeadingBody),
      );
      rows.push({ prob: Number.isFinite(out.p) ? out.p : null, quality: out.verdict || null });
    }
    verdicts[fam] = rows;
  }
  return { drift, policy, change, note, live: liveChunks.length, verdicts, appendedProbeCount: appendedProbes.length };
}

// AUC of answerable (positive) vs off-domain (negative) by probability.
function auc(pos, neg) {
  const P = pos.filter((r) => Number.isFinite(r.prob)).map((r) => r.prob);
  const N = neg.filter((r) => Number.isFinite(r.prob)).map((r) => r.prob);
  if (!P.length || !N.length) return null;
  let wins = 0;
  for (const p of P) for (const n of N) wins += p > n ? 1 : p === n ? 0.5 : 0;
  return wins / (P.length * N.length);
}
const meanProb = (rows) => {
  const v = rows.filter((r) => Number.isFinite(r.prob)).map((r) => r.prob);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
const abstainRate = (rows) => {
  const q = rows.filter((r) => r.quality);
  return q.length ? q.filter((r) => r.quality === 'abstain').length / q.length : null;
};

const results = [];
for (const policy of policies) {
  for (const drift of DRIFTS) {
    if (drift === 0 && results.some((r) => r.drift === 0)) { results.push({ ...results.find((r) => r.drift === 0), policy }); continue; }
    console.error(`\n--- drift ${drift} (${policy}) ---`);
    const r = await runConfig(drift, policy);
    const row = {
      drift: r.drift, policy: r.policy, change: r.change, live: r.live, note: r.note,
      auc_answerable_vs_offdomain: auc(r.verdicts.answerable, r.verdicts.offDomain),
      // At drift 0 nothing is appended, so this family's source records are
      // absent from the corpus by construction; the number is reported for
      // completeness but is not comparable to the drifted rows.
      auc_appended_vs_offdomain: auc(r.verdicts.appended, r.verdicts.offDomain),
      appended_probes: r.appendedProbeCount,
      // Every appended probe's source is live in this configuration.
      appended_family_sources_live: r.appendedProbeCount > 0,
      mean_prob_answerable: meanProb(r.verdicts.answerable),
      mean_prob_appended: meanProb(r.verdicts.appended),
      mean_prob_offdomain: meanProb(r.verdicts.offDomain),
      abstain_answerable: abstainRate(r.verdicts.answerable),
      abstain_appended: abstainRate(r.verdicts.appended),
      abstain_offdomain: abstainRate(r.verdicts.offDomain),
    };
    console.error(`  ${r.note} | live ${r.live} | AUC ans/off ${row.auc_answerable_vs_offdomain?.toFixed(4)} app/off ${row.auc_appended_vs_offdomain?.toFixed(4)}`);
    results.push(row);
  }
}

embedder.dispose();
const out = {
  dataset, foreign: FOREIGN, base_frac: BASE_FRAC, seed: SEED, probes: PROBES,
  base_records: baseChunks.length,
  fit: { threshold: asset.threshold ?? null, features: asset.features || null },
  note: 'drift splits half appends / half tombstones; probes fixed across levels; retrieval is exact so verdict changes are the fit\'s. '
    + 'The answerable family is the fit\'s own goldenQueries, which are derived from base records: under drift some of their source '
    + 'records are tombstoned, so a fall in mean_prob_answerable conflates "the fit degraded" with "the answer was deleted". '
    + 'Read auc_appended_vs_offdomain for the fit-only signal: the appended family is rebuilt per configuration from the records '
    + 'that configuration appends, so every one of its source records is live and none was seen by the fit. Because the two '
    + 'policies append different records, compare each policy against its OWN drift trend, not cross-policy at one drift level.',
  results,
};
console.log(JSON.stringify(out, null, 2));
if (jsonOut) { writeFileSync(jsonOut, `${JSON.stringify(out, null, 2)}\n`); console.error(`-> ${jsonOut}`); }
