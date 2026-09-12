#!/usr/bin/env node
// Measurement for spec/COMPLETE_PROFILE.md section 8 question 6: what would
// per-row vector commitments cost at the reader's real access pattern?
//
// Method: open the wiki-scale sketch (456k rows, 384 B/row), replay the
// 200-query eval set at the measured operating point (C = recommendedRerank),
// capture the exact candidate-id sets fetchRows() is asked for, and model —
// per digest-page geometry — the reads a verifier would add: pages touched,
// digest bytes under the reader's own gap-coalescing, extra requests,
// resident anchor size, file overhead, and cross-query page reuse. Three
// id-set families separate corpus locality from luck:
//   real        the captured candidate sets;
//   uniform     random ids, same set sizes (no locality);
//   adversarial ids spaced N/C apart (maximal dispersion: every candidate in
//               its own page, zero coalescing — the analytic worst case).
// Also modeled: an interleaved layout (each 16/32-row digest page stored
// immediately before its rows) where digest and row fetches share coalesced
// runs, and a Merkle-proof-per-run upper bound.
//
//   node measure-row-commitments.mjs
//
// Output: tables per workload, poc/row-commitment-results.json, and
// poc/row-commitment-replay.json (per-query run lists for the HTTP replay in
// measure-row-commitments-http.mjs).

import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { PikeletSketchArtifact } = require('../../../pikelet-artifact.js');

const DATA = path.join(here, '..', '..', '..', 'examples', '04-static-wiki-pack', 'data-full');
const SKETCH_PATH = path.join(DATA, 'wiki.pikelet-sketch');
const QUERIES_PATH = path.join(DATA, 'eval-queries.f32');
const DIM = 384;
const ROW_BYTES = DIM; // u8 rows
const ROW_GAP = 16 * 1024; // the reader's default coalescing gap for rows
const PAGE_SIZES = [16, 32, 64, 128, 256, 512, 1024]; // rows per digest page
const DIGEST_BYTES = [16, 32];
const DIGEST_GAPS = [0, 4096, 16384];
const REPLAY_GEOMETRY = { P: 16, D: 16 }; // the recommended point, exported for HTTP replay

function loadQueries() {
    const raw = fs.readFileSync(QUERIES_PATH);
    const f32 = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const queries = [];
    for (let q = 0; q * DIM < f32.length; q++) {
        queries.push(f32.subarray(q * DIM, (q + 1) * DIM));
    }
    return queries;
}

// Coalesce sorted [offset, length] extents with a gap. Returns run count,
// total fetched bytes (runs span their internal gaps), and the run list.
function coalesce(extents, gap) {
    if (extents.length === 0) return { runs: 0, bytes: 0, list: [] };
    const list = [];
    let [runStart, runEnd] = [extents[0][0], extents[0][0] + extents[0][1]];
    for (let i = 1; i < extents.length; i++) {
        const [off, len] = extents[i];
        if (off <= runEnd + gap) {
            runEnd = Math.max(runEnd, off + len);
        } else {
            list.push([runStart, runEnd - runStart]);
            runStart = off;
            runEnd = off + len;
        }
    }
    list.push([runStart, runEnd - runStart]);
    return { runs: list.length, bytes: list.reduce((s, [, l]) => s + l, 0), list };
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

function quantiles(sorted, qs) {
    return qs.map((q) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]);
}

// Separate-region model: rows in the plain vectors region, digest table in
// its own region, fetched independently.
function analyze(label, idSets, N) {
    const perQuery = { rowBytes: [], rowRuns: [], candidates: [] };
    const gapSamples = [];
    for (const ids of idSets) {
        const sorted = [...new Set(ids)].sort((a, b) => a - b);
        perQuery.candidates.push(sorted.length);
        for (let i = 1; i < sorted.length; i++) gapSamples.push(sorted[i] - sorted[i - 1]);
        const { runs, bytes } = coalesce(sorted.map((id) => [id * ROW_BYTES, ROW_BYTES]), ROW_GAP);
        perQuery.rowRuns.push(runs);
        perQuery.rowBytes.push(bytes);
    }
    gapSamples.sort((a, b) => a - b);
    const [g50, g90, g99] = quantiles(gapSamples, [0.5, 0.9, 0.99]);
    const meanRowBytes = mean(perQuery.rowBytes);
    const meanRowRuns = mean(perQuery.rowRuns);
    console.log(`\n=== ${label} ===`);
    console.log(`queries ${idSets.length}, mean candidates/query ${mean(perQuery.candidates).toFixed(0)}, `
        + `id-gap p50/p90/p99: ${g50}/${g90}/${g99} rows`);
    console.log(`row reads (gap ${ROW_GAP / 1024}K): mean ${(meanRowBytes / 1024).toFixed(0)} KiB in ${meanRowRuns.toFixed(0)} runs/query`);
    console.log('rows/pg  dig  | pages/q  reuse% | naive KiB | coalesced KiB (runs) @gap 0 / 4K / 16K | ampl | resident | table');

    const rows = [];
    for (const P of PAGE_SIZES) {
        for (const D of DIGEST_BYTES) {
            const pageBytes = P * D;
            const seen = new Set();
            let pagesTotal = 0;
            let reuseHits = 0;
            const perGap = DIGEST_GAPS.map(() => ({ bytes: [], runs: [] }));
            const pagesPerQuery = [];
            for (const ids of idSets) {
                const pages = [...new Set(ids.map((id) => Math.floor(id / P)))].sort((a, b) => a - b);
                pagesPerQuery.push(pages.length);
                pagesTotal += pages.length;
                for (const p of pages) {
                    if (seen.has(p)) reuseHits++;
                    else seen.add(p);
                }
                const extents = pages.map((p) => [p * pageBytes, pageBytes]);
                DIGEST_GAPS.forEach((gap, gi) => {
                    const { runs, bytes } = coalesce(extents, gap);
                    perGap[gi].bytes.push(bytes);
                    perGap[gi].runs.push(runs);
                });
            }
            const meanPages = mean(pagesPerQuery);
            const co = perGap.map((g) => ({ bytes: mean(g.bytes), runs: mean(g.runs) }));
            const row = {
                P, D,
                meanPages: Number(meanPages.toFixed(1)),
                reusePct: Number((100 * reuseHits / pagesTotal).toFixed(1)),
                naiveKiB: Number((meanPages * pageBytes / 1024).toFixed(1)),
                coalesced: co.map((g) => ({ KiB: Number((g.bytes / 1024).toFixed(1)), runs: Number(g.runs.toFixed(1)) })),
                amplification16K: Number((co[2].bytes / meanRowBytes).toFixed(2)),
                residentKiB: Number((Math.ceil(N / P) * 32 / 1024).toFixed(1)),
                tableMiB: Number((N * D / 1048576).toFixed(2)),
                tableOverheadPct: Number((100 * D / ROW_BYTES).toFixed(1)),
            };
            rows.push(row);
            console.log(
                `${String(P).padStart(6)}  ${String(D).padStart(3)}B | ${String(row.meanPages).padStart(7)} ${String(row.reusePct).padStart(6)}% | ${String(row.naiveKiB).padStart(9)} | `
                + row.coalesced.map((g) => `${String(g.KiB).padStart(7)} (${String(g.runs).padStart(5)})`).join(' / ')
                + ` | ${String(row.amplification16K).padStart(5)}x | ${String(row.residentKiB).padStart(7)}K | ${String(row.tableMiB).padStart(6)}M (${row.tableOverheadPct}%)`);
        }
    }

    const depth = Math.ceil(Math.log2(N));
    const merklePerQuery = meanRowRuns * 2 * depth * 32;
    console.log(`merkle-per-run upper bound: depth ${depth}, mean ${meanRowRuns.toFixed(0)} runs/query -> `
        + `<= ${(merklePerQuery / 1024).toFixed(0)} KiB proofs/query (inline proofs; no resident table, no reuse modeled)`);

    return {
        label,
        queries: idSets.length,
        meanCandidates: mean(perQuery.candidates),
        idGap: { p50: g50, p90: g90, p99: g99 },
        rowReads: { meanKiB: meanRowBytes / 1024, meanRuns: meanRowRuns, gap: ROW_GAP },
        merkleUpperBoundKiB: merklePerQuery / 1024,
        geometries: rows,
    };
}

// Interleaved model: the vectors region is laid out as blocks of
// [P*D digest page][P rows], so a row fetch and its page digest share a
// coalesced run whenever they are within the gap — for P=16, the digest is
// at most 15*384 B before the first needed row, always inside the 16K gap.
function interleavedExtents(ids, P, D) {
    const pageBytes = P * D;
    const blockBytes = pageBytes + P * ROW_BYTES;
    const extents = [];
    const pages = new Set();
    for (const id of [...new Set(ids)].sort((a, b) => a - b)) {
        const page = Math.floor(id / P);
        if (!pages.has(page)) {
            pages.add(page);
            extents.push([page * blockBytes, pageBytes]);
        }
        extents.push([page * blockBytes + pageBytes + (id % P) * ROW_BYTES, ROW_BYTES]);
    }
    extents.sort((a, b) => a[0] - b[0]);
    return extents;
}

function analyzeInterleaved(label, idSets, P, D, gaps = [4096, ROW_GAP]) {
    const out = { label, P, D, gaps: {} };
    console.log(`\n--- interleaved layout (${P} rows + ${D} B digests per block): ${label} ---`);
    for (const gap of gaps) {
        const bytes = [];
        const runs = [];
        const rowOnlyBytes = [];
        const rowOnlyRuns = [];
        for (const ids of idSets) {
            const all = coalesce(interleavedExtents(ids, P, D), gap);
            bytes.push(all.bytes);
            runs.push(all.runs);
            // Rows alone in the same (padded) layout, for a like-for-like base.
            const sorted = [...new Set(ids)].sort((a, b) => a - b);
            const pageBytes = P * D;
            const blockBytes = pageBytes + P * ROW_BYTES;
            const rowsOnly = coalesce(sorted.map((id) => [
                Math.floor(id / P) * blockBytes + pageBytes + (id % P) * ROW_BYTES, ROW_BYTES,
            ]), gap);
            rowOnlyBytes.push(rowsOnly.bytes);
            rowOnlyRuns.push(rowsOnly.runs);
        }
        const extraKiB = (mean(bytes) - mean(rowOnlyBytes)) / 1024;
        const extraRuns = mean(runs) - mean(rowOnlyRuns);
        out.gaps[gap] = {
            totalKiB: Number((mean(bytes) / 1024).toFixed(1)),
            totalRuns: Number(mean(runs).toFixed(1)),
            rowsOnlyKiB: Number((mean(rowOnlyBytes) / 1024).toFixed(1)),
            rowsOnlyRuns: Number(mean(rowOnlyRuns).toFixed(1)),
            extraKiB: Number(extraKiB.toFixed(1)),
            extraRuns: Number(extraRuns.toFixed(1)),
        };
        console.log(`gap ${String(gap / 1024).padStart(2)}K: rows+digests ${(mean(bytes) / 1024).toFixed(0)} KiB in ${mean(runs).toFixed(0)} runs `
            + `(rows alone ${(mean(rowOnlyBytes) / 1024).toFixed(0)} KiB in ${mean(rowOnlyRuns).toFixed(0)} runs) -> `
            + `verification adds ${extraKiB.toFixed(0)} KiB and ${extraRuns.toFixed(1)} extra requests/query`);
    }
    return out;
}

function makeRand(seedInit) {
    let seed = seedInit;
    return () => {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

async function main() {
    const sketch = await PikeletSketchArtifact.openFile(SKETCH_PATH);
    const N = sketch.count;
    console.log(`sketch: ${N} rows, dim ${sketch.dim}, recommendedRerank ${sketch.recommendedRerank}, resident ${(sketch.residentBytes / 1048576).toFixed(1)} MiB`);
    const queries = loadQueries();
    console.log(`queries: ${queries.length}`);

    const original = sketch.fetchRows.bind(sketch);
    let capture = null;
    sketch.fetchRows = async (ids, options) => {
        if (capture) capture.push([...ids]);
        return original(ids, options);
    };

    const results = { generatedFor: 'COMPLETE_PROFILE.md section 8 q6', N, dim: DIM, workloads: [], interleaved: [] };
    const replay = { N, rowBytes: ROW_BYTES, geometry: REPLAY_GEOMETRY, scenarios: {} };

    for (const C of [sketch.recommendedRerank || 600, 120]) {
        capture = [];
        const t0 = Date.now();
        for (const q of queries) await sketch.search(q, 10, { rerank: C });
        const elapsed = Date.now() - t0;
        const real = capture;
        capture = null;
        console.log(`\nworkload C=${C}: ${queries.length} queries in ${(elapsed / 1000).toFixed(1)}s`);

        // Uniform baseline (same sizes) and the analytic worst case: ids
        // spaced N/C apart so every candidate sits in its own digest page
        // and nothing coalesces.
        const rand = makeRand(0xC0FFEE ^ C);
        const uniform = real.map((ids) => {
            const set = new Set();
            while (set.size < ids.length) set.add(Math.floor(rand() * N));
            return [...set];
        });
        const adversarial = real.map((ids, qi) => {
            const stride = Math.floor(N / ids.length);
            const phase = (qi * 7919) % stride;
            return Array.from({ length: ids.length }, (_, i) => (phase + i * stride) % N);
        });

        results.workloads.push(analyze(`real wiki queries, C=${C}`, real, N));
        results.workloads.push(analyze(`uniform baseline, C=${C}`, uniform, N));
        results.workloads.push(analyze(`adversarial (evenly spread), C=${C}`, adversarial, N));

        const { P, D } = REPLAY_GEOMETRY;
        results.interleaved.push(analyzeInterleaved(`real, C=${C}`, real, P, D));
        results.interleaved.push(analyzeInterleaved(`adversarial, C=${C}`, adversarial, P, D));

        // Per-query run lists for the HTTP replay: rows alone; rows plus
        // separate-region digests (gap 0 and 4K on the digest side); and the
        // interleaved layout at the reader's row gap.
        const pageBytes = P * D;
        const tableBase = N * ROW_BYTES; // digest table modeled after the rows region
        replay.scenarios[`C${C}`] = real.map((ids) => {
            const sorted = [...new Set(ids)].sort((a, b) => a - b);
            const rowRuns = coalesce(sorted.map((id) => [id * ROW_BYTES, ROW_BYTES]), ROW_GAP).list;
            const pages = [...new Set(sorted.map((id) => Math.floor(id / P)))].sort((a, b) => a - b);
            const digestExtents = pages.map((p) => [tableBase + p * pageBytes, pageBytes]);
            return {
                rows: rowRuns,
                digests0: coalesce(digestExtents, 0).list,
                digests4k: coalesce(digestExtents, 4096).list,
                interleaved: coalesce(interleavedExtents(sorted, P, D), ROW_GAP).list,
            };
        });
    }

    await sketch.close();
    fs.writeFileSync(path.join(here, 'row-commitment-results.json'), `${JSON.stringify(results, null, 2)}\n`);
    fs.writeFileSync(path.join(here, 'row-commitment-replay.json'), `${JSON.stringify(replay)}\n`);
    console.log('\nresults written to row-commitment-results.json and row-commitment-replay.json');
}

main().catch((err) => { console.error(err); process.exit(1); });
