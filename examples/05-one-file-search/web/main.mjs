// Browser host for the one-file reader: HTTP range source + a thin UI.
// The reader, encoder, sketch scan, and calibration are the same modules
// Node runs; kind-3 uses reader-owned WASM kernels, no server-side search.

import { openPikeletFile } from '../pikelet-file-reader.mjs';
import { httpRangeSource } from '../sources.mjs';

// Prefer the content-addressed URL from the pointer (immutable-cacheable,
// safe across rebuilds); the bare filename stays as the fallback for hosts
// serving the file without serve.mjs's pointer route.
const FALLBACK_URL = '/pikelet-docs.pikelet';
async function resolveFileUrl() {
    try {
        const pointer = await (await fetch('/pikelet-latest.json')).json();
        if (typeof pointer?.url === 'string') return pointer.url;
    } catch { /* no pointer route on this host */ }
    return FALLBACK_URL;
}
const FILE_URL = await resolveFileUrl();

const statusEl = document.getElementById('status');
const inputEl = document.getElementById('q');
const verdictEl = document.getElementById('verdict');
const resultsEl = document.getElementById('results');

const source = httpRangeSource(FILE_URL);
await source.init();
const openStart = performance.now();
const search = await openPikeletFile(source, {
    // Resident-scan acceleration: at wiki scale the pure-JS sketch scan is
    // ~500 ms of every warm query. The factory runs in the background after
    // open — the engine entrypoint (and its wasm, a lazy Vite asset) only
    // downloads then, queries serve from the JS scan until it's staged, and
    // a load failure just leaves the JS scan serving. Docs-scale artifacts
    // scan in single-digit ms either way; the reader uses whatever arrives.
    sketchScanner: async (sketch) => {
        const { default: Pikelet } = await import('../../../pikelet.web.mjs');
        return Pikelet.createSketchScanner(sketch, { maxRerank: 4096 });
    },
});
const openMs = performance.now() - openStart;
const info = search.info();
// Devtools affordance: poke the live reader from the console —
// pikeletSearchDemo.search.info().residentScan shows whether the WASM
// scanner has staged ('engine') or the JS scan is serving ('js').
window.pikeletSearchDemo = { search, source };

statusEl.textContent = `${(info.fileBytes / 1048576).toFixed(2)} MiB file - `
    + `open ${openMs.toFixed(0)} ms, ${source.stats.requests} requests / ${(source.stats.bytes / 1024).toFixed(0)} KiB - `
    + `${info.records} chunks - identity ${info.identity.slice(0, 12)}... - hash verified: ${info.residentVerified}`;
inputEl.disabled = false;
inputEl.placeholder = info.sampleQueries[0] || 'ask the docs anything...';

let running = false;
async function run() {
    const text = inputEl.value.trim();
    if (!text || running) return;
    running = true;
    const before = { ...source.stats };
    const t0 = performance.now();
    try {
        const out = await search.query(text, { k: 5 });
        const ms = performance.now() - t0;
        const requests = source.stats.requests - before.requests;
        const kib = (source.stats.bytes - before.bytes) / 1024;
        const badge = `<span class="badge ${out.matchQuality}">${out.matchQuality}</span>`;
        const confidence = out.confidence !== undefined ? ` confidence ${out.confidence.toFixed(3)} -` : '';
        verdictEl.innerHTML = `${badge}${confidence} ${ms.toFixed(0)} ms - ${requests} range requests - ${kib.toFixed(1)} KiB fetched`;
        resultsEl.innerHTML = out.results.length === 0
            ? '<p class="p">No results - the file knows this corpus cannot answer that.</p>'
            : out.results.map((r) => `
                <div class="hit">
                  <div class="t">${escapeHtml(r.title)}</div>
                  <div class="p">${escapeHtml(r.sourcePath)}${r.anchor ? '#' + escapeHtml(r.anchor) : ''} - distance ${r.distance.toFixed(3)}</div>
                  <div class="x">${escapeHtml((r.preview || r.text).slice(0, 220))}...</div>
                </div>`).join('');
    } catch (err) {
        verdictEl.textContent = String(err && err.message ? err.message : err);
    } finally {
        running = false;
    }
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

inputEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') run();
});
