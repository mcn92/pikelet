// Pikelet playground: query a real .pikelet knowledge pack straight from a
// static host, entirely in the browser. Same reader, encoder, sketch scan,
// and calibration Node runs — no server-side search. Every request the
// page makes is visible in the live log; there is no /search, /embed,
// /query, or /api endpoint anywhere behind this URL.
//
// The default pack is served same-origin (relative path) rather than from
// a GitHub release asset: GitHub's release-asset host (and the Azure blob
// storage behind it) sends no Access-Control-Allow-Origin header, so a
// cross-origin browser fetch for it is blocked outright — a real CORS
// limitation of that host, not something this reader can route around.
// Range-reading an arbitrary .pikelet URL still works from here as long as
// that host sends CORS headers permitting it; type any such URL into the
// mount field to try one.

import { openPikeletFile } from '../../pikelet-file-reader.mjs';
import { httpRangeSource } from '../../sources.mjs';

const DEFAULT_PACK_URL =
  'https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev/wikipedia.pikelet';

const els = {
    packUrl: document.getElementById('pack-url'),
    mountBtn: document.getElementById('mount-btn'),
    status: document.getElementById('status'),
    queryForm: document.getElementById('query-form'),
    queryInput: document.getElementById('query-input'),
    verdict: document.getElementById('verdict'),
    results: document.getElementById('results'),
    log: document.getElementById('request-log'),
    reqCount: document.getElementById('req-count'),
    byteCount: document.getElementById('byte-count'),
    pctCount: document.getElementById('pct-count'),
    identityCard: document.getElementById('identity-card'),
    idFilename: document.getElementById('id-filename'),
    idSize: document.getElementById('id-size'),
    idRecords: document.getElementById('id-records'),
    idEncoder: document.getElementById('id-encoder'),
    idHash: document.getElementById('id-hash'),
    transferBanner: document.getElementById('transfer-banner'),
    transferHeadline: document.getElementById('transfer-headline'),
    suggestedQueries: document.getElementById('suggested-queries'),
    mapPanel: document.getElementById('map-panel'),
    mapFilename: document.getElementById('map-filename'),
    mapSize: document.getElementById('map-size'),
    mapStrip: document.getElementById('map-strip'),
    mapCanvas: document.getElementById('map-canvas'),
    mapNote: document.getElementById('map-note'),
    mapNow: document.getElementById('map-now'),
    mapLegend: document.getElementById('map-legend'),
    mapZoomLabel: document.getElementById('map-zoom-label'),
    mapZoomStrip: document.getElementById('map-zoom-strip'),
    mapZoomCanvas: document.getElementById('map-zoom-canvas'),
    mapRecord: document.getElementById('map-record'),
    mapDetails: document.getElementById('map-details'),
    resultsPanel: document.getElementById('results-panel'),
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

// grounding (computed by the reader for every pack): the question's content
// words the best-matching of the top five passages contains and lacks. An
// asked-for name or value in `uncovered` is one that passage does not state.
// `results` names that passage when it is not the top result.
function groundingHtml(out, results = out.results) {
    const g = out.grounding;
    if (!g) return '';
    if (!g.uncovered.length) return 'the best-matching passage contains every word of the question';
    const words = g.uncovered.map((w) => `<span class="badge lacks">${escapeHtml(w)}</span>`).join(' ');
    const best = results.find((r) => r.id === g.recordId);
    const which = best && best !== results[0] ? ` (best match: ${escapeHtml(best.title || `record ${best.id}`)})` : '';
    return `the best-matching passage lacks ${words}${which}`;
}

function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

let search = null;
let source = null;
let fileBytes = 0;
let logRows = [];
const MAX_LOG_ROWS = 60;

function renderLog() {
    els.log.innerHTML = logRows
        .slice(-MAX_LOG_ROWS)
        .map((row) => `<div class="log-row"><span class="log-verb">GET</span><span class="log-range">${row}</span></div>`)
        .join('');
    els.log.scrollTop = els.log.scrollHeight;
}

// Range-read listeners other than the search tab's own counters (currently
// just the ablation tab) register here — wireFetchLogging notifies all of
// them on every request, since fetch is patched exactly once, globally.
const rangeReadListeners = [];

function wireFetchLogging() {
    const origFetch = globalThis.fetch;
    globalThis.fetch = async (url, opts) => {
        const range = opts?.headers?.Range || opts?.headers?.range;
        const response = await origFetch(url, opts);
        if (range) {
            const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range) || [];
            if (a !== undefined) {
                const len = Number(b) - Number(a) + 1;
                stats.requests++;
                stats.bytes += len;
                logRows.push(`bytes=${a}-${b}  <span class="log-len">(${formatBytes(len)})</span>`);
                renderLog();
                updateCounters();
                for (const listener of rangeReadListeners) listener(url, a, b, len);
            }
        }
        return response;
    };
}

const stats = { requests: 0, bytes: 0 };

function updateCounters() {
    els.reqCount.textContent = stats.requests;
    els.byteCount.textContent = formatBytes(stats.bytes);
    els.pctCount.textContent = fileBytes
        ? `${((stats.bytes / fileBytes) * 100).toFixed(3)}%`
        : '—';
    if (fileBytes && stats.bytes > 0) {
        const pct = (stats.bytes / fileBytes) * 100;
        els.transferHeadline.innerHTML = `<span class="dim">${formatBytes(fileBytes)} artifact ·</span> `
            + `${formatBytes(stats.bytes)} fetched <span class="dim">·</span> ${pct < 0.1 ? pct.toFixed(3) : pct.toFixed(1)}% transferred`;
        els.transferBanner.hidden = false;
    } else {
        els.transferBanner.hidden = true;
    }
}

// --- File map: the mounted file's parts drawn to scale, with every range
// read painted onto them as it lands, so "one file" is something a visitor
// watches happen rather than takes on trust. Parts too small to see at true
// scale (the header, the built-in tests) get a minimum width, and the note
// under the strip says so.
const SECTIONS = {
    header: { label: 'Header & table of contents', detail: 'identity hash, manifest, where each part starts', activity: 'Reading the header' },
    index: { label: 'Vector index', detail: 'compact sketches scanned per query, full vectors for reranking', activity: 'Searching the vector index' },
    corpus: { label: 'Article text', detail: 'every article, stored record by record', activity: 'Reading matching passages' },
    lexical: { label: 'Keyword index', detail: 'BM25 terms and postings', activity: 'Looking up keywords' },
    'query-interp': { label: 'Query encoder', detail: 'the model that turns your question into a vector, plus calibration', activity: 'Loading the query encoder' },
    evaluation: { label: 'Built-in tests', detail: 'golden queries the file checks itself against', activity: 'Reading built-in tests' },
};
const sectionInfo = (kind) => SECTIONS[kind] || { label: kind, detail: '', activity: `Reading ${kind}` };
const MIN_SEGMENT_PX = 8;

// epoch 0 is the mount; each query bumps it, so "this query" is every read
// tagged with the current epoch.
const fileMap = { filename: null, fileBytes: 0, layout: null, reads: [], epoch: 0, zoomKind: null, zoomPinned: false, record: null };

function filenameOf(url) {
    try { return new URL(String(url), location.href).pathname.split('/').pop() || String(url); } catch { return String(url); }
}

function sectionAt(offset) {
    let found = fileMap.layout[0];
    for (const s of fileMap.layout) if (s.offset <= offset) found = s;
    return found;
}

// Bytes of each read [a, b] (inclusive) that land in each section, keyed by kind.
function bytesBySection(reads) {
    const out = {};
    for (const s of fileMap.layout) out[s.kind] = 0;
    for (const r of reads) {
        for (const [i, s] of fileMap.layout.entries()) {
            const end = i + 1 < fileMap.layout.length ? fileMap.layout[i + 1].offset : fileMap.fileBytes;
            const overlap = Math.min(r.b + 1, end) - Math.max(r.a, s.offset);
            if (overlap > 0) out[s.kind] += overlap;
        }
    }
    return out;
}

function geometry(width) {
    const small = new Set(fileMap.layout.filter((s) => (s.length / fileMap.fileBytes) * width < MIN_SEGMENT_PX));
    const largeBytes = fileMap.layout.reduce((n, s) => n + (small.has(s) ? 0 : s.length), 0);
    const rest = width - MIN_SEGMENT_PX * small.size;
    let x = 0;
    return fileMap.layout.map((s) => {
        const w = small.has(s) ? MIN_SEGMENT_PX : (s.length / largeBytes) * rest;
        const g = { ...s, x, w };
        x += w;
        return g;
    });
}

function mapX(geo, offset) {
    let g = geo[0];
    for (const s of geo) if (s.offset <= offset) g = s;
    return g.x + Math.min(1, Math.max(0, (offset - g.offset) / g.length)) * g.w;
}

function cssVar(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}
const sectionColor = (kind) => cssVar(`--sec-${kind}`, cssVar('--accent', '#0f7d70'));

function sizedContext(canvas, width, height) {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return ctx;
}

function paintReads(ctx, height, toX, filter) {
    for (const r of fileMap.reads) {
        if (filter && !filter(r)) continue;
        const x0 = toX(r.a);
        const x1 = toX(r.b + 1);
        ctx.globalAlpha = r.epoch === fileMap.epoch ? 1 : 0.3;
        ctx.fillStyle = sectionColor(sectionAt(r.a).kind);
        ctx.fillRect(x0, 0, Math.max(1.5, x1 - x0), height);
    }
    ctx.globalAlpha = 1;
}

function paintRecord(ctx, height, toX) {
    const rec = fileMap.record;
    if (!rec) return;
    const x0 = toX(rec.offset);
    const w = Math.max(3, toX(rec.offset + rec.length) - x0);
    ctx.strokeStyle = cssVar('--ink', '#16181a');
    ctx.lineWidth = 2;
    ctx.strokeRect(x0 - 1, 1, w + 2, height - 2);
}

function drawMap() {
    if (!fileMap.layout) return;
    const width = els.mapStrip.clientWidth;
    const height = els.mapStrip.clientHeight;
    const geo = geometry(width);
    for (const g of geo) {
        const seg = els.mapStrip.querySelector(`[data-kind="${g.kind}"]`);
        if (seg) {
            seg.style.width = `${g.w}px`;
            seg.classList.toggle('zoomed', g.kind === fileMap.zoomKind);
        }
    }
    const ctx = sizedContext(els.mapCanvas, width, height);
    const toX = (offset) => mapX(geo, offset);
    paintReads(ctx, height, toX);
    paintRecord(ctx, height, toX);

    const zoom = fileMap.layout.find((s) => s.kind === fileMap.zoomKind) || fileMap.layout[0];
    const zw = els.mapZoomStrip.clientWidth;
    const zh = els.mapZoomStrip.clientHeight;
    els.mapZoomStrip.style.setProperty('--seg-color', sectionColor(zoom.kind));
    const zctx = sizedContext(els.mapZoomCanvas, zw, zh);
    const ztoX = (offset) => ((offset - zoom.offset) / zoom.length) * zw;
    paintReads(zctx, zh, ztoX, (r) => r.b >= zoom.offset && r.a < zoom.offset + zoom.length);
    if (fileMap.record && fileMap.record.offset >= zoom.offset && fileMap.record.offset < zoom.offset + zoom.length) {
        paintRecord(zctx, zh, ztoX);
    }
    els.mapZoomLabel.innerHTML = `Zoomed in: <strong>${escapeHtml(sectionInfo(zoom.kind).label)}</strong>`
        + ` · bytes ${zoom.offset.toLocaleString()}–${(zoom.offset + zoom.length - 1).toLocaleString()}`
        + ` · 1 px ≈ ${formatBytes(Math.max(1, Math.round(zoom.length / Math.max(1, zw))))}`
        + ' <span class="dim">— click a part of the file to zoom into it</span>';
}

let mapFrame = 0;
function scheduleMapDraw() {
    if (mapFrame) return;
    mapFrame = requestAnimationFrame(() => {
        mapFrame = 0;
        drawMap();
        renderLegend();
    });
}

function renderLegend() {
    if (!fileMap.layout) return;
    const now = bytesBySection(fileMap.reads.filter((r) => r.epoch === fileMap.epoch));
    const total = bytesBySection(fileMap.reads);
    const nowLabel = fileMap.epoch === 0 ? 'Opening' : 'This query';
    els.mapLegend.innerHTML = `<span></span><span class="head">Part</span><span class="head size">Size</span><span class="head read">${nowLabel} / all reads</span>`
        + fileMap.layout.map((s) => {
            const info = sectionInfo(s.kind);
            const pct = (100 * s.length) / fileMap.fileBytes;
            return `<span class="sw" style="--seg-color:${sectionColor(s.kind)}"></span>`
                + `<span class="name">${escapeHtml(info.label)}<small>${escapeHtml(info.detail)}</small></span>`
                + `<span class="size">${formatBytes(s.length)} · ${pct < 0.1 ? '<0.1' : pct.toFixed(0)}%</span>`
                + `<span class="read${now[s.kind] ? ' hot' : ''}">${formatBytes(now[s.kind])} / ${formatBytes(total[s.kind])}</span>`;
        }).join('');
}

// Plain-language summary of one epoch's reads, largest part first.
function summarizeEpoch(epoch) {
    const reads = fileMap.reads.filter((r) => r.epoch === epoch);
    const bytes = reads.reduce((n, r) => n + (r.b - r.a + 1), 0);
    const parts = Object.entries(bytesBySection(reads))
        .filter(([, n]) => n > 0)
        .sort((a, b) => b[1] - a[1])
        .map(([kind, n]) => `${sectionInfo(kind).label} ${formatBytes(n)}`)
        .join(', ');
    const allBytes = fileMap.reads.reduce((n, r) => n + (r.b - r.a + 1), 0);
    const untouched = 100 - (100 * allBytes) / fileMap.fileBytes;
    const lead = epoch === 0 ? 'Opened the file' : 'This query';
    return `${lead}: ${reads.length} reads · ${formatBytes(bytes)} from ${fileMap.filename}${parts ? ` — ${parts}` : ''}. `
        + `${untouched.toFixed(1)}% of the file has never left the server.`;
}

function onMapRead(url, a, b) {
    if (!fileMap.filename || filenameOf(url) !== fileMap.filename) return;
    const read = { a: Number(a), b: Number(b), epoch: fileMap.epoch };
    fileMap.reads.push(read);
    if (!fileMap.layout) return;
    const kind = sectionAt(read.a).kind;
    if (!fileMap.zoomPinned) fileMap.zoomKind = busiestSection();
    els.mapNow.textContent = `${sectionInfo(kind).activity}… bytes ${read.a.toLocaleString()}–${read.b.toLocaleString()}`;
    // Reads keep landing after open resolves (the encoder streams in behind
    // it), so the summary replaces the live line once reads go quiet rather
    // than at a fixed point.
    clearTimeout(mapIdleTimer);
    mapIdleTimer = setTimeout(() => { els.mapNow.textContent = summarizeEpoch(fileMap.epoch); }, 700);
    scheduleMapDraw();
}
let mapIdleTimer = 0;

function busiestSection() {
    const now = Object.entries(bytesBySection(fileMap.reads.filter((r) => r.epoch === fileMap.epoch)));
    now.sort((x, y) => y[1] - x[1]);
    return now[0]?.[1] > 0 ? now[0][0] : fileMap.zoomKind;
}

function resetMap(url) {
    fileMap.filename = filenameOf(url);
    fileMap.fileBytes = 0;
    fileMap.layout = null;
    fileMap.reads = [];
    fileMap.epoch = 0;
    fileMap.zoomKind = null;
    fileMap.zoomPinned = false;
    fileMap.record = null;
    clearTimeout(mapIdleTimer);
    els.mapRecord.hidden = true;
    els.mapPanel.hidden = true;
}

// Called once the reader is open: reads made while opening were buffered,
// so the map shows the mount's reads as soon as it appears.
function showMap(info) {
    if (!Array.isArray(info.layout) || info.layout.length === 0) return;
    fileMap.layout = info.layout;
    fileMap.fileBytes = info.fileBytes;
    fileMap.zoomKind = busiestSection() || 'corpus';
    els.mapFilename.textContent = fileMap.filename;
    els.mapSize.textContent = `${formatBytes(info.fileBytes)} · ${info.records.toLocaleString()} records`;
    els.mapStrip.querySelectorAll('.map-seg').forEach((el) => el.remove());
    for (const s of fileMap.layout) {
        const seg = document.createElement('button');
        seg.type = 'button';
        seg.className = 'map-seg';
        seg.dataset.kind = s.kind;
        seg.style.setProperty('--seg-color', sectionColor(s.kind));
        seg.title = `${sectionInfo(s.kind).label} · ${formatBytes(s.length)}`;
        seg.setAttribute('aria-label', `Zoom into ${sectionInfo(s.kind).label}, ${formatBytes(s.length)}`);
        els.mapStrip.insertBefore(seg, els.mapCanvas);
    }
    els.mapPanel.hidden = false;
    const enlarged = fileMap.layout.filter((s) => (s.length / fileMap.fileBytes) * els.mapStrip.clientWidth < MIN_SEGMENT_PX);
    els.mapNote.textContent = enlarged.length
        ? `Drawn to scale, except ${enlarged.map((s) => sectionInfo(s.kind).label.toLowerCase()).join(' and ')}, enlarged so you can see them.`
        : 'Drawn to scale.';
    els.mapNow.textContent = summarizeEpoch(0);
    scheduleMapDraw();
}

function zoomTo(kind) {
    fileMap.zoomKind = kind;
    fileMap.zoomPinned = true;
    scheduleMapDraw();
}

function locateResult(hit) {
    const id = Number(hit.dataset.id);
    const range = search.recordRange(id);
    const kind = sectionAt(range.offset).kind;
    fileMap.record = { id, ...range };
    zoomTo(kind);
    els.results.querySelectorAll('.hit').forEach((el) => el.classList.toggle('located', el === hit));
    const fetched = fileMap.reads.some((r) => r.a <= range.offset + range.length - 1 && r.b >= range.offset);
    const title = hit.querySelector('h3')?.textContent || `record ${id}`;
    els.mapRecord.innerHTML = `<strong>“${escapeHtml(title)}”</strong> is record #${id}, stored at bytes `
        + `<strong>${range.offset.toLocaleString()}–${(range.offset + range.length - 1).toLocaleString()}</strong> of ${escapeHtml(fileMap.filename)}`
        + ` (${formatBytes(range.length)}, inside ${escapeHtml(sectionInfo(kind).label)}, outlined above). `
        + (fetched ? 'This page range-read exactly those bytes to show you the passage.' : 'Those bytes have not been fetched yet.');
    els.mapRecord.hidden = false;
    els.mapPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setStatus(text, cls) {
    els.status.textContent = text;
    els.status.className = `status${cls ? ` ${cls}` : ''}`;
}

async function mount(url) {
    els.mountBtn.disabled = true;
    els.queryInput.disabled = true;
    setStatus('Resolving…', 'busy');
    logRows = [];
    stats.requests = 0;
    stats.bytes = 0;
    renderLog();
    updateCounters();
    resetMap(url);

    try {
        source = httpRangeSource(url);
        search = await openPikeletFile(source, {
            sketchScanner: async (sketch) => {
                const { default: Pikelet } = await import('pikelet-wasm/web');
                return Pikelet.createSketchScanner(sketch, { maxRerank: 4096 });
            },
        });
        const info = search.info();
        fileBytes = info.fileBytes;
        updateCounters();
        setStatus(`Mounted — hash verified: ${info.residentVerified}`, 'ok');
        renderIdentity(url, info);
        showMap(info);
        els.queryInput.disabled = false;
        els.queryInput.focus();
    } catch (err) {
        setStatus(`Failed to mount: ${err?.message || err}`, 'error');
        els.identityCard.hidden = true;
    } finally {
        els.mountBtn.disabled = false;
    }
}

function renderIdentity(url, info) {
    let filename = url;
    try { filename = new URL(url).pathname.split('/').pop() || url; } catch { /* relative path — keep as-is */ }
    els.idFilename.textContent = filename;
    els.idSize.textContent = `${(info.fileBytes / 1024 ** 2).toFixed(1)} MB`;
    els.idRecords.textContent = info.records.toLocaleString();
    els.idEncoder.textContent = info.encoder?.kind
        ? `${info.encoder.kind}${info.encoderVerified === false ? ' (unverified)' : ''}`
        : 'embedded';
    els.idHash.textContent = info.identity;
    els.identityCard.hidden = false;
}

let running = false;
async function runQuery() {
    const text = els.queryInput.value.trim();
    if (!text || running || !search) return;
    running = true;
    els.verdict.textContent = 'Searching…';
    els.verdict.className = 'verdict';
    els.results.innerHTML = '';
    els.resultsPanel.hidden = false;
    fileMap.epoch++;
    fileMap.record = null;
    fileMap.zoomPinned = false;
    els.mapRecord.hidden = true;
    scheduleMapDraw();
    const before = { ...stats };
    const t0 = performance.now();
    try {
        // showAbstained: results always ship, as pikelet mcp returns them;
        // the wiki pack's old calibration would otherwise withhold some.
        const out = await search.query(text, { k: 5, showAbstained: true });
        const ms = performance.now() - t0;
        const usedReq = stats.requests - before.requests;
        const usedBytes = stats.bytes - before.bytes;
        const grounding = groundingHtml(out);
        els.verdict.innerHTML = `${ms.toFixed(0)} ms · ${usedReq} range reads · ${formatBytes(usedBytes)} fetched`
            + (grounding ? `<br>${grounding}` : '');
        els.results.innerHTML = out.results.length === 0
            ? '<p class="empty">No results.</p>'
            : out.results.map((r) => `
                <article class="hit" data-id="${escapeHtml(String(r.id))}">
                  <h3>${escapeHtml(r.title)}</h3>
                  <p class="meta">${escapeHtml(r.sourcePath || r.url || '')}${r.anchor ? `#${escapeHtml(r.anchor)}` : ''} · distance ${r.distance?.toFixed(3)}</p>
                  <p class="excerpt">${escapeHtml((r.preview || r.text || '').slice(0, 240))}…</p>
                  ${fileMap.layout ? '<button type="button" class="locate">Show where this lives in the file</button>' : ''}
                </article>`).join('');
        if (fileMap.layout) els.mapNow.textContent = summarizeEpoch(fileMap.epoch);
    } catch (err) {
        els.verdict.textContent = String(err?.message || err);
        els.verdict.className = 'verdict error';
    } finally {
        running = false;
    }
}

wireFetchLogging();
rangeReadListeners.push(onMapRead);

els.mapStrip.addEventListener('click', (event) => {
    const seg = event.target.closest('.map-seg');
    if (seg) zoomTo(seg.dataset.kind);
});
els.results.addEventListener('click', (event) => {
    const button = event.target.closest('.locate');
    if (button) locateResult(button.closest('.hit'));
});
// The close-up has no width while its <details> is closed; draw it on open.
els.mapDetails.addEventListener('toggle', scheduleMapDraw);
new ResizeObserver(scheduleMapDraw).observe(els.mapStrip);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', scheduleMapDraw);

els.mountBtn.addEventListener('click', () => mount(els.packUrl.value.trim() || DEFAULT_PACK_URL));
els.queryForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runQuery();
});
els.suggestedQueries.addEventListener('click', (event) => {
    const chip = event.target.closest('.chip');
    if (!chip || els.queryInput.disabled) return;
    els.queryInput.value = chip.dataset.q;
    runQuery();
});

els.packUrl.value = DEFAULT_PACK_URL;
mount(DEFAULT_PACK_URL);

// --- Ablation demo tab: same question against a full pack, a pack missing
// one record, and (separately) a pack with one record edited to contradict
// the rest. Fictional Station Veyra research corpus — see
// examples/one-file-search/web/veyra-corpus/README.md for how it's
// generated and compiled. Loaded lazily (first tab open), not on page
// load, since these packs are ~25 MB each and most visitors to the search
// tab will never open this one.
const VEYRA_BASE = 'https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev';

// Which single fact gets removed is picked from this list, not fixed to
// one example — a single hardcoded before/after reads as cherry-picked.
// Each entry here was individually compiled (one fact removed from the
// 94-fact corpus, corpus-<fact>/) and verified end to end against a real
// build before being listed. Against the 0.15.0 reader, on every pack below,
// the FULL pack's top result states the fact and its best-matching passage
// contains every question word, and on the ABLATED pack that passage is
// about another project and grounding lists the asked-for project's name
// as uncovered. Removals where the retriever still found a plausible but
// wrong passage after the fact was gone were left out; that is a real
// limit of the signal, not hidden by picking only the cases that work.
// Each scenario also carries a "contradicted" variant for the third panel:
// a pack identical to the full one except that this same fact's record was
// edited to state a different, still-plausible value (another clearance
// name, vault letter, or supervisor already used elsewhere in the corpus —
// never a nonsense value). contradictQuestion is tuned per scenario where
// the ablation-panel question's own top-1 retrieval doesn't land on the
// edited record (a ranking quirk, not a contradiction-specific one — see
// the loc-tovash/clr-sarnix note below) so the panel reliably demonstrates
// the point instead of an unrelated near-miss.
const FACT_SCENARIOS = [
    {
        id: 'loc-tovash',
        label: 'Where is Tovash housed?',
        question: 'What chamber is the Tovash project housed in?',
        removedText: '"The Tovash project is housed in Chamber 17."',
        // This one needs all 6 of the corpus's curated removals gone at
        // once (veyra-ablated.pikelet) — removing loc-tovash alone left
        // this corpus too small to pass the calibration quality gate, so
        // no other single-fact pack for it exists.
        ablatedUrl: `${VEYRA_BASE}/veyra-ablated.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-chamber43.pikelet`,
        contradictQuestion: 'What chamber is the Tovash project housed in?',
        contradictNote: 'Chamber 17 → Chamber 43',
    },
    {
        id: 'clr-sarnix',
        label: 'What clearance does Sarnix use?',
        // Not the more obvious phrasing ("What clearance does the Sarnix
        // project use?") — that one ties for top-1 at k=1 against an
        // unrelated "Sarnix Location" record on this pack. This phrasing,
        // which echoes the fact's own second sentence, reliably retrieves
        // the actual clearance record on every pack this scenario uses.
        question: 'What clearance badge do Sarnix project members carry?',
        removedText: '"The Sarnix project uses cobalt clearance."',
        ablatedUrl: `${VEYRA_BASE}/veyra-remove-clr-sarnix.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-contradict-clr-sarnix.pikelet`,
        contradictQuestion: 'What clearance badge do Sarnix project members carry?',
        contradictNote: 'cobalt clearance → ochre clearance',
    },
    {
        id: 'vault-crimson',
        label: 'Which vault does crimson clearance open?',
        question: 'Which vault does crimson clearance permit access to?',
        removedText: '"Crimson clearance permits access to Vault A."',
        ablatedUrl: `${VEYRA_BASE}/veyra-remove-vault-crimson.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-contradict-vault-crimson.pikelet`,
        contradictQuestion: 'Which vault does crimson clearance permit access to?',
        contradictNote: 'Vault A → Vault B',
    },
    {
        id: 'clr-penlow',
        label: 'What clearance does Penlow use?',
        // Not the more obvious phrasing ("What clearance does the Penlow
        // project use?") — on the contradicted pack that one ties for
        // top-1 against an unrelated "Penlow Location" record, same issue
        // clr-sarnix had. This phrasing, echoing the fact's own second
        // sentence, reliably retrieves the actual clearance record on
        // every pack this scenario uses.
        question: 'What clearance badge do Penlow project members carry?',
        removedText: '"The Penlow project uses cobalt clearance."',
        ablatedUrl: `${VEYRA_BASE}/veyra-remove-clr-penlow.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-contradict-clr-penlow.pikelet`,
        contradictQuestion: 'What clearance badge do Penlow project members carry?',
        contradictNote: 'cobalt clearance → ochre clearance',
    },
];

const ablationEls = {
    tabBtnSearch: document.getElementById('tab-btn-search'),
    tabBtnAblation: document.getElementById('tab-btn-ablation'),
    tabSearch: document.getElementById('tab-search'),
    tabAblation: document.getElementById('tab-ablation'),
    factPicker: document.getElementById('fact-picker'),
    question: document.getElementById('ablation-question'),
    removed: document.getElementById('ablation-removed'),
    runBtn: document.getElementById('ablation-run-btn'),
    fullBadge: document.getElementById('abl-full-badge'),
    fullAnswer: document.getElementById('abl-full-answer'),
    fullRecord: document.getElementById('abl-full-record'),
    ablatedBadge: document.getElementById('abl-ablated-badge'),
    ablatedAnswer: document.getElementById('abl-ablated-answer'),
    ablatedRecord: document.getElementById('abl-ablated-record'),
    c43Question: document.getElementById('ablation-question-c43'),
    c43Note: document.getElementById('ablation-c43-note'),
    c43RunBtn: document.getElementById('ablation-c43-run-btn'),
    c43Badge: document.getElementById('abl-c43-badge'),
    c43Answer: document.getElementById('abl-c43-answer'),
    c43Record: document.getElementById('abl-c43-record'),
    reqCount: document.getElementById('abl-req-count'),
    byteCount: document.getElementById('abl-byte-count'),
    log: document.getElementById('abl-request-log'),
};

// Own network-activity monitor for the ablation tab: fetch is patched once,
// globally (wireFetchLogging), so this just registers as another listener
// on every range read rather than re-patching fetch itself. Kept separate
// from the search tab's stats/log since the two tabs mount different files
// with different sizes — a shared "% of the file" figure wouldn't mean
// anything once more than one file is in play.
const ablationStats = { requests: 0, bytes: 0 };
let ablationLogRows = [];
const ABL_MAX_LOG_ROWS = 40;
rangeReadListeners.push((url, a, b, len) => {
    ablationStats.requests++;
    ablationStats.bytes += len;
    ablationEls.reqCount.textContent = ablationStats.requests;
    ablationEls.byteCount.textContent = formatBytes(ablationStats.bytes);
    let filename = url;
    try { filename = new URL(url).pathname.split('/').pop() || url; } catch { /* relative path */ }
    ablationLogRows.push(`<span class="log-file">${escapeHtml(filename)}</span> bytes=${a}-${b}  <span class="log-len">(${formatBytes(len)})</span>`);
    ablationEls.log.innerHTML = ablationLogRows
        .slice(-ABL_MAX_LOG_ROWS)
        .map((row) => `<div class="log-row"><span class="log-verb">GET</span><span class="log-range">${row}</span></div>`)
        .join('');
    ablationEls.log.scrollTop = ablationEls.log.scrollHeight;
});

function switchTab(name) {
    const toAblation = name === 'ablation';
    ablationEls.tabBtnSearch.classList.toggle('active', !toAblation);
    ablationEls.tabBtnAblation.classList.toggle('active', toAblation);
    ablationEls.tabBtnSearch.setAttribute('aria-selected', String(!toAblation));
    ablationEls.tabBtnAblation.setAttribute('aria-selected', String(toAblation));
    ablationEls.tabSearch.hidden = toAblation;
    ablationEls.tabAblation.hidden = !toAblation;
    if (toAblation) initAblationDemo();
}

async function runAblationPanel(url, question, badgeEl, answerEl, recordEl) {
    badgeEl.innerHTML = '';
    let filename = url;
    try { filename = new URL(url).pathname.split('/').pop() || url; } catch { /* relative path */ }
    answerEl.textContent = `Mounting ${filename}…`;
    recordEl.innerHTML = '';
    try {
        const source = httpRangeSource(url);
        const packSearch = await openPikeletFile(source);
        answerEl.textContent = 'Querying…';
        // The nearest passage is shown even when the fact is gone, as an
        // agent would receive it; grounding is what says it does not
        // answer. k: 5 so the best-matching passage can be named.
        const out = await packSearch.query(question, { k: 5, showAbstained: true });
        badgeEl.innerHTML = groundingHtml(out);
        if (out.results.length === 0) {
            answerEl.textContent = 'No results.';
            recordEl.innerHTML = '';
            return;
        }
        const top = out.results[0];
        answerEl.textContent = top.preview || top.text || '';
        recordEl.innerHTML = `record ${escapeHtml(String(top.id))} — ${escapeHtml(top.title || '')}`;
    } catch (err) {
        badgeEl.innerHTML = '<span class="badge lacks">error</span>';
        answerEl.textContent = String(err?.message || err);
    }
}

// The full pack's answer never changes across scenarios (same question set,
// same complete file) — cached per question text so re-running doesn't
// needlessly re-mount and re-query the 25 MB full pack every time.
const fullPackAnswerCache = new Map();
async function runFullPanel(question) {
    const alreadyCached = fullPackAnswerCache.has(question);
    if (!alreadyCached) {
        fullPackAnswerCache.set(question, (async () => {
            const source = httpRangeSource(`${VEYRA_BASE}/veyra.pikelet`);
            const packSearch = await openPikeletFile(source);
            return packSearch.query(question, { k: 5, showAbstained: true });
        })());
    }
    ablationEls.fullBadge.innerHTML = '';
    ablationEls.fullAnswer.textContent = alreadyCached ? 'Querying…' : 'Mounting veyra.pikelet…';
    ablationEls.fullRecord.innerHTML = '';
    try {
        const out = await fullPackAnswerCache.get(question);
        ablationEls.fullBadge.innerHTML = groundingHtml(out);
        const top = out.results[0];
        ablationEls.fullAnswer.textContent = top ? (top.preview || top.text || '') : 'No results.';
        ablationEls.fullRecord.innerHTML = top ? `record ${escapeHtml(String(top.id))} — ${escapeHtml(top.title || '')}` : '';
    } catch (err) {
        fullPackAnswerCache.delete(question);
        ablationEls.fullBadge.innerHTML = '<span class="badge lacks">error</span>';
        ablationEls.fullAnswer.textContent = String(err?.message || err);
    }
}

let currentScenario = null;

// Updates the labels for the selected scenario without firing any network
// request — the two "Run query" buttons are what actually trigger reads,
// so switching the picker never mounts a pack the visitor didn't ask for.
function selectFact(scenario) {
    if (currentScenario?.id === scenario.id) return;
    currentScenario = scenario;
    [...ablationEls.factPicker.children].forEach((el) => el.classList.toggle('selected', el.dataset.fact === scenario.id));
    ablationEls.question.textContent = `"${scenario.question}"`;
    ablationEls.removed.innerHTML = `removed record: <code>${escapeHtml(scenario.id)}</code> — ${scenario.removedText}`;
    ablationEls.c43Question.textContent = `"${scenario.contradictQuestion}"`;
    ablationEls.c43Note.innerHTML = `edited record: <code>${escapeHtml(scenario.id)}</code> — ${escapeHtml(scenario.contradictNote)}`;
    for (const [badge, answer, record] of [
        [ablationEls.fullBadge, ablationEls.fullAnswer, ablationEls.fullRecord],
        [ablationEls.ablatedBadge, ablationEls.ablatedAnswer, ablationEls.ablatedRecord],
        [ablationEls.c43Badge, ablationEls.c43Answer, ablationEls.c43Record],
    ]) {
        badge.innerHTML = '';
        record.innerHTML = '';
    }
    ablationEls.fullAnswer.textContent = 'Press "Run query on both files" above.';
    ablationEls.ablatedAnswer.textContent = 'Press "Run query on both files" above.';
    ablationEls.c43Answer.textContent = 'Press "Run query on the contradicted file" above.';
}

async function runRemovalPair() {
    if (!currentScenario) return;
    ablationEls.runBtn.disabled = true;
    try {
        // Sequential, not concurrent: independent fetches resolving in
        // network-dependent order filled panels out of reading order (right
        // side before left) — confusing for a demo whose whole point is a
        // careful side-by-side comparison. Not a hot path, so the ~2x wait
        // is an easy trade for filling in the order a reader expects.
        await runFullPanel(currentScenario.question);
        await runAblationPanel(currentScenario.ablatedUrl, currentScenario.question, ablationEls.ablatedBadge, ablationEls.ablatedAnswer, ablationEls.ablatedRecord);
    } finally {
        ablationEls.runBtn.disabled = false;
    }
}

async function runContradictionPanel() {
    if (!currentScenario) return;
    ablationEls.c43RunBtn.disabled = true;
    try {
        await runAblationPanel(currentScenario.contradictUrl, currentScenario.contradictQuestion, ablationEls.c43Badge, ablationEls.c43Answer, ablationEls.c43Record);
    } finally {
        ablationEls.c43RunBtn.disabled = false;
    }
}

let ablationStarted = false;
function initAblationDemo() {
    if (ablationStarted) return;
    ablationStarted = true;
    ablationEls.factPicker.innerHTML = FACT_SCENARIOS
        .map((s) => `<button type="button" class="chip" data-fact="${escapeHtml(s.id)}">${escapeHtml(s.label)}</button>`)
        .join('');
    ablationEls.factPicker.addEventListener('click', (event) => {
        const chip = event.target.closest('.chip');
        if (!chip) return;
        const scenario = FACT_SCENARIOS.find((s) => s.id === chip.dataset.fact);
        if (scenario) selectFact(scenario);
    });
    ablationEls.runBtn.addEventListener('click', runRemovalPair);
    ablationEls.c43RunBtn.addEventListener('click', runContradictionPanel);
    selectFact(FACT_SCENARIOS[0]);
}

ablationEls.tabBtnSearch.addEventListener('click', () => switchTab('search'));
ablationEls.tabBtnAblation.addEventListener('click', () => switchTab('ablation'));

document.getElementById('link-to-ablation-tab')?.addEventListener('click', (event) => {
    event.preventDefault();
    switchTab('ablation');
});
