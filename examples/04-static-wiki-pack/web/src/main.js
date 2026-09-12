// Wiki knowledge-pack demo: the full pipeline in-browser — self-hosted fp16
// MiniLM encoder, resident WASM sketch scan, ranged rerank fetches against a
// static artifact, coalesced hydration, and calibrated abstention — with the
// network meter as the page's centerpiece. Headless hooks (__bench, __probes)
// are load-bearing for CI/measurement and must keep their shapes.
import Pikelet from '../../../../pikelet.web.mjs';
import { pipeline, env } from '@huggingface/transformers';
import { createAbstentionScorer } from './abstention.js';
import './style.css';

const PACK_BASE = '/pack';
const K = 10;
const params = new URLSearchParams(location.search);
const FETCH_PARALLELISM = Number(params.get('p') || 32);
const FETCH_GAP = Number(params.get('gap') || 16384);
const RERANK = params.get('C') ? Number(params.get('C')) : undefined;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/models/';
env.backends.onnx.wasm.wasmPaths = '/ort/';

const SAMPLE_QUERIES = [
    { q: 'what causes earthquakes', note: '' },
    { q: 'what are eggs?', note: 'ranking depth' },
    { q: 'who won the 2026 world cup', note: 'finds the pre-event article' },
    { q: 'reset netgear router admin password', note: 'abstains' },
    { q: 'zzyzx qwfp jkl glorp', note: 'abstains' },
];

const els = Object.fromEntries(
    ['status', 'query', 'go', 'timing', 'results', 'chips', 'bootPanel',
     'bootStages', 'meterSession', 'meterBreakdown', 'meterQuery']
        .map((id) => [id, document.getElementById(id)]));

// ---- meter ----------------------------------------------------------------
const meter = { requests: 0, bytes: 0, note: {} };
function meterAdd(kind, bytes, requests = 1) {
    meter.requests += requests;
    meter.bytes += bytes;
    meter.note[kind] = (meter.note[kind] || 0) + bytes;
    els.meterSession.textContent = `${meter.requests} requests · ${(meter.bytes / 1e6).toFixed(1)} MB`;
    els.meterBreakdown.textContent = Object.entries(meter.note)
        .map(([k, v]) => `${k} ${(v / 1e6).toFixed(1)} MB`).join(' · ');
}

function createHttpRangeSource(url, kind) {
    let fullBuffer = null;
    return {
        async read(offset, length) {
            if (fullBuffer) return fullBuffer.subarray(offset, offset + length);
            // The range rides in the query string as well as the header:
            // Chromium serializes concurrent fetches of one cacheable URL on
            // its cache-entry write lock, so same-URL range reads execute
            // one at a time no matter the requested parallelism (measured
            // live: 5.7-18.6 s per query before this line, 0.6-1.5 s after).
            // Distinct URLs per range dissolve the lock; the server ignores
            // the query for routing and normalizes its own cache key.
            const response = await fetch(`${url}?r=${offset}-${offset + length - 1}`, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
            if (response.status !== 206 && response.status !== 200) {
                throw new Error(`Range read failed: ${response.status}`);
            }
            const bytes = new Uint8Array(await response.arrayBuffer());
            meterAdd(kind, bytes.byteLength);
            if (response.status === 200 && bytes.byteLength > length) {
                fullBuffer = bytes; // host ignored Range: degrade to one full download
                return fullBuffer.subarray(offset, offset + length);
            }
            if (bytes.byteLength !== length) throw new Error(`short range read: ${bytes.byteLength}/${length}`);
            return bytes;
        },
    };
}

// ---- boot -----------------------------------------------------------------
const state = {};
const timings = { boot: {} };

const BOOT_STAGES = [
    ['manifest', 'Pack manifest (version pointer)'],
    ['modelLoad', 'Encoder — MiniLM-L6 fp16, runs in this tab'],
    ['encoderWarmup', 'Encoder warmup'],
    ['packOpen', 'Resident index tier — 4-bit sketches of every chunk'],
    ['scannerBuild', 'WASM SIMD scanner'],
    ['offsetsLoad', 'Corpus offset table'],
    ['abstention', 'Abstention calibration + vocabulary filter'],
];
const activeStages = new Set();
function renderStages() {
    els.bootStages.innerHTML = '';
    for (const [key, label] of BOOT_STAGES) {
        const li = document.createElement('li');
        const done = timings.boot[key] !== undefined;
        li.className = done ? 'done' : activeStages.has(key) ? 'active' : '';
        const name = document.createElement('span');
        name.className = 'stage-name';
        name.textContent = label;
        const detail = document.createElement('span');
        detail.className = 'stage-detail';
        detail.textContent = done ? `${timings.boot[key]} ms` : '';
        li.append(name, detail);
        els.bootStages.appendChild(li);
    }
}
async function stage(key, fn) {
    activeStages.add(key);
    renderStages();
    const t0 = performance.now();
    try {
        return await fn();
    } finally {
        timings.boot[key] = Math.round(performance.now() - t0);
        activeStages.delete(key);
        renderStages();
    }
}

async function boot() {
    // Manifest first: it is the version pointer (short cache), and its
    // packVersion threads a content-hash segment into every other pack URL
    // so those can be cached as immutable without a purge story.
    els.status.textContent = 'Loading the pack — one-time download, cached by your browser.';
    const manifest = await stage('manifest', async () =>
        (await fetch(`${PACK_BASE}/pack-manifest.json`)).json());
    state.manifest = manifest;
    state.packBase = manifest.packVersion ? `${PACK_BASE}/${manifest.packVersion}` : PACK_BASE;

    // Everything after the manifest is independent, so it loads in parallel:
    // the encoder download (the largest asset) fully overlaps the resident
    // fetch — measured sequentially these were ~6 s + ~6 s of a ~13 s cold
    // boot. Time-to-first-query is the max of the branches, not the sum.
    const encoderReady = stage('modelLoad', () =>
        pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'fp16' })
    ).then(async (embedder) => {
        state.embedder = embedder;
        for (const entry of performance.getEntriesByType('resource')) {
            if (entry.name.includes('/models/') || entry.name.includes('/ort/')) {
                meterAdd('encoder', entry.transferSize || entry.encodedBodySize || 0);
            }
        }
        await stage('encoderWarmup', () => embed('warmup query'));
    });

    const packReady = stage('packOpen', () =>
        Pikelet.SketchArtifact.open(createHttpRangeSource(`${state.packBase}/wiki.pikelet-sketch`, 'pack'), {})
    ).then(async (artifact) => {
        state.artifact = artifact;
        state.scanner = await stage('scannerBuild', () => Pikelet.createSketchScanner(artifact));
    });

    const offsetsReady = stage('offsetsLoad', async () => {
        const r = await fetch(`${state.packBase}/corpus-offsets.u32`);
        const b = await r.arrayBuffer();
        meterAdd('offsets', b.byteLength);
        return new Uint32Array(b);
    }).then((offsets) => {
        state.offsets = offsets;
        state.corpusSource = createHttpRangeSource(`${state.packBase}/corpus.bin`, 'hydration');
    });

    const abstentionReady = stage('abstention', () => Promise.all([
        fetch(`${state.packBase}/wiki-abstention.json`).then((r) => r.json()),
        fetch(`${state.packBase}/wiki-vocab.bloom`).then((r) => r.arrayBuffer()),
    ])).then(([abstentionAsset, bloomBuf]) => {
        meterAdd('abstention', bloomBuf.byteLength);
        state.abstention = createAbstentionScorer(abstentionAsset, bloomBuf);
    });

    await Promise.all([encoderReady, packReady, offsetsReady, abstentionReady]);

    els.bootPanel.classList.add('collapsed');
    els.status.innerHTML = '';
    const ready = document.createElement('span');
    ready.textContent = `Ready — ${state.artifact.count.toLocaleString()} chunks from ${manifest.articles.toLocaleString()} articles, `
        + `${(state.artifact.residentBytes / 1e6).toFixed(1)} MB resident`
        + (state.artifact.residentVerified ? ', hash verified ' : ' ');
    els.status.appendChild(ready);
    if (manifest.packVersion) {
        const hash = document.createElement('span');
        hash.className = 'hash';
        hash.textContent = manifest.packVersion;
        els.status.appendChild(hash);
    }
    els.query.disabled = false;
    els.go.disabled = false;
    els.query.focus();
    window.__wikiPackReady = true;
}

async function embed(text) {
    const out = await state.embedder(text, { pooling: 'mean', normalize: true });
    return Float32Array.from(out.data);
}

async function hydrate(ids) {
    // The cluster layout often makes top-k chunks physically adjacent, so
    // coalesce hydration reads the same way the rerank fetch does.
    const spans = ids
        .map((id) => ({ id, start: state.offsets[id], end: state.offsets[id + 1] }))
        .sort((a, b) => a.start - b.start);
    const HYDRATE_GAP = 4096;
    const groups = [];
    for (const s of spans) {
        const g = groups[groups.length - 1];
        if (g && s.start - g.end <= HYDRATE_GAP) {
            g.spans.push(s);
            g.end = Math.max(g.end, s.end);
        } else {
            groups.push({ start: s.start, end: s.end, spans: [s] });
        }
    }
    const byId = new Map();
    await Promise.all(groups.map(async (g) => {
        const bytes = await state.corpusSource.read(g.start, g.end - g.start);
        for (const s of g.spans) {
            byId.set(s.id, JSON.parse(new TextDecoder().decode(bytes.subarray(s.start - g.start, s.end - g.start))));
        }
    }));
    return ids.map((id) => byId.get(id));
}

async function runQuery(text) {
    const t = {};
    const m0 = { requests: meter.requests, bytes: meter.bytes };
    let t0 = performance.now();
    const qv = await embed(text);
    t.embed = performance.now() - t0;

    t0 = performance.now();
    const { results } = await state.artifact.search(qv, K, { scanner: state.scanner, parallelism: FETCH_PARALLELISM, gap: FETCH_GAP, rerank: RERANK });
    t.search = performance.now() - t0;

    const abstention = state.abstention ? state.abstention.score(text, results) : null;

    t0 = performance.now();
    const rows = abstention && abstention.verdict === 'abstain'
        ? [] : await hydrate(results.map((r) => r.id));
    t.hydrate = performance.now() - t0;
    t.total = t.embed + t.search + t.hydrate;
    t.queryRequests = meter.requests - m0.requests;
    t.queryBytes = meter.bytes - m0.bytes;
    return { results, rows, t, abstention };
}

// ---- rendering ------------------------------------------------------------
function renderQuery(text, { results, rows, t, abstention }) {
    els.meterQuery.textContent = `${t.queryRequests} requests · ${(t.queryBytes / 1024).toFixed(0)} KiB · ${t.total.toFixed(0)} ms`;
    els.timing.textContent = `embed ${t.embed.toFixed(0)} · search ${t.search.toFixed(0)} · hydrate ${t.hydrate.toFixed(0)} ms`;
    els.results.innerHTML = '';

    if (abstention && abstention.verdict === 'abstain') {
        const card = document.createElement('div');
        card.className = 'verdict verdict-abstain';
        const head = document.createElement('b');
        head.textContent = 'Nothing useful in this pack for that.';
        const body = document.createElement('span');
        body.textContent = 'Abstaining rather than showing noise — note the meter: no text was even fetched.';
        const fine = document.createElement('div');
        fine.className = 'verdict-fine';
        fine.textContent = `confidence ${abstention.p.toFixed(2)} · best distance ${abstention.signals.d0.toFixed(2)} · known vocabulary ${(abstention.signals.known_frac * 100).toFixed(0)}%`;
        card.append(head, body, fine);
        els.results.appendChild(card);
        return;
    }
    if (abstention && abstention.verdict === 'weak') {
        const card = document.createElement('div');
        card.className = 'verdict verdict-weak';
        card.append('the closest content is distant from the question; showing it anyway.');
        els.results.appendChild(card);
    }
    results.forEach((r, i) => {
        const row = rows[i];
        const div = document.createElement('div');
        div.className = 'hit';
        const title = document.createElement('div');
        title.className = 'hit-title';
        title.textContent = `${i + 1}. `;
        if (row.url && /^https:\/\/simple\.wikipedia\.org\//.test(row.url)) {
            const link = document.createElement('a');
            link.href = row.url;
            link.target = '_blank';
            link.rel = 'noopener';
            link.textContent = row.title;
            title.appendChild(link);
        } else {
            title.append(row.title);
        }
        const dist = document.createElement('span');
        dist.className = 'hit-dist';
        dist.textContent = `distance ${r.distance.toFixed(3)}`;
        title.appendChild(dist);
        const body = document.createElement('div');
        body.className = 'hit-body';
        body.textContent = row.text.slice(row.title.length + 2, row.title.length + 300);
        div.append(title, body);
        els.results.appendChild(div);
    });
}

for (const { q, note } of SAMPLE_QUERIES) {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.append(q);
    if (note) {
        const small = document.createElement('small');
        small.textContent = ` · ${note}`;
        chip.appendChild(small);
    }
    chip.addEventListener('click', () => {
        els.query.value = q;
        els.go.click();
    });
    els.chips.appendChild(chip);
}

els.go.addEventListener('click', async () => {
    const q = els.query.value.trim();
    if (!q || els.go.disabled) return;
    els.go.disabled = true;
    try {
        renderQuery(q, await runQuery(q));
    } catch (e) {
        els.timing.textContent = `query failed: ${e.message}`;
        console.error(e);
    } finally {
        els.go.disabled = false;
    }
});
els.query.addEventListener('keydown', (e) => { if (e.key === 'Enter') els.go.click(); });

// Headless benchmark hook: window.__bench(queries) -> per-query timings.
window.__bench = async (queries) => {
    const out = { boot: timings.boot, queries: [] };
    for (const q of queries) {
        const { results, t, abstention } = await runQuery(q);
        out.queries.push({ q, t, topDistance: results[0]?.distance, verdict: abstention?.verdict });
    }
    out.meter = { requests: meter.requests, bytes: meter.bytes, note: meter.note };
    return out;
};

// Golden-probe hook: verify the shipped calibration in the real browser.
window.__probes = async () => {
    const probes = await (await fetch(`${state.packBase}/wiki-abstention-probes.json`)).json();
    const failures = [];
    for (const probe of probes) {
        const { abstention } = await runQuery(probe.text);
        const expected = Array.isArray(probe.expect) ? probe.expect : [probe.expect];
        if (!expected.includes(abstention.verdict)) {
            failures.push({ text: probe.text, expected, got: abstention.verdict, p: abstention.p });
        }
    }
    return { total: probes.length, failures };
};

// Service worker: caches the versioned pack assets, the encoder, and the
// ONNX runtime so a repeat visit boots from disk. Best-effort — the demo is
// fully functional without it.
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
}

boot().catch((e) => {
    els.status.textContent = `boot failed: ${e.message}`;
    console.error(e);
    window.__wikiPackError = String(e);
});
