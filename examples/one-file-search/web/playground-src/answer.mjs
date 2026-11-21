// Pikelet answers: search the Wikipedia pack in the browser, then have
// SmolLM2 answer from the top three passages in a worker (llm/worker.mjs,
// the decoder spike's Llama kernel). The prompt and the name check are the
// ones the spike scored (decoder-spike/rag_ref.py, rag_score.mjs gate
// "names"): 54/70 answerable questions right with 360M u8 from the top
// three passages, and every question about something fictional declined.

import { openPikeletFile } from '../../pikelet-file-reader.mjs';
import { httpRangeSource } from '../../sources.mjs';

const R2 = 'https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev';
const PACK_URL = `${R2}/wikipedia.pikelet`;
// The weights and tokenizer files; the worker checks them against the hashes
// in llm/models.json, served from this page's origin.
const MODEL_BASE = `${R2}/models/`;
const SYSTEM = 'Answer the question using only the passages below. Reply with a short answer taken from the passages. '
    + 'If the passages do not contain the answer, reply exactly: Not in the passages.';
const PASSAGES = 3;

const $ = (id) => document.getElementById(id);
const els = {
    loadBtn: $('load-btn'), modelStatus: $('model-status'), progress: $('load-progress'),
    input: $('query-input'), askBtn: $('ask-btn'), form: $('query-form'), packStatus: $('pack-status'),
    chips: $('suggested-queries'), answerPanel: $('answer-panel'), answer: $('answer'),
    anyway: $('anyway-btn'), stats: $('answer-stats'), passagesPanel: $('passages-panel'), passages: $('passages'),
};

let search = null;
let worker = null, modelReady = false, modelInfo = null;
let busy = false;
let pending = null;  // the question the name check declined, for "Answer anyway"

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function setStatus(el, text, cls) {
    el.textContent = text;
    el.className = `status${cls ? ` ${cls}` : ''}`;
}

const mb = (n) => `${(n / 1024 ** 2).toFixed(0)} MB`;

function updateControls() {
    const ready = modelReady && search && !busy;
    els.input.disabled = !ready;
    els.askBtn.disabled = !ready;
    els.input.placeholder = modelReady ? 'ask about something Wikipedia covers…' : 'load the model, then ask…';
}

async function mountPack() {
    try {
        search = await openPikeletFile(httpRangeSource(PACK_URL), {
            sketchScanner: async (sketch) => {
                const { default: Pikelet } = await import('pikelet-wasm/web');
                return Pikelet.createSketchScanner(sketch, { maxRerank: 4096 });
            },
        });
        const info = search.info();
        setStatus(els.packStatus, `wikipedia.pikelet mounted: ${info.records.toLocaleString()} records, hash verified: ${info.residentVerified}`, 'ok');
    } catch (err) {
        setStatus(els.packStatus, `Could not mount the pack: ${err?.message || err}`, 'error');
    }
    updateControls();
}

// Resolves with the worker's next message of one of the given types.
function nextMessage(types, onOther) {
    return new Promise((resolve, reject) => {
        worker.onmessage = ({ data }) => {
            if (types.includes(data.type)) resolve(data);
            else if (data.type === 'error') reject(new Error(data.message));
            else onOther?.(data);
        };
        worker.onerror = (event) => reject(new Error(event.message || 'the model worker failed to start'));
    });
}

async function loadModel() {
    const name = document.querySelector('input[name="model"]:checked').value;
    if (worker) worker.terminate();
    modelReady = false;
    busy = true;
    els.loadBtn.disabled = true;
    updateControls();
    els.progress.hidden = false;
    els.progress.value = 0;
    setStatus(els.modelStatus, 'Starting…', 'busy');
    // Built from the page URL, not a static import, so Vite leaves the worker
    // and the kernel files beside it unbundled.
    worker = new Worker(new URL('llm/worker.mjs', document.baseURI), { type: 'module' });
    const threads = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 2));
    worker.postMessage({ type: 'load', name, base: MODEL_BASE, threads });
    try {
        const ready = await nextMessage(['ready'], (msg) => {
            if (msg.type !== 'progress') return;
            els.progress.value = msg.loaded / msg.total;
            const verb = msg.phase === 'cache' ? 'Reading the saved copy' : 'Downloading';
            setStatus(els.modelStatus, `${verb}: ${mb(msg.loaded)} of ${mb(msg.total)}`, 'busy');
        });
        modelReady = true;
        modelInfo = { name, ...ready };
        const how = ready.isolated ? `${ready.threads} threads` : 'one thread (this page is not cross-origin isolated)';
        const from = ready.source === 'cache' ? 'from this browser\'s saved copy'
            : ready.saveError ? `downloaded (not saved: ${ready.saveError})` : 'downloaded and saved in this browser';
        setStatus(els.modelStatus, `${name} ready, ${from}, in ${(ready.loadMs / 1000).toFixed(1)} s · ${how}`, 'ok');
    } catch (err) {
        worker.terminate();
        worker = null;
        setStatus(els.modelStatus, `Could not load the model: ${err.message}`, 'error');
    } finally {
        els.progress.hidden = true;
        els.loadBtn.disabled = false;
        busy = false;
        updateControls();
    }
}

// rag_score.mjs's "names" gate: a capitalized word after the first, or a word
// with a digit, that none of the passages contains.
const toks = (s) => s.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
function missingNames(question, texts) {
    const ctx = new Set(toks(texts.join(' ')).map((t) => t.toLowerCase()));
    const names = toks(question).filter((t, i) => /\p{N}/u.test(t) || (i > 0 && /^\p{Lu}/u.test(t)));
    return [...new Set(names.filter((t) => !ctx.has(t.toLowerCase())))];
}

function showPassages(results) {
    els.passages.innerHTML = results.map((r, i) => `
        <article class="hit">
          <h3>[${i + 1}] ${escapeHtml(r.title || `record ${r.id}`)}</h3>
          <p class="meta">${escapeHtml(r.sourcePath || r.url || '')} · distance ${r.distance?.toFixed(3)}</p>
          <p class="excerpt full">${escapeHtml(r.text || r.preview || '')}</p>
        </article>`).join('');
    els.passagesPanel.hidden = results.length === 0;
}

function generate(question, texts, searchMs) {
    els.answer.textContent = '';
    els.answer.className = 'answer-box';
    els.anyway.hidden = true;
    els.stats.textContent = 'Reading the passages…';
    let first = true;
    worker.postMessage({ type: 'answer', system: SYSTEM, passages: texts, question, maxNew: 128 });
    return nextMessage(['done'], (msg) => {
        if (msg.type !== 'token') return;
        if (first) { els.stats.textContent = 'Writing…'; first = false; }
        els.answer.textContent += msg.text;
    }).then((d) => {
        if (!els.answer.textContent.trim()) els.answer.textContent = '(no answer)';
        const dropped = d.passages < texts.length ? ` (${texts.length - d.passages} passage dropped to fit)` : '';
        els.stats.textContent = `search ${searchMs.toFixed(0)} ms · read ${d.readTokens} tokens${dropped} in ${(d.readMs / 1000).toFixed(1)} s · `
            + `wrote ${d.wrote} at ${(d.wrote / (d.genMs / 1000) || 0).toFixed(1)} tokens/s · ${modelInfo.name}, `
            + `${modelInfo.isolated ? `${modelInfo.threads} threads` : '1 thread'}`;
    });
}

async function ask(question, { skipCheck = false } = {}) {
    if (!question || busy || !modelReady || !search) return;
    busy = true;
    updateControls();
    els.answerPanel.hidden = false;
    try {
        let texts, searchMs;
        if (skipCheck && pending?.question === question) {
            ({ texts, searchMs } = pending);
        } else {
            els.answer.textContent = '';
            els.answer.className = 'answer-box';
            els.anyway.hidden = true;
            els.stats.textContent = 'Searching the pack…';
            const t0 = performance.now();
            const out = await search.query(question, { k: 5, showAbstained: true });
            searchMs = performance.now() - t0;
            const top = out.results.slice(0, PASSAGES);
            showPassages(top);
            texts = top.map((r) => r.text || r.preview || '');
        }
        pending = null;
        const missing = skipCheck ? [] : missingNames(question, texts);
        if (missing.length) {
            pending = { question, texts, searchMs };
            els.answer.className = 'answer-box declined';
            els.answer.textContent = `None of the passages found mentions ${missing.map((w) => `"${w}"`).join(', ')}, `
                + 'so they can\'t answer this. The search may have missed it, or Wikipedia may not cover it.';
            els.anyway.hidden = false;
            els.stats.textContent = `search ${searchMs.toFixed(0)} ms · model not run`;
            return;
        }
        await generate(question, texts, searchMs);
    } catch (err) {
        els.answer.className = 'answer-box declined';
        els.answer.textContent = String(err?.message || err);
        els.stats.textContent = '';
    } finally {
        busy = false;
        updateControls();
    }
}

els.loadBtn.addEventListener('click', loadModel);
els.form.addEventListener('submit', (event) => {
    event.preventDefault();
    ask(els.input.value.trim());
});
els.chips.addEventListener('click', (event) => {
    const chip = event.target.closest('.chip');
    if (!chip || els.input.disabled) return;
    els.input.value = chip.dataset.q;
    ask(chip.dataset.q);
});
els.anyway.addEventListener('click', () => ask(pending?.question, { skipCheck: true }));

mountPack();
