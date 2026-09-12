// Minimal two-panel ablation demo: same question against a full pack and
// a pack missing one record, to show the answer is a function of the file.
import { openPikeletFile } from '../pikelet-file-reader.mjs';
import { httpRangeSource } from '../sources.mjs';

const QUESTION = 'What chamber is the Tovash project housed in?';

async function runPanel(url, badgeEl, answerEl, recordEl) {
    const source = httpRangeSource(url);
    await source.init();
    const search = await openPikeletFile(source);
    const out = await search.query(QUESTION, { k: 1 });

    badgeEl.innerHTML = `<span class="badge ${out.matchQuality}">${out.matchQuality}</span> confidence ${out.confidence.toFixed(3)}`;

    if (out.results.length === 0) {
        answerEl.textContent = 'Cannot answer from this pack.';
        recordEl.innerHTML = '';
        return;
    }

    const top = out.results[0];
    answerEl.textContent = top.preview || top.text;
    recordEl.innerHTML = `<div class="rid">record ${top.id} — ${escapeHtml(top.title)} (${escapeHtml(top.sourcePath)})</div>`;
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

await Promise.all([
    runPanel('/veyra.pikelet',
        document.getElementById('left-badge'),
        document.getElementById('left-answer'),
        document.getElementById('left-record')),
    runPanel('/veyra-ablated.pikelet',
        document.getElementById('right-badge'),
        document.getElementById('right-answer'),
        document.getElementById('right-record')),
]);
