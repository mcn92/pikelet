import Pikelet from 'pikelet-wasm/web';
import { loadStudentModel, embedTextWithStudent } from '../../../../03-edge-docs-search/student-embedder.mjs';
import { createAbstentionScorer } from './abstention.js';
import './style.css';

const ARTIFACT_URL = '/artifacts/pikelet-docs.pikelet-range';
const MODEL_URL = '/models/docs-student.bin';
const CORPUS_URL = '/corpus/docs-corpus.json';
const ABSTENTION_URL = '/abstention/docs-abstention.json';

const SAMPLE_QUERIES = [
  'how does compaction work',
  'How do Cloudflare Workers restore snapshots from R2?',
  'What are the memory tradeoffs for quantized indexes?',
  'banana pikelet recipe',
];

const els = {
  artifactLabel: document.querySelector('#artifactLabel'),
  queryInput: document.querySelector('#queryInput'),
  sampleQueries: document.querySelector('#sampleQueries'),
  kInput: document.querySelector('#kInput'),
  efInput: document.querySelector('#efInput'),
  gapInput: document.querySelector('#gapInput'),
  searchButton: document.querySelector('#searchButton'),
  wallMetric: document.querySelector('#wallMetric'),
  embedMetric: document.querySelector('#embedMetric'),
  requestsMetric: document.querySelector('#requestsMetric'),
  bytesMetric: document.querySelector('#bytesMetric'),
  missMetric: document.querySelector('#missMetric'),
  cacheMetric: document.querySelector('#cacheMetric'),
  resultsList: document.querySelector('#resultsList'),
  roundsList: document.querySelector('#roundsList'),
  matchBanner: document.querySelector('#matchBanner'),
};

let artifact;
let model;
let corpusById;
let abstention;

function createHttpRangeSource(url) {
  let fullBuffer = null;
  return {
    get fallbackActive() {
      return fullBuffer !== null;
    },
    async read(offset, length) {
      if (fullBuffer) return fullBuffer.subarray(offset, offset + length);
      const response = await fetch(url, {
        headers: { Range: `bytes=${offset}-${offset + length - 1}` },
      });
      if (response.status !== 206 && response.status !== 200) {
        throw new Error(`Range read failed: ${response.status} ${response.statusText}`.trim());
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (response.status === 200 && bytes.byteLength > length) {
        // Host ignored the Range header and returned the whole file (for
        // example Cloudflare Pages). Keep the buffer and serve every future
        // read as a local slice: one full download instead of a hard failure.
        fullBuffer = bytes;
        return fullBuffer.subarray(offset, offset + length);
      }
      if (bytes.byteLength !== length) {
        throw new Error(`Range read returned ${bytes.byteLength} bytes; expected ${length}`);
      }
      return bytes;
    },
  };
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1048576).toFixed(2)} MiB`;
}

function formatMs(ms) {
  return `${ms.toFixed(ms < 10 ? 2 : 1)} ms`;
}

function setBusy(busy) {
  els.searchButton.disabled = busy;
  els.searchButton.textContent = busy ? 'Searching' : 'Search';
}

function renderResults(rows) {
  els.resultsList.replaceChildren(...rows.map((row) => {
    const chunk = corpusById.get(row.id);
    const li = document.createElement('li');
    li.className = 'result';

    const header = document.createElement('div');
    header.className = 'resultHeader';
    const title = document.createElement('strong');
    title.textContent = chunk?.title || `Chunk ${row.id}`;
    const dist = document.createElement('span');
    dist.textContent = row.distance.toFixed(4);
    header.append(title, dist);

    const source = document.createElement('div');
    source.className = 'resultSource';
    source.textContent = chunk?.anchor
      ? `${chunk.sourcePath} #${chunk.anchor}`
      : chunk?.sourcePath || '';

    const preview = document.createElement('p');
    preview.className = 'resultPreview';
    const text = chunk?.preview || chunk?.text || '';
    preview.textContent = text.length > 240 ? `${text.slice(0, 240)}…` : text;

    li.append(header, source, preview);
    return li;
  }));
}

function renderRounds(rounds) {
  const visible = rounds.filter((round) => round.requests > 0);
  if (!visible.length) {
    const div = document.createElement('div');
    div.className = 'round';
    const label = document.createElement('span');
    label.textContent = 'All records served from the local cache';
    div.append(label);
    els.roundsList.replaceChildren(div);
    return;
  }
  els.roundsList.replaceChildren(...visible.map((round, index) => {
    const div = document.createElement('div');
    div.className = 'round';
    const label = document.createElement('span');
    label.textContent = `${index + 1}. ${round.phase || 'round'}`;
    const details = document.createElement('strong');
    details.textContent = `${round.requests} req / ${formatBytes(round.bytes)}`;
    div.append(label, details);
    return div;
  }));
}

function renderMatchBanner(quality) {
  const banner = els.matchBanner;
  banner.className = 'matchBanner';
  if (!quality || quality.match_quality === 'unscored') {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  const label = quality.match_quality;
  banner.classList.add(`match-${label}`);
  const confidence = quality.confidence !== undefined ? ` (confidence ${quality.confidence.toFixed(3)})` : '';
  if (label === 'strong') {
    banner.textContent = `Strong match${confidence}`;
  } else if (label === 'weak') {
    banner.textContent = `No strong match — showing the closest results${confidence}`;
  } else {
    banner.textContent = `No match in this corpus — the artifact abstains rather than guessing${confidence}`;
  }
}

async function runSearch() {
  if (!artifact || !model) return;
  const text = els.queryInput.value.trim();
  if (!text) return;
  setBusy(true);
  try {
    const k = Number(els.kInput.value);
    const efSearch = Number(els.efInput.value);
    const gap = Number(els.gapInput.value);
    const before = artifact.stats();
    const t0 = performance.now();
    const embedded = embedTextWithStudent(text, model);
    const embedMs = performance.now() - t0;
    const result = await artifact.search(embedded.vector, k, {
      efSearch,
      gap,
      expansionBatch: 8,
      rangeParallelism: 6,
    });
    const wallMs = performance.now() - t0;
    const after = artifact.stats();
    const requests = after.rangeRequests - before.rangeRequests;
    const bytes = after.rangeBytes - before.rangeBytes;
    const missRounds = result.rounds.filter((round) => round.requests > 0).length;

    const quality = abstention
      ? (abstention.scorePreSearch(embedded) || abstention.score(result.results, embedded))
      : { match_quality: 'unscored' };

    els.wallMetric.textContent = formatMs(wallMs);
    els.embedMetric.textContent = formatMs(embedMs);
    els.requestsMetric.textContent = requests.toLocaleString();
    els.bytesMetric.textContent = formatBytes(bytes);
    els.missMetric.textContent = missRounds.toLocaleString();
    els.cacheMetric.textContent = after.cachedNodes.toLocaleString();
    renderMatchBanner(quality);
    renderResults(quality.match_quality === 'none' ? [] : result.results);
    renderRounds(result.rounds);
  } catch (error) {
    els.resultsList.replaceChildren();
    els.roundsList.textContent = error && error.message ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

async function boot() {
  SAMPLE_QUERIES.forEach((sample) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = sample;
    chip.addEventListener('click', () => {
      els.queryInput.value = sample;
      runSearch();
    });
    els.sampleQueries.append(chip);
  });

  setBusy(true);
  const [modelBytes, corpus, abstentionAsset] = await Promise.all([
    fetch(MODEL_URL).then((r) => {
      if (!r.ok) throw new Error(`Model fetch failed: ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(CORPUS_URL).then((r) => {
      if (!r.ok) throw new Error(`Corpus fetch failed: ${r.status}`);
      return r.json();
    }),
    fetch(ABSTENTION_URL).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  model = loadStudentModel(new Uint8Array(modelBytes));
  corpusById = new Map(corpus.map((chunk) => [chunk.id, chunk]));
  abstention = createAbstentionScorer(abstentionAsset);

  const source = createHttpRangeSource(ARTIFACT_URL);
  artifact = await Pikelet.RangeArtifact.open(source);
  const labelParts = [
    `${artifact.count.toLocaleString()} doc chunks`,
    `${artifact.dim}D`,
    `router ${artifact.routerResident.records.toLocaleString()} records`,
    `${formatBytes(artifact.routerResident.bytes)} resident`,
    `encoder ${formatBytes(modelBytes.byteLength)} local`,
  ];
  if (source.fallbackActive) {
    labelParts.push('host ignores Range: full artifact fetched once');
  }
  els.artifactLabel.textContent = labelParts.join(' / ');
  setBusy(false);
  els.queryInput.focus();
}

els.searchButton.addEventListener('click', runSearch);
els.queryInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') runSearch();
});

boot().catch((error) => {
  setBusy(false);
  els.artifactLabel.textContent = error && error.message ? error.message : String(error);
});
