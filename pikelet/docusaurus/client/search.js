import './search.css';
import { computeMatchQuality, embedTextWithStudent, loadStudentModel } from '../../src/student-embedder.mjs';

const DEFAULT_K = 8;
const DEFAULT_EF_SEARCH = 120;
const DEFAULT_ASSET_BASE = '/pikelet-search';

function normalizeBase(value) {
  return String(value || DEFAULT_ASSET_BASE).replace(/\/+$/g, '');
}

function createRangeSource(url) {
  let fullBuffer = null;
  return {
    async read(offset, length) {
      if (fullBuffer) return fullBuffer.subarray(offset, offset + length);
      const response = await fetch(`${url}?r=${offset}-${offset + length - 1}`, {
        headers: { Range: `bytes=${offset}-${offset + length - 1}` },
      });
      if (response.status !== 206 && response.status !== 200) {
        throw new Error(`Range read failed: ${response.status} ${response.statusText}`.trim());
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (response.status === 200 && bytes.byteLength > length) {
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

function resultUrl(chunk) {
  if (!chunk) return '#';
  const url = chunk.url || chunk.sourcePath || '#';
  if (!chunk.anchor) return url;
  return `${url.replace(/#.*$/, '')}#${chunk.anchor}`;
}

function htmlEscape(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderRows(results, corpusById, { debug = false } = {}) {
  if (!results.length) return '<p class="pikelet-search-empty">No results</p>';
  return `<ol class="pikelet-search-results">${results.map((hit) => {
    const chunk = corpusById?.get(hit.id) || hit;
    const crumb = Array.isArray(chunk?.headingPath) && chunk.headingPath.length
      ? `<span class="pikelet-search-crumb">${htmlEscape(chunk.headingPath.join(' › '))}</span>`
      : '';
    return `<li>
      <a href="${htmlEscape(resultUrl(chunk))}">${htmlEscape(chunk?.title || `Chunk ${hit.id}`)}</a>
      ${crumb}
      <p>${htmlEscape(chunk?.preview || chunk?.text || '')}</p>
      ${debug ? `<span>${hit.distance.toFixed(4)}</span>` : ''}
    </li>`;
  }).join('')}</ol>`;
}

async function loadBrowserRuntime() {
  if (typeof window === 'undefined') throw new Error('Pikelet search can only load in a browser');
  // The widget only needs the range-artifact reader, which is pure JS. The
  // full pikelet-wasm/web entrypoint carries Vite-specific `?url` WASM asset
  // imports that Docusaurus's webpack build cannot process.
  const artifactModule = await import('pikelet-wasm/artifact');
  const contract = artifactModule.default || artifactModule;
  return {
    RangeArtifact: contract.PikeletRangeArtifact,
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampPanel(root) {
  const rect = root.getBoundingClientRect();
  const margin = 12;
  const left = clamp(rect.left, margin, Math.max(margin, window.innerWidth - rect.width - margin));
  const top = clamp(rect.top, margin, Math.max(margin, window.innerHeight - rect.height - margin));
  root.style.left = `${left}px`;
  root.style.top = `${top}px`;
  root.style.right = 'auto';
  root.style.bottom = 'auto';
}

function enableDrag(root, handle) {
  let drag = null;
  handle.addEventListener('pointerdown', (event) => {
    if (event.button !== undefined && event.button !== 0) return;
    const rect = root.getBoundingClientRect();
    drag = {
      pointerId: event.pointerId,
      dx: event.clientX - rect.left,
      dy: event.clientY - rect.top,
    };
    root.classList.add('pikelet-search-dragging');
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener('pointermove', (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const rect = root.getBoundingClientRect();
    const margin = 12;
    const left = clamp(event.clientX - drag.dx, margin, Math.max(margin, window.innerWidth - rect.width - margin));
    const top = clamp(event.clientY - drag.dy, margin, Math.max(margin, window.innerHeight - rect.height - margin));
    root.style.left = `${left}px`;
    root.style.top = `${top}px`;
    root.style.right = 'auto';
    root.style.bottom = 'auto';
  });
  function endDrag(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    root.classList.remove('pikelet-search-dragging');
    drag = null;
  }
  handle.addEventListener('pointerup', endDrag);
  handle.addEventListener('pointercancel', endDrag);
  window.addEventListener('resize', () => clampPanel(root));
}

class PikeletDocusaurusSearch {
  constructor(options = {}) {
    this.assetBase = normalizeBase(options.assetBase);
    this.k = Number(options.k || DEFAULT_K);
    this.efSearch = Number(options.efSearch || DEFAULT_EF_SEARCH);
    this.state = null;
    this.ready = null;
  }

  async load() {
    if (this.ready) return this.ready;
    this.ready = this._load();
    return this.ready;
  }

  async _load() {
    const manifestUrl = `${this.assetBase}/manifest.json`;
    const manifest = await fetch(manifestUrl).then((r) => {
      if (!r.ok) throw new Error(`Manifest fetch failed: ${r.status}`);
      return r.json();
    });
    const urls = manifest.docusaurus || {
      artifactUrl: `${this.assetBase}/index.pikelet-range`,
      corpusUrl: `${this.assetBase}/corpus.json`,
      studentModelUrl: `${this.assetBase}/student-model.bin`,
      abstentionUrl: `${this.assetBase}/student-abstention.json`,
    };
    if (urls.completeArtifactUrl) {
      // Range-read execution: the reader opens on the manifest, resident
      // sketch, corpus tables, and lexical index — a fraction of the file —
      // and prefetches the kind-3 encoder in the background from the moment
      // the panel opens. Hosts that ignore Range degrade to a bounded
      // download-once inside httpRangeSource itself.
      const { openCompletePikeletUrl } = await import('./complete-reader.mjs');
      const completeSearch = await openCompletePikeletUrl(urls.completeArtifactUrl);
      this.state = {
        manifest,
        completeSearch,
        completeInfo: completeSearch.info(),
        corpusById: null,
      };
      return this.state;
    }
    const { RangeArtifact } = await loadBrowserRuntime();
    const [corpus, studentBytes, abstention, artifact] = await Promise.all([
      fetch(urls.corpusUrl).then((r) => {
        if (!r.ok) throw new Error(`Corpus fetch failed: ${r.status}`);
        return r.json();
      }),
      fetch(urls.studentModelUrl).then((r) => {
        if (!r.ok) throw new Error(`Student model fetch failed: ${r.status}`);
        return r.arrayBuffer();
      }),
      urls.abstentionUrl
        ? fetch(urls.abstentionUrl).then((r) => {
            if (r.status === 404) return null;
            if (!r.ok) throw new Error(`Abstention fetch failed: ${r.status}`);
            return r.json();
          })
        : null,
      RangeArtifact.open(createRangeSource(urls.artifactUrl)),
    ]);
    const studentModel = loadStudentModel(studentBytes);
    this.state = {
      manifest,
      artifact,
      studentModel,
      abstention,
      corpusById: new Map(corpus.map((chunk) => [chunk.id, chunk])),
    };
    return this.state;
  }

  async search(query, options = {}) {
    const state = await this.load();
    if (state.completeSearch) {
      const result = await state.completeSearch.query(query, { k: Number(options.k || this.k) });
      return {
        ...result,
        match_quality: result.matchQuality,
        rows: result.results,
      };
    }
    const prefixed = `${state.manifest.prefixPolicy?.query || ''}${query}`;
    const embedded = embedTextWithStudent(prefixed, state.studentModel);
    // The student featurizer only recognizes Latin-script tokens; a query
    // with no recognized terms embeds to the zero vector, which the cosine
    // artifact rejects. Report a graceful no-match instead.
    if (!(embedded.preNorm > 0)) {
      return { results: [], rows: [], rerank: 0, match_quality: 'none' };
    }
    const embedding = embedded.vector;
    const result = await state.artifact.search(embedding, Number(options.k || this.k), {
      efSearch: Number(options.efSearch || state.manifest.efSearch || this.efSearch),
    });
    const quality = computeMatchQuality(result.results, embedded, state.abstention);
    const results = quality.match_quality === 'none' ? [] : result.results;
    return {
      ...result,
      ...quality,
      results,
      rows: results.map((hit) => state.corpusById.get(hit.id)),
    };
  }
}

function mountSearch() {
  const root = document.querySelector('[data-pikelet-search]');
  if (!root || root.dataset.pikeletMounted === '1') return;
  root.dataset.pikeletMounted = '1';
  const assetBase = root.dataset.pikeletAssetBase || window.__PIKELET_SEARCH__?.assetBase;
  const search = new PikeletDocusaurusSearch({ assetBase });
  root.innerHTML = `
    <button class="pikelet-search-launcher" type="button" aria-label="Open search">Search</button>
    <section class="pikelet-search-card" aria-label="Search documentation" hidden>
      <div class="pikelet-search-titlebar">
        <button class="pikelet-search-drag-handle" type="button" aria-label="Move search panel">
          <span>Search docs</span>
        </button>
        <button class="pikelet-search-close" type="button" aria-label="Close search">×</button>
      </div>
      <form class="pikelet-search-form">
        <input class="pikelet-search-input" type="search" autocomplete="off" placeholder="Search docs" />
        <button class="pikelet-search-button" type="submit">Search</button>
      </form>
      <div class="pikelet-search-status">Loading search...</div>
      <div class="pikelet-search-output"></div>
    </section>
  `;
  const launcher = root.querySelector('.pikelet-search-launcher');
  const card = root.querySelector('.pikelet-search-card');
  const close = root.querySelector('.pikelet-search-close');
  const handle = root.querySelector('.pikelet-search-drag-handle');
  const form = root.querySelector('form');
  const input = root.querySelector('input');
  const button = root.querySelector('.pikelet-search-button');
  const status = root.querySelector('.pikelet-search-status');
  const output = root.querySelector('.pikelet-search-output');

  enableDrag(root, handle);
  // Nothing loads until the panel first opens: page load costs zero search
  // bytes, and opening the panel starts the small range-read open (plus,
  // for kind-3 artifacts, the background encoder prefetch) while the user
  // types their first query.
  launcher.addEventListener('click', () => {
    card.hidden = false;
    launcher.hidden = true;
    clampPanel(root);
    input.focus();
    search.load()
      .then((state) => {
        const count = state.completeInfo?.records ?? state.artifact.count;
        status.textContent = `Ready - ${count.toLocaleString()} chunks`;
      })
      .catch((error) => { status.textContent = error.message || String(error); });
  });
  close.addEventListener('click', () => {
    card.hidden = true;
    launcher.hidden = false;
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const query = input.value.trim();
    if (!query) return;
    button.disabled = true;
    status.textContent = 'Searching...';
    try {
      const t0 = performance.now();
      const result = await search.search(query);
      const quality = result.match_quality && result.match_quality !== 'unscored'
        ? ` - ${result.match_quality}`
        : '';
      status.textContent = `${result.results.length} results${quality} in ${(performance.now() - t0).toFixed(0)} ms`;
      output.innerHTML = renderRows(result.results, search.state.corpusById, { debug: root.dataset.pikeletDebug === '1' });
    } catch (error) {
      status.textContent = error.message || String(error);
    } finally {
      button.disabled = false;
    }
  });
}

if (typeof window !== 'undefined') {
  window.PikeletDocusaurusSearch = PikeletDocusaurusSearch;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountSearch);
  else mountSearch();
}

export { PikeletDocusaurusSearch, mountSearch };
