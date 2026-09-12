import Pikelet, { PikeletError, PIKELET_ERROR_CODES } from 'pikelet-wasm';
import * as encoder from './encoder.js';
import SNAPSHOT_ASSET from './assets/snapshot.pnck';
import CORPUS_ASSET from './assets/corpus.json';
import MANIFEST_ASSET from './assets/manifest.json';
import UI_HTML from './ui.html';

const DEFAULT_MAX_JSON_BYTES = 256 * 1024;
const MAX_RESULTS = 8;
const MAX_EF_SEARCH = 400;
const RATE_LIMIT_WINDOW_MS = 60_000;
const CLIENT_ERROR_CODES = new Set([
  PIKELET_ERROR_CODES.INVALID_ARGUMENT,
  PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
  PIKELET_ERROR_CODES.INVALID_VECTOR,
  PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
  PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
  PIKELET_ERROR_CODES.SNAPSHOT_CAPACITY_EXCEEDED,
]);

let index = null;
let corpus = [];
let corpusById = new Map();
let manifest = null;
let loadPromise = null;
const rateLimitMap = new Map();
const state = {
  restoreCount: 0,
  restoredAt: null,
  lastRestoreMs: null,
};

function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...extraHeaders,
    },
  });
}

function htmlResponse(body, env) {
  return new Response(body, {
    headers: {
      'content-type': 'text/html; charset=utf-8',
      ...corsHeaders(env),
    },
  });
}

function corsHeaders(env) {
  const origin = String(env?.ALLOWED_ORIGIN || '').trim();
  return origin ? { 'access-control-allow-origin': origin } : {};
}

function withCors(response, env) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders(env))) headers.set(key, value);
  return new Response(response.body, { status: response.status, headers });
}

function isReadOnly(env) {
  const value = String(env?.READ_ONLY || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

function allowInsecureAdmin(env) {
  const value = String(env?.ALLOW_INSECURE_ADMIN || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

function requireAdminAuth(request, env) {
  if (!env?.API_KEY && !allowInsecureAdmin(env)) {
    return jsonResponse({ error: 'Admin routes require API_KEY or explicit ALLOW_INSECURE_ADMIN=1' }, 403);
  }
  if (!env?.API_KEY) return null;
  const expected = `Bearer ${env.API_KEY}`;
  return timingSafeEqual(request.headers.get('authorization') || '', expected)
    ? null
    : jsonResponse({ error: 'Unauthorized' }, 401);
}

function timingSafeEqual(actual, expected) {
  const encoder = new TextEncoder();
  const a = encoder.encode(actual);
  const b = encoder.encode(expected);
  const length = Math.max(a.length, b.length);
  let mismatch = a.length ^ b.length;
  for (let i = 0; i < length; i++) mismatch |= (a[i] || 0) ^ (b[i] || 0);
  return mismatch === 0;
}

function positiveEnvInt(env, name, fallback) {
  const raw = env?.[name];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function isRateLimited(request, env) {
  const maxRpm = positiveEnvInt(env, 'RATE_LIMIT_RPM', 0);
  if (!maxRpm) return false;
  const ip = request.headers.get('cf-connecting-ip') || 'local';
  const now = Date.now();
  for (const [key, entries] of rateLimitMap) {
    while (entries.length && now - entries[0] >= RATE_LIMIT_WINDOW_MS) entries.shift();
    if (!entries.length) rateLimitMap.delete(key);
  }
  const entries = rateLimitMap.get(ip) || [];
  rateLimitMap.set(ip, entries);
  while (entries.length && now - entries[0] >= RATE_LIMIT_WINDOW_MS) entries.shift();
  if (entries.length >= maxRpm) return true;
  entries.push(now);
  return false;
}

async function readJson(request, env) {
  const maxBytes = positiveEnvInt(env, 'MAX_JSON_BYTES', DEFAULT_MAX_JSON_BYTES);
  const reader = request.body?.getReader();
  if (!reader) return {};
  const chunks = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      try { await reader.cancel(); } catch {}
      return { error: `JSON body exceeds MAX_JSON_BYTES (${total} > ${maxBytes})`, status: 413 };
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return { body: bytes.byteLength ? JSON.parse(new TextDecoder().decode(bytes)) : {} };
  } catch {
    return { error: 'Invalid JSON', status: 400 };
  }
}

function assetBytes(asset, label) {
  if (asset instanceof ArrayBuffer) return new Uint8Array(asset);
  if (ArrayBuffer.isView(asset)) return new Uint8Array(asset.buffer, asset.byteOffset, asset.byteLength);
  throw new Error(`${label} was not bundled as binary data`);
}

function assertManifestMatches(loaded) {
  encoder.assertEncoderManifest(loaded);
  if (loaded.dims !== 384 && loaded.dim !== 384) {
    throw Object.assign(new Error('Manifest dimension mismatch'), { code: 'MANIFEST_MISMATCH' });
  }
  if (!loaded.prefixPolicy || typeof loaded.prefixPolicy.query !== 'string') {
    throw Object.assign(new Error('Manifest prefix policy missing'), { code: 'MANIFEST_MISMATCH' });
  }
  if ((loaded.pooling || 'mean') !== 'mean') {
    throw Object.assign(new Error('Manifest pooling mismatch'), { code: 'MANIFEST_MISMATCH' });
  }
}

async function restoreAssets() {
  const t0 = performance.now();
  const loadedManifest = MANIFEST_ASSET;
  assertManifestMatches(loadedManifest);
  const snapshotBytes = assetBytes(SNAPSHOT_ASSET, 'Snapshot');
  const restored = await Pikelet.restore(snapshotBytes, {
    maxElements: loadedManifest.maxElements,
    efSearch: loadedManifest.efSearch,
  });
  corpus = CORPUS_ASSET;
  corpusById = new Map(corpus.map((entry) => [entry.id, entry]));
  manifest = loadedManifest;
  index = restored;
  state.restoreCount += 1;
  state.restoredAt = new Date().toISOString();
  state.lastRestoreMs = performance.now() - t0;
}

async function ensureLoaded() {
  if (index && manifest) return { cold: false, restoreMs: 0 };
  const cold = !loadPromise;
  if (!loadPromise) {
    loadPromise = restoreAssets().finally(() => {
      loadPromise = null;
    });
  }
  await loadPromise;
  return { cold, restoreMs: cold ? state.lastRestoreMs : 0 };
}

function parseSearchParams(request) {
  const url = new URL(request.url);
  return {
    query: url.searchParams.get('q') || '',
    k: Number.parseInt(url.searchParams.get('k') || '5', 10),
    efSearch: Number.parseInt(url.searchParams.get('efSearch') || url.searchParams.get('ef') || '120', 10),
  };
}

function buildResult(hit) {
  const chunk = corpusById.get(hit.id);
  return {
    id: hit.id,
    distance: hit.distance,
    title: chunk?.title || `Chunk ${hit.id}`,
    preview: chunk?.preview || chunk?.text?.slice(0, 220) || '',
    source_path: chunk?.sourcePath || '',
    anchor: chunk?.anchor || '',
    url: chunk?.url || '',
  };
}

async function handleSearch(request, env) {
  let params = parseSearchParams(request);
  if (request.method === 'POST') {
    const parsed = await readJson(request, env);
    if (parsed.error) return jsonResponse({ error: parsed.error }, parsed.status);
    params = {
      ...params,
      query: parsed.body.query ?? parsed.body.q ?? params.query,
      k: parsed.body.k ?? params.k,
      efSearch: parsed.body.efSearch ?? parsed.body.ef ?? params.efSearch,
    };
  }

  const query = String(params.query || '').trim();
  if (!query) return jsonResponse({ error: 'query is required' }, 400);
  const maxQueryChars = positiveEnvInt(env, 'MAX_QUERY_CHARS', manifest?.maxQueryChars || 4096);
  if (query.length > maxQueryChars) {
    return jsonResponse({ error: `query exceeds MAX_QUERY_CHARS (${query.length} > ${maxQueryChars})` }, 400);
  }
  let k = Number.isInteger(params.k) && params.k > 0 ? params.k : 5;
  let efSearch = Number.isInteger(params.efSearch) && params.efSearch > 0 ? params.efSearch : manifest?.efSearch || 120;
  k = Math.min(k, MAX_RESULTS);
  efSearch = Math.min(efSearch, MAX_EF_SEARCH);

  const load = await ensureLoaded();
  const embedStart = performance.now();
  const { vector: queryVector, embedded } = await encoder.embedQuery(query, manifest, env);
  const embeddingMs = performance.now() - embedStart;
  const searchStart = performance.now();
  const rawHits = index.search(queryVector, k, { efSearch });
  const searchMs = performance.now() - searchStart;
  const quality = encoder.scoreHits(rawHits, embedded);
  const hits = quality?.match_quality === 'none' ? [] : rawHits;
  return jsonResponse({
    query,
    result_count: hits.length,
    ...(quality ? { match_quality: quality.match_quality, match_confidence: quality.confidence ?? null } : {}),
    cache_state: load.cold ? 'cold-restored' : 'warm-cache',
    restore_ms: load.restoreMs,
    embedding_ms: embeddingMs,
    search_ms: searchMs,
    corpus_chunks: corpus.length,
    dim: manifest.dims,
    ef_search: efSearch,
    restore_count: state.restoreCount,
    results: hits.map(buildResult),
  });
}

function healthBody(env) {
  return {
    ok: true,
    loaded: !!index,
    corpus_chunks: corpus.length,
    dim: manifest?.dims || null,
    read_only: isReadOnly(env),
    restore_count: state.restoreCount,
    restored_at: state.restoredAt,
    last_restore_ms: state.lastRestoreMs,
    runtime_mode: 'snapshot',
    ...encoder.encoderInfo(env, manifest),
  };
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      if (request.method === 'OPTIONS') {
        return new Response(null, {
          status: corsHeaders(env)['access-control-allow-origin'] ? 204 : 403,
          headers: {
            ...corsHeaders(env),
            'access-control-allow-methods': 'GET, POST, OPTIONS',
            'access-control-allow-headers': 'content-type, authorization',
          },
        });
      }
      if (url.pathname === '/') return htmlResponse(UI_HTML, env);
      if (url.pathname !== '/health' && isRateLimited(request, env)) {
        return withCors(jsonResponse({ error: 'Rate limit exceeded' }, 429), env);
      }
      if (url.pathname === '/health') return withCors(jsonResponse(healthBody(env)), env);
      if (url.pathname === '/readiness') {
        const auth = requireAdminAuth(request, env);
        if (auth) return withCors(auth, env);
        return withCors(jsonResponse({ ready: true, loaded: !!index, manifest: MANIFEST_ASSET, read_only: isReadOnly(env) }), env);
      }
      if (url.pathname === '/reset_cache' && request.method === 'POST') {
        const auth = requireAdminAuth(request, env);
        if (auth) return withCors(auth, env);
        if (isReadOnly(env)) return withCors(jsonResponse({ error: 'Read-only mode' }, 403), env);
        if (index) index.dispose();
        index = null;
        manifest = null;
        corpus = [];
        corpusById = new Map();
        return withCors(jsonResponse({ cleared: true }), env);
      }
      if (url.pathname === '/search' && (request.method === 'GET' || request.method === 'POST')) {
        return withCors(await handleSearch(request, env), env);
      }
      return withCors(jsonResponse({ error: 'Not found' }, 404), env);
    } catch (error) {
      const status = error?.code === 'EMBED_UNAVAILABLE'
        ? 503
        : error?.code === 'MANIFEST_MISMATCH'
          ? 500
          : error instanceof PikeletError && CLIENT_ERROR_CODES.has(error.code)
            ? 400
            : 500;
      return withCors(jsonResponse({ error: error.message || String(error), code: error.code }, status), env);
    }
  },
};
