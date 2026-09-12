import Pikelet, { PikeletError, PIKELET_ERROR_CODES } from 'pikelet-wasm';
import * as encoder from './encoder.js';
import ARTIFACT_ASSET from './assets/index.pikelet-range';
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

let artifact = null;
let corpus = [];
let corpusById = new Map();
let manifest = null;
let loadPromise = null;
const rateLimitMap = new Map();
const state = {
  loadCount: 0,
  loadedAt: null,
  lastLoadMs: null,
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

function nonNegativeEnvInt(env, name, fallback) {
  const raw = env?.[name];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
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

function createBundledRangeSource(asset, label) {
  const bytes = assetBytes(asset, label);
  return {
    async read(offset, length) {
      if (!Number.isInteger(offset) || !Number.isInteger(length) || offset < 0 || length < 0) {
        throw new Error(`Invalid range request for ${label}`);
      }
      const end = offset + length;
      if (end > bytes.byteLength) {
        throw new Error(`Range request exceeds ${label} size (${end} > ${bytes.byteLength})`);
      }
      return bytes.slice(offset, end);
    },
  };
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

async function loadAssets(env) {
  const t0 = performance.now();
  const loadedManifest = MANIFEST_ASSET;
  assertManifestMatches(loadedManifest);
  const source = createBundledRangeSource(ARTIFACT_ASSET, 'Search artifact');
  const cacheMb = positiveEnvInt(env, 'ARTIFACT_CACHE_MB', 64);
  const loadedArtifact = await Pikelet.RangeArtifact.open(source, { maxCacheBytes: cacheMb * 1024 * 1024 });
  if (loadedArtifact.dim !== loadedManifest.dims) {
    throw Object.assign(new Error(`Search Artifact dimension mismatch (${loadedArtifact.dim} !== ${loadedManifest.dims})`), { code: 'MANIFEST_MISMATCH' });
  }
  if (loadedArtifact.count !== CORPUS_ASSET.length) {
    throw Object.assign(new Error(`Search Artifact corpus count mismatch (${loadedArtifact.count} !== ${CORPUS_ASSET.length})`), { code: 'MANIFEST_MISMATCH' });
  }
  corpus = CORPUS_ASSET;
  corpusById = new Map(corpus.map((entry) => [entry.id, entry]));
  manifest = loadedManifest;
  artifact = loadedArtifact;
  state.loadCount += 1;
  state.loadedAt = new Date().toISOString();
  state.lastLoadMs = performance.now() - t0;
}

async function ensureLoaded(env) {
  if (artifact && manifest) return { cold: false, loadMs: 0 };
  const cold = !loadPromise;
  if (!loadPromise) {
    loadPromise = loadAssets(env).finally(() => {
      loadPromise = null;
    });
  }
  await loadPromise;
  return { cold, loadMs: cold ? state.lastLoadMs : 0 };
}

function parseSearchParams(request) {
  const url = new URL(request.url);
  return {
    query: url.searchParams.get('q') || '',
    k: Number.parseInt(url.searchParams.get('k') || '5', 10),
    efSearch: Number.parseInt(url.searchParams.get('efSearch') || url.searchParams.get('ef') || '120', 10),
    rangeGap: Number.parseInt(url.searchParams.get('rangeGap') || url.searchParams.get('gap') || '', 10),
    expansionBatch: Number.parseInt(url.searchParams.get('expansionBatch') || url.searchParams.get('batch') || '', 10),
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
      rangeGap: parsed.body.rangeGap ?? parsed.body.gap ?? params.rangeGap,
      expansionBatch: parsed.body.expansionBatch ?? parsed.body.batch ?? params.expansionBatch,
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
  const rangeGap = Number.isInteger(params.rangeGap) && params.rangeGap >= 0 ? params.rangeGap : nonNegativeEnvInt(env, 'RANGE_GAP_BYTES', 0);
  const expansionBatch = Number.isInteger(params.expansionBatch) && params.expansionBatch > 0 ? params.expansionBatch : positiveEnvInt(env, 'EXPANSION_BATCH', 1);
  k = Math.min(k, MAX_RESULTS);
  efSearch = Math.min(efSearch, MAX_EF_SEARCH);

  const load = await ensureLoaded(env);
  const embedStart = performance.now();
  const { vector: queryVector, embedded } = await encoder.embedQuery(query, manifest, env);
  const embeddingMs = performance.now() - embedStart;
  const searchStart = performance.now();
  const beforeArtifactStats = artifact.stats();
  const artifactResult = await artifact.search(queryVector, k, { efSearch, gap: rangeGap, expansionBatch });
  const searchMs = performance.now() - searchStart;
  const artifactStats = artifact.stats();
  const quality = encoder.scoreHits(artifactResult.results, embedded);
  const hits = quality?.match_quality === 'none' ? [] : artifactResult.results;
  return jsonResponse({
    query,
    result_count: hits.length,
    ...(quality ? { match_quality: quality.match_quality, match_confidence: quality.confidence ?? null } : {}),
    cache_state: load.cold ? 'cold-loaded-artifact' : 'warm-artifact',
    load_ms: load.loadMs,
    embedding_ms: embeddingMs,
    search_ms: searchMs,
    corpus_chunks: corpus.length,
    dim: manifest.dims,
    ef_search: efSearch,
    range_gap_bytes: rangeGap,
    expansion_batch: expansionBatch,
    load_count: state.loadCount,
    artifact: {
      router_resident_records: artifactStats.routerResident.records,
      router_resident_bytes: artifactStats.routerResident.bytes,
      query_range_requests: artifactStats.rangeRequests - beforeArtifactStats.rangeRequests,
      query_range_bytes: artifactStats.rangeBytes - beforeArtifactStats.rangeBytes,
      query_cached_nodes_added: artifactStats.cachedNodes - beforeArtifactStats.cachedNodes,
      total_range_requests: artifactStats.rangeRequests,
      total_range_bytes: artifactStats.rangeBytes,
      cached_nodes: artifactStats.cachedNodes,
      rounds: artifactResult.rounds.length,
    },
    results: hits.map(buildResult),
  });
}

function healthBody(env) {
  return {
    ok: true,
    loaded: !!artifact,
    corpus_chunks: corpus.length,
    dim: manifest?.dims || null,
    read_only: isReadOnly(env),
    load_count: state.loadCount,
    loaded_at: state.loadedAt,
    last_load_ms: state.lastLoadMs,
    runtime_mode: 'artifact',
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
        return withCors(jsonResponse({ ready: true, loaded: !!artifact, manifest: MANIFEST_ASSET, read_only: isReadOnly(env), runtime_mode: 'artifact' }), env);
      }
      if (url.pathname === '/reset_cache' && request.method === 'POST') {
        const auth = requireAdminAuth(request, env);
        if (auth) return withCors(auth, env);
        if (isReadOnly(env)) return withCors(jsonResponse({ error: 'Read-only mode' }, 403), env);
        await artifact?.close?.();
        artifact = null;
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
