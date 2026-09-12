/**
 * Pikelet Search - Cloudflare Workers Entrypoint
 *
 * This example intentionally uses the public Pikelet API. It should be useful
 * as deployment documentation and as a benchmark of the package surface users
 * actually call, not as a copy of the raw WASM ABI.
 */

import Pikelet, { PikeletError, PIKELET_ERROR_CODES } from '../../../pikelet.workerd.mjs';

let index = null;
let indexConfig = null;
let localSnapshot = null;
let restorePromise = null;
let mutationPromise = Promise.resolve();
let snapshotSequence = 0;
let restoreGeneration = 0;

const MAX_RESULTS = 100;
const MAX_DIMS = 4096;
const DEFAULT_MAX_ELEMENTS = 5_000;
const DEFAULT_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024;
const DEFAULT_MAX_JSON_BYTES = 1 * 1024 * 1024;
const MAX_EF = 4096;
const MAX_M = 128;
const SNAPSHOT_KEY_PREFIX = 'pikelet-index-';
const LEGACY_SNAPSHOT_KEY = 'pikelet-index.bin';
const PUBLIC_ROUTES = new Set(['/health', '/search']);
const ADMIN_ROUTES = new Set([
  '/init',
  '/add',
  '/add_batch',
  '/delete',
  '/compact',
  '/import',
  '/export',
  '/reset_cache',
  '/search_debug'
]);

const rateLimitMap = new Map();
const RATE_LIMIT_WINDOW_MS = 60_000;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
const CLIENT_ERROR_CODES = new Set([
  PIKELET_ERROR_CODES.INVALID_ARGUMENT,
  PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
  PIKELET_ERROR_CODES.INVALID_VECTOR,
  PIKELET_ERROR_CODES.INDEX_FULL,
  PIKELET_ERROR_CODES.COMPACTION_REQUIRED,
  PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
  PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
  PIKELET_ERROR_CODES.SNAPSHOT_CAPACITY_EXCEEDED
]);
const restoreState = {
  restoreCount: 0,
  restoredAt: null,
  lastRestoreMs: null,
  lastFetchMs: null,
  lastDeserializeMs: null,
  lastSnapshotBytes: null,
  lastSnapshotKey: null
};

function isPositiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

class RequestError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.name = 'RequestError';
    this.status = status;
  }
}

function jsonResponse(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...headers
    }
  });
}

function binaryResponse(bytes, headers = {}) {
  return new Response(bytes, {
    headers: {
      'Content-Type': 'application/octet-stream',
      ...headers
    }
  });
}

function corsHeaders(env) {
  const origin = String(env?.ALLOWED_ORIGIN || '').trim();
  const headers = {
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
  };
  // CORS is opt-in: without ALLOWED_ORIGIN the header is omitted entirely.
  if (origin) headers['Access-Control-Allow-Origin'] = origin;
  return headers;
}

function withCors(response, env) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders(env))) {
    headers.set(key, value);
  }
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

function isAdminAllowed(request, env) {
  // Without API_KEY, non-admin routes stay open; admin routes fail closed
  // earlier in handleRequest() unless ALLOW_INSECURE_ADMIN is set.
  if (!env?.API_KEY) return true;
  const expected = `Bearer ${env.API_KEY}`;
  return timingSafeEqual(request.headers.get('Authorization') || '', expected);
}

function timingSafeEqual(actual, expected) {
  const a = textEncoder.encode(actual);
  const b = textEncoder.encode(expected);
  const length = Math.max(a.length, b.length);
  let mismatch = a.length ^ b.length;
  for (let i = 0; i < length; i++) mismatch |= (a[i] || 0) ^ (b[i] || 0);
  return mismatch === 0;
}

function getRateLimit(env) {
  const raw = env?.RATE_LIMIT_RPM;
  if (raw === undefined || raw === null || raw === '') return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function isRateLimited(request, env) {
  const maxRpm = getRateLimit(env);
  if (maxRpm === null) return false;

  const ip = request.headers.get('CF-Connecting-IP') || 'local';
  const now = Date.now();
  for (const [key, entries] of rateLimitMap) {
    while (entries.length > 0 && now - entries[0] >= RATE_LIMIT_WINDOW_MS) entries.shift();
    if (entries.length === 0) rateLimitMap.delete(key);
  }
  let timestamps = rateLimitMap.get(ip);
  if (!timestamps) {
    timestamps = [];
    rateLimitMap.set(ip, timestamps);
  }

  while (timestamps.length > 0 && now - timestamps[0] >= RATE_LIMIT_WINDOW_MS) {
    timestamps.shift();
  }
  if (timestamps.length >= maxRpm) return true;
  timestamps.push(now);
  return false;
}

async function readJson(request, env) {
  const maxBytes = positiveEnvInt(env, 'MAX_JSON_BYTES', DEFAULT_MAX_JSON_BYTES);
  const buffer = await readLimitedBody(request, maxBytes, 'JSON body');
  if (buffer.byteLength === 0) return {};
  const text = textDecoder.decode(buffer);
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new RequestError(`Invalid JSON body: ${error.message}`);
  }
}

async function readBytes(request, env) {
  const maxBytes = positiveEnvInt(env, 'MAX_SNAPSHOT_BYTES', DEFAULT_MAX_SNAPSHOT_BYTES);
  return new Uint8Array(await readLimitedBody(request, maxBytes, 'Snapshot'));
}

async function readLimitedBody(request, maxBytes, label) {
  const contentLength = request.headers.get('Content-Length');
  if (contentLength !== null) {
    const declared = Number.parseInt(contentLength, 10);
    if (Number.isFinite(declared) && declared > maxBytes) {
      throw new RequestError(`${label} exceeds limit (${declared} > ${maxBytes})`, 413);
    }
  }
  if (!request.body) return new Uint8Array(0);

  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      try { await reader.cancel(); } catch {}
      throw new RequestError(`${label} exceeds limit (${total} > ${maxBytes})`, 413);
    }
    chunks.push(value);
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function positiveEnvInt(env, name, fallback) {
  const raw = env?.[name];
  if (raw === undefined || raw === null || raw === '') return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function validateCreateConfig(config, label, maxElementsLimit) {
  if (!isPositiveInteger(config.dim) || config.dim > MAX_DIMS) {
    throw new RequestError(`${label}: dims must be an integer between 1 and ${MAX_DIMS}`);
  }
  if (!isPositiveInteger(config.maxElements) || config.maxElements > maxElementsLimit) {
    throw new RequestError(`${label}: maxElements must be an integer between 1 and ${maxElementsLimit}`);
  }
  if (!Number.isInteger(config.M) || config.M < 2 || config.M > MAX_M) {
    throw new RequestError(`${label}: M must be an integer between 2 and ${MAX_M}`);
  }
  if (!isPositiveInteger(config.efConstruction) || config.efConstruction > MAX_EF) {
    throw new RequestError(`${label}: efConstruction must be an integer between 1 and ${MAX_EF}`);
  }
  if (!isPositiveInteger(config.efSearch) || config.efSearch > MAX_EF) {
    throw new RequestError(`${label}: efSearch must be an integer between 1 and ${MAX_EF}`);
  }
}

function makeCreateConfig(body, env) {
  const config = {
    dim: body.dims ?? body.dim,
    maxElements: body.maxElements ?? DEFAULT_MAX_ELEMENTS,
    metric: body.metric || 'cosine',
    quantized: body.quantized !== false,
    M: body.M ?? 16,
    efConstruction: body.efConstruction ?? body.efC ?? 200,
    efSearch: body.efSearch ?? body.efS ?? 100
  };
  validateCreateConfig(config, 'Invalid index config', positiveEnvInt(env, 'MAX_ELEMENTS_LIMIT', DEFAULT_MAX_ELEMENTS));
  return config;
}

function configForMetadata(config) {
  return {
    dims: config.dim,
    maxElements: config.maxElements,
    metric: config.metric,
    quantized: config.quantized,
    initParams: {
      M: config.M,
      efC: config.efConstruction,
      efS: config.efSearch
    }
  };
}

function configFromMetadata(metadata, fallbackDims, env) {
  const initParams = metadata?.initParams || {};
  const config = {
    dim: metadata?.dims ?? fallbackDims,
    maxElements: metadata?.maxElements ?? DEFAULT_MAX_ELEMENTS,
    metric: metadata?.metric || 'cosine',
    quantized: metadata?.quantized !== false,
    M: initParams.M ?? 16,
    efConstruction: initParams.efC ?? initParams.efConstruction ?? 200,
    efSearch: initParams.efS ?? initParams.efSearch ?? 100
  };
  validateCreateConfig(config, 'Invalid snapshot config', positiveEnvInt(env, 'MAX_ELEMENTS_LIMIT', DEFAULT_MAX_ELEMENTS));
  return config;
}

function nextSnapshotKey() {
  const timestamp = String(Date.now()).padStart(13, '0');
  const sequence = String(snapshotSequence++).padStart(6, '0');
  return `${SNAPSHOT_KEY_PREFIX}${timestamp}-${sequence}.pnck`;
}

async function findLatestSnapshot(env) {
  if (!env?.INDEX_BUCKET) return null;

  let latest = null;
  let cursor;
  do {
    const listed = await env.INDEX_BUCKET.list({ prefix: SNAPSHOT_KEY_PREFIX, cursor });
    for (const obj of listed.objects) {
      if (!latest || obj.key > latest.key) latest = obj;
    }
    cursor = listed.truncated ? listed.cursor : undefined;
  } while (cursor);

  if (latest) return latest.key;

  const legacy = await env.INDEX_BUCKET.head(LEGACY_SNAPSHOT_KEY);
  return legacy ? LEGACY_SNAPSHOT_KEY : null;
}

async function persistIndex(env, options = {}) {
  if (!index || !indexConfig) return false;
  const compact = options.compact !== false;
  if (index.ghostCount > 0) {
    if (!compact) return false;
    index.compact();
  }
  const snapshot = index.export();
  localSnapshot = {
    bytes: snapshot,
    config: { ...indexConfig },
    key: 'local-memory-snapshot'
  };
  if (!env?.INDEX_BUCKET) return true;
  // Keys are append-only in this example. Configure an R2 lifecycle rule or
  // delete older snapshots when adapting this persistence scheme for production.
  const key = nextSnapshotKey();
  await env.INDEX_BUCKET.put(key, snapshot, {
    customMetadata: Object.fromEntries(
      Object.entries(configForMetadata(indexConfig)).map(([k, v]) => [k, typeof v === 'string' ? v : JSON.stringify(v)])
    ),
    httpMetadata: { contentType: 'application/octet-stream' }
  });
  restoreState.lastSnapshotKey = key;
  return true;
}

async function restoreIndex(env) {
  if (index) return false;
  const generation = restoreGeneration;
  if (!env?.INDEX_BUCKET && localSnapshot) {
    const start = performance.now();
    const restored = await Pikelet.restore(localSnapshot.bytes, {
      maxElements: localSnapshot.config.maxElements,
      efSearch: localSnapshot.config.efSearch
    });
    if (generation !== restoreGeneration || index) {
      restored.dispose();
      return false;
    }
    index = restored;
    indexConfig = { ...restored.config };
    restoreState.restoreCount += 1;
    restoreState.restoredAt = new Date().toISOString();
    restoreState.lastRestoreMs = performance.now() - start;
    restoreState.lastFetchMs = 0;
    restoreState.lastDeserializeMs = restoreState.lastRestoreMs;
    restoreState.lastSnapshotBytes = localSnapshot.bytes.byteLength;
    restoreState.lastSnapshotKey = localSnapshot.key;
    return true;
  }
  if (!env?.INDEX_BUCKET) return false;
  const key = await findLatestSnapshot(env);
  if (!key) return false;

  const fetchStart = performance.now();
  const obj = await env.INDEX_BUCKET.get(key);
  if (!obj) return false;
  const maxBytes = positiveEnvInt(env, 'MAX_SNAPSHOT_BYTES', DEFAULT_MAX_SNAPSHOT_BYTES);
  if (Number.isFinite(obj.size) && obj.size > maxBytes) {
    throw new RequestError(`Snapshot exceeds MAX_SNAPSHOT_BYTES (${obj.size} > ${maxBytes})`, 413);
  }
  const bytes = new Uint8Array(await obj.arrayBuffer());
  const fetchMs = performance.now() - fetchStart;
  if (bytes.byteLength > maxBytes) {
    throw new RequestError(`Snapshot exceeds MAX_SNAPSHOT_BYTES (${bytes.byteLength} > ${maxBytes})`, 413);
  }

  const metadata = parseCustomMetadata(obj.customMetadata || {});
  const inspected = Pikelet.inspectSnapshot(bytes);
  const maxElementsLimit = positiveEnvInt(env, 'MAX_ELEMENTS_LIMIT', DEFAULT_MAX_ELEMENTS);
  const restoredCapacity = metadata.maxElements ?? Math.max(DEFAULT_MAX_ELEMENTS, inspected.count);
  if (!isPositiveInteger(restoredCapacity) || restoredCapacity > maxElementsLimit || restoredCapacity < inspected.count) {
    throw new RequestError(`Invalid snapshot maxElements ${restoredCapacity}`);
  }
  let overrides;
  if (inspected.format === 'raw') {
    overrides = configFromMetadata(metadata, inspected.dim, env);
  } else {
    overrides = {
      maxElements: restoredCapacity,
      efSearch: metadata?.initParams?.efS ?? metadata?.initParams?.efSearch ?? 100
    };
  }

  const deserializeStart = performance.now();
  const restored = await Pikelet.restore(bytes, overrides);
  if (generation !== restoreGeneration || index) {
    restored.dispose();
    return false;
  }

  index = restored;
  indexConfig = { ...restored.config };
  restoreState.restoreCount += 1;
  restoreState.restoredAt = new Date().toISOString();
  restoreState.lastRestoreMs = performance.now() - fetchStart;
  restoreState.lastFetchMs = fetchMs;
  restoreState.lastDeserializeMs = performance.now() - deserializeStart;
  restoreState.lastSnapshotBytes = bytes.byteLength;
  restoreState.lastSnapshotKey = key;
  return true;
}

async function ensureIndex(env) {
  if (index) return true;
  if (restorePromise === null) {
    restorePromise = restoreIndex(env).finally(() => {
      restorePromise = null;
    });
  }
  return restorePromise;
}

function parseCustomMetadata(rawMetadata = {}) {
  const metadata = {};
  for (const [key, value] of Object.entries(rawMetadata)) {
    try { metadata[key] = JSON.parse(value); } catch { metadata[key] = value; }
  }
  return metadata;
}

function metadataSnapshotSummary(metadata, size) {
  const hasConfig = metadata.dims !== undefined || metadata.maxElements !== undefined || metadata.initParams !== undefined;
  if (!hasConfig) return null;
  return {
    format: 'pikelet',
    dim: metadata.dims ?? null,
    maxElements: metadata.maxElements ?? null,
    metric: metadata.metric ?? null,
    quantized: metadata.quantized ?? null,
    objectBytes: size ?? null,
    source: 'r2-metadata'
  };
}

async function inspectStoredSnapshot(env) {
  if (!env?.INDEX_BUCKET && localSnapshot) {
    return {
      available: true,
      key: localSnapshot.key,
      metadata: Pikelet.inspectSnapshot(localSnapshot.bytes)
    };
  }
  const key = await findLatestSnapshot(env);
  if (!key) return { available: false, key: null, metadata: null };
  const maxBytes = positiveEnvInt(env, 'MAX_SNAPSHOT_BYTES', DEFAULT_MAX_SNAPSHOT_BYTES);
  const head = await env.INDEX_BUCKET.head(key);
  if (!head) return { available: false, key, metadata: null };
  if (Number.isFinite(head.size) && head.size > maxBytes) {
    throw new RequestError(`Snapshot exceeds MAX_SNAPSHOT_BYTES (${head.size} > ${maxBytes})`, 413);
  }

  const metadata = parseCustomMetadata(head.customMetadata || {});
  const summary = metadataSnapshotSummary(metadata, head.size);
  if (summary) return { available: true, key, metadata: summary };

  const obj = await env.INDEX_BUCKET.get(key, { range: { offset: 0, length: 4096 } });
  if (!obj) return { available: false, key, metadata: null };
  const bytes = new Uint8Array(await obj.arrayBuffer());
  try {
    return { available: true, key, metadata: Pikelet.inspectSnapshot(bytes) };
  } catch {
    return {
      available: true,
      key,
      metadata: {
        format: 'unknown',
        objectBytes: head.size ?? bytes.byteLength,
        source: 'r2-range'
      }
    };
  }
}

async function runMutation(fn) {
  const previous = mutationPromise;
  let release;
  mutationPromise = new Promise(resolve => { release = resolve; });
  await previous.catch(() => {});
  try {
    return await fn();
  } finally {
    release();
  }
}

function healthBody(env) {
  return {
    ok: true,
    initialized: !!index,
    dims: indexConfig?.dim ?? null,
    count: index ? index.count : 0,
    read_only: isReadOnly(env),
    restore: restoreState
  };
}

function statsBody() {
  const memory = index.memoryUsage;
  return {
    dims: indexConfig.dim,
    count: index.count,
    live_count: index.liveCount,
    deleted_count: index.deletedCount,
    deleted_ratio: index.deletedRatio,
    capacity: index.capacity,
    remaining_capacity: index.remainingCapacity,
    memory: memory.logicalIndexBytes,
    memory_usage: memory,
    // Legacy response names retained for clients of the example.
    ghost_count: index.ghostCount,
    ghost_ratio: index.ghostRatio
  };
}

async function handleRequest(request, env, ctx) {
  const url = new URL(request.url);
  const method = request.method.toUpperCase();

  if (method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders(env) });
  }

  const adminRoute = ADMIN_ROUTES.has(url.pathname);
  if (adminRoute && isReadOnly(env)) {
    return jsonResponse({ error: 'Read-only mode' }, 403);
  }
  if (adminRoute && !env?.API_KEY && !allowInsecureAdmin(env)) {
    return jsonResponse({ error: 'Admin routes require API_KEY or explicit ALLOW_INSECURE_ADMIN=1' }, 403);
  }
  if (!PUBLIC_ROUTES.has(url.pathname) && !isAdminAllowed(request, env)) {
    return jsonResponse({ error: 'Unauthorized' }, 401);
  }
  if (url.pathname !== '/health' && isRateLimited(request, env)) {
    return jsonResponse({ error: 'Rate limit exceeded' }, 429);
  }

  if (url.pathname === '/health' && method === 'GET') {
    return jsonResponse(healthBody(env));
  }

  if (url.pathname === '/readiness' && method === 'GET') {
    const snapshot = await inspectStoredSnapshot(env);
    return jsonResponse({
      ...healthBody(env),
      snapshot_available: snapshot.available,
      snapshot_key: snapshot.key,
      snapshot: snapshot.metadata
    });
  }

  if (url.pathname === '/reset_cache' && method === 'POST') {
    return runMutation(async () => {
      restoreGeneration += 1;
      restorePromise = null;
      if (index) index.dispose();
      index = null;
      indexConfig = null;
      return jsonResponse({ ok: true, reset: true });
    });
  }

  if (url.pathname === '/init' && method === 'POST') {
    return runMutation(async () => {
      const body = await readJson(request, env);
      const config = makeCreateConfig(body, env);
      const next = await Pikelet.create(config);
      try {
        if (Array.isArray(body.vectors) && body.vectors.length > 0) {
          next.addBatch(body.vectors);
        }
      } catch (error) {
        next.dispose();
        throw error;
      }
      restoreGeneration += 1;
      if (index) index.dispose();
      index = next;
      indexConfig = { ...next.config };
      const persisted = await persistIndex(env);
      return jsonResponse({
        ok: true,
        dims: config.dim,
        count: index.count,
        inserted: Array.isArray(body.vectors) ? body.vectors.length : 0,
        persisted
      });
    });
  }

  if (url.pathname === '/import' && method === 'POST') {
    return runMutation(async () => {
      const bytes = await readBytes(request, env);
      const inspected = Pikelet.inspectSnapshot(bytes);
      const maxElements = Number.parseInt(
        url.searchParams.get('maxElements') || String(Math.max(DEFAULT_MAX_ELEMENTS, inspected.count)), 10
      );
      const efSearch = Number.parseInt(url.searchParams.get('efSearch') || '100', 10);
      let overrides = { maxElements, efSearch };
      if (inspected.format === 'raw') {
        overrides = makeCreateConfig({
          dims: Number.parseInt(url.searchParams.get('dims') || String(inspected.dim), 10),
          maxElements,
          metric: url.searchParams.get('metric') || inspected.metric,
          quantized: inspected.quantized,
          M: Number.parseInt(url.searchParams.get('M') || String(inspected.M), 10),
          efConstruction: Number.parseInt(
            url.searchParams.get('efConstruction') || String(inspected.efConstruction), 10
          ),
          efSearch
        }, env);
      } else {
        validateCreateConfig({
          dim: inspected.dim,
          maxElements,
          metric: inspected.metric,
          quantized: inspected.quantized,
          M: inspected.M,
          efConstruction: inspected.efConstruction,
          efSearch
        }, 'Invalid import config', positiveEnvInt(env, 'MAX_ELEMENTS_LIMIT', DEFAULT_MAX_ELEMENTS));
      }
      const next = await Pikelet.restore(bytes, overrides);
      restoreGeneration += 1;
      if (index) index.dispose();
      index = next;
      indexConfig = { ...next.config };
      const persisted = await persistIndex(env);
      return jsonResponse({ ok: true, count: index.count, dims: indexConfig.dim, persisted });
    });
  }

  if (url.pathname === '/stats' && method === 'GET') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    return jsonResponse(statsBody());
  }

  if (url.pathname === '/add' && method === 'POST') {
    return runMutation(async () => {
      if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
      const body = await readJson(request, env);
      const id = index.add(body.vector);
      const persisted = await persistIndex(env, { compact: false });
      return jsonResponse({ ok: true, id, count: index.count, persisted });
    });
  }

  if (url.pathname === '/add_batch' && method === 'POST') {
    return runMutation(async () => {
      if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
      const body = await readJson(request, env);
      if (!Array.isArray(body.vectors)) throw new RequestError('add_batch requires vectors');
      const ids = index.addBatch(body.vectors);
      const persisted = await persistIndex(env, { compact: false });
      return jsonResponse({ ok: true, ids, inserted: ids.length, count: index.count, persisted });
    });
  }

  if (url.pathname === '/delete' && method === 'POST') {
    return runMutation(async () => {
      if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
      const body = await readJson(request, env);
      const deleted = index.delete(body.id);
      return jsonResponse({
        ok: true,
        deleted,
        count: index.count,
        deleted_count: index.deletedCount,
        persisted: false
      });
    });
  }

  if (url.pathname === '/compact' && method === 'POST') {
    return runMutation(async () => {
      if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
      index.compact();
      const persisted = await persistIndex(env);
      return jsonResponse({ ok: true, count: index.count, ghost_count: index.ghostCount, persisted });
    });
  }

  if ((url.pathname === '/search' || url.pathname === '/search_debug') && method === 'POST') {
    if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
    const body = await readJson(request, env);
    const k = Math.min(Number.parseInt(body.k ?? '10', 10), MAX_RESULTS);
    const efSearch = body.efSearch ?? body.ef;
    if (efSearch !== undefined && (!isPositiveInteger(efSearch) || efSearch > MAX_EF)) {
      return jsonResponse({ error: `efSearch must be an integer between 1 and ${MAX_EF}` }, 400);
    }
    const searchOptions = efSearch === undefined ? undefined : { efSearch };
    const start = performance.now();
    const neighbors = body.allowedIds
      ? index.searchFiltered(body.query, k, new Set(body.allowedIds), searchOptions)
      : index.search(body.query, k, searchOptions);
    return jsonResponse({
      ok: true,
      neighbors,
      results: neighbors,
      search_ms: performance.now() - start,
      count: index.count
    });
  }

  if (url.pathname === '/export' && method === 'POST') {
    return runMutation(async () => {
      if (!await ensureIndex(env)) return jsonResponse({ error: 'Index not initialized. Call /init first.' }, 503);
      if (index.ghostCount > 0) index.compact();
      const persisted = await persistIndex(env, { compact: false });
      return binaryResponse(localSnapshot?.bytes ?? index.export(), {
        'X-Pikelet-Persisted': persisted ? '1' : '0'
      });
    });
  }

  return jsonResponse({
    ok: true,
    routes: {
      'GET /health': 'Public health check',
      'GET /readiness': 'Authenticated snapshot/readiness check',
      'GET /stats': 'Index stats',
      'POST /init': '{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }',
      'POST /search': '{ query: float[], k?, efSearch?, allowedIds? }',
      'POST /add': '{ vector: float[] }',
      'POST /add_batch': '{ vectors: float[][] }',
      'POST /delete': '{ id: number }',
      'POST /compact': 'Compact deleted entries',
      'POST /export': 'Export Pikelet snapshot',
      'POST /import': 'Import Pikelet snapshot',
      'POST /reset_cache': 'Drop warm in-memory index'
    }
  });
}

export default {
  async fetch(request, env, ctx) {
    try {
      return withCors(await handleRequest(request, env, ctx), env);
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      const status = error instanceof RequestError
        ? error.status
        : (error instanceof PikeletError && CLIENT_ERROR_CODES.has(error.code) ? 400 : 500);
      const body = { error: status >= 500 ? 'Internal server error' : message };
      if (status < 500 && error instanceof PikeletError) body.code = error.code;
      return withCors(jsonResponse(body, status), env);
    }
  }
};
