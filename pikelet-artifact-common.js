'use strict';
// Shared by the range and sketch artifact readers/builders: read budgets and
// range validation, range sources, the result heap, snapshot parsing, and
// SHA-256 helpers. Not a public entry; pikelet-artifact.js re-exports what is public.
// Split out of pikelet-artifact.js (the public entry, which re-exports the
// three parts); see that file for the module map.

const { pikeletError, PIKELET_ERROR_CODES } = require('./pikelet-errors.js');

const PIKELET_MAGIC = 0x504E434B;
const V1_ENVELOPE_HEADER_SIZE = 24;
const V2_ENVELOPE_HEADER_SIZE = 20;
const V3_ENVELOPE_HEADER_SIZE = 32;
const MAPPING_ENTRY_SIZE = 8;
const UINT8_HNSW_MAGIC_V1 = 0x49384831;

// Read limits are layered, because header fields are untrusted and a hostile
// count/offset must not turn into a multi-GB read or allocation before the
// truncation checks run:
// - MAX_ARTIFACT_READ_BYTES is the absolute backstop on any single read,
//   blocking the 32-bit-overflow class of pathological requests. It is also
//   enforced independently inside NodeFileRangeSource.
// - Open-path reads (id map, router, resident sketch tier) default to the
//   much tighter DEFAULT_OPEN_READ_BYTES, configurable per open() via
//   options.maxReadBytes — resident segments are small by design, so a read
//   near this limit means a hostile or misconfigured header.
// - Query-path coalesced ranges are split at MAX_COALESCED_RANGE_BYTES so no
//   gap/parallelism combination can drive one giant fetch.
// - Whole-segment verification streams in VERIFY_CHUNK_BYTES chunks where a
//   streaming hash exists (Node); WebCrypto cannot stream SHA-256, so other
//   runtimes fall back to a single read bounded by the open budget.
const MAX_ARTIFACT_READ_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_OPEN_READ_BYTES = 256 * 1024 * 1024;
const MAX_COALESCED_RANGE_BYTES = 16 * 1024 * 1024;
const VERIFY_CHUNK_BYTES = 64 * 1024 * 1024;

async function mapLimit(items, limit, fn) {
    const n = !Number.isFinite(limit) ? items.length : Math.min(Math.max(1, Math.trunc(limit)), items.length);
    if (n === 0) return [];
    const out = new Array(items.length);
    let next = 0;
    await Promise.all(Array.from({ length: n }, async () => {
        for (let i = next++; i < items.length; i = next++) {
            out[i] = await fn(items[i], i);
        }
    }));
    return out;
}

// Validate an artifact-derived (offset, length) before issuing source.read.
// Rejects non-integer/negative/overflowing ranges, anything past the cap, and
// — when the source reports a numeric size — anything past the source end, so
// the read is refused instead of attempted. Returns the length for convenience.
function checkArtifactRange(source, offset, length, label, limit = MAX_ARTIFACT_READ_BYTES) {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length)
        || offset < 0 || length < 0 || !Number.isSafeInteger(offset + length)) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} range is out of bounds`, { offset, length });
    }
    const max = Math.min(limit, MAX_ARTIFACT_READ_BYTES);
    if (length > max) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} range exceeds the maximum read size`, { length, max });
    }
    if (source && Number.isSafeInteger(source.size) && offset + length > source.size) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} range extends past the source size`, { offset, length, size: source.size });
    }
    return length;
}

// Read a checked artifact region: validate (offset, length) before touching
// the source so a hostile header cannot drive a giant read or allocation.
async function readChecked(source, offset, length, label, limit) {
    checkArtifactRange(source, offset, length, label, limit);
    return asUint8Array(await source.read(offset, length));
}

// Per-open read budget: undefined keeps the default, Infinity defers to the
// absolute backstop, anything else must be a positive finite number and is
// honored strictly — a budget too small for the artifact's resident segments
// fails the open with a coded error rather than being silently raised.
function resolveMaxReadBytes(value) {
    if (value === undefined) return DEFAULT_OPEN_READ_BYTES;
    if (value === Infinity) return MAX_ARTIFACT_READ_BYTES;
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
            'maxReadBytes must be a positive number or Infinity', { maxReadBytes: value });
    }
    return value;
}
function asUint8Array(bytes) {
    if (bytes instanceof Uint8Array) return bytes;
    if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
    if (ArrayBuffer.isView(bytes)) {
        return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    }
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'Range source returned a non-binary value');
}

// Both readers accept a maxCacheBytes option: undefined keeps the default,
// Infinity disables the bound, and any other value must be a positive number.
// Each reader raises small budgets to a floor so the LRU always holds enough
// records for one search round.
function resolveMaxCacheBytes(value, floorBytes, defaultBytes) {
    if (value === undefined) return defaultBytes;
    if (value === Infinity) return Infinity;
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'maxCacheBytes must be a positive number or Infinity', { maxCacheBytes: value });
    }
    return Math.max(value, floorBytes);
}

function readU32(view, state) {
    const value = view.getUint32(state.offset, true);
    state.offset += 4;
    return value;
}

function readU16(view, state) {
    const value = view.getUint16(state.offset, true);
    state.offset += 2;
    return value;
}

function readF32(view, state) {
    const value = view.getFloat32(state.offset, true);
    state.offset += 4;
    return value;
}

function compareDistancesAsc(a, b) {
    return a.distance - b.distance || a.id - b.id;
}

class MinHeap {
    constructor(compare) {
        this.items = [];
        this.compare = compare;
    }

    get size() {
        return this.items.length;
    }

    push(value) {
        const items = this.items;
        items.push(value);
        this.bubbleUp(items.length - 1);
    }

    bubbleUp(index) {
        const items = this.items;
        const value = items[index];
        while (index > 0) {
            const parent = (index - 1) >> 1;
            if (this.compare(items[parent], value) <= 0) break;
            items[index] = items[parent];
            index = parent;
        }
        items[index] = value;
    }

    pop() {
        const items = this.items;
        if (items.length === 0) return undefined;
        const top = items[0];
        const value = items.pop();
        if (items.length > 0) {
            let index = 0;
            while (true) {
                let child = index * 2 + 1;
                if (child >= items.length) break;
                if (child + 1 < items.length && this.compare(items[child + 1], items[child]) < 0) {
                    child++;
                }
                if (this.compare(items[child], value) >= 0) break;
                items[index] = items[child];
                index = child;
            }
            items[index] = value;
        }
        return top;
    }

    peek() {
        return this.items[0];
    }

    sorted() {
        return [...this.items].sort(this.compare);
    }
}

class NodeFileRangeSource {
    constructor(filePath) {
        const fs = require('fs');
        this.fs = fs;
        this.filePath = filePath;
        this.fd = fs.openSync(filePath, 'r');
        try {
            // fstat on the descriptor we hold: no second path lookup to race,
            // and if it fails the descriptor is released before the throw.
            this.size = fs.fstatSync(this.fd).size;
        } catch (err) {
            fs.closeSync(this.fd);
            this.fd = null;
            throw err;
        }
        this.preferredParallelism = Infinity;
        this.preferredGapBytes = 2048;
    }

    async read(offset, length) {
        if (this.fd === null) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'NodeFileRangeSource is closed', { filePath: this.filePath });
        }
        // Defense in depth: the callers validate artifact-derived ranges, but
        // a future caller must not be able to drive a giant Buffer.alloc or a
        // read past the file end through this source directly.
        if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length)
            || offset < 0 || length < 0 || length > MAX_ARTIFACT_READ_BYTES
            || offset + length > this.size) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'NodeFileRangeSource.read() range is out of bounds', { offset, length, size: this.size });
        }
        const buffer = Buffer.alloc(length);
        let bytesRead = 0;
        while (bytesRead < length) {
            const chunk = this.fs.readSync(this.fd, buffer, bytesRead, length - bytesRead, offset + bytesRead);
            if (chunk === 0) break;
            bytesRead += chunk;
        }
        // Return only the bytes actually read so callers' truncation checks
        // fire instead of parsing an unwritten buffer tail.
        return bytesRead === length ? buffer : buffer.subarray(0, bytesRead);
    }

    async close() {
        if (this.fd !== null) {
            this.fs.closeSync(this.fd);
            this.fd = null;
        }
    }
}

// The artifact readers apply the same query-vector contract as the core
// engine (pikelet-core.js assertNumericVector / validateVectorValues): a
// plain array must hold real numbers (Float32Array.from would coerce '1',
// '', true silently), every component must be finite for every metric (an
// L2 search over NaN would otherwise return distance: NaN results), and a
// cosine query must have a usable norm.
function normalizeQuery(query, dim, metric = 0) {
    if (!(query instanceof Float32Array)) {
        if (!Array.isArray(query)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR, 'search() query must be a Float32Array or number[]');
        }
        for (let i = 0; i < query.length; i++) {
            if (typeof query[i] !== 'number') {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR,
                    `search() query must contain only numbers; found ${typeof query[i]} at index ${i}`, { index: i, actualType: typeof query[i] });
            }
        }
        query = Float32Array.from(query);
    }
    if (query.length !== dim) {
        throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH, `search() query dimension ${query.length} does not match artifact dimension ${dim}`, { queryDim: query.length, dim });
    }
    let norm = 0;
    for (let i = 0; i < query.length; i++) {
        const value = query[i];
        if (!Number.isFinite(value)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR, 'search() query contains non-finite value (NaN or Infinity)', { index: i, reason: 'non_finite' });
        }
        norm += value * value;
    }
    if (metric === 1) {
        norm = Math.sqrt(norm);
        if (!Number.isFinite(norm) || norm <= 1e-30) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR, 'search() query has invalid cosine norm', { reason: 'invalid_cosine_norm' });
        }
        const normalized = new Float32Array(query.length);
        for (let i = 0; i < query.length; i++) normalized[i] = query[i] / norm;
        return normalized;
    }
    return query;
}

// Positive-integer search arguments (k, rerank, efSearch) shared by both
// artifact readers: integers only, and bounded by the artifact's row count
// before they size any allocation — an artifact cannot return more than
// count results, so a k of 1e9 over 300 rows must not allocate 1e9 slots.
function resolveSearchK(k, count) {
    if (!Number.isSafeInteger(k) || k < 1) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'search() k must be a positive integer', { k });
    }
    return Math.min(k, count);
}
// undefined, null, and 0 all mean "use the default" — 0 kept its historical
// `options.rerank || default` meaning so existing callers do not break;
// anything else must be a positive safe integer.
function resolveOptionalPositiveInt(value, name, fallback) {
    if (value === undefined || value === null || value === 0) return fallback;
    if (!Number.isSafeInteger(value) || value < 1) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `search() ${name} must be a positive integer (0 or undefined selects the default)`, { [name]: value });
    }
    return value;
}

function unwrapSnapshot(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.byteLength < 4 || view.getUint32(0, true) !== PIKELET_MAGIC) return bytes;
    // Envelope fields are untrusted: every size read from them must be
    // checked against the actual buffer before slicing, so a corrupt
    // envelope fails closed with a coded error instead of a raw RangeError.
    if (bytes.byteLength < 8) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Pikelet envelope is truncated', { byteLength: bytes.byteLength });
    }
    const version = view.getUint32(4, true);
    if (version === 1 || version === 2) {
        const headerSize = version === 1 ? V1_ENVELOPE_HEADER_SIZE : V2_ENVELOPE_HEADER_SIZE;
        if (bytes.byteLength < headerSize) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Pikelet envelope is truncated', { version, byteLength: bytes.byteLength });
        }
        return bytes.subarray(headerSize);
    }
    if (version === 3) {
        if (bytes.byteLength < V3_ENVELOPE_HEADER_SIZE) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Pikelet envelope is truncated', { version, byteLength: bytes.byteLength });
        }
        const mappingCount = view.getUint32(24, true);
        const rawSize = view.getUint32(28, true);
        const rawOffset = V3_ENVELOPE_HEADER_SIZE + mappingCount * MAPPING_ENTRY_SIZE;
        if (rawOffset + rawSize > bytes.byteLength) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Pikelet envelope declares more data than the snapshot contains', { mappingCount, rawSize, byteLength: bytes.byteLength });
        }
        return bytes.subarray(rawOffset, rawOffset + rawSize);
    }
    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `Unsupported Pikelet envelope version ${version}`, { version });
}

function parseUint8Snapshot(bytes) {
    const raw = unwrapSnapshot(asUint8Array(bytes));
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    let offset = 0;
    // Snapshot bytes are untrusted: sizes read from the header drive every
    // subsequent read, so each read checks the remaining buffer and fails
    // closed with a coded error instead of a raw RangeError.
    const truncated = () => pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot is truncated', { offset, byteLength: raw.byteLength });
    const u32 = () => {
        if (raw.byteLength - offset < 4) throw truncated();
        const value = view.getUint32(offset, true);
        offset += 4;
        return value;
    };
    const f32 = () => {
        if (raw.byteLength - offset < 4) throw truncated();
        const value = view.getFloat32(offset, true);
        offset += 4;
        return value;
    };

    const magic = u32();
    const dim = u32();
    const version = u32();
    const count = u32();
    const entryPoint = u32();
    const maxLevel = u32();
    const M = u32();
    const M0 = u32();
    const metric = u32();
    const efConstruction = u32();
    if (magic !== UINT8_HNSW_MAGIC_V1) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Search Artifact export currently supports uint8 Pikelet snapshots only');
    }
    // Contract §6: reject unknown future versions instead of parsing them
    // as v2 — a changed layout must fail closed, not misparse.
    if (version > 2) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported uint8 snapshot format version', { version });
    }
    if (metric !== 0 && metric !== 1) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported uint8 snapshot metric', { metric });
    }
    // Same structural sanity the engine's own deserialize enforces; the
    // level cap mirrors MAX_DESERIALIZE_LEVEL in src/uint8_float_hnsw.hpp.
    if (dim < 1 || count < 1 || entryPoint >= count || maxLevel > 64
        || !Number.isSafeInteger(count * dim)
        || raw.byteLength - offset < count * 8 + count * dim) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot header is inconsistent', { dim, count, entryPoint, maxLevel, byteLength: raw.byteLength });
    }

    const scales = new Float32Array(count);
    const offsets = new Float32Array(count);
    for (let i = 0; i < count; i++) scales[i] = f32();
    for (let i = 0; i < count; i++) offsets[i] = f32();
    const qdata = raw.subarray(offset, offset + count * dim);
    offset += qdata.byteLength;

    // Adjacency bounds mirror the engine's own deserializer: a level's edge
    // count is capped by M0 (layer 0) / M (upper layers), the bytes it claims
    // (4 per id + 4 per serialized distance) must still be in the buffer, and
    // every neighbor id must address a node — all checked before the
    // Uint32Array for that level is allocated, so a crafted count cannot
    // drive a giant allocation that truncation would only catch afterwards.
    // Same envelope as the engine: create() accepts M in [2, 128] and fixes
    // M0 = 2*M, and its own import rejects M outside that range.
    if (!(M >= 2 && M <= 128) || !(M0 >= M && M0 <= 256)) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot graph parameters are implausible', { M, M0 });
    }
    // Edge layout follows the engine's deserializer (uint8_float_hnsw.hpp):
    // version >= 2 stores {u32 id, f32 distance} per edge, version 1 stores
    // the u32 id only (distances are recomputed on load). The artifact
    // readers never use the stored distance, so both parse to the same graph.
    const edgeBytes = version >= 2 ? 8 : 4;
    const levels = new Uint16Array(count);
    const base = new Array(count);
    const upper = Array.from({ length: count }, () => []);
    for (let id = 0; id < count; id++) {
        const level = u32();
        if (level > maxLevel) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot node level exceeds header maxLevel', { id, level, maxLevel });
        }
        levels[id] = level;
        for (let l = 0; l <= level; l++) {
            const size = u32();
            const cap = l === 0 ? M0 : M;
            if (size > cap) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot adjacency exceeds the graph parameter bound', { id, level: l, size, cap });
            }
            if (raw.byteLength - offset < size * edgeBytes) throw truncated();
            const edges = new Uint32Array(size);
            for (let e = 0; e < size; e++) {
                const neighbor = u32();
                if (neighbor >= count) {
                    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'uint8 snapshot neighbor id is out of range', { id, level: l, neighbor, count });
                }
                edges[e] = neighbor;
                if (version >= 2) offset += 4; // the serialized float edge distance, unused here
            }
            if (l === 0) base[id] = edges;
            else upper[id][l - 1] = edges;
        }
        if (!base[id]) base[id] = new Uint32Array(0);
    }

    return { kind: 'u8', dim, version, count, entryPoint, maxLevel, M, M0, metric, efConstruction, scales, offsets, qdata, levels, base, upper };
}
function sha256Bytes(bytes) {
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(bytes).digest();
}

async function sha256BytesAsync(bytes) {
    if (globalThis.crypto && globalThis.crypto.subtle) {
        const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
        return new Uint8Array(digest);
    }
    try {
        return new Uint8Array(sha256Bytes(bytes));
    } catch {
        return null; // no crypto available in this environment
    }
}

// Verify bytes against an expected 32-byte SHA-256 digest. Verification was
// requested by the caller, so a missing crypto backend fails closed rather
// than admitting unverified bytes.
async function verifySha256(bytes, expected, label) {
    const digest = await sha256BytesAsync(bytes);
    if (!digest) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            'Artifact verification requested but no crypto backend is available; pass verify:false to skip');
    }
    for (let b = 0; b < 32; b++) {
        if (digest[b] !== expected[b]) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `${label} failed hash verification`);
        }
    }
}

// Verify a whole segment against its digest without buffering it entirely.
// With a streaming hash backend (Node crypto) the segment is read and hashed
// in bounded chunks, so peak memory stays at chunkBytes no matter how large
// the segment. WebCrypto has no incremental SHA-256, so without a streaming
// backend the verify falls back to one bounded read — and refuses segments
// larger than the one-shot limit rather than buffering them.
async function verifySegmentSha256(source, offset, totalBytes, expected, label, options = {}) {
    const chunkBytes = Math.max(4096, Math.trunc(options.chunkBytes || VERIFY_CHUNK_BYTES));
    let hash = null;
    try {
        hash = require('crypto').createHash('sha256');
    } catch {
        hash = null;
    }
    if (hash) {
        let done = 0;
        while (done < totalBytes) {
            const len = Math.min(chunkBytes, totalBytes - done);
            const bytes = await readChecked(source, offset + done, len, label);
            if (bytes.byteLength !== len) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    `Artifact ${label} is truncated`, { offset: offset + done, expected: len, actual: bytes.byteLength });
            }
            hash.update(bytes);
            done += len;
        }
        const digest = new Uint8Array(hash.digest());
        for (let b = 0; b < 32; b++) {
            if (digest[b] !== expected[b]) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `${label} failed hash verification`);
            }
        }
        return;
    }
    const oneShotLimit = options.oneShotLimit || DEFAULT_OPEN_READ_BYTES;
    if (totalBytes > oneShotLimit) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} is too large to verify without a streaming crypto backend`,
            { totalBytes, oneShotLimit });
    }
    const bytes = await readChecked(source, offset, totalBytes, label, oneShotLimit);
    if (bytes.byteLength !== totalBytes) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            `Artifact ${label} is truncated`, { offset, expected: totalBytes, actual: bytes.byteLength });
    }
    await verifySha256(bytes, expected, label);
}

module.exports = {
    resolveSearchK,
    resolveOptionalPositiveInt,
    PIKELET_MAGIC,
    V1_ENVELOPE_HEADER_SIZE,
    V2_ENVELOPE_HEADER_SIZE,
    V3_ENVELOPE_HEADER_SIZE,
    MAPPING_ENTRY_SIZE,
    UINT8_HNSW_MAGIC_V1,
    MAX_ARTIFACT_READ_BYTES,
    DEFAULT_OPEN_READ_BYTES,
    MAX_COALESCED_RANGE_BYTES,
    VERIFY_CHUNK_BYTES,
    mapLimit,
    checkArtifactRange,
    readChecked,
    resolveMaxReadBytes,
    asUint8Array,
    resolveMaxCacheBytes,
    readU32,
    readU16,
    readF32,
    compareDistancesAsc,
    MinHeap,
    NodeFileRangeSource,
    normalizeQuery,
    unwrapSnapshot,
    parseUint8Snapshot,
    sha256Bytes,
    sha256BytesAsync,
    verifySha256,
    verifySegmentSha256,
};
