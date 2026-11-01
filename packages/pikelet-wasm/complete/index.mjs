// The one-file reader: open a .pikelet (spec/COMPLETE_PROFILE.md) and query
// it. Environment-neutral — the same module runs in Node (file path or
// range source) and the browser (HTTP range source, via a bundler for the
// CJS sketch-reader dependency). The query path needs no core HNSW engine:
// encoder, sketch reference scan, hydration, and calibration all run in
// pure JS; kind-3 artifacts load the reader-owned transformer WASM kernels,
// and at scale the resident scan optionally accelerates through the
// engine's SIMD scan kernel (options.sketchScanner — auto-staged in Node,
// injected in browsers), which only ever changes speed, never results.
//
//   const search = await openPikeletFile('docs.pikelet');           // Node
//   const search = await openPikeletFile(httpRangeSource(url));     // browser
//   const out = await search.query('how do workers restore snapshots');
//
// Every byte of the artifact is untrusted input (contract section 7): each
// offset, length, and count is validated — safe integer, within the header's
// fileBytes and the source's size, under the per-open read budget — before a
// read is issued or a buffer allocated, and every read must return exactly
// the bytes asked for. Format 2 files (profile pikelet-complete-v2) carry
// per-record corpus digests, so each hydrated record is verified on its own
// range read; format 1 files verify eager segments whole and hydrate records
// under the segment digest only (reported via info().corpusIntegrity).

import { loadStudentModel, embedTextWithStudent } from './student-embedder.mjs';
import { computeMatchQuality, computePreSearchAbstention } from './student-abstention.mjs';
import { createAbstentionScorer } from './retrieval-abstention.mjs';
import { openLexicalIndex, openLexicalIndexLazy } from './lexical.mjs';
import { parseRowMapSegment } from './rowmap.mjs';
import { PikeletSketchArtifact } from '../src/artifact/pikelet-artifact.js';
import { blobCacheGet, blobCachePut } from '../src/artifact/pikelet-artifact-common.js';
import { MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KINDS, KIND_NAMES } from './format.mjs';

import {
    KERNEL_LAYOUT, KERNEL_MAX_LAYERS, expectedBlobBytes, parseInlineTransformerEncoder,
    createInlineTransformerEmbedder, INLINE_TEST_VECTOR_TEXTS,
    buildInlineTestVectors, verifyInlineTestVectors,
    validateExpectedEmbedding, validateTestVectorTolerance,
} from './inline-transformer.mjs';

export { httpRangeSource } from './sources.mjs';
// Re-exported for the builder side: calibration fits coverage against the
// same fused (vector+BM25) ranking the reader serves, which needs an
// in-memory lexical searcher over the segment bytes already assembled by
// buildLexicalSegment, before the artifact is written to disk.
export { openLexicalIndex } from './lexical.mjs';
export { fuseCandidates, FUSION_DEFAULTS } from './fusion.mjs';
import { fuseCandidates } from './fusion.mjs';
export {
    KERNEL_LAYOUT, KERNEL_MAX_LAYERS, expectedBlobBytes, parseInlineTransformerEncoder,
    createInlineTransformerEmbedder, INLINE_TEST_VECTOR_TEXTS,
    buildInlineTestVectors, verifyInlineTestVectors,
    validateExpectedEmbedding, validateTestVectorTolerance,
};

// Supported container formats: header formatVersion -> manifest profile.
export const SUPPORTED_PROFILES = Object.freeze({ 1: 'pikelet-complete-v1', 2: 'pikelet-complete-v2' });
// Pre-rename profile strings (2026-09): every .pikelet published before this
// date — including live GitHub release assets — carries "pancake-complete-
// vN" in its manifest. Readers accept both spellings indefinitely so an
// already-published pack does not go dark on a reader upgrade; only
// SUPPORTED_PROFILES (above) is what a fresh build ever writes.
const LEGACY_PROFILES = Object.freeze({ 1: 'pancake-complete-v1', 2: 'pancake-complete-v2' });
export const CORPUS_LAYOUT_V2 = 'records-v2';

// Read budgets. Open-path reads (manifest, segment table, query-interp,
// corpus tables, evaluation) default to DEFAULT_OPEN_READ_BYTES per read and
// are configurable per open via options.maxReadBytes; Infinity defers to
// the absolute backstop. Records hydrate under options.maxRecordBytes.
const DEFAULT_OPEN_READ_BYTES = 256 * 1024 * 1024;
const MAX_ARTIFACT_READ_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_RECORD_BYTES = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_SEGMENTS = 64;
const MAX_RECORDS = 2 ** 31 - 1;
const MAX_PAGE_RECORDS = 65536;
const DIGEST_PAGE_CACHE = 64;
const RECORD_CACHE = 256;
// The embedded index is a .pikelet-sketch artifact (SKETCH_PROFILE.md);
// its fixed-size header is what format 2 commits to in the manifest.
const SKETCH_HEADER_BYTES = 256;
// Hybrid retrieval: BM25 candidates fetched from the lexical segment per
// query; the fusion rule and its constants live in fusion.mjs, shared with
// the calibrator so fit-time rankings match what is served.
export const LEXICAL_CANDIDATES = 24;
// scoreQuality's coverage term hydrates the fused top passagesNeeded
// records (asset.coverage.topK, 5 by default — see calibrate.mjs's
// COVERAGE_TOP_PASSAGES) once retrieval returns. The scorer that knows
// the real number is created inside retrievalScorer()'s own closure, not
// reachable from here, so this is a fixed approximation used to decide
// how many of fusedFull's records to include in the batched hydration
// below, alongside phrase-pinning's own candidates — hydrating a couple
// more than coverage strictly needs just costs a little extra bandwidth
// on a cache entry that goes unread; hydrating fewer only narrows how
// much of coverage's later hydration hits warm cache. Either way is
// harmless: it's a batch-sizing choice, not a correctness one.
const COVERAGE_BATCH_COUNT = 5;
// Only BM25 hits within this fraction of the top lexical score join fusion.
// Common query terms give every matching document a near-tied score (idf
// collapses them flat); a wide cutoff let that tied mass ride into fusion on
// a technicality. Kept identical to pikelet/src/calibrate.mjs's
// LEXICAL_CUTOFF so calibration is fit against the same fusion the reader
// actually serves.
export const LEXICAL_CUTOFF = 1.5;
// Query-interpretation segments above this size open lazily (kind 3's
// inline encoder is ~25 MiB); smaller ones keep the one-read eager path.
const LAZY_QI_BYTES = 4 * 1024 * 1024;
// Lexical segments above this size open lazily too (a wiki-scale inverted
// index runs to tens of MiB): only the header and doclen array become
// resident, term lookups interpolate into the remote table. Lazy reads are
// committed via the manifest digest but not individually verified — the
// transitional stance format-1 sketch rows carry.
const LAZY_LEXICAL_BYTES = 8 * 1024 * 1024;
// The resident sketch scan is O(count * sketchDims) per query in pure JS —
// ~500 ms at wiki scale (456k rows x 192 dims). At or above this many
// cells the reader loads the engine's SIMD scan kernel in the background
// (Node only; browser bundles inject one via options.sketchScanner) and
// serves queries from it once staged. Below it, docs-scale corpora scan in
// single-digit milliseconds and the engine load costs more than it saves.
const SCANNER_AUTO_CELLS = 16 * 1024 * 1024;
// The scanner's candidate output buffers are sized once at creation; the
// sketch reader falls back to the JS scan for any query whose C exceeds
// this, so an oversized rerank degrades speed, never recall.
const SCANNER_MAX_RERANK = 4096;

function bytesEqual(a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}
const METRIC_NAMES = { 0: 'l2', 1: 'cosine' };

const decoder = new TextDecoder();
const MAX_SAFE_BIG = BigInt(Number.MAX_SAFE_INTEGER);

function toHex(bytes) {
    let out = '';
    for (const b of bytes) out += b.toString(16).padStart(2, '0');
    return out;
}

const isSha256Hex = (value) => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);

// SHA-256 via WebCrypto (browser, workerd, Node 18+); node:crypto fallback
// for older Node, gated on an actual Node runtime check rather than just
// "WebCrypto is missing" — a browser can lack crypto.subtle too (privacy
// settings, a hardened build), and a bare dynamic import('node:crypto')
// there isn't a Node module resolution failure, it's a cross-origin module
// fetch a browser refuses outright, surfacing as an opaque module-loading
// error instead of the clear message below. Fails closed either way:
// verification is not optional in this reader.
async function sha256hex(bytes) {
    if (globalThis.crypto?.subtle) {
        return toHex(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes)));
    }
    if (typeof process !== 'undefined' && process.versions?.node) {
        const { createHash } = await import(/* webpackIgnore: true */ /* @vite-ignore */ 'node:crypto');
        return createHash('sha256').update(bytes).digest('hex');
    }
    throw new Error('No crypto backend available (crypto.subtle is missing) — '
        + 'cannot verify artifact integrity in this environment');
}

function asUint8Array(bytes) {
    if (bytes instanceof Uint8Array) return bytes;
    if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
    if (ArrayBuffer.isView(bytes)) return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    throw new Error('.pikelet range source returned a non-binary value');
}

function resolveBudget(value, label, fallback) {
    if (value === undefined) return fallback;
    if (value === Infinity) return MAX_ARTIFACT_READ_BYTES;
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw new Error(`${label} must be a positive number or Infinity`);
    }
    return Math.min(value, MAX_ARTIFACT_READ_BYTES);
}

// u64 fields become JS numbers only when they are exactly representable.
function u64(view, offset, label) {
    const big = view.getBigUint64(offset, true);
    if (big > MAX_SAFE_BIG) throw new Error(`.pikelet ${label} exceeds the safe integer range`);
    return Number(big);
}

// Validate an artifact-derived (offset, length) before touching the source:
// safe non-negative integers, no overflow, within the read budget, within
// the header's fileBytes, and within the source's size when it reports one.
function checkRange(source, offset, length, label, limit, fileBytes) {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length)
        || offset < 0 || length < 0 || !Number.isSafeInteger(offset + length)) {
        throw new Error(`.pikelet ${label} range is out of bounds`);
    }
    if (length > limit) {
        throw new Error(`.pikelet ${label} read of ${length} bytes exceeds the ${limit}-byte read budget`);
    }
    if (fileBytes !== undefined && offset + length > fileBytes) {
        throw new Error(`.pikelet ${label} range extends past the file (${offset}+${length} > ${fileBytes})`);
    }
    if (Number.isSafeInteger(source.size) && offset + length > source.size) {
        throw new Error(`.pikelet ${label} range extends past the source (${offset}+${length} > ${source.size})`);
    }
}

async function readChecked(source, offset, length, label, limit, fileBytes) {
    checkRange(source, offset, length, label, limit, fileBytes);
    const bytes = asUint8Array(await source.read(offset, length));
    if (bytes.length !== length) {
        throw new Error(`.pikelet ${label} read returned ${bytes.length} of ${length} bytes (truncated or misbehaving source)`);
    }
    return bytes;
}

async function fileSource(filePath) {
    const fs = await import(/* webpackIgnore: true */ /* @vite-ignore */ 'node:fs');
    let fd = fs.openSync(filePath, 'r');
    let size;
    try {
        size = fs.fstatSync(fd).size;
    } catch (err) {
        fs.closeSync(fd);
        throw err;
    }
    return {
        size,
        preferredParallelism: Infinity,
        preferredGapBytes: 2048,
        async read(offset, length) {
            if (fd === null) throw new Error('.pikelet file source is closed');
            const buffer = new Uint8Array(length);
            let bytesRead = 0;
            while (bytesRead < length) {
                const chunk = fs.readSync(fd, buffer, bytesRead, length - bytesRead, offset + bytesRead);
                if (chunk === 0) break;
                bytesRead += chunk;
            }
            return buffer.subarray(0, bytesRead);
        },
        // Idempotent: a second close() is a no-op, not EBADF.
        async close() {
            if (fd !== null) {
                const handle = fd;
                fd = null;
                fs.closeSync(handle);
            }
        },
    };
}

// A source restricted to one segment's window, for handing a container
// region to a reader that expects offset 0 = its own start (the embedded
// sketch artifact). Reads outside the window are refused here, before the
// underlying source sees them.
function windowSource(source, offset, length) {
    return {
        size: length,
        preferredParallelism: source.preferredParallelism,
        preferredGapBytes: source.preferredGapBytes,
        async read(off, len) {
            if (!Number.isSafeInteger(off) || !Number.isSafeInteger(len) || off < 0 || len < 0 || off + len > length) {
                throw new Error('.pikelet index segment read is outside the segment window');
            }
            return source.read(offset + off, len);
        },
        async close() {},
    };
}

function align16(n) {
    return Math.ceil(n / 16) * 16;
}

const viewOf = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

function toFloat32(vector, dim, what) {
    const v = vector instanceof Float32Array ? vector
        : (Array.isArray(vector) || ArrayBuffer.isView(vector)) ? Float32Array.from(vector) : null;
    if (!v || v.length !== dim) {
        throw new Error(`${what} must return a ${dim}-dimensional vector`);
    }
    for (let d = 0; d < dim; d++) {
        if (!Number.isFinite(v[d])) throw new Error(`${what} returned a non-finite component`);
    }
    return v;
}

export function base64Bytes(text) {
    if (typeof text !== 'string') return new Uint8Array(0);
    return typeof Buffer !== 'undefined'
        ? Buffer.from(text, 'base64')
        : Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
}

// Contract 4.4 mode 2: a host-supplied encoder for a pinned external model
// is checked against the declaration's verification vectors before the
// artifact serves a single query. Dimension and every component within the
// declared tolerance; any disagreement refuses the open.
export async function verifyHostEncoder(declaration, encodeQuery, dim) {
    const vectors = declaration.testVectors;
    if (!Array.isArray(vectors) || vectors.length === 0) {
        throw new Error('.pikelet kind-2 declaration carries no verification vectors');
    }
    if (declaration.dim !== undefined && declaration.dim !== dim) {
        throw new Error(`.pikelet kind-2 declaration dim ${declaration.dim} disagrees with manifest dim ${dim}`);
    }
    for (const tv of vectors) {
        if (!tv || typeof tv.text !== 'string') {
            throw new Error('.pikelet kind-2 verification vector is malformed');
        }
        // The expected embedding must be dim finite numbers: a malformed
        // entry would turn every comparison into NaN, and NaN > maxDiff is
        // false — the vector would "pass" without comparing anything.
        const label = `.pikelet kind-2 verification vector "${tv.text.slice(0, 40)}…"`;
        validateExpectedEmbedding(tv.embedding, dim, label);
        const tolerance = validateTestVectorTolerance(tv.tolerance, label);
        const vector = toFloat32(await encodeQuery(tv.text), dim, 'options.encodeQuery');
        let maxDiff = 0;
        for (let d = 0; d < dim; d++) {
            const diff = Math.abs(vector[d] - tv.embedding[d]);
            if (diff > maxDiff) maxDiff = diff;
        }
        if (!(maxDiff <= tolerance)) {
            throw new Error(`host encoder failed the artifact's verification vector "${tv.text.slice(0, 40)}…": `
                + `max component diff ${maxDiff.toExponential(2)} exceeds tolerance ${tolerance} `
                + `(declared model ${declaration.model})`);
        }
    }
    return { checked: vectors.length };
}

/**
 * Open a .pikelet complete artifact from a file path (Node) or a
 * { read(offset, length), size? } range source (any runtime).
 * Returns { query(text, {k}), info(), evaluation(), close() }.
 */
/**
 * Read a chain member's manifest and segment table without opening the corpus,
 * sketch or lexical machinery — LAYERED_PROFILE.md 4.3.
 *
 * A tombstone-only layer carries only a query-interp (kind 4) and a tombstones
 * segment. It is a chain member but not a search tier (3.1), so the full open
 * has nothing to do for it and every corpus/index step would need a branch.
 * The chain reader uses this instead; the header, manifest identity and each
 * segment digest are verified exactly as the full open verifies them.
 *
 * Returns { tombstoneOnly, tier, close }. When `tombstoneOnly` is false the
 * caller should close this and use openPikeletFile() instead: the member is a
 * search tier and needs the real open.
 */
/**
 * Read and identity-verify just a member's manifest, without opening its
 * segments. 5.1.1's locator walk needs this: to find a chain's ancestors from
 * the head alone it must read each member's `layer.parent` and stop at the
 * base, and paying for a full open per hop only to discard it is wasteful when
 * the walk may then reject the location.
 *
 * Returns { identity, manifest, isLayer, parent }. The identity check is the
 * same one openPikeletFile makes, so a tampered manifest fails here too.
 */
export async function readMemberManifest(input, options = {}) {
    const source = typeof input === 'string' ? await fileSource(input) : input;
    const owned = typeof input === 'string';
    const maxReadBytes = resolveBudget(options.maxReadBytes, 'maxReadBytes', DEFAULT_OPEN_READ_BYTES);
    try {
        const header = await readChecked(source, 0, HEADER_BYTES, 'header', maxReadBytes);
        const hv = viewOf(header);
        if (hv.getUint32(0, true) !== MAGIC) throw new Error('not a .pikelet file (bad magic)');
        const manifestBytes = hv.getUint32(8, true);
        const identity = [...new Uint8Array(header.buffer, header.byteOffset + 24, 32)]
            .map((b) => b.toString(16).padStart(2, '0')).join('');
        const fileBytes = u64(hv, 16, 'header fileBytes');
        const manifestBuf = await readChecked(source, HEADER_BYTES, manifestBytes, 'manifest', maxReadBytes, fileBytes);
        if (await sha256hex(manifestBuf) !== identity) {
            throw new Error('.pikelet manifest failed identity verification');
        }
        const manifest = JSON.parse(decoder.decode(manifestBuf));
        const isLayer = manifest.profile === 'pikelet-layer-v1';
        return {
            identity,
            manifest,
            isLayer,
            parent: isLayer && manifest.layer?.parent ? { ...manifest.layer.parent } : null,
        };
    } finally {
        if (owned && source.close) await source.close();
    }
}

/**
 * Validate a segment table against the identity-committed manifest and return
 * kind -> {offset, length, sha256}. The table sits OUTSIDE the identity, so
 * every entry must agree with the manifest entry at the same index: length,
 * digest shape, contiguous 16-aligned placement inside the file, and kind. A
 * known kind appears at most once. Unknown kinds are skipped (spec 3.3), but
 * the manifest must not name them as a kind this reader knows. Shared by the
 * full open and the chain-member shell so neither can drift from the other.
 */
function parseSegmentTable(table, manifest, segmentCount, manifestBytes, fileBytes) {
    const tableView = viewOf(table);
    const segments = new Map();
    let expectedOffset = align16(HEADER_BYTES + manifestBytes + segmentCount * TABLE_ENTRY_BYTES);
    for (let i = 0; i < segmentCount; i++) {
        const entry = i * TABLE_ENTRY_BYTES;
        const kindNumber = tableView.getUint32(entry, true);
        const kind = KIND_NAMES[kindNumber];
        const offset = u64(tableView, entry + 8, `segment ${i} offset`);
        const length = u64(tableView, entry + 16, `segment ${i} length`);
        const declared = manifest.segments[i];
        if (!declared || typeof declared !== 'object' || declared.bytes !== length || !isSha256Hex(declared.sha256)
            || offset !== expectedOffset || !Number.isSafeInteger(offset + length) || offset + length > fileBytes) {
            throw new Error(`.pikelet segment table disagrees with manifest at entry ${i}`);
        }
        if (kind === undefined) {
            if (typeof declared.kind !== 'string' || declared.kind in KINDS) {
                throw new Error(`.pikelet segment table entry ${i} has unknown kind ${kindNumber} but the manifest names it ${declared.kind}`);
            }
        } else {
            if (declared.kind !== kind) {
                throw new Error(`.pikelet segment table disagrees with manifest at entry ${i}`);
            }
            if (segments.has(kind)) throw new Error(`.pikelet carries more than one ${kind} segment`);
            segments.set(kind, { offset, length, sha256: declared.sha256 });
        }
        expectedOffset = align16(offset + length);
    }
    return segments;
}

export async function readChainMemberShell(input, options = {}) {
    const source = typeof input === 'string' ? await fileSource(input) : input;
    const owned = typeof input === 'string';
    const maxReadBytes = resolveBudget(options.maxReadBytes, 'maxReadBytes', DEFAULT_OPEN_READ_BYTES);
    const close = async () => { if (owned && source.close) await source.close(); };
    try {
        const header = await readChecked(source, 0, HEADER_BYTES, 'header', maxReadBytes);
        const hv = viewOf(header);
        if (hv.getUint32(0, true) !== MAGIC) throw new Error('not a .pikelet file (bad magic)');
        const formatVersion = hv.getUint32(4, true);
        const manifestBytes = hv.getUint32(8, true);
        const segmentCount = hv.getUint32(12, true);
        const identity = [...new Uint8Array(header.buffer, header.byteOffset + 24, 32)]
            .map((b) => b.toString(16).padStart(2, '0')).join('');
        const fileBytes = u64(hv, 16, 'header fileBytes');
        if (manifestBytes < 2 || manifestBytes > MAX_MANIFEST_BYTES || segmentCount < 1 || segmentCount > MAX_SEGMENTS
            || HEADER_BYTES + manifestBytes + segmentCount * TABLE_ENTRY_BYTES > fileBytes) {
            throw new Error('.pikelet header is implausible');
        }
        const manifestBuf = await readChecked(source, HEADER_BYTES, manifestBytes, 'manifest', maxReadBytes, fileBytes);
        if (await sha256hex(manifestBuf) !== identity) {
            throw new Error('.pikelet manifest failed identity verification');
        }
        const manifest = JSON.parse(decoder.decode(manifestBuf));
        if (manifest.profile !== 'pikelet-layer-v1' || formatVersion !== 2) {
            return { tombstoneOnly: false, tier: null, close };
        }
        if (manifest.layer?.records !== 0) {
            return { tombstoneOnly: false, tier: null, close };
        }
        if (!Array.isArray(manifest.segments) || manifest.segments.length !== segmentCount) {
            throw new Error('.pikelet manifest segment list disagrees with the header segment count');
        }
        // The same table checks as the full open. This path used to keep only
        // length and digest shape: it never compared the table's kind with the
        // manifest's and let a later duplicate overwrite an earlier one, so
        // whoever controls the table bytes (a mirror) could relabel another
        // committed segment as the tombstone mask without changing identity.
        const table = await readChecked(source, HEADER_BYTES + manifestBytes, segmentCount * TABLE_ENTRY_BYTES, 'segment table', maxReadBytes, fileBytes);
        const segments = parseSegmentTable(table, manifest, segmentCount, manifestBytes, fileBytes);
        for (const required of ['query-interp', 'tombstones']) {
            if (!segments.has(required)) throw new Error(`a tombstone-only layer is missing the ${required} segment`);
        }
        // 4.3: a tombstone-only layer carries no records, so it MUST NOT carry
        // the segments that would hold them.
        for (const forbidden of ['index', 'corpus', 'lexical']) {
            if (segments.has(forbidden)) throw new Error(`a tombstone-only layer must not carry a ${forbidden} segment (4.3)`);
        }
        return {
            tombstoneOnly: true,
            tier: {
                manifest, identity, recordCount: 0, dim: manifest.dim,
                sketch: null, lexicalIndex: null, hydrate: null,
                encoderInfo: null, segments,
                async readSegmentBytes(seg) {
                    const bytes = await readChecked(source, seg.offset, seg.length, 'chain segment', maxReadBytes, fileBytes);
                    if (await sha256hex(bytes) !== seg.sha256) {
                        throw new Error('.pikelet chain segment failed hash verification');
                    }
                    return bytes;
                },
            },
            close,
        };
    } catch (err) {
        await close().catch(() => {});
        throw err;
    }
}

export async function openPikeletFile(input, options = {}) {
    const source = typeof input === 'string' ? await fileSource(input) : input;
    const owned = typeof input === 'string';
    if (!source || typeof source.read !== 'function') {
        throw new Error('openPikeletFile() requires a file path or a range source with read(offset, length)');
    }
    const maxReadBytes = resolveBudget(options.maxReadBytes, 'maxReadBytes', DEFAULT_OPEN_READ_BYTES);
    const maxRecordBytes = resolveBudget(options.maxRecordBytes, 'maxRecordBytes', DEFAULT_RECORD_BYTES);
    const verifyRecords = options.verifyRecords !== false;
    // Opt-in full authentication of the lazy rerank rows at open: one
    // streamed pass over the vectors segment against the (identity-anchored)
    // vectorsSha256. Without it, rows that feed reranking are covered by the
    // whole-segment commitment but not verified per read — the documented
    // transitional stance (spec section 6) — and verifyVectors() can run the
    // same pass at any later point.
    const verifyIndexVectors = options.verifyIndexVectors === true;
    const verifyEncoder = options.verifyEncoder !== false;
    // Declared outside the try so a failure after the encoder is created
    // (count mismatch, bad corpus tables, ...) still releases its buffers.
    let disposeEncoder = () => {};
    // Kind-3 lazy encoder: the pending fetch+init promise. close() settles
    // it before disposing so an in-flight prefetch cannot leak its kernel.
    let encoderPending = null;
    // Set by the kind-3 branch below. Hoisted because `append` needs the base's
    // own encoder through the __chainTier seam (6.1), and the branch's own
    // ensureEmbedder is scoped to that branch.
    let resolveEmbedder = null;
    const settleEncoder = async () => {
        if (encoderPending) { try { await encoderPending; } catch { /* surfaced to queries */ } }
    };
    // WASM sketch scanner: staged in the background like the encoder, used
    // by queries only once ready (never awaited — the JS scan serves until
    // then, and permanently if staging fails). Reader-created scanners are
    // disposed on close; a caller-supplied scanner object stays caller-owned.
    let scanner = null;
    let scannerPending = null;
    let disposeScanner = () => {};
    const settleScanner = async () => {
        if (scannerPending) { try { await scannerPending; } catch { /* JS scan remains */ } }
    };
    try {
        // Header: the only read whose bounds come from nowhere but the spec.
        const header = await readChecked(source, 0, HEADER_BYTES, 'header', maxReadBytes);
        const headerView = viewOf(header);
        if (headerView.getUint32(0, true) !== MAGIC) {
            throw new Error('not a .pikelet file (bad magic)');
        }
        const formatVersion = headerView.getUint32(4, true);
        const profile = SUPPORTED_PROFILES[formatVersion];
        if (!profile) throw new Error(`unsupported .pikelet format version ${formatVersion}`);
        const manifestBytes = headerView.getUint32(8, true);
        const segmentCount = headerView.getUint32(12, true);
        const fileBytes = u64(headerView, 16, 'header fileBytes');
        if (manifestBytes < 2 || manifestBytes > MAX_MANIFEST_BYTES || segmentCount < 1 || segmentCount > MAX_SEGMENTS) {
            throw new Error('.pikelet header is implausible');
        }
        if (Number.isSafeInteger(source.size) && source.size !== fileBytes) {
            throw new Error(`.pikelet is truncated or padded: header says ${fileBytes} bytes, source has ${source.size}`);
        }
        const tableBytes = segmentCount * TABLE_ENTRY_BYTES;
        if (HEADER_BYTES + manifestBytes + tableBytes > fileBytes) {
            throw new Error('.pikelet header places the manifest or segment table past the file');
        }
        const identity = toHex(header.subarray(24, 56));
        // A caller pinning an expected identity (a shelf entry, a lockfile,
        // `--pack url#sha256`) is refused at the earliest possible moment —
        // one 64-byte header read in. Wrong knowledge object: don't touch
        // the knowledge object. The manifest hash check below still binds
        // the header's claim to the actual manifest bytes.
        if (options.expectedIdentity !== undefined) {
            if (typeof options.expectedIdentity !== 'string' || !/^[0-9a-f]{64}$/i.test(options.expectedIdentity)) {
                throw new Error('options.expectedIdentity must be a sha256 hex string');
            }
            if (identity !== options.expectedIdentity.toLowerCase()) {
                throw new Error(`.pikelet identity mismatch: expected ${options.expectedIdentity.toLowerCase()}, found ${identity}`);
            }
        }

        const manifestBuf = await readChecked(source, HEADER_BYTES, manifestBytes, 'manifest', maxReadBytes, fileBytes);
        if (await sha256hex(manifestBuf) !== identity) {
            throw new Error('.pikelet manifest failed identity verification');
        }
        let manifest;
        try { manifest = JSON.parse(decoder.decode(manifestBuf)); } catch (err) {
            throw new Error('.pikelet manifest is not valid JSON', { cause: err });
        }
        // 4.1 of the layered profile: a layer carries `pikelet-layer-v1` under
        // formatVersion 2, and a reader MUST refuse it by default — a layer
        // opened alone would silently serve a fraction of a corpus. Only the
        // chain reader, which opens every ancestor and verifies the hash chain,
        // passes asChainMember to accept one.
        const isLayerProfile = manifest?.profile === 'pikelet-layer-v1' && formatVersion === 2;
        const acceptLayer = isLayerProfile && options.asChainMember === true;
        if (!manifest || typeof manifest !== 'object'
            || (manifest.profile !== profile && manifest.profile !== LEGACY_PROFILES[formatVersion] && !acceptLayer)) {
            throw new Error(`unsupported profile ${manifest?.profile} for format version ${formatVersion}`);
        }
        if (!Array.isArray(manifest.segments) || manifest.segments.length !== segmentCount) {
            throw new Error('.pikelet manifest segment list disagrees with the header segment count');
        }
        const dim = manifest.dim;
        if (!Number.isSafeInteger(dim) || dim < 1 || dim > 65536) {
            throw new Error('.pikelet manifest dim is implausible');
        }
        if (!manifest.corpus || typeof manifest.corpus !== 'object') {
            throw new Error('.pikelet manifest has no corpus block');
        }

        const table = await readChecked(source, HEADER_BYTES + manifestBytes, tableBytes, 'segment table', maxReadBytes, fileBytes);
        const segments = parseSegmentTable(table, manifest, segmentCount, manifestBytes, fileBytes);
        // The evaluation segment, read and hash-verified. Named so both the
        // reader's own evaluation() and the __chainTier seam can reach it: a
        // chain has no evaluation segment of its own, and `verify_pack` called
        // `search.evaluation()` on a chain reader that had no such method,
        // which threw a TypeError instead of reporting the base's goldens.
        const readEvaluationSegment = async () => {
            const ev = segments.get('evaluation');
            if (!ev) return null;
            const bytes = await readChecked(source, ev.offset, ev.length, 'evaluation segment', maxReadBytes, fileBytes);
            if (await sha256hex(bytes) !== ev.sha256) {
                throw new Error('.pikelet evaluation segment failed hash verification');
            }
            let parsed;
            try { parsed = JSON.parse(decoder.decode(bytes)); } catch (err) {
                throw new Error('.pikelet evaluation segment is not valid JSON', { cause: err });
            }
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
                throw new Error('.pikelet evaluation segment must be a JSON object');
            }
            return parsed;
        };
        // The lineage segment (kind 7, LAYERED_PROFILE.md 6.3), parsed and
        // validated on first use. Present only on a base compacted from a
        // chain. buildLineageSegment wrote it and parseLineageSegment validated
        // it, but nothing on the read side ever reached either: a real
        // compacted pack could not translate a citation, which is the whole
        // point of the segment ("what lets a consumer treat a translated
        // citation as evidence rather than as a hint").
        //
        // Cached because the validation is not free and the segment is
        // immutable; the cache holds the parse, never a translation.
        let lineageCache;
        const readLineageSegment = async () => {
            if (lineageCache !== undefined) return lineageCache;
            const seg = segments.get('lineage');
            if (!seg) { lineageCache = null; return null; }
            const bytes = await readChecked(source, seg.offset, seg.length, 'lineage segment', maxReadBytes, fileBytes);
            if (await sha256hex(bytes) !== seg.sha256) {
                throw new Error('.pikelet lineage segment failed hash verification');
            }
            const { parseLineageSegment } = await import('./layer-compact.mjs');
            // Cross-checked against the manifest: the parser refuses a segment
            // whose last head is not compactedFrom.identity, or whose counts
            // disagree, so a tampered pair fails rather than translating.
            lineageCache = parseLineageSegment(bytes, manifest.compactedFrom || null);
            return lineageCache;
        };

        // oldId -> new dense id, by 6.3's rule: a surviving id loses one
        // position per tombstone below it. The per-byte prefix table is built
        // once and reused, because the naive form rescans the bitset from zero
        // on every call -- O(id) per citation, quadratic over a run of them on
        // a large pack.
        let compactPrefix;
        const compactMapFor = (lineage) => {
            if (compactPrefix === undefined) {
                const bytes = lineage.bitset.length;
                compactPrefix = new Uint32Array(bytes + 1);
                for (let b = 0; b < bytes; b++) {
                    let n = 0;
                    for (let k = 0; k < 8; k++) if ((lineage.bitset[b] >> k) & 1) n++;
                    compactPrefix[b + 1] = compactPrefix[b] + n;
                }
            }
            return (oldId) => {
                if (oldId < 0 || oldId >= lineage.oldRowTotal) return null;
                if ((lineage.bitset[oldId >> 3] >> (oldId & 7)) & 1) return null;
                const byte = oldId >> 3;
                let below = compactPrefix[byte];
                for (let k = 0; k < (oldId & 7); k++) {
                    if ((lineage.bitset[byte] >> k) & 1) below++;
                }
                return oldId - below;
            };
        };

        // Required segments are per-profile (layered profile 4.3). A layer with
        // records carries a tombstones segment on top of the complete set; a
        // TOMBSTONE-ONLY layer has neither an index nor a corpus segment and is
        // therefore not opened through this function at all — the chain reader
        // reads its two segments directly (see openPikeletChain), because
        // threading "no index, no corpus" through the whole of this open would
        // put a branch on every step of the single-file path.
        if (acceptLayer && manifest.layer?.records === 0) {
            throw new Error('a tombstone-only layer is read by the chain reader, not opened as a member');
        }
        const requiredSegments = acceptLayer
            ? ['index', 'corpus', 'query-interp', 'tombstones']
            : ['index', 'corpus', 'query-interp'];
        for (const required of requiredSegments) {
            if (!segments.has(required)) throw new Error(`.pikelet is missing the ${required} segment`);
        }

        // Corpus tables: count + offsets (+ page table for layout v2),
        // one contiguous region at the segment start, sized from the
        // identity-verified manifest and cross-checked against the segment's
        // own words after the read.
        const qi = segments.get('query-interp');
        const idx = segments.get('index');
        const corpus = segments.get('corpus');
        const declaredRecords = manifest.corpus.records;
        if (!Number.isSafeInteger(declaredRecords) || declaredRecords < 0 || declaredRecords > MAX_RECORDS) {
            throw new Error('.pikelet manifest corpus.records is implausible');
        }
        const layout = manifest.corpus.layout;
        const perRecord = formatVersion >= 2;
        if (perRecord && layout !== CORPUS_LAYOUT_V2) {
            throw new Error(`.pikelet format ${formatVersion} requires corpus layout ${CORPUS_LAYOUT_V2}, manifest says ${layout}`);
        }
        if (!perRecord && layout !== undefined) {
            throw new Error(`.pikelet format 1 carries corpus layout v1, manifest says ${layout}`);
        }
        let pageRecords = 0;
        let pages = 0;
        let offsetsAt;
        let tablesBytes;
        let recordDigestsAt = 0;
        let recordsAt;
        if (perRecord) {
            pageRecords = manifest.corpus.pageRecords;
            if (!Number.isSafeInteger(pageRecords) || pageRecords < 1 || pageRecords > MAX_PAGE_RECORDS) {
                throw new Error('.pikelet manifest corpus.pageRecords is implausible');
            }
            pages = Math.ceil(declaredRecords / pageRecords);
            if (manifest.corpus.pages !== pages || !isSha256Hex(manifest.corpus.pageTableSha256)
                || manifest.corpus.recordDigest !== 'sha256') {
                throw new Error('.pikelet manifest corpus integrity block is inconsistent');
            }
            offsetsAt = 8;
            tablesBytes = offsetsAt + 8 * (declaredRecords + 1) + 32 * pages;
            recordDigestsAt = tablesBytes;
            recordsAt = recordDigestsAt + 32 * declaredRecords;
        } else {
            offsetsAt = 4;
            tablesBytes = offsetsAt + 8 * (declaredRecords + 1);
            recordsAt = tablesBytes;
        }
        if (!Number.isSafeInteger(recordsAt) || recordsAt > corpus.length) {
            throw new Error('.pikelet manifest corpus.records is implausible for the corpus segment');
        }
        // Format 2 commits to the embedded sketch's 256-byte header
        // (metric/dim/count and the sketch's residentSha256/vectorsSha256),
        // anchoring the sketch's own integrity chain to the identity; the
        // sketch's self-checks alone verify against fields an attacker who
        // rewrites the segment also controls.
        if (perRecord && (!isSha256Hex(manifest.index?.headerSha256) || idx.length < SKETCH_HEADER_BYTES)) {
            throw new Error('.pikelet manifest carries no index header commitment (index.headerSha256)');
        }

        // After the segment table, every remaining open-path extent is known,
        // so the query-interp segment, the embedded sketch (two dependent
        // reads of its own), and the corpus tables fetch as one concurrent
        // wave instead of dependent rounds — at 100 ms/request that is most
        // of the open time.
        // A large query-interpretation segment (kind 3's inline encoder is
        // ~25 MiB) is not read at open: only its 16-byte header joins the
        // wave, the calibration region follows as one small dependent read,
        // and the encoder bytes arrive lazily — prefetched in the background
        // by default, awaited by the first query. The whole-segment digest
        // is verified when those bytes arrive, and the header/calibration
        // slices used at open are byte-compared against the verified
        // segment before any query can complete, so nothing unverified ever
        // influences a returned result. Small segments (kinds 1/2) keep the
        // one-read eager path unchanged.
        const qiDeferrable = qi.length > LAZY_QI_BYTES;
        const [qiFirst, sketch, tables, lexicalBytes, rowMapBytes] = await Promise.all([
            (async () => {
                if (qiDeferrable) {
                    return readChecked(source, qi.offset, Math.min(16, qi.length), 'query-interp header', maxReadBytes, fileBytes);
                }
                const bytes = await readChecked(source, qi.offset, qi.length, 'query-interp segment', maxReadBytes, fileBytes);
                if (await sha256hex(bytes) !== qi.sha256) {
                    throw new Error('.pikelet query-interp segment failed hash verification');
                }
                return bytes;
            })(),
            // Format 2's header commitment is checked inside open(), against
            // the same bytes the sketch parses (see expectedHeaderSha256).
            PikeletSketchArtifact.open(windowSource(source, idx.offset, idx.length), {
                maxReadBytes,
                ...(options.blobCache ? { blobCache: options.blobCache } : {}),
                ...(perRecord ? { expectedHeaderSha256: manifest.index.headerSha256 } : {}),
            }),
            readChecked(source, corpus.offset, tablesBytes, 'corpus tables', maxReadBytes, fileBytes),
            // Lexical index (kind 5, OPTIONAL). Small segments: eager
            // whole-segment read, digest-verified like the query-interp
            // segment. Wiki-scale segments open lazily — header + doclens
            // resident, term table and postings remote (see
            // LAZY_LEXICAL_BYTES for the integrity stance). Readers that
            // predate the kind skip it (spec 3.3).
            segments.has('lexical') ? (async () => {
                const seg = segments.get('lexical');
                if (seg.length > LAZY_LEXICAL_BYTES) {
                    // Confined to the segment's own window: lazy reads are
                    // not individually verified, so no field they carry may
                    // steer a read into another segment.
                    const lexRead = (off, len) => {
                        if (!Number.isSafeInteger(off) || !Number.isSafeInteger(len) || off < 0 || len < 0 || off + len > seg.length) {
                            throw new Error('.pikelet lexical segment read is outside the segment window');
                        }
                        return readChecked(source, seg.offset + off, len, 'lexical segment', maxReadBytes, fileBytes);
                    };
                    return { lazyIndex: await openLexicalIndexLazy(lexRead, seg.length) };
                }
                const bytes = await readChecked(source, seg.offset, seg.length, 'lexical segment', maxReadBytes, fileBytes);
                if (await sha256hex(bytes) !== seg.sha256) {
                    throw new Error('.pikelet lexical segment failed hash verification');
                }
                return { bytes };
            })() : null,
            // Row map (kind 8, OPTIONAL): a few bytes per record, read
            // eagerly and digest-verified. Present when the index holds
            // several vectors per record (sentence windows).
            segments.has('rowmap') ? (async () => {
                const seg = segments.get('rowmap');
                const bytes = await readChecked(source, seg.offset, seg.length, 'row map segment', maxReadBytes, fileBytes);
                if (await sha256hex(bytes) !== seg.sha256) {
                    throw new Error('.pikelet row map segment failed hash verification');
                }
                return bytes;
            })() : null,
        ]);

        const lexicalIndex = lexicalBytes
            ? (lexicalBytes.lazyIndex || openLexicalIndex(lexicalBytes.bytes))
            : null;
        if (lexicalIndex && lexicalIndex.docCount !== declaredRecords) {
            throw new Error(`.pikelet lexical segment covers ${lexicalIndex.docCount} records, corpus declares ${declaredRecords}`);
        }

        // Query interpretation: version word, kind, two length-prefixed
        // regions that must tile the segment exactly.
        if (qi.length < 16) throw new Error('.pikelet query-interp segment is too short');
        const qiView = viewOf(qiFirst);
        const qiVersion = qiView.getUint32(0, true);
        if (qiVersion !== 1) throw new Error(`unsupported query-interpretation version ${qiVersion}`);
        const qiKind = qiView.getUint32(4, true);
        const encoderLen = qiView.getUint32(8, true);
        const calibrationLen = qiView.getUint32(12, true);
        if (16 + encoderLen + calibrationLen !== qi.length) {
            throw new Error('.pikelet query-interp segment layout is inconsistent');
        }
        // Laziness only pays for kind 3 (the deferrable bulk IS the
        // encoder); a large segment of any other kind falls back to the
        // eager whole-segment read and verification, one dependent round.
        let qiBytes = qiFirst;
        let qiDeferred = qiDeferrable;
        if (qiDeferrable && qiKind !== 3) {
            qiBytes = await readChecked(source, qi.offset, qi.length, 'query-interp segment', maxReadBytes, fileBytes);
            if (await sha256hex(qiBytes) !== qi.sha256) {
                throw new Error('.pikelet query-interp segment failed hash verification');
            }
            qiDeferred = false;
        }
        // Deferred path: the calibration region arrives now (small), the
        // encoder region stays remote. Both header and calibration are
        // unverified until the full segment fetch cross-checks them.
        const calibrationRegion = qiDeferred
            ? await readChecked(source, qi.offset + 16 + encoderLen, calibrationLen, 'calibration region', maxReadBytes, fileBytes)
            : qiBytes.subarray(16 + encoderLen);
        const encoderBytes = qiDeferred ? null : qiBytes.subarray(16, 16 + encoderLen);
        let calibrationJson;
        try { calibrationJson = JSON.parse(decoder.decode(calibrationRegion)); } catch (err) {
            throw new Error('.pikelet calibration is not valid JSON', { cause: err });
        }
        if (!calibrationJson || typeof calibrationJson !== 'object') {
            throw new Error('.pikelet calibration must be a JSON object');
        }

        // kind 1 (student-inline-v1): pure-JS inline encoder + feature-stream
        // abstention. kind 2 (external-transformers-v1): the encoder is a
        // pinned declaration the HOST must satisfy via options.encodeQuery,
        // verified against the declaration's test vectors before serving;
        // calibration scores retrieval signals + query text. kind 3
        // (inline-transformer-v1): the pinned teacher as data, executed by
        // reader-owned kernels, verified against its own test vectors.
        let embed;
        let preScore = () => null;
        let scoreQuality;
        let encoderInfo;
        let encoderVerified = null;
        const retrievalScorer = () => {
            const scorer = createAbstentionScorer(calibrationJson.asset, base64Bytes(calibrationJson.vocabBloomBase64));
            const VERDICTS = { answer: 'strong', weak: 'weak', abstain: 'none' };
            return async (hits, context, fusedHits, scoreOptions) => {
                if (!scorer) return { match_quality: 'unscored' };
                // The coverage term grounds the verdict in the top passages'
                // text, so those records hydrate before scoring. Hydrated
                // from the fused (RRF) order when hybrid retrieval produced
                // one — calibrate.mjs fits coverage against the same fused
                // ranking, and a chunk BM25 ranks first but the embedding
                // ranks far down is exactly the case coverage exists to
                // catch; scoring it against vector-only order would miss it
                // again. Falls back to vector order (hits) when there is no
                // lexical segment or fusion did not change anything. When
                // the query is answered the pages are already hot for
                // result hydration; an abstained query costs these extra
                // reads. Hydration failures propagate — on format 2 a
                // record that fails its digest must fail the query, not
                // skew its verdict.
                const passageSource = fusedHits?.length ? fusedHits : hits;
                // A record's text starts with its own heading echoed as the
                // first line (ingest.mjs's section-to-chunk join). A
                // heading match still grounds the query but at reduced
                // credit — see retrieval-abstention.mjs's coverageFrac and
                // calibrate.mjs's HEADING_COVERAGE_WEIGHT, which this split
                // must match exactly (build-time and serve-time coverage
                // must agree, or the fitted thresholds are scoring a
                // different signal than the one they were calibrated
                // against).
                const splitHeadingBody = (text) => {
                    const nl = String(text || '').indexOf('\n');
                    return nl === -1 ? { heading: text, body: '' } : { heading: text.slice(0, nl), body: text.slice(nl + 1) };
                };
                // A chain supplies the passages itself (context.passages): its
                // ids are GLOBAL, and this hydrate only knows this member's
                // local ids, so letting it fetch them would throw on any id a
                // layer owns. Single-file callers pass none and it hydrates.
                const topTexts = Array.isArray(context.passages)
                    ? context.passages.slice(0, scorer.passagesNeeded || 1).map(splitHeadingBody)
                    : scorer.usesPassage && passageSource.length
                        ? (await Promise.all(passageSource.slice(0, scorer.passagesNeeded || 1)
                            .map((hit) => hydrate(hit.id)))).map((record) => splitHeadingBody(record?.text))
                        : [];
                // scoreOptions.vocabBloom is the chain's union bloom (4.5).
                // Absent for a single-file pack, where the fit's own bloom is
                // the whole vocabulary.
                const scored = await scorer.score(context.text, hits, topTexts, {
                    vocabBloom: scoreOptions?.vocabBloom,
                });
                return { match_quality: VERDICTS[scored.verdict] || scored.verdict, confidence: scored.p, grounding: scored.grounding || null };
            };
        };
        if (qiKind === 1) {
            const student = loadStudentModel(encoderBytes);
            encoderInfo = { kind: 'student-inline-v1' };
            embed = (text) => {
                const embedded = embedTextWithStudent(text, student);
                return { vector: toFloat32(embedded.vector, dim, 'inline student encoder'), embedded };
            };
            // Pre-search abstention skips the index entirely when it fires,
            // exactly as the Worker does.
            preScore = (context) => computePreSearchAbstention(context.embedded, calibrationJson);
            scoreQuality = (hits, context) => computeMatchQuality(hits, context.embedded, calibrationJson);
        } else if (qiKind === 2) {
            let declaration;
            try { declaration = JSON.parse(decoder.decode(encoderBytes)); } catch (err) {
                throw new Error('.pikelet kind-2 encoder declaration is not valid JSON', { cause: err });
            }
            if (!declaration || typeof declaration !== 'object') throw new Error('.pikelet kind-2 encoder declaration must be a JSON object');
            encoderInfo = declaration;
            const encodeQuery = options.encodeQuery;
            if (encodeQuery !== undefined && typeof encodeQuery !== 'function') {
                throw new Error('options.encodeQuery must be a function');
            }
            const hasVectors = Array.isArray(declaration.testVectors) && declaration.testVectors.length > 0;
            if (encodeQuery && verifyEncoder) {
                if (hasVectors) {
                    await verifyHostEncoder(declaration, encodeQuery, dim);
                    encoderVerified = true;
                } else if (options.allowUnverifiedEncoder === true) {
                    encoderVerified = false;
                } else {
                    throw new Error(`.pikelet declares an external encoder (${declaration.model}) without verification vectors, `
                        + 'so the host encoder cannot be checked against it (contract section 4.4); '
                        + 'pass options.allowUnverifiedEncoder: true to serve it anyway, marked unverified');
                }
            } else if (encodeQuery) {
                encoderVerified = false;
            }
            embed = async (text) => {
                if (!encodeQuery) {
                    throw new Error(`.pikelet declares an external encoder (${declaration.model}); `
                        + 'pass options.encodeQuery to openPikeletFile');
                }
                return { vector: toFloat32(await encodeQuery(text), dim, 'options.encodeQuery'), text };
            };
            scoreQuality = retrievalScorer();
        } else if (qiKind === 3) {
            // Kernels are reader-owned (spec 3.6) and load dynamically so
            // kind-1/2 files never touch them; the node-only glue is hidden
            // from bundlers, the web glue stays analyzable so browser builds
            // carry it. options.createEncoder overrides the loader.
            let createEncoder = options.createEncoder;
            if (!createEncoder) {
                try {
                    createEncoder = globalThis.process?.versions?.node
                        ? (await import(/* webpackIgnore: true */ /* @vite-ignore */ './encoder-kernels/encoder.node.mjs')).default
                        : (await import('./encoder-kernels/encoder.mjs')).default;
                } catch (err) {
                    throw new Error('kind-3 artifact requires the inline-transformer kernels at '
                        + 'complete/encoder-kernels/ (pikelet-wasm/complete); run '
                        + 'examples/one-file-search/encoder-spike/build-encoder.sh to rebuild them',
                    { cause: err });
                }
            }
            const openHeader = qiDeferred ? qiFirst.slice(0, 16) : null;
            const loadEmbedder = async () => {
                let encBytes = encoderBytes;
                if (qiDeferred) {
                    // The ~25 MiB encoder is the other bulk of a cold mount;
                    // options.blobCache serves it by the manifest's segment
                    // digest (hash-checked on the way out, as a fetch is).
                    const cached = await blobCacheGet(options.blobCache, qi.sha256, qi.length);
                    const full = cached || await readChecked(source, qi.offset, qi.length, 'query-interp segment', maxReadBytes, fileBytes);
                    if (!cached) {
                        if (await sha256hex(full) !== qi.sha256) {
                            throw new Error('.pikelet query-interp segment failed hash verification');
                        }
                        blobCachePut(options.blobCache, qi.sha256, full);
                    }
                    // The header and calibration slices used at open were
                    // unverified; the digest-verified segment must contain
                    // them byte for byte, or the open state is poisoned and
                    // every query must fail.
                    if (!bytesEqual(full.subarray(0, 16), openHeader)
                        || !bytesEqual(full.subarray(16 + encoderLen), calibrationRegion)) {
                        throw new Error('.pikelet query-interp segment bytes changed between open and encoder fetch');
                    }
                    encBytes = full.subarray(16, 16 + encoderLen);
                }
                const { declaration, vocabText, blob } = parseInlineTransformerEncoder(encBytes);
                if (declaration.dim !== undefined && declaration.dim !== dim) {
                    throw new Error(`.pikelet kind-3 declaration dim ${declaration.dim} disagrees with manifest dim ${dim}`);
                }
                const embedder = await createInlineTransformerEmbedder({ declaration, vocabText, blob, createEncoder, verify: verifyEncoder });
                encoderInfo = declaration;
                encoderVerified = verifyEncoder && Array.isArray(declaration.testVectors) && declaration.testVectors.length > 0;
                disposeEncoder = () => embedder.dispose();
                return { embedder, declaration };
            };
            const ensureEmbedder = () => (encoderPending ??= loadEmbedder());
            resolveEmbedder = ensureEmbedder;
            // Until the encoder arrives, info() serves the identity-verified
            // manifest declaration; encoderVerified stays null (unknown, not
            // unverified) and flips once test vectors have run.
            encoderInfo = {
                ...(manifest.encoder && typeof manifest.encoder === 'object' ? manifest.encoder : {}),
                kind: 'inline-transformer-v1',
                ...(qiDeferred ? { deferred: true } : {}),
            };
            if (!qiDeferred) {
                await ensureEmbedder();
            } else if (options.prefetchEncoder !== false) {
                // Background prefetch: the ~25 MiB encoder downloads while
                // the caller renders/collects input; failures are held and
                // surface on the first query, which awaits this promise.
                ensureEmbedder().catch(() => {});
            }
            embed = async (text) => {
                const { embedder, declaration } = await ensureEmbedder();
                const { vector } = await embedder.embed(`${declaration.prefixPolicy?.query || ''}${text}`);
                return { vector: toFloat32(vector, dim, 'inline transformer encoder'), text };
            };
            scoreQuality = retrievalScorer();
        } else if (qiKind === 4 && acceptLayer) {
            // Kind 4, `inherited-v1` (layered profile 4.5): a layer carries a
            // COMMITMENT to the base's encoder, never encoder bytes. This open
            // therefore installs no embedder — the chain reader resolves the
            // encoder from the base it opened and validates the commitment
            // itself (layer-reader.mjs), and a query on this member alone is
            // refused rather than answered with a missing encoder.
            embed = async () => {
                throw new Error('a layer inherits its encoder from the base: query the chain, not the layer (4.5)');
            };
            scoreQuality = null;
        } else {
            throw new Error(`unsupported query-interpretation kind ${qiKind}`);
        }

        // Index: the embedded sketch artifact, opened in the wave above
        // against a segment-windowed source (resident hash verified there).
        // The index binds to the corpus positionally (3.4) unless a row map
        // (3.9) says how its rows group into records; the manifest names the
        // policy that produced the rows, so a row map can neither be dropped
        // nor added without changing the identity.
        const rowMapPolicy = manifest.index?.rowMap ?? null;
        if (!!rowMapPolicy !== !!rowMapBytes) {
            throw new Error(rowMapPolicy
                ? '.pikelet manifest declares index.rowMap but carries no row map segment'
                : '.pikelet carries a row map segment its manifest does not declare (index.rowMap)');
        }
        const rowMap = rowMapBytes ? parseRowMapSegment(rowMapBytes, declaredRecords, sketch.count) : null;
        if (!rowMap && sketch.count !== declaredRecords) {
            throw new Error(`.pikelet index count ${sketch.count} != corpus records ${declaredRecords}`);
        }
        if (sketch.dim !== undefined && sketch.dim !== dim) {
            throw new Error(`.pikelet index dim ${sketch.dim} != manifest dim ${dim}`);
        }
        // The metric decides the search semantics; the sketch header's copy
        // must agree with the identity-verified manifest (on format 2 the
        // header itself is hash-anchored above; on format 1 this cross-check
        // is the only binding).
        if (METRIC_NAMES[sketch.metric] !== undefined && typeof manifest.metric === 'string'
            && METRIC_NAMES[sketch.metric] !== manifest.metric) {
            throw new Error(`.pikelet index metric ${METRIC_NAMES[sketch.metric]} != manifest metric ${manifest.metric}`);
        }
        if (verifyIndexVectors) {
            await sketch.verifyVectors();
        }

        // Resident-scan acceleration. options.sketchScanner:
        //   false      — pure JS scan always (the pre-0.7 behavior);
        //   an object  — a ready scanner (createSketchScanner's shape),
        //                caller-owned, used as-is;
        //   a function — async factory (sketchArtifact) => scanner, invoked
        //                in the background once the sketch is fully resident
        //                (the injection point for browser bundles, which
        //                cannot load the Node engine entrypoint);
        //   undefined  — auto: in Node, when the corpus is large enough
        //                that the JS scan dominates query time, stage the
        //                engine's SIMD scan kernel in the background.
        // Staging failures are silent by design: every query the scanner
        // would have served is answered identically (and identically
        // correctly) by the JS scan, just slower.
        const scannerOption = options.sketchScanner;
        if (scannerOption !== undefined && scannerOption !== false
            && typeof scannerOption !== 'object' && typeof scannerOption !== 'function') {
            throw new Error('options.sketchScanner must be false, a scanner object, or a factory function');
        }
        if (scannerOption && typeof scannerOption === 'object') {
            scanner = scannerOption;
        } else if (typeof scannerOption === 'function') {
            scannerPending = (async () => {
                await sketch.fullyResident;
                const created = await scannerOption(sketch);
                if (!created || typeof created.scan !== 'function') {
                    throw new Error('sketchScanner factory must resolve to an object with scan()');
                }
                disposeScanner = () => { if (typeof created.dispose === 'function') created.dispose(); };
                scanner = created;
            })();
            scannerPending.catch(() => {});
        } else if (scannerOption === undefined
            && sketch.count * (sketch.sketchDims || 0) >= SCANNER_AUTO_CELLS
            && globalThis.process?.versions?.node) {
            scannerPending = (async () => {
                // Hidden from bundlers like the node encoder kernels: the
                // web entrypoint's ?url wasm imports would break a browser
                // build that statically pulled this in.
                const { default: Pikelet } = await import(/* webpackIgnore: true */ /* @vite-ignore */ '../src/pikelet.node.mjs');
                await sketch.fullyResident;
                const created = await Pikelet.createSketchScanner(sketch, {
                    maxRerank: Math.min(sketch.count, SCANNER_MAX_RERANK),
                });
                disposeScanner = () => created.dispose();
                scanner = created;
            })();
            scannerPending.catch(() => {});
        }

        // Corpus tables: cross-check the segment's own words, then the
        // offsets — monotonic, starting exactly where the tables end, ending
        // inside the segment — and for layout v2 the page table's digest.
        const tablesView = viewOf(tables);
        const recordCount = tablesView.getUint32(0, true);
        if (recordCount !== declaredRecords) {
            throw new Error('.pikelet corpus count disagrees with manifest');
        }
        if (perRecord && tablesView.getUint32(4, true) !== pageRecords) {
            throw new Error('.pikelet corpus page size disagrees with manifest');
        }
        const recordOffsets = new Float64Array(recordCount + 1);
        for (let i = 0; i <= recordCount; i++) {
            const value = u64(tablesView, offsetsAt + 8 * i, `corpus offset ${i}`);
            if (value > corpus.length || (i > 0 && value < recordOffsets[i - 1]) || (i === 0 && value !== recordsAt)) {
                throw new Error('.pikelet corpus offsets are inconsistent');
            }
            recordOffsets[i] = value;
        }
        let pageTable = null;
        if (perRecord) {
            pageTable = tables.slice(offsetsAt + 8 * (recordCount + 1), tablesBytes);
            if (await sha256hex(pageTable) !== manifest.corpus.pageTableSha256) {
                throw new Error('.pikelet corpus page table failed hash verification');
            }
        }
        const corpusIntegrity = perRecord ? (verifyRecords ? 'per-record-sha256' : 'per-record-sha256 (unverified by option)') : 'segment-sha256';

        // Per-page record-digest cache: a page is fetched (one read of
        // 32 * pageRecords bytes), verified against the page table, and kept
        // for the records that share it. Pending fetches are shared so
        // concurrent hydrations of one page cost one read.
        const digestPages = new Map();
        const pageDigests = (page) => {
            if (digestPages.has(page)) {
                const cached = digestPages.get(page);
                digestPages.delete(page);
                digestPages.set(page, cached);
                return cached;
            }
            const pending = (async () => {
                const from = Math.min(recordCount, page * pageRecords);
                const to = Math.min(recordCount, (page + 1) * pageRecords);
                const bytes = await readChecked(source, corpus.offset + recordDigestsAt + 32 * from, 32 * (to - from),
                    `corpus digest page ${page}`, maxReadBytes, fileBytes);
                if (await sha256hex(bytes) !== toHex(pageTable.subarray(32 * page, 32 * page + 32))) {
                    digestPages.delete(page);
                    throw new Error(`.pikelet corpus digest page ${page} failed hash verification`);
                }
                return bytes;
            })();
            digestPages.set(page, pending);
            // A failed fetch (transport error, short read, hash mismatch) must
            // not stay cached as a rejected promise that every later record in
            // the page re-throws; evict it so the next hydration retries.
            pending.catch(() => { if (digestPages.get(page) === pending) digestPages.delete(page); });
            if (digestPages.size > DIGEST_PAGE_CACHE) digestPages.delete(digestPages.keys().next().value);
            return pending;
        };

        const recordCache = new Map();
        const cacheRecord = (id, record) => {
            if (recordCache.has(id)) recordCache.delete(id);
            recordCache.set(id, record);
            if (recordCache.size > RECORD_CACHE) recordCache.delete(recordCache.keys().next().value);
            return record;
        };

        // One record's stored bytes, read and (on format 2) verified against
        // its per-record digest. hydrate() parses these; `compact` (6.3)
        // copies them verbatim through the chain's __recordBytes seam.
        const readRecordBytes = async (id) => {
            if (!Number.isSafeInteger(id) || id < 0 || id >= recordCount) {
                throw new Error(`.pikelet result id ${id} is outside the corpus`);
            }
            const start = recordOffsets[id];
            const end = recordOffsets[id + 1];
            const bytes = await readChecked(source, corpus.offset + start, end - start, `corpus record ${id}`, maxRecordBytes, fileBytes);
            if (perRecord && verifyRecords) {
                const page = Math.floor(id / pageRecords);
                const digests = await pageDigests(page);
                const at = 32 * (id - page * pageRecords);
                if (await sha256hex(bytes) !== toHex(digests.subarray(at, at + 32))) {
                    throw new Error(`.pikelet corpus record ${id} failed integrity verification`);
                }
            }
            return bytes;
        };

        const hydrate = async (id) => {
            if (!Number.isSafeInteger(id) || id < 0 || id >= recordCount) {
                throw new Error(`.pikelet result id ${id} is outside the corpus`);
            }
            if (recordCache.has(id)) {
                const cached = recordCache.get(id);
                recordCache.delete(id);
                recordCache.set(id, cached);
                return cached;
            }
            const bytes = await readRecordBytes(id);
            let record;
            try { record = JSON.parse(decoder.decode(bytes)); } catch (err) {
                throw new Error(`.pikelet corpus record ${id} is not valid JSON`, { cause: err });
            }
            if (!record || typeof record !== 'object' || Array.isArray(record)) {
                throw new Error(`.pikelet corpus record ${id} is not a JSON object`);
            }
            return cacheRecord(id, record);
        };

        // Lifecycle: close() is idempotent (the encoder's WASM buffers and the
        // file descriptor are released once), and the query surface refuses
        // use after close instead of reading through a freed encoder.
        let closed = false;
        const assertOpen = () => {
            if (closed) throw new Error('.pikelet reader is closed');
        };

        // The rerank's default coalescing gap. The identity-verified manifest
        // may carry a fetch hint: cluster-ordered layouts only coalesce at a
        // gap at least matched to their cluster geometry (the wiki pack's
        // k-means layout needs ~16 KiB; a 2 KiB gap squanders it — measured
        // 440 vs 200 requests/query at C=600). The hint is a floor: a wider
        // gap merges a superset of the same runs, so a source that prefers
        // more (HTTP's 256 KiB) keeps its own. undefined leaves the sketch
        // reader on the source's preference. A chain reads it per tier
        // through __chainTier.
        const rerankGapHint = Number.isFinite(manifest.recommendedGap)
            ? Math.max(manifest.recommendedGap, Number(source.preferredGapBytes) || 0)
            : undefined;

        // The one retrieval path behind query(): resident sketch scan plus
        // BM25 candidates, exact rerank, rank fusion and phrase pinning.
        // Compile-time calibration fits on exactly this (through
        // __calibrationEvidence below), so the evidence a verdict was fit on
        // is the evidence query() scores. excludeIds removes records from
        // both candidate sources before either is cut to size — calibration
        // ablations use it; query() never does.
        const retrieveCandidates = async (trimmed, context, k, queryOptions, excludeIds = null) => {
            let fusedFull = null;
            // Hybrid retrieval when the artifact carries a lexical
            // segment: the BM25 top matches join the sketch's exact
            // rerank as extra candidates (scored by true vector
            // distance like any candidate — this is what recovers
            // known-item lookups the sketch scan's top-C missed),
            // and the final ordering fuses the distance ranking
            // with the BM25 ranking by reciprocal rank (RRF, k=60,
            // untuned). Abstention still scores the distance-sorted
            // top-k exactly as calibrated; fusion affects result
            // order only. fullRerankOutput returns every reranked
            // candidate (all already fetched and exactly scored) so
            // fusion sees the true distance of every lexical match.
            // Common query terms give every matching document a
            // small, near-tied BM25 score (idf collapses them);
            // that flat mass carries no ranking information and
            // dilutes rank fusion. Keep only hits within a factor
            // of the best lexical score — for a known-item lookup
            // that is typically the one document holding the rare
            // terms.
            // retrieval: 'hybrid' (default when the artifact
            // carries a lexical segment) | 'vector' | 'lexical' —
            // the last two exist for measurement and debugging
            // (retrieval-quality bakeoffs run all three on one
            // artifact). Abstention scores the distance-sorted
            // top-k identically in every mode.
            // 'augmented' unions the lexical candidates into the
            // exact rerank but keeps pure distance order — the mode
            // for workloads scored against exact nearest neighbors,
            // where RRF's reordering reads as loss by definition.
            const retrieval = queryOptions.retrieval ?? 'hybrid';
            if (!['hybrid', 'vector', 'lexical', 'augmented'].includes(retrieval)) {
                throw new Error(`query() retrieval must be hybrid, vector, lexical, or augmented, got ${retrieval}`);
            }
            // An excluded record must not take a candidate slot on either
            // side, so both sources skip it before cutting to size — BM25
            // before its top-n sort (scored with the full corpus's term
            // statistics, the one way this differs from a corpus without
            // those records) and the sketch before its top-C cut.
            const excluded = excludeIds && excludeIds.size ? excludeIds : null;
            let excludeBits = null;
            if (excluded) {
                excludeBits = new Uint8Array(Math.ceil(sketch.count / 8));
                for (const id of excluded) {
                    // With a row map, excluding a record masks every row it owns.
                    const [from, to] = rowMap ? [rowMap.offsets[id], rowMap.offsets[id + 1]] : [id, id + 1];
                    for (let row = from; row < to; row++) excludeBits[row >> 3] |= 1 << (row & 7);
                }
            }
            const lexicalRaw = lexicalIndex && retrieval !== 'vector'
                ? await lexicalIndex.search(trimmed, LEXICAL_CANDIDATES, excluded ? { exclude: (id) => excluded.has(id) } : {})
                : [];
            const lexicalHits = lexicalRaw.filter((h) => h.score >= lexicalRaw[0].score / LEXICAL_CUTOFF);
            // With a row map (3.9) the sketch searches rows: a record's lexical
            // match joins the rerank with every row it owns, k is scaled to the
            // mean rows per record, and the reranked rows collapse to records
            // below, each scored by its best row.
            const extraRows = rowMap
                ? lexicalHits.flatMap((h) => Array.from({ length: rowMap.offsets[h.id + 1] - rowMap.offsets[h.id] }, (_, i) => rowMap.offsets[h.id] + i))
                : lexicalHits.map((h) => h.id);
            const searchK = rowMap ? Math.min(sketch.count, Math.ceil(k * sketch.count / recordCount)) : k;
            const rerankedRows = (await sketch.search(context.vector, searchK, {
                rerank: queryOptions.rerank,
                parallelism: queryOptions.parallelism ?? queryOptions.rerankParallelism ?? options.rerankParallelism,
                gap: queryOptions.gap ?? queryOptions.rerankGap ?? options.rerankGap ?? rerankGapHint,
                maxRangeBytes: queryOptions.maxRangeBytes ?? options.rerankMaxRangeBytes,
                // Present once background staging finished; the
                // sketch reader ignores it for any query whose C
                // exceeds the scanner's buffers.
                ...(scanner ? { scanner } : {}),
                ...(extraRows.length ? { extraCandidates: extraRows } : {}),
                ...(excludeBits ? { exclude: excludeBits } : {}),
                // Always return the full reranked candidate list
                // (already fetched and exactly scored) rather than
                // truncating to k: abstention is fit at a fixed
                // top-10 window (calibrate.mjs) independent of the
                // caller's k, and needs that full window to score
                // consistently — see retrieval-abstention.mjs.
                fullRerankOutput: true,
            })).results;
            // The rerank output is distance-ordered, so a record's first row
            // is its best: keeping first occurrences gives the records in
            // order of their best row's distance.
            let searched = rerankedRows;
            if (rowMap) {
                const seen = new Set();
                searched = [];
                for (const row of rerankedRows) {
                    const id = rowMap.owner[row.id];
                    if (!seen.has(id)) { seen.add(id); searched.push({ id, distance: row.distance }); }
                }
            }
            const hits = searched.slice(0, k);
            // Full-window fused order (not k-truncated) for scoring:
            // coverage grounds its verdict in the passage fusion
            // actually ranks first, and calibrate.mjs fits it the
            // same way — a k-truncated fused list would silently
            // drop the fused top hit for callers requesting a small k.
            if (retrieval === 'lexical') {
                const byId = new Map(searched.map((hit) => [hit.id, hit]));
                fusedFull = lexicalHits.map((h) => byId.get(h.id)).filter(Boolean);
            } else if (retrieval !== 'augmented' && lexicalHits.length) {
                const lexRank = new Map(lexicalHits.map((h, i) => [h.id, i]));
                fusedFull = fuseCandidates(searched, lexicalHits.map((h) => h.id));
                // One batched hydration for everything this block and
                // scoreQuality's coverage term (below) will need, in
                // place of two separate staged fetches (phrase-pin's
                // own, then a later cold start for coverage): over a
                // remote range-read source each hydrate() a caller
                // hasn't already paid for is its own network round
                // trip, and sequential/staged awaits pay that cost
                // once per stage instead of once per query. The
                // union of phrase-pin's candidates (top lexical
                // hits) and coverage's (fusedFull's top few — phrase
                // pinning only ever reorders that window, never
                // changes its membership) is known right here, so
                // both stages can share one round trip. A record
                // that fails to hydrate here does not fail the
                // query: the id is just missing from the map, and
                // each site below falls through exactly as it would
                // on a cold cache miss (recs[i] undefined; coverage
                // sees fewer usable passages) — hydrate() raising
                // for a bad id is still a real failure, but only
                // once a site actually needs that specific record,
                // not speculatively for every candidate touched here.
                const phraseCandidates = lexicalHits.slice(0, 3);
                const coverageCandidates = fusedFull.slice(0, COVERAGE_BATCH_COUNT);
                const batchIds = [...new Set([...phraseCandidates, ...coverageCandidates].map((h) => h.id))];
                const batch = new Map();
                await Promise.all(batchIds.map(async (id) => {
                    try { batch.set(id, await hydrate(id)); } catch { /* see comment above */ }
                }));
                // Phrase pinning: a verbatim occurrence of the query (>= 3
                // content words) in one of the top lexical hits is the
                // strongest evidence of support there is; rank it first.
                const norm = (t) => String(t || '').toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim();
                const phrase = norm(trimmed);
                if (phrase.split(' ').length >= 3) {
                    for (const h of phraseCandidates) {
                        if (norm(batch.get(h.id)?.text).includes(phrase)) {
                            fusedFull = [searched.find((x) => x.id === h.id), ...fusedFull.filter((x) => x.id !== h.id)].filter(Boolean);
                            break;
                        }
                    }
                }
                // Rank transparency: a consumer can see a verbatim hit sitting
                // at fused rank 3 behind two semantic near-misses.
                for (const [i, hit] of fusedFull.entries()) { hit.fusedRank = i + 1; hit.lexicalRank = lexRank.has(hit.id) ? lexRank.get(hit.id) + 1 : null; hit.vectorRank = searched.findIndex((x) => x.id === hit.id) + 1; }
            }
            const fused = retrieval === 'augmented' ? null : fusedFull?.slice(0, k) ?? null;
            return { hits, fused, searched, fusedFull };
        };

        return {
            info() {
                return {
                    identity,
                    formatVersion,
                    profile,
                    // 6.3: present when this base was compacted from a chain,
                    // and the key to whether citation() can say anything. It
                    // was in the manifest all along and reported nowhere, so a
                    // compacted pack looked indistinguishable from a freshly
                    // compiled one.
                    compactedFrom: manifest.compactedFrom
                        && typeof manifest.compactedFrom === 'object'
                        && !Array.isArray(manifest.compactedFrom)
                        ? { ...manifest.compactedFrom }
                        : null,
                    // Human-readable pack name from the identity-verified
                    // manifest (compile records it under corpus.provenance);
                    // null when the builder recorded none.
                    name: typeof manifest.corpus?.provenance?.name === 'string'
                        ? manifest.corpus.provenance.name
                        : (typeof manifest.name === 'string' ? manifest.name : null),
                    // Content license recorded at compile (SPDX id or free
                    // text); null when the builder recorded none.
                    license: typeof manifest.corpus?.provenance?.license === 'string'
                        ? manifest.corpus.provenance.license : null,
                    records: recordCount,
                    // Index rows and the policy that produced them when the
                    // index holds several vectors per record (row map, 3.9);
                    // otherwise one row per record and rowMap is null.
                    indexRows: sketch.count,
                    rowMap: rowMapPolicy ? { ...rowMapPolicy } : null,
                    dim,
                    metric: manifest.metric,
                    encoder: encoderInfo,
                    encoderVerified,
                    corpusIntegrity,
                    fileBytes,
                    residentBytes: sketch.residentBytes + 8 * (recordCount + 1) + (pageTable ? pageTable.length : 0),
                    residentVerified: sketch.stats().residentVerified,
                    // True only after a full vectors pass (verifyIndexVectors
                    // at open, or verifyVectors() later).
                    vectorsVerified: sketch.vectorsVerified === true,
                    // 'per-row-sha256' when the embedded sketch is format 2
                    // (every rerank row verified on its own read against the
                    // identity-anchored page-hash table); 'segment-sha256'
                    // for format-1 sketches, where rows are committed but
                    // unverified until a full vectors pass.
                    indexRowIntegrity: sketch.formatVersion >= 2 && sketch.verifyRows !== false
                        ? 'per-row-sha256' : 'segment-sha256',
                    // 'engine' once a WASM scan kernel serves the resident
                    // scan (auto-staged or injected); 'js' before staging
                    // resolves and forever if it fails or was disabled.
                    residentScan: scanner ? 'engine' : 'js',
                    lexical: lexicalIndex
                        ? {
                            terms: lexicalIndex.termCount,
                            docCount: lexicalIndex.docCount,
                            // Lazy reads are bounded to the segment but not
                            // individually verified (LAZY_LEXICAL_BYTES).
                            integrity: lexicalIndex.lazy ? 'unverified-lazy-reads' : 'segment-sha256',
                            ...(lexicalIndex.lazy ? { lazy: true } : {}),
                        }
                        : null,
                    sampleQueries: Array.isArray(manifest.sampleQueries) ? manifest.sampleQueries : [],
                    // Where each part of the file lives, in file order: the
                    // header (fixed header, manifest and segment table, up to
                    // the first segment's aligned offset) and every segment of
                    // a known kind. Lets a viewer attribute range reads to
                    // what they fetched.
                    layout: [
                        { kind: 'header', offset: 0, length: align16(HEADER_BYTES + manifestBytes + tableBytes) },
                        ...[...segments].map(([kind, seg]) => ({ kind, offset: seg.offset, length: seg.length }))
                            .sort((a, b) => a.offset - b.offset),
                    ],
                };
            },

            async query(text, queryOptions = {}) {
                assertOpen();
                // k: absent means the default; anything supplied must be a
                // positive integer (an explicit 0, -1, or 2.5 is an error,
                // not a silent fallback) and is capped to the corpus size
                // before it sizes the search.
                let k = 5;
                if (queryOptions.k !== undefined) {
                    if (!Number.isSafeInteger(queryOptions.k) || queryOptions.k < 1) {
                        throw new Error('query() k must be a positive integer');
                    }
                    k = queryOptions.k;
                }
                k = Math.min(k, Math.max(recordCount, 1));
                const trimmed = String(text || '').trim();
                if (!trimmed) throw new Error('query text is required');
                const context = await embed(trimmed);
                const pre = preScore(context);
                let hits = [];
                let fused = null;
                let searched = null;
                let fusedFull = null;
                if (!pre) ({ hits, fused, searched, fusedFull } = await retrieveCandidates(trimmed, context, k, queryOptions));
                // Abstention scores the full reranked candidate window
                // (searched), not the k-truncated hits: calibrate.mjs fits
                // its signals at a fixed top-10 (top-5 for coverage)
                // regardless of how many results a caller ultimately
                // requests, so scoring must use the same window to match.
                // preScore (kind 1) runs before search and has no window to
                // fix, so it is unaffected. fusedFull (RRF over the same
                // full window, not k-truncated) is threaded through
                // separately so coverage grades the passage fusion actually
                // ranks first, while d0/margin/mean10 still read vector
                // order (searched) exactly as calibrated.
                const quality = pre || await scoreQuality(searched ?? hits, context, fusedFull);
                // A 'none' verdict withholds results by default — the
                // calibrated abstention signal is doing its job, and most
                // callers want that. showAbstained is an explicit opt-out
                // for a caller that wants the raw retrieval anyway (e.g. to
                // inspect why the classifier abstained); it never changes
                // matchQuality or confidence, only whether results ships.
                const returned = quality.match_quality === 'none' && !queryOptions.showAbstained
                    ? [] : (fused || hits);
                // The search's id and distance are authoritative: they are
                // written last so a corpus record carrying its own `id` or
                // `distance` field cannot overwrite them (reserved names).
                const results = await Promise.all(returned.map(async (hit) => {
                    const record = await hydrate(hit.id);
                    return { ...record, id: hit.id, distance: hit.distance, ...(hit.fusedRank ? { fusedRank: hit.fusedRank, lexicalRank: hit.lexicalRank, vectorRank: hit.vectorRank } : {}) };
                }));
                // Grounding transparency: which record's text grounded the verdict
                // and which query words it covered, so a consumer can tell a
                // topical match ("much money darcy make year") from an answer.
                const passageOrder = fusedFull?.length ? fusedFull : (searched ?? hits);
                const grounding = quality.grounding ? { ...quality.grounding, recordId: passageOrder[quality.grounding.passageIndex]?.id ?? null } : null;
                return { matchQuality: quality.match_quality, confidence: quality.confidence, results, ...(grounding ? { grounding } : {}) };
            },

            /**
             * The retrieval evidence query() scores, for compile-time
             * calibration: `vector` is the distance-ordered rerank window
             * d0/margin/mean10 read, `fused` the order coverage reads its
             * passages from. Calibration consumes this instead of
             * reimplementing retrieval, so a verdict is fit on the evidence
             * it is later scored on. Not part of the documented reader API.
             */
            async __calibrationEvidence(text, { excludeIds = null } = {}) {
                assertOpen();
                const trimmed = String(text || '').trim();
                if (!trimmed) throw new Error('query text is required');
                // The 10-wide window calibration fits at; any k up to the
                // pack's rerank depth gives query() the same candidates.
                const k = Math.min(10, Math.max(recordCount, 1));
                const { searched, fusedFull } = await retrieveCandidates(trimmed, await embed(trimmed), k, {}, excludeIds);
                const plain = (list) => list.map((hit) => ({ id: hit.id, distance: hit.distance }));
                return { vector: plain(searched), fused: plain(fusedFull?.length ? fusedFull : searched) };
            },

            /** Hydrate one corpus record by id (verified per record on format 2). */
            async record(id) {
                assertOpen();
                return hydrate(id);
            },

            /** The absolute byte range one corpus record's stored bytes occupy in the file. */
            recordRange(id) {
                assertOpen();
                if (!Number.isSafeInteger(id) || id < 0 || id >= recordCount) {
                    throw new Error(`.pikelet record id ${id} is outside the corpus`);
                }
                const start = recordOffsets[id];
                return { offset: corpus.offset + start, length: recordOffsets[id + 1] - start };
            },

            /**
             * Full pass over the index's lazy vector rows against the
             * identity-anchored vectorsSha256 (streamed where possible).
             * Until this resolves (or verifyIndexVectors ran at open), rows
             * that feed reranking are committed but not verified per read.
             */
            async verifyVectors(verifyOptions) {
                assertOpen();
                return sketch.verifyVectors(verifyOptions);
            },

            async evaluation() {
                assertOpen();
                return readEvaluationSegment();
            },

            /**
             * 6.3: translate a citation made against any head of the chain this
             * base was compacted from. Returns null when this pack carries no
             * lineage segment, i.e. it was not compacted from a chain.
             *
             * Forward translation only, and the shape says so: `{ok: true,
             * newId, via}` for a record that survives, `{ok: true,
             * deleted: true, ...}` for one that does not, and `{ok: false,
             * reason}` for an id or identity this history cannot speak to.
             * It answers what an old id BECAME, never what an old head served
             * — the old record's bytes are not in this file (see the spec's
             * "Translation is not dereference").
             */
            async citation(identity, id) {
                assertOpen();
                const lineage = await readLineageSegment();
                if (!lineage) return null;
                const { translateCitation } = await import('./layer-compact.mjs');
                // oldId -> new dense id, by the spec's own rule: a live id
                // loses one position per tombstone below it. Derived from the
                // segment's bitset rather than passed in, so a caller cannot
                // supply a mapping that disagrees with the bytes.
                return translateCitation(lineage, compactMapFor(lineage), identity, id);
            },

            /**
             * The parsed lineage segment, or null. Exposed so a consumer can
             * enumerate the heads a citation may name without guessing.
             */
            async lineage() {
                assertOpen();
                const parsed = await readLineageSegment();
                if (!parsed) return null;
                return {
                    heads: parsed.heads.map((h) => ({ ...h })),
                    oldRowTotal: parsed.oldRowTotal,
                    tombstoneCount: parsed.tombstoneCount,
                    supersessions: parsed.supersessions.map((e) => [...e]),
                };
            },

            /**
             * Internal seam for the layered profile (LAYERED_PROFILE.md 5.1
             * step 3): a chain reader mounts each member through this same
             * open, then needs each search tier's sketch, lexical index and
             * hydration to merge across tiers (5.3). Exposing them keeps one
             * container parser rather than a second one — decision 1.
             *
             * Deliberately not part of the documented reader API: the names
             * carry no compatibility promise and a caller outside this package
             * should use query()/record() instead.
             */
            get __chainTier() {
                assertOpen();
                return {
                    manifest,
                    identity,
                    recordCount,
                    dim,
                    sketch,
                    // A chain maps sketch rows to records positionally, so
                    // it needs to know when a member's rows are not records.
                    rowMap,
                    lexicalIndex,
                    hydrate,
                    readRecordBytes,
                    encoderInfo,
                    segments,
                    // The query-interpretation kind, verbatim from the
                    // segment header. `append` needs it to decide
                    // appendability (6.1), and reverse-mapping it from
                    // encoderInfo.kind loses kind 2, whose encoderInfo is
                    // shaped by the host declaration rather than a fixed name.
                    qiKind,
                    // This member's own evaluation segment, hash-verified.
                    // A chain surfaces the base's through it (6.2).
                    evaluation: readEvaluationSegment,
                    // The tier's staged WASM scan kernel, or null before
                    // background staging resolves. A chain queries
                    // sketch.search directly, so without passing this through
                    // every tier fell back to the JS scan: on the 456k-record
                    // wiki base that is ~450 ms a query against ~50 ms with
                    // the kernel. A getter, not a captured value, because
                    // staging completes after open() returns.
                    get scanner() { return scanner; },
                    // The rerank's default coalescing gap for this member
                    // (manifest hint as a floor under the source's own), so
                    // a chain fetches each tier's rows as this member would.
                    rerankGap: rerankGapHint,
                    // The base's OWN encoder, for `append` (6.1): "a kind-3
                    // base carries the teacher, and append loads it from the
                    // base's own query-interp segment — no model download, no
                    // external dependency: a pack is sufficient to compile its
                    // own successors." Resolving it here rather than letting a
                    // producer construct one is what makes substituting a
                    // different encoder impossible.
                    //
                    // Returns null for kinds that cannot embed passages: kind 1
                    // carries a query encoder only (6.1 refuses it), and kind 4
                    // is a layer, which inherits.
                    async passageEmbedder() {
                        assertOpen();
                        if (qiKind !== 3 || !resolveEmbedder) return null;
                        const { embedder, declaration } = await resolveEmbedder();
                        return {
                            declaration,
                            // The passage prefix is part of the contract the
                            // base was built with; applying the wrong one (or
                            // none) silently produces vectors in a different
                            // space from the base's.
                            embed: async (text) => {
                                const prefix = declaration.prefixPolicy?.passage || '';
                                const { vector } = await embedder.embed(`${prefix}${text}`);
                                return vector;
                            },
                            // A refit (6.3 step 4) embeds generated QUERIES,
                            // which take the query prefix, not the passage one:
                            // the asymmetry is the encoder's contract and
                            // applying the wrong side puts the fit's probes in
                            // a different space from the corpus it scores.
                            embedQuery: async (text) => {
                                const prefix = declaration.prefixPolicy?.query || '';
                                const { vector } = await embedder.embed(`${prefix}${text}`);
                                return vector;
                            },
                            // Word-level vectors for the comparison feature.
                            // Unprefixed by design -- a single word is not a
                            // query needing an instruction prefix. Null when
                            // the kernel cannot do it, which the caller reads
                            // as "fit without that feature".
                            embedWords: typeof embedder.embedWords === 'function'
                                ? (text) => embedder.embedWords(text)
                                : null,
                        };
                    },
                    // The base's own abstention scorer, for a chain serving
                    // under `inherited` (5.5): "the base's
                    // retrieval-signals-v1 scorer runs unchanged over the
                    // merged searched window and fused passages, with the union
                    // bloom of 4.5". Null when the base carries no fit, which
                    // is what makes a chain report `none` rather than silently
                    // omitting the verdict.
                    scoreQuality: typeof scoreQuality === 'function' ? scoreQuality : null,
                    // The base's fit, for a chain's calibration status (5.5).
                    // It lives in the query-interp segment's calibration
                    // region, not in the manifest — `asset` present means the
                    // base carries a retrieval-signals fit, and
                    // `asset.driftLimit` is the producer's declared validity
                    // envelope when it declared one.
                    calibrationJson,
                    // Raw segment bytes for the chain's structural parsing
                    // (tombstones, kind-4). Bounded and digest-checked exactly
                    // as any other segment read: the digest comes from the
                    // identity-verified manifest, so bytes that fail it never
                    // reach a parser.
                    async readSegmentBytes(seg) {
                        assertOpen();
                        const bytes = await readChecked(source, seg.offset, seg.length, 'chain segment', maxReadBytes, fileBytes);
                        if (await sha256hex(bytes) !== seg.sha256) {
                            throw new Error('.pikelet chain segment failed hash verification');
                        }
                        return bytes;
                    },
                };
            },

            async close() {
                if (closed) return;
                closed = true;
                await settleEncoder();
                disposeEncoder();
                await settleScanner();
                disposeScanner();
                await sketch.close();
                if (owned) await source.close();
            },
        };
    } catch (err) {
        await settleEncoder();
        try { disposeEncoder(); } catch { /* already released */ }
        await settleScanner();
        try { disposeScanner(); } catch { /* already released */ }
        if (owned && source.close) await source.close().catch(() => {});
        throw err;
    }
}
