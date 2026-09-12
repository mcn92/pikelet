'use strict';

const {
    PikeletError,
    PIKELET_ERROR_CODES,
    pikeletError,
} = require('./pikelet-errors.js');

// Envelope header for validated export/import
const PIKELET_MAGIC = 0x504E434B; // "PNCK"
const ENVELOPE_VERSION = 3;
const V1_ENVELOPE_HEADER_SIZE = 24; // magic(4) + version(4) + dim(4) + compressed(4) + metric(4) + quantized(4)
const V2_ENVELOPE_HEADER_SIZE = 20; // magic(4) + version(4) + dim(4) + metric(4) + quantized(4)
const V3_ENVELOPE_HEADER_SIZE = 32; // v2 + nextExtId(4) + mappingCount(4) + wasmSize(4)
const MAPPING_ENTRY_SIZE = 8; // intId(4) + extId(4)
const FLOAT_HNSW_MAGIC_V0 = 0x464C4857; // "FLHW"
const FLOAT_HNSW_MAGIC_V1 = 0x464C4831; // "FLH1"
const UINT8_HNSW_MAGIC_V0 = 0x49384857; // "I8HW"
const UINT8_HNSW_MAGIC_V1 = 0x49384831; // "I8H1"
const MAX_EF = 4096;
// The wasm32 heap tops out at 2 GiB (no MAXIMUM_MEMORY override, no
// MEMORY64). The index arena gets 1.5 GiB of that; the rest is headroom for
// the 16 MiB initial heap, query scratch, export buffers, and envelope
// copies during import.
const MAX_INDEX_ARENA_BYTES = 1536 * 1024 * 1024;
const DEFAULT_MAX_ELEMENTS = 100000;
const DEFAULT_M = 12;
const DEFAULT_EF_CONSTRUCTION = 75;
const DEFAULT_EF_SEARCH = 100;
const DEFAULT_SEED = 108;

function validateEfSearch(value, label = 'efSearch') {
    if (!Number.isInteger(value) || value < 1 || value > MAX_EF) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
            `${label} must be an integer between 1 and ${MAX_EF}`,
            { argument: label, value, min: 1, max: MAX_EF });
    }
    return value;
}

function resolveSearchEf(options, fallback, methodName) {
    if (options === undefined) return fallback;
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
            `${methodName}() options must be an object`, { argument: 'options' });
    }
    return options.efSearch === undefined
        ? fallback
        : validateEfSearch(options.efSearch, `${methodName}() options.efSearch`);
}

// Reject non-numeric elements in a plain-array vector input. A Float32Array
// already guarantees numeric storage, but `new Float32Array([...])` silently
// coerces a plain array's elements via Number() — so '1'->1, ''->0, true->1,
// [2]->2 would be accepted as valid coordinates. That turns malformed input
// (e.g. an empty CSV field) into a silently-wrong vector instead of a loud
// error. Require real numbers so callers fail closed on bad data. Float32Array
// inputs skip this (already numeric). NaN/Infinity are still caught downstream.
function assertNumericVector(vec, label) {
    if (vec instanceof Float32Array) return;
    if (vec == null || typeof vec.length !== 'number') return; // length check handles these
    for (let i = 0; i < vec.length; i++) {
        if (typeof vec[i] !== 'number') {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR,
                `${label} must contain only numbers; found ${typeof vec[i]} at index ${i}`,
                { index: i, actualType: typeof vec[i] });
        }
    }
}

function validateVectorValues(f32, label, validateCosineNorm) {
    let normSq = 0;
    for (let i = 0; i < f32.length; i++) {
        const value = f32[i];
        if (!Number.isFinite(value)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR,
                `${label} contains non-finite value (NaN or Infinity)`, { reason: 'non_finite' });
        }
        if (validateCosineNorm) normSq += value * value;
    }
    if (validateCosineNorm && (!(normSq > 0) || !Number.isFinite(normSq))) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR,
            `${label} has invalid cosine norm`, { reason: 'invalid_cosine_norm' });
    }
}

class PikeletIndex {
    constructor(engine, opts, handle, vecPtr, idPtr, distPtr, bufferCapacity) {
        this._e = engine;
        this._dim = opts.dim;
        this._maxElements = opts.maxElements;
        this._quantized = !!opts.quantized;
        this._isL2 = (opts.metric === 'l2');
        this._M = opts.M;
        this._efConstruction = opts.efConstruction;
        this._efSearch = opts.efSearch;
        this._seed = opts.seed;
        this._handle = handle;
        this._vecPtr = vecPtr;
        this._idPtr = idPtr;
        this._distPtr = distPtr;
        this._bufferCapacity = bufferCapacity;
        this._snapshotBufferBytes = 0;
        this._disposed = false;

        // ID translation layer -- WASM compact() reassigns sequential IDs,
        // so we maintain a stable external ID space for the consumer.
        this._nextExtId = 0;
        this._intToExt = new Map(); // internal (WASM) id -> external (user) id
        this._extToInt = new Map(); // external (user) id -> internal (WASM) id
        this._deletedExt = new Set(); // external ids that have been soft-deleted
    }

    add(vec) {
        this._checkDisposed();
        assertNumericVector(vec, 'Vector');
        const f32 = vec instanceof Float32Array ? vec : new Float32Array(vec);
        if (f32.length !== this._dim) {
            throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
                `Expected vector of length ${this._dim}, got ${f32.length}`,
                { expected: this._dim, actual: f32.length });
        }
        validateVectorValues(f32, 'Vector', !this._isL2);
        // The v3 snapshot stores every external id and the next-id counter as
        // u32. Assigning past that boundary would produce ids the snapshot
        // cannot represent (export would truncate, restore would reject), so
        // the limit is enforced before the engine insert, leaving the index
        // unmutated. Ids survive deletes, so a long-lived high-churn index
        // can reach this without ever holding 2^32 vectors at once.
        if (this._nextExtId >= 0xFFFFFFFF) {
            throw pikeletError(PIKELET_ERROR_CODES.INDEX_LIMIT,
                'Stable id space exhausted: ids are serialized as uint32 and the next id would not fit',
                { nextExtId: this._nextExtId });
        }
        this._e.HEAPF32.set(f32, this._vecPtr >> 2);
        const intId = this._e._pikelet_add(this._handle, this._vecPtr);
        if (intId === 0xFFFFFFFF || intId < 0) {
            throw pikeletError(PIKELET_ERROR_CODES.INDEX_FULL,
                'Insert failed (index full or not initialized)');
        }

        const extId = this._nextExtId++;
        this._intToExt.set(intId, extId);
        this._extToInt.set(extId, intId);
        return extId;
    }

    addBatch(vectors) {
        this._checkDisposed();
        if (!Array.isArray(vectors)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'addBatch() requires an array of vectors');
        }
        if (vectors.length === 0) return [];
        if (this.count + vectors.length > this._maxElements) {
            throw pikeletError(PIKELET_ERROR_CODES.INDEX_FULL,
                'Insert failed (index full or not initialized)');
        }
        // Same u32 stable-id bound as add(), preflighted for the whole batch
        // so the failure happens before any row is inserted.
        if (this._nextExtId + vectors.length > 0xFFFFFFFF) {
            throw pikeletError(PIKELET_ERROR_CODES.INDEX_LIMIT,
                'Stable id space exhausted: ids are serialized as uint32 and this batch would run past the last representable id',
                { nextExtId: this._nextExtId, batch: vectors.length });
        }

        // Validate ALL vectors first before mutating the index, so a bad
        // input mid-batch can't leave the index in a partially-inserted state.
        // Keep the converted Float32Arrays so the write pass below doesn't
        // convert plain arrays a second time.
        const converted = new Array(vectors.length);
        for (let i = 0; i < vectors.length; i++) {
            const v = vectors[i];
            const len = (v instanceof Float32Array) ? v.length : (v && v.length);
            if (len !== this._dim) {
                throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
                    `Expected vector of length ${this._dim}, got ${len} at index ${i}`,
                    { expected: this._dim, actual: len, index: i });
            }
            assertNumericVector(v, `Vector at index ${i}`);
            const f32 = v instanceof Float32Array ? v : new Float32Array(v);
            validateVectorValues(f32, `Vector at index ${i}`, !this._isL2);
            converted[i] = f32;
        }

        // Allocate the WASM buffer once, write directly into the heap (no
        // intermediate JS Float32Array), call bulk_insert once.
        const totalFloats = vectors.length * this._dim;
        const dataPtr = this._e._emsc_malloc(totalFloats * 4);
        if (!dataPtr) throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED,
            'WASM malloc failed for bulk insert');

        try {
            const heapOffset = dataPtr >> 2;
            for (let i = 0; i < vectors.length; i++) {
                this._e.HEAPF32.set(converted[i], heapOffset + i * this._dim);
            }
            const countBefore = this.count;
            const inserted = this._e._pikelet_bulk_insert(this._handle, dataPtr, vectors.length);
            const ids = this._recordInsertedRange(countBefore, inserted);
            if (inserted !== vectors.length) {
                throw pikeletError(PIKELET_ERROR_CODES.INTERNAL_INVARIANT,
                    'bulk_insert inserted fewer vectors than the prevalidated batch',
                    { requested: vectors.length, inserted, ids });
            }
            return ids;
        } finally {
            this._e._emsc_free(dataPtr);
        }
    }

    search(query, k, options) {
        this._checkDisposed();
        if (!Number.isSafeInteger(k) || k < 0) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'search() requires a non-negative integer k', { argument: 'k', value: k });
        }
        assertNumericVector(query, 'Query vector');
        const f32 = query instanceof Float32Array ? query : new Float32Array(query);
        if (f32.length !== this._dim) {
            throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
                `Expected query of length ${this._dim}, got ${f32.length}`,
                { expected: this._dim, actual: f32.length });
        }
        validateVectorValues(f32, 'Query vector', !this._isL2);
        const efSearch = resolveSearchEf(options, this._efSearch, 'search');
        // The C ABI takes a signed 32-bit k and the WASM allocations take
        // 32-bit byte sizes. Passing a larger JS integer through either boundary
        // can wrap and allocate undersized output buffers. An index cannot
        // return more than count results, so cap k before allocating or calling
        // into WASM while preserving the documented k > count behavior.
        const boundedK = Math.min(k, this.count);
        if (boundedK === 0) return [];
        this._ensureSearchCapacity(boundedK);
        this._e.HEAPF32.set(f32, this._vecPtr >> 2);
        const found = this._e._pikelet_query(
            this._handle, this._vecPtr, boundedK, efSearch, this._idPtr, this._distPtr
        );
        return this._readResults(found);
    }

    searchFiltered(query, k, allowedIds, options) {
        this._checkDisposed();
        if (!Number.isSafeInteger(k) || k < 0) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'searchFiltered() requires a non-negative integer k', { argument: 'k', value: k });
        }
        assertNumericVector(query, 'Query vector');
        const f32 = query instanceof Float32Array ? query : new Float32Array(query);
        if (f32.length !== this._dim) {
            throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
                `Expected query of length ${this._dim}, got ${f32.length}`,
                { expected: this._dim, actual: f32.length });
        }
        validateVectorValues(f32, 'Query vector', !this._isL2);
        const efSearch = resolveSearchEf(options, this._efSearch, 'searchFiltered');
        if (!(allowedIds instanceof Set)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'searchFiltered() requires allowedIds to be a Set<number>', { argument: 'allowedIds' });
        }
        const boundedK = Math.min(k, this.count);
        if (boundedK === 0 || allowedIds.size === 0) return [];
        this._ensureSearchCapacity(boundedK);

        // Build bitset over internal IDs
        const count = this._e._pikelet_count(this._handle);
        const bitsetLen = (count + 7) >> 3;
        const bitsetPtr = this._e._emsc_malloc(bitsetLen);
        if (!bitsetPtr) throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED,
            'WASM malloc failed for filter bitset');

        try {
            // Zero the bitset
            this._e.HEAPU8.fill(0, bitsetPtr, bitsetPtr + bitsetLen);

            // Set bits for allowed internal IDs
            for (const extId of allowedIds) {
                const intId = this._extToInt.get(extId);
                if (intId !== undefined && !this._deletedExt.has(extId)) {
                    this._e.HEAPU8[bitsetPtr + (intId >> 3)] |= (1 << (intId & 7));
                }
            }

            this._e.HEAPF32.set(f32, this._vecPtr >> 2);
            const found = this._e._pikelet_query_filtered(
                this._handle, this._vecPtr, boundedK, efSearch,
                this._idPtr, this._distPtr,
                bitsetPtr, bitsetLen
            );
            return this._readResults(found);
        } finally {
            this._e._emsc_free(bitsetPtr);
        }
    }

    delete(id) {
        this._checkDisposed();
        const intId = this._extToInt.get(id);
        if (intId === undefined || this._deletedExt.has(id)) return false;
        this._e._pikelet_delete(this._handle, intId);
        this._deletedExt.add(id);
        return true;
    }

    has(id) {
        this._checkDisposed();
        return this._extToInt.has(id) && !this._deletedExt.has(id);
    }

    isDeleted(id) {
        this._checkDisposed();
        return this._deletedExt.has(id);
    }

    compact() {
        this._checkDisposed();

        // Use the engine's own old->new remap rather than re-deriving it in
        // JS: how compaction assigns new IDs is the engine's contract, and
        // re-implementing it here would silently corrupt the external-ID
        // translation if the engine's assignment order ever changed.
        const countBefore = this._e._pikelet_count(this._handle);
        if (countBefore === 0) {
            this._e._pikelet_compact(this._handle);
            return;
        }

        const mapPtr = this._e._emsc_malloc(countBefore * 4);
        if (!mapPtr) throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED,
            'WASM malloc failed for compact remap');

        try {
            const written = this._e._pikelet_compact_remap(this._handle, mapPtr, countBefore);
            const base = mapPtr >> 2;
            const remapped = [];
            for (let oldInt = 0; oldInt < written; oldInt++) {
                const newInt = this._e.HEAPU32[base + oldInt];
                if (newInt === 0xFFFFFFFF) continue; // deleted by compaction
                const extId = this._intToExt.get(oldInt);
                if (extId === undefined) continue;
                remapped.push([newInt, extId]);
            }

            this._intToExt.clear();
            this._extToInt.clear();
            this._deletedExt.clear();
            for (const [newInt, extId] of remapped) {
                this._intToExt.set(newInt, extId);
                this._extToInt.set(extId, newInt);
            }

            const liveCount = this._e._pikelet_count(this._handle);
            if (this._intToExt.size !== liveCount) {
                this._clearMappings();
                throw pikeletError(PIKELET_ERROR_CODES.INTERNAL_INVARIANT,
                    `compact() remap mismatch: engine reports ${liveCount} live vectors, remap yielded ${this._intToExt.size}`,
                    { liveCount, mappingCount: this._intToExt.size });
            }
        } finally {
            this._e._emsc_free(mapPtr);
        }
    }

    export() {
        this._checkDisposed();
        if (this.ghostCount > 0) {
            throw pikeletError(PIKELET_ERROR_CODES.COMPACTION_REQUIRED,
                'Export failed: compact() required before export when ghostCount > 0',
                { deletedCount: this.ghostCount });
        }

        const sizePtr = this._e._emsc_malloc(4);
        if (!sizePtr) throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED,
            'WASM malloc failed for export');
        try {
            const dataPtr = this._e._pikelet_export(this._handle, sizePtr);
            if (!dataPtr) {
                throw pikeletError(PIKELET_ERROR_CODES.INTERNAL_INVARIANT, 'Export failed');
            }

            const wasmSize = this._e.HEAPU32[sizePtr >> 2];
            this._snapshotBufferBytes = wasmSize;
            const liveMappings = Array.from(this._intToExt.entries()).sort((a, b) => a[0] - b[0]);
            const mappingBytes = liveMappings.length * MAPPING_ENTRY_SIZE;
            const result = new Uint8Array(V3_ENVELOPE_HEADER_SIZE + mappingBytes + wasmSize);
            const view = new DataView(result.buffer);
            view.setUint32(0, PIKELET_MAGIC, true);
            view.setUint32(4, ENVELOPE_VERSION, true);
            view.setUint32(8, this._dim, true);
            view.setUint32(12, this._isL2 ? 0 : 1, true);
            view.setUint32(16, this._quantized ? 1 : 0, true);
            view.setUint32(20, this._nextExtId, true);
            view.setUint32(24, liveMappings.length, true);
            view.setUint32(28, wasmSize, true);

            let offset = V3_ENVELOPE_HEADER_SIZE;
            for (const [intId, extId] of liveMappings) {
                view.setUint32(offset, intId, true);
                view.setUint32(offset + 4, extId, true);
                offset += MAPPING_ENTRY_SIZE;
            }

            result.set(this._e.HEAPU8.subarray(dataPtr, dataPtr + wasmSize), offset);
            return result;
        } finally {
            this._e._emsc_free(sizePtr);
        }
    }

    import(data) {
        this._checkDisposed();
        const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);

        let wasmBytes = bytes;
        let pendingMappings = null;
        let pendingNextExtId = null;

        if (bytes.length >= V2_ENVELOPE_HEADER_SIZE) {
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

            if (view.getUint32(0, true) === PIKELET_MAGIC) {
                const version = view.getUint32(4, true);

                let dim;
                let metricVal;
                let quantizedVal;
                let nextExtId = null;
                let mappingCount = null;
                let mappingOffset = null;
                let wasmOffset = null;
                let wasmSize = null;

                if (version === 1) {
                    if (bytes.length < V1_ENVELOPE_HEADER_SIZE) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Import failed: truncated v1 envelope');
                    }

                    dim = view.getUint32(8, true);
                    const compressed = view.getUint32(12, true);
                    if (compressed !== dim) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                            'Import failed: DCT/PCA indexes are no longer supported');
                    }

                    metricVal = view.getUint32(16, true);
                    quantizedVal = view.getUint32(20, true);
                } else if (version === ENVELOPE_VERSION) {
                    if (bytes.length < V3_ENVELOPE_HEADER_SIZE) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Import failed: truncated v3 envelope');
                    }

                    dim = view.getUint32(8, true);
                    metricVal = view.getUint32(12, true);
                    quantizedVal = view.getUint32(16, true);
                    nextExtId = view.getUint32(20, true);
                    mappingCount = view.getUint32(24, true);
                    wasmSize = view.getUint32(28, true);
                    mappingOffset = V3_ENVELOPE_HEADER_SIZE;
                    wasmOffset = mappingOffset + mappingCount * MAPPING_ENTRY_SIZE;

                    if (wasmOffset > bytes.length || wasmOffset + wasmSize > bytes.length) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                            'Import failed: truncated v3 envelope payload');
                    }
                } else {
                    if (version !== 2) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                            `Import failed: unsupported envelope version ${version}`, { version });
                    }

                    dim = view.getUint32(8, true);
                    metricVal = view.getUint32(12, true);
                    quantizedVal = view.getUint32(16, true);
                }

                if (dim !== this._dim) {
                    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                        `Import failed: dim mismatch (exported ${dim}, expected ${this._dim})`,
                        { field: 'dim', exported: dim, expected: this._dim });
                }

                const exportedL2 = metricVal === 0;
                if (exportedL2 !== this._isL2) {
                    const got = exportedL2 ? 'l2' : 'cosine';
                    const want = this._isL2 ? 'l2' : 'cosine';
                    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                        `Import failed: metric mismatch (exported ${got}, expected ${want})`,
                        { field: 'metric', exported: got, expected: want });
                }

                if (!!quantizedVal !== this._quantized) {
                    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                        'Import failed: quantized mismatch',
                        { field: 'quantized', exported: !!quantizedVal, expected: this._quantized });
                }

                if (version === ENVELOPE_VERSION) {
                    wasmBytes = bytes.subarray(wasmOffset, wasmOffset + wasmSize);
                    const metadata = parseRawSnapshotMetadata(wasmBytes);
                    if (metadata === null) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                            'Import failed: unsupported raw snapshot format');
                    }
                    this._validateRawSnapshotMetadata(metadata);
                    if (mappingCount !== metadata.count) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                            'Import failed: envelope mapping count mismatch');
                    }
                    pendingMappings = new Map();
                    let offset = mappingOffset;
                    for (let i = 0; i < mappingCount; i++) {
                        const intId = view.getUint32(offset, true);
                        const extId = view.getUint32(offset + 4, true);
                        pendingMappings.set(intId, extId);
                        offset += MAPPING_ENTRY_SIZE;
                    }
                    this._validateV3Mappings(pendingMappings, nextExtId, metadata.count);
                    pendingNextExtId = nextExtId;
                } else {
                    // The v1 header carries an extra `compressed` field, so its
                    // payload starts 4 bytes later than v2's.
                    wasmBytes = bytes.subarray(
                        version === 1 ? V1_ENVELOPE_HEADER_SIZE : V2_ENVELOPE_HEADER_SIZE
                    );
                }
            }
        }

        // Validate the embedded raw snapshot even when an envelope is present.
        // The engine metric is restored from the raw header, so trusting only
        // the envelope would allow the public wrapper and backend to disagree
        // about both query behavior and distance interpretation.
        const rawMetadata = parseRawSnapshotMetadata(wasmBytes);
        if (rawMetadata === null) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'Import failed: unsupported raw snapshot format');
        }
        this._validateRawSnapshotMetadata(rawMetadata);

        const dataPtr = this._e._emsc_malloc(wasmBytes.length);
        if (!dataPtr) throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED,
            'WASM malloc failed for import');

        let status;
        try {
            this._e.HEAPU8.set(wasmBytes, dataPtr);
            status = this._e._pikelet_import(this._handle, dataPtr, wasmBytes.length);
        } finally {
            this._e._emsc_free(dataPtr);
        }

        if (status !== 0) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Import failed');
        }

        const count = this._e._pikelet_count(this._handle);
        if (pendingMappings === null) {
            this._setIdentityMappings(count);
            return;
        }

        try {
            this._commitMappings(pendingMappings, pendingNextExtId);
        } catch (err) {
            this._clearMappings();
            throw err;
        }
    }

    get count() {
        this._checkDisposed();
        return this._e._pikelet_count(this._handle);
    }

    get ghostCount() {
        this._checkDisposed();
        return this._e._pikelet_ghost_count(this._handle);
    }

    get ghostRatio() {
        this._checkDisposed();
        return this._e._pikelet_ghost_ratio(this._handle);
    }

    get memory() {
        this._checkDisposed();
        return this._e._pikelet_memory(this._handle);
    }

    get dim() { return this._dim; }

    get liveCount() {
        return this.count - this.deletedCount;
    }

    get deletedCount() {
        return this.ghostCount;
    }

    get deletedRatio() {
        return this.ghostRatio;
    }

    get capacity() {
        this._checkDisposed();
        return this._maxElements;
    }

    get remainingCapacity() {
        return this.capacity - this.count;
    }

    get config() {
        this._checkDisposed();
        return Object.freeze({
            dim: this._dim,
            maxElements: this._maxElements,
            metric: this._isL2 ? 'l2' : 'cosine',
            quantized: this._quantized,
            M: this._M,
            efConstruction: this._efConstruction,
            efSearch: this._efSearch,
            seed: this._seed,
        });
    }

    get memoryUsage() {
        this._checkDisposed();
        return Object.freeze({
            logicalIndexBytes: this._e._pikelet_memory(this._handle),
            wasmHeapBytes: this._e.HEAPU8.buffer.byteLength,
            snapshotBufferBytes: this._snapshotBufferBytes,
        });
    }

    dispose() {
        if (this._disposed) return;
        let thrown = null;
        try {
            this._e._pikelet_dispose(this._handle);
        } catch (err) {
            thrown = err;
        } finally {
            for (const ptr of [this._vecPtr, this._idPtr, this._distPtr]) {
                try {
                    this._e._emsc_free(ptr);
                } catch (err) {
                    if (thrown === null) thrown = err;
                }
            }
            this._disposed = true;
        }
        if (thrown !== null) throw thrown;
    }

    setEfSearch(ef) {
        this._checkDisposed();
        this._efSearch = validateEfSearch(ef, 'setEfSearch() efSearch');
    }

    _checkDisposed() {
        if (this._disposed) {
            throw pikeletError(
                PIKELET_ERROR_CODES.INDEX_DISPOSED,
                'PikeletIndex has been disposed'
            );
        }
    }

    _ensureSearchCapacity(k) {
        if (k <= this._bufferCapacity) return;

        const newIdPtr = this._e._emsc_malloc(k * 8);
        const newDistPtr = this._e._emsc_malloc(k * 4);
        if (!newIdPtr || !newDistPtr) {
            if (newIdPtr) this._e._emsc_free(newIdPtr);
            if (newDistPtr) this._e._emsc_free(newDistPtr);
            throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED,
                `WASM malloc failed while growing search buffers to k=${k}`, { k });
        }

        this._e._emsc_free(this._idPtr);
        this._e._emsc_free(this._distPtr);
        this._idPtr = newIdPtr;
        this._distPtr = newDistPtr;
        this._bufferCapacity = k;
    }

    _recordInsertedRange(firstIntId, count) {
        const ids = new Array(count);
        for (let i = 0; i < count; i++) {
            const intId = firstIntId + i;
            const extId = this._nextExtId++;
            this._intToExt.set(intId, extId);
            this._extToInt.set(extId, intId);
            ids[i] = extId;
        }
        return ids;
    }

    _clearMappings() {
        this._intToExt.clear();
        this._extToInt.clear();
        this._deletedExt.clear();
        this._nextExtId = 0;
    }

    _commitMappings(intToExt, nextExtId) {
        this._clearMappings();
        for (const [intId, extId] of intToExt) {
            this._intToExt.set(intId, extId);
            this._extToInt.set(extId, intId);
        }
        this._nextExtId = nextExtId;
    }

    _validateV3Mappings(intToExt, nextExtId, count) {
        if (intToExt.size !== count) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'Import failed: envelope mapping count mismatch');
        }
        if (!Number.isInteger(nextExtId) || nextExtId < count) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'Import failed: envelope nextExtId is invalid');
        }

        const extIds = new Set();
        let maxExtId = -1;
        for (const [intId, extId] of intToExt) {
            if (!Number.isInteger(intId) || intId < 0 || intId >= count) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    'Import failed: envelope mapping contains invalid internal ID');
            }
            if (!Number.isInteger(extId) || extId < 0) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    'Import failed: envelope mapping contains invalid external ID');
            }
            if (extIds.has(extId)) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    'Import failed: envelope mapping contains duplicates');
            }
            extIds.add(extId);
            if (extId > maxExtId) maxExtId = extId;
        }
        if (nextExtId <= maxExtId) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'Import failed: envelope nextExtId is invalid');
        }
    }

    _setIdentityMappings(count) {
        this._clearMappings();
        for (let i = 0; i < count; i++) {
            this._intToExt.set(i, i);
            this._extToInt.set(i, i);
        }
        this._nextExtId = count;
    }

    _validateRawSnapshotMetadata(metadata) {
        // Unsupported major format versions must be rejected, not parsed as
        // the newest known layout (Search Artifact Contract §6). Current
        // writers emit v2 (quantized) / v1 (float); a future v3 would
        // otherwise be silently misparsed as v2 by the open-ended version
        // reads in the native deserializers.
        const maxVersion = metadata.quantized ? 2 : 1;
        if (!Number.isInteger(metadata.version) || metadata.version > maxVersion) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                `Import failed: unsupported snapshot format version ${metadata.version}`,
                { version: metadata.version, maxSupported: maxVersion });
        }
        if (!Number.isInteger(metadata.count) || metadata.count < 0) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'Import failed: snapshot count is invalid');
        }
        if (metadata.count > this._maxElements) {
            throw pikeletError(
                PIKELET_ERROR_CODES.SNAPSHOT_CAPACITY_EXCEEDED,
                `Import failed: snapshot count ${metadata.count} exceeds maxElements ${this._maxElements}`,
                { count: metadata.count, maxElements: this._maxElements }
            );
        }
        if (metadata.dim !== this._dim) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                `Import failed: dim mismatch (exported ${metadata.dim}, expected ${this._dim})`,
                { field: 'dim', exported: metadata.dim, expected: this._dim });
        }

        const exportedL2 = metadata.metric === 0;
        if (exportedL2 !== this._isL2) {
            const got = exportedL2 ? 'l2' : 'cosine';
            const want = this._isL2 ? 'l2' : 'cosine';
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                `Import failed: metric mismatch (exported ${got}, expected ${want})`,
                { field: 'metric', exported: got, expected: want });
        }

        if (metadata.quantized !== this._quantized) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                'Import failed: quantized mismatch',
                { field: 'quantized', exported: metadata.quantized, expected: this._quantized });
        }
        if (metadata.M !== this._M) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                `Import failed: M mismatch (exported ${metadata.M}, expected ${this._M})`,
                { field: 'M', exported: metadata.M, expected: this._M });
        }
        if (metadata.efConstruction !== this._efConstruction) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                `Import failed: efConstruction mismatch (exported ${metadata.efConstruction}, expected ${this._efConstruction})`,
                { field: 'efConstruction', exported: metadata.efConstruction, expected: this._efConstruction });
        }
    }

    _readResults(n) {
        const dv = new DataView(this._e.HEAPU8.buffer);
        const results = new Array(n);
        for (let i = 0; i < n; i++) {
            const lo = dv.getUint32(this._idPtr + i * 8, true);
            const hi = dv.getUint32(this._idPtr + i * 8 + 4, true);
            const intId = hi * 0x100000000 + lo;
            let distance = this._e.HEAPF32[(this._distPtr >> 2) + i];
            if (this._isL2) distance = Math.sqrt(distance);
            const extId = this._intToExt.get(intId);
            if (extId === undefined) {
                throw pikeletError(
                    PIKELET_ERROR_CODES.INTERNAL_INVARIANT,
                    `Search invariant failed: missing external ID mapping for internal ID ${intId}`,
                    { internalId: intId }
                );
            }
            results[i] = { id: extId, distance };
        }
        return results;
    }
}

function createPikeletApi(loadEngineImpl) {
    function isVectorInput(value) {
        return value instanceof Float32Array || Array.isArray(value);
    }

    function extractVectorRecord(item) {
        if (isVectorInput(item)) {
            return { vector: item, hasSourceId: false, sourceId: undefined };
        }
        if (!item || typeof item !== 'object' || !('vector' in item)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'fromVectors() requires vectors or { vector, id? } records');
        }
        const sourceId = item.id;
        const hasSourceId = Object.prototype.hasOwnProperty.call(item, 'id');
        return { vector: item.vector, hasSourceId, sourceId };
    }

    async function create(opts) {
        if (!opts || !opts.dim) throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
            'opts.dim is required', { argument: 'dim' });
        if (!Number.isInteger(opts.dim) || opts.dim <= 0) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'opts.dim must be a positive integer', { argument: 'dim', value: opts.dim });
        }
        if ('compressed' in opts) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'opts.compressed has been removed');
        }
        if ('varianceSample' in opts) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'opts.varianceSample has been removed');
        }
        if (opts.metric !== undefined && opts.metric !== 'cosine' && opts.metric !== 'l2') {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                "opts.metric must be 'cosine' or 'l2'", { argument: 'metric', value: opts.metric });
        }
        // Upper bound matches the engine ABI: pancake_init takes max_elem as a
        // C int, so anything above 2^31-1 would truncate or go negative.
        if (opts.maxElements !== undefined && (!Number.isInteger(opts.maxElements) || opts.maxElements <= 0 || opts.maxElements > 0x7fffffff)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'opts.maxElements must be an integer between 1 and 2147483647', { argument: 'maxElements', value: opts.maxElements });
        }
        if (opts.M !== undefined && (!Number.isInteger(opts.M) || opts.M <= 1 || opts.M > 128)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'opts.M must be an integer between 2 and 128', { argument: 'M', value: opts.M });
        }
        if (opts.efConstruction !== undefined && (!Number.isInteger(opts.efConstruction) || opts.efConstruction <= 0 || opts.efConstruction > 4096)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'opts.efConstruction must be an integer between 1 and 4096',
                { argument: 'efConstruction', value: opts.efConstruction });
        }
        if (opts.efSearch !== undefined) validateEfSearch(opts.efSearch, 'opts.efSearch');
        if (opts.seed !== undefined && (!Number.isInteger(opts.seed) || opts.seed <= 0 || opts.seed > 0x7fffffff)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'opts.seed must be an integer between 1 and 2147483647',
                { argument: 'seed', value: opts.seed });
        }

        const dim = opts.dim;
        const metric = (opts.metric === 'l2') ? 0 : 1;
        const maxElements = opts.maxElements ?? DEFAULT_MAX_ELEMENTS;
        const isQuantized = opts.quantized !== undefined ? !!opts.quantized : true;
        const quantized = isQuantized ? 1 : 0;
        const M = opts.M ?? DEFAULT_M;
        const efConstruction = opts.efConstruction ?? DEFAULT_EF_CONSTRUCTION;
        const efSearch = opts.efSearch ?? DEFAULT_EF_SEARCH;
        const seed = opts.seed ?? DEFAULT_SEED;

        // Capacity guard: the backend allocates its arena eagerly at
        // construction, and a request the wasm32 heap cannot satisfy throws
        // std::bad_alloc inside pancake_init — reject it here with the
        // estimate instead. Per-element cost mirrors the constructor
        // allocations (uint8_float_hnsw.hpp / float_hnsw.hpp): quantized rows
        // cost dim bytes plus 16*M edge bytes plus ~39 bytes of bookkeeping;
        // float rows cost 4*dim plus 8*M plus ~23.
        const bytesPerElement = isQuantized ? dim + 16 * M + 39 : 4 * dim + 8 * M + 23;
        const estimatedBytes = maxElements * bytesPerElement;
        if (estimatedBytes > MAX_INDEX_ARENA_BYTES) {
            throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED,
                `Index configuration needs ~${Math.ceil(estimatedBytes / (1024 * 1024))} MiB `
                + `(${isQuantized ? 'quantized' : 'float32'} backend: maxElements ${maxElements} × ~${bytesPerElement} B/element), `
                + `above the ${Math.floor(MAX_INDEX_ARENA_BYTES / (1024 * 1024))} MiB wasm32 index budget; `
                + 'lower maxElements or dim, or shard across indexes',
                { estimatedBytes, budgetBytes: MAX_INDEX_ARENA_BYTES, dim, maxElements, M, quantized: isQuantized });
        }

        const resolvedOpts = {
            ...opts,
            dim,
            maxElements,
            metric: metric === 0 ? 'l2' : 'cosine',
            quantized: isQuantized,
            M,
            efConstruction,
            efSearch,
            seed,
        };

        const e = await loadEngineImpl();

        const vecPtr = e._emsc_malloc(dim * 4);
        const initialBufferCapacity = 16;
        const idPtr = e._emsc_malloc(initialBufferCapacity * 8);
        const distPtr = e._emsc_malloc(initialBufferCapacity * 4);

        if (!vecPtr || !idPtr || !distPtr) {
            if (vecPtr) e._emsc_free(vecPtr);
            if (idPtr) e._emsc_free(idPtr);
            if (distPtr) e._emsc_free(distPtr);
            throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED, 'WASM malloc failed');
        }

        const handle = e._pikelet_init(dim, maxElements, quantized, metric, M, efConstruction, efSearch, seed);

        // The engine returns uint32_t INVALID_HANDLE (0xFFFFFFFF), which the
        // Emscripten i32 ABI delivers to JS as -1.
        if ((handle >>> 0) === 0xFFFFFFFF) {
            e._emsc_free(vecPtr);
            e._emsc_free(idPtr);
            e._emsc_free(distPtr);
            throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED, 'Backend init failed');
        }

        return new PikeletIndex(e, resolvedOpts, handle, vecPtr, idPtr, distPtr, initialBufferCapacity);
    }

    async function fromVectors(items, opts = {}) {
        if (!Array.isArray(items)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'fromVectors() requires an array');
        }
        if (items.length === 0) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'fromVectors() requires at least one vector');
        }

        const vectors = new Array(items.length);
        const sourceIds = new Array(items.length);
        const hasSourceIds = new Array(items.length);

        let inferredDim = null;
        for (let i = 0; i < items.length; i++) {
            const { vector, hasSourceId, sourceId } = extractVectorRecord(items[i]);
            if (!isVectorInput(vector)) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR,
                    `fromVectors() expected a vector at index ${i}`, { index: i });
            }
            const dim = vector.length;
            if (!Number.isInteger(dim) || dim <= 0) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR,
                    `fromVectors() expected a non-empty vector at index ${i}`, { index: i });
            }
            if (inferredDim === null) {
                inferredDim = dim;
            } else if (dim !== inferredDim) {
                throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
                    `fromVectors() found mixed vector dimensions (${inferredDim} and ${dim})`,
                    { expected: inferredDim, actual: dim, index: i });
            }
            vectors[i] = vector;
            sourceIds[i] = sourceId;
            hasSourceIds[i] = hasSourceId;
        }

        if (opts.dim !== undefined && opts.dim !== inferredDim) {
            throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
                `fromVectors() dim mismatch (inferred ${inferredDim}, got opts.dim=${opts.dim})`,
                { expected: inferredDim, actual: opts.dim });
        }

        const createOpts = {
            ...opts,
            dim: opts.dim ?? inferredDim,
            maxElements: opts.maxElements ?? items.length,
        };

        const index = await create(createOpts);
        try {
            const ids = index.addBatch(vectors);
            const idMap = new Map();
            for (let i = 0; i < ids.length; i++) {
                if (hasSourceIds[i]) {
                    idMap.set(ids[i], sourceIds[i]);
                }
            }
            return { index, ids, idMap };
        } catch (err) {
            try {
                index.dispose();
            } catch {}
            throw err;
        }
    }

    async function withIndex(options, fn) {
        if (typeof fn !== 'function') {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'withIndex() requires a callback function', { argument: 'fn' });
        }
        const index = await create(options);
        try {
            return await fn(index);
        } finally {
            index.dispose();
        }
    }

    function inspectSnapshot(data) {
        return inspectSnapshotMetadata(data);
    }

    async function restore(snapshot, overrides = {}) {
        if (!overrides || typeof overrides !== 'object' || Array.isArray(overrides)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'restore() overrides must be an object', { argument: 'overrides' });
        }

        const metadata = inspectSnapshotMetadata(snapshot);
        const fixedConfig = {
            dim: metadata.dim,
            metric: metadata.metric,
            quantized: metadata.quantized,
            M: metadata.M,
            efConstruction: metadata.efConstruction,
        };

        if (metadata.format === 'raw') {
            for (const field of Object.keys(fixedConfig)) {
                if (overrides[field] === undefined) {
                    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                        `restore() requires overrides.${field} for legacy raw snapshots`,
                        { argument: field, format: 'raw' });
                }
            }
        }

        for (const [field, expected] of Object.entries(fixedConfig)) {
            if (overrides[field] !== undefined && overrides[field] !== expected) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_CONFIG_MISMATCH,
                    `restore() ${field} override does not match the snapshot`,
                    { field, exported: expected, override: overrides[field] });
            }
        }

        const options = {
            ...overrides,
            ...fixedConfig,
            maxElements: overrides.maxElements ?? Math.max(1, metadata.count),
            efSearch: overrides.efSearch ?? DEFAULT_EF_SEARCH,
            seed: overrides.seed ?? DEFAULT_SEED,
        };
        const index = await create(options);
        try {
            index.import(snapshot);
            return index;
        } catch (error) {
            try {
                index.dispose();
            } catch {}
            throw error;
        }
    }

    return {
        create,
        fromVectors,
        withIndex,
        restore,
        inspectSnapshot,
        PikeletError,
        PIKELET_ERROR_CODES,
    };
}

function parseRawSnapshotMetadata(bytes) {
    if (!bytes || bytes.length < 12) return null;

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const magic = view.getUint32(0, true);

    if (magic === FLOAT_HNSW_MAGIC_V1 || magic === UINT8_HNSW_MAGIC_V1) {
        if (bytes.length < 40) return null;
        return {
            version: view.getUint32(8, true),
            dim: view.getUint32(4, true),
            count: view.getUint32(12, true),
            M: view.getUint32(24, true),
            M0: view.getUint32(28, true),
            metric: view.getUint32(32, true),
            efConstruction: view.getUint32(36, true),
            quantized: magic === UINT8_HNSW_MAGIC_V1,
        };
    }

    if (magic === FLOAT_HNSW_MAGIC_V0 || magic === UINT8_HNSW_MAGIC_V0) {
        if (bytes.length < 36) return null;
        return {
            version: 0,
            dim: view.getUint32(4, true),
            count: view.getUint32(8, true),
            M: view.getUint32(20, true),
            M0: view.getUint32(24, true),
            metric: view.getUint32(28, true),
            efConstruction: 200,
            quantized: magic === UINT8_HNSW_MAGIC_V0,
        };
    }

    return null;
}

function snapshotBytes(data, methodName) {
    if (data instanceof Uint8Array) return data;
    if (data instanceof ArrayBuffer) return new Uint8Array(data);
    if (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer) {
        return new Uint8Array(data);
    }
    if (ArrayBuffer.isView(data)) {
        return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    }
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
        `${methodName}() requires snapshot bytes`, { argument: 'snapshot' });
}

function inspectSnapshotMetadata(data) {
    const bytes = snapshotBytes(data, 'inspectSnapshot');
    if (bytes.byteLength < 4) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Snapshot is too small to inspect');
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let rawBytes = bytes;
    let format = 'raw';
    let envelopeVersion = null;
    let envelopeDim = null;
    let envelopeMetric = null;
    let envelopeQuantized = null;
    let nextId = null;

    if (view.getUint32(0, true) === PIKELET_MAGIC) {
        format = 'pikelet';
        if (bytes.byteLength < 8) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Snapshot envelope is truncated');
        }
        envelopeVersion = view.getUint32(4, true);
        let rawOffset;
        if (envelopeVersion === 1) {
            if (bytes.byteLength < V1_ENVELOPE_HEADER_SIZE) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Snapshot v1 envelope is truncated');
            }
            envelopeDim = view.getUint32(8, true);
            if (view.getUint32(12, true) !== envelopeDim) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    'DCT/PCA snapshots are no longer supported');
            }
            envelopeMetric = view.getUint32(16, true);
            envelopeQuantized = view.getUint32(20, true);
            rawOffset = V1_ENVELOPE_HEADER_SIZE;
        } else if (envelopeVersion === 2) {
            if (bytes.byteLength < V2_ENVELOPE_HEADER_SIZE) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Snapshot v2 envelope is truncated');
            }
            envelopeDim = view.getUint32(8, true);
            envelopeMetric = view.getUint32(12, true);
            envelopeQuantized = view.getUint32(16, true);
            rawOffset = V2_ENVELOPE_HEADER_SIZE;
        } else if (envelopeVersion === ENVELOPE_VERSION) {
            if (bytes.byteLength < V3_ENVELOPE_HEADER_SIZE) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Snapshot v3 envelope is truncated');
            }
            envelopeDim = view.getUint32(8, true);
            envelopeMetric = view.getUint32(12, true);
            envelopeQuantized = view.getUint32(16, true);
            nextId = view.getUint32(20, true);
            const mappingCount = view.getUint32(24, true);
            const rawSize = view.getUint32(28, true);
            rawOffset = V3_ENVELOPE_HEADER_SIZE + mappingCount * MAPPING_ENTRY_SIZE;
            if (!Number.isSafeInteger(rawOffset) || rawOffset > bytes.byteLength ||
                rawSize > bytes.byteLength - rawOffset) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    'Snapshot v3 envelope payload is truncated');
            }
            rawBytes = bytes.subarray(rawOffset, rawOffset + rawSize);
        } else {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                `Unsupported snapshot envelope version ${envelopeVersion}`, { version: envelopeVersion });
        }
        if (envelopeVersion !== ENVELOPE_VERSION) rawBytes = bytes.subarray(rawOffset);
    }

    const raw = parseRawSnapshotMetadata(rawBytes);
    if (raw === null) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            'Unsupported or truncated raw snapshot format');
    }
    if (raw.dim <= 0 || raw.metric > 1 || raw.M < 2 || raw.M > 128 ||
        raw.M0 !== raw.M * 2 || raw.efConstruction < 1 || raw.efConstruction > 4096) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            'Snapshot header contains invalid index configuration');
    }
    if (format === 'pikelet' && (envelopeDim !== raw.dim || envelopeMetric !== raw.metric ||
        !!envelopeQuantized !== raw.quantized)) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
            'Snapshot envelope does not match its raw index header');
    }
    if (nextId === null) nextId = raw.count;
    if (nextId < raw.count) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Snapshot nextId is invalid');
    }

    return Object.freeze({
        format,
        version: format === 'pikelet' ? envelopeVersion : raw.version,
        dim: raw.dim,
        count: raw.count,
        metric: raw.metric === 0 ? 'l2' : 'cosine',
        quantized: raw.quantized,
        M: raw.M,
        efConstruction: raw.efConstruction,
        nextId,
    });
}

module.exports = createPikeletApi;
module.exports.default = createPikeletApi;
module.exports.PikeletError = PikeletError;
module.exports.PIKELET_ERROR_CODES = PIKELET_ERROR_CODES;

if (typeof Symbol.dispose === 'symbol') {
    PikeletIndex.prototype[Symbol.dispose] = function disposeWithSymbol() {
        this.dispose();
    };
}
