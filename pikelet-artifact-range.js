'use strict';
// Range artifact profile (.pikelet-range) — spec/SEARCH_ARTIFACT_CONTRACT.md 9.2:
// PikeletRangeArtifact reader and the range-artifact builder.
// Split out of pikelet-artifact.js (the public entry, which re-exports the
// three parts); see that file for the module map.

const { pikeletError, PIKELET_ERROR_CODES } = require('./pikelet-errors.js');
const {
    MAX_COALESCED_RANGE_BYTES,
    readChecked,
    resolveMaxReadBytes,
    resolveMaxCacheBytes,
    readU32,
    readU16,
    readF32,
    compareDistancesAsc,
    MinHeap,
    NodeFileRangeSource,
    normalizeQuery,
    parseUint8Snapshot,
    verifySha256,
    verifySegmentSha256,
    resolveSearchK,
    resolveOptionalPositiveInt,
} = require('./pikelet-artifact-common.js');

const RANGE_MAGIC = 0x31415250; // PRA1
const HEADER_BYTES = 128;
const HEADER_BYTES_V2 = 256;
const RANGE_KIND_U8 = 1;
// v3 appends three whole-segment SHA-256 digests (id map, router, base) in
// the second half of the 256-byte header, after the v2 field block.
const RANGE_VERSION = 3;
const RANGE_DIGESTS_OFFSET = 128;
const RANGE_DIGEST_BYTES = 32;
const ROUTER_LOCATION_MASK = 0x80000000;
const LOCATION_ORDINAL_MASK = 0x7fffffff;
class PikeletRangeArtifact {
    constructor(source, header, idMap, options = {}) {
        this.source = source;
        this.version = header.version;
        this.kind = header.kind;
        this.dim = header.dim;
        this.count = header.count;
        this.entryPoint = header.entryPoint;
        this.maxLevel = header.maxLevel;
        this.M = header.M;
        this.M0 = header.M0;
        this.metric = header.metric;
        this.recordBytes = header.recordBytes;
        this.idMapOffset = header.idMapOffset;
        this.recordsOffset = header.recordsOffset;
        this.parts = header.parts;
        this.routerCount = header.routerCount;
        this.baseCount = header.baseCount;
        this.routerRecordsOffset = header.routerRecordsOffset;
        this.baseRecordsOffset = header.baseRecordsOffset;
        this.originalToLocation = idMap;
        this.cache = new Map();
        // Lazily fetched records live in a byte-budgeted LRU so a long-lived
        // reader stays bounded; the router segment (this.cache) is permanent.
        this.lazyCache = new Map();
        this.lazyCacheBytes = 0;
        this.maxCacheBytes = resolveMaxCacheBytes(options.maxCacheBytes, 64 * this.recordBytes, 64 * 1024 * 1024);
        this.currentRanges = [];
        this.routerResident = { records: 0, bytes: 0 };
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.rangeNodesDecoded = 0;
        this.loadRouter = options.loadRouter !== false;
        this.maxReadBytes = resolveMaxReadBytes(options.maxReadBytes);
        this.verify = options.verify !== false;
        // v3 artifacts carry whole-segment digests; older versions have none.
        this.digests = null;
        this.segmentVerified = { idMap: false, router: false, base: false };
    }

    static async open(source, options = {}) {
        if (!source || typeof source.read !== 'function') {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'PikeletRangeArtifact.open() requires a range source with read(offset, length)');
        }
        const headerBytes = await readChecked(source, 0, HEADER_BYTES, 'header');
        if (headerBytes.byteLength < HEADER_BYTES) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact header is truncated');
        }
        const header = parseHeader(headerBytes);
        // Resolved before construction: the id-map read below is driven by an
        // untrusted header count and must respect the caller's budget.
        const maxReadBytes = resolveMaxReadBytes(options.maxReadBytes);
        let digests = null;
        if (header.version >= 3) {
            const digestBytes = await readChecked(source, RANGE_DIGESTS_OFFSET, RANGE_DIGEST_BYTES * 3, 'header digests');
            if (digestBytes.byteLength !== RANGE_DIGEST_BYTES * 3) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact header digests are truncated');
            }
            digests = {
                idMap: new Uint8Array(digestBytes.subarray(0, RANGE_DIGEST_BYTES)),
                router: new Uint8Array(digestBytes.subarray(RANGE_DIGEST_BYTES, RANGE_DIGEST_BYTES * 2)),
                base: new Uint8Array(digestBytes.subarray(RANGE_DIGEST_BYTES * 2, RANGE_DIGEST_BYTES * 3)),
            };
        }
        const idMapBytes = await readChecked(source, header.idMapOffset, header.count * 4, 'id map', maxReadBytes);
        if (idMapBytes.byteLength !== header.count * 4) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact id map is truncated');
        }
        const copied = new Uint8Array(idMapBytes.byteLength);
        copied.set(idMapBytes);
        const idMap = new Uint32Array(copied.buffer);
        const artifact = new PikeletRangeArtifact(source, header, idMap, options);
        artifact.digests = digests;
        if (digests && artifact.verify) {
            await verifySha256(copied, digests.idMap, 'Range artifact id map');
            artifact.segmentVerified.idMap = true;
        }
        if (artifact.loadRouter && artifact.version >= 2) {
            artifact.routerResident = await artifact.loadRouterSegment();
            artifact.resetStats();
        }
        return artifact;
    }

    static async openFile(filePath, options = {}) {
        // The source owns a file descriptor from construction; a rejected
        // open must release it, or every corrupt artifact leaks an fd.
        const source = new NodeFileRangeSource(filePath);
        try {
            return await PikeletRangeArtifact.open(source, options);
        } catch (err) {
            await source.close().catch(() => {});
            throw err;
        }
    }

    async close() {
        if (this.source && typeof this.source.close === 'function') {
            await this.source.close();
        }
    }

    resetStats() {
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.rangeNodesDecoded = 0;
        this.currentRanges = [];
    }

    clearCache(options = {}) {
        this.cache.clear();
        this.lazyCache.clear();
        this.lazyCacheBytes = 0;
        if (options.reloadRouter !== false && this.loadRouter && this.version >= 2) {
            return this.loadRouterSegment().then((resident) => {
                this.routerResident = resident;
                this.resetStats();
                return resident;
            });
        }
        return Promise.resolve({ records: 0, bytes: 0 });
    }

    stats() {
        return {
            rangeRequests: this.rangeRequests,
            rangeBytes: this.rangeBytes,
            rangeNodesDecoded: this.rangeNodesDecoded,
            cachedNodes: this.cache.size + this.lazyCache.size,
            lazyCacheBytes: this.lazyCacheBytes,
            routerResident: { ...this.routerResident },
            segmentVerified: { ...this.segmentVerified },
        };
    }

    // Verify the lazily-read base segment against the header's whole-segment
    // digest (v3+). Reads the segment in bounded chunks with a streaming hash
    // (Node); runtimes without one fall back to a single read bounded by the
    // open budget. Intended for producers, CI, and hosts that materialize
    // artifacts, not the query path. Verifying individual lazy ranges needs
    // per-chunk commitments, which arrive with the contract's
    // complete-profile manifest.
    async verifyBaseSegment(options = {}) {
        if (!this.digests) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT,
                'Range artifact predates segment digests (format v3); nothing to verify against', { version: this.version });
        }
        const bytes = this.baseCount * this.recordBytes;
        await verifySegmentSha256(this.source, this.baseRecordsOffset, bytes, this.digests.base,
            'Range artifact base segment', { chunkBytes: options.chunkBytes, oneShotLimit: this.maxReadBytes });
        this.segmentVerified.base = true;
        return true;
    }

    markRanges() {
        return this.currentRanges.length;
    }

    rangesSince(mark) {
        return this.currentRanges.slice(mark);
    }

    recordAddressForId(id) {
        if (!Number.isInteger(id) || id < 0 || id >= this.count) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `Node id ${id} is outside artifact bounds`, { id, count: this.count });
        }
        const location = this.originalToLocation[id];
        if (this.version >= 2) {
            const ordinal = location & LOCATION_ORDINAL_MASK;
            if ((location & ROUTER_LOCATION_MASK) !== 0) {
                if (ordinal >= this.routerCount) throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `Router ordinal ${ordinal} is outside artifact`);
                return this.routerRecordsOffset + ordinal * this.recordBytes;
            }
            if (ordinal >= this.baseCount) throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `Base ordinal ${ordinal} is outside artifact`);
            return this.baseRecordsOffset + ordinal * this.recordBytes;
        }
        if (location >= this.count) throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `Node id ${id} is not addressable in artifact`);
        return this.recordsOffset + location * this.recordBytes;
    }

    cachedNode(id) {
        const resident = this.cache.get(id);
        if (resident !== undefined) return resident;
        const lazy = this.lazyCache.get(id);
        if (lazy !== undefined) {
            // Touch for LRU: move to the tail of insertion order.
            this.lazyCache.delete(id);
            this.lazyCache.set(id, lazy);
            return lazy;
        }
        return undefined;
    }

    cacheLazyNode(id, node) {
        if (this.lazyCache.has(id)) return;
        this.lazyCache.set(id, node);
        this.lazyCacheBytes += this.recordBytes;
        while (this.lazyCacheBytes > this.maxCacheBytes && this.lazyCache.size > 1) {
            const oldest = this.lazyCache.keys().next().value;
            this.lazyCache.delete(oldest);
            this.lazyCacheBytes -= this.recordBytes;
        }
    }

    async readNode(id) {
        const cached = this.cachedNode(id);
        if (cached !== undefined) return cached;
        await this.prefetch([id]);
        const node = this.cachedNode(id);
        if (node === undefined) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'Range artifact record for id was not resolved by its address', { id });
        }
        return node;
    }

    async prefetch(ids, options = {}) {
        const addresses = [];
        const seen = new Set();
        for (const id of ids) {
            if (this.cache.has(id) || this.lazyCache.has(id) || seen.has(id)) continue;
            seen.add(id);
            addresses.push(this.recordAddressForId(id));
        }
        if (addresses.length === 0) return 0;
        addresses.sort((a, b) => a - b);
        const gap = Math.max(0, options.gap || 0);
        const gapBytes = this.version >= 2 ? gap : gap * this.recordBytes;
        const rangeParallelism = Math.max(1, Math.trunc(options.parallelism || 1));
        // Coalesced runs are split at maxRangeBytes (record-aligned) so no
        // gap/parallelism combination — or a bulk prefetch of the whole base
        // segment — turns into one unbounded read and allocation.
        const maxRangeBytes = Math.max(this.recordBytes,
            Math.floor(Math.trunc(options.maxRangeBytes || MAX_COALESCED_RANGE_BYTES) / this.recordBytes) * this.recordBytes);
        const ranges = [];
        let runStart = addresses[0];
        let runEnd = addresses[0] + this.recordBytes;
        const flush = () => {
            for (let piece = runStart; piece < runEnd; piece += maxRangeBytes) {
                ranges.push([piece, Math.min(piece + maxRangeBytes, runEnd)]);
            }
        };
        const readRange = async ([start, end]) => {
            const bytes = end - start;
            const buffer = await readChecked(this.source, start, bytes, 'record');
            if (buffer.byteLength !== bytes) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record read returned a truncated range', { offset: start, bytes, actual: buffer.byteLength });
            }
            return { start, end, bytes, buffer };
        };
        const decodeRange = ({ start, end, bytes, buffer }) => {
            this.rangeRequests++;
            this.rangeBytes += bytes;
            this.currentRanges.push([start, end]);
            for (let off = 0; off + this.recordBytes <= bytes; off += this.recordBytes) {
                const record = buffer.subarray(off, off + this.recordBytes);
                const originalId = new DataView(record.buffer, record.byteOffset, record.byteLength).getUint32(0, true);
                // The embedded id is untrusted: it must map back to the byte
                // address this record was read from, or a lying record would
                // poison the cache under an attacker-chosen key and searches
                // for the real id would die uncoded.
                if (originalId >= this.count || this.recordAddressForId(originalId) !== start + off) {
                    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                        'Range artifact record id does not match its address',
                        { originalId, address: start + off });
                }
                if (!this.cache.has(originalId) && !this.lazyCache.has(originalId)) {
                    this.cacheLazyNode(originalId, this.decodeNode(record));
                    this.rangeNodesDecoded++;
                }
            }
        };
        for (let i = 1; i < addresses.length; i++) {
            const address = addresses[i];
            if (address <= runEnd + gapBytes) {
                runEnd = address + this.recordBytes;
            } else {
                flush();
                runStart = address;
                runEnd = address + this.recordBytes;
            }
        }
        flush();
        for (let i = 0; i < ranges.length; i += rangeParallelism) {
            const batch = ranges.slice(i, i + rangeParallelism);
            const results = await Promise.all(batch.map(readRange));
            for (const result of results) decodeRange(result);
        }
        return ranges.length;
    }

    async loadRouterSegment() {
        if (this.version < 2 || this.routerCount === 0) return { records: 0, bytes: 0 };
        const bytes = this.routerCount * this.recordBytes;
        const buffer = await readChecked(this.source, this.routerRecordsOffset, bytes, 'router segment', this.maxReadBytes);
        if (buffer.byteLength !== bytes) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact router segment is truncated');
        }
        if (this.digests && this.verify) {
            await verifySha256(buffer, this.digests.router, 'Range artifact router segment');
            this.segmentVerified.router = true;
        }
        for (let i = 0; i < this.routerCount; i++) {
            const record = buffer.subarray(i * this.recordBytes, (i + 1) * this.recordBytes);
            const originalId = new DataView(record.buffer, record.byteOffset, record.byteLength).getUint32(0, true);
            const address = this.routerRecordsOffset + i * this.recordBytes;
            // Same untrusted-id check as decodeRange: the embedded id must map
            // back to this router slot's address.
            if (originalId >= this.count || this.recordAddressForId(originalId) !== address) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    'Range artifact router record id does not match its address',
                    { originalId, address });
            }
            if (!this.cache.has(originalId)) this.cache.set(originalId, this.decodeNode(record));
        }
        return { records: this.routerCount, bytes };
    }

    decodeNode(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const state = { offset: 0 };
        const id = readU32(view, state);
        const level = readU16(view, state);
        const baseCount = readU16(view, state);
        // Records are untrusted bytes. Counts beyond the header geometry would
        // read filler (or zeros) as edges; fail closed instead.
        if (level > this.maxLevel || baseCount > this.M0) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record structure is inconsistent', { id, level, baseCount });
        }
        const upperCounts = new Uint16Array(this.maxLevel);
        for (let i = 0; i < this.maxLevel; i++) {
            upperCounts[i] = readU16(view, state);
            if (upperCounts[i] > this.M) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record structure is inconsistent', { id, level: i + 1, edges: upperCounts[i] });
            }
        }
        const qdata = new Uint8Array(this.dim);
        qdata.set(bytes.subarray(state.offset, state.offset + this.dim));
        state.offset += this.dim;
        const scale = readF32(view, state);
        const offset = readF32(view, state);
        const base = new Uint32Array(baseCount);
        for (let i = 0; i < this.M0; i++) {
            const neighbor = readU32(view, state);
            if (i < baseCount) {
                if (neighbor >= this.count) {
                    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record has an out-of-bounds neighbor', { id, neighbor, count: this.count });
                }
                base[i] = neighbor;
            }
        }
        const upper = Array.from({ length: this.maxLevel }, () => new Uint32Array(0));
        for (let levelIndex = 0; levelIndex < this.maxLevel; levelIndex++) {
            const edges = new Uint32Array(upperCounts[levelIndex]);
            for (let i = 0; i < this.M; i++) {
                const neighbor = readU32(view, state);
                if (i < edges.length) {
                    if (neighbor >= this.count) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact record has an out-of-bounds neighbor', { id, neighbor, count: this.count });
                    }
                    edges[i] = neighbor;
                }
            }
            upper[levelIndex] = edges;
        }
        return { id, level, base, upper, qdata, scale, offset };
    }

    distance(query, node) {
        if (this.metric === 1) {
            let dot = 0;
            for (let d = 0; d < this.dim; d++) {
                dot += query[d] * (node.offset + node.scale * node.qdata[d]);
            }
            if (dot > 1) dot = 1;
            else if (dot < -1) dot = -1;
            return 1 - dot;
        }
        let sum = 0;
        for (let d = 0; d < this.dim; d++) {
            const decoded = node.offset + node.scale * node.qdata[d];
            const diff = query[d] - decoded;
            sum += diff * diff;
        }
        return sum;
    }

    async search(queryInput, k, options = {}) {
        const query = normalizeQuery(queryInput, this.dim, this.metric);
        k = resolveSearchK(k, this.count);
        if (k === 0) return { results: [], rounds: [], stats: this.stats() };
        const efSearch = Math.max(resolveOptionalPositiveInt(options.efSearch, 'efSearch', 100), k);
        const expansionBatch = Math.max(1, Math.trunc(options.expansionBatch || 1));
        const rounds = [];
        const prefetchRound = async (ids) => {
            if (!ids.length) return;
            const beforeRequests = this.rangeRequests;
            const beforeBytes = this.rangeBytes;
            const beforeRanges = this.markRanges();
            await this.prefetch(ids, { gap: options.gap || 0, parallelism: options.rangeParallelism || options.parallelism || 1, maxRangeBytes: options.maxRangeBytes });
            const ranges = this.rangesSince(beforeRanges);
            rounds.push({
                ids: ids.length,
                requests: this.rangeRequests - beforeRequests,
                bytes: this.rangeBytes - beforeBytes,
                rangeBytes: ranges.map(([start, end]) => end - start),
            });
        };

        let current = this.entryPoint;
        await prefetchRound([current]);
        let currentNode = await this.readNode(current);
        let currentDistance = this.distance(query, currentNode);

        for (let level = this.maxLevel; level > 0; level--) {
            let changed = true;
            while (changed) {
                changed = false;
                const edges = currentNode.upper[level - 1] || [];
                await prefetchRound(edges);
                for (const neighbor of edges) {
                    const node = await this.readNode(neighbor);
                    const distance = this.distance(query, node);
                    if (distance < currentDistance) {
                        current = neighbor;
                        currentNode = node;
                        currentDistance = distance;
                        changed = true;
                    }
                }
            }
        }

        const visited = new Uint8Array(this.count);
        const candidates = new MinHeap(compareDistancesAsc);
        const results = new MinHeap((a, b) => b.distance - a.distance || b.id - a.id);
        currentNode = await this.readNode(current);
        const distance0 = this.distance(query, currentNode);
        candidates.push({ distance: distance0, id: current });
        results.push({ distance: distance0, id: current });
        visited[current] = 1;

        while (candidates.size > 0) {
            const batch = [];
            while (batch.length < expansionBatch && candidates.size > 0) {
                const candidate = candidates.peek();
                const worst = results.peek();
                if (results.size >= efSearch && worst && candidate && candidate.distance > worst.distance) break;
                batch.push(candidates.pop());
            }
            if (!batch.length) break;

            const toVisit = [];
            for (const candidate of batch) {
                const node = await this.readNode(candidate.id);
                for (const neighbor of node.base) {
                    if (visited[neighbor]) continue;
                    visited[neighbor] = 1;
                    toVisit.push(neighbor);
                }
            }
            await prefetchRound(toVisit);
            for (const neighbor of toVisit) {
                const neighborNode = await this.readNode(neighbor);
                const distance = this.distance(query, neighborNode);
                const currentWorst = results.peek();
                if (results.size < efSearch || distance < currentWorst.distance) {
                    candidates.push({ distance, id: neighbor });
                    results.push({ distance, id: neighbor });
                    if (results.size > efSearch) results.pop();
                }
            }
        }

        const top = results.items.sort(compareDistancesAsc).slice(0, k);
        // Traversal orders by squared L2; the API contract (README "Distance
        // values") reports Euclidean, matching PikeletIndex.search.
        if (this.metric !== 1) {
            for (const hit of top) hit.distance = Math.sqrt(hit.distance);
        }
        return {
            results: top,
            rounds,
            stats: this.stats(),
        };
    }
}

function parseHeader(headerBytes) {
    const view = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);
    const state = { offset: 0 };
    const magic = readU32(view, state);
    if (magic !== RANGE_MAGIC) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Not a Pikelet range artifact', { magic });
    }
    const version = readU32(view, state);
    const kind = readU32(view, state);
    if (kind !== RANGE_KIND_U8) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported Pikelet range artifact kind', { kind });
    }
    const header = {
        version,
        kind,
        dim: readU32(view, state),
        count: readU32(view, state),
        entryPoint: readU32(view, state),
        maxLevel: readU32(view, state),
        M: readU32(view, state),
        M0: readU32(view, state),
        metric: readU32(view, state),
        recordBytes: readU32(view, state),
        idMapOffset: readU32(view, state),
        recordsOffset: readU32(view, state),
        parts: readU32(view, state),
        routerCount: 0,
        baseCount: 0,
        routerRecordsOffset: 0,
        baseRecordsOffset: 0,
    };
    if (header.metric !== 0) {
        if (header.metric !== 1) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported Pikelet range artifact metric', { metric: header.metric });
        }
    }
    if (version >= 2) {
        header.routerCount = readU32(view, state);
        header.baseCount = readU32(view, state);
        header.routerRecordsOffset = readU32(view, state);
        header.baseRecordsOffset = readU32(view, state);
    } else {
        header.baseCount = header.count;
        header.baseRecordsOffset = header.recordsOffset;
    }
    if (version < 1 || version > RANGE_VERSION) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported Pikelet range artifact version', { version });
    }
    // recordBytes must agree with the layout the geometry implies, or
    // decodeNode would read past record boundaries on a corrupt header.
    const expectedRecordBytes = 4 + 2 + 2 + header.maxLevel * 2 + header.dim
        + 8 + header.M0 * 4 + header.maxLevel * header.M * 4;
    if (header.dim < 1 || header.count < 1 || header.M < 1 || header.M0 < 1
        || header.maxLevel > 64
        || header.entryPoint >= header.count
        || header.recordBytes !== expectedRecordBytes
        || (version >= 2 && header.routerCount + header.baseCount !== header.count)) {
        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Range artifact header geometry is inconsistent', {
            dim: header.dim, count: header.count, maxLevel: header.maxLevel,
            M: header.M, M0: header.M0, recordBytes: header.recordBytes, expectedRecordBytes,
        });
    }
    return header;
}

function artifactRecordBytes(index) {
    return 4 + 2 + 2 + index.maxLevel * 2 + index.dim + 8 + index.M0 * 4 + index.maxLevel * index.M * 4;
}

function writeNodeRecord(index, id, record) {
    record.fill(0);
    let offset = 0;
    record.writeUInt32LE(id, offset); offset += 4;
    record.writeUInt16LE(index.levels[id], offset); offset += 2;
    const baseEdges = index.base[id];
    record.writeUInt16LE(baseEdges.length, offset); offset += 2;
    for (let level = 1; level <= index.maxLevel; level++) {
        record.writeUInt16LE((index.upper[id][level - 1] || []).length, offset);
        offset += 2;
    }
    Buffer.from(index.qdata.buffer, index.qdata.byteOffset + id * index.dim, index.dim).copy(record, offset);
    offset += index.dim;
    record.writeFloatLE(index.scales[id], offset); offset += 4;
    record.writeFloatLE(index.offsets[id], offset); offset += 4;
    for (let i = 0; i < index.M0; i++) {
        record.writeUInt32LE(i < baseEdges.length ? baseEdges[i] : 0xFFFFFFFF, offset);
        offset += 4;
    }
    for (let level = 1; level <= index.maxLevel; level++) {
        const edges = index.upper[id][level - 1] || [];
        for (let i = 0; i < index.M; i++) {
            record.writeUInt32LE(i < edges.length ? edges[i] : 0xFFFFFFFF, offset);
            offset += 4;
        }
    }
}

function reverseCuthillMckeePermutation(index) {
    const count = index.count;
    const degrees = new Uint32Array(count);
    for (let id = 0; id < count; id++) degrees[id] = index.base[id].length;
    const starts = Array.from({ length: count }, (_, id) => id);
    starts.sort((a, b) => degrees[a] - degrees[b] || a - b);
    const visited = new Uint8Array(count);
    const order = new Uint32Array(count);
    const queue = new Uint32Array(count);
    let orderLen = 0;
    for (const start of starts) {
        if (visited[start]) continue;
        let read = 0;
        let write = 0;
        queue[write++] = start;
        visited[start] = 1;
        while (read < write) {
            const id = queue[read++];
            order[orderLen++] = id;
            const neighbors = Array.from(index.base[id]).filter((neighbor) => !visited[neighbor]);
            neighbors.sort((a, b) => degrees[a] - degrees[b] || a - b);
            for (const neighbor of neighbors) {
                if (visited[neighbor]) continue;
                visited[neighbor] = 1;
                queue[write++] = neighbor;
            }
        }
    }
    const ordinalToOriginal = new Uint32Array(count);
    for (let pos = 0; pos < orderLen; pos++) ordinalToOriginal[pos] = order[orderLen - 1 - pos];
    return ordinalToOriginal;
}

function identityPermutation(count) {
    const out = new Uint32Array(count);
    for (let i = 0; i < count; i++) out[i] = i;
    return out;
}

function buildOrdinalToOriginal(index, layout) {
    if (layout === 'identity') return identityPermutation(index.count);
    if (layout === 'rcm') return reverseCuthillMckeePermutation(index);
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `Unsupported Search Artifact layout '${layout}'`, { layout });
}

function exportSplitArtifact(index, outPath, options = {}) {
    const fs = require('fs');
    const layoutName = options.layout || 'rcm';
    const ordinalToOriginal = buildOrdinalToOriginal(index, layoutName);
    const recBytes = artifactRecordBytes(index);
    const idMapOffset = HEADER_BYTES_V2;
    const routerIds = [];
    const baseIds = [];
    const locationMap = new Uint32Array(index.count);
    for (let ordinal = 0; ordinal < index.count; ordinal++) {
        const id = ordinalToOriginal[ordinal];
        if (index.levels[id] > 0) routerIds.push(id);
        else baseIds.push(id);
    }
    for (let i = 0; i < routerIds.length; i++) locationMap[routerIds[i]] = ROUTER_LOCATION_MASK | i;
    for (let i = 0; i < baseIds.length; i++) locationMap[baseIds[i]] = i;

    const routerRecordsOffset = idMapOffset + index.count * 4;
    const baseRecordsOffset = routerRecordsOffset + routerIds.length * recBytes;
    const totalBytes = baseRecordsOffset + baseIds.length * recBytes;
    const fd = fs.openSync(outPath, 'w');
    // Whole-segment digests (v3): streamed while the segments are written so
    // the file is hashed exactly once, then stamped into the header at the
    // end. The id map and router are verified by the reader at open; the base
    // segment is verifiable on demand via verifyBaseSegment().
    const crypto = require('crypto');
    const idMapBuffer = Buffer.from(locationMap.buffer);
    const idMapDigest = crypto.createHash('sha256').update(idMapBuffer).digest();
    const routerHash = crypto.createHash('sha256');
    const baseHash = crypto.createHash('sha256');
    let routerDigest;
    let baseDigest;
    try {
        fs.writeSync(fd, idMapBuffer, 0, index.count * 4, idMapOffset);

        const record = Buffer.alloc(recBytes);
        for (let i = 0; i < routerIds.length; i++) {
            writeNodeRecord(index, routerIds[i], record);
            routerHash.update(record);
            fs.writeSync(fd, record, 0, record.length, routerRecordsOffset + i * recBytes);
        }
        for (let i = 0; i < baseIds.length; i++) {
            writeNodeRecord(index, baseIds[i], record);
            baseHash.update(record);
            fs.writeSync(fd, record, 0, record.length, baseRecordsOffset + i * recBytes);
        }

        const header = Buffer.alloc(HEADER_BYTES_V2);
        let h = 0;
        header.writeUInt32LE(RANGE_MAGIC, h); h += 4;
        header.writeUInt32LE(RANGE_VERSION, h); h += 4;
        header.writeUInt32LE(RANGE_KIND_U8, h); h += 4;
        header.writeUInt32LE(index.dim, h); h += 4;
        header.writeUInt32LE(index.count, h); h += 4;
        header.writeUInt32LE(index.entryPoint, h); h += 4;
        header.writeUInt32LE(index.maxLevel, h); h += 4;
        header.writeUInt32LE(index.M, h); h += 4;
        header.writeUInt32LE(index.M0, h); h += 4;
        header.writeUInt32LE(index.metric, h); h += 4;
        header.writeUInt32LE(recBytes, h); h += 4;
        header.writeUInt32LE(idMapOffset, h); h += 4;
        header.writeUInt32LE(routerRecordsOffset, h); h += 4;
        header.writeUInt32LE(options.parts || 0, h); h += 4;
        header.writeUInt32LE(routerIds.length, h); h += 4;
        header.writeUInt32LE(baseIds.length, h); h += 4;
        header.writeUInt32LE(routerRecordsOffset, h); h += 4;
        header.writeUInt32LE(baseRecordsOffset, h); h += 4;
        routerDigest = routerHash.digest();
        baseDigest = baseHash.digest();
        idMapDigest.copy(header, RANGE_DIGESTS_OFFSET);
        routerDigest.copy(header, RANGE_DIGESTS_OFFSET + RANGE_DIGEST_BYTES);
        baseDigest.copy(header, RANGE_DIGESTS_OFFSET + RANGE_DIGEST_BYTES * 2);
        fs.writeSync(fd, header, 0, header.length, 0);
    } finally {
        fs.closeSync(fd);
    }
    return {
        format: 'pikelet-range-artifact',
        formatVersion: RANGE_VERSION,
        integrity: {
            idMapSha256: idMapDigest.toString('hex'),
            routerSha256: routerDigest.toString('hex'),
            baseSha256: baseDigest.toString('hex'),
        },
        file: outPath,
        sizeBytes: totalBytes,
        kind: 'u8-affine',
        metric: index.metric === 0 ? 'l2' : 'cosine',
        layout: { permutation: layoutName, split: 'router_then_base' },
        graph: {
            count: index.count,
            dim: index.dim,
            entryPoint: index.entryPoint,
            maxLevel: index.maxLevel,
            M: index.M,
            M0: index.M0,
        },
        addressing: {
            headerBytes: HEADER_BYTES_V2,
            idMapOffset,
            recordBytes: recBytes,
            routerCount: routerIds.length,
            routerBytes: routerIds.length * recBytes,
            routerRecordsOffset,
            baseCount: baseIds.length,
            baseBytes: baseIds.length * recBytes,
            baseRecordsOffset,
        },
    };
}

function buildRangeArtifact(snapshotBytes, outPath, options = {}) {
    if (typeof outPath !== 'string' || outPath.length === 0) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'buildRangeArtifact() requires an output path');
    }
    const index = parseUint8Snapshot(snapshotBytes);
    return exportSplitArtifact(index, outPath, options);
}

function buildRangeArtifactFile(snapshotPath, outPath, options = {}) {
    const fs = require('fs');
    const snapshot = fs.readFileSync(snapshotPath);
    return buildRangeArtifact(snapshot, outPath, options);
}

module.exports = {
    PikeletRangeArtifact,
    buildRangeArtifact,
    buildRangeArtifactFile,
    exportSplitArtifact,
    parseHeader,
};
