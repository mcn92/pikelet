'use strict';
// Sketch artifact profile (.pikelet-sketch) — spec/SKETCH_PROFILE.md:
// PikeletSketchArtifact reader, the sketch-artifact builders, and the
// engine-backed resident scanner.
// Split out of pikelet-artifact.js (the public entry, which re-exports the
// three parts); see that file for the module map.

const { pikeletError, PIKELET_ERROR_CODES } = require('./pikelet-errors.js');
const {
    DEFAULT_OPEN_READ_BYTES,
    MAX_COALESCED_RANGE_BYTES,
    mapLimit,
    readChecked,
    resolveMaxReadBytes,
    resolveMaxCacheBytes,
    NodeFileRangeSource,
    parseUint8Snapshot,
    sha256Bytes,
    sha256BytesAsync,
    verifySegmentSha256,
} = require('./pikelet-artifact-common.js');
const {
    normalizeQuery: normalizeQueryVector,
    resolveSearchK,
    resolveOptionalPositiveInt,
} = require('./pikelet-artifact-common.js');

// ============================================================================
// Sketch artifact profile (.pikelet-sketch) — spec/SKETCH_PROFILE.md
// ============================================================================
//
// Layout: 256-byte header | scales f32[count] | offsets f32[count] |
// packed pooled sketches | zero padding to 16-byte alignment | raw u8 rows.
// Everything before vectorsOffset is the resident prefix; row addresses are
// computed (vectorsOffset + id*dim), never looked up.

const SKETCH_MAGIC = 0x31415350; // PSA1
const SKETCH_HEADER_BYTES = 256;
const SKETCH_KIND_U8 = 1;

function buildSketchArtifact(snapshotBytes, outPath, options = {}) {
    if (typeof outPath !== 'string' || outPath.length === 0) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'buildSketchArtifact() requires an output path');
    }
    return exportSketchArtifact(parseUint8Snapshot(snapshotBytes), outPath, options);
}

// Accepts any index-like source of quantized rows ({ dim, count, metric,
// qdata, scales, offsets }) — the profile is derivable from a snapshot, a
// range artifact, or any producer that emits row-affine u8 vectors.
function exportSketchArtifact(index, outPath, options = {}) {
    const dim = index.dim;
    const count = index.count;
    const sketchDims = options.sketchDims || (dim % 2 === 0 ? dim / 2 : dim);
    const sketchBits = options.sketchBits || 4;
    const recommendedRerank = options.recommendedRerank || 0;
    if (![4, 8].includes(sketchBits)) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'sketchBits must be 4 or 8', { sketchBits });
    }
    if (!Number.isInteger(sketchDims) || sketchDims < 1 || dim % sketchDims !== 0) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'sketchDims must divide dim', { sketchDims, dim });
    }
    if (sketchBits === 4 && sketchDims % 2 !== 0) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'sketchDims must be even for 4-bit sketches', { sketchDims });
    }
    // Optional staged-boot micro tier: a second, coarser pooling of the same
    // quantized rows, stored after the full sketches so v1 readers see it
    // only as opaque resident tail bytes (their layout check permits a gap
    // between sketches end and vectorsOffset, and the resident hash already
    // covers it). Pooling commutes with the per-row affine map, so the micro
    // tier reuses scales/offsets unchanged, exactly like the full tier.
    const microDims = options.microDims || 0;
    // 4-bit default: the 2026-08-04 sweep on the 456k wiki corpus measured
    // bit depth as worthless at fixed bytes (48d/4b == 48d/8b within noise)
    // while halved pooling at the same bytes gained +28 recall points
    // (96d/4b 87.8% vs 48d/8b 59.4% capture at C=800). Micro tiers should
    // spend their byte budget on dims, never on bits.
    const microBits = microDims ? (options.microBits || 4) : 0;
    if (microDims) {
        if (!Number.isInteger(microDims) || microDims < 1 || sketchDims % microDims !== 0 || microDims >= sketchDims) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'microDims must divide sketchDims and be smaller', { microDims, sketchDims });
        }
        if (![4, 8].includes(microBits) || (microBits === 4 && microDims % 2 !== 0)) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'invalid micro sketch encoding', { microBits, microDims });
        }
    }
    // Per-row integrity (format version 2, spec section 2.4): the vectors
    // region is written as blocks of rowsPerBlock rows, each preceded by a
    // digest page of truncated per-row SHA-256s, and a 32-byte-per-block
    // page-hash table sits at the end of the resident prefix — so
    // residentSha256 (and any identity that commits to the header) anchors
    // every lazily fetched row. The 2026-08-25 measurement chose the
    // defaults: 16 rows x 16-byte digests ride inside the reader's existing
    // coalesced runs at ~0% latency cost. rowIntegrity:false emits v1.
    const rowIntegrity = options.rowIntegrity !== false;
    const rowsPerBlock = options.rowsPerBlock || 16;
    const rowDigestBytes = options.rowDigestBytes || 16;
    if (rowIntegrity) {
        if (!Number.isInteger(rowsPerBlock) || rowsPerBlock < 1 || rowsPerBlock > 4096) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'rowsPerBlock must be an integer in [1, 4096]', { rowsPerBlock });
        }
        if (!Number.isInteger(rowDigestBytes) || rowDigestBytes < 8 || rowDigestBytes > 32) {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'rowDigestBytes must be an integer in [8, 32]', { rowDigestBytes });
        }
    }
    const pool = dim / sketchDims;
    const sketchRowBytes = (sketchDims * sketchBits) / 8;
    const microPool = microDims ? dim / microDims : 0;
    const microRowBytes = microDims ? (microDims * microBits) / 8 : 0;

    const scalesOffset = SKETCH_HEADER_BYTES;
    const offsetsOffset = scalesOffset + count * 4;
    const sketchesOffset = offsetsOffset + count * 4;
    const sketchesEnd = sketchesOffset + count * sketchRowBytes;
    const microOffset = microDims ? sketchesEnd : 0;
    const tiersEnd = microDims ? microOffset + count * microRowBytes : sketchesEnd;
    const numBlocks = rowIntegrity ? Math.ceil(count / rowsPerBlock) : 0;
    const pageTableOffset = rowIntegrity ? tiersEnd : 0;
    const residentEnd = rowIntegrity ? pageTableOffset + numBlocks * 32 : tiersEnd;
    const vectorsOffset = Math.ceil(residentEnd / 16) * 16;
    const pageBytes = rowIntegrity ? rowsPerBlock * rowDigestBytes : 0;
    const blockBytes = rowIntegrity ? pageBytes + rowsPerBlock * dim : 0;
    const vectorsRegionBytes = rowIntegrity
        ? (numBlocks - 1) * blockBytes + pageBytes + (count - (numBlocks - 1) * rowsPerBlock) * dim
        : count * dim;
    const fileBytes = vectorsOffset + vectorsRegionBytes;

    const out = Buffer.alloc(fileBytes);
    out.writeUInt32LE(SKETCH_MAGIC, 0);
    out.writeUInt32LE(rowIntegrity ? 2 : 1, 4);
    out.writeUInt32LE(SKETCH_KIND_U8, 8);
    out.writeUInt32LE(index.metric, 12);
    out.writeUInt32LE(dim, 16);
    out.writeUInt32LE(count, 20);
    out.writeUInt32LE(sketchDims, 24);
    out.writeUInt32LE(sketchBits, 28);
    out.writeUInt32LE(scalesOffset, 32);
    out.writeUInt32LE(offsetsOffset, 36);
    out.writeUInt32LE(sketchesOffset, 40);
    out.writeUInt32LE(vectorsOffset, 44);
    out.writeBigUInt64LE(BigInt(fileBytes), 48);
    out.writeUInt32LE(recommendedRerank, 120);
    if (microDims) {
        out.writeUInt32LE(microDims, 124);
        out.writeUInt32LE(microBits, 128);
        out.writeUInt32LE(microOffset, 132);
    }
    if (rowIntegrity) {
        out.writeUInt32LE(rowsPerBlock, 168);
        out.writeUInt32LE(rowDigestBytes, 172);
        out.writeUInt32LE(pageTableOffset, 176);
    }

    for (let i = 0; i < count; i++) out.writeFloatLE(index.scales[i], scalesOffset + i * 4);
    for (let i = 0; i < count; i++) out.writeFloatLE(index.offsets[i], offsetsOffset + i * 4);

    for (let i = 0; i < count; i++) {
        const rowBase = i * dim;
        for (let sd = 0; sd < sketchDims; sd++) {
            let acc = 0;
            for (let j = 0; j < pool; j++) acc += index.qdata[rowBase + sd * pool + j];
            const pooled = Math.round(acc / pool);
            if (sketchBits === 8) {
                out[sketchesOffset + i * sketchRowBytes + sd] = pooled;
            } else {
                const q = Math.min(15, Math.round(pooled / 17));
                const byteIndex = sketchesOffset + i * sketchRowBytes + (sd >> 1);
                if (sd % 2 === 0) out[byteIndex] = (out[byteIndex] & 0xf0) | q;
                else out[byteIndex] = (out[byteIndex] & 0x0f) | (q << 4);
            }
        }
        if (microDims) {
            for (let sd = 0; sd < microDims; sd++) {
                let acc = 0;
                for (let j = 0; j < microPool; j++) acc += index.qdata[rowBase + sd * microPool + j];
                const pooled = Math.round(acc / microPool);
                if (microBits === 8) {
                    out[microOffset + i * microRowBytes + sd] = pooled;
                } else {
                    const q = Math.min(15, Math.round(pooled / 17));
                    const byteIndex = microOffset + i * microRowBytes + (sd >> 1);
                    if (sd % 2 === 0) out[byteIndex] = (out[byteIndex] & 0xf0) | q;
                    else out[byteIndex] = (out[byteIndex] & 0x0f) | (q << 4);
                }
            }
        }
    }

    if (rowIntegrity) {
        // Interleaved vectors region: [digest page][rows] per block. Digest
        // pages are written first per block so the page hash (into the
        // resident page table) covers the final page bytes; unused digest
        // slots in the last block stay zero.
        for (let b = 0; b < numBlocks; b++) {
            const blockStart = vectorsOffset + b * blockBytes;
            const rowsInBlock = Math.min(rowsPerBlock, count - b * rowsPerBlock);
            for (let j = 0; j < rowsInBlock; j++) {
                const id = b * rowsPerBlock + j;
                const row = Buffer.from(index.qdata.buffer, index.qdata.byteOffset + id * dim, dim);
                out.set(row, blockStart + pageBytes + j * dim);
                out.set(sha256Bytes(row).subarray(0, rowDigestBytes), blockStart + j * rowDigestBytes);
            }
            out.set(sha256Bytes(out.subarray(blockStart, blockStart + pageBytes)), pageTableOffset + b * 32);
        }
    } else {
        Buffer.from(index.qdata.buffer, index.qdata.byteOffset, count * dim).copy(out, vectorsOffset);
    }

    sha256Bytes(out.subarray(SKETCH_HEADER_BYTES, vectorsOffset)).copy(out, 56);
    sha256Bytes(out.subarray(vectorsOffset)).copy(out, 88);
    if (microDims) {
        // Stage-1 hash: scales+offsets plus the micro segment (and, on v2,
        // the page-hash table, so staged readers can verify rows during the
        // micro window) — exactly the bytes a staged open serves from,
        // verifiable before the full sketches arrive. The full resident hash
        // still covers everything.
        const stage1Parts = [
            out.subarray(SKETCH_HEADER_BYTES, sketchesOffset),
            out.subarray(microOffset, microOffset + count * microRowBytes),
        ];
        if (rowIntegrity) stage1Parts.push(out.subarray(pageTableOffset, pageTableOffset + numBlocks * 32));
        sha256Bytes(Buffer.concat(stage1Parts)).copy(out, 136);
    }

    const manifest = {
        format: 'pikelet-sketch-artifact',
        formatVersion: rowIntegrity ? 2 : 1,
        sizeBytes: fileBytes,
        metric: index.metric === 1 ? 'cosine' : 'l2',
        graph: { count, dim },
        sketch: { sketchDims, sketchBits, pool, residentBytes: vectorsOffset },
        micro: microDims ? { microDims, microBits, microPool, stage1Bytes: sketchesOffset + count * microRowBytes } : null,
        rowIntegrity: rowIntegrity
            ? { rowsPerBlock, rowDigestBytes, blocks: numBlocks, pageTableBytes: numBlocks * 32, regionBytes: vectorsRegionBytes }
            : null,
        addressing: { scalesOffset, offsetsOffset, sketchesOffset, vectorsOffset },
        recommendedRerank,
    };

    if (outPath === null) return { bytes: out, manifest };

    const fs = require('fs');
    fs.writeFileSync(outPath, out);
    return { ...manifest, file: outPath };
}

// Bytes-in/bytes-out variant for producers that assemble segments in memory
// (e.g. the complete-profile compiler): identical output to
// buildSketchArtifact, no filesystem involved. Returns { bytes, manifest }.
function buildSketchArtifactBytes(snapshotBytes, options = {}) {
    return exportSketchArtifact(parseUint8Snapshot(snapshotBytes), null, options);
}

function buildSketchArtifactFile(snapshotPath, outPath, options = {}) {
    const fs = require('fs');
    return buildSketchArtifact(fs.readFileSync(snapshotPath), outPath, options);
}

class PikeletSketchArtifact {
    constructor(source) {
        this.source = source;
        // Fetched rows live in a byte-budgeted LRU; search correctness never
        // depends on retention because fetchRows returns rows directly.
        this.cache = new Map();
        this.cacheBytes = 0;
        this.maxCacheBytes = 64 * 1024 * 1024;
        this.rangeRequests = 0;
        this.rangeBytes = 0;
        this.maxReadBytes = DEFAULT_OPEN_READ_BYTES;
        this.residentVerified = false;
        this.vectorsVerified = false;
    }

    cachedRow(id) {
        const row = this.cache.get(id);
        if (row !== undefined) {
            this.cache.delete(id);
            this.cache.set(id, row);
        }
        return row;
    }

    // Drop all cached rows. Cache state never affects result semantics, so
    // this only changes fetch behavior (every subsequent search starts cold).
    clearCache() {
        this.cache.clear();
        this.cacheBytes = 0;
    }

    cacheRow(id, row) {
        if (this.cache.has(id)) return;
        this.cache.set(id, row);
        this.cacheBytes += this.dim;
        while (this.cacheBytes > this.maxCacheBytes && this.cache.size > 1) {
            const oldest = this.cache.keys().next().value;
            this.cache.delete(oldest);
            this.cacheBytes -= this.dim;
        }
    }

    static async open(source, options = {}) {
        if (!source || typeof source.read !== 'function') {
            throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'SketchArtifact.open() requires a range source with read(offset, length)');
        }
        const artifact = new PikeletSketchArtifact(source);
        artifact.maxReadBytes = resolveMaxReadBytes(options.maxReadBytes);
        const header = await readChecked(source, 0, SKETCH_HEADER_BYTES, 'header');
        if (header.byteLength !== SKETCH_HEADER_BYTES) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact header is truncated');
        }
        const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
        if (view.getUint32(0, true) !== SKETCH_MAGIC) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Not a Pikelet sketch artifact (bad magic)');
        }
        const version = view.getUint32(4, true);
        if (version !== 1 && version !== 2) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact version', { version });
        }
        artifact.formatVersion = version;
        if (view.getUint32(8, true) !== SKETCH_KIND_U8) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact kind');
        }
        artifact.metric = view.getUint32(12, true);
        artifact.dim = view.getUint32(16, true);
        artifact.count = view.getUint32(20, true);
        artifact.sketchDims = view.getUint32(24, true);
        artifact.sketchBits = view.getUint32(28, true);
        const scalesOffset = view.getUint32(32, true);
        const offsetsOffset = view.getUint32(36, true);
        const sketchesOffset = view.getUint32(40, true);
        artifact.vectorsOffset = view.getUint32(44, true);
        const fileBytes = Number(view.getBigUint64(48, true));
        artifact.recommendedRerank = view.getUint32(120, true);
        // Retained so the lazy tier stays verifiable after open — see
        // verifyVectors(). Copied out of the header read buffer.
        artifact.vectorsSha256 = new Uint8Array(header.subarray(88, 120));
        artifact.microDims = view.getUint32(124, true);
        artifact.microBits = artifact.microDims ? view.getUint32(128, true) : 0;
        const microOffset = artifact.microDims ? view.getUint32(132, true) : 0;
        artifact.rowsPerBlock = version >= 2 ? view.getUint32(168, true) : 0;
        artifact.rowDigestBytes = version >= 2 ? view.getUint32(172, true) : 0;
        const pageTableOffset = version >= 2 ? view.getUint32(176, true) : 0;
        artifact.maxCacheBytes = resolveMaxCacheBytes(options.maxCacheBytes, 256 * artifact.dim, artifact.maxCacheBytes);

        const { metric, dim, count, sketchDims, sketchBits, vectorsOffset } = artifact;
        if (metric !== 0 && metric !== 1) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Unsupported sketch artifact metric', { metric });
        }
        if (dim < 1 || count < 1 || sketchDims < 1 || dim % sketchDims !== 0) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Invalid sketch artifact geometry', { dim, count, sketchDims });
        }
        if (![4, 8].includes(sketchBits) || (sketchBits === 4 && sketchDims % 2 !== 0)) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Invalid sketch encoding', { sketchBits, sketchDims });
        }
        const sketchRowBytes = (sketchDims * sketchBits) / 8;
        // Version 2 interleaves a digest page ahead of each rowsPerBlock rows
        // and stores a 32-byte-per-block page-hash table at the end of the
        // resident prefix; the vectors region size follows from that geometry.
        if (version >= 2) {
            const P = artifact.rowsPerBlock;
            const D = artifact.rowDigestBytes;
            if (!Number.isInteger(P) || P < 1 || P > 4096 || !Number.isInteger(D) || D < 8 || D > 32) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Invalid sketch row-integrity geometry', { rowsPerBlock: P, rowDigestBytes: D });
            }
            artifact.numBlocks = Math.ceil(count / P);
            artifact.pageBytes = P * D;
            artifact.blockBytes = artifact.pageBytes + P * dim;
            artifact.vectorsRegionBytes = (artifact.numBlocks - 1) * artifact.blockBytes
                + artifact.pageBytes + (count - (artifact.numBlocks - 1) * P) * dim;
            if (!Number.isSafeInteger(artifact.vectorsRegionBytes)
                || pageTableOffset < sketchesOffset + count * sketchRowBytes
                || pageTableOffset + artifact.numBlocks * 32 > vectorsOffset) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact page table layout is inconsistent', { pageTableOffset });
            }
        } else {
            artifact.numBlocks = 0;
            artifact.vectorsRegionBytes = count * dim;
        }
        const expectVectors = artifact.vectorsRegionBytes;
        if (!Number.isSafeInteger(fileBytes) || !Number.isSafeInteger(expectVectors)
            || scalesOffset !== SKETCH_HEADER_BYTES
            || offsetsOffset !== scalesOffset + count * 4
            || sketchesOffset !== offsetsOffset + count * 4
            || vectorsOffset < sketchesOffset + count * sketchRowBytes
            || vectorsOffset % 16 !== 0
            || fileBytes !== vectorsOffset + expectVectors) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact layout is inconsistent');
        }
        let microRowBytes = 0;
        if (artifact.microDims) {
            microRowBytes = (artifact.microDims * artifact.microBits) / 8;
            if (sketchDims % artifact.microDims !== 0 || artifact.microDims >= sketchDims
                || ![4, 8].includes(artifact.microBits) || (artifact.microBits === 4 && artifact.microDims % 2 !== 0)
                || microOffset !== sketchesOffset + count * sketchRowBytes
                || microOffset + count * microRowBytes > vectorsOffset) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact micro tier layout is inconsistent', { microDims: artifact.microDims });
            }
        }
        artifact.microOffset = microOffset;
        if (version >= 2) {
            // The page table must sit exactly at the end of the sketch tiers,
            // still inside the resident prefix residentSha256 covers.
            const tiersEnd = artifact.microDims ? microOffset + count * microRowBytes : sketchesOffset + count * sketchRowBytes;
            if (pageTableOffset !== tiersEnd) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact page table layout is inconsistent', { pageTableOffset, tiersEnd });
            }
        }
        artifact.pageTableOffset = pageTableOffset;

        const verify = options.verify !== false;
        // Per-row verification on format 2 follows the same switch as the
        // resident hash: on by default, verify:false opts the whole open out.
        artifact.verifyRows = version >= 2 && verify;
        const staged = options.staged === true && artifact.microDims > 0;
        artifact.tier = 'full';
        artifact.fullyResident = Promise.resolve(artifact);

        if (!staged) {
            const resident = await readChecked(source, SKETCH_HEADER_BYTES, vectorsOffset - SKETCH_HEADER_BYTES, 'resident prefix', artifact.maxReadBytes);
            if (resident.byteLength !== vectorsOffset - SKETCH_HEADER_BYTES) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact resident prefix is truncated');
            }
            // Copy into an aligned buffer so the typed-array views are valid
            // regardless of the source's byteOffset.
            const residentCopy = new Uint8Array(resident.byteLength);
            residentCopy.set(resident);
            artifact._adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset);
            if (verify) await artifact._verifyFullResident(residentCopy, header);
            return artifact;
        }

        // Staged boot: serve queries from the coarse micro tier after
        // fetching only scales+offsets and the micro segment (one parallel
        // wave after the header), then complete residency in the background.
        // Stage 1 is independently hash-verified; stage 2 re-verifies the
        // full resident hash before the tier swap, so both states honor the
        // integrity contract. Result semantics differ between tiers, so the
        // tier is reported on every search result rather than hidden.
        artifact.tier = 'micro';
        // On v2 the page-hash table joins the stage-1 wave (and its hash), so
        // rows fetched during the micro window are verifiable too.
        const pageTableBytes = version >= 2 ? artifact.numBlocks * 32 : 0;
        const [affineBytes, microBytes, pageTableRead] = await Promise.all([
            readChecked(source, SKETCH_HEADER_BYTES, sketchesOffset - SKETCH_HEADER_BYTES, 'stage-1 affine', artifact.maxReadBytes),
            readChecked(source, microOffset, count * microRowBytes, 'stage-1 micro', artifact.maxReadBytes),
            version >= 2 ? readChecked(source, pageTableOffset, pageTableBytes, 'stage-1 page table', artifact.maxReadBytes) : null,
        ]);
        if (affineBytes.byteLength !== sketchesOffset - SKETCH_HEADER_BYTES || microBytes.byteLength !== count * microRowBytes
            || (version >= 2 && pageTableRead.byteLength !== pageTableBytes)) {
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-1 read is truncated');
        }
        const affineCopy = new Uint8Array(affineBytes.byteLength);
        affineCopy.set(affineBytes);
        const microCopy = new Uint8Array(microBytes.byteLength);
        microCopy.set(microBytes);
        const pageTableCopy = version >= 2 ? new Uint8Array(pageTableRead.byteLength) : null;
        if (pageTableCopy) pageTableCopy.set(pageTableRead);
        if (verify) {
            const stage1 = new Uint8Array(affineCopy.byteLength + microCopy.byteLength + (pageTableCopy ? pageTableCopy.byteLength : 0));
            stage1.set(affineCopy, 0);
            stage1.set(microCopy, affineCopy.byteLength);
            if (pageTableCopy) stage1.set(pageTableCopy, affineCopy.byteLength + microCopy.byteLength);
            const digest = await sha256BytesAsync(stage1);
            if (!digest) {
                // Verification requested but no crypto backend: fail closed.
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                    'Sketch artifact verification requested but no crypto backend is available; pass verify:false to skip');
            }
            const expected = header.subarray(136, 168);
            for (let b = 0; b < 32; b++) {
                if (digest[b] !== expected[b]) {
                    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-1 prefix failed hash verification');
                }
            }
            artifact.residentVerified = true;
        }
        artifact.scales = new Float32Array(affineCopy.buffer, 0, count);
        artifact.offsets = new Float32Array(affineCopy.buffer, count * 4, count);
        artifact.sketches = null;
        artifact.microSketches = microCopy;
        if (pageTableCopy) artifact.pageTable = pageTableCopy;
        artifact.residentBytes = SKETCH_HEADER_BYTES + affineCopy.byteLength + microCopy.byteLength + (pageTableCopy ? pageTableCopy.byteLength : 0);

        const onStage = typeof options.onStage === 'function' ? options.onStage : null;
        if (onStage) onStage({ tier: 'micro', residentBytes: artifact.residentBytes });
        artifact.fullyResident = (async () => {
            const rest = await readChecked(source, sketchesOffset, vectorsOffset - sketchesOffset, 'stage-2 sketches', artifact.maxReadBytes);
            if (rest.byteLength !== vectorsOffset - sketchesOffset) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact stage-2 read is truncated');
            }
            const residentCopy = new Uint8Array((sketchesOffset - SKETCH_HEADER_BYTES) + rest.byteLength);
            residentCopy.set(affineCopy, 0);
            residentCopy.set(rest, sketchesOffset - SKETCH_HEADER_BYTES);
            if (verify) await artifact._verifyFullResident(residentCopy, header);
            artifact._adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset);
            artifact.tier = 'full';
            if (onStage) onStage({ tier: 'full', residentBytes: artifact.residentBytes });
            return artifact;
        })();
        // Surfacing failures is the caller's job via the promise; an
        // unobserved rejection must not crash the process mid-boot.
        artifact.fullyResident.catch(() => {});
        return artifact;
    }

    _adoptResident(residentCopy, count, sketchRowBytes, microRowBytes, microOffset) {
        this.scales = new Float32Array(residentCopy.buffer, 0, count);
        this.offsets = new Float32Array(residentCopy.buffer, count * 4, count);
        this.sketches = new Uint8Array(residentCopy.buffer, count * 8, count * sketchRowBytes);
        if (this.microDims) {
            this.microSketches = new Uint8Array(residentCopy.buffer, microOffset - SKETCH_HEADER_BYTES, count * microRowBytes);
        }
        if (this.formatVersion >= 2) {
            this.pageTable = new Uint8Array(residentCopy.buffer, this.pageTableOffset - SKETCH_HEADER_BYTES, this.numBlocks * 32);
        }
        this.residentBytes = this.vectorsOffset;
    }

    async _verifyFullResident(residentCopy, header) {
        const digest = await sha256BytesAsync(residentCopy);
        if (!digest) {
            // Verification was requested (callers gate this on verify) but no
            // crypto backend is available. Fail closed rather than admit
            // unverified bytes.
            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                'Sketch artifact verification requested but no crypto backend is available; pass verify:false to skip');
        }
        const expected = header.subarray(56, 88);
        for (let b = 0; b < 32; b++) {
            if (digest[b] !== expected[b]) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact resident prefix failed hash verification');
            }
        }
        this.residentVerified = true;
    }

    static async openFile(filePath, options = {}) {
        // The source owns a file descriptor from construction; a rejected
        // open must release it, or every corrupt artifact leaks an fd.
        const source = new NodeFileRangeSource(filePath);
        try {
            return await PikeletSketchArtifact.open(source, options);
        } catch (err) {
            await source.close().catch(() => {});
            throw err;
        }
    }

    stats() {
        return {
            rangeRequests: this.rangeRequests,
            rangeBytes: this.rangeBytes,
            cachedRows: this.cache.size,
            cacheBytes: this.cacheBytes,
            residentBytes: this.residentBytes,
            residentVerified: this.residentVerified,
            vectorsVerified: this.vectorsVerified,
        };
    }

    // Verify the lazy vectors segment against the header's whole-segment
    // hash. Reads the segment in bounded chunks with a streaming hash (Node);
    // runtimes without one fall back to a single read bounded by the open
    // budget. Intended for producers, CI, and hosts that materialize
    // artifacts, not the query path. Verifying individual row ranges needs
    // per-chunk commitments, which arrive with the contract's
    // complete-profile manifest.
    async verifyVectors(options = {}) {
        const bytes = this.vectorsRegionBytes;
        await verifySegmentSha256(this.source, this.vectorsOffset, bytes, this.vectorsSha256,
            'Sketch artifact vectors segment', { chunkBytes: options.chunkBytes, oneShotLimit: this.maxReadBytes });
        this.vectorsVerified = true;
        return true;
    }

    sketchValue(id, sd) {
        if (this.sketchBits === 8) return this.sketches[id * this.sketchDims + sd];
        const byte = this.sketches[id * (this.sketchDims >> 1) + (sd >> 1)];
        const nibble = sd % 2 === 0 ? (byte & 0x0f) : (byte >> 4);
        return nibble * 17;
    }

    microValue(id, sd) {
        if (this.microBits === 8) return this.microSketches[id * this.microDims + sd];
        const byte = this.microSketches[id * (this.microDims >> 1) + (sd >> 1)];
        const nibble = sd % 2 === 0 ? (byte & 0x0f) : (byte >> 4);
        return nibble * 17;
    }

    // Active resident tier for candidate selection. 'full' whenever the full
    // sketches are adopted; 'micro' only during a staged boot window.
    activeTier() {
        return this.tier === 'micro'
            ? { name: 'micro', dims: this.microDims, value: (i, sd) => this.microValue(i, sd) }
            : { name: 'full', dims: this.sketchDims, value: (i, sd) => this.sketchValue(i, sd) };
    }

    async fetchRows(ids, options = {}) {
        if (this.formatVersion >= 2) return this._fetchRowsVerified(ids, options);
        const defaultGap = this.source.preferredGapBytes === undefined ? 2048 : this.source.preferredGapBytes;
        const gap = Math.max(0, options.gap === undefined ? defaultGap : options.gap);
        const preferredParallelism = this.source.preferredParallelism === undefined ? 32 : this.source.preferredParallelism;
        const requestedParallelism = options.parallelism === undefined ? preferredParallelism : options.parallelism;
        const parallelism = requestedParallelism === 0 ? Infinity : Math.max(1, Math.trunc(requestedParallelism || 32));
        const rows = new Map();
        const missing = [];
        for (const id of ids) {
            if (rows.has(id)) continue;
            if (!Number.isInteger(id) || id < 0 || id >= this.count) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'row id out of range', { id });
            }
            const cached = this.cachedRow(id);
            if (cached !== undefined) rows.set(id, cached);
            else {
                rows.set(id, null);
                missing.push(id);
            }
        }
        if (!missing.length) return rows;
        missing.sort((a, b) => a - b);
        const dim = this.dim;
        // Coalesced runs are split at maxRangeBytes (row-aligned) so no gap
        // setting — or a bulk fetch of every row — turns into one unbounded
        // read and allocation.
        const maxRunRows = Math.max(1, Math.floor(Math.trunc(options.maxRangeBytes || MAX_COALESCED_RANGE_BYTES) / dim));
        const ranges = [];
        const pushRun = (startId, endId) => {
            for (let piece = startId; piece < endId; piece += maxRunRows) {
                ranges.push([piece, Math.min(piece + maxRunRows, endId)]);
            }
        };
        let runStartId = missing[0];
        let runEndId = missing[0] + 1;
        for (let i = 1; i < missing.length; i++) {
            const id = missing[i];
            if (id * dim <= runEndId * dim + gap) {
                runEndId = id + 1;
            } else {
                pushRun(runStartId, runEndId);
                runStartId = id;
                runEndId = id + 1;
            }
        }
        pushRun(runStartId, runEndId);
        const buffers = await mapLimit(ranges, parallelism, async ([startId, endId]) => {
            const offset = this.vectorsOffset + startId * dim;
            const length = (endId - startId) * dim;
            const bytes = await readChecked(this.source, offset, length, 'row');
            if (bytes.byteLength !== length) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact row read returned a truncated range', { offset, length });
            }
            this.rangeRequests++;
            this.rangeBytes += length;
            return { startId, endId, bytes };
        });
        for (const { startId, endId, bytes } of buffers) {
            for (let id = startId; id < endId; id++) {
                    // Copy each row out of the coalesced fetch buffer: a view
                    // would pin the whole range in the LRU, so cacheBytes
                    // would count dim bytes while retaining the full fetch.
                    // (Explicit copy constructor — .slice() is unreliable here
                    // because Node Buffers override it with view semantics.)
                    const row = new Uint8Array(bytes.subarray((id - startId) * dim, (id - startId + 1) * dim));
                if (rows.has(id)) rows.set(id, row);
                this.cacheRow(id, row);
            }
        }
        return rows;
    }

    // Format-2 row fetch: rows live in interleaved blocks
    // [digest page][rowsPerBlock rows], so each needed block is fetched from
    // its page through the last needed row (the page rides inside the same
    // read), the page is verified against the resident page-hash table —
    // which residentSha256, and any identity committing to the header,
    // anchors — and each row against its truncated digest in the page,
    // before anything is returned or cached. Cached rows are already
    // verified, so repeat hits cost nothing.
    async _fetchRowsVerified(ids, options = {}) {
        const defaultGap = this.source.preferredGapBytes === undefined ? 2048 : this.source.preferredGapBytes;
        const gap = Math.max(0, options.gap === undefined ? defaultGap : options.gap);
        const preferredParallelism = this.source.preferredParallelism === undefined ? 32 : this.source.preferredParallelism;
        const requestedParallelism = options.parallelism === undefined ? preferredParallelism : options.parallelism;
        const parallelism = requestedParallelism === 0 ? Infinity : Math.max(1, Math.trunc(requestedParallelism || 32));
        const rows = new Map();
        const missing = [];
        for (const id of ids) {
            if (rows.has(id)) continue;
            if (!Number.isInteger(id) || id < 0 || id >= this.count) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'row id out of range', { id });
            }
            const cached = this.cachedRow(id);
            if (cached !== undefined) rows.set(id, cached);
            else {
                rows.set(id, null);
                missing.push(id);
            }
        }
        if (!missing.length) return rows;
        missing.sort((a, b) => a - b);
        const { dim, rowsPerBlock, rowDigestBytes, pageBytes, blockBytes, vectorsOffset } = this;
        const blocks = new Map();
        for (const id of missing) {
            const b = Math.floor(id / rowsPerBlock);
            let entry = blocks.get(b);
            if (!entry) { entry = { maxIdx: 0, ids: [] }; blocks.set(b, entry); }
            entry.ids.push(id);
            const idx = id - b * rowsPerBlock;
            if (idx > entry.maxIdx) entry.maxIdx = idx;
        }
        // Coalesce block spans into runs; a run always holds whole block
        // spans and is split before exceeding maxRangeBytes (floor one block,
        // so a single block is always fetchable).
        const maxRunBytes = Math.max(blockBytes, Math.trunc(options.maxRangeBytes || MAX_COALESCED_RANGE_BYTES));
        const runs = [];
        let run = null;
        for (const b of [...blocks.keys()].sort((x, y) => x - y)) {
            const start = vectorsOffset + b * blockBytes;
            const end = start + pageBytes + (blocks.get(b).maxIdx + 1) * dim;
            if (run && start <= run.end + gap && end - run.start <= maxRunBytes) {
                run.end = end;
                run.blocks.push(b);
            } else {
                if (run) runs.push(run);
                run = { start, end, blocks: [b] };
            }
        }
        runs.push(run);
        const buffers = await mapLimit(runs, parallelism, async (r) => {
            const length = r.end - r.start;
            const bytes = await readChecked(this.source, r.start, length, 'row block');
            if (bytes.byteLength !== length) {
                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact row read returned a truncated range', { offset: r.start, length });
            }
            this.rangeRequests++;
            this.rangeBytes += length;
            return { r, bytes };
        });
        for (const { r, bytes } of buffers) {
            for (const b of r.blocks) {
                const rel = vectorsOffset + b * blockBytes - r.start;
                const page = bytes.subarray(rel, rel + pageBytes);
                if (this.verifyRows) {
                    const pageDigest = await sha256BytesAsync(page);
                    if (!pageDigest) {
                        throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID,
                            'Sketch artifact row verification requested but no crypto backend is available; pass verify:false to skip');
                    }
                    const expected = this.pageTable.subarray(b * 32, b * 32 + 32);
                    for (let i = 0; i < 32; i++) {
                        if (pageDigest[i] !== expected[i]) {
                            throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact row digest page failed hash verification', { block: b });
                        }
                    }
                }
                for (const id of blocks.get(b).ids) {
                    const idx = id - b * rowsPerBlock;
                    const row = new Uint8Array(bytes.subarray(rel + pageBytes + idx * dim, rel + pageBytes + (idx + 1) * dim));
                    if (this.verifyRows) {
                        const digest = await sha256BytesAsync(row);
                        const slot = page.subarray(idx * rowDigestBytes, (idx + 1) * rowDigestBytes);
                        for (let i = 0; i < rowDigestBytes; i++) {
                            if (digest[i] !== slot[i]) {
                                throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, 'Sketch artifact row failed digest verification', { id });
                            }
                        }
                    }
                    if (rows.has(id)) rows.set(id, row);
                    this.cacheRow(id, row);
                }
            }
        }
        return rows;
    }

    async search(queryInput, k, options = {}) {
        // Same query contract as the core engine and the range reader:
        // numeric, finite, right dimension, usable cosine norm (normalized
        // here for cosine; the pooling below then sees a unit vector).
        const query = normalizeQueryVector(queryInput, this.dim, this.metric);
        k = resolveSearchK(k, this.count);
        if (k === 0) return { results: [], stats: this.stats() };
        const tier = this.activeTier();
        // The micro tier is coarser, so its candidate pool defaults wider:
        // an explicit options.rerank always wins; otherwise recommendedRerank
        // is scaled by microBoost while serving from the micro tier. The
        // candidate pool C can never usefully exceed the row count, and it
        // sizes the selection buffers below, so it is bounded by count before
        // anything is allocated — a k or rerank of 1e9 over a small artifact
        // must not try to allocate 1e9 slots.
        const base = resolveOptionalPositiveInt(options.rerank, 'rerank', 0);
        const boost = Math.max(1, Math.trunc(options.microBoost || 4));
        const rec = this.recommendedRerank || Math.max(100, k * 10);
        const C = Math.min(this.count, Math.max(k, base > 0 ? base : tier.name === 'micro' ? rec * boost : rec));
        const { dim, count } = this;
        const tierDims = tier.dims;
        const pool = dim / tierDims;

        let q = query;
        if (this.metric === 1) {
            let norm = 0;
            for (let d = 0; d < dim; d++) norm += query[d] * query[d];
            norm = Math.sqrt(norm);
            if (!(norm > 0) || !Number.isFinite(norm)) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'cosine query must have a nonzero finite norm');
            }
            q = new Float32Array(dim);
            for (let d = 0; d < dim; d++) q[d] = query[d] / norm;
        }
        const qPool = new Float32Array(tierDims);
        for (let sd = 0; sd < tierDims; sd++) {
            let acc = 0;
            for (let j = 0; j < pool; j++) acc += q[sd * pool + j];
            qPool[sd] = acc / pool;
        }

        let ids;
        let scanner = options.scanner || null;
        if (scanner && scanner.sketchDims !== undefined && scanner.sketchDims !== tierDims) scanner = null;
        if (!scanner && options.microScanner && options.microScanner.sketchDims === tierDims) scanner = options.microScanner;
        // A scanner's output buffers are sized at creation (maxRerank); a
        // C beyond them would silently shrink the candidate pool — recall
        // loss, not an error. Fall back to the JS scan for that query.
        if (scanner && scanner.maxRerank !== undefined && C > scanner.maxRerank) scanner = null;
        if (scanner) {
            // A scanner scores the resident sketches itself, so it must
            // implement this artifact's metric. Cosine requires an explicit
            // declaration: a metric-blind scanner silently loses recall there.
            const scannerMetric = scanner.metric;
            if (scannerMetric !== undefined && scannerMetric !== this.metric) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'scanner metric does not match artifact metric', { scannerMetric, metric: this.metric });
            }
            if (this.metric === 1 && scannerMetric !== 1) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'cosine sketch artifacts require a metric-aware scanner (scanner.metric === 1)', { metric: this.metric });
            }
            ids = scanner.scan(qPool, C);
        } else if (C >= count) {
            // Every row is a candidate: the exact rerank below scores them
            // all, so the resident scan has nothing to select. Without this
            // the selection loop degenerates to 2*count^2 slot operations
            // (candMax stays Infinity until the last slot fills).
            ids = Array.from({ length: count }, (_, i) => i);
        } else {
            const candDist = new Float64Array(C).fill(Infinity);
            const candId = new Int32Array(C).fill(-1);
            let candMax = Infinity;
            for (let i = 0; i < count; i++) {
                const s = this.scales[i];
                const o = this.offsets[i];
                let metricAcc = 0;
                if (this.metric === 1) {
                    for (let sd = 0; sd < tierDims; sd++) {
                        metricAcc += qPool[sd] * (o + s * tier.value(i, sd));
                    }
                    metricAcc = 1 - Math.max(-1, Math.min(1, metricAcc * pool));
                } else {
                    for (let sd = 0; sd < tierDims; sd++) {
                        const diff = qPool[sd] - (o + s * tier.value(i, sd));
                        metricAcc += diff * diff;
                    }
                }
                if (metricAcc < candMax) {
                    let worst = 0;
                    for (let j = 1; j < C; j++) if (candDist[j] > candDist[worst]) worst = j;
                    candDist[worst] = metricAcc;
                    candId[worst] = i;
                    candMax = 0;
                    for (let j = 0; j < C; j++) if (candDist[j] > candMax) candMax = candDist[j];
                }
            }
            ids = [];
            for (let j = 0; j < C; j++) if (candId[j] >= 0) ids.push(candId[j]);
        }

        // Externally supplied candidates (e.g. a lexical index's matches in
        // a hybrid query) join the exact rerank alongside the sketch scan's
        // selection: they are scored by true distance like any candidate,
        // so a lexically-found row that is also a near neighbor surfaces
        // even when the resident scan's top-C missed it.
        if (options.extraCandidates !== undefined) {
            if (!Array.isArray(options.extraCandidates)) {
                throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'extraCandidates must be an array of row ids');
            }
            if (!Array.isArray(ids)) ids = Array.from(ids); // a custom scanner may return a typed array
            const have = new Set(ids);
            for (const id of options.extraCandidates) {
                if (!Number.isInteger(id) || id < 0 || id >= count) {
                    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'extraCandidates ids must be integers in [0, count)', { id, count });
                }
                if (!have.has(id)) {
                    have.add(id);
                    ids.push(id);
                }
            }
        }

        const rows = await this.fetchRows(ids, options);

        const exact = [];
        for (const id of ids) {
            const row = rows.get(id);
            const s = this.scales[id];
            const o = this.offsets[id];
            let acc = 0;
            if (this.metric === 1) {
                for (let d = 0; d < dim; d++) acc += q[d] * (o + s * row[d]);
                acc = 1 - Math.max(-1, Math.min(1, acc));
            } else {
                for (let d = 0; d < dim; d++) {
                    const diff = q[d] - (o + s * row[d]);
                    acc += diff * diff;
                }
            }
            exact.push([acc, id]);
        }
        exact.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        // Rerank accumulates squared L2; the API contract (README "Distance
        // values") reports Euclidean, matching PikeletIndex.search.
        const sqrtL2 = this.metric !== 1;
        // fullRerankOutput returns every reranked candidate in distance
        // order instead of the top k — the candidates are already fetched
        // and exactly scored, so this changes only the output slice. Hybrid
        // callers use it so rank fusion sees the true distance of every
        // lexical candidate.
        const selected = options.fullRerankOutput ? exact : exact.slice(0, k);
        return {
            results: selected.map(([distance, id]) => ({ id, distance: sqrtL2 ? Math.sqrt(distance) : distance })),
            rerank: ids.length,
            tier: tier.name,
        };
    }

    async close() {
        if (this.source && typeof this.source.close === 'function') await this.source.close();
    }
}

// Build a WASM-backed scanner for a sketch artifact's resident tier, usable
// as the `scanner` option to PikeletSketchArtifact.search(). The engine's
// pikelet_sketch_scan SIMD kernel runs the O(count*sketchDims) resident scan
// that is otherwise the browser query bottleneck. `loadEngine` is the
// entrypoint's own async engine loader (Node/web/workerd all supply one);
// the sketch tier is staged in the engine heap once at creation.
async function createSketchScanner(loadEngine, artifact, options = {}) {
    if (typeof loadEngine !== 'function') {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'createSketchScanner() requires an engine loader');
    }
    if (!artifact || !(artifact instanceof PikeletSketchArtifact)) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'createSketchScanner() requires a sketch artifact');
    }
    const engine = await loadEngine();
    const { count, metric } = artifact;
    const tierName = options.tier === 'micro' ? 'micro' : 'full';
    if (tierName === 'micro' && !artifact.microDims) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'artifact has no micro tier');
    }
    if (tierName === 'full' && !artifact.sketches) {
        throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'full sketches are not resident yet (staged open still in stage 1; await artifact.fullyResident)');
    }
    const sketchDims = tierName === 'micro' ? artifact.microDims : artifact.sketchDims;
    const tierBits = tierName === 'micro' ? artifact.microBits : artifact.sketchBits;
    const maxC = Math.max(1, Math.trunc(options.maxRerank || 1024));

    // The kernel reads u8 sketch values; expand 4-bit nibbles to their
    // reconstructed u8 once so the scan needs no per-element unpacking.
    const expanded = new Uint8Array(count * sketchDims);
    if (tierBits === 4) {
        for (let i = 0; i < count; i++) {
            for (let sd = 0; sd < sketchDims; sd++) expanded[i * sketchDims + sd] = tierName === 'micro' ? artifact.microValue(i, sd) : artifact.sketchValue(i, sd);
        }
    } else {
        expanded.set(tierName === 'micro' ? artifact.microSketches : artifact.sketches);
    }

    const sketchesPtr = engine._emsc_malloc(expanded.length);
    const scalesPtr = engine._emsc_malloc(count * 4);
    const offsetsPtr = engine._emsc_malloc(count * 4);
    const queryPtr = engine._emsc_malloc(sketchDims * 4);
    const outIdsPtr = engine._emsc_malloc(maxC * 4);
    const outDistsPtr = engine._emsc_malloc(maxC * 4);
    if (!sketchesPtr || !scalesPtr || !offsetsPtr || !queryPtr || !outIdsPtr || !outDistsPtr) {
        // Free whichever allocations succeeded before one failed, so a
        // partial failure does not leak WASM heap (matches create()).
        for (const ptr of [sketchesPtr, scalesPtr, offsetsPtr, queryPtr, outIdsPtr, outDistsPtr]) {
            if (ptr) engine._emsc_free(ptr);
        }
        throw pikeletError(PIKELET_ERROR_CODES.WASM_ALLOCATION_FAILED, 'sketch scanner heap allocation failed');
    }
    engine.HEAPU8.set(expanded, sketchesPtr);
    engine.HEAPF32.set(artifact.scales, scalesPtr >> 2);
    engine.HEAPF32.set(artifact.offsets, offsetsPtr >> 2);

    let disposed = false;
    return {
        maxRerank: maxC,
        metric,
        sketchDims,
        tier: tierName,
        scan(pooledQuery, c) {
            if (disposed) throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, 'sketch scanner disposed');
            const query = pooledQuery instanceof Float32Array ? pooledQuery : Float32Array.from(pooledQuery);
            // The query is copied into a queryPtr sized for exactly sketchDims
            // floats: an over- or under-sized input would read/write outside
            // that buffer in the WASM heap. Validate before the copy.
            if (query.length !== sketchDims) {
                throw pikeletError(PIKELET_ERROR_CODES.DIMENSION_MISMATCH,
                    `scan() pooled query has ${query.length} values, expected ${sketchDims}`,
                    { expected: sketchDims, actual: query.length });
            }
            for (let i = 0; i < query.length; i++) {
                if (!Number.isFinite(query[i])) {
                    throw pikeletError(PIKELET_ERROR_CODES.INVALID_VECTOR,
                        'scan() pooled query contains a non-finite value', { index: i });
                }
            }
            engine.HEAPF32.set(query, queryPtr >> 2);
            const topC = Math.min(Math.max(1, Math.trunc(c)), maxC);
            const n = engine._pikelet_sketch_scan(
                sketchesPtr, scalesPtr, offsetsPtr, count, sketchDims, queryPtr, metric, topC, outIdsPtr, outDistsPtr
            );
            return Array.from(engine.HEAPU32.subarray(outIdsPtr >> 2, (outIdsPtr >> 2) + n));
        },
        dispose() {
            if (disposed) return;
            disposed = true;
            for (const ptr of [sketchesPtr, scalesPtr, offsetsPtr, queryPtr, outIdsPtr, outDistsPtr]) {
                engine._emsc_free(ptr);
            }
        },
    };
}

module.exports = {
    PikeletSketchArtifact,
    createSketchScanner,
    buildSketchArtifact,
    buildSketchArtifactBytes,
    buildSketchArtifactFile,
    exportSketchArtifact,
};
