import crypto from 'node:crypto';
import fs from 'node:fs';
import { expectedBlobBytes } from './inline-transformer.mjs';
import { MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KINDS, KIND_NAMES } from './format.mjs';

export { MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KINDS, KIND_NAMES };

// The inline-transformer kernel via the web glue with an explicitly read
// wasm binary — the form that works under both native ESM and jiti's CJS
// transform (the node glue's createRequire bootstrap breaks there). Node
// builders only; the reader picks its own kernel.
export async function loadInlineEncoderKernel() {
  const { default: createEncoderModule } = await import('./encoder-kernels/encoder.mjs');
  const wasmBinary = fs.readFileSync(new URL('./encoder-kernels/encoder.wasm', import.meta.url));
  return () => createEncoderModule({ wasmBinary });
}

export const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
export const align16 = (n) => Math.ceil(n / 16) * 16;

// Canonical form = JSON.stringify semantics with sorted object keys:
// undefined / functions / symbols are omitted as object properties and
// become null as array elements, exactly as JSON.stringify treats them —
// previously they leaked through as the literal text `undefined`, producing
// output JSON.parse rejects.
export function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map((v) => canonicalJson(v) ?? 'null').join(',')}]`;
  }
  if (value && typeof value === 'object') {
    const parts = [];
    for (const k of Object.keys(value).sort()) {
      const serialized = canonicalJson(value[k]);
      if (serialized !== undefined) parts.push(`${JSON.stringify(k)}:${serialized}`);
    }
    return `{${parts.join(',')}}`;
  }
  return JSON.stringify(value);
}

export function buildQueryInterpSegment(kind, encoderBytes, calibrationBytes) {
  const out = Buffer.alloc(16 + encoderBytes.length + calibrationBytes.length);
  out.writeUInt32LE(1, 0);
  out.writeUInt32LE(kind, 4);
  out.writeUInt32LE(encoderBytes.length, 8);
  out.writeUInt32LE(calibrationBytes.length, 12);
  out.set(encoderBytes, 16); // set(), not copy(): inputs may be plain Uint8Arrays
  out.set(calibrationBytes, 16 + encoderBytes.length);
  return out;
}

// Lexical index segment (kind 5, OPTIONAL), layout bm25-v1: a static
// inverted index over the corpus records for hybrid retrieval. Terms are
// addressed by a 64-bit hash (two fnv1a-32 passes, seeds 0 and 0x9e3779b9)
// in a fixed-width table sorted by (hashHi, hashLo), so a reader can binary
// search without materializing term strings — lazy-friendly by
// construction, even though the phase-1 reader loads the segment eagerly.
// Postings are (docIdDelta, tf) varint pairs. Document length (indexed
// tokens per record) and the corpus token total ship for BM25
// normalization. Hash collisions merge two terms' postings; at 64 bits over
// realistic vocabularies the probability is negligible and the failure mode
// is one slightly-wrong lexical score, not corruption.
//
//   [0,4)   u32 version = 1
//   [4,8)   u32 docCount
//   [8,12)  u32 termCount
//   [12,20) u64 totalTokens
//   [20,24) u32 doclenOffset          absolute within the segment
//   [24,28) u32 termTableOffset
//   [28,32) u32 postingsOffset
//   [32,40) u64 postingsBytes
//   [40,64) reserved, zero
//   doclens   u32[docCount]
//   termTable termCount x 24 bytes: u32 hashLo, u32 hashHi,
//             u64 postingsOffset (relative to postingsOffset), u32 bytes,
//             u32 df
//   postings  per term: varint docId (first absolute, then deltas),
//             varint tf, repeated df times
//
// Indexing tokenizes on [a-z0-9']+ lowercased, keeps tokens of 2..32 chars,
// and skips a small function-word stopword list — readers need no
// coordination with the list, because an unindexed query token simply finds
// no postings.
const LEXICAL_LAYOUT = 'bm25-v1';
const LEXICAL_STOPWORDS = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'into', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these', 'those', 'as', 'their', 'they', 'them', 'his', 'her', 'has', 'have', 'had', 'not', 'can', 'will', 'if', 'we', 'you', 'your', 'i']);
const lexTokenize = (text) => (String(text).toLowerCase().match(/[a-z0-9']+/g) || [])
  .filter((w) => w.length >= 2 && w.length <= 32 && !LEXICAL_STOPWORDS.has(w));

function fnv1a32(str, seed) {
  let h = 0x811c9dc5 ^ seed;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function varintBytes(n) {
  const out = [];
  let v = n >>> 0;
  do {
    let b = v & 0x7f;
    v >>>= 7;
    if (v) b |= 0x80;
    out.push(b);
  } while (v);
  return out;
}

// Counting-sort construction over typed arrays: the obvious
// Map-of-Maps build dies around tens of millions of posting entries
// (a 456k-chunk wiki corpus has ~30M), so terms are interned to dense
// indexes on a first tokenize pass (df counted per term), entry arrays
// are preallocated exactly, and a second tokenize pass scatters
// (docId, tf) pairs into per-term slots. Entries land doc-ascending per
// term by construction, which is what the delta varints need. Output
// bytes are identical to the original builder's.
export function buildLexicalSegment(texts) {
  const docCount = texts.length;
  const doclens = new Uint32Array(docCount);
  const termIndex = new Map(); // "hashHi:hashLo" -> dense index
  const hashLoArr = [];
  const hashHiArr = [];
  const dfArr = [];
  let totalTokens = 0;
  let entryCount = 0;

  const tallyDoc = (id, onTerm) => {
    const tokens = lexTokenize(texts[id]);
    const tf = new Map(); // dense term index -> tf within this doc
    for (const t of tokens) {
      const key = `${fnv1a32(t, 0x9e3779b9)}:${fnv1a32(t, 0)}`;
      let idx = termIndex.get(key);
      if (idx === undefined) {
        idx = hashLoArr.length;
        termIndex.set(key, idx);
        hashLoArr.push(fnv1a32(t, 0));
        hashHiArr.push(fnv1a32(t, 0x9e3779b9));
        dfArr.push(0);
      }
      tf.set(idx, (tf.get(idx) || 0) + 1);
    }
    for (const [idx, count] of tf) onTerm(idx, count);
    return tokens.length;
  };

  for (let id = 0; id < docCount; id++) {
    const tokens = tallyDoc(id, (idx) => { dfArr[idx] += 1; entryCount += 1; });
    doclens[id] = tokens;
    totalTokens += tokens;
  }

  const termCount = hashLoArr.length;
  // Per-term slot offsets in the entry arrays (prefix sums of df).
  const slotStart = new Uint32Array(termCount + 1);
  for (let i = 0; i < termCount; i++) slotStart[i + 1] = slotStart[i] + dfArr[i];
  const cursor = slotStart.slice(0, termCount);
  const eDoc = new Uint32Array(entryCount);
  const eTf = new Uint32Array(entryCount);
  for (let id = 0; id < docCount; id++) {
    tallyDoc(id, (idx, tf) => {
      const at = cursor[idx]++;
      eDoc[at] = id;
      eTf[at] = tf;
    });
  }

  // Serialize in (hashHi, hashLo) order. Sizes first, then one fill pass.
  const order = Array.from({ length: termCount }, (_, i) => i)
    .sort((a, b) => (hashHiArr[a] - hashHiArr[b]) || (hashLoArr[a] - hashLoArr[b]));
  const varintLen = (n) => (n < 0x80 ? 1 : n < 0x4000 ? 2 : n < 0x200000 ? 3 : n < 0x10000000 ? 4 : 5);
  let postingsCursor = 0;
  const termPostingsOffset = new Float64Array(termCount); // by sorted position
  const termPostingsBytes = new Uint32Array(termCount);
  order.forEach((idx, pos) => {
    let bytes = 0;
    let prev = 0;
    for (let at = slotStart[idx]; at < slotStart[idx + 1]; at++) {
      bytes += varintLen(at === slotStart[idx] ? eDoc[at] : eDoc[at] - prev) + varintLen(eTf[at]);
      prev = eDoc[at];
    }
    termPostingsOffset[pos] = postingsCursor;
    termPostingsBytes[pos] = bytes;
    postingsCursor += bytes;
  });

  const doclenOffset = 64;
  const termTableOffset = doclenOffset + 4 * docCount;
  const postingsOffset = termTableOffset + 24 * termCount;
  const out = Buffer.alloc(postingsOffset + postingsCursor);
  out.writeUInt32LE(1, 0);
  out.writeUInt32LE(docCount, 4);
  out.writeUInt32LE(termCount, 8);
  out.writeBigUInt64LE(BigInt(totalTokens), 12);
  out.writeUInt32LE(doclenOffset, 20);
  out.writeUInt32LE(termTableOffset, 24);
  out.writeUInt32LE(postingsOffset, 28);
  out.writeBigUInt64LE(BigInt(postingsCursor), 32);
  for (let id = 0; id < docCount; id++) out.writeUInt32LE(doclens[id], doclenOffset + 4 * id);
  const writeVarint = (value, at) => {
    let v = value >>> 0;
    do {
      let b = v & 0x7f;
      v >>>= 7;
      if (v) b |= 0x80;
      out[at++] = b;
    } while (v);
    return at;
  };
  order.forEach((idx, pos) => {
    const entry = termTableOffset + 24 * pos;
    out.writeUInt32LE(hashLoArr[idx], entry);
    out.writeUInt32LE(hashHiArr[idx], entry + 4);
    out.writeBigUInt64LE(BigInt(termPostingsOffset[pos]), entry + 8);
    out.writeUInt32LE(termPostingsBytes[pos], entry + 16);
    out.writeUInt32LE(dfArr[idx], entry + 20);
    let at = postingsOffset + termPostingsOffset[pos];
    let prev = 0;
    for (let slot = slotStart[idx]; slot < slotStart[idx + 1]; slot++) {
      at = writeVarint(slot === slotStart[idx] ? eDoc[slot] : eDoc[slot] - prev, at);
      at = writeVarint(eTf[slot], at);
      prev = eDoc[slot];
    }
  });
  return {
    bytes: out,
    meta: { layout: LEXICAL_LAYOUT, terms: termCount, docCount, totalTokens },
  };
}

// Corpus layout v1 (format 1, profile pikelet-complete-v1): count + offsets +
// records, integrity by whole-segment digest only. Kept for producing
// format-1 files (compatibility fixtures, readers that predate format 2);
// new artifacts use buildCorpusSegment() below.
export function buildCorpusSegmentFromBuffers(records) {
  const count = records.length;
  const prefix = 4 + 8 * (count + 1);
  const total = prefix + records.reduce((sum, b) => sum + b.length, 0);
  const out = Buffer.alloc(total);
  out.writeUInt32LE(count, 0);
  let cursor = prefix;
  for (let id = 0; id < count; id++) {
    out.writeBigUInt64LE(BigInt(cursor), 4 + 8 * id);
    out.set(records[id], cursor); // set(), not copy(): records may be plain Uint8Arrays
    cursor += records[id].length;
  }
  out.writeBigUInt64LE(BigInt(cursor), 4 + 8 * count);
  return out;
}

// Corpus layout v2 (format 2, profile pikelet-complete-v2): per-record
// integrity that a reader can check on the single range read that hydrates
// a record, without fetching the rest of the segment (contract 4.1 / 7):
//
//   [0, 4)                 u32 count
//   [4, 8)                 u32 pageRecords (P)
//   [8, 8 + 8*(count+1))   u64 offsets[count+1]   record i = [offsets[i], offsets[i+1])
//   [A, A + 32*pages)      pageSha256[pages]      pages = ceil(count / P); page p is
//                                                 the digest of recordSha256[p*P .. min(count,(p+1)*P))
//   [B, B + 32*count)      recordSha256[count]    digest of each record's bytes
//   [C, ...)               records                offsets[0] == C
//
// The manifest carries pageTableSha256 (digest of the page table bytes), so
// the chain is identity -> page table (read at open, 32 bytes per P
// records) -> one page of record digests (read when a record in it is first
// hydrated) -> the record. Returns the segment bytes plus the manifest's
// corpus fields; callers spread them under manifest.corpus.
export const CORPUS_LAYOUT_V2 = 'records-v2';
export const DEFAULT_CORPUS_PAGE_RECORDS = 256;
export function buildCorpusSegment(records, options = {}) {
  const count = records.length;
  const pageRecords = options.pageRecords ?? DEFAULT_CORPUS_PAGE_RECORDS;
  if (!Number.isInteger(pageRecords) || pageRecords < 1 || pageRecords > 65536) {
    throw new Error('pageRecords must be an integer in [1, 65536]');
  }
  const pages = Math.ceil(count / pageRecords);
  const offsetsAt = 8;
  const pageTableAt = offsetsAt + 8 * (count + 1);
  const recordDigestsAt = pageTableAt + 32 * pages;
  const recordsAt = recordDigestsAt + 32 * count;
  const total = recordsAt + records.reduce((sum, b) => sum + b.length, 0);
  const out = Buffer.alloc(total);
  out.writeUInt32LE(count, 0);
  out.writeUInt32LE(pageRecords, 4);
  let cursor = recordsAt;
  for (let id = 0; id < count; id++) {
    out.writeBigUInt64LE(BigInt(cursor), offsetsAt + 8 * id);
    out.set(records[id], cursor); // set(), not copy(): records may be plain Uint8Arrays
    out.set(sha256(records[id]), recordDigestsAt + 32 * id);
    cursor += records[id].length;
  }
  out.writeBigUInt64LE(BigInt(cursor), offsetsAt + 8 * count);
  for (let p = 0; p < pages; p++) {
    const from = recordDigestsAt + 32 * p * pageRecords;
    const to = recordDigestsAt + 32 * Math.min(count, (p + 1) * pageRecords);
    sha256(out.subarray(from, to)).copy(out, pageTableAt + 32 * p);
  }
  const pageTableSha256 = sha256(out.subarray(pageTableAt, recordDigestsAt)).toString('hex');
  return {
    bytes: out,
    corpus: {
      records: count,
      layout: CORPUS_LAYOUT_V2,
      pageRecords,
      pages,
      recordDigest: 'sha256',
      pageTableSha256,
    },
  };
}

export function buildInlineTransformerEncoderSegment({ declaration, vocabBytes, weightBytes }) {
  if (declaration.layout) {
    const expected = expectedBlobBytes(declaration.layout);
    if (weightBytes.length !== expected) {
      throw new Error(`inline-encoder weights are ${weightBytes.length} bytes but the declared layout implies ${expected}`);
    }
  }
  const declarationBytes = Buffer.from(canonicalJson(declaration), 'utf8');
  const out = Buffer.alloc(12 + declarationBytes.length + vocabBytes.length + weightBytes.length);
  out.writeUInt32LE(declarationBytes.length, 0);
  out.writeUInt32LE(vocabBytes.length, 4);
  out.writeUInt32LE(weightBytes.length, 8);
  out.set(declarationBytes, 12);
  out.set(vocabBytes, 12 + declarationBytes.length); // set(), not copy(): inputs may be plain Uint8Arrays
  out.set(weightBytes, 12 + declarationBytes.length + vocabBytes.length);
  return out;
}

// The spec's measured operating point (SKETCH_PROFILE.md section 5, mandatory
// in the complete profile's evaluation segment): open the just-built sketch
// bytes in memory, replay a query set at increasing rerank C against the
// exact full-rerank top-k, and return the smallest C whose recall reaches
// targetRecall — plus the whole curve for the evaluation segment. Queries
// are the corpus's own float embeddings when the caller has them, else rows
// dequantized out of the snapshot.
export async function measureRecommendedRerank({
  artifactModule,
  sketchBytes,
  queryVectors = null,
  snapshotBytes = null,
  k = 10,
  targetRecall = 0.98,
  maxQueries = 256,
  sweep = null,
}) {
  const bytes = sketchBytes instanceof Uint8Array ? sketchBytes : new Uint8Array(sketchBytes);
  const source = {
    size: bytes.length,
    preferredParallelism: Infinity,
    preferredGapBytes: 0,
    async read(offset, length) { return bytes.subarray(offset, offset + length); },
    async close() {},
  };
  const sketch = await artifactModule.PikeletSketchArtifact.open(source);
  try {
    const count = sketch.count;
    if (count === 0) throw new Error('cannot measure rerank on an empty sketch');
    const topK = Math.min(k, count);
    let queries;
    let querySource;
    if (queryVectors && queryVectors.length > 0) {
      queries = sampleEvenly(queryVectors, maxQueries);
      querySource = 'corpus-embeddings';
    } else {
      queries = selfQueriesFromSnapshot(artifactModule.parseUint8Snapshot(snapshotBytes), maxQueries);
      querySource = 'dequantized-rows';
    }
    const truth = [];
    for (const q of queries) {
      truth.push(new Set((await sketch.search(q, topK, { rerank: count })).results.map((r) => r.id)));
    }
    const ladder = (sweep || [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640])
      .filter((c) => c >= topK && c < count);
    ladder.push(count);
    const curve = [];
    let recommendedRerank = null;
    let recommendedRecall = null;
    for (const C of ladder) {
      let hits = 0;
      let total = 0;
      for (let i = 0; i < queries.length; i++) {
        for (const r of (await sketch.search(queries[i], topK, { rerank: C })).results) {
          if (truth[i].has(r.id)) hits++;
        }
        total += truth[i].size;
      }
      const recall = total > 0 ? Number((hits / total).toFixed(4)) : 1;
      curve.push({ rerank: C, recall });
      if (recommendedRerank === null && recall >= targetRecall) {
        recommendedRerank = C;
        recommendedRecall = recall;
      }
    }
    if (recommendedRerank === null) {
      recommendedRerank = count;
      recommendedRecall = curve[curve.length - 1].recall;
    }
    return {
      recommendedRerank,
      recall: recommendedRecall,
      k: topK,
      targetRecall,
      queries: queries.length,
      querySource,
      curve,
    };
  } finally {
    await sketch.close();
  }
}

function sampleEvenly(items, max) {
  if (items.length <= max) return items;
  const stride = items.length / max;
  const out = [];
  for (let i = 0; i < max; i++) out.push(items[Math.floor(i * stride)]);
  return out;
}

function selfQueriesFromSnapshot(graph, maxQueries) {
  const n = Math.min(graph.count, maxQueries);
  const stride = graph.count / n;
  const queries = [];
  for (let i = 0; i < n; i++) {
    const id = Math.floor(i * stride);
    const v = new Float32Array(graph.dim);
    const base = id * graph.dim;
    for (let d = 0; d < graph.dim; d++) v[d] = graph.offsets[id] + graph.scales[id] * graph.qdata[base + d];
    queries.push(v);
  }
  return queries;
}

// Container format version follows the profile string: pikelet-complete-v1
// files carry corpus layout v1 and header version 1; pikelet-complete-v2
// files carry corpus layout records-v2 (buildCorpusSegment) and header
// version 2, so readers that predate per-record integrity reject them
// explicitly instead of misreading the corpus tables.
export const PROFILE_V1 = 'pikelet-complete-v1';
export const PROFILE_V2 = 'pikelet-complete-v2';
export const FORMAT_VERSIONS = { [PROFILE_V1]: 1, [PROFILE_V2]: 2 };

export function assemblePikeletFile(manifestFields, segments, outPath) {
  const formatVersion = FORMAT_VERSIONS[manifestFields.profile];
  if (!formatVersion) {
    throw new Error(`manifest.profile must be ${PROFILE_V1} or ${PROFILE_V2}, got ${manifestFields.profile}`);
  }
  const layout = manifestFields.corpus?.layout;
  if (formatVersion === 2 && layout !== CORPUS_LAYOUT_V2) {
    throw new Error(`${PROFILE_V2} requires manifest.corpus from buildCorpusSegment() (layout ${CORPUS_LAYOUT_V2})`);
  }
  if (formatVersion === 1 && layout !== undefined) {
    throw new Error(`${PROFILE_V1} carries corpus layout v1 (buildCorpusSegmentFromBuffers); got layout ${layout}`);
  }
  if (!Number.isInteger(manifestFields.corpus?.records) || manifestFields.corpus.records < 0) {
    throw new Error('manifest.corpus.records must be a non-negative integer');
  }
  // Format 2 additionally commits to the embedded sketch's 256-byte header
  // under the identity. The header carries the sketch's metric/dim/count and
  // its residentSha256 / vectorsSha256, so anchoring the header anchors the
  // sketch's own integrity chain to the complete identity without forcing a
  // whole-segment read at open (the index segment stays lazy).
  let indexCommitment;
  if (formatVersion >= 2) {
    const index = segments.find((s) => s.kind === 'index');
    if (!index || index.bytes.length < 256) {
      throw new Error('index segment must carry a 256-byte sketch header');
    }
    indexCommitment = { ...manifestFields.index, headerSha256: sha256(index.bytes.subarray(0, 256)).toString('hex') };
  }
  const manifest = {
    ...manifestFields,
    ...(indexCommitment ? { index: indexCommitment } : {}),
    segments: segments.map((s) => ({
      kind: s.kind,
      sha256: sha256(s.bytes).toString('hex'),
      bytes: s.bytes.length,
    })),
  };
  const manifestBytes = Buffer.from(canonicalJson(manifest), 'utf8');
  const identity = sha256(manifestBytes);

  const tableOffset = HEADER_BYTES + manifestBytes.length;
  let cursor = align16(tableOffset + segments.length * TABLE_ENTRY_BYTES);
  const table = Buffer.alloc(segments.length * TABLE_ENTRY_BYTES);
  const placed = [];
  for (let i = 0; i < segments.length; i++) {
    const entry = i * TABLE_ENTRY_BYTES;
    // Known kinds map by name; a producer may carry an extra segment under
    // a name this spec revision does not define by giving its numeric kind
    // explicitly (readers skip unknown kinds, spec 3.3).
    const kindNumber = KINDS[segments[i].kind] ?? segments[i].kindNumber;
    if (!Number.isInteger(kindNumber) || kindNumber < 1 || kindNumber > 0xffffffff) {
      throw new Error(`segment ${i} (${segments[i].kind}) has no kind number`);
    }
    table.writeUInt32LE(kindNumber, entry);
    table.writeBigUInt64LE(BigInt(cursor), entry + 8);
    table.writeBigUInt64LE(BigInt(segments[i].bytes.length), entry + 16);
    placed.push({ offset: cursor, bytes: segments[i].bytes });
    cursor = align16(cursor + segments[i].bytes.length);
  }
  const last = placed[placed.length - 1];
  const fileBytes = last.offset + last.bytes.length;

  const header = Buffer.alloc(HEADER_BYTES);
  header.writeUInt32LE(MAGIC, 0);
  header.writeUInt32LE(formatVersion, 4);
  header.writeUInt32LE(manifestBytes.length, 8);
  header.writeUInt32LE(segments.length, 12);
  header.writeBigUInt64LE(BigInt(fileBytes), 16);
  identity.copy(header, 24);

  const fd = fs.openSync(outPath, 'w');
  try {
    fs.writeSync(fd, header, 0, header.length, 0);
    fs.writeSync(fd, manifestBytes, 0, manifestBytes.length, HEADER_BYTES);
    fs.writeSync(fd, table, 0, table.length, tableOffset);
    for (const { offset, bytes } of placed) {
      fs.writeSync(fd, bytes, 0, bytes.length, offset);
    }
    fs.ftruncateSync(fd, fileBytes);
  } finally {
    fs.closeSync(fd);
  }
  return { outPath, fileBytes, formatVersion, identity: identity.toString('hex'), manifest };
}
