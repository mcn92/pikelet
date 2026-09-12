/**
 * pikelet-wasm/complete/builder — Node-only assembly of complete .pikelet
 * artifacts (spec/COMPLETE_PROFILE.md). The wire-format constants here are
 * the same objects the reader uses (complete/format.mjs).
 */

export declare const MAGIC: number;
export declare const HEADER_BYTES: number;
export declare const TABLE_ENTRY_BYTES: number;
export declare const KINDS: Record<string, number>;
export declare const KIND_NAMES: Record<number, string>;

export declare function sha256(bytes: Uint8Array | Buffer): Buffer;
export declare function canonicalJson(value: unknown): string;
export declare const align16: (n: number) => number;

/** Manifest profile strings and the header formatVersion each one implies. */
export declare const PROFILE_V1: 'pikelet-complete-v1';
export declare const PROFILE_V2: 'pikelet-complete-v2';
export declare const FORMAT_VERSIONS: Record<string, number>;
/** Corpus layout name for format 2 (per-record digests behind a page table). */
export declare const CORPUS_LAYOUT_V2: 'records-v2';
export declare const DEFAULT_CORPUS_PAGE_RECORDS: number;

export declare function buildQueryInterpSegment(kind: number, encoderBytes: Buffer | Uint8Array, calibrationBytes: Buffer | Uint8Array): Buffer;
/** Corpus layout v1 (format 1): count + offsets + records, whole-segment integrity only. */
export declare function buildCorpusSegmentFromBuffers(records: Array<Buffer | Uint8Array>): Buffer;
/**
 * Corpus layout v2 (format 2): count, pageRecords, offsets, a page table of
 * SHA-256s over runs of record digests, one SHA-256 per record, then the
 * records. Spread the returned `corpus` block into manifest.corpus.
 */
export declare function buildCorpusSegment(records: Array<Buffer | Uint8Array>, options?: { pageRecords?: number }): {
  bytes: Buffer;
  corpus: {
    records: number;
    layout: 'records-v2';
    pageRecords: number;
    pages: number;
    recordDigest: 'sha256';
    pageTableSha256: string;
  };
};
export declare function buildInlineTransformerEncoderSegment(input: {
  declaration: Record<string, unknown>;
  vocabBytes: Buffer | Uint8Array;
  weightBytes: Buffer | Uint8Array;
}): Buffer;
/**
 * Writes the container. manifestFields.profile selects the header format
 * version (PROFILE_V1 -> 1 with a v1 corpus, PROFILE_V2 -> 2 with a
 * buildCorpusSegment() corpus); mismatches throw. Segments with a kind this
 * spec does not name must carry an explicit kindNumber (readers skip them).
 */
export declare function assemblePikeletFile(
  manifestFields: Record<string, unknown> & { profile: string; corpus: { records: number } & Record<string, unknown> },
  segments: Array<{ kind: string; bytes: Buffer | Uint8Array; kindNumber?: number }>,
  outPath: string
): { outPath: string; fileBytes: number; formatVersion: number; identity: string; manifest: Record<string, unknown> };

/** Measured recall-vs-rerank operating point (SKETCH_PROFILE.md section 5). */
export declare function measureRecommendedRerank(input: {
  artifactModule: unknown;
  sketchBytes: Uint8Array | ArrayBufferLike;
  queryVectors?: Float32Array[] | null;
  snapshotBytes?: Uint8Array | ArrayBufferLike | null;
  k?: number;
  targetRecall?: number;
  maxQueries?: number;
  sweep?: number[] | null;
}): Promise<{
  recommendedRerank: number;
  recall: number;
  k: number;
  targetRecall: number;
  queries: number;
  querySource: string;
  curve: Array<{ rerank: number; recall: number }>;
}>;

/** Kind-3 kernel via the web glue with an explicit wasm binary (jiti-safe). */
export declare function loadInlineEncoderKernel(): Promise<() => unknown>;
