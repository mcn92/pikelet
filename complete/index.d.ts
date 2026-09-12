/**
 * pikelet-wasm/complete — reader for complete .pikelet artifacts
 * (spec/COMPLETE_PROFILE.md, profile pikelet-complete-v1). Opens any kind
 * (1 student-inline, 2 declared-external, 3 inline-transformer) from a file
 * path in Node or a range source anywhere.
 */

/** A byte-range source: local file, HTTP, memory, or anything else. */
export interface CompleteRangeSource {
  /** Total bytes, when known. Verified against the header when present. */
  size?: number;
  preferredParallelism?: number;
  preferredGapBytes?: number;
  read(offset: number, length: number): Promise<Uint8Array | ArrayBufferLike>;
  close?(): Promise<void>;
}

export interface HttpRangeSourceStats {
  requests: number;
  bytes: number;
  acceptRanges: string | null;
  etag: string | null;
  fullFallback: boolean;
  /** Transient-status (429/502/503/504) retries issued, across all reads. */
  retries: number;
  /**
   * Redirect hops resolved while pinning the final URL. A redirecting host
   * (GitHub release assets) is resolved once and range reads go straight
   * to the target, so its rate-limited front door is charged per mount,
   * not per read; expired signed URLs (401/403) re-resolve automatically.
   */
  redirects: number;
}

export interface HttpRangeSource extends CompleteRangeSource {
  stats: HttpRangeSourceStats;
  init(): Promise<void>;
}

export declare function httpRangeSource(url: string, options?: {
  preferredParallelism?: number;
  preferredGapBytes?: number;
  /** Cap on the one-time full download when a host ignores Range, enforced on received bytes. Default 64 MiB. */
  maxFullFallbackBytes?: number;
  /**
   * Query parameter appended per range read so each read gets its own HTTP
   * cache key (defeats Chromium's same-URL cache-entry lock). Default 'r';
   * pass null for hosts that sign the full query string (presigned URLs) —
   * range reads then share the unmodified URL.
   */
  cacheKeyParam?: string | null | false;
  /**
   * Retries per read for transient statuses (429/502/503/504), with capped
   * exponential backoff (Retry-After honored when sane). Default 6; 0
   * disables. CDNs rate-limit the parallel range bursts a query issues.
   */
  maxRetries?: number;
}): HttpRangeSource;

export interface CompleteQueryResult {
  matchQuality: 'strong' | 'weak' | 'none' | 'unscored';
  confidence?: number;
  /**
   * Each result is the hydrated corpus record plus the search's `id` and
   * `distance`. Those two names are reserved: they always come from the
   * search, and a record field of the same name is overwritten, never the
   * other way round.
   */
  results: Array<{ id: number; distance: number } & Record<string, unknown>>;
}

export interface CompleteSearch {
  info(): {
    identity: string;
    /** Container header formatVersion (1 or 2). */
    formatVersion: number;
    /** Manifest profile: 'pikelet-complete-v1' | 'pikelet-complete-v2'. */
    profile: string;
    /**
     * Human-readable pack name from the identity-verified manifest
     * (compile records it under corpus.provenance.name); null when the
     * builder recorded none.
     */
    name: string | null;
    /**
     * Content license recorded at compile (SPDX id or free text) under
     * corpus.provenance.license; null when the builder recorded none.
     */
    license: string | null;
    records: number;
    dim: number;
    metric: string | number;
    encoder: Record<string, unknown>;
    /**
     * Kind 2: true when the host encoder passed the declaration's verification
     * vectors, false when served unverified (verifyEncoder:false or
     * allowUnverifiedEncoder), null when no host encoder was supplied.
     * Kind 3: whether the inline encoder verified against its own vectors —
     * null while a lazily-opened encoder has not arrived yet (unknown, not
     * unverified; it resolves before the first query completes). While
     * deferred, `encoder` serves the identity-verified manifest declaration
     * with `deferred: true`, replaced by the full declaration on arrival.
     * Kind 1: null (inline student encoder, nothing external to verify).
     */
    encoderVerified: boolean | null;
    /**
     * 'per-record-sha256' on format 2 (each hydrated record verified on its
     * own range read); 'segment-sha256' on format 1 (whole-segment digests
     * only, lazy record reads not independently verifiable).
     */
    corpusIntegrity: string;
    fileBytes: number;
    residentBytes: number;
    residentVerified: boolean;
    /**
     * True only after a full vectors pass (verifyIndexVectors at open, or
     * verifyVectors() later). Until then, lazily fetched rerank rows are
     * committed via the identity-anchored vectorsSha256 but not verified
     * per read.
     */
    vectorsVerified: boolean;
    /**
     * Present when the artifact carries a lexical index segment (kind 5):
     * queries run hybrid (BM25 candidates join the exact rerank; result
     * order fuses distance and BM25 rankings by reciprocal rank). Null on
     * vector-only artifacts.
     */
    lexical: { terms: number; docCount: number; lazy?: true } | null;
    /**
     * 'engine' once a WASM scan kernel serves the resident scan
     * (auto-staged or injected via options.sketchScanner); 'js' before
     * background staging resolves and forever if it fails or was disabled.
     */
    residentScan: 'engine' | 'js';
    sampleQueries: string[];
  };
  /**
   * Verify the index's lazy vector rows against the identity-anchored
   * vectorsSha256 (streamed where a streaming hash exists). Resolves true
   * on match; throws on mismatch.
   */
  verifyVectors(options?: { chunkBytes?: number }): Promise<true>;
  query(text: string, options?: {
    k?: number;
    /**
     * Retrieval mode on artifacts carrying a lexical segment: 'hybrid'
     * (default — BM25 candidates join the rerank, RRF result order),
     * 'vector' (ignore the lexical index), 'lexical' (BM25 ranking only),
     * 'augmented' (BM25 candidates join the exact rerank but results stay
     * in pure distance order — the mode for workloads scored against
     * exact nearest neighbors, where RRF's reordering reads as loss by
     * definition; measured on the 456k wiki eval it lifts recall@10 from
     * 82.8% to 85.3% while RRF-ordered hybrid scores 45.9% on that
     * metric). Abstention scores identically in every mode.
     */
    retrieval?: 'hybrid' | 'vector' | 'lexical' | 'augmented';
    rerank?: number;
    parallelism?: number;
    gap?: number;
    maxRangeBytes?: number;
    /**
     * A matchQuality: 'none' verdict withholds results by default. Set
     * true to see the raw retrieval anyway (e.g. to inspect why the
     * classifier abstained) — matchQuality and confidence are unaffected,
     * this only controls whether results is populated. Default false.
     */
    showAbstained?: boolean;
  }): Promise<CompleteQueryResult>;
  /** Hydrate one corpus record by id (verified per record on format 2). */
  record(id: number): Promise<Record<string, unknown>>;
  /** The evaluation segment's JSON object, or null when the artifact carries none. */
  evaluation(): Promise<Record<string, unknown> | null>;
  close(): Promise<void>;
}

export declare function openPikeletFile(
  input: string | CompleteRangeSource,
  options?: {
    /**
     * Required to serve kind-2 (declared external encoder) artifacts. It is
     * run against the declaration's verification vectors at open and the
     * open fails if any disagree beyond the declared tolerance.
     */
    encodeQuery?: (text: string) => Promise<Float32Array | number[]> | Float32Array | number[];
    /** Skip encoder verification (kind 2 host vectors, kind 3 inline vectors). Default true. */
    verifyEncoder?: boolean;
    /** Serve a kind-2 declaration that carries no verification vectors, marked unverified. Default false. */
    allowUnverifiedEncoder?: boolean;
    /** Verify each hydrated record against its digest on format 2. Default true. */
    verifyRecords?: boolean;
    /**
     * Pin the expected manifest identity (sha256 hex). Checked against the
     * header one 64-byte read in — a mismatched artifact is refused before
     * anything else is fetched; the manifest-hash verification then binds
     * the header's claim to the manifest bytes as always.
     */
    expectedIdentity?: string;
    /**
     * Kind-3 artifacts open lazily: the ~25 MiB inline-encoder region is not
     * read at open, and by default starts downloading in the background the
     * moment open resolves (the first query awaits it; the whole segment is
     * digest-verified on arrival, and the header/calibration slices used at
     * open are byte-compared against it before any result is returned).
     * Set false to defer the transfer entirely to the first query. Default
     * true.
     */
    prefetchEncoder?: boolean;
    /**
     * Fully verify the index's lazy vector rows at open (one streamed pass
     * against the identity-anchored vectorsSha256). Default false: without
     * it, rows that feed reranking are covered by the whole-segment
     * commitment but not verified per read until verifyVectors() runs —
     * the documented transitional stance (spec section 6).
     */
    verifyIndexVectors?: boolean;
    /**
     * Budget for any single open-path read (manifest, segment table,
     * query-interp, corpus tables, evaluation, digest pages), honored
     * strictly. Default 256 MiB; Infinity defers to the 2 GiB backstop.
     */
    maxReadBytes?: number;
    /** Budget for a single corpus record read. Default 16 MiB. */
    maxRecordBytes?: number;
    /** Override the kind-3 wasm kernel loader. */
    createEncoder?: () => Promise<unknown> | unknown;
    /**
     * Resident-scan acceleration. Default (undefined): in Node, when the
     * corpus is large enough that the pure-JS sketch scan dominates query
     * time (~16M sketch cells), the engine's SIMD scan kernel is staged in
     * the background and serves queries once ready — the JS scan answers
     * identically until then and permanently if staging fails. Pass false
     * to force the JS scan; a ready scanner object (createSketchScanner's
     * shape — caller-owned, not disposed by close()); or an async factory
     * (sketchArtifact) => scanner, invoked once the sketch is fully
     * resident (the injection point for browser bundles, which cannot load
     * the Node engine entrypoint).
     */
    sketchScanner?: false | { scan: (pooledQuery: unknown, c: number) => unknown }
      | ((sketch: unknown) => unknown | Promise<unknown>);
    rerankParallelism?: number;
    rerankGap?: number;
    rerankMaxRangeBytes?: number;
  }
): Promise<CompleteSearch>;

/** Header formatVersion -> manifest profile accepted by this reader. */
export declare const SUPPORTED_PROFILES: Readonly<Record<number, string>>;
export declare const CORPUS_LAYOUT_V2: 'records-v2';
/**
 * Check a host encoder against a kind-2 declaration's verification vectors
 * (contract 4.4 mode 2) without opening an artifact. Throws on dimension
 * or tolerance disagreement; resolves with the number of vectors checked.
 */
export declare function verifyHostEncoder(
  declaration: Record<string, unknown>,
  encodeQuery: (text: string) => Promise<Float32Array | number[]> | Float32Array | number[],
  dim: number
): Promise<{ checked: number }>;

/** Kernel layout compiled into the kind-3 encoder (mirrors encoder.cpp). */
export declare const KERNEL_LAYOUT: Record<'V' | 'P' | 'T' | 'D' | 'F' | 'L' | 'B' | 'H', number>;
export declare function expectedBlobBytes(layout?: Record<string, number>): number;
export declare function parseInlineTransformerEncoder(encoderBytes: Uint8Array): {
  declaration: Record<string, unknown>;
  vocabText: string;
  blob: Uint8Array;
};
export declare function createInlineTransformerEmbedder(input: {
  declaration: Record<string, unknown>;
  vocabText: string;
  blob: Uint8Array;
  createEncoder: () => Promise<unknown> | unknown;
  verify?: boolean;
}): Promise<{
  declaration: Record<string, unknown>;
  dim: number;
  maxSeq: number;
  embed(text: string): Promise<{ vector: Float32Array; text: string; windows: number }>;
  dispose(): void;
}>;
/**
 * Assert an expected verification-vector embedding is exactly dim finite
 * numbers; throws otherwise. Malformed expectations would otherwise make
 * every comparison NaN and "pass".
 */
export declare function validateExpectedEmbedding(values: unknown, dim: number, label: string): void;
/** Validate (or default to 1e-3) a verification vector's tolerance. */
export declare function validateTestVectorTolerance(tolerance: unknown, label: string): number;
export declare const INLINE_TEST_VECTOR_TEXTS: string[];
export declare function buildInlineTestVectors(embedder: unknown, texts?: string[]): Promise<Array<{
  text: string; windows: number; embedding: number[]; tolerance: number;
}>>;
export declare function verifyInlineTestVectors(embedder: unknown): Promise<{ checked: number }>;
