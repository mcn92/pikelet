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
  /** The URL as given, before any redirect; a layer's parent locator resolves against it. */
  url: string;
  init(): Promise<void>;
}

export declare function httpRangeSource(url: string, options?: {
  preferredParallelism?: number;
  /**
   * Rows of the vector rerank closer than this many bytes are fetched in
   * one request, reading the bytes between them. Default 256 KiB. A pack's
   * manifest recommendedGap applies only when it is wider.
   */
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
  /**
   * Called with each redirect target before it is followed; throw to refuse
   * the hop. When set, range reads never auto-follow redirects. A chain walk
   * sets it for a parent named by a layer's locator, confining every hop to
   * the child's origin and directory (LAYERED_PROFILE.md 5.1.1).
   */
  redirectGuard?: (target: string) => void;
  /**
   * The fetch used for every request. Default nodeFetch under Node
   * (the package's node export condition), else globalThis.fetch. A runtime
   * whose fetch transparently decodes response bodies can pass one that does
   * not.
   */
  fetch?: (input: string, init?: {
    method?: string;
    headers?: Record<string, string>;
    redirect?: 'follow' | 'manual' | 'error';
  }) => Promise<any>;
}): HttpRangeSource;

/**
 * Node only (the 'node' export condition): a fetch over node:http/node:https
 * that never decompresses, sends Accept-Encoding: identity and refuses any
 * other Content-Encoding. httpRangeSource defaults to it under Node.
 */
export declare function nodeFetch(input: string, init?: {
  method?: 'GET' | 'HEAD';
  headers?: Record<string, string>;
  redirect?: 'follow' | 'manual' | 'error';
  signal?: unknown;
  /** Inflate a gzip/deflate body a server sent despite identity (crawled HTML); Brotli/zstd still refused. */
  inflateGzip?: boolean;
}): Promise<any>;

/** openPikeletFile's options.blobCache: bytes keyed by their sha256 (lowercase hex). */
export interface BlobCache {
  get(key: string): Promise<Uint8Array | null> | Uint8Array | null;
  put(key: string, bytes: Uint8Array): Promise<void> | void;
  delete?(key: string): Promise<void> | void;
}

/**
 * Node only (the 'node' export condition): a BlobCache on disk, one file per
 * key under `dir` (default defaultBlobCacheDir()), evicting least recently
 * used entries past maxBytes (default 2 GiB). flush() settles pending writes.
 */
export declare function diskBlobCache(dir?: string, options?: { maxBytes?: number }): BlobCache & {
  dir: string;
  flush(): Promise<void>;
};
/** $PIKELET_CACHE_DIR/blobs, else $XDG_CACHE_HOME/pikelet/blobs, else ~/.cache/pikelet/blobs. */
export declare function defaultBlobCacheDir(env?: Record<string, string | undefined>): string;
export declare const DEFAULT_BLOB_CACHE_BYTES: number;

export interface CompleteQueryResult {
  /**
   * The pack's calibrated verdict on this retrieval. On packs calibrated
   * by pikelet 0.10 or later (evaluation calibration.target 'relevance')
   * it measures retrieval relevance: 'none' means the pack does not appear
   * to cover the query's topic, 'strong' that it does — not that a
   * returned passage states the specific fact asked for. 'unscored': the
   * pack ships no fitted calibrator.
   */
  matchQuality: 'strong' | 'weak' | 'none' | 'unscored';
  confidence?: number;
  /**
   * Each result is the hydrated corpus record plus the search's `id` and
   * `distance`. Those two names are reserved: they always come from the
   * search, and a record field of the same name is overwritten, never the
   * other way round.
   */
  results: Array<{ id: number; distance: number } & Record<string, unknown>>;
  /**
   * Which passage grounded the verdict and which of the query's content
   * words it contains. Present when a fitted calibrator scored the query
   * (absent on 'unscored'). An asked-for value in `uncovered` is one no top
   * passage states, whatever matchQuality says.
   */
  grounding?: {
    /** Id of the passage the coverage score was taken from. */
    recordId: number | null;
    /** Index of that passage in the fused (or distance-ordered) window. */
    passageIndex: number;
    /** Weighted share of the query's content words the passage contains, 0-1. */
    coverage: number;
    covered: string[];
    uncovered: string[];
  };
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
    /** Rows in the index: equal to records unless a row map groups several rows per record. */
    indexRows: number;
    /** The policy behind a multi-row index (COMPLETE_PROFILE.md 3.9), e.g. sentence windows; null otherwise. */
    rowMap: { kind: string; sentences: number; rows: number } | null;
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
    lexical: {
        terms: number;
        docCount: number;
        /** 'segment-sha256' when read eagerly and verified whole; 'unverified-lazy-reads' for lazily opened segments. */
        integrity: 'segment-sha256' | 'unverified-lazy-reads';
        lazy?: true;
    } | null;
    /**
     * 'engine' once a WASM scan kernel serves the resident scan
     * (auto-staged or injected via options.sketchScanner); 'js' before
     * background staging resolves and forever if it fails or was disabled.
     */
    residentScan: 'engine' | 'js';
    sampleQueries: string[];
    /**
     * The file's parts in file order: 'header' (fixed header, manifest and
     * segment table) followed by each segment of a known kind ('index',
     * 'corpus', 'lexical', 'query-interp', 'evaluation', ...). Absolute
     * byte offsets.
     */
    layout: Array<{ kind: string; offset: number; length: number }>;
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
  /** The absolute byte range one corpus record's stored bytes occupy in the file. */
  recordRange(id: number): { offset: number; length: number };
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
     * A store for the regions a pack commits to by sha256 that dominate a
     * cold URL mount: the sketch's resident prefix and the query-interp
     * segment holding the inline encoder. Keys are those digests (lowercase
     * hex), so entries never go stale. Every hit is hashed against its key
     * before use and a mismatch is treated as a miss (and deleted), so the
     * store need not be trusted; get/put/delete faults never fail an open.
     * diskBlobCache() (Node) is a ready implementation. Ignored by an open
     * whose resident verification is off.
     */
    blobCache?: BlobCache;
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

/** Rank-fusion parameters shared by the reader, the calibrator and the BEIR ladder. */
export interface FusionOptions {
  /** Reciprocal-rank-fusion constant (default 60). */
  rrfK?: number;
  /** Weight of the lexical (BM25) rank term relative to the vector term (default 0.5). */
  lexicalWeight?: number;
  /**
   * Relative distance margin (top-2 minus top-1, over top-1) at or above which
   * the vector top-1 keeps rank 1 (default 0.05; 0 disables the guard).
   */
  guardMargin?: number;
}
export declare const FUSION_DEFAULTS: Readonly<Required<FusionOptions>>;
/**
 * Fuse the exact-reranked vector order (every lexical candidate included) with the BM25 order.
 * Returns the same hit objects, reordered.
 */
export declare function fuseCandidates<T extends { id: number; distance: number }>(
  vectorOrder: T[],
  lexicalIds: number[],
  options?: FusionOptions,
): T[];
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
/** Largest layer count (layout L) the kind-3 kernel runs; L may be 1 to this. */
export declare const KERNEL_MAX_LAYERS: number;
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
