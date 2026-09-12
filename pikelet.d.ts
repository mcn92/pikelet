export type Metric = 'cosine' | 'l2';

export const PIKELET_ERROR_CODES: Readonly<{
  INVALID_ARGUMENT: 'INVALID_ARGUMENT';
  DIMENSION_MISMATCH: 'DIMENSION_MISMATCH';
  INVALID_VECTOR: 'INVALID_VECTOR';
  INDEX_FULL: 'INDEX_FULL';
  INDEX_DISPOSED: 'INDEX_DISPOSED';
  COMPACTION_REQUIRED: 'COMPACTION_REQUIRED';
  SNAPSHOT_INVALID: 'SNAPSHOT_INVALID';
  SNAPSHOT_CONFIG_MISMATCH: 'SNAPSHOT_CONFIG_MISMATCH';
  SNAPSHOT_CAPACITY_EXCEEDED: 'SNAPSHOT_CAPACITY_EXCEEDED';
  WASM_LOAD_FAILED: 'WASM_LOAD_FAILED';
  WASM_ALLOCATION_FAILED: 'WASM_ALLOCATION_FAILED';
  FILE_IO_FAILED: 'FILE_IO_FAILED';
  PARSE_FAILED: 'PARSE_FAILED';
  INTERNAL_INVARIANT: 'INTERNAL_INVARIANT';
  INDEX_LIMIT: 'INDEX_LIMIT';
}>;

export type PikeletErrorCode = typeof PIKELET_ERROR_CODES[keyof typeof PIKELET_ERROR_CODES];

export class PikeletError extends Error {
  readonly code: PikeletErrorCode;
  readonly details?: Readonly<Record<string, unknown>>;
  readonly cause?: unknown;
  constructor(code: PikeletErrorCode, message: string, details?: Readonly<Record<string, unknown>>, cause?: unknown);
}

export type VectorInput = Float32Array | readonly number[];

export interface VectorRecord<Id = unknown> {
  id?: Id;
  vector: VectorInput;
  [key: string]: unknown;
}

export interface CreateOptions {
  /** Input vector dimension (required). */
  dim: number;
  /**
   * Maximum index capacity (default: 100000). The backend allocates its
   * arena eagerly at ~(dim + 16*M + 39) bytes per element quantized, or
   * ~(4*dim + 8*M + 23) float32; configurations estimated above the 1.5 GiB
   * wasm32 index budget are rejected with WASM_ALLOCATION_FAILED at
   * create() rather than aborting inside the engine. Hard ABI ceiling:
   * 2147483647.
   */
  maxElements?: number;
  /** Distance metric (default: 'cosine'). */
  metric?: Metric;
  /** Use uint8 quantized storage (default: true). */
  quantized?: boolean;
  /** HNSW graph connectivity (default: 12). Valid range: 2-128. */
  M?: number;
  /** Build-time search breadth (default: 75). Valid range: 1-4096. */
  efConstruction?: number;
  /** Query-time search breadth (default: 100). Valid range: 1-4096. */
  efSearch?: number;
  /** RNG seed for deterministic HNSW level assignment (default: 108). */
  seed?: number;
}

export interface SearchOptions {
  /** Override query-time search breadth for this call. Valid range: 1-4096. */
  efSearch?: number;
}

export interface FromVectorsResult<Id = unknown> {
  /** Created and populated index. Caller owns dispose(). */
  index: PikeletIndex;
  /** Stable Pikelet IDs returned in insertion order. */
  ids: number[];
  /** Mapping from Pikelet IDs back to caller-provided source IDs. */
  idMap: Map<number, Id>;
}

export interface JsonFileOptions extends Omit<CreateOptions, 'dim'> {
  /** Input vector dimension. Default: inferred from the first vector in the file. */
  dim?: number;
  /** Force JSON or JSONL parsing. Default: inferred from file extension. */
  format?: 'json' | 'jsonl';
  /** Record field holding the vector. Default: 'vector'. */
  vectorKey?: string;
  /** Record field holding the caller's source ID. Default: 'id'. */
  idKey?: string;
  /** Maximum JSON/JSONL file size in bytes. Default: 64 MiB. */
  maxFileBytes?: number;
}

export interface RestoreOptions extends Partial<CreateOptions> {}

export type SnapshotFormat = 'pikelet' | 'raw';

export interface SnapshotInspection {
  readonly format: SnapshotFormat;
  readonly version: number;
  readonly dim: number;
  readonly count: number;
  readonly metric: Metric;
  readonly quantized: boolean;
  readonly M: number;
  readonly efConstruction: number;
  readonly nextId: number;
}

export interface SnapshotFileOptions extends RestoreOptions {
  /** Maximum snapshot file size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
}

export interface SearchResult {
  /** Vector ID as returned by add(). */
  id: number;
  /** Distance from query (lower is more similar). */
  distance: number;
}

export interface RangeReadSource {
  /** Return exactly length bytes starting at offset. */
  read(offset: number, length: number): Uint8Array | ArrayBuffer | Promise<Uint8Array | ArrayBuffer>;
  /** Optional close hook for file-backed sources. */
  close?(): void | Promise<void>;
}

export interface RangeArtifactSearchOptions {
  /** Query-time search breadth. Default: 100. */
  efSearch?: number;
  /** Merge adjacent v1 record runs separated by this many skipped records. Default: 0. */
  gap?: number;
  /** Candidates expanded per fetch round (deeper rounds trade requests for latency). Default: 1. */
  expansionBatch?: number;
  /** Concurrent range reads per fetch round. Default: 1. */
  parallelism?: number;
  /** Overrides `parallelism` for range reads when both are set. */
  rangeParallelism?: number;
  /**
   * Split coalesced record runs at this many bytes (record-aligned, floor
   * one record) so no single range read grows unbounded. Default: 16 MiB.
   */
  maxRangeBytes?: number;
}

/** A decoded graph record, as returned by {@link PikeletRangeArtifact.readNode}. */
export interface RangeArtifactNode {
  readonly id: number;
  readonly level: number;
  readonly base: Uint32Array;
  readonly upper: readonly Uint32Array[];
  readonly qdata: Uint8Array;
  readonly scale: number;
  readonly offset: number;
}

export interface RangeArtifactStats {
  readonly rangeRequests: number;
  readonly rangeBytes: number;
  readonly rangeNodesDecoded: number;
  readonly cachedNodes: number;
  readonly lazyCacheBytes: number;
  readonly routerResident: {
    readonly records: number;
    readonly bytes: number;
  };
  /**
   * Which segments have passed their v3 whole-segment digests. Always false
   * for pre-v3 artifacts (which carry no digests) and under verify:false;
   * base becomes true only after a successful verifyBaseSegment().
   */
  readonly segmentVerified: {
    readonly idMap: boolean;
    readonly router: boolean;
    readonly base: boolean;
  };
}

export interface RangeArtifactRound {
  readonly ids: number;
  readonly requests: number;
  readonly bytes: number;
  readonly rangeBytes: readonly number[];
}

export interface RangeArtifactSearchResult {
  readonly results: SearchResult[];
  readonly rounds: RangeArtifactRound[];
  readonly stats: RangeArtifactStats;
}

export interface RangeArtifactOpenOptions {
  /** Eagerly load the v2 router segment. Default: true. */
  loadRouter?: boolean;
  /**
   * Verify the id map and router segments against the header digests at open
   * (v3 artifacts only; pre-v3 artifacts carry no digests and open
   * unverified either way). Default: true.
   */
  verify?: boolean;
  /**
   * Budget for any single open-path read (id map, router segment), honored
   * strictly: a budget below what the artifact's resident segments need
   * fails the open with SNAPSHOT_INVALID. Default: 256 MiB; Infinity defers
   * to the 2 GiB absolute backstop. Other non-positive or non-numeric
   * values throw INVALID_ARGUMENT.
   */
  maxReadBytes?: number;
  /**
   * Byte budget for the lazily-fetched record cache (LRU eviction; the
   * resident router segment is not counted). Default: 64 MiB. Pass Infinity
   * for the pre-0.3 unbounded behavior. Budgets below 64 records are raised
   * to that floor; other non-positive or non-numeric values throw
   * INVALID_ARGUMENT.
   */
  maxCacheBytes?: number;
}

export interface RangeArtifactBuildOptions {
  /** Deterministic base-layer layout. Default: 'rcm'. */
  layout?: 'rcm' | 'identity';
}

export interface RangeArtifactBuildManifest {
  readonly format: 'pikelet-range-artifact';
  readonly formatVersion: number;
  /** Hex SHA-256 whole-segment digests stamped into the v3 header. */
  readonly integrity: Readonly<{
    idMapSha256: string;
    routerSha256: string;
    baseSha256: string;
  }>;
  readonly file: string;
  readonly sizeBytes: number;
  readonly kind: string;
  readonly metric: Metric;
  readonly layout: Readonly<Record<string, unknown>>;
  readonly graph: Readonly<Record<string, number>>;
  readonly addressing: Readonly<Record<string, number>>;
}

/**
 * Node-only file-backed range source. This is NOT a named runtime export of
 * any entrypoint: reach the constructor via the Node API object
 * (`Pikelet.NodeFileRangeSource`) or the `pikelet-wasm/artifact` subpath.
 */
export interface NodeFileRangeSource extends RangeReadSource {
  readonly filePath: string;
  readonly size: number;
  read(offset: number, length: number): Promise<Uint8Array>;
  close(): Promise<void>;
}

export interface NodeFileRangeSourceConstructor {
  new (filePath: string): NodeFileRangeSource;
  readonly prototype: NodeFileRangeSource;
}

/**
 * A range-readable Search Artifact reader. The class is NOT a named runtime
 * export of any entrypoint: reach it via `Pikelet.RangeArtifact` or the
 * `pikelet-wasm/artifact` subpath.
 */
export interface PikeletRangeArtifact {
  readonly version: number;
  readonly kind: number;
  readonly dim: number;
  readonly count: number;
  readonly entryPoint: number;
  readonly maxLevel: number;
  readonly M: number;
  readonly M0: number;
  readonly metric: number;
  readonly recordBytes: number;
  readonly routerCount: number;
  readonly baseCount: number;
  readonly routerResident: { readonly records: number; readonly bytes: number };
  search(query: VectorInput, k: number, options?: RangeArtifactSearchOptions): Promise<RangeArtifactSearchResult>;
  stats(): RangeArtifactStats;
  /**
   * Verify the lazily-read base segment against the v3 header digest (not a
   * query-path operation). Streams the segment in bounded chunks where a
   * streaming hash exists (Node); other runtimes fall back to one read
   * bounded by maxReadBytes. Resolves true on match; throws
   * SNAPSHOT_INVALID on mismatch, INVALID_ARGUMENT for pre-v3 artifacts,
   * which carry no digests.
   */
  verifyBaseSegment(options?: { chunkBytes?: number }): Promise<true>;
  resetStats(): void;
  clearCache(options?: { reloadRouter?: boolean }): Promise<{ records: number; bytes: number }>;
  /** Fetch (and cache) the given node ids in coalesced ranges. Returns the number of range reads issued. */
  prefetch(ids: readonly number[], options?: { gap?: number; parallelism?: number; maxRangeBytes?: number }): Promise<number>;
  /** Read a single node, fetching it if not cached. */
  readNode(id: number): Promise<RangeArtifactNode>;
  /** Byte offset of a node's record within the artifact. */
  recordAddressForId(id: number): number;
  /** Mark the current position in the range-read log (see rangesSince). */
  markRanges(): number;
  /** Ranges read since a markRanges() mark, as [start, end) byte offsets. */
  rangesSince(mark: number): Array<[number, number]>;
  close(): Promise<void>;
}

export interface PikeletRangeArtifactConstructor {
  open(source: RangeReadSource, options?: RangeArtifactOpenOptions): Promise<PikeletRangeArtifact>;
  /** Node entrypoints only. */
  openFile(filePath: string, options?: RangeArtifactOpenOptions): Promise<PikeletRangeArtifact>;
  readonly prototype: PikeletRangeArtifact;
}

/** Resident sketch tier: the coarse staged-boot tier or the full tier. */
export type SketchTier = 'micro' | 'full';

export interface SketchStageEvent {
  readonly tier: SketchTier;
  readonly residentBytes: number;
}

export interface SketchArtifactOpenOptions {
  /** Verify the resident prefix hash when a crypto backend exists. Default: true. */
  verify?: boolean;
  /**
   * Budget for any single open-path read (resident prefix, staged tiers),
   * honored strictly: a budget below what the artifact's resident tier
   * needs fails the open with SNAPSHOT_INVALID. Default: 256 MiB; Infinity
   * defers to the 2 GiB absolute backstop. Other non-positive or
   * non-numeric values throw INVALID_ARGUMENT.
   */
  maxReadBytes?: number;
  /**
   * Byte budget for the fetched-row cache (LRU eviction; the resident sketch
   * tier is not counted). Default: 64 MiB. Pass Infinity to disable eviction.
   * Budgets below 256 rows are raised to that floor; other non-positive or
   * non-numeric values throw INVALID_ARGUMENT.
   */
  maxCacheBytes?: number;
  /**
   * Staged boot: when the artifact carries a micro tier, become searchable
   * after loading only the stage-1 prefix and swap to the full tier in the
   * background (see {@link PikeletSketchArtifact.fullyResident}). Ignored for
   * artifacts without a micro tier. Default: false.
   */
  staged?: boolean;
  /** Called on each residency transition of a staged open (micro, then full). */
  onStage?: (event: SketchStageEvent) => void;
}

export interface SketchScanner {
  /**
   * Return the top-C row ids for a pooled query, ascending by sketch
   * distance. `c` is silently clamped to {@link maxRerank}.
   */
  scan(pooledQuery: Float32Array | readonly number[], c: number): number[];
  /** Free the scanner's WASM heap allocations. */
  dispose(): void;
  readonly maxRerank: number;
  /** The artifact metric this scanner scores (0 = l2, 1 = cosine). */
  readonly metric: number;
  /** Pooled dimensionality of the tier this scanner was built for. */
  readonly sketchDims: number;
  /** The resident tier this scanner scores. */
  readonly tier: SketchTier;
}

export interface SketchScannerOptions {
  /** Upper bound on rerank depth the scanner will return. Default: 1024. */
  maxRerank?: number;
  /**
   * Which resident tier to build the scanner over. 'micro' requires an
   * artifact with a micro tier. Default: 'full'.
   */
  tier?: SketchTier;
}

export interface SketchArtifactSearchOptions {
  /**
   * Rerank depth (top-C sketch candidates fetched for exact scoring). A
   * scanner-backed search additionally clamps this to the scanner's
   * maxRerank.
   */
  rerank?: number;
  /** Byte gap for coalescing row fetches. Default: 2048. */
  gap?: number;
  /** Concurrent range reads per fetch round. Default: 8. */
  parallelism?: number;
  /**
   * Row ids that join the exact rerank alongside the sketch scan's
   * selection (e.g. a lexical index's matches in a hybrid query). Scored
   * by true distance like any candidate. Ids must be integers in
   * [0, count).
   */
  extraCandidates?: readonly number[];
  /**
   * Return every reranked candidate in distance order instead of the top
   * k. The candidates are already fetched and exactly scored; this
   * changes only the output slice. Hybrid callers use it so rank fusion
   * sees the true distance of every lexical candidate.
   */
  fullRerankOutput?: boolean;
  /**
   * Split coalesced row runs at this many bytes (row-aligned, floor one
   * row) so no single range read grows unbounded. Default: 16 MiB.
   */
  maxRangeBytes?: number;
  /**
   * External top-C scanner (e.g. from createSketchScanner). Must implement
   * the artifact's metric: cosine artifacts require `metric === 1`, and a
   * declared metric that disagrees with the artifact throws
   * INVALID_ARGUMENT. A scanner whose tier does not match the artifact's
   * current resident tier is bypassed, not misused.
   */
  scanner?: SketchScanner | { metric?: number; scan(pooledQuery: Float32Array, c: number): number[] };
  /**
   * While serving from the micro tier, multiply the candidate pool by this
   * factor to compensate for the coarser sketches. Default: 4.
   */
  microBoost?: number;
  /**
   * Scanner built over the micro tier (createSketchScanner with
   * `tier: 'micro'`); engaged only while the micro tier is the active
   * resident tier and `sketchDims` matches it.
   */
  microScanner?: SketchScanner | { sketchDims: number; metric?: number; scan(pooledQuery: Float32Array, c: number): number[] };
}

export interface SketchArtifactSearchResult {
  readonly results: SearchResult[];
  /** Number of candidates fetched for exact rerank. */
  readonly rerank: number;
  /** The resident tier that served this search. */
  readonly tier: SketchTier;
}

export interface SketchArtifactStats {
  readonly rangeRequests: number;
  readonly rangeBytes: number;
  readonly cachedRows: number;
  readonly cacheBytes: number;
  readonly residentBytes: number;
  readonly residentVerified: boolean;
  /** True after a successful verifyVectors(). */
  readonly vectorsVerified: boolean;
}

export interface SketchArtifactBuildOptions {
  /** Pooled sketch dimensionality; must divide dim. Default: dim / 2. */
  sketchDims?: number;
  /** Bits per sketch dimension. Default: 4. */
  sketchBits?: 4 | 8;
  /** Producer-recommended rerank depth recorded in the header. */
  recommendedRerank?: number;
  /**
   * Add a staged-boot micro tier at this pooled dimensionality; must divide
   * sketchDims and be smaller. Default: no micro tier.
   */
  microDims?: number;
  /** Bits per micro-tier dimension. Default: 4. */
  microBits?: 4 | 8;
  /**
   * Per-row integrity (format version 2, the default): the vectors region is
   * written as blocks of rowsPerBlock rows, each preceded by a digest page
   * of truncated per-row SHA-256s, anchored by a page-hash table in the
   * resident prefix. Pass false to emit a version-1 artifact.
   */
  rowIntegrity?: boolean;
  /** Rows per digest block (format 2). Default: 16. */
  rowsPerBlock?: number;
  /** Truncated per-row digest size in bytes, 8..32 (format 2). Default: 16. */
  rowDigestBytes?: number;
}

export interface SketchArtifactBuildManifest {
  readonly format: 'pikelet-sketch-artifact';
  readonly formatVersion: number;
  /** Output path; absent on the bytes-in/bytes-out builder. */
  readonly file?: string;
  readonly sizeBytes: number;
  readonly metric: Metric;
  readonly graph: Readonly<Record<string, number>>;
  readonly sketch: Readonly<Record<string, number>>;
  /** Micro-tier geometry, or null when built without one. */
  readonly micro: Readonly<{
    microDims: number;
    microBits: number;
    microPool: number;
    stage1Bytes: number;
  }> | null;
  readonly addressing: Readonly<Record<string, number>>;
  readonly recommendedRerank: number;
}

/**
 * A sketch Search Artifact reader. The class is NOT a named runtime export of
 * any entrypoint: reach it via `Pikelet.SketchArtifact` or the
 * `pikelet-wasm/artifact` subpath.
 */
export interface PikeletSketchArtifact {
  readonly metric: number;
  readonly dim: number;
  readonly count: number;
  readonly sketchDims: number;
  readonly sketchBits: number;
  /** Micro-tier pooled dimensionality; 0 when the artifact has none. */
  readonly microDims: number;
  /** Micro-tier bits per dimension; 0 when the artifact has none. */
  readonly microBits: number;
  /** The currently active resident tier ('micro' only during a staged boot). */
  readonly tier: SketchTier;
  /**
   * Resolves with this artifact once the full tier is resident and verified.
   * Already resolved for non-staged opens; rejects if the background stage-2
   * load fails.
   */
  readonly fullyResident: Promise<PikeletSketchArtifact>;
  readonly recommendedRerank: number;
  readonly residentBytes: number;
  readonly residentVerified: boolean;
  /** True after a successful verifyVectors(). */
  readonly vectorsVerified: boolean;
  search(query: VectorInput, k: number, options?: SketchArtifactSearchOptions): Promise<SketchArtifactSearchResult>;
  stats(): SketchArtifactStats;
  /** Drop all cached rows; subsequent searches fetch cold. Never affects result semantics. */
  clearCache(): void;
  /**
   * Verify the lazy vectors segment against the header's whole-segment hash
   * (not a query-path operation). Streams the segment in bounded chunks
   * where a streaming hash exists (Node); other runtimes fall back to one
   * read bounded by maxReadBytes. Resolves true on match; throws
   * SNAPSHOT_INVALID on mismatch or when verification is impossible (no
   * crypto backend).
   */
  verifyVectors(options?: { chunkBytes?: number }): Promise<true>;
  close(): Promise<void>;
}

export interface PikeletSketchArtifactConstructor {
  open(source: RangeReadSource, options?: SketchArtifactOpenOptions): Promise<PikeletSketchArtifact>;
  /** Node entrypoints only. */
  openFile(filePath: string, options?: SketchArtifactOpenOptions): Promise<PikeletSketchArtifact>;
  readonly prototype: PikeletSketchArtifact;
}

export interface MemoryUsage {
  /** Backend-owned graph and vector storage estimate. */
  readonly logicalIndexBytes: number;
  /** Current byte length of this index's isolated WebAssembly heap. */
  readonly wasmHeapBytes: number;
  /** Raw backend snapshot buffer retained by the most recent export(). */
  readonly snapshotBufferBytes: number;
}

export interface ResolvedConfig {
  readonly dim: number;
  readonly maxElements: number;
  readonly metric: Metric;
  readonly quantized: boolean;
  readonly M: number;
  readonly efConstruction: number;
  readonly efSearch: number;
  readonly seed: number;
}

export interface PikeletIndex {
  /** Insert a single vector. Returns its stable external ID. */
  add(vector: VectorInput): number;
  /** Insert multiple vectors. Returns stable external IDs in insertion order. */
  addBatch(vectors: readonly VectorInput[]): number[];
  /** Find the k nearest neighbors of a query vector. */
  search(query: VectorInput, k: number, options?: SearchOptions): SearchResult[];
  /** Find the k nearest neighbors restricted to IDs in allowedIds. */
  searchFiltered(query: VectorInput, k: number, allowedIds: Set<number>, options?: SearchOptions): SearchResult[];
  /** Change the default query-time search breadth for future searches. */
  setEfSearch(efSearch: number): void;
  /** Soft-delete a live vector. Returns false for unknown or already-deleted IDs. */
  delete(id: number): boolean;
  /** Whether an ID currently refers to a live vector. */
  has(id: number): boolean;
  /** Whether an ID is awaiting reclamation by compact(). */
  isDeleted(id: number): boolean;
  /** Rebuild the graph without soft-deleted entries. */
  compact(): void;
  /** Serialize index state. Throws if ghostCount > 0; call compact() first after deletions. */
  export(): Uint8Array;
  /**
   * Restore index state from a previous export().
   * Validation failures leave the target index unchanged.
   */
  import(data: Uint8Array | ArrayBufferLike): void;
  /** Free WASM buffers. The instance is unusable after this. */
  dispose(): void;
  /** Dispose support for JavaScript `using` declarations. */
  [Symbol.dispose](): void;
  /** Vectors stored by the backend, including soft-deleted entries until compact(). */
  readonly count: number;
  /** Live vectors available to search. */
  readonly liveCount: number;
  /** Preferred name for soft-deleted vectors awaiting compaction. */
  readonly deletedCount: number;
  /** Preferred name for the soft-deleted fraction. */
  readonly deletedRatio: number;
  /** Fixed insertion capacity. */
  readonly capacity: number;
  /** Remaining insertion slots; soft deletion does not increase this value. */
  readonly remainingCapacity: number;
  /** Soft-deleted vectors awaiting compaction. */
  readonly ghostCount: number;
  /** Ratio reported by the backend; after soft deletes this behaves as ghostCount / count. */
  readonly ghostRatio: number;
  /** Estimated index memory in bytes. */
  readonly memory: number;
  /** Logical index, isolated WASM heap, and retained snapshot-buffer memory. */
  readonly memoryUsage: MemoryUsage;
  /** Fully resolved construction and current query-policy values. */
  readonly config: ResolvedConfig;
  /** Input vector dimension. */
  readonly dim: number;
}

/**
 * The portable API exposed by every entrypoint (Node, browser, Workers).
 * The browser/Workers entrypoints expose exactly this surface; the Node
 * entrypoints add file helpers (see {@link NodePikeletApi}).
 */
export interface PikeletApi {
  readonly PikeletError: typeof PikeletError;
  readonly PIKELET_ERROR_CODES: typeof PIKELET_ERROR_CODES;
  readonly RangeArtifact: PikeletRangeArtifactConstructor;
  readonly SketchArtifact: PikeletSketchArtifactConstructor;
  /** Build a WASM-backed SIMD scanner for a sketch artifact's resident tier. */
  createSketchScanner(artifact: PikeletSketchArtifact, options?: SketchScannerOptions): Promise<SketchScanner>;
  /** Create a new Pikelet index using the runtime-specific packaged entrypoint. */
  create(opts: CreateOptions): Promise<PikeletIndex>;
  /** Restore an envelope snapshot, inferring its construction config. */
  restore(snapshot: Uint8Array | ArrayBufferLike, overrides?: RestoreOptions): Promise<PikeletIndex>;
  /** Validate and inspect snapshot headers without creating a WASM index. */
  inspectSnapshot(snapshot: Uint8Array | ArrayBufferLike): SnapshotInspection;
  /** Create an index, run a callback, and always dispose the index afterward. */
  withIndex<T>(opts: CreateOptions, fn: (index: PikeletIndex) => T | Promise<T>): Promise<T>;
  /** Create and populate an index from raw vectors. Infers dim and maxElements by default. */
  fromVectors(vectors: readonly VectorInput[], opts?: Omit<CreateOptions, 'dim'> & Partial<Pick<CreateOptions, 'dim'>>): Promise<FromVectorsResult<never>>;
  /** Create and populate an index from { id, vector } records. */
  fromVectors<Id = unknown>(records: readonly VectorRecord<Id>[], opts?: Omit<CreateOptions, 'dim'> & Partial<Pick<CreateOptions, 'dim'>>): Promise<FromVectorsResult<Id>>;
}

/**
 * The Node.js API: the portable surface plus filesystem helpers. These helpers
 * exist only on the Node entrypoints (`pikelet-wasm`, `pikelet-wasm/node`); the
 * browser/Workers entrypoints do not expose them, and importing those returns
 * the narrower {@link PikeletApi}.
 */
export interface NodePikeletApi extends PikeletApi {
  readonly NodeFileRangeSource: NodeFileRangeSourceConstructor;
  /**
   * Build a range-readable Search Artifact from a uint8 Pikelet snapshot.
   * @deprecated The `.pikelet-range` profile is deprecated
   * (spec/SEARCH_ARTIFACT_CONTRACT.md 9.2); build a `.pikelet-sketch`
   * artifact instead. Readers stay supported for existing artifacts.
   */
  buildRangeArtifact(snapshot: Uint8Array | ArrayBufferLike, outPath: string, opts?: RangeArtifactBuildOptions): RangeArtifactBuildManifest;
  /**
   * Build a range-readable Search Artifact from a uint8 Pikelet snapshot file.
   * @deprecated The `.pikelet-range` profile is deprecated
   * (spec/SEARCH_ARTIFACT_CONTRACT.md 9.2); build a `.pikelet-sketch`
   * artifact instead. Readers stay supported for existing artifacts.
   */
  buildRangeArtifactFile(snapshotPath: string, outPath: string, opts?: RangeArtifactBuildOptions): RangeArtifactBuildManifest;
  /** Open a range-readable Search Artifact from a local file. */
  openRangeArtifactFile(filePath: string, opts?: RangeArtifactOpenOptions): Promise<PikeletRangeArtifact>;
  /** Build a sketch Search Artifact from a uint8 Pikelet snapshot. */
  buildSketchArtifact(snapshot: Uint8Array | ArrayBufferLike, outPath: string, opts?: SketchArtifactBuildOptions): SketchArtifactBuildManifest;
  /** Build a sketch Search Artifact from a uint8 Pikelet snapshot file. */
  buildSketchArtifactFile(snapshotPath: string, outPath: string, opts?: SketchArtifactBuildOptions): SketchArtifactBuildManifest;
  /**
   * Bytes-in/bytes-out sketch build for producers assembling segments in
   * memory (e.g. a complete-artifact compiler); output identical to
   * buildSketchArtifact, no filesystem involved.
   */
  buildSketchArtifactBytes(snapshot: Uint8Array | ArrayBufferLike, opts?: SketchArtifactBuildOptions): { bytes: Uint8Array; manifest: SketchArtifactBuildManifest };
  /** Open a sketch Search Artifact from a local file. */
  openSketchArtifactFile(filePath: string, opts?: SketchArtifactOpenOptions): Promise<PikeletSketchArtifact>;
  /** Load vectors from a JSON/JSONL file and build an index. */
  loadJsonFile<Id = unknown>(filePath: string, opts?: JsonFileOptions): Promise<FromVectorsResult<Id>>;
  /**
   * Load a previously exported Pikelet snapshot from disk. Envelope snapshots
   * carry their own config, so `opts` is only required for raw engine
   * snapshots (which need the full create config).
   */
  loadSnapshotFile(filePath: string, opts?: SnapshotFileOptions): Promise<PikeletIndex>;
}

declare const Pikelet: NodePikeletApi;

export default Pikelet;
