// Types for the browser / Cloudflare Workers entrypoints (`pikelet-wasm/web`).
// These runtimes expose the portable API, including range-readable Search
// Artifacts. Only PikeletError and PIKELET_ERROR_CODES exist as named runtime
// exports; every other re-export below is type-only. Node-only helpers
// (NodeFileRangeSource, the build*/open*File functions, loadJsonFile /
// loadSnapshotFile) are absent from this surface so using them is a compile
// error rather than a runtime throw.
export { PikeletError, PIKELET_ERROR_CODES } from './pikelet.js';
export type {
  Metric,
  PikeletErrorCode,
  VectorInput,
  VectorRecord,
  CreateOptions,
  SearchOptions,
  FromVectorsResult,
  RestoreOptions,
  SnapshotFormat,
  SnapshotInspection,
  SearchResult,
  MemoryUsage,
  ResolvedConfig,
  PikeletIndex,
  RangeReadSource,
  RangeArtifactSearchOptions,
  RangeArtifactNode,
  RangeArtifactStats,
  RangeArtifactRound,
  RangeArtifactSearchResult,
  RangeArtifactOpenOptions,
  PikeletRangeArtifact,
  PikeletRangeArtifactConstructor,
  SketchTier,
  SketchStageEvent,
  SketchArtifactOpenOptions,
  SketchScanner,
  SketchScannerOptions,
  SketchArtifactSearchOptions,
  SketchArtifactSearchResult,
  SketchArtifactStats,
  PikeletSketchArtifact,
  PikeletSketchArtifactConstructor,
  PikeletApi,
} from './pikelet.js';

import type { PikeletApi } from './pikelet.js';

declare const Pikelet: PikeletApi;

export default Pikelet;
