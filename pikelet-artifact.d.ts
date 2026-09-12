// Types for the `pikelet-wasm/artifact` subpath: the engine-free CJS module
// holding the Search Artifact readers, builders, and snapshot parser. Unlike
// the entrypoints, this module DOES provide these as named exports.

import type {
  NodeFileRangeSourceConstructor,
  PikeletRangeArtifactConstructor,
  PikeletSketchArtifact,
  PikeletSketchArtifactConstructor,
  RangeArtifactBuildManifest,
  RangeArtifactBuildOptions,
  SketchArtifactBuildManifest,
  SketchArtifactBuildOptions,
  SketchScanner,
  SketchScannerOptions,
} from './pikelet.js';

/** A parsed uint8 engine snapshot, as returned by {@link parseUint8Snapshot}. */
export interface Uint8SnapshotGraph {
  readonly kind: 'u8';
  readonly version: number;
  readonly dim: number;
  readonly count: number;
  readonly entryPoint: number;
  readonly maxLevel: number;
  readonly M: number;
  readonly M0: number;
  /** 0 = l2, 1 = cosine. */
  readonly metric: number;
  readonly efConstruction: number;
  readonly scales: Float32Array;
  readonly offsets: Float32Array;
  readonly qdata: Uint8Array;
  readonly levels: Uint16Array;
  readonly base: Uint32Array[];
  readonly upper: Uint32Array[][];
}

/**
 * Any index-like source of quantized rows: a {@link Uint8SnapshotGraph} or a
 * live quantized index exposing the same row fields.
 */
export interface SketchRowSource {
  readonly dim: number;
  readonly count: number;
  /** 0 = l2, 1 = cosine. */
  readonly metric: number;
  readonly qdata: Uint8Array;
  readonly scales: Float32Array;
  readonly offsets: Float32Array;
}

export declare const PikeletRangeArtifact: PikeletRangeArtifactConstructor;
export declare const PikeletSketchArtifact: PikeletSketchArtifactConstructor;
export declare const NodeFileRangeSource: NodeFileRangeSourceConstructor;

/**
 * Raw scanner factory. The entrypoints curry the engine loader; here it must
 * be passed explicitly as the runtime's engine-module loader.
 */
export declare function createSketchScanner(
  loadEngine: () => Promise<unknown>,
  artifact: PikeletSketchArtifact,
  options?: SketchScannerOptions
): Promise<SketchScanner>;

/**
 * @deprecated The `.pikelet-range` profile is deprecated
 * (spec/SEARCH_ARTIFACT_CONTRACT.md 9.2); build a `.pikelet-sketch`
 * artifact instead. Readers stay supported for existing artifacts.
 */
export declare function buildRangeArtifact(
  snapshot: Uint8Array | ArrayBufferLike,
  outPath: string,
  opts?: RangeArtifactBuildOptions
): RangeArtifactBuildManifest;

/**
 * @deprecated The `.pikelet-range` profile is deprecated
 * (spec/SEARCH_ARTIFACT_CONTRACT.md 9.2); build a `.pikelet-sketch`
 * artifact instead. Readers stay supported for existing artifacts.
 */
export declare function buildRangeArtifactFile(
  snapshotPath: string,
  outPath: string,
  opts?: RangeArtifactBuildOptions
): RangeArtifactBuildManifest;

export declare function buildSketchArtifact(
  snapshot: Uint8Array | ArrayBufferLike,
  outPath: string,
  opts?: SketchArtifactBuildOptions
): SketchArtifactBuildManifest;

export declare function buildSketchArtifactFile(
  snapshotPath: string,
  outPath: string,
  opts?: SketchArtifactBuildOptions
): SketchArtifactBuildManifest;

/** Write a sketch artifact directly from quantized rows. */
export declare function exportSketchArtifact(
  index: SketchRowSource,
  outPath: string,
  options?: SketchArtifactBuildOptions
): SketchArtifactBuildManifest;

/** Parse and validate an untrusted uint8 engine snapshot (fails closed). */
export declare function parseUint8Snapshot(bytes: Uint8Array | ArrayBufferLike): Uint8SnapshotGraph;
