// Type-conformance checks for the shipped .d.ts files. Compiled by
// `npm run test:types` (tsc --noEmit); nothing here runs. Each block asserts
// that the declarations match the documented runtime surface — including
// what must NOT type-check, via @ts-expect-error.

import Pikelet, { PikeletError, PIKELET_ERROR_CODES } from 'pikelet-wasm';
import WebPikelet from 'pikelet-wasm/web';
import {
  PikeletRangeArtifact as RangeArtifactCtor,
  PikeletSketchArtifact as SketchArtifactCtor,
  NodeFileRangeSource,
  buildSketchArtifactFile,
  exportSketchArtifact,
  parseUint8Snapshot,
  createSketchScanner,
} from 'pikelet-wasm/artifact';
import type {
  PikeletIndex,
  PikeletSketchArtifact,
  SearchResult,
  SketchArtifactSearchResult,
  SketchStageEvent,
  SketchTier,
} from 'pikelet-wasm';

// The entrypoints provide PikeletError / PIKELET_ERROR_CODES as named runtime
// exports, but NOT the artifact classes — those are properties of the API
// object (or named exports of pikelet-wasm/artifact). Importing the name only
// yields a type; using it as a value must not compile.
import { PikeletRangeArtifact as NotAValue } from 'pikelet-wasm';

async function nodeSurface(): Promise<void> {
  const index: PikeletIndex = await Pikelet.create({ dim: 8, metric: 'l2', maxElements: 100 });
  const hits: SearchResult[] = index.search(new Float32Array(8), 3, { efSearch: 50 });
  hits[0]?.distance.toFixed(3);
  index.dispose();

  // Staged sketch open with residency callbacks.
  const artifact: PikeletSketchArtifact = await Pikelet.openSketchArtifactFile('x.pikelet-sketch', {
    staged: true,
    onStage: (event: SketchStageEvent) => {
      const tier: SketchTier = event.tier;
      void tier;
      event.residentBytes.toFixed(0);
    },
  });
  const tier: SketchTier = artifact.tier;
  void tier;
  artifact.microDims.toFixed(0);
  artifact.microBits.toFixed(0);
  const settled: PikeletSketchArtifact = await artifact.fullyResident;
  void settled;

  const micro = await Pikelet.createSketchScanner(artifact, { tier: 'micro', maxRerank: 512 });
  micro.sketchDims.toFixed(0);
  const microTier: SketchTier = micro.tier;
  void microTier;

  const result: SketchArtifactSearchResult = await artifact.search(new Float32Array(artifact.dim), 5, {
    rerank: 200,
    microBoost: 4,
    microScanner: micro,
  });
  const servedBy: SketchTier = result.tier;
  void servedBy;
  result.rerank.toFixed(0);
  micro.dispose();
  await artifact.close();

  // Build options carry the micro-tier geometry; the manifest reports it.
  const manifest = Pikelet.buildSketchArtifactFile('snap.pnck', 'out.pikelet-sketch', {
    sketchDims: 192,
    sketchBits: 4,
    microDims: 48,
    microBits: 4,
  });
  if (manifest.micro) manifest.micro.stage1Bytes.toFixed(0);

  // @ts-expect-error search results always carry a tier field
  const missingTier: SketchArtifactSearchResult = { results: [], rerank: 0 };
  void missingTier;
}

async function webSurface(): Promise<void> {
  const artifact = await WebPikelet.RangeArtifact.open({
    async read(offset: number, length: number): Promise<Uint8Array> {
      void offset;
      return new Uint8Array(length);
    },
  });
  const searched = await artifact.search(new Float32Array(artifact.dim), 5);
  searched.results[0]?.distance.toFixed(3);
  await artifact.close();

  // @ts-expect-error the web surface has no Node file helpers
  WebPikelet.loadSnapshotFile;
}

async function artifactSubpath(): Promise<void> {
  const source = new NodeFileRangeSource('artifact.pikelet-range');
  const artifact = await RangeArtifactCtor.open(source);
  await artifact.close();

  const parsed = parseUint8Snapshot(new Uint8Array(0));
  parsed.scales.length.toFixed(0);
  const manifest = exportSketchArtifact(parsed, 'out.pikelet-sketch', { microDims: 24 });
  manifest.sizeBytes.toFixed(0);
  buildSketchArtifactFile('snap.pnck', 'out.pikelet-sketch');

  const sketch = await SketchArtifactCtor.openFile('out.pikelet-sketch');
  const scanner = await createSketchScanner(async () => ({}), sketch, { tier: 'full' });
  scanner.scan(new Float32Array(sketch.sketchDims), 100);
  scanner.dispose();
  await sketch.close();
}

void nodeSurface;
void webSurface;
void artifactSubpath;
// @ts-expect-error PikeletRangeArtifact is a type, not a value, on the entrypoints
void NotAValue;
void PikeletError;
void PIKELET_ERROR_CODES;
