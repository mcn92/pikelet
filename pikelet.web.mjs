import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import simdWasmAsset from './dist/engine.wasm?url';
import scalarWasmAsset from './dist/engine.scalar.wasm?url';
import createPikeletApi from './pikelet-core.js';
import errorContract from './pikelet-errors.js';
import loaderContract from './pikelet-loader.js';
import artifactContract from './pikelet-artifact.js';
const { PikeletError, PIKELET_ERROR_CODES, pikeletError } = errorContract;
const { createCachedModuleLoader } = loaderContract;
const { PikeletRangeArtifact, PikeletSketchArtifact, createSketchScanner } = artifactContract;
let engineVariantPromise = null;

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return pikeletError(PIKELET_ERROR_CODES.WASM_LOAD_FAILED, `${message}: ${detail}`, undefined, error);
}

function isUrlLikeWasmAsset(asset) {
  return typeof asset === 'string' || asset instanceof URL;
}

async function loadWasmAsset(asset) {
  if (!isUrlLikeWasmAsset(asset)) return asset;
  const href = asset instanceof URL ? asset.href : asset;
  const response = await fetch(href, { credentials: 'same-origin' });
  if (!response.ok) {
    throw pikeletError(PIKELET_ERROR_CODES.WASM_LOAD_FAILED,
      `${response.status} ${response.statusText}`.trim(), { status: response.status });
  }
  return response.arrayBuffer();
}

const moduleLoader = createCachedModuleLoader((variant) =>
  loadWasmAsset(variant === 'simd' ? simdWasmAsset : scalarWasmAsset)
);

function selectEngineVariant() {
  engineVariantPromise ??= moduleLoader.supports('simd')
    .then((supported) => supported ? 'simd' : 'scalar')
    .catch((error) => { throw makeLoadError('Failed to probe Pikelet SIMD WASM support', error); });
  return engineVariantPromise;
}

async function loadWebEngine() {
  const engineVariant = await selectEngineVariant();

  if (engineVariant === 'scalar') {
    try {
      return await moduleLoader.instantiate(loadScalarEngine, 'scalar');
    } catch (error) {
      throw makeLoadError('Pikelet failed to load the scalar WASM engine', error);
    }
  }

  try {
    return await moduleLoader.instantiate(loadEngine, 'simd');
  } catch (simdError) {
    try {
      const engine = await moduleLoader.instantiate(loadScalarEngine, 'scalar');
      engineVariantPromise = Promise.resolve('scalar');
      return engine;
    } catch (scalarError) {
      throw makeLoadError(
        `Pikelet failed to load either the SIMD or scalar WASM engine (SIMD error: ${simdError && simdError.message ? simdError.message : String(simdError)})`,
        scalarError
      );
    }
  }
}

const Pikelet = createPikeletApi(loadWebEngine);
export { PikeletError, PIKELET_ERROR_CODES };

Pikelet.RangeArtifact = PikeletRangeArtifact;
Pikelet.SketchArtifact = PikeletSketchArtifact;
Pikelet.createSketchScanner = (artifact, options) => createSketchScanner(loadWebEngine, artifact, options);

function unsupportedNodeFileHelper(name) {
  return async function unsupported() {
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `${name}() is only available in the Node.js entrypoints`);
  };
}

Pikelet.loadSnapshotFile = unsupportedNodeFileHelper('loadSnapshotFile');
Pikelet.loadJsonFile = unsupportedNodeFileHelper('loadJsonFile');

export default Pikelet;
