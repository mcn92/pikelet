import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import loadEngine from './dist/engine.js';
import loadScalarEngine from './dist/engine.scalar.js';
import createPikeletApi from './pikelet-core.js';
import errorContract from './pikelet-errors.js';
import loaderContract from './pikelet-loader.js';
import artifactContract from './pikelet-artifact.js';
const { PikeletError, PIKELET_ERROR_CODES, pikeletError } = errorContract;
const { createCachedModuleLoader } = loaderContract;
const { PikeletRangeArtifact, PikeletSketchArtifact, createSketchScanner, NodeFileRangeSource, buildRangeArtifact, buildRangeArtifactFile, buildSketchArtifact, buildSketchArtifactBytes, buildSketchArtifactFile } = artifactContract;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
let engineVariantPromise = null;

const DEFAULT_JSON_FILE_BYTES = 64 * 1024 * 1024;
const DEFAULT_SNAPSHOT_FILE_BYTES = 512 * 1024 * 1024;
const SUPPORTED_SNAPSHOT_MAGICS = new Set([
  0x504E434B, // PNCK envelope
  0x464C4857, // FLHW raw float v0
  0x464C4831, // FLH1 raw float v1
  0x49384857, // I8HW raw uint8 v0
  0x49384831, // I8H1 raw uint8 v1
]);

function readWasmBinary(fileName) {
  try {
    const wasmBinary = readFileSync(path.join(__dirname, 'dist', fileName));
    return wasmBinary.buffer.slice(
      wasmBinary.byteOffset,
      wasmBinary.byteOffset + wasmBinary.byteLength
    );
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    throw pikeletError(PIKELET_ERROR_CODES.WASM_LOAD_FAILED, `Failed to load Pikelet WASM binary (${fileName}): ${message}`, { fileName }, error);
  }
}

function makeLoadError(message, error) {
  const detail = error && error.message ? error.message : String(error);
  return pikeletError(PIKELET_ERROR_CODES.WASM_LOAD_FAILED, `${message}: ${detail}`, undefined, error);
}

const moduleLoader = createCachedModuleLoader((variant) =>
  readWasmBinary(variant === 'simd' ? 'engine.wasm' : 'engine.scalar.wasm')
);

function selectEngineVariant() {
  engineVariantPromise ??= moduleLoader.supports('simd')
    .then((supported) => supported ? 'simd' : 'scalar');
  return engineVariantPromise;
}

function parseJsonLines(text, filePath) {
  const rows = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length === 0) continue;
    try {
      rows.push(JSON.parse(line));
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      throw pikeletError(PIKELET_ERROR_CODES.PARSE_FAILED, `Failed to parse JSONL in ${filePath} at line ${i + 1}: ${message}`, { filePath, line: i + 1 }, error);
    }
  }
  return rows;
}

function inferJsonFormat(filePath) {
  if (/\.json$/i.test(filePath)) return 'json';
  if (/\.(jsonl|ndjson)$/i.test(filePath)) return 'jsonl';
  throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `loadJsonFile() could not infer format from '${filePath}'. Use a .json/.jsonl/.ndjson extension or pass opts.format.`, { filePath });
}

function remapJsonRows(rows, vectorKey, idKey) {
  if (!Array.isArray(rows)) {
    throw pikeletError(PIKELET_ERROR_CODES.PARSE_FAILED, 'loadJsonFile() expects a JSON array or JSONL sequence of vectors/records');
  }
  return rows.map((row, i) => {
    if (row instanceof Float32Array || Array.isArray(row)) {
      return row;
    }
    if (!row || typeof row !== 'object') {
      throw pikeletError(PIKELET_ERROR_CODES.PARSE_FAILED, `loadJsonFile() expected an object or vector at index ${i}`, { index: i });
    }
    if (!(vectorKey in row)) {
      throw pikeletError(PIKELET_ERROR_CODES.PARSE_FAILED, `loadJsonFile() missing vectorKey '${vectorKey}' at index ${i}`, { index: i, vectorKey });
    }
    const mapped = { vector: row[vectorKey] };
    if (Object.prototype.hasOwnProperty.call(row, idKey)) {
      mapped.id = row[idKey];
    }
    return mapped;
  });
}

function validateFilePath(filePath, helperName) {
  if (typeof filePath !== 'string' || filePath.length === 0) {
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `${helperName}() requires a non-empty file path`);
  }
}

function validateMaxFileBytes(maxFileBytes, helperName) {
  if (!Number.isInteger(maxFileBytes) || maxFileBytes <= 0) {
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `${helperName}() maxFileBytes must be a positive integer`, { maxFileBytes });
  }
}

function statRegularFile(filePath, helperName) {
  let stat;
  try {
    stat = statSync(filePath);
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    throw pikeletError(PIKELET_ERROR_CODES.FILE_IO_FAILED, `${helperName}() could not stat ${filePath}: ${message}`, { filePath }, error);
  }
  if (!stat.isFile()) {
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `${helperName}() expected a regular file: ${filePath}`, { filePath });
  }
  return stat;
}

function enforceFileSize(stat, maxFileBytes, helperName, filePath) {
  if (stat.size > maxFileBytes) {
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `${helperName}() file exceeds maxFileBytes (${stat.size} > ${maxFileBytes}): ${filePath}`, { filePath, fileBytes: stat.size, maxFileBytes });
  }
}

function readUtf8FileWithLimit(filePath, helperName, maxFileBytes) {
  validateFilePath(filePath, helperName);
  validateMaxFileBytes(maxFileBytes, helperName);
  const stat = statRegularFile(filePath, helperName);
  enforceFileSize(stat, maxFileBytes, helperName, filePath);
  try {
    return readFileSync(filePath, 'utf8');
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    throw pikeletError(PIKELET_ERROR_CODES.FILE_IO_FAILED, `${helperName}() could not read ${filePath}: ${message}`, { filePath }, error);
  }
}

function readBinaryFileWithLimit(filePath, helperName, maxFileBytes) {
  validateFilePath(filePath, helperName);
  validateMaxFileBytes(maxFileBytes, helperName);
  const stat = statRegularFile(filePath, helperName);
  enforceFileSize(stat, maxFileBytes, helperName, filePath);
  try {
    return readFileSync(filePath);
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    throw pikeletError(PIKELET_ERROR_CODES.FILE_IO_FAILED, `${helperName}() could not read ${filePath}: ${message}`, { filePath }, error);
  }
}

function validateSnapshotBytes(snapshot, filePath) {
  if (!snapshot || snapshot.byteLength < 4) {
    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `loadSnapshotFile() snapshot is too small to be valid: ${filePath}`, { filePath });
  }
  const view = new DataView(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength);
  const magic = view.getUint32(0, true);
  if (!SUPPORTED_SNAPSHOT_MAGICS.has(magic)) {
    throw pikeletError(PIKELET_ERROR_CODES.SNAPSHOT_INVALID, `loadSnapshotFile() unsupported snapshot file type: ${filePath}`, { filePath, magic });
  }
}

async function loadNodeEngine() {
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

const Pikelet = createPikeletApi(loadNodeEngine);
export { PikeletError, PIKELET_ERROR_CODES };

Pikelet.RangeArtifact = PikeletRangeArtifact;
Pikelet.NodeFileRangeSource = NodeFileRangeSource;
Pikelet.buildRangeArtifact = buildRangeArtifact;
Pikelet.buildRangeArtifactFile = buildRangeArtifactFile;

Pikelet.openRangeArtifactFile = async function openRangeArtifactFile(filePath, opts) {
  validateFilePath(filePath, 'openRangeArtifactFile');
  return PikeletRangeArtifact.openFile(filePath, opts);
};

Pikelet.SketchArtifact = PikeletSketchArtifact;
Pikelet.createSketchScanner = (artifact, options) => createSketchScanner(loadNodeEngine, artifact, options);
Pikelet.buildSketchArtifact = buildSketchArtifact;
Pikelet.buildSketchArtifactBytes = buildSketchArtifactBytes;
Pikelet.buildSketchArtifactFile = buildSketchArtifactFile;

Pikelet.openSketchArtifactFile = async function openSketchArtifactFile(filePath, opts) {
  validateFilePath(filePath, 'openSketchArtifactFile');
  return PikeletSketchArtifact.openFile(filePath, opts);
};

Pikelet.loadSnapshotFile = async function loadSnapshotFile(filePath, opts) {
  const {
    maxFileBytes = DEFAULT_SNAPSHOT_FILE_BYTES,
    ...createOpts
  } = opts || {};

  const snapshot = readBinaryFileWithLimit(filePath, 'loadSnapshotFile', maxFileBytes);
  validateSnapshotBytes(snapshot, filePath);
  return Pikelet.restore(snapshot, createOpts);
};

Pikelet.loadJsonFile = async function loadJsonFile(filePath, opts = {}) {
  const {
    format = inferJsonFormat(filePath),
    vectorKey = 'vector',
    idKey = 'id',
    maxFileBytes = DEFAULT_JSON_FILE_BYTES,
    ...createOpts
  } = opts;

  if (format !== 'json' && format !== 'jsonl') {
    throw pikeletError(PIKELET_ERROR_CODES.INVALID_ARGUMENT, `loadJsonFile() format must be 'json' or 'jsonl', got '${format}'`, { format });
  }

  const text = readUtf8FileWithLimit(filePath, 'loadJsonFile', maxFileBytes);
  let rows;
  if (format === 'jsonl') {
    rows = parseJsonLines(text, filePath);
  } else {
    try {
      rows = JSON.parse(text);
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      throw pikeletError(PIKELET_ERROR_CODES.PARSE_FAILED, `Failed to parse JSON in ${filePath}: ${message}`, { filePath }, error);
    }
  }
  return Pikelet.fromVectors(remapJsonRows(rows, vectorKey, idKey), createOpts);
};

export default Pikelet;
