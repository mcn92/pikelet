import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);

let profile = false;
for (const arg of args) {
  if (arg === '--profile') {
    profile = true;
    continue;
  }

  console.error(`Unknown option: ${arg}`);
  console.error('Usage: node scripts/build-engine.mjs [--profile]');
  process.exit(1);
}

const env = process.env;
const outBasename = env.OUT_BASENAME ?? 'engine';
const patchEngineJs = env.PATCH_ENGINE_JS ?? '1';
const wasmSimd = env.WASM_SIMD ?? '1';
const wasmRelaxedSimd = env.WASM_RELAXED_SIMD ?? '0';
const homeDir = os.homedir();
const defaultEmCache = path.join(homeDir, 'emsdk', 'upstream', 'emscripten', 'cache');
const fallbackEmCache = env.EM_CACHE_FALLBACK ?? path.join(os.tmpdir(), 'pikelet-emcc-cache');

function run(command, commandArgs, options = {}) {
  const result = spawnSync(command, commandArgs, {
    stdio: 'inherit',
    env: options.env ?? env,
    shell: options.shell ?? false,
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }

  return result;
}

function capture(command, commandArgs, options = {}) {
  const result = spawnSync(command, commandArgs, {
    encoding: 'utf8',
    env: options.env ?? env,
    shell: options.shell ?? false,
  });

  if (result.error) {
    return null;
  }

  if (result.status !== 0) {
    return null;
  }

  return result.stdout;
}

function pythonCandidates() {
  return process.platform === 'win32'
    ? [['py', ['-3']], ['python3', []], ['python', []]]
    : [['python3', []], ['python', []]];
}

function shellWords(value) {
  const words = value.match(/(?:[^\s"]+|"[^"]*")+/g);
  if (!words) return [];
  return words.map((word) => word.replace(/^"|"$/g, ''));
}

function canWriteDir(dirPath) {
  try {
    fs.accessSync(dirPath, fs.constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

function nodeSupportsRelaxedSimd() {
  const output = capture('node', ['--v8-options']);
  return output ? output.includes('--experimental-wasm-relaxed-simd') : false;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = -1;
  do {
    value /= 1024;
    unitIndex += 1;
  } while (value >= 1024 && unitIndex < units.length - 1);
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function printFileDetails(filePath) {
  const stats = fs.statSync(filePath);
  console.log(`${formatSize(stats.size)}  ${filePath}`);
}

const buildEnv = { ...env };
if (!buildEnv.EM_CACHE && !canWriteDir(defaultEmCache)) {
  buildEnv.EM_CACHE = fallbackEmCache;
  fs.mkdirSync(buildEnv.EM_CACHE, { recursive: true });
  console.log(`Using fallback Emscripten cache: ${buildEnv.EM_CACHE}`);
}

const debugSymbols = buildEnv.DEBUG_SYMBOLS === '1';
const buildDesc = debugSymbols ? 'Debug' : 'Optimized';
const optFlags = debugSymbols ? ['-O2'] : ['-O3'];
const debugFlags = debugSymbols
  ? ['-gsource-map', '-g', '--source-map-base', 'http://localhost/']
  : ['-g0'];

console.log('==============================================');
console.log(`Building Pikelet WASM Engine (${buildDesc})`);
console.log('==============================================');

fs.mkdirSync('dist', { recursive: true });

let simdDesc = 'scalar';
const simdFlags = [];
let nodeWasmFlags = [];
if (wasmSimd === '1') {
  simdFlags.push('-msimd128');
  simdDesc = 'WASM SIMD';
  if (wasmRelaxedSimd === '1') {
    simdFlags.push('-mrelaxed-simd');
    simdDesc = 'WASM SIMD + relaxed SIMD';
    if (nodeSupportsRelaxedSimd()) {
      nodeWasmFlags = ['--experimental-wasm-relaxed-simd'];
    }
  }
}

const profileFlags = [];
let profileDesc = 'profiling off';
if (profile) {
  profileFlags.push('-D', 'PIKELET_UINT8_HNSW_BUILD_PROFILE=1');
  profileDesc = 'profiling on';
}

console.log(`Compiling with ${simdDesc} + Memory Access (${profileDesc}, exceptions enabled)...`);

const emccCommand = shellWords(
  buildEnv.EMCC ?? `python3 ${path.join(homeDir, 'emsdk', 'upstream', 'emscripten', 'emcc.py')}`
);

if (emccCommand.length === 0) {
  console.error('EMCC command is empty');
  process.exit(1);
}

const emccArgs = [
  ...emccCommand.slice(1),
  ...optFlags,
  ...debugFlags,
  '--closure',
  '0',
  '-s',
  'WASM=1',
  '-s',
  'ALLOW_MEMORY_GROWTH=1',
  '-s',
  'INITIAL_MEMORY=16777216',
  '-s',
  'MALLOC=emmalloc',
  '-s',
  'WASM_BIGINT=1',
  '-s',
  'ALLOW_TABLE_GROWTH=0',
  '-s',
  'DYNAMIC_EXECUTION=0',
  '-s',
  'EMULATE_FUNCTION_POINTER_CASTS=0',
  '-s',
  'SUPPORT_LONGJMP=0',
  ...simdFlags,
  '-mnontrapping-fptoint',
  '-msign-ext',
  '-mbulk-memory',
  ...profileFlags,
  '-D',
  'PIKELET_WASM_BUILD=1',
  '-fno-merge-all-constants',
  '-s',
  'EXPORTED_FUNCTIONS=["_pikelet_init","_pikelet_add","_pikelet_bulk_insert","_pikelet_query","_pikelet_query_filtered","_pikelet_delete","_pikelet_compact","_pikelet_compact_remap","_pikelet_count","_pikelet_memory","_pikelet_ghost_count","_pikelet_ghost_ratio","_pikelet_export","_pikelet_import","_pikelet_dispose","_pikelet_dimension","_pikelet_sketch_scan","_pikelet_shutdown_all","_shutdown_all","_emsc_malloc","_emsc_free","_pikelet_profile_print","_pikelet_profile_reset"]',
  '-s',
  'EXPORTED_RUNTIME_METHODS=["ccall","cwrap","HEAPF32","HEAPU8","HEAPU32","HEAP32"]',
  '-s',
  'MODULARIZE=1',
  '-s',
  'EXPORT_NAME=P',
  '-s',
  'ENVIRONMENT=web,node',
  '-s',
  'FILESYSTEM=0',
  '-s',
  'ASSERTIONS=0',
  '-s',
  'DISABLE_EXCEPTION_CATCHING=0',
  '-s',
  'SINGLE_FILE=0',
  '-s',
  'STACK_SIZE=65536',
  '-fno-rtti',
  '-ffast-math',
  '-fvectorize',
  '-fslp-vectorize',
  '--no-entry',
  '-Isrc',
  '-o',
  `dist/${outBasename}.js`,
  'src/engine.cpp',
];

run(emccCommand[0], emccArgs, { env: buildEnv });

if (patchEngineJs === '1' && outBasename === 'engine') {
  console.log('');
  console.log('Applying engine.js patches...');
  let patched = false;
  for (const [command, baseArgs] of pythonCandidates()) {
    const result = spawnSync(command, [...baseArgs, 'scripts/patch_engine.py'], {
      stdio: 'inherit',
      env: buildEnv,
    });
    if (!result.error && result.status === 0) {
      patched = true;
      break;
    }
  }
  if (!patched) {
    console.error('Unable to run patch_engine.py with an available Python interpreter');
    process.exit(1);
  }
}

console.log('');
console.log('Build complete!');
printFileDetails(`dist/${outBasename}.js`);
printFileDetails(`dist/${outBasename}.wasm`);
console.log('Running test...');
run('node', [...nodeWasmFlags, 'run_tests.js'], { env: buildEnv });
