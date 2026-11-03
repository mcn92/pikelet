// Shared by the pikelet modules: package paths and version, config
// defaults, the model table, CliError, and the loaders that resolve pikelet-wasm
// (engine, artifact layer, complete profile). All three resolve the bare
// package: the workspace link inside this repository, the installed
// dependency for npm consumers.

import crypto from 'node:crypto';
import fssync from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { nodeFetch } from 'pikelet-wasm/complete/node-fetch.mjs';

const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest();
let completeModulesPromise = null;
function loadCompleteModules() {
  completeModulesPromise ??= (async () => {
    try {
      return {
        builder: await import('pikelet-wasm/complete/builder'),
        reader: await import('pikelet-wasm/complete'),
      };
    } catch (error) {
      throw new CliError(`could not resolve pikelet-wasm/complete; install pikelet-wasm >= 0.3 alongside pikelet. ${error.message.split('\n')[0]}`, 2);
    }
  })();
  return completeModulesPromise;
}

// A pack's digests catch altered bytes on any transport, but plain http
// still shows every observer which pack and which passages are read, and
// lets one stall or block them. Loopback is exempt: that is local testing.
function plainHttpWarning(location) {
  let url;
  try { url = new URL(location); } catch { return null; }
  if (url.protocol !== 'http:') return null;
  if (url.hostname === 'localhost' || url.hostname === '[::1]' || /^127\./.test(url.hostname)) return null;
  return `warning: ${location} is plain http: the pack's digests still refuse altered bytes, but anyone on the path `
    + 'can see what is read and can stall or block it. Serve packs over https.';
}

/**
 * Resolve chain members for a producer: a `https?://` location becomes an
 * initialized range source, anything else a resolved local path. Layered
 * producers (`append --parent`, `compact --head`) handed their arguments to
 * openPikeletChain verbatim, so a URL was treated as a filename and failed at
 * open -- while `mcp` had mounted remote chains this way all along. The
 * capability was the reader's; only the producers were local-only.
 *
 * `location#<sha256>` pins that member's manifest identity, the same form
 * `mcp --pack` takes and every producer's usage line advertises. Pins used to
 * pass through untouched: a local `x.pikelet#<sha>` failed as a missing file,
 * and a URL's fragment never reached the wire, so the pin was silently
 * ignored. Each pinned member's manifest is now read and compared before the
 * caller opens anything. Chain links commit to parent identities, so pinning
 * the last member pins the whole chain.
 */
async function resolveChainMembers(locations, log = () => {}) {
  const { reader } = await loadCompleteModules();
  const resolved = [];
  for (const raw of locations) {
    const pin = /^(.*)#([0-9a-f]{64})$/i.exec(raw);
    const loc = pin ? pin[1] : raw;
    // A malformed pin must not degrade to "unpinned": a URL fragment never
    // reaches the server, and a short hex suffix on a path reads as a typo'd
    // pin, not a filename.
    if (!pin && (/^https?:\/\/[^#]*#/i.test(raw) || /#[0-9a-f]+$/i.test(raw))) {
      throw new CliError(`${raw}: a '#' pin must be the full 64-hex sha256 manifest identity`);
    }
    let member;
    if (/^https?:\/\//i.test(loc)) {
      const insecure = plainHttpWarning(loc);
      if (insecure) log(insecure);
      const source = reader.httpRangeSource(loc, { fetch: nodeFetch });
      await source.init();
      log(`Reading ${loc} over HTTP range requests`);
      member = source;
    } else {
      member = path.resolve(process.cwd(), loc);
    }
    if (pin) {
      const expected = pin[2].toLowerCase();
      let identity;
      try {
        ({ identity } = await reader.readMemberManifest(member));
      } catch (err) {
        throw new CliError(`${loc}: ${err.message}`);
      }
      if (identity !== expected) {
        throw new CliError(`${loc}: identity mismatch: pinned ${expected}, found ${identity}; `
          + 'refusing to build on a member that is not the one named');
      }
      log(`Pinned ${path.basename(loc)} at ${expected.slice(0, 12)}…`);
    }
    resolved.push(member);
  }
  return resolved;
}

/**
 * The `parent.locator` a new layer records (LAYERED_PROFILE.md 5.1.1): the
 * explicit --parent-locator, validated here rather than written blind and
 * refused by every reader later; else, when the parent is a local file at or
 * below the output's directory, its relative path. Producers wrote no locator
 * unless told to, so a layer from `append` or `rebase` could not be walked
 * from its head, and the next rebase of it needed --old-parent spelled out.
 * A URL parent, or one outside the output directory, gets none: a locator is
 * a relative reference under the child and cannot name either.
 */
async function parentLocatorFor(flags, parentMember, outPath, log = () => {}) {
  const { validateLocatorShape } = await import('pikelet-wasm/complete/layer-locator.mjs');
  const explicit = flags['parent-locator'];
  if (explicit) {
    try {
      validateLocatorShape(explicit);
    } catch (err) {
      throw new CliError(`--parent-locator ${explicit}: ${err.message} (5.1.1)`);
    }
    return explicit;
  }
  if (typeof parentMember !== 'string') return null;
  const rel = path.relative(path.dirname(outPath), parentMember).split(path.sep).join('/');
  try {
    validateLocatorShape(rel);
  } catch {
    log(`No parent locator recorded: ${parentMember} is not under the output directory `
      + '(pass --parent-locator to name one)');
    return null;
  }
  log(`Parent locator: ${rel}`);
  return rel;
}

/**
 * Read ONE segment's raw bytes out of a .pikelet, digest-verified, without
 * opening it as a searchable reader.
 *
 * `rebase` (LAYERED_PROFILE.md 6.2) needs this: it copies a layer's index,
 * corpus, lexical and query-interp segments verbatim, and a layer with records
 * cannot be opened alone (4.1 refuses it) nor read through
 * readChainMemberShell, which returns a tier only for the tombstone-only case.
 *
 * Returns null when the segment is absent. Throws if it is present but fails
 * its digest, so a copy never carries corrupt bytes forward.
 */
async function readChainMemberShellBytes(filePath, kind) {
  const crypto = await import('node:crypto');
  // Header and table geometry come from the format module, never from a literal
  // here: TABLE_ENTRY_BYTES is 48, and a hardcoded 24 reads garbage offsets
  // while still parsing, which is the worst possible failure for a byte copy.
  const { MAGIC, HEADER_BYTES, TABLE_ENTRY_BYTES, KINDS, KIND_NAMES } = await import('pikelet-wasm/complete/format.mjs');
  if (typeof filePath !== 'string') {
    throw new CliError('rebase copies a layer\'s bytes from a local file; download the layer first');
  }
  const fh = await (await import('node:fs/promises')).open(filePath, 'r');
  // A short read leaves zero-filled bytes that still parse; refuse it instead.
  const readExact = async (length, position, what) => {
    const buf = Buffer.alloc(length);
    const { bytesRead } = await fh.read(buf, 0, length, position);
    if (bytesRead !== length) throw new CliError(`${filePath}: truncated ${what} (${bytesRead} of ${length} bytes)`);
    return buf;
  };
  try {
    const fileBytes = (await fh.stat()).size;
    const head = await readExact(HEADER_BYTES, 0, 'header');
    if (head.readUInt32LE(0) !== MAGIC) throw new CliError(`${filePath}: not a .pikelet file (bad magic)`);
    const manifestBytes = head.readUInt32LE(8);
    const segmentCount = head.readUInt32LE(12);
    if (HEADER_BYTES + manifestBytes + segmentCount * TABLE_ENTRY_BYTES > fileBytes) {
      throw new CliError(`${filePath}: header declares a manifest and segment table larger than the file`);
    }
    const manifestBuf = await readExact(manifestBytes, HEADER_BYTES, 'manifest');
    // The manifest digest IS the identity; the per-segment digests below are
    // only as good as this check.
    const identity = head.subarray(24, 56).toString('hex');
    if (crypto.createHash('sha256').update(manifestBuf).digest('hex') !== identity) {
      throw new CliError(`${filePath}: manifest fails identity verification; refusing to copy from it`);
    }
    const manifest = JSON.parse(manifestBuf.toString('utf8'));
    if (!Array.isArray(manifest.segments) || manifest.segments.length !== segmentCount) {
      throw new CliError(`${filePath}: segment table disagrees with the manifest`);
    }
    const table = await readExact(segmentCount * TABLE_ENTRY_BYTES, HEADER_BYTES + manifestBytes, 'segment table');
    let found = null;
    for (let i = 0; i < segmentCount; i++) {
      const at = i * TABLE_ENTRY_BYTES;
      const tableKind = KIND_NAMES[table.readUInt32LE(at)];
      const declared = manifest.segments[i];
      // An extension kind unknown to both sides is skipped (spec 3.3).
      if (tableKind === undefined && !Object.hasOwn(KINDS, declared.kind)) continue;
      // The table sits outside the identity; the manifest's kind is what the
      // identity commits to, so the two must agree before bytes are trusted.
      if (tableKind !== declared.kind) {
        throw new CliError(`${filePath}: segment ${i} is ${tableKind} in the table but ${declared.kind} in the manifest`);
      }
      if (tableKind !== kind) continue;
      if (found) throw new CliError(`${filePath}: carries more than one ${kind} segment`);
      const offset = Number(table.readBigUInt64LE(at + 8));
      const length = Number(table.readBigUInt64LE(at + 16));
      if (length !== declared.bytes || offset + length > fileBytes) {
        throw new CliError(`${filePath}: the ${kind} segment's extent disagrees with the manifest`);
      }
      const bytes = await readExact(length, offset, `${kind} segment`);
      const actual = crypto.createHash('sha256').update(bytes).digest('hex');
      if (declared.sha256 !== actual) {
        throw new CliError(`${filePath}: the ${kind} segment fails its manifest digest; refusing to copy it`);
      }
      found = new Uint8Array(bytes);
    }
    return found;
  } finally {
    await fh.close();
  }
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PACKAGE_ROOT = path.resolve(__dirname, '..');
const STUDENT_TRAINER = path.join(PACKAGE_ROOT, 'tools', 'train_student.py');
const OWN_PACKAGE = JSON.parse(fssync.readFileSync(path.join(PACKAGE_ROOT, 'package.json'), 'utf8'));
const CLI_VERSION = OWN_PACKAGE.version;
// The generated project must run its artifacts on a pikelet-wasm at least as
// new as the one that built them (range artifact format revisions are not
// readable by older readers), so it inherits this package's own range rather
// than carrying a second hardcoded one that drifts.
const PIKELET_WASM_RANGE = OWN_PACKAGE.dependencies['pikelet-wasm'];
const DEFAULT_PREFIX = 'Represent this sentence for searching relevant passages: ';
const CONFIG_SCHEMA_URL = 'https://raw.githubusercontent.com/mcn92/pikelet/main/packages/pikelet/schemas/v1/pikelet.config.schema.json';
const DEFAULT_CONFIG = Object.freeze({
  chunking: { targetTokens: 256, overlapPercent: 15 },
  embedding: {
    mode: 'workers-ai',
    buildModel: 'bge-small-en-v1.5',
    dims: 384,
    prefixPolicy: { passage: '', query: DEFAULT_PREFIX },
    pooling: 'mean',
    normalize: true,
  },
  index: { metric: 'cosine', quantized: true, M: 16, efConstruction: 200, efSearch: 120 },
  runtime: { mode: 'snapshot', storage: 'bundled' },
  validation: { minRecallAt10: 0.98 },
});
const RANGE_ARTIFACT_MAGIC = 0x31415250;
// workers-ai builds embed passages locally with the inline-transformer kernel
// running a u8 export of the same model Workers AI serves for queries
// (examples/one-file-search/encoder-spike/export_encoder_blob.py --model
// BAAI/bge-small-en-v1.5 --bits 8 at hfRevision, which reproduces the pinned
// bytes). Its vocabulary is the packaged bert-base-uncased WordPiece table.
// Measured on BEIR scifact against live Workers AI query vectors: nDCG@10
// 0.6991, against 0.6992 for the @xenova/transformers path it replaced.
const MODEL_MAP = Object.freeze({
  'bge-small-en-v1.5': {
    dims: 384,
    workersAiModel: '@cf/baai/bge-small-en-v1.5',
    hfModel: 'BAAI/bge-small-en-v1.5',
    hfRevision: '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
    license: 'mit',
    maxInputTokens: 512,
    layout: { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 12, B: 64, H: 12 },
    weights: {
      file: 'bge-small-en-v1.5-u8.bin',
      url: 'https://github.com/mcn92/pikelet/releases/download/build-encoder-bge-v1/bge-small-en-v1.5-u8.bin',
      sha256: '3be1ede506f1826717a76593e6d614a15e15de8caa34fa209ab0089f0d613385',
      urlEnv: 'PIKELET_BGE_WEIGHTS_URL',
    },
  },
});
class CliError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.exitCode = exitCode;
  }
}
// The time stamped into build metadata (the evaluation segment's
// generatedAt, the project manifest). Those bytes are inside the artifact's
// identity, so a wall-clock stamp gives every rebuild of identical inputs a
// new identity. SOURCE_DATE_EPOCH (reproducible-builds.org: integer seconds
// since the epoch) pins it, making the build byte-for-byte reproducible;
// unset, it is the current time as before.
function buildTimestamp() {
  const raw = process.env.SOURCE_DATE_EPOCH;
  if (raw === undefined || raw === '') return new Date().toISOString();
  const seconds = /^\d+$/.test(raw) ? Number(raw) : NaN;
  // 8.64e12 s is the last instant a JavaScript Date can represent.
  if (!Number.isSafeInteger(seconds) || seconds > 8.64e12) {
    throw new CliError(`SOURCE_DATE_EPOCH must be a whole number of seconds since 1970-01-01, got ${JSON.stringify(raw)}`, 1);
  }
  return new Date(seconds * 1000).toISOString();
}
async function loadArtifactContract() {
  let mod;
  try {
    mod = await import('pikelet-wasm/artifact');
  } catch (error) {
    throw new CliError(`could not resolve pikelet-wasm/artifact; install pikelet-wasm alongside pikelet. ${error.message.split('\n')[0]}`, 2);
  }
  const contract = mod.default || mod;
  if (typeof contract.buildSketchArtifactBytes === 'function') return contract;
  throw new CliError('pikelet-wasm/artifact does not expose buildSketchArtifactBytes; install pikelet-wasm >= 0.3 alongside pikelet.', 2);
}
async function loadPikelet() {
  return (await import('pikelet-wasm')).default;
}

export {
  sha256,
  completeModulesPromise,
  loadCompleteModules,
  resolveChainMembers,
  plainHttpWarning,
  parentLocatorFor,
  readChainMemberShellBytes,
  __filename,
  __dirname,
  PACKAGE_ROOT,
  STUDENT_TRAINER,
  OWN_PACKAGE,
  CLI_VERSION,
  PIKELET_WASM_RANGE,
  DEFAULT_PREFIX,
  CONFIG_SCHEMA_URL,
  DEFAULT_CONFIG,
  RANGE_ARTIFACT_MAGIC,
  MODEL_MAP,
  CliError,
  buildTimestamp,
  loadArtifactContract,
  loadPikelet,
};
