// Complete-profile (kind-3 inline transformer) artifact assembly: the encoder
// declaration, the pinned weights download, and buildCompleteArtifact.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { sha256, loadCompleteModules, CliError, packBuildTimestamp, loadArtifactContract } from './common.mjs';
import { publicChunk, ingestDeclaration, SENTENCE_WINDOWS_KIND } from './ingest.mjs';
import { pickGoldenQueries, GOLDEN_K } from './goldens.mjs';
import { nodeFetch } from 'pikelet-wasm/complete/node-fetch.mjs';

async function buildCompleteArtifact({ projectDir, assetsDir, config, chunks, snapshot, vectors, rowOffsets = null, windowSize = 0, log = () => {}, logDetail = log }) {
  const artifactContract = await loadArtifactContract();
  const {
    assemblePikeletFile, buildCorpusSegment, buildInlineTransformerEncoderSegment, PROFILE_V2,
    buildQueryInterpSegment, buildLexicalSegment, measureRecommendedRerank, loadInlineEncoderKernel, buildRowMapSegment,
  } = (await loadCompleteModules()).builder;
  const { createInlineTransformerEmbedder, buildInlineTestVectors, openPikeletFile } = (await loadCompleteModules()).reader;
  const runtime = config.runtime || {};
  const { encoder, vocabPath, weightsPath } = await resolveInlineEncoderInputs(config, projectDir);

  const declaration = inlineEncoderDeclaration(config, encoder);
  const vocabText = await fs.readFile(vocabPath, 'utf8');
  const weightBytes = await fs.readFile(weightsPath);
  // The embedder for the declaration's verification vectors; created on
  // first need, disposed once they are built.
  let embedder = null;
  const getEmbedder = async () => {
    if (!embedder) {
      embedder = await createInlineTransformerEmbedder({
        declaration,
        vocabText,
        blob: weightBytes,
        createEncoder: await loadInlineEncoderKernel(),
      });
    }
    return embedder;
  };
  const lexical = buildLexicalSegment(chunks.map((chunk) => chunk.text || ''));
  logDetail(`Built lexical index: ${lexical.meta.terms.toLocaleString()} terms over ${lexical.meta.docCount.toLocaleString()} records (${(lexical.bytes.length / 1024).toFixed(0)} KiB)`);
  try {
    if (!Array.isArray(declaration.testVectors) || declaration.testVectors.length === 0) {
      // Contract section 4.4 mode 1: an inline encoder carries verification
      // vectors, so any reader can prove its kernel reproduces this build's
      // encoder before serving. One probe exceeds the kernel window to pin
      // the windowed mean-pool path.
      declaration.testVectors = await buildInlineTestVectors(await getEmbedder());
      logDetail(`Embedded ${declaration.testVectors.length} encoder verification vectors in the declaration`);
    }
  } finally {
    embedder?.dispose();
  }
  const inlineEncoderBytes = buildInlineTransformerEncoderSegment({
    declaration,
    vocabBytes: Buffer.from(vocabText, 'utf8'),
    weightBytes,
  });
  const unscored = Buffer.from(JSON.stringify({ kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '' }), 'utf8');
  const provisionalSketch = artifactContract.buildSketchArtifactBytes(snapshot, {
    sketchDims: runtime.sketchDims,
    sketchBits: runtime.sketchBits,
  });
  const rerankSweep = await measureRecommendedRerank({
    artifactModule: artifactContract,
    sketchBytes: provisionalSketch.bytes,
    queryVectors: vectors,
    snapshotBytes: snapshot,
  });
  logDetail(`Measured rerank operating point: C=${rerankSweep.recommendedRerank} `
    + `(recall@${rerankSweep.k} ${rerankSweep.recall} over ${rerankSweep.queries} ${rerankSweep.querySource} queries)`);
  if (rerankSweep.targetReached === false) {
    log(`warn: recall@${rerankSweep.k} target ${rerankSweep.targetRecall} not reached at the reader's default `
      + `rerank ceiling C=${rerankSweep.rerankCeiling}; queries are served at recall ${rerankSweep.recall} `
      + 'unless a caller passes a deeper rerank');
  }
  const sketch = artifactContract.buildSketchArtifactBytes(snapshot, {
    sketchDims: runtime.sketchDims,
    sketchBits: runtime.sketchBits,
    recommendedRerank: rerankSweep.recommendedRerank,
  });
  const records = chunks.map((chunk) => Buffer.from(JSON.stringify(publicChunk(chunk)), 'utf8'));
  // Absent unless SOURCE_DATE_EPOCH pins it, so the same inputs give the
  // same identity by default; pinned, the bytes are as they always were.
  const generatedAt = packBuildTimestamp();
  const evaluationBytes = (goldens) => Buffer.from(JSON.stringify({
    kind: 'docs-site-build-v1',
    ...(generatedAt ? { generatedAt } : {}),
    querySet: runtime.evaluation?.queries || [],
    // Each verified at build time to retrieve its source through the
    // assembled file's own query() (goldens.mjs), so the tests the file
    // carries are true of it by construction. expectId names a chunk id,
    // expectTitle any chunk of the titled document.
    goldenQueries: goldens,
    // COMPLETE_PROFILE.md section 5.4: the recall-vs-C measurements behind
    // this artifact's recommendedRerank.
    rerankSweep,
  }), 'utf8');
  const outPath = path.join(assetsDir, runtime.fileName || 'search.pikelet');
  const corpusSegment = buildCorpusSegment(records);
  const rowMapBytes = rowOffsets ? buildRowMapSegment(rowOffsets) : null;
  // lexical: the BM25 segment for hybrid retrieval (kind 5), built above.
  // Older readers skip the segment.
  const assemble = (goldens, calibration, target = outPath) => assemblePikeletFile({
    profile: PROFILE_V2,
    corpus: {
      ...corpusSegment.corpus,
      // How the records were made (LAYERED_PROFILE.md 6.1): with it a layer
      // can append without --assert-ingest, and cannot re-chunk mid-chain.
      ingest: ingestDeclaration(config.chunking),
      provenance: {
        source: config.source.type,
        name: config.name,
        // SPDX identifier (or free text) for the compiled content — set
        // with compile --license; packs meant for redistribution should
        // carry one.
        ...(config.license ? { license: config.license } : {}),
      },
    },
    dim: config.embedding.dims,
    metric: config.index.metric,
    encoder: {
      kind: 'inline-transformer-v1',
      model: declaration.model,
      pooling: declaration.pooling,
      normalized: declaration.normalized,
      maxTokens: declaration.maxTokens,
    },
    recommendedRerank: rerankSweep.recommendedRerank,
    lexical: lexical.meta,
    sampleQueries: runtime.sampleQueries || [],
    // Several index rows per record (sentence windows): the policy that made
    // the rows, committed under the identity alongside the row map itself.
    ...(rowMapBytes ? { index: { rowMap: { kind: SENTENCE_WINDOWS_KIND, sentences: windowSize, rows: rowOffsets[rowOffsets.length - 1] } } } : {}),
  }, [
    { kind: 'index', bytes: Buffer.from(sketch.bytes) },
    { kind: 'corpus', bytes: corpusSegment.bytes },
    { kind: 'query-interp', bytes: buildQueryInterpSegment(3, inlineEncoderBytes, calibration) },
    { kind: 'evaluation', bytes: evaluationBytes(goldens) },
    { kind: 'lexical', bytes: lexical.bytes },
    ...(rowMapBytes ? [{ kind: 'rowmap', bytes: Buffer.from(rowMapBytes) }] : []),
  ], target);
  // A pack carries a calibration only when one is supplied prebuilt
  // (encoder.calibrationPath: compile --calibration, the Docusaurus
  // plugin's completeProfile.calibration); compile fits none, and the
  // reader reports such a pack "unscored".
  const calibrationBytes = encoder.calibrationPath
    ? await fs.readFile(path.resolve(projectDir, encoder.calibrationPath))
    : unscored;
  let result = assemble([], calibrationBytes);
  // Golden queries are verified against the assembled file's own retrieval,
  // then embedded by reassembling it; the evaluation segment feeds no
  // retrieval, so they stay true of the final bytes. The query encoder comes
  // from loadInlineEncoderKernel (the web glue), as for the build's own
  // embedder: under the Docusaurus plugin this module runs through jiti's
  // CJS transform, where the reader's default node glue fails with
  // "require is not a function".
  const pack = await openPikeletFile(outPath, { createEncoder: await loadInlineEncoderKernel() });
  let picked;
  try {
    picked = await pickGoldenQueries({
      chunks, config, projectDir, log,
      query: async (text) => (await pack.query(text, { k: GOLDEN_K, showAbstained: true })).results,
    });
  } finally {
    await pack.close();
  }
  if (picked.goldens.length) result = assemble(picked.goldens, calibrationBytes);
  log(`Embedded ${picked.goldens.length} golden queries, each verified to retrieve its source`
    + (picked.authored ? ` (${picked.authoredKept} of your ${picked.authored} questions)` : ''));
  logDetail(`  ${picked.generatedKept} of ${picked.generated} generated questions retrieved their source in the top ${GOLDEN_K}`);
  return {
    profile: PROFILE_V2,
    kind: 'inline-transformer-v1',
    path: path.basename(outPath),
    bytes: result.fileBytes,
    sha256: sha256(fssync.readFileSync(outPath)).toString('hex'),
    identity: result.identity,
    segments: result.manifest.segments,
  };
}

function inlineEncoderDeclaration(config, encoder = {}) {
  const model = encoder.model || 'sentence-transformers/all-MiniLM-L6-v2';
  const pooling = config.embedding.pooling || 'mean';
  return encoder.declaration || {
    kind: 'inline-transformer-v1',
    model,
    license: encoder.license || 'apache-2.0',
    attribution: encoder.attribution || `${model} (quantized derivative)`,
    dim: config.embedding.dims,
    pooling,
    normalized: config.embedding.normalize !== false,
    // The compiled kernel's window is 512 tokens (encoder.cpp MAXSEQ); the
    // declaration states what this artifact's encoder actually does, so a
    // larger configured value cannot be declared.
    maxTokens: Math.min(encoder.maxTokens || 512, 512),
    // cls pooling only reads the first window (see inline-transformer.mjs
    // embed()); mean pooling combines every window into one average.
    longInputs: pooling === 'cls' ? 'first-window-cls' : 'windowed-mean-pool',
    prefixPolicy: {
      passage: config.embedding.prefixPolicy?.passage || '',
      query: config.embedding.prefixPolicy?.query || '',
    },
    layout: encoder.layout || { V: 30522, P: 512, T: 2, D: 384, F: 1536, L: 6, B: 64, H: 12 },
  };
}

// The 24.3 MiB weight blob is excluded from published tarballs
// (src/inline-encoder/.npmignore), so npm consumers fetch it once, pinned by
// digest, from the release asset into the user model cache (see
// defaultInlineWeightsPath in cli.mjs). A repo checkout carries the file in
// git, so this path never runs in-tree.
const INLINE_WEIGHTS_URL = 'https://github.com/mcn92/pikelet/releases/download/inline-encoder-v1/encoder-weights.bin';
const INLINE_WEIGHTS_SHA256 = '3b14685a73bd7f30477be8dad89902b6e4bb55e49ec325c9e071c462cf89089b';

// Fetches a digest-pinned weight blob to weightsPath. The default pin is the
// packaged MiniLM blob; create's build-time BGE encoder passes its own
// (MODEL_MAP in common.mjs). The bytes land under a temporary name and are
// renamed into place, so an interrupted or concurrent download never leaves
// a partial blob at weightsPath.
async function fetchInlineEncoderWeights(weightsPath, pin = { url: INLINE_WEIGHTS_URL, sha256: INLINE_WEIGHTS_SHA256, urlEnv: 'PIKELET_ENCODER_WEIGHTS_URL' }) {
  const url = process.env[pin.urlEnv] || pin.url;
  let body;
  try {
    const response = await nodeFetch(url, { redirect: 'follow' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    body = Buffer.from(await response.arrayBuffer());
  } catch (error) {
    throw new CliError(`Encoder weights not found at ${weightsPath} and the download from ${url} failed (${error.message}). `
      + `Next: place the encoder blob there yourself (sha256 ${pin.sha256}), or set ${pin.urlEnv} to a mirror.`, 2);
  }
  const digest = sha256(body).toString('hex');
  if (digest !== pin.sha256) {
    throw new CliError(`Downloaded encoder weights failed verification: sha256 ${digest} != pinned ${pin.sha256}. Refusing to use them.`, 2);
  }
  await fs.mkdir(path.dirname(weightsPath), { recursive: true });
  const partial = `${weightsPath}.${process.pid}.partial`;
  await fs.writeFile(partial, body);
  await fs.rename(partial, weightsPath);
}

async function resolveInlineEncoderInputs(config, projectDir) {
  const encoder = config.runtime?.inlineEncoder || {};
  const resolveInput = (value, label) => {
    if (!value) throw new CliError(`runtime.inlineEncoder.${label} is required for complete kind-3 artifacts`, 1);
    return path.resolve(projectDir, value);
  };
  const vocabPath = resolveInput(encoder.vocabPath, 'vocabPath');
  const weightsPath = resolveInput(encoder.weightsPath, 'weightsPath');
  if (!fssync.existsSync(vocabPath)) throw new CliError(`Inline encoder vocab not found: ${vocabPath}`, 1);
  if (!fssync.existsSync(weightsPath)) {
    // Only the packaged default blob is auto-fetched — the digest pin is
    // for that exact file. A non-default model must supply its own weights
    // even if its output happens to be named encoder-weights.bin too (the
    // export script's default output name): matching on the declared model
    // rather than just the basename avoids silently overwriting a missing
    // custom blob with MiniLM's.
    const isDefaultModel = !encoder.model || encoder.model === 'sentence-transformers/all-MiniLM-L6-v2';
    if (path.basename(weightsPath) !== 'encoder-weights.bin' || !isDefaultModel) {
      throw new CliError(`Inline encoder weights not found: ${weightsPath}`, 1);
    }
    await fetchInlineEncoderWeights(weightsPath);
  }
  return { encoder, vocabPath, weightsPath };
}

export {
  buildCompleteArtifact,
  inlineEncoderDeclaration,
  INLINE_WEIGHTS_URL,
  INLINE_WEIGHTS_SHA256,
  fetchInlineEncoderWeights,
  resolveInlineEncoderInputs,
};
