// Complete-profile (kind-3 inline transformer) artifact assembly: the encoder
// declaration, the pinned weights download, and buildCompleteArtifact.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { sha256, loadCompleteModules, CliError, loadArtifactContract } from './common.mjs';
import { publicChunk } from './ingest.mjs';
import { calibrateRetrievalAbstention } from './calibrate.mjs';

async function buildCompleteArtifact({ Pikelet, projectDir, assetsDir, config, chunks, snapshot, vectors, log = () => {} }) {
  const artifactContract = await loadArtifactContract();
  const {
    assemblePikeletFile, buildCorpusSegment, buildInlineTransformerEncoderSegment, PROFILE_V2,
    buildQueryInterpSegment, buildLexicalSegment, measureRecommendedRerank, loadInlineEncoderKernel,
  } = (await loadCompleteModules()).builder;
  const { createInlineTransformerEmbedder, buildInlineTestVectors, openPikeletFile, openLexicalIndex } = (await loadCompleteModules()).reader;
  const runtime = config.runtime || {};
  const { encoder, vocabPath, weightsPath } = await resolveInlineEncoderInputs(config, projectDir);

  const declaration = inlineEncoderDeclaration(config, encoder);
  const vocabText = await fs.readFile(vocabPath, 'utf8');
  const weightBytes = await fs.readFile(weightsPath);
  // One embedder serves both the declaration's verification vectors and
  // abstention calibration; created on first need, disposed after both.
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
  // Built early (before calibration, not in its usual position after
  // corpusSegment below) so calibration can fit coverage against the same
  // fused vector+BM25 ranking the reader serves, not a vector-only
  // approximation of it.
  const lexical = buildLexicalSegment(chunks.map((chunk) => chunk.text || ''));
  log(`Built lexical index: ${lexical.meta.terms.toLocaleString()} terms over ${lexical.meta.docCount.toLocaleString()} records (${(lexical.bytes.length / 1024).toFixed(0)} KiB)`);
  let calibrationBytes = null;
  let goldenQueries = [];
  try {
    if (!Array.isArray(declaration.testVectors) || declaration.testVectors.length === 0) {
      // Contract section 4.4 mode 1: an inline encoder carries verification
      // vectors, so any reader can prove its kernel reproduces this build's
      // encoder before serving. One probe exceeds the kernel window to pin
      // the windowed mean-pool path.
      declaration.testVectors = await buildInlineTestVectors(await getEmbedder());
      log(`Embedded ${declaration.testVectors.length} encoder verification vectors in the declaration`);
    }
    if (encoder.calibrationPath) {
      calibrationBytes = await fs.readFile(path.resolve(projectDir, encoder.calibrationPath));
    } else if (runtime.calibration === 'auto') {
      const embedQuery = async (text) => (await (await getEmbedder()).embed(`${declaration.prefixPolicy?.query || ''}${text}`)).vector;
      const lexicalIndex = openLexicalIndex(lexical.bytes);
      const calibrated = await calibrateRetrievalAbstention({ Pikelet, chunks, vectors, config, embedQuery, lexicalIndex, log });
      if (calibrated) {
        calibrationBytes = Buffer.from(JSON.stringify(calibrated.calibrationJson), 'utf8');
        // Retrieval-verified positives double as golden queries: embedded
        // in the evaluation segment, so any later reader can re-run the
        // pack's own tests from inside the file (mcp verify_pack).
        goldenQueries = calibrated.goldenQueries || [];
        const { verifiedPositiveQueries, foreignNegativeQueries, syntheticGibberishQueries, heldOutNegativeQueries, recombinationNegativeQueries, weakQueries, fitAuc, cvAuc, cvAucHard } = calibrated.summary;
        log(`Calibrated abstention: ${verifiedPositiveQueries} answerable / ${heldOutNegativeQueries + recombinationNegativeQueries} hard in-domain (${heldOutNegativeQueries} held-out-doc, ${recombinationNegativeQueries} recombination) / `
          + `${foreignNegativeQueries} off-domain / ${syntheticGibberishQueries} gibberish / ${weakQueries} weak queries, `
          + `5-fold CV AUC ${cvAuc ?? 'n/a'} pooled, ${cvAucHard ?? 'n/a'} vs hard negatives (fit AUC ${fitAuc}, in-sample)`);
      }
    }
  } finally {
    embedder?.dispose();
  }
  const inlineEncoderBytes = buildInlineTransformerEncoderSegment({
    declaration,
    vocabBytes: Buffer.from(vocabText, 'utf8'),
    weightBytes,
  });
  calibrationBytes ??= Buffer.from(JSON.stringify({ kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '' }), 'utf8');
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
  log(`Measured rerank operating point: C=${rerankSweep.recommendedRerank} `
    + `(recall@${rerankSweep.k} ${rerankSweep.recall} over ${rerankSweep.queries} ${rerankSweep.querySource} queries)`);
  const sketch = artifactContract.buildSketchArtifactBytes(snapshot, {
    sketchDims: runtime.sketchDims,
    sketchBits: runtime.sketchBits,
    recommendedRerank: rerankSweep.recommendedRerank,
  });
  const records = chunks.map((chunk) => Buffer.from(JSON.stringify(publicChunk(chunk)), 'utf8'));
  const generatedAt = new Date().toISOString();
  const evaluationBytes = (goldens) => Buffer.from(JSON.stringify({
    kind: 'docs-site-build-v1',
    generatedAt,
    querySet: runtime.evaluation?.queries || [],
    // Each verified at build time to retrieve its source — first against
    // the calibration index, then replayed through the assembled artifact
    // itself (below), so the tests the file carries are true of the file
    // by construction. expectId names a chunk id, expectTitle any chunk
    // of the titled document.
    goldenQueries: goldens,
    // COMPLETE_PROFILE.md section 5.4: the recall-vs-C measurements behind
    // this artifact's recommendedRerank.
    rerankSweep,
  }), 'utf8');
  const outPath = path.join(assetsDir, runtime.fileName || 'search.pikelet');
  const corpusSegment = buildCorpusSegment(records);
  // lexical (BM25 segment for hybrid retrieval, kind 5) was built earlier,
  // above, so calibration could fit against the same fused ranking it backs
  // at serve time. Older readers skip the segment.
  const assemble = (goldens) => assemblePikeletFile({
    profile: PROFILE_V2,
    corpus: {
      ...corpusSegment.corpus,
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
  }, [
    { kind: 'index', bytes: Buffer.from(sketch.bytes) },
    { kind: 'corpus', bytes: corpusSegment.bytes },
    { kind: 'query-interp', bytes: buildQueryInterpSegment(3, inlineEncoderBytes, calibrationBytes) },
    { kind: 'evaluation', bytes: evaluationBytes(goldens) },
    { kind: 'lexical', bytes: lexical.bytes },
  ], outPath);
  let result = assemble(goldenQueries);
  // Replay the candidate goldens through the assembled artifact's own
  // retrieval (sketch scan + rerank + hybrid ordering, inline encoder) —
  // the configuration a reader will actually run, which the calibration
  // index only approximates. A golden that does not reproduce there is
  // dropped and the file reassembled: verify_pack and acceptance tests
  // must pass by construction on the shipped bytes. (The evaluation
  // segment does not feed retrieval, so the replay stays valid across
  // the reassembly.)
  if (goldenQueries.length) {
    const replay = await openPikeletFile(outPath);
    let kept;
    try {
      kept = [];
      for (const golden of goldenQueries) {
        const out = await replay.query(golden.text, { k: 10 });
        const hit = golden.expectId !== undefined
          ? out.results.some((r) => r.id === golden.expectId)
          : out.results.some((r) => (r.title || '').trim() === golden.expectTitle);
        if (hit) kept.push(golden);
      }
    } finally {
      await replay.close();
    }
    if (kept.length !== goldenQueries.length) {
      log(`Golden replay against the assembled artifact: ${kept.length}/${goldenQueries.length} reproduce; embedding the ${kept.length} that do`);
      result = assemble(kept);
    } else {
      log(`Golden replay against the assembled artifact: ${kept.length}/${kept.length} reproduce`);
    }
  }
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
// digest, from the release asset. A repo checkout carries the file in git,
// so this path never runs in-tree.
const INLINE_WEIGHTS_URL = 'https://github.com/mcn92/pikelet/releases/download/inline-encoder-v1/encoder-weights.bin';
const INLINE_WEIGHTS_SHA256 = '3b14685a73bd7f30477be8dad89902b6e4bb55e49ec325c9e071c462cf89089b';

async function fetchInlineEncoderWeights(weightsPath) {
  const url = process.env.PIKELET_ENCODER_WEIGHTS_URL || INLINE_WEIGHTS_URL;
  let body;
  try {
    const response = await fetch(url, { redirect: 'follow' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    body = Buffer.from(await response.arrayBuffer());
  } catch (error) {
    throw new CliError(`Inline encoder weights not found at ${weightsPath} and the download from ${url} failed (${error.message}). `
      + `Next: place the encoder blob there yourself (sha256 ${INLINE_WEIGHTS_SHA256}), or set PIKELET_ENCODER_WEIGHTS_URL to a mirror.`, 2);
  }
  const digest = sha256(body).toString('hex');
  if (digest !== INLINE_WEIGHTS_SHA256) {
    throw new CliError(`Downloaded encoder weights failed verification: sha256 ${digest} != pinned ${INLINE_WEIGHTS_SHA256}. Refusing to use them.`, 2);
  }
  await fs.mkdir(path.dirname(weightsPath), { recursive: true });
  await fs.writeFile(weightsPath, body);
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
