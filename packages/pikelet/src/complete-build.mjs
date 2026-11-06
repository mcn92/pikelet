// Complete-profile (kind-3 inline transformer) artifact assembly: the encoder
// declaration, the pinned weights download, and buildCompleteArtifact.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { sha256, loadCompleteModules, CliError, packBuildTimestamp, loadArtifactContract } from './common.mjs';
import { publicChunk, ingestDeclaration, SENTENCE_WINDOWS_KIND } from './ingest.mjs';
import { calibrateRetrievalAbstention } from './calibrate.mjs';
import { nodeFetch } from 'pikelet-wasm/complete/node-fetch.mjs';

async function buildCompleteArtifact({ projectDir, assetsDir, config, chunks, snapshot, vectors, rowOffsets = null, windowSize = 0, log = () => {} }) {
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
  let calibrationSummary = null;
  try {
    if (!Array.isArray(declaration.testVectors) || declaration.testVectors.length === 0) {
      // Contract section 4.4 mode 1: an inline encoder carries verification
      // vectors, so any reader can prove its kernel reproduces this build's
      // encoder before serving. One probe exceeds the kernel window to pin
      // the windowed mean-pool path.
      declaration.testVectors = await buildInlineTestVectors(await getEmbedder());
      log(`Embedded ${declaration.testVectors.length} encoder verification vectors in the declaration`);
    }
  } catch (error) {
    embedder?.dispose();
    throw error;
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
  log(`Measured rerank operating point: C=${rerankSweep.recommendedRerank} `
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
    // Each verified at build time to retrieve its source — first by
    // calibration, through the uncalibrated artifact's own retrieval, then
    // replayed through the assembled artifact's query() (below), so the
    // tests the file carries are true of the file by construction. expectId names a chunk id, expectTitle any chunk
    // of the titled document.
    goldenQueries: goldens,
    // COMPLETE_PROFILE.md section 5.4: the recall-vs-C measurements behind
    // this artifact's recommendedRerank.
    rerankSweep,
    // Calibration quality, kept for audit after the file ships. method and
    // target say what matchQuality measures (self-templates-v6: relevance,
    // fit against off-topic queries). fitAuc is in-sample and optimistic;
    // cvAuc is the pooled 5-fold held-out AUC against off-topic queries
    // (the gate); cvAucHard is the in-domain-unanswerable diagnostic,
    // never fit or gated. realQueryAuc/realQueryAbstentionRate are the
    // only real validation this asset can have: human-written queries
    // (runtime.calibrationQueries), held out of the fit entirely. null
    // when the corpus shipped none. A pack that shipped without
    // calibration (encoder.calibrationPath set, or the corpus failed the
    // fit gates) carries calibration: null here.
    ...(calibrationSummary ? {
      calibration: {
        method: calibrationSummary.method,
        target: calibrationSummary.target,
        fitAuc: calibrationSummary.fitAuc,
        cvAuc: calibrationSummary.cvAuc,
        cvAucHard: calibrationSummary.cvAucHard,
        // A regression specific to the substituted (low-coverage) class is
        // exactly the failure the pooled numbers above can hide — see
        // calibrate.mjs's design note.
        cvAucByGenKind: calibrationSummary.cvAucByGenKind,
        verifiedPositiveQueries: calibrationSummary.verifiedPositiveQueries,
        positivesByGenKind: calibrationSummary.positivesByGenKind,
        ablationNegativeQueries: calibrationSummary.ablationNegativeQueries,
        entitySwapNegativeQueries: calibrationSummary.entitySwapNegativeQueries,
        weakQueries: calibrationSummary.weakQueries,
        humanCalibrationQueries: calibrationSummary.humanCalibrationQueries,
        realQueryAuc: calibrationSummary.realQueryAuc,
        realQueryAbstentionRate: calibrationSummary.realQueryAbstentionRate,
        // Comparison-only: how a semantic (embedding) word-grounding signal
        // would separate positives from hard negatives versus the lexical
        // coverage1 feature that actually feeds the fit. Not used for
        // scoring — two attempts at blending it in (see calibrate.mjs's
        // GROUNDING_FEAT design note) both made real-query behavior worse.
        maxSimVsCoverage: calibrationSummary.maxSimVsCoverage,
        maxSimInFit: calibrationSummary.maxSimInFit,
      },
    } : { calibration: null }),
  }), 'utf8');
  const outPath = path.join(assetsDir, runtime.fileName || 'search.pikelet');
  const corpusSegment = buildCorpusSegment(records);
  const rowMapBytes = rowOffsets ? buildRowMapSegment(rowOffsets) : null;
  // lexical (BM25 segment for hybrid retrieval, kind 5) was built earlier,
  // above, so calibration could fit against the same fused ranking it backs
  // at serve time. Older readers skip the segment.
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
  try {
    if (encoder.calibrationPath) {
      calibrationBytes = await fs.readFile(path.resolve(projectDir, encoder.calibrationPath));
    } else if (runtime.calibration === 'auto') {
      const embedQuery = async (text) => (await (await getEmbedder()).embed(`${declaration.prefixPolicy?.query || ''}${text}`)).vector;
      // Word-level vectors for maxSimFrac (comparison feature — see
      // calibrate.mjs's design note): no query/passage prefix, since a
      // single word is not itself a query needing an instruction prefix —
      // that would distort the word's own meaning vector for no benefit.
      const embedWordVecs = async (text) => (await getEmbedder()).embedWords(text);
      // Calibration retrieves through the artifact itself: the complete
      // file is assembled unscored, opened with the reader, and every probe
      // runs the reader's own retrieval (sketch scan, BM25 candidates,
      // exact rerank, fusion, phrase pin). The calibration segment feeds
      // none of that, so the evidence gathered here is exactly what the
      // calibrated file below serves.
      const uncalibratedPath = `${outPath}.uncalibrated`;
      let calibrated;
      try {
        assemble([], unscored, uncalibratedPath);
        const uncalibrated = await openPikeletFile(uncalibratedPath);
        try {
          const evidence = (text, options) => uncalibrated.__calibrationEvidence(text, options);
          calibrated = await calibrateRetrievalAbstention({ chunks, config, embedQuery, embedWordVecs, evidence, log, projectDir });
        } finally {
          await uncalibrated.close();
        }
      } finally {
        fssync.rmSync(uncalibratedPath, { force: true });
      }
      if (calibrated) {
        calibrationBytes = Buffer.from(JSON.stringify(calibrated.calibrationJson), 'utf8');
        // Retrieval-verified positives double as golden queries: embedded
        // in the evaluation segment, so any later reader can re-run the
        // pack's own tests from inside the file (mcp verify_pack).
        goldenQueries = calibrated.goldenQueries || [];
        // Kept for the evaluation segment (below) so a shipped pack's own
        // calibration quality is auditable after the fact — verify_pack can
        // report it instead of the numbers being build-log-only.
        calibrationSummary = calibrated.summary;
        const {
          verifiedPositiveQueries, foreignNegativeQueries, syntheticGibberishQueries,
          ablationNegativeQueries, entitySwapNegativeQueries, heldOutNegativeQueries, weakQueries, fitAuc, cvAuc, cvAucHard,
          humanCalibrationQueries, realQueryAuc, realQueryAbstentionRate, maxSimVsCoverage, maxSimInFit,
        } = calibrated.summary;
        const validation = humanCalibrationQueries
          ? `${humanCalibrationQueries} human queries: AUC ${realQueryAuc ?? 'n/a'}, abstention rate ${((realQueryAbstentionRate ?? 0) * 100).toFixed(0)}%`
          : 'no human queries (runtime.calibrationQueries) — validated by cross-validation only';
        log(`Calibrated relevance: ${verifiedPositiveQueries} on-topic / ${foreignNegativeQueries} off-domain + `
          + `${syntheticGibberishQueries} gibberish off-topic / ${weakQueries} weak queries, `
          + `5-fold CV AUC ${cvAuc ?? 'n/a'} vs off-topic (fit AUC ${fitAuc}, in-sample), validation: ${validation}`);
        log(`  in-domain-unanswerable diagnostic (not used in the fit): AUC ${cvAucHard ?? 'n/a'} vs `
          + `${ablationNegativeQueries} ablation + ${entitySwapNegativeQueries} entity-swap + ${heldOutNegativeQueries} held-out-doc `
          + 'hard negatives; matchQuality measures relevance, so "strong" does not by itself mean the passage holds the fact');
        // maxSim1 is comparison-only (see calibrate.mjs's GROUNDING_FEAT
        // design note: max(coverage1, maxSim1) has been tried twice —
        // unrescaled, then rescaled and damped — and reverted both times,
        // the second time for making real-query behavior worse despite a
        // better AUC number). Logged so it's visible without pulling
        // apart the artifact.
        if (maxSimVsCoverage?.maxSimSeparationAuc !== null && maxSimVsCoverage !== undefined) {
          log(`  maxSim1 vs coverage1 vs ablation hard negatives (comparison only, not used in the fit): separation AUC `
            + `${maxSimVsCoverage.maxSimSeparationAuc ?? 'n/a'} vs ${maxSimVsCoverage.coverageSeparationAuc ?? 'n/a'}; `
            + `mean positive/hard-negative ${maxSimVsCoverage.meanMaxSimPositive}/${maxSimVsCoverage.meanMaxSimHardNegative} `
            + `vs ${maxSimVsCoverage.meanCoveragePositive}/${maxSimVsCoverage.meanCoverageHardNegative}`);
          const para = maxSimVsCoverage.paraphrase;
          if (para?.coverage1) {
            log(`  maxSim1 vs coverage1 on real paraphrases (base vs. substituted positives, lower AUC = more `
              + `paraphrase-robust): maxSim1 ${para.maxSim1 ? para.maxSim1.separationAuc : 'n/a'} `
              + `(mean ${para.maxSim1 ? para.maxSim1.meanBase : 'n/a'} -> ${para.maxSim1 ? para.maxSim1.meanSubstituted : 'n/a'}) `
              + `vs coverage1 ${para.coverage1.separationAuc} (mean ${para.coverage1.meanBase} -> ${para.coverage1.meanSubstituted})`);
          }
        }
      }
    }
  } finally {
    embedder?.dispose();
  }
  calibrationBytes ??= unscored;
  let result = assemble(goldenQueries, calibrationBytes);
  // Replay the candidate goldens through the assembled artifact's own
  // retrieval (sketch scan + rerank + hybrid ordering, inline encoder) —
  // the configuration a reader will actually run, at k = 10, which the
  // calibration's own checks need not match. A golden that does not
  // reproduce there is dropped and the file reassembled (with the same
  // calibration): verify_pack and acceptance tests
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
      result = assemble(kept, calibrationBytes);
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
