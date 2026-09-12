// The build pipeline: buildAssets (ingest -> chunk -> embed -> index -> artifact),
// config validation, the manifest, student asset publishing, bundle sizing.

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { sha256, CLI_VERSION, DEFAULT_CONFIG, RANGE_ARTIFACT_MAGIC, MODEL_MAP, CliError, loadPikelet } from './common.mjs';
import { ingestFolder, ingestUrl, chunkDocs, dedupeChunks, applySourceRoutes, publicChunk } from './ingest.mjs';
import { buildCompleteArtifact } from './complete-build.mjs';
import { embedChunks, validateSelfRecall } from './embed.mjs';

async function buildSearchAssets(projectDir, config, options = {}) {
  await buildAssets(projectDir, config, options);
}
async function buildAssets(projectDir, config, options = {}) {
  validateConfig(config);
  const logLines = [];
  const log = (msg) => {
    logLines.push(msg);
    console.log(msg);
  };
  const sourceRoot = config.source.type === 'folder'
    ? path.resolve(options.sourceBaseDir || projectDir, config.source.path)
    : config.source.url;
  const docs = config.source.type === 'folder'
    ? await ingestFolder(sourceRoot, config.source, log)
    : await ingestUrl(config.source, log);
  if (docs.length === 0) {
    const detail = config.source.type === 'folder'
      ? `resolved path ${sourceRoot}; include ${JSON.stringify(config.source.include || ['**/*.{md,mdx,html,txt}'])}; exclude ${JSON.stringify(config.source.exclude || [])}`
      : `seed URL ${sourceRoot}; maxPages ${config.source.maxPages || 500}`;
    throw new CliError(`Ingest produced 0 documents from ${detail}. Next: check --source/include/exclude and ensure files contain enough text.`, 1);
  }
  const chunked = chunkDocs(docs, config.chunking);
  const chunks = dedupeChunks(chunked);
  if (chunks.length === 0) {
    const dropped = (chunked.dropped || [])
      .slice(0, 8)
      .map((doc) => `${doc.sourcePath || doc.url || doc.title || doc.id} (${doc.tokens} tokens)`)
      .join(', ');
    const suffix = dropped ? ` Dropped as too short: ${dropped}${chunked.dropped.length > 8 ? ', ...' : ''}.` : '';
    throw new CliError(`Chunking produced 0 chunks. Chunking counts whitespace-separated tokens and drops documents under 25 of them.${suffix} Next: use longer whitespace-delimited content or adjust source filters.`, 1);
  }
  applySourceRoutes(chunks, config);
  log(`Ingested ${docs.length} docs -> ${chunks.length} chunks`);

  const vectors = await embedChunks(chunks, config, log, projectDir);
  const Pikelet = await loadPikelet();
  const maxElements = Math.max(chunks.length, Math.ceil(chunks.length * 1.25));
  const index = await Pikelet.create({ ...config.index, dim: config.embedding.dims, maxElements });
  let snapshot;
  try {
    index.addBatch(vectors);
    validateSelfRecall(index, chunks, vectors, log);
    snapshot = index.export();
  } finally {
    index.dispose();
  }

  const assetsDir = options.assetsDir
    ? path.resolve(options.assetsDir)
    : path.join(projectDir, 'assets');
  await fs.mkdir(assetsDir, { recursive: true });
  const artifactPath = config.runtime?.mode === 'artifact' && config.runtime.artifactPath
    ? path.resolve(projectDir, config.runtime.artifactPath)
    : null;
  let artifact = null;
  let artifactInfo = null;
  if (config.runtime?.mode === 'artifact') {
    log('Warning: the artifact runtime builds the deprecated .pikelet-range profile. Prefer the snapshot runtime, or `pikelet compile` for a complete .pikelet file.');
    if (artifactPath && !fssync.existsSync(artifactPath)) {
      throw new CliError(`Configured Search Artifact not found: ${artifactPath}\nNext: update runtime.artifactPath in pikelet.config.json or rebuild with --artifact <file>.`, 2);
    }
    if (artifactPath) {
      artifact = await fs.readFile(artifactPath);
      artifactInfo = inspectRangeArtifactHeader(artifact, artifactPath);
      if (artifactInfo.dim !== config.embedding.dims) {
        throw new CliError(`Search Artifact dimension mismatch (${artifactInfo.dim} !== ${config.embedding.dims}). Next: supply an artifact built from the same embedding model.`, 2);
      }
      if (artifactInfo.count !== chunks.length) {
        throw new CliError(`Search Artifact count mismatch (${artifactInfo.count} !== ${chunks.length}). Next: rebuild the artifact from this generated corpus.`, 2);
      }
    }
  }
  const manifest = makeManifest(config, chunks, snapshot, vectors, artifact, artifactInfo);
  if (config.runtime?.mode === 'complete') {
    artifactInfo = await buildCompleteArtifact({
      Pikelet,
      projectDir,
      assetsDir,
      config,
      chunks,
      snapshot,
      vectors,
      log,
    });
    manifest.artifactSha256 = artifactInfo.sha256;
    manifest.artifact = artifactInfo;
    manifest.runtime = { ...manifest.runtime, mode: 'complete', artifactUrl: 'search.pikelet' };
  } else if (config.runtime?.mode === 'artifact') {
    const outPath = path.join(assetsDir, 'index.pikelet-range');
    if (artifact) {
      await fs.writeFile(outPath, artifact);
    } else {
      const buildManifest = Pikelet.buildRangeArtifact(snapshot, outPath, { layout: 'rcm' });
      artifactInfo = {
        version: buildManifest.formatVersion,
        kind: 1,
        dim: buildManifest.graph.dim,
        count: buildManifest.graph.count,
        entryPoint: buildManifest.graph.entryPoint,
        maxLevel: buildManifest.graph.maxLevel,
        M: buildManifest.graph.M,
        M0: buildManifest.graph.M0,
        metric: config.index.metric === 'l2' ? 0 : 1,
        recordBytes: buildManifest.addressing.recordBytes,
        parts: 0,
        routerCount: buildManifest.addressing.routerCount,
        baseCount: buildManifest.addressing.baseCount,
      };
      artifact = await fs.readFile(outPath);
    }
    manifest.artifactSha256 = crypto.createHash('sha256').update(artifact).digest('hex');
    manifest.artifact = artifactInfo;
  } else {
    await fs.writeFile(path.join(assetsDir, 'snapshot.pnck'), snapshot);
  }
  if (config.embedding.mode === 'student') {
    await publishStudentAssets(projectDir, config, assetsDir, manifest, log);
  }
  await fs.writeFile(path.join(assetsDir, 'corpus.json'), `${JSON.stringify(chunks.map(publicChunk), null, 2)}\n`);
  await fs.writeFile(path.join(assetsDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  await fs.mkdir(path.join(projectDir, '.pikelet'), { recursive: true });
  await fs.writeFile(path.join(projectDir, '.pikelet', 'last-build.log'), `${logLines.join('\n')}\n`);
  const gzipBytes = options.skipBundleSizeCheck ? null : await projectedGzipBytes(projectDir);
  if (config.runtime?.mode === 'complete') {
    log(`Built complete .pikelet artifact with ${(artifactInfo.bytes / 1024 / 1024).toFixed(2)} MB`);
  } else if (config.runtime?.mode === 'artifact') {
    log(`Built corpus assets with ${(artifact.byteLength / 1024 / 1024).toFixed(2)} MB Search Artifact`);
  } else {
    log(`Built index: ${(snapshot.byteLength / 1024 / 1024).toFixed(2)} MB snapshot`);
  }
  if (gzipBytes !== null) log(`Projected bundled gzip size: ${(gzipBytes / 1024 / 1024).toFixed(2)} MB`);
  if (gzipBytes !== null && gzipBytes > 3 * 1024 * 1024) {
    throw new CliError('Projected Worker bundle exceeds the free-plan 3 MB compressed limit. Next: reduce source scope or wait for the R2-backed tier.', 2);
  }
}
function validateConfig(config) {
  if (!config || config.version !== 1) throw new CliError('pikelet.config.json version must be 1', 1);
  if (!config.name) throw new CliError('pikelet.config.json name is required', 1);
  if (!config.source || !['folder', 'url'].includes(config.source.type)) throw new CliError('source.type must be folder or url', 1);
  const embeddingMode = config.embedding?.mode || 'workers-ai';
  if (!['workers-ai', 'student', 'inline-transformer'].includes(embeddingMode)) throw new CliError('embedding.mode must be workers-ai, student, or inline-transformer', 1);
  if (embeddingMode === 'workers-ai') {
    const model = MODEL_MAP[config.embedding?.buildModel];
    if (!model) throw new CliError(`Unsupported embedding.buildModel ${config.embedding?.buildModel}`, 1);
    if (config.embedding.dims !== model.dims) throw new CliError(`embedding.dims must be ${model.dims}`, 1);
  } else if (embeddingMode === 'student' && !config.embedding.studentModelPath) {
    throw new CliError('embedding.studentModelPath is required when embedding.mode is student', 1);
  } else if (embeddingMode === 'student' && !config.embedding.trainStudent?.enabled && !config.embedding.teacherVectorsPath) {
    throw new CliError('embedding.teacherVectorsPath is required when embedding.mode is student and trainStudent is disabled', 1);
  }
  const runtimeMode = config.runtime?.mode || 'snapshot';
  if (!['snapshot', 'artifact', 'complete'].includes(runtimeMode)) throw new CliError('runtime.mode must be snapshot, artifact, or complete', 1);
  if (embeddingMode === 'inline-transformer' && runtimeMode !== 'complete') {
    throw new CliError('embedding.mode inline-transformer requires runtime.mode complete: snapshot/artifact runtimes deploy the Workers AI query encoder, which cannot serve inline-transformer builds', 1);
  }
  if (runtimeMode === 'artifact' && config.runtime?.storage !== 'bundled') throw new CliError('runtime.storage must be bundled for artifact mode in this release', 1);
  if (runtimeMode === 'complete' && config.runtime?.profile !== 'kind3') throw new CliError('runtime.profile must be kind3 for complete mode', 1);
}
function inspectRangeArtifactHeader(bytes, label = 'Search Artifact') {
  if (!bytes || bytes.byteLength < 56) {
    throw new CliError(`${label} is too small to be a Pikelet Search Artifact`, 2);
  }
  const magic = bytes.readUInt32LE(0);
  if (magic !== RANGE_ARTIFACT_MAGIC) {
    throw new CliError(`${label} is not a Pikelet Search Artifact`, 2);
  }
  const version = bytes.readUInt32LE(4);
  const kind = bytes.readUInt32LE(8);
  const dim = bytes.readUInt32LE(12);
  const count = bytes.readUInt32LE(16);
  const entryPoint = bytes.readUInt32LE(20);
  const maxLevel = bytes.readUInt32LE(24);
  const M = bytes.readUInt32LE(28);
  const M0 = bytes.readUInt32LE(32);
  const metric = bytes.readUInt32LE(36);
  const recordBytes = bytes.readUInt32LE(40);
  const parts = bytes.readUInt32LE(52);
  const routerCount = version >= 2 && bytes.byteLength >= 64 ? bytes.readUInt32LE(56) : 0;
  const baseCount = version >= 2 && bytes.byteLength >= 68 ? bytes.readUInt32LE(60) : count;
  return { version, kind, dim, count, entryPoint, maxLevel, M, M0, metric, recordBytes, parts, routerCount, baseCount };
}
async function publishStudentAssets(projectDir, config, assetsDir, manifest, log) {
  const modelPath = path.resolve(projectDir, config.embedding.studentModelPath);
  if (!fssync.existsSync(modelPath)) {
    throw new CliError(`Student model was not produced: ${modelPath}. Next: rerun training or pass --student-model.`, 2);
  }
  const modelBytes = await fs.readFile(modelPath);
  await fs.writeFile(path.join(assetsDir, 'student-model.bin'), modelBytes);
  // The abstention scorer travels beside the model both in the trainer's out
  // dir and when staged externally. A `null` placeholder keeps the Worker's
  // static asset import valid when no calibrated scorer exists.
  const abstentionPath = path.join(path.dirname(modelPath), 'student-abstention.json');
  let abstention = null;
  if (fssync.existsSync(abstentionPath)) {
    const abstentionBytes = await fs.readFile(abstentionPath);
    await fs.writeFile(path.join(assetsDir, 'student-abstention.json'), abstentionBytes);
    abstention = {
      bytes: abstentionBytes.byteLength,
      sha256: crypto.createHash('sha256').update(abstentionBytes).digest('hex'),
    };
  } else {
    await fs.writeFile(path.join(assetsDir, 'student-abstention.json'), 'null\n');
  }
  manifest.encoder = {
    ...manifest.encoder,
    studentModelBytes: modelBytes.byteLength,
    studentModelSha256: crypto.createHash('sha256').update(modelBytes).digest('hex'),
    abstentionBytes: abstention?.bytes || null,
    abstentionSha256: abstention?.sha256 || null,
  };
  log(`Bundled student encoder (${(modelBytes.byteLength / 1024).toFixed(1)} KiB${abstention ? ', calibrated abstention' : ', no abstention scorer'})`);
}
function makeManifest(config, chunks, snapshot, vectors, artifact = null, artifactInfo = null) {
  const model = config.embedding.mode === 'student' || config.embedding.mode === 'inline-transformer'
    ? null
    : MODEL_MAP[config.embedding.buildModel];
  const firstVectorHash = vectors[0]
    ? crypto.createHash('sha256').update(Buffer.from(vectors[0].buffer, vectors[0].byteOffset, vectors[0].byteLength)).digest('hex')
    : null;
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    cliVersion: CLI_VERSION,
    name: config.name,
    model: config.embedding.mode === 'student'
      ? 'pikelet-distilled-student'
      : config.embedding.mode === 'inline-transformer'
        ? config.runtime?.inlineEncoder?.model || 'sentence-transformers/all-MiniLM-L6-v2'
        : config.embedding.buildModel,
    workersAiModel: model?.workersAiModel || null,
    encoder: config.embedding.mode === 'student'
      ? {
          mode: 'student',
          format: 'pstu',
          studentModelPath: config.embedding.studentModelPath,
          teacherVectorsPath: config.embedding.teacherVectorsPath || null,
          trainedDuringBuild: config.embedding.trainStudent?.enabled === true,
        }
      : config.embedding.mode === 'inline-transformer'
        ? {
            mode: 'inline-transformer',
            kind: 'inline-transformer-v1',
            teacherVectorsPath: config.embedding.teacherVectorsPath,
            runtimeWeightsPath: config.runtime?.inlineEncoder?.weightsPath || null,
            runtimeVocabPath: config.runtime?.inlineEncoder?.vocabPath || null,
          }
      : {
          mode: 'workers-ai',
          hfModel: model.hfModel,
          workersAiModel: model.workersAiModel,
        },
    dims: config.embedding.dims,
    dim: config.embedding.dims,
    metric: config.index.metric,
    quantized: config.index.quantized,
    M: config.index.M,
    efConstruction: config.index.efConstruction,
    efSearch: config.index.efSearch,
    maxElements: Math.max(chunks.length, Math.ceil(chunks.length * 1.25)),
    chunkCount: chunks.length,
    prefixPolicy: config.embedding.prefixPolicy || DEFAULT_CONFIG.embedding.prefixPolicy,
    pooling: config.embedding.pooling || 'mean',
    normalize: config.embedding.normalize !== false,
    maxInputTokens: model?.maxInputTokens || null,
    maxQueryChars: 4096,
    runtime: config.runtime || DEFAULT_CONFIG.runtime,
    firstVectorSha256: firstVectorHash,
    snapshotSha256: crypto.createHash('sha256').update(snapshot).digest('hex'),
    artifactSha256: artifact ? crypto.createHash('sha256').update(artifact).digest('hex') : null,
    artifact: artifactInfo,
    configHash: crypto.createHash('sha256').update(JSON.stringify(config)).digest('hex'),
  };
}
async function projectedGzipBytes(projectDir) {
  const files = [
    'worker.js',
    'encoder.js',
    'ui.html',
    'assets/corpus.json',
    'assets/manifest.json',
  ];
  const completePath = path.join(projectDir, 'assets', 'search.pikelet');
  const artifactPath = path.join(projectDir, 'assets', 'index.pikelet-range');
  files.push(fssync.existsSync(completePath)
    ? 'assets/search.pikelet'
    : fssync.existsSync(artifactPath) ? 'assets/index.pikelet-range' : 'assets/snapshot.pnck');
  for (const studentFile of ['student-embedder.mjs', 'assets/student-model.bin', 'assets/student-abstention.json']) {
    if (fssync.existsSync(path.join(projectDir, studentFile))) files.push(studentFile);
  }
  const buffers = [];
  for (const file of files) buffers.push(await fs.readFile(path.join(projectDir, file)));
  return zlib.gzipSync(Buffer.concat(buffers)).byteLength;
}

export {
  buildSearchAssets,
  buildAssets,
  validateConfig,
  inspectRangeArtifactHeader,
  publishStudentAssets,
  makeManifest,
  projectedGzipBytes,
};
