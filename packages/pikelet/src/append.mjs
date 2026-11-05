// `pikelet append` — LAYERED_PROFILE.md 6.1.
//
// Compiles one layer on top of a mounted chain. The planning and assembly rules
// live in pikelet-wasm/complete/layer-append.mjs (and are conformance-tested
// there); this module is the operator-facing wiring: resolve the chain, load the
// base's own encoder, ingest and embed the new records, then write the file.
//
// The encoder comes from the BASE, not from this CLI's packaged weights. That is
// the point of 6.1's "a pack is sufficient to compile its own successors": a
// kind-3 base carries its teacher in its own query-interp segment, so appending
// needs no model download and cannot silently substitute a different encoder.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { CliError, loadCompleteModules, DEFAULT_CONFIG } from './common.mjs';
import { nodeFetch } from 'pikelet-wasm/complete/node-fetch.mjs';
import { ingestFolder, ingestUrl, chunkDocs, describeDroppedDocs, MIN_CHUNK_TOKENS, dedupeChunks, ingestDeclaration, INGEST_CHUNKER } from './ingest.mjs';

const defaultLog = (line) => console.log(line);

/**
 * @param {object} flags parsed CLI flags
 */
/**
 * Resolve each `--supersede <oldId>=<path>` target to the local index of the
 * chunk that path produced (LAYERED_PROFILE.md 3.5).
 *
 * Exported for testing: this lived inline in appendLayer, where the only way
 * to exercise it was to run the whole command -- so the test for supersede
 * resolution drove planLayer directly, never this, and missed that a chunk's
 * `sourcePath` is RELATIVE to the root it was ingested from ("pricing.md")
 * rather than absolute. Comparing against a cwd-resolved target therefore
 * matched nothing, and `--supersede 2=/abs/v2/pricing.md` failed on a path
 * that plainly existed.
 *
 * Accepts the target as written, as an absolute path, and resolved against
 * each --source root, so `2=pricing.md`, `2=v2/pricing.md` and
 * `2=/abs/v2/pricing.md` all name the same record. Basename alone is
 * deliberately NOT accepted: two sources can each hold a `pricing.md`, and
 * guessing between them is the silent wrong edge this resolution prevents.
 *
 * @param {Array<[number, string]>} pairs   parsed --supersede pairs
 * @param {Array<{sourcePath?: string}>} chunks  this layer's chunks, in order
 * @param {string[]} sources                the --source roots
 * @returns {Array<[number, number]>}  [oldId, local chunk index]
 */
export function resolveSupersedeTargets(pairs, chunks, sources) {
  return pairs.map(([oldId, target]) => {
    if (!sources.length) {
      throw new CliError(`--supersede ${oldId}=${target} needs a --source: the path names which new record replaces id ${oldId}`);
    }
    const candidates = new Set([target, path.resolve(process.cwd(), target)]);
    const matches = [];
    for (let i = 0; i < chunks.length; i++) {
      const sp = chunks[i].sourcePath;
      if (!sp) continue;
      if (candidates.has(sp)) { matches.push(i); continue; }
      const hit = sources.some((src) => {
        const root = /^https?:\/\//i.test(src) ? null : path.resolve(process.cwd(), src);
        return root !== null && candidates.has(path.resolve(root, sp));
      });
      if (hit) matches.push(i);
    }
    if (!matches.length) {
      throw new CliError(`--supersede ${oldId}=${target} matched no ingested record; `
        + `the path must be one the --source ingested (${chunks.length} chunk(s) produced)`);
    }
    if (matches.length > 1) {
      throw new CliError(`--supersede ${oldId}=${target} matched ${matches.length} records `
        + `(local ids ${matches.join(', ')}); a supersession edge names exactly one successor, `
        + 'so split the source or supersede by resolved id');
    }
    return [oldId, matches[0]];
  });
}

// `options` is for programmatic callers (the journal): `log` replaces the
// console logger — an MCP server's stdout is its protocol channel — and
// `chunks` supplies ready-made records instead of ingesting --source, one
// record per chunk, unchunked. Returns what was written.
export async function appendLayer(flags, options = {}) {
  const log = options.log ?? defaultLog;
  if (!flags.parent) throw new CliError('append requires --parent <file|url>[#identity]');
  if (!flags.out) throw new CliError('append requires --out <file>');
  const outPath = path.resolve(process.cwd(), flags.out);
  if (fssync.existsSync(outPath) && !flags.force) {
    throw new CliError(`Output file already exists: ${outPath}\nNext: rerun with --force or choose --out`);
  }

  const removeIds = toIdList(flags.remove, '--remove');
  const supersedePairs = toSupersedeList(flags.supersede);
  const sources = flags.source ? [].concat(flags.source) : [];
  const givenChunks = Array.isArray(options.chunks) ? options.chunks : null;
  if (givenChunks && sources.length) throw new CliError('append takes --source or ready-made chunks, not both');
  if (!sources.length && !givenChunks?.length && !removeIds.length && !supersedePairs.length) {
    // 6.1's "a layer must do something", caught before any work: the planner
    // enforces it too, but failing here costs no mount and no embedding.
    throw new CliError('append needs at least one of --source, --remove or --supersede: a layer that introduces neither a record nor a tombstone is refused (6.1)');
  }

  const { reader } = await loadCompleteModules();
  const {
    planAppendability, planLayer, buildLayerSegments, checkAppendDrift, canonicalJson,
  } = await import('pikelet-wasm/complete/layer-append.mjs');
  const { openPikeletChain } = await import('pikelet-wasm/complete/layer-reader.mjs');

  // --- 1. Mount the parent chain (6.1 step 1) ---------------------------------
  // The parent may itself be a chain; the caller lists its members base-first,
  // which is also what --lineage would supply. A bare --parent is a base.
  const { resolveChainMembers, parentLocatorFor } = await import('./common.mjs');
  const members = await resolveChainMembers([].concat(flags.parent), log);
  // Before the mount and any embedding, so a bad --parent-locator costs nothing.
  const parentLocator = await parentLocatorFor(flags, members[members.length - 1], outPath, log);
  const chain = await openPikeletChain(members, {
    fetch: nodeFetch,
    ...(flags['reader-drift-limit'] ? { readerDriftLimit: Number(flags['reader-drift-limit']) } : {}),
  });
  try {
    const info = chain.info();
    log(`Mounted chain: ${info.layers} member(s), ${info.records} records (${info.liveRecords} live), depth ${info.layers - 1}`);

    const head = chain.__head;
    if (!head) {
      throw new CliError('this reader build does not expose the chain head needed to append');
    }

    // --- 2. Appendability and the ingestion declaration (6.1) ----------------
    const givenAssertion = flags['assert-ingest']
      ? JSON.parse(await fs.readFile(path.resolve(process.cwd(), flags['assert-ingest']), 'utf8'))
      : null;
    // A chain whose depth-1 layer asserted the declaration already carries
    // it (validated at mount), and every later layer repeats it (4.1), so
    // only the first append onto such a base needs --assert-ingest. One
    // given again must be the same declaration: a later layer may not
    // change it.
    const chainAssertion = head.corpusIngest == null && head.chainIngest?.asserted
      ? JSON.parse(head.chainIngest.canonical)
      : null;
    if (chainAssertion && givenAssertion && canonicalJson(givenAssertion) !== head.chainIngest.canonical) {
      throw new CliError('--assert-ingest differs from the declaration this chain already asserted at depth 1; '
        + 'a later layer may not change it (6.1). Drop --assert-ingest to append under the chain\'s declaration.');
    }
    const assertIngest = givenAssertion ?? chainAssertion;
    if (head.corpusIngest == null && !assertIngest) {
      // A base compiled before compile recorded corpus.ingest. Say what to
      // assert rather than only that something must be: this CLI's own
      // declaration, which is right when the base came from compile with its
      // default chunking.
      throw new CliError('the base carries no corpus.ingest (it predates compile recording one), so record '
        + 'granularity cannot be checked (6.1). If it was compiled by pikelet with default chunking, assert '
        + 'that declaration:\n'
        + `  echo '${JSON.stringify(ingestDeclaration(DEFAULT_CONFIG.chunking))}' > ingest.json\n`
        + '  pikelet append ... --assert-ingest ingest.json\n'
        + 'The assertion is your statement, recorded in every layer of the chain.');
    }
    const plan0 = planAppendability({
      qiKind: head.qiKind,
      corpusIngest: head.corpusIngest,
      assertIngest,
    });
    // A declaration naming another chunker describes records this CLI cannot
    // reproduce; chunking new sources anyway would change record granularity
    // mid-chain, which is the thing the declaration exists to prevent.
    if (plan0.ingest.chunker !== undefined && plan0.ingest.chunker !== INGEST_CHUNKER) {
      throw new CliError(`the chain's records were made by chunker ${JSON.stringify(plan0.ingest.chunker)}, `
        + `but this pikelet chunks with ${JSON.stringify(INGEST_CHUNKER)}; appending would change record `
        + 'granularity mid-chain (6.1). Use the pikelet version that compiled the base, or compile a new base.');
    }
    log(`Base is appendable: encoder from the ${plan0.encoderSource === 'pack' ? 'pack itself' : 'host'}`
      + `${plan0.ingestAsserted ? ', ingestion declaration asserted by the operator' : ''}`);

    // --- 3. Ingest and chunk under the chain's declaration (6.1 step 2) ------
    let chunks = givenChunks ? [...givenChunks] : [];
    if (sources.length) {
      const docs = [];
      for (const src of sources) {
        const isUrl = /^https?:\/\//i.test(src);
        if (isUrl) {
          docs.push(...await ingestUrl({ type: 'url', url: src, maxPages: Number(flags['max-pages'] || 500) }, log));
        } else {
          docs.push(...await ingestFolder(path.resolve(process.cwd(), src), {
            include: flags.include || ['**/*.{md,mdx,html,txt}'],
            exclude: flags.exclude || ['**/node_modules/**', '**/.git/**'],
          }, log));
        }
      }
      // Records are NOT deduplicated against the chain (decision 9): a producer
      // that wants "append only what is new" compares candidate digests against
      // the chain's live recordSha256 values as its own policy.
      // Chunk under the CHAIN's declaration, not this CLI's defaults: record
      // granularity is part of what the calibration saw and must not change
      // mid-chain (6.1). plan0.ingest is the chain's declaration — the base's
      // own when it carries one, the operator's assertion otherwise.
      const chunking = {
        targetTokens: plan0.ingest.targetTokens ?? DEFAULT_CONFIG.chunking.targetTokens,
        overlapPercent: plan0.ingest.overlapPercent ?? DEFAULT_CONFIG.chunking.overlapPercent,
      };
      const chunked = chunkDocs(docs, chunking);
      chunks = dedupeChunks(chunked);
      log(`Ingested ${docs.length} doc(s) -> ${chunks.length} chunk(s)`);
      // A short page is dropped whole; name it, or it silently never reaches
      // the layer (and with nothing else to append, the refusal below would
      // not say why).
      const dropped = describeDroppedDocs(chunked.dropped);
      if (dropped) log(`Skipped ${chunked.dropped.length} doc(s) under ${MIN_CHUNK_TOKENS} tokens: ${dropped}`);
      if (!chunks.length && !removeIds.length && !supersedePairs.length) {
        throw new CliError('the sources produced no chunks, and no --remove or --supersede was given: nothing to append (6.1)'
          + (dropped ? `. Every document was under ${MIN_CHUNK_TOKENS} whitespace-separated tokens: ${dropped}` : ''));
      }
    }

    // Which new record replaces which old one (3.5). planLayer used to receive
    // the raw pairs and destructure the path away, assigning targets by
    // argument order; see resolveSupersedeTargets for what that cost.
    const resolvedSupersede = resolveSupersedeTargets(supersedePairs, chunks, sources);

    // --- 4-5. Plan ids, mask and supersessions (6.1 steps 4-5) --------------
    const plan = planLayer({
      parentRowBase: head.rowBase,
      parentRecords: head.records,
      parentBitset: head.bitset,
      parentDepth: head.depth,
      newRecordCount: chunks.length,
      remove: removeIds,
      supersede: resolvedSupersede,
      ...(flags['max-depth'] ? { maxDepth: Number(flags['max-depth']) } : {}),
    });

    // Drift: append MUST refuse past the effective limit unless --allow-drift.
    const drift = checkAppendDrift({
      baseRecords: head.baseRecords,
      appendedBefore: head.appendedBefore,
      newRecords: plan.records,
      headTombstones: plan.tombstones,
      effectiveLimit: info.driftLimit,
      allowDrift: flags['allow-drift'] === true,
    });

    log(`Planned layer: depth ${plan.depth}, rowBase ${plan.rowBase}, +${plan.records} record(s), `
      + `${plan.delta.length} new tombstone(s), ${plan.supersessions.length} supersession(s)`);
    log(`  calibrationDrift would be ${drift.drift.toFixed(4)} against limit ${info.driftLimit}`
      + `${drift.exceedsLimit ? ' — EXCEEDED, the chain will serve unscored' : ''}`);

    // --- 6. Embed with the BASE's encoder (6.1 step 3) ----------------------
    // Not this CLI's packaged weights: the base's own, resolved by the base's
    // open from its own query-interp segment. That is what makes "a pack is
    // sufficient to compile its own successors" true, and what makes
    // substituting a different encoder impossible.
    let vectors = [];
    if (chunks.length) {
      const enc = await head.passageEmbedder();
      if (!enc) {
        throw new CliError(`the base's query-interpretation kind ${head.qiKind} cannot embed passages; only a kind-3 base carries its teacher (6.1)`);
      }
      log(`Embedding ${chunks.length} chunk(s) with the base's own encoder`
        + `${enc.declaration.model ? ` (${enc.declaration.model})` : ''}`);
      for (const chunk of chunks) vectors.push(await enc.embed(chunk.text || ''));
      if (vectors[0].length !== head.dim) {
        throw new CliError(`the base's encoder produced ${vectors[0].length} dims but the chain declares ${head.dim}`);
      }
    }

    // --- 7. Build this layer's segments (6.1 step 4) ------------------------
    const { loadPikelet, loadArtifactContract } = await import('./common.mjs');
    const Pikelet = await loadPikelet();
    const artifactContract = await loadArtifactContract();
    const builder = await import('pikelet-wasm/complete/builder');

    let indexBytes = null;
    let corpusMeta = null;
    let corpusBytes = null;
    let lexicalBytes = null;
    if (chunks.length) {
      // The layer's own index over its own rows, with the base's geometry by
      // default (4.6 allows a layer its own; the default keeps a future
      // resident index over the union simple).
      const index = await Pikelet.create({
        dim: head.dim, metric: head.metric, quantized: true,
        maxElements: Math.max(chunks.length, Math.ceil(chunks.length * 1.25)),
      });
      try {
        // addBatch takes a plain array of vectors; ids are assigned by
        // position, which is what the layer's local numbering already is.
        index.addBatch(vectors);
        const snapshot = index.export();
        indexBytes = artifactContract.buildSketchArtifactBytes(snapshot, {
          // recommendedRerank applies to THIS layer's index only (4.1); a
          // small layer is cheapest to rerank exhaustively. Capped at the
          // reader's default ceiling, which a larger declaration would not
          // get anyway.
          recommendedRerank: Math.min(chunks.length, artifactContract.MAX_RECOMMENDED_RERANK ?? chunks.length),
        }).bytes;
      } finally {
        index.dispose();
      }
      const { publicChunk } = await import('./ingest.mjs');
      const built = builder.buildCorpusSegment(
        chunks.map((c) => Buffer.from(JSON.stringify(publicChunk(c)), 'utf8')),
      );
      corpusBytes = built.bytes;
      corpusMeta = built.corpus;
      // The base's tokenizer, via the same builder the base used.
      lexicalBytes = builder.buildLexicalSegment(chunks.map((c) => c.text || '')).bytes;
    }

    // The layer's own vocabulary bloom, with the BASE's geometry (4.5, 6.1
    // step 4). Without it the chain degrades to unscored: 4.5 says a layer
    // shipping `{"kind":"none"}` while the base carries a fit takes the whole
    // chain to drift-exceeded, and 5.3 says the union is what keeps the
    // known-token signal from under-reading every query about a layer's
    // content.
    let layerBloom = null;
    if (chunks.length && head.baseBloomGeometry) {
      const { buildLayerVocabBloom, BLOOM_HASH_NAMES } = await import('./calibrate.mjs');
      const geom = head.baseBloomGeometry;
      const built = buildLayerVocabBloom(chunks, { bits: geom.bits, minCount: geom.minCount ?? 1 });
      layerBloom = {
        geometry: { bits: geom.bits, hashes: geom.hashes ?? BLOOM_HASH_NAMES },
        bytes: built.bloom,
      };
      log(`Built the layer's vocabulary bloom: ${built.keptWords} term(s) at the base's geometry `
        + `(${geom.bits} bits)`);
    } else if (chunks.length) {
      log('The base carries no vocabulary bloom, so this layer ships none (the chain stays unscored)');
    }

    const { buildLayerSegments } = await import('pikelet-wasm/complete/layer-append.mjs');
    const assembled = buildLayerSegments({
      layerBloom,
      plan,
      baseIdentity: head.baseIdentity,
      parentIdentity: head.identity,
      baseQueryInterpSha256: head.baseQueryInterpSha256,
      parentLocator,
      ingest: plan0.ingest,
      ingestAsserted: plan0.ingestAsserted,
      dim: head.dim,
      metric: head.metric,
      encoder: head.encoder,
      corpus: corpusMeta || { records: 0 },
    });

    // --- 8. Write the file (6.1 step 6) ------------------------------------
    const segments = [];
    if (indexBytes) segments.push({ kind: 'index', bytes: indexBytes });
    if (corpusBytes) segments.push({ kind: 'corpus', bytes: corpusBytes });
    segments.push({ kind: 'query-interp', bytes: assembled.queryInterp });
    if (lexicalBytes) segments.push({ kind: 'lexical', bytes: lexicalBytes });
    segments.push({ kind: 'tombstones', bytes: assembled.tombstones });

    await fs.mkdir(path.dirname(outPath), { recursive: true });
    const written = builder.assemblePikeletFile(assembled.manifestFields, segments, outPath);
    log(`Wrote ${outPath}`);
    log(`  ${(written.fileBytes / 1024).toFixed(1)} KiB, depth ${plan.depth}, +${plan.records} record(s), `
      + `identity ${written.identity}`);
    log(`  calibrationDrift ${drift.drift.toFixed(4)}`
      + `${drift.exceedsLimit ? ' (drift-exceeded: the chain serves unscored)' : ''}`);
    // Mounting the newest layer walks its parent locators down to the base, and
    // the head identity pins every member. (The library's openPikeletChain
    // takes a query vector, not text, so it is no copy-paste next step.)
    log(`Serve the chain: pikelet mcp --pack ${outPath}#${written.identity}`);
    return {
      outPath, identity: written.identity, fileBytes: written.fileBytes,
      rowBase: plan.rowBase, records: plan.records, depth: plan.depth,
    };
  } finally {
    await chain.close();
  }
}

function toIdList(value, flagName) {
  if (value === undefined) return [];
  return [].concat(value).map((raw) => {
    const n = parseRecordId(raw);
    if (n === null) throw new CliError(`${flagName} takes one non-negative record id per flag, got ${raw}`);
    return n;
  });
}

// Strict: parseInt read `--remove 1,2,3` as 1 and `--remove 12abc` as 12,
// tombstoning a different set than the one named without a word.
function parseRecordId(raw) {
  const text = String(raw).trim();
  if (!/^\d+$/.test(text)) return null;
  const n = Number(text);
  return Number.isSafeInteger(n) ? n : null;
}

function toSupersedeList(value) {
  if (value === undefined) return [];
  return [].concat(value).map((raw) => {
    const at = String(raw).indexOf('=');
    if (at < 1) throw new CliError(`--supersede takes <oldId>=<sourcePath>, got ${raw}`);
    const oldId = parseRecordId(String(raw).slice(0, at));
    if (oldId === null) throw new CliError(`--supersede oldId must be a non-negative id, got ${raw}`);
    return [oldId, String(raw).slice(at + 1)];
  });
}
