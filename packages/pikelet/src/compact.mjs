// `pikelet compact` — LAYERED_PROFILE.md 6.3.
//
// Collapses a chain back into one pikelet-complete-v2 base. This is SEMANTIC
// maintenance, not only physical consolidation: it restores exact lexical
// statistics over the live set, a live-only vocabulary bloom, no tombstones,
// depth zero and reclaimed id space. The drift and depth limits exist to force it to happen.
//
// What it preserves is corpus meaning and provenance — the compacted base's
// records are the old head's live records, in ascending old-id order, bytes
// unchanged. It does NOT preserve rankings, and is not meant to: the layered
// path scored BM25 with statistics that counted tombstoned documents and
// generated candidates per tier, and compaction scores one unified corpus with
// exact statistics. A compaction that reproduced the chain's rankings would have
// failed at its job.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { CliError } from './common.mjs';
import { nodeFetch } from 'pikelet-wasm/complete/node-fetch.mjs';

const defaultLog = (line) => console.log(line);

// `options.log` replaces the console logger for programmatic callers (the
// journal, inside an MCP server whose stdout is the protocol channel).
// Returns what was written.
export async function compactChain(flags, options = {}) {
  const log = options.log ?? defaultLog;
  if (!flags.head) throw new CliError('compact requires --head <file|url>[#identity], listing the chain base-first');
  if (!flags.out) throw new CliError('compact requires --out <file>');
  const outPath = path.resolve(process.cwd(), flags.out);
  if (fssync.existsSync(outPath) && !flags.force) {
    throw new CliError(`Output file already exists: ${outPath}\nNext: rerun with --force or choose --out`);
  }

  const { openPikeletChain } = await import('pikelet-wasm/complete/layer-reader.mjs');
  const { planCompaction, buildLineageSegment } = await import('pikelet-wasm/complete/layer-compact.mjs');
  const builder = await import('pikelet-wasm/complete/builder');
  const { loadArtifactContract } = await import('./common.mjs');

  const { resolveChainMembers, refuseOutputOverInput } = await import('./common.mjs');
  const members = await resolveChainMembers([].concat(flags.head), log);
  refuseOutputOverInput(outPath, members);
  const chain = await openPikeletChain(members, { fetch: nodeFetch });
  try {
    const info = chain.info();
    const chainMembers = chain.__members;
    const head = chain.__head;
    log(`Mounted chain: ${info.layers} member(s), ${info.records} records (${info.liveRecords} live), `
      + `${info.tombstones} tombstone(s), depth ${info.layers - 1}`);
    if (info.layers === 1) {
      throw new CliError('this is already a depth-0 base: there is no chain to compact');
    }

    // --- 1-2. Enumerate live ids and renumber densely (6.3) -----------------
    const headBitset = chainMembers[chainMembers.length - 1].bitset
      ?? chainMembers.findLast((m) => m.bitset)?.bitset
      ?? new Uint8Array(0);
    const plan = planCompaction({ rowTotal: info.records, headBitset });
    log(`Compacting to ${plan.liveRecords} live record(s), reclaiming ${plan.tombstones} id(s)`);

    // Records are copied verbatim — the compacted base's records ARE the old
    // head's live records, bytes unchanged. Re-embedding would break that.
    const records = [];
    const texts = [];
    // The stored bytes, not record(): record() lays chain fields over the
    // parsed record, including a global `id` that replaces the record's own,
    // so stripping them and re-serializing dropped every record's own id and
    // could reorder keys or reformat numbers.
    const liveBytes = await chain.__recordBytes(plan.liveIds);
    for (const bytes of liveBytes) {
      records.push(Buffer.from(bytes));
      const own = JSON.parse(Buffer.from(bytes).toString('utf8'));
      texts.push(own.text || '');
    }

    // --- 2. Rebuild the index over live rows --------------------------------
    // 6.3: copy each live row's quantized bytes and its per-row scale/offset
    // verbatim. Same encoder, same quantizer, no re-embedding — the compacted
    // base's vectors ARE the head's live vectors, exactly as its records are.
    //
    // This re-embedded every live record with the base's own encoder instead,
    // on the stated grounds that the engine exposed no row readback. It does:
    // the sketch reader's fetchRows() (format 2 verifies each row's digest as
    // it reads) with the resident scales/offsets, reached through the chain's
    // __fetchRows seam. Re-embedding cost a forward pass per record and only
    // reproduced the same bytes if the encoder was bit-identical, which is a
    // stronger assumption than a copy needs to make.
    const artifactContract = await loadArtifactContract();
    log(`Copying ${plan.liveIds.length} row(s) verbatim from the head's tiers`);
    const fetched = await chain.__fetchRows(plan.liveIds);
    const dim = head.dim;
    const qdata = new Uint8Array(plan.liveRecords * dim);
    const scales = new Float32Array(plan.liveRecords);
    const offsets = new Float32Array(plan.liveRecords);
    for (let newId = 0; newId < fetched.length; newId++) {
      const { row, scale, offset } = fetched[newId];
      if (row.length !== dim) {
        throw new CliError(`row for old id ${plan.liveIds[newId]} is ${row.length} bytes, expected ${dim}`);
      }
      qdata.set(row, newId * dim);
      scales[newId] = scale;
      offsets[newId] = offset;
    }
    const indexBytes = artifactContract.exportSketchArtifact(
      { dim, count: plan.liveRecords, metric: head.metricCode, qdata, scales, offsets },
      null,
      { recommendedRerank: Math.min(plan.liveRecords, 100) },
    ).bytes;

    // --- 3. Corpus and lexical over live records; statistics now exact ------
    const corpus = builder.buildCorpusSegment(records);
    const lexical = builder.buildLexicalSegment(texts);
    log(`Rebuilt lexical index: ${lexical.meta.terms.toLocaleString()} terms over `
      + `${lexical.meta.docCount.toLocaleString()} live record(s) — statistics are now exact`);

    // --- 5. The lineage segment (kind 7), REQUIRED ---------------------------
    const heads = chainMembers.map((m, i) => ({
      identity: m.identity,
      rowTotal: chainMembers.slice(0, i + 1).reduce((n, x) => n + x.records, 0),
      depth: i,
    }));
    const edges = [];
    for (const m of chainMembers) {
      for (const [oldId, newId] of m.supersessions || []) edges.push([oldId, newId, m.depth]);
    }
    const lineage = buildLineageSegment({
      heads, headBitset, oldRowTotal: info.records, supersessions: edges,
    });
    log(`Lineage segment: ${heads.length} head(s) in the table, ${edges.length} edge(s) — `
      + 'a citation against any of them still translates (6.3)');

    // The ENCODER half of the query-interp segment is copied from the base:
    // compaction MUST NOT change encoder, dim, metric, ingestion or
    // tokenization — those are a new compile, not a compaction. The
    // CALIBRATION half is replaced, because 6.3 forbids carrying the
    // inherited fit across a compaction: the fit was made against a corpus
    // that no longer exists.
    //
    // This used to copy the whole segment, so the base's fit shipped verbatim
    // while the log said "shipping unscored". A compacted pack scored `strong`
    // with a confidence, from a fit whose corpus had changed underneath it —
    // exactly what step 4's MUST exists to prevent.
    const baseQi = await head.readBaseQueryInterp?.();
    if (!baseQi) {
      throw new CliError('compaction needs the base\'s query-interp segment to copy; this reader build does not expose it');
    }
    const qiView = new DataView(baseQi.buffer, baseQi.byteOffset, baseQi.byteLength);
    const qiVersion = qiView.getUint32(0, true);
    const qiKindRead = qiView.getUint32(4, true);
    const qiEncoderLen = qiView.getUint32(8, true);
    const qiCalibrationLen = qiView.getUint32(12, true);
    if (16 + qiEncoderLen + qiCalibrationLen !== baseQi.length) {
      throw new CliError(`the base's query-interp segment declares ${qiEncoderLen}+${qiCalibrationLen} bytes `
        + `in a ${baseQi.length}-byte segment; refusing to rebuild it`);
    }
    const encoderBytes = baseQi.subarray(16, 16 + qiEncoderLen);
    if (qiVersion !== 1) {
      throw new CliError(`the base's query-interp segment is version ${qiVersion}; this build writes version 1`);
    }

    const assemble = (calibration, target) => builder.assemblePikeletFile({
      profile: builder.PROFILE_V2,
      dim: head.dim,
      metric: head.metric,
      encoder: head.encoder,
      corpus: {
        ...corpus.corpus,
        // The compacted base carries the chain's ingestion declaration so it is
        // appendable on its own terms (6.1).
        ...(head.corpusIngest ? { ingest: head.corpusIngest } : {}),
      },
      index: {},
      compactedFrom: {
        identity: info.identity,
        depth: info.layers - 1,
        liveRecords: plan.liveRecords,
        tombstones: plan.tombstones,
      },
    }, [
      { kind: 'index', bytes: indexBytes },
      { kind: 'corpus', bytes: corpus.bytes },
      { kind: 'query-interp', bytes: builder.buildQueryInterpSegment(qiKindRead, encoderBytes, calibration) },
      { kind: 'lexical', bytes: lexical.bytes },
      { kind: 'lineage', bytes: lineage },
    ], target);

    // --- 4. Calibration ----------------------------------------------------
    // 6.3 step 4 forbids carrying the inherited fit across a compaction, and
    // pikelet no longer fits one, so the compacted base ships unscored.
    // --no-refit is accepted and changes nothing.
    const calibration = Buffer.from(JSON.stringify({
      kind: 'retrieval-signals-v1', asset: null, vocabBloomBase64: '',
    }), 'utf8');

    const written = assemble(calibration, outPath);

    log(`Wrote ${outPath}`);
    log(`  ${(written.fileBytes / 1024 / 1024).toFixed(2)} MiB, ${plan.liveRecords} records, depth 0, `
      + `identity ${written.identity}`);
    log(`  compacted from ${info.identity.slice(0, 12)}… at depth ${info.layers - 1}`);
    return { outPath, identity: written.identity, records: plan.liveRecords };
  } finally {
    await chain.close();
  }
}

export async function rebaseLayer(flags) {
  const log = defaultLog;
  // 6.2: recompile a layer onto a different head. The layer's BYTES are copied
  // verbatim — u8 rows, sketch, records, postings, vocabulary bloom — and
  // nothing is re-embedded; local ids do not change. What is recomputed is the
  // layer's position in a new history: rowBase, the cumulative bitset, the
  // supersession edges (translated and conflict-resolved) and the manifest.
  //
  // The rules live in pikelet-wasm/complete/layer-rebase.mjs and are
  // conformance-tested there. This is the command around them: it mounts three
  // artifacts, because computing what the layer DID (rather than what it
  // inherited) needs the layer's ORIGINAL parent as well as the new head.
  if (!flags.layer) throw new CliError('rebase requires --layer <file>: the layer to move');
  if (!flags.onto) throw new CliError('rebase requires --onto <file|...>: the new head, base-first for a chain');
  if (!flags.out) throw new CliError('rebase requires --out <file>');
  const outPath = path.resolve(process.cwd(), flags.out);
  if (fssync.existsSync(outPath) && !flags.force) {
    throw new CliError(`Output file already exists: ${outPath}\nNext: rerun with --force or choose --out`);
  }
  const onConflict = flags['on-conflict'] || 'refuse';
  const onForeign = flags['on-foreign'] || 'refuse';

  const { loadCompleteModules, resolveChainMembers, parentLocatorFor, refuseOutputOverInput } = await import('./common.mjs');
  const { builder } = await loadCompleteModules();
  const { openPikeletChain } = await import('pikelet-wasm/complete/layer-reader.mjs');
  const { readMemberManifest } = await import('pikelet-wasm/complete');
  const { planRebase, forkPoint, checkRebasePreconditions, ON_CONFLICT, ON_FOREIGN } =
    await import('pikelet-wasm/complete/layer-rebase.mjs');
  const { buildLayerSegments } = await import('pikelet-wasm/complete/layer-append.mjs');
  const { chainIngestDeclaration } = await import('pikelet-wasm/complete/layer-manifest.mjs');

  for (const [flag, value, allowed] of [['--on-conflict', onConflict, Object.values(ON_CONFLICT)],
    ['--on-foreign', onForeign, Object.values(ON_FOREIGN)]]) {
    if (!allowed.includes(value)) throw new CliError(`${flag} must be one of ${allowed.join(', ')}`);
  }

  // --- 1. The layer, read as a manifest only: it is not mountable alone ------
  const layerPath = (await resolveChainMembers([flags.layer], log))[0];
  refuseOutputOverInput(outPath, [layerPath]);
  const layerPeek = await readMemberManifest(layerPath);
  if (!layerPeek.isLayer) throw new CliError(`${flags.layer} is not a layer (profile ${layerPeek.manifest.profile})`);
  const L = layerPeek.manifest.layer;
  log(`Layer ${layerPeek.identity.slice(0, 12)}…: depth ${L.depth}, rowBase ${L.rowBase}, `
    + `${L.records} record(s), ${L.tombstones} tombstone(s), ${L.supersessions ?? 0} edge(s)`);

  // --- 2. The layer's ORIGINAL parent chain (6.2) ----------------------------
  // "To compute ΔT_L the rebase MUST open L's original parent by
  // L.layer.parent.identity; if that artifact is unavailable, the delta is
  // unknowable and the rebase MUST refuse." T_L is cumulative; replaying it
  // would import ancestors' operations into a branch that never performed them.
  const oldParentSpec = flags['old-parent']
    ? [].concat(flags['old-parent'])
    : (L.parent?.locator ? [path.resolve(path.dirname(layerPath), L.parent.locator)] : null);
  if (!oldParentSpec) {
    throw new CliError('rebase needs the layer\'s original parent to compute what the layer itself deleted '
      + `(6.2). The layer names parent ${String(L.parent?.identity).slice(0, 12)}… but carries no locator; `
      + 'pass --old-parent <file> [--old-parent <layer> ...], base-first.');
  }
  const oldChainMembers = await resolveChainMembers(oldParentSpec, log);
  refuseOutputOverInput(outPath, oldChainMembers);
  let oldParent;
  try {
    oldParent = await openPikeletChain(oldChainMembers, { followParents: false, fetch: nodeFetch });
  } catch (err) {
    // 6.2: "if that artifact is unavailable, the delta is unknowable and the
    // rebase MUST refuse." Say that, rather than leaking an ENOENT: the
    // operator needs to know WHY the file matters, since the obvious next move
    // (rebase the cumulative bitset instead) would be wrong.
    throw new CliError(`cannot open the layer's original parent `
      + `${String(L.parent?.identity).slice(0, 12)}… (${err.message}). A rebase replays what the layer `
      + 'itself deleted, which is its cumulative mask MINUS its parent\'s; without the parent that '
      + 'difference is unknowable, and replaying the cumulative mask would import its ancestors\' '
      + 'deletions into a history that never performed them (6.2). Pass --old-parent <file> '
      + '[--old-parent <layer> ...], base-first.');
  }
  let head = null;
  try {
    const oldInfo = oldParent.info();
    if (oldInfo.identity !== L.parent.identity) {
      throw new CliError(`the original parent resolved to ${oldInfo.identity.slice(0, 12)}… but the layer `
        + `commits to ${String(L.parent.identity).slice(0, 12)}…: that is not the parent this layer was cut against (6.2)`);
    }
    const oldHead = oldParent.__head;
    log(`Original parent: ${oldInfo.layers} member(s), ${oldInfo.records} rows, depth ${oldInfo.layers - 1}`);

    // --- 3. The new head ----------------------------------------------------
    const ontoMembers = await resolveChainMembers([].concat(flags.onto), log);
    refuseOutputOverInput(outPath, ontoMembers);
    const parentLocator = await parentLocatorFor(flags, ontoMembers[ontoMembers.length - 1], outPath, log);
    head = await openPikeletChain(ontoMembers);
    const headInfo = head.info();
    const H = head.__head;
    log(`New head: ${headInfo.layers} member(s), ${headInfo.records} rows, depth ${headInfo.layers - 1}`);

    // --- 4. Preconditions (6.2) --------------------------------------------
    // Each history's chain declaration in the {canonical, asserted} form the
    // check compares. The head's is the one validated at its mount, or its
    // base's corpus.ingest when it is a lone base (null for a legacy base).
    // The layer's history is its original parent PLUS the layer, so when that
    // parent is a lone legacy base the layer itself is the depth-1 member that
    // asserts the declaration.
    const baseDecl = (ingest) => (ingest == null ? null
      : chainIngestDeclaration({ corpus: { ingest } }, null));
    const headChainDecl = H.chainIngest ?? baseDecl(H.corpusIngest);
    let layerChainDecl;
    try {
      layerChainDecl = oldHead.chainIngest
        ?? chainIngestDeclaration({ corpus: { ingest: oldHead.corpusIngest } }, layerPeek.manifest);
    } catch (err) {
      throw new CliError(`rebase refuses: ${err.message}`);
    }
    const decl = checkRebasePreconditions({
      layerManifest: layerPeek.manifest,
      headChainDecl,
      layerChainDecl,
      headIsLegacyBase: headInfo.layers === 1 && H.corpusIngest == null,
      headBaseIdentity: headInfo.baseIdentity,
      layerBaseIdentity: L.baseIdentity,
    });

    // The fork point is the longest common prefix BY IDENTITY, base first.
    const fork = forkPoint(oldInfo.members, headInfo.members);
    log(`Fork point: ${fork.sharedPrefix} shared member(s), forkRowBase ${fork.forkRowBase}`);

    // --- 5. Plan -----------------------------------------------------------
    const plan = planRebase({
      layerBitset: await layerCumulativeBitset(layerPath, L),
      oldParentBitset: oldHead.bitset,
      oldRowBase: L.rowBase,
      layerRecords: L.records,
      layerSupersessions: await layerOwnEdges(layerPath, L),
      newRowBase: H.rowBase + H.records,
      headBitset: H.bitset,
      headSupersessions: headEdgeMap(headInfo, head),
      forkRowBase: fork.forkRowBase,
      onForeign, onConflict,
    });
    if (plan.droppedForeign?.length) {
      log(`Dropped ${plan.droppedForeign.length} foreign id(s) at --on-foreign drop: ${plan.droppedForeign.join(', ')}`);
    }
    if (plan.skippedEdges?.length) {
      // skippedEdges holds the conflicting OLD ids (planRebase pushes x, not a
      // pair), and each y stays as an ordinary appended record (6.2).
      log(`Skipped ${plan.skippedEdges.length} conflicting edge(s) at --on-conflict skip: `
        + `id(s) ${plan.skippedEdges.join(', ')} keep the head's successor; this layer's replacement `
        + 'stays as an ordinary record');
    }
    // 6.2's terminal rule.
    if (!plan.emit) {
      log('Nothing to emit: the rebased operation is empty because the target history already '
        + 'represents it (every deletion this layer introduced is already performed on the new head, '
        + 'and the layer adds no records). No file written (6.2).');
      return;
    }
    log(`Planned rebase: rowBase ${L.rowBase} -> ${plan.newRowBase}, ${plan.records} record(s) copied verbatim, `
      + `${plan.tombstones} tombstone(s) (${plan.newlySet.length} newly set), ${plan.supersessions.length} edge(s)`);

    // --- 6. Copy the layer's own segments VERBATIM (6.2) --------------------
    const carried = await copyLayerSegments(layerPath);
    const segments = buildLayerSegments({
      plan: { ...plan, depth: headInfo.layers, rowBase: plan.newRowBase, tombstoneOnly: plan.records === 0 },
      baseIdentity: headInfo.baseIdentity,
      parentIdentity: headInfo.identity,
      baseQueryInterpSha256: H.baseQueryInterpSha256,
      parentLocator,
      // What the precondition resolved against the NEW history, not the
      // layer's own flags: a layer rebased from depth 2 onto a legacy base
      // becomes the depth-1 member and must carry the assertion.
      ingest: decl.ingest,
      ingestAsserted: decl.ingestAsserted,
      dim: layerPeek.manifest.dim,
      metric: layerPeek.manifest.metric,
      encoder: layerPeek.manifest.encoder,
      corpus: layerPeek.manifest.corpus,
      layerBloom: null,   // the bloom rides in the copied query-interp segment
    });

    const written = builder.assemblePikeletFile(segments.manifestFields, [
      ...(carried.index ? [{ kind: 'index', bytes: carried.index }] : []),
      ...(carried.corpus ? [{ kind: 'corpus', bytes: carried.corpus }] : []),
      { kind: 'query-interp', bytes: carried.queryInterp },
      ...(carried.lexical ? [{ kind: 'lexical', bytes: carried.lexical }] : []),
      { kind: 'tombstones', bytes: segments.tombstones },
    ], outPath);

    log(`Wrote ${outPath}`);
    log(`  ${(written.fileBytes / 1024).toFixed(1)} KiB, depth ${headInfo.layers}, ${plan.records} record(s), `
      + `identity ${written.identity}`);
    log('  Chain-level golden queries dropped (goldenQueriesDropped: "rebase"): their expected global '
      + 'ids moved, and 6.2 forbids copying them.');
    log(`  The old layer ${layerPeek.identity.slice(0, 12)}… is untouched; citations against it still resolve `
      + 'against the old files.');
  } finally {
    await oldParent.close();
    if (head) await head.close();
  }
}

// The layer's cumulative bitset T_L, read from its own tombstone segment.
async function layerCumulativeBitset(layerPath, L) {
  const { readChainMemberShellBytes } = await import('./common.mjs');
  const { parseTombstoneSegment } = await import('pikelet-wasm/complete/tombstones.mjs');
  const bytes = await readChainMemberShellBytes(layerPath, 'tombstones');
  const parsed = parseTombstoneSegment(bytes, { rowBase: L.rowBase, tombstones: L.tombstones, records: L.records });
  return parsed.bitset;
}

// S_L: the supersession edges physically recorded in L's OWN segment (3.5).
async function layerOwnEdges(layerPath, L) {
  const { readChainMemberShellBytes } = await import('./common.mjs');
  const { parseTombstoneSegment } = await import('pikelet-wasm/complete/tombstones.mjs');
  const bytes = await readChainMemberShellBytes(layerPath, 'tombstones');
  const parsed = parseTombstoneSegment(bytes, { rowBase: L.rowBase, tombstones: L.tombstones, records: L.records });
  return parsed.supersessions ?? [];
}

// S_H: the head chain's union supersession map, as planRebase wants it.
function headEdgeMap(headInfo, head) {
  const map = new Map();
  for (const m of head.__members) {
    for (const [x, y] of m.supersessions || []) {
      if (!map.has(x)) map.set(x, []);
      map.get(x).push({ newId: y, depth: m.depth });
    }
  }
  return map;
}

// The layer's own bytes, copied without interpretation (6.2).
async function copyLayerSegments(layerPath) {
  const { readChainMemberShellBytes } = await import('./common.mjs');
  const out = {};
  for (const kind of ['index', 'corpus', 'query-interp', 'lexical']) {
    out[kind === 'query-interp' ? 'queryInterp' : kind] =
      // Null means absent. A digest or structure failure throws and the
      // rebase refuses: catching it here dropped a corrupt segment and wrote
      // an unmountable layer with exit 0.
      await readChainMemberShellBytes(layerPath, kind);
  }
  if (!out.queryInterp) throw new CliError('the layer carries no query-interp segment to copy (4.5)');
  return out;
}
