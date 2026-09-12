#!/usr/bin/env node
/**
 * Benchmark the complete-profile remote-range execution model over real HTTP
 * 206 requests. This intentionally measures storage/execution behaviour, not
 * retrieval relevance (see bakeoff-retrieval.mjs for relevance).
 *
 * Typical use:
 *
 *   npm run bench:range-storage -- \
 *     --queries test/relevance/nodeapi-queries.json \
 *     artifacts/1k.pikelet artifacts/10k.pikelet artifacts/100k.pikelet
 *
 * Each artifact is benchmarked in a fresh child process so RSS/heap numbers
 * do not accumulate caches from earlier corpus sizes. The child serves the
 * artifact from a loopback HTTP server that implements HEAD + byte ranges,
 * then opens it through complete/httpRangeSource(), exercising the same 206
 * path used against a CDN/object store.
 *
 * Reported phases:
 *   open       bytes/ranges needed before the first query (query model,
 *              resident sketch, corpus tables, eager lexical index, etc.)
 *   first-pass each supplied query once in one fresh search session
 *   repeat     the same queries again, showing the effect of row/record caches
 *
 * Loopback latency is intentionally not a WAN/CDN latency claim. Use
 * --server-delay-ms to inject a fixed delay before every HTTP response when
 * you want to see how request count interacts with a rough RTT model.
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(SCRIPT), '..');
const RESULT_PREFIX = '__PANCAKE_RANGE_BENCH_RESULT__';

function usage(exitCode = 0) {
  const out = exitCode ? console.error : console.log;
  out(`usage: node scripts/benchmark-range-storage.mjs [options] <artifact.pikelet> [...]

Options:
  --queries <file>           JSON array of strings or {text} query objects.
                             If omitted, each artifact's sampleQueries are used.
  --query-limit <n>          Maximum queries to run (default 20).
  --k <n>                    Results requested per query (default 5).
  --retrieval <mode>         hybrid | vector | lexical (default hybrid).
  --rerank <n>               Override artifact recommended rerank C.
  --parallelism <n>          Override rerank range-read parallelism.
  --gap <bytes>              Override range coalescing gap.
  --max-range-bytes <bytes>  Override maximum coalesced rerank range.
  --server-delay-ms <ms>     Fixed delay before each HTTP response (default 0).
  --encoder-module <file>    For kind-2 artifacts: ESM module exporting
                             encodeQuery(text) (named or default export).
  --no-verify-records        Disable per-record corpus integrity verification.
  --no-prefetch-encoder      Do not prefetch a lazy kind-3 encoder at open;
                             the first query then pays the encoder transfer,
                             making the open/query split visible.
  --no-verify-encoder        Disable query-encoder verification.
  --json <file>              Write full machine-readable results.
  --csv <file>               Write the summary table as CSV.
  -h, --help                 Show this help.

Examples:
  node scripts/benchmark-range-storage.mjs --queries queries.json search.pikelet

  node scripts/benchmark-range-storage.mjs --server-delay-ms 25 \\
    --queries queries.json 1k.pikelet 10k.pikelet 100k.pikelet
`);
  process.exit(exitCode);
}

function parseArgs(argv) {
  const options = {
    queries: null,
    queryLimit: 20,
    k: 5,
    retrieval: 'hybrid',
    rerank: undefined,
    parallelism: undefined,
    gap: undefined,
    maxRangeBytes: undefined,
    serverDelayMs: 0,
    encoderModule: null,
    verifyRecords: true,
    verifyEncoder: true,
    prefetchEncoder: true,
    json: null,
    csv: null,
    artifacts: [],
  };

  const take = (i, flag) => {
    if (i + 1 >= argv.length) throw new Error(`${flag} requires a value`);
    return argv[i + 1];
  };

  const number = (raw, flag, { min = 0, integer = true } = {}) => {
    const value = Number(raw);
    if (!Number.isFinite(value) || value < min || (integer && !Number.isInteger(value))) {
      throw new Error(`${flag} must be ${integer ? 'an integer' : 'a number'} >= ${min}`);
    }
    return value;
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') usage(0);
    if (arg === '--queries') { options.queries = take(i, arg); i++; continue; }
    if (arg === '--query-limit') { options.queryLimit = number(take(i, arg), arg, { min: 1 }); i++; continue; }
    if (arg === '--k') { options.k = number(take(i, arg), arg, { min: 1 }); i++; continue; }
    if (arg === '--retrieval') {
      options.retrieval = take(i, arg); i++;
      if (!['hybrid', 'vector', 'lexical'].includes(options.retrieval)) {
        throw new Error('--retrieval must be hybrid, vector, or lexical');
      }
      continue;
    }
    if (arg === '--rerank') { options.rerank = number(take(i, arg), arg, { min: 1 }); i++; continue; }
    if (arg === '--parallelism') { options.parallelism = number(take(i, arg), arg, { min: 1 }); i++; continue; }
    if (arg === '--gap') { options.gap = number(take(i, arg), arg, { min: 0 }); i++; continue; }
    if (arg === '--max-range-bytes') { options.maxRangeBytes = number(take(i, arg), arg, { min: 1 }); i++; continue; }
    if (arg === '--server-delay-ms') { options.serverDelayMs = number(take(i, arg), arg, { min: 0, integer: false }); i++; continue; }
    if (arg === '--encoder-module') { options.encoderModule = take(i, arg); i++; continue; }
    if (arg === '--no-verify-records') { options.verifyRecords = false; continue; }
    if (arg === '--no-prefetch-encoder') { options.prefetchEncoder = false; continue; }
    if (arg === '--no-verify-encoder') { options.verifyEncoder = false; continue; }
    if (arg === '--json') { options.json = take(i, arg); i++; continue; }
    if (arg === '--csv') { options.csv = take(i, arg); i++; continue; }
    if (arg.startsWith('-')) throw new Error(`unknown option ${arg}`);
    options.artifacts.push(arg);
  }
  return options;
}

function resolvePaths(options) {
  const resolveMaybe = (p) => (p ? path.resolve(p) : null);
  return {
    ...options,
    queries: resolveMaybe(options.queries),
    encoderModule: resolveMaybe(options.encoderModule),
    json: resolveMaybe(options.json),
    csv: resolveMaybe(options.csv),
    artifacts: options.artifacts.map((p) => path.resolve(p)),
  };
}

function loadQueryTexts(filePath, limit) {
  if (!filePath) return null;
  const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const rows = Array.isArray(raw) ? raw : Array.isArray(raw?.queries) ? raw.queries : null;
  if (!rows) throw new Error(`query file must be an array or {queries:[...]}: ${filePath}`);
  const seen = new Set();
  const texts = [];
  for (const row of rows) {
    const text = typeof row === 'string' ? row : row?.text;
    const normalized = String(text || '').trim();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    texts.push(normalized);
    if (texts.length >= limit) break;
  }
  if (!texts.length) throw new Error(`query file contains no usable query text: ${filePath}`);
  return texts;
}

const mib = (bytes) => bytes / (1024 * 1024);
const kib = (bytes) => bytes / 1024;
const round = (n, digits = 2) => Number(n.toFixed(digits));

function percentile(values, p) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.max(0, Math.ceil(p * sorted.length) - 1);
  return sorted[index];
}

function mean(values) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function summarizeSamples(samples) {
  const latencies = samples.map((s) => s.ms);
  const bytes = samples.map((s) => s.bytes);
  const requests = samples.map((s) => s.requests);
  return {
    n: samples.length,
    bytesTotal: bytes.reduce((a, b) => a + b, 0),
    bytesMean: mean(bytes),
    bytesP95: percentile(bytes, 0.95),
    requestsTotal: requests.reduce((a, b) => a + b, 0),
    requestsMean: mean(requests),
    requestsP95: percentile(requests, 0.95),
    latencyMeanMs: mean(latencies),
    latencyP50Ms: percentile(latencies, 0.50),
    latencyP95Ms: percentile(latencies, 0.95),
    abstained: samples.filter((s) => s.results === 0).length,
    resultCountMean: mean(samples.map((s) => s.results)),
  };
}

function memorySnapshot() {
  if (globalThis.gc) globalThis.gc();
  const m = process.memoryUsage();
  return { rss: m.rss, heapUsed: m.heapUsed, external: m.external, arrayBuffers: m.arrayBuffers };
}

function memoryDelta(after, before) {
  const out = {};
  for (const key of Object.keys(after)) out[key] = after[key] - before[key];
  return out;
}

function statDelta(after, before) {
  return { requests: after.requests - before.requests, bytes: after.bytes - before.bytes };
}

const sleep = (ms) => (ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve());

async function serveRangeFile(filePath, delayMs = 0) {
  const stat = fs.statSync(filePath);
  const size = stat.size;
  const etag = `"bench-${size}-${Math.trunc(stat.mtimeMs)}"`;
  const counters = { head: 0, gets: 0, ranges: 0, fullGets: 0, responseBytes: 0 };

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', 'http://127.0.0.1');
    if (url.pathname !== '/artifact.pikelet') {
      res.writeHead(404).end('not found');
      return;
    }
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('ETag', etag);
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
    res.setHeader('Access-Control-Allow-Origin', '*');

    if (req.method === 'HEAD') {
      counters.head++;
      await sleep(delayMs);
      res.setHeader('Content-Length', String(size));
      res.writeHead(200).end();
      return;
    }
    if (req.method !== 'GET') {
      res.writeHead(405).end();
      return;
    }
    counters.gets++;
    const range = req.headers.range;
    if (!range) {
      counters.fullGets++;
      counters.responseBytes += size;
      await sleep(delayMs);
      res.setHeader('Content-Length', String(size));
      res.writeHead(200);
      fs.createReadStream(filePath).pipe(res);
      return;
    }
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    if (!match) {
      res.setHeader('Content-Range', `bytes */${size}`);
      res.writeHead(416).end();
      return;
    }
    const start = Number(match[1]);
    const requestedEnd = Number(match[2]);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(requestedEnd)
      || start < 0 || requestedEnd < start || start >= size) {
      res.setHeader('Content-Range', `bytes */${size}`);
      res.writeHead(416).end();
      return;
    }
    const end = Math.min(requestedEnd, size - 1);
    const length = end - start + 1;
    counters.ranges++;
    counters.responseBytes += length;
    await sleep(delayMs);
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
    res.setHeader('Content-Length', String(length));
    res.writeHead(206);
    fs.createReadStream(filePath, { start, end }).pipe(res);
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  const url = `http://127.0.0.1:${address.port}/artifact.pikelet`;
  return {
    url,
    size,
    counters,
    async close() {
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

async function loadExternalEncoder(modulePath) {
  if (!modulePath) return null;
  const mod = await import(pathToFileURL(modulePath).href);
  const fn = mod.encodeQuery || mod.default;
  if (typeof fn !== 'function') {
    throw new Error(`encoder module must export encodeQuery(text) or a default function: ${modulePath}`);
  }
  return fn;
}

async function measureOneQuery(search, source, text, queryOptions) {
  const before = { requests: source.stats.requests, bytes: source.stats.bytes };
  const started = performance.now();
  const out = await search.query(text, queryOptions);
  const ms = performance.now() - started;
  const after = { requests: source.stats.requests, bytes: source.stats.bytes };
  return {
    text,
    ms,
    ...statDelta(after, before),
    results: out.results.length,
    matchQuality: out.matchQuality,
  };
}

async function worker(options) {
  const artifactPath = options.artifacts[0];
  if (!artifactPath || !fs.existsSync(artifactPath)) {
    throw new Error(`artifact not found: ${artifactPath || '(missing)'}`);
  }
  const server = await serveRangeFile(artifactPath, options.serverDelayMs);
  const warnings = [];
  try {
    const baselineMemory = memorySnapshot();
    const { openPikeletFile, httpRangeSource } = await import(
      pathToFileURL(path.join(ROOT, 'complete', 'index.mjs')).href
    );
    const encodeQuery = await loadExternalEncoder(options.encoderModule);
    const source = httpRangeSource(server.url, {
      preferredParallelism: options.parallelism,
    });
    await source.init();

    const beforeOpen = { requests: source.stats.requests, bytes: source.stats.bytes };
    const openStarted = performance.now();
    const search = await openPikeletFile(source, {
      ...(encodeQuery ? { encodeQuery } : {}),
      verifyRecords: options.verifyRecords,
      verifyEncoder: options.verifyEncoder,
      prefetchEncoder: options.prefetchEncoder,
      ...(options.parallelism !== undefined ? { rerankParallelism: options.parallelism } : {}),
      ...(options.gap !== undefined ? { rerankGap: options.gap } : {}),
      ...(options.maxRangeBytes !== undefined ? { rerankMaxRangeBytes: options.maxRangeBytes } : {}),
    });
    const openMs = performance.now() - openStarted;
    const afterOpenStats = { requests: source.stats.requests, bytes: source.stats.bytes };
    const openTransfer = statDelta(afterOpenStats, beforeOpen);
    const afterOpenMemory = memorySnapshot();
    const info = search.info();

    let queryTexts = loadQueryTexts(options.queries, options.queryLimit);
    if (!queryTexts) {
      queryTexts = (info.sampleQueries || [])
        .map(String).map((s) => s.trim()).filter(Boolean)
        .slice(0, options.queryLimit);
      if (!queryTexts.length) {
        queryTexts = ['documentation search'];
        warnings.push('artifact has no sampleQueries and --queries was omitted; used a generic query');
      } else if (queryTexts.length < 5) {
        warnings.push(`only ${queryTexts.length} sampleQueries available; use --queries for a more stable benchmark`);
      }
    }
    if (info.encoder?.kind === 'external-transformers-v1' && !encodeQuery) {
      throw new Error('artifact uses an external query encoder; pass --encoder-module <file> so queries can run');
    }

    const queryOptions = {
      k: options.k,
      retrieval: options.retrieval,
      ...(options.rerank !== undefined ? { rerank: options.rerank } : {}),
      ...(options.parallelism !== undefined ? { parallelism: options.parallelism } : {}),
      ...(options.gap !== undefined ? { gap: options.gap } : {}),
      ...(options.maxRangeBytes !== undefined ? { maxRangeBytes: options.maxRangeBytes } : {}),
    };

    const firstPass = [];
    for (const text of queryTexts) {
      firstPass.push(await measureOneQuery(search, source, text, queryOptions));
    }
    const afterFirstMemory = memorySnapshot();

    const repeatPass = [];
    for (const text of queryTexts) {
      repeatPass.push(await measureOneQuery(search, source, text, queryOptions));
    }
    const afterRepeatMemory = memorySnapshot();

    const sourceEnd = { requests: source.stats.requests, bytes: source.stats.bytes };
    const sourceBytes = sourceEnd.bytes;
    const serverBytes = server.counters.responseBytes;
    if (source.stats.fullFallback) {
      warnings.push('server unexpectedly triggered full-download fallback');
    }
    if (server.counters.fullGets) {
      warnings.push(`server observed ${server.counters.fullGets} full GET request(s)`);
    }
    // HEAD bodies have zero bytes, so requested range bytes should equal
    // actual response bytes when every GET was a 206.
    if (!server.counters.fullGets && sourceBytes !== serverBytes) {
      warnings.push(`range-source requested ${sourceBytes} bytes but server sent ${serverBytes}`);
    }

    const result = {
      artifact: artifactPath,
      artifactName: path.basename(artifactPath),
      records: info.records,
      dim: info.dim,
      metric: info.metric,
      fileBytes: info.fileBytes,
      residentBytes: info.residentBytes,
      lexicalTerms: info.lexical?.terms ?? null,
      encoderKind: info.encoder?.kind ?? null,
      recommendedRerank: info.recommendedRerank ?? null,
      config: {
        queryCount: queryTexts.length,
        k: options.k,
        retrieval: options.retrieval,
        rerank: options.rerank ?? null,
        parallelism: options.parallelism ?? null,
        gap: options.gap ?? null,
        maxRangeBytes: options.maxRangeBytes ?? null,
        serverDelayMs: options.serverDelayMs,
        prefetchEncoder: options.prefetchEncoder,
        verifyRecords: options.verifyRecords,
        verifyEncoder: options.verifyEncoder,
      },
      open: {
        ms: openMs,
        ...openTransfer,
        memoryDelta: memoryDelta(afterOpenMemory, baselineMemory),
      },
      firstQuery: firstPass[0],
      firstPass: summarizeSamples(firstPass),
      repeatPass: summarizeSamples(repeatPass),
      memory: {
        baseline: baselineMemory,
        afterOpen: afterOpenMemory,
        afterFirstPass: afterFirstMemory,
        afterRepeatPass: afterRepeatMemory,
        afterFirstPassDelta: memoryDelta(afterFirstMemory, baselineMemory),
        afterRepeatPassDelta: memoryDelta(afterRepeatMemory, baselineMemory),
      },
      transport: {
        rangeSource: {
          requests: source.stats.requests,
          bytes: source.stats.bytes,
          fullFallback: source.stats.fullFallback,
        },
        server: { ...server.counters },
      },
      warnings,
      samples: { firstPass, repeatPass },
    };

    await search.close();
    return result;
  } finally {
    await server.close();
  }
}

function csvEscape(value) {
  const s = String(value ?? '');
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function summaryRow(r) {
  return {
    artifact: r.artifactName,
    records: r.records,
    artifactMiB: round(mib(r.fileBytes), 2),
    initMiB: round(mib(r.open.bytes), 2),
    initRanges: r.open.requests,
    openMs: round(r.open.ms, 1),
    firstKiBPerQuery: round(kib(r.firstPass.bytesMean), 1),
    firstRangesPerQuery: round(r.firstPass.requestsMean, 2),
    firstP50Ms: round(r.firstPass.latencyP50Ms, 1),
    firstP95Ms: round(r.firstPass.latencyP95Ms, 1),
    repeatKiBPerQuery: round(kib(r.repeatPass.bytesMean), 1),
    repeatRangesPerQuery: round(r.repeatPass.requestsMean, 2),
    repeatP50Ms: round(r.repeatPass.latencyP50Ms, 1),
    rssDeltaMiB: round(mib(r.memory.afterRepeatPassDelta.rss), 1),
    abstainedFirstPass: r.firstPass.abstained,
  };
}

function printHuman(results, options) {
  console.log(`\nPikelet HTTP range-storage benchmark`);
  console.log(`retrieval=${options.retrieval}, k=${options.k}, query-limit=${options.queryLimit}, server-delay=${options.serverDelayMs}ms`);
  console.log('latencies are loopback client+HTTP+search timings, not public-CDN latency\n');
  console.table(results.map(summaryRow));

  for (const r of results) {
    console.log(`${r.artifactName}: ${r.records.toLocaleString()} records, ${mib(r.fileBytes).toFixed(2)} MiB artifact`);
    console.log(`  open: ${mib(r.open.bytes).toFixed(2)} MiB in ${r.open.requests} range GETs, ${r.open.ms.toFixed(1)} ms`);
    console.log(`  first query: ${kib(r.firstQuery.bytes).toFixed(1)} KiB in ${r.firstQuery.requests} ranges, ${r.firstQuery.ms.toFixed(1)} ms`);
    console.log(`  first pass: ${kib(r.firstPass.bytesMean).toFixed(1)} KiB/query, ${r.firstPass.requestsMean.toFixed(2)} ranges/query, `
      + `p50 ${r.firstPass.latencyP50Ms.toFixed(1)} ms, p95 ${r.firstPass.latencyP95Ms.toFixed(1)} ms`);
    console.log(`  repeat pass: ${kib(r.repeatPass.bytesMean).toFixed(1)} KiB/query, ${r.repeatPass.requestsMean.toFixed(2)} ranges/query, `
      + `p50 ${r.repeatPass.latencyP50Ms.toFixed(1)} ms, p95 ${r.repeatPass.latencyP95Ms.toFixed(1)} ms`);
    console.log(`  memory: RSS delta after repeat pass ${mib(r.memory.afterRepeatPassDelta.rss).toFixed(1)} MiB `
      + `(heap ${mib(r.memory.afterRepeatPassDelta.heapUsed).toFixed(1)} MiB, external ${mib(r.memory.afterRepeatPassDelta.external).toFixed(1)} MiB)`);
    for (const warning of r.warnings) console.log(`  warning: ${warning}`);
  }
}

function runChild(artifact, options) {
  const childOptions = { ...options, artifacts: [artifact], json: null, csv: null };
  const child = spawnSync(
    process.execPath,
    ['--expose-gc', SCRIPT, '--worker'],
    {
      cwd: ROOT,
      env: { ...process.env, PANCAKE_RANGE_BENCH_OPTIONS: JSON.stringify(childOptions) },
      encoding: 'utf8',
      maxBuffer: 32 * 1024 * 1024,
    },
  );
  if (child.status !== 0) {
    const detail = [child.stderr, child.stdout].filter(Boolean).join('\n').trim();
    throw new Error(`benchmark failed for ${artifact}${detail ? `:\n${detail}` : ''}`);
  }
  const line = child.stdout.split(/\r?\n/).find((s) => s.startsWith(RESULT_PREFIX));
  if (!line) throw new Error(`benchmark child returned no result for ${artifact}`);
  return JSON.parse(line.slice(RESULT_PREFIX.length));
}

async function main() {
  if (process.argv[2] === '--worker') {
    const raw = process.env.PANCAKE_RANGE_BENCH_OPTIONS;
    if (!raw) throw new Error('worker options missing');
    const result = await worker(JSON.parse(raw));
    process.stdout.write(`${RESULT_PREFIX}${JSON.stringify(result)}\n`);
    return;
  }

  let options;
  try {
    options = resolvePaths(parseArgs(process.argv.slice(2)));
  } catch (err) {
    console.error(`error: ${err.message}`);
    usage(1);
  }
  if (!options.artifacts.length) usage(1);
  if (options.queries && !fs.existsSync(options.queries)) {
    throw new Error(`query file not found: ${options.queries}`);
  }
  if (options.encoderModule && !fs.existsSync(options.encoderModule)) {
    throw new Error(`encoder module not found: ${options.encoderModule}`);
  }
  for (const artifact of options.artifacts) {
    if (!fs.existsSync(artifact)) throw new Error(`artifact not found: ${artifact}`);
  }

  const results = [];
  for (const artifact of options.artifacts) {
    console.log(`benchmarking ${path.basename(artifact)} ...`);
    results.push(runChild(artifact, options));
  }
  results.sort((a, b) => a.records - b.records);
  printHuman(results, options);

  if (options.json) {
    fs.mkdirSync(path.dirname(options.json), { recursive: true });
    fs.writeFileSync(options.json, `${JSON.stringify({
      generatedAt: new Date().toISOString(),
      options,
      results,
    }, null, 2)}\n`);
    console.log(`\nwrote ${options.json}`);
  }
  if (options.csv) {
    const rows = results.map(summaryRow);
    const headers = Object.keys(rows[0] || {});
    const csv = [
      headers.join(','),
      ...rows.map((row) => headers.map((h) => csvEscape(row[h])).join(',')),
    ].join('\n') + '\n';
    fs.mkdirSync(path.dirname(options.csv), { recursive: true });
    fs.writeFileSync(options.csv, csv);
    console.log(`wrote ${options.csv}`);
  }
}

main().catch((err) => {
  console.error(err?.stack || err?.message || String(err));
  process.exit(1);
});
