#!/usr/bin/env node
// The full loop: mount the wiki pack over HTTP into the real `pikelet mcp`
// server, then hand it to a REAL headless Claude subprocess
// (`claude -p --mcp-config ... --strict-mcp-config`) restricted to only
// the wiki-pack MCP tools — no built-in knowledge tools, no filesystem, no
// web. Every answer requires an actual `search` tool call against the
// pack served from a deliberately dumb static HTTP server. Reports the
// LLM's full text response next to the dumb server's own request/byte log
// for that query, so "what the LLM said" and "what actually went over the
// wire" are shown from two independent vantage points.
//
// Follows the spawn pattern already proven in
// local-packs/synthbench/llm-driver.mjs (plain `claude -p` text output,
// stdin ignored from the start, a hard kill-timeout) rather than the
// --output-format stream-json + execFileSync approach, which hung
// unpredictably nested two subprocesses deep.
//
// Caveat this script deliberately surfaces rather than hides: each
// `claude -p` call is an independent process with its own fresh `pikelet
// mcp` subprocess — there is no shared mount/cache across queries here,
// unlike mcp-proof.mjs (one long-lived session, one mount, many queries).
// Every query below pays full mount cost again from scratch.
//
// Usage: node bench/range-proof/llm-proof.mjs [path/to/pack.pikelet]
// Requires the `claude` CLI on PATH.

import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { startDumbServer } from './dumb-server.mjs';
import { openPikeletFile } from '../../complete/index.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const PIKELET_BIN = path.join(ROOT, 'pikelet/bin/pikelet.mjs');
const packPath = path.resolve(process.argv[2] || path.join(ROOT, 'examples/05-one-file-search/pancake-wiki-inline.pancake'));
const packDir = path.dirname(packPath);
const packFile = path.basename(packPath);

const QUERIES = [
  'Who was the first person on the moon?',
  'How do volcanoes form?',
  'What causes earthquakes?',
  'How does photosynthesis work?',
  'What is the capital of France?',
];

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
}

function requestsAndBytes(logSlice) {
  let requests = 0;
  let bytes = 0;
  for (const entry of logSlice) {
    if (entry.method !== 'HEAD') requests += 1;
    const m = entry.range && /^bytes=(\d+)-(\d+)$/.exec(entry.range);
    if (m) bytes += Number(m[2]) - Number(m[1]) + 1;
  }
  return { requests, bytes, heads: logSlice.filter((e) => e.method === 'HEAD').length };
}

// Proven pattern from local-packs/synthbench/llm-driver.mjs: stdin ignored
// from the start (not a pipe left open), plain text output (no
// stream-json), a hard kill-timeout so a stuck call can never hang the
// whole run.
function runClaude(question, mcpConfigPath) {
  return new Promise((resolve) => {
    const args = [
      '-p', `${question} Use the wiki-pack tools to answer and cite the record id(s) you used.`,
      '--mcp-config', mcpConfigPath,
      '--strict-mcp-config',
      '--allowedTools', 'mcp__wiki-pack__search', 'mcp__wiki-pack__list_packs', 'mcp__wiki-pack__get_record',
    ];
    const child = spawn('claude', args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), 120000);
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('close', (code) => { clearTimeout(timer); resolve({ code, out, err }); });
  });
}

async function main() {
  console.log('='.repeat(76));
  console.log('Start a deliberately dumb static file server');
  console.log('='.repeat(76));
  const server = await startDumbServer(packDir, { port: 0, logToStdout: false });
  console.log(`serving ${path.relative(ROOT, packDir)} on ${server.url}`);
  console.log('(fs.createReadStream + Range support only — no database, no index, no query logic)');
  console.log();

  const localSearch = await openPikeletFile(packPath);
  const identity = localSearch.info().identity;
  await localSearch.close();

  const url = `${server.url}/${encodeURIComponent(packFile)}#${identity}`;
  const mcpConfigPath = path.join(ROOT, '.tmp-test-work', 'wiki-mcp-config.json');
  fs.mkdirSync(path.dirname(mcpConfigPath), { recursive: true });
  fs.writeFileSync(mcpConfigPath, JSON.stringify({
    mcpServers: { 'wiki-pack': { command: 'node', args: [PIKELET_BIN, 'mcp', url] } },
  }, null, 2));

  console.log(`MCP config points at: ${url}`);
  console.log('Each query below spawns an independent `claude -p` process, restricted to');
  console.log('only the wiki-pack MCP tools (--strict-mcp-config) — no other knowledge source.');
  console.log();

  const rows = [];
  for (const q of QUERIES) {
    console.log('='.repeat(76));
    console.log(`USER: ${q}`);
    console.log('='.repeat(76));
    const before = server.log.length;
    const t0 = Date.now();
    const res = await runClaude(q, mcpConfigPath);
    const ms = Date.now() - t0;
    const after = server.log.length;
    console.log(res.out.trim() || `(no output, exit code ${res.code})`);
    if (res.code !== 0 && res.err) console.log(`[stderr] ${res.err.trim().slice(0, 500)}`);
    const { requests, bytes, heads } = requestsAndBytes(server.log.slice(before, after));
    console.log(`\n[network] ${requests} range requests (+ ${heads} HEAD probe), ${fmtBytes(bytes)} transferred, ${ms} ms wall time`);
    console.log();
    rows.push({ query: q, requests, bytes, heads, ms, code: res.code });
  }

  console.log('='.repeat(76));
  console.log('SUMMARY');
  console.log('='.repeat(76));
  console.log(`${'Query'.padEnd(38)}  Requests  Bytes         ms  exit`);
  for (const r of rows) {
    console.log(`${r.query.slice(0, 38).padEnd(38)}  ${String(r.requests).padStart(8)}  ${fmtBytes(r.bytes).padStart(11)}  ${String(r.ms).padStart(5)}  ${r.code}`);
  }
  const totalRequests = rows.reduce((s, r) => s + r.requests, 0);
  const totalBytes = rows.reduce((s, r) => s + r.bytes, 0);
  console.log();
  console.log(`Total: ${totalRequests} range requests, ${fmtBytes(totalBytes)} transferred across ${rows.length} independent, cold LLM sessions.`);
  console.log();
  console.log('Caveat: each session above is a fresh process with its own fresh pikelet mcp');
  console.log('mount — no shared cache between queries, so every query pays full mount cost');
  console.log('again. See mcp-proof.mjs for the one-session, many-queries number (mount');
  console.log('amortized once), which is the number that matters for a live agent session.');

  await server.close();
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
