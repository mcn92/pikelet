#!/usr/bin/env node
// End-to-end check of pikelet as a user experiences it:
// scaffold a project (stub embeddings, no Cloudflare account), install it,
// start its Worker with `wrangler dev`, and query /search — once per runtime.
// Also exercises `compile` (complete kind-3 .pikelet from a fixture corpus,
// opened and queried with the in-repo reader).
//
// Two things this catches that the in-repo smoke does not:
//   1. The generated project's pikelet-wasm dependency range drifting from
//      the scaffolder's own (0.3.0 shipped ^0.2.0 and every artifact-runtime
//      project answered SNAPSHOT_INVALID: the 0.2.x reader cannot open the
//      format-v3 artifacts the 0.3.0 builder writes). Asserted directly, and
//      the in-repo pikelet-wasm version must satisfy the generated range.
//   2. The generated Worker not actually serving queries against the
//      artifacts this tree builds. The project is installed with the
//      in-repo pikelet-wasm tarball (npm pack), so the reader under test is
//      the one about to be published, not whatever the registry has.
//
// Usage: node scripts/test-scaffold-e2e.mjs [--runtime artifact|snapshot]...
//        (default: both). Needs network for the generated project's other
//        dependencies (wrangler) and a free local port; no Cloudflare login.

import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CPS_DIR = path.join(ROOT, 'pikelet');
const CPS_BIN = path.join(CPS_DIR, 'bin', 'pikelet.mjs');
const requestedRuntimes = process.argv.slice(2).flatMap((arg, i, all) => (arg === '--runtime' ? [all[i + 1]] : []));
const RUNTIMES = requestedRuntimes.length ? requestedRuntimes : ['artifact', 'snapshot'];

let passed = 0;
let failed = 0;
function ok(condition, label, detail = '') {
  if (condition) { passed++; console.log(`  ok: ${label}`); }
  else { failed++; console.log(`  FAIL: ${label}${detail ? ` — ${detail}` : ''}`); }
}
function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], ...opts });
  if (res.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} failed (${res.status}):\n${res.stdout}\n${res.stderr}`);
  }
  return res.stdout;
}
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';

// Minimal caret-range check, enough for the ranges this repo writes; anything
// else is reported as a failure so a reviewer looks at it.
function caretSatisfies(range, version) {
  const m = /^\^(\d+)\.(\d+)\.(\d+)$/.exec(range);
  const v = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!m || !v) return null;
  const [rMaj, rMin, rPat] = m.slice(1).map(Number);
  const [vMaj, vMin, vPat] = v.slice(1).map(Number);
  if (rMaj > 0) return vMaj === rMaj && (vMin > rMin || (vMin === rMin && vPat >= rPat));
  if (rMin > 0) return vMaj === 0 && vMin === rMin && vPat >= rPat;
  return vMaj === 0 && vMin === 0 && vPat === rPat;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => { const { port } = srv.address(); srv.close((e) => (e ? reject(e) : resolve(port))); });
    srv.on('error', reject);
  });
}
async function waitFor(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try { const r = await fetch(url); if (r.ok) return true; } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}
function killTree(proc) {
  if (!proc || proc.exitCode !== null) return;
  try { process.kill(-proc.pid, 'SIGTERM'); } catch { try { proc.kill('SIGTERM'); } catch { /* gone */ } }
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'pancake-scaffold-e2e-'));
const rootPkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const cpsPkg = JSON.parse(fs.readFileSync(path.join(CPS_DIR, 'package.json'), 'utf8'));
const cpsRange = cpsPkg.dependencies?.['pikelet-wasm'];

console.log(`Scaffold e2e: pikelet-wasm ${rootPkg.version} (in-repo), pikelet ${cpsPkg.version}, runtimes ${RUNTIMES.join(', ')}`);
console.log(`work dir: ${work}`);

// In-repo tarballs for BOTH packages, installed into each generated project:
// pikelet-wasm so the reader under test is the one about to be published,
// and pikelet because the generated devDependency pins the
// in-repo version, which does not exist on the registry until it is
// published — before a release, resolving it from npm fails with ETARGET.
const packOut = run(npm, ['pack', '--pack-destination', work, '--ignore-scripts'], { cwd: ROOT });
const tarball = path.join(work, packOut.trim().split('\n').pop());
ok(fs.existsSync(tarball), `packed in-repo pancake-wasm: ${path.basename(tarball)}`);
const cpsPackOut = run(npm, ['pack', '--pack-destination', work, '--ignore-scripts'], { cwd: CPS_DIR });
const cpsTarball = path.join(work, cpsPackOut.trim().split('\n').pop());
ok(fs.existsSync(cpsTarball), `packed in-repo pikelet: ${path.basename(cpsTarball)}`);

// The compile command: a complete kind-3 .pikelet from a small fixture
// corpus, opened and queried with the in-repo complete reader. A fixture
// rather than ROOT/docs because the inline transformer embeds for real
// (no stub path), and the guard rail: --runtime complete must point at
// compile instead of silently building a snapshot project (the pre-0.6
// behavior).
console.log('\n--- compile ---');
{
  const fixtureDir = path.join(work, 'compile-fixture');
  fs.mkdirSync(fixtureDir, { recursive: true });
  const fixtures = {
    'volcanoes.md': 'Volcanoes form where magma rises through cracks in the crust and erupts onto the surface. Repeated eruptions of lava and ash build the cone over thousands of years, usually along tectonic plate boundaries where one plate subducts beneath another and melts into fresh magma below.',
    'routers.md': 'A network router forwards packets between different networks by reading the destination address in each packet header and consulting its routing table. Routes are configured statically or learned through routing protocols, and home routers also perform network address translation with a built-in firewall.',
    'sourdough.md': 'Sourdough bread rises without commercial yeast by relying on a starter, a live culture of wild yeast and lactic acid bacteria kept alive with regular feedings of flour and water. The long fermentation develops flavor and structure, and the acidity gives the crumb its characteristic tang.',
  };
  for (const [name, text] of Object.entries(fixtures)) fs.writeFileSync(path.join(fixtureDir, name), `# ${name}\n\n${(text + ' ').repeat(3)}`);
  const outFile = path.join(work, 'compiled', 'search.pikelet');
  run(process.execPath, [CPS_BIN, 'compile', '--source', fixtureDir, '--out', outFile], { cwd: CPS_DIR });
  ok(fs.existsSync(outFile), 'compile wrote the .pikelet artifact');

  const { openPikeletFile } = await import(path.join(ROOT, 'complete', 'index.mjs'));
  const search = await openPikeletFile(outFile);
  try {
    const info = search.info();
    ok(info.corpusIntegrity === 'per-record-sha256' && info.indexRowIntegrity === 'per-row-sha256',
      'compiled artifact is format 2 with per-record and per-row integrity', JSON.stringify({ corpus: info.corpusIntegrity, index: info.indexRowIntegrity }));
    const out = await search.query('how does bread rise without yeast', { k: 2 });
    ok(out.results[0]?.sourcePath?.includes('sourdough') || out.results[0]?.title?.includes('sourdough'),
      'compiled artifact answers a natural-language query with no host encoder', JSON.stringify(out.results[0] || {}).slice(0, 200));
    ok(out.matchQuality === 'strong', 'on-topic query scores strong via the self-calibrated abstention model', `got ${out.matchQuality} conf ${out.confidence}`);
    const offDomain = await search.query('medicare part d formulary exception', { k: 2 });
    ok(offDomain.matchQuality === 'none' && offDomain.results.length === 0,
      'off-domain query abstains with no results', `got ${offDomain.matchQuality} with ${offDomain.results.length} results`);
    const gibberish = await search.query('xqzvw plorth grimbleflax snorp', { k: 2 });
    ok(gibberish.matchQuality === 'none' && gibberish.results.length === 0,
      'gibberish query abstains with no results', `got ${gibberish.matchQuality} with ${gibberish.results.length} results`);
  } finally {
    await search.close();
  }

  const rejected = spawnSync(process.execPath,
    [CPS_BIN, '--name', path.join(work, 'proj-complete'), '--source', fixtureDir, '--runtime', 'complete', '--no-deploy', '--yes'],
    { encoding: 'utf8', cwd: CPS_DIR });
  ok(rejected.status !== 0 && `${rejected.stderr}${rejected.stdout}`.includes('compile --source'),
    'scaffold --runtime complete is rejected with a pointer to compile', `${rejected.status} ${rejected.stderr.slice(0, 200)}`);

  // The MCP server: the compiled pack attached the way an LLM client
  // attaches it — stdio JSON-RPC, tools discovered and called, provenance
  // and calibrated abstention crossing the protocol intact.
  console.log('\n--- mcp ---');
  {
    const child = spawn(process.execPath, [CPS_BIN, 'mcp', '--pack', outFile, '--pack', outFile],
      { cwd: CPS_DIR, stdio: ['pipe', 'pipe', 'pipe'] });
    const lines = [];
    let buffer = '';
    child.stdout.on('data', (d) => {
      buffer += d;
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        lines.push(JSON.parse(buffer.slice(0, idx)));
        buffer = buffer.slice(idx + 1);
      }
    });
    const send = (m) => child.stdin.write(`${JSON.stringify(m)}\n`);
    const waitFor = (id) => new Promise((resolve, reject) => {
      const deadline = Date.now() + 120000;
      const timer = setInterval(() => {
        const hit = lines.find((l) => l.id === id);
        if (hit) { clearInterval(timer); resolve(hit); }
        else if (Date.now() > deadline) { clearInterval(timer); reject(new Error(`mcp reply ${id} timed out`)); }
      }, 20);
    });
    const callTool = async (id, name, args) => {
      send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });
      const res = (await waitFor(id)).result;
      const isError = res.isError === true;
      // Error results carry plain text, successful ones carry JSON.
      return { isError, body: isError ? res.content[0].text : JSON.parse(res.content[0].text) };
    };
    try {
      send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'e2e', version: '0' } } });
      const init = (await waitFor(1)).result;
      ok(init.protocolVersion === '2025-06-18' && init.serverInfo?.name === 'pikelet-knowledge-packs',
        'mcp initialize handshakes with the requested protocol version', JSON.stringify(init).slice(0, 120));
      send({ jsonrpc: '2.0', method: 'notifications/initialized' });
      send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
      const tools = (await waitFor(2)).result.tools.map((t) => t.name).sort();
      ok(JSON.stringify(tools) === JSON.stringify(['get_record', 'list_packs', 'search', 'verify_pack']),
        'mcp lists the four pack tools', tools.join(','));

      const packs = (await callTool(3, 'list_packs', {})).body.packs;
      ok(packs.length === 2 && packs[0].name !== packs[1].name
        && packs[0].identity === packs[1].identity && /^[0-9a-f]{64}$/.test(packs[0].identity),
        'duplicate pack names get suffixed; identities stay the manifest sha256', JSON.stringify(packs.map((p) => p.name)));

      const packName = packs[0].name;
      const hit = (await callTool(4, 'search', { query: 'how does bread rise without yeast', pack: packName, k: 2 })).body;
      const section = hit.sections[0];
      const top = section.results[0];
      ok(section.matchQuality === 'strong' && top && top.pack === packName
        && top.packIdentity === packs[0].identity && typeof top.text === 'string'
        && Number.isFinite(top.distance) && (top.source || top.title),
        'mcp search returns provenanced results with calibrated quality', JSON.stringify({ q: section.matchQuality, top: { ...top, text: (top.text || '').slice(0, 40) } }));

      const off = (await callTool(5, 'search', { query: 'medicare part d formulary exception' })).body;
      ok(off.sections.every((s) => s.matchQuality === 'none' && s.results.length === 0) && typeof off.note === 'string',
        'off-domain query abstains in every pack and says so', JSON.stringify(off).slice(0, 160));

      const record = (await callTool(6, 'get_record', { pack: packName, id: top.id })).body;
      ok(record.pack === packName && record.id === top.id && typeof record.text === 'string' && record.text.length >= top.text.length,
        'get_record hydrates the full provenanced chunk', JSON.stringify(record).slice(0, 120));

      const badPack = await callTool(7, 'search', { query: 'x', pack: 'nope' });
      const badTool = await callTool(8, 'no_such_tool', {}).catch(() => null);
      ok(badPack.isError === true, 'unknown pack is a correctable tool error, not a protocol error');
      ok(badTool === null || badTool.isError === true, 'unknown tool is reported as an error');
      const ambiguous = await callTool(9, 'get_record', { id: 0 });
      ok(ambiguous.isError === true, 'get_record without pack errors when multiple packs are mounted');

      // verify_pack: the compiled fixture embeds retrieval-verified golden
      // queries (calibration positives), runnable as tests from inside the
      // file.
      const verify = (await callTool(10, 'verify_pack', { pack: packName })).body;
      ok(verify.verdict === 'pass' && verify.goldenQueries.total > 0
        && verify.goldenQueries.passed === verify.goldenQueries.total
        && verify.encoderVerified === true && verify.corpusIntegrity === 'per-record-sha256',
        'verify_pack runs the embedded goldens and they pass', JSON.stringify({ verdict: verify.verdict, goldens: verify.goldenQueries.total, enc: verify.encoderVerified }));
    } finally {
      child.stdin.end();
      await new Promise((resolve) => child.on('close', resolve));
    }

    // URL mount with identity pinning: range-served over local HTTP, the
    // wrong pin refused, the right pin served.
    const http = await import('node:http');
    const packBytes = fs.statSync(outFile).size;
    const rangeSrv = http.createServer((req, res) => {
      const r = req.headers.range && /^bytes=(\d+)-(\d+)?$/.exec(req.headers.range);
      if (r) {
        const start = +r[1];
        const end = r[2] !== undefined ? Math.min(+r[2], packBytes - 1) : packBytes - 1;
        res.writeHead(206, { 'accept-ranges': 'bytes', 'content-range': `bytes ${start}-${end}/${packBytes}`, 'content-length': end - start + 1 });
        fs.createReadStream(outFile, { start, end }).pipe(res);
        return;
      }
      res.writeHead(req.method === 'HEAD' ? 200 : 200, { 'accept-ranges': 'bytes', 'content-length': packBytes });
      if (req.method === 'HEAD') res.end();
      else fs.createReadStream(outFile).pipe(res);
    });
    await new Promise((resolve) => rangeSrv.listen(0, '127.0.0.1', resolve));
    const packUrl = `http://127.0.0.1:${rangeSrv.address().port}/search.pikelet`;
    const { openPikeletFile: openForIdentity } = await import(path.join(ROOT, 'complete', 'index.mjs'));
    const identityReader = await openForIdentity(outFile);
    const packIdentity = identityReader.info().identity;
    await identityReader.close();
    // Async spawn, not spawnSync: the child's background encoder prefetch
    // reads from rangeSrv in THIS process — a synchronous wait deadlocks.
    const badPin = await new Promise((resolve) => {
      const child = spawn(process.execPath, [CPS_BIN, 'mcp', '--pack', `${packUrl}#${'f'.repeat(64)}`],
        { cwd: CPS_DIR, stdio: ['pipe', 'ignore', 'pipe'] });
      let stderr = '';
      child.stderr.on('data', (d) => { stderr += d; });
      child.stdin.end();
      child.on('close', (status) => resolve({ status, stderr }));
    });
    ok(badPin.status !== 0 && badPin.stderr.includes('identity mismatch'),
      'a wrong identity pin refuses the mount', badPin.stderr.slice(0, 160));
    {
      const urlChild = spawn(process.execPath, [CPS_BIN, 'mcp', '--pack', `${packUrl}#${packIdentity}`],
        { cwd: CPS_DIR, stdio: ['pipe', 'pipe', 'pipe'] });
      const urlLines = [];
      let urlBuf = '';
      urlChild.stdout.on('data', (d) => {
        urlBuf += d;
        let idx;
        while ((idx = urlBuf.indexOf('\n')) >= 0) { urlLines.push(JSON.parse(urlBuf.slice(0, idx))); urlBuf = urlBuf.slice(idx + 1); }
      });
      urlChild.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'list_packs', arguments: {} } })}\n`);
      const listed = await new Promise((resolve, reject) => {
        const deadline = Date.now() + 120000;
        const timer = setInterval(() => {
          const hit = urlLines.find((l) => l.id === 1);
          if (hit) { clearInterval(timer); resolve(hit); }
          else if (Date.now() > deadline) { clearInterval(timer); reject(new Error('url mount timed out')); }
        }, 20);
      });
      const urlPacks = JSON.parse(listed.result.content[0].text).packs;
      ok(urlPacks.length === 1 && urlPacks[0].remote === true && urlPacks[0].identity === packIdentity,
        'a URL pack mounts over range reads with the pinned identity', JSON.stringify(urlPacks[0]));
      urlChild.stdin.end();
      await new Promise((resolve) => urlChild.on('close', resolve));
    }
    rangeSrv.close();

    // mcp install: writes the client config as a file operation, no server.
    const installDir = path.join(work, 'install-target');
    fs.mkdirSync(installDir, { recursive: true });
    const { installMcpConfig } = await import(path.join(CPS_DIR, 'src', 'mcp.mjs'));
    const written = await installMcpConfig({ packPaths: [outFile], client: 'claude-code', cwd: installDir });
    const config = JSON.parse(fs.readFileSync(written.configPath, 'utf8'));
    const entry = config.mcpServers['knowledge-packs'];
    ok(written.configPath === path.join(installDir, '.mcp.json') && entry.command === 'npx'
      && entry.args.includes('--pack') && entry.args.includes(outFile),
      'mcp install writes a claude-code .mcp.json with absolute pack paths', JSON.stringify(entry));
    const refused = await installMcpConfig({ packPaths: [outFile], client: 'claude-code', cwd: installDir }).catch((e) => e);
    ok(refused instanceof Error && /--force/.test(refused.message),
      'mcp install refuses to replace an existing entry without --force');
  }

  // URL crawl filters: pattern semantics and the folder-glob/URL-pattern
  // mismatch erroring instead of silently no-opping.
  const { urlCrawlFilter } = await import(path.join(CPS_DIR, 'src', 'ingest.mjs'));
  const defaultFilter = urlCrawlFilter({ url: 'https://docs.example.com/' });
  ok(!defaultFilter('https://docs.example.com/print.html') && defaultFilter('https://docs.example.com/keymap.html'),
    'URL crawl excludes print.html by default and keeps normal pages');
  const scoped = urlCrawlFilter({ url: 'https://docs.example.com/', includeUrl: ['/guide/*'], excludeUrl: ['*/draft-*'] });
  ok(scoped('https://docs.example.com/guide/setup.html') && !scoped('https://docs.example.com/blog/post.html')
    && !scoped('https://docs.example.com/guide/draft-new.html'),
    'include-url and exclude-url patterns scope the crawl frontier');
  // Seed redirects are followed (bounded); mid-crawl redirects stay skipped.
  const { resolveSeedUrl } = await import(path.join(CPS_DIR, 'src', 'ingest.mjs'));
  const http = await import('node:http');
  const redirectPort = await freePort();
  const redirectSrv = http.createServer((req, res) => {
    if (req.url === '/') { res.writeHead(301, { location: '/hop/' }); res.end(); return; }
    if (req.url === '/hop/') {
      res.writeHead(200, { 'content-type': 'text/html' });
      res.end('<!DOCTYPE html>\n<meta http-equiv="refresh" content="0;url=/en/start/">');
      return;
    }
    if (req.url === '/en/start/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<html><body>start</body></html>'); return; }
    res.writeHead(404); res.end();
  });
  await new Promise((resolve) => redirectSrv.listen(redirectPort, '127.0.0.1', resolve));
  try {
    const hops = [];
    const resolved = await resolveSeedUrl(new URL(`http://127.0.0.1:${redirectPort}/`), (m) => hops.push(m));
    ok(resolved.pathname === '/en/start/'
      && hops.some((m) => m.includes('seed redirected:'))
      && hops.some((m) => m.includes('meta refresh')),
      'seed follows HTTP and meta-refresh redirects to its target', `${resolved.href} ${hops.join(';')}`);
  } finally {
    redirectSrv.close();
  }
  const globOnUrl = spawnSync(process.execPath,
    [CPS_BIN, 'compile', '--source', 'https://docs.example.com', '--exclude', '*print*', '--out', path.join(work, 'x.pikelet')],
    { encoding: 'utf8', cwd: CPS_DIR });
  ok(globOnUrl.status !== 0 && `${globOnUrl.stderr}${globOnUrl.stdout}`.includes('--exclude-url'),
    'folder globs against a URL source error instead of silently no-opping', globOnUrl.stderr.slice(0, 200));
}

let worker = null;
try {
  for (const runtime of RUNTIMES) {
    console.log(`\n--- runtime: ${runtime} ---`);
    const projectDir = path.join(work, `proj-${runtime}`);
    run(process.execPath, [CPS_BIN, '--name', projectDir, '--source', path.join(ROOT, 'docs'), '--runtime', runtime, '--no-deploy', '--yes', '--force'], {
      cwd: CPS_DIR,
      env: { ...process.env, PIKELET_SEARCH_STUB_EMBEDDINGS: '1' },
    });
    const genPkgPath = path.join(projectDir, 'package.json');
    ok(fs.existsSync(genPkgPath), 'scaffold wrote package.json');
    const genPkg = JSON.parse(fs.readFileSync(genPkgPath, 'utf8'));
    const genRange = genPkg.dependencies?.['pikelet-wasm'];
    ok(genRange === cpsRange, `generated pikelet-wasm range equals the scaffolder's own (${cpsRange})`, `generated ${genRange}`);
    const sat = caretSatisfies(genRange, rootPkg.version);
    ok(sat === true, `in-repo pikelet-wasm ${rootPkg.version} satisfies generated range ${genRange}`, sat === null ? 'unrecognized range shape' : 'not satisfied');

    // Install as generated, with both in-repo tarballs on top.
    run(npm, ['install', '--no-audit', '--no-fund', '--no-progress', tarball, cpsTarball], { cwd: projectDir });
    const installed = JSON.parse(fs.readFileSync(path.join(projectDir, 'node_modules', 'pikelet-wasm', 'package.json'), 'utf8')).version;
    ok(installed === rootPkg.version, `generated project runs pikelet-wasm ${installed}`);

    const localToml = fs.existsSync(path.join(projectDir, 'wrangler.local.toml')) ? 'wrangler.local.toml' : 'wrangler.toml';
    const port = await freePort();
    const args = ['wrangler', 'dev', '--config', localToml, '--port', String(port), '--log-level', 'error'];
    if (localToml === 'wrangler.toml') args.push('--var', 'LOCAL_STUB_AI:1');
    let workerLog = '';
    worker = spawn(npx, args, { cwd: projectDir, detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
    worker.stdout.on('data', (d) => { workerLog += d; });
    worker.stderr.on('data', (d) => { workerLog += d; });

    const base = `http://127.0.0.1:${port}`;
    const up = await waitFor(`${base}/health`, 120_000);
    ok(up, `wrangler dev answers /health on :${port}`, workerLog.slice(-800));
    if (up) {
      const health = await (await fetch(`${base}/health`)).json();
      ok(health.runtime_mode === runtime, `/health reports runtime_mode ${runtime}`, JSON.stringify(health).slice(0, 300));
      const res = await fetch(`${base}/search?q=${encodeURIComponent('pikelet search artifact')}&k=3`);
      const body = await res.json().catch(() => ({}));
      ok(res.status === 200 && !body.error, `/search returns 200 without error`, `${res.status} ${JSON.stringify(body).slice(0, 300)}`);
      ok(Array.isArray(body.results) && body.results.length > 0 && typeof body.results[0].title === 'string',
        `/search returns hydrated results (${body.result_count ?? '?'} for k=3)`, JSON.stringify(body).slice(0, 300));
      if (runtime === 'artifact') {
        ok(body.artifact && body.artifact.query_range_requests > 0, 'artifact runtime served the query via range reads', JSON.stringify(body.artifact || {}).slice(0, 200));
      }
      const second = await (await fetch(`${base}/search?q=${encodeURIComponent('worker snapshot restore')}&k=2`)).json().catch(() => ({}));
      ok(Array.isArray(second.results) && second.results.length > 0, 'second query (warm) returns results');
    }
    killTree(worker);
    await new Promise((r) => setTimeout(r, 1500));
    worker = null;
  }
} finally {
  killTree(worker);
}

console.log(`\nScaffold e2e: ${passed} passed, ${failed} failed`);
if (failed === 0) fs.rmSync(work, { recursive: true, force: true });
else console.log(`kept ${work} for inspection`);
process.exit(failed === 0 ? 0 : 1);
