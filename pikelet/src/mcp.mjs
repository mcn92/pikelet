// MCP server over .pikelet knowledge packs: `pikelet mcp
// --pack a.pikelet --pack b.pikelet` speaks Model Context Protocol on
// stdio, so any MCP client (Claude Code, Claude Desktop, an agent
// framework) can attach compiled packs as a retrieval tool — no vector
// database, embedding service, or retrieval backend. The pack carries the
// corpus, indexes, query encoder, and calibration; this server carries
// nothing but the reader.
//
// Design stances:
// - Provenance is first-class: every result names its pack, the pack's
//   immutable manifest identity (sha256), and the chunk's title, heading
//   path, anchor, and source — so an answer can cite the exact knowledge
//   state it was derived from.
// - Abstention is surfaced, not smoothed over: a pack that cannot answer
//   says so per pack (matchQuality 'none' with zero results), which is
//   exactly what a grounding layer must be able to tell a model.
// - Multi-pack search returns per-pack sections, never a cross-pack
//   fusion: distances are only comparable within one pack's encoder and
//   corpus, and keeping packs separate keeps provenance separable.
// - The protocol layer is hand-rolled: MCP stdio is newline-delimited
//   JSON-RPC 2.0 and this server needs four methods; a protocol SDK would
//   be this package's third runtime dependency for less code than this
//   comment block.
//
// stdout carries only protocol frames (the MCP stdio contract); all
// logging goes to stderr.

import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';

// Dual-era protocol support (spec 2026-07-28 "Versioning and
// Compatibility"): modern revisions are stateless — every request carries
// its protocol version in _meta, servers answer server/discover, results
// carry resultType and serverInfo metadata — while legacy revisions open
// with an initialize handshake. This server serves both concurrently:
// a request with modern _meta gets modern semantics, an initialize gets
// the legacy session. Legacy responses stay byte-shaped as before —
// modern decoration is only added where modern clients look for it.
const MODERN_VERSIONS = ['2026-07-28'];
const LEGACY_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];
const META_VERSION = 'io.modelcontextprotocol/protocolVersion';
const META_SERVER_INFO = 'io.modelcontextprotocol/serverInfo';
// tools/list is static for the life of the process (packs mount once), so
// modern CacheableResult hints let clients cache it for an hour; private
// because mounted packs are the user's own configuration.
const CACHE_HINTS = { ttlMs: 3600000, cacheScope: 'private' };
// Result text is chunk-sized by construction (compile targets ~256
// tokens); k stays small so one tool result cannot flood a context window.
const MAX_K = 20;

function toolDefinitions(packs) {
  const packNames = [...packs.keys()];
  const packDescription = packs.size === 1
    ? `Only one pack is mounted (${packNames[0]}); the parameter is optional.`
    : `Omit to search every mounted pack (${packNames.join(', ')}); results stay grouped per pack.`;
  return [
    {
      name: 'search',
      description: 'Search the mounted .pikelet knowledge packs. Returns the most relevant '
        + 'chunks with full provenance (pack, immutable pack identity, title, heading path, '
        + 'source) plus a calibrated matchQuality per pack: "strong" means the pack answers '
        + 'this, "weak" means treat results with caution, "none" means this pack does not '
        + 'contain the answer — do not force a citation from a pack that says none.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Natural-language question or keyword query.' },
          pack: { type: 'string', description: `Pack to search. ${packDescription}` },
          k: { type: 'number', description: `Results per pack, 1-${MAX_K} (default 5).` },
        },
        required: ['query'],
      },
    },
    {
      name: 'list_packs',
      description: 'List the mounted knowledge packs: name, immutable identity (sha256 of the '
        + 'pack manifest — cite it to pin the exact knowledge state an answer used), record '
        + 'count, encoder, and sample queries the pack was built to answer.',
      inputSchema: { type: 'object', properties: {} },
    },
    {
      name: 'verify_pack',
      description: 'Run the tests a pack carries inside itself: golden queries (each verified '
        + 'at build time to retrieve its source) and abstention probes (queries the pack must '
        + 'answer or must refuse). Reports pass/fail per test plus the pack\'s encoder and '
        + 'integrity verification state. This is self-verification: it proves the artifact is '
        + 'intact and behaves as it did when built — not that its content is true or its '
        + 'publisher trustworthy.',
      inputSchema: {
        type: 'object',
        properties: {
          pack: { type: 'string', description: 'Pack to verify (optional when only one pack is mounted).' },
          limit: { type: 'number', description: 'Max tests to run per category (default 24).' },
        },
      },
    },
    {
      name: 'get_record',
      description: 'Fetch one full record from a pack by the id a search result reported — '
        + 'the complete chunk text plus its provenance, integrity-verified from the pack.',
      inputSchema: {
        type: 'object',
        properties: {
          pack: { type: 'string', description: 'Pack name (optional when only one pack is mounted).' },
          id: { type: 'number', description: 'Record id from a search result.' },
        },
        required: ['id'],
      },
    },
  ];
}

function provenanced(packName, identity, result) {
  return {
    pack: packName,
    packIdentity: identity,
    id: result.id,
    title: result.title ?? null,
    headingPath: Array.isArray(result.headingPath) ? result.headingPath : [],
    anchor: result.anchor ?? null,
    source: result.url ?? result.sourcePath ?? null,
    distance: result.distance,
    text: result.text ?? result.preview ?? null,
  };
}

function resolvePack(packs, requested, toolName) {
  if (requested !== undefined) {
    if (typeof requested !== 'string' || !packs.has(requested)) {
      throw new Error(`${toolName}: unknown pack ${JSON.stringify(requested)} — mounted packs: ${[...packs.keys()].join(', ')}`);
    }
    return [requested];
  }
  return [...packs.keys()];
}

async function callSearch(packs, args) {
  if (typeof args?.query !== 'string' || !args.query.trim()) {
    throw new Error('search: query must be a non-empty string');
  }
  let k = 5;
  if (args.k !== undefined) {
    if (!Number.isSafeInteger(args.k) || args.k < 1 || args.k > MAX_K) {
      throw new Error(`search: k must be an integer between 1 and ${MAX_K}`);
    }
    k = args.k;
  }
  const names = resolvePack(packs, args.pack, 'search');
  const sections = [];
  for (const name of names) {
    const mounted = packs.get(name);
    const out = await mounted.search.query(args.query, { k });
    sections.push({
      pack: name,
      packIdentity: mounted.identity,
      matchQuality: out.matchQuality,
      ...(out.confidence !== undefined ? { confidence: out.confidence } : {}),
      results: out.results.map((r) => provenanced(name, mounted.identity, r)),
    });
  }
  const answered = sections.filter((s) => s.results.length > 0);
  return {
    query: args.query,
    packsSearched: names,
    ...(answered.length === 0 ? {
      note: 'No mounted pack contains an answer to this query. Say so rather than guessing.',
    } : {}),
    sections,
  };
}

async function callGetRecord(packs, args) {
  if (!Number.isSafeInteger(args?.id) || args.id < 0) {
    throw new Error('get_record: id must be a non-negative integer');
  }
  const names = resolvePack(packs, args.pack, 'get_record');
  if (names.length !== 1) {
    throw new Error('get_record: pack is required when more than one pack is mounted');
  }
  const mounted = packs.get(names[0]);
  const record = await mounted.search.record(args.id);
  const shaped = provenanced(names[0], mounted.identity, { ...record, id: args.id });
  delete shaped.distance;
  return shaped;
}

async function callVerifyPack(packs, args) {
  const names = resolvePack(packs, args?.pack, 'verify_pack');
  if (names.length !== 1) {
    throw new Error('verify_pack: pack is required when more than one pack is mounted');
  }
  let limit = 24;
  if (args?.limit !== undefined) {
    if (!Number.isSafeInteger(args.limit) || args.limit < 1 || args.limit > 200) {
      throw new Error('verify_pack: limit must be an integer between 1 and 200');
    }
    limit = args.limit;
  }
  const mounted = packs.get(names[0]);
  const evaluation = await mounted.search.evaluation().catch(() => null);
  const goldens = Array.isArray(evaluation?.goldenQueries) ? evaluation.goldenQueries.slice(0, limit) : [];
  const probes = Array.isArray(evaluation?.abstentionProbes) ? evaluation.abstentionProbes.slice(0, limit) : [];
  const goldenResults = [];
  for (const golden of goldens) {
    if (typeof golden?.text !== 'string') continue;
    const out = await mounted.search.query(golden.text, { k: 10 });
    const pass = golden.expectId !== undefined
      ? out.results.some((r) => r.id === golden.expectId)
      : golden.expectTitle !== undefined
        ? out.results.some((r) => (r.title || '').trim() === golden.expectTitle)
        : golden.expectedTopId !== undefined
          ? out.results[0]?.id === golden.expectedTopId
          : out.results.length > 0;
    goldenResults.push({ text: golden.text, pass, ...(pass ? {} : { got: out.results.slice(0, 3).map((r) => ({ id: r.id, title: r.title })) }) });
  }
  const probeResults = [];
  for (const probe of probes) {
    if (typeof probe?.text !== 'string') continue;
    const out = await mounted.search.query(probe.text, { k: 5 });
    const answered = out.matchQuality !== 'none';
    const pass = probe.expect === 'abstain' ? !answered : answered;
    probeResults.push({ text: probe.text, expect: probe.expect, got: out.matchQuality, pass });
  }
  const goldenPassed = goldenResults.filter((g) => g.pass).length;
  const probesPassed = probeResults.filter((g) => g.pass).length;
  // Read info after the queries: a deferred kind-3 encoder verifies its
  // test vectors on first load, so encoderVerified is only meaningful now.
  const info = mounted.search.info();
  return {
    pack: names[0],
    packIdentity: mounted.identity,
    encoderVerified: info.encoderVerified,
    corpusIntegrity: info.corpusIntegrity,
    indexRowIntegrity: info.indexRowIntegrity,
    goldenQueries: { total: goldenResults.length, passed: goldenPassed, results: goldenResults },
    abstentionProbes: { total: probeResults.length, passed: probesPassed, results: probeResults },
    ...(goldenResults.length === 0 && probeResults.length === 0
      ? { note: 'This pack embeds no runnable tests (older build, or calibration was skipped); encoder and integrity state above still apply.' }
      : {}),
    verdict: goldenPassed === goldenResults.length && probesPassed === probeResults.length
      ? (goldenResults.length + probeResults.length > 0 ? 'pass' : 'no-tests')
      : 'fail',
  };
}

function callListPacks(packs) {
  return {
    packs: [...packs.entries()].map(([name, mounted]) => {
      const info = mounted.search.info();
      return {
        name,
        identity: info.identity,
        file: mounted.file,
        records: info.records,
        encoder: info.encoder?.model ?? info.encoder?.kind ?? null,
        license: info.license,
        remote: mounted.remote === true,
        hybridLexical: info.lexical !== null,
        sampleQueries: info.sampleQueries.slice(0, 5),
      };
    }),
  };
}

async function mountPack(packs, spec, { openPikeletFile, httpRangeSource, log }) {
  const isUrl = /^https?:\/\//i.test(spec.location);
  // URL packs are the format's native habitat: range-read off dumb HTTP,
  // nothing downloaded but the resident slice and per-query ranges. The
  // reader's own bounded full-download fallback covers hosts that ignore
  // Range (`pikelet doctor <url>` certifies a host).
  // A pinned identity is enforced by the reader itself, one header read
  // in — a mismatched pack is refused before any of it is opened.
  const openOptions = spec.identity ? { expectedIdentity: spec.identity } : {};
  let search;
  try {
    search = isUrl
      ? await (async () => {
        const source = httpRangeSource(spec.location);
        await source.init();
        return openPikeletFile(source, openOptions);
      })()
      : await openPikeletFile(path.resolve(spec.location), openOptions);
  } catch (err) {
    if (/identity mismatch/.test(String(err?.message))) {
      throw new Error(`${spec.location}: ${err.message} — the pack at this location is not the `
        + 'knowledge state the mount pinned; refusing to serve it.');
    }
    throw err;
  }
  const info = search.info();
  if (info.encoder && info.encoder.kind === 'external-transformers-v1') {
    await search.close();
    throw new Error(`${spec.location}: kind-2 packs need a host encoder and cannot be mounted self-contained; `
      + 'compile packs with the inline encoder (the compile default) to serve them over MCP');
  }
  let name = spec.name || info.name
    || path.basename(isUrl ? new URL(spec.location).pathname : spec.location).replace(/\.pikelet$/, '');
  // Names address packs in every tool call; collisions get a stable
  // numeric suffix rather than silently shadowing an earlier mount.
  if (packs.has(name)) {
    let n = 2;
    while (packs.has(`${name}-${n}`)) n += 1;
    name = `${name}-${n}`;
  }
  packs.set(name, { search, identity: info.identity, file: isUrl ? spec.location : path.resolve(spec.location), remote: isUrl });
  log(`mounted ${name} (${info.records} records, identity ${info.identity.slice(0, 12)}…, ${spec.identity ? 'identity-pinned' : 'unpinned'}) from ${spec.location}`);
}


/**
 * Write (or merge) an MCP client config entry that launches this server —
 * turns the documentation page into a verb. Supported clients:
 * 'claude-code' (./.mcp.json in the current project) and 'claude-desktop'
 * (the per-platform Claude Desktop config file). Existing config is
 * preserved; an existing server of the same name is only replaced with
 * force.
 */
export async function installMcpConfig({ packPaths, shelf, client = 'claude-code', serverName = 'knowledge-packs', force = false, version = undefined, homedir = undefined, cwd = undefined }) {
  const os = await import('node:os');
  const home = homedir || os.homedir();
  const base = cwd || process.cwd();
  if ((!Array.isArray(packPaths) || packPaths.length === 0) && !shelf) {
    throw new Error('mcp install requires at least one --pack <file-or-url> or a --shelf');
  }
  let configPath;
  let hint;
  if (client === 'claude-code') {
    configPath = path.join(base, '.mcp.json');
    hint = 'Claude Code picks it up on the next session in this project.';
  } else if (client === 'claude-desktop') {
    const platform = process.platform;
    configPath = platform === 'darwin'
      ? path.join(home, 'Library', 'Application Support', 'Claude', 'claude_desktop_config.json')
      : platform === 'win32'
        ? path.join(process.env.APPDATA || path.join(home, 'AppData', 'Roaming'), 'Claude', 'claude_desktop_config.json')
        : path.join(home, '.config', 'Claude', 'claude_desktop_config.json');
    hint = 'Restart Claude Desktop to load it.';
  } else {
    throw new Error(`mcp install: unknown --client ${client} (use claude-code or claude-desktop)`);
  }
  // The runtime is version-pinned: the packs are content-addressed and
  // immutable, and a reader that silently floats to whatever npm serves
  // next month would undercut exactly that reproducibility.
  const args = ['-y', version ? `pikelet@${version}` : 'pikelet', 'mcp'];
  for (const raw of packPaths || []) {
    // Local paths are pinned absolute so the config works from any cwd;
    // URLs (and #identity pins) pass through untouched.
    const hash = /^(.*)#([0-9a-f]{64})$/i.exec(raw);
    const location = hash ? hash[1] : raw;
    const suffix = hash ? `#${hash[2].toLowerCase()}` : '';
    args.push('--pack', /^https?:\/\//i.test(location) ? `${location}${suffix}` : `${path.resolve(base, location)}${suffix}`);
  }
  if (shelf) args.push('--shelf', /^https?:\/\//i.test(shelf) ? shelf : path.resolve(base, shelf));

  let existing = {};
  try {
    existing = JSON.parse(await fs.readFile(configPath, 'utf8'));
    if (!existing || typeof existing !== 'object' || Array.isArray(existing)) {
      throw new Error(`${configPath} exists but is not a JSON object; refusing to overwrite it`);
    }
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
  const servers = existing.mcpServers && typeof existing.mcpServers === 'object' ? existing.mcpServers : {};
  if (servers[serverName] && !force) {
    throw new Error(`${configPath} already has an mcpServers entry named ${JSON.stringify(serverName)}; rerun with --force to replace it or choose --server-name`);
  }
  const merged = { ...existing, mcpServers: { ...servers, [serverName]: { command: 'npx', args } } };
  await fs.mkdir(path.dirname(configPath), { recursive: true });
  await fs.writeFile(configPath, `${JSON.stringify(merged, null, 2)}\n`, 'utf8');
  return { configPath, serverName, hint };
}

/**
 * A shelf is a registry that is also just a file: a static JSON listing of
 * packs — { packs: [{ name?, url|path, identity?, description? }] }.
 * Relative paths resolve against the shelf's own location. Entries with an
 * identity are pinned at mount.
 */
export async function loadShelf(location) {
  const isUrl = /^https?:\/\//i.test(location);
  let text;
  if (isUrl) {
    const response = await fetch(location, { redirect: 'follow' });
    if (!response.ok) throw new Error(`shelf ${location}: HTTP ${response.status}`);
    text = await response.text();
  } else {
    text = await fs.readFile(path.resolve(location), 'utf8');
  }
  let shelf;
  try { shelf = JSON.parse(text); } catch (err) {
    throw new Error(`shelf ${location} is not valid JSON: ${err.message}`);
  }
  if (!Array.isArray(shelf?.packs) || shelf.packs.length === 0) {
    throw new Error(`shelf ${location} must be a JSON object with a non-empty "packs" array`);
  }
  return shelf.packs.map((entry, i) => {
    const target = entry.url ?? entry.path;
    if (typeof target !== 'string' || !target) throw new Error(`shelf ${location} entry ${i} needs a url or path`);
    if (entry.identity !== undefined && !/^[0-9a-f]{64}$/i.test(entry.identity)) {
      throw new Error(`shelf ${location} entry ${i}: identity must be a sha256 hex string`);
    }
    let resolved;
    if (/^https?:\/\//i.test(target)) resolved = target;
    else if (isUrl) resolved = new URL(target, location).href;
    else resolved = path.resolve(path.dirname(path.resolve(location)), target);
    return {
      location: resolved,
      ...(entry.identity ? { identity: entry.identity.toLowerCase() } : {}),
      ...(typeof entry.name === 'string' && entry.name ? { name: entry.name } : {}),
    };
  });
}

/**
 * Mount packs and serve MCP on stdio until stdin closes. `openPack` is
 * injected (the CLI passes pikelet-wasm/complete's openPikeletFile) so
 * tests can stub it.
 */
export async function runMcpServer({ packPaths, openPikeletFile, httpRangeSource, serverVersion = '0.0.0', stdin = process.stdin, stdout = process.stdout, log = (line) => process.stderr.write(`${line}\n`) }) {
  if (!Array.isArray(packPaths) || packPaths.length === 0) {
    throw new Error('mcp requires at least one --pack <file.pikelet>');
  }
  const packs = new Map();
  const specs = [];
  for (const raw of packPaths) {
    if (raw && typeof raw === 'object') { specs.push(raw); continue; }
    // `location#<sha256>` pins the pack's manifest identity at the mount:
    // the shape a shelf entry, a README, or a lockfile can carry.
    const hash = /^(.*)#([0-9a-f]{64})$/i.exec(raw);
    specs.push(hash ? { location: hash[1], identity: hash[2].toLowerCase() } : { location: raw });
  }
  for (const spec of specs) {
    await mountPack(packs, spec, { openPikeletFile, httpRangeSource, log });
  }
  // Warm every pack in the background so the first tool call pays for
  // retrieval, not for staging: the query forces the deferred encoder,
  // kernel init, and (URL mounts) the first rerank round trips. Failures
  // surface on real queries, not here.
  for (const [, mounted] of packs) {
    const probe = mounted.search.info().sampleQueries[0] || 'what is this about';
    mounted.warmup = mounted.search.query(probe, { k: 1 }).catch(() => {});
  }

  const serverInfo = { name: 'pikelet-knowledge-packs', version: serverVersion };
  const supportedVersions = [...MODERN_VERSIONS, ...LEGACY_VERSIONS];
  const send = (message) => stdout.write(`${JSON.stringify(message)}\n`);
  const replyError = (id, code, message, data) => send({ jsonrpc: '2.0', id, error: { code, message, ...(data ? { data } : {}) } });

  const rl = readline.createInterface({ input: stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let message;
    try { message = JSON.parse(trimmed); } catch {
      replyError(null, -32700, 'parse error');
      continue;
    }
    const { id, method, params } = message;
    // Notifications (no id) get no response, per JSON-RPC.
    const isRequest = id !== undefined && id !== null;
    // Modern requests declare their version per request; server/discover
    // is modern by definition (it is the stdio compatibility probe).
    const metaVersion = typeof params?._meta?.[META_VERSION] === 'string' ? params._meta[META_VERSION] : null;
    const modern = metaVersion !== null || method === 'server/discover';
    // Modern results carry resultType and server identity; legacy results
    // keep their exact pre-modern shape.
    const reply = (replyId, result) => send({
      jsonrpc: '2.0',
      id: replyId,
      result: modern
        ? { resultType: 'complete', ...result, _meta: { [META_SERVER_INFO]: serverInfo, ...(result._meta || {}) } }
        : result,
    });
    try {
      if (metaVersion !== null && !supportedVersions.includes(metaVersion)) {
        if (isRequest) {
          replyError(id, -32022, 'Unsupported protocol version', { supported: supportedVersions, requested: metaVersion });
        }
      } else if (method === 'server/discover') {
        reply(id, {
          supportedVersions,
          capabilities: { tools: {} },
          instructions: 'This server mounts .pikelet knowledge packs. Use search to retrieve '
            + 'provenanced passages; matchQuality "none" means the pack does not contain the '
            + 'answer — say so rather than guessing. verify_pack runs the tests a pack carries '
            + 'inside itself; list_packs reports identities for citation pinning.',
          ...CACHE_HINTS,
        });
      } else if (method === 'initialize') {
        const requested = params?.protocolVersion;
        reply(id, {
          protocolVersion: LEGACY_VERSIONS.includes(requested) ? requested : LEGACY_VERSIONS[0],
          capabilities: { tools: {} },
          serverInfo,
        });
      } else if (method === 'ping') {
        // Removed in 2026-07-28 but kept for legacy clients; a modern
        // caller who sends it anyway gets a well-formed empty result.
        if (isRequest) reply(id, {});
      } else if (method === 'tools/list') {
        // Deterministic order (mount order) per the modern caching rules.
        reply(id, { tools: toolDefinitions(packs), ...(modern ? CACHE_HINTS : {}) });
      } else if (method === 'tools/call') {
        const toolName = params?.name;
        const args = params?.arguments ?? {};
        let result;
        if (toolName === 'search') result = await callSearch(packs, args);
        else if (toolName === 'list_packs') result = callListPacks(packs);
        else if (toolName === 'get_record') result = await callGetRecord(packs, args);
        else if (toolName === 'verify_pack') result = await callVerifyPack(packs, args);
        else throw new Error(`unknown tool ${JSON.stringify(toolName)}`);
        reply(id, { content: [{ type: 'text', text: JSON.stringify(result, null, 1) }] });
      } else if (typeof method === 'string' && method.startsWith('notifications/')) {
        // initialized, cancelled, ... — nothing to do.
      } else if (isRequest) {
        replyError(id, -32601, `method not found: ${method}`);
      }
    } catch (err) {
      const text = err && err.message ? err.message : String(err);
      if (method === 'tools/call' && isRequest) {
        // Tool-level failures are results with isError, not protocol
        // errors — the model sees them and can correct the call.
        reply(id, { content: [{ type: 'text', text }], isError: true });
      } else if (isRequest) {
        replyError(id, -32603, text);
      } else {
        log(`error handling ${method}: ${text}`);
      }
    }
  }

  for (const mounted of packs.values()) await mounted.search.close();
}
