// MCP protocol conformance for the knowledge-pack server, both eras:
// modern stateless (2026-07-28 — server/discover, per-request _meta
// version, resultType, CacheableResult hints, UnsupportedProtocolVersion)
// and legacy handshake (2025-11-25 and earlier — initialize/initialized).
// Runs against a stubbed pack (runMcpServer takes an injected
// openPikeletFile), so it is protocol-only and fast enough for every CI
// run — no compile, no encoder, no network.

import { PassThrough } from 'node:stream';
import { runMcpServer } from '../pikelet/src/mcp.mjs';

let passed = 0;
let failed = 0;
const check = (label, cond, detail = '') => {
    if (cond) { passed++; console.log(`  ok: ${label}`); }
    else { failed++; console.log(`  FAIL: ${label}${detail ? ` - ${detail}` : ''}`); }
};

const META_VERSION = 'io.modelcontextprotocol/protocolVersion';
const META_SERVER_INFO = 'io.modelcontextprotocol/serverInfo';

function stubPack() {
    return {
        info: () => ({
            identity: 'a'.repeat(64),
            name: 'stub-pack',
            license: 'CC0-1.0',
            records: 2,
            encoder: { kind: 'inline-transformer-v1', model: 'stub' },
            encoderVerified: true,
            corpusIntegrity: 'per-record-sha256',
            indexRowIntegrity: 'per-row-sha256',
            lexical: null,
            sampleQueries: ['what is a stub'],
        }),
        query: async () => ({
            matchQuality: 'strong',
            confidence: 0.9,
            results: [{ id: 0, title: 'Stub', headingPath: [], anchor: null, sourcePath: 'stub.md', text: 'stub text', distance: 0.1 }],
        }),
        record: async (id) => ({ title: 'Stub', text: 'stub text', sourcePath: 'stub.md', id }),
        evaluation: async () => null,
        close: async () => {},
    };
}

async function withServer(fn) {
    const stdin = new PassThrough();
    const stdout = new PassThrough();
    const replies = [];
    let buffer = '';
    stdout.on('data', (d) => {
        buffer += d;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
            replies.push(JSON.parse(buffer.slice(0, idx)));
            buffer = buffer.slice(idx + 1);
        }
    });
    const done = runMcpServer({
        packPaths: ['stub.pikelet'],
        openPikeletFile: async () => stubPack(),
        httpRangeSource: () => { throw new Error('no network in conformance'); },
        serverVersion: '0.0.0-test',
        stdin,
        stdout,
        log: () => {},
    });
    const send = (m) => stdin.write(`${JSON.stringify(m)}\n`);
    const waitFor = (id) => new Promise((resolve, reject) => {
        const deadline = Date.now() + 10000;
        const timer = setInterval(() => {
            const hit = replies.find((r) => r.id === id);
            if (hit) { clearInterval(timer); resolve(hit); }
            else if (Date.now() > deadline) { clearInterval(timer); reject(new Error(`no reply for ${id}`)); }
        }, 5);
    });
    try {
        await fn({ send, waitFor, replies });
    } finally {
        stdin.end();
        await done;
    }
}

console.log('modern era (2026-07-28, stateless)');
await withServer(async ({ send, waitFor }) => {
    const meta = { [META_VERSION]: '2026-07-28' };

    // server/discover: the mandatory RPC and the stdio compatibility probe.
    send({ jsonrpc: '2.0', id: 1, method: 'server/discover', params: { _meta: meta } });
    const discover = (await waitFor(1)).result;
    check('server/discover advertises supported versions',
        Array.isArray(discover.supportedVersions) && discover.supportedVersions.includes('2026-07-28')
        && discover.supportedVersions.includes('2025-06-18'), JSON.stringify(discover.supportedVersions));
    check('discover result carries resultType complete', discover.resultType === 'complete');
    check('discover result carries capabilities and cache hints',
        discover.capabilities?.tools !== undefined && Number.isFinite(discover.ttlMs)
        && ['public', 'private'].includes(discover.cacheScope));
    check('discover result identifies the server in _meta',
        discover._meta?.[META_SERVER_INFO]?.name === 'pikelet-knowledge-packs');

    // No handshake: a tools/list with per-request _meta just works.
    send({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: { _meta: meta } });
    const list = (await waitFor(2)).result;
    check('stateless tools/list works without initialize',
        Array.isArray(list.tools) && list.tools.length === 4 && list.resultType === 'complete');
    check('modern tools/list is a CacheableResult',
        Number.isFinite(list.ttlMs) && ['public', 'private'].includes(list.cacheScope));
    send({ jsonrpc: '2.0', id: 3, method: 'tools/list', params: { _meta: meta } });
    const again = (await waitFor(3)).result;
    check('tools/list order is deterministic',
        JSON.stringify(list.tools.map((t) => t.name)) === JSON.stringify(again.tools.map((t) => t.name)));

    // Stateless tools/call, decorated as a modern result.
    send({ jsonrpc: '2.0', id: 4, method: 'tools/call', params: { _meta: meta, name: 'search', arguments: { query: 'what is a stub' } } });
    const call = (await waitFor(4)).result;
    check('modern tools/call returns content with resultType and serverInfo',
        call.resultType === 'complete' && Array.isArray(call.content)
        && call._meta?.[META_SERVER_INFO]?.name === 'pikelet-knowledge-packs');

    // Unknown version: UnsupportedProtocolVersionError (-32022) with data.
    send({ jsonrpc: '2.0', id: 5, method: 'tools/list', params: { _meta: { [META_VERSION]: '1900-01-01' } } });
    const bad = await waitFor(5);
    check('unknown version returns UnsupportedProtocolVersionError',
        bad.error?.code === -32022 && bad.error.data?.requested === '1900-01-01'
        && Array.isArray(bad.error.data?.supported), JSON.stringify(bad.error));

    // Modern tool errors are still isError results, not protocol errors.
    send({ jsonrpc: '2.0', id: 6, method: 'tools/call', params: { _meta: meta, name: 'search', arguments: {} } });
    const toolError = (await waitFor(6)).result;
    check('modern tool failure is an isError result', toolError.isError === true && toolError.resultType === 'complete');
});

console.log('\nlegacy era (initialize handshake)');
await withServer(async ({ send, waitFor }) => {
    send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '0' } } });
    const init = (await waitFor(1)).result;
    check('initialize echoes a supported requested version', init.protocolVersion === '2025-06-18');
    check('legacy initialize result keeps the legacy shape (no modern decoration)',
        init.resultType === undefined && init._meta === undefined && init.serverInfo?.name === 'pikelet-knowledge-packs',
        JSON.stringify(init));
    send({ jsonrpc: '2.0', method: 'notifications/initialized' });
    send({ jsonrpc: '2.0', id: 2, method: 'ping' });
    check('legacy ping answers an empty result', JSON.stringify((await waitFor(2)).result) === '{}');
    send({ jsonrpc: '2.0', id: 3, method: 'tools/list' });
    const list = (await waitFor(3)).result;
    check('legacy tools/list keeps the legacy shape',
        Array.isArray(list.tools) && list.tools.length === 4 && list.resultType === undefined && list.ttlMs === undefined);
    send({ jsonrpc: '2.0', id: 4, method: 'initialize', params: { protocolVersion: '2099-01-01' } });
    check('unknown legacy version negotiates down to the newest legacy revision',
        (await waitFor(4)).result.protocolVersion === '2025-11-25');
    send({ jsonrpc: '2.0', id: 5, method: 'no/such/method' });
    check('unknown method is -32601', (await waitFor(5)).error?.code === -32601);
});

console.log(`\nMCP protocol conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
