// MCP protocol conformance for the knowledge-pack server, both eras:
// modern stateless (2026-07-28 — server/discover, per-request _meta
// version, resultType, CacheableResult hints, UnsupportedProtocolVersion)
// and legacy handshake (2025-11-25 and earlier — initialize/initialized).
// Runs against a stubbed pack (runMcpServer takes an injected
// openPikeletFile), so it is protocol-only and fast enough for every CI
// run — no compile, no encoder, no network.

import { PassThrough } from 'node:stream';
import { runMcpServer } from '../packages/pikelet/src/mcp.mjs';

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
        // Mirrors the real reader's showAbstained contract (complete/index.mjs):
        // a 'none' verdict withholds results unless showAbstained is true, and
        // showAbstained never changes matchQuality or confidence.
        query: async (queryText, opts = {}) => {
            if (queryText === 'abstained query') {
                const withheld = [{ id: 1, title: 'Withheld', headingPath: [], anchor: null, sourcePath: 'withheld.md', text: 'withheld text', distance: 0.3 }];
                return {
                    matchQuality: 'none',
                    confidence: 0.1,
                    results: opts.showAbstained ? withheld : [],
                };
            }
            // Mirrors the real reader's unscored contract (complete/index.mjs):
            // no fitted calibrator means no confidence field at all, and
            // results always ship — there is nothing to withhold against.
            if (queryText === 'unscored query') {
                return {
                    matchQuality: 'unscored',
                    confidence: undefined,
                    results: [{ id: 2, title: 'Unscored', headingPath: [], anchor: null, sourcePath: 'unscored.md', text: 'unscored text', distance: 0.4 }],
                };
            }
            return {
                matchQuality: 'strong',
                confidence: 0.9,
                results: [{ id: 0, title: 'Stub', headingPath: [], anchor: null, sourcePath: 'stub.md', text: 'stub text', distance: 0.1 }],
            };
        },
        record: async (id) => ({ title: 'Stub', text: 'stub text', sourcePath: 'stub.md', id }),
        evaluation: async () => null,
        close: async () => {},
    };
}

async function withServer(fn, { packPaths = ['stub.pikelet'], openPikeletFile = async () => stubPack(), onExit = null, toolsListGraceMs } = {}) {
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
        packPaths,
        openPikeletFile,
        httpRangeSource: () => { throw new Error('no network in conformance'); },
        serverVersion: '0.0.0-test',
        ...(toolsListGraceMs === undefined ? {} : { toolsListGraceMs }),
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
        if (onExit) onExit(await done.then(() => null, (err) => err));
        else await done;
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

console.log('\nsearch tool: showAbstained');
await withServer(async ({ send, waitFor }) => {
    const call = async (id, args) => {
        send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name: 'search', arguments: args } });
        const result = (await waitFor(id)).result;
        return JSON.parse(result.content[0].text);
    };

    const inputSchemaList = () => new Promise((resolve) => {
        send({ jsonrpc: '2.0', id: 100, method: 'tools/list' });
        waitFor(100).then((r) => resolve(r.result.tools.find((t) => t.name === 'search')));
    });
    const searchTool = await inputSchemaList();
    check('search inputSchema declares showAbstained as boolean',
        searchTool.inputSchema.properties.showAbstained?.type === 'boolean');

    const shown = await call(1, { query: 'abstained query' });
    check('a none verdict shows results by default (showAbstained defaults true)',
        shown.sections[0].matchQuality === 'none' && shown.sections[0].results.length === 1
        && shown.sections[0].results[0].id === 1);
    check('default-shown response carries a caveat note, not a withholding one',
        typeof shown.note === 'string' && shown.note.includes('weigh'));

    const withheld = await call(2, { query: 'abstained query', showAbstained: false });
    check('showAbstained: false withholds results under none',
        withheld.sections[0].matchQuality === 'none' && withheld.sections[0].results.length === 0);
    check('showAbstained never changes matchQuality or confidence',
        shown.sections[0].matchQuality === withheld.sections[0].matchQuality
        && shown.sections[0].confidence === withheld.sections[0].confidence);
    check('withheld response points back at showAbstained: false to recover the note',
        typeof withheld.note === 'string' && withheld.note.includes('showAbstained: false'));

    const answered = await call(3, { query: 'what is a stub' });
    check('a strong verdict carries no note', answered.note === undefined && answered.sections[0].matchQuality === 'strong');

    // A rejected call is a tool error (isError, plain text), not JSON — call
    // its own protocol round trip directly instead of the JSON-parsing helper.
    send({ jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'search', arguments: { query: 'q', showAbstained: 'yes' } } });
    const bad = (await waitFor(4)).result;
    check('non-boolean showAbstained is rejected as a tool error',
        bad.isError === true && bad.content[0].text.includes('showAbstained'));

    const unscored = await call(5, { query: 'unscored query' });
    check('an unscored section always ships its results',
        unscored.sections[0].matchQuality === 'unscored' && unscored.sections[0].results.length === 1);
    check('an unscored section carries an unscoredWarning naming the pack',
        typeof unscored.unscoredWarning === 'string' && unscored.unscoredWarning.includes('stub-pack')
        && unscored.unscoredWarning.includes('no calibrated abstention'));
    check('unscored does not also trigger the none-verdict note',
        unscored.note === undefined);

    const unscoredStrict = await call(6, { query: 'unscored query', showAbstained: false });
    check('showAbstained: false has no effect on an unscored section',
        unscoredStrict.sections[0].matchQuality === 'unscored' && unscoredStrict.sections[0].results.length === 1
        && typeof unscoredStrict.unscoredWarning === 'string');
});


console.log('mount isolation');
{
    // One pack that cannot mount (dead shelf host, refused pin) must not take
    // down the packs listed after it: the survivors serve, list_packs names
    // the missing one, and the process still exits non-zero.
    let exitErr;
    await withServer(async ({ send, waitFor }) => {
        const meta = { [META_VERSION]: '2026-07-28' };
        send({ jsonrpc: '2.0', id: 1, method: 'tools/list', params: { _meta: meta } });
        const list = await waitFor(1);
        check('tools/list answers despite an earlier failed mount', Array.isArray(list.result?.tools), JSON.stringify(list.error));
        send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { _meta: meta, name: 'search', arguments: { query: 'what is a stub' } } });
        const call = (await waitFor(2)).result;
        check('the pack after the failed one serves searches', call && !call.isError && call.content[0].text.includes('stub text'),
            call?.content?.[0]?.text);
        send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { _meta: meta, name: 'list_packs', arguments: {} } });
        const listed = JSON.parse((await waitFor(3)).result.content[0].text);
        check('list_packs names the unavailable pack and why',
            listed.packs.length === 1 && listed.unavailable?.length === 1
            && listed.unavailable[0].location === 'dead.pikelet' && /host unreachable/.test(listed.unavailable[0].error),
            JSON.stringify(listed.unavailable));
    }, {
        packPaths: ['dead.pikelet', 'stub.pikelet'],
        openPikeletFile: async (target) => {
            if (String(target).endsWith('dead.pikelet')) throw new Error('host unreachable');
            return stubPack();
        },
        onExit: (err) => { exitErr = err; },
    });
    check('a partial mount failure still exits non-zero', exitErr?.exitCode === 1 && /dead\.pikelet/.test(exitErr.message),
        exitErr?.message);

    // Nothing mounted: every tool call reports why, rather than an empty server.
    let allExit;
    await withServer(async ({ send, waitFor }) => {
        const meta = { [META_VERSION]: '2026-07-28' };
        send({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { _meta: meta, name: 'search', arguments: { query: 'x' } } });
        const call = (await waitFor(1)).result;
        check('with no pack mounted, a tool call reports the mount failure',
            call?.isError === true && /no pack mounted.*host unreachable/.test(call.content[0].text), call?.content?.[0]?.text);
    }, {
        packPaths: ['dead.pikelet'],
        openPikeletFile: async () => { throw new Error('host unreachable'); },
        onExit: (err) => { allExit = err; },
    });
    check('a total mount failure exits non-zero', allExit?.exitCode === 1);
}

console.log('slow mount');
{
    // A URL mount on a slow host can take tens of seconds. tools/list must not
    // wait it out: a client that times out on it starts with no tools at all.
    // It answers provisionally after the grace period, and list_changed tells
    // the client to fetch the real list once the mounts settle.
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    await withServer(async ({ send, waitFor, replies }) => {
        const meta = { [META_VERSION]: '2026-07-28' };
        send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 't', version: '0' } } });
        const init = (await waitFor(1)).result;
        check('initialize advertises tools.listChanged', init.capabilities?.tools?.listChanged === true);
        send({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: { _meta: meta } });
        const provisional = (await waitFor(2)).result;
        const names = (provisional?.tools || []).map((t) => t.name).join(',');
        check('tools/list answers while a mount is still in flight', names === 'search,list_packs,verify_pack,get_record', names);
        check('the provisional list carries no cache hints', provisional.ttlMs === undefined && provisional.cacheScope === undefined);
        check('the provisional list says packs are still mounting',
            provisional.tools[0].inputSchema.properties.pack.description.includes('still mounting'));
        check('no list_changed before the mounts settle',
            !replies.some((r) => r.method === 'notifications/tools/list_changed'));
        // A tool call made now waits for the mount rather than failing.
        send({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { _meta: meta, name: 'search', arguments: { query: 'what is a stub' } } });
        release();
        const call = (await waitFor(3)).result;
        check('a call made during the mount waits for it and serves', call && !call.isError && call.content[0].text.includes('stub text'),
            call?.content?.[0]?.text);
        check('list_changed follows once the mounts settle',
            replies.some((r) => r.method === 'notifications/tools/list_changed' && r.id === undefined));
        send({ jsonrpc: '2.0', id: 4, method: 'tools/list', params: { _meta: meta } });
        const full = (await waitFor(4)).result;
        check('the list after settling names the pack and is cacheable',
            full.tools[0].inputSchema.properties.pack.description.includes('stub-pack') && Number.isFinite(full.ttlMs));
    }, {
        toolsListGraceMs: 20,
        openPikeletFile: async () => { await gate; return stubPack(); },
    });

    // A mount that settles inside the grace period gets the exact list and no
    // notification: the common local-pack case is unchanged.
    await withServer(async ({ send, waitFor, replies }) => {
        send({ jsonrpc: '2.0', id: 1, method: 'tools/list' });
        const list = (await waitFor(1)).result;
        check('a mount inside the grace period gets the named list',
            list.tools[0].inputSchema.properties.pack.description.includes('stub-pack'));
        send({ jsonrpc: '2.0', id: 2, method: 'ping' });
        await waitFor(2);
        check('no list_changed when the first list was already exact',
            !replies.some((r) => r.method === 'notifications/tools/list_changed'));
    }, {
        toolsListGraceMs: 5000,
        openPikeletFile: async () => { await new Promise((r) => setTimeout(r, 30)); return stubPack(); },
    });
}

console.log(`\nMCP protocol conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
