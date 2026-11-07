// The agent journal (docs/agent-journal.md): a private chain an agent may
// write notes to over MCP. Builds real journals (the inline encoder, no
// calibration) in a temp directory and drives the MCP server in-process.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import { openPikeletFile, httpRangeSource } from 'pikelet-wasm/complete';
import {
    initJournal, shapeNote, appendNote, openJournalChain, latestNotes, compactJournal,
    resolveJournalDir, defaultJournalRoot, NOTE_LIMITS,
} from '../packages/pikelet/src/journal.mjs';
import { runMcpServer, installMcpConfig } from '../packages/pikelet/src/mcp.mjs';

let passed = 0;
let failed = 0;
const check = (label, cond, detail = '') => {
    if (cond) { passed++; console.log(`  ok: ${label}`); }
    else { failed++; console.log(`  FAIL: ${label}${detail ? ` — ${detail}` : ''}`); }
};
const throwsLike = (label, fn, pattern) => {
    try { fn(); check(label, false, 'did not throw'); } catch (err) {
        check(label, pattern.test(String(err?.message)), String(err?.message).slice(0, 160));
    }
};
const rejectsLike = async (label, fn, pattern) => {
    try { await fn(); check(label, false, 'resolved'); } catch (err) {
        check(label, pattern.test(String(err?.message)), String(err?.message).slice(0, 160));
    }
};

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-journal-'));
const quiet = () => {};
const meta = (i = 0) => ({ created: new Date(Date.UTC(2026, 9, 13, 12, 0, i)).toISOString(), authoredBy: { client: 'test' }, session: 's1' });

console.log('journal: naming and note validation');
{
    check('a bare name lives under ~/.pikelet/journals', resolveJournalDir('proj') === path.join(defaultJournalRoot(), 'proj'));
    check('a path is taken as a path', resolveJournalDir('./x/y') === path.resolve('./x/y'));
    throwsLike('a name with odd characters is refused', () => resolveJournalDir('a b'), /must be letters/);

    const ok = shapeNote({ kind: 'decision', title: 'T', text: 'Body', tags: ['A', 'a', 'b'] }, meta());
    check('a note becomes one record whose text starts with its title', ok.text === 'T\nBody' && ok.title === 'T');
    check('tags are lower-cased and de-duplicated', JSON.stringify(ok.note.tags) === '["a","b"]');
    check('server-set fields come from the server', ok.note.created === meta().created && ok.note.authoredBy.client === 'test' && ok.note.session === 's1');
    throwsLike('an unknown kind is refused', () => shapeNote({ kind: 'gossip', title: 'T', text: 'B' }, meta()), /kind must be one of/);
    throwsLike('a multi-line title is refused', () => shapeNote({ kind: 'todo', title: 'a\nb', text: 'B' }, meta()), /one line/);
    throwsLike('an empty body is refused', () => shapeNote({ kind: 'todo', title: 'T', text: '  ' }, meta()), /text must be/);
    throwsLike('an oversized body is refused',
        () => shapeNote({ kind: 'todo', title: 'T', text: 'x'.repeat(NOTE_LIMITS.textBytes + 1) }, meta()), /larger than/);
    throwsLike('a note cannot set its own author or time',
        () => shapeNote({ kind: 'todo', title: 'T', text: 'B', authoredBy: 'someone else' }, meta()), /does not take authoredBy/);
    throwsLike('a malformed source citation is refused',
        () => shapeNote({ kind: 'todo', title: 'T', text: 'B', sources: [{ identity: 'abc', id: 1 }] }, meta()), /sources\[0\]/);
    throwsLike('an unknown repo field is refused',
        () => shapeNote({ kind: 'todo', title: 'T', text: 'B', repo: { secrets: 'x' } }, meta()), /repo does not take/);
}

console.log('journal: init, write, read, compact');
const dir = path.join(tmp, 'j');
{
    await initJournal({ nameOrPath: dir, log: quiet });
    const state = JSON.parse(fs.readFileSync(path.join(dir, 'journal.json'), 'utf8'));
    check('init writes a one-member journal', state.format === 'pikelet-journal-v1' && state.members.length === 1);
    await rejectsLike('init refuses to replace a journal without --force',
        () => initJournal({ nameOrPath: dir, log: quiet }), /already holds a journal/);

    const a = await appendNote(dir, shapeNote({ kind: 'decision', title: 'Reframe matchQuality as relevance',
        text: 'Fit the calibrator against off-topic queries; answerable none went from 12 of 44 to 0.', tags: ['calibration'],
        repo: { commits: ['c9927bf'] } }, meta(1)));
    const b = await appendNote(dir, shapeNote({ kind: 'dead-end', title: 'BGE-small gave no gain on small packs',
        text: 'Real kernel matched the simulation, but answers on the 68-record docs pack matched MiniLM.' }, meta(2)));
    check('notes get consecutive chain-global ids after the seed', a.id === 1 && b.id === 2, `${a.id}, ${b.id}`);

    const { chain } = await openJournalChain(dir);
    check('the journal is unscored by design', chain.info().calibrationStatus === 'none');
    const latest = await latestNotes(chain, { limit: 10 });
    check('latest lists notes newest first, without the seed', latest.map((r) => r.id).join(',') === '2,1');
    check('latest filters by kind', (await latestNotes(chain, { kind: 'decision' })).map((r) => r.id).join(',') === '1');
    check('latest filters by tag', (await latestNotes(chain, { tag: 'CALIBRATION' })).map((r) => r.id).join(',') === '1');
    const rec = await chain.record(1);
    check('a note record keeps its metadata', rec.note?.kind === 'decision' && rec.note.repo?.commits?.[0] === 'c9927bf'
        && rec.note.created === meta(1).created, JSON.stringify(rec.note));
    const enc = await chain.__head.passageEmbedder();
    const out = await chain.query(await enc.embedQuery('why did we stop using BGE'), 3, { text: 'why did we stop using BGE', showAbstained: true });
    check('search finds the note that answers', out.results[0]?.id === 2, out.results.map((r) => r.id).join(','));
    await chain.close();

    // Fill to the depth limit (base + 8 layers): the next write compacts first.
    for (let i = 0; i < 6; i++) {
        await appendNote(dir, shapeNote({ kind: 'measurement', title: `Run ${i}`, text: `Median ${100 + i} ms over twelve questions.` }, meta(3 + i)));
    }
    const before = JSON.parse(fs.readFileSync(path.join(dir, 'journal.json'), 'utf8')).members.length;
    const c = await appendNote(dir, shapeNote({ kind: 'summary', title: 'After compaction', text: 'Written after the journal folded.' }, meta(20)));
    const after = JSON.parse(fs.readFileSync(path.join(dir, 'journal.json'), 'utf8'));
    check('at depth 8 the next write compacts first', before === 9 && after.members.length === 2 && after.generation === 2,
        `${before} -> ${after.members.length} (generation ${after.generation})`);
    check('compaction removes the folded files', fs.readdirSync(dir).filter((f) => f.endsWith('.pikelet')).length === 2);
    const { chain: c2 } = await openJournalChain(dir);
    const all = await latestNotes(c2, { limit: 50 });
    check('every note survives compaction with its id', all.length === 9 && all[0].id === c.id && (await c2.record(1)).title === 'Reframe matchQuality as relevance');
    await c2.close();
    const folded = await compactJournal(dir, { log: quiet });
    check('compact folds the remaining layer', folded.members.length === 1);
}

console.log('journal: one writer at a time');
{
    const lock = path.join(dir, '.lock');
    fs.writeFileSync(lock, JSON.stringify({ pid: process.pid, at: Date.now() }));
    await rejectsLike('a live lock held past the wait refuses a second writer',
        () => appendNote(dir, shapeNote({ kind: 'todo', title: 'T', text: 'B' }, meta(30)), { lockWaitMs: 300 }), /is locked by process/);
    // A live lock released while a writer waits: the writer gets in.
    setTimeout(() => fs.rmSync(lock, { force: true }), 200);
    const waited = await appendNote(dir, shapeNote({ kind: 'todo', title: 'After a wait', text: 'The holder finished first.' }, meta(32)));
    check('a writer waits out a live lock', Number.isSafeInteger(waited.id) && !fs.existsSync(lock));
    fs.writeFileSync(lock, JSON.stringify({ pid: 2 ** 22 + 12345, at: Date.now() }));
    const w = await appendNote(dir, shapeNote({ kind: 'todo', title: 'After a stale lock', text: 'The dead writer\'s lock was taken over.' }, meta(31)));
    check('a lock left by a dead process is taken over', Number.isSafeInteger(w.id) && !fs.existsSync(lock));

    // A .lock that cannot be read is refused at once, not retried in a loop
    // that never reaches the deadline.
    fs.mkdirSync(lock);
    const t0 = Date.now();
    await rejectsLike('an unreadable lock is refused, not spun on',
        () => appendNote(dir, shapeNote({ kind: 'todo', title: 'T', text: 'B' }, meta(33)), { lockWaitMs: 60000 }), /cannot read the journal lock/);
    check('the refusal is immediate', Date.now() - t0 < 5000, `${Date.now() - t0} ms`);
    fs.rmdirSync(lock);

    // Stale locks are broken under .lock.break: a live breaker excludes the
    // others, and one left by a dead breaker is itself broken.
    const brk = path.join(dir, '.lock.break');
    fs.writeFileSync(lock, JSON.stringify({ pid: 2 ** 22 + 12345, at: Date.now() }));
    fs.writeFileSync(brk, JSON.stringify({ pid: process.pid, at: Date.now() }));
    await rejectsLike('a stale lock is not broken while another breaker holds .lock.break',
        () => appendNote(dir, shapeNote({ kind: 'todo', title: 'T', text: 'B' }, meta(34)), { lockWaitMs: 300 }), /is locked by process/);
    fs.writeFileSync(brk, JSON.stringify({ pid: 2 ** 22 + 12346, at: Date.now() }));
    const b2 = await appendNote(dir, shapeNote({ kind: 'todo', title: 'After a dead breaker', text: 'Both stale locks were broken.' }, meta(35)));
    check('a .lock.break left by a dead breaker is broken too', Number.isSafeInteger(b2.id) && !fs.existsSync(lock) && !fs.existsSync(brk));
}

console.log('journal: recovery and concurrent writers');
{
    // Writers at once: each waits its turn and lands in its own layer —
    // never two writers on one layer file.
    const results = await Promise.allSettled([0, 1, 2].map((i) => appendNote(dir,
        shapeNote({ kind: 'todo', title: `Concurrent ${i}`, text: `Written by writer ${i}.` }, meta(40 + i)))));
    const won = results.filter((r) => r.status === 'fulfilled').map((r) => r.value.id);
    const refused = results.filter((r) => r.status === 'rejected');
    check('concurrent writers all land, one after another',
        won.length === 3 && refused.length === 0,
        results.map((r) => r.status === 'fulfilled' ? r.value.id : r.reason?.message).join(' | '));
    const { chain } = await openJournalChain(dir);
    const titles = (await latestNotes(chain, { limit: 50 })).map((n) => n.title).filter((t) => t.startsWith('Concurrent'));
    check('every note a writer was told landed is in the chain', titles.length === won.length && new Set(won).size === won.length,
        `${titles.length} in chain, ${won.length} reported`);
    await chain.close();
    check('no lock is left behind', !fs.existsSync(path.join(dir, '.lock')));

    // A layer file left by a write that died before journal.json was updated.
    const state = JSON.parse(fs.readFileSync(path.join(dir, 'journal.json'), 'utf8'));
    const orphan = `${state.members[0].replace(/\.pikelet$/, '')}.${String(state.members.length).padStart(4, '0')}.pikelet`;
    fs.writeFileSync(path.join(dir, orphan), 'half-written layer');
    const after = await appendNote(dir, shapeNote({ kind: 'todo', title: 'After an interrupted write', text: 'The orphan layer was cleared.' }, meta(50)));
    check('an orphan layer file does not block the next write', Number.isSafeInteger(after.id));

    // Replacing a journal that has layers clears them, so its first write
    // does not collide with the old chain's files.
    const re = path.join(tmp, 'reinit');
    await initJournal({ nameOrPath: re, log: quiet });
    await appendNote(re, shapeNote({ kind: 'todo', title: 'Old', text: 'Before re-init.' }, meta(51)));
    await initJournal({ nameOrPath: re, force: true, log: quiet });
    const fresh = await appendNote(re, shapeNote({ kind: 'todo', title: 'New', text: 'After re-init.' }, meta(52)));
    check('init --force on a journal with layers leaves it writable', fresh.id === 1, `id ${fresh.id}`);
}

console.log('journal: private file modes');
if (process.platform === 'win32') {
    console.log('  skip: Windows has no Unix modes');
} else {
    // Under a permissive umask, init, append and compact still leave the
    // directory 0700 and every file 0600; a journal loosened from outside
    // is tightened again by the next write.
    const prevUmask = process.umask(0o022);
    try {
        const pdir = path.join(tmp, 'private');
        const modes = () => {
            const out = { dir: fs.statSync(pdir).mode & 0o777 };
            for (const f of fs.readdirSync(pdir)) out[f] = fs.statSync(path.join(pdir, f)).mode & 0o777;
            return out;
        };
        const allPrivate = (m) => m.dir === 0o700 && Object.entries(m).every(([k, v]) => k === 'dir' || v === 0o600);
        const show = (m) => JSON.stringify(Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v.toString(8)])));
        await initJournal({ nameOrPath: pdir, log: quiet });
        let m = modes();
        check('init leaves the directory 0700 and its files 0600', allPrivate(m), show(m));
        await appendNote(pdir, shapeNote({ kind: 'todo', title: 'Private', text: 'Mode check.' }, meta(70)));
        m = modes();
        check('a note leaves every file 0600', allPrivate(m) && Object.keys(m).length === 4, show(m));
        await compactJournal(pdir, { log: quiet });
        m = modes();
        check('compaction leaves every file 0600', allPrivate(m), show(m));
        fs.chmodSync(pdir, 0o755);
        for (const f of fs.readdirSync(pdir)) fs.chmodSync(path.join(pdir, f), 0o644);
        await appendNote(pdir, shapeNote({ kind: 'todo', title: 'Tightened', text: 'Loosened from outside.' }, meta(71)));
        m = modes();
        check('a write tightens a loosened journal', allPrivate(m), show(m));
        const lines = [];
        await initJournal({ nameOrPath: path.join(tmp, 'private-2'), log: (l) => lines.push(l) });
        check('no permissions warning where modes are enforced', !lines.some((l) => /does not enforce/.test(l)), lines.join(' | '));
    } finally {
        process.umask(prevUmask);
    }
}

console.log('journal: alongside a single corpus pack');
{
    // --journal added to a one-pack server: get_record and verify_pack
    // without `pack` still mean the corpus, as they did before.
    const corpusDir = path.join(tmp, 'corpus');
    await initJournal({ nameOrPath: corpusDir, log: quiet });
    const stdin = new PassThrough();
    const stdout = new PassThrough();
    const replies = [];
    let buffer = '';
    stdout.on('data', (d) => {
        buffer += d;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) { replies.push(JSON.parse(buffer.slice(0, idx))); buffer = buffer.slice(idx + 1); }
    });
    const done = runMcpServer({
        packPaths: [path.join(corpusDir, 'base-0001.pikelet')], journal: dir, openPikeletFile, httpRangeSource,
        serverVersion: '0.0.0-test', stdin, stdout, log: quiet,
    });
    const waitFor = (id) => new Promise((resolve, reject) => {
        const deadline = Date.now() + 60000;
        const timer = setInterval(() => {
            const hit = replies.find((r) => r.id === id);
            if (hit) { clearInterval(timer); resolve(hit); }
            else if (Date.now() > deadline) { clearInterval(timer); reject(new Error(`no reply for ${id}`)); }
        }, 5);
    });
    stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'get_record', arguments: { id: 0 } } })}\n`);
    const g = (await waitFor(1)).result;
    check('get_record without pack reads the one corpus pack', !g.isError && JSON.parse(g.content[0].text).title === 'Journal: corpus',
        g.content[0].text.slice(0, 160));
    stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'verify_pack', arguments: {} } })}\n`);
    const v = (await waitFor(2)).result;
    check('verify_pack without pack verifies the one corpus pack', !v.isError, v.content[0].text.slice(0, 160));
    stdin.end();
    await done;
}

console.log('journal: MCP tools');
{
    const stdin = new PassThrough();
    const stdout = new PassThrough();
    const replies = [];
    let nonJson = 0;
    let buffer = '';
    stdout.on('data', (d) => {
        buffer += d;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 1);
            // Every byte on stdout is protocol: append and compact log to the
            // server's log, never here.
            try { replies.push(JSON.parse(line)); } catch { nonJson++; }
        }
    });
    const done = runMcpServer({
        packPaths: [], journal: dir, openPikeletFile, httpRangeSource,
        serverVersion: '0.0.0-test', stdin, stdout, log: quiet,
    });
    const send = (m) => stdin.write(`${JSON.stringify(m)}\n`);
    const waitFor = (id) => new Promise((resolve, reject) => {
        const deadline = Date.now() + 60000;
        const timer = setInterval(() => {
            const hit = replies.find((r) => r.id === id);
            if (hit) { clearInterval(timer); resolve(hit); }
            else if (Date.now() > deadline) { clearInterval(timer); reject(new Error(`no reply for ${id}`)); }
        }, 5);
    });
    const call = async (id, name, args) => {
        send({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } });
        const r = await waitFor(id);
        return { error: r.result?.isError === true, body: r.result?.isError ? r.result.content[0].text : JSON.parse(r.result.content[0].text) };
    };
    send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test-client', version: '1.2.3' } } });
    await waitFor(1);
    send({ jsonrpc: '2.0', method: 'notifications/initialized' });
    send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
    const tools = (await waitFor(2)).result.tools.map((t) => t.name);
    check('a journal adds write_note and latest to the tools', tools.includes('write_note') && tools.includes('latest'), tools.join(','));

    const w = await call(3, 'write_note', { kind: 'handoff', title: 'Handoff over MCP', text: 'Goal, state, next steps.', tags: ['mcp'] });
    check('write_note returns a citation', !w.error && Number.isSafeInteger(w.body.citation?.id) && /^[0-9a-f]{64}$/.test(w.body.citation.identity), JSON.stringify(w.body).slice(0, 200));
    const l = await call(4, 'latest', { limit: 1 });
    check('the next read sees the note (the server remounted)', l.body.notes?.[0]?.id === w.body.citation.id && l.body.notes[0].title === 'Handoff over MCP');
    check('the author is the client from initialize', l.body.notes?.[0]?.authoredBy?.client === 'test-client' && l.body.notes[0].authoredBy.clientVersion === '1.2.3');
    const s = await call(5, 'search', { query: 'handoff over mcp', k: 2 });
    const sec = s.body.sections?.[0];
    check('search labels journal results as agent-written', sec?.journal === true && sec.results[0]?.agentWritten === true
        && sec.results[0]?.note?.kind === 'handoff' && typeof s.body.journalNote === 'string' && !s.body.unscoredWarning);
    const g = await call(6, 'get_record', { id: w.body.citation.id });
    check('get_record returns the full note', g.body.note?.kind === 'handoff' && g.body.text?.includes('Goal, state'));
    const bad = await call(7, 'write_note', { kind: 'handoff', title: 'T', text: 'B', created: '1999-01-01' });
    check('a note cannot set server fields over MCP', bad.error && /does not take created/.test(bad.body));
    const lp = await call(8, 'list_packs', {});
    check('list_packs marks the journal writable', lp.body.packs?.[0]?.journal === true && lp.body.packs[0].writable === true);
    check('the citation does not promise its identity resolves later',
        /compaction replaces that head/.test(w.body.note) && !/stay valid/.test(w.body.note), w.body.note);

    // Another session writes, then compacts: this server sees both without
    // restarting.
    const other = await appendNote(dir, shapeNote({ kind: 'decision', title: 'Written by another session', text: 'Not through this server.' }, meta(60)));
    const l2 = await call(9, 'latest', { limit: 1 });
    check('a note another session wrote is seen on the next call', l2.body.notes?.[0]?.id === other.id, JSON.stringify(l2.body.notes?.[0]).slice(0, 160));
    await compactJournal(dir, { log: quiet });
    const l3 = await call(10, 'latest', { limit: 2 });
    const g2 = await call(11, 'get_record', { id: w.body.citation.id });
    check('after a compaction elsewhere the server reopens and ids still hold',
        l3.body.notes?.[0]?.id === other.id && g2.body.title === 'Handoff over MCP', JSON.stringify(l3.body).slice(0, 160));

    // A modern request names its client in its own _meta.
    send({ jsonrpc: '2.0', id: 12, method: 'tools/call', params: {
        name: 'write_note', arguments: { kind: 'todo', title: 'From a modern client', text: 'Stateless request.' },
        _meta: { 'io.modelcontextprotocol/protocolVersion': '2026-07-28', 'io.modelcontextprotocol/clientCapabilities': {},
            'io.modelcontextprotocol/clientInfo': { name: 'modern-client', version: '9.9' } } } });
    const m = await waitFor(12);
    check('a modern write_note succeeds', m.result && !m.result.isError, JSON.stringify(m).slice(0, 200));
    const l4 = await call(13, 'latest', { limit: 1 });
    check('a modern request is authored by its own clientInfo',
        l4.body.notes?.[0]?.authoredBy?.client === 'modern-client' && l4.body.notes[0].authoredBy.clientVersion === '9.9',
        JSON.stringify(l4.body.notes?.[0]?.authoredBy));
    check('nothing but protocol reached stdout', nonJson === 0, `${nonJson} non-JSON line(s)`);
    stdin.end();
    await done;
}

console.log('journal: mcp install');
{
    const cwd = fs.mkdtempSync(path.join(tmp, 'install-'));
    const written = await installMcpConfig({ packPaths: [], journal: dir, cwd, homedir: cwd });
    const cfg = JSON.parse(fs.readFileSync(written.configPath, 'utf8'));
    const args = cfg.mcpServers['knowledge-packs'].args;
    check('install pins the journal by absolute path', args.includes('--journal') && args[args.indexOf('--journal') + 1] === dir, args.join(' '));
}

fs.rmSync(tmp, { recursive: true, force: true });
console.log(`\nJournal conformance: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
