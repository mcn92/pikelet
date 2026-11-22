// create's build-time passage encoder (workers-ai mode): the inline-transformer
// kernel running the pinned u8 BGE-small blob must embed like the fp32 model
// Workers AI serves for queries (test/fixtures/build-encoder), and the blob's
// download and cache must refuse anything that fails the pin.
//
// The first run downloads the 36 MiB blob into ~/.cache/pikelet/models
// (PIKELET_MODEL_CACHE overrides it); CI caches that directory.

import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { embedChunks, studentOutDir, trainStudentVectors } from '../packages/pikelet/src/embed.mjs';
import { DEFAULT_CONFIG, MODEL_MAP } from '../packages/pikelet/src/common.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(fs.readFileSync(path.join(HERE, 'fixtures/build-encoder/bge-small-reference.json'), 'utf8'));
const pin = MODEL_MAP['bge-small-en-v1.5'].weights;
let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
    if (ok) { passed++; console.log(`  ok   ${name}`); } else { failed++; console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`); }
}
async function rejects(name, fn, pattern) {
    try { await fn(); check(name, false, 'did not throw'); } catch (err) { check(name, pattern.test(String(err?.message)), String(err?.message)); }
}
const config = { embedding: { ...DEFAULT_CONFIG.embedding } };
const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const logs = [];
const log = (m) => logs.push(m);

console.log('build encoder vs fp32 BGE-small');
// The query prefix is part of the text here: passages carry no prefix, so
// embedChunks embeds exactly what it is given.
const items = [
    ...ref.queries.map((q) => ({ text: `${ref.queryPrefix}${q.text}`, vector: q.vector, label: `query "${q.text.slice(0, 30)}"` })),
    ...ref.passages.map((p) => ({ text: p.text, vector: p.vector, label: `passage "${p.text.slice(0, 30)}"` })),
];
const vectors = await embedChunks(items.map(({ text }) => ({ text })), config, log, process.cwd());
let worst = 1;
items.forEach((item, i) => {
    const c = cosine(vectors[i], item.vector);
    worst = Math.min(worst, c);
    // Measured 0.99994 for inputs within the 512-token window; the hosted
    // Workers AI copy itself sits at ~0.983 from fp32.
    check(`${item.label} cosine ${c.toFixed(5)} >= 0.9995`, c >= 0.9995);
});
check('every vector is unit length', vectors.every((v) => Math.abs(Math.hypot(...v) - 1) < 1e-4));
const again = await embedChunks(items.slice(0, 3).map(({ text }) => ({ text })), config, () => {}, process.cwd());
check('re-embedding is bit-identical', again.every((v, i) => Buffer.from(v.buffer).equals(Buffer.from(vectors[i].buffer))));
console.log(`  worst cosine ${worst.toFixed(5)}`);

console.log('pinned download and cache');
const cacheDir = process.env.PIKELET_MODEL_CACHE
    || path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'pikelet', 'models');
const blob = fs.readFileSync(path.join(cacheDir, pin.file));
check('the cached blob matches its pin', crypto.createHash('sha256').update(blob).digest('hex') === pin.sha256);
const tampered = Buffer.from(blob);
tampered[tampered.length - 1] ^= 1;
let requests = 0;
const server = http.createServer((req, res) => {
    requests++;
    const body = req.url === '/good' ? blob : req.url === '/tampered' ? tampered : null;
    if (!body) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-length': body.length });
    res.end(body);
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-build-encoder-'));
const saved = { cache: process.env.PIKELET_MODEL_CACHE, url: process.env[pin.urlEnv] };
const one = [{ text: ref.passages[0].text }];
try {
    process.env.PIKELET_MODEL_CACHE = tmp;
    const cached = path.join(tmp, pin.file);

    process.env[pin.urlEnv] = `${origin}/tampered`;
    await rejects('a download that fails the pin is refused', () => embedChunks(one, config, () => {}, tmp), /failed verification/);
    check('a refused download leaves nothing in the cache', fs.readdirSync(tmp).length === 0, fs.readdirSync(tmp).join(', '));

    process.env[pin.urlEnv] = `${origin}/missing`;
    await rejects('a failed download names the pin and the mirror variable', () => embedChunks(one, config, () => {}, tmp),
        new RegExp(`HTTP 404.*${pin.sha256}.*${pin.urlEnv}`));

    process.env[pin.urlEnv] = `${origin}/good`;
    const first = await embedChunks(one, config, () => {}, tmp);
    check('a missing blob is downloaded into the cache', fs.existsSync(cached) && fs.statSync(cached).size === blob.length);
    check('the downloaded blob embeds identically', Buffer.from(first[0].buffer).equals(Buffer.from(vectors[2].buffer)));

    const before = requests;
    await embedChunks(one, config, () => {}, tmp);
    check('a valid cached blob is not downloaded again', requests === before);

    fs.writeFileSync(cached, blob.subarray(0, 1000));
    const msgs = [];
    await embedChunks(one, config, (m) => msgs.push(m), tmp);
    check('a truncated cache entry is downloaded again', requests === before + 1 && msgs.some((m) => /fails its pinned digest/.test(m)));
    check('and replaced with the pinned bytes', fs.readFileSync(cached).equals(blob));
} finally {
    for (const [key, value] of [['PIKELET_MODEL_CACHE', saved.cache], [pin.urlEnv, saved.url]]) {
        if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    server.close();
    fs.rmSync(tmp, { recursive: true, force: true });
}

// The student trainer's outDir is deleted and recreated on every build, so a
// config value that resolves to the project, its parent, or anywhere outside
// it must be refused before anything is removed.
console.log('student outDir');
{
    const parent = fs.mkdtempSync(path.join(os.tmpdir(), 'pikelet-outdir-'));
    const project = path.join(parent, 'project');
    fs.mkdirSync(project);
    const sentinel = path.join(project, 'keep.txt');
    const sibling = path.join(parent, 'sibling-keep.txt');
    fs.writeFileSync(sentinel, 'keep');
    fs.writeFileSync(sibling, 'keep');
    try {
        for (const bad of ['.', '..', '../elsewhere', 'student/../..', project, parent, `${project}-other`]) {
            let error = null;
            try {
                await trainStudentVectors([], { trainStudent: { enabled: true, outDir: bad } }, () => {}, project);
            } catch (e) { error = e; }
            check(`trainStudent.outDir ${JSON.stringify(bad)} is refused before anything is deleted`,
                /must be a folder inside the project/.test(error?.message || '') && fs.existsSync(sentinel) && fs.existsSync(sibling),
                error?.message);
        }
        check('the default and nested folders inside the project are accepted',
            studentOutDir(project, undefined) === path.join(project, 'student')
            && studentOutDir(project, 'student') === path.join(project, 'student')
            && studentOutDir(project, 'build/student') === path.join(project, 'build', 'student')
            && studentOutDir(project, path.join(project, 'x')) === path.join(project, 'x'));
    } finally {
        fs.rmSync(parent, { recursive: true, force: true });
    }
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
