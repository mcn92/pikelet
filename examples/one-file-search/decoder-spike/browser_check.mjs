// node browser_check.mjs -- the decoder's browser build (decoder.web.mjs,
// build.sh) in Chromium, Firefox and WebKit against the Node build
// (decoder.mjs), on workload.mjs: every logits row must be bit-identical to
// Node's (the first differing position is reported if not), greedy output
// identical, plus decode / prefill / generation speed per engine.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import createDecoder from './decoder.mjs';
import { runWorkload } from './workload.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.resolve(HERE, '../../../package.json'));
const { chromium, firefox, webkit } = require('playwright');
const longIds = JSON.parse(readFileSync(path.join(HERE, 'eval_ref.json'), 'utf8')).long_ids;
const prompts = JSON.parse(readFileSync(path.join(HERE, 'refs.json'), 'utf8')).prompts.map((p) => p.ids);
const args = { longIds, prompts, nLong: Number(process.env.NLONG || 600), nGen: Number(process.env.NGEN || 100) };

const t0 = performance.now();
const node = await runWorkload(await createDecoder(), async (f) => readFileSync(path.join(HERE, f)), args);
console.log(`node: done in ${((performance.now() - t0) / 1000).toFixed(0)} s`);

const SERVED = /^\/(decoder\.web\.(mjs|wasm)|workload\.mjs|[\w-]+\.bin)$/;
const TYPES = { '.mjs': 'text/javascript', '.wasm': 'application/wasm' };
const server = http.createServer(async (req, res) => {
  const p = new URL(req.url, 'http://x').pathname;
  if (p === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>decoder</title>'); return; }
  if (!SERVED.test(p)) { res.writeHead(404); res.end(); return; }
  try {
    const body = await readFile(path.join(HERE, p));
    res.writeHead(200, { 'content-type': TYPES[path.extname(p)] || 'application/octet-stream' });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;

const firstDiff = (a, b) => { if (!a && !b) return -1; for (let i = 0; i < Math.max(a.length, b.length); i++) if (a[i] !== b[i]) return i; return -1; };
const rows = [['node', node]];
let failed = 0;
const ENGINES = (process.env.ENGINES || 'chromium,firefox,webkit').split(',');
for (const [name, engine] of Object.entries({ chromium, firefox, webkit }).filter(([n]) => ENGINES.includes(n))) {
  let browser;
  try {
    browser = await engine.launch({ headless: true });
    const page = await browser.newPage();
    page.setDefaultTimeout(0);
    page.on('console', (m) => console.log(`  [${name}] ${m.text()}`));
    await page.goto(origin);
    const t = performance.now();
    const got = await page.evaluate(async (args) => {
      const { default: createDecoder } = await import('/decoder.web.mjs');
      const { runWorkload } = await import('/workload.mjs');
      const load = async (f) => new Uint8Array(await (await fetch(`/${f}`)).arrayBuffer());
      return runWorkload(await createDecoder(), load, args, (m) => console.log(m));
    }, args);
    console.log(`${name}: done in ${((performance.now() - t) / 1000).toFixed(0)} s (${await browser.version()})`);
    got.forEach((g, i) => {
      const n = node[i], bad = [];
      for (const key of ['decode', 'prefill']) {
        const d = firstDiff(g[key], n[key]);
        if (d >= 0) bad.push(`${key} logits differ from position ${d}`);
      }
      const gd = g.greedy.filter((x, k) => x.join() !== n.greedy[k].join()).length;
      if (gd) bad.push(`greedy differs on ${gd}/${g.greedy.length} prompts`);
      if (bad.length) { failed++; console.log(`  ${g.name}: FAIL ${bad.join('; ')}`); }
      else console.log(`  ${g.name}: bit-identical to Node (${g.decode.length} decode rows${g.prefill ? `, ${g.prefill.length} prefill rows` : ''}, ${g.greedy.flat().length} greedy tokens)`);
    });
    rows.push([name, got]);
  } catch (err) {
    failed++;
    console.log(`${name}: FAIL ${err.message.split('\n')[0]}`);
  } finally {
    await browser?.close();
  }
}
server.close();

console.log(`\nspeed, tok/s: decode (${args.nLong} tokens one per call) / prefill (200-token calls) / greedy generation`);
console.log(['config', ...rows.map(([n]) => n)].join(' | '));
node.forEach((c, i) => console.log([c.name, ...rows.map(([, r]) => {
  const x = r[i];
  return `${x.decodeTps.toFixed(0)} / ${x.prefillTps ? x.prefillTps.toFixed(0) : '-'} / ${x.genTps.toFixed(0)}`;
})].join(' | ')));
console.log(failed ? `\nFAILED (${failed})` : '\nall engines bit-identical to Node');
process.exit(failed ? 1 : 0);
