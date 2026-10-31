// node bench.mjs [module.mjs ...] -- decode speed, one token per call, over
// the first N tokens of eval_ref.json's long text (no EOS stops), per weight
// format, each module, interleaved rounds. Default: ./decoder.mjs.
import { readFileSync, existsSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const HERE = new URL('.', import.meta.url);
const V = 50257, N = Number(process.env.BENCH_TOKENS || 300), ROUNDS = Number(process.env.BENCH_ROUNDS || 3);
const ids = JSON.parse(readFileSync(new URL('eval_ref.json', HERE), 'utf8')).long_ids.slice(0, N);
const mods = process.argv.slice(2).length ? process.argv.slice(2) : ['./decoder.mjs'];
const formats = (process.env.BENCH_BITS || '8,4,5').split(',').map(Number);

async function setup(modPath, bits) {
  const { default: create } = await import(pathToFileURL(path.resolve(modPath)).href);
  const mod = await create();
  const bytes = readFileSync(new URL(`tinystories-u${bits}.bin`, HERE));
  const blob = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, blob);
  const idsPtr = mod._malloc(4), logits = mod._malloc(4 * V);
  return () => {
    const t0 = performance.now();
    for (let p = 0; p < ids.length; p++) {
      mod.HEAP32[idsPtr >> 2] = ids[p];
      const rc = mod._decoder_forward(blob, idsPtr, 1, p, logits, bits, 0);
      if (rc) throw new Error(`${modPath} u${bits}: rc ${rc}`);
    }
    return ids.length / (performance.now() - t0) * 1000;
  };
}

for (const bits of formats) {
  if (!existsSync(new URL(`tinystories-u${bits}.bin`, HERE))) continue;
  const runs = [];
  for (const m of mods) {
    try { runs.push({ m, run: await setup(m, bits), tps: [] }); } catch (e) { console.log(`${m} u${bits}: ${e.message}`); }
  }
  for (let r = 0; r < ROUNDS; r++) for (const x of runs) {
    try { x.tps.push(x.run()); } catch (e) { x.err = e.message; }
  }
  for (const x of runs) {
    const s = x.tps.slice().sort((a, b) => a - b);
    console.log(x.err ? `u${bits} ${x.m}: ${x.err}` : `u${bits} ${x.m}: median ${s[s.length >> 1].toFixed(1)} tok/s (rounds ${x.tps.map((t) => t.toFixed(1)).join(', ')}; ${N} tokens from pos 0)`);
  }
}
