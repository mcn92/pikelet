// node speed.mjs [profile-build.mjs]
//   prompt reading: n tokens in calls of up to 256, logits for the last only
//   generation: per-token time for 32 tokens after a context of c tokens
//   with a PROFILE build (bash build.sh then emcc -DPROFILE, see README):
//   the per-phase split of generation and of prompt reading
// Formats: u8, u4, and u5 expanded at load (format 6).
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const HERE = new URL('.', import.meta.url);
const V = 50257, MAXSEQ = 256;
const ids = JSON.parse(readFileSync(new URL('eval_ref.json', HERE), 'utf8')).long_ids;
const modPath = process.argv[2] ? pathToFileURL(path.resolve(process.argv[2])).href : new URL('decoder.mjs', HERE).href;
const { default: create } = await import(modPath);
const PHASES = ['embedding', 'LayerNorm+residual', 'Q/K/V GEMMs', 'attention', 'output GEMM', 'FFN up GEMM', 'GELU', 'FFN down GEMM', 'LM head'];

async function load(fmt) {
  const mod = await create();
  const bytes = readFileSync(new URL(`tinystories-u${fmt === 6 ? 5 : fmt}.bin`, HERE));
  let blob = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, blob);
  if (fmt === 6) {
    const out = mod._malloc(Number(mod._decoder_expand_u5(blob, 0)));
    mod._decoder_expand_u5(blob, out);
    mod._free(blob);
    blob = out;
  }
  const idsPtr = mod._malloc(4 * MAXSEQ), logits = mod._malloc(4 * V);
  const call = (chunk, pos0) => {
    mod.HEAP32.set(chunk, idsPtr >> 2);
    const rc = mod._decoder_forward(blob, idsPtr, chunk.length, pos0, logits, fmt, 0);
    if (rc) throw new Error(`rc ${rc}`);
  };
  const read = (n) => { for (let p = 0; p < n; p += MAXSEQ) call(ids.slice(p, Math.min(n, p + MAXSEQ)), p); };
  const prof = mod._decoder_profile ? () => new Float64Array(mod.HEAPF64.buffer, mod._decoder_profile(), PHASES.length) : null;
  return { read, call, prof };
}
const median = (f, reps = 3) => { const t = []; for (let r = 0; r < reps; r++) t.push(f()); return t.sort((a, b) => a - b)[reps >> 1]; };
const timed = (f) => { const t0 = performance.now(); f(); return performance.now() - t0; };

for (const fmt of [8, 4, 6]) {
  const name = fmt === 6 ? 'u5 (expanded)' : `u${fmt}`;
  const d = await load(fmt);
  const readRow = [16, 64, 256, 512, 1000].map((n) => {
    const ms = median(() => timed(() => d.read(n)));
    return `${n}: ${ms.toFixed(0)} ms (${(n / ms * 1000).toFixed(0)} tok/s)`;
  });
  console.log(`${name} prompt reading  ${readRow.join(' | ')}`);
  const genRow = [16, 256, 512, 968].map((c) => {
    const ms = median(() => { d.read(c); return timed(() => { for (let k = 0; k < 32; k++) d.call([ids[c + k]], c + k); }) / 32; });
    return `ctx ${c}: ${ms.toFixed(1)} ms/token (${(1000 / ms).toFixed(0)} tok/s)`;
  });
  console.log(`${name} generation      ${genRow.join(' | ')}`);
  if (d.prof) {
    const split = (label, f) => {
      d.prof().fill(0);
      f();
      const p = Array.from(d.prof()), total = p.reduce((a, b) => a + b, 0);
      console.log(`${name} ${label}: ${PHASES.map((ph, i) => `${ph} ${(100 * p[i] / total).toFixed(1)}%`).join(', ')}`);
    };
    split('split, generating 32 tokens after 16', () => { d.read(16); d.prof().fill(0); for (let k = 0; k < 32; k++) d.call([ids[16 + k]], 16 + k); });
    split('split, generating 32 tokens after 968', () => { d.read(968); d.prof().fill(0); for (let k = 0; k < 32; k++) d.call([ids[968 + k]], 968 + k); });
    split('split, reading a 1000-token prompt', () => d.read(1000));
  }
}
