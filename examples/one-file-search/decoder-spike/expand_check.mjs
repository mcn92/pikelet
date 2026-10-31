// node expand_check.mjs -- u5 expanded at load (format 6) against u5 decoded
// in the kernel (format 5): logits must be bit-identical, token by token;
// then decode speed for both, and the expanded size.
import { readFileSync } from 'node:fs';
import createDecoder from './decoder.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, N = 300;
const ids = JSON.parse(readFileSync(new URL('eval_ref.json', HERE), 'utf8')).long_ids.slice(0, N);
const mod = await createDecoder();
const bytes = readFileSync(new URL('tinystories-u5.bin', HERE));
const u5 = mod._malloc(bytes.length);
mod.HEAPU8.set(bytes, u5);
const size = Number(mod._decoder_expand_u5(u5, 0));
const u6 = mod._malloc(size);
const t0 = performance.now();
if (Number(mod._decoder_expand_u5(u5, u6)) !== size) throw new Error('expand size mismatch');
const expandMs = performance.now() - t0;
const idsPtr = mod._malloc(4), logits = mod._malloc(4 * V);

function run(blob, bits, keep) {
  const out = [];
  const t = performance.now();
  for (let p = 0; p < ids.length; p++) {
    mod.HEAP32[idsPtr >> 2] = ids[p];
    const rc = mod._decoder_forward(blob, idsPtr, 1, p, logits, bits, 0);
    if (rc) throw new Error(`rc ${rc}`);
    if (keep) out.push(mod.HEAPF32.slice(logits >> 2, (logits >> 2) + V));
  }
  return { out, tps: ids.length / (performance.now() - t) * 1000 };
}

const a = run(u5, 5, true), b = run(u6, 6, true);
let mismatched = 0;
a.out.forEach((row, p) => { for (let i = 0; i < V; i++) if (row[i] !== b.out[p][i] && !(Number.isNaN(row[i]) && Number.isNaN(b.out[p][i]))) { mismatched++; break; } });
console.log(`format 6 vs 5 logits: ${mismatched === 0 ? 'bit-identical' : `${mismatched} positions differ`} over ${N} tokens`);
console.log(`expanded: ${(size / 1e6).toFixed(1)} MB in memory from a ${(bytes.length / 1e6).toFixed(1)} MB file, in ${expandMs.toFixed(0)} ms`);
const speed = { 5: [], 6: [] };
for (let r = 0; r < 3; r++) { speed[5].push(run(u5, 5).tps); speed[6].push(run(u6, 6).tps); }
const med = (x) => x.slice().sort((p, q) => p - q)[1].toFixed(1);
console.log(`decode, ${N} tokens from pos 0: u5 in-kernel ${med(speed[5])} tok/s, u5 expanded ${med(speed[6])} tok/s`);
