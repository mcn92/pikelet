// node shortlist_check.mjs [bits] -- the vocabulary shortlist against the
// full LM head, teacher-forced over eval_ref.json's 1000-token text, one
// token per call: shortlisted logits must equal the full head's bit for bit;
// how often the full argmax and the full top-40 survive the shortlist; decode
// speed. Then greedy generation, 3 prompts x 200 tokens, full vs shortlist.
import { readFileSync } from 'node:fs';
import createDecoder from './decoder.mjs';
import { createBpe } from './bpe.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, EOS = 50256;
const bits = Number(process.argv[2] || 8);
const ids = JSON.parse(readFileSync(new URL('eval_ref.json', HERE), 'utf8')).long_ids;
const bpe = createBpe(JSON.parse(readFileSync(new URL('tokenizer/vocab.json', HERE), 'utf8')),
  readFileSync(new URL('tokenizer/merges.txt', HERE), 'utf8'));
const mod = await createDecoder();
const bytes = readFileSync(new URL(`tinystories-u${bits === 6 ? 5 : bits}.bin`, HERE));
let blob = mod._malloc(bytes.length);
mod.HEAPU8.set(bytes, blob);
if (bits === 6) { const out = mod._malloc(Number(mod._decoder_expand_u5(blob, 0))); mod._decoder_expand_u5(blob, out); blob = out; }
const idsPtr = mod._malloc(4 * 256), lg = mod._malloc(4 * V);
const step = (chunk, pos) => {
  mod.HEAP32.set(chunk, idsPtr >> 2);
  if (mod._decoder_forward(blob, idsPtr, chunk.length, pos, lg, bits, 0)) throw new Error('forward');
  return mod.HEAPF32.slice(lg >> 2, (lg >> 2) + V);
};
const argmax = (r) => { let b = 0; for (let i = 1; i < V; i++) if (r[i] > r[b]) b = i; return b; };
const topk = (r, k) => Array.from(r.keys()).sort((a, b) => r[b] - r[a]).slice(0, k);

function teacher() {
  const rows = [];
  const t0 = performance.now();
  for (let p = 0; p < ids.length; p++) rows.push(step([ids[p]], p));
  return { rows, tps: ids.length / (performance.now() - t0) * 1000 };
}
function greedy(prompt, n) {
  const p = bpe.encode(prompt), out = [];
  let r = step(p, 0), pos = p.length;
  for (let k = 0; k < n; k++) { const id = argmax(r); if (id === EOS) break; out.push(id); r = step([id], pos++); }
  return out;
}
const PROMPTS = ['Once upon a time, there was a little dog named', 'Lily wanted to bake a cake for her mom, but', 'The robot looked at the moon and said'];

mod._decoder_set_shortlist(0, 0, 0);
const full = teacher();
const fullGreedy = PROMPTS.map((p) => greedy(p, 200));
const fullTop = full.rows.map((r) => ({ arg: argmax(r), top40: topk(r, 40) }));
console.log(`full head: ${full.tps.toFixed(0)} tok/s`);
for (const r of [64, 128, 192, 256]) {
  let sl;
  try { sl = readFileSync(new URL(`shortlist-r${r}.bin`, HERE)); } catch { continue; }
  const slPtr = mod._malloc(sl.length);
  mod.HEAPU8.set(sl, slPtr);
  for (const K of [64, 128, 256]) {
    if (mod._decoder_set_shortlist(slPtr, r, K)) throw new Error('set_shortlist');
    const s = teacher();
    let argIn = 0, top40 = 0, exact = true;
    s.rows.forEach((row, p) => {
      const f = full.rows[p];
      for (let i = 0; i < V; i++) if (row[i] !== -Infinity && row[i] !== f[i]) exact = false;
      if (row[fullTop[p].arg] !== -Infinity) argIn++;
      top40 += fullTop[p].top40.filter((i) => row[i] !== -Infinity).length;
    });
    const g = PROMPTS.map((p) => greedy(p, 200));
    const gSame = g.filter((x, i) => x.join() === fullGreedy[i].join()).length;
    console.log(`r ${r} K ${K}: argmax kept ${argIn}/${ids.length}, top-40 recall ${(100 * top40 / (40 * ids.length)).toFixed(2)}%, shortlisted logits ${exact ? 'bit-identical' : 'DIFFER'}, greedy identical ${gSame}/3; ${s.tps.toFixed(0)} tok/s (${(s.tps / full.tps).toFixed(2)}x)`);
  }
  mod._decoder_set_shortlist(0, 0, 0);
  mod._free(slPtr);
}
