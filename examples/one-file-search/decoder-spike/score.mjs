// node score.mjs [bits] -- the kernel as a scorer on score_items.json
// (score_ref.py): read each prefix once, then score every candidate from the
// prefix's end in the KV cache (each candidate overwrites the last one's
// positions). Accuracy and choices against torch fp32, and time per item.
import { readFileSync } from 'node:fs';
import createDecoder from './decoder.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, MAXSEQ = 256;
const bits = Number(process.argv[2] || 8);
const items = JSON.parse(readFileSync(new URL('score_items.json', HERE), 'utf8'));
const mod = await createDecoder();
const bytes = readFileSync(new URL(`tinystories-u${bits === 6 ? 5 : bits}.bin`, HERE));
let blob = mod._malloc(bytes.length);
mod.HEAPU8.set(bytes, blob);
if (bits === 6) { const out = mod._malloc(Number(mod._decoder_expand_u5(blob, 0))); mod._decoder_expand_u5(blob, out); blob = out; }
const idsPtr = mod._malloc(4 * MAXSEQ), lg = mod._malloc(4 * V * MAXSEQ);
const fwd = (ids, pos0, all) => {
  mod.HEAP32.set(ids, idsPtr >> 2);
  if (mod._decoder_forward(blob, idsPtr, ids.length, pos0, lg, bits, all ? 1 : 0)) throw new Error('forward');
};
const logSoftmaxAt = (row, id) => {
  let mx = -Infinity; for (let i = 0; i < V; i++) if (row[i] > mx) mx = row[i];
  let s = 0; for (let i = 0; i < V; i++) s += Math.exp(row[i] - mx);
  return row[id] - mx - Math.log(s);
};
const argmax = (a) => a.indexOf(Math.max(...a));

let sumOk = 0, meanOk = 0, sumAgree = 0, meanAgree = 0, maxDiff = 0;
const t0 = performance.now();
for (const it of items) {
  const p = it.prefix;
  fwd(p.slice(0, -1), 0, false);  // prefix minus its last token, cached once
  const sums = [], means = [];
  for (const c of it.cands) {
    const chunk = [p[p.length - 1], ...c];  // row r predicts chunk[r + 1]
    fwd(chunk, p.length - 1, true);
    let s = 0;
    for (let r = 0; r < c.length; r++) s += logSoftmaxAt(mod.HEAPF32.subarray((lg >> 2) + r * V, (lg >> 2) + (r + 1) * V), c[r]);
    sums.push(s); means.push(s / c.length);
  }
  it.torch_sum.forEach((x, k) => { maxDiff = Math.max(maxDiff, Math.abs(x - sums[k]) / it.cands[k].length); });
  sumOk += argmax(sums) === it.answer; meanOk += argmax(means) === it.answer;
  sumAgree += argmax(sums) === argmax(it.torch_sum); meanAgree += argmax(means) === argmax(it.torch_mean);
}
const ms = (performance.now() - t0) / items.length;
const n = items.length, f = (x) => (x / n).toFixed(3);
const tAcc = (key) => f(items.filter((it) => argmax(it[key]) === it.answer).length);
console.log(`format ${bits}: accuracy summed ${f(sumOk)} (torch fp32 ${tAcc('torch_sum')}), per-token mean ${f(meanOk)} (torch ${tAcc('torch_mean')}); same choice as torch ${sumAgree}/${n} and ${meanAgree}/${n}; max per-token log-likelihood diff vs fp32 ${maxDiff.toFixed(4)}; ${ms.toFixed(0)} ms per item (prefix + 4 candidates)`);
