// node eval.mjs <8|4>: the kernel against eval_ref.json (eval_ref.py).
//   long: 1000 teacher-forced tokens across the 256 window, by the decode
//         path (one token per call) and the prefill path (200-token calls)
//   ppl:  perplexity over the validation stories (prefill path)
import { readFileSync } from 'node:fs';
import createDecoder from './decoder.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, MAXSEQ = 256;
const bits = Number(process.argv[2] || 8);
const ref = JSON.parse(readFileSync(new URL('eval_ref.json', HERE), 'utf8'));
const mine = ref.variants[`u${bits}`];
const mod = await createDecoder();
const blobBytes = readFileSync(new URL(`tinystories-u${bits}.bin`, HERE));
const blob = mod._malloc(blobBytes.length);
mod.HEAPU8.set(blobBytes, blob);
const idsPtr = mod._malloc(4 * MAXSEQ);
const logitsPtr = mod._malloc(4 * V * MAXSEQ);

// Logits rows for ids at positions pos0.. (all rows, or just the last).
function forward(ids, pos0, all) {
  mod.HEAP32.set(ids, idsPtr >> 2);
  const rc = mod._decoder_forward(blob, idsPtr, ids.length, pos0, logitsPtr, bits, all ? 1 : 0);
  if (rc) throw new Error(`decoder_forward ${rc}`);
  const rows = all ? ids.length : 1;
  return Array.from({ length: rows }, (_, r) => mod.HEAPF32.subarray((logitsPtr >> 2) + r * V, (logitsPtr >> 2) + (r + 1) * V));
}
function logSoftmaxAt(row, id) {
  let mx = -Infinity;
  for (let i = 0; i < V; i++) if (row[i] > mx) mx = row[i];
  let s = 0;
  for (let i = 0; i < V; i++) s += Math.exp(row[i] - mx);
  return row[id] - mx - Math.log(s);
}
function argmax(row) { let b = 0; for (let i = 1; i < V; i++) if (row[i] > row[b]) b = i; return b; }
function npy(buf) { return new Float32Array(buf.buffer.slice(buf.byteOffset + 128, buf.byteOffset + buf.length)); }

// long
const ids = ref.long_ids, n = ids.length;
const cps = new Map(ref.checkpoints.map((p, k) => [p, k]));
const refCp = npy(readFileSync(new URL(`long_u${bits}.npy`, HERE)));
for (const path of ['decode', 'prefill']) {
  const t0 = performance.now();
  const am = [], lp = [], cpDiff = [];
  const take = (row, p) => {
    am[p] = argmax(row);
    if (p < n - 1) lp[p] = logSoftmaxAt(row, ids[p + 1]);
    if (cps.has(p)) {
      const k = cps.get(p);
      let d = 0;
      for (let i = 0; i < V; i++) d = Math.max(d, Math.abs(row[i] - refCp[k * V + i]));
      cpDiff.push(`${p}:${d.toFixed(4)}`);
    }
  };
  if (path === 'decode') for (let p = 0; p < n; p++) take(forward([ids[p]], p, false)[0], p);
  else for (let p0 = 0; p0 < n; p0 += 200) forward(ids.slice(p0, p0 + 200), p0, true).forEach((row, r) => take(row, p0 + r));
  const ms = performance.now() - t0;
  const agree = (lo, hi) => { let a = 0; for (let p = lo; p < hi; p++) a += am[p] === mine.long_argmax[p]; return `${a}/${hi - lo}`; };
  let dlp = 0, dlpLate = 0;
  for (let p = 0; p < n - 1; p++) { const d = Math.abs(lp[p] - mine.long_next_logprob[p]); dlp = Math.max(dlp, d); if (p >= 256) dlpLate = Math.max(dlpLate, d); }
  console.log(`long u${bits} ${path}: argmax == torch-u${bits} pos<256 ${agree(0, 256)}, pos>=256 ${agree(256, n)}; max|d logprob(next)| ${dlp.toFixed(4)} (pos>=256 ${dlpLate.toFixed(4)}); checkpoint max|d logit| ${cpDiff.join(' ')}; ${(n / ms * 1000).toFixed(0)} tok/s`);
}

// ppl
const t0 = performance.now();
let nll = 0, ntok = 0, maxStoryDiff = 0;
ref.story_ids.forEach((s, si) => {
  let storyNll = 0;
  for (let p0 = 0; p0 < s.length - 1; p0 += MAXSEQ) {
    const chunk = s.slice(p0, Math.min(p0 + MAXSEQ, s.length - 1));
    forward(chunk, p0, true).forEach((row, r) => { storyNll -= logSoftmaxAt(row, s[p0 + r + 1]); });
  }
  nll += storyNll;
  ntok += s.length - 1;
  maxStoryDiff = Math.max(maxStoryDiff, Math.abs(storyNll - mine.story_nll[si]) / (s.length - 1));
});
const ms = performance.now() - t0;
console.log(`ppl u${bits}: kernel ${Math.exp(nll / ntok).toFixed(4)} | torch-u${bits} ${mine.ppl.toFixed(4)} | torch-fp32 ${ref.variants.fp32.ppl.toFixed(4)} | ${ntok} tokens, max per-token NLL diff per story ${maxStoryDiff.toExponential(2)}; ${(ntok / ms * 1000).toFixed(0)} tok/s (prefill)`);
