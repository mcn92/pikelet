// node record_candidates.mjs -- the vocabulary shortlist's candidate sets,
// for remote_rows_sim.py (could the exact-rescoring wte rows be fetched on
// demand instead of shipped?). u8 blob, rank-192 shortlist; the candidates
// of a step are its logits that are not -inf. Writes candidates/:
//   calib-k256.bin   teacher-forced validation stories 6000+ (~10k steps),
//                    only to rank rows by how often they are candidates
//   test-k{K}.bin    for K = 128 and 256: the 1000-token long text
//                    teacher-forced, then 50 prompts (first sentence of
//                    stories 5000+) x 200 greedy tokens, then the same
//                    prompts x 200 tokens sampled (temp 0.8, top-40, seeded)
//   index.json       per K: segments [{kind, steps, prompt ids}] in file order
// Each .bin is Int32 K ids per step, in step order.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import createDecoder from './decoder.mjs';
import { createBpe } from './bpe.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, EOS = 50256, R = 192, CALIB_STEPS = 10000, N_PROMPTS = 50, N_GEN = 200;
const bpe = createBpe(JSON.parse(readFileSync(new URL('tokenizer/vocab.json', HERE), 'utf8')),
  readFileSync(new URL('tokenizer/merges.txt', HERE), 'utf8'));
const stories = readFileSync(new URL('data/TinyStories-valid.txt', HERE), 'utf8').split('<|endoftext|>').map((s) => s.trim()).filter(Boolean);
const longIds = JSON.parse(readFileSync(new URL('eval_ref.json', HERE), 'utf8')).long_ids;
const prompts = stories.slice(5000, 5000 + N_PROMPTS).map((s) => bpe.encode(s.split(/(?<=[.!?])\s/)[0]));

const mod = await createDecoder();
const bytes = readFileSync(new URL('tinystories-u8.bin', HERE));
const blob = mod._malloc(bytes.length);
mod.HEAPU8.set(bytes, blob);
const slBytes = readFileSync(new URL(`shortlist-r${R}.bin`, HERE));
const sl = mod._malloc(slBytes.length);
mod.HEAPU8.set(slBytes, sl);
const idsPtr = mod._malloc(4 * 256), lg = mod._malloc(4 * V);

let out = [];
const step = (ids, pos0, K) => {
  mod.HEAP32.set(ids, idsPtr >> 2);
  if (mod._decoder_forward(blob, idsPtr, ids.length, pos0, lg, 8, 0)) throw new Error('forward');
  const r = mod.HEAPF32.subarray(lg >> 2, (lg >> 2) + V);
  const c = [];
  for (let i = 0; i < V; i++) if (r[i] !== -Infinity) c.push(i);
  if (c.length !== K) throw new Error(`${c.length} candidates, expected ${K}`);
  out.push(...c);
  return r;
};
let seed = 1;
const rand = () => { seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
function pick(r, temp) {
  let best = 0;
  for (let i = 1; i < V; i++) if (r[i] > r[best]) best = i;
  if (!temp) return best;
  const idx = [];
  for (let i = 0; i < V; i++) {
    if (r[i] === -Infinity || (idx.length === 40 && r[i] <= r[idx[39]])) continue;
    let k = idx.length;
    while (k > 0 && r[idx[k - 1]] < r[i]) k--;
    idx.splice(k, 0, i);
    if (idx.length > 40) idx.pop();
  }
  const w = idx.map((i) => Math.exp((r[i] - r[best]) / temp));
  let x = rand() * w.reduce((a, b) => a + b);
  for (let k = 0; k < idx.length; k++) if ((x -= w[k]) <= 0) return idx[k];
  return idx[0];
}
// Prompt read in one call (its last step's candidates are recorded), then
// one step per generated token.
function generate(p, K, temp) {
  let r = step(p, 0, K), pos = p.length, steps = 1;
  for (let k = 0; k < N_GEN; k++) {
    const id = pick(r, temp);
    if (id === EOS) break;
    r = step([id], pos++, K); steps++;
  }
  return steps;
}
const flush = (name) => { writeFileSync(new URL(`candidates/${name}.bin`, HERE), new Int32Array(out)); const n = out.length; out = []; return n; };
mkdirSync(new URL('candidates', HERE), { recursive: true });

const t0 = performance.now();
if (mod._decoder_set_shortlist(sl, R, 256)) throw new Error('set_shortlist');
let calib = 0;
for (const s of stories.slice(6000)) {
  const ids = [EOS, ...bpe.encode(s)].slice(0, 1024);
  for (let p = 0; p < ids.length; p++) step([ids[p]], p, 256);
  calib += ids.length;
  if (calib >= CALIB_STEPS) break;
}
flush('calib-k256');
console.log(`calib: ${calib} steps (${((performance.now() - t0) / 1000).toFixed(0)} s)`);

const index = {};
for (const K of [128, 256]) {
  if (mod._decoder_set_shortlist(sl, R, K)) throw new Error('set_shortlist');
  const segs = [];
  for (let p = 0; p < longIds.length; p++) step([longIds[p]], p, K);
  segs.push({ kind: 'long', steps: longIds.length, prompt: [] });
  for (const temp of [0, 0.8]) {
    seed = 1;
    for (const p of prompts) segs.push({ kind: temp ? 'sampled' : 'greedy', steps: generate(p, K, temp), prompt: p });
  }
  const n = flush(`test-k${K}`);
  index[K] = segs;
  console.log(`K ${K}: ${n / K} steps in ${segs.length} segments (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
}
writeFileSync(new URL('candidates/index.json', HERE), JSON.stringify(index));
