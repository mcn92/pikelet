// node kernel_run.mjs <name> <bits: 8|5> <rag_out file> -- reruns a
// rag_ref.py condition through llama.cpp from the same prompt ids (greedy,
// 48 tokens), reports how many replies equal torch's and the speed, and
// writes the kernel's replies as rag_out-<name>-kernel-u<bits>-<condition>.json
// for rag_score.mjs.
import { readFileSync, writeFileSync } from 'node:fs';
import createLlama from './llama.mjs';
import { createBpeFromJson } from './bpe.mjs';

const HERE = new URL('.', import.meta.url);
const [NAME, BITS, FILE] = process.argv.slice(2);
const cond = FILE.match(/-(closed|rag\d)\.json$/)[1];
const torch = JSON.parse(readFileSync(new URL(FILE, HERE), 'utf8'));
const tokJson = JSON.parse(readFileSync(new URL(`llama-tokenizer/${NAME}.json`, HERE), 'utf8'));
const bpe = createBpeFromJson(tokJson);
const dec = new TextDecoder();
const STOP = new Set([tokJson.eos, tokJson.specials['<|im_end|>']]);

const mod = await createLlama();
const raw = readFileSync(new URL(`${NAME}-u${BITS}.bin`, HERE));
let blob = mod._malloc(raw.length);
mod.HEAPU8.set(raw, blob);
if (BITS === '5') {
  const out = mod._malloc(Number(mod._llama_expand_u5(blob, 0)));
  mod._llama_expand_u5(blob, out);
  mod._free(blob);
  blob = out;
}
if (mod._llama_init(blob, BITS === '5' ? 6 : 8, 2048)) throw new Error('llama_init');
const V = mod._llama_dim(0);
const idsPtr = mod._malloc(4 * 256), lg = mod._malloc(4 * V);
const fwd = (ids, pos0) => {
  mod.HEAP32.set(ids, idsPtr >> 2);
  if (mod._llama_forward(blob, idsPtr, ids.length, pos0, lg, 0)) throw new Error('llama_forward');
  const row = mod.HEAPF32.subarray(lg >> 2, (lg >> 2) + V);
  let b = 0; for (let i = 1; i < V; i++) if (row[i] > row[b]) b = i;
  return b;
};
const out = [];
let same = 0, promptTok = 0, promptMs = 0, genTok = 0, genMs = 0;
for (const [k, t] of torch.entries()) {
  let t0 = performance.now(), next;
  for (let p = 0; p < t.ids.length; p += 256) next = fwd(t.ids.slice(p, p + 256), p);
  promptMs += performance.now() - t0; promptTok += t.ids.length;
  const gen = [];
  let pos = t.ids.length;
  t0 = performance.now();
  while (gen.length < 48 && !STOP.has(next)) { gen.push(next); next = fwd([next], pos++); }
  genMs += performance.now() - t0; genTok += gen.length;
  const reply = dec.decode(bpe.decodeBytes(gen)).trim();
  if (reply === t.reply) same++;
  out.push({ id: t.id, reply });
  if ((k + 1) % 20 === 0) console.log(`${k + 1}/${torch.length}`);
}
writeFileSync(new URL(`rag_out-${NAME}-kernel-u${BITS}-${cond}.json`, HERE), JSON.stringify(out));
console.log(`kernel u${BITS} ${cond}: ${same}/${torch.length} replies identical to torch; prompt ${(promptTok / promptMs * 1000).toFixed(0)} tok/s (mean ${(promptTok / torch.length).toFixed(0)} tokens, ${(promptMs / torch.length / 1000).toFixed(1)} s each), generation ${(genTok / genMs * 1000).toFixed(1)} tok/s`);
