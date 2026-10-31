// node grounded.mjs <name> <bits: 8|5> -- grounded QA over SQuAD v2 passages
// (data/grounded_items.json, from grounded_items.py) with llama.cpp: each
// item's passage and question in the model's chat format (grounded_score.mjs
// SYSTEM), greedy reply up to 48 tokens, scored SQuAD-style. Writes
// grounded_results-<name>-u<bits>.json and prints scores and speed.
import { readFileSync, writeFileSync } from 'node:fs';
import createLlama from './llama.mjs';
import { createBpeFromJson } from './bpe.mjs';
import { chatPrompt, score, summarize } from './grounded_score.mjs';

const HERE = new URL('.', import.meta.url);
const [NAME = 'smollm2-360m', BITS = '8'] = process.argv.slice(2);
const items = JSON.parse(readFileSync(new URL('data/grounded_items.json', HERE), 'utf8'));
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

const rows = [];
let promptTok = 0, promptMs = 0, genTok = 0, genMs = 0;
for (const [k, item] of items.entries()) {
  const ids = bpe.encode(chatPrompt(item));
  let t = performance.now(), next;
  for (let p = 0; p < ids.length; p += 256) next = fwd(ids.slice(p, p + 256), p);
  promptMs += performance.now() - t; promptTok += ids.length;
  const out = [];
  let pos = ids.length;
  t = performance.now();
  while (out.length < 48 && !STOP.has(next)) { out.push(next); next = fwd([next], pos++); }
  genMs += performance.now() - t; genTok += out.length;
  const reply = dec.decode(bpe.decodeBytes(out)).trim();
  rows.push({ id: item.id, promptTokens: ids.length, reply, score: score(item, reply) });
  if ((k + 1) % 20 === 0) console.log(`${k + 1}/${items.length}`);
}
const summary = summarize(rows);
summary.promptTokS = +(promptTok / promptMs * 1000).toFixed(0);
summary.genTokS = +(genTok / genMs * 1000).toFixed(1);
summary.meanPromptTokens = +(promptTok / items.length).toFixed(0);
writeFileSync(new URL(`grounded_results-${NAME}-u${BITS}.json`, HERE), JSON.stringify({ summary, rows }, null, 1));
console.log(JSON.stringify(summary));
