// node llama_check.mjs <name> <bits: 8|5> -- llama.cpp against the torch
// references from llama_ref.py (the model with the blob's own dequantized
// weights): tokenizer ids, chat-prompt logits and greedy text, and a
// teacher-forced long text decoded one token per call and in 200-token
// calls (argmax agreement, next-token log-prob and checkpoint logits).
// bits 5 runs the u5 blob expanded to format 6.
import { readFileSync } from 'node:fs';
import createLlama from './llama.mjs';
import { createBpeFromJson } from './bpe.mjs';

const HERE = new URL('.', import.meta.url);
const [NAME = 'smollm2-135m', BITS = '8'] = process.argv.slice(2);
const ref = JSON.parse(readFileSync(new URL(`llama_ref-${NAME}-u${BITS}.json`, HERE), 'utf8'));
const tokJson = JSON.parse(readFileSync(new URL(`llama-tokenizer/${NAME}.json`, HERE), 'utf8'));
const bpe = createBpeFromJson(tokJson);
const dec = new TextDecoder();

let tokOk = 0;
for (const t of ref.tokenizer) {
  const got = bpe.encode(t.text);
  if (got.join() === t.ids.join()) tokOk++;
  else console.log(`  tokenizer differs on ${JSON.stringify(t.text)}:\n    ours ${got.join(' ')}\n    hf   ${t.ids.join(' ')}`);
}
console.log(`tokenizer: ${tokOk}/${ref.tokenizer.length} test strings identical to Hugging Face`);

const mod = await createLlama();
const raw = readFileSync(new URL(`${NAME}-u${BITS}.bin`, HERE));
let blob = mod._malloc(raw.length);
mod.HEAPU8.set(raw, blob);
let fmt = 8;
if (BITS === '5') {
  const size = Number(mod._llama_expand_u5(blob, 0));
  const out = mod._malloc(size);
  mod._llama_expand_u5(blob, out);
  mod._free(blob);
  blob = out;
  fmt = 6;
}
const MAXCTX = 2048;
const rc = mod._llama_init(blob, fmt, MAXCTX);
if (rc) throw new Error(`llama_init ${rc}`);
const V = mod._llama_dim(0);
const idsPtr = mod._malloc(4 * 256), lg = mod._malloc(4 * V * 200);
const fwd = (ids, pos0, all = false) => {
  mod.HEAP32.set(ids, idsPtr >> 2);
  const r = mod._llama_forward(blob, idsPtr, ids.length, pos0, lg, all ? 1 : 0);
  if (r) throw new Error(`llama_forward ${r}`);
  return mod.HEAPF32.subarray(lg >> 2, (lg >> 2) + V * (all ? ids.length : 1));
};
const argmax = (row) => { let b = 0; for (let i = 1; i < row.length; i++) if (row[i] > row[b]) b = i; return b; };
const logsumexp = (row) => { let m = -Infinity; for (const x of row) if (x > m) m = x; let s = 0; for (const x of row) s += Math.exp(x - m); return m + Math.log(s); };
const EOS = new Set([tokJson.eos, tokJson.specials['<|im_end|>']].filter((x) => x !== undefined));

for (const [ci, c] of ref.quant.chats.entries()) {
  let row = null;
  for (let p = 0; p < c.ids.length; p += 256) row = fwd(c.ids.slice(p, p + 256), p);
  let maxd = 0;
  c.top.forEach((id, k) => { maxd = Math.max(maxd, Math.abs(row[id] - c.top_logits[k])); });
  const gen = [];
  let next = argmax(row), pos = c.ids.length;
  const t0 = performance.now();
  while (gen.length < c.greedy.length) { gen.push(next); if (EOS.has(next) && gen.length >= c.greedy.length) break; row = fwd([next], pos++); next = argmax(row); }
  const ms = performance.now() - t0;
  let same = 0; while (same < c.greedy.length && gen[same] === c.greedy[same]) same++;
  console.log(`chat ${ci}: top-20 logits max|d| ${maxd.toFixed(4)} (max |logit| ${c.maxabs.toFixed(1)}); greedy == torch for ${same}/${c.greedy.length} tokens; ${(gen.length / ms * 1000).toFixed(0)} tok/s`);
  console.log(`  ${JSON.stringify(dec.decode(bpe.decodeBytes(gen)).slice(0, 120))}`);
}

const long = ref.quant.long;
const n = long.ids.length;
for (const mode of ['decode', 'prefill']) {
  let agree = 0, maxlp = 0;
  const cps = {};
  const t0 = performance.now();
  const check = (row, i) => {
    if (argmax(row) === long.argmax[i]) agree++;
    if (i < n - 1) maxlp = Math.max(maxlp, Math.abs(row[long.ids[i + 1]] - logsumexp(row) - long.next_logprob[i]));
    if (long.checkpoints[i]) { let d = 0; long.checkpoints[i].forEach((x, k) => { d = Math.max(d, Math.abs(row[k] - x)); }); cps[i] = d.toFixed(4); }
  };
  if (mode === 'decode') {
    for (let i = 0; i < n; i++) check(fwd([long.ids[i]], i), i);
  } else {
    for (let p = 0; p < n; p += 200) {
      const chunk = long.ids.slice(p, p + 200), rows = fwd(chunk, p, true);
      for (let r = 0; r < chunk.length; r++) check(rows.subarray(r * V, (r + 1) * V), p + r);
    }
  }
  const s = (performance.now() - t0) / 1000;
  console.log(`long ${mode} (${n} tokens): argmax == torch ${agree}/${n}; max |d next-token logprob| ${maxlp.toFixed(4)}; checkpoint max|d logit| ${JSON.stringify(cps)}; ${(n / s).toFixed(0)} tok/s`);
}
// Perplexity of the long text, kernel vs torch (quantized and fp32).
const ppl = (lps) => Math.exp(-lps.reduce((a, b) => a + b, 0) / lps.length);
console.log(`long-text perplexity: torch fp32 ${ppl(ref.fp32.long.next_logprob).toFixed(4)}, torch u${BITS} ${ppl(long.next_logprob).toFixed(4)}`);
