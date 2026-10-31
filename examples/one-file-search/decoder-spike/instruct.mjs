// DECODER_BLOB=tinystories-instruct-u8.bin node instruct.mjs -- the kernel
// running TinyStories-Instruct-33M (u8) on instruct_items.json
// (instruct_ref.py), greedy, against torch fp32's greedy outputs:
// required-word use in written stories, ROUGE-1 F1 of one-line summaries
// against the dataset's reference, and how often the text matches torch's.
import { readFileSync } from 'node:fs';
import createDecoder from './decoder.mjs';
import { createBpe } from './bpe.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, EOS = 50256, MAXSEQ = 256;
const { words, summary } = JSON.parse(readFileSync(new URL('instruct_items.json', HERE), 'utf8'));
const bpe = createBpe(JSON.parse(readFileSync(new URL('tokenizer/vocab.json', HERE), 'utf8')),
  readFileSync(new URL('tokenizer/merges.txt', HERE), 'utf8'));
const mod = await createDecoder();
const bytes = readFileSync(new URL(process.env.DECODER_BLOB || 'tinystories-instruct-u8.bin', HERE));
const blob = mod._malloc(bytes.length);
mod.HEAPU8.set(bytes, blob);
const idsPtr = mod._malloc(4 * MAXSEQ), lg = mod._malloc(4 * V);
const fwd = (ids, pos0) => {
  mod.HEAP32.set(ids, idsPtr >> 2);
  if (mod._decoder_forward(blob, idsPtr, ids.length, pos0, lg, 8, 0)) throw new Error('forward');
  const r = mod.HEAPF32.subarray(lg >> 2, (lg >> 2) + V);
  let b = 0; for (let i = 1; i < V; i++) if (r[i] > r[b]) b = i;
  return b;
};
const dec = new TextDecoder();
let readMs = 0, readTok = 0, genMs = 0, genTok = 0;
function greedy(ids, n, stopNewline) {
  let t = performance.now(), next;
  for (let p = 0; p < ids.length; p += MAXSEQ) next = fwd(ids.slice(p, p + MAXSEQ), p);
  readMs += performance.now() - t; readTok += ids.length;
  t = performance.now();
  const out = [];
  let pos = ids.length;
  for (let k = 0; k < n && next !== EOS; k++) {
    out.push(next);
    if (stopNewline && dec.decode(bpe.decodeBytes([next])).includes('\n')) break;
    next = fwd([next], pos++);
  }
  genMs += performance.now() - t; genTok += out.length;
  const text = dec.decode(bpe.decodeBytes(out));
  return stopNewline ? text.split('\n')[0] : text;
}
const used = (text, ws) => ws.filter((w) => new RegExp(`\\b${w.toLowerCase().replace(/[^a-z]/g, '')}`, 'i').test(text)).length;
const toks = (s) => s.toLowerCase().match(/[a-z']+/g) || [];
function rouge1(a, b) {
  const A = toks(a), B = toks(b), cnt = new Map();
  B.forEach((w) => cnt.set(w, (cnt.get(w) || 0) + 1));
  let hit = 0;
  A.forEach((w) => { if (cnt.get(w)) { hit++; cnt.set(w, cnt.get(w) - 1); } });
  return hit ? 2 * hit / (A.length + B.length) : 0;
}

let wk = 0, wt = 0, wTotal = 0, wSame = 0;
for (const it of words) {
  const text = greedy(it.ids, 250, false);
  wk += used(text, it.words); wt += used(it.torch, it.words); wTotal += it.words.length;
  wSame += text.trim() === it.torch.trim();
}
let rk = 0, rt = 0, sSame = 0;
const examples = [];
for (const it of summary) {
  const text = greedy(it.ids, 60, true).trim();
  rk += rouge1(text, it.reference); rt += rouge1(it.torch.trim(), it.reference);
  sSame += text === it.torch.trim();
  if (examples.length < 3) examples.push({ kernel: text, reference: it.reference });
}
console.log(`words: required words used, kernel ${wk}/${wTotal} (${(100 * wk / wTotal).toFixed(1)}%), torch fp32 ${wt}/${wTotal} (${(100 * wt / wTotal).toFixed(1)}%); story identical to torch ${wSame}/${words.length}`);
console.log(`summary: ROUGE-1 F1 vs reference, kernel ${(rk / summary.length).toFixed(3)}, torch fp32 ${(rt / summary.length).toFixed(3)}; summary identical to torch ${sSame}/${summary.length}`);
console.log(`speed: prompt reading ${(readTok / readMs * 1000).toFixed(0)} tok/s (${readTok} tokens), generation ${(genTok / genMs * 1000).toFixed(0)} tok/s (${genTok} tokens)`);
examples.forEach((e, i) => console.log(`example ${i}: kernel: ${e.kernel}\n           reference: ${e.reference}`));
