// node pragmatic.mjs -- can the decoder produce meaningful text? Runs the 40
// prompts of pragmatic_prompts.json (grammar, narrative, instruction,
// factual, grounded) through TinyStories-33M and TinyStories-Instruct-33M
// (u8, exact head, greedy, from <|endoftext|>), applies the automatic checks
// (answer keywords in the first quoted reply, required words, dialogue,
// required sentence) and writes pragmatic_results.json for judging.
//   PRAG_CONFIG=u5     u5 weights (expanded at load, bit-identical to u5)
//   PRAG_CONFIG=u5-sl  u5 plus the r192 K256 vocabulary shortlist
// write pragmatic_results-<config>.json instead; the default is u8 with the
// full head.
import { readFileSync, writeFileSync } from 'node:fs';
import createDecoder from './decoder.mjs';
import { createBpe } from './bpe.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, EOS = 50256, MAXSEQ = 256;
const CONFIG = process.env.PRAG_CONFIG || 'u8';
if (!['u8', 'u5', 'u5-sl'].includes(CONFIG)) throw new Error(`PRAG_CONFIG must be u8, u5 or u5-sl, got ${CONFIG}`);
const BITS = CONFIG === 'u8' ? 8 : 6;
const MODELS = CONFIG === 'u8'
  ? { base: 'tinystories-u8.bin', instruct: 'tinystories-instruct-u8.bin' }
  : { base: 'tinystories-u5.bin', instruct: 'tinystories-instruct-u5.bin' };
const SHORTLISTS = { base: 'shortlist-r192.bin', instruct: 'instruct-shortlist-r192.bin' };
const { prompts } = JSON.parse(readFileSync(new URL('pragmatic_prompts.json', HERE), 'utf8'));
const bpe = createBpe(JSON.parse(readFileSync(new URL('tokenizer/vocab.json', HERE), 'utf8')),
  readFileSync(new URL('tokenizer/merges.txt', HERE), 'utf8'));
const dec = new TextDecoder();

// The reply a question gets: the generated text up to its closing quote.
const reply = (text) => text.split('"')[0].trim();
const usesWord = (text, w) => new RegExp(`\\b${w}`, 'i').test(text);

const results = prompts.map((p) => ({ ...p, out: {} }));
for (const [model, file] of Object.entries(MODELS)) {
  const mod = await createDecoder();
  const bytes = readFileSync(new URL(file, HERE));
  let blob = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, blob);
  if (BITS === 6) {
    const raw = blob;
    blob = mod._malloc(Number(mod._decoder_expand_u5(raw, 0)));
    mod._decoder_expand_u5(raw, blob);
    mod._free(raw);
  }
  if (CONFIG === 'u5-sl') {
    const sl = readFileSync(new URL(SHORTLISTS[model], HERE));
    const slPtr = mod._malloc(sl.length);
    mod.HEAPU8.set(sl, slPtr);
    if (mod._decoder_set_shortlist(slPtr, 192, 256)) throw new Error('set_shortlist');
  }
  const idsPtr = mod._malloc(4 * MAXSEQ), lg = mod._malloc(4 * V);
  const fwd = (ids, pos0) => {
    mod.HEAP32.set(ids, idsPtr >> 2);
    if (mod._decoder_forward(blob, idsPtr, ids.length, pos0, lg, BITS, 0)) throw new Error('forward');
    const r = mod.HEAPF32.subarray(lg >> 2, (lg >> 2) + V);
    let b = 0; for (let i = 1; i < V; i++) if (r[i] > r[b]) b = i;
    return b;
  };
  for (const r of results) {
    const ids = [EOS, ...bpe.encode(r.prompt)];
    let next;
    for (let p = 0; p < ids.length; p += MAXSEQ) next = fwd(ids.slice(p, p + MAXSEQ), p);
    const out = [];
    let pos = ids.length;
    while (out.length < r.max && next !== EOS) {
      out.push(next);
      if (r.stopNewline && dec.decode(bpe.decodeBytes(out)).trim().includes('\n')) break;
      next = fwd([next], pos++);
    }
    let text = dec.decode(bpe.decodeBytes(out));
    if (r.stopNewline) text = text.trim().split('\n')[0];
    const checks = {};
    if (r.answer) checks.answer = r.answer.some((a) => usesWord(reply(text), a));
    if (r.words) checks.words = r.words.filter((w) => usesWord(text, w)).length + '/' + r.words.length;
    if (r.dialogue) checks.dialogue = /"[^"]+"/.test(text);
    if (r.contains) checks.contains = text.toLowerCase().includes(r.contains);
    r.out[model] = { text, checks, ...(r.answer ? { reply: reply(text) } : {}) };
  }
  console.log(`${model} ${CONFIG}: done`);
}
writeFileSync(new URL(CONFIG === 'u8' ? 'pragmatic_results.json' : `pragmatic_results-${CONFIG}.json`, HERE), JSON.stringify(results, null, 1));
for (const cat of ['factual', 'grounded']) {
  for (const m of Object.keys(MODELS)) {
    const rs = results.filter((r) => r.cat === cat);
    console.log(`${cat} ${m}: ${rs.filter((r) => r.out[m].checks.answer).length}/${rs.length} correct by answer key`);
  }
}
