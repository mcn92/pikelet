// node mt_check.mjs <module.mjs> <threads> [name] [bits] [ref name] -- one chat prompt
// read in one call, then 64 greedy tokens one per call; prints a SHA-256
// over every logit row (bit-identity across builds and thread counts) and
// both speeds, and the generated text. One configuration per process, so
// modules don't share memory. [ref name] takes the prompt from another
// model's llama_ref file with the same tokenizer (no torch reference needed).
import { readFileSync } from 'node:fs';
import { loadBlob } from './load_blob.mjs';
import { createHash } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
import { createBpeFromJson } from './bpe.mjs';

const [MOD = './llama.mjs', THREADS = '1', NAME = 'smollm2-360m', BITS = '8', REF = NAME] = process.argv.slice(2);
const HERE = new URL('.', import.meta.url);
const { default: create } = await import(pathToFileURL(path.resolve(MOD)).href);
const mod = await create();
const threads = mod._llama_set_threads ? mod._llama_set_threads(Number(THREADS)) : 1;
const ref = JSON.parse(readFileSync(new URL(`llama_ref-${REF}-u${BITS === '5' ? 8 : BITS}.json`, HERE), 'utf8'));
// Pointers can pass 2 GB with a large blob: read them unsigned (>>> 0) and
// index the heap with >>>, never >>. The blob is read straight into the
// module's memory (load_blob.mjs), with no JavaScript copy kept.
let blob = loadBlob(mod, fileURLToPath(new URL(`${NAME}-u${BITS}.bin`, HERE))).ptr;
let fmt = 8;
if (BITS === '5') {
  const size = Number(mod._llama_expand_u5(blob, 0));
  const out = mod._malloc(size) >>> 0;
  mod._llama_expand_u5(blob, out);
  mod._free(blob);
  blob = out;
  fmt = 6;
}
const rc = mod._llama_init(blob, fmt, 512);
if (rc) throw new Error(`llama_init ${rc}`);
const V = mod._llama_dim(0);
const idsPtr = mod._malloc(4 * 256) >>> 0, lg = mod._malloc(4 * V) >>> 0;
const hash = createHash('sha256');
const fwd = (ids, pos0) => {
  mod.HEAP32.set(ids, idsPtr >>> 2);
  const r = mod._llama_forward(blob, idsPtr, ids.length, pos0, lg, 0);
  if (r) throw new Error(`llama_forward ${r}`);
  const row = mod.HEAPF32.slice(lg >>> 2, (lg >>> 2) + V);
  hash.update(Buffer.from(row.buffer));
  return row;
};
const argmax = (row) => { let b = 0; for (let i = 1; i < row.length; i++) if (row[i] > row[b]) b = i; return b; };
// The longest chat prompt, padded by repetition to 256 tokens.
const base = ref.quant.chats.map((c) => c.ids).sort((a, b) => b.length - a.length)[0];
const prompt = [];
while (prompt.length < 256) prompt.push(...base);
prompt.length = 256;
let t0 = performance.now();
let row = fwd(prompt, 0);
const prefillS = (performance.now() - t0) / 1000;
const N = 64;
const gen = [];
t0 = performance.now();
for (let i = 0; i < N; i++) { gen.push(argmax(row)); row = fwd([gen[i]], 256 + i); }
const decodeS = (performance.now() - t0) / 1000;
console.log(JSON.stringify({ module: path.basename(MOD), threads, prefillTokS: +(256 / prefillS).toFixed(1), decodeTokS: +(N / decodeS).toFixed(1), sha256: hash.digest('hex').slice(0, 16) }));
const bpe = createBpeFromJson(JSON.parse(readFileSync(new URL(`llama-tokenizer/${NAME}.json`, HERE), 'utf8')));
if (process.env.SHOW_TEXT) console.log(JSON.stringify(new TextDecoder().decode(bpe.decodeBytes(gen))));
process.exit(0);
