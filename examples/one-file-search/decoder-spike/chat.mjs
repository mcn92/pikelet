// Chat with a SmolLM2 blob on the threaded kernel (llama-mt.mjs).
//
//   node chat.mjs [--model smollm2-1.7b] [--threads n] [--temp t] [--seed s] ["question"]
//
// With a question, answers it and exits; without, a conversation on stdin
// (empty line or Ctrl-D ends it). The conversation stays in the KV cache, so
// each turn feeds only its new tokens. --temp 0 (the default) is greedy;
// above 0, tokens are sampled from the top-p 0.9 set with a seeded generator,
// so a run is repeatable. Needs <model>-u8.bin and llama-tokenizer/<model>.json
// (export_llama.py).
import os from 'node:os';
import readline from 'node:readline';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import createLlama from './llama-mt.mjs';
import { createBpeFromJson } from './bpe.mjs';
import { loadBlob } from './load_blob.mjs';

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  if (i < 0) return fallback;
  const [, value] = args.splice(i, 2);
  return value;
};
const NAME = opt('model', 'smollm2-1.7b');
const THREADS = Number(opt('threads', Math.min(6, Math.max(1, os.availableParallelism() - 2))));
const TEMP = Number(opt('temp', 0));
let seed = Number(opt('seed', 1)) >>> 0;
const question = args.join(' ').trim();
const MAXCTX = 2048, MAXNEW = 512;
const SYSTEM = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face';

const HERE = new URL('.', import.meta.url);
const tokJson = JSON.parse(readFileSync(new URL(`llama-tokenizer/${NAME}.json`, HERE), 'utf8'));
const bpe = createBpeFromJson(tokJson);
const STOP = new Set([tokJson.eos, tokJson.specials['<|im_end|>']].filter((x) => x !== undefined));

const mod = await createLlama();
const threads = mod._llama_set_threads(THREADS);
let t0 = performance.now();
const blob = loadBlob(mod, fileURLToPath(new URL(`${NAME}-u8.bin`, HERE))).ptr;
const rc = mod._llama_init(blob, 8, MAXCTX);
if (rc) throw new Error(`llama_init ${rc}`);
const V = mod._llama_dim(0);
const idsPtr = mod._malloc(4 * 256) >>> 0, lg = mod._malloc(4 * V) >>> 0;
process.stderr.write(`[${NAME}, ${threads} threads, loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s]\n`);

let pos = 0;  // positions already in the KV cache
// Feeds ids at the end of the cache, 256 at a time; returns the last logits.
function feed(ids) {
  if (pos + ids.length > MAXCTX) throw new Error(`the conversation would pass ${MAXCTX} tokens`);
  for (let i = 0; i < ids.length; i += 256) {
    const part = ids.slice(i, i + 256);
    mod.HEAP32.set(part, idsPtr >>> 2);
    const r = mod._llama_forward(blob, idsPtr, part.length, pos, lg, 0);
    if (r) throw new Error(`llama_forward ${r}`);
    pos += part.length;
  }
  return mod.HEAPF32.subarray(lg >>> 2, (lg >>> 2) + V);
}

// mulberry32: a small seeded generator, so sampled runs repeat.
function random() {
  seed = (seed + 0x6D2B79F5) >>> 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function pick(logits) {
  let best = 0;
  for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
  if (TEMP <= 0) return best;
  const order = [];
  const probs = new Float64Array(V);
  let sum = 0;
  for (let i = 0; i < V; i++) { probs[i] = Math.exp((logits[i] - logits[best]) / TEMP); sum += probs[i]; }
  for (let i = 0; i < V; i++) if (probs[i] / sum > 1e-6) order.push(i);
  order.sort((a, b) => probs[b] - probs[a]);
  let mass = 0, cut = 0;
  while (cut < order.length && mass < 0.9 * sum) mass += probs[order[cut++]];
  let r = random() * mass;
  for (let k = 0; k < cut; k++) { r -= probs[order[k]]; if (r <= 0) return order[k]; }
  return order[cut - 1];
}

// One turn: feed the user's message, stream the reply until an end token.
function turn(text, first) {
  const prompt = (first ? `<|im_start|>system\n${SYSTEM}<|im_end|>\n` : '')
    + `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
  const ids = bpe.encode(prompt);
  let t = performance.now();
  let logits = feed(ids);
  const readS = (performance.now() - t) / 1000;
  const utf8 = new TextDecoder('utf-8');
  t = performance.now();
  let n = 0;
  for (; n < MAXNEW; n++) {
    const id = pick(logits);
    if (STOP.has(id)) { feed([id]); break; }  // the end token stays in the cache, as the template has it
    process.stdout.write(utf8.decode(bpe.decodeBytes([id]), { stream: true }));
    logits = feed([id]);
  }
  process.stdout.write(utf8.decode() + '\n');
  feed(bpe.encode('\n'));
  const genS = (performance.now() - t) / 1000;
  process.stderr.write(`[read ${ids.length} tokens in ${readS.toFixed(1)} s; wrote ${n} at ${(n / genS).toFixed(1)} tok/s; ${pos}/${MAXCTX} context used]\n`);
}

if (question) {
  turn(question, true);
  process.exit(0);
}
const rl = readline.createInterface({ input: process.stdin, output: process.stderr, prompt: '> ' });
let first = true;
rl.prompt();
for await (const line of rl) {
  if (!line.trim()) break;
  try {
    turn(line.trim(), first);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    break;
  }
  first = false;
  rl.prompt();
}
process.exit(0);
