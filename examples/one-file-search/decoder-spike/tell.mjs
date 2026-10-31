// node tell.mjs check                 -- match the torch references
// node tell.mjs "prompt" [n] [temp]   -- generate (temp 0 = greedy)
import { readFileSync } from 'node:fs';
import { createBpe } from './bpe.mjs';
import createDecoder from './decoder.mjs';

const HERE = new URL('.', import.meta.url);
const V = 50257, EOS = 50256;
const bpe = createBpe(JSON.parse(readFileSync(new URL('tokenizer/vocab.json', HERE), 'utf8')),
  readFileSync(new URL('tokenizer/merges.txt', HERE), 'utf8'));
const mod = await createDecoder();
const blobBytes = readFileSync(new URL(process.env.DECODER_BLOB || 'tinystories-u8.bin', HERE));  // a u8 blob
const blob = mod._malloc(blobBytes.length);
mod.HEAPU8.set(blobBytes, blob);
const idsPtr = mod._malloc(4 * 256);
const logitsPtr = mod._malloc(4 * V);

function forward(ids, pos0) {
  mod.HEAP32.set(ids, idsPtr >> 2);
  const rc = mod._decoder_forward(blob, idsPtr, ids.length, pos0, logitsPtr, 8, 0);
  if (rc) throw new Error(`decoder_forward ${rc}`);
  return mod.HEAPF32.subarray(logitsPtr >> 2, (logitsPtr >> 2) + V);
}

function pick(logits, temp, topK = 40) {
  let best = 0;
  for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
  if (!temp) return best;
  const idx = [];  // top-k by insertion, no full sort
  for (let i = 0; i < V; i++) {
    if (idx.length === topK && logits[i] <= logits[idx[topK - 1]]) continue;
    let k = idx.length;
    while (k > 0 && logits[idx[k - 1]] < logits[i]) k--;
    idx.splice(k, 0, i);
    if (idx.length > topK) idx.pop();
  }
  const w = idx.map((i) => Math.exp((logits[i] - logits[best]) / temp));
  let r = Math.random() * w.reduce((a, b) => a + b);
  for (let k = 0; k < idx.length; k++) if ((r -= w[k]) <= 0) return idx[k];
  return idx[0];
}

function* generate(prompt, n, temp) {
  const ids = bpe.encode(prompt);
  let logits = forward(ids, 0);
  let pos = ids.length;
  for (let k = 0; k < n; k++) {
    const id = pick(logits, temp);
    if (id === EOS) return;
    yield id;
    logits = forward([id], pos++);
  }
}

if (process.argv[2] === 'check') {
  const refs = JSON.parse(readFileSync(new URL('refs.json', HERE), 'utf8'));
  const npy = (buf) => new Float32Array(buf.buffer.slice(buf.byteOffset + 128, buf.byteOffset + buf.length));  // 1-D f32, 128-byte header
  refs.prompts.forEach((r, pi) => {
    const ids = bpe.encode(r.text);
    const tokOk = ids.join() === r.ids.join();
    const ref = npy(readFileSync(new URL(`logits_u8_${pi}.npy`, HERE)));
    const got = forward(ids, 0);
    let maxd = 0;
    for (let i = 0; i < V; i++) maxd = Math.max(maxd, Math.abs(got[i] - ref[i]));
    const gen = [...generate(r.text, r.greedy_u8.length, 0)];
    const same = gen.filter((t, i) => t === r.greedy_u8[i]).length;
    let prefix = 0;
    while (prefix < gen.length && gen[prefix] === r.greedy_u8[prefix]) prefix++;
    console.log(`prompt ${pi}: tokenizer ${tokOk ? 'match' : 'MISMATCH'}; first-step logits max|diff| ${maxd.toFixed(4)} (max |logit| ${r.logits_u8_full_maxabs.toFixed(1)}); greedy vs torch-u8: identical prefix ${prefix}/${r.greedy_u8.length}`);
  });
} else {
  const prompt = process.argv[2] || 'Once upon a time,';
  const n = Number(process.argv[3] || 200), temp = Number(process.argv[4] ?? 0.8);
  const dec = new TextDecoder();
  process.stdout.write(prompt);
  const t0 = performance.now();
  let count = 0;
  for (const id of generate(prompt, n, temp)) {
    process.stdout.write(dec.decode(bpe.decodeBytes([id]), { stream: true }));
    count++;
  }
  const ms = performance.now() - t0;
  process.stdout.write(`\n\n[${count} tokens in ${(ms / 1000).toFixed(2)} s, ${(count / ms * 1000).toFixed(1)} tok/s]\n`);
}
