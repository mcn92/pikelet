#!/usr/bin/env node
// Writes the answer page's model manifest: for each published weights file,
// its size and the SHA-256 of every CHUNK-byte piece, and the tokenizer's
// SHA-256. The page serves the manifest from its own origin, and the worker
// checks every piece it downloads against it before the kernel reads it or a
// copy is saved, so the weights host cannot change what the page runs.
//
//   node model_manifest.mjs --out <manifest.json> \
//     --tokenizer <local.json> <published name> \
//     --model <page name> <local .bin> <published name> [--model ...]
import { createHash } from 'node:crypto';
import { openSync, readSync, closeSync, fstatSync, readFileSync, writeFileSync } from 'node:fs';

const CHUNK = 16 << 20;
const args = process.argv.slice(2);
let out = null, tokenizer = null;
const models = {};
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--out') out = args[++i];
  else if (args[i] === '--tokenizer') tokenizer = { local: args[++i], file: args[++i] };
  else if (args[i] === '--model') models[args[++i]] = { local: args[++i], file: args[++i] };
  else throw new Error(`unknown argument ${args[i]}`);
}
if (!out || !tokenizer || !Object.keys(models).length) {
  throw new Error('usage: model_manifest.mjs --out <json> --tokenizer <local> <name> --model <page name> <local> <name> ...');
}

const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');

function chunkHashes(path) {
  const fd = openSync(path, 'r');
  try {
    const size = fstatSync(fd).size;
    const buf = Buffer.alloc(CHUNK);
    const whole = createHash('sha256');
    const chunks = [];
    for (let off = 0; off < size; off += CHUNK) {
      const n = Math.min(CHUNK, size - off);
      let got = 0;
      while (got < n) got += readSync(fd, buf, got, n - got, off + got);
      const part = buf.subarray(0, n);
      chunks.push(sha(part));
      whole.update(part);
    }
    return { size, sha256: whole.digest('hex'), chunks };
  } finally {
    closeSync(fd);
  }
}

const manifest = {
  chunkSize: CHUNK,
  tokenizer: { file: tokenizer.file, sha256: sha(readFileSync(tokenizer.local)) },
  models: {},
};
for (const [name, { local, file }] of Object.entries(models)) {
  manifest.models[name] = { file, ...chunkHashes(local) };
  console.log(`${name}: ${file}, ${manifest.models[name].size} bytes, ${manifest.models[name].chunks.length} chunks, sha256 ${manifest.models[name].sha256}`);
}
writeFileSync(out, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`wrote ${out}`);
