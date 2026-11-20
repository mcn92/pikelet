// The answer page's model worker: loads a SmolLM2 blob into the decoder
// spike's Llama kernel and streams replies. Served unbundled from llm/ next
// to the kernel builds, which start their own threads from their own URL.
//
// Threads need a cross-origin-isolated page (SharedArrayBuffer); without one
// the single-threaded build runs the same code. The output is the same
// either way, at any thread count.
//
// Weights are fetched once and kept in the origin's private file system
// (OPFS) when the browser allows it. A copy counts only once its marker file
// records the full size, so an interrupted download is fetched again.
//
// Messages in:  {type:'load', name, url, size, tokenizerUrl, threads}
//               {type:'answer', system, passages, question, maxNew}
//               {type:'hash', ids, steps}   (tests: logits hash, as mt_check.mjs)
// Messages out: {type:'progress', phase, loaded, total}
//               {type:'ready', threads, isolated, source, saveError, loadMs}
//               {type:'token', text}
//               {type:'done', passages, readTokens, readMs, wrote, genMs}
//               {type:'hashed', sha256, prefillTokS, decodeTokS}
//               {type:'error', message}
import { createBpeFromJson } from './bpe.mjs';

// Three passages and a question ran to 919 tokens at most in the spike's
// test; 1,280 positions keep 1.7B's KV cache (393 KB a position) near 0.5 GB.
const MAXCTX = 1280;
const CHUNK = 256;  // prompt tokens per forward call
let mod, blob, bpe, stop, V, idsPtr, lg;

const post = (msg) => self.postMessage(msg);

async function cacheDir() {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle('models', { create: true });
  } catch {
    return null;
  }
}

async function cachedSize(dir, name) {
  try {
    const marker = await (await dir.getFileHandle(`${name}.ok`)).getFile();
    return Number(await marker.text());
  } catch {
    return -1;
  }
}

// Reads a cached copy into the heap at ptr, 16 MB at a time.
async function readCache(dir, name, ptr, size) {
  const file = await (await dir.getFileHandle(`${name}.bin`)).getFile();
  if (file.size !== size) return false;
  const STEP = 16 << 20;
  for (let off = 0; off < size; off += STEP) {
    const part = new Uint8Array(await file.slice(off, Math.min(size, off + STEP)).arrayBuffer());
    mod.HEAPU8.set(part, ptr + off);
    post({ type: 'progress', phase: 'cache', loaded: off + part.length, total: size });
  }
  return true;
}

// Streams the download into the heap at ptr, writing the cache alongside.
// Returns null once a copy is saved, or why none was.
async function download(url, dir, name, ptr, size) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  let handle = null, saveError = dir ? null : 'this browser gives the page no private file storage';
  const dropCopy = async (why) => {
    saveError = why;
    try { handle?.close(); } catch { /* already closed */ }
    handle = null;
    await dir.removeEntry(`${name}.bin`).catch(() => {});
  };
  if (dir) {
    try {
      await dir.removeEntry(`${name}.ok`).catch(() => {});
      await dir.removeEntry(`${name}.bin`).catch(() => {});
      const { quota = Infinity, usage = 0 } = await navigator.storage.estimate().catch(() => ({}));
      if (quota - usage < size) {
        saveError = `the browser allows this site ${Math.floor((quota - usage) / 2 ** 20)} MB of storage`;
      } else {
        handle = await (await dir.getFileHandle(`${name}.bin`, { create: true })).createSyncAccessHandle();
      }
    } catch (error) {
      await dropCopy(String(error?.message || error));
    }
  }
  const reader = res.body.getReader();
  let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (off + value.length > size) throw new Error(`${url}: larger than the expected ${size} bytes`);
    mod.HEAPU8.set(value, ptr + off);
    if (handle) {
      try {
        if (handle.write(value, { at: off }) !== value.length) throw new Error('short write');
      } catch (error) {
        await dropCopy(`writing failed at ${Math.floor(off / 2 ** 20)} MB: ${error?.message || error}`);
      }
    }
    off += value.length;
    post({ type: 'progress', phase: 'download', loaded: off, total: size });
  }
  if (off !== size) throw new Error(`${url}: got ${off} bytes, expected ${size}`);
  if (handle) {
    try {
      handle.flush();
      handle.close();
      // A sync handle, not createWritable(), which WebKit lacks.
      const marker = await (await dir.getFileHandle(`${name}.ok`, { create: true })).createSyncAccessHandle();
      marker.truncate(0);
      marker.write(new TextEncoder().encode(String(size)), { at: 0 });
      marker.flush();
      marker.close();
    } catch (error) {
      await dropCopy(String(error?.message || error));
    }
  }
  return saveError;
}

async function load({ name, url, size, tokenizerUrl, threads }) {
  const t0 = performance.now();
  const isolated = self.crossOriginIsolated === true;
  const { default: create } = await import(isolated ? './llama-mt.web.mjs' : './llama.web.mjs');
  const tokJson = await (await fetch(tokenizerUrl)).json();
  bpe = createBpeFromJson(tokJson);
  stop = new Set([tokJson.eos, tokJson.specials['<|im_end|>']].filter((x) => x !== undefined));
  mod = await create();
  const n = isolated ? mod._llama_set_threads(threads) : 1;
  // Pointers can pass 2 GB: read them unsigned and index the heap with >>>.
  blob = mod._malloc(size) >>> 0;
  if (!blob) throw new Error(`could not allocate ${size} bytes for the weights`);
  const dir = await cacheDir();
  let source = 'network', saveError = null;
  if (dir && (await cachedSize(dir, name)) === size && (await readCache(dir, name, blob, size).catch(() => false))) {
    source = 'cache';
  } else {
    saveError = await download(url, dir, name, blob, size);
  }
  const rc = mod._llama_init(blob, 8, MAXCTX);
  if (rc) throw new Error(`llama_init returned ${rc}: not a usable weights file`);
  V = mod._llama_dim(0);
  idsPtr = mod._malloc(4 * CHUNK) >>> 0;
  lg = mod._malloc(4 * V) >>> 0;
  post({ type: 'ready', threads: n, isolated, source, saveError, loadMs: performance.now() - t0 });
}

// Feeds ids at positions pos.., CHUNK at a time; returns the last logits row.
function feed(ids, pos) {
  if (pos + ids.length > MAXCTX) throw new Error(`the prompt passes ${MAXCTX} tokens`);
  for (let i = 0; i < ids.length; i += CHUNK) {
    const part = ids.slice(i, i + CHUNK);
    mod.HEAP32.set(part, idsPtr >>> 2);
    const r = mod._llama_forward(blob, idsPtr, part.length, pos + i, lg, 0);
    if (r) throw new Error(`llama_forward returned ${r}`);
  }
  return mod.HEAPF32.subarray(lg >>> 2, (lg >>> 2) + V);
}

function argmax(row) {
  let b = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[b]) b = i;
  return b;
}

// SmolLM2's chat template, with the passages numbered as rag_ref.py has them.
function promptIds(system, passages, question) {
  const body = passages.map((p, i) => `[${i + 1}] ${p}`).join('\n\n');
  return bpe.encode(`<|im_start|>system\n${system}<|im_end|>\n`
    + `<|im_start|>user\nPassages:\n${body}\n\nQuestion: ${question}<|im_end|>\n<|im_start|>assistant\n`);
}

// Greedy reply from an empty cache. Passages are dropped from the end until
// the prompt and reply fit the context.
function answer({ system, passages, question, maxNew = 128 }) {
  let kept = passages.length;
  let ids = promptIds(system, passages, question);
  while (ids.length + maxNew > MAXCTX && kept > 1) ids = promptIds(system, passages.slice(0, --kept), question);
  let t = performance.now();
  let row = feed(ids, 0);
  const readMs = performance.now() - t;
  const utf8 = new TextDecoder('utf-8');
  t = performance.now();
  let pos = ids.length, n = 0;
  for (; n < maxNew && pos < MAXCTX; n++) {
    const id = argmax(row);
    if (stop.has(id)) break;
    const text = utf8.decode(bpe.decodeBytes([id]), { stream: true });
    if (text) post({ type: 'token', text });
    row = feed([id], pos++);
  }
  const tail = utf8.decode();
  if (tail) post({ type: 'token', text: tail });
  post({ type: 'done', passages: kept, readTokens: ids.length, readMs, wrote: n, genMs: performance.now() - t });
}

// mt_check.mjs's procedure: ids in one call, then greedy tokens one per call;
// SHA-256 over every logits row.
async function hash({ ids, steps }) {
  const rows = [];
  let t = performance.now();
  let row = feed(ids, 0);
  const prefillS = (performance.now() - t) / 1000;
  rows.push(row.slice());
  t = performance.now();
  for (let i = 0; i < steps; i++) {
    row = feed([argmax(row)], ids.length + i);
    rows.push(row.slice());
  }
  const decodeS = (performance.now() - t) / 1000;
  const all = new Uint8Array(rows.length * V * 4);
  rows.forEach((r, i) => all.set(new Uint8Array(r.buffer), i * V * 4));
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', all));
  const sha256 = [...digest].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
  post({ type: 'hashed', sha256, prefillTokS: ids.length / prefillS, decodeTokS: steps / decodeS });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') await load(data);
    else if (data.type === 'answer') answer(data);
    else if (data.type === 'hash') await hash(data);
  } catch (error) {
    post({ type: 'error', message: String(error?.message || error) });
  }
};
