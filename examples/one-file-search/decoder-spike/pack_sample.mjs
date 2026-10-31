// node pack_sample.mjs -- samples candidate passages from the wiki pack for
// the pack-native QA set: seeded random record ids, keeping prose passages
// of 400-1500 characters (no stubs, lists or category residue), one per
// article title. Writes data/pack_candidates.json (id, title, text).
import { writeFileSync } from 'node:fs';
import { openPikeletFile } from '../../../packages/pikelet-wasm/complete/index.node.mjs';

const HERE = new URL('.', import.meta.url);
const pack = await openPikeletFile(new URL('data/wikipedia.pikelet', HERE).pathname);
const N = pack.info().records;
let s = 20261028;
const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32; };
const out = [], titles = new Set();
while (out.length < 140) {
  const id = Math.floor(rand() * N);
  const r = await pack.record(id);
  const text = String(r.text);
  const body = text.slice(text.indexOf(':') + 1);
  const sentences = body.split(/[.!?]\s/).length;
  if (text.length < 400 || text.length > 1500 || sentences < 4 || titles.has(r.title)) continue;
  if (/\b(births|deaths|Category|References)\b/.test(body.slice(-200)) && body.split('\n').length > 6) continue;
  titles.add(r.title);
  out.push({ id, title: r.title, text });
}
await pack.close();
writeFileSync(new URL('data/pack_candidates.json', HERE), JSON.stringify(out, null, 1));
console.log(out.length, 'candidates');
