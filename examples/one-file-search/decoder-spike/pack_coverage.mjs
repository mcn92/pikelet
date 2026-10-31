// node pack_coverage.mjs -- how much of the grounded-QA set the wiki pack
// (data/wikipedia.pikelet, Simple English Wikipedia) can answer: for each
// question, hybrid retrieval top 10 (showAbstained), the pack's matchQuality,
// and the rank of the first passage containing a gold answer (normalized as
// grounded_score.mjs). Writes data/pack_retrieval.json and prints how many
// gold answers the pack surfaces in the top 1, 3, 5 and 10.
import { writeFileSync, readFileSync } from 'node:fs';
import { openPikeletFile } from '../../../packages/pikelet-wasm/complete/index.node.mjs';

const HERE = new URL('.', import.meta.url);
const items = JSON.parse(readFileSync(new URL('data/grounded_items.json', HERE), 'utf8'));
const pack = await openPikeletFile(new URL('data/wikipedia.pikelet', HERE).pathname);
const norm = (s) => ` ${s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\b(a|an|the)\b/g, ' ').split(/\s+/).filter(Boolean).join(' ')} `;
const out = [];
const t0 = performance.now();
for (const it of items) {
  const r = await pack.query(it.question, { k: 10, showAbstained: true });
  const passages = r.results.map((x) => ({ id: x.id, title: x.title, text: x.text }));
  const golds = it.answers.map((a) => norm(a).trim()).filter(Boolean);
  const goldRank = passages.findIndex((p) => golds.some((g) => norm(p.text).includes(` ${g} `)));
  out.push({ id: it.id, answerable: it.answers.length > 0, matchQuality: r.matchQuality, confidence: r.confidence, goldRank, passages });
}
const ms = performance.now() - t0;
await pack.close();
writeFileSync(new URL('data/pack_retrieval.json', HERE), JSON.stringify(out));
const a = out.filter((x) => x.answerable), u = out.filter((x) => !x.answerable);
const at = (rows, k) => rows.filter((x) => x.goldRank >= 0 && x.goldRank < k).length;
const mq = (rows) => ['strong', 'weak', 'none'].map((q) => `${q} ${rows.filter((x) => x.matchQuality === q).length}`).join(', ');
console.log(`answerable (${a.length}): gold answer in the pack's top 1 / 3 / 5 / 10: ${at(a, 1)} / ${at(a, 3)} / ${at(a, 5)} / ${at(a, 10)}`);
console.log(`matchQuality, answerable: ${mq(a)}; unanswerable: ${mq(u)}`);
console.log(`answerable with gold in top 5, by matchQuality: ${['strong', 'weak', 'none'].map((q) => `${q} ${at(a.filter((x) => x.matchQuality === q), 5)}/${a.filter((x) => x.matchQuality === q).length}`).join(', ')}`);
console.log(`${(ms / items.length).toFixed(0)} ms per query`);
