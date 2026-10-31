// node pack_rag_retrieve.mjs -- retrieval half of the pack-native QA test:
// each question in data/pack_qa_items.json against the wiki pack (hybrid,
// top 5, showAbstained so a 'none' verdict still shows what it found).
// Records the pack's matchQuality/confidence, the rank of the passage the
// question was written from, and the rank of the first passage containing a
// gold answer. Writes data/pack_rag_retrieval.json.
import { readFileSync, writeFileSync } from 'node:fs';
import { openPikeletFile } from '../../../packages/pikelet-wasm/complete/index.node.mjs';

const HERE = new URL('.', import.meta.url);
const items = JSON.parse(readFileSync(new URL('data/pack_qa_items.json', HERE), 'utf8'));
const pack = await openPikeletFile(new URL('data/wikipedia.pikelet', HERE).pathname);
const norm = (s) => ` ${s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\b(a|an|the)\b/g, ' ').split(/\s+/).filter(Boolean).join(' ')} `;
const out = [];
for (const it of items) {
  const r = await pack.query(it.question, { k: 5, showAbstained: true });
  const passages = r.results.map((x) => ({ id: x.id, title: x.title, text: x.text }));
  const golds = it.answers.map((a) => norm(a).trim()).filter(Boolean);
  out.push({
    id: it.id, matchQuality: r.matchQuality, confidence: r.confidence,
    sourceRank: passages.findIndex((p) => p.id === it.sourceId),
    goldRank: passages.findIndex((p) => golds.some((g) => norm(p.text).includes(` ${g} `))),
    passages,
  });
}
await pack.close();
writeFileSync(new URL('data/pack_rag_retrieval.json', HERE), JSON.stringify(out));
const a = out.filter((x, i) => items[i].kind === 'answerable'), u = out.filter((x, i) => items[i].kind !== 'answerable');
const at = (k) => a.filter((x) => x.goldRank >= 0 && x.goldRank < k).length;
const mq = (rows) => ['strong', 'weak', 'none'].map((q) => `${q} ${rows.filter((x) => x.matchQuality === q).length}`).join(', ');
console.log(`answerable (${a.length}): gold answer in top 1 / 3 / 5: ${at(1)} / ${at(3)} / ${at(5)}; source passage in top 5: ${a.filter((x) => x.sourceRank >= 0).length}`);
console.log(`matchQuality: answerable ${mq(a)}; unanswerable ${mq(u)}`);
for (const st of ['missing-detail', 'fictional']) {
  const rows = out.filter((x, i) => items[i].subtype === st);
  console.log(`  ${st}: ${mq(rows)}`);
}
