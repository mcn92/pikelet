// node rag_score.mjs <rag_out file>... -- scores pack-native QA replies
// (rag_ref.py or kernel_run.mjs output) with no gate and with three gates
// applied after the fact:
//   verdict  answer "not in the pack" whenever the pack's matchQuality is none
//   support  also reject an answer whose content words (beyond the
//            question's) are not all in the passages the model was shown
//   names    also decline when a name or number in the question (a
//            capitalized word after the first, or a word with a digit) is in
//            none of the passages shown: a search result's grounding.uncovered,
//            computed client-side
// Answerable: correct = contains a gold answer and did not abstain.
// Unanswerable: correct = abstained.
import { readFileSync } from 'node:fs';

const HERE = new URL('.', import.meta.url);
const items = new Map(JSON.parse(readFileSync(new URL('data/pack_qa_items.json', HERE), 'utf8')).map((i) => [i.id, i]));
const ret = new Map(JSON.parse(readFileSync(new URL('data/pack_rag_retrieval.json', HERE), 'utf8')).map((r) => [r.id, r]));
const ABSTAIN = /not in the passage|i don'?t know|i do not know|does not (say|mention|contain|specify|state)|doesn't (say|mention)|no information|not mentioned|cannot be determined|not (stated|specified|given)/i;
const STOP = new Set('a an the of in on at to and or by for is was were are be been it its that this as with from what who whom which when where how many much did does do'.split(' '));
const words = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter((w) => w && !STOP.has(w));
const norm = (s) => ` ${s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\b(a|an|the)\b/g, ' ').split(/\s+/).filter(Boolean).join(' ')} `;

const toks = (s) => s.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
const names = (q) => toks(q).filter((t, i) => /\p{N}/u.test(t) || (i > 0 && /^\p{Lu}/u.test(t)));

export function grade(item, reply, shown, gate) {
  let abstained = ABSTAIN.test(reply);
  const r = ret.get(item.id);
  if (gate !== 'none' && r.matchQuality === 'none') abstained = true;
  if (gate === 'support' && !abstained && shown) {
    const ctx = new Set(words(shown)), q = new Set(words(item.question));
    if (!words(reply).filter((w) => !q.has(w)).every((w) => ctx.has(w))) abstained = true;
  }
  if (gate === 'names' && !abstained && shown) {
    const ctx = new Set(toks(shown).map((t) => t.toLowerCase()));
    if (names(item.question).some((t) => !ctx.has(t.toLowerCase()))) abstained = true;
  }
  if (item.kind !== 'answerable') return abstained ? `declined-ok-${item.subtype}` : 'made-up';
  if (abstained) return 'declined-wrongly';
  return item.answers.some((g) => norm(reply).includes(norm(g))) ? 'correct' : 'wrong';
}

const k = { closed: 0, rag1: 1, rag3: 3 };
for (const file of process.argv.slice(2)) {
  const cond = file.match(/-(closed|rag\d)\.json$/)[1];
  const rows = JSON.parse(readFileSync(new URL(file, HERE), 'utf8'));
  console.log(file);
  for (const gate of cond === 'closed' ? ['none'] : ['none', 'verdict', 'support', 'names']) {
    const c = {};
    for (const row of rows) {
      const it = items.get(row.id);
      const shown = k[cond] ? ret.get(row.id).passages.slice(0, k[cond]).map((p) => p.text).join(' ') : '';
      const g = grade(it, row.reply, shown, gate);
      c[g] = (c[g] || 0) + 1;
    }
    const ans = 70, un = 30;
    const missing = c['declined-ok-missing-detail'] || 0, fictional = c['declined-ok-fictional'] || 0;
    console.log(`  gate ${gate.padEnd(8)} answerable: correct ${c.correct || 0}/${ans}, wrong ${c.wrong || 0}, declined ${c['declined-wrongly'] || 0} | unanswerable: declined ${missing + fictional}/${un} (${missing} missing-detail, ${fictional} fictional), made up ${c['made-up'] || 0} | trustworthy ${(c.correct || 0) + missing + fictional}/100, confident errors ${(c.wrong || 0) + (c['made-up'] || 0)}`);
  }
}
