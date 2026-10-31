// SQuAD-style scoring shared by grounded.mjs and the torch reference:
// normalized exact match, token F1 (max over gold answers), whether the
// reply contains a gold answer, and abstention detection for unanswerable
// items.
export const SYSTEM = 'You answer questions using only the given passage. Reply with a short answer taken from the passage. '
  + 'If the passage does not contain the answer, reply exactly: Not in the passage.';
export const chatPrompt = (item) => `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n`
  + `Passage:\n${item.context}\n\nQuestion: ${item.question}<|im_end|>\n<|im_start|>assistant\n`;

const norm = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\b(a|an|the)\b/g, ' ').split(/\s+/).filter(Boolean);
const ABSTAIN = /not in the passage|does not (say|mention|contain|specify|state)|doesn't (say|mention)|no information|not mentioned|cannot be determined|not (stated|specified|given)/i;

export function score(item, reply) {
  const abstained = ABSTAIN.test(reply);
  if (!item.answers.length) return { answerable: false, abstained, correct: abstained, em: 0, f1: 0, contains: false };
  const r = norm(reply);
  let em = 0, f1 = 0, contains = false;
  for (const gold of item.answers) {
    const g = norm(gold);
    if (r.join(' ') === g.join(' ')) em = 1;
    if (` ${r.join(' ')} `.includes(` ${g.join(' ')} `)) contains = true;
    const counts = new Map();
    for (const t of g) counts.set(t, (counts.get(t) || 0) + 1);
    let common = 0;
    for (const t of r) if (counts.get(t) > 0) { common++; counts.set(t, counts.get(t) - 1); }
    if (common) f1 = Math.max(f1, (2 * common) / (r.length + g.length));
  }
  return { answerable: true, abstained, correct: contains && !abstained, em, f1, contains };
}

export function summarize(rows) {
  const a = rows.filter((x) => x.score.answerable), u = rows.filter((x) => !x.score.answerable);
  const pct = (n, d) => `${(100 * n / d).toFixed(1)}%`;
  return {
    answerable: a.length, unanswerable: u.length,
    containsGold: pct(a.filter((x) => x.score.contains && !x.score.abstained).length, a.length),
    exactMatch: pct(a.filter((x) => x.score.em).length, a.length),
    f1: (a.reduce((s, x) => s + x.score.f1, 0) / a.length).toFixed(3),
    falseAbstain: pct(a.filter((x) => x.score.abstained).length, a.length),
    abstainOnUnanswerable: pct(u.filter((x) => x.score.abstained).length, u.length),
  };
}
