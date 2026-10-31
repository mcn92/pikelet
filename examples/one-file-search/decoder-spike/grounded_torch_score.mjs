// node grounded_torch_score.mjs <name> <variant> -- scores grounded_ref.py's
// torch replies like grounded.mjs and checks that bpe.mjs tokenizes every
// prompt to the same ids as Hugging Face's chat template.
import { readFileSync } from 'node:fs';
import { createBpeFromJson } from './bpe.mjs';
import { chatPrompt, score, summarize } from './grounded_score.mjs';

const HERE = new URL('.', import.meta.url);
const [NAME, VARIANT] = process.argv.slice(2);
const items = new Map(JSON.parse(readFileSync(new URL('data/grounded_items.json', HERE), 'utf8')).map((i) => [i.id, i]));
const bpe = createBpeFromJson(JSON.parse(readFileSync(new URL(`llama-tokenizer/${NAME}.json`, HERE), 'utf8')));
const torch = JSON.parse(readFileSync(new URL(`grounded_torch-${NAME}-${VARIANT}.json`, HERE), 'utf8'));
let same = 0;
const rows = torch.map((t) => {
  if (bpe.encode(chatPrompt(items.get(t.id))).join() === t.ids.join()) same++;
  return { id: t.id, reply: t.reply, score: score(items.get(t.id), t.reply) };
});
console.log(`prompt ids identical to Hugging Face's chat template: ${same}/${torch.length}`);
console.log(JSON.stringify(summarize(rows)));
