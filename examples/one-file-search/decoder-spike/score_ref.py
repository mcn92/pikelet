"""Scoring test items + torch fp32 reference: 200 cloze items from validation
stories 3000+ (disjoint from the perplexity and shortlist stories). Each item
is a story's first k sentences (k >= 2) and four candidate next sentences: the
real one and three from other items' stories. Writes score_items.json (token
ids) with torch's summed and per-token-mean log-likelihoods per candidate."""
import json, random, re
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M').eval()
stories = [s.strip() for s in (HERE / 'data/TinyStories-valid.txt').read_text(encoding='utf-8').split('<|endoftext|>')]
stories = [s for s in stories if s][3000:]
rng = random.Random(0)
split = lambda s: [x for x in re.split(r'(?<=[.!?])\s+', s.replace('\n', ' ')) if x]

raw = []
for s in stories:
    sents = split(s)
    if len(sents) < 4:
        continue
    k = rng.randint(2, min(4, len(sents) - 1))
    raw.append((' '.join(sents[:k]), sents[k], sents))
    if len(raw) == 200:
        break
items = []
for i, (prefix, true, _) in enumerate(raw):
    others = rng.sample([j for j in range(len(raw)) if j != i], 3)
    cands = [true] + [rng.choice(split(' '.join(raw[j][2][1:]))) for j in others]
    order = list(range(4)); rng.shuffle(order)
    items.append({'prefix': [50256] + tok(prefix).input_ids,
                  'cands': [tok(' ' + cands[o]).input_ids for o in order],
                  'answer': order.index(0)})


def score(prefix, cand):
    ids = torch.tensor([prefix + cand])
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0], -1)
    n = len(prefix)
    t = lp[torch.arange(n - 1, n - 1 + len(cand)), torch.tensor(cand)]
    return float(t.sum()), float(t.mean())


for it in items:
    s = [score(it['prefix'], c) for c in it['cands']]
    it['torch_sum'] = [x[0] for x in s]
    it['torch_mean'] = [x[1] for x in s]
acc = lambda key: np.mean([int(np.argmax(it[key]) == it['answer']) for it in items])
print(f'{len(items)} items; torch fp32 accuracy: summed {acc("torch_sum"):.3f}, per-token mean {acc("torch_mean"):.3f} (chance 0.25)')
(HERE / 'score_items.json').write_text(json.dumps(items))
