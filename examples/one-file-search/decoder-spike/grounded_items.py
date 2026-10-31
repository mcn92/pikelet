"""Grounded-QA items for grounded.mjs: 100 questions from the SQuAD v2
validation set (CC BY-SA 4.0, Wikipedia paragraphs), 70 answerable and 30
unanswerable (the paragraph is on topic but does not contain the answer),
seeded, paragraphs of at most 220 words. Writes data/grounded_items.json."""
import json
import random
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
ds = load_dataset('rajpurkar/squad_v2', split='validation')
rng = random.Random(20261028)
idx = list(range(len(ds)))
rng.shuffle(idx)
items, ans, unans, seen = [], 0, 0, set()
for i in idx:
    r = ds[i]
    if len(r['context'].split()) > 220 or r['context'] in seen:
        continue
    answerable = len(r['answers']['text']) > 0
    if answerable and ans < 70:
        ans += 1
    elif not answerable and unans < 30:
        unans += 1
    else:
        continue
    seen.add(r['context'])
    items.append({'id': r['id'], 'title': r['title'], 'context': r['context'], 'question': r['question'],
                  'answers': sorted(set(r['answers']['text']))})
    if ans == 70 and unans == 30:
        break
(HERE / 'data').mkdir(exist_ok=True)
(HERE / 'data/grounded_items.json').write_text(json.dumps(items, indent=1))
print(len(items), 'items;', ans, 'answerable,', unans, 'unanswerable')
