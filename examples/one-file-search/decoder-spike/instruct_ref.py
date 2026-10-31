"""Instruction-following items for TinyStories-Instruct-33M + torch fp32
greedy outputs, from data/TinyStories-Instruct-valid.txt (entries 1000+):
  words:   header lines (Words:, Features:, Summary:, ...) then "Story:"; the
           story must use the required words (max 250 tokens)
  summary: a story, then "Summary:"; one line summarizing it (max 60 tokens)
Writes instruct_items.json."""
import json, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
N = 50
tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-Instruct-33M').eval()
entries = (HERE / 'data/TinyStories-Instruct-valid.txt').read_text(encoding='utf-8').split('<|endoftext|>')[1000:]

words, summary = [], []
for e in entries:
    e = e.strip('\n')
    m = re.search(r'^Words: (.+)$', e, re.M)
    if m and 'Story:' in e and not re.search(r'Story:.*\nSummary:', e, re.S) and len(words) < N:
        head = e[:e.index('Story:')] + 'Story: \n'
        words.append({'prompt': head, 'words': [w.strip() for w in m.group(1).split(',')]})
    ms = re.search(r'^(.*Story:.*)\nSummary: (.+)$', e, re.S | re.M)
    if ms and len(summary) < N:
        prompt = ms.group(1) + '\nSummary:'
        if len(tok(prompt).input_ids) <= 900:
            summary.append({'prompt': prompt, 'reference': ms.group(2).strip().split('\n')[0]})
    if len(words) == N and len(summary) == N:
        break


def greedy(prompt, n, stop_newline):
    ids = [50256] + tok(prompt).input_ids
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                             max_new_tokens=n, do_sample=False, pad_token_id=50256)
    text = tok.decode(out[0, len(ids):], skip_special_tokens=True)
    return ids, (text.split('\n')[0] if stop_newline else text)


for it in words:
    it['ids'], it['torch'] = greedy(it['prompt'], 250, False)
for it in summary:
    it['ids'], it['torch'] = greedy(it['prompt'], 60, True)
(HERE / 'instruct_items.json').write_text(json.dumps({'words': words, 'summary': summary}))
print(f'{len(words)} words items, {len(summary)} summary items')
print('example summary prompt tail:', repr(summary[0]['prompt'][-200:]))
print('torch summary:', repr(summary[0]['torch']), '| reference:', repr(summary[0]['reference']))
