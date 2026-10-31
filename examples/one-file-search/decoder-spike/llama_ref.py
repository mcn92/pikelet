"""Torch references for llama.cpp (llama_check.mjs): `python3 llama_ref.py
<hf id> <name> <bits>` loads the model with the blob's dequantized weights
(<name>-u<bits>.deq.pt from export_llama.py) and writes llama_ref-<name>-u<bits>.json:
HF token ids for tokenizer test strings; for chat prompts, the last-token
logits (top 20 and max |logit|) and 40 greedy tokens; and a teacher-forced
long text (up to 1000 tokens) with each position's argmax and log-prob of
the next token, plus every position's full logits for a few checkpoints.
The same run with the fp32 weights (no deq) is stored alongside for the
quantization gap."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL, NAME, BITS = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

TOKENIZER_TESTS = [
    'Hello, world!', 'The year 2024 had 366 days; pi is 3.14159.', "  leading spaces and\ttabs\n\nnewlines ",
    "It's what we've done, isn't it? They'll see.", 'naïve café — “quotes” and emoji 🎉👍', '<|im_start|>user\nHi<|im_end|>\n',
    'def f(x):\n    return x**2 + 10_000', 'ÄÖÜ ß Здравствуйте 你好 こんにちは', 'a1b22c333 99 bottles', '   ',
]
CHATS = [
    [{'role': 'user', 'content': 'What is the capital of France?'}],
    [{'role': 'user', 'content': 'Write a haiku about the sea.'}],
    [{'role': 'user', 'content': 'Lily bought a yellow balloon and a box of crayons. On the way home, the balloon flew away. What color was the balloon that flew away?'}],
]
data = (HERE / 'data/TinyStories-valid.txt').read_text(encoding='utf-8')
LONG = data[:20000]


def run(model):
    out = {'chats': []}
    with torch.no_grad():
        for msgs in CHATS:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
            logits = model(torch.tensor([ids])).logits[0, -1].numpy()
            top = np.argsort(-logits)[:20]
            gen = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                 max_new_tokens=40, do_sample=False)[0, len(ids):].tolist()
            out['chats'].append({'ids': ids, 'top': top.tolist(), 'top_logits': logits[top].tolist(),
                                 'maxabs': float(np.abs(logits).max()), 'greedy': gen, 'text': tok.decode(gen)})
        ids = tok(LONG).input_ids[:1000]
        logits = model(torch.tensor([ids])).logits[0].numpy()
        lp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) - logits.max(1, keepdims=True)
        out['long'] = {'ids': ids, 'argmax': logits.argmax(1).tolist(),
                       'next_logprob': [float(lp[i, ids[i + 1]]) for i in range(len(ids) - 1)],
                       'checkpoints': {str(p): logits[p].tolist() for p in (0, 10, 100, 500, 999) if p < len(ids)}}
    return out


ref = {'model': MODEL, 'tokenizer': [{'text': t, 'ids': tok(t).input_ids} for t in TOKENIZER_TESTS]}
ref['fp32'] = run(model)
deq = torch.load(HERE / f'{NAME}-u{BITS}.deq.pt')
sd = model.state_dict()
with torch.no_grad():
    for k, v in deq.items():
        sd[k].copy_(v)
    if model.config.tie_word_embeddings:
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
ref['quant'] = run(model)
(HERE / f'llama_ref-{NAME}-u{BITS}.json').write_text(json.dumps(ref))
for a, b in zip(ref['fp32']['chats'], ref['quant']['chats']):
    print('fp32:', repr(a['text'][:80])); print('u   :', repr(b['text'][:80]))
