"""Torch side of the grounded-QA test: `python3 grounded_ref.py <hf id> <name>
<fp32|8|5>` answers data/grounded_items.json with the fp32 model or with the
blob's dequantized weights (<name>-u<bits>.deq.pt), greedy, 48 tokens, on
the prompt grounded.mjs builds (the system text is read from
grounded_score.mjs so the two cannot drift). Writes
grounded_torch-<name>-<variant>.json: each item's prompt ids and reply;
grounded_torch_score.mjs scores it and checks the ids against bpe.mjs."""
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL, NAME, VARIANT = sys.argv[1], sys.argv[2], sys.argv[3]
src = (HERE / 'grounded_score.mjs').read_text()
SYSTEM = ''.join(re.findall(r"'([^']*)'", src.split('export const SYSTEM =')[1].split(';')[0]))
items = json.loads((HERE / 'data/grounded_items.json').read_text())
tok = AutoTokenizer.from_pretrained(MODEL)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
if VARIANT != 'fp32':
    deq = torch.load(HERE / f'{NAME}-u{VARIANT}.deq.pt')
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in deq.items():
            sd[k].copy_(v)
model.to(dev)
stop = [tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')]
out = []
for it in items:
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f"Passage:\n{it['context']}\n\nQuestion: {it['question']}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    with torch.no_grad():
        gen = model.generate(torch.tensor([ids], device=dev), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev),
                             max_new_tokens=48, do_sample=False, eos_token_id=stop, pad_token_id=stop[0])[0, len(ids):].tolist()
    gen = [t for t in gen if t not in stop]
    out.append({'id': it['id'], 'ids': ids, 'reply': tok.decode(gen).strip()})
(HERE / f'grounded_torch-{NAME}-{VARIANT}.json').write_text(json.dumps(out))
print('wrote', len(out), 'replies on', dev)
