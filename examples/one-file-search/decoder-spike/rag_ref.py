"""Model half of the pack-native QA test, in torch (the kernel gives the
same replies; kernel_run.mjs confirms a chosen condition):
`python3 rag_ref.py <hf id> <name> <fp32|8|5> <condition>` answers
data/pack_qa_items.json greedily (48 tokens) under one condition:
  closed  no passages; the model answers from what it knows
  rag3    the wiki pack's top 3 passages (data/pack_rag_retrieval.json)
  rag1    the top passage only
Writes rag_out-<name>-<variant>-<condition>.json: each item's prompt ids
(for kernel_run.mjs) and reply. rag_score.mjs scores them."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL, NAME, VARIANT, COND = sys.argv[1:5]
CLOSED_SYS = "Answer the question with a short answer. If you do not know the answer, reply exactly: I don't know."
RAG_SYS = ('Answer the question using only the passages below. Reply with a short answer taken from the passages. '
           'If the passages do not contain the answer, reply exactly: Not in the passages.')
items = json.loads((HERE / 'data/pack_qa_items.json').read_text())
ret = {r['id']: r for r in json.loads((HERE / 'data/pack_rag_retrieval.json').read_text())}
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


def messages(it):
    if COND == 'closed':
        return [{'role': 'system', 'content': CLOSED_SYS}, {'role': 'user', 'content': it['question']}]
    k = {'rag3': 3, 'rag1': 1}[COND]
    ps = ret[it['id']]['passages'][:k]
    body = '\n\n'.join(f'[{i + 1}] {p["text"]}' for i, p in enumerate(ps))
    return [{'role': 'system', 'content': RAG_SYS}, {'role': 'user', 'content': f"Passages:\n{body}\n\nQuestion: {it['question']}"}]


out = []
for it in items:
    ids = tok.apply_chat_template(messages(it), add_generation_prompt=True)
    with torch.no_grad():
        gen = model.generate(torch.tensor([ids], device=dev), attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev),
                             max_new_tokens=48, do_sample=False, eos_token_id=stop, pad_token_id=stop[0])[0, len(ids):].tolist()
    gen = [t for t in gen if t not in stop]
    out.append({'id': it['id'], 'ids': ids, 'reply': tok.decode(gen).strip()})
(HERE / f'rag_out-{NAME}-{VARIANT}-{COND}.json').write_text(json.dumps(out))
print(f'{COND}: {len(out)} replies, mean prompt {sum(len(o["ids"]) for o in out) / len(out):.0f} tokens, on {dev}')
