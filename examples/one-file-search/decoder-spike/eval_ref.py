"""Torch references for the decoder tests, fp32 and the dequantized u8/u4/u5
models: (1) teacher-forced logits over ~1000 tokens of concatenated stories
(crosses the local layers' 256-token window), (2) perplexity on the first N
validation stories. Writes eval_ref.json plus checkpoint logits .npy.
`python3 eval_ref.py 300 [variant ...]` (fp32 u8 u4 u5)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant_u5 import quant_u5

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
EOS, LONG = 50256, 1000
CHECKPOINTS = [50, 254, 255, 256, 257, 400, 511, 700, 999]
tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
stories = [s.strip() for s in (HERE / 'data/TinyStories-valid.txt').read_text(encoding='utf-8').split('<|endoftext|>')]
stories = [s for s in stories if s][:N + 50]
story_ids = [[EOS] + tok(s).input_ids for s in stories[:N]]
long_ids = [EOS]
for s in stories[N:]:
    long_ids += tok(s).input_ids + [EOS]
    if len(long_ids) >= LONG:
        break
long_ids = long_ids[:LONG]


def quant_blocks(w, bits):
    B, LEVELS = (64, 255) if bits == 8 else (32, 15)
    rows, cols = w.shape
    c = w.reshape(rows, cols // B, B)
    lo, hi = c.min(2), c.max(2)
    scale = np.where(hi - lo <= 0, 1e-12, (hi - lo) / LEVELS).astype(np.float32)
    q = np.clip(np.round((c - lo[..., None]) / scale[..., None]), 0, LEVELS)
    return (q * scale[..., None] + lo[..., None]).reshape(rows, cols).astype(np.float32)


def load(bits):
    m = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M').eval()
    if bits:
        with torch.no_grad():
            for k, v in m.state_dict().items():
                # Every matrix the blob quantizes. lm_head shares wte's storage:
                # quantizing it too would quantize wte twice, which u5's
                # least-squares fit and f16 rounding do not leave unchanged.
                if v.dim() == 2 and v.dtype == torch.float32 and k != 'lm_head.weight':
                    v.copy_(torch.from_numpy(quant_u5(v.numpy())[3] if bits == 5 else quant_blocks(v.numpy(), bits)))
    return m


# Variants named on the command line after N (default all) are (re)computed
# and merged into an existing eval_ref.json.
ALL = {'fp32': 0, 'u8': 8, 'u4': 4, 'u5': 5}
names = sys.argv[2:] or list(ALL)
prev = HERE / 'eval_ref.json'
out = {'N': N, 'story_ids': story_ids, 'long_ids': long_ids, 'checkpoints': CHECKPOINTS, 'variants': {}}
if prev.exists() and sys.argv[2:]:
    old = json.loads(prev.read_text())
    assert old['story_ids'] == story_ids and old['long_ids'] == long_ids
    out['variants'] = old['variants']
for name, bits in ((n, ALL[n]) for n in names):
    m = load(bits)
    with torch.no_grad():
        nll = []
        for ids in story_ids:
            lp = torch.log_softmax(m(torch.tensor([ids])).logits[0, :-1], -1)
            nll.append(float(-lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].sum()))
        logits = m(torch.tensor([long_ids])).logits[0]
    lp = torch.log_softmax(logits[:-1], -1)
    ntok = sum(len(i) - 1 for i in story_ids)
    out['variants'][name] = {
        'story_nll': nll, 'ppl': float(np.exp(sum(nll) / ntok)),
        'long_argmax': logits.argmax(-1).tolist(),
        'long_next_logprob': lp[torch.arange(LONG - 1), torch.tensor(long_ids[1:])].tolist(),
    }
    np.save(HERE / f'long_{name}.npy', logits[CHECKPOINTS].numpy())
    print(name, 'ppl', round(out['variants'][name]['ppl'], 4), 'over', ntok, 'tokens')
(HERE / 'eval_ref.json').write_text(json.dumps(out))
