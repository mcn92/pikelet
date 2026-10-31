"""Low-rank vocabulary shortlist for the LM head: wte ~= A B (rank r), fitted
to the hidden states the model produces rather than to wte itself: with
C = E[x x^T] over final-LayerNorm outputs (stories 2000-2399 of the
validation file, disjoint from the evaluation stories), C = L L^T,
SVD(wte L) = U S V^T, A = U_r S_r, B = V_r^T L^-1, which minimizes
E||wte x - A B x||^2. A (V x r) is stored as u8 blocks of 64 in the blob's
quant layout (codes, f32 scales, f32 offsets), then B (r x D) f32. The
kernel scores every token with A (B x), keeps the top K, and rescores only
those rows exactly against the real wte.
`python3 export_shortlist.py 128` writes shortlist-r128.bin. For the Instruct
model: DECODER_MODEL=roneneldan/TinyStories-Instruct-33M
DECODER_NAME=instruct writes instruct-shortlist-r128.bin, fitted on entries
3000-3399 of the Instruct validation file."""
import os
import sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 128
assert R % 64 == 0
MODEL = os.environ.get('DECODER_MODEL', 'roneneldan/TinyStories-33M')
NAME = os.environ.get('DECODER_NAME')
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
wte = model.state_dict()['transformer.wte.weight'].double().numpy()

data = 'data/TinyStories-Instruct-valid.txt' if NAME else 'data/TinyStories-valid.txt'
stories = [s.strip() for s in (HERE / data).read_text(encoding='utf-8').split('<|endoftext|>')]
stories = [s for s in stories if s][3000:3400] if NAME else [s for s in stories if s][2000:2400]
C = np.zeros((wte.shape[1], wte.shape[1]))
n = 0
with torch.no_grad():
    for s in stories:
        h = model.transformer(torch.tensor([[50256] + tok(s).input_ids[:255]])).last_hidden_state[0].double().numpy()
        C += h.T @ h
        n += len(h)
C /= n
Lc = np.linalg.cholesky(C + 1e-6 * np.trace(C) / len(C) * np.eye(len(C)))
U, S, Vt = np.linalg.svd(wte @ Lc, full_matrices=False)
A = (U[:, :R] * S[:R]).astype(np.float32)
Bm = (Vt[:R] @ np.linalg.inv(Lc)).astype(np.float32)
print(f'rank {R}: {n} hidden states; logit energy kept {float((S[:R] ** 2).sum() / (S ** 2).sum()):.4f}')

rows, cols = A.shape
c = A.reshape(rows, cols // 64, 64)
lo, hi = c.min(2), c.max(2)
scale = np.where(hi - lo <= 0, 1e-12, (hi - lo) / 255).astype(np.float32)
q = np.clip(np.round((c - lo[..., None]) / scale[..., None]), 0, 255).astype(np.uint8).reshape(rows, cols)
blob = q.tobytes() + scale.tobytes() + lo.astype(np.float32).tobytes() + Bm.tobytes()
OUT = f'{NAME}-shortlist-r{R}.bin' if NAME else f'shortlist-r{R}.bin'
(HERE / OUT).write_bytes(blob)
print(f'{OUT} {len(blob) / 1e6:.1f} MB')
