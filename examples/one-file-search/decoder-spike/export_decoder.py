"""TinyStories-33M (GPT-Neo) -> the inline encoder's block-affine weight
format: `python3 export_decoder.py` writes tinystories-u8.bin (u8, blocks
of 64) plus the GPT-2 BPE tokenizer files and the torch references (fp32 and
the dequantized-u8 model) behind `node tell.mjs check`;
`python3 export_decoder.py 4` writes tinystories-u4.bin (u4, blocks of 32);
`python3 export_decoder.py 5` writes tinystories-u5.bin (see quant_u5.py).
Another GPT-Neo checkpoint of the same shape: DECODER_MODEL=<hf id>
DECODER_NAME=<prefix> writes <prefix>-u<bits>.bin (blob only)."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant_u5 import quant_u5, pack_u5

HERE = Path(__file__).resolve().parent
BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # 8: u8 blocks of 64; 4: u4 blocks of 32; 5: u5
MODEL = os.environ.get('DECODER_MODEL', 'roneneldan/TinyStories-33M')
NAME = os.environ.get('DECODER_NAME', 'tinystories')
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
sd = {k: v.float().numpy() for k, v in model.state_dict().items()}


def quant_blocks(w, bits=8):
    B, LEVELS = (64, 255) if bits == 8 else (32, 15)
    rows, cols = w.shape
    nb = cols // B
    c = w.reshape(rows, nb, B)
    lo, hi = c.min(2), c.max(2)
    scale = np.where(hi - lo <= 0, 1e-12, (hi - lo) / LEVELS).astype(np.float32)
    q = np.clip(np.round((c - lo[..., None]) / scale[..., None]), 0, LEVELS).astype(np.uint8)
    deq = (q.astype(np.float32) * scale[..., None] + lo[..., None]).reshape(rows, cols)
    return q.reshape(rows, cols), scale, lo.astype(np.float32), deq


blob = bytearray()
deq_sd = {}


def emit(arr):
    assert len(blob) % 16 == 0
    blob.extend(np.ascontiguousarray(arr).tobytes())


def pack_u4(q):
    # Per block of 32: weight j in the low nibble of byte j, j+16 in the high.
    b = q.reshape(q.shape[0], -1, 2, 16)
    return (b[:, :, 0, :] | (b[:, :, 1, :] << 4)).astype(np.uint8).reshape(q.shape[0], -1)


def emit_quant(name):
    if BITS == 5:
        q, s16, o16, deq = quant_u5(sd[name])
        nib, hib = pack_u5(q)
        emit(nib); emit(hib); emit(s16); emit(o16)
        deq_sd[name] = deq
        return
    q, s, o, deq = quant_blocks(sd[name], BITS)
    emit(q if BITS == 8 else pack_u4(q)); emit(s); emit(o)
    deq_sd[name] = deq


def emit_f32(name):
    emit(sd[name].astype(np.float32))


emit_quant('transformer.wte.weight')
emit_quant('transformer.wpe.weight')
for i in range(model.config.num_layers):
    p = f'transformer.h.{i}.'
    emit_f32(p + 'ln_1.weight'); emit_f32(p + 'ln_1.bias')
    for w in ('q_proj', 'k_proj', 'v_proj'):
        emit_quant(p + f'attn.attention.{w}.weight')
    emit_quant(p + 'attn.attention.out_proj.weight'); emit_f32(p + 'attn.attention.out_proj.bias')
    emit_f32(p + 'ln_2.weight'); emit_f32(p + 'ln_2.bias')
    emit_quant(p + 'mlp.c_fc.weight'); emit_f32(p + 'mlp.c_fc.bias')
    emit_quant(p + 'mlp.c_proj.weight'); emit_f32(p + 'mlp.c_proj.bias')
emit_f32('transformer.ln_f.weight'); emit_f32('transformer.ln_f.bias')
(HERE / f'{NAME}-u{BITS}.bin').write_bytes(bytes(blob))
print('blob', len(blob) / 1e6, 'MB')
if BITS != 8 or NAME != 'tinystories':
    sys.exit(0)  # the references below are for the u8 check

tok.save_pretrained(HERE / 'tokenizer')  # vocab.json + merges.txt

PROMPTS = ['Once upon a time, there was a little dog named',
           'Lily wanted to bake a cake for her mom, but',
           'The robot looked at the moon and said']
refs = {'prompts': []}


def greedy(m, ids, n=60):
    with torch.no_grad():
        out = m.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                         max_new_tokens=n, do_sample=False, pad_token_id=50256)
    return out[0, len(ids):].tolist()


def first_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0, -1].numpy()


fp32 = {p: (greedy(model, tok(p).input_ids), first_logits(model, tok(p).input_ids)) for p in PROMPTS}
with torch.no_grad():
    for k, v in deq_sd.items():
        model.state_dict()[k].copy_(torch.from_numpy(v))
for p in PROMPTS:
    ids = tok(p).input_ids
    g8, l8 = greedy(model, ids), first_logits(model, ids)
    g32, l32 = fp32[p]
    refs['prompts'].append({'text': p, 'ids': ids, 'greedy_fp32': g32, 'greedy_u8': g8,
                            'logits_u8_top': np.argsort(-l8)[:10].tolist(),
                            'logits_u8': l8[np.argsort(-l8)[:10]].tolist(),
                            'logits_u8_full_maxabs': float(np.abs(l8).max())})
    np.save(HERE / f'logits_u8_{len(refs["prompts"]) - 1}.npy', l8)
    print(repr(p + tok.decode(g8)))
    print('  fp32 == u8 greedy tokens:', sum(a == b for a, b in zip(g32, g8)), '/', len(g32))
(HERE / 'refs.json').write_text(json.dumps(refs, indent=1))
