"""A Llama-architecture checkpoint (SmolLM2 by default) -> llama.cpp's blob:
`python3 export_llama.py [8|5] [hf id] [name]` writes <name>-u<bits>.bin
(u8 in blocks of 64, or u5 in blocks of 32, see quant_u5.py; u5 rows' high-bit
plane padded to a multiple of 128 weights) and the
tokenizer as llama-tokenizer/<name>.json (vocab, merges, special tokens).

Blob: a 64-byte header of int32 fields (magic 'LLM1', V, D, F, L, H, KVH,
HD, format, tied) and f32 fields (rope_theta, rms_eps), padded to 64 bytes;
then embed (V x D); per layer: attn_norm (D), wq (H*HD x D), wk (KVH*HD x
D), wv (KVH*HD x D), wo (D x H*HD), ffn_norm (D), w_gate (F x D), w_up
(F x D), w_down (D x F); final norm (D); lm_head (V x D) only if untied.
Every array is a multiple of 16 bytes, as in export_decoder.py."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant_u5 import quant_u5, pack_u5

HERE = Path(__file__).resolve().parent
BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'HuggingFaceTB/SmolLM2-135M-Instruct'
NAME = sys.argv[3] if len(sys.argv) > 3 else 'smollm2-135m'
assert BITS in (8, 5)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
cfg = model.config
assert cfg.hidden_act == 'silu' and not getattr(cfg, 'attention_bias', False) and cfg.rope_scaling is None
sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
V, D, F, L = cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
H, KVH = cfg.num_attention_heads, cfg.num_key_value_heads
HD = D // H
tied = bool(cfg.tie_word_embeddings)


def quant_u8(w):
    rows, cols = w.shape
    c = w.reshape(rows, cols // 64, 64)
    lo, hi = c.min(2), c.max(2)
    scale = np.where(hi - lo <= 0, 1e-12, (hi - lo) / 255).astype(np.float32)
    q = np.clip(np.round((c - lo[..., None]) / scale[..., None]), 0, 255).astype(np.uint8)
    deq = (q.astype(np.float32) * scale[..., None] + lo[..., None]).reshape(rows, cols)
    return q.reshape(rows, cols), scale, lo.astype(np.float32), deq


blob = bytearray()
deq_sd = {}


def emit(arr):
    assert len(blob) % 16 == 0
    blob.extend(np.ascontiguousarray(arr).tobytes())


def emit_quant(name):
    if BITS == 5:
        # pack_u5's high-bit plane runs in groups of 128 weights; SmolLM2's
        # D (576, 960) is not a multiple of 128, so each row's codes are
        # zero-padded to one for that plane (ceil(cols / 128) * 16 bytes per
        # row) and the nibble plane is cut back to cols / 2. llama.cpp only
        # reads u5 when expanding it (format 6), never in the GEMM.
        q, s16, o16, deq = quant_u5(sd[name])
        rows, cols = q.shape
        pad = -cols % 128
        nib, hib = pack_u5(np.pad(q, ((0, 0), (0, pad))))
        emit(nib[:, :cols // 2]); emit(hib); emit(s16); emit(o16)
    else:
        q, s, o, deq = quant_u8(sd[name])
        emit(q); emit(s); emit(o)
    deq_sd[name] = deq


header = np.zeros(16, dtype=np.int32)
header[:10] = [0x314D4C4C, V, D, F, L, H, KVH, HD, BITS, int(tied)]
header[10:12] = np.array([cfg.rope_theta, cfg.rms_norm_eps], dtype=np.float32).view(np.int32)
emit(header)
emit_quant('model.embed_tokens.weight')
for i in range(L):
    p = f'model.layers.{i}.'
    emit(sd[p + 'input_layernorm.weight'])
    for w in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        emit_quant(p + f'self_attn.{w}.weight')
    emit(sd[p + 'post_attention_layernorm.weight'])
    for w in ('gate_proj', 'up_proj', 'down_proj'):
        emit_quant(p + f'mlp.{w}.weight')
emit(sd['model.norm.weight'])
if not tied:
    emit_quant('lm_head.weight')
(HERE / f'{NAME}-u{BITS}.bin').write_bytes(bytes(blob))
print(f'{NAME}-u{BITS}.bin {len(blob) / 1e6:.1f} MB  V={V} D={D} F={F} L={L} H={H} KVH={KVH} HD={HD} tied={tied}')

# The dequantized weights, for torch references of exactly what the blob holds.
torch.save({k: torch.from_numpy(v) for k, v in deq_sd.items()}, HERE / f'{NAME}-u{BITS}.deq.pt')

tok = AutoTokenizer.from_pretrained(MODEL)
tj = json.loads(tok.backend_tokenizer.to_str())
merges = [m if isinstance(m, str) else ' '.join(m) for m in tj['model']['merges']]
specials = {t['content']: t['id'] for t in tj['added_tokens']}
digits = any(p.get('type') == 'Digits' for p in (tj['pre_tokenizer'].get('pretokenizers') or [tj['pre_tokenizer']]))
out = HERE / 'llama-tokenizer'
out.mkdir(exist_ok=True)
(out / f'{NAME}.json').write_text(json.dumps({'vocab': tj['model']['vocab'], 'merges': merges, 'specials': specials,
                                               'individualDigits': digits, 'eos': tok.eos_token_id}))
