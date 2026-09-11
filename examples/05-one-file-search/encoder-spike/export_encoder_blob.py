#!/usr/bin/env python3
"""Export the full MiniLM-L6 as the inline-encoder weight blob (v0 draft of
the kind-3 segment layout) plus per-stage torch references for kernel parity.

Blob layout — plain concatenation, every tensor byte-size is a multiple of
16, offsets are running sums that encoder.cpp recomputes with the same
arithmetic (V=30522 P=512 T=2 D=384 F=1536 L=6 B=64; u8 matrices are
block-64 affine along the last dim, scales/offsets f32 per block; biases
and LayerNorm params are f32):

  word_q, word_s, word_o,
  pos_q, pos_s, pos_o,
  type_q, type_s, type_o,
  embln_g, embln_b,
  then per layer 0..5:
    wq_q,wq_s,wq_o,bq,  wk_q,wk_s,wk_o,bk,  wv_q,wv_s,wv_o,bv,
    wo_q,wo_s,wo_o,bo,  ln1_g,ln1_b,
    wu_q,wu_s,wu_o,bu,  wd_q,wd_s,wd_o,bd,  ln2_g,ln2_b

References (real/): token ids for a test query, hidden states after the
embedding LayerNorm and after each layer, and the pooled+normalized output
— all from the SAME fake-quantized computation the blob encodes, so the
WASM forward should match to f32 accumulation noise.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                     help="HF model id. Must match the kernel's compiled-in shape "
                          "(encoder.cpp constexpr V,P,T,D,F,L,B,H) exactly.")
parser.add_argument("--pooling", choices=["mean", "cls"], default="mean",
                     help="mean: average every token's final hidden state (MiniLM's "
                          "native pooling). cls: use the [CLS] token's hidden state "
                          "(e.g. Snowflake arctic-embed models). Must match "
                          "inline-transformer.mjs's embed() pooling exactly, or the "
                          "reference vectors won't match what the reader produces.")
args = parser.parse_args()

HERE = Path(__file__).resolve().parent
OUT = HERE / "real"
OUT.mkdir(exist_ok=True)
MODEL = args.model
B = 64
TEST_TEXT = "how do volcanoes form and why do they erupt"


def quant_blocks(w: np.ndarray):
    """Block-64 affine u8 along the last dim. Returns (q, scales, offsets)
    and leaves w unchanged."""
    rows, cols = w.shape
    nb = cols // B
    q = np.zeros((rows, cols), dtype=np.uint8)
    scales = np.zeros((rows, nb), dtype=np.float32)
    offsets = np.zeros((rows, nb), dtype=np.float32)
    for bi in range(nb):
        chunk = w[:, bi * B:(bi + 1) * B]
        lo = chunk.min(axis=1)
        hi = chunk.max(axis=1)
        scale = np.where(hi - lo <= 0, 1e-12, (hi - lo) / 255.0).astype(np.float32)
        q[:, bi * B:(bi + 1) * B] = np.clip(np.round((chunk - lo[:, None]) / scale[:, None]), 0, 255).astype(np.uint8)
        scales[:, bi] = scale
        offsets[:, bi] = lo
    return q, scales, offsets


def dequant(q, scales, offsets):
    rows, cols = q.shape
    out = np.zeros((rows, cols), dtype=np.float32)
    for bi in range(cols // B):
        out[:, bi * B:(bi + 1) * B] = q[:, bi * B:(bi + 1) * B].astype(np.float32) * scales[:, bi:bi + 1] + offsets[:, bi:bi + 1]
    return out


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
model.eval()

# The kernel's shape is compiled in (encoder.cpp constexpr), not read from
# the blob or the declaration, so a mismatched model produces a blob that
# silently corrupts every embedding rather than failing loudly. Check the
# shape the config actually implies before spending time exporting it.
cfg = model.config
expected = {"vocab_size": 30522, "max_position_embeddings": 512, "type_vocab_size": 2,
            "hidden_size": 384, "intermediate_size": 1536, "num_hidden_layers": 6,
            "num_attention_heads": 12}
mismatches = {k: (getattr(cfg, k, None), v) for k, v in expected.items() if getattr(cfg, k, None) != v}
if mismatches:
    lines = "\n".join(f"  {k}: model has {got}, kernel requires {want}" for k, (got, want) in mismatches.items())
    raise SystemExit(f"{MODEL} does not match the compiled-in kernel shape:\n{lines}\n"
                      f"Rebuild encoder.cpp for this shape first, or pick a model with this exact shape.")
if cfg.hidden_act != "gelu":
    raise SystemExit(f"{MODEL} uses hidden_act={cfg.hidden_act!r}; the kernel's activation is hardcoded to "
                      f"(erf) gelu. A different activation needs a kernel change, not just new weights.")

blob = bytearray()
manifest = []


def emit(name, arr):
    data = np.ascontiguousarray(arr).tobytes()
    assert len(data) % 16 == 0, f"{name}: {len(data)} not 16-aligned"
    manifest.append({"name": name, "offset": len(blob), "bytes": len(data)})
    blob.extend(data)


def emit_quant(name, w):
    q, s, o = quant_blocks(w)
    emit(f"{name}_q", q)
    emit(f"{name}_s", s)
    emit(f"{name}_o", o)
    return dequant(q, s, o)


# Fake-quantized copies feed the reference computation so refs match the blob.
deq = {}
emb = model.embeddings
deq["word"] = emit_quant("word", emb.word_embeddings.weight.detach().numpy())
deq["pos"] = emit_quant("pos", emb.position_embeddings.weight.detach().numpy())
deq["type"] = emit_quant("type", emb.token_type_embeddings.weight.detach().numpy())
emit("embln_g", emb.LayerNorm.weight.detach().numpy().astype(np.float32))
emit("embln_b", emb.LayerNorm.bias.detach().numpy().astype(np.float32))

for i, layer in enumerate(model.encoder.layer):
    pieces = {
        "wq": layer.attention.self.query, "wk": layer.attention.self.key,
        "wv": layer.attention.self.value, "wo": layer.attention.output.dense,
        "wu": layer.intermediate.dense, "wd": layer.output.dense,
    }
    for key in ["wq", "wk", "wv"]:
        deq[f"l{i}.{key}"] = emit_quant(f"l{i}_{key}", pieces[key].weight.detach().numpy())
        emit(f"l{i}_{key}_b", pieces[key].bias.detach().numpy().astype(np.float32))
    deq[f"l{i}.wo"] = emit_quant(f"l{i}_wo", pieces["wo"].weight.detach().numpy())
    emit(f"l{i}_wo_b", pieces["wo"].bias.detach().numpy().astype(np.float32))
    emit(f"l{i}_ln1_g", layer.attention.output.LayerNorm.weight.detach().numpy().astype(np.float32))
    emit(f"l{i}_ln1_b", layer.attention.output.LayerNorm.bias.detach().numpy().astype(np.float32))
    deq[f"l{i}.wu"] = emit_quant(f"l{i}_wu", pieces["wu"].weight.detach().numpy())
    emit(f"l{i}_wu_b", pieces["wu"].bias.detach().numpy().astype(np.float32))
    deq[f"l{i}.wd"] = emit_quant(f"l{i}_wd", pieces["wd"].weight.detach().numpy())
    emit(f"l{i}_wd_b", pieces["wd"].bias.detach().numpy().astype(np.float32))
    emit(f"l{i}_ln2_g", layer.output.LayerNorm.weight.detach().numpy().astype(np.float32))
    emit(f"l{i}_ln2_b", layer.output.LayerNorm.bias.detach().numpy().astype(np.float32))

(OUT / "encoder-weights.bin").write_bytes(bytes(blob))
(OUT / "encoder-blob-manifest.json").write_text(json.dumps(manifest))
print(f"[blob] {len(blob) / 1048576:.2f} MiB, {len(manifest)} tensors")

# ---- Per-stage references via numpy, using the SAME dequantized weights ----
def layernorm(x, g, b):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-12) * g + b

def erf_gelu(x):
    from scipy.special import erf  # noqa: F401  (fallback below if absent)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

try:
    import scipy  # noqa: F401
    gelu = erf_gelu
except ImportError:
    def gelu(x):
        t = torch.erf(torch.from_numpy(x / np.sqrt(2.0)))
        return (0.5 * x * (1.0 + t.numpy())).astype(np.float32)

ids = tokenizer(TEST_TEXT)["input_ids"]
seq = len(ids)
gnp = lambda name: model.embeddings.LayerNorm  # noqa: E731 (unused helper guard)

ln_g = {"emb": emb.LayerNorm.weight.detach().numpy(), }
x = deq["word"][ids] + deq["pos"][:seq] + deq["type"][0]
x = layernorm(x.astype(np.float32), emb.LayerNorm.weight.detach().numpy(), emb.LayerNorm.bias.detach().numpy())
stages = {"emb": x.copy()}

H, HD = 12, 32
for i, layer in enumerate(model.encoder.layer):
    bq = layer.attention.self.query.bias.detach().numpy()
    bk = layer.attention.self.key.bias.detach().numpy()
    bv = layer.attention.self.value.bias.detach().numpy()
    bo = layer.attention.output.dense.bias.detach().numpy()
    bu = layer.intermediate.dense.bias.detach().numpy()
    bd = layer.output.dense.bias.detach().numpy()
    q = x @ deq[f"l{i}.wq"].T + bq
    k = x @ deq[f"l{i}.wk"].T + bk
    v = x @ deq[f"l{i}.wv"].T + bv
    ctx = np.zeros_like(x)
    for h in range(H):
        qs = q[:, h * HD:(h + 1) * HD]
        ks = k[:, h * HD:(h + 1) * HD]
        vs = v[:, h * HD:(h + 1) * HD]
        scores = qs @ ks.T / np.sqrt(HD)
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        a = e / e.sum(axis=1, keepdims=True)
        ctx[:, h * HD:(h + 1) * HD] = a @ vs
    x = layernorm(x + ctx @ deq[f"l{i}.wo"].T + bo,
                  layer.attention.output.LayerNorm.weight.detach().numpy(),
                  layer.attention.output.LayerNorm.bias.detach().numpy())
    hgelu = gelu((x @ deq[f"l{i}.wu"].T + bu).astype(np.float32))
    x = layernorm(x + hgelu @ deq[f"l{i}.wd"].T + bd,
                  layer.output.LayerNorm.weight.detach().numpy(),
                  layer.output.LayerNorm.bias.detach().numpy())
    stages[f"layer{i}"] = x.copy()

pooled = x[0] if args.pooling == "cls" else x.mean(axis=0)
pooled = pooled / np.linalg.norm(pooled)

np.array(ids, dtype=np.int32).tofile(OUT / "ref-ids.i32")
np.concatenate([stages["emb"][None], *[stages[f"layer{i}"][None] for i in range(6)]]).astype(np.float32).tofile(OUT / "ref-stages.f32")
pooled.astype(np.float32).tofile(OUT / "ref-pooled.f32")
(OUT / "ref-meta.json").write_text(json.dumps({"text": TEST_TEXT, "seq": seq, "ids": ids}))
print(f"[refs] seq={seq} ids={ids}")
print(f"[refs] wrote ref-stages.f32 (7 x {seq} x 384) + ref-pooled.f32")
