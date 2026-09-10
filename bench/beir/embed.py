#!/usr/bin/env python3.11
"""Encode BEIR corpus records or queries with plain float32 MiniLM
(sentence-transformers/all-MiniLM-L6-v2) — mean pooling, L2-normalized,
exactly the recipe examples/04-static-wiki-pack/embed_corpus.py uses, and
the recipe Pikelet's inline quantized encoder is checked against.

This is Configuration A's encoder: no chunking (one BEIR record -> one
embedding, unlike embed_corpus.py which re-chunks), no quantization. Every
other configuration in the ablation ladder starts by comparing against this.

Usage:
    python3.11 bench/beir/embed.py scifact corpus
    python3.11 bench/beir/embed.py scifact queries

Reads bench/beir/work/<dataset>/records.jsonl (corpus) or
bench/beir/cache/<dataset>/queries.jsonl (queries), in the exact row order
convert.mjs / the BEIR download already fixed. Writes
bench/beir/work/<dataset>/vectors-<corpus|queries>-float32.f32 (row-major
float32, dim 384) plus a .ids.json sidecar recording the id for each row so
downstream scripts never have to re-derive order.
"""
import sys
import os
import json
import time

HERE = os.path.dirname(__file__)
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
MAX_TOKENS = 256  # matches embed_corpus.py; the bundled inline encoder's
                   # compiled window is smaller (128) and windowed-mean-pools
                   # longer inputs — config A intentionally uses the
                   # upstream model's own native window, not Pikelet's.


def load_rows(dataset: str, which: str):
    if which == "corpus":
        path = os.path.join(HERE, "work", dataset, "records.jsonl")
        id_key = "_id"
    elif which == "queries":
        path = os.path.join(HERE, "cache", dataset, "queries.jsonl")
        id_key = "_id"
    else:
        raise ValueError(which)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run convert.mjs first" if which == "corpus" else path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row["text"] if which == "corpus" else row["text"]
            rows.append((row[id_key], text))
    return rows


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("corpus", "queries"):
        print(__doc__)
        sys.exit(1)
    dataset, which = sys.argv[1], sys.argv[2]

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    rows = load_rows(dataset, which)
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    work_dir = os.path.join(HERE, "work", dataset)
    os.makedirs(work_dir, exist_ok=True)
    vec_path = os.path.join(work_dir, f"vectors-{which}-float32.f32")
    ids_path = os.path.join(work_dir, f"vectors-{which}-float32.ids.json")

    batch_size = 64
    t0 = time.time()
    with open(vec_path, "wb") as vf:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            with torch.no_grad():
                enc = tok(batch, padding=True, truncation=True, max_length=MAX_TOKENS,
                          return_tensors="pt").to(device)
                out = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out * mask).sum(1) / mask.sum(1)
                emb = torch.nn.functional.normalize(emb, dim=1)
            vf.write(emb.cpu().numpy().astype(np.float32).tobytes())
            if (start // batch_size) % 20 == 0:
                print(f"{which} {start}/{len(texts)}", file=sys.stderr)

    with open(ids_path, "w") as f:
        json.dump({"ids": ids, "dim": DIM, "model": MODEL_ID, "count": len(ids)}, f)

    dt = time.time() - t0
    print(f"{dataset} {which}: {len(ids)} rows x {DIM}d -> {vec_path} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
