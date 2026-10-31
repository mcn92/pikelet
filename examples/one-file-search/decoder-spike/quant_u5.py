"""u5 block format: 32 weights per block, w = s*q + o with q in 0..31,
fitted by least squares (s, o refit given the codes, codes re-rounded given
s, o), and s, o stored as f16 with the codes re-rounded against the stored
values. Shared by export_decoder.py and eval_ref.py so the blob and the torch
reference hold the same weights.

Per block, 24 bytes (6.0 bits per weight, the size of u4 with f32 s/o):
  16 bytes  nibble plane: low 4 bits of weight j in the low nibble of byte j,
            of weight j+16 in the high nibble (the u4 layout)
   4 bytes  high-bit plane: the plane runs in 16-byte groups of 128
            weights (4 blocks), bit g of byte j holding bit 4 of the group's
            weight g*16 + j, so a 16-lane half-block unpacks with one shift
   2 + 2    s and o as f16
Each plane is stored as its own array per matrix, like the f32 s/o arrays of
the u8/u4 formats."""
import numpy as np

B, LEVELS, LS_ITERS = 32, 31, 6


def quant_u5(w):
    """w: (rows, cols) f32. Returns (codes uint8 (rows, cols), s f16 (rows, nb),
    o f16 (rows, nb), dequantized f32 (rows, cols))."""
    rows, cols = w.shape
    t = w.reshape(rows, cols // B, B).astype(np.float64)
    lo, hi = t.min(2, keepdims=True), t.max(2, keepdims=True)
    s = np.where(hi - lo <= 0, 1e-12, (hi - lo) / LEVELS)
    o = lo
    q = np.clip(np.round((t - o) / s), 0, LEVELS)
    for _ in range(LS_ITERS):
        qm, tm = q.mean(2, keepdims=True), t.mean(2, keepdims=True)
        var = ((q - qm) ** 2).mean(2, keepdims=True)
        cov = ((q - qm) * (t - tm)).mean(2, keepdims=True)
        s_new = np.where(var > 0, cov / np.maximum(var, 1e-30), s)
        s = np.where(s_new > 0, s_new, s)
        o = tm - s * qm
        q = np.clip(np.round((t - o) / s), 0, LEVELS)
    s16 = np.maximum(s, 6e-8).astype(np.float16)
    o16 = o.astype(np.float16)
    q = np.clip(np.round((t - o16.astype(np.float64)) / s16.astype(np.float64)), 0, LEVELS)
    deq = (q * s16.astype(np.float32) + o16.astype(np.float32)).astype(np.float32)
    return (q.astype(np.uint8).reshape(rows, cols), s16[..., 0], o16[..., 0], deq.reshape(rows, cols))


def pack_u5(codes):
    """codes: (rows, cols) 0..31, cols % 128 == 0 -> (nibble plane
    (rows, cols/2), high-bit plane (rows, cols/8))."""
    rows, cols = codes.shape
    b = (codes & 0x0F).reshape(rows, -1, 2, 16)
    nib = (b[:, :, 0, :] | (b[:, :, 1, :] << 4)).astype(np.uint8).reshape(rows, -1)
    assert cols % 128 == 0
    hb = ((codes >> 4) & 1).astype(np.uint8).reshape(rows, cols // 128, 8, 16)
    hib = (hb << np.arange(8, dtype=np.uint8)[None, None, :, None]).sum(2).astype(np.uint8).reshape(rows, -1)
    return nib, hib
