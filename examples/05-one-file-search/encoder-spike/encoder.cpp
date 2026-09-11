// Six-layer MiniLM-L6 forward over the block-affine u8 weight blob emitted
// by export_encoder_blob.py — the inline-encoder (kind 3) kernel spike.
// Float weight tensors are never materialized: every GEMV dequantizes
// inside the dot product. LayerNorm, softmax, GELU, biases, and residuals
// run in f32. Correctness-first: attention is direct (seq <= MAXSEQ, the
// score tile is KBs), the FFN intermediate is materialized per token;
// the streamed-accumulator variant is a later optimization. Attention is
// O(seq^2) per head per layer, so MAXSEQ trades embedding latency for
// window width — pinned to P (the position-embedding table size, 512),
// since a wider window has no positions to encode past that anyway.
//
// Layout constants mirror the exporter exactly; offsets are running sums
// in the same emit order. Everything is 16-byte aligned by construction.

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

namespace {

constexpr int V = 30522, P = 512, T = 2, D = 384, F = 1536, L = 6, B = 64;
constexpr int H = 12, HD = 32;
constexpr int NB = D / B;    // blocks per D-wide row
constexpr int NBF = F / B;   // blocks per F-wide row
constexpr int MAXSEQ = 512;
constexpr float LN_EPS = 1e-12f;

struct QuantMat {
    const uint8_t* q;
    const float* s;
    const float* o;
};

struct Layer {
    QuantMat wq, wk, wv, wo, wu, wd;
    const float *bq, *bk, *bv, *bo, *bu, *bd;
    const float *ln1_g, *ln1_b, *ln2_g, *ln2_b;
};

struct Layout {
    QuantMat word, pos, type;
    const float *embln_g, *embln_b;
    Layer layer[L];
};

const uint8_t* cursor_base;
size_t cursor;

const uint8_t* take(size_t bytes) {
    const uint8_t* p = cursor_base + cursor;
    cursor += bytes;
    return p;
}

QuantMat take_quant(int rows, int cols) {
    QuantMat m;
    const int nb = cols / B;
    m.q = take((size_t)rows * cols);
    m.s = (const float*)take((size_t)rows * nb * 4);
    m.o = (const float*)take((size_t)rows * nb * 4);
    return m;
}

void fill_layout(const uint8_t* blob, Layout& lay) {
    cursor_base = blob;
    cursor = 0;
    lay.word = take_quant(V, D);
    lay.pos = take_quant(P, D);
    lay.type = take_quant(T, D);
    lay.embln_g = (const float*)take(D * 4);
    lay.embln_b = (const float*)take(D * 4);
    for (int i = 0; i < L; i++) {
        Layer& ly = lay.layer[i];
        ly.wq = take_quant(D, D); ly.bq = (const float*)take(D * 4);
        ly.wk = take_quant(D, D); ly.bk = (const float*)take(D * 4);
        ly.wv = take_quant(D, D); ly.bv = (const float*)take(D * 4);
        ly.wo = take_quant(D, D); ly.bo = (const float*)take(D * 4);
        ly.ln1_g = (const float*)take(D * 4);
        ly.ln1_b = (const float*)take(D * 4);
        ly.wu = take_quant(F, D); ly.bu = (const float*)take(F * 4);
        ly.wd = take_quant(D, F); ly.bd = (const float*)take(D * 4);
        ly.ln2_g = (const float*)take(D * 4);
        ly.ln2_b = (const float*)take(D * 4);
    }
}

// Fused u8 GEMV: y = A x + bias, A block-affine u8 (rows x cols).
void gemv(const QuantMat& m, int rows, int cols, const float* x, const float* bias, float* y) {
    const int nblocks = cols / B;
    float xsums[NBF];
    for (int b = 0; b < nblocks; b++) {
        float s = 0.f;
        for (int c = 0; c < B; c++) s += x[b * B + c];
        xsums[b] = s;
    }
#ifdef __wasm_simd128__
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = m.q + (size_t)r * cols;
        const float* srow = m.s + (size_t)r * nblocks;
        const float* orow = m.o + (size_t)r * nblocks;
        float acc = bias ? bias[r] : 0.f;
        for (int b = 0; b < nblocks; b++) {
            const uint8_t* ab = arow + b * B;
            const float* xb = x + b * B;
            v128_t vacc = wasm_f32x4_const_splat(0.f);
            for (int c = 0; c < B; c += 16) {
                const v128_t bytes = wasm_v128_load(ab + c);
                const v128_t lo16 = wasm_u16x8_extend_low_u8x16(bytes);
                const v128_t hi16 = wasm_u16x8_extend_high_u8x16(bytes);
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(
                    wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(lo16)), wasm_v128_load(xb + c)));
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(
                    wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(lo16)), wasm_v128_load(xb + c + 4)));
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(
                    wasm_f32x4_convert_u32x4(wasm_u32x4_extend_low_u16x8(hi16)), wasm_v128_load(xb + c + 8)));
                vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(
                    wasm_f32x4_convert_u32x4(wasm_u32x4_extend_high_u16x8(hi16)), wasm_v128_load(xb + c + 12)));
            }
            const float dot = wasm_f32x4_extract_lane(vacc, 0) + wasm_f32x4_extract_lane(vacc, 1)
                + wasm_f32x4_extract_lane(vacc, 2) + wasm_f32x4_extract_lane(vacc, 3);
            acc += srow[b] * dot + orow[b] * xsums[b];
        }
        y[r] = acc;
    }
#else
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = m.q + (size_t)r * cols;
        float acc = bias ? bias[r] : 0.f;
        for (int b = 0; b < nblocks; b++) {
            float dot = 0.f;
            for (int c = 0; c < B; c++) dot += (float)arow[b * B + c] * x[b * B + c];
            acc += m.s[(size_t)r * nblocks + b] * dot + m.o[(size_t)r * nblocks + b] * xsums[b];
        }
        y[r] = acc;
    }
#endif
}

// Dequantize one u8 row into f32 (embedding gather).
void dequant_row(const QuantMat& m, int row, float* out) {
    const uint8_t* q = m.q + (size_t)row * D;
    const float* s = m.s + (size_t)row * NB;
    const float* o = m.o + (size_t)row * NB;
    for (int b = 0; b < NB; b++) {
        for (int c = 0; c < B; c++) out[b * B + c] = (float)q[b * B + c] * s[b] + o[b];
    }
}

void layernorm(float* x, const float* g, const float* bta) {
    float mu = 0.f;
    for (int d = 0; d < D; d++) mu += x[d];
    mu /= D;
    float var = 0.f;
    for (int d = 0; d < D; d++) { const float dv = x[d] - mu; var += dv * dv; }
    var /= D;
    const float inv = 1.f / sqrtf(var + LN_EPS);
    for (int d = 0; d < D; d++) x[d] = (x[d] - mu) * inv * g[d] + bta[d];
}

inline float gelu(float v) {
    return 0.5f * v * (1.f + erff(v * 0.70710678f));
}

// Working buffers (BSS; single query at a time).
float bx[MAXSEQ * D];
float bq_[MAXSEQ * D];
float bk_[MAXSEQ * D];
float bv_[MAXSEQ * D];
float bctx[MAXSEQ * D];
float btmp[D];
float bh[F];
float bscores[MAXSEQ];

} // namespace

extern "C" {

// ids: i32 token ids. outHidden: seq*D floats (final hidden states).
// dbgStages: null, or (1+L)*seq*D floats — after-embedding-LN plus each
// layer's output, for stage parity against the torch references.
// Returns seq on success, negative on error.
int encoder_forward(const uint8_t* blob, const int* ids, int seq, float* outHidden, float* dbgStages) {
    if (seq < 1 || seq > MAXSEQ) return -1;
    Layout lay;
    fill_layout(blob, lay);

    for (int t = 0; t < seq; t++) {
        const int id = ids[t];
        if (id < 0 || id >= V) return -2;
        float* x = bx + t * D;
        dequant_row(lay.word, id, x);
        dequant_row(lay.pos, t, btmp);
        for (int d = 0; d < D; d++) x[d] += btmp[d];
        dequant_row(lay.type, 0, btmp);
        for (int d = 0; d < D; d++) x[d] += btmp[d];
        layernorm(x, lay.embln_g, lay.embln_b);
    }
    if (dbgStages) {
        for (int i = 0; i < seq * D; i++) dbgStages[i] = bx[i];
    }

    const float invSqrtHd = 1.f / sqrtf((float)HD);
    for (int li = 0; li < L; li++) {
        const Layer& ly = lay.layer[li];
        for (int t = 0; t < seq; t++) {
            gemv(ly.wq, D, D, bx + t * D, ly.bq, bq_ + t * D);
            gemv(ly.wk, D, D, bx + t * D, ly.bk, bk_ + t * D);
            gemv(ly.wv, D, D, bx + t * D, ly.bv, bv_ + t * D);
        }
        for (int h = 0; h < H; h++) {
            const int off = h * HD;
            for (int ti = 0; ti < seq; ti++) {
                const float* qv = bq_ + ti * D + off;
                float maxScore = -1e30f;
                for (int tj = 0; tj < seq; tj++) {
                    const float* kv = bk_ + tj * D + off;
                    float s = 0.f;
                    for (int d = 0; d < HD; d++) s += qv[d] * kv[d];
                    s *= invSqrtHd;
                    bscores[tj] = s;
                    if (s > maxScore) maxScore = s;
                }
                float denom = 0.f;
                for (int tj = 0; tj < seq; tj++) {
                    bscores[tj] = expf(bscores[tj] - maxScore);
                    denom += bscores[tj];
                }
                float* out = bctx + ti * D + off;
                for (int d = 0; d < HD; d++) out[d] = 0.f;
                for (int tj = 0; tj < seq; tj++) {
                    const float a = bscores[tj] / denom;
                    const float* vv = bv_ + tj * D + off;
                    for (int d = 0; d < HD; d++) out[d] += a * vv[d];
                }
            }
        }
        for (int t = 0; t < seq; t++) {
            gemv(ly.wo, D, D, bctx + t * D, ly.bo, btmp);
            float* x = bx + t * D;
            for (int d = 0; d < D; d++) x[d] += btmp[d];
            layernorm(x, ly.ln1_g, ly.ln1_b);
            gemv(ly.wu, F, D, x, ly.bu, bh);
            for (int f = 0; f < F; f++) bh[f] = gelu(bh[f]);
            gemv(ly.wd, D, F, bh, ly.bd, btmp);
            for (int d = 0; d < D; d++) x[d] += btmp[d];
            layernorm(x, ly.ln2_g, ly.ln2_b);
        }
        if (dbgStages) {
            float* dst = dbgStages + (1 + li) * seq * D;
            for (int i = 0; i < seq * D; i++) dst[i] = bx[i];
        }
    }

    for (int i = 0; i < seq * D; i++) outHidden[i] = bx[i];
    return seq;
}

} // extern "C"
