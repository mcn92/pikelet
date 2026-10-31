// GPT-Neo decoder (TinyStories-33M shape) over block-affine weights (the
// inline encoder's u8 blocks of 64 or u4 blocks of 32, or u5 blocks of 32
// with f16 scale/offset; see quant_u5.py), reusing the encoder's integer
// GEMM: activations quantized to int16 per token, i32x4.dot_i16x8_s per
// weight block. Differences from
// the encoder: causal attention with a KV cache (one call processes tokens
// at positions pos0..pos0+seq-1 and appends their K/V), pre-LN blocks,
// unscaled scores (GPT-Neo), local layers windowed to 256, tanh GELU, final
// LayerNorm and a tied LM head (the wte table as a V x D GEMM).
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <wasm_simd128.h>

// PROFILE builds time each phase (decoder_profile() returns the ms totals):
// 0 embedding, 1 LayerNorm + residual adds, 2 Q/K/V GEMMs, 3 attention,
// 4 output GEMM, 5 FFN up GEMM, 6 GELU, 7 FFN down GEMM, 8 LM head.
#ifdef PROFILE
#include <emscripten.h>
double g_prof[9];
#define PROF(i, ...) { const double prof_t0 = emscripten_get_now(); __VA_ARGS__; g_prof[i] += emscripten_get_now() - prof_t0; }
#else
#define PROF(i, ...) { __VA_ARGS__; }
#endif

namespace {

constexpr int V = 50257, P = 2048, D = 768, F = 3072, L = 4, H = 16, HD = 48;
// Weight formats (g_fmt): 8 = u8 in blocks of 64, 4 = u4 in blocks of 32,
// both with f32 s/o; 5 = u5 in blocks of 32 with f16 s/o; 6 = u5 expanded
// in memory by decoder_expand_u5 (u8 codes in blocks of 32, f32 s/o): the
// u5 file's quality at near-u8 decode speed, for 10 bits per weight of RAM.
constexpr int B8 = 64, B4 = 32, QB = 32, WINDOW = 256;
constexpr int MAXCTX = 1024, MAXSEQ = 256;  // context / tokens per call
constexpr float LN_EPS = 1e-5f;

// q: codes (u8), or the low-nibble plane (u4, u5); h: u5's high-bit plane;
// s/o: f32 (u8, u4) or f16 bits (u5).
struct QuantMat { const uint8_t* q; const uint8_t* h; const void* s; const void* o; };
struct Layer {
    const float *ln1_g, *ln1_b, *ln2_g, *ln2_b, *bo, *bu, *bd;
    QuantMat wq, wk, wv, wo, wu, wd;
};
struct Layout { QuantMat wte, wpe; Layer layer[L]; const float *lnf_g, *lnf_b; };

const uint8_t* cur;
const float* take_f32(int n) { const float* p = (const float*)cur; cur += (size_t)n * 4; return p; }
int g_fmt;
QuantMat take_quant(int rows, int cols) {
    QuantMat m;
    const size_t n = (size_t)rows * cols;
    m.q = cur; cur += (g_fmt == 8 || g_fmt == 6) ? n : n / 2;
    m.h = cur; if (g_fmt == 5) cur += n / 8;
    const size_t nb = n / (g_fmt == 8 ? B8 : B4), sb = g_fmt == 5 ? 2 : 4;
    m.s = cur; cur += nb * sb;
    m.o = cur; cur += nb * sb;
    return m;
}

// f16 bits -> f32, exact for normals, subnormals and zero (no inf/NaN in a
// blob): sign to bit 31, exponent+mantissa into f32 position, rescale by
// 2^112. Branch-free.
inline float f16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)(h & 0x8000) << 16) | ((uint32_t)(h & 0x7FFF) << 13);
    return v.f * 5.192296858534828e33f;  // 2^112
}
template <int FMT> inline float load_scalar(const void* base, size_t i) {
    return FMT == 5 ? f16_to_f32(((const uint16_t*)base)[i]) : ((const float*)base)[i];
}
void fill_layout(const uint8_t* blob, Layout& lay) {
    cur = blob;
    lay.wte = take_quant(V, D);
    lay.wpe = take_quant(P, D);
    for (int i = 0; i < L; i++) {
        Layer& ly = lay.layer[i];
        ly.ln1_g = take_f32(D); ly.ln1_b = take_f32(D);
        ly.wq = take_quant(D, D); ly.wk = take_quant(D, D); ly.wv = take_quant(D, D);
        ly.wo = take_quant(D, D); ly.bo = take_f32(D);
        ly.ln2_g = take_f32(D); ly.ln2_b = take_f32(D);
        ly.wu = take_quant(F, D); ly.bu = take_f32(F);
        ly.wd = take_quant(D, F); ly.bd = take_f32(D);
    }
    lay.lnf_g = take_f32(D); lay.lnf_b = take_f32(D);
}

// Per-token int16 activations and per-block exact sums (the encoder's
// quantize_acts, for one block size).
alignas(16) int16_t g_xq[MAXSEQ * F];
float g_xs[MAXSEQ];
float g_xsum[MAXSEQ * (F / QB)];   // per B4 block
float g_xsum8[MAXSEQ * (F / B8)];  // per B8 block

void quantize_acts(const float* X, int seq, int cols) {
    const int nb = cols / QB;
    for (int t = 0; t < seq; t++) {
        const float* x = X + t * cols;
        float m = 0.f;
        for (int c = 0; c < cols; c++) { const float a = fabsf(x[c]); if (a > m) m = a; }
        const float inv = m > 0.f ? 32767.f / m : 0.f;
        g_xs[t] = m / 32767.f;
        int16_t* q = g_xq + t * cols;
        const v128_t vinv = wasm_f32x4_splat(inv);
        for (int c = 0; c < cols; c += 8) {
            const v128_t a = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(x + c), vinv)));
            const v128_t b = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(x + c + 4), vinv)));
            wasm_v128_store(q + c, wasm_i16x8_narrow_i32x4(a, b));
        }
        for (int b = 0; b < nb; b++) {
            float sum = 0.f;
            for (int c = 0; c < QB; c++) sum += x[b * QB + c];
            g_xsum[t * nb + b] = sum;
        }
        for (int b = 0; b < cols / B8; b++) g_xsum8[t * (cols / B8) + b] = g_xsum[t * nb + 2 * b] + g_xsum[t * nb + 2 * b + 1];
    }
}

// u5 high-bit plane: 16 bytes per 128 weights, bit g of byte j holding
// weight g*16 + j's bit 4, so the 16 lanes of half-block g need only one
// shift (fixed per g) and an AND to land the bit at value 16.
template <int G> inline v128_t high_bits(v128_t v) {
    const v128_t m = wasm_u8x16_splat(0x10);
    if constexpr (G < 4) return wasm_v128_and(wasm_i8x16_shl(v, 4 - G), m);
    else if constexpr (G == 4) return wasm_v128_and(v, m);
    else return wasm_v128_and(wasm_u8x16_shr(v, G - 4), m);
}

// The encoder's gemm_int_tile, plus u5: the nibble plane unpacks as u4 and
// the high bits are ORed in (high_bits); u5 blocks run four at a time so each
// block's half-block indices are compile-time constants.
template <int NT, int FMT>
void gemm_int_tile(const QuantMat& m, int rows, int cols, int t0, const float* bias, float* Y) {
    constexpr int BS = FMT == 8 ? B8 : B4;
    constexpr bool NIB = FMT == 4 || FMT == 5;  // nibble plane; else u8 codes
    constexpr int NV = BS / 8;
    const int nblocks = cols / BS;
    const size_t rowBytes = NIB ? (size_t)cols / 2 : (size_t)cols;
    const int16_t* xq0 = g_xq + t0 * cols;
    const float* xsum0 = (FMT == 8 ? g_xsum8 : g_xsum) + t0 * nblocks;
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = m.q + r * rowBytes;
        const uint8_t* hrow = m.h + r * ((size_t)cols / 8);
        const size_t sbase = (size_t)r * nblocks;
        v128_t facc[NT];
        float off[NT];
        for (int t = 0; t < NT; t++) { facc[t] = wasm_f32x4_const_splat(0.f); off[t] = 0.f; }
        auto block = [&]<int K>(int b) {
            v128_t w[NV];
            if constexpr (NIB) {
                const v128_t packed = wasm_v128_load(arow + b * (B4 / 2));
                v128_t lo = wasm_v128_and(packed, wasm_u8x16_splat(0x0F));
                v128_t hi = wasm_u8x16_shr(packed, 4);
                if constexpr (FMT == 5) {
                    const v128_t hv = wasm_v128_load(hrow + (b >> 2) * 16);
                    lo = wasm_v128_or(lo, high_bits<2 * K>(hv));
                    hi = wasm_v128_or(hi, high_bits<2 * K + 1>(hv));
                }
                w[0] = wasm_u16x8_extend_low_u8x16(lo);
                w[1] = wasm_u16x8_extend_high_u8x16(lo);
                w[2] = wasm_u16x8_extend_low_u8x16(hi);
                w[3] = wasm_u16x8_extend_high_u8x16(hi);
            } else {
                for (int c = 0; c < BS; c += 16) {
                    const v128_t bytes = wasm_v128_load(arow + b * BS + c);
                    w[c / 8] = wasm_u16x8_extend_low_u8x16(bytes);
                    w[c / 8 + 1] = wasm_u16x8_extend_high_u8x16(bytes);
                }
            }
            const v128_t sv = wasm_f32x4_splat(load_scalar<FMT>(m.s, sbase + b));
            const float ob = load_scalar<FMT>(m.o, sbase + b);
            for (int t = 0; t < NT; t++) {
                const int16_t* xq = xq0 + t * cols + b * BS;
                v128_t vacc = wasm_i32x4_dot_i16x8(w[0], wasm_v128_load(xq));
                for (int k = 1; k < NV; k++) vacc = wasm_i32x4_add(vacc, wasm_i32x4_dot_i16x8(w[k], wasm_v128_load(xq + 8 * k)));
                facc[t] = wasm_f32x4_add(facc[t], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vacc), sv));
                off[t] += ob * xsum0[t * nblocks + b];
            }
        };
        if constexpr (FMT == 5) {
            for (int b = 0; b < nblocks; b += 4) {  // nblocks % 4 == 0 (cols % 128)
                block.template operator()<0>(b);
                block.template operator()<1>(b + 1);
                block.template operator()<2>(b + 2);
                block.template operator()<3>(b + 3);
            }
        } else {
            for (int b = 0; b < nblocks; b++) block.template operator()<0>(b);
        }
        for (int t = 0; t < NT; t++) {
            const float dot = wasm_f32x4_extract_lane(facc[t], 0) + wasm_f32x4_extract_lane(facc[t], 1)
                + wasm_f32x4_extract_lane(facc[t], 2) + wasm_f32x4_extract_lane(facc[t], 3);
            Y[(t0 + t) * rows + r] = (bias ? bias[r] : 0.f) + g_xs[t0 + t] * dot + off[t];
        }
    }
}

template <int FMT>
void gemm_q(const QuantMat& m, int rows, int cols, int seq, const float* bias, float* Y) {
    int t = 0;
    for (; t + 8 <= seq; t += 8) gemm_int_tile<8, FMT>(m, rows, cols, t, bias, Y);
    for (; t + 4 <= seq; t += 4) gemm_int_tile<4, FMT>(m, rows, cols, t, bias, Y);
    for (; t + 2 <= seq; t += 2) gemm_int_tile<2, FMT>(m, rows, cols, t, bias, Y);
    for (; t < seq; t++) gemm_int_tile<1, FMT>(m, rows, cols, t, bias, Y);
}

// Y = A X (+ bias); quantize=false reuses the previous call's activations.
void gemm(const QuantMat& m, int rows, int cols, const float* X, int seq, const float* bias, float* Y, bool quantize = true) {
    if (quantize) quantize_acts(X, seq, cols);
    if (g_fmt == 6) gemm_q<6>(m, rows, cols, seq, bias, Y);
    else if (g_fmt == 5) gemm_q<5>(m, rows, cols, seq, bias, Y);
    else if (g_fmt == 4) gemm_q<4>(m, rows, cols, seq, bias, Y);
    else gemm_q<8>(m, rows, cols, seq, bias, Y);
}

void dequant_row(const QuantMat& m, int row, float* out) {
    const int bs = g_fmt == 8 ? B8 : B4;
    const bool nib = g_fmt == 4 || g_fmt == 5;
    const uint8_t* q = m.q + (size_t)row * (nib ? D / 2 : D);
    const uint8_t* h = m.h + (size_t)row * (D / 8);
    const size_t sb = (size_t)row * (D / bs);
    for (int c = 0; c < D; c++) {
        const int b = c / bs, k = c % bs;
        int w = !nib ? q[c] : (k < 16 ? q[b * 16 + k] & 0x0F : q[b * 16 + k - 16] >> 4);
        if (g_fmt == 5) w |= ((h[(c / 128) * 16 + c % 16] >> ((c % 128) / 16)) & 1) << 4;
        const float s = g_fmt == 5 ? load_scalar<5>(m.s, sb + b) : load_scalar<8>(m.s, sb + b);
        const float o = g_fmt == 5 ? load_scalar<5>(m.o, sb + b) : load_scalar<8>(m.o, sb + b);
        out[c] = (float)w * s + o;
    }
}

void layernorm(const float* x, const float* g, const float* bta, float* y) {
    float mu = 0.f;
    for (int d = 0; d < D; d++) mu += x[d];
    mu /= D;
    float var = 0.f;
    for (int d = 0; d < D; d++) { const float dv = x[d] - mu; var += dv * dv; }
    var /= D;
    const float inv = 1.f / sqrtf(var + LN_EPS);
    for (int d = 0; d < D; d++) y[d] = (x[d] - mu) * inv * g[d] + bta[d];
}

// exp(x) for x in [-88, 0] (the encoder's Cephes expf: n = round(x log2 e),
// two-part ln 2 reduction, degree-6 polynomial, 2^n through the exponent).
inline v128_t exp_neg_f32x4(v128_t x) {
    x = wasm_f32x4_max(x, wasm_f32x4_splat(-87.3f));
    const v128_t n = wasm_f32x4_nearest(wasm_f32x4_mul(x, wasm_f32x4_splat(1.44269504f)));
    v128_t r = wasm_f32x4_sub(x, wasm_f32x4_mul(n, wasm_f32x4_splat(0.693359375f)));
    r = wasm_f32x4_sub(r, wasm_f32x4_mul(n, wasm_f32x4_splat(-2.12194440e-4f)));
    v128_t p = wasm_f32x4_splat(1.9875691500e-4f);
    p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(1.3981999507e-3f));
    p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(8.3334519073e-3f));
    p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(4.1665795894e-2f));
    p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(1.6666665459e-1f));
    p = wasm_f32x4_add(wasm_f32x4_mul(p, r), wasm_f32x4_splat(5.0000001201e-1f));
    p = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(p, wasm_f32x4_mul(r, r)), r), wasm_f32x4_splat(1.f));
    const v128_t pow2n = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(n), wasm_i32x4_splat(127)), 23);
    return wasm_f32x4_mul(p, pow2n);
}

// GPT-2's tanh GELU, 0.5 v (1 + tanh(0.798 (v + 0.0447 v^3))), four lanes at
// a time: tanh|y| = (1 - e) / (1 + e) with e = exp(-2|y|), sign restored.
inline v128_t gelu_new_f32x4(v128_t v) {
    const v128_t y = wasm_f32x4_mul(wasm_f32x4_splat(0.7978845608f),
        wasm_f32x4_add(v, wasm_f32x4_mul(wasm_f32x4_splat(0.044715f), wasm_f32x4_mul(v, wasm_f32x4_mul(v, v)))));
    const v128_t e = exp_neg_f32x4(wasm_f32x4_mul(wasm_f32x4_splat(-2.f), wasm_f32x4_abs(y)));
    const v128_t one = wasm_f32x4_splat(1.f);
    const v128_t th = wasm_f32x4_div(wasm_f32x4_sub(one, e), wasm_f32x4_add(one, e));
    const v128_t t = wasm_v128_or(th, wasm_v128_and(y, wasm_i32x4_splat((int32_t)0x80000000)));
    return wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_splat(0.5f), v), wasm_f32x4_add(one, t));
}

inline float hsum(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)
        + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

constexpr int HV = HD / 4;  // f32x4 vectors per head

// One query row against the cached keys j0..i of one head: scores, softmax,
// weighted sum of the cached values into out. The query and the output stay
// in registers (12 vectors each); exp is exp_neg_f32x4 over the whole score
// row, with lanes past i written but never summed.
void attend(const float* q, const float* kc, const float* vc, int j0, int i, float* sc, float* out) {
    v128_t qv[HV];
    for (int k = 0; k < HV; k++) qv[k] = wasm_v128_load(q + 4 * k);
    const int n = i - j0 + 1;
    float mx = -1e30f;
    for (int j = 0; j < n; j++) {
        const float* kv = kc + (size_t)(j0 + j) * D;
        v128_t acc = wasm_f32x4_mul(qv[0], wasm_v128_load(kv));
        for (int k = 1; k < HV; k++) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(qv[k], wasm_v128_load(kv + 4 * k)));
        const float s = hsum(acc);
        sc[j] = s;
        if (s > mx) mx = s;
    }
    const v128_t vmx = wasm_f32x4_splat(mx);
    for (int j = 0; j < n; j += 4) wasm_v128_store(sc + j, exp_neg_f32x4(wasm_f32x4_sub(wasm_v128_load(sc + j), vmx)));
    float den = 0.f;
    for (int j = 0; j < n; j++) den += sc[j];
    const float inv = 1.f / den;
    v128_t ov[HV];
    for (int k = 0; k < HV; k++) ov[k] = wasm_f32x4_const_splat(0.f);
    for (int j = 0; j < n; j++) {
        const v128_t a = wasm_f32x4_splat(sc[j] * inv);
        const float* vv = vc + (size_t)(j0 + j) * D;
        for (int k = 0; k < HV; k++) ov[k] = wasm_f32x4_add(ov[k], wasm_f32x4_mul(a, wasm_v128_load(vv + 4 * k)));
    }
    for (int k = 0; k < HV; k++) wasm_v128_store(out + 4 * k, ov[k]);
}

float bx[MAXSEQ * D];   // residual stream
float bn[MAXSEQ * D];   // LayerNorm output
float bq[MAXSEQ * D];
float bctx[MAXSEQ * D];
float bo[MAXSEQ * D];
float bh[MAXSEQ * F];
float kcache[L][MAXCTX * D];
float vcache[L][MAXCTX * D];
alignas(16) float scores[MAXCTX + 4];  // + one vector of exp padding

// Vocabulary shortlist (decoder_set_shortlist): A (V x r, u8 blocks of 64,
// quant layout) and B (r x D, f32) from export_shortlist.py. Coarse scores
// A (B x) rank every token; the top g_slK are rescored exactly against wte.
const uint8_t* g_slA;
const float* g_slB;
int g_slR, g_slK;
float g_coarse[V];
float g_y[256];
int g_top[1024];

// Exact logit of one wte row, by the same integer tile as the full LM head
// (same quantized activations, same order: bit-identical to it).
template <int FMT>
float wte_row_logit(const QuantMat& m, int row) {
    const int cols = D, nb = cols / (FMT == 8 ? B8 : B4);
    const size_t qbytes = (FMT == 4 || FMT == 5) ? (size_t)cols / 2 : (size_t)cols, sbytes = FMT == 5 ? 2 : 4;
    QuantMat r;
    r.q = m.q + (size_t)row * qbytes;
    r.h = m.h + (size_t)row * (cols / 8);
    r.s = (const uint8_t*)m.s + (size_t)row * nb * sbytes;
    r.o = (const uint8_t*)m.o + (size_t)row * nb * sbytes;
    float y;
    gemm_int_tile<1, FMT>(r, 1, cols, 0, nullptr, &y);
    return y;
}

// Shortlist LM head for one token's final hidden state x: logits[] gets the
// exact logit for the top g_slK coarse candidates and -inf elsewhere.
void shortlist_head(const QuantMat& wte, const float* x, float* logits) {
    const int R = g_slR, K = g_slK;
    for (int k = 0; k < R; k++) {  // y = B x
        const float* b = g_slB + (size_t)k * D;
        v128_t acc = wasm_f32x4_const_splat(0.f);
        for (int d = 0; d < D; d += 4) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(b + d), wasm_v128_load(x + d)));
        g_y[k] = hsum(acc);
    }
    QuantMat A;
    A.q = g_slA;
    A.h = nullptr;
    A.s = g_slA + (size_t)V * R;
    A.o = (const uint8_t*)A.s + (size_t)V * (R / B8) * 4;
    const int fmt = g_fmt;
    g_fmt = 8;
    gemm(A, V, R, g_y, 1, nullptr, g_coarse);
    g_fmt = fmt;
    // Top K by coarse score: a min-heap of indices.
    int n = 0;
    auto less = [](int a, int b) { return g_coarse[a] < g_coarse[b]; };
    auto down = [&](int i) {
        for (;;) {
            int c = 2 * i + 1;
            if (c >= n) return;
            if (c + 1 < n && less(g_top[c + 1], g_top[c])) c++;
            if (!less(g_top[c], g_top[i])) return;
            const int t = g_top[c]; g_top[c] = g_top[i]; g_top[i] = t; i = c;
        }
    };
    for (int v = 0; v < V; v++) {
        if (n < K) {
            g_top[n++] = v;
            for (int i = n - 1; i > 0;) {
                const int p = (i - 1) / 2;
                if (!less(g_top[i], g_top[p])) break;
                const int t = g_top[p]; g_top[p] = g_top[i]; g_top[i] = t; i = p;
            }
        } else if (g_coarse[v] > g_coarse[g_top[0]]) {
            g_top[0] = v;
            down(0);
        }
    }
    for (int v = 0; v < V; v++) logits[v] = -INFINITY;
    quantize_acts(x, 1, D);
    for (int k = 0; k < n; k++) {
        const int v = g_top[k];
        logits[v] = fmt == 6 ? wte_row_logit<6>(wte, v) : fmt == 5 ? wte_row_logit<5>(wte, v)
            : fmt == 4 ? wte_row_logit<4>(wte, v) : wte_row_logit<8>(wte, v);
    }
}

} // namespace

extern "C" {

// Installs (or, with a null sl, removes) the vocabulary shortlist used when
// decoder_forward is called without allLogits. r: the table's rank (a
// multiple of 64, at most 256); k: candidates rescored exactly (at most
// 1024). Returns 0, or -1 on bad arguments.
int decoder_set_shortlist(const uint8_t* sl, int r, int k) {
    if (!sl) { g_slA = nullptr; return 0; }
    if (r < 64 || r > 256 || r % 64 || k < 1 || k > 1024) return -1;
    g_slA = sl;
    g_slR = r;
    g_slK = k;
    g_slB = (const float*)(sl + (size_t)V * r + (size_t)V * (r / B8) * 8);
    return 0;
}

// Runs tokens ids[0..seq) at positions pos0.. (the cache must hold
// positions 0..pos0-1 from earlier calls) and writes the last token's V
// logits (allLogits: every token's, seq*V). bits: 8, 4, 5 or 6, the blob format. Returns 0, or negative on error.
// Expands a u5 blob (format 5) into format 6 at dst, in the same order;
// with dst null, only returns the size dst needs. Returns bytes written.
size_t decoder_expand_u5(const uint8_t* src, uint8_t* dst) {
    g_fmt = 5;
    Layout in;
    fill_layout(src, in);
    size_t pos = 0;
    auto vec = [&](const float* v, int n) {
        if (dst) __builtin_memcpy(dst + pos, v, (size_t)n * 4);
        pos += (size_t)n * 4;
    };
    auto quant = [&](const QuantMat& m, int rows, int cols) {
        const int nb = cols / B4;
        uint8_t* q = dst ? dst + pos : nullptr;
        pos += (size_t)rows * cols;
        float* sv = dst ? (float*)(dst + pos) : nullptr;
        pos += (size_t)rows * nb * 4;
        float* ov = dst ? (float*)(dst + pos) : nullptr;
        pos += (size_t)rows * nb * 4;
        if (!dst) return;
        for (int r = 0; r < rows; r++) {
            const uint8_t* nibs = m.q + (size_t)r * cols / 2;
            const uint8_t* h = m.h + (size_t)r * cols / 8;
            for (int c = 0; c < cols; c++) {
                const int b = c / B4, k = c % B4;
                const int lo = k < 16 ? nibs[b * 16 + k] & 0x0F : nibs[b * 16 + k - 16] >> 4;
                q[(size_t)r * cols + c] = (uint8_t)(lo | (((h[(c / 128) * 16 + c % 16] >> ((c % 128) / 16)) & 1) << 4));
            }
            for (int b = 0; b < nb; b++) {
                sv[(size_t)r * nb + b] = load_scalar<5>(m.s, (size_t)r * nb + b);
                ov[(size_t)r * nb + b] = load_scalar<5>(m.o, (size_t)r * nb + b);
            }
        }
    };
    quant(in.wte, V, D);
    quant(in.wpe, P, D);
    for (int i = 0; i < L; i++) {
        const Layer& ly = in.layer[i];
        vec(ly.ln1_g, D); vec(ly.ln1_b, D);
        quant(ly.wq, D, D); quant(ly.wk, D, D); quant(ly.wv, D, D);
        quant(ly.wo, D, D); vec(ly.bo, D);
        vec(ly.ln2_g, D); vec(ly.ln2_b, D);
        quant(ly.wu, F, D); vec(ly.bu, F);
        quant(ly.wd, D, F); vec(ly.bd, D);
    }
    vec(in.lnf_g, D); vec(in.lnf_b, D);
    return pos;
}

int decoder_forward(const uint8_t* blob, const int* ids, int seq, int pos0, float* logits, int bits, int allLogits) {
    if (seq < 1 || seq > MAXSEQ || pos0 < 0 || pos0 + seq > MAXCTX) return -1;
    if (bits != 8 && bits != 4 && bits != 5 && bits != 6) return -4;
    g_fmt = bits;
    Layout lay;
    fill_layout(blob, lay);
    for (int t = 0; t < seq; t++) if (ids[t] < 0 || ids[t] >= V) return -2;
    PROF(0,
    for (int t = 0; t < seq; t++) {
        float* x = bx + t * D;
        dequant_row(lay.wte, ids[t], x);
        dequant_row(lay.wpe, pos0 + t, bo);
        for (int d = 0; d < D; d++) x[d] += bo[d];
    })
    for (int li = 0; li < L; li++) {
        const Layer& ly = lay.layer[li];
        const bool local = li % 2 == 1;
        PROF(1, for (int t = 0; t < seq; t++) layernorm(bx + t * D, ly.ln1_g, ly.ln1_b, bn + t * D))
        float* kc = kcache[li];
        float* vc = vcache[li];
        PROF(2,
        gemm(ly.wq, D, D, bn, seq, nullptr, bq);
        gemm(ly.wk, D, D, bn, seq, nullptr, kc + (size_t)pos0 * D, false);
        gemm(ly.wv, D, D, bn, seq, nullptr, vc + (size_t)pos0 * D, false))
        PROF(3,
        for (int t = 0; t < seq; t++) {
            const int i = pos0 + t;
            const int j0 = local && i >= WINDOW ? i - WINDOW + 1 : 0;
            for (int h = 0; h < H; h++)
                attend(bq + t * D + h * HD, kc + h * HD, vc + h * HD, j0, i, scores, bctx + t * D + h * HD);
        })
        PROF(4, gemm(ly.wo, D, D, bctx, seq, ly.bo, bo))
        PROF(1,
        for (int i = 0; i < seq * D; i++) bx[i] += bo[i];
        for (int t = 0; t < seq; t++) layernorm(bx + t * D, ly.ln2_g, ly.ln2_b, bn + t * D))
        PROF(5, gemm(ly.wu, F, D, bn, seq, ly.bu, bh))
        PROF(6, for (int i = 0; i < seq * F; i += 4) wasm_v128_store(bh + i, gelu_new_f32x4(wasm_v128_load(bh + i))))
        PROF(7, gemm(ly.wd, D, F, bh, seq, ly.bd, bo))
        PROF(1, for (int i = 0; i < seq * D; i++) bx[i] += bo[i])
    }
    const int first = allLogits ? 0 : seq - 1;
    PROF(8,
    for (int t = first; t < seq; t++) layernorm(bx + t * D, lay.lnf_g, lay.lnf_b, bn + (t - first) * D);
    if (g_slA && !allLogits) shortlist_head(lay.wte, bn, logits);
    else gemm(lay.wte, V, D, bn, seq - first, nullptr, logits))
    return 0;
}

#ifdef PROFILE
double* decoder_profile() { return g_prof; }
#endif

} // extern "C"
