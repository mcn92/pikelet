// Llama-architecture decoder (SmolLM2 and the like) over the same
// block-affine weights and integer GEMM as decoder.cpp: activations quantized
// to int16 per token, i32x4.dot_i16x8_s per weight block. Differences from
// the GPT-Neo kernel: the model's shape comes from the blob's header
// (export_llama.py) instead of constants, so buffers are allocated by
// llama_init; RMSNorm instead of LayerNorm; rotary position embeddings
// (rotate-half, as Hugging Face's Llama) instead of a position table;
// grouped-query attention scaled by 1/sqrt(HD) with no window; a SwiGLU FFN
// (silu(gate) * up, then down) and no biases. Weight formats: 8 (u8 blocks
// of 64, f32 s/o) and 6 (u5 expanded at load by llama_expand_u5: u8 codes in
// blocks of 32, f32 s/o). The u5 file format (5) is only read by the
// expansion.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <wasm_simd128.h>

namespace {

constexpr int B8 = 64, QB = 32, MAXSEQ = 256, MAXL = 64, MAXHD = 128;
constexpr int32_t MAGIC = 0x314D4C4C;  // "LLM1"

int V, D, F, L, H, KVH, HD, KVD, QD, MAXCTX, g_fmt;
bool g_tied;
float g_theta, g_eps;

struct QuantMat { const uint8_t* q; const uint8_t* h; const void* s; const void* o; };
struct Layer { const float *attn_norm, *ffn_norm; QuantMat wq, wk, wv, wo, wg, wu, wd; };
struct Layout { QuantMat embed, head; Layer layer[MAXL]; const float* norm; };

const uint8_t* cur;
const float* take_f32(int n) { const float* p = (const float*)cur; cur += (size_t)n * 4; return p; }
// Format 5 (u5 file): nibble plane rows*cols/2, high-bit plane
// rows*ceil(cols/128)*16, f16 s/o per block of 32. Formats 8 and 6: u8
// codes, f32 s/o per block of 64 (8) or 32 (6).
QuantMat take_quant(int fmt, int rows, int cols) {
    QuantMat m;
    const size_t n = (size_t)rows * cols;
    m.q = cur; cur += fmt == 5 ? n / 2 : n;
    m.h = cur; if (fmt == 5) cur += (size_t)rows * ((cols + 127) / 128) * 16;
    const size_t nb = n / (fmt == 8 ? B8 : QB), sb = fmt == 5 ? 2 : 4;
    m.s = cur; cur += nb * sb;
    m.o = cur; cur += nb * sb;
    return m;
}
void fill_layout(const uint8_t* blob, int fmt, Layout& lay) {
    cur = blob + 64;
    lay.embed = take_quant(fmt, V, D);
    for (int i = 0; i < L; i++) {
        Layer& ly = lay.layer[i];
        ly.attn_norm = take_f32(D);
        ly.wq = take_quant(fmt, QD, D); ly.wk = take_quant(fmt, KVD, D); ly.wv = take_quant(fmt, KVD, D);
        ly.wo = take_quant(fmt, D, QD);
        ly.ffn_norm = take_f32(D);
        ly.wg = take_quant(fmt, F, D); ly.wu = take_quant(fmt, F, D); ly.wd = take_quant(fmt, D, F);
    }
    lay.norm = take_f32(D);
    lay.head = g_tied ? lay.embed : take_quant(fmt, V, D);
}

// Per-token int16 activations and per-block exact sums.
int16_t* g_xq;
float* g_xs;
float* g_xsum;   // per block of 32
float* g_xsum8;  // per block of 64

void quantize_acts(const float* X, int seq, int cols) {
    const int nb = cols / QB;
    for (int t = 0; t < seq; t++) {
        const float* x = X + (size_t)t * cols;
        float m = 0.f;
        for (int c = 0; c < cols; c++) { const float a = fabsf(x[c]); if (a > m) m = a; }
        const float inv = m > 0.f ? 32767.f / m : 0.f;
        g_xs[t] = m / 32767.f;
        int16_t* q = g_xq + (size_t)t * cols;
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

// decoder.cpp's gemm_int_tile for u8 codes (formats 8 and 6).
template <int NT, int FMT>
void gemm_int_tile(const QuantMat& m, int rows, int cols, int t0, float* Y) {
    constexpr int BS = FMT == 8 ? B8 : QB;
    constexpr int NV = BS / 8;
    const int nblocks = cols / BS;
    const int16_t* xq0 = g_xq + (size_t)t0 * cols;
    const float* xsum0 = (FMT == 8 ? g_xsum8 : g_xsum) + t0 * nblocks;
    const float* S = (const float*)m.s;
    const float* O = (const float*)m.o;
    for (int r = 0; r < rows; r++) {
        const uint8_t* arow = m.q + (size_t)r * cols;
        const size_t sbase = (size_t)r * nblocks;
        v128_t facc[NT];
        float off[NT];
        for (int t = 0; t < NT; t++) { facc[t] = wasm_f32x4_const_splat(0.f); off[t] = 0.f; }
        for (int b = 0; b < nblocks; b++) {
            v128_t w[NV];
            for (int c = 0; c < BS; c += 16) {
                const v128_t bytes = wasm_v128_load(arow + b * BS + c);
                w[c / 8] = wasm_u16x8_extend_low_u8x16(bytes);
                w[c / 8 + 1] = wasm_u16x8_extend_high_u8x16(bytes);
            }
            const v128_t sv = wasm_f32x4_splat(S[sbase + b]);
            const float ob = O[sbase + b];
            for (int t = 0; t < NT; t++) {
                const int16_t* xq = xq0 + (size_t)t * cols + b * BS;
                v128_t vacc = wasm_i32x4_dot_i16x8(w[0], wasm_v128_load(xq));
                for (int k = 1; k < NV; k++) vacc = wasm_i32x4_add(vacc, wasm_i32x4_dot_i16x8(w[k], wasm_v128_load(xq + 8 * k)));
                facc[t] = wasm_f32x4_add(facc[t], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vacc), sv));
                off[t] += ob * xsum0[t * nblocks + b];
            }
        }
        for (int t = 0; t < NT; t++) {
            const float dot = wasm_f32x4_extract_lane(facc[t], 0) + wasm_f32x4_extract_lane(facc[t], 1)
                + wasm_f32x4_extract_lane(facc[t], 2) + wasm_f32x4_extract_lane(facc[t], 3);
            Y[(size_t)(t0 + t) * rows + r] = g_xs[t0 + t] * dot + off[t];
        }
    }
}

template <int FMT>
void gemm_q(const QuantMat& m, int rows, int cols, int seq, float* Y) {
    int t = 0;
    for (; t + 8 <= seq; t += 8) gemm_int_tile<8, FMT>(m, rows, cols, t, Y);
    for (; t + 4 <= seq; t += 4) gemm_int_tile<4, FMT>(m, rows, cols, t, Y);
    for (; t + 2 <= seq; t += 2) gemm_int_tile<2, FMT>(m, rows, cols, t, Y);
    for (; t < seq; t++) gemm_int_tile<1, FMT>(m, rows, cols, t, Y);
}

// Y = A X; quantize=false reuses the previous call's activations.
void gemm(const QuantMat& m, int rows, int cols, const float* X, int seq, float* Y, bool quantize = true) {
    if (quantize) quantize_acts(X, seq, cols);
    if (g_fmt == 6) gemm_q<6>(m, rows, cols, seq, Y);
    else gemm_q<8>(m, rows, cols, seq, Y);
}

void dequant_row(const QuantMat& m, int row, int cols, float* out) {
    const int bs = g_fmt == 8 ? B8 : QB;
    const uint8_t* q = m.q + (size_t)row * cols;
    const float* S = (const float*)m.s + (size_t)row * (cols / bs);
    const float* O = (const float*)m.o + (size_t)row * (cols / bs);
    for (int c = 0; c < cols; c++) out[c] = (float)q[c] * S[c / bs] + O[c / bs];
}

void rmsnorm(const float* x, const float* g, float* y) {
    float ss = 0.f;
    for (int d = 0; d < D; d++) ss += x[d] * x[d];
    const float inv = 1.f / sqrtf(ss / D + g_eps);
    for (int d = 0; d < D; d++) y[d] = g[d] * (x[d] * inv);
}

// exp(x) for x in [-88, 0] (decoder.cpp's Cephes expf).
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

// silu(g) * u, four lanes at a time: sigmoid(g) = 1 / (1 + e) for g >= 0 and
// e / (1 + e) below, with e = exp(-|g|).
inline v128_t swiglu_f32x4(v128_t g, v128_t u) {
    const v128_t one = wasm_f32x4_splat(1.f);
    const v128_t e = exp_neg_f32x4(wasm_f32x4_neg(wasm_f32x4_abs(g)));
    const v128_t den = wasm_f32x4_add(one, e);
    const v128_t sig = wasm_v128_bitselect(wasm_f32x4_div(one, den), wasm_f32x4_div(e, den), wasm_f32x4_ge(g, wasm_f32x4_const_splat(0.f)));
    return wasm_f32x4_mul(wasm_f32x4_mul(g, sig), u);
}

inline float hsum(v128_t v) {
    return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1)
        + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

float* g_cos;  // MAXCTX x HD/2
float* g_sin;

// Rotate-half RoPE on one head vector at position pos: pairs (i, i + HD/2).
void rope(float* x, int pos) {
    const int half = HD / 2;
    const float* c = g_cos + (size_t)pos * half;
    const float* s = g_sin + (size_t)pos * half;
    for (int i = 0; i < half; i++) {
        const float a = x[i], b = x[i + half];
        x[i] = a * c[i] - b * s[i];
        x[i + half] = b * c[i] + a * s[i];
    }
}

// One query row (already scaled) against cached keys 0..i of one KV head
// (row stride KVD): scores, softmax, weighted sum of the cached values.
void attend(const float* q, const float* kc, const float* vc, int i, float* sc, float* out) {
    const int hv = HD / 4;
    v128_t qv[MAXHD / 4];
    for (int k = 0; k < hv; k++) qv[k] = wasm_v128_load(q + 4 * k);
    const int n = i + 1;
    float mx = -1e30f;
    for (int j = 0; j < n; j++) {
        const float* kv = kc + (size_t)j * KVD;
        v128_t acc = wasm_f32x4_mul(qv[0], wasm_v128_load(kv));
        for (int k = 1; k < hv; k++) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(qv[k], wasm_v128_load(kv + 4 * k)));
        const float s = hsum(acc);
        sc[j] = s;
        if (s > mx) mx = s;
    }
    const v128_t vmx = wasm_f32x4_splat(mx);
    for (int j = 0; j < n; j += 4) wasm_v128_store(sc + j, exp_neg_f32x4(wasm_f32x4_sub(wasm_v128_load(sc + j), vmx)));
    float den = 0.f;
    for (int j = 0; j < n; j++) den += sc[j];
    const float inv = 1.f / den;
    v128_t ov[MAXHD / 4];
    for (int k = 0; k < hv; k++) ov[k] = wasm_f32x4_const_splat(0.f);
    for (int j = 0; j < n; j++) {
        const v128_t a = wasm_f32x4_splat(sc[j] * inv);
        const float* vv = vc + (size_t)j * KVD;
        for (int k = 0; k < hv; k++) ov[k] = wasm_f32x4_add(ov[k], wasm_f32x4_mul(a, wasm_v128_load(vv + 4 * k)));
    }
    for (int k = 0; k < hv; k++) wasm_v128_store(out + 4 * k, ov[k]);
}

float *bx, *bn, *bq, *bctx, *bo, *bg, *bu, *kcache, *vcache, *scores;

int read_header(const uint8_t* blob) {
    const int32_t* h = (const int32_t*)blob;
    if (h[0] != MAGIC) return -10;
    V = h[1]; D = h[2]; F = h[3]; L = h[4]; H = h[5]; KVH = h[6]; HD = h[7];
    g_tied = h[9] != 0;
    g_theta = ((const float*)h)[10];
    g_eps = ((const float*)h)[11];
    if (L < 1 || L > MAXL || H < 1 || KVH < 1 || H % KVH || HD < 4 || HD > MAXHD || HD % 4 || HD % 2
        || D % B8 || F % B8 || (H * HD) % B8) return -11;
    QD = H * HD;
    KVD = KVH * HD;
    return h[8];
}

template <typename T> T* alloc(size_t n) { return (T*)aligned_alloc(16, ((n * sizeof(T)) + 15) / 16 * 16); }

} // namespace

extern "C" {

// Reads the blob's header and allocates the KV cache (maxctx positions) and
// work buffers. Call once per module before llama_forward. bits: the
// blob's format, 8 or 6. Returns 0, or negative on error.
int llama_init(const uint8_t* blob, int bits, int maxctx) {
    const int fmt = read_header(blob);
    if (fmt < 0) return fmt;
    if (fmt != bits || (bits != 8 && bits != 6)) return -4;
    if (maxctx < 1 || maxctx > 32768) return -1;
    g_fmt = bits;
    MAXCTX = maxctx;
    const int maxcols = D > F ? (D > QD ? D : QD) : (F > QD ? F : QD);
    g_xq = alloc<int16_t>((size_t)MAXSEQ * maxcols);
    g_xs = alloc<float>(MAXSEQ);
    g_xsum = alloc<float>((size_t)MAXSEQ * (maxcols / QB));
    g_xsum8 = alloc<float>((size_t)MAXSEQ * (maxcols / B8));
    bx = alloc<float>((size_t)MAXSEQ * D);
    bn = alloc<float>((size_t)MAXSEQ * D);
    bq = alloc<float>((size_t)MAXSEQ * QD);
    bctx = alloc<float>((size_t)MAXSEQ * QD);
    bo = alloc<float>((size_t)MAXSEQ * D);
    bg = alloc<float>((size_t)MAXSEQ * F);
    bu = alloc<float>((size_t)MAXSEQ * F);
    kcache = alloc<float>((size_t)L * MAXCTX * KVD);
    vcache = alloc<float>((size_t)L * MAXCTX * KVD);
    scores = alloc<float>((size_t)MAXCTX + 4);
    const int half = HD / 2;
    g_cos = alloc<float>((size_t)MAXCTX * half);
    g_sin = alloc<float>((size_t)MAXCTX * half);
    if (!g_xq || !kcache || !vcache || !g_cos || !g_sin || !bg || !bu) return -5;
    // As transformers: inv_freq = 1 / theta^(2i/HD) in f32, angle = pos *
    // inv_freq in f32, then cos/sin.
    for (int i = 0; i < half; i++) {
        const float inv = 1.f / powf(g_theta, (float)(2 * i) / (float)HD);
        for (int p = 0; p < MAXCTX; p++) {
            const float a = (float)p * inv;
            g_cos[(size_t)p * half + i] = cosf(a);
            g_sin[(size_t)p * half + i] = sinf(a);
        }
    }
    return 0;
}

// Expands a u5 blob (format 5) into format 6 at dst, header included; with
// dst null, only returns the size dst needs.
size_t llama_expand_u5(const uint8_t* src, uint8_t* dst) {
    if (read_header(src) != 5) return 0;
    Layout in;
    fill_layout(src, 5, in);
    size_t pos = 0;
    auto bytes = [&](const void* p, size_t n) { if (dst) __builtin_memcpy(dst + pos, p, n); pos += n; };
    auto vec = [&](const float* v, int n) { bytes(v, (size_t)n * 4); };
    auto quant = [&](const QuantMat& m, int rows, int cols) {
        const int nb = cols / QB;
        const size_t hstride = (size_t)((cols + 127) / 128) * 16;
        uint8_t* q = dst ? dst + pos : nullptr;
        pos += (size_t)rows * cols;
        float* sv = dst ? (float*)(dst + pos) : nullptr;
        pos += (size_t)rows * nb * 4;
        float* ov = dst ? (float*)(dst + pos) : nullptr;
        pos += (size_t)rows * nb * 4;
        if (!dst) return;
        const uint16_t* s16 = (const uint16_t*)m.s;
        const uint16_t* o16 = (const uint16_t*)m.o;
        auto f16 = [](uint16_t h) {
            union { uint32_t u; float f; } v;
            v.u = ((uint32_t)(h & 0x8000) << 16) | ((uint32_t)(h & 0x7FFF) << 13);
            return v.f * 5.192296858534828e33f;  // 2^112
        };
        for (int r = 0; r < rows; r++) {
            const uint8_t* nibs = m.q + (size_t)r * cols / 2;
            const uint8_t* h = m.h + (size_t)r * hstride;
            for (int c = 0; c < cols; c++) {
                const int b = c / QB, k = c % QB;
                const int lo = k < 16 ? nibs[b * 16 + k] & 0x0F : nibs[b * 16 + k - 16] >> 4;
                q[(size_t)r * cols + c] = (uint8_t)(lo | (((h[(c / 128) * 16 + c % 16] >> ((c % 128) / 16)) & 1) << 4));
            }
            for (int b = 0; b < nb; b++) {
                sv[(size_t)r * nb + b] = f16(s16[(size_t)r * nb + b]);
                ov[(size_t)r * nb + b] = f16(o16[(size_t)r * nb + b]);
            }
        }
    };
    int32_t header[16];
    __builtin_memcpy(header, src, 64);
    header[8] = 6;
    bytes(header, 64);
    quant(in.embed, V, D);
    for (int i = 0; i < L; i++) {
        const Layer& ly = in.layer[i];
        vec(ly.attn_norm, D);
        quant(ly.wq, QD, D); quant(ly.wk, KVD, D); quant(ly.wv, KVD, D); quant(ly.wo, D, QD);
        vec(ly.ffn_norm, D);
        quant(ly.wg, F, D); quant(ly.wu, F, D); quant(ly.wd, D, F);
    }
    vec(in.norm, D);
    if (!g_tied) quant(in.head, V, D);
    return pos;
}

// The model's shape, for the caller: [V, D, F, L, H, KVH, HD, MAXCTX].
int llama_dim(int i) {
    const int dims[8] = { V, D, F, L, H, KVH, HD, MAXCTX };
    return i >= 0 && i < 8 ? dims[i] : -1;
}

// Runs tokens ids[0..seq) at positions pos0.. (the cache holds positions
// 0..pos0-1 from earlier calls) and writes the last token's V logits
// (allLogits: every token's, seq*V). Returns 0, or negative on error.
int llama_forward(const uint8_t* blob, const int* ids, int seq, int pos0, float* logits, int allLogits) {
    if (!kcache) return -6;
    if (seq < 1 || seq > MAXSEQ || pos0 < 0 || pos0 + seq > MAXCTX) return -1;
    if (read_header(blob) != g_fmt) return -4;
    Layout lay;
    fill_layout(blob, g_fmt, lay);
    for (int t = 0; t < seq; t++) if (ids[t] < 0 || ids[t] >= V) return -2;
    for (int t = 0; t < seq; t++) dequant_row(lay.embed, ids[t], D, bx + (size_t)t * D);
    const float qscale = 1.f / sqrtf((float)HD);
    const int group = H / KVH;
    for (int li = 0; li < L; li++) {
        const Layer& ly = lay.layer[li];
        float* kc = kcache + (size_t)li * MAXCTX * KVD;
        float* vc = vcache + (size_t)li * MAXCTX * KVD;
        for (int t = 0; t < seq; t++) rmsnorm(bx + (size_t)t * D, ly.attn_norm, bn + (size_t)t * D);
        gemm(ly.wq, QD, D, bn, seq, bq);
        gemm(ly.wk, KVD, D, bn, seq, kc + (size_t)pos0 * KVD, false);
        gemm(ly.wv, KVD, D, bn, seq, vc + (size_t)pos0 * KVD, false);
        for (int t = 0; t < seq; t++) {
            const int i = pos0 + t;
            for (int h = 0; h < H; h++) {
                float* q = bq + (size_t)t * QD + h * HD;
                rope(q, i);
                for (int d = 0; d < HD; d++) q[d] *= qscale;
            }
            for (int h = 0; h < KVH; h++) rope(kc + (size_t)i * KVD + h * HD, i);
        }
        for (int t = 0; t < seq; t++) {
            const int i = pos0 + t;
            for (int h = 0; h < H; h++) {
                const int kvh = h / group;
                attend(bq + (size_t)t * QD + h * HD, kc + kvh * HD, vc + kvh * HD, i, scores, bctx + (size_t)t * QD + h * HD);
            }
        }
        gemm(ly.wo, D, QD, bctx, seq, bo);
        for (size_t i = 0; i < (size_t)seq * D; i++) bx[i] += bo[i];
        for (int t = 0; t < seq; t++) rmsnorm(bx + (size_t)t * D, ly.ffn_norm, bn + (size_t)t * D);
        gemm(ly.wg, F, D, bn, seq, bg);
        gemm(ly.wu, F, D, bn, seq, bu, false);
        for (size_t i = 0; i < (size_t)seq * F; i += 4)
            wasm_v128_store(bg + i, swiglu_f32x4(wasm_v128_load(bg + i), wasm_v128_load(bu + i)));
        gemm(ly.wd, D, F, bg, seq, bo);
        for (size_t i = 0; i < (size_t)seq * D; i++) bx[i] += bo[i];
    }
    const int first = allLogits ? 0 : seq - 1;
    for (int t = first; t < seq; t++) rmsnorm(bx + (size_t)t * D, lay.norm, bn + (size_t)(t - first) * D);
    gemm(lay.head, V, D, bn, seq - first, logits);
    return 0;
}

} // extern "C"
