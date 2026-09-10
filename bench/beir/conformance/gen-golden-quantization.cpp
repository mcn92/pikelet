// Standalone conformance harness: emits known quantization rows using the
// EXACT same code path as src/uint8_float_hnsw.hpp's insert() (copied
// verbatim below, since the real class is templated on the whole HNSW graph
// and not easily invoked standalone without the full engine). Compiled with
// plain g++, no Emscripten needed — this only tests the quantization math.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

// Verbatim from src/uint8_float_hnsw.hpp:198-234 (cosine path)
void quantize_row(const float* src, size_t dims, std::vector<uint8_t>& q, float& scale, float& offset) {
    float vmin = src[0], vmax = src[0];
    for (size_t d = 1; d < dims; d++) {
        if (src[d] < vmin) vmin = src[d];
        if (src[d] > vmax) vmax = src[d];
    }
    float range = vmax - vmin;
    if (range < 1e-30f) range = 1.0f;
    scale = range / 255.0f;
    offset = vmin;
    float inv_scale = 255.0f / range;
    q.resize(dims);
    for (size_t d = 0; d < dims; d++) {
        float v = (src[d] - vmin) * inv_scale + 0.5f;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        q[d] = static_cast<uint8_t>(v);
    }
}

bool normalize_cosine(const float* in, float* out, size_t dims) {
    float normSq = 0;
    for (size_t d = 0; d < dims; d++) normSq += in[d] * in[d];
    if (!(normSq > 0)) return false;
    float inv = 1.0f / std::sqrt(normSq);
    for (size_t d = 0; d < dims; d++) out[d] = in[d] * inv;
    return true;
}

int main() {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const size_t dim = 8;
    const int numRows = 5;

    printf("[\n");
    for (int r = 0; r < numRows; r++) {
        std::vector<float> raw(dim), normed(dim);
        for (size_t d = 0; d < dim; d++) raw[d] = dist(rng);
        normalize_cosine(raw.data(), normed.data(), dim);

        std::vector<uint8_t> q;
        float scale, offset;
        quantize_row(normed.data(), dim, q, scale, offset);

        std::vector<float> query(dim);
        for (size_t d = 0; d < dim; d++) query[d] = dist(rng);
        std::vector<float> queryNormed(dim);
        normalize_cosine(query.data(), queryNormed.data(), dim);

        double exactDot = 0, sumQuery = 0;
        for (size_t d = 0; d < dim; d++) { exactDot += queryNormed[d] * normed[d]; sumQuery += queryNormed[d]; }

        printf("  {\n");
        printf("    \"inputRow\": [");
        for (size_t d = 0; d < dim; d++) printf("%.9g%s", raw[d], d + 1 < dim ? ", " : "");
        printf("],\n");
        printf("    \"normalizedRow\": [");
        for (size_t d = 0; d < dim; d++) printf("%.9g%s", normed[d], d + 1 < dim ? ", " : "");
        printf("],\n");
        printf("    \"scale\": %.9g,\n", scale);
        printf("    \"offset\": %.9g,\n", offset);
        printf("    \"q\": [");
        for (size_t d = 0; d < dim; d++) printf("%d%s", q[d], d + 1 < dim ? ", " : "");
        printf("],\n");
        printf("    \"query\": [");
        for (size_t d = 0; d < dim; d++) printf("%.9g%s", queryNormed[d], d + 1 < dim ? ", " : "");
        printf("],\n");
        printf("    \"expectedExactDot\": %.9g,\n", exactDot);
        printf("    \"sumQuery\": %.9g\n", sumQuery);
        printf("  }%s\n", r + 1 < numRows ? "," : "");
    }
    printf("]\n");
    return 0;
}
