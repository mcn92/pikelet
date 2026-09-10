#include "float_hnsw.hpp"
#include "uint8_float_hnsw.hpp"
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <cstring>
#include <memory>

#if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
#endif

using namespace pikelet::wasm;

namespace {

constexpr size_t DEFAULT_MAX_ELEMENTS = 100000;
constexpr size_t DEFAULT_M = 12;
constexpr size_t DEFAULT_EF_CONSTRUCTION = 75;
constexpr size_t DEFAULT_EF_SEARCH = 100;
constexpr uint32_t DEFAULT_SEED = 108;

size_t serialized_index_count_hint(const uint8_t* data, size_t size, uint32_t versioned_magic) {
    if (!data || size < 12) return DEFAULT_MAX_ELEMENTS;

    uint32_t magic = 0;
    std::memcpy(&magic, data, sizeof(magic));

    size_t count_offset = (magic == versioned_magic) ? 12 : 8;
    if (size < count_offset + 4) return DEFAULT_MAX_ELEMENTS;

    uint32_t count = 0;
    std::memcpy(&count, data + count_offset, sizeof(count));
    return static_cast<size_t>(count);
}

} // namespace

// =============================================================================
// Handle-based index system
// =============================================================================

class IndexWrapper {
public:
    virtual ~IndexWrapper() = default;
    virtual uint32_t insert(const float* vec) = 0;
    virtual int bulk_insert(const float* vecs, int n) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k, size_t ef_search) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len, size_t ef_search) = 0;
    virtual void mark_delete(uint32_t id) = 0;
    virtual void compact() = 0;
    virtual void compact(std::vector<uint32_t>& out_map) = 0;
    virtual size_t count() const = 0;
    virtual size_t ghost_count() const = 0;
    virtual float ghost_ratio() const = 0;
    virtual size_t memory_bytes() const = 0;
    virtual std::vector<uint8_t> serialize() const = 0;
    virtual bool deserialize(const uint8_t* data, size_t size) = 0;
    virtual size_t dimension() const = 0;
};


class FloatHNSWWrapper : public IndexWrapper {
    std::unique_ptr<FloatHNSW> impl_;
    FloatHNSWConfig cfg_;
    size_t dims_;
public:
    FloatHNSWWrapper(size_t dims, const FloatHNSWConfig& cfg)
        : impl_(std::make_unique<FloatHNSW>(dims, cfg)), cfg_(cfg), dims_(dims) {}

    uint32_t insert(const float* vec) override { return impl_->insert(vec); }
    int bulk_insert(const float* vecs, int n) override {
        int inserted = 0;
        for (int i = 0; i < n; i++) {
            uint32_t id = impl_->insert(vecs + i * dims_);
            if (id != 0xFFFFFFFF) inserted++;
        }
        return inserted;
    }
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k, size_t ef_search) override {
        return impl_->search(query, k, ef_search);
    }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len, size_t ef_search) override {
        return impl_->search_filtered(query, k, bitset, bitset_len, ef_search);
    }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        size_t cnt = serialized_index_count_hint(data, size, 0x464C4831);
        FloatHNSWConfig cfg = cfg_;
        if (cnt > cfg.max_elements) return false;

        auto next = std::make_unique<FloatHNSW>(dims_, cfg);
        if (!next->deserialize(data, size)) return false;
        impl_ = std::move(next);
        return true;
    }
    size_t dimension() const override { return dims_; }
};

class Uint8FloatHNSWWrapper : public IndexWrapper {
    std::unique_ptr<Uint8FloatHNSW> impl_;
    Uint8FloatHNSWConfig cfg_;
    size_t dims_;
public:
    Uint8FloatHNSWWrapper(size_t dims, const Uint8FloatHNSWConfig& cfg)
        : impl_(std::make_unique<Uint8FloatHNSW>(dims, cfg)), cfg_(cfg), dims_(dims) {}

    uint32_t insert(const float* vec) override { return impl_->insert(vec); }
    int bulk_insert(const float* vecs, int n) override { return impl_->bulk_insert(vecs, n); }
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k, size_t ef_search) override {
        return impl_->search(query, k, ef_search);
    }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len, size_t ef_search) override {
        return impl_->search_filtered(query, k, bitset, bitset_len, ef_search);
    }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        size_t cnt = serialized_index_count_hint(data, size, 0x49384831);
        Uint8FloatHNSWConfig cfg = cfg_;
        if (cnt > cfg.max_elements) return false;

        auto next = std::make_unique<Uint8FloatHNSW>(dims_, cfg);
        if (!next->deserialize(data, size)) return false;
        impl_ = std::move(next);
        return true;
    }
    size_t dimension() const override { return dims_; }
};

// Handle table
constexpr uint32_t MAX_HANDLES = 64;
constexpr uint32_t INVALID_HANDLE = 0xFFFFFFFF;

struct HandleSlot {
    IndexWrapper* index = nullptr;
    size_t dim = 0;
};

static HandleSlot g_handles[MAX_HANDLES];
static std::vector<uint8_t> g_export_bufs[MAX_HANDLES];

static uint32_t alloc_handle() {
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        if (!g_handles[i].index) return i;
    }
    return INVALID_HANDLE;
}

static void free_handle(uint32_t h) {
    if (h < MAX_HANDLES && g_handles[h].index) {
        delete g_handles[h].index;
        g_handles[h].index = nullptr;
        g_handles[h].dim = 0;
        g_export_bufs[h].clear();
    }
}


// =============================================================================
// C ABI
// =============================================================================

extern "C" {

// =============================================================================
// Handle-based pancake_* API
// =============================================================================

uint32_t pancake_init(int dim, int max_elem, int quantized, int metric,
                      int M, int ef_c, int ef_s, int seed) {
    if (dim <= 0 || max_elem <= 0) return INVALID_HANDLE;

    uint32_t h = alloc_handle();
    if (h == INVALID_HANDLE) return INVALID_HANDLE;

    g_handles[h].dim = static_cast<size_t>(dim);
    bool use_cosine = (metric == 1);

    // The constructors allocate the whole index arena eagerly; a request the
    // heap cannot satisfy throws std::bad_alloc. Catch it and hand back the
    // sentinel so one oversized create fails cleanly instead of aborting the
    // WASM instance under every other live handle (mirrors pancake_import).
    try {
        if (quantized) {
            Uint8FloatHNSWConfig u8cfg;
            u8cfg.max_elements = static_cast<size_t>(max_elem);
            u8cfg.M = (M > 0) ? static_cast<size_t>(M) : DEFAULT_M;
            u8cfg.ef_construction = (ef_c > 0) ? static_cast<size_t>(ef_c) : DEFAULT_EF_CONSTRUCTION;
            u8cfg.ef_search = (ef_s > 0) ? static_cast<size_t>(ef_s) : DEFAULT_EF_SEARCH;
            u8cfg.seed = (seed > 0) ? static_cast<uint32_t>(seed) : DEFAULT_SEED;
            u8cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
            u8cfg.use_heuristic = true;
            g_handles[h].index = new Uint8FloatHNSWWrapper(dim, u8cfg);
        } else {
            FloatHNSWConfig cfg;
            cfg.max_elements = static_cast<size_t>(max_elem);
            cfg.M = (M > 0) ? static_cast<size_t>(M) : DEFAULT_M;
            cfg.ef_construction = (ef_c > 0) ? static_cast<size_t>(ef_c) : DEFAULT_EF_CONSTRUCTION;
            cfg.ef_search = (ef_s > 0) ? static_cast<size_t>(ef_s) : DEFAULT_EF_SEARCH;
            cfg.seed = (seed > 0) ? static_cast<uint32_t>(seed) : DEFAULT_SEED;
            cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
            g_handles[h].index = new FloatHNSWWrapper(dim, cfg);
        }
    } catch (...) {
        g_handles[h].index = nullptr;
        g_handles[h].dim = 0;
        return INVALID_HANDLE;
    }

    return h;
}

uint32_t pancake_add(uint32_t h, const float* vec) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0xFFFFFFFF;
    return g_handles[h].index->insert(vec);
}

int pancake_bulk_insert(uint32_t h, const float* vecs, int n) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->bulk_insert(vecs, n);
}

int pancake_query(uint32_t h, const float* qv, int k, int ef_search, uint64_t* ids, float* dists) {
    if (h >= MAX_HANDLES || !g_handles[h].index || !qv || !ids || !dists ||
        k <= 0 || ef_search <= 0 || ef_search > 4096) return 0;
    const size_t bounded_k = std::min(static_cast<size_t>(k), g_handles[h].index->count());
    auto res = g_handles[h].index->search(qv, bounded_k, static_cast<size_t>(ef_search));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = static_cast<uint64_t>(res[j].first);
        dists[j] = res[j].second;
    }
    return static_cast<int>(res.size());
}

int pancake_query_filtered(uint32_t h, const float* qv, int k, int ef_search,
                           uint64_t* ids, float* dists,
                           const uint8_t* bitset, size_t bitset_len) {
    if (h >= MAX_HANDLES || !g_handles[h].index || !qv || !ids || !dists ||
        (!bitset && bitset_len != 0) || k <= 0 || ef_search <= 0 || ef_search > 4096) return 0;
    const size_t bounded_k = std::min(static_cast<size_t>(k), g_handles[h].index->count());
    auto res = g_handles[h].index->search_filtered(qv, bounded_k, bitset, bitset_len,
                                                    static_cast<size_t>(ef_search));
    for (size_t j = 0; j < res.size(); ++j) {
        ids[j] = static_cast<uint64_t>(res[j].first);
        dists[j] = res[j].second;
    }
    return static_cast<int>(res.size());
}

void pancake_delete(uint32_t h, uint32_t id) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return;
    g_handles[h].index->mark_delete(id);
}

void pancake_compact(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return;
    g_handles[h].index->compact();
}

// Compact and write the old→new ID remap into caller-allocated buffer.
// out_buf[old_id] = new_id, or 0xFFFFFFFF for deleted vectors.
// Returns the number of entries written (= pre-compaction count).
size_t pancake_compact_remap(uint32_t h, uint32_t* out_buf, size_t out_capacity) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    if (!out_buf || out_capacity == 0) return 0;
    std::vector<uint32_t> map;
    g_handles[h].index->compact(map);
    size_t n = std::min(map.size(), out_capacity);
    std::memcpy(out_buf, map.data(), n * sizeof(uint32_t));
    return n;
}

size_t pancake_count(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->count();
}

size_t pancake_memory(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->memory_bytes();
}

size_t pancake_ghost_count(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return g_handles[h].index->ghost_count();
}

float pancake_ghost_ratio(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0.0f;
    return g_handles[h].index->ghost_ratio();
}

int pancake_bulk_insert_flat(uint32_t h, const float* vecs, int n) {
    return pancake_bulk_insert(h, vecs, n);
}

uint8_t* pancake_export(uint32_t h, size_t* out_size) {
    if (h >= MAX_HANDLES || !g_handles[h].index) {
        if (out_size) *out_size = 0;
        return nullptr;
    }
    g_export_bufs[h] = g_handles[h].index->serialize();
    *out_size = g_export_bufs[h].size();
    return g_export_bufs[h].data();
}

int pancake_import(uint32_t h, const uint8_t* data, size_t size) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return -1;
    // deserialize() parses an untrusted buffer. Bounds and level/scale caps make
    // a hostile snapshot fail closed, but a remaining oversized resize() could
    // still throw; catch it here so import returns an error instead of aborting
    // the whole WASM instance.
    try {
        return g_handles[h].index->deserialize(data, size) ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

void pancake_dispose(uint32_t h) {
    free_handle(h);
}

int pancake_dimension(uint32_t h) {
    if (h >= MAX_HANDLES || !g_handles[h].index) return 0;
    return static_cast<int>(g_handles[h].index->dimension());
}


// =============================================================================
// Utility exports
// =============================================================================

void* emsc_malloc(size_t size) { return malloc(size); }
void emsc_free(void* ptr) { free(ptr); }

void pancake_profile_print(uint32_t range_start, uint32_t range_end) {
#if defined(PIKELET_UINT8_HNSW_BUILD_PROFILE)
    pikelet::wasm::g_build_profile.print(range_start, range_end);
#else
    (void)range_start;
    (void)range_end;
#endif
}

void pancake_profile_reset() {
#if defined(PIKELET_UINT8_HNSW_BUILD_PROFILE)
    pikelet::wasm::g_build_profile.reset();
#endif
}

// =============================================================================
// Sketch scan
// =============================================================================
//
// Brute-force top-C scan over a resident tier of row-quantized sketches
// against a float32 query. L2 (metric 0) dequantizes each row per lane
// (offset + scale * byte); cosine (metric 1) uses the factored form
// (offset*sum(query) + scale*(raw dot)) instead — see the comment above
// query_sum below. Stateless — operates on caller-provided heap buffers,
// no index handle involved. Used by the range-artifact sketch-rerank
// geometry, where this scan selects the records to fetch remotely.
//
// metric 0 scores rows by squared L2; metric 1 scores by negated dot
// product against a caller-normalized query, so ascending order is best
// cosine first (out_dists then hold -dot; the reference reader's clamped
// 1 - pool*dot transform is monotone in -dot, so the selected top-C is
// identical up to clamp ties).
//
// Returns the number of results written to out_ids/out_dists (ascending by
// score). dims must be a multiple of 16 for the SIMD path; any dims works
// on the scalar tail.

int pancake_sketch_scan(const uint8_t* sketches,
                        const float* scales,
                        const float* offsets,
                        uint32_t count,
                        uint32_t dims,
                        const float* query,
                        uint32_t metric,
                        uint32_t top_c,
                        uint32_t* out_ids,
                        float* out_dists) {
    if (!sketches || !scales || !offsets || !query || !out_ids || !out_dists) return 0;
    if (count == 0 || dims == 0 || top_c == 0 || metric > 1) return 0;

    // Cosine's factored form (y*x = offset*sum(y) + scale*(y*q), see
    // asymmetric_cosine in uint8_float_hnsw.hpp) needs sum(y) once per
    // query, not once per row — every row in this scan shares the same
    // query, so it is computed here rather than inside the row loop.
    // L2 keeps its per-lane dequantize: factoring a squared difference
    // needs three separate sums (query-offset squared, a cross term,
    // byte-squared) rather than fewer operations, so there's no win there.
    float query_sum = 0.0f;
    if (metric == 1) {
        for (uint32_t d = 0; d < dims; d++) query_sum += query[d];
    }

    // Max-heap over the current top-C (root = worst kept distance).
    uint32_t heap_size = 0;
    auto sift_down = [&](uint32_t i) {
        for (;;) {
            uint32_t left = 2 * i + 1;
            uint32_t right = 2 * i + 2;
            uint32_t largest = i;
            if (left < heap_size && out_dists[left] > out_dists[largest]) largest = left;
            if (right < heap_size && out_dists[right] > out_dists[largest]) largest = right;
            if (largest == i) return;
            std::swap(out_dists[i], out_dists[largest]);
            std::swap(out_ids[i], out_ids[largest]);
            i = largest;
        }
    };
    auto sift_up = [&](uint32_t i) {
        while (i > 0) {
            uint32_t parent = (i - 1) / 2;
            if (out_dists[parent] >= out_dists[i]) return;
            std::swap(out_dists[i], out_dists[parent]);
            std::swap(out_ids[i], out_ids[parent]);
            i = parent;
        }
    };

    for (uint32_t row = 0; row < count; row++) {
        const uint8_t* data = sketches + static_cast<size_t>(row) * dims;
        const float s = scales[row];
        const float o = offsets[row];
        float sum = 0.0f;
        uint32_t d = 0;

#ifdef UINT8_HNSW_WASM_SIMD
        v128_t acc0 = wasm_f32x4_splat(0.0f);
        v128_t acc1 = wasm_f32x4_splat(0.0f);
        v128_t acc2 = wasm_f32x4_splat(0.0f);
        v128_t acc3 = wasm_f32x4_splat(0.0f);

        if (metric == 1) {
            // Factored form: accumulate the raw y*byte dot; offset*sum(y)
            // (sum(y) computed once above, outside this row loop) and
            // scale come in once at the end below, instead of
            // dequantizing (offset + scale*byte) per lane — so, unlike
            // the L2 branch below, this path never needs v_scale/v_offset.
            for (; d + 16 <= dims; d += 16) {
                v128_t bytes = wasm_v128_load(data + d);
                v128_t u16_lo = wasm_u16x8_extend_low_u8x16(bytes);
                v128_t u16_hi = wasm_u16x8_extend_high_u8x16(bytes);

                v128_t f0 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_lo));
                acc0 = WFMA(acc0, wasm_v128_load(query + d), f0);

                v128_t f1 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_lo));
                acc1 = WFMA(acc1, wasm_v128_load(query + d + 4), f1);

                v128_t f2 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_hi));
                acc2 = WFMA(acc2, wasm_v128_load(query + d + 8), f2);

                v128_t f3 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_hi));
                acc3 = WFMA(acc3, wasm_v128_load(query + d + 12), f3);
            }
        } else {
            v128_t v_scale = wasm_f32x4_splat(s);
            v128_t v_offset = wasm_f32x4_splat(o);
            for (; d + 16 <= dims; d += 16) {
                v128_t bytes = wasm_v128_load(data + d);
                v128_t u16_lo = wasm_u16x8_extend_low_u8x16(bytes);
                v128_t u16_hi = wasm_u16x8_extend_high_u8x16(bytes);

                v128_t f0 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_lo));
                v128_t val0 = WFMA(v_offset, f0, v_scale);
                v128_t diff0 = wasm_f32x4_sub(wasm_v128_load(query + d), val0);
                acc0 = WFMA(acc0, diff0, diff0);

                v128_t f1 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_lo));
                v128_t val1 = WFMA(v_offset, f1, v_scale);
                v128_t diff1 = wasm_f32x4_sub(wasm_v128_load(query + d + 4), val1);
                acc1 = WFMA(acc1, diff1, diff1);

                v128_t f2 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_low_u16x8(u16_hi));
                v128_t val2 = WFMA(v_offset, f2, v_scale);
                v128_t diff2 = wasm_f32x4_sub(wasm_v128_load(query + d + 8), val2);
                acc2 = WFMA(acc2, diff2, diff2);

                v128_t f3 = wasm_f32x4_convert_i32x4(wasm_u32x4_extend_high_u16x8(u16_hi));
                v128_t val3 = WFMA(v_offset, f3, v_scale);
                v128_t diff3 = wasm_f32x4_sub(wasm_v128_load(query + d + 12), val3);
                acc3 = WFMA(acc3, diff3, diff3);
            }
        }

        v128_t acc = wasm_f32x4_add(wasm_f32x4_add(acc0, acc1),
                                    wasm_f32x4_add(acc2, acc3));
        sum = wasm_f32x4_extract_lane(acc, 0) + wasm_f32x4_extract_lane(acc, 1) +
              wasm_f32x4_extract_lane(acc, 2) + wasm_f32x4_extract_lane(acc, 3);
#endif

        if (metric == 1) {
            for (; d < dims; d++) {
                sum += query[d] * static_cast<float>(data[d]);
            }
            // Contract unchanged: out_dists holds -dot (unclamped), not the
            // reference reader's clamped 1 - dot — see the function-level
            // comment. Ascending order is identical either way; this just
            // avoids computing the clamp for a value nothing reads.
            sum = -(o * query_sum + s * sum);
        } else {
            for (; d < dims; d++) {
                const float diff = query[d] - (o + s * static_cast<float>(data[d]));
                sum += diff * diff;
            }
        }

        if (heap_size < top_c) {
            out_dists[heap_size] = sum;
            out_ids[heap_size] = row;
            heap_size++;
            sift_up(heap_size - 1);
        } else if (sum < out_dists[0]) {
            out_dists[0] = sum;
            out_ids[0] = row;
            sift_down(0);
        }
    }

    // Heap-sort in place so results come back ascending by distance.
    uint32_t n = heap_size;
    while (n > 1) {
        n--;
        std::swap(out_dists[0], out_dists[n]);
        std::swap(out_ids[0], out_ids[n]);
        uint32_t saved = heap_size;
        heap_size = n;
        sift_down(0);
        heap_size = saved;
    }
    return static_cast<int>(heap_size);
}

// =============================================================================
// Global cleanup
// =============================================================================

void pancake_shutdown_all() {
    // Free all handle-based indexes
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        free_handle(i);
    }
}

void shutdown_all() {
    pancake_shutdown_all();
}

} // extern "C"
