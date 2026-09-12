/**
 * pancake_napi.cpp — Node.js N-API binding for the Pikelet HNSW engine.
 *
 * Wraps the same C++ engine (engine.cpp) that the WASM build uses, but
 * compiled natively with SSE2 SIMD. Provides an identical API surface so
 * the JS benchmark harness can swap between WASM and native transparently.
 *
 * Exposed JS API:
 *   pancake_init(dim, maxElem, quantized, metric, M, efC, efS, seed?) -> handle
 *   pancake_add(handle, Float32Array) -> internalId
 *   pancake_bulk_insert(handle, Float32Array, n) -> count
 *   pancake_query(handle, Float32Array, k) -> { ids: Uint32Array, distances: Float32Array, count }
 *   pancake_set_ef(handle, ef)
 *   pancake_delete(handle, id)
 *   pancake_compact(handle)
 *   pancake_count(handle) -> number
 *   pancake_ghost_count(handle) -> number
 *   pancake_ghost_ratio(handle) -> number
 *   pancake_memory(handle) -> number
 *   pancake_dimension(handle) -> number
 *   pancake_export(handle) -> Buffer
 *   pancake_import(handle, Buffer) -> 0 on success
 *   pancake_dispose(handle)
 */

#include <napi.h>

// Pull in the engine directly — it's header-only behind the wrappers.
// We define away the Emscripten-specific bits.
#include "float_hnsw.hpp"
#include "uint8_float_hnsw.hpp"

#include <vector>
#include <memory>
#include <cstring>
#include <string>
#include <algorithm>

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

// The count field in a snapshot header is untrusted, and the wrappers below
// size max_elements (and thus the rebuilt index's allocations) from it. A
// snapshot actually holding `count` vectors must contain at least the raw
// vector payload (count * bytes_per_vector), so reject counts the buffer
// cannot possibly hold before they drive an allocation. This is a
// conservative pre-check only — deserialize() remains the authoritative
// validator (magic, dims, levels, edge bounds, ...).
bool snapshot_count_plausible(const uint8_t* data, size_t size, uint32_t versioned_magic, size_t bytes_per_vector) {
    if (!data || size < 12) return true;  // header unreadable; deserialize() rejects it cheaply

    uint32_t magic = 0;
    std::memcpy(&magic, data, sizeof(magic));

    size_t count_offset = (magic == versioned_magic) ? 12 : 8;
    if (size < count_offset + 4) return true;

    uint32_t count = 0;
    std::memcpy(&count, data + count_offset, sizeof(count));
    if (count == 0 || bytes_per_vector == 0) return true;
    return static_cast<size_t>(count) <= size / bytes_per_vector;
}

} // namespace

// ============================================================================
// Handle table (same structure as engine.cpp)
// ============================================================================

class IndexWrapper {
public:
    virtual ~IndexWrapper() = default;
    virtual uint32_t insert(const float* vec) = 0;
    virtual int bulk_insert(const float* vecs, int n) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) = 0;
    virtual void mark_delete(uint32_t id) = 0;
    virtual void compact() = 0;
    virtual void compact(std::vector<uint32_t>& out_map) = 0;
    virtual size_t count() const = 0;
    virtual size_t ghost_count() const = 0;
    virtual float ghost_ratio() const = 0;
    virtual size_t memory_bytes() const = 0;
    virtual std::vector<uint8_t> serialize() const = 0;
    virtual bool snapshot_plausible(const uint8_t* data, size_t size) const = 0;
    virtual bool deserialize(const uint8_t* data, size_t size) = 0;
    virtual void set_ef_search(size_t ef) = 0;
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
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) override { return impl_->search(query, k); }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) override { return impl_->search_filtered(query, k, bitset, bitset_len); }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool snapshot_plausible(const uint8_t* data, size_t size) const override {
        return snapshot_count_plausible(data, size, 0x464C4831, dims_ * sizeof(float));
    }
    bool deserialize(const uint8_t* data, size_t size) override {
        // The header count is untrusted and sizes max_elements below; a
        // crafted count would otherwise drive a multi-GB allocation here.
        if (!snapshot_plausible(data, size)) return false;
        size_t cnt = serialized_index_count_hint(data, size, 0x464C4831);
        FloatHNSWConfig cfg = cfg_;
        cfg.max_elements = std::max(static_cast<size_t>(cnt * 1.2), static_cast<size_t>(100000));
        // Deserialize into a fresh index and swap only on success so a failed
        // import leaves the live index unchanged (same contract as the WASM
        // engine wrappers).
        auto next = std::make_unique<FloatHNSW>(dims_, cfg);
        if (!next->deserialize(data, size)) return false;
        impl_ = std::move(next);
        return true;
    }
    void set_ef_search(size_t ef) override { impl_->set_ef(ef); }
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
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) override { return impl_->search(query, k); }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) override { return impl_->search_filtered(query, k, bitset, bitset_len); }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool snapshot_plausible(const uint8_t* data, size_t size) const override {
        return snapshot_count_plausible(data, size, 0x49384831, dims_ * sizeof(uint8_t));
    }
    bool deserialize(const uint8_t* data, size_t size) override {
        // The header count is untrusted and sizes max_elements below; a
        // crafted count would otherwise drive a multi-GB allocation here.
        if (!snapshot_plausible(data, size)) return false;
        size_t cnt = serialized_index_count_hint(data, size, 0x49384831);
        Uint8FloatHNSWConfig cfg = cfg_;
        cfg.max_elements = std::max(static_cast<size_t>(cnt * 1.2), static_cast<size_t>(100000));
        // Deserialize into a fresh index and swap only on success so a failed
        // import leaves the live index unchanged (same contract as the WASM
        // engine wrappers).
        auto next = std::make_unique<Uint8FloatHNSW>(dims_, cfg);
        if (!next->deserialize(data, size)) return false;
        impl_ = std::move(next);
        return true;
    }
    void set_ef_search(size_t ef) override { impl_->set_ef_search(ef); }
    size_t dimension() const override { return dims_; }
};

constexpr uint32_t MAX_HANDLES = 64;
constexpr uint32_t INVALID_HANDLE = 0xFFFFFFFF;

static IndexWrapper* g_handles[MAX_HANDLES] = {};

static uint32_t alloc_handle() {
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        if (!g_handles[i]) return i;
    }
    return INVALID_HANDLE;
}

static void free_handle(uint32_t h) {
    if (h < MAX_HANDLES && g_handles[h]) {
        delete g_handles[h];
        g_handles[h] = nullptr;
    }
}

// ============================================================================
// N-API wrappers
// ============================================================================

Napi::Value Init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int dim     = info[0].As<Napi::Number>().Int32Value();
    int maxElem = info[1].As<Napi::Number>().Int32Value();
    int quant   = info[2].As<Napi::Number>().Int32Value();
    int metric  = info[3].As<Napi::Number>().Int32Value();
    int M       = info[4].As<Napi::Number>().Int32Value();
    int efC     = info[5].As<Napi::Number>().Int32Value();
    int efS     = info[6].As<Napi::Number>().Int32Value();
    int seed    = (info.Length() > 7 && info[7].IsNumber())
        ? info[7].As<Napi::Number>().Int32Value()
        : static_cast<int>(DEFAULT_SEED);

    if (dim <= 0 || maxElem <= 0)
        return Napi::Number::New(env, INVALID_HANDLE);

    uint32_t h = alloc_handle();
    if (h == INVALID_HANDLE)
        return Napi::Number::New(env, INVALID_HANDLE);

    bool use_cosine = (metric == 1);

    if (quant) {
        Uint8FloatHNSWConfig cfg;
        cfg.max_elements = maxElem;
        cfg.M = (M > 0) ? static_cast<size_t>(M) : DEFAULT_M;
        cfg.ef_construction = (efC > 0) ? static_cast<size_t>(efC) : DEFAULT_EF_CONSTRUCTION;
        cfg.ef_search = (efS > 0) ? static_cast<size_t>(efS) : DEFAULT_EF_SEARCH;
        cfg.seed = (seed > 0) ? static_cast<uint32_t>(seed) : DEFAULT_SEED;
        cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        cfg.use_heuristic = true;
        g_handles[h] = new Uint8FloatHNSWWrapper(dim, cfg);
    } else {
        FloatHNSWConfig cfg;
        cfg.max_elements = maxElem;
        cfg.M = (M > 0) ? static_cast<size_t>(M) : DEFAULT_M;
        cfg.ef_construction = (efC > 0) ? static_cast<size_t>(efC) : DEFAULT_EF_CONSTRUCTION;
        cfg.ef_search = (efS > 0) ? static_cast<size_t>(efS) : DEFAULT_EF_SEARCH;
        cfg.seed = (seed > 0) ? static_cast<uint32_t>(seed) : DEFAULT_SEED;
        cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        g_handles[h] = new FloatHNSWWrapper(dim, cfg);
    }

    return Napi::Number::New(env, h);
}

Napi::Value Add(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(env, INVALID_HANDLE);
    if (!info[1].IsTypedArray()) {
        Napi::TypeError::New(env, "pancake_add: vector must be a Float32Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::TypedArray typed = info[1].As<Napi::TypedArray>();
    if (typed.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "pancake_add: vector must be a Float32Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Float32Array vec = typed.As<Napi::Float32Array>();
    if (vec.ElementLength() < g_handles[h]->dimension()) {
        Napi::RangeError::New(env, "pancake_add: vector is shorter than the index dimension")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t id = g_handles[h]->insert(vec.Data());
    return Napi::Number::New(env, id);
}

Napi::Value BulkInsert(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    int n = info[2].As<Napi::Number>().Int32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(env, 0);
    if (n < 0) {
        Napi::RangeError::New(env, "pancake_bulk_insert: n must be non-negative")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (!info[1].IsTypedArray()) {
        Napi::TypeError::New(env, "pancake_bulk_insert: vectors must be a Float32Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::TypedArray typed = info[1].As<Napi::TypedArray>();
    if (typed.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "pancake_bulk_insert: vectors must be a Float32Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Float32Array vecs = typed.As<Napi::Float32Array>();
    const size_t dims = g_handles[h]->dimension();
    const size_t requested = static_cast<size_t>(n);
    if (dims != 0 && requested > vecs.ElementLength() / dims) {
        Napi::RangeError::New(env, "pancake_bulk_insert: vectors are shorter than n * dimension")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    int count = g_handles[h]->bulk_insert(vecs.Data(), n);
    return Napi::Number::New(env, count);
}

Napi::Value Query(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    int k = info[2].As<Napi::Number>().Int32Value();

    if (h >= MAX_HANDLES || !g_handles[h]) {
        Napi::Object result = Napi::Object::New(env);
        result.Set("count", 0);
        return result;
    }
    if (k < 0) {
        Napi::RangeError::New(env, "pancake_query: k must be non-negative")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (!info[1].IsTypedArray()) {
        Napi::TypeError::New(env, "pancake_query: query must be a Float32Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::TypedArray typed = info[1].As<Napi::TypedArray>();
    if (typed.TypedArrayType() != napi_float32_array) {
        Napi::TypeError::New(env, "pancake_query: query must be a Float32Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    Napi::Float32Array qv = typed.As<Napi::Float32Array>();
    if (qv.ElementLength() < g_handles[h]->dimension()) {
        Napi::RangeError::New(env, "pancake_query: query is shorter than the index dimension")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    const size_t bounded_k = std::min(static_cast<size_t>(k), g_handles[h]->count());
    auto res = g_handles[h]->search(qv.Data(), bounded_k);
    int count = static_cast<int>(res.size());

    Napi::Uint32Array ids = Napi::Uint32Array::New(env, count);
    Napi::Float32Array dists = Napi::Float32Array::New(env, count);
    for (int i = 0; i < count; i++) {
        ids[i] = res[i].first;
        dists[i] = res[i].second;
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("ids", ids);
    result.Set("distances", dists);
    result.Set("count", count);
    return result;
}

Napi::Value SetEf(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    int ef = info[1].As<Napi::Number>().Int32Value();
    if (h < MAX_HANDLES && g_handles[h] && ef > 0)
        g_handles[h]->set_ef_search(ef);
    return info.Env().Undefined();
}

Napi::Value Delete(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    uint32_t id = info[1].As<Napi::Number>().Uint32Value();
    if (h < MAX_HANDLES && g_handles[h])
        g_handles[h]->mark_delete(id);
    return info.Env().Undefined();
}

Napi::Value Compact(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h < MAX_HANDLES && g_handles[h])
        g_handles[h]->compact();
    return info.Env().Undefined();
}

Napi::Value Count(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->count()));
}

Napi::Value GhostCount(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->ghost_count()));
}

Napi::Value GhostRatio(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), g_handles[h]->ghost_ratio());
}

Napi::Value Memory(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->memory_bytes()));
}

Napi::Value Dimension(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->dimension()));
}

Napi::Value Export(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return env.Null();
    auto data = g_handles[h]->serialize();
    auto buf = Napi::Buffer<uint8_t>::Copy(env, data.data(), data.size());
    return buf;
}

Napi::Value Import(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    Napi::Buffer<uint8_t> buf = info[1].As<Napi::Buffer<uint8_t>>();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(env, -1);
    // The snapshot header's count field is untrusted and sizes the rebuilt
    // index; reject counts the buffer cannot possibly hold before allocating.
    if (!g_handles[h]->snapshot_plausible(buf.Data(), buf.Length())) {
        Napi::TypeError::New(env, "pancake_import: snapshot count field exceeds buffer capacity")
            .ThrowAsJavaScriptException();
        return Napi::Number::New(env, -1);
    }
    // deserialize() parses an untrusted buffer. Bounds and level caps make a
    // hostile snapshot fail closed, but a remaining oversized resize() could
    // still throw; convert any C++ exception into a JS error here instead of
    // letting it escape the N-API boundary and abort the process.
    bool ok = false;
    try {
        ok = g_handles[h]->deserialize(buf.Data(), buf.Length());
    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("pancake_import: ") + e.what())
            .ThrowAsJavaScriptException();
        return Napi::Number::New(env, -1);
    } catch (...) {
        Napi::Error::New(env, "pancake_import: snapshot rejected (C++ exception)")
            .ThrowAsJavaScriptException();
        return Napi::Number::New(env, -1);
    }
    return Napi::Number::New(env, ok ? 0 : -1);
}

Napi::Value Dispose(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    free_handle(h);
    return info.Env().Undefined();
}

// ============================================================================
// Module init
// ============================================================================

Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
    exports.Set("pancake_init",         Napi::Function::New(env, Init));
    exports.Set("pancake_add",          Napi::Function::New(env, Add));
    exports.Set("pancake_bulk_insert",  Napi::Function::New(env, BulkInsert));
    exports.Set("pancake_query",        Napi::Function::New(env, Query));
    exports.Set("pancake_set_ef",       Napi::Function::New(env, SetEf));
    exports.Set("pancake_delete",       Napi::Function::New(env, Delete));
    exports.Set("pancake_compact",      Napi::Function::New(env, Compact));
    exports.Set("pancake_count",        Napi::Function::New(env, Count));
    exports.Set("pancake_ghost_count",  Napi::Function::New(env, GhostCount));
    exports.Set("pancake_ghost_ratio",  Napi::Function::New(env, GhostRatio));
    exports.Set("pancake_memory",       Napi::Function::New(env, Memory));
    exports.Set("pancake_dimension",    Napi::Function::New(env, Dimension));
    exports.Set("pancake_export",       Napi::Function::New(env, Export));
    exports.Set("pancake_import",       Napi::Function::New(env, Import));
    exports.Set("pancake_dispose",      Napi::Function::New(env, Dispose));
    return exports;
}

NODE_API_MODULE(pikelet_native, InitModule)
