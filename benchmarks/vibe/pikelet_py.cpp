// pikelet_py.cpp — pybind11 wrapper around the header-only Pikelet HNSW engine.
//
// This exposes the SAME C++ engine used by the WASM and N-API builds
// (src/uint8_float_hnsw.hpp / src/float_hnsw.hpp) to Python, so neutral runners
// such as ANN-Benchmarks and VIBE can benchmark the native engine without
// passing through Node or WebAssembly.
//
// Build (out of the pikelet repo root, with the headers on the include path):
//   c++ -O3 -std=c++17 -shared -fPIC -DPIKELET_ENABLE_AVX2_SIMD -mavx2 \
//       $(python3 -m pybind11 --includes) \
//       -Isrc pikelet_py.cpp -o pikelet_py$(python3-config --extension-suffix)
//
// Notes:
//  * We accept float32 vectors and let the engine do its OWN asymmetric
//    uint8 quantization internally (quantized=true). This is the whole point
//    of pikelet — see the uint8 header's "WHY ASYMMETRIC" comment — so we do
//    NOT consume VIBE's pre-quantized uint8 datasets. The Python wrapper only
//    handles the float32 path; the VIBE config declares pikelet under `float:`.
//  * Single-threaded build/query to match VIBE's measurement convention
//    (OMP_NUM_THREADS=1 etc. are set in the base image).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "uint8_float_hnsw.hpp"
#include "float_hnsw.hpp"

namespace py = pybind11;
using pikelet::wasm::DistanceMetric;
using pikelet::wasm::Uint8FloatHNSW;
using pikelet::wasm::Uint8FloatHNSWConfig;
using pikelet::wasm::FloatHNSW;
using pikelet::wasm::FloatHNSWConfig;

class PikeletIndex {
public:
    PikeletIndex(size_t dim, size_t max_elements, bool quantized,
                 const std::string& metric, size_t M, size_t ef_construction)
        : dim_(dim), quantized_(quantized)
    {
        DistanceMetric m = (metric == "cosine") ? DistanceMetric::Cosine
                                                 : DistanceMetric::L2;
        if (quantized_) {
            Uint8FloatHNSWConfig cfg;
            cfg.max_elements = max_elements;
            cfg.M = M;
            cfg.ef_construction = ef_construction;
            cfg.ef_search = 100;
            cfg.metric = m;
            cfg.use_heuristic = true;
            i8_ = new Uint8FloatHNSW(dim, cfg);
        } else {
            FloatHNSWConfig cfg;
            cfg.max_elements = max_elements;
            cfg.M = M;
            cfg.ef_construction = ef_construction;
            cfg.ef_search = 100;
            cfg.metric = m;
            cfg.use_heuristic = true;
            f32_ = new FloatHNSW(dim, cfg);
        }
    }

    ~PikeletIndex() { delete i8_; delete f32_; }

    // X: (n, dim) float32, C-contiguous.
    void fit(py::array_t<float, py::array::c_style | py::array::forcecast> X) {
        auto buf = X.request();
        size_t n = static_cast<size_t>(buf.shape[0]);
        const float* data = static_cast<const float*>(buf.ptr);
        for (size_t i = 0; i < n; ++i) {
            const float* vec = data + i * dim_;
            uint32_t id = quantized_ ? i8_->insert(vec) : f32_->insert(vec);
            if (id == UINT32_MAX) {
                throw std::runtime_error(
                    "Pikelet rejected benchmark vector " + std::to_string(i)
                );
            }
            if (id != i) {
                throw std::runtime_error(
                    "Pikelet internal ID diverged from benchmark row ID at " +
                    std::to_string(i)
                );
            }
        }
    }

    void set_ef(size_t ef) {
        if (quantized_) i8_->set_ef_search(ef);
        else            f32_->set_ef(ef);
    }

    // Returns the neighbor IDs only — VIBE recomputes distances itself.
    py::array_t<int64_t> query(py::array_t<float, py::array::c_style | py::array::forcecast> v,
                               int k) {
        auto buf = v.request();
        const float* q = static_cast<const float*>(buf.ptr);
        std::vector<std::pair<uint32_t, float>> res =
            quantized_ ? i8_->search(q, static_cast<size_t>(k))
                       : f32_->search(q, static_cast<size_t>(k));
        py::array_t<int64_t> out(static_cast<py::ssize_t>(res.size()));
        auto ob = out.request();
        int64_t* op = static_cast<int64_t*>(ob.ptr);
        for (size_t i = 0; i < res.size(); ++i) op[i] = static_cast<int64_t>(res[i].first);
        return out;
    }

    size_t memory_bytes() const {
        return quantized_ ? i8_->memory_bytes() : f32_->memory_bytes();
    }

private:
    size_t dim_;
    bool quantized_;
    Uint8FloatHNSW* i8_ = nullptr;
    FloatHNSW*     f32_ = nullptr;
};

PYBIND11_MODULE(pikelet_py, m) {
    m.doc() = "Native Python binding for the Pikelet HNSW engine (VIBE).";
    py::class_<PikeletIndex>(m, "PikeletIndex")
        .def(py::init<size_t, size_t, bool, const std::string&, size_t, size_t>(),
             py::arg("dim"), py::arg("max_elements"), py::arg("quantized"),
             py::arg("metric"), py::arg("M"), py::arg("ef_construction"))
        .def("fit", &PikeletIndex::fit)
        .def("set_ef", &PikeletIndex::set_ef)
        .def("query", &PikeletIndex::query)
        .def("memory_bytes", &PikeletIndex::memory_bytes);
}
