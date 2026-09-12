{
  "targets": [
    {
      "target_name": "pikelet_native",
      "sources": ["pikelet_napi.cpp"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../src"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": [
        "-O3",
        "-std=c++17",
        "-ffast-math",
        "-ftree-vectorize",
        "-fno-rtti",
        "-march=native",
        "-mavx2",
        "-msse2",
        "-DPIKELET_ENABLE_AVX512_SIMD",
        "-DPIKELET_ENABLE_AVX2_SIMD",
        "-DPIKELET_ENABLE_SSE2_SIMD"
      ],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "OTHER_CPLUSPLUSFLAGS": [
              "-O3", "-ffast-math", "-fvectorize", "-fslp-vectorize",
              "-fno-rtti", "-mavx2",
              "-DPIKELET_ENABLE_AVX512_SIMD",
              "-DPIKELET_ENABLE_AVX2_SIMD", "-DPIKELET_ENABLE_SSE2_SIMD"
            ]
          }
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": ["/O2", "/std:c++17", "/arch:AVX2", "/DPIKELET_ENABLE_AVX2_SIMD", "/DPIKELET_ENABLE_SSE2_SIMD"]
            }
          }
        }]
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"]
    }
  ]
}
