#pragma once

#include <immintrin.h>

struct stage;

typedef void blend(struct stage*,
                   void* dst, const void* vsrc, const void* vcov,
                   __m128 d, __m128 s, int n);

struct stage {
    struct stage* next;
    blend* fn;
    void* ctx;
};

blend load_srgb_dst,
      load_srgb_src,  // ctx == src pointer
      srcover,
      lerp_a8_cov,    // ctx == cov pointer
      store_srgb;
