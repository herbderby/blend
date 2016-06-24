#pragma once

#include <immintrin.h>

struct stage;

typedef void blend(struct stage*,
                   void* dp, const void* sp, const void* cp,
                   __m128 d, __m128 s, int n);

struct stage {
    struct stage* next;
    blend* fn;
    void* ctx;
};

blend load_srgb_dst,
      load_srgb_src,
      srcover,
      lerp_a8_cov,
      store_srgb;
