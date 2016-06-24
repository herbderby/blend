#pragma once

#include <immintrin.h>

struct stage;

typedef void blend(struct stage*, size_t n, void* dp, __m128 d, __m128 s);

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
