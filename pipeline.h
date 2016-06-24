#pragma once

#include <immintrin.h>

struct stage;

typedef void blend(struct stage*, size_t n, void* dp, __m128 d, __m128 s);

struct stage {
    struct stage* next;
    blend* fn;
    void* ctx;
};

blend load_d_srgb,
      load_s_srgb,
      srcover,
      lerp_u8,
      store_s_load_d_srgb;
