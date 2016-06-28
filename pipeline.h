#pragma once

#include <immintrin.h>

struct stage;

typedef void stage_fn(const struct stage*, size_t n, void* dp, __m128 d, __m128 s);

struct stage {
    stage_fn* next_fn;
    const void* ctx;
};

stage_fn done_yet,
         load_d_srgb,
         load_s_srgb,
         srcover,
         lerp_u8,
         store_s_srgb,
         loop;
