#pragma once

#include <immintrin.h>

struct stage;

typedef void blend(struct stage*, size_t n, void* dp, __m128 d, __m128 s);

struct stage {
    struct stage* next;
    blend* fn;
    const void* const_ctx;
    void* ctx;
};

blend splice_next,
      done_yet,
      load_d_srgb,
      load_s_srgb,
      srcover,
      lerp_u8,
      store_s_srgb,
      store_s_done_yet_load_d_srgb;
