#pragma once

#include <immintrin.h>

struct stage;

#if 0
    #define ABI __attribute__((vectorcall))
#elif 0
    #define ABI __attribute__((sysv_abi))
#else
    #define ABI
#endif

typedef ABI void stage_fn(const struct stage*, size_t x, void* dp, __m128 d, __m128 s);

struct stage {
    stage_fn* next;
    const void* ctx;
};

stage_fn shortcircuit_srcover_both_rgba8888,

         load_d_srgb,
         load_s_srgb,
         srcover,
         lerp_u8,
         store_s_srgb;

void run_pipeline(const struct stage*, stage_fn* start, void* dp, size_t n);
