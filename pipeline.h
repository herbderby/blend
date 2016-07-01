#pragma once

#include <immintrin.h>
#include <stdint.h>
#include <vector>

struct stage;

#if 0
    #define ABI __attribute__((vectorcall))
#elif 0
    #define ABI __attribute__((sysv_abi))
#else
    #define ABI
#endif

typedef ABI void stage_fn(const stage*, size_t x, void* dp, __m128 d, __m128 s);

struct stage {
    stage_fn* next;
    const void* ctx;
};

stage_fn shortcircuit_srcover_both_srgb,
         load_d_srgb,
         load_s_srgb,
         srcover,
         lerp_u8,
         store_s_srgb;

void fused(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n);

struct pipeline {
    void add_stage(stage_fn*, const void* ctx);
    void ready();

    void call(void* dp, size_t n) const;

    std::vector<stage> stages;
};
