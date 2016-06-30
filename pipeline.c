#include "pipeline.h"
#include "srgb.h"
#include "sse.h"
#include "unused.h"
#include <stdbool.h>

static bool shortcircuit_srcover_both_rgba8888_(const void* ctx, size_t x, void* dp,
                                                __m128* UNUSED, __m128* UNUSED) {
    const uint32_t* src = ctx;
    uint32_t* dst = dp;
    switch (src[x] >> 24) {
        case 255: dst[x] = src[x]; return true;
        case   0:                  return true;
    }
    return false;
}

static bool load_d_srgb_(const void* UNUSED, size_t x, void* dp, __m128* d, __m128* UNUSED) {
    uint32_t* dst = dp;
    *d = srgb_to_linear(dst[x]);
    return false;
}

static bool load_s_srgb_(const void* ctx, size_t x, void* UNUSED, __m128* UNUSED, __m128* s) {
    const uint32_t* src = ctx;
    *s = srgb_to_linear(src[x]);
    return false;
}

static bool srcover_(const void* UNUSED, size_t UNUSED, void* UNUSED, __m128* d, __m128* s) {
    __m128 a = _mm_shuffle_ps(*s,*s, 0xff);
    *s = _mm_add_ps(*s, _mm_mul_ps(*d, _mm_sub_ps(_mm_set1_ps(1), a)));
    return false;
}

static bool lerp_u8_(const void* ctx, size_t x, void* UNUSED, __m128* d, __m128* s) {
    const uint8_t* cov = ctx;
    __m128 c = _mm_mul_ps(better_cvtsi32_ss(cov[x]), _mm_set1_ps(1/255.0f)),
           C = _mm_sub_ps(_mm_set1_ps(1), c);
    *s = _mm_add_ps(_mm_mul_ps(*s, c), _mm_mul_ps(*d, C));
    return false;
}

static bool store_s_srgb_(const void* UNUSED, size_t x, void* dp, __m128* UNUSED, __m128* s) {
    uint32_t* dst = dp;
    dst[x] = linear_to_srgb(*s);
    return true;
}


#define STAGE(name)                                                                    \
    ABI void name(const struct stage* stage, size_t x, void* dp, __m128 d, __m128 s) { \
        if (!name##_(stage->ctx, x,dp,&d,&s)) {                                        \
            stage->next(stage+1, x,dp,d,s);                                            \
        }                                                                              \
    }

    STAGE(shortcircuit_srcover_both_rgba8888)
    STAGE(load_d_srgb)
    STAGE(load_s_srgb)
    STAGE(srcover)
    STAGE(lerp_u8)
    STAGE(store_s_srgb)

#undef STAGE

void run_pipeline(const struct stage* stages, stage_fn* start, void* dp, size_t n) {
    __m128 d = _mm_undefined_ps(),
           s = _mm_undefined_ps();
    for (size_t x = 0; x < n; x++) {
        start(stages, x, dp, d,s);
    }
}

void fused(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n) {
    __m128 d = _mm_undefined_ps(),
           s = _mm_undefined_ps();
    for (size_t x = 0; x < n; x++) {
        //if (shortcircuit_srcover_both_rgba8888_( src, x,dst,&d,&s)) continue;
        if (                       load_d_srgb_(NULL, x,dst,&d,&s)) continue;
        if (                       load_s_srgb_( src, x,dst,&d,&s)) continue;
        if (                           srcover_(NULL, x,dst,&d,&s)) continue;
        if (                           lerp_u8_( cov, x,dst,&d,&s)) continue;
        if (                      store_s_srgb_(NULL, x,dst,&d,&s)) continue;
    }
}
