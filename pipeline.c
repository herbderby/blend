#include "pipeline.h"
#include "srgb.h"

static void next(const struct stage* stage, size_t x, void* dp, __m128 d, __m128 s) {
    stage->next(stage+1, x,dp,d,s);
}

ABI void shortcircuit_srcover_both_rgba8888(const struct stage* stage, size_t x, void* dp,
                                            __m128 d, __m128 s) {
    const uint32_t* src = stage->ctx;
    uint32_t* dst = dp;
    switch (src[x] >> 24) {
        case 255: dst[x] = src[x];
        case   0: return;
    }

    next(stage,x,dp,d,s);
}

ABI void load_d_srgb(const struct stage* stage, size_t x, void* dp, __m128 d, __m128 s) {
    uint32_t* dst = dp;
    d = srgb_to_linear(dst[x]);

    next(stage,x,dp,d,s);
}

ABI void load_s_srgb(const struct stage* stage, size_t x, void* dp, __m128 d, __m128 s) {
    const uint32_t* src = stage->ctx;
    s = srgb_to_linear(src[x]);

    next(stage,x,dp,d,s);
}

ABI void srcover(const struct stage* stage, size_t x, void* dp, __m128 d, __m128 s) {
    __m128 a = _mm_shuffle_ps(s,s, 0xff);
    s = _mm_add_ps(s, _mm_mul_ps(d, _mm_sub_ps(_mm_set1_ps(1), a)));

    next(stage,x,dp,d,s);
}

ABI void lerp_u8(const struct stage* stage, size_t x, void* dp, __m128 d, __m128 s) {
    const uint8_t* cov = stage->ctx;
    __m128 c = _mm_set1_ps(cov[x] * (1/255.0f)),
           C = _mm_sub_ps(_mm_set1_ps(1), c);
    s = _mm_add_ps(_mm_mul_ps(s, c), _mm_mul_ps(d, C));

    next(stage,x,dp,d,s);
}

ABI void store_s_srgb(const struct stage* stage, size_t x, void* dp, __m128 d, __m128 s) {
    uint32_t* dst = dp;
    dst[x] = linear_to_srgb(s);

    (void)stage;
    (void)x;
    (void)d;
}

void run_pipeline(const struct stage* stages, stage_fn* start, void* dp, size_t n) {
    __m128 d = _mm_undefined_ps(),
           s = _mm_undefined_ps();
    d = _mm_setzero_ps();  // comment out this line to run at half speed... wtf?
    for (size_t x = 0; x < n; x++) {
        //_mm256_zeroall();  // when using vectorcall, keep this line commented to run at half speed
        start(stages, x, dp, d,s);
    }
}
