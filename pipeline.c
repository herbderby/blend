#include "pipeline.h"
#include "srgb.h"

static inline void next(const struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    stage->next_fn(stage+1, n,dp,d,s);
}

void load_d_srgb(const struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    int* dst = dp;
    d = srgb_to_linear(dst[n]);

    next(stage,n,dp,d,s);
}

void load_s_srgb(const struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    const int* src = stage->ctx;
    s = srgb_to_linear(src[n]);

    next(stage,n,dp,d,s);
}

void srcover(const struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    __m128 a = _mm_shuffle_ps(s,s, 0xff);
    s = _mm_add_ps(s, _mm_mul_ps(d, _mm_sub_ps(_mm_set1_ps(1), a)));

    next(stage,n,dp,d,s);
}

void lerp_u8(const struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    const char* cov = stage->ctx;
    __m128 c = _mm_set1_ps(cov[n] * (1/255.0f)),
           C = _mm_sub_ps(_mm_set1_ps(1), c);
    s = _mm_add_ps(_mm_mul_ps(s, c), _mm_mul_ps(d, C));

    next(stage,n,dp,d,s);
}

void store_s_srgb(const struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    int* dst = dp;
    dst[n] = linear_to_srgb(s);

    next(stage,n,dp,d,s);
}

void done_yet(const struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    if (n-- == 0) {
        return;
    }
    stage->next_fn(stage->ctx, n,dp,d,s);
}
