#include "pipeline.h"
#include "srgb.h"

#define NEXT stage->next->fn(stage->next, n,dp,d,s)

void just_next(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    NEXT;
}

void done_yet(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    if (n-- == 0) {
        return;
    }

    NEXT;
}

void load_d_srgb(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    int* dst = dp;
    d = srgb_to_linear(dst[n]);

    NEXT;
}

void load_s_srgb(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    const int* src = stage->const_ctx;
    s = srgb_to_linear(src[n]);

    NEXT;
}

void srcover(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    __m128 a = _mm_shuffle_ps(s,s, 0xff);
    s = _mm_add_ps(s, _mm_mul_ps(d, _mm_sub_ps(_mm_set1_ps(1), a)));

    NEXT;
}

void lerp_u8(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    const char* cov = stage->const_ctx;
    __m128 c = _mm_set1_ps(cov[n] * (1/255.0f)),
           C = _mm_sub_ps(_mm_set1_ps(1), c);
    s = _mm_add_ps(_mm_mul_ps(s, c), _mm_mul_ps(d, C));

    NEXT;
}

void store_s_srgb(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    int* dst = dp;
    dst[n] = linear_to_srgb(s);

    NEXT;
}

void store_s_done_yet_load_d_srgb(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    int* dst = dp;
    dst[n] = linear_to_srgb(s);

    if (n-- == 0) {
        return;
    }

    d = srgb_to_linear(dst[n]);

    NEXT;
}
