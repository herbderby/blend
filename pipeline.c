#include "pipeline.h"

void load_srgb_dst(struct stage* stage, size_t n, void* dp, const void* sp, __m128 d, __m128 s) {
    int* dst = dp;
    d = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(dst[n])));
    d = _mm_mul_ps(d, _mm_set1_ps(1/255.0f));
    d = _mm_blend_ps(_mm_mul_ps(d,d), d, 0x08);

    stage->next->fn(stage->next, n,dp,sp,d,s);
}

void load_srgb_src(struct stage* stage, size_t n, void* dp, const void* sp, __m128 d, __m128 s) {
    const int* src = sp;
    s = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(src[n])));
    s = _mm_mul_ps(s, _mm_set1_ps(1/255.0f));
    s = _mm_blend_ps(_mm_mul_ps(s,s), s, 0x08);

    stage->next->fn(stage->next, n,dp,sp,d,s);
}

void srcover(struct stage* stage, size_t n, void* dp, const void* sp, __m128 d, __m128 s) {
    __m128 a = _mm_shuffle_ps(s,s, 0xff);
    s = _mm_add_ps(s, _mm_mul_ps(d, _mm_sub_ps(_mm_set1_ps(1), a)));

    stage->next->fn(stage->next, n,dp,sp,d,s);
}

void lerp_a8_cov(struct stage* stage, size_t n, void* dp, const void* sp, __m128 d, __m128 s) {
    const char* cov = stage->ctx;
    __m128 c = _mm_set1_ps(cov[n] * (1/255.0f)),
           C = _mm_sub_ps(_mm_set1_ps(1), c);
    s = _mm_add_ps(_mm_mul_ps(s, c), _mm_mul_ps(d, C));

    stage->next->fn(stage->next, n,dp,sp,d,s);
}

void store_srgb(struct stage* stage, size_t n, void* dp, const void* sp, __m128 d, __m128 s) {
    int* dst = dp;
    s = _mm_mul_ps(_mm_set1_ps(255), _mm_blend_ps(_mm_rcp_ps(_mm_rsqrt_ps(s)), s, 0x08));
    dst[n] = _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(s),
                                                _mm_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0)));

    if (n-- == 0) {
        return;
    }

    d = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(dst[n])));
    d = _mm_mul_ps(d, _mm_set1_ps(1/255.0f));
    d = _mm_blend_ps(_mm_mul_ps(d,d), d, 0x08);

    stage->next->fn(stage->next, n,dp,sp,d,s);
}
