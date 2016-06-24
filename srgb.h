#pragma once

#include <immintrin.h>

extern const float srgb_to_linear_table[256];

static inline __m128 srgb_to_linear(int srgb) {
    unsigned u = (unsigned)srgb;
    return _mm_setr_ps(srgb_to_linear_table[(u    ) & 0xff],
                       srgb_to_linear_table[(u>> 8) & 0xff],
                       srgb_to_linear_table[(u>>16) & 0xff],
                       (u>>24) * (1/255.0f));
}

static inline int linear_to_srgb(__m128 linear) {
    __m128 rsqrt = _mm_rsqrt_ps(linear),
            sqrt = _mm_rcp_ps(rsqrt),
            ftrt = _mm_rsqrt_ps(rsqrt);

    __m128 hi = _mm_add_ps(_mm_set1_ps(-0.101115084998961f * 255.0f),
                _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                           _mm_mul_ps(_mm_set1_ps(+0.422602055039580f * 255.0f), ftrt)));

    __m128 lo = _mm_mul_ps(_mm_set1_ps(12.92f * 255.0f), linear);

    __m128 srgb = _mm_blendv_ps(hi, lo, _mm_cmplt_ps(linear, _mm_set1_ps(0.00349f)));

    srgb = _mm_setr_ps(srgb[0], srgb[1], srgb[2], (linear[3] * 255.0f));

    return _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(srgb),
                                              _mm_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0)));
}
