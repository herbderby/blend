#pragma once

#include <immintrin.h>

extern const float srgb_to_linear_table[256];

static inline __m128 srgb_to_linear(int srgb) {
    const __m128 rgb_mask = _mm_setr_ps(-0.0f, -0.0f, -0.0f, 0.0f);
    __m128 l = _mm_undefined_ps();
    l = _mm_mask_i32gather_ps(l, srgb_to_linear_table,
                              _mm_cvtepu8_epi32(_mm_cvtsi32_si128(srgb)), rgb_mask, 4);
    l = _mm_setr_ps(l[0], l[1], l[2], ((unsigned)srgb >> 24) * (1/255.0f));
    return l;
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
