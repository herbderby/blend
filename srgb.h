#pragma once

#include "sse.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

extern const float srgb_to_float[256];

static inline __m128 srgb_to_floats(uint32_t srgb) {
    __m128 a = _mm_mul_ps(better_cvtsi32_ss(srgb>>24), _mm_set1_ps(1/255.0f));

    return _mm_setr_ps(srgb_to_float[(srgb    ) & 0xff],
                       srgb_to_float[(srgb>> 8) & 0xff],
                       srgb_to_float[(srgb>>16) & 0xff],
                       a[0]);
}

static inline void srgb_to_floats_T(const uint32_t srgb[4],
                                    __m128* r, __m128* g, __m128* b, __m128* a) {
    *r = _mm_setr_ps(srgb_to_float[(srgb[0] >>  0) & 0xff],
                     srgb_to_float[(srgb[1] >>  0) & 0xff],
                     srgb_to_float[(srgb[2] >>  0) & 0xff],
                     srgb_to_float[(srgb[3] >>  0) & 0xff]);

    *g = _mm_setr_ps(srgb_to_float[(srgb[0] >>  8) & 0xff],
                     srgb_to_float[(srgb[1] >>  8) & 0xff],
                     srgb_to_float[(srgb[2] >>  8) & 0xff],
                     srgb_to_float[(srgb[3] >>  8) & 0xff]);

    *b = _mm_setr_ps(srgb_to_float[(srgb[0] >> 16) & 0xff],
                     srgb_to_float[(srgb[1] >> 16) & 0xff],
                     srgb_to_float[(srgb[2] >> 16) & 0xff],
                     srgb_to_float[(srgb[3] >> 16) & 0xff]);

    auto p = reinterpret_cast<const __m128i*>(srgb);
    *a = _mm_mul_ps(_mm_set1_ps(1/255.0f),
                    _mm_cvtepi32_ps(_mm_srli_epi32(_mm_loadu_si128(p), 24)));
}

static inline __m128 clamp_0_255(__m128 srgb) {
    // max/min order and argument order both matter.  This clamps NaN to 0.
    srgb = _mm_max_ps(srgb, _mm_set1_ps(0));
    srgb = _mm_min_ps(srgb, _mm_set1_ps(255));
    return srgb;
}

static inline __m128 floats_to_srgb_floats(__m128 l) {
    __m128 rsqrt = _mm_rsqrt_ps(l),
            sqrt = _mm_rcp_ps(rsqrt),
            ftrt = _mm_rsqrt_ps(rsqrt);

    __m128 lo = _mm_mul_ps(_mm_set1_ps(12.92f * 255.0f), l);

    __m128 hi = _mm_add_ps(_mm_set1_ps(-0.101115084998961f * 255.0f),
                _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                           _mm_mul_ps(_mm_set1_ps(+0.422602055039580f * 255.0f), ftrt)));

    return _mm_blendv_ps(hi, lo, _mm_cmplt_ps(l, _mm_set1_ps(0.00349f)));
}

static inline uint32_t floats_to_srgb(__m128 l) {
    auto srgb = floats_to_srgb_floats(l);
    srgb = _mm_setr_ps(srgb[0], srgb[1], srgb[2], (l[3] * 255.0f));
    srgb = clamp_0_255(srgb);
    return static_cast<uint32_t>(
        _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(srgb),
                                           _mm_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0))));
}

static inline void floats_to_srgb_T(uint32_t srgb[4], __m128 r, __m128 g, __m128 b, __m128 a) {
    r = clamp_0_255(floats_to_srgb_floats(r));
    g = clamp_0_255(floats_to_srgb_floats(g));
    b = clamp_0_255(floats_to_srgb_floats(b));
    a = clamp_0_255(_mm_mul_ps(a, _mm_set1_ps(255.0f)));

    __m128i rgba = _mm_or_si128(               _mm_cvtps_epi32(r)     ,
                   _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(g),  8),
                   _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(b), 16),
                                _mm_slli_epi32(_mm_cvtps_epi32(a), 24))));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(srgb), rgba);
}
