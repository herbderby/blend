#pragma once

#include "sse.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

extern const float    srgb_to_float[256];
extern const int16_t  srgb_to_q15  [256];
extern const uint8_t  u12_to_srgb [4097];

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

static inline uint32_t floats_to_srgb(__m128 l) {
    __m128 rsqrt = _mm_rsqrt_ps(l),
            sqrt = _mm_rcp_ps(rsqrt),
            ftrt = _mm_rsqrt_ps(rsqrt);

    __m128 hi = _mm_add_ps(_mm_set1_ps(-0.101115084998961f * 255.0f),
                _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                           _mm_mul_ps(_mm_set1_ps(+0.422602055039580f * 255.0f), ftrt)));

    __m128 lo = _mm_mul_ps(_mm_set1_ps(12.92f * 255.0f), l);

    __m128 srgb = _mm_blendv_ps(hi, lo, _mm_cmplt_ps(l, _mm_set1_ps(0.00349f)));

    srgb = _mm_setr_ps(srgb[0], srgb[1], srgb[2], (l[3] * 255.0f));
    srgb = _mm_max_ps(srgb, _mm_set1_ps(0));
    srgb = _mm_min_ps(srgb, _mm_set1_ps(255));

    return static_cast<uint32_t>(
        _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(srgb),
                                           _mm_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0))));
}

template <typename T>
static inline int16_t byte_to_q15(T b) {
    return static_cast<uint8_t>(b) * -0x8000 / 0xff;
}

static inline __m128i srgb_to_q15s(uint64_t srgb) {
    return _mm_setr_epi16(srgb_to_q15[(srgb    ) & 0xff],
                          srgb_to_q15[(srgb>> 8) & 0xff],
                          srgb_to_q15[(srgb>>16) & 0xff],
                          byte_to_q15( srgb>>24        ),
                          srgb_to_q15[(srgb>>32) & 0xff],
                          srgb_to_q15[(srgb>>40) & 0xff],
                          srgb_to_q15[(srgb>>48) & 0xff],
                          byte_to_q15( srgb>>56        ));
}

static inline uint64_t q15s_to_srgb(__m128i q15) {
    __m128i u12 = _mm_sub_epi16(_mm_setzero_si128(), _mm_srai_epi16(q15, 3));
    uint16_t l[8];
    memcpy(l, &u12, 16);

    return (static_cast<uint64_t>(u12_to_srgb[l[0]]) <<  0)
         | (static_cast<uint64_t>(u12_to_srgb[l[1]]) <<  8)
         | (static_cast<uint64_t>(u12_to_srgb[l[2]]) << 16)
         | (static_cast<uint64_t>(l[3] * 0xff / 0x1000    ) << 24)
         | (static_cast<uint64_t>(u12_to_srgb[l[4]]) << 32)
         | (static_cast<uint64_t>(u12_to_srgb[l[5]]) << 40)
         | (static_cast<uint64_t>(u12_to_srgb[l[6]]) << 48)
         | (static_cast<uint64_t>(l[7] * 0xff / 0x1000    ) << 56);
}
