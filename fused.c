#include "fused.h"
#include <immintrin.h>

void fused(int* dst, const int* src, const char* cov, int n) {
    __m128 d, s;
    while (n --> 0) {
        d = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*dst)));
        d = _mm_mul_ps(d, _mm_set1_ps(1/255.0f));
        d = _mm_blend_ps(_mm_mul_ps(d,d), d, 0x08);

        s = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*src)));
        s = _mm_mul_ps(s, _mm_set1_ps(1/255.0f));
        s = _mm_blend_ps(_mm_mul_ps(s,s), s, 0x08);

        __m128 a = _mm_shuffle_ps(s,s, 0xff);
        s = _mm_add_ps(s, _mm_mul_ps(d, _mm_sub_ps(_mm_set1_ps(1), a)));

        __m128 c = _mm_set1_ps(*cov * (1/255.0f)),
               C = _mm_sub_ps(_mm_set1_ps(1), c);
        s = _mm_add_ps(_mm_mul_ps(s, c), _mm_mul_ps(d, C));

        s = _mm_mul_ps(_mm_set1_ps(255), _mm_blend_ps(_mm_rcp_ps(_mm_rsqrt_ps(s)), s, 0x08));
        *dst = _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(s),
                                                  _mm_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0)));

        dst++;
        src++;
        cov++;
    }
}
