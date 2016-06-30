#include "speed_of_light.h"
#include "srgb.h"
#include <immintrin.h>

void speed_of_light(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n) {
    __m128 d, s;
    while (n --> 0) {
#if 0
        switch (*src >> 24) {
            case 255: *dst = *src;  // fallthrough
            case   0: dst++; src++; cov++; continue;
        }
#endif

        d = srgb_to_linear(*dst);
        s = srgb_to_linear(*src);

        __m128 a = _mm_shuffle_ps(s,s, 0xff);
        s = _mm_add_ps(s, _mm_mul_ps(d, _mm_sub_ps(_mm_set1_ps(1), a)));

        __m128 c = _mm_set1_ps(*cov * (1/255.0f)),
               C = _mm_sub_ps(_mm_set1_ps(1), c);
        s = _mm_add_ps(_mm_mul_ps(s, c), _mm_mul_ps(d, C));

        *dst = linear_to_srgb(s);

        dst++;
        src++;
        cov++;
    }
}
