#include "pipeline.h"

void check_n(struct stage* stage, void* vdst, const void* vsrc, const void* vcov, __m128 d, __m128 s, int n) {
    if (n--) {
        stage->next->fn(stage->next, vdst,vsrc,vcov,d,s,n);
    }
}

void load_srgb_dst(struct stage* stage, void* vdst, const void* vsrc, const void* vcov, __m128 d, __m128 s, int n) {
    int* dst = vdst;
    d = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*dst)));
    d = _mm_mul_ps(d, _mm_set1_ps(1/255.0f));
    d = _mm_blend_ps(_mm_mul_ps(d,d), d, 0x08);

    stage->next->fn(stage->next, vdst,vsrc,vcov,d,s,n);
}

void load_srgb_src(struct stage* stage, void* vdst, const void* vsrc, const void* vcov, __m128 d, __m128 s, int n) {
    const int* src = vsrc;
    vsrc = src+1;

    s = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*src)));
    s = _mm_mul_ps(s, _mm_set1_ps(1/255.0f));
    s = _mm_blend_ps(_mm_mul_ps(s,s), s, 0x08);

    stage->next->fn(stage->next, vdst,vsrc,vcov,d,s,n);
}

void srcover(struct stage* stage, void* vdst, const void* vsrc, const void* vcov, __m128 d, __m128 s, int n) {
    __m128 a = _mm_shuffle_ps(s,s, 0xff);
    s = _mm_add_ps(s, _mm_mul_ps(d, _mm_sub_ps(_mm_set1_ps(1), a)));

    stage->next->fn(stage->next, vdst,vsrc,vcov,d,s,n);
}

void lerp_a8_cov(struct stage* stage, void* vdst, const void* vsrc, const void* vcov, __m128 d, __m128 s, int n) {
    const char* cov = vcov;
    vcov = cov+1;

    __m128 c = _mm_set1_ps(*cov * (1/255.0f)),
           C = _mm_sub_ps(_mm_set1_ps(1), c);
    s = _mm_add_ps(_mm_mul_ps(s, c), _mm_mul_ps(d, C));

    stage->next->fn(stage->next, vdst,vsrc,vcov,d,s,n);
}

void store_srgb(struct stage* stage, void* vdst, const void* vsrc, const void* vcov, __m128 d, __m128 s, int n) {
    int* dst = vdst;
    vdst = dst+1;

    s = _mm_mul_ps(_mm_set1_ps(255), _mm_blend_ps(_mm_rcp_ps(_mm_rsqrt_ps(s)), s, 0x08));
    *dst = _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(s),
                                              _mm_setr_epi8(0,4,8,12,0,0,0,0,0,0,0,0,0,0,0,0)));

    stage->next->fn(stage->next, vdst,vsrc,vcov,d,s,n);
}
