#include "abi.h"
#include "pipeline.h"
#include "srgb.h"
#include <assert.h>
#include <immintrin.h>

using f8 = __m256;

static inline void srgb_to_floats(const uint32_t srgb[8], f8* r, f8* g, f8* b, f8* a) {
    *r = { srgb_to_float[(srgb[0] >>  0) & 0xff],
           srgb_to_float[(srgb[1] >>  0) & 0xff],
           srgb_to_float[(srgb[2] >>  0) & 0xff],
           srgb_to_float[(srgb[3] >>  0) & 0xff],
           srgb_to_float[(srgb[4] >>  0) & 0xff],
           srgb_to_float[(srgb[5] >>  0) & 0xff],
           srgb_to_float[(srgb[6] >>  0) & 0xff],
           srgb_to_float[(srgb[7] >>  0) & 0xff] };

    *g = { srgb_to_float[(srgb[0] >>  8) & 0xff],
           srgb_to_float[(srgb[1] >>  8) & 0xff],
           srgb_to_float[(srgb[2] >>  8) & 0xff],
           srgb_to_float[(srgb[3] >>  8) & 0xff],
           srgb_to_float[(srgb[4] >>  8) & 0xff],
           srgb_to_float[(srgb[5] >>  8) & 0xff],
           srgb_to_float[(srgb[6] >>  8) & 0xff],
           srgb_to_float[(srgb[7] >>  8) & 0xff] };

    *b = { srgb_to_float[(srgb[0] >> 16) & 0xff],
           srgb_to_float[(srgb[1] >> 16) & 0xff],
           srgb_to_float[(srgb[2] >> 16) & 0xff],
           srgb_to_float[(srgb[3] >> 16) & 0xff],
           srgb_to_float[(srgb[4] >> 16) & 0xff],
           srgb_to_float[(srgb[5] >> 16) & 0xff],
           srgb_to_float[(srgb[6] >> 16) & 0xff],
           srgb_to_float[(srgb[7] >> 16) & 0xff] };

    auto p = reinterpret_cast<const __m256i*>(srgb);
    *a = _mm256_mul_ps(_mm256_set1_ps(1/255.0f),
                       _mm256_cvtepi32_ps(_mm256_srli_epi32(_mm256_loadu_si256(p), 24)));
}

static inline void floats_to_srgb(uint32_t srgb[8], f8 r, f8 g, f8 b, f8 a) {
    auto to_srgb = [](f8 l) {
        f8 rsqrt = _mm256_rsqrt_ps(l),
            sqrt = _mm256_rcp_ps(rsqrt),
            ftrt = _mm256_rsqrt_ps(rsqrt);

        f8 lo = _mm256_mul_ps(_mm256_set1_ps(12.92f * 255.0f), l);

        f8 hi = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(+0.422602055039580f * 255.0f), ftrt),
                _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                                            _mm256_set1_ps(-0.101115084998961f * 255.0f)));

        return _mm256_blendv_ps(hi, lo, _mm256_cmp_ps(l, _mm256_set1_ps(0.00349f), _CMP_LT_OS));
    };

    auto clamp_0_255 = [](f8 x) {
        // max/min order and argument order both matter.  This clamps NaN to 0.
        x = _mm256_max_ps(x, _mm256_set1_ps(0));
        x = _mm256_min_ps(x, _mm256_set1_ps(255));
        return x;
    };

    r = clamp_0_255(to_srgb(r));
    g = clamp_0_255(to_srgb(g));
    b = clamp_0_255(to_srgb(b));
    a = clamp_0_255(_mm256_mul_ps(a, _mm256_set1_ps(255.0f)));

    __m256i rgba = _mm256_or_si256(                  _mm256_cvtps_epi32(r)     ,
                   _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(g),  8),
                   _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(b), 16),
                                   _mm256_slli_epi32(_mm256_cvtps_epi32(a), 24))));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(srgb), rgba);
}

static f8 load_u8(const uint8_t cov[8]) {
    auto cov8 = reinterpret_cast<const __m128i*>(cov);
    return _mm256_mul_ps(_mm256_set1_ps(1/255.0f),
                         _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(cov8))));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

using avx2_fn = ABI void(*)(stage*, size_t, f8, f8, f8, f8);

static void next(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) {
    auto jump = reinterpret_cast<avx2_fn>(st->next);
    jump(st+1, x, r,g,b,a);
}

static ABI void load_srgb(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) {
    auto src = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(src+x, &r,&g,&b,&a);

    next(st,x,r,g,b,a);
}

static ABI void load_f16(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) {
    auto src = static_cast<const int64_t*>(st->ctx);

    auto lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src+x+0)),
         hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src+x+4));

    auto even = _mm256_unpacklo_epi16(lo, hi),
          odd = _mm256_unpackhi_epi16(lo, hi);

    auto rg = _mm256_unpacklo_epi16(even, odd),
         ba = _mm256_unpackhi_epi16(even, odd);

    rg = _mm256_permute4x64_epi64(rg, 0xd8);
    ba = _mm256_permute4x64_epi64(ba, 0xd8);

    r = _mm256_cvtph_ps(_mm256_extracti128_si256(rg, 0));
    g = _mm256_cvtph_ps(_mm256_extracti128_si256(rg, 1));
    b = _mm256_cvtph_ps(_mm256_extracti128_si256(ba, 0));
    a = _mm256_cvtph_ps(_mm256_extracti128_si256(ba, 1));

    next(st,x,r,g,b,a);
}

static ABI void scale_u8(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) {
    auto cov  = static_cast<const uint8_t*>(st->ctx);
    f8 c = load_u8(cov+x);
    r *= c;
    g *= c;
    b *= c;
    a *= c;

    next(st,x,r,g,b,a);
}

static ABI void srcover_srgb(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) {
    auto dst = static_cast<const uint32_t*>(st->ctx);
    f8 dr,dg,db,da;
    srgb_to_floats(dst+x, &dr,&dg,&db,&da);

    f8 A = _mm256_sub_ps(_mm256_set1_ps(1), a);
    r += dr * A;
    g += dg * A;
    b += db * A;
    a += da * A;

    next(st,x,r,g,b,a);
}

static ABI void lerp_u8_srgb(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) {
    auto cov = static_cast<const uint8_t*>(st->ctx);
    f8 c = load_u8(cov+x);

    auto dst = static_cast<const uint32_t*>(st->dtx);
    f8 dr,dg,db,da;
    srgb_to_floats(dst+x, &dr,&dg,&db,&da);

    r = dr + (r-dr)*c;
    g = dg + (g-dg)*c;
    b = db + (b-db)*c;
    a = da + (a-da)*c;

    next(st,x,r,g,b,a);
}

static ABI void store_srgb(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) {
    auto dst = static_cast<uint32_t*>(st->dtx);
    floats_to_srgb(dst+x, r,g,b,a);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void pipeline::add_avx2(Stage st, const void* ctx, void* dtx) {
    if (avx2_stages.size() == 0) {
        avx2_stages.reserve(8);
    }

    avx2_fn f = nullptr;
    switch (st) {
        case load_srgb:     f = ::load_srgb;     break;
        case load_f16:      f = ::load_f16;      break;
        case scale_u8:      f = ::scale_u8;      break;
        case srcover_srgb:  f = ::srcover_srgb;  break;
        case lerp_u8_srgb:  f = ::lerp_u8_srgb;  break;
        case store_srgb:    f = ::store_srgb;    break;
    }
    avx2_stages.push_back({ reinterpret_cast<void(*)(void)>(f), ctx, dtx });
}

void pipeline::call_avx2(size_t* x, size_t* n) {
    assert (avx2_stages.size() > 0);

    f8 u = _mm256_undefined_ps();
    auto start = reinterpret_cast<avx2_fn>(avx2_stages.back().next);
    while (*n >= 8) {
        start(avx2_stages.data(), *x, u,u,u,u);
        *x += 8;
        *n -= 8;
    }
}
