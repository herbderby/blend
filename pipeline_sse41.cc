#include "abi.h"
#include "pipeline.h"
#include "srgb.h"
#include <assert.h>
#include <immintrin.h>

using f4 = __m128;

static inline void srgb_to_floats(const uint32_t srgb[4], f4* r, f4* g, f4* b, f4* a) {
    *r = { srgb_to_float[(srgb[0] >>  0) & 0xff],
           srgb_to_float[(srgb[1] >>  0) & 0xff],
           srgb_to_float[(srgb[2] >>  0) & 0xff],
           srgb_to_float[(srgb[3] >>  0) & 0xff] };

    *g = { srgb_to_float[(srgb[0] >>  8) & 0xff],
           srgb_to_float[(srgb[1] >>  8) & 0xff],
           srgb_to_float[(srgb[2] >>  8) & 0xff],
           srgb_to_float[(srgb[3] >>  8) & 0xff] };

    *b = { srgb_to_float[(srgb[0] >> 16) & 0xff],
           srgb_to_float[(srgb[1] >> 16) & 0xff],
           srgb_to_float[(srgb[2] >> 16) & 0xff],
           srgb_to_float[(srgb[3] >> 16) & 0xff] };

    auto p = reinterpret_cast<const __m128i*>(srgb);
    *a = _mm_mul_ps(_mm_set1_ps(1/255.0f),
                    _mm_cvtepi32_ps(_mm_srli_epi32(_mm_loadu_si128(p), 24)));
}

static inline void floats_to_srgb(uint32_t srgb[4], f4 r, f4 g, f4 b, f4 a) {
    auto to_srgb = [](f4 l) {
        f4 rsqrt = _mm_rsqrt_ps(l),
            sqrt = _mm_rcp_ps(rsqrt),
            ftrt = _mm_rsqrt_ps(rsqrt);

        f4 lo = _mm_mul_ps(_mm_set1_ps(12.92f * 255.0f), l);

        f4 hi = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.422602055039580f * 255.0f), ftrt),
                _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                                      _mm_set1_ps(-0.101115084998961f * 255.0f)));

        return _mm_blendv_ps(hi, lo, _mm_cmplt_ps(l, _mm_set1_ps(0.00349f)));
    };

    auto clamp_0_255 = [](f4 x) {
        // max/min order and argument order both matter.  This clamps NaN to 0.
        x = _mm_max_ps(x, _mm_set1_ps(0));
        x = _mm_min_ps(x, _mm_set1_ps(255));
        return x;
    };

    r = clamp_0_255(to_srgb(r));
    g = clamp_0_255(to_srgb(g));
    b = clamp_0_255(to_srgb(b));
    a = clamp_0_255(_mm_mul_ps(a, _mm_set1_ps(255.0f)));

    __m128i rgba = _mm_or_si128(               _mm_cvtps_epi32(r)     ,
                   _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(g),  8),
                   _mm_or_si128(_mm_slli_epi32(_mm_cvtps_epi32(b), 16),
                                _mm_slli_epi32(_mm_cvtps_epi32(a), 24))));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(srgb), rgba);
}

static f4 load_u8(const uint8_t cov[4]) {
    auto cov4 = reinterpret_cast<const int*>(cov);
    return _mm_mul_ps(_mm_set1_ps(1/255.0f),
                      _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*cov4))));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

using sse41_fn = ABI void(*)(stage*, size_t, f4,f4,f4,f4, f4,f4,f4,f4);

static void next(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                      f4 dr, f4 dg, f4 db, f4 da) {
    auto next = reinterpret_cast<sse41_fn>(st->next);
    next(st+1, x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void load_s_srgb(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                                 f4 dr, f4 dg, f4 db, f4 da) {
    auto ptr = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(ptr+x, &sr,&sg,&sb,&sa);

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void load_d_srgb(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                                 f4 dr, f4 dg, f4 db, f4 da) {
    auto ptr = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(ptr+x, &dr,&dg,&db,&da);

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void srcover(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                             f4 dr, f4 dg, f4 db, f4 da) {
    f4 A = _mm_sub_ps(_mm_set1_ps(1), sa);
    sr += dr * A;
    sg += dg * A;
    sb += db * A;
    sa += da * A;

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void scale_u8(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                              f4 dr, f4 dg, f4 db, f4 da) {
    auto cov = static_cast<const uint8_t*>(st->ctx);
    f4 c = load_u8(cov+x);

    sr *= c;
    sg *= c;
    sb *= c;
    sa *= c;

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void lerp_u8(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                             f4 dr, f4 dg, f4 db, f4 da) {
    auto cov = static_cast<const uint8_t*>(st->ctx);
    f4 c = load_u8(cov+x);

    sr = dr + (sr-dr)*c;
    sg = dg + (sg-dg)*c;
    sb = db + (sb-db)*c;
    sa = da + (sa-da)*c;

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void store_srgb(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                                f4   , f4   , f4   , f4   ) {
    auto ptr = static_cast<uint32_t*>(st->ctx);
    floats_to_srgb(ptr+x, sr,sg,sb,sa);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void pipeline::add_sse41(Stage st, void* ctx) {
    if (sse41_stages.size() == 0) {
        sse41_stages.reserve(8);
    }

    sse41_fn f = nullptr;
    switch (st) {
        case load_s_srgb: f = ::load_s_srgb;   break;
        case load_d_srgb: f = ::load_d_srgb;   break;
        case     srcover: f = ::srcover;       break;
        case    scale_u8: f = ::scale_u8;      break;
        case     lerp_u8: f = ::lerp_u8;       break;
        case  store_srgb: f = ::store_srgb;    break;
    }
    sse41_stages.push_back({ reinterpret_cast<void(*)(void)>(f), ctx });
}

void pipeline::call_sse41(size_t* x, size_t* n) {
    assert (sse41_stages.size() > 0);

    f4 u = _mm_undefined_ps();
    auto start = reinterpret_cast<sse41_fn>(sse41_stages.back().next);
    while (*n >= 4) {
        start(sse41_stages.data(), *x, u,u,u,u, u,u,u,u);
        *x += 4;
        *n -= 4;
    }
}
