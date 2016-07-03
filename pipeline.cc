#include "abi.h"
#include "pipeline.h"
#include "srgb.h"
#include <assert.h>
#include <immintrin.h>

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

static inline void srgb_to_floats(const uint32_t srgb[1], f1* r, f1* g, f1* b, f1* a) {
    *r = srgb_to_float[(*srgb >>  0) & 0xff];
    *g = srgb_to_float[(*srgb >>  8) & 0xff];
    *b = srgb_to_float[(*srgb >> 16) & 0xff];
    *a = (*srgb >> 24) * (1/255.0f);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

static inline f4 clamp_0_255(f4 x) {
    // max/min order and argument order both matter.  This clamps NaN to 0.
    x = _mm_max_ps(x, _mm_set1_ps(0));
    x = _mm_min_ps(x, _mm_set1_ps(255));
    return x;
}

static inline f4 to_srgb(f4 l) {
    f4 rsqrt = _mm_rsqrt_ps(l),
        sqrt = _mm_rcp_ps(rsqrt),
        ftrt = _mm_rsqrt_ps(rsqrt);

    f4 lo = _mm_mul_ps(_mm_set1_ps(12.92f * 255.0f), l);

    f4 hi = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.422602055039580f * 255.0f), ftrt),
            _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                                  _mm_set1_ps(-0.101115084998961f * 255.0f)));

    return _mm_blendv_ps(hi, lo, _mm_cmplt_ps(l, _mm_set1_ps(0.00349f)));
}

static inline void floats_to_srgb(uint32_t srgb[4], f4 r, f4 g, f4 b, f4 a) {
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

static inline void floats_to_srgb(uint32_t srgb[1], f1 r, f1 g, f1 b, f1 a) {
    f4 s = to_srgb(f4{r,g,b,0});
    s = { s[0], s[1], s[2], a*255 };
    s = clamp_0_255(s);

    *srgb = static_cast<uint32_t>(
        _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(s),
                                           _mm_setr_epi8(0,4,8,12, 0,0,0,0,0,0,0,0,0,0,0,0))));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

using f4_fn = ABI void(*)(stage*, size_t, f4, f4, f4, f4);

static void next(stage* st, size_t x, f4 r, f4 g, f4 b, f4 a) {
    auto next = reinterpret_cast<f4_fn>(st->next);
    next(st+1, x, r,g,b,a);
}

static void load_srgb(stage* st, size_t x, f4 r, f4 g, f4 b, f4 a) {
    auto src = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(src+x, &r,&g,&b,&a);

    next(st,x,r,g,b,a);
}

static void scale_u8(stage* st, size_t x, f4 r, f4 g, f4 b, f4 a) {
    auto cov  = static_cast<const uint8_t*>(st->ctx);
    auto cov4 = reinterpret_cast<const int*>(cov+x);
    f4 c = _mm_mul_ps(_mm_set1_ps(1/255.0f),
                      _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*cov4))));
    r *= c;
    g *= c;
    b *= c;
    a *= c;

    next(st,x,r,g,b,a);
}

static void srcover_srgb(stage* st, size_t x, f4 r, f4 g, f4 b, f4 a) {
    auto dst = static_cast<uint32_t*>(st->ctx);
    f4 dr,dg,db,da;
    srgb_to_floats(dst+x, &dr,&dg,&db,&da);

    f4 A = _mm_sub_ps(_mm_set1_ps(1), a);
    r += dr * A;
    g += dg * A;
    b += db * A;
    a += da * A;

    floats_to_srgb(dst+x, r, g, b, a);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
//
using f1_fn = ABI void(*)(stage*, size_t, f1, f1, f1, f1);

static void next(stage* st, size_t x, f1 r, f1 g, f1 b, f1 a) {
    auto next = reinterpret_cast<f1_fn>(st->next);
    next(st+1, x, r,g,b,a);
}

static void load_srgb(stage* st, size_t x, f1 r, f1 g, f1 b, f1 a) {
    auto src = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(src+x, &r,&g,&b,&a);

    next(st,x,r,g,b,a);
}

static void scale_u8(stage* st, size_t x, f1 r, f1 g, f1 b, f1 a) {
    auto cov  = static_cast<const uint8_t*>(st->ctx);
    f1 c = cov[x] * (1/255.0f);
    r *= c;
    g *= c;
    b *= c;
    a *= c;

    next(st,x,r,g,b,a);
}

static void srcover_srgb(stage* st, size_t x, f1 r, f1 g, f1 b, f1 a) {
    auto dst = static_cast<uint32_t*>(st->ctx);
    f1 dr,dg,db,da;
    srgb_to_floats(dst+x, &dr,&dg,&db,&da);

    f1 A = 1.0f - a;
    r += dr * A;
    g += dg * A;
    b += db * A;
    a += da * A;

    floats_to_srgb(dst+x, r, g, b, a);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void pipeline::add_stage(Stage st, const void* ctx) {
    if (f4_stages.size() == 0) {
        f4_stages.reserve(8);
        f1_stages.reserve(8);
    }

    this->add_f8_stage(st, ctx);

    {
        f4_fn f = nullptr;
        switch (st) {
            case load_srgb:    f = ::load_srgb;    break;
            case scale_u8:     f = ::scale_u8;     break;
            case srcover_srgb: f = ::srcover_srgb; break;
        }
        f4_stages.push_back({ reinterpret_cast<void(*)(void)>(f), const_cast<void*>(ctx) });
    }

    {
        f1_fn f = nullptr;
        switch (st) {
            case load_srgb:    f = ::load_srgb;    break;
            case scale_u8:     f = ::scale_u8;     break;
            case srcover_srgb: f = ::srcover_srgb; break;
        }
        f1_stages.push_back({ reinterpret_cast<void(*)(void)>(f), const_cast<void*>(ctx) });
    }
}

void pipeline::ready() {
    auto rewire = [](std::vector<stage>* stages) {
        auto start = (*stages)[0].next;
        for (size_t i = 0; i < stages->size(); i++) {
            (*stages)[i].next = (*stages)[i+1].next;
        }
        (*stages)[stages->size() - 1].next = start;
    };

    rewire(&f8_stages);
    rewire(&f4_stages);
    rewire(&f1_stages);
}

void pipeline::call(size_t n) {
    assert (f4_stages.size() > 0);
    assert (f1_stages.size() > 0);

    size_t x = this->call_f8(n);
    n -= x;

    while (n >= 4) {
        f4 u = _mm_undefined_ps();
        auto start = reinterpret_cast<f4_fn>(f4_stages.back().next);
        start(f4_stages.data(), x, u,u,u,u);

        x += 4;
        n -= 4;
    }
    while (n > 0) {
        f1 u = _mm_undefined_ps()[0];
        auto start = reinterpret_cast<f1_fn>(f1_stages.back().next);
        start(f1_stages.data(), x, u,u,u,u);

        x += 1;
        n -= 1;
    }
}
