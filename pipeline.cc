#include "abi.h"
#include "pipeline.h"
#include "srgb.h"
#include "sse.h"
#include <assert.h>
#include <immintrin.h>

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

    __m128 hi = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.422602055039580f * 255.0f), ftrt),
                _mm_add_ps(_mm_mul_ps(_mm_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                                      _mm_set1_ps(-0.101115084998961f * 255.0f)));

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

using narrow_xmm = ABI void(*)(const pipeline::stage*, size_t, void*, __m128, __m128);

static bool load_d_srgb(const void*, size_t x, void* dp, __m128* d, __m128*) {
    auto dst = static_cast<uint32_t*>(dp);
    *d = srgb_to_floats(dst[x]);
    return false;
}

static bool load_s_srgb(const void* ctx, size_t x, void*, __m128*, __m128* s) {
    auto src = static_cast<const uint32_t*>(ctx);
    *s = srgb_to_floats(src[x]);
    return false;
}

static bool srcover(const void*, size_t, void*, __m128* d, __m128* s) {
    __m128 a = _mm_shuffle_ps(*s,*s, 0xff);
    *s = _mm_add_ps(*s, _mm_mul_ps(*d, _mm_sub_ps(_mm_set1_ps(1), a)));
    return false;
}

static bool lerp_u8(const void* ctx, size_t x, void*, __m128* d, __m128* s) {
    auto cov = static_cast<const uint8_t*>(ctx);
    __m128 c = better_cvtsi32_ss(cov[x]);
           c = _mm_mul_ps(_mm_shuffle_ps(c,c,0x00), _mm_set1_ps(1/255.0f));
    __m128 C = _mm_sub_ps(_mm_set1_ps(1), c);
    *s = _mm_add_ps(_mm_mul_ps(*s, c), _mm_mul_ps(*d, C));
    return false;
}

static bool store_s_srgb(const void*, size_t x, void* dp, __m128*, __m128* s) {
    auto dst = static_cast<uint32_t*>(dp);
    dst[x] = floats_to_srgb(*s);
    return true;
}

#define EXPORT_STAGE(name)                                                                  \
  static ABI void name(const pipeline::stage* st, size_t x, void* dp, __m128 d, __m128 s) { \
      if (!name(st->ctx, x,dp,&d,&s)) {                                                     \
          auto next = reinterpret_cast<narrow_xmm>(st->next);                               \
          next(st+1, x,dp,d,s);                                                             \
      }                                                                                     \
  }

    EXPORT_STAGE(load_d_srgb)
    EXPORT_STAGE(load_s_srgb)
    EXPORT_STAGE(srcover)
    EXPORT_STAGE(lerp_u8)
    EXPORT_STAGE(store_s_srgb)

#undef EXPORT_STAGE

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

using wide_xmm = ABI void(*)(const pipeline::stage*, size_t, void*,
                             __m128, __m128, __m128, __m128,
                             __m128, __m128, __m128, __m128);

static bool load_d_srgb(const void*, size_t x, void* dp,
                        __m128* dr, __m128* dg, __m128* db, __m128* da,
                        __m128*   , __m128*   , __m128*   , __m128*   ) {
    auto dst = static_cast<uint32_t*>(dp);
    srgb_to_floats_T(dst+x, dr,dg,db,da);
    return false;
}

static bool load_s_srgb(const void* ctx, size_t x, void*,
                        __m128*   , __m128*   , __m128*   , __m128*   ,
                        __m128* sr, __m128* sg, __m128* sb, __m128* sa) {
    auto src = static_cast<const uint32_t*>(ctx);
    srgb_to_floats_T(src+x, sr,sg,sb,sa);
    return false;
}

static bool srcover(const void*, size_t, void*,
                    __m128* dr, __m128* dg, __m128* db, __m128* da,
                    __m128* sr, __m128* sg, __m128* sb, __m128* sa) {
    __m128 A = _mm_sub_ps(_mm_set1_ps(1), *sa);
    *sr = _mm_add_ps(*sr, _mm_mul_ps(*dr, A));
    *sg = _mm_add_ps(*sg, _mm_mul_ps(*dg, A));
    *sb = _mm_add_ps(*sb, _mm_mul_ps(*db, A));
    *sa = _mm_add_ps(*sa, _mm_mul_ps(*da, A));
    return false;
}

static bool lerp_u8(const void* ctx, size_t x, void*,
                    __m128* dr, __m128* dg, __m128* db, __m128* da,
                    __m128* sr, __m128* sg, __m128* sb, __m128* sa) {
    auto cov  = static_cast<const uint8_t*>(ctx);
    auto cov4 = reinterpret_cast<const int*>(cov+x);
    __m128 c = _mm_mul_ps(_mm_set1_ps(1/255.0f),
                          _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*cov4))));
    __m128 C = _mm_sub_ps(_mm_set1_ps(1), c);

    *sr = _mm_add_ps(_mm_mul_ps(*sr, c), _mm_mul_ps(*dr, C));
    *sg = _mm_add_ps(_mm_mul_ps(*sg, c), _mm_mul_ps(*dg, C));
    *sb = _mm_add_ps(_mm_mul_ps(*sb, c), _mm_mul_ps(*db, C));
    *sa = _mm_add_ps(_mm_mul_ps(*sa, c), _mm_mul_ps(*da, C));
    return false;
}

static bool store_s_srgb(const void*, size_t x, void* dp,
                         __m128*   , __m128*   , __m128*   , __m128*   ,
                         __m128* sr, __m128* sg, __m128* sb, __m128* sa) {
    auto dst = static_cast<uint32_t*>(dp);
    floats_to_srgb_T(dst+x, *sr, *sg, *sb, *sa);
    return true;
}

#define EXPORT_STAGE(name)                                            \
  static ABI void name(const pipeline::stage* st, size_t x, void* dp, \
                       __m128 dr, __m128 dg, __m128 db, __m128 da,    \
                       __m128 sr, __m128 sg, __m128 sb, __m128 sa) {  \
      if (!name(st->ctx, x,dp, &dr,&dg,&db,&da, &sr,&sg,&sb,&sa)) {   \
          auto next = reinterpret_cast<wide_xmm>(st->next);           \
          next(st+1, x,dp, dr,dg,db,da, sr,sg,sb,sa);                 \
      }                                                               \
  }

    EXPORT_STAGE(load_d_srgb)
    EXPORT_STAGE(load_s_srgb)
    EXPORT_STAGE(srcover)
    EXPORT_STAGE(lerp_u8)
    EXPORT_STAGE(store_s_srgb)

#undef EXPORT_STAGE

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void pipeline::add_stage(Stage st, const void* ctx) {
    if (narrow_stages.size() == 0) {
        narrow_stages.reserve(8);
    }
    if (wide_stages.size() == 0) {
        wide_stages.reserve(8);
    }

    narrow_xmm n = nullptr;
    switch (st) {
        case load_d_srgb:   n = ::load_d_srgb;  break;
        case load_s_srgb:   n = ::load_s_srgb;  break;
        case srcover:       n = ::srcover;      break;
        case lerp_u8:       n = ::lerp_u8;      break;
        case store_s_srgb:  n = ::store_s_srgb; break;
    }
    narrow_stages.push_back({ reinterpret_cast<void(*)(void)>(n), ctx });

    wide_xmm w = nullptr;
    switch (st) {
        case load_d_srgb:   w = ::load_d_srgb;  break;
        case load_s_srgb:   w = ::load_s_srgb;  break;
        case srcover:       w = ::srcover;      break;
        case lerp_u8:       w = ::lerp_u8;      break;
        case store_s_srgb:  w = ::store_s_srgb; break;
    }
    wide_stages.push_back({ reinterpret_cast<void(*)(void)>(w), ctx });

    this->add_avx2_stage(st, ctx);
}

static void rewire(std::vector<pipeline::stage>* stages) {
    assert (stages->size() > 0);

    auto start = (*stages)[0].next;
    for (size_t i = 0; i < stages->size(); i++) {
        (*stages)[i].next = (*stages)[i+1].next;
    }
    (*stages)[stages->size() - 1].next = start;
}

void pipeline::ready() {
    rewire(&narrow_stages);
    rewire(&  wide_stages);
    rewire(&  avx2_stages);
}

void pipeline::call(void* dp, size_t n) const {
    assert (narrow_stages.size() > 0);
    assert (  wide_stages.size() > 0);

    n = this->call_avx2_stages(dp, n);

    __m128 u = _mm_undefined_ps();

    size_t stride = 4;
    size_t mask = stride-1;
    for (size_t x = 0; x < (n & ~mask); x += stride) {
        auto start = reinterpret_cast<wide_xmm>(wide_stages.back().next);
        start(wide_stages.data(), x, dp, u,u,u,u, u,u,u,u);
    }
    n &= mask;

    for (size_t x = 0; x < n; x++) {
        auto start = reinterpret_cast<narrow_xmm>(narrow_stages.back().next);
        start(narrow_stages.data(), x, dp, u,u);
    }
}

void fused(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n) {
    __m128 d = _mm_undefined_ps(),
           s = _mm_undefined_ps();
    for (size_t x = 0; x < n; x++) {
        if (                   load_d_srgb(NULL, x,dst,&d,&s)) continue;
        if (                   load_s_srgb( src, x,dst,&d,&s)) continue;
        if (                       srcover(NULL, x,dst,&d,&s)) continue;
        if (                       lerp_u8( cov, x,dst,&d,&s)) continue;
        if (                  store_s_srgb(NULL, x,dst,&d,&s)) continue;
    }
}
