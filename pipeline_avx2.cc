#include "abi.h"
#include "pipeline.h"
#include "srgb.h"
#include "sse.h"
#include <assert.h>
#include <immintrin.h>

#if 0
static inline void srgb_to_floats_T(const uint32_t srgb[8],
                                    __m256* r, __m256* g, __m256* b, __m256* a) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srgb));

    *r = _mm256_i32gather_ps(srgb_to_float,
                             _mm256_and_si256(                  v     , _mm256_set1_epi32(0xff)),
                             4);
    *g = _mm256_i32gather_ps(srgb_to_float,
                             _mm256_and_si256(_mm256_srli_epi32(v,  8), _mm256_set1_epi32(0xff)),
                             4);
    *b = _mm256_i32gather_ps(srgb_to_float,
                             _mm256_and_si256(_mm256_srli_epi32(v, 16), _mm256_set1_epi32(0xff)),
                             4);

    *a = _mm256_mul_ps(_mm256_set1_ps(1/255.0f),
                       _mm256_cvtepi32_ps(_mm256_srli_epi32(v, 24)));
}
#else

static inline void srgb_to_floats_T(const uint32_t srgb[8],
                                    __m256* r, __m256* g, __m256* b, __m256* a) {
    *r = _mm256_setr_ps(srgb_to_float[(srgb[0] >>  0) & 0xff],
                        srgb_to_float[(srgb[1] >>  0) & 0xff],
                        srgb_to_float[(srgb[2] >>  0) & 0xff],
                        srgb_to_float[(srgb[3] >>  0) & 0xff],
                        srgb_to_float[(srgb[4] >>  0) & 0xff],
                        srgb_to_float[(srgb[5] >>  0) & 0xff],
                        srgb_to_float[(srgb[6] >>  0) & 0xff],
                        srgb_to_float[(srgb[7] >>  0) & 0xff]);

    *g = _mm256_setr_ps(srgb_to_float[(srgb[0] >>  8) & 0xff],
                        srgb_to_float[(srgb[1] >>  8) & 0xff],
                        srgb_to_float[(srgb[2] >>  8) & 0xff],
                        srgb_to_float[(srgb[3] >>  8) & 0xff],
                        srgb_to_float[(srgb[4] >>  8) & 0xff],
                        srgb_to_float[(srgb[5] >>  8) & 0xff],
                        srgb_to_float[(srgb[6] >>  8) & 0xff],
                        srgb_to_float[(srgb[7] >>  8) & 0xff]);

    *b = _mm256_setr_ps(srgb_to_float[(srgb[0] >> 16) & 0xff],
                        srgb_to_float[(srgb[1] >> 16) & 0xff],
                        srgb_to_float[(srgb[2] >> 16) & 0xff],
                        srgb_to_float[(srgb[3] >> 16) & 0xff],
                        srgb_to_float[(srgb[4] >> 16) & 0xff],
                        srgb_to_float[(srgb[5] >> 16) & 0xff],
                        srgb_to_float[(srgb[6] >> 16) & 0xff],
                        srgb_to_float[(srgb[7] >> 16) & 0xff]);

    auto p = reinterpret_cast<const __m256i*>(srgb);
    *a = _mm256_mul_ps(_mm256_set1_ps(1/255.0f),
                       _mm256_cvtepi32_ps(_mm256_srli_epi32(_mm256_loadu_si256(p), 24)));
}
#endif

static inline __m256 clamp_0_255(__m256 srgb) {
    // max/min order and argument order both matter.  This clamps NaN to 0.
    srgb = _mm256_max_ps(srgb, _mm256_set1_ps(0));
    srgb = _mm256_min_ps(srgb, _mm256_set1_ps(255));
    return srgb;
}

static inline __m256 floats_to_srgb_floats(__m256 l) {
    __m256 rsqrt = _mm256_rsqrt_ps(l),
            sqrt = _mm256_rcp_ps(rsqrt),
            ftrt = _mm256_rsqrt_ps(rsqrt);

    __m256 lo = _mm256_mul_ps(_mm256_set1_ps(12.92f * 255.0f), l);

    __m256 hi = _mm256_add_ps(_mm256_set1_ps(-0.101115084998961f * 255.0f),
                _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(+0.678513029959381f * 255.0f), sqrt),
                              _mm256_mul_ps(_mm256_set1_ps(+0.422602055039580f * 255.0f), ftrt)));

    return _mm256_blendv_ps(hi, lo, _mm256_cmp_ps(l, _mm256_set1_ps(0.00349f), _CMP_LT_OS));
}

static inline void floats_to_srgb_T(uint32_t srgb[8], __m256 r, __m256 g, __m256 b, __m256 a) {
    r = clamp_0_255(floats_to_srgb_floats(r));
    g = clamp_0_255(floats_to_srgb_floats(g));
    b = clamp_0_255(floats_to_srgb_floats(b));
    a = clamp_0_255(_mm256_mul_ps(a, _mm256_set1_ps(255.0f)));

    __m256i rgba = _mm256_or_si256(                  _mm256_cvtps_epi32(r)     ,
                   _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(g),  8),
                   _mm256_or_si256(_mm256_slli_epi32(_mm256_cvtps_epi32(b), 16),
                                   _mm256_slli_epi32(_mm256_cvtps_epi32(a), 24))));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(srgb), rgba);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

using wide_ymm = ABI void(*)(const pipeline::stage*, size_t, void*,
                             __m256, __m256, __m256, __m256,
                             __m256, __m256, __m256, __m256);

static bool load_d_srgb(const void*, size_t x, void* dp,
                        __m256* dr, __m256* dg, __m256* db, __m256* da,
                        __m256*   , __m256*   , __m256*   , __m256*   ) {
    auto dst = static_cast<uint32_t*>(dp);
    srgb_to_floats_T(dst+x, dr,dg,db,da);
    return false;
}

static bool load_s_srgb(const void* ctx, size_t x, void*,
                        __m256*   , __m256*   , __m256*   , __m256*   ,
                        __m256* sr, __m256* sg, __m256* sb, __m256* sa) {
    auto src = static_cast<const uint32_t*>(ctx);
    srgb_to_floats_T(src+x, sr,sg,sb,sa);
    return false;
}

static bool srcover(const void*, size_t, void*,
                    __m256* dr, __m256* dg, __m256* db, __m256* da,
                    __m256* sr, __m256* sg, __m256* sb, __m256* sa) {
    __m256 A = _mm256_sub_ps(_mm256_set1_ps(1), *sa);
    *sr = _mm256_add_ps(*sr, _mm256_mul_ps(*dr, A));
    *sg = _mm256_add_ps(*sg, _mm256_mul_ps(*dg, A));
    *sb = _mm256_add_ps(*sb, _mm256_mul_ps(*db, A));
    *sa = _mm256_add_ps(*sa, _mm256_mul_ps(*da, A));
    return false;
}

static bool lerp_u8(const void* ctx, size_t x, void*,
                    __m256* dr, __m256* dg, __m256* db, __m256* da,
                    __m256* sr, __m256* sg, __m256* sb, __m256* sa) {
    auto cov  = static_cast<const uint8_t*>(ctx);
    auto cov8 = reinterpret_cast<const int64_t*>(cov+x);
    __m256 c = _mm256_mul_ps(_mm256_set1_ps(1/255.0f),
                             _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(*cov8))));
    __m256 C = _mm256_sub_ps(_mm256_set1_ps(1), c);

    *sr = _mm256_add_ps(_mm256_mul_ps(*sr, c), _mm256_mul_ps(*dr, C));
    *sg = _mm256_add_ps(_mm256_mul_ps(*sg, c), _mm256_mul_ps(*dg, C));
    *sb = _mm256_add_ps(_mm256_mul_ps(*sb, c), _mm256_mul_ps(*db, C));
    *sa = _mm256_add_ps(_mm256_mul_ps(*sa, c), _mm256_mul_ps(*da, C));
    return false;
}

static bool store_s_srgb(const void*, size_t x, void* dp,
                         __m256*   , __m256*   , __m256*   , __m256*   ,
                         __m256* sr, __m256* sg, __m256* sb, __m256* sa) {
    auto dst = static_cast<uint32_t*>(dp);
    floats_to_srgb_T(dst+x, *sr, *sg, *sb, *sa);
    return true;
}

#define EXPORT_STAGE(name)                                            \
  static ABI void name(const pipeline::stage* st, size_t x, void* dp, \
                       __m256 dr, __m256 dg, __m256 db, __m256 da,    \
                       __m256 sr, __m256 sg, __m256 sb, __m256 sa) {  \
      if (!name(st->ctx, x,dp, &dr,&dg,&db,&da, &sr,&sg,&sb,&sa)) {   \
          auto next = reinterpret_cast<wide_ymm>(st->next);           \
          next(st+1, x,dp, dr,dg,db,da, sr,sg,sb,sa);                 \
      }                                                               \
  }

    EXPORT_STAGE(load_d_srgb);
    EXPORT_STAGE(load_s_srgb);
    EXPORT_STAGE(srcover);
    EXPORT_STAGE(lerp_u8);
    EXPORT_STAGE(store_s_srgb);

#undef EXPORT_STAGE

void pipeline::add_avx2_stage(Stage st, const void* ctx) {
    if (avx2_stages.size() == 0) {
        avx2_stages.reserve(8);
    }

    wide_ymm w = nullptr;
    switch (st) {
        case load_d_srgb:   w = ::load_d_srgb;  break;
        case load_s_srgb:   w = ::load_s_srgb;  break;
        case srcover:       w = ::srcover;      break;
        case lerp_u8:       w = ::lerp_u8;      break;
        case store_s_srgb:  w = ::store_s_srgb; break;
    }
    avx2_stages.push_back({ reinterpret_cast<void(*)(void)>(w), ctx });
}

size_t pipeline::call_avx2_stages(void* dp, size_t n) const {
    assert (avx2_stages.size() > 0);

    __m256 u = _mm256_undefined_ps();

    size_t stride = 8;
    size_t mask = stride-1;
    for (size_t x = 0; x < (n & ~mask); x += stride) {
        auto start = reinterpret_cast<wide_ymm>(avx2_stages.back().next);
        start(avx2_stages.data(), x, dp, u,u,u,u, u,u,u,u);
    }
    return n&mask;
}
