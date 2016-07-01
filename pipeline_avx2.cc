#include "abi.h"
#include "pipeline.h"
#include "srgb.h"
#include "sse.h"
#include <immintrin.h>

using wide_ymm = ABI void(*)(const pipeline::stage*, size_t, void*,
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
