#include "pipeline.h"
#include "srgb.h"
#include "sse.h"
#include <algorithm>
#include <assert.h>
#include <immintrin.h>
#include <vector>

static bool shortcircuit_srcover_both_srgb(const void* ctx, size_t x, void* dp,
                                           __m128*, __m128*) {
    auto src = static_cast<const uint32_t*>(ctx);
    auto dst = static_cast<      uint32_t*>( dp);
    switch (src[x] >> 24) {
        case 255: dst[x] = src[x]; return true;
        case   0:                  return true;
    }
    return false;
}

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
    *s = _mm_add_ps(_mm_mul_ps(*s, c), _mm_mul_ps(*d, _mm_sub_ps(_mm_set1_ps(1), c)));
    return false;
}

static bool store_s_srgb(const void*, size_t x, void* dp, __m128*, __m128* s) {
    auto dst = static_cast<uint32_t*>(dp);
    dst[x] = floats_to_srgb(*s);
    return true;
}

void fused(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n) {
    __m128 d = _mm_undefined_ps(),
           s = _mm_undefined_ps();
    for (size_t x = 0; x < n; x++) {
        //if (shortcircuit_srcover_both_srgb( src, x,dst,&d,&s)) continue;
        if (                   load_d_srgb(NULL, x,dst,&d,&s)) continue;
        if (                   load_s_srgb( src, x,dst,&d,&s)) continue;
        if (                       srcover(NULL, x,dst,&d,&s)) continue;
        if (                       lerp_u8( cov, x,dst,&d,&s)) continue;
        if (                  store_s_srgb(NULL, x,dst,&d,&s)) continue;
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

#if 0
    #define ABI __attribute__((vectorcall))
#elif 0
    #define ABI __attribute__((sysv_abi))
#else
    #define ABI
#endif

using narrow_xmm = void(*)(const pipeline::stage*, size_t, void*, __m128, __m128);

#define EXPORT_STAGE(name)                                                                  \
  static ABI void name(const pipeline::stage* st, size_t x, void* dp, __m128 d, __m128 s) { \
      if (!name(st->ctx, x,dp,&d,&s)) {                                                     \
          auto next = reinterpret_cast<narrow_xmm>(st->next);                               \
          next(st+1, x,dp,d,s);                                                             \
      }                                                                                     \
  }                                                                                         \

    EXPORT_STAGE(shortcircuit_srcover_both_srgb)
    EXPORT_STAGE(load_d_srgb)
    EXPORT_STAGE(load_s_srgb)
    EXPORT_STAGE(srcover)
    EXPORT_STAGE(lerp_u8)
    EXPORT_STAGE(store_s_srgb)

#undef EXPORT_STAGE

void pipeline::add_stage(Stage st, const void* ctx) {
    if (stages.size() == 0) {
        stages.reserve(8);
    }

    narrow_xmm f = nullptr;
    switch (st) {
        case shortcircuit_srcover_both_srgb: f =  ::shortcircuit_srcover_both_srgb; break;
        case load_d_srgb:                    f =  ::load_d_srgb;                    break;
        case load_s_srgb:                    f =  ::load_s_srgb;                    break;
        case srcover:                        f =      ::srcover;                    break;
        case lerp_u8:                        f =      ::lerp_u8;                    break;
        case store_s_srgb:                   f = ::store_s_srgb;                    break;
    }
    stages.push_back({ reinterpret_cast<void(*)(void)>(f), ctx });
}

void pipeline::ready() {
    assert (stages.size() > 0);

    auto start = stages[0].next;
    for (size_t i = 0; i < stages.size(); i++) {
        stages[i].next = stages[i+1].next;
    }
    stages[stages.size() - 1].next = start;
}

void pipeline::call(void* dp, size_t n) const {
    assert (stages.size() > 0);

    for (size_t x = 0; x < n; x++) {
        auto start = reinterpret_cast<narrow_xmm>(stages.back().next);
        start(stages.data(), x, dp, _mm_undefined_ps(), _mm_undefined_ps());
    }
}
