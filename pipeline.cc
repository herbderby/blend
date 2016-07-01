#include "pipeline.h"
#include "srgb.h"
#include "sse.h"
#include <algorithm>
#include <assert.h>
#include <immintrin.h>

static bool shortcircuit_srcover_both_srgb(const void* ctx, size_t x, void* dp, __m128*, __m128*) {
    auto src = static_cast<const uint32_t*>(ctx);
    auto dst = static_cast<      uint32_t*>( dp);
    switch (src[x] >> 24) {
        case 255: dst[x] = src[x]; return true;
        case   0:                  return true;
    }
    return false;
}
static bool shortcircuit_srcover_both_srgb(const void* ctx, size_t x, void* dp, __m128i*, __m128i*) {
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
    *d = srgb_to_linear_floats(dst[x]);
    return false;
}
static bool load_d_srgb(const void*, size_t x, void* dp, __m128i* d, __m128i*) {
    auto dst = static_cast<uint64_t*>(dp);
    *d = srgb_to_linear_q15s(dst[x]);
    return false;
}

static bool load_s_srgb(const void* ctx, size_t x, void*, __m128*, __m128* s) {
    auto src = static_cast<const uint32_t*>(ctx);
    *s = srgb_to_linear_floats(src[x]);
    return false;
}
static bool load_s_srgb(const void* ctx, size_t x, void*, __m128i*, __m128i* s) {
    auto src = static_cast<const uint64_t*>(ctx);
    *s = srgb_to_linear_q15s(src[x]);
    return false;
}

static bool srcover(const void*, size_t, void*, __m128* d, __m128* s) {
    __m128 a = _mm_shuffle_ps(*s,*s, 0xff);
    *s = _mm_add_ps(*s, _mm_mul_ps(*d, _mm_sub_ps(_mm_set1_ps(1), a)));
    return false;
}
static bool srcover(const void*, size_t, void*, __m128i* d, __m128i* s) {
    __m128i a = _mm_shufflehi_epi16(_mm_shufflelo_epi16(*s, 0xff), 0xff);
    *s = _mm_add_epi16(*s, mul_q15(*d, _mm_sub_epi16(_mm_set1_epi16(-0x8000), a)));
    return false;
}

static bool lerp_u8(const void* ctx, size_t x, void*, __m128* d, __m128* s) {
    auto cov = static_cast<const uint8_t*>(ctx);
    __m128 c = _mm_mul_ps(better_cvtsi32_ss(cov[x]), _mm_set1_ps(1/255.0f));
    c = _mm_shuffle_ps(c,c,0x00);
    __m128 C = _mm_sub_ps(_mm_set1_ps(1), c);
    *s = _mm_add_ps(_mm_mul_ps(*s, c), _mm_mul_ps(*d, C));
    return false;
}
static bool lerp_u8(const void* ctx, size_t x, void*, __m128i* d, __m128i* s) {
    auto cov = static_cast<const uint16_t*>(ctx);

    int16_t c0 = byte_to_linear_q15(cov[x] & 0xff),
            c1 = byte_to_linear_q15(cov[x] >>   8);

    __m128i c = _mm_shuffle_epi8(_mm_cvtsi32_si128((c1 << 16) | c0),
                                 _mm_setr_epi8(0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3));
    __m128i C = _mm_sub_epi16(_mm_set1_epi16(-0x8000), c);

    *s = _mm_add_epi16(mul_q15(*s, c), mul_q15(*d, C));
    return false;
}

static bool store_s_srgb(const void*, size_t x, void* dp, __m128*, __m128* s) {
    auto dst = static_cast<uint32_t*>(dp);
    dst[x] = linear_floats_to_srgb(*s);
    return true;
}
static bool store_s_srgb(const void*, size_t x, void* dp, __m128i*, __m128i* s) {
    auto dst = static_cast<uint64_t*>(dp);
    dst[x] = linear_q15s_to_srgb(*s);
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

typedef
ABI void stage_f_fn  (const pipeline::stage_f*,   size_t x, void* dp, __m128  d, __m128  s);
typedef
ABI void stage_q15_fn(const pipeline::stage_q15*, size_t x, void* dp, __m128i d, __m128i s);

struct pipeline::stage_f {
    stage_f_fn* next;
    const void* ctx;
};
struct pipeline::stage_q15 {
    stage_q15_fn* next;
    const void*   ctx;
};

#define EXPORT_STAGE(name)                                                                        \
  static ABI void name(const pipeline::stage_f* st, size_t x, void* dp, __m128 d, __m128 s) {     \
      if (!name(st->ctx, x,dp,&d,&s)) {                                                           \
          st->next(st+1, x,dp,d,s);                                                               \
      }                                                                                           \
  }                                                                                               \
  static ABI void name(const pipeline::stage_q15* st, size_t x, void* dp, __m128i d, __m128i s) { \
      if (!name(st->ctx, x,dp,&d,&s)) {                                                           \
          st->next(st+1, x,dp,d,s);                                                               \
      }                                                                                           \
  }

    EXPORT_STAGE(shortcircuit_srcover_both_srgb)
    EXPORT_STAGE(load_d_srgb)
    EXPORT_STAGE(load_s_srgb)
    EXPORT_STAGE(srcover)
    EXPORT_STAGE(lerp_u8)
    EXPORT_STAGE(store_s_srgb)

#undef EXPORT_STAGE

pipeline::pipeline() : stages_f  (new std::vector<stage_f  >)
                     , stages_q15(new std::vector<stage_q15>) {
    stages_f  ->reserve(8);
    stages_q15->reserve(8);
}
pipeline::~pipeline() {}

void pipeline::add_stage(Stage stage, const void* ctx) {
    stage_f_fn* f = nullptr;
    switch (stage) {
        case Stage::shortcircuit_srcover_both_srgb: f =  shortcircuit_srcover_both_srgb; break;
        case Stage::load_d_srgb:                    f =  load_d_srgb;                    break;
        case Stage::load_s_srgb:                    f =  load_s_srgb;                    break;
        case Stage::srcover:                        f =      srcover;                    break;
        case Stage::lerp_u8:                        f =      lerp_u8;                    break;
        case Stage::store_s_srgb:                   f = store_s_srgb;                    break;
    }
    stages_f->push_back({ f, ctx });

    stage_q15_fn* q15 = nullptr;
    switch (stage) {
        case Stage::shortcircuit_srcover_both_srgb: q15 =  shortcircuit_srcover_both_srgb; break;
        case Stage::load_d_srgb:                    q15 =  load_d_srgb;                    break;
        case Stage::load_s_srgb:                    q15 =  load_s_srgb;                    break;
        case Stage::srcover:                        q15 =      srcover;                    break;
        case Stage::lerp_u8:                        q15 =      lerp_u8;                    break;
        case Stage::store_s_srgb:                   q15 = store_s_srgb;                    break;
    }
    stages_q15->push_back({ q15, ctx });
}

template <typename T>
static void ready_stages(std::vector<T>* stages) {
    assert (stages->size() > 0);
    auto start = (*stages)[0].next;
    for (size_t i = 0; i < stages->size(); i++) {
        (*stages)[i].next = (*stages)[i+1].next;
    }
    (*stages)[stages->size() - 1].next = start;
}

void pipeline::ready() {
    ready_stages(stages_f  .get());
    ready_stages(stages_q15.get());
}

void pipeline::call(void* dp, size_t n, bool use_float_stages) const {
    if (use_float_stages) {
        assert (stages_f->size() > 0);

        __m128 d = _mm_undefined_ps(),
               s = _mm_undefined_ps();
        for (size_t x = 0; x < n; x++) {
            auto start = stages_f->back().next;
            start(stages_f->data(), x, dp, d,s);
        }
    } else {
        assert (stages_q15->size() > 0);

        __m128i d = _mm_undefined_si128(),
                s = _mm_undefined_si128();
        for (size_t x = 0; x < n/2; x++) {  // TODO: handle odd n
            auto start = stages_q15->back().next;
            start(stages_q15->data(), x, dp, d,s);
        }
    }
}
