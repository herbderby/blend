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

static bool load_d_srgb(const void*, size_t x, void* dp, __m128* d, __m128*) {
    auto dst = static_cast<uint32_t*>(dp);
    *d = srgb_to_linear_floats(dst[x]);
    return false;
}

static bool load_s_srgb(const void* ctx, size_t x, void*, __m128*, __m128* s) {
    auto src = static_cast<const uint32_t*>(ctx);
    *s = srgb_to_linear_floats(src[x]);
    return false;
}

static bool srcover(const void*, size_t, void*, __m128* d, __m128* s) {
    __m128 a = _mm_shuffle_ps(*s,*s, 0xff);
    *s = _mm_add_ps(*s, _mm_mul_ps(*d, _mm_sub_ps(_mm_set1_ps(1), a)));
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

static bool store_s_srgb(const void*, size_t x, void* dp, __m128*, __m128* s) {
    auto dst = static_cast<uint32_t*>(dp);
    dst[x] = linear_floats_to_srgb(*s);
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

typedef ABI void stage_fn(const pipeline::stage*, size_t x, void* dp, __m128 d, __m128 s);

struct pipeline::stage {
    stage_fn* next;
    const void* ctx;
};

#define EXPORT_STAGE(name)                                                                       \
    static ABI void name(const pipeline::stage* stage, size_t x, void* dp, __m128 d, __m128 s) { \
        if (!name(stage->ctx, x,dp,&d,&s)) {                                                     \
            stage->next(stage+1, x,dp,d,s);                                                      \
        }                                                                                        \
    }

    EXPORT_STAGE(shortcircuit_srcover_both_srgb)
    EXPORT_STAGE(load_d_srgb)
    EXPORT_STAGE(load_s_srgb)
    EXPORT_STAGE(srcover)
    EXPORT_STAGE(lerp_u8)
    EXPORT_STAGE(store_s_srgb)

#undef EXPORT_STAGE

pipeline::pipeline() : stages(new std::vector<stage>) {}
pipeline::~pipeline() {}

void pipeline::add_stage(Stage stage, const void* ctx) {
    stage_fn* fn = nullptr;
    switch (stage) {
        case Stage::shortcircuit_srcover_both_srgb: fn =  shortcircuit_srcover_both_srgb; break;
        case Stage::load_d_srgb:                    fn =  load_d_srgb;                    break;
        case Stage::load_s_srgb:                    fn =  load_s_srgb;                    break;
        case Stage::srcover:                        fn =      srcover;                    break;
        case Stage::lerp_u8:                        fn =      lerp_u8;                    break;
        case Stage::store_s_srgb:                   fn = store_s_srgb;                    break;
    }
    stages->push_back({ fn, ctx });
}

void pipeline::ready() {
    assert (stages->size() > 0);

    auto start = (*stages)[0].next;
    for (size_t i = 0; i < stages->size(); i++) {
        (*stages)[i].next = (*stages)[i+1].next;
    }
    (*stages)[stages->size() - 1].next = start; // Not really, just a convenient place to stash it.
}

void pipeline::call(void* dp, size_t n) const {
    assert (stages->size() > 0);

    __m128 d = _mm_undefined_ps(),
           s = _mm_undefined_ps();
    for (size_t x = 0; x < n; x++) {
        auto start = stages->back().next;  // See pipeline::ready().
        start(stages->data(), x, dp, d,s);
    }
}
