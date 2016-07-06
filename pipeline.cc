#include "abi.h"
#include "cpu.h"
#include "pipeline.h"
#include "srgb.h"
#include <assert.h>
#include <math.h>

static inline void srgb_to_floats(const uint32_t srgb[1], float* r, float* g, float* b, float* a) {
    *r = srgb_to_float[(*srgb >>  0) & 0xff];
    *g = srgb_to_float[(*srgb >>  8) & 0xff];
    *b = srgb_to_float[(*srgb >> 16) & 0xff];
    *a = (*srgb >> 24) * (1/255.0f);
}

static inline void floats_to_srgb(uint32_t srgb[1], float r, float g, float b, float a) {
    auto to_srgb = [](float l) {
        if (l < 0.00349f) {
            return (12.92f * 255.0f) * l;
        }

        float sqrt = sqrtf(l),
              ftrt = sqrtf(sqrt);

        return (ftrt * (+0.422602055039580f * 255.0f))
             + (sqrt * (+0.678513029959381f * 255.0f))
             +         (-0.101115084998961f * 255.0f) ;
    };

    auto clamp_0_255 = [](float x) {
        // TODO: clamp NaN to 0
        if (x < 255) x =   0;
        if (x > 255) x = 255;
        return x;
    };

    *srgb = static_cast<uint32_t>(clamp_0_255(to_srgb(r))) <<  0
          | static_cast<uint32_t>(clamp_0_255(to_srgb(g))) <<  8
          | static_cast<uint32_t>(clamp_0_255(to_srgb(b))) << 16
          | static_cast<uint32_t>(clamp_0_255(        a )) << 24;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

using float_fn = ABI void(*)(stage*, size_t, float, float, float, float);

static void next(stage* st, size_t x, float r, float g, float b, float a) {
    auto next = reinterpret_cast<float_fn>(st->next);
    next(st+1, x, r,g,b,a);
}

static ABI void load_srgb(stage* st, size_t x, float r, float g, float b, float a) {
    auto src = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(src+x, &r,&g,&b,&a);

    next(st,x,r,g,b,a);
}

static ABI void load_f16(stage* st, size_t x, float r, float g, float b, float a) {
    auto src = static_cast<const __fp16*>(st->ctx);

    r = static_cast<float>(src[4*x+0]);
    g = static_cast<float>(src[4*x+1]);
    b = static_cast<float>(src[4*x+2]);
    a = static_cast<float>(src[4*x+3]);

    next(st,x,r,g,b,a);
}

static ABI void scale_u8(stage* st, size_t x, float r, float g, float b, float a) {
    auto cov = static_cast<const uint8_t*>(st->ctx);
    float c = cov[x] * (1/255.0f);

    r *= c;
    g *= c;
    b *= c;
    a *= c;

    next(st,x,r,g,b,a);
}

static ABI void srcover_srgb(stage* st, size_t x, float r, float g, float b, float a) {
    auto dst = static_cast<const uint32_t*>(st->ctx);
    float dr,dg,db,da;
    srgb_to_floats(dst+x, &dr,&dg,&db,&da);

    float A = 1 - a;
    r += dr * A;
    g += dg * A;
    b += db * A;
    a += da * A;

    next(st,x,r,g,b,a);
}

static ABI void lerp_u8_srgb(stage* st, size_t x, float r, float g, float b, float a) {
    auto cov = static_cast<const uint8_t*>(st->ctx);
    float c = cov[x] * (1/255.0f);

    auto dst = static_cast<const uint32_t*>(st->dtx);
    float dr,dg,db,da;
    srgb_to_floats(dst+x, &dr,&dg,&db,&da);

    r = dr + (r-dr)*c;
    g = dg + (g-dg)*c;
    b = db + (b-db)*c;
    a = da + (a-da)*c;

    next(st,x,r,g,b,a);
}

static ABI void store_srgb(stage* st, size_t x, float r, float g, float b, float a) {
    auto dst = static_cast<uint32_t*>(st->dtx);
    floats_to_srgb(dst+x, r,g,b,a);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void pipeline::add_stage(Stage st, const void* ctx, void* dtx) {
    if (float_stages.size() == 0) {
        float_stages.reserve(8);
    }

    this->add_avx2 (st, ctx, dtx);
    this->add_sse41(st, ctx, dtx);

    float_fn f = nullptr;
    switch (st) {
        case load_srgb:     f = ::load_srgb;     break;
        case load_f16:      f = ::load_f16;      break;
        case scale_u8:      f = ::scale_u8;      break;
        case srcover_srgb:  f = ::srcover_srgb;  break;
        case lerp_u8_srgb:  f = ::lerp_u8_srgb;  break;
        case store_srgb:    f = ::store_srgb;    break;
    }
    float_stages.push_back({ reinterpret_cast<void(*)(void)>(f), ctx, dtx });
}

void pipeline::ready() {
    auto rewire = [](std::vector<stage>* stages) {
        auto start = (*stages)[0].next;
        for (size_t i = 0; i < stages->size(); i++) {
            (*stages)[i].next = (*stages)[i+1].next;
        }
        (*stages)[stages->size() - 1].next = start;
    };

    rewire(& avx2_stages);
    rewire(&sse41_stages);
    rewire(&float_stages);
}

void pipeline::call(size_t n) {
    assert (float_stages.size() > 0);

    size_t x = 0;
    if (cpu::supports(cpu::AVX2 | cpu::FMA | cpu::BMI1 | cpu::F16C)) { this->call_avx2 (&x, &n); }
    if (cpu::supports(cpu::SSE41                                  )) { this->call_sse41(&x, &n); }

    while (n > 0) {
        auto start = reinterpret_cast<float_fn>(float_stages.back().next);
        start(float_stages.data(), x, 0,0,0,0);

        x += 1;
        n -= 1;
    }
}
