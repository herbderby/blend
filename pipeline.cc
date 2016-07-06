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

using float_fn = ABI void(*)(stage*, size_t, float, float, float, float,
                                             float, float, float, float);

static void next(stage* st, size_t x, float sr, float sg, float sb, float sa,
                                      float dr, float dg, float db, float da) {
    auto next = reinterpret_cast<float_fn>(st->next);
    next(st+1, x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void load_s_srgb(stage* st, size_t x, float sr, float sg, float sb, float sa,
                                                 float dr, float dg, float db, float da) {
    auto ptr = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(ptr+x, &sr,&sg,&sb,&sa);

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void load_d_srgb(stage* st, size_t x, float sr, float sg, float sb, float sa,
                                                 float dr, float dg, float db, float da) {
    auto ptr = static_cast<const uint32_t*>(st->ctx);
    srgb_to_floats(ptr+x, &dr,&dg,&db,&da);

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void srcover(stage* st, size_t x, float sr, float sg, float sb, float sa,
                                             float dr, float dg, float db, float da) {
    float A = 1 - sa;
    sr += dr * A;
    sg += dg * A;
    sb += db * A;
    sa += da * A;

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void scale_u8(stage* st, size_t x, float sr, float sg, float sb, float sa,
                                              float dr, float dg, float db, float da) {
    auto cov = static_cast<const uint8_t*>(st->ctx);
    float c = cov[x] * (1/255.0f);

    sr *= c;
    sg *= c;
    sb *= c;
    sa *= c;

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}


static ABI void lerp_u8(stage* st, size_t x, float sr, float sg, float sb, float sa,
                                             float dr, float dg, float db, float da) {
    auto cov = static_cast<const uint8_t*>(st->ctx);
    float c = cov[x] * (1/255.0f);

    sr = dr + (sr-dr)*c;
    sg = dg + (sg-dg)*c;
    sb = db + (sb-db)*c;
    sa = da + (sa-da)*c;

    next(st,x, sr,sg,sb,sa, dr,dg,db,da);
}

static ABI void store_srgb(stage* st, size_t x, float sr, float sg, float sb, float sa,
                                                float   , float   , float   , float   ) {
    auto ptr = static_cast<uint32_t*>(st->ctx);
    floats_to_srgb(ptr+x, sr,sg,sb,sa);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void pipeline::add_stage(Stage st, const void* const_ctx) {
    auto ctx = const_cast<void*>(const_ctx);
    if (float_stages.size() == 0) {
        float_stages.reserve(8);
    }

    if ((1) && cpu::supports(cpu::AVX2 | cpu::FMA | cpu::BMI1)) {
        this->add_avx2(st, ctx);
    } else {
        this->add_avx (st, ctx);
    }

    this->add_sse41(st, ctx);

    float_fn f = nullptr;
    switch (st) {
        case load_s_srgb: f = ::load_s_srgb;   break;
        case load_d_srgb: f = ::load_d_srgb;   break;
        case     srcover: f = ::srcover;       break;
        case    scale_u8: f = ::scale_u8;      break;
        case     lerp_u8: f = ::lerp_u8;       break;
        case  store_srgb: f = ::store_srgb;    break;
    }
    float_stages.push_back({ reinterpret_cast<void(*)(void)>(f), ctx });
}

void pipeline::ready() {
    auto rewire = [](std::vector<stage>* stages) {
        auto start = (*stages)[0].next;
        for (size_t i = 0; i < stages->size(); i++) {
            (*stages)[i].next = (*stages)[i+1].next;
        }
        (*stages)[stages->size() - 1].next = start;
    };

    rewire(&  ymm_stages);
    rewire(&  xmm_stages);
    rewire(&float_stages);
}

void pipeline::call(size_t n) {
    assert (float_stages.size() > 0);

    size_t x = 0;
    if ((1) && cpu::supports(cpu::AVX  )) { this->call_ymm(&x, &n); }
    if ((1) && cpu::supports(cpu::SSE41)) { this->call_xmm(&x, &n); }

    while (n > 0) {
        auto start = reinterpret_cast<float_fn>(float_stages.back().next);
        start(float_stages.data(), x, 0,0,0,0, 0,0,0,0);

        x += 1;
        n -= 1;
    }
}
