#pragma once

#include "abi.h"
#include <stddef.h>
#include <stdint.h>
#include <vector>

struct stage {
    void (*next)(void);
    void* ctx;
};

struct pipeline {
    enum Stage {
        load_s_srgb,
        load_d_srgb,
            srcover,
           scale_u8,
            lerp_u8,
         store_srgb
    };
    void add_stage(Stage, const void* ctx);
    void ready();

    void call(size_t n);

private:
    void add_avx2 (Stage, void* ctx);
    void add_avx  (Stage, void* ctx);
    void add_sse41(Stage, void* ctx);

    void call_ymm(size_t* x, size_t* n);
    void call_xmm(size_t* x, size_t* n);

    std::vector<stage> ymm_stages,
                       xmm_stages,
                     float_stages;
};

using f4 = float __attribute__((vector_size(16)));
using f8 = float __attribute__((vector_size(32)));

using xmm_fn = ABI void(*)(stage*, size_t, f4,f4,f4,f4, f4,f4,f4,f4);
using ymm_fn = ABI void(*)(stage*, size_t, f8,f8,f8,f8, f8,f8,f8,f8);

static inline void next(stage* st, size_t x, f4 sr, f4 sg, f4 sb, f4 sa,
                                             f4 dr, f4 dg, f4 db, f4 da) {
    auto next = reinterpret_cast<xmm_fn>(st->next);
    next(st+1, x, sr,sg,sb,sa, dr,dg,db,da);
}

static inline void next(stage* st, size_t x, f8 sr, f8 sg, f8 sb, f8 sa,
                                             f8 dr, f8 dg, f8 db, f8 da) {
    auto next = reinterpret_cast<ymm_fn>(st->next);
    next(st+1, x, sr,sg,sb,sa, dr,dg,db,da);
}

