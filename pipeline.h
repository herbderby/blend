#pragma once

#include <stdint.h>
#include <stddef.h>
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
    void add_sse41(Stage, void* ctx);

    void call_avx2 (size_t* x, size_t* n);
    void call_sse41(size_t* x, size_t* n);

    std::vector<stage> avx2_stages,
                      sse41_stages,
                      float_stages;
};
