#pragma once

#include <stdint.h>
#include <stddef.h>
#include <vector>

struct stage {
    void (*next)(void);
    const void* ctx;
    void* dtx;
};

struct pipeline {
    enum Stage { load_srgb, scale_u8, srcover_srgb, lerp_u8_srgb, store_srgb };

    void add_stage(Stage, const void* ctx, void* dtx);
    void ready();

    void call(size_t n);

private:
    void add_avx2 (Stage, const void* ctx, void* dtx);
    void add_sse41(Stage, const void* ctx, void* dtx);

    void call_avx2 (size_t* x, size_t* n);
    void call_sse41(size_t* x, size_t* n);

    std::vector<stage> avx2_stages,
                      sse41_stages,
                      float_stages;
};
