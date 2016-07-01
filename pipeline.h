#pragma once

#include <stdint.h>
#include <stddef.h>
#include <vector>

void fused(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n);

struct pipeline {
    enum Stage {
        load_d_srgb,
        load_s_srgb,
        srcover,
        lerp_u8,
        store_s_srgb,
    };
    void add_stage(Stage, const void* ctx);
    void ready();

    void call(void* dp, size_t n) const;

    struct stage {
        void (*next)(void);
        const void* ctx;
    };

private:
    void add_avx2_stage(Stage, const void* ctx);

    std::vector<stage> narrow_stages,
                         wide_stages,
                         avx2_stages;
};
