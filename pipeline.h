#pragma once

#include <stdint.h>
#include <vector>

void fused(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n);

enum class Stage {
    shortcircuit_srcover_both_srgb,
    load_d_srgb,
    load_s_srgb,
    srcover,
    lerp_u8,
    store_s_srgb,
};

struct pipeline {
    pipeline();
    ~pipeline();

    void add_stage(Stage, const void* ctx);
    void ready();

    void call(void* dp, size_t n, bool use_float_stages) const;

    struct stage_f;
    std::unique_ptr<std::vector<stage_f>> stages_f;

    struct stage_q15;
    std::unique_ptr<std::vector<stage_q15>> stages_q15;
};
