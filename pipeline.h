#pragma once

#include <stdint.h>
#include <memory>

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

    void call(void* dp, size_t n, bool use_q15_stages) const;

    struct Impl;
    std::unique_ptr<Impl> impl;
};
