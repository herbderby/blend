#pragma once

#include <stdint.h>
#include <stddef.h>
#include <vector>

struct stage {
    void (*next)(void);
    void* ctx;
};

struct pipeline {
    enum Stage { load_srgb, scale_u8, srcover_srgb };

    void add_stage(Stage, const void* ctx);
    void ready();

    void call(size_t n);

private:
    void add_f8_stage(Stage, const void* ctx);

    size_t call_f8(size_t n);

    std::vector<stage> f8_stages,
                       f4_stages,
                       f1_stages;
};
