#pragma once

#include <stdint.h>

struct cpu {
    enum : uint32_t {
        SSE41 = 1 << 0,
        AVX   = 1 << 1,
        F16C  = 1 << 2,
        FMA   = 1 << 3,
        AVX2  = 1 << 4,
        BMI1  = 1 << 5,
        BMI2  = 1 << 6
    };

    static bool supports(uint32_t mask) {
        return (features & mask) == mask;
    }

    static void read_features();
    static uint32_t features;
};
