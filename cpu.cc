#include "cpu.h"
#include <cpuid.h>

static uint32_t cpu_features() {
    auto xgetbv = [](uint32_t xcr) {
        uint32_t eax, edx;
        __asm__ __volatile__("xgetbv": "=a"(eax), "=d"(edx) : "c"(xcr));
        return static_cast<uint64_t>(edx) << 32 | static_cast<uint64_t>(eax);
    };

    uint32_t features = 0;
    uint32_t eax,ebx,ecx,edx;

    __cpuid(1, eax, ebx, ecx, edx);

    if (ecx & (1<<19)) { features |= cpu::SSE41; }

    if ((ecx & (3<<26)) == (3<<26) && (xgetbv(0) & 6) == 6) {
        if (ecx & (1<<28)) { features |= cpu:: AVX; }
        if (ecx & (1<<29)) { features |= cpu::F16C; }
        if (ecx & (1<<12)) { features |= cpu:: FMA; }

        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        if (ebx & (1<< 5)) { features |= cpu::AVX2; }
        if (ebx & (1<< 3)) { features |= cpu::BMI1; }
        if (ebx & (1<< 8)) { features |= cpu::BMI2; }
    }

    return features;
}

bool cpu::supports(uint32_t mask) {
    static uint32_t features = cpu_features();
    return (features & mask) == mask;
}
