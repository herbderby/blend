#include "cpu.h"
#include "pipeline.h"
#include <stdlib.h>

static void srcover(uint32_t* dp, const uint32_t* sp, size_t n) {
    pipeline p;
    p.add_stage(pipeline::   load_srgb, sp,      nullptr);
    p.add_stage(pipeline::srcover_srgb, nullptr, dp);
    p.ready();

    p.call(n);
}

static void srcover_mask(uint32_t* dp, const uint32_t* sp, const uint8_t* cp, size_t n) {
    pipeline p;
    p.add_stage(pipeline::   load_srgb, sp,      nullptr);
    p.add_stage(pipeline::    scale_u8, cp,      nullptr);
    p.add_stage(pipeline::srcover_srgb, nullptr, dp);
    p.ready();

    p.call(n);
}

static void src_mask(uint32_t* dp, const uint32_t* sp, const uint8_t* cp, size_t n) {
    pipeline p;
    p.add_stage(pipeline::    load_srgb, sp, nullptr);
    p.add_stage(pipeline::store_u8_srgb, cp, dp);
    p.ready();

    p.call(n);
}

static uint32_t dst[1023], src[1023];
static uint8_t cov[1023];

int main(int argc, char** argv) {
    cpu::read_features();

    int choice = argc > 1 ? atoi(argv[1]) : 0;

    for (int j = 0; j < 100000; j++) {
        switch(choice) {
            case 1: srcover     (dst, src,      1023); break;
            case 2: srcover_mask(dst, src, cov, 1023); break;
            case 3:     src_mask(dst, src, cov, 1023); break;
        }
    }

    srcover     (dst, src,      1023);
    srcover_mask(dst, src, cov, 1023);
        src_mask(dst, src, cov, 1023);

    return 0;
}
