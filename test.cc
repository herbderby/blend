#include "pipeline.h"
#include <stdlib.h>

static void with_pipeline(uint32_t* dp, const uint32_t* sp, const uint8_t* cp, size_t n) {
    pipeline p;
    p.add_stage(pipeline::   load_srgb, sp);
    p.add_stage(pipeline::    scale_u8, cp);
    p.add_stage(pipeline::srcover_srgb, dp);
    p.ready();

    p.call(n);
}

static uint32_t dst[1023], src[1023];
static uint8_t cov[1023];

int main(int argc, char** argv) {
    int choice = argc > 1 ? atoi(argv[1]) : 0;

    if (choice) {
        for (int j = 0; j < 100000; j++) {
            switch (choice) {
                case 1: with_pipeline(dst, src, cov, 1023); break;
            }
        }
        return 0;
    }

    with_pipeline(dst, src, cov, 1023);

    return 0;
}
