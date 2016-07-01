#include "pipeline.h"
#include <assert.h>
#include <stdlib.h>

static void with_pipeline(uint32_t* dp, const uint32_t* sp, const uint8_t* cp, size_t n) {
    static pipeline* p = nullptr;
    if (!p) {
        p = new pipeline;
        p->add_stage( load_d_srgb, nullptr);
        p->add_stage( load_s_srgb,    sp  );
        p->add_stage(     srcover, nullptr);
        p->add_stage(     lerp_u8,    cp  );
        p->add_stage(store_s_srgb, nullptr);
        p->ready();
    }

    p->call(dp, n);
}

static uint32_t dst[1024], src[1024];
static uint8_t cov[1024];

int main(int argc, char** argv) {
    int choice = argc > 1 ? atoi(argv[1]) : 0;

    if (choice) {
        for (int j = 0; j < 100000; j++) {
            switch (choice) {
                case 1: fused        (dst, src, cov, 1024); break;
                case 2: with_pipeline(dst, src, cov, 1024); break;
            }
        }
        return 0;
    }

    fused        (dst, src, cov, 1024);
    with_pipeline(dst, src, cov, 1024);

    return 0;
}
