#include "fused.h"
#include "pipeline.h"
#include <assert.h>
#include <stdlib.h>

static void pipeline(int* dp, const int* sp, const char* cp, size_t n) {
    stage_fn* start = load_d_srgb;
    struct stage stages[] = {
        {  load_s_srgb, NULL },  // load_d_srgb
        {      srcover,  sp  },  // load_s_srgb
        {      lerp_u8, NULL },  // srcover
        { store_s_srgb,  cp  },  // lerp_u8
        {         NULL, NULL },  // store_s_srgb
    };

    run_pipeline(stages, start, dp, n);
}

static int dst[1024], src[1024];
static char cov[1024];

int main(int argc, char** argv) {
    int choice = argc > 1 ? atoi(argv[1]) : 0;

    if (choice) {
        for (int j = 0; j < 100000; j++) {
            switch (choice) {
                case 1: fused   (dst, src, cov, 1024); break;
                case 2: pipeline(dst, src, cov, 1024); break;
            }
        }
        return 0;
    }

    fused   (dst, src, cov, 1024);
    pipeline(dst, src, cov, 1024);

    return 0;
}
