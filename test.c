#include "fused.h"
#include "pipeline.h"
#include <assert.h>
#include <stdlib.h>

static void simplest_pipeline(int* dp, const int* sp, const char* cp, int n) {
    struct stage stages[] = {
        {  load_d_srgb,       NULL },  // done_yet()
        {  load_s_srgb,       NULL },  // load_d
        {      srcover,         sp },  // load_s
        {      lerp_u8,       NULL },  // srcover
        { store_s_srgb,         cp },  // lerp_u8
        {         loop,       NULL },  // store_s_srgb
        {     done_yet,  &stages[0]},  // loop
    };

    done_yet(&stages[0],
             (size_t)n, dp, _mm_setzero_ps(), _mm_setzero_ps());
}

static int dst[1024], src[1024];
static char cov[1024];

int main(int argc, char** argv) {
    int choice = argc > 1 ? atoi(argv[1]) : 0;

    if (choice) {
        for (int j = 0; j < 100000; j++) {
            switch (choice) {
                case 1:             fused(dst, src, cov, 1024); break;
                case 2: simplest_pipeline(dst, src, cov, 1024); break;
            }
        }
        return 0;
    }

    fused            (dst, src, cov, 1024);
    simplest_pipeline(dst, src, cov, 1024);

    return 0;
}
