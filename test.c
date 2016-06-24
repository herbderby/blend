#include "fused.h"
#include "pipeline.h"
#include <stdlib.h>

static int dst[1024], src[1024];
static char cov[1024];

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1;

    for (int j = 0; j < n; j++) {
        fused(dst, src, cov, 1024);
    }

    for (int j = 0; j < n; j++) {
        struct stage stages[] = {
            { (void*)1,         load_d_srgb, NULL },
            { (void*)2,         load_s_srgb,  src },
            { (void*)3,             srcover, NULL },
            { (void*)4,             lerp_u8,  cov },
            { (void*)1, store_s_load_d_srgb, NULL },
        };

        for (size_t i = 0; i < sizeof(stages) / sizeof(*stages); i++) {
            size_t next = (size_t)stages[i].next;
            stages[i].next = &stages[next];
        }

        stages[0].fn(&stages[0], 1023, dst, _mm_setzero_ps(), _mm_setzero_ps());
    }

    return 0;
}
