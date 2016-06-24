#include "fused.h"
#include "pipeline.h"
#include <stdlib.h>

static int dst[1024], src[1024];
static char cov[1024];

int main(int argc, char** argv) {
    int n = 1;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    for (int j = 0; j < n; j++) {
        fused(dst, src, cov, 1024);
    }

    for (int j = 0; j < n; j++) {
        struct stage stages[] = {
            { NULL, load_srgb_dst,  dst },
            { NULL, load_srgb_src,  src },
            { NULL,       srcover, NULL },
            { NULL,   lerp_a8_cov,  cov },
            { NULL,    store_srgb,  dst },
        };

        for (int i = 0; i < 4; i++) {
            stages[i].next = &stages[i+1];
        }
        stages[4].next = &stages[1];

        stages[0].fn(&stages[0], 1023,_mm_setzero_ps(), _mm_setzero_ps());
    }

    return 0;
}
