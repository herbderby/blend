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
            { NULL,       check_n, NULL },
            { NULL, load_srgb_dst, NULL },
            { NULL, load_srgb_src,  src },
            { NULL,       srcover, NULL },
            { NULL,   lerp_a8_cov,  cov },
            { NULL,    store_srgb, NULL },
        };

        for (int i = 0; i < 5; i++) {
            stages[i].next = &stages[i+1];
        }
        stages[5].next = &stages[0];

        stages[0].fn(&stages[0], dst, _mm_setzero_ps(), _mm_setzero_ps(), 1024);
    }

    return 0;
}
