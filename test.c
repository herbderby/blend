#include "fused.h"
#include "pipeline.h"
#include <assert.h>
#include <stdlib.h>

static void wire(struct stage* stages, size_t nstages) {
    for (size_t i = 0; i < nstages; i++) {
        size_t next = (size_t)stages[i].next;
        stages[i].next = &stages[next];
    }
}

static void simplest_pipeline(int* dp, const int* sp, const char* cp, int n) {
    struct stage stages[] = {
        {  load_d_srgb,   (void*)1,   NULL },
        {  load_s_srgb,   (void*)2,   NULL },
        {      srcover,   (void*)3,     sp },
        {      lerp_u8,   (void*)4,   NULL },
        { store_s_srgb,   (void*)5,     cp },
        {     done_yet,   (void*)0,   NULL },
    };
    size_t nstages = sizeof(stages)/sizeof(*stages);
    wire(stages, nstages);

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
