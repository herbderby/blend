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
        { (void*)1,   load_d_srgb, NULL },
        { (void*)2,   load_s_srgb, NULL },
        { (void*)3,       srcover,   sp },
        { (void*)4,       lerp_u8, NULL },
        { (void*)5,  store_s_srgb,   cp },
        { (void*)0,      done_yet, NULL },
    };
    size_t nstages = sizeof(stages)/sizeof(*stages);
    wire(stages, nstages);

    done_yet(&stages[0],
             (size_t)n, dp, _mm_setzero_ps(), _mm_setzero_ps());
}

static void fastest_pipeline(int* dp, const int* sp, const char* cp, int n) {
    assert (n > 0);
    struct stage stages[] = {
        { (void*)1,                  load_d_srgb, NULL },  // start
        { (void*)2,                  load_s_srgb, NULL },  // d loaded

        { (void*)3,                      srcover,   sp },  // d+s loaded
        { (void*)4,                      lerp_u8, NULL },  // s = srcover(d,s)
        { (void*)5, store_s_done_yet_load_d_srgb,   cp },  // s = lerp(d,s,cov)
        { (void*)2,                  load_s_srgb, NULL },  // d loaded
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
                case 3:  fastest_pipeline(dst, src, cov, 1024); break;
            }
        }
        return 0;
    }

    fused            (dst, src, cov, 1024);
    simplest_pipeline(dst, src, cov, 1024);
    fastest_pipeline (dst, src, cov, 1024);

    return 0;
}
