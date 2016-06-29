#include "fused.h"
#include "pipeline.h"
#include <assert.h>
#include <stdlib.h>

static void go(const struct stage* start, size_t n, void* dp) {
    done_yet(start, n, dp, _mm_setzero_ps(), _mm_setzero_ps());
}

static void simple(int* dp, const int* sp, const char* cp, size_t n) {
    struct stage stages[] = {
        {  load_d_srgb, &stages[1] },  // done_yet

        {  load_s_srgb,       NULL },  // load_d
        {      srcover,         sp },  // load_s
        {      lerp_u8,       NULL },  // srcover
        { store_s_srgb,         cp },  // lerp_u8
        {     done_yet,       NULL },  // store_s_srgb
        {  load_d_srgb, &stages[1] },  // done_yet
    };
    go(stages, n, dp);
}

static void faster(int* dp, const int* sp, const char* cp, size_t n) {
    struct stage stages[] = {
        {  load_d_srgb, &stages[1] },  // done_yet
        {  load_s_srgb,       NULL },  // load_d

        {      srcover,         sp },  // load_s
        {      lerp_u8,       NULL },  // srcover
        {        super,         cp },  // lerp_u8
        {  load_s_srgb, &stages[2] },  // super
    };
    go(stages, n, dp);
}

static int dst[1024], src[1024];
static char cov[1024];

int main(int argc, char** argv) {
    int choice = argc > 1 ? atoi(argv[1]) : 0;

    if (choice) {
        for (int j = 0; j < 100000; j++) {
            switch (choice) {
                case 1: fused (dst, src, cov, 1024); break;
                case 2: simple(dst, src, cov, 1024); break;
                case 3: faster(dst, src, cov, 1024); break;
            }
        }
        return 0;
    }

    fused (dst, src, cov, 1024);
    simple(dst, src, cov, 1024);
    faster(dst, src, cov, 1024);

    return 0;
}
