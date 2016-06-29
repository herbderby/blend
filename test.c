#include "fused.h"
#include "pipeline.h"
#include <assert.h>
#include <stdlib.h>

static void simple(int* dp, const int* sp, const char* cp, size_t n) {
    stage_fn* start = done_yet;
    struct stage stages[] = {
        {  load_d_srgb, &stages[1] },  // done_yet

        {  load_s_srgb,       NULL },  // load_d
        {      srcover,         sp },  // load_s
        {      lerp_u8,       NULL },  // srcover
        { store_s_srgb,         cp },  // lerp_u8
        {     done_yet,       NULL },  // store_s_srgb
        {  load_d_srgb, &stages[1] },  // done_yet
    };
    start(stages, n, dp, _mm_setzero_ps(), _mm_setzero_ps());
}

static void safe(int* dp, const int* sp, const char* cp, size_t n) {
    stage_fn* start = load_d_srgb;
    struct stage stages[] = {
        {  load_s_srgb, NULL },  // load_d_srgb
        {      srcover,  sp  },  // load_s_srgb
        {      lerp_u8, NULL },  // srcover
        { store_s_srgb,  cp  },  // lerp_u8
        {     just_ret, NULL },  // store_s_srgb
        {         NULL, NULL },  // just_ret
    };

    for (size_t i = 0; i < n; i++) {
        start(stages, i, dp, _mm_setzero_ps(), _mm_setzero_ps());
    }
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
                case 3: safe  (dst, src, cov, 1024); break;
            }
        }
        return 0;
    }

    fused (dst, src, cov, 1024);
    simple(dst, src, cov, 1024);
    safe  (dst, src, cov, 1024);

    return 0;
}
