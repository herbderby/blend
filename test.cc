#include "pipeline.h"
#include "sse.h"
#include <assert.h>
#include <stdlib.h>

static void with_pipeline(uint32_t* dp, const uint32_t* sp, const uint8_t* cp, size_t n,
                          bool use_float_stages) {
    pipeline p;
    p.add_stage(Stage:: load_d_srgb, nullptr);
    p.add_stage(Stage:: load_s_srgb,    sp  );
    p.add_stage(Stage::     srcover, nullptr);
    p.add_stage(Stage::     lerp_u8,    cp  );
    p.add_stage(Stage::store_s_srgb, nullptr);
    p.ready();

    p.call(dp, n, use_float_stages);
}

static uint32_t dst[1024], src[1024];
static uint8_t cov[1024];

int main(int argc, char** argv) {
    int choice = argc > 1 ? atoi(argv[1]) : 0;

    if (choice) {
        for (int j = 0; j < 100000; j++) {
            switch (choice) {
                case 1: fused        (dst, src, cov, 1024       ); break;
                case 2: with_pipeline(dst, src, cov, 1024,  true); break;
                case 3: with_pipeline(dst, src, cov, 1024, false); break;
            }
        }
        return 0;
    }

    fused        (dst, src, cov, 1024       );
    with_pipeline(dst, src, cov, 1024,  true);
    with_pipeline(dst, src, cov, 1024, false);

    auto test_mul_q15 = [](int16_t x, int16_t y, int16_t xy) {
        int16_t actual = static_cast<int16_t>(_mm_extract_epi16(mul_q15(_mm_set1_epi16(x),
                                                                        _mm_set1_epi16(y)), 0));
        printf("%d %d\n", xy, actual);
        assert (xy == actual);
    };

    for (int i = 0; i <= 0x8000; i++) {
        auto x = static_cast<int16_t>(-i);

        test_mul_q15(0, x, 0);
        test_mul_q15(x, 0, 0);

        test_mul_q15(-0x8000, x, x);
        test_mul_q15(x, -0x8000, x);
    }

    test_mul_q15(-0x8000, -0x8000, -0x8000);
    test_mul_q15(-0x4000, -0x4000, -0x2000);


    return 0;
}
