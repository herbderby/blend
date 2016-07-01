#pragma once

#include <immintrin.h>

static inline __m128 better_cvtsi32_ss(int x) {
    // Using movd + cvtdq2ps instead of cvtsi2ss breaks the dependency on the
    // previous top three lanes of whatever scratch register this uses,
    // allowing much better out-of-order execution.
    return _mm_cvtepi32_ps(_mm_cvtsi32_si128(x));
}
