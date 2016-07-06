#include "pipeline.h"
#include <assert.h>
#include <immintrin.h>

void pipeline::call_xmm(size_t* x, size_t* n) {
    assert (xmm_stages.size() > 0);

    f4 u = _mm_undefined_ps();
    auto start = reinterpret_cast<xmm_fn>(xmm_stages.back().next);
    while (*n >= 4) {
        start(xmm_stages.data(), *x, u,u,u,u, u,u,u,u);
        *x += 4;
        *n -= 4;
    }
}
