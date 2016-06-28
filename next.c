#include "pipeline.h"

void next(struct stage* stage, size_t n, void* dp, __m128 d, __m128 s) {
    stage->next->fn(stage->next, n, dp, d, s);
}
