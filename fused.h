#pragma once

#include <stddef.h>
#include <stdint.h>

void fused(uint32_t* dst, const uint32_t* src, const uint8_t* cov, size_t n);
