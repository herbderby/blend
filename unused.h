#pragma once

#define UNUSED3(n) __attribute__((unused)) unused##n
#define UNUSED2(n) UNUSED3(n)
#define UNUSED UNUSED2(__COUNTER__)
