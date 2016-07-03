#pragma once
#include <stddef.h>

// What registers can we use to tail call arguments through a pipeline?
//
//    x86-64 System V: rdi,rsi,rdx,rcx,r8,r9, xmm0-xmm7
//
//    __vectorcall 32-bit: ecx,edx, xmm0-xmm5
//
//    __vectorcall 64-bit: rcx/xmm0, rdx/xmm1, r8/xmm2, r9/xmm3, xmm4, xmm5
//
//    AAPCS:   r0-r3, q0-q3
//    AAPCS64: r0-r7, v0-v7

#if 0
    #define ABI __attribute__((vectorcall))
#elif 0
    #define ABI __attribute__((sysv_abi))
#else
    #define ABI
#endif

using f1 = float;
using f4 = float __attribute__((__vector_size__(16)));
using f8 = float __attribute__((__vector_size__(32)));

struct stage;
using f1_fn = ABI void(*)(stage*, size_t, f1, f1, f1, f1);
using f4_fn = ABI void(*)(stage*, size_t, f4, f4, f4, f4);
using f8_fn = ABI void(*)(stage*, size_t, f8, f8, f8, f8);

#define EXPORT_F1(name)                                               \
  static ABI void name(stage* st, size_t x, f1 r, f1 g, f1 b, f1 a) { \
      if (!name(st->ctx, x, &r,&g,&b,&a)) {                           \
          auto next = reinterpret_cast<f1_fn>(st->next);              \
          next(st+1, x, r,g,b,a);                                     \
      }                                                               \
  }

#define EXPORT_F4(name)                                               \
  static ABI void name(stage* st, size_t x, f4 r, f4 g, f4 b, f4 a) { \
      if (!name(st->ctx, x, &r,&g,&b,&a)) {                           \
          auto next = reinterpret_cast<f4_fn>(st->next);              \
          next(st+1, x, r,g,b,a);                                     \
      }                                                               \
  }

#define EXPORT_F8(name)                                               \
  static ABI void name(stage* st, size_t x, f8 r, f8 g, f8 b, f8 a) { \
      if (!name(st->ctx, x, &r,&g,&b,&a)) {                           \
          auto next = reinterpret_cast<f8_fn>(st->next);              \
          next(st+1, x, r,g,b,a);                                     \
      }                                                               \
  }
