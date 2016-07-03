#pragma once

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

#define EXPORT_NARROW_XMM(name)                                       \
  static ABI void name(const pipeline::stage* st, size_t x, void* dp, \
                       __m128 d, __m128 s) {                          \
      if (!name(st->ctx, x,dp,&d,&s)) {                               \
          auto next = reinterpret_cast<narrow_xmm>(st->next);         \
          next(st+1, x,dp,d,s);                                       \
      }                                                               \
  }

#define EXPORT_WIDE_XMM(name)                                         \
  static ABI void name(const pipeline::stage* st, size_t x, void* dp, \
                       __m128 dr, __m128 dg, __m128 db, __m128 da,    \
                       __m128 sr, __m128 sg, __m128 sb, __m128 sa) {  \
      if (!name(st->ctx, x,dp, &dr,&dg,&db,&da, &sr,&sg,&sb,&sa)) {   \
          auto next = reinterpret_cast<wide_xmm>(st->next);           \
          next(st+1, x,dp, dr,dg,db,da, sr,sg,sb,sa);                 \
      }                                                               \
  }

#define EXPORT_WIDE_YMM(name)                                         \
  static ABI void name(const pipeline::stage* st, size_t x, void* dp, \
                       __m256 dr, __m256 dg, __m256 db, __m256 da,    \
                       __m256 sr, __m256 sg, __m256 sb, __m256 sa) {  \
      if (!name(st->ctx, x,dp, &dr,&dg,&db,&da, &sr,&sg,&sb,&sa)) {   \
          auto next = reinterpret_cast<wide_ymm>(st->next);           \
          next(st+1, x,dp, dr,dg,db,da, sr,sg,sb,sa);                 \
      }                                                               \
  }
