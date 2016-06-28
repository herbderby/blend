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
