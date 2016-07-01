#pragma once

#if 0
    #define ABI __attribute__((vectorcall))
#elif 0
    #define ABI __attribute__((sysv_abi))
#else
    #define ABI
#endif
