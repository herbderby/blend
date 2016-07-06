cmake_minimum_required(VERSION 3.5)
project(blend)

add_compile_options(-std=c++11 -O2 -march=native -momit-leaf-frame-pointer -ffp-contract=fast
        -Werror)

set(SOURCE_FILES
    cpu.cc
    cpu.h
    pipeline.cc
    pipeline.h
    pipeline_sse41.cc
    pipeline_avx2.cc
    srgb.cc
    srgb.h
    test.cc)

set_property(SOURCE pipeline_sse41.cc APPEND PROPERTY COMPILE_FLAGS -msse4.1)
set_property(SOURCE pipeline_avx2.cc APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx2 -mfma -mbmi ")
set_property(SOURCE pipeline_avx2.cc APPEND_STRING PROPERTY COMPILE_FLAGS " -ffp-contract=fast ")


add_executable(blend ${SOURCE_FILES})
