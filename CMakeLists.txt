cmake_minimum_required(VERSION 3.16)
project(llm_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include src)

# 聚合所有的源文件 (Sprint 1 + Sprint 2 + Sprint 3)
add_library(llm_engine
    src/backends/cpu/reference/math_ref.cpp
    src/backends/cpu/arm_neon/pack.cpp
    src/backends/cpu/arm_neon/gemm_kernel_8x12_neon.cpp
    src/backends/cpu/arm_neon/neon_ops.cpp
    src/backends/cpu/arm_neon/neon_backend.cpp

    src/graph/graph.cpp
    src/graph/compiler.cpp
)

# 纯 ARM 环境，直接拉满极致优化
target_compile_options(llm_engine PRIVATE
    -O3
    -march=armv8.2-a+fp16 
)

enable_testing()
add_subdirectory(tests)

# 新增：挂载 Benchmark 目录
# add_subdirectory(benchmarks)