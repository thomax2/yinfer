cmake_minimum_required(VERSION 3.16)
project(llm_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include src)

# 公共源文件（非架构相关）
set(LLM_ENGINE_SOURCES
    # Sprint 1: 基础数学运算
    src/backends/cpu/reference/math_ref.cpp

    # Sprint 3: 计算图
    src/graph/graph.cpp
    src/graph/compiler.cpp

    # Sprint 4: 内存管理
    src/memory/memory_pool.cpp
    src/memory/workspace.cpp

    src/memory/kv_cache.cpp
)

# 如果是在 ARM 架构（如 aarch64），才添加 Neon 源并启用 ARM 专属优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    list(APPEND LLM_ENGINE_SOURCES
        src/backends/cpu/arm_neon/pack.cpp
        src/backends/cpu/arm_neon/gemm_kernel_8x12_neon.cpp
        src/backends/cpu/arm_neon/neon_ops.cpp
        src/backends/cpu/arm_neon/neon_backend.cpp
        src/backends/cpu/arm_neon/rms_norm_neon.cpp
        src/backends/cpu/arm_neon/rope_neon.cpp
        src/backends/cpu/arm_neon/swiglu_neon.cpp
    )
endif()

add_library(llm_engine ${LLM_ENGINE_SOURCES})

# 针对不同平台选择合适的编译选项
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    target_compile_options(llm_engine PRIVATE -O3 -march=armv8.2-a+fp16)
else()
    target_compile_options(llm_engine PRIVATE -O3)
endif()

enable_testing()
add_subdirectory(tests)

# 新增：挂载 Benchmark 目录
 add_subdirectory(benchmarks)