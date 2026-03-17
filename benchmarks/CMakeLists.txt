# 1. 直接在系统中寻找已经安装好的 benchmark 库
find_package(benchmark REQUIRED)

# 2. 编译你的跑分程序
add_executable(bench_gemm bench_gemm.cpp)

# 3. 链接库
# 注意：find_package 找到的 benchmark 目标名称通常是 benchmark::benchmark
target_link_libraries(bench_gemm PRIVATE 
    llm_engine 
    benchmark::benchmark 
    pthread
)