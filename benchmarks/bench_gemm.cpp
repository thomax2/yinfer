#include <benchmark/benchmark.h>
#include <random>
#include <cstring>
#include <vector> // 引入 vector 管理临时内存

#include "llm_engine/tensor.h"
#include "llm_engine/memory/memory_pool.h"
#include "backends/cpu/reference/math_ref.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"

using namespace llm_engine;

static void init_memory_pool() {
    static bool initialized = false;
    if (!initialized) {
        // 分配 512MB 内存池
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
        initialized = true;
    }
}

void fill_random(Tensor& t) {
    float* ptr = t.ptr<float>();
    size_t num = t.size();
    for (size_t i = 0; i < num; ++i) {
        ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

static void BM_matmul_ref(benchmark::State& state) {
    init_memory_pool(); // ✅ 补上救命的初始化

    int N = state.range(0);
    Tensor A({N, N});
    Tensor B({N, N});
    Tensor C({N, N});

    fill_random(A);
    fill_random(B);

    for (auto _ : state) {
        reference::matmul_ref(A, B, C);
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * N * N * N,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000 
    );
}

static void BM_matmul_neon(benchmark::State& state) {
    init_memory_pool(); 

    int N = state.range(0);
    Tensor A({N, N});
    Tensor B({N, N});
    Tensor C({N, N});

    fill_random(A);
    fill_random(B);

    using namespace arm_neon;
    int mp = (N + MR - 1) / MR; 
    int np = (N + NR - 1) / NR; 
    size_t num_elements = mp * MR * N + np * NR * N;
    
    // ✅ 使用 std::vector 安全管理测试时的生命周期，防止撑爆自定义 MemoryPool
    std::vector<float> workspace_vec(num_elements, 0.0f);
    float* workspace = workspace_vec.data();

    for (auto _ : state) {
        arm_neon::matmul_neon(A, B, C, workspace, false, nullptr);
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * N * N * N,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000 
    );
}

BENCHMARK(BM_matmul_ref)->Arg(128)->Arg(512)->Arg(1024);
BENCHMARK(BM_matmul_neon)->Arg(128)->Arg(512)->Arg(1024);

BENCHMARK_MAIN();