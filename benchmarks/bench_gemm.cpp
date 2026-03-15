#include <benchmark/benchmark.h>
#include <random>

// 注意根据你实际的路径修改 include
#include "llm_engine/tensor.h"
#include "backends/cpu/reference/math_ref.h"
#include "backends/cpu/arm_neon/neon_ops.h"

using namespace llm_engine;

// 核心防御：填充随机数，模拟真实的大模型激活值与权重，防止 CPU 零值优化作弊
void fill_random(Tensor& t) {
    float* ptr = t.ptr<float>();
    size_t num = t.size();
    for (size_t i = 0; i < num; ++i) {
        // 生成 0.0 ~ 1.0 的随机浮点数
        ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

static void BM_matmul_ref(benchmark::State& state) {
    int N = state.range(0);
    Tensor A({N, N});
    Tensor B({N, N});
    Tensor C({N, N});

    fill_random(A);
    fill_random(B);

    for (auto _ : state) {
        reference::matmul_ref(A, B, C);
    }
}

static void BM_matmul_neon(benchmark::State& state) {
    int N = state.range(0);
    Tensor A({N, N});
    Tensor B({N, N});
    Tensor C({N, N});

    fill_random(A);
    fill_random(B);

    for (auto _ : state) {
        // 调用你的汇编级优化算子
        arm_neon::matmul_neon(A, B, C);
    }
}

// 定义测试矩阵的维度：128x128, 512x512, 1024x1024
BENCHMARK(BM_matmul_ref)->Arg(128)->Arg(512)->Arg(1024);
BENCHMARK(BM_matmul_neon)->Arg(128)->Arg(512)->Arg(1024);

BENCHMARK_MAIN();