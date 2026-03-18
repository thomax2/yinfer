#include <gtest/gtest.h>
#include <cmath>

#include "llm_engine/tensor.h"
#include "llm_engine/memory/memory_pool.h"

#include "backends/cpu/reference/math_ref.h"
#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"

using namespace llm_engine;

static void init_memory_pool() {
    static bool initialized = false;
    if (!initialized) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
        initialized = true;
    }
}


/*
在你写测试代码之前，我必须提醒你一个底层常识：千万不要用 EXPECT_EQ 去判断两个浮点矩阵是否相等！

Reference 算子：使用的是标准的 c += a * b;，这是离散的乘法和加法。

NEON 算子：你使用了 vfmaq_f32 指令。这叫融合乘加（FMA, Fused Multiply-Accumulate）。它在底层硬件里只做一次舍入（Rounding），精度实际上比纯 C++ 还要高。

同时，NEON 的 8x12 分块改变了浮点数相加的顺序。因为浮点数加法不满足结合律（即 (a+b)+c 在计算机里未必等于 a+(b+c)），所以两者的最终结果在小数点后几位一定会产生微小的偏差。
*/


TEST(NeonTest, MatmulAccuracy) {
    // 选一个既有典型性又能触发 Pack 边界的维度，比如 137x137
    // 不要只测 128 这种 8 或 12 的整数倍，要测非对齐的维度！
    init_memory_pool();

    int M = 137, N = 137, K = 137; 

    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C_ref({M, N});
    Tensor C_neon({M, N});

    // 1. 填充随机数据
    float* a_ptr = A.ptr<float>();
    float* b_ptr = B.ptr<float>();
    for (size_t i = 0; i < A.size(); ++i) a_ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < B.size(); ++i) b_ptr[i] = static_cast<float>(rand()) / RAND_MAX;

    // 2. 跑基准真值
    reference::matmul_ref(A, B, C_ref);

    // 3. 跑你手写的 NEON 加速版

    // ✅ 分配 workspace（关键）
    using namespace arm_neon;
    int mp = (M + MR - 1) / MR; // A 的块数
    int np = (N + NR - 1) / NR; // B 的块数
    size_t ws_size = (mp * MR * K + np * NR * K) * sizeof(float);
    float* workspace = (float*)g_memory_pool->allocate(ws_size);

    std::memset(workspace, 0, ws_size);

    arm_neon::matmul_neon(A, B, C_neon, workspace);

    // 4. 逐元素比对 (Element-wise Comparison)
    float* c_ref_ptr = C_ref.ptr<float>();
    float* c_neon_ptr = C_neon.ptr<float>();
    
    // 允许的误差阈值
    float epsilon = 1e-4; 

    for (size_t i = 0; i < C_ref.size(); ++i) {
        // 使用 EXPECT_NEAR 而不是 EXPECT_EQ
        EXPECT_NEAR(c_ref_ptr[i], c_neon_ptr[i], epsilon) 
            << "Matrix mismatch at flat index " << i;
    }
}