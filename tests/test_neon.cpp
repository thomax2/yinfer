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

// =========================================================================
// 新增测试：RMSNorm
// =========================================================================
TEST(NeonTest, RMSNormAccuracy) {
    // 故意选一个非 4 的倍数，测试你的尾部 (Tail case) 处理逻辑
    int n = 4093; 
    float eps = 1e-5f;

    Tensor X({1, n});
    Tensor W({1, n});
    Tensor Y_ref({1, n});
    Tensor Y_neon({1, n});

    float* x_ptr = X.ptr<float>();
    float* w_ptr = W.ptr<float>();
    float* y_ref_ptr = Y_ref.ptr<float>();
    float* y_neon_ptr = Y_neon.ptr<float>();

    // 1. 填充随机数据
    for (int i = 0; i < n; ++i) {
        x_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f; // [-1.0, 1.0]
        w_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f + 0.9f; // 权重通常在 1.0 附近
    }

    // 2. 跑 C++ 标准基准真值 (Reference)
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x_ptr[i] * x_ptr[i];
    }
    float mean = sum_sq / n;
    float scale = 1.0f / std::sqrt(mean + eps);
    for (int i = 0; i < n; ++i) {
        y_ref_ptr[i] = x_ptr[i] * scale * w_ptr[i];
    }

    // 3. 跑你写的 NEON 算子
    arm_neon::rmsnorm_neon(x_ptr, w_ptr, y_neon_ptr, n, eps);

    // 4. 对比误差 (RMSNorm 的标准 C++ 与 NEON 精度高度一致)
    float epsilon = 1e-4;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y_ref_ptr[i], y_neon_ptr[i], epsilon) 
            << "RMSNorm mismatch at index " << i;
    }
}

// =========================================================================
// 新增测试：SwiGLU
// =========================================================================
TEST(NeonTest, SwiGLUAccuracy) {
    int n = 4095; // 非 4 的倍数

    Tensor X({1, n});
    Tensor Up({1, n});
    Tensor Y_ref({1, n});
    Tensor Y_neon({1, n});

    float* x_ptr = X.ptr<float>();
    float* up_ptr = Up.ptr<float>();
    float* y_ref_ptr = Y_ref.ptr<float>();
    float* y_neon_ptr = Y_neon.ptr<float>();

    // 1. 填充数据 (包含正负数，因为 Swish 对负数很敏感)
    for (int i = 0; i < n; ++i) {
        x_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f; // [-5.0, 5.0]
        up_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    // 2. 跑 C++ 标准基准真值
    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-x_ptr[i]));
        y_ref_ptr[i] = x_ptr[i] * sig * up_ptr[i];
    }

    // 3. 跑 NEON 算子
    arm_neon::swiglu_neon(x_ptr, up_ptr, y_neon_ptr, n);

    // 4. 对比误差 
    // 注意：因为我们用了工业级的近似 exp，所以和标准库的 std::exp 会有极微小差异。
    // 工业上允许这种逼近误差，阈值设为 1e-3 是非常安全的。
    float epsilon = 1e-3;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y_ref_ptr[i], y_neon_ptr[i], epsilon) 
            << "SwiGLU mismatch at index " << i << ". X: " << x_ptr[i];
    }
}

// =========================================================================
// 新增测试：RoPE (旋转位置编码)
// =========================================================================
TEST(NeonTest, RoPEAccuracy) {
    int n = 128; // RoPE 通常处理 head_dim，一般是 64 或 128，必须是 4 的倍数

    // RoPE 是 In-place (原地) 修改数据的，所以我们需要两份 X
    Tensor X_ref({1, n});
    Tensor X_neon({1, n});
    Tensor Cos({1, n});
    Tensor Sin({1, n});

    float* x_ref_ptr = X_ref.ptr<float>();
    float* x_neon_ptr = X_neon.ptr<float>();
    float* cos_ptr = Cos.ptr<float>();
    float* sin_ptr = Sin.ptr<float>();

    // 1. 填充数据
    for (int i = 0; i < n; ++i) {
        float val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        x_ref_ptr[i] = val;
        x_neon_ptr[i] = val; // 保持一致
    }

    // 填充 Cos/Sin 表。注意内存排布必须是相邻重复：[c0, c0, c1, c1...]
    for (int i = 0; i < n; i += 2) {
        float theta = (static_cast<float>(rand()) / RAND_MAX) * 3.14159f;
        float c = std::cos(theta);
        float s = std::sin(theta);
        cos_ptr[i] = c; cos_ptr[i+1] = c;
        sin_ptr[i] = s; sin_ptr[i+1] = s;
    }

    // 2. 跑 C++ 标准基准真值 (复数乘法展开)
    for (int i = 0; i < n; i += 2) {
        float x0 = x_ref_ptr[i];
        float x1 = x_ref_ptr[i+1];
        float c = cos_ptr[i]; 
        float s = sin_ptr[i];
        x_ref_ptr[i]   = x0 * c - x1 * s;
        x_ref_ptr[i+1] = x1 * c + x0 * s;
    }

    // 3. 跑 NEON 算子
    arm_neon::rope_neon(x_neon_ptr, cos_ptr, sin_ptr, n);

    // 4. 对比误差
    float epsilon = 1e-4;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x_ref_ptr[i], x_neon_ptr[i], epsilon) 
            << "RoPE mismatch at index " << i;
    }
}