#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>  // for rand(), RAND_MAX

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
⚠️ 重要提醒：不要用 EXPECT_EQ 判断浮点数相等！
- Reference: 离散乘加，多次舍入
- NEON: vfmaq_f32 是 FMA 指令，单次舍入，精度更高
- 分块改变加法顺序 → 浮点结合律不满足 → 微小偏差正常
*/

// ============================================================
// 🔥 测试 1: Matmul 精度验证
// ============================================================
TEST(NeonTest, MatmulAccuracy) {
    init_memory_pool();
    srand(42);  // 固定随机种子，保证可复现

    // 选非对齐维度，触发 Pack 边界处理
    int M = 137, N = 137, K = 137; 

    Tensor A({M, K});
    Tensor B({K, N});
    Tensor C_ref({M, N});
    Tensor C_neon({M, N});

    // 🔥【关键修复】延迟分配后，必须手动确保内存已分配
    A.ensure_allocated();
    B.ensure_allocated();
    C_ref.ensure_allocated();
    C_neon.ensure_allocated();

    // 1. 填充随机数据
    float* a_ptr = A.ptr<float>();
    float* b_ptr = B.ptr<float>();
    for (size_t i = 0; i < A.size(); ++i) a_ptr[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < B.size(); ++i) b_ptr[i] = static_cast<float>(rand()) / RAND_MAX;

    // 2. 跑基准真值 (C++ Reference)
    reference::matmul_ref(A, B, C_ref);

    // 3. 跑 NEON 加速版
    using namespace arm_neon;
    int mp = (M + MR - 1) / MR;
    int np = (N + NR - 1) / NR;
    size_t ws_size = (mp * MR * K + np * NR * K) * sizeof(float);
    float* workspace = (float*)g_memory_pool->allocate(ws_size);
    std::memset(workspace, 0, ws_size);

    arm_neon::matmul_neon(A, B, C_neon, workspace, false, nullptr);

    // 4. 逐元素比对 (允许微小误差)
    float* c_ref_ptr = C_ref.ptr<float>();
    float* c_neon_ptr = C_neon.ptr<float>();
    float epsilon = 1e-4f; 

    for (size_t i = 0; i < C_ref.size(); ++i) {
        EXPECT_NEAR(c_ref_ptr[i], c_neon_ptr[i], epsilon) 
            << "Matrix mismatch at flat index " << i 
            << " (ref=" << c_ref_ptr[i] << ", neon=" << c_neon_ptr[i] << ")";
    }

    // 清理 workspace（Tensor 析构会自动释放自己的内存）
    g_memory_pool->free_block(workspace);
}

// ============================================================
// 🔥 测试 2: RMSNorm 精度验证
// ============================================================
TEST(NeonTest, RMSNormAccuracy) {
    init_memory_pool();
    srand(42);

    // 故意选非 4 倍数，测试 Tail Case
    int n = 4093; 
    float eps = 1e-5f;

    Tensor X({1, n});
    Tensor W({1, n});
    Tensor Y_ref({1, n});
    Tensor Y_neon({1, n});

    // 🔥 确保内存已分配
    X.ensure_allocated();
    W.ensure_allocated();
    Y_ref.ensure_allocated();
    Y_neon.ensure_allocated();

    float* x_ptr = X.ptr<float>();
    float* w_ptr = W.ptr<float>();
    float* y_ref_ptr = Y_ref.ptr<float>();
    float* y_neon_ptr = Y_neon.ptr<float>();

    // 1. 填充随机数据
    for (int i = 0; i < n; ++i) {
        x_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
        w_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f + 0.9f;  // [0.9, 1.0]
    }

    // 2. C++ Reference 实现
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x_ptr[i] * x_ptr[i];
    }
    float mean = sum_sq / n;
    float scale = 1.0f / std::sqrt(mean + eps);
    for (int i = 0; i < n; ++i) {
        y_ref_ptr[i] = x_ptr[i] * scale * w_ptr[i];
    }

    // 3. NEON 实现
    arm_neon::rmsnorm_neon(x_ptr, w_ptr, y_neon_ptr, n, eps);

    // 4. 对比误差
    float epsilon = 1e-4f;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y_ref_ptr[i], y_neon_ptr[i], epsilon) 
            << "RMSNorm mismatch at index " << i;
    }
}

// ============================================================
// 🔥 测试 3: SwiGLU 精度验证
// ============================================================
TEST(NeonTest, SwiGLUAccuracy) {
    init_memory_pool();
    srand(42);

    int n = 4095;  // 非 4 倍数

    Tensor X({1, n});
    Tensor Up({1, n});
    Tensor Y_ref({1, n});
    Tensor Y_neon({1, n});

    // 🔥 确保内存已分配
    X.ensure_allocated();
    Up.ensure_allocated();
    Y_ref.ensure_allocated();
    Y_neon.ensure_allocated();

    float* x_ptr = X.ptr<float>();
    float* up_ptr = Up.ptr<float>();
    float* y_ref_ptr = Y_ref.ptr<float>();
    float* y_neon_ptr = Y_neon.ptr<float>();

    // 1. 填充数据（覆盖正负区间，测试 Swish 非线性）
    for (int i = 0; i < n; ++i) {
        x_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f;  // [-5, 5]
        up_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }

    // 2. C++ Reference: y = x * sigmoid(x) * up
    for (int i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-x_ptr[i]));
        y_ref_ptr[i] = x_ptr[i] * sig * up_ptr[i];
    }

    // 3. NEON 实现（使用近似 exp）
    arm_neon::swiglu_neon(x_ptr, up_ptr, y_neon_ptr, n);

    // 4. 对比误差（近似 exp 允许稍大误差）
    float epsilon = 1e-3f;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(y_ref_ptr[i], y_neon_ptr[i], epsilon) 
            << "SwiGLU mismatch at index " << i << ", x=" << x_ptr[i];
    }
}

// ============================================================
// 🔥 测试 4: RoPE 精度验证（In-place 算子）
// ============================================================
TEST(NeonTest, RoPEAccuracy) {
    init_memory_pool();
    srand(42);

    int n = 128;  // head_dim，必须是 4 的倍数

    // RoPE 是 In-place 修改，需要两份独立拷贝
    Tensor X_ref({1, n});
    Tensor X_neon({1, n});
    Tensor Cos({1, n});
    Tensor Sin({1, n});

    // 🔥 确保内存已分配
    X_ref.ensure_allocated();
    X_neon.ensure_allocated();
    Cos.ensure_allocated();
    Sin.ensure_allocated();

    float* x_ref_ptr = X_ref.ptr<float>();
    float* x_neon_ptr = X_neon.ptr<float>();
    float* cos_ptr = Cos.ptr<float>();
    float* sin_ptr = Sin.ptr<float>();

    // 1. 填充输入数据
    for (int i = 0; i < n; ++i) {
        float val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        x_ref_ptr[i] = val;
        x_neon_ptr[i] = val;  // 保持两份输入完全一致
    }

    // 2. 填充 Cos/Sin 表（NEON 要求相邻重复: [c0,c0,c1,c1,...]）
    for (int i = 0; i < n; i += 2) {
        float theta = (static_cast<float>(rand()) / RAND_MAX) * 3.14159f;
        float c = std::cos(theta);
        float s = std::sin(theta);
        cos_ptr[i] = c; cos_ptr[i+1] = c;
        sin_ptr[i] = s; sin_ptr[i+1] = s;
    }

    // 3. C++ Reference: 复数乘法展开
    // [x0, x1] * [c, -s; s, c] = [x0*c - x1*s, x1*c + x0*s]
    for (int i = 0; i < n; i += 2) {
        float x0 = x_ref_ptr[i];
        float x1 = x_ref_ptr[i+1];
        float c = cos_ptr[i]; 
        float s = sin_ptr[i];
        x_ref_ptr[i]   = x0 * c - x1 * s;
        x_ref_ptr[i+1] = x1 * c + x0 * s;
    }

    // 4. NEON 实现（In-place 修改 X_neon）
    arm_neon::rope_neon(x_neon_ptr, cos_ptr, sin_ptr, n);

    // 5. 对比误差
    float epsilon = 1e-4f;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x_ref_ptr[i], x_neon_ptr[i], epsilon) 
            << "RoPE mismatch at index " << i 
            << " (ref=" << x_ref_ptr[i] << ", neon=" << x_neon_ptr[i] << ")";
    }
}

// ============================================================
// 🔥 测试: Softmax_NEON 精度验证
// ============================================================
TEST(NeonTest, SoftmaxAccuracy) {
    init_memory_pool();
    srand(42);

    int num_rep = 8;       // 模拟有 8 个 Query Head 一起算
    int seq_len = 128;     // 模拟当前序列长度

    Tensor input({num_rep, seq_len});
    Tensor output_neon({num_rep, seq_len});
    Tensor output_ref({num_rep, seq_len}); // 用于存放 C++ 跑出来的标准答案

    input.ensure_allocated();
    output_neon.ensure_allocated();
    output_ref.ensure_allocated();

    float* in_ptr = input.ptr<float>();
    float* out_neon_ptr = output_neon.ptr<float>();
    float* out_ref_ptr = output_ref.ptr<float>();

    // 1. 填充随机输入
    for (int i = 0; i < input.size(); ++i) {
        in_ptr[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f; // -5 到 5 之间
    }

    // 2. 运行你写的 NEON 版本
    // 我们假设 softmax_neon 是 out-of-place 或者 in-place，这里为了方便比对，先把输入拷给 output_neon
    memcpy(out_neon_ptr, in_ptr, input.bytes());
    llm_engine::arm_neon::softmax_neon(output_neon, output_neon);

    // 3. 运行标准的 C++ Reference 版本
    for (int b = 0; b < num_rep; ++b) {
        const float* in_row = in_ptr + b * seq_len;
        float* out_row = out_ref_ptr + b * seq_len;

        float max_val = std::numeric_limits<float>::lowest();
        for (int i = 0; i < seq_len; ++i) max_val = std::max(max_val, in_row[i]);

        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            out_row[i] = std::exp(in_row[i] - max_val);
            sum_exp += out_row[i];
        }
        for (int i = 0; i < seq_len; ++i) out_row[i] /= sum_exp;
    }

    // 4. 对比结果 (因为涉及 exp 近似和浮点累加，允许 1e-3 的误差)
    for (int i = 0; i < output_neon.size(); ++i) {
        EXPECT_NEAR(out_neon_ptr[i], out_ref_ptr[i], 1e-3);
    }
}