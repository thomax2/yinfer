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

// Paged Prefill 第一阶段：完整 Score + 显式 causal mask。
// 使用跨两个物理页且逻辑页顺序被打乱的数据，验证 Multi-Query 路径与
// “逐 Query 调用现有 Paged Decode Kernel”的数值结果一致。
TEST(NeonTest, PagedPrefillMultiQueryMatchesDecodeReference) {
    constexpr int query_rows = 5;      // 覆盖 4-row tile 和尾部 active_rows=1
    constexpr int start_position = 14;
    constexpr int seq_len = start_position + query_rows;
    constexpr int block_size = 16;
    constexpr int physical_blocks = 2;
    constexpr int num_layers = 1;
    constexpr int num_kv_heads = 1;
    constexpr int num_rep = 2;
    constexpr int head_dim = 8;
    constexpr int q_row_stride = num_rep * head_dim;

    // Logical block 0/1 分别映射到 Physical block 1/0，确保内核真正查表。
    const int block_table[2] = {1, 0};
    const size_t page_elements =
        (size_t)physical_blocks * num_layers * num_kv_heads *
        block_size * head_dim;
    std::vector<fp16_t> k_pages(page_elements, (fp16_t)0);
    std::vector<fp16_t> v_pages(page_elements, (fp16_t)0);
    std::vector<fp16_t> q((size_t)query_rows * q_row_stride);

    auto page_ptr = [&](std::vector<fp16_t>& pages, int physical_block) {
        return pages.data() + (size_t)physical_block * block_size * head_dim;
    };
    for (int token = 0; token < seq_len; ++token) {
        const int logical_block = token / block_size;
        const int offset = token % block_size;
        fp16_t* k = page_ptr(k_pages, block_table[logical_block]) +
            (size_t)offset * head_dim;
        fp16_t* v = page_ptr(v_pages, block_table[logical_block]) +
            (size_t)offset * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            k[d] = (fp16_t)(0.01f * (token + 1) + 0.002f * d);
            v[d] = (fp16_t)(0.015f * (token + 1) - 0.001f * d);
        }
    }
    for (int row = 0; row < query_rows; ++row) {
        for (int i = 0; i < q_row_stride; ++i) {
            q[(size_t)row * q_row_stride + i] =
                (fp16_t)(0.02f * (row + 1) + 0.003f * i);
        }
    }

    std::vector<fp16_t> actual((size_t)query_rows * q_row_stride, (fp16_t)0);
    std::vector<fp16_t> online((size_t)query_rows * q_row_stride, (fp16_t)0);
    std::vector<fp16_t> expected((size_t)query_rows * q_row_stride, (fp16_t)0);
    std::vector<fp16_t> score((size_t)query_rows * num_rep * seq_len);
    std::vector<fp16_t> reference_score((size_t)num_rep * seq_len);
    std::vector<fp16_t> k_tile((size_t)head_dim * 16);
    std::vector<float> online_out_acc((size_t)4 * head_dim);
    const float scale = 1.0f / std::sqrt((float)head_dim);

    ASSERT_EQ(
        arm_neon::attention_prefill_paged_f16_neon_public(
            q.data(), actual.data(), query_rows, start_position,
            q_row_stride, num_rep, head_dim, scale,
            k_pages.data(), v_pages.data(), block_table, 2, block_size,
            0, 0, num_layers, num_kv_heads, physical_blocks,
            score.data(), score.size(), k_tile.data(), k_tile.size()),
        Status::SUCCESS);

    // 第二阶段 Online Softmax 使用相同的 Paged KV 与乱序 Block Table，
    // 但不接收完整 Score Workspace；这里只提供一个 K Tile 和 4 行 FP32 O。
    ASSERT_EQ(
        arm_neon::attention_prefill_paged_online_f16_neon_public(
            q.data(), online.data(), query_rows, start_position,
            q_row_stride, num_rep, head_dim, scale,
            k_pages.data(), v_pages.data(), block_table, 2, block_size,
            0, 0, num_layers, num_kv_heads, physical_blocks,
            k_tile.data(), k_tile.size(),
            online_out_acc.data(), online_out_acc.size()),
        Status::SUCCESS);

    for (int row = 0; row < query_rows; ++row) {
        const int visible = start_position + row + 1;
        arm_neon::attention_decode_score_paged_f16_neon_public(
            q.data() + (size_t)row * q_row_stride,
            reference_score.data(), num_rep, visible, head_dim, scale,
            k_pages.data(), block_table, 2, block_size, 0, 0,
            num_layers, num_kv_heads, physical_blocks);
        ASSERT_EQ(
            arm_neon::softmax_f16_inplace_neon(
                reference_score.data(), num_rep, visible),
            Status::SUCCESS);
        ASSERT_EQ(
            arm_neon::attention_decode_value_paged_f16_neon_public(
                reference_score.data(),
                expected.data() + (size_t)row * q_row_stride,
                num_rep, visible, head_dim, v_pages.data(), block_table, 2,
                block_size, 0, 0, num_layers, num_kv_heads, physical_blocks),
            Status::SUCCESS);
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR((float)actual[i], (float)expected[i], 2e-2f)
            << "paged prefill mismatch at element " << i;
        EXPECT_NEAR((float)online[i], (float)expected[i], 2e-2f)
            << "online paged prefill mismatch at element " << i;
        EXPECT_NEAR((float)online[i], (float)actual[i], 2e-2f)
            << "online/dense paged prefill mismatch at element " << i;
    }
}
