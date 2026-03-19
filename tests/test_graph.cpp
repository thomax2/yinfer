#include <gtest/gtest.h>

#include "llm_engine/graph/graph.h"
#include "llm_engine/graph/compiler.h"
#include "llm_engine/memory/memory_pool.h"

using namespace llm_engine;

static void init_memory_pool() {
    static bool initialized = false;
    if (!initialized) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
        initialized = true;
    }
}

TEST(GraphTest, ComplexDagWithMemoryCheck) {
    init_memory_pool();
    ComputationGraph g;

    // 1. 创建受保护的 Tensor (输入和权重)
    auto* X  = g.create_tensor({2, 2}); // 输入
    auto* W1 = g.create_tensor({2, 2}); // 权重 1
    auto* W2 = g.create_tensor({2, 2}); // 权重 2
    auto* B  = g.create_tensor({2, 2}); // 权重 3

    // 2. 创建中间临时 Tensor (算完应该被自动 Free)
    auto* M1 = g.create_tensor({2, 2});
    auto* M2 = g.create_tensor({2, 2});
    auto* A1 = g.create_tensor({2, 2});
    auto* M3 = g.create_tensor({2, 2});

    // 3. 创建受保护的 Tensor (最终输出)
    auto* Out = g.create_tensor({2, 2});

    // 4. 初始化明确的数据 (方便手动计算结果)
    float* x_ptr = X->ptr<float>();
    x_ptr[0]=1; x_ptr[1]=2; x_ptr[2]=3; x_ptr[3]=4;

    float* w1_ptr = W1->ptr<float>(); // 设为单位阵
    w1_ptr[0]=1; w1_ptr[1]=0; w1_ptr[2]=0; w1_ptr[3]=1;

    float* w2_ptr = W2->ptr<float>(); // 设为单位阵
    w2_ptr[0]=1; w2_ptr[1]=0; w2_ptr[2]=0; w2_ptr[3]=1;

    float* b_ptr = B->ptr<float>();   // 设为 0.5 倍的单位阵
    b_ptr[0]=0.5; b_ptr[1]=0; b_ptr[2]=0; b_ptr[3]=0.5;

    // 5. 构建复杂的有向无环图 (DAG)
    // 模拟大模型的一个带残差的 Block: Out = ((X*W1) + (X*W2)) * B + X
    g.add_matmul(X, W1, M1); // M1 = X
    g.add_matmul(X, W2, M2); // M2 = X
    g.add_add(M1, M2, A1);   // A1 = 2X
    g.add_matmul(A1, B, M3); // M3 = 2X * 0.5 = X
    g.add_add(M3, X, Out);   // Out = X + X = 2X

    // 6. 编译计算图
    GraphCompiler compiler;
    auto plan = compiler.compile(g);

    // 7. 执行计算图
    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    // ==========================================================
    // 核心校验点 1：验证数学结果是否正确 
    // Out 应该等于 2 * X = [2, 4, 6, 8]
    // ==========================================================
    float* out_ptr = Out->ptr<float>();
    EXPECT_FLOAT_EQ(out_ptr[0], 2.0f);
    EXPECT_FLOAT_EQ(out_ptr[1], 4.0f);
    EXPECT_FLOAT_EQ(out_ptr[2], 6.0f);
    EXPECT_FLOAT_EQ(out_ptr[3], 8.0f);

    // ==========================================================
    // 核心校验点 2：验证 Compiler 的自动内存回收 (FreeNode) 是否生效
    // 中间变量的 data 指针应该被置为了 nullptr
    // ==========================================================
    EXPECT_EQ(M1->data, nullptr) << "M1 should have been freed!";
    EXPECT_EQ(M2->data, nullptr) << "M2 should have been freed!";
    EXPECT_EQ(A1->data, nullptr) << "A1 should have been freed!";
    EXPECT_EQ(M3->data, nullptr) << "M3 should have been freed!";

    // ==========================================================
    // 核心校验点 3：验证受保护的 Tensor 是否安全活了下来
    // ==========================================================
    EXPECT_NE(X->data, nullptr)  << "X is an input, should NOT be freed!";
    EXPECT_NE(W1->data, nullptr) << "W1 is a weight, should NOT be freed!";
    EXPECT_NE(B->data, nullptr)  << "B is a weight, should NOT be freed!";
    EXPECT_NE(Out->data, nullptr)<< "Out is the final output, should NOT be freed!";
}