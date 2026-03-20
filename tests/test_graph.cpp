#include <gtest/gtest.h>
#include <iostream>

#include "llm_engine/graph/graph.h"
#include "llm_engine/graph/compiler.h"
#include "llm_engine/memory/memory_pool.h"

using namespace llm_engine;

// ===== 辅助：初始化全局内存池 =====
static void init_memory_pool() {
    static bool initialized = false;
    if (!initialized) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
        initialized = true;
    }
}

// ===== 自定义 In-place 算子：模拟 RoPE =====
class InPlaceRoPENode : public GraphNode {
public:
    InPlaceRoPENode(Tensor* input) {
        inputs.push_back(input);
        outputs.push_back(input);  // 🔥 关键：output 就是 input 本身！
    }
    
    Status forward() override {
        float* data = inputs[0]->ptr<float>();
        size_t size = inputs[0]->size();
        for(size_t i = 0; i < size; i++) {
            data[i] *= 1.1f;
        }
        return Status::SUCCESS;
    }
};

void add_inplace_rope(ComputationGraph& g, Tensor* input) {
    g.nodes.push_back(std::make_unique<InPlaceRoPENode>(input));
}

// ============================================================
// 🔥 测试 1: In-place + 后续还要使用 → 绝不能提前释放
// ============================================================
TEST(InplaceMemoryTest, InplaceWithSubsequentUse) {
    init_memory_pool();
    ComputationGraph g;

    auto* X = g.create_tensor({2, 2});  
    auto* W = g.create_tensor({2, 2});  
    auto* Out = g.create_tensor({2, 2});

    // 建图: X -> RoPE(X) [inplace] -> Matmul(X, W) -> Out
    add_inplace_rope(g, X);  
    g.add_matmul(X, W, Out);  

    // 【关键修复】：先编译分配内存！
    GraphCompiler compiler;
    auto plan = compiler.compile(g);

    // 再写数据
    float* x_ptr = X->ptr<float>();
    for(int i = 0; i < 4; i++) x_ptr[i] = 1.0f + i;  // [1,2,3,4]

    float* w_ptr = W->ptr<float>();
    w_ptr[0]=1; w_ptr[1]=0; w_ptr[2]=0; w_ptr[3]=1; // 单位阵

    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    // 校验
    float* out_ptr = Out->ptr<float>();
    EXPECT_FLOAT_EQ(out_ptr[0], 1.1f);
    EXPECT_FLOAT_EQ(out_ptr[3], 4.4f);
    
    EXPECT_NE(X->data, nullptr);
    std::cout << "[PASS] Inplace + subsequent use\n";
}

// ============================================================
// 🔥 测试 2: In-place + 是最终输出 → protected 保护
// ============================================================
TEST(InplaceMemoryTest, InplaceAsFinalOutput) {
    init_memory_pool();
    ComputationGraph g;

    auto* X = g.create_tensor({2, 2});
    add_inplace_rope(g, X); 

    GraphCompiler compiler;
    auto plan = compiler.compile(g);

    float* x_ptr = X->ptr<float>();
    for(int i = 0; i < 4; i++) x_ptr[i] = 10.0f + i;  

    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    float* result = X->ptr<float>(); 
    EXPECT_FLOAT_EQ(result[0], 11.0f);  
    EXPECT_NE(X->data, nullptr);
    std::cout << "[PASS] Inplace as final output\n";
}

// ============================================================
// 🔥 测试 3: In-place死节点内存回收 (极其核心的边界验证)
// ============================================================
TEST(InplaceMemoryTest, InplaceDeadValueShouldRecycle) {
    init_memory_pool();
    ComputationGraph g;

    // 为了让 X 成为“中间变量”，我们用 Matmul 生产它
    auto* Input1 = g.create_tensor({2, 2});
    auto* W1 = g.create_tensor({2, 2});
    auto* X = g.create_tensor({2, 2}); // 中间变量 X
    
    // Step 0: Input1 * W1 -> X
    g.add_matmul(Input1, W1, X);

    // Step 1: RoPE(X) 
    // X 在这里被 in-place 修改，之后再无节点使用，它在 Step 1 寿终正寝！
    add_inplace_rope(g, X); 
    
    // Step 2: 产生一个新的毫无关联的运算，用来“吃掉” X 吐出来的内存
    auto* Input2 = g.create_tensor({2, 2});
    auto* W2 = g.create_tensor({2, 2});
    auto* Y = g.create_tensor({2, 2}); // 中间变量 Y
    auto* FinalOut = g.create_tensor({2, 2});
    
    // Input2 * W2 -> Y -> FinalOut 
    g.add_matmul(Input2, W2, Y);
    g.add_add(Y, Input2, FinalOut);

    // 编译
    GraphCompiler compiler;
    auto plan = compiler.compile(g);

    // 🔍 终极校验：如果你的回收逻辑是对的，Y 一定会 100% 复用 X 的物理内存！
    EXPECT_EQ(X->data, Y->data) << "FATAL: X was not recycled into free_blocks!";
    
    std::cout << "[PASS] Inplace dead value strictly recycled!\n";
}

// ============================================================
// 🔥 测试 4: 压力测试 - 连续多个 inplace 算子
// ============================================================
TEST(InplaceMemoryTest, StressMultipleInplace) {
    init_memory_pool();
    ComputationGraph g;

    auto* X = g.create_tensor({4, 4}); 
    auto* Zero = g.create_tensor({4, 4}); 
    auto* Out = g.create_tensor({4, 4});

    // 连续 3 次 inplace
    add_inplace_rope(g, X);
    add_inplace_rope(g, X);
    add_inplace_rope(g, X);
    
    // 用 Out = X + 0 代替 Identity，避免 X 成为最终输出
    g.add_add(X, Zero, Out);  

    GraphCompiler compiler;
    auto plan = compiler.compile(g);

    float* x_ptr = X->ptr<float>();
    float* zero_ptr = Zero->ptr<float>();
    for(int i = 0; i < 16; i++) {
        x_ptr[i] = 1.0f;
        zero_ptr[i] = 0.0f;
    }

    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    float* out_ptr = Out->ptr<float>();
    // 1.0 * 1.1 * 1.1 * 1.1 = 1.331
    EXPECT_NEAR(out_ptr[0], 1.331f, 1e-5);
    
    std::cout << "[PASS] Stress test: multiple inplace ops passed!\n";
}