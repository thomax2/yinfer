#include <gtest/gtest.h>
#include <iostream>
#include <vector>

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
        outputs.push_back(input);  // 关键：output 就是 input 本身！
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
// 🔥 测试 1: In-place + 后续还要使用 
// (全外部内存分配，完美零拷贝)
// ============================================================
TEST(InplaceMemoryTest, InplaceWithSubsequentUse_External) {
    init_memory_pool();
    ComputationGraph g;

    // 1. 在引擎外部准备好所有的边界内存 (Inputs, Weights, Outputs)
    std::vector<float> ext_X = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> ext_W = {1.0f, 0.0f, 0.0f, 1.0f}; // 单位阵
    std::vector<float> ext_Out(4, 0.0f);                 // 接收结果的 Buffer

    // 2. 将外部内存绑入图中 (边界张量必须用 from_ptr)
    auto* X   = g.create_tensor_from_ptr({2, 2}, ext_X.data());  
    auto* W   = g.create_tensor_from_ptr({2, 2}, ext_W.data());  
    auto* Out = g.create_tensor_from_ptr({2, 2}, ext_Out.data());

    // 3. 建图: X -> RoPE(X) [inplace] -> Matmul(X, W) -> Out
    add_inplace_rope(g, X);  
    g.add_matmul(X, W, Out);  

    // 4. 编译与执行
    GraphCompiler compiler;
    auto plan = compiler.compile(g);
    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    // 5. 极其优雅的校验：直接检查我们自己的 std::vector！
    EXPECT_FLOAT_EQ(ext_Out[0], 1.1f);
    EXPECT_FLOAT_EQ(ext_Out[3], 4.4f);
    
    std::cout << "[PASS] Test 1: Inplace + subsequent use (External Binding)\n";
}

// ============================================================
// 🔥 测试 2: In-place 作为最终输出
// ============================================================
TEST(InplaceMemoryTest, InplaceAsFinalOutput_External) {
    init_memory_pool();
    ComputationGraph g;

    // 外部准备内存
    std::vector<float> ext_X = {10.0f, 11.0f, 12.0f, 13.0f};

    // 绑入图中
    auto* X = g.create_tensor_from_ptr({2, 2}, ext_X.data());
    
    // 建图
    add_inplace_rope(g, X); 

    GraphCompiler compiler;
    auto plan = compiler.compile(g);
    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    // 校验：X 既是输入也是最终输出，原位被修改
    EXPECT_FLOAT_EQ(ext_X[0], 11.0f);  // 10 * 1.1
    EXPECT_FLOAT_EQ(ext_X[3], 14.3f);  // 13 * 1.1
    std::cout << "[PASS] Test 2: Inplace as final output (External Binding)\n";
}

// ============================================================
// 🔥 测试 3: 内部 Arena 内存复用 (极其核心的内外区分测试)
// ============================================================
TEST(InplaceMemoryTest, InternalArenaRecycling) {
    init_memory_pool();
    ComputationGraph g;

    std::vector<float> ext_In1(4, 1.0f), ext_W1(4, 1.0f);
    std::vector<float> ext_W2(4, 1.0f);
    std::vector<float> ext_Final(4, 0.0f);

    auto* Input1   = g.create_tensor_from_ptr({2, 2}, ext_In1.data());
    auto* W1       = g.create_tensor_from_ptr({2, 2}, ext_W1.data());
    auto* W2       = g.create_tensor_from_ptr({2, 2}, ext_W2.data());
    auto* FinalOut = g.create_tensor_from_ptr({2, 2}, ext_Final.data());

    // 内部临时变量 (交给引擎 Arena 管理)
    auto* X = g.create_tensor({2, 2}); 
    auto* Dummy = g.create_tensor({2, 2}); 
    auto* Y = g.create_tensor({2, 2}); 
    
    // Step 0: 生产 X
    g.add_matmul(Input1, W1, X);

    // Step 1: In-place 修改 X
    add_inplace_rope(g, X); 
    
    // Step 2: 消耗 X 生产 Dummy (【X 在这一步阵亡】，Offset 被放回 free_blocks)
    g.add_matmul(X, W2, Dummy);

    // Step 3: 消耗 Dummy 生产 Y (【Y 在这一步出生】，它 100% 会捡走 X 刚空出来的内存！)
    g.add_add(Dummy, Dummy, Y);

    // Step 4: 将 Y 写入外部输出边界
    g.add_add(Y, Y, FinalOut);

    GraphCompiler compiler;
    auto plan = compiler.compile(g);

    // 🔍 终极校验：由于 Y 出生在 X 死亡之后，它一定会完美复用 X 的指针！
    EXPECT_EQ(X->data, Y->data) << "FATAL: Internal Arena did not recycle intermediate tensors!";
    
    std::cout << "[PASS] Test 3: Internal Arena strictly recycled!\n";
}


// ============================================================
// 🔥 测试 4: 压力测试 - 连续多个 inplace 算子
// ============================================================
TEST(InplaceMemoryTest, StressMultipleInplace_External) {
    init_memory_pool();
    ComputationGraph g;

    std::vector<float> ext_X(16, 1.0f); 
    std::vector<float> ext_Zero(16, 0.0f); 
    std::vector<float> ext_Out(16, 0.0f);

    auto* X    = g.create_tensor_from_ptr({4, 4}, ext_X.data()); 
    auto* Zero = g.create_tensor_from_ptr({4, 4}, ext_Zero.data()); 
    auto* Out  = g.create_tensor_from_ptr({4, 4}, ext_Out.data());

    // 连续 3 次 inplace
    add_inplace_rope(g, X);
    add_inplace_rope(g, X);
    add_inplace_rope(g, X);
    
    g.add_add(X, Zero, Out);  

    GraphCompiler compiler;
    auto plan = compiler.compile(g);
    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    // 直接校验外部的 ext_Out 数组
    EXPECT_NEAR(ext_Out[0], 1.331f, 1e-5);
    
    std::cout << "[PASS] Test 4: Stress test passed!\n";
}

// ============================================================
// 🚨 测试 5: 验证编译器的异常拦截防线 (你刚写的抛异常逻辑)
// ============================================================
TEST(CompilerTest, ShouldThrowIfBoundaryNotBound) {
    init_memory_pool();
    ComputationGraph g;

    // 错误示范：用户想创建一个纯输入张量，但却懒惰地用了 create_tensor (没有绑指针)
    auto* X = g.create_tensor({2, 2});
    auto* Out = g.create_tensor({2, 2});

    // Dummy 操作
    g.add_add(X, X, Out);

    GraphCompiler compiler;
    
    // 校验：编译器必须精准抛出 std::runtime_error 拦截这种行为！
    EXPECT_THROW({
        compiler.compile(g);
    }, std::runtime_error);

    std::cout << "[PASS] Test 5: Compiler successfully threw exception for unbound boundary tensors!\n";
}