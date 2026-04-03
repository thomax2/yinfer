#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>

#include "llm_engine/tensor.h"
#include "llm_engine/memory/memory_pool.h"
#include "llm_engine/memory/workspace.h"
#include "backends/cpu/arm_neon/neon_ops.h"

using namespace llm_engine;

namespace {
// 复用路径解析函数
std::string resolve_data_path(const std::string& filepath) {
    namespace fs = std::filesystem;
    fs::path p(filepath);
    if (fs::exists(p)) return p.string();
#ifdef TEST_DATA_DIR
    fs::path from_source_data = fs::path(TEST_DATA_DIR) / p.filename();
    if (fs::exists(from_source_data)) return from_source_data.string();
#endif
    fs::path from_build_tests = fs::path("../..") / "tests" / p;
    if (fs::exists(from_build_tests)) return from_build_tests.lexically_normal().string();
    return filepath;
}
} // namespace

// 辅助函数：加载 bin 文件
static void load_tensor_from_bin(const std::string& filepath, Tensor& tensor) {
    std::string resolved_path = resolve_data_path(filepath);
    std::ifstream file(resolved_path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to open " << filepath;
    file.read(reinterpret_cast<char*>(tensor.ptr<float>()), tensor.bytes());
    file.close();
}

// 辅助函数：比对数组
static void expect_tensor_near(const float* actual, const float* expected, int size, float tol = 1e-4) {
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "Mismatch at index " << i << ". Expected: " << expected[i] << ", Actual: " << actual[i];
    }
}

TEST(FFNTest, PyTorchAlignment) {
    // 1. 初始化内存池
    if (g_memory_pool == nullptr) {
        g_memory_pool = new MemoryPool(256 * 1024 * 1024); // 256MB
    }
    
    // 配置 FFN 参数: 假设 hidden_dim=64, intermediate_size=128 (通常 LLaMA 中是 8/3 倍)
    arm_neon::FFNConfig config = {64, 128}; 
    int num_tokens = 1; // 测试 Decode 阶段单 Token，也可以改为 >1 测试 Prefill 阶段

    // 2. 分配 C++ Tensor
    Tensor hidden_states({num_tokens, config.hidden_dim});
    Tensor ffn_output({num_tokens, config.hidden_dim});
    
    // 注意：C++ 的 Matmul 权重通常是 [in_features, out_features] 的内存排布
    Tensor w_gate({config.hidden_dim, config.intermediate_size});
    Tensor w_up({config.hidden_dim, config.intermediate_size});
    Tensor w_down({config.intermediate_size, config.hidden_dim});

    hidden_states.ensure_allocated(); 
    ffn_output.ensure_allocated();
    w_gate.ensure_allocated(); 
    w_up.ensure_allocated(); 
    w_down.ensure_allocated();

    // 3. 从 Python 生成的黄金数据加载输入和权重
    load_tensor_from_bin("data/ffn_hidden_states.bin", hidden_states);
    load_tensor_from_bin("data/ffn_w_gate.bin", w_gate);
    load_tensor_from_bin("data/ffn_w_up.bin", w_up);
    load_tensor_from_bin("data/ffn_w_down.bin", w_down);

    // 4. 准备 Workspace 
    // FFN 需要暂存 gate 和 up，且还需要 matmul 的 pack 缓存
    // 按照你 ffn_neon.cpp 中的严格校验，2MB 绝对足够应付这里的测试小尺寸
    Workspace workspace(2 * 1024 * 1024); 

    // 5. 运行 C++ 核心 FFN 算子
    Status status = arm_neon::ffn_neon(
        hidden_states, ffn_output,
        w_gate, w_up, w_down,
        config, workspace
    );
    ASSERT_EQ(status, Status::SUCCESS);

    // 6. 验证：对比最终输出
    Tensor golden_out({num_tokens, config.hidden_dim});
    golden_out.ensure_allocated();
    load_tensor_from_bin("data/golden_ffn_out.bin", golden_out);
    
    // 允许 1e-4 的数值误差 (NEON 指令重排及 float32 累加顺序会导致微小尾差)
    expect_tensor_near(ffn_output.ptr<float>(), golden_out.ptr<float>(), num_tokens * config.hidden_dim, 1e-4);
}