#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>

#include "llm_engine/tensor.h"
#include "llm_engine/memory/memory_pool.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/memory/kv_cache.h"
#include "backends/cpu/arm_neon/neon_ops.h"

using namespace llm_engine;

namespace {

std::string resolve_data_path(const std::string& filepath) {
    namespace fs = std::filesystem;
    fs::path p(filepath);

    if (fs::exists(p)) {
        return p.string();
    }

#ifdef TEST_DATA_DIR
    fs::path from_source_data = fs::path(TEST_DATA_DIR) / p.filename();
    if (fs::exists(from_source_data)) {
        return from_source_data.string();
    }
#endif

    // 兼容从 build/tests 目录直接执行二进制的场景
    fs::path from_build_tests = fs::path("../..") / "tests" / p;
    if (fs::exists(from_build_tests)) {
        return from_build_tests.lexically_normal().string();
    }

    return filepath;
}

} // namespace

// 辅助函数：从二进制文件加载数据到 Tensor 中
void load_tensor_from_bin(const std::string& filepath, Tensor& tensor) {
    std::string resolved_path = resolve_data_path(filepath);
    std::ifstream file(resolved_path, std::ios::binary);
    ASSERT_TRUE(file.is_open())
        << "Failed to open " << filepath
        << " (resolved: " << resolved_path
        << ", cwd: " << std::filesystem::current_path().string() << ")";
    file.read(reinterpret_cast<char*>(tensor.ptr<float>()), tensor.bytes());
    file.close();
}

// 辅助函数：对比两个数组是否一致
void expect_tensor_near(const float* actual, const float* expected, int size, float tol = 1e-4) {
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "Mismatch at index " << i;
    }
}

TEST(AttentionTest, PyTorchAlignment) {
    // 1. 初始化内存池和环境
    if (g_memory_pool == nullptr) {
        g_memory_pool = new MemoryPool(256 * 1024 * 1024); // 256MB
    }
    
    arm_neon::AttentionConfig config = {64, 4, 2, 16}; // 对应 Python 脚本的配置
    int max_seq_len = 128;
    int layer_id = 0;
    int current_pos = 0; // 我们测试第 0 步 (生成第一个 Token)

    // 2. 分配 C++ Tensor
    Tensor hidden_states({1, config.hidden_dim});
    Tensor attn_output({1, config.hidden_dim});
    Tensor w_q({config.hidden_dim, config.num_q_heads * config.head_dim});
    Tensor w_k({config.hidden_dim, config.num_kv_heads * config.head_dim});
    Tensor w_v({config.hidden_dim, config.num_kv_heads * config.head_dim});
    Tensor w_o({config.num_q_heads * config.head_dim, config.hidden_dim});
    Tensor cos_tensor({config.num_q_heads * config.head_dim});
    Tensor sin_tensor({config.num_q_heads * config.head_dim});

    hidden_states.ensure_allocated(); attn_output.ensure_allocated();
    w_q.ensure_allocated(); w_k.ensure_allocated(); w_v.ensure_allocated(); w_o.ensure_allocated();
    cos_tensor.ensure_allocated(); sin_tensor.ensure_allocated();

    // 3. 从 Python 生成的黄金数据加载输入
    load_tensor_from_bin("data/hidden_states.bin", hidden_states);
    load_tensor_from_bin("data/w_q.bin", w_q);
    load_tensor_from_bin("data/w_k.bin", w_k);
    load_tensor_from_bin("data/w_v.bin", w_v);
    load_tensor_from_bin("data/w_o.bin", w_o);
    load_tensor_from_bin("data/cos.bin", cos_tensor);
    load_tensor_from_bin("data/sin.bin", sin_tensor);

    // 4. 准备 Workspace 和 KVCache
    Workspace workspace(2 * 1024 * 1024); // 2MB Workspace
    KVCache kv_cache(1, max_seq_len, config.num_kv_heads, config.head_dim);

    // 5. 🔥 运行 C++ 核心 Attention 算子
    Status status = arm_neon::attention_neon(
        hidden_states, attn_output,
        w_q, w_k, w_v, w_o,
        cos_tensor.ptr<float>(), sin_tensor.ptr<float>(),
        kv_cache, layer_id, current_pos, config, workspace
    );
    ASSERT_EQ(status, Status::SUCCESS);

    // 6. 验证 1：对比最终输出 Attention Output
    Tensor golden_out({1, config.hidden_dim});
    golden_out.ensure_allocated();
    load_tensor_from_bin("data/golden_attn_out.bin", golden_out);
    
    // 允许 1e-4 的数值误差 (由于 NEON FMA 和浮点计算顺序不同)
    expect_tensor_near(attn_output.ptr<float>(), golden_out.ptr<float>(), config.hidden_dim, 1e-4);

    // 7. 验证 2：顺带精准测试 KV Cache 的副作用 (Side-effect)
    Tensor golden_k({1, config.num_kv_heads, 1, config.head_dim}); // PyTorch 导出的形状
    golden_k.ensure_allocated();
    load_tensor_from_bin("data/golden_k_cache.bin", golden_k);
    
    float* golden_k_ptr = golden_k.ptr<float>();
    for (int h = 0; h < config.num_kv_heads; ++h) {
        // 获取 C++ KVCache 里的指针
        float* c_cache_ptr = kv_cache.get_k_head_ptr(layer_id, h);
        float* py_cache_ptr = golden_k_ptr + h * config.head_dim; 
        
        // 检查这一层、这个 Head 的当前 Token 写入是否正确
        expect_tensor_near(c_cache_ptr, py_cache_ptr, config.head_dim, 1e-5);
    }
}