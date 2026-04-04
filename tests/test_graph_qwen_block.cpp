#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <filesystem>
#include <iostream>

#include "llm_engine/graph/graph.h"
#include "llm_engine/graph/compiler.h"
#include "llm_engine/memory/memory_pool.h"
#include "llm_engine/memory/kv_cache.h"
#include "backends/cpu/arm_neon/neon_ops.h"

using namespace llm_engine;

// 全局内存池外部声明
// extern MemoryPool* g_memory_pool;

namespace {
std::string resolve_data_path(const std::string& filepath) {
    namespace fs = std::filesystem;
    fs::path p(filepath);
    if (fs::exists(p)) return p.string();
    fs::path from_build_tests = fs::path("../..") / "tests" / p;
    if (fs::exists(from_build_tests)) return from_build_tests.lexically_normal().string();
    return filepath;
}
} // namespace

// 辅助加载函数 (适配 std::vector，模拟外部加载的常驻权重内存)
static void load_bin_to_vector(const std::string& filepath, std::vector<float>& vec) {
    std::string resolved_path = resolve_data_path(filepath);
    std::ifstream file(resolved_path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to open " << filepath;
    file.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(float));
    file.close();
}

static void expect_tensor_close(const float* actual, const float* expected, int size, float rtol = 1e-3, float atol = 1e-3) {
    for (int i = 0; i < size; ++i) {
        float allowed_error = atol + rtol * std::abs(expected[i]);
        EXPECT_LE(std::abs(actual[i] - expected[i]), allowed_error) 
            << "Mismatch at index " << i << ". Expected: " << expected[i] << ", Actual: " << actual[i];
    }
}

TEST(GraphIntegrationTest, QwenBlockNode) {
    // 1. 初始化引擎的全局内存池 (至少 256MB 保证充裕)
    if (g_memory_pool == nullptr) {
        g_memory_pool = new MemoryPool(256 * 1024 * 1024);
    }
    
    ComputationGraph g;
    
    // 配置参数 (需与 export_qwen_block.py 中的维度保持绝对一致)
    arm_neon::AttentionConfig attn_config = {64, 4, 2, 16};
    arm_neon::FFNConfig ffn_config = {64, 128};
    float rms_norm_eps = 1e-6;
    int max_seq_len = 128, layer_id = 0, num_tokens = 1;
    
    // 当前生成的 Token 序号，采用指针透传给 Node
    int current_pos = 0; 
    KVCache kv_cache(1, max_seq_len, attn_config.num_kv_heads, attn_config.head_dim);

    // 2. 在外部申请并读取真实的数据 (模拟模型加载)
    std::vector<float> h_states_data(num_tokens * attn_config.hidden_dim);
    std::vector<float> norm1_w_data(attn_config.hidden_dim);
    std::vector<float> w_q_data(attn_config.hidden_dim * attn_config.num_q_heads * attn_config.head_dim);
    std::vector<float> w_k_data(attn_config.hidden_dim * attn_config.num_kv_heads * attn_config.head_dim);
    std::vector<float> w_v_data(attn_config.hidden_dim * attn_config.num_kv_heads * attn_config.head_dim);
    std::vector<float> w_o_data(attn_config.num_q_heads * attn_config.head_dim * attn_config.hidden_dim);
    std::vector<float> b_q_data(attn_config.num_q_heads * attn_config.head_dim);
    std::vector<float> b_k_data(attn_config.num_kv_heads * attn_config.head_dim);
    std::vector<float> b_v_data(attn_config.num_kv_heads * attn_config.head_dim);
    std::vector<float> cos_data(attn_config.num_q_heads * attn_config.head_dim);
    std::vector<float> sin_data(attn_config.num_q_heads * attn_config.head_dim);
    std::vector<float> norm2_w_data(attn_config.hidden_dim);
    std::vector<float> w_gate_data(attn_config.hidden_dim * ffn_config.intermediate_size);
    std::vector<float> w_up_data(attn_config.hidden_dim * ffn_config.intermediate_size);
    std::vector<float> w_down_data(ffn_config.intermediate_size * attn_config.hidden_dim);

    load_bin_to_vector("data/block_hidden_states.bin", h_states_data);
    load_bin_to_vector("data/block_norm1_w.bin", norm1_w_data);
    load_bin_to_vector("data/block_w_q.bin", w_q_data);
    load_bin_to_vector("data/block_w_k.bin", w_k_data);
    load_bin_to_vector("data/block_w_v.bin", w_v_data);
    load_bin_to_vector("data/block_w_o.bin", w_o_data);
    load_bin_to_vector("data/block_b_q.bin", b_q_data);
    load_bin_to_vector("data/block_b_k.bin", b_k_data);
    load_bin_to_vector("data/block_b_v.bin", b_v_data);
    load_bin_to_vector("data/block_cos.bin", cos_data);
    load_bin_to_vector("data/block_sin.bin", sin_data);
    load_bin_to_vector("data/block_norm2_w.bin", norm2_w_data);
    load_bin_to_vector("data/block_w_gate.bin", w_gate_data);
    load_bin_to_vector("data/block_w_up.bin", w_up_data);
    load_bin_to_vector("data/block_w_down.bin", w_down_data);

    // 3. 将这 15 个外部数据指针注册为图(Graph)中的 Tensor (生命周期归 Graph 追踪)
    auto* t_hidden = g.create_tensor_from_ptr({num_tokens, attn_config.hidden_dim}, h_states_data.data());
    auto* t_n1_w = g.create_tensor_from_ptr({attn_config.hidden_dim}, norm1_w_data.data());
    auto* t_w_q = g.create_tensor_from_ptr({attn_config.hidden_dim, attn_config.num_q_heads * attn_config.head_dim}, w_q_data.data());
    auto* t_w_k = g.create_tensor_from_ptr({attn_config.hidden_dim, attn_config.num_kv_heads * attn_config.head_dim}, w_k_data.data());
    auto* t_w_v = g.create_tensor_from_ptr({attn_config.hidden_dim, attn_config.num_kv_heads * attn_config.head_dim}, w_v_data.data());
    auto* t_w_o = g.create_tensor_from_ptr({attn_config.num_q_heads * attn_config.head_dim, attn_config.hidden_dim}, w_o_data.data());
    auto* t_b_q = g.create_tensor_from_ptr({attn_config.num_q_heads * attn_config.head_dim}, b_q_data.data());
    auto* t_b_k = g.create_tensor_from_ptr({attn_config.num_kv_heads * attn_config.head_dim}, b_k_data.data());
    auto* t_b_v = g.create_tensor_from_ptr({attn_config.num_kv_heads * attn_config.head_dim}, b_v_data.data());
    auto* t_cos = g.create_tensor_from_ptr({attn_config.num_q_heads * attn_config.head_dim}, cos_data.data());
    auto* t_sin = g.create_tensor_from_ptr({attn_config.num_q_heads * attn_config.head_dim}, sin_data.data());
    auto* t_n2_w = g.create_tensor_from_ptr({attn_config.hidden_dim}, norm2_w_data.data());
    auto* t_w_gate = g.create_tensor_from_ptr({attn_config.hidden_dim, ffn_config.intermediate_size}, w_gate_data.data());
    auto* t_w_up = g.create_tensor_from_ptr({attn_config.hidden_dim, ffn_config.intermediate_size}, w_up_data.data());
    auto* t_w_down = g.create_tensor_from_ptr({ffn_config.intermediate_size, attn_config.hidden_dim}, w_down_data.data());

    // 4. 利用图接口创建 QwenBlock 节点 (不关心返回值，内部已存储到 graph.nodes)
    g.add_qwen_block(
        t_hidden, t_n1_w, 
        t_w_q, t_w_k, t_w_v, t_w_o, t_b_q, t_b_k, t_b_v,
        t_cos, t_sin, 
        t_n2_w, t_w_gate, t_w_up, t_w_down,
        &kv_cache, layer_id, &current_pos, // 💡 指针形式传入 current_pos
        attn_config, ffn_config, rms_norm_eps
    );

    // 5. 编译图，生成执行计划 (测试静态内存规划)
    GraphCompiler compiler;
    auto plan = compiler.compile(g); 
    
    // 6. 执行图 (测试动态 Workspace 借用和归还)
    for (auto* node : plan) {
        Status s = node->forward();
        ASSERT_EQ(s, Status::SUCCESS);
    }

    // 7. 验证最终结果是否对齐 PyTorch 的输出
    std::vector<float> golden_out(num_tokens * attn_config.hidden_dim);
    load_bin_to_vector("data/golden_block_out.bin", golden_out);
    
    // 允许容差
    expect_tensor_close(h_states_data.data(), golden_out.data(), num_tokens * attn_config.hidden_dim, 2e-3, 2e-3);
}