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
    if (fs::exists(p)) return p.string();
    fs::path from_build_tests = fs::path("../..") / "tests" / p;
    if (fs::exists(from_build_tests)) return from_build_tests.lexically_normal().string();
    return filepath;
}
}

static void load_tensor_from_bin(const std::string& filepath, Tensor& tensor) {
    std::string resolved_path = resolve_data_path(filepath);
    std::ifstream file(resolved_path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to open " << filepath;
    file.read(reinterpret_cast<char*>(tensor.ptr<float>()), tensor.bytes());
    file.close();
}

static void expect_tensor_close(const float* actual, const float* expected, int size, float rtol = 1e-3, float atol = 1e-3) {
    for (int i = 0; i < size; ++i) {
        float a = actual[i];
        float e = expected[i];
        float allowed_error = atol + rtol * std::abs(e);
        EXPECT_LE(std::abs(a - e), allowed_error) 
            << "Mismatch at index " << i << ". Expected: " << e << ", Actual: " << a;
    }
}

TEST(QwenBlockTest, PyTorchAlignment) {
    if (g_memory_pool == nullptr) g_memory_pool = new MemoryPool(256 * 1024 * 1024);
    
    arm_neon::AttentionConfig attn_config = {64, 4, 2, 16};
    arm_neon::FFNConfig ffn_config = {64, 128};
    float rms_norm_eps = 1e-6;
    int max_seq_len = 128, layer_id = 0, current_pos = 0, num_tokens = 1;

    // 1. 分配所有的张量
    Tensor hidden_states({num_tokens, attn_config.hidden_dim});
    Tensor norm1_weight({attn_config.hidden_dim});
    
    Tensor w_q({attn_config.hidden_dim, attn_config.num_q_heads * attn_config.head_dim});
    Tensor w_k({attn_config.hidden_dim, attn_config.num_kv_heads * attn_config.head_dim});
    Tensor w_v({attn_config.hidden_dim, attn_config.num_kv_heads * attn_config.head_dim});
    Tensor w_o({attn_config.num_q_heads * attn_config.head_dim, attn_config.hidden_dim});
    
    // ✅ 新增：分配 QKV Bias 的 Tensor (一维)
    Tensor b_q({attn_config.num_q_heads * attn_config.head_dim});
    Tensor b_k({attn_config.num_kv_heads * attn_config.head_dim});
    Tensor b_v({attn_config.num_kv_heads * attn_config.head_dim});

    Tensor cos_tensor({attn_config.num_q_heads * attn_config.head_dim});
    Tensor sin_tensor({attn_config.num_q_heads * attn_config.head_dim});
    Tensor norm2_weight({attn_config.hidden_dim});
    
    Tensor w_gate({attn_config.hidden_dim, ffn_config.intermediate_size});
    Tensor w_up({attn_config.hidden_dim, ffn_config.intermediate_size});
    Tensor w_down({ffn_config.intermediate_size, attn_config.hidden_dim});

    // 2. 确保内存已分配
    hidden_states.ensure_allocated(); norm1_weight.ensure_allocated();
    w_q.ensure_allocated(); w_k.ensure_allocated(); w_v.ensure_allocated(); w_o.ensure_allocated();
    b_q.ensure_allocated(); b_k.ensure_allocated(); b_v.ensure_allocated(); // ✅ 新增
    cos_tensor.ensure_allocated(); sin_tensor.ensure_allocated(); norm2_weight.ensure_allocated();
    w_gate.ensure_allocated(); w_up.ensure_allocated(); w_down.ensure_allocated();

    // 3. 从 bin 文件读取数据
    load_tensor_from_bin("data/block_hidden_states.bin", hidden_states);
    load_tensor_from_bin("data/block_norm1_w.bin", norm1_weight);
    load_tensor_from_bin("data/block_w_q.bin", w_q);
    load_tensor_from_bin("data/block_w_k.bin", w_k);
    load_tensor_from_bin("data/block_w_v.bin", w_v);
    load_tensor_from_bin("data/block_w_o.bin", w_o);
    
    // ✅ 新增：加载 Bias 数据
    load_tensor_from_bin("data/block_b_q.bin", b_q);
    load_tensor_from_bin("data/block_b_k.bin", b_k);
    load_tensor_from_bin("data/block_b_v.bin", b_v);

    load_tensor_from_bin("data/block_cos.bin", cos_tensor);
    load_tensor_from_bin("data/block_sin.bin", sin_tensor);
    load_tensor_from_bin("data/block_norm2_w.bin", norm2_weight);
    load_tensor_from_bin("data/block_w_gate.bin", w_gate);
    load_tensor_from_bin("data/block_w_up.bin", w_up);
    load_tensor_from_bin("data/block_w_down.bin", w_down);

    Workspace workspace(4 * 1024 * 1024); // 4MB Workspace
    KVCache kv_cache(1, max_seq_len, attn_config.num_kv_heads, attn_config.head_dim);

    // 4. 调用新接口
    Status status = arm_neon::qwen_block_neon(
        hidden_states, norm1_weight, 
        w_q, w_k, w_v, w_o, 
        b_q.ptr<float>(), b_k.ptr<float>(), b_v.ptr<float>(), // ✅ 传入 Bias 指针
        cos_tensor.ptr<float>(), sin_tensor.ptr<float>(), 
        norm2_weight, w_gate, w_up, w_down, 
        kv_cache, layer_id, current_pos, attn_config, ffn_config, rms_norm_eps, workspace
    );
    ASSERT_EQ(status, Status::SUCCESS);

    // 5. 对比验证
    Tensor golden_out({num_tokens, attn_config.hidden_dim});
    golden_out.ensure_allocated();
    load_tensor_from_bin("data/golden_block_out.bin", golden_out);
    
    expect_tensor_close(hidden_states.ptr<float>(), golden_out.ptr<float>(), num_tokens * attn_config.hidden_dim, 2e-3, 2e-3);
}