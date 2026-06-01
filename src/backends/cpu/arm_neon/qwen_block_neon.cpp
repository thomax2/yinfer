#include "backends/cpu/arm_neon/neon_ops.h"
#include <vector>
#include <cstring> // for memcpy

namespace llm_engine {
namespace arm_neon {

// Qwen Block 计算流
Status qwen_block_neon(
    Tensor& hidden_states,       // 输入并作为最终输出 (In-place)
    const Tensor& norm1_weight,  // Attention 前的 RMSNorm 权重
    // Attention 参数
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o,
    const Tensor& w_q_pack, const Tensor& w_k_pack, const Tensor& w_v_pack, const Tensor& w_o_pack,
    const float* q_bias, const float* k_bias, const float* v_bias,  // ✅ 新增 QKV Bias
    const float* cos_ptr, const float* sin_ptr,
    const Tensor& norm2_weight,  // FFN 前的 RMSNorm 权重
    // FFN 参数 (Qwen2.5 的 FFN 是无 Bias 的)
    const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down,
    const Tensor& w_gate_pack, const Tensor& w_up_pack, const Tensor& w_down_pack,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& attn_config,
    const FFNConfig& ffn_config,
    float rms_norm_eps,
    Workspace& workspace
) {
    int num_tokens = hidden_states.shape[0];
    int hidden_dim = hidden_states.shape[1];

    // ==========================================
    // 0. 从 Workspace 中安全切分 Buffer (零分配开销)
    // ==========================================
    size_t bytes_needed = num_tokens * hidden_dim * sizeof(float);
    
    char* ws_base = static_cast<char*>(workspace.data());
    
    // 切出 residual 的内存
    float* res_ptr = reinterpret_cast<float*>(ws_base);
    ws_base += bytes_needed;
    
    // 切出 norm_out 的内存
    float* norm_ptr = reinterpret_cast<float*>(ws_base);
    ws_base += bytes_needed;

    Tensor residual({num_tokens, hidden_dim}, res_ptr);
    Tensor norm_out({num_tokens, hidden_dim}, norm_ptr);

    // ⭐ 新增：构造剩下的内存区域为 sub_workspace
    size_t used_bytes = 2 * bytes_needed;
    if (workspace.size() < used_bytes) {
        return Status::OUT_OF_MEMORY; // 内存不足保护
    }
    Workspace sub_workspace(ws_base, workspace.size() - used_bytes);

    // ==========================================
    // 1. residual = hidden_states
    // ==========================================
    std::memcpy(residual.ptr<float>(), hidden_states.ptr<float>(), num_tokens * hidden_dim * sizeof(float));

    // ==========================================
    // 2. norm_out = RMSNorm(hidden_states)
    // ==========================================
    for (int i = 0; i < num_tokens; ++i) {
        rmsnorm_neon(
            hidden_states.ptr<float>() + i * hidden_dim,
            norm1_weight.ptr<float>(),
            norm_out.ptr<float>() + i * hidden_dim,
            hidden_dim, rms_norm_eps
        );
    }

    // ==========================================
    // 3. hidden_states = Attention(norm_out)
    // ==========================================
    Status status = attention_neon(
        norm_out, hidden_states, 
        w_q, w_k, w_v, w_o,
        w_q_pack, w_k_pack, w_v_pack, w_o_pack,
        q_bias, k_bias, v_bias,
        cos_ptr, sin_ptr, 
        kv_cache, layer_id, current_pos, attn_config,
        sub_workspace
    );
    if (status != Status::SUCCESS) return status;

    // ==========================================
    // 4. hidden_states = Add(residual, hidden_states)
    // ==========================================
    add_neon(residual, hidden_states, hidden_states);

    // ==========================================
    // 5. residual = hidden_states
    // ==========================================
    std::memcpy(residual.ptr<float>(), hidden_states.ptr<float>(), num_tokens * hidden_dim * sizeof(float));

    // ==========================================
    // 6. norm_out = RMSNorm(hidden_states)
    // ==========================================
    for (int i = 0; i < num_tokens; ++i) {
        rmsnorm_neon(
            hidden_states.ptr<float>() + i * hidden_dim,
            norm2_weight.ptr<float>(),
            norm_out.ptr<float>() + i * hidden_dim,
            hidden_dim, rms_norm_eps
        );
    }

    // ==========================================
    // 7. hidden_states = FFN(norm_out)
    // ==========================================
    status = ffn_neon(
        norm_out, hidden_states,
        w_gate, w_up, w_down,
        w_gate_pack, w_up_pack, w_down_pack,
        ffn_config,
        sub_workspace
    );
    if (status != Status::SUCCESS) return status;

    // ==========================================
    // 8. hidden_states = Add(residual, hidden_states)
    // ==========================================
    add_neon(residual, hidden_states, hidden_states);

    return Status::SUCCESS;
}

Status qwen_block_neon(
    Tensor& hidden_states,
    const Tensor& norm1_weight,
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr,
    const Tensor& norm2_weight,
    const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& attn_config,
    const FFNConfig& ffn_config,
    float rms_norm_eps,
    Workspace& workspace
) {
    Tensor empty;
    return qwen_block_neon(
        hidden_states, norm1_weight,
        w_q, w_k, w_v, w_o,
        empty, empty, empty, empty,
        q_bias, k_bias, v_bias,
        cos_ptr, sin_ptr,
        norm2_weight,
        w_gate, w_up, w_down,
        empty, empty, empty,
        kv_cache, layer_id, current_pos,
        attn_config, ffn_config, rms_norm_eps,
        workspace
    );
}

} // namespace arm_neon
} // namespace llm_engine
