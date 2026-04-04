#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/workspace.h"

namespace llm_engine {
namespace arm_neon {

Status matmul_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float* workspace,
    bool transB = false,
    const float* bias = nullptr
);

Status bmm_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float* workspace,
    bool transB = false
);

Status softmax_neon(
    const Tensor& input,
    Tensor& output
);

void add_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
);

void rmsnorm_neon(
    const float* x,
    const float* weight,
    float* y,
    int n,
    float eps
);

void rope_neon(
    float* x,
    const float* cos,
    const float* sin,
    int n
);

void swiglu_neon(
    const float* x,
    const float* up,
    float* y,
    int n
);

struct AttentionConfig {
    int hidden_dim;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
};

Status attention_neon(
    const Tensor& hidden_states, 
    Tensor& attn_output,         
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o, 
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr, 
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
);

struct FFNConfig {
    int hidden_dim;
    int intermediate_size;
};

Status ffn_neon(
    const Tensor& hidden_states, 
    Tensor& ffn_output,          
    const Tensor& w_gate,        
    const Tensor& w_up,          
    const Tensor& w_down,        
    const FFNConfig& config,     
    Workspace& workspace
);

Status qwen_block_neon(
    Tensor& hidden_states,       // 输入并作为最终输出 (In-place)
    const Tensor& norm1_weight,  // Attention 前的 RMSNorm 权重
    // Attention 参数
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr,
    const Tensor& norm2_weight,  // FFN 前的 RMSNorm 权重
    // FFN 参数 (Qwen2.5 的 FFN 是无 Bias 的)
    const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& attn_config,
    const FFNConfig& ffn_config,
    float rms_norm_eps,
    Workspace& workspace
);

}
}