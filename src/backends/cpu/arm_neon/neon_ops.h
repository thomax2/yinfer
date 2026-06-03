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

Status gemv_neon_transposed(
    const Tensor& A,
    const Tensor& B_T,
    Tensor& C,
    const float* bias = nullptr
);

Status linear_decode_prepacked_neon(
    const float* x,
    const float* w_pack,
    float* y,
    int K,
    int N,
    const float* bias = nullptr
);

// 在 [panel_begin, panel_end) 区间内计算 packed GEMV 的输出。
// panel_begin/panel_end 是 panel 维度（一个 panel = NR 个输出通道）。
// 不申请内存，不使用线程池，不访问 g_memory_pool。
Status linear_decode_prepacked_range_neon(
    const float* x,
    const float* w_pack,
    float* y,
    int K,
    int N,
    int panel_begin,
    int panel_end,
    const float* bias = nullptr
);

// 多线程版 packed GEMV：输出完整 y[0..N)。
// 当线程池不可用或 N 太小时，自动 fallback 串行。
Status linear_decode_prepacked_parallel_neon(
    const float* x,
    const float* w_pack,
    float* y,
    int K,
    int N,
    const float* bias = nullptr
);

struct ArgmaxResult {
    int index;
    float value;
};

// LM Head 专用：fused parallel argmax，不输出完整 logits。
// 每个 worker 维护一个 local 最大值，最后归并。
ArgmaxResult linear_decode_prepacked_argmax_parallel_neon(
    const float* x,
    const float* w_pack,
    int K,
    int N
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
    const Tensor& w_q_pack, const Tensor& w_k_pack, const Tensor& w_v_pack, const Tensor& w_o_pack,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
);

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
    const Tensor& w_gate_pack,
    const Tensor& w_up_pack,
    const Tensor& w_down_pack,
    const FFNConfig& config,
    Workspace& workspace
);

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
    Tensor& hidden_states,
    const Tensor& norm1_weight,
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o,
    const Tensor& w_q_pack, const Tensor& w_k_pack, const Tensor& w_v_pack, const Tensor& w_o_pack,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr,
    const Tensor& norm2_weight,
    const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down,
    const Tensor& w_gate_pack, const Tensor& w_up_pack, const Tensor& w_down_pack,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& attn_config,
    const FFNConfig& ffn_config,
    float rms_norm_eps,
    Workspace& workspace
);

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
);

}
}
