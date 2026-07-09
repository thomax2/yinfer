#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/workspace.h"
#include "backends/cpu/arm_neon/quant_gptq.h"

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

Status matmul_f16_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    fp16_t* workspace,
    bool transB = false,
    const fp16_t* bias = nullptr
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

// LM Head 专用：fused 串行 argmax，仅供 debug / 一致性比对使用。
// 不申请内存、不使用线程池。
ArgmaxResult linear_decode_prepacked_argmax_serial_neon(
    const float* x,
    const float* w_pack,
    int K,
    int N
);

Status linear_gptq_int8_decode_neon(
    const fp16_t* x,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias = nullptr,
    void* workspace = nullptr,
    size_t workspace_bytes = 0
);

Status linear_gptq_int8_batch_neon(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& w,
    fp16_t* y,
    const fp16_t* bias = nullptr,
    void* workspace = nullptr,
    size_t workspace_bytes = 0
);

ArgmaxResult linear_gptq_int8_decode_argmax_neon(
    const fp16_t* x,
    const GPTQInt8Weight& w,
    void* workspace = nullptr,
    size_t workspace_bytes = 0
);

Status fused_gate_up_swiglu_gptq_int8_decode_neon(
    const fp16_t* x,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    fp16_t* y,
    void* workspace = nullptr,
    size_t workspace_bytes = 0
);

// FFN 专用：在 [panel_begin, panel_end) 范围内同时计算 gate / up，
// 并 fused 完成 SwiGLU。y[i] = silu(gate[i]) * up[i]。
// w_gate_pack / w_up_pack 与 linear_decode_prepacked_neon 使用同一种 packed 布局。
// 不申请内存、不使用线程池、不访问 g_memory_pool。
Status fused_gate_up_swiglu_prepacked_range_neon(
    const float* x,
    const float* w_gate_pack,
    const float* w_up_pack,
    float* y,
    int K,
    int N,
    int panel_begin,
    int panel_end
);

// FFN 专用：fused gate+up+SwiGLU 的多线程版本。
// 当线程池不可用或 N 太小时自动 fallback 串行 range kernel。
Status fused_gate_up_swiglu_prepacked_parallel_neon(
    const float* x,
    const float* w_gate_pack,
    const float* w_up_pack,
    float* y,
    int K,
    int N
);

Status fused_gate_up_swiglu_gptq_int8_batch_neon(
    const fp16_t* x,
    int rows,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    fp16_t* y,
    void* workspace = nullptr,
    size_t workspace_bytes = 0
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

Status softmax_f16_neon(
    const Tensor& input,
    Tensor& output
);

void attention_decode_score_f16_neon_public(
    const fp16_t* q,
    const fp16_t* k_cache,
    fp16_t* score,
    int num_rep,
    int seq_len,
    int head_dim,
    float scale
);

void attention_decode_value_f16_neon_public(
    const fp16_t* score,
    const fp16_t* v_cache,
    fp16_t* out,
    int num_rep,
    int seq_len,
    int head_dim
);

Status attention_decode_score_paged_f16_neon_public(
    const fp16_t* q,
    fp16_t* score,
    int num_rep,
    int seq_len,
    int head_dim,
    float scale,
    const fp16_t* raw_k_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks
);

Status attention_decode_value_paged_f16_neon_public(
    const fp16_t* score,
    fp16_t* out,
    int num_rep,
    int seq_len,
    int head_dim,
    const fp16_t* raw_v_pages,
    const int* block_table,
    int block_table_size,
    int block_size,
    int layer_id,
    int kv_head,
    int num_layers,
    int num_kv_heads,
    int num_physical_blocks
);

void add_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
);

void add_f16_neon(
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

void rmsnorm_f16_neon(
    const fp16_t* x,
    const fp16_t* weight,
    fp16_t* y,
    int n,
    float eps
);

void rope_neon(
    float* x,
    const float* cos,
    const float* sin,
    int n
);

void rope_f16_neon(
    fp16_t* x,
    const fp16_t* cos,
    const fp16_t* sin,
    int n
);

void swiglu_neon(
    const float* x,
    const float* up,
    float* y,
    int n
);

void swiglu_f16_neon(
    const fp16_t* gate,
    const fp16_t* up,
    fp16_t* y,
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

Status attention_f16_gptq_neon(
    const Tensor& hidden_states,
    Tensor& attn_output,
    const GPTQInt8Weight& q_proj,
    const GPTQInt8Weight& k_proj,
    const GPTQInt8Weight& v_proj,
    const GPTQInt8Weight& o_proj,
    const fp16_t* q_bias,
    const fp16_t* k_bias,
    const fp16_t* v_bias,
    const fp16_t* cos_ptr,
    const fp16_t* sin_ptr,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
);

Status attention_f16_gptq_prefill_neon(
    const Tensor& hidden_states,
    Tensor& attn_output,
    const GPTQInt8Weight& q_proj,
    const GPTQInt8Weight& k_proj,
    const GPTQInt8Weight& v_proj,
    const GPTQInt8Weight& o_proj,
    const fp16_t* q_bias,
    const fp16_t* k_bias,
    const fp16_t* v_bias,
    const fp16_t* cos_base,
    const fp16_t* sin_base,
    KVCache& kv_cache,
    int layer_id,
    int start_pos,
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

Status ffn_f16_gptq_neon(
    const Tensor& hidden_states,
    Tensor& ffn_output,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    const GPTQInt8Weight& down_proj,
    const FFNConfig& config,
    Workspace& workspace
);

Status ffn_f16_gptq_batch_neon(
    const Tensor& hidden_states,
    Tensor& ffn_output,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    const GPTQInt8Weight& down_proj,
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

Status qwen_block_f16_gptq_neon(
    Tensor& hidden_states,
    const Tensor& norm1_weight,
    const GPTQInt8Weight& q_proj,
    const GPTQInt8Weight& k_proj,
    const GPTQInt8Weight& v_proj,
    const GPTQInt8Weight& o_proj,
    const fp16_t* q_bias,
    const fp16_t* k_bias,
    const fp16_t* v_bias,
    const fp16_t* cos_ptr,
    const fp16_t* sin_ptr,
    const Tensor& norm2_weight,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    const GPTQInt8Weight& down_proj,
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& attn_config,
    const FFNConfig& ffn_config,
    float rms_norm_eps,
    Workspace& workspace
);

Status qwen_block_f16_gptq_prefill_neon(
    Tensor& hidden_states,
    const Tensor& norm1_weight,
    const GPTQInt8Weight& q_proj,
    const GPTQInt8Weight& k_proj,
    const GPTQInt8Weight& v_proj,
    const GPTQInt8Weight& o_proj,
    const fp16_t* q_bias,
    const fp16_t* k_bias,
    const fp16_t* v_bias,
    const fp16_t* cos_base,
    const fp16_t* sin_base,
    const Tensor& norm2_weight,
    const GPTQInt8Weight& gate_proj,
    const GPTQInt8Weight& up_proj,
    const GPTQInt8Weight& down_proj,
    KVCache& kv_cache,
    int layer_id,
    int start_pos,
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
