#include "backends/cpu/arm_neon/neon_ops.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/tensor.h"
#include <cmath>
#include <arm_neon.h>

namespace llm_engine {
namespace arm_neon {

Status attention_neon(
    const Tensor& hidden_states, // [1, hidden_dim]
    Tensor& attn_output,         // [1, hidden_dim]
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o, 
    const float* cos_ptr, const float* sin_ptr, 
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
) {
    // ==========================================
    // 0. Workspace 内存切分
    // ==========================================
    float* ws_base = static_cast<float*>(workspace.data()); 
    
    int q_size = config.num_q_heads * config.head_dim;
    int k_size = config.num_kv_heads * config.head_dim;
    int v_size = config.num_kv_heads * config.head_dim;
    int max_seq_len = kv_cache.get_max_seq_len();
    int num_rep = config.num_q_heads / config.num_kv_heads;
    
    float* q_proj_ptr = ws_base;                ws_base += q_size;
    float* k_proj_ptr = ws_base;                ws_base += k_size;
    float* v_proj_ptr = ws_base;                ws_base += v_size;
    
    // 💡 修复问题2：Score 大小改为 num_rep * max_seq_len，满足成组计算
    float* score_ptr  = ws_base;                ws_base += num_rep * max_seq_len; 
    float* attn_out_ptr = ws_base;              ws_base += q_size; 
    float* matmul_ws_ptr = ws_base; 

    // 💡 修复问题1：全面回归纯净的 2D 形状
    Tensor Q_proj({1, q_size}, q_proj_ptr);
    Tensor K_proj({1, k_size}, k_proj_ptr);
    Tensor V_proj({1, v_size}, v_proj_ptr);
    Tensor Attn_Out_Buf({1, q_size}, attn_out_ptr);

    // ==========================================
    // 1 & 2. QKV 投影与 RoPE
    // ==========================================
    matmul_neon(hidden_states, w_q, Q_proj, matmul_ws_ptr);
    matmul_neon(hidden_states, w_k, K_proj, matmul_ws_ptr);
    matmul_neon(hidden_states, w_v, V_proj, matmul_ws_ptr);

    rope_neon(q_proj_ptr, cos_ptr, sin_ptr, q_size);
    rope_neon(k_proj_ptr, cos_ptr, sin_ptr, k_size);

    // ==========================================
    // 3. 更新 KV Cache
    // ==========================================
    kv_cache.update(layer_id, current_pos, k_proj_ptr, v_proj_ptr);

    // ==========================================
    // 4. GQA 核心计算循环 (💡 修复问题3：按 KV Head 循环！)
    // ==========================================
    float scale = 1.0f / std::sqrt((float)config.head_dim);
    int current_seq_len = current_pos + 1; 

    for (int kv_head = 0; kv_head < config.num_kv_heads; ++kv_head) {
        
        // --- 准备 K 和 V (完全纯净的 2D Tensor) ---
        float* k_cache_ptr = kv_cache.get_k_head_ptr(layer_id, kv_head);
        float* v_cache_ptr = kv_cache.get_v_head_ptr(layer_id, kv_head);
        
        // 💡 修复问题1：直接使用 [seq_len, head_dim] 2D 形状
        Tensor K_cache({current_seq_len, config.head_dim}, k_cache_ptr);
        Tensor V_cache({current_seq_len, config.head_dim}, v_cache_ptr);

        // --- 准备成组的 Q (Grouped Q) ---
        // 提取与此 kv_head 绑定的 num_rep 个连续 Q head
        float* q_group_ptr = q_proj_ptr + kv_head * num_rep * config.head_dim;
        // 把它们看作一个矩阵: [num_rep, head_dim]
        Tensor Q_group({num_rep, config.head_dim}, q_group_ptr);

        // --- 准备 Score ---
        // Score 尺寸为 [num_rep, current_seq_len]
        Tensor Score({num_rep, current_seq_len}, score_ptr);

        // A. 极致性能：一次计算 Score = Q_group * K_cache^T
        // 因为 K_cache 是复用的，底层会自动 pack 唯一一次 K，然后服务于 num_rep 行的 Q！
        matmul_neon(Q_group, K_cache, Score, matmul_ws_ptr, true);

        // B. Score 缩放 (按总元素个数循环)
        int total_score_elements = num_rep * current_seq_len;
        int i = 0;
        float32x4_t v_scale = vdupq_n_f32(scale);
        for (; i <= total_score_elements - 4; i += 4) {
            float32x4_t v_score = vld1q_f32(score_ptr + i);
            v_score = vmulq_f32(v_score, v_scale);
            vst1q_f32(score_ptr + i, v_score);
        }
        for (; i < total_score_elements; i++) {
            score_ptr[i] *= scale;
        }

        // C. Softmax
        // 你写的 softmax_neon 内部是按最后维度 (seq_len) 循环的
        // 所以传 [num_rep, seq_len] 进去，它会自动按行对 num_rep 组各自做 softmax，完美兼容！
        softmax_neon(Score, Score);

        // D. 极致性能：一次计算 Output = Score * V_cache
        // 同理，V 只被 pack 唯一一次
        float* out_group_ptr = attn_out_ptr + kv_head * num_rep * config.head_dim;
        Tensor Out_group({num_rep, config.head_dim}, out_group_ptr);
        
        matmul_neon(Score, V_cache, Out_group, matmul_ws_ptr, false);
    }

    // ==========================================
    // 5. 最终输出投影
    // ==========================================
    matmul_neon(Attn_Out_Buf, w_o, attn_output, matmul_ws_ptr);

    return Status::SUCCESS;
}

} // namespace arm_neon
} // namespace llm_engine