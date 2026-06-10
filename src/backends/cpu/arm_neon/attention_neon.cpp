#include "backends/cpu/arm_neon/neon_ops.h"
#include "llm_engine/memory/kv_cache.h"
#include "llm_engine/memory/workspace.h"
#include "llm_engine/tensor.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <arm_neon.h>

namespace llm_engine {
namespace arm_neon {

namespace {
bool debug_numeric_enabled() {
    const char* v = std::getenv("LLM_DEBUG_NUMERIC");
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

void dump_f16_stats(const char* tag, int layer_id, const fp16_t* data, int n) {
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        float v = (float)data[i];
        if (std::isnan(v)) {
            nan_count++;
        } else if (!std::isfinite(v)) {
            inf_count++;
        } else {
            finite_count++;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    std::cerr << "[NUMERIC] layer=" << layer_id
              << " stage=" << tag
              << " finite=" << finite_count
              << " nan=" << nan_count
              << " inf=" << inf_count
              << " min=" << min_v
              << " max=" << max_v
              << std::endl;
}
} // namespace

Status attention_neon(
    const Tensor& hidden_states, // [1, hidden_dim]
    Tensor& attn_output,         // [1, hidden_dim]
    const Tensor& w_q, const Tensor& w_k, const Tensor& w_v, const Tensor& w_o,
    const Tensor& w_q_pack, const Tensor& w_k_pack, const Tensor& w_v_pack, const Tensor& w_o_pack,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* cos_ptr, const float* sin_ptr, 
    KVCache& kv_cache,
    int layer_id,
    int current_pos,
    const AttentionConfig& config,
    Workspace& workspace
) {

    int num_tokens = hidden_states.shape[0];
    int hidden_dim = hidden_states.shape[1];

    int q_out_dim = config.num_q_heads * config.head_dim;
    int kv_out_dim = config.num_kv_heads * config.head_dim;

    // ==========================================
    // 🛡️ 防御 1：严格的 Shape 校验
    // ==========================================
    if (w_q.shape[1] != q_out_dim || w_k.shape[1] != kv_out_dim || w_v.shape[1] != kv_out_dim) {
        return Status::INVALID_ARGUMENT; 
    }
    // 严格绑定输入特征维度，防止配置错误
    if (w_o.shape[0] != q_out_dim || w_o.shape[1] != hidden_dim) {
        return Status::INVALID_ARGUMENT;
    }

    // ==========================================
    // 🛡️ 防御 2：极致优化的 Workspace 大小检查
    // ==========================================
    // 计算 Q, K, V 投影所需的空间
    size_t q_bytes = num_tokens * q_out_dim * sizeof(float);
    size_t k_bytes = num_tokens * kv_out_dim * sizeof(float);
    size_t v_bytes = num_tokens * kv_out_dim * sizeof(float);

    // 计算 GQA 复用组大小
    int num_rep = config.num_q_heads / config.num_kv_heads;
    
    // Score 缓冲区：只需要存当前 GQA 组的分数 (极限省内存！)
    size_t score_bytes = num_tokens * num_rep * kv_cache.get_max_seq_len() * sizeof(float);

    // Attention 融合后的输出缓存 (乘 w_o 之前)
    size_t attn_out_bytes = num_tokens * q_out_dim * sizeof(float);

    // 给底层的 GEMM 预留汇编指令 Pack 打包的缓存区
    size_t pack_bytes = 1024 * 1024; // 1MB，够跑 0.5B 模型了

    size_t required = q_bytes + k_bytes + v_bytes + score_bytes + attn_out_bytes + pack_bytes;

    if (workspace.size() < required) {
        return Status::OUT_OF_MEMORY;
    }

    // ==========================================
    // 0. Workspace 内存切分
    // ==========================================
    float* ws_base = static_cast<float*>(workspace.data()); 
    
    int q_size = config.num_q_heads * config.head_dim;
    int k_size = config.num_kv_heads * config.head_dim;
    int v_size = config.num_kv_heads * config.head_dim;
    int max_seq_len = kv_cache.get_max_seq_len();
    
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


    // w_q 必须是 [hidden_dim, q_size]
    assert(w_q.shape[1] == q_size);
    assert(w_k.shape[1] == k_size);
    assert(w_v.shape[1] == v_size);


    // ==========================================
    // 1 & 2. QKV 投影与 RoPE
    // ==========================================
    if (num_tokens == 1 && w_q_pack.data && w_k_pack.data && w_v_pack.data) {
        Status status = linear_decode_prepacked_neon(
            hidden_states.ptr<float>(), w_q_pack.ptr<float>(), q_proj_ptr,
            hidden_dim, q_size, q_bias
        );
        if (status != Status::SUCCESS) return status;

        status = linear_decode_prepacked_neon(
            hidden_states.ptr<float>(), w_k_pack.ptr<float>(), k_proj_ptr,
            hidden_dim, k_size, k_bias
        );
        if (status != Status::SUCCESS) return status;

        status = linear_decode_prepacked_neon(
            hidden_states.ptr<float>(), w_v_pack.ptr<float>(), v_proj_ptr,
            hidden_dim, v_size, v_bias
        );
        if (status != Status::SUCCESS) return status;
    } else {
        matmul_neon(hidden_states, w_q, Q_proj, matmul_ws_ptr, false, q_bias);
        matmul_neon(hidden_states, w_k, K_proj, matmul_ws_ptr, false, k_bias);
        matmul_neon(hidden_states, w_v, V_proj, matmul_ws_ptr, false, v_bias);
    }

    // rope_neon(q_proj_ptr, cos_ptr, sin_ptr, q_size);
    // rope_neon(k_proj_ptr, cos_ptr, sin_ptr, k_size);
    // 遍历每一个 Q Head，分别应用长度为 head_dim 的 ROPE
    for(int h = 0; h < config.num_q_heads; ++h){
        rope_neon(q_proj_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }

    // 遍历每一个 Kv Head，分别应用长度为 head dim 的 ROPE
    for (int h = 0; h < config.num_kv_heads; ++h){
        rope_neon(k_proj_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }

    // ==========================================
    // 3. 更新 KV Cache
    // ==========================================
    kv_cache.update(
        layer_id,
        current_pos,
        reinterpret_cast<const fp16_t*>(k_proj_ptr),
        reinterpret_cast<const fp16_t*>(v_proj_ptr));

    // ==========================================
    // 4. GQA 核心计算循环 (💡 修复问题3：按 KV Head 循环！)
    // ==========================================
    float scale = 1.0f / std::sqrt((float)config.head_dim);
    int current_seq_len = current_pos + 1; 

    for (int kv_head = 0; kv_head < config.num_kv_heads; ++kv_head) {
        
        // --- 准备 K 和 V (完全纯净的 2D Tensor) ---
        float* k_cache_ptr = reinterpret_cast<float*>(kv_cache.get_k_head_ptr(layer_id, kv_head));
        float* v_cache_ptr = reinterpret_cast<float*>(kv_cache.get_v_head_ptr(layer_id, kv_head));
        
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
        matmul_neon(Q_group, K_cache, Score, matmul_ws_ptr, true, nullptr);

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
        
        matmul_neon(Score, V_cache, Out_group, matmul_ws_ptr, false, nullptr);
    }

    // ==========================================
    // 5. 最终输出投影
    // ==========================================
    if (num_tokens == 1 && w_o_pack.data) {
        Status status = linear_decode_prepacked_neon(
            attn_out_ptr, w_o_pack.ptr<float>(), attn_output.ptr<float>(),
            q_size, hidden_dim, nullptr
        );
        if (status != Status::SUCCESS) return status;
    } else {
        matmul_neon(Attn_Out_Buf, w_o, attn_output, matmul_ws_ptr, false, nullptr);
    }

    return Status::SUCCESS;
}

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
) {
    Tensor empty;
    return attention_neon(
        hidden_states, attn_output,
        w_q, w_k, w_v, w_o,
        empty, empty, empty, empty,
        q_bias, k_bias, v_bias,
        cos_ptr, sin_ptr,
        kv_cache, layer_id, current_pos, config, workspace
    );
}

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
) {
    bool debug_numeric = debug_numeric_enabled();
    if (hidden_states.dtype != DataType::FP16 || attn_output.dtype != DataType::FP16) {
        return Status::INVALID_ARGUMENT;
    }

    int hidden_dim = hidden_states.shape[1];
    int q_size = config.num_q_heads * config.head_dim;
    int kv_size = config.num_kv_heads * config.head_dim;
    int num_rep = config.num_q_heads / config.num_kv_heads;
    int max_seq_len = kv_cache.get_max_seq_len();

    size_t q_bytes = align_size((size_t)q_size * sizeof(fp16_t));
    size_t k_bytes = align_size((size_t)kv_size * sizeof(fp16_t));
    size_t v_bytes = align_size((size_t)kv_size * sizeof(fp16_t));
    size_t score_bytes = align_size((size_t)num_rep * max_seq_len * sizeof(fp16_t));
    size_t attn_bytes = align_size((size_t)q_size * sizeof(fp16_t));
    size_t matmul_bytes = align_size(32ULL * 1024 * 1024);
    size_t required = q_bytes + k_bytes + v_bytes + score_bytes + attn_bytes + matmul_bytes;
    if (workspace.size() < required) {
        return Status::OUT_OF_MEMORY;
    }

    char* base = static_cast<char*>(workspace.data());
    base = align_ptr(base); fp16_t* q_ptr = reinterpret_cast<fp16_t*>(base); base += q_bytes;
    base = align_ptr(base); fp16_t* k_ptr = reinterpret_cast<fp16_t*>(base); base += k_bytes;
    base = align_ptr(base); fp16_t* v_ptr = reinterpret_cast<fp16_t*>(base); base += v_bytes;
    base = align_ptr(base); fp16_t* score_ptr = reinterpret_cast<fp16_t*>(base); base += score_bytes;
    base = align_ptr(base); fp16_t* attn_out_ptr = reinterpret_cast<fp16_t*>(base); base += attn_bytes;
    base = align_ptr(base); fp16_t* matmul_ws = reinterpret_cast<fp16_t*>(base);

    Status status = linear_gptq_int8_decode_neon(
        hidden_states.ptr<fp16_t>(), q_proj, q_ptr, q_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    status = linear_gptq_int8_decode_neon(
        hidden_states.ptr<fp16_t>(), k_proj, k_ptr, k_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    status = linear_gptq_int8_decode_neon(
        hidden_states.ptr<fp16_t>(), v_proj, v_ptr, v_bias, nullptr, 0);
    if (status != Status::SUCCESS) return status;
    if (debug_numeric) {
        dump_f16_stats("attn_norm_in", layer_id, hidden_states.ptr<fp16_t>(), hidden_dim);
        dump_f16_stats("q_decode", layer_id, q_ptr, q_size);
        dump_f16_stats("k_decode", layer_id, k_ptr, kv_size);
        dump_f16_stats("v_decode", layer_id, v_ptr, kv_size);
    }

    for (int h = 0; h < config.num_q_heads; ++h) {
        rope_f16_neon(q_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }
    for (int h = 0; h < config.num_kv_heads; ++h) {
        rope_f16_neon(k_ptr + h * config.head_dim, cos_ptr, sin_ptr, config.head_dim);
    }
    if (debug_numeric) {
        dump_f16_stats("q_rope", layer_id, q_ptr, q_size);
        dump_f16_stats("k_rope", layer_id, k_ptr, kv_size);
    }

    kv_cache.update(layer_id, current_pos, k_ptr, v_ptr);

    int current_seq_len = current_pos + 1;
    float scale = 1.0f / std::sqrt((float)config.head_dim);

    for (int kv_head = 0; kv_head < config.num_kv_heads; ++kv_head) {
        fp16_t* k_cache_ptr = kv_cache.get_k_head_ptr(layer_id, kv_head);
        fp16_t* v_cache_ptr = kv_cache.get_v_head_ptr(layer_id, kv_head);
        fp16_t* q_group_ptr = q_ptr + kv_head * num_rep * config.head_dim;
        fp16_t* out_group_ptr = attn_out_ptr + kv_head * num_rep * config.head_dim;

        Tensor Q_group({num_rep, config.head_dim}, q_group_ptr, DataType::FP16);
        Tensor K_cache({current_seq_len, config.head_dim}, k_cache_ptr, DataType::FP16);
        Tensor V_cache({current_seq_len, config.head_dim}, v_cache_ptr, DataType::FP16);
        Tensor Score({num_rep, current_seq_len}, score_ptr, DataType::FP16);
        Tensor Out_group({num_rep, config.head_dim}, out_group_ptr, DataType::FP16);

        status = matmul_f16_neon(Q_group, K_cache, Score, matmul_ws, true, nullptr);
        if (status != Status::SUCCESS) return status;
        if (debug_numeric) {
            dump_f16_stats("score_matmul", layer_id, score_ptr, num_rep * current_seq_len);
        }

        int total_score = num_rep * current_seq_len;
        float32x4_t v_scale = vdupq_n_f32(scale);
        int i = 0;
        for (; i <= total_score - 8; i += 8) {
            float16x8_t h = vld1q_f16(score_ptr + i);
            float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(h)), v_scale);
            float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(h)), v_scale);
            vst1q_f16(score_ptr + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
        }
        for (; i < total_score; ++i) {
            score_ptr[i] = (fp16_t)((float)score_ptr[i] * scale);
        }
        if (debug_numeric) {
            dump_f16_stats("score_scaled", layer_id, score_ptr, total_score);
        }

        status = softmax_f16_neon(Score, Score);
        if (status != Status::SUCCESS) return status;
        if (debug_numeric) {
            dump_f16_stats("score_softmax", layer_id, score_ptr, total_score);
        }

        status = matmul_f16_neon(Score, V_cache, Out_group, matmul_ws, false, nullptr);
        if (status != Status::SUCCESS) return status;
        if (debug_numeric) {
            dump_f16_stats("attn_group_out", layer_id, out_group_ptr, num_rep * config.head_dim);
        }
    }

    status = linear_gptq_int8_decode_neon(
        attn_out_ptr, o_proj, attn_output.ptr<fp16_t>(), nullptr, nullptr, 0);
    if (debug_numeric) {
        dump_f16_stats("o_proj_out", layer_id, attn_output.ptr<fp16_t>(), hidden_dim);
    }
    return status;
}

} // namespace arm_neon
} // namespace llm_engine
