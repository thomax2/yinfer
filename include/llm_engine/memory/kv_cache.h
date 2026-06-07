#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>

#include "llm_engine/tensor.h"

namespace llm_engine {

class KVCache {
public:
    // 构造函数：预分配最大容量的内存
    KVCache(int num_layers, int max_seq_len, int num_kv_heads, int head_dim);

    // 将当前生成的 Token 的 K, V 写入到 Cache 中
    // k_curr, v_curr 是当前 token 的输出，形状为 [num_kv_heads, head_dim]
    // current_pos 是当前 Token 在序列中的位置索引 (例如：第一个字是 0，第二个字是 1)
    void update(int layer_id, int current_pos, const fp16_t* k_curr, const fp16_t* v_curr);

    // 获取某一层的 K 和 V 缓存的首地址，用于传给 BMM 算子
    fp16_t* get_k_head_ptr(int layer_id, int kv_head_id);
    fp16_t* get_v_head_ptr(int layer_id, int kv_head_id);
    
    int get_max_seq_len() const { return max_seq_len; }

    void clear() {
        std::fill(k_cache.begin(), k_cache.end(), (fp16_t)0);
        std::fill(v_cache.begin(), v_cache.end(), (fp16_t)0);
    }

private:
    int num_layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;
    
    // 我们用一维 std::vector 来模拟高维张量：
    // 逻辑形状为: [num_layers, max_seq_len, num_kv_heads, head_dim]
    std::vector<fp16_t> k_cache;
    std::vector<fp16_t> v_cache;
};

} // namespace llm_enginematmul_neon
