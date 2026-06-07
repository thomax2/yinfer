#include "llm_engine/memory/kv_cache.h"
#include <cstring>

namespace llm_engine {

KVCache::KVCache(int num_layers, int max_seq_len, int num_kv_heads, int head_dim)
    : num_layers(num_layers), max_seq_len(max_seq_len), 
      num_kv_heads(num_kv_heads), head_dim(head_dim) {
    size_t total_elements = (size_t)num_layers * num_kv_heads * max_seq_len * head_dim;
    k_cache.resize(total_elements, (fp16_t)0);
    v_cache.resize(total_elements, (fp16_t)0);
}

void KVCache::update(int layer_id, int current_pos, const fp16_t* k_curr, const fp16_t* v_curr) {
    // k_curr 里的物理布局是 [num_kv_heads, head_dim]
    // 我们需要将其“拆散”，放入 [num_kv_heads, max_seq_len, head_dim] 的对应位置
    for (int h = 0; h < num_kv_heads; ++h) {
        // 目标地址：跳过前面的层和头，定位到当前头的起始，再加上 current_pos 的偏移
        fp16_t* k_dest = get_k_head_ptr(layer_id, h) + current_pos * head_dim;
        fp16_t* v_dest = get_v_head_ptr(layer_id, h) + current_pos * head_dim;
        
        // 源地址：直接按头偏移
        const fp16_t* k_src = k_curr + h * head_dim;
        const fp16_t* v_src = v_curr + h * head_dim;
        
        // 每次拷贝一个 Head_dim 的大小
        std::memcpy(k_dest, k_src, head_dim * sizeof(fp16_t));
        std::memcpy(v_dest, v_src, head_dim * sizeof(fp16_t));
    }
}

fp16_t* KVCache::get_k_head_ptr(int layer_id, int kv_head_id) {
    size_t layer_stride = (size_t)num_kv_heads * max_seq_len * head_dim;
    size_t head_stride = (size_t)max_seq_len * head_dim;
    return k_cache.data() + layer_id * layer_stride + kv_head_id * head_stride;
}

fp16_t* KVCache::get_v_head_ptr(int layer_id, int kv_head_id) {
    size_t layer_stride = (size_t)num_kv_heads * max_seq_len * head_dim;
    size_t head_stride = (size_t)max_seq_len * head_dim;
    return v_cache.data() + layer_id * layer_stride + kv_head_id * head_stride;
}

} // namespace llm_engine
