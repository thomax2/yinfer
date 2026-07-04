#pragma once

#include <cstddef>
#include <vector>

#include "llm_engine/tensor.h"

namespace llm_engine {

enum class KVCacheLayout {
    CONTIGUOUS,
    PAGED
};

struct PagedKVView {
    int block_size = 0;
    int seq_len = 0;
    int num_layers = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int num_physical_blocks = 0;
    const std::vector<int>* block_table = nullptr;
};

class KVCache {
public:
    KVCache(int num_layers, int max_seq_len, int num_kv_heads, int head_dim);

    void update(int layer_id, int current_pos, const fp16_t* k_curr, const fp16_t* v_curr);

    fp16_t* get_k_head_ptr(int layer_id, int kv_head_id);
    fp16_t* get_v_head_ptr(int layer_id, int kv_head_id);

    int get_max_seq_len() const { return max_seq_len; }
    bool is_paged() const { return layout_ == KVCacheLayout::PAGED; }
    int block_size() const { return block_size_; }
    int num_blocks() const { return num_logical_blocks_; }
    int allocated_blocks() const { return num_physical_blocks_; }
    int max_written_pos() const { return max_written_pos_; }
    size_t bytes_per_block() const;
    size_t total_kv_bytes() const;
    bool valid_physical_block(int block_id) const;
    void clear_physical_block(int block_id);
    bool get_active_paged_view(PagedKVView* out) const;
    const fp16_t* get_paged_k_token_ptr(
        int physical_block, int layer_id, int kv_head_id, int offset_in_block) const;
    const fp16_t* get_paged_v_token_ptr(
        int physical_block, int layer_id, int kv_head_id, int offset_in_block) const;
    const fp16_t* get_paged_k_token_ptr_by_pos(
        int layer_id, int kv_head_id, int logical_pos) const;
    const fp16_t* get_paged_v_token_ptr_by_pos(
        int layer_id, int kv_head_id, int logical_pos) const;
    const fp16_t* raw_k_pages() const;
    const fp16_t* raw_v_pages() const;
    size_t paged_elements_per_block() const;

    void set_active_sequence(std::vector<int>* block_table, int* max_written_pos);
    void clear_active_sequence();

    void clear();

private:
    fp16_t* get_contiguous_k_head_ptr(int layer_id, int kv_head_id);
    fp16_t* get_contiguous_v_head_ptr(int layer_id, int kv_head_id);

    fp16_t* paged_k_token_ptr(int physical_block, int layer_id, int kv_head_id, int offset_in_block);
    fp16_t* paged_v_token_ptr(int physical_block, int layer_id, int kv_head_id, int offset_in_block);
    const fp16_t* paged_k_token_ptr(int physical_block, int layer_id, int kv_head_id, int offset_in_block) const;
    const fp16_t* paged_v_token_ptr(int physical_block, int layer_id, int kv_head_id, int offset_in_block) const;

    fp16_t* gather_head(bool gather_k, int layer_id, int kv_head_id);

    bool debug_enabled() const;
    bool debug_verbose_enabled() const;
    void debug_log_config() const;
    void debug_log_gather(bool gather_k, int layer_id, int kv_head_id, int valid_len);

    int num_layers;
    int max_seq_len;
    int num_kv_heads;
    int head_dim;

    KVCacheLayout layout_ = KVCacheLayout::CONTIGUOUS;
    int block_size_ = 16;
    int num_logical_blocks_ = 0;
    int num_physical_blocks_ = 0;
    int max_written_pos_ = -1;
    int gather_log_count_ = 0;
    std::vector<int>* active_block_table_ = nullptr;
    int* active_max_written_pos_ = nullptr;

    std::vector<fp16_t> k_cache;
    std::vector<fp16_t> v_cache;

    std::vector<int> block_table_;
    std::vector<fp16_t> k_pages_;
    std::vector<fp16_t> v_pages_;
    std::vector<fp16_t> gather_k_buffer_;
    std::vector<fp16_t> gather_v_buffer_;
};

} // namespace llm_engine
