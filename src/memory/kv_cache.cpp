#include "llm_engine/memory/kv_cache.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace llm_engine {

namespace {

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

int env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;

    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v || x <= 0) return default_value;
    return static_cast<int>(x);
}

} // namespace

KVCache::KVCache(int num_layers, int max_seq_len, int num_kv_heads, int head_dim)
    : num_layers(num_layers), max_seq_len(max_seq_len), 
      num_kv_heads(num_kv_heads), head_dim(head_dim) {
    layout_ = env_flag("LLM_PAGED_KV") ? KVCacheLayout::PAGED : KVCacheLayout::CONTIGUOUS;

    if (layout_ == KVCacheLayout::PAGED) {
        block_size_ = env_int("LLM_KV_BLOCK_SIZE", 16);
        num_logical_blocks_ = (max_seq_len + block_size_ - 1) / block_size_;
        num_physical_blocks_ = env_int("LLM_KV_MAX_BLOCKS", num_logical_blocks_);

        block_table_.resize((size_t)num_logical_blocks_);
        for (int i = 0; i < num_logical_blocks_; ++i) {
            block_table_[(size_t)i] = i < num_physical_blocks_ ? i : -1;
        }

        size_t page_elements =
            (size_t)num_physical_blocks_ * num_layers * num_kv_heads * block_size_ * head_dim;
        k_pages_.resize(page_elements, (fp16_t)0);
        v_pages_.resize(page_elements, (fp16_t)0);
        gather_k_buffer_.resize((size_t)max_seq_len * head_dim, (fp16_t)0);
        gather_v_buffer_.resize((size_t)max_seq_len * head_dim, (fp16_t)0);
    } else {
        size_t total_elements = (size_t)num_layers * num_kv_heads * max_seq_len * head_dim;
        k_cache.resize(total_elements, (fp16_t)0);
        v_cache.resize(total_elements, (fp16_t)0);
    }

    debug_log_config();
}

void KVCache::update(int layer_id, int current_pos, const fp16_t* k_curr, const fp16_t* v_curr) {
    if (current_pos < 0 || current_pos >= max_seq_len ||
        layer_id < 0 || layer_id >= num_layers) {
        if (debug_enabled()) {
            std::cerr << "[KVCache] update ignored"
                      << " layer=" << layer_id
                      << " pos=" << current_pos
                      << std::endl;
        }
        return;
    }

    if (layout_ == KVCacheLayout::CONTIGUOUS) {
        for (int h = 0; h < num_kv_heads; ++h) {
            fp16_t* k_dest = get_contiguous_k_head_ptr(layer_id, h) + (size_t)current_pos * head_dim;
            fp16_t* v_dest = get_contiguous_v_head_ptr(layer_id, h) + (size_t)current_pos * head_dim;
            const fp16_t* k_src = k_curr + (size_t)h * head_dim;
            const fp16_t* v_src = v_curr + (size_t)h * head_dim;
            std::memcpy(k_dest, k_src, (size_t)head_dim * sizeof(fp16_t));
            std::memcpy(v_dest, v_src, (size_t)head_dim * sizeof(fp16_t));
        }
        max_written_pos_ = std::max(max_written_pos_, current_pos);
        return;
    }

    std::vector<int>* active_table = active_block_table_ ? active_block_table_ : &block_table_;
    int logical_block = current_pos / block_size_;
    int offset_in_block = current_pos % block_size_;
    if (logical_block < 0 || logical_block >= static_cast<int>(active_table->size())) {
        if (debug_enabled()) {
            std::cerr << "[KVCache] update missing logical block"
                      << " logical_block=" << logical_block
                      << " table_size=" << active_table->size()
                      << std::endl;
        }
        return;
    }
    int physical_block = (*active_table)[(size_t)logical_block];
    if (physical_block < 0 || physical_block >= num_physical_blocks_) {
        if (debug_enabled()) {
            std::cerr << "[KVCache] update missing physical block"
                      << " logical_block=" << logical_block
                      << " physical_block=" << physical_block
                      << std::endl;
        }
        return;
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        fp16_t* k_dest = paged_k_token_ptr(physical_block, layer_id, h, offset_in_block);
        fp16_t* v_dest = paged_v_token_ptr(physical_block, layer_id, h, offset_in_block);
        const fp16_t* k_src = k_curr + (size_t)h * head_dim;
        const fp16_t* v_src = v_curr + (size_t)h * head_dim;
        std::memcpy(k_dest, k_src, (size_t)head_dim * sizeof(fp16_t));
        std::memcpy(v_dest, v_src, (size_t)head_dim * sizeof(fp16_t));
    }
    if (active_max_written_pos_) {
        *active_max_written_pos_ = std::max(*active_max_written_pos_, current_pos);
    } else {
        max_written_pos_ = std::max(max_written_pos_, current_pos);
    }
}

fp16_t* KVCache::get_k_head_ptr(int layer_id, int kv_head_id) {
    if (layout_ == KVCacheLayout::PAGED) {
        return gather_head(true, layer_id, kv_head_id);
    }
    return get_contiguous_k_head_ptr(layer_id, kv_head_id);
}

fp16_t* KVCache::get_v_head_ptr(int layer_id, int kv_head_id) {
    if (layout_ == KVCacheLayout::PAGED) {
        return gather_head(false, layer_id, kv_head_id);
    }
    return get_contiguous_v_head_ptr(layer_id, kv_head_id);
}

void KVCache::clear() {
    if (layout_ == KVCacheLayout::PAGED) {
        std::fill(k_pages_.begin(), k_pages_.end(), (fp16_t)0);
        std::fill(v_pages_.begin(), v_pages_.end(), (fp16_t)0);
        std::fill(gather_k_buffer_.begin(), gather_k_buffer_.end(), (fp16_t)0);
        std::fill(gather_v_buffer_.begin(), gather_v_buffer_.end(), (fp16_t)0);
    } else {
        std::fill(k_cache.begin(), k_cache.end(), (fp16_t)0);
        std::fill(v_cache.begin(), v_cache.end(), (fp16_t)0);
    }
    max_written_pos_ = -1;
    gather_log_count_ = 0;

    if (debug_enabled()) {
        std::cerr << "[KVCache] reset layout="
                  << (is_paged() ? "paged" : "contiguous")
                  << std::endl;
    }
}

size_t KVCache::bytes_per_block() const {
    if (!is_paged()) {
        return 0;
    }
    return (size_t)num_layers * 2 * num_kv_heads * block_size_ * head_dim * sizeof(fp16_t);
}

size_t KVCache::total_kv_bytes() const {
    if (is_paged()) {
        return bytes_per_block() * (size_t)num_physical_blocks_;
    }
    return (size_t)num_layers * 2 * num_kv_heads * max_seq_len * head_dim * sizeof(fp16_t);
}

bool KVCache::valid_physical_block(int block_id) const {
    return is_paged() && block_id >= 0 && block_id < num_physical_blocks_;
}

void KVCache::clear_physical_block(int block_id) {
    if (!valid_physical_block(block_id)) {
        return;
    }

    size_t elements_per_plane =
        (size_t)num_layers * num_kv_heads * block_size_ * head_dim;
    size_t begin = (size_t)block_id * elements_per_plane;
    size_t end = begin + elements_per_plane;
    if (end > k_pages_.size() || end > v_pages_.size()) {
        return;
    }

    std::fill(k_pages_.begin() + begin, k_pages_.begin() + end, (fp16_t)0);
    std::fill(v_pages_.begin() + begin, v_pages_.begin() + end, (fp16_t)0);
}

bool KVCache::get_active_paged_view(PagedKVView* out) const {
    if (!out || !is_paged()) {
        return false;
    }

    const std::vector<int>* table = active_block_table_ ? active_block_table_ : &block_table_;
    int max_written = active_max_written_pos_ ? *active_max_written_pos_ : max_written_pos_;
    int seq_len = max_written + 1;
    if (seq_len <= 0) {
        return false;
    }
    if (seq_len > max_seq_len) {
        seq_len = max_seq_len;
    }

    out->block_size = block_size_;
    out->seq_len = seq_len;
    out->num_layers = num_layers;
    out->num_kv_heads = num_kv_heads;
    out->head_dim = head_dim;
    out->num_physical_blocks = num_physical_blocks_;
    out->block_table = table;
    return table != nullptr && !table->empty();
}

const fp16_t* KVCache::get_paged_k_token_ptr(
    int physical_block, int layer_id, int kv_head_id, int offset_in_block) const {
    if (!valid_physical_block(physical_block) ||
        layer_id < 0 || layer_id >= num_layers ||
        kv_head_id < 0 || kv_head_id >= num_kv_heads ||
        offset_in_block < 0 || offset_in_block >= block_size_) {
        return nullptr;
    }
    return paged_k_token_ptr(physical_block, layer_id, kv_head_id, offset_in_block);
}

const fp16_t* KVCache::get_paged_v_token_ptr(
    int physical_block, int layer_id, int kv_head_id, int offset_in_block) const {
    if (!valid_physical_block(physical_block) ||
        layer_id < 0 || layer_id >= num_layers ||
        kv_head_id < 0 || kv_head_id >= num_kv_heads ||
        offset_in_block < 0 || offset_in_block >= block_size_) {
        return nullptr;
    }
    return paged_v_token_ptr(physical_block, layer_id, kv_head_id, offset_in_block);
}

const fp16_t* KVCache::get_paged_k_token_ptr_by_pos(
    int layer_id, int kv_head_id, int logical_pos) const {
    PagedKVView view;
    if (!get_active_paged_view(&view) ||
        logical_pos < 0 || logical_pos >= view.seq_len ||
        !view.block_table) {
        return nullptr;
    }
    int logical_block = logical_pos / block_size_;
    int offset_in_block = logical_pos % block_size_;
    if (logical_block < 0 || logical_block >= static_cast<int>(view.block_table->size())) {
        return nullptr;
    }
    int physical_block = (*view.block_table)[(size_t)logical_block];
    return get_paged_k_token_ptr(physical_block, layer_id, kv_head_id, offset_in_block);
}

const fp16_t* KVCache::get_paged_v_token_ptr_by_pos(
    int layer_id, int kv_head_id, int logical_pos) const {
    PagedKVView view;
    if (!get_active_paged_view(&view) ||
        logical_pos < 0 || logical_pos >= view.seq_len ||
        !view.block_table) {
        return nullptr;
    }
    int logical_block = logical_pos / block_size_;
    int offset_in_block = logical_pos % block_size_;
    if (logical_block < 0 || logical_block >= static_cast<int>(view.block_table->size())) {
        return nullptr;
    }
    int physical_block = (*view.block_table)[(size_t)logical_block];
    return get_paged_v_token_ptr(physical_block, layer_id, kv_head_id, offset_in_block);
}

const fp16_t* KVCache::raw_k_pages() const {
    return is_paged() && !k_pages_.empty() ? k_pages_.data() : nullptr;
}

const fp16_t* KVCache::raw_v_pages() const {
    return is_paged() && !v_pages_.empty() ? v_pages_.data() : nullptr;
}

size_t KVCache::paged_elements_per_block() const {
    if (!is_paged()) {
        return 0;
    }
    return (size_t)num_layers * num_kv_heads * block_size_ * head_dim;
}

void KVCache::set_active_sequence(std::vector<int>* block_table, int* max_written_pos) {
    active_block_table_ = block_table;
    active_max_written_pos_ = max_written_pos;
}

void KVCache::clear_active_sequence() {
    active_block_table_ = nullptr;
    active_max_written_pos_ = nullptr;
}

fp16_t* KVCache::get_contiguous_k_head_ptr(int layer_id, int kv_head_id) {
    size_t layer_stride = (size_t)num_kv_heads * max_seq_len * head_dim;
    size_t head_stride = (size_t)max_seq_len * head_dim;
    return k_cache.data() + (size_t)layer_id * layer_stride + (size_t)kv_head_id * head_stride;
}

fp16_t* KVCache::get_contiguous_v_head_ptr(int layer_id, int kv_head_id) {
    size_t layer_stride = (size_t)num_kv_heads * max_seq_len * head_dim;
    size_t head_stride = (size_t)max_seq_len * head_dim;
    return v_cache.data() + (size_t)layer_id * layer_stride + (size_t)kv_head_id * head_stride;
}

fp16_t* KVCache::paged_k_token_ptr(
    int physical_block, int layer_id, int kv_head_id, int offset_in_block) {
    return k_pages_.data() +
           (((((size_t)physical_block * num_layers + layer_id) * num_kv_heads + kv_head_id) *
                 block_size_ +
             offset_in_block) *
            head_dim);
}

fp16_t* KVCache::paged_v_token_ptr(
    int physical_block, int layer_id, int kv_head_id, int offset_in_block) {
    return v_pages_.data() +
           (((((size_t)physical_block * num_layers + layer_id) * num_kv_heads + kv_head_id) *
                 block_size_ +
             offset_in_block) *
            head_dim);
}

const fp16_t* KVCache::paged_k_token_ptr(
    int physical_block, int layer_id, int kv_head_id, int offset_in_block) const {
    return k_pages_.data() +
           (((((size_t)physical_block * num_layers + layer_id) * num_kv_heads + kv_head_id) *
                 block_size_ +
             offset_in_block) *
            head_dim);
}

const fp16_t* KVCache::paged_v_token_ptr(
    int physical_block, int layer_id, int kv_head_id, int offset_in_block) const {
    return v_pages_.data() +
           (((((size_t)physical_block * num_layers + layer_id) * num_kv_heads + kv_head_id) *
                 block_size_ +
             offset_in_block) *
            head_dim);
}

fp16_t* KVCache::gather_head(bool gather_k, int layer_id, int kv_head_id) {
    std::vector<fp16_t>& buffer = gather_k ? gather_k_buffer_ : gather_v_buffer_;
    std::fill(buffer.begin(), buffer.end(), (fp16_t)0);

    std::vector<int>* active_table = active_block_table_ ? active_block_table_ : &block_table_;
    int max_written = active_max_written_pos_ ? *active_max_written_pos_ : max_written_pos_;
    int valid_len = max_written + 1;
    if (valid_len < 0) valid_len = 0;
    if (valid_len > max_seq_len) valid_len = max_seq_len;

    for (int pos = 0; pos < valid_len; ++pos) {
        int logical_block = pos / block_size_;
        int offset_in_block = pos % block_size_;
        if (logical_block < 0 || logical_block >= static_cast<int>(active_table->size())) {
            if (debug_enabled()) {
                std::cerr << "[KVCache] gather missing logical block"
                          << " logical_block=" << logical_block
                          << " table_size=" << active_table->size()
                          << std::endl;
            }
            continue;
        }
        int physical_block = (*active_table)[(size_t)logical_block];
        if (physical_block < 0 || physical_block >= num_physical_blocks_) {
            if (debug_enabled()) {
                std::cerr << "[KVCache] gather missing physical block"
                          << " logical_block=" << logical_block
                          << " physical_block=" << physical_block
                          << std::endl;
            }
            continue;
        }
        const fp16_t* src = gather_k
            ? paged_k_token_ptr(physical_block, layer_id, kv_head_id, offset_in_block)
            : paged_v_token_ptr(physical_block, layer_id, kv_head_id, offset_in_block);
        fp16_t* dst = buffer.data() + (size_t)pos * head_dim;
        std::memcpy(dst, src, (size_t)head_dim * sizeof(fp16_t));
    }

    debug_log_gather(gather_k, layer_id, kv_head_id, valid_len);
    return buffer.data();
}

bool KVCache::debug_enabled() const {
    return env_flag("LLM_DEBUG_KV");
}

bool KVCache::debug_verbose_enabled() const {
    return env_flag("LLM_DEBUG_KV_VERBOSE");
}

void KVCache::debug_log_config() const {
    if (!debug_enabled()) return;

    double kv_mb = total_kv_bytes() / 1024.0 / 1024.0;

    std::cerr << "[KVCache] layout=" << (is_paged() ? "paged" : "contiguous")
              << " max_seq_len=" << max_seq_len
              << " layers=" << num_layers
              << " kv_heads=" << num_kv_heads
              << " head_dim=" << head_dim
              << " total_kv_bytes=" << total_kv_bytes()
              << " kv_mb=" << kv_mb
              << std::endl;

    if (is_paged()) {
        std::cerr << "[KVCache] block_size=" << block_size_
                  << " logical_blocks=" << num_logical_blocks_
                  << " physical_blocks=" << num_physical_blocks_
                  << " bytes_per_block=" << bytes_per_block()
                  << std::endl;
    }
}

void KVCache::debug_log_gather(bool gather_k, int layer_id, int kv_head_id, int valid_len) {
    if (!debug_enabled()) return;
    if (!debug_verbose_enabled() && gather_log_count_ >= 4) return;

    std::cerr << "[KVCache] gather " << (gather_k ? "K" : "V")
              << " layer=" << layer_id
              << " head=" << kv_head_id
              << " valid_len=" << valid_len
              << std::endl;
    gather_log_count_++;
}

} // namespace llm_engine
