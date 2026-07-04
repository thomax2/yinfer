#include "llm_engine/memory/kv_cache_manager.h"

#include <algorithm>
#include <cstdlib>
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

} // namespace

KVCacheManager::KVCacheManager(
    KVCache& cache,
    int max_seq_len,
    int block_size,
    int total_physical_blocks)
    : cache_(cache),
      max_seq_len_(max_seq_len),
      block_size_(block_size),
      num_logical_blocks_((max_seq_len + block_size - 1) / block_size),
      total_physical_blocks_(total_physical_blocks) {
    blocks_.resize((size_t)total_physical_blocks_);
    free_list_.reserve((size_t)total_physical_blocks_);
    for (int i = 0; i < total_physical_blocks_; ++i) {
        blocks_[(size_t)i].block_id = i;
        blocks_[(size_t)i].state = KVBlockState::FREE;
        blocks_[(size_t)i].ref_count = 0;
        free_list_.push_back(i);
    }

    log_stats("init");
}

bool KVCacheManager::init_sequence(SequenceState& seq) {
    if (seq.block_table.size() != (size_t)num_logical_blocks_) {
        seq.block_table.assign((size_t)num_logical_blocks_, -1);
    }
    return true;
}

bool KVCacheManager::ensure_block_for_position(SequenceState& seq, int position) {
    if (position < 0 || position >= max_seq_len_) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "KV position out of range";
        return false;
    }

    init_sequence(seq);
    int logical_block = position / block_size_;
    int& physical_block = seq.block_table[(size_t)logical_block];
    if (physical_block >= 0) {
        return true;
    }

    physical_block = allocate_block();
    if (physical_block < 0) {
        seq.status = SequenceStatus::FAILED;
        seq.error_message = "KV block pool exhausted";
        if (debug_enabled()) {
            std::cerr << "[KVManager] allocation failed"
                      << " session=" << seq.session_id
                      << " position=" << position
                      << " logical_block=" << logical_block
                      << " free=" << free_blocks()
                      << " used=" << used_blocks()
                      << std::endl;
        }
        return false;
    }

    if (debug_enabled()) {
        std::cerr << "[KVManager] alloc"
                  << " session=" << seq.session_id
                  << " logical_block=" << logical_block
                  << " physical_block=" << physical_block
                  << " free=" << free_blocks()
                  << " used=" << used_blocks()
                  << std::endl;
    }
    return true;
}

bool KVCacheManager::ensure_blocks_for_range(SequenceState& seq, int begin_pos, int end_pos) {
    if (begin_pos >= end_pos) {
        return true;
    }
    for (int pos = begin_pos; pos < end_pos; ++pos) {
        if (!ensure_block_for_position(seq, pos)) {
            return false;
        }
    }
    return true;
}

void KVCacheManager::free_sequence(SequenceState& seq) {
    if (seq.block_table.empty()) {
        return;
    }

    for (int& block_id : seq.block_table) {
        if (block_id >= 0) {
            release_block(block_id);
            block_id = -1;
        }
    }

    if (debug_enabled()) {
        std::cerr << "[KVManager] free_sequence"
                  << " session=" << seq.session_id
                  << " free=" << free_blocks()
                  << " used=" << used_blocks()
                  << std::endl;
    }
}

void KVCacheManager::reset() {
    free_list_.clear();
    for (int i = 0; i < total_physical_blocks_; ++i) {
        blocks_[(size_t)i].state = KVBlockState::FREE;
        blocks_[(size_t)i].ref_count = 0;
        free_list_.push_back(i);
    }
    log_stats("reset");
}

int KVCacheManager::allocate_block() {
    if (free_list_.empty()) {
        return -1;
    }
    int block_id = free_list_.back();
    free_list_.pop_back();

    KVBlockMeta& block = blocks_[(size_t)block_id];
    block.state = KVBlockState::USED;
    block.ref_count = 1;
    return block_id;
}

void KVCacheManager::release_block(int block_id) {
    if (block_id < 0 || block_id >= total_physical_blocks_) {
        return;
    }

    KVBlockMeta& block = blocks_[(size_t)block_id];
    if (block.state == KVBlockState::FREE) {
        return;
    }

    block.ref_count = std::max(0, block.ref_count - 1);
    if (block.ref_count == 0) {
        block.state = KVBlockState::FREE;
        free_list_.push_back(block_id);
    }
}

bool KVCacheManager::debug_enabled() const {
    return env_flag("LLM_DEBUG_KV_MANAGER");
}

void KVCacheManager::log_stats(const char* tag) const {
    if (!debug_enabled()) return;

    std::cerr << "[KVManager] " << tag
              << " block_size=" << block_size_
              << " logical_blocks=" << num_logical_blocks_
              << " physical_blocks=" << total_physical_blocks_
              << " bytes_per_block=" << cache_.bytes_per_block()
              << " total_kv_bytes=" << cache_.total_kv_bytes()
              << " free=" << free_blocks()
              << " used=" << used_blocks()
              << std::endl;
}

} // namespace llm_engine
