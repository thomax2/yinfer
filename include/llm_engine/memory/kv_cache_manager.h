#pragma once

#include <cstdint>
#include <vector>

#include "llm_engine/cache/hash.h"
#include "llm_engine/engine/sequence_state.h"
#include "llm_engine/memory/kv_cache.h"

namespace llm_engine {

class PrefixCache;

enum class KVBlockState {
    FREE,
    ACTIVE,
    CACHED
};

struct KVBlockMeta {
    int block_id = -1;
    int ref_count = 0;
    KVBlockState state = KVBlockState::FREE;
    bool in_lru = false;
    int lru_prev = -1;
    int lru_next = -1;
    uint64_t last_access_tick = 0;
    bool has_hash = false;
    uint64_t debug_hash = 0;
    HashValue block_hash;
    HashValue parent_hash;
    int token_count = 0;
};

class KVCacheManager {
public:
    KVCacheManager(KVCache& cache, int max_seq_len, int block_size, int total_physical_blocks);

    bool init_sequence(SequenceState& seq);
    bool ensure_block_for_position(SequenceState& seq, int position);
    bool ensure_blocks_for_range(SequenceState& seq, int begin_pos, int end_pos);
    bool retain_block(int block_id);
    void release_block_ref(int block_id);
    bool mark_sequence_active(SequenceState& seq);
    void release_sequence_to_cache(SequenceState& seq);
    void discard_sequence(SequenceState& seq);
    void free_sequence(SequenceState& seq);
    bool evict_one_cached_block();
    int evict_until_free_block_available();
    void clear_cached_blocks();
    void clear_all();
    void reset();
    void set_prefix_cache(PrefixCache* cache);
    bool attach_hash_to_block(
        int block_id,
        const HashValue& block_hash,
        const HashValue& parent_hash,
        int token_count);
    bool block_has_hash(int block_id) const;
    HashValue block_hash(int block_id) const;
    int block_token_count(int block_id) const;
    int block_ref_count(int block_id) const;

    int total_blocks() const { return total_physical_blocks_; }
    int free_blocks() const { return static_cast<int>(free_list_.size()); }
    int used_blocks() const { return total_physical_blocks_ - free_blocks(); }
    int active_blocks() const;
    int cached_blocks() const;
    int lru_size() const { return lru_size_; }
    bool check_invariants() const;

private:
    int allocate_block();
    int pop_free_block();
    void push_free_block(int block_id);
    void make_block_free(int block_id, bool clear_page);
    void clear_block_hash(KVBlockMeta& block);
    void push_lru_tail(int block_id);
    void remove_from_lru(int block_id);
    int pop_lru_head();
    bool debug_enabled() const;
    bool invariant_debug_enabled() const;
    bool evict_zero_enabled() const;
    void maybe_check_invariants(const char* tag) const;
    void log_stats(const char* tag) const;
    const char* state_name(KVBlockState state) const;

    KVCache& cache_;
    PrefixCache* prefix_cache_ = nullptr;
    int max_seq_len_ = 0;
    int block_size_ = 16;
    int num_logical_blocks_ = 0;
    int total_physical_blocks_ = 0;

    std::vector<KVBlockMeta> blocks_;
    std::vector<int> free_list_;
    int lru_head_ = -1;
    int lru_tail_ = -1;
    int lru_size_ = 0;
    uint64_t access_tick_ = 0;
};

} // namespace llm_engine
