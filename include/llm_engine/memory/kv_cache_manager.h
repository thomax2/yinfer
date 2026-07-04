#pragma once

#include <vector>

#include "llm_engine/engine/sequence_state.h"
#include "llm_engine/memory/kv_cache.h"

namespace llm_engine {

enum class KVBlockState {
    FREE,
    USED
};

struct KVBlockMeta {
    int block_id = -1;
    int ref_count = 0;
    KVBlockState state = KVBlockState::FREE;
};

class KVCacheManager {
public:
    KVCacheManager(KVCache& cache, int max_seq_len, int block_size, int total_physical_blocks);

    bool init_sequence(SequenceState& seq);
    bool ensure_block_for_position(SequenceState& seq, int position);
    bool ensure_blocks_for_range(SequenceState& seq, int begin_pos, int end_pos);
    void free_sequence(SequenceState& seq);
    void reset();

    int total_blocks() const { return total_physical_blocks_; }
    int free_blocks() const { return static_cast<int>(free_list_.size()); }
    int used_blocks() const { return total_physical_blocks_ - free_blocks(); }

private:
    int allocate_block();
    void release_block(int block_id);
    bool debug_enabled() const;
    void log_stats(const char* tag) const;

    KVCache& cache_;
    int max_seq_len_ = 0;
    int block_size_ = 16;
    int num_logical_blocks_ = 0;
    int total_physical_blocks_ = 0;

    std::vector<KVBlockMeta> blocks_;
    std::vector<int> free_list_;
};

} // namespace llm_engine
