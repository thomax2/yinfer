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
                      << " active=" << active_blocks()
                      << " cached=" << cached_blocks()
                      << " lru=" << lru_size()
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
                  << " active=" << active_blocks()
                  << " cached=" << cached_blocks()
                  << " lru=" << lru_size()
                  << std::endl;
    }
    maybe_check_invariants("ensure_block");
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

bool KVCacheManager::retain_block(int block_id) {
    if (block_id < 0 || block_id >= total_physical_blocks_) {
        return false;
    }

    KVBlockMeta& block = blocks_[(size_t)block_id];
    if (block.state == KVBlockState::FREE) {
        return false;
    }

    if (block.state == KVBlockState::CACHED) {
        remove_from_lru(block_id);
        block.state = KVBlockState::ACTIVE;
    }

    block.ref_count++;
    block.last_access_tick = ++access_tick_;
    maybe_check_invariants("retain_block");
    return true;
}

void KVCacheManager::release_block_ref(int block_id) {
    if (block_id < 0 || block_id >= total_physical_blocks_) {
        return;
    }

    KVBlockMeta& block = blocks_[(size_t)block_id];
    if (block.state == KVBlockState::FREE) {
        if (debug_enabled()) {
            std::cerr << "[KVManager] release ignored free block=" << block_id << std::endl;
        }
        return;
    }

    if (block.ref_count <= 0) {
        if (debug_enabled()) {
            std::cerr << "[KVManager] release ignored invalid ref"
                      << " block=" << block_id
                      << " state=" << state_name(block.state)
                      << " ref=" << block.ref_count
                      << std::endl;
        }
        return;
    }

    block.ref_count--;
    block.last_access_tick = ++access_tick_;
    if (block.ref_count == 0) {
        block.state = KVBlockState::CACHED;
        push_lru_tail(block_id);
    }
    maybe_check_invariants("release_block_ref");
}

bool KVCacheManager::mark_sequence_active(SequenceState& seq) {
    init_sequence(seq);

    std::vector<int> retained;
    retained.reserve(seq.block_table.size());
    for (int block_id : seq.block_table) {
        if (block_id < 0) {
            continue;
        }
        if (!retain_block(block_id)) {
            for (int retained_block : retained) {
                release_block_ref(retained_block);
            }
            seq.status = SequenceStatus::FAILED;
            seq.error_message = "KV sequence references an unavailable block";
            return false;
        }
        retained.push_back(block_id);
    }

    seq.status = SequenceStatus::RUNNING;
    return true;
}

void KVCacheManager::release_sequence_to_cache(SequenceState& seq) {
    if (seq.block_table.empty()) {
        return;
    }

    for (int& block_id : seq.block_table) {
        if (block_id >= 0) {
            release_block_ref(block_id);
            block_id = -1;
        }
    }

    seq.history_pos = 0;
    seq.max_written_pos = -1;
    seq.status = SequenceStatus::IDLE;
    seq.error_message.clear();

    if (debug_enabled()) {
        std::cerr << "[KVManager] release_sequence_to_cache"
                  << " session=" << seq.session_id
                  << " free=" << free_blocks()
                  << " active=" << active_blocks()
                  << " cached=" << cached_blocks()
                  << " lru=" << lru_size()
                  << std::endl;
    }
    maybe_check_invariants("release_sequence_to_cache");
}

void KVCacheManager::discard_sequence(SequenceState& seq) {
    if (seq.block_table.empty()) {
        return;
    }

    for (int& block_id : seq.block_table) {
        if (block_id < 0 || block_id >= total_physical_blocks_) {
            block_id = -1;
            continue;
        }

        KVBlockMeta& block = blocks_[(size_t)block_id];
        if (block.state == KVBlockState::ACTIVE && block.ref_count > 0) {
            block.ref_count--;
            if (block.ref_count > 0) {
                block_id = -1;
                continue;
            }
        }
        if (block.state != KVBlockState::FREE && block.ref_count <= 0) {
            make_block_free(block_id, evict_zero_enabled());
        }
        block_id = -1;
    }

    seq.history_pos = 0;
    seq.max_written_pos = -1;
    seq.status = SequenceStatus::IDLE;
    seq.error_message.clear();

    if (debug_enabled()) {
        std::cerr << "[KVManager] discard_sequence"
                  << " session=" << seq.session_id
                  << " free=" << free_blocks()
                  << " active=" << active_blocks()
                  << " cached=" << cached_blocks()
                  << " lru=" << lru_size()
                  << std::endl;
    }
    maybe_check_invariants("discard_sequence");
}

void KVCacheManager::free_sequence(SequenceState& seq) {
    release_sequence_to_cache(seq);
}

bool KVCacheManager::evict_one_cached_block() {
    int block_id = pop_lru_head();
    if (block_id < 0) {
        return false;
    }

    KVBlockMeta& block = blocks_[(size_t)block_id];
    if (block.state != KVBlockState::CACHED || block.ref_count != 0) {
        if (debug_enabled()) {
            std::cerr << "[KVManager] evict skipped invalid cached block"
                      << " block=" << block_id
                      << " state=" << state_name(block.state)
                      << " ref=" << block.ref_count
                      << std::endl;
        }
        return false;
    }

    make_block_free(block_id, evict_zero_enabled());
    if (debug_enabled()) {
        std::cerr << "[KVManager] evict_cached"
                  << " block=" << block_id
                  << " free=" << free_blocks()
                  << " active=" << active_blocks()
                  << " cached=" << cached_blocks()
                  << " lru=" << lru_size()
                  << std::endl;
    }
    maybe_check_invariants("evict_one_cached_block");
    return true;
}

int KVCacheManager::evict_until_free_block_available() {
    int evicted = 0;
    while (free_list_.empty()) {
        if (!evict_one_cached_block()) {
            break;
        }
        evicted++;
    }
    return evicted;
}

void KVCacheManager::clear_cached_blocks() {
    while (evict_one_cached_block()) {
    }
    maybe_check_invariants("clear_cached_blocks");
}

void KVCacheManager::clear_all() {
    free_list_.clear();
    lru_head_ = -1;
    lru_tail_ = -1;
    lru_size_ = 0;
    for (int i = 0; i < total_physical_blocks_; ++i) {
        KVBlockMeta& block = blocks_[(size_t)i];
        block.block_id = i;
        block.state = KVBlockState::FREE;
        block.ref_count = 0;
        block.in_lru = false;
        block.lru_prev = -1;
        block.lru_next = -1;
        block.last_access_tick = 0;
        block.has_hash = false;
        block.debug_hash = 0;
        free_list_.push_back(i);
        if (evict_zero_enabled()) {
            cache_.clear_physical_block(i);
        }
    }
    log_stats("clear_all");
    maybe_check_invariants("clear_all");
}

void KVCacheManager::reset() {
    clear_all();
}

int KVCacheManager::allocate_block() {
    if (free_list_.empty()) {
        evict_until_free_block_available();
        if (free_list_.empty()) {
            return -1;
        }
    }
    int block_id = pop_free_block();
    if (block_id < 0) {
        return -1;
    }

    KVBlockMeta& block = blocks_[(size_t)block_id];
    block.state = KVBlockState::ACTIVE;
    block.ref_count = 1;
    block.in_lru = false;
    block.lru_prev = -1;
    block.lru_next = -1;
    block.last_access_tick = ++access_tick_;
    block.has_hash = false;
    block.debug_hash = 0;
    maybe_check_invariants("allocate_block");
    return block_id;
}

int KVCacheManager::pop_free_block() {
    while (!free_list_.empty()) {
        int block_id = free_list_.back();
        free_list_.pop_back();
        if (block_id >= 0 && block_id < total_physical_blocks_ &&
            blocks_[(size_t)block_id].state == KVBlockState::FREE) {
            return block_id;
        }
    }
    return -1;
}

void KVCacheManager::push_free_block(int block_id) {
    if (block_id < 0 || block_id >= total_physical_blocks_) {
        return;
    }
    free_list_.push_back(block_id);
}

void KVCacheManager::make_block_free(int block_id, bool clear_page) {
    if (block_id < 0 || block_id >= total_physical_blocks_) {
        return;
    }
    KVBlockMeta& block = blocks_[(size_t)block_id];
    remove_from_lru(block_id);

    block.state = KVBlockState::FREE;
    block.ref_count = 0;
    block.in_lru = false;
    block.lru_prev = -1;
    block.lru_next = -1;
    block.last_access_tick = ++access_tick_;
    block.has_hash = false;
    block.debug_hash = 0;

    if (clear_page) {
        cache_.clear_physical_block(block_id);
    }
    push_free_block(block_id);
}

void KVCacheManager::push_lru_tail(int block_id) {
    if (block_id < 0 || block_id >= total_physical_blocks_) {
        return;
    }
    KVBlockMeta& block = blocks_[(size_t)block_id];
    if (block.in_lru) {
        remove_from_lru(block_id);
    }

    block.in_lru = true;
    block.lru_prev = lru_tail_;
    block.lru_next = -1;
    if (lru_tail_ >= 0) {
        blocks_[(size_t)lru_tail_].lru_next = block_id;
    } else {
        lru_head_ = block_id;
    }
    lru_tail_ = block_id;
    lru_size_++;
}

void KVCacheManager::remove_from_lru(int block_id) {
    if (block_id < 0 || block_id >= total_physical_blocks_) {
        return;
    }
    KVBlockMeta& block = blocks_[(size_t)block_id];
    if (!block.in_lru) {
        block.lru_prev = -1;
        block.lru_next = -1;
        return;
    }

    if (block.lru_prev >= 0) {
        blocks_[(size_t)block.lru_prev].lru_next = block.lru_next;
    } else {
        lru_head_ = block.lru_next;
    }
    if (block.lru_next >= 0) {
        blocks_[(size_t)block.lru_next].lru_prev = block.lru_prev;
    } else {
        lru_tail_ = block.lru_prev;
    }

    block.in_lru = false;
    block.lru_prev = -1;
    block.lru_next = -1;
    if (lru_size_ > 0) {
        lru_size_--;
    }
}

int KVCacheManager::pop_lru_head() {
    int block_id = lru_head_;
    if (block_id >= 0) {
        remove_from_lru(block_id);
    }
    return block_id;
}

int KVCacheManager::active_blocks() const {
    int count = 0;
    for (const KVBlockMeta& block : blocks_) {
        if (block.state == KVBlockState::ACTIVE) {
            count++;
        }
    }
    return count;
}

int KVCacheManager::cached_blocks() const {
    int count = 0;
    for (const KVBlockMeta& block : blocks_) {
        if (block.state == KVBlockState::CACHED) {
            count++;
        }
    }
    return count;
}

bool KVCacheManager::check_invariants() const {
    if (total_physical_blocks_ < 0 ||
        blocks_.size() != (size_t)total_physical_blocks_) {
        return false;
    }

    std::vector<int> free_seen((size_t)total_physical_blocks_, 0);
    for (int block_id : free_list_) {
        if (block_id < 0 || block_id >= total_physical_blocks_) {
            return false;
        }
        if (++free_seen[(size_t)block_id] != 1) {
            return false;
        }
    }

    std::vector<int> lru_seen((size_t)total_physical_blocks_, 0);
    int lru_count = 0;
    int prev = -1;
    for (int block_id = lru_head_; block_id >= 0;) {
        if (block_id >= total_physical_blocks_) {
            return false;
        }
        const KVBlockMeta& block = blocks_[(size_t)block_id];
        if (++lru_seen[(size_t)block_id] != 1) {
            return false;
        }
        if (!block.in_lru || block.lru_prev != prev ||
            block.state != KVBlockState::CACHED || block.ref_count != 0) {
            return false;
        }
        prev = block_id;
        block_id = block.lru_next;
        lru_count++;
    }
    if (prev != lru_tail_ || lru_count != lru_size_) {
        return false;
    }

    int free_count = 0;
    int active_count = 0;
    int cached_count = 0;
    for (int i = 0; i < total_physical_blocks_; ++i) {
        const KVBlockMeta& block = blocks_[(size_t)i];
        if (block.block_id != i) {
            return false;
        }

        bool in_free = free_seen[(size_t)i] != 0;
        bool in_lru = lru_seen[(size_t)i] != 0;
        if (block.state == KVBlockState::FREE) {
            if (block.ref_count != 0 || block.in_lru || in_lru || !in_free) {
                return false;
            }
            free_count++;
        } else if (block.state == KVBlockState::ACTIVE) {
            if (block.ref_count <= 0 || block.in_lru || in_lru || in_free) {
                return false;
            }
            active_count++;
        } else if (block.state == KVBlockState::CACHED) {
            if (block.ref_count != 0 || !block.in_lru || !in_lru || in_free) {
                return false;
            }
            cached_count++;
        } else {
            return false;
        }
    }

    return free_count == free_blocks() &&
           active_count == active_blocks() &&
           cached_count == cached_blocks() &&
           cached_count == lru_size_ &&
           free_count + active_count + cached_count == total_physical_blocks_;
}

bool KVCacheManager::debug_enabled() const {
    return env_flag("LLM_DEBUG_KV_MANAGER");
}

bool KVCacheManager::invariant_debug_enabled() const {
    return env_flag("LLM_DEBUG_KV_INVARIANTS");
}

bool KVCacheManager::evict_zero_enabled() const {
    return env_flag("LLM_DEBUG_KV_EVICT_ZERO");
}

void KVCacheManager::maybe_check_invariants(const char* tag) const {
    if (!invariant_debug_enabled()) {
        return;
    }
    if (!check_invariants()) {
        std::cerr << "[KVManager] invariant_failed"
                  << " tag=" << tag
                  << " free=" << free_blocks()
                  << " active=" << active_blocks()
                  << " cached=" << cached_blocks()
                  << " lru=" << lru_size()
                  << std::endl;
    }
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
              << " active=" << active_blocks()
              << " cached=" << cached_blocks()
              << " used=" << used_blocks()
              << " lru=" << lru_size()
              << std::endl;
}

const char* KVCacheManager::state_name(KVBlockState state) const {
    switch (state) {
        case KVBlockState::FREE:
            return "FREE";
        case KVBlockState::ACTIVE:
            return "ACTIVE";
        case KVBlockState::CACHED:
            return "CACHED";
    }
    return "UNKNOWN";
}

} // namespace llm_engine
