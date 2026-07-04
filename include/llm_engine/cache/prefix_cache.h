#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "llm_engine/cache/hash.h"

namespace llm_engine {

struct PrefixCacheEntry {
    HashValue hash;
    HashValue parent_hash;
    int physical_block = -1;
    int token_count = 0;
    std::vector<int> tokens;
    uint64_t hit_count = 0;
    uint64_t last_hit_tick = 0;
};

class PrefixCache {
public:
    explicit PrefixCache(PrefixCacheConfig config);

    bool lookup(
        const HashValue& hash,
        const std::vector<int>& prompt_tokens,
        int token_begin,
        int block_size,
        PrefixCacheEntry* out_entry);

    bool insert(
        const HashValue& hash,
        const HashValue& parent_hash,
        const std::vector<int>& all_tokens,
        int token_begin,
        int block_size,
        int physical_block);

    void erase(const HashValue& hash);
    void erase_block(int physical_block);
    bool contains_block(int physical_block) const;

    int size() const { return static_cast<int>(table_.size()); }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t inserts() const { return inserts_; }
    uint64_t evictions() const { return evictions_; }
    const PrefixCacheConfig& config() const { return config_; }

private:
    bool debug_enabled() const;
    bool force_collision_enabled() const;
    bool same_tokens(
        const std::vector<int>& a,
        int begin,
        const std::vector<int>& b,
        int block_size) const;
    void log_stats(const char* tag) const;

    PrefixCacheConfig config_;
    std::unordered_map<HashValue, PrefixCacheEntry, HashValueHasher> table_;
    std::unordered_map<int, HashValue> block_to_hash_;
    uint64_t tick_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t inserts_ = 0;
    uint64_t evictions_ = 0;
};

} // namespace llm_engine
