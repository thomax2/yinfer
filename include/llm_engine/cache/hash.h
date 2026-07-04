#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm_engine {

struct HashValue {
    uint64_t hi = 0;
    uint64_t lo = 0;

    bool operator==(const HashValue& other) const {
        return hi == other.hi && lo == other.lo;
    }

    bool empty() const {
        return hi == 0 && lo == 0;
    }
};

struct HashValueHasher {
    size_t operator()(const HashValue& h) const {
        return static_cast<size_t>(
            h.hi ^ (h.lo + 0x9e3779b97f4a7c15ULL + (h.hi << 6) + (h.hi >> 2)));
    }
};

struct PrefixCacheConfig {
    uint64_t model_hash = 0;
    uint64_t tokenizer_hash = 0;
    uint64_t chat_template_hash = 0;
    uint64_t cache_salt = 0;
};

HashValue hash_token_block(
    const HashValue& parent_hash,
    const std::vector<int>& tokens,
    int begin,
    int block_size,
    const PrefixCacheConfig& config);

} // namespace llm_engine
