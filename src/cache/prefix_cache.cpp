#include "llm_engine/cache/prefix_cache.h"

#include <cstdio>
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

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void fold(uint64_t v, uint64_t& hi, uint64_t& lo) {
    hi = mix64(hi ^ v ^ (lo << 1));
    lo = mix64(lo ^ v ^ (hi >> 1));
}

std::string hash_string(const HashValue& h) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%016llx%016llx",
        static_cast<unsigned long long>(h.hi),
        static_cast<unsigned long long>(h.lo));
    return std::string(buf);
}

} // namespace

HashValue hash_token_block(
    const HashValue& parent_hash,
    const std::vector<int>& tokens,
    int begin,
    int block_size,
    const PrefixCacheConfig& config) {
    if (env_flag("LLM_PREFIX_FORCE_COLLISION")) {
        return HashValue{0xabcdef1234567890ULL, 0x1234567890abcdefULL};
    }

    uint64_t hi = 0x6a09e667f3bcc909ULL;
    uint64_t lo = 0xbb67ae8584caa73bULL;
    fold(parent_hash.hi, hi, lo);
    fold(parent_hash.lo, hi, lo);
    fold(config.model_hash, hi, lo);
    fold(config.tokenizer_hash, hi, lo);
    fold(config.chat_template_hash, hi, lo);
    fold(config.cache_salt, hi, lo);
    fold(static_cast<uint64_t>(block_size), hi, lo);

    for (int i = 0; i < block_size; ++i) {
        uint64_t token = static_cast<uint64_t>(static_cast<int64_t>(tokens[(size_t)(begin + i)]));
        fold(token ^ static_cast<uint64_t>(i), hi, lo);
    }

    return HashValue{hi, lo};
}

PrefixCache::PrefixCache(PrefixCacheConfig config) : config_(config) {
    if (debug_enabled()) {
        std::cerr << "[PREFIX] enabled"
                  << " salt=" << config_.cache_salt
                  << " model_hash=" << config_.model_hash
                  << " tokenizer_hash=" << config_.tokenizer_hash
                  << " chat_template_hash=" << config_.chat_template_hash
                  << std::endl;
    }
}

bool PrefixCache::lookup(
    const HashValue& hash,
    const std::vector<int>& prompt_tokens,
    int token_begin,
    int block_size,
    PrefixCacheEntry* out_entry) {
    tick_++;
    auto it = table_.find(hash);
    if (it == table_.end()) {
        misses_++;
        if (debug_enabled()) {
            std::cerr << "[PREFIX] lookup hash=" << hash_string(hash)
                      << " begin=" << token_begin
                      << " miss=not_found"
                      << std::endl;
        }
        return false;
    }

    PrefixCacheEntry& entry = it->second;
    if (entry.physical_block < 0 || entry.token_count != block_size) {
        misses_++;
        if (debug_enabled()) {
            std::cerr << "[PREFIX] lookup hash=" << hash_string(hash)
                      << " miss=invalid_entry"
                      << " physical=" << entry.physical_block
                      << " token_count=" << entry.token_count
                      << std::endl;
        }
        return false;
    }

    if (!same_tokens(prompt_tokens, token_begin, entry.tokens, block_size)) {
        misses_++;
        std::cerr << "[PREFIX] collision hash=" << hash_string(hash)
                  << " physical=" << entry.physical_block
                  << " begin=" << token_begin
                  << std::endl;
        return false;
    }

    hits_++;
    entry.hit_count++;
    entry.last_hit_tick = tick_;
    if (out_entry) {
        *out_entry = entry;
    }
    if (debug_enabled()) {
        std::cerr << "[PREFIX] lookup hash=" << hash_string(hash)
                  << " hit physical=" << entry.physical_block
                  << " hits=" << hits_
                  << std::endl;
    }
    return true;
}

bool PrefixCache::insert(
    const HashValue& hash,
    const HashValue& parent_hash,
    const std::vector<int>& all_tokens,
    int token_begin,
    int block_size,
    int physical_block) {
    if (physical_block < 0 || block_size <= 0 ||
        token_begin < 0 ||
        token_begin + block_size > static_cast<int>(all_tokens.size())) {
        return false;
    }

    auto it = table_.find(hash);
    if (it != table_.end()) {
        PrefixCacheEntry& entry = it->second;
        if (!same_tokens(all_tokens, token_begin, entry.tokens, block_size)) {
            std::cerr << "[PREFIX] insert collision hash=" << hash_string(hash)
                      << " old_physical=" << entry.physical_block
                      << " new_physical=" << physical_block
                      << std::endl;
            return false;
        }
        block_to_hash_.erase(entry.physical_block);
        entry.physical_block = physical_block;
        entry.parent_hash = parent_hash;
        block_to_hash_[physical_block] = hash;
        if (debug_enabled()) {
            std::cerr << "[PREFIX] insert update hash=" << hash_string(hash)
                      << " physical=" << physical_block
                      << std::endl;
        }
        return true;
    }

    PrefixCacheEntry entry;
    entry.hash = hash;
    entry.parent_hash = parent_hash;
    entry.physical_block = physical_block;
    entry.token_count = block_size;
    entry.tokens.assign(
        all_tokens.begin() + token_begin,
        all_tokens.begin() + token_begin + block_size);

    table_[hash] = entry;
    block_to_hash_[physical_block] = hash;
    inserts_++;
    if (debug_enabled()) {
        std::cerr << "[PREFIX] insert hash=" << hash_string(hash)
                  << " physical=" << physical_block
                  << " begin=" << token_begin
                  << " tokens=" << block_size
                  << " size=" << size()
                  << std::endl;
    }
    return true;
}

void PrefixCache::erase(const HashValue& hash) {
    auto it = table_.find(hash);
    if (it == table_.end()) {
        return;
    }
    int physical_block = it->second.physical_block;
    table_.erase(it);
    block_to_hash_.erase(physical_block);
    evictions_++;
    if (debug_enabled()) {
        std::cerr << "[PREFIX] erase hash=" << hash_string(hash)
                  << " physical=" << physical_block
                  << " size=" << size()
                  << std::endl;
    }
}

void PrefixCache::erase_block(int physical_block) {
    auto it = block_to_hash_.find(physical_block);
    if (it == block_to_hash_.end()) {
        return;
    }
    HashValue hash = it->second;
    table_.erase(hash);
    block_to_hash_.erase(it);
    evictions_++;
    if (debug_enabled()) {
        std::cerr << "[PREFIX] evict physical=" << physical_block
                  << " hash=" << hash_string(hash)
                  << " size=" << size()
                  << std::endl;
    }
}

bool PrefixCache::contains_block(int physical_block) const {
    return block_to_hash_.find(physical_block) != block_to_hash_.end();
}

bool PrefixCache::debug_enabled() const {
    return env_flag("LLM_DEBUG_PREFIX_CACHE");
}

bool PrefixCache::force_collision_enabled() const {
    return env_flag("LLM_PREFIX_FORCE_COLLISION");
}

bool PrefixCache::same_tokens(
    const std::vector<int>& a,
    int begin,
    const std::vector<int>& b,
    int block_size) const {
    if (begin < 0 || block_size < 0 ||
        begin + block_size > static_cast<int>(a.size()) ||
        block_size != static_cast<int>(b.size())) {
        return false;
    }
    for (int i = 0; i < block_size; ++i) {
        if (a[(size_t)(begin + i)] != b[(size_t)i]) {
            return false;
        }
    }
    return true;
}

void PrefixCache::log_stats(const char* tag) const {
    if (!debug_enabled()) return;
    std::cerr << "[PREFIX] stats"
              << " tag=" << tag
              << " hits=" << hits_
              << " misses=" << misses_
              << " inserts=" << inserts_
              << " evictions=" << evictions_
              << " size=" << size()
              << std::endl;
}

} // namespace llm_engine
