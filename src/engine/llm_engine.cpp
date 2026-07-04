#include "llm_engine/engine/llm_engine.h"

#include "../../model/model.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

namespace llm_engine {

namespace {

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

uint64_t env_u64(const char* name, uint64_t default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 0);
    if (end == v) return default_value;
    return static_cast<uint64_t>(x);
}

} // namespace

LLMEngine::LLMEngine(QwenModel& model) : model_(model) {
    bool requested_session_cache = env_flag("LLM_ENABLE_SESSION_CACHE");
    bool requested_prefix_cache = env_flag("LLM_ENABLE_PREFIX_CACHE");
    if (requested_session_cache && model_.kv_cache && model_.kv_cache->is_paged()) {
        session_cache_enabled_ = true;
        kv_manager_ = std::make_unique<KVCacheManager>(
            *model_.kv_cache,
            model_.kv_cache->get_max_seq_len(),
            model_.kv_cache->block_size(),
            model_.kv_cache->allocated_blocks());
        if (debug_session_enabled()) {
            std::cerr << "[SESSION] enabled default_session=" << default_session_id_
                      << std::endl;
        }
        if (requested_prefix_cache) {
            PrefixCacheConfig config;
            config.model_hash = env_u64("LLM_MODEL_HASH", 0x2515b0015b8f1001ULL);
            config.tokenizer_hash = env_u64("LLM_TOKENIZER_HASH", 0x746f6b656e697a31ULL);
            config.chat_template_hash = env_u64("LLM_CHAT_TEMPLATE_HASH", 0x636861746d6c0001ULL);
            config.cache_salt = env_u64("LLM_PREFIX_CACHE_SALT", 0);
            prefix_cache_ = std::make_unique<PrefixCache>(config);
            kv_manager_->set_prefix_cache(prefix_cache_.get());
            prefix_cache_enabled_ = true;
        }
    } else if (requested_session_cache) {
        std::cerr << "[SESSION] LLM_ENABLE_SESSION_CACHE ignored because paged KV is not enabled"
                  << std::endl;
    }
    if (requested_prefix_cache && !prefix_cache_enabled_) {
        std::cerr << "[PREFIX] LLM_ENABLE_PREFIX_CACHE ignored because paged session cache is not enabled"
                  << std::endl;
    }
}

RequestId LLMEngine::submit(
    const std::vector<int>& prompt_tokens,
    const SamplingParams& sampling,
    TokenCallback callback
) {
    return submit(default_session_id_, prompt_tokens, sampling, std::move(callback));
}

RequestId LLMEngine::submit(
    SessionId session_id,
    const std::vector<int>& prompt_tokens,
    const SamplingParams& sampling,
    TokenCallback callback
) {
    RequestId id = next_request_id_++;

    RequestState request;
    request.id = id;
    request.session_id = session_id;
    request.prompt_tokens = prompt_tokens;
    request.sampling = sampling;
    request.callback = std::move(callback);
    request.status = RequestStatus::RUNNING;

    auto inserted = requests_.emplace(id, std::move(request));
    RequestState& state = inserted.first->second;
    debug_log_submit(state);

    bool callback_stopped = false;
    try {
        auto wrapped_callback = [this, id, &callback_stopped](int token_id) {
            auto it = requests_.find(id);
            if (it == requests_.end()) {
                callback_stopped = true;
                return false;
            }

            RequestState& active = it->second;
            if (active.status == RequestStatus::ABORTED) {
                callback_stopped = true;
                return false;
            }

            active.generated_tokens.push_back(token_id);
            active.token_count++;

            if (active.callback && !active.callback(token_id)) {
                callback_stopped = true;
                return false;
            }
            return true;
        };

        if (session_cache_enabled_) {
            SequenceState& seq = get_or_create_session(session_id);
            if (debug_session_enabled()) {
                std::cerr << "[SESSION] submit"
                          << " session=" << session_id
                          << " request=" << id
                          << " history_pos=" << seq.history_pos
                          << " prompt_tokens=" << state.prompt_tokens.size()
                          << std::endl;
            }
            apply_prefix_cache(seq, state.prompt_tokens);
            model_.generate_for_sequence(
                seq,
                state.prompt_tokens,
                state.sampling.max_new_tokens,
                *kv_manager_,
                prefix_cache_.get(),
                wrapped_callback);
            if (seq.status == SequenceStatus::FAILED) {
                state.status = RequestStatus::FAILED;
                state.error_message = seq.error_message;
                debug_log_failed(state);
                state.callback = TokenCallback{};
                return id;
            }
            if (seq.status == SequenceStatus::ABORTED) {
                callback_stopped = true;
            }
            if (debug_session_enabled()) {
                std::cerr << "[SESSION] finished"
                          << " session=" << session_id
                          << " request=" << id
                          << " history_pos=" << seq.history_pos
                          << " max_written_pos=" << seq.max_written_pos
                          << " cached_prefix_tokens=" << seq.cached_prefix_tokens
                          << " cached_prefix_blocks=" << seq.cached_prefix_blocks
                          << " computed_tokens=" << seq.num_computed_tokens
                          << std::endl;
            }
        } else {
            model_.generate(
                state.prompt_tokens,
                state.sampling.max_new_tokens,
                wrapped_callback);
        }

        state.status = callback_stopped ? RequestStatus::ABORTED : RequestStatus::FINISHED;
        debug_log_finished(state);
    } catch (const std::exception& e) {
        state.status = RequestStatus::FAILED;
        state.error_message = e.what();
        debug_log_failed(state);
    } catch (...) {
        state.status = RequestStatus::FAILED;
        state.error_message = "unknown exception";
        debug_log_failed(state);
    }

    state.callback = TokenCallback{};
    return id;
}

void LLMEngine::abort(RequestId id) {
    auto it = requests_.find(id);
    if (it != requests_.end() &&
        it->second.status != RequestStatus::FINISHED &&
        it->second.status != RequestStatus::FAILED) {
        it->second.status = RequestStatus::ABORTED;
    }
}

void LLMEngine::clear_history() {
    if (debug_enabled()) {
        std::cerr << "[ENGINE] clear_history" << std::endl;
    }
    clear_session(default_session_id_);
}

void LLMEngine::clear_session(SessionId session_id) {
    if (!session_cache_enabled_) {
        model_.clear_history();
        return;
    }

    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        bool discard_blocks = env_flag("LLM_SESSION_CLEAR_DISCARD");
        if (kv_manager_) {
            if (discard_blocks) {
                kv_manager_->discard_sequence(it->second);
                kv_manager_->clear_cached_blocks();
            } else {
                kv_manager_->release_sequence_to_cache(it->second);
            }
        }
        it->second.reset();
        it->second.session_id = session_id;
        if (debug_session_enabled()) {
            std::cerr << "[SESSION] clear"
                      << " session=" << session_id
                      << " mode=" << (discard_blocks ? "discard" : "cache")
                      << std::endl;
        }
    }
    if (model_.kv_cache) {
        model_.kv_cache->clear_active_sequence();
    }
}

const RequestState* LLMEngine::get_request(RequestId id) const {
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return nullptr;
    }
    return &it->second;
}

const SequenceState* LLMEngine::get_session(SessionId session_id) const {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool LLMEngine::debug_enabled() const {
    return env_flag("LLM_DEBUG_ENGINE");
}

bool LLMEngine::debug_session_enabled() const {
    return env_flag("LLM_DEBUG_SESSION");
}

bool LLMEngine::debug_prefix_enabled() const {
    return env_flag("LLM_DEBUG_PREFIX_CACHE");
}

void LLMEngine::apply_prefix_cache(SequenceState& seq, const std::vector<int>& prompt_tokens) {
    if (!prefix_cache_enabled_ || !prefix_cache_ || !kv_manager_) {
        return;
    }
    if (seq.history_pos != 0) {
        if (debug_prefix_enabled()) {
            std::cerr << "[PREFIX] skip lookup reason=session_history"
                      << " history_pos=" << seq.history_pos
                      << std::endl;
        }
        return;
    }

    kv_manager_->init_sequence(seq);
    seq.cached_prefix_tokens = 0;
    seq.cached_prefix_blocks = 0;
    seq.num_computed_tokens = 0;
    seq.last_prefix_hash = {};
    seq.all_tokens.clear();

    int block_size = model_.kv_cache ? model_.kv_cache->block_size() : 0;
    if (block_size <= 0) {
        return;
    }
    int full_blocks = static_cast<int>(prompt_tokens.size()) / block_size;
    HashValue parent_hash;
    for (int block_idx = 0; block_idx < full_blocks; ++block_idx) {
        if (block_idx >= static_cast<int>(seq.block_table.size())) {
            break;
        }
        int begin = block_idx * block_size;
        HashValue h = hash_token_block(
            parent_hash,
            prompt_tokens,
            begin,
            block_size,
            prefix_cache_->config());

        PrefixCacheEntry entry;
        if (!prefix_cache_->lookup(h, prompt_tokens, begin, block_size, &entry)) {
            if (debug_prefix_enabled()) {
                std::cerr << "[PREFIX] stop lookup"
                          << " block=" << block_idx
                          << " reason=miss"
                          << std::endl;
            }
            break;
        }
        if (!kv_manager_->retain_block(entry.physical_block)) {
            if (debug_prefix_enabled()) {
                std::cerr << "[PREFIX] stop lookup"
                          << " block=" << block_idx
                          << " reason=retain_failed"
                          << " physical=" << entry.physical_block
                          << std::endl;
            }
            break;
        }

        seq.block_table[(size_t)block_idx] = entry.physical_block;
        seq.history_pos += block_size;
        seq.max_written_pos = seq.history_pos - 1;
        seq.cached_prefix_blocks++;
        seq.cached_prefix_tokens += block_size;
        seq.last_prefix_hash = h;
        seq.all_tokens.insert(
            seq.all_tokens.end(),
            prompt_tokens.begin() + begin,
            prompt_tokens.begin() + begin + block_size);
        parent_hash = h;

        if (debug_prefix_enabled()) {
            std::cerr << "[PREFIX] retain"
                      << " block=" << block_idx
                      << " physical=" << entry.physical_block
                      << " ref_count=" << kv_manager_->block_ref_count(entry.physical_block)
                      << " cached_tokens=" << seq.cached_prefix_tokens
                      << std::endl;
        }
    }
}

void LLMEngine::debug_log_submit(const RequestState& request) const {
    if (!debug_enabled()) return;
    std::cerr << "[ENGINE] submit"
              << " id=" << request.id
              << " session=" << request.session_id
              << " prompt_tokens=" << request.prompt_tokens.size()
              << " max_new_tokens=" << request.sampling.max_new_tokens
              << std::endl;
}

void LLMEngine::debug_log_finished(const RequestState& request) const {
    if (!debug_enabled()) return;
    const char* status = request.status == RequestStatus::ABORTED ? "aborted" : "finished";
    std::cerr << "[ENGINE] " << status
              << " id=" << request.id
              << " generated_tokens=" << request.generated_tokens.size()
              << std::endl;
}

void LLMEngine::debug_log_failed(const RequestState& request) const {
    if (!debug_enabled()) return;
    std::cerr << "[ENGINE] failed"
              << " id=" << request.id
              << " error=" << request.error_message
              << std::endl;
}

SequenceState& LLMEngine::get_or_create_session(SessionId session_id) {
    auto it = sessions_.find(session_id);
    if (it != sessions_.end()) {
        return it->second;
    }

    SequenceState seq;
    seq.session_id = session_id;
    auto inserted = sessions_.emplace(session_id, std::move(seq));
    if (kv_manager_) {
        kv_manager_->init_sequence(inserted.first->second);
    }
    if (debug_session_enabled()) {
        std::cerr << "[SESSION] create"
                  << " session=" << session_id
                  << std::endl;
    }
    return inserted.first->second;
}

} // namespace llm_engine
