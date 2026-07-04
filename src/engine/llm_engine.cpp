#include "llm_engine/engine/llm_engine.h"

#include "../../model/model.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

namespace llm_engine {

namespace {

constexpr int QWEN_EOT_ID = 151643;
constexpr int QWEN_IM_START_ID = 151644;
constexpr int QWEN_IM_END_ID = 151645;

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

uint64_t env_u64(const char* name, uint64_t default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    unsigned long long x = std::strtoull(v, &end, 0);
    if (end == v) return default_value;
    return static_cast<uint64_t>(x);
}

bool is_stop_token(int token_id) {
    return token_id == QWEN_EOT_ID ||
           token_id == QWEN_IM_START_ID ||
           token_id == QWEN_IM_END_ID;
}

} // namespace

LLMEngine::LLMEngine(QwenModel& model) : model_(model) {
    scheduler_enabled_ = env_flag("LLM_ENABLE_SCHEDULER");
    prefill_step_tokens_ = env_int("LLM_PREFILL_STEP_TOKENS", 1);
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] enabled=" << (scheduler_enabled_ ? 1 : 0)
                  << " prefill_step_tokens=" << prefill_step_tokens_
                  << std::endl;
    }

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
    if (scheduler_enabled_) {
        RequestId id = submit_async(session_id, prompt_tokens, sampling, std::move(callback));
        run_until_finished(id);
        return id;
    }

    RequestId id = next_request_id_++;

    RequestState request;
    request.id = id;
    request.session_id = session_id;
    request.prompt_tokens = prompt_tokens;
    request.sampling = sampling;
    request.callback = std::move(callback);
    request.status = RequestStatus::RUNNING_PREFILL;

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

RequestId LLMEngine::submit_async(
    const std::vector<int>& prompt_tokens,
    const SamplingParams& sampling,
    TokenCallback callback
) {
    return submit_async(default_session_id_, prompt_tokens, sampling, std::move(callback));
}

RequestId LLMEngine::submit_async(
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
    request.status = RequestStatus::WAITING;

    requests_.emplace(id, std::move(request));
    waiting_queue_.push_back(id);

    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] submit_async"
                  << " id=" << id
                  << " session=" << session_id
                  << " prompt_tokens=" << prompt_tokens.size()
                  << " max_new_tokens=" << sampling.max_new_tokens
                  << " queue=" << waiting_queue_.size()
                  << std::endl;
    }
    return id;
}

void LLMEngine::abort(RequestId id) {
    auto it = requests_.find(id);
    if (it != requests_.end() &&
        it->second.status != RequestStatus::FINISHED &&
        it->second.status != RequestStatus::ABORTED &&
        it->second.status != RequestStatus::FAILED) {
        it->second.status = RequestStatus::ABORTED;
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] abort"
                      << " id=" << id
                      << " active=" << active_request_id_
                      << std::endl;
        }
    }
}

bool LLMEngine::step_once() {
    if (active_request_id_ == 0) {
        schedule_next_request();
    }
    if (active_request_id_ == 0) {
        return false;
    }

    auto it = requests_.find(active_request_id_);
    if (it == requests_.end()) {
        active_request_id_ = 0;
        return true;
    }

    RequestState& request = it->second;
    try {
        if (is_terminal(request.status)) {
            if (debug_scheduler_enabled()) {
                std::cerr << "[SCHED] active reset"
                          << " id=" << request.id
                          << " status=" << request_status_name(request.status)
                          << std::endl;
            }
            active_request_id_ = 0;
            return true;
        }

        if (request.status == RequestStatus::RUNNING_PREFILL) {
            step_prefill(request);
            return true;
        }
        if (request.status == RequestStatus::RUNNING_DECODE) {
            step_decode(request);
            return true;
        }
    } catch (const std::exception& e) {
        fail_request(request, e.what());
        return true;
    } catch (...) {
        fail_request(request, "unknown scheduler exception");
        return true;
    }

    return false;
}

void LLMEngine::run_until_idle() {
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] run_until_idle start" << std::endl;
    }
    int guard = 0;
    while (has_pending_requests()) {
        if (!step_once()) {
            break;
        }
        if (++guard > 1000000) {
            std::cerr << "[SCHED] run_until_idle guard break" << std::endl;
            break;
        }
    }
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] run_until_idle end" << std::endl;
    }
}

void LLMEngine::run_until_finished(RequestId id) {
    int guard = 0;
    while (true) {
        auto it = requests_.find(id);
        if (it == requests_.end() || is_terminal(it->second.status)) {
            return;
        }
        if (!step_once()) {
            return;
        }
        int max_steps = static_cast<int>(it->second.prompt_tokens.size()) +
                        it->second.sampling.max_new_tokens * 2 + 1024;
        if (++guard > max_steps) {
            fail_request(it->second, "scheduler step guard exceeded");
            return;
        }
    }
}

bool LLMEngine::has_pending_requests() const {
    if (active_request_id_ != 0) {
        return true;
    }
    for (RequestId id : waiting_queue_) {
        auto it = requests_.find(id);
        if (it != requests_.end() && it->second.status == RequestStatus::WAITING) {
            return true;
        }
    }
    return false;
}

bool LLMEngine::request_finished(RequestId id) const {
    auto it = requests_.find(id);
    return it == requests_.end() || is_terminal(it->second.status);
}

bool LLMEngine::scheduler_enabled() const {
    return scheduler_enabled_;
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

bool LLMEngine::debug_scheduler_enabled() const {
    return env_flag("LLM_DEBUG_SCHEDULER");
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
    int max_cacheable_tokens = static_cast<int>(prompt_tokens.size()) - 1;
    if (max_cacheable_tokens < 0) {
        max_cacheable_tokens = 0;
    }
    int full_blocks = max_cacheable_tokens / block_size;
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

void LLMEngine::schedule_next_request() {
    while (!waiting_queue_.empty()) {
        RequestId id = waiting_queue_.front();
        waiting_queue_.pop_front();
        auto it = requests_.find(id);
        if (it == requests_.end() || it->second.status != RequestStatus::WAITING) {
            continue;
        }

        RequestState& request = it->second;
        if (request.prompt_tokens.empty()) {
            fail_request(request, "empty prompt is not supported");
            return;
        }

        if (session_cache_enabled_) {
            SequenceState& seq = get_or_create_session(request.session_id);
            apply_prefix_cache(seq, request.prompt_tokens);
            request.prefix_applied = prefix_cache_enabled_;
            request.cached_prefix_tokens = seq.cached_prefix_tokens;
            request.cached_prefix_blocks = seq.cached_prefix_blocks;
            request.prompt_cursor = seq.cached_prefix_tokens;
        }

        request.status = RequestStatus::RUNNING_PREFILL;
        active_request_id_ = request.id;
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] schedule"
                      << " id=" << request.id
                      << " session=" << request.session_id
                      << " status=" << request_status_name(request.status)
                      << " prompt_cursor=" << request.prompt_cursor
                      << " prompt_tokens=" << request.prompt_tokens.size()
                      << " cached_prefix_tokens=" << request.cached_prefix_tokens
                      << " cached_prefix_blocks=" << request.cached_prefix_blocks
                      << " queue=" << waiting_queue_.size()
                      << std::endl;
        }
        return;
    }
}

void LLMEngine::step_prefill(RequestState& request) {
    if (!session_cache_enabled_) {
        run_legacy_request(request);
        return;
    }

    SequenceState& seq = get_or_create_session(request.session_id);
    int budget = prefill_step_tokens_;
    while (budget-- > 0 &&
           request.prompt_cursor < static_cast<int>(request.prompt_tokens.size())) {
        int next = model_.prefill_one_for_sequence(
            seq,
            request.prompt_tokens,
            request.prompt_cursor,
            *kv_manager_,
            prefix_cache_.get());
        if (seq.status == SequenceStatus::FAILED) {
            fail_request(request, seq.error_message.empty() ? "prefill failed" : seq.error_message);
            return;
        }
        if (next < 0) {
            fail_request(request, "prefill forward failed");
            return;
        }
        request.next_token = next;
        request.prompt_cursor++;
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] prefill"
                      << " id=" << request.id
                      << " prompt_cursor=" << request.prompt_cursor
                      << " prompt_tokens=" << request.prompt_tokens.size()
                      << " next_token=" << request.next_token
                      << std::endl;
        }
    }

    if (request.prompt_cursor >= static_cast<int>(request.prompt_tokens.size())) {
        if (request.next_token < 0) {
            fail_request(request, "prefill produced no next token");
            return;
        }
        request.status = RequestStatus::RUNNING_DECODE;
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] transition"
                      << " id=" << request.id
                      << " status=" << request_status_name(request.status)
                      << std::endl;
        }
    }
}

void LLMEngine::step_decode(RequestState& request) {
    if (!session_cache_enabled_) {
        run_legacy_request(request);
        return;
    }
    if (request.next_token < 0) {
        fail_request(request, "decode next_token is invalid");
        return;
    }
    if (request.num_generated_tokens >= request.sampling.max_new_tokens) {
        finish_request(request);
        return;
    }

    SequenceState& seq = get_or_create_session(request.session_id);
    int token_id = request.next_token;
    if (is_stop_token(token_id)) {
        model_.decode_one_for_sequence(seq, token_id, *kv_manager_, prefix_cache_.get());
        if (seq.status == SequenceStatus::FAILED) {
            fail_request(request, seq.error_message.empty() ? "stop token forward failed" : seq.error_message);
            return;
        }
        finish_request(request);
        return;
    }

    request.generated_tokens.push_back(token_id);
    request.token_count++;
    request.num_generated_tokens++;
    seq.generated_tokens.push_back(token_id);

    bool keep_going = true;
    if (request.callback) {
        keep_going = request.callback(token_id);
    }
    if (!keep_going) {
        request.callback_stopped = true;
        request.status = RequestStatus::ABORTED;
        seq.status = SequenceStatus::ABORTED;
        request.callback = TokenCallback{};
        if (active_request_id_ == request.id) {
            active_request_id_ = 0;
        }
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] aborted"
                      << " id=" << request.id
                      << " generated=" << request.num_generated_tokens
                      << std::endl;
        }
        return;
    }

    int next = model_.decode_one_for_sequence(seq, token_id, *kv_manager_, prefix_cache_.get());
    if (seq.status == SequenceStatus::FAILED) {
        fail_request(request, seq.error_message.empty() ? "decode failed" : seq.error_message);
        return;
    }
    if (next < 0) {
        fail_request(request, "decode forward failed");
        return;
    }
    request.next_token = next;

    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] decode"
                  << " id=" << request.id
                  << " generated=" << request.num_generated_tokens
                  << " next_token=" << request.next_token
                  << std::endl;
    }

    if (request.num_generated_tokens >= request.sampling.max_new_tokens) {
        finish_request(request);
    }
}

void LLMEngine::run_legacy_request(RequestState& request) {
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] legacy_prefill"
                  << " id=" << request.id
                  << " prompt_tokens=" << request.prompt_tokens.size()
                  << std::endl;
    }
    request.status = RequestStatus::RUNNING_DECODE;
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] transition"
                  << " id=" << request.id
                  << " status=" << request_status_name(request.status)
                  << std::endl;
    }

    bool callback_stopped = false;
    auto wrapped_callback = [this, &request, &callback_stopped](int token_id) {
        if (request.status == RequestStatus::ABORTED) {
            callback_stopped = true;
            return false;
        }
        request.generated_tokens.push_back(token_id);
        request.token_count++;
        request.num_generated_tokens++;
        if (request.callback && !request.callback(token_id)) {
            callback_stopped = true;
            return false;
        }
        return true;
    };

    try {
        model_.generate(request.prompt_tokens, request.sampling.max_new_tokens, wrapped_callback);
        request.callback_stopped = callback_stopped;
        if (callback_stopped) {
            request.status = RequestStatus::ABORTED;
            request.callback = TokenCallback{};
            if (active_request_id_ == request.id) {
                active_request_id_ = 0;
            }
            if (debug_scheduler_enabled()) {
                std::cerr << "[SCHED] aborted"
                          << " id=" << request.id
                          << " generated=" << request.num_generated_tokens
                          << std::endl;
            }
            return;
        }
        finish_request(request);
        return;
    } catch (const std::exception& e) {
        fail_request(request, e.what());
        return;
    } catch (...) {
        fail_request(request, "unknown legacy scheduler exception");
        return;
    }
}

void LLMEngine::fail_request(RequestState& request, const std::string& error) {
    request.status = RequestStatus::FAILED;
    request.error_message = error;
    request.callback = TokenCallback{};
    if (active_request_id_ == request.id) {
        active_request_id_ = 0;
    }
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] failed"
                  << " id=" << request.id
                  << " error=" << error
                  << std::endl;
    }
}

void LLMEngine::finish_request(RequestState& request) {
    request.status = RequestStatus::FINISHED;
    auto seq_it = sessions_.find(request.session_id);
    if (seq_it != sessions_.end()) {
        seq_it->second.status = SequenceStatus::FINISHED;
    }
    request.callback = TokenCallback{};
    if (active_request_id_ == request.id) {
        active_request_id_ = 0;
    }
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] finished"
                  << " id=" << request.id
                  << " generated=" << request.num_generated_tokens
                  << std::endl;
    }
}

bool LLMEngine::is_terminal(RequestStatus status) const {
    return status == RequestStatus::FINISHED ||
           status == RequestStatus::ABORTED ||
           status == RequestStatus::FAILED;
}

const char* LLMEngine::request_status_name(RequestStatus status) const {
    switch (status) {
        case RequestStatus::WAITING:
            return "WAITING";
        case RequestStatus::RUNNING_PREFILL:
            return "RUNNING_PREFILL";
        case RequestStatus::RUNNING_DECODE:
            return "RUNNING_DECODE";
        case RequestStatus::FINISHED:
            return "FINISHED";
        case RequestStatus::ABORTED:
            return "ABORTED";
        case RequestStatus::FAILED:
            return "FAILED";
    }
    return "UNKNOWN";
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
