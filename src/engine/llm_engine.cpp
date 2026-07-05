#include "llm_engine/engine/llm_engine.h"

#include "../../model/model.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <algorithm>
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

bool env_flag_default(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    std::string s(v);
    if (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON") {
        return true;
    }
    if (s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF") {
        return false;
    }
    return default_value;
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
    continuous_batching_enabled_ = scheduler_enabled_ && env_flag("LLM_ENABLE_CONTINUOUS_BATCHING");
    max_active_decode_requests_ = env_int("LLM_CONT_BATCH_MAX_ACTIVE_DECODE", 8);
    cont_batch_decode_first_ = env_flag_default("LLM_CONT_BATCH_DECODE_FIRST", true);
    cont_batch_prefill_when_decode_empty_ =
        env_flag_default("LLM_CONT_BATCH_PREFILL_WHEN_DECODE_EMPTY", true);
    cont_batch_prefill_after_decode_ = env_flag("LLM_CONT_BATCH_PREFILL_AFTER_DECODE");
    cont_batch_max_prefill_chunks_per_step_ =
        env_int("LLM_CONT_BATCH_MAX_PREFILL_CHUNKS_PER_STEP", 1);
    cont_batch_conservative_executor_ =
        env_flag_default("LLM_CONT_BATCH_CONSERVATIVE_EXECUTOR", true);
    prefill_step_tokens_ = env_int("LLM_PREFILL_STEP_TOKENS", 1);
    chunked_prefill_enabled_ = env_flag("LLM_ENABLE_CHUNKED_PREFILL");
    chunked_prefill_strict_ = env_flag("LLM_CHUNKED_PREFILL_STRICT");
    prefill_chunk_size_ = env_int("LLM_PREFILL_CHUNK_SIZE", prefill_step_tokens_);
    if (prefill_chunk_size_ <= 0) {
        prefill_chunk_size_ = 1;
    }
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] enabled=" << (scheduler_enabled_ ? 1 : 0)
                  << " prefill_step_tokens=" << prefill_step_tokens_
                  << " chunked_prefill=" << (chunked_prefill_enabled_ ? 1 : 0)
                  << " prefill_chunk_size=" << prefill_chunk_size_
                  << std::endl;
    }
    if (debug_cont_batch_enabled()) {
        std::cerr << "[SCHED_V2] enabled=" << (continuous_batching_enabled_ ? 1 : 0)
                  << " max_active_decode=" << max_active_decode_requests_
                  << " decode_first=" << (cont_batch_decode_first_ ? 1 : 0)
                  << " prefill_when_decode_empty="
                  << (cont_batch_prefill_when_decode_empty_ ? 1 : 0)
                  << " prefill_after_decode=" << (cont_batch_prefill_after_decode_ ? 1 : 0)
                  << " max_prefill_chunks_per_step=" << cont_batch_max_prefill_chunks_per_step_
                  << " conservative_executor=" << (cont_batch_conservative_executor_ ? 1 : 0)
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
    request.enqueue_time_us = now_us();
    request.schedule_time_us = request.enqueue_time_us;
    request.paged_attention_baseline = snapshot_paged_attention_stats();
    request.metrics.request_id = id;
    request.metrics.session_id = session_id;
    request.metrics.prompt_tokens = static_cast<int>(prompt_tokens.size());
    request.metrics.max_new_tokens = sampling.max_new_tokens;
    request.scheduler_v2 = continuous_batching_enabled_;
    request.queued_waiting = continuous_batching_enabled_;
    request.metrics.continuous_batching_enabled = continuous_batching_enabled_;
    init_request_sampling(request);

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
            active.num_generated_tokens++;
            mark_first_token(active);

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
            state.cached_prefix_tokens = seq.cached_prefix_tokens;
            state.cached_prefix_blocks = seq.cached_prefix_blocks;
            state.metrics.cached_prefix_tokens = seq.cached_prefix_tokens;
            state.metrics.cached_prefix_blocks = seq.cached_prefix_blocks;
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
                callback_stopped = false;
            } else if (seq.status == SequenceStatus::ABORTED) {
                callback_stopped = true;
            }
            state.metrics.computed_prefill_tokens = seq.num_computed_tokens;
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

        if (state.status != RequestStatus::FAILED) {
            state.status = callback_stopped ? RequestStatus::ABORTED : RequestStatus::FINISHED;
            debug_log_finished(state);
        }
        state.metrics.generated_tokens = state.num_generated_tokens;
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
    emit_metrics_once(state);
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
    request.enqueue_time_us = now_us();
    request.paged_attention_baseline = snapshot_paged_attention_stats();
    request.metrics.request_id = id;
    request.metrics.session_id = session_id;
    request.metrics.prompt_tokens = static_cast<int>(prompt_tokens.size());
    request.metrics.max_new_tokens = sampling.max_new_tokens;
    request.scheduler_v2 = continuous_batching_enabled_;
    request.queued_waiting = continuous_batching_enabled_;
    request.metrics.continuous_batching_enabled = continuous_batching_enabled_;
    init_request_sampling(request);

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
        remove_request_from_all_v2_queues(id);
        emit_metrics_once(it->second);
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] abort"
                      << " id=" << id
                      << " active=" << active_request_id_
                      << std::endl;
        }
    }
}

bool LLMEngine::step_once() {
    if (continuous_batching_enabled_) {
        return step_once_continuous();
    }
    return step_once_legacy();
}

bool LLMEngine::step_once_legacy() {
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
    if (continuous_batching_enabled_) {
        if (!active_decode_requests_.empty() ||
            !prefill_queue_.empty() ||
            !decode_ready_queue_.empty()) {
            return true;
        }
    }
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
    if (continuous_batching_enabled_) {
        std::vector<RequestId> to_abort;
        for (const auto& kv : requests_) {
            const RequestState& request = kv.second;
            if (request.session_id == session_id && !is_terminal(request.status)) {
                to_abort.push_back(kv.first);
            }
        }
        for (RequestId id : to_abort) {
            auto req_it = requests_.find(id);
            if (req_it != requests_.end() && !is_terminal(req_it->second.status)) {
                req_it->second.status = RequestStatus::ABORTED;
                req_it->second.error_message = "session cleared";
                req_it->second.callback = TokenCallback{};
                remove_request_from_all_v2_queues(id);
                emit_metrics_once(req_it->second);
            }
        }
    }

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

bool LLMEngine::debug_cont_batch_enabled() const {
    return env_flag("LLM_CONT_BATCH_DEBUG");
}

bool LLMEngine::debug_cont_batch_verbose_enabled() const {
    return env_flag("LLM_CONT_BATCH_DEBUG_VERBOSE");
}

bool LLMEngine::debug_chunked_prefill_enabled() const {
    return env_flag("LLM_DEBUG_CHUNKED_PREFILL");
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

bool LLMEngine::step_once_continuous() {
    cleanup_terminal_requests_v2();
    admit_waiting_requests_v2();
    activate_decode_requests();

    bool did_work = false;
    if (cont_batch_decode_first_ && !active_decode_requests_.empty()) {
        did_work = run_decode_batch_step() || did_work;
        cleanup_terminal_requests_v2();
        activate_decode_requests();
        if (!cont_batch_prefill_after_decode_) {
            return did_work;
        }
    }

    int prefill_steps = 0;
    const int max_prefill_steps = std::max(1, cont_batch_max_prefill_chunks_per_step_);
    bool may_prefill = cont_batch_prefill_after_decode_ ||
                       active_decode_requests_.empty() ||
                       cont_batch_prefill_when_decode_empty_;
    while (may_prefill &&
           prefill_steps < max_prefill_steps &&
           !prefill_queue_.empty()) {
        if (!active_decode_requests_.empty() &&
            cont_batch_decode_first_ &&
            !cont_batch_prefill_after_decode_) {
            for (RequestId id : prefill_queue_) {
                auto it = requests_.find(id);
                if (it != requests_.end() && !is_terminal(it->second.status)) {
                    it->second.metrics.prefill_scheduler_yield_count++;
                }
            }
            break;
        }
        if (!run_prefill_chunk_step()) {
            break;
        }
        did_work = true;
        prefill_steps++;
        cleanup_terminal_requests_v2();
        activate_decode_requests();
        may_prefill = cont_batch_prefill_after_decode_ ||
                      active_decode_requests_.empty() ||
                      cont_batch_prefill_when_decode_empty_;
    }

    if (!cont_batch_decode_first_ && !active_decode_requests_.empty()) {
        did_work = run_decode_batch_step() || did_work;
        cleanup_terminal_requests_v2();
    }

    if (debug_cont_batch_verbose_enabled()) {
        std::cerr << "[SCHED_V2] queues"
                  << " waiting=" << waiting_queue_.size()
                  << " prefill=" << prefill_queue_.size()
                  << " decode_ready=" << decode_ready_queue_.size()
                  << " active_decode=" << active_decode_requests_.size()
                  << std::endl;
    }
    return did_work;
}

void LLMEngine::admit_waiting_requests_v2() {
    while (!waiting_queue_.empty()) {
        RequestId id = waiting_queue_.front();
        waiting_queue_.pop_front();
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            continue;
        }

        RequestState& request = it->second;
        request.queued_waiting = false;
        if (request.status != RequestStatus::WAITING) {
            continue;
        }
        if (request.prompt_tokens.empty()) {
            fail_request(request, "empty prompt is not supported");
            continue;
        }

        if (session_cache_enabled_) {
            SequenceState& seq = get_or_create_session(request.session_id);
            apply_prefix_cache(seq, request.prompt_tokens);
            request.prefix_applied = prefix_cache_enabled_;
            request.cached_prefix_tokens = seq.cached_prefix_tokens;
            request.cached_prefix_blocks = seq.cached_prefix_blocks;
            request.prompt_cursor = seq.cached_prefix_tokens;
            request.metrics.cached_prefix_tokens = seq.cached_prefix_tokens;
            request.metrics.cached_prefix_blocks = seq.cached_prefix_blocks;
        }

        request.status = RequestStatus::RUNNING_PREFILL;
        request.schedule_time_us = now_us();
        request.metrics.queue_wait_ms = elapsed_ms(request.enqueue_time_us, request.schedule_time_us);
        request.metrics.continuous_batching_enabled = true;
        add_to_prefill_queue(request);
        if (debug_cont_batch_enabled()) {
            std::cerr << "[SCHED_V2] enqueue_prefill"
                      << " id=" << request.id
                      << " session=" << request.session_id
                      << " prompt_cursor=" << request.prompt_cursor
                      << " prompt_tokens=" << request.prompt_tokens.size()
                      << " cached_prefix_tokens=" << request.cached_prefix_tokens
                      << " cached_prefix_blocks=" << request.cached_prefix_blocks
                      << " prefill_queue=" << prefill_queue_.size()
                      << std::endl;
        }
    }
}

bool LLMEngine::run_decode_batch_step() {
    activate_decode_requests();
    if (active_decode_requests_.empty()) {
        return false;
    }

    std::vector<RequestId> batch = active_decode_requests_;
    int batch_size = static_cast<int>(batch.size());
    if (debug_cont_batch_enabled()) {
        std::cerr << "[SCHED_V2] decode_batch"
                  << " size=" << batch_size
                  << " active_decode=" << active_decode_requests_.size()
                  << std::endl;
    }

    bool did_work = false;
    for (RequestId id : batch) {
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            remove_from_active_decode(id);
            continue;
        }
        RequestState& request = it->second;
        if (is_terminal(request.status) || request.status != RequestStatus::RUNNING_DECODE) {
            remove_from_active_decode(id);
            continue;
        }

        request.metrics.continuous_batching_enabled = true;
        request.metrics.scheduler_v2_steps++;
        request.metrics.scheduler_v2_decode_steps++;
        request.metrics.decode_batch_steps++;
        request.metrics.decode_batch_size_sum += batch_size;
        request.metrics.decode_batch_size_max =
            std::max(request.metrics.decode_batch_size_max, batch_size);

        try {
            step_decode(request);
        } catch (const std::exception& e) {
            fail_request(request, e.what());
        } catch (...) {
            fail_request(request, "unknown scheduler v2 decode exception");
        }
        did_work = true;
        if (is_terminal(request.status)) {
            remove_from_active_decode(id);
        }
    }
    return did_work;
}

bool LLMEngine::run_prefill_chunk_step() {
    while (!prefill_queue_.empty()) {
        RequestId id = prefill_queue_.front();
        prefill_queue_.pop_front();
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            continue;
        }

        RequestState& request = it->second;
        request.queued_prefill = false;
        if (is_terminal(request.status)) {
            continue;
        }
        if (request.status != RequestStatus::RUNNING_PREFILL) {
            if (request.status == RequestStatus::RUNNING_DECODE) {
                add_to_decode_ready_queue(request);
            }
            continue;
        }

        request.metrics.continuous_batching_enabled = true;
        request.metrics.scheduler_v2_steps++;
        request.metrics.scheduler_v2_prefill_steps++;
        request.metrics.prefill_chunk_steps++;
        if (debug_cont_batch_enabled()) {
            std::cerr << "[SCHED_V2] prefill_chunk"
                      << " id=" << request.id
                      << " cursor=" << request.prompt_cursor
                      << " prompt_tokens=" << request.prompt_tokens.size()
                      << std::endl;
        }

        try {
            step_prefill(request);
        } catch (const std::exception& e) {
            fail_request(request, e.what());
        } catch (...) {
            fail_request(request, "unknown scheduler v2 prefill exception");
        }

        if (request.status == RequestStatus::RUNNING_PREFILL) {
            add_to_prefill_queue(request);
        } else if (request.status == RequestStatus::RUNNING_DECODE) {
            add_to_decode_ready_queue(request);
        } else if (is_terminal(request.status)) {
            remove_request_from_all_v2_queues(request.id);
        }
        return true;
    }
    return false;
}

void LLMEngine::add_to_prefill_queue(RequestState& request) {
    if (request.queued_prefill || is_terminal(request.status)) {
        return;
    }
    request.queued_prefill = true;
    prefill_queue_.push_back(request.id);
}

void LLMEngine::add_to_decode_ready_queue(RequestState& request) {
    if (request.queued_decode_ready || request.active_decode || is_terminal(request.status)) {
        return;
    }
    request.queued_decode_ready = true;
    decode_ready_queue_.push_back(request.id);
    if (debug_cont_batch_enabled()) {
        std::cerr << "[SCHED_V2] enqueue_decode_ready"
                  << " id=" << request.id
                  << " decode_ready=" << decode_ready_queue_.size()
                  << std::endl;
    }
}

void LLMEngine::activate_decode_requests() {
    while (!decode_ready_queue_.empty() &&
           static_cast<int>(active_decode_requests_.size()) < max_active_decode_requests_) {
        RequestId id = decode_ready_queue_.front();
        decode_ready_queue_.pop_front();
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            continue;
        }
        RequestState& request = it->second;
        request.queued_decode_ready = false;
        if (request.status != RequestStatus::RUNNING_DECODE || is_terminal(request.status)) {
            continue;
        }
        if (request.active_decode) {
            continue;
        }
        request.active_decode = true;
        active_decode_requests_.push_back(id);
        if (debug_cont_batch_enabled()) {
            std::cerr << "[SCHED_V2] activate_decode"
                      << " id=" << request.id
                      << " active_decode=" << active_decode_requests_.size()
                      << std::endl;
        }
    }
}

void LLMEngine::remove_from_active_decode(RequestId id) {
    active_decode_requests_.erase(
        std::remove(active_decode_requests_.begin(), active_decode_requests_.end(), id),
        active_decode_requests_.end());
    auto it = requests_.find(id);
    if (it != requests_.end()) {
        it->second.active_decode = false;
    }
}

void LLMEngine::remove_request_from_all_v2_queues(RequestId id) {
    waiting_queue_.erase(
        std::remove(waiting_queue_.begin(), waiting_queue_.end(), id),
        waiting_queue_.end());
    prefill_queue_.erase(
        std::remove(prefill_queue_.begin(), prefill_queue_.end(), id),
        prefill_queue_.end());
    decode_ready_queue_.erase(
        std::remove(decode_ready_queue_.begin(), decode_ready_queue_.end(), id),
        decode_ready_queue_.end());
    active_decode_requests_.erase(
        std::remove(active_decode_requests_.begin(), active_decode_requests_.end(), id),
        active_decode_requests_.end());
    auto it = requests_.find(id);
    if (it != requests_.end()) {
        it->second.queued_waiting = false;
        it->second.queued_prefill = false;
        it->second.queued_decode_ready = false;
        it->second.active_decode = false;
    }
}

void LLMEngine::cleanup_terminal_requests_v2() {
    std::vector<RequestId> terminal_ids;
    for (const auto& kv : requests_) {
        if (kv.second.scheduler_v2 && is_terminal(kv.second.status)) {
            terminal_ids.push_back(kv.first);
        }
    }
    for (RequestId id : terminal_ids) {
        remove_request_from_all_v2_queues(id);
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
            request.metrics.cached_prefix_tokens = seq.cached_prefix_tokens;
            request.metrics.cached_prefix_blocks = seq.cached_prefix_blocks;
        }

        request.status = RequestStatus::RUNNING_PREFILL;
        request.schedule_time_us = now_us();
        request.metrics.queue_wait_ms = elapsed_ms(request.enqueue_time_us, request.schedule_time_us);
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
    int budget = (continuous_batching_enabled_ && request.scheduler_v2)
        ? 1
        : std::max(1, prefill_step_tokens_);
    while (budget-- > 0 &&
           request.prompt_cursor < static_cast<int>(request.prompt_tokens.size())) {
        int begin = request.prompt_cursor;
        int end = begin + 1;
        bool used_chunk = false;
        int next = -1;

        if (chunked_prefill_enabled_) {
            int remaining = static_cast<int>(request.prompt_tokens.size()) - begin;
            int chunk = std::max(1, std::min(prefill_chunk_size_, remaining));
            end = begin + chunk;
            {
                ScopedTimer timer(&request.metrics.prefill_ms);
                next = model_.prefill_chunk_for_sequence(
                    seq,
                    request.prompt_tokens,
                    begin,
                    end,
                    *kv_manager_,
                    prefix_cache_.get());
            }
            request.metrics.prefill_chunks++;
            request.metrics.batch_prefill_ms += model_.last_prefill_chunk_stats.batch_ms;
            request.metrics.token_loop_prefill_ms += model_.last_prefill_chunk_stats.token_loop_ms;
            if (model_.last_prefill_chunk_stats.real_batch_used) {
                request.metrics.real_batch_prefill_chunks++;
            }
            if (model_.last_prefill_chunk_stats.token_loop_used) {
                request.metrics.token_loop_prefill_chunks++;
            }
            if (model_.last_prefill_chunk_stats.fallback) {
                request.metrics.batch_prefill_fallbacks++;
            }
            if (model_.last_prefill_chunk_stats.compare_mismatch) {
                request.metrics.batch_prefill_compare_mismatches++;
            }
            used_chunk = next >= 0 && seq.status != SequenceStatus::FAILED;
        } else {
            ScopedTimer timer(&request.metrics.prefill_ms);
            next = model_.prefill_one_for_sequence(
                seq,
                request.prompt_tokens,
                begin,
                *kv_manager_,
                prefix_cache_.get());
            request.metrics.prefill_chunks++;
            request.metrics.token_loop_prefill_chunks++;
        }
        if (seq.status == SequenceStatus::FAILED) {
            fail_request(request, seq.error_message.empty() ? "prefill failed" : seq.error_message);
            return;
        }
        if (next < 0) {
            fail_request(request, "prefill forward failed");
            return;
        }
        request.next_token = next;
        request.prompt_cursor = end;
        request.metrics.computed_prefill_tokens += (end - begin);
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] prefill"
                      << " id=" << request.id
                      << " prompt_cursor=" << request.prompt_cursor
                      << " prompt_tokens=" << request.prompt_tokens.size()
                      << " next_token=" << request.next_token
                      << " chunk=" << (used_chunk ? (end - begin) : 1)
                      << std::endl;
        }
        if (chunked_prefill_enabled_) {
            break;
        }
    }

    if (request.prompt_cursor >= static_cast<int>(request.prompt_tokens.size())) {
        if (request.next_token < 0) {
            fail_request(request, "prefill produced no next token");
            return;
        }
        if (sampling_enabled(request.sampling)) {
            SamplingRuntimeStats sampling_stats;
            int sampled = model_.sample_next_token_from_last_logits(
                request.sampling,
                request.rng,
                &sampling_stats);
            if (sampled >= 0) {
                request.next_token = sampled;
            }
            request.metrics.sampled_tokens += sampling_stats.sampled_tokens;
            request.metrics.greedy_tokens += sampling_stats.greedy_tokens;
            request.metrics.sampling_ms += sampling_stats.sampling_ms;
        } else {
            request.metrics.greedy_tokens++;
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
    ScopedTimer decode_timer(&request.metrics.decode_ms);
    int token_id = request.next_token;
    if (request_stop_token(request, token_id)) {
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
    mark_first_token(request);

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
        emit_metrics_once(request);
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] aborted"
                      << " id=" << request.id
                      << " generated=" << request.num_generated_tokens
                      << std::endl;
        }
        return;
    }

    int next = -1;
    if (sampling_enabled(request.sampling)) {
        SamplingRuntimeStats sampling_stats;
        next = model_.decode_one_for_sequence_sampled(
            seq,
            token_id,
            *kv_manager_,
            prefix_cache_.get(),
            request.sampling,
            request.rng,
            &sampling_stats);
        request.metrics.sampled_tokens += sampling_stats.sampled_tokens;
        request.metrics.greedy_tokens += sampling_stats.greedy_tokens;
        request.metrics.sampling_ms += sampling_stats.sampling_ms;
    } else {
        next = model_.decode_one_for_sequence(seq, token_id, *kv_manager_, prefix_cache_.get());
        request.metrics.greedy_tokens++;
    }
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
        mark_first_token(request);
        if (request.callback && !request.callback(token_id)) {
            callback_stopped = true;
            return false;
        }
        return true;
    };

    try {
        if (sampling_enabled(request.sampling)) {
            ScopedTimer timer(&request.metrics.decode_ms);
            if (!model_.kv_cache) {
                fail_request(request, "missing kv cache");
                return;
            }

            int next_token = -1;
            for (int tok : request.prompt_tokens) {
                if (model_.history_pos >= model_.config.max_seq_len) {
                    fail_request(request, "sequence length exceeded");
                    return;
                }
                next_token = model_.forward(tok, model_.history_pos, *model_.kv_cache);
                model_.history_pos++;
                if (next_token < 0) {
                    fail_request(request, "prompt forward failed");
                    return;
                }
            }

            SamplingRuntimeStats sampling_stats;
            int sampled = model_.sample_next_token_from_last_logits(
                request.sampling,
                request.rng,
                &sampling_stats);
            if (sampled >= 0) {
                next_token = sampled;
            }
            request.metrics.sampled_tokens += sampling_stats.sampled_tokens;
            request.metrics.greedy_tokens += sampling_stats.greedy_tokens;
            request.metrics.sampling_ms += sampling_stats.sampling_ms;

            int current_token = next_token;
            for (int i = 0;
                 i < request.sampling.max_new_tokens && current_token >= 0;
                 ++i) {
                if (request_stop_token(request, current_token)) {
                    if (model_.history_pos < model_.config.max_seq_len) {
                        int ignored = model_.forward(
                            current_token,
                            model_.history_pos,
                            *model_.kv_cache);
                        model_.history_pos++;
                        if (ignored < 0) {
                            fail_request(request, "stop token forward failed");
                            return;
                        }
                    }
                    break;
                }

                if (!wrapped_callback(current_token)) {
                    callback_stopped = true;
                    break;
                }
                if (model_.history_pos >= model_.config.max_seq_len) {
                    break;
                }

                int greedy_next = model_.forward(
                    current_token,
                    model_.history_pos,
                    *model_.kv_cache);
                model_.history_pos++;
                if (greedy_next < 0) {
                    fail_request(request, "decode forward failed");
                    return;
                }

                SamplingRuntimeStats step_stats;
                int sampled_next = model_.sample_next_token_from_last_logits(
                    request.sampling,
                    request.rng,
                    &step_stats);
                request.metrics.sampled_tokens += step_stats.sampled_tokens;
                request.metrics.greedy_tokens += step_stats.greedy_tokens;
                request.metrics.sampling_ms += step_stats.sampling_ms;
                current_token = sampled_next >= 0 ? sampled_next : greedy_next;
            }

            request.callback_stopped = callback_stopped;
            if (callback_stopped) {
                request.status = RequestStatus::ABORTED;
                request.callback = TokenCallback{};
                if (active_request_id_ == request.id) {
                    active_request_id_ = 0;
                }
                emit_metrics_once(request);
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
        }

        ScopedTimer timer(&request.metrics.decode_ms);
        model_.generate(request.prompt_tokens, request.sampling.max_new_tokens, wrapped_callback);
        request.callback_stopped = callback_stopped;
        if (callback_stopped) {
            request.status = RequestStatus::ABORTED;
            request.callback = TokenCallback{};
            if (active_request_id_ == request.id) {
                active_request_id_ = 0;
            }
            emit_metrics_once(request);
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
    if (request.scheduler_v2) {
        request.metrics.active_decode_batch_size_at_finish =
            std::max(request.metrics.active_decode_batch_size_at_finish,
                     static_cast<int>(active_decode_requests_.size()));
        remove_request_from_all_v2_queues(request.id);
    }
    emit_metrics_once(request);
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
    if (request.scheduler_v2) {
        request.metrics.active_decode_batch_size_at_finish =
            std::max(request.metrics.active_decode_batch_size_at_finish,
                     static_cast<int>(active_decode_requests_.size()));
        remove_request_from_all_v2_queues(request.id);
    }
    emit_metrics_once(request);
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] finished"
                  << " id=" << request.id
                  << " generated=" << request.num_generated_tokens
                  << std::endl;
    }
}

void LLMEngine::emit_metrics_once(RequestState& request) {
    if (request.metrics_emitted) {
        return;
    }
    request.metrics_emitted = true;
    request.finish_time_us = now_us();

    request.metrics.request_id = request.id;
    request.metrics.session_id = request.session_id;
    request.metrics.prompt_tokens = static_cast<int>(request.prompt_tokens.size());
    request.metrics.generated_tokens = request.num_generated_tokens;
    request.metrics.max_new_tokens = request.sampling.max_new_tokens;
    request.metrics.cached_prefix_tokens = request.cached_prefix_tokens;
    request.metrics.cached_prefix_blocks = request.cached_prefix_blocks;
    request.metrics.final_status = request_status_name(request.status);
    request.metrics.error_message = request.error_message;
    request.metrics.continuous_batching_enabled =
        request.metrics.continuous_batching_enabled || request.scheduler_v2;
    if (request.metrics.decode_batch_steps > 0) {
        request.metrics.decode_batch_size_avg =
            static_cast<double>(request.metrics.decode_batch_size_sum) /
            static_cast<double>(request.metrics.decode_batch_steps);
    }
    if (request.scheduler_v2 && request.metrics.active_decode_batch_size_at_finish <= 0) {
        request.metrics.active_decode_batch_size_at_finish =
            static_cast<int>(active_decode_requests_.size());
    }
    request.metrics.total_ms = elapsed_ms(request.enqueue_time_us, request.finish_time_us);
    if (request.metrics.first_token_ms <= 0.0 && request.first_token_time_us > 0) {
        request.metrics.first_token_ms =
            elapsed_ms(request.enqueue_time_us, request.first_token_time_us);
    }
    if (request.metrics.total_ms > 0.0 && request.metrics.generated_tokens > 0) {
        request.metrics.tokens_per_second =
            static_cast<double>(request.metrics.generated_tokens) /
            (request.metrics.total_ms / 1000.0);
    }

    if (kv_manager_) {
        request.metrics.kv_total_blocks = kv_manager_->total_blocks();
        request.metrics.kv_free_blocks = kv_manager_->free_blocks();
        request.metrics.kv_active_blocks = kv_manager_->active_blocks();
        request.metrics.kv_cached_blocks = kv_manager_->cached_blocks();
        request.metrics.kv_lru_size = kv_manager_->lru_size();
    }

    PagedAttentionStats end = snapshot_paged_attention_stats();
    PagedAttentionStats delta = diff_paged_attention_stats(
        request.paged_attention_baseline,
        end);
    request.metrics.paged_attention_calls = delta.calls;
    request.metrics.paged_attention_fallbacks = delta.fallbacks;
    request.metrics.paged_attention_compare_warnings = delta.compare_warnings;

    emit_request_metrics(request.metrics);
}

void LLMEngine::mark_first_token(RequestState& request) {
    if (request.first_token_emitted) {
        return;
    }
    request.first_token_emitted = true;
    request.first_token_time_us = now_us();
    request.metrics.first_token_ms =
        elapsed_ms(request.enqueue_time_us, request.first_token_time_us);
}

void LLMEngine::init_request_sampling(RequestState& request) {
    if (request.sampling.temperature < 0.0f) {
        request.sampling.temperature = 0.0f;
    }
    if (request.sampling.top_k < 0) {
        request.sampling.top_k = 0;
    }
    if (request.sampling.top_p <= 0.0f || request.sampling.top_p > 1.0f) {
        request.sampling.top_p = 1.0f;
    }
    if (request.sampling.temperature > 0.0f) {
        request.sampling.greedy = false;
    }

    uint64_t seed = request.sampling.has_seed
        ? request.sampling.seed
        : (now_us() ^ (request.id * 0x9e3779b97f4a7c15ULL));
    request.sampling.seed = seed;
    request.sampling.has_seed = true;
    request.rng.seed(seed);

    request.metrics.sampling_enabled = sampling_enabled(request.sampling);
    request.metrics.temperature = request.sampling.temperature;
    request.metrics.top_k = request.sampling.top_k;
    request.metrics.top_p = request.sampling.top_p;
    request.metrics.seed = seed;
}

bool LLMEngine::request_stop_token(const RequestState& request, int token_id) const {
    if (is_stop_token(token_id)) {
        return true;
    }
    for (int stop_id : request.sampling.stop_token_ids) {
        if (token_id == stop_id) {
            return true;
        }
    }
    return false;
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
