#include "llm_engine/engine/llm_engine.h"
#include "llm_engine/engine/mixed_batch.h"

#include "../../model/model.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
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

struct LLMEngine::SelectiveDecodeScratch {
    std::vector<RequestId> selected;
    std::vector<RequestId> model_ids;
    std::vector<QwenModel::SelectiveDecodeItem> items;
    std::vector<QwenModel::SelectiveDecodeOutput> outputs;
};

struct LLMEngine::MixedBatchScratch {
    std::vector<RequestId> request_ids;
    std::vector<int> prompt_begins;
    std::vector<bool> finish_after_forward;
    std::vector<int> token_ids;
    std::vector<QwenModel::MixedBatchItem> items;
    std::vector<QwenModel::MixedBatchOutput> outputs;

    void clear() {
        request_ids.clear();
        prompt_begins.clear();
        finish_after_forward.clear();
        token_ids.clear();
        items.clear();
        outputs.clear();
    }
};

LLMEngine::LLMEngine(QwenModel& model) : model_(model) {
    scheduler_enabled_ = env_flag("LLM_ENABLE_SCHEDULER");
    selective_decode_enabled_ = scheduler_enabled_;
    selective_decode_max_batch_ =
        env_int("LLM_SELECTIVE_DECODE_MAX_BATCH", 8);
    selective_decode_min_batch_ =
        env_int("LLM_SELECTIVE_DECODE_MIN_BATCH", 2);
    selective_decode_debug_ =
        env_flag("LLM_SELECTIVE_DECODE_DEBUG");
    max_batched_tokens_ = env_int("LLM_MAX_BATCHED_TOKENS", 64);
    max_batched_tokens_ = std::max(1, std::min(max_batched_tokens_, 256));
    prefill_chunk_size_ = env_int("LLM_PREFILL_CHUNK_SIZE", max_batched_tokens_);
    max_prefill_tokens_with_decode_ =
        env_int("LLM_MAX_PREFILL_TOKENS_WITH_DECODE", max_batched_tokens_);
    max_prefill_tokens_with_decode_ = std::min(
        max_prefill_tokens_with_decode_, max_batched_tokens_);
    if (prefill_chunk_size_ <= 0) {
        prefill_chunk_size_ = 1;
    }
    if (selective_decode_max_batch_ <= 0) {
        selective_decode_max_batch_ = 1;
    }
    if (selective_decode_min_batch_ <= 0) {
        selective_decode_min_batch_ = 1;
    }
    selective_decode_max_batch_ = std::min(selective_decode_max_batch_, 8);
    selective_decode_min_batch_ = std::min(selective_decode_min_batch_, selective_decode_max_batch_);
    selective_decode_scratch_ = std::make_unique<SelectiveDecodeScratch>();
    selective_decode_scratch_->selected.reserve(8);
    selective_decode_scratch_->model_ids.reserve(8);
    selective_decode_scratch_->items.reserve(8);
    selective_decode_scratch_->outputs.reserve(8);
    mixed_batch_scratch_ = std::make_unique<MixedBatchScratch>();
    mixed_batch_scratch_->request_ids.reserve((size_t)max_batched_tokens_);
    mixed_batch_scratch_->prompt_begins.reserve((size_t)max_batched_tokens_);
    mixed_batch_scratch_->finish_after_forward.reserve((size_t)max_batched_tokens_);
    mixed_batch_scratch_->token_ids.reserve((size_t)max_batched_tokens_);
    mixed_batch_scratch_->items.reserve((size_t)max_batched_tokens_);
    mixed_batch_scratch_->outputs.reserve((size_t)max_batched_tokens_);
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] enabled=" << (scheduler_enabled_ ? 1 : 0)
                  << " prefill_chunk_size=" << prefill_chunk_size_
                  << " max_batched_tokens=" << max_batched_tokens_
                  << " max_prefill_with_decode=" << max_prefill_tokens_with_decode_
                  << " mixed_batch=" << (scheduler_enabled_ ? 1 : 0)
                  << std::endl;
    }
    if (selective_decode_debug_) {
        std::cerr << "[SELECTIVE_DECODE] enabled=" << (selective_decode_enabled_ ? 1 : 0)
                  << " max_batch=" << selective_decode_max_batch_
                  << " min_batch=" << selective_decode_min_batch_
                  << std::endl;
    }

    bool requested_prefix_cache = env_flag("LLM_ENABLE_PREFIX_CACHE");
    if (scheduler_enabled_ && model_.kv_cache && model_.kv_cache->is_paged()) {
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
    } else if (scheduler_enabled_) {
        throw std::runtime_error("scheduler requires paged KV cache");
    }
    if (requested_prefix_cache && !prefix_cache_enabled_) {
        std::cerr << "[PREFIX] LLM_ENABLE_PREFIX_CACHE ignored because paged session cache is not enabled"
                  << std::endl;
    }

    if (scheduler_enabled_) {
        // Mixed Plan 的节点结构与本轮 token/Sequence 无关。引擎初始化时使用
        // Scheduler 的总 token budget 一次性完成建图、编译和固定 Workspace 分配；
        // 首个请求及后续请求都只执行 plan.run()，不再根据实际 T 扩容。
        model_.build_mixed_batch_plan(max_batched_tokens_);
    }
}

LLMEngine::~LLMEngine() = default;

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
    request.metrics.scheduler_enabled = false;
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

        model_.generate(
            state.prompt_tokens,
            state.sampling.max_new_tokens,
            wrapped_callback);

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
    if (!scheduler_enabled_) {
        throw std::logic_error("submit_async requires LLM_ENABLE_SCHEDULER=1");
    }
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
    request.queued_waiting = true;
    request.metrics.scheduler_enabled = true;
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
        remove_request_from_all_queues(id);
        emit_metrics_once(it->second);
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] abort"
                      << " id=" << id
                      << std::endl;
        }
    }
}

bool LLMEngine::step_once() {
    return scheduler_enabled_ && step_once_scheduled();
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
    if (!decode_queue_.empty() || !prefill_queue_.empty()) {
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
    if (scheduler_enabled_) {
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
                remove_request_from_all_queues(id);
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

bool LLMEngine::step_once_scheduled() {
    cleanup_terminal_requests();
    admit_waiting_requests();

    // Greedy 主路径每个 tick 只执行一次模型：Decode 行先占 token budget，
    // Prefill chunk 再填剩余行，避免同一层权重被 Decode/Prefill 分别扫描两遍。
    bool did_work = run_mixed_batch_step();
    cleanup_terminal_requests();

    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] queues"
                  << " waiting=" << waiting_queue_.size()
                  << " prefill=" << prefill_queue_.size()
                  << " decode=" << decode_queue_.size()
                  << std::endl;
    }
    return did_work;
}

void LLMEngine::admit_waiting_requests() {
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
        request.metrics.scheduler_enabled = true;
        add_to_prefill_queue(request);
        if (debug_scheduler_enabled()) {
            std::cerr << "[SCHED] enqueue_prefill"
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

bool LLMEngine::has_sampling_work() const {
    auto request_uses_sampling = [&](RequestId id) {
        auto it = requests_.find(id);
        return it != requests_.end() && !is_terminal(it->second.status) &&
               sampling_enabled(it->second.sampling);
    };
    for (RequestId id : decode_queue_) {
        if (request_uses_sampling(id)) return true;
    }
    for (RequestId id : prefill_queue_) {
        if (request_uses_sampling(id)) return true;
    }
    return false;
}

bool LLMEngine::run_mixed_batch_step() {
    // 当前融合 LM Head 只产生 Argmax。Sampling 需要完整 logits，因此整轮回退到
    // 原有路径，不能把 greedy 结果误当成 top-k/top-p 结果。
    if (has_sampling_work()) {
        bool did_work = false;
        if (!decode_queue_.empty()) {
            did_work = run_decode_batch_step_conservative() || did_work;
        }
        if (!prefill_queue_.empty()) {
            did_work = run_prefill_step() || did_work;
        }
        return did_work;
    }

    MixedBatchScratch& scratch = *mixed_batch_scratch_;
    scratch.clear();
    bool did_work = false;
    int decode_rows = 0;

    // 第一阶段：Decode 优先。每个活跃请求最多贡献一行。
    const std::deque<RequestId> decode_snapshot = decode_queue_;
    for (RequestId id : decode_snapshot) {
        if (decode_rows >= max_batched_tokens_) break;
        auto it = requests_.find(id);
        if (it == requests_.end()) continue;
        RequestState& request = it->second;
        if (is_terminal(request.status) || request.status != RequestStatus::RUNNING_DECODE) {
            remove_request_from_all_queues(id);
            continue;
        }
        if (request.next_token < 0) {
            fail_request(request, "decode next_token is invalid");
            did_work = true;
            continue;
        }
        if (request.num_generated_tokens >= request.sampling.max_new_tokens) {
            finish_request(request);
            did_work = true;
            continue;
        }

        SequenceState& seq = get_or_create_session(request.session_id);
        if (seq.history_pos >= model_.config.max_seq_len) {
            fail_request(request, "sequence length exceeded");
            did_work = true;
            continue;
        }

        const int token_id = request.next_token;
        const bool stop_token = request_stop_token(request, token_id);
        bool finish_after_forward = stop_token;

        if (!stop_token) {
            // 与旧 Decode 语义一致：先向用户发出当前 token，再把它送进模型求下一 token。
            request.generated_tokens.push_back(token_id);
            request.token_count++;
            request.num_generated_tokens++;
            seq.generated_tokens.push_back(token_id);
            mark_first_token(request);

            bool keep_going = true;
            if (request.callback) keep_going = request.callback(token_id);
            if (!keep_going) {
                request.callback_stopped = true;
                request.status = RequestStatus::ABORTED;
                seq.status = SequenceStatus::ABORTED;
                request.callback = TokenCallback{};
                remove_request_from_all_queues(request.id);
                emit_metrics_once(request);
                did_work = true;
                continue;
            }
            finish_after_forward =
                request.num_generated_tokens >= request.sampling.max_new_tokens;
        }

        const int row_begin = static_cast<int>(scratch.token_ids.size());
        scratch.token_ids.push_back(token_id);
        scratch.items.push_back(QwenModel::MixedBatchItem{
            QwenModel::MixedBatchItemKind::DECODE,
            &seq,
            row_begin,
            1,
            seq.history_pos,
            !finish_after_forward});
        scratch.request_ids.push_back(id);
        scratch.prompt_begins.push_back(-1);
        scratch.finish_after_forward.push_back(finish_after_forward);
        decode_rows++;
    }

    // 第二阶段：用剩余 token budget 填充一个或多个 Prefill chunk。
    const MixedBatchBudget budget = plan_mixed_batch_budget(
        max_batched_tokens_, decode_rows, std::numeric_limits<int>::max(),
        max_prefill_tokens_with_decode_);
    int prefill_budget = budget.prefill_rows;
    size_t candidates = prefill_queue_.size();
    while (prefill_budget > 0 && candidates-- > 0 && !prefill_queue_.empty()) {
        const RequestId id = prefill_queue_.front();
        prefill_queue_.pop_front();
        auto it = requests_.find(id);
        if (it == requests_.end()) continue;

        RequestState& request = it->second;
        request.queued_prefill = false;
        if (is_terminal(request.status)) continue;
        if (request.status != RequestStatus::RUNNING_PREFILL) {
            if (request.status == RequestStatus::RUNNING_DECODE) {
                add_to_decode_queue(request);
            }
            continue;
        }

        SequenceState& seq = get_or_create_session(request.session_id);
        const int begin = request.prompt_cursor;
        const int prompt_remaining =
            static_cast<int>(request.prompt_tokens.size()) - begin;
        if (prompt_remaining <= 0) {
            transition_prefill_complete(request);
            did_work = true;
            continue;
        }

        const int chunk_rows = std::min({
            prompt_remaining,
            prefill_chunk_size_,
            prefill_budget,
            model_.config.max_seq_len - seq.history_pos});
        if (chunk_rows <= 0) {
            fail_request(request, "sequence length exceeded");
            did_work = true;
            continue;
        }

        const int row_begin = static_cast<int>(scratch.token_ids.size());
        scratch.token_ids.insert(
            scratch.token_ids.end(),
            request.prompt_tokens.begin() + begin,
            request.prompt_tokens.begin() + begin + chunk_rows);
        const bool finishes_prompt = begin + chunk_rows >=
            static_cast<int>(request.prompt_tokens.size());
        scratch.items.push_back(QwenModel::MixedBatchItem{
            QwenModel::MixedBatchItemKind::PREFILL,
            &seq,
            row_begin,
            chunk_rows,
            seq.history_pos,
            finishes_prompt});
        scratch.request_ids.push_back(id);
        scratch.prompt_begins.push_back(begin);
        scratch.finish_after_forward.push_back(false);
        prefill_budget -= chunk_rows;
    }

    if (scratch.items.empty()) return did_work;

    QwenModel::MixedBatchStats stats;
    const bool ok = model_.run_mixed_batch_for_sequences(
        scratch.token_ids,
        scratch.items,
        *kv_manager_,
        prefix_cache_.get(),
        &scratch.outputs,
        &stats);
    did_work = true;

    if (!ok || scratch.outputs.size() != scratch.items.size()) {
        const std::string error = stats.state_modified
            ? "mixed batch failed after KV mutation"
            : "mixed batch execution failed";
        for (RequestId id : scratch.request_ids) {
            auto it = requests_.find(id);
            if (it != requests_.end() && !is_terminal(it->second.status)) {
                it->second.metrics.mixed_batch_enabled = true;
                it->second.metrics.mixed_batch_fallbacks++;
                it->second.metrics.mixed_batch_mode = "mixed_failed";
                it->second.metrics.selective_decode_fallbacks++;
                it->second.metrics.selective_decode_mode = "mixed_failed";
                fail_request(it->second, error);
            }
        }
        return true;
    }

    // 第三阶段：模型成功后，再分别提交 RequestState 的 cursor/next_token/status。
    for (size_t item_index = 0; item_index < scratch.items.size(); ++item_index) {
        auto request_it = requests_.find(scratch.request_ids[item_index]);
        if (request_it == requests_.end()) continue;
        RequestState& request = request_it->second;
        if (is_terminal(request.status)) continue;

        const QwenModel::MixedBatchItem& item = scratch.items[item_index];
        const QwenModel::MixedBatchOutput& output = scratch.outputs[item_index];
        if (!output.success) {
            fail_request(request, output.error_message.empty()
                ? "mixed batch item failed" : output.error_message);
            continue;
        }

        request.metrics.scheduler_enabled = true;
        request.metrics.scheduler_steps++;
        request.metrics.mixed_batch_enabled = true;
        request.metrics.mixed_batch_steps++;
        request.metrics.mixed_batch_total_rows_sum += stats.total_rows;
        request.metrics.mixed_batch_total_rows_max =
            std::max(request.metrics.mixed_batch_total_rows_max, stats.total_rows);
        request.metrics.mixed_batch_decode_rows += stats.decode_rows;
        request.metrics.mixed_batch_prefill_rows += stats.prefill_rows;
        request.metrics.mixed_batch_attention_segments += stats.attention_sequence_segments;
        request.metrics.mixed_batch_lm_head_rows += stats.lm_head_rows;
        request.metrics.mixed_batch_model_ms += stats.model_ms;
        request.metrics.mixed_batch_mode = "decode_first_token_budget";
        auto share_counter = [&](uint64_t total) {
            const uint64_t count = static_cast<uint64_t>(scratch.items.size());
            return total / count + (item_index < total % count ? 1ULL : 0ULL);
        };
        request.metrics.gptq_batch_kernel_calls += share_counter(stats.gptq_batch_kernel_calls);
        request.metrics.gptq_batch_rows_total += share_counter(stats.gptq_batch_rows_total);
        request.metrics.gptq_batch_output_panel_tasks +=
            share_counter(stats.gptq_batch_output_panel_tasks);
        request.metrics.gptq_batch_row_gemv_fallbacks +=
            share_counter(stats.gptq_batch_row_gemv_fallbacks);
        request.metrics.gptq_batch_weight_vector_loads +=
            share_counter(stats.gptq_batch_weight_vector_loads);
        request.metrics.gptq_batch_dequant_vector_ops +=
            share_counter(stats.gptq_batch_dequant_vector_ops);
        request.metrics.gptq_batch_argmax_calls += share_counter(stats.gptq_batch_argmax_calls);
        request.metrics.gptq_batch_argmax_rows += share_counter(stats.gptq_batch_argmax_rows);
        request.metrics.gptq_batch_full_logits_elements_written +=
            share_counter(stats.gptq_batch_full_logits_elements_written);
        request.metrics.gptq_batch_compare_mismatches +=
            share_counter(stats.gptq_batch_compare_mismatches);
        request.metrics.selective_decode_hotpath_allocations +=
            share_counter(stats.mixed_batch_hotpath_allocations);
        request.metrics.selective_decode_workspace_reallocations +=
            share_counter(stats.mixed_batch_workspace_reallocations);
        if (item.kind == QwenModel::MixedBatchItemKind::DECODE) {
            request.metrics.scheduler_decode_steps++;
            request.metrics.decode_batch_steps++;
            request.metrics.decode_batch_size_sum += stats.decode_rows;
            request.metrics.decode_batch_size_max =
                std::max(request.metrics.decode_batch_size_max, stats.decode_rows);
            request.metrics.selective_decode_enabled = true;
            request.metrics.selective_decode_steps++;
            request.metrics.selective_decode_size_sum += stats.decode_rows;
            request.metrics.selective_decode_size_max =
                std::max(request.metrics.selective_decode_size_max, stats.decode_rows);
            request.metrics.selective_decode_linear_batch_rows += stats.linear_batch_rows;
            request.metrics.selective_decode_attention_per_sequence_calls +=
                stats.attention_sequence_segments;
            request.metrics.selective_decode_lm_head_rows += stats.lm_head_rows;
            request.metrics.selective_decode_model_ms += stats.model_ms;
            request.metrics.selective_decode_mode = "mixed_decode_prefill";
            request.metrics.decode_ms += stats.model_ms;

            if (scratch.finish_after_forward[item_index]) {
                finish_request(request);
            } else if (output.next_token < 0) {
                fail_request(request, "mixed decode produced no next token");
            } else {
                request.next_token = output.next_token;
                request.metrics.greedy_tokens++;
            }
        } else {
            const int begin = scratch.prompt_begins[item_index];
            request.prompt_cursor = begin + item.row_count;
            request.metrics.scheduler_prefill_steps++;
            request.metrics.prefill_chunk_steps++;
            request.metrics.prefill_chunks++;
            request.metrics.computed_prefill_tokens += item.row_count;
            request.metrics.prefill_ms += stats.model_ms;
            request.metrics.batch_prefill_ms += stats.model_ms;
            request.metrics.real_batch_prefill_chunks++;

            if (request.prompt_cursor >= static_cast<int>(request.prompt_tokens.size())) {
                if (output.next_token < 0) {
                    fail_request(request, "mixed prefill produced no next token");
                } else {
                    request.next_token = output.next_token;
                    transition_prefill_complete(request);
                }
            }
            if (request.status == RequestStatus::RUNNING_PREFILL) {
                add_to_prefill_queue(request);
            }
        }
    }

    if (debug_scheduler_enabled()) {
        std::cerr << "[MIXED_BATCH] items=" << stats.item_count
                  << " total_rows=" << stats.total_rows
                  << " decode_rows=" << stats.decode_rows
                  << " prefill_rows=" << stats.prefill_rows
                  << " lm_head_rows=" << stats.lm_head_rows
                  << " attention_segments=" << stats.attention_sequence_segments
                  << std::endl;
    }
    return did_work;
}

bool LLMEngine::run_decode_batch_step() {
    if (selective_decode_enabled_ && run_selective_decode_batch_step()) {
        return true;
    }
    return run_decode_batch_step_conservative();
}

bool LLMEngine::run_decode_batch_step_conservative() {
    if (decode_queue_.empty()) {
        return false;
    }

    std::deque<RequestId> batch = decode_queue_;
    int batch_size = static_cast<int>(batch.size());
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] decode_batch"
                  << " size=" << batch_size
                  << " decode=" << decode_queue_.size()
                  << std::endl;
    }

    bool did_work = false;
    for (RequestId id : batch) {
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            remove_request_from_all_queues(id);
            continue;
        }
        RequestState& request = it->second;
        if (is_terminal(request.status) || request.status != RequestStatus::RUNNING_DECODE) {
            remove_request_from_all_queues(id);
            continue;
        }

        request.metrics.scheduler_enabled = true;
        request.metrics.scheduler_steps++;
        request.metrics.scheduler_decode_steps++;
        request.metrics.decode_batch_steps++;
        request.metrics.decode_batch_size_sum += batch_size;
        request.metrics.decode_batch_size_max =
            std::max(request.metrics.decode_batch_size_max, batch_size);

        try {
            step_decode(request);
        } catch (const std::exception& e) {
            fail_request(request, e.what());
        } catch (...) {
            fail_request(request, "unknown scheduler decode exception");
        }
        did_work = true;
        if (is_terminal(request.status)) {
            remove_request_from_all_queues(id);
        }
    }
    return did_work;
}

bool LLMEngine::build_selective_decode_batch(std::vector<RequestId>* selected) {
    if (!selected) {
        return false;
    }
    selected->clear();
    for (RequestId id : decode_queue_) {
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            continue;
        }
        RequestState& request = it->second;
        if (is_terminal(request.status) ||
            request.status != RequestStatus::RUNNING_DECODE ||
            request.next_token < 0 ||
            request.num_generated_tokens >= request.sampling.max_new_tokens) {
            continue;
        }
        if (request_stop_token(request, request.next_token)) {
            continue;
        }
        if (sampling_enabled(request.sampling)) {
            request.metrics.selective_decode_fallbacks++;
            request.metrics.selective_decode_mode = "fallback_sampling";
            continue;
        }
        selected->push_back(id);
        if (static_cast<int>(selected->size()) >= selective_decode_max_batch_) {
            break;
        }
    }
    if (static_cast<int>(selected->size()) < selective_decode_min_batch_) {
        if (selective_decode_debug_) {
            std::cerr << "[SELECTIVE_DECODE] fallback reason=batch_too_small"
                      << " decode=" << decode_queue_.size()
                      << " selected=" << selected->size()
                      << std::endl;
        }
        for (RequestId id : *selected) {
            auto it = requests_.find(id);
            if (it != requests_.end()) {
                it->second.metrics.selective_decode_fallbacks++;
                it->second.metrics.selective_decode_mode = "fallback_batch_too_small";
            }
        }
        selected->clear();
        return false;
    }
    return true;
}

bool LLMEngine::run_decode_post_emit_conservative(
    RequestState& request,
    SequenceState& seq,
    int token_id
) {
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
        return false;
    }
    if (next < 0) {
        fail_request(request, "decode forward failed");
        return false;
    }
    request.next_token = next;
    if (request.num_generated_tokens >= request.sampling.max_new_tokens) {
        finish_request(request);
    }
    return true;
}

bool LLMEngine::run_selective_decode_batch_step() {
    SelectiveDecodeScratch& scratch = *selective_decode_scratch_;
    scratch.selected.clear();
    scratch.model_ids.clear();
    scratch.items.clear();
    scratch.outputs.clear();
    std::vector<RequestId>& selected = scratch.selected;
    if (!build_selective_decode_batch(&selected)) {
        return false;
    }

    std::vector<RequestId>& model_ids = scratch.model_ids;
    std::vector<QwenModel::SelectiveDecodeItem>& items = scratch.items;

    for (RequestId id : selected) {
        auto it = requests_.find(id);
        if (it == requests_.end()) {
            continue;
        }
        RequestState& request = it->second;
        if (is_terminal(request.status) || request.status != RequestStatus::RUNNING_DECODE) {
            continue;
        }
        SequenceState& seq = get_or_create_session(request.session_id);
        int token_id = request.next_token;

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
            emit_metrics_once(request);
            remove_request_from_all_queues(id);
            continue;
        }

        model_ids.push_back(id);
        items.push_back(QwenModel::SelectiveDecodeItem{&seq, token_id});
    }

    if (items.empty()) {
        return true;
    }

    if (static_cast<int>(items.size()) < selective_decode_min_batch_) {
        for (RequestId id : model_ids) {
            auto it = requests_.find(id);
            if (it == requests_.end() || is_terminal(it->second.status)) {
                continue;
            }
            RequestState& request = it->second;
            SequenceState& seq = get_or_create_session(request.session_id);
            request.metrics.selective_decode_fallbacks++;
            request.metrics.selective_decode_mode = "fallback_after_callback";
            run_decode_post_emit_conservative(request, seq, request.generated_tokens.back());
        }
        return true;
    }

    if (selective_decode_debug_) {
        std::cerr << "[SELECTIVE_DECODE] build size=" << items.size()
                  << " decode=" << decode_queue_.size()
                  << std::endl;
    }

    std::vector<QwenModel::SelectiveDecodeOutput>& outputs = scratch.outputs;
    QwenModel::SelectiveDecodeStats stats;
    bool ok = model_.decode_selective_batch_for_sequences(
        items,
        *kv_manager_,
        prefix_cache_.get(),
        &outputs,
        &stats);

    if (!ok || outputs.size() != items.size()) {
        if (selective_decode_debug_) {
            std::cerr << "[SELECTIVE_DECODE] fallback reason=model_failed"
                      << " size=" << items.size()
                      << std::endl;
        }
        for (RequestId id : model_ids) {
            auto it = requests_.find(id);
            if (it == requests_.end() || is_terminal(it->second.status)) {
                continue;
            }
            RequestState& request = it->second;
            request.metrics.selective_decode_fallbacks++;
            request.metrics.selective_decode_mode = stats.state_modified
                ? "failed_after_kv_write"
                : "fallback_model_preflight";
            if (stats.state_modified) {
                if (stats.state_modified) {
                    SequenceState& seq = get_or_create_session(request.session_id);
                    seq.status = SequenceStatus::FAILED;
                    seq.error_message = "selective decode failed after KV write";
                }
                std::string error = "selective decode failed";
                if (!outputs.empty() && !outputs[0].error_message.empty()) {
                    error = outputs[0].error_message;
                }
                fail_request(request, error);
                continue;
            }
            SequenceState& seq = get_or_create_session(request.session_id);
            run_decode_post_emit_conservative(request, seq, request.generated_tokens.back());
        }
        return true;
    }

    int batch_size = static_cast<int>(items.size());
    double per_request_model_ms =
        batch_size > 0 ? stats.model_ms / static_cast<double>(batch_size) : 0.0;
    int per_request_linear_rows =
        batch_size > 0 ? stats.linear_batch_rows / batch_size : stats.linear_batch_rows;
    int per_request_attention_calls =
        batch_size > 0
            ? stats.attention_per_sequence_calls / batch_size
            : stats.attention_per_sequence_calls;
    int per_request_lm_head_rows =
        batch_size > 0 ? stats.lm_head_rows / batch_size : stats.lm_head_rows;
    auto share_counter = [batch_size](uint64_t value, size_t index) -> uint64_t {
        if (batch_size <= 0) return value;
        uint64_t base = value / (uint64_t)batch_size;
        uint64_t remainder = value % (uint64_t)batch_size;
        return base + (index < remainder ? 1u : 0u);
    };
    for (size_t i = 0; i < model_ids.size(); ++i) {
        RequestId id = model_ids[i];
        auto it = requests_.find(id);
        if (it == requests_.end() || is_terminal(it->second.status)) {
            continue;
        }
        RequestState& request = it->second;
        SequenceState& seq = get_or_create_session(request.session_id);
        request.metrics.scheduler_enabled = true;
        request.metrics.scheduler_steps++;
        request.metrics.scheduler_decode_steps++;
        request.metrics.decode_batch_steps++;
        request.metrics.decode_batch_size_sum += batch_size;
        request.metrics.decode_batch_size_max =
            std::max(request.metrics.decode_batch_size_max, batch_size);
        request.metrics.selective_decode_enabled = true;
        request.metrics.selective_decode_steps++;
        request.metrics.selective_decode_size_sum += batch_size;
        request.metrics.selective_decode_size_max =
            std::max(request.metrics.selective_decode_size_max, batch_size);
        request.metrics.selective_decode_linear_batch_rows += per_request_linear_rows;
        request.metrics.selective_decode_attention_per_sequence_calls +=
            per_request_attention_calls;
        request.metrics.selective_decode_lm_head_rows += per_request_lm_head_rows;
        request.metrics.selective_decode_model_ms += per_request_model_ms;
        request.metrics.gptq_batch_kernel_calls +=
            share_counter(stats.gptq_batch_kernel_calls, i);
        request.metrics.gptq_batch_rows_total +=
            share_counter(stats.gptq_batch_rows_total, i);
        request.metrics.gptq_batch_output_panel_tasks +=
            share_counter(stats.gptq_batch_output_panel_tasks, i);
        request.metrics.gptq_batch_row_gemv_fallbacks +=
            share_counter(stats.gptq_batch_row_gemv_fallbacks, i);
        request.metrics.gptq_batch_weight_vector_loads +=
            share_counter(stats.gptq_batch_weight_vector_loads, i);
        request.metrics.gptq_batch_dequant_vector_ops +=
            share_counter(stats.gptq_batch_dequant_vector_ops, i);
        request.metrics.gptq_batch_argmax_calls +=
            share_counter(stats.gptq_batch_argmax_calls, i);
        request.metrics.gptq_batch_argmax_rows +=
            share_counter(stats.gptq_batch_argmax_rows, i);
        request.metrics.gptq_batch_full_logits_elements_written +=
            share_counter(stats.gptq_batch_full_logits_elements_written, i);
        request.metrics.gptq_batch_compare_mismatches +=
            share_counter(stats.gptq_batch_compare_mismatches, i);
        request.metrics.selective_decode_hotpath_allocations +=
            share_counter(stats.selective_decode_hotpath_allocations, i);
        request.metrics.selective_decode_workspace_reallocations +=
            share_counter(stats.selective_decode_workspace_reallocations, i);
        request.metrics.selective_decode_mode = "selective_batch_decode";
        request.metrics.decode_ms += per_request_model_ms;

        if (!outputs[i].success || outputs[i].next_token < 0) {
            fail_request(
                request,
                outputs[i].error_message.empty()
                    ? "selective decode output failed"
                    : outputs[i].error_message);
            continue;
        }
        if (seq.status == SequenceStatus::FAILED) {
            fail_request(request, seq.error_message.empty() ? "decode failed" : seq.error_message);
            continue;
        }
        request.next_token = outputs[i].next_token;
        request.metrics.greedy_tokens++;
        if (request.num_generated_tokens >= request.sampling.max_new_tokens) {
            finish_request(request);
        }
    }
    return true;
}

bool LLMEngine::run_prefill_step() {
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
                add_to_decode_queue(request);
            }
            continue;
        }

        try {
            SequenceState& seq = get_or_create_session(request.session_id);
            const int begin = request.prompt_cursor;
            const int end = std::min(
                begin + prefill_chunk_size_,
                static_cast<int>(request.prompt_tokens.size()));
            if (begin >= end) {
                if (request.next_token < 0) {
                    fail_request(request, "prefill produced no next token");
                } else {
                    transition_prefill_complete(request);
                }
            } else if (seq.history_pos + (end - begin) > model_.config.max_seq_len) {
                fail_request(request, "sequence length exceeded");
            } else {
                int next = -1;
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
                if (seq.status == SequenceStatus::FAILED) {
                    fail_request(request, seq.error_message.empty() ? "prefill failed" : seq.error_message);
                } else if (next < 0) {
                    fail_request(request, "prefill forward failed");
                } else {
                    request.next_token = next;
                    request.prompt_cursor = end;
                    request.metrics.computed_prefill_tokens += end - begin;
                    request.metrics.scheduler_enabled = true;
                    request.metrics.scheduler_steps++;
                    request.metrics.scheduler_prefill_steps++;
                    request.metrics.prefill_chunk_steps++;
                    if (request.prompt_cursor >= static_cast<int>(request.prompt_tokens.size())) {
                        transition_prefill_complete(request);
                    }
                }
            }
        } catch (const std::exception& e) {
            fail_request(request, e.what());
        } catch (...) {
            fail_request(request, "unknown scheduler prefill exception");
        }

        if (request.status == RequestStatus::RUNNING_PREFILL) {
            add_to_prefill_queue(request);
        } else if (is_terminal(request.status)) {
            remove_request_from_all_queues(request.id);
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

void LLMEngine::add_to_decode_queue(RequestState& request) {
    if (request.queued_decode || is_terminal(request.status) ||
        request.status != RequestStatus::RUNNING_DECODE) {
        return;
    }
    request.queued_decode = true;
    decode_queue_.push_back(request.id);
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] enqueue_decode"
                  << " id=" << request.id
                  << " decode=" << decode_queue_.size()
                  << std::endl;
    }
}

void LLMEngine::remove_request_from_all_queues(RequestId id) {
    waiting_queue_.erase(
        std::remove(waiting_queue_.begin(), waiting_queue_.end(), id),
        waiting_queue_.end());
    prefill_queue_.erase(
        std::remove(prefill_queue_.begin(), prefill_queue_.end(), id),
        prefill_queue_.end());
    decode_queue_.erase(
        std::remove(decode_queue_.begin(), decode_queue_.end(), id),
        decode_queue_.end());
    auto it = requests_.find(id);
    if (it != requests_.end()) {
        it->second.queued_waiting = false;
        it->second.queued_prefill = false;
        it->second.queued_decode = false;
    }
}

void LLMEngine::cleanup_terminal_requests() {
    std::vector<RequestId> terminal_ids;
    for (const auto& kv : requests_) {
        if (is_terminal(kv.second.status)) {
            terminal_ids.push_back(kv.first);
        }
    }
    for (RequestId id : terminal_ids) {
        remove_request_from_all_queues(id);
    }
}

void LLMEngine::transition_prefill_complete(RequestState& request) {
    if (request.prompt_cursor < static_cast<int>(request.prompt_tokens.size())) {
        return;
    }
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
    // Prefill 已经产生首个 next_token，下一次 scheduler tick 可直接 Decode。
    // 当前实现不再区分 decode-ready 与 active，也不设置独立的 Decode 并发上限。
    add_to_decode_queue(request);
    if (debug_scheduler_enabled()) {
        std::cerr << "[SCHED] transition"
                  << " id=" << request.id
                  << " status=" << request_status_name(request.status)
                  << std::endl;
    }
}

void LLMEngine::step_decode(RequestState& request) {
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
        remove_request_from_all_queues(request.id);
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

void LLMEngine::fail_request(RequestState& request, const std::string& error) {
    request.status = RequestStatus::FAILED;
    request.error_message = error;
    request.callback = TokenCallback{};
    request.metrics.decode_queue_size_at_finish =
        std::max(request.metrics.decode_queue_size_at_finish,
                 static_cast<int>(decode_queue_.size()));
    remove_request_from_all_queues(request.id);
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
    request.metrics.decode_queue_size_at_finish =
        std::max(request.metrics.decode_queue_size_at_finish,
                 static_cast<int>(decode_queue_.size()));
    remove_request_from_all_queues(request.id);
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
    request.metrics.scheduler_enabled = scheduler_enabled_;
    if (request.metrics.decode_batch_steps > 0) {
        request.metrics.decode_batch_size_avg =
            static_cast<double>(request.metrics.decode_batch_size_sum) /
            static_cast<double>(request.metrics.decode_batch_steps);
    }
    if (request.metrics.selective_decode_steps > 0) {
        request.metrics.selective_decode_size_avg =
            static_cast<double>(request.metrics.selective_decode_size_sum) /
            static_cast<double>(request.metrics.selective_decode_steps);
    }
    if (scheduler_enabled_ && request.metrics.decode_queue_size_at_finish <= 0) {
        request.metrics.decode_queue_size_at_finish =
            static_cast<int>(decode_queue_.size());
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
