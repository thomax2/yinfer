#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm_engine/cache/prefix_cache.h"
#include "llm_engine/engine/sampling.h"
#include "llm_engine/engine/sequence_state.h"
#include "llm_engine/memory/kv_cache_manager.h"
#include "llm_engine/metrics/metrics.h"

namespace llm_engine {

class QwenModel;

using RequestId = uint64_t;
using TokenCallback = std::function<bool(int token_id)>;

enum class RequestStatus {
    WAITING,
    RUNNING_PREFILL,
    RUNNING_DECODE,
    FINISHED,
    ABORTED,
    FAILED
};

struct PrefillChunkItem {
    RequestId request_id = 0;
    SessionId session_id = 0;
    int prompt_begin = 0;
    int prompt_end = 0;
    int token_count = 0;
    int seq_history_pos_begin = 0;
    int seq_history_pos_end = 0;
    bool is_tail = false;
};

struct PrefillMicroBatch {
    std::vector<PrefillChunkItem> items;
    int target_chunk_size = 0;
    int full_chunks = 0;
    int tail_chunks = 0;
    bool contains_tail = false;
};

struct RequestState {
    RequestId id = 0;
    SessionId session_id = 0;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    SamplingParams sampling;
    TokenCallback callback;
    RequestStatus status = RequestStatus::WAITING;
    std::string error_message;
    int token_count = 0;
    int prompt_cursor = 0;
    int next_token = -1;
    int num_generated_tokens = 0;
    bool callback_stopped = false;
    bool prefix_applied = false;
    int cached_prefix_tokens = 0;
    int cached_prefix_blocks = 0;
    RequestMetrics metrics;
    PagedAttentionStats paged_attention_baseline;
    uint64_t enqueue_time_us = 0;
    uint64_t schedule_time_us = 0;
    uint64_t first_token_time_us = 0;
    uint64_t finish_time_us = 0;
    bool first_token_emitted = false;
    bool metrics_emitted = false;
    bool scheduler_v2 = false;
    bool queued_waiting = false;
    bool queued_prefill = false;
    bool queued_decode_ready = false;
    bool active_decode = false;
    int prefill_microbatch_steps = 0;
    int prefill_microbatch_size_sum = 0;
    int prefill_microbatch_size_max = 0;
    int prefill_full_chunk_steps = 0;
    int prefill_tail_chunk_steps = 0;
    int prefill_tail_wait_steps = 0;
    int prefill_requeue_count = 0;
    bool prefill_blocked_for_batch = false;
    std::mt19937_64 rng;
};

class LLMEngine {
public:
    explicit LLMEngine(QwenModel& model);

    RequestId submit(
        const std::vector<int>& prompt_tokens,
        const SamplingParams& sampling,
        TokenCallback callback);
    RequestId submit(
        SessionId session_id,
        const std::vector<int>& prompt_tokens,
        const SamplingParams& sampling,
        TokenCallback callback);
    RequestId submit_async(
        const std::vector<int>& prompt_tokens,
        const SamplingParams& sampling,
        TokenCallback callback);
    RequestId submit_async(
        SessionId session_id,
        const std::vector<int>& prompt_tokens,
        const SamplingParams& sampling,
        TokenCallback callback);

    bool step_once();
    void run_until_idle();
    void run_until_finished(RequestId id);
    bool has_pending_requests() const;
    bool request_finished(RequestId id) const;
    bool scheduler_enabled() const;

    void abort(RequestId id);
    void clear_history();
    void clear_session(SessionId session_id);

    const RequestState* get_request(RequestId id) const;
    const SequenceState* get_session(SessionId session_id) const;

private:
    bool debug_enabled() const;
    bool debug_session_enabled() const;
    bool debug_prefix_enabled() const;
    bool debug_scheduler_enabled() const;
    bool debug_cont_batch_enabled() const;
    bool debug_cont_batch_verbose_enabled() const;
    bool debug_prefill_batch_enabled() const;
    bool debug_prefill_batch_verbose_enabled() const;
    bool debug_chunked_prefill_enabled() const;
    void apply_prefix_cache(SequenceState& seq, const std::vector<int>& prompt_tokens);
    bool step_once_legacy();
    bool step_once_continuous();
    void schedule_next_request();
    void admit_waiting_requests_v2();
    bool run_decode_batch_step();
    bool run_prefill_chunk_step();
    bool run_prefill_microbatch_step();
    PrefillMicroBatch build_prefill_microbatch();
    bool execute_prefill_microbatch_conservative(const PrefillMicroBatch& batch);
    bool execute_prefill_microbatch_true_batch(const PrefillMicroBatch& batch);
    bool make_prefill_chunk_item(RequestState& request, PrefillChunkItem* out);
    void update_prefill_microbatch_metrics(
        RequestState& request,
        const PrefillMicroBatch& batch,
        const PrefillChunkItem& item);
    void requeue_prefill_request(RequestId id);
    bool request_has_prefill_remaining(const RequestState& request) const;
    void transition_prefill_complete(RequestState& request);
    void add_to_prefill_queue(RequestState& request);
    void add_to_decode_ready_queue(RequestState& request);
    void activate_decode_requests();
    void remove_from_active_decode(RequestId id);
    void remove_request_from_all_v2_queues(RequestId id);
    void cleanup_terminal_requests_v2();
    void step_prefill(RequestState& request);
    void step_decode(RequestState& request);
    void run_legacy_request(RequestState& request);
    void fail_request(RequestState& request, const std::string& error);
    void finish_request(RequestState& request);
    void emit_metrics_once(RequestState& request);
    void mark_first_token(RequestState& request);
    void init_request_sampling(RequestState& request);
    bool request_stop_token(const RequestState& request, int token_id) const;
    bool is_terminal(RequestStatus status) const;
    const char* request_status_name(RequestStatus status) const;
    void debug_log_submit(const RequestState& request) const;
    void debug_log_finished(const RequestState& request) const;
    void debug_log_failed(const RequestState& request) const;
    SequenceState& get_or_create_session(SessionId session_id);

    QwenModel& model_;
    RequestId next_request_id_ = 1;
    SessionId default_session_id_ = 1;
    bool session_cache_enabled_ = false;
    bool prefix_cache_enabled_ = false;
    bool scheduler_enabled_ = false;
    bool chunked_prefill_enabled_ = false;
    bool chunked_prefill_strict_ = false;
    bool continuous_batching_enabled_ = false;
    bool cont_batch_decode_first_ = true;
    bool cont_batch_prefill_when_decode_empty_ = true;
    bool cont_batch_prefill_after_decode_ = false;
    bool cont_batch_conservative_executor_ = true;
    bool prefill_batching_enabled_ = false;
    int prefill_microbatch_max_requests_ = 4;
    int prefill_microbatch_chunk_size_ = 8;
    int prefill_microbatch_min_requests_ = 2;
    bool prefill_microbatch_allow_tail_ = true;
    int prefill_microbatch_tail_max_wait_steps_ = 2;
    int prefill_microbatch_scan_limit_ = 32;
    bool prefill_microbatch_after_decode_ = false;
    bool prefill_microbatch_when_decode_empty_ = true;
    std::string prefill_microbatch_executor_ = "conservative";
    bool prefill_microbatch_strict_ = false;
    int max_active_decode_requests_ = 8;
    int cont_batch_max_prefill_chunks_per_step_ = 1;
    int prefill_step_tokens_ = 1;
    int prefill_chunk_size_ = 1;
    RequestId active_request_id_ = 0;
    std::unique_ptr<KVCacheManager> kv_manager_;
    std::unique_ptr<PrefixCache> prefix_cache_;
    std::deque<RequestId> waiting_queue_;
    std::deque<RequestId> prefill_queue_;
    std::deque<RequestId> decode_ready_queue_;
    std::vector<RequestId> active_decode_requests_;
    std::unordered_map<RequestId, RequestState> requests_;
    std::unordered_map<SessionId, SequenceState> sessions_;
};

} // namespace llm_engine
