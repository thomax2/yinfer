#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm_engine/cache/prefix_cache.h"
#include "llm_engine/engine/sequence_state.h"
#include "llm_engine/memory/kv_cache_manager.h"
#include "llm_engine/metrics/metrics.h"

namespace llm_engine {

class QwenModel;

using RequestId = uint64_t;
using TokenCallback = std::function<bool(int token_id)>;

struct SamplingParams {
    int max_new_tokens = 512;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    bool greedy = true;
};

enum class RequestStatus {
    WAITING,
    RUNNING_PREFILL,
    RUNNING_DECODE,
    FINISHED,
    ABORTED,
    FAILED
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
    bool debug_chunked_prefill_enabled() const;
    void apply_prefix_cache(SequenceState& seq, const std::vector<int>& prompt_tokens);
    void schedule_next_request();
    void step_prefill(RequestState& request);
    void step_decode(RequestState& request);
    void run_legacy_request(RequestState& request);
    void fail_request(RequestState& request, const std::string& error);
    void finish_request(RequestState& request);
    void emit_metrics_once(RequestState& request);
    void mark_first_token(RequestState& request);
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
    int prefill_step_tokens_ = 1;
    int prefill_chunk_size_ = 1;
    RequestId active_request_id_ = 0;
    std::unique_ptr<KVCacheManager> kv_manager_;
    std::unique_ptr<PrefixCache> prefix_cache_;
    std::deque<RequestId> waiting_queue_;
    std::unordered_map<RequestId, RequestState> requests_;
    std::unordered_map<SessionId, SequenceState> sessions_;
};

} // namespace llm_engine
