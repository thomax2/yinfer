#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm_engine/cache/prefix_cache.h"
#include "llm_engine/engine/sequence_state.h"
#include "llm_engine/memory/kv_cache_manager.h"

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
    RUNNING,
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

    void abort(RequestId id);
    void clear_history();
    void clear_session(SessionId session_id);

    const RequestState* get_request(RequestId id) const;
    const SequenceState* get_session(SessionId session_id) const;

private:
    bool debug_enabled() const;
    bool debug_session_enabled() const;
    bool debug_prefix_enabled() const;
    void apply_prefix_cache(SequenceState& seq, const std::vector<int>& prompt_tokens);
    void debug_log_submit(const RequestState& request) const;
    void debug_log_finished(const RequestState& request) const;
    void debug_log_failed(const RequestState& request) const;
    SequenceState& get_or_create_session(SessionId session_id);

    QwenModel& model_;
    RequestId next_request_id_ = 1;
    SessionId default_session_id_ = 1;
    bool session_cache_enabled_ = false;
    bool prefix_cache_enabled_ = false;
    std::unique_ptr<KVCacheManager> kv_manager_;
    std::unique_ptr<PrefixCache> prefix_cache_;
    std::unordered_map<RequestId, RequestState> requests_;
    std::unordered_map<SessionId, SequenceState> sessions_;
};

} // namespace llm_engine
