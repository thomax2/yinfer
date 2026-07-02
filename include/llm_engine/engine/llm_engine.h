#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

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

    void abort(RequestId id);
    void clear_history();

    const RequestState* get_request(RequestId id) const;

private:
    bool debug_enabled() const;
    void debug_log_submit(const RequestState& request) const;
    void debug_log_finished(const RequestState& request) const;
    void debug_log_failed(const RequestState& request) const;

    QwenModel& model_;
    RequestId next_request_id_ = 1;
    std::unordered_map<RequestId, RequestState> requests_;
};

} // namespace llm_engine
