#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "llm_engine/engine/llm_engine.h"
#include "llm_engine/metrics/metrics.h"

namespace llm_engine {

struct EngineRequestSnapshot {
    bool exists = false;
    RequestId request_id = 0;
    SessionId session_id = 0;
    RequestStatus status = RequestStatus::FAILED;
    std::string error_message;
    int token_count = 0;
    int num_generated_tokens = 0;
    RequestMetrics metrics;
};

class EngineService {
public:
    explicit EngineService(LLMEngine& engine);
    ~EngineService();

    EngineService(const EngineService&) = delete;
    EngineService& operator=(const EngineService&) = delete;

    void start();
    void stop();

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
    void clear_session(SessionId session_id);
    void clear_history();

    bool wait_until_finished(RequestId id);
    bool request_snapshot(RequestId id, EngineRequestSnapshot* out) const;
    bool is_running() const;

private:
    struct EngineCommand;

    void engine_loop();
    void handle_command(EngineCommand& command);
    void enqueue_command(std::unique_ptr<EngineCommand> command);
    void enqueue_void_command(std::unique_ptr<EngineCommand> command);
    void refresh_request_status(RequestId id);
    void refresh_all_tracked_statuses();
    bool debug_enabled() const;
    bool is_terminal(RequestStatus status) const;
    const char* request_status_name(RequestStatus status) const;

    LLMEngine& engine_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<EngineCommand>> commands_;

    mutable std::mutex status_mu_;
    std::condition_variable status_cv_;
    std::unordered_map<RequestId, EngineRequestSnapshot> snapshots_;
};

} // namespace llm_engine
