#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llm_engine/engine/engine_service.h"
#include "llm_engine/server/server_metrics.h"

namespace llm_engine {

struct ChatMessage {
    std::string role;
    std::string content;
};

std::string build_qwen_chatml_prompt(const std::vector<ChatMessage>& messages);

class HttpSseServer {
public:
    using EncodeFn = std::function<std::vector<int>(const std::string&)>;
    using DecodeFn = std::function<std::string(int)>;

    HttpSseServer(
        EngineService& service,
        EncodeFn encode_text,
        EncodeFn encode_chatml,
        DecodeFn decode,
        int default_max_new_tokens);

    bool run_forever(const std::string& host, int port);
    void stop();

private:
    struct StreamState;
    enum class ResponseKind { Chat, Completion };

    void handle_client(int fd);
    bool route_request(int fd, const struct HttpRequest& request);
    bool handle_chat_completions(int fd, const struct HttpRequest& request);
    bool handle_completions(int fd, const struct HttpRequest& request);
    bool handle_models(int fd);
    bool handle_health(int fd);
    bool handle_metrics(int fd, const struct HttpRequest& request);
    bool handle_options(int fd);

    bool check_auth(const struct HttpRequest& request, bool metrics_path, int fd);
    bool send_json(int fd, int status, const std::string& body);
    bool send_error(int fd, int status, const std::string& message, const std::string& type);
    bool send_sse_headers(int fd);
    void apply_cors(struct HttpResponse* response) const;

    bool parse_chat_request(
        const std::string& body,
        std::vector<int>* prompt_tokens,
        SamplingParams* sampling,
        bool* stream,
        uint64_t* requested_session_id,
        std::string* error);
    bool parse_completion_request(
        const std::string& body,
        std::vector<int>* prompt_tokens,
        SamplingParams* sampling,
        bool* stream,
        uint64_t* requested_session_id,
        std::string* error);
    bool validate_limits(
        int prompt_tokens,
        SamplingParams* sampling,
        int fd);
    SessionId choose_session_id(uint64_t requested_session_id);
    bool submit_and_stream(
        int fd,
        const std::vector<int>& prompt_tokens,
        SamplingParams sampling,
        ResponseKind kind,
        SessionId session_id);
    bool submit_and_collect(
        int fd,
        const std::vector<int>& prompt_tokens,
        SamplingParams sampling,
        ResponseKind kind,
        SessionId session_id);
    bool wait_request(
        RequestId request_id,
        EngineRequestSnapshot* snapshot,
        bool* timed_out);
    void writer_loop(
        int fd,
        std::shared_ptr<StreamState> state,
        ResponseKind kind,
        const std::string& object_id,
        const std::string& model_name,
        uint64_t created);
    std::string decode_tokens(const std::vector<int>& tokens);
    std::string make_usage_json(int prompt_tokens, int completion_tokens) const;
    bool debug_enabled() const;
    bool debug_chatml_enabled() const;
    bool debug_prefix_enabled() const;
    bool debug_limits_enabled() const;

    EngineService& service_;
    EncodeFn encode_text_;
    EncodeFn encode_chatml_;
    DecodeFn decode_;
    int default_max_new_tokens_ = 512;
    std::string model_name_;
    std::string api_key_;
    bool enable_cors_ = true;
    bool openai_stateless_ = true;
    bool prefix_cache_friendly_ = true;
    std::string session_strategy_;
    SessionId default_session_id_ = 1;
    int max_pending_requests_ = 16;
    int max_active_connections_ = 32;
    int request_timeout_ms_ = 120000;
    int response_queue_limit_ = 2048;
    int max_prompt_tokens_ = 4096;
    int max_new_tokens_ = 512;
    int max_total_tokens_ = 8192;
    size_t max_header_bytes_ = 16 * 1024;
    size_t max_body_bytes_ = 256 * 1024;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> listen_fd_{-1};
    std::atomic<int> pending_requests_{0};
    std::atomic<uint64_t> next_session_id_{1000000};
    std::mutex decode_mu_;
    ServerMetrics metrics_;
    uint64_t start_time_us_ = 0;
};

} // namespace llm_engine
