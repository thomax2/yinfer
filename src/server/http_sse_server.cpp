#include "llm_engine/server/http_sse_server.h"

#include "llm_engine/server/http_types.h"
#include "llm_engine/server/json_lite.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <io.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace llm_engine {

namespace {

bool env_flag(const char* name, bool default_value = false) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

int env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<int>(x);
}

std::string env_string(const char* name, const std::string& default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    return std::string(v);
}

uint64_t now_epoch_sec() {
    return static_cast<uint64_t>(std::time(nullptr));
}

uint64_t steady_ms() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            clock::now().time_since_epoch()).count());
}

const char* status_name(RequestStatus status) {
    switch (status) {
        case RequestStatus::WAITING: return "WAITING";
        case RequestStatus::RUNNING_PREFILL: return "RUNNING_PREFILL";
        case RequestStatus::RUNNING_DECODE: return "RUNNING_DECODE";
        case RequestStatus::FINISHED: return "FINISHED";
        case RequestStatus::ABORTED: return "ABORTED";
        case RequestStatus::FAILED: return "FAILED";
    }
    return "UNKNOWN";
}

bool is_terminal(RequestStatus status) {
    return status == RequestStatus::FINISHED ||
           status == RequestStatus::ABORTED ||
           status == RequestStatus::FAILED;
}

#ifndef _WIN32
void close_fd(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}
#endif

std::string openai_error_json(const std::string& message, const std::string& type) {
    std::ostringstream os;
    os << "{\"error\":{\"message\":\"" << json_escape(message)
       << "\",\"type\":\"" << json_escape(type)
       << "\",\"code\":null}}";
    return os.str();
}

void parse_sampling_common(const JsonValue& root, SamplingParams* sampling, int default_max_new_tokens) {
    sampling->max_new_tokens = default_max_new_tokens;
    sampling->greedy = true;
    int max_tokens = 0;
    if (get_int(root, "max_tokens", &max_tokens) && max_tokens > 0) {
        sampling->max_new_tokens = max_tokens;
    }
    double temperature = 0.0;
    if (get_double(root, "temperature", &temperature)) {
        sampling->temperature = static_cast<float>(temperature);
    }
    int top_k = 0;
    if (get_int(root, "top_k", &top_k)) {
        sampling->top_k = top_k;
    }
    double top_p = 1.0;
    if (get_double(root, "top_p", &top_p)) {
        sampling->top_p = static_cast<float>(top_p);
    }
    uint64_t seed = 0;
    if (get_u64(root, "seed", &seed)) {
        sampling->seed = seed;
        sampling->has_seed = true;
    }
    bool greedy = sampling->greedy;
    if (get_bool(root, "greedy", &greedy)) {
        sampling->greedy = greedy;
    }
    if (sampling->temperature > 0.0f) {
        sampling->greedy = false;
    }
}

} // namespace

struct HttpSseServer::StreamState {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<int> token_ids;
    bool finished = false;
    bool closed = false;
    RequestId request_id = 0;
    std::string finish_reason = "stop";
    std::string error_message;
};

std::string build_qwen_chatml_prompt(const std::vector<ChatMessage>& messages) {
    std::string prompt = "\n";
    for (const ChatMessage& message : messages) {
        prompt += "<|im_start|>";
        prompt += message.role;
        prompt += "\n";
        prompt += message.content;
        prompt += "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

HttpSseServer::HttpSseServer(
    EngineService& service,
    EncodeFn encode_text,
    EncodeFn encode_chatml,
    DecodeFn decode,
    int default_max_new_tokens
) : service_(service),
    encode_text_(std::move(encode_text)),
    encode_chatml_(std::move(encode_chatml)),
    decode_(std::move(decode)),
    default_max_new_tokens_(default_max_new_tokens) {
    model_name_ = env_string("LLM_API_MODEL_NAME", "qwen2.5-1.5b-rk3588");
    api_key_ = env_string("LLM_API_KEY", "");
    enable_cors_ = env_flag("LLM_ENABLE_CORS", true);
    openai_stateless_ = env_flag("LLM_OPENAI_STATELESS", true);
    prefix_cache_friendly_ = env_flag("LLM_HTTP_PREFIX_CACHE_FRIENDLY", true);
    session_strategy_ = env_string("LLM_HTTP_REQUEST_SESSION_STRATEGY", "stateless");
    default_session_id_ = static_cast<SessionId>(std::max(1, env_int("LLM_HTTP_DEFAULT_SESSION_ID", 1)));
    max_pending_requests_ = std::max(1, env_int("LLM_SERVER_MAX_PENDING_REQUESTS", 16));
    max_active_connections_ = std::max(1, env_int("LLM_SERVER_MAX_ACTIVE_CONNECTIONS", 32));
    request_timeout_ms_ = std::max(1000, env_int("LLM_SERVER_REQUEST_TIMEOUT_MS", 120000));
    response_queue_limit_ = std::max(1, env_int("LLM_SERVER_RESPONSE_QUEUE_LIMIT", 2048));
    max_prompt_tokens_ = std::max(1, env_int("LLM_MAX_PROMPT_TOKENS", 4096));
    max_new_tokens_ = std::max(1, env_int("LLM_MAX_NEW_TOKENS", default_max_new_tokens_));
    max_total_tokens_ = std::max(1, env_int("LLM_MAX_TOTAL_TOKENS", 8192));
    max_header_bytes_ = static_cast<size_t>(std::max(1024, env_int("LLM_HTTP_MAX_HEADER_BYTES", 16384)));
    max_body_bytes_ = static_cast<size_t>(std::max(1024, env_int("LLM_HTTP_MAX_BODY_BYTES", 262144)));
    start_time_us_ = now_us();
}

bool HttpSseServer::run_forever(const std::string& host, int port) {
#ifdef _WIN32
    (void)host;
    (void)port;
    std::cerr << "[HTTP] HTTP server is only implemented for POSIX sockets"
              << std::endl;
    return false;
#else
    signal(SIGPIPE, SIG_IGN);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::cerr << "[HTTP] socket failed: " << std::strerror(errno) << std::endl;
        return false;
    }
    listen_fd_.store(fd);
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (host.empty() || host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "[HTTP] invalid host=" << host << std::endl;
        close_fd(fd);
        listen_fd_.store(-1);
        return false;
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "[HTTP] bind failed: " << std::strerror(errno) << std::endl;
        close_fd(fd);
        listen_fd_.store(-1);
        return false;
    }
    if (listen(fd, env_int("LLM_HTTP_BACKLOG", 16)) != 0) {
        std::cerr << "[HTTP] listen failed: " << std::strerror(errno) << std::endl;
        close_fd(fd);
        listen_fd_.store(-1);
        return false;
    }

    std::cerr << "[HTTP] listening host=" << host << " port=" << port << std::endl;
    while (!stop_requested_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (stop_requested_.load()) {
                break;
            }
            std::cerr << "[HTTP] accept failed: " << std::strerror(errno) << std::endl;
            continue;
        }

        if (metrics_.active_connections() >= max_active_connections_) {
            HttpResponse response;
            response.status = 503;
            response.body = openai_error_json("Too many active connections", "server_error");
            apply_cors(&response);
            write_http_response(client_fd, response);
            close_fd(client_fd);
            continue;
        }
        metrics_.connection_opened();
        std::thread(&HttpSseServer::handle_client, this, client_fd).detach();
    }
    close_fd(fd);
    listen_fd_.store(-1);
    return true;
#endif
}

void HttpSseServer::stop() {
    stop_requested_.store(true);
#ifndef _WIN32
    int fd = listen_fd_.exchange(-1);
    if (fd >= 0) {
        shutdown(fd, SHUT_RDWR);
        close_fd(fd);
    }
#endif
}

void HttpSseServer::handle_client(int fd) {
#ifdef _WIN32
    (void)fd;
#else
    int timeout_ms = std::max(1, env_int("LLM_SERVER_WRITE_TIMEOUT_MS", 5000));
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    HttpRequest request;
    HttpReadLimits limits{max_header_bytes_, max_body_bytes_};
    int error_status = 400;
    std::string error;
    if (!read_http_request(fd, limits, &request, &error_status, &error)) {
        send_error(fd, error_status, error.empty() ? "Bad request" : error, "invalid_request_error");
        close_fd(fd);
        metrics_.connection_closed();
        return;
    }

    if (debug_enabled()) {
        std::cerr << "[HTTP] request method=" << request.method
                  << " path=" << request.path
                  << " body_bytes=" << request.body.size()
                  << std::endl;
    }
    route_request(fd, request);
    close_fd(fd);
    metrics_.connection_closed();
#endif
}

bool HttpSseServer::route_request(int fd, const HttpRequest& request) {
    if (request.method == "OPTIONS") {
        return handle_options(fd);
    }
    if (request.path == "/health") {
        return handle_health(fd);
    }
    if (request.path == "/metrics") {
        return handle_metrics(fd, request);
    }
    if (request.path == "/v1/models") {
        if (!check_auth(request, false, fd)) return false;
        if (request.method != "GET") return send_error(fd, 405, "Method not allowed", "invalid_request_error");
        return handle_models(fd);
    }
    if (request.path == "/v1/chat/completions") {
        if (!check_auth(request, false, fd)) return false;
        if (request.method != "POST") return send_error(fd, 405, "Method not allowed", "invalid_request_error");
        return handle_chat_completions(fd, request);
    }
    if (request.path == "/v1/completions") {
        if (!check_auth(request, false, fd)) return false;
        if (request.method != "POST") return send_error(fd, 405, "Method not allowed", "invalid_request_error");
        return handle_completions(fd, request);
    }
    return send_error(fd, 404, "Not found", "invalid_request_error");
}

bool HttpSseServer::check_auth(const HttpRequest& request, bool metrics_path, int fd) {
    if (api_key_.empty()) {
        return true;
    }
    if (metrics_path && !env_flag("LLM_METRICS_REQUIRE_AUTH", true)) {
        return true;
    }
    std::string auth = header_get_lower(request.headers, "authorization");
    std::string expected = std::string("Bearer ") + api_key_;
    if (auth == expected) {
        return true;
    }
    std::cerr << "[AUTH] unauthorized path=" << request.path << std::endl;
    return send_error(fd, 401, "Unauthorized", "authentication_error");
}

bool HttpSseServer::handle_options(int fd) {
    HttpResponse response;
    response.status = 204;
    response.body.clear();
    apply_cors(&response);
    return write_http_response(fd, response);
}

bool HttpSseServer::handle_models(int fd) {
    std::ostringstream os;
    os << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(model_name_)
       << "\",\"object\":\"model\",\"owned_by\":\"local\"}]}";
    return send_json(fd, 200, os.str());
}

bool HttpSseServer::handle_health(int fd) {
    uint64_t uptime = (now_us() - start_time_us_) / 1000000ULL;
    std::ostringstream os;
    os << "{\"ok\":" << (service_.is_running() ? "true" : "false")
       << ",\"model_loaded\":true"
       << ",\"service_running\":" << (service_.is_running() ? "true" : "false")
       << ",\"scheduler_enabled\":true"
       << ",\"paged_kv\":" << (env_flag("LLM_PAGED_KV") ? "true" : "false")
       << ",\"session_cache\":" << (env_flag("LLM_ENABLE_SESSION_CACHE") ? "true" : "false")
       << ",\"prefix_cache\":" << (env_flag("LLM_ENABLE_PREFIX_CACHE") ? "true" : "false")
       << ",\"http_server\":true"
       << ",\"uptime_sec\":" << uptime
       << ",\"active_requests\":" << metrics_.active_requests()
       << ",\"pending_requests\":" << pending_requests_.load()
       << "}";
    return send_json(fd, service_.is_running() ? 200 : 503, os.str());
}

bool HttpSseServer::handle_metrics(int fd, const HttpRequest& request) {
    if (!check_auth(request, true, fd)) {
        return false;
    }
    if (request.method != "GET") {
        return send_error(fd, 405, "Method not allowed", "invalid_request_error");
    }
    metrics_.set_pending_requests(pending_requests_.load());
    return send_json(fd, 200, metrics_.to_json());
}

bool HttpSseServer::handle_chat_completions(int fd, const HttpRequest& request) {
    std::vector<int> prompt_tokens;
    SamplingParams sampling;
    bool stream = true;
    uint64_t requested_session_id = 0;
    std::string error;
    if (!parse_chat_request(
            request.body, &prompt_tokens, &sampling, &stream, &requested_session_id, &error)) {
        return send_error(fd, 400, error, "invalid_request_error");
    }
    if (!validate_limits(static_cast<int>(prompt_tokens.size()), &sampling, fd)) {
        return false;
    }
    SessionId session_id = choose_session_id(requested_session_id);
    if (stream) {
        return submit_and_stream(fd, prompt_tokens, sampling, ResponseKind::Chat, session_id);
    }
    return submit_and_collect(fd, prompt_tokens, sampling, ResponseKind::Chat, session_id);
}

bool HttpSseServer::handle_completions(int fd, const HttpRequest& request) {
    std::vector<int> prompt_tokens;
    SamplingParams sampling;
    bool stream = true;
    uint64_t requested_session_id = 0;
    std::string error;
    if (!parse_completion_request(
            request.body, &prompt_tokens, &sampling, &stream, &requested_session_id, &error)) {
        return send_error(fd, 400, error, "invalid_request_error");
    }
    if (!validate_limits(static_cast<int>(prompt_tokens.size()), &sampling, fd)) {
        return false;
    }
    SessionId session_id = choose_session_id(requested_session_id);
    if (stream) {
        return submit_and_stream(fd, prompt_tokens, sampling, ResponseKind::Completion, session_id);
    }
    return submit_and_collect(fd, prompt_tokens, sampling, ResponseKind::Completion, session_id);
}

bool HttpSseServer::parse_chat_request(
    const std::string& body,
    std::vector<int>* prompt_tokens,
    SamplingParams* sampling,
    bool* stream,
    uint64_t* requested_session_id,
    std::string* error
) {
    JsonValue root;
    if (!parse_json(body, &root, error) || !root.is_object()) {
        if (error && error->empty()) *error = "invalid JSON object";
        return false;
    }
    const JsonValue* messages = get_field(root, "messages");
    if (!messages || !messages->is_array() || messages->as_array().empty()) {
        if (error) *error = "missing messages";
        return false;
    }

    std::vector<ChatMessage> parsed;
    parsed.reserve(messages->as_array().size());
    for (const JsonValue& item : messages->as_array()) {
        if (!item.is_object()) {
            if (error) *error = "message must be object";
            return false;
        }
        ChatMessage message;
        if (!get_string(item, "role", &message.role) ||
            !get_string(item, "content", &message.content)) {
            if (error) *error = "message role/content must be string";
            return false;
        }
        if (message.role != "system" &&
            message.role != "user" &&
            message.role != "assistant") {
            if (error) *error = "unsupported message role";
            return false;
        }
        parsed.push_back(std::move(message));
    }
    *stream = true;
    get_bool(root, "stream", stream);
    parse_sampling_common(root, sampling, default_max_new_tokens_);
    get_u64(root, "session_id", requested_session_id);

    std::string prompt = build_qwen_chatml_prompt(parsed);
    *prompt_tokens = encode_chatml_(prompt);
    if (debug_chatml_enabled()) {
        std::cerr << "[HTTP_CHATML] messages=" << parsed.size()
                  << " chars=" << prompt.size()
                  << " prompt_tokens=" << prompt_tokens->size()
                  << std::endl;
    }
    return true;
}

bool HttpSseServer::parse_completion_request(
    const std::string& body,
    std::vector<int>* prompt_tokens,
    SamplingParams* sampling,
    bool* stream,
    uint64_t* requested_session_id,
    std::string* error
) {
    JsonValue root;
    if (!parse_json(body, &root, error) || !root.is_object()) {
        if (error && error->empty()) *error = "invalid JSON object";
        return false;
    }
    std::string prompt;
    if (!get_string(root, "prompt", &prompt)) {
        if (error) *error = "missing prompt";
        return false;
    }
    *stream = true;
    get_bool(root, "stream", stream);
    parse_sampling_common(root, sampling, default_max_new_tokens_);
    get_u64(root, "session_id", requested_session_id);
    *prompt_tokens = encode_text_(prompt);
    return true;
}

bool HttpSseServer::validate_limits(int prompt_tokens, SamplingParams* sampling, int fd) {
    if (prompt_tokens > max_prompt_tokens_) {
        if (debug_limits_enabled()) {
            std::cerr << "[LIMIT] prompt too long prompt_tokens=" << prompt_tokens
                      << " max=" << max_prompt_tokens_ << std::endl;
        }
        return send_error(fd, 413, "prompt too long", "invalid_request_error");
    }
    if (sampling->max_new_tokens <= 0) {
        sampling->max_new_tokens = default_max_new_tokens_;
    }
    if (sampling->max_new_tokens > max_new_tokens_) {
        if (debug_limits_enabled()) {
            std::cerr << "[LIMIT] clamp max_tokens requested=" << sampling->max_new_tokens
                      << " max=" << max_new_tokens_ << std::endl;
        }
        sampling->max_new_tokens = max_new_tokens_;
    }
    if (prompt_tokens + sampling->max_new_tokens > max_total_tokens_) {
        return send_error(fd, 400, "prompt_tokens + max_tokens exceeds limit", "invalid_request_error");
    }
    int current = pending_requests_.load();
    if (current >= max_pending_requests_) {
        if (debug_limits_enabled()) {
            std::cerr << "[LIMIT] pending exceeded current=" << current
                      << " max=" << max_pending_requests_ << std::endl;
        }
        return send_error(fd, 429, "too many pending requests", "rate_limit_error");
    }
    return true;
}

SessionId HttpSseServer::choose_session_id(uint64_t requested_session_id) {
    if (session_strategy_ == "explicit_session" && requested_session_id > 0) {
        return static_cast<SessionId>(requested_session_id);
    }
    if (openai_stateless_ || prefix_cache_friendly_ || session_strategy_ == "stateless") {
        return static_cast<SessionId>(next_session_id_.fetch_add(1));
    }
    return default_session_id_;
}

bool HttpSseServer::submit_and_stream(
    int fd,
    const std::vector<int>& prompt_tokens,
    SamplingParams sampling,
    ResponseKind kind,
    SessionId session_id
) {
    if (!send_sse_headers(fd)) {
        return false;
    }

    auto state = std::make_shared<StreamState>();
    pending_requests_.fetch_add(1);
    metrics_.request_started(static_cast<int>(prompt_tokens.size()));
    metrics_.set_pending_requests(pending_requests_.load());
    if (openai_stateless_ || prefix_cache_friendly_) {
        service_.clear_session(session_id);
    }

    RequestId request_id = 0;
    try {
        request_id = service_.submit(
            session_id,
            prompt_tokens,
            sampling,
            [state, this](int token_id) {
                std::lock_guard<std::mutex> lk(state->mu);
                if (state->closed) {
                    return false;
                }
                if (state->token_ids.size() >= static_cast<size_t>(response_queue_limit_)) {
                    state->closed = true;
                    return false;
                }
                state->token_ids.push_back(token_id);
                state->cv.notify_one();
                return true;
            });
    } catch (const std::exception& e) {
        pending_requests_.fetch_sub(1);
        metrics_.request_failed();
        metrics_.set_pending_requests(pending_requests_.load());
        return write_sse_event(fd, openai_error_json(e.what(), "server_error")) &&
               write_sse_event(fd, "[DONE]");
    }

    const std::string object_id =
        (kind == ResponseKind::Chat ? "chatcmpl-" : "cmpl-") + std::to_string(request_id);
    {
        std::lock_guard<std::mutex> lk(state->mu);
        state->request_id = request_id;
    }
    uint64_t created = now_epoch_sec();
    std::thread writer(
        &HttpSseServer::writer_loop,
        this,
        fd,
        state,
        kind,
        object_id,
        model_name_,
        created);

    EngineRequestSnapshot snapshot;
    bool timed_out = false;
    bool ok = wait_request(request_id, &snapshot, &timed_out);
    if (timed_out) {
        service_.abort(request_id);
        state->finish_reason = "timeout";
        state->error_message = "request timeout";
    } else if (!ok || snapshot.status == RequestStatus::FAILED) {
        state->finish_reason = "error";
        state->error_message = snapshot.error_message.empty() ? "request failed" : snapshot.error_message;
    } else if (snapshot.status == RequestStatus::ABORTED) {
        state->finish_reason = "abort";
        state->error_message = snapshot.error_message;
    }

    {
        std::lock_guard<std::mutex> lk(state->mu);
        state->finished = true;
    }
    state->cv.notify_one();
    if (writer.joinable()) {
        writer.join();
    }

    pending_requests_.fetch_sub(1);
    metrics_.set_pending_requests(pending_requests_.load());
    if (snapshot.exists) {
        if (snapshot.status == RequestStatus::FINISHED) {
            metrics_.request_finished(snapshot.metrics);
        } else if (snapshot.status == RequestStatus::ABORTED) {
            metrics_.request_aborted();
        } else {
            metrics_.request_failed();
        }
    } else {
        metrics_.request_failed();
    }
    if (openai_stateless_ || prefix_cache_friendly_) {
        service_.clear_session(session_id);
    }
    if (debug_prefix_enabled()) {
        std::cerr << "[HTTP_PREFIX] mode=" << session_strategy_
                  << " request_session=" << session_id
                  << " cached_prefix_tokens=" << snapshot.metrics.cached_prefix_tokens
                  << " cached_prefix_blocks=" << snapshot.metrics.cached_prefix_blocks
                  << std::endl;
    }
    return !state->closed;
}

bool HttpSseServer::submit_and_collect(
    int fd,
    const std::vector<int>& prompt_tokens,
    SamplingParams sampling,
    ResponseKind kind,
    SessionId session_id
) {
    std::mutex token_mu;
    std::vector<int> tokens;

    pending_requests_.fetch_add(1);
    metrics_.request_started(static_cast<int>(prompt_tokens.size()));
    metrics_.set_pending_requests(pending_requests_.load());
    if (openai_stateless_ || prefix_cache_friendly_) {
        service_.clear_session(session_id);
    }

    RequestId request_id = 0;
    try {
        request_id = service_.submit(
            session_id,
            prompt_tokens,
            sampling,
            [&token_mu, &tokens](int token_id) {
                std::lock_guard<std::mutex> lk(token_mu);
                tokens.push_back(token_id);
                return true;
            });
    } catch (const std::exception& e) {
        pending_requests_.fetch_sub(1);
        metrics_.request_failed();
        metrics_.set_pending_requests(pending_requests_.load());
        return send_error(fd, 500, e.what(), "server_error");
    }

    EngineRequestSnapshot snapshot;
    bool timed_out = false;
    bool ok = wait_request(request_id, &snapshot, &timed_out);
    if (timed_out) {
        service_.abort(request_id);
        pending_requests_.fetch_sub(1);
        metrics_.request_aborted();
        metrics_.set_pending_requests(pending_requests_.load());
        if (openai_stateless_ || prefix_cache_friendly_) {
            service_.clear_session(session_id);
        }
        return send_error(fd, 504, "request timeout", "server_error");
    }

    pending_requests_.fetch_sub(1);
    metrics_.set_pending_requests(pending_requests_.load());
    if (snapshot.exists && snapshot.status == RequestStatus::FINISHED) {
        metrics_.request_finished(snapshot.metrics);
    } else if (snapshot.exists && snapshot.status == RequestStatus::ABORTED) {
        metrics_.request_aborted();
    } else {
        metrics_.request_failed();
    }
    if (openai_stateless_ || prefix_cache_friendly_) {
        service_.clear_session(session_id);
    }
    if (!ok || !snapshot.exists || snapshot.status != RequestStatus::FINISHED) {
        return send_error(fd, 500, snapshot.error_message.empty() ? "request failed" : snapshot.error_message, "server_error");
    }

    std::vector<int> token_copy;
    {
        std::lock_guard<std::mutex> lk(token_mu);
        token_copy = tokens;
    }
    std::string text = decode_tokens(token_copy);
    std::string object_id =
        (kind == ResponseKind::Chat ? "chatcmpl-" : "cmpl-") + std::to_string(request_id);
    uint64_t created = now_epoch_sec();
    std::ostringstream os;
    if (kind == ResponseKind::Chat) {
        os << "{\"id\":\"" << object_id
           << "\",\"object\":\"chat.completion\",\"created\":" << created
           << ",\"model\":\"" << json_escape(model_name_)
           << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
           << json_escape(text)
           << "\"},\"finish_reason\":\"stop\"}],\"usage\":"
           << make_usage_json(static_cast<int>(prompt_tokens.size()), snapshot.num_generated_tokens)
           << "}";
    } else {
        os << "{\"id\":\"" << object_id
           << "\",\"object\":\"text_completion\",\"created\":" << created
           << ",\"model\":\"" << json_escape(model_name_)
           << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(text)
           << "\",\"finish_reason\":\"stop\"}],\"usage\":"
           << make_usage_json(static_cast<int>(prompt_tokens.size()), snapshot.num_generated_tokens)
           << "}";
    }
    return send_json(fd, 200, os.str());
}

bool HttpSseServer::wait_request(
    RequestId request_id,
    EngineRequestSnapshot* snapshot,
    bool* timed_out
) {
    uint64_t begin = steady_ms();
    if (timed_out) *timed_out = false;
    while (true) {
        EngineRequestSnapshot current;
        if (service_.request_snapshot(request_id, &current) && is_terminal(current.status)) {
            if (snapshot) *snapshot = current;
            return true;
        }
        if (steady_ms() - begin > static_cast<uint64_t>(request_timeout_ms_)) {
            if (timed_out) *timed_out = true;
            if (snapshot) *snapshot = current;
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void HttpSseServer::writer_loop(
    int fd,
    std::shared_ptr<StreamState> state,
    ResponseKind kind,
    const std::string& object_id,
    const std::string& model_name,
    uint64_t created
) {
    while (true) {
        int token_id = -1;
        bool done = false;
        std::string finish_reason;
        std::string error_message;
        {
            std::unique_lock<std::mutex> lk(state->mu);
            state->cv.wait(lk, [&] {
                return state->closed || state->finished || !state->token_ids.empty();
            });
            if (state->closed) {
                break;
            }
            if (!state->token_ids.empty()) {
                token_id = state->token_ids.front();
                state->token_ids.pop_front();
            } else if (state->finished) {
                done = true;
                finish_reason = state->finish_reason;
                error_message = state->error_message;
            }
        }

        if (token_id >= 0) {
            std::string piece;
            {
                std::lock_guard<std::mutex> lk(decode_mu_);
                piece = decode_(token_id);
            }
            std::ostringstream os;
            if (kind == ResponseKind::Chat) {
                os << "{\"id\":\"" << object_id
                   << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
                   << ",\"model\":\"" << json_escape(model_name)
                   << "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
                   << json_escape(piece)
                   << "\"},\"finish_reason\":null}]}";
            } else {
                os << "{\"id\":\"" << object_id
                   << "\",\"object\":\"text_completion.chunk\",\"created\":" << created
                   << ",\"model\":\"" << json_escape(model_name)
                   << "\",\"choices\":[{\"index\":0,\"text\":\""
                   << json_escape(piece)
                   << "\",\"finish_reason\":null}]}";
            }
            if (!write_sse_event(fd, os.str())) {
                RequestId request_to_abort = 0;
                std::lock_guard<std::mutex> lk(state->mu);
                state->closed = true;
                request_to_abort = state->request_id;
                if (request_to_abort != 0) {
                    service_.abort(request_to_abort);
                }
                break;
            }
        }
        if (done) {
            if (!error_message.empty()) {
                write_sse_event(fd, openai_error_json(error_message, "server_error"));
            }
            std::ostringstream os;
            if (kind == ResponseKind::Chat) {
                os << "{\"id\":\"" << object_id
                   << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
                   << ",\"model\":\"" << json_escape(model_name)
                   << "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\""
                   << json_escape(finish_reason) << "\"}]}";
                write_sse_event(fd, os.str());
            }
            write_sse_event(fd, "[DONE]");
            break;
        }
    }
}

std::string HttpSseServer::decode_tokens(const std::vector<int>& tokens) {
    std::string out;
    std::lock_guard<std::mutex> lk(decode_mu_);
    for (int token : tokens) {
        out += decode_(token);
    }
    return out;
}

std::string HttpSseServer::make_usage_json(int prompt_tokens, int completion_tokens) const {
    std::ostringstream os;
    os << "{\"prompt_tokens\":" << prompt_tokens
       << ",\"completion_tokens\":" << completion_tokens
       << ",\"total_tokens\":" << (prompt_tokens + completion_tokens)
       << "}";
    return os.str();
}

bool HttpSseServer::send_json(int fd, int status, const std::string& body) {
    HttpResponse response;
    response.status = status;
    response.content_type = "application/json";
    response.body = body;
    apply_cors(&response);
    return write_http_response(fd, response);
}

bool HttpSseServer::send_error(
    int fd,
    int status,
    const std::string& message,
    const std::string& type
) {
    std::cerr << "[HTTP] error status=" << status
              << " message=" << message << std::endl;
    return send_json(fd, status, openai_error_json(message, type));
}

bool HttpSseServer::send_sse_headers(int fd) {
    std::ostringstream os;
    os << "HTTP/1.1 200 OK\r\n"
       << "Content-Type: text/event-stream\r\n"
       << "Cache-Control: no-cache\r\n"
       << "Connection: close\r\n";
    if (enable_cors_) {
        os << "Access-Control-Allow-Origin: *\r\n"
           << "Access-Control-Allow-Headers: authorization, content-type\r\n"
           << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    }
    os << "\r\n";
    return write_raw(fd, os.str());
}

void HttpSseServer::apply_cors(HttpResponse* response) const {
    if (!response || !enable_cors_) {
        return;
    }
    response->headers["Access-Control-Allow-Origin"] = "*";
    response->headers["Access-Control-Allow-Headers"] = "authorization, content-type";
    response->headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
}

bool HttpSseServer::debug_enabled() const {
    return env_flag("LLM_DEBUG_HTTP_SERVER");
}

bool HttpSseServer::debug_chatml_enabled() const {
    return env_flag("LLM_DEBUG_HTTP_CHATML");
}

bool HttpSseServer::debug_prefix_enabled() const {
    return env_flag("LLM_DEBUG_HTTP_PREFIX");
}

bool HttpSseServer::debug_limits_enabled() const {
    return env_flag("LLM_DEBUG_SERVER_LIMITS");
}

} // namespace llm_engine
