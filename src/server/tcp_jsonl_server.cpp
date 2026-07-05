#include "llm_engine/server/tcp_jsonl_server.h"

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
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

constexpr size_t kMaxLineBytes = 64 * 1024;

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
    if (end == v) return default_value;
    return static_cast<int>(x);
}

std::string json_escape(const std::string& s) {
    std::ostringstream os;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    os << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
                } else {
                    os << c;
                }
                break;
        }
    }
    return os.str();
}

bool skip_ws(const std::string& s, size_t* pos) {
    while (*pos < s.size() &&
           (s[*pos] == ' ' || s[*pos] == '\t' || s[*pos] == '\r' || s[*pos] == '\n')) {
        ++(*pos);
    }
    return *pos < s.size();
}

bool extract_string_field(const std::string& json, const char* key, std::string* out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    if (!skip_ws(json, &pos) || json[pos] != '"') return false;
    ++pos;

    std::string value;
    while (pos < json.size()) {
        char c = json[pos++];
        if (c == '"') {
            *out = value;
            return true;
        }
        if (c == '\\' && pos < json.size()) {
            char e = json[pos++];
            switch (e) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: value.push_back(e); break;
            }
        } else {
            value.push_back(c);
        }
    }
    return false;
}

bool extract_int_field(const std::string& json, const char* key, int64_t* out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    if (!skip_ws(json, &pos)) return false;

    char* end = nullptr;
    long long value = std::strtoll(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;
    *out = static_cast<int64_t>(value);
    return true;
}

bool extract_u64_field(const std::string& json, const char* key, uint64_t* out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    if (!skip_ws(json, &pos)) return false;

    char* end = nullptr;
    unsigned long long value = std::strtoull(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;
    *out = static_cast<uint64_t>(value);
    return true;
}

bool extract_float_field(const std::string& json, const char* key, float* out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    if (!skip_ws(json, &pos)) return false;

    char* end = nullptr;
    float value = std::strtof(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return false;
    *out = value;
    return true;
}

bool extract_bool_field(const std::string& json, const char* key, bool* out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    if (!skip_ws(json, &pos)) return false;
    if (json.compare(pos, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
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

#ifndef _WIN32
void close_fd(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}
#endif

} // namespace

struct TcpJsonlServer::ServerResponse {
    enum class Type { RAW, TOKEN, FINISH, ERROR };
    Type type = Type::RAW;
    std::string raw_json;
    RequestId request_id = 0;
    int token_id = -1;
    std::string error;
    RequestStatus status = RequestStatus::FAILED;
    int generated_tokens = 0;
};

struct TcpJsonlServer::ClientConnection {
    int fd = -1;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<ServerResponse> responses;
    std::atomic<bool> closed{false};
    std::atomic<RequestId> active_request_id{0};
    std::thread writer_thread;
};

TcpJsonlServer::TcpJsonlServer(
    EngineService& service,
    EncodeFn encode,
    DecodeFn decode,
    int default_max_new_tokens
) : service_(service),
    encode_(std::move(encode)),
    decode_(std::move(decode)),
    default_max_new_tokens_(default_max_new_tokens) {
    response_queue_limit_ = std::max(1, env_int("LLM_SERVER_RESPONSE_QUEUE_LIMIT", 1024));
}

bool TcpJsonlServer::run_forever(const std::string& host, int port) {
#ifdef _WIN32
    (void)host;
    (void)port;
    std::cerr << "[SERVER] TCP/JSONL server is only implemented for POSIX sockets"
              << std::endl;
    return false;
#else
    signal(SIGPIPE, SIG_IGN);

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::cerr << "[SERVER] socket failed: " << std::strerror(errno) << std::endl;
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
        std::cerr << "[SERVER] invalid host=" << host << std::endl;
        close_fd(fd);
        listen_fd_.store(-1);
        return false;
    }

    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "[SERVER] bind failed: " << std::strerror(errno) << std::endl;
        close_fd(fd);
        listen_fd_.store(-1);
        return false;
    }
    if (listen(fd, env_int("LLM_SERVER_BACKLOG", 16)) != 0) {
        std::cerr << "[SERVER] listen failed: " << std::strerror(errno) << std::endl;
        close_fd(fd);
        listen_fd_.store(-1);
        return false;
    }

    std::cerr << "[SERVER] listening host=" << host << " port=" << port << std::endl;
    while (!stop_requested_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (stop_requested_.load()) {
                break;
            }
            std::cerr << "[SERVER] accept failed: " << std::strerror(errno) << std::endl;
            continue;
        }
        std::thread(&TcpJsonlServer::handle_client, this, client_fd).detach();
    }

    close_fd(fd);
    listen_fd_.store(-1);
    return true;
#endif
}

void TcpJsonlServer::stop() {
    stop_requested_.store(true);
#ifndef _WIN32
    int fd = listen_fd_.exchange(-1);
    if (fd >= 0) {
        shutdown(fd, SHUT_RDWR);
        close_fd(fd);
    }
#endif
}

void TcpJsonlServer::handle_client(int fd) {
#ifdef _WIN32
    (void)fd;
#else
    auto client = std::make_shared<ClientConnection>();
    client->fd = fd;

    int timeout_ms = std::max(1, env_int("LLM_SERVER_WRITE_TIMEOUT_MS", 5000));
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    client->writer_thread = std::thread(&TcpJsonlServer::writer_loop, this, client);
    if (debug_enabled()) {
        std::cerr << "[SERVER] client connected fd=" << fd << std::endl;
    }

    std::string buffer;
    char chunk[4096];
    while (!stop_requested_.load() && !client->closed.load()) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            break;
        }
        buffer.append(chunk, chunk + n);
        if (buffer.size() > kMaxLineBytes) {
            enqueue_response(
                client,
                ServerResponse{ServerResponse::Type::ERROR, "", 0, -1, "line too large"},
                false);
            break;
        }

        size_t newline = std::string::npos;
        while ((newline = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, newline);
            buffer.erase(0, newline + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty() && !handle_line(client, line)) {
                client->closed.store(true);
                break;
            }
        }
    }

    client->closed.store(true);
    RequestId active = client->active_request_id.exchange(0);
    if (active != 0) {
        service_.abort(active);
    }
    shutdown(fd, SHUT_RDWR);
    client->cv.notify_all();
    if (client->writer_thread.joinable()) {
        client->writer_thread.join();
    }
    if (debug_enabled()) {
        std::cerr << "[SERVER] client disconnected fd=" << fd << std::endl;
    }
    close_fd(fd);
#endif
}

bool TcpJsonlServer::handle_line(
    const std::shared_ptr<ClientConnection>& client,
    const std::string& line
) {
    std::string command;
    extract_string_field(line, "command", &command);

    int64_t session_value = 1;
    extract_int_field(line, "session_id", &session_value);
    SessionId session_id = static_cast<SessionId>(std::max<int64_t>(1, session_value));

    try {
        if (command == "clear") {
            service_.clear_session(session_id);
            std::ostringstream os;
            os << "{\"ok\":true,\"session_id\":" << session_id << ",\"command\":\"clear\"}";
            return enqueue_response(
                client,
                ServerResponse{ServerResponse::Type::RAW, os.str()},
                true);
        }
        if (command == "abort") {
            int64_t id = 0;
            if (!extract_int_field(line, "request_id", &id) || id <= 0) {
                return enqueue_response(
                    client,
                    ServerResponse{ServerResponse::Type::ERROR, "", 0, -1, "missing request_id"},
                    true);
            }
            service_.abort(static_cast<RequestId>(id));
            std::ostringstream os;
            os << "{\"ok\":true,\"request_id\":" << id << ",\"command\":\"abort\"}";
            return enqueue_response(
                client,
                ServerResponse{ServerResponse::Type::RAW, os.str()},
                true);
        }

        bool stream = true;
        extract_bool_field(line, "stream", &stream);
        if (!stream) {
            return enqueue_response(
                client,
                ServerResponse{ServerResponse::Type::ERROR, "", 0, -1, "stream=false is not implemented"},
                true);
        }

        std::string prompt;
        if (!extract_string_field(line, "prompt", &prompt)) {
            return enqueue_response(
                client,
                ServerResponse{ServerResponse::Type::ERROR, "", 0, -1, "missing prompt"},
                true);
        }
        int64_t max_new_tokens_value = default_max_new_tokens_;
        extract_int_field(line, "max_new_tokens", &max_new_tokens_value);
        int max_new_tokens = static_cast<int>(
            std::max<int64_t>(1, std::min<int64_t>(4096, max_new_tokens_value)));

        SamplingParams sampling;
        sampling.max_new_tokens = max_new_tokens;
        sampling.greedy = true;
        extract_float_field(line, "temperature", &sampling.temperature);
        int64_t top_k = 0;
        if (extract_int_field(line, "top_k", &top_k)) {
            sampling.top_k = static_cast<int>(top_k);
        }
        extract_float_field(line, "top_p", &sampling.top_p);
        bool greedy = sampling.greedy;
        if (extract_bool_field(line, "greedy", &greedy)) {
            sampling.greedy = greedy;
        }
        uint64_t seed = 0;
        if (extract_u64_field(line, "seed", &seed)) {
            sampling.seed = seed;
            sampling.has_seed = true;
        }
        if (sampling.temperature > 0.0f) {
            sampling.greedy = false;
        }

        std::vector<int> prompt_tokens = encode_(prompt);
        RequestId request_id = service_.submit(
            session_id,
            prompt_tokens,
            sampling,
            [this, client](int token_id) {
                if (client->closed.load()) {
                    return false;
                }
                ServerResponse response;
                response.type = ServerResponse::Type::TOKEN;
                response.request_id = client->active_request_id.load();
                response.token_id = token_id;
                if (!enqueue_response(client, std::move(response), true)) {
                    std::cerr << "[SERVER] response queue overflow" << std::endl;
                    return false;
                }
                return true;
            });
        client->active_request_id.store(request_id);

        if (debug_enabled()) {
            std::cerr << "[SERVER] submit"
                      << " request=" << request_id
                      << " session=" << session_id
                      << " prompt_tokens=" << prompt_tokens.size()
                      << std::endl;
        }

        service_.wait_until_finished(request_id);
        EngineRequestSnapshot snapshot;
        service_.request_snapshot(request_id, &snapshot);
        ServerResponse finish;
        finish.type = ServerResponse::Type::FINISH;
        finish.request_id = request_id;
        finish.status = snapshot.status;
        finish.generated_tokens = snapshot.num_generated_tokens;
        finish.error = snapshot.error_message;
        enqueue_response(client, std::move(finish), true);
        client->active_request_id.store(0);
        return !client->closed.load();
    } catch (const std::exception& e) {
        return enqueue_response(
            client,
            ServerResponse{ServerResponse::Type::ERROR, "", 0, -1, e.what()},
            true);
    }
}

void TcpJsonlServer::writer_loop(std::shared_ptr<ClientConnection> client) {
    if (debug_queue_enabled()) {
        std::cerr << "[SERVER_QUEUE] writer start fd=" << client->fd << std::endl;
    }
    while (true) {
        ServerResponse response;
        {
            std::unique_lock<std::mutex> lk(client->mu);
            client->cv.wait(lk, [&] {
                return client->closed.load() || !client->responses.empty();
            });
            if (client->responses.empty()) {
                if (client->closed.load()) {
                    break;
                }
                continue;
            }
            response = std::move(client->responses.front());
            client->responses.pop_front();
        }

        std::ostringstream os;
        if (response.type == ServerResponse::Type::RAW) {
            os << response.raw_json;
        } else if (response.type == ServerResponse::Type::TOKEN) {
            std::string token;
            try {
                std::lock_guard<std::mutex> lk(decode_mu_);
                token = decode_(response.token_id);
            } catch (const std::exception& e) {
                token = std::string("<decode_error:") + e.what() + ">";
            }
            os << "{\"request_id\":" << response.request_id
               << ",\"token_id\":" << response.token_id
               << ",\"token\":\"" << json_escape(token) << "\"}";
        } else if (response.type == ServerResponse::Type::FINISH) {
            os << "{\"request_id\":" << response.request_id
               << ",\"finished\":true"
               << ",\"status\":\"" << status_name(response.status) << "\""
               << ",\"generated_tokens\":" << response.generated_tokens
               << ",\"error_message\":\"" << json_escape(response.error) << "\"}";
        } else {
            os << "{\"error\":\"" << json_escape(response.error) << "\"}";
        }

        if (!write_json_line(client->fd, os.str())) {
            client->closed.store(true);
            RequestId active = client->active_request_id.exchange(0);
            if (active != 0) {
                service_.abort(active);
            }
            break;
        }
    }
    if (debug_queue_enabled()) {
        std::cerr << "[SERVER_QUEUE] writer exit fd=" << client->fd << std::endl;
    }
}

bool TcpJsonlServer::enqueue_response(
    const std::shared_ptr<ClientConnection>& client,
    ServerResponse response,
    bool count_limit
) {
    if (!client || client->closed.load()) {
        return false;
    }
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lk(client->mu);
        if (count_limit &&
            client->responses.size() >= static_cast<size_t>(response_queue_limit_)) {
            client->closed.store(true);
            return false;
        }
        client->responses.push_back(std::move(response));
        size = client->responses.size();
    }
    client->cv.notify_one();
    if (debug_queue_enabled()) {
        std::cerr << "[SERVER_QUEUE] push"
                  << " fd=" << client->fd
                  << " size=" << size
                  << std::endl;
    }
    return true;
}

bool TcpJsonlServer::write_json_line(int fd, const std::string& json) {
#ifdef _WIN32
    (void)fd;
    (void)json;
    return false;
#else
    std::string line = json;
    line.push_back('\n');
    const char* data = line.data();
    size_t remaining = line.size();
    while (remaining > 0) {
        ssize_t n = send(fd, data, remaining, 0);
        if (n <= 0) {
            return false;
        }
        data += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
#endif
}

bool TcpJsonlServer::debug_enabled() const {
    return env_flag("LLM_DEBUG_SERVER");
}

bool TcpJsonlServer::debug_queue_enabled() const {
    return env_flag("LLM_DEBUG_SERVER_QUEUE");
}

} // namespace llm_engine
