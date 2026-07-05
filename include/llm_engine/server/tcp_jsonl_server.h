#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llm_engine/engine/engine_service.h"

namespace llm_engine {

class TcpJsonlServer {
public:
    using EncodeFn = std::function<std::vector<int>(const std::string&)>;
    using DecodeFn = std::function<std::string(int)>;

    TcpJsonlServer(
        EngineService& service,
        EncodeFn encode,
        DecodeFn decode,
        int default_max_new_tokens);

    bool run_forever(const std::string& host, int port);
    void stop();

private:
    struct ClientConnection;
    struct ServerResponse;

    void handle_client(int fd);
    bool handle_line(const std::shared_ptr<ClientConnection>& client, const std::string& line);
    void writer_loop(std::shared_ptr<ClientConnection> client);
    bool enqueue_response(
        const std::shared_ptr<ClientConnection>& client,
        ServerResponse response,
        bool count_limit);
    bool write_json_line(int fd, const std::string& json);
    bool debug_enabled() const;
    bool debug_queue_enabled() const;

    EngineService& service_;
    EncodeFn encode_;
    DecodeFn decode_;
    int default_max_new_tokens_ = 512;
    int response_queue_limit_ = 1024;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> listen_fd_{-1};
    std::mutex decode_mu_;
};

} // namespace llm_engine
