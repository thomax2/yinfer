#pragma once

#include <atomic>
#include <functional>
#include <string>
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
    void handle_client(int fd);
    bool handle_line(int fd, const std::string& line);
    bool write_json_line(int fd, const std::string& json);
    bool debug_enabled() const;

    EngineService& service_;
    EncodeFn encode_;
    DecodeFn decode_;
    int default_max_new_tokens_ = 512;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> listen_fd_{-1};
};

} // namespace llm_engine
