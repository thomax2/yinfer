#pragma once

#include <string>
#include <unordered_map>

namespace llm_engine {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

struct HttpReadLimits {
    size_t max_header_bytes = 16 * 1024;
    size_t max_body_bytes = 256 * 1024;
};

std::string http_status_text(int status);
std::string lower_header_key(const std::string& key);
std::string header_get_lower(
    const std::unordered_map<std::string, std::string>& headers,
    const std::string& key,
    const std::string& default_value = "");

bool read_http_request(
    int fd,
    const HttpReadLimits& limits,
    HttpRequest* out,
    int* error_status,
    std::string* error);

bool write_http_response(int fd, const HttpResponse& response);
bool write_raw(int fd, const std::string& data);
bool write_sse_event(int fd, const std::string& data);

} // namespace llm_engine
