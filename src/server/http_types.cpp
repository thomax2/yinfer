#include "llm_engine/server/http_types.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>

#ifdef _WIN32
#include <io.h>
#else
#include <sys/socket.h>
#endif

namespace llm_engine {

namespace {

bool recv_more(int fd, std::string* buffer) {
#ifdef _WIN32
    (void)fd;
    (void)buffer;
    return false;
#else
    char chunk[4096];
    ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
    if (n <= 0) {
        return false;
    }
    buffer->append(chunk, chunk + n);
    return true;
#endif
}

std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() &&
           (s[begin] == ' ' || s[begin] == '\t' ||
            s[begin] == '\r' || s[begin] == '\n')) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin &&
           (s[end - 1] == ' ' || s[end - 1] == '\t' ||
            s[end - 1] == '\r' || s[end - 1] == '\n')) {
        --end;
    }
    return s.substr(begin, end - begin);
}

bool parse_content_length(const std::string& text, size_t* out) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *out = static_cast<size_t>(value);
    return true;
}

bool parse_request_head(
    const std::string& head,
    HttpRequest* out,
    int* error_status,
    std::string* error
) {
    std::istringstream is(head);
    std::string line;
    if (!std::getline(is, line)) {
        if (error_status) *error_status = 400;
        if (error) *error = "empty request";
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    std::istringstream first(line);
    std::string target;
    std::string version;
    if (!(first >> out->method >> target >> version)) {
        if (error_status) *error_status = 400;
        if (error) *error = "invalid request line";
        return false;
    }
    size_t q = target.find('?');
    if (q == std::string::npos) {
        out->path = target;
    } else {
        out->path = target.substr(0, q);
        out->query = target.substr(q + 1);
    }

    while (std::getline(is, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            break;
        }
        size_t colon = line.find(':');
        if (colon == std::string::npos) {
            if (error_status) *error_status = 400;
            if (error) *error = "invalid header";
            return false;
        }
        std::string key = lower_header_key(trim(line.substr(0, colon)));
        std::string value = trim(line.substr(colon + 1));
        out->headers[key] = value;
    }
    return true;
}

} // namespace

std::string http_status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
        default: return "Error";
    }
}

std::string lower_header_key(const std::string& key) {
    std::string out = key;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::string header_get_lower(
    const std::unordered_map<std::string, std::string>& headers,
    const std::string& key,
    const std::string& default_value
) {
    auto it = headers.find(lower_header_key(key));
    if (it == headers.end()) {
        return default_value;
    }
    return it->second;
}

bool read_http_request(
    int fd,
    const HttpReadLimits& limits,
    HttpRequest* out,
    int* error_status,
    std::string* error
) {
    if (!out) {
        if (error_status) *error_status = 500;
        if (error) *error = "null request output";
        return false;
    }

    std::string buffer;
    size_t header_end = std::string::npos;
    while ((header_end = buffer.find("\r\n\r\n")) == std::string::npos) {
        if (buffer.size() > limits.max_header_bytes) {
            if (error_status) *error_status = 413;
            if (error) *error = "request headers too large";
            return false;
        }
        if (!recv_more(fd, &buffer)) {
            if (error_status) *error_status = 400;
            if (error) *error = "failed to read request";
            return false;
        }
    }
    if (header_end > limits.max_header_bytes) {
        if (error_status) *error_status = 413;
        if (error) *error = "request headers too large";
        return false;
    }

    std::string head = buffer.substr(0, header_end + 2);
    if (!parse_request_head(head, out, error_status, error)) {
        return false;
    }

    size_t body_begin = header_end + 4;
    std::string content_length_text = header_get_lower(out->headers, "content-length");
    size_t content_length = 0;
    if (out->method == "POST") {
        if (content_length_text.empty()) {
            if (error_status) *error_status = 411;
            if (error) *error = "missing Content-Length";
            return false;
        }
        if (!parse_content_length(content_length_text, &content_length)) {
            if (error_status) *error_status = 400;
            if (error) *error = "invalid Content-Length";
            return false;
        }
        if (content_length > limits.max_body_bytes) {
            if (error_status) *error_status = 413;
            if (error) *error = "request body too large";
            return false;
        }
    }

    size_t have = buffer.size() - body_begin;
    while (have < content_length) {
        if (!recv_more(fd, &buffer)) {
            if (error_status) *error_status = 400;
            if (error) *error = "incomplete request body";
            return false;
        }
        have = buffer.size() - body_begin;
        if (have > limits.max_body_bytes) {
            if (error_status) *error_status = 413;
            if (error) *error = "request body too large";
            return false;
        }
    }
    if (content_length > 0) {
        out->body = buffer.substr(body_begin, content_length);
    }
    return true;
}

bool write_raw(int fd, const std::string& data) {
#ifdef _WIN32
    (void)fd;
    (void)data;
    return false;
#else
    const char* p = data.data();
    size_t remaining = data.size();
    while (remaining > 0) {
        ssize_t n = send(fd, p, remaining, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
#endif
}

bool write_http_response(int fd, const HttpResponse& response) {
    std::ostringstream os;
    os << "HTTP/1.1 " << response.status << ' '
       << http_status_text(response.status) << "\r\n";
    os << "Content-Type: " << response.content_type << "\r\n";
    os << "Content-Length: " << response.body.size() << "\r\n";
    os << "Connection: close\r\n";
    for (const auto& kv : response.headers) {
        os << kv.first << ": " << kv.second << "\r\n";
    }
    os << "\r\n";
    os << response.body;
    return write_raw(fd, os.str());
}

bool write_sse_event(int fd, const std::string& data) {
    return write_raw(fd, std::string("data: ") + data + "\n\n");
}

} // namespace llm_engine
