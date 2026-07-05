#include "llm_engine/server/json_lite.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace llm_engine {

JsonValue::JsonValue() = default;
JsonValue::JsonValue(std::nullptr_t) : type_(Type::Null) {}
JsonValue::JsonValue(bool value) : type_(Type::Bool), bool_value_(value) {}
JsonValue::JsonValue(double value) : type_(Type::Number), number_value_(value) {}
JsonValue::JsonValue(std::string value)
    : type_(Type::String), string_value_(std::move(value)) {}
JsonValue::JsonValue(Array value)
    : type_(Type::Array), array_value_(std::move(value)) {}
JsonValue::JsonValue(Object value)
    : type_(Type::Object), object_value_(std::move(value)) {}

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    bool parse(JsonValue* out, std::string* error) {
        skip_ws();
        if (!parse_value(out, error)) {
            return false;
        }
        skip_ws();
        if (pos_ != text_.size()) {
            set_error(error, "trailing characters");
            return false;
        }
        return true;
    }

private:
    void skip_ws() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                break;
            }
            ++pos_;
        }
    }

    bool consume(char expected) {
        if (pos_ >= text_.size() || text_[pos_] != expected) {
            return false;
        }
        ++pos_;
        return true;
    }

    bool consume_literal(const char* literal) {
        size_t begin = pos_;
        for (const char* p = literal; *p; ++p) {
            if (pos_ >= text_.size() || text_[pos_] != *p) {
                pos_ = begin;
                return false;
            }
            ++pos_;
        }
        return true;
    }

    void set_error(std::string* error, const std::string& message) const {
        if (error) {
            std::ostringstream os;
            os << message << " at byte " << pos_;
            *error = os.str();
        }
    }

    bool parse_value(JsonValue* out, std::string* error) {
        skip_ws();
        if (pos_ >= text_.size()) {
            set_error(error, "unexpected end of input");
            return false;
        }
        char c = text_[pos_];
        if (c == 'n') {
            if (!consume_literal("null")) {
                set_error(error, "invalid null literal");
                return false;
            }
            *out = JsonValue(nullptr);
            return true;
        }
        if (c == 't') {
            if (!consume_literal("true")) {
                set_error(error, "invalid true literal");
                return false;
            }
            *out = JsonValue(true);
            return true;
        }
        if (c == 'f') {
            if (!consume_literal("false")) {
                set_error(error, "invalid false literal");
                return false;
            }
            *out = JsonValue(false);
            return true;
        }
        if (c == '"') {
            std::string s;
            if (!parse_string(&s, error)) {
                return false;
            }
            *out = JsonValue(std::move(s));
            return true;
        }
        if (c == '[') {
            return parse_array(out, error);
        }
        if (c == '{') {
            return parse_object(out, error);
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number(out, error);
        }
        set_error(error, "unexpected character");
        return false;
    }

    bool parse_array(JsonValue* out, std::string* error) {
        if (!consume('[')) {
            set_error(error, "expected array");
            return false;
        }
        JsonValue::Array values;
        skip_ws();
        if (consume(']')) {
            *out = JsonValue(std::move(values));
            return true;
        }
        while (true) {
            JsonValue value;
            if (!parse_value(&value, error)) {
                return false;
            }
            values.push_back(std::move(value));
            skip_ws();
            if (consume(']')) {
                *out = JsonValue(std::move(values));
                return true;
            }
            if (!consume(',')) {
                set_error(error, "expected ',' or ']'");
                return false;
            }
        }
    }

    bool parse_object(JsonValue* out, std::string* error) {
        if (!consume('{')) {
            set_error(error, "expected object");
            return false;
        }
        JsonValue::Object fields;
        skip_ws();
        if (consume('}')) {
            *out = JsonValue(std::move(fields));
            return true;
        }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(&key, error)) {
                return false;
            }
            skip_ws();
            if (!consume(':')) {
                set_error(error, "expected ':'");
                return false;
            }
            JsonValue value;
            if (!parse_value(&value, error)) {
                return false;
            }
            fields[std::move(key)] = std::move(value);
            skip_ws();
            if (consume('}')) {
                *out = JsonValue(std::move(fields));
                return true;
            }
            if (!consume(',')) {
                set_error(error, "expected ',' or '}'");
                return false;
            }
        }
    }

    bool parse_number(JsonValue* out, std::string* error) {
        size_t begin = pos_;
        if (text_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= text_.size()) {
            set_error(error, "invalid number");
            return false;
        }
        if (text_[pos_] == '0') {
            ++pos_;
        } else if (text_[pos_] >= '1' && text_[pos_] <= '9') {
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            set_error(error, "invalid number");
            return false;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                set_error(error, "invalid number fraction");
                return false;
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                set_error(error, "invalid number exponent");
                return false;
            }
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                ++pos_;
            }
        }
        std::string number = text_.substr(begin, pos_ - begin);
        errno = 0;
        char* end = nullptr;
        double value = std::strtod(number.c_str(), &end);
        if (end == number.c_str() || errno == ERANGE || !std::isfinite(value)) {
            set_error(error, "number out of range");
            return false;
        }
        *out = JsonValue(value);
        return true;
    }

    bool parse_string(std::string* out, std::string* error) {
        if (!consume('"')) {
            set_error(error, "expected string");
            return false;
        }
        std::string value;
        while (pos_ < text_.size()) {
            unsigned char c = static_cast<unsigned char>(text_[pos_++]);
            if (c == '"') {
                *out = std::move(value);
                return true;
            }
            if (c < 0x20) {
                set_error(error, "control character in string");
                return false;
            }
            if (c != '\\') {
                value.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= text_.size()) {
                set_error(error, "unterminated string escape");
                return false;
            }
            char e = text_[pos_++];
            switch (e) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u': {
                    uint32_t code = 0;
                    if (!parse_hex4(&code)) {
                        set_error(error, "invalid unicode escape");
                        return false;
                    }
                    append_utf8(code, &value);
                    break;
                }
                default:
                    set_error(error, "invalid string escape");
                    return false;
            }
        }
        set_error(error, "unterminated string");
        return false;
    }

    bool parse_hex4(uint32_t* out) {
        if (pos_ + 4 > text_.size()) {
            return false;
        }
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<uint32_t>(10 + c - 'a');
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<uint32_t>(10 + c - 'A');
            } else {
                return false;
            }
        }
        *out = value;
        return true;
    }

    static void append_utf8(uint32_t code, std::string* out) {
        if (code <= 0x7f) {
            out->push_back(static_cast<char>(code));
        } else if (code <= 0x7ff) {
            out->push_back(static_cast<char>(0xc0 | (code >> 6)));
            out->push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else {
            out->push_back(static_cast<char>(0xe0 | (code >> 12)));
            out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            out->push_back(static_cast<char>(0x80 | (code & 0x3f)));
        }
    }

    const std::string& text_;
    size_t pos_ = 0;
};

} // namespace

bool parse_json(const std::string& text, JsonValue* out, std::string* error) {
    if (!out) {
        if (error) {
            *error = "null output";
        }
        return false;
    }
    Parser parser(text);
    return parser.parse(out, error);
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
                    os << "\\u"
                       << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(c)
                       << std::dec << std::setfill(' ');
                } else {
                    os << c;
                }
                break;
        }
    }
    return os.str();
}

const JsonValue* get_field(const JsonValue& obj, const std::string& key) {
    if (!obj.is_object()) {
        return nullptr;
    }
    auto it = obj.as_object().find(key);
    if (it == obj.as_object().end()) {
        return nullptr;
    }
    return &it->second;
}

bool get_string(const JsonValue& obj, const std::string& key, std::string* out) {
    const JsonValue* v = get_field(obj, key);
    if (!v || !v->is_string()) {
        return false;
    }
    if (out) {
        *out = v->as_string();
    }
    return true;
}

bool get_int(const JsonValue& obj, const std::string& key, int* out) {
    int64_t value = 0;
    if (!get_int64(obj, key, &value)) {
        return false;
    }
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    if (out) {
        *out = static_cast<int>(value);
    }
    return true;
}

bool get_int64(const JsonValue& obj, const std::string& key, int64_t* out) {
    const JsonValue* v = get_field(obj, key);
    if (!v || !v->is_number()) {
        return false;
    }
    double d = v->as_number();
    if (!std::isfinite(d) ||
        d < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        d > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    if (out) {
        *out = static_cast<int64_t>(d);
    }
    return true;
}

bool get_u64(const JsonValue& obj, const std::string& key, uint64_t* out) {
    const JsonValue* v = get_field(obj, key);
    if (!v || !v->is_number()) {
        return false;
    }
    double d = v->as_number();
    if (!std::isfinite(d) || d < 0.0 ||
        d > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        return false;
    }
    if (out) {
        *out = static_cast<uint64_t>(d);
    }
    return true;
}

bool get_double(const JsonValue& obj, const std::string& key, double* out) {
    const JsonValue* v = get_field(obj, key);
    if (!v || !v->is_number()) {
        return false;
    }
    if (out) {
        *out = v->as_number();
    }
    return true;
}

bool get_bool(const JsonValue& obj, const std::string& key, bool* out) {
    const JsonValue* v = get_field(obj, key);
    if (!v || !v->is_bool()) {
        return false;
    }
    if (out) {
        *out = v->as_bool();
    }
    return true;
}

} // namespace llm_engine
