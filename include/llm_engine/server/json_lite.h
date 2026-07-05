#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm_engine {

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    using Array = std::vector<JsonValue>;
    using Object = std::unordered_map<std::string, JsonValue>;

    JsonValue();
    explicit JsonValue(std::nullptr_t);
    explicit JsonValue(bool value);
    explicit JsonValue(double value);
    explicit JsonValue(std::string value);
    explicit JsonValue(Array value);
    explicit JsonValue(Object value);

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool() const { return bool_value_; }
    double as_number() const { return number_value_; }
    const std::string& as_string() const { return string_value_; }
    const Array& as_array() const { return array_value_; }
    const Object& as_object() const { return object_value_; }

private:
    Type type_ = Type::Null;
    bool bool_value_ = false;
    double number_value_ = 0.0;
    std::string string_value_;
    Array array_value_;
    Object object_value_;
};

bool parse_json(const std::string& text, JsonValue* out, std::string* error);
std::string json_escape(const std::string& s);

const JsonValue* get_field(const JsonValue& obj, const std::string& key);
bool get_string(const JsonValue& obj, const std::string& key, std::string* out);
bool get_int(const JsonValue& obj, const std::string& key, int* out);
bool get_int64(const JsonValue& obj, const std::string& key, int64_t* out);
bool get_u64(const JsonValue& obj, const std::string& key, uint64_t* out);
bool get_double(const JsonValue& obj, const std::string& key, double* out);
bool get_bool(const JsonValue& obj, const std::string& key, bool* out);

} // namespace llm_engine
