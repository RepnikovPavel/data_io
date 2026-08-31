// Minimal JSON parser — just enough for HF tokenizer.json (objects, arrays,
// strings with \uXXXX and the standard escapes, numbers, true/false/null).
// No external dependencies; errors throw std::runtime_error with an offset.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace minijson {

struct Value;
using Object = std::vector<std::pair<std::string, Value>>;  // insertion order
using Array = std::vector<Value>;

struct Value {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    Array arr;
    Object obj;

    const Value& at(const std::string& key) const {
        for (const auto& [k, v] : obj)
            if (k == key) return v;
        throw std::runtime_error("json: missing key '" + key + "'");
    }
    uint64_t as_u64() const { return (uint64_t)num; }
};

class Parser {
  public:
    explicit Parser(const std::string& s) : s_(s) {}

    Value parse() {
        Value v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) throw err("trailing data");
        return v;
    }

  private:
    const std::string& s_;
    size_t pos_ = 0;

    std::runtime_error err(const std::string& msg) const {
        return std::runtime_error("json parse error at byte " + std::to_string(pos_) + ": " + msg);
    }
    void skip_ws() {
        while (pos_ < s_.size() &&
               (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
            ++pos_;
    }
    char peek() {
        if (pos_ >= s_.size()) throw err("unexpected eof");
        return s_[pos_];
    }
    char get() {
        char c = peek();
        ++pos_;
        return c;
    }
    void expect(char c) {
        if (get() != c) throw err(std::string("expected '") + c + "'");
    }
    void expect_lit(const char* lit) {
        for (const char* p = lit; *p; ++p) expect(*p);
    }

    Value parse_value() {
        skip_ws();
        switch (peek()) {
            case '{': return parse_obj();
            case '[': return parse_arr();
            case '"': {
                Value v;
                v.kind = Value::Str;
                v.str = parse_str();
                return v;
            }
            case 't': expect_lit("true"); { Value v; v.kind = Value::Bool; v.b = true; return v; }
            case 'f': expect_lit("false"); { Value v; v.kind = Value::Bool; return v; }
            case 'n': expect_lit("null"); return Value{};
            default: return parse_num();
        }
    }

    Value parse_obj() {
        Value v;
        v.kind = Value::Obj;
        expect('{');
        skip_ws();
        if (peek() == '}') { ++pos_; return v; }
        for (;;) {
            skip_ws();
            std::string key = parse_str();
            skip_ws();
            expect(':');
            v.obj.emplace_back(std::move(key), parse_value());
            skip_ws();
            char c = get();
            if (c == '}') return v;
            if (c != ',') throw err("expected ',' or '}'");
        }
    }

    Value parse_arr() {
        Value v;
        v.kind = Value::Arr;
        expect('[');
        skip_ws();
        if (peek() == ']') { ++pos_; return v; }
        for (;;) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = get();
            if (c == ']') return v;
            if (c != ',') throw err("expected ',' or ']'");
        }
    }

    void encode_utf8(uint32_t cp, std::string& out) {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xc0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3f)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xe0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back((char)(0x80 | (cp & 0x3f)));
        } else {
            out.push_back((char)(0xf0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back((char)(0x80 | (cp & 0x3f)));
        }
    }

    uint32_t hex4() {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = get();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else throw err("bad \\u escape");
        }
        return v;
    }

    std::string parse_str() {
        expect('"');
        std::string out;
        for (;;) {
            char c = get();
            if (c == '"') return out;
            if (c == '\\') {
                switch (get()) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = hex4();
                        if (cp >= 0xd800 && cp <= 0xdbff) {  // surrogate pair
                            if (get() != '\\' || get() != 'u') throw err("bad surrogate");
                            uint32_t lo = hex4();
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        }
                        encode_utf8(cp, out);
                        break;
                    }
                    default: throw err("bad escape");
                }
            } else {
                out.push_back(c);  // raw UTF-8 passes through
            }
        }
    }

    Value parse_num() {
        size_t start = pos_;
        while (pos_ < s_.size() &&
               (s_[pos_] == '-' || s_[pos_] == '+' || s_[pos_] == '.' || s_[pos_] == 'e' ||
                s_[pos_] == 'E' || (s_[pos_] >= '0' && s_[pos_] <= '9')))
            ++pos_;
        if (start == pos_) throw err("bad value");
        Value v;
        v.kind = Value::Num;
        v.num = std::stod(s_.substr(start, pos_ - start));
        return v;
    }
};

inline Value parse(const std::string& s) { return Parser(s).parse(); }

// Read a whole file into a string.
inline std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string s(n, '\0');
    if (n && fread(s.data(), 1, n, f) != (size_t)n) {
        fclose(f);
        throw std::runtime_error("short read: " + path);
    }
    fclose(f);
    return s;
}

}  // namespace minijson
