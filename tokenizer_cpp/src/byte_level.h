// ByteLevel bytes <-> unicode mapping (GPT-2 style, identical to the
// `tokenizers` crate's ByteLevel pre-tokenizer and to our Rust trainer).
//
// Bytes in the printable ranges [0x21..=0x7E], [0xA1..=0xAC], [0xAE..=0xFF]
// map to the codepoint of the same value; all other bytes map to U+0100+n
// where n counts up over the non-printable bytes in increasing byte order.
//
// The map is a bijection bytes <-> chars, which is why the training loop can
// count words and pairs in raw byte space and only convert to byte-level char
// strings when writing tokenizer.json.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bytelevel {

// byte -> unicode codepoint of its ByteLevel char
inline std::vector<uint32_t> bytes_to_unicode_table() {
    std::vector<uint32_t> table(256);
    auto printable = [](uint32_t b) {
        return (0x21 <= b && b <= 0x7e) || (0xa1 <= b && b <= 0xac) ||
               (0xae <= b && b <= 0xff);
    };
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        if (printable(b)) {
            table[b] = b;
        } else {
            table[b] = 0x100 + n;
            ++n;
        }
    }
    return table;
}

// Append the UTF-8 encoding of a codepoint (codepoints here are <= U+0143,
// so 1-2 bytes suffice, but implement the general path for clarity).
inline void encode_utf8(uint32_t cp, std::string& out) {
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

// raw token bytes -> byte-level char string (UTF-8)
inline std::string bytes_to_bpe_string(const std::string& bytes,
                                       const std::vector<uint32_t>& table) {
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char b : bytes) encode_utf8(table[b], out);
    return out;
}

// Human-readable rendering of raw token bytes for progress logs: printable
// ASCII as-is, everything else as \xNN.
inline std::string show_bytes(const std::string& bytes) {
    std::string out = "\"";
    char buf[8];
    for (unsigned char b : bytes) {
        if (b >= 0x20 && b < 0x7f && b != '"' && b != '\\') {
            out.push_back((char)b);
        } else {
            snprintf(buf, sizeof buf, "\\x%02x", b);
            out += buf;
        }
    }
    out.push_back('"');
    return out;
}

}  // namespace bytelevel
