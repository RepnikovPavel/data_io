#include "json_writer.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "byte_level.h"

namespace json_writer {

namespace {

// serde_json-compatible string escaping: " -> \" and \ -> \\, control chars
// < 0x20 via the \b \t \n \f \r shortcuts or \u00XX; everything else raw
// UTF-8. (Token strings only ever contain printable chars and the byte-level
// U+0100..U+0143 range, so in practice only \" and \\ ever fire.)
void write_json_string(std::ostream& os, const std::string& s) {
    os.put('"');
    char buf[8];
    for (unsigned char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b"; break;
            case '\t': os << "\\t"; break;
            case '\n': os << "\\n"; break;
            case '\f': os << "\\f"; break;
            case '\r': os << "\\r"; break;
            default:
                if (c < 0x20) {
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    os << buf;
                } else {
                    os.put((char)c);
                }
        }
    }
    os.put('"');
}

// The fixed ByteLevel block used for post_processor and decoder.
const char* kByteLevel =
    "{\n"
    "    \"type\": \"ByteLevel\",\n"
    "    \"add_prefix_space\": false,\n"
    "    \"trim_offsets\": false,\n"
    "    \"use_regex\": false\n"
    "  }";

}  // namespace

void write_tokenizer_json(const std::string& path,
                          const std::vector<std::string>& specials,
                          const std::vector<std::string>& vocab_bytes,
                          const std::vector<std::pair<uint32_t, uint32_t>>& merges) {
    const auto table = bytelevel::bytes_to_unicode_table();
    const size_t n_specials = specials.size();

    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) throw std::runtime_error("cannot create " + path);

    os << "{\n";
    os << "  \"version\": \"1.0\",\n";
    os << "  \"truncation\": null,\n";
    os << "  \"padding\": null,\n";

    // added_tokens: the special tokens with their ids and fixed flags.
    os << "  \"added_tokens\": [\n";
    for (size_t i = 0; i < n_specials; ++i) {
        os << "    {\n";
        os << "      \"id\": " << i << ",\n";
        os << "      \"content\": ";
        write_json_string(os, specials[i]);
        os << ",\n";
        os << "      \"single_word\": false,\n";
        os << "      \"lstrip\": false,\n";
        os << "      \"rstrip\": false,\n";
        os << "      \"normalized\": false,\n";
        os << "      \"special\": true\n";
        os << "    }" << (i + 1 < n_specials ? ",\n" : "\n");
    }
    os << "  ],\n";

    os << "  \"normalizer\": {\n";
    os << "    \"type\": \"NFC\"\n";
    os << "  },\n";

    // pre_tokenizer: Sequence[Split(regex, Isolated), ByteLevel(...)]
    os << "  \"pre_tokenizer\": {\n";
    os << "    \"type\": \"Sequence\",\n";
    os << "    \"pretokenizers\": [\n";
    os << "      {\n";
    os << "        \"type\": \"Split\",\n";
    os << "        \"pattern\": {\n";
    os << "          \"Regex\": ";
    write_json_string(
        os, "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
            "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
    os << "\n";
    os << "        },\n";
    os << "        \"behavior\": \"Isolated\",\n";
    os << "        \"invert\": false\n";
    os << "      },\n";
    os << "      {\n";
    os << "        \"type\": \"ByteLevel\",\n";
    os << "        \"add_prefix_space\": false,\n";
    os << "        \"trim_offsets\": false,\n";
    os << "        \"use_regex\": false\n";
    os << "      }\n";
    os << "    ]\n";
    os << "  },\n";

    os << "  \"post_processor\": " << kByteLevel << ",\n";
    os << "  \"decoder\": " << kByteLevel << ",\n";

    os << "  \"model\": {\n";
    os << "    \"type\": \"BPE\",\n";
    os << "    \"dropout\": null,\n";
    os << "    \"unk_token\": null,\n";
    os << "    \"continuing_subword_prefix\": null,\n";
    os << "    \"end_of_word_suffix\": null,\n";
    os << "    \"fuse_unk\": false,\n";
    os << "    \"byte_fallback\": false,\n";
    os << "    \"ignore_merges\": false,\n";

    // vocab ordered by id (the reference serializes with OrderedVocabIter).
    os << "    \"vocab\": {\n";
    for (size_t id = 0; id < vocab_bytes.size(); ++id) {
        os << "      ";
        // Specials are stored as their literal bytes; everything else is
        // converted to the byte-level char string.
        std::string tok = id < n_specials
                              ? vocab_bytes[id]
                              : bytelevel::bytes_to_bpe_string(vocab_bytes[id], table);
        write_json_string(os, tok);
        os << ": " << id << (id + 1 < vocab_bytes.size() ? ",\n" : "\n");
    }
    os << "    },\n";

    os << "    \"merges\": [\n";
    for (size_t i = 0; i < merges.size(); ++i) {
        auto [a, b] = merges[i];
        os << "      [\n";
        os << "        ";
        write_json_string(os, bytelevel::bytes_to_bpe_string(vocab_bytes[a], table));
        os << ",\n";
        os << "        ";
        write_json_string(os, bytelevel::bytes_to_bpe_string(vocab_bytes[b], table));
        os << "\n";
        os << "      ]" << (i + 1 < merges.size() ? ",\n" : "\n");
    }
    os << "    ]\n";

    os << "  }\n";
    os << "}";  // no trailing newline, same as the reference
    if (!os) throw std::runtime_error("write failed: " + path);
}

void write_config_json(const std::string& tokenizer_json_path) {
    namespace fs = std::filesystem;
    fs::path p = fs::path(tokenizer_json_path).parent_path();
    fs::path cfg = p.empty() ? fs::path("config.json") : p / "config.json";
    std::ofstream os(cfg, std::ios::binary | std::ios::trunc);
    if (!os) throw std::runtime_error("cannot create " + cfg.string());
    os << "{\"model_type\": \"qwen3\"}";
}

}  // namespace json_writer
