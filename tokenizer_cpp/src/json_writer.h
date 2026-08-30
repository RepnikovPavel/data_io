// Writes tokenizer.json with exactly the same bytes as the reference
// artifacts (serde_json pretty, 2-space indent, raw UTF-8, keys in the
// tokenizers-crate serialization order), plus config.json {"model_type":
// "qwen3"} alongside it.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace json_writer {

// specials: token contents for ids 0..specials.size()-1.
// vocab_bytes: id -> raw byte content (specials hold their literal bytes).
// merges: (a, b) token ids in rank order.
void write_tokenizer_json(const std::string& path,
                          const std::vector<std::string>& specials,
                          const std::vector<std::string>& vocab_bytes,
                          const std::vector<std::pair<uint32_t, uint32_t>>& merges);

// Writes {"model_type": "qwen3"} next to the tokenizer.json (same as the
// reference trainer).
void write_config_json(const std::string& tokenizer_json_path);

}  // namespace json_writer
