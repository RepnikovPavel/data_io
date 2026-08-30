// Standalone input mode B: read a plain text corpus (one document per line),
// split each document into words with the exact pre-tokenizer regex the
// reference pipeline uses, and count word frequencies.
//
// HONEST LIMITATION (documented in tokenizer/docs): unlike the real pipeline,
// mode B does NOT apply NFC normalization (no unicode normalization library
// is linked) and applies no per-file sampling limits or truncation. Word
// counts therefore differ slightly from the reference on non-NFC text. This
// mode exists for readability and small-scale experiments, not for
// reproducing the official artifacts — use mode A (--words words.bin) for
// that.
//
// ByteLevel mapping is a bijection on bytes, so words are counted as raw
// UTF-8 bytes (same as the Rust trainer's words.bin contents).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pretok {

// Returns (word bytes, count) sorted by bytes. Requires PCRE2 at compile
// time (TOKENIZER_CPP_HAVE_PCRE2); otherwise throws std::runtime_error.
std::vector<std::pair<std::string, uint64_t>> count_words_from_corpus(const std::string& path);

constexpr bool kHavePcre2 =
#ifdef TOKENIZER_CPP_HAVE_PCRE2
    true;
#else
    false;
#endif

}  // namespace pretok
