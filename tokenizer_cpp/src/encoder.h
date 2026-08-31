// BPE encoder: loads a HF tokenizer.json (byte-level BPE as produced by the
// trainers in this repo) and encodes text exactly like the `tokenizers`
// crate / python lib does with this pipeline:
//
//   1. Split out added special tokens (all 31 are normalized:false, so they
//      are matched on the RAW text, leftmost, longest-at-position), each
//      special match counts as 1 token and is not pretokenized.
//   2. NFC-normalize each non-special span (utf8proc). Fast path: pure-ASCII
//      spans skip NFC entirely (ASCII is NFC-stable).
//   3. Split with the GPT-2 regex (PCRE2, UTF+UCP) — behavior Isolated, so
//      each regex match is one word.
//   4. Map each word's bytes through the ByteLevel table and apply merges
//      greedily by rank (repeatedly merge the lowest-rank adjacent pair).
//      Bytes whose byte-level char is missing from the vocab (never seen in
//      training) are dropped, same as the reference model with no unk_token
//      and no byte_fallback.
//
// encode_count() returns the number of tokens. Word results are memoized per
// Encoder instance (the corpus has extreme word repetition, so the cache is
// the main speedup). NOT thread-safe by itself — use one Encoder per thread
// (Encoder::clone_shallow shares the read-only tables).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace encoder {

struct Tables {
    // byte -> token id of its byte-level char. uint16_t suffices: vocab_size
    // is 65536, so ids fit 0..65535; the 0xFFFF sentinel is safe for THIS
    // array because alphabet (single-byte) ids are all < 272. Merged-token
    // ids up to 65535 never appear here.
    std::vector<uint16_t> byte_id;
    // (a << 32 | b) -> (rank << 32) | new_id
    std::unordered_map<uint64_t, uint64_t> merge_rank;
    // special token contents (ids 0..30)
    std::vector<std::string> specials;
    // PCRE2 compiled pattern (opaque; owned). Thread-safe to SHARE;
    // match data is NOT shared — each Encoder owns its own.
    void* re = nullptr;
    ~Tables();
};

// Transparent hash/equal so the word cache can be queried with a
// string_view without constructing a std::string (hot path).
struct StrHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const {
        return std::hash<std::string_view>{}(s);
    }
};
struct StrEq {
    using is_transparent = void;
    bool operator()(std::string_view a, std::string_view b) const { return a == b; }
};

class Encoder {
  public:
    // Load from a tokenizer.json path. Throws on format problems.
    explicit Encoder(const std::string& tokenizer_json_path);
    explicit Encoder(std::shared_ptr<Tables> tables);
    Encoder clone_shallow() const { return Encoder(t_); }
    ~Encoder();
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&& o) noexcept
        : t_(std::move(o.t_)), md_(o.md_), cache_(std::move(o.cache_)) {
        o.md_ = nullptr;
    }

    uint64_t encode_count(std::string_view text);

    // Debug: print the word split of `text`, one escaped word per line
    // (prefix "W"), specials as "S <content>". Used by --dump-splits.
    void dump_doc(std::string_view text, FILE* out);

  private:    std::shared_ptr<Tables> t_;
    void* md_ = nullptr;  // per-Encoder PCRE2 match data (thread-local state)
    std::unordered_map<std::string, uint64_t, StrHash, StrEq> cache_;  // word -> tokens
    std::vector<uint16_t> scratch_;  // reusable token buffer for count_word

    uint64_t count_word(std::string_view word);        // steps 3-4 for one word
    uint64_t count_span(const char* data, size_t len); // steps 2-4 for a span
};

}  // namespace encoder
