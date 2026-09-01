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
#include <cstring>
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

// Fast 64-bit hash for short byte strings (multiplicative mix over 8-byte
// chunks). Used as the flat-cache key; hits are verified bytewise, so a hash
// collision can never change a count.
inline uint64_t word_hash(const char* p, size_t n) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        memcpy(&v, p + i, 8);
        h ^= v * 0x9E3779B185EBCA87ull;
        h = (h << 27) | (h >> 37);
        h *= 0xC2B2AE3D27D4EB4Full;
    }
    if (i < n) {
        uint64_t v = 0;
        memcpy(&v, p + i, n - i);
        h ^= v * 0x9E3779B185EBCA87ull;
        h = (h << 27) | (h >> 37);
    }
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return h;
}

// Flat open-addressing word -> count cache. One cache line per probe (no
// node pointers); the word bytes are copied into a bump arena on insert and
// hits verify bytewise. Insertions stop at 70% load (late-corpus misses just
// run the BPE directly, which is cheap) — this bounds memory and never
// changes counts.
class FlatCache {
  public:
    explicit FlatCache(size_t cap = 1 << 21) { reset(cap); }

    static constexpr uint32_t EMPTY = UINT32_MAX;

    void reset(size_t cap) {
        cap_ = cap;
        mask_ = cap - 1;
        slots_.assign(cap, Slot{0, 0, EMPTY, 0});
        arena_.clear();
        size_ = 0;
    }
    void clear() {
        std::fill(slots_.begin(), slots_.end(), Slot{0, 0, EMPTY, 0});
        arena_.clear();
        size_ = 0;
    }

    // Returns the count, or UINT64_MAX on miss.
    uint64_t get(std::string_view w) const {
        uint64_t h = word_hash(w.data(), w.size());
        size_t idx = h & mask_;
        for (;;) {
            const Slot& s = slots_[idx];
            if (s.len == EMPTY) return UINT64_MAX;
            if (s.hash == h && s.len == w.size() &&
                memcmp(arena_.data() + s.off, w.data(), w.size()) == 0)
                return s.count;
            idx = (idx + 1) & mask_;
        }
    }

    void put(std::string_view w, uint64_t count) {
        if (size_ >= (cap_ * 7) / 10) return;  // stop inserting when 70% full
        uint64_t h = word_hash(w.data(), w.size());
        size_t idx = h & mask_;
        while (slots_[idx].len != EMPTY) idx = (idx + 1) & mask_;
        uint32_t off = (uint32_t)arena_.size();
        arena_.insert(arena_.end(), w.begin(), w.end());
        slots_[idx] = Slot{h, off, (uint32_t)w.size(), count};
        ++size_;
    }

  private:
    struct Slot {
        uint64_t hash;
        uint32_t off;
        uint32_t len;
        uint64_t count;
    };
    std::vector<Slot> slots_;
    std::vector<char> arena_;
    size_t cap_ = 0, mask_ = 0, size_ = 0;
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
    FlatCache cache_;     // word bytes -> tokens
    std::vector<uint16_t> scratch_;  // reusable token buffer for count_word

    uint64_t count_word(std::string_view word);  // byte map + merges for one word
    uint64_t count_span(const char* data, size_t len);  // scanner path (fused)
    uint64_t count_span_fused(const char* data, size_t len, bool nfc_done);
    uint64_t count_span_pcre2(const char* data, size_t len);  // PCRE2 reference path
};

}  // namespace encoder
