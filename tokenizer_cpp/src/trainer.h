// The BPE merge loop — bug-for-bug compatible with the `tokenizers` 0.22
// crate's BpeTrainer (and therefore with our Rust iterative trainer, whose
// module docstring documents these semantics in detail):
//
//  * Initial vocab = special tokens (ids 0..30) + the byte-level chars
//    PRESENT in the corpus (not all 256), sorted by char codepoint.
//  * Pair counts are i32 with wrapping arithmetic. Pairs occurring more than
//    i32::MAX times wrap negative; since the priority queue only admits
//    positive counts, such pairs are never merged. Queue entries store the
//    reference's `count as u64` (sign-extended) value.
//  * Merge selection: highest count; ties broken by ascending token-id pair
//    (the reference's Merge::cmp).
//  * After each merge, pair counts are updated with per-merge-site deltas
//    (-1 for the two outer pairs, +1 for the two new outer pairs, weighted
//    by the word's count); the merged pair's own count is never decremented.
//  * Fresh queue entries are pushed only for pairs that received a POSITIVE
//    change (this is what keeps overflowed pairs out of the queue forever).
//  * Stale queue entries are re-pushed with the current (sign-extended)
//    count, exactly like the reference's lazy heap fix-up.
//  * Stop when vocab_size is reached, or the best count drops below
//    min_frequency (2), or the queue empties.
//  * A merge whose concatenated content already exists in the vocab (e.g. a
//    special token string) reuses the existing id; the merge is still
//    recorded (reference behavior).
#pragma once

#include <cstdint>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trainer {

// A pair of token ids packed into one key: a << 32 | b.
using PairKey = uint64_t;
inline PairKey pair_key(uint32_t a, uint32_t b) {
    return ((uint64_t)a << 32) | b;
}

// Wrapping i32 arithmetic (signed overflow is UB in C++, so go through
// unsigned). Mirrors release-mode Rust i32 arithmetic in the reference.
inline int32_t wrap_add(int32_t x, int32_t y) {
    return (int32_t)((uint32_t)x + (uint32_t)y);
}
inline int32_t wrap_mul(int32_t x, int32_t y) {
    return (int32_t)((uint32_t)x * (uint32_t)y);
}
// u64 word count -> i32 with truncation, as in the reference (`as i32`).
inline int32_t count_as_i32(uint64_t c) { return (int32_t)(uint32_t)c; }
// The reference stores queue counts as `count as u64` (sign-extended i32).
inline uint64_t queue_count(int32_t c) { return (uint64_t)(int64_t)c; }

// Priority queue entry. Top = highest count; ties -> smallest (a, b).
struct QEntry {
    uint64_t count;
    uint32_t a, b;
};
inline bool operator<(const QEntry& x, const QEntry& y) {
    if (x.count != y.count) return x.count < y.count;
    return std::tie(x.a, x.b) > std::tie(y.a, y.b);
}

struct TrainState {
    // id -> raw byte content (specials: their literal UTF-8 bytes)
    std::vector<std::string> vocab_bytes;
    // raw byte content -> id
    std::unordered_map<std::string, uint32_t> content_to_id;
    std::vector<std::pair<uint32_t, uint32_t>> merges;
    // per unique word: token id sequence + corpus count
    std::vector<std::vector<uint32_t>> toks;
    std::vector<uint64_t> counts;
    // weighted adjacent-pair counts (wrapping i32)
    std::unordered_map<PairKey, int32_t> pair_counts;
    // pair -> indices of words containing it (append-only, may be stale;
    // sorted + deduped and re-validated when the pair is selected)
    std::unordered_map<PairKey, std::vector<uint32_t>> where_pair;
    std::priority_queue<QEntry> heap;
};

struct TrainOptions {
    size_t vocab_size = 65536;
    std::string checkpoint_dir = ".";
    size_t checkpoint_interval = 100;
    // Stop after N merges without writing tokenizer.json (testing).
    size_t max_merges = 0;  // 0 = no cap
    unsigned threads = 0;   // 0 = hardware_concurrency
};

// Build the initial state from word counts (words.bin or corpus mode B).
// Resumes from the latest merges checkpoint in opts.checkpoint_dir if one
// exists: recorded merges are replayed onto the words (greedy by merge rank,
// which reproduces the training-time state exactly), then pair counts are
// rebuilt from scratch.
TrainState init_state(const std::vector<std::pair<std::string, uint64_t>>& words,
                      const std::vector<std::string>& special_tokens,
                      const TrainOptions& opts);

// Run the merge loop until vocab_size / min_frequency / exhaustion.
// Returns true when training completed (tokenizer.json should be written),
// false when it stopped early due to opts.max_merges (checkpoint written,
// no tokenizer.json). On SIGINT/SIGTERM, checkpoints and exits with code 2.
// Checkpoints every opts.checkpoint_interval merges.
bool run_merge_loop(TrainState& st, const TrainOptions& opts);

}  // namespace trainer
