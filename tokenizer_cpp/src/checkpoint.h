// Checkpoint IO: words.bin (word counts, written by the load phase of the
// Rust iterative trainer) and merges_N.bin (merge-loop checkpoints).
//
// Formats are std-only, little-endian, length-prefixed, and cross-compatible
// with tokenizer/src/bin/train_tokenizer_iter.rs:
//
// words.bin:
//   u32 WORDS_MAGIC ("WBI1" = 0x57424931), u64 docs_total, u64 bytes_total,
//   u64 n_words, then per word: u32 len, len bytes (raw UTF-8 of the
//   pre-tokenizer word), u64 corpus count.
//
// merges_N.bin:
//   u32 MERGES_MAGIC ("MBI1" = 0x4D424931), u64 n_vocab, then per vocab entry
//   (id order: specials + alphabet + merged tokens): u32 len, len bytes;
//   u64 n_merges, then per merge: u32 a, u32 b (token ids).
//
// N in the file name is the number of merges recorded inside; resume picks
// the file with the largest N.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

struct WordsFile {
    uint64_t docs_total = 0;
    uint64_t bytes_total = 0;
    // (raw word bytes, corpus count), sorted by bytes (deterministic)
    std::vector<std::pair<std::string, uint64_t>> words;
};

struct MergesFile {
    std::vector<std::string> vocab_bytes;  // id -> raw byte content
    std::vector<std::pair<uint32_t, uint32_t>> merges;
};

// Throws std::runtime_error on any format/IO problem.
WordsFile read_words(const std::string& path);
void write_merges(const std::string& path, const std::vector<std::string>& vocab_bytes,
                  const std::vector<std::pair<uint32_t, uint32_t>>& merges);
MergesFile read_merges(const std::string& path);

// Latest merges_<N>.bin in dir, or {"", 0} with found=false.
struct LatestMerges {
    bool found = false;
    size_t n = 0;
    std::string path;
};
LatestMerges latest_merges_checkpoint(const std::string& dir);

}  // namespace checkpoint
