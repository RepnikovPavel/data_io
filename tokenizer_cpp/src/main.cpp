// train_tokenizer_cpp — a from-scratch, dependency-light C++17 BPE tokenizer
// trainer, algorithmically identical (bug-for-bug) to the reference
// `tokenizers`-crate BpeTrainer and to our Rust iterative trainer
// (tokenizer/src/bin/train_tokenizer_iter.rs, whose module docstring
// documents the exact semantics).
//
// Input modes:
//   --words <words.bin>   mode A (primary): word counts from the load phase
//                         of the Rust iterative trainer (cross-compatible
//                         binary format). With the official words.bin this
//                         reproduces the official tokenizer byte-for-byte.
//   --corpus <file.txt>   mode B (standalone/readability): one document per
//                         line, split with the same regex via PCRE2.
//                         NOTE: mode B skips NFC normalization (no unicode
//                         normalization library is linked) and applies no
//                         sampling limits — for experiments only.
//
// Output: HF tokenizer.json (byte-compatible with the reference artifacts)
// + config.json {"model_type": "qwen3"} alongside.
//
// Checkpoints: merges_<N>.bin every --checkpoint-interval merges in
// --checkpoint-dir (same binary format as the Rust trainer); on SIGINT /
// SIGTERM a checkpoint is written and the process exits with code 2; re-run
// the same command to resume from the latest checkpoint.
//
// Build (plain g++):
//   g++ -O3 -std=c++17 -pthread src/*.cpp -o train_tokenizer_cpp
//       $(pkg-config --cflags --libs libpcre2-8) -DTOKENIZER_CPP_HAVE_PCRE2
// or with CMake (PCRE2 optional — without it, mode B is disabled):
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "checkpoint.h"
#include "json_writer.h"
#include "pretok.h"
#include "trainer.h"

namespace {

// The 31 special tokens (ids 0..30), same list as the reference trainer.
const std::vector<std::string>& special_tokens() {
    static const std::vector<std::string> kSpecials = {
        "<|PAD|>",
        "<|direct|>", "<|cot|>", "<|noisy|>", "<|synth|>",
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>",
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
        "<|repo_name|>", "<|file_sep|>",
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"};
    return kSpecials;
}

void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s (--words <words.bin> | --corpus <file.txt>) -o <tokenizer.json>\n"
            "          [--vocab-size 65536] [--checkpoint-dir .] [--checkpoint-interval 100]\n"
            "          [--threads N] [--max-merges N]\n",
            argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string words_path, corpus_path, out_path;
    trainer::TrainOptions opts;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--words") words_path = next("--words");
        else if (a == "--corpus") corpus_path = next("--corpus");
        else if (a == "-o" || a == "--out") out_path = next("-o");
        else if (a == "--vocab-size") opts.vocab_size = std::stoull(next("--vocab-size"));
        else if (a == "--checkpoint-dir") opts.checkpoint_dir = next("--checkpoint-dir");
        else if (a == "--checkpoint-interval")
            opts.checkpoint_interval = std::stoull(next("--checkpoint-interval"));
        else if (a == "--threads") opts.threads = std::stoul(next("--threads"));
        else if (a == "--max-merges") opts.max_merges = std::stoull(next("--max-merges"));
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }

    if ((words_path.empty() == corpus_path.empty()) || out_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    try {
        std::vector<std::pair<std::string, uint64_t>> words;
        if (!words_path.empty()) {
            printf("[load] reading %s ...\n", words_path.c_str());
            auto wf = checkpoint::read_words(words_path);
            printf("[load] %llu docs, %.2f GiB text, %zu unique words\n",
                   (unsigned long long)wf.docs_total, wf.bytes_total / 1073741824.0,
                   wf.words.size());
            words = std::move(wf.words);
        } else {
            if (!pretok::kHavePcre2) {
                fprintf(stderr, "mode B unavailable: this build has no PCRE2 support\n");
                return 1;
            }
            printf("[load] counting words in %s (mode B: NO NFC normalization, no sampling "
                   "limits) ...\n", corpus_path.c_str());
            words = pretok::count_words_from_corpus(corpus_path);
            printf("[load] %zu unique words\n", words.size());
        }

        auto st = trainer::init_state(words, special_tokens(), opts);
        if (!trainer::run_merge_loop(st, opts)) return 0;  // stopped early (--max-merges)

        json_writer::write_tokenizer_json(out_path, special_tokens(), st.vocab_bytes, st.merges);
        json_writer::write_config_json(out_path);
        printf("[train] saved tokenizer to %s\n", out_path.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
