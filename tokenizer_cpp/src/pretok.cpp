#include "pretok.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#ifdef TOKENIZER_CPP_HAVE_PCRE2
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#endif

namespace pretok {

// The exact pre-tokenizer regex (GPT-2 style) shared by the whole pipeline.
#ifdef TOKENIZER_CPP_HAVE_PCRE2
static const char* kSplitRegex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::vector<std::pair<std::string, uint64_t>> count_words_from_corpus(const std::string& path) {
    int errcode = 0;
    PCRE2_SIZE erroffset = 0;
    // PCRE2_UTF + PCRE2_UCP give \p{L} / \p{N} / \s their unicode meaning;
    // PCRE2_MATCH_INVALID_UTF lets matching survive invalid UTF-8 in input.
    pcre2_code* re = pcre2_compile((PCRE2_SPTR)kSplitRegex, PCRE2_ZERO_TERMINATED,
                                   PCRE2_UTF | PCRE2_UCP | PCRE2_MATCH_INVALID_UTF, &errcode,
                                   &erroffset, nullptr);
    if (!re) {
        PCRE2_UCHAR buf[256];
        pcre2_get_error_message(errcode, buf, sizeof buf);
        throw std::runtime_error(std::string("PCRE2 compile failed: ") + (char*)buf);
    }
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);

    std::unordered_map<std::string, uint64_t> counts;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        PCRE2_SIZE offset = 0;
        while (offset <= line.size()) {
            int rc = pcre2_match(re, (PCRE2_SPTR)line.data(), line.size(), offset, 0, md, nullptr);
            if (rc < 0) break;  // no more matches
            PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
            if (ov[1] > ov[0]) counts[line.substr(ov[0], ov[1] - ov[0])] += 1;
            offset = ov[1] > ov[0] ? ov[1] : ov[1] + 1;  // guard against empty matches
        }
    }
    pcre2_match_data_free(md);
    pcre2_code_free(re);

    std::vector<std::pair<std::string, uint64_t>> words(counts.begin(), counts.end());
    std::sort(words.begin(), words.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    return words;
}

#else  // !TOKENIZER_CPP_HAVE_PCRE2

std::vector<std::pair<std::string, uint64_t>> count_words_from_corpus(const std::string&) {
    throw std::runtime_error(
        "this build has no PCRE2 support (mode B unavailable); "
        "install libpcre2-dev and rebuild, or use --words <words.bin>");
}

#endif

}  // namespace pretok
