// Hand-rolled replacement for the PCRE2 pre-tokenizer regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// behavior Isolated (each match is one word; the pattern matches at every
// position, so the document partitions into consecutive matches).
//
// Motivation: pure speed — the PCRE2 interpreter is slow on this pattern and
// even JIT only gets ~3x; this scanner is table-driven. The semantics below
// replicate PCRE2 leftmost-first alternation + greedy quantifiers exactly
// (verified by differential tests against PCRE2 on real corpus data and
// adversarial fuzz strings — see scripts/test_count_tokens.sh and the
// COUNT_SPLIT=pcre2 fallback).
//
// Unicode classes (matching PCRE2 UTF+UCP):
//   \p{L} = general category L* (Lu,Ll,Lt,Lm,Lo; utf8proc 1..5)
//   \p{N} = general category N* (Nd,Nl,No; utf8proc 9..11)
//   \s    = Unicode White_Space property (exactly the set PCRE2 UCP uses):
//           09-0D, 20, 85, A0, 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000
#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include <utf8proc.h>

namespace splitter {

enum Cls : uint8_t { CLS_L, CLS_N, CLS_WS, CLS_OTHER };

inline bool is_ws_cp(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || cp == 0x20 || cp == 0x85 || cp == 0xa0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

// ASCII fast table: bit0 = letter, bit1 = digit, bit2 = whitespace.
inline uint8_t ascii_class(uint8_t c) {
    static uint8_t table[128] = {0};
    static bool init = [] {
        for (uint8_t c = 'a'; c <= 'z'; ++c) table[c] |= 1;
        for (uint8_t c = 'A'; c <= 'Z'; ++c) table[c] |= 1;
        for (uint8_t c = '0'; c <= '9'; ++c) table[c] |= 2;
        for (uint8_t c = 0x09; c <= 0x0d; ++c) table[c] |= 4;
        table[0x20] |= 4;
        return true;
    }();
    (void)init;
    return table[c];
}

inline Cls classify(uint32_t cp) {
    if (cp < 128) {
        uint8_t t = ascii_class((uint8_t)cp);
        if (t & 1) return CLS_L;
        if (t & 2) return CLS_N;
        if (t & 4) return CLS_WS;
        return CLS_OTHER;
    }
    int cat = utf8proc_category(cp);
    if (cat >= 1 && cat <= 5) return CLS_L;   // L*
    if (cat >= 9 && cat <= 11) return CLS_N;  // N*
    if (is_ws_cp(cp)) return CLS_WS;
    return CLS_OTHER;
}

// Decode one UTF-8 codepoint at s[i]; invalid bytes decode as U+FFFD with
// advance 1 (corpus text is valid UTF-8; this path is a safety net).
inline uint32_t decode(const char* s, size_t len, size_t i, size_t& adv) {
    uint8_t b = (uint8_t)s[i];
    if (b < 0x80) {
        adv = 1;
        return b;
    }
    uint32_t cp;
    size_t n;
    if ((b & 0xe0) == 0xc0) {
        cp = b & 0x1f;
        n = 2;
    } else if ((b & 0xf0) == 0xe0) {
        cp = b & 0x0f;
        n = 3;
    } else if ((b & 0xf8) == 0xf0) {
        cp = b & 0x07;
        n = 4;
    } else {
        adv = 1;
        return 0xfffd;
    }
    if (i + n > len) {
        adv = 1;
        return 0xfffd;
    }
    for (size_t k = 1; k < n; ++k) {
        uint8_t cb = (uint8_t)s[i + k];
        if ((cb & 0xc0) != 0x80) {
            adv = 1;
            return 0xfffd;
        }
        cp = (cp << 6) | (cb & 0x3f);
    }
    adv = n;
    return cp;
}

inline bool ci_equal(char a, char b) {  // ASCII case-insensitive
    if (a >= 'A' && a <= 'Z') a += 32;
    if (b >= 'A' && b <= 'Z') b += 32;
    return a == b;
}

// Consume a maximal run of class CLS_L starting at i; returns end offset.
inline size_t take_letters(const char* s, size_t len, size_t i) {
    size_t e = i;
    while (e < len) {
        size_t adv;
        uint32_t cp = decode(s, len, e, adv);
        if (classify(cp) != CLS_L) break;
        e += adv;
    }
    return e;
}

// Returns the end offset of the match starting at i (i < len guaranteed).
// Alternatives in pattern order:
//   A1 (?i:'s|'t|'re|'ve|'m|'ll|'d)   contraction suffixes
//   A2 [^\r\n\p{L}\p{N}]?\p{L}+       optional prefix char + letters
//   A3 \p{N}                          single number char
//   A4 ' '?[^\s\p{L}\p{N}]+[\r\n]*    optional space + symbol run + newlines
//   A5 \s*[\r\n]+                     whitespace run ending in newline(s)
//   A6 \s+(?!\S)                      whitespace run (see note)
//   A7 \s+                            whitespace run
inline size_t match_at(const char* s, size_t len, size_t i) {
    // A1: contractions (ordered alternation; all two-after-quote chars are
    // distinct so order only matters for length).
    if (s[i] == '\'') {
        static const char* alts[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        for (const char* a : alts) {
            size_t n = strlen(a);
            if (i + n <= len) {
                bool ok = true;
                for (size_t k = 0; k < n; ++k)
                    if (!ci_equal(s[i + k], a[k])) {
                        ok = false;
                        break;
                    }
                if (ok) return i + n;
            }
        }
    }

    size_t adv0;
    uint32_t cp0 = decode(s, len, i, adv0);
    Cls c0 = classify(cp0);

    // A2: [^\r\n\p{L}\p{N}]?\p{L}+
    if (c0 == CLS_L) {
        return take_letters(s, len, i);
    }
    if (c0 != CLS_N && cp0 != '\r' && cp0 != '\n') {
        // c0 may serve as the optional prefix char (punct, symbol, non-CRLF
        // whitespace), but only if letters follow.
        size_t j = i + adv0;
        if (j < len) {
            size_t adv;
            uint32_t cp = decode(s, len, j, adv);
            if (classify(cp) == CLS_L) return take_letters(s, len, j);
        }
        // A2 fails; fall through (A3 impossible, A4 may apply for punct).
    }

    // A3: single number char
    if (c0 == CLS_N) return i + adv0;

    // A4: optional literal space, then 1+ chars that are not
    // (whitespace/letter/number), then trailing CR/LFs.
    {
        size_t j = i;
        if (j < len && s[j] == ' ') ++j;
        size_t e = j;
        while (e < len) {
            size_t adv;
            uint32_t cp = decode(s, len, e, adv);
            if (classify(cp) != CLS_OTHER) break;
            e += adv;
        }
        if (e > j) {
            while (e < len && (s[e] == '\r' || s[e] == '\n')) ++e;
            return e;
        }
        // A4 fails (space not followed by symbols); fall through.
    }

    // A5-A7: whitespace run [i, e)
    if (c0 == CLS_WS) {
        size_t e = i, last = i;  // last = start offset of the final whitespace char
        while (e < len) {
            size_t adv;
            uint32_t cp = decode(s, len, e, adv);
            if (classify(cp) != CLS_WS) break;
            last = e;
            e += adv;
        }
        // A5: \s*[\r\n]+ — greedy \s* backtracks to the LAST \r/\n in the run;
        // [\r\n]+ then covers just that char (the rest of the run after it is
        // non-newline whitespace by construction).
        for (size_t q = e; q-- > i;)
            if (s[q] == '\r' || s[q] == '\n') return q + 1;
        // A6: \s+(?!\S) — greedy \s+ with the lookahead satisfied whenever the
        // next char is whitespace or EOS; followed by non-whitespace it
        // backtracks ONE CHAR (all-but-the-last whitespace char; needs a run
        // of >= 2 chars).
        if (e == len) return e;
        if (last > i) return last;
        // A7: \s+
        return e;
    }

    return i + adv0;  // unreachable: CLS_OTHER is consumed by A4
}

// Split a document into regex words (views into `text`).
inline void split_words(const char* s, size_t len, std::vector<std::string_view>& out) {
    out.clear();
    size_t i = 0;
    while (i < len) {
        size_t e = match_at(s, len, i);
        out.emplace_back(s + i, e - i);
        i = e;
    }
}

}  // namespace splitter
