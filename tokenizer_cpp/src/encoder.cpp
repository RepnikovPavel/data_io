#include "encoder.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include <utf8proc.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "byte_level.h"
#include "minijson.h"
#include "splitter.h"

namespace encoder {

Tables::~Tables() {
    if (re) pcre2_code_free_8((pcre2_code_8*)re);
}

static const char* kSplitRegex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// Split implementation: default = hand-rolled scanner (splitter.h);
// COUNT_SPLIT=pcre2 forces the PCRE2 regex path (used for differential tests).
// TEMP profiling hooks (COUNT_ABLATE=scan|cache).
static int ablate_mode() {
    static int v = [] {
        const char* e = getenv("COUNT_ABLATE");
        if (!e) return 0;
        if (std::string(e) == "scan") return 1;   // specials+NFC+split, no word counting
        if (std::string(e) == "cache") return 2;  // + byte map, no merge loop
        return 0;
    }();
    return v;
}

static bool use_scanner() {
    static bool v = [] {
        const char* e = getenv("COUNT_SPLIT");
        return !(e && std::string(e) == "pcre2");
    }();
    return v;
}

Encoder::Encoder(const std::string& tokenizer_json_path) {
    auto root = minijson::parse(minijson::read_file(tokenizer_json_path));
    auto t = std::make_shared<Tables>();

    // vocab: byte-level char string -> id
    std::unordered_map<std::string, uint32_t> vocab;
    const auto& model = root.at("model");
    for (const auto& [tok, id] : model.at("vocab").obj) vocab.emplace(tok, (uint32_t)id.as_u64());

    // merges: (a_id, b_id) -> (rank, new_id)
    t->merge_rank.reserve(model.at("merges").arr.size() * 2);
    uint32_t rank = 0;
    for (const auto& m : model.at("merges").arr) {
        const auto& pair = m.arr;
        auto ia = vocab.find(pair[0].str);
        auto ib = vocab.find(pair[1].str);
        if (ia == vocab.end() || ib == vocab.end())
            throw std::runtime_error("merge token out of vocab");
        std::string merged = pair[0].str + pair[1].str;
        auto inew = vocab.find(merged);
        if (inew == vocab.end()) throw std::runtime_error("merge result out of vocab");
        t->merge_rank.emplace(((uint64_t)ia->second << 32) | ib->second,
                              ((uint64_t)rank << 32) | inew->second);
        ++rank;
    }

    // byte-level alphabet: single-char vocab entries -> byte
    const auto b2u = bytelevel::bytes_to_unicode_table();
    t->byte_id.assign(256, UINT16_MAX);
    for (uint32_t b = 0; b < 256; ++b) {
        std::string ch;
        bytelevel::encode_utf8(b2u[b], ch);
        auto it = vocab.find(ch);
        if (it != vocab.end()) {
            if (it->second > 65534)
                throw std::runtime_error("alphabet id unexpectedly high");
            t->byte_id[b] = (uint16_t)it->second;
        }
    }

    // special tokens (added_tokens, ordered by id)
    const auto& added = root.at("added_tokens").arr;
    t->specials.resize(added.size());
    for (const auto& at : added) t->specials[at.at("id").as_u64()] = at.at("content").str;

    // pre-tokenizer regex via PCRE2 (UTF + UCP for \p{L} etc.)
    int errcode = 0;
    PCRE2_SIZE erroffset = 0;
    pcre2_code_8* re = pcre2_compile_8((PCRE2_SPTR8)kSplitRegex, PCRE2_ZERO_TERMINATED,
                                       PCRE2_UTF | PCRE2_UCP | PCRE2_MATCH_INVALID_UTF, &errcode,
                                       &erroffset, nullptr);
    if (!re) {
        PCRE2_UCHAR8 buf[256];
        pcre2_get_error_message_8(errcode, buf, sizeof buf);
        throw std::runtime_error(std::string("PCRE2 compile failed: ") + (char*)buf);
    }
    // JIT-compile the pattern (large speedup on the GPT-2 regex, which is
    // slow in the PCRE2 interpreter). Fall back to the interpreter if JIT
    // is unavailable.
    if (pcre2_jit_compile_8(re, PCRE2_JIT_COMPLETE) != 0)
        fprintf(stderr, "[count] warning: PCRE2 JIT unavailable, using interpreter\n");
    t->re = re;
    t_ = std::move(t);
}

Encoder::Encoder(std::shared_ptr<Tables> tables) : t_(std::move(tables)) {
    md_ = pcre2_match_data_create_from_pattern_8((pcre2_code_8*)t_->re, nullptr);
    if (!md_) throw std::runtime_error("pcre2_match_data_create failed");
}

Encoder::~Encoder() {
    if (md_) pcre2_match_data_free_8((pcre2_match_data_8*)md_);
}

// Steps 3-4 for one regex word (raw bytes): ByteLevel map + greedy rank
// merges. Memoized; the lookup is by string_view (no allocation on hit).
uint64_t Encoder::count_word(std::string_view word) {
    if (uint64_t c = cache_.get(word); c != UINT64_MAX) return c;

    std::vector<uint16_t>& toks = scratch_;  // reused buffer (no per-word alloc)
    toks.clear();
    toks.reserve(word.size());
    for (unsigned char b : word) {
        uint16_t id = t_->byte_id[b];
        if (id != UINT16_MAX) toks.push_back(id);  // unknown byte: dropped (no unk/fallback)
    }
    // Greedy lowest-rank merge loop (standard BPE application).
    while (ablate_mode() < 2 && toks.size() > 1) {  // TEMP ablate
        uint64_t best = UINT64_MAX;  // packed (rank << 32) | new_id
        uint16_t ba = 0, bb = 0;
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            auto it = t_->merge_rank.find(((uint64_t)toks[i] << 32) | toks[i + 1]);
            if (it != t_->merge_rank.end() && it->second < best) {
                best = it->second;
                ba = toks[i];
                bb = toks[i + 1];
            }
        }
        if (best == UINT64_MAX) break;
        uint16_t new_id = (uint16_t)best;
        size_t w = 0;
        for (size_t i = 0; i < toks.size();) {
            if (i + 1 < toks.size() && toks[i] == ba && toks[i + 1] == bb) {
                toks[w++] = new_id;
                i += 2;
            } else {
                toks[w++] = toks[i];
                i += 1;
            }
        }
        toks.resize(w);
    }
    uint64_t n = toks.size();
    cache_.put(word, n);
    return n;
}

// Steps 2-4 for a non-special span, PCRE2 reference path (differential
// testing; COUNT_SPLIT=pcre2): NFC (utf8proc; skipped for pure ASCII) +
// regex split + count.
uint64_t Encoder::count_span_pcre2(const char* data, size_t len) {
    if (len == 0) return 0;
    // NFC fast path: ASCII-only spans are NFC-stable. Word-at-a-time high-bit
    // check (8 bytes per step).
    bool ascii = true;
    size_t k = 0;
    for (; k + 8 <= len; k += 8) {
        uint64_t v;
        memcpy(&v, data + k, 8);
        if (v & 0x8080808080808080ull) {
            ascii = false;
            break;
        }
    }
    for (; ascii && k < len; ++k)
        if ((unsigned char)data[k] >= 0x80) ascii = false;

    utf8proc_uint8_t* nfc = nullptr;
    const char* text = data;
    size_t text_len = len;
    if (!ascii) {
        utf8proc_size_t r = utf8proc_map((const utf8proc_uint8_t*)data, (utf8proc_size_t)len,
                                         &nfc,
                                         (utf8proc_option_t)(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
        if (!nfc || r < 0) throw std::runtime_error("utf8proc NFC failed");
        text = (const char*)nfc;
        text_len = (size_t)r;
    }
    uint64_t total = 0;
    PCRE2_SIZE offset = 0;
    auto* md = (pcre2_match_data_8*)md_;
    auto* re = (pcre2_code_8*)t_->re;
    // PCRE2_NO_UTF_CHECK is safe: input is valid UTF-8 (corpus strings) or
    // utf8proc output (guaranteed valid).
    while (offset <= text_len) {
        int rc = pcre2_match_8(re, (PCRE2_SPTR8)text, text_len, offset, PCRE2_NO_UTF_CHECK, md,
                               nullptr);
        if (rc < 0) break;
        PCRE2_SIZE* ov = pcre2_get_ovector_pointer_8(md);
        if (ov[1] > ov[0]) {
            total += count_word(std::string_view(text + ov[0], ov[1] - ov[0]));
        }
        offset = ov[1] > ov[0] ? ov[1] : ov[1] + 1;
    }
    if (nfc) free(nfc);
    return total;
}

// Fused single pass (scanner mode): specials matched in-place on the raw
// text, split via the hand-rolled scanner, non-ASCII detection folded in
// (word-level: a word containing a byte >= 0x80 triggers an NFC redo of the
// whole span — exact reference behavior since specials are pure ASCII and
// the NFC'd redo re-matches them identically).
uint64_t Encoder::count_span_fused(const char* data, size_t len, bool nfc_done) {
    uint64_t total = 0;
    size_t i = 0;
    while (i < len) {
        char c = data[i];
        if (c == '<') {
            size_t best_len = 0;
            for (const auto& sp : t_->specials)
                if (sp.size() > best_len && len - i >= sp.size() &&
                    memcmp(data + i, sp.data(), sp.size()) == 0)
                    best_len = sp.size();
            if (best_len) {
                total += 1;  // special token, counts as 1
                i += best_len;
                continue;
            }
        }
        if (!nfc_done && (unsigned char)c >= 0x80) return UINT64_MAX;  // signal: needs NFC
        size_t e = splitter::match_at(data, len, i);
        std::string_view w(data + i, e - i);
        if (!nfc_done)
            for (char wc : w)
                if ((unsigned char)wc >= 0x80) return UINT64_MAX;  // needs NFC
        if (ablate_mode() == 1) total += w.size();  // TEMP ablate: skip counting
        else total += count_word(w);
        i = e;
    }
    return total;
}

uint64_t Encoder::count_span(const char* data, size_t len) {
    if (len == 0) return 0;
    uint64_t total = count_span_fused(data, len, false);
    if (total != UINT64_MAX) return total;
    // NFC needed: redo the whole span on the NFC'd text (rare path).
    utf8proc_uint8_t* nfc = nullptr;
    utf8proc_size_t r = utf8proc_map((const utf8proc_uint8_t*)data, (utf8proc_size_t)len, &nfc,
                                     (utf8proc_option_t)(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (!nfc || r < 0) throw std::runtime_error("utf8proc NFC failed");
    total = count_span_fused((const char*)nfc, (size_t)r, true);
    free(nfc);
    return total;
}

uint64_t Encoder::encode_count(std::string_view text) {
    if (use_scanner()) {
        // Default: fused single pass (splitter.h scanner; COUNT_SPLIT=pcre2
        // selects the PCRE2 reference path used for differential tests).
        return count_span(text.data(), text.size());
    }
    // PCRE2 reference path: specials via memchr, then count_span_pcre2.
    uint64_t total = 0;
    size_t span_start = 0;
    size_t i = 0;
    while (i < text.size()) {
        const char* hit = (const char*)memchr(text.data() + i, '<', text.size() - i);
        if (!hit) break;
        i = (size_t)(hit - text.data());
        size_t best_len = 0;
        for (const auto& sp : t_->specials)
            if (sp.size() > best_len && text.compare(i, sp.size(), sp) == 0)
                best_len = sp.size();
        if (best_len) {
            total += count_span_pcre2(text.data() + span_start, i - span_start);
            total += 1;  // the special token itself
            i += best_len;
            span_start = i;
        } else {
            ++i;
        }
    }
    total += count_span_pcre2(text.data() + span_start, text.size() - span_start);
    return total;
}

}  // namespace encoder

// --- debug dump (out-of-line) -----------------------------------------------
#include <cstdio>

namespace encoder {

// Shared split driver for dump_doc: specials + NFC + word split, no counting.
void Encoder::dump_doc(std::string_view text, FILE* out) {
    auto dump_span = [&](const char* data, size_t len) {
        if (len == 0) return;
        bool ascii = true;
        for (size_t i = 0; i < len; ++i)
            if ((unsigned char)data[i] >= 0x80) {
                ascii = false;
                break;
            }
        std::string nfc;
        const char* p = data;
        size_t n = len;
        if (!ascii) {
            utf8proc_uint8_t* r = nullptr;
            utf8proc_size_t rn = utf8proc_map((const utf8proc_uint8_t*)data, (utf8proc_size_t)len,
                                              &r,
                                              (utf8proc_option_t)(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
            if (!r || rn < 0) throw std::runtime_error("utf8proc NFC failed");
            nfc.assign((const char*)r, rn);
            free(r);
            p = nfc.data();
            n = nfc.size();
        }
        size_t i = 0;
        while (i < n) {
            size_t e;
            if (use_scanner()) {
                e = splitter::match_at(p, n, i);
            } else {
                PCRE2_SIZE offset = i;
                int rc = pcre2_match_8((pcre2_code_8*)t_->re, (PCRE2_SPTR8)p, n, offset,
                                       PCRE2_NO_UTF_CHECK, (pcre2_match_data_8*)md_, nullptr);
                if (rc < 0) break;
                PCRE2_SIZE* ov = pcre2_get_ovector_pointer_8((pcre2_match_data_8*)md_);
                e = ov[1] > ov[0] ? ov[1] : ov[1] + 1;
            }
            if (e > i) {
                std::string w(p + i, e - i);
                fprintf(out, "W %s\n", bytelevel::show_bytes(w).c_str());
            }
            i = e;
        }
    };
    size_t span_start = 0, i = 0;
    while (i < text.size()) {
        if (text[i] == '<') {
            size_t best_len = 0;
            for (const auto& sp : t_->specials)
                if (sp.size() > best_len && text.compare(i, sp.size(), sp) == 0)
                    best_len = sp.size();
            if (best_len) {
                dump_span(text.data() + span_start, i - span_start);
                fprintf(out, "S %s\n", std::string(text.substr(i, best_len)).c_str());
                i += best_len;
                span_start = i;
                continue;
            }
        }
        ++i;
    }
    dump_span(text.data() + span_start, text.size() - span_start);
}

}  // namespace encoder
