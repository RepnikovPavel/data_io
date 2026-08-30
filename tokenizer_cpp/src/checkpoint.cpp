#include "checkpoint.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace checkpoint {

namespace {
constexpr uint32_t WORDS_MAGIC = 0x57424931;   // "WBI1"
constexpr uint32_t MERGES_MAGIC = 0x4d424931;  // "MBI1"

struct Reader {
    std::ifstream f;
    explicit Reader(const std::string& path) : f(path, std::ios::binary) {
        if (!f) throw std::runtime_error("cannot open " + path);
    }
    template <typename T>
    T read() {
        T v;
        f.read(reinterpret_cast<char*>(&v), sizeof v);
        if (!f) throw std::runtime_error("short read");
        return v;  // file is little-endian; x86/ARM hosts are too
    }
    std::string read_bytes(uint32_t len) {
        std::string s(len, '\0');
        f.read(s.data(), len);
        if (!f) throw std::runtime_error("short read");
        return s;
    }
};

struct Writer {
    std::ofstream f;
    explicit Writer(const std::string& path) : f(path, std::ios::binary | std::ios::trunc) {
        if (!f) throw std::runtime_error("cannot create " + path);
    }
    template <typename T>
    void write(T v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof v);
    }
    void write_bytes(const std::string& s) { f.write(s.data(), (std::streamsize)s.size()); }
};
}  // namespace

WordsFile read_words(const std::string& path) {
    Reader r(path);
    if (r.read<uint32_t>() != WORDS_MAGIC)
        throw std::runtime_error(path + ": bad words.bin magic");
    WordsFile out;
    out.docs_total = r.read<uint64_t>();
    out.bytes_total = r.read<uint64_t>();
    uint64_t n_words = r.read<uint64_t>();
    out.words.reserve(n_words);
    for (uint64_t i = 0; i < n_words; ++i) {
        uint32_t len = r.read<uint32_t>();
        std::string word = r.read_bytes(len);
        uint64_t count = r.read<uint64_t>();
        out.words.emplace_back(std::move(word), count);
    }
    return out;
}

void write_merges(const std::string& path, const std::vector<std::string>& vocab_bytes,
                  const std::vector<std::pair<uint32_t, uint32_t>>& merges) {
    // Atomic write: tmp file + rename (a crash mid-write never leaves a
    // truncated checkpoint under the final name).
    std::string tmp = path + ".tmp";
    {
        Writer w(tmp);
        w.write(MERGES_MAGIC);
        w.write((uint64_t)vocab_bytes.size());
        for (const auto& tok : vocab_bytes) {
            w.write((uint32_t)tok.size());
            w.write_bytes(tok);
        }
        w.write((uint64_t)merges.size());
        for (auto [a, b] : merges) {
            w.write(a);
            w.write(b);
        }
        if (!w.f) throw std::runtime_error("write failed: " + tmp);
    }
    std::filesystem::rename(tmp, path);
}

MergesFile read_merges(const std::string& path) {
    Reader r(path);
    if (r.read<uint32_t>() != MERGES_MAGIC)
        throw std::runtime_error(path + ": bad merges checkpoint magic");
    MergesFile out;
    uint64_t n_vocab = r.read<uint64_t>();
    out.vocab_bytes.reserve(n_vocab);
    for (uint64_t i = 0; i < n_vocab; ++i) {
        uint32_t len = r.read<uint32_t>();
        out.vocab_bytes.push_back(r.read_bytes(len));
    }
    uint64_t n_merges = r.read<uint64_t>();
    out.merges.reserve(n_merges);
    for (uint64_t i = 0; i < n_merges; ++i) {
        uint32_t a = r.read<uint32_t>();
        uint32_t b = r.read<uint32_t>();
        out.merges.emplace_back(a, b);
    }
    return out;
}

LatestMerges latest_merges_checkpoint(const std::string& dir) {
    LatestMerges best;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return best;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("merges_", 0) != 0) continue;
        const std::string stem = name.substr(7);
        if (stem.size() < 5 || stem.compare(stem.size() - 4, 4, ".bin") != 0) continue;
        const std::string num = stem.substr(0, stem.size() - 4);
        if (num.empty() || num.find_first_not_of("0123456789") != std::string::npos) continue;
        size_t n = std::stoull(num);
        if (!best.found || n > best.n) {
            best.found = true;
            best.n = n;
            best.path = entry.path().string();
        }
    }
    return best;
}

}  // namespace checkpoint
