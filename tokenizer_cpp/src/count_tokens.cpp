// count_tokens — count tokens over the whole transformed corpus with a
// trained tokenizer.json, using the C++ encoder (tokenizer_cpp/src/encoder.*).
//
// Input: .tokbin files (see scripts/stage_corpus_tokbin.py):
//   u32 magic "TKB1", u64 n_docs, u64 src_size, u64 src_mtime,
//   then n_docs x { u32 len, u8 bytes[len] }   (instruction and response are
//   separate docs; no truncation, no sampling limits)
//
// Usage:
//   count_tokens --tokenizer <tokenizer.json> -o <out.tsv> [--threads N] <root-or-file>...
//   count_tokens --tokenizer <tokenizer.json> --per-doc <file.tokbin>
//       (self-test mode: print one token count per doc to stdout)
//
// Output: one TSV line per input file `relpath<TAB>token_count` (also written
// to --out), then per-dataset and grand totals on stdout. Progress line every
// 15s: files done/total, docs, GiB, GiB/s, ETA.
//
// Parallelism: files are streamed in batches; each batch is split into
// per-thread doc ranges, so a single huge file cannot tail (chunk-level
// parallelism). Peak memory is bounded by the batch size plus the per-thread
// word caches — far below 50G for this corpus.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "encoder.h"

namespace {

constexpr uint32_t TOKBIN_MAGIC = 0x314b4254;  // "TKB1" little-endian
constexpr size_t BATCH_TARGET_BYTES = 64ull << 20;  // ~64 MiB of text per batch
constexpr size_t BATCH_MAX_DOCS = 1 << 18;

struct Progress {
    std::atomic<uint64_t> docs{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> files_done{0};
    std::atomic<uint64_t> total_files{0};
    std::atomic<bool> stop{false};
};

void progress_thread(Progress* p, std::chrono::steady_clock::time_point t0) {
    while (!p->stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(15));
        double mins = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 60.0;
        uint64_t done = p->files_done.load(), total = p->total_files.load();
        uint64_t docs = p->docs.load(), bytes = p->bytes.load();
        double gib = bytes / 1073741824.0;
        double eta = done ? mins * (total - done) / done : -1.0;
        printf("[count] %llu/%llu files, %llu docs, %.2f GiB, elapsed %.1f min, ETA %.1f min\n",
               (unsigned long long)done, (unsigned long long)total, (unsigned long long)docs, gib,
               mins, eta);
        fflush(stdout);
    }
}

// Process one .tokbin file: mmap it (zero-copy), walk docs in batches,
// count each batch in parallel, return (docs, tokens).
// If per_doc is set, print one count per line per doc (self-test mode).
std::pair<uint64_t, uint64_t> count_file(const std::string& path,
                                         const encoder::Encoder& proto, unsigned nthreads,
                                         Progress& prog, bool per_doc = false) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) throw std::runtime_error("cannot stat " + path);
    size_t fsize = (size_t)st.st_size;
    const char* base = (const char*)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
    madvise((void*)base, fsize, MADV_WILLNEED);  // prefetch; harmless if ignored

    auto rd = [&](size_t off) -> uint64_t {
        uint64_t v;
        memcpy(&v, base + off, sizeof v);
        return v;
    };
    uint32_t magic;
    memcpy(&magic, base, 4);
    if (magic != TOKBIN_MAGIC) throw std::runtime_error("bad magic: " + path);
    uint64_t n_docs = rd(4);
    uint64_t docs_left = n_docs;
    size_t pos = 4 + 8 + 8 + 8;  // magic + n_docs + src_size + src_mtime

    // Per-thread encoders (share read-only tables, own word cache).
    std::vector<encoder::Encoder> encs;
    for (unsigned t = 0; t < nthreads; ++t) encs.push_back(proto.clone_shallow());

    uint64_t file_tokens = 0;
    const bool dump_splits = getenv("COUNT_DUMP_SPLITS") != nullptr;
    while (docs_left > 0) {
        // Walk one batch: up to BATCH_MAX_DOCS docs / BATCH_TARGET_BYTES
        // bytes; docs are (offset, len) into the mapped file — zero copy.
        std::vector<std::pair<uint64_t, uint32_t>> docs;
        docs.reserve(std::min<uint64_t>(docs_left, BATCH_MAX_DOCS));
        size_t batch_bytes = 0;
        uint64_t batch_docs = 0;
        while (docs_left > 0 && batch_docs < BATCH_MAX_DOCS && batch_bytes < BATCH_TARGET_BYTES) {
            uint32_t len;
            memcpy(&len, base + pos, 4);
            pos += 4;
            docs.emplace_back(pos, len);
            pos += len;
            batch_bytes += 4 + len;
            ++batch_docs;
            --docs_left;
        }
        prog.docs.fetch_add(batch_docs);
        prog.bytes.fetch_add(batch_bytes);

        if (per_doc || dump_splits) {
            // Self-test/debug mode: sequential, deterministic.
            uint64_t sum = 0;
            for (auto& [off, len] : docs) {
                if (dump_splits) {
                    encs[0].dump_doc(std::string_view(base + off, len), stdout);
                    continue;
                }
                uint64_t n = encs[0].encode_count(std::string_view(base + off, len));
                printf("%llu\n", (unsigned long long)n);
                sum += n;
            }
            file_tokens += sum;
            continue;
        }

        // Split the batch into per-thread doc ranges; count in parallel.
        std::vector<uint64_t> partial(nthreads, 0);
        {
            std::vector<std::thread> pool;
            size_t chunk = (docs.size() + nthreads - 1) / nthreads;
            for (unsigned t = 0; t < nthreads; ++t) {
                size_t lo = t * chunk, hi = std::min(docs.size(), lo + chunk);
                if (lo >= hi) break;
                pool.emplace_back([&, t, lo, hi] {
                    uint64_t sum = 0;
                    for (size_t i = lo; i < hi; ++i)
                        sum += encs[t].encode_count(
                            std::string_view(base + docs[i].first, docs[i].second));
                    partial[t] = sum;
                });
            }
            for (auto& th : pool) th.join();
        }
        for (uint64_t s : partial) file_tokens += s;
    }
    munmap((void*)base, fsize);
    close(fd);
    prog.files_done.fetch_add(1);
    return {n_docs, file_tokens};
}

// Dataset key from the relative tokbin path, matching the registry names in
// scripts/docs/generate_docs.py:
//   data_clustered/<name>/... -> <name>
//   data/Platypus/arb_*.tokbin -> "arb"
//   data/Platypus/<stem>.tokbin / data/<stem>.tokbin -> <stem>
std::string dataset_key(const std::string& rel) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t i = 0; i <= rel.size(); ++i)
        if (i == rel.size() || rel[i] == '/') {
            parts.push_back(rel.substr(start, i - start));
            start = i + 1;
        }
    if (parts.size() >= 2 && parts[0] == "data_clustered") {
        // Registry name in scripts/docs/generate_docs.py is lowercase.
        if (parts[1] == "SYNTH") return "synth";
        return parts[1];
    }
    std::string stem = parts.back();
    if (auto dot = stem.rfind('.'); dot != std::string::npos) stem.resize(dot);
    if (parts.size() >= 3 && parts[0] == "data" && parts[1] == "Platypus" &&
        stem.rfind("arb_", 0) == 0)
        return "arb";
    return stem;
}

// --- throughput benchmark mode ----------------------------------------------
// `count_tokens --tokenizer T --bench <file.tokbin>`: encode docs from the
// file (bucketed by length: ~100 / ~1k / ~10k chars), single-threaded and
// all-threads, and print a tokens/sec + MiB/s table. Encoder warm-up (word
// cache fill) is included, matching real corpus conditions.
struct TokbinDoc { uint64_t off; uint32_t len; };

int run_bench(const std::string& tok_path, const encoder::Encoder& proto, unsigned nthreads) {
    int fd = open(tok_path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + tok_path);
    struct stat st;
    if (fstat(fd, &st) != 0) throw std::runtime_error("cannot stat " + tok_path);
    size_t fsize = (size_t)st.st_size;
    const char* base = (const char*)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + tok_path);
    madvise((void*)base, fsize, MADV_WILLNEED);
    uint64_t n_docs;
    memcpy(&n_docs, base + 4, 8);
    size_t pos = 28;
    std::vector<TokbinDoc> docs;
    docs.reserve(n_docs);
    for (uint64_t i = 0; i < n_docs; ++i) {
        uint32_t len;
        memcpy(&len, base + pos, 4);
        pos += 4;
        docs.push_back({pos, len});
        pos += len;
    }

    // Length buckets: [50,200) ~100 chars, [500,2000) ~1k, [5000,20000) ~10k.
    const struct { const char* name; uint32_t lo, hi; } buckets[] = {
        {"~100 chars", 50, 200}, {"~1k chars", 500, 2000}, {"~10k chars", 5000, 20000},
    };
    printf("\n[bench] %s (%llu docs, %.2f GiB), %u threads\n", tok_path.c_str(),
           (unsigned long long)n_docs, fsize / 1073741824.0, nthreads);
    printf("%-12s %9s %10s %10s | %14s %12s | %14s %12s\n", "bucket", "docs", "MiB", "tokens",
           "1-thread tok/s", "MiB/s", "all-threads tok/s", "MiB/s");
    printf("------------------------------------------------------------------------------------------\n");
    for (const auto& [name, lo, hi] : buckets) {
        // Up to 2000 evenly spaced docs in the length band.
        std::vector<TokbinDoc> sample;
        for (const auto& d : docs)
            if (d.len >= lo && d.len < hi) sample.push_back(d);
        if (sample.size() > 2000) {
            std::vector<TokbinDoc> thinned;
            double step = (double)sample.size() / 2000;
            for (size_t i = 0; i < 2000; ++i) thinned.push_back(sample[(size_t)(i * step)]);
            sample = std::move(thinned);
        }
        if (sample.empty()) {
            printf("%-12s %9s\n", name, "(none)");
            continue;
        }
        uint64_t bytes = 0;
        for (const auto& d : sample) bytes += d.len;

        auto timed = [&](unsigned threads) -> std::pair<double, uint64_t> {
            std::vector<encoder::Encoder> encs;
            for (unsigned t = 0; t < threads; ++t) encs.push_back(proto.clone_shallow());
            std::atomic<uint64_t> tokens{0};
            auto t0 = std::chrono::steady_clock::now();
            if (threads == 1) {
                uint64_t sum = 0;
                for (const auto& d : sample)
                    sum += encs[0].encode_count(std::string_view(base + d.off, d.len));
                tokens = sum;
            } else {
                std::vector<std::thread> pool;
                size_t chunk = (sample.size() + threads - 1) / threads;
                for (unsigned t = 0; t < threads; ++t) {
                    size_t b = t * chunk, e = std::min(sample.size(), b + chunk);
                    if (b >= e) break;
                    pool.emplace_back([&, t, b, e] {
                        uint64_t sum = 0;
                        for (size_t i = b; i < e; ++i)
                            sum += encs[t].encode_count(
                                std::string_view(base + sample[i].off, sample[i].len));
                        tokens.fetch_add(sum);
                    });
                }
                for (auto& th : pool) th.join();
            }
            double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return {secs, tokens.load()};
        };

        auto [s1, t1] = timed(1);
        auto [sn, tn] = timed(nthreads);
        printf("%-12s %9zu %10.1f %10llu | %14.0f %12.1f | %14.0f %12.1f\n", name,
               sample.size(), bytes / 1048576.0, (unsigned long long)t1, t1 / s1,
               bytes / 1048576.0 / s1, tn / sn, bytes / 1048576.0 / sn);
    }
    munmap((void*)base, fsize);
    close(fd);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string tokenizer_path, out_path, bench_path;
    bool per_doc = false;
    unsigned nthreads = std::thread::hardware_concurrency();
    std::vector<std::string> roots;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--tokenizer") tokenizer_path = next("--tokenizer");
        else if (a == "-o" || a == "--out") out_path = next("-o");
        else if (a == "--threads") nthreads = std::stoul(next("--threads"));
        else if (a == "--per-doc") per_doc = true;
        else if (a == "--bench") bench_path = next("--bench");
        else roots.push_back(a);
    }
    if (tokenizer_path.empty() || (roots.empty() && bench_path.empty())) {
        fprintf(stderr, "usage: count_tokens --tokenizer <tokenizer.json> -o <out.tsv> [--threads N] <root-or-file>...\n"
                        "       count_tokens --tokenizer <tokenizer.json> --bench <file.tokbin>\n");
        return 1;
    }
    if (per_doc) nthreads = 1;
    if (nthreads == 0) nthreads = 1;

    try {
        encoder::Encoder proto(tokenizer_path);
        if (!bench_path.empty())
            return run_bench(bench_path, proto, nthreads);
        fprintf(per_doc ? stderr : stdout, "[count] tokenizer loaded from %s\n",
                tokenizer_path.c_str());

        // Collect .tokbin files.
        std::vector<std::string> files;
        for (const auto& root : roots) {
            std::error_code ec;
            if (std::filesystem::is_regular_file(root, ec)) {
                files.push_back(root);
            } else {
                for (const auto& e : std::filesystem::recursive_directory_iterator(root, ec))
                    if (e.is_regular_file() && e.path().extension() == ".tokbin")
                        files.push_back(e.path().string());
            }
        }
        std::sort(files.begin(), files.end());
        fprintf(per_doc ? stderr : stdout, "[count] %zu .tokbin files on %u threads\n",
                files.size(), nthreads);

        Progress prog;
        prog.total_files = files.size();
        const auto t0 = std::chrono::steady_clock::now();
        std::thread monitor;
        if (!per_doc) monitor = std::thread(progress_thread, &prog, t0);

        FILE* out = nullptr;
        if (!out_path.empty()) {
            out = fopen(out_path.c_str(), "w");
            if (!out) throw std::runtime_error("cannot create " + out_path);
        }
        // Longest-file-first so big files start early (less tail).
        std::vector<std::pair<uint64_t, size_t>> by_size;
        for (size_t i = 0; i < files.size(); ++i)
            by_size.emplace_back(std::filesystem::file_size(files[i]), i);
        std::sort(by_size.rbegin(), by_size.rend());

        std::map<std::string, uint64_t> per_dataset;  // dataset -> tokens
        std::map<std::string, uint64_t> per_dataset_docs;
        std::vector<std::string> relpath_of(files.size());

        for (auto& [sz, idx] : by_size) {
            const std::string& path = files[idx];
            // rel path vs the common root (first root arg)
            std::string rel = path;
            std::string root0 = roots[0];
            if (rel.rfind(root0 + "/", 0) == 0) rel = rel.substr(root0.size() + 1);
            relpath_of[idx] = rel;
            auto [n_docs, n_toks] = count_file(path, proto, nthreads, prog, per_doc);
            per_dataset[dataset_key(rel)] += n_toks;
            per_dataset_docs[dataset_key(rel)] += n_docs;
            if (out)
                fprintf(out, "%s\t%llu\t%llu\n", rel.c_str(), (unsigned long long)n_docs,
                        (unsigned long long)n_toks);
        }

        prog.stop.store(true);
        if (monitor.joinable()) monitor.join();
        if (out) fclose(out);
        if (per_doc) return 0;  // self-test mode: no summary on stdout

        double mins = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 60.0;
        uint64_t total_docs = prog.docs.load(), total_bytes = prog.bytes.load();
        printf("\n[count] per-dataset totals (docs / tokens):\n");
        uint64_t grand = 0, grand_docs = 0;
        for (const auto& [name, toks] : per_dataset) {
            printf("  %-24s %14llu docs %18llu tokens\n", name.c_str(),
                   (unsigned long long)per_dataset_docs[name], (unsigned long long)toks);
            grand += toks;
            grand_docs += per_dataset_docs[name];
        }
        printf("[count] done: %zu files, %llu docs, %.2f GiB text, %llu tokens total, "
               "in %.1f min (%.2f GiB/s)\n",
               files.size(), (unsigned long long)total_docs, total_bytes / 1073741824.0,
               (unsigned long long)grand, mins, total_bytes / 1073741824.0 / (mins * 60.0));
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
