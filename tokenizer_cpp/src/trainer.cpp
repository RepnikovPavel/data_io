#include "trainer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "byte_level.h"
#include "checkpoint.h"

namespace trainer {

namespace {

constexpr int32_t MIN_FREQUENCY = 2;  // same as the reference BpeTrainer

// Set by the SIGINT/SIGTERM handler; checked once per merge.
volatile std::sig_atomic_t g_interrupted = 0;
void on_signal(int) { g_interrupted = 1; }

// Run f(begin..end) split over nthreads contiguous chunks.
template <typename F>
void parallel_chunks(size_t begin, size_t end, unsigned nthreads, F&& f) {
    if (nthreads == 0) nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 1;
    size_t n = end - begin;
    if (n == 0) return;
    nthreads = (unsigned)std::min<size_t>(nthreads, n);
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    size_t chunk = (n + nthreads - 1) / nthreads;
    for (unsigned t = 0; t < nthreads; ++t) {
        size_t lo = begin + t * chunk;
        size_t hi = std::min(end, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&, lo, hi] { f(lo, hi); });
    }
    for (auto& th : pool) th.join();
}

// Merge all (non-overlapping, left-to-right) occurrences of (a, b) in toks.
// Same pass structure as the reference's Word::merge.
void merge_pair_in_place(std::vector<uint32_t>& toks, uint32_t a, uint32_t b, uint32_t new_id) {
    std::vector<uint32_t> out;
    out.reserve(toks.size());
    size_t i = 0;
    while (i < toks.size()) {
        if (i + 1 < toks.size() && toks[i] == a && toks[i + 1] == b) {
            out.push_back(new_id);
            i += 2;
        } else {
            out.push_back(toks[i]);
            i += 1;
        }
    }
    toks = std::move(out);
}

// Replay recorded merges on one word: repeatedly merge the lowest-rank
// adjacent pair (all occurrences at once). Equivalent to applying the merges
// in order, because a merge can only affect a word when its pair is adjacent
// in that word, and ranks are processed low-to-high either way.
void replay_merges_on_word(
    std::vector<uint32_t>& toks,
    const std::vector<std::pair<uint32_t, uint32_t>>& merges,
    const std::unordered_map<PairKey, uint32_t>& ranks,
    const std::vector<std::string>& vocab_bytes,
    const std::unordered_map<std::string, uint32_t>& content_to_id) {
    for (;;) {
        uint32_t best_rank = UINT32_MAX;
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            auto it = ranks.find(pair_key(toks[i], toks[i + 1]));
            if (it != ranks.end() && it->second < best_rank) best_rank = it->second;
        }
        if (best_rank == UINT32_MAX) break;
        auto [a, b] = merges[best_rank];
        std::string content = vocab_bytes[a] + vocab_bytes[b];
        uint32_t new_id = content_to_id.at(content);
        merge_pair_in_place(toks, a, b, new_id);
    }
}

// Initial pair counts (parallel per-chunk maps, merged with wrapping i32
// adds; counts are commutative so the result is thread-count independent).
void count_pairs(TrainState& st, unsigned threads) {
    st.pair_counts.clear();
    st.where_pair.clear();
    const size_t n_words = st.toks.size();
    unsigned nthreads = threads ? threads : std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 1;
    std::vector<std::unordered_map<PairKey, int32_t>> pc_parts(nthreads);
    std::vector<std::unordered_map<PairKey, std::vector<uint32_t>>> wp_parts(nthreads);
    std::vector<std::thread> pool;
    size_t chunk = (n_words + nthreads - 1) / nthreads;
    for (unsigned t = 0; t < nthreads; ++t) {
        size_t lo = t * chunk, hi = std::min(n_words, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&, t, lo, hi] {
            auto& pc = pc_parts[t];
            auto& wp = wp_parts[t];
            for (size_t i = lo; i < hi; ++i) {
                const auto& toks = st.toks[i];
                int32_t cnt = count_as_i32(st.counts[i]);
                for (size_t j = 0; j + 1 < toks.size(); ++j) {
                    PairKey p = pair_key(toks[j], toks[j + 1]);
                    auto [it, _] = pc.try_emplace(p, 0);
                    it->second = wrap_add(it->second, cnt);
                    wp[p].push_back((uint32_t)i);
                }
            }
        });
    }
    for (auto& th : pool) th.join();
    for (auto& part : pc_parts) {
        for (auto& [p, c] : part) {
            auto [it, _] = st.pair_counts.try_emplace(p, 0);
            it->second = wrap_add(it->second, c);
        }
    }
    for (auto& part : wp_parts)
        for (auto& [p, v] : part) {
            auto& dst = st.where_pair[p];
            dst.insert(dst.end(), v.begin(), v.end());
        }
}

}  // namespace

TrainState init_state(const std::vector<std::pair<std::string, uint64_t>>& words,
                      const std::vector<std::string>& special_tokens,
                      const TrainOptions& opts) {
    TrainState st;
    const auto table = bytelevel::bytes_to_unicode_table();
    const size_t n_words = words.size();
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // Alphabet: byte-level chars present in the corpus (a byte is present iff
    // its ByteLevel char is), sorted by char codepoint — exactly the reference
    // trainer's compute_alphabet. Parallel present-byte scan.
    std::atomic<uint64_t> present_bits[4] = {};  // 256-bit bitmap
    parallel_chunks(0, n_words, opts.threads, [&](size_t lo, size_t hi) {
        uint64_t bits[4] = {};
        for (size_t i = lo; i < hi; ++i)
            for (unsigned char b : words[i].first) bits[b >> 6] |= 1ull << (b & 63);
        for (int k = 0; k < 4; ++k)
            if (bits[k]) present_bits[k].fetch_or(bits[k], std::memory_order_relaxed);
    });
    std::vector<uint8_t> alphabet;
    for (uint32_t b = 0; b < 256; ++b)
        if (present_bits[b >> 6] & (1ull << (b & 63))) alphabet.push_back((uint8_t)b);
    std::sort(alphabet.begin(), alphabet.end(),
              [&](uint8_t x, uint8_t y) { return table[x] < table[y]; });

    // Base vocab: specials (ids 0..n_specials-1) + present byte-level entries.
    st.vocab_bytes.reserve(std::max<size_t>(opts.vocab_size, 1024));
    st.content_to_id.reserve(opts.vocab_size * 2);
    for (const auto& tok : special_tokens) {
        st.content_to_id.emplace(tok, (uint32_t)st.vocab_bytes.size());
        st.vocab_bytes.push_back(tok);
    }
    std::vector<uint32_t> byte_to_id(256, 0);
    for (uint8_t b : alphabet) {
        uint32_t id = (uint32_t)st.vocab_bytes.size();
        byte_to_id[b] = id;
        st.content_to_id.emplace(std::string(1, (char)b), id);
        st.vocab_bytes.push_back(std::string(1, (char)b));
    }
    const size_t n_initial = st.vocab_bytes.size();
    printf("[train] initial vocab: %zu specials + %zu corpus-present byte-level entries = %zu\n",
           special_tokens.size(), alphabet.size(), n_initial);

    // Tokenize words to initial token id sequences (parallel).
    st.toks.resize(n_words);
    st.counts.resize(n_words);
    parallel_chunks(0, n_words, opts.threads, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
            const std::string& w = words[i].first;
            std::vector<uint32_t> t;
            t.reserve(w.size());
            for (unsigned char b : w) t.push_back(byte_to_id[b]);
            st.toks[i] = std::move(t);
            st.counts[i] = words[i].second;
        }
    });

    // Resume from the latest merges checkpoint if present.
    //
    // Queue-eligibility subtlety: the reference only ever queues a pair if it
    // had a positive (i32) count at init or received a positive change from
    // some merge. Pairs of two INITIAL tokens can never be created by a merge
    // (positive changes always involve the new token), so a pair that was
    // non-positive at init (i32 overflow) stays out of the queue forever. To
    // resume identically, we compute the pre-replay pair counts and treat as
    // eligible: pairs positive at init, plus any pair involving a merged
    // token (id >= n_initial).
    std::unordered_set<PairKey> eligible_at_init;
    auto latest = checkpoint::latest_merges_checkpoint(opts.checkpoint_dir);
    if (latest.found) {
        count_pairs(st, opts.threads);
        for (const auto& [p, c] : st.pair_counts)
            if (c > 0) eligible_at_init.insert(p);

        auto ckpt = checkpoint::read_merges(latest.path);
        if (ckpt.vocab_bytes.size() < n_initial ||
            !std::equal(ckpt.vocab_bytes.begin(), ckpt.vocab_bytes.begin() + n_initial,
                        st.vocab_bytes.begin()))
            throw std::runtime_error(
                latest.path + " does not match the current special tokens / alphabet -- "
                "cannot resume");
        if (ckpt.merges.size() != latest.n)
            throw std::runtime_error(latest.path + ": file name N does not match contents");
        st.vocab_bytes = std::move(ckpt.vocab_bytes);
        st.content_to_id.clear();
        st.content_to_id.reserve(st.vocab_bytes.size() * 2);
        for (uint32_t i = 0; i < st.vocab_bytes.size(); ++i)
            st.content_to_id.emplace(st.vocab_bytes[i], i);
        st.merges = std::move(ckpt.merges);
        printf("[train] resuming from %s: %zu merges, vocab %zu\n", latest.path.c_str(),
               st.merges.size(), st.vocab_bytes.size());

        std::unordered_map<PairKey, uint32_t> ranks;
        ranks.reserve(st.merges.size() * 2);
        for (uint32_t i = 0; i < st.merges.size(); ++i)
            ranks.emplace(pair_key(st.merges[i].first, st.merges[i].second), i);
        parallel_chunks(0, n_words, opts.threads, [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i)
                replay_merges_on_word(st.toks[i], st.merges, ranks, st.vocab_bytes,
                                      st.content_to_id);
        });
        printf("[train] replayed %zu merges on %zu words in %.1fs\n", st.merges.size(), n_words,
               elapsed());
    } else {
        printf("[train] no merges checkpoint found, starting fresh\n");
    }

    // Pair counts over the current (possibly replayed) word token sequences.
    {
        const auto t1 = std::chrono::steady_clock::now();
        count_pairs(st, opts.threads);
        printf("[train] %zu distinct pairs counted in %.1fs\n", st.pair_counts.size(),
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
    }

    // The reference queues only pairs with a positive (i32) count; the queued
    // value is `count as u64` (sign-extended).
    for (const auto& [p, c] : st.pair_counts) {
        if (c <= 0) continue;
        if (latest.found) {
            // Resume: keep the run's admission rule (see above).
            uint32_t a = (uint32_t)(p >> 32), b = (uint32_t)p;
            bool involves_merged = a >= n_initial || b >= n_initial;
            if (!involves_merged && !eligible_at_init.count(p)) continue;
        }
        st.heap.push(QEntry{queue_count(c), (uint32_t)(p >> 32), (uint32_t)p});
    }

    return st;
}

bool run_merge_loop(TrainState& st, const TrainOptions& opts) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // Initial vocab size = current vocab minus merges applied (display only;
    // content collisions can make this off by a little).
    const size_t n_initial = st.vocab_bytes.size() - st.merges.size();
    const auto t_start = std::chrono::steady_clock::now();
    auto t_window = t_start;
    size_t window_start_merges = st.merges.size();
    size_t next_checkpoint = opts.checkpoint_interval == 0
                                 ? SIZE_MAX
                                 : (st.merges.size() / opts.checkpoint_interval + 1) *
                                       opts.checkpoint_interval;

    printf("[train] starting BPE merge loop: vocab %zu -> %zu, %zu merges already done\n",
           st.vocab_bytes.size(), opts.vocab_size, st.merges.size());

    auto write_checkpoint = [&] {
        std::string path =
            opts.checkpoint_dir + "/merges_" + std::to_string(st.merges.size()) + ".bin";
        checkpoint::write_merges(path, st.vocab_bytes, st.merges);
        return path;
    };

    while (true) {
        if (st.vocab_bytes.size() >= opts.vocab_size) break;
        if (opts.max_merges && st.merges.size() >= opts.max_merges) {
            std::string path = write_checkpoint();
            printf("[train] --max-merges %zu reached, wrote %s and stopping without "
                   "tokenizer.json\n",
                   opts.max_merges, path.c_str());
            return false;
        }
        if (g_interrupted) {
            std::string path = write_checkpoint();
            printf("[train] interrupted, wrote %s (%zu merges) -- re-run to resume\n",
                   path.c_str(), st.merges.size());
            fflush(stdout);
            _Exit(2);
        }

        // Pop until a non-stale entry; re-push stale ones with the current
        // (sign-extended) count, like the reference's lazy heap fix-up.
        QEntry top{};
        bool have_top = false;
        while (!st.heap.empty()) {
            QEntry e = st.heap.top();
            st.heap.pop();
            int32_t cur = 0;
            auto it = st.pair_counts.find(pair_key(e.a, e.b));
            if (it != st.pair_counts.end()) cur = it->second;
            if (queue_count(cur) == e.count) {
                top = e;
                have_top = true;
                break;
            }
            st.heap.push(QEntry{queue_count(cur), e.a, e.b});
        }
        if (!have_top) {
            printf("[train] no more pairs to merge\n");
            break;
        }
        // Reference stop condition on the (u64-cast) count.
        if (top.count < 1 || (uint64_t)MIN_FREQUENCY > top.count) {
            printf("[train] best pair count %" PRIu64 " < min_frequency %d, stopping\n",
                   top.count, MIN_FREQUENCY);
            break;
        }

        const uint32_t a = top.a, b = top.b;
        // New token content; on collision with an existing token (e.g. a
        // special token string) reuse its id, like the reference.
        std::string content = st.vocab_bytes[a] + st.vocab_bytes[b];
        uint32_t new_id;
        if (auto it = st.content_to_id.find(content); it != st.content_to_id.end()) {
            new_id = it->second;
        } else {
            new_id = (uint32_t)st.vocab_bytes.size();
            st.content_to_id.emplace(content, new_id);
            st.vocab_bytes.push_back(std::move(content));
        }
        st.merges.emplace_back(a, b);

        // Apply the merge to every word containing (a, b), with per-site
        // count changes exactly like the reference's Word::merge: per site,
        // -1 for the two outer pairs and +1 for the two new outer pairs;
        // the merged pair's own count is never decremented.
        std::unordered_set<PairKey> positive;  // pairs with at least one +1 change
        {
            auto node = st.where_pair.find(pair_key(a, b));
            std::vector<uint32_t> idxs =
                node != st.where_pair.end() ? std::move(node->second) : std::vector<uint32_t>();
            if (node != st.where_pair.end()) st.where_pair.erase(node);
            std::sort(idxs.begin(), idxs.end());
            idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
            for (uint32_t widx : idxs) {
                auto& toks = st.toks[widx];
                int32_t cnt = count_as_i32(st.counts[widx]);
                // Re-validate: the word index may be stale.
                bool has = false;
                for (size_t i = 0; i + 1 < toks.size(); ++i)
                    if (toks[i] == a && toks[i + 1] == b) {
                        has = true;
                        break;
                    }
                if (!has) continue;

                std::vector<uint32_t> out;
                out.reserve(toks.size());
                size_t i = 0;
                while (i < toks.size()) {
                    if (i + 1 < toks.size() && toks[i] == a && toks[i + 1] == b) {
                        if (!out.empty()) {
                            PairKey p1 = pair_key(out.back(), a);
                            PairKey p2 = pair_key(out.back(), new_id);
                            auto e1 = st.pair_counts.try_emplace(p1, 0).first;
                            e1->second = wrap_add(e1->second, wrap_mul(-1, cnt));
                            auto e2 = st.pair_counts.try_emplace(p2, 0).first;
                            e2->second = wrap_add(e2->second, cnt);
                            st.where_pair[p2].push_back(widx);
                            positive.insert(p2);
                        }
                        if (i + 2 < toks.size()) {
                            PairKey p3 = pair_key(b, toks[i + 2]);
                            PairKey p4 = pair_key(new_id, toks[i + 2]);
                            auto e3 = st.pair_counts.try_emplace(p3, 0).first;
                            e3->second = wrap_add(e3->second, wrap_mul(-1, cnt));
                            auto e4 = st.pair_counts.try_emplace(p4, 0).first;
                            e4->second = wrap_add(e4->second, cnt);
                            st.where_pair[p4].push_back(widx);
                            positive.insert(p4);
                        }
                        out.push_back(new_id);
                        i += 2;
                    } else {
                        out.push_back(toks[i]);
                        i += 1;
                    }
                }
                toks = std::move(out);
            }
        }
        // Fresh queue entries only for pairs with a positive change AND a
        // positive current count — the reference's `if count > 0` on the
        // drained update set.
        for (PairKey p : positive) {
            int32_t c = st.pair_counts[p];
            if (c > 0)
                st.heap.push(QEntry{queue_count(c), (uint32_t)(p >> 32), (uint32_t)p});
        }

        size_t done = st.merges.size();
        if (done % 25 == 0 || done == 1) {
            double window_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_window)
                    .count();
            double rate = (done - window_start_merges) / std::max(window_s, 1e-9);
            size_t remaining =
                opts.vocab_size > st.vocab_bytes.size() ? opts.vocab_size - st.vocab_bytes.size() : 0;
            double eta_min = remaining / std::max(rate, 1e-9) / 60.0;
            printf("[train] merge %zu/%zu pair=(%s, %s) count=%" PRIu64
                   " %.1f merges/s ETA %.1f min\n",
                   done, n_initial + (opts.vocab_size > n_initial ? opts.vocab_size - n_initial : 0),
                   bytelevel::show_bytes(st.vocab_bytes[a]).c_str(),
                   bytelevel::show_bytes(st.vocab_bytes[b]).c_str(), top.count, rate, eta_min);
            t_window = std::chrono::steady_clock::now();
            window_start_merges = done;
        }

        if (done >= next_checkpoint) {
            std::string path = write_checkpoint();
            printf("[train] checkpoint %s\n", path.c_str());
            next_checkpoint += opts.checkpoint_interval;
        }
    }

    double mins = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                      .count() /
                  60.0;
    printf("[train] merge loop done: %zu merges, vocab %zu in %.1f min\n", st.merges.size(),
           st.vocab_bytes.size(), mins);
    return true;
}

}  // namespace trainer
