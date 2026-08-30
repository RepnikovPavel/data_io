//! Iterative, checkpointable BPE tokenizer trainer.
//!
//! Same pipeline and sampling rules as `train_tokenizer.rs` (NFC normalizer,
//! Split+ByteLevel pre-tokenizer, 31 special tokens at ids 0..30, 65536 vocab,
//! min_frequency 2, per-doc 10k-char truncation, per-file sampling from
//! prefix_config.yaml with limit = limit_mul_factor * max_per_file and a
//! Pcg64 seeded by DefaultHasher(seed, safe_name)), but split into two
//! checkpointable phases:
//!
//! * `load`  : stream all input files (rayon parallel), truncate/sample,
//!   NFC-normalize, split into words with the exact same regex the
//!   reference trainer uses, and count word frequencies into a
//!   map. Persisted to `<checkpoint-dir>/words.bin`. Deterministic:
//!   per-thread maps are merged commutatively, so word counts do
//!   not depend on the thread count.
//! * `train` : load `words.bin`, then run the classic BPE merge loop with
//!   incremental pair counting. Checkpoints every
//!   `--checkpoint-interval` merges to
//!   `<checkpoint-dir>/merges_<N>.bin` and resumes from the latest
//!   one. On completion writes `tokenizer.json` (+ `config.json`)
//!   with exactly the same structure as the reference trainer.
//!
//! Words are counted as raw UTF-8 bytes of the NFC-normalized regex words
//! instead of ByteLevel-mapped char strings. The ByteLevel bytes->unicode map
//! is a bijection, so word counts are identical; byte strings are converted to
//! ByteLevel char strings only when assembling the final vocab/merges.
//!
//! Parity with the reference trainer (tokenizers 0.22 `BpeTrainer::do_train`)
//! is exact, bug-for-bug:
//! * initial vocab = special tokens (ids 0..N-1) + the byte-level chars
//!   *present in the corpus* (the reference derives the alphabet from the
//!   word counts, it does NOT use all 256 chars), sorted by char codepoint;
//! * pair counts are i32 and updated with release-mode (wrapping) arithmetic,
//!   exactly like the reference (`AHashMap<Pair, i32>`, `change * count as
//!   i32`). On this corpus several pairs occur more than i32::MAX times and
//!   their counts wrap to negative; the reference only queues pairs with
//!   count > 0, which permanently excludes those pairs from merging. We
//!   replicate this faithfully (including the `count as u64` sign-extension
//!   in queue entries and the stale-entry re-push semantics) rather than
//!   "fixing" it, because the goal is to match the reference artifact;
//! * merge selection = max queue count, ties broken by ascending token-id
//!   pair (the reference's `Merge::cmp`: `count.cmp` then reversed
//!   `pair.cmp`), so merge order and id assignment match merge-for-merge;
//! * min_frequency = 2, merges stop when the best pair's count drops below 2
//!   or the vocab is full; merges whose concatenation already exists in the
//!   vocab (e.g. a special token string) reuse the existing id, as the
//!   reference does.
//!
//! Training complexity: pair counts are maintained incrementally. The initial
//! pass counts all adjacent pairs once, O(total tokens). Each merge only
//! rewrites the words that contain the selected pair (tracked via a
//! pair -> word-indices map, append-only with dedup at use), so one merge
//! costs O(total length of affected words) plus O(#changed distinct pairs)
//! heap pushes. Argmax uses a lazy max-heap; stale heap entries are detected
//! by comparing against the current pair count and re-pushed with the fresh
//! (sign-extended) count, exactly like the reference's queue.

use anyhow::{Context, Error, Result, bail};
use clap::{Parser, ValueEnum};
use rand::seq::SliceRandom;
use rand_pcg::Pcg64;
use rand_pcg::rand_core::SeedableRng;
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use tokenizers::models::bpe::{BpeBuilder, Vocab};
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{
    AddedToken, NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString,
    PreTokenizer, TokenizerBuilder,
};

use tokenizer::{FoundFile, read_any_stream, scan_inputs};

// Optimize memory allocation (same as the reference trainer).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The exact pre-tokenizer regex used by the reference trainer.
const SPLIT_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
/// Same min_frequency as the reference BpeTrainer setup.
const MIN_FREQUENCY: u64 = 2;

const WORDS_MAGIC: u32 = 0x5742_4931; // "WBI1"
const MERGES_MAGIC: u32 = 0x4d42_4931; // "MBI1"

/// Unique words (raw UTF-8 bytes) with their corpus counts.
type WordCounts = Vec<(Vec<u8>, u64)>;
/// A pair of token ids.
type Pair = (u32, u32);
/// Weighted adjacent-pair counts. i32 with wrapping arithmetic, replicating
/// the reference trainer's `AHashMap<Pair, i32>` (which overflows on pairs
/// occurring more than i32::MAX times -- pairs with a wrapped non-positive
/// count are never merged by the reference, and neither by us).
type PairCounts = HashMap<Pair, i32>;
/// pair -> indices of words containing it.
type PairLocations = HashMap<Pair, Vec<u32>>;

#[derive(Parser, Debug)]
#[command(name = "train_tokenizer_iter")]
struct TrainArgs {
    /// Input directories/files to scan for .jsonl / .parquet data
    #[arg(required = true, num_args = 1..)]
    dirs: Vec<PathBuf>,
    /// Output tokenizer.json path
    #[arg(short, long, required = true)]
    out: PathBuf,
    /// prefix_config.yaml with per-file sampling limits
    #[arg(long, required = true)]
    prefix_config: PathBuf,
    /// Directory for words.bin and merges_N.bin checkpoints
    #[arg(long, required = true)]
    checkpoint_dir: PathBuf,

    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 65536)]
    vocab_size: usize,
    #[arg(long, default_value = "bpe")]
    tokenizer_type: String,

    /// Which phase to run: load (scan+count words), train (BPE merges),
    /// or all (load, then train). `all` reuses an existing words.bin.
    #[arg(long, default_value = "all")]
    phase: Phase,

    /// Takes first truncate_len characters of each document.
    #[arg(long, default_value_t = 10_000)]
    truncate_len: usize,
    /// Per-file doc limit = limit_mul_factor * max_per_file.
    #[arg(long, default_value_t = 10)]
    limit_mul_factor: usize,

    /// Write a merges checkpoint every N merges.
    #[arg(long, default_value_t = 100)]
    checkpoint_interval: usize,

    /// Stop after N merges without writing tokenizer.json (for testing
    /// checkpoint/resume). Hidden from normal usage.
    #[arg(long, hide = true)]
    max_merges: Option<usize>,

    /// Special token list (ids 0..N-1, same default as the reference trainer)
    #[arg(long, value_delimiter = ',', default_values = [
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
    "<think>", "</think>"
    ])]
    special_tokens: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Phase {
    All,
    Load,
    Train,
}

#[derive(Debug, Deserialize)]
struct PrefixConfigItem {
    prefix: String,
    max_per_file: Option<usize>,
}

fn truncate_safe(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        // Optimization: len() in bytes >= len() in chars
        return s;
    }
    // Find the byte index of the Nth character
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s, // String has fewer than max_chars characters
    }
}

// ---------------------------------------------------------------------------
// ByteLevel bytes <-> unicode table (GPT-2 style, same as tokenizers ByteLevel)
// ---------------------------------------------------------------------------

fn bytes_to_unicode_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let is_printable =
        |b: u8| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut n: u32 = 0;
    for b in 0..=255u8 {
        table[b as usize] = if is_printable(b) {
            b as char
        } else {
            let c = char::from_u32(256 + n).expect("byte-level char");
            n += 1;
            c
        };
    }
    table
}

/// Alphabet id order for the bytes present in the corpus: sorted by the
/// codepoint of their ByteLevel char. Exactly the reference trainer's
/// `compute_alphabet`: chars seen in the word counts, kept sorted by codepoint
/// for determinism.
fn alphabet_order(table: &[char; 256], present: &[bool; 256]) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..=255).filter(|&b| present[b as usize]).collect();
    bytes.sort_by_key(|&b| table[b as usize] as u32);
    bytes
}

fn bytes_to_bpe_string(bytes: &[u8], table: &[char; 256]) -> String {
    bytes.iter().map(|&b| table[b as usize]).collect()
}

/// Human-readable rendering of raw token bytes for progress logs.
fn show_bytes(bytes: &[u8]) -> String {
    format!("{:?}", String::from_utf8_lossy(bytes))
}

// ---------------------------------------------------------------------------
// Checkpoint formats (std-only, little-endian, length-prefixed)
//
// words.bin:
//   u32 WORDS_MAGIC, u64 docs_total, u64 bytes_total, u64 n_words,
//   then per word: u32 len, len bytes (raw UTF-8 of the regex word), u64 count
//
// merges_N.bin:
//   u32 MERGES_MAGIC, u64 n_vocab,
//   per vocab entry (id order, specials + alphabet + merged): u32 len, bytes,
//   u64 n_merges, per merge: u32 a, u32 b
// ---------------------------------------------------------------------------

fn write_words_checkpoint(
    path: &Path,
    docs_total: u64,
    bytes_total: u64,
    words: &[(Vec<u8>, u64)],
) -> Result<()> {
    let tmp = path.with_extension("bin.tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(&WORDS_MAGIC.to_le_bytes())?;
        w.write_all(&docs_total.to_le_bytes())?;
        w.write_all(&bytes_total.to_le_bytes())?;
        w.write_all(&(words.len() as u64).to_le_bytes())?;
        for (word, count) in words {
            w.write_all(&(word.len() as u32).to_le_bytes())?;
            w.write_all(word)?;
            w.write_all(&count.to_le_bytes())?;
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_words_header(path: &Path) -> Result<(u64, u64, u64)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut u32buf = [0u8; 4];
    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u32buf)?;
    if u32::from_le_bytes(u32buf) != WORDS_MAGIC {
        bail!("{:?}: bad words.bin magic", path);
    }
    let mut next_u64 = |r: &mut BufReader<File>| -> Result<u64> {
        r.read_exact(&mut u64buf)?;
        Ok(u64::from_le_bytes(u64buf))
    };
    let docs = next_u64(&mut r)?;
    let bytes = next_u64(&mut r)?;
    let n_words = next_u64(&mut r)?;
    Ok((docs, bytes, n_words))
}

fn read_words_checkpoint(path: &Path) -> Result<(u64, u64, WordCounts)> {
    let mut r = BufReader::with_capacity(1 << 24, File::open(path)?);
    let mut u32buf = [0u8; 4];
    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u32buf)?;
    if u32::from_le_bytes(u32buf) != WORDS_MAGIC {
        bail!("{:?}: bad words.bin magic", path);
    }
    r.read_exact(&mut u64buf)?;
    let docs_total = u64::from_le_bytes(u64buf);
    r.read_exact(&mut u64buf)?;
    let bytes_total = u64::from_le_bytes(u64buf);
    r.read_exact(&mut u64buf)?;
    let n_words = u64::from_le_bytes(u64buf) as usize;

    let mut words = Vec::with_capacity(n_words);
    for _ in 0..n_words {
        r.read_exact(&mut u32buf)?;
        let len = u32::from_le_bytes(u32buf) as usize;
        let mut word = vec![0u8; len];
        r.read_exact(&mut word)?;
        r.read_exact(&mut u64buf)?;
        words.push((word, u64::from_le_bytes(u64buf)));
    }
    Ok((docs_total, bytes_total, words))
}

fn write_merges_checkpoint(
    path: &Path,
    vocab_bytes: &[Vec<u8>],
    merges: &[(u32, u32)],
) -> Result<()> {
    let tmp = path.with_extension("bin.tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(&MERGES_MAGIC.to_le_bytes())?;
        w.write_all(&(vocab_bytes.len() as u64).to_le_bytes())?;
        for token in vocab_bytes {
            w.write_all(&(token.len() as u32).to_le_bytes())?;
            w.write_all(token)?;
        }
        w.write_all(&(merges.len() as u64).to_le_bytes())?;
        for &(a, b) in merges {
            w.write_all(&a.to_le_bytes())?;
            w.write_all(&b.to_le_bytes())?;
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_merges_checkpoint(path: &Path) -> Result<(Vec<Vec<u8>>, Vec<Pair>)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut u32buf = [0u8; 4];
    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u32buf)?;
    if u32::from_le_bytes(u32buf) != MERGES_MAGIC {
        bail!("{:?}: bad merges checkpoint magic", path);
    }
    r.read_exact(&mut u64buf)?;
    let n_vocab = u64::from_le_bytes(u64buf) as usize;
    let mut vocab_bytes = Vec::with_capacity(n_vocab);
    for _ in 0..n_vocab {
        r.read_exact(&mut u32buf)?;
        let len = u32::from_le_bytes(u32buf) as usize;
        let mut token = vec![0u8; len];
        r.read_exact(&mut token)?;
        vocab_bytes.push(token);
    }
    r.read_exact(&mut u64buf)?;
    let n_merges = u64::from_le_bytes(u64buf) as usize;
    let mut merges = Vec::with_capacity(n_merges);
    for _ in 0..n_merges {
        r.read_exact(&mut u32buf)?;
        let a = u32::from_le_bytes(u32buf);
        r.read_exact(&mut u32buf)?;
        let b = u32::from_le_bytes(u32buf);
        merges.push((a, b));
    }
    Ok((vocab_bytes, merges))
}

/// Find the latest `merges_<N>.bin` in the checkpoint dir.
fn latest_merges_checkpoint(dir: &Path) -> Result<Option<(usize, PathBuf)>> {
    let mut best: Option<(usize, PathBuf)> = None;
    if !dir.is_dir() {
        return Ok(None);
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(rest) = name.strip_prefix("merges_")
            && let Some(num) = rest.strip_suffix(".bin")
            && let Ok(n) = num.parse::<usize>()
            && best.as_ref().is_none_or(|(bn, _)| n > *bn)
        {
            best = Some((n, path));
        }
    }
    Ok(best)
}

// ---------------------------------------------------------------------------
// Phase: load
// ---------------------------------------------------------------------------

/// Fields are read by the unit tests and by callers that want the numbers
/// (run_load already prints them, hence the allow).
#[allow(dead_code)]
struct LoadStats {
    n_files: usize,
    docs: u64,
    bytes: u64,
    unique_words: usize,
    elapsed_secs: f64,
}

/// Read one input file, apply truncation + per-file sampling limit (identical
/// to the reference trainer), then NFC-normalize + regex-split every document
/// and count words (as raw UTF-8 bytes) into `acc`.
fn process_file(
    f: &FoundFile,
    args: &TrainArgs,
    prefix_configs: &[PrefixConfigItem],
    splitter: &Split,
    acc: &mut HashMap<Vec<u8>, u64>,
) -> Result<(usize, usize)> {
    let mut local_docs: Vec<String> = Vec::new();
    read_any_stream(&f.path, |_, inst, resp| {
        local_docs.push(truncate_safe(inst, args.truncate_len).to_string());
        local_docs.push(truncate_safe(resp, args.truncate_len).to_string());
    })?;

    // Find first match of prefix, else use default (no limit)
    let config = prefix_configs.iter().find(|c| f.safe_name.starts_with(&c.prefix));
    if let Some(prefix_limit) = config.and_then(|c| c.max_per_file) {
        // Scale limit by a certain factor for diversity.
        let limit = args.limit_mul_factor * prefix_limit;
        if local_docs.len() > limit {
            // Create a unique seed for THIS specific file
            let mut hasher = DefaultHasher::new();
            args.seed.hash(&mut hasher);
            f.safe_name.hash(&mut hasher);
            let file_specific_seed = hasher.finish();
            // Take first `limit` items
            let mut rng = Pcg64::seed_from_u64(file_specific_seed);
            local_docs.partial_shuffle(&mut rng, limit);
            local_docs.truncate(limit);
        }
    }

    let n_docs = local_docs.len();
    let n_bytes: usize = local_docs.iter().map(|s| s.len()).sum();

    for doc in &local_docs {
        if doc.is_empty() {
            continue;
        }
        // Same normalization + split as the reference trainer's pipeline
        // (NFC normalizer, then the Split pre-tokenizer with Isolated
        // behavior). ByteLevel is a bijection on bytes, so we count raw
        // bytes instead of ByteLevel chars.
        let mut normalized = NormalizedString::from(doc.as_str());
        NFC.normalize(&mut normalized).map_err(Error::msg)?;
        let mut pretokenized = PreTokenizedString::from(normalized);
        splitter.pre_tokenize(&mut pretokenized).map_err(Error::msg)?;
        for (piece, _, _) in pretokenized.get_splits(OffsetReferential::Normalized, OffsetType::Byte)
        {
            if piece.is_empty() {
                continue;
            }
            *acc.entry(piece.as_bytes().to_vec()).or_default() += 1;
        }
    }
    Ok((n_docs, n_bytes))
}

fn run_load(args: &TrainArgs) -> Result<LoadStats> {
    let words_path = args.checkpoint_dir.join("words.bin");
    if words_path.exists() {
        let (docs, bytes, n_words) = read_words_header(&words_path)?;
        println!(
            "[load] found existing {:?}: {} docs, {:.2} GiB, {} unique words -- reusing (delete it to re-run)",
            words_path,
            docs,
            bytes as f64 / 2f64.powi(30),
            n_words
        );
        return Ok(LoadStats {
            n_files: 0,
            docs,
            bytes,
            unique_words: n_words as usize,
            elapsed_secs: 0.0,
        });
    }

    let prefix_configs: Vec<PrefixConfigItem> =
        serde_saphyr::from_reader(File::open(&args.prefix_config)?)?;
    std::fs::create_dir_all(&args.checkpoint_dir)?;

    println!("[load] scanning and loading data...");
    let files = scan_inputs(&args.dirs)?;
    let n_files = files.len();
    println!("[load] found {} input files", n_files);

    // Progress lines every 15s (indicatif bars are invisible in non-tty logs).
    let files_done = Arc::new(AtomicUsize::new(0));
    let docs_loaded = Arc::new(AtomicUsize::new(0));
    let bytes_loaded = Arc::new(AtomicUsize::new(0));
    let stop_monitor = Arc::new(AtomicBool::new(false));
    {
        let files_done = files_done.clone();
        let docs_loaded = docs_loaded.clone();
        let bytes_loaded = bytes_loaded.clone();
        let stop = stop_monitor.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            while !stop.load(AtomicOrdering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(15));
                let done = files_done.load(AtomicOrdering::Relaxed);
                let docs = docs_loaded.load(AtomicOrdering::Relaxed);
                let bytes = bytes_loaded.load(AtomicOrdering::Relaxed);
                let mins = t0.elapsed().as_secs_f64() / 60.0;
                let eta = if done > 0 {
                    mins * (n_files.saturating_sub(done)) as f64 / done as f64
                } else {
                    f64::NAN
                };
                println!(
                    "[load] {}/{} files, {} docs, {:.2} GiB, elapsed {:.1} min, ETA {:.1} min",
                    done,
                    n_files,
                    docs,
                    bytes as f64 / 2f64.powi(30),
                    mins,
                    eta
                );
            }
        });
    }

    let load_started = Instant::now();
    let splitter = Split::new(
        SplitPattern::Regex(SPLIT_REGEX.to_string()),
        tokenizers::SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(Error::msg)?;

    // Per-thread word maps, merged commutatively at the end: word counts are
    // deterministic and independent of the rayon thread count.
    let word_map = files
        .par_iter()
        .fold(
            HashMap::<Vec<u8>, u64>::new,
            |mut acc, f| {
                match process_file(f, args, &prefix_configs, &splitter, &mut acc) {
                    Ok((docs, bytes)) => {
                        docs_loaded.fetch_add(docs, AtomicOrdering::Relaxed);
                        bytes_loaded.fetch_add(bytes, AtomicOrdering::Relaxed);
                    }
                    Err(e) => eprintln!("[load] ERROR processing {:?}: {:#}", f.path, e),
                }
                files_done.fetch_add(1, AtomicOrdering::Relaxed);
                acc
            },
        )
        .reduce(HashMap::<Vec<u8>, u64>::new, |mut a, b| {
            if a.len() < b.len() {
                // Merge the smaller map into the larger one.
                let (mut small, large) = (a, b);
                a = large;
                for (k, v) in small.drain() {
                    *a.entry(k).or_default() += v;
                }
                a
            } else {
                for (k, v) in b {
                    *a.entry(k).or_default() += v;
                }
                a
            }
        });

    stop_monitor.store(true, AtomicOrdering::Relaxed);
    let docs_total = docs_loaded.load(AtomicOrdering::Relaxed) as u64;
    let bytes_total = bytes_loaded.load(AtomicOrdering::Relaxed) as u64;

    // Sort for a deterministic checkpoint file, then persist.
    let mut words: Vec<(Vec<u8>, u64)> = word_map.into_iter().collect();
    words.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
    write_words_checkpoint(&words_path, docs_total, bytes_total, &words)?;

    let elapsed = load_started.elapsed().as_secs_f64();
    let ckpt_bytes = std::fs::metadata(&words_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "[load] done: {} files, {} documents, {:.2} GiB text, {} unique words, \
         words.bin {:.2} GiB, in {:.1} min ({:.1} MiB/s)",
        n_files,
        docs_total,
        bytes_total as f64 / 2f64.powi(30),
        words.len(),
        ckpt_bytes as f64 / 2f64.powi(30),
        elapsed / 60.0,
        bytes_total as f64 / 2f64.powi(20) / elapsed.max(1e-9),
    );

    Ok(LoadStats {
        n_files,
        docs: docs_total,
        bytes: bytes_total,
        unique_words: words.len(),
        elapsed_secs: elapsed,
    })
}

// ---------------------------------------------------------------------------
// Phase: train
// ---------------------------------------------------------------------------

/// Lazy max-heap entry: best = highest count, ties broken by ascending
/// token-id pair -- exactly the reference BpeTrainer's `Merge::cmp`
/// (`count.cmp` then reversed `pair.cmp`). `count` is the i32 pair count
/// after the reference's `count as u64` sign-extending cast, so entries for
/// overflowed (negative) counts sort exactly as the reference's do.
#[derive(Eq, PartialEq)]
struct QEntry {
    count: u64,
    pair: Pair,
}

impl PartialOrd for QEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            // Reversed: for equal counts the *smaller* id pair is greater.
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

struct TrainState {
    /// id -> raw byte content (specials: their literal UTF-8 bytes)
    vocab_bytes: Vec<Vec<u8>>,
    /// raw byte content -> id
    content_to_id: HashMap<Vec<u8>, u32>,
    merges: Vec<(u32, u32)>,
    /// (token sequence, corpus count) per unique word
    words: Vec<(Vec<u32>, u64)>,
    /// weighted adjacent-pair counts
    pair_counts: PairCounts,
    /// pair -> indices of words containing it (append-only, may be stale;
    /// sorted + deduped and re-validated when the pair is selected)
    where_pair: HashMap<(u32, u32), Vec<u32>>,
    heap: BinaryHeap<QEntry>,
}

fn merge_pair_in_place(toks: &mut Vec<u32>, a: u32, b: u32, new_id: u32) {
    let mut out = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        if i + 1 < toks.len() && toks[i] == a && toks[i + 1] == b {
            out.push(new_id);
            i += 2;
        } else {
            out.push(toks[i]);
            i += 1;
        }
    }
    *toks = out;
}

/// Merge all (non-overlapping, left-to-right) occurrences of (a, b) in one
/// word, returning the pair-count changes exactly as the reference's
/// `Word::merge` computes them: per merge site, -1 for the two outer pairs
/// ((left, a) and (b, right)) and +1 for the two new outer pairs
/// ((left, new_id) and (new_id, right)). The merged pair's own count is never
/// decremented -- same as the reference.
fn merge_word_with_changes(toks: &mut Vec<u32>, a: u32, b: u32, new_id: u32) -> Vec<(Pair, i32)> {
    let mut changes = Vec::new();
    let mut out: Vec<u32> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        if i + 1 < toks.len() && toks[i] == a && toks[i + 1] == b {
            if let Some(&left) = out.last() {
                changes.push(((left, a), -1));
                changes.push(((left, new_id), 1));
            }
            if i + 2 < toks.len() {
                changes.push(((b, toks[i + 2]), -1));
                changes.push(((new_id, toks[i + 2]), 1));
            }
            out.push(new_id);
            i += 2;
        } else {
            out.push(toks[i]);
            i += 1;
        }
    }
    *toks = out;
    changes
}

/// Replay recorded merges on one word's token sequence: repeatedly merge the
/// lowest-rank adjacent pair (all occurrences at once), which reproduces the
/// training-time state exactly.
fn replay_merges_on_word(
    toks: &mut Vec<u32>,
    merges: &[(u32, u32)],
    ranks: &HashMap<(u32, u32), u32>,
    vocab_bytes: &[Vec<u8>],
    content_to_id: &HashMap<Vec<u8>, u32>,
) {
    loop {
        let mut best: Option<u32> = None; // lowest applicable rank
        for w in toks.windows(2) {
            if let Some(&r) = ranks.get(&(w[0], w[1]))
                && best.is_none_or(|br| r < br)
            {
                best = Some(r);
            }
        }
        let Some(r) = best else { break };
        let (a, b) = merges[r as usize];
        let mut content = vocab_bytes[a as usize].clone();
        content.extend_from_slice(&vocab_bytes[b as usize]);
        let new_id = content_to_id[&content];
        merge_pair_in_place(toks, a, b, new_id);
    }
}

/// Initial pair count + pair -> word-indices maps (parallel, deterministic:
/// counts are sums; index lists are sorted+deduped before use).
fn count_pairs(
    words: &[(Vec<u32>, u64)],
) -> (PairCounts, PairLocations) {
    words
        .par_iter()
        .enumerate()
        .fold(
            || (HashMap::new(), HashMap::new()),
            |(mut pc, mut wp): (PairCounts, PairLocations),
             (i, (toks, cnt))| {
                // Wrapping i32 accumulate, exactly like the reference.
                let cnt = *cnt as i32;
                for w in toks.windows(2) {
                    let pair = (w[0], w[1]);
                    let e = pc.entry(pair).or_insert(0);
                    *e = e.wrapping_add(cnt);
                    wp.entry(pair).or_default().push(i as u32);
                }
                (pc, wp)
            },
        )
        .reduce(
            || (HashMap::new(), HashMap::new()),
            |(mut pc1, mut wp1), (pc2, wp2)| {
                for (k, v) in pc2 {
                    let e = pc1.entry(k).or_insert(0);
                    *e = e.wrapping_add(v);
                }
                for (k, v) in wp2 {
                    wp1.entry(k).or_default().extend(v);
                }
                (pc1, wp1)
            },
        )
}

fn init_train_state(args: &TrainArgs) -> Result<TrainState> {
    let words_path = args.checkpoint_dir.join("words.bin");
    if !words_path.exists() {
        bail!(
            "{:?} not found -- run the load phase first (--phase load or --phase all)",
            words_path
        );
    }
    let n_specials = args.special_tokens.len();
    let table = bytes_to_unicode_table();

    println!("[train] reading {:?} ...", words_path);
    let t0 = Instant::now();
    let (docs_total, bytes_total, raw_words) = read_words_checkpoint(&words_path)?;
    let total_tokens: usize = raw_words.iter().map(|(w, _)| w.len()).sum();
    println!(
        "[train] {} unique words ({} docs, {:.2} GiB text, {} total tokens) read in {:.1}s",
        raw_words.len(),
        docs_total,
        bytes_total as f64 / 2f64.powi(30),
        total_tokens,
        t0.elapsed().as_secs_f64()
    );

    // Alphabet: byte-level chars present in the corpus (a byte is present iff
    // its ByteLevel char is), sorted by char codepoint -- exactly the
    // reference trainer's `compute_alphabet`.
    let present = raw_words
        .par_iter()
        .fold(
            || [false; 256],
            |mut acc, (w, _)| {
                for &b in w {
                    acc[b as usize] = true;
                }
                acc
            },
        )
        .reduce(
            || [false; 256],
            |mut a, b| {
                for i in 0..256 {
                    a[i] |= b[i];
                }
                a
            },
        );
    let alpha_order = alphabet_order(&table, &present);

    // Base vocab: specials (ids 0..n_specials-1) + present byte-level entries.
    let mut vocab_bytes: Vec<Vec<u8>> = Vec::with_capacity(args.vocab_size.max(1024));
    let mut content_to_id: HashMap<Vec<u8>, u32> = HashMap::new();
    for tok in &args.special_tokens {
        let bytes = tok.clone().into_bytes();
        content_to_id.insert(bytes.clone(), (vocab_bytes.len()) as u32);
        vocab_bytes.push(bytes);
    }
    let mut byte_to_id = [0u32; 256];
    for &b in &alpha_order {
        let id = vocab_bytes.len() as u32;
        byte_to_id[b as usize] = id;
        content_to_id.insert(vec![b], id);
        vocab_bytes.push(vec![b]);
    }
    let n_initial = vocab_bytes.len();
    println!(
        "[train] initial vocab: {} specials + {} corpus-present byte-level entries = {}",
        n_specials,
        alpha_order.len(),
        n_initial
    );

    // Tokenize words to initial token id sequences.
    let mut words: Vec<(Vec<u32>, u64)> = raw_words
        .into_par_iter()
        .map(|(w, c)| (w.iter().map(|&b| byte_to_id[b as usize]).collect(), c))
        .collect();

    // Resume from the latest merges checkpoint if present.
    //
    // Queue-eligibility subtlety: the reference only ever queues a pair if it
    // had a positive (i32) count at init or received a positive change from
    // some merge. Pairs of two INITIAL tokens can never be created by a merge
    // (positive changes always involve the new token), so a pair that was
    // non-positive at init (i32 overflow) stays out of the queue forever. To
    // resume identically, count pairs on the PRE-REPLAY state and treat as
    // eligible: pairs positive at init, plus any pair involving a merged
    // token (id >= n_initial).
    let resume_ckpt = latest_merges_checkpoint(&args.checkpoint_dir)?;
    let mut eligible_at_init: Option<HashSet<Pair>> = None;
    if resume_ckpt.is_some() {
        let (pc0, _) = count_pairs(&words);
        eligible_at_init = Some(pc0.into_iter().filter(|&(_, c)| c > 0).map(|(p, _)| p).collect());
    }
    let mut merges: Vec<(u32, u32)> = Vec::new();
    if let Some((n, path)) = resume_ckpt {
        let (ckpt_vocab, ckpt_merges) = read_merges_checkpoint(&path)?;
        if ckpt_vocab.len() < n_initial || ckpt_vocab[..n_initial] != vocab_bytes[..] {
            bail!(
                "{:?} does not match the current special tokens / alphabet -- cannot resume",
                path
            );
        }
        if ckpt_merges.len() != n {
            bail!("{:?}: filename N={} but contains {} merges", path, n, ckpt_merges.len());
        }
        vocab_bytes = ckpt_vocab;
        content_to_id = vocab_bytes
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        merges = ckpt_merges;
        println!(
            "[train] resuming from {:?}: {} merges, vocab {}",
            path,
            merges.len(),
            vocab_bytes.len()
        );

        // Replay the recorded merges on every word (greedy by merge rank).
        let ranks: HashMap<(u32, u32), u32> = merges
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i as u32))
            .collect();
        let t0 = Instant::now();
        words.par_iter_mut().for_each(|(toks, _)| {
            replay_merges_on_word(toks, &merges, &ranks, &vocab_bytes, &content_to_id);
        });
        println!(
            "[train] replayed {} merges on {} words in {:.1}s",
            merges.len(),
            words.len(),
            t0.elapsed().as_secs_f64()
        );
    } else {
        println!("[train] no merges checkpoint found, starting fresh");
    }

    // Initial pair counts (rebuilt from the current word token sequences).
    let t0 = Instant::now();
    let (pair_counts, where_pair) = count_pairs(&words);
    println!(
        "[train] {} distinct pairs counted in {:.1}s",
        pair_counts.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut heap = BinaryHeap::with_capacity(pair_counts.len());
    for (&pair, &count) in &pair_counts {
        // The reference queues only pairs with a positive (i32) count; the
        // queued value is `count as u64` (sign-extended).
        if count <= 0 {
            continue;
        }
        if let Some(eligible) = &eligible_at_init {
            // Resume: keep the run's admission rule (see above).
            let involves_merged =
                pair.0 as usize >= n_initial || pair.1 as usize >= n_initial;
            if !involves_merged && !eligible.contains(&pair) {
                continue;
            }
        }
        heap.push(QEntry {
            count: count as u64,
            pair,
        });
    }

    Ok(TrainState {
        vocab_bytes,
        content_to_id,
        merges,
        words,
        pair_counts,
        where_pair,
        heap,
    })
}

/// Apply merge (a, b) -> new_id to every word that contains it, updating
/// pair_counts with wrapping i32 deltas from per-site changes (exactly like
/// the reference's `Word::merge` + changes loop). Returns the pairs that
/// received at least one *positive* change: only those get fresh queue
/// entries in the reference (`if change > 0 { where_to_update... }`), which
/// is what permanently keeps pairs whose count was non-positive at init (i32
/// overflow) out of the queue even if their count later wraps positive
/// through pure destruction.
fn apply_merge(state: &mut TrainState, a: u32, b: u32, new_id: u32) -> HashSet<Pair> {
    let idxs = state.where_pair.remove(&(a, b)).unwrap_or_default();
    let mut idxs = idxs;
    idxs.sort_unstable();
    idxs.dedup();

    let mut positive: HashSet<Pair> = HashSet::new();
    let words = &mut state.words;
    let pair_counts = &mut state.pair_counts;
    let where_pair = &mut state.where_pair;

    for widx in idxs {
        let (toks, cnt) = &mut words[widx as usize];
        let cnt = *cnt as i32; // wrapping cast, same as the reference
        if !toks.windows(2).any(|w| w[0] == a && w[1] == b) {
            continue; // stale index entry
        }
        let changes = merge_word_with_changes(toks, a, b, new_id);
        for (pair, delta) in changes {
            // `change * counts[iw] as i32`, wrapping, as in the reference.
            let e = pair_counts.entry(pair).or_insert(0);
            *e = e.wrapping_add(delta.wrapping_mul(cnt));
            if delta > 0 {
                where_pair.entry(pair).or_default().push(widx);
                positive.insert(pair);
            }
        }
    }
    positive
}

/// Assemble and save tokenizer.json (+ config.json) with exactly the same
/// structure as the reference trainer.
fn save_tokenizer(args: &TrainArgs, vocab_bytes: &[Vec<u8>], merges: &[(u32, u32)]) -> Result<()> {
    let n_specials = args.special_tokens.len();
    let table = bytes_to_unicode_table();

    let mut vocab: Vocab = Vocab::default();
    for (id, bytes) in vocab_bytes.iter().enumerate() {
        let token = if id < n_specials {
            String::from_utf8(bytes.clone()).context("special token is not valid UTF-8")?
        } else {
            bytes_to_bpe_string(bytes, &table)
        };
        vocab.insert(token, id as u32);
    }
    let merges_str: Vec<(String, String)> = merges
        .iter()
        .map(|&(a, b)| {
            (
                bytes_to_bpe_string(&vocab_bytes[a as usize], &table),
                bytes_to_bpe_string(&vocab_bytes[b as usize], &table),
            )
        })
        .collect();

    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges_str)
        .build()
        .map_err(Error::msg)?;

    let create_byte_level =
        || ByteLevel::default().add_prefix_space(false).trim_offsets(false).use_regex(false);
    let create_pre_tokenizer = || {
        Sequence::new(vec![
            Split::new(
                SplitPattern::Regex(SPLIT_REGEX.to_string()),
                tokenizers::SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap()
            .into(),
            create_byte_level().into(),
        ])
    };

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(bpe)
        .with_normalizer(Some(NFC))
        .with_pre_tokenizer(Some(create_pre_tokenizer()))
        .with_post_processor(Some(create_byte_level()))
        .with_decoder(Some(create_byte_level()))
        .build()
        .map_err(Error::msg)?;

    let special_tokens: Vec<AddedToken> = args
        .special_tokens
        .iter()
        .map(|s| AddedToken::from(s, true))
        .collect();
    tokenizer.add_special_tokens(&special_tokens);

    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    tokenizer.save(&args.out, true).map_err(Error::msg)?;

    // Save config.json for compatibility with Transformers
    let config_path = args
        .out
        .parent()
        .map(|p| p.join("config.json"))
        .unwrap_or_else(|| PathBuf::from("config.json"));
    std::fs::write(&config_path, "{\"model_type\": \"qwen3\"}")?;
    Ok(())
}

fn run_train(args: &TrainArgs) -> Result<()> {
    if args.tokenizer_type.to_lowercase() != "bpe" {
        return Err(Error::msg(format!(
            "Unsupported tokenizer type: {} (train_tokenizer_iter only implements bpe).",
            args.tokenizer_type
        )));
    }

    let mut state = init_train_state(args)?;
    std::fs::create_dir_all(&args.checkpoint_dir)?;

    // Checkpoint on SIGINT/SIGTERM. The handler can only be installed once per
    // process (relevant for tests calling run_train repeatedly).
    static INTERRUPTED: AtomicBool = AtomicBool::new(false);
    static INSTALL_HANDLER: std::sync::Once = std::sync::Once::new();
    INSTALL_HANDLER.call_once(|| {
        ctrlc::set_handler(move || {
            INTERRUPTED.store(true, AtomicOrdering::SeqCst);
        })
        .expect("failed to install ctrlc handler");
    });
    let interrupted = &INTERRUPTED;

    // Initial vocab size = current vocab minus merges applied so far
    // (display only; content collisions can make this off by a little).
    let n_initial = state.vocab_bytes.len() - state.merges.len();
    let total_merges_target = args.vocab_size.saturating_sub(n_initial);
    let train_started = Instant::now();
    let mut window_started = Instant::now();
    let mut window_start_merges = state.merges.len();
    let mut next_checkpoint = if args.checkpoint_interval == 0 {
        usize::MAX
    } else {
        (state.merges.len() / args.checkpoint_interval + 1) * args.checkpoint_interval
    };

    println!(
        "[train] starting BPE merge loop: vocab {} -> {}, {} merges already done",
        state.vocab_bytes.len(),
        args.vocab_size,
        state.merges.len()
    );

    loop {
        if state.vocab_bytes.len() >= args.vocab_size {
            break;
        }
        if let Some(cap) = args.max_merges
            && state.merges.len() >= cap
        {
            let path = args
                .checkpoint_dir
                .join(format!("merges_{}.bin", state.merges.len()));
            write_merges_checkpoint(&path, &state.vocab_bytes, &state.merges)?;
            println!(
                "[train] --max-merges {} reached, wrote {:?} and stopping without tokenizer.json",
                cap, path
            );
            return Ok(());
        }
        if interrupted.load(AtomicOrdering::SeqCst) {
            let path = args
                .checkpoint_dir
                .join(format!("merges_{}.bin", state.merges.len()));
            write_merges_checkpoint(&path, &state.vocab_bytes, &state.merges)?;
            println!(
                "[train] interrupted, wrote {:?} ({} merges) -- re-run to resume",
                path,
                state.merges.len()
            );
            std::process::exit(2);
        }

        // Pop until we find a non-stale heap entry; re-push stale entries
        // with the current (sign-extended) count, like the reference.
        let top = loop {
            let Some(entry) = state.heap.pop() else {
                break None;
            };
            let current = state.pair_counts.get(&entry.pair).copied().unwrap_or(0) as u64;
            if current == entry.count {
                break Some(entry);
            }
            state.heap.push(QEntry {
                count: current,
                pair: entry.pair,
            });
        };
        let Some(top) = top else {
            println!("[train] no more pairs to merge");
            break;
        };
        if top.count < 1 || MIN_FREQUENCY > top.count {
            println!(
                "[train] best pair count {} < min_frequency {}, stopping",
                top.count, MIN_FREQUENCY
            );
            break;
        }

        let (a, b) = top.pair;
        // Insert the new token (on content collision with an existing token,
        // e.g. a special token string, reuse its id -- same as the reference
        // trainer).
        let mut content = state.vocab_bytes[a as usize].clone();
        content.extend_from_slice(&state.vocab_bytes[b as usize]);
        let new_id = match state.content_to_id.get(&content) {
            Some(&id) => id,
            None => {
                let id = state.vocab_bytes.len() as u32;
                state.content_to_id.insert(content.clone(), id);
                state.vocab_bytes.push(content);
                id
            }
        };
        state.merges.push((a, b));

        let positive = apply_merge(&mut state, a, b, new_id);
        for pair in positive {
            // Fresh queue entries only for pairs with a positive (i32) count,
            // like the reference's `if count > 0 { queue.push(...) }`.
            if let Some(&count) = state.pair_counts.get(&pair)
                && count > 0
            {
                state.heap.push(QEntry {
                    count: count as u64,
                    pair,
                });
            }
        }

        let done = state.merges.len();
        if done % 25 == 0 || done == 1 {
            let window_secs = window_started.elapsed().as_secs_f64();
            let window_merges = done - window_start_merges;
            let rate = window_merges as f64 / window_secs.max(1e-9);
            let remaining = args.vocab_size.saturating_sub(state.vocab_bytes.len());
            let eta_min = remaining as f64 / rate.max(1e-9) / 60.0;
            println!(
                "[train] merge {}/{} pair=({}, {}) count={} {:.1} merges/s ETA {:.1} min",
                done,
                n_initial + total_merges_target,
                show_bytes(&state.vocab_bytes[a as usize]),
                show_bytes(&state.vocab_bytes[b as usize]),
                top.count,
                rate,
                eta_min
            );
            window_started = Instant::now();
            window_start_merges = done;
        }

        if done >= next_checkpoint {
            let path = args.checkpoint_dir.join(format!("merges_{}.bin", done));
            write_merges_checkpoint(&path, &state.vocab_bytes, &state.merges)?;
            println!("[train] checkpoint {:?}", path);
            next_checkpoint += args.checkpoint_interval;
        }
    }

    let train_secs = train_started.elapsed().as_secs_f64();
    println!(
        "[train] merge loop done: {} merges, vocab {} in {:.1} min",
        state.merges.len(),
        state.vocab_bytes.len(),
        train_secs / 60.0
    );

    save_tokenizer(args, &state.vocab_bytes, &state.merges)?;
    println!("[train] saved tokenizer to {:?}", args.out);
    Ok(())
}

fn main() -> Result<()> {
    let args = TrainArgs::parse();
    match args.phase {
        Phase::Load => {
            run_load(&args)?;
        }
        Phase::Train => {
            run_train(&args)?;
        }
        Phase::All => {
            run_load(&args)?;
            run_train(&args)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_specials() -> Vec<String> {
        [
            "<|PAD|>",
            "<|direct|>",
            "<|cot|>",
            "<|noisy|>",
            "<|synth|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<think>",
            "</think>",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn write_jsonl(path: &Path, n_rows: usize) {
        // NOTE: the pre-tokenizer regex splits letters from digits, so unique
        // words only come from varying alphabetic content.
        const SYLLABLES: [&str; 16] = [
            "ab", "ka", "zu", "le", "mi", "ro", "te", "xa", "ur", "pi", "os", "en", "ga", "fu",
            "ly", "wo",
        ];
        let word = |i: usize, salt: usize| {
            // base-16 digits of i (plus salt) -> decorrelated syllable combos
            let j = i + salt * 4096;
            format!(
                "{}{}{}",
                SYLLABLES[j % 16],
                SYLLABLES[(j / 16) % 16],
                SYLLABLES[(j / 256) % 16]
            )
        };
        let mut f = File::create(path).unwrap();
        for i in 0..n_rows {
            let inst = format!(
                "Question number {}: what is {} plus {}? Please explain step by step, \
                 carefully and slowly. café naïve 数学 🎉 {} {}",
                i,
                i % 97,
                i % 89,
                word(i, 1),
                word(i, 2)
            );
            let resp = format!(
                "The answer is {}. Let me think carefully about this problem. \
                 First, {} plus {} equals {}. This is a well known result in \
                 elementary arithmetic. {} {} {}\n\
                 Second line with\ttabs and   spaces.",
                (i % 97) + (i % 89),
                i % 97,
                i % 89,
                (i % 97) + (i % 89),
                word(i, 3),
                word(i, 4),
                word(i, 5)
            );
            writeln!(
                f,
                "{}",
                serde_json::json!({
                    "condition": "test",
                    "instruction": inst,
                    "response": resp,
                })
            )
            .unwrap();
        }
    }

    fn make_args(dir: &Path, max_merges: Option<usize>) -> TrainArgs {
        let corpus = dir.join("corpus");
        let ckpt = dir.join("ckpt");
        let prefix_config = dir.join("prefix_config.yaml");
        std::fs::write(
            &prefix_config,
            "- prefix: \"limited__\"\n  max_per_file: 50\n",
        )
        .unwrap();
        TrainArgs {
            dirs: vec![corpus],
            out: dir.join("out").join("tokenizer.json"),
            prefix_config,
            checkpoint_dir: ckpt,
            seed: 0,
            vocab_size: 700,
            tokenizer_type: "bpe".to_string(),
            phase: Phase::All,
            truncate_len: 10_000,
            limit_mul_factor: 10,
            checkpoint_interval: 4,
            max_merges,
            special_tokens: default_specials(),
        }
    }

    fn build_corpus(dir: &Path) {
        let corpus = dir.join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        write_jsonl(&corpus.join("plain.jsonl"), 8000);
        // 1000 rows -> 2000 docs > limit (10 * 50 = 500) -> sampling kicks in
        write_jsonl(&corpus.join("limited__big.jsonl"), 1000);
    }

    #[test]
    fn end_to_end_and_resume() {
        let dir = std::env::temp_dir().join(format!("tti_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        build_corpus(&dir);

        // --- Phase load ---
        let args = make_args(&dir, None);
        let stats = run_load(&args).unwrap();
        // plain.jsonl: 8000 rows -> 16000 docs; limited__big.jsonl sampled to 500
        assert_eq!(stats.docs, 16000 + 500);
        assert!(stats.unique_words > 100);
        let words_path = dir.join("ckpt").join("words.bin");
        assert!(words_path.exists());
        let (docs, bytes, words) = read_words_checkpoint(&words_path).unwrap();
        assert_eq!(docs, stats.docs);
        assert_eq!(bytes, stats.bytes);
        assert_eq!(words.len(), stats.unique_words);
        // counts are positive and sorted keys (deterministic checkpoint)
        assert!(words.iter().all(|(_, c)| *c > 0));
        assert!(words.windows(2).all(|w| w[0].0 < w[1].0));
        let n_alphabet = {
            let mut present = [false; 256];
            for (w, _) in &words {
                for &b in w {
                    present[b as usize] = true;
                }
            }
            present.iter().filter(|&&p| p).count()
        };

        // --- Phase train, interrupted after 8 merges (simulated kill) ---
        let args = make_args(&dir, Some(8));
        run_train(&args).unwrap();
        assert!(dir.join("ckpt").join("merges_8.bin").exists());
        assert!(!args.out.exists(), "tokenizer.json must not be written yet");
        let (ckpt_vocab, ckpt_merges) =
            read_merges_checkpoint(&dir.join("ckpt").join("merges_8.bin")).unwrap();
        assert_eq!(ckpt_merges.len(), 8);
        assert_eq!(ckpt_vocab.len(), 31 + n_alphabet + 8);

        // --- Resume: completes from merges_8.bin ---
        let args = make_args(&dir, None);
        run_train(&args).unwrap();
        assert!(args.out.exists());

        // --- Validate tokenizer.json structure ---
        let json: serde_json::Value =
            serde_json::from_reader(File::open(&args.out).unwrap()).unwrap();
        assert_eq!(json["normalizer"]["type"], "NFC");
        let pt = &json["pre_tokenizer"];
        assert_eq!(pt["type"], "Sequence");
        assert_eq!(pt["pretokenizers"][0]["type"], "Split");
        assert_eq!(pt["pretokenizers"][0]["pattern"]["Regex"], SPLIT_REGEX);
        assert_eq!(pt["pretokenizers"][0]["behavior"], "Isolated");
        assert_eq!(pt["pretokenizers"][0]["invert"], false);
        for wrapper in [
            &pt["pretokenizers"][1],
            &json["post_processor"],
            &json["decoder"],
        ] {
            assert_eq!(wrapper["type"], "ByteLevel");
            assert_eq!(wrapper["add_prefix_space"], false);
            assert_eq!(wrapper["trim_offsets"], false);
            assert_eq!(wrapper["use_regex"], false);
        }
        assert_eq!(json["model"]["type"], "BPE");
        let added = json["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 31);
        for (i, tok) in added.iter().enumerate() {
            assert_eq!(tok["id"], i as u64);
            assert_eq!(tok["special"], true);
            assert_eq!(tok["single_word"], false);
            assert_eq!(tok["lstrip"], false);
            assert_eq!(tok["rstrip"], false);
            assert_eq!(tok["normalized"], false);
        }
        assert_eq!(added[0]["content"], "<|PAD|>");
        assert_eq!(added[30]["content"], "</think>");
        assert_eq!(json["model"]["vocab"].as_object().unwrap().len(), 700);
        assert_eq!(
            json["model"]["merges"].as_array().unwrap().len(),
            700 - 31 - n_alphabet
        );

        // config.json written next to tokenizer.json
        let cfg: serde_json::Value = serde_json::from_reader(
            File::open(dir.join("out").join("config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(cfg["model_type"], "qwen3");

        // --- Round-trip via the tokenizers crate ---
        let tokenizer = tokenizers::Tokenizer::from_file(&args.out).unwrap();
        assert_eq!(tokenizer.get_vocab_size(true), 700);
        assert_eq!(tokenizer.token_to_id("<|PAD|>"), Some(0));
        assert_eq!(tokenizer.token_to_id("<|endoftext|>"), Some(5));
        assert_eq!(tokenizer.token_to_id("</think>"), Some(30));
        // NOTE: round-trip only works for bytes that occur in the corpus
        // (the alphabet is corpus-derived, same as the reference trainer).
        for text in [
            "Please explain step by step, carefully and slowly.",
            "café naïve 数学 🎉",
            "  leading spaces\n\nnewlines\t tabs  ",
            "The answer is 42. Let me think carefully.",
        ] {
            let enc = tokenizer.encode(text, true).unwrap();
            let dec = tokenizer.decode(enc.get_ids(), false).unwrap();
            assert_eq!(dec, text, "round-trip failed for {:?}", text);
        }
        // Special tokens are recognized and keep their ids in encodings.
        let enc = tokenizer.encode("<|PAD|> hello <|im_end|>", true).unwrap();
        assert_eq!(enc.get_ids()[0], 0);
        assert_eq!(*enc.get_ids().last().unwrap(), 7);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Merge-for-merge and vocab parity with the reference trainer (the
    /// tokenizers crate's BpeTrainer) on identical word counts.
    #[test]
    fn parity_with_reference_bpe_trainer() {
        use tokenizers::models::bpe::{BPE, BpeTrainer};

        let dir = std::env::temp_dir().join(format!("tti_parity_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        build_corpus(&dir);

        let args = make_args(&dir, None);
        run_load(&args).unwrap();
        run_train(&args).unwrap();

        // My tokenizer.json.
        let json: serde_json::Value =
            serde_json::from_reader(File::open(&args.out).unwrap()).unwrap();

        // Reference: crate BpeTrainer::do_train on the same word counts,
        // words mapped to byte-level char space.
        let table = bytes_to_unicode_table();
        let (_, _, words) = read_words_checkpoint(&dir.join("ckpt").join("words.bin")).unwrap();
        let mut wc: ahash::AHashMap<compact_str::CompactString, u64> = Default::default();
        for (w, c) in &words {
            wc.insert(
                compact_str::CompactString::from(bytes_to_bpe_string(w, &table)),
                *c,
            );
        }
        let special_tokens: Vec<AddedToken> = default_specials()
            .iter()
            .map(|s| AddedToken::from(s, true))
            .collect();
        let trainer = BpeTrainer::builder()
            .vocab_size(700)
            .min_frequency(2)
            .special_tokens(special_tokens)
            .show_progress(false)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&wc, &mut model).unwrap();

        // Save the reference model through the same TokenizerBuilder/save
        // path the reference trainer uses, then compare at the JSON level.
        let create_byte_level =
            || ByteLevel::default().add_prefix_space(false).trim_offsets(false).use_regex(false);
        let ref_tok = TokenizerBuilder::new()
            .with_model(model)
            .with_normalizer(Some(NFC))
            .with_pre_tokenizer(Some(Sequence::new(vec![
                Split::new(
                    SplitPattern::Regex(SPLIT_REGEX.to_string()),
                    tokenizers::SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap()
                .into(),
                create_byte_level().into(),
            ])))
            .with_post_processor(Some(create_byte_level()))
            .with_decoder(Some(create_byte_level()))
            .build()
            .map_err(Error::msg)
            .unwrap();
        let ref_path = dir.join("ref_tokenizer.json");
        ref_tok.save(&ref_path, true).unwrap();
        let ref_json: serde_json::Value =
            serde_json::from_reader(File::open(&ref_path).unwrap()).unwrap();

        // Vocab must be identical (token -> id).
        assert_eq!(
            json["model"]["vocab"], ref_json["model"]["vocab"],
            "vocab mismatch"
        );
        // Merges must be identical, in rank order.
        assert_eq!(
            json["model"]["merges"], ref_json["model"]["merges"],
            "merge order mismatch"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
