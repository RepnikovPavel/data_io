use anyhow::{Result, Error};
use clap::Parser;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::path::PathBuf;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand_pcg::rand_core::SeedableRng;
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::models::unigram::{Unigram, UnigramTrainer};
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::{AddedToken, TokenizerBuilder};

use tokenizer::{read_any_stream, scan_inputs};

// Optimize memory allocation
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
struct TrainArgs {
    #[arg(required = true, num_args = 1..)]
    dirs: Vec<PathBuf>,
    #[arg(short, required = true)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(short, long, default_value = "../prefix_config.yaml")]
    prefix_config: PathBuf,
    #[arg(long, default_value_t = 10_000)]
    truncate_len: usize, // Takes first truncate_len characters. Maybe larger for context longer than 2048.
    #[arg(long, default_value_t = 10)]
    limit_mul_factor: usize,

    #[arg(long, default_value = "bpe")]
    tokenizer_type: String,
    #[arg(long, default_value_t = 65536)]
    vocab_size: usize,

    /// Special token list
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

#[derive(Debug, Deserialize)]
struct PrefixConfigItem {
    prefix: String,
    max_per_file: Option<usize>,
}

fn truncate_safe(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars { // Optimization: len() in bytes >= len() in chars
        return s;
    }
    // Find the byte index of the Nth character
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s, // String has fewer than max_chars characters
    }
}

fn main() -> Result<()> {
    let args = TrainArgs::parse();
    let prefix_configs: Vec<PrefixConfigItem> = serde_saphyr::from_reader(File::open(&args.prefix_config)?)?;

    // Load data
    println!("Scanning and loading data...");
    let files = scan_inputs(&args.dirs)?;
    let n_files = files.len();
    println!("Found {} input files", n_files);

    // Counters for log-friendly progress (indicatif bars are invisible in
    // non-tty logs, so emit explicit progress lines every 15s).
    let files_done = Arc::new(AtomicUsize::new(0));
    let docs_loaded = Arc::new(AtomicUsize::new(0));
    let stop_monitor = Arc::new(AtomicBool::new(false));
    {
        let files_done = files_done.clone();
        let docs_loaded = docs_loaded.clone();
        let stop = stop_monitor.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(15));
                let done = files_done.load(Ordering::Relaxed);
                let docs = docs_loaded.load(Ordering::Relaxed);
                let mins = t0.elapsed().as_secs_f64() / 60.0;
                let eta = if done > 0 {
                    mins * (n_files - done) as f64 / done as f64
                } else { f64::NAN };
                println!("[load] {}/{} files, {} docs, elapsed {:.1} min, ETA {:.1} min",
                         done, n_files, docs, mins, eta);
            }
        });
    }

    let pb = ProgressBar::new(n_files as u64);
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(2));
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {msg} {bar:40} {pos}/{len} {elapsed} ETA {eta}")?);
    let load_started = Instant::now();

    let all_documents: Vec<String> = files.into_par_iter().progress_with(pb).filter_map(|f| {
        let mut local_docs = Vec::new();
        read_any_stream(&f.path, |_, inst, resp| {
            local_docs.push(truncate_safe(inst, args.truncate_len).to_string());
            local_docs.push(truncate_safe(resp, args.truncate_len).to_string());
        }).unwrap();

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
        docs_loaded.fetch_add(local_docs.len(), Ordering::Relaxed);
        files_done.fetch_add(1, Ordering::Relaxed);
        Some(local_docs)
    }).flatten().collect();

    stop_monitor.store(true, Ordering::Relaxed);
    let load_secs = load_started.elapsed().as_secs_f64();
    let total_bytes: usize = all_documents.iter().map(|s| s.len()).sum();
    println!("[load] done: {} files, {} documents, {:.1} GiB text in {:.1} min",
             n_files, all_documents.len(), total_bytes as f64 / 2f64.powi(30), load_secs / 60.0);

    // Train
    println!("[train] starting {} training (vocab_size={}) on {} documents...",
             args.tokenizer_type, args.vocab_size, all_documents.len());
    let train_started = Instant::now();
    let stop_heartbeat = Arc::new(AtomicBool::new(false));
    {
        let stop = stop_heartbeat.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(60));
                println!("[train] in progress, elapsed {:.1} min",
                         train_started.elapsed().as_secs_f64() / 60.0);
            }
        });
    }

    let create_byte_level = || { ByteLevel::default().add_prefix_space(false).trim_offsets(false).use_regex(false) };
    let create_pre_tokenizer = || { Sequence::new(vec![
            Split::new(
                SplitPattern::Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_string()),
                tokenizers::SplitDelimiterBehavior::Isolated,
                false
            ).unwrap().into(),
            create_byte_level().into(),
        ])
    };
    let special_tokens: Vec<AddedToken> = args.special_tokens.iter()
        .map(|s| AddedToken::from(s, true))
        .collect();

    match args.tokenizer_type.to_lowercase().as_str() {
        "bpe" => {
            let mut builder = TokenizerBuilder::new()
                .with_model(BPE::default())
                .with_normalizer(Some(NFC))
                .with_pre_tokenizer(Some(create_pre_tokenizer()))
                .with_post_processor(Some(create_byte_level()))
                .with_decoder(Some(create_byte_level()))
                .build().map_err(Error::msg)?;

            let mut trainer = BpeTrainer::builder()
                .vocab_size(args.vocab_size)
                .min_frequency(2)  // best default for BPE
                .special_tokens(special_tokens)
                .show_progress(true)
                .build();

            builder.train(&mut trainer, all_documents.iter()).map_err(Error::msg)?;
            builder.save(&args.out, true).map_err(Error::msg)?;

            // Save config.json for compatibility with Transformers
            let config_path = args.out.parent()
                .map(|p| p.join("config.json"))
                .unwrap_or_else(|| PathBuf::from("config.json"));

            std::fs::write(&config_path, "{\"model_type\": \"qwen3\"}")?;
        },
        "unigram" => {
            // "BPE-like" unigram. Byte-level split, then run Unigram EM algorithm to "merge" the vocab instead of BPE merging
            let mut builder = TokenizerBuilder::new()
                .with_model(Unigram::default())
                .with_normalizer(Some(NFC))
                .with_pre_tokenizer(Some(create_pre_tokenizer()))
                .with_post_processor(Some(create_byte_level()))
                .with_decoder(Some(create_byte_level()))
                .build().map_err(Error::msg)?;

            let mut trainer = UnigramTrainer::builder()
                .vocab_size(args.vocab_size as u32)
                .special_tokens(special_tokens)
                .show_progress(true)
                // No extra hyperparams (like shrinking_factor) set, use default.
                .build()
                .map_err(Error::msg)?;

            builder.train(&mut trainer, all_documents.iter()).map_err(Error::msg)?;
            builder.save(&args.out, true).map_err(Error::msg)?;

            // Check for missed bytes except invalid UTF-8 bytes:
            // Banned C0, C1
            // Empty future planes F1, F2
            // Out-of-bounds F5 - FF
        },
        other => {
            return Err(Error::msg(format!("Unsupported tokenizer type: {}.", other)));
        }
    }

    stop_heartbeat.store(true, Ordering::Relaxed);
    println!("[train] done in {:.1} min -> {:?}",
             train_started.elapsed().as_secs_f64() / 60.0, args.out);

    Ok(())
}
