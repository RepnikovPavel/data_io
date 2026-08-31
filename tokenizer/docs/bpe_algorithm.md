# BPE: training and inference, in pseudocode

↑ [index](README.md)

Byte Pair Encoding for a reader who has never seen it. This is exactly what
`train_tokenizer_iter` (Rust) and `tokenizer_cpp` implement.

**The whole idea:** start with a vocabulary of single bytes. Repeatedly find
the most frequent *adjacent pair* of symbols in the corpus and merge it into
one new symbol. After 65264 merges the vocabulary has 65536 entries: frequent
substrings (" the", "ing", "Ġthe") became single tokens; rare text still
splits into bytes, so nothing is ever unencodable.

## Notation and data formats

| artifact | what it is | format | size on our corpus |
|---|---|---|---|
| corpus | 5221 parquet/jsonl files, columns `condition, instruction, response` | parquet (snappy) / JSONL | 345 GiB compressed |
| documents | `instruction` and `response` of each row, as separate text strings | in-memory UTF-8 | 410,012,296 docs, 170.2 GiB text |
| `words.bin` | **unique pretokenized words with counts** — the only input the merge loop needs | binary, spec below | 288 MiB, 12,127,412 unique words |
| `merges_N.bin` | resume checkpoint after N merges | binary, spec below | ~1.2 MB at N=65200 |
| `tokenizer.json` | the trained tokenizer | HF tokenizers JSON | 4.7 MB |

### What exactly is inside words.bin

A "word" = one piece produced by the pre-tokenizer regex from NFC-normalized
text. Real examples (top by count) — note these are *text pieces*, not tokens
yet; spaces and punctuation are words too:

```
" the"  1,412,457,831
","     1,278,416,784
" "       994,680,691
"."       954,394,480
"1"       721,120,752
" of"     717,145,171
"0"       692,848,596
" and"    671,535,133
```

Emoji / Cyrillic / any Unicode appear as their raw UTF-8 bytes. Special
tokens (`<|direct|>` etc.) are NOT here — they are injected into the
vocabulary directly at training time.

Binary layout (little-endian, written atomically via tmp+rename):

```
u32  magic "WBI1"
u64  docs_total        # 410,012,296
u64  bytes_total       # text bytes scanned (170.2 GiB)
u64  n_words           # 12,127,412
n_words × { u32 len | u8 word_bytes[len] | u64 count }
```

**Words are stored sorted lexicographically** — so the file is deterministic
(bit-identical across runs and thread counts) and diffable.

### The pre-tokenizer regex

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)        # English contractions: "don't" -> "don", "'t"
| [^\r\n\p{L}\p{N}]?\p{L}+          # optional punct/space + letter run: " the", "!Hello"
| \p{N}                             # single digit/number char
| ?[^\s\p{L}\p{N}]+[\r\n]*          # punctuation run (optional leading space)
| \s*[\r\n]+                        # newline runs
| \s+(?!\S)                         # trailing whitespace
| \s+                               # other whitespace runs
```

Input: one document (a plain string). Output: a list of pieces — letters
sticks to letters, digits split apart, punctuation clumps, whitespace is its
own piece. Example:

```
"Hello world! 2+2=4"  ->  ["Hello", " world", "!", " 2", "+", "2", "=", "4"]
```

## Phase 1 — count words (`words.bin`)

```python
# input: corpus files; output: words.bin
word_count = {}                     # dict: word_bytes -> u64

for file in corpus_files:           # parallel over files
    for (condition, instruction, response) in file:
        for text in (instruction, response):
            text = first_chars(text, 10_000)     # long-tail cut, see below
            text = NFC(text)                     # canonical unicode form
            for word in regex_split(text):       # pieces like " the", ",", "!"
                word_count[word] += 1

save_sorted(word_count, "words.bin")  # lexicographic order -> deterministic
```

Only *unique* words survive — 410M documents / 170 GiB of text compress to
288 MiB of statistics. The merge loop never touches the raw corpus again.

(`first_chars(text, 10_000)` cuts the tails of very long documents, tokenizer
training only. Measured impact: 0.37% of documents, ≈0.2% of characters.)

## Phase 2 — the merge loop (`merges_N.bin` checkpoints)

```python
# input: words.bin; output: merges (65264 rules), vocab (65536)
vocab  = special_tokens(31) + alphabet_of_bytes_present_in_corpus(241)
merges = []
tokens = {word: list_of_byte_symbols(word) for word in word_count}

pair_count   = count_all_adjacent_pairs(tokens, weighted_by=word_count)  # i32!
pair_to_word = index pairs -> words containing them

while len(vocab) < 65536:
    (a, b) = argmax(pair_count)        # ties: smaller token-id pair wins
    if pair_count[(a, b)] < 2: break   # min_frequency

    new = a + b                        # e.g. "Ġ" + "t" -> "Ġt"
    vocab.append(new); merges.append((a, b))

    for word in pair_to_word[(a, b)]:  # only words containing the pair
        update_neighbour_pair_counts(word)   # pairs touching (a,b) get
        tokens[word] = merge(tokens[word], a, b, new)  # remeasured; others untouched

    if len(merges) % 100 == 0:
        save_checkpoint(merges, vocab)  # merges_N.bin; resume replays from here
```

Checkpoint interval justification: the loop does **65264 merges** at ~250–500
merges/s on our hardware (measured: full loop in 1–4.5 min), so a checkpoint
every 100 merges costs a ~1 MB write every fraction of a second (~700 MB total
over a run) and bounds crash-recovery loss to <1 s of work. The format:

```
u32 magic "MBI1"
u64 n_vocab; n_vocab × { u32 len | bytes }      # vocab in id order
u64 n_merges; n_merges × { u32 id_a | u32 id_b } # merge rules in learning order
```

## Inference (encoding with the trained tokenizer)

```python
# input: tokenizer.json, raw text; output: token ids (uint16 — vocab is 65536)
merges_rank = {pair: rank for rank, pair in enumerate(merges)}  # rank = learning order

def encode(text):
    text = NFC(text)
    ids = []
    for piece in split_special_tokens(text):       # "<|direct|>" -> its own id
        if piece is special: ids.append(special_id(piece)); continue
        for word in regex_split(piece):
            symbols = byte_level_symbols(word)     # one symbol per byte
            while True:                            # greedy: always merge the
                best = argmin over adjacent pairs of merges_rank  # earliest-learned rule
                if no adjacent pair is a known merge: break
                symbols = merge(symbols, best)
            ids += [vocab_id(s) for s in symbols]
    return ids                                     # fits uint16: max id 65535
```

Key point: encoding applies the learned rules **in the order they were
learned** (lowest rank first), which is what makes it deterministic and equal
to training-time segmentation.

## Measured compression (this tokenizer, this corpus)

Tokens are uint16 (2 bytes). Bytes/token and storage ratio (2 bytes × tokens /
input bytes), measured per dataset:

| dataset | bytes/token | uint16 size / raw text size |
|---|---|---|
| SYNTH | 4.96 | 0.40 |
| flan | 3.73 | 0.54 |
| gsm8k | 3.49 | 0.57 |
| openmathinstruct2 | 2.71 | 0.74 |
| dmmath | 1.69 | **1.19** (short symbolic lines compress poorly — uint16 output is *larger* than the text) |

So "compression" ranges from 2.5× (prose) to negative (symbolic math). The
uint16 choice matters: with the authors' int32 fallback all ratios above
double. (Corpus-wide token counts: see `scripts/docs/token_counts.md`.)

## Incremental training (new data arrives — retrain from scratch?)

**Is BPE permutation-invariant?** The word *counts* are (order of documents
does not matter for phase 1). The *merge sequence* is not: it is a greedy
path — each merge depends on all previous ones.

**train(A) then continue on B  vs  train(A ∪ B) from scratch — same result?**
No. Once merges 1..N are learned from A, they are frozen; training on B can
only choose *subsequent* merges. A fresh run on A ∪ B would re-evaluate every
pair with combined counts and can pick different merges from the very first
one. Continuing on B therefore yields a vocabulary skewed toward A; how much
worse depends on how similar B is to A.

**Does the current code support it?** Partially: `words.bin` is just counts —
counts for new data can be added (a `--phase load` over the new files, then
summing the maps), and `--phase train` resumes from any merge checkpoint. So
"train on A, later add B's counts, continue merging" is implementable from the
existing checkpoints. What is *not* possible is revising already-committed
merges.

**Is it justified here?** No. A full retrain costs ~35 min / a couple of
dollars (see [benchmarks.md](benchmarks.md)); vocabulary drift between model
checkpoints costs far more than that. Recommendation: retrain from scratch on
A ∪ B. Incremental continuation is only worth it when retraining is truly
expensive and the vocab shift is acceptable.

## Parity-relevant quirks (replicated bug-for-bug)

For byte-identical output with the reference `tokenizers` crate, the merge
loop keeps: wrapping **i32** pair counts (pairs above 2.15B occurrences wrap
negative and are never merged — our corpus has such pairs), an alphabet of
only the 241 byte symbols actually present, and tie-breaks by token-id pair.
Details: [gotchas.md](gotchas.md).
