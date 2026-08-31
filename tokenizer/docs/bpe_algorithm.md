# BPE training — the algorithm in pseudocode

↑ [index](README.md)

For a reader who has never seen BPE (Byte Pair Encoding). This describes
exactly what `train_tokenizer_iter` / `tokenizer_cpp` implement, at the level
of data structures. No ML background needed: BPE training is just counting
and merging symbols.

## Input / output

**Input:** a corpus of text documents (in this project: 410M documents,
170 GiB, after the sampling limits of `prefix_config.yaml`), plus parameters:
`vocab_size` (65536), `min_frequency` (2), a list of special tokens, a
pre-tokenizer regex.

**Output:** `tokenizer.json` —
- `vocab`: map `token string -> id` (65536 entries; ids 0..30 are the special
  tokens, then the byte alphabet, then merged tokens in merge order),
- `merges`: ordered list of 65264 pairs `[A, B]` — the *rules* learned below.
- plus the fixed machinery: NFC normalizer, pre-tokenizer regex, byte-level
  decoder.

At *inference* time the learned tokenizer is applied greedily: split text by
the same regex, map to bytes, then repeatedly apply the merge rules **in the
order they were learned** (rule 1 first). The merge order is what makes the
encoding deterministic.

## Key idea

Start with a vocabulary of single bytes (256 symbols) + the 31 special tokens.
Repeatedly find the most frequent *adjacent pair* of symbols in the corpus and
merge it into one new symbol. Stop when the vocabulary reaches `vocab_size`.
Frequent substrings become single tokens; rare text stays split into bytes.

## Phase 1 — build word statistics (the "load" phase, `words.bin`)

```text
word_count = {}                       # map: tuple of byte-symbols -> u64

for file in input_files:              # parquet/jsonl, in parallel
    for (condition, instruction, response) in file:
        for text in (instruction, response):
            text = first_10000_chars(text)
            text = NFC(text)                       # normalize unicode
            for word in regex_split(text):         # GPT-2-style regex:
                                                   # words, numbers, punctuation,
                                                   # whitespace runs
                symbols = byte_level_encode(word)  # each byte -> one symbol
                                                   # (space -> "Ġ", etc.)
                word_count[symbols] += 1

save words.bin                        # {word_symbols: count}, sorted
```

Notes:
- The corpus is never kept in memory in this phase — only *unique* words with
  counts survive (12.1M unique words for our corpus).
- `first_10000_chars` (`--truncate-len`, default 10 000) cuts only the tails of
  very long documents, and only for tokenizer *training* (the model-training
  data is not truncated). Measured on our corpus: 0.37% of documents affected
  (mostly long SYNTH answers), ≈0.2% of characters lost. Rationale: BPE merge
  statistics need frequent patterns, not rare long tails; 10k chars ≈ well
  beyond the 4096-token training context.
- Everything downstream works on **weighted words**: a word with count 1000
  counts as 1000 occurrences when we count pairs.

## Phase 2 — the merge loop (the "train" phase, `merges_N.bin`)

```text
vocab  = special_tokens (ids 0..30) + alphabet (byte symbols present in corpus)
merges = []                            # ordered merge rules
tokens = { word: list_of_symbols }     # every unique word as a symbol sequence

# initial pair statistics over all words, weighted by count
pair_count = {}
pair_to_words = {}                     # pair -> set of words containing it
for word, cnt in word_count:
    for each adjacent pair (a, b) in tokens[word]:
        pair_count[(a, b)] += cnt      # NOTE: i32 with wraparound (see gotchas.md)
        pair_to_words[(a, b)].add(word)

while len(vocab) < vocab_size:         # 65536 -> exactly 65264 merges
    # pick the most frequent pair; ties: smaller token-id pair wins
    (a, b) = argmax pair_count         # skip if count < min_frequency (2)

    new = concat(a, b)                 # the merged symbol, e.g. "Ġ" + "t" = "Ġt"
    vocab.add(new, id = next_id)
    merges.append([a, b])              # the rule, in learning order

    # apply the merge ONLY to words that contain the pair (incremental update)
    for word in pair_to_words[(a, b)]:
        for each adjacent position where (a, b) occurs in tokens[word]:
            # a pair at the boundary affects its neighbours:
            pair_count[(left, a)] -= cnt       # old neighbour pair dies
            pair_count[(b, right)] -= cnt
            pair_count[(left, new)] += cnt     # new neighbour pairs appear
            pair_count[(new, right)] += cnt
        tokens[word] = merge tokens[word] at (a, b) -> new

    if merge_index % 100 == 0: checkpoint(merges, vocab)   # crash-safe resume
```

After the loop: write `tokenizer.json` (vocab + merges + normalizer /
pre-tokenizer / decoder config) and `config.json`.

## Worked micro-example

Corpus: `"the cat"` ×5, `"the car"` ×3. After phase 1 (per word, byte-level
symbols; `Ġ` = space):

```text
word_count:  Ġthe:(Ġ,t,h,e)x8   Ġcat:(Ġ,c,a,t)x5   Ġcar:(Ġ,c,a,r)x3
```

Merge loop:

| step | most frequent pair | count | new token | merges |
|---|---|---|---|---|
| 1 | (Ġ, t) | 8 | `Ġt` | `[["Ġ","t"]]` |
| 2 | (Ġt, h) | 8 | `Ġth` | + `[["Ġt","h"]]` |
| 3 | (Ġth, e) | 8 | `Ġthe` | + `[["Ġth","e"]]` |
| 4 | (Ġ, c) | 8 | `Ġc` | … |
| 5 | (Ġc, a) | 8 | `Ġca` | … |
| 6 | (c, a)... pair counts drop below neighbours; continues until vocab_size or min_frequency |

Encoding `"the car"` afterwards: split → `the`, `Ġcar`; apply merge rules in
order → `Ġthe`, `Ġcar` — two tokens. A rare word like `"zebra"` gets no
merges and falls back to bytes: `z e b r a`.

## Complexity / cost notes

- Phase 1 is O(corpus size), parallel over files; memory = unique words.
- Phase 2 with the incremental update is O(tokens) once + O(words touching
  the pair) per merge — this is why our trainers finish 65264 merges in
  ~1 minute. A naive re-count over the whole corpus per merge would take days.
- `pair_to_words` is the index that makes merges cheap; checkpoints every 100
  merges make the loop resumable (resume replays merges onto words and
  recounts pairs).
- Exact parity with the reference (`tokenizers` crate) requires replicating
  its quirks: wrapping **i32** pair counts (pairs above 2.1B occurrences
  overflow and never get merged — see [gotchas.md](gotchas.md)), corpus-present
  alphabet (241 of 256 byte symbols on our corpus), and tie-break by token-id
  pair.
