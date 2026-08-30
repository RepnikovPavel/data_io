# Special tokens

← [README.md](README.md)

31 added tokens, ids 0..30, all with `special: true`, `single_word: false`,
`lstrip: false`, `rstrip: false`, `normalized: false`. Names mostly mirror
Qwen2.5/Qwen3 tokenizer conventions; ids 1–4 are this project's addition.
Ids are fixed by the order of the `--special-tokens` list in
`tokenizer/src/bin/train_tokenizer.rs` (same default in
`train_tokenizer_iter.rs`).

| id | token | purpose in THIS project | origin |
|---|---|---|---|
| 0 | `<\|PAD\|>` | padding — note: currently unused (`sample_tokenized.py` never emits it) | GPT-2/Qwen convention |
| 1 | `<\|direct\|>` | condition marker: answer-only rows | **this project** (training-mix tags, see `scripts/docs/README.md` condition section) |
| 2 | `<\|cot\|>` | condition marker: chain-of-thought rows | this project |
| 3 | `<\|noisy\|>` | condition marker: unverified machine-generated/scraped rows | this project |
| 4 | `<\|synth\|>` | condition marker: synthetic rows | this project |
| 5 | `<\|endoftext\|>` | document/EOS marker | GPT-2/Qwen convention |
| 6 | `<\|im_start\|>` | used as **BOQ** (begin of question) — `main.rs --boq` default | ChatML (Qwen chat models) |
| 7 | `<\|im_end\|>` | used as **EOQ** (end of question) — `main.rs --eoq` default | ChatML (Qwen chat models) |
| 8 | `<\|object_ref_start\|>` | — (unused here; see quirk below) | Qwen vision: object reference |
| 9 | `<\|object_ref_end\|>` | — | Qwen vision: object reference |
| 10 | `<\|box_start\|>` | — | Qwen vision: bounding box |
| 11 | `<\|box_end\|>` | — | Qwen vision: bounding box |
| 12 | `<\|quad_start\|>` | — | Qwen vision: quad |
| 13 | `<\|quad_end\|>` | — | Qwen vision: quad |
| 14 | `<\|vision_start\|>` | — | Qwen vision |
| 15 | `<\|vision_end\|>` | — | Qwen vision |
| 16 | `<\|vision_pad\|>` | — | Qwen vision |
| 17 | `<\|image_pad\|>` | — | Qwen VL |
| 18 | `<\|video_pad\|>` | — | Qwen VL |
| 19 | `<\|fim_prefix\|>` | — | fill-in-the-middle, Qwen code models |
| 20 | `<\|fim_middle\|>` | — | fill-in-the-middle |
| 21 | `<\|fim_suffix\|>` | — | fill-in-the-middle |
| 22 | `<\|fim_pad\|>` | — | fill-in-the-middle |
| 23 | `<\|repo_name\|>` | — | code file structure, Qwen code models |
| 24 | `<\|file_sep\|>` | — | code file structure |
| 25 | `<tool_call>` | — | agent/tool-use markup, Qwen3 |
| 26 | `</tool_call>` | — | Qwen3 |
| 27 | `<tool_response>` | — | Qwen3 |
| 28 | `</tool_response>` | — | Qwen3 |
| 29 | `<think>` | — | reasoning-trace delimiters, Qwen3 convention |
| 30 | `</think>` | — | Qwen3 |

## The `--conditions` quirk in `main.rs` (tokenization stage)

`tokenize_data` (`tokenizer/src/main.rs`) emits, per row:
`BOQ + condition tokens + instruction + EOQ + response + EOA`.
Its defaults are tuned to the stock Qwen3 tokenizer, which has NO dedicated
condition tokens:

- `--conditions` defaults to `direct=<|object_ref_start|>,
  cot=<|object_ref_end|>, noisy=<|quad_start|>, synth=<|quad_end|>` — i.e. it
  **reuses Qwen vision-loc tokens as condition markers**.
- `--boq`/`--eoq` default to `<|im_start|>`/`<|im_end|>`, `--eoa` defaults to
  `<|box_end|>`.

With OUR trained tokenizer the dedicated tokens exist, so pass them
explicitly or the condition markers will silently be the vision tokens:

```sh
tokenize_data <DIRS>... -o <OUT_DIR> -t /mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe \
  --conditions 'direct=<|direct|>,cot=<|cot|>,noisy=<|noisy|>,synth=<|synth|>' \
  --eoa '<|endoftext|>'
```

(`--boq`/`<|im_start|>` and `--eoq`/`<|im_end|>` defaults are fine — those ids
6/7 exist with the same meaning.)
