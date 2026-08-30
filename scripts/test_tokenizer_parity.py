#!/usr/bin/env python3
"""Tokenizer parity check: compare N candidate tokenizer.json files against a reference.

Reports per candidate:
  * structural equality (normalizer / pre_tokenizer / post_processor /
    decoder / added_tokens / model fields except vocab+merges)
  * vocab size and token-set overlap
  * merge list: length, common prefix, total equal positions
  * segmentation equality (token STRINGS, not ids) on fixed test strings
    (English, Cyrillic, code, LaTeX, special tokens, long text) plus a random
    sample of records from a real data file (instruction + response)

Exit code: 0 if all candidates load and are structurally equal to the
reference; 1 otherwise. Vocab/merge/segmentation differences are reported
but do not affect the exit code.

Usage:
  test_tokenizer_parity.py <reference.json> <candidate.json> [candidate2.json ...]
      [--data /path/to/gsm8k_train.jsonl] [--n-samples 1000]
"""

import argparse
import json
import random
import sys

from tokenizers import Tokenizer

FIXED_STRINGS = [
    # English
    "The answer is 42. Let me think carefully about this problem.",
    "What is 2 plus 2? Please explain step by step, carefully and slowly.",
    "It's a well-known fact that we're testing tokenizers; they'll split contractions.",
    # Cyrillic
    "Привет, мир! Это проверка токенизатора на кириллице.",
    "Теорема Пифагора: квадрат гипотенузы равен сумме квадратов катетов.",
    # code
    "def f(x):\n    return x * 2  # comment\n",
    "int main() { printf(\"%d\\n\", 42); return 0; }",
    "fn main() { let v = vec![1, 2, 3]; println!(\"{:?}\", v); }",
    # LaTeX / math
    "x^{2} + y^{2} = z^{2}, \\frac{a}{b}, $\\sqrt{2} \\approx 1.4142$",
    "\\begin{align} E &= mc^2 \\\\ F &= ma \\end{align}",
    # special tokens (encoded with add_special_tokens=True)
    "<|im_start|>What is 1+1?<|im_end|>",
    "<|direct|>some row<|endoftext|>",
    "<think>reasoning trace</think><tool_call>{}</tool_call>",
    # whitespace / control
    "  leading spaces\n\nnewlines\t tabs  \r\n crlf",
    "trailing spaces   \n",
    # unicode / emoji / CJK
    "café naïve 数学 🎉 übermensch Æsir",
    "日本語のテキストと中文文本混合 test 123",
    # digits / misc
    "0123456789 3.14159265358979 1e-10 0xDEADBEEF",
    # long text
    ("The quick brown fox jumps over the lazy dog. " * 40).strip(),
    ("Step 1: consider the problem. Step 2: apply the theorem. " * 30).strip(),
]


def structural_sections(d):
    m = {k: v for k, v in d["model"].items() if k not in ("vocab", "merges")}
    return {
        "normalizer": d.get("normalizer"),
        "pre_tokenizer": d.get("pre_tokenizer"),
        "post_processor": d.get("post_processor"),
        "decoder": d.get("decoder"),
        "added_tokens": d.get("added_tokens"),
        "model_fields": m,
    }


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizers", nargs="+", help="reference first, then candidates")
    ap.add_argument("--data", default="/mnt/hdd2/datasets_text_transformed/HRM-Text/data/gsm8k_train.jsonl")
    ap.add_argument("--n-samples", type=int, default=1000)
    args = ap.parse_args()

    if len(args.tokenizers) < 2:
        ap.error("need at least a reference and one candidate")

    # Sampled real records (shared across candidates)
    samples = []
    try:
        with open(args.data) as f:
            rows = [json.loads(line) for line in f]
        random.seed(0)
        for row in random.sample(rows, min(args.n_samples, len(rows))):
            samples.append(row["instruction"])
            samples.append(row["response"])
        print(f"sampled {len(samples)} documents from {args.data}")
    except FileNotFoundError:
        print(f"WARNING: {args.data} not found -- skipping record sampling")

    ref_path = args.tokenizers[0]
    ref_json = json.load(open(ref_path))
    ref_tok = Tokenizer.from_file(ref_path)
    ref_struct = structural_sections(ref_json)
    ref_vocab = ref_json["model"]["vocab"]
    ref_merges = [tuple(m) for m in ref_json["model"]["merges"]]
    ref_seg_fixed = [ref_tok.encode(s, add_special_tokens=True).tokens for s in FIXED_STRINGS]
    ref_seg_data = [ref_tok.encode(s, add_special_tokens=False).tokens for s in samples]

    print(f"\nreference: {ref_path}  (vocab {len(ref_vocab)}, merges {len(ref_merges)})")
    print(f"fixed test strings: {len(FIXED_STRINGS)}")

    ok = True
    for cand_path in args.tokenizers[1:]:
        print("\n" + "=" * 100)
        print(f"candidate: {cand_path}")
        try:
            cand_json = json.load(open(cand_path))
            cand_tok = Tokenizer.from_file(cand_path)
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            ok = False
            continue

        cand_struct = structural_sections(cand_json)
        cand_vocab = cand_json["model"]["vocab"]
        cand_merges = [tuple(m) for m in cand_json["model"]["merges"]]

        # structural
        struct_diffs = [k for k in ref_struct if ref_struct[k] != cand_struct[k]]
        struct_ok = not struct_diffs
        ok &= struct_ok

        # vocab
        overlap = len(set(ref_vocab) & set(cand_vocab))
        # merges
        prefix = common_prefix(ref_merges, cand_merges)
        equal_pos = sum(1 for x, y in zip(ref_merges, cand_merges) if x == y)

        # segmentation (token strings, not ids)
        fixed_mismatch = sum(
            1 for s, r in zip(FIXED_STRINGS, ref_seg_fixed)
            if cand_tok.encode(s, add_special_tokens=True).tokens != r
        )
        data_mismatch = sum(
            1 for s, r in zip(samples, ref_seg_data)
            if cand_tok.encode(s, add_special_tokens=False).tokens != r
        )
        seg_ok = fixed_mismatch == 0 and data_mismatch == 0
        # Segmentation differences are reported but do not fail the check.

        print(f"  {'OK  ' if struct_ok else 'FAIL'} structure"
              + ("" if struct_ok else f" (differ: {', '.join(struct_diffs)})"))
        print(f"       vocab: {len(cand_vocab)} (ref {len(ref_vocab)}), "
              f"token set overlap {overlap}/{len(ref_vocab)} ({overlap/len(ref_vocab):.2%})")
        print(f"       merges: {len(cand_merges)} (ref {len(ref_merges)}), "
              f"common prefix {prefix}, equal positions {equal_pos}/{len(ref_merges)}")
        print(f"  {'OK  ' if seg_ok else 'DIFF'} segmentation: fixed strings "
              f"{len(FIXED_STRINGS)-fixed_mismatch}/{len(FIXED_STRINGS)} equal, "
              f"sampled records {len(samples)-data_mismatch}/{len(samples)} equal")

    print("\n" + "=" * 100)
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
