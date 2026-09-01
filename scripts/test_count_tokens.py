#!/usr/bin/env python3
"""Self-test: C++ count_tokens encoder vs python `tokenizers` lib, per doc.

Builds a .tokbin from N random docs (default: 1000 rows of gsm8k_train =
2000 docs, instruction+response separately, plus fixed edge-case strings),
runs the C++ count_tokens --per-doc on it (must be invoked separately, in the
cpp image), and compares against python tokenizers counts for the same docs.

Two halves, selected by flag (so each can run in its own image):
  --write-tokbin <out.tokbin>   (clean image: needs only stdlib)
  --check <tokbin> <cpp_per_doc_output.txt>  (clean image: needs `tokenizers`)

Usage via scripts/test_count_tokens.sh (runs both halves + the C++ side).
"""

import argparse
import json
import random
import struct
import sys

TOKBIN_MAGIC = 0x324B4254  # "TKB2": data concatenated, lens array at EOF
HEADER = struct.Struct("<IQQQ")

EDGE_CASES = [
    "",  # empty
    " ",  # whitespace only
    "\n\n\n",
    "café naïve 数学 🎉 übermensch Æsir",
    "Привет, мир! Теорема Пифагора.",
    "x^{2} + y^{2} = z^{2}, \\frac{a}{b}, $\\sqrt{2}$",
    "def f(x):\n    return x * 2\n",
    "<|im_start|>special tokens inside text<|im_end|> and <|direct|> mid-word<|cot|>",
    "don't you're I'll we've they'd he's it's",  # contractions
    "  trailing   spaces  \n\n",
    "a" * 5000,  # long single word
    "Mixed CASE Words With 123 Numbers 4.56 and 0xFF",
    "e\u0301 vs é combining marks",  # NFC-relevant
    "ﬁ ligature and ① circled",  # more NFC-relevant
]


def write_tokbin(path, docs):
    lens = []
    with open(path, "wb") as f:
        f.write(HEADER.pack(TOKBIN_MAGIC, len(docs), 0, 0))
        for d in docs:
            b = d.encode("utf-8")
            lens.append(len(b))
            f.write(b)
        f.write(struct.pack(f"<{len(lens)}I", *lens))


def sample_docs(data_path, n_rows):
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]
    random.seed(0)
    docs = []
    for row in random.sample(rows, min(n_rows, len(rows))):
        docs.append(row["instruction"])
        docs.append(row["response"])
    return docs + EDGE_CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hdd2/datasets_text_transformed/HRM-Text/data/gsm8k_train.jsonl")
    ap.add_argument("--n-rows", type=int, default=1000)
    ap.add_argument("--tokenizer", default="/mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe/tokenizer.json")
    ap.add_argument("--write-tokbin")
    ap.add_argument("--check", nargs=2, metavar=("TOKBIN", "CPP_COUNTS"))
    args = ap.parse_args()

    if args.write_tokbin:
        docs = sample_docs(args.data, args.n_rows)
        write_tokbin(args.write_tokbin, docs)
        print(f"wrote {len(docs)} docs to {args.write_tokbin}")
        return 0

    if args.check:
        tokbin, cpp_path = args.check
        raw = open(tokbin, "rb").read()
        magic, n_docs, _, _ = HEADER.unpack(raw[: HEADER.size])
        assert magic == TOKBIN_MAGIC
        lens = struct.unpack(f"<{n_docs}I", raw[len(raw) - 4 * n_docs :])
        docs = []
        pos = HEADER.size
        for ln in lens:
            docs.append(raw[pos : pos + ln].decode("utf-8"))
            pos += ln
        cpp_counts = [int(x) for x in open(cpp_path).read().split()]
        assert len(cpp_counts) == n_docs, (len(cpp_counts), n_docs)

        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
        py_counts = [len(tok.encode(d, add_special_tokens=True).ids) for d in docs]

        mismatches = [(i, c, p, docs[i][:80]) for i, (c, p) in enumerate(zip(cpp_counts, py_counts)) if c != p]
        total_cpp = sum(cpp_counts)
        total_py = sum(py_counts)
        print(f"docs={n_docs}  total_cpp={total_cpp}  total_py={total_py}")
        if mismatches:
            print(f"MISMATCH on {len(mismatches)} docs; first 10:")
            for i, c, p, preview in mismatches[:10]:
                print(f"  doc {i}: cpp={c} py={p} {preview!r}")
            return 1
        print("EXACT MATCH on all docs")
        return 0

    ap.error("need --write-tokbin or --check")


if __name__ == "__main__":
    sys.exit(main())
