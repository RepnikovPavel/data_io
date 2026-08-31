#!/usr/bin/env python3
"""Aggregate count_tokens TSV output into token_counts.json + token_counts.md.

Input TSV rows: relpath<TAB>docs<TAB>tokens  (docs = instruction+response
counted separately; rows = docs/2).

Usage: aggregate_token_counts.py <counts.tsv> <out_dir> <tokenizer_path>
Writes <out_dir>/token_counts.json and <out_dir>/token_counts.md.
"""

import datetime
import json
import os
import sys


def dataset_key(rel):
    """Same mapping as count_tokens.cpp: registry dataset name from relpath."""
    parts = rel.split("/")
    if parts[0] == "data_clustered" and len(parts) >= 2:
        # registry name in scripts/docs/generate_docs.py is lowercase
        return "synth" if parts[1] == "SYNTH" else parts[1]
    stem = os.path.splitext(parts[-1])[0]
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "Platypus" \
            and stem.startswith("arb_"):
        return "arb"
    return stem


def main():
    tsv_path, out_dir, tokenizer_path = sys.argv[1], sys.argv[2], sys.argv[3]

    per_dataset = {}  # name -> [docs, tokens]
    with open(tsv_path) as f:
        for line in f:
            rel, docs, toks = line.rstrip("\n").split("\t")
            d = per_dataset.setdefault(dataset_key(rel), [0, 0])
            d[0] += int(docs)
            d[1] += int(toks)

    datasets = {}
    for name in sorted(per_dataset):
        docs, toks = per_dataset[name]
        rows = docs // 2
        datasets[name] = {
            "rows": rows,
            "tokens": toks,
            "tokens_per_row": round(toks / rows, 1) if rows else 0.0,
        }
    total_rows = sum(d["rows"] for d in datasets.values())
    total_tokens = sum(d["tokens"] for d in datasets.values())

    payload = {
        "generated": datetime.date.today().isoformat(),
        "tokenizer": tokenizer_path,
        "vocab_size": 65536,
        "note": "instruction and response counted as separate docs; "
                "no truncation, no sampling limits; rows = docs/2",
        "datasets": datasets,
        "total": {
            "rows": total_rows,
            "tokens": total_tokens,
            "tokens_per_row": round(total_tokens / total_rows, 1) if total_rows else 0.0,
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "token_counts.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Token counts over the transformed corpus",
        "",
        f"Tokenizer: `{tokenizer_path}` (vocab 65536). "
        "instruction and response counted as separate documents; "
        "no truncation, no sampling limits. "
        f"Generated {payload['generated']} by `scripts/count_tokens.sh`.",
        "",
        "| dataset | rows | tokens | tokens/row avg |",
        "|---|---|---|---|",
    ]
    for name, d in datasets.items():
        lines.append(f"| {name} | {d['rows']:,} | {d['tokens']:,} | "
                     f"{d['tokens_per_row']:.1f} |")
    t = payload["total"]
    lines.append(f"| **TOTAL** | **{t['rows']:,}** | **{t['tokens']:,}** | "
                 f"**{t['tokens_per_row']:.1f}** |")
    lines.append("")
    with open(os.path.join(out_dir, "token_counts.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"[aggregate] {len(datasets)} datasets, {total_rows:,} rows, "
          f"{total_tokens:,} tokens -> {out_dir}/token_counts.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
