#!/usr/bin/env python3
"""Bar plot of the token distribution per domain over the transformed corpus.

Reads scripts/docs/token_counts.json (produced by scripts/count_tokens.sh) and
the dataset->domain mapping META from scripts/docs/generate_docs.py, sums
tokens per domain and renders a LaTeX-style bar chart sorted by descending
token count: domain labels rotated 45 deg on the x axis, total tokens on the
y axis and above each bar, formatted K/M/B/T.

Usage: plot_domain_tokens.py [token_counts.json] [out.png] [--linear]
                             [--width W] [--height H] [--dpi D] [--wrap N]
                             [--color C] [--no-title]

Run via scripts/plot_domain_tokens.sh (docker, hrm_text_clean_image).
"""

import argparse
import json
import os
import sys
import textwrap

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
sys.path.insert(0, DOCS_DIR)
from generate_docs import META  # noqa: E402  name -> (created, domain, evidence)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_COUNTS = os.path.join(DOCS_DIR, "token_counts.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "assets", "domain_token_distribution.png")


def fmt_tokens(n):
    """Human-readable token count: 137.9B, 22.3B, 648M, 54K."""
    for suffix, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + suffix
    return str(int(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("counts", nargs="?", default=DEFAULT_COUNTS,
                        help="token_counts.json path")
    parser.add_argument("out", nargs="?", default=DEFAULT_OUT,
                        help="output image path (png/pdf)")
    parser.add_argument("--linear", action="store_true",
                        help="linear y axis (log by default: the domain token "
                             "counts span 6 orders of magnitude)")
    parser.add_argument("--width", type=float, default=17.0)
    parser.add_argument("--height", type=float, default=7.5)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--wrap", type=int, default=24,
                        help="wrap domain labels at N chars (0 = no wrap)")
    parser.add_argument("--color", default="#4C72B0")
    parser.add_argument("--label-size", type=float, default=8.5)
    parser.add_argument("--no-title", action="store_true")
    args = parser.parse_args()

    with open(args.counts) as f:
        counts = json.load(f)

    per_domain = {}
    for name, d in counts["datasets"].items():
        domain = META[name][1]
        per_domain[domain] = per_domain.get(domain, 0) + d["tokens"]
    items = sorted(per_domain.items(), key=lambda kv: kv[1], reverse=True)
    labels = [textwrap.fill(k, args.wrap, break_long_words=False,
                            break_on_hyphens=False) if args.wrap else k
              for k, _ in items]
    values = [v for _, v in items]
    total = counts["total"]["tokens"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    bars = ax.bar(range(len(values)), values, width=0.7,
                  color=args.color, edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=args.label_size)
    ax.set_ylabel("tokens")
    ax.set_xlabel("domain")
    if not args.no_title:
        ax.set_title("HRM-Text training corpus — tokens per domain "
                     f"(total {fmt_tokens(total)})", fontsize=13, pad=12)

    if not args.linear:
        ax.set_yscale("log")
        ax.set_ylim(top=max(values) * 6)
    else:
        ax.set_ylim(0, max(values) * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_tokens(v)))
    ax.grid(axis="y", which="major", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar_label(bars, labels=[fmt_tokens(v) for v in values],
                 padding=3, fontsize=8.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot] {len(items)} domains, {total:,} tokens total -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
