"""Generate documentation for the data-cleaning pipeline.

For every registered dataset this script writes scripts/docs/<name>.md with:
purpose (informed by the dataset card), raw storage layout with precise types
(derived from the actual files), the transformed output contract, real row
counts, keyword tables (categorical fields with real value counts), and real
before/after example records rendered in the storage's own format (tables for
parquet/arrow rows, raw lines for JSONL/txt/CSV). Also writes
scripts/docs/README.md as the navigable index.

Datasets whose output does not exist yet are skipped (re-runnable).
Run inside hrm_text_clean_image:

    docker run --rm -v <repo>:/workspace \
        -v /mnt/hdd2/datasets_text:/mnt/hdd2/datasets_text \
        -v /mnt/hdd2/datasets_text_transformed:/mnt/hdd2/datasets_text_transformed \
        -w /workspace -e HF_HOME=/mnt/hdd2/datasets_text/.hf_cache \
        -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 \
        hrm_text_clean_image python scripts/docs/generate_docs.py
"""

import argparse
import datetime
import glob as globmod
import json
import os
import string
import sys
import tarfile
from collections import Counter

# Repo root on sys.path so `utils` is importable regardless of CWD.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

os.environ.setdefault("HF_HOME", "/mnt/hdd2/datasets_text/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_ROOT = "/mnt/hdd2/datasets_text_transformed/HRM-Text"
RAW_ROOT = "/mnt/hdd2/datasets_text"
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
TRUNC = 600
MAX_KEYWORD_EXAMPLES = 8
HIGH_CARDINALITY = 20  # above this many values: counts table + 2 examples only

CONDITION_MEANING = {
    "direct": ("short answer only, no reasoning shown",
               "controls how much 'answer-only' behavior the mix teaches"),
    "cot": ("response contains the full chain-of-thought / worked solution",
            "controls how much explicit reasoning the mix teaches"),
    "noisy": ("content was machine-generated or scraped and not fully verified",
              "flags rows the trainer may want to down-weight for quality"),
    "synth": ("synthetically generated (model- or program-produced) data",
              "separates synthetic from human/curated data in the mix"),
}


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def truncate_str(s):
    if isinstance(s, str) and len(s) > TRUNC:
        return s[:TRUNC] + f"… [truncated, {len(s)} chars total]"
    return s


def truncate(value):
    """Recursively truncate long strings for display."""
    if isinstance(value, str):
        return truncate_str(value)
    if isinstance(value, list):
        return [truncate(v) for v in value]
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def cell(value):
    """Render one value as a markdown table cell."""
    if value is None:
        return "∅ (null)"
    if isinstance(value, (dict, list)):
        value = json.dumps(truncate(value), ensure_ascii=False)
    else:
        value = truncate_str(str(value))
    return value.replace("|", "\\|").replace("\r", "").replace("\n", "⏎")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def row_as_table(columns, record):
    """One tabular record (parquet/arrow row) as a markdown table."""
    return md_table(columns, [[cell(record.get(c)) for c in columns]])


def fence(text, lang="text"):
    n = 4
    while "`" * n in text:
        n += 1
    ticks = "`" * n
    return f"{ticks}{lang}\n{text}\n{ticks}"


def jsonl_line(record):
    """Record rendered as one raw JSONL line (valid JSON, fields truncated)."""
    return json.dumps(truncate(record), ensure_ascii=False)


def fmt_rows(n):
    return f"{n:,}" if isinstance(n, int) else "?"


def expand_outputs(patterns):
    files = []
    for pat in patterns:
        full = os.path.join(OUT_ROOT, pat)
        if globmod.has_magic(full):
            files.extend(sorted(globmod.glob(full)))
        elif os.path.isfile(full):
            files.append(full)
    return sorted(set(files))


# ---------------------------------------------------------------------------
# type resolvers — types are read from the actual files, never guessed
# ---------------------------------------------------------------------------

def hf_type_str(feature):
    """datasets feature -> precise type string."""
    import datasets
    if isinstance(feature, datasets.Value):
        t = feature.dtype
        if t in ("string", "large_string"):
            return f"UTF-8 text (arrow `{t}`)"
        if t.startswith("int") or t.startswith("uint"):
            return f"integer (arrow `{t}`)"
        if t.startswith("float"):
            return f"float (arrow `{t}`)"
        if t == "bool":
            return "boolean (arrow `bool`)"
        return f"arrow `{t}`"
    seq_types = tuple(t for t in (getattr(datasets, "List", None),
                                  getattr(datasets, "Sequence", None)) if t)
    if isinstance(feature, seq_types):
        return f"arrow list — {hf_type_str(feature.feature)}"
    if isinstance(feature, dict):
        inner = ", ".join(f"{k}: {hf_type_str(v)}" for k, v in feature.items())
        return "arrow struct {" + inner + "}"
    if isinstance(feature, datasets.Image):
        return "image (arrow `struct<bytes, path>`, PIL-decoded)"
    return str(feature)


def parquet_col_types(path):
    """{column: precise type} from the parquet file's physical+arrow schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    phys = {pf.schema.column(i).name: pf.schema.column(i).physical_type
            for i in range(len(pf.schema.names))}
    out = {}
    for f in pf.schema_arrow:
        p = phys.get(f.name, "?")
        t = f.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            out[f.name] = (f"parquet {p} (logical String) → arrow `{t}` — "
                           "UTF-8 text")
        elif pa.types.is_integer(t):
            out[f.name] = f"parquet {p} → arrow `{t}` — integer"
        elif pa.types.is_floating(t):
            out[f.name] = f"parquet {p} → arrow `{t}` — float"
        elif pa.types.is_boolean(t):
            out[f.name] = f"parquet {p} → arrow `{t}` — boolean"
        else:
            out[f.name] = f"parquet {p} → arrow `{t}`"
    return out


def json_type_str(value):
    if isinstance(value, str):
        return "JSON string — UTF-8 text"
    if isinstance(value, bool):
        return "JSON boolean"
    if isinstance(value, (int, float)):
        return "JSON number"
    if isinstance(value, list):
        return "JSON array"
    if isinstance(value, dict):
        return "JSON object"
    return "JSON null"


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------

def count_jsonl(path, cond_values):
    """One pass over a jsonl file: line count + per-condition byte counts."""
    pats = {v: b'"condition":"' + v.encode() + b'"' for v in cond_values}
    n_lines, counts = 0, Counter()
    with open(path, "rb") as f:
        tail = b""
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            n_lines += chunk.count(b"\n")
            buf = tail + chunk
            # count patterns only in complete lines; patterns never contain
            # '\n', so this is exact across read boundaries
            complete, sep, tail = buf.rpartition(b"\n")
            region = complete + sep
            for v, p in pats.items():
                counts[v] += region.count(p)
    for v, p in pats.items():  # final line without trailing newline
        counts[v] += tail.count(p)
    return n_lines, counts


def parquet_num_rows(path):
    import pyarrow.parquet as pq
    return pq.read_metadata(path).num_rows


def gather_condition_counts(entry, files, total):
    """-> (counts: Counter, note: str)."""
    spec = entry["condition_counts"]
    kind = spec["kind"]
    if kind == "jsonl_scan":
        counts = Counter()
        for path in files:
            _, c = count_jsonl(path, entry["conditions"])
            counts.update(c)
        other = total - sum(counts.values())
        note = "exact counts (full scan of the jsonl file)"
        if other:
            note += f"; {fmt_rows(other)} rows carry a condition value not " \
                    "listed above"
        return counts, note
    if kind == "file_map":
        fn = spec["fn"]
        counts = Counter()
        for path in files:
            n = count_jsonl(path, [])[0] if entry["out_format"] == "jsonl" \
                else parquet_num_rows(path)
            counts[fn(os.path.basename(path))] += n
        return counts, "exact counts (condition is constant per file; " \
                       "rows summed from file metadata/line counts)"
    if kind == "col_sample":
        import pyarrow.parquet as pq
        n_files = min(spec["files"], len(files))
        counts = Counter()
        sampled_rows = 0
        for path in files[:n_files]:
            col = pq.read_table(path, columns=["condition"]).column("condition")
            sampled_rows += len(col)
            counts.update(x for x in col.to_pylist() if x is not None)
        if n_files < len(files):
            note = (f"exact within the first {n_files} of {len(files)} files "
                    f"({fmt_rows(sampled_rows)} of {fmt_rows(total)} rows); "
                    "files are homogeneous, so the mix is representative")
        else:
            note = "exact counts (condition column read in full)"
        return counts, note
    raise ValueError(kind)


def file_map_counts(files, fmt, fn):
    """Group output rows by fn(filename) — for input-side keywords like tier."""
    counts = Counter()
    for path in files:
        n = count_jsonl(path, [])[0] if fmt == "jsonl" else parquet_num_rows(path)
        counts[fn(os.path.basename(path))] += n
    return counts, "exact (rows summed per file from metadata/line counts)"


# ---------------------------------------------------------------------------
# example collection
# ---------------------------------------------------------------------------

def jsonl_condition_examples(files, cond_values, max_bytes=1 << 26):
    """First record per condition value (early exit once all found)."""
    wanted = set(cond_values)
    found = {}
    for path in files:
        read = 0
        with open(path, "rb") as f:
            for line in f:
                read += len(line)
                if read > max_bytes:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                v = rec.get("condition")
                if v in wanted and v not in found:
                    found[v] = (path, rec)
                if len(found) == len(wanted):
                    return found
    return found


def parquet_condition_examples(files, cond_values, max_files=24):
    """First row per condition value, scanning at most max_files files."""
    import pyarrow.parquet as pq
    wanted = set(cond_values)
    found = {}
    for path in files[:max_files]:
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        for batch in pf.iter_batches(batch_size=64):
            for rec in batch.to_pylist():
                v = rec.get("condition")
                if v in wanted and v not in found:
                    found[v] = (path, rec, cols)
            if len(found) == len(wanted):
                return found
        if len(found) == len(wanted):
            break
    return found


def flan_example_files(files):
    """One file per subset prefix — finds both conditions with 2 reads."""
    by_prefix = {}
    for f in files:
        by_prefix.setdefault(os.path.basename(f).split("__", 1)[0], f)
    return sorted(by_prefix.values())


# ---------------------------------------------------------------------------
# raw example loaders — each returns (label, layout, record, render_hint)
# render_hint: "table" (tabular row) or "raw" (verbatim text/json block)
# ---------------------------------------------------------------------------

def _hf_dataset(spec, collapse=True):
    from utils import load_local_dataset
    ds = load_local_dataset(spec["repo"], spec.get("config"),
                            split=spec.get("split"))
    if collapse and isinstance(ds, dict):  # DatasetDict (no split given)
        ds = ds[sorted(ds.keys())[0]]
    return ds


def load_hf(spec):
    ds = _hf_dataset(spec)
    row = ds[0]
    match = spec.get("match")
    if match:
        key, want = match
        for i in range(min(len(ds), spec.get("match_scan", 20000))):
            if ds[i][key] == want:
                row = ds[i]
                break
    fields = spec["fields"]
    label = f"HF `{spec['repo']}`"
    if spec.get("config"):
        label += f" (config `{spec['config']}`)"
    if spec.get("split"):
        label += f", split `{spec['split']}`"
    layout = f"One row of {label} (an arrow table in the local HF cache), " \
             "shown as a table with the columns the transform reads:"
    return (label, layout, {k: row[k] for k in fields}, "table",
            {"features": ds.features, "ds": ds})


def load_hf_flan(spec):
    """First row of one subset's first parquet + the matching output file."""
    import pyarrow.parquet as pq
    subset_dir = os.path.join(spec["input_dir"], spec["subset"])
    first_file = sorted(globmod.glob(os.path.join(subset_dir, "*.parquet")))[0]
    pf = pq.ParquetFile(first_file)
    row = next(pf.iter_batches(batch_size=1)).to_pylist()[0]
    label = f"`{spec['input_dir']}/{spec['subset']}/{os.path.basename(first_file)}`"
    layout = f"One row of {label} (parquet table), shown with the columns " \
             "the transform reads:"
    data = {k: row[k] for k in spec["fields"]}
    safe = "".join(c if c in set(string.ascii_letters + string.digits + "_-. ")
                   else "_" for c in row["_task_name"])
    out_file = os.path.join(OUT_ROOT, spec["out_dir"],
                            f"{spec['subset']}__{safe}.parquet")
    extra = {"types": parquet_col_types(first_file),
             "paired_output": out_file if os.path.isfile(out_file) else None}
    return label, layout, data, "table", extra


def load_json_list_file(spec):
    """First element of a JSON file containing a list of objects. If
    `prefer` is set, scan the first few files for a record where that key
    is truthy (e.g. a non-empty solution)."""
    paths = sorted(globmod.glob(spec["glob"]))
    row, path = None, paths[0]
    for cand in paths[:8]:
        with open(cand, "rb") as f:
            data = json.loads(f.read())
        first = data[0] if isinstance(data, list) else data
        if row is None:
            row, path = first, cand
        if spec.get("prefer") and first.get(spec["prefer"]):
            row, path = first, cand
            break
    label = f"`{path}`"
    layout = f"First element of the JSON array in {label} (UTF-8 JSON file):"
    return label, layout, {k: row.get(k) for k in spec["fields"]}, "raw_json", \
        {"path": path}


def load_khan(spec):
    import orjson
    path = sorted(globmod.glob(os.path.join(spec["input_dir"], "**", "*.json"),
                               recursive=True))[0]
    with open(path, "rb") as f:
        row = orjson.loads(f.read())
    label = f"`{path}`"
    layout = f"Full content of {label} (one JSON object per file, UTF-8):"
    return label, layout, {k: row[k] for k in spec["fields"]}, "raw_json", \
        {"path": path}


def load_txt_pairs(spec):
    path = sorted(globmod.glob(os.path.join(spec["input_dir"], spec["subset"],
                                            "*.txt")))[0]
    with open(path) as f:
        lines = [next(f).rstrip("\n") for _ in range(4)]
    label = f"`{path}`"
    layout = f"First 4 lines of {label} (UTF-8 text; line 2k = question, " \
             "line 2k+1 = answer):"
    record = {"question_line": lines[0], "answer_line": lines[1]}
    return label, layout, record, "raw_lines", {"lines": lines, "path": path}


def load_parquet_file(spec):
    import pyarrow.parquet as pq
    path = sorted(globmod.glob(spec["glob"]))[0]
    row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
    label = f"`{path}`"
    layout = f"One row of {label} (parquet table), shown with the columns " \
             "the transform reads:"
    return label, layout, {k: row[k] for k in spec["fields"]}, "table", \
        {"types": parquet_col_types(path)}


def load_amps_tar(spec):
    found = None
    with tarfile.open(spec["tar"], "r:gz") as tar:
        for member in tar:  # streams; stops after the first matching member
            if member.isfile() and member.name.endswith(".txt") \
                    and "/mathematica/" in member.name:
                f = tar.extractfile(member)
                if f:
                    found = (member.name,
                             f.read().decode("utf-8", errors="replace"))
                break
    if not found:
        raise RuntimeError("no mathematica .txt member found in tar")
    name, content = found
    label = f"`{spec['tar']}` member `{name}`"
    layout = f"Content of {label} (gzip tar of small UTF-8 text files, " \
             "'Problem: ... Answer: ...'):"
    return label, layout, {"raw_content": content}, "raw_lines", \
        {"lines": content.splitlines()}


def load_sudoku_csv(spec):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset")
    with open(path, newline="") as f:
        header = f.readline().rstrip("\n")
        first = f.readline().rstrip("\n")
    label = f"HF `{spec['repo']}` file `{spec['file']}`"
    layout = f"Header + first data row of {label} (CSV, UTF-8):"
    record = dict(zip(header.split(","), first.split(",")))
    return label, layout, record, "raw_lines", \
        {"lines": [header, first], "path": path}


RAW_LOADERS = {
    "hf": load_hf,
    "flan": load_hf_flan,
    "json_list_file": load_json_list_file,
    "khan_json": load_khan,
    "txt_pairs": load_txt_pairs,
    "parquet_file": load_parquet_file,
    "amps_tar": load_amps_tar,
    "sudoku_csv": load_sudoku_csv,
}


# ---------------------------------------------------------------------------
# input-keyword counting / examples
# ---------------------------------------------------------------------------

def hf_keyword_counts(ds, field, n):
    """Counter over a HF dataset column; n=None -> full column."""
    if n is None or n >= len(ds):
        return Counter(str(v) for v in ds[field]), \
            f"exact counts (all {fmt_rows(len(ds))} rows)"
    sub = ds.select(range(n))
    return Counter(str(v) for v in sub[field]), \
        f"estimated from the first {fmt_rows(n)} of {fmt_rows(len(ds))} rows"


def hf_keyword_examples(ds, field, counts, cap):
    """First row index per top-`cap` value (within a bounded scan)."""
    values = [v for v, _ in counts.most_common(cap)]
    wanted, found = set(values), {}
    n = min(len(ds), 50000)
    for i in range(n):
        v = str(ds[i][field])
        if v in wanted and v not in found:
            found[v] = ds[i]
        if len(found) == len(wanted):
            break
    return [(v, found[v]) for v in values if v in found]


def parquet_keyword_counts(files, column, n_files):
    import pyarrow.parquet as pq
    counts, rows = Counter(), 0
    for path in files[:n_files]:
        col = pq.read_table(path, columns=[column]).column(column)
        rows += len(col)
        counts.update(str(x) for x in col.to_pylist())
    total_files = len(files)
    note = f"exact (all {total_files} files)" if n_files >= total_files else \
        f"estimated from the first {n_files} of {total_files} files " \
        f"({fmt_rows(rows)} rows)"
    return counts, note


def parquet_keyword_examples(files, fields, key_field, counts, cap,
                             max_rows=20000):
    """First row per top-`cap` keyword value, scanning batches of file 1..k."""
    import pyarrow.parquet as pq
    values = [v for v, _ in counts.most_common(cap)]
    wanted, found = set(values), {}
    scanned = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=fields):
            for rec in batch.to_pylist():
                scanned += 1
                v = str(rec.get(key_field))
                if v in wanted and v not in found:
                    found[v] = (path, rec)
            if scanned >= max_rows or len(found) == len(wanted):
                break
        if scanned >= max_rows or len(found) == len(wanted):
            break
    return [(v, found[v]) for v in values if v in found]


def csv_keyword_counts(path, col_idx, n_lines):
    counts = Counter()
    with open(path, newline="") as f:
        next(f)  # header
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            counts[line.split(",", 1)[0]] += 1
    return counts, f"estimated from the first {fmt_rows(min(n_lines, i + 1))} " \
                   "data rows of the CSV"


# ---------------------------------------------------------------------------
# dataset registry
# ---------------------------------------------------------------------------

D = RAW_ROOT

REGISTRY = [
    {
        "name": "gsm8k_train",
        "script": "pipe/clean_gsm8k_train.py",
        "purpose": "GSM8K (OpenAI): 7,473 crowdsourced grade-school math word "
                   "problems (train split; the socratic config is unused). Raw "
                   "answers embed calculator annotations `<<...>>` and end in "
                   "`#### <final>`; the transform keeps only the final answer, "
                   "teaching terse numeric answers to word problems.",
        "source": "HF `openai/gsm8k` (config `main`, split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("question", "the word problem"),
            ("answer", "worked solution ending in `#### <final>`; only the part after `####` is kept"),
        ],
        "raw_example": {"kind": "hf", "repo": "openai/gsm8k", "config": "main",
                        "split": "train", "fields": ["question", "answer"]},
        "outputs": ["data/gsm8k_train.jsonl"],
        "out_format": "jsonl",
        "conditions": {"direct": ("response is the final numeric answer only",
                                  "pure answer-only supervision for easy arithmetic")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [],
    },
    {
        "name": "math_train",
        "script": "pipe/clean_math_train.py",
        "purpose": "MATH (Hendrycks): 12.5k competition problems in 7 subjects "
                   "with LaTeX worked solutions (train splits used here). Each "
                   "problem yields the full solution (cot) plus the last "
                   "`\\boxed{...}` content as a bare answer (direct).",
        "source": "HF `EleutherAI/hendrycks_math` (7 subject configs, split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache, one config per subject",
        "raw_fields": [
            ("problem", "LaTeX problem statement"),
            ("solution", "worked solution containing `\\boxed{...}`"),
        ],
        "raw_example": {"kind": "hf", "repo": "EleutherAI/hendrycks_math",
                        "config": "algebra", "split": "train",
                        "fields": ["problem", "solution"]},
        "outputs": ["data/math_train.jsonl"],
        "out_format": "jsonl",
        "conditions": {
            "cot": ("full worked LaTeX solution",
                    "teaches competition-level step-by-step reasoning"),
            "direct": ("content of the last `\\boxed{...}` (row omitted when no boxed answer)",
                       "same problems as answer-only supervision"),
        },
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "subject config", "side": "input",
            "intro": "The 7 HF configs (subjects) concatenated by the script.",
            "count": {"kind": "hf_configs",
                      "repo": "EleutherAI/hendrycks_math",
                      "configs": ["algebra", "counting_and_probability",
                                  "geometry", "intermediate_algebra",
                                  "number_theory", "prealgebra",
                                  "precalculus"]},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "natural_reasoning",
        "script": "pipe/clean_natural_reasoning.py",
        "purpose": "NaturalReasoning (Meta): 1.1M challenging reasoning "
                   "questions backtranslated from DCLM/FineMath pretraining "
                   "corpora, deduplicated and decontaminated against MATH/GPQA/"
                   "MMLU. The transform keeps the reference answer extracted "
                   "from the source document (rows with empty answers or "
                   "proof-style questions are dropped).",
        "source": "HF `facebook/natural_reasoning` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("question", "the question"),
            ("reference_answer", "reference answer from the source document (stripped; empty -> dropped)"),
        ],
        "raw_example": {"kind": "hf", "repo": "facebook/natural_reasoning",
                        "split": "train",
                        "fields": ["question", "reference_answer"]},
        "outputs": ["data/natural_reasoning.jsonl"],
        "out_format": "jsonl",
        "conditions": {"noisy,direct": ("reference answer as-is, unverified",
                                        "broad reasoning QA whose answers were not model-verified")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [],
    },
    {
        "name": "no_robots",
        "script": "pipe/clean_no_robots.py",
        "purpose": "No Robots (HuggingFace): 10k instructions written by human "
                   "annotators (not GPT), modelled after InstructGPT, across 10 "
                   "categories (Generation, Open QA, ...). Only the first "
                   "user->assistant turn is kept (system prompt prepended).",
        "source": "HF `HuggingFaceH4/no_robots` (splits `train`+`test`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("messages", "conversation turns; only the optional system message + first user/assistant pair are read"),
        ],
        "raw_example": {"kind": "hf", "repo": "HuggingFaceH4/no_robots",
                        "split": "train", "fields": ["messages"]},
        "outputs": ["data/no_robots.jsonl"],
        "out_format": "jsonl",
        "conditions": {"cot": ("free-form human-written response (tagged cot by convention)",
                               "human-quality SFT signal for general instruction following")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "category", "side": "input",
            "intro": "Human-assigned instruction category (from the dataset card taxonomy).",
            "count": {"kind": "hf", "repo": "HuggingFaceH4/no_robots",
                      "split": "train", "field": "category", "n": None},
            "values": None, "examples": 2,
            "fields": ["messages", "category"],
        }],
    },
    {
        "name": "numinamath",
        "script": "pipe/clean_numinamath.py",
        "purpose": "NuminaMath 1.5: ~900k competition-level math problems with "
                   "CoT solutions, sourced from Chinese high-school exercises to "
                   "international olympiads. The transform drops synthetic rows, "
                   "invalid problems/solutions, and URL/translation artifacts; "
                   "kept rows yield the full solution plus, for non-proofs, the "
                   "short answer.",
        "source": "HF `AI-MO/NuminaMath-1.5` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("problem", "the problem"),
            ("solution", "worked CoT solution"),
            ("answer", "short final answer ('proof' for proofs)"),
            ("synthetic", "filter: synthetic rows are dropped"),
            ("problem_is_valid", "filter: must be 'Yes'"),
            ("solution_is_valid", "filter: must be 'Yes'"),
            ("question_type", "filter: 'proof' rows get no direct record"),
        ],
        "raw_example": {"kind": "hf", "repo": "AI-MO/NuminaMath-1.5",
                        "split": "train",
                        "fields": ["problem", "solution", "answer",
                                   "synthetic", "problem_is_valid",
                                   "solution_is_valid", "question_type"]},
        "outputs": ["data/numinamath.jsonl"],
        "out_format": "jsonl",
        "conditions": {
            "noisy,cot": ("full solution, source quality unverified",
                          "bulk competition-math reasoning signal"),
            "noisy,direct": ("short answer only, source quality unverified",
                             "answer-only variant of the same problems"),
        },
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "question_type", "side": "input",
            "intro": "Problem form per the dataset card; drives whether a "
                     "direct record is emitted.",
            "count": {"kind": "hf", "repo": "AI-MO/NuminaMath-1.5",
                      "split": "train", "field": "question_type", "n": 30000},
            "values": {
                "math-word-problem": ("problem with a concrete answer",
                                      "gets both a cot and a direct record"),
                "proof": ("prove/show statement",
                          "cot record only — no short answer exists"),
                "MCQ": ("multiple choice", "rare here; answer is the option"),
                "other": ("uncategorized", "treated like math-word-problem"),
            },
            "examples": "per_value",
            "fields": ["problem", "answer", "question_type"],
        }],
    },
    {
        "name": "omnimath",
        "script": "pipe/clean_omnimath.py",
        "purpose": "Omni-MATH: 4,428 olympiad-level problems spanning 33 "
                   "sub-domains and 10 difficulty levels, published as a "
                   "benchmark — here the test split is reused as training data. "
                   "Each problem yields the full solution (cot) and the short "
                   "final answer (direct).",
        "source": "HF `KbsdJames/Omni-MATH` (split `test`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("problem", "the problem"),
            ("solution", "worked solution"),
            ("answer", "short final answer"),
        ],
        "raw_example": {"kind": "hf", "repo": "KbsdJames/Omni-MATH",
                        "split": "test",
                        "fields": ["problem", "solution", "answer"]},
        "outputs": ["data/omnimath.jsonl"],
        "out_format": "jsonl",
        "conditions": {
            "cot": ("full worked solution", "hardest tier of reasoning training data"),
            "direct": ("short final answer", "answer-only variant of the same problems"),
        },
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "source", "side": "input",
            "intro": "Originating competition per problem (unused by the "
                     "transform; shown for context).",
            "count": {"kind": "hf", "repo": "KbsdJames/Omni-MATH",
                      "split": "test", "field": "source", "n": None},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "principia_collection",
        "script": "pipe/clean_principia_collection.py",
        "purpose": "Principia Collection (Meta): ~550k synthetic STEM problems "
                   "(proposed by GPT-OSS-120B) over PhySH/MSC-2020 topics, in "
                   "two splits: `mathematical_object` (derive an equation/"
                   "expression) and `numerical` (numeric answer). Loaded "
                   "verbatim as question->answer pairs.",
        "source": "HF `facebook/principia-collection` (splits `mathematical_object`+`numerical`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("problem_statement", "the problem"),
            ("answer", "the answer"),
        ],
        "raw_example": {"kind": "hf", "repo": "facebook/principia-collection",
                        "fields": ["problem_statement", "answer"]},
        "outputs": ["data/principia_collection.jsonl"],
        "out_format": "jsonl",
        "conditions": {"synth,direct": ("synthetic question->answer pair",
                                        "large-volume synthetic STEM drill data")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "split", "side": "input",
            "intro": "The two HF splits, concatenated in order by the script.",
            "count": {"kind": "hf_splits", "repo": "facebook/principia-collection"},
            "values": {
                "mathematical_object": ("answer is an equation/expression",
                                        "teaches symbolic derivation"),
                "numerical": ("answer is numeric",
                              "teaches numeric problem solving"),
            },
            "examples": None,
        }],
    },
    {
        "name": "webinstruct_verified",
        "script": "pipe/clean_webinstruct_verified.py",
        "purpose": "WebInstruct-verified (TIGER-Lab, General-Reasoner project): "
                   "~230k web-mined questions across many domains whose answers "
                   "were verified for correctness/verifiability. Loaded verbatim "
                   "as direct QA pairs.",
        "source": "HF `TIGER-Lab/WebInstruct-verified` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("question", "the question"),
            ("answer", "verified answer"),
        ],
        "raw_example": {"kind": "hf", "repo": "TIGER-Lab/WebInstruct-verified",
                        "split": "train", "fields": ["question", "answer"]},
        "outputs": ["data/webinstruct_verified.jsonl"],
        "out_format": "jsonl",
        "conditions": {"direct": ("verified answer, no reasoning shown",
                                  "broad-domain verifiable QA supervision")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "category", "side": "input",
            "intro": "Domain of the question (unused by the transform; shown "
                     "for context).",
            "count": {"kind": "hf", "repo": "TIGER-Lab/WebInstruct-verified",
                      "split": "train", "field": "category", "n": 30000},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "amps_khan",
        "script": "pipe/clean_amps_khan.py",
        "purpose": "Khan Academy exercises from the AMPS dataset (Hendrycks et "
                   "al.) — one JSON file per problem. The step hints are joined "
                   "into the response; hint quality varies, hence noisy.",
        "source": f"`{D}/amps/khan/**/*.json` (extracted from `{D}/amps.tar.gz`)",
        "raw_layout": "one JSON object per .json file, UTF-8",
        "raw_fields": [
            ("problem", "the exercise text"),
            ("hints", "step hints, joined with newlines into the response"),
        ],
        "raw_example": {"kind": "khan_json", "input_dir": f"{D}/amps/khan",
                        "fields": ["problem", "hints"]},
        "outputs": ["data/amps_khan.jsonl"],
        "out_format": "jsonl",
        "conditions": {"noisy,cot": ("hint sequence as pseudo-solution, quality unverified",
                                     "cheap step-wise math signal of mixed quality")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [],
    },
    {
        "name": "arb",
        "script": "pipe/clean_platypus/clean_arb.py",
        "purpose": "ARB: an advanced reasoning benchmark with graduate-level "
                   "problems in math, physics, science, reading and law "
                   "(distributed with Platypus, one JSON file per subject). The "
                   "transform prepends a fixed task description per subject and "
                   "tags law as direct (option letter), the rest as cot.",
        "source": f"`{D}/Platypus/ARB/*.json` (5 subject files)",
        "raw_layout": "one JSON array of objects per subject file, UTF-8",
        "raw_fields": [
            ("instruction", "the problem (a subject-specific description is prepended)"),
            ("response", "the solution"),
        ],
        "raw_example": {"kind": "json_list_file",
                        "glob": f"{D}/Platypus/ARB/math.json",
                        "fields": ["instruction", "response"]},
        "outputs": ["data/Platypus/arb_*.jsonl"],
        "out_format": "jsonl",
        "conditions": {
            "cot": ("math / reading / science / physics: worked solution",
                    "advanced graduate-level reasoning"),
            "direct": ("law: correct option letter only",
                       "MCQ-style answer-only supervision"),
        },
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "subject file", "side": "input",
            "intro": "One input JSON file per subject; the file determines the "
                     "prepended instruction prefix and the condition tag.",
            "count": {"kind": "file_map",
                      "fn": lambda b: b.removeprefix("arb_").removesuffix(".jsonl")},
            "values": {
                "math": ("'Solve the math problem.'", "tagged cot"),
                "reading": ("'Solve the reading comprehension problem.'", "tagged cot"),
                "law": ("'Choose the correct option letter.'", "tagged direct"),
                "science": ("'Solve the science problem.'", "tagged cot"),
                "physics": ("'Solve the physics problem.'", "tagged cot"),
            },
            "examples": None,
        }],
    },
    {
        "name": "openbookqa",
        "script": "pipe/clean_platypus/clean_openbookqa.py",
        "purpose": "OpenBookQA: ~6k elementary science multiple-choice "
                   "questions ('additional' config, which adds the supporting "
                   "fact1). Question, options and fact are rendered into one "
                   "instruction; the response is the option letter.",
        "source": "HF `allenai/openbookqa` (config `additional`, splits `train`+`validation`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("question_stem", "the question"),
            ("fact1", "supporting fact appended to the instruction"),
            ("choices", "answer options, rendered as A:/B:/C:/D:"),
            ("answerKey", "correct option letter (the response)"),
        ],
        "raw_example": {"kind": "hf", "repo": "allenai/openbookqa",
                        "config": "additional", "split": "train",
                        "fields": ["question_stem", "fact1", "choices",
                                   "answerKey"]},
        "outputs": ["data/Platypus/openbookqa.jsonl"],
        "out_format": "jsonl",
        "conditions": {"direct": ("correct option letter",
                                  "fact-grounded MCQ answering")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "split", "side": "input",
            "intro": "The two HF splits concatenated by the script "
                     "(test is unused).",
            "count": {"kind": "hf_splits", "repo": "allenai/openbookqa",
                      "config": "additional", "only": ["train", "validation"]},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "reclor",
        "script": "pipe/clean_platypus/clean_reclor.py",
        "purpose": "ReClor: logical-reasoning reading comprehension MCQs from "
                   "LSAT/GMAT preparation material. Context, question and "
                   "options are rendered into one instruction; the response is "
                   "the option letter.",
        "source": "HF `metaeval/reclor` (splits `train`+`validation`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("context", "passage the question refers to"),
            ("question", "the question"),
            ("answers", "options, rendered as A:/B:/C:/D:"),
            ("label", "index of the correct option (response is the letter)"),
        ],
        "raw_example": {"kind": "hf", "repo": "metaeval/reclor", "split": "train",
                        "fields": ["context", "question", "answers", "label"]},
        "outputs": ["data/Platypus/reclor.jsonl"],
        "out_format": "jsonl",
        "conditions": {"direct": ("correct option letter",
                                  "logical-reasoning MCQ answering")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [],
    },
    {
        "name": "scibench",
        "script": "pipe/clean_platypus/clean_scibench.py",
        "purpose": "SciBench: college-level scientific problems from textbook "
                   "JSON files (physics, chemistry, math; `*_sol.json` files "
                   "carry worked solutions). The transform emits the worked "
                   "solution when present (cot) and always the final numeric "
                   "answer (direct).",
        "source": f"`{D}/Platypus/scibench/dataset/original/*.json`",
        "raw_layout": "one JSON array of objects per textbook file, UTF-8",
        "raw_fields": [
            ("problem_text", "the problem"),
            ("solution", "worked solution (cot record emitted only when non-empty)"),
            ("answer_latex", "final answer in LaTeX (falls back to answer_number)"),
            ("answer_number", "numeric final answer; used when answer_latex is redundant"),
        ],
        "raw_example": {"kind": "json_list_file",
                        "glob": f"{D}/Platypus/scibench/dataset/original/*.json",
                        "prefer": "solution",
                        "fields": ["problem_text", "solution", "answer_latex",
                                   "answer_number"]},
        "outputs": ["data/Platypus/scibench.jsonl"],
        "out_format": "jsonl",
        "conditions": {
            "cot": ("worked solution (only when the source has one)",
                    "scientific problem solving with steps"),
            "direct": ("final answer (answer_latex, simplified when it duplicates answer_number)",
                       "answer-only variant of every problem"),
        },
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [],
    },
    {
        "name": "scienceqa",
        "script": "pipe/clean_platypus/clean_scienceqa.py",
        "purpose": "ScienceQA (text-only subset): grade-school science MCQs "
                   "with lectures and rationales. Rows with a rationale yield a "
                   "cot record (rationale + answer letter); every row also "
                   "yields a direct record (bare letter). A lecture, when "
                   "present, is appended to the instruction.",
        "source": "HF `metaeval/ScienceQA_text_only` (splits `train`+`validation`+`test`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("question", "the question"),
            ("choices", "options, rendered as A:/B:/C:/..."),
            ("lecture", "background text, appended to the instruction when non-empty"),
            ("solution", "rationale; when empty no cot record is emitted"),
            ("answer", "index of the correct option (response is the letter)"),
        ],
        "raw_example": {"kind": "hf", "repo": "metaeval/ScienceQA_text_only",
                        "split": "train",
                        "fields": ["question", "choices", "lecture", "solution",
                                   "answer"]},
        "outputs": ["data/Platypus/scienceqa.jsonl"],
        "out_format": "jsonl",
        "conditions": {
            "cot": ("rationale + 'Answer: X'",
                    "teaches explaining the answer before giving it"),
            "direct": ("bare option letter", "answer-only variant of every question"),
        },
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [],
    },
    {
        "name": "theoremqa",
        "script": "pipe/clean_platypus/clean_theoremqa.py",
        "purpose": "TheoremQA: 800 expert-curated university-level questions "
                   "driven by 350+ STEM theorems (math, EE&CS, physics, "
                   "finance). Rows containing a picture are dropped; the rest "
                   "become direct question->answer pairs.",
        "source": "HF `TIGER-Lab/TheoremQA` (split `test`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("Question", "the question"),
            ("Answer", "the answer"),
            ("Picture", "filter only: rows with a picture are dropped"),
        ],
        "raw_example": {"kind": "hf", "repo": "TIGER-Lab/TheoremQA",
                        "split": "test",
                        "fields": ["Question", "Answer", "Picture"]},
        "outputs": ["data/Platypus/theoremqa.jsonl"],
        "out_format": "jsonl",
        "conditions": {"direct": ("final answer only",
                                  "theorem-application questions, answer-verified style")},
        "condition_counts": {"kind": "jsonl_scan"},
        "keywords": [{
            "field": "Answer_type", "side": "input",
            "intro": "Declared answer type per question (unused by the "
                     "transform; shown for context).",
            "count": {"kind": "hf", "repo": "TIGER-Lab/TheoremQA",
                      "split": "test", "field": "Answer_type", "n": None},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "flan",
        "script": "pipe_clustered/clean_flan.py",
        "purpose": "FLAN v2 instruction-tuning collection (Open-Orca parquet "
                   "dump): templated tasks from the Flan/T0/NIV2/CoT/dialog "
                   "submixtures, in few-shot/zero-shot and with/without-options "
                   "variants. 14 subsets are included; each output parquet "
                   "holds one (subset, task) pair.",
        "source": f"`{D}/Open-Orca/FLAN/<subset>/*.parquet`",
        "raw_layout": "one parquet file set per subset directory",
        "raw_fields": [
            ("_task_name", "source task id; becomes part of the output filename"),
            ("inputs", "prompt text -> instruction"),
            ("targets", "target text -> response"),
        ],
        "raw_example": {"kind": "flan", "input_dir": f"{D}/Open-Orca/FLAN",
                        "subset": "cot_fsopt_data",
                        "fields": ["_task_name", "inputs", "targets"],
                        "out_dir": "data_clustered/flan"},
        "outputs": ["data_clustered/flan/*.parquet"],
        "out_format": "parquet",
        "conditions": {
            "direct": ("12 subsets: dialog/flan/niv2/t0 in fsopt/fsnoopt/zsopt/zsnoopt variants",
                       "classic instruction-following supervision"),
            "cot": ("2 subsets: cot_fsopt_data, cot_zsopt_data",
                    "chain-of-thought prompts with reasoning targets"),
        },
        "condition_counts": {"kind": "file_map",
                             "fn": lambda b: "cot" if b.startswith("cot_")
                             else "direct"},
        "keywords": [{
            "field": "subset", "side": "input",
            "intro": "Subset directory = submixture x prompt style "
                     "(fs = few-shot, zs = zero-shot, opt/noopt = answer "
                     "options present/absent). One example pair below is drawn "
                     "from cot_fsopt_data.",
            "count": {"kind": "file_map",
                      "fn": lambda b: b.split("__", 1)[0]},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "synth",
        "script": "pipe_clustered/clean_SYNTH.py",
        "purpose": "SYNTH (PleIAs): ~80M synthetic samples amplified from "
                   "~59k Wikipedia 'vital articles', with model-written "
                   "reasoning traces; ~20% non-English. The transform keeps "
                   "English only, drops self-knowledge queries and cooking "
                   "exercises, and derives the condition tag from the exercise "
                   "type.",
        "source": f"`{D}/PleIAs/SYNTH/*.parquet`",
        "raw_layout": "500 parquet files (synth_001.parquet ...), one row per sample",
        "raw_fields": [
            ("query", "the prompt (constraints appended for rag exercises)"),
            ("constraints", "extra requirements (rag exercises only)"),
            ("synthetic_answer", "generated answer -> response"),
            ("exercise", "exercise type; drives the condition tag and the 'cooking' filter"),
            ("language", "filter: only 'en' kept"),
            ("query_seed_url", "filter: 'Pleias self-knowledge' rows dropped"),
        ],
        "raw_example": {"kind": "parquet_file",
                        "glob": f"{D}/PleIAs/SYNTH/*.parquet",
                        "fields": ["query", "constraints", "synthetic_answer",
                                   "exercise", "language", "query_seed_url"]},
        "outputs": ["data_clustered/synth/*.parquet"],
        "out_format": "parquet",
        "conditions": {
            "synth,cot": ("creative writing / rag / memorization / constrained writing / editing",
                          "synthetic reasoning-trace supervision"),
            "synth,direct": ("math mcq / mcq",
                             "synthetic answer-only MCQ supervision"),
        },
        # NOTE: the script also maps exercise=="math exercise" to
        # "synth,noisy,cot", but every math-exercise row has a null
        # query_seed_url, which the null-rejecting filter drops — verified
        # against all 500 output files: 0 rows carry that condition.
        "condition_counts": {"kind": "col_sample", "files": 500},
        "keywords": [
            {
                "field": "exercise", "side": "input",
                "intro": "Exercise type — the field that decides the condition "
                         "tag (and whether the row is dropped).",
                "count": {"kind": "parquet_col", "column": "exercise",
                          "files": 3},
                "values": {
                    "memorization": ("recall facts from the seed article",
                                     "mapped to synth,cot; by far the largest slice"),
                    "mcq": ("multiple-choice question",
                            "mapped to synth,direct"),
                    "constrained writing": ("write text under explicit constraints",
                                            "mapped to synth,cot"),
                    "math exercise": ("open-ended math problem",
                                      "mapped to synth,noisy,cot in the script, but every such row has a null query_seed_url and is dropped by the filter — 0 rows in the output"),
                    "rag": ("retrieval-augmented QA (constraints appended to the query)",
                            "mapped to synth,cot"),
                    "math mcq": ("multiple-choice math",
                                 "mapped to synth,direct"),
                    "creative writing": ("stories/poems etc.",
                                         "mapped to synth,cot"),
                    "editing": ("rewrite/improve a text",
                                "mapped to synth,cot"),
                    "cooking": ("recipe exercises",
                                "DROPPED by the transform (no output rows)"),
                },
                "examples": "per_value",
                "fields": ["query", "constraints", "synthetic_answer",
                           "exercise", "language"],
            },
            {
                "field": "language", "side": "input",
                "intro": "Sample language; only `en` survives the transform "
                         "(~80% of the raw data per the dataset card).",
                "count": {"kind": "parquet_col", "column": "language",
                          "files": 3},
                "values": None, "examples": None,
            },
        ],
    },
    {
        "name": "dmmath",
        "script": "pipe_clustered/clean_dmmath.py",
        "purpose": "DeepMind mathematics_dataset-v1.0: procedurally generated "
                   "school math over ~120 task types, split into train-easy/"
                   "medium/hard tiers for curriculum training. Each .txt holds "
                   "alternating question/answer lines; one output parquet per "
                   "(tier, task).",
        "source": f"`{D}/mathematics_dataset-v1.0/{{train-easy,train-medium,train-hard}}/*.txt`",
        "raw_layout": "plain UTF-8 text: line 2k = question, line 2k+1 = answer",
        "raw_fields": [
            ("question_line", "odd lines: the question"),
            ("answer_line", "even lines: the short answer"),
        ],
        "raw_example": {"kind": "txt_pairs",
                        "input_dir": f"{D}/mathematics_dataset-v1.0",
                        "subset": "train-easy"},
        "outputs": ["data_clustered/dmmath/*.parquet"],
        "out_format": "parquet",
        "conditions": {"direct": ("generated short answer",
                                  "unlimited-volume clean drill data")},
        "condition_counts": {"kind": "col_sample", "files": 12},
        "keywords": [{
            "field": "difficulty tier", "side": "input",
            "intro": "Directory tier, encoded in the output filename prefix "
                     "(train-easy__... etc.).",
            "count": {"kind": "file_map", "fn": lambda b: b.split("__", 1)[0]},
            "values": {
                "train-easy": ("easy tier", "curriculum stage 1"),
                "train-medium": ("medium tier", "curriculum stage 2"),
                "train-hard": ("hard tier", "curriculum stage 3"),
            },
            "examples": None,
        }],
    },
    {
        "name": "acereason",
        "script": "pipe_clustered/clean_acereason.py",
        "purpose": "AceReason-1.1-SFT (nvidia): math+code SFT data whose "
                   "responses were generated by DeepSeek-R1 and decontaminated "
                   "against benchmarks. The transform keeps the math category "
                   "only and strips the `<think>...</think>` block from each "
                   "response.",
        "source": "HF `nvidia/AceReason-1.1-SFT` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("input", "the problem -> instruction"),
            ("output", "response with a <think> block (stripped)"),
            ("category", "filter: only 'math' rows kept"),
        ],
        "raw_example": {"kind": "hf", "repo": "nvidia/AceReason-1.1-SFT",
                        "split": "train", "match": ("category", "math"),
                        "fields": ["input", "output", "category"]},
        "outputs": ["data_clustered/acereason/all.parquet"],
        "out_format": "parquet",
        "conditions": {"synth,cot": ("R1-generated math reasoning, think block removed",
                                     "strong verified-style math reasoning signal")},
        "condition_counts": {"kind": "col_sample", "files": 1},
        "keywords": [{
            "field": "category", "side": "input",
            "intro": "Row category; the transform keeps only `math` "
                     "(per the dataset card: 2.67M math / 1.30M code samples).",
            "count": {"kind": "hf", "repo": "nvidia/AceReason-1.1-SFT",
                      "split": "train", "field": "category", "n": 30000},
            "values": {
                "math": ("math problem", "KEPT (the code rows are dropped)"),
                "code": ("coding problem", "dropped by the transform"),
            },
            "examples": None,
        }],
    },
    {
        "name": "ampsmathematica",
        "script": "pipe_clustered/clean_ampsmathematica.py",
        "purpose": "AMPS Mathematica: machine-generated math exercises with "
                   "Mathematica-produced answers, read straight from the tar "
                   "archive. Files under a `*_w_steps` task folder carry "
                   "step-by-step answers (cot), the rest final answers "
                   "(direct); output is grouped one parquet per topic_subtask.",
        "source": f"`{D}/amps.tar.gz` (members `amps/mathematica/<topic>/<task>/*.txt`)",
        "raw_layout": "gzip tar of small UTF-8 text files: 'Problem: ... Answer: ...'",
        "raw_fields": [
            ("raw_content", "whole file: 'Problem:' prefix stripped, split on the first 'Answer:'"),
        ],
        "raw_example": {"kind": "amps_tar", "tar": f"{D}/amps.tar.gz"},
        "outputs": ["data_clustered/ampsmathematica/*.parquet"],
        "out_format": "parquet",
        "conditions": {
            "noisy,cot": ("task folder ends with `_w_steps`: worked steps",
                          "step-wise math signal of mixed quality"),
            "noisy,direct": ("other task folders: final answer only",
                             "answer-only math drill"),
        },
        "condition_counts": {"kind": "file_map",
                             "fn": lambda b: "noisy,cot" if "_w_steps" in b
                             else "noisy,direct"},
        "keywords": [{
            "field": "task folder suffix", "side": "input",
            "intro": "The `_w_steps` suffix on the tar member's task folder "
                     "decides the condition (and the output file).",
            "count": {"kind": "file_map",
                      "fn": lambda b: "w_steps" if "_w_steps" in b else "plain"},
            "values": {
                "w_steps": ("answer contains worked steps", "tagged noisy,cot"),
                "plain": ("answer is the final result", "tagged noisy,direct"),
            },
            "examples": None,
        }],
    },
    {
        "name": "openmathinstruct2",
        "script": "pipe_clustered/clean_openmathinstruct2.py",
        "purpose": "OpenMathInstruct-2 (nvidia): 14M problem-solution pairs "
                   "generated by Llama3.1-405B from GSM8K/MATH training "
                   "problems (solution augmentation + problem augmentation; "
                   "answers for augmented problems are majority-voted). Every "
                   "row goes to cot.parquet; rows not from the original "
                   "datasets also go to direct.parquet.",
        "source": "HF `nvidia/OpenMathInstruct-2` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("problem", "the problem -> instruction"),
            ("generated_solution", "model-generated solution (cot.parquet response)"),
            ("expected_answer", "short answer (direct.parquet response)"),
            ("problem_source", "filter: rows from 'math'/'gsm8k' are excluded from direct.parquet"),
        ],
        "raw_example": {"kind": "hf", "repo": "nvidia/OpenMathInstruct-2",
                        "split": "train",
                        "fields": ["problem", "generated_solution",
                                   "expected_answer", "problem_source"]},
        "outputs": ["data_clustered/openmathinstruct2/cot.parquet",
                    "data_clustered/openmathinstruct2/direct.parquet"],
        "out_format": "parquet",
        "conditions": {
            "synth,cot": ("cot.parquet: full generated solution",
                          "very large synthetic math reasoning corpus"),
            "synth,direct": ("direct.parquet: short expected answer (augmented problems only)",
                             "answer-only variant; original-problem answers are already covered by gsm8k/math_train"),
        },
        "condition_counts": {"kind": "file_map",
                             "fn": lambda b: "synth,cot" if b.startswith("cot")
                             else "synth,direct"},
        "keywords": [{
            "field": "problem_source", "side": "input",
            "intro": "Provenance per row; decides direct.parquet membership.",
            "count": {"kind": "hf", "repo": "nvidia/OpenMathInstruct-2",
                      "split": "train", "field": "problem_source", "n": 30000},
            "values": {
                "gsm8k": ("original GSM8K train problem",
                          "cot only — excluded from direct.parquet"),
                "math": ("original MATH train problem",
                         "cot only — excluded from direct.parquet"),
                "augmented_gsm8k": ("new problem derived from GSM8K",
                                    "goes to both cot and direct"),
                "augmented_math": ("new problem derived from MATH",
                                   "goes to both cot and direct"),
            },
            "examples": "per_value",
            "fields": ["problem", "expected_answer", "problem_source"],
        }],
    },
    {
        "name": "openthoughts2",
        "script": "pipe_clustered/clean_openthoughts2.py",
        "purpose": "OpenThoughts2-1M: 1.1M synthetic reasoning traces (math, "
                   "science, code, puzzles) with R1-style `<think>` blocks. The "
                   "transform drops code-related sources and code-looking rows, "
                   "strips the think block, and keeps the remaining answer.",
        "source": "HF `open-thoughts/OpenThoughts2-1M` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("conversations", "exactly one user + one assistant turn; both used"),
            ("source", "filter: code/math-duplicate sources dropped (dolphin, magicoder, sharegpt, nvidia_math, ...)"),
        ],
        "raw_example": {"kind": "hf", "repo": "open-thoughts/OpenThoughts2-1M",
                        "split": "train", "match": ("source", "automath"),
                        "fields": ["conversations", "source"]},
        "outputs": ["data_clustered/openthoughts2/all.parquet"],
        "out_format": "parquet",
        "conditions": {"synth,cot": ("reasoning trace with <think> block removed",
                                     "large synthetic reasoning corpus, code filtered out")},
        "condition_counts": {"kind": "col_sample", "files": 1},
        "keywords": [{
            "field": "source", "side": "input",
            "intro": "Upstream data source; the REMOVE_SOURCES set (dolphin, "
                     "evolcodegolf, glaive, magicoder, sharegpt, codefeedback, "
                     "nvidia_math) is dropped as code or duplicate.",
            "count": {"kind": "hf", "repo": "open-thoughts/OpenThoughts2-1M",
                      "split": "train", "field": "source", "n": 50000},
            "values": None, "examples": 2,
            "fields": ["conversations", "source"],
        }],
    },
    {
        "name": "sudoku_extreme",
        "script": "pipe_clustered/clean_sudoku.py",
        "purpose": "sudoku-extreme (Sapient): 3.8M training puzzles mixing easy "
                   "sets with the hardest community-collected ones; exact-"
                   "deduped, unique solutions. The puzzle string ('.' -> '0') "
                   "gets a fixed 'Solve the Sudoku' prefix; the response is the "
                   "solved 81-char grid. Teaches long-horizon constraint "
                   "reasoning.",
        "source": "HF `sapientinc/sudoku-extreme` (file `train.csv`)",
        "raw_layout": "CSV (UTF-8): header `source,question,answer,rating`; puzzle/answer are 81-char strings",
        "raw_fields": [
            ("question", "puzzle, 81 chars, '.' = empty cell ('.' -> '0' in the output)"),
            ("answer", "solved grid, 81 chars -> response"),
        ],
        "raw_example": {"kind": "sudoku_csv", "repo": "sapientinc/sudoku-extreme",
                        "file": "train.csv"},
        "outputs": ["data_clustered/sudoku_extreme/all.parquet"],
        "out_format": "parquet",
        "conditions": {"direct": ("solved grid, no reasoning",
                                  "pure pattern/constraint output, no intermediate steps")},
        "condition_counts": {"kind": "col_sample", "files": 1},
        "keywords": [{
            "field": "source", "side": "input",
            "intro": "Puzzle collection the row came from (per the dataset "
                     "card: puzzles0-2 are easy, puzzles3+ are the hardest "
                     "known). Unused by the transform; shown for context.",
            "count": {"kind": "csv_col", "repo": "sapientinc/sudoku-extreme",
                      "file": "train.csv", "n": 200000},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "tasksource",
        "script": "pipe_clustered/clean_tasksource.py",
        "purpose": "tasksource-instruct-v0: 5.3M instruction examples recast "
                   "from 485 curated HF datasets (mostly discriminative: NLI, "
                   "classification, tagging, MCQ), capped at 30k rows per task. "
                   "The transform keeps a curated ~180-task subset and writes "
                   "one parquet per task.",
        "source": "HF `tasksource/tasksource-instruct-v0` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("task", "task id; filter (TASK_SET) and output filename"),
            ("inputs", "prompt -> instruction"),
            ("targets", "target -> response (trailing '.' removed)"),
        ],
        "raw_example": {"kind": "hf", "repo": "tasksource/tasksource-instruct-v0",
                        "split": "train", "fields": ["task", "inputs", "targets"]},
        "outputs": ["data_clustered/tasksource/*.parquet"],
        "out_format": "parquet",
        "conditions": {"direct": ("short task target",
                                  "broad discriminative-task supervision")},
        "condition_counts": {"kind": "col_sample", "files": 12},
        "keywords": [{
            "field": "task", "side": "output",
            "intro": "One output parquet per kept task; the task name is the "
                     "filename. High cardinality — counts as file count only.",
            "count": {"kind": "n_output_files"},
            "values": None, "examples": None,
        }],
    },
    {
        "name": "textbookreasoning",
        "script": "pipe_clustered/clean_textbookreasoning.py",
        "purpose": "TextbookReasoning (MegaScience): 650k questions with "
                   "truthful reference answers extracted from 12k university "
                   "textbooks across 7 scientific disciplines. Every row goes "
                   "to cot.parquet (full answer); non-proof rows also go to "
                   "direct.parquet (short reference answer).",
        "source": "HF `MegaScience/TextbookReasoning` (split `train`)",
        "raw_layout": "HF dataset (arrow table) in the prefetched local cache",
        "raw_fields": [
            ("question", "the question"),
            ("answer", "full answer (cot.parquet response)"),
            ("reference_answer", "short answer (direct.parquet response; 'prove'/'show that' questions excluded)"),
        ],
        "raw_example": {"kind": "hf", "repo": "MegaScience/TextbookReasoning",
                        "split": "train",
                        "fields": ["question", "answer", "reference_answer"]},
        "outputs": ["data_clustered/textbookreasoning/cot.parquet",
                    "data_clustered/textbookreasoning/direct.parquet"],
        "out_format": "parquet",
        "conditions": {
            "synth,cot": ("cot.parquet: full extracted answer",
                          "textbook-grade scientific reasoning"),
            "noisy,direct": ("direct.parquet: short reference answer",
                             "answer-only variant for verifiable-style training"),
        },
        "condition_counts": {"kind": "file_map",
                             "fn": lambda b: "synth,cot" if b.startswith("cot")
                             else "noisy,direct"},
        "keywords": [{
            "field": "subject", "side": "input",
            "intro": "Scientific discipline (unused by the transform; shown "
                     "for context).",
            "count": {"kind": "hf", "repo": "MegaScience/TextbookReasoning",
                      "split": "train", "field": "subject", "n": 30000},
            "values": None, "examples": None,
        }],
    },
]


# ---------------------------------------------------------------------------
# keyword processing
# ---------------------------------------------------------------------------

_HF_CACHE = {}


def hf_cached(repo, config=None, split=None, collapse=True):
    key = (repo, config, split, collapse)
    if key not in _HF_CACHE:
        _HF_CACHE[key] = _hf_dataset({"repo": repo, "config": config,
                                      "split": split}, collapse=collapse)
    return _HF_CACHE[key]


def process_keyword(entry, block, out_files):
    """-> (counts: Counter, note: str, examples: [(value, record)])."""
    spec = block["count"]
    kind = spec["kind"]
    counts, note = Counter(), ""
    ds = None
    raw_files = None

    if kind == "hf":
        ds = hf_cached(spec["repo"], spec.get("config"), spec.get("split"))
        n = spec.get("n")
        full = n is None or n >= len(ds)
        sub = ds if full else ds.select(range(n))
        counts = Counter(str(v) for v in sub[spec["field"]])
        note = f"exact counts (all {fmt_rows(len(ds))} rows)" if full else \
            f"estimated from the first {fmt_rows(n)} of " \
            f"{fmt_rows(len(ds))} rows"
        ds = sub
    elif kind == "hf_configs":
        for c in spec["configs"]:
            counts[c] = hf_cached(spec["repo"], c, "train").num_rows
        note = "exact (HF split sizes, train split per config)"
    elif kind == "hf_splits":
        dsd = hf_cached(spec["repo"], spec.get("config"), collapse=False)
        for name, s in dsd.items():
            if spec.get("only") and name not in spec["only"]:
                continue
            counts[name] = s.num_rows
        note = "exact (HF split sizes)"
    elif kind == "parquet_col":
        raw_files = sorted(globmod.glob(entry["raw_example"]["glob"]))
        counts, note = parquet_keyword_counts(raw_files, spec["column"],
                                              spec["files"])
    elif kind == "csv_col":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset")
        counts, note = csv_keyword_counts(path, 0, spec["n"])
    elif kind == "file_map":
        counts, note = file_map_counts(out_files, entry["out_format"],
                                       spec["fn"])
    elif kind == "n_output_files":
        counts = Counter({"distinct tasks (one parquet file each)":
                          len(out_files)})
        note = "exact (output file count)"
    else:
        raise ValueError(kind)

    # examples for this keyword
    examples = []
    want = block.get("examples")
    if want and counts:
        hi_card = len(counts) > HIGH_CARDINALITY
        cap = 2 if (want == 2 or hi_card) else MAX_KEYWORD_EXAMPLES
        if kind == "hf" and ds is not None:
            field = spec["field"]
            values = [v for v, _ in counts.most_common(cap)]
            wanted = set(values)
            col = ds[field]
            idx = {}
            for i, v in enumerate(col):
                v = str(v)
                if v in wanted and v not in idx:
                    idx[v] = i
                if len(idx) == len(wanted):
                    break
            examples = [(v, ds[idx[v]]) for v in values if v in idx]
        elif kind == "parquet_col" and raw_files:
            fields = list(dict.fromkeys(
                block.get("fields", []) + [spec["column"]]))
            examples = [(v, rec) for v, (_, rec) in parquet_keyword_examples(
                raw_files, fields, spec["column"], counts, cap)]
    return counts, note, examples


# ---------------------------------------------------------------------------
# dataset creation date + domain
# Sources: dataset paper (preferred), HF dataset card, HF API createdAt
# (repo creation date, fallback). Year-month where known, year otherwise.
# ---------------------------------------------------------------------------

META = {
    # name: (created, domain, evidence)
    "gsm8k_train": ("2021-10", "math word problems",
                    "GSM8K paper, arXiv:2110.14168"),
    "math_train": ("2021-03", "competition math",
                   "MATH paper, arXiv:2103.03874"),
    "natural_reasoning": ("2025-02", "web-derived reasoning QA",
                          "NaturalReasoning paper, arXiv:2502.13124"),
    "no_robots": ("2023-11", "general instruction following (human-written)",
                  "HF release (repo createdAt 2023-11)"),
    "numinamath": ("2025-02", "competition math",
                   "NuminaMath-1.5 release (repo createdAt 2025-02)"),
    "omnimath": ("2024-10", "olympiad math", "Omni-MATH paper, arXiv:2410.07985"),
    "principia_collection": ("2025-11", "synthetic STEM problems",
                             "HF release (repo createdAt 2025-11; paper pending per card)"),
    "webinstruct_verified": ("2025-05", "web-mined QA (multi-domain)",
                             "General-Reasoner paper, arXiv:2505.14652"),
    "amps_khan": ("2021-03", "math exercises (Khan Academy)",
                  "AMPS, released with the MATH paper, arXiv:2103.03874"),
    "arb": ("2023-07", "advanced reasoning (graduate STEM/law/reading)",
            "ARB paper, arXiv:2307.13692"),
    "openbookqa": ("2018-09", "elementary science MCQ",
                   "OpenBookQA paper (EMNLP 2018), arXiv:1809.02789"),
    "reclor": ("2020-02", "logical reasoning MCQ",
               "ReClor paper (ICLR 2020), arXiv:2002.04326"),
    "scibench": ("2023-07", "college-level science problems",
                 "SciBench paper, arXiv:2307.10635"),
    "scienceqa": ("2022-09", "grade-school science MCQ",
                  "ScienceQA paper (NeurIPS 2022), arXiv:2209.09513"),
    "theoremqa": ("2023-05", "university STEM theorem QA",
                  "TheoremQA paper, arXiv:2305.12524"),
    "flan": ("2023-01", "instruction-tuning mixture",
             "FLAN v2 collection, arXiv:2301.13688 (Open-Orca parquet dump, 2023-07)"),
    "synth": ("2025-04", "synthetic general knowledge (Wikipedia-derived)",
              "PleIAs SYNTH release (blog announcement 2025-04)"),
    "dmmath": ("2019-04", "procedural school math",
               "DeepMind mathematics_dataset paper, arXiv:1904.01557"),
    "acereason": ("2025-06", "math reasoning SFT (DeepSeek-R1 distilled)",
                  "AceReason paper, arXiv:2506.13284"),
    "ampsmathematica": ("2021-03", "synthetic math exercises (Mathematica)",
                        "AMPS, released with the MATH paper, arXiv:2103.03874"),
    "openmathinstruct2": ("2024-10", "synthetic math instruction",
                          "OpenMathInstruct-2 paper, arXiv:2410.01559"),
    "openthoughts2": ("2025-04", "synthetic reasoning traces",
                      "OpenThoughts2 release (blog 'thinkagain', 2025-04)"),
    "sudoku_extreme": ("2024-10", "logic puzzles (sudoku)",
                       "HF release (repo createdAt 2024-10; Sapient)"),
    "tasksource": ("2023-05", "multi-task NLP (classification/NLI/MCQ)",
                   "tasksource-instruct-v0 HF release (repo createdAt 2023-05)"),
    "textbookreasoning": ("2025-07", "textbook science QA",
                          "MegaScience paper, arXiv:2507.16812"),
}

for _e in REGISTRY:
    _e["created"], _e["domain"], _e["created_src"] = META[_e["name"]]


# ---------------------------------------------------------------------------
# evaluation.md — static analysis doc (paper facts verified against
# ~/hrmseries/2/document.md; script claims verified against pipe*/)
# ---------------------------------------------------------------------------

EVALUATION_MD = """\
# Evaluation & benchmarks — relation to the training-data pipeline

↑ [index](README.md)

How the benchmarks in the HRM-Text paper (arXiv:2605.20613) relate to the
datasets processed by this repo, and how accuracy is computed given that the
training data was transformed. Paper facts are quoted from the paper; pipeline
claims are verified against the scripts in `pipe/`, `pipe/clean_platypus/`,
`pipe_clustered/`.

## The benchmarks

Table 4 of the paper evaluates: **MMLU, ARC-C, HellaSwag, Winogrande, BoolQ,
DROP, GSM8K, MATH** — against Llama/Qwen/Gemma/OLMo/Huginn/Ouro baselines.

## Key point: eval does not consume the transformed data

Benchmark accuracy is computed from the **original benchmark repositories**
(test/validation splits), prompted in the original format. The scripts in this
repo only repackage *training-side* data into `(instruction, response,
condition)` records for pretraining; they never produce eval prompts or gold
answers. **The transform therefore has no effect on how accuracy is computed —
only on what the model saw during training.**

One intentional coupling exists (paper §4.1): the model is trained with
condition tags, and at inference a condition tag is prepended to the
instruction to select the response style ("four primary conditions: direct
(answer-only), cot (chain-of-thought), synth, noisy"). The tags in the
`condition` column documented per dataset are exactly this mechanism.

## Eval protocol (paper Table 8 and §4.3)

| setting | value |
|---|---|
| max context | 3072 tokens |
| decoding | temperature 0 (deterministic) |
| system prompt | none |
| prompt contents | original benchmark question only; few-shot exemplars only where the benchmark protocol requires them |
| few-shot evals | vLLM |
| chain-of-thought evals | `lm_eval_harness` |
| checkpoint | EMA of weights (decay 0.9999) — used for the final evals and the released weights |

## Split separation: what is trained on vs. what is evaluated

| training dataset (this repo) | in the training mix | relation to Table 4 eval |
|---|---|---|
| gsm8k_train | GSM8K **train** split (7,473 rows) | GSM8K **test** (1,319) is evaluated — disjoint by construction |
| math_train | MATH **train**, 7 subjects (14,996 records) | MATH **test** (5,000) is evaluated — disjoint |
| openmathinstruct2, acereason | synthetic solutions derived from GSM8K/MATH **train** problems only | same train/test separation holds upstream |
| flan | train splits of many NLP benchmarks — verified in the output: `flan_*__ai2_arc_*`, `__drop_*`, `__hellaswag_*`, `__winogrande_*`, `*boolq*` task files (ARC-C/E, DROP, HellaSwag, Winogrande, BoolQ) | eval uses those benchmarks' test/validation splits; standard FLAN practice; the eval *questions* are covered by the n-gram decontamination test below |
| tasksource | train splits of 182 curated NLP tasks | none of the Table 4 benchmarks directly; same decontamination coverage |
| omnimath | Omni-MATH **test** split is used as *training* data — deliberate: Omni-MATH is not among the Table 4 benchmarks | **flag:** do not add Omni-MATH to the eval suite without removing it from the mix |
| theoremqa, scibench, reclor, arb, openbookqa, scienceqa | benchmarks repurposed as training data (Platypus collection): TheoremQA test, ReClor train+val, OpenBookQA train+val, ScienceQA all splits, SciBench full | none are Table 4 benchmarks; **flag:** evaluating on any of them requires excluding it from training first |
| synth, dmmath, ampsmathematica, amps_khan, sudoku_extreme, principia_collection, natural_reasoning, webinstruct_verified, no_robots, numinamath, openthoughts2, textbookreasoning | synthetic, procedural, or web-derived corpora | no direct benchmark split relationship; covered by the decontamination test |

## Decontamination (paper §4.2)

Method (adapted from the Llama family): tokenize the questions of **all
evaluated benchmarks** (excluding few-shot exemplars) and find **n-gram
matches (n = 13 and n = 20) against the fully tokenized pretraining corpus**.
A sample's contamination % is the fraction of its tokens inside matched
n-grams. Eval samples are partitioned into Clean (<20%), Not Clean (≥20%),
Not Dirty (<80%), Dirty (≥80%); contamination is deemed significant only if
|Z| > 2 on all four subsets (clean must do worse, dirty better).

Result: HRM-Text 0.6B — no significant contamination at either n. HRM-Text 1B
— significant only on DROP at n = 13 (not at n = 20), and it still scores
81.1 on the strictly-clean DROP subset (0% contamination, 5,904 samples).
Conclusion: benchmark performance is unlikely to be driven by exposure to
test examples.

## Practical rules for extending the recipe

1. **Before adding a benchmark to eval**, check it is not in the training
   mix: this index plus the split table above, and for aggregate mixtures
   (FLAN, tasksource) check the per-task output files —
   `ls /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/flan | grep -i <benchmark>`.
2. **Run the paper's n-gram decontamination test** (n = 13 and n = 20,
   benchmark questions vs. the tokenized corpus, clean/dirty subset analysis)
   for any new benchmark.
3. **Never train on the eval split** of an active benchmark. Training on a
   non-evaluated benchmark's test split (as done for Omni-MATH) is a
   deliberate one-way decision: document it in that dataset's doc page.
"""


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_raw_block(label, layout, record, hint, extra):
    lines = [f"### Raw — {label}", "", layout, ""]
    if hint == "table":
        lines.append(row_as_table(list(record.keys()), record))
    elif hint == "raw_json":
        lines.append(fence(json.dumps(truncate(record), indent=2,
                                      ensure_ascii=False), "json"))
    elif hint == "raw_lines":
        lines.append(fence("\n".join(truncate_str(l)
                                     for l in extra["lines"])))
    lines.append("")
    return lines


def render_output_record(entry, rel_file, record, label, cols=None):
    lines = [f"### {label}", ""]
    if entry["out_format"] == "jsonl":
        lines.append(f"One line of `{rel_file}` (JSONL — one JSON object per "
                     "line, UTF-8):")
        lines.append("")
        lines.append(fence(jsonl_line(record), "jsonl"))
    else:
        cols = cols or list(record.keys())
        lines.append(f"One row of `{rel_file}` (parquet table, columns: "
                     f"{', '.join(f'`{c}`' for c in cols)}), shown as a "
                     "table:")
        lines.append("")
        lines.append(row_as_table(cols, record))
    lines.append("")
    return lines


def render_keyword_section(entry, block, counts, note):
    lines = [f"## Keyword: `{block['field']}` ({block['side']})", ""]
    if block.get("intro"):
        lines += [block["intro"], ""]
    meanings = block.get("values")
    shown = counts.most_common()
    truncated_list = False
    if len(shown) > HIGH_CARDINALITY:
        shown = shown[:12]
        truncated_list = True
    if meanings:
        lines.append(md_table(
            ["value", "rows", "meaning", "why it matters"],
            [[f"`{v}`", fmt_rows(n)] + list(meanings.get(v, ("—", "—")))
             for v, n in shown]))
    else:
        lines.append(md_table(["value", "rows"],
                              [[f"`{v}`", fmt_rows(n)] for v, n in shown]))
    if truncated_list:
        lines.append(f"\n… {len(counts)} distinct values total, top 12 shown.")
    lines += ["", f"_{note}_", ""]
    return lines


def render_doc(entry, prev_name, next_name, data):
    lines = []
    nav = "↑ [index](README.md)"
    if prev_name:
        nav += f" · ← [{prev_name}]({prev_name}.md)"
    if next_name:
        nav += f" · [next → {next_name}]({next_name}.md)"
    lines += [f"# {entry['name']}", "", nav, ""]
    lines += [f"**Script:** `{entry['script']}`", ""]

    lines += ["## Purpose", "", entry["purpose"], "",
              f"Created: {entry['created']} · Domain: {entry['domain']} "
              f"_(date source: {entry['created_src']})_", ""]

    # --- before
    lines += ["## Before (raw storage)", ""]
    lines.append(f"- Source: {entry['source']}")
    lines.append(f"- Storage: {entry['raw_layout']}")
    lines.append("")
    lines.append("Fields read by the transform (types derived from the actual "
                 "files):")
    lines.append("")
    rt = data["raw_types"]
    lines.append(md_table(
        ["field", "type", "meaning"],
        [[f"`{f}`", rt.get(f, "—"), m] for f, m in entry["raw_fields"]]))
    lines.append("")

    # --- after
    files, total = data["files"], data["total"]
    lines += ["## After (transformed)", ""]
    locs = ", ".join(f"`{p}`" for p in entry["outputs"])
    lines.append(f"- Location: {locs} (under `{OUT_ROOT}`; "
                 f"{len(files)} file(s))")
    if entry["out_format"] == "jsonl":
        lines.append("- Storage: JSONL — one JSON object per line, UTF-8; "
                     "keys `condition`, `instruction`, `response`")
    else:
        lines.append("- Storage: parquet; columns `instruction`, `response`, "
                     "`condition`")
    lines.append(f"- Rows: {fmt_rows(total)}")
    lines.append("")
    ot = data["out_types"]
    out_meanings = {
        "instruction": "the prompt",
        "response": "the target",
        "condition": "comma-separated tags (see keyword table below)",
    }
    lines.append(md_table(
        ["column", "type", "meaning"],
        [[f"`{c}`", ot.get(c, "—"), out_meanings.get(c, "—")]
         for c in ot]))
    lines.append("")

    # --- condition keyword
    cond_counts, cond_note = data["cond_counts"], data["cond_note"]
    block = {"field": "condition", "side": "output",
             "intro": "Every output record carries this tag; training samples/"
                      "mixes by it.",
             "values": {k: list(v) for k, v in entry["conditions"].items()}}
    lines += render_keyword_section(entry, block, cond_counts, cond_note)

    # --- extra keywords
    for block, counts, note in data["keyword_stats"]:
        lines += render_keyword_section(entry, block, counts, note)

    # --- examples
    lines += ["## Examples", "",
              f"Fields longer than {TRUNC} chars are truncated "
              "(`… [truncated, N chars total]`); in tables, `⏎` marks a "
              "newline inside the value.", ""]

    label, layout, record, hint, extra = data["raw_example"]
    lines += render_raw_block(label, layout, record, hint, extra)

    if data.get("paired_output"):
        path, rec, cols = data["paired_output"]
        rel = os.path.relpath(path, OUT_ROOT)
        lines += render_output_record(
            entry, rel, rec,
            "Transformed — the same record (row 1 of the parquet for this "
            "exact (subset, task) pair)", cols)

    # keyword examples: (field, value, record)
    for field, value, rec in data["keyword_examples"]:
        cols = [f for f, _ in entry["raw_fields"] if f in rec]
        if not cols:
            cols = list(rec.keys())
        lines += [f"### Raw — `{field}` = `{value}`", ""]
        lines.append("One raw row with this value (same storage as above), "
                     "shown as a table:")
        lines.append("")
        lines.append(row_as_table(list(rec.keys()), rec))
        lines.append("")

    for value, path, rec, cols in data["cond_examples"]:
        rel = os.path.relpath(path, OUT_ROOT)
        lines += render_output_record(
            entry, rel, rec, f"Transformed — condition=`{value}`", cols)

    lines += ["---", "", nav, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# per-dataset processing
# ---------------------------------------------------------------------------

def process(entry):
    name = entry["name"]
    files = expand_outputs(entry["outputs"])
    if not files:
        print(f"skip: {name} (no output yet)", flush=True)
        return None
    fmt = entry["out_format"]
    data = {"files": files}

    # total rows + condition counts
    if fmt == "jsonl":
        total = 0
        counts = Counter()
        for path in files:
            n, c = count_jsonl(path, entry["conditions"])
            total += n
            counts.update(c)
        other = total - sum(counts.values())
        note = "exact counts (full scan of the jsonl file)"
        if other:
            note += (f"; {fmt_rows(other)} rows carry a condition value not "
                     "listed above")
        data["cond_counts"], data["cond_note"] = counts, note
    elif entry["condition_counts"]["kind"] == "file_map":
        fn = entry["condition_counts"]["fn"]
        counts = Counter()
        for path in files:
            counts[fn(os.path.basename(path))] += parquet_num_rows(path)
        total = sum(counts.values())
        data["cond_counts"] = counts
        data["cond_note"] = ("exact counts (condition is constant per file; "
                             "rows summed from parquet metadata)")
    else:  # col_sample
        total = sum(parquet_num_rows(p) for p in files)
        counts, note = gather_condition_counts(entry, files, total)
        data["cond_counts"], data["cond_note"] = counts, note
    data["total"] = total

    # raw example
    data["raw_example"] = RAW_LOADERS[entry["raw_example"]["kind"]](
        entry["raw_example"])
    _, _, record, hint, extra = data["raw_example"]

    # raw field types, derived from the actual data
    if "features" in extra:  # hf
        feats = extra["features"]
        data["raw_types"] = {f: hf_type_str(feats[f])
                             for f, _ in entry["raw_fields"]}
    elif "types" in extra:  # raw parquet
        data["raw_types"] = {f: extra["types"].get(f, "—")
                             for f, _ in entry["raw_fields"]}
    elif hint == "raw_json":
        data["raw_types"] = {f: json_type_str(record.get(f))
                             for f, _ in entry["raw_fields"]}
    elif hint == "raw_lines" and "path" in extra and \
            extra["path"].endswith(".csv"):
        data["raw_types"] = {f: "CSV field — UTF-8 text"
                             for f, _ in entry["raw_fields"]}
    elif entry["raw_example"]["kind"] == "txt_pairs":
        data["raw_types"] = {f: "UTF-8 text line"
                             for f, _ in entry["raw_fields"]}
    else:  # amps tar
        data["raw_types"] = {f: "UTF-8 text file inside a gzip tar"
                             for f, _ in entry["raw_fields"]}

    # output schema types from the actual file
    if fmt == "parquet":
        data["out_types"] = parquet_col_types(files[0])
    else:
        with open(files[0], "rb") as f:
            first = json.loads(f.readline())
        data["out_types"] = {k: json_type_str(v) for k, v in first.items()}

    # paired output example (flan)
    if extra.get("paired_output"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(extra["paired_output"])
        rec = next(pf.iter_batches(batch_size=1)).to_pylist()[0]
        data["paired_output"] = (extra["paired_output"], rec,
                                 pf.schema_arrow.names)

    # condition examples: one per discovered condition value
    cond_values = list(dict.fromkeys(
        list(entry["conditions"]) + list(data["cond_counts"])))
    if fmt == "jsonl":
        found = jsonl_condition_examples(files, cond_values)
        data["cond_examples"] = [(v, p, r, None)
                                 for v, (p, r) in found.items()]
    else:
        scan_files = flan_example_files(files) if entry["name"] == "flan" \
            else files
        found = parquet_condition_examples(scan_files, cond_values)
        data["cond_examples"] = [(v, p, r, c)
                                 for v, (p, r, c) in found.items()]

    # extra keyword blocks
    data["keyword_stats"] = []
    data["keyword_examples"] = []
    for block in entry["keywords"]:
        counts, note, examples = process_keyword(entry, block, files)
        data["keyword_stats"].append((block, counts, note))
        for v, rec in examples:
            keep = [f for f in block.get("fields", rec.keys()) if f in rec]
            data["keyword_examples"].append(
                (block["field"], v, {k: rec[k] for k in keep}))

    print(f"done: {name} ({fmt_rows(total)} rows, {len(files)} files)",
          flush=True)
    return data


def render_readme(generated):
    lines = []
    lines.append("# Data-cleaning pipeline — dataset docs")
    lines.append("")
    lines.append("Generated by `scripts/docs/generate_docs.py` on "
                 f"{datetime.date.today().isoformat()} (row counts and "
                 "examples are computed from the actual files; re-run the "
                 "script to refresh).")
    lines.append("")
    lines.append("**→ [evaluation.md](evaluation.md): how the HRM-Text paper's "
                 "benchmarks relate to this training data, and how accuracy is "
                 "computed given the transforms.**")
    lines.append("")
    lines.append("## Unified output contract")
    lines.append("")
    lines.append("Every cleaning script (`pipe/*.py`, "
                 "`pipe/clean_platypus/*.py`, `pipe_clustered/*.py`) reduces "
                 "one raw dataset to records of the same shape:")
    lines.append("")
    lines.append("- `instruction` (string) — the prompt")
    lines.append("- `response` (string) — the target")
    lines.append("- `condition` (string) — comma-separated tags describing "
                 "the record")
    lines.append("")
    lines.append(f"Storage: JSONL under `{OUT_ROOT}/data/` (and "
                 f"`data/Platypus/`), parquet under "
                 f"`{OUT_ROOT}/data_clustered/<name>/`. Clustered datasets "
                 "are split into many parquet files (one per task/subset) so "
                 "large mixes can be read in parallel and per-task.")
    lines.append("")
    lines.append("`condition` tags:")
    lines.append("")
    for tag, (meaning, why) in CONDITION_MEANING.items():
        lines.append(f"- `{tag}` — {meaning}; {why}")
    lines.append("")
    lines.append("Why the transform exists: the raw datasets are heterogeneous "
                 "(HF datasets, JSON dumps, tar archives, line-based txt, "
                 "CSV). Unifying them into one schema lets the training code "
                 "sample and mix data by condition tags (e.g. how much cot vs "
                 "direct, how much noisy/synthetic data) without any "
                 "per-dataset code.")
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    lines.append("| dataset | created | domain | source | condition tags | rows | doc |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry in generated:
        tags = ", ".join(f"`{c}`" for c in entry["conditions"])
        lines.append(f"| {entry['name']} | {entry['created']} | "
                     f"{entry['domain']} | {entry['source']} | {tags} | "
                     f"{fmt_rows(entry['_total'])} | "
                     f"[{entry['name']}.md]({entry['name']}.md) |")
    lines.append("")
    skipped = [e["name"] for e in REGISTRY if e not in generated]
    if skipped:
        lines.append("Skipped (no transformed output yet when this was "
                     "generated): " + ", ".join(skipped) + ".")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="generate docs for a single dataset")
    args = parser.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)

    processed = []  # (entry, data)
    for entry in REGISTRY:
        if args.only and entry["name"] != args.only:
            continue
        try:
            data = process(entry)
        except Exception as exc:  # never fail the whole run
            print(f"error: {entry['name']}: {exc!r}", flush=True)
            data = None
        if data:
            processed.append((entry, data))

    for i, (entry, data) in enumerate(processed):
        prev_name = processed[i - 1][0]["name"] if i else None
        next_name = (processed[i + 1][0]["name"]
                     if i + 1 < len(processed) else None)
        if args.only:
            prev_name = next_name = None
        doc = render_doc(entry, prev_name, next_name, data)
        path = os.path.join(DOCS_DIR, f"{entry['name']}.md")
        with open(path, "w") as f:
            f.write(doc)

    if not args.only:
        for entry, data in processed:
            entry["_total"] = data["total"]
        with open(os.path.join(DOCS_DIR, "README.md"), "w") as f:
            f.write(render_readme([e for e, _ in processed]))
        with open(os.path.join(DOCS_DIR, "evaluation.md"), "w") as f:
            f.write(EVALUATION_MD)
        print(f"done: README.md + evaluation.md ({len(processed)} datasets "
              "documented)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
