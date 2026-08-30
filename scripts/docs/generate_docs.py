"""Generate documentation for the data-cleaning pipeline.

For every registered dataset this script writes scripts/docs/<name>.md with:
purpose, raw storage layout (only the columns the transform actually reads),
the transformed output contract, real row counts, and real before/after
example records. Also writes scripts/docs/README.md as the index.

Datasets whose output does not exist yet are skipped (re-runnable).
Run inside hrm_text_clean_image with HF_HOME / HF_HUB_OFFLINE set (see
scripts/docs/README.md header once generated).
"""

import argparse
import datetime
import glob as globmod
import json
import os
import sys
import tarfile

# Repo root on sys.path so `utils` is importable regardless of CWD.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

os.environ.setdefault("HF_HOME", "/mnt/hdd2/datasets_text/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT_ROOT = "/mnt/hdd2/datasets_text_transformed/HRM-Text"
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
TRUNC = 600

CONDITION_MEANING = {
    "direct": "short answer only, no reasoning shown",
    "cot": "response contains the full chain-of-thought / worked solution",
    "noisy": "content was machine-generated or scraped and not fully verified",
    "synth": "synthetically generated (model- or program-produced) data",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def truncate(value):
    """Recursively truncate long strings for display."""
    if isinstance(value, str):
        if len(value) > TRUNC:
            return value[:TRUNC] + f"\n… [truncated, {len(value)} chars total]"
        return value
    if isinstance(value, list):
        return [truncate(v) for v in value]
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def fence(text):
    n = 4
    while "`" * n in text:
        n += 1
    ticks = "`" * n
    return f"{ticks}text\n{text}\n{ticks}"


def fmt_rows(n):
    return f"{n:,}" if isinstance(n, int) else "?"


def expand_outputs(patterns):
    """Resolve output path patterns (relative to OUT_ROOT) to sorted files."""
    files = []
    for pat in patterns:
        full = os.path.join(OUT_ROOT, pat)
        if globmod.has_magic(full):
            files.extend(sorted(globmod.glob(full)))
        elif os.path.isfile(full):
            files.append(full)
    return sorted(set(files))


def count_rows(files, fmt):
    if fmt == "jsonl":
        total = 0
        for path in files:
            n = 0
            with open(path, "rb") as f:
                while chunk := f.read(1 << 24):
                    n += chunk.count(b"\n")
            total += n
        return total
    import pyarrow.parquet as pq
    total = 0
    for path in files:
        total += pq.read_metadata(path).num_rows
    return total


def output_examples(files, fmt, n=2, preferred=None):
    """First n records of the preferred (or first) output file."""
    path = preferred if preferred and os.path.isfile(preferred) else files[0]
    records = []
    if fmt == "jsonl":
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                if len(records) >= n:
                    break
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=n):
            records.extend(batch.to_pylist())
            if len(records) >= n:
                break
    return path, records[:n]


# --------------------------------------------------------------------------
# raw example loaders — each returns (source_label, dict of used fields)
# --------------------------------------------------------------------------

def _hf_first_row(spec):
    from utils import load_local_dataset
    ds = load_local_dataset(spec["repo"], spec.get("config"),
                            split=spec.get("split"))
    if isinstance(ds, dict):  # DatasetDict (no split given)
        ds = ds[sorted(ds.keys())[0]]
    match = spec.get("match")
    row = None
    if match:
        key, want = match
        for i in range(min(len(ds), 500)):
            if ds[i][key] == want:
                row = ds[i]
                break
    if row is None:
        row = ds[0]
    fields = spec["fields"]
    label = f"HF `{spec['repo']}`"
    if spec.get("config"):
        label += f" (config `{spec['config']}`)"
    if spec.get("split"):
        label += f", split `{spec['split']}`"
    return label, {k: row[k] for k in fields}


def load_hf(spec):
    label, row = _hf_first_row(spec)
    return label, row


def load_hf_flan(spec):
    """First row of the first parquet of one FLAN subset, plus the matching
    output file name (subset__task)."""
    import pyarrow.parquet as pq
    subset_dir = os.path.join(spec["input_dir"], spec["subset"])
    first_file = sorted(globmod.glob(os.path.join(subset_dir, "*.parquet")))[0]
    pf = pq.ParquetFile(first_file)
    row = next(pf.iter_batches(batch_size=1)).to_pylist()[0]
    label = f"`{spec['input_dir']}/{spec['subset']}/{os.path.basename(first_file)}`"
    data = {k: row[k] for k in spec["fields"]}
    # mirror pipe_clustered/clean_flan.py: safe_filename(task) naming
    import string
    safe = "".join(c if c in set(string.ascii_letters + string.digits + "_-. ")
                   else "_" for c in row["_task_name"])
    out_file = os.path.join(
        OUT_ROOT, spec["out_dir"], f"{spec['subset']}__{safe}.parquet")
    return label, data, out_file if os.path.isfile(out_file) else None


def load_json_list_file(spec):
    """First element of a JSON file containing a list of objects."""
    path = sorted(globmod.glob(spec["glob"]))[0]
    with open(path, "rb") as f:
        data = json.loads(f.read())
    row = data[0] if isinstance(data, list) else data
    return f"`{path}`", {k: row.get(k) for k in spec["fields"]}


def load_khan(spec):
    import orjson
    path = sorted(globmod.glob(os.path.join(spec["input_dir"], "**", "*.json"),
                               recursive=True))[0]
    with open(path, "rb") as f:
        row = orjson.loads(f.read())
    return f"`{path}`", {k: row[k] for k in spec["fields"]}


def load_txt_pairs(spec):
    """dmmath: alternating question/answer lines."""
    path = sorted(globmod.glob(os.path.join(spec["input_dir"], spec["subset"],
                                            "*.txt")))[0]
    with open(path) as f:
        question = f.readline().strip()
        answer = f.readline().strip()
    return f"`{path}`", {"question_line": question, "answer_line": answer}


def load_parquet_file(spec):
    import pyarrow.parquet as pq
    path = sorted(globmod.glob(spec["glob"]))[0]
    row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
    return f"`{path}`", {k: row[k] for k in spec["fields"]}


def load_amps_tar(spec):
    """First .txt members of amps.tar.gz (streams; does not scan the tar)."""
    found = []
    with tarfile.open(spec["tar"], "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".txt") \
                    and "/mathematica/" in member.name:
                f = tar.extractfile(member)
                if f:
                    found.append((member.name,
                                  f.read().decode("utf-8", errors="replace")))
            if len(found) >= 1:
                break
    if not found:
        raise RuntimeError("no mathematica .txt member found in tar")
    name, content = found[0]
    return f"`{spec['tar']}` member `{name}`", {"raw_content": content}


def load_sudoku_csv(spec):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset")
    with open(path, newline="") as f:
        header = f.readline().rstrip("\n").split(",")
        first = f.readline().rstrip("\n").split(",")
    return (f"HF `{spec['repo']}` file `{spec['file']}`",
            dict(zip(header, first)))


RAW_LOADERS = {
    "hf": load_hf,
    "json_list_file": load_json_list_file,
    "khan_json": load_khan,
    "txt_pairs": load_txt_pairs,
    "parquet_file": load_parquet_file,
    "amps_tar": load_amps_tar,
    "sudoku_csv": load_sudoku_csv,
}


# --------------------------------------------------------------------------
# dataset registry
# --------------------------------------------------------------------------

D = "/mnt/hdd2/datasets_text"

REGISTRY = [
    {
        "name": "gsm8k_train",
        "script": "pipe/clean_gsm8k_train.py",
        "purpose": "Grade-school math word problems (GSM8K train split). "
                   "Teaches short arithmetic problem solving with a bare final "
                   "numeric answer — the annotated calculator steps are discarded.",
        "source": f"HF `openai/gsm8k` (config `main`, split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("question", "string", "the word problem"),
            ("answer", "string", "worked solution ending in `#### <final>`; only the part after `####` is kept"),
        ],
        "raw_example": {"kind": "hf", "repo": "openai/gsm8k", "config": "main",
                        "split": "train", "fields": ["question", "answer"]},
        "outputs": ["data/gsm8k_train.jsonl"],
        "out_format": "jsonl",
        "conditions": [("direct", "response is the final numeric answer only")],
    },
    {
        "name": "math_train",
        "script": "pipe/clean_math_train.py",
        "purpose": "MATH (Hendrycks) competition problems, 7 subjects, train "
                   "split. Teaches competition-level math: each problem yields a "
                   "full worked LaTeX solution (cot) plus the extracted boxed "
                   "final answer (direct).",
        "source": "HF `EleutherAI/hendrycks_math` (7 subject configs, split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("problem", "string", "LaTeX problem statement"),
            ("solution", "string", "worked solution containing `\\boxed{...}`"),
        ],
        "raw_example": {"kind": "hf", "repo": "EleutherAI/hendrycks_math",
                        "config": "algebra", "split": "train",
                        "fields": ["problem", "solution"]},
        "outputs": ["data/math_train.jsonl"],
        "out_format": "jsonl",
        "conditions": [
            ("cot", "full worked LaTeX solution"),
            ("direct", "content of the last `\\boxed{...}` in the solution (row omitted when no boxed answer)"),
        ],
    },
    {
        "name": "natural_reasoning",
        "script": "pipe/clean_natural_reasoning.py",
        "purpose": "General-knowledge reasoning questions with reference "
                   "answers (facebook/natural_reasoning). Rows asking to "
                   "'prove'/'show that' and empty answers are filtered out.",
        "source": "HF `facebook/natural_reasoning` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("question", "string", "the question"),
            ("reference_answer", "string", "reference answer (stripped; empty -> dropped)"),
        ],
        "raw_example": {"kind": "hf", "repo": "facebook/natural_reasoning",
                        "split": "train",
                        "fields": ["question", "reference_answer"]},
        "outputs": ["data/natural_reasoning.jsonl"],
        "out_format": "jsonl",
        "conditions": [("noisy,direct", "reference answer as-is; quality not verified")],
    },
    {
        "name": "no_robots",
        "script": "pipe/clean_no_robots.py",
        "purpose": "Human-written instruction-following conversations "
                   "(HuggingFaceH4/no_robots). Only the first user->assistant "
                   "turn is kept; an optional system prompt is prepended to the "
                   "instruction. Broad human-quality SFT data.",
        "source": "HF `HuggingFaceH4/no_robots` (all splits)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("messages", "list[{role, content}]", "conversation turns; only the optional system message + first user/assistant pair are read"),
        ],
        "raw_example": {"kind": "hf", "repo": "HuggingFaceH4/no_robots",
                        "split": "train", "fields": ["messages"]},
        "outputs": ["data/no_robots.jsonl"],
        "out_format": "jsonl",
        "conditions": [("cot", "free-form human-written response (tagged cot by convention)")],
    },
    {
        "name": "numinamath",
        "script": "pipe/clean_numinamath.py",
        "purpose": "Large competition/olympiad math corpus (NuminaMath-1.5). "
                   "Synthetic rows, invalid problems/solutions, and rows with "
                   "URLs or translation artifacts are dropped. Each kept row "
                   "yields the full solution plus, for non-proof rows, the short "
                   "final answer.",
        "source": "HF `AI-MO/NuminaMath-1.5` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("problem", "string", "the problem"),
            ("solution", "string", "worked solution"),
            ("answer", "string", "short final answer ('proof' for proofs)"),
            ("synthetic", "bool", "filter: synthetic rows are dropped"),
            ("problem_is_valid / solution_is_valid", "string", "filter: must both be 'Yes'"),
            ("question_type", "string", "filter: 'proof' rows get no direct record"),
        ],
        "raw_example": {"kind": "hf", "repo": "AI-MO/NuminaMath-1.5",
                        "split": "train",
                        "fields": ["problem", "solution", "answer",
                                   "synthetic", "problem_is_valid",
                                   "solution_is_valid", "question_type"]},
        "outputs": ["data/numinamath.jsonl"],
        "out_format": "jsonl",
        "conditions": [
            ("noisy,cot", "full solution, source quality unverified"),
            ("noisy,direct", "short answer only, source quality unverified"),
        ],
    },
    {
        "name": "omnimath",
        "script": "pipe/clean_omnimath.py",
        "purpose": "Omni-MATH olympiad-level problems (the test split is reused "
                   "as training data). Each problem yields both the full "
                   "solution (cot) and the short final answer (direct).",
        "source": "HF `KbsdJames/Omni-MATH` (split `test`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("problem", "string", "the problem"),
            ("solution", "string", "worked solution"),
            ("answer", "string", "short final answer"),
        ],
        "raw_example": {"kind": "hf", "repo": "KbsdJames/Omni-MATH",
                        "split": "test",
                        "fields": ["problem", "solution", "answer"]},
        "outputs": ["data/omnimath.jsonl"],
        "out_format": "jsonl",
        "conditions": [
            ("cot", "full worked solution"),
            ("direct", "short final answer"),
        ],
    },
    {
        "name": "principia_collection",
        "script": "pipe/clean_principia_collection.py",
        "purpose": "Synthetic STEM problems generated from textbook/exam "
                   "material (facebook/principia-collection). Straight "
                   "question->answer pairs.",
        "source": "HF `facebook/principia-collection` (all splits)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("problem_statement", "string", "the problem"),
            ("answer", "string", "the answer"),
        ],
        "raw_example": {"kind": "hf", "repo": "facebook/principia-collection",
                        "fields": ["problem_statement", "answer"]},
        "outputs": ["data/principia_collection.jsonl"],
        "out_format": "jsonl",
        "conditions": [("synth,direct", "synthetic question->answer pair")],
    },
    {
        "name": "webinstruct_verified",
        "script": "pipe/clean_webinstruct_verified.py",
        "purpose": "Web-mined QA pairs whose answers were verified by LLM "
                   "judges (TIGER-Lab/WebInstruct-verified). Broad-domain direct "
                   "supervision.",
        "source": "HF `TIGER-Lab/WebInstruct-verified` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("question", "string", "the question"),
            ("answer", "string", "verified answer"),
        ],
        "raw_example": {"kind": "hf", "repo": "TIGER-Lab/WebInstruct-verified",
                        "split": "train", "fields": ["question", "answer"]},
        "outputs": ["data/webinstruct_verified.jsonl"],
        "out_format": "jsonl",
        "conditions": [("direct", "verified answer, no reasoning shown")],
    },
    {
        "name": "amps_khan",
        "script": "pipe/clean_amps_khan.py",
        "purpose": "Khan Academy exercises from the AMPS dataset (one JSON file "
                   "per problem). The step hints are joined into the response — "
                   "hint quality varies, hence noisy.",
        "source": f"`{D}/amps/khan/**/*.json` (extracted from `{D}/amps.tar.gz`)",
        "raw_format": "one JSON object per file",
        "raw_fields": [
            ("problem", "string", "the exercise text"),
            ("hints", "list[string]", "step hints, joined with newlines into the response"),
        ],
        "raw_example": {"kind": "khan_json", "input_dir": f"{D}/amps/khan",
                        "fields": ["problem", "hints"]},
        "outputs": ["data/amps_khan.jsonl"],
        "out_format": "jsonl",
        "conditions": [("noisy,cot", "hint sequence as pseudo-solution, quality unverified")],
    },
    {
        "name": "arb",
        "script": "pipe/clean_platypus/clean_arb.py",
        "purpose": "ARB advanced reasoning benchmark (math, physics, science, "
                   "reading, law) as distributed with Platypus. A fixed task "
                   "description is prepended to each problem depending on the "
                   "subject file.",
        "source": f"`{D}/Platypus/ARB/*.json` (5 subject files)",
        "raw_format": "JSON list of objects per subject file",
        "raw_fields": [
            ("instruction", "string", "the problem (a subject-specific description is prepended)"),
            ("response", "string", "the solution"),
        ],
        "raw_example": {"kind": "json_list_file",
                        "glob": f"{D}/Platypus/ARB/math.json",
                        "fields": ["instruction", "response"]},
        "outputs": ["data/Platypus/arb_*.jsonl"],
        "out_format": "jsonl",
        "conditions": [
            ("cot", "math / reading / science / physics: worked solution"),
            ("direct", "law: correct option letter only"),
        ],
    },
    {
        "name": "openbookqa",
        "script": "pipe/clean_platypus/clean_openbookqa.py",
        "purpose": "OpenBookQA elementary science multiple-choice questions "
                   "('additional' config with the supporting fact1). Question, "
                   "options and fact are rendered into one instruction.",
        "source": "HF `allenai/openbookqa` (config `additional`, splits `train`+`validation`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("question_stem", "string", "the question"),
            ("fact1", "string", "supporting fact appended to the instruction"),
            ("choices", "{text: list[string], label: list[string]}", "answer options, rendered as A:/B:/C:/D:"),
            ("answerKey", "string", "correct option letter (the response)"),
        ],
        "raw_example": {"kind": "hf", "repo": "allenai/openbookqa",
                        "config": "additional", "split": "train",
                        "fields": ["question_stem", "fact1", "choices",
                                   "answerKey"]},
        "outputs": ["data/Platypus/openbookqa.jsonl"],
        "out_format": "jsonl",
        "conditions": [("direct", "correct option letter")],
    },
    {
        "name": "reclor",
        "script": "pipe/clean_platypus/clean_reclor.py",
        "purpose": "ReClor logical reasoning multiple-choice questions "
                   "(LSAT/GMAT style). Context, question and options are "
                   "rendered into one instruction.",
        "source": "HF `metaeval/reclor` (splits `train`+`validation`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("context", "string", "passage the question refers to"),
            ("question", "string", "the question"),
            ("answers", "list[string]", "options, rendered as A:/B:/C:/D:"),
            ("label", "int", "index of the correct option (response is the letter)"),
        ],
        "raw_example": {"kind": "hf", "repo": "metaeval/reclor", "split": "train",
                        "fields": ["context", "question", "answers", "label"]},
        "outputs": ["data/Platypus/reclor.jsonl"],
        "out_format": "jsonl",
        "conditions": [("direct", "correct option letter")],
    },
    {
        "name": "scibench",
        "script": "pipe/clean_platypus/clean_scibench.py",
        "purpose": "SciBench college-level scientific problems (per-textbook "
                   "JSON files). Emits the worked solution when present (cot) "
                   "and always the final numeric answer (direct).",
        "source": f"`{D}/Platypus/scibench/dataset/original/*.json`",
        "raw_format": "JSON list of objects per textbook file",
        "raw_fields": [
            ("problem_text", "string", "the problem"),
            ("solution", "string", "worked solution (cot record emitted only when non-empty)"),
            ("answer_latex", "string", "final answer in LaTeX (falls back to answer_number)"),
            ("answer_number", "string", "numeric final answer; used when answer_latex is redundant"),
        ],
        "raw_example": {"kind": "json_list_file",
                        "glob": f"{D}/Platypus/scibench/dataset/original/*.json",
                        "fields": ["problem_text", "solution", "answer_latex",
                                   "answer_number"]},
        "outputs": ["data/Platypus/scibench.jsonl"],
        "out_format": "jsonl",
        "conditions": [
            ("cot", "worked solution (only when the source has one)"),
            ("direct", "final answer (answer_latex, simplified when it duplicates answer_number)"),
        ],
    },
    {
        "name": "scienceqa",
        "script": "pipe/clean_platypus/clean_scienceqa.py",
        "purpose": "ScienceQA text-only multiple-choice science questions. "
                   "Rows with a rationale yield a cot record (rationale + answer "
                   "letter); every row also yields a direct record (bare letter). "
                   "A lecture, when present, is appended to the instruction.",
        "source": "HF `metaeval/ScienceQA_text_only` (splits `train`+`validation`+`test`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("question", "string", "the question"),
            ("choices", "list[string]", "options, rendered as A:/B:/C:/..."),
            ("lecture", "string", "background text, appended to the instruction when non-empty"),
            ("solution", "string", "rationale; when empty no cot record is emitted"),
            ("answer", "int", "index of the correct option (response is the letter)"),
        ],
        "raw_example": {"kind": "hf", "repo": "metaeval/ScienceQA_text_only",
                        "split": "train",
                        "fields": ["question", "choices", "lecture", "solution",
                                   "answer"]},
        "outputs": ["data/Platypus/scienceqa.jsonl"],
        "out_format": "jsonl",
        "conditions": [
            ("cot", "rationale + 'Answer: X'"),
            ("direct", "bare option letter"),
        ],
    },
    {
        "name": "theoremqa",
        "script": "pipe/clean_platypus/clean_theoremqa.py",
        "purpose": "TheoremQA university-level math/science questions (test "
                   "split). Rows containing a picture are dropped; the rest are "
                   "direct question->answer pairs.",
        "source": "HF `TIGER-Lab/TheoremQA` (split `test`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("Question", "string", "the question"),
            ("Answer", "string", "the answer"),
            ("Picture", "string|null", "filter only: rows with a picture are dropped"),
        ],
        "raw_example": {"kind": "hf", "repo": "TIGER-Lab/TheoremQA",
                        "split": "test",
                        "fields": ["Question", "Answer", "Picture"]},
        "outputs": ["data/Platypus/theoremqa.jsonl"],
        "out_format": "jsonl",
        "conditions": [("direct", "final answer only")],
    },
    {
        "name": "flan",
        "script": "pipe_clustered/clean_flan.py",
        "purpose": "FLAN v2 instruction-tuning collection (Open-Orca parquet "
                   "dump). 14 subsets are included; every output parquet holds "
                   "one (subset, task) pair. Few-shot/zero-shot option subsets "
                   "are tagged direct, the two cot_* subsets cot.",
        "source": f"`{D}/Open-Orca/FLAN/<subset>/*.parquet`",
        "raw_format": "parquet files per subset",
        "raw_fields": [
            ("_task_name", "string", "source task id; becomes part of the output filename"),
            ("inputs", "string", "prompt text -> instruction"),
            ("targets", "string", "target text -> response"),
        ],
        "raw_example": {"kind": "flan", "input_dir": f"{D}/Open-Orca/FLAN",
                        "subset": "cot_fsopt_data",
                        "fields": ["_task_name", "inputs", "targets"],
                        "out_dir": "data_clustered/flan"},
        "outputs": ["data_clustered/flan/*.parquet"],
        "out_format": "parquet",
        "conditions": [
            ("direct", "12 subsets: dialog/flan/niv2/t0 fsopt+fsnoopt+zsopt+zsnoopt"),
            ("cot", "2 subsets: cot_fsopt_data, cot_zsopt_data"),
        ],
    },
    {
        "name": "synth",
        "script": "pipe_clustered/clean_SYNTH.py",
        "purpose": "PleIAs SYNTH: a large synthetic instruction dataset. Kept "
                   "English-only; self-knowledge queries and cooking exercises "
                   "filtered out. The condition tag is derived from the exercise "
                   "type.",
        "source": f"`{D}/PleIAs/SYNTH/*.parquet`",
        "raw_format": "parquet files",
        "raw_fields": [
            ("query", "string", "the prompt (constraints appended for rag exercises)"),
            ("constraints", "string", "extra requirements (rag exercises only)"),
            ("synthetic_answer", "string", "generated answer -> response"),
            ("exercise", "string", "exercise type; drives the condition tag and the 'cooking' filter"),
            ("language", "string", "filter: only 'en' kept"),
            ("query_seed_url", "string", "filter: 'Pleias self-knowledge' rows dropped"),
        ],
        "raw_example": {"kind": "parquet_file",
                        "glob": f"{D}/PleIAs/SYNTH/*.parquet",
                        "fields": ["query", "constraints", "synthetic_answer",
                                   "exercise", "language", "query_seed_url"]},
        "outputs": ["data_clustered/synth/*.parquet"],
        "out_format": "parquet",
        "conditions": [
            ("synth,cot", "creative writing / rag / memorization / constrained writing / editing"),
            ("synth,direct", "math mcq / mcq"),
            ("synth,noisy,cot", "math exercise"),
        ],
    },
    {
        "name": "dmmath",
        "script": "pipe_clustered/clean_dmmath.py",
        "purpose": "DeepMind mathematics_dataset-v1.0: procedurally generated "
                   "school-level math across ~120 task types and 3 difficulty "
                   "tiers. Each .txt file holds alternating question/answer "
                   "lines; one output parquet per (tier, task).",
        "source": f"`{D}/mathematics_dataset-v1.0/{{train-easy,train-medium,train-hard}}/*.txt`",
        "raw_format": "plain text: line 2k = question, line 2k+1 = answer",
        "raw_fields": [
            ("question_line", "string", "odd lines: the question"),
            ("answer_line", "string", "even lines: the short answer"),
        ],
        "raw_example": {"kind": "txt_pairs",
                        "input_dir": f"{D}/mathematics_dataset-v1.0",
                        "subset": "train-easy"},
        "outputs": ["data_clustered/dmmath/*.parquet"],
        "out_format": "parquet",
        "conditions": [("direct", "generated short answer")],
    },
    {
        "name": "acereason",
        "script": "pipe_clustered/clean_acereason.py",
        "purpose": "AceReason-1.1 SFT (nvidia), math category only. The "
                   "<think>...</think> block is stripped from each response, "
                   "leaving the final solution text.",
        "source": "HF `nvidia/AceReason-1.1-SFT` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("input", "string", "the problem -> instruction"),
            ("output", "string", "response with a <think> block (stripped)"),
            ("category", "string", "filter: only 'math' rows kept"),
        ],
        "raw_example": {"kind": "hf", "repo": "nvidia/AceReason-1.1-SFT",
                        "split": "train", "match": ("category", "math"),
                        "fields": ["input", "output", "category"]},
        "outputs": ["data_clustered/acereason/all.parquet"],
        "out_format": "parquet",
        "conditions": [("synth,cot", "model-generated reasoning, think block removed")],
    },
    {
        "name": "ampsmathematica",
        "script": "pipe_clustered/clean_ampsmathematica.py",
        "purpose": "AMPS Mathematica synthetic math exercises, read directly "
                   "from the tar archive. Files under a `*_w_steps` task folder "
                   "carry step-by-step answers (cot), the rest final answers "
                   "(direct). Output is grouped one parquet per topic_subtask.",
        "source": f"`{D}/amps.tar.gz` (members `amps/mathematica/<topic>/<task>/*.txt`)",
        "raw_format": "gzipped tar of small .txt files: 'Problem: ... Answer: ...'",
        "raw_fields": [
            ("raw_content", "string", "whole file: 'Problem:' prefix stripped, split on the first 'Answer:'"),
        ],
        "raw_example": {"kind": "amps_tar", "tar": f"{D}/amps.tar.gz"},
        "outputs": ["data_clustered/ampsmathematica/*.parquet"],
        "out_format": "parquet",
        "conditions": [
            ("noisy,cot", "task folder ends with `_w_steps`: worked steps"),
            ("noisy,direct", "other task folders: final answer only"),
        ],
    },
    {
        "name": "openmathinstruct2",
        "script": "pipe_clustered/clean_openmathinstruct2.py",
        "purpose": "OpenMathInstruct-2 (nvidia): ~14M synthetic math solutions. "
                   "Every row goes to cot.parquet (full generated solution); "
                   "rows not derived from the original math/gsm8k data also go "
                   "to direct.parquet (short expected answer).",
        "source": "HF `nvidia/OpenMathInstruct-2` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("problem", "string", "the problem -> instruction"),
            ("generated_solution", "string", "model-generated solution (cot.parquet response)"),
            ("expected_answer", "string", "short answer (direct.parquet response)"),
            ("problem_source", "string", "filter: rows from 'math'/'gsm8k' are excluded from direct.parquet"),
        ],
        "raw_example": {"kind": "hf", "repo": "nvidia/OpenMathInstruct-2",
                        "split": "train",
                        "fields": ["problem", "generated_solution",
                                   "expected_answer", "problem_source"]},
        "outputs": ["data_clustered/openmathinstruct2/cot.parquet",
                    "data_clustered/openmathinstruct2/direct.parquet"],
        "out_format": "parquet",
        "conditions": [
            ("synth,cot", "cot.parquet: full generated solution"),
            ("synth,direct", "direct.parquet: short expected answer (non-original sources only)"),
        ],
    },
    {
        "name": "openthoughts2",
        "script": "pipe_clustered/clean_openthoughts2.py",
        "purpose": "OpenThoughts2-1M reasoning traces. Code-related sources and "
                   "code-looking rows are filtered out; <think> blocks are "
                   "stripped from the assistant reply.",
        "source": "HF `open-thoughts/OpenThoughts2-1M` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("conversations", "list[{from, value}]", "exactly one user + one assistant turn; both used"),
            ("source", "string", "filter: code/math-duplicate sources dropped (dolphin, magicoder, sharegpt, nvidia_math, ...)"),
        ],
        "raw_example": {"kind": "hf", "repo": "open-thoughts/OpenThoughts2-1M",
                        "split": "train",
                        "fields": ["conversations", "source"]},
        "outputs": ["data_clustered/openthoughts2/all.parquet"],
        "out_format": "parquet",
        "conditions": [("synth,cot", "reasoning trace with <think> block removed")],
    },
    {
        "name": "sudoku_extreme",
        "script": "pipe_clustered/clean_sudoku.py",
        "purpose": "sudoku-extreme: millions of 81-cell Sudoku puzzles. The "
                   "puzzle string ('.' -> '0') gets a fixed prompt prefix; the "
                   "response is the solved grid. Teaches long-horizon constraint "
                   "reasoning.",
        "source": "HF `sapientinc/sudoku-extreme` (file `train.csv`)",
        "raw_format": "CSV: source,question,answer (81-char strings)",
        "raw_fields": [
            ("question", "string", "puzzle, 81 chars, '.' = empty cell"),
            ("answer", "string", "solved grid, 81 chars"),
        ],
        "raw_example": {"kind": "sudoku_csv", "repo": "sapientinc/sudoku-extreme",
                        "file": "train.csv"},
        "outputs": ["data_clustered/sudoku_extreme/all.parquet"],
        "out_format": "parquet",
        "conditions": [("direct", "solved grid, no reasoning")],
    },
    {
        "name": "tasksource",
        "script": "pipe_clustered/clean_tasksource.py",
        "purpose": "tasksource-instruct-v0: ~200 curated NLP tasks (NLI, "
                   "classification, tagging, MCQ). Rows outside the curated "
                   "TASK_SET are dropped; one output parquet per task.",
        "source": "HF `tasksource/tasksource-instruct-v0` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("task", "string", "task id; filter (TASK_SET) and output filename"),
            ("inputs", "string", "prompt -> instruction"),
            ("targets", "string", "target -> response (trailing '.' removed)"),
        ],
        "raw_example": {"kind": "hf", "repo": "tasksource/tasksource-instruct-v0",
                        "split": "train", "match": ("task", "reclor"),
                        "fields": ["task", "inputs", "targets"]},
        "outputs": ["data_clustered/tasksource/*.parquet"],
        "out_format": "parquet",
        "conditions": [("direct", "short task target")],
    },
    {
        "name": "textbookreasoning",
        "script": "pipe_clustered/clean_textbookreasoning.py",
        "purpose": "TextbookReasoning (MegaScience): QA extracted from "
                   "textbooks. Every row goes to cot.parquet (full answer); "
                   "non-proof rows also go to direct.parquet (short reference "
                   "answer).",
        "source": "HF `MegaScience/TextbookReasoning` (split `train`)",
        "raw_format": "HF dataset (arrow) in the prefetched local cache",
        "raw_fields": [
            ("question", "string", "the question"),
            ("answer", "string", "full answer (cot.parquet response)"),
            ("reference_answer", "string", "short answer (direct.parquet response; 'prove'/'show that' questions excluded)"),
        ],
        "raw_example": {"kind": "hf", "repo": "MegaScience/TextbookReasoning",
                        "split": "train",
                        "fields": ["question", "answer", "reference_answer"]},
        "outputs": ["data_clustered/textbookreasoning/cot.parquet",
                    "data_clustered/textbookreasoning/direct.parquet"],
        "out_format": "parquet",
        "conditions": [
            ("synth,cot", "cot.parquet: full extracted answer"),
            ("noisy,direct", "direct.parquet: short reference answer"),
        ],
    },
]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_doc(entry, out_files, n_rows, raw_label, raw_record, raw_note,
               ex_file, ex_records):
    lines = []
    lines.append(f"# {entry['name']}")
    lines.append("")
    lines.append(f"**Script:** `{entry['script']}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(entry["purpose"])
    lines.append("")
    lines.append("## Before (raw storage)")
    lines.append("")
    lines.append(f"- Source: {entry['source']}")
    lines.append(f"- Format: {entry['raw_format']}")
    lines.append("")
    lines.append("Columns actually read by the transform:")
    lines.append("")
    lines.append("| column | type | meaning |")
    lines.append("|---|---|---|")
    for col, typ, desc in entry["raw_fields"]:
        lines.append(f"| `{col}` | {typ} | {desc} |")
    lines.append("")
    lines.append("## After (transformed)")
    lines.append("")
    if entry["out_format"] == "jsonl":
        fmt_desc = "JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)"
    else:
        fmt_desc = "parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)"
    locs = ", ".join(f"`{p}`" for p in entry["outputs"])
    lines.append(f"- Location: {locs} (under `{OUT_ROOT}`; {len(out_files)} file(s))")
    lines.append(f"- Format: {fmt_desc}")
    lines.append(f"- Rows: {fmt_rows(n_rows)}")
    lines.append("")
    lines.append("`condition` values used here:")
    lines.append("")
    for value, meaning in entry["conditions"]:
        lines.append(f"- `{value}` — {meaning}")
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    lines.append("Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).")
    lines.append("")
    lines.append(f"### Raw record ({raw_label})")
    if raw_note:
        lines.append(f"\n_{raw_note}_")
    lines.append("")
    lines.append(fence(json.dumps(truncate(raw_record), indent=2,
                                  ensure_ascii=False)))
    lines.append("")
    for i, rec in enumerate(ex_records):
        rel = os.path.relpath(ex_file, OUT_ROOT)
        suffix = f", record {i + 1}" if len(ex_records) > 1 else ""
        lines.append(f"### Transformed record (`{rel}`{suffix})")
        lines.append("")
        lines.append(fence(json.dumps(truncate(rec), indent=2,
                                      ensure_ascii=False)))
        lines.append("")
    return "\n".join(lines)


def generate(entry):
    name = entry["name"]
    out_files = expand_outputs(entry["outputs"])
    if not out_files:
        print(f"skip: {name} (no output yet)", flush=True)
        return None

    n_rows = count_rows(out_files, entry["out_format"])

    spec = entry["raw_example"]
    raw_note = None
    preferred = None
    if spec["kind"] == "flan":
        raw_label, raw_record, preferred = load_hf_flan(spec)
        raw_note = ("first row of the subset's first parquet file; the "
                    "transformed record below is row 1 of the parquet for that "
                    "exact (subset, task) pair")
    else:
        raw_label, raw_record = RAW_LOADERS[spec["kind"]](spec)

    ex_file, ex_records = output_examples(out_files, entry["out_format"],
                                          n=2, preferred=preferred)

    doc = render_doc(entry, out_files, n_rows, raw_label, raw_record, raw_note,
                     ex_file, ex_records)
    path = os.path.join(DOCS_DIR, f"{name}.md")
    with open(path, "w") as f:
        f.write(doc)
    print(f"done: {name} ({fmt_rows(n_rows)} rows, {len(out_files)} files) -> "
          f"{os.path.relpath(path)}", flush=True)
    return {"name": name, "n_rows": n_rows, "entry": entry}


def render_readme(results):
    lines = []
    lines.append("# Data-cleaning pipeline — dataset docs")
    lines.append("")
    lines.append("Generated by `scripts/docs/generate_docs.py` on "
                 f"{datetime.date.today().isoformat()} (row counts and examples "
                 "are computed from the actual files; re-run the script to refresh).")
    lines.append("")
    lines.append("## Unified output contract")
    lines.append("")
    lines.append("Every cleaning script (`pipe/*.py`, `pipe/clean_platypus/*.py`, "
                 "`pipe_clustered/*.py`) reduces one raw dataset to records of "
                 "the same shape:")
    lines.append("")
    lines.append("- `instruction` (string) — the prompt")
    lines.append("- `response` (string) — the target")
    lines.append("- `condition` (string) — comma-separated tags describing the record")
    lines.append("")
    lines.append(f"Storage: JSONL under `{OUT_ROOT}/data/` (and `data/Platypus/`), "
                 f"parquet under `{OUT_ROOT}/data_clustered/<name>/`. "
                 "Clustered datasets are split into many parquet files "
                 "(one per task/subset) so large mixes can be read in parallel "
                 "and per-task.")
    lines.append("")
    lines.append("`condition` tags:")
    lines.append("")
    for tag, meaning in CONDITION_MEANING.items():
        lines.append(f"- `{tag}` — {meaning}")
    lines.append("")
    lines.append("Why the transform exists: the raw datasets are heterogeneous "
                 "(HF datasets, JSON dumps, tar archives, line-based txt, CSV). "
                 "Unifying them into one schema lets the training code sample "
                 "and mix data by condition tags (e.g. how much cot vs direct, "
                 "how much noisy/synthetic data) without any per-dataset code.")
    lines.append("")
    lines.append("## Datasets")
    lines.append("")
    lines.append("| dataset | source | condition tags | rows | doc |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        e = r["entry"]
        tags = ", ".join(f"`{c}`" for c, _ in e["conditions"])
        lines.append(f"| {e['name']} | {e['source']} | {tags} | "
                     f"{fmt_rows(r['n_rows'])} | [{e['name']}.md]({e['name']}.md) |")
    lines.append("")
    lines.append("Datasets not listed above had no transformed output yet when "
                 "this file was generated (the cleaning queue is still running); "
                 "the generator skips them and picks them up on the next run.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="generate docs for a single dataset")
    args = parser.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)
    results = []
    for entry in REGISTRY:
        if args.only and entry["name"] != args.only:
            continue
        try:
            r = generate(entry)
        except Exception as exc:  # never fail the whole run
            print(f"error: {entry['name']}: {exc!r}", flush=True)
            r = None
        if r:
            results.append(r)

    if not args.only:
        with open(os.path.join(DOCS_DIR, "README.md"), "w") as f:
            f.write(render_readme(results))
        print(f"done: README.md ({len(results)} datasets documented)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
