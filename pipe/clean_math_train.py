import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from utils import load_local_dataset, write_jsonl


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()


DATASET_NAME = "EleutherAI/hendrycks_math"


def _init_worker():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _process_subset(subset):
    """Load one subset (train split) and build its records.

    Returns (subset_name, records, seconds). Record order within the
    subset matches the original row order (cot then direct per row).
    """
    t0 = time.time()
    dataset = load_local_dataset(DATASET_NAME, subset, split="train")

    records = []
    for row in dataset:
        records.append({
            "condition": "cot",
            "instruction": row["problem"],
            "response": row["solution"].strip()
        })

        # No CoT variant
        ground_truth_answer = _last_boxed_only_string(row["solution"])
        if ground_truth_answer:
            records.append({
                "condition": "direct",
                "instruction": row["problem"],
                "response": ground_truth_answer.strip()
            })

    return subset, records, time.time() - t0


def clean_math_train(output_path: str, workers: int):
    started = time.time()

    # Hardcoded so the script works fully offline (HF_HUB_OFFLINE=1) once the
    # dataset has been prefetched by the download stage; get_dataset_config_names
    # would require network access.
    subsets = ["algebra", "counting_and_probability", "geometry",
               "intermediate_algebra", "number_theory", "prealgebra",
               "precalculus"]
    print(f"Subsets: {subsets}")

    # 2. Load subsets in parallel; results are re-assembled in the original
    # subset order so the output is byte-identical to sequential processing.
    results = [None] * len(subsets)
    with ProcessPoolExecutor(
            max_workers=max(1, min(workers, len(subsets))),
            initializer=_init_worker) as pool:
        futures = {pool.submit(_process_subset, s): i
                   for i, s in enumerate(subsets)}
        with tqdm(total=len(subsets), desc="subsets") as bar:
            for i, future in enumerate(as_completed(futures), 1):
                subset, records, secs = future.result()
                results[futures[future]] = records
                bar.update(1)
                bar.set_postfix(subset=subset, done=f"{i}/{len(subsets)}")
                tqdm.write(f"[{i}/{len(subsets)}] {subset}: "
                           f"{len(records)} records in {secs:.1f}s")

    result = [record for records in results for record in records]

    print(f"Total records loaded: {len(result)}")
    write_jsonl(output_path, result)
    elapsed = time.time() - started
    print(f"Done: {len(result)} records -> {output_path} in {elapsed:.1f}s",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/math_train.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of subsets processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_math_train(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
