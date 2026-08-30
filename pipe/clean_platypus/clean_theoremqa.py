import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from utils import load_local_dataset, write_jsonl

DATASET_NAME = "TIGER-Lab/TheoremQA"
# Below this many rows multiprocessing costs more than it saves.
MAP_MIN_ROWS = 100_000


def transform_batch(batch):
    # Keep only rows without a picture, matching the original `row['Picture'] is None`.
    conditions, instructions, responses = [], [], []
    for question, answer, picture in zip(
            batch["Question"], batch["Answer"], batch["Picture"]):
        if picture is None:
            conditions.append("direct")
            instructions.append(question)
            responses.append(answer)
    return {"condition": conditions, "instruction": instructions, "response": responses}


def clean_theoremqa(output_path: str, workers: int):
    started = time.time()

    dataset = load_local_dataset(DATASET_NAME, split="test")
    total = len(dataset)
    print(f"Loaded {total} rows from {DATASET_NAME} (test)", flush=True)

    # Batched map preserves the original record order (also with num_proc > 1).
    mapped = dataset.map(
        transform_batch,
        batched=True,
        batch_size=1000,
        num_proc=workers if workers > 1 and total >= MAP_MIN_ROWS else None,
        remove_columns=dataset.column_names,
        desc="formatting",
    )
    n_rows = len(mapped)

    write_jsonl(output_path, tqdm(mapped, total=n_rows, desc="writing", unit="rows"))

    elapsed = time.time() - started
    print(f"[1/1] {os.path.basename(output_path)}: {n_rows} rows in {elapsed:.0f}s",
          flush=True)
    print(f"Done: {n_rows} rows -> {output_path} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus/theoremqa.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of processes for datasets.map on large inputs '
             '(default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_theoremqa(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
