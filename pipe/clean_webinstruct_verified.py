import argparse
import os
import time

import orjson
from tqdm import tqdm

from utils import load_local_dataset

DATASET_ID = "TIGER-Lab/WebInstruct-verified"
SPLIT = "train"
BATCH_SIZE = 8192


def _to_hrm(batch):
    n = len(batch["question"])
    return {
        "condition": ["direct"] * n,
        "instruction": batch["question"],
        "response": batch["answer"],
    }


def clean_webinstruct_verified(output_path: str, workers: int) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    dataset = load_local_dataset(DATASET_ID, split=SPLIT)
    n_rows = len(dataset)
    print(f"Loaded {DATASET_ID} ({SPLIT}): {n_rows} rows "
          f"({workers} workers)", flush=True)

    # Batched, multi-process transform; datasets.map preserves row order.
    transformed = dataset.map(
        _to_hrm,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=workers,
        remove_columns=dataset.column_names,
        desc="transform",
    )

    # Incremental write: one JSON object per line, never holding the full
    # result list in RAM (same format as utils.write_jsonl).
    n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
    n_written = 0
    with open(output_path, "wb") as f, tqdm(
            total=n_rows, desc="write", unit="row", unit_scale=True) as bar:
        for i, batch in enumerate(transformed.iter(batch_size=BATCH_SIZE), 1):
            t0 = time.time()
            f.write(b"".join(
                orjson.dumps({
                    "condition": condition,
                    "instruction": question,
                    "response": answer,
                }) + b"\n"
                for condition, question, answer in zip(
                    batch["condition"], batch["instruction"], batch["response"])
            ))
            n = len(batch["condition"])
            n_written += n
            bar.update(n)
            tqdm.write(f"[{i}/{n_batches}] webinstruct_verified: "
                       f"{n} rows in {time.time() - t0:.1f}s")

    elapsed = time.time() - started
    print(f"Done: {n_written} rows -> {output_path} in {elapsed:.1f}s",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/webinstruct_verified.jsonl',
        help='absolute path to the output .jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='num_proc for the batched transform (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_webinstruct_verified(
        output_path=args.output_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
