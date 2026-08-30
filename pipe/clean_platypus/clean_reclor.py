import argparse
import os
import time

from datasets import concatenate_datasets
from tqdm import tqdm

from utils import load_local_dataset, write_jsonl

DATASET_NAME = "metaeval/reclor"
SPLITS = ("train", "validation")
OUTPUT_FILENAME = "reclor.jsonl"


def format_batch(batch):
    conditions, instructions, responses = [], [], []
    for context, question, answers, label in zip(
            batch["context"], batch["question"], batch["answers"], batch["label"]):
        instruction = f"{question}\n\n{context}\n\nOptions:"
        for i, ans in enumerate(answers):
            instruction += "\n" + chr(65 + i) + ": " + ans
        instructions.append(instruction)
        conditions.append("direct")
        responses.append(chr(65 + label))
    return {"condition": conditions, "instruction": instructions, "response": responses}


def clean_reclor(output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, OUTPUT_FILENAME)
    started = time.time()

    data = load_local_dataset(DATASET_NAME)
    # Keep the original record order: all train rows, then all validation rows.
    dataset = concatenate_datasets([data[split_name] for split_name in SPLITS])
    total = len(dataset)
    print(f"Loaded {total} rows from {DATASET_NAME} ({', '.join(SPLITS)})", flush=True)

    mapped = dataset.map(
        format_batch,
        batched=True,
        batch_size=1000,
        num_proc=workers if workers > 1 else None,
        remove_columns=dataset.column_names,
        desc="formatting",
    )

    write_jsonl(out_file, tqdm(mapped, total=total, desc="writing", unit="rows"))

    elapsed = time.time() - started
    print(f"[1/1] {OUTPUT_FILENAME}: {total} rows in {elapsed:.0f}s", flush=True)
    print(f"Done: {total} rows -> {out_file} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus',
        help='output directory; reclor.jsonl is written inside it')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of processes for datasets.map (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_reclor(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
