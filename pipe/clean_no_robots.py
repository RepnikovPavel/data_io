import argparse
import os
import time

from tqdm import tqdm

from utils import load_local_dataset, write_jsonl

DATASET_ID = "HuggingFaceH4/no_robots"


def _transform_batch(batch):
    """Extract the first user -> assistant turn (with optional system prompt)."""
    conditions, instructions, responses = [], [], []
    for msgs in batch["messages"]:
        if len(msgs) < 2:
            continue

        # Check for system prompt at index 0
        system_content = ""
        start_idx = 0

        if msgs[0]["role"] == "system":
            system_content = msgs[0]["content"] + "\n\n"
            start_idx = 1

        # Ensure valid User -> Assistant structure for the first turn
        if (len(msgs) > start_idx + 1 and
                msgs[start_idx]["role"] == "user" and
                msgs[start_idx + 1]["role"] == "assistant"):
            conditions.append("cot")
            instructions.append(system_content + msgs[start_idx]["content"])
            responses.append(msgs[start_idx + 1]["content"])
    return {"condition": conditions, "instruction": instructions,
            "response": responses}


def clean_no_robots(output_path: str, workers: int):
    started = time.time()
    dataset = load_local_dataset(DATASET_ID)
    splits = list(dataset.items())  # keep the original split order

    result = []
    with tqdm(total=sum(len(s) for _, s in splits), desc="rows",
              unit="row") as bar:
        for i, (split_name, split) in enumerate(splits, 1):
            t0 = time.time()
            mapped = split.map(
                _transform_batch,
                batched=True,
                num_proc=workers,
                remove_columns=split.column_names,
            )
            result.extend(mapped)
            bar.update(len(split))
            tqdm.write(f"[{i}/{len(splits)}] {split_name}: {len(mapped)} records "
                       f"from {len(split)} rows in {time.time() - t0:.0f}s")

    write_jsonl(output_path, result)
    elapsed = time.time() - started
    print(f"Done: {len(result)} records -> {output_path} in {elapsed:.0f}s",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/no_robots.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='num_proc for datasets.map (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_no_robots(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
