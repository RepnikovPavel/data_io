import argparse
import os
import time

import orjson
from tqdm import tqdm

from utils import load_local_dataset

DATASET_ID = "facebook/principia-collection"
CONDITION = "synth,direct"
BATCH_SIZE = 10_000


def _to_output(batch):
    n = len(batch["problem_statement"])
    return {
        "condition": [CONDITION] * n,
        "instruction": batch["problem_statement"],
        "response": batch["answer"],
    }


def clean_principia_collection(output_path: str, workers: int):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    dataset_dict = load_local_dataset(DATASET_ID)
    total_rows = sum(ds.num_rows for ds in dataset_dict.values())
    n_splits = len(dataset_dict)
    print(f"Processing {n_splits} splits, {total_rows} rows total "
          f"({workers} workers)", flush=True)

    n_rows = 0
    with open(output_path, "wb") as f, tqdm(
            total=total_rows, desc="rows", unit="row", unit_scale=True) as bar:
        for i, (split_name, dataset) in enumerate(dataset_dict.items(), 1):
            t0 = time.time()
            dataset = dataset.map(
                _to_output,
                batched=True,
                num_proc=workers,
                remove_columns=dataset.column_names,
                desc=f"map[{split_name}]",
            )
            split_rows = 0
            for batch in dataset.iter(batch_size=BATCH_SIZE):
                for cond, inst, resp in zip(batch["condition"],
                                            batch["instruction"],
                                            batch["response"]):
                    f.write(orjson.dumps({
                        "condition": cond,
                        "instruction": inst,
                        "response": resp,
                    }))
                    f.write(b"\n")
                split_rows += len(batch["condition"])
                bar.update(len(batch["condition"]))
            n_rows += split_rows
            tqdm.write(f"[{i}/{n_splits}] {split_name}: {split_rows} rows "
                       f"in {time.time() - t0:.0f}s")

    elapsed = time.time() - started
    print(f"Done: {n_rows} rows -> {output_path} in {elapsed / 60:.1f} min",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/principia_collection.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='num_proc for datasets.map (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_principia_collection(output_path=args.output_path,
                               workers=args.workers)


if __name__ == "__main__":
    main()
