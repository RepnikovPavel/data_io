import argparse
import os
import time

from tqdm import tqdm

from utils import load_local_dataset, write_jsonl

DATASET_ID = "KbsdJames/Omni-MATH"
CHUNK_SIZE = 1000


def _iter_records(rows):
    """Yield the two output records per input row (cot first, then direct)."""
    for problem, solution, answer in zip(
            rows["problem"], rows["solution"], rows["answer"]):
        yield {
            "condition": "cot",
            "instruction": problem,
            "response": solution.strip(),
        }
        yield {
            "condition": "direct",
            "instruction": problem,
            "response": answer.strip(),
        }


def clean_omnimath(output_path: str):
    dataset = load_local_dataset(DATASET_ID, split="test")
    n_rows = len(dataset)
    n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    started = time.time()

    def _gen():
        n_written = 0
        with tqdm(total=n_rows, desc="omnimath", unit="row") as bar:
            for i in range(n_chunks):
                t0 = time.time()
                rows = dataset[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
                n_chunk_rows = len(rows["problem"])
                for record in _iter_records(rows):
                    n_written += 1
                    yield record
                bar.update(n_chunk_rows)
                tqdm.write(f"[{i + 1}/{n_chunks}] omnimath: {n_chunk_rows} rows "
                           f"({2 * n_chunk_rows} records) in {time.time() - t0:.1f}s")
        tqdm.write(f"[summary] wrote {n_written} records from {n_rows} rows")

    # write_jsonl streams line by line, so records are never held in RAM at once.
    write_jsonl(output_path, _gen())

    elapsed = time.time() - started
    print(f"Done: {n_rows} rows -> {output_path} "
          f"({2 * n_rows} records) in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/omnimath.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--input_dir', type=str, default=None,
        help='unused: the dataset is loaded from the Hugging Face hub '
             f'({DATASET_ID}); kept for interface compatibility')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='unused: the dataset is small enough to process sequentially '
             '(default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_omnimath(output_path=args.output_path)


if __name__ == "__main__":
    main()
