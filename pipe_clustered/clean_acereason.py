import argparse
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from utils import load_local_dataset

DATASET_ID = 'nvidia/AceReason-1.1-SFT'
BATCH_SIZE = 20_000

# re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL) in RE2 syntax
THINK_PATTERN = r'(?s)<think>.*?</think>'


def _process_batch(batch):
    """Filter to math rows, strip think tags. Vectorized via pyarrow compute.

    Returns a pydict with columns instruction/response/condition
    (same order and values as the original row-wise script).
    """
    table = pa.table(batch)
    math_rows = table.filter(pc.equal(table["category"], "math"))

    outputs = math_rows["output"]
    # Original asserts row["output"].count("<think>") == 1 for every math row
    # (and crashes on a null output, since None has no .count).
    if pc.any(pc.is_null(outputs)).as_py():
        raise AssertionError("null output in math row")
    n_think = pc.count_substring(outputs, "<think>")
    # min_count=0: an empty selection (no math rows in this batch) must pass,
    # not raise — pc.all on an empty array is otherwise null.
    if pc.all(pc.equal(n_think, 1), min_count=0).as_py() is not True:
        raise AssertionError("expected exactly one <think> tag per math output")
    cleaned = pc.replace_substring_regex(outputs, THINK_PATTERN, "")

    n = math_rows.num_rows
    return {
        "instruction": math_rows["input"].to_pylist(),
        "response": cleaned.to_pylist(),
        "condition": ["synth,cot"] * n,
    }, math_rows.num_rows


def clean_acereason(output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    started = time.time()

    dataset = load_local_dataset(DATASET_ID, split='train')
    n_rows = len(dataset)
    n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {n_rows} rows from {DATASET_ID} "
          f"({workers} workers, batches of {BATCH_SIZE})", flush=True)

    out_file = os.path.join(output_path, "all.parquet")
    kept = 0
    writer = None
    pending = deque()  # (future, batch_input_rows, t0) in submission order
    batch_iter = iter(dataset.iter(batch_size=BATCH_SIZE))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool, \
                tqdm(total=n_rows, desc="rows", unit="row",
                     unit_scale=True) as bar:
            done_batches = 0
            # Keep a bounded window of in-flight batches (RAM stays bounded);
            # results are consumed in submission order, so row order is kept.
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < max(1, 2 * workers):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(
                        (pool.submit(_process_batch, batch),
                         len(batch["input"]), time.time()))

                future, in_rows, t0 = pending.popleft()
                result, n_kept = future.result()
                table = pa.Table.from_pydict(result)
                if writer is None:
                    writer = pq.ParquetWriter(out_file, table.schema)
                if table.num_rows:
                    writer.write_table(table)
                kept += n_kept
                done_batches += 1
                bar.update(in_rows)
                bar.set_postfix(kept=kept)
                tqdm.write(f"[{done_batches}/{n_batches}] batch: {n_kept} kept / "
                           f"{in_rows} rows in {time.time() - t0:.1f}s")
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - started
    print(f"Done: {kept} rows -> {out_file} in {elapsed / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/acereason',
        help='absolute path to data_clustered/acereason')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of batches processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_acereason(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
