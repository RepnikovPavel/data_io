import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import orjson
import polars as pl
from tqdm import tqdm

from utils import load_local_dataset

DATASET_NAME = "facebook/natural_reasoning"
CONDITION = "noisy,direct"
BATCH_SIZE = 10_000


def _init_worker(polars_threads):
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)


def _process_shard(job):
    """Filter one contiguous shard of the dataset, write a temp jsonl part.

    Streams the shard in batches (bounded RAM), filters with polars
    expressions (no row-wise Python), appends orjson lines per batch.
    Returns (shard_idx, n_in, n_out, part_path, seconds).
    """
    shard_idx, start, end, part_path = job
    t0 = time.time()
    ds = load_local_dataset(DATASET_NAME, split="train")
    shard = ds.select(range(start, end))

    n_in = 0
    n_out = 0
    with open(part_path, "wb") as f:
        for batch in shard.iter(batch_size=BATCH_SIZE):
            n_in += len(batch["question"])
            df = (
                pl.DataFrame(
                    {"question": batch["question"],
                     "reference_answer": batch["reference_answer"]})
                .with_columns(
                    pl.col("reference_answer").str.strip_chars().alias("response"))
                .filter(
                    (pl.col("response").str.len_chars() > 0)
                    # Skip all "proofs". Rough rule-based filter.
                    & ~pl.col("question").str.to_lowercase()
                        .str.contains("prove", literal=True)
                    & ~pl.col("question").str.to_lowercase()
                        .str.contains("show that", literal=True)
                )
                .select(
                    pl.lit(CONDITION).alias("condition"),
                    pl.col("question").alias("instruction"),
                    pl.col("response"),
                )
            )
            for line in df.iter_rows():
                f.write(orjson.dumps(
                    dict(zip(("condition", "instruction", "response"), line))))
                f.write(b"\n")
            n_out += df.height
    return shard_idx, n_in, n_out, part_path, time.time() - t0


def clean_natural_reasoning(output_path: str, workers: int):
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, ".tmp_natural_reasoning_parts")
    os.makedirs(tmp_dir, exist_ok=True)
    started = time.time()

    # Download/read the dataset once in the main process so workers only
    # hit the warm HF cache (no concurrent-download races).
    ds = load_local_dataset(DATASET_NAME, split="train")
    total = len(ds)
    del ds

    n_shards = min(workers, max(1, total // BATCH_SIZE))
    bounds = [total * i // n_shards for i in range(n_shards + 1)]
    jobs = [
        (i, bounds[i], bounds[i + 1], os.path.join(tmp_dir, f"part_{i:05d}.jsonl"))
        for i in range(n_shards)
    ]
    polars_threads = max(1, (os.cpu_count() or 1) // workers)
    print(f"Processing {total} rows in {n_shards} shards "
          f"({workers} workers, {polars_threads} polars threads/worker)", flush=True)

    parts = {}
    done_in = 0
    done_out = 0
    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(polars_threads,)) as pool:
        futures = [pool.submit(_process_shard, job) for job in jobs]
        with tqdm(total=total, desc="rows", unit="row", unit_scale=True) as bar:
            for i, future in enumerate(as_completed(futures), 1):
                shard_idx, n_in, n_out, part_path, secs = future.result()
                parts[shard_idx] = part_path
                done_in += n_in
                done_out += n_out
                bar.update(n_in)
                tqdm.write(f"[{i}/{len(jobs)}] shard {shard_idx}: "
                           f"{n_out}/{n_in} rows kept in {secs:.0f}s")

    # Concatenate shard parts in original dataset order, streaming copy.
    with open(output_path, "wb") as out:
        for i in range(len(jobs)):
            with open(parts[i], "rb") as f:
                shutil.copyfileobj(f, out, length=1 << 20)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - started
    print(f"Done: {done_out}/{done_in} rows kept -> {output_path} "
          f"in {elapsed / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/natural_reasoning.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of shards processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_natural_reasoning(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
