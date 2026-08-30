import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from utils import load_local_dataset


DATASET_ID = "nvidia/OpenMathInstruct-2"
ORIGINAL_SOURCES = {"math", "gsm8k"}
COT_CONDITION = "synth,cot"
DIRECT_CONDITION = "synth,direct"
BATCH_ROWS = 65536

# Original output was built via pa.Table.from_pydict on python strings
# -> all columns are plain utf8 `string` in this order.
SCHEMA = pa.schema([
    ("instruction", pa.string()),
    ("response", pa.string()),
    ("condition", pa.string()),
])


def _string_col(table, name):
    col = table.column(name)
    if col.type != pa.string():
        col = col.cast(pa.string())
    return col


def _condition_col(n, condition):
    return pa.chunked_array([pa.array([condition] * n, type=pa.string())])


def _process_shard(job):
    """Process one contiguous [start, end) slice of the dataset, writing
    cot/direct part files to tmp_dir.

    The dataset is read from the HF cache as a memory-mapped arrow table and
    consumed in zero-copy BATCH_ROWS slices, so memory stays bounded by the
    batch size regardless of dataset size (~14M rows / ~30GB).
    Returns (shard_idx, n_rows, n_cot, n_direct, seconds).
    """
    shard_idx, start, end, tmp_dir = job
    t0 = time.time()
    ds = load_local_dataset(DATASET_ID, split="train")  # cache hit: main downloaded it
    table = ds.data.table  # full memory-mapped table, zero-copy
    n_rows = end - start

    cot_path = os.path.join(tmp_dir, f"cot_{shard_idx:05d}.parquet")
    direct_path = os.path.join(tmp_dir, f"direct_{shard_idx:05d}.parquet")
    cot_writer = pq.ParquetWriter(cot_path, SCHEMA, compression="snappy")
    direct_writer = None
    n_direct = 0
    value_set = pa.array(sorted(ORIGINAL_SOURCES),
                         type=table.column("problem_source").type)

    try:
        for offset in range(start, end, BATCH_ROWS):
            batch = table.slice(offset, min(BATCH_ROWS, end - offset))
            # Every row goes to cot: problem -> instruction,
            # generated_solution -> response, condition "synth,cot".
            cot_writer.write_table(pa.table({
                "instruction": _string_col(batch, "problem"),
                "response": _string_col(batch, "generated_solution"),
                "condition": _condition_col(batch.num_rows, COT_CONDITION),
            }))
            # Only rows whose problem_source is not one of the original
            # (non-synthetic) sources also go to direct, with expected_answer.
            mask = pc.invert(pc.is_in(batch.column("problem_source"),
                                      value_set=value_set))
            dbatch = batch.filter(mask)
            if dbatch.num_rows:
                if direct_writer is None:
                    direct_writer = pq.ParquetWriter(direct_path, SCHEMA,
                                                     compression="snappy")
                n_direct += dbatch.num_rows
                direct_writer.write_table(pa.table({
                    "instruction": _string_col(dbatch, "problem"),
                    "response": _string_col(dbatch, "expected_answer"),
                    "condition": _condition_col(dbatch.num_rows, DIRECT_CONDITION),
                }))
    finally:
        cot_writer.close()
        if direct_writer is not None:
            direct_writer.close()
    return shard_idx, n_rows, n_rows, n_direct, time.time() - t0


def _merge_parts(tmp_dir, prefix, out_path, desc):
    """Stream part files (in shard order) into one final parquet file.

    Reads and writes in batches: constant memory, no full-file load.
    """
    parts = [os.path.join(tmp_dir, f) for f in sorted(os.listdir(tmp_dir))
             if f.startswith(prefix)]
    total = sum(os.path.getsize(p) for p in parts)
    with pq.ParquetWriter(out_path, SCHEMA, compression="snappy") as writer, \
            tqdm(total=total, desc=desc, unit="B", unit_scale=True,
                 unit_divisor=1024) as bar:
        for p in parts:
            pf = pq.ParquetFile(p)
            for record_batch in pf.iter_batches(batch_size=BATCH_ROWS):
                writer.write_batch(record_batch)
            bar.update(os.path.getsize(p))


def clean_openmathinstruct2(output_path: str, workers: int, tmp_dir: str):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    started = time.time()

    ds = load_local_dataset(DATASET_ID, split="train")  # populates the HF cache
    n_rows = ds.num_rows
    num_shards = max(1, min(workers, n_rows))
    print(f"Processing {DATASET_ID} train: {n_rows} rows in {num_shards} "
          f"contiguous shards ({workers} workers)", flush=True)

    # Contiguous shard bounds; merging parts in shard order preserves the
    # original dataset row order in both output files.
    bounds = [(i * n_rows // num_shards, (i + 1) * n_rows // num_shards)
              for i in range(num_shards)]
    jobs = [(i, start, end, tmp_dir) for i, (start, end) in enumerate(bounds)]

    total_cot = total_direct = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_shard, job) for job in jobs]
        with tqdm(total=n_rows, desc="shards", unit="row",
                  unit_scale=True) as bar:
            for i, future in enumerate(as_completed(futures), 1):
                shard_idx, shard_rows, n_cot, n_direct, secs = future.result()
                total_cot += n_cot
                total_direct += n_direct
                bar.update(shard_rows)
                bar.set_postfix(done=f"{i}/{num_shards}")
                tqdm.write(f"[{i}/{num_shards}] shard {shard_idx}: "
                           f"{n_cot} cot + {n_direct} direct rows in {secs:.0f}s")

    _merge_parts(tmp_dir, "cot_", os.path.join(output_path, "cot.parquet"),
                 "merge cot")
    _merge_parts(tmp_dir, "direct_", os.path.join(output_path, "direct.parquet"),
                 "merge direct")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.time() - started
    print(f"Done: {total_cot} cot + {total_direct} direct rows -> {output_path} "
          f"in {elapsed / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/openmathinstruct2',
        help='absolute path to data_clustered/openmathinstruct2')
    parser.add_argument(
        '--tmp_dir', type=str, default=None,
        help='directory for intermediate per-shard files '
             '(default: <output_path>/.tmp_parts)')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of shards processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()
    output_path = args.output_path
    tmp_dir = args.tmp_dir or os.path.join(output_path, ".tmp_parts")

    clean_openmathinstruct2(
        output_path=output_path,
        workers=args.workers,
        tmp_dir=tmp_dir,
    )


if __name__ == "__main__":
    main()
