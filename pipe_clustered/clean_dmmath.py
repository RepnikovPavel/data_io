import argparse
import itertools
import os
import string
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SUBSETS = ["train-easy", "train-medium", "train-hard"]
BATCH_SIZE = 65536
_SENTINEL = object()


def safe_filename(filename):
    # Remove or replace unsafe characters
    safe_chars = set(string.ascii_letters + string.digits + "_-. ")
    return "".join(c if c in safe_chars else "_" for c in filename)


def _process_file(job):
    """Convert one .txt file (alternating question/answer lines) to one parquet.

    Streams the file in batches and appends row groups via ParquetWriter, so
    memory stays bounded regardless of file size.
    Returns (task_name, n_rows, n_bytes_in, seconds).
    """
    set_name, filename, output_path = job
    t0 = time.time()
    task_name = safe_filename(
        f"{set_name}__{os.path.basename(filename).removesuffix('.txt')}")
    out_file = os.path.join(output_path, f"{task_name}.parquet")

    n_rows = 0
    with open(filename, "r") as f:
        stripped = (line.strip() for line in f)

        def pairs():
            # consecutive lines -> (instruction, response); odd trailing line
            # must fail, matching the original's `assert len(lines) % 2 == 0`
            while True:
                x = next(stripped, _SENTINEL)
                if x is _SENTINEL:
                    return
                y = next(stripped, _SENTINEL)
                if y is _SENTINEL:
                    raise AssertionError(f"odd number of lines in {filename}")
                yield (x, y)

        with pq.ParquetWriter(out_file, schema=pa.schema([
                ("instruction", pa.string()),
                ("response", pa.string()),
                ("condition", pa.string())])) as writer:
            while True:
                batch = list(itertools.islice(pairs(), BATCH_SIZE))
                if not batch:
                    break
                n = len(batch)
                xs, ys = zip(*batch)
                writer.write_table(pa.table({
                    "instruction": list(xs),
                    "response": list(ys),
                    "condition": ["direct"] * n,
                }))
                n_rows += n

    if n_rows == 0:
        # Match the original output for empty files (columns inferred as null).
        pq.write_table(
            pa.Table.from_pydict(
                {"instruction": [], "response": [], "condition": []}),
            out_file)

    return task_name, n_rows, os.path.getsize(filename), time.time() - t0


def clean_dmmath(input_dir: str, output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    started = time.time()

    jobs = []
    for set_name in SUBSETS:
        for filename in sorted(glob(os.path.join(input_dir, set_name, "*.txt"))):
            jobs.append((set_name, filename, output_path))
    if not jobs:
        print(f"[warn] no .txt files found under {input_dir}", flush=True)
        return

    total_bytes = sum(os.path.getsize(f) for _, f, _ in jobs)
    print(f"Processing {len(jobs)} files, {total_bytes / 2**30:.1f} GiB total "
          f"({workers} workers)", flush=True)

    total_rows = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_file, job) for job in jobs]
        with tqdm(total=total_bytes, desc="files", unit="B", unit_scale=True,
                  unit_divisor=1024) as bar:
            for i, future in enumerate(as_completed(futures), 1):
                task_name, n_rows, n_bytes, secs = future.result()
                total_rows += n_rows
                bar.update(n_bytes)
                bar.set_postfix(file=task_name, done=f"{i}/{len(jobs)}")
                tqdm.write(f"[{i}/{len(jobs)}] {task_name}: "
                           f"{n_rows} rows in {secs:.0f}s")

    elapsed = time.time() - started
    print(f"Done: {len(jobs)} files, {total_rows} rows -> {output_path} "
          f"in {elapsed / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str,
        default='/mnt/hdd2/datasets_text/mathematics_dataset-v1.0',
        help='path to mathematics_dataset-v1.0 '
             '(with train-easy/train-medium/train-hard subdirs)')
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/dmmath',
        help='absolute path to data_clustered/dmmath')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of files processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    clean_dmmath(
        input_dir=args.input_dir,
        output_path=args.output_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
