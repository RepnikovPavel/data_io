import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import orjson
from tqdm import tqdm


def _read_files(filenames):
    """Read a batch of per-problem JSON files and build output records."""
    records = []
    for filename in filenames:
        with open(filename, "rb") as f:
            item = orjson.loads(f.read())
        records.append({
            "condition": "noisy,cot",
            "instruction": item["problem"],
            "response": "\n".join(item["hints"]).strip()
        })
    return records


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def clean_amps_khan(input_dir: str, output_path: str, workers: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    started = time.time()

    filenames = sorted(glob(os.path.join(input_dir, "**", "*.json"), recursive=True))
    n_files = len(filenames)
    if n_files == 0:
        tqdm.write(f"[warn] no json files under {input_dir}")
        open(output_path, "wb").close()
        print(f"Done: 0 files -> {output_path} in 0s", flush=True)
        return

    chunk_size = max(1, min(2000, (n_files + workers * 20 - 1) // (workers * 20)))
    batches = list(_chunks(filenames, chunk_size))
    print(f"Processing {n_files} files in {len(batches)} batches "
          f"({workers} workers)", flush=True)

    n_rows = 0
    with open(output_path, "wb") as out, ProcessPoolExecutor(
            max_workers=workers) as pool:
        # pool.map preserves input order, so the jsonl record order matches
        # the sorted file order while results stream in batch by batch.
        with tqdm(total=n_files, desc="files", unit="file") as bar:
            t_batch = time.time()
            for i, records in enumerate(pool.map(_read_files, batches), 1):
                for record in records:
                    out.write(orjson.dumps(record))
                    out.write(b"\n")
                n_rows += len(records)
                bar.update(len(records))
                now = time.time()
                tqdm.write(f"[{i}/{len(batches)}] batch: {len(records)} rows "
                           f"in {now - t_batch:.1f}s")
                t_batch = now

    elapsed = time.time() - started
    print(f"Done: {n_files} files -> {n_rows} rows -> {output_path} "
          f"in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str,
        default='/mnt/hdd2/datasets_text/amps/khan',
        help='absolute path to raw_data/amps/khan')
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/amps_khan.jsonl',
        help='absolute path to data/amps_khan.jsonl')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of parallel reader processes (default: min(8, cpu_count))')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    clean_amps_khan(
        input_dir=args.input_dir,
        output_path=args.output_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
