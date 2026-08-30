import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import orjson
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from utils import write_jsonl

DESCRIPTION_MAP = {
    "math.json": ("Solve the math problem.", "cot"),
    "reading.json": ("Solve the reading comprehension problem.", "cot"),
    "law.json": ("Choose the correct option letter.", "direct"),
    "science.json": ("Solve the science problem.", "cot"),
    "physics.json": ("Solve the physics problem.", "cot")
}


def _process_file(job):
    """Transform one ARB json file into one jsonl output file.

    Returns (filename, n_rows, n_bytes_in, seconds).
    """
    filename, description, condition, input_dir, output_path = job
    t0 = time.time()
    in_file = os.path.join(input_dir, filename)
    with open(in_file, "rb") as f:
        data = orjson.loads(f.read())

    records = (
        {
            "instruction": f"{description}\n\n{x['instruction']}",
            "response": x["response"],
            "condition": condition,
        } for x in data
    )

    # Preserve the original naming quirk: arb_<filename>l, e.g. arb_math.jsonl
    write_jsonl(os.path.join(output_path, f"arb_{filename}l"), records)
    n_bytes = os.path.getsize(in_file)
    return filename, len(data), n_bytes, time.time() - t0


def clean_arb(input_dir: str, output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    started = time.time()

    jobs = []
    for filename, (description, condition) in DESCRIPTION_MAP.items():
        if not os.path.exists(os.path.join(input_dir, filename)):
            tqdm.write(f"[warn] missing input file {filename}")
            continue
        jobs.append((filename, description, condition, input_dir, output_path))

    total_bytes = sum(os.path.getsize(os.path.join(input_dir, job[0])) for job in jobs)
    print(f"Processing {len(jobs)} files, {total_bytes / 2**20:.1f} MiB total "
          f"({workers} workers)", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_file, job) for job in jobs]
        with tqdm(total=total_bytes, desc="files", unit="B", unit_scale=True,
                  unit_divisor=1024) as bar:
            for i, future in enumerate(as_completed(futures), 1):
                filename, n_rows, n_bytes, secs = future.result()
                bar.update(n_bytes)
                bar.set_postfix(file=filename, done=f"{i}/{len(jobs)}")
                tqdm.write(f"[{i}/{len(jobs)}] {filename}: {n_rows} rows, "
                           f"{n_bytes / 2**20:.2f} MiB in {secs:.1f}s")

    elapsed = time.time() - started
    print(f"Done: {len(jobs)} files -> {output_path} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str,
        default='/mnt/hdd2/datasets_text/Platypus/ARB',
        help='absolute path to raw_data/Platypus/ARB')
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus',
        help='absolute path to data/Platypus')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of files processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    clean_arb(
        input_dir=args.input_dir,
        output_path=args.output_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
