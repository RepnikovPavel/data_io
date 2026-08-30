import argparse
import math
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from utils import load_local_dataset


DATASET_ID = 'open-thoughts/OpenThoughts2-1M'
REMOVE_SOURCES = {
    'dolphin', 'evolcodegolf', 'glaive', 'magicoder', 'sharegpt', 'codefeedback',  # Remove code
    'nvidia_math'  # Already included
}
THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
SCHEMA = pa.schema([
    ("instruction", pa.string()),
    ("response", pa.string()),
    ("condition", pa.string()),
])


def remove_think_tags(text):
    return THINK_PATTERN.sub('', text)


def _process_chunk(job):
    """Filter/transform one contiguous row range, write one part parquet.

    Streams the slice in batches through a ParquetWriter: constant memory.
    Returns (chunk_id, n_rows, n_kept, seconds).
    """
    chunk_id, start, end, tmp_dir = job
    t0 = time.time()
    dataset = load_local_dataset(DATASET_ID, split='train')
    part_path = os.path.join(tmp_dir, f"part_{chunk_id:05d}.parquet")

    writer = pq.ParquetWriter(part_path, SCHEMA)
    n_kept = 0
    try:
        for batch in dataset.select(range(start, end)).iter(batch_size=4096):
            instructions, responses = [], []
            for source, convs in zip(batch["source"], batch["conversations"]):
                if source in REMOVE_SOURCES:
                    continue
                # Check there are exactly two conversations: user and assistant
                assert len(convs) == 2
                assert convs[0]["from"] == "user"
                assert convs[1]["from"] == "assistant"

                input = convs[0]["value"]
                output = convs[1]["value"]

                # Filter out code based on simple heuristics
                output_lower = output.lower()
                if "python" not in input.lower() and "python" not in output_lower \
                        and "```" not in output_lower:
                    instructions.append(input)
                    responses.append(remove_think_tags(output))

            if instructions:
                table = pa.Table.from_pydict(
                    {
                        "instruction": instructions,
                        "response": responses,
                        "condition": ["synth,cot"] * len(instructions),
                    },
                    schema=SCHEMA,
                )
                writer.write_table(table)
                n_kept += len(instructions)
    finally:
        writer.close()
    return chunk_id, end - start, n_kept, time.time() - t0


def clean_openthoughts2(output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    tmp_dir = os.path.join(output_path, ".tmp_parts")
    os.makedirs(tmp_dir, exist_ok=True)
    started = time.time()

    dataset = load_local_dataset(DATASET_ID, split='train')  # populates the HF cache once
    n_rows = len(dataset)
    del dataset

    n_chunks = max(1, workers * 4)
    chunk_size = math.ceil(n_rows / n_chunks)
    jobs = [
        (i, start, min(start + chunk_size, n_rows), tmp_dir)
        for i, start in enumerate(range(0, n_rows, chunk_size))
    ]
    print(f"Processing {n_rows} rows in {len(jobs)} chunks "
          f"({workers} workers)", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_chunk, job) for job in jobs]
        with tqdm(total=n_rows, desc="rows", unit="row", unit_scale=True) as bar:
            for i, future in enumerate(as_completed(futures), 1):
                chunk_id, chunk_rows, n_kept, secs = future.result()
                bar.update(chunk_rows)
                bar.set_postfix(chunk=chunk_id, done=f"{i}/{len(jobs)}")
                tqdm.write(f"[{i}/{len(jobs)}] chunk {chunk_id}: "
                           f"{n_kept}/{chunk_rows} rows kept in {secs:.0f}s")

    # Merge part files in chunk order (= dataset order) into all.parquet,
    # streaming batches so memory stays bounded.
    out_file = os.path.join(output_path, "all.parquet")
    writer = pq.ParquetWriter(out_file, SCHEMA)
    total_kept = 0
    try:
        for chunk_id, _, _, _ in jobs:
            part_path = os.path.join(tmp_dir, f"part_{chunk_id:05d}.parquet")
            for batch in pq.ParquetFile(part_path).iter_batches():
                total_kept += batch.num_rows
                writer.write_table(pa.Table.from_batches([batch]))
    finally:
        writer.close()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.time() - started
    print(f"Done: {total_kept}/{n_rows} rows -> {out_file} "
          f"in {elapsed / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/openthoughts2',
        help='absolute path to data_clustered/openthoughts2')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of chunks processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_openthoughts2(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
