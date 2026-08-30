import argparse
import os
import struct
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import tarfile
from tqdm import tqdm

# Files per worker task; small enough for fine-grained progress,
# large enough to amortize IPC cost.
BATCH_FILES = 256
# Flush a group's buffered rows to its parquet writer at this size,
# so memory stays bounded regardless of archive size.
FLUSH_ROWS = 2000
# Cap on concurrently queued batches (backpressure on archive reading).
MAX_PENDING_FACTOR = 3


def _parse_batch(items):
    """Parse a batch of (task_key, condition, raw_bytes) file payloads.

    Returns (records, n_skipped) where records is a list of
    (task_key, condition, instruction, response).
    """
    records = []
    n_skipped = 0
    for task_key, condition, raw in items:
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            n_skipped += 1
            continue

        # 1. Check if starts with "Problem:" and remove it
        content = content.removeprefix("Problem:")

        # 2. Split by the first "Answer:"
        # maxsplit=1 ensures we only split on the first occurrence
        parts = content.split("Answer:", 1)

        if len(parts) < 2:
            # "Answer:" tag missing
            n_skipped += 1
            continue

        instruction_text = parts[0].strip()
        response_text = parts[1].strip()

        if not instruction_text or not response_text:
            n_skipped += 1
            continue

        records.append((task_key, condition, instruction_text, response_text))
    return records, n_skipped


def _uncompressed_size(path):
    """ISIZE footer of a gzip file: uncompressed size mod 2**32."""
    try:
        with open(path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            return struct.unpack("<I", f.read(4))[0]
    except OSError:
        return None


class GroupedParquetWriter:
    """One ParquetWriter per topic_subtask group, fed with bounded buffers."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.buffers = defaultdict(list)
        self.writers = {}
        self.rows_per_group = defaultdict(int)

    def add(self, records):
        for task_key, condition, instruction, response in records:
            buf = self.buffers[task_key]
            buf.append((condition, instruction, response))
            if len(buf) >= FLUSH_ROWS:
                self._flush(task_key)

    def _flush(self, task_key):
        buf = self.buffers.pop(task_key, [])
        if not buf:
            return
        table = pa.Table.from_pydict({
            "condition": [r[0] for r in buf],
            "instruction": [r[1] for r in buf],
            "response": [r[2] for r in buf],
        })
        writer = self.writers.get(task_key)
        if writer is None:
            file_path = os.path.join(self.output_dir, f"{task_key}.parquet")
            writer = pq.ParquetWriter(file_path, table.schema)
            self.writers[task_key] = writer
        writer.write_table(table)
        self.rows_per_group[task_key] += len(buf)

    def close(self):
        for task_key in list(self.buffers):
            self._flush(task_key)
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


def process_amps_archive(input_path, output_dir, workers):
    started = time.time()
    skipped_count = 0
    total_records = 0

    os.makedirs(output_dir, exist_ok=True)
    total_bytes = _uncompressed_size(input_path)
    print(f"Processing {input_path} ({workers} workers)...", flush=True)

    sink = GroupedParquetWriter(output_dir)
    pending = set()
    batch = []

    def drain(futures):
        nonlocal skipped_count, total_records
        for future in futures:
            records, n_skipped = future.result()
            skipped_count += n_skipped
            if records:
                sink.add(records)
                total_records += len(records)

    with tarfile.open(input_path, "r:gz") as tar, tqdm(
            total=total_bytes, desc="Reading archive",
            unit="B", unit_scale=True, unit_divisor=1024) as bar, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        prev_offset = 0
        for member in tar:
            # tqdm tracks uncompressed bytes consumed
            if member.offset_data > prev_offset:
                bar.update(member.offset_data - prev_offset)
                prev_offset = member.offset_data

            if not member.isfile() or not member.name.endswith('.txt'):
                continue

            # Path parsing: amps/mathematica/<topic>/<task>/filename.txt
            path_parts = member.name.strip("/").split("/")

            # Ensure we have enough depth for negative indexing
            if len(path_parts) < 3:
                continue

            # Validation: Ensure it's part of the mathematica dataset
            if len(path_parts) >= 4 and path_parts[-4] != "mathematica":
                continue

            topic = path_parts[-3]
            subtask = path_parts[-2]

            # Create the specific key for grouping and filename
            task_key = f"{topic}_{subtask}"

            # Determine condition based on the subtask folder name
            condition = "noisy,cot" if subtask.endswith("w_steps") else "noisy,direct"

            f = tar.extractfile(member)
            if f is None:
                skipped_count += 1
                continue

            batch.append((task_key, condition, f.read()))
            if len(batch) >= BATCH_FILES:
                pending.add(pool.submit(_parse_batch, batch))
                batch = []
                if len(pending) >= workers * MAX_PENDING_FACTOR:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    drain(done)

        if batch:
            pending.add(pool.submit(_parse_batch, batch))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            drain(done)

    print(f"Parsing complete. Found {total_records} records.")
    print(f"Writing parquet files to '{output_dir}'...")
    sink.close()

    groups = sorted(sink.rows_per_group.items())
    for i, (task_name, n_rows) in enumerate(groups, 1):
        print(f"[{i}/{len(groups)}] {task_name}: {n_rows} rows", flush=True)

    elapsed = time.time() - started
    print(f"Done: {total_records} records -> {len(groups)} parquet files in "
          f"'{output_dir}' in {elapsed / 60:.1f} min. "
          f"Skipped files: {skipped_count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str,
        default='/mnt/hdd2/datasets_text/amps.tar.gz',
        help='absolute path to the amps.tar.gz archive')
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/ampsmathematica',
        help='absolute path to data_clustered/ampsmathematica')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of parallel parse workers (default: min(8, cpu_count))')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    process_amps_archive(args.input_dir, args.output_path, args.workers)


if __name__ == "__main__":
    main()
