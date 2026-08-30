import argparse
import os
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DATASET = "sapientinc/sudoku-extreme"
DATASET_FILE = "train.csv"
CONDITION = "direct"
BLOCK_SIZE = 64 << 20  # 64 MiB read blocks -> bounded RAM, handful of chunks


def _count_rows(csv_path: str) -> int:
    # Puzzle/answer fields never contain newlines, so rows = '\n' count - header.
    n = 0
    with open(csv_path, "rb") as f:
        while chunk := f.read(1 << 26):
            n += chunk.count(b"\n")
        f.seek(-1, os.SEEK_END)
        ends_with_nl = f.read(1) == b"\n"
    return n - 1 + (0 if ends_with_nl else 1)


def clean_sudoku(output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    started = time.time()
    pa.set_cpu_count(max(1, workers))
    pa.set_io_thread_count(max(1, workers))

    t0 = time.time()
    csv_path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset")
    size_bytes = os.path.getsize(csv_path)
    total_rows = _count_rows(csv_path)
    n_chunks = max(1, (size_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE)
    print(f"Input: {csv_path} ({size_bytes / 2**20:.1f} MiB, {total_rows:,} rows, "
          f"~{n_chunks} chunks) ready in {time.time() - t0:.0f}s", flush=True)

    with open(csv_path, newline="") as f:
        header = f.readline().rstrip("\n").split(",")
    q_col, a_col = header[1], header[2]  # positional, like the original csv.reader

    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE),
        convert_options=pacsv.ConvertOptions(
            column_types={q_col: pa.string(), a_col: pa.string()}),
    )

    out_file = os.path.join(output_path, "all.parquet")
    writer = None
    rows_done = 0
    chunk_i = 0
    try:
        with tqdm(total=total_rows, desc="rows", unit="row",
                  unit_scale=True) as bar:
            while True:
                t_chunk = time.time()
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                q = batch.column(q_col)
                a = batch.column(a_col)
                if not pc.all(pc.and_(
                        pc.equal(pc.utf8_length(q), 81),
                        pc.equal(pc.utf8_length(a), 81))).as_py():
                    raise AssertionError(
                        "question/answer with length != 81 encountered")
                instruction = pc.binary_join_element_wise(
                    "Solve the Sudoku\n\n",
                    pc.replace_substring(q, ".", "0"),
                    "",
                )
                table = pa.Table.from_arrays(
                    [instruction, a, pa.repeat(CONDITION, batch.num_rows)],
                    names=["instruction", "response", "condition"])
                if writer is None:
                    writer = pq.ParquetWriter(out_file, table.schema,
                                              compression="snappy")
                writer.write_table(table)
                chunk_i += 1
                rows_done += batch.num_rows
                bar.update(batch.num_rows)
                tqdm.write(f"[{chunk_i}/{n_chunks}] {DATASET_FILE}: "
                           f"{batch.num_rows:,} rows in "
                           f"{time.time() - t_chunk:.1f}s")
        if rows_done != total_rows:
            raise AssertionError(f"expected {total_rows} rows, wrote {rows_done}")
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - started
    print(f"Done: {rows_done:,} rows -> {out_file} in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/sudoku_extreme',
        help='absolute path to data_clustered/sudoku_extreme')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='threads for pyarrow CSV parsing/compute '
             '(default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_sudoku(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
