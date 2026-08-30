import argparse
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from utils import load_local_dataset

DATASET_ID = "MegaScience/TextbookReasoning"

# Same column names/order as the original script's pydict.
OUTPUT_SCHEMA = pa.schema([
    ("instruction", pa.string()),
    ("response", pa.string()),
    ("condition", pa.string()),
])

BATCH_SIZE = 4096


def _process_batch(batch):
    """Split one batch into the two output pydicts.

    Same rule as the original script: every row goes to cot (question/answer,
    "synth,cot"); rows whose lowercased question contains neither "prove" nor
    "show that" also go to direct (question/reference_answer, "noisy,direct").
    """
    questions = batch["question"]
    n = len(questions)
    cot = {
        "instruction": questions,
        "response": batch["answer"],
        "condition": ["synth,cot"] * n,
    }
    direct = {"instruction": [], "response": [], "condition": []}
    for question, reference_answer in zip(questions, batch["reference_answer"]):
        # Skip all "proofs". Rough rule-based filter.
        lower = question.lower()
        if "prove" not in lower and "show that" not in lower:
            direct["instruction"].append(question)
            direct["response"].append(reference_answer)
            direct["condition"].append("noisy,direct")
    return cot, direct


def clean_textbookreasoning(output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    started = time.time()

    ds = load_local_dataset(DATASET_ID, split="train")
    n_rows = len(ds)
    n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {n_rows} rows from {DATASET_ID} "
          f"({workers} workers, batches of {BATCH_SIZE})", flush=True)

    # Single pass over the (disk-cached) dataset, writing both files at once.
    # Constant memory: a bounded window of in-flight batches; results are
    # consumed in submission order, so row order matches the original.
    cot_file = os.path.join(output_path, "cot.parquet")
    direct_file = os.path.join(output_path, "direct.parquet")
    n_cot = n_direct = 0
    cot_writer = pq.ParquetWriter(cot_file, OUTPUT_SCHEMA)
    direct_writer = pq.ParquetWriter(direct_file, OUTPUT_SCHEMA)
    pending = deque()  # (future, batch_rows, t0) in submission order
    batch_iter = iter(ds.iter(batch_size=BATCH_SIZE))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool, \
                tqdm(total=n_rows, desc="rows", unit="row",
                     unit_scale=True) as bar:
            done_batches = 0
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
                         len(batch["question"]), time.time()))

                future, in_rows, t0 = pending.popleft()
                cot, direct = future.result()
                cot_writer.write_table(
                    pa.Table.from_pydict(cot, schema=OUTPUT_SCHEMA))
                n_cot += len(cot["instruction"])
                n_direct_batch = len(direct["instruction"])
                if n_direct_batch:
                    direct_writer.write_table(
                        pa.Table.from_pydict(direct, schema=OUTPUT_SCHEMA))
                    n_direct += n_direct_batch
                done_batches += 1
                bar.update(in_rows)
                bar.set_postfix(cot=n_cot, direct=n_direct)
                tqdm.write(f"[{done_batches}/{n_batches}] batch: "
                           f"{in_rows} cot / {n_direct_batch} direct rows in "
                           f"{time.time() - t0:.1f}s")
    finally:
        cot_writer.close()
        direct_writer.close()

    elapsed = time.time() - started
    print(f"Done: cot={n_cot} rows -> {cot_file}; direct={n_direct} rows -> "
          f"{direct_file} in {elapsed / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/textbookreasoning',
        help='absolute path to data_clustered/textbookreasoning')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of batches processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_textbookreasoning(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
