import argparse
import os
import time

import orjson
from tqdm import tqdm

from utils import load_local_dataset

DATASET_ID = "AI-MO/NuminaMath-1.5"

MAP_BATCH_SIZE = 1000
WRITE_BATCH_SIZE = 10_000


def _transform_batch(batch):
    """Emit one 'noisy,cot' record per valid row, plus a 'noisy,direct'
    record when the answer qualifies. Order within and across batches
    matches the original row-wise loop."""
    conditions, instructions, responses = [], [], []
    for synthetic, p_valid, s_valid, problem, solution, answer, qtype in zip(
            batch["synthetic"], batch["problem_is_valid"],
            batch["solution_is_valid"], batch["problem"], batch["solution"],
            batch["answer"], batch["question_type"]):
        if synthetic or p_valid != "Yes" or s_valid != "Yes" \
                or solution is None or answer is None:
            continue
        if "http" in problem or "http" in solution \
                or "Translate the text above into English" in solution:
            continue
        problem = problem.strip()
        solution = solution.strip()
        if not problem or not solution:
            continue
        conditions.append("noisy,cot")
        instructions.append(problem)
        responses.append(solution)
        if qtype != "proof" and answer != "proof":
            answer = answer.strip()
            if answer:
                conditions.append("noisy,direct")
                instructions.append(problem)
                responses.append(answer)
    return {"condition": conditions, "instruction": instructions,
            "response": responses}


def clean_numinamath(output_path: str, workers: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    started = time.time()

    dataset = load_local_dataset(DATASET_ID, split="train")
    n_rows = len(dataset)
    print(f"Loaded {DATASET_ID} train: {n_rows} rows "
          f"({workers} map workers)", flush=True)

    # Batched parallel transform; num_proc preserves row order (shards are
    # reassembled in order). Streams through the arrow cache on disk, so RAM
    # stays bounded regardless of dataset size.
    mapped = dataset.map(
        _transform_batch,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        num_proc=workers,
        remove_columns=dataset.column_names,
        desc="transform",
    )

    # Incremental write: stream the mapped dataset in chunks, append orjson
    # lines as we go (same format as utils.write_jsonl).
    n_out = len(mapped)
    n_chunks = (n_out + WRITE_BATCH_SIZE - 1) // WRITE_BATCH_SIZE
    written = 0
    with open(output_path, "wb") as f, \
            tqdm(total=n_out, desc="write", unit="row",
                 unit_scale=True) as bar:
        for i, batch in enumerate(mapped.iter(batch_size=WRITE_BATCH_SIZE), 1):
            t0 = time.time()
            chunk = b"".join(
                orjson.dumps({"condition": c, "instruction": inst,
                              "response": r}) + b"\n"
                for c, inst, r in zip(batch["condition"], batch["instruction"],
                                      batch["response"]))
            f.write(chunk)
            written += len(batch["condition"])
            bar.update(len(batch["condition"]))
            tqdm.write(f"[{i}/{n_chunks}] write: {len(batch['condition'])} rows "
                       f"in {time.time() - t0:.1f}s")

    elapsed = time.time() - started
    print(f"Done: {written} rows -> {output_path} in {elapsed / 60:.1f} min",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/numinamath.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='num_proc for the batched map (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_numinamath(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
