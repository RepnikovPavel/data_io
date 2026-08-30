import argparse
import os
import time

from tqdm import tqdm

from utils import load_local_dataset, write_jsonl

DATASET_NAME = "openai/gsm8k"
BATCH_SIZE = 1000


def _transform_batch(batch):
    instructions, responses = [], []
    for question, answer in zip(batch["question"], batch["answer"]):
        parts = answer.split("#### ")
        assert len(parts) == 2
        instructions.append(question.strip())
        responses.append(parts[-1].strip())
    return {"instruction": instructions, "response": responses}


def clean_gsm8k_train(output_path: str, workers: int):
    started = time.time()
    dataset = load_local_dataset(DATASET_NAME, "main", split="train")
    total = len(dataset)

    # datasets.map preserves the original record order, also with num_proc > 1
    mapped = dataset.map(
        _transform_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=workers,
        remove_columns=dataset.column_names,
        desc="transform",
    )

    result = []
    n_chunks = (total + BATCH_SIZE - 1) // BATCH_SIZE
    with tqdm(total=total, desc="gsm8k_train", unit="row") as bar:
        for i, batch in enumerate(mapped.iter(batch_size=BATCH_SIZE), 1):
            t0 = time.time()
            n_rows = len(batch["instruction"])
            result.extend(
                {
                    "condition": "direct",
                    "instruction": instruction,
                    "response": response,
                }
                for instruction, response in zip(batch["instruction"], batch["response"])
            )
            bar.update(n_rows)
            tqdm.write(f"[{i}/{n_chunks}] gsm8k_train: {n_rows} rows "
                       f"in {time.time() - t0:.1f}s")

    print(f"Total records loaded: {len(result)}")
    write_jsonl(output_path, result)
    print(f"Done: {len(result)} rows -> {output_path} in {time.time() - started:.1f}s",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/gsm8k_train.jsonl',
        help='absolute path to gsm8k_train.jsonl')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of processes for datasets.map (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_gsm8k_train(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
