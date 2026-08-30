import argparse
import os
import time

from tqdm import tqdm

from utils import load_local_dataset, write_jsonl

SPLITS = ["train", "validation"]


def _format_batch(batch):
    conditions, instructions, responses = [], [], []
    for question_stem, fact1, choices, answer_key in zip(
            batch["question_stem"], batch["fact1"],
            batch["choices"], batch["answerKey"]):
        # Convert the choices into multiple choice format
        choice_texts = choices["text"]
        formatted_choices = "\n".join(
            f"\n{chr(65 + i)}: {choice}" if i == 0 else f"{chr(65 + i)}: {choice}"
            for i, choice in enumerate(choice_texts))
        instruction = (
            "Based on the given fact, which of the following option is the "
            f"correct answer to the question?\n\n{question_stem} "
            f"{formatted_choices}\n\nFact: {fact1}")
        conditions.append("direct")
        instructions.append(instruction)
        responses.append(answer_key)
    return {"condition": conditions, "instruction": instructions,
            "response": responses}


def clean_openbookqa(output_path: str, workers: int):
    started = time.time()
    dataset = load_local_dataset("allenai/openbookqa", name="additional")

    # Small dataset (~5.5k rows): collect formatted records in memory,
    # preserving the original order (train first, then validation).
    records = []
    for i, split in enumerate(SPLITS, 1):
        t0 = time.time()
        ds = dataset[split]
        mapped = ds.map(
            _format_batch,
            batched=True,
            num_proc=min(workers, len(ds)),
            remove_columns=ds.column_names,
            desc=f"format {split}",
        )
        rows = mapped.to_list()
        records.extend(rows)
        tqdm.write(f"[{i}/{len(SPLITS)}] {split}: {len(rows)} rows in "
                   f"{time.time() - t0:.1f}s")

    write_jsonl(output_path, records)
    elapsed = time.time() - started
    print(f"Done: {len(records)} rows -> {output_path} in {elapsed:.1f}s",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus/openbookqa.jsonl',
        help='absolute path to the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='num_proc for datasets.map (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_openbookqa(output_path=args.output_path, workers=args.workers)


if __name__ == "__main__":
    main()
