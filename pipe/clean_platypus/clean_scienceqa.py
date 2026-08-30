import argparse
import os
import time

from tqdm import tqdm

from utils import load_local_dataset, write_jsonl

SPLITS = ("train", "validation", "test")


def transform_batch(batch):
    """Vectorized equivalent of the original per-row loop.

    Emits, per input row and in order: a "cot" record when the solution is
    non-empty, then always a "direct" record.
    """
    conditions = []
    instructions = []
    responses = []
    for question, choices, lecture, solution, answer in zip(
            batch["question"], batch["choices"], batch["lecture"],
            batch["solution"], batch["answer"]):
        question = question.strip()
        lecture = lecture.strip()
        solution = solution.strip()

        formatted_choices = '\n'.join(
            f'{chr(65 + i)}: {choice.strip()}' for i, choice in enumerate(choices))

        if solution:
            # with rationale
            if lecture:
                instruction = (
                    "Solve the following question using the information provided in the lecture."
                    f"\n\n{question}\nOptions:\n{formatted_choices}\n\nLecture: {lecture}")
            else:
                instruction = f"{question}\nOptions:\n{formatted_choices}"
            conditions.append("cot")
            instructions.append(instruction)
            responses.append(f"{solution}\n\nAnswer: {chr(65 + answer)}")

        # without rationale
        if lecture:
            instruction = (
                "Choose the correct option letter for the following question based on the "
                f"information from the lecture.\n\n{question}\nOptions:\n{formatted_choices}"
                f"\n\nLecture: {lecture}")
        else:
            instruction = f"{question}\nOptions:\n{formatted_choices}"
        conditions.append("direct")
        instructions.append(instruction)
        responses.append(f"{chr(65 + answer)}")

    return {"condition": conditions, "instruction": instructions, "response": responses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus/scienceqa.jsonl',
        help='output jsonl file path')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='num_proc for datasets.map (default: min(8, cpu_count))')
    args = parser.parse_args()

    started = time.time()
    dataset = load_local_dataset('metaeval/ScienceQA_text_only')

    new_data = []
    with tqdm(total=sum(len(dataset[s]) for s in SPLITS),
              desc="splits", unit="row") as bar:
        for i, split in enumerate(SPLITS, 1):
            t0 = time.time()
            mapped = dataset[split].map(
                transform_batch,
                batched=True,
                num_proc=args.workers,
                remove_columns=dataset[split].column_names,
                desc=f"transform {split}",
            )
            n_in = len(dataset[split])
            new_data.extend(
                {"condition": c, "instruction": ins, "response": r}
                for c, ins, r in zip(mapped["condition"], mapped["instruction"],
                                     mapped["response"]))
            bar.update(n_in)
            tqdm.write(f"[{i}/{len(SPLITS)}] {split}: {n_in} input rows -> "
                       f"{len(mapped)} records in {time.time() - t0:.1f}s")

    write_jsonl(args.output_path, new_data)
    elapsed = time.time() - started
    print(f"Done: {len(new_data)} records -> {args.output_path} in {elapsed:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
