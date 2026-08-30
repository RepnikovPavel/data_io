import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import orjson
from tqdm import tqdm


def _process_file(file_path):
    """Read one scibench json file and transform it into
    condition/instruction/response records."""
    t0 = time.time()
    with open(file_path, 'r') as f:
        file_data = json.load(f)

    records = []
    for d in file_data:
        problem = d.get('problem_text', '').strip()
        solution = d.get('solution', '').strip()
        answer_latex = d.get('answer_latex', '').strip()
        answer_number = d.get('answer_number', '').strip()
        if answer_latex == f"${answer_number}$":
            answer_latex = answer_number

        if solution:
            records.append({
                "condition": "cot",
                "instruction": problem,
                "response": solution
            })

        records.append({
            "condition": "direct",
            "instruction": problem,
            "response": answer_latex
        })

    return os.path.basename(file_path), records, os.path.getsize(file_path), \
        time.time() - t0


def _iter_results(file_paths, workers):
    # pool.map yields results in input order, so the original record order
    # (os.listdir order, files processed one after another) is preserved.
    if workers > 1 and len(file_paths) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(_process_file, file_paths)
    else:
        for file_path in file_paths:
            yield _process_file(file_path)


def clean_scibench(input_dir: str, output_path: str, workers: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    started = time.time()

    file_paths = [os.path.join(input_dir, f)
                  for f in os.listdir(input_dir) if f.endswith('.json')]
    total_bytes = sum(os.path.getsize(f) for f in file_paths)
    print(f"Processing {len(file_paths)} files, {total_bytes / 2**20:.1f} MiB total "
          f"({workers} workers)", flush=True)

    n_records = 0
    with open(output_path, "wb") as out, \
            tqdm(total=len(file_paths), desc="files", unit="file") as bar:
        for i, (name, records, n_bytes, secs) in enumerate(
                _iter_results(file_paths, workers), 1):
            for record in records:
                out.write(orjson.dumps(record))
                out.write(b"\n")
            n_records += len(records)
            bar.update(1)
            tqdm.write(f"[{i}/{len(file_paths)}] {name}: "
                       f"{len(records)} rows in {secs:.1f}s")

    elapsed = time.time() - started
    print(f"Done: {n_records} rows from {len(file_paths)} files -> {output_path} "
          f"in {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str,
        default='/mnt/hdd2/datasets_text/Platypus/scibench/dataset/original',
        help='directory containing the scibench *.json files')
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data/Platypus/scibench.jsonl',
        help='path of the output jsonl file')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of files processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    clean_scibench(
        input_dir=args.input_dir,
        output_path=args.output_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
