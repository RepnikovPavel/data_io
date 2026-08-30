import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import polars as pl
from tqdm import tqdm


def _init_worker(polars_threads):
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)


def _process_file(job):
    """Filter + transform one parquet file, stream result to output."""
    file_path, output_path = job
    t0 = time.time()
    (
        pl.scan_parquet(file_path)
        .filter(
            (pl.col("language") == "en") &
            (~pl.col("query_seed_url").str.contains("Pleias self-knowledge")) &
            (pl.col("exercise") != "cooking")
        )
        .select([
            # instruction logic
            pl.when(pl.col("exercise") == "rag")
            .then(pl.col("query") + pl.col("constraints"))
            .otherwise(pl.col("query"))
            .alias("instruction"),

            # condition logic
            pl.when(pl.col("exercise").is_in({'creative writing', 'rag', 'memorization', 'constrained writing', 'editing'})).then(pl.lit("synth,cot"))
            .when(pl.col("exercise").is_in({'math mcq', 'mcq'})).then(pl.lit("synth,direct"))
            .when(pl.col("exercise") == "math exercise").then(pl.lit("synth,noisy,cot"))
            .otherwise(pl.lit(""))
            .alias("condition"),

            # response
            pl.col("synthetic_answer").alias("response")
        ])
        .sink_parquet(
            os.path.join(output_path, os.path.basename(file_path)),
            compression="snappy",
            engine="streaming",
        )
    )
    return os.path.basename(file_path), os.path.getsize(file_path), time.time() - t0


def clean_synth(input_dir: str, output_path: str, workers: int):
    os.makedirs(output_path, exist_ok=True)
    started = time.time()

    files = sorted(glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files in {input_dir}")
    total_bytes = sum(os.path.getsize(f) for f in files)
    polars_threads = max(1, (os.cpu_count() or 1) // workers)
    print(f"Processing {len(files)} files, {total_bytes / 2**30:.1f} GiB total "
          f"({workers} workers, {polars_threads} polars threads/worker)", flush=True)

    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(polars_threads,)) as pool:
        futures = [pool.submit(_process_file, (f, output_path)) for f in files]
        with tqdm(total=total_bytes, desc="files", unit="B", unit_scale=True,
                  unit_divisor=1024) as bar:
            for i, future in enumerate(as_completed(futures), 1):
                name, n_bytes, secs = future.result()
                bar.update(n_bytes)
                bar.set_postfix(file=name, done=f"{i}/{len(files)}")
                tqdm.write(f"[{i}/{len(files)}] {name}: "
                           f"{n_bytes / 2**30:.2f} GiB in {secs:.0f}s")

    elapsed = time.time() - started
    print(f"Done: {len(files)} files -> {output_path} in {elapsed / 60:.1f} min",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str,
        default='/mnt/hdd2/datasets_text/PleIAs/SYNTH',
        help='absolute path to raw SYNTH parquet files')
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/SYNTH',
        help='absolute path to data_clustered/SYNTH')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of files processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()

    clean_synth(
        input_dir=args.input_dir,
        output_path=args.output_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
