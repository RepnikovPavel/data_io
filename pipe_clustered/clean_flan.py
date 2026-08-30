import argparse
import os
import shutil
import string
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import polars as pl
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*PartitionBy.*unstable.*")


INCLUDED_SUBSETS = {
    "direct": [
        # Few-shot
        "dialog_fsopt_data",
        "flan_fsopt_data", "flan_fsnoopt_data",
        "niv2_fsopt_data",
        "t0_fsopt_data", "t0_fsnoopt_data",

        # Zero-shot
        "dialog_zsopt_data",
        "flan_zsopt_data", "flan_zsnoopt_data",
        "niv2_zsopt_data",
        "t0_zsopt_data", "t0_zsnoopt_data",
    ],
    "cot": [
        "cot_fsopt_data",
        "cot_zsopt_data",
    ]
}


def safe_filename(filename):
    # Remove or replace unsafe characters
    safe_chars = set(string.ascii_letters + string.digits + "_-. ")
    return "".join(c if c in safe_chars else "_" for c in filename)


def _init_worker(polars_threads):
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)
    warnings.filterwarnings("ignore", message=".*PartitionBy.*unstable.*")


def _process_subset(job):
    """Stream one subset: read all parquet files, write one parquet per task.

    Uses polars streaming engine with PartitionBy: constant memory,
    single pass over the data (no intermediate dump + merge).
    Returns (subset_name, n_tasks, n_bytes_in, seconds).
    """
    subset_name, condition, files, output_path, tmp_root = job
    t0 = time.time()
    tmp_dir = os.path.join(tmp_root, subset_name)
    (
        pl.scan_parquet(files, extra_columns="ignore")
        .select(
            "_task_name",
            pl.col("inputs").alias("instruction"),
            pl.col("targets").alias("response"),
        )
        .with_columns(pl.lit(condition).alias("condition"))
        .sink_parquet(
            pl.PartitionBy(tmp_dir, key="_task_name", include_key=False),
            compression="snappy",
            mkdir=True,
            engine="streaming",
        )
    )
    n_tasks = 0
    for task_dir in glob(os.path.join(tmp_dir, "*")):
        task_name = os.path.basename(task_dir).split("=", 1)[1]
        parts = sorted(glob(os.path.join(task_dir, "*.parquet")))
        out_file = os.path.join(
            output_path, f"{subset_name}__{safe_filename(task_name)}.parquet")
        if len(parts) == 1:
            os.replace(parts[0], out_file)
        else:
            # key exceeded the single-file size limit: merge parts
            pl.scan_parquet(parts).sink_parquet(out_file, compression="snappy")
        n_tasks += 1
    shutil.rmtree(tmp_dir, ignore_errors=True)
    n_bytes = sum(os.path.getsize(f) for f in files)
    return subset_name, n_tasks, n_bytes, time.time() - t0


def clean_flan(input_dir: str, included_subsets: dict[str, list[str]], output_path: str,
               workers: int, tmp_dir: str):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    started = time.time()

    jobs = []
    for condition, subset_names in included_subsets.items():
        for subset_name in subset_names:
            filenames = sorted(glob(os.path.join(input_dir, subset_name, "*.parquet")))
            if not filenames:
                tqdm.write(f"[warn] no parquet files for subset {subset_name}")
                continue
            jobs.append((subset_name, condition, filenames, output_path, tmp_dir))

    total_bytes = sum(os.path.getsize(f) for job in jobs for f in job[2])
    polars_threads = max(1, (os.cpu_count() or 1) // workers)
    print(f"Processing {len(jobs)} subsets, {total_bytes / 2**30:.1f} GiB total "
          f"({workers} workers, {polars_threads} polars threads/worker)", flush=True)

    # Live throughput monitor: subset-level bar updates are rare (a subset can
    # take tens of minutes), so report written bytes + rate every 15s.
    stop_monitor = threading.Event()

    def _monitor():
        while not stop_monitor.wait(15):
            written = 0
            for dirpath, _, filenames in os.walk(output_path):
                for f in filenames:
                    try:
                        written += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
            now = time.time()
            rate = (written - last[0]) / max(now - last[1], 1e-9) / 2**20
            tqdm.write(f"[progress] written {written / 2**30:.1f} GiB, "
                       f"rate {rate:.0f} MiB/s, elapsed {(now - started) / 60:.1f} min")
            last[0], last[1] = written, now

    last = [0, time.time()]
    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    done_bytes = 0
    try:
        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(polars_threads,)) as pool:
            futures = [pool.submit(_process_subset, job) for job in jobs]
            with tqdm(total=total_bytes, desc="subsets", unit="B", unit_scale=True,
                      unit_divisor=1024) as bar:
                for i, future in enumerate(as_completed(futures), 1):
                    subset_name, n_tasks, n_bytes, secs = future.result()
                    done_bytes += n_bytes
                    bar.update(n_bytes)
                    bar.set_postfix(subset=subset_name, done=f"{i}/{len(jobs)}")
                    tqdm.write(f"[{i}/{len(jobs)}] {subset_name}: {n_tasks} tasks, "
                               f"{n_bytes / 2**30:.2f} GiB in {secs:.0f}s")
    finally:
        stop_monitor.set()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.time() - started
    print(f"Done: {len(jobs)} subsets -> {output_path} in {elapsed / 60:.1f} min",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str,
        default='/mnt/hdd2/datasets_text/Open-Orca/FLAN',
        help='absolute path to raw_data/FLAN')
    parser.add_argument(
        '--output_path', type=str,
        default='/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/flan',
        help='absolute path to data_clustered/flan')
    parser.add_argument(
        '--tmp_dir', type=str, default=None,
        help='directory for intermediate per-task files '
             '(default: <output_path>/.tmp_parts)')
    parser.add_argument(
        '--workers', type=int, default=min(8, os.cpu_count() or 1),
        help='number of subsets processed in parallel (default: min(8, cpu_count))')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_path = args.output_path
    tmp_dir = args.tmp_dir or os.path.join(output_path, ".tmp_parts")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(str(input_dir))

    clean_flan(
        input_dir=input_dir,
        included_subsets=INCLUDED_SUBSETS,
        output_path=output_path,
        workers=args.workers,
        tmp_dir=tmp_dir,
    )


if __name__ == "__main__":
    main()
