#!/usr/bin/env python3
"""Stage the transformed corpus into .tokbin files for the C++ count_tokens.

Walks <corpus_root>/data (jsonl) and <corpus_root>/data_clustered (parquet),
streams each file and writes a .tokbin mirror under <out_root> preserving
relative paths (extension replaced with .tokbin). instruction and response
are separate docs; NO truncation, NO sampling limits — the full corpus.

.tokbin format (little-endian):
  u32 magic "TKB1" (0x314B4254), u64 n_docs, u64 src_size, u64 src_mtime_ns,
  then n_docs x { u32 len, u8 bytes[len] }.

Parallel over files (ProcessPoolExecutor), tqdm progress, and skips outputs
whose recorded source size+mtime still match (resumable).

Parquet is read with pyarrow iter_batches (true streaming, constant memory);
polars' streaming engine was considered but collect() materializes whole
files, which breaks the memory budget on the biggest ones.

Run inside hrm_text_clean_image (needs pyarrow, orjson, tqdm):

    python3 scripts/stage_corpus_tokbin.py [CORPUS_ROOT] [OUT_ROOT] [--workers N]
"""

import argparse
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

TOKBIN_MAGIC = 0x314B4254  # "TKB1"
HEADER = struct.Struct("<IQQQ")  # magic, n_docs, src_size, src_mtime_ns


def write_tokbin(src_path, out_path, kind):
    """Stream one jsonl/parquet file into its .tokbin mirror."""
    import pyarrow.parquet as pq

    tmp = out_path + ".tmp"
    n_docs = 0
    with open(tmp, "wb") as out:
        out.write(HEADER.pack(TOKBIN_MAGIC, 0, 0, 0))  # patched at close
        buf = bytearray()
        pack = struct.Struct("<I").pack

        def flush():
            if buf:
                out.write(buf)
                buf.clear()

        def emit(s):
            nonlocal n_docs, buf
            b = s.encode("utf-8") if s else b""
            buf += pack(len(b))
            buf += b
            n_docs += 1
            if len(buf) >= 1 << 24:
                flush()

        if kind == "jsonl":
            import orjson

            with open(src_path, "rb") as f:
                for line in f:
                    rec = orjson.loads(line)
                    emit(rec["instruction"])
                    emit(rec["response"])
        else:  # parquet
            pf = pq.ParquetFile(src_path)
            for batch in pf.iter_batches(batch_size=65536,
                                         columns=["instruction", "response"]):
                inst = batch.column("instruction").to_pylist()
                resp = batch.column("response").to_pylist()
                for a, b in zip(inst, resp):
                    emit(a)
                    emit(b)
        flush()

    st = os.stat(src_path)
    with open(tmp, "r+b") as out:  # patch header with real values
        out.write(HEADER.pack(TOKBIN_MAGIC, n_docs, st.st_size, st.st_mtime_ns))
    os.replace(tmp, out_path)
    return n_docs


def is_fresh(src_path, out_path):
    """Skip if the .tokbin exists and recorded src size/mtime still match."""
    try:
        st = os.stat(src_path)
        with open(out_path, "rb") as f:
            magic, _n, src_size, src_mtime = HEADER.unpack(f.read(HEADER.size))
        return magic == TOKBIN_MAGIC and src_size == st.st_size \
            and src_mtime == st.st_mtime_ns
    except (OSError, struct.error):
        return False


def process_one(args):
    src_path, out_path, kind = args
    if is_fresh(src_path, out_path):
        return out_path, None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = write_tokbin(src_path, out_path, kind)
    return out_path, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_root", nargs="?",
                    default=os.path.expanduser("~/hrm_text_tokenizer_cache"))
    ap.add_argument("out_root", nargs="?",
                    default=os.path.expanduser("~/hrm_text_tokbin"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    tasks = []
    for sub, kind in (("data", "jsonl"), ("data_clustered", "parquet")):
        root = os.path.join(args.corpus_root, sub)
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                src = os.path.join(dirpath, fn)
                rel = os.path.relpath(src, args.corpus_root)
                stem, ext = os.path.splitext(rel)
                if (kind == "jsonl" and ext != ".jsonl") or \
                   (kind == "parquet" and ext != ".parquet"):
                    continue
                out = os.path.join(args.out_root, stem + ".tokbin")
                tasks.append((src, out, kind))

    print(f"[stage] {len(tasks)} source files -> {args.out_root}", flush=True)
    done_docs = 0
    skipped = 0
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, t) for t in tasks]
        with tqdm(total=len(futures), desc="staging", mininterval=5) as bar:
            for fut in as_completed(futures):
                out_path, n = fut.result()
                if n is None:
                    skipped += 1
                else:
                    done_docs += n
                bar.update(1)
                bar.set_postfix(new_docs=f"{done_docs:,}", skip=skipped)
    print(f"[stage] done: {done_docs:,} docs written, {skipped} files up-to-date",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
