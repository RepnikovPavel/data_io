from typing import Callable
import os
import re

import orjson
import pyarrow.parquet as pq


def load_local_dataset(repo_id, *args, **kwargs):
    """load_dataset() fully offline: resolves repo_id to the prefetched HF
    cache snapshot (download stage) and loads from local files.
    Works under HF_HUB_OFFLINE=1."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id, repo_type="dataset", local_files_only=True)
    return load_dataset(path, *args, **kwargs)


def read_jsonl(filename):
    with open(filename, "rb") as f:
        return map(orjson.loads, f.readlines())


def write_jsonl(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        for item in data:
            f.write(orjson.dumps(item))
            f.write(b"\n")


def read_data_file(filename: str):
    # Read packed
    if os.path.basename(filename).startswith("packed__"):
        assert filename.endswith(".jsonl"), f"{filename}: Packed mode only support jsonl format"

        return read_jsonl(filename)
    
    # Read normal
    m = re.fullmatch(r"(\d+)__(.*)", os.path.basename(filename))
    assert m, f"Unknown filename format {filename}"

    if filename.endswith(".jsonl"):
        data = read_jsonl(filename)
    elif filename.endswith(".parquet"):
        data = pq.read_table(filename).to_pylist()
    else:
        raise NotImplementedError(f"Unknown type of file {filename}")
    
    return [{
        "rsize": int(m.group(1)),
        "name": m.group(2),
        "data": data
    }]
