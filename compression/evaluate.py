#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import multiprocessing
from datasets import load_dataset, DatasetDict

HERE = Path(__file__).resolve().parent

archive_path = Path(
    os.environ.get("PACKED_ARCHIVE", HERE / "compression_challenge_submission.zip")
)
unpacked_archive = Path(
    os.environ.get(
        "UNPACKED_ARCHIVE", HERE / "compression_challenge_submission_decompressed"
    )
)


def compare(example):
    path = Path(example["path"])
    decompressed_path = unpacked_archive / path.name
    tokens = np.load(decompressed_path)
    original_tokens = np.load(path)
    assert np.all(
        tokens == original_tokens
    ), f"decompressed data does not match original data for {path}"


if __name__ == "__main__":
    num_proc = multiprocessing.cpu_count()
    # Define local data files for splits '0' and '1'
    splits = ["0", "1"]
    data_files = {
        "0": str(HERE.parent / "data" / "data_0_to_2500.zip"),
        "1": str(HERE.parent / "data" / "data_2500_to_5000.zip"),
    }
    # Load dataset from local files
    ds = load_dataset(
        "commaai/commavq", num_proc=num_proc, split=splits, data_files=data_files
    )
    ds = DatasetDict(zip(splits, ds))
    # Compare decompressed files to originals
    ds.map(compare, desc="compare", num_proc=num_proc, load_from_cache_file=False)
    # Calculate and print compression rate
    rate = (
        sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8
    ) / archive_path.stat().st_size
    print(f"Compression rate: {rate:.1f}")
