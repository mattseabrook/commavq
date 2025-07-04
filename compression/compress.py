#!/usr/bin/env python3
import os
import lzma
import multiprocessing
import shutil
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

output_dir = HERE / "./compression_challenge_submission/"


def compress_tokens(tokens: np.ndarray) -> bytes:
    tokens = (
        tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
    )  # transposing increases compression rate ;)
    return lzma.compress(tokens)


def compress_example(example):
    path = Path(example["path"])
    tokens = np.load(path)
    compressed = compress_tokens(tokens)
    compression_rate = (tokens.size * 10 / 8) / len(compressed)  # 10 bits per token
    with open(output_dir / path.name, "wb") as f:
        f.write(compressed)
    example["compression_rate"] = compression_rate
    return example


def create_zip_with_progress(source_dir, output_path):
    total_size = sum(f.stat().st_size for f in source_dir.glob("**/*") if f.is_file())
    with ZipFile(output_path, "w", ZIP_DEFLATED) as zipf:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Zipping") as pbar:
            for file_path in source_dir.glob("**/*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)
                    pbar.update(file_path.stat().st_size)


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    num_proc = multiprocessing.cpu_count()

    # Load split 0 and 1
    splits = ["0", "1"]
    data_files = {
        "0": str(HERE.parent / "data" / "data_0_to_2500.zip"),
        "1": str(HERE.parent / "data" / "data_2500_to_5000.zip"),
    }
    ds = load_dataset(
        "commaai/commavq", num_proc=num_proc, split=splits, data_files=data_files
    )
    ds = DatasetDict(zip(splits, ds))

    # Compress with built-in progress bar
    ratios = ds.map(
        compress_example,
        desc="Compressing",
        num_proc=num_proc,
        load_from_cache_file=False,
    )

    # Make archive with progress
    shutil.copy(HERE / "decompress.py", output_dir)
    zip_path = HERE / "compression_challenge_submission.zip"
    create_zip_with_progress(output_dir, zip_path)

    # Print compression rate
    rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / os.path.getsize(zip_path)
    print(f"Compression rate: {rate:.1f}")
