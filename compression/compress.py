#!/usr/bin/env python3
import os
import lzma
import multiprocessing
import shutil
import numpy as np
import cupy as cp  # Import CuPy for GPU acceleration
from zipfile import ZipFile, ZIP_STORED
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
output_dir = HERE / "./compression_challenge_submission/"


def delta_encode(tokens_gpu: cp.ndarray) -> cp.ndarray:
    """Encodes tokens using delta encoding on GPU."""
    return cp.diff(tokens_gpu.astype(cp.int32), prepend=0).astype(cp.int16)


def compress_tokens(tokens: np.ndarray) -> bytes:
    """Delta-encodes and compresses tokens using LZMA with max settings."""
    # Move data to GPU
    tokens_gpu = cp.asarray(tokens.astype(cp.int16))
    tokens_raveled_gpu = tokens_gpu.reshape(-1, 128).T.ravel()
    encoded_tokens_gpu = delta_encode(tokens_raveled_gpu)

    # Move back to CPU for LZMA compression
    encoded_tokens = cp.asnumpy(encoded_tokens_gpu)

    # Compress with LZMA at max settings (preset=9)
    compressor = lzma.LZMACompressor(preset=9)
    compressed = compressor.compress(encoded_tokens.tobytes())
    compressed += compressor.flush()
    return compressed


def compress_example(example):
    path = Path(example["path"])
    tokens = np.load(path)
    compressed = compress_tokens(tokens)
    compression_rate = (tokens.size * 10 / 8) / len(compressed)
    with open(output_dir / path.name, "wb") as f:
        f.write(compressed)
    example["compression_rate"] = compression_rate
    return example


def create_zip_with_progress(source_dir, output_path):
    file_paths = [f for f in source_dir.glob("**/*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in file_paths)

    def read_file(file_path):
        with open(file_path, "rb") as f:
            return file_path, f.read()

    with ThreadPoolExecutor(max_workers=32) as executor:
        file_contents = list(
            tqdm(
                executor.map(read_file, file_paths),
                total=len(file_paths),
                desc="Reading files",
            )
        )

    with ZipFile(output_path, "w", ZIP_STORED) as zipf:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Writing to zip"
        ) as pbar:
            for file_path, content in file_contents:
                arcname = file_path.relative_to(source_dir)
                zipf.writestr(str(arcname), content)
                pbar.update(len(content))


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    num_proc = multiprocessing.cpu_count()

    # Load dataset splits
    splits = ["0", "1"]
    data_files = {
        "0": str(HERE.parent / "data" / "data_0_to_2500.zip"),
        "1": str(HERE.parent / "data" / "data_2500_to_5000.zip"),
    }
    ds = load_dataset(
        "commaai/commavq", num_proc=num_proc, split=splits, data_files=data_files
    )
    ds = DatasetDict(zip(splits, ds))

    # Compress files with progress using LZMA
    ratios = ds.map(
        compress_example,
        desc="Compressing",
        num_proc=num_proc,
        load_from_cache_file=False,
    )

    # Create ZIP archive with progress
    shutil.copy(HERE / "decompress.py", output_dir)
    zip_path = HERE / "compression_challenge_submission.zip"
    create_zip_with_progress(output_dir, zip_path)

    # Report compression rate
    rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / os.path.getsize(zip_path)
    print(f"Compression rate: {rate:.1f}")
