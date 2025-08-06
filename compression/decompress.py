#!/usr/bin/env python3
import os
import zstandard as zstd
import numpy as np
from pathlib import Path
import multiprocessing
from datasets import load_dataset, DatasetDict
import time  # For timing

HERE = Path(__file__).resolve().parent
compressed_dir = HERE / "compression_challenge_submission"
output_dir = Path(
    os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed")
)


def decompress_bytes(x: bytes) -> np.ndarray:
    dctx = zstd.ZstdDecompressor()
    decompressed_data = dctx.decompress(x)
    tokens = np.frombuffer(decompressed_data, dtype=np.int16)
    return tokens.reshape(128, -1).T.reshape(-1, 8, 16)


def decompress_example(example, compressed_dir, output_dir):
    path = Path(example["path"])
    compressed_path = compressed_dir / path.name
    with open(compressed_path, "rb") as f:
        tokens = decompress_bytes(f.read())
    decompressed_path = output_dir / path.name
    np.save(decompressed_path, tokens)
    original_tokens = np.load(path)
    assert np.all(tokens == original_tokens), f"Decompression failed for {path}"


if __name__ == "__main__":
    num_proc = multiprocessing.cpu_count()
    os.makedirs(output_dir, exist_ok=True)
    splits = ["0", "1"]
    data_files = {
        "0": str(HERE.parent / "data" / "data_0_to_2500.zip"),
        "1": str(HERE.parent / "data" / "data_2500_to_5000.zip"),
    }

    # Load dataset
    print("Starting dataset loading...")
    start_time = time.time()
    ds = load_dataset(
        "commaai/commavq", num_proc=num_proc, split=splits, data_files=data_files
    )
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    ds = DatasetDict(zip(splits, ds))

    # Decompress with progress and timing
    print("Starting decompression...")
    start_time = time.time()
    ds.map(
        decompress_example,
        desc="Decompressing files",
        num_proc=num_proc,
        load_from_cache_file=False,
        fn_kwargs={"compressed_dir": compressed_dir, "output_dir": output_dir},
    )
    print(f"Decompression completed in {time.time() - start_time:.2f} seconds")
