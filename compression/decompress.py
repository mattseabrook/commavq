#!/usr/bin/env python3
import os
import lzma
import numpy as np
from pathlib import Path
import multiprocessing
from datasets import load_dataset, DatasetDict

HERE = Path(__file__).resolve().parent

# Directory where compressed files are stored (input)
compressed_dir = HERE / "compression_challenge_submission"
# Directory where decompressed files will be written (output)
output_dir = Path(
    os.environ.get("OUTPUT_DIR", HERE / "compression_challenge_submission_decompressed")
)


def decompress_bytes(x: bytes) -> np.ndarray:
    tokens = np.frombuffer(lzma.decompress(x), dtype=np.int16)
    return tokens.reshape(128, -1).T.reshape(-1, 8, 16)


def decompress_example(example, compressed_dir, output_dir):
    path = Path(example["path"])
    # Read the compressed file
    compressed_path = compressed_dir / path.name
    with open(compressed_path, "rb") as f:
        tokens = decompress_bytes(f.read())
    # Write the decompressed file
    decompressed_path = output_dir / path.name
    np.save(decompressed_path, tokens)
    # Load the original file for comparison
    original_tokens = np.load(path)
    assert np.all(
        tokens == original_tokens
    ), f"decompressed data does not match original data for {path}"


if __name__ == "__main__":
    num_proc = multiprocessing.cpu_count()
    # Create the output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)
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
    # Decompress with multiprocessing, passing directories as kwargs
    ds.map(
        decompress_example,
        desc="decompress_example",
        num_proc=num_proc,
        load_from_cache_file=False,
        fn_kwargs={"compressed_dir": compressed_dir, "output_dir": output_dir},
    )
