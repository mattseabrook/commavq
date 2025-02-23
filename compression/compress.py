#!/usr/bin/env python3
import os
import multiprocessing
import shutil
import numpy as np
import struct
from pathlib import Path
from datasets import load_dataset, DatasetDict

HERE = Path(__file__).resolve().parent
output_dir = HERE / './compression_challenge_submission/'
os.makedirs(output_dir, exist_ok=True)

# --- Fenwick Tree Helper Functions ---
def fenwicks_init(size):
    # Create a Fenwick tree for 'size' elements (1-indexed)
    return [0] * (size + 1)

def fenwicks_update(fenwicks, index, delta):
    # Update Fenwick tree at 1-indexed position 'index' by adding delta
    while index < len(fenwicks):
        fenwicks[index] += delta
        index += index & -index

def fenwicks_sum(fenwicks, index):
    # Return cumulative frequency sum from index 1 up to 'index'
    s = 0
    while index > 0:
        s += fenwicks[index]
        index -= index & -index
    return s

# --- Delta Arithmetic Encoder using Fenwick Tree ---
class DeltaArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.pending_bits = 0
        self.output_bits = []
        # For mapped deltas: valid range 0 .. 2046 (i.e. -1023 to +1023)
        self.size = 2047
        self.fenwicks = fenwicks_init(self.size)
        # Initialize each frequency to 1 so that total = 2047
        for i in range(1, self.size + 1):
            fenwicks_update(self.fenwicks, i, 1)
        self.total = fenwicks_sum(self.fenwicks, self.size)

    def _write_bit(self, bit):
        self.output_bits.append(bit)
        while self.pending_bits:
            self.output_bits.append(1 - bit)
            self.pending_bits -= 1

    def _update_range(self, symbol):
        # Use Fenwick tree queries instead of full np.cumsum
        cum_low = fenwicks_sum(self.fenwicks, symbol)       # cumulative freq for symbols < symbol
        cum_high = fenwicks_sum(self.fenwicks, symbol + 1)    # cumulative freq for symbols <= symbol
        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * cum_high // self.total) - 1
        self.low = self.low + (range_width * cum_low // self.total)

        while True:
            if self.high < self.half_range:
                self._write_bit(0)
                self.low <<= 1
                self.high = (self.high << 1) | 1
            elif self.low >= self.half_range:
                self._write_bit(1)
                self.low = (self.low - self.half_range) << 1
                self.high = ((self.high - self.half_range) << 1) | 1
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.pending_bits += 1
                self.low = (self.low - self.quarter_range) << 1
                self.high = ((self.high - self.quarter_range) << 1) | 1
            else:
                break

        # Update frequency for the symbol in the Fenwicks tree
        fenwicks_update(self.fenwicks, symbol + 1, 1)
        self.total += 1

    def encode_deltas(self, arr):
        # Encode the first token as-is, then encode deltas
        self._update_range(arr[0])
        prev = arr[0]
        for val in arr[1:]:
            delta = val - prev
            mapped = delta + 1023  # Map delta into range 0..2046
            self._update_range(mapped)
            prev = val

    def finish(self):
        self.pending_bits += 1
        self._write_bit(1 if self.low >= self.quarter_range else 0)
        bytes_out = bytearray()
        current_byte = 0
        bits_in_byte = 0
        for bit in self.output_bits:
            current_byte |= bit << bits_in_byte
            bits_in_byte += 1
            if bits_in_byte == 8:
                bytes_out.append(current_byte)
                current_byte = 0
                bits_in_byte = 0
        if bits_in_byte:
            bytes_out.append(current_byte)
        return bytes_out

def compress_tokens(tokens: np.ndarray) -> bytes:
    # Apply the standalone's token transformation:
    # (1200,8,16) -> (8,16,1200) then flatten.
    tokens = tokens.astype(np.int16).reshape(1200, 8, 16).transpose(1, 2, 0).ravel()
    
    encoder = DeltaArithmeticEncoder()
    # Process each spatial position's time series (each with 1200 tokens)
    for i in range(0, len(tokens), 1200):
        chunk = tokens[i:i+1200]
        encoder.encode_deltas(chunk)
    
    encoded_data = encoder.finish()
    # Prepend a 4-byte little-endian integer with the token count
    return struct.pack("<I", tokens.size) + encoded_data

def compress_example(example):
    path = Path(example['path'])
    tokens = np.load(path)
    compressed = compress_tokens(tokens)
    # Write the compressed output into the evaluation folder (their process)
    out_path = output_dir / path.name
    with open(out_path, "wb") as f:
        f.write(compressed)
    # Record the compression ratio (raw int16 data: tokens.size * 2 bytes)
    example['compression_ratio'] = (tokens.size * 2) / len(compressed)
    return example

if __name__ == '__main__':
    num_proc = multiprocessing.cpu_count()

    splits = ['0', '1']
    data_files = {'0': 'data_0_to_2500.zip', '1': 'data_2500_to_5000.zip'}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits, data_files=data_files)
    ds = DatasetDict(zip(splits, ds))

    # Process each example in parallel and write out the compressed files in the evaluation folder
    ds.map(compress_example, desc="compress_example", num_proc=num_proc, load_from_cache_file=False)

    # For evaluation, copy the decompression script and archive the entire folder (as per their test)
    shutil.copy(HERE / 'decompress.py', output_dir)
    shutil.make_archive(HERE / 'compression_challenge_submission', 'zip', output_dir)

    # Their evaluation calculation uses:
    # original_size = (number of examples * 1200 * 128 * 10/8) bytes
    original_size = sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8
    zip_size = os.path.getsize(HERE / "compression_challenge_submission.zip")
    rate = original_size / zip_size
    print(f"Compression rate: {rate:.1f}")
