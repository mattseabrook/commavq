#!/usr/bin/env python3
import os
import numpy as np
import argparse
import struct
from pathlib import Path

# ----------------- Arithmetic Coding Classes -----------------
class ArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.pending_bits = 0
        self.output_bits = []  # List of bits (0 or 1)

    def write_bit(self, bit):
        self.output_bits.append(bit)
        while self.pending_bits:
            self.output_bits.append(1 - bit)
            self.pending_bits -= 1

    def encode_symbol(self, symbol, cum_freq, total):
        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * cum_freq[symbol + 1] // total) - 1
        self.low = self.low + (range_width * cum_freq[symbol] // total)
        while True:
            if self.high < self.half_range:
                self.write_bit(0)
                self.low <<= 1
                self.high = (self.high << 1) | 1
            elif self.low >= self.half_range:
                self.write_bit(1)
                self.low = (self.low - self.half_range) << 1
                self.high = ((self.high - self.half_range) << 1) | 1
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.pending_bits += 1
                self.low = (self.low - self.quarter_range) << 1
                self.high = ((self.high - self.quarter_range) << 1) | 1
            else:
                break

    def finish(self):
        self.pending_bits += 1
        if self.low < self.quarter_range:
            self.write_bit(0)
        else:
            self.write_bit(1)
        # Pack bits into bytes
        out_bytes = bytearray()
        current_byte = 0
        bits_in_byte = 0
        for bit in self.output_bits:
            current_byte |= (bit << bits_in_byte)
            bits_in_byte += 1
            if bits_in_byte == 8:
                out_bytes.append(current_byte)
                current_byte = 0
                bits_in_byte = 0
        if bits_in_byte:
            out_bytes.append(current_byte)
        return bytes(out_bytes)

# ----------------- Compression Pipeline -----------------
def compress_tokens(tokens: np.ndarray) -> bytes:
    # Reorder tokens: (1200,8,16) -> reshape to (1200,128) -> transpose to (128,1200) -> flatten.
    tokens = tokens.astype(np.int16).reshape(1200, 128).T.ravel()
    n = tokens.shape[0]  # should be 153600 tokens

    # Build frequency table (alphabet: 0..1023) with add-one smoothing.
    alphabet_size = 1024
    freq = [1] * alphabet_size
    for token in tokens:
        freq[int(token)] += 1
    total = sum(freq)
    cum_freq = [0]
    for f in freq:
        cum_freq.append(cum_freq[-1] + f)

    encoder = ArithmeticEncoder(precision=32)
    for token in tokens:
        encoder.encode_symbol(int(token), cum_freq, total)
    encoded_bits = encoder.finish()

    # Build header:
    # 1. 4 bytes: number of tokens (unsigned int)
    # 2. Frequency table: 1024 unsigned ints (4 bytes each)
    header = bytearray()
    header += struct.pack("<I", n)
    for f in freq:
        header += struct.pack("<I", f)

    return bytes(header) + encoded_bits

def main():
    parser = argparse.ArgumentParser(
        description="Compress a .npy file using arithmetic coding on transposed tokens (no delta)."
    )
    parser.add_argument("input_file", help="Path to the input .npy file")
    parser.add_argument("output_file", nargs="?", help="Output file (default: input_file + .ac)")
    args = parser.parse_args()
    input_path = Path(args.input_file)
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_suffix(input_path.suffix + ".ac")
    tokens = np.load(input_path)
    compressed = compress_tokens(tokens)
    with open(output_path, "wb") as f:
        f.write(compressed)
    # Compression ratio is calculated based on original size (10 bits per token).
    original_bits = tokens.size * 10
    ratio = original_bits / (len(compressed) * 8)
    print(f"Compression ratio: {ratio:.2f}")
    print(f"Compressed file saved to: {output_path}")

if __name__ == "__main__":
    main()
