#!/usr/bin/env python3
import os
import numpy as np
import argparse
import struct
from pathlib import Path

# ----------------- Arithmetic Decoder Class -----------------
class ArithmeticDecoder:
    def __init__(self, bitstream, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.code = 0
        self.bitstream = bitstream
        self.bit_index = 0
        self.total_bits = len(bitstream) * 8
        for _ in range(precision):
            self.code = (self.code << 1) | self.read_bit()

    def read_bit(self):
        if self.bit_index >= self.total_bits:
            return 0
        byte_index = self.bit_index // 8
        bit_pos = self.bit_index % 8
        self.bit_index += 1
        return (self.bitstream[byte_index] >> bit_pos) & 1

    def decode_symbol(self, cum_freq, total):
        range_width = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // range_width
        low_idx = 0
        high_idx = len(cum_freq) - 1
        while low_idx < high_idx - 1:
            mid = (low_idx + high_idx) // 2
            if cum_freq[mid] > value:
                high_idx = mid
            else:
                low_idx = mid
        symbol = low_idx
        self.high = self.low + (range_width * cum_freq[symbol + 1] // total) - 1
        self.low = self.low + (range_width * cum_freq[symbol] // total)
        while True:
            if self.high < self.half_range:
                pass
            elif self.low >= self.half_range:
                self.code -= self.half_range
                self.low -= self.half_range
                self.high -= self.half_range
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.code -= self.quarter_range
                self.low -= self.quarter_range
                self.high -= self.quarter_range
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self.read_bit()
        return symbol

# ----------------- Decompression Pipeline -----------------
def decompress_bytes(x: bytes) -> np.ndarray:
    offset = 0
    # Read header: first 4 bytes = number of tokens.
    n = struct.unpack("<I", x[offset:offset+4])[0]
    offset += 4
    alphabet_size = 1024
    freq = []
    for _ in range(alphabet_size):
        f = struct.unpack("<I", x[offset:offset+4])[0]
        freq.append(f)
        offset += 4
    total = sum(freq)
    cum_freq = [0]
    for f in freq:
        cum_freq.append(cum_freq[-1] + f)
    bitstream = x[offset:]
    decoder = ArithmeticDecoder(bitstream, precision=32)
    tokens_decoded = []
    for _ in range(n):
        sym = decoder.decode_symbol(cum_freq, total)
        tokens_decoded.append(sym)
    tokens = np.array(tokens_decoded, dtype=np.int16)
    # Reverse the reordering: tokens were produced via .reshape(1200,128).T.ravel()
    tokens = tokens.reshape(128, -1).T.reshape(1200, 8, 16)
    return tokens

def main():
    parser = argparse.ArgumentParser(
        description="Decompress a file compressed with arithmetic coding on transposed tokens (no delta)."
    )
    parser.add_argument("compressed_file", help="Path to the compressed file")
    parser.add_argument("--verify", help="Path to the original .npy file for verification")
    parser.add_argument("--output", help="Output .npy file (default: <compressed_file>_decompressed.npy)")
    args = parser.parse_args()
    comp_path = Path(args.compressed_file)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = comp_path.with_name(comp_path.stem + "_decompressed.npy")
    with open(comp_path, "rb") as f:
        data = f.read()
    tokens = decompress_bytes(data)
    np.save(out_path, tokens)
    print(f"Decompressed file saved to: {out_path}")
    if args.verify:
        original = np.load(Path(args.verify))
        if np.array_equal(original, tokens):
            print("Verification successful: decompressed data matches the original.")
        else:
            print("Verification failed: decompressed data does NOT match the original.")

if __name__ == "__main__":
    main()
