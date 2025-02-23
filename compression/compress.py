#!/usr/bin/env python3
import numpy as np
import argparse
import struct
from pathlib import Path

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
        self.freq = [1] * 2047  # -1023 to +1023 → 0-2046
        self.total = 2047

    def _write_bit(self, bit):
        self.output_bits.append(bit)
        while self.pending_bits:
            self.output_bits.append(1 - bit)
            self.pending_bits -= 1

    def _update_range(self, symbol):
        cum_freq = np.cumsum([0] + self.freq)
        total = self.total
        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * cum_freq[symbol+1] // total) - 1
        self.low = self.low + (range_width * cum_freq[symbol] // total)

        while True:
            if self.high < self.half_range:
                self._write_bit(0)
                self.low <<= 1
                self.high = (self.high << 1) | 1
            elif self.low >= self.half_range:
                self._write_bit(1)
                self.low = (self.low - self.half_range) << 1
                self.high = ((self.high - self.half_range) << 1) | 1
            elif (self.low >= self.quarter_range and 
                  self.high < 3 * self.quarter_range):
                self.pending_bits += 1
                self.low = (self.low - self.quarter_range) << 1
                self.high = ((self.high - self.quarter_range) << 1) | 1
            else:
                break
        
        self.freq[symbol] += 1
        self.total += 1

    def encode_deltas(self, arr):
        # Store first value as-is
        self._update_range(arr[0])
        # Encode deltas for remaining values
        prev = arr[0]
        for val in arr[1:]:
            delta = val - prev
            mapped = delta + 1023  # Convert to 0-2046
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
    # Reorganize: (1200,8,16) → (8,16,1200)
    tokens = tokens.reshape(1200, 8, 16).transpose(1, 2, 0).ravel()
    
    encoder = DeltaArithmeticEncoder()
    
    # Process each spatial position's time series
    for i in range(0, len(tokens), 1200):
        chunk = tokens[i:i+1200]
        encoder.encode_deltas(chunk)
    
    encoded_data = encoder.finish()
    return struct.pack("<I", tokens.size) + encoded_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file", nargs="?")
    args = parser.parse_args()
    
    tokens = np.load(args.input_file)
    compressed = compress_tokens(tokens)
    
    output_path = Path(args.output_file or f"{args.input_file}.aac")
    with open(output_path, "wb") as f:
        f.write(compressed)
    
    orig_size = tokens.size * 2  # int16 = 2 bytes
    ratio = orig_size / len(compressed)
    print(f"Compression ratio: {ratio:.2f}")

if __name__ == "__main__":
    main()