#!/usr/bin/env python3
import numpy as np
import argparse
import struct
from pathlib import Path

class AdaptiveArithmeticEncoder:
    def __init__(self, alphabet_size=1024, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.pending_bits = 0
        self.output_bits = []
        self.alphabet_size = alphabet_size
        self.freq = [1] * alphabet_size
        self.total = alphabet_size  # Initial sum

    def _write_bit(self, bit):
        self.output_bits.append(bit)
        while self.pending_bits > 0:
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
        
        # Update frequency after encoding symbol
        self.freq[symbol] += 1
        self.total += 1

    def encode_symbol(self, symbol):
        self._update_range(symbol)

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
    tokens = tokens.reshape(1200, 8, 16).transpose(1, 2, 0).ravel()  # Optimized grouping
    tokens = tokens.astype(np.int16)
    
    encoder = AdaptiveArithmeticEncoder()
    for symbol in tokens:
        encoder.encode_symbol(symbol)
    
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