#!/usr/bin/env python3
import numpy as np
import argparse
import struct
from pathlib import Path

class AdaptiveArithmeticDecoder:
    def __init__(self, bitstream, alphabet_size=1024, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.code = 0
        self.bitstream = bitstream
        self.bit_ptr = 0
        self.alphabet_size = alphabet_size
        self.freq = [1] * alphabet_size
        self.total = alphabet_size
        
        for _ in range(precision):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        if self.bit_ptr >= len(self.bitstream) * 8:
            return 0
        byte = self.bitstream[self.bit_ptr // 8]
        bit = (byte >> (self.bit_ptr % 8)) & 1
        self.bit_ptr += 1
        return bit

    def _decode_symbol(self):
        cum_freq = np.cumsum([0] + self.freq)
        total = self.total
        range_width = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // range_width
        symbol = np.searchsorted(cum_freq, value, side='right') - 1
        
        # Update range
        self.high = self.low + (range_width * cum_freq[symbol+1] // total) - 1
        self.low = self.low + (range_width * cum_freq[symbol] // total)
        
        while True:
            if self.high < self.half_range:
                pass
            elif self.low >= self.half_range:
                self.code -= self.half_range
                self.low -= self.half_range
                self.high -= self.half_range
            elif (self.low >= self.quarter_range and 
                  self.high < 3 * self.quarter_range):
                self.code -= self.quarter_range
                self.low -= self.quarter_range
                self.high -= self.quarter_range
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._read_bit()
        
        # Update frequency
        self.freq[symbol] += 1
        self.total += 1
        return symbol

    def decode_stream(self, num_symbols):
        return [self._decode_symbol() for _ in range(num_symbols)]

def decompress_bytes(data: bytes) -> np.ndarray:
    n = struct.unpack("<I", data[:4])[0]
    decoder = AdaptiveArithmeticDecoder(data[4:])
    tokens = np.array(decoder.decode_stream(n), dtype=np.int16)
    tokens = tokens.reshape(8, 16, 1200).transpose(2, 0, 1)  # Reverse grouping
    return tokens.astype(np.int16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compressed_file")
    parser.add_argument("--output")
    parser.add_argument("--verify", default=None)
    args = parser.parse_args()
    
    with open(args.compressed_file, "rb") as f:
        data = f.read()
    
    tokens = decompress_bytes(data)
    output_path = Path(args.output or f"{args.compressed_file}.npy")
    np.save(output_path, tokens)
    
    if args.verify:
        orig = np.load(args.verify)
        assert np.array_equal(orig, tokens), "Verification failed!"
        print("Verification passed")

if __name__ == "__main__":
    main()