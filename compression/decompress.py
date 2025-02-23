#!/usr/bin/env python3
import os
import numpy as np
import struct
from pathlib import Path
import multiprocessing
from datasets import DatasetDict, load_dataset

HERE = Path(__file__).resolve().parent
output_dir = Path(os.environ.get('OUTPUT_DIR', HERE/'./compression_challenge_submission_decompressed/'))

class DeltaArithmeticDecoder:
    # Keep our optimized decoder class
    def __init__(self, bitstream, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.code = 0
        self.bitstream = bitstream
        self.bit_ptr = 0
        self.freq = [1] * 2047
        self.total = 2047
        
        for _ in range(precision):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        if self.bit_ptr >= len(self.bitstream)*8:
            return 0
        byte = self.bitstream[self.bit_ptr//8]
        bit = (byte >> (self.bit_ptr%8)) & 1
        self.bit_ptr += 1
        return bit

    def _decode_symbol(self):
        cum_freq = np.cumsum([0] + self.freq)
        total = self.total
        range_width = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // range_width
        symbol = np.searchsorted(cum_freq, value, side='right') - 1
        
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
        
        self.freq[symbol] += 1
        self.total += 1
        return symbol

    def decode_deltas(self, num_symbols):
        output = []
        first_val = self._decode_symbol()
        output.append(first_val)
        prev = first_val
        for _ in range(num_symbols-1):
            mapped = self._decode_symbol()
            delta = mapped - 1023
            val = prev + delta
            output.append(val)
            prev = val
        return output

def decompress_bytes(x: bytes) -> np.ndarray:
    n = struct.unpack("<I", x[:4])[0]
    decoder = DeltaArithmeticDecoder(x[4:])
    
    tokens = []
    for _ in range(0, n, 1200):
        chunk = decoder.decode_deltas(1200)
        tokens.extend(chunk)
    
    # Maintain original reshape order
    tokens = np.array(tokens, dtype=np.int16)
    return tokens.reshape(128, -1).T.reshape(-1, 8, 16)

def decompress_example(example):
    path = Path(example['path'])
    with open(output_dir/path.name, 'rb') as f:
        tokens = decompress_bytes(f.read())
    np.save(output_dir/path.name, tokens)
    assert np.all(tokens == np.load(path)), f"Verification failed for {path}"

if __name__ == '__main__':
    num_proc = multiprocessing.cpu_count()
    splits = ['0', '1']
    ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits)
    ds = DatasetDict(zip(splits, ds))
    ds.map(decompress_example, desc="decompress_example", num_proc=num_proc, load_from_cache_file=False)