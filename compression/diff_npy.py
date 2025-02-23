import numpy as np
import sys
import os

def calculate_difference_percentage(file1, file2):
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    if data1.shape != data2.shape:
        raise ValueError("Files have different shapes")

    difference = np.sum(data1 != data2)
    total_elements = data1.size
    difference_percentage = (difference / total_elements) * 100
    return difference_percentage

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diff_npy.py <file1.npy> <file2.npy>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"One of the files does not exist: {file1}, {file2}")
        sys.exit(1)

    try:
        diff_percentage = calculate_difference_percentage(file1, file2)
        print(f"Difference between {file1} and {file2}: {diff_percentage:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
