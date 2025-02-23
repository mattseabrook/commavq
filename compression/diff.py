def load_bytes_from_file(file_path, length=500):
    with open(file_path, 'rb') as f:
        return list(f.read(length))

# Load byte arrays from files
original = load_bytes_from_file('./data_2500_to_5000/fff6e3424ce88d99f1227a971c14a384_13.npy')
compressed = load_bytes_from_file('./compression_challenge_submission/fff6e3424ce88d99f1227a971c14a384_13.npy')
decompressed = load_bytes_from_file('./compression_challenge_submission_decompressed/fff6e3424ce88d99f1227a971c14a384_13.npy')

### Deep Dive Analysis

#### Compare Byte-by-Byte
def compare_data(original, decompressed):
    discrepancies = []
    for i in range(len(original)):
        if original[i] != decompressed[i]:
            discrepancies.append((i, original[i], decompressed[i]))
    return discrepancies

discrepancies = compare_data(original, decompressed)
for i, orig, decomp in discrepancies:
    print(f"Byte {i}: original = {hex(orig)}, decompressed = {hex(decomp)}")
