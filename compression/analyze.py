import numpy as np
from pathlib import Path


def analyze_within_file(file_path, log_file):
    """Analyze chunks within a single .npy file."""
    data = np.load(file_path)
    if data.shape != (1200, 8, 16) or data.dtype != np.int16:
        msg = f"Unexpected shape or dtype in {file_path.name}\n"
        log_file.write(msg)
        print(msg)
        return

    # Check 10-bit range (0-1023)
    if np.any(data < 0) or np.any(data > 1023):
        msg = f"Values out of 10-bit range in {file_path.name}\n"
    else:
        msg = f"All values within 0-1023 in {file_path.name}\n"
    log_file.write(msg)
    print(msg)

    # Display binary representation of the first 5 16-bit values from the first frame
    sample_values = data[
        0, 0, :5
    ]  # First 5 elements of the first row of the first frame
    binary_values = [f"{int(val):016b}" for val in sample_values]
    msg = f"First 5 binary values: {binary_values}\n"
    log_file.write(msg)
    print(msg)

    # Compute frame-to-frame deltas (keeping this as is per your original script)
    deltas = np.diff(data, axis=0)  # Shape: (1199, 8, 16)
    mean_delta = np.mean(np.abs(deltas))
    msg = f"Mean absolute delta between frames: {mean_delta:.2f}\n"
    log_file.write(msg)
    print(msg)


def analyze_across_files(files, log_file):
    """Analyze redundancy across files by comparing corresponding chunks."""
    # Load first frame of each file
    first_frames = [np.load(f)[0] for f in files]  # Shape: (10, 8, 16)
    msg = "\nCross-file analysis (comparing frame 0 across files):\n"
    log_file.write(msg)
    print(msg)

    # Pairwise comparison
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            diff = np.mean(np.abs(first_frames[i] - first_frames[j]))
            msg = f"Mean abs diff between {files[i].name} and {files[j].name}: {diff:.2f}\n"
            log_file.write(msg)
            print(msg)


def main():
    test_dir = Path("test")
    log_path = test_dir / "log.txt"

    # Get list of 10 .npy files
    files = sorted(test_dir.glob("*.npy"))[:10]
    if len(files) != 10:
        print(f"Expected 10 files, found {len(files)}")
        return

    with open(log_path, "w") as log_file:
        # Phase 1: Analyze each file individually
        for file in files:
            msg = f"\nAnalyzing {file.name}:\n"
            log_file.write(msg)
            print(msg)
            analyze_within_file(file, log_file)

        # Phase 2: Analyze across files
        analyze_across_files(files, log_file)


if __name__ == "__main__":
    main()
