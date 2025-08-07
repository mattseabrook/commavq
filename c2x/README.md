`commavq.exe` is a Windows x86_64 executable that compresses and decompresses .token.npy files from the commavq dataset using delta encoding. It is written in C and designed for maximum performance on large datasets, with a focus on minimal overhead and efficient handling of temporal redundancies in tokenized video data.

**Table-of-Contents**
- [Architecture](#architecture)
  - [Token Format](#token-format)
- [Encoding](#encoding)
  - [Header Handling](#header-handling)
  - [Bitpacking](#bitpacking)
  - [Keyframe](#keyframe)
  - [Delta Frames](#delta-frames)
    - [Mask-Based Encoding](#mask-based-encoding)
- [Usage](#usage)
  - [`-c <FILE>`](#-c-file)
  - [`-d <FILE>`](#-d-file)
  - [`-v`](#-v)
- [Visualization](#visualization)
- [Developers](#developers)
  - [Examples](#examples)
  - [Windows SDK on Linux](#windows-sdk-on-linux)

# Architecture

## Token Format

The .token.npy files in the commavq dataset store tokenized representations of driving video frames, compressed using a VQ-VAE model. Each file corresponds to a 1-minute segment at 20 FPS, resulting in 1200 frames. Tokens are 10-bit values (range 0-1023) stored as little-endian int16 (2 bytes per token), with the upper 6 bits always zero.

- **Shape**: (1200, 8, 16) – 1200 frames, each tokenized into an 8x16 grid (128 tokens per frame).
- **File Structure**: 128-byte NumPy header followed by 307,200 data bytes (1200 frames * 256 bytes/frame).
- **Endianness**: Little-endian, matching standard x86/x64 systems.
- **Notes**: The visualization tool (viz.html) confirms the upper 6 bits are padding; tokens are effectively packed in the lower 10 bits.

# Encoding

Temporal redundancy exploits between consecutive frames, where many tokens remain unchanged (visible as blackouts in the visualization). The encoder skips the NumPy header in output files (.cmp), storing only the keyframe and compressed deltas. Decompression reconstructs the full .npy with the static header.

## Header Handling

The 128-byte NumPy header is identical across all files and hardcoded as a static array. It is skipped during compression and prepended during decompression.

## Bitpacking

Before delta encoding, each frame's 128 tokens (256 bytes) are packed by extracting the lower 10 bits and concatenating them into 160 bytes (128 * 10 / 8). This removes the 6-bit padding per token, reducing the frame size by 37.5% upfront.

- **Process**: Loop over int16 tokens, mask & shift lower 10 bits, pack into uint8_t buffer (bit-by-bit accumulation).
- **Notes**: Packing is lossless since upper bits are always zero. Total unpacked data: 307,200 bytes → packed: 192,000 bytes before deltas.

## Keyframe

The first frame (256 bytes) is always stored verbatim as the reference for subsequent deltas.

## Delta Frames

Each of the remaining 1199 frames is encoded as a bitmask (20 bytes, since packed frames are 160 bytes) followed by variable changed bytes. Unchanged positions copy from the previous frame.

### Mask-Based Encoding

- **Mask**: 20-byte bitmask (1 bit per packed byte); set if changed.
- **Changed Bytes**: Only differing bytes are appended after the mask.
- **Notes**: If no changes, mask is all-zero (20 bytes + 0 changes). Encoding scans for differences; decoding copies previous and overwrites changes.

# Usage

## `-c <FILE>`

Compresses (delta encodes) the `.token.npy` file, outputting `<file>.cmp` without the NumPy header.

```cmd
commavq.exe -c example.token.npy
```

## `-d <FILE>`

Decompresses the `.token.npy.cmp` file, reconstructing the full .npy with the static header as `<file>.dec`.

```cmd
commavq.exe -d example.token.npy.cmp
```

## `-v`

Prints the version string.

```cmd
commavq.exe -v
```

# Visualization

`viz.html` is an interactive HTML/JS tool for inspecting `.token.npy` files. It supports drag-and-drop loading and three views:

- **Default**: 64-column hex byte grid with value-based coloring (purple gradient).
- **Token View**: 32-column int16 token display (combines byte pairs).
- **Frame View**: Highlights 4-row frames (256 bytes) with white borders; blackouts unchanged bytes (deltas from previous).

The tool validates file size/extension, skips the header, and uses Uint8Array/DataView for little-endian reads. Spinner handles rendering lag.

# Developers

`build.sh` is a Bash script for compiling the C source on Linux, producing either a Windows x64 EXE (using `clang-cl` for MSVC compatibility) or a Linux x64 ELF executable.

- **Dependencies**: Requires `xwin` for Windows SDK setup (auto-installs if missing for Windows builds). Uses `clang-cl`/`lld-link` for Windows, native `clang`/`ld` for Linux.
- **Modes**: windows (cross-compile to commavq.exe), linux (native compile to commavq), clean (remove artifacts).
- **Steps**: Detects SDK paths (for Windows), compiles .c files in src/ to .obj/o, links with platform libs. No external libraries; bare-metal C. Outputs ~10-20KB executable.
- **Notes**: Wildcard support for src/*.c and include/; automatic platform config. For Windows: Uses MSVC runtime libs (libcmt, libucrt). For Linux: Optimized with -O3, -march=native.

## Examples 

```bash
./build.sh windows   # Build Windows version
./build.sh linux     # Build Linux version
./build.sh clean     # Clean all builds
```

## Windows SDK on Linux

```bash
# Windows SDK
yay -S xwin
sudo xwin --accept-license splat --output /opt/winsdk
```