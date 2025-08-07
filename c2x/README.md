`commavq.exe` is a Windows x86_64 executable that compresses and decompresses .token.npy files from the commavq dataset using delta encoding. It is written in C and designed for maximum performance on large datasets, with a focus on minimal overhead and efficient handling of temporal redundancies in tokenized video data.

**Table-of-Contents**
- [Architecture](#architecture)
  - [Token Format](#token-format)
- [Encoding](#encoding)
  - [Header Handling](#header-handling)
  - [Keyframe](#keyframe)
  - [Delta Frames](#delta-frames)
    - [Mask-Based Encoding](#mask-based-encoding)
- [Usage](#usage)
  - [`-c <FILE>`](#-c-file)
  - [`-d <FILE>`](#-d-file)
  - [`-v`](#-v)
- [Visualization](#visualization)
- [Developers](#developers)
  - [Build Process](#build-process)
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

## Build Process

`build.sh` is a Bash script for cross-compiling the C source to a Windows x64 EXE on Linux, using `clang-cl` for MSVC compatibility.

**Dependencies**: Requires `xwin` for Windows SDK setup (auto-installs if missing).
- **Modes**: release (default, optimized), debug (symbols), clean (remove artifacts).
- **Steps**: Detects SDK paths, compiles .c files in src/ to .obj, links with MSVC libs.
- **Notes**: No external libraries; bare-metal C. Outputs commavq.exe (~10-20KB).

## Windows SDK on Linux

```bash
# Windows SDK
yay -S xwin
sudo xwin --accept-license splat --output /opt/winsdk
```