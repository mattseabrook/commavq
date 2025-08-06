#!/bin/bash
#
# build.sh – commavq CLI utility
#
# Cross-compiles src/*.c → Windows x64 EXE (MSVC runtime)
#

set -e

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PROJECT_ROOT=$(pwd)
TARGET_TRIPLE="x86_64-pc-windows-msvc"
WINSDK_BASE="/opt/winsdk"
BUILD_DIR="build"
EXE_NAME="commavq.exe"

# ------------------------------------------------------------------------------
# Windows-SDK helper
# ------------------------------------------------------------------------------
setup_winsdk() {
    echo "=== Setting up Windows SDK ==="

    if [[ ! -d "$WINSDK_BASE" ]]; then
        echo "Windows SDK not found → installing via xwin..."
        command -v xwin >/dev/null || { echo "Installing xwin..."; cargo install xwin; }
        sudo mkdir -p "$WINSDK_BASE"
        sudo chown "$USER":"$USER" "$WINSDK_BASE"
        xwin --accept-license splat --output "$WINSDK_BASE"
    fi

    # Detect layout produced by xwin
    DETECTED_SDK_INCLUDE="$WINSDK_BASE/sdk/include"
    DETECTED_SDK_LIB="$WINSDK_BASE/sdk/lib"
    DETECTED_CRT_INCLUDE="$WINSDK_BASE/crt/include"
    DETECTED_CRT_LIB="$WINSDK_BASE/crt/lib"
    DETECTED_LIB_ARCH="x86_64"

    # Pick the first SDK version directory we find
    for d in "$DETECTED_SDK_INCLUDE"/*/; do
        [[ -d "$d" ]] && { DETECTED_SDK_VERSION=$(basename "$d"); break; }
    done
    DETECTED_SDK_VERSION="${DETECTED_SDK_VERSION:-10.0.26100}"

    # Sanity check
    for p in "$DETECTED_SDK_INCLUDE/$DETECTED_SDK_VERSION/um" \
             "$DETECTED_SDK_INCLUDE/$DETECTED_SDK_VERSION/ucrt" \
             "$DETECTED_SDK_LIB/um/$DETECTED_LIB_ARCH" \
             "$DETECTED_CRT_LIB/$DETECTED_LIB_ARCH"; do
        [[ -d "$p" ]] || { echo "Missing path: $p"; exit 1; }
    done
    echo "Windows SDK ready."
}

# ------------------------------------------------------------------------------
# Clean build artifacts
# ------------------------------------------------------------------------------
clean() {
    echo "=== Cleaning build artifacts ==="
    rm -rf "$BUILD_DIR" "$EXE_NAME"
    echo "Clean complete."
}

# ------------------------------------------------------------------------------
# Main build process
# ------------------------------------------------------------------------------
build() {
    setup_winsdk

    echo "=== Preparing build directory ==="
    mkdir -p "$BUILD_DIR"

    echo "=== Compiling sources ==="
    OBJECTS=()
    for src in src/*.c; do
        [[ -f "$src" ]] || { echo "No source files found in src/"; exit 1; }
        obj="$BUILD_DIR/$(basename "${src%.c}").obj"
        OBJECTS+=("$obj")
        echo "Compiling $(basename "$src")..."
        echo "  Source: $src"
        echo "  Object: $obj"
        
        echo "  Attempting compilation..."
        set -x  # Enable command tracing
        clang-cl --target="$TARGET_TRIPLE" \
            -std=c2x \
            /EHsc \
            /MT \
            /O2 \
            /DNDEBUG \
            "/imsvc$DETECTED_CRT_INCLUDE" \
            "/imsvc$DETECTED_SDK_INCLUDE/$DETECTED_SDK_VERSION/ucrt" \
            "/imsvc$DETECTED_SDK_INCLUDE/$DETECTED_SDK_VERSION/um" \
            "/imsvc$DETECTED_SDK_INCLUDE/$DETECTED_SDK_VERSION/shared" \
            "/I$PROJECT_ROOT/include" \
            /c \
            "/Fo:$obj" \
            "$src"
        set +x  # Disable command tracing
        
        if [[ ! -f "$obj" ]]; then
            echo "ERROR: Failed to compile $src"
            exit 1
        fi
        echo "  ✓ Compiled successfully"
    done

    echo "Linking $EXE_NAME..."
    echo "Object files to link: ${OBJECTS[*]}"

    # Link flags
    LINK_FLAGS=(
        /subsystem:console
        /defaultlib:libcmt
        /defaultlib:libucrt
        /nodefaultlib:msvcrt.lib
        /nodefaultlib:ucrt.lib
        /libpath:"$DETECTED_SDK_LIB/ucrt/$DETECTED_LIB_ARCH"
        /libpath:"$DETECTED_SDK_LIB/um/$DETECTED_LIB_ARCH"
        /libpath:"$DETECTED_CRT_LIB/$DETECTED_LIB_ARCH"
        /opt:ref
        /opt:icf
        kernel32.lib
        user32.lib
        shell32.lib
        ole32.lib
        uuid.lib
        advapi32.lib
    )

    clang-cl --target="$TARGET_TRIPLE" -fuse-ld=lld-link \
        "${OBJECTS[@]}" -o "$EXE_NAME" /link "${LINK_FLAGS[@]}"

    if [[ $? -eq 0 ]]; then
        echo "✅ Build complete → ./$EXE_NAME"
        if [[ -f "$EXE_NAME" ]]; then
            file_size=$(stat -c%s "$EXE_NAME" 2>/dev/null || stat -f%z "$EXE_NAME" 2>/dev/null || echo "unknown")
            echo "Executable size: ${file_size} bytes"
        fi
    else
        echo "❌ Build failed!"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------------------
case "${1:-release}" in
    clean)
        clean
        ;;
    debug|release)
        build
        ;;
    *)
        echo "Usage: $0 [clean|debug|release]"
        exit 1
        ;;
esac