#!/bin/bash
#
# build.sh – commavq CLI utility
#
# Cross-compiles src/*.c → Windows x64 EXE (MSVC runtime) OR Linux x64 ELF
#

set -e

# Show banner
echo -e "\033[1;97m                         _____ _____ 
 ___ ___ _____ _____ ___|  |  |     |
|  _| . |     |     | .'|  |  |  |  |
|___|___|_|_|_|_|_|_|__,|\___/|__  _|
                                 |__|
\e[0m"
echo 'commaVQ VQ-VAE Experimental post-processor'
echo 'v.0.1 - 2025/08/07 - matt@lakeshoretechnical.com'
echo

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PROJECT_ROOT=$(pwd)
BUILD_DIR="build"

# Platform-specific configuration
setup_platform_config() {
    case "$1" in
        windows)
            TARGET_TRIPLE="x86_64-pc-windows-msvc"
            WINSDK_BASE="/opt/winsdk"
            EXE_NAME="commavq.exe"
            ;;
        linux)
            TARGET_TRIPLE="x86_64-unknown-linux-gnu"
            EXE_NAME="commavq"
            ;;
        *)
            echo "ERROR: Unknown platform '$1'"
            echo "Supported platforms: windows, linux"
            exit 1
            ;;
    esac
}

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
    echo "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR" "commavq.exe" "commavq" *.token.npy.cmp *.token.npy.cmp.dec
    echo "  ✓ Cleaned successfully"
}

# ------------------------------------------------------------------------------
# Linux build process
# ------------------------------------------------------------------------------
build_linux() {
    echo "=== Building for Linux x64 ===
    "

    echo "Preparing build directory..."
    mkdir -p "$BUILD_DIR"

    echo "Compiling sources..."
    OBJECTS=()
    for src in src/*.c; do
        [[ -f "$src" ]] || { echo "No source files found in src/"; exit 1; }
        obj="$BUILD_DIR/$(basename "${src%.c}").o"
        OBJECTS+=("$obj")
        echo "$(basename "$src")..."
        echo "  Source: $src"
        echo "  Object: $obj"
        
        echo "Attempting compilation..."
        set -x  # Enable command tracing
        clang \
            -std=c2x \
            -O3 \
            -march=native \
            -mtune=native \
            -flto \
            -ffast-math \
            -funroll-loops \
            -fomit-frame-pointer \
            -DNDEBUG \
            -Wall \
            -Wextra \
            -Wpedantic \
            -I"$PROJECT_ROOT/include" \
            -c \
            -o "$obj" \
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

    set -x  # Enable command tracing
    clang \
        -std=c2x \
        -O3 \
        -march=native \
        -mtune=native \
        -flto \
        -ffast-math \
        -funroll-loops \
        -fomit-frame-pointer \
        -DNDEBUG \
        -s \
        "${OBJECTS[@]}" \
        -o "$EXE_NAME" \
        -lm \
        -lpthread \
        -lz
    set +x  # Disable command tracing

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
# Windows build process  
# ------------------------------------------------------------------------------
build_windows() {
    echo "=== Building for Windows x64 ==="
    
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
        synchronization.lib
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

# ------------------------------------------------------------------------------s
# Script entry point
# ------------------------------------------------------------------------------
case "${1:-}" in
    clean)
        clean
        ;;
    windows)
        setup_platform_config windows
        build_windows
        ;;
    linux)
        setup_platform_config linux
        build_linux
        ;;
    debug|release)
        # Backwards compatibility - default to Windows build
        echo "⚠️  Warning: 'debug/release' is deprecated. Use 'windows' or 'linux' instead."
        setup_platform_config windows
        build_windows
        ;;
    *)
        echo "Usage: $0 [clean|windows|linux]"
        echo ""
        echo "Build targets:"
        echo "  windows  - Cross-compile for Windows x64 (MSVC runtime) → commavq.exe"
        echo "  linux    - Build for Linux x64 with intense optimizations → commavq"
        echo "  clean    - Remove all build artifacts"
        echo ""
        echo "Examples:"
        echo "  $0 windows   # Build Windows version"
        echo "  $0 linux     # Build Linux version" 
        echo "  $0 clean     # Clean all builds"
        exit 1
        ;;
esac