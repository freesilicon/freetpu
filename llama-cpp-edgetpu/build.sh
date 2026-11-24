#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
SRC="${LLAMA_CPP_DIR:-$ROOT/upstream/llama.cpp-b12608}"
OB="${OPENBUENOS_DIR:-$REPO_ROOT/libopenbuenos}"
CMAKE="${CMAKE:-$(command -v cmake || echo /opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/cmake)}"
NINJA="${NINJA:-$(command -v ninja || echo /opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/ninja)}"
NDK="${NDK:-/opt/homebrew/share/android-commandlinetools/ndk/26.1.10909125}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

mode="${1:-all}"

overlay_backend() {
    if [[ ! -d "$SRC/ggml" ]]; then
        echo "ERROR: llama.cpp source tree not found at LLAMA_CPP_DIR='$SRC'" >&2
        echo "Clone ggml-org/llama.cpp (tag b12608) first, e.g.:" >&2
        echo "  git clone --depth 1 --branch b12608 https://github.com/ggml-org/llama.cpp \"$SRC\"" >&2
        exit 1
    fi

    mkdir -p "$SRC/ggml/src/ggml-edgetpu"
    cp "$ROOT/ggml-edgetpu.h" "$SRC/ggml/include/ggml-edgetpu.h"
    cp "$ROOT/ggml-edgetpu.cpp" "$ROOT/openbuenos_shim.c" "$ROOT/CMakeLists.txt" "$SRC/ggml/src/ggml-edgetpu/"

    if ! grep -q "GGML_EDGETPU" "$SRC/ggml/CMakeLists.txt"; then
        git -C "$SRC" apply "$ROOT/patches/llama-cpp-b12608-edgetpu.patch" || \
            patch -d "$SRC" -p1 < "$ROOT/patches/llama-cpp-b12608-edgetpu.patch"
    fi
    if ! grep -q "compute_graph_with_accel_fallback" "$SRC/ggml/src/ggml-rpc/ggml-rpc.cpp"; then
        git -C "$SRC" apply "$ROOT/patches/llama-cpp-b12608-rpc-accel-fallback.patch" || \
            patch -d "$SRC" -p1 < "$ROOT/patches/llama-cpp-b12608-rpc-accel-fallback.patch"
    fi
}

build_host() {
    overlay_backend
    echo "=== Building host (build-host) ==="
    "$CMAKE" -S "$SRC" -B "$ROOT/build-host" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_EDGETPU=ON \
        -DGGML_RPC=ON \
        -DGGML_RPC_RDMA=OFF \
        -DGGML_METAL=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_OPENMP=OFF \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_OPENSSL=OFF \
        -DOPENBUENOS_DIR="$OB"
    "$CMAKE" --build "$ROOT/build-host" --target test-backend-ops llama-cli llama-bench llama-completion ggml-rpc-server -j"$JOBS"
}

build_android() {
    overlay_backend
    echo "=== Cross-building Android ARM64 (build-android-arm64) ==="
    "$CMAKE" -S "$SRC" -B "$ROOT/build-android-arm64" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-34 \
        -DANDROID_STL=c++_static \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_EDGETPU=ON \
        -DGGML_RPC=ON \
        -DGGML_RPC_RDMA=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+fp16+i8mm" \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_OPENSSL=OFF \
        -DOPENBUENOS_DIR="$OB"
    "$CMAKE" --build "$ROOT/build-android-arm64" --target test-backend-ops llama-cli llama-bench llama-completion ggml-rpc-server -j"$JOBS"

    echo "=== Stripping & verifying zero-Google-.so Android ARM64 binaries ==="
    local llvm_bin="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin"
    local strip_dir="$ROOT/build-android-arm64/stripped"
    mkdir -p "$strip_dir"
    local bins=(llama-cli llama-bench llama-completion test-backend-ops ggml-rpc-server)
    for b in "${bins[@]}"; do
        "$llvm_bin/llvm-strip" -o "$strip_dir/$b" "$ROOT/build-android-arm64/bin/$b"
        local needed
        needed="$("$llvm_bin/llvm-readelf" -d "$strip_dir/$b" | grep 'NEEDED' || true)"
        echo "--- $b NEEDED ---"
        echo "$needed"
        if echo "$needed" | grep -Eqi 'libedgetpu|libdarwinn|libgxp|liblitert'; then
            echo "ERROR: Forbidden Google TPU library found in $b!" >&2
            exit 1
        fi
    done
    ls -lh "$strip_dir"
}

build_linux_arm64() {
    echo "=== Building libopenbuenos (build/linux-arm64) ==="
    "$CMAKE" -S "$OB" -B "$OB/build/linux-arm64" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    "$CMAKE" --build "$OB/build/linux-arm64" -j"$JOBS"

    overlay_backend
    echo "=== Building Linux ARM64 llama.cpp + ggml-edgetpu (build-linux-arm64) ==="
    "$CMAKE" -S "$SRC" -B "$ROOT/build-linux-arm64" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_EDGETPU=ON \
        -DGGML_RPC=ON \
        -DGGML_RPC_RDMA=OFF \
        -DGGML_OPENMP=ON \
        -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+fp16+i8mm" \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_OPENSSL=OFF \
        -DOPENBUENOS_DIR="$OB"
    "$CMAKE" --build "$ROOT/build-linux-arm64" --target test-backend-ops llama-cli llama-bench llama-completion ggml-rpc-server -j"$JOBS"

    if command -v readelf >/dev/null 2>&1; then
        echo "=== Verifying zero-Google-.so Linux ARM64 binaries ==="
        for b in llama-cli llama-bench llama-completion test-backend-ops ggml-rpc-server; do
            local needed
            needed="$(readelf -d "$ROOT/build-linux-arm64/bin/$b" | grep 'NEEDED' || true)"
            if echo "$needed" | grep -Eqi 'libedgetpu|libdarwinn|libgxp|liblitert'; then
                echo "ERROR: Forbidden Google TPU library found in $b!" >&2
                exit 1
            fi
        done
    fi
}

case "$mode" in
    overlay)     overlay_backend ;;
    host)        build_host ;;
    linux-arm64) build_linux_arm64 ;;
    android)     build_android ;;
    all)         build_host; build_android ;;
    *)           echo "Usage: $0 [overlay|host|linux-arm64|android|all]" >&2; exit 2 ;;
esac
