#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="${LLAMA_CPP_DIR:-$ROOT/upstream/llama.cpp-b12608}"
CMAKE="${CMAKE:-$(command -v cmake || echo /opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/cmake)}"
CTEST="${CTEST:-$(command -v ctest || echo "$(dirname "$CMAKE")/ctest")}"
NINJA="${NINJA:-$(command -v ninja || echo /opt/homebrew/share/android-commandlinetools/cmake/3.22.1/bin/ninja)}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

mode="${1:-host}"

overlay_backend() {
    if [[ ! -d "$SRC/ggml" ]]; then
        echo "ERROR: llama.cpp source tree not found at LLAMA_CPP_DIR='$SRC'" >&2
        echo "Clone ggml-org/llama.cpp (tag b12608) first, e.g.:" >&2
        echo "  git clone --depth 1 --branch b12608 https://github.com/ggml-org/llama.cpp \"$SRC\"" >&2
        exit 1
    fi

    local patch_file="$ROOT/patches/llama-cpp-b12608-cloudtpu.patch"
    local need_apply=0
    if git -C "$SRC" apply --reverse --check "$patch_file" 2>/dev/null; then
        : # Complete overlay patch is already applied.
    elif git -C "$SRC" apply --check "$patch_file" 2>/dev/null; then
        need_apply=1
    else
        echo "ERROR: llama.cpp tree at '$SRC' has a partially applied or conflicting CLOUDTPU overlay." >&2
        echo "Reset '$SRC' to clean tag b12608 before re-running $0." >&2
        exit 1
    fi

    mkdir -p "$SRC/ggml/src/ggml-cloudtpu/third_party/openxla"
    cp "$ROOT/ggml-cloudtpu.h" "$SRC/ggml/include/ggml-cloudtpu.h"
    cp "$ROOT/ggml-cloudtpu.cpp" \
       "$ROOT/pjrt_runtime.hpp" "$ROOT/pjrt_runtime.cpp" \
       "$ROOT/stablehlo_emitter.hpp" "$ROOT/stablehlo_emitter.cpp" \
       "$ROOT/CMakeLists.txt" \
       "$SRC/ggml/src/ggml-cloudtpu/"
    cp "$ROOT/../libopentpu/third_party/openxla/pjrt_c_api.h" "$SRC/ggml/src/ggml-cloudtpu/third_party/openxla/"

    if [[ "$need_apply" -eq 1 ]]; then
        git -C "$SRC" apply "$patch_file"
    fi
}

configure_and_build() {
    local bdir="$1"
    overlay_backend
    local gen_args=()
    if [[ -n "$NINJA" && -x "$NINJA" ]]; then
        gen_args=(-G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA")
    fi
    "$CMAKE" -S "$SRC" -B "$bdir" "${gen_args[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CLOUDTPU=ON \
        -DGGML_METAL=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_OPENMP=OFF \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_OPENSSL=OFF
    "$CMAKE" --build "$bdir" --target test-backend-ops llama-cli llama-bench llama-server -j"$JOBS"
}

build_standalone_tests() {
    local bdir="$ROOT/build-standalone"
    local gen_args=()
    if [[ -n "$NINJA" && -x "$NINJA" ]]; then
        gen_args=(-G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA")
    fi
    "$CMAKE" -S "$ROOT" -B "$bdir" "${gen_args[@]}" -DCMAKE_BUILD_TYPE=Release
    "$CMAKE" --build "$bdir" -j"$JOBS"
    (cd "$bdir" && "$CTEST" --output-on-failure)
}

case "$mode" in
    overlay)    overlay_backend ;;
    standalone) build_standalone_tests ;;
    host)       configure_and_build "$ROOT/build-host" ;;
    tpu)        configure_and_build "$ROOT/build-tpu" ;;
    *)          echo "Usage: $0 [overlay|standalone|host|tpu]" >&2; exit 2 ;;
esac
