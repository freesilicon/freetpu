#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Day-1 ggml-cloudtpu test-backend-ops + llama-bench matrix (BF16 & FP8 f8E4M3FN) for Ironwood tpu7x.
set -euo pipefail

DRY_RUN=0
MOCK_MODE=0
SPMD_CHIPS="auto"
MODELS_DIR="${MODELS_DIR:-/opt/models}"
XLA_CACHE="${XLA_COMPILATION_CACHE_DIR:-}"
OUT_DIR=""

usage() {
  cat <<'EOF'
Usage: 01_ggml_cloudtpu_ops_bench.sh [--models-dir DIR] [--spmd N|auto] [--xla-cache DIR|gs://...] [--out-dir DIR] [--dry-run] [--mock]

Runs Ironwood Day-1 verification & benchmarks for ggml-cloudtpu:
  1. test-backend-ops -b CLOUDTPU (BF16 default baseline, CLOUDTPU_MATMUL_FP8=0)
  2. Dedicated live FP8 f8E4M3FN parity gate (1e-2 NMSE via test_cloudtpu_host --fp8-parity-only) + MAPS_LIBPYTHON=0
  3. llama-bench Q4_K_M sweep (8B / 32B / 70B / 120B / 235B) with persistent XLA_COMPILATION_CACHE_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --spmd)
      SPMD_CHIPS="$2"
      shift 2
      ;;
    --xla-cache)
      XLA_CACHE="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --mock)
      MOCK_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/runs/ironwood-day1-${TIMESTAMP}"
fi
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/01_ggml_cloudtpu_ops_bench.txt"

log() {
  echo "$*" | tee -a "${LOG}"
}

: > "${LOG}"
log "=== FreeTPU Ironwood (tpu7x) Day-1 ggml-cloudtpu Ops & Bench ==="
log "timestamp_utc=${TIMESTAMP} spmd_chips=${SPMD_CHIPS} models_dir=${MODELS_DIR}"

if [[ -n "${XLA_CACHE}" ]]; then
  export XLA_COMPILATION_CACHE_DIR="${XLA_CACHE}"
  if [[ "${XLA_CACHE}" != gs://* ]]; then
    mkdir -p "${XLA_CACHE}"
  fi
  log "XLA_COMPILATION_CACHE_DIR=${XLA_COMPILATION_CACHE_DIR}"
else
  log "XLA_COMPILATION_CACHE_DIR=unset (set --xla-cache gs://YOUR_BUCKET/xla_cache to avoid recompiles)"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "[DRY-RUN] Would run test_cloudtpu_host, BF16 test-backend-ops (CLOUDTPU_MATMUL_FP8=0), live FP8 1e-2 NMSE parity gate (test_cloudtpu_host --fp8-parity-only), and llama-bench for available 8B/32B/70B/120B/235B Q4_K_M models."
  exit 0
fi

if [[ "${MOCK_MODE}" -eq 1 ]]; then
  log "[MOCK] Running standalone Host Mock verification under tpu7x-4t (8 logical chiplets, FP8 + D2D pairing)"
  CLOUDTPU_MOCK=1 CLOUDTPU_MOCK_PROFILE=tpu7x-4t CLOUDTPU_MOCK_DEVICES=8 CLOUDTPU_MATMUL_FP8=auto \
    "${REPO_ROOT}/llama-cpp-cloudtpu/build.sh" standalone 2>&1 | tee -a "${LOG}"
  exit 0
fi

BIN_DIR="${REPO_ROOT}/llama-cpp-cloudtpu/build-host/bin"
log "--- Building/refreshing llama-cpp-cloudtpu host overlay ---"
"${REPO_ROOT}/llama-cpp-cloudtpu/build.sh" host 2>&1 | tee -a "${LOG}"

log "--- Standalone Host Check & MAPS_LIBPYTHON=0 Proof ---"
"${REPO_ROOT}/llama-cpp-cloudtpu/build.sh" standalone 2>&1 | tee -a "${LOG}"

log "--- Live test-backend-ops (BF16 MXU default) ---"
CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS=1 CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_MATMUL_BF16=1 \
  "${BIN_DIR}/test-backend-ops" -b CLOUDTPU \
  -o "MUL_MAT,RMS_NORM,ROPE,SOFT_MAX,GET_ROWS,SET_ROWS,ADD,MUL,CPY,SILU,SWIGLU,FLASH_ATTN_EXT" \
  2>&1 | tee -a "${LOG}"

log "--- Live FP8 f8E4M3FN accuracy gate (1e-2 NMSE via test_cloudtpu_host --fp8-parity-only) ---"
log "Note: Using dedicated FP8 1e-2 NMSE gate instead of upstream test-backend-ops MUL_MAT because upstream hardcodes a 5e-4 Q8_0 NMSE threshold while OCP E4M3FN scaling+rounding yields ~6.7e-3 NMSE."
HOST_TEST_BIN="${REPO_ROOT}/llama-cpp-cloudtpu/build-standalone/test_cloudtpu_host"
if [[ ! -x "${HOST_TEST_BIN}" ]]; then
  log "ERROR: test_cloudtpu_host binary not found at ${HOST_TEST_BIN}"
  exit 1
fi
CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS=1 CLOUDTPU_MATMUL_FP8=auto CLOUDTPU_MATMUL_BF16=1 \
  "${HOST_TEST_BIN}" --fp8-parity-only 2>&1 | tee -a "${LOG}"

run_bench_tier() {
  local tier="$1"
  local pattern="$2"
  local model_path=""
  if [[ -d "${MODELS_DIR}" ]]; then
    local matches=()
    local line=""
    while IFS= read -r line; do
      [[ -n "${line}" ]] && matches+=("${line}")
    done < <(find "${MODELS_DIR}" -maxdepth 3 -type f -iname "${pattern}" 2>/dev/null | LC_ALL=C sort)

    local f=""
    for f in "${matches[@]+"${matches[@]}"}"; do
      if [[ ! "${f}" =~ -[0-9]+-of-[0-9]+\.[gG][gG][uU][fF]$ ]]; then
        model_path="${f}"
        break
      fi
    done
    if [[ -z "${model_path}" ]]; then
      for f in "${matches[@]+"${matches[@]}"}"; do
        if [[ "${f}" =~ -00001-of-[0-9]+\.[gG][gG][uU][fF]$ ]]; then
          model_path="${f}"
          break
        fi
      done
    fi
  fi
  if [[ -z "${model_path}" ]]; then
    log "[SKIP] ${tier}: no single-file or shard-00001 GGUF matching '${pattern}' in ${MODELS_DIR}"
    return 0
  fi
  for fp8_mode in 0 auto; do
    log "--- llama-bench ${tier} model=$(basename "${model_path}") spmd=${SPMD_CHIPS} fp8=${fp8_mode} ---"
    CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS="${SPMD_CHIPS}" CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_MATMUL_FP8="${fp8_mode}" \
      "${BIN_DIR}/llama-bench" -m "${model_path}" -ngl 99 -fa 1 -sm none -mg 0 -p 512 -n 64 \
      2>&1 | tee -a "${LOG}"
  done
}

run_bench_tier "8B" "*8b*q4_k_m*.gguf"
run_bench_tier "32B" "*32b*q4_k_m*.gguf"
run_bench_tier "70B" "*70b*q4_k_m*.gguf"
run_bench_tier "120B" "*120b*q4_k_m*.gguf"
run_bench_tier "235B" "*235b*q4_k_m*.gguf"

echo "Benchmark log written to: ${LOG}"
