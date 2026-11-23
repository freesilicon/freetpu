#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Day-1 ggml-cloudtpu test-backend-ops + FreeTile Mosaic + llama-bench matrix (BF16 & FP8 f8E4M3FN) for Ironwood tpu7x.
set -euo pipefail

DRY_RUN=0
MOCK_MODE=0
RUN_MOSAIC=1
SKIP_GCS_SYNC=0
INCLUDE_671B=0
SPMD_CHIPS="auto"
MODELS_DIR="${MODELS_DIR:-/opt/models}"
GCS_MODEL_BUCKET="${GCS_MODEL_BUCKET:-}"
TPU_ZONE="${TPU_ZONE:-}"
XLA_CACHE="${XLA_COMPILATION_CACHE_DIR:-}"
PCI_SYSFS_DIR="${PCI_SYSFS_DIR:-/sys/bus/pci/devices}"
STAGED_RUNS_ROOT="${STAGED_RUNS_ROOT:-}"
OUT_DIR=""

usage() {
  cat <<'EOF'
Usage: 01_ggml_cloudtpu_ops_bench.sh [--models-dir DIR] [--gcs-model-bucket gs://...] [--tpu-zone ZONE]
                                     [--skip-gcs-sync] [--spmd N|auto] [--xla-cache DIR|gs://...]
                                     [--no-mosaic] [--include-671b] [--out-dir DIR] [--dry-run] [--mock]

Runs Ironwood Day-1 verification & benchmarks for ggml-cloudtpu:
  0. Auto-discovers staged libtpu.so from Step 00 (runs/ironwood-day1*/) when LIBTPU_PATH is unset,
     optional same-region GCS model staging (when GCS_MODEL_BUCKET / --gcs-model-bucket is set), &
     persistent XLA_COMPILATION_CACHE_DIR setup (gs://... or local fallback)
  1. test-backend-ops -b CLOUDTPU (BF16 pure StableHLO baseline, CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_FREETILE_MOSAIC=0)
  2. Dedicated live FP8 f8E4M3FN parity gate (1e-2 NMSE via test_cloudtpu_host --fp8-parity-only) + MAPS_LIBPYTHON=0
  3. FreeTile Mosaic Q4_K/Q6_K pass (CLOUDTPU_FREETILE_MOSAIC=1) verifying actual Mosaic kernel dispatch
     (reports MOSAIC_COMPARISON_STATUS=FALLBACK_ONLY if Mosaic falls back to pure StableHLO)
  4. llama-bench sweep across named 1t bracket tiers (8B, Qwen3-32B, Llama-3.3-70B, gpt-oss-120b, Qwen3-235B)
     plus DeepSeek-V3/R1 671B (FP8/Q4) hero tpu7x-standard-4t tier (gated on >=8 chiplets / 768 GiB HBM or --include-671b)
EOF
}

require_arg() {
  if [[ "$1" -lt 2 ]]; then
    echo "Option $2 requires an argument" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir)
      require_arg "$#" "$1"
      MODELS_DIR="$2"
      shift 2
      ;;
    --gcs-model-bucket)
      require_arg "$#" "$1"
      GCS_MODEL_BUCKET="$2"
      shift 2
      ;;
    --tpu-zone)
      require_arg "$#" "$1"
      TPU_ZONE="$2"
      shift 2
      ;;
    --skip-gcs-sync)
      SKIP_GCS_SYNC=1
      shift
      ;;
    --spmd)
      require_arg "$#" "$1"
      SPMD_CHIPS="$2"
      shift 2
      ;;
    --xla-cache)
      require_arg "$#" "$1"
      XLA_CACHE="$2"
      shift 2
      ;;
    --no-mosaic)
      RUN_MOSAIC=0
      shift
      ;;
    --include-671b)
      INCLUDE_671B=1
      shift
      ;;
    --out-dir)
      require_arg "$#" "$1"
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
if [[ -z "${STAGED_RUNS_ROOT}" ]]; then
  STAGED_RUNS_ROOT="${REPO_ROOT}/runs"
fi
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/runs/ironwood-day1-${TIMESTAMP}"
fi
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/01_ggml_cloudtpu_ops_bench.txt"

log() {
  echo "$*" | tee -a "${LOG}"
}

resolve_staged_libtpu() {
  if [[ -n "${LIBTPU_PATH:-}" ]]; then
    log "[LIBTPU] Using caller-provided LIBTPU_PATH=${LIBTPU_PATH}"
    return 0
  fi
  if [[ -n "${TPU_LIBRARY_PATH:-}" ]]; then
    export LIBTPU_PATH="${TPU_LIBRARY_PATH}"
    log "[LIBTPU] Using caller-provided TPU_LIBRARY_PATH=${LIBTPU_PATH}"
    return 0
  fi
  local search_dirs=()
  if [[ -d "${OUT_DIR}" ]]; then
    search_dirs+=("${OUT_DIR}")
  fi
  local d=""
  for d in "${STAGED_RUNS_ROOT}/ironwood-day1" "${STAGED_RUNS_ROOT}"/ironwood-day1-*; do
    if [[ -d "${d}" && "${d}" != "${OUT_DIR}" ]]; then
      search_dirs+=("${d}")
    fi
  done
  if [[ "${#search_dirs[@]}" -eq 0 ]]; then
    return 0
  fi
  local latest_so=""
  if command -v python3 >/dev/null 2>&1; then
    latest_so="$(python3 - "${search_dirs[@]}" <<'PYEOF'
import os
import sys
from pathlib import Path

best_path = ""
best_mtime = -1.0
for raw_dir in sys.argv[1:]:
    p = Path(raw_dir)
    if not p.is_dir():
        continue
    for entry in p.glob("libtpu*.so"):
        try:
            st = entry.stat()
        except OSError:
            continue
        if st.st_size > 0 and st.st_mtime >= best_mtime:
            best_mtime = st.st_mtime
            best_path = str(entry)
print(best_path)
PYEOF
)"
  else
    latest_so="$(ls -t "${search_dirs[@]/%//libtpu*.so}" 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -n "${latest_so}" && -s "${latest_so}" ]]; then
    export LIBTPU_PATH="${latest_so}"
    log "[LIBTPU] Auto-discovered staged libtpu.so from Step 00: LIBTPU_PATH=${LIBTPU_PATH}"
  else
    log "[LIBTPU] LIBTPU_PATH unset and no staged libtpu*.so found under ${OUT_DIR} or ${STAGED_RUNS_ROOT}/ironwood-day1* (will rely on system/Python libtpu.so)"
  fi
}

resolve_effective_chips() {
  if [[ "${SPMD_CHIPS}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${SPMD_CHIPS}"
    return 0
  fi
  local survey_file="${OUT_DIR}/00_pci_vfio_survey.txt"
  if [[ -r "${survey_file}" ]]; then
    local pjrt_n=""
    pjrt_n="$(sed -n 's/^PJRT_LOGICAL_DEVICES=\([0-9][0-9]*\).*/\1/p' "${survey_file}" | tail -n 1)"
    if [[ "${pjrt_n}" =~ ^[0-9]+$ && "${pjrt_n}" -gt 0 ]]; then
      printf '%s' "${pjrt_n}"
      return 0
    fi
  fi
  local chips=0
  if [[ -d "${PCI_SYSFS_DIR}" ]]; then
    local dev_path="" vid="" did="" cls=""
    for dev_path in "${PCI_SYSFS_DIR}"/*; do
      [[ -e "${dev_path}/vendor" ]] || continue
      vid="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/vendor" 2>/dev/null || echo unknown)"
      [[ "${vid}" == "0x1ae0" ]] || continue
      cls="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/class" 2>/dev/null || echo unknown)"
      case "${cls}" in
        0x02*) continue ;;
        0xff*|0x12*|0x0b*|0x08*) ;;
        *) continue ;;
      esac
      did="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/device" 2>/dev/null || echo unknown)"
      case "${did}" in
        0x0075|0x0076) chips=$((chips + 2)) ;;
        *) chips=$((chips + 1)) ;;
      esac
    done
  fi
  printf '%s' "${chips}"
}

evaluate_live_mosaic_dispatch() {
  local label="$1"
  local run_output="$2"
  local summary_counts=""
  summary_counts="$(printf '%s\n' "${run_output}" | awk '
    /ggml-cloudtpu: summary/ {
      if (match($0, /mosaic_compiles=[0-9]+/)) {
        mc += substr($0, RSTART + 16, RLENGTH - 16) + 0
        saw = 1
      }
      if (match($0, /mosaic_fallbacks=[0-9]+/)) {
        mf += substr($0, RSTART + 17, RLENGTH - 17) + 0
        saw = 1
      }
    }
    END { printf "%d %d %d\n", saw + 0, mc + 0, mf + 0 }
  ')"
  local saw=0 mc=0 mf=0
  read -r saw mc mf <<< "${summary_counts}"
  if [[ "${saw}" -eq 1 && "${mc}" -gt 0 ]]; then
    log "MOSAIC_COMPARISON_STATUS=PASS (label=${label} mosaic_compiles=${mc} mosaic_fallbacks=${mf})"
  elif [[ "${mf}" -gt 0 ]] || printf '%s\n' "${run_output}" | grep -Fq "FreeTile Mosaic compile failed"; then
    log "WARN: FreeTile Mosaic fell back to pure StableHLO without dispatching a Mosaic kernel (label=${label} mosaic_compiles=${mc} mosaic_fallbacks=${mf})"
    log "MOSAIC_COMPARISON_STATUS=FALLBACK_ONLY (label=${label} mosaic_compiles=${mc} mosaic_fallbacks=${mf})"
  else
    log "WARN: No FreeTile Mosaic kernel dispatch observed in ggml-cloudtpu summary (label=${label} summary_seen=${saw} mosaic_compiles=${mc} mosaic_fallbacks=${mf})"
    log "MOSAIC_COMPARISON_STATUS=FALLBACK_ONLY (label=${label} mosaic_compiles=0 mosaic_fallbacks=${mf})"
  fi
}

sync_gcs_models() {
  if [[ -z "${GCS_MODEL_BUCKET}" ]]; then
    log "[GCS] GCS_MODEL_BUCKET=unset — skipping GCS model staging (using local MODELS_DIR=${MODELS_DIR})"
    return 0
  fi
  if [[ "${SKIP_GCS_SYNC}" -eq 1 ]]; then
    log "[GCS] --skip-gcs-sync set — keeping GCS_MODEL_BUCKET=${GCS_MODEL_BUCKET} for cache defaults only"
    return 0
  fi
  if [[ "${DRY_RUN}" -eq 1 || "${MOCK_MODE}" -eq 1 ]]; then
    log "[GCS] Would sync same-region models (tpu_zone=${TPU_ZONE:-auto}): gcloud storage rsync --recursive --exclude='^xla-cache/.*' ${GCS_MODEL_BUCKET} ${MODELS_DIR}"
    return 0
  fi
  mkdir -p "${MODELS_DIR}"
  log "--- Staging models from same-region GCS (${GCS_MODEL_BUCKET}, tpu_zone=${TPU_ZONE:-auto}) -> ${MODELS_DIR} ---"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync --recursive --exclude='^xla-cache/.*' "${GCS_MODEL_BUCKET}" "${MODELS_DIR}" 2>&1 | tee -a "${LOG}"
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r -x '^xla-cache/.*' "${GCS_MODEL_BUCKET}" "${MODELS_DIR}" 2>&1 | tee -a "${LOG}"
  else
    log "ERROR: GCS_MODEL_BUCKET=${GCS_MODEL_BUCKET} is set, but neither gcloud nor gsutil is installed."
    return 1
  fi
}

: > "${LOG}"
log "=== FreeTPU Ironwood (tpu7x) Day-1 ggml-cloudtpu Ops & Bench ==="
log "timestamp_utc=${TIMESTAMP} spmd_chips=${SPMD_CHIPS} models_dir=${MODELS_DIR} tpu_zone=${TPU_ZONE:-unset} gcs_model_bucket=${GCS_MODEL_BUCKET:-unset} run_mosaic=${RUN_MOSAIC} include_671b=${INCLUDE_671B}"

resolve_staged_libtpu

if [[ -z "${XLA_CACHE}" ]]; then
  if [[ -n "${GCS_MODEL_BUCKET}" ]]; then
    gcs_bucket_root="$(printf '%s' "${GCS_MODEL_BUCKET}" | sed -E 's#^(gs://[^/]+).*#\1#')"
    XLA_CACHE="${gcs_bucket_root}/xla-cache/tpu7x"
  else
    XLA_CACHE="${OUT_DIR}/xla_cache"
  fi
fi

export XLA_COMPILATION_CACHE_DIR="${XLA_CACHE}"
if [[ "${XLA_CACHE}" != gs://* ]]; then
  mkdir -p "${XLA_CACHE}"
fi
log "XLA_COMPILATION_CACHE_DIR=${XLA_COMPILATION_CACHE_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  sync_gcs_models
  log "[DRY-RUN] Would run test_cloudtpu_host, BF16 pure StableHLO test-backend-ops (CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_FREETILE_MOSAIC=0), live FP8 1e-2 NMSE parity gate (test_cloudtpu_host --fp8-parity-only), FreeTile Mosaic Q4_K/Q6_K pass (CLOUDTPU_FREETILE_MOSAIC=1 CLOUDTPU_VERBOSE=1 verifying mosaic_compiles>0), and llama-bench for named 1t bracket tiers (8B / Qwen3-32B / Llama-3.3-70B / gpt-oss-120b / Qwen3-235B) + DeepSeek-V3/R1 671B FP8/Q4 (when effective_chips>=8 or --include-671b)."
  exit 0
fi

if [[ "${MOCK_MODE}" -eq 1 ]]; then
  sync_gcs_models
  log "[MOCK] Pass 1/2: Pure StableHLO Host Mock verification under tpu7x-4t (8 logical chiplets, FP8 + D2D pairing, CLOUDTPU_FREETILE_MOSAIC=0)"
  CLOUDTPU_MOCK=1 CLOUDTPU_MOCK_PROFILE=tpu7x-4t CLOUDTPU_MOCK_DEVICES=8 CLOUDTPU_MATMUL_FP8=auto CLOUDTPU_FREETILE_MOSAIC=0 \
    "${REPO_ROOT}/llama-cpp-cloudtpu/build.sh" standalone 2>&1 | tee -a "${LOG}"
  if [[ "${RUN_MOSAIC}" -eq 1 ]]; then
    HOST_TEST_BIN="${REPO_ROOT}/llama-cpp-cloudtpu/build-standalone/test_cloudtpu_host"
    if [[ ! -x "${HOST_TEST_BIN}" ]]; then
      log "ERROR: test_cloudtpu_host binary not found at ${HOST_TEST_BIN}"
      exit 1
    fi
    log "[MOCK] Pass 2/2: FreeTile Mosaic Q4_K/Q6_K Host Mock comparison (CLOUDTPU_FREETILE_MOSAIC=1 vs pure StableHLO baseline)"
    mosaic_mock_out="$(CLOUDTPU_MOCK=1 CLOUDTPU_MOCK_PROFILE=tpu7x-4t CLOUDTPU_MOCK_DEVICES=8 CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_FREETILE_MOSAIC=1 CLOUDTPU_VERBOSE=1 "${HOST_TEST_BIN}" 2>&1)"
    printf '%s\n' "${mosaic_mock_out}" | tee -a "${LOG}"
    if ! printf '%s\n' "${mosaic_mock_out}" | grep -Fq "[PASS] FreeTile Mosaic lowering (Q4_K_I4 S4/U4/F32-scales, Q4_0_I4 U4, Q6_K_SPLIT, fallback & dialect guard)"; then
      log "ERROR: Mock FreeTile Mosaic run did not report expected FreeTile Mosaic Q4_K/Q6_K dispatch proof marker."
      log "MOSAIC_COMPARISON_STATUS=FALLBACK_ONLY (mock: missing FreeTile Mosaic dispatch marker)"
      exit 1
    fi
    log "MOSAIC_COMPARISON_STATUS=PASS (mock: pure_stablehlo=PASS freetile_mosaic_q4k_q6k_dispatched=PASS)"
  fi
  exit 0
fi

sync_gcs_models

BIN_DIR="${REPO_ROOT}/llama-cpp-cloudtpu/build-host/bin"
log "--- Building/refreshing llama-cpp-cloudtpu host overlay ---"
"${REPO_ROOT}/llama-cpp-cloudtpu/build.sh" host 2>&1 | tee -a "${LOG}"

log "--- Standalone Host Check & MAPS_LIBPYTHON=0 Proof ---"
"${REPO_ROOT}/llama-cpp-cloudtpu/build.sh" standalone 2>&1 | tee -a "${LOG}"

log "--- Live test-backend-ops (BF16 MXU pure StableHLO baseline, CLOUDTPU_FREETILE_MOSAIC=0) ---"
CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS=1 CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_FREETILE_MOSAIC=0 \
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
CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS=1 CLOUDTPU_MATMUL_FP8=auto CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_FREETILE_MOSAIC=0 \
  "${HOST_TEST_BIN}" --fp8-parity-only 2>&1 | tee -a "${LOG}"

if [[ "${RUN_MOSAIC}" -eq 1 ]]; then
  log "--- Live FreeTile Mosaic Q4_K/Q6_K pass (CLOUDTPU_FREETILE_MOSAIC=1 CLOUDTPU_VERBOSE=1 vs pure StableHLO) ---"
  mosaic_ops_out="$(CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS=1 CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_FREETILE_MOSAIC=1 CLOUDTPU_VERBOSE=1 \
    "${BIN_DIR}/test-backend-ops" -b CLOUDTPU -o "MUL_MAT" 2>&1)"
  printf '%s\n' "${mosaic_ops_out}" | tee -a "${LOG}"
  evaluate_live_mosaic_dispatch "test-backend-ops:MUL_MAT" "${mosaic_ops_out}"
fi

find_model_gguf() {
  local model_regex="$1"
  local size_regex="$2"
  local quant_regex="${3:-q4_k_m}"
  local exclude_regex="${4:-}"
  local model_path=""
  if [[ -d "${MODELS_DIR}" ]]; then
    local matches=()
    local line=""
    local base_lc=""
    while IFS= read -r line; do
      [[ -n "${line}" ]] || continue
      base_lc="$(basename "${line}" | tr '[:upper:]' '[:lower:]')"
      if [[ -n "${exclude_regex}" ]] && printf '%s\n' "${base_lc}" | grep -Eq "${exclude_regex}"; then
        continue
      fi
      if [[ -n "${model_regex}" ]] && ! printf '%s\n' "${base_lc}" | grep -Eq "${model_regex}"; then
        continue
      fi
      if printf '%s\n' "${base_lc}" | grep -Eq "(^|[^0-9a-z.])${size_regex}([^0-9a-z]|$)" &&
         printf '%s\n' "${base_lc}" | grep -Eq "${quant_regex}"; then
        matches+=("${line}")
      fi
    done < <(find -L "${MODELS_DIR}" -maxdepth 3 -type f -iname "*.gguf" 2>/dev/null | LC_ALL=C sort)

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
  printf '%s' "${model_path}"
}

run_bench_tier() {
  local tier="$1"
  local model_regex="$2"
  local size_regex="$3"
  local quant_regex="${4:-q4_k_m}"
  local exclude_regex="${5:-}"
  local model_path=""
  model_path="$(find_model_gguf "${model_regex}" "${size_regex}" "${quant_regex}" "${exclude_regex}")"
  if [[ -z "${model_path}" ]]; then
    log "[SKIP] ${tier}: no single-file or shard-00001 GGUF matching model='${model_regex:-any}' size='${size_regex}' quant='${quant_regex}' in ${MODELS_DIR}"
    return 0
  fi
  local ckpt_name=""
  ckpt_name="$(basename "${model_path}")"
  log "[MATCH] ${tier} -> checkpoint=${ckpt_name} path=${model_path}"

  log "--- llama-bench ${tier} checkpoint=${ckpt_name} spmd=${SPMD_CHIPS} fp8=0 mosaic=0 (pure StableHLO BF16) ---"
  CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS="${SPMD_CHIPS}" CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_FREETILE_MOSAIC=0 \
    "${BIN_DIR}/llama-bench" -m "${model_path}" -ngl 99 -fa 1 -sm none -mg 0 -p 512 -n 64 \
    2>&1 | tee -a "${LOG}"

  if [[ "${RUN_MOSAIC}" -eq 1 ]]; then
    log "--- llama-bench ${tier} checkpoint=${ckpt_name} spmd=${SPMD_CHIPS} fp8=0 mosaic=1 (FreeTile Mosaic Q4_K/Q6_K) ---"
    local mosaic_bench_out=""
    mosaic_bench_out="$(CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS="${SPMD_CHIPS}" CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_MATMUL_FP8=0 CLOUDTPU_FREETILE_MOSAIC=1 CLOUDTPU_VERBOSE=1 \
      "${BIN_DIR}/llama-bench" -m "${model_path}" -ngl 99 -fa 1 -sm none -mg 0 -p 512 -n 64 2>&1)"
    printf '%s\n' "${mosaic_bench_out}" | tee -a "${LOG}"
    evaluate_live_mosaic_dispatch "llama-bench:${tier}:${ckpt_name}" "${mosaic_bench_out}"
  fi

  log "--- llama-bench ${tier} checkpoint=${ckpt_name} spmd=${SPMD_CHIPS} fp8=auto mosaic=0 (FP8 f8E4M3FN) ---"
  CLOUDTPU_MOCK=0 CLOUDTPU_SPMD_CHIPS="${SPMD_CHIPS}" CLOUDTPU_MATMUL_BF16=1 CLOUDTPU_MATMUL_FP8=auto CLOUDTPU_FREETILE_MOSAIC=0 \
    "${BIN_DIR}/llama-bench" -m "${model_path}" -ngl 99 -fa 1 -sm none -mg 0 -p 512 -n 64 \
    2>&1 | tee -a "${LOG}"
}

should_run_671b_tier() {
  local eff_chips="$1"
  if [[ "${INCLUDE_671B}" -eq 1 || "${eff_chips}" -ge 8 ]]; then
    return 0
  fi
  return 1
}

# 1t bracket tiers (fit within 192 GiB single-package bracket, also sharded across 8 chiplets on 4t)
run_bench_tier "8B (1t bracket)" "" "8b" "q4_k_m" "distill"
run_bench_tier "Qwen3-32B (1t bracket)" "qwen[-_.]?3" "32b" "q4_k_m" "distill"
run_bench_tier "Llama-3.3-70B (1t bracket)" "llama" "70b" "q4_k_m" "distill"
run_bench_tier "gpt-oss-120b (1t bracket)" "gpt[-_.]?oss" "120b" "q4_k_m"
run_bench_tier "Qwen3-235B (1t bracket)" "qwen[-_.]?3" "235b" "q4_k_m" "distill"

# Hero 4t tier (tpu7x-standard-4t: 768 GiB across 8 logical chiplets)
EFFECTIVE_CHIPS="$(resolve_effective_chips)"
if should_run_671b_tier "${EFFECTIVE_CHIPS}"; then
  run_bench_tier "DeepSeek-V3/R1-671B (4t hero 768GiB)" "deepseek[-_.]?(v3|r1)" "(671b|deepseek[-_.]?(v3|r1))" "(q4_k_m|q4_k|q4_0|iq4|fp8|f8e4m3)" "(distill|1\.5b|7b|8b|14b|32b|70b)"
else
  log "SKIP: 671B tier requires >=8 Ironwood chiplets (768 GiB HBM) or --include-671b (effective_chips=${EFFECTIVE_CHIPS}, spmd=${SPMD_CHIPS})"
fi

echo "Benchmark log written to: ${LOG}"
