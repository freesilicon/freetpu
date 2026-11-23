#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Safe libopentpu /dev/vfio/* probe & MAPS_LIBTPU=0 capture for Ironwood Day-1 qualification.
set -euo pipefail

DRY_RUN=0
MOCK_MODE=0
LIVE_VFIO=0
SKIP_BUILD="${SKIP_BUILD:-0}"
PCI_SYSFS_DIR="${PCI_SYSFS_DIR:-/sys/bus/pci/devices}"
OUT_DIR=""

usage() {
  cat <<'EOF'
Usage: 02_libopentpu_vfio_probe.sh [--out-dir DIR] [--live-vfio] [--dry-run] [--mock]

Verifies libopentpu clean-room build, multi-gen HAL geometry, and MAPS_LIBTPU=0 / MAPS_LIBPYTHON=0 capture.
Live /dev/vfio/* device probing requires explicit --live-vfio on Linux and only accepts either:
  - PASS with MAPS_LIBTPU=0 and MAPS_LIBPYTHON=0 verified, or
  - Explicit pre-reset refusal ("Refusing to reset/program unsupported VFIO PCI device")
    with MAPS_LIBTPU=0 and MAPS_LIBPYTHON=0 verified.
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
    --out-dir)
      require_arg "$#" "$1"
      OUT_DIR="$2"
      shift 2
      ;;
    --live-vfio)
      LIVE_VFIO=1
      shift
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
LOG="${OUT_DIR}/02_libopentpu_vfio_probe.txt"

log() {
  echo "$*" | tee -a "${LOG}"
}

assert_cleanroom_maps() {
  local output="$1"
  local allow_non_linux_skip="${2:-0}"
  if printf '%s\n' "${output}" | grep -Eq '(^|[[:space:]])MAPS_LIBTPU=0([[:space:]]|$)' &&
     printf '%s\n' "${output}" | grep -Eq '(^|[[:space:]])MAPS_LIBPYTHON=0([[:space:]]|$)'; then
    log "CLEANROOM_MAPS_STATUS=PASS (MAPS_LIBTPU=0 MAPS_LIBPYTHON=0)"
    return 0
  fi
  if [[ "${allow_non_linux_skip}" -eq 1 && "$(uname -s)" != "Linux" ]] &&
     printf '%s\n' "${output}" | grep -Fq "MAPS_CHECK=SKIPPED"; then
    log "CLEANROOM_MAPS_STATUS=SKIPPED_NON_LINUX (/proc/self/maps unavailable outside Linux)"
    return 0
  fi
  log "ERROR: Clean-room proof assertion failed — probe output did not report both MAPS_LIBTPU=0 and MAPS_LIBPYTHON=0."
  log "CLEANROOM_MAPS_STATUS=FAIL"
  return 1
}

: > "${LOG}"
log "=== FreeTPU Ironwood (tpu7x) Day-1 libopentpu VFIO Probe ==="
log "timestamp_utc=${TIMESTAMP} live_vfio=${LIVE_VFIO} mock_mode=${MOCK_MODE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "[DRY-RUN] Would build libopentpu host targets, run opentpu_host_test (incl. tpu7x HAL geometry + ReBAR tests), and run opentpu_vfio_smoke (--mock or live) asserting MAPS_LIBTPU=0, MAPS_LIBPYTHON=0, and explicit pre-reset refusal on unvalidated PCI DIDs."
  exit 0
fi

if [[ "${SKIP_BUILD}" -ne 1 ]]; then
  make -C "${REPO_ROOT}/libopentpu" verify-header-host host test 2>&1 | tee -a "${LOG}"
fi

SMOKE_BIN_OVERRIDE="${SMOKE_BIN:-}"
SMOKE_BIN="${SMOKE_BIN:-${REPO_ROOT}/libopentpu/build/host/opentpu_vfio_smoke}"
if [[ ! -x "${SMOKE_BIN}" ]]; then
  log "ERROR: smoke binary not found or not executable: ${SMOKE_BIN}"
  exit 1
fi

if [[ "${MOCK_MODE}" -eq 1 || "${LIVE_VFIO}" -eq 0 || ( "$(uname -s)" != "Linux" && -z "${SMOKE_BIN_OVERRIDE}" ) ]]; then
  log "--- Running opentpu_vfio_smoke --mock under tpu7x profile ---"
  mock_out="$(OPENTPU_MOCK_CHIP=tpu7x "${SMOKE_BIN}" --mock 2>&1)"
  printf '%s\n' "${mock_out}" | tee -a "${LOG}"
  assert_cleanroom_maps "${mock_out}" 1
  log "MOCK_VFIO_PROBE_STATUS=PASS"
  echo "Probe complete: ${LOG}"
  exit 0
fi

first_vfio_tpu_bdf=""
for dev_path in "${PCI_SYSFS_DIR}"/*; do
  [[ -r "${dev_path}/vendor" && -L "${dev_path}/driver" ]] || continue
  vid="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/vendor" 2>/dev/null || true)"
  cls="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/class" 2>/dev/null || true)"
  drv="$(basename "$(readlink -f "${dev_path}/driver" 2>/dev/null || true)" 2>/dev/null || true)"
  case "${cls}" in
    0x02*) continue ;;
    0xff*|0x12*|0x0b*|0x08*) ;;
    *) continue ;;
  esac
  if [[ "${vid}" == "0x1ae0" && "${drv}" == "vfio-pci" ]]; then
    first_vfio_tpu_bdf="$(basename "${dev_path}")"
    break
  fi
done

probe_cmd=("${SMOKE_BIN}")
if [[ -n "${first_vfio_tpu_bdf}" ]]; then
  probe_cmd+=(--bdf "${first_vfio_tpu_bdf}")
fi

log "--- Probing live /dev/vfio/* with ${probe_cmd[*]} (expect PASS on validated 0x0063 or explicit pre-reset refusal on unvalidated TPU PCI DID) ---"
probe_rc=0
probe_out="$("${probe_cmd[@]}" 2>&1)" || probe_rc=$?
printf '%s\n' "${probe_out}" | tee -a "${LOG}"

if ! assert_cleanroom_maps "${probe_out}" 0; then
  log "LIVE_VFIO_PROBE_STATUS=CLEANROOM_MAPS_CHECK_FAILED (exit=${probe_rc})"
  exit 1
fi

if [[ "${probe_rc}" -eq 0 ]]; then
  log "LIVE_VFIO_PROBE_STATUS=PASS"
elif [[ "${probe_rc}" -eq 1 ]] && printf '%s\n' "${probe_out}" | grep -Fq "Refusing to reset/program unsupported VFIO PCI device"; then
  log "LIVE_VFIO_PROBE_STATUS=EXPECTED_UNVALIDATED_DEVICE_REFUSAL (exit=${probe_rc})"
else
  log "LIVE_VFIO_PROBE_STATUS=UNEXPECTED_FAILURE (exit=${probe_rc})"
  exit "${probe_rc:-1}"
fi

echo "Probe complete: ${LOG}"
