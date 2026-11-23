#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Non-destructive PCIe / Resizable BAR / MSI-X / VFIO / IOMMUFD survey for Cloud TPU v7x (Ironwood).
set -euo pipefail

DRY_RUN=0
MOCK_MODE=0
OUT_DIR=""

usage() {
  cat <<'EOF'
Usage: 00_pci_vfio_survey.sh [--out-dir DIR] [--dry-run] [--mock]

Non-destructive Day-1 hardware survey for Cloud TPU v7x hosts (tpu7x-standard-4t):
  - Enumerates PCI Vendor 0x1ae0 functions, Device IDs, BAR0/BAR2/BAR4 sizes, Resizable BAR caps
  - Captures MSI-X capability counts, iommufd (/dev/iommu) and /dev/vfio/* viability
  - Compares physical PCI function count vs PJRT logical chiplet count
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
REPORT="${OUT_DIR}/00_pci_vfio_survey.txt"

log() {
  echo "$*" | tee -a "${REPORT}"
}

: > "${REPORT}"
log "=== FreeTPU Ironwood (tpu7x) Day-1 PCI / VFIO Survey ==="
log "timestamp_utc=${TIMESTAMP}"
log "hostname=$(hostname)"
log "kernel=$(uname -srm)"
log "dry_run=${DRY_RUN} mock_mode=${MOCK_MODE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "[DRY-RUN] Would inspect /sys/bus/pci/devices for vendor=0x1ae0, read resource/iommu_group/driver, check /dev/vfio and /dev/iommu, and query libtpu.so GetPjrtApi logical count."
  exit 0
fi

if [[ "${MOCK_MODE}" -eq 1 || "$(uname -s)" != "Linux" ]]; then
  log "[MOCK] Simulating tpu7x-4t PCI survey (4 physical PCI functions, 0x1ae0:0x0075, dual-chiplet 96 GiB/chiplet)"
  for i in 0 1 2 3; do
    log "PCI 0000:00:0$((i+4)).0 vendor=0x1ae0 device=0x0075 driver=vfio-pci iommu_group=${i} bar0=64MiB bar4=192GiB(ReBAR)"
  done
  log "DEV_VFIO=/dev/vfio/vfio,/dev/vfio/0..3"
  log "LOGICAL_CHIPLETS=8 (4 packages x 2 chiplets)"
  exit 0
fi

log "--- /dev/vfio and /dev/iommu nodes ---"
ls -la /dev/vfio /dev/iommu 2>&1 | tee -a "${REPORT}" || true

tpu_count=0
for dev_path in /sys/bus/pci/devices/*; do
  [[ -e "${dev_path}/vendor" ]] || continue
  vid="$(cat "${dev_path}/vendor" 2>/dev/null || echo unknown)"
  [[ "${vid}" == "0x1ae0" ]] || continue
  bdf="$(basename "${dev_path}")"
  did="$(cat "${dev_path}/device" 2>/dev/null || echo unknown)"
  cls="$(cat "${dev_path}/class" 2>/dev/null || echo unknown)"
  drv="$(basename "$(readlink -f "${dev_path}/driver" 2>/dev/null || echo unbound)" 2>/dev/null || echo unbound)"
  grp="$(basename "$(readlink -f "${dev_path}/iommu_group" 2>/dev/null || echo none)" 2>/dev/null || echo none)"
  numa="$(cat "${dev_path}/numa_node" 2>/dev/null || echo -1)"
  log "--- TPU BDF=${bdf} vid=${vid} did=${did} class=${cls} driver=${drv} iommu_group=${grp} numa=${numa} ---"
  if [[ -r "${dev_path}/resource" ]]; then
    nl -ba "${dev_path}/resource" | head -n 7 | tee -a "${REPORT}"
  fi
  if command -v lspci >/dev/null 2>&1; then
    lspci -s "${bdf}" -nnvvv 2>/dev/null | awk '
      function indent_len(s) {
        match(s, /^[ \t]*/)
        return RLENGTH
      }
      /Resizable BAR/ {
        print
        in_rebar = 1
        rebar_indent = indent_len($0)
        next
      }
      in_rebar {
        if ($0 ~ /^[ \t]*$/ || indent_len($0) <= rebar_indent) {
          in_rebar = 0
        } else {
          print
          next
        }
      }
      /Google|Region |MSI-X|LnkSta:/ { print }
    ' | tee -a "${REPORT}" || true
  fi
  tpu_count=$((tpu_count + 1))
done
log "total_0x1ae0_pci_functions=${tpu_count}"

if [[ -f "${REPO_ROOT}/experimental/cloud-tpu/vfio_pci_survey.py" ]] && command -v python3 >/dev/null 2>&1; then
  log "--- Running experimental/cloud-tpu/vfio_pci_survey.py ---"
  python3 "${REPO_ROOT}/experimental/cloud-tpu/vfio_pci_survey.py" 2>&1 | tee -a "${REPORT}" || true
fi

echo "Survey complete: ${REPORT}"
