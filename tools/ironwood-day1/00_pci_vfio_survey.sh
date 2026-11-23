#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Non-destructive PCIe / Resizable BAR / MSI-X / VFIO / IOMMUFD survey for Cloud TPU v7x (Ironwood).
set -euo pipefail

DRY_RUN=0
MOCK_MODE=0
ALLOW_NO_PJRT=0
STAGE_LIBTPU=0
LIBTPU_VERSION="${LIBTPU_VERSION:-0.0.53.1}"
LIBTPU_DEST="${LIBTPU_DEST:-}"
EXPECTED_PCI_PACKAGES="${EXPECTED_PCI_PACKAGES:-4}"
EXPECTED_LOGICAL_DEVICES="${EXPECTED_LOGICAL_DEVICES:-8}"
OUT_DIR=""
PCI_SYSFS_DIR="${PCI_SYSFS_DIR:-/sys/bus/pci/devices}"

usage() {
  cat <<'EOF'
Usage: 00_pci_vfio_survey.sh [--out-dir DIR] [--expected-packages N] [--expected-devices N]
                             [--stage-libtpu] [--libtpu-version VER] [--libtpu-dest PATH]
                             [--dry-run] [--mock] [--allow-no-pjrt]

Non-destructive Day-1 hardware survey for Cloud TPU v7x hosts (tpu7x-standard-4t):
  - Enumerates PCI Vendor 0x1ae0 TPU/accelerator functions (excluding gVNIC / 0x02* network controllers),
    Device IDs, BAR0/BAR2/BAR4 sizes, and Resizable BAR capability size fields
  - Captures MSI-X capability counts, iommufd (/dev/iommu) and /dev/vfio/* viability
  - Optionally stages standalone libtpu.so from the PyPI cp312 wheel (--stage-libtpu, default 0.0.53.1)
    without installing JAX
  - Queries live PJRT device descriptions (id, coords, core_on_chip) via native C++ PJRT probe with
    automatic JAX local_devices(backend="tpu") fallback, and validates observed counts against
    expected Ironwood package/chiplet counts (default: 4 packages / 8 logical devices on tpu7x-4t)
    (pass --allow-no-pjrt to warn instead of failing if live PJRT is unavailable)
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
    --expected-packages)
      require_arg "$#" "$1"
      EXPECTED_PCI_PACKAGES="$2"
      shift 2
      ;;
    --expected-devices)
      require_arg "$#" "$1"
      EXPECTED_LOGICAL_DEVICES="$2"
      shift 2
      ;;
    --stage-libtpu)
      STAGE_LIBTPU=1
      shift
      ;;
    --libtpu-version)
      require_arg "$#" "$1"
      LIBTPU_VERSION="$2"
      shift 2
      ;;
    --libtpu-dest)
      require_arg "$#" "$1"
      LIBTPU_DEST="$2"
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
    --allow-no-pjrt)
      ALLOW_NO_PJRT=1
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

for _count_var in EXPECTED_PCI_PACKAGES EXPECTED_LOGICAL_DEVICES; do
  if [[ ! "${!_count_var}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${_count_var} must be a non-negative integer (got '${!_count_var}')" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${REPO_ROOT}/runs/ironwood-day1-${TIMESTAMP}"
fi
mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/00_pci_vfio_survey.txt"
if [[ -z "${LIBTPU_DEST}" ]]; then
  LIBTPU_DEST="${OUT_DIR}/libtpu-${LIBTPU_VERSION}.so"
fi

log() {
  echo "$*" | tee -a "${REPORT}"
}

emit_topology_status() {
  local obs_pci="$1"
  local obs_pjrt="$2"
  if [[ "${obs_pjrt}" =~ ^[0-9]+$ && "${obs_pci}" -eq "${EXPECTED_PCI_PACKAGES}" && "${obs_pjrt}" -eq "${EXPECTED_LOGICAL_DEVICES}" ]]; then
    log "STATUS: PJRT_TOPOLOGY_CAPTURED (pci_tpu_functions=${obs_pci} expected_packages=${EXPECTED_PCI_PACKAGES} pjrt_logical_devices=${obs_pjrt} expected_devices=${EXPECTED_LOGICAL_DEVICES})"
  else
    log "WARN: TOPOLOGY_COUNT_MISMATCH observed_pci_functions=${obs_pci} expected_packages=${EXPECTED_PCI_PACKAGES} observed_pjrt_devices=${obs_pjrt:-unknown} expected_devices=${EXPECTED_LOGICAL_DEVICES}"
    log "STATUS: PJRT_TOPOLOGY_MISMATCH (pci_tpu_functions=${obs_pci} expected_packages=${EXPECTED_PCI_PACKAGES} pjrt_logical_devices=${obs_pjrt:-unknown} expected_devices=${EXPECTED_LOGICAL_DEVICES})"
  fi
}

compute_file_sha256() {
  local target="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${target}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${target}" | awk '{print $1}'
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "${target}" <<'PYEOF'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PYEOF
  else
    return 1
  fi
}

check_staged_libtpu_stamp() {
  local dest="$1"
  local want_ver="$2"
  local stamp="${dest}.version"
  LIBTPU_STAMP_REASON=""
  if [[ ! -s "${dest}" ]]; then
    LIBTPU_STAMP_REASON="missing_binary"
    return 1
  fi
  if [[ ! -r "${stamp}" ]]; then
    LIBTPU_STAMP_REASON="missing_version_stamp"
    return 1
  fi
  local got_ver="" got_sha="" actual_sha=""
  got_ver="$(sed -n 's/^version=//p' "${stamp}" | head -n 1)"
  got_sha="$(sed -n 's/^sha256=//p' "${stamp}" | head -n 1)"
  if [[ "${got_ver}" != "${want_ver}" ]]; then
    LIBTPU_STAMP_REASON="version_mismatch(have=${got_ver:-unknown},want=${want_ver})"
    return 1
  fi
  if [[ -z "${got_sha}" ]]; then
    LIBTPU_STAMP_REASON="missing_sha256_in_stamp"
    return 1
  fi
  actual_sha="$(compute_file_sha256 "${dest}" 2>/dev/null || true)"
  if [[ -z "${actual_sha}" || "${actual_sha}" != "${got_sha}" ]]; then
    LIBTPU_STAMP_REASON="sha256_mismatch"
    return 1
  fi
  LIBTPU_STAMP_REASON="verified(version=${got_ver},sha256=${actual_sha})"
  return 0
}

stage_libtpu_wheel() {
  if [[ "${STAGE_LIBTPU}" -ne 1 ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" -eq 1 || "${MOCK_MODE}" -eq 1 ]]; then
    if check_staged_libtpu_stamp "${LIBTPU_DEST}" "${LIBTPU_VERSION}"; then
      log "[LIBTPU-STAGE] Would reuse verified staged libtpu.so (${LIBTPU_STAMP_REASON}): ${LIBTPU_DEST}"
    else
      log "[LIBTPU-STAGE] Would extract libtpu/libtpu.so from PyPI libtpu==${LIBTPU_VERSION} cp312 x86_64 wheel to ${LIBTPU_DEST} (reason=${LIBTPU_STAMP_REASON}, zero JAX install required)"
    fi
    return 0
  fi
  if check_staged_libtpu_stamp "${LIBTPU_DEST}" "${LIBTPU_VERSION}"; then
    log "[LIBTPU-STAGE] Reusing verified staged libtpu.so (${LIBTPU_STAMP_REASON}): ${LIBTPU_DEST}"
    export LIBTPU_PATH="${LIBTPU_DEST}"
    return 0
  fi
  if [[ -e "${LIBTPU_DEST}" ]]; then
    log "[LIBTPU-STAGE] Existing ${LIBTPU_DEST} cannot be reused (${LIBTPU_STAMP_REASON}); re-staging libtpu==${LIBTPU_VERSION}"
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    log "ERROR: --stage-libtpu requires python3 to download and extract the cp312 wheel."
    return 1
  fi
  log "[LIBTPU-STAGE] Extracting libtpu/libtpu.so from PyPI libtpu==${LIBTPU_VERSION} cp312 x86_64 wheel -> ${LIBTPU_DEST}"
  mkdir -p "$(dirname "${LIBTPU_DEST}")"
  python3 - "${LIBTPU_VERSION}" "${LIBTPU_DEST}" <<'PYEOF' 2>&1 | tee -a "${REPORT}"
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

version, dest = sys.argv[1], sys.argv[2]
meta_url = f"https://pypi.org/pypi/libtpu/{version}/json"
with urllib.request.urlopen(meta_url, timeout=30) as resp:
    meta = json.load(resp)

cands = [
    u for u in meta.get("urls", [])
    if "cp312" in u.get("filename", "") and "x86_64" in u.get("filename", "") and u.get("filename", "").endswith(".whl")
]
if not cands:
    raise SystemExit(f"No cp312 x86_64 wheel found for libtpu=={version}")

whl = cands[0]
expected_sha = (whl.get("digests") or {}).get("sha256")
with urllib.request.urlopen(whl["url"], timeout=120) as resp:
    whl_bytes = resp.read()

actual_sha = hashlib.sha256(whl_bytes).hexdigest()
if expected_sha and actual_sha != expected_sha:
    raise SystemExit(f"Wheel sha256 mismatch: expected={expected_sha} actual={actual_sha}")

with zipfile.ZipFile(io.BytesIO(whl_bytes)) as zf:
    so_members = [n for n in zf.namelist() if n.endswith("/libtpu.so")]
    if not so_members:
        raise SystemExit(f"libtpu.so not found inside {whl['filename']}")
    so_bytes = zf.read(so_members[0])

so_sha = hashlib.sha256(so_bytes).hexdigest()
tmp_dest = f"{dest}.tmp.{os.getpid()}"
with open(tmp_dest, "wb") as f:
    f.write(so_bytes)
os.chmod(tmp_dest, 0o755)
os.replace(tmp_dest, dest)

stamp_path = f"{dest}.version"
tmp_stamp = f"{stamp_path}.tmp.{os.getpid()}"
with open(tmp_stamp, "w", encoding="utf-8") as f:
    f.write(f"version={version}\nsha256={so_sha}\nwheel={whl['filename']}\nwheel_sha256={actual_sha}\n")
os.replace(tmp_stamp, stamp_path)
print(f"LIBTPU_STAGED path={dest} version={version} wheel={whl['filename']} bytes={len(so_bytes)} so_sha256={so_sha} wheel_sha256={actual_sha}")
PYEOF
  export LIBTPU_PATH="${LIBTPU_DEST}"
}

: > "${REPORT}"
log "=== FreeTPU Ironwood (tpu7x) Day-1 PCI / VFIO Survey ==="
log "timestamp_utc=${TIMESTAMP}"
log "hostname=$(hostname)"
log "kernel=$(uname -srm)"
log "dry_run=${DRY_RUN} mock_mode=${MOCK_MODE} expected_packages=${EXPECTED_PCI_PACKAGES} expected_devices=${EXPECTED_LOGICAL_DEVICES} stage_libtpu=${STAGE_LIBTPU}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  stage_libtpu_wheel
  log "[DRY-RUN] Would inspect ${PCI_SYSFS_DIR} for vendor=0x1ae0, read resource/iommu_group/driver, check /dev/vfio and /dev/iommu, and query libtpu.so GetPjrtApi (with JAX fallback) against expected ${EXPECTED_PCI_PACKAGES} packages / ${EXPECTED_LOGICAL_DEVICES} logical devices."
  exit 0
fi

if [[ "${MOCK_MODE}" -eq 1 || ( "$(uname -s)" != "Linux" && "${PCI_SYSFS_DIR}" == "/sys/bus/pci/devices" ) ]]; then
  stage_libtpu_wheel
  log "[MOCK] Simulating tpu7x-4t PCI survey (4 physical PCI functions, 0x1ae0:0x0075, dual-chiplet 96 GiB/chiplet)"
  for i in 0 1 2 3; do
    log "PCI 0000:00:0$((i+4)).0 vendor=0x1ae0 device=0x0075 class=0xff0000 driver=vfio-pci iommu_group=${i} bar0=64MiB bar4=192GiB(ReBAR)"
  done
  log "DEV_VFIO=/dev/vfio/vfio,/dev/vfio/0..3"
  log "total_tpu_pci_functions=4"
  log "PJRT_DEVICE index=0 id=0 local_hardware_id=0 kind=\"TPU v7x\" coords=[0,0,0] core_on_chip=0"
  log "PJRT_DEVICE index=1 id=1 local_hardware_id=1 kind=\"TPU v7x\" coords=[0,0,0] core_on_chip=1"
  log "PJRT_DEVICE index=2 id=2 local_hardware_id=2 kind=\"TPU v7x\" coords=[1,0,0] core_on_chip=0"
  log "PJRT_DEVICE index=3 id=3 local_hardware_id=3 kind=\"TPU v7x\" coords=[1,0,0] core_on_chip=1"
  log "PJRT_DEVICE index=4 id=4 local_hardware_id=4 kind=\"TPU v7x\" coords=[1,1,0] core_on_chip=0"
  log "PJRT_DEVICE index=5 id=5 local_hardware_id=5 kind=\"TPU v7x\" coords=[1,1,0] core_on_chip=1"
  log "PJRT_DEVICE index=6 id=6 local_hardware_id=6 kind=\"TPU v7x\" coords=[0,1,0] core_on_chip=0"
  log "PJRT_DEVICE index=7 id=7 local_hardware_id=7 kind=\"TPU v7x\" coords=[0,1,0] core_on_chip=1"
  log "PJRT_LOGICAL_DEVICES=8 (4 packages x 2 chiplets)"
  emit_topology_status 4 8
  exit 0
fi

stage_libtpu_wheel

is_tpu_pci_class() {
  local cls_lc
  cls_lc="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  # Cloud TPUs expose class 0xff0000 (vendor-specific/unassigned) or accelerator/processor
  # classes (0x12*, 0x0b*, 0x08*). Explicitly exclude network controllers (0x02*, e.g. gVNIC 0x020000)
  # and other non-accelerator Google virtual PCI devices.
  case "${cls_lc}" in
    0x02*) return 1 ;;
    0xff*|0x12*|0x0b*|0x08*) return 0 ;;
    *) return 1 ;;
  esac
}

log "--- /dev/vfio and /dev/iommu nodes ---"
ls -la /dev/vfio /dev/iommu 2>&1 | tee -a "${REPORT}" || true

tpu_count=0
non_tpu_google_count=0
for dev_path in "${PCI_SYSFS_DIR}"/*; do
  [[ -e "${dev_path}/vendor" ]] || continue
  vid="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/vendor" 2>/dev/null || echo unknown)"
  [[ "${vid}" == "0x1ae0" ]] || continue
  bdf="$(basename "${dev_path}")"
  did="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/device" 2>/dev/null || echo unknown)"
  cls="$(tr '[:upper:]' '[:lower:]' < "${dev_path}/class" 2>/dev/null || echo unknown)"
  if ! is_tpu_pci_class "${cls}"; then
    log "skip_non_tpu_0x1ae0 bdf=${bdf} vid=${vid} did=${did} class=${cls}"
    non_tpu_google_count=$((non_tpu_google_count + 1))
    continue
  fi
  if [[ -L "${dev_path}/driver" && -e "${dev_path}/driver" ]]; then
    drv="$(basename "$(readlink -f "${dev_path}/driver" 2>/dev/null)" 2>/dev/null || echo unbound)"
  else
    drv="unbound"
  fi
  if [[ -L "${dev_path}/iommu_group" && -e "${dev_path}/iommu_group" ]]; then
    grp="$(basename "$(readlink -f "${dev_path}/iommu_group" 2>/dev/null)" 2>/dev/null || echo none)"
  else
    grp="none"
  fi
  numa="$(cat "${dev_path}/numa_node" 2>/dev/null || echo -1)"
  log "--- TPU BDF=${bdf} vid=${vid} did=${did} class=${cls} driver=${drv} iommu_group=${grp} numa=${numa} ---"
  if [[ -r "${dev_path}/resource" ]]; then
    nl -ba "${dev_path}/resource" | head -n 7 | tee -a "${REPORT}"
  fi
  if command -v lspci >/dev/null 2>&1; then
    lspci_out="$(lspci -s "${bdf}" -nnvvv 2>/dev/null | awk '
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
      tolower($0) ~ /<access denied>/ { print; next }
      /Google|Region |MSI-X|LnkSta:/ { print }
    ' || true)"
    if [[ -n "${lspci_out}" ]]; then
      printf '%s\n' "${lspci_out}" | tee -a "${REPORT}"
      if printf '%s\n' "${lspci_out}" | grep -qi "<access denied>"; then
        log "WARN: PCI capability reads were denied (<access denied>) for ${bdf}; re-run with sudo to inspect Resizable BAR and MSI-X fields."
      fi
    fi
  fi
  tpu_count=$((tpu_count + 1))
done
log "total_tpu_pci_functions=${tpu_count} skipped_non_tpu_0x1ae0=${non_tpu_google_count}"

if [[ -f "${REPO_ROOT}/experimental/cloud-tpu/vfio_pci_survey.py" ]] && command -v python3 >/dev/null 2>&1; then
  log "--- Running experimental/cloud-tpu/vfio_pci_survey.py ---"
  python3 "${REPO_ROOT}/experimental/cloud-tpu/vfio_pci_survey.py" 2>&1 | tee -a "${REPORT}" || true
fi

extract_pjrt_logical_devices() {
  printf '%s\n' "$1" | sed -n 's/^PJRT_LOGICAL_DEVICES=\([0-9][0-9]*\).*/\1/p' | tail -n 1
}

probe_live_pjrt_topology() {
  PJRT_LOGICAL_DEVICES_COUNT=""
  PJRT_NATIVE_FAIL_REASON=""
  local pjrt_header="${REPO_ROOT}/libopentpu/third_party/openxla/pjrt_c_api.h"
  local libtpu_path="${LIBTPU_PATH:-${TPU_LIBRARY_PATH:-}}"
  if [[ -z "${libtpu_path}" ]]; then
    local cand=""
    for cand in /lib/libtpu.so /usr/lib/libtpu.so /usr/local/lib/libtpu.so; do
      if [[ -r "${cand}" ]]; then
        libtpu_path="${cand}"
        break
      fi
    done
  fi
  if [[ -z "${libtpu_path}" ]] && command -v python3 >/dev/null 2>&1; then
    libtpu_path="$(python3 -c 'import libtpu; print(libtpu.get_library_path())' 2>/dev/null || true)"
  fi

  local cxx_bin=""
  for cand in c++ g++ clang++; do
    if command -v "${cand}" >/dev/null 2>&1; then
      cxx_bin="${cand}"
      break
    fi
  done

  if [[ -n "${libtpu_path}" && -r "${libtpu_path}" && -n "${cxx_bin}" && -r "${pjrt_header}" ]]; then
    local tmp_dir=""
    tmp_dir="$(mktemp -d)"
    cat > "${tmp_dir}/pjrt_topo_probe.cpp" <<'CPPEOF'
#include "pjrt_c_api.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>

static void print_err(const PJRT_Api* api, PJRT_Error* err, const char* step) {
  if (!err) return;
  const char* msg = "";
  size_t len = 0;
  if (api && api->PJRT_Error_Message) {
    PJRT_Error_Message_Args em = {};
    em.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    em.error = err;
    api->PJRT_Error_Message(&em);
    if (em.message) { msg = em.message; len = em.message_size; }
  }
  std::fprintf(stderr, "PJRT %s failed: %.*s\n", step, static_cast<int>(len), msg);
  if (api && api->PJRT_Error_Destroy) {
    PJRT_Error_Destroy_Args ed = {};
    ed.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    ed.error = err;
    api->PJRT_Error_Destroy(&ed);
  }
}

int main(int argc, char** argv) {
  if (argc < 2 || !argv[1][0]) return 2;
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", argv[1], dlerror());
    return 3;
  }
  using GetApiFn = const PJRT_Api* (*)();
  auto get_api = reinterpret_cast<GetApiFn>(dlsym(h, "GetPjrtApi"));
  if (!get_api) {
    std::fprintf(stderr, "GetPjrtApi symbol not found in %s\n", argv[1]);
    return 4;
  }
  const PJRT_Api* api = get_api();
  if (!api || !api->PJRT_Client_Create || !api->PJRT_Client_AddressableDevices) {
    std::fprintf(stderr, "Incomplete PJRT_Api returned by %s\n", argv[1]);
    return 5;
  }
  if (api->PJRT_Plugin_Initialize) {
    PJRT_Plugin_Initialize_Args pi = {};
    pi.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    if (PJRT_Error* e = api->PJRT_Plugin_Initialize(&pi)) {
      print_err(api, e, "Plugin_Initialize");
      return 6;
    }
  }
  PJRT_Client_Create_Args cc = {};
  cc.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  if (PJRT_Error* e = api->PJRT_Client_Create(&cc)) {
    print_err(api, e, "Client_Create");
    return 7;
  }
  PJRT_Client_AddressableDevices_Args ad = {};
  ad.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  ad.client = cc.client;
  if (PJRT_Error* e = api->PJRT_Client_AddressableDevices(&ad)) {
    print_err(api, e, "Client_AddressableDevices");
    return 8;
  }
  std::printf("PJRT_LIBRARY=%s\n", argv[1]);
  std::printf("PJRT_LOGICAL_DEVICES=%zu\n", ad.num_addressable_devices);
  for (size_t i = 0; i < ad.num_addressable_devices; ++i) {
    PJRT_Device* dev = ad.addressable_devices[i];
    int id = -1, hw_id = -1;
    std::string kind = "unknown";
    std::string coords = "[]";
    long long core = -1;
    if (api->PJRT_Device_LocalHardwareId) {
      PJRT_Device_LocalHardwareId_Args lh = {};
      lh.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
      lh.device = dev;
      if (!api->PJRT_Device_LocalHardwareId(&lh)) hw_id = lh.local_hardware_id;
    }
    if (api->PJRT_Device_GetDescription) {
      PJRT_Device_GetDescription_Args gd = {};
      gd.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
      gd.device = dev;
      if (!api->PJRT_Device_GetDescription(&gd) && gd.device_description) {
        if (api->PJRT_DeviceDescription_Id) {
          PJRT_DeviceDescription_Id_Args ia = {};
          ia.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
          ia.device_description = gd.device_description;
          if (!api->PJRT_DeviceDescription_Id(&ia)) id = ia.id;
        }
        if (api->PJRT_DeviceDescription_Kind) {
          PJRT_DeviceDescription_Kind_Args ka = {};
          ka.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
          ka.device_description = gd.device_description;
          if (!api->PJRT_DeviceDescription_Kind(&ka) && ka.device_kind) {
            kind.assign(ka.device_kind, ka.device_kind_size);
          }
        }
        if (api->PJRT_DeviceDescription_Attributes) {
          PJRT_DeviceDescription_Attributes_Args aa = {};
          aa.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
          aa.device_description = gd.device_description;
          if (!api->PJRT_DeviceDescription_Attributes(&aa)) {
            for (size_t ai = 0; ai < aa.num_attributes; ++ai) {
              const PJRT_NamedValue& a = aa.attributes[ai];
              std::string aname(a.name ? a.name : "", a.name_size);
              if (aname == "coords" && a.type == PJRT_NamedValue_kInt64List && a.int64_array_value) {
                coords = "[";
                for (size_t ci = 0; ci < a.value_size; ++ci) {
                  if (ci > 0) coords += ",";
                  coords += std::to_string(a.int64_array_value[ci]);
                }
                coords += "]";
              } else if (aname == "core_on_chip" && a.type == PJRT_NamedValue_kInt64) {
                core = static_cast<long long>(a.int64_value);
              }
            }
          }
        }
      }
    }
    std::printf("PJRT_DEVICE index=%zu id=%d local_hardware_id=%d kind=\"%s\" coords=%s core_on_chip=%lld\n",
                i, id, hw_id, kind.c_str(), coords.c_str(), core);
  }
  if (api->PJRT_Client_Destroy) {
    PJRT_Client_Destroy_Args cd = {};
    cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    cd.client = cc.client;
    if (PJRT_Error* de = api->PJRT_Client_Destroy(&cd)) {
      print_err(api, de, "Client_Destroy");
    }
  }
  return ad.num_addressable_devices > 0 ? 0 : 9;
}
CPPEOF
    local probe_rc=0
    local probe_out=""
    if "${cxx_bin}" -std=c++17 -O2 -I"$(dirname "${pjrt_header}")" \
        "${tmp_dir}/pjrt_topo_probe.cpp" -ldl -o "${tmp_dir}/pjrt_topo_probe" 2>"${tmp_dir}/build.err"; then
      probe_out="$("${tmp_dir}/pjrt_topo_probe" "${libtpu_path}" 2>&1)" || probe_rc=$?
      rm -rf "${tmp_dir}"
      if [[ "${probe_rc}" -eq 0 ]]; then
        printf '%s\n' "${probe_out}" | tee -a "${REPORT}"
        PJRT_LOGICAL_DEVICES_COUNT="$(extract_pjrt_logical_devices "${probe_out}")"
        return 0
      fi
      printf '%s\n' "${probe_out}" | tee -a "${REPORT}"
      PJRT_NATIVE_FAIL_REASON="libtpu.so PJRT probe exited ${probe_rc}"
      PJRT_FAIL_REASON="${PJRT_NATIVE_FAIL_REASON}"
      log "WARN: native C++ PJRT probe failed (${PJRT_NATIVE_FAIL_REASON}); falling back to python3 jax.local_devices(backend='tpu')"
    else
      local berr=""
      berr="$(tr '\n' ' ' < "${tmp_dir}/build.err")"
      rm -rf "${tmp_dir}"
      PJRT_NATIVE_FAIL_REASON="failed to compile PJRT probe: ${berr}"
      PJRT_FAIL_REASON="${PJRT_NATIVE_FAIL_REASON}"
      log "WARN: ${PJRT_NATIVE_FAIL_REASON}; falling back to python3 jax.local_devices(backend='tpu')"
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    local py_rc=0
    local py_out=""
    py_out="$(python3 - <<'PYEOF' 2>&1
import jax
try:
    devs = jax.local_devices(backend="tpu")
except Exception as exc:
    print(f"JAX TPU backend unavailable: {exc}")
    raise SystemExit(10)
if not devs:
    print("JAX TPU backend returned 0 devices")
    raise SystemExit(9)
print("PJRT_BACKEND=jax-tpu")
print(f"PJRT_LOGICAL_DEVICES={len(devs)}")
for i, d in enumerate(devs):
    coords = getattr(d, "coords", None)
    core = getattr(d, "core_on_chip", -1)
    kind = getattr(d, "device_kind", "unknown")
    hw = getattr(d, "local_hardware_id", d.id)
    coords_str = "[" + ",".join(str(c) for c in coords) + "]" if coords is not None else "[]"
    print(f'PJRT_DEVICE index={i} id={d.id} local_hardware_id={hw} kind="{kind}" coords={coords_str} core_on_chip={core}')
PYEOF
)" || py_rc=$?
    if [[ "${py_rc}" -eq 0 ]]; then
      printf '%s\n' "${py_out}" | tee -a "${REPORT}"
      PJRT_LOGICAL_DEVICES_COUNT="$(extract_pjrt_logical_devices "${py_out}")"
      if [[ -n "${PJRT_NATIVE_FAIL_REASON}" ]]; then
        log "NOTE: JAX fallback succeeded after native PJRT probe failure (${PJRT_NATIVE_FAIL_REASON})"
      fi
      return 0
    fi
    if [[ -n "${py_out}" ]]; then
      printf '%s\n' "${py_out}" | tee -a "${REPORT}"
    fi
    PJRT_FAIL_REASON="${PJRT_FAIL_REASON:-libtpu.so not found}; jax fallback exited ${py_rc}"
    return 1
  fi

  PJRT_FAIL_REASON="${PJRT_FAIL_REASON:-neither libtpu.so + C++ compiler nor python3+jax available}"
  return 1
}

log "--- Live PJRT Logical Device & Chiplet Topology ---"
PJRT_FAIL_REASON=""
PJRT_NATIVE_FAIL_REASON=""
PJRT_LOGICAL_DEVICES_COUNT=""
if probe_live_pjrt_topology; then
  emit_topology_status "${tpu_count}" "${PJRT_LOGICAL_DEVICES_COUNT}"
else
  log "STATUS: PJRT_TOPOLOGY_UNAVAILABLE (${PJRT_FAIL_REASON})"
  if [[ "${ALLOW_NO_PJRT}" -ne 1 ]]; then
    log "ERROR: Live PJRT topology probe failed; re-run with --stage-libtpu (extracts cp312 libtpu ${LIBTPU_VERSION} without JAX), LIBTPU_PATH=/path/to/libtpu.so, or --allow-no-pjrt."
    exit 1
  fi
fi

echo "Survey complete: ${REPORT}"
