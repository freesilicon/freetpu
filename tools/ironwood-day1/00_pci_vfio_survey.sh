#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Non-destructive PCIe / Resizable BAR / MSI-X / VFIO / IOMMUFD survey for Cloud TPU v7x (Ironwood).
set -euo pipefail

DRY_RUN=0
MOCK_MODE=0
ALLOW_NO_PJRT=0
OUT_DIR=""
PCI_SYSFS_DIR="${PCI_SYSFS_DIR:-/sys/bus/pci/devices}"

usage() {
  cat <<'EOF'
Usage: 00_pci_vfio_survey.sh [--out-dir DIR] [--dry-run] [--mock] [--allow-no-pjrt]

Non-destructive Day-1 hardware survey for Cloud TPU v7x hosts (tpu7x-standard-4t):
  - Enumerates PCI Vendor 0x1ae0 TPU/accelerator functions (excluding gVNIC / 0x02* network controllers),
    Device IDs, BAR0/BAR2/BAR4 sizes, and Resizable BAR capability size fields
  - Captures MSI-X capability counts, iommufd (/dev/iommu) and /dev/vfio/* viability
  - Queries live PJRT device descriptions (id, coords, core_on_chip) and compares logical chiplets vs PCI functions
    (pass --allow-no-pjrt to warn instead of failing if live PJRT is unavailable)
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
  exit 0
fi

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

probe_live_pjrt_topology() {
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
        return 0
      fi
      printf '%s\n' "${probe_out}" | tee -a "${REPORT}"
      PJRT_FAIL_REASON="libtpu.so PJRT probe exited ${probe_rc}"
      return 1
    else
      local berr=""
      berr="$(tr '\n' ' ' < "${tmp_dir}/build.err")"
      rm -rf "${tmp_dir}"
      PJRT_FAIL_REASON="failed to compile PJRT probe: ${berr}"
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
      return 0
    fi
    PJRT_FAIL_REASON="${PJRT_FAIL_REASON:-libtpu.so not found}; jax fallback exited ${py_rc}"
    return 1
  fi

  PJRT_FAIL_REASON="${PJRT_FAIL_REASON:-neither libtpu.so + C++ compiler nor python3+jax available}"
  return 1
}

log "--- Live PJRT Logical Device & Chiplet Topology ---"
PJRT_FAIL_REASON=""
if probe_live_pjrt_topology; then
  log "STATUS: PJRT_TOPOLOGY_CAPTURED (pci_tpu_functions=${tpu_count})"
else
  log "STATUS: PJRT_TOPOLOGY_UNAVAILABLE (${PJRT_FAIL_REASON})"
  if [[ "${ALLOW_NO_PJRT}" -ne 1 ]]; then
    log "ERROR: Live PJRT topology probe failed; re-run with LIBTPU_PATH=/path/to/libtpu.so (or pass --allow-no-pjrt to record PCI-only survey)."
    exit 1
  fi
fi

echo "Survey complete: ${REPORT}"
