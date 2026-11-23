#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Day-1 Ops & Router smoke test: CDI v1.1 + tpu-exporter + router c=16..128 SSE concurrency sweep.
set -euo pipefail

DRY_RUN=0
MOCK_MODE=0
OUT_DIR=""
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-}"
CONCURRENCIES="${CONCURRENCIES:-16,32,64,128}"
TIER_TIMEOUT_S="${TIER_TIMEOUT_S:-30}"
CDI_VERSION="${CDI_VERSION:-1.1.0}"
RUNTIME_METRICS_ADDR="${RUNTIME_METRICS_ADDR:-127.0.0.1:8431}"

usage() {
  cat <<'EOF'
Usage: 03_ops_and_router_smoke.sh [--out-dir DIR] [--backend-url URL] [--model NAME]
                                  [--concurrencies 16,32,64,128] [--tier-timeout-s SEC]
                                  [--cdi-version 1.1.0] [--runtime-metrics-addr HOST:PORT]
                                  [--dry-run] [--mock]

Runs bounded Ironwood Day-1 checks for the FreeTPU ops & serving control plane:
  1. CNCF CDI v1.1 spec generation & host validation (ops/tpu-cdi/freetpu_cdi.py generate + validate)
  2. Prometheus telemetry one-shot capture (ops/tpu-exporter/tpu_exporter.py --once) validating
     actual non-comment Prometheus samples (freetpu_tpu_devices_discovered >= 1 and explicit
     per-collector verification for direct_grpc, shm, and aer; live runs with missing collectors
     report EXPORTER_SMOKE_STATUS=PARTIAL_UNVERIFIED)
  3. FreeTPU router config check (--check-config) + bounded SSE concurrency sweep (c=16,32,64,128)
     with an explicit per-tier wall-clock deadline (--tier-timeout-s), rejecting SSE error events
     and asserting active_inflight == 0 after every tier. In --mock mode, a barrier-synchronized
     backend enforces hard peak_inflight >= c; in live mode, observed peak_inflight is sampled
     and tiers that complete without observing peak_inflight >= c report COMPLETED_UNVERIFIED_PEAK.

In --mock mode (or on non-Linux hosts), creates a synthetic tpu7x-standard-4t sysfs/devfs tree for
CDI/exporter and runs the router sweep against a local loopback barrier-synchronized OpenAI SSE backend.
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
    --backend-url)
      require_arg "$#" "$1"
      BACKEND_URL="$2"
      shift 2
      ;;
    --model)
      require_arg "$#" "$1"
      MODEL_NAME="$2"
      shift 2
      ;;
    --concurrencies)
      require_arg "$#" "$1"
      CONCURRENCIES="$2"
      shift 2
      ;;
    --tier-timeout-s)
      require_arg "$#" "$1"
      TIER_TIMEOUT_S="$2"
      shift 2
      ;;
    --cdi-version)
      require_arg "$#" "$1"
      CDI_VERSION="$2"
      shift 2
      ;;
    --runtime-metrics-addr)
      require_arg "$#" "$1"
      RUNTIME_METRICS_ADDR="$2"
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
LOG="${OUT_DIR}/03_ops_and_router_smoke.txt"

log() {
  echo "$*" | tee -a "${LOG}"
}

: > "${LOG}"
log "=== FreeTPU Ironwood (tpu7x) Day-1 Ops & Router Smoke ==="
log "timestamp_utc=${TIMESTAMP} mock_mode=${MOCK_MODE} cdi_version=${CDI_VERSION} concurrencies=${CONCURRENCIES} tier_timeout_s=${TIER_TIMEOUT_S} backend_url=${BACKEND_URL}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "[DRY-RUN] Would run freetpu_cdi.py generate/validate (--cdi-version ${CDI_VERSION}), tpu_exporter.py --once (validating non-comment Prometheus samples and direct_grpc/shm/aer collectors), and freetpu_router.py --check-config + SSE concurrency sweep (${CONCURRENCIES}, tier_timeout_s=${TIER_TIMEOUT_S}, barrier peak_inflight>=c in --mock / observed peak_inflight in live mode, rejecting SSE error events, asserting active_inflight==0)."
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  log "ERROR: python3 is required for CDI, tpu-exporter, and router checks."
  exit 1
fi

TMP_ROOT="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_ROOT}"
}
trap cleanup EXIT

CDI_SCRIPT="${CDI_SCRIPT:-${REPO_ROOT}/ops/tpu-cdi/freetpu_cdi.py}"
EXPORTER_SCRIPT="${EXPORTER_SCRIPT:-${REPO_ROOT}/ops/tpu-exporter/tpu_exporter.py}"
ROUTER_SCRIPT="${ROUTER_SCRIPT:-${REPO_ROOT}/router/freetpu_router.py}"
CDI_SPEC="${OUT_DIR}/freetpu-cdi-${CDI_VERSION}.json"

EFFECTIVE_MOCK="${MOCK_MODE}"
if [[ "$(uname -s)" != "Linux" ]]; then
  EFFECTIVE_MOCK=1
fi

# --- Part 1 & 2: CNCF CDI v1.1 + tpu-exporter --once ---
CDI_EXTRA_ARGS=()
EXPORTER_EXTRA_ARGS=()
if [[ "${EFFECTIVE_MOCK}" -eq 1 ]]; then
  log "--- [MOCK] Building synthetic tpu7x-standard-4t sysfs/devfs/procfs tree for CDI v${CDI_VERSION} & tpu-exporter ---"
  python3 - "${TMP_ROOT}/mock_host" <<'PYEOF'
import os
import struct
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
sysfs_root = root / "sys"
dev_root = root / "dev"
proc_root = root / "proc"
shm_root = root / "shm"

pci_devices = sysfs_root / "bus" / "pci" / "devices"
pci_devices.mkdir(parents=True, exist_ok=True)
iommu_groups = sysfs_root / "kernel" / "iommu_groups"
iommu_groups.mkdir(parents=True, exist_ok=True)
vfio_dir = dev_root / "vfio"
vfio_dir.mkdir(parents=True, exist_ok=True)
proc_root.mkdir(parents=True, exist_ok=True)
shm_root.mkdir(parents=True, exist_ok=True)

# VFIO container node + sysfs major:minor
(vfio_dir / "vfio").write_text("", encoding="utf-8")
misc_vfio = sysfs_root / "class" / "misc" / "vfio" / "dev"
misc_vfio.parent.mkdir(parents=True, exist_ok=True)
misc_vfio.write_text("10:196\n", encoding="utf-8")

drv = sysfs_root / "bus" / "pci" / "drivers" / "vfio-pci"
drv.mkdir(parents=True, exist_ok=True)

for idx in range(4):
    bdf = f"0000:00:0{idx + 4}.0"
    grp_id = str(idx)
    (vfio_dir / grp_id).write_text("", encoding="utf-8")
    grp_dev = sysfs_root / "class" / "vfio" / grp_id / "dev"
    grp_dev.parent.mkdir(parents=True, exist_ok=True)
    grp_dev.write_text(f"242:{idx}\n", encoding="utf-8")

    d = pci_devices / bdf
    d.mkdir(parents=True, exist_ok=True)
    (d / "vendor").write_text("0x1ae0\n", encoding="utf-8")
    (d / "device").write_text("0x0076\n", encoding="utf-8")
    (d / "class").write_text("0xff0000\n", encoding="utf-8")
    (d / "numa_node").write_text("0\n", encoding="utf-8")
    (d / "aer_dev_correctable").write_text("RxErr 0\nBadTLP 0\nTOTAL_ERR_COR 0\n", encoding="utf-8")
    (d / "aer_dev_fatal").write_text("DLP 0\nTOTAL_ERR_FATAL 0\n", encoding="utf-8")

    grp_path = iommu_groups / grp_id
    grp_devs = grp_path / "devices"
    grp_devs.mkdir(parents=True, exist_ok=True)
    os.symlink(str(d), grp_devs / bdf)
    os.symlink(str(grp_path), d / "iommu_group")
    os.symlink(str(drv), d / "driver")

# Write a clean v1 SeqLock SHM telemetry snapshot for device 0 (0000:00:04.0)
# Layout matches ops/tpu-exporter/freetpu_metrics_shm.h (56-byte struct freetpu_shm_metrics_v1)
os.chmod(shm_root, 0o750)
now_ns = time.time_ns()
shm_blob = struct.pack(
    "<IIQQQQQQ",
    0x55505446,       # FREETPU_METRICS_SHM_MAGIC ("FTPU")
    1,                # FREETPU_METRICS_SHM_VERSION
    2,                # even seqlock seq -> committed snapshot
    96 * 1024**3,     # hbm_allocated_bytes
    12 * 1024**3,     # hbm_used_bytes
    12345678,         # mxu_active_cycles
    8 * 1024**3,      # dma_bytes_transferred
    now_ns,           # timestamp_ns (CLOCK_REALTIME ns)
)
shm_file = shm_root / "freetpu_metrics_0000:00:04.0"
shm_file.write_bytes(shm_blob)
os.chmod(shm_file, 0o600)
PYEOF
  CDI_EXTRA_ARGS=(--sysfs-root "${TMP_ROOT}/mock_host/sys" --dev-root "${TMP_ROOT}/mock_host/dev")
  EXPORTER_EXTRA_ARGS=(
    --sysfs-root "${TMP_ROOT}/mock_host/sys"
    --dev-root "${TMP_ROOT}/mock_host/dev"
    --proc-root "${TMP_ROOT}/mock_host/proc"
    --shm-root "${TMP_ROOT}/mock_host/shm"
    --shm-trusted-uid "$(id -u)"
    --no-pid-scan
    --tpu-info-bin ""
    --runtime-metrics-addr ""
  )
else
  EXPORTER_EXTRA_ARGS=(--runtime-metrics-addr "${RUNTIME_METRICS_ADDR}")
fi

log "--- Step 1/3: CNCF CDI v${CDI_VERSION} generate + validate ---"
if [[ "${EFFECTIVE_MOCK}" -eq 1 ]]; then
  python3 "${CDI_SCRIPT}" generate \
    --cdi-version "${CDI_VERSION}" \
    --stdout \
    --allow-missing-vfio \
    --validate \
    "${CDI_EXTRA_ARGS[@]}" > "${CDI_SPEC}"

  python3 "${CDI_SCRIPT}" validate \
    "${CDI_SPEC}" 2>&1 | tee -a "${LOG}"
else
  python3 "${CDI_SCRIPT}" generate \
    --cdi-version "${CDI_VERSION}" \
    --output "${CDI_SPEC}" \
    --validate \
    "${CDI_EXTRA_ARGS[@]}" 2>&1 | tee -a "${LOG}"

  python3 "${CDI_SCRIPT}" validate \
    --check-host \
    "${CDI_EXTRA_ARGS[@]}" \
    "${CDI_SPEC}" 2>&1 | tee -a "${LOG}"
fi

python3 - "${CDI_SPEC}" "${CDI_VERSION}" <<'PYEOF' 2>&1 | tee -a "${LOG}"
import json
import sys

spec_path, expected_ver = sys.argv[1], sys.argv[2]
with open(spec_path, encoding="utf-8") as f:
    spec = json.load(f)
got_ver = spec.get("cdiVersion")
devices = spec.get("devices", [])
if got_ver != expected_ver:
    raise SystemExit(f"CDI version mismatch: expected={expected_ver} got={got_ver}")
if not devices:
    raise SystemExit("CDI spec contains 0 devices")
print(f"CDI_SMOKE_STATUS=PASS cdiVersion={got_ver} devices={len(devices)}")
PYEOF

log "--- Step 2/3: tpu-exporter --once (gRPC / SeqLock SHM / PCIe AER) ---"
EXPORTER_OUT="${OUT_DIR}/tpu_exporter_once.prom"
python3 "${EXPORTER_SCRIPT}" --once "${EXPORTER_EXTRA_ARGS[@]}" > "${EXPORTER_OUT}"
head -n 40 "${EXPORTER_OUT}" | tee -a "${LOG}"

python3 - "${EXPORTER_OUT}" "${EFFECTIVE_MOCK}" <<'PYEOF' 2>&1 | tee -a "${LOG}"
import math
import re
import sys

prom_path, is_mock = sys.argv[1], sys.argv[2] == "1"
sample_re = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)(?:\s+\d+)?$"
)

samples = []
with open(prom_path, encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = sample_re.match(line)
        if not m:
            continue
        name, labels, val_s = m.group(1), (m.group(2) or ""), m.group(3)
        try:
            val = float(val_s)
        except ValueError:
            continue
        samples.append((name, labels, val))

if not samples:
    raise SystemExit(f"tpu-exporter emitted 0 valid non-comment Prometheus samples ({prom_path})")

discovered_vals = [val for name, _lbl, val in samples if name == "freetpu_tpu_devices_discovered" and math.isfinite(val)]
if not discovered_vals:
    raise SystemExit("tpu-exporter missing non-comment freetpu_tpu_devices_discovered sample")
devices_discovered = int(discovered_vals[-1])
if devices_discovered < 1:
    raise SystemExit(f"tpu-exporter discovered {devices_discovered} TPU devices (expected >= 1)")

telemetry_source_re = re.compile(r'telemetry_source="([^"]*)"')
tpu7x_info_samples = 0
direct_grpc_samples = 0
tpu_info_cli_samples = 0
shm_info_samples = 0

for name, lbl, _v in samples:
    if name != "freetpu_tpu_info":
        continue
    if 'generation="tpu7x"' in lbl:
        tpu7x_info_samples += 1
    m_src = telemetry_source_re.search(lbl)
    src_tokens = {tok.strip() for tok in m_src.group(1).split("+") if tok.strip()} if m_src else set()
    if "libtpu-grpc" in src_tokens or "grpc" in src_tokens:
        direct_grpc_samples += 1
    elif "tpu-info" in src_tokens or "tpu-info-cli" in src_tokens:
        tpu_info_cli_samples += 1
    if "libopentpu-shm" in src_tokens:
        shm_info_samples += 1

shm_mxu_samples = sum(1 for name, _lbl, _v in samples if name == "freetpu_tpu_mxu_active_cycles_total")
shm_dma_samples = sum(1 for name, _lbl, _v in samples if name == "freetpu_tpu_dma_bytes_transferred_total")
shm_samples = shm_mxu_samples + shm_dma_samples + shm_info_samples
aer_corr_samples = sum(1 for name, _lbl, _v in samples if name == "freetpu_pcie_aer_correctable_total")
aer_fatal_samples = sum(1 for name, _lbl, _v in samples if name == "freetpu_pcie_aer_fatal_total")
aer_samples = aer_corr_samples + aer_fatal_samples

direct_grpc_status = "PASS" if direct_grpc_samples > 0 else "UNVERIFIED"
tpu_info_cli_status = "FALLBACK" if tpu_info_cli_samples > 0 else "NONE"
shm_status = "PASS" if (shm_mxu_samples > 0 and shm_dma_samples > 0) else "UNVERIFIED"
aer_status = "PASS" if (aer_corr_samples > 0 and aer_fatal_samples > 0) else "UNVERIFIED"

if is_mock:
    errors = []
    if devices_discovered != 4:
        errors.append(f"devices_discovered={devices_discovered} (expected 4)")
    if tpu7x_info_samples < 4:
        errors.append(f"tpu7x_info_samples={tpu7x_info_samples} (expected >= 4)")
    if shm_mxu_samples < 1 or shm_dma_samples < 1 or shm_info_samples < 1:
        errors.append(
            f"shm_samples incomplete (mxu={shm_mxu_samples} dma={shm_dma_samples} info={shm_info_samples})"
        )
    if aer_corr_samples < 4 or aer_fatal_samples < 4:
        errors.append(f"aer_samples incomplete (correctable={aer_corr_samples} fatal={aer_fatal_samples})")
    if errors:
        raise SystemExit("tpu-exporter mock sample verification failed: " + "; ".join(errors))
    exporter_status = "PASS"
else:
    if (direct_grpc_status == "PASS" or shm_status == "PASS") and aer_status == "PASS":
        exporter_status = "PASS"
    else:
        exporter_status = "PARTIAL_UNVERIFIED"

print(
    f"EXPORTER_SMOKE_STATUS={exporter_status} devices_discovered={devices_discovered} "
    f"tpu7x_info_samples={tpu7x_info_samples} "
    f"direct_grpc={direct_grpc_status}({direct_grpc_samples}) "
    f"tpu_info_cli={tpu_info_cli_status}({tpu_info_cli_samples}) "
    f"shm={shm_status}({shm_samples}) aer={aer_status}({aer_samples}) "
    f"total_samples={len(samples)} prom_file={prom_path}"
)
PYEOF

# --- Step 3/3: FreeTPU Router --check-config + c=16..128 SSE concurrency sweep ---
log "--- Step 3/3: router/freetpu_router.py --check-config + c=${CONCURRENCIES} SSE concurrency sweep (tier_timeout_s=${TIER_TIMEOUT_S}) ---"
python3 - "${ROUTER_SCRIPT}" "${TMP_ROOT}" "${EFFECTIVE_MOCK}" "${BACKEND_URL}" "${MODEL_NAME}" "${CONCURRENCIES}" "${TIER_TIMEOUT_S}" <<'PYEOF' 2>&1 | tee -a "${LOG}"
import concurrent.futures
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

router_script = sys.argv[1]
tmp_root = sys.argv[2]
is_mock = sys.argv[3] == "1"
backend_url = sys.argv[4].rstrip("/")
model_name = sys.argv[5].strip()
concurrencies = [int(x.strip()) for x in sys.argv[6].split(",") if x.strip()]
tier_timeout_s = float(sys.argv[7])
if not concurrencies or any(c <= 0 for c in concurrencies):
    raise SystemExit(f"Invalid concurrencies list: {sys.argv[6]!r}")
if tier_timeout_s <= 0:
    raise SystemExit(f"Invalid tier_timeout_s: {tier_timeout_s!r}")


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _MockBackendServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 512

    def __init__(self, server_address, handler_cls):
        super().__init__(server_address, handler_cls)
        self.tier_lock = threading.Lock()
        self.target_conc = 0
        self.arrived = 0
        self.in_flight = 0
        self.peak_backend_inflight = 0
        self.all_arrived = threading.Event()
        self.release_tier = threading.Event()
        self.release_tier.set()
        self.barrier_timeout_s = 15.0

    def arm_tier(self, target_conc: int, barrier_timeout_s: float) -> None:
        with self.tier_lock:
            self.target_conc = target_conc
            self.arrived = 0
            self.in_flight = 0
            self.peak_backend_inflight = 0
            self.barrier_timeout_s = barrier_timeout_s
            self.all_arrived.clear()
            self.release_tier.clear()


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
        elif self.path in ("/v1/models", "/models"):
            body = json.dumps({
                "object": "list",
                "data": [{"id": "ironwood-mock-model", "object": "model"}],
            }).encode("utf-8")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        req = json.loads(raw.decode("utf-8") or "{}")
        req_model = req.get("model") or "ironwood-mock-model"
        stream = bool(req.get("stream", False))

        if stream:
            chunk1 = (
                "data: "
                + json.dumps({
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
                })
                + "\n\n"
            ).encode("utf-8")
            chunk2 = (
                "data: "
                + json.dumps({
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
                })
                + "\n\ndata: [DONE]\n\n"
            ).encode("utf-8")

            srv = self.server
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(chunk1) + len(chunk2)))
            self.end_headers()
            self.wfile.write(chunk1)
            self.wfile.flush()

            with srv.tier_lock:
                srv.in_flight += 1
                srv.arrived += 1
                if srv.in_flight > srv.peak_backend_inflight:
                    srv.peak_backend_inflight = srv.in_flight
                target = srv.target_conc
                if target > 0 and srv.arrived >= target:
                    srv.all_arrived.set()
                barrier_timeout = srv.barrier_timeout_s

            try:
                if target > 0:
                    srv.release_tier.wait(timeout=barrier_timeout)
                self.wfile.write(chunk2)
                self.wfile.flush()
            finally:
                with srv.tier_lock:
                    srv.in_flight = max(0, srv.in_flight - 1)
        else:
            body = json.dumps({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1700000000,
                "model": req_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


mock_server = None
mock_thread = None
router_proc = None
router_log = None

try:
    if is_mock:
        mock_port = pick_free_port()
        mock_server = _MockBackendServer(("127.0.0.1", mock_port), _MockOpenAIHandler)
        mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
        mock_thread.start()
        backend_url = f"http://127.0.0.1:{mock_port}/v1"
        if not model_name:
            model_name = "ironwood-mock-model"
        print(f"[MOCK] Started barrier-synchronized OpenAI SSE mock backend at {backend_url} model={model_name}")
    else:
        try:
            with urllib.request.urlopen(f"{backend_url}/models", timeout=5.0) as resp:
                models_doc = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise SystemExit(
                f"Live backend {backend_url}/models unreachable ({exc}); "
                "start llama-server/vllm-tpu or pass --mock."
            )
        if not model_name:
            data = models_doc.get("data") or []
            if not data or not isinstance(data[0], dict) or not data[0].get("id"):
                raise SystemExit(f"Could not auto-discover model from {backend_url}/models; pass --model NAME.")
            model_name = str(data[0]["id"])
        print(f"[LIVE] Verified backend {backend_url} model={model_name}")

    router_port = pick_free_port()
    router_cfg_path = os.path.join(tmp_root, "router-ironwood-smoke.json")
    router_cfg = {
        "host": "127.0.0.1",
        "port": router_port,
        "policy": "round_robin",
        "health_poll_interval_s": 30.0,
        "failure_cooldown_s": 5.0,
        "request_timeout_s": max(60.0, tier_timeout_s * 2.0),
        "probe_timeout_s": 2.5,
        "auth": {"type": "allow_all"},
        "usage_meter": {"type": "noop"},
        "scaler": {"type": "noop"},
        "backends": [
            {
                "id": "ironwood-tpu7x-backend",
                "type": "cloud-tpu-v7x",
                "local": is_mock,
                "spot": False,
                "base_url": backend_url,
                "health_path": "/health",
                "models": [model_name],
                "discover_models": True,
                "max_inflight": max(256, max(concurrencies) * 2),
                "timeout_s": max(60.0, tier_timeout_s * 2.0),
                "priority": 10,
            }
        ],
    }
    with open(router_cfg_path, "w", encoding="utf-8") as f:
        json.dump(router_cfg, f, indent=2)

    chk = subprocess.run(
        [sys.executable, router_script, "--config", router_cfg_path, "--check-config"],
        capture_output=True,
        text=True,
        check=False,
    )
    if chk.returncode != 0:
        raise SystemExit(f"freetpu_router.py --check-config failed (rc={chk.returncode}): {chk.stdout} {chk.stderr}")
    print(f"ROUTER_CHECK_CONFIG=PASS {chk.stdout.strip()}")

    router_log_path = os.path.join(tmp_root, "router.stderr.log")
    router_log = open(router_log_path, "w", encoding="utf-8")
    router_proc = subprocess.Popen(
        [sys.executable, router_script, "--config", router_cfg_path, "--no-initial-probe"],
        stdout=subprocess.DEVNULL,
        stderr=router_log,
    )

    deadline = time.monotonic() + 10.0
    healthy = False
    while time.monotonic() < deadline:
        if router_proc.poll() is not None:
            router_log.close()
            err_txt = open(router_log_path, encoding="utf-8").read()
            raise SystemExit(f"freetpu_router.py exited early (rc={router_proc.returncode}): {err_txt}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{router_port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except Exception:
            time.sleep(0.1)
    if not healthy:
        raise SystemExit("Timed out waiting for freetpu_router.py /health")

    req_body = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "max_tokens": 8,
        "stream": True,
    }).encode("utf-8")

    def fire_sse_request(_idx: int, tier_deadline: float) -> float:
        t0 = time.monotonic()
        rem = tier_deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError(f"Tier wall-clock deadline ({tier_timeout_s}s) exceeded before connect")
        conn = http.client.HTTPConnection("127.0.0.1", router_port, timeout=rem)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=req_body,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            rem = tier_deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"Tier wall-clock deadline ({tier_timeout_s}s) exceeded awaiting headers")
            if conn.sock is not None:
                conn.sock.settimeout(rem)
            resp = conn.getresponse()
            buf = bytearray()
            while True:
                rem = tier_deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError(f"Tier wall-clock deadline ({tier_timeout_s}s) exceeded mid-stream")
                if conn.sock is not None:
                    conn.sock.settimeout(rem)
                try:
                    chunk = resp.read1(4096)
                except (socket.timeout, TimeoutError) as exc:
                    raise TimeoutError(
                        f"Tier wall-clock deadline ({tier_timeout_s}s) exceeded reading SSE chunk"
                    ) from exc
                if not chunk:
                    break
                buf.extend(chunk)
            data = buf.decode("utf-8", errors="replace")
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {data[:200]}")
            validate_sse_response(data)
            return (time.monotonic() - t0) * 1000.0
        finally:
            conn.close()

    def validate_sse_response(data: str) -> None:
        saw_done = False
        valid_chunks = 0
        for raw_line in data.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                ev_name = line[6:].strip().lower()
                if ev_name == "error":
                    raise RuntimeError(f"SSE stream emitted event: error ({data[:200]})")
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                saw_done = True
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"SSE stream emitted invalid JSON payload: {payload[:200]}") from exc
            if isinstance(obj, dict) and ("error" in obj or obj.get("object") == "error"):
                raise RuntimeError(f"SSE stream emitted error payload: {payload[:200]}")
            valid_chunks += 1
        if not saw_done:
            raise RuntimeError(f"Missing SSE [DONE] sentinel: {data[:200]}")
        if valid_chunks < 1:
            raise RuntimeError(f"SSE stream completed with [DONE] but 0 valid data chunks: {data[:200]}")

    def read_active_inflight() -> tuple[int, int]:
        with urllib.request.urlopen(f"http://127.0.0.1:{router_port}/health", timeout=5.0) as resp:
            health_doc = json.loads(resp.read().decode("utf-8"))
        health_inf = int(health_doc.get("active_inflight", -1))

        with urllib.request.urlopen(f"http://127.0.0.1:{router_port}/metrics", timeout=5.0) as resp:
            metrics_txt = resp.read().decode("utf-8")
        prom_sum = 0
        saw_metric = False
        for line in metrics_txt.splitlines():
            if line.startswith("freetpu_router_active_inflight"):
                saw_metric = True
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    prom_sum += int(float(parts[1]))
        if not saw_metric:
            raise RuntimeError("freetpu_router_active_inflight metric missing from /metrics")
        return health_inf, prom_sum

    all_tiers_verified_peak = True
    for conc in concurrencies:
        t_level = time.monotonic()
        tier_deadline = t_level + tier_timeout_s
        latencies_ms = []
        errors = []
        peak_state = {"router": 0, "prom": 0}
        backend_peak = 0

        if is_mock and mock_server is not None:
            mock_server.arm_tier(conc, barrier_timeout_s=tier_timeout_s)

        stop_sampler = threading.Event()
        sampler_thread = None

        def _live_sampler(state=peak_state, stop_ev=stop_sampler) -> None:
            while not stop_ev.is_set():
                try:
                    hi, pi = read_active_inflight()
                    if hi > state["router"]:
                        state["router"] = hi
                    if pi > state["prom"]:
                        state["prom"] = pi
                except Exception:
                    pass
                stop_ev.wait(0.025)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=conc)
        try:
            futs = [pool.submit(fire_sse_request, i, tier_deadline) for i in range(conc)]
            if is_mock and mock_server is not None:
                try:
                    rem = max(0.1, tier_deadline - time.monotonic())
                    reached = mock_server.all_arrived.wait(timeout=rem)
                    hi, pi = read_active_inflight()
                    peak_state["router"] = max(peak_state["router"], hi)
                    peak_state["prom"] = max(peak_state["prom"], pi)
                    with mock_server.tier_lock:
                        backend_peak = mock_server.peak_backend_inflight
                    if not reached:
                        errors.append(
                            f"Concurrency barrier timed out: arrived={mock_server.arrived}/{conc} "
                            f"backend_peak={backend_peak} router_peak={peak_state['router']}"
                        )
                finally:
                    mock_server.release_tier.set()
            else:
                sampler_thread = threading.Thread(target=_live_sampler, daemon=True)
                sampler_thread.start()

            wait_rem = max(0.0, tier_deadline - time.monotonic())
            done, not_done = concurrent.futures.wait(futs, timeout=wait_rem)
            for fut in done:
                try:
                    latencies_ms.append(fut.result())
                except Exception as exc:
                    errors.append(str(exc))
            if not_done:
                for fut in not_done:
                    fut.cancel()
                errors.append(
                    f"Tier wall-clock deadline ({tier_timeout_s}s) exceeded: {len(not_done)}/{conc} streams still open"
                )
        finally:
            if is_mock and mock_server is not None:
                mock_server.release_tier.set()
            stop_sampler.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=1.0)
            pool.shutdown(wait=False, cancel_futures=True)

        wall_ms = (time.monotonic() - t_level) * 1000.0
        if errors:
            raise SystemExit(
                f"ROUTER_SWEEP c={conc} FAILED: {len(errors)} error(s) (first: {errors[0]})"
            )

        peak_router_inflight = peak_state["router"]
        peak_prom_inflight = peak_state["prom"]
        if is_mock:
            if backend_peak < conc or peak_router_inflight < conc or peak_prom_inflight < conc:
                raise SystemExit(
                    f"ROUTER_SWEEP c={conc} UNDER-CONCURRENT: "
                    f"backend_peak={backend_peak} router_peak={peak_router_inflight} "
                    f"prom_peak={peak_prom_inflight} (expected >= {conc})"
                )

        # Poll briefly for final SSE teardown accounting on the server thread
        settle_deadline = time.monotonic() + 2.0
        health_inf, prom_inf = read_active_inflight()
        while (health_inf != 0 or prom_inf != 0) and time.monotonic() < settle_deadline:
            time.sleep(0.05)
            health_inf, prom_inf = read_active_inflight()

        if health_inf != 0 or prom_inf != 0:
            raise SystemExit(
                f"ROUTER_SWEEP c={conc} LEAK DETECTED: health.active_inflight={health_inf} "
                f"prometheus.freetpu_router_active_inflight={prom_inf}"
            )

        latencies_ms.sort()
        p50 = latencies_ms[len(latencies_ms) // 2]
        p95 = latencies_ms[min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.95))]
        if is_mock:
            tier_status = "PASS"
            peak_info = f"peak_inflight={peak_router_inflight} prom_peak={peak_prom_inflight} backend_peak={backend_peak}"
        else:
            observed_peak = max(peak_router_inflight, peak_prom_inflight)
            if observed_peak >= conc:
                tier_status = "PASS"
            else:
                tier_status = "COMPLETED_UNVERIFIED_PEAK"
                all_tiers_verified_peak = False
            peak_info = f"observed_peak_inflight={peak_router_inflight} prom_peak={peak_prom_inflight}"
        print(
            f"ROUTER_SWEEP c={conc} ok={len(latencies_ms)}/{conc} {peak_info} "
            f"wall_ms={wall_ms:.1f} p50_ms={p50:.1f} p95_ms={p95:.1f} "
            f"active_inflight={health_inf} prom_inflight={prom_inf} status={tier_status}"
        )

    router_overall = "PASS" if all_tiers_verified_peak else "COMPLETED_UNVERIFIED_PEAK"
    print(f"ROUTER_SMOKE_STATUS={router_overall}")
finally:
    if mock_server is not None:
        mock_server.release_tier.set()
    if router_proc is not None and router_proc.poll() is None:
        router_proc.terminate()
        try:
            router_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            router_proc.kill()
            router_proc.wait(timeout=5.0)
    if router_log is not None and not router_log.closed:
        router_log.close()
    if mock_server is not None:
        mock_server.shutdown()
        mock_server.server_close()
PYEOF

EXPORTER_FINAL="$(sed -n 's/^EXPORTER_SMOKE_STATUS=\([^[:space:]]*\).*/\1/p' "${LOG}" | tail -n 1)"
ROUTER_FINAL="$(sed -n 's/^ROUTER_SMOKE_STATUS=\([^[:space:]]*\).*/\1/p' "${LOG}" | tail -n 1)"
if [[ "${EXPORTER_FINAL}" == "PASS" && "${ROUTER_FINAL}" == "PASS" ]]; then
  log "OPS_AND_ROUTER_SMOKE_STATUS=PASS"
else
  log "OPS_AND_ROUTER_SMOKE_STATUS=PARTIAL_UNVERIFIED (exporter=${EXPORTER_FINAL:-UNKNOWN} router=${ROUTER_FINAL:-UNKNOWN})"
fi
echo "Smoke log written to: ${LOG}"
