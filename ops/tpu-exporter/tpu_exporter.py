#!/usr/bin/env python3
"""FreeTPU Prometheus Telemetry Exporter (DCGM-exporter equivalent for Cloud TPUs).

Discovers Google TPU PCI devices (vendor 0x1ae0) under /sys/bus/pci/devices/*
and their /dev/vfio/* IOMMU device nodes, scrapes live HBM usage, MXU duty
cycle, active PIDs, and chip health via `tpu-info` (with pure sysfs/procfs
fallback), and serves Prometheus text exposition format 0.0.4 on HTTP
`/metrics`, or prints a one-shot snapshot with `--once` / `--json`.

Zero third-party dependencies: uses Python standard library only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GOOGLE_PCI_VENDOR_ID = "0x1ae0"

# Known Google Cloud TPU PCI device IDs under vendor 0x1ae0.
# Values: (generation_slug, display_name, nominal_hbm_bytes)
TPU_PCI_DEVICES: Dict[str, Tuple[str, str, int]] = {
    "0x0027": ("v2-v3", "TPU v2/v3", 16 * 1024**3),
    "0x0056": ("v4lite", "TPU v4lite", 8 * 1024**3),
    "0x005e": ("v4", "TPU v4", 32 * 1024**3),
    "0x0062": ("v5p", "TPU v5p", 95 * 1024**3),
    "0x0063": ("v5e", "TPU v5e", 16 * 1024**3),
    "0x006e": ("v6e-pf", "TPU v6e (PF)", 32 * 1024**3),
    "0x006f": ("v6e", "TPU v6e (Trillium)", 32 * 1024**3),
}

_UNIT_MULTIPLIERS: Dict[str, int] = {
    "B": 1,
    "KIB": 1024,
    "KB": 1000,
    "MIB": 1024**2,
    "MB": 1000**2,
    "GIB": 1024**3,
    "GB": 1000**3,
    "TIB": 1024**4,
    "TB": 1000**4,
}

_MEM_PAIR_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")


@dataclass
class TpuTelemetry:
    chip_index: int
    pci_bdf: str
    vendor_id: str
    device_id: str
    generation: str
    model_name: str
    driver: str
    iommu_group: str
    vfio_device: str
    vfio_present: bool
    vfio_is_char: bool
    vfio_container_present: bool
    pci_enabled: bool
    numa_node: Optional[int]
    healthy: bool
    hbm_used_bytes: Optional[int]
    hbm_total_bytes: Optional[int]
    hbm_nominal_bytes: int
    duty_cycle_percent: Optional[float]
    active_pids: List[int] = field(default_factory=list)
    telemetry_source: str = "sysfs-fallback"


@dataclass
class ExporterSnapshot:
    timestamp_unix: float
    scrape_duration_seconds: float
    tpu_info_available: bool
    tpu_info_error: Optional[str]
    devices: List[TpuTelemetry]


def _normalize_hex_id(raw: str) -> str:
    val = raw.strip().lower()
    if not val:
        return ""
    if not val.startswith("0x"):
        val = "0x" + val
    return val


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_symlink_name(path: Path) -> Optional[str]:
    try:
        if path.is_symlink():
            return os.path.basename(os.readlink(path))
    except OSError:
        return None
    # Allow plain text files in synthetic test fixtures if not a symlink.
    if path.is_file():
        return _read_text(path)
    return None


def _parse_bytes(value_str: str, unit_str: str) -> int:
    mult = _UNIT_MULTIPLIERS[unit_str.upper()]
    return int(round(float(value_str) * mult))


def _scan_proc_vfio_pids(
    proc_root: Path,
    vfio_paths: Sequence[str],
) -> Dict[str, List[int]]:
    """Find PIDs holding open file descriptors to any /dev/vfio/<group> node."""
    targets = {os.path.normpath(p): p for p in vfio_paths if p}
    result: Dict[str, List[int]] = {p: [] for p in vfio_paths if p}
    if not targets or not proc_root.is_dir():
        return result

    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return result

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd_link in fds:
            try:
                target = os.path.normpath(os.readlink(fd_link))
            except OSError:
                continue
            if target in targets:
                canonical = targets[target]
                if pid not in result[canonical]:
                    result[canonical].append(pid)

    for pids in result.values():
        pids.sort()
    return result


@dataclass
class _TpuInfoRow:
    chip_key: str
    chip_type: str = ""
    pids: List[int] = field(default_factory=list)
    hbm_used_bytes: Optional[int] = None
    hbm_total_bytes: Optional[int] = None
    duty_cycle_percent: Optional[float] = None


def parse_tpu_info_output(output: str) -> Dict[str, _TpuInfoRow]:
    """Parse `tpu-info` stdout tables (both Unicode box and ASCII pipe styles).

    Returns a mapping keyed by normalized chip key (e.g. '/dev/vfio/0' or '0').
    """
    rows: Dict[str, _TpuInfoRow] = {}
    section = "unknown"

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if "tpu chips" in lower:
            section = "chips"
            continue
        if "tpu runtime utilization" in lower or "runtime utilization" in lower:
            section = "utilization"
            continue

        normalized = (
            line.replace("│", "|")
            .replace("┃", "|")
            .replace("║", "|")
        )
        if "|" not in normalized:
            continue

        cells = [c.strip() for c in normalized.strip("|").split("|")]
        if not cells or all(not c or set(c) <= set("-=─━═+┼ ") for c in cells):
            continue

        first_lower = cells[0].lower()
        if first_lower in ("chip", "device", "tpu"):
            continue

        chip_key = cells[0]
        if not chip_key:
            continue

        row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))

        # Check if any cell has HBM usage (e.g. "1.50 GiB / 15.75 GiB")
        joined = " | ".join(cells[1:])
        mem_match = _MEM_PAIR_RE.search(joined)
        pct_match = _PCT_RE.search(joined)

        if mem_match:
            row.hbm_used_bytes = _parse_bytes(mem_match.group(1), mem_match.group(2))
            row.hbm_total_bytes = _parse_bytes(mem_match.group(3), mem_match.group(4))
        if pct_match:
            row.duty_cycle_percent = float(pct_match.group(1))

        if section == "chips" or (
            chip_key.startswith("/dev/vfio/") and not mem_match
        ):
            if len(cells) >= 2 and cells[1]:
                row.chip_type = cells[1]
            # PID is typically the last column in `TPU Chips`.
            if len(cells) >= 4:
                pid_cell = cells[-1]
                for tok in re.findall(r"\b[0-9]+\b", pid_cell):
                    pid = int(tok)
                    if pid not in row.pids:
                        row.pids.append(pid)

    return rows


def _run_tpu_info(
    tpu_info_bin: Optional[str],
    timeout_s: float,
) -> Tuple[bool, Optional[str], Dict[str, _TpuInfoRow]]:
    if not tpu_info_bin:
        return False, "tpu-info disabled", {}

    resolved = tpu_info_bin
    if os.path.sep not in tpu_info_bin:
        found = shutil.which(tpu_info_bin)
        if not found:
            return False, f"{tpu_info_bin} not found in PATH", {}
        resolved = found
    elif not os.path.exists(resolved):
        return False, f"{resolved} does not exist", {}

    try:
        proc = subprocess.run(
            [resolved],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), {}

    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if proc.returncode != 0:
        err_msg = combined.strip().splitlines()[-1] if combined.strip() else f"exit {proc.returncode}"
        return False, err_msg, {}

    parsed = parse_tpu_info_output(proc.stdout)
    return True, None, parsed


def discover_tpu_devices(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
) -> List[TpuTelemetry]:
    """Scan sysfs PCI devices for vendor 0x1ae0 and resolve VFIO nodes."""
    pci_dir = sysfs_root / "bus" / "pci" / "devices"
    if not pci_dir.is_dir():
        return []

    vfio_container_present = (dev_root / "vfio" / "vfio").exists()
    discovered: List[TpuTelemetry] = []

    for entry in sorted(pci_dir.iterdir(), key=lambda p: p.name):
        vendor_raw = _read_text(entry / "vendor")
        if not vendor_raw:
            continue
        vendor_id = _normalize_hex_id(vendor_raw)
        if vendor_id != GOOGLE_PCI_VENDOR_ID:
            continue

        device_id = _normalize_hex_id(_read_text(entry / "device") or "0x0000")
        gen, model_name, nominal_hbm = TPU_PCI_DEVICES.get(
            device_id, ("unknown", f"Google TPU ({device_id})", 0)
        )
        driver = _read_symlink_name(entry / "driver") or "unbound"
        iommu_group = _read_symlink_name(entry / "iommu_group") or ""
        enable_raw = _read_text(entry / "enable")
        pci_enabled = enable_raw != "0"

        numa_raw = _read_text(entry / "numa_node")
        numa_node: Optional[int] = None
        if numa_raw is not None:
            try:
                numa_node = int(numa_raw)
            except ValueError:
                numa_node = None

        vfio_path_obj = (dev_root / "vfio" / iommu_group) if iommu_group else None
        vfio_device = f"/dev/vfio/{iommu_group}" if iommu_group else ""
        vfio_present = bool(vfio_path_obj and vfio_path_obj.exists())
        vfio_is_char = False
        if vfio_path_obj and vfio_present:
            try:
                vfio_is_char = stat.S_ISCHR(vfio_path_obj.stat().st_mode)
            except OSError:
                vfio_is_char = False

        healthy = (
            driver == "vfio-pci"
            and bool(iommu_group)
            and vfio_present
            and vfio_container_present
            and pci_enabled
        )

        discovered.append(
            TpuTelemetry(
                chip_index=len(discovered),
                pci_bdf=entry.name,
                vendor_id=vendor_id,
                device_id=device_id,
                generation=gen,
                model_name=model_name,
                driver=driver,
                iommu_group=iommu_group,
                vfio_device=vfio_device,
                vfio_present=vfio_present,
                vfio_is_char=vfio_is_char,
                vfio_container_present=vfio_container_present,
                pci_enabled=pci_enabled,
                numa_node=numa_node,
                healthy=healthy,
                hbm_used_bytes=None,
                hbm_total_bytes=nominal_hbm if nominal_hbm > 0 else None,
                hbm_nominal_bytes=nominal_hbm,
                duty_cycle_percent=None,
            )
        )

    return discovered


def collect_snapshot(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    proc_root: Path = Path("/proc"),
    tpu_info_bin: Optional[str] = "tpu-info",
    tpu_info_timeout_s: float = 5.0,
) -> ExporterSnapshot:
    """Collect PCI/VFIO discovery, procfs PID handles, and `tpu-info` metrics."""
    t0 = time.monotonic()
    devices = discover_tpu_devices(sysfs_root=sysfs_root, dev_root=dev_root)

    # Scan /proc/<pid>/fd for open handles to <dev_root>/vfio/<group>.
    real_vfio_paths = [
        str(dev_root / "vfio" / d.iommu_group)
        for d in devices
        if d.iommu_group
    ]
    proc_pids = _scan_proc_vfio_pids(proc_root, real_vfio_paths)

    tpu_info_ok, tpu_info_err, parsed_rows = _run_tpu_info(
        tpu_info_bin=tpu_info_bin,
        timeout_s=tpu_info_timeout_s,
    )

    for dev in devices:
        dev_real_vfio = str(dev_root / "vfio" / dev.iommu_group) if dev.iommu_group else ""
        pids = list(proc_pids.get(dev_real_vfio, []))

        matched_rows: List[_TpuInfoRow] = []
        candidate_keys = [
            dev.vfio_device,
            str(dev.chip_index),
        ]
        if dev.iommu_group and dev.iommu_group != str(dev.chip_index):
            candidate_keys.append(dev.iommu_group)

        for key in candidate_keys:
            if key and key in parsed_rows:
                matched_rows.append(parsed_rows[key])

        if matched_rows:
            dev.telemetry_source = "tpu-info"
            for row in matched_rows:
                for p in row.pids:
                    if p not in pids:
                        pids.append(p)
                if row.hbm_used_bytes is not None:
                    dev.hbm_used_bytes = row.hbm_used_bytes
                if row.hbm_total_bytes is not None:
                    dev.hbm_total_bytes = row.hbm_total_bytes
                if row.duty_cycle_percent is not None:
                    dev.duty_cycle_percent = row.duty_cycle_percent
        elif tpu_info_ok:
            # tpu-info succeeded (e.g., returned empty utilization when idle).
            dev.telemetry_source = "tpu-info"
            if dev.hbm_used_bytes is None:
                dev.hbm_used_bytes = 0
            if dev.duty_cycle_percent is None:
                dev.duty_cycle_percent = 0.0

        dev.active_pids = sorted(pids)

    elapsed = time.monotonic() - t0
    return ExporterSnapshot(
        timestamp_unix=time.time(),
        scrape_duration_seconds=elapsed,
        tpu_info_available=tpu_info_ok,
        tpu_info_error=tpu_info_err,
        devices=devices,
    )


def _esc_label(value: Any) -> str:
    s = str(value)
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt_labels(labels: Dict[str, Any]) -> str:
    if not labels:
        return ""
    pairs = [f'{k}="{_esc_label(v)}"' for k, v in labels.items()]
    return "{" + ",".join(pairs) + "}"


def _fmt_float(value: float) -> str:
    if value == 0.0:
        return "0"
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def format_prometheus_metrics(snapshot: ExporterSnapshot) -> str:
    """Render an ExporterSnapshot in Prometheus text exposition format 0.0.4."""
    lines: List[str] = []

    def add_metric(
        name: str,
        help_text: str,
        mtype: str,
        samples: Iterable[Tuple[Dict[str, Any], str]],
    ) -> None:
        sample_list = list(samples)
        if not sample_list:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        for labels, value in sample_list:
            lines.append(f"{name}{_fmt_labels(labels)} {value}")

    add_metric(
        "freetpu_exporter_up",
        "Whether the FreeTPU exporter scrape succeeded (1 = up).",
        "gauge",
        [({}, "1")],
    )
    add_metric(
        "freetpu_tpu_devices_discovered",
        "Total Google TPU PCI devices (vendor 0x1ae0) discovered in sysfs.",
        "gauge",
        [({}, str(len(snapshot.devices)))],
    )
    add_metric(
        "freetpu_tpu_scrape_duration_seconds",
        "Time taken to scrape TPU PCI/VFIO and telemetry sources in seconds.",
        "gauge",
        [({}, _fmt_float(snapshot.scrape_duration_seconds))],
    )
    add_metric(
        "freetpu_tpu_info_available",
        "Whether the tpu-info CLI returned telemetry cleanly (1 = yes, 0 = fallback).",
        "gauge",
        [({}, "1" if snapshot.tpu_info_available else "0")],
    )

    info_samples: List[Tuple[Dict[str, Any], str]] = []
    health_samples: List[Tuple[Dict[str, Any], str]] = []
    hbm_used_samples: List[Tuple[Dict[str, Any], str]] = []
    hbm_total_samples: List[Tuple[Dict[str, Any], str]] = []
    hbm_ratio_samples: List[Tuple[Dict[str, Any], str]] = []
    duty_samples: List[Tuple[Dict[str, Any], str]] = []
    pid_count_samples: List[Tuple[Dict[str, Any], str]] = []
    pid_active_samples: List[Tuple[Dict[str, Any], str]] = []

    for d in snapshot.devices:
        base_labels = {
            "chip": str(d.chip_index),
            "pci_bdf": d.pci_bdf,
            "device_id": d.device_id,
            "generation": d.generation,
            "vfio_device": d.vfio_device or "none",
        }
        info_labels = {
            **base_labels,
            "vendor": d.vendor_id,
            "model": d.model_name,
            "driver": d.driver,
            "iommu_group": d.iommu_group or "none",
            "telemetry_source": d.telemetry_source,
        }
        info_samples.append((info_labels, "1"))
        health_samples.append((base_labels, "1" if d.healthy else "0"))

        if d.hbm_used_bytes is not None:
            hbm_used_samples.append((base_labels, str(d.hbm_used_bytes)))
        if d.hbm_total_bytes is not None:
            hbm_total_samples.append((base_labels, str(d.hbm_total_bytes)))
        if (
            d.hbm_used_bytes is not None
            and d.hbm_total_bytes is not None
            and d.hbm_total_bytes > 0
        ):
            ratio = d.hbm_used_bytes / float(d.hbm_total_bytes)
            hbm_ratio_samples.append((base_labels, _fmt_float(ratio)))
        if d.duty_cycle_percent is not None:
            duty_samples.append((base_labels, _fmt_float(d.duty_cycle_percent)))

        pid_count_samples.append((base_labels, str(len(d.active_pids))))
        for pid in d.active_pids:
            pid_active_samples.append(({**base_labels, "pid": str(pid)}, "1"))

    add_metric(
        "freetpu_tpu_info",
        "Static inventory metadata for each discovered TPU device.",
        "gauge",
        info_samples,
    )
    add_metric(
        "freetpu_tpu_health_status",
        "TPU chip readiness (1 = vfio-pci bound, VFIO nodes present, PCI enabled).",
        "gauge",
        health_samples,
    )
    add_metric(
        "freetpu_tpu_hbm_used_bytes",
        "TPU High Bandwidth Memory (HBM) currently allocated in bytes.",
        "gauge",
        hbm_used_samples,
    )
    add_metric(
        "freetpu_tpu_hbm_total_bytes",
        "Total TPU High Bandwidth Memory (HBM) capacity in bytes.",
        "gauge",
        hbm_total_samples,
    )
    add_metric(
        "freetpu_tpu_hbm_utilization_ratio",
        "Fraction of TPU HBM currently allocated (0.0 to 1.0).",
        "gauge",
        hbm_ratio_samples,
    )
    add_metric(
        "freetpu_tpu_duty_cycle_percent",
        "TPU MXU/TensorCore duty cycle utilization percentage (0 to 100).",
        "gauge",
        duty_samples,
    )
    add_metric(
        "freetpu_tpu_active_pids",
        "Number of active OS PIDs holding or utilizing this TPU chip.",
        "gauge",
        pid_count_samples,
    )
    add_metric(
        "freetpu_tpu_process_active",
        "Active process indicator per TPU chip and PID (1 = active).",
        "gauge",
        pid_active_samples,
    )

    return "\n".join(lines) + "\n"


def snapshot_to_json(snapshot: ExporterSnapshot) -> str:
    return json.dumps(asdict(snapshot), indent=2, sort_keys=False) + "\n"


class _ExporterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        sysfs_root: Path,
        dev_root: Path,
        proc_root: Path,
        tpu_info_bin: Optional[str],
        tpu_info_timeout_s: float,
    ) -> None:
        self.sysfs_root = sysfs_root
        self.dev_root = dev_root
        self.proc_root = proc_root
        self.tpu_info_bin = tpu_info_bin
        self.tpu_info_timeout_s = tpu_info_timeout_s
        self._scrape_lock = threading.Lock()
        super().__init__(server_address, request_handler_class)

    def scrape(self) -> ExporterSnapshot:
        with self._scrape_lock:
            return collect_snapshot(
                sysfs_root=self.sysfs_root,
                dev_root=self.dev_root,
                proc_root=self.proc_root,
                tpu_info_bin=self.tpu_info_bin,
                tpu_info_timeout_s=self.tpu_info_timeout_s,
            )


class _ExporterHandler(BaseHTTPRequestHandler):
    server: _ExporterServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep stdout clean when serving; log errors to stderr.
        if os.environ.get("FREETPU_EXPORTER_VERBOSE") == "1":
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send(self, status: int, content_type: str, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/metrics", "/metrics/"):
            snap = self.server.scrape()
            self._send(
                200,
                "text/plain; version=0.0.4; charset=utf-8",
                format_prometheus_metrics(snap),
            )
            return
        if path in ("/metrics.json", "/json"):
            snap = self.server.scrape()
            self._send(200, "application/json; charset=utf-8", snapshot_to_json(snap))
            return
        if path in ("/healthz", "/readyz", "/"):
            self._send(200, "application/json; charset=utf-8", '{"status":"ok"}\n')
            return
        self._send(404, "text/plain; charset=utf-8", "not found\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpu-exporter",
        description="FreeTPU Prometheus telemetry exporter for Google Cloud TPUs (0x1ae0 / VFIO).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP bind address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9400,
        help="HTTP listen port (default: 9400).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect metrics once, print to stdout, and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON snapshot (implies --once unless serving).",
    )
    parser.add_argument(
        "--sysfs-root",
        type=Path,
        default=Path("/sys"),
        help="Sysfs root path (default: /sys).",
    )
    parser.add_argument(
        "--dev-root",
        type=Path,
        default=Path("/dev"),
        help="Devfs root path (default: /dev).",
    )
    parser.add_argument(
        "--proc-root",
        type=Path,
        default=Path("/proc"),
        help="Procfs root path (default: /proc).",
    )
    parser.add_argument(
        "--tpu-info-bin",
        default=os.environ.get("FREETPU_EXPORTER_TPU_INFO_BIN", "tpu-info"),
        help="Path or binary name for tpu-info (empty string disables tpu-info).",
    )
    parser.add_argument(
        "--tpu-info-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds when invoking tpu-info (default: 5.0).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tpu_info_bin = args.tpu_info_bin if args.tpu_info_bin else None

    if args.once or args.json:
        snap = collect_snapshot(
            sysfs_root=args.sysfs_root,
            dev_root=args.dev_root,
            proc_root=args.proc_root,
            tpu_info_bin=tpu_info_bin,
            tpu_info_timeout_s=args.tpu_info_timeout,
        )
        if args.json:
            sys.stdout.write(snapshot_to_json(snap))
        else:
            sys.stdout.write(format_prometheus_metrics(snap))
        return 0

    server = _ExporterServer(
        (args.host, args.port),
        _ExporterHandler,
        sysfs_root=args.sysfs_root,
        dev_root=args.dev_root,
        proc_root=args.proc_root,
        tpu_info_bin=tpu_info_bin,
        tpu_info_timeout_s=args.tpu_info_timeout,
    )
    sys.stderr.write(
        f"freetpu tpu-exporter listening on http://{args.host}:{server.server_address[1]}/metrics\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
