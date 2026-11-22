#!/usr/bin/env python3
"""FreeTPU Prometheus Telemetry Exporter (DCGM-exporter equivalent for Cloud TPUs).

Discovers Google TPU PCI devices (vendor 0x1ae0) under /sys/bus/pci/devices/*
and their /dev/vfio/* IOMMU device nodes on VFIO-backed Cloud TPU VMs (v5e,
v5p, v6e Trillium), scrapes live HBM usage, MXU duty cycle, active PIDs, and
VFIO chip readiness via `tpu-info` (with pure sysfs/procfs fallback), and serves
Prometheus text exposition format 0.0.4 on HTTP `/metrics`, or prints a one-shot
snapshot with `--once` / `--json`.

Note: Legacy `/dev/accel*` Cloud TPU hosts (e.g. TPU v2/v3/v4 using the `accel`
driver instead of `vfio-pci`) are recognized in PCI inventory, but VFIO
readiness (`freetpu_tpu_health_status`) and `/dev/vfio/*` PID tracking are
scoped to VFIO-backed hosts.

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
import socket
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
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


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
    pid_telemetry_available: bool = True
    runtime_chip_id: Optional[str] = None
    telemetry_source: str = "sysfs-fallback"


@dataclass
class RuntimeTelemetry:
    runtime_chip: str
    mapped_pci_bdf: Optional[str]
    hbm_used_bytes: Optional[int]
    hbm_total_bytes: Optional[int]
    duty_cycle_percent: Optional[float]


@dataclass
class ExporterSnapshot:
    timestamp_unix: float
    scrape_duration_seconds: float
    tpu_info_available: bool
    tpu_info_error: Optional[str]
    pid_telemetry_available: bool
    devices: List[TpuTelemetry]
    runtime_utilization: List[RuntimeTelemetry] = field(default_factory=list)


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
) -> Tuple[Dict[str, List[int]], bool, int]:
    """Find PIDs holding open file descriptors to any /dev/vfio/<group> node.

    Returns (pids_by_vfio_path, pid_telemetry_accessible, unreadable_pid_dirs).
    Permission failures on /proc/<pid>/fd mean PID telemetry may be incomplete,
    so callers must not confuse missing reads with zero active workloads.
    """
    targets = {os.path.normpath(p): p for p in vfio_paths if p}
    result: Dict[str, List[int]] = {p: [] for p in vfio_paths if p}
    if not targets:
        return result, True, 0
    if not proc_root.is_dir():
        return result, False, 1

    try:
        entries = list(proc_root.iterdir())
    except PermissionError:
        return result, False, 1
    except OSError:
        return result, False, 1

    unreadable = 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except PermissionError:
            unreadable += 1
            continue
        except OSError:
            # Process may have exited concurrently; not an access denial.
            continue
        for fd_link in fds:
            try:
                target = os.path.normpath(os.readlink(fd_link))
            except PermissionError:
                unreadable += 1
                continue
            except OSError:
                continue
            if target in targets:
                canonical = targets[target]
                if pid not in result[canonical]:
                    result[canonical].append(pid)

    for pids in result.values():
        pids.sort()
    return result, unreadable == 0, unreadable


@dataclass
class _TpuInfoRow:
    chip_key: str
    chip_type: str = ""
    pids: List[int] = field(default_factory=list)
    hbm_used_bytes: Optional[int] = None
    hbm_total_bytes: Optional[int] = None
    duty_cycle_percent: Optional[float] = None
    in_chips_table: bool = False
    in_util_table: bool = False


def _split_table_cells(line: str) -> Optional[List[str]]:
    """Split a Rich Unicode or ASCII table row into cells without splitting inside `[...]` gauges.

    Upstream `tpu-info` renders bar gauges with literal `|` characters (e.g.
    `[|||||.....] 50.8%`). Splitting naively on every `|` corrupts column indices.
    """
    if any(ch in line for ch in ("│", "┃", "║")):
        trimmed = re.sub(r"^[│┃║]+|[│┃║]+$", "", line.strip())
        return [c.strip() for c in re.split(r"[│┃║]", trimmed)]

    if "|" not in line:
        return None

    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]

    cells: List[str] = []
    buf: List[str] = []
    bracket_depth = 0
    for ch in trimmed:
        if ch == "[":
            bracket_depth += 1
            buf.append(ch)
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            buf.append(ch)
        elif ch == "|" and bracket_depth == 0:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


def parse_tpu_info_output(output: str) -> Dict[str, _TpuInfoRow]:
    """Parse `tpu-info` stdout tables (both Unicode box and ASCII pipe styles).

    Supports both:
    - Modern `tpu-info` dashboard tables: `TPU Device Status` (Memory Usage +
      TensorCore Duty Cycle, ignoring Memory Bar Gauge / HBM Bandwidth Util
      percentages) and `TPU Process Monitor` (Device Path / VFIO Path + PID).
    - Classic `tpu-info` tables: `TPU Chips` and `TPU Runtime Utilization`,
      keeping standalone `TensorCore Utilization` tables isolated.
    """
    rows: Dict[str, _TpuInfoRow] = {}
    section = "unknown"
    col_map: Dict[str, int] = {}

    for raw_line in output.splitlines():
        line = _ANSI_RE.sub("", raw_line).strip()
        if not line:
            continue
        lower = line.lower()
        is_box_or_table = any(
            ch in line for ch in ("|", "│", "┃", "║", "┌", "┏", "└", "┗", "├", "┝", "+", "─", "━", "═")
        )
        if not is_box_or_table:
            col_map = {}
            if "tpu chips" in lower:
                section = "chips"
                continue
            if "tpu runtime utilization" in lower:
                section = "utilization"
                continue
            if "tpu device status" in lower:
                section = "device_status"
                continue
            if "tpu process monitor" in lower:
                section = "process_monitor"
                continue
            if "tensorcore" in lower or "tensor core" in lower or "utilization" in lower:
                section = "other"
                continue
            section = "other"
            continue

        cells = _split_table_cells(line)
        if not cells or all(not c or set(c) <= set("-=─━═+┼ ") for c in cells):
            continue

        first_lower = cells[0].lower()
        if first_lower in ("core", "tensorcore", "tensorcore id"):
            section = "other"
            col_map = {}
            continue
        if first_lower in ("chip", "device", "device id", "id", "tpu"):
            for idx, col_hdr in enumerate(cells):
                hdr = col_hdr.lower()
                if "memory usage" in hdr or "hbm usage" in hdr:
                    col_map["mem"] = idx
                elif "duty cycle" in hdr:
                    col_map["duty"] = idx
                elif "path" in hdr or "vfio" in hdr:
                    col_map["path"] = idx
                elif hdr == "pid":
                    col_map["pid"] = idx
            continue

        if section not in ("chips", "utilization", "device_status", "process_monitor"):
            continue

        chip_key = cells[0]
        if not chip_key:
            # Handle wrapped multi-line Rich headers where col 0 is blank.
            for idx, col_hdr in enumerate(cells):
                hdr = col_hdr.lower()
                if "mem" not in col_map and ("memory usage" in hdr or "hbm usage" in hdr):
                    col_map["mem"] = idx
                elif "duty" not in col_map and "duty cycle" in hdr:
                    col_map["duty"] = idx
                elif "path" not in col_map and ("path" in hdr or "vfio" in hdr):
                    col_map["path"] = idx
                elif "pid" not in col_map and hdr == "pid":
                    col_map["pid"] = idx
            continue

        if section == "utilization":
            row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))
            row.in_util_table = True
            mem_idx = col_map.get("mem")
            duty_idx = col_map.get("duty")
            mem_text = (
                cells[mem_idx]
                if mem_idx is not None and mem_idx < len(cells)
                else " | ".join(cells[1:])
            )
            duty_text = (
                cells[duty_idx]
                if duty_idx is not None and duty_idx < len(cells)
                else " | ".join(cells[1:])
            )
            mem_match = _MEM_PAIR_RE.search(mem_text)
            pct_match = _PCT_RE.search(duty_text)
            if mem_match:
                row.hbm_used_bytes = _parse_bytes(mem_match.group(1), mem_match.group(2))
                row.hbm_total_bytes = _parse_bytes(mem_match.group(3), mem_match.group(4))
            if pct_match:
                row.duty_cycle_percent = float(pct_match.group(1))
        elif section == "device_status":
            row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))
            row.in_util_table = True
            mem_idx = col_map.get("mem", 1)
            duty_idx = col_map.get("duty", 4)
            if mem_idx < len(cells):
                mem_match = _MEM_PAIR_RE.search(cells[mem_idx])
                if mem_match:
                    row.hbm_used_bytes = _parse_bytes(mem_match.group(1), mem_match.group(2))
                    row.hbm_total_bytes = _parse_bytes(mem_match.group(3), mem_match.group(4))
            if duty_idx < len(cells):
                pct_match = _PCT_RE.search(cells[duty_idx])
                if pct_match:
                    row.duty_cycle_percent = float(pct_match.group(1))
        elif section == "chips":
            row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))
            row.in_chips_table = True
            if len(cells) >= 2 and cells[1]:
                row.chip_type = cells[1]
            # PID is typically the last column in `TPU Chips`.
            pid_idx = col_map.get("pid", len(cells) - 1 if len(cells) >= 4 else -1)
            if 0 <= pid_idx < len(cells):
                pid_cell = cells[pid_idx]
                for tok in re.findall(r"\b[0-9]+\b", pid_cell):
                    pid = int(tok)
                    if pid not in row.pids:
                        row.pids.append(pid)
        elif section == "process_monitor":
            path_idx = col_map.get("path", 1)
            pid_idx = col_map.get("pid", 2)
            if path_idx >= len(cells):
                continue
            dev_path = cells[path_idx]
            if not dev_path or dev_path.lower() in ("n/a", "none"):
                continue
            row = rows.setdefault(dev_path, _TpuInfoRow(chip_key=dev_path))
            row.in_chips_table = True
            if pid_idx < len(cells):
                pid_cell = cells[pid_idx]
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

    run_env = dict(os.environ)
    run_env.setdefault("COLUMNS", "240")
    run_env.setdefault("NO_COLOR", "1")
    try:
        proc = subprocess.run(
            [resolved],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=run_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), {}

    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if proc.returncode != 0:
        err_msg = combined.strip().splitlines()[-1] if combined.strip() else f"exit {proc.returncode}"
        return False, err_msg, {}

    parsed = parse_tpu_info_output(proc.stdout)
    if not parsed:
        return False, "no recognized tpu-info tables in output", {}
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
        if device_id not in TPU_PCI_DEVICES:
            # Filter out non-TPU Google PCI functions (e.g. gVNIC 0x0042).
            continue
        gen, model_name, nominal_hbm = TPU_PCI_DEVICES[device_id]
        driver = _read_symlink_name(entry / "driver") or "unbound"
        iommu_group = _read_symlink_name(entry / "iommu_group") or ""
        enable_raw = _read_text(entry / "enable")
        pci_enabled = enable_raw == "1"
        runtime_chip_id = _read_text(entry / "chip_id") or None

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

        # On vfio-pci, PCI `enable` is 0 while idle (no process holds /dev/vfio/*);
        # readiness depends on driver binding and VFIO device nodes, not active opens.
        healthy = (
            driver == "vfio-pci"
            and bool(iommu_group)
            and vfio_present
            and vfio_container_present
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
                runtime_chip_id=runtime_chip_id,
            )
        )

    return discovered


def collect_snapshot(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    proc_root: Path = Path("/proc"),
    tpu_info_bin: Optional[str] = "tpu-info",
    tpu_info_timeout_s: float = 5.0,
    chip_id_map: Optional[Dict[str, str]] = None,
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
    proc_pids, proc_fd_accessible, _unreadable = _scan_proc_vfio_pids(
        proc_root, real_vfio_paths
    )

    tpu_info_ok, tpu_info_err, parsed_rows = _run_tpu_info(
        tpu_info_bin=tpu_info_bin,
        timeout_s=tpu_info_timeout_s,
    )
    util_rows = {k: r for k, r in parsed_rows.items() if r.in_util_table}
    mapped_util_keys: Dict[str, str] = {}
    overall_pid_available = True

    for dev in devices:
        dev_real_vfio = str(dev_root / "vfio" / dev.iommu_group) if dev.iommu_group else ""
        pids = list(proc_pids.get(dev_real_vfio, []))

        # Attach `TPU Chips` rows strictly by verified /dev/vfio/<group> or PCI BDF.
        chips_row: Optional[_TpuInfoRow] = None
        for key in (dev.vfio_device, dev.pci_bdf):
            if key and key in parsed_rows and parsed_rows[key].in_chips_table:
                chips_row = parsed_rows[key]
                break

        if chips_row is not None:
            dev.telemetry_source = "tpu-info"
            for p in chips_row.pids:
                if p not in pids:
                    pids.append(p)

        # Attach `TPU Runtime Utilization` rows strictly via verified hardware/map IDs
        # (vfio_device, pci_bdf, sysfs chip_id, or explicit chip_id_map). Never infer
        # runtime chip IDs from PCI enumeration order or IOMMU group numbers.
        verified_runtime_id = dev.runtime_chip_id
        if not verified_runtime_id and chip_id_map:
            verified_runtime_id = chip_id_map.get(dev.pci_bdf) or chip_id_map.get(dev.vfio_device)
            if verified_runtime_id is not None:
                verified_runtime_id = str(verified_runtime_id)
        if verified_runtime_id:
            dev.runtime_chip_id = verified_runtime_id

        util_candidates = [dev.vfio_device, dev.pci_bdf]
        if verified_runtime_id:
            util_candidates.append(verified_runtime_id)

        matched_util: Optional[_TpuInfoRow] = None
        for key in util_candidates:
            if key and key in util_rows:
                matched_util = util_rows[key]
                mapped_util_keys[key] = dev.pci_bdf
                break

        if matched_util is not None:
            if (
                matched_util.hbm_used_bytes is not None
                or matched_util.hbm_total_bytes is not None
                or matched_util.duty_cycle_percent is not None
            ):
                dev.telemetry_source = "tpu-info"
            if matched_util.hbm_used_bytes is not None:
                dev.hbm_used_bytes = matched_util.hbm_used_bytes
            if matched_util.hbm_total_bytes is not None:
                dev.hbm_total_bytes = matched_util.hbm_total_bytes
            if matched_util.duty_cycle_percent is not None:
                dev.duty_cycle_percent = matched_util.duty_cycle_percent

        # Upstream `tpu-info` also scans `/proc/*/fd/*` under the caller's privileges
        # and reports `PID=None` when another user's process is unreadable. Only mark
        # PID telemetry available when the `/proc/<pid>/fd` scan succeeded without
        # permission denials so incomplete scans never publish false zeros.
        dev.pid_telemetry_available = proc_fd_accessible
        if not dev.pid_telemetry_available:
            overall_pid_available = False

        dev.active_pids = sorted(pids)

    runtime_telemetry: List[RuntimeTelemetry] = [
        RuntimeTelemetry(
            runtime_chip=key,
            mapped_pci_bdf=mapped_util_keys.get(key),
            hbm_used_bytes=row.hbm_used_bytes,
            hbm_total_bytes=row.hbm_total_bytes,
            duty_cycle_percent=row.duty_cycle_percent,
        )
        for key, row in util_rows.items()
        if (
            row.hbm_used_bytes is not None
            or row.hbm_total_bytes is not None
            or row.duty_cycle_percent is not None
        )
    ]

    elapsed = time.monotonic() - t0
    return ExporterSnapshot(
        timestamp_unix=time.time(),
        scrape_duration_seconds=elapsed,
        tpu_info_available=tpu_info_ok,
        tpu_info_error=tpu_info_err,
        pid_telemetry_available=overall_pid_available,
        devices=devices,
        runtime_utilization=runtime_telemetry,
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
    pci_enabled_samples: List[Tuple[Dict[str, Any], str]] = []
    hbm_used_samples: List[Tuple[Dict[str, Any], str]] = []
    hbm_total_samples: List[Tuple[Dict[str, Any], str]] = []
    hbm_ratio_samples: List[Tuple[Dict[str, Any], str]] = []
    duty_samples: List[Tuple[Dict[str, Any], str]] = []
    pid_avail_samples: List[Tuple[Dict[str, Any], str]] = []
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
        pci_enabled_samples.append((base_labels, "1" if d.pci_enabled else "0"))
        pid_avail_samples.append((base_labels, "1" if d.pid_telemetry_available else "0"))

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

        if d.pid_telemetry_available:
            pid_count_samples.append((base_labels, str(len(d.active_pids))))
            for pid in d.active_pids:
                pid_active_samples.append(({**base_labels, "pid": str(pid)}, "1"))

    runtime_hbm_used: List[Tuple[Dict[str, Any], str]] = []
    runtime_hbm_total: List[Tuple[Dict[str, Any], str]] = []
    runtime_duty: List[Tuple[Dict[str, Any], str]] = []
    for rt in snapshot.runtime_utilization:
        rt_labels = {
            "runtime_chip": rt.runtime_chip,
            "pci_bdf": rt.mapped_pci_bdf or "unmapped",
        }
        if rt.hbm_used_bytes is not None:
            runtime_hbm_used.append((rt_labels, str(rt.hbm_used_bytes)))
        if rt.hbm_total_bytes is not None:
            runtime_hbm_total.append((rt_labels, str(rt.hbm_total_bytes)))
        if rt.duty_cycle_percent is not None:
            runtime_duty.append((rt_labels, _fmt_float(rt.duty_cycle_percent)))

    add_metric(
        "freetpu_tpu_info",
        "Static inventory metadata for each discovered TPU device.",
        "gauge",
        info_samples,
    )
    add_metric(
        "freetpu_tpu_health_status",
        "VFIO TPU chip readiness on VFIO-backed hosts (1 = vfio-pci bound and /dev/vfio nodes present; idle chips stay ready; 0 on non-VFIO /dev/accel* hosts).",
        "gauge",
        health_samples,
    )
    add_metric(
        "freetpu_tpu_pci_enabled",
        "PCI sysfs enable reference state (1 = actively enabled by open VFIO handle, 0 = idle).",
        "gauge",
        pci_enabled_samples,
    )
    add_metric(
        "freetpu_tpu_pid_telemetry_available",
        "Whether active PID telemetry could be read cleanly for this chip (1 = available, 0 = unavailable).",
        "gauge",
        pid_avail_samples,
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
        "TPU MXU duty cycle utilization percentage (0 to 100).",
        "gauge",
        duty_samples,
    )
    add_metric(
        "freetpu_tpu_active_pids",
        "Number of active OS PIDs holding or utilizing this TPU chip (omitted when unreadable).",
        "gauge",
        pid_count_samples,
    )
    add_metric(
        "freetpu_tpu_process_active",
        "Active process indicator per TPU chip and PID (1 = active).",
        "gauge",
        pid_active_samples,
    )
    add_metric(
        "freetpu_tpu_runtime_hbm_used_bytes",
        "TPU Runtime Utilization HBM allocated bytes keyed by libtpu runtime chip ID.",
        "gauge",
        runtime_hbm_used,
    )
    add_metric(
        "freetpu_tpu_runtime_hbm_total_bytes",
        "TPU Runtime Utilization HBM total capacity bytes keyed by libtpu runtime chip ID.",
        "gauge",
        runtime_hbm_total,
    )
    add_metric(
        "freetpu_tpu_runtime_duty_cycle_percent",
        "TPU Runtime Utilization MXU duty cycle percentage keyed by libtpu runtime chip ID.",
        "gauge",
        runtime_duty,
    )

    return "\n".join(lines) + "\n"


def snapshot_to_json(snapshot: ExporterSnapshot) -> str:
    return json.dumps(asdict(snapshot), indent=2, sort_keys=False) + "\n"


DEFAULT_MAX_HTTP_WORKERS = 16
DEFAULT_HEALTH_RESERVED_WORKERS = 4
DEFAULT_HTTP_SOCKET_TIMEOUT_S = 10.0


class _HealthHandler(BaseHTTPRequestHandler):
    timeout = DEFAULT_HTTP_SOCKET_TIMEOUT_S

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
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
        if path in ("/healthz", "/readyz", "/"):
            self._send(200, "application/json; charset=utf-8", '{"status":"ok"}\n')
            return
        self._send(404, "text/plain; charset=utf-8", "not found\n")


class _HealthServer(ThreadingHTTPServer):
    """Dedicated lightweight liveness/readiness server isolated from scrape traffic."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(
        self,
        server_address: Tuple[str, int],
        max_workers: int = DEFAULT_HEALTH_RESERVED_WORKERS,
    ) -> None:
        self._worker_sem = threading.BoundedSemaphore(max(1, max_workers))
        super().__init__(server_address, _HealthHandler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self._worker_sem.acquire(blocking=False):
            try:
                request.settimeout(1.0)
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: text/plain; charset=utf-8\r\n"
                    b"Content-Length: 17\r\n"
                    b"Connection: close\r\n\r\n"
                    b"exporter busy\r\n\r\n"
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._worker_sem.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_sem.release()


class _ExporterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

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
        chip_id_map: Optional[Dict[str, str]] = None,
        max_workers: int = DEFAULT_MAX_HTTP_WORKERS,
        scrape_timeout_s: float = 10.0,
    ) -> None:
        self.sysfs_root = sysfs_root
        self.dev_root = dev_root
        self.proc_root = proc_root
        self.tpu_info_bin = tpu_info_bin
        self.tpu_info_timeout_s = tpu_info_timeout_s
        self.chip_id_map = chip_id_map
        self.scrape_timeout_s = scrape_timeout_s
        self._scrape_lock = threading.Lock()
        scrape_cap = max(1, max_workers)
        self._scrape_sem = threading.BoundedSemaphore(scrape_cap)
        # Reserve extra connection slots above the scrape cap so /healthz on the
        # main port is not rejected when scrape slots are saturated.
        self._worker_sem = threading.BoundedSemaphore(
            scrape_cap + DEFAULT_HEALTH_RESERVED_WORKERS
        )
        super().__init__(server_address, request_handler_class)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self._worker_sem.acquire(blocking=False):
            try:
                request.settimeout(1.0)
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: text/plain; charset=utf-8\r\n"
                    b"Content-Length: 17\r\n"
                    b"Connection: close\r\n\r\n"
                    b"exporter busy\r\n\r\n"
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._worker_sem.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_sem.release()

    def scrape(self) -> ExporterSnapshot:
        if not self._scrape_lock.acquire(timeout=self.scrape_timeout_s):
            raise TimeoutError("timed out waiting for active TPU telemetry scrape")
        try:
            return collect_snapshot(
                sysfs_root=self.sysfs_root,
                dev_root=self.dev_root,
                proc_root=self.proc_root,
                tpu_info_bin=self.tpu_info_bin,
                tpu_info_timeout_s=self.tpu_info_timeout_s,
                chip_id_map=self.chip_id_map,
            )
        finally:
            self._scrape_lock.release()


class _ExporterHandler(BaseHTTPRequestHandler):
    server: _ExporterServer
    timeout = DEFAULT_HTTP_SOCKET_TIMEOUT_S

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
            if not self.server._scrape_sem.acquire(blocking=False):
                self._send(503, "text/plain; charset=utf-8", "exporter busy\n")
                return
            try:
                snap = self.server.scrape()
            except TimeoutError as exc:
                self._send(503, "text/plain; charset=utf-8", f"{exc}\n")
                return
            finally:
                self.server._scrape_sem.release()
            self._send(
                200,
                "text/plain; version=0.0.4; charset=utf-8",
                format_prometheus_metrics(snap),
            )
            return
        if path in ("/metrics.json", "/json"):
            if not self.server._scrape_sem.acquire(blocking=False):
                self._send(503, "text/plain; charset=utf-8", "exporter busy\n")
                return
            try:
                snap = self.server.scrape()
            except TimeoutError as exc:
                self._send(503, "text/plain; charset=utf-8", f"{exc}\n")
                return
            finally:
                self.server._scrape_sem.release()
            self._send(200, "application/json; charset=utf-8", snapshot_to_json(snap))
            return
        if path in ("/healthz", "/readyz", "/"):
            self._send(200, "application/json; charset=utf-8", '{"status":"ok"}\n')
            return
        self._send(404, "text/plain; charset=utf-8", "not found\n")


def _load_chip_id_map(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"--chip-id-map must be a JSON object: {path}")
    return {str(k): str(v) for k, v in raw.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpu-exporter",
        description="FreeTPU Prometheus telemetry exporter for Google Cloud TPUs (0x1ae0 / VFIO).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9400,
        help="HTTP listen port (default: 9400).",
    )
    parser.add_argument(
        "--health-host",
        default=None,
        help="Bind address for optional dedicated health listener (default: same as --host).",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="Optional dedicated HTTP port for /healthz and /readyz isolated from scrape saturation.",
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
    parser.add_argument(
        "--chip-id-map",
        type=Path,
        default=None,
        help="Optional JSON mapping {pci_bdf_or_vfio_path: runtime_chip_id} for verified tpu-info joins.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_HTTP_WORKERS,
        help=f"Maximum concurrent HTTP handler threads (default: {DEFAULT_MAX_HTTP_WORKERS}).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tpu_info_bin = args.tpu_info_bin if args.tpu_info_bin else None
    chip_id_map = _load_chip_id_map(args.chip_id_map)

    if args.once or args.json:
        snap = collect_snapshot(
            sysfs_root=args.sysfs_root,
            dev_root=args.dev_root,
            proc_root=args.proc_root,
            tpu_info_bin=tpu_info_bin,
            tpu_info_timeout_s=args.tpu_info_timeout,
            chip_id_map=chip_id_map,
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
        chip_id_map=chip_id_map,
        max_workers=args.max_workers,
    )
    health_server: Optional[_HealthServer] = None
    health_thread: Optional[threading.Thread] = None
    if args.health_port is not None:
        health_host = args.health_host if args.health_host is not None else args.host
        health_server = _HealthServer((health_host, args.health_port))
        health_thread = threading.Thread(target=health_server.serve_forever, daemon=True)
        health_thread.start()
        sys.stderr.write(
            f"freetpu tpu-exporter health listening on http://{health_host}:{health_server.server_address[1]}/healthz\n"
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
        if health_server is not None:
            health_server.shutdown()
            health_server.server_close()
        if health_thread is not None:
            health_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
