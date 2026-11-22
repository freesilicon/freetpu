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
import importlib
import importlib.util
import inspect
import io
import json
import math
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
# Values: (generation_slug, display_name, nominal_hbm_bytes_per_physical_chip)
TPU_PCI_DEVICES: Dict[str, Tuple[str, str, int]] = {
    "0x0027": ("v2-v3", "TPU v2/v3", 16 * 1024**3),
    "0x0056": ("v4lite", "TPU v4lite", 8 * 1024**3),
    "0x005e": ("v4", "TPU v4", 32 * 1024**3),
    "0x0062": ("v5p", "TPU v5p", 95 * 1024**3),
    "0x0063": ("v5e", "TPU v5e", 16 * 1024**3),
    "0x006e": ("v6e-pf", "TPU v6e (PF)", 32 * 1024**3),
    "0x006f": ("v6e", "TPU v6e (Trillium)", 32 * 1024**3),
}

# Legacy Cloud TPU v2/v3 (0x0027) expose two PCI core functions per physical chip
# under the same base PCI address (`DDDD:BB:DD.0` and `.1`). Subsystem device ID
# distinguishes v2 (2x8 GiB = 16 GiB/chip) from v3 (2x16 GiB = 32 GiB/chip).
LEGACY_DUAL_FUNCTION_DEVICE_IDS = frozenset({"0x0027"})
LEGACY_V2_V3_SUBSYSTEM_DEVICES: Dict[str, Tuple[str, str, int]] = {
    "0x004e": ("v2", "TPU v2", 16 * 1024**3),
    "0x004f": ("v3", "TPU v3", 32 * 1024**3),
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
    r"(?<![0-9A-Za-z._+-])([+-]?[0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)\s*/\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"(?<![0-9A-Za-z._+-])([+-]?[0-9]+(?:\.[0-9]+)?)\s*%")
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
    vfio_group_exclusive: bool = False


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


_LINUX_O_PATH = getattr(os, "O_PATH", 0o10000000)

_GEN_TO_TPU_CHIP_ATTR: Dict[str, str] = {
    "v2": "V2",
    "v3": "V3",
    "v4": "V4",
    "v4lite": "V4",
    "v5e": "V5E",
    "v5p": "V5P",
    "v6e": "V6E",
    "v6e-pf": "V6E",
}


def _read_fdinfo_flags(fdinfo_path: Path) -> Tuple[Optional[int], bool]:
    """Read octal open `flags:` from `/proc/<pid>/fdinfo/<fd>`.

    Returns `(flags_int_or_None, permission_denied)`. Failing closed when
    `fdinfo` is missing or unreadable prevents unverified or `O_PATH`
    descriptors from being counted as active VFIO reservations.
    """
    try:
        text = fdinfo_path.read_text(encoding="utf-8")
    except PermissionError:
        return None, True
    except OSError:
        return None, False
    for line in text.splitlines():
        if line.startswith("flags:"):
            raw = line.split(":", 1)[1].strip()
            try:
                return int(raw, 8), False
            except ValueError:
                return None, False
    return None, False


_VFIO_DEVNUM_RE = re.compile(r"^[0-9]+:[0-9]+$")


def _read_kernel_vfio_devnum(sysfs_root: Path, group: Optional[str] = None) -> Optional[str]:
    """Read kernel-exported `<major>:<minor>` from sysfs for `/dev/vfio/<group>` or `/dev/vfio/vfio`."""
    if group is None:
        candidates = (
            sysfs_root / "class" / "misc" / "vfio" / "dev",
            sysfs_root / "devices" / "virtual" / "misc" / "vfio" / "dev",
        )
    else:
        if not group:
            return None
        candidates = (
            sysfs_root / "class" / "vfio" / group / "dev",
            sysfs_root / "devices" / "virtual" / "vfio" / group / "dev",
        )
    for path in candidates:
        raw = _read_text(path)
        if raw and _VFIO_DEVNUM_RE.fullmatch(raw):
            return raw
    return None


def _stat_non_symlink_char_devnum(path: Path) -> Optional[str]:
    """Return '<major>:<minor>' iff `path` itself (not a symlink) is a character device."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISCHR(st.st_mode):
        return f"{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}"
    return None


DEFAULT_PROC_SCAN_TIMEOUT_S = 2.0
_PROC_SCAN_LOCK = threading.Lock()
_IN_FLIGHT_PROC_SCANS: set[str] = set()


def _scan_proc_vfio_pids_sync(
    proc_root: Path,
    rdev_to_paths: Dict[int, List[str]],
    result: Dict[str, List[int]],
    deadline: float,
) -> Tuple[bool, int]:
    """Synchronously scan `/proc/<pid>/fd/*` via `os.scandir()` within `deadline`."""
    timed_out = False
    unreadable = 0
    fd_checks = 0

    try:
        proc_iter = os.scandir(proc_root)
    except PermissionError:
        return False, 1
    except OSError:
        return False, 1

    with proc_iter:
        for entry in proc_iter:
            if not entry.name.isdigit():
                continue
            if time.monotonic() > deadline:
                timed_out = True
                break
            pid = int(entry.name)
            fd_dir_path = os.path.join(entry.path, "fd")
            try:
                fd_iter = os.scandir(fd_dir_path)
            except PermissionError:
                unreadable += 1
                continue
            except OSError:
                # Process may have exited concurrently; not an access denial.
                continue
            with fd_iter:
                if not rdev_to_paths:
                    continue
                for fd_entry in fd_iter:
                    fd_checks += 1
                    if (fd_checks & 0xFF) == 0 and time.monotonic() > deadline:
                        timed_out = True
                        break
                    try:
                        fd_st = os.stat(fd_entry.path)
                    except PermissionError:
                        unreadable += 1
                        continue
                    except OSError:
                        continue
                    if not stat.S_ISCHR(fd_st.st_mode):
                        continue
                    matched_paths = rdev_to_paths.get(fd_st.st_rdev)
                    if not matched_paths:
                        continue
                    flags, perm_denied = _read_fdinfo_flags(
                        Path(entry.path) / "fdinfo" / fd_entry.name
                    )
                    if perm_denied:
                        unreadable += 1
                        continue
                    if flags is None or (flags & _LINUX_O_PATH) != 0:
                        continue
                    for canonical in matched_paths:
                        if pid not in result[canonical]:
                            result[canonical].append(pid)
            if timed_out:
                break

    for pids in result.values():
        pids.sort()
    return (unreadable == 0 and not timed_out), unreadable + (1 if timed_out else 0)


def _scan_proc_vfio_pids(
    proc_root: Path,
    vfio_paths: Sequence[str],
    expected_devnums: Optional[Dict[str, Optional[str]]] = None,
    scan_timeout_s: float = DEFAULT_PROC_SCAN_TIMEOUT_S,
) -> Tuple[Dict[str, List[int]], bool, int, Dict[str, int]]:
    """Find PIDs holding active non-`O_PATH` char-device FDs matching host `/dev/vfio/<group>` `st_rdev`.

    Streams `/proc` and `/proc/<pid>/fd` with `os.scandir()` inside a single-flight
    deadline-bounded daemon worker per `proc_root`. If a host process holds an open
    descriptor on an unresponsive hard-mounted filesystem (e.g. NFS) and `os.stat()`
    blocks past `scan_timeout_s`, this function returns promptly with
    `pid_telemetry_accessible=False` without holding the scrape lock or spawning
    unbounded extra threads, so PCI inventory and health metrics remain available.
    """
    unique_paths = list(dict.fromkeys(p for p in vfio_paths if p))
    result: Dict[str, List[int]] = {}
    verified_rdevs: Dict[str, int] = {}
    if not unique_paths:
        return result, True, 0, verified_rdevs
    if not proc_root.is_dir():
        return result, False, 1, verified_rdevs

    rdev_to_paths: Dict[int, List[str]] = {}
    for vp in unique_paths:
        try:
            host_st = os.lstat(vp)
        except OSError:
            continue
        if not stat.S_ISCHR(host_st.st_mode):
            continue
        if expected_devnums is not None:
            kernel_dn = expected_devnums.get(vp)
            actual_dn = f"{os.major(host_st.st_rdev)}:{os.minor(host_st.st_rdev)}"
            if not kernel_dn or actual_dn != kernel_dn:
                continue
        verified_rdevs[vp] = host_st.st_rdev
        result[vp] = []
        rdev_to_paths.setdefault(host_st.st_rdev, []).append(vp)

    budget_s = float(scan_timeout_s)
    if budget_s <= 0.0:
        return {vp: [] for vp in verified_rdevs}, False, 1, verified_rdevs

    proc_key = os.fspath(proc_root)
    with _PROC_SCAN_LOCK:
        if proc_key in _IN_FLIGHT_PROC_SCANS:
            return {vp: [] for vp in verified_rdevs}, False, 1, verified_rdevs
        _IN_FLIGHT_PROC_SCANS.add(proc_key)

    worker_result: Dict[str, List[int]] = {vp: [] for vp in result}
    outcome: Dict[str, Any] = {"ok": False, "unreadable": 1}
    done_ev = threading.Event()
    deadline = time.monotonic() + budget_s

    def _worker() -> None:
        try:
            ok, unrd = _scan_proc_vfio_pids_sync(
                proc_root, rdev_to_paths, worker_result, deadline
            )
            outcome["ok"] = ok
            outcome["unreadable"] = unrd
        finally:
            with _PROC_SCAN_LOCK:
                _IN_FLIGHT_PROC_SCANS.discard(proc_key)
            done_ev.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    if not done_ev.wait(timeout=budget_s):
        return {vp: [] for vp in verified_rdevs}, False, 1, verified_rdevs
    return worker_result, bool(outcome["ok"]), int(outcome["unreadable"]), verified_rdevs


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
                used_bytes = _parse_bytes(mem_match.group(1), mem_match.group(2))
                total_bytes = _parse_bytes(mem_match.group(3), mem_match.group(4))
                if used_bytes >= 0 and total_bytes > 0:
                    row.hbm_used_bytes = used_bytes
                    row.hbm_total_bytes = total_bytes
            if pct_match:
                pct_val = float(pct_match.group(1))
                if math.isfinite(pct_val) and pct_val >= 0.0:
                    row.duty_cycle_percent = pct_val
        elif section == "device_status":
            row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))
            row.in_util_table = True
            mem_idx = col_map.get("mem", 1)
            duty_idx = col_map.get("duty", 4)
            if mem_idx < len(cells):
                mem_match = _MEM_PAIR_RE.search(cells[mem_idx])
                if mem_match:
                    used_bytes = _parse_bytes(mem_match.group(1), mem_match.group(2))
                    total_bytes = _parse_bytes(mem_match.group(3), mem_match.group(4))
                    if used_bytes >= 0 and total_bytes > 0:
                        row.hbm_used_bytes = used_bytes
                        row.hbm_total_bytes = total_bytes
            if duty_idx < len(cells):
                pct_match = _PCT_RE.search(cells[duty_idx])
                if pct_match:
                    pct_val = float(pct_match.group(1))
                    if math.isfinite(pct_val) and pct_val >= 0.0:
                        row.duty_cycle_percent = pct_val
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


_PY_METRICS_SUBPROCESS_HELPER = (
    "import inspect, json, sys, tpu_info.device as d, tpu_info.metrics as m\n"
    "chip_attr, addr, timeout_s = sys.argv[1], sys.argv[2], float(sys.argv[3])\n"
    "chip = getattr(d.TpuChip, chip_attr)\n"
    "sig = inspect.signature(m.get_chip_usage)\n"
    "kw = {}\n"
    "if 'addr' in sig.parameters:\n"
    "    kw['addr'] = addr\n"
    "if 'timeout' in sig.parameters:\n"
    "    kw['timeout'] = timeout_s\n"
    "rows = m.get_chip_usage(chip, **kw)\n"
    "out = [\n"
    "    {\n"
    "        'device_id': getattr(r, 'device_id', None),\n"
    "        'memory_usage': getattr(r, 'memory_usage', None),\n"
    "        'total_memory': getattr(r, 'total_memory', None),\n"
    "        'duty_cycle_pct': getattr(r, 'duty_cycle_pct', None),\n"
    "    }\n"
    "    for r in (rows or [])\n"
    "]\n"
    "sys.stdout.write(json.dumps(out))\n"
)


def _parse_python_usage_entries(
    entries: Iterable[Any],
    addr: str,
) -> Tuple[bool, Optional[str], Dict[str, _TpuInfoRow]]:
    rows: Dict[str, _TpuInfoRow] = {}
    for item in entries:
        if isinstance(item, dict):
            dev_id = item.get("device_id")
            mem_used = item.get("memory_usage")
            mem_total = item.get("total_memory")
            duty = item.get("duty_cycle_pct")
        else:
            dev_id = getattr(item, "device_id", None)
            mem_used = getattr(item, "memory_usage", None)
            mem_total = getattr(item, "total_memory", None)
            duty = getattr(item, "duty_cycle_pct", None)
        if dev_id is None:
            continue
        key = str(dev_id)
        row = rows.setdefault(key, _TpuInfoRow(chip_key=key))
        row.in_util_table = True
        if isinstance(mem_used, (int, float)) and not isinstance(mem_used, bool) and mem_used >= 0:
            row.hbm_used_bytes = int(mem_used)
        if isinstance(mem_total, (int, float)) and not isinstance(mem_total, bool) and mem_total > 0:
            row.hbm_total_bytes = int(mem_total)
        if isinstance(duty, (int, float)) and not isinstance(duty, bool) and duty >= 0:
            row.duty_cycle_percent = float(duty)

    if not rows:
        return False, f"tpu_info.metrics returned no parseable chip entries from {addr}", {}
    return True, None, rows


def _query_tpu_info_python_metrics(
    devices: Sequence[TpuTelemetry],
    addr: Optional[str] = "127.0.0.1:8431",
    timeout_s: float = 5.0,
) -> Tuple[bool, Optional[str], Dict[str, _TpuInfoRow]]:
    """Query `libtpu` runtime HBM & duty-cycle metrics via `tpu_info.metrics` with a hard deadline.

    Decouples live utilization collection from `/proc/*/fd/*` PID scanning and
    `/dev/vfio/*` access so Kubernetes sidecars can run without
    `shareProcessNamespace`, `SYS_PTRACE`, or cross-AppArmor-profile `/proc` reads.
    Runs real installed `tpu_info` queries in a deadline-bounded subprocess so an
    unresponsive `libtpu` gRPC server (upstream client sets no RPC deadline) is
    terminated cleanly and never holds `_ExporterServer._scrape_lock` indefinitely.
    """
    if not addr:
        return False, "runtime metrics gRPC address disabled", {}

    deadline_s = max(0.05, float(timeout_s))
    chip_attr = None
    for dev in devices:
        candidate = _GEN_TO_TPU_CHIP_ATTR.get(dev.generation)
        if candidate:
            chip_attr = candidate
            break
    if chip_attr is None:
        return False, "unable to map discovered TPU generation to tpu_info.device.TpuChip", {}

    # Support in-memory mocked `tpu_info.metrics` modules in unit tests with a
    # bounded worker thread; for real installed packages, execute in an isolated
    # subprocess with a hard OS timeout so hung RPCs cannot leak threads or locks.
    cached_metrics_mod = sys.modules.get("tpu_info.metrics")
    if cached_metrics_mod is not None and not getattr(cached_metrics_mod, "__file__", None):
        tpu_device_mod = sys.modules.get("tpu_info.device")
        get_chip_usage = getattr(cached_metrics_mod, "get_chip_usage", None)
        tpu_chip_enum = getattr(tpu_device_mod, "TpuChip", None) if tpu_device_mod is not None else None
        if not callable(get_chip_usage) or tpu_chip_enum is None or not hasattr(tpu_chip_enum, chip_attr):
            return False, "tpu_info.metrics.get_chip_usage unavailable", {}
        chip_type = getattr(tpu_chip_enum, chip_attr)

        box: Dict[str, Any] = {}
        done_ev = threading.Event()

        def _worker() -> None:
            try:
                sig = inspect.signature(get_chip_usage)
                kw: Dict[str, Any] = {}
                if "addr" in sig.parameters:
                    kw["addr"] = addr
                if "timeout" in sig.parameters:
                    kw["timeout"] = deadline_s
                box["rows"] = get_chip_usage(chip_type, **kw)
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                done_ev.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        if not done_ev.wait(timeout=deadline_s):
            return False, f"tpu_info.metrics query timed out after {deadline_s}s ({addr})", {}
        if "error" in box:
            return False, f"tpu_info.metrics query failed ({addr}): {box['error']}", {}
        usage_list = box.get("rows")
        if not usage_list:
            return False, f"tpu_info.metrics returned no runtime usage rows from {addr}", {}
        return _parse_python_usage_entries(usage_list, addr)

    try:
        if importlib.util.find_spec("tpu_info") is None:
            return False, "tpu_info python API unavailable: No module named 'tpu_info'", {}
    except Exception as exc:  # noqa: BLE001
        return False, f"tpu_info python API unavailable: {exc}", {}

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PY_METRICS_SUBPROCESS_HELPER, chip_attr, addr, str(deadline_s)],
            capture_output=True,
            text=True,
            timeout=deadline_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"tpu_info.metrics query timed out after {deadline_s}s ({addr})", {}
    except OSError as exc:
        return False, f"tpu_info.metrics subprocess failed ({addr}): {exc}", {}

    if proc.returncode != 0:
        err_lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        err_msg = err_lines[-1] if err_lines else f"exit {proc.returncode}"
        return False, f"tpu_info.metrics query failed ({addr}): {err_msg}", {}

    try:
        parsed_json = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, f"tpu_info.metrics returned invalid JSON ({addr}): {exc}", {}

    if not isinstance(parsed_json, list) or not parsed_json:
        return False, f"tpu_info.metrics returned no runtime usage rows from {addr}", {}
    return _parse_python_usage_entries(parsed_json, addr)


def _resolve_tpu_device_spec(device_id: str, subsystem_device: str) -> Tuple[str, str, int]:
    if device_id == "0x0027" and subsystem_device in LEGACY_V2_V3_SUBSYSTEM_DEVICES:
        return LEGACY_V2_V3_SUBSYSTEM_DEVICES[subsystem_device]
    return TPU_PCI_DEVICES[device_id]


def _collect_iommu_group_membership(
    sysfs_root: Path,
    pci_entries: Sequence[Path],
) -> Tuple[Dict[str, set[str]], Dict[str, bool]]:
    """Collect full PCI member sets per IOMMU group across sysfs.

    Returns `(members_by_group, group_devices_dir_readable)`. A group is only
    considered verifiable for exclusivity when its authoritative
    `/sys/kernel/iommu_groups/<group>/devices` directory was present and readable.
    """
    members: Dict[str, set[str]] = {}
    readable: Dict[str, bool] = {}

    for entry in pci_entries:
        grp = _read_symlink_name(entry / "iommu_group") or ""
        if grp:
            members.setdefault(grp, set()).add(entry.name)

    iommu_root = sysfs_root / "kernel" / "iommu_groups"
    if iommu_root.is_dir():
        try:
            grp_dirs = list(iommu_root.iterdir())
        except OSError:
            grp_dirs = []
        for grp_dir in grp_dirs:
            dev_dir = grp_dir / "devices"
            if not dev_dir.is_dir():
                continue
            try:
                grp_members = {m.name for m in dev_dir.iterdir()}
            except OSError:
                continue
            readable[grp_dir.name] = True
            members.setdefault(grp_dir.name, set()).update(grp_members)

    return members, readable


def discover_tpu_devices(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
) -> List[TpuTelemetry]:
    """Scan sysfs PCI devices for vendor 0x1ae0 and resolve VFIO nodes."""
    pci_dir = sysfs_root / "bus" / "pci" / "devices"
    if not pci_dir.is_dir():
        return []

    pci_entries = sorted(pci_dir.iterdir(), key=lambda p: p.name)
    group_members, group_dir_readable = _collect_iommu_group_membership(
        sysfs_root, pci_entries
    )

    vfio_container_path = dev_root / "vfio" / "vfio"
    vfio_container_present = vfio_container_path.exists()
    vfio_container_dn = _stat_non_symlink_char_devnum(vfio_container_path)
    kernel_container_dn = _read_kernel_vfio_devnum(sysfs_root, group=None)
    vfio_container_verified = bool(
        vfio_container_dn is not None
        and kernel_container_dn is not None
        and vfio_container_dn == kernel_container_dn
    )
    discovered: List[TpuTelemetry] = []
    legacy_by_base_addr: Dict[str, TpuTelemetry] = {}

    for entry in pci_entries:
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
        subsystem_device = _normalize_hex_id(_read_text(entry / "subsystem_device") or "")
        gen, model_name, nominal_hbm = _resolve_tpu_device_spec(device_id, subsystem_device)
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
        vfio_group_dn = _stat_non_symlink_char_devnum(vfio_path_obj) if vfio_path_obj else None
        vfio_is_char = vfio_group_dn is not None
        kernel_group_dn = _read_kernel_vfio_devnum(sysfs_root, iommu_group) if iommu_group else None
        vfio_group_verified = bool(
            vfio_group_dn is not None
            and kernel_group_dn is not None
            and vfio_group_dn == kernel_group_dn
        )

        # On vfio-pci, PCI `enable` is 0 while idle (no process holds /dev/vfio/*);
        # readiness requires vfio-pci binding plus verified non-symlink character
        # device nodes matching kernel sysfs major:minor for both /dev/vfio/vfio
        # and /dev/vfio/<group>.
        healthy = (
            driver == "vfio-pci"
            and bool(iommu_group)
            and vfio_group_verified
            and vfio_container_verified
        )

        # Group dual-core legacy TPU v2/v3 (0x0027) PCI functions sharing the same
        # base PCI address (`DDDD:BB:DD`) into one physical chip record.
        if device_id in LEGACY_DUAL_FUNCTION_DEVICE_IDS:
            base_addr = entry.name.rsplit(".", 1)[0]
            existing = legacy_by_base_addr.get(base_addr)
            if existing is not None:
                existing.pci_enabled = existing.pci_enabled or pci_enabled
                existing.healthy = existing.healthy and healthy
                if existing.generation == "v2-v3" and gen != "v2-v3":
                    existing.generation = gen
                    existing.model_name = model_name
                    existing.hbm_nominal_bytes = nominal_hbm
                    existing.hbm_total_bytes = nominal_hbm if nominal_hbm > 0 else None
                continue

        # A VFIO group is exclusive to this TPU ONLY when we verified the full
        # kernel `/sys/kernel/iommu_groups/<group>/devices` membership and it
        # contains strictly this single PCI BDF (no sibling TPUs or non-TPU devices).
        full_members = group_members.get(iommu_group, set()) | ({entry.name} if iommu_group else set())
        vfio_group_exclusive = (
            driver == "vfio-pci"
            and bool(iommu_group)
            and group_dir_readable.get(iommu_group, False)
            and full_members == {entry.name}
        )

        chip = TpuTelemetry(
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
            vfio_group_exclusive=vfio_group_exclusive,
        )
        discovered.append(chip)
        if device_id in LEGACY_DUAL_FUNCTION_DEVICE_IDS:
            legacy_by_base_addr[entry.name.rsplit(".", 1)[0]] = chip

    return discovered


def _shared_vfio_paths(devices: Sequence[TpuTelemetry]) -> set[str]:
    """Return `/dev/vfio/<group>` paths shared by more than one discovered TPU chip."""
    counts: Dict[str, int] = {}
    for d in devices:
        if d.vfio_device:
            counts[d.vfio_device] = counts.get(d.vfio_device, 0) + 1
    return {vp for vp, count in counts.items() if count > 1}


def validate_chip_id_map(
    devices: Sequence[TpuTelemetry],
    chip_id_map: Optional[Dict[str, str]],
) -> None:
    """Validate that `--chip-id-map` never uses a shared VFIO path or duplicates a runtime ID."""
    if not chip_id_map:
        return

    shared_vfio = _shared_vfio_paths(devices)
    for raw_key in chip_id_map:
        norm_key = os.path.normpath(raw_key) if raw_key.startswith("/") else raw_key
        if raw_key in shared_vfio or norm_key in shared_vfio:
            raise ValueError(
                f"--chip-id-map key {raw_key!r} targets a shared VFIO group; "
                "use explicit PCI BDF keys for chips sharing a VFIO group"
            )

    seen_runtime_targets: Dict[str, str] = {}
    for dev in devices:
        mapped = chip_id_map.get(dev.pci_bdf)
        if mapped is None and dev.vfio_device and dev.vfio_device not in shared_vfio:
            mapped = chip_id_map.get(dev.vfio_device)
        if mapped is None:
            continue
        rt_id = str(mapped)
        prev_bdf = seen_runtime_targets.get(rt_id)
        if prev_bdf is not None and prev_bdf != dev.pci_bdf:
            raise ValueError(
                f"--chip-id-map assigns runtime chip {rt_id!r} to multiple PCI BDFs "
                f"({prev_bdf} and {dev.pci_bdf})"
            )
        seen_runtime_targets[rt_id] = dev.pci_bdf


DEFAULT_RUNTIME_METRICS_ADDR = "127.0.0.1:8431"
MAX_HTTP_HEADER_BYTES = 16384
HTTP_HEADER_READ_TIMEOUT_S = 2.0


def collect_snapshot(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    proc_root: Path = Path("/proc"),
    tpu_info_bin: Optional[str] = "tpu-info",
    tpu_info_timeout_s: float = 5.0,
    chip_id_map: Optional[Dict[str, str]] = None,
    *,
    skip_pid_scan: bool = False,
    runtime_metrics_addr: Optional[str] = DEFAULT_RUNTIME_METRICS_ADDR,
) -> ExporterSnapshot:
    """Collect PCI/VFIO discovery, optional procfs PID handles, and `tpu-info` runtime metrics."""
    t0 = time.monotonic()
    devices = discover_tpu_devices(sysfs_root=sysfs_root, dev_root=dev_root)
    shared_vfio = _shared_vfio_paths(devices)
    validate_chip_id_map(devices, chip_id_map)

    # Scan /proc/<pid>/fd for open handles to <dev_root>/vfio/<group> unless
    # decoupled (`skip_pid_scan=True`, used by unprivileged workload sidecars).
    real_vfio_paths = [
        str(dev_root / "vfio" / d.iommu_group)
        for d in devices
        if d.iommu_group
    ]
    expected_vfio_devnums: Dict[str, Optional[str]] = {
        str(dev_root / "vfio" / d.iommu_group): _read_kernel_vfio_devnum(sysfs_root, d.iommu_group)
        for d in devices
        if d.iommu_group
    }
    if skip_pid_scan:
        proc_pids: Dict[str, List[int]] = {}
        proc_fd_accessible = False
        vfio_rdevs: Dict[str, int] = {}
    else:
        proc_pids, proc_fd_accessible, _unreadable, vfio_rdevs = _scan_proc_vfio_pids(
            proc_root, real_vfio_paths, expected_devnums=expected_vfio_devnums
        )

    runtime_budget_s = max(0.05, float(tpu_info_timeout_s))
    runtime_deadline = time.monotonic() + runtime_budget_s
    explicit_non_default_runtime_addr = bool(
        runtime_metrics_addr and runtime_metrics_addr != DEFAULT_RUNTIME_METRICS_ADDR
    )
    if not tpu_info_bin:
        # Empty/None `--tpu-info-bin` explicitly disables all `tpu-info` runtime
        # telemetry (both CLI and Python gRPC) across every mode.
        tpu_info_ok, tpu_info_err, parsed_rows = False, "tpu-info disabled", {}
    elif skip_pid_scan or explicit_non_default_runtime_addr:
        # Sidecar / no-PID mode OR explicit non-default `--runtime-metrics-addr`:
        # query `tpu_info.metrics` directly at `runtime_metrics_addr` with a hard
        # deadline. Never run the `tpu-info` CLI subprocess against its hardcoded
        # default port 8431 when an explicit non-default runtime endpoint was
        # configured (or when PID scanning is disabled).
        tpu_info_ok, tpu_info_err, parsed_rows = _query_tpu_info_python_metrics(
            devices,
            addr=runtime_metrics_addr,
            timeout_s=runtime_budget_s,
        )
    else:
        tpu_info_ok, tpu_info_err, parsed_rows = _run_tpu_info(
            tpu_info_bin=tpu_info_bin,
            timeout_s=runtime_budget_s,
        )
        # If `tpu-info` CLI failed OR exited cleanly with only inventory/`N/A` rows
        # (no live HBM/duty-cycle readings, e.g. when its default endpoint differs from
        # `--runtime-metrics-addr`), fall back to direct `tpu_info.metrics` gRPC queries
        # ONLY within the remaining shared deadline so a stalled `libtpu` endpoint
        # cannot double-spend the scrape timeout before PCI and PID metrics are returned.
        has_live_util = any(
            r.in_util_table
            and (
                r.hbm_used_bytes is not None
                or r.hbm_total_bytes is not None
                or r.duty_cycle_percent is not None
            )
            for r in parsed_rows.values()
        )
        remaining_s = runtime_deadline - time.monotonic()
        if (
            (not tpu_info_ok or not has_live_util)
            and tpu_info_bin
            and runtime_metrics_addr
            and remaining_s >= 0.05
        ):
            py_ok, _py_err, py_rows = _query_tpu_info_python_metrics(
                devices,
                addr=runtime_metrics_addr,
                timeout_s=remaining_s,
            )
            if py_ok:
                merged_rows = dict(parsed_rows)
                for k, py_r in py_rows.items():
                    existing = merged_rows.get(k)
                    if existing is not None:
                        if py_r.hbm_used_bytes is not None:
                            existing.hbm_used_bytes = py_r.hbm_used_bytes
                        if py_r.hbm_total_bytes is not None:
                            existing.hbm_total_bytes = py_r.hbm_total_bytes
                        if py_r.duty_cycle_percent is not None:
                            existing.duty_cycle_percent = py_r.duty_cycle_percent
                        existing.in_util_table = True
                    else:
                        merged_rows[k] = py_r
                tpu_info_ok, tpu_info_err, parsed_rows = True, None, merged_rows
    util_rows = {k: r for k, r in parsed_rows.items() if r.in_util_table}
    mapped_util_keys: Dict[str, str] = {}
    overall_pid_available = True

    # Combine sysfs `chip_id` and explicit `--chip-id-map` across all discovered chips
    # first so any runtime chip ID (or utilization key) claimed by >1 physical chip is
    # detected as ambiguous and left keyed strictly by runtime ID (`pci_bdf="unmapped"`).
    resolved_rt_ids: Dict[str, Optional[str]] = {}
    rt_id_claimants: Dict[str, List[str]] = {}
    prospective_util_key: Dict[str, Optional[str]] = {}
    util_key_claimants: Dict[str, List[str]] = {}

    for dev in devices:
        is_shared_vfio = bool(dev.vfio_device and dev.vfio_device in shared_vfio)
        verified_runtime_id = dev.runtime_chip_id
        if not verified_runtime_id and chip_id_map:
            mapped = chip_id_map.get(dev.pci_bdf)
            if mapped is None and not is_shared_vfio and dev.vfio_device:
                mapped = chip_id_map.get(dev.vfio_device)
            if mapped is not None:
                verified_runtime_id = str(mapped)
        resolved_rt_ids[dev.pci_bdf] = verified_runtime_id
        if verified_runtime_id:
            rt_id_claimants.setdefault(verified_runtime_id, []).append(dev.pci_bdf)

    ambiguous_rt_ids = {rt for rt, bdfs in rt_id_claimants.items() if len(bdfs) > 1}

    for dev in devices:
        is_shared_vfio = bool(dev.vfio_device and dev.vfio_device in shared_vfio)
        rt_id = resolved_rt_ids.get(dev.pci_bdf)
        if rt_id in ambiguous_rt_ids:
            rt_id = None
        util_candidates: List[str] = [dev.pci_bdf]
        if dev.vfio_device and not is_shared_vfio:
            util_candidates.append(dev.vfio_device)
        if rt_id:
            util_candidates.append(rt_id)

        matched_key: Optional[str] = None
        for key in util_candidates:
            if key and key in util_rows:
                matched_key = key
                break
        prospective_util_key[dev.pci_bdf] = matched_key
        if matched_key is not None:
            util_key_claimants.setdefault(matched_key, []).append(dev.pci_bdf)

    ambiguous_util_keys = {k for k, bdfs in util_key_claimants.items() if len(bdfs) > 1}

    for dev in devices:
        dev_real_vfio = str(dev_root / "vfio" / dev.iommu_group) if dev.iommu_group else ""
        pids = list(proc_pids.get(dev_real_vfio, []))
        host_rdev = vfio_rdevs.get(dev_real_vfio) if dev_real_vfio else None

        # Attach `TPU Chips` rows strictly by verified /dev/vfio/<group> or PCI BDF.
        chips_row: Optional[_TpuInfoRow] = None
        for key in (dev.vfio_device, dev.pci_bdf):
            if key and key in parsed_rows and parsed_rows[key].in_chips_table:
                chips_row = parsed_rows[key]
                break

        if chips_row is not None and not skip_pid_scan:
            dev.telemetry_source = "tpu-info"

        rt_id = resolved_rt_ids.get(dev.pci_bdf)
        if rt_id in ambiguous_rt_ids:
            dev.runtime_chip_id = None
        elif rt_id:
            dev.runtime_chip_id = rt_id

        matched_key = prospective_util_key.get(dev.pci_bdf)
        matched_util: Optional[_TpuInfoRow] = None
        if matched_key is not None and matched_key not in ambiguous_util_keys:
            matched_util = util_rows[matched_key]
            mapped_util_keys[matched_key] = dev.pci_bdf

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
        # PID telemetry available when BOTH the `/proc/<pid>/fd` scan succeeded without
        # permission denials AND this device's `/dev/vfio/<group>` character device
        # number (`st_rdev`) was verified, so a missing/unreadable/non-char VFIO node
        # never emits a false zero PID count as valid billing coverage.
        dev.pid_telemetry_available = proc_fd_accessible and (host_rdev is not None)
        if not dev.pid_telemetry_available:
            overall_pid_available = False
            dev.active_pids = []
        else:
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
            "vfio_group_exclusive": "true" if d.vfio_group_exclusive else "false",
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
        skip_pid_scan: bool = False,
        runtime_metrics_addr: Optional[str] = "127.0.0.1:8431",
        max_workers: int = DEFAULT_MAX_HTTP_WORKERS,
        scrape_timeout_s: float = 10.0,
    ) -> None:
        self.sysfs_root = sysfs_root
        self.dev_root = dev_root
        self.proc_root = proc_root
        self.tpu_info_bin = tpu_info_bin
        self.tpu_info_timeout_s = tpu_info_timeout_s
        self.chip_id_map = chip_id_map
        self.skip_pid_scan = skip_pid_scan
        self.runtime_metrics_addr = runtime_metrics_addr
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
                skip_pid_scan=self.skip_pid_scan,
                runtime_metrics_addr=self.runtime_metrics_addr,
            )
        finally:
            self._scrape_lock.release()


class _ExporterHandler(BaseHTTPRequestHandler):
    server: _ExporterServer
    timeout = DEFAULT_HTTP_SOCKET_TIMEOUT_S

    def handle_one_request(self) -> None:
        """Enforce aggregate header size and total wall-clock header read deadline."""
        self.close_connection = True
        deadline = time.monotonic() + HTTP_HEADER_READ_TIMEOUT_S
        buf = bytearray()
        header_complete = False
        try:
            while len(buf) <= MAX_HTTP_HEADER_BYTES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._send_raw_error(408, "Request Timeout", "header read timeout\n")
                    return
                self.connection.settimeout(remaining)
                chunk = self.connection.recv(min(4096, MAX_HTTP_HEADER_BYTES + 1 - len(buf)))
                if not chunk:
                    return
                buf.extend(chunk)
                if b"\r\n\r\n" in buf or b"\n\n" in buf:
                    header_complete = True
                    break
            if not header_complete:
                self._send_raw_error(
                    431, "Request Header Fields Too Large", "request headers too large\n"
                )
                return
            self.connection.settimeout(self.timeout)
            self.rfile = io.BytesIO(bytes(buf))
            super().handle_one_request()
        except socket.timeout:
            self._send_raw_error(408, "Request Timeout", "header read timeout\n")
        except OSError:
            return
        finally:
            self.close_connection = True

    def _send_raw_error(self, code: int, reason: str, body: str) -> None:
        payload = body.encode("utf-8")
        resp = (
            f"HTTP/1.1 {code} {reason}\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + payload
        try:
            self.connection.settimeout(1.0)
            self.connection.sendall(resp)
        except OSError:
            pass

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
            except ValueError as exc:
                self._send(500, "text/plain; charset=utf-8", f"{exc}\n")
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
            except ValueError as exc:
                self._send(500, "text/plain; charset=utf-8", f"{exc}\n")
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
        default="127.0.0.1",
        help="Bind address for optional dedicated health listener (default: 127.0.0.1, loopback-only).",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="Optional dedicated loopback HTTP port for /healthz and /readyz isolated from scrape and pod-network traffic.",
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
        help="Shared total timeout in seconds for runtime telemetry (tpu-info CLI plus Python gRPC fallback; default: 5.0).",
    )
    parser.add_argument(
        "--chip-id-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping {pci_bdf_or_unique_vfio_path: runtime_chip_id} "
            "for verified tpu-info joins (shared VFIO groups require explicit PCI BDF keys)."
        ),
    )
    parser.add_argument(
        "--no-pid-scan",
        action="store_true",
        help=(
            "Disable `/proc/*/fd` PID scanning and query `tpu_info.metrics` directly "
            "over loopback gRPC (recommended for unprivileged workload sidecars without "
            "`shareProcessNamespace` or `SYS_PTRACE`)."
        ),
    )
    parser.add_argument(
        "--runtime-metrics-addr",
        default="127.0.0.1:8431",
        help=(
            "Loopback `libtpu` gRPC address for direct `tpu_info.metrics` queries "
            "(default: 127.0.0.1:8431; empty string disables direct gRPC queries)."
        ),
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
    runtime_metrics_addr = args.runtime_metrics_addr if args.runtime_metrics_addr else None
    try:
        chip_id_map = _load_chip_id_map(args.chip_id_map)
        if chip_id_map:
            initial_devices = discover_tpu_devices(
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
            )
            validate_chip_id_map(initial_devices, chip_id_map)
    except ValueError as exc:
        sys.stderr.write(f"freetpu tpu-exporter error: {exc}\n")
        return 1

    if args.once or args.json:
        try:
            snap = collect_snapshot(
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
                proc_root=args.proc_root,
                tpu_info_bin=tpu_info_bin,
                tpu_info_timeout_s=args.tpu_info_timeout,
                chip_id_map=chip_id_map,
                skip_pid_scan=args.no_pid_scan,
                runtime_metrics_addr=runtime_metrics_addr,
            )
        except ValueError as exc:
            sys.stderr.write(f"freetpu tpu-exporter error: {exc}\n")
            return 1
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
        skip_pid_scan=args.no_pid_scan,
        runtime_metrics_addr=runtime_metrics_addr,
        max_workers=args.max_workers,
    )
    health_server: Optional[_HealthServer] = None
    health_thread: Optional[threading.Thread] = None
    if args.health_port is not None:
        health_host = args.health_host
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
