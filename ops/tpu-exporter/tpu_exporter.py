#!/usr/bin/env python3
"""FreeTPU Prometheus Telemetry Exporter (DCGM-exporter equivalent for Cloud TPUs).

Discovers Google TPU PCI devices (vendor 0x1ae0) under /sys/bus/pci/devices/*
and their /dev/vfio/* IOMMU device nodes on VFIO-backed Cloud TPU VMs (v5e,
v5p, v6e Trillium, and tpu7x pending live HW confirmation), scrapes:
  - Direct `libtpu` gRPC / `RuntimeMetricService` (`localhost:8431`) when
    `grpcio` and protobuf stubs are available (with graceful fallback and zero
    unconditional `tpu-info` subprocess churn)
  - Lockless SeqLock `/dev/shm/freetpu_metrics_<bdf>` `pread` telemetry for
    bare-metal `libopentpu.so` (`ops/tpu-exporter/freetpu_metrics_shm.h`),
    authenticated via opened-fd owner UID and non-group/world-writable mode
  - Pure `libtpu`-free sysfs PCIe AER counters (`aer_dev_correctable`,
    `aer_dev_fatal`)
  - Optional GCP Spot preemption metadata gauge
  - Legacy `tpu-info` CLI / Python fallback and `/proc/<pid>/fd` VFIO handles

and serves Prometheus text exposition format 0.0.4 on HTTP `/metrics`, or
prints a one-shot snapshot with `--once` / `--json`.

Note: Legacy `/dev/accel*` Cloud TPU hosts (e.g. TPU v2/v3/v4 using the `accel`
driver instead of `vfio-pci`) are recognized in PCI inventory, but VFIO
readiness (`freetpu_tpu_health_status`) and `/dev/vfio/*` PID tracking are
scoped to VFIO-backed hosts.

Zero third-party dependencies required: uses Python standard library only
(with optional `grpcio` / `tpu_metric_service` proto stubs for direct gRPC).
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
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request

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
    "0x0075": ("tpu7x", "TPU tpu7x (Ironwood, pending live hardware confirmation #21)", 192 * 1024**3),
    "0x0076": ("tpu7x", "TPU tpu7x (Ironwood)", 192 * 1024**3),
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

FREETPU_SHM_MAGIC = 0x46545055
FREETPU_SHM_VERSION = 1
_SHM_HEADER_STRUCT = struct.Struct("<IIQ")
_SHM_PAYLOAD_STRUCT = struct.Struct("<QQQQ")
_SHM_SEQ_STRUCT = struct.Struct("<Q")
_SHM_FULL_STRUCT = struct.Struct("<IIQQQQQ")  # 48 bytes; matches freetpu_shm_metrics_v1

DEFAULT_GCP_METADATA_PREEMPTED_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
)


@dataclass
class ShmMetrics:
    hbm_allocated_bytes: int
    hbm_used_bytes: int
    mxu_active_cycles: int
    dma_bytes_transferred: int
    seq: int


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
    hbm_bandwidth_utilization_percent: Optional[float] = None
    aer_correctable_total: Optional[int] = None
    aer_fatal_total: Optional[int] = None
    shm_hbm_allocated_bytes: Optional[int] = None
    mxu_active_cycles: Optional[int] = None
    dma_bytes_transferred: Optional[int] = None


@dataclass
class RuntimeTelemetry:
    runtime_chip: str
    mapped_pci_bdf: Optional[str]
    hbm_used_bytes: Optional[int]
    hbm_total_bytes: Optional[int]
    duty_cycle_percent: Optional[float]
    hbm_bandwidth_utilization_percent: Optional[float] = None


@dataclass
class ExporterSnapshot:
    timestamp_unix: float
    scrape_duration_seconds: float
    tpu_info_available: bool
    tpu_info_error: Optional[str]
    pid_telemetry_available: bool
    devices: List[TpuTelemetry]
    runtime_utilization: List[RuntimeTelemetry] = field(default_factory=list)
    gcp_preempted_status: Optional[int] = None


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
    "tpu7x": "V7X",
}

# Logical runtime devices/cores per physical TPU chip (matches upstream
# `tpu_info.device.TpuChip`: v2/v3/v7x expose 2 logical devices per physical
# chip while v4/v5e/v5p/v6e expose 1). `libtpu` reports HBM metrics per logical
# device/core and `tpu.runtime.tensorcore.dutycycle.percent` per physical chip.
_GEN_DEVICES_PER_CHIP: Dict[str, int] = {
    "v2": 2,
    "v3": 2,
    "v2-v3": 2,
    "v4": 1,
    "v4lite": 1,
    "v5e": 1,
    "v5p": 1,
    "v6e": 1,
    "v6e-pf": 1,
    "tpu7x": 2,
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


def _procfs_hides_pids(proc_root: Path) -> bool:
    """Return True if procfs mount options restrict PID visibility (`hidepid`) for this caller."""
    mountinfo_path = proc_root / "self" / "mountinfo"
    is_real_linux_proc = sys.platform.startswith("linux") and os.fspath(proc_root) == "/proc"
    if not mountinfo_path.exists() and not is_real_linux_proc:
        return False

    try:
        raw = mountinfo_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True

    target_mounts = {os.fspath(proc_root), "/proc"}
    matched_opts: Optional[List[str]] = None
    for line in raw.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            continue
        mount_point = left_fields[4]
        fs_type = right_fields[0]
        if fs_type == "proc" and mount_point in target_mounts:
            matched_opts = left_fields[5].split(",") + right_fields[2].split(",")

    if matched_opts is None:
        return is_real_linux_proc

    hidepid_val: Optional[str] = None
    gid_val: Optional[str] = None
    for opt in matched_opts:
        if opt.startswith("hidepid="):
            hidepid_val = opt.split("=", 1)[1].strip().lower()
        elif opt.startswith("gid="):
            gid_val = opt.split("=", 1)[1].strip()

    if not hidepid_val or hidepid_val in ("0", "off"):
        return False

    # Linux `hidepid=ptraceable` (numeric 4, HIDEPID_NOT_PTRACEABLE) applies a
    # ptrace visibility check BEFORE considering `gid=`; fail closed so restricted
    # ptrace never produces a silent undercount marked available.
    if hidepid_val in ("4", "ptraceable"):
        return True
    try:
        if (int(hidepid_val, 0) & 4) != 0:
            return True
    except ValueError:
        pass

    try:
        caller_gids = {os.getegid(), *os.getgroups()}
    except OSError:
        caller_gids = {os.getegid()} if hasattr(os, "getegid") else set()

    if gid_val is not None:
        try:
            bypass_gid = int(gid_val)
        except ValueError:
            return True
        return bypass_gid not in caller_gids

    # Without an explicit `gid=` mount option, Linux defaults `pid_gid` to
    # GLOBAL_ROOT_GID (0).
    return not (hasattr(os, "geteuid") and os.geteuid() == 0 and 0 in caller_gids)


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

    if _procfs_hides_pids(proc_root):
        return {vp: [] for vp in verified_rdevs}, False, 1, verified_rdevs

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
    hbm_bw_util_percent: Optional[float] = None
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


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    for pipe in (proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


def _run_subprocess_bounded(
    cmd: Sequence[str],
    *,
    timeout_s: float,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a child command in its own process group and kill the full group on timeout."""
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(
            args=list(cmd),
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except BaseException:
        _terminate_process_group(proc)
        raise


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
        proc = _run_subprocess_bounded(
            [resolved],
            timeout_s=timeout_s,
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
        proc = _run_subprocess_bounded(
            [sys.executable, "-c", _PY_METRICS_SUBPROCESS_HELPER, chip_attr, addr, str(deadline_s)],
            timeout_s=deadline_s,
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


_GRPC_CHANNEL_LOCK = threading.Lock()
_GRPC_CHANNELS: Dict[str, Any] = {}
_GRPC_METRIC_NAMES = (
    "tpu.runtime.tensorcore.dutycycle.percent",
    "tpu.runtime.hbm.memory.usage.bytes",
    "tpu.runtime.hbm.memory.total.bytes",
    "tpu.runtime.hbm.utilization.percent",
)


def _import_optional_module(name: str) -> Optional[Any]:
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None


def _load_grpc_runtime_modules() -> Optional[Tuple[Any, Any, Any]]:
    grpc_mod = _import_optional_module("grpc")
    if grpc_mod is None:
        return None
    for pb2_name, pb2_grpc_name in (
        ("tpu_info.proto.tpu_metric_service_pb2", "tpu_info.proto.tpu_metric_service_pb2_grpc"),
        ("tpu_metric_service_pb2", "tpu_metric_service_pb2_grpc"),
    ):
        pb2 = _import_optional_module(pb2_name)
        pb2_grpc = _import_optional_module(pb2_grpc_name)
        if pb2 is not None and pb2_grpc is not None:
            return grpc_mod, pb2, pb2_grpc
    return None


def _get_grpc_channel(grpc_mod: Any, addr: str) -> Any:
    with _GRPC_CHANNEL_LOCK:
        channel = _GRPC_CHANNELS.get(addr)
        if channel is None:
            channel = grpc_mod.insecure_channel(addr)
            _GRPC_CHANNELS[addr] = channel
        return channel


def _extract_grpc_chip_id(metric_item: Any) -> Optional[str]:
    """Decode device/chip ID from published `tpu_metric_service.proto` `Metric.attribute.value`."""
    attr = getattr(metric_item, "attribute", None)
    if attr is None:
        return None
    val = getattr(attr, "value", None)
    if val is None:
        return None

    which_fn = getattr(val, "WhichOneof", None)
    if callable(which_fn):
        try:
            selected = which_fn("attr")
        except Exception:  # noqa: BLE001
            selected = None
        if selected == "string_attr":
            str_val = getattr(val, "string_attr", None)
            return str_val.strip() if isinstance(str_val, str) and str_val.strip() else None
        if selected == "int_attr":
            int_val = getattr(val, "int_attr", None)
            return str(int_val) if isinstance(int_val, int) and not isinstance(int_val, bool) else None
        return None

    str_val = getattr(val, "string_attr", None)
    if isinstance(str_val, str) and str_val.strip():
        return str_val.strip()
    int_val = getattr(val, "int_attr", None)
    if isinstance(int_val, int) and not isinstance(int_val, bool):
        return str(int_val)
    return None


def _extract_grpc_gauge_value(metric_item: Any) -> Optional[float]:
    """Decode numeric gauge reading from `tpu_metric_service.proto` `Metric.gauge`."""
    gauge = getattr(metric_item, "gauge", None)
    if gauge is None:
        return None
    which_fn = getattr(gauge, "WhichOneof", None)
    if callable(which_fn):
        try:
            selected = which_fn("value")
        except Exception:  # noqa: BLE001
            selected = None
        if selected not in ("as_double", "as_int"):
            return None
        raw = getattr(gauge, selected, None)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            fval = float(raw)
            return fval if math.isfinite(fval) else None
        return None

    for field_name in ("as_double", "as_int"):
        raw = getattr(gauge, field_name, None)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            fval = float(raw)
            if math.isfinite(fval):
                return fval
    return None


def _infer_devices_per_chip(devices: Sequence[TpuTelemetry]) -> int:
    """Infer logical runtime devices per physical chip from discovered PCI generations."""
    counts = {
        _GEN_DEVICES_PER_CHIP[d.generation]
        for d in devices
        if d.generation in _GEN_DEVICES_PER_CHIP
    }
    if len(counts) == 1:
        return next(iter(counts))
    return 1


def _query_grpc_runtime_metrics(
    devices: Sequence[TpuTelemetry],
    addr: Optional[str] = "127.0.0.1:8431",
    timeout_s: float = 5.0,
) -> Tuple[bool, Optional[str], Dict[str, _TpuInfoRow]]:
    """Query `libtpu` RuntimeMetricService directly in-process over gRPC (`localhost:8431`).

    Eliminates `tpu-info` CLI/Python subprocess churn and works cleanly when
    `grpcio` and `tpu_metric_service` proto stubs are installed. Maps per-chip
    `tpu.runtime.tensorcore.dutycycle.percent` readings across each chip's
    logical runtime devices using `_GEN_DEVICES_PER_CHIP` (matching upstream
    `tpu_info/metrics.py` & `device.py`, e.g. 2 logical cores per physical chip
    on Ironwood `tpu7x` and v2/v3) before joining with per-core HBM series.
    Falls back gracefully when `grpcio` or proto stubs are unavailable.
    """
    if not addr:
        return False, "runtime metrics gRPC address disabled", {}

    mods = _load_grpc_runtime_modules()
    if mods is None:
        return False, "grpcio/tpu_metric_service stubs unavailable", {}
    grpc_mod, pb2, pb2_grpc = mods

    stub_cls = getattr(pb2_grpc, "RuntimeMetricServiceStub", None)
    req_cls = getattr(pb2, "MetricRequest", None)
    if stub_cls is None or req_cls is None:
        return False, "RuntimeMetricServiceStub or MetricRequest missing from proto stubs", {}

    deadline = time.monotonic() + max(0.05, float(timeout_s))
    try:
        channel = _get_grpc_channel(grpc_mod, addr)
        stub = stub_cls(channel)
    except Exception as exc:  # noqa: BLE001
        return False, f"gRPC channel init failed ({addr}): {exc}", {}

    rpc_fn = getattr(stub, "GetRuntimeMetric", None)
    if not callable(rpc_fn):
        return False, "RuntimeMetricServiceStub.GetRuntimeMetric unavailable", {}

    devices_per_chip = _infer_devices_per_chip(devices)
    rows: Dict[str, _TpuInfoRow] = {}
    duty_by_chip: Dict[str, float] = {}
    last_err: Optional[str] = None
    any_rpc_ok = False
    for metric_name in _GRPC_METRIC_NAMES:
        remaining = deadline - time.monotonic()
        if remaining <= 0.01:
            last_err = f"gRPC RuntimeMetricService timed out ({addr})"
            break
        try:
            resp = rpc_fn(req_cls(metric_name=metric_name), timeout=remaining)
            any_rpc_ok = True
        except Exception as exc:  # noqa: BLE001
            last_err = f"gRPC GetRuntimeMetric({metric_name}) failed ({addr}): {exc}"
            continue

        metric_obj = getattr(resp, "metric", None)
        entries = getattr(metric_obj, "metrics", None) if metric_obj is not None else None
        if not entries:
            continue
        for item in entries:
            chip_id = _extract_grpc_chip_id(item)
            val = _extract_grpc_gauge_value(item)
            if chip_id is None or val is None or val < 0.0:
                continue
            if metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                duty_by_chip[chip_id] = float(val)
                continue
            row = rows.setdefault(chip_id, _TpuInfoRow(chip_key=chip_id))
            row.in_util_table = True
            if metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                row.hbm_used_bytes = int(round(val))
            elif metric_name == "tpu.runtime.hbm.memory.total.bytes" and val > 0:
                row.hbm_total_bytes = int(round(val))
            elif metric_name == "tpu.runtime.hbm.utilization.percent":
                row.hbm_bw_util_percent = float(val)

    # Join per-chip duty cycle onto logical runtime cores using devices_per_chip:
    # chip `c` covers logical cores `c * devices_per_chip + offset` (core_id // devices_per_chip == c).
    for chip_key, duty_val in duty_by_chip.items():
        if devices_per_chip > 1 and chip_key.isdigit():
            chip_idx = int(chip_key)
            for offset in range(devices_per_chip):
                core_key = str(chip_idx * devices_per_chip + offset)
                row = rows.setdefault(core_key, _TpuInfoRow(chip_key=core_key))
                row.in_util_table = True
                row.duty_cycle_percent = duty_val
        else:
            row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))
            row.in_util_table = True
            row.duty_cycle_percent = duty_val

    if not rows:
        if not any_rpc_ok and last_err:
            return False, last_err, {}
        return False, last_err or f"gRPC RuntimeMetricService returned no live metrics from {addr}", {}
    return True, None, rows


def read_shm_seqlock_metrics(
    pci_bdf: str,
    shm_root: Optional[Path] = Path("/dev/shm"),
    *,
    max_retries: int = 64,
    trusted_uids: Optional[Sequence[int]] = None,
) -> Optional[ShmMetrics]:
    """Read bare-metal `libopentpu.so` telemetry from `<shm_root>/freetpu_metrics_<bdf>` via SeqLock `pread`.

    Authenticates the publisher on the opened file descriptor before reading:
    - rejects world-writable directories that lack the sticky bit (`S_ISVTX`);
    - opens with `O_NOFOLLOW | O_NONBLOCK` to avoid symlink/FIFO TOCTOU races;
    - requires a regular file owned by an allowed UID (default: current `euid`)
      with neither group-write (`0o020`) nor world-write (`0o002`) permissions;
    - reads via `os.pread` (never `mmap`) so concurrent truncation cannot raise
      `SIGBUS` and instead fails closed with `None`.
    """
    if shm_root is None or not pci_bdf:
        return None
    if "/" in pci_bdf or "\\" in pci_bdf or ".." in pci_bdf:
        return None

    shm_dir = Path(shm_root)
    try:
        dir_st = os.stat(shm_dir)
    except OSError:
        return None
    if not stat.S_ISDIR(dir_st.st_mode):
        return None
    if (dir_st.st_mode & stat.S_IWOTH) != 0 and (dir_st.st_mode & stat.S_ISVTX) == 0:
        return None

    allowed_uids = (
        {int(u) for u in trusted_uids}
        if trusted_uids is not None
        else {os.geteuid()}
    )

    shm_path = shm_dir / f"freetpu_metrics_{pci_bdf}"
    open_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(shm_path, open_flags)
    except OSError:
        return None
    try:
        fd_st = os.fstat(fd)
        if not stat.S_ISREG(fd_st.st_mode):
            return None
        if fd_st.st_uid not in allowed_uids:
            return None
        if (fd_st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) != 0:
            return None
        if fd_st.st_size < _SHM_FULL_STRUCT.size:
            return None

        for _ in range(max(1, max_retries)):
            hdr = os.pread(fd, _SHM_HEADER_STRUCT.size, 0)
            if len(hdr) < _SHM_HEADER_STRUCT.size:
                return None
            magic, version, seq1 = _SHM_HEADER_STRUCT.unpack_from(hdr, 0)
            if magic != FREETPU_SHM_MAGIC or version != FREETPU_SHM_VERSION:
                return None
            if (seq1 & 1) != 0:
                time.sleep(0)
                continue
            payload = os.pread(fd, _SHM_PAYLOAD_STRUCT.size, 16)
            if len(payload) < _SHM_PAYLOAD_STRUCT.size:
                return None
            seq2_raw = os.pread(fd, _SHM_SEQ_STRUCT.size, 8)
            if len(seq2_raw) < _SHM_SEQ_STRUCT.size:
                return None
            (seq2,) = _SHM_SEQ_STRUCT.unpack_from(seq2_raw, 0)
            if seq1 == seq2:
                hbm_alloc, hbm_used, mxu_cycles, dma_bytes = _SHM_PAYLOAD_STRUCT.unpack_from(
                    payload, 0
                )
                return ShmMetrics(
                    hbm_allocated_bytes=hbm_alloc,
                    hbm_used_bytes=hbm_used,
                    mxu_active_cycles=mxu_cycles,
                    dma_bytes_transferred=dma_bytes,
                    seq=seq1,
                )
        return None
    except (OSError, ValueError, struct.error):
        return None
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _parse_sysfs_aer_file(path: Path) -> Optional[int]:
    """Parse Linux sysfs PCIe AER file (`aer_dev_correctable` or `aer_dev_fatal`)."""
    raw = _read_text(path)
    if raw is None or not raw:
        return None
    try:
        val = int(raw, 0)
        return val if val >= 0 else None
    except ValueError:
        pass

    subtotal = 0
    saw_field = False
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        key, num_str = parts
        try:
            count = int(num_str, 0)
        except ValueError:
            continue
        if count < 0:
            continue
        if key.upper().startswith("TOTAL_ERR_"):
            return count
        subtotal += count
        saw_field = True
    return subtotal if saw_field else None


def read_pcie_aer_counters(pci_entry: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read pure `libtpu`-free sysfs PCIe AER correctable and fatal error counts."""
    return (
        _parse_sysfs_aer_file(pci_entry / "aer_dev_correctable"),
        _parse_sysfs_aer_file(pci_entry / "aer_dev_fatal"),
    )


def query_gcp_preemption_status(
    url: str = DEFAULT_GCP_METADATA_PREEMPTED_URL,
    timeout_s: float = 0.5,
) -> Optional[int]:
    """Query GCP instance preemption metadata (`1` if preempted, `0` if healthy, `None` if unavailable)."""
    req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(req, timeout=max(0.05, float(timeout_s))) as resp:
            body = resp.read(256).decode("utf-8", errors="replace").strip().lower()
    except (OSError, urllib.error.URLError, ValueError):
        return None
    if body == "true":
        return 1
    if body == "false":
        return 0
    return None


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


_VFIO_VIABLE_DRIVERS = frozenset({"vfio-pci", "pci-stub", "pcieport"})


def _is_vfio_group_viable(
    sysfs_root: Path,
    iommu_group: str,
    group_members: Dict[str, set[str]],
    group_dir_readable: Dict[str, bool],
) -> bool:
    """Return True iff every endpoint in the IOMMU group is unbound or bound to a VFIO/stub driver.

    Linux VFIO refuses to attach a group if any non-bridge device in that IOMMU
    group remains bound to a host driver. Fail closed when the authoritative
    `/sys/kernel/iommu_groups/<group>/devices` directory is missing/unreadable.
    """
    if not iommu_group or not group_dir_readable.get(iommu_group, False):
        return False
    members = group_members.get(iommu_group, set())
    if not members:
        return False
    pci_dir = sysfs_root / "bus" / "pci" / "devices"
    iommu_dev_root = sysfs_root / "kernel" / "iommu_groups" / iommu_group / "devices"
    for member_bdf in members:
        member_path = pci_dir / member_bdf
        if not member_path.exists():
            alt_path = iommu_dev_root / member_bdf
            if alt_path.is_dir():
                member_path = alt_path
            else:
                return False
        class_raw = _normalize_hex_id(_read_text(member_path / "class") or "")
        # Linux VFIO ignores PCI host/PCIe bridges (0x0600xx / 0x0604xx) for viability.
        if class_raw.startswith("0x0604") or class_raw.startswith("0x0600"):
            continue
        member_driver = _read_symlink_name(member_path / "driver")
        if member_driver and member_driver not in _VFIO_VIABLE_DRIVERS:
            return False
    return True


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
        pci_enabled = False
        if enable_raw is not None:
            try:
                pci_enabled = int(enable_raw, 0) > 0
            except ValueError:
                pci_enabled = False
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

        # On vfio-pci, PCI `enable` is 0 while idle; VFIO readiness requires
        # vfio-pci binding, a viable IOMMU group (no sibling endpoint bound to
        # a conflicting host driver), and verified non-symlink character device
        # nodes matching kernel sysfs major:minor for /dev/vfio/vfio and /dev/vfio/<group>.
        vfio_group_viable = _is_vfio_group_viable(
            sysfs_root, iommu_group, group_members, group_dir_readable
        )
        healthy = (
            driver == "vfio-pci"
            and bool(iommu_group)
            and vfio_group_viable
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

        aer_correctable, aer_fatal = read_pcie_aer_counters(entry)

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
            aer_correctable_total=aer_correctable,
            aer_fatal_total=aer_fatal,
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


def _has_live_utilization(rows: Dict[str, _TpuInfoRow]) -> bool:
    return any(
        r.in_util_table
        and (
            r.hbm_used_bytes is not None
            or r.hbm_total_bytes is not None
            or r.duty_cycle_percent is not None
            or r.hbm_bw_util_percent is not None
        )
        for r in rows.values()
    )


def _merge_runtime_rows(
    base_rows: Dict[str, _TpuInfoRow],
    extra_rows: Dict[str, _TpuInfoRow],
) -> Dict[str, _TpuInfoRow]:
    merged = dict(base_rows)
    for k, r in extra_rows.items():
        existing = merged.get(k)
        if existing is not None:
            if r.hbm_used_bytes is not None:
                existing.hbm_used_bytes = r.hbm_used_bytes
            if r.hbm_total_bytes is not None:
                existing.hbm_total_bytes = r.hbm_total_bytes
            if r.duty_cycle_percent is not None:
                existing.duty_cycle_percent = r.duty_cycle_percent
            if r.hbm_bw_util_percent is not None:
                existing.hbm_bw_util_percent = r.hbm_bw_util_percent
            existing.in_util_table = True
        else:
            merged[k] = r
    return merged


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
    shm_root: Optional[Path] = Path("/dev/shm"),
    shm_trusted_uids: Optional[Sequence[int]] = None,
    gcp_metadata_preemption: bool = False,
    gcp_metadata_url: str = DEFAULT_GCP_METADATA_PREEMPTED_URL,
    gcp_metadata_timeout_s: float = 0.5,
) -> ExporterSnapshot:
    """Collect PCI/VFIO discovery, SeqLock SHM, AER counters, optional procfs PIDs, and gRPC/`tpu-info` metrics."""
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
    runtime_source_label = "tpu-info"
    if not tpu_info_bin:
        # Empty/None `--tpu-info-bin` explicitly disables `libtpu` runtime
        # telemetry (direct gRPC, `tpu_info.metrics`, and CLI) across every mode.
        tpu_info_ok, tpu_info_err, parsed_rows = False, "tpu-info disabled", {}
    elif skip_pid_scan or explicit_non_default_runtime_addr:
        # Sidecar / no-PID mode OR explicit non-default `--runtime-metrics-addr`:
        # try direct in-process gRPC first, then fall back to `tpu_info.metrics`
        # within the remaining deadline. Never spawn the `tpu-info` CLI subprocess.
        tpu_info_ok, tpu_info_err, parsed_rows = _query_grpc_runtime_metrics(
            devices,
            addr=runtime_metrics_addr,
            timeout_s=runtime_budget_s,
        )
        if tpu_info_ok:
            runtime_source_label = "libtpu-grpc"
        else:
            remaining_s = runtime_deadline - time.monotonic()
            if remaining_s >= 0.05:
                tpu_info_ok, tpu_info_err, parsed_rows = _query_tpu_info_python_metrics(
                    devices,
                    addr=runtime_metrics_addr,
                    timeout_s=remaining_s,
                )
    else:
        # Prefer direct in-process `RuntimeMetricService` gRPC over spawning
        # `tpu-info` subprocesses when default `"tpu-info"` is configured; if a
        # caller supplied an explicit CLI script/path (e.g. custom binary or test
        # fixture) or gRPC stubs/endpoint are unavailable, fall back cleanly within
        # the single shared scrape deadline.
        custom_cli_path = os.path.sep in tpu_info_bin
        tpu_info_ok = False
        tpu_info_err = None
        parsed_rows = {}
        if not custom_cli_path and runtime_metrics_addr:
            tpu_info_ok, tpu_info_err, parsed_rows = _query_grpc_runtime_metrics(
                devices,
                addr=runtime_metrics_addr,
                timeout_s=runtime_budget_s,
            )
            if tpu_info_ok and _has_live_utilization(parsed_rows):
                runtime_source_label = "libtpu-grpc"

        remaining_s = runtime_deadline - time.monotonic()
        if (not tpu_info_ok or not _has_live_utilization(parsed_rows)) and remaining_s >= 0.05:
            cli_ok, cli_err, cli_rows = _run_tpu_info(
                tpu_info_bin=tpu_info_bin,
                timeout_s=remaining_s,
            )
            if cli_ok:
                tpu_info_ok = True
                tpu_info_err = None
                parsed_rows = _merge_runtime_rows(parsed_rows, cli_rows) if parsed_rows else cli_rows
            elif not tpu_info_ok:
                tpu_info_err = cli_err

            remaining_s = runtime_deadline - time.monotonic()
            if (
                (not tpu_info_ok or not _has_live_utilization(parsed_rows))
                and runtime_metrics_addr
                and remaining_s >= 0.05
            ):
                py_ok, _py_err, py_rows = _query_tpu_info_python_metrics(
                    devices,
                    addr=runtime_metrics_addr,
                    timeout_s=remaining_s,
                )
                if py_ok:
                    parsed_rows = _merge_runtime_rows(parsed_rows, py_rows)
                    tpu_info_ok, tpu_info_err = True, None
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
            dev.telemetry_source = runtime_source_label

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
                or matched_util.hbm_bw_util_percent is not None
            ):
                dev.telemetry_source = runtime_source_label
            if matched_util.hbm_used_bytes is not None:
                dev.hbm_used_bytes = matched_util.hbm_used_bytes
            if matched_util.hbm_total_bytes is not None:
                dev.hbm_total_bytes = matched_util.hbm_total_bytes
            if matched_util.duty_cycle_percent is not None:
                dev.duty_cycle_percent = matched_util.duty_cycle_percent
            if matched_util.hbm_bw_util_percent is not None:
                dev.hbm_bandwidth_utilization_percent = matched_util.hbm_bw_util_percent

        # Bare-metal libopentpu.so SeqLock /dev/shm/freetpu_metrics_<bdf> telemetry.
        shm = read_shm_seqlock_metrics(
            dev.pci_bdf,
            shm_root=shm_root,
            trusted_uids=shm_trusted_uids,
        )
        if shm is not None:
            dev.shm_hbm_allocated_bytes = shm.hbm_allocated_bytes
            dev.mxu_active_cycles = shm.mxu_active_cycles
            dev.dma_bytes_transferred = shm.dma_bytes_transferred
            if dev.hbm_used_bytes is None:
                dev.hbm_used_bytes = shm.hbm_used_bytes
            if shm.hbm_allocated_bytes > 0 and (
                dev.hbm_total_bytes is None or dev.telemetry_source == "sysfs-fallback"
            ):
                dev.hbm_total_bytes = shm.hbm_allocated_bytes
            if dev.telemetry_source == "sysfs-fallback":
                dev.telemetry_source = "libopentpu-shm"

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
            hbm_bandwidth_utilization_percent=row.hbm_bw_util_percent,
        )
        for key, row in util_rows.items()
        if (
            row.hbm_used_bytes is not None
            or row.hbm_total_bytes is not None
            or row.duty_cycle_percent is not None
            or row.hbm_bw_util_percent is not None
        )
    ]

    gcp_preempted: Optional[int] = None
    if gcp_metadata_preemption:
        gcp_preempted = query_gcp_preemption_status(
            url=gcp_metadata_url,
            timeout_s=gcp_metadata_timeout_s,
        )

    elapsed = time.monotonic() - t0
    return ExporterSnapshot(
        timestamp_unix=time.time(),
        scrape_duration_seconds=elapsed,
        tpu_info_available=tpu_info_ok,
        tpu_info_error=tpu_info_err,
        pid_telemetry_available=overall_pid_available,
        devices=devices,
        runtime_utilization=runtime_telemetry,
        gcp_preempted_status=gcp_preempted,
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
    hbm_bw_samples: List[Tuple[Dict[str, Any], str]] = []
    duty_samples: List[Tuple[Dict[str, Any], str]] = []
    aer_corr_samples: List[Tuple[Dict[str, Any], str]] = []
    aer_fatal_samples: List[Tuple[Dict[str, Any], str]] = []
    shm_hbm_alloc_samples: List[Tuple[Dict[str, Any], str]] = []
    mxu_cycles_samples: List[Tuple[Dict[str, Any], str]] = []
    dma_bytes_samples: List[Tuple[Dict[str, Any], str]] = []
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
        if d.hbm_bandwidth_utilization_percent is not None:
            hbm_bw_samples.append((base_labels, _fmt_float(d.hbm_bandwidth_utilization_percent)))
        if d.duty_cycle_percent is not None:
            duty_samples.append((base_labels, _fmt_float(d.duty_cycle_percent)))
        if d.aer_correctable_total is not None:
            aer_corr_samples.append((base_labels, str(d.aer_correctable_total)))
        if d.aer_fatal_total is not None:
            aer_fatal_samples.append((base_labels, str(d.aer_fatal_total)))
        if d.shm_hbm_allocated_bytes is not None:
            shm_hbm_alloc_samples.append((base_labels, str(d.shm_hbm_allocated_bytes)))
        if d.mxu_active_cycles is not None:
            mxu_cycles_samples.append((base_labels, str(d.mxu_active_cycles)))
        if d.dma_bytes_transferred is not None:
            dma_bytes_samples.append((base_labels, str(d.dma_bytes_transferred)))

        if d.pid_telemetry_available:
            pid_count_samples.append((base_labels, str(len(d.active_pids))))
            for pid in d.active_pids:
                pid_active_samples.append(({**base_labels, "pid": str(pid)}, "1"))

    runtime_hbm_used: List[Tuple[Dict[str, Any], str]] = []
    runtime_hbm_total: List[Tuple[Dict[str, Any], str]] = []
    runtime_duty: List[Tuple[Dict[str, Any], str]] = []
    runtime_hbm_bw: List[Tuple[Dict[str, Any], str]] = []
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
        if rt.hbm_bandwidth_utilization_percent is not None:
            runtime_hbm_bw.append((rt_labels, _fmt_float(rt.hbm_bandwidth_utilization_percent)))

    add_metric(
        "freetpu_tpu_info",
        "Static inventory metadata for each discovered TPU device.",
        "gauge",
        info_samples,
    )
    add_metric(
        "freetpu_tpu_health_status",
        "VFIO TPU chip readiness on VFIO-backed hosts (1 = vfio-pci bound, viable IOMMU group, and verified /dev/vfio nodes; idle chips stay ready; 0 on non-VFIO /dev/accel* hosts).",
        "gauge",
        health_samples,
    )
    add_metric(
        "freetpu_tpu_pci_enabled",
        "PCI sysfs enable state (1 when sysfs enable reference count > 0, 0 otherwise).",
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
        "freetpu_tpu_hbm_bandwidth_utilization_percent",
        "TPU HBM memory bandwidth utilization percentage (0 to 100).",
        "gauge",
        hbm_bw_samples,
    )
    add_metric(
        "freetpu_tpu_duty_cycle_percent",
        "TPU MXU duty cycle utilization percentage (0 to 100).",
        "gauge",
        duty_samples,
    )
    add_metric(
        "freetpu_pcie_aer_correctable_total",
        "Cumulative PCIe Advanced Error Reporting (AER) correctable error count from sysfs.",
        "counter",
        aer_corr_samples,
    )
    add_metric(
        "freetpu_pcie_aer_fatal_total",
        "Cumulative PCIe Advanced Error Reporting (AER) fatal error count from sysfs.",
        "counter",
        aer_fatal_samples,
    )
    add_metric(
        "freetpu_tpu_shm_hbm_allocated_bytes",
        "Bare-metal libopentpu.so shared-memory HBM pool capacity in bytes.",
        "gauge",
        shm_hbm_alloc_samples,
    )
    add_metric(
        "freetpu_tpu_mxu_active_cycles_total",
        "Cumulative MXU active cycles reported via libopentpu.so SeqLock shared memory.",
        "counter",
        mxu_cycles_samples,
    )
    add_metric(
        "freetpu_tpu_dma_bytes_transferred_total",
        "Cumulative DMA bytes transferred reported via libopentpu.so SeqLock shared memory.",
        "counter",
        dma_bytes_samples,
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
    add_metric(
        "freetpu_tpu_runtime_hbm_bandwidth_utilization_percent",
        "TPU Runtime Utilization HBM bandwidth utilization percentage keyed by libtpu runtime chip ID.",
        "gauge",
        runtime_hbm_bw,
    )
    if snapshot.gcp_preempted_status is not None:
        add_metric(
            "gcp_metadata_preempted_status",
            "GCP Spot/Preemptible VM preemption status from instance metadata (1 = preempted, 0 = active).",
            "gauge",
            [({}, str(snapshot.gcp_preempted_status))],
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
        shm_root: Optional[Path] = Path("/dev/shm"),
        shm_trusted_uids: Optional[Sequence[int]] = None,
        gcp_metadata_preemption: bool = False,
        gcp_metadata_url: str = DEFAULT_GCP_METADATA_PREEMPTED_URL,
        gcp_metadata_timeout_s: float = 0.5,
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
        self.shm_root = shm_root
        self.shm_trusted_uids = tuple(shm_trusted_uids) if shm_trusted_uids is not None else None
        self.gcp_metadata_preemption = gcp_metadata_preemption
        self.gcp_metadata_url = gcp_metadata_url
        self.gcp_metadata_timeout_s = gcp_metadata_timeout_s
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
                shm_root=self.shm_root,
                shm_trusted_uids=self.shm_trusted_uids,
                gcp_metadata_preemption=self.gcp_metadata_preemption,
                gcp_metadata_url=self.gcp_metadata_url,
                gcp_metadata_timeout_s=self.gcp_metadata_timeout_s,
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
        dest="no_pid_scan",
        action="store_true",
        default=False,
        help=(
            "Disable `/proc/*/fd` PID scanning and query `tpu_info.metrics` directly "
            "over loopback gRPC (recommended for containers without host PID visibility)."
        ),
    )
    parser.add_argument(
        "--pid-scan",
        dest="no_pid_scan",
        action="store_false",
        help=(
            "Explicitly enable `/proc/*/fd` host PID scanning (overrides `--no-pid-scan`, "
            "e.g. when running the Docker image with `--pid=host --cap-add=SYS_PTRACE`)."
        ),
    )
    parser.add_argument(
        "--runtime-metrics-addr",
        default="127.0.0.1:8431",
        help=(
            "Loopback `libtpu` gRPC address for direct RuntimeMetricService queries "
            "(default: 127.0.0.1:8431; empty string disables direct gRPC queries)."
        ),
    )
    parser.add_argument(
        "--shm-root",
        type=Path,
        default=Path("/dev/shm"),
        help="Shared-memory directory for libopentpu.so SeqLock files (default: /dev/shm).",
    )
    parser.add_argument(
        "--shm-trusted-uid",
        dest="shm_trusted_uids",
        action="append",
        type=int,
        default=None,
        help=(
            "Trusted owner UID for SeqLock SHM files (repeatable; defaults to "
            "the exporter's effective UID only; group/world-writable files are always rejected)."
        ),
    )
    parser.add_argument(
        "--gcp-metadata-preemption",
        action="store_true",
        default=False,
        help="Opt-in fast poll of GCP Spot/Preemptible instance metadata for gcp_metadata_preempted_status.",
    )
    parser.add_argument(
        "--gcp-metadata-url",
        default=DEFAULT_GCP_METADATA_PREEMPTED_URL,
        help=f"Override GCP metadata preemption URL (default: {DEFAULT_GCP_METADATA_PREEMPTED_URL}).",
    )
    parser.add_argument(
        "--gcp-metadata-timeout",
        type=float,
        default=0.5,
        help="Fast timeout in seconds for GCP metadata preemption queries (default: 0.5).",
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
                shm_root=args.shm_root,
                shm_trusted_uids=args.shm_trusted_uids,
                gcp_metadata_preemption=args.gcp_metadata_preemption,
                gcp_metadata_url=args.gcp_metadata_url,
                gcp_metadata_timeout_s=args.gcp_metadata_timeout,
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
        shm_root=args.shm_root,
        shm_trusted_uids=args.shm_trusted_uids,
        gcp_metadata_preemption=args.gcp_metadata_preemption,
        gcp_metadata_url=args.gcp_metadata_url,
        gcp_metadata_timeout_s=args.gcp_metadata_timeout,
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

    prev_sigterm = None
    if threading.current_thread() is threading.main_thread():
        def _handle_sigterm(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        prev_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        sys.stderr.write(
            f"freetpu tpu-exporter listening on http://{args.host}:{server.server_address[1]}/metrics\n"
        )
        sys.stderr.flush()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)
        server.server_close()
        if health_server is not None:
            health_server.shutdown()
            health_server.server_close()
        if health_thread is not None:
            health_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
