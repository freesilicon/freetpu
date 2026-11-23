#!/usr/bin/env python3
"""FreeTPU Prometheus Telemetry Exporter (DCGM-exporter equivalent for Cloud TPUs).

Discovers Google TPU PCI devices (vendor 0x1ae0) under /sys/bus/pci/devices/*
and their /dev/vfio/* IOMMU device nodes on VFIO-backed Cloud TPU VMs (v5e,
v5p, v6e Trillium, and tpu7x pending live HW confirmation), scrapes:
  - Direct `libtpu` gRPC / `RuntimeMetricService` (`localhost:8431`) when
    `grpcio` and protobuf stubs are available (with graceful fallback and zero
    unconditional `tpu-info` subprocess churn)
  - Lockless SeqLock `/run/freetpu-metrics/freetpu_metrics_<bdf>` `pread` telemetry
    for bare-metal `libopentpu.so` (`ops/tpu-exporter/freetpu_metrics_shm.h`),
    authenticated via publisher-controlled directory + opened-fd owner UID and
    non-group/world-writable permissions
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
import http.client
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import ipaddress
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
import urllib.parse
import urllib.request

GOOGLE_PCI_VENDOR_ID = "0x1ae0"

# Known Google Cloud TPU PCI device IDs under vendor 0x1ae0.
# Values: (generation_slug, display_name, nominal_hbm_bytes_per_pci_device)
# Note: Ironwood (`tpu7x`, `0x0075`/`0x0076`) uses a dual-chiplet architecture
# (192 GiB per physical chip = 96 GiB per exposed PCI function/chiplet, so a
# 4-chip / 8-PCI-function VM aggregates to 8 * 96 GiB = 768 GiB).
TPU_PCI_DEVICES: Dict[str, Tuple[str, str, int]] = {
    "0x0027": ("v2-v3", "TPU v2/v3", 16 * 1024**3),
    "0x0056": ("v4lite", "TPU v4lite", 8 * 1024**3),
    "0x005e": ("v4", "TPU v4", 32 * 1024**3),
    "0x0062": ("v5p", "TPU v5p", 95 * 1024**3),
    "0x0063": ("v5e", "TPU v5e", 16 * 1024**3),
    "0x006e": ("v6e-pf", "TPU v6e (PF)", 32 * 1024**3),
    "0x006f": ("v6e", "TPU v6e (Trillium)", 32 * 1024**3),
    "0x0075": ("tpu7x", "TPU tpu7x (Ironwood, pending live hardware confirmation #21)", 96 * 1024**3),
    "0x0076": ("tpu7x", "TPU tpu7x (Ironwood)", 96 * 1024**3),
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

FREETPU_SHM_MAGIC = 0x55505446  # ASCII "FTPU" in little-endian byte order ('F','T','P','U')
FREETPU_SHM_VERSION = 1
DEFAULT_SHM_ROOT = Path("/run/freetpu-metrics")
DEFAULT_SHM_MAX_AGE_S = 30.0
_SHM_HEADER_STRUCT = struct.Struct("<IIQ")
_SHM_PAYLOAD_STRUCT = struct.Struct("<QQQQQ")
_SHM_SEQ_STRUCT = struct.Struct("<Q")
_SHM_FULL_STRUCT = struct.Struct("<IIQQQQQQ")  # 56 bytes; matches freetpu_shm_metrics_v1

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
    timestamp_ns: int = 0


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
    sibling_aer_counters: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)
    sibling_iommu_groups: List[str] = field(default_factory=list)
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
    telemetry_source: Optional[str] = None


@dataclass
class ExporterSnapshot:
    timestamp_unix: float
    scrape_duration_seconds: float
    tpu_info_available: bool
    tpu_info_error: Optional[str]
    pid_telemetry_available: bool
    devices: List[TpuTelemetry]
    runtime_utilization: List[RuntimeTelemetry] = field(default_factory=list)
    runtime_unjoined_duty_cycle: Dict[str, float] = field(default_factory=dict)
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
    unjoined_duty_cycle_percent: Optional[float] = None
    hbm_bw_util_percent: Optional[float] = None
    in_chips_table: bool = False
    in_util_table: bool = False
    duty_unset_in_grpc: bool = False
    duty_ids_verified: bool = False
    sources: List[str] = field(default_factory=list)


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
    - Modern `tpu-info` dashboard tables: `TPU Device Status` (Memory Usage,
      HBM Bandwidth Util, and TensorCore Duty Cycle, ignoring Memory Bar Gauge
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
                elif "bandwidth" in hdr or ("hbm" in hdr and "util" in hdr):
                    col_map["hbm_bw"] = idx
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
                elif "hbm_bw" not in col_map and (
                    "bandwidth" in hdr or ("hbm" in hdr and "util" in hdr)
                ):
                    col_map["hbm_bw"] = idx
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
            bw_idx = col_map.get("hbm_bw")
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
            if bw_idx is not None and bw_idx < len(cells):
                bw_match = _PCT_RE.search(cells[bw_idx])
                if bw_match:
                    bw_val = float(bw_match.group(1))
                    if math.isfinite(bw_val) and bw_val >= 0.0:
                        row.hbm_bw_util_percent = bw_val
        elif section == "device_status":
            row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))
            row.in_util_table = True
            mem_idx = col_map.get("mem", 1)
            bw_idx = col_map.get("hbm_bw", 3 if len(cells) >= 5 else -1)
            duty_idx = col_map.get("duty", 4)
            if mem_idx < len(cells):
                mem_match = _MEM_PAIR_RE.search(cells[mem_idx])
                if mem_match:
                    used_bytes = _parse_bytes(mem_match.group(1), mem_match.group(2))
                    total_bytes = _parse_bytes(mem_match.group(3), mem_match.group(4))
                    if used_bytes >= 0 and total_bytes > 0:
                        row.hbm_used_bytes = used_bytes
                        row.hbm_total_bytes = total_bytes
            if 0 <= bw_idx < len(cells):
                bw_match = _PCT_RE.search(cells[bw_idx])
                if bw_match:
                    bw_val = float(bw_match.group(1))
                    if math.isfinite(bw_val) and bw_val >= 0.0:
                        row.hbm_bw_util_percent = bw_val
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
            if dev_path.startswith("/"):
                dev_path = os.path.normpath(dev_path)
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
    "import inspect, json, math, sys, tpu_info.device as d, tpu_info.metrics as m\n"
    "chip_attr, addr, timeout_s = sys.argv[1], sys.argv[2], float(sys.argv[3])\n"
    "layout = sys.argv[4] if len(sys.argv) > 4 else 'auto'\n"
    "chip = getattr(d.TpuChip, chip_attr)\n"
    "def _invoke(fn):\n"
    "    sig = inspect.signature(fn)\n"
    "    kw = {}\n"
    "    if 'addr' in sig.parameters:\n"
    "        kw['addr'] = addr\n"
    "    if 'timeout' in sig.parameters:\n"
    "        kw['timeout'] = timeout_s\n"
    "    return fn(chip, **kw)\n"
    "def _verify_raw_ironwood_duty(usage_rows, dpc, *, per_chip_mode):\n"
    "    raw_fn = getattr(m, 'get_metric_data', None)\n"
    "    if not callable(raw_fn):\n"
    "        return None\n"
    "    try:\n"
    "        resp = raw_fn(addr, 'tpu.runtime.tensorcore.dutycycle.percent')\n"
    "    except Exception:\n"
    "        return None\n"
    "    metric_obj = getattr(resp, 'metric', None)\n"
    "    entries = getattr(metric_obj, 'metrics', None) if metric_obj is not None else None\n"
    "    if not entries:\n"
    "        return False\n"
    "    raw_ids = set()\n"
    "    for item in entries:\n"
    "        attr = getattr(item, 'attribute', None)\n"
    "        aval = getattr(attr, 'value', None) if attr is not None else None\n"
    "        if aval is None:\n"
    "            return False\n"
    "        aw = aval.WhichOneof('attr') if hasattr(aval, 'WhichOneof') else None\n"
    "        cid = getattr(aval, 'int_attr', None) if aw == 'int_attr' else getattr(aval, 'string_attr', None)\n"
    "        try:\n"
    "            cid_int = int(str(cid).strip(), 10)\n"
    "        except Exception:\n"
    "            return False\n"
    "        g = getattr(item, 'gauge', None)\n"
    "        gw = g.WhichOneof('value') if (g is not None and hasattr(g, 'WhichOneof')) else None\n"
    "        if gw not in ('as_double', 'as_int'):\n"
    "            return False\n"
    "        gv = float(getattr(g, gw))\n"
    "        if not math.isfinite(gv) or gv < 0.0:\n"
    "            return False\n"
    "        raw_ids.add(cid_int)\n"
    "    hbm_ids = set()\n"
    "    for r in (usage_rows or []):\n"
    "        did = getattr(r, 'device_id', None)\n"
    "        try:\n"
    "            hbm_ids.add(int(str(did).strip(), 10))\n"
    "        except Exception:\n"
    "            return False\n"
    "    if per_chip_mode:\n"
    "        expected_hbm = {cid * dpc + off for cid in raw_ids for off in range(dpc)}\n"
    "        return bool(raw_ids) and expected_hbm == hbm_ids\n"
    "    return bool(raw_ids) and raw_ids == hbm_ids\n"
    "per_chip_fallback = False\n"
    "duty_ids_verified = False\n"
    "new_fn = getattr(m, 'get_chip_usage_new', None) if chip_attr == 'V7X' else None\n"
    "if chip_attr == 'V7X' and layout == 'per-chip':\n"
    "    rows = _invoke(m.get_chip_usage)\n"
    "    v_ok = _verify_raw_ironwood_duty(rows, 2, per_chip_mode=True)\n"
    "    if v_ok is True:\n"
    "        duty_ids_verified = True\n"
    "    elif v_ok is False:\n"
    "        per_chip_fallback = True\n"
    "elif callable(new_fn):\n"
    "    try:\n"
    "        rows = _invoke(new_fn)\n"
    "        v_ok = _verify_raw_ironwood_duty(rows, 2, per_chip_mode=False)\n"
    "        if v_ok is True:\n"
    "            duty_ids_verified = True\n"
    "        elif v_ok is False:\n"
    "            per_chip_fallback = True\n"
    "    except (AssertionError, ValueError):\n"
    "        rows = _invoke(m.get_chip_usage)\n"
    "        per_chip_fallback = True\n"
    "else:\n"
    "    rows = _invoke(m.get_chip_usage)\n"
    "out = [\n"
    "    {\n"
    "        'device_id': getattr(r, 'device_id', None),\n"
    "        'memory_usage': getattr(r, 'memory_usage', None),\n"
    "        'total_memory': getattr(r, 'total_memory', None),\n"
    "        'duty_cycle_pct': getattr(r, 'duty_cycle_pct', None),\n"
    "        '_per_chip_fallback': per_chip_fallback,\n"
    "        '_duty_ids_verified': duty_ids_verified,\n"
    "    }\n"
    "    for r in (rows or [])\n"
    "]\n"
    "sys.stdout.write(json.dumps(out))\n"
)


_MAX_RUNTIME_DEVICE_ID = 65535


def _parse_runtime_device_id(raw: Any) -> Optional[int]:
    """Parse a bounded non-negative ASCII decimal runtime device ID (`0..65535`).

    Rejects booleans, negative integers, Unicode digits (such as `'²'` which
    `str.isdigit()` accepts but `int()` rejects), non-numeric strings, and
    oversized decimal strings (exceeding Python's `int_max_str_digits`) without
    raising.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw <= _MAX_RUNTIME_DEVICE_ID else None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or len(text) > 5 or not text.isascii() or not text.isdigit():
            return None
        try:
            val = int(text)
        except ValueError:
            return None
        return val if 0 <= val <= _MAX_RUNTIME_DEVICE_ID else None
    return None


def _verify_raw_python_per_chip_duty(
    metrics_mod: Any,
    addr: str,
    usage_rows: Sequence[Any],
    devices_per_chip: int,
    *,
    per_chip_mode: bool = True,
) -> Optional[bool]:
    """Verify raw `tensorcore.dutycycle` IDs and populated protobuf gauges when `get_metric_data` is available.

    Returns:
      - `True` when raw duty IDs exist, every gauge is populated/valid (not unset
        protobuf default `0.0`), and IDs match HBM core IDs 1:1 (expanding
        `chip_id * devices_per_chip + offset` when `per_chip_mode=True`, or
        direct `raw_ids == hbm_ids` when `per_chip_mode=False`).
      - `False` when raw duty data is queried and either mismatches HBM core IDs
        or contains an unset/invalid protobuf gauge.
      - `None` when `get_metric_data` is unavailable or raises.
    """
    raw_fn = getattr(metrics_mod, "get_metric_data", None)
    if not callable(raw_fn):
        return None
    try:
        resp = raw_fn(addr, "tpu.runtime.tensorcore.dutycycle.percent")
    except Exception:  # noqa: BLE001
        return None
    metric_obj = getattr(resp, "metric", None)
    entries = getattr(metric_obj, "metrics", None) if metric_obj is not None else None
    if not entries:
        return False
    raw_ids: set[int] = set()
    for item in entries:
        cid_str = _extract_grpc_chip_id(item)
        cid_int = _parse_runtime_device_id(cid_str)
        if cid_int is None:
            return False
        val = _extract_grpc_gauge_value(item)
        if val is None or val < 0.0:
            return False
        raw_ids.add(cid_int)
    hbm_ids: set[int] = set()
    for row in usage_rows:
        did = row.get("device_id") if isinstance(row, dict) else getattr(row, "device_id", None)
        did_int = _parse_runtime_device_id(did)
        if did_int is None:
            return False
        hbm_ids.add(did_int)
    if per_chip_mode:
        expected_hbm = {
            cid * devices_per_chip + off
            for cid in raw_ids
            for off in range(devices_per_chip)
        }
        return bool(raw_ids) and expected_hbm == hbm_ids
    return bool(raw_ids) and raw_ids == hbm_ids


def _parse_python_usage_entries(
    entries: Iterable[Any],
    addr: str,
    *,
    per_chip_fallback: bool = False,
    duty_ids_verified: bool = False,
    devices_per_chip: int = 2,
) -> Tuple[bool, Optional[str], Dict[str, _TpuInfoRow]]:
    rows: Dict[str, _TpuInfoRow] = {}
    for item in entries:
        if isinstance(item, dict):
            dev_id = item.get("device_id")
            mem_used = item.get("memory_usage")
            mem_total = item.get("total_memory")
            duty = item.get("duty_cycle_pct")
            item_per_chip_fb = bool(item.get("_per_chip_fallback", per_chip_fallback))
            item_duty_verified = bool(item.get("_duty_ids_verified", duty_ids_verified))
        else:
            dev_id = getattr(item, "device_id", None)
            mem_used = getattr(item, "memory_usage", None)
            mem_total = getattr(item, "total_memory", None)
            duty = getattr(item, "duty_cycle_pct", None)
            item_per_chip_fb = per_chip_fallback
            item_duty_verified = duty_ids_verified
        parsed_id = _parse_runtime_device_id(dev_id)
        if parsed_id is None:
            continue
        key = str(parsed_id)
        row = rows.setdefault(key, _TpuInfoRow(chip_key=key))
        row.in_util_table = True
        if (
            isinstance(mem_used, (int, float))
            and not isinstance(mem_used, bool)
            and math.isfinite(float(mem_used))
            and mem_used >= 0
        ):
            row.hbm_used_bytes = int(mem_used)
        if (
            isinstance(mem_total, (int, float))
            and not isinstance(mem_total, bool)
            and math.isfinite(float(mem_total))
            and mem_total > 0
        ):
            row.hbm_total_bytes = int(mem_total)
        if (
            isinstance(duty, (int, float))
            and not isinstance(duty, bool)
            and math.isfinite(float(duty))
            and duty >= 0
        ):
            if not item_per_chip_fb:
                row.duty_cycle_percent = float(duty)
                row.duty_ids_verified = item_duty_verified
            # When `get_chip_usage` ran only as a fallback after `get_chip_usage_new`
            # failed in `auto` mode (or raw duty chip IDs mismatched HBM core IDs),
            # preserve HBM usage/total rows and omit unverified duty readings.

    if not rows:
        return False, f"tpu_info.metrics returned no parseable chip entries from {addr}", {}
    return True, None, rows


def _query_tpu_info_python_metrics(
    devices: Sequence[TpuTelemetry],
    addr: Optional[str] = "127.0.0.1:8431",
    timeout_s: float = 5.0,
    grpc_duty_layout: str = "auto",
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
    inferred_dpc = _infer_devices_per_chip(devices) or 2

    # Support in-memory mocked `tpu_info.metrics` modules in unit tests with a
    # bounded worker thread; for real installed packages, execute in an isolated
    # subprocess with a hard OS timeout so hung RPCs cannot leak threads or locks.
    cached_metrics_mod = sys.modules.get("tpu_info.metrics")
    if cached_metrics_mod is not None and not getattr(cached_metrics_mod, "__file__", None):
        tpu_device_mod = sys.modules.get("tpu_info.device")
        get_chip_usage_new = (
            getattr(cached_metrics_mod, "get_chip_usage_new", None)
            if chip_attr == "V7X"
            else None
        )
        get_chip_usage_legacy = getattr(cached_metrics_mod, "get_chip_usage", None)
        tpu_chip_enum = getattr(tpu_device_mod, "TpuChip", None) if tpu_device_mod is not None else None
        if (
            (not callable(get_chip_usage_new) and not callable(get_chip_usage_legacy))
            or tpu_chip_enum is None
            or not hasattr(tpu_chip_enum, chip_attr)
        ):
            return False, "tpu_info.metrics.get_chip_usage unavailable", {}
        chip_type = getattr(tpu_chip_enum, chip_attr)

        box: Dict[str, Any] = {}
        done_ev = threading.Event()

        def _invoke_mock(fn: Any) -> Any:
            sig = inspect.signature(fn)
            kw: Dict[str, Any] = {}
            if "addr" in sig.parameters:
                kw["addr"] = addr
            if "timeout" in sig.parameters:
                kw["timeout"] = deadline_s
            return fn(chip_type, **kw)

        def _worker() -> None:
            try:
                if chip_attr == "V7X" and grpc_duty_layout == "per-chip" and callable(get_chip_usage_legacy):
                    box["rows"] = _invoke_mock(get_chip_usage_legacy)
                    v_ok = _verify_raw_python_per_chip_duty(
                        cached_metrics_mod, addr, box["rows"] or [], inferred_dpc, per_chip_mode=True
                    )
                    if v_ok is True:
                        box["duty_ids_verified"] = True
                    elif v_ok is False:
                        box["per_chip_fallback"] = True
                elif callable(get_chip_usage_new):
                    try:
                        box["rows"] = _invoke_mock(get_chip_usage_new)
                        v_ok = _verify_raw_python_per_chip_duty(
                            cached_metrics_mod, addr, box["rows"] or [], inferred_dpc, per_chip_mode=False
                        )
                        if v_ok is True:
                            box["duty_ids_verified"] = True
                        elif v_ok is False:
                            box["per_chip_fallback"] = True
                    except (AssertionError, ValueError):
                        if callable(get_chip_usage_legacy):
                            box["rows"] = _invoke_mock(get_chip_usage_legacy)
                            box["per_chip_fallback"] = True
                        else:
                            raise
                else:
                    box["rows"] = _invoke_mock(get_chip_usage_legacy)
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
        return _parse_python_usage_entries(
            usage_list,
            addr,
            per_chip_fallback=bool(box.get("per_chip_fallback", False)),
            duty_ids_verified=bool(box.get("duty_ids_verified", False)),
            devices_per_chip=inferred_dpc,
        )

    try:
        if importlib.util.find_spec("tpu_info") is None:
            return False, "tpu_info python API unavailable: No module named 'tpu_info'", {}
    except Exception as exc:  # noqa: BLE001
        return False, f"tpu_info python API unavailable: {exc}", {}

    try:
        proc = _run_subprocess_bounded(
            [
                sys.executable,
                "-c",
                _PY_METRICS_SUBPROCESS_HELPER,
                chip_attr,
                addr,
                str(deadline_s),
                grpc_duty_layout,
            ],
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
    return _parse_python_usage_entries(parsed_json, addr, devices_per_chip=inferred_dpc)


_GRPC_CHANNEL_LOCK = threading.Lock()
_GRPC_CHANNELS: Dict[str, Any] = {}
_GRPC_METRIC_NAMES = (
    "tpu.runtime.tensorcore.dutycycle.percent",
    "tpu.runtime.hbm.memory.usage.bytes",
    "tpu.runtime.hbm.memory.total.bytes",
    "tpu.runtime.hbm.utilization.percent",
)


_MIN_SAFE_INPROC_TPU_INFO_VERSION = (0, 14, 2)
_PEP440_PREFIX_RE = re.compile(r"^([0-9]+)\.([0-9]+)(?:\.([0-9]+))?(.*)$")


def _parse_safe_tpu_info_version(version_str: str) -> bool:
    """Return True when `version_str` is a release >= `0.14.2` (excludes pre/dev releases of 0.14.2)."""
    m = _PEP440_PREFIX_RE.match(version_str.strip())
    if not m:
        return False
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3) or 0)
    suffix = (m.group(4) or "").strip().lower()
    ver = (major, minor, patch)
    if ver > _MIN_SAFE_INPROC_TPU_INFO_VERSION:
        return True
    if ver < _MIN_SAFE_INPROC_TPU_INFO_VERSION:
        return False
    # Exact 0.14.2: accept final or post-release (e.g. "0.14.2", "0.14.2.post1"),
    # reject pre/dev releases (e.g. "0.14.2a1", "0.14.2rc1", "0.14.2.dev0").
    return suffix == "" or suffix.startswith(".post") or suffix.startswith("+")


def _can_safely_import_tpu_info_inproc() -> bool:
    """Gate in-process `tpu_info.*` imports on `tpu-info >= 0.14.2` to avoid fatal import-time crashes."""
    if (
        "tpu_info.proto.tpu_metric_service_pb2" in sys.modules
        and "tpu_info.proto.tpu_metric_service_pb2_grpc" in sys.modules
    ):
        return True
    try:
        installed_ver = importlib.metadata.version("tpu-info")
    except Exception:  # noqa: BLE001
        return False
    return _parse_safe_tpu_info_version(installed_ver)


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
    candidates: List[Tuple[str, str]] = []
    if _can_safely_import_tpu_info_inproc():
        candidates.append(
            ("tpu_info.proto.tpu_metric_service_pb2", "tpu_info.proto.tpu_metric_service_pb2_grpc")
        )
    # Standalone generated stubs never trigger `tpu_info/__init__.py`.
    candidates.append(("tpu_metric_service_pb2", "tpu_metric_service_pb2_grpc"))
    for pb2_name, pb2_grpc_name in candidates:
        pb2 = _import_optional_module(pb2_name)
        pb2_grpc = _import_optional_module(pb2_grpc_name)
        if pb2 is not None and pb2_grpc is not None:
            return grpc_mod, pb2, pb2_grpc
    return None


def _parse_local_grpc_target(addr: str) -> str:
    """Validate that `addr` targets a local transport and return `'uds'` or `'tcp'`.

    Matches upstream `tpu_info/metrics.py` local-only transport guarantees so
    process proxy settings or non-loopback targets cannot divert local `libtpu`
    telemetry to a remote endpoint.
    """
    target = addr.strip()
    if not target:
        raise ValueError("Empty gRPC runtime metrics address")
    if target.startswith("unix:") or target.startswith("unix-abstract:"):
        return "uds"

    raw = target
    for scheme in ("ipv4:", "ipv6:", "dns:///"):
        if raw.startswith(scheme):
            raw = raw[len(scheme):]
            break

    host = ""
    if raw.startswith("["):
        end_bracket = raw.find("]")
        if end_bracket <= 1:
            raise ValueError(f"Malformed IPv6 gRPC runtime metrics address: {addr!r}")
        host = raw[1:end_bracket]
    elif ":" in raw:
        host, _port = raw.rsplit(":", 1)
    else:
        host = raw

    host_lower = host.strip().lower()
    if host_lower == "localhost":
        return "tcp"
    try:
        if ipaddress.ip_address(host_lower).is_loopback:
            return "tcp"
    except ValueError:
        pass
    raise ValueError(
        f"Refusing non-local runtime metrics gRPC target {addr!r}: "
        "only loopback TCP (127.0.0.1, ::1, localhost) and local UDS endpoints are permitted"
    )


def _get_grpc_channel(grpc_mod: Any, addr: str) -> Any:
    transport = _parse_local_grpc_target(addr)
    with _GRPC_CHANNEL_LOCK:
        channel = _GRPC_CHANNELS.get(addr)
        if channel is None:
            local_creds_fn = getattr(grpc_mod, "local_channel_credentials", None)
            local_conn_type = getattr(grpc_mod, "LocalConnectionType", None)
            secure_channel_fn = getattr(grpc_mod, "secure_channel", None)
            if not (
                callable(local_creds_fn)
                and local_conn_type is not None
                and callable(secure_channel_fn)
            ):
                raise RuntimeError(
                    "grpc module lacks local_channel_credentials / secure_channel required for local-only libtpu telemetry"
                )
            conn_type = (
                local_conn_type.UDS if transport == "uds" else local_conn_type.LOCAL_TCP
            )
            creds = local_creds_fn(conn_type)
            channel = secure_channel_fn(
                addr,
                creds,
                options=[("grpc.enable_http_proxy", 0)],
            )
            _GRPC_CHANNELS[addr] = channel
        return channel


def _extract_grpc_chip_id(metric_item: Any) -> Optional[str]:
    """Decode and validate a bounded numeric device/chip ID from `tpu_metric_service.proto`."""
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
            parsed = _parse_runtime_device_id(getattr(val, "string_attr", None))
            return str(parsed) if parsed is not None else None
        if selected == "int_attr":
            parsed = _parse_runtime_device_id(getattr(val, "int_attr", None))
            return str(parsed) if parsed is not None else None
        return None

    str_parsed = _parse_runtime_device_id(getattr(val, "string_attr", None))
    if str_parsed is not None:
        return str(str_parsed)
    int_parsed = _parse_runtime_device_id(getattr(val, "int_attr", None))
    if int_parsed is not None:
        return str(int_parsed)
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


def _infer_devices_per_chip(devices: Sequence[TpuTelemetry]) -> Optional[int]:
    """Infer logical runtime devices per physical chip from discovered PCI generations.

    Returns `None` when PCI inventory is unavailable (e.g. sidecar mode without
    `/sys/bus/pci`), unrecognized, or mixed across generations with different
    logical-device counts so callers never assume 1:1 chip-to-core joins.
    """
    counts = {
        _GEN_DEVICES_PER_CHIP[d.generation]
        for d in devices
        if d.generation in _GEN_DEVICES_PER_CHIP
    }
    if len(counts) == 1:
        return next(iter(counts))
    return None


def _verified_dual_function_chip_count(
    devices: Sequence[TpuTelemetry],
    devices_per_chip: int,
) -> Optional[int]:
    """Return verified physical chip count ONLY when every visible PCI function
    belongs to a verified sibling group of size `devices_per_chip` (multifunction
    PCI slot siblings `DDDD:BB:DD.*` OR shared sysfs `chip_id` siblings) AND the
    host exposes at least 4 physical chips.

    Returns `None` on incomplete PCI inventory (e.g. 6 functions = 2 pairs + 2
    orphans, or only 2 of 4 chips visible) so `len(devices) // devices_per_chip`
    never understates the physical chip upper bound.
    """
    if devices_per_chip <= 1 or not devices:
        return None
    slot_counts: Dict[Tuple[str, str], int] = {}
    rid_counts: Dict[Tuple[str, str], int] = {}
    for d in devices:
        base_slot = d.pci_bdf.rsplit(".", 1)[0]
        slot_counts[(d.generation, base_slot)] = slot_counts.get((d.generation, base_slot), 0) + 1
        if d.runtime_chip_id:
            rid_counts[(d.generation, d.runtime_chip_id)] = (
                rid_counts.get((d.generation, d.runtime_chip_id), 0) + 1
            )

    verified_groups: set[Tuple[str, str, str]] = set()
    for d in devices:
        base_slot = d.pci_bdf.rsplit(".", 1)[0]
        if slot_counts.get((d.generation, base_slot), 0) == devices_per_chip:
            verified_groups.add(("slot", d.generation, base_slot))
        elif (
            d.runtime_chip_id
            and rid_counts.get((d.generation, d.runtime_chip_id), 0) == devices_per_chip
        ):
            verified_groups.add(("rid", d.generation, d.runtime_chip_id))
        else:
            # Unpaired or degraded function: cannot verify a tight upper bound.
            return None

    chip_count = len(verified_groups)
    # Require a full 4-chip host floor so partial container/degraded PCI views
    # exposing only 2 chips never mistake 4 per-chip IDs for per-core IDs.
    return chip_count if chip_count >= 4 else None


VALID_GRPC_DUTY_LAYOUTS = ("auto", "per-chip", "per-core")


def _query_grpc_runtime_metrics(
    devices: Sequence[TpuTelemetry],
    addr: Optional[str] = "127.0.0.1:8431",
    timeout_s: float = 5.0,
    grpc_duty_layout: str = "auto",
) -> Tuple[bool, Optional[str], Dict[str, _TpuInfoRow]]:
    """Query `libtpu` RuntimeMetricService directly in-process over gRPC (`localhost:8431`).

    Eliminates `tpu-info` CLI/Python subprocess churn and works cleanly when
    `grpcio` and `tpu_metric_service` proto stubs are installed. In `auto` mode,
    joins per-core duty cycle directly on single-device-per-chip generations
    (`v5e`, `v6e`, `v5p`, `v4`) and on verified complete dual-function hosts
    when duty IDs prove `per_core` via pigeonhole (`len(raw_duty_ids) >
    verified_chip_count`); expands legacy `v2`/`v3` per-chip replies only when
    verified full-host coverage matches; and retains ambiguous Ironwood (`tpu7x`)
    or sidecar replies in the separate unjoined series unless an explicit
    `--grpc-duty-layout` (`per-core` or `per-chip`) is configured.
    Falls back gracefully when `grpcio` or proto stubs are unavailable.
    """
    if grpc_duty_layout not in VALID_GRPC_DUTY_LAYOUTS:
        raise ValueError(
            f"Invalid --grpc-duty-layout {grpc_duty_layout!r}; "
            f"expected one of {', '.join(VALID_GRPC_DUTY_LAYOUTS)}"
        )
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
    unset_duty_ids: set[str] = set()
    raw_duty_ids: set[str] = set()
    raw_hbm_ids: set[str] = set()
    last_err: Optional[str] = None
    any_rpc_ok = False
    total_rpcs = len(_GRPC_METRIC_NAMES)
    for idx, metric_name in enumerate(_GRPC_METRIC_NAMES):
        remaining = deadline - time.monotonic()
        if remaining <= 0.01:
            last_err = f"gRPC RuntimeMetricService timed out ({addr})"
            break
        remaining_rpcs = total_rpcs - idx
        per_rpc_timeout = min(remaining, max(0.05, remaining / remaining_rpcs))
        try:
            resp = rpc_fn(req_cls(metric_name=metric_name), timeout=per_rpc_timeout)
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
            if chip_id is None:
                continue
            if metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                raw_duty_ids.add(chip_id)
            else:
                raw_hbm_ids.add(chip_id)
            val = _extract_grpc_gauge_value(item)
            if val is None or val < 0.0:
                if metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                    unset_duty_ids.add(chip_id)
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

    # Join duty-cycle readings with per-core HBM rows using ALL raw response IDs
    # collected BEFORE filtering unset/NaN/negative gauges:
    # - Explicit `--grpc-duty-layout=per-core` joins 1:1 directly.
    # - Explicit `--grpc-duty-layout=per-chip` expands `chip c -> c * dpc + off`
    #   when `devices_per_chip` is known from PCI inventory (otherwise remains
    #   `ambiguous` so we never guess a multiplier).
    # - In default `auto` mode:
    #   - When `devices_per_chip is None` (sidecar without `/sys/bus/pci` or
    #     mixed/unrecognized generations), keep readings `ambiguous` (unjoined).
    #   - On `devices_per_chip == 1` generations (`v5e`, `v6e`, `v5p`, `v4`),
    #     always join 1:1.
    #   - On Ironwood (`tpu7x`, `devices_per_chip == 2`), upstream `tpu-info`
    #     uses the per-device path (`metrics.get_chip_usage_new`), so a 4-ID
    #     duty reply `{0..3}` alongside 8 HBM IDs `{0..7}` is ambiguous with a
    #     partial per-core reply. Join `per_core` ONLY when a verified complete
    #     sibling PCI inventory (`verified_chip_count >= 4`, zero unpaired
    #     functions) proves `len(raw_duty_int_ids) > verified_chip_count` via
    #     pigeonhole; never auto-expand `tpu7x` per-chip without explicit
    #     `--grpc-duty-layout=per-chip`.
    #   - On legacy `v2`/`v3` (where `discover_tpu_devices` collapses dual-
    #     function 0x0027 into one record per physical chip and upstream
    #     `metrics.get_chip_usage` strictly expands per chip), allow full-host
    #     per-chip expansion when `len(raw_duty_int_ids) == len(devices) >= 4`
    #     and `expanded(raw_duty_int_ids) == all_hbm_core_ids`.
    if grpc_duty_layout == "per-core":
        duty_layout = "per_core"
    elif grpc_duty_layout == "per-chip":
        if devices_per_chip is None:
            duty_layout = "ambiguous"
        elif devices_per_chip == 1:
            duty_layout = "per_core"
        else:
            duty_layout = "per_chip"
    elif devices_per_chip is None:
        duty_layout = "ambiguous"
    elif devices_per_chip == 1:
        duty_layout = "per_core"
    else:
        duty_layout = "ambiguous"
        if raw_duty_ids:
            parsed_duty = [_parse_runtime_device_id(k) for k in raw_duty_ids]
            if all(cid is not None for cid in parsed_duty):
                raw_duty_int_ids = {cid for cid in parsed_duty if cid is not None}
                all_hbm_core_ids = {
                    hid
                    for k in raw_hbm_ids
                    for hid in (_parse_runtime_device_id(k),)
                    if hid is not None
                }
                known_gens = {
                    d.generation
                    for d in devices
                    if d.generation in _GEN_DEVICES_PER_CHIP
                }
                verified_chip_count = _verified_dual_function_chip_count(
                    devices, devices_per_chip
                )
                if known_gens == {"tpu7x"}:
                    # Google Cloud Ironwood (`tpu7x`) VMs have an architectural
                    # ceiling of 4 physical chips per host (8 logical cores), so
                    # >max(4, verified_chip_count) distinct duty IDs prove a
                    # per-core duty reply independently of PCI sibling grouping.
                    if len(raw_duty_int_ids) > max(4, verified_chip_count or 4):
                        duty_layout = "per_core"
                elif known_gens and known_gens <= {"v2", "v3", "v2-v3"} and len(devices) >= 4:
                    # Cloud TPU v2/v3 VMs can host up to 8 physical chips (16
                    # logical cores); a filtered container may see only 4 chips
                    # while libtpu reports all 8 physical chips. Never compare
                    # `len(raw_duty_int_ids)` against a possibly-filtered visible
                    # chip count to infer `per_core` — require exceeding the
                    # architectural 8-chip VM ceiling (or explicit `--grpc-duty-layout=per-core`).
                    legacy_chip_count = len(devices)
                    legacy_core_count = legacy_chip_count * devices_per_chip
                    expanded_raw = {
                        cid * devices_per_chip + off
                        for cid in raw_duty_int_ids
                        for off in range(devices_per_chip)
                    }
                    if len(raw_duty_int_ids) > max(8, legacy_chip_count):
                        duty_layout = "per_core"
                    elif (
                        len(raw_duty_int_ids) == legacy_chip_count
                        and len(all_hbm_core_ids) == legacy_core_count
                        and expanded_raw == all_hbm_core_ids
                    ):
                        duty_layout = "per_chip"
                elif verified_chip_count is not None:
                    if len(raw_duty_int_ids) > max(8, verified_chip_count):
                        duty_layout = "per_core"

    if duty_layout == "per_chip" and devices_per_chip is not None:
        for chip_key, duty_val in duty_by_chip.items():
            chip_idx = _parse_runtime_device_id(chip_key)
            if chip_idx is None:
                continue
            for offset in range(devices_per_chip):
                core_key = str(chip_idx * devices_per_chip + offset)
                row = rows.setdefault(core_key, _TpuInfoRow(chip_key=core_key))
                row.in_util_table = True
                row.duty_cycle_percent = duty_val
    elif duty_layout == "per_core":
        for chip_key, duty_val in duty_by_chip.items():
            row = rows.setdefault(chip_key, _TpuInfoRow(chip_key=chip_key))
            row.in_util_table = True
            row.duty_cycle_percent = duty_val
    elif duty_by_chip:
        # Layout is ambiguous (e.g. sidecar without PCI inventory, incomplete
        # PCI inventory, or partial/missing HBM/duty response): keep
        # `duty_cycle_percent` unset so we never duplicate or misjoin chip IDs
        # onto logical core HBM rows, preserve the raw readings in
        # `unjoined_duty_cycle_percent` for separate export, and allow fallback
        # to run within the remaining scrape budget.
        for raw_key, duty_val in duty_by_chip.items():
            row = rows.setdefault(raw_key, _TpuInfoRow(chip_key=raw_key))
            row.in_util_table = True
            row.unjoined_duty_cycle_percent = duty_val

    # Record explicitly unset/invalid duty-cycle gauges returned by live gRPC so
    # stock `tpu-info` / `tpu_info.metrics` fallbacks (which read unset protobuf
    # doubles as `0.0` without checking field presence) can never turn an unknown
    # duty reading into a false 0% collapse alert.
    unresolved_unset = unset_duty_ids - set(duty_by_chip)
    if unresolved_unset:
        unset_target_keys: set[str] = set()
        if duty_layout == "per_chip" and devices_per_chip is not None:
            for chip_key in unresolved_unset:
                chip_idx = _parse_runtime_device_id(chip_key)
                if chip_idx is None:
                    unset_target_keys.add(chip_key)
                    continue
                for offset in range(devices_per_chip):
                    unset_target_keys.add(str(chip_idx * devices_per_chip + offset))
        elif duty_layout == "per_core":
            unset_target_keys.update(unresolved_unset)
        else:
            for chip_key in unresolved_unset:
                unset_target_keys.add(chip_key)
                if devices_per_chip is not None and devices_per_chip > 1:
                    chip_idx = _parse_runtime_device_id(chip_key)
                    if chip_idx is not None:
                        for offset in range(devices_per_chip):
                            unset_target_keys.add(str(chip_idx * devices_per_chip + offset))
        for tk in unset_target_keys:
            row = rows.setdefault(tk, _TpuInfoRow(chip_key=tk))
            row.in_util_table = True
            row.duty_unset_in_grpc = True

    if not _has_live_utilization(rows):
        if not any_rpc_ok and last_err:
            return False, last_err, {}
        return (
            False,
            last_err or f"gRPC RuntimeMetricService returned no live metrics from {addr}",
            rows if any(r.duty_unset_in_grpc for r in rows.values()) else {},
        )
    return True, None, rows


def read_shm_seqlock_metrics(
    pci_bdf: str,
    shm_root: Optional[Path] = DEFAULT_SHM_ROOT,
    *,
    max_retries: int = 64,
    trusted_uids: Optional[Sequence[int]] = None,
    max_age_s: float = DEFAULT_SHM_MAX_AGE_S,
    now_ns: Optional[int] = None,
) -> Optional[ShmMetrics]:
    """Read bare-metal `libopentpu.so` telemetry from `<shm_root>/freetpu_metrics_<bdf>` via SeqLock `pread`.

    Authenticates the publisher directory, metric file, and sample freshness on
    opened file descriptors:
    - requires `shm_root` (default `/run/freetpu-metrics`, under root-owned `/run`
      and provisioned via systemd `RuntimeDirectory=freetpu-metrics` / `tmpfiles.d`
      to prevent name squatting under world-writable `/dev/shm`) to be a non-symlink
      publisher-controlled directory opened with `O_DIRECTORY | O_NOFOLLOW`,
      owned by an allowed UID (default: current `euid`) with neither group-write
      (`0o020`) nor world-write (`0o002`) permissions (rejecting shared `/dev/shm`
      so unrelated local users cannot pre-create predictable filenames);
    - opens `freetpu_metrics_<bdf>` relative to the verified directory fd with
      `O_NOFOLLOW | O_NONBLOCK` to avoid symlink/FIFO/rename TOCTOU races;
    - requires a regular file owned by an allowed UID with neither group-write
      (`0o020`) nor world-write (`0o002`) permissions;
    - reads via `os.pread` (never `mmap`) so concurrent truncation cannot raise
      `SIGBUS` and instead fails closed with `None`;
    - requires a fresh publisher heartbeat `timestamp_ns` within `max_age_s`
      (default 30s) so exited/crashed publishers expire cleanly instead of
      pinning stale HBM/OOM readings indefinitely.
    """
    if shm_root is None or not pci_bdf:
        return None
    if "/" in pci_bdf or "\\" in pci_bdf or ".." in pci_bdf:
        return None

    allowed_uids = (
        {int(u) for u in trusted_uids}
        if trusted_uids is not None
        else {os.geteuid()}
    )

    shm_dir = Path(shm_root)
    dir_open_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        dir_fd = os.open(shm_dir, dir_open_flags)
    except OSError:
        return None

    open_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        dir_st = os.fstat(dir_fd)
        if not stat.S_ISDIR(dir_st.st_mode):
            return None
        if dir_st.st_uid not in allowed_uids:
            return None
        if (dir_st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) != 0:
            return None
        try:
            fd = os.open(f"freetpu_metrics_{pci_bdf}", open_flags, dir_fd=dir_fd)
        except OSError:
            return None
    finally:
        os.close(dir_fd)
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

        effective_now_ns = time.time_ns() if now_ns is None else int(now_ns)
        max_age_ns = int(max(0.0, float(max_age_s)) * 1_000_000_000)
        max_future_skew_ns = max(5_000_000_000, max_age_ns)

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
                hbm_alloc, hbm_used, mxu_cycles, dma_bytes, ts_ns = _SHM_PAYLOAD_STRUCT.unpack_from(
                    payload, 0
                )
                if ts_ns <= 0:
                    return None
                if effective_now_ns - ts_ns > max_age_ns:
                    return None
                if ts_ns - effective_now_ns > max_future_skew_ns:
                    return None
                return ShmMetrics(
                    hbm_allocated_bytes=hbm_alloc,
                    hbm_used_bytes=hbm_used,
                    mxu_active_cycles=mxu_cycles,
                    dma_bytes_transferred=dma_bytes,
                    seq=seq1,
                    timestamp_ns=ts_ns,
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


_GCP_METADATA_DIRECT_HOSTS = frozenset({"metadata.google.internal", "169.254.169.254"})


def query_gcp_preemption_status(
    url: str = DEFAULT_GCP_METADATA_PREEMPTED_URL,
    timeout_s: float = 0.5,
) -> Optional[int]:
    """Query GCP instance preemption metadata (`1` if preempted, `0` if healthy, `None` if unavailable).

    Bypasses process HTTP proxies for Compute Engine's built-in metadata endpoints
    (`metadata.google.internal` and `169.254.169.254`) so VM proxy settings do not
    silently break Spot preemption polling when `NO_PROXY` omits the metadata host.
    """
    try:
        req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        effective_timeout = max(0.05, float(timeout_s))
        if host in _GCP_METADATA_DIRECT_HOSTS:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            resp_ctx = opener.open(req, timeout=effective_timeout)
        else:
            resp_ctx = urllib.request.urlopen(req, timeout=effective_timeout)
        with resp_ctx as resp:
            body = resp.read(256).decode("utf-8", errors="replace").strip().lower()
    except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError):
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

        aer_correctable, aer_fatal = read_pcie_aer_counters(entry)

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

        # Group dual-core legacy TPU v2/v3 (0x0027) PCI functions sharing the same
        # base PCI address (`DDDD:BB:DD`) into one physical chip record while
        # retaining per-function PCIe AER counters keyed by each sibling's own BDF.
        # Verify sibling IOMMU group membership exclusivity before attributing
        # sibling VFIO group PIDs or keeping `vfio_group_exclusive=True`.
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
                if aer_correctable is not None or aer_fatal is not None:
                    existing.sibling_aer_counters[entry.name] = (aer_correctable, aer_fatal)
                if iommu_group and iommu_group != existing.iommu_group:
                    existing.vfio_group_exclusive = (
                        existing.vfio_group_exclusive and vfio_group_exclusive
                    )
                    if vfio_group_exclusive and iommu_group not in existing.sibling_iommu_groups:
                        existing.sibling_iommu_groups.append(iommu_group)
                else:
                    same_group_exclusive = (
                        driver == "vfio-pci"
                        and bool(iommu_group)
                        and group_dir_readable.get(iommu_group, False)
                        and full_members <= {existing.pci_bdf, entry.name}
                    )
                    existing.vfio_group_exclusive = (
                        existing.vfio_group_exclusive and same_group_exclusive
                    )
                continue

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


def _row_has_live_utilization(row: _TpuInfoRow) -> bool:
    return row.in_util_table and (
        row.hbm_used_bytes is not None
        or row.hbm_total_bytes is not None
        or row.duty_cycle_percent is not None
        or row.unjoined_duty_cycle_percent is not None
        or row.hbm_bw_util_percent is not None
    )


def _has_live_utilization(rows: Dict[str, _TpuInfoRow]) -> bool:
    return any(_row_has_live_utilization(r) for r in rows.values())


def _row_has_complete_utilization(row: _TpuInfoRow) -> bool:
    return (
        row.in_util_table
        and row.hbm_used_bytes is not None
        and row.hbm_total_bytes is not None
        and (row.duty_cycle_percent is not None or row.duty_unset_in_grpc)
    )


def _resolve_raw_device_runtime_ids(
    devices: Sequence[TpuTelemetry],
    shared_vfio: set[str],
    chip_id_map: Optional[Dict[str, str]],
) -> Tuple[Dict[str, Optional[str]], set[str]]:
    """Resolve per-BDF runtime IDs and post-override ambiguous IDs.

    Checks raw sysfs `chip_id` sharing BEFORE applying `--chip-id-map` overrides:
    if >1 PCI device shares a raw sysfs `chip_id` (e.g. Ironwood sibling PCI
    functions sharing a physical chip ID), neither that shared raw ID nor any
    unmapped function of that multi-core generation may inherit a physical sysfs
    `chip_id` as a logical runtime core ID when a partial `--chip-id-map` maps
    only its sibling.
    """
    raw_sysfs_counts: Dict[str, int] = {}
    raw_sysfs_gen_counts: Dict[Tuple[str, str], int] = {}
    tpu7x_slot_rids: Dict[str, List[str]] = {}
    for dev in devices:
        if dev.runtime_chip_id:
            raw_sysfs_counts[dev.runtime_chip_id] = (
                raw_sysfs_counts.get(dev.runtime_chip_id, 0) + 1
            )
            raw_sysfs_gen_counts[(dev.generation, dev.runtime_chip_id)] = (
                raw_sysfs_gen_counts.get((dev.generation, dev.runtime_chip_id), 0) + 1
            )
            if dev.generation == "tpu7x":
                base_slot = dev.pci_bdf.rsplit(".", 1)[0]
                tpu7x_slot_rids.setdefault(base_slot, []).append(dev.runtime_chip_id)
    shared_raw_sysfs_ids = {rid for rid, cnt in raw_sysfs_counts.items() if cnt > 1}
    physical_sysfs_gens = {
        gen
        for (gen, _rid), cnt in raw_sysfs_gen_counts.items()
        if cnt > 1 and _GEN_DEVICES_PER_CHIP.get(gen, 1) > 1
    }
    # On Ironwood (`tpu7x`, 2 logical cores/PCI functions per physical chip), a
    # filtered PCI view exposing only one function per chip makes physical sysfs
    # `chip_id`s appear unique even though they are NOT logical core IDs.
    # Require positive host-level evidence that `tpu7x` sysfs `chip_id` is
    # per-core (at least one multifunction sibling pair on the same PCI slot
    # reporting distinct non-empty `chip_id`s, with zero shared `tpu7x` raw
    # `chip_id`s) before trusting raw sysfs `chip_id` without `--chip-id-map`.
    tpu7x_sysfs_proved_per_core = (
        "tpu7x" not in physical_sysfs_gens
        and any(
            len(rids) == _GEN_DEVICES_PER_CHIP["tpu7x"] and len(set(rids)) == len(rids)
            for rids in tpu7x_slot_rids.values()
        )
    )

    resolved_by_bdf: Dict[str, Optional[str]] = {}
    claimants: Dict[str, List[str]] = {}
    for dev in devices:
        is_shared_vfio = bool(dev.vfio_device and dev.vfio_device in shared_vfio)
        rt_id: Optional[str] = None
        if chip_id_map:
            mapped = chip_id_map.get(dev.pci_bdf)
            if mapped is None and not is_shared_vfio and dev.vfio_device:
                mapped = chip_id_map.get(dev.vfio_device)
            if mapped is not None:
                rt_id = str(mapped).strip() or None
        if not rt_id and dev.runtime_chip_id:
            if (
                dev.runtime_chip_id not in shared_raw_sysfs_ids
                and dev.generation not in physical_sysfs_gens
                and (dev.generation != "tpu7x" or tpu7x_sysfs_proved_per_core)
            ):
                rt_id = dev.runtime_chip_id
        resolved_by_bdf[dev.pci_bdf] = rt_id
        if rt_id:
            claimants.setdefault(rt_id, []).append(dev.pci_bdf)
    ambiguous = {rt for rt, bdfs in claimants.items() if len(bdfs) > 1}
    return resolved_by_bdf, ambiguous


def _verified_row_alias_map(
    devices: Sequence[TpuTelemetry],
    shared_vfio: set[str],
    chip_id_map: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Map every verified non-ambiguous alias for a TPU (`pci_bdf`, unique
    `/dev/vfio/<group>`, verified `runtime_id`) to one canonical row key so
    multi-backend fallback merges combine fields onto the same device row."""
    resolved_by_bdf, ambiguous = _resolve_raw_device_runtime_ids(
        devices, shared_vfio, chip_id_map
    )
    aliases: Dict[str, str] = {}
    for dev in devices:
        rt_id = resolved_by_bdf.get(dev.pci_bdf)
        if rt_id in ambiguous:
            rt_id = None
        canonical = rt_id if rt_id else dev.pci_bdf
        aliases[dev.pci_bdf] = canonical
        if dev.vfio_device and dev.vfio_device not in shared_vfio:
            aliases[dev.vfio_device] = canonical
        if rt_id:
            aliases[rt_id] = canonical
    return aliases


def _expected_runtime_ids_for_devices(
    devices: Sequence[TpuTelemetry],
    shared_vfio: set[str],
    chip_id_map: Optional[Dict[str, str]],
) -> set[str]:
    """Return the set of non-ambiguous expected runtime IDs established by
    verified sysfs `chip_id` and/or explicit `--chip-id-map`."""
    resolved_by_bdf, ambiguous = _resolve_raw_device_runtime_ids(
        devices, shared_vfio, chip_id_map
    )
    return {
        rt
        for rt in resolved_by_bdf.values()
        if rt is not None and rt not in ambiguous
    }


def _has_complete_utilization(
    rows: Dict[str, _TpuInfoRow],
    expected_runtime_ids: Optional[set[str]] = None,
) -> bool:
    """Return True when every observed runtime device row (and every expected
    runtime ID established by PCI inventory / `--chip-id-map`) has both HBM
    usage/total and duty-cycle."""
    util_rows = {k: r for k, r in rows.items() if r.in_util_table}
    if not util_rows:
        return False
    if not all(_row_has_complete_utilization(r) for r in util_rows.values()):
        return False
    if expected_runtime_ids:
        complete_ids = {
            k for k, r in util_rows.items() if _row_has_complete_utilization(r)
        }
        if not expected_runtime_ids.issubset(complete_ids):
            return False
    return True


def _sanitize_fallback_duty_rows(
    rows: Dict[str, _TpuInfoRow],
    devices: Sequence[TpuTelemetry],
    *,
    unset_duty_ids: Optional[set[str]] = None,
    grpc_duty_layout: str = "auto",
) -> Dict[str, _TpuInfoRow]:
    """Strip unverified fallback `duty_cycle_percent` joins on Ironwood (`tpu7x`) hosts
    and block stock fallbacks from turning gRPC-unset duty gauges into `0.0%`.

    Upstream `tpu_info.metrics.get_chip_usage_new` (used strictly on Ironwood
    `V7X` by stock `tpu-info` CLI and `auto`/`per-core` Python fallback) sorts
    HBM and duty-cycle replies separately and zips them by position without
    checking that IDs match, and stock `tpu-info` / `tpu_info.metrics` read
    unset protobuf duty gauges as `0.0` without checking field presence.
    Explicit `--grpc-duty-layout=per-chip` in the Python fallback calls
    `get_chip_usage()` instead (which expands chip readings into per-core rows
    cleanly even on a filtered PCI sidecar), and legacy `v2`/`v3`/`v4`/`v5e`/
    `v5p`/`v6e` always use `get_chip_usage()`.
    On Ironwood (`tpu7x`) `get_chip_usage_new` paths, accept fallback
    `duty_cycle_percent` ONLY when fallback covers the full contiguous verified
    host core set (`0..verified_chips * devices_per_chip - 1`) with populated
    gauges, where equal-length positional zipping is provably 1:1.
    """
    if not rows:
        return rows
    working = rows
    if unset_duty_ids:
        stripped: Dict[str, _TpuInfoRow] = {}
        for k, r in rows.items():
            copied = _copy_tpu_info_row(r)
            if k in unset_duty_ids and copied.duty_cycle_percent is not None:
                copied.duty_cycle_percent = None
            stripped[k] = copied
        working = stripped
    # Restrict the Ironwood positional-zip guard strictly to hosts with
    # Ironwood (`tpu7x`) devices; sidecars without PCI inventory and legacy
    # `v2`/`v3`/`v4`/`v5e`/`v5p`/`v6e` hosts preserve fallback duty readings.
    if not any(d.generation == "tpu7x" for d in devices):
        return working
    inferred_dpc = _infer_devices_per_chip(devices)
    if inferred_dpc is None or inferred_dpc <= 1:
        return working

    def _strip_unverified_ironwood_zeros(src_rows: Dict[str, _TpuInfoRow]) -> Dict[str, _TpuInfoRow]:
        """Clear `0.0` duty readings whose protobuf gauge presence was not verified
        so stock `tpu-info` / `get_chip_usage_new()` / `get_chip_usage()` reading
        unset protobuf defaults as `0.0` can never trigger `TpuMxuDutyCycleCollapse`."""
        cleaned: Dict[str, _TpuInfoRow] = {}
        for k, r in src_rows.items():
            copied = _copy_tpu_info_row(r)
            if (
                copied.duty_cycle_percent == 0.0
                and not copied.duty_ids_verified
            ):
                copied.duty_cycle_percent = None
            cleaned[k] = copied
        return cleaned

    # In explicit `per-chip` Python fallback (`get_chip_usage()`), upstream
    # sorts HBM and duty replies separately and zips by position after checking
    # only `2 * len(duty) == len(hbm)`. Accept partial subsets ONLY when raw
    # duty chip IDs and gauge presence were independently verified 1:1 against
    # HBM core IDs; otherwise require full contiguous 8-core host coverage
    # (`0..7`) and strip any unverified `0.0` fallback duty values.
    if grpc_duty_layout == "per-chip":
        duty_rows = [
            r for r in working.values() if r.in_util_table and r.duty_cycle_percent is not None
        ]
        if duty_rows and all(r.duty_ids_verified for r in duty_rows):
            return working
        raw_duty_ids = {
            pid
            for k, r in rows.items()
            if r.in_util_table and r.duty_cycle_percent is not None
            for pid in (_parse_runtime_device_id(k),)
            if pid is not None
        }
        if raw_duty_ids == set(range(8)):
            return _strip_unverified_ironwood_zeros(working)
        sanitized_pc: Dict[str, _TpuInfoRow] = {}
        for k, r in working.items():
            copied = _copy_tpu_info_row(r)
            if copied.duty_cycle_percent is not None:
                copied.duty_cycle_percent = None
            sanitized_pc[k] = copied
        return sanitized_pc

    duty_rows_auto = [
        r for r in working.values() if r.in_util_table and r.duty_cycle_percent is not None
    ]
    if duty_rows_auto and all(r.duty_ids_verified for r in duty_rows_auto):
        return working

    expected_core_count: Optional[int] = None
    verified_chips = _verified_dual_function_chip_count(devices, inferred_dpc)
    if verified_chips is not None:
        expected_core_count = verified_chips * inferred_dpc
    elif all(d.generation == "tpu7x" for d in devices) and len(devices) == 8:
        expected_core_count = 8

    if expected_core_count is not None:
        duty_ids = {
            pid
            for k, r in working.items()
            if r.in_util_table and r.duty_cycle_percent is not None
            for pid in (_parse_runtime_device_id(k),)
            if pid is not None
        }
        if duty_ids == set(range(expected_core_count)):
            return _strip_unverified_ironwood_zeros(working)

    sanitized: Dict[str, _TpuInfoRow] = {}
    for k, r in working.items():
        copied = _copy_tpu_info_row(r)
        if copied.duty_cycle_percent is not None:
            copied.duty_cycle_percent = None
        sanitized[k] = copied
    return sanitized


def _tag_row_sources(rows: Dict[str, _TpuInfoRow], source_label: str) -> Dict[str, _TpuInfoRow]:
    """Attach `source_label` provenance to rows returned by a telemetry backend."""
    tagged: Dict[str, _TpuInfoRow] = {}
    for k, r in rows.items():
        copied = _copy_tpu_info_row(r)
        if not copied.sources and (
            copied.in_chips_table or _row_has_live_utilization(copied)
        ):
            copied.sources = [source_label]
        tagged[k] = copied
    return tagged


def _format_row_source(row: _TpuInfoRow, default: str = "tpu-info") -> str:
    return "+".join(row.sources) if row.sources else default


def _copy_tpu_info_row(row: _TpuInfoRow) -> _TpuInfoRow:
    return _TpuInfoRow(
        chip_key=row.chip_key,
        chip_type=row.chip_type,
        pids=list(row.pids),
        hbm_used_bytes=row.hbm_used_bytes,
        hbm_total_bytes=row.hbm_total_bytes,
        duty_cycle_percent=row.duty_cycle_percent,
        unjoined_duty_cycle_percent=row.unjoined_duty_cycle_percent,
        hbm_bw_util_percent=row.hbm_bw_util_percent,
        in_chips_table=row.in_chips_table,
        in_util_table=row.in_util_table,
        duty_unset_in_grpc=row.duty_unset_in_grpc,
        duty_ids_verified=row.duty_ids_verified,
        sources=list(row.sources),
    )


def _merge_runtime_rows(
    base_rows: Dict[str, _TpuInfoRow],
    extra_rows: Dict[str, _TpuInfoRow],
    *,
    prefer_base: bool = False,
    alias_to_canonical: Optional[Dict[str, str]] = None,
) -> Dict[str, _TpuInfoRow]:
    def _merge_one(
        target: Dict[str, _TpuInfoRow],
        raw_key: str,
        r: _TpuInfoRow,
        *,
        keep_existing: bool,
    ) -> None:
        k = alias_to_canonical.get(raw_key, raw_key) if alias_to_canonical else raw_key
        existing = target.get(k)
        if existing is not None:
            used_extra = False
            if r.hbm_used_bytes is not None and (not keep_existing or existing.hbm_used_bytes is None):
                existing.hbm_used_bytes = r.hbm_used_bytes
                used_extra = True
            if r.hbm_total_bytes is not None and (not keep_existing or existing.hbm_total_bytes is None):
                existing.hbm_total_bytes = r.hbm_total_bytes
                used_extra = True
            if r.duty_cycle_percent is not None and (
                not keep_existing
                or (existing.duty_cycle_percent is None and not existing.duty_unset_in_grpc)
            ):
                existing.duty_cycle_percent = r.duty_cycle_percent
                used_extra = True
            existing.duty_unset_in_grpc = existing.duty_unset_in_grpc or r.duty_unset_in_grpc
            if r.unjoined_duty_cycle_percent is not None and (
                not keep_existing or existing.unjoined_duty_cycle_percent is None
            ):
                existing.unjoined_duty_cycle_percent = r.unjoined_duty_cycle_percent
                used_extra = True
            if r.hbm_bw_util_percent is not None and (
                not keep_existing or existing.hbm_bw_util_percent is None
            ):
                existing.hbm_bw_util_percent = r.hbm_bw_util_percent
                used_extra = True
            if r.chip_type and not existing.chip_type:
                existing.chip_type = r.chip_type
            for pid in r.pids:
                if pid not in existing.pids:
                    existing.pids.append(pid)
            if r.in_chips_table and not existing.in_chips_table:
                used_extra = True
            existing.in_chips_table = existing.in_chips_table or r.in_chips_table
            existing.in_util_table = existing.in_util_table or r.in_util_table
            if used_extra:
                for src in r.sources:
                    if src and src not in existing.sources:
                        existing.sources.append(src)
        else:
            copied = _copy_tpu_info_row(r)
            copied.chip_key = k
            target[k] = copied

    merged: Dict[str, _TpuInfoRow] = {}
    for k, r in base_rows.items():
        _merge_one(merged, k, r, keep_existing=True)
    for k, r in extra_rows.items():
        _merge_one(merged, k, r, keep_existing=prefer_base)
    return merged


def _resolve_device_runtime_mapping(
    devices: Sequence[TpuTelemetry],
    parsed_rows: Dict[str, _TpuInfoRow],
    shared_vfio: set[str],
    chip_id_map: Optional[Dict[str, str]],
) -> Tuple[
    Dict[str, Optional[str]],
    set[str],
    Dict[str, Optional[str]],
    set[str],
]:
    """Resolve authoritative runtime chip IDs and utilization row keys for discovered PCI devices.

    Only authoritative sources are accepted for runtime-ordinal-to-PCI mapping:
    1. Verified sysfs `chip_id` (`dev.runtime_chip_id`)
    2. Explicit `--chip-id-map` (`pci_bdf` or non-shared `/dev/vfio/<group>`)

    `tpu-info` CLI Process Monitor guesses `/dev/vfio/<runtime_id>` when runtime IDs
    differ from IOMMU groups, so it is NEVER treated as authoritative PCI identity.
    Any runtime ID or utilization row claimed by >1 physical PCI device is flagged
    ambiguous so it remains keyed strictly by runtime ID (`pci_bdf="unmapped"`).
    """
    util_rows = {k: r for k, r in parsed_rows.items() if r.in_util_table}
    resolved_rt_ids, ambiguous_rt_ids = _resolve_raw_device_runtime_ids(
        devices, shared_vfio, chip_id_map
    )
    prospective_util_key: Dict[str, Optional[str]] = {}
    util_key_claimants: Dict[str, List[str]] = {}

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
    return resolved_rt_ids, ambiguous_rt_ids, prospective_util_key, ambiguous_util_keys


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
    grpc_duty_layout: str = "auto",
    shm_root: Optional[Path] = DEFAULT_SHM_ROOT,
    shm_trusted_uids: Optional[Sequence[int]] = None,
    shm_max_age_s: float = DEFAULT_SHM_MAX_AGE_S,
    gcp_metadata_preemption: bool = False,
    gcp_metadata_url: str = DEFAULT_GCP_METADATA_PREEMPTED_URL,
    gcp_metadata_timeout_s: float = 0.5,
) -> ExporterSnapshot:
    """Collect PCI/VFIO discovery, SeqLock SHM, AER counters, optional procfs PIDs, and gRPC/`tpu-info` metrics."""
    if grpc_duty_layout not in VALID_GRPC_DUTY_LAYOUTS:
        raise ValueError(
            f"Invalid --grpc-duty-layout {grpc_duty_layout!r}; "
            f"expected one of {', '.join(VALID_GRPC_DUTY_LAYOUTS)}"
        )
    t0 = time.monotonic()
    devices = discover_tpu_devices(sysfs_root=sysfs_root, dev_root=dev_root)
    shared_vfio = _shared_vfio_paths(devices)
    validate_chip_id_map(devices, chip_id_map)

    # Scan /proc/<pid>/fd for open handles to <dev_root>/vfio/<group> (including
    # any separate-group sibling PCI functions collapsed onto legacy v2/v3 chips)
    # unless decoupled (`skip_pid_scan=True`, used by unprivileged workload sidecars).
    real_vfio_paths: List[str] = []
    expected_vfio_devnums: Dict[str, Optional[str]] = {}
    for d in devices:
        dev_groups = ([d.iommu_group] if d.iommu_group else []) + [
            sg for sg in d.sibling_iommu_groups if sg and sg != d.iommu_group
        ]
        for grp in dev_groups:
            vp = str(dev_root / "vfio" / grp)
            if vp not in expected_vfio_devnums:
                real_vfio_paths.append(vp)
                expected_vfio_devnums[vp] = _read_kernel_vfio_devnum(sysfs_root, grp)
    if skip_pid_scan:
        proc_pids: Dict[str, List[int]] = {}
        proc_fd_accessible = False
        vfio_rdevs: Dict[str, int] = {}
    else:
        proc_pids, proc_fd_accessible, _unreadable, vfio_rdevs = _scan_proc_vfio_pids(
            proc_root, real_vfio_paths, expected_devnums=expected_vfio_devnums
        )

    expected_runtime_ids = _expected_runtime_ids_for_devices(
        devices, shared_vfio, chip_id_map
    )
    verified_row_aliases = _verified_row_alias_map(devices, shared_vfio, chip_id_map)
    runtime_budget_s = max(0.05, float(tpu_info_timeout_s))
    runtime_deadline = time.monotonic() + runtime_budget_s
    explicit_non_default_runtime_addr = bool(
        runtime_metrics_addr and runtime_metrics_addr != DEFAULT_RUNTIME_METRICS_ADDR
    )
    if not tpu_info_bin:
        # Empty/None `--tpu-info-bin` explicitly disables `libtpu` runtime
        # telemetry (direct gRPC, `tpu_info.metrics`, and CLI) across every mode.
        tpu_info_ok, tpu_info_err, parsed_rows = False, "tpu-info disabled", {}
    elif skip_pid_scan or explicit_non_default_runtime_addr:
        # Sidecar / no-PID mode OR explicit non-default `--runtime-metrics-addr`:
        # try direct in-process gRPC first (capped at 65% of the shared runtime
        # budget so 35% remains reserved for the Python `tpu_info.metrics`
        # fallback if gRPC RPCs fail, stall, or return only partial families
        # such as HBM without duty-cycle). Never spawn the `tpu-info` CLI subprocess.
        grpc_budget_s = runtime_budget_s * 0.65
        tpu_info_ok, tpu_info_err, parsed_rows = _query_grpc_runtime_metrics(
            devices,
            addr=runtime_metrics_addr,
            timeout_s=grpc_budget_s,
            grpc_duty_layout=grpc_duty_layout,
        )
        if parsed_rows:
            parsed_rows = _merge_runtime_rows(
                {},
                _tag_row_sources(parsed_rows, "libtpu-grpc"),
                prefer_base=True,
                alias_to_canonical=verified_row_aliases,
            )
        if not tpu_info_ok or not _has_complete_utilization(
            parsed_rows, expected_runtime_ids=expected_runtime_ids
        ):
            remaining_s = runtime_deadline - time.monotonic()
            if remaining_s >= 0.01:
                py_ok, py_err, py_rows = _query_tpu_info_python_metrics(
                    devices,
                    addr=runtime_metrics_addr,
                    timeout_s=remaining_s,
                    grpc_duty_layout=grpc_duty_layout,
                )
                if py_ok:
                    grpc_unset_duty_ids = {
                        k for k, r in parsed_rows.items() if r.duty_unset_in_grpc
                    }
                    py_rows = _sanitize_fallback_duty_rows(
                        py_rows,
                        devices,
                        unset_duty_ids=grpc_unset_duty_ids,
                        grpc_duty_layout=grpc_duty_layout,
                    )
                    py_rows = _tag_row_sources(py_rows, "tpu-info-python")
                    parsed_rows = _merge_runtime_rows(
                        parsed_rows,
                        py_rows,
                        prefer_base=True,
                        alias_to_canonical=verified_row_aliases,
                    )
                    tpu_info_ok, tpu_info_err = True, None
                elif not tpu_info_ok:
                    tpu_info_err = py_err
    else:
        # Prefer direct in-process `RuntimeMetricService` gRPC over spawning
        # `tpu-info` subprocesses ONLY for the exact default `"tpu-info"` value;
        # any custom executable name or path (e.g. `my-wrapper` on PATH or
        # `/path/to/wrapper`) invokes that configured CLI directly.
        # Note: `tpu-info` CLI Process Monitor guesses `/dev/vfio/<runtime_id>`
        # and cannot establish authoritative PCI identity, so we do NOT spawn the
        # CLI solely to guess mappings when direct gRPC already returned complete
        # HBM + duty-cycle readings. If direct gRPC returned only partial
        # utilization (e.g. HBM succeeded while duty-cycle timed out, or entire
        # expected devices were omitted), run fallback within the remaining
        # scrape budget and merge only the missing fields (`prefer_base=True`) so
        # successful gRPC readings are preserved while per-row source provenance
        # records which chips used fallback.
        use_default_grpc_first = tpu_info_bin == "tpu-info"
        tpu_info_ok = False
        tpu_info_err = None
        parsed_rows = {}
        attempted_grpc = False
        if use_default_grpc_first and runtime_metrics_addr:
            attempted_grpc = True
            # Reserve 35% of the shared runtime scrape budget for the fallback
            # path when fallback is enabled so a stalled gRPC RPC cannot starve
            # fallback telemetry.
            grpc_budget_s = (
                runtime_budget_s * 0.65 if tpu_info_bin else runtime_budget_s
            )
            tpu_info_ok, tpu_info_err, parsed_rows = _query_grpc_runtime_metrics(
                devices,
                addr=runtime_metrics_addr,
                timeout_s=grpc_budget_s,
                grpc_duty_layout=grpc_duty_layout,
            )
            if parsed_rows:
                parsed_rows = _merge_runtime_rows(
                    {},
                    _tag_row_sources(parsed_rows, "libtpu-grpc"),
                    prefer_base=True,
                    alias_to_canonical=verified_row_aliases,
                )

        remaining_s = runtime_deadline - time.monotonic()
        # When the configured CLI is the primary/first backend (e.g. a custom
        # executable or disabled direct gRPC), run whenever positive time
        # remains — even at the minimum 50 ms scrape budget after setup overhead.
        cli_has_budget = (remaining_s > 0.0) if not attempted_grpc else (remaining_s >= 0.01)
        if (
            not tpu_info_ok
            or not _has_complete_utilization(
                parsed_rows, expected_runtime_ids=expected_runtime_ids
            )
        ) and cli_has_budget:
            cli_ok, cli_err, cli_rows = _run_tpu_info(
                tpu_info_bin=tpu_info_bin,
                timeout_s=remaining_s,
            )
            if cli_ok:
                # Only apply the upstream `tpu-info` V7X positional-zip duty guard
                # to stock `tpu-info` executables; preserve duty-cycle readings
                # produced by custom operator wrappers passed via `--tpu-info-bin`.
                if Path(tpu_info_bin).name in ("tpu-info", "tpu_info"):
                    grpc_unset_duty_ids = {
                        k for k, r in parsed_rows.items() if r.duty_unset_in_grpc
                    }
                    cli_rows = _sanitize_fallback_duty_rows(
                        cli_rows, devices, unset_duty_ids=grpc_unset_duty_ids
                    )
                cli_rows = _tag_row_sources(cli_rows, "tpu-info")
                tpu_info_ok = True
                tpu_info_err = None
                parsed_rows = _merge_runtime_rows(
                    parsed_rows,
                    cli_rows,
                    prefer_base=True,
                    alias_to_canonical=verified_row_aliases,
                )
            elif not tpu_info_ok:
                tpu_info_err = cli_err

            remaining_s = runtime_deadline - time.monotonic()
            if (
                (
                    not tpu_info_ok
                    or not _has_complete_utilization(
                        parsed_rows, expected_runtime_ids=expected_runtime_ids
                    )
                )
                and runtime_metrics_addr
                and remaining_s >= 0.01
            ):
                py_ok, _py_err, py_rows = _query_tpu_info_python_metrics(
                    devices,
                    addr=runtime_metrics_addr,
                    timeout_s=remaining_s,
                    grpc_duty_layout=grpc_duty_layout,
                )
                if py_ok:
                    grpc_unset_duty_ids = {
                        k for k, r in parsed_rows.items() if r.duty_unset_in_grpc
                    }
                    py_rows = _sanitize_fallback_duty_rows(
                        py_rows,
                        devices,
                        unset_duty_ids=grpc_unset_duty_ids,
                        grpc_duty_layout=grpc_duty_layout,
                    )
                    py_rows = _tag_row_sources(py_rows, "tpu-info-python")
                    parsed_rows = _merge_runtime_rows(
                        parsed_rows,
                        py_rows,
                        prefer_base=True,
                        alias_to_canonical=verified_row_aliases,
                    )
                    tpu_info_ok, tpu_info_err = True, None
    util_rows = {k: r for k, r in parsed_rows.items() if r.in_util_table}
    mapped_util_keys: Dict[str, str] = {}
    overall_pid_available = True

    (
        resolved_rt_ids,
        ambiguous_rt_ids,
        prospective_util_key,
        ambiguous_util_keys,
    ) = _resolve_device_runtime_mapping(devices, parsed_rows, shared_vfio, chip_id_map)

    for dev in devices:
        dev_groups = ([dev.iommu_group] if dev.iommu_group else []) + [
            sg for sg in dev.sibling_iommu_groups if sg and sg != dev.iommu_group
        ]
        dev_vfio_paths = [str(dev_root / "vfio" / grp) for grp in dev_groups]
        pids_set: set[int] = set()
        all_rdevs_verified = bool(dev_vfio_paths)
        for vp in dev_vfio_paths:
            pids_set.update(proc_pids.get(vp, []))
            if vfio_rdevs.get(vp) is None:
                all_rdevs_verified = False
        pids = sorted(pids_set)

        rt_id = resolved_rt_ids.get(dev.pci_bdf)
        if rt_id is None or rt_id in ambiguous_rt_ids:
            dev.runtime_chip_id = None
            verified_rt_id = None
        else:
            dev.runtime_chip_id = rt_id
            verified_rt_id = rt_id

        # Attach `TPU Chips` rows strictly by verified /dev/vfio/<group>, PCI BDF,
        # or canonicalized verified runtime ID for this device.
        chips_row: Optional[_TpuInfoRow] = None
        for key in (dev.vfio_device, dev.pci_bdf, verified_rt_id):
            if key and key in parsed_rows and parsed_rows[key].in_chips_table:
                chips_row = parsed_rows[key]
                break

        if chips_row is not None and not skip_pid_scan:
            dev.telemetry_source = _format_row_source(chips_row)

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
                dev.telemetry_source = _format_row_source(matched_util)
            if matched_util.hbm_used_bytes is not None:
                dev.hbm_used_bytes = matched_util.hbm_used_bytes
            if matched_util.hbm_total_bytes is not None:
                dev.hbm_total_bytes = matched_util.hbm_total_bytes
            if matched_util.duty_cycle_percent is not None:
                dev.duty_cycle_percent = matched_util.duty_cycle_percent
            if matched_util.hbm_bw_util_percent is not None:
                dev.hbm_bandwidth_utilization_percent = matched_util.hbm_bw_util_percent

        # Bare-metal libopentpu.so SeqLock /run/freetpu-metrics/freetpu_metrics_<bdf> telemetry.
        shm = read_shm_seqlock_metrics(
            dev.pci_bdf,
            shm_root=shm_root,
            trusted_uids=shm_trusted_uids,
            max_age_s=shm_max_age_s,
        )
        if shm is not None:
            dev.shm_hbm_allocated_bytes = shm.hbm_allocated_bytes
            dev.mxu_active_cycles = shm.mxu_active_cycles
            dev.dma_bytes_transferred = shm.dma_bytes_transferred
            if dev.hbm_used_bytes is None:
                dev.hbm_used_bytes = shm.hbm_used_bytes
            # Preserve known device HBM capacity (from runtime telemetry or PCI
            # nominal capacity); expose the libopentpu pool size through the
            # dedicated `freetpu_tpu_shm_hbm_allocated_bytes` metric and use it
            # as `hbm_total_bytes` ONLY when device capacity is otherwise absent.
            if dev.hbm_total_bytes is None and shm.hbm_allocated_bytes > 0:
                dev.hbm_total_bytes = shm.hbm_allocated_bytes
            if dev.telemetry_source == "sysfs-fallback":
                dev.telemetry_source = "libopentpu-shm"

        # Upstream `tpu-info` also scans `/proc/*/fd/*` under the caller's privileges
        # and reports `PID=None` when another user's process is unreadable. Only mark
        # PID telemetry available when BOTH the `/proc/<pid>/fd` scan succeeded without
        # permission denials AND this device's `/dev/vfio/<group>` character device
        # number (`st_rdev`) was verified, so a missing/unreadable/non-char VFIO node
        # never emits a false zero PID count as valid billing coverage.
        dev.pid_telemetry_available = proc_fd_accessible and all_rdevs_verified
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
            telemetry_source=_format_row_source(row) if row.sources else None,
        )
        for key, row in util_rows.items()
        if (
            row.hbm_used_bytes is not None
            or row.hbm_total_bytes is not None
            or row.duty_cycle_percent is not None
            or row.hbm_bw_util_percent is not None
        )
    ]

    unjoined_duty_candidates: Dict[str, float] = {
        key: row.unjoined_duty_cycle_percent
        for key, row in util_rows.items()
        if row.unjoined_duty_cycle_percent is not None
    }
    joined_duty_keys: set[str] = {
        key for key, row in util_rows.items() if row.duty_cycle_percent is not None
    }
    inferred_dpc = _infer_devices_per_chip(devices)

    def _is_unjoined_duty_covered(raw_key: str) -> bool:
        """Return True ONLY when fallback joined rows provably cover `raw_key`
        under BOTH per-core (`raw_id`) and per-chip (`raw_id * dpc + offset`)
        interpretations using a verified PCI `devices_per_chip` multiplier.
        """
        if inferred_dpc is None or inferred_dpc < 1:
            return False
        raw_id = _parse_runtime_device_id(raw_key)
        if raw_id is None:
            return False
        if str(raw_id) not in joined_duty_keys:
            return False
        return all(
            str(raw_id * inferred_dpc + off) in joined_duty_keys
            for off in range(inferred_dpc)
        )

    runtime_unjoined_duty: Dict[str, float] = {
        key: val
        for key, val in unjoined_duty_candidates.items()
        if not _is_unjoined_duty_covered(key)
    }

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
        runtime_unjoined_duty_cycle=runtime_unjoined_duty,
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
        # Monotonic hardware and SHM counters MUST NOT include scan-order `chip`
        # (or dynamic VFIO labels) so losing a lower-index TPU never resets
        # surviving series identity and breaks `rate()` / `increase()`.
        stable_counter_labels = {
            "pci_bdf": d.pci_bdf,
            "device_id": d.device_id,
            "generation": d.generation,
        }
        if d.aer_correctable_total is not None:
            aer_corr_samples.append((stable_counter_labels, str(d.aer_correctable_total)))
        if d.aer_fatal_total is not None:
            aer_fatal_samples.append((stable_counter_labels, str(d.aer_fatal_total)))
        for sib_bdf, (sib_corr, sib_fatal) in sorted(d.sibling_aer_counters.items()):
            sib_aer_labels = {
                "pci_bdf": sib_bdf,
                "device_id": d.device_id,
                "generation": d.generation,
            }
            if sib_corr is not None:
                aer_corr_samples.append((sib_aer_labels, str(sib_corr)))
            if sib_fatal is not None:
                aer_fatal_samples.append((sib_aer_labels, str(sib_fatal)))
        if d.shm_hbm_allocated_bytes is not None:
            shm_hbm_alloc_samples.append((base_labels, str(d.shm_hbm_allocated_bytes)))
        if d.mxu_active_cycles is not None:
            mxu_cycles_samples.append((stable_counter_labels, str(d.mxu_active_cycles)))
        if d.dma_bytes_transferred is not None:
            dma_bytes_samples.append((stable_counter_labels, str(d.dma_bytes_transferred)))

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
    if snapshot.runtime_unjoined_duty_cycle:
        def _unjoined_sort_key(kv: Tuple[str, float]) -> Tuple[int, Any]:
            parsed_id = _parse_runtime_device_id(kv[0])
            return (0, parsed_id) if parsed_id is not None else (1, kv[0])

        unjoined_duty_samples = [
            ({"runtime_device_id": raw_id}, _fmt_float(val))
            for raw_id, val in sorted(
                snapshot.runtime_unjoined_duty_cycle.items(),
                key=_unjoined_sort_key,
            )
        ]
        add_metric(
            "freetpu_tpu_runtime_unjoined_duty_cycle_percent",
            "TPU Runtime Utilization MXU duty cycle percentage keyed by raw libtpu reported ID when chip-to-core layout cannot be verified from PCI inventory.",
            "gauge",
            unjoined_duty_samples,
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
        grpc_duty_layout: str = "auto",
        shm_root: Optional[Path] = DEFAULT_SHM_ROOT,
        shm_trusted_uids: Optional[Sequence[int]] = None,
        shm_max_age_s: float = DEFAULT_SHM_MAX_AGE_S,
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
        self.grpc_duty_layout = grpc_duty_layout
        self.shm_root = shm_root
        self.shm_trusted_uids = tuple(shm_trusted_uids) if shm_trusted_uids is not None else None
        self.shm_max_age_s = shm_max_age_s
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
                grpc_duty_layout=self.grpc_duty_layout,
                shm_root=self.shm_root,
                shm_trusted_uids=self.shm_trusted_uids,
                shm_max_age_s=self.shm_max_age_s,
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


def _load_chip_id_map(raw: Optional[Path | str]) -> Optional[Dict[str, str]]:
    if raw is None:
        return None
    if isinstance(raw, Path):
        text = raw.read_text(encoding="utf-8")
    else:
        text = str(raw).strip()
        if not text:
            return None
        # Detect inline JSON before probing the filesystem so long inline maps
        # (>255 bytes) never fail with ENAMETOOLONG in Path.is_file().
        if not text.startswith(("{", "[")):
            text = Path(text).read_text(encoding="utf-8")

    def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        seen_norm: Dict[str, str] = {}
        for k, v in pairs:
            if not isinstance(k, str) or not k.strip():
                raise ValueError(
                    f"Invalid --chip-id-map key {k!r}: must be a non-empty string"
                )
            norm_k = k.strip()
            norm_cmp = norm_k.lower()
            if norm_cmp in seen_norm:
                raise ValueError(
                    f"Duplicate --chip-id-map key {k!r} (conflicts with {seen_norm[norm_cmp]!r})"
                )
            seen_norm[norm_cmp] = k
            store_k = norm_k if norm_k.startswith("/") else norm_cmp
            obj[store_k] = v
        return obj

    parsed = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(parsed, dict):
        raise ValueError(f"--chip-id-map must be a JSON object: {raw}")
    return {str(k): str(v).strip() for k, v in parsed.items()}


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
        type=str,
        default=None,
        help=(
            "Optional inline JSON or file path mapping {pci_bdf_or_unique_vfio_path: runtime_chip_id} "
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
        "--grpc-duty-layout",
        choices=VALID_GRPC_DUTY_LAYOUTS,
        default="auto",
        help=(
            "Direct gRPC duty-cycle ID layout: 'auto' (default: 1:1 on single-core-per-chip "
            "generations and verified pigeonhole Ironwood per-core replies, keeping ambiguous "
            "replies unjoined), 'per-core' (explicit 1:1 join), or 'per-chip' (expand across "
            "devices_per_chip when PCI inventory is present)."
        ),
    )
    parser.add_argument(
        "--shm-root",
        type=Path,
        default=DEFAULT_SHM_ROOT,
        help=(
            f"Publisher-controlled shared-memory directory for libopentpu.so SeqLock files "
            f"(default: {DEFAULT_SHM_ROOT}; must be owned by a trusted UID and not group/world-writable)."
        ),
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
        "--shm-max-age-s",
        type=float,
        default=DEFAULT_SHM_MAX_AGE_S,
        help=(
            f"Maximum age in seconds for SeqLock SHM publisher heartbeat timestamp_ns "
            f"(default: {DEFAULT_SHM_MAX_AGE_S}; expired samples are omitted)."
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
                grpc_duty_layout=args.grpc_duty_layout,
                shm_root=args.shm_root,
                shm_trusted_uids=args.shm_trusted_uids,
                shm_max_age_s=args.shm_max_age_s,
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
        grpc_duty_layout=args.grpc_duty_layout,
        shm_root=args.shm_root,
        shm_trusted_uids=args.shm_trusted_uids,
        shm_max_age_s=args.shm_max_age_s,
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
