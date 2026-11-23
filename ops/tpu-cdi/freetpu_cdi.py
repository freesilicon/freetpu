#!/usr/bin/env python3
"""FreeTPU Open CNCF CDI Spec Generator & Container Passthrough Toolkit.

Scans /sys/bus/pci/devices/* (vendor 0x1ae0) and VFIO / iommufd device nodes
on VFIO-backed Cloud TPU hosts (v5e, v5p, v6e Trillium, and tpu7x Ironwood)
to generate a boot-scoped CNCF Container Device Interface (CDI) specification
(default: CDI v0.6.0 for broad containerd 2.0/2.1+ compatibility, with opt-in
`--cdi-version 1.1.0`, output default: /var/run/cdi/freetpu.json) exposing:
  - `freetpu.dev/tpu=pci-<bdf>` (and `group-pci-...` for shared IOMMU groups)
  - NUMA / PCIe-root-complex-aware capacity-pinned sub-host presets
    (`numa0-pcie-0000-00-4`, `numa1-pcie-0000-80-4`, `pcie-0000-00-4` when NUMA
    is unavailable, plus verified range aliases `0-3`, `4-7`) on multi-root hosts (e.g. Cloud
    TPU v6e-8 `0000:00` vs `0000:80`)
  - `freetpu.dev/tpu=all`

Also supports:
  - Legacy VFIO (`/dev/vfio/vfio` + `/dev/vfio/<iommu_group>`) preserved as
    default when usable, plus explicit `--vfio-mode iommufd` (or `cdev`) opt-in
    and pure-iommufd fallback (`/dev/iommu` + verified `<pci>/vfio-dev/vfioN`)
  - Opt-in single-host isolated container networking via `--single-host-env`
    (or `--network-mode isolated`), injecting `MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353`,
    `TPU_HOST_BOUNDS=1,1,1`, `TPU_WORKER_HOSTNAMES=localhost`, `TPU_WORKER_ID=0`,
    `TPU_SKIP_MDS_QUERY=true` (plus physical `TPU_VISIBLE_CHIPS`, logical
    `TPU_VISIBLE_DEVICES`, and physical-chip-derived `TPU_CHIPS_PER_HOST_BOUNDS`
    on full-allocation selectors `numa*`/`pcie*`/`0-3`/`all` only). By default
    (`--network-mode host`, without `--single-host-env`), no `libtpu` environment
    overrides are injected so existing multi-host workloads retain their
    configured peer discovery, visibility, and bounds.
  - `validate`: strict CNCF CDI structural & live host VFIO/iommufd validator
  - `sync`: boot-scoped generator/reconciler that invalidates stale specs on drift
  - `docker-args`: prints Docker / Podman CLI flags (memlock default; optional IPC_LOCK)

Note: Legacy `/dev/accel*` Cloud TPU hosts (e.g. Cloud TPU v4 using Google's
`accel` driver) are not supported by VFIO CDI passthrough.

Zero third-party dependencies: uses Python standard library only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Default to CDI v0.6.0 — the minimum spec version needed for all generated
# fields (annotations, deviceNodes.hostPath, env) and compatible with supported
# containerd releases (unaffected 2.0.x, patched 2.1.x >= 2.1.9, 2.2.x >= 2.2.5,
# 2.3.x >= 2.3.2, >= 2.4.0; see GHSA-33vj-92qq-66hc). Operators may opt into
# newer CDI versions (e.g. "1.1.0") via --cdi-version.
DEFAULT_CDI_VERSION = "0.6.0"
CDI_VERSION = DEFAULT_CDI_VERSION
SUPPORTED_CDI_VERSIONS: Tuple[str, ...] = ("0.6.0", "0.7.0", "0.8.0", "1.0.0", "1.1.0")
CDI_KIND = "freetpu.dev/tpu"
GOOGLE_PCI_VENDOR_ID = "0x1ae0"
# Boot-scoped CNCF CDI dynamic spec directory (/var/run is tmpfs on Linux so
# stale IOMMU group assignments never persist across reboots).
DEFAULT_OUTPUT_PATH = Path("/var/run/cdi/freetpu.json")
NETWORK_MODES = frozenset({"isolated", "host"})
VFIO_MODES = frozenset({"auto", "legacy", "iommufd", "cdev"})

# Known Google Cloud TPU PCI device IDs under vendor 0x1ae0.
TPU_PCI_DEVICES: Dict[str, Tuple[str, str]] = {
    "0x0027": ("v2-v3", "TPU v2/v3"),
    "0x0056": ("v4lite", "TPU v4lite"),
    "0x005e": ("v4", "TPU v4"),
    "0x0062": ("v5p", "TPU v5p"),
    "0x0063": ("v5e", "TPU v5e"),
    "0x006e": ("v6e-pf", "TPU v6e (PF)"),
    "0x006f": ("v6e", "TPU v6e (Trillium)"),
    "0x0075": ("tpu7x", "TPU tpu7x (Ironwood, pending live hardware confirmation #21)"),
    "0x0076": ("tpu7x", "TPU tpu7x (Ironwood)"),
}

# CNCF CDI parser grammar (ValidateVendorName / ValidateClassName):
# vendor and class must each start with an ASCII letter, contain only
# [A-Za-z0-9_.-], end with an ASCII letter or digit, and have length >= 2
# (CDI v0.8.0+ / containerd 2.0.0 panics on 1-character components).
_CDI_KIND_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.-]*[A-Za-z0-9]/[A-Za-z][A-Za-z0-9_.-]*[A-Za-z0-9]\Z"
)
# CNCF CDI device names must start and end with an ASCII alphanumeric char.
_CDI_DEVICE_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_CDI_CONTAINER_EDITS_KEYS_V06 = frozenset({"env", "deviceNodes", "hooks", "mounts"})
_CDI_CONTAINER_EDITS_KEYS_V07 = _CDI_CONTAINER_EDITS_KEYS_V06 | frozenset(
    {"intelRdt", "additionalGids"}
)
_CDI_CONTAINER_EDITS_KEYS_V11 = _CDI_CONTAINER_EDITS_KEYS_V07 | frozenset({"netDevices"})
# Exact CNCF CDI specs-go/config.go IntelRdt JSON keys:
# v0.7.0 - v1.0.0: closID (string), l3CacheSchema (string), memBwSchema (string),
#                  enableCMT (bool), enableMBM (bool)
# v1.1.0:          closID (string), l3CacheSchema (string), memBwSchema (string),
#                  schemata ([]string), enableMonitoring (bool)
#                  (enableCMT and enableMBM were removed in v1.1.0)
_ALLOWED_INTEL_RDT_KEYS_V07 = frozenset(
    {"closID", "l3CacheSchema", "memBwSchema", "enableCMT", "enableMBM"}
)
_ALLOWED_INTEL_RDT_KEYS_V11 = frozenset(
    {"closID", "l3CacheSchema", "memBwSchema", "schemata", "enableMonitoring"}
)
_ALLOWED_NET_DEVICE_KEYS = frozenset({"hostInterfaceName", "name"})
# PCI functions (chiplets/cores) exposed per physical TPU chip.
# Ironwood (`tpu7x`: 0x0075/0x0076) and legacy v2/v3 (`0x0027`) expose 2 PCI
# functions per physical chip; single-function generations default to 1.
_PCI_FUNCTIONS_PER_PHYSICAL_CHIP: Dict[str, int] = {
    "v2-v3": 2,
    "tpu7x": 2,
}
_SYSFS_PCI_ROOT_RE = re.compile(r"^pci([0-9a-fA-F]{4,}:[0-9a-fA-F]{2})\Z")
_ALLOWED_HOOK_NAMES = frozenset(
    {
        "prestart",
        "createRuntime",
        "createContainer",
        "startContainer",
        "poststart",
        "poststop",
    }
)
_ALLOWED_DEVICE_TYPES = frozenset({"c", "b", "p", "u"})
_ALLOWED_PERMISSIONS_CHARS = frozenset({"r", "w", "m"})
_VFIO_CDEV_NAME_RE = re.compile(r"^vfio[0-9]+\Z")


def _is_valid_iommufd_cdev_path(path: str) -> bool:
    """Return True iff `path` is a direct `/dev/vfio/devices/vfio<N>` cdev node path."""
    prefix = "/dev/vfio/devices/"
    if not path.startswith(prefix):
        return False
    basename = path[len(prefix):]
    return "/" not in basename and bool(_VFIO_CDEV_NAME_RE.match(basename))


@dataclass(frozen=True)
class DiscoveredChip:
    index: int
    pci_bdf: str
    vendor_id: str
    device_id: str
    generation: str
    model_name: str
    driver: str
    iommu_group: str
    vfio_host_path: Path
    vfio_container_path: str
    non_tpu_group_members: Tuple[str, ...] = ()
    numa_node: Optional[int] = None
    pcie_root: str = ""
    vfio_mode: str = "vfio-group"
    iommufd_cdev: Optional[str] = None
    runtime_chip_id: Optional[int] = None


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
    if path.is_file():
        return _read_text(path)
    return None


def _make_device_node(
    container_path: str,
    host_path: Path,
    *,
    permissions: str = "rw",
) -> Dict[str, Any]:
    # Omit static major/minor from deviceNodes (Linux allocates VFIO group minor
    # numbers dynamically and independently of IOMMU group IDs), while recording
    # live char device numbers in `freetpu.dev/vfio-devnums` annotations.
    # Note: CDI v0.8.0 (containerd 2.0) resolves omitted major/minor on first
    # injection and caches them in memory until the spec file changes or is
    # removed, and refreshing a spec does not revoke device nodes already
    # injected into running containers. Before any PCI/VFIO unbind or rebind,
    # operators MUST: (1) stop/drain affected container consumers, (2) stop or
    # pause `freetpu_cdi.py sync` so it cannot recreate the spec during
    # maintenance, (3) remove `/var/run/cdi/freetpu.json` and confirm the
    # runtime has withdrawn the CDI selectors, (4) perform the rebind, and
    # (5) regenerate or resume `freetpu_cdi.py sync`.
    return {
        "path": container_path,
        "hostPath": os.path.abspath(host_path),
        "type": "c",
        "permissions": permissions,
    }


def _collect_iommu_group_members(
    sysfs_root: Path,
    pci_entries: Sequence[Path],
) -> Tuple[Dict[str, set[str]], Dict[str, set[str]]]:
    """Map every IOMMU group ID to PCI BDFs seen in sysfs and verified kernel group listings.

    Returns `(all_members, kernel_members)` where `kernel_members[group]` is populated
    ONLY when `<sysfs_root>/kernel/iommu_groups/<group>/devices` exists and is readable.
    Callers MUST require a verified `kernel_members` entry containing the candidate TPU
    before publishing a VFIO group so a filtered or partially mounted `--sysfs-root`
    cannot hide sibling devices in the same IOMMU group.
    """
    members: Dict[str, set[str]] = {}
    kernel_members: Dict[str, set[str]] = {}
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
                listed = {member.name for member in dev_dir.iterdir()}
            except OSError:
                continue
            kernel_members[grp_dir.name] = listed
            members.setdefault(grp_dir.name, set()).update(listed)
    return members, kernel_members


_DEVNUM_RE = re.compile(r"^[0-9]+:[0-9]+$")


def _read_kernel_vfio_devnum(sysfs_root: Path, *, group: Optional[str] = None) -> Optional[str]:
    """Read kernel-exported `<major>:<minor>` from sysfs for `/dev/vfio/<group>` or `/dev/vfio/vfio`.

    Checks `/sys/class/...` first and falls back to the identical canonical
    `/sys/devices/virtual/...` path (used inside minimal container mounts that
    mount `/sys/devices` read-only without mounting `/sys/class`).
    """
    if group is None:
        candidates = (
            sysfs_root / "class" / "misc" / "vfio" / "dev",
            sysfs_root / "devices" / "virtual" / "misc" / "vfio" / "dev",
        )
    else:
        candidates = (
            sysfs_root / "class" / "vfio" / group / "dev",
            sysfs_root / "devices" / "virtual" / "vfio" / group / "dev",
        )
    for path in candidates:
        raw = _read_text(path)
        if raw and _DEVNUM_RE.fullmatch(raw):
            return raw
    return None


def _read_kernel_iommu_devnum(sysfs_root: Path) -> Optional[str]:
    """Read kernel-exported `<major>:<minor>` from sysfs for `/dev/iommu` when exported."""
    candidates = (
        sysfs_root / "class" / "misc" / "iommu" / "dev",
        sysfs_root / "devices" / "virtual" / "misc" / "iommu" / "dev",
    )
    for path in candidates:
        raw = _read_text(path)
        if raw and _DEVNUM_RE.fullmatch(raw):
            return raw
    return None


def _normalize_vfio_mode(vfio_mode: str) -> str:
    if vfio_mode not in VFIO_MODES:
        raise ValueError(
            f"Invalid vfio_mode {vfio_mode!r}: expected one of {sorted(VFIO_MODES)}"
        )
    return "iommufd" if vfio_mode == "cdev" else vfio_mode


def _resolve_iommufd_cdev(
    pci_entry: Path,
    sysfs_root: Path,
) -> Tuple[str, Optional[str]]:
    """Resolve the verified Linux 6.8+ iommufd cdev (`vfioN`) bound directly to `pci_entry`.

    Requires an explicit `<pci_entry>/vfio-dev/vfio<N>` sysfs association tying
    the character device directly to this PCI function. Never guesses `vfioN`
    from TPU enumeration order.
    """
    vfio_dev_dir = pci_entry / "vfio-dev"
    if not vfio_dev_dir.is_dir():
        raise RuntimeError(
            f"Cannot resolve verified iommufd VFIO cdev for PCI device {pci_entry.name}: "
            f"missing sysfs association directory {vfio_dev_dir}"
        )
    try:
        cdevs = sorted(
            (p for p in vfio_dev_dir.iterdir() if _VFIO_CDEV_NAME_RE.match(p.name)),
            key=lambda p: p.name,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to read sysfs iommufd VFIO association directory {vfio_dev_dir}: {exc}"
        ) from exc

    if not cdevs:
        raise RuntimeError(
            f"Cannot resolve verified iommufd VFIO cdev for PCI device {pci_entry.name}: "
            f"no vfio<N> entry under {vfio_dev_dir}"
        )
    if len(cdevs) > 1:
        names = ", ".join(p.name for p in cdevs)
        raise RuntimeError(
            f"Ambiguous iommufd VFIO cdev entries under {vfio_dev_dir} for {pci_entry.name}: {names}"
        )

    cdev_dir = cdevs[0]
    devnum = _read_text(cdev_dir / "dev")
    if devnum and not _DEVNUM_RE.fullmatch(devnum):
        devnum = None
    if devnum is None:
        for fallback_dev in (
            sysfs_root / "class" / "vfio-dev" / cdev_dir.name / "dev",
            sysfs_root / "devices" / "virtual" / "vfio-dev" / cdev_dir.name / "dev",
        ):
            class_devnum = _read_text(fallback_dev)
            if class_devnum and _DEVNUM_RE.fullmatch(class_devnum):
                devnum = class_devnum
                break
    return cdev_dir.name, devnum


def _pcie_root_complex(pci_bdf: str) -> str:
    """Return `<domain>:<bus>` prefix from a PCI BDF (fallback when sysfs root bridge is absent)."""
    parts = pci_bdf.strip().split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}"
    return pci_bdf.strip()


def _resolve_pcie_root_complex(pci_entry: Path) -> str:
    """Derive the PCIe root complex (`<domain>:<root_bus>`) from resolved sysfs ancestry.

    On Linux, `/sys/bus/pci/devices/<bdf>` symlinks into
    `/sys/devices/pci<domain>:<root_bus>/<bridges...>/<bdf>`. Walking the
    resolved path finds the host bridge (`pciXXXX:XX`) so chips behind PCIe
    switches on different downstream bus numbers stay in the same topology
    domain. Falls back to `_pcie_root_complex(pci_entry.name)` for flat
    synthetic fixtures without a `pciXXXX:XX` ancestor directory.
    """
    try:
        resolved = Path(os.path.realpath(pci_entry))
    except OSError:
        resolved = pci_entry
    for part in resolved.parts:
        m = _SYSFS_PCI_ROOT_RE.match(part)
        if m:
            return m.group(1).lower()
    return _pcie_root_complex(pci_entry.name)


def _parse_numa_node(pci_entry: Path) -> Optional[int]:
    raw = _read_text(pci_entry / "numa_node")
    if raw is None:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val if val >= 0 else None


def _parse_runtime_chip_id(pci_entry: Path) -> Optional[int]:
    """Read optional sysfs `chip_id` integer when exported by the kernel driver."""
    raw = _read_text(pci_entry / "chip_id")
    if raw is None:
        return None
    try:
        val = int(raw, 10)
    except ValueError:
        return None
    return val if val >= 0 else None


def _load_chip_id_map(raw: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse `--chip-id-map` JSON object or file path mapping PCI BDF -> non-negative runtime int ID."""
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip()
    # Parse inline JSON before probing the filesystem so valid inline maps
    # exceeding NAME_MAX (255 bytes) never fail with ENAMETOOLONG in is_file().
    if not text.startswith(("{", "[")):
        path = Path(text)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("--chip-id-map must be a JSON object mapping PCI BDF strings to integer IDs")
    result: Dict[str, int] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid --chip-id-map key {k!r}: must be a non-empty PCI BDF string")
        if isinstance(v, bool):
            raise ValueError(f"Invalid --chip-id-map value for {k!r}: boolean is not a chip ID")
        if isinstance(v, str) and v.strip().isdigit():
            iv = int(v.strip(), 10)
        elif isinstance(v, int) and v >= 0:
            iv = v
        else:
            raise ValueError(
                f"Invalid --chip-id-map value {v!r} for {k!r}: must be a non-negative integer"
            )
        result[k.strip()] = iv
    return result


def _legacy_vfio_control_usable(
    legacy_control: Path,
    effective_iommu: Path,
    sysfs_root: Path,
    *,
    legacy_is_iommu_compat: bool,
    allow_missing_vfio: bool,
) -> bool:
    """Return True iff the legacy VFIO control node exists and matches kernel sysfs."""
    control_node = (
        effective_iommu
        if (legacy_is_iommu_compat and legacy_control.is_symlink())
        else legacy_control
    )
    if not control_node.exists():
        return False
    if allow_missing_vfio:
        return True
    actual_dn = _stat_char_devnum(control_node)
    if actual_dn is None:
        return False
    kernel_dn = (
        _read_kernel_iommu_devnum(sysfs_root)
        if legacy_is_iommu_compat
        else _read_kernel_vfio_devnum(sysfs_root, group=None)
    )
    return kernel_dn is not None and actual_dn == kernel_dn


def _iommufd_control_usable(
    effective_iommu: Path,
    sysfs_root: Path,
    *,
    allow_missing_vfio: bool,
) -> bool:
    """Return True iff `/dev/iommu` exists and matches kernel sysfs `misc/iommu/dev`."""
    if not effective_iommu.exists():
        return False
    if allow_missing_vfio:
        return True
    actual_dn = _stat_char_devnum(effective_iommu)
    if actual_dn is None:
        return False
    kernel_dn = _read_kernel_iommu_devnum(sysfs_root)
    return kernel_dn is not None and actual_dn == kernel_dn


def _iommufd_cdev_usable(
    tpu_candidates: Sequence[Tuple[Path, str, str, str]],
    sysfs_root: Path,
    dev_root: Path,
    *,
    allow_missing_vfio: bool,
) -> bool:
    """Return True iff every discovered TPU exposes a verified iommufd VFIO cdev."""
    if not tpu_candidates:
        return False
    for entry, _device_id, _gen, _model_name in tpu_candidates:
        try:
            cdev_name, kernel_cdev_dn = _resolve_iommufd_cdev(entry, sysfs_root)
        except RuntimeError:
            return False
        if not allow_missing_vfio:
            cdev_path = dev_root / "vfio" / "devices" / cdev_name
            if not cdev_path.exists():
                return False
            actual_cdev_dn = _stat_char_devnum(cdev_path)
            if (
                actual_cdev_dn is None
                or kernel_cdev_dn is None
                or actual_cdev_dn != kernel_cdev_dn
            ):
                return False
    return True


def _legacy_vfio_groups_usable(
    tpu_candidates: Sequence[Tuple[Path, str, str, str]],
    sysfs_root: Path,
    dev_root: Path,
    *,
    allow_missing_vfio: bool,
) -> bool:
    """Return True iff every discovered TPU exposes a verified legacy `/dev/vfio/<group>` node."""
    if not tpu_candidates:
        return False
    for entry, _device_id, _gen, _model_name in tpu_candidates:
        iommu_group = _read_symlink_name(entry / "iommu_group") or ""
        if not iommu_group:
            return False
        if not allow_missing_vfio:
            group_path = dev_root / "vfio" / iommu_group
            if not group_path.exists():
                return False
            actual_group_dn = _stat_char_devnum(group_path)
            kernel_group_dn = _read_kernel_vfio_devnum(sysfs_root, group=iommu_group)
            if (
                actual_group_dn is None
                or kernel_group_dn is None
                or actual_group_dn != kernel_group_dn
            ):
                return False
    return True


def discover_tpu_chips(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    *,
    allow_missing_vfio: bool = False,
    iommu_dev_path: Optional[Path] = None,
    vfio_mode: str = "auto",
) -> Tuple[List[DiscoveredChip], Path]:
    """Scan sysfs for VFIO/iommufd-backed Google 0x1ae0 TPU PCI devices and device nodes.

    In default `vfio_mode="auto"`, legacy VFIO (`/dev/vfio/vfio` + `/dev/vfio/<group>`)
    is preserved whenever `/dev/vfio/vfio` is present so hosts with working legacy VFIO
    never break or withdraw group nodes merely because `/dev/iommu` also exists.
    `iommufd` cdev mode is selected either explicitly via `vfio_mode="iommufd"` (or `"cdev"`)
    or in `"auto"` mode when legacy `/dev/vfio/vfio` is absent and `/dev/iommu` is present.
    """
    norm_mode = _normalize_vfio_mode(vfio_mode)
    sysfs_root = Path(os.path.abspath(sysfs_root))
    dev_root = Path(os.path.abspath(dev_root))
    pci_dir = sysfs_root / "bus" / "pci" / "devices"
    if not pci_dir.is_dir():
        raise FileNotFoundError(f"PCI sysfs directory not found: {pci_dir}")

    pci_entries = sorted(pci_dir.iterdir(), key=lambda p: p.name)
    group_members, kernel_group_members = _collect_iommu_group_members(sysfs_root, pci_entries)

    tpu_candidates: List[Tuple[Path, str, str, str]] = []
    tpu_bdfs: set[str] = set()
    for entry in pci_entries:
        vendor_raw = _read_text(entry / "vendor")
        if not vendor_raw or _normalize_hex_id(vendor_raw) != GOOGLE_PCI_VENDOR_ID:
            continue

        device_id = _normalize_hex_id(_read_text(entry / "device") or "0x0000")
        if device_id not in TPU_PCI_DEVICES:
            # Skip non-TPU Google PCI functions under vendor 0x1ae0 (e.g. gVNIC 0x0042).
            continue
        gen, model_name = TPU_PCI_DEVICES[device_id]
        tpu_candidates.append((entry, device_id, gen, model_name))
        tpu_bdfs.add(entry.name)

    legacy_control = dev_root / "vfio" / "vfio"
    effective_iommu = (
        Path(os.path.abspath(iommu_dev_path))
        if iommu_dev_path is not None
        else (dev_root / "iommu")
    )
    has_legacy_control = legacy_control.exists() or legacy_control.is_symlink()
    has_iommu_control = effective_iommu.exists() or effective_iommu.is_symlink()

    legacy_is_iommu_compat = False
    if has_legacy_control and has_iommu_control:
        if legacy_control.is_symlink():
            try:
                legacy_is_iommu_compat = os.path.realpath(legacy_control) == os.path.realpath(
                    effective_iommu
                )
            except OSError:
                legacy_is_iommu_compat = False
        elif _read_kernel_vfio_devnum(sysfs_root) is None:
            kernel_iommu_dn = _read_kernel_iommu_devnum(sysfs_root)
            if (
                kernel_iommu_dn is not None
                and _stat_char_devnum(legacy_control) == kernel_iommu_dn
            ):
                legacy_is_iommu_compat = True

    if norm_mode == "iommufd":
        use_iommufd = True
    elif norm_mode == "legacy":
        use_iommufd = False
    else:
        # auto: prefer legacy VFIO whenever BOTH its control node and all
        # `/dev/vfio/<iommu_group>` nodes are verified and usable. On Linux 6.8+
        # (`CONFIG_IOMMUFD_VFIO_CONTAINER`), `/dev/vfio/vfio` may be exposed as
        # either a compatibility symlink or a real misc char device even when
        # legacy `/dev/vfio/<group>` nodes are omitted; in that case fall back
        # to iommufd cdev whenever `/dev/iommu` and every TPU's `vfio-dev/vfioN`
        # cdev are verified and usable. Never fall back past an unrecognized
        # symlink at `/dev/vfio/vfio`.
        if not has_legacy_control:
            use_iommufd = has_iommu_control
        elif has_iommu_control and not (
            legacy_control.is_symlink() and not legacy_is_iommu_compat
        ):
            legacy_ctrl_usable = _legacy_vfio_control_usable(
                legacy_control,
                effective_iommu,
                sysfs_root,
                legacy_is_iommu_compat=legacy_is_iommu_compat,
                allow_missing_vfio=allow_missing_vfio,
            )
            groups_usable = _legacy_vfio_groups_usable(
                tpu_candidates,
                sysfs_root,
                dev_root,
                allow_missing_vfio=allow_missing_vfio,
            )
            if legacy_ctrl_usable and groups_usable:
                use_iommufd = False
            else:
                iommu_ctrl_usable = _iommufd_control_usable(
                    effective_iommu,
                    sysfs_root,
                    allow_missing_vfio=allow_missing_vfio,
                )
                cdevs_usable = _iommufd_cdev_usable(
                    tpu_candidates,
                    sysfs_root,
                    dev_root,
                    allow_missing_vfio=allow_missing_vfio,
                )
                use_iommufd = iommu_ctrl_usable and cdevs_usable
        else:
            use_iommufd = False

    if use_iommufd:
        control_node = effective_iommu
        if not allow_missing_vfio:
            if not has_iommu_control:
                raise FileNotFoundError(
                    f"iommufd control device not found: {effective_iommu}"
                )
            actual_iommu_dn = _stat_char_devnum(effective_iommu)
            if actual_iommu_dn is None:
                raise RuntimeError(
                    f"iommufd control node is not a character device: {effective_iommu}"
                )
            kernel_iommu_dn = _read_kernel_iommu_devnum(sysfs_root)
            if kernel_iommu_dn is None:
                raise RuntimeError(
                    f"Unable to verify kernel device number for iommufd control node {effective_iommu} "
                    f"(missing {sysfs_root}/class/misc/iommu/dev and {sysfs_root}/devices/virtual/misc/iommu/dev)"
                )
            if actual_iommu_dn != kernel_iommu_dn:
                raise RuntimeError(
                    f"iommufd control node {effective_iommu} device number {actual_iommu_dn!r} "
                    f"does not match kernel sysfs misc/iommu/dev {kernel_iommu_dn!r}"
                )
    else:
        control_node = (
            effective_iommu
            if (legacy_is_iommu_compat and legacy_control.is_symlink())
            else legacy_control
        )
        if not allow_missing_vfio:
            if not control_node.exists():
                raise FileNotFoundError(
                    f"VFIO control device not found: {control_node} "
                    "(freetpu-cdi requires VFIO-backed Cloud TPUs using vfio-pci; "
                    "/dev/accel* hosts such as Cloud TPU v4 are not supported)"
                )
            actual_container_dn = _stat_char_devnum(control_node)
            if actual_container_dn is None:
                raise RuntimeError(
                    f"VFIO control node is not a character device: {control_node}"
                )
            if legacy_is_iommu_compat:
                kernel_container_dn = _read_kernel_iommu_devnum(sysfs_root)
                kernel_label = "misc/iommu/dev"
            else:
                kernel_container_dn = _read_kernel_vfio_devnum(sysfs_root, group=None)
                kernel_label = "misc/vfio/dev"
            if kernel_container_dn is None:
                raise RuntimeError(
                    f"Unable to verify kernel device number for VFIO control node {control_node} "
                    f"(missing {sysfs_root}/class/{kernel_label} and {sysfs_root}/devices/virtual/{kernel_label})"
                )
            if actual_container_dn != kernel_container_dn:
                raise RuntimeError(
                    f"VFIO control node {control_node} device number {actual_container_dn!r} "
                    f"does not match kernel sysfs {kernel_label} {kernel_container_dn!r}"
                )

    chips: List[DiscoveredChip] = []
    for entry, device_id, gen, model_name in tpu_candidates:
        driver = _read_symlink_name(entry / "driver") or "unbound"
        if driver != "vfio-pci" and not allow_missing_vfio:
            raise RuntimeError(
                f"TPU PCI device {entry.name} is bound to driver {driver!r}, not 'vfio-pci' "
                "(freetpu-cdi supports VFIO-backed Cloud TPUs only; /dev/accel* hosts are not supported)"
            )

        iommu_group = _read_symlink_name(entry / "iommu_group") or ""
        if not iommu_group:
            if allow_missing_vfio:
                continue
            raise RuntimeError(
                f"TPU PCI device {entry.name} has no iommu_group in {entry}"
            )

        verified_kernel_members = kernel_group_members.get(iommu_group)
        if verified_kernel_members is None:
            raise RuntimeError(
                f"Refusing VFIO group {iommu_group} for TPU {entry.name}: "
                f"cannot verify full IOMMU group membership because "
                f"{sysfs_root}/kernel/iommu_groups/{iommu_group}/devices is missing or unreadable"
            )
        if entry.name not in verified_kernel_members:
            raise RuntimeError(
                f"Refusing VFIO group {iommu_group} for TPU {entry.name}: "
                f"kernel IOMMU group membership list "
                f"({sysfs_root}/kernel/iommu_groups/{iommu_group}/devices) does not contain {entry.name}"
            )

        all_in_group = group_members.get(iommu_group, set()) | verified_kernel_members
        non_tpu_members = tuple(sorted(all_in_group - tpu_bdfs))
        if non_tpu_members:
            raise RuntimeError(
                f"Refusing VFIO group {iommu_group} for TPU {entry.name}: "
                f"IOMMU group also exposes non-TPU PCI device(s): {', '.join(non_tpu_members)}"
            )

        numa_node = _parse_numa_node(entry)
        pcie_root = _resolve_pcie_root_complex(entry)

        if use_iommufd:
            cdev_name, kernel_cdev_dn = _resolve_iommufd_cdev(entry, sysfs_root)
            vfio_host_path = dev_root / "vfio" / "devices" / cdev_name
            vfio_container_path = f"/dev/vfio/devices/{cdev_name}"
            chip_vfio_mode = "iommufd"
            iommufd_cdev = cdev_name
            if not allow_missing_vfio:
                if not vfio_host_path.exists():
                    raise FileNotFoundError(
                        f"iommufd VFIO cdev not found for {entry.name}: {vfio_host_path}"
                    )
                actual_cdev_dn = _stat_char_devnum(vfio_host_path)
                if actual_cdev_dn is None:
                    raise RuntimeError(
                        f"iommufd VFIO cdev for {entry.name} is not a character device: {vfio_host_path}"
                    )
                if kernel_cdev_dn is None:
                    raise RuntimeError(
                        f"Unable to verify kernel device number for iommufd VFIO cdev {cdev_name} ({entry.name}): "
                        f"missing {entry}/vfio-dev/{cdev_name}/dev"
                    )
                if actual_cdev_dn != kernel_cdev_dn:
                    raise RuntimeError(
                        f"iommufd VFIO cdev {vfio_host_path} for {entry.name} has device number "
                        f"{actual_cdev_dn!r}, which does not match kernel sysfs {cdev_name}/dev {kernel_cdev_dn!r}"
                    )
        else:
            vfio_host_path = dev_root / "vfio" / iommu_group
            vfio_container_path = f"/dev/vfio/{iommu_group}"
            chip_vfio_mode = "vfio-group"
            iommufd_cdev = None
            if not allow_missing_vfio:
                if not vfio_host_path.exists():
                    raise FileNotFoundError(
                        f"VFIO group node not found for {entry.name}: {vfio_host_path}"
                    )
                actual_group_dn = _stat_char_devnum(vfio_host_path)
                if actual_group_dn is None:
                    raise RuntimeError(
                        f"VFIO group node for {entry.name} is not a character device: {vfio_host_path}"
                    )
                kernel_group_dn = _read_kernel_vfio_devnum(sysfs_root, group=iommu_group)
                if kernel_group_dn is None:
                    raise RuntimeError(
                        f"Unable to verify kernel device number for VFIO group {iommu_group} ({entry.name}): "
                        f"missing {sysfs_root}/class/vfio/{iommu_group}/dev and "
                        f"{sysfs_root}/devices/virtual/vfio/{iommu_group}/dev"
                    )
                if actual_group_dn != kernel_group_dn:
                    raise RuntimeError(
                        f"VFIO group node {vfio_host_path} for {entry.name} has device number "
                        f"{actual_group_dn!r}, which does not match kernel sysfs group {iommu_group} "
                        f"device number {kernel_group_dn!r}"
                    )

        chips.append(
            DiscoveredChip(
                index=len(chips),
                pci_bdf=entry.name,
                vendor_id=GOOGLE_PCI_VENDOR_ID,
                device_id=device_id,
                generation=gen,
                model_name=model_name,
                driver=driver,
                iommu_group=iommu_group,
                vfio_host_path=vfio_host_path,
                vfio_container_path=vfio_container_path,
                non_tpu_group_members=non_tpu_members,
                numa_node=numa_node,
                pcie_root=pcie_root,
                vfio_mode=chip_vfio_mode,
                iommufd_cdev=iommufd_cdev,
                runtime_chip_id=_parse_runtime_chip_id(entry),
            )
        )

    if not chips:
        raise RuntimeError(
            f"No known Google TPU PCI devices (vendor={GOOGLE_PCI_VENDOR_ID}) found under {pci_dir}"
        )

    return chips, control_node


def _pci_bdf_token(pci_bdf: str) -> str:
    """Convert a PCI BDF (e.g. 0000:00:06.0) into a CDI-safe hardware token."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", pci_bdf.strip()).strip("-")


def _single_chip_selector(pci_bdf: str) -> str:
    return f"pci-{_pci_bdf_token(pci_bdf)}"


def _shared_group_selector(pci_bdfs: Sequence[str]) -> str:
    tokens = [_pci_bdf_token(b) for b in sorted(pci_bdfs)]
    return f"group-pci-{'_'.join(tokens)}"


def _stat_char_devnum(path: Path) -> Optional[str]:
    """Return '<major>:<minor>' if path itself (not a symlink) is a character device, else None."""
    try:
        st = path.lstat()
    except OSError:
        return None
    if stat.S_ISCHR(st.st_mode):
        return f"{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}"
    return None


def _format_vfio_devnums(entries: Sequence[Tuple[str, Optional[str]]]) -> Optional[str]:
    pairs = sorted((container_path, dn) for container_path, dn in entries if dn)
    if not pairs:
        return None
    return ",".join(f"{cp}={dn}" for cp, dn in pairs)


def _map_physical_chip_indices(
    chips: Sequence[DiscoveredChip],
    *,
    chip_id_map: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Map each discovered PCI BDF to its physical TPU chip ordinal (`0..N-1`).

    Generations with 1 PCI function per physical chip (`v5e`, `v5p`, `v6e`, etc.)
    map 1:1. Dual-function/chiplet generations (`tpu7x` Ironwood `0x0075`/`0x0076`
    and legacy `v2-v3`) pair two PCI functions into one physical chip ONLY when an
    explicit verified sibling relationship exists:
    - multifunction siblings sharing the exact same PCI slot `DDDD:BB:DD`
      (`slot_count == per_chip`), OR
    - multiple functions explicitly mapped to the SAME physical chip ID from
      sysfs `chip_id` or `--chip-id-map` within the same topology domain
      (`rid_count == per_chip`).

    Distinct unique IDs are NEVER paired by integer division (since one surviving
    function per chip across 4 chips can expose unique physical IDs `0,1,2,3`),
    and unmatched functions are left as incomplete singletons so degraded chips
    cannot form bogus multi-chip slices.
    """
    bdf_to_phys: Dict[str, int] = {}
    slot_counts: Dict[Tuple[str, str], int] = {}
    for c in chips:
        base_slot = c.pci_bdf.rsplit(".", 1)[0]
        key = (c.generation, base_slot)
        slot_counts[key] = slot_counts.get(key, 0) + 1

    # Identify functions explicitly sharing the SAME physical chip ID within one
    # topology domain (never pairing distinct unique IDs via integer division).
    # Preserve raw sysfs `chip_id` for physical sibling pairing so passing a
    # logical per-function `--chip-id-map` (e.g. `0..7` for visibility masks on
    # Ironwood where siblings share physical sysfs `chip_id` `0..3`) never masks
    # the sysfs sibling relationship. Fall back to `chip_id_map` only when sysfs
    # `chip_id` is absent.
    shared_sysfs_counts: Dict[Tuple[str, int], int] = {}
    shared_sysfs_domains: Dict[Tuple[str, int], set[Tuple[str, Optional[int], str]]] = {}
    shared_map_counts: Dict[Tuple[str, int], int] = {}
    shared_map_domains: Dict[Tuple[str, int], set[Tuple[str, Optional[int], str]]] = {}
    for c in chips:
        per_chip = _PCI_FUNCTIONS_PER_PHYSICAL_CHIP.get(c.generation, 1)
        if per_chip <= 1:
            continue
        if c.runtime_chip_id is not None:
            rkey = (c.generation, int(c.runtime_chip_id))
            shared_sysfs_counts[rkey] = shared_sysfs_counts.get(rkey, 0) + 1
            shared_sysfs_domains.setdefault(rkey, set()).add(_topology_domain_key(c))
        elif chip_id_map and c.pci_bdf in chip_id_map:
            mkey = (c.generation, int(chip_id_map[c.pci_bdf]))
            shared_map_counts[mkey] = shared_map_counts.get(mkey, 0) + 1
            shared_map_domains.setdefault(mkey, set()).add(_topology_domain_key(c))

    slot_to_phys: Dict[Tuple[str, str], int] = {}
    shared_rid_to_phys: Dict[Tuple[str, str, Tuple[str, Optional[int], str], int], int] = {}
    next_phys = 0

    for c in chips:
        per_chip = _PCI_FUNCTIONS_PER_PHYSICAL_CHIP.get(c.generation, 1)
        if per_chip <= 1:
            bdf_to_phys[c.pci_bdf] = next_phys
            next_phys += 1
            continue

        base_slot = c.pci_bdf.rsplit(".", 1)[0]
        slot_key = (c.generation, base_slot)
        if slot_counts.get(slot_key, 0) == per_chip:
            if slot_key not in slot_to_phys:
                slot_to_phys[slot_key] = next_phys
                next_phys += 1
            bdf_to_phys[c.pci_bdf] = slot_to_phys[slot_key]
            continue

        if c.runtime_chip_id is not None:
            rkey = (c.generation, int(c.runtime_chip_id))
            if (
                shared_sysfs_counts.get(rkey, 0) == per_chip
                and len(shared_sysfs_domains.get(rkey, set())) == 1
            ):
                vkey = ("sysfs", c.generation, _topology_domain_key(c), int(c.runtime_chip_id))
                if vkey not in shared_rid_to_phys:
                    shared_rid_to_phys[vkey] = next_phys
                    next_phys += 1
                bdf_to_phys[c.pci_bdf] = shared_rid_to_phys[vkey]
                continue
        elif chip_id_map and c.pci_bdf in chip_id_map:
            mkey = (c.generation, int(chip_id_map[c.pci_bdf]))
            if (
                shared_map_counts.get(mkey, 0) == per_chip
                and len(shared_map_domains.get(mkey, set())) == 1
            ):
                vkey = ("map", c.generation, _topology_domain_key(c), int(chip_id_map[c.pci_bdf]))
                if vkey not in shared_rid_to_phys:
                    shared_rid_to_phys[vkey] = next_phys
                    next_phys += 1
                bdf_to_phys[c.pci_bdf] = shared_rid_to_phys[vkey]
                continue

        # Unmatched or single-function chiplet: assign its own singleton physical
        # ID so `_has_complete_chip_functions` marks it incomplete.
        bdf_to_phys[c.pci_bdf] = next_phys
        next_phys += 1

    return bdf_to_phys


def _distinct_physical_indices(
    subset: Sequence[DiscoveredChip],
    bdf_to_phys: Dict[str, int],
) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for c in subset:
        idx = bdf_to_phys[c.pci_bdf]
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


# Supported single-host Cloud TPU physical chip topologies per generation.
# Per Google Cloud TPU docs (`tpu-machines#tpu_topology`):
# - `v6e` and `v5e` single-VM hosts support `{1, 4, 8}` physical chips (NOT 2,
#   because PCIe/NUMA locality alone cannot verify that two surviving chips form
#   a contiguous `1,2,1` ICI mesh rather than a 2x2 diagonal).
# - `v5p`, `v4`, and `tpu7x` (Ironwood) VMs are strictly 4-chip-per-VM hosts
#   (`v5p-8`, `v4-8`, `tpu7x-8`), so degraded 1-, 2-, or 3-chip inventory can
#   still be published without injecting unsupported `tpu7x-2`/`v5p-2` or
#   `1,1,1` isolated runtime overrides.
# Sub-host slice selectors (`numa*`, `pcie-*`, range aliases) are emitted ONLY
# when a domain's live physical chip count is supported for its generation and
# every chip has its full PCI function set.
_SUPPORTED_SINGLE_HOST_CHIP_COUNTS_BY_GEN: Dict[str, frozenset[int]] = {
    "v6e": frozenset({1, 4, 8}),
    "v6e-pf": frozenset({1, 4, 8}),
    "v5e": frozenset({1, 4, 8}),
    "v5p": frozenset({4}),
    "v4": frozenset({4}),
    "tpu7x": frozenset({4}),
}
_DEFAULT_SUPPORTED_SINGLE_HOST_CHIP_COUNTS: frozenset[int] = frozenset()


def _is_supported_single_host_chip_count(
    physical_chip_count: int,
    generations: Sequence[str],
    *,
    subhost_slice: bool = False,
) -> bool:
    """Return True iff `physical_chip_count` is a supported single-host topology for `generations`."""
    if subhost_slice and physical_chip_count < 2:
        return False
    norm_gens = {"v6e" if g == "v6e-pf" else g for g in generations if g}
    if len(norm_gens) != 1:
        return False
    gen = next(iter(norm_gens))
    allowed = _SUPPORTED_SINGLE_HOST_CHIP_COUNTS_BY_GEN.get(
        gen, _DEFAULT_SUPPORTED_SINGLE_HOST_CHIP_COUNTS
    )
    return physical_chip_count in allowed


def _is_supported_all_topology(
    chips: Sequence[DiscoveredChip],
    bdf_to_phys: Dict[str, int],
    generations: Sequence[str],
) -> bool:
    """Return True iff the live `all` chip set forms a verified single-host topology.

    Prevents injecting single-host bounds/accelerator-type overrides (such as
    `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1` / `TPU_ACCELERATOR_TYPE=v6e-4`) onto a
    degraded cross-root set (e.g. a v6e-8 degraded to 3+1 or 2+2 chips across
    its two NUMA roots, where a supported v6e-4 2x2 slice must sit within one
    NUMA/PCIe root domain).
    """
    if not chips or not _has_complete_chip_functions(chips, bdf_to_phys):
        return False
    n_chips = len(_distinct_physical_indices(chips, bdf_to_phys))
    if not _is_supported_single_host_chip_count(n_chips, generations):
        return False

    domains: Dict[Tuple[str, Optional[int], str], List[DiscoveredChip]] = {}
    for c in chips:
        domains.setdefault(_topology_domain_key(c), []).append(c)
    domain_phys_counts = sorted(
        len(_distinct_physical_indices(d_chips, bdf_to_phys))
        for d_chips in domains.values()
    )

    norm_gens = {"v6e" if g == "v6e-pf" else g for g in generations if g}
    gen = next(iter(norm_gens))

    if n_chips == 1:
        return domain_phys_counts == [1]
    if n_chips == 2:
        return domain_phys_counts == [2]
    if n_chips == 4:
        if gen in ("tpu7x", "v5p"):
            return domain_phys_counts in ([4], [2, 2])
        return domain_phys_counts == [4]
    if n_chips == 8:
        return domain_phys_counts in ([8], [4, 4])
    return False


def _count_complete_chips(
    subset: Sequence[DiscoveredChip],
    bdf_to_phys: Dict[str, int],
) -> Tuple[int, int]:
    """Return `(complete_chip_count, incomplete_pci_function_count)` for `subset`.

    Only physical chips exposing their full expected PCI function set (e.g.
    both chiplets on `tpu7x` Ironwood or legacy `v2-v3`) are counted in
    `complete_chip_count`. Unpaired/orphaned functions are excluded from chip
    capacity and tallied in `incomplete_pci_function_count`.
    """
    counts_by_phys: Dict[int, Tuple[int, str]] = {}
    for c in subset:
        phys = bdf_to_phys[c.pci_bdf]
        cnt, gen = counts_by_phys.get(phys, (0, c.generation))
        counts_by_phys[phys] = (cnt + 1, gen)
    complete_chips = 0
    incomplete_fns = 0
    for cnt, gen in counts_by_phys.values():
        expected = _PCI_FUNCTIONS_PER_PHYSICAL_CHIP.get(gen, 1)
        if cnt == expected:
            complete_chips += 1
        else:
            incomplete_fns += cnt
    return complete_chips, incomplete_fns


def _has_complete_chip_functions(
    subset: Sequence[DiscoveredChip],
    bdf_to_phys: Dict[str, int],
) -> bool:
    """Return True when every physical chip in `subset` exposes its full expected PCI function set."""
    if not subset:
        return False
    complete_chips, incomplete_fns = _count_complete_chips(subset, bdf_to_phys)
    return complete_chips > 0 and incomplete_fns == 0


def _resolve_verified_visibility_ordinals(
    subset: Sequence[DiscoveredChip],
    all_chips: Sequence[DiscoveredChip],
    bdf_to_phys: Dict[str, int],
    *,
    chip_id_map: Optional[Dict[str, int]] = None,
) -> Optional[Tuple[List[int], List[int]]]:
    """Return `(visible_chip_ordinals, visible_device_ordinals)` ONLY when verified.

    - Requires every physical chip in `subset` to have its complete PCI function set.
    - Requires authoritative runtime ordinals from sysfs `chip_id` or explicit
      `chip_id_map` for EVERY chip in `subset` (checking uniqueness across all
      host chips so conflicting or incomplete maps fail closed).
    - Never assumes runtime ordinals are `0..N-1` (even on `all`, since Cloud TPU
      hosts can expose non-contiguous runtime IDs such as `8,9,12,13`). When
      verified ordinals are unavailable, returns `None` so CDI omits
      `TPU_VISIBLE_CHIPS` and `TPU_VISIBLE_DEVICES` and preserves workload-provided
      visibility settings.
    """
    if not subset or not _has_complete_chip_functions(subset, bdf_to_phys):
        return None

    def _raw_id(chip: DiscoveredChip) -> Optional[int]:
        if chip_id_map and chip.pci_bdf in chip_id_map:
            return chip_id_map[chip.pci_bdf]
        return chip.runtime_chip_id

    subset_raw: List[Optional[int]] = [_raw_id(c) for c in subset]
    if all(rid is not None for rid in subset_raw):
        per_chip_set = {
            _PCI_FUNCTIONS_PER_PHYSICAL_CHIP.get(c.generation, 1) for c in subset
        }
        if len(per_chip_set) == 1:
            per_chip = next(iter(per_chip_set))
            all_raw_by_bdf: Dict[str, int] = {}
            for c in all_chips:
                rid = _raw_id(c)
                if rid is not None:
                    all_raw_by_bdf[c.pci_bdf] = rid

            if per_chip == 1:
                # Every claimed runtime ID across the host must be unique.
                if len(set(all_raw_by_bdf.values())) == len(all_raw_by_bdf):
                    ords = sorted(int(rid) for rid in subset_raw if rid is not None)
                    return ords, list(ords)
                return None

            # Dual-function/chiplet generations (e.g. Ironwood per_chip=2):
            # Case A: distinct logical device IDs per PCI function (e.g. 4,5,6,7 -> chips 2,3).
            sub_ids = [int(rid) for rid in subset_raw if rid is not None]
            if len(set(sub_ids)) == len(sub_ids) and len(set(all_raw_by_bdf.values())) == len(
                all_raw_by_bdf
            ):
                phys_set = sorted({oid // per_chip for oid in sub_ids})
                expected_logical = [
                    p * per_chip + off for p in phys_set for off in range(per_chip)
                ]
                if sorted(sub_ids) == expected_logical:
                    return phys_set, expected_logical
                return None

            # Case B: physical chip ID shared by both PCI functions of each chip.
            phys_to_rids: Dict[int, set[int]] = {}
            for c in all_chips:
                rid = all_raw_by_bdf.get(c.pci_bdf)
                if rid is not None:
                    phys_to_rids.setdefault(bdf_to_phys[c.pci_bdf], set()).add(rid)
            if all(len(rids) == 1 for rids in phys_to_rids.values()):
                claimed_phys_rids = [next(iter(rids)) for rids in phys_to_rids.values()]
                if len(set(claimed_phys_rids)) == len(claimed_phys_rids):
                    sub_phys = sorted(
                        {
                            next(iter(phys_to_rids[bdf_to_phys[c.pci_bdf]]))
                            for c in subset
                        }
                    )
                    logical = [
                        p * per_chip + off for p in sub_phys for off in range(per_chip)
                    ]
                    return sub_phys, logical
            return None

    return None


def _chips_per_host_bounds(chip_count: int, generations: Sequence[str]) -> str:
    """Return `TPU_CHIPS_PER_HOST_BOUNDS` (`X,Y,Z`) matching physical chip count and generation."""
    if chip_count <= 1:
        return "1,1,1"
    if chip_count == 2:
        return "1,2,1"
    if chip_count == 4:
        return "2,2,1"
    if chip_count == 8:
        if any(g in ("v4", "v4lite", "v5p", "tpu7x") for g in generations):
            return "2,2,2"
        return "2,4,1"
    return f"1,{chip_count},1"


def _logical_device_indices(
    subset: Sequence[DiscoveredChip],
    physical_chip_indices: Sequence[int],
) -> List[int]:
    """Return logical runtime device ordinals corresponding to `subset` / `physical_chip_indices`.

    On 1-device-per-chip generations (`v5e`, `v5p`, `v6e`), logical device IDs
    equal physical chip ordinals. On 2-device-per-chip generations (`tpu7x`
    Ironwood, `v2-v3`), physical chip `c` exposes logical devices `2*c` and
    `2*c + 1` (e.g. physical chips `2,3` -> logical devices `4,5,6,7`).
    """
    per_chip_set = {
        _PCI_FUNCTIONS_PER_PHYSICAL_CHIP.get(c.generation, 1) for c in subset
    }
    if len(per_chip_set) == 1:
        per_chip = next(iter(per_chip_set))
        if per_chip > 1:
            return [
                phys_idx * per_chip + offset
                for phys_idx in physical_chip_indices
                for offset in range(per_chip)
            ]
    return [c.index for c in subset]


def _tpu_accelerator_type(
    physical_chip_count: int,
    generations: Sequence[str],
) -> Optional[str]:
    """Return Google Cloud / PyTorch-XLA `TPU_ACCELERATOR_TYPE` for a supported single-host slice.

    Matches PyTorch/XLA `torch_xla/_internal/tpu.py` (`^(?:v|tpu)(\\d)([A-Za-z]?){7}-(\\d+)$`):
    - `v6e` / `v6e-pf` -> `v6e-<chips>` (e.g. `v6e-4`, `v6e-8`)
    - `v5e` -> `v5litepod-<chips>` (e.g. `v5litepod-1`, `v5litepod-4`, `v5litepod-8`)
    - `v5p` -> `v5p-<cores>` (`chips * 2`, e.g. `v5p-8`)
    - `v4` -> `v4-<cores>` (`chips * 2`, e.g. `v4-8`)
    - `tpu7x` (Ironwood) -> `tpu7x-<cores>` (`chips * 2`, e.g. `tpu7x-8`)
    """
    if not _is_supported_single_host_chip_count(physical_chip_count, generations):
        return None
    norm_gens = {"v6e" if g == "v6e-pf" else g for g in generations if g}
    if len(norm_gens) != 1:
        return None
    gen = next(iter(norm_gens))
    if gen == "v6e":
        return f"v6e-{physical_chip_count}"
    if gen == "v5e":
        return f"v5litepod-{physical_chip_count}"
    if gen == "v5p":
        return f"v5p-{physical_chip_count * 2}"
    if gen == "v4":
        return f"v4-{physical_chip_count * 2}"
    if gen == "tpu7x":
        return f"tpu7x-{physical_chip_count * 2}"
    return None


def _build_device_env(
    physical_chip_count: int,
    generations: Sequence[str],
    *,
    single_host_env: bool = False,
    include_bounds_env: bool = True,
    verified_visibility: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    omit_visible_env: bool = False,
) -> List[str]:
    """Build per-device CDI `containerEdits.env` entries.

    By default (`single_host_env=False`), returns an empty list so CDI never
    overwrites workload-configured multi-host peer discovery, visibility, or
    host bounds. When `single_host_env=True` is explicitly opted in:
    - All selectors receive isolated loopback Megascale / single-host overrides.
    - Full-allocation / slice selectors (`numa*`, `pcie*`, `0-3`/`4-7`, `all`)
      with supported complete chip counts receive `TPU_CHIPS_PER_HOST_BOUNDS`
      and `TPU_ACCELERATOR_TYPE` (required by PyTorch/XLA when metadata
      discovery is disabled via `TPU_SKIP_MDS_QUERY=true`).
    - `TPU_VISIBLE_CHIPS` and `TPU_VISIBLE_DEVICES` are emitted ONLY when
      `verified_visibility` supplies verified runtime ordinals (from sysfs
      `chip_id` or explicit `--chip-id-map`) AND `omit_visible_env` is False.
      Pass `omit_visible_env=True` (`--omit-visible-env`) for multi-process
      launchers such as PyTorch/XLA `xmp.spawn()` that assign per-rank chip
      visibility via `os.environ.setdefault`.
    - Independently composable unit selectors (`pci-*`, `group-pci-*`) omit
      visibility, host bounds, and accelerator type so combining multiple
      selectors never overwrites OCI env settings.
    """
    if not single_host_env:
        return []

    env: List[str] = []
    has_complete_topology = False
    if (
        include_bounds_env
        and _is_supported_single_host_chip_count(physical_chip_count, generations)
    ):
        if verified_visibility is not None and not omit_visible_env:
            vis_chips, vis_devices = verified_visibility
            env.append("TPU_VISIBLE_CHIPS=" + ",".join(str(idx) for idx in vis_chips))
            env.append("TPU_VISIBLE_DEVICES=" + ",".join(str(idx) for idx in vis_devices))
        bounds = _chips_per_host_bounds(physical_chip_count, generations)
        env.append(f"TPU_CHIPS_PER_HOST_BOUNDS={bounds}")
        accel_type = _tpu_accelerator_type(physical_chip_count, generations)
        if accel_type is not None:
            env.append(f"TPU_ACCELERATOR_TYPE={accel_type}")
            has_complete_topology = True
    env.extend(
        [
            "MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353",
            "TPU_HOST_BOUNDS=1,1,1",
            "TPU_WORKER_HOSTNAMES=localhost",
            "TPU_WORKER_ID=0",
        ]
    )
    # Gate `TPU_SKIP_MDS_QUERY=true` strictly on complete topology injection so
    # composable `pci-*`/`group-pci-*` selectors and degraded `all` selectors
    # that omit `TPU_ACCELERATOR_TYPE`/`TPU_CHIPS_PER_HOST_BOUNDS` do not break
    # PyTorch/XLA metadata discovery.
    if has_complete_topology:
        env.append("TPU_SKIP_MDS_QUERY=true")
    return env


@dataclass(frozen=True)
class _TopologySlice:
    chips: Tuple[DiscoveredChip, ...]
    pci_bdfs: Tuple[str, ...]
    physical_indices: Tuple[int, ...]
    logical_indices: Tuple[int, ...]
    generations: Tuple[str, ...]
    numa_node: Optional[int]
    pcie_root: str
    pcie_domain: int
    allocation_unit: str
    aliases: Tuple[str, ...]


def _compute_topology_slices(
    chips: Sequence[DiscoveredChip],
    bdf_to_phys: Dict[str, int],
    *,
    reserved_names: Optional[set[str]] = None,
    chip_id_map: Optional[Dict[str, int]] = None,
) -> List[_TopologySlice]:
    """Compute valid single-domain NUMA / PCIe sub-host slices and their canonical aliases.

    Numeric range aliases (e.g. `0-3`, `4-7`) are emitted ONLY from verified
    stable runtime chip IDs (sysfs `chip_id` or explicit `--chip-id-map`) so
    losing a chip in one NUMA domain never renumbers a healthy sibling slice
    (e.g. turning `4-7` into `3-6`). When verified chip IDs are unavailable,
    only the topology selector (`numa*` or `pcie*`) is emitted.
    """
    domains: Dict[Tuple[str, Optional[int], str], List[DiscoveredChip]] = {}
    for chip in chips:
        domains.setdefault(_topology_domain_key(chip), []).append(chip)

    if not domains:
        return []

    used_names = set(reserved_names or ())

    # Map every IOMMU group to the set of domain keys that touch it so we
    # never emit a sub-host slice that splits an IOMMU group across domains
    # (enforced in BOTH legacy VFIO and iommufd/cdev modes).
    group_to_domains: Dict[str, set[Tuple[str, Optional[int], str]]] = {}
    for d_key, d_chips in domains.items():
        for c in d_chips:
            group_to_domains.setdefault(c.iommu_group, set()).add(d_key)

    slices: List[_TopologySlice] = []
    for dom_ordinal, (d_key, d_chips) in enumerate(domains.items()):
        if len(d_chips) < 2:
            continue
        if any(
            len(group_to_domains.get(c.iommu_group, set())) > 1 for c in d_chips
        ):
            continue

        slice_bdfs = tuple(sorted(c.pci_bdf for c in d_chips))
        slice_phys_indices = tuple(_distinct_physical_indices(d_chips, bdf_to_phys))
        slice_gens: List[str] = []
        for c in d_chips:
            if c.generation not in slice_gens:
                slice_gens.append(c.generation)
        if (
            not _is_supported_single_host_chip_count(
                len(slice_phys_indices), slice_gens, subhost_slice=True
            )
            or not _has_complete_chip_functions(d_chips, bdf_to_phys)
        ):
            continue
        slice_logical_indices = tuple(_logical_device_indices(d_chips, slice_phys_indices))

        discovered_numa = d_key[1]
        pcie_root = d_key[2]
        slice_chip_count = len(slice_phys_indices)
        # Keep every slice selector permanently bound to its exact hardware
        # domain regardless of what other roots or NUMA nodes disappear or
        # return across refreshes:
        # - When sysfs NUMA locality is known, publish ONLY the domain-qualified
        #   `numa<node>-pcie-<root>-<chips>` selector (never scan-dependent
        #   `numa<node>-<chips>` or `pcie-<root>-<chips>` shortcuts that could
        #   rebind to a different NUMA node or PCIe root after an outage).
        # - When NUMA locality is absent (`None`), publish `pcie-<root>-<chips>`.
        pcie_root_token = _pci_bdf_token(pcie_root)
        candidate_names: List[str] = []
        if discovered_numa is not None:
            candidate_names.append(
                f"numa{discovered_numa}-pcie-{pcie_root_token}-{slice_chip_count}"
            )
        else:
            candidate_names.append(f"pcie-{pcie_root_token}-{slice_chip_count}")
        verified_vis = _resolve_verified_visibility_ordinals(
            d_chips,
            chips,
            bdf_to_phys,
            chip_id_map=chip_id_map,
        )
        if verified_vis is not None:
            vis_chips, _vis_devs = verified_vis
            if vis_chips and vis_chips == list(range(vis_chips[0], vis_chips[0] + len(vis_chips))):
                candidate_names.append(f"{vis_chips[0]}-{vis_chips[-1]}")

        aliases: List[str] = []
        for s_name in candidate_names:
            if s_name in used_names or s_name == "all":
                continue
            used_names.add(s_name)
            aliases.append(s_name)

        if aliases:
            slices.append(
                _TopologySlice(
                    chips=tuple(d_chips),
                    pci_bdfs=slice_bdfs,
                    physical_indices=slice_phys_indices,
                    logical_indices=slice_logical_indices,
                    generations=tuple(slice_gens),
                    numa_node=discovered_numa,
                    pcie_root=pcie_root,
                    pcie_domain=dom_ordinal,
                    allocation_unit="numa-slice" if discovered_numa is not None else "pcie-slice",
                    aliases=tuple(aliases),
                )
            )
    return slices


def _dedup_device_nodes(chips: Sequence[DiscoveredChip]) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    nodes: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    devnums: Dict[str, Optional[str]] = {}
    for chip in chips:
        cp = chip.vfio_container_path
        if cp not in seen_paths:
            seen_paths.add(cp)
            nodes.append(_make_device_node(cp, chip.vfio_host_path))
            devnums[cp] = _stat_char_devnum(chip.vfio_host_path)
    return nodes, devnums


def _topology_domain_key(chip: DiscoveredChip) -> Tuple[str, Optional[int], str]:
    """Group chips strictly within matching NUMA node and PCIe root complex boundaries."""
    root = chip.pcie_root or _pcie_root_complex(chip.pci_bdf)
    if chip.numa_node is not None and chip.numa_node >= 0:
        return ("numa", chip.numa_node, root)
    return ("pcie", None, root)


def build_cdi_spec(
    chips: Sequence[DiscoveredChip],
    vfio_container_host_path: Path,
    *,
    kind: str = CDI_KIND,
    network_mode: str = "host",
    single_host_env: bool = False,
    cdi_version: str = DEFAULT_CDI_VERSION,
    chip_id_map: Optional[Dict[str, int]] = None,
    omit_visible_env: bool = False,
) -> Dict[str, Any]:
    """Build a CNCF CDI JSON specification dictionary (default: CDI v0.6.0, host-safe env).

    By default (`network_mode="host"`, `single_host_env=False`), no `libtpu`
    environment overrides are injected so existing multi-host workloads retain
    their configured peer discovery, visibility, and host bounds. Pass
    `single_host_env=True` (or `network_mode="isolated"`) to explicitly opt
    into single-host isolated container environment injection.
    """
    if not isinstance(cdi_version, str) or cdi_version not in SUPPORTED_CDI_VERSIONS:
        raise ValueError(
            f"Unsupported CDI version {cdi_version!r}: expected one of {list(SUPPORTED_CDI_VERSIONS)}"
        )
    if not isinstance(kind, str) or not _CDI_KIND_RE.match(kind):
        raise ValueError(
            f"Invalid CDI kind {kind!r}: vendor and class must each start with an ASCII letter, "
            "contain only [A-Za-z0-9_.-], and end with an ASCII letter or digit"
        )
    if network_mode not in NETWORK_MODES:
        raise ValueError(
            f"Invalid network_mode {network_mode!r}: expected one of {sorted(NETWORK_MODES)}"
        )
    if not chips:
        raise ValueError("Cannot build CDI spec with zero discovered TPU chips")

    for chip in chips:
        if chip.non_tpu_group_members:
            raise ValueError(
                f"Refusing VFIO group {chip.iommu_group} for TPU {chip.pci_bdf}: "
                f"IOMMU group contains non-TPU device(s): {', '.join(chip.non_tpu_group_members)}"
            )

    inject_single_host_env = bool(single_host_env or network_mode == "isolated")
    effective_network_mode = "isolated" if inject_single_host_env else "host"

    is_iommufd = any(c.vfio_mode == "iommufd" for c in chips)
    control_container_path = "/dev/iommu" if is_iommufd else "/dev/vfio/vfio"
    container_devnum = _stat_char_devnum(vfio_container_host_path)
    global_edits: Dict[str, Any] = {
        "deviceNodes": [
            _make_device_node(control_container_path, vfio_container_host_path),
        ],
        "env": [
            f"FREETPU_CDI_VERSION={cdi_version}",
            f"FREETPU_IOMMU_MODE={'iommufd' if is_iommufd else 'vfio-group'}",
            f"FREETPU_NETWORK_MODE={effective_network_mode}",
            f"FREETPU_VFIO_CONTAINER={control_container_path}",
        ],
    }

    bdf_to_phys = _map_physical_chip_indices(chips, chip_id_map=chip_id_map)

    devices: List[Dict[str, Any]] = []
    seen_group_nodes: Dict[str, Dict[str, Any]] = {}
    seen_group_devnums: Dict[str, Optional[str]] = {}
    units: Dict[str, List[DiscoveredChip]] = {}
    all_bdfs: List[str] = []
    all_generations: List[str] = []

    # Group sibling chips sharing an IOMMU group into ONE allocation unit in BOTH
    # legacy VFIO and iommufd/cdev modes (Linux VFIO permits only one DMA owner
    # per IOMMU group regardless of whether group or cdev nodes are used).
    for chip in chips:
        units.setdefault(chip.iommu_group, []).append(chip)
        all_bdfs.append(chip.pci_bdf)
        if chip.generation not in all_generations:
            all_generations.append(chip.generation)

    for _unit_key, unit_chips in units.items():
        rep = unit_chips[0]
        unit_nodes, unit_devnums = _dedup_device_nodes(unit_chips)
        for node in unit_nodes:
            seen_group_nodes.setdefault(node["path"], node)
        for cp, dn in unit_devnums.items():
            seen_group_devnums.setdefault(cp, dn)
        unit_bdfs = sorted(c.pci_bdf for c in unit_chips)
        unit_gens: List[str] = []
        for c in unit_chips:
            if c.generation not in unit_gens:
                unit_gens.append(c.generation)
        unit_phys_indices = _distinct_physical_indices(unit_chips, bdf_to_phys)
        unit_env = _build_device_env(
            len(unit_phys_indices),
            unit_gens,
            single_host_env=inject_single_host_env,
            include_bounds_env=False,
            verified_visibility=None,
        )
        unit_edits: Dict[str, Any] = {"deviceNodes": [dict(n) for n in unit_nodes]}
        if unit_env:
            unit_edits["env"] = unit_env
        pcie_root = rep.pcie_root or _pcie_root_complex(rep.pci_bdf)
        devnums_str = _format_vfio_devnums(list(unit_devnums.items()))

        if len(unit_chips) == 1:
            selector = _single_chip_selector(rep.pci_bdf)
            per_chip = _PCI_FUNCTIONS_PER_PHYSICAL_CHIP.get(rep.generation, 1)
            unit_label = "single-chiplet" if per_chip > 1 else "single-chip"
            ann: Dict[str, str] = {
                "freetpu.dev/allocation-unit": unit_label,
                "freetpu.dev/pci-bdf": rep.pci_bdf,
                "freetpu.dev/vendor-id": rep.vendor_id,
                "freetpu.dev/device-id": rep.device_id,
                "freetpu.dev/generation": rep.generation,
                "freetpu.dev/model": rep.model_name,
                "freetpu.dev/driver": rep.driver,
                "freetpu.dev/iommu-group": rep.iommu_group,
                "freetpu.dev/pcie-root": pcie_root,
                "freetpu.dev/optional-cap-add": "IPC_LOCK",
                "freetpu.dev/required-ulimit-memlock": "-1",
            }
            if rep.numa_node is not None:
                ann["freetpu.dev/numa-node"] = str(rep.numa_node)
            if devnums_str:
                ann["freetpu.dev/vfio-devnums"] = devnums_str
            devices.append(
                {
                    "name": selector,
                    "annotations": ann,
                    "containerEdits": unit_edits,
                }
            )
        else:
            # Expose shared IOMMU groups strictly as a single hardware-backed
            # allocation unit so callers never assume device-level VFIO isolation
            # between sibling chips and selectors never shift onto different chips.
            unit_complete_chips, unit_incomplete_fns = _count_complete_chips(
                unit_chips, bdf_to_phys
            )
            selector = _shared_group_selector(unit_bdfs)
            ann = {
                "freetpu.dev/allocation-unit": "shared-iommu-group",
                "freetpu.dev/shared-iommu-group": "true",
                "freetpu.dev/chip-count": str(unit_complete_chips),
                "freetpu.dev/pci-bdfs": ",".join(unit_bdfs),
                "freetpu.dev/generations": ",".join(unit_gens),
                "freetpu.dev/iommu-group": rep.iommu_group,
                "freetpu.dev/pcie-root": pcie_root,
                "freetpu.dev/optional-cap-add": "IPC_LOCK",
                "freetpu.dev/required-ulimit-memlock": "-1",
            }
            if len(unit_chips) != unit_complete_chips or unit_incomplete_fns > 0:
                ann["freetpu.dev/pci-function-count"] = str(len(unit_chips))
            if unit_incomplete_fns > 0:
                ann["freetpu.dev/incomplete-pci-functions"] = str(unit_incomplete_fns)
                ann["freetpu.dev/capacity-complete"] = "false"
            if rep.numa_node is not None:
                ann["freetpu.dev/numa-node"] = str(rep.numa_node)
            if devnums_str:
                ann["freetpu.dev/vfio-devnums"] = devnums_str
            devices.append(
                {
                    "name": selector,
                    "annotations": ann,
                    "containerEdits": unit_edits,
                }
            )

    # Emit NUMA / PCIe-root-complex-aware capacity-pinned sub-host slice presets
    # (e.g. `numa0-pcie-0000-00-4`, `numa1-pcie-0000-80-4`, `0-3`, `4-7` on `v6e-8`)
    # strictly within matching NUMA / PCIe root complex boundaries.
    used_names = {d["name"] for d in devices}
    for topo_slice in _compute_topology_slices(
        chips, bdf_to_phys, reserved_names=used_names, chip_id_map=chip_id_map
    ):
        slice_nodes, slice_devnums = _dedup_device_nodes(topo_slice.chips)
        slice_visibility = _resolve_verified_visibility_ordinals(
            topo_slice.chips,
            chips,
            bdf_to_phys,
            chip_id_map=chip_id_map,
        )
        slice_env = _build_device_env(
            len(topo_slice.physical_indices),
            topo_slice.generations,
            single_host_env=inject_single_host_env,
            include_bounds_env=True,
            verified_visibility=slice_visibility,
            omit_visible_env=omit_visible_env,
        )
        slice_ann_base: Dict[str, str] = {
            "freetpu.dev/allocation-unit": topo_slice.allocation_unit,
            "freetpu.dev/chip-count": str(len(topo_slice.physical_indices)),
            "freetpu.dev/pci-bdfs": ",".join(topo_slice.pci_bdfs),
            "freetpu.dev/generations": ",".join(topo_slice.generations),
            "freetpu.dev/pcie-root": topo_slice.pcie_root,
            "freetpu.dev/pcie-domain": str(topo_slice.pcie_domain),
            "freetpu.dev/optional-cap-add": "IPC_LOCK",
            "freetpu.dev/required-ulimit-memlock": "-1",
        }
        # Record freetpu.dev/numa-node ONLY with the actual sysfs NUMA node;
        # never substitute a PCIe domain ordinal or invent a node when sysfs has none.
        if topo_slice.numa_node is not None:
            slice_ann_base["freetpu.dev/numa-node"] = str(topo_slice.numa_node)
        if len(topo_slice.chips) != len(topo_slice.physical_indices):
            slice_ann_base["freetpu.dev/pci-function-count"] = str(len(topo_slice.chips))
        slice_devnums_str = _format_vfio_devnums(list(slice_devnums.items()))
        if slice_devnums_str:
            slice_ann_base["freetpu.dev/vfio-devnums"] = slice_devnums_str

        for s_name in topo_slice.aliases:
            slice_edits: Dict[str, Any] = {
                "deviceNodes": [dict(n) for n in slice_nodes],
            }
            if slice_env:
                slice_edits["env"] = list(slice_env)
            s_ann = dict(slice_ann_base)
            if chip_id_map and _RANGE_SLICE_RE.match(s_name):
                s_ann["freetpu.dev/chip-id-source"] = "chip-id-map"
            devices.append(
                {
                    "name": s_name,
                    "annotations": s_ann,
                    "containerEdits": slice_edits,
                }
            )

    all_complete_chips, all_incomplete_fns = _count_complete_chips(chips, bdf_to_phys)
    all_supported_topology = _is_supported_all_topology(chips, bdf_to_phys, all_generations)
    all_visibility = _resolve_verified_visibility_ordinals(
        chips,
        chips,
        bdf_to_phys,
        chip_id_map=chip_id_map,
    )
    all_ann: Dict[str, str] = {
        "freetpu.dev/allocation-unit": "all-devices",
        "freetpu.dev/chip-count": str(all_complete_chips),
        "freetpu.dev/pci-bdfs": ",".join(sorted(all_bdfs)),
        "freetpu.dev/generations": ",".join(all_generations),
        "freetpu.dev/optional-cap-add": "IPC_LOCK",
        "freetpu.dev/required-ulimit-memlock": "-1",
    }
    if len(chips) != all_complete_chips or all_incomplete_fns > 0:
        all_ann["freetpu.dev/pci-function-count"] = str(len(chips))
    if all_incomplete_fns > 0:
        all_ann["freetpu.dev/incomplete-pci-functions"] = str(all_incomplete_fns)
        all_ann["freetpu.dev/capacity-complete"] = "false"
    all_devnums = _format_vfio_devnums(list(seen_group_devnums.items()))
    if all_devnums:
        all_ann["freetpu.dev/vfio-devnums"] = all_devnums

    all_env = _build_device_env(
        all_complete_chips,
        all_generations,
        single_host_env=inject_single_host_env,
        include_bounds_env=all_supported_topology,
        verified_visibility=all_visibility,
        omit_visible_env=omit_visible_env,
    )
    all_edits: Dict[str, Any] = {"deviceNodes": list(seen_group_nodes.values())}
    if all_env:
        all_edits["env"] = all_env

    devices.append(
        {
            "name": "all",
            "annotations": all_ann,
            "containerEdits": all_edits,
        }
    )

    root_ann: Dict[str, str] = {
        "freetpu.dev/generated-by": "freetpu-cdi",
        "freetpu.dev/iommu-mode": "iommufd" if is_iommufd else "vfio-group",
        "freetpu.dev/network-mode": effective_network_mode,
        "freetpu.dev/optional-cap-add": "IPC_LOCK",
        "freetpu.dev/required-ulimit-memlock": "-1",
    }
    # Record live VFIO/iommufd char major:minor in spec annotations (while keeping
    # deviceNodes free of pinned major/minor) so driver rebinds that change
    # dynamic VFIO minors produce a spec update that invalidates CDI's cache.
    root_devnums = _format_vfio_devnums(
        [(control_container_path, container_devnum), *seen_group_devnums.items()]
    )
    if root_devnums:
        root_ann["freetpu.dev/vfio-devnums"] = root_devnums

    return {
        "cdiVersion": cdi_version,
        "kind": kind,
        "annotations": root_ann,
        "containerEdits": global_edits,
        "devices": devices,
    }


_ALLOWED_SPEC_ROOT_KEYS = frozenset(
    {"cdiVersion", "kind", "annotations", "containerEdits", "devices"}
)
_ALLOWED_DEVICE_ENTRY_KEYS = frozenset({"name", "annotations", "containerEdits"})
_ALLOWED_DEVICE_NODE_KEYS = frozenset(
    {"path", "hostPath", "type", "major", "minor", "fileMode", "permissions", "uid", "gid"}
)
_ALLOWED_MOUNT_KEYS = frozenset({"hostPath", "containerPath", "type", "options"})
_ALLOWED_HOOK_KEYS = frozenset({"hookName", "path", "args", "env", "timeout"})
_HOST_VALIDATED_CONTAINER_EDITS_KEYS = frozenset({"env", "deviceNodes"})
_HOST_VALIDATED_DEVICE_NODE_KEYS = frozenset({"path", "hostPath", "type", "permissions"})


def _load_json_strict_keys(raw: str) -> Any:
    """Parse JSON while rejecting duplicate keys (case-insensitively) within any object.

    Containerd 2.0's Go CDI decoder matches JSON fields case-insensitively; rejecting
    duplicate or case-colliding keys prevents smuggling shadow fields (e.g. `Major`).
    """

    def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        seen_lower: Dict[str, str] = {}
        for k, v in pairs:
            kl = k.lower()
            if kl in seen_lower:
                raise ValueError(
                    f"duplicate or case-colliding JSON key {k!r} (conflicts with {seen_lower[kl]!r})"
                )
            seen_lower[kl] = k
            obj[k] = v
        return obj

    return json.loads(raw, object_pairs_hook=_no_duplicate_keys)


def _allowed_container_edits_keys(cdi_version: str) -> frozenset[str]:
    if cdi_version == "0.6.0":
        return _CDI_CONTAINER_EDITS_KEYS_V06
    if cdi_version in ("0.7.0", "0.8.0", "1.0.0"):
        return _CDI_CONTAINER_EDITS_KEYS_V07
    return _CDI_CONTAINER_EDITS_KEYS_V11


def _allowed_intel_rdt_keys(cdi_version: str) -> frozenset[str]:
    if cdi_version == "1.1.0":
        return _ALLOWED_INTEL_RDT_KEYS_V11
    return _ALLOWED_INTEL_RDT_KEYS_V07


def _validate_container_edits(
    edits: Any,
    context: str,
    errors: List[str],
    *,
    cdi_version: str = DEFAULT_CDI_VERSION,
) -> None:
    if not isinstance(edits, dict):
        errors.append(f"{context}: must be a JSON object")
        return

    allowed_edits_keys = _allowed_container_edits_keys(cdi_version)
    unknown_keys = set(edits.keys()) - allowed_edits_keys
    if unknown_keys:
        errors.append(
            f"{context}: unsupported CDI v{cdi_version} containerEdits keys: {sorted(unknown_keys)}"
        )

    if "env" in edits:
        env = edits["env"]
        if not isinstance(env, list) or not all(
            isinstance(x, str) and "=" in x and bool(x.split("=", 1)[0]) for x in env
        ):
            errors.append(f"{context}.env: must be a list of non-empty KEY=VALUE strings")

    if "additionalGids" in edits and "additionalGids" in allowed_edits_keys:
        gids = edits["additionalGids"]
        if not isinstance(gids, list) or not all(
            isinstance(g, int) and not isinstance(g, bool) and 0 <= g <= 0xFFFFFFFF
            for g in gids
        ):
            errors.append(f"{context}.additionalGids: must be a list of unsigned 32-bit integers")

    if "intelRdt" in edits and "intelRdt" in allowed_edits_keys:
        irdt = edits["intelRdt"]
        if not isinstance(irdt, dict):
            errors.append(f"{context}.intelRdt: must be a JSON object")
        else:
            allowed_irdt_keys = _allowed_intel_rdt_keys(cdi_version)
            unknown_irdt = set(irdt.keys()) - allowed_irdt_keys
            if unknown_irdt:
                errors.append(
                    f"{context}.intelRdt: unsupported CDI v{cdi_version} keys: {sorted(unknown_irdt)}"
                )
            for str_field in ("closID", "l3CacheSchema", "memBwSchema"):
                if str_field in irdt and not isinstance(irdt[str_field], str):
                    errors.append(f"{context}.intelRdt.{str_field}: must be a string")
            if "closID" in irdt and isinstance(irdt["closID"], str):
                clos_id = irdt["closID"]
                # Matches CNCF CDI `IntelRdt.Validate()` (`pkg/cdi/container-edits.go`):
                # `""` (unset) is always permitted; `"/"` (root CLOS) is permitted ONLY
                # in CDI >= 1.1.0 (rejected in 0.7.0, 0.8.0, and 1.0.0); otherwise must
                # be a valid Linux filename (< 4096 bytes, not "." or "..", and no "/",
                # "\n", or "\x00").
                try:
                    cdi_ver_tuple = tuple(int(p) for p in cdi_version.split("."))
                except ValueError:
                    cdi_ver_tuple = (0, 0, 0)
                allow_root_clos = cdi_ver_tuple >= (1, 1, 0)
                if clos_id != "" and not (allow_root_clos and clos_id == "/"):
                    if (
                        len(clos_id.encode("utf-8")) >= 4096
                        or clos_id in (".", "..")
                        or any(ch in clos_id for ch in ("/", "\n", "\x00"))
                    ):
                        root_hint = "'/' for root CLOS or " if allow_root_clos else ""
                        errors.append(
                            f"{context}.intelRdt.closID: invalid closID {clos_id!r} "
                            f"(must be {root_hint}a valid Linux filename without '/', newline, or NUL)"
                        )
            for bool_field in ("enableCMT", "enableMBM", "enableMonitoring"):
                if bool_field in irdt and bool_field in allowed_irdt_keys:
                    if not isinstance(irdt[bool_field], bool):
                        errors.append(f"{context}.intelRdt.{bool_field}: must be a boolean")
            if "schemata" in irdt and "schemata" in allowed_irdt_keys:
                schemata = irdt["schemata"]
                if not isinstance(schemata, list) or not all(
                    isinstance(item, str) for item in schemata
                ):
                    errors.append(f"{context}.intelRdt.schemata: must be a list of strings")

    if "netDevices" in edits and "netDevices" in allowed_edits_keys:
        net_devs = edits["netDevices"]
        if not isinstance(net_devs, list):
            errors.append(f"{context}.netDevices: must be a list")
        else:
            seen_host_ifaces: set[str] = set()
            seen_net_names: set[str] = set()
            for idx, nd in enumerate(net_devs):
                ndc = f"{context}.netDevices[{idx}]"
                if not isinstance(nd, dict):
                    errors.append(f"{ndc}: must be an object")
                    continue
                unknown_nd = set(nd.keys()) - _ALLOWED_NET_DEVICE_KEYS
                if unknown_nd:
                    errors.append(
                        f"{ndc}: unsupported CDI v{cdi_version} netDevice keys: {sorted(unknown_nd)}"
                    )
                host_iface = nd.get("hostInterfaceName")
                if not isinstance(host_iface, str) or not host_iface:
                    errors.append(f"{ndc}.hostInterfaceName: must be a non-empty string")
                elif host_iface in seen_host_ifaces:
                    errors.append(f"{ndc}.hostInterfaceName: duplicate hostInterfaceName {host_iface!r}")
                else:
                    seen_host_ifaces.add(host_iface)
                iface_name = nd.get("name")
                if not isinstance(iface_name, str) or not iface_name:
                    errors.append(f"{ndc}.name: must be a non-empty string")
                elif iface_name in seen_net_names:
                    errors.append(f"{ndc}.name: duplicate netDevice name {iface_name!r}")
                else:
                    seen_net_names.add(iface_name)

    if "deviceNodes" in edits:
        nodes = edits["deviceNodes"]
        if not isinstance(nodes, list):
            errors.append(f"{context}.deviceNodes: must be a list")
        else:
            for idx, node in enumerate(nodes):
                nc = f"{context}.deviceNodes[{idx}]"
                if not isinstance(node, dict):
                    errors.append(f"{nc}: must be an object")
                    continue
                unknown_node_keys = set(node.keys()) - _ALLOWED_DEVICE_NODE_KEYS
                if unknown_node_keys:
                    errors.append(
                        f"{nc}: unsupported CDI v{cdi_version} deviceNode keys: {sorted(unknown_node_keys)}"
                    )
                path = node.get("path")
                if not isinstance(path, str) or not path.startswith("/"):
                    errors.append(f"{nc}.path: must be an absolute path")
                if "hostPath" in node:
                    hp = node["hostPath"]
                    if not isinstance(hp, str) or not hp.startswith("/"):
                        errors.append(f"{nc}.hostPath: must be an absolute path")
                if "type" in node:
                    node_type = node["type"]
                    if not isinstance(node_type, str) or node_type not in _ALLOWED_DEVICE_TYPES:
                        errors.append(f"{nc}.type: invalid device type {node_type!r}")
                if "permissions" in node:
                    perm = node["permissions"]
                    if (
                        not isinstance(perm, str)
                        or not perm
                        or not set(perm) <= _ALLOWED_PERMISSIONS_CHARS
                    ):
                        errors.append(f"{nc}.permissions: invalid permissions {perm!r}")
                for field_name in ("major", "minor"):
                    if field_name in node:
                        val = node[field_name]
                        if (
                            not isinstance(val, int)
                            or isinstance(val, bool)
                            or val < 0
                            or val > 0x7FFFFFFFFFFFFFFF
                        ):
                            errors.append(f"{nc}.{field_name}: must be a non-negative 64-bit integer")
                for field_name in ("uid", "gid", "fileMode"):
                    if field_name in node:
                        val = node[field_name]
                        if (
                            not isinstance(val, int)
                            or isinstance(val, bool)
                            or val < 0
                            or val > 0xFFFFFFFF
                        ):
                            errors.append(f"{nc}.{field_name}: must be an unsigned 32-bit integer")

    if "mounts" in edits:
        mounts = edits["mounts"]
        if not isinstance(mounts, list):
            errors.append(f"{context}.mounts: must be a list")
        else:
            for idx, m in enumerate(mounts):
                mc = f"{context}.mounts[{idx}]"
                if not isinstance(m, dict):
                    errors.append(f"{mc}: must be an object")
                    continue
                unknown_mount_keys = set(m.keys()) - _ALLOWED_MOUNT_KEYS
                if unknown_mount_keys:
                    errors.append(
                        f"{mc}: unsupported CDI v{cdi_version} mount keys: {sorted(unknown_mount_keys)}"
                    )
                if not isinstance(m.get("hostPath"), str) or not m["hostPath"].startswith("/"):
                    errors.append(f"{mc}.hostPath: must be an absolute path")
                if (
                    not isinstance(m.get("containerPath"), str)
                    or not m["containerPath"].startswith("/")
                ):
                    errors.append(f"{mc}.containerPath: must be an absolute path")
                if "type" in m:
                    m_type = m["type"]
                    if not isinstance(m_type, str):
                        errors.append(f"{mc}.type: must be a string")
                if "options" in m:
                    m_opts = m["options"]
                    if not isinstance(m_opts, list) or not all(
                        isinstance(opt, str) for opt in m_opts
                    ):
                        errors.append(f"{mc}.options: must be a list of strings")

    if "hooks" in edits:
        hooks = edits["hooks"]
        if not isinstance(hooks, list):
            errors.append(f"{context}.hooks: must be a list")
        else:
            for idx, h in enumerate(hooks):
                hc = f"{context}.hooks[{idx}]"
                if not isinstance(h, dict):
                    errors.append(f"{hc}: must be an object")
                    continue
                unknown_hook_keys = set(h.keys()) - _ALLOWED_HOOK_KEYS
                if unknown_hook_keys:
                    errors.append(
                        f"{hc}: unsupported CDI v{cdi_version} hook keys: {sorted(unknown_hook_keys)}"
                    )
                hook_name = h.get("hookName")
                if not isinstance(hook_name, str) or hook_name not in _ALLOWED_HOOK_NAMES:
                    errors.append(f"{hc}.hookName: invalid OCI hookName {hook_name!r}")
                if not isinstance(h.get("path"), str) or not h["path"].startswith("/"):
                    errors.append(f"{hc}.path: must be an absolute executable path")
                if "args" in h:
                    h_args = h["args"]
                    if not isinstance(h_args, list) or not all(
                        isinstance(a, str) for a in h_args
                    ):
                        errors.append(f"{hc}.args: must be a list of strings")
                if "env" in h:
                    h_env = h["env"]
                    if not isinstance(h_env, list) or not all(
                        isinstance(x, str) and "=" in x and bool(x.split("=", 1)[0]) for x in h_env
                    ):
                        errors.append(f"{hc}.env: must be a list of non-empty KEY=VALUE strings")
                if "timeout" in h:
                    h_timeout = h["timeout"]
                    if (
                        not isinstance(h_timeout, int)
                        or isinstance(h_timeout, bool)
                        or h_timeout <= 0
                    ):
                        errors.append(f"{hc}.timeout: must be a positive integer")


def validate_cdi_spec(spec: Any) -> List[str]:
    """Validate a parsed CDI JSON object against CNCF CDI rules and FreeTPU VFIO/iommufd rules."""
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ["CDI spec root must be a JSON object"]

    raw_version = spec.get("cdiVersion")
    if not isinstance(raw_version, str) or raw_version not in SUPPORTED_CDI_VERSIONS:
        errors.append(
            f"cdiVersion must be one of {list(SUPPORTED_CDI_VERSIONS)}, got {raw_version!r}"
        )
        effective_version = DEFAULT_CDI_VERSION
    else:
        effective_version = raw_version

    unknown_root = set(spec.keys()) - _ALLOWED_SPEC_ROOT_KEYS
    if unknown_root:
        errors.append(f"unsupported CDI v{effective_version} root keys: {sorted(unknown_root)}")

    kind = spec.get("kind")
    if not isinstance(kind, str) or not _CDI_KIND_RE.match(kind):
        errors.append(f"kind must match CNCF CDI vendor/class syntax, got {kind!r}")

    if "annotations" in spec:
        ann = spec["annotations"]
        if not isinstance(ann, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in ann.items()
        ):
            errors.append("annotations must be a string-to-string map")

    global_paths: set[str] = set()
    if "containerEdits" in spec:
        _validate_container_edits(
            spec["containerEdits"],
            "containerEdits",
            errors,
            cdi_version=effective_version,
        )
        if isinstance(spec["containerEdits"], dict):
            nodes = spec["containerEdits"].get("deviceNodes")
            if isinstance(nodes, list):
                for n in nodes:
                    if isinstance(n, dict) and isinstance(n.get("path"), str):
                        global_paths.add(n["path"])

    devices = spec.get("devices")
    if not isinstance(devices, list) or not devices:
        errors.append("devices must be a non-empty list")
        return errors

    allowed_edits_keys = _allowed_container_edits_keys(effective_version)
    seen_names: set[str] = set()
    for idx, dev in enumerate(devices):
        dc = f"devices[{idx}]"
        if not isinstance(dev, dict):
            errors.append(f"{dc}: must be a JSON object")
            continue
        unknown_dev = set(dev.keys()) - _ALLOWED_DEVICE_ENTRY_KEYS
        if unknown_dev:
            errors.append(
                f"{dc}: unsupported CDI v{effective_version} device keys: {sorted(unknown_dev)}"
            )
        name = dev.get("name")
        if not isinstance(name, str) or not _CDI_DEVICE_NAME_RE.match(name):
            errors.append(f"{dc}.name: invalid CDI device name {name!r}")
        elif name in seen_names:
            errors.append(f"{dc}.name: duplicate device name {name!r}")
        else:
            seen_names.add(name)

        if "annotations" in dev:
            ann = dev["annotations"]
            if not isinstance(ann, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in ann.items()
            ):
                errors.append(f"{dc}.annotations: must be a string-to-string map")

        dev_paths: set[str] = set()
        if "containerEdits" not in dev:
            errors.append(f"{dc}.containerEdits: required on every CDI device")
        else:
            _validate_container_edits(
                dev["containerEdits"],
                f"{dc}.containerEdits",
                errors,
                cdi_version=effective_version,
            )
            if isinstance(dev["containerEdits"], dict):
                edits_dict = dev["containerEdits"]
                has_any_edit = any(
                    (isinstance(edits_dict.get(k), list) and len(edits_dict[k]) > 0)
                    or (k == "intelRdt" and isinstance(edits_dict.get(k), dict) and len(edits_dict[k]) > 0)
                    for k in allowed_edits_keys
                )
                if not has_any_edit:
                    errors.append(
                        f"{dc}.containerEdits: must contain at least one edit "
                        f"(CDI v{effective_version} rejects empty per-device edits)"
                    )
                nodes = edits_dict.get("deviceNodes")
                if isinstance(nodes, list):
                    for n in nodes:
                        if isinstance(n, dict) and isinstance(n.get("path"), str):
                            dev_paths.add(n["path"])

        effective_paths = global_paths | dev_paths
        label = name if isinstance(name, str) else dc
        has_iommufd = "/dev/iommu" in effective_paths
        has_vfio_ctrl = "/dev/vfio/vfio" in effective_paths
        if not has_iommufd and not has_vfio_ctrl:
            errors.append(
                f"{dc} ({label!r}): effective containerEdits must expose /dev/iommu or /dev/vfio/vfio container node"
            )
        elif has_iommufd:
            if not any(_is_valid_iommufd_cdev_path(p) for p in effective_paths):
                errors.append(
                    f"{dc} ({label!r}): effective containerEdits with /dev/iommu must expose at least one /dev/vfio/devices/vfioN device node"
                )
        else:
            if not any(
                p.startswith("/dev/vfio/")
                and p != "/dev/vfio/vfio"
                and not p.startswith("/dev/vfio/devices/")
                for p in effective_paths
            ):
                errors.append(
                    f"{dc} ({label!r}): effective containerEdits must expose at least one /dev/vfio/<iommu_group> device node"
                )

    return errors


def _resolve_effective_host_path(node: Dict[str, Any]) -> Path:
    """Return the host path CDI actually resolves (`hostPath`, falling back to `path`)."""
    raw = node.get("hostPath")
    if isinstance(raw, str) and raw:
        return Path(raw)
    return Path(str(node.get("path", "")))


def _parse_vfio_devnums_annotation(raw: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not isinstance(raw, str) or not raw.strip():
        return result
    for part in raw.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            result[k] = v
    return result


def _check_host_safe_container_edits(edits: Any, context: str, errors: List[str]) -> None:
    """Reject unchecked mounts, OCI hooks, or other non-VFIO edits in host-validated specs."""
    if not isinstance(edits, dict):
        return
    disallowed = sorted(set(edits.keys()) - _HOST_VALIDATED_CONTAINER_EDITS_KEYS)
    if disallowed:
        errors.append(
            f"{context}: disallowed containerEdits in host-validated TPU spec: {disallowed} "
            "(only exact 'env' and 'deviceNodes' keys are permitted in --check-host mode)"
        )


def _check_host_safe_device_node_keys(node: Dict[str, Any], context: str, errors: List[str]) -> None:
    """Enforce exact allowlist of deviceNode keys and character-device type for VFIO host validation."""
    disallowed = sorted(set(node.keys()) - _HOST_VALIDATED_DEVICE_NODE_KEYS)
    if disallowed:
        errors.append(
            f"{context}: disallowed deviceNode keys in host-validated TPU spec: {disallowed} "
            "(explicit major/minor and non-canonical keys are rejected in --check-host mode)"
        )
    node_type = node.get("type")
    if node_type is not None and node_type not in ("", "c"):
        errors.append(
            f"{context}: VFIO deviceNode type must be omitted or 'c' (character device), got {node_type!r}"
        )


_RANGE_SLICE_RE = re.compile(r"^[0-9]+-[0-9]+\Z")


def _detect_spec_vfio_mode(spec: Dict[str, Any]) -> str:
    """Infer whether a CDI spec targets 'iommufd' or 'legacy' VFIO from its deviceNodes."""
    paths: set[str] = set()
    global_edits = spec.get("containerEdits")
    if isinstance(global_edits, dict) and isinstance(global_edits.get("deviceNodes"), list):
        for n in global_edits["deviceNodes"]:
            if isinstance(n, dict) and isinstance(n.get("path"), str):
                paths.add(n["path"])
    devices = spec.get("devices")
    if isinstance(devices, list):
        for dev in devices:
            if isinstance(dev, dict) and isinstance(dev.get("containerEdits"), dict):
                nodes = dev["containerEdits"].get("deviceNodes")
                if isinstance(nodes, list):
                    for n in nodes:
                        if isinstance(n, dict) and isinstance(n.get("path"), str):
                            paths.add(n["path"])
    if "/dev/iommu" in paths and "/dev/vfio/vfio" not in paths:
        return "iommufd"
    if "/dev/vfio/vfio" in paths:
        return "legacy"
    return "auto"


def _verify_spec_against_host_detailed(
    spec: Any,
    *,
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    iommu_dev_path: Optional[Path] = None,
    vfio_mode: str = "auto",
    chip_id_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[str]]:
    """Revalidate all effective global and per-device CDI nodes against live host VFIO/iommufd state.

    Returns `(confirmed_errors, unverifiable_errors)`:
    - `confirmed_errors`: confirmed hardware, PCI, VFIO, devnum, or topology
      mismatches that justify invalidating/unlinking a stale CDI spec.
    - `unverifiable_errors`: numeric range aliases (e.g. `0-3`, `4-7`) whose
      underlying VFIO/PCI/NUMA hardware still matches an intact live host slice
      but cannot be verified because neither sysfs `chip_id` nor `--chip-id-map`
      was supplied to validation. `validate --check-host --remove-if-stale`
      preserves live specs when only `unverifiable_errors` occur.
    """
    sysfs_root = Path(os.path.abspath(sysfs_root))
    dev_root = Path(os.path.abspath(dev_root))
    errors: List[str] = []
    unverifiable: List[str] = []
    if not isinstance(spec, dict):
        return ["CDI spec root must be a JSON object"], []

    unknown_root = sorted(set(spec.keys()) - _ALLOWED_SPEC_ROOT_KEYS)
    if unknown_root:
        errors.append(f"disallowed root keys in host-validated TPU spec: {unknown_root}")

    try:
        norm_mode = _normalize_vfio_mode(vfio_mode)
    except ValueError as exc:
        return [str(exc)], []
    effective_mode = _detect_spec_vfio_mode(spec) if norm_mode == "auto" else norm_mode

    try:
        live_chips, vfio_container = discover_tpu_chips(
            sysfs_root=sysfs_root,
            dev_root=dev_root,
            allow_missing_vfio=False,
            iommu_dev_path=iommu_dev_path,
            vfio_mode=effective_mode,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"live host VFIO discovery failed: {exc}"], []

    is_iommufd = any(c.vfio_mode == "iommufd" for c in live_chips)
    expected_control_container_path = "/dev/iommu" if is_iommufd else "/dev/vfio/vfio"
    live_container_resolved = str(vfio_container.resolve())
    live_by_host_resolved: Dict[str, Tuple[str, List[str], Optional[str]]] = {}
    live_devnums_by_container_path: Dict[str, str] = {}
    container_dn = _stat_char_devnum(vfio_container)
    if container_dn:
        live_devnums_by_container_path[expected_control_container_path] = container_dn

    live_chips_by_bdf = {c.pci_bdf: c for c in live_chips}
    live_units: Dict[str, List[DiscoveredChip]] = {}
    for chip in live_chips:
        live_units.setdefault(chip.iommu_group, []).append(chip)
    live_reserved_names: set[str] = {
        _single_chip_selector(u[0].pci_bdf)
        if len(u) == 1
        else _shared_group_selector(sorted(c.pci_bdf for c in u))
        for u in live_units.values()
    }
    live_bdf_to_phys = _map_physical_chip_indices(live_chips, chip_id_map=chip_id_map)
    expected_slices_by_name: Dict[str, _TopologySlice] = {}
    expected_slices_by_bdfs: Dict[Tuple[str, ...], _TopologySlice] = {}
    for topo_slice in _compute_topology_slices(
        live_chips,
        live_bdf_to_phys,
        reserved_names=live_reserved_names,
        chip_id_map=chip_id_map,
    ):
        expected_slices_by_bdfs[topo_slice.pci_bdfs] = topo_slice
        for alias in topo_slice.aliases:
            expected_slices_by_name[alias] = topo_slice
    groups_bdfs: Dict[str, List[str]] = {}
    groups_host: Dict[str, Path] = {}
    for chip in live_chips:
        groups_bdfs.setdefault(chip.vfio_container_path, []).append(chip.pci_bdf)
        groups_host.setdefault(chip.vfio_container_path, chip.vfio_host_path)

    for container_path, bdfs in groups_bdfs.items():
        bdfs.sort()
        host_p = groups_host[container_path]
        dn = _stat_char_devnum(host_p)
        if dn:
            live_devnums_by_container_path[container_path] = dn
        live_by_host_resolved[str(host_p.resolve())] = (container_path, bdfs, dn)

    # 1. Validate global containerEdits (applied to EVERY selector).
    global_edits = spec.get("containerEdits")
    _check_host_safe_container_edits(global_edits, "containerEdits", errors)
    global_nodes: List[Dict[str, Any]] = []
    if isinstance(global_edits, dict) and isinstance(global_edits.get("deviceNodes"), list):
        for n_idx, node in enumerate(global_edits["deviceNodes"]):
            nc = f"containerEdits.deviceNodes[{n_idx}]"
            if not isinstance(node, dict):
                errors.append(f"{nc}: must be an object")
                continue
            global_nodes.append(node)
            _check_host_safe_device_node_keys(node, nc, errors)
            eff_host = _resolve_effective_host_path(node)
            if not eff_host.exists():
                errors.append(f"{nc}: effective hostPath {eff_host} does not exist on host")
                continue
            if _stat_char_devnum(eff_host) is None:
                errors.append(
                    f"{nc}: effective hostPath {eff_host} is not a non-symlink character device"
                )
                continue
            resolved_host = str(eff_host.resolve())
            if (
                resolved_host != live_container_resolved
                or node.get("path") != expected_control_container_path
            ):
                errors.append(
                    f"{nc}: global deviceNode exposes unauthorized host device {eff_host} "
                    f"(expected only {vfio_container} mapped to {expected_control_container_path})"
                )

    # Check root recorded char devnums whenever host VFIO nodes are char devices.
    root_ann = spec.get("annotations") if isinstance(spec.get("annotations"), dict) else {}
    recorded_root_devnums = _parse_vfio_devnums_annotation(root_ann.get("freetpu.dev/vfio-devnums"))
    for cp, live_dn in live_devnums_by_container_path.items():
        if recorded_root_devnums.get(cp) != live_dn:
            errors.append(
                f"annotations[freetpu.dev/vfio-devnums]: stale or missing device number for {cp} "
                f"(spec recorded {recorded_root_devnums.get(cp)!r}, host has {live_dn!r})"
            )

    devices = spec.get("devices")
    if not isinstance(devices, list) or not devices:
        return errors + ["devices must be a non-empty list"]

    # 2. Validate every device's effective containerEdits and deviceNodes (global + per-device).
    for idx, dev in enumerate(devices):
        if not isinstance(dev, dict):
            continue
        name = dev.get("name", f"devices[{idx}]")
        unknown_dev = sorted(set(dev.keys()) - _ALLOWED_DEVICE_ENTRY_KEYS)
        if unknown_dev:
            errors.append(f"device {name!r}: disallowed device keys in host-validated TPU spec: {unknown_dev}")
        edits = dev.get("containerEdits")
        _check_host_safe_container_edits(edits, f"device {name!r} containerEdits", errors)
        dev_nodes = (
            edits.get("deviceNodes")
            if isinstance(edits, dict) and isinstance(edits.get("deviceNodes"), list)
            else []
        )

        matched_group_paths: List[str] = []
        expected_live_bdfs: List[str] = []
        has_container_node = any(
            _stat_char_devnum(_resolve_effective_host_path(gn)) is not None
            and str(_resolve_effective_host_path(gn).resolve()) == live_container_resolved
            and gn.get("path") == expected_control_container_path
            for gn in global_nodes
            if _resolve_effective_host_path(gn).exists()
        )

        for n_idx, node in enumerate(dev_nodes):
            nc = f"device {name!r} deviceNodes[{n_idx}]"
            if not isinstance(node, dict):
                errors.append(f"{nc}: must be an object")
                continue
            _check_host_safe_device_node_keys(node, nc, errors)
            eff_host = _resolve_effective_host_path(node)
            if not eff_host.exists():
                errors.append(f"{nc}: effective hostPath {eff_host} does not exist on host")
                continue
            if _stat_char_devnum(eff_host) is None:
                errors.append(
                    f"{nc}: effective hostPath {eff_host} is not a non-symlink character device"
                )
                continue
            resolved_host = str(eff_host.resolve())
            if resolved_host == live_container_resolved:
                if node.get("path") != expected_control_container_path:
                    errors.append(
                        f"{nc}: VFIO container {eff_host} must map to {expected_control_container_path}, got {node.get('path')!r}"
                    )
                else:
                    has_container_node = True
                continue
            if resolved_host not in live_by_host_resolved:
                errors.append(
                    f"{nc}: effective hostPath {eff_host} is not a verified TPU-only VFIO group on this host"
                )
                continue
            live_cp, live_bdfs, _live_dn = live_by_host_resolved[resolved_host]
            if node.get("path") != live_cp:
                errors.append(
                    f"{nc}: container path {node.get('path')!r} does not match host VFIO node {live_cp!r} for {eff_host}"
                )
            matched_group_paths.append(live_cp)
            expected_live_bdfs.extend(live_bdfs)

        if not has_container_node:
            errors.append(f"device {name!r}: effective deviceNodes missing live VFIO container {vfio_container}")
        if not matched_group_paths:
            errors.append(f"device {name!r}: no verified host VFIO/iommufd deviceNodes to verify")
            continue

        expected_live_bdfs = sorted(set(expected_live_bdfs))
        touched_groups = {
            live_chips_by_bdf[b].iommu_group
            for b in expected_live_bdfs
            if b in live_chips_by_bdf
        }
        required_group_bdfs = sorted(
            c.pci_bdf for c in live_chips if c.iommu_group in touched_groups
        )
        if expected_live_bdfs != required_group_bdfs:
            errors.append(
                f"device {name!r}: partial IOMMU group allocation; exposes {expected_live_bdfs} "
                f"without full IOMMU group membership {required_group_bdfs}"
            )

        ann = dev.get("annotations") if isinstance(dev.get("annotations"), dict) else {}
        recorded_bdfs: List[str] = []
        if isinstance(ann.get("freetpu.dev/pci-bdf"), str) and ann["freetpu.dev/pci-bdf"].strip():
            recorded_bdfs = [ann["freetpu.dev/pci-bdf"].strip()]
        elif isinstance(ann.get("freetpu.dev/pci-bdfs"), str) and ann["freetpu.dev/pci-bdfs"].strip():
            recorded_bdfs = sorted(
                b.strip() for b in ann["freetpu.dev/pci-bdfs"].split(",") if b.strip()
            )
        else:
            errors.append(
                f"device {name!r}: missing freetpu.dev/pci-bdf(s) annotation required to verify VFIO group identity"
            )
            continue

        if recorded_bdfs != expected_live_bdfs:
            errors.append(
                f"device {name!r}: stale VFIO group identity for {','.join(matched_group_paths)} "
                f"(spec expects PCI BDF(s) {recorded_bdfs}, host currently exposes {expected_live_bdfs})"
            )

        if isinstance(name, str):
            if name.startswith("pci-"):
                if len(expected_live_bdfs) != 1 or name != _single_chip_selector(expected_live_bdfs[0]):
                    expected_sel = (
                        _single_chip_selector(expected_live_bdfs[0])
                        if len(expected_live_bdfs) == 1
                        else _shared_group_selector(expected_live_bdfs)
                    )
                    errors.append(
                        f"device {name!r}: PCI selector name does not match verified VFIO group "
                        f"membership {expected_live_bdfs} (expected selector {expected_sel!r})"
                    )
            elif name.startswith("group-pci-"):
                expected_sel = _shared_group_selector(expected_live_bdfs)
                if name != expected_sel:
                    errors.append(
                        f"device {name!r}: shared-group PCI selector name does not match verified "
                        f"VFIO group membership {expected_live_bdfs} (expected selector {expected_sel!r})"
                    )
            elif name.startswith("numa") or name.startswith("pcie") or _RANGE_SLICE_RE.match(name):
                slice_domains = {
                    _topology_domain_key(live_chips_by_bdf[b])
                    for b in expected_live_bdfs
                    if b in live_chips_by_bdf
                }
                if len(slice_domains) > 1:
                    errors.append(
                        f"device {name!r}: sub-host slice crosses NUMA / PCIe root complex boundaries ({expected_live_bdfs})"
                    )
                expected_slice = expected_slices_by_name.get(name)
                check_slice = expected_slice
                range_match = _RANGE_SLICE_RE.match(name)
                live_slice_for_bdfs = (
                    expected_slices_by_bdfs.get(tuple(expected_live_bdfs))
                    if range_match is not None
                    else None
                )
                range_width_matches = False
                if range_match is not None and live_slice_for_bdfs is not None:
                    start_idx, end_idx = (int(p) for p in name.split("-", 1))
                    range_width_matches = (
                        end_idx >= start_idx
                        and (end_idx - start_idx + 1)
                        == len(live_slice_for_bdfs.physical_indices)
                    )
                uses_explicit_chip_map = ann.get("freetpu.dev/chip-id-source") == "chip-id-map"
                if expected_slice is None:
                    if (
                        range_width_matches
                        and chip_id_map is None
                        and live_slice_for_bdfs is not None
                        and (
                            uses_explicit_chip_map
                            or _resolve_verified_visibility_ordinals(
                                live_slice_for_bdfs.chips,
                                live_chips,
                                live_bdf_to_phys,
                                chip_id_map=None,
                            )
                            is None
                        )
                    ):
                        unverifiable.append(
                            f"device {name!r}: numeric range alias cannot be verified without "
                            "--chip-id-map (spec depends on explicit --chip-id-map or host sysfs "
                            "exposes no chip_id; underlying VFIO/PCI topology still matches live host)"
                        )
                        check_slice = live_slice_for_bdfs
                    else:
                        errors.append(
                            f"device {name!r}: slice alias does not match any live host topology slice "
                            f"(live aliases: {sorted(expected_slices_by_name)})"
                        )
                elif expected_live_bdfs != list(expected_slice.pci_bdfs):
                    if (
                        range_width_matches
                        and chip_id_map is None
                        and live_slice_for_bdfs is not None
                        and uses_explicit_chip_map
                    ):
                        unverifiable.append(
                            f"device {name!r}: numeric range alias was generated with --chip-id-map "
                            "and cannot be verified without that map (underlying VFIO/PCI topology "
                            "still matches live host)"
                        )
                        check_slice = live_slice_for_bdfs
                    else:
                        errors.append(
                            f"device {name!r}: slice alias PCI membership {expected_live_bdfs} does not "
                            f"match live host topology for {name!r} ({list(expected_slice.pci_bdfs)})"
                        )
                        check_slice = None

                if check_slice is not None:
                    if (
                        "freetpu.dev/pcie-root" in ann
                        and ann.get("freetpu.dev/pcie-root") != check_slice.pcie_root
                    ):
                        errors.append(
                            f"device {name!r}: stale freetpu.dev/pcie-root "
                            f"{ann.get('freetpu.dev/pcie-root')!r} (host has {check_slice.pcie_root!r})"
                        )
                    if check_slice.numa_node is None and "freetpu.dev/numa-node" in ann:
                        errors.append(
                            f"device {name!r}: unexpected freetpu.dev/numa-node "
                            f"{ann.get('freetpu.dev/numa-node')!r} when host sysfs reports no NUMA node"
                        )
                    elif (
                        check_slice.numa_node is not None
                        and "freetpu.dev/numa-node" in ann
                        and ann.get("freetpu.dev/numa-node") != str(check_slice.numa_node)
                    ):
                        errors.append(
                            f"device {name!r}: stale freetpu.dev/numa-node "
                            f"{ann.get('freetpu.dev/numa-node')!r} (host has {str(check_slice.numa_node)!r})"
                        )
            elif name == "all":
                all_live_bdfs = sorted(c.pci_bdf for c in live_chips)
                if expected_live_bdfs != all_live_bdfs:
                    errors.append(
                        f"device 'all': exposed VFIO groups {expected_live_bdfs} do not match "
                        f"all live host TPU chips {all_live_bdfs}"
                    )

        recorded_dev_dn = _parse_vfio_devnums_annotation(ann.get("freetpu.dev/vfio-devnums"))
        for cp in matched_group_paths:
            live_dn = live_devnums_by_container_path.get(cp)
            if live_dn and recorded_dev_dn.get(cp) != live_dn:
                errors.append(
                    f"device {name!r}: stale or missing VFIO device number for {cp} "
                    f"(spec recorded {recorded_dev_dn.get(cp)!r}, host has {live_dn!r})"
                )

    return errors, unverifiable


def verify_spec_against_host(
    spec: Any,
    *,
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    iommu_dev_path: Optional[Path] = None,
    vfio_mode: str = "auto",
    chip_id_map: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Revalidate all effective global and per-device CDI nodes against live host VFIO/iommufd state."""
    confirmed, unverifiable = _verify_spec_against_host_detailed(
        spec,
        sysfs_root=sysfs_root,
        dev_root=dev_root,
        iommu_dev_path=iommu_dev_path,
        vfio_mode=vfio_mode,
        chip_id_map=chip_id_map,
    )
    return confirmed + unverifiable


def _write_spec_atomic(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _invalidate_spec(path: Path) -> Tuple[bool, Optional[str]]:
    """Remove a stale CDI spec file if present. Returns (removed, cleanup_error)."""
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
            return True, None
    except OSError as exc:
        return False, str(exc)
    return False, None


def _format_invalidation_suffix(path: Path, removed: bool, cleanup_err: Optional[str]) -> str:
    if removed:
        return f" (removed stale {path})"
    if cleanup_err:
        return f" (FAILED to remove stale {path}: {cleanup_err})"
    return ""


def reconcile_cdi_spec(
    output_path: Path,
    *,
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    kind: str = CDI_KIND,
    network_mode: str = "host",
    single_host_env: bool = False,
    iommu_dev_path: Optional[Path] = None,
    cdi_version: str = DEFAULT_CDI_VERSION,
    vfio_mode: str = "auto",
    chip_id_map: Optional[Dict[str, int]] = None,
    omit_visible_env: bool = False,
) -> Tuple[bool, str]:
    """Generate, validate, and atomically publish a live CDI spec, invalidating stale files on error."""
    if not isinstance(kind, str) or not _CDI_KIND_RE.match(kind):
        return False, f"Invalid CDI kind {kind!r} (existing spec preserved)"
    if network_mode not in NETWORK_MODES:
        return False, f"Invalid network_mode {network_mode!r} (existing spec preserved)"
    if not isinstance(cdi_version, str) or cdi_version not in SUPPORTED_CDI_VERSIONS:
        return False, f"Unsupported CDI version {cdi_version!r} (existing spec preserved)"
    if vfio_mode not in VFIO_MODES:
        return False, f"Invalid vfio_mode {vfio_mode!r} (existing spec preserved)"
    try:
        chips, vfio_container = discover_tpu_chips(
            sysfs_root=sysfs_root,
            dev_root=dev_root,
            allow_missing_vfio=False,
            iommu_dev_path=iommu_dev_path,
            vfio_mode=vfio_mode,
        )
        spec = build_cdi_spec(
            chips,
            vfio_container,
            kind=kind,
            network_mode=network_mode,
            single_host_env=single_host_env,
            cdi_version=cdi_version,
            chip_id_map=chip_id_map,
            omit_visible_env=omit_visible_env,
        )
    except Exception as exc:  # noqa: BLE001
        removed, cleanup_err = _invalidate_spec(output_path)
        suffix = _format_invalidation_suffix(output_path, removed, cleanup_err)
        return False, f"discovery failed: {exc}{suffix}"

    errors = validate_cdi_spec(spec)
    if not errors:
        errors.extend(
            verify_spec_against_host(
                spec,
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                iommu_dev_path=iommu_dev_path,
                vfio_mode=vfio_mode,
                chip_id_map=chip_id_map,
            )
        )
    if errors:
        removed, cleanup_err = _invalidate_spec(output_path)
        suffix = _format_invalidation_suffix(output_path, removed, cleanup_err)
        return False, f"validation failed: {'; '.join(errors)}{suffix}"

    rendered = json.dumps(spec, indent=2) + "\n"
    try:
        if output_path.exists() and output_path.read_text(encoding="utf-8") == rendered:
            return True, f"unchanged ({len(chips)} chip(s), kind={kind}) at {output_path}"
    except OSError:
        pass

    try:
        _write_spec_atomic(output_path, rendered)
    except OSError as exc:
        removed, cleanup_err = _invalidate_spec(output_path)
        suffix = _format_invalidation_suffix(output_path, removed, cleanup_err)
        return False, f"publication failed: {exc}{suffix}"

    return True, f"updated CNCF CDI v{cdi_version} spec ({len(chips)} chip(s), kind={kind}) at {output_path}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freetpu-cdi",
        description=f"FreeTPU CNCF CDI (default v{DEFAULT_CDI_VERSION}) spec generator & container toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help=(
            f"Write a point-in-time boot/topology CNCF CDI (default v{DEFAULT_CDI_VERSION}) JSON snapshot "
            "(before any VFIO/PCI unbind or rebind: stop consumers, stop `sync` if "
            "running, remove the CDI spec and confirm runtime withdrawal, rebind, "
            "then regenerate)."
        ),
    )
    gen.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CDI JSON path (default boot-scoped tmpfs: {DEFAULT_OUTPUT_PATH}).",
    )
    gen.add_argument(
        "--stdout",
        action="store_true",
        help="Print generated CDI JSON to stdout instead of writing a file.",
    )
    gen.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated CDI spec against CNCF CDI rules before emitting.",
    )
    gen.add_argument(
        "--kind",
        default=CDI_KIND,
        help=f"CDI vendor/class kind (default: {CDI_KIND}).",
    )
    gen.add_argument(
        "--cdi-version",
        choices=list(SUPPORTED_CDI_VERSIONS),
        default=DEFAULT_CDI_VERSION,
        help=(
            f"CNCF CDI spec version to emit (default: {DEFAULT_CDI_VERSION}, "
            "the minimum version needed by generated fields and compatible with "
            "unaffected/patched containerd releases: 2.0.x, >=2.1.9, >=2.2.5, "
            ">=2.3.2, >=2.4.0)."
        ),
    )
    gen.add_argument(
        "--vfio-mode",
        choices=sorted(VFIO_MODES),
        default="auto",
        help=(
            "VFIO device interface mode: 'auto' (default: preserves legacy VFIO "
            "/dev/vfio/<group> + /dev/vfio/vfio whenever usable, falling back to "
            "iommufd only when legacy /dev/vfio/vfio is absent and /dev/iommu exists), "
            "'legacy', or 'iommufd'/'cdev' (explicit Linux 6.8+ iommufd cdev opt-in)."
        ),
    )
    gen.add_argument(
        "--network-mode",
        choices=sorted(NETWORK_MODES),
        default="host",
        help=(
            "Container networking mode (default: 'host', preserving workload-provided "
            "libtpu peer discovery, visibility, and host bounds for multi-host compatibility; "
            "pass 'isolated' or --single-host-env to opt into single-host loopback "
            "Megascale/bounds/MDS-skip overrides)."
        ),
    )
    gen.add_argument(
        "--single-host-env",
        action="store_true",
        default=False,
        help=(
            "Opt in to injecting single-host isolated libtpu env overrides "
            "(MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353, TPU_HOST_BOUNDS=1,1,1, "
            "TPU_WORKER_HOSTNAMES=localhost, TPU_WORKER_ID=0, TPU_SKIP_MDS_QUERY=true, "
            "plus TPU_CHIPS_PER_HOST_BOUNDS, TPU_ACCELERATOR_TYPE, and verified "
            "TPU_VISIBLE_CHIPS / TPU_VISIBLE_DEVICES when runtime ordinals are verified "
            "via sysfs chip_id or --chip-id-map unless --omit-visible-env is set)."
        ),
    )
    gen.add_argument(
        "--omit-visible-env",
        action="store_true",
        default=False,
        help=(
            "With --single-host-env, omit TPU_VISIBLE_CHIPS and TPU_VISIBLE_DEVICES "
            "while keeping isolated networking and topology settings (for slice-aware "
            "multi-process launchers such as PyTorch/XLA that set both process count "
            "and each rank's assigned chip ID rather than inheriting a full-slice mask)."
        ),
    )
    gen.add_argument(
        "--chip-id-map",
        default=None,
        help=(
            "Optional JSON object or file path mapping PCI BDF -> verified runtime "
            "chip/device ordinal (used with --single-host-env to emit authoritative "
            "TPU_VISIBLE_CHIPS / TPU_VISIBLE_DEVICES on sub-host slices)."
        ),
    )
    gen.add_argument(
        "--sysfs-root",
        type=Path,
        default=Path("/sys"),
        help="Sysfs root path (default: /sys).",
    )
    gen.add_argument(
        "--dev-root",
        type=Path,
        default=Path("/dev"),
        help="Devfs root path (default: /dev).",
    )
    gen.add_argument(
        "--iommu-dev",
        type=Path,
        default=None,
        help="Optional override path for /dev/iommu probing (default: <dev-root>/iommu).",
    )
    gen.add_argument(
        "--allow-missing-vfio",
        action="store_true",
        help="Do not fail if /dev/vfio nodes are absent (dry-run inspection only; requires --stdout).",
    )

    val = sub.add_parser(
        "validate",
        help="Validate a CNCF CDI JSON spec file (or '-' for stdin).",
    )
    val.add_argument(
        "spec_path",
        help="Path to CDI JSON file, or '-' to read from stdin.",
    )
    val.add_argument(
        "--check-host",
        action="store_true",
        help="Also revalidate every VFIO/iommufd unit's PCI BDF identity and isolation against live sysfs.",
    )
    val.add_argument(
        "--remove-if-stale",
        action="store_true",
        help="With --check-host, unlink spec_path if live host VFIO verification fails.",
    )
    val.add_argument(
        "--vfio-mode",
        choices=sorted(VFIO_MODES),
        default="auto",
        help="VFIO interface mode for --check-host (default: auto, matching spec's declared mode).",
    )
    val.add_argument(
        "--sysfs-root",
        type=Path,
        default=Path("/sys"),
        help="Sysfs root path for --check-host (default: /sys).",
    )
    val.add_argument(
        "--dev-root",
        type=Path,
        default=Path("/dev"),
        help="Devfs root path for --check-host (default: /dev).",
    )
    val.add_argument(
        "--iommu-dev",
        type=Path,
        default=None,
        help="Optional override path for /dev/iommu probing in --check-host mode.",
    )
    val.add_argument(
        "--chip-id-map",
        default=None,
        help=(
            "Optional JSON object or file path mapping PCI BDF -> verified runtime "
            "chip/device ordinal (used with --check-host to verify numeric range aliases)."
        ),
    )

    sync = sub.add_parser(
        "sync",
        help=(
            "Continuously reconcile a boot-scoped CDI spec against live VFIO/iommufd topology, "
            "removing stale specs on drift/error (before any planned VFIO/PCI unbind or "
            "rebind: stop affected consumers, stop/pause `sync` so it cannot recreate "
            "the spec during maintenance, delete the CDI spec and confirm the runtime "
            "withdrew its selectors, rebind, then resume `sync`)."
        ),
    )
    sync.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CDI JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    sync.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Revalidation interval in seconds (default: 15.0).",
    )
    sync.add_argument(
        "--once",
        action="store_true",
        help="Run a single reconciliation pass and exit (0 if valid, 1 if invalidated).",
    )
    sync.add_argument(
        "--unlink-on-exit",
        action="store_true",
        help="Remove the published CDI spec file on graceful shutdown (SIGTERM/SIGINT).",
    )
    sync.add_argument(
        "--kind",
        default=CDI_KIND,
        help=f"CDI vendor/class kind (default: {CDI_KIND}).",
    )
    sync.add_argument(
        "--cdi-version",
        choices=list(SUPPORTED_CDI_VERSIONS),
        default=DEFAULT_CDI_VERSION,
        help=f"CNCF CDI spec version to emit (default: {DEFAULT_CDI_VERSION}).",
    )
    sync.add_argument(
        "--vfio-mode",
        choices=sorted(VFIO_MODES),
        default="auto",
        help="VFIO device interface mode ('auto' default, 'legacy', or 'iommufd'/'cdev').",
    )
    sync.add_argument(
        "--network-mode",
        choices=sorted(NETWORK_MODES),
        default="host",
        help="Container networking mode ('host' default preserving workload env; 'isolated' or --single-host-env opts into single-host overrides).",
    )
    sync.add_argument(
        "--single-host-env",
        action="store_true",
        default=False,
        help="Opt in to injecting single-host isolated libtpu environment overrides.",
    )
    sync.add_argument(
        "--omit-visible-env",
        action="store_true",
        default=False,
        help=(
            "With --single-host-env, omit TPU_VISIBLE_CHIPS and TPU_VISIBLE_DEVICES "
            "for slice-aware multi-process launchers (e.g. PyTorch/XLA) that set both "
            "process count and each rank's assigned chip ID."
        ),
    )
    sync.add_argument(
        "--sysfs-root",
        type=Path,
        default=Path("/sys"),
        help="Sysfs root path (default: /sys).",
    )
    sync.add_argument(
        "--dev-root",
        type=Path,
        default=Path("/dev"),
        help="Devfs root path (default: /dev).",
    )
    sync.add_argument(
        "--iommu-dev",
        type=Path,
        default=None,
        help="Optional override path for /dev/iommu probing (default: <dev-root>/iommu).",
    )
    sync.add_argument(
        "--chip-id-map",
        default=None,
        help=(
            "Optional JSON object or file path mapping PCI BDF -> verified runtime "
            "chip/device ordinal for --single-host-env sub-host slice visibility."
        ),
    )

    dargs = sub.add_parser(
        "docker-args",
        help="Print recommended Docker / Podman CLI flags for FreeTPU CDI passthrough.",
    )
    dargs.add_argument(
        "--device",
        default="all",
        help="CDI device selector suffix (default: all -> <kind>=all).",
    )
    dargs.add_argument(
        "--kind",
        default=CDI_KIND,
        help=f"CDI vendor/class kind (default: {CDI_KIND}).",
    )
    dargs.add_argument(
        "--network-mode",
        choices=sorted(NETWORK_MODES),
        default="isolated",
        help="Container networking mode (default: isolated omits --net=host; 'host' adds --net=host).",
    )
    dargs.add_argument(
        "--cap-ipc-lock",
        action="store_true",
        help="Emit --cap-add=IPC_LOCK instead of --ulimit memlock=-1:-1 (fallback when RLIMIT_MEMLOCK cannot be raised).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        if not isinstance(args.kind, str) or not _CDI_KIND_RE.match(args.kind):
            sys.stderr.write(f"freetpu-cdi generate failed: Invalid CDI kind {args.kind!r}\n")
            return 1
        if args.allow_missing_vfio and not args.stdout:
            sys.stderr.write(
                "freetpu-cdi generate failed: --allow-missing-vfio is for dry-run inspection only "
                "and requires --stdout (refusing to overwrite a live CDI spec file)\n"
            )
            return 1

        try:
            parsed_chip_map = _load_chip_id_map(args.chip_id_map)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"freetpu-cdi generate failed: invalid --chip-id-map: {exc}\n")
            return 1

        try:
            chips, vfio_container = discover_tpu_chips(
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
                allow_missing_vfio=args.allow_missing_vfio,
                iommu_dev_path=args.iommu_dev,
                vfio_mode=args.vfio_mode,
            )
            spec = build_cdi_spec(
                chips,
                vfio_container,
                kind=args.kind,
                network_mode=args.network_mode,
                single_host_env=args.single_host_env,
                cdi_version=args.cdi_version,
                chip_id_map=parsed_chip_map,
                omit_visible_env=args.omit_visible_env,
            )
        except Exception as exc:  # noqa: BLE001
            if not args.stdout and not args.allow_missing_vfio:
                removed, cleanup_err = _invalidate_spec(args.output)
                if removed:
                    sys.stderr.write(f"Removed stale CDI spec at {args.output}\n")
                elif cleanup_err:
                    sys.stderr.write(f"Failed to remove stale CDI spec at {args.output}: {cleanup_err}\n")
            sys.stderr.write(f"freetpu-cdi generate failed: {exc}\n")
            return 1

        if args.validate:
            errors = validate_cdi_spec(spec)
            if not errors and not args.allow_missing_vfio:
                errors.extend(
                    verify_spec_against_host(
                        spec,
                        sysfs_root=args.sysfs_root,
                        dev_root=args.dev_root,
                        iommu_dev_path=args.iommu_dev,
                        vfio_mode=args.vfio_mode,
                        chip_id_map=parsed_chip_map,
                    )
                )
            if errors:
                if not args.stdout and not args.allow_missing_vfio:
                    removed, cleanup_err = _invalidate_spec(args.output)
                    if removed:
                        sys.stderr.write(f"Removed stale CDI spec at {args.output}\n")
                    elif cleanup_err:
                        sys.stderr.write(f"Failed to remove stale CDI spec at {args.output}: {cleanup_err}\n")
                for err in errors:
                    sys.stderr.write(f"CDI validation error: {err}\n")
                return 1

        rendered = json.dumps(spec, indent=2) + "\n"
        if args.stdout:
            sys.stdout.write(rendered)
            return 0

        try:
            _write_spec_atomic(args.output, rendered)
        except OSError as exc:
            removed, cleanup_err = _invalidate_spec(args.output)
            if removed:
                sys.stderr.write(f"Removed stale CDI spec at {args.output}\n")
            elif cleanup_err:
                sys.stderr.write(f"Failed to remove stale CDI spec at {args.output}: {cleanup_err}\n")
            sys.stderr.write(f"freetpu-cdi generate failed to publish {args.output}: {exc}\n")
            return 1
        sys.stderr.write(
            f"Wrote CNCF CDI v{args.cdi_version} spec ({len(chips)} chip(s), kind={args.kind}) to {args.output}\n"
        )
        return 0

    if args.command == "validate":
        if args.spec_path == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(args.spec_path).read_text(encoding="utf-8")
        try:
            parsed = _load_json_strict_keys(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            if args.check_host and args.remove_if_stale and args.spec_path != "-":
                removed, cleanup_err = _invalidate_spec(Path(args.spec_path))
                if removed:
                    sys.stderr.write(f"Removed stale CDI spec at {args.spec_path}\n")
                elif cleanup_err:
                    sys.stderr.write(f"Failed to remove stale CDI spec at {args.spec_path}: {cleanup_err}\n")
            sys.stderr.write(f"Invalid JSON: {exc}\n")
            return 1
        errors = validate_cdi_spec(parsed)
        unverifiable: List[str] = []
        if not errors and args.check_host:
            try:
                val_chip_map = _load_chip_id_map(args.chip_id_map)
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"freetpu-cdi validate failed: invalid --chip-id-map: {exc}\n")
                return 1
            host_confirmed, unverifiable = _verify_spec_against_host_detailed(
                parsed,
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
                iommu_dev_path=args.iommu_dev,
                vfio_mode=args.vfio_mode,
                chip_id_map=val_chip_map,
            )
            errors.extend(host_confirmed)
        if errors:
            if args.check_host and args.remove_if_stale and args.spec_path != "-":
                removed, cleanup_err = _invalidate_spec(Path(args.spec_path))
                if removed:
                    sys.stderr.write(f"Removed stale CDI spec at {args.spec_path}\n")
                elif cleanup_err:
                    sys.stderr.write(f"Failed to remove stale CDI spec at {args.spec_path}: {cleanup_err}\n")
            for err in errors + unverifiable:
                sys.stderr.write(f"CDI validation error: {err}\n")
            return 1
        if unverifiable:
            if args.check_host and args.remove_if_stale and args.spec_path != "-":
                sys.stderr.write(
                    f"Preserved live CDI spec at {args.spec_path}: hardware/VFIO identity matched "
                    "but numeric range aliases require --chip-id-map to verify\n"
                )
            for warn in unverifiable:
                sys.stderr.write(f"CDI validation error: {warn}\n")
            return 1
        suffix = " (host VFIO identity verified)" if args.check_host else ""
        spec_ver = parsed.get("cdiVersion", DEFAULT_CDI_VERSION)
        sys.stdout.write(
            f"VALID CNCF CDI v{spec_ver} spec ({parsed.get('kind')}, {len(parsed.get('devices', []))} device entries){suffix}\n"
        )
        return 0

    if args.command == "sync":
        if not isinstance(args.kind, str) or not _CDI_KIND_RE.match(args.kind):
            sys.stderr.write(f"freetpu-cdi sync failed: Invalid CDI kind {args.kind!r}\n")
            return 1
        try:
            parsed_chip_map = _load_chip_id_map(args.chip_id_map)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"freetpu-cdi sync failed: invalid --chip-id-map: {exc}\n")
            return 1
        if args.once:
            ok, msg = reconcile_cdi_spec(
                args.output,
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
                kind=args.kind,
                network_mode=args.network_mode,
                single_host_env=args.single_host_env,
                iommu_dev_path=args.iommu_dev,
                cdi_version=args.cdi_version,
                vfio_mode=args.vfio_mode,
                chip_id_map=parsed_chip_map,
                omit_visible_env=args.omit_visible_env,
            )
            sys.stderr.write(f"freetpu-cdi sync: {msg}\n")
            return 0 if ok else 1

        stop_requested = False

        def _handle_signal(_signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        last_msg = ""
        interval = max(1.0, float(args.interval))
        while not stop_requested:
            try:
                ok, msg = reconcile_cdi_spec(
                    args.output,
                    sysfs_root=args.sysfs_root,
                    dev_root=args.dev_root,
                    kind=args.kind,
                    network_mode=args.network_mode,
                    single_host_env=args.single_host_env,
                    iommu_dev_path=args.iommu_dev,
                    cdi_version=args.cdi_version,
                    vfio_mode=args.vfio_mode,
                    chip_id_map=parsed_chip_map,
                    omit_visible_env=args.omit_visible_env,
                )
            except Exception as exc:  # noqa: BLE001
                removed, cleanup_err = _invalidate_spec(args.output)
                suffix = _format_invalidation_suffix(args.output, removed, cleanup_err)
                ok = False
                msg = f"unexpected sync error: {exc}{suffix}"
            if msg != last_msg or not ok:
                sys.stderr.write(f"freetpu-cdi sync: {msg}\n")
                sys.stderr.flush()
                last_msg = msg
            deadline = time.monotonic() + interval
            while not stop_requested and time.monotonic() < deadline:
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        if args.unlink_on_exit:
            removed, cleanup_err = _invalidate_spec(args.output)
            if removed:
                sys.stderr.write(f"freetpu-cdi sync: withdrew CDI spec at {args.output} on exit\n")
            elif cleanup_err:
                sys.stderr.write(
                    f"freetpu-cdi sync: failed to withdraw CDI spec at {args.output} on exit: {cleanup_err}\n"
                )
        return 0

    if args.command == "docker-args":
        if not isinstance(args.kind, str) or not _CDI_KIND_RE.match(args.kind):
            sys.stderr.write(f"freetpu-cdi docker-args failed: invalid CDI kind {args.kind!r}\n")
            return 1
        flags = [f"--device={args.kind}={args.device}"]
        if args.network_mode == "host":
            flags.append("--net=host")
        if args.cap_ipc_lock:
            flags.append("--cap-add=IPC_LOCK")
        else:
            flags.append("--ulimit memlock=-1:-1")
        sys.stdout.write(" ".join(flags) + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
