#!/usr/bin/env python3
"""FreeTPU Open CNCF CDI Spec Generator & Container Passthrough Toolkit.

Scans /sys/bus/pci/devices/* (vendor 0x1ae0) and VFIO / iommufd device nodes
on VFIO-backed Cloud TPU hosts (v5e, v5p, v6e Trillium, and tpu7x pending live
HW confirmation) to generate a boot-scoped CNCF Container Device Interface (CDI)
v1.1.0 specification (default: /var/run/cdi/freetpu.json) exposing:
  - `freetpu.dev/tpu=pci-<bdf>` (and `group-pci-...` for shared legacy groups)
  - NUMA / PCIe-root-complex-aware 4-chip sub-host presets (`numa0`, `numa1`,
    `0-3`, `4-7`) on multi-root hosts (e.g. Cloud TPU v6e-8 `0000:00` vs `0000:80`)
  - `freetpu.dev/tpu=all`

Also supports:
  - Dual-path Linux 6.8+ `iommufd` (`/dev/iommu` + `/dev/vfio/devices/vfioN`) and
    legacy VFIO (`/dev/vfio/vfio` + `/dev/vfio/<iommu_group>`) auto-detection
  - Isolated container networking without `--net=host` via CDI
    `containerEdits.env` (`MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353`,
    `TPU_HOST_BOUNDS=1,1,1`, slice-scoped `TPU_CHIPS_PER_HOST_BOUNDS`,
    `TPU_VISIBLE_DEVICES`, and `TPU_SKIP_MDS_QUERY=1`), or explicit
    `--network-mode host` for multi-host slices
  - `validate`: strict CNCF CDI v1.1.0 structural & live host VFIO/iommufd validator
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

CDI_VERSION = "1.1.0"
CDI_KIND = "freetpu.dev/tpu"
GOOGLE_PCI_VENDOR_ID = "0x1ae0"
# Boot-scoped CNCF CDI dynamic spec directory (/var/run is tmpfs on Linux so
# stale IOMMU group assignments never persist across reboots).
DEFAULT_OUTPUT_PATH = Path("/var/run/cdi/freetpu.json")
NETWORK_MODES = frozenset({"isolated", "host"})

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
# CNCF CDI v1.1.0 containerEdits keys (includes intelRdt & additionalGids from v0.7+/v0.8+).
_ALLOWED_CONTAINER_EDITS_KEYS = frozenset(
    {"env", "deviceNodes", "hooks", "mounts", "intelRdt", "additionalGids"}
)
_ALLOWED_INTEL_RDT_KEYS = frozenset(
    {"closID", "l3CacheAllocation", "memBwAllocation", "schemata", "enableMonitoring"}
)
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


def _resolve_iommufd_cdev(
    pci_entry: Path,
    sysfs_root: Path,
    fallback_index: int,
) -> Tuple[str, Optional[str]]:
    """Resolve the Linux 6.8+ iommufd cdev name (`vfioN`) and optional kernel devnum."""
    vfio_dev_dir = pci_entry / "vfio-dev"
    if vfio_dev_dir.is_dir():
        try:
            cdevs = sorted(
                (p for p in vfio_dev_dir.iterdir() if p.name.startswith("vfio")),
                key=lambda p: p.name,
            )
        except OSError:
            cdevs = []
        if cdevs:
            cdev_dir = cdevs[0]
            devnum = _read_text(cdev_dir / "dev")
            if devnum and not _DEVNUM_RE.fullmatch(devnum):
                devnum = None
            if devnum is None:
                class_devnum = _read_text(sysfs_root / "class" / "vfio-dev" / cdev_dir.name / "dev")
                if class_devnum and _DEVNUM_RE.fullmatch(class_devnum):
                    devnum = class_devnum
            return cdev_dir.name, devnum

    cdev_name = f"vfio{fallback_index}"
    class_devnum = _read_text(sysfs_root / "class" / "vfio-dev" / cdev_name / "dev")
    if class_devnum and _DEVNUM_RE.fullmatch(class_devnum):
        devnum = class_devnum
    else:
        devnum = None
    return cdev_name, devnum


def _pcie_root_complex(pci_bdf: str) -> str:
    """Return `<domain>:<bus>` PCIe root-complex prefix for a BDF (e.g. `0000:00` or `0000:80`)."""
    parts = pci_bdf.strip().split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}"
    return pci_bdf.strip()


def _parse_numa_node(pci_entry: Path) -> Optional[int]:
    raw = _read_text(pci_entry / "numa_node")
    if raw is None:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val if val >= 0 else None


def discover_tpu_chips(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    *,
    allow_missing_vfio: bool = False,
    iommu_dev_path: Optional[Path] = None,
) -> Tuple[List[DiscoveredChip], Path]:
    """Scan sysfs for VFIO/iommufd-backed Google 0x1ae0 TPU PCI devices and device nodes."""
    sysfs_root = Path(os.path.abspath(sysfs_root))
    dev_root = Path(os.path.abspath(dev_root))
    pci_dir = sysfs_root / "bus" / "pci" / "devices"
    if not pci_dir.is_dir():
        raise FileNotFoundError(f"PCI sysfs directory not found: {pci_dir}")

    effective_iommu = (
        Path(os.path.abspath(iommu_dev_path))
        if iommu_dev_path is not None
        else (dev_root / "iommu")
    )
    use_iommufd = effective_iommu.exists() or effective_iommu.is_symlink()

    if use_iommufd:
        control_node = effective_iommu
        if not allow_missing_vfio:
            actual_iommu_dn = _stat_char_devnum(effective_iommu)
            if actual_iommu_dn is None:
                raise RuntimeError(
                    f"iommufd control node is not a character device: {effective_iommu}"
                )
            kernel_iommu_dn = _read_kernel_iommu_devnum(sysfs_root)
            if kernel_iommu_dn is not None and actual_iommu_dn != kernel_iommu_dn:
                raise RuntimeError(
                    f"iommufd control node {effective_iommu} device number {actual_iommu_dn!r} "
                    f"does not match kernel sysfs misc/iommu/dev {kernel_iommu_dn!r}"
                )
    else:
        control_node = dev_root / "vfio" / "vfio"
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
            kernel_container_dn = _read_kernel_vfio_devnum(sysfs_root, group=None)
            if kernel_container_dn is None:
                raise RuntimeError(
                    f"Unable to verify kernel device number for VFIO control node {control_node} "
                    f"(missing {sysfs_root}/class/misc/vfio/dev and {sysfs_root}/devices/virtual/misc/vfio/dev)"
                )
            if actual_container_dn != kernel_container_dn:
                raise RuntimeError(
                    f"VFIO control node {control_node} device number {actual_container_dn!r} "
                    f"does not match kernel sysfs misc/vfio/dev {kernel_container_dn!r}"
                )

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
        pcie_root = _pcie_root_complex(entry.name)

        if use_iommufd:
            cdev_name, kernel_cdev_dn = _resolve_iommufd_cdev(
                entry, sysfs_root, len(chips)
            )
            vfio_host_path = dev_root / "vfio" / "devices" / cdev_name
            vfio_container_path = f"/dev/vfio/devices/{cdev_name}"
            vfio_mode = "iommufd"
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
                if kernel_cdev_dn is not None and actual_cdev_dn != kernel_cdev_dn:
                    raise RuntimeError(
                        f"iommufd VFIO cdev {vfio_host_path} for {entry.name} has device number "
                        f"{actual_cdev_dn!r}, which does not match kernel sysfs {cdev_name}/dev {kernel_cdev_dn!r}"
                    )
        else:
            vfio_host_path = dev_root / "vfio" / iommu_group
            vfio_container_path = f"/dev/vfio/{iommu_group}"
            vfio_mode = "vfio-group"
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
                vfio_mode=vfio_mode,
                iommufd_cdev=iommufd_cdev,
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


def _chips_per_host_bounds(chip_count: int, generations: Sequence[str]) -> str:
    """Return `TPU_CHIPS_PER_HOST_BOUNDS` (`X,Y,Z`) matching the slice size and generation."""
    if chip_count <= 1:
        return "1,1,1"
    if chip_count == 2:
        return "1,2,1"
    if chip_count == 4:
        return "2,2,1"
    if chip_count == 8:
        if any(g in ("v4", "v4lite", "v5p") for g in generations):
            return "2,2,2"
        return "2,4,1"
    return f"1,{chip_count},1"


def _build_device_env(
    chip_indices: Sequence[int],
    generations: Sequence[str],
    *,
    network_mode: str = "isolated",
) -> List[str]:
    """Build per-device CDI `containerEdits.env` entries."""
    visible = ",".join(str(idx) for idx in chip_indices)
    bounds = _chips_per_host_bounds(len(chip_indices), generations)
    env = [
        f"TPU_VISIBLE_DEVICES={visible}",
        f"TPU_CHIPS_PER_HOST_BOUNDS={bounds}",
    ]
    if network_mode == "isolated":
        env.extend(
            [
                "MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353",
                "TPU_HOST_BOUNDS=1,1,1",
                "TPU_SKIP_MDS_QUERY=1",
            ]
        )
    return env


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
    network_mode: str = "isolated",
) -> Dict[str, Any]:
    """Build a CNCF CDI v1.1.0 JSON specification dictionary."""
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

    is_iommufd = any(c.vfio_mode == "iommufd" for c in chips) or (
        vfio_container_host_path.name == "iommu"
    )
    control_container_path = "/dev/iommu" if is_iommufd else "/dev/vfio/vfio"
    container_devnum = _stat_char_devnum(vfio_container_host_path)
    global_edits: Dict[str, Any] = {
        "deviceNodes": [
            _make_device_node(control_container_path, vfio_container_host_path),
        ],
        "env": [
            f"FREETPU_CDI_VERSION={CDI_VERSION}",
            f"FREETPU_IOMMU_MODE={'iommufd' if is_iommufd else 'vfio-group'}",
            f"FREETPU_NETWORK_MODE={network_mode}",
            f"FREETPU_VFIO_CONTAINER={control_container_path}",
        ],
    }

    devices: List[Dict[str, Any]] = []
    seen_group_nodes: Dict[str, Dict[str, Any]] = {}
    seen_group_devnums: Dict[str, Optional[str]] = {}
    units: Dict[str, List[DiscoveredChip]] = {}
    all_bdfs: List[str] = []
    all_generations: List[str] = []

    for chip in chips:
        unit_key = chip.pci_bdf if is_iommufd else chip.iommu_group
        units.setdefault(unit_key, []).append(chip)
        all_bdfs.append(chip.pci_bdf)
        if chip.generation not in all_generations:
            all_generations.append(chip.generation)

    for _unit_key, unit_chips in units.items():
        rep = unit_chips[0]
        node = _make_device_node(rep.vfio_container_path, rep.vfio_host_path)
        seen_group_nodes.setdefault(rep.vfio_container_path, node)
        group_devnum = _stat_char_devnum(rep.vfio_host_path)
        seen_group_devnums.setdefault(rep.vfio_container_path, group_devnum)
        unit_bdfs = sorted(c.pci_bdf for c in unit_chips)
        unit_gens: List[str] = []
        for c in unit_chips:
            if c.generation not in unit_gens:
                unit_gens.append(c.generation)
        unit_env = _build_device_env(
            [c.index for c in unit_chips],
            unit_gens,
            network_mode=network_mode,
        )
        pcie_root = rep.pcie_root or _pcie_root_complex(rep.pci_bdf)

        if len(unit_chips) == 1:
            selector = _single_chip_selector(rep.pci_bdf)
            ann: Dict[str, str] = {
                "freetpu.dev/allocation-unit": "single-chip",
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
            devnums_str = _format_vfio_devnums([(rep.vfio_container_path, group_devnum)])
            if devnums_str:
                ann["freetpu.dev/vfio-devnums"] = devnums_str
            devices.append(
                {
                    "name": selector,
                    "annotations": ann,
                    "containerEdits": {
                        "deviceNodes": [node],
                        "env": unit_env,
                    },
                }
            )
        else:
            # Expose shared IOMMU groups strictly as a single hardware-backed
            # allocation unit so callers never assume device-level VFIO isolation
            # between sibling chips and selectors never shift onto different chips.
            selector = _shared_group_selector(unit_bdfs)
            ann = {
                "freetpu.dev/allocation-unit": "shared-iommu-group",
                "freetpu.dev/shared-iommu-group": "true",
                "freetpu.dev/chip-count": str(len(unit_chips)),
                "freetpu.dev/pci-bdfs": ",".join(unit_bdfs),
                "freetpu.dev/generations": ",".join(unit_gens),
                "freetpu.dev/iommu-group": rep.iommu_group,
                "freetpu.dev/pcie-root": pcie_root,
                "freetpu.dev/optional-cap-add": "IPC_LOCK",
                "freetpu.dev/required-ulimit-memlock": "-1",
            }
            if rep.numa_node is not None:
                ann["freetpu.dev/numa-node"] = str(rep.numa_node)
            devnums_str = _format_vfio_devnums([(rep.vfio_container_path, group_devnum)])
            if devnums_str:
                ann["freetpu.dev/vfio-devnums"] = devnums_str
            devices.append(
                {
                    "name": selector,
                    "annotations": ann,
                    "containerEdits": {
                        "deviceNodes": [node],
                        "env": unit_env,
                    },
                }
            )

    # Emit NUMA / PCIe-root-complex-aware sub-host slice presets (e.g. `numa0`,
    # `numa1`, `0-3`, `4-7` on `v6e-8`) strictly within matching NUMA / PCIe root
    # complex boundaries so host-to-device DMA never crosses UPI/QPI.
    domains: Dict[Tuple[str, Optional[int], str], List[DiscoveredChip]] = {}
    for chip in chips:
        domains.setdefault(_topology_domain_key(chip), []).append(chip)

    if len(domains) >= 2:
        used_names = {d["name"] for d in devices}
        explicit_numa_ids = [
            key[1] for key in domains if key[0] == "numa" and key[1] is not None
        ]
        use_explicit_numa = (
            len(explicit_numa_ids) == len(domains)
            and len(set(explicit_numa_ids)) == len(domains)
        )
        # Map each legacy IOMMU group to the set of domain keys that touch it so
        # we never emit a sub-host slice that shares a legacy group with another domain.
        group_to_domains: Dict[str, set[Tuple[str, Optional[int], str]]] = {}
        for d_key, d_chips in domains.items():
            for c in d_chips:
                group_to_domains.setdefault(c.iommu_group, set()).add(d_key)

        for dom_ordinal, (d_key, d_chips) in enumerate(domains.items()):
            if len(d_chips) < 2 or len(d_chips) >= len(chips):
                continue
            if not is_iommufd and any(
                len(group_to_domains.get(c.iommu_group, set())) > 1 for c in d_chips
            ):
                continue

            slice_nodes, slice_devnums = _dedup_device_nodes(d_chips)
            slice_bdfs = sorted(c.pci_bdf for c in d_chips)
            slice_indices = [c.index for c in d_chips]
            slice_gens: List[str] = []
            for c in d_chips:
                if c.generation not in slice_gens:
                    slice_gens.append(c.generation)
            slice_env = _build_device_env(
                slice_indices,
                slice_gens,
                network_mode=network_mode,
            )
            numa_label = str(d_key[1]) if use_explicit_numa and d_key[1] is not None else str(dom_ordinal)
            pcie_root = d_key[2]
            slice_ann_base: Dict[str, str] = {
                "freetpu.dev/allocation-unit": "numa-slice",
                "freetpu.dev/chip-count": str(len(d_chips)),
                "freetpu.dev/pci-bdfs": ",".join(slice_bdfs),
                "freetpu.dev/generations": ",".join(slice_gens),
                "freetpu.dev/numa-node": numa_label,
                "freetpu.dev/pcie-root": pcie_root,
                "freetpu.dev/optional-cap-add": "IPC_LOCK",
                "freetpu.dev/required-ulimit-memlock": "-1",
            }
            slice_devnums_str = _format_vfio_devnums(list(slice_devnums.items()))
            if slice_devnums_str:
                slice_ann_base["freetpu.dev/vfio-devnums"] = slice_devnums_str

            candidate_names = [f"numa{numa_label}"]
            if slice_indices == list(range(slice_indices[0], slice_indices[0] + len(slice_indices))):
                candidate_names.append(f"{slice_indices[0]}-{slice_indices[-1]}")

            for s_name in candidate_names:
                if s_name in used_names or s_name == "all":
                    continue
                used_names.add(s_name)
                devices.append(
                    {
                        "name": s_name,
                        "annotations": dict(slice_ann_base),
                        "containerEdits": {
                            "deviceNodes": [dict(n) for n in slice_nodes],
                            "env": list(slice_env),
                        },
                    }
                )

    all_ann: Dict[str, str] = {
        "freetpu.dev/allocation-unit": "all-devices",
        "freetpu.dev/chip-count": str(len(chips)),
        "freetpu.dev/pci-bdfs": ",".join(sorted(all_bdfs)),
        "freetpu.dev/generations": ",".join(all_generations),
        "freetpu.dev/optional-cap-add": "IPC_LOCK",
        "freetpu.dev/required-ulimit-memlock": "-1",
    }
    all_devnums = _format_vfio_devnums(list(seen_group_devnums.items()))
    if all_devnums:
        all_ann["freetpu.dev/vfio-devnums"] = all_devnums

    devices.append(
        {
            "name": "all",
            "annotations": all_ann,
            "containerEdits": {
                "deviceNodes": list(seen_group_nodes.values()),
                "env": _build_device_env(
                    [c.index for c in chips],
                    all_generations,
                    network_mode=network_mode,
                ),
            },
        }
    )

    root_ann: Dict[str, str] = {
        "freetpu.dev/generated-by": "freetpu-cdi",
        "freetpu.dev/iommu-mode": "iommufd" if is_iommufd else "vfio-group",
        "freetpu.dev/network-mode": network_mode,
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
        "cdiVersion": CDI_VERSION,
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


def _validate_container_edits(edits: Any, context: str, errors: List[str]) -> None:
    if not isinstance(edits, dict):
        errors.append(f"{context}: must be a JSON object")
        return

    unknown_keys = set(edits.keys()) - _ALLOWED_CONTAINER_EDITS_KEYS
    if unknown_keys:
        errors.append(
            f"{context}: unsupported CDI v{CDI_VERSION} containerEdits keys: {sorted(unknown_keys)}"
        )

    if "env" in edits:
        env = edits["env"]
        if not isinstance(env, list) or not all(
            isinstance(x, str) and "=" in x and bool(x.split("=", 1)[0]) for x in env
        ):
            errors.append(f"{context}.env: must be a list of non-empty KEY=VALUE strings")

    if "additionalGids" in edits:
        gids = edits["additionalGids"]
        if not isinstance(gids, list) or not all(
            isinstance(g, int) and not isinstance(g, bool) and 0 <= g <= 0xFFFFFFFF
            for g in gids
        ):
            errors.append(f"{context}.additionalGids: must be a list of unsigned 32-bit integers")

    if "intelRdt" in edits:
        irdt = edits["intelRdt"]
        if not isinstance(irdt, dict):
            errors.append(f"{context}.intelRdt: must be a JSON object")
        else:
            unknown_irdt = set(irdt.keys()) - _ALLOWED_INTEL_RDT_KEYS
            if unknown_irdt:
                errors.append(
                    f"{context}.intelRdt: unsupported keys: {sorted(unknown_irdt)}"
                )

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
                        f"{nc}: unsupported CDI v{CDI_VERSION} deviceNode keys: {sorted(unknown_node_keys)}"
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
                        f"{mc}: unsupported CDI v{CDI_VERSION} mount keys: {sorted(unknown_mount_keys)}"
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
                        f"{hc}: unsupported CDI v{CDI_VERSION} hook keys: {sorted(unknown_hook_keys)}"
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
    """Validate a parsed CDI JSON object against CNCF CDI v1.1.0 and FreeTPU VFIO/iommufd rules."""
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ["CDI spec root must be a JSON object"]

    unknown_root = set(spec.keys()) - _ALLOWED_SPEC_ROOT_KEYS
    if unknown_root:
        errors.append(f"unsupported CDI v{CDI_VERSION} root keys: {sorted(unknown_root)}")

    if spec.get("cdiVersion") != CDI_VERSION:
        errors.append(f"cdiVersion must be {CDI_VERSION!r}, got {spec.get('cdiVersion')!r}")

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
        _validate_container_edits(spec["containerEdits"], "containerEdits", errors)
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

    seen_names: set[str] = set()
    for idx, dev in enumerate(devices):
        dc = f"devices[{idx}]"
        if not isinstance(dev, dict):
            errors.append(f"{dc}: must be a JSON object")
            continue
        unknown_dev = set(dev.keys()) - _ALLOWED_DEVICE_ENTRY_KEYS
        if unknown_dev:
            errors.append(f"{dc}: unsupported CDI v{CDI_VERSION} device keys: {sorted(unknown_dev)}")
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
            _validate_container_edits(dev["containerEdits"], f"{dc}.containerEdits", errors)
            if isinstance(dev["containerEdits"], dict):
                edits_dict = dev["containerEdits"]
                has_any_edit = any(
                    (isinstance(edits_dict.get(k), list) and len(edits_dict[k]) > 0)
                    or (k == "intelRdt" and isinstance(edits_dict.get(k), dict) and len(edits_dict[k]) > 0)
                    for k in ("env", "deviceNodes", "hooks", "mounts", "additionalGids", "intelRdt")
                )
                if not has_any_edit:
                    errors.append(
                        f"{dc}.containerEdits: must contain at least one edit "
                        "(CDI v1.1.0 rejects empty per-device edits)"
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
            if not any(p.startswith("/dev/vfio/devices/vfio") for p in effective_paths):
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


def verify_spec_against_host(
    spec: Any,
    *,
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    iommu_dev_path: Optional[Path] = None,
) -> List[str]:
    """Revalidate all effective global and per-device CDI nodes against live host VFIO/iommufd state.

    Enforces exact allowlists on root, device, containerEdits, and deviceNode keys
    (blocking case-folded Go JSON decoder fields like `Major`, `HostPath`, `Hooks`,
    `Mounts`), verifies the exact host device CDI will inject (`hostPath` or fallback
    `path`), validates global `containerEdits.deviceNodes`, checks that every VFIO/iommufd
    unit still maps to the exact TPU PCI BDF(s) with zero non-TPU members, enforces
    NUMA / PCIe root-complex boundaries on sub-host slices, and flags changed char
    device numbers.
    """
    sysfs_root = Path(os.path.abspath(sysfs_root))
    dev_root = Path(os.path.abspath(dev_root))
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ["CDI spec root must be a JSON object"]

    unknown_root = sorted(set(spec.keys()) - _ALLOWED_SPEC_ROOT_KEYS)
    if unknown_root:
        errors.append(f"disallowed root keys in host-validated TPU spec: {unknown_root}")

    try:
        live_chips, vfio_container = discover_tpu_chips(
            sysfs_root=sysfs_root,
            dev_root=dev_root,
            allow_missing_vfio=False,
            iommu_dev_path=iommu_dev_path,
        )
    except Exception as exc:  # noqa: BLE001
        return [f"live host VFIO discovery failed: {exc}"]

    is_iommufd = any(c.vfio_mode == "iommufd" for c in live_chips)
    expected_control_container_path = "/dev/iommu" if is_iommufd else "/dev/vfio/vfio"
    live_container_resolved = str(vfio_container.resolve())
    live_by_host_resolved: Dict[str, Tuple[str, List[str], Optional[str]]] = {}
    live_devnums_by_container_path: Dict[str, str] = {}
    container_dn = _stat_char_devnum(vfio_container)
    if container_dn:
        live_devnums_by_container_path[expected_control_container_path] = container_dn

    live_chips_by_bdf = {c.pci_bdf: c for c in live_chips}
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
            elif name.startswith("numa") or _RANGE_SLICE_RE.match(name):
                slice_domains = {
                    _topology_domain_key(live_chips_by_bdf[b])
                    for b in expected_live_bdfs
                    if b in live_chips_by_bdf
                }
                if len(slice_domains) > 1:
                    errors.append(
                        f"device {name!r}: sub-host slice crosses NUMA / PCIe root complex boundaries ({expected_live_bdfs})"
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

    return errors


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
    network_mode: str = "isolated",
    iommu_dev_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Generate, validate, and atomically publish a live CDI spec, invalidating stale files on error."""
    if not isinstance(kind, str) or not _CDI_KIND_RE.match(kind):
        return False, f"Invalid CDI kind {kind!r} (existing spec preserved)"
    if network_mode not in NETWORK_MODES:
        return False, f"Invalid network_mode {network_mode!r} (existing spec preserved)"
    try:
        chips, vfio_container = discover_tpu_chips(
            sysfs_root=sysfs_root,
            dev_root=dev_root,
            allow_missing_vfio=False,
            iommu_dev_path=iommu_dev_path,
        )
        spec = build_cdi_spec(
            chips,
            vfio_container,
            kind=kind,
            network_mode=network_mode,
        )
    except Exception as exc:  # noqa: BLE001
        removed, cleanup_err = _invalidate_spec(output_path)
        suffix = _format_invalidation_suffix(output_path, removed, cleanup_err)
        return False, f"discovery failed: {exc}{suffix}"

    errors = validate_cdi_spec(spec)
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

    return True, f"updated CNCF CDI v{CDI_VERSION} spec ({len(chips)} chip(s), kind={kind}) at {output_path}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freetpu-cdi",
        description=f"FreeTPU CNCF CDI v{CDI_VERSION} spec generator & container toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help=(
            f"Write a point-in-time boot/topology CNCF CDI v{CDI_VERSION} JSON snapshot "
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
        help=f"Validate generated CDI spec against CNCF CDI v{CDI_VERSION} rules before emitting.",
    )
    gen.add_argument(
        "--kind",
        default=CDI_KIND,
        help=f"CDI vendor/class kind (default: {CDI_KIND}).",
    )
    gen.add_argument(
        "--network-mode",
        choices=sorted(NETWORK_MODES),
        default="isolated",
        help=(
            "Container networking mode (default: isolated, injecting single-host "
            "MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353, TPU_HOST_BOUNDS=1,1,1, "
            "and TPU_SKIP_MDS_QUERY=1 so --net=host is not required; use 'host' "
            "for multi-host slices)."
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
        help=f"Validate a CNCF CDI v{CDI_VERSION} JSON spec file (or '-' for stdin).",
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
        "--network-mode",
        choices=sorted(NETWORK_MODES),
        default="isolated",
        help="Container networking mode ('isolated' default, or 'host' for multi-host slices).",
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
            chips, vfio_container = discover_tpu_chips(
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
                allow_missing_vfio=args.allow_missing_vfio,
                iommu_dev_path=args.iommu_dev,
            )
            spec = build_cdi_spec(
                chips,
                vfio_container,
                kind=args.kind,
                network_mode=args.network_mode,
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
            f"Wrote CNCF CDI v{CDI_VERSION} spec ({len(chips)} chip(s), kind={args.kind}) to {args.output}\n"
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
        if not errors and args.check_host:
            errors.extend(
                verify_spec_against_host(
                    parsed,
                    sysfs_root=args.sysfs_root,
                    dev_root=args.dev_root,
                    iommu_dev_path=args.iommu_dev,
                )
            )
        if errors:
            if args.check_host and args.remove_if_stale and args.spec_path != "-":
                removed, cleanup_err = _invalidate_spec(Path(args.spec_path))
                if removed:
                    sys.stderr.write(f"Removed stale CDI spec at {args.spec_path}\n")
                elif cleanup_err:
                    sys.stderr.write(f"Failed to remove stale CDI spec at {args.spec_path}: {cleanup_err}\n")
            for err in errors:
                sys.stderr.write(f"CDI validation error: {err}\n")
            return 1
        suffix = " (host VFIO identity verified)" if args.check_host else ""
        sys.stdout.write(
            f"VALID CNCF CDI v{CDI_VERSION} spec ({parsed.get('kind')}, {len(parsed.get('devices', []))} device entries){suffix}\n"
        )
        return 0

    if args.command == "sync":
        if not isinstance(args.kind, str) or not _CDI_KIND_RE.match(args.kind):
            sys.stderr.write(f"freetpu-cdi sync failed: Invalid CDI kind {args.kind!r}\n")
            return 1
        if args.once:
            ok, msg = reconcile_cdi_spec(
                args.output,
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
                kind=args.kind,
                network_mode=args.network_mode,
                iommu_dev_path=args.iommu_dev,
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
                    iommu_dev_path=args.iommu_dev,
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
