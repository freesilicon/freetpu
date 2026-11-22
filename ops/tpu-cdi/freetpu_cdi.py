#!/usr/bin/env python3
"""FreeTPU Open CNCF CDI Spec Generator & Container Passthrough Toolkit.

Scans /sys/bus/pci/devices/* (vendor 0x1ae0) and /dev/vfio/* on VFIO-backed
Cloud TPU hosts (v5e, v5p, v6e Trillium bound to `vfio-pci`) to generate a
boot-scoped CNCF Container Device Interface (CDI) v0.6.0 specification
(default: /var/run/cdi/freetpu.json) exposing stable hardware-backed selectors
`freetpu.dev/tpu=pci-<bdf>`, `freetpu.dev/tpu=group-pci-...`, and
`freetpu.dev/tpu=all`.

Note: Legacy `/dev/accel*` Cloud TPU hosts (e.g. Cloud TPU v4 using Google's
`accel` driver) are not supported by VFIO CDI passthrough.

Also provides:
  - `validate`: strict CNCF CDI v0.6.0 structural & live host VFIO validator
  - `sync`: boot-scoped generator/reconciler that invalidates stale specs on drift
  - `docker-args`: prints Docker / Podman CLI flags (including IPC_LOCK + memlock)

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

CDI_VERSION = "0.6.0"
CDI_KIND = "freetpu.dev/tpu"
GOOGLE_PCI_VENDOR_ID = "0x1ae0"
# Boot-scoped CNCF CDI dynamic spec directory (/var/run is tmpfs on Linux so
# stale IOMMU group assignments never persist across reboots).
DEFAULT_OUTPUT_PATH = Path("/var/run/cdi/freetpu.json")

# Known Google Cloud TPU PCI device IDs under vendor 0x1ae0.
TPU_PCI_DEVICES: Dict[str, Tuple[str, str]] = {
    "0x0027": ("v2-v3", "TPU v2/v3"),
    "0x0056": ("v4lite", "TPU v4lite"),
    "0x005e": ("v4", "TPU v4"),
    "0x0062": ("v5p", "TPU v5p"),
    "0x0063": ("v5e", "TPU v5e"),
    "0x006e": ("v6e-pf", "TPU v6e (PF)"),
    "0x006f": ("v6e", "TPU v6e (Trillium)"),
}

# CNCF CDI v0.6.0 / v0.8.0 parser grammar (ValidateVendorName / ValidateClassName):
# vendor and class must each start with an ASCII letter, contain only
# [A-Za-z0-9_.-], and end with an ASCII letter or digit.
_CDI_KIND_RE = re.compile(
    r"^[A-Za-z](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/[A-Za-z](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$"
)
_CDI_DEVICE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_ALLOWED_CONTAINER_EDITS_KEYS = frozenset(
    {"env", "deviceNodes", "hooks", "mounts"}
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
        "hostPath": str(host_path),
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


def discover_tpu_chips(
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    *,
    allow_missing_vfio: bool = False,
) -> Tuple[List[DiscoveredChip], Path]:
    """Scan sysfs for VFIO-backed Google 0x1ae0 TPU PCI devices and /dev/vfio/* nodes."""
    pci_dir = sysfs_root / "bus" / "pci" / "devices"
    if not pci_dir.is_dir():
        raise FileNotFoundError(f"PCI sysfs directory not found: {pci_dir}")

    vfio_container = dev_root / "vfio" / "vfio"
    if not allow_missing_vfio:
        if not vfio_container.exists():
            raise FileNotFoundError(
                f"VFIO control device not found: {vfio_container} "
                "(freetpu-cdi requires VFIO-backed Cloud TPUs using vfio-pci; "
                "/dev/accel* hosts such as Cloud TPU v4 are not supported)"
            )
        actual_container_dn = _stat_char_devnum(vfio_container)
        if actual_container_dn is None:
            raise RuntimeError(
                f"VFIO control node is not a character device: {vfio_container}"
            )
        kernel_container_dn = _read_kernel_vfio_devnum(sysfs_root, group=None)
        if kernel_container_dn is None:
            raise RuntimeError(
                f"Unable to verify kernel device number for VFIO control node {vfio_container} "
                f"(missing {sysfs_root}/class/misc/vfio/dev and {sysfs_root}/devices/virtual/misc/vfio/dev)"
            )
        if actual_container_dn != kernel_container_dn:
            raise RuntimeError(
                f"VFIO control node {vfio_container} device number {actual_container_dn!r} "
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

        vfio_host_path = dev_root / "vfio" / iommu_group
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
                vfio_container_path=f"/dev/vfio/{iommu_group}",
                non_tpu_group_members=non_tpu_members,
            )
        )

    if not chips:
        raise RuntimeError(
            f"No known Google TPU PCI devices (vendor={GOOGLE_PCI_VENDOR_ID}) found under {pci_dir}"
        )

    return chips, vfio_container


def _pci_bdf_token(pci_bdf: str) -> str:
    """Convert a PCI BDF (e.g. 0000:00:06.0) into a CDI-safe hardware token."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", pci_bdf.strip()).strip("-")


def _single_chip_selector(pci_bdf: str) -> str:
    return f"pci-{_pci_bdf_token(pci_bdf)}"


def _shared_group_selector(pci_bdfs: Sequence[str]) -> str:
    tokens = [_pci_bdf_token(b) for b in sorted(pci_bdfs)]
    return f"group-pci-{'_'.join(tokens)}"


def _stat_char_devnum(path: Path) -> Optional[str]:
    """Return '<major>:<minor>' if path is a character device, else None."""
    try:
        st = path.stat()
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


def build_cdi_spec(
    chips: Sequence[DiscoveredChip],
    vfio_container_host_path: Path,
    *,
    kind: str = CDI_KIND,
) -> Dict[str, Any]:
    """Build a CNCF CDI v0.6.0 JSON specification dictionary."""
    if not isinstance(kind, str) or not _CDI_KIND_RE.match(kind):
        raise ValueError(
            f"Invalid CDI kind {kind!r}: vendor and class must each start with an ASCII letter, "
            "contain only [A-Za-z0-9_.-], and end with an ASCII letter or digit"
        )
    if not chips:
        raise ValueError("Cannot build CDI spec with zero discovered TPU chips")

    for chip in chips:
        if chip.non_tpu_group_members:
            raise ValueError(
                f"Refusing VFIO group {chip.iommu_group} for TPU {chip.pci_bdf}: "
                f"IOMMU group contains non-TPU device(s): {', '.join(chip.non_tpu_group_members)}"
            )

    container_devnum = _stat_char_devnum(vfio_container_host_path)
    global_edits: Dict[str, Any] = {
        "deviceNodes": [
            _make_device_node("/dev/vfio/vfio", vfio_container_host_path),
        ],
        "env": [
            f"FREETPU_CDI_VERSION={CDI_VERSION}",
            "FREETPU_VFIO_CONTAINER=/dev/vfio/vfio",
        ],
    }

    devices: List[Dict[str, Any]] = []
    seen_group_nodes: Dict[str, Dict[str, Any]] = {}
    seen_group_devnums: Dict[str, Optional[str]] = {}
    groups: Dict[str, List[DiscoveredChip]] = {}
    all_bdfs: List[str] = []
    all_generations: List[str] = []

    for chip in chips:
        groups.setdefault(chip.iommu_group, []).append(chip)
        all_bdfs.append(chip.pci_bdf)
        if chip.generation not in all_generations:
            all_generations.append(chip.generation)

    for iommu_group, group_chips in groups.items():
        rep = group_chips[0]
        node = _make_device_node(rep.vfio_container_path, rep.vfio_host_path)
        seen_group_nodes.setdefault(rep.vfio_container_path, node)
        group_devnum = _stat_char_devnum(rep.vfio_host_path)
        seen_group_devnums.setdefault(rep.vfio_container_path, group_devnum)
        group_bdfs = sorted(c.pci_bdf for c in group_chips)
        group_gens: List[str] = []
        for c in group_chips:
            if c.generation not in group_gens:
                group_gens.append(c.generation)

        if len(group_chips) == 1:
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
                "freetpu.dev/required-cap-add": "IPC_LOCK",
                "freetpu.dev/required-ulimit-memlock": "-1",
            }
            devnums_str = _format_vfio_devnums([(rep.vfio_container_path, group_devnum)])
            if devnums_str:
                ann["freetpu.dev/vfio-devnums"] = devnums_str
            devices.append(
                {
                    "name": selector,
                    "annotations": ann,
                    "containerEdits": {
                        "deviceNodes": [node],
                        "env": [
                            f"FREETPU_CDI_DEVICE={selector}",
                            f"FREETPU_PCI_BDFS={rep.pci_bdf}",
                            f"FREETPU_TPU_GENERATION={rep.generation}",
                            f"FREETPU_VFIO_DEVICES={rep.vfio_container_path}",
                        ],
                    },
                }
            )
        else:
            # Expose shared IOMMU groups strictly as a single hardware-backed
            # allocation unit so callers never assume device-level VFIO isolation
            # between sibling chips and selectors never shift onto different chips.
            selector = _shared_group_selector(group_bdfs)
            ann = {
                "freetpu.dev/allocation-unit": "shared-iommu-group",
                "freetpu.dev/shared-iommu-group": "true",
                "freetpu.dev/chip-count": str(len(group_chips)),
                "freetpu.dev/pci-bdfs": ",".join(group_bdfs),
                "freetpu.dev/generations": ",".join(group_gens),
                "freetpu.dev/iommu-group": iommu_group,
                "freetpu.dev/required-cap-add": "IPC_LOCK",
                "freetpu.dev/required-ulimit-memlock": "-1",
            }
            devnums_str = _format_vfio_devnums([(rep.vfio_container_path, group_devnum)])
            if devnums_str:
                ann["freetpu.dev/vfio-devnums"] = devnums_str
            devices.append(
                {
                    "name": selector,
                    "annotations": ann,
                    "containerEdits": {
                        "deviceNodes": [node],
                        "env": [
                            f"FREETPU_CDI_DEVICE={selector}",
                            f"FREETPU_CHIP_COUNT={len(group_chips)}",
                            f"FREETPU_PCI_BDFS={','.join(group_bdfs)}",
                            f"FREETPU_TPU_GENERATION={','.join(group_gens)}",
                            f"FREETPU_VFIO_DEVICES={rep.vfio_container_path}",
                        ],
                    },
                }
            )

    all_ann: Dict[str, str] = {
        "freetpu.dev/allocation-unit": "all-devices",
        "freetpu.dev/chip-count": str(len(chips)),
        "freetpu.dev/pci-bdfs": ",".join(sorted(all_bdfs)),
        "freetpu.dev/generations": ",".join(all_generations),
        "freetpu.dev/required-cap-add": "IPC_LOCK",
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
                "env": [
                    "FREETPU_CDI_DEVICE=all",
                    f"FREETPU_CHIP_COUNT={len(chips)}",
                    f"FREETPU_PCI_BDFS={','.join(sorted(all_bdfs))}",
                    f"FREETPU_TPU_GENERATION={','.join(all_generations)}",
                    f"FREETPU_VFIO_DEVICES={','.join(seen_group_nodes.keys())}",
                ],
            },
        }
    )

    root_ann: Dict[str, str] = {
        "freetpu.dev/generated-by": "freetpu-cdi",
        "freetpu.dev/required-cap-add": "IPC_LOCK",
        "freetpu.dev/required-ulimit-memlock": "-1",
    }
    # Record live VFIO char major:minor in spec annotations (while keeping
    # deviceNodes free of pinned major/minor) so driver rebinds that change
    # dynamic VFIO minors produce a spec update that invalidates CDI's cache.
    root_devnums = _format_vfio_devnums(
        [("/dev/vfio/vfio", container_devnum), *seen_group_devnums.items()]
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
            f"{context}: unsupported CDI v0.6.0 containerEdits keys: {sorted(unknown_keys)}"
        )

    if "env" in edits:
        env = edits["env"]
        if not isinstance(env, list) or not all(isinstance(x, str) and "=" in x for x in env):
            errors.append(f"{context}.env: must be a list of KEY=VALUE strings")

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
                        f"{nc}: unsupported CDI v0.6.0 deviceNode keys: {sorted(unknown_node_keys)}"
                    )
                path = node.get("path")
                if not isinstance(path, str) or not path.startswith("/"):
                    errors.append(f"{nc}.path: must be an absolute path")
                if "hostPath" in node:
                    hp = node["hostPath"]
                    if not isinstance(hp, str) or not hp.startswith("/"):
                        errors.append(f"{nc}.hostPath: must be an absolute path")
                if "type" in node and node["type"] not in _ALLOWED_DEVICE_TYPES:
                    errors.append(f"{nc}.type: invalid device type {node['type']!r}")
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
                        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                            errors.append(f"{nc}.{field_name}: must be a non-negative integer")

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
                if not isinstance(m.get("hostPath"), str) or not m["hostPath"].startswith("/"):
                    errors.append(f"{mc}.hostPath: must be an absolute path")
                if (
                    not isinstance(m.get("containerPath"), str)
                    or not m["containerPath"].startswith("/")
                ):
                    errors.append(f"{mc}.containerPath: must be an absolute path")

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
                if h.get("hookName") not in _ALLOWED_HOOK_NAMES:
                    errors.append(f"{hc}.hookName: invalid OCI hookName {h.get('hookName')!r}")
                if not isinstance(h.get("path"), str) or not h["path"].startswith("/"):
                    errors.append(f"{hc}.path: must be an absolute executable path")


def validate_cdi_spec(spec: Any) -> List[str]:
    """Validate a parsed CDI JSON object against CNCF CDI v0.6.0 and FreeTPU VFIO rules."""
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ["CDI spec root must be a JSON object"]

    unknown_root = set(spec.keys()) - _ALLOWED_SPEC_ROOT_KEYS
    if unknown_root:
        errors.append(f"unsupported CDI v0.6.0 root keys: {sorted(unknown_root)}")

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
            errors.append(f"{dc}: unsupported CDI v0.6.0 device keys: {sorted(unknown_dev)}")
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
                nodes = dev["containerEdits"].get("deviceNodes")
                if isinstance(nodes, list):
                    for n in nodes:
                        if isinstance(n, dict) and isinstance(n.get("path"), str):
                            dev_paths.add(n["path"])

        effective_paths = global_paths | dev_paths
        label = name if isinstance(name, str) else dc
        if "/dev/vfio/vfio" not in effective_paths:
            errors.append(
                f"{dc} ({label!r}): effective containerEdits must expose /dev/vfio/vfio container node"
            )
        if not any(
            p.startswith("/dev/vfio/") and p != "/dev/vfio/vfio"
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
    """Enforce exact allowlist of deviceNode keys so case-folded Go JSON fields cannot bypass checks."""
    disallowed = sorted(set(node.keys()) - _HOST_VALIDATED_DEVICE_NODE_KEYS)
    if disallowed:
        errors.append(
            f"{context}: disallowed deviceNode keys in host-validated TPU spec: {disallowed} "
            "(explicit major/minor and non-canonical keys are rejected in --check-host mode)"
        )


def verify_spec_against_host(
    spec: Any,
    *,
    sysfs_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
) -> List[str]:
    """Revalidate all effective global and per-device CDI nodes against live host VFIO state.

    Enforces exact allowlists on root, device, containerEdits, and deviceNode keys
    (blocking case-folded Go JSON decoder fields like `Major`, `HostPath`, `Hooks`,
    `Mounts`), verifies the exact host device CDI will inject (`hostPath` or fallback
    `path`), validates global `containerEdits.deviceNodes`, checks that every VFIO group
    still maps to the exact TPU PCI BDF(s) with zero non-TPU members, and flags changed
    char device numbers.
    """
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
        )
    except Exception as exc:  # noqa: BLE001
        return [f"live host VFIO discovery failed: {exc}"]

    live_container_resolved = str(vfio_container.resolve())
    live_by_host_resolved: Dict[str, Tuple[str, List[str], Optional[str]]] = {}
    live_devnums_by_container_path: Dict[str, str] = {}
    container_dn = _stat_char_devnum(vfio_container)
    if container_dn:
        live_devnums_by_container_path["/dev/vfio/vfio"] = container_dn

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
            resolved_host = str(eff_host.resolve())
            if resolved_host != live_container_resolved or node.get("path") != "/dev/vfio/vfio":
                errors.append(
                    f"{nc}: global deviceNode exposes unauthorized host device {eff_host} "
                    f"(expected only {vfio_container} mapped to /dev/vfio/vfio)"
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
            str(_resolve_effective_host_path(gn).resolve()) == live_container_resolved
            and gn.get("path") == "/dev/vfio/vfio"
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
            resolved_host = str(eff_host.resolve())
            if resolved_host == live_container_resolved:
                if node.get("path") != "/dev/vfio/vfio":
                    errors.append(
                        f"{nc}: VFIO container {eff_host} must map to /dev/vfio/vfio, got {node.get('path')!r}"
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
                    f"{nc}: container path {node.get('path')!r} does not match host VFIO group {live_cp!r} for {eff_host}"
                )
            matched_group_paths.append(live_cp)
            expected_live_bdfs.extend(live_bdfs)

        if not has_container_node:
            errors.append(f"device {name!r}: effective deviceNodes missing live VFIO container {vfio_container}")
        if not matched_group_paths:
            errors.append(f"device {name!r}: no verified host /dev/vfio/<group> deviceNodes to verify")
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
) -> Tuple[bool, str]:
    """Generate, validate, and atomically publish a live CDI spec, invalidating stale files on error."""
    try:
        chips, vfio_container = discover_tpu_chips(
            sysfs_root=sysfs_root,
            dev_root=dev_root,
            allow_missing_vfio=False,
        )
        spec = build_cdi_spec(chips, vfio_container, kind=kind)
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
        description="FreeTPU CNCF CDI v0.6.0 spec generator & container toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help=(
            "Write a point-in-time boot/topology CNCF CDI v0.6.0 JSON snapshot "
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
        help="Validate generated CDI spec against CNCF CDI v0.6.0 rules before emitting.",
    )
    gen.add_argument(
        "--kind",
        default=CDI_KIND,
        help=f"CDI vendor/class kind (default: {CDI_KIND}).",
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
        "--allow-missing-vfio",
        action="store_true",
        help="Do not fail if /dev/vfio nodes are absent (dry-run inspection only; requires --stdout).",
    )

    val = sub.add_parser(
        "validate",
        help="Validate a CNCF CDI v0.6.0 JSON spec file (or '-' for stdin).",
    )
    val.add_argument(
        "spec_path",
        help="Path to CDI JSON file, or '-' to read from stdin.",
    )
    val.add_argument(
        "--check-host",
        action="store_true",
        help="Also revalidate every VFIO group's PCI BDF identity and isolation against live sysfs.",
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

    sync = sub.add_parser(
        "sync",
        help=(
            "Continuously reconcile a boot-scoped CDI spec against live VFIO topology, "
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
        "--kind",
        default=CDI_KIND,
        help=f"CDI vendor/class kind (default: {CDI_KIND}).",
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

    dargs = sub.add_parser(
        "docker-args",
        help="Print recommended Docker / Podman CLI flags for FreeTPU CDI passthrough.",
    )
    dargs.add_argument(
        "--device",
        default="all",
        help="CDI device selector suffix (default: all -> freetpu.dev/tpu=all).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
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
            )
            spec = build_cdi_spec(
                chips,
                vfio_container,
                kind=args.kind,
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
        if args.once:
            ok, msg = reconcile_cdi_spec(
                args.output,
                sysfs_root=args.sysfs_root,
                dev_root=args.dev_root,
                kind=args.kind,
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
        return 0

    if args.command == "docker-args":
        sys.stdout.write(
            f"--device={CDI_KIND}={args.device} --cap-add=IPC_LOCK --ulimit memlock=-1:-1\n"
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
