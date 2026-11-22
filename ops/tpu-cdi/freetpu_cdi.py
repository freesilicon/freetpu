#!/usr/bin/env python3
"""FreeTPU Open CNCF CDI Spec Generator & Container Passthrough Toolkit.

Scans /sys/bus/pci/devices/* (vendor 0x1ae0) and /dev/vfio/* on VFIO-backed
Cloud TPU hosts (v5e, v5p, v6e Trillium bound to `vfio-pci`) to generate a
CNCF Container Device Interface (CDI) v0.6.0 specification (e.g.
/etc/cdi/freetpu.json) exposing `freetpu.dev/tpu=0`, ..., `freetpu.dev/tpu=all`.

Note: Legacy `/dev/accel*` Cloud TPU hosts (e.g. Cloud TPU v4 using Google's
`accel` driver) are not supported by VFIO CDI passthrough.

Also provides:
  - `validate`: strict CNCF CDI v0.6.0 structural & per-device VFIO validator
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
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

CDI_VERSION = "0.6.0"
CDI_KIND = "freetpu.dev/tpu"
GOOGLE_PCI_VENDOR_ID = "0x1ae0"
DEFAULT_OUTPUT_PATH = Path("/etc/cdi/freetpu.json")

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

# CNCF CDI v0.6.0 spec grammar rules.
_CDI_KIND_RE = re.compile(
    r"^[a-z0-9]+([.-][a-z0-9]+)*/[a-zA-Z0-9]+([._-][a-zA-Z0-9]+)*$"
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
    # Omit major and minor intentionally: Linux allocates VFIO group minor
    # numbers dynamically and independently of IOMMU group IDs, so CDI must
    # resolve live device numbers from hostPath at container injection time.
    return {
        "path": container_path,
        "hostPath": str(host_path),
        "type": "c",
        "permissions": permissions,
    }


def _collect_iommu_group_members(
    sysfs_root: Path,
    pci_entries: Sequence[Path],
) -> Dict[str, set[str]]:
    """Map every IOMMU group ID to all PCI device BDFs belonging to that group."""
    members: Dict[str, set[str]] = {}
    for entry in pci_entries:
        grp = _read_symlink_name(entry / "iommu_group") or ""
        if grp:
            members.setdefault(grp, set()).add(entry.name)

    iommu_root = sysfs_root / "kernel" / "iommu_groups"
    if iommu_root.is_dir():
        for grp_dir in iommu_root.iterdir():
            dev_dir = grp_dir / "devices"
            if not dev_dir.is_dir():
                continue
            try:
                for member in dev_dir.iterdir():
                    members.setdefault(grp_dir.name, set()).add(member.name)
            except OSError:
                continue
    return members


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
    if not vfio_container.exists() and not allow_missing_vfio:
        raise FileNotFoundError(
            f"VFIO control device not found: {vfio_container} "
            "(freetpu-cdi requires VFIO-backed Cloud TPUs using vfio-pci; "
            "/dev/accel* hosts such as Cloud TPU v4 are not supported)"
        )

    pci_entries = sorted(pci_dir.iterdir(), key=lambda p: p.name)
    group_members = _collect_iommu_group_members(sysfs_root, pci_entries)

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

        all_in_group = group_members.get(iommu_group, set()) | {entry.name}
        non_tpu_members = tuple(sorted(all_in_group - tpu_bdfs))
        if non_tpu_members:
            raise RuntimeError(
                f"Refusing VFIO group {iommu_group} for TPU {entry.name}: "
                f"IOMMU group also exposes non-TPU PCI device(s): {', '.join(non_tpu_members)}"
            )

        vfio_host_path = dev_root / "vfio" / iommu_group
        if not vfio_host_path.exists() and not allow_missing_vfio:
            raise FileNotFoundError(
                f"VFIO group node not found for {entry.name}: {vfio_host_path}"
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


def build_cdi_spec(
    chips: Sequence[DiscoveredChip],
    vfio_container_host_path: Path,
    *,
    kind: str = CDI_KIND,
) -> Dict[str, Any]:
    """Build a CNCF CDI v0.6.0 JSON specification dictionary."""
    if not chips:
        raise ValueError("Cannot build CDI spec with zero discovered TPU chips")

    for chip in chips:
        if chip.non_tpu_group_members:
            raise ValueError(
                f"Refusing VFIO group {chip.iommu_group} for TPU {chip.pci_bdf}: "
                f"IOMMU group contains non-TPU device(s): {', '.join(chip.non_tpu_group_members)}"
            )

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
        group_bdfs = [c.pci_bdf for c in group_chips]
        group_gens: List[str] = []
        for c in group_chips:
            if c.generation not in group_gens:
                group_gens.append(c.generation)

        if len(group_chips) == 1:
            devices.append(
                {
                    "name": str(rep.index),
                    "annotations": {
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
                    },
                    "containerEdits": {
                        "deviceNodes": [node],
                        "env": [
                            f"FREETPU_VISIBLE_DEVICES={rep.index}",
                            f"FREETPU_PCI_BDFS={rep.pci_bdf}",
                            f"FREETPU_TPU_GENERATION={rep.generation}",
                            f"FREETPU_VFIO_DEVICES={rep.vfio_container_path}",
                        ],
                    },
                }
            )
        else:
            # Expose shared IOMMU groups strictly as a single allocation unit so
            # callers never assume device-level VFIO isolation between sibling chips.
            devices.append(
                {
                    "name": f"group-{iommu_group}",
                    "annotations": {
                        "freetpu.dev/allocation-unit": "shared-iommu-group",
                        "freetpu.dev/shared-iommu-group": "true",
                        "freetpu.dev/chip-count": str(len(group_chips)),
                        "freetpu.dev/chip-indices": ",".join(str(c.index) for c in group_chips),
                        "freetpu.dev/pci-bdfs": ",".join(group_bdfs),
                        "freetpu.dev/generations": ",".join(group_gens),
                        "freetpu.dev/iommu-group": iommu_group,
                        "freetpu.dev/required-cap-add": "IPC_LOCK",
                        "freetpu.dev/required-ulimit-memlock": "-1",
                    },
                    "containerEdits": {
                        "deviceNodes": [node],
                        "env": [
                            f"FREETPU_VISIBLE_DEVICES={','.join(str(c.index) for c in group_chips)}",
                            f"FREETPU_CHIP_COUNT={len(group_chips)}",
                            f"FREETPU_PCI_BDFS={','.join(group_bdfs)}",
                            f"FREETPU_TPU_GENERATION={','.join(group_gens)}",
                            f"FREETPU_VFIO_DEVICES={rep.vfio_container_path}",
                        ],
                    },
                }
            )

    devices.append(
        {
            "name": "all",
            "annotations": {
                "freetpu.dev/allocation-unit": "all-devices",
                "freetpu.dev/chip-count": str(len(chips)),
                "freetpu.dev/pci-bdfs": ",".join(all_bdfs),
                "freetpu.dev/generations": ",".join(all_generations),
                "freetpu.dev/required-cap-add": "IPC_LOCK",
                "freetpu.dev/required-ulimit-memlock": "-1",
            },
            "containerEdits": {
                "deviceNodes": list(seen_group_nodes.values()),
                "env": [
                    f"FREETPU_VISIBLE_DEVICES={','.join(str(c.index) for c in chips)}",
                    f"FREETPU_CHIP_COUNT={len(chips)}",
                    f"FREETPU_PCI_BDFS={','.join(all_bdfs)}",
                    f"FREETPU_TPU_GENERATION={','.join(all_generations)}",
                    f"FREETPU_VFIO_DEVICES={','.join(seen_group_nodes.keys())}",
                ],
            },
        }
    )

    return {
        "cdiVersion": CDI_VERSION,
        "kind": kind,
        "annotations": {
            "freetpu.dev/generated-by": "freetpu-cdi",
            "freetpu.dev/required-cap-add": "IPC_LOCK",
            "freetpu.dev/required-ulimit-memlock": "-1",
        },
        "containerEdits": global_edits,
        "devices": devices,
    }


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freetpu-cdi",
        description="FreeTPU CNCF CDI v0.6.0 spec generator & container toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help="Scan PCI 0x1ae0 + /dev/vfio/* and emit CNCF CDI v0.6.0 JSON.",
    )
    gen.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CDI JSON path (default: {DEFAULT_OUTPUT_PATH}).",
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
        help="Do not fail if /dev/vfio nodes are absent (dry-run inspection only).",
    )

    val = sub.add_parser(
        "validate",
        help="Validate a CNCF CDI v0.6.0 JSON spec file (or '-' for stdin).",
    )
    val.add_argument(
        "spec_path",
        help="Path to CDI JSON file, or '-' to read from stdin.",
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
            sys.stderr.write(f"freetpu-cdi generate failed: {exc}\n")
            return 1

        if args.validate:
            errors = validate_cdi_spec(spec)
            if errors:
                for err in errors:
                    sys.stderr.write(f"CDI validation error: {err}\n")
                return 1

        rendered = json.dumps(spec, indent=2) + "\n"
        if args.stdout:
            sys.stdout.write(rendered)
            return 0

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
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
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"Invalid JSON: {exc}\n")
            return 1
        errors = validate_cdi_spec(parsed)
        if errors:
            for err in errors:
                sys.stderr.write(f"CDI validation error: {err}\n")
            return 1
        sys.stdout.write(
            f"VALID CNCF CDI v{CDI_VERSION} spec ({parsed.get('kind')}, {len(parsed.get('devices', []))} device entries)\n"
        )
        return 0

    if args.command == "docker-args":
        sys.stdout.write(
            f"--device={CDI_KIND}={args.device} --cap-add=IPC_LOCK --ulimit memlock=-1:-1\n"
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
