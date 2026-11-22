#!/usr/bin/env python3
"""Deterministic offline self-tests for ops/tpu-cdi/freetpu_cdi.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CDI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CDI_DIR))

import freetpu_cdi  # noqa: E402


def _make_mock_pci_vfio(
    root: Path,
    devices: list[dict[str, str]],
    *,
    create_vfio_container: bool = True,
    create_vfio_nodes: list[str] | None = None,
) -> tuple[Path, Path]:
    sysfs_root = root / "sys"
    dev_root = root / "dev"

    pci_devices = sysfs_root / "bus" / "pci" / "devices"
    pci_devices.mkdir(parents=True, exist_ok=True)
    iommu_groups = sysfs_root / "kernel" / "iommu_groups"
    iommu_groups.mkdir(parents=True, exist_ok=True)

    vfio_dir = dev_root / "vfio"
    vfio_dir.mkdir(parents=True, exist_ok=True)
    if create_vfio_container:
        (vfio_dir / "vfio").write_text("", encoding="utf-8")
    for grp in create_vfio_nodes or []:
        (vfio_dir / grp).write_text("", encoding="utf-8")

    for dev in devices:
        d = pci_devices / dev["bdf"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "vendor").write_text(dev["vendor"] + "\n", encoding="utf-8")
        (d / "device").write_text(dev["device"] + "\n", encoding="utf-8")
        if dev.get("iommu_group"):
            grp = iommu_groups / dev["iommu_group"]
            grp.mkdir(parents=True, exist_ok=True)
            os.symlink(str(grp), d / "iommu_group")
        if dev.get("driver"):
            drv = sysfs_root / "bus" / "pci" / "drivers" / dev["driver"]
            drv.mkdir(parents=True, exist_ok=True)
            os.symlink(str(drv), d / "driver")

    return sysfs_root, dev_root


class FreeTpuCdiTests(unittest.TestCase):
    def test_generate_single_v5e_spec_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:04.0",
                        "vendor": "0x1af4",
                        "device": "0x1041",
                        "driver": "virtio-pci",
                        "iommu_group": "9",
                    },
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                ],
                create_vfio_nodes=["0"],
            )

            chips, vfio_container = freetpu_cdi.discover_tpu_chips(
                sysfs_root=sysfs_root,
                dev_root=dev_root,
            )
            self.assertEqual(len(chips), 1)
            self.assertEqual(chips[0].generation, "v5e")

            spec = freetpu_cdi.build_cdi_spec(chips, vfio_container)
            errors = freetpu_cdi.validate_cdi_spec(spec)
            self.assertEqual(errors, [])

            self.assertEqual(spec["cdiVersion"], "0.6.0")
            self.assertEqual(spec["kind"], "freetpu.dev/tpu")
            names = [d["name"] for d in spec["devices"]]
            self.assertEqual(names, ["0", "all"])
            self.assertEqual(
                spec["containerEdits"]["deviceNodes"][0]["path"], "/dev/vfio/vfio"
            )
            self.assertEqual(
                spec["devices"][0]["containerEdits"]["deviceNodes"][0]["path"],
                "/dev/vfio/0",
            )

    def test_generate_multi_chip_v6e_deduplicates_all_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",  # shared IOMMU group
                    },
                    {
                        "bdf": "0000:00:07.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                    },
                ],
                create_vfio_nodes=["0", "1"],
            )

            out_file = root / "etc" / "cdi" / "freetpu.json"
            script = CDI_DIR / "freetpu_cdi.py"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "generate",
                    "--output",
                    str(out_file),
                    "--validate",
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                    "--with-oci-hook",
                    "/usr/local/bin/freetpu-cdi",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            spec = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(freetpu_cdi.validate_cdi_spec(spec), [])
            names = [d["name"] for d in spec["devices"]]
            self.assertEqual(names, ["0", "1", "2", "all"])
            all_nodes = [
                n["path"] for n in spec["devices"][-1]["containerEdits"]["deviceNodes"]
            ]
            self.assertEqual(all_nodes, ["/dev/vfio/0", "/dev/vfio/1"])
            self.assertEqual(
                spec["containerEdits"]["hooks"][0]["hookName"], "createContainer"
            )

            val_res = subprocess.run(
                [sys.executable, str(script), "validate", str(out_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("VALID CNCF CDI v0.6.0 spec", val_res.stdout)

    def test_validate_rejects_invalid_specs(self) -> None:
        bad_version = {"cdiVersion": "0.5.0", "kind": "freetpu.dev/tpu", "devices": []}
        errs = freetpu_cdi.validate_cdi_spec(bad_version)
        self.assertTrue(any("cdiVersion" in e for e in errs))

        bad_edits = {
            "cdiVersion": "0.6.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                "bogusKey": True,
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                }
            ],
        }
        errs2 = freetpu_cdi.validate_cdi_spec(bad_edits)
        self.assertTrue(any("bogusKey" in e for e in errs2))

    def test_missing_vfio_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_container=False,
                create_vfio_nodes=[],
            )
            with self.assertRaises(FileNotFoundError):
                freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

    def test_oci_hook_patches_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            config_path = bundle / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ociVersion": "1.1.0",
                        "process": {
                            "capabilities": {"bounding": ["CAP_NET_BIND_SERVICE"]},
                            "rlimits": [{"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            script = CDI_DIR / "freetpu_cdi.py"
            subprocess.run(
                [sys.executable, str(script), "oci-hook"],
                input=json.dumps({"bundle": str(bundle)}),
                text=True,
                check=True,
            )
            updated = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("bounding", "effective", "inheritable", "permitted"):
                self.assertIn("CAP_IPC_LOCK", updated["process"]["capabilities"][key])
            memlock = [
                r for r in updated["process"]["rlimits"] if r["type"] == "RLIMIT_MEMLOCK"
            ]
            self.assertEqual(memlock, [{"type": "RLIMIT_MEMLOCK", "hard": -1, "soft": -1}])


if __name__ == "__main__":
    unittest.main()
