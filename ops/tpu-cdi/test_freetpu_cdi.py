#!/usr/bin/env python3
"""Deterministic offline self-tests for ops/tpu-cdi/freetpu_cdi.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

CDI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CDI_DIR))

import freetpu_cdi  # noqa: E402


@contextlib.contextmanager
def _mock_vfio_char_devices():
    """Patch Path.stat/lstat so synthetic non-symlink `<dev_root>/vfio/*`, `vfio/devices/*`, and `iommu` behave as Linux char devices."""
    orig_stat = Path.stat
    orig_lstat = Path.lstat

    def _maybe_char(p: Path, st):
        if not stat.S_ISLNK(st.st_mode):
            if p.parent.name == "vfio":
                fake = mock.MagicMock()
                fake.st_mode = stat.S_IFCHR | 0o600
                if p.name == "vfio":
                    fake.st_rdev = os.makedev(10, 196)
                elif p.name.isdigit():
                    fake.st_rdev = os.makedev(242, int(p.name))
                else:
                    fake.st_rdev = os.makedev(242, 250)
                return fake
            if p.name == "iommu":
                fake = mock.MagicMock()
                fake.st_mode = stat.S_IFCHR | 0o600
                fake.st_rdev = os.makedev(10, 197)
                return fake
            if p.parent.name == "devices" and p.parent.parent.name == "vfio":
                fake = mock.MagicMock()
                fake.st_mode = stat.S_IFCHR | 0o600
                suffix = p.name.removeprefix("vfio")
                minor = int(suffix) if suffix.isdigit() else 251
                fake.st_rdev = os.makedev(243, minor)
                return fake
        return st

    def _char_stat(p: Path, *args, **kwargs):
        return _maybe_char(p, orig_stat(p, *args, **kwargs))

    def _char_lstat(p: Path, *args, **kwargs):
        return _maybe_char(p, orig_lstat(p, *args, **kwargs))

    with (
        mock.patch.object(Path, "stat", autospec=True, side_effect=_char_stat),
        mock.patch.object(Path, "lstat", autospec=True, side_effect=_char_lstat),
    ):
        yield


def _run_cli_inproc(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = freetpu_cdi.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _make_mock_pci_vfio(
    root: Path,
    devices: list[dict[str, str]],
    *,
    create_vfio_container: bool = True,
    create_vfio_nodes: list[str] | None = None,
    create_iommu: bool = False,
    create_iommufd_cdevs: list[str] | None = None,
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
        misc_dev = sysfs_root / "class" / "misc" / "vfio" / "dev"
        misc_dev.parent.mkdir(parents=True, exist_ok=True)
        misc_dev.write_text("10:196\n", encoding="utf-8")
    if create_iommu:
        (dev_root / "iommu").write_text("", encoding="utf-8")
        iommu_misc_dev = sysfs_root / "class" / "misc" / "iommu" / "dev"
        iommu_misc_dev.parent.mkdir(parents=True, exist_ok=True)
        iommu_misc_dev.write_text("10:197\n", encoding="utf-8")
    for cdev in create_iommufd_cdevs or []:
        cdev_path = vfio_dir / "devices" / cdev
        cdev_path.parent.mkdir(parents=True, exist_ok=True)
        cdev_path.write_text("", encoding="utf-8")
        suffix = cdev.removeprefix("vfio")
        minor = int(suffix) if suffix.isdigit() else 251
        cdev_sys = sysfs_root / "class" / "vfio-dev" / cdev / "dev"
        cdev_sys.parent.mkdir(parents=True, exist_ok=True)
        cdev_sys.write_text(f"243:{minor}\n", encoding="utf-8")
    for grp in create_vfio_nodes or []:
        (vfio_dir / grp).write_text("", encoding="utf-8")
        minor = int(grp) if grp.isdigit() else 250
        grp_dev = sysfs_root / "class" / "vfio" / grp / "dev"
        grp_dev.parent.mkdir(parents=True, exist_ok=True)
        grp_dev.write_text(f"242:{minor}\n", encoding="utf-8")

    for dev in devices:
        d = pci_devices / dev["bdf"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "vendor").write_text(dev["vendor"] + "\n", encoding="utf-8")
        (d / "device").write_text(dev["device"] + "\n", encoding="utf-8")
        if "numa_node" in dev:
            (d / "numa_node").write_text(dev["numa_node"] + "\n", encoding="utf-8")
        if "iommufd_cdev" in dev:
            cdev_name = dev["iommufd_cdev"]
            suffix = cdev_name.removeprefix("vfio")
            minor = int(suffix) if suffix.isdigit() else 251
            cdev_dir = d / "vfio-dev" / cdev_name
            cdev_dir.mkdir(parents=True, exist_ok=True)
            (cdev_dir / "dev").write_text(f"243:{minor}\n", encoding="utf-8")
        if dev.get("iommu_group"):
            grp = iommu_groups / dev["iommu_group"]
            grp_devs = grp / "devices"
            grp_devs.mkdir(parents=True, exist_ok=True)
            os.symlink(str(d), grp_devs / dev["bdf"])
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
                        "device": "0x0042",  # gVNIC NIC (must be skipped, no VFIO node)
                        "driver": "gve",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                ],
                create_vfio_nodes=["0"],
            )

            with _mock_vfio_char_devices():
                chips, vfio_container = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                )
                self.assertEqual(len(chips), 1)
                self.assertEqual(chips[0].pci_bdf, "0000:00:06.0")
                self.assertEqual(chips[0].generation, "v5e")

                spec = freetpu_cdi.build_cdi_spec(chips, vfio_container)
                errors = freetpu_cdi.validate_cdi_spec(spec)
                self.assertEqual(errors, [])

            self.assertEqual(spec["cdiVersion"], "0.6.0")
            self.assertEqual(spec["kind"], "freetpu.dev/tpu")
            names = [d["name"] for d in spec["devices"]]
            self.assertEqual(names, ["pci-0000-00-06-0", "all"])
            self.assertEqual(
                spec["containerEdits"]["deviceNodes"][0]["path"], "/dev/vfio/vfio"
            )
            self.assertEqual(
                spec["devices"][0]["containerEdits"]["deviceNodes"][0]["path"],
                "/dev/vfio/0",
            )
            # Default host-safe mode preserves workload-provided libtpu env vars
            # (no MEGASCALE/TPU_VISIBLE_DEVICES/TPU_CHIPS_PER_HOST_BOUNDS/TPU_SKIP_MDS_QUERY).
            self.assertNotIn("env", spec["devices"][0]["containerEdits"])
            self.assertNotIn("env", spec["devices"][-1]["containerEdits"])
            self.assertIn("FREETPU_NETWORK_MODE=host", spec["containerEdits"]["env"])

            # Opt-in single_host_env injects isolated overrides; composable single-chip
            # PCI selector still omits TPU_VISIBLE_DEVICES / TPU_CHIPS_PER_HOST_BOUNDS,
            # while `all` injects full visibility and bounds.
            with _mock_vfio_char_devices():
                iso_spec = freetpu_cdi.build_cdi_spec(
                    chips, vfio_container, single_host_env=True
                )
            self.assertEqual(
                iso_spec["devices"][0]["containerEdits"]["env"],
                [
                    "MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353",
                    "TPU_HOST_BOUNDS=1,1,1",
                    "TPU_WORKER_HOSTNAMES=localhost",
                    "TPU_WORKER_ID=0",
                    "TPU_SKIP_MDS_QUERY=true",
                ],
            )
            self.assertEqual(
                iso_spec["devices"][-1]["containerEdits"]["env"],
                [
                    "TPU_VISIBLE_CHIPS=0",
                    "TPU_VISIBLE_DEVICES=0",
                    "TPU_CHIPS_PER_HOST_BOUNDS=1,1,1",
                    "MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353",
                    "TPU_HOST_BOUNDS=1,1,1",
                    "TPU_WORKER_HOSTNAMES=localhost",
                    "TPU_WORKER_ID=0",
                    "TPU_SKIP_MDS_QUERY=true",
                ],
            )
            # Device numbers are omitted from deviceNodes (with live st_rdev tracked in annotations to invalidate CDI caches on rebind).
            for node in (
                spec["containerEdits"]["deviceNodes"]
                + spec["devices"][0]["containerEdits"]["deviceNodes"]
                + spec["devices"][-1]["containerEdits"]["deviceNodes"]
            ):
                self.assertNotIn("major", node)
                self.assertNotIn("minor", node)

    def test_generate_multi_chip_v6e_exposes_shared_group_as_single_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:04.0",
                        "vendor": "0x1ae0",
                        "device": "0x0042",  # gVNIC must be skipped
                        "driver": "gve",
                    },
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

            out_file = root / "var" / "run" / "cdi" / "freetpu.json"
            with _mock_vfio_char_devices():
                rc, _stdout, stderr = _run_cli_inproc(
                    [
                        "generate",
                        "--output",
                        str(out_file),
                        "--validate",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertEqual(rc, 0, stderr)

            spec = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(freetpu_cdi.validate_cdi_spec(spec), [])
            names = [d["name"] for d in spec["devices"]]
            # Shared group 0 becomes a single hardware-backed unit; isolated chip 0000:00:07.0 uses its stable PCI selector.
            self.assertEqual(
                names,
                ["group-pci-0000-00-05-0_0000-00-06-0", "pci-0000-00-07-0", "all"],
            )
            self.assertNotIn("pci-0000-00-05-0", names)
            self.assertNotIn("pci-0000-00-06-0", names)
            self.assertEqual(
                spec["devices"][0]["annotations"]["freetpu.dev/allocation-unit"],
                "shared-iommu-group",
            )
            self.assertEqual(
                [n["path"] for n in spec["devices"][0]["containerEdits"]["deviceNodes"]],
                ["/dev/vfio/0"],
            )
            all_nodes = [
                n["path"] for n in spec["devices"][-1]["containerEdits"]["deviceNodes"]
            ]
            self.assertEqual(all_nodes, ["/dev/vfio/0", "/dev/vfio/1"])
            self.assertNotIn("hooks", spec["containerEdits"])

            script = CDI_DIR / "freetpu_cdi.py"
            val_res = subprocess.run(
                [sys.executable, str(script), "validate", str(out_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("VALID CNCF CDI v0.6.0 spec", val_res.stdout)

    def test_validate_rejects_invalid_and_incomplete_per_device_specs(self) -> None:
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

        # Device "0" lacks any /dev/vfio/<group> node even though "all" provides one.
        incomplete_per_device = {
            "cdiVersion": "0.6.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/vfio"}]},
            "devices": [
                {"name": "0", "containerEdits": {"env": ["FOO=bar"]}},
                {"name": "all", "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]}},
            ],
        }
        errs3 = freetpu_cdi.validate_cdi_spec(incomplete_per_device)
        self.assertTrue(any("devices[0]" in e and "/dev/vfio/<iommu_group>" in e for e in errs3))

        # CDI v1.1.0 supports exact specs-go IntelRdt keys (`closID`, `l3CacheSchema`,
        # `memBwSchema`, `schemata`, `enableMonitoring`), `additionalGids`, and `netDevices`.
        v11_spec = {
            "cdiVersion": "1.1.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                "intelRdt": {
                    "closID": "tpu",
                    "l3CacheSchema": "L3:0=7f0",
                    "memBwSchema": "MB:0=50",
                    "schemata": ["L3:0=7f0"],
                    "enableMonitoring": True,
                },
                "additionalGids": [1000],
                "netDevices": [{"hostInterfaceName": "eth1", "name": "tpu0"}],
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                }
            ],
        }
        self.assertEqual(freetpu_cdi.validate_cdi_spec(v11_spec), [])

        # CDI v0.6.0 must reject v0.7+/v1.1+ fields (intelRdt, additionalGids, netDevices).
        v06_with_v11 = json.loads(json.dumps(v11_spec))
        v06_with_v11["cdiVersion"] = "0.6.0"
        errs_v06 = freetpu_cdi.validate_cdi_spec(v06_with_v11)
        self.assertTrue(any("intelRdt" in e and "0.6.0" in e for e in errs_v06), errs_v06)

        # CDI v0.7.0 - v1.0.0 accept boolean enableCMT and enableMBM on intelRdt,
        # reject non-boolean values, and reject v1.1.0-only schemata/enableMonitoring.
        for older_ver in ("0.7.0", "0.8.0", "1.0.0"):
            older_ok = {
                "cdiVersion": older_ver,
                "kind": "freetpu.dev/tpu",
                "containerEdits": {
                    "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                    "intelRdt": {
                        "closID": "tpu",
                        "l3CacheSchema": "L3:0=7f0",
                        "memBwSchema": "MB:0=50",
                        "enableCMT": True,
                        "enableMBM": False,
                    },
                },
                "devices": [
                    {
                        "name": "0",
                        "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                    }
                ],
            }
            self.assertEqual(freetpu_cdi.validate_cdi_spec(older_ok), [], older_ver)

        v08_bad_cmt = {
            "cdiVersion": "0.8.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                "intelRdt": {
                    "enableCMT": "true",
                    "enableMBM": 1,
                },
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                }
            ],
        }
        errs_v08_cmt = freetpu_cdi.validate_cdi_spec(v08_bad_cmt)
        self.assertTrue(any("intelRdt.enableCMT" in e for e in errs_v08_cmt), errs_v08_cmt)
        self.assertTrue(any("intelRdt.enableMBM" in e for e in errs_v08_cmt), errs_v08_cmt)

        v08_with_v11_irdt = {
            "cdiVersion": "0.8.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                "intelRdt": {
                    "closID": "tpu",
                    "l3CacheSchema": "L3:0=7f0",
                    "memBwSchema": "MB:0=50",
                    "schemata": ["L3:0=7f0"],
                    "enableMonitoring": True,
                },
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                }
            ],
        }
        errs_v08 = freetpu_cdi.validate_cdi_spec(v08_with_v11_irdt)
        self.assertTrue(any("schemata" in e and "enableMonitoring" in e for e in errs_v08), errs_v08)

        # CDI v1.1.0 removes enableCMT/enableMBM, rejects legacy/invalid IntelRdt keys
        # and wrong types, and requires non-empty unique netDevices[].name.
        bad_v11_spec = {
            "cdiVersion": "1.1.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                "intelRdt": {
                    "closID": 42,
                    "l3CacheSchema": ["bad"],
                    "memBwSchema": True,
                    "schemata": "L3:0=7f0",
                    "enableMonitoring": "yes",
                    "enableCMT": True,
                    "enableMBM": True,
                    "l3CacheAllocation": "L3:0=7f0",
                    "memBwAllocation": "MB:0=50",
                },
                "additionalGids": ["not-an-int"],
                "netDevices": [
                    {"hostInterfaceName": "eth1"},
                    {"hostInterfaceName": "eth2", "name": ""},
                    {"hostInterfaceName": "eth3", "name": "tpu0"},
                    {"hostInterfaceName": "eth4", "name": "tpu0"},
                ],
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                }
            ],
        }
        errs4 = freetpu_cdi.validate_cdi_spec(bad_v11_spec)
        self.assertTrue(any("l3CacheAllocation" in e and "memBwAllocation" in e for e in errs4), errs4)
        self.assertTrue(any("enableCMT" in e and "enableMBM" in e for e in errs4), errs4)
        self.assertTrue(any("intelRdt.closID" in e for e in errs4), errs4)
        self.assertTrue(any("intelRdt.l3CacheSchema" in e for e in errs4), errs4)
        self.assertTrue(any("intelRdt.memBwSchema" in e for e in errs4), errs4)
        self.assertTrue(any("intelRdt.schemata" in e for e in errs4), errs4)
        self.assertTrue(any("intelRdt.enableMonitoring" in e for e in errs4), errs4)
        self.assertTrue(any("additionalGids" in e for e in errs4), errs4)
        self.assertTrue(any("netDevices[0].name: must be a non-empty string" in e for e in errs4), errs4)
        self.assertTrue(any("netDevices[1].name: must be a non-empty string" in e for e in errs4), errs4)
        self.assertTrue(any("duplicate netDevice name" in e for e in errs4), errs4)

        # CNCF CDI v1.1.0 IntelRdt.closID validation: "/" (root CLOS), "" (unset), and
        # valid Linux filenames pass; paths containing "/", ".", "..", "\n", "\x00", or
        # >= 4096 bytes must fail validation.
        for valid_clos in ("", "/", "tpu-tenant-0"):
            ok_clos_spec = {
                "cdiVersion": "1.1.0",
                "kind": "freetpu.dev/tpu",
                "containerEdits": {
                    "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                    "intelRdt": {"closID": valid_clos},
                },
                "devices": [
                    {
                        "name": "0",
                        "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                    }
                ],
            }
            self.assertEqual(
                freetpu_cdi.validate_cdi_spec(ok_clos_spec),
                [],
                f"Expected closID {valid_clos!r} to be valid",
            )

        for bad_clos in ("tenant/tpu", "/tenant", ".", "..", "a\nb", "a\x00b", "a" * 4096):
            bad_clos_spec = {
                "cdiVersion": "1.1.0",
                "kind": "freetpu.dev/tpu",
                "containerEdits": {
                    "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                    "intelRdt": {"closID": bad_clos},
                },
                "devices": [
                    {
                        "name": "0",
                        "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                    }
                ],
            }
            clos_errs = freetpu_cdi.validate_cdi_spec(bad_clos_spec)
            self.assertTrue(
                any("intelRdt.closID: invalid closID" in e for e in clos_errs),
                f"Expected invalid closID error for {bad_clos!r}, got {clos_errs}",
            )

    def test_rejects_vfio_group_with_non_tpu_devices_and_non_vfio_driver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Group 0 contains both a recognized TPU and a non-TPU PCI function bound to vfio-pci.
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1af4",
                        "device": "0x1041",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                    {
                        "bdf": "0000:00:07.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                    },
                ],
                create_vfio_nodes=["0", "1"],
            )
            with _mock_vfio_char_devices():
                with self.assertRaisesRegex(RuntimeError, "non-TPU PCI device.*0000:00:05.0"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Non-TPU member listed only under /sys/kernel/iommu_groups/<group>/devices/.
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            extra_dev = sysfs_root / "kernel" / "iommu_groups" / "0" / "devices" / "0000:00:09.0"
            extra_dev.parent.mkdir(parents=True, exist_ok=True)
            extra_dev.write_text("", encoding="utf-8")
            with _mock_vfio_char_devices():
                with self.assertRaisesRegex(RuntimeError, "0000:00:09.0"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

        # Direct build_cdi_spec defense-in-depth check.
        bad_chip = freetpu_cdi.DiscoveredChip(
            index=0,
            pci_bdf="0000:00:06.0",
            vendor_id="0x1ae0",
            device_id="0x0063",
            generation="v5e",
            model_name="TPU v5e",
            driver="vfio-pci",
            iommu_group="0",
            vfio_host_path=Path("/dev/vfio/0"),
            vfio_container_path="/dev/vfio/0",
            non_tpu_group_members=("0000:00:05.0",),
        )
        with self.assertRaisesRegex(ValueError, "non-TPU"):
            freetpu_cdi.build_cdi_spec([bad_chip], Path("/dev/vfio/vfio"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Cloud TPU v4 bound to accel driver (non-vfio-pci) must fail closed.
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x005e",
                        "driver": "accel",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            with _mock_vfio_char_devices():
                with self.assertRaisesRegex(RuntimeError, "vfio-pci"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

    def test_device_nodes_omit_major_minor_for_char_devices(self) -> None:
        import stat as _stat
        from unittest import mock

        fake_stat = mock.MagicMock()
        fake_stat.st_mode = _stat.S_IFCHR | 0o600
        fake_stat.st_rdev = os.makedev(242, 7)

        with (
            mock.patch.object(Path, "stat", return_value=fake_stat),
            mock.patch.object(Path, "lstat", return_value=fake_stat),
        ):
            node = freetpu_cdi._make_device_node("/dev/vfio/0", Path("/dev/vfio/0"))
        self.assertEqual(node["path"], "/dev/vfio/0")
        self.assertEqual(node["hostPath"], "/dev/vfio/0")
        self.assertEqual(node["type"], "c")
        self.assertNotIn("major", node)
        self.assertNotIn("minor", node)

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

            # With an empty /dev/vfio directory (as created by DirectoryOrCreate when
            # vfio-pci failed to initialize), reconcile_cdi_spec must invalidate any stale spec.
            stale_spec = root / "var" / "run" / "cdi" / "freetpu.json"
            stale_spec.parent.mkdir(parents=True, exist_ok=True)
            stale_spec.write_text('{"stale": true}\n', encoding="utf-8")
            ok, msg = freetpu_cdi.reconcile_cdi_spec(
                stale_spec, sysfs_root=sysfs_root, dev_root=dev_root
            )
            self.assertFalse(ok)
            self.assertIn("removed stale", msg)
            self.assertFalse(stale_spec.exists())

    def test_docker_args_prints_required_vfio_flags(self) -> None:
        script = CDI_DIR / "freetpu_cdi.py"
        res = subprocess.run(
            [sys.executable, str(script), "docker-args", "--device", "group-0"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--device=freetpu.dev/tpu=group-0", res.stdout)
        self.assertIn("--ulimit memlock=-1:-1", res.stdout)
        self.assertNotIn("--cap-add=IPC_LOCK", res.stdout)

        res_cap = subprocess.run(
            [sys.executable, str(script), "docker-args", "--device", "group-0", "--cap-ipc-lock"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--cap-add=IPC_LOCK", res_cap.stdout)
        self.assertNotIn("--ulimit memlock=-1:-1", res_cap.stdout)

        res_kind = subprocess.run(
            [sys.executable, str(script), "docker-args", "--kind", "custom.io/tpu", "--device", "all"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--device=custom.io/tpu=all", res_kind.stdout)

    def test_default_output_is_boot_scoped_and_dry_run_requires_stdout(self) -> None:
        self.assertEqual(freetpu_cdi.DEFAULT_OUTPUT_PATH, Path("/var/run/cdi/freetpu.json"))
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
            live_spec = root / "var" / "run" / "cdi" / "freetpu.json"
            live_spec.parent.mkdir(parents=True, exist_ok=True)
            live_spec.write_text('{"sentinel": "keep"}\n', encoding="utf-8")

            script = CDI_DIR / "freetpu_cdi.py"
            res_fail = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "generate",
                    "--allow-missing-vfio",
                    "--output",
                    str(live_spec),
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(res_fail.returncode, 0)
            self.assertIn("--stdout", res_fail.stderr)
            # Must NEVER overwrite or remove the existing live spec on dry-run refusal.
            self.assertEqual(live_spec.read_text(encoding="utf-8"), '{"sentinel": "keep"}\n')

            res_ok = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "generate",
                    "--allow-missing-vfio",
                    "--stdout",
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parsed = json.loads(res_ok.stdout)
            self.assertEqual(parsed["cdiVersion"], "0.6.0")

            # Explicit --cdi-version 1.1.0 opt-in works on CLI.
            res_v11 = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "generate",
                    "--allow-missing-vfio",
                    "--stdout",
                    "--validate",
                    "--cdi-version",
                    "1.1.0",
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parsed_v11 = json.loads(res_v11.stdout)
            self.assertEqual(parsed_v11["cdiVersion"], "1.1.0")

    def test_host_revalidation_and_sync_invalidate_stale_vfio_group_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            spec_path = root / "var" / "run" / "cdi" / "freetpu.json"

            with _mock_vfio_char_devices():
                # Initial sync writes a valid spec.
                ok, _msg = freetpu_cdi.reconcile_cdi_spec(
                    spec_path,
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                )
                self.assertTrue(ok)
                self.assertTrue(spec_path.exists())

                # Live host validation succeeds while topology matches.
                rc_ok, out_ok, err_ok = _run_cli_inproc(
                    [
                        "validate",
                        str(spec_path),
                        "--check-host",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertEqual(rc_ok, 0, err_ok)
                self.assertIn("host VFIO identity verified", out_ok)

                # Simulate topology shift on the same host (same dev_root, shifted sysfs):
                # group 0 is reassigned to a different TPU BDF while spec still expects
                # 0000:00:06.0 in group 0.
                shifted_sysfs, _unused_dev = _make_mock_pci_vfio(
                    root / "shifted",
                    devices=[
                        {
                            "bdf": "0000:00:09.0",
                            "vendor": "0x1ae0",
                            "device": "0x0063",
                            "driver": "vfio-pci",
                            "iommu_group": "0",
                        }
                    ],
                    create_vfio_nodes=["0"],
                )
                rc_stale, _out_stale, err_stale = _run_cli_inproc(
                    [
                        "validate",
                        str(spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(shifted_sysfs),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertNotEqual(rc_stale, 0)
                self.assertIn("stale VFIO group identity", err_stale)
                self.assertFalse(spec_path.exists())

                # Re-create spec, then simulate non-TPU device entering group 0:
                # sync must invalidate/unlink the stale spec immediately.
                ok2, _msg2 = freetpu_cdi.reconcile_cdi_spec(
                    spec_path,
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                )
                self.assertTrue(ok2)
                self.assertTrue(spec_path.exists())

                extra_non_tpu = sysfs_root / "kernel" / "iommu_groups" / "0" / "devices" / "0000:00:05.0"
                extra_non_tpu.parent.mkdir(parents=True, exist_ok=True)
                extra_non_tpu.write_text("", encoding="utf-8")

                ok3, msg3 = freetpu_cdi.reconcile_cdi_spec(
                    spec_path,
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                )
                self.assertFalse(ok3)
                self.assertIn("removed stale", msg3)
                self.assertFalse(spec_path.exists())

    def test_selectors_stay_bound_to_physical_chip_when_earlier_chip_disappears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_2, dev_2 = _make_mock_pci_vfio(
                root / "two",
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                    },
                ],
                create_vfio_nodes=["0", "1"],
            )
            spec_path = root / "var" / "run" / "cdi" / "freetpu.json"
            with _mock_vfio_char_devices():
                ok1, _ = freetpu_cdi.reconcile_cdi_spec(spec_path, sysfs_root=sysfs_2, dev_root=dev_2)
                self.assertTrue(ok1)
                spec1 = json.loads(spec_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    [d["name"] for d in spec1["devices"]],
                    ["pci-0000-00-05-0", "pci-0000-00-06-0", "all"],
                )

                # Earlier chip 0000:00:05.0 disappears; 0000:00:06.0 remains.
                sysfs_1, dev_1 = _make_mock_pci_vfio(
                    root / "one",
                    devices=[
                        {
                            "bdf": "0000:00:06.0",
                            "vendor": "0x1ae0",
                            "device": "0x0063",
                            "driver": "vfio-pci",
                            "iommu_group": "1",
                        }
                    ],
                    create_vfio_nodes=["1"],
                )
                ok2, _ = freetpu_cdi.reconcile_cdi_spec(spec_path, sysfs_root=sysfs_1, dev_root=dev_1)
                self.assertTrue(ok2)
                spec2 = json.loads(spec_path.read_text(encoding="utf-8"))
                names2 = [d["name"] for d in spec2["devices"]]
                self.assertEqual(names2, ["pci-0000-00-06-0", "all"])
                self.assertNotIn("pci-0000-00-05-0", names2)
                self.assertEqual(
                    spec2["devices"][0]["annotations"]["freetpu.dev/pci-bdf"],
                    "0000:00:06.0",
                )

    def test_reconcile_refreshes_spec_when_vfio_char_devnums_change(self) -> None:
        import stat as _stat
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            spec_path = root / "var" / "run" / "cdi" / "freetpu.json"
            orig_stat = Path.stat
            orig_lstat = Path.lstat
            current_minor = {"group0": 0}

            def _maybe_mock(p: Path, st):
                if p.parent.name == "vfio" and p.name in ("vfio", "0") and not _stat.S_ISLNK(st.st_mode):
                    fake = mock.MagicMock()
                    fake.st_mode = _stat.S_IFCHR | 0o600
                    fake.st_rdev = (
                        os.makedev(10, 196)
                        if p.name == "vfio"
                        else os.makedev(242, current_minor["group0"])
                    )
                    return fake
                return st

            def _mock_stat(p: Path, *args, **kwargs):
                return _maybe_mock(p, orig_stat(p, *args, **kwargs))

            def _mock_lstat(p: Path, *args, **kwargs):
                return _maybe_mock(p, orig_lstat(p, *args, **kwargs))

            with (
                mock.patch.object(Path, "stat", autospec=True, side_effect=_mock_stat),
                mock.patch.object(Path, "lstat", autospec=True, side_effect=_mock_lstat),
            ):
                ok1, msg1 = freetpu_cdi.reconcile_cdi_spec(
                    spec_path, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(ok1)
                self.assertIn("updated", msg1)
                spec1 = json.loads(spec_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    spec1["annotations"]["freetpu.dev/vfio-devnums"],
                    "/dev/vfio/0=242:0,/dev/vfio/vfio=10:196",
                )
                # deviceNodes must still omit major/minor.
                self.assertNotIn("major", spec1["devices"][0]["containerEdits"]["deviceNodes"][0])

                # No change -> unchanged.
                ok_same, msg_same = freetpu_cdi.reconcile_cdi_spec(
                    spec_path, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(ok_same)
                self.assertIn("unchanged", msg_same)

                # Driver rebind changes /dev/vfio/0 minor from 0 -> 3 while BDF & group stay identical.
                current_minor["group0"] = 3
                (sysfs_root / "class" / "vfio" / "0" / "dev").write_text(
                    "242:3\n", encoding="utf-8"
                )
                errs_stale = freetpu_cdi.verify_spec_against_host(
                    spec1, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("242:3" in e for e in errs_stale))

                ok2, msg2 = freetpu_cdi.reconcile_cdi_spec(
                    spec_path, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(ok2)
                self.assertIn("updated", msg2)
                spec2 = json.loads(spec_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    spec2["annotations"]["freetpu.dev/vfio-devnums"],
                    "/dev/vfio/0=242:3,/dev/vfio/vfio=10:196",
                )
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        spec2, sysfs_root=sysfs_root, dev_root=dev_root
                    ),
                    [],
                )

    def test_check_host_validates_effective_host_path_global_nodes_and_pinned_devnums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0", "99"],
            )
            with _mock_vfio_char_devices():
                chips, vfio_container = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                base_spec = freetpu_cdi.build_cdi_spec(chips, vfio_container)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        base_spec, sysfs_root=sysfs_root, dev_root=dev_root
                    ),
                    [],
                )

                # 1. Tampered per-device hostPath pointing to /dev/vfio/99 while path says /dev/vfio/0.
                tampered_hp = json.loads(json.dumps(base_spec))
                tampered_hp["devices"][0]["containerEdits"]["deviceNodes"][0]["hostPath"] = str(
                    dev_root / "vfio" / "99"
                )
                errs_hp = freetpu_cdi.verify_spec_against_host(
                    tampered_hp, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("not a verified TPU-only VFIO group" in e for e in errs_hp))

                # 2. Rogue device added to global containerEdits.deviceNodes (affects every selector).
                rogue_global = json.loads(json.dumps(base_spec))
                rogue_global["containerEdits"]["deviceNodes"].append(
                    {
                        "path": "/dev/vfio/99",
                        "hostPath": str(dev_root / "vfio" / "99"),
                        "type": "c",
                        "permissions": "rw",
                    }
                )
                errs_global = freetpu_cdi.verify_spec_against_host(
                    rogue_global, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("unauthorized host device" in e for e in errs_global))

                # 3. Explicit pinned major/minor on a VFIO node must be rejected by --check-host.
                pinned_spec = json.loads(json.dumps(base_spec))
                pinned_spec["devices"][0]["containerEdits"]["deviceNodes"][0]["major"] = 242
                pinned_spec["devices"][0]["containerEdits"]["deviceNodes"][0]["minor"] = 99
                errs_pinned = freetpu_cdi.verify_spec_against_host(
                    pinned_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("explicit major/minor" in e for e in errs_pinned))

                # 4. Global and per-device mounts and OCI hooks must be rejected in --check-host mode.
                global_mount_spec = json.loads(json.dumps(base_spec))
                global_mount_spec["containerEdits"]["mounts"] = [
                    {"hostPath": "/tmp", "containerPath": "/dev/vfio"}
                ]
                errs_gm = freetpu_cdi.verify_spec_against_host(
                    global_mount_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("mounts" in e for e in errs_gm))

                global_hook_spec = json.loads(json.dumps(base_spec))
                global_hook_spec["containerEdits"]["hooks"] = [
                    {"hookName": "createRuntime", "path": "/bin/sh"}
                ]
                errs_gh = freetpu_cdi.verify_spec_against_host(
                    global_hook_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("hooks" in e for e in errs_gh))

                dev_mount_spec = json.loads(json.dumps(base_spec))
                dev_mount_spec["devices"][0]["containerEdits"]["mounts"] = [
                    {"hostPath": "/tmp", "containerPath": "/dev/vfio"}
                ]
                errs_dm = freetpu_cdi.verify_spec_against_host(
                    dev_mount_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("mounts" in e for e in errs_dm))

                dev_hook_spec = json.loads(json.dumps(base_spec))
                dev_hook_spec["devices"][0]["containerEdits"]["hooks"] = [
                    {"hookName": "createRuntime", "path": "/bin/sh"}
                ]
                errs_dh = freetpu_cdi.verify_spec_against_host(
                    dev_hook_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("hooks" in e for e in errs_dh))

                # 5. Case-variant / unknown keys (Go JSON decoder matches case-insensitively!)
                # must be rejected on deviceNodes, containerEdits, device entries, and root.
                case_major_spec = json.loads(json.dumps(base_spec))
                case_major_spec["devices"][0]["containerEdits"]["deviceNodes"][0]["Major"] = 242
                case_major_spec["devices"][0]["containerEdits"]["deviceNodes"][0]["Minor"] = 99
                errs_case_major = freetpu_cdi.verify_spec_against_host(
                    case_major_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("Major" in e and "Minor" in e for e in errs_case_major))
                self.assertTrue(
                    any("Major" in e for e in freetpu_cdi.validate_cdi_spec(case_major_spec))
                )

                case_hp_spec = json.loads(json.dumps(base_spec))
                case_hp_spec["devices"][0]["containerEdits"]["deviceNodes"][0]["HostPath"] = str(
                    dev_root / "vfio" / "99"
                )
                errs_case_hp = freetpu_cdi.verify_spec_against_host(
                    case_hp_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("HostPath" in e for e in errs_case_hp))

                case_hooks_spec = json.loads(json.dumps(base_spec))
                case_hooks_spec["containerEdits"]["Hooks"] = [
                    {"hookName": "createRuntime", "path": "/bin/sh"}
                ]
                errs_case_hooks = freetpu_cdi.verify_spec_against_host(
                    case_hooks_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("Hooks" in e for e in errs_case_hooks))

                case_mounts_spec = json.loads(json.dumps(base_spec))
                case_mounts_spec["devices"][0]["containerEdits"]["Mounts"] = [
                    {"hostPath": "/tmp", "containerPath": "/dev/vfio"}
                ]
                errs_case_mounts = freetpu_cdi.verify_spec_against_host(
                    case_mounts_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(any("Mounts" in e for e in errs_case_mounts))

                # Duplicate/case-colliding JSON keys in raw spec file must fail CLI validate --check-host.
                raw_spec_path = root / "dup-keys.json"
                raw_spec_path.write_text(
                    json.dumps(case_major_spec),
                    encoding="utf-8",
                )
                rc_dup, _out_dup, _err_dup = _run_cli_inproc(
                    [
                        "validate",
                        str(raw_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertNotEqual(rc_dup, 0)
                self.assertFalse(raw_spec_path.exists())

    def test_regular_file_at_vfio_path_rejected_in_live_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Plain regular files (NOT char devices) at /dev/vfio/vfio and /dev/vfio/12.
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "12",
                    }
                ],
                create_vfio_container=True,
                create_vfio_nodes=["12"],
            )
            with self.assertRaisesRegex(RuntimeError, "not a character device"):
                freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

            # Real subprocess CLI generate must also reject regular files cleanly.
            script = CDI_DIR / "freetpu_cdi.py"
            out_path = root / "var" / "run" / "cdi" / "freetpu.json"
            res = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "generate",
                    "--output",
                    str(out_path),
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("not a character device", res.stderr)
            self.assertFalse(out_path.exists())

            # Host validation against regular files must also fail.
            with _mock_vfio_char_devices():
                chips, vfio_container = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                spec = freetpu_cdi.build_cdi_spec(chips, vfio_container)
            errs = freetpu_cdi.verify_spec_against_host(
                spec, sysfs_root=sysfs_root, dev_root=dev_root
            )
            self.assertTrue(any("not a character device" in e for e in errs))

    def test_reconcile_invalidates_stale_spec_on_publication_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            spec_path = root / "var" / "run" / "cdi" / "freetpu.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text('{"stale": "old-group"}\n', encoding="utf-8")

            with _mock_vfio_char_devices():
                # 1. Publication OSError (e.g. ENOSPC on /run) must unlink stale spec without raising.
                with mock.patch.object(
                    freetpu_cdi,
                    "_write_spec_atomic",
                    side_effect=OSError("No space left on device"),
                ):
                    ok, msg = freetpu_cdi.reconcile_cdi_spec(
                        spec_path, sysfs_root=sysfs_root, dev_root=dev_root
                    )
                self.assertFalse(ok)
                self.assertIn("publication failed", msg)
                self.assertIn("removed stale", msg)
                self.assertFalse(spec_path.exists())

                # 2. If cleanup unlink also fails, error message explicitly reports cleanup failure.
                spec_path.write_text('{"stale": "stuck"}\n', encoding="utf-8")
                with (
                    mock.patch.object(
                        freetpu_cdi,
                        "_write_spec_atomic",
                        side_effect=OSError("No space left on device"),
                    ),
                    mock.patch.object(
                        freetpu_cdi,
                        "_invalidate_spec",
                        return_value=(False, "Permission denied"),
                    ),
                ):
                    ok2, msg2 = freetpu_cdi.reconcile_cdi_spec(
                        spec_path, sysfs_root=sysfs_root, dev_root=dev_root
                    )
                self.assertFalse(ok2)
                self.assertIn("FAILED to remove stale", msg2)
                self.assertIn("Permission denied", msg2)

                # 3. Once disk space recovers, next retry publishes cleanly.
                ok3, msg3 = freetpu_cdi.reconcile_cdi_spec(
                    spec_path, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(ok3, msg3)
                self.assertTrue(spec_path.exists())

    def test_custom_kind_grammar_matches_cdi_runtime_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            with _mock_vfio_char_devices():
                chips, vfio_container = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                base_spec = freetpu_cdi.build_cdi_spec(chips, vfio_container)

                # Invalid CDI kinds rejected by containerd's CDI parser (digit-leading vendor/class,
                # trailing punctuation, and 1-char components that panic CDI v0.8.0 / containerd 2.0.0).
                invalid_kinds = [
                    "1vendor/tpu",
                    "freetpu.dev/1tpu",
                    "0example.com/0tpu",
                    "freetpu.dev-/tpu",
                    "freetpu.dev/tpu_",
                    ".freetpu.dev/tpu",
                    "freetpu.dev/.tpu",
                    "a/b",
                    "a/tpu",
                    "freetpu.dev/t",
                ]
                for bad_kind in invalid_kinds:
                    with self.assertRaisesRegex(ValueError, "Invalid CDI kind"):
                        freetpu_cdi.build_cdi_spec(chips, vfio_container, kind=bad_kind)

                    bad_spec = json.loads(json.dumps(base_spec))
                    bad_spec["kind"] = bad_kind
                    errs = freetpu_cdi.validate_cdi_spec(bad_spec)
                    self.assertTrue(
                        any("kind" in e for e in errs),
                        f"expected validate_cdi_spec to reject kind={bad_kind!r}, got {errs}",
                    )

                    out_path = root / "var" / "run" / "cdi" / "bad.json"
                    rc, _out, err = _run_cli_inproc(
                        [
                            "generate",
                            "--kind",
                            bad_kind,
                            "--validate",
                            "--output",
                            str(out_path),
                            "--sysfs-root",
                            str(sysfs_root),
                            "--dev-root",
                            str(dev_root),
                        ]
                    )
                    self.assertNotEqual(rc, 0, f"CLI generate should reject kind={bad_kind!r}")
                    self.assertIn("Invalid CDI kind", err)
                    self.assertFalse(out_path.exists())

                # Valid CDI kinds accepted by containerd's CDI parser.
                for good_kind in ["freetpu.dev/tpu", "Example_Org.io/TPU-v6e", "ab/cd", "v1.org/tpu0"]:
                    spec = freetpu_cdi.build_cdi_spec(chips, vfio_container, kind=good_kind)
                    self.assertEqual(freetpu_cdi.validate_cdi_spec(spec), [])

                # Device names ending in punctuation must be rejected per CDI v0.6.0.
                bad_dev_spec = json.loads(json.dumps(base_spec))
                bad_dev_spec["devices"][0]["name"] = "all-"
                bad_dev_errs = freetpu_cdi.validate_cdi_spec(bad_dev_spec)
                self.assertTrue(
                    any("invalid CDI device name" in e for e in bad_dev_errs), bad_dev_errs
                )

    def test_rejects_vfio_devnum_mismatch_against_kernel_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )

            with _mock_vfio_char_devices():
                chips, vfio_container = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                valid_spec = freetpu_cdi.build_cdi_spec(chips, vfio_container)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        valid_spec, sysfs_root=sysfs_root, dev_root=dev_root
                    ),
                    [],
                )

                # 1. Stale/mismatched /dev/vfio/0 (242:0) vs kernel /sys/class/vfio/0/dev (242:99):
                #    BOTH discovery AND host validation (even with matching annotation) MUST reject!
                (sysfs_root / "class" / "vfio" / "0" / "dev").write_text(
                    "242:99\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "does not match kernel sysfs group 0"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

                errs_mismatch = freetpu_cdi.verify_spec_against_host(
                    valid_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(
                    any("does not match kernel sysfs group 0" in e for e in errs_mismatch),
                    errs_mismatch,
                )

                # 2. Missing kernel /sys/class/vfio/0/dev (and virtual fallback) fails closed.
                (sysfs_root / "class" / "vfio" / "0" / "dev").unlink()
                with self.assertRaisesRegex(RuntimeError, "Unable to verify kernel device number"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

                # 3. Canonical /sys/devices/virtual/vfio/<group>/dev fallback works when
                #    /sys/class/vfio is not mounted (narrow DaemonSet mounts).
                virt_group_dev = sysfs_root / "devices" / "virtual" / "vfio" / "0" / "dev"
                virt_group_dev.parent.mkdir(parents=True, exist_ok=True)
                virt_group_dev.write_text("242:0\n", encoding="utf-8")
                (sysfs_root / "class" / "misc" / "vfio" / "dev").unlink()
                virt_misc_dev = sysfs_root / "devices" / "virtual" / "misc" / "vfio" / "dev"
                virt_misc_dev.parent.mkdir(parents=True, exist_ok=True)
                virt_misc_dev.write_text("10:196\n", encoding="utf-8")

                chips_virt, _ = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertEqual(len(chips_virt), 1)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        valid_spec, sysfs_root=sysfs_root, dev_root=dev_root
                    ),
                    [],
                )

                # 4. Mismatched /dev/vfio/vfio control device number fails closed.
                virt_misc_dev.write_text("10:250\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "misc/vfio/dev"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

    def test_rejects_missing_or_incomplete_kernel_iommu_group_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                ],
                create_vfio_nodes=["0"],
            )
            with _mock_vfio_char_devices():
                # Baseline passes when kernel iommu_groups/0/devices contains the TPU.
                chips, _ = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertEqual(len(chips), 1)

                # Removing candidate TPU entry from kernel membership list fails closed.
                member_link = sysfs_root / "kernel" / "iommu_groups" / "0" / "devices" / "0000:00:06.0"
                member_link.unlink()
                with self.assertRaisesRegex(RuntimeError, "does not contain 0000:00:06.0"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

                # Removing the kernel iommu_groups/0/devices directory fails closed.
                (sysfs_root / "kernel" / "iommu_groups" / "0" / "devices").rmdir()
                with self.assertRaisesRegex(RuntimeError, "missing or unreadable"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

    def test_rejects_symlinked_vfio_nodes_and_withdraws_spec_on_graceful_exit(self) -> None:
        import signal

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                ],
                create_vfio_nodes=["0"],
            )
            spec_out = root / "cdi" / "freetpu.json"
            with _mock_vfio_char_devices():
                # 1. Symlinked /dev/vfio/0 MUST be rejected (matching CDI v0.8.0 lstat check).
                target_char = dev_root / "vfio" / "real0"
                (dev_root / "vfio" / "0").rename(target_char)
                os.symlink(str(target_char), dev_root / "vfio" / "0")
                with self.assertRaisesRegex(RuntimeError, "not a character device"):
                    freetpu_cdi.discover_tpu_chips(sysfs_root=sysfs_root, dev_root=dev_root)

                # Restore real node for sync --unlink-on-exit test.
                (dev_root / "vfio" / "0").unlink()
                target_char.rename(dev_root / "vfio" / "0")

                # 2. `sync --unlink-on-exit` publishes spec then removes it on graceful SIGTERM.
                def _stop_after_first_sleep(_secs: float) -> None:
                    self.assertTrue(spec_out.exists())
                    os.kill(os.getpid(), signal.SIGTERM)

                with mock.patch.object(freetpu_cdi.time, "sleep", side_effect=_stop_after_first_sleep):
                    rc, _, err = _run_cli_inproc(
                        [
                            "sync",
                            "--output",
                            str(spec_out),
                            "--interval",
                            "1",
                            "--unlink-on-exit",
                            "--sysfs-root",
                            str(sysfs_root),
                            "--dev-root",
                            str(dev_root),
                        ]
                    )
                self.assertEqual(rc, 0, err)
                self.assertFalse(spec_out.exists())
                self.assertIn("withdrew CDI spec", err)

                # 3. Tampered PCI selector name pointing to live VFIO group fails --check-host.
                chips_live, vfio_c = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                tampered = freetpu_cdi.build_cdi_spec(chips_live, vfio_c)
                for dev in tampered["devices"]:
                    if dev["name"].startswith("pci-"):
                        dev["name"] = "pci-0000-00-99-0"
                errs_sel = freetpu_cdi.verify_spec_against_host(
                    tampered, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(
                    any("PCI selector name does not match" in e for e in errs_sel),
                    errs_sel,
                )

                # 4. Non-character deviceNode type (e.g. 'b') fails --check-host.
                bad_type = freetpu_cdi.build_cdi_spec(chips_live, vfio_c)
                bad_type["devices"][0]["containerEdits"]["deviceNodes"][0]["type"] = "b"
                errs_type = freetpu_cdi.verify_spec_against_host(
                    bad_type, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(
                    any("must be omitted or 'c'" in e for e in errs_type),
                    errs_type,
                )

                # 5. Symlinked hostPath in a spec pointing to a live char node fails --check-host.
                sym_node = dev_root / "vfio" / "sym0"
                os.symlink(str(dev_root / "vfio" / "0"), sym_node)
                sym_spec = freetpu_cdi.build_cdi_spec(chips_live, vfio_c)
                sym_spec["devices"][0]["containerEdits"]["deviceNodes"][0]["hostPath"] = str(sym_node)
                errs_sym = freetpu_cdi.verify_spec_against_host(
                    sym_spec, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(
                    any("not a non-symlink character device" in e for e in errs_sym),
                    errs_sym,
                )

                # 6. Non-string/unhashable JSON fields in type/hookName/mounts/hooks produce validation errors (no TypeError).
                malformed_spec = {
                    "cdiVersion": "0.6.0",
                    "kind": "freetpu.dev/tpu",
                    "devices": [
                        {
                            "name": "all",
                            "containerEdits": {
                                "env": ["=empty-key"],
                                "deviceNodes": [
                                    {"path": "/dev/vfio/0", "type": ["c"], "uid": -1, "gid": 2**32, "fileMode": -1}
                                ],
                                "mounts": [
                                    {
                                        "hostPath": "/tmp/a",
                                        "containerPath": "/tmp/b",
                                        "options": "ro",
                                        "type": 123,
                                        "extraField": True,
                                    }
                                ],
                                "hooks": [
                                    {
                                        "hookName": {"bad": 1},
                                        "path": "/bin/true",
                                        "args": "not-a-list",
                                        "timeout": 0,
                                        "unknownKey": 1,
                                    }
                                ],
                            },
                        }
                    ],
                }
                mal_errs = freetpu_cdi.validate_cdi_spec(malformed_spec)
                self.assertTrue(any("non-empty KEY=VALUE" in e for e in mal_errs), mal_errs)
                self.assertTrue(any("invalid device type" in e for e in mal_errs), mal_errs)
                self.assertTrue(any(".uid:" in e for e in mal_errs), mal_errs)
                self.assertTrue(any(".gid:" in e for e in mal_errs), mal_errs)
                self.assertTrue(any(".fileMode:" in e for e in mal_errs), mal_errs)
                self.assertTrue(any(".options: must be a list of strings" in e for e in mal_errs), mal_errs)
                self.assertTrue(any(".type: must be a string" in e for e in mal_errs), mal_errs)
                self.assertTrue(any("unsupported CDI v0.6.0 mount keys" in e for e in mal_errs), mal_errs)
                self.assertTrue(any("invalid OCI hookName" in e for e in mal_errs), mal_errs)
                self.assertTrue(any(".args: must be a list of strings" in e for e in mal_errs), mal_errs)
                self.assertTrue(any(".timeout: must be a positive integer" in e for e in mal_errs), mal_errs)
                self.assertTrue(any("unsupported CDI v0.6.0 hook keys" in e for e in mal_errs), mal_errs)

                # 7. Default per-device containerEdits preserves workload env (no libtpu overrides);
                #    opt-in `single_host_env=True` injects isolated libtpu env vars.
                clean_spec = freetpu_cdi.build_cdi_spec(chips_live, vfio_c)
                for dev_entry in clean_spec["devices"]:
                    self.assertNotIn("env", dev_entry["containerEdits"])
                    self.assertIn("annotations", dev_entry)

                isolated_spec = freetpu_cdi.build_cdi_spec(
                    chips_live, vfio_c, single_host_env=True
                )
                for dev_entry in isolated_spec["devices"]:
                    self.assertIn("env", dev_entry["containerEdits"])
                    self.assertIn(
                        "MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353",
                        dev_entry["containerEdits"]["env"],
                    )
                    self.assertIn("TPU_HOST_BOUNDS=1,1,1", dev_entry["containerEdits"]["env"])
                    self.assertIn("TPU_WORKER_HOSTNAMES=localhost", dev_entry["containerEdits"]["env"])
                    self.assertIn("TPU_WORKER_ID=0", dev_entry["containerEdits"]["env"])
                    self.assertIn("TPU_SKIP_MDS_QUERY=true", dev_entry["containerEdits"]["env"])

                # 8. Empty per-device containerEdits is rejected (CDI requires >= 1 edit per device).
                empty_dev_spec = {
                    "cdiVersion": "0.6.0",
                    "kind": "freetpu.dev/tpu",
                    "containerEdits": {
                        "deviceNodes": [
                            {"path": "/dev/vfio/vfio", "type": "c", "permissions": "rw"},
                            {"path": "/dev/vfio/0", "type": "c", "permissions": "rw"},
                        ]
                    },
                    "devices": [{"name": "all", "containerEdits": {}}],
                }
                empty_errs = freetpu_cdi.validate_cdi_spec(empty_dev_spec)
                self.assertTrue(
                    any("must contain at least one edit" in e for e in empty_errs),
                    empty_errs,
                )

                # 9. Relative `--dev-root` emits absolute `hostPath` entries that pass `validate_cdi_spec`.
                rel_dev = Path(os.path.relpath(dev_root, Path.cwd()))
                rel_sys = Path(os.path.relpath(sysfs_root, Path.cwd()))
                rel_chips, rel_vfio_c = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=rel_sys, dev_root=rel_dev
                )
                rel_spec = freetpu_cdi.build_cdi_spec(rel_chips, rel_vfio_c)
                self.assertEqual(freetpu_cdi.validate_cdi_spec(rel_spec), [])
                self.assertTrue(
                    rel_spec["containerEdits"]["deviceNodes"][0]["hostPath"].startswith("/")
                )
                for dev_entry in rel_spec["devices"]:
                    for dn in dev_entry["containerEdits"]["deviceNodes"]:
                        self.assertTrue(dn["hostPath"].startswith("/"))

                # 10. Invalid `--kind` in `generate` or `sync` fails without unlinking an existing spec.
                existing_out = root / "published_freetpu.json"
                existing_out.write_text('{"keep": true}\n', encoding="utf-8")
                ok_bad_k, msg_bad_k = freetpu_cdi.reconcile_cdi_spec(
                    existing_out, sysfs_root=sysfs_root, dev_root=dev_root, kind="badkind"
                )
                self.assertFalse(ok_bad_k)
                self.assertIn("Invalid CDI kind", msg_bad_k)
                self.assertTrue(existing_out.exists())
                rc_gen_bad = freetpu_cdi.main(
                    [
                        "generate",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                        "--output",
                        str(existing_out),
                        "--kind",
                        "badkind",
                    ]
                )
                self.assertEqual(rc_gen_bad, 1)
                self.assertTrue(existing_out.exists())
                rc_sync_bad = freetpu_cdi.main(
                    [
                        "sync",
                        "--once",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                        "--output",
                        str(existing_out),
                        "--kind",
                        "badkind",
                    ]
                )
                self.assertEqual(rc_sync_bad, 1)
                self.assertTrue(existing_out.exists())

    def test_cdi_v11_network_mode_isolated_vs_host_and_docker_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            devices = [
                {
                    "bdf": f"0000:00:{idx + 1:02x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x006f",
                    "driver": "vfio-pci",
                    "iommu_group": str(idx),
                }
                for idx in range(4)
            ]
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root, devices=devices, create_vfio_nodes=[str(i) for i in range(4)]
            )
            with _mock_vfio_char_devices():
                chips, control_node = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                iso_spec = freetpu_cdi.build_cdi_spec(
                    chips, control_node, network_mode="isolated"
                )
                self.assertEqual(freetpu_cdi.validate_cdi_spec(iso_spec), [])
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        iso_spec, sysfs_root=sysfs_root, dev_root=dev_root
                    ),
                    [],
                )
                all_env = {
                    kv.split("=", 1)[0]: kv.split("=", 1)[1]
                    for kv in iso_spec["devices"][-1]["containerEdits"]["env"]
                }
                self.assertEqual(all_env["TPU_VISIBLE_DEVICES"], "0,1,2,3")
                self.assertEqual(all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(all_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")
                self.assertEqual(all_env["TPU_HOST_BOUNDS"], "1,1,1")
                self.assertEqual(all_env["TPU_WORKER_HOSTNAMES"], "localhost")
                self.assertEqual(all_env["TPU_WORKER_ID"], "0")
                self.assertEqual(all_env["TPU_SKIP_MDS_QUERY"], "true")

                # Composable PCI selectors omit TPU_VISIBLE_CHIPS / TPU_VISIBLE_DEVICES /
                # TPU_CHIPS_PER_HOST_BOUNDS so combining multiple PCI selectors never
                # overwrites visibility in OCI env.
                pci0_iso_env = {
                    kv.split("=", 1)[0]: kv.split("=", 1)[1]
                    for kv in iso_spec["devices"][0]["containerEdits"]["env"]
                }
                self.assertNotIn("TPU_VISIBLE_CHIPS", pci0_iso_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", pci0_iso_env)
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", pci0_iso_env)
                self.assertEqual(pci0_iso_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")
                self.assertEqual(pci0_iso_env["TPU_WORKER_HOSTNAMES"], "localhost")
                self.assertEqual(pci0_iso_env["TPU_WORKER_ID"], "0")
                self.assertEqual(pci0_iso_env["TPU_SKIP_MDS_QUERY"], "true")

                host_spec = freetpu_cdi.build_cdi_spec(
                    chips, control_node, network_mode="host"
                )
                self.assertEqual(freetpu_cdi.validate_cdi_spec(host_spec), [])
                # In default/host mode, NO selector injects libtpu env overrides so
                # multi-host jobs preserve workload-configured visibility, bounds, and coordinator.
                for dev_entry in host_spec["devices"]:
                    self.assertNotIn("env", dev_entry["containerEdits"])

                # CLI `generate` defaults to host-safe mode; `--single-host-env` opts in.
                rc_def, out_def, err_def = _run_cli_inproc(
                    [
                        "generate",
                        "--stdout",
                        "--validate",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertEqual(rc_def, 0, err_def)
                parsed_def = json.loads(out_def)
                for dev_entry in parsed_def["devices"]:
                    self.assertNotIn("env", dev_entry["containerEdits"])

                rc_sh, out_sh, err_sh = _run_cli_inproc(
                    [
                        "generate",
                        "--stdout",
                        "--validate",
                        "--single-host-env",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertEqual(rc_sh, 0, err_sh)
                parsed_sh = json.loads(out_sh)
                sh_all_env = dict(
                    kv.split("=", 1) for kv in parsed_sh["devices"][-1]["containerEdits"]["env"]
                )
                self.assertEqual(sh_all_env["TPU_VISIBLE_DEVICES"], "0,1,2,3")
                self.assertEqual(sh_all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(sh_all_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")

        rc_iso, out_iso, _ = _run_cli_inproc(["docker-args", "--device", "numa0"])
        self.assertEqual(rc_iso, 0)
        self.assertNotIn("--net=host", out_iso)
        self.assertIn("--device=freetpu.dev/tpu=numa0", out_iso)

        rc_host, out_host, _ = _run_cli_inproc(
            ["docker-args", "--device", "all", "--network-mode", "host"]
        )
        self.assertEqual(rc_host, 0)
        self.assertIn("--net=host", out_host)

    def test_iommufd_preserves_legacy_default_requires_sysfs_cdev_and_groups_iommu_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Pure iommufd host (no /dev/vfio/vfio, /dev/iommu present, verified vfio-dev/vfioN).
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root / "pure_iommufd",
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                        "iommufd_cdev": "vfio0",
                    },
                    {
                        "bdf": "0000:00:0c.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "2",
                        "iommufd_cdev": "vfio1",
                    },
                ],
                create_vfio_container=False,
                create_vfio_nodes=[],
                create_iommu=True,
                create_iommufd_cdevs=["vfio0", "vfio1"],
            )

            with _mock_vfio_char_devices():
                chips, control_node = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    iommu_dev_path=dev_root / "iommu",
                )
                self.assertEqual(control_node, dev_root / "iommu")
                self.assertEqual([c.vfio_mode for c in chips], ["iommufd", "iommufd"])
                self.assertEqual(
                    [c.vfio_container_path for c in chips],
                    ["/dev/vfio/devices/vfio0", "/dev/vfio/devices/vfio1"],
                )
                spec = freetpu_cdi.build_cdi_spec(chips, control_node)
                self.assertEqual(freetpu_cdi.validate_cdi_spec(spec), [])
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        spec,
                        sysfs_root=sysfs_root,
                        dev_root=dev_root,
                        iommu_dev_path=dev_root / "iommu",
                    ),
                    [],
                )
                self.assertEqual(
                    spec["containerEdits"]["deviceNodes"][0]["path"], "/dev/iommu"
                )
                self.assertEqual(
                    spec["devices"][0]["containerEdits"]["deviceNodes"][0]["path"],
                    "/dev/vfio/devices/vfio0",
                )
                self.assertEqual(
                    spec["devices"][1]["containerEdits"]["deviceNodes"][0]["path"],
                    "/dev/vfio/devices/vfio1",
                )
                self.assertIn("FREETPU_IOMMU_MODE=iommufd", spec["containerEdits"]["env"])

                # Incomplete iommufd spec lacking /dev/vfio/devices/vfioN must fail validation.
                bad_iommufd = {
                    "cdiVersion": "0.6.0",
                    "kind": "freetpu.dev/tpu",
                    "containerEdits": {"deviceNodes": [{"path": "/dev/iommu"}]},
                    "devices": [{"name": "0", "containerEdits": {"env": ["FOO=bar"]}}],
                }
                bad_errs = freetpu_cdi.validate_cdi_spec(bad_iommufd)
                self.assertTrue(
                    any("/dev/vfio/devices/vfioN" in e for e in bad_errs), bad_errs
                )

            # 2. Dual-interface host: BOTH legacy VFIO and /dev/iommu exist.
            #    Default `vfio_mode="auto"` MUST preserve legacy VFIO (even when TPU
            #    cdevs are absent or present); explicit `--vfio-mode iommufd`/`cdev` opts in.
            dual_sysfs, dual_dev = _make_mock_pci_vfio(
                root / "dual_interface",
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                        "iommufd_cdev": "vfio4",
                    }
                ],
                create_vfio_container=True,
                create_vfio_nodes=["3"],
                create_iommu=True,
                create_iommufd_cdevs=["vfio4"],
            )
            with _mock_vfio_char_devices():
                chips_auto, ctrl_auto = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=dual_sysfs, dev_root=dual_dev
                )
                self.assertEqual(ctrl_auto, dual_dev / "vfio" / "vfio")
                self.assertEqual(chips_auto[0].vfio_mode, "vfio-group")
                self.assertEqual(chips_auto[0].vfio_container_path, "/dev/vfio/3")
                legacy_spec = freetpu_cdi.build_cdi_spec(chips_auto, ctrl_auto)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        legacy_spec, sysfs_root=dual_sysfs, dev_root=dual_dev
                    ),
                    [],
                )

                # Explicit opt-in `--vfio-mode cdev` / `iommufd` selects cdev on the same host
                # and validates cleanly with `verify_spec_against_host`!
                chips_cdev, ctrl_cdev = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=dual_sysfs, dev_root=dual_dev, vfio_mode="cdev"
                )
                self.assertEqual(ctrl_cdev, dual_dev / "iommu")
                self.assertEqual(chips_cdev[0].vfio_mode, "iommufd")
                self.assertEqual(chips_cdev[0].vfio_container_path, "/dev/vfio/devices/vfio4")
                cdev_spec = freetpu_cdi.build_cdi_spec(chips_cdev, ctrl_cdev)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        cdev_spec, sysfs_root=dual_sysfs, dev_root=dual_dev
                    ),
                    [],
                )

            # 3. Missing `<pci>/vfio-dev/vfioN` association MUST fail closed even when
            #    `/dev/vfio/devices/vfio0` and `/sys/class/vfio-dev/vfio0/dev` exist
            #    (never guess `vfio0` from TPU enumeration index!).
            no_assoc_sysfs, no_assoc_dev = _make_mock_pci_vfio(
                root / "no_cdev_assoc",
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                        # intentionally omit iommufd_cdev!
                    }
                ],
                create_vfio_container=True,
                create_vfio_nodes=["3"],
                create_iommu=True,
                create_iommufd_cdevs=["vfio0"],  # belongs to some other device!
            )
            with _mock_vfio_char_devices():
                # Default auto still succeeds via legacy VFIO without failing or invalidating!
                chips_safe, ctrl_safe = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=no_assoc_sysfs, dev_root=no_assoc_dev
                )
                self.assertEqual(ctrl_safe, no_assoc_dev / "vfio" / "vfio")
                self.assertEqual(chips_safe[0].vfio_mode, "vfio-group")

                # Explicit iommufd mode MUST fail closed instead of guessing vfio0.
                with self.assertRaisesRegex(RuntimeError, "missing sysfs association directory"):
                    freetpu_cdi.discover_tpu_chips(
                        sysfs_root=no_assoc_sysfs,
                        dev_root=no_assoc_dev,
                        vfio_mode="iommufd",
                    )

            # 4. Shared IOMMU group in iommufd mode MUST group sibling cdevs into ONE
            #    allocation unit and suppress cross-domain sub-host slices.
            shared_sysfs, shared_dev = _make_mock_pci_vfio(
                root / "shared_iommufd",
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "7",
                        "numa_node": "0",
                        "iommufd_cdev": "vfio0",
                    },
                    {
                        "bdf": "0000:00:0c.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "7",  # shared IOMMU group with 0000:00:0b.0
                        "numa_node": "0",
                        "iommufd_cdev": "vfio1",
                    },
                    {
                        "bdf": "0000:80:00.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "7",  # crosses NUMA 0 / NUMA 1 boundary!
                        "numa_node": "1",
                        "iommufd_cdev": "vfio2",
                    },
                    {
                        "bdf": "0000:80:01.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "8",
                        "numa_node": "1",
                        "iommufd_cdev": "vfio3",
                    },
                ],
                create_vfio_container=False,
                create_vfio_nodes=[],
                create_iommu=True,
                create_iommufd_cdevs=["vfio0", "vfio1", "vfio2", "vfio3"],
            )
            with _mock_vfio_char_devices():
                chips_sh, ctrl_sh = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=shared_sysfs, dev_root=shared_dev, vfio_mode="iommufd"
                )
                spec_sh = freetpu_cdi.build_cdi_spec(chips_sh, ctrl_sh)
                self.assertEqual(freetpu_cdi.validate_cdi_spec(spec_sh), [])
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        spec_sh, sysfs_root=shared_sysfs, dev_root=shared_dev
                    ),
                    [],
                )
                sh_names = [d["name"] for d in spec_sh["devices"]]
                # Group 7 siblings become ONE shared-group unit; cross-domain numa0/numa1 suppressed!
                self.assertEqual(
                    sh_names,
                    [
                        "group-pci-0000-00-0b-0_0000-00-0c-0_0000-80-00-0",
                        "pci-0000-80-01-0",
                        "all",
                    ],
                )
                self.assertNotIn("numa0", sh_names)
                self.assertNotIn("numa1", sh_names)
                self.assertEqual(
                    [n["path"] for n in spec_sh["devices"][0]["containerEdits"]["deviceNodes"]],
                    [
                        "/dev/vfio/devices/vfio0",
                        "/dev/vfio/devices/vfio1",
                        "/dev/vfio/devices/vfio2",
                    ],
                )
                # Tampering spec to expose only ONE chip of shared group 7 must fail --check-host!
                partial_spec = json.loads(json.dumps(spec_sh))
                partial_spec["devices"][0]["name"] = "pci-0000-00-0b-0"
                partial_spec["devices"][0]["annotations"]["freetpu.dev/pci-bdfs"] = "0000:00:0b.0"
                partial_spec["devices"][0]["containerEdits"]["deviceNodes"] = [
                    partial_spec["devices"][0]["containerEdits"]["deviceNodes"][0]
                ]
                errs_partial = freetpu_cdi.verify_spec_against_host(
                    partial_spec, sysfs_root=shared_sysfs, dev_root=shared_dev
                )
                self.assertTrue(
                    any("partial IOMMU group allocation" in e for e in errs_partial),
                    errs_partial,
                )

    def test_v6e8_numa_pcie_root_complex_subhost_slices_and_tpu7x(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v6e8_devices = [
                {
                    "bdf": f"0000:00:0{0x0b + i:x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x006f",
                    "driver": "vfio-pci",
                    "iommu_group": str(i),
                    "numa_node": "0",
                }
                for i in range(4)
            ] + [
                {
                    "bdf": f"0000:80:0{i:x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x006f",
                    "driver": "vfio-pci",
                    "iommu_group": str(4 + i),
                    "numa_node": "1",
                }
                for i in range(4)
            ]
            sysfs_root, dev_root = _make_mock_pci_vfio(
                root,
                devices=v6e8_devices,
                create_vfio_nodes=[str(i) for i in range(8)],
            )
            with _mock_vfio_char_devices():
                chips, control_node = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_root, dev_root=dev_root
                )
                spec = freetpu_cdi.build_cdi_spec(
                    chips, control_node, single_host_env=True
                )
                self.assertEqual(freetpu_cdi.validate_cdi_spec(spec), [])
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        spec, sysfs_root=sysfs_root, dev_root=dev_root
                    ),
                    [],
                )

                by_name = {d["name"]: d for d in spec["devices"]}
                for slice_name in ("numa0", "0-3", "numa1", "4-7", "all"):
                    self.assertIn(slice_name, by_name)

                # NUMA 0 slice strictly binds 0000:00:* chips 0..3 with 2,2,1 bounds.
                numa0 = by_name["numa0"]
                self.assertEqual(numa0["annotations"]["freetpu.dev/pcie-root"], "0000:00")
                self.assertEqual(numa0["annotations"]["freetpu.dev/numa-node"], "0")
                self.assertTrue(
                    all(
                        b.startswith("0000:00:")
                        for b in numa0["annotations"]["freetpu.dev/pci-bdfs"].split(",")
                    )
                )
                n0_env = dict(kv.split("=", 1) for kv in numa0["containerEdits"]["env"])
                self.assertEqual(n0_env["TPU_VISIBLE_DEVICES"], "0,1,2,3")
                self.assertEqual(n0_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(n0_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")

                # NUMA 1 slice strictly binds 0000:80:* chips 4..7 with 2,2,1 bounds.
                numa1 = by_name["numa1"]
                self.assertEqual(numa1["annotations"]["freetpu.dev/pcie-root"], "0000:80")
                self.assertEqual(numa1["annotations"]["freetpu.dev/numa-node"], "1")
                self.assertTrue(
                    all(
                        b.startswith("0000:80:")
                        for b in numa1["annotations"]["freetpu.dev/pci-bdfs"].split(",")
                    )
                )
                n1_env = dict(kv.split("=", 1) for kv in numa1["containerEdits"]["env"])
                self.assertEqual(n1_env["TPU_VISIBLE_DEVICES"], "4,5,6,7")
                self.assertEqual(n1_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")

                # Full 8-chip v6e-8 `all` uses 2,4,1 bounds.
                all_env = dict(kv.split("=", 1) for kv in by_name["all"]["containerEdits"]["env"])
                self.assertEqual(all_env["TPU_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                self.assertEqual(all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,4,1")

                # Tampering numa0 to mix 0000:00 and 0000:80 must fail --check-host!
                tampered = json.loads(json.dumps(spec))
                for dev in tampered["devices"]:
                    if dev["name"] == "numa0":
                        dev["containerEdits"]["deviceNodes"][3] = by_name["numa1"][
                            "containerEdits"
                        ]["deviceNodes"][0]
                        bdfs = dev["annotations"]["freetpu.dev/pci-bdfs"].split(",")
                        bdfs[3] = "0000:80:00.0"
                        dev["annotations"]["freetpu.dev/pci-bdfs"] = ",".join(sorted(bdfs))
                errs_cross = freetpu_cdi.verify_spec_against_host(
                    tampered, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(
                    any("crosses NUMA / PCIe root complex boundaries" in e for e in errs_cross),
                    errs_cross,
                )

                # Swapping/renaming `numa1` -> `numa0` (and `4-7` -> `0-3`) while keeping
                # valid single-domain nodes and annotations MUST fail --check-host!
                swapped = json.loads(json.dumps(spec))
                for dev in swapped["devices"]:
                    if dev["name"] == "numa0":
                        dev["name"] = "numa1"
                    elif dev["name"] == "numa1":
                        dev["name"] = "numa0"
                    elif dev["name"] == "0-3":
                        dev["name"] = "4-7"
                    elif dev["name"] == "4-7":
                        dev["name"] = "0-3"
                errs_swap = freetpu_cdi.verify_spec_against_host(
                    swapped, sysfs_root=sysfs_root, dev_root=dev_root
                )
                for bad_alias in ("numa0", "numa1", "0-3", "4-7"):
                    self.assertTrue(
                        any(
                            f"device '{bad_alias}'" in e
                            and "does not match live host topology" in e
                            for e in errs_swap
                        ),
                        f"Expected swapped alias error for {bad_alias}, got {errs_swap}",
                    )

                # Unknown slice alias (e.g. `numa2` on a 2-NUMA host) must also fail --check-host!
                unknown_alias = json.loads(json.dumps(spec))
                for dev in unknown_alias["devices"]:
                    if dev["name"] == "numa1":
                        dev["name"] = "numa2"
                errs_unk = freetpu_cdi.verify_spec_against_host(
                    unknown_alias, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(
                    any(
                        "device 'numa2'" in e
                        and "does not match any live host topology slice" in e
                        for e in errs_unk
                    ),
                    errs_unk,
                )

            # Verify tpu7x Ironwood 4-chip VM (8 PCI functions across 0x0076 / 0x0075):
            # all 8 PCI functions remain allocated in VFIO deviceNodes while host bounds
            # use the 4 physical chips (`2,2,1`, NOT `2,4,1`).
            root_7x = root / "tpu7x"
            ironwood_devices = [
                {
                    "bdf": f"0000:00:0{4 + i:x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x0076" if i < 3 else "0x0075",
                    "driver": "vfio-pci",
                    "iommu_group": str(i),
                    "numa_node": "0",
                }
                for i in range(4)
            ] + [
                {
                    "bdf": f"0000:80:0{i:x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x0076",
                    "driver": "vfio-pci",
                    "iommu_group": str(4 + i),
                    "numa_node": "1",
                }
                for i in range(4)
            ]
            sysfs_7x, dev_7x = _make_mock_pci_vfio(
                root_7x,
                devices=ironwood_devices,
                create_vfio_nodes=[str(i) for i in range(8)],
            )
            with _mock_vfio_char_devices():
                chips_7x, ctrl_7x = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_7x, dev_root=dev_7x
                )
                self.assertEqual(len(chips_7x), 8)
                self.assertTrue(all(c.generation == "tpu7x" for c in chips_7x))
                self.assertIn("0x0075", {c.device_id for c in chips_7x})
                self.assertIn("0x0076", {c.device_id for c in chips_7x})
                spec_7x = freetpu_cdi.build_cdi_spec(
                    chips_7x, ctrl_7x, single_host_env=True
                )
                self.assertEqual(freetpu_cdi.validate_cdi_spec(spec_7x), [])
                by_name_7x = {d["name"]: d for d in spec_7x["devices"]}
                all_7x = by_name_7x["all"]
                # All 8 VFIO function nodes are attached.
                self.assertEqual(len(all_7x["containerEdits"]["deviceNodes"]), 8)
                self.assertEqual(all_7x["annotations"]["freetpu.dev/chip-count"], "4")
                self.assertEqual(all_7x["annotations"]["freetpu.dev/pci-function-count"], "8")
                all_7x_env = dict(kv.split("=", 1) for kv in all_7x["containerEdits"]["env"])
                self.assertEqual(all_7x_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(all_7x_env["TPU_VISIBLE_CHIPS"], "0,1,2,3")
                self.assertEqual(all_7x_env["TPU_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                # Each NUMA half holds 2 physical chips (4 logical devices):
                # numa0 -> chips 0,1 / logical devices 0,1,2,3; numa1 -> chips 2,3 / devices 4,5,6,7.
                numa0_7x_env = dict(
                    kv.split("=", 1) for kv in by_name_7x["numa0"]["containerEdits"]["env"]
                )
                self.assertEqual(len(by_name_7x["numa0"]["containerEdits"]["deviceNodes"]), 4)
                self.assertEqual(numa0_7x_env["TPU_CHIPS_PER_HOST_BOUNDS"], "1,2,1")
                self.assertEqual(numa0_7x_env["TPU_VISIBLE_CHIPS"], "0,1")
                self.assertEqual(numa0_7x_env["TPU_VISIBLE_DEVICES"], "0,1,2,3")

                numa1_7x_env = dict(
                    kv.split("=", 1) for kv in by_name_7x["numa1"]["containerEdits"]["env"]
                )
                self.assertEqual(numa1_7x_env["TPU_CHIPS_PER_HOST_BOUNDS"], "1,2,1")
                self.assertEqual(numa1_7x_env["TPU_VISIBLE_CHIPS"], "2,3")
                self.assertEqual(numa1_7x_env["TPU_VISIBLE_DEVICES"], "4,5,6,7")

            # Verify PCIe root resolution from sysfs ancestry when endpoints sit behind
            # PCIe switches on distinct downstream bus numbers (without numa_node files).
            # When sysfs supplies NO numa_node, `freetpu.dev/numa-node` MUST be omitted
            # and slices use `pcie0`/`pcie1` + `freetpu.dev/pcie-domain` (never inventing NUMA IDs).
            sw_root = root / "pcie_switch"
            sw_sys = sw_root / "sys"
            sw_dev = sw_root / "dev"
            (sw_sys / "bus" / "pci" / "devices").mkdir(parents=True, exist_ok=True)
            (sw_sys / "kernel" / "iommu_groups").mkdir(parents=True, exist_ok=True)
            (sw_dev / "vfio").mkdir(parents=True, exist_ok=True)
            (sw_dev / "vfio" / "vfio").write_text("", encoding="utf-8")
            misc_d = sw_sys / "class" / "misc" / "vfio" / "dev"
            misc_d.parent.mkdir(parents=True, exist_ok=True)
            misc_d.write_text("10:196\n", encoding="utf-8")
            drv_vfio = sw_sys / "bus" / "pci" / "drivers" / "vfio-pci"
            drv_vfio.mkdir(parents=True, exist_ok=True)

            switch_specs = [
                ("pci0000:00", "0000:00:01.0/0000:01:00.0", f"0000:0{2 + i}:00.0", str(i))
                for i in range(4)
            ] + [
                ("pci0000:80", "0000:80:01.0/0000:81:00.0", f"0000:8{2 + i}:00.0", str(4 + i))
                for i in range(4)
            ]
            for host_bridge, bridge_chain, bdf, grp in switch_specs:
                real_dev = sw_sys / "devices" / host_bridge / bridge_chain / bdf
                real_dev.mkdir(parents=True, exist_ok=True)
                (real_dev / "vendor").write_text("0x1ae0\n", encoding="utf-8")
                (real_dev / "device").write_text("0x006f\n", encoding="utf-8")
                os.symlink(str(drv_vfio), real_dev / "driver")
                grp_dir = sw_sys / "kernel" / "iommu_groups" / grp
                (grp_dir / "devices").mkdir(parents=True, exist_ok=True)
                os.symlink(str(real_dev), grp_dir / "devices" / bdf)
                os.symlink(str(grp_dir), real_dev / "iommu_group")
                os.symlink(str(real_dev), sw_sys / "bus" / "pci" / "devices" / bdf)
                (sw_dev / "vfio" / grp).write_text("", encoding="utf-8")
                gd = sw_sys / "class" / "vfio" / grp / "dev"
                gd.parent.mkdir(parents=True, exist_ok=True)
                gd.write_text(f"242:{grp}\n", encoding="utf-8")

            with _mock_vfio_char_devices():
                sw_chips, sw_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sw_sys, dev_root=sw_dev
                )
                self.assertEqual(
                    [c.pcie_root for c in sw_chips],
                    ["0000:00"] * 4 + ["0000:80"] * 4,
                )
                sw_spec = freetpu_cdi.build_cdi_spec(sw_chips, sw_ctrl)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        sw_spec, sysfs_root=sw_sys, dev_root=sw_dev
                    ),
                    [],
                )
                sw_by_name = {d["name"]: d for d in sw_spec["devices"]}
                self.assertTrue({"pcie0", "0-3", "pcie1", "4-7", "all"} <= set(sw_by_name), sw_by_name.keys())
                self.assertNotIn("numa0", sw_by_name)
                self.assertNotIn("numa1", sw_by_name)
                self.assertNotIn("freetpu.dev/numa-node", sw_by_name["pcie0"]["annotations"])
                self.assertNotIn("freetpu.dev/numa-node", sw_by_name["pcie1"]["annotations"])
                self.assertEqual(sw_by_name["pcie0"]["annotations"]["freetpu.dev/pcie-domain"], "0")
                self.assertEqual(sw_by_name["pcie1"]["annotations"]["freetpu.dev/pcie-domain"], "1")

            # When two PCIe roots SHARE Linux NUMA node 0, BOTH slices must keep
            # `freetpu.dev/numa-node == "0"` (never writing domain ordinal "1" as NUMA node 1).
            shared_numa_root = root / "shared_numa0"
            shared_numa_devs = [
                {
                    "bdf": f"0000:00:0{i:x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x006f",
                    "driver": "vfio-pci",
                    "iommu_group": str(i),
                    "numa_node": "0",
                }
                for i in range(2)
            ] + [
                {
                    "bdf": f"0000:80:0{i:x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x006f",
                    "driver": "vfio-pci",
                    "iommu_group": str(2 + i),
                    "numa_node": "0",
                }
                for i in range(2)
            ]
            sn_sys, sn_dev = _make_mock_pci_vfio(
                shared_numa_root,
                devices=shared_numa_devs,
                create_vfio_nodes=["0", "1", "2", "3"],
            )
            with _mock_vfio_char_devices():
                sn_chips, sn_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sn_sys, dev_root=sn_dev
                )
                sn_spec = freetpu_cdi.build_cdi_spec(sn_chips, sn_ctrl)
                sn_by_name = {d["name"]: d for d in sn_spec["devices"]}
                self.assertIn("pcie0", sn_by_name)
                self.assertIn("pcie1", sn_by_name)
                self.assertNotIn("numa1", sn_by_name)
                self.assertEqual(sn_by_name["pcie0"]["annotations"]["freetpu.dev/numa-node"], "0")
                self.assertEqual(sn_by_name["pcie1"]["annotations"]["freetpu.dev/numa-node"], "0")
                self.assertEqual(sn_by_name["pcie0"]["annotations"]["freetpu.dev/pcie-domain"], "0")
                self.assertEqual(sn_by_name["pcie1"]["annotations"]["freetpu.dev/pcie-domain"], "1")

            # Verify Kubernetes DRA example manifest uses stable resource.k8s.io/v1 API
            # and documents Topology Manager + Guaranteed QoS CPU/memory prerequisites.
            dra_path = CDI_DIR / "manifests" / "k8s-tpu-dra-example.yaml"
            dra_text = dra_path.read_text(encoding="utf-8")
            self.assertNotIn("resource.k8s.io/v1beta1", dra_text)
            self.assertEqual(dra_text.count("apiVersion: resource.k8s.io/v1"), 3)
            self.assertNotIn("basic:", dra_text)
            self.assertIn("exactly:", dra_text)
            self.assertIn("single-numa-node", dra_text)
            self.assertIn("cpuManagerPolicy", dra_text)
            self.assertIn("memoryManagerPolicy", dra_text)
            self.assertIn("Guaranteed", dra_text)
            self.assertIn('cpu: "4"', dra_text)
            self.assertIn('memory: "32Gi"', dra_text)
            self.assertIn("TPU_WORKER_HOSTNAMES=localhost", dra_text)
            self.assertIn("TPU_WORKER_ID=0", dra_text)
            self.assertIn("TPU_SKIP_MDS_QUERY=true", dra_text)

            # Verify README keeps containerd GHSA-33vj-92qq-66hc security prerequisite
            # before the general Quickstart section (not only under DaemonSet instructions).
            readme_text = (CDI_DIR / "README.md").read_text(encoding="utf-8")
            qs_idx = readme_text.index("## Quickstart")
            sec_idx = readme_text.index("GHSA-33vj-92qq-66hc")
            self.assertLess(sec_idx, qs_idx)
            self.assertNotIn("2.1.x+", readme_text)


if __name__ == "__main__":
    unittest.main()
