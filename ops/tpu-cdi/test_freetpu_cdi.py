#!/usr/bin/env python3
"""Deterministic offline self-tests for ops/tpu-cdi/freetpu_cdi.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
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
        if "chip_id" in dev:
            (d / "chip_id").write_text(dev["chip_id"] + "\n", encoding="utf-8")
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
            # PCI selector omits TPU_VISIBLE_DEVICES / TPU_CHIPS_PER_HOST_BOUNDS /
            # TPU_ACCELERATOR_TYPE / TPU_SKIP_MDS_QUERY (preserving metadata discovery
            # when topology is not injected), while `all` injects complete topology + MDS skip.
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
                ],
            )
            self.assertEqual(
                iso_spec["devices"][-1]["containerEdits"]["env"],
                [
                    "TPU_CHIPS_PER_HOST_BOUNDS=1,1,1",
                    "TPU_ACCELERATOR_TYPE=v5litepod-1",
                    "MEGASCALE_COORDINATOR_ADDRESS=127.0.0.1:8353",
                    "TPU_HOST_BOUNDS=1,1,1",
                    "TPU_WORKER_HOSTNAMES=localhost",
                    "TPU_WORKER_ID=0",
                    "TPU_SKIP_MDS_QUERY=true",
                ],
            )
            with _mock_vfio_char_devices():
                iso_mapped = freetpu_cdi.build_cdi_spec(
                    chips,
                    vfio_container,
                    single_host_env=True,
                    chip_id_map={"0000:00:06.0": 0},
                )
            self.assertEqual(
                iso_mapped["devices"][-1]["containerEdits"]["env"],
                [
                    "TPU_VISIBLE_CHIPS=0",
                    "TPU_VISIBLE_DEVICES=0",
                    "TPU_CHIPS_PER_HOST_BOUNDS=1,1,1",
                    "TPU_ACCELERATOR_TYPE=v5litepod-1",
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

        # CDI v1.1.0 cross-scope netDevices collision: global containerEdits and a
        # selected device's containerEdits apply together, so mapping two different
        # host interfaces to the same fixed container-side name MUST be rejected!
        cross_net_conflict = {
            "cdiVersion": "1.1.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                "netDevices": [{"hostInterfaceName": "eth1", "name": "tpu0"}],
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {
                        "deviceNodes": [{"path": "/dev/vfio/0"}],
                        "netDevices": [{"hostInterfaceName": "eth2", "name": "tpu0"}],
                    },
                }
            ],
        }
        cross_net_errs = freetpu_cdi.validate_cdi_spec(cross_net_conflict)
        self.assertTrue(
            any("conflicts with global containerEdits.netDevices" in e for e in cross_net_errs),
            cross_net_errs,
        )
        cross_net_ok = {
            **cross_net_conflict,
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {
                        "deviceNodes": [{"path": "/dev/vfio/0"}],
                        "netDevices": [{"hostInterfaceName": "eth2", "name": "tpu1"}],
                    },
                }
            ],
        }
        self.assertEqual(freetpu_cdi.validate_cdi_spec(cross_net_ok), [])

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

        # Root CLOS `"/"` is valid ONLY on CDI 1.1.0; CDI 0.7.0, 0.8.0, and 1.0.0
        # MUST reject `"/"` while still permitting `""` and normal Linux filenames.
        for pre_11_ver in ("0.7.0", "0.8.0", "1.0.0"):
            root_clos_pre11 = {
                "cdiVersion": pre_11_ver,
                "kind": "freetpu.dev/tpu",
                "containerEdits": {
                    "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                    "intelRdt": {"closID": "/"},
                },
                "devices": [
                    {
                        "name": "0",
                        "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                    }
                ],
            }
            pre11_errs = freetpu_cdi.validate_cdi_spec(root_clos_pre11)
            self.assertTrue(
                any("intelRdt.closID: invalid closID" in e for e in pre11_errs),
                f"Expected CDI {pre_11_ver} to reject root closID '/', got {pre11_errs}",
            )
            valid_pre11 = {
                **root_clos_pre11,
                "containerEdits": {
                    "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                    "intelRdt": {"closID": "tpu-tenant"},
                },
            }
            self.assertEqual(freetpu_cdi.validate_cdi_spec(valid_pre11), [])

        # Per CNCF CDI v1.1.0, a non-null empty `intelRdt: {}` object on a device
        # counts as a valid edit (resetting global intelRdt settings).
        empty_irdt_reset_spec = {
            "cdiVersion": "1.1.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [
                    {"path": "/dev/vfio/vfio"},
                    {"path": "/dev/vfio/0"},
                ],
                "intelRdt": {"closID": "tpu-global"},
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {"intelRdt": {}},
                }
            ],
        }
        self.assertEqual(freetpu_cdi.validate_cdi_spec(empty_irdt_reset_spec), [])

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
                    if dev_entry["name"] == "all":
                        self.assertIn("TPU_SKIP_MDS_QUERY=true", dev_entry["containerEdits"]["env"])
                    else:
                        self.assertNotIn("TPU_SKIP_MDS_QUERY=true", dev_entry["containerEdits"]["env"])

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
                # Without verified sysfs `chip_id` or complete `--chip-id-map`, even `all`
                # MUST omit TPU_VISIBLE_CHIPS and TPU_VISIBLE_DEVICES so non-contiguous
                # runtime IDs (e.g. 8,9,12,13) or workload masks are never overwritten.
                self.assertNotIn("TPU_VISIBLE_CHIPS", all_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", all_env)
                self.assertEqual(all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(all_env["TPU_ACCELERATOR_TYPE"], "v6e-4")
                self.assertEqual(all_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")
                self.assertEqual(all_env["TPU_HOST_BOUNDS"], "1,1,1")
                self.assertEqual(all_env["TPU_WORKER_HOSTNAMES"], "localhost")
                self.assertEqual(all_env["TPU_WORKER_ID"], "0")
                self.assertEqual(all_env["TPU_SKIP_MDS_QUERY"], "true")

                # Incomplete map (verifying only 1 of 4 chips as ID 4) MUST still omit
                # visibility masks on `all` (never falling back to `0,1,2,3`!).
                partial_map_spec = freetpu_cdi.build_cdi_spec(
                    chips,
                    control_node,
                    network_mode="isolated",
                    chip_id_map={"0000:00:01.0": 4},
                )
                partial_all_env = dict(
                    kv.split("=", 1) for kv in partial_map_spec["devices"][-1]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_VISIBLE_CHIPS", partial_all_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", partial_all_env)

                # Complete verified map with non-contiguous v6e runtime IDs (8,9,12,13)
                # emits those exact verified masks on `all`.
                noncontig_map = {
                    "0000:00:01.0": 8,
                    "0000:00:02.0": 9,
                    "0000:00:03.0": 12,
                    "0000:00:04.0": 13,
                }
                verified_all_spec = freetpu_cdi.build_cdi_spec(
                    chips,
                    control_node,
                    network_mode="isolated",
                    chip_id_map=noncontig_map,
                )
                verified_all_env = dict(
                    kv.split("=", 1)
                    for kv in verified_all_spec["devices"][-1]["containerEdits"]["env"]
                )
                self.assertEqual(verified_all_env["TPU_VISIBLE_CHIPS"], "8,9,12,13")
                self.assertEqual(verified_all_env["TPU_VISIBLE_DEVICES"], "8,9,12,13")

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
                self.assertNotIn("TPU_ACCELERATOR_TYPE", pci0_iso_env)
                self.assertNotIn("TPU_SKIP_MDS_QUERY", pci0_iso_env)
                self.assertEqual(pci0_iso_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")
                self.assertEqual(pci0_iso_env["TPU_WORKER_HOSTNAMES"], "localhost")
                self.assertEqual(pci0_iso_env["TPU_WORKER_ID"], "0")

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
                        "--chip-id-map",
                        json.dumps(noncontig_map),
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
                self.assertEqual(sh_all_env["TPU_VISIBLE_CHIPS"], "8,9,12,13")
                self.assertEqual(sh_all_env["TPU_VISIBLE_DEVICES"], "8,9,12,13")
                self.assertEqual(sh_all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(sh_all_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")

        rc_iso, out_iso, _ = _run_cli_inproc(["docker-args", "--device", "numa0-4"])
        self.assertEqual(rc_iso, 0)
        self.assertNotIn("--net=host", out_iso)
        self.assertIn("--device=freetpu.dev/tpu=numa0-4", out_iso)

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

                # 1b. Linux iommufd compatibility symlink (`/dev/vfio/vfio -> ../iommu`
                #     without native `misc/vfio/dev`):
                #     - Without `/dev/vfio/<group>` nodes, when every TPU exposes a verified
                #       `vfio-dev/vfioN` cdev, `vfio_mode="auto"` selects iommufd cdev.
                compat_link = dev_root / "vfio" / "vfio"
                os.symlink("../iommu", compat_link)
                try:
                    compat_chips, compat_ctrl = freetpu_cdi.discover_tpu_chips(
                        sysfs_root=sysfs_root,
                        dev_root=dev_root,
                        vfio_mode="auto",
                    )
                    self.assertEqual(compat_ctrl, dev_root / "iommu")
                    self.assertTrue(all(c.vfio_mode == "iommufd" for c in compat_chips))
                    compat_spec = freetpu_cdi.build_cdi_spec(compat_chips, compat_ctrl)
                    self.assertEqual(freetpu_cdi.validate_cdi_spec(compat_spec), [])
                    self.assertEqual(
                        freetpu_cdi.verify_spec_against_host(
                            compat_spec,
                            sysfs_root=sysfs_root,
                            dev_root=dev_root,
                        ),
                        [],
                    )
                finally:
                    compat_link.unlink(missing_ok=True)

                # 1c. Linux `CONFIG_IOMMUFD_VFIO_CONTAINER` compat host WITHOUT
                #     `CONFIG_VFIO_DEVICE_CDEV`: `/dev/vfio/vfio -> ../iommu` plus live
                #     `/dev/vfio/<group>` nodes and NO `vfio-dev/vfioN` cdevs.
                #     `vfio_mode="auto"` MUST keep legacy `/dev/vfio/<group>` access
                #     (never assuming cdevs exist!) and validate cleanly end-to-end.
                compat_grp_sys, compat_grp_dev = _make_mock_pci_vfio(
                    root / "compat_legacy_groups_only",
                    devices=[
                        {
                            "bdf": "0000:00:0b.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": "1",
                        },
                        {
                            "bdf": "0000:00:0c.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": "2",
                        },
                    ],
                    create_vfio_container=False,
                    create_vfio_nodes=["1", "2"],
                    create_iommu=True,
                    create_iommufd_cdevs=[],
                )
                os.symlink("../iommu", compat_grp_dev / "vfio" / "vfio")
                cg_chips, cg_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=compat_grp_sys,
                    dev_root=compat_grp_dev,
                    vfio_mode="auto",
                )
                self.assertEqual(cg_ctrl, compat_grp_dev / "iommu")
                self.assertTrue(all(c.vfio_mode == "vfio-group" for c in cg_chips))
                self.assertEqual(
                    [c.vfio_container_path for c in cg_chips],
                    ["/dev/vfio/1", "/dev/vfio/2"],
                )
                cg_spec = freetpu_cdi.build_cdi_spec(cg_chips, cg_ctrl)
                self.assertEqual(
                    cg_spec["containerEdits"]["deviceNodes"][0]["path"],
                    "/dev/vfio/vfio",
                )
                self.assertEqual(
                    cg_spec["containerEdits"]["deviceNodes"][0]["hostPath"],
                    str(compat_grp_dev / "iommu"),
                )
                self.assertEqual(freetpu_cdi.validate_cdi_spec(cg_spec), [])
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        cg_spec,
                        sysfs_root=compat_grp_sys,
                        dev_root=compat_grp_dev,
                    ),
                    [],
                )
                cg_out = root / "compat_legacy_groups_only.json"
                cg_ok, cg_msg = freetpu_cdi.reconcile_cdi_spec(
                    cg_out,
                    sysfs_root=compat_grp_sys,
                    dev_root=compat_grp_dev,
                    vfio_mode="auto",
                )
                self.assertTrue(cg_ok, cg_msg)

                # 1d. Decoy `/dev/vfio/vfio` symlink NOT targeting `/dev/iommu` MUST fail closed.
                decoy_target = compat_grp_dev / "decoy_node"
                decoy_target.write_text("", encoding="utf-8")
                (compat_grp_dev / "vfio" / "vfio").unlink()
                os.symlink("../decoy_node", compat_grp_dev / "vfio" / "vfio")
                with self.assertRaises(RuntimeError):
                    freetpu_cdi.discover_tpu_chips(
                        sysfs_root=compat_grp_sys,
                        dev_root=compat_grp_dev,
                        vfio_mode="auto",
                    )

                # Incomplete or malformed iommufd cdev paths (`/dev/vfio/devices/vfio`,
                # non-numeric `/dev/vfio/devices/vfioabc`, or nested paths) must fail validation.
                for bad_cdev_path in (
                    None,
                    "/dev/vfio/devices/vfio",
                    "/dev/vfio/devices/vfioabc",
                    "/dev/vfio/devices/vfio0/extra",
                ):
                    dev_edits = (
                        {"env": ["FOO=bar"]}
                        if bad_cdev_path is None
                        else {"deviceNodes": [{"path": bad_cdev_path, "type": "c"}]}
                    )
                    bad_iommufd = {
                        "cdiVersion": "0.6.0",
                        "kind": "freetpu.dev/tpu",
                        "containerEdits": {"deviceNodes": [{"path": "/dev/iommu"}]},
                        "devices": [{"name": "0", "containerEdits": dev_edits}],
                    }
                    bad_errs = freetpu_cdi.validate_cdi_spec(bad_iommufd)
                    self.assertTrue(
                        any("/dev/vfio/devices/vfioN" in e for e in bad_errs),
                        f"Expected cdev validation error for {bad_cdev_path!r}, got {bad_errs}",
                    )

            # 1e. Linux 6.8+ `CONFIG_IOMMUFD_VFIO_CONTAINER` real char `/dev/vfio/vfio`
            #     with its OWN misc devnum (`10:196`) when legacy `/dev/vfio/<group>`
            #     nodes are omitted from the device mount, while `/dev/iommu` + verified
            #     cdevs ARE present: `vfio_mode="auto"` MUST check legacy group usability
            #     and fall back to verified iommufd cdev instead of failing discovery!
            compat_char_sys, compat_char_dev = _make_mock_pci_vfio(
                root / "compat_char_no_groups_cdev_ready",
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
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
                create_vfio_container=True,   # real char `/dev/vfio/vfio` with own misc/vfio/dev!
                create_vfio_nodes=[],         # legacy `/dev/vfio/<group>` omitted!
                create_iommu=True,
                create_iommufd_cdevs=["vfio0", "vfio1"],
            )
            with _mock_vfio_char_devices():
                cc_chips, cc_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=compat_char_sys,
                    dev_root=compat_char_dev,
                    vfio_mode="auto",
                )
                self.assertEqual(cc_ctrl, compat_char_dev / "iommu")
                self.assertEqual([c.vfio_mode for c in cc_chips], ["iommufd", "iommufd"])
                cc_spec = freetpu_cdi.build_cdi_spec(cc_chips, cc_ctrl)
                self.assertEqual(freetpu_cdi.validate_cdi_spec(cc_spec), [])
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        cc_spec,
                        sysfs_root=compat_char_sys,
                        dev_root=compat_char_dev,
                    ),
                    [],
                )
                cc_out = root / "compat_char_cdev.json"
                cc_ok, cc_msg = freetpu_cdi.reconcile_cdi_spec(
                    cc_out,
                    sysfs_root=compat_char_sys,
                    dev_root=compat_char_dev,
                    vfio_mode="auto",
                )
                self.assertTrue(cc_ok, cc_msg)

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
                self.assertFalse(any(n.startswith("numa") for n in sh_names), sh_names)
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
                # Without verified sysfs `chip_id` or `--chip-id-map`, only permanently
                # domain-qualified topology selectors (`numa0-pcie-0000-00-4`,
                # `numa1-pcie-0000-80-4`, `all`) are emitted — scan-dependent shorter
                # aliases (`numa0-4`, `numa1-4`, `pcie-0000-00-4`, `pcie-0000-80-4`)
                # and discovery-order range aliases (`0-3`, `4-7`) are omitted so hardware
                # outages/returns can never silently rebind a selector to different TPUs.
                for slice_name in ("numa0-pcie-0000-00-4", "numa1-pcie-0000-80-4", "all"):
                    self.assertIn(slice_name, by_name)
                for forbidden in (
                    "numa0",
                    "numa1",
                    "numa0-4",
                    "numa1-4",
                    "pcie-0000-00-4",
                    "pcie-0000-80-4",
                    "0-3",
                    "4-7",
                ):
                    self.assertNotIn(forbidden, by_name)

                # NUMA 0 slice strictly binds 0000:00:* chips 0..3 with 2,2,1 bounds.
                numa0 = by_name["numa0-pcie-0000-00-4"]
                self.assertEqual(numa0["annotations"]["freetpu.dev/pcie-root"], "0000:00")
                self.assertEqual(numa0["annotations"]["freetpu.dev/numa-node"], "0")
                self.assertTrue(
                    all(
                        b.startswith("0000:00:")
                        for b in numa0["annotations"]["freetpu.dev/pci-bdfs"].split(",")
                    )
                )
                # Without verified sysfs `chip_id` or explicit `--chip-id-map`, sub-host
                # slices MUST omit TPU_VISIBLE_CHIPS / TPU_VISIBLE_DEVICES so workload-
                # provided visibility is preserved (never guessed from PCI discovery order),
                # while still setting slice-scoped TPU_CHIPS_PER_HOST_BOUNDS=2,2,1.
                n0_env = dict(kv.split("=", 1) for kv in numa0["containerEdits"]["env"])
                self.assertNotIn("TPU_VISIBLE_CHIPS", n0_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", n0_env)
                self.assertEqual(n0_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(n0_env["TPU_ACCELERATOR_TYPE"], "v6e-4")
                self.assertEqual(n0_env["MEGASCALE_COORDINATOR_ADDRESS"], "127.0.0.1:8353")

                # NUMA 1 slice strictly binds 0000:80:* with 2,2,1 bounds (visibility omitted).
                numa1 = by_name["numa1-pcie-0000-80-4"]
                self.assertEqual(numa1["annotations"]["freetpu.dev/pcie-root"], "0000:80")
                self.assertEqual(numa1["annotations"]["freetpu.dev/numa-node"], "1")
                self.assertTrue(
                    all(
                        b.startswith("0000:80:")
                        for b in numa1["annotations"]["freetpu.dev/pci-bdfs"].split(",")
                    )
                )
                n1_env = dict(kv.split("=", 1) for kv in numa1["containerEdits"]["env"])
                self.assertNotIn("TPU_VISIBLE_CHIPS", n1_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", n1_env)
                self.assertEqual(n1_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")

                # With explicit `--chip-id-map` where runtime IDs differ from PCI order
                # (NUMA 0 -> runtime 4,5,6,7; NUMA 1 -> runtime 0,1,2,3), sub-host slices
                # emit the VERIFIED runtime ordinals (never PCI discovery order!).
                reordered_map = {
                    **{v6e8_devices[i]["bdf"]: 4 + i for i in range(4)},
                    **{v6e8_devices[4 + i]["bdf"]: i for i in range(4)},
                }
                mapped_spec = freetpu_cdi.build_cdi_spec(
                    chips, control_node, single_host_env=True, chip_id_map=reordered_map
                )
                mapped_by_name = {d["name"]: d for d in mapped_spec["devices"]}
                m_n0_env = dict(
                    kv.split("=", 1)
                    for kv in mapped_by_name["numa0-pcie-0000-00-4"]["containerEdits"]["env"]
                )
                self.assertEqual(m_n0_env["TPU_VISIBLE_CHIPS"], "4,5,6,7")
                self.assertEqual(m_n0_env["TPU_VISIBLE_DEVICES"], "4,5,6,7")
                m_n1_env = dict(
                    kv.split("=", 1)
                    for kv in mapped_by_name["numa1-pcie-0000-80-4"]["containerEdits"]["env"]
                )
                self.assertEqual(m_n1_env["TPU_VISIBLE_CHIPS"], "0,1,2,3")
                self.assertEqual(m_n1_env["TPU_VISIBLE_DEVICES"], "0,1,2,3")
                self.assertIn("4-7", mapped_by_name)
                self.assertIn("0-3", mapped_by_name)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        mapped_spec,
                        sysfs_root=sysfs_root,
                        dev_root=dev_root,
                        chip_id_map=reordered_map,
                    ),
                    [],
                )

                # Both `generate --validate` AND `reconcile_cdi_spec` (`sync`) MUST run
                # publication-time `verify_spec_against_host` forwarding `--chip-id-map`,
                # `--iommu-dev`, and `--vfio-mode` overrides!
                chip_map_file = root / "reordered_chip_map.json"
                chip_map_file.write_text(json.dumps(reordered_map), encoding="utf-8")
                gen_out = root / "gen_mapped.json"
                with mock.patch.object(
                    freetpu_cdi,
                    "verify_spec_against_host",
                    wraps=freetpu_cdi.verify_spec_against_host,
                ) as spy_verify:
                    rc_gen, _out_gen, err_gen = _run_cli_inproc(
                        [
                            "generate",
                            "--sysfs-root",
                            str(sysfs_root),
                            "--dev-root",
                            str(dev_root),
                            "--single-host-env",
                            "--chip-id-map",
                            str(chip_map_file),
                            "--output",
                            str(gen_out),
                            "--validate",
                        ]
                    )
                    self.assertEqual(rc_gen, 0, err_gen)
                    self.assertEqual(spy_verify.call_count, 1)
                    self.assertEqual(spy_verify.call_args.kwargs["chip_id_map"], reordered_map)

                    sync_out = root / "sync_mapped.json"
                    ok_sync, msg_sync = freetpu_cdi.reconcile_cdi_spec(
                        sync_out,
                        sysfs_root=sysfs_root,
                        dev_root=dev_root,
                        single_host_env=True,
                        chip_id_map=reordered_map,
                    )
                    self.assertTrue(ok_sync, msg_sync)
                    self.assertEqual(spy_verify.call_count, 2)
                    self.assertEqual(spy_verify.call_args.kwargs["chip_id_map"], reordered_map)

                # Running `validate --check-host --remove-if-stale` WITHOUT `--chip-id-map`
                # on a live spec generated with `--chip-id-map` (containing `0-3`/`4-7`)
                # where host sysfs has no `chip_id` MUST preserve the live spec (NOT unlink it!),
                # while still unlinking on a confirmed hardware mismatch!
                rc_nomap, _out_nm, err_nm = _run_cli_inproc(
                    [
                        "validate",
                        str(gen_out),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertNotEqual(rc_nomap, 0)
                self.assertIn("Preserved live CDI spec", err_nm)
                self.assertIn("--chip-id-map", err_nm)
                self.assertTrue(gen_out.exists(), "must NOT unlink live spec when only --chip-id-map is missing")

                rc_withmap, _out_wm, err_wm = _run_cli_inproc(
                    [
                        "validate",
                        str(gen_out),
                        "--check-host",
                        "--remove-if-stale",
                        "--chip-id-map",
                        str(chip_map_file),
                        "--sysfs-root",
                        str(sysfs_root),
                        "--dev-root",
                        str(dev_root),
                    ]
                )
                self.assertEqual(rc_withmap, 0, err_wm)
                self.assertTrue(gen_out.exists())

                # Even when host sysfs DOES expose `chip_id`s (0..3 on NUMA 0, 4..7 on NUMA 1),
                # a spec generated with a swapped `--chip-id-map` (NUMA 0 -> 4..7, NUMA 1 -> 0..3)
                # records `freetpu.dev/chip-id-source=chip-id-map` on numeric range aliases so
                # `validate --check-host --remove-if-stale` without `--chip-id-map` treats the
                # alias mismatch as unverifiable and PRESERVES the healthy live spec!
                swap_root = root / "v6e8_sysfs_swapped_map"
                swap_sysfs, swap_dev = _make_mock_pci_vfio(
                    swap_root,
                    devices=[
                        {
                            "bdf": f"0000:00:{4 + i:02x}.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": str(i),
                            "numa_node": "0",
                            "chip_id": str(i),
                        }
                        for i in range(4)
                    ]
                    + [
                        {
                            "bdf": f"0000:80:{4 + i:02x}.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": str(4 + i),
                            "numa_node": "1",
                            "chip_id": str(4 + i),
                        }
                        for i in range(4)
                    ],
                    create_vfio_nodes=[str(i) for i in range(8)],
                )
                swapped_map = {
                    **{f"0000:00:{4 + i:02x}.0": 4 + i for i in range(4)},
                    **{f"0000:80:{4 + i:02x}.0": i for i in range(4)},
                }
                swapped_map_file = swap_root / "swapped-map.json"
                swapped_map_file.write_text(json.dumps(swapped_map), encoding="utf-8")
                swapped_spec_path = swap_root / "freetpu-tpu.json"
                rc_sw_gen, _o_sw, err_sw = _run_cli_inproc(
                    [
                        "generate",
                        "--sysfs-root",
                        str(swap_sysfs),
                        "--dev-root",
                        str(swap_dev),
                        "--chip-id-map",
                        str(swapped_map_file),
                        "--output",
                        str(swapped_spec_path),
                        "--validate",
                    ]
                )
                self.assertEqual(rc_sw_gen, 0, err_sw)
                sw_loaded = json.loads(swapped_spec_path.read_text(encoding="utf-8"))
                sw_by_name = {d["name"]: d for d in sw_loaded["devices"]}
                self.assertEqual(
                    sw_by_name["0-3"]["annotations"].get("freetpu.dev/chip-id-source"),
                    "chip-id-map",
                )
                self.assertEqual(
                    sw_by_name["4-7"]["annotations"].get("freetpu.dev/chip-id-source"),
                    "chip-id-map",
                )
                rc_sw_nomap, _o_swnm, err_swnm = _run_cli_inproc(
                    [
                        "validate",
                        str(swapped_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(swap_sysfs),
                        "--dev-root",
                        str(swap_dev),
                    ]
                )
                self.assertNotEqual(rc_sw_nomap, 0)
                self.assertIn("Preserved live CDI spec", err_swnm)
                self.assertIn("--chip-id-map", err_swnm)
                self.assertTrue(
                    swapped_spec_path.exists(),
                    "must preserve healthy live spec generated with swapped --chip-id-map when map is omitted",
                )
                rc_sw_withmap, _o_swwm, err_swwm = _run_cli_inproc(
                    [
                        "validate",
                        str(swapped_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--chip-id-map",
                        str(swapped_map_file),
                        "--sysfs-root",
                        str(swap_sysfs),
                        "--dev-root",
                        str(swap_dev),
                    ]
                )
                self.assertEqual(rc_sw_withmap, 0, err_swwm)
                self.assertTrue(swapped_spec_path.exists())

                # Non-contiguous `--chip-id-map` (`0,1,4,5` on NUMA 0; `2,3,6,7` on NUMA 1)
                # emits NO numeric range aliases, only `numa0-pcie-...` / `numa1-pcie-...`
                # with injected `TPU_VISIBLE_CHIPS` / `TPU_VISIBLE_DEVICES`. When the map
                # later changes (`0,1,4,5` -> `2,3,6,7`), `validate --check-host --remove-if-stale`
                # with the updated map MUST detect stale injected visibility masks and unlink!
                noncontig_map_a = {
                    "0000:00:04.0": 0,
                    "0000:00:05.0": 1,
                    "0000:00:06.0": 4,
                    "0000:00:07.0": 5,
                    "0000:80:04.0": 2,
                    "0000:80:05.0": 3,
                    "0000:80:06.0": 6,
                    "0000:80:07.0": 7,
                }
                noncontig_map_b = {
                    "0000:00:04.0": 2,
                    "0000:00:05.0": 3,
                    "0000:00:06.0": 6,
                    "0000:00:07.0": 7,
                    "0000:80:04.0": 0,
                    "0000:80:05.0": 1,
                    "0000:80:06.0": 4,
                    "0000:80:07.0": 5,
                }
                nc_map_a_file = swap_root / "nc-map-a.json"
                nc_map_b_file = swap_root / "nc-map-b.json"
                nc_map_a_file.write_text(json.dumps(noncontig_map_a), encoding="utf-8")
                nc_map_b_file.write_text(json.dumps(noncontig_map_b), encoding="utf-8")
                nc_spec_path = swap_root / "freetpu-noncontig.json"
                rc_nc_gen, _o_nc, err_nc = _run_cli_inproc(
                    [
                        "generate",
                        "--sysfs-root",
                        str(swap_sysfs),
                        "--dev-root",
                        str(swap_dev),
                        "--single-host-env",
                        "--chip-id-map",
                        str(nc_map_a_file),
                        "--output",
                        str(nc_spec_path),
                        "--validate",
                    ]
                )
                self.assertEqual(rc_nc_gen, 0, err_nc)
                nc_loaded = json.loads(nc_spec_path.read_text(encoding="utf-8"))
                nc_names = {d["name"] for d in nc_loaded["devices"]}
                self.assertNotIn("0-3", nc_names)
                self.assertNotIn("4-7", nc_names)
                # Without --chip-id-map, preserve the live map-generated spec.
                rc_nc_nomap, _o_ncnm, err_ncnm = _run_cli_inproc(
                    [
                        "validate",
                        str(nc_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(swap_sysfs),
                        "--dev-root",
                        str(swap_dev),
                    ]
                )
                self.assertNotEqual(rc_nc_nomap, 0)
                self.assertIn("Preserved live CDI spec", err_ncnm)
                self.assertTrue(nc_spec_path.exists())
                # With the UPDATED map (2,3,6,7 on NUMA 0), stale injected masks MUST
                # fail validation AND unlink the stale spec!
                rc_nc_stale, _o_ncs, err_ncs = _run_cli_inproc(
                    [
                        "validate",
                        str(nc_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--chip-id-map",
                        str(nc_map_b_file),
                        "--sysfs-root",
                        str(swap_sysfs),
                        "--dev-root",
                        str(swap_dev),
                    ]
                )
                self.assertNotEqual(rc_nc_stale, 0)
                self.assertIn("stale injected TPU_VISIBLE_CHIPS", err_ncs)
                self.assertFalse(
                    nc_spec_path.exists(),
                    "must unlink stale CDI spec when injected TPU_VISIBLE_CHIPS/DEVICES no longer match --chip-id-map",
                )

                # Continuous `sync --chip-id-map <file>` MUST reload the file before
                # each reconciliation tick (skipping transient bad JSON without crashing)
                # so operator map corrections republish updated masks immediately!
                live_map_file = swap_root / "live-sync-map.json"
                live_map_file.write_text(json.dumps(noncontig_map_a), encoding="utf-8")
                sync_out_path = swap_root / "freetpu-sync-reload.json"
                sync_sig_handlers: Dict[int, Any] = {}
                sync_ticks = {"n": 0}
                fake_clock = {"t": 0.0}

                def _fake_signal(sig_num, handler):
                    sync_sig_handlers[sig_num] = handler
                    return None

                def _fake_monotonic():
                    return fake_clock["t"]

                def _fake_sleep(_dt):
                    fake_clock["t"] += 2.0
                    sync_ticks["n"] += 1
                    if sync_ticks["n"] == 1:
                        # Tick 1 published map A; corrupt file transiently for Tick 2.
                        snap_a = json.loads(sync_out_path.read_text(encoding="utf-8"))
                        numa0_a = {d["name"]: d for d in snap_a["devices"]}["numa0-pcie-0000-00-4"]
                        env_a = dict(kv.split("=", 1) for kv in numa0_a["containerEdits"]["env"])
                        self.assertEqual(env_a["TPU_VISIBLE_CHIPS"], "0,1,4,5")
                        live_map_file.write_text("{broken-json", encoding="utf-8")
                    elif sync_ticks["n"] == 2:
                        # Tick 2 skipped bad JSON; now write corrected map B for Tick 3.
                        live_map_file.write_text(json.dumps(noncontig_map_b), encoding="utf-8")
                    else:
                        # Stop cleanly after Tick 3 published map B.
                        sync_sig_handlers[freetpu_cdi.signal.SIGTERM](
                            freetpu_cdi.signal.SIGTERM, None
                        )

                with (
                    mock.patch.object(freetpu_cdi.signal, "signal", side_effect=_fake_signal),
                    mock.patch.object(freetpu_cdi.time, "monotonic", side_effect=_fake_monotonic),
                    mock.patch.object(freetpu_cdi.time, "sleep", side_effect=_fake_sleep),
                ):
                    rc_sync_loop, _o_sl, err_sl = _run_cli_inproc(
                        [
                            "sync",
                            "--sysfs-root",
                            str(swap_sysfs),
                            "--dev-root",
                            str(swap_dev),
                            "--single-host-env",
                            "--chip-id-map",
                            str(live_map_file),
                            "--output",
                            str(sync_out_path),
                            "--interval",
                            "1",
                        ]
                    )
                self.assertEqual(rc_sync_loop, 0, err_sl)
                self.assertIn("failed to reload --chip-id-map", err_sl)
                self.assertTrue(sync_out_path.exists())
                snap_b = json.loads(sync_out_path.read_text(encoding="utf-8"))
                numa0_b = {d["name"]: d for d in snap_b["devices"]}["numa0-pcie-0000-00-4"]
                env_b = dict(kv.split("=", 1) for kv in numa0_b["containerEdits"]["env"])
                self.assertEqual(env_b["TPU_VISIBLE_CHIPS"], "2,3,6,7")
                self.assertEqual(env_b["TPU_VISIBLE_DEVICES"], "2,3,6,7")

                # If `--chip-id-map` cannot be reloaded AND live PCI/VFIO hardware
                # drifts (e.g. a TPU VFIO node disappears), continuous `sync` MUST
                # still detect confirmed hardware drift and withdraw the stale spec!
                drift_root = root / "sync_broken_map_hw_drift"
                drift_sys, drift_dev = _make_mock_pci_vfio(
                    drift_root,
                    devices=[
                        {
                            "bdf": f"0000:00:{4 + i:02x}.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": str(i),
                            "numa_node": "0",
                        }
                        for i in range(4)
                    ],
                    create_vfio_nodes=[str(i) for i in range(4)],
                )
                drift_map_file = drift_root / "drift-map.json"
                drift_map_file.write_text(
                    json.dumps({f"0000:00:{4 + i:02x}.0": i for i in range(4)}),
                    encoding="utf-8",
                )
                drift_spec_path = drift_root / "freetpu-drift.json"
                drift_sigs: Dict[int, Any] = {}
                drift_ticks = {"n": 0}
                drift_clock = {"t": 0.0}

                def _drift_signal(sig_num, handler):
                    drift_sigs[sig_num] = handler
                    return None

                def _drift_monotonic():
                    return drift_clock["t"]

                def _drift_sleep(_dt):
                    drift_clock["t"] += 2.0
                    drift_ticks["n"] += 1
                    if drift_ticks["n"] == 1:
                        self.assertTrue(drift_spec_path.exists())
                        # Break chip-id-map AND remove a VFIO group node simultaneously.
                        drift_map_file.write_text("{broken", encoding="utf-8")
                        (drift_dev / "vfio" / "3").unlink()
                    else:
                        drift_sigs[freetpu_cdi.signal.SIGTERM](
                            freetpu_cdi.signal.SIGTERM, None
                        )

                with (
                    mock.patch.object(freetpu_cdi.signal, "signal", side_effect=_drift_signal),
                    mock.patch.object(freetpu_cdi.time, "monotonic", side_effect=_drift_monotonic),
                    mock.patch.object(freetpu_cdi.time, "sleep", side_effect=_drift_sleep),
                ):
                    rc_dr, _o_dr, err_dr = _run_cli_inproc(
                        [
                            "sync",
                            "--sysfs-root",
                            str(drift_sys),
                            "--dev-root",
                            str(drift_dev),
                            "--chip-id-map",
                            str(drift_map_file),
                            "--output",
                            str(drift_spec_path),
                            "--interval",
                            "1",
                        ]
                    )
                self.assertEqual(rc_dr, 0, err_dr)
                self.assertIn("confirmed hardware drift", err_dr)
                self.assertFalse(
                    drift_spec_path.exists(),
                    "must withdraw stale CDI spec on confirmed VFIO drift even when --chip-id-map reload fails",
                )

                # Same hardware-drift revalidation MUST also run if `--chip-id-map`
                # fails to load at `sync` STARTUP (e.g. service restart after crash).
                start_drift_spec = drift_root / "freetpu-startup-drift.json"
                # Regenerate with a temporary valid map while node 3 is absent? Write a stale spec first!
                (drift_dev / "vfio" / "3").touch()
                drift_map_file.write_text(
                    json.dumps({f"0000:00:{4 + i:02x}.0": i for i in range(4)}),
                    encoding="utf-8",
                )
                rc_sd_gen, _o_sdg, err_sdg = _run_cli_inproc(
                    [
                        "generate",
                        "--sysfs-root",
                        str(drift_sys),
                        "--dev-root",
                        str(drift_dev),
                        "--chip-id-map",
                        str(drift_map_file),
                        "--output",
                        str(start_drift_spec),
                        "--validate",
                    ]
                )
                self.assertEqual(rc_sd_gen, 0, err_sdg)
                self.assertTrue(start_drift_spec.exists())
                # Now break the map file AND delete VFIO node 3 BEFORE `sync` starts.
                drift_map_file.write_text("{malformed-at-startup", encoding="utf-8")
                (drift_dev / "vfio" / "3").unlink()
                rc_sd_sync, _o_sds, err_sds = _run_cli_inproc(
                    [
                        "sync",
                        "--once",
                        "--sysfs-root",
                        str(drift_sys),
                        "--dev-root",
                        str(drift_dev),
                        "--chip-id-map",
                        str(drift_map_file),
                        "--output",
                        str(start_drift_spec),
                    ]
                )
                self.assertNotEqual(rc_sd_sync, 0)
                self.assertIn("confirmed hardware drift", err_sds)
                self.assertFalse(
                    start_drift_spec.exists(),
                    "must withdraw stale CDI spec on confirmed VFIO drift when --chip-id-map fails to load at sync startup",
                )

                # Sysfs-derived range alias in DEFAULT host-safe mode (NO --single-host-env,
                # NO injected TPU_VISIBLE_CHIPS, NO --chip-id-map): when live sysfs `chip_id`s
                # later conflict (duplicate "0"), `validate --check-host --remove-if-stale`
                # MUST withdraw the stale `0-3` range alias as confirmed drift!
                default_conf_root = root / "sysfs_conflict_default"
                default_conf_spec = default_conf_root / "freetpu-default-sysfs.json"
                sc_sys2, sc_dev2 = _make_mock_pci_vfio(
                    default_conf_root,
                    devices=[
                        {
                            "bdf": f"0000:00:{4 + i:02x}.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": str(i),
                            "numa_node": "0",
                            "chip_id": str(i),
                        }
                        for i in range(4)
                    ],
                    create_vfio_nodes=[str(i) for i in range(4)],
                )
                rc_dc_gen, _o_dcg, err_dcg = _run_cli_inproc(
                    [
                        "generate",
                        "--sysfs-root",
                        str(sc_sys2),
                        "--dev-root",
                        str(sc_dev2),
                        "--output",
                        str(default_conf_spec),
                        "--validate",
                    ]
                )
                self.assertEqual(rc_dc_gen, 0, err_dcg)
                self.assertIn(
                    "0-3",
                    {d["name"] for d in json.loads(default_conf_spec.read_text(encoding="utf-8"))["devices"]},
                )
                (sc_sys2 / "bus/pci/devices/0000:00:05.0/chip_id").write_text(
                    "0\n", encoding="utf-8"
                )
                rc_dc_val, _o_dcv, err_dcv = _run_cli_inproc(
                    [
                        "validate",
                        str(default_conf_spec),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(sc_sys2),
                        "--dev-root",
                        str(sc_dev2),
                    ]
                )
                self.assertNotEqual(rc_dc_val, 0)
                self.assertFalse(
                    default_conf_spec.exists(),
                    "must unlink default-mode sysfs-derived 0-3 alias when live sysfs chip_id values conflict",
                )

                # Sysfs-derived visibility mask (generated WITHOUT --chip-id-map):
                # if live sysfs `chip_id`s later conflict (duplicate "0"), `validate
                # --check-host --remove-if-stale` MUST treat it as confirmed drift
                # and withdraw the unsubstantiated spec!
                sysfs_conf_root = root / "sysfs_conflict_vis"
                sc_sys, sc_dev = _make_mock_pci_vfio(
                    sysfs_conf_root,
                    devices=[
                        {
                            "bdf": f"0000:00:{4 + i:02x}.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": str(i),
                            "numa_node": "0",
                            "chip_id": str(i),
                        }
                        for i in range(4)
                    ],
                    create_vfio_nodes=[str(i) for i in range(4)],
                )
                sc_spec_path = sysfs_conf_root / "freetpu-sysfs.json"
                rc_sc_gen, _o_scg, err_scg = _run_cli_inproc(
                    [
                        "generate",
                        "--sysfs-root",
                        str(sc_sys),
                        "--dev-root",
                        str(sc_dev),
                        "--single-host-env",
                        "--output",
                        str(sc_spec_path),
                        "--validate",
                    ]
                )
                self.assertEqual(rc_sc_gen, 0, err_scg)
                self.assertTrue(sc_spec_path.exists())
                # Create a live sysfs chip_id conflict (duplicate "0").
                (sc_sys / "bus/pci/devices/0000:00:05.0/chip_id").write_text(
                    "0\n", encoding="utf-8"
                )
                rc_sc_val, _o_scv, err_scv = _run_cli_inproc(
                    [
                        "validate",
                        str(sc_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(sc_sys),
                        "--dev-root",
                        str(sc_dev),
                    ]
                )
                self.assertNotEqual(rc_sc_val, 0)
                self.assertIn("conflicting or missing sysfs chip_id", err_scv)
                self.assertFalse(
                    sc_spec_path.exists(),
                    "must unlink sysfs-derived CDI spec when live sysfs chip_id values conflict",
                )

                # If publication-time host verification fails, BOTH `generate --validate`
                # and `reconcile_cdi_spec` MUST fail closed and remove any stale spec file!
                with mock.patch.object(
                    freetpu_cdi,
                    "verify_spec_against_host",
                    return_value=["injected host mismatch"],
                ):
                    rc_fail, _o_f, err_f = _run_cli_inproc(
                        [
                            "generate",
                            "--sysfs-root",
                            str(sysfs_root),
                            "--dev-root",
                            str(dev_root),
                            "--output",
                            str(gen_out),
                            "--validate",
                        ]
                    )
                    self.assertNotEqual(rc_fail, 0)
                    self.assertIn("injected host mismatch", err_f)
                    self.assertFalse(gen_out.exists())

                    ok_sync_fail, msg_sync_fail = freetpu_cdi.reconcile_cdi_spec(
                        sync_out,
                        sysfs_root=sysfs_root,
                        dev_root=dev_root,
                        chip_id_map=reordered_map,
                    )
                    self.assertFalse(ok_sync_fail)
                    self.assertIn("injected host mismatch", msg_sync_fail)
                    self.assertFalse(sync_out.exists())

                # Full 8-chip v6e-8 `all` uses 2,4,1 bounds and omits visibility when unverified;
                # with verified chip_id_map, `all` emits the verified 0..7 masks.
                all_env = dict(kv.split("=", 1) for kv in by_name["all"]["containerEdits"]["env"])
                self.assertNotIn("TPU_VISIBLE_CHIPS", all_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", all_env)
                self.assertEqual(all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,4,1")
                self.assertEqual(all_env["TPU_ACCELERATOR_TYPE"], "v6e-8")
                mapped_all_env = dict(
                    kv.split("=", 1) for kv in mapped_by_name["all"]["containerEdits"]["env"]
                )
                self.assertEqual(mapped_all_env["TPU_VISIBLE_CHIPS"], "0,1,2,3,4,5,6,7")
                self.assertEqual(mapped_all_env["TPU_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")

                # Tampering numa0-pcie-0000-00-4 to mix 0000:00 and 0000:80 must fail --check-host!
                tampered = json.loads(json.dumps(spec))
                for dev in tampered["devices"]:
                    if dev["name"] == "numa0-pcie-0000-00-4":
                        dev["containerEdits"]["deviceNodes"][3] = by_name["numa1-pcie-0000-80-4"][
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

                # Swapping/renaming `numa1-pcie-0000-80-4` <-> `numa0-pcie-0000-00-4` while keeping valid single-domain
                # nodes and annotations MUST fail --check-host!
                swapped = json.loads(json.dumps(spec))
                for dev in swapped["devices"]:
                    if dev["name"] == "numa0-pcie-0000-00-4":
                        dev["name"] = "numa1-pcie-0000-80-4"
                    elif dev["name"] == "numa1-pcie-0000-80-4":
                        dev["name"] = "numa0-pcie-0000-00-4"
                errs_swap = freetpu_cdi.verify_spec_against_host(
                    swapped, sysfs_root=sysfs_root, dev_root=dev_root
                )
                for bad_alias in ("numa0-pcie-0000-00-4", "numa1-pcie-0000-80-4"):
                    self.assertTrue(
                        any(
                            f"device '{bad_alias}'" in e
                            and "does not match live host topology" in e
                            for e in errs_swap
                        ),
                        f"Expected swapped alias error for {bad_alias}, got {errs_swap}",
                    )
                # Also verify swapping numeric range aliases (`0-3` <-> `4-7`) on a spec
                # with verified chip IDs fails --check-host!
                swapped_mapped = json.loads(json.dumps(mapped_spec))
                for dev in swapped_mapped["devices"]:
                    if dev["name"] == "0-3":
                        dev["name"] = "4-7"
                    elif dev["name"] == "4-7":
                        dev["name"] = "0-3"
                errs_swap_mapped = freetpu_cdi.verify_spec_against_host(
                    swapped_mapped,
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    chip_id_map=reordered_map,
                )
                for bad_alias in ("0-3", "4-7"):
                    self.assertTrue(
                        any(
                            f"device '{bad_alias}'" in e
                            and "does not match live host topology" in e
                            for e in errs_swap_mapped
                        ),
                        f"Expected swapped range alias error for {bad_alias}, got {errs_swap_mapped}",
                    )

                # Unknown/unpinned slice aliases (e.g. `numa2-4` or bare `numa0`) MUST fail --check-host!
                unknown_alias = json.loads(json.dumps(spec))
                for dev in unknown_alias["devices"]:
                    if dev["name"] == "numa1-pcie-0000-80-4":
                        dev["name"] = "numa2-4"
                errs_unk = freetpu_cdi.verify_spec_against_host(
                    unknown_alias, sysfs_root=sysfs_root, dev_root=dev_root
                )
                self.assertTrue(
                    any(
                        "device 'numa2-4'" in e
                        and "does not match any live host topology slice" in e
                        for e in errs_unk
                    ),
                    errs_unk,
                )

                # Degraded NUMA 0 (1 TPU disappears -> 3 chips on NUMA 0, 4 on NUMA 1):
                # `numa0-pcie-0000-00-4` and `0-2` MUST be withdrawn (never emitting unsupported 1,3,1 bounds),
                # healthy 4-chip `numa1-pcie-0000-80-4` remains published, and `all` on 7 chips omits bogus
                # TPU_CHIPS_PER_HOST_BOUNDS. Previously published 4-chip `numa0-pcie-0000-00-4` fails --check-host!
                deg_root = root / "v6e8_degraded_numa0"
                deg_sys, deg_dev = _make_mock_pci_vfio(
                    deg_root,
                    devices=v6e8_devices[1:],  # drop 1 chip from NUMA 0 (leaving 3 + 4 = 7)
                    create_vfio_nodes=[str(i) for i in range(1, 8)],
                )
                deg_chips, deg_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=deg_sys, dev_root=deg_dev
                )
                deg_spec = freetpu_cdi.build_cdi_spec(
                    deg_chips, deg_ctrl, single_host_env=True
                )
                deg_by_name = {d["name"]: d for d in deg_spec["devices"]}
                self.assertNotIn("numa0-pcie-0000-00-4", deg_by_name)
                self.assertNotIn("numa0-pcie-0000-00-3", deg_by_name)
                self.assertNotIn("0-2", deg_by_name)
                self.assertIn("numa1-pcie-0000-80-4", deg_by_name)
                self.assertEqual(
                    deg_by_name["numa1-pcie-0000-80-4"]["annotations"]["freetpu.dev/chip-count"],
                    "4",
                )
                deg_all_env = dict(
                    kv.split("=", 1) for kv in deg_by_name["all"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", deg_all_env)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        deg_spec, sysfs_root=deg_sys, dev_root=deg_dev
                    ),
                    [],
                )
                errs_deg_old = freetpu_cdi.verify_spec_against_host(
                    spec, sysfs_root=deg_sys, dev_root=deg_dev
                )
                self.assertTrue(
                    any("device 'numa0-pcie-0000-00-4'" in e for e in errs_deg_old),
                    errs_deg_old,
                )
                # Confirmed hardware degradation on a chip-id-map spec STILL unlinks under
                # `validate --check-host --remove-if-stale` even without `--chip-id-map`!
                deg_mapped_copy = root / "deg_mapped_copy.json"
                deg_mapped_copy.write_text(json.dumps(mapped_spec), encoding="utf-8")
                rc_deg_rm, _o_drm, err_drm = _run_cli_inproc(
                    [
                        "validate",
                        str(deg_mapped_copy),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(deg_sys),
                        "--dev-root",
                        str(deg_dev),
                    ]
                )
                self.assertNotEqual(rc_deg_rm, 0)
                self.assertIn("Removed stale CDI spec", err_drm)
                self.assertFalse(deg_mapped_copy.exists())

                # Degraded NUMA 0 (2 TPUs disappear -> 2 chips on NUMA 0, 4 on NUMA 1):
                # Per Google Cloud v6e docs, single-VM v6e supports {1, 4, 8} chips ONLY
                # (2 surviving chips in a degraded 2x2 slice could be a torus diagonal!).
                # BOTH `numa0-pcie-0000-00-4` AND `numa0-pcie-0000-00-2` MUST be withdrawn,
                # while healthy 4-chip `numa1-pcie-0000-80-4` stays published!
                deg2_root = root / "v6e8_degraded_numa0_two_chips"
                deg2_sys, deg2_dev = _make_mock_pci_vfio(
                    deg2_root,
                    devices=v6e8_devices[2:],  # drop 2 chips from NUMA 0 (leaving 2 + 4 = 6)
                    create_vfio_nodes=[str(i) for i in range(2, 8)],
                )
                deg2_out = root / "v6e8_degraded_numa0_two_chips.json"
                ok_deg2, msg_deg2 = freetpu_cdi.reconcile_cdi_spec(
                    deg2_out,
                    sysfs_root=deg2_sys,
                    dev_root=deg2_dev,
                    single_host_env=True,
                )
                self.assertTrue(ok_deg2, msg_deg2)
                deg2_spec = json.loads(deg2_out.read_text(encoding="utf-8"))
                deg2_by_name = {d["name"]: d for d in deg2_spec["devices"]}
                self.assertNotIn("numa0-pcie-0000-00-4", deg2_by_name)
                self.assertNotIn("numa0-pcie-0000-00-2", deg2_by_name)
                self.assertNotIn("numa0", deg2_by_name)
                self.assertIn("numa1-pcie-0000-80-4", deg2_by_name)
                self.assertEqual(
                    deg2_by_name["numa1-pcie-0000-80-4"]["annotations"]["freetpu.dev/chip-count"],
                    "4",
                )
                deg2_all_env = dict(
                    kv.split("=", 1) for kv in deg2_by_name["all"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", deg2_all_env)
                self.assertNotIn("TPU_ACCELERATOR_TYPE", deg2_all_env)
                errs_deg2_old = freetpu_cdi.verify_spec_against_host(
                    spec, sysfs_root=deg2_sys, dev_root=deg2_dev
                )
                self.assertTrue(
                    any("device 'numa0-pcie-0000-00-4'" in e for e in errs_deg2_old),
                    errs_deg2_old,
                )

                # With verified sysfs `chip_id` (0..3 on NUMA 0, 4..7 on NUMA 1), losing
                # chip 0 on NUMA 0 MUST keep healthy NUMA 1 stable as `numa1-pcie-0000-80-4` + `4-7`
                # and NEVER renumber it to `3-6`!
                deg_vid_root = root / "v6e8_degraded_verified_ids"
                v6e8_verified = [
                    {**d, "chip_id": str(idx)} for idx, d in enumerate(v6e8_devices)
                ]
                deg_vid_sys, deg_vid_dev = _make_mock_pci_vfio(
                    deg_vid_root,
                    devices=v6e8_verified[1:],  # chip 0 gone; NUMA 0 has 1,2,3; NUMA 1 has 4,5,6,7
                    create_vfio_nodes=[str(i) for i in range(1, 8)],
                )
                deg_vid_chips, deg_vid_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=deg_vid_sys, dev_root=deg_vid_dev
                )
                deg_vid_spec = freetpu_cdi.build_cdi_spec(
                    deg_vid_chips, deg_vid_ctrl, single_host_env=True
                )
                deg_vid_by_name = {d["name"]: d for d in deg_vid_spec["devices"]}
                self.assertNotIn("numa0-pcie-0000-00-4", deg_vid_by_name)
                self.assertNotIn("0-2", deg_vid_by_name)
                self.assertNotIn("3-6", deg_vid_by_name)
                self.assertIn("numa1-pcie-0000-80-4", deg_vid_by_name)
                self.assertIn("4-7", deg_vid_by_name)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        deg_vid_spec, sysfs_root=deg_vid_sys, dev_root=deg_vid_dev
                    ),
                    [],
                )

                # Losing EVERY PCI function on NUMA 0 (`0000:00`) leaves only NUMA 1's
                # 4 healthy chips (`0000:80`). `sync` MUST preserve `numa1-pcie-0000-80-4` and its
                # verified `4-7` range alias so surviving DRA/Docker consumers keep
                # running, while withdrawing `numa0-pcie-0000-00-4` / `0-3`!
                lost_root0_sys, lost_root0_dev = _make_mock_pci_vfio(
                    root / "v6e8_lost_entire_numa0",
                    devices=v6e8_verified[4:],  # NUMA 0 completely gone; NUMA 1 intact (4..7)
                    create_vfio_nodes=[str(i) for i in range(4, 8)],
                )
                lost_root0_out = root / "v6e8_lost_entire_numa0.json"
                ok_lr0, msg_lr0 = freetpu_cdi.reconcile_cdi_spec(
                    lost_root0_out,
                    sysfs_root=lost_root0_sys,
                    dev_root=lost_root0_dev,
                    single_host_env=True,
                )
                self.assertTrue(ok_lr0, msg_lr0)
                lr0_spec = json.loads(lost_root0_out.read_text(encoding="utf-8"))
                lr0_by_name = {d["name"]: d for d in lr0_spec["devices"]}
                self.assertNotIn("numa0-pcie-0000-00-4", lr0_by_name)
                self.assertNotIn("0-3", lr0_by_name)
                self.assertIn("numa1-pcie-0000-80-4", lr0_by_name)
                self.assertIn("4-7", lr0_by_name)
                self.assertEqual(
                    lr0_by_name["numa1-pcie-0000-80-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:80",
                )
                self.assertEqual(
                    lr0_by_name["numa1-pcie-0000-80-4"]["annotations"]["freetpu.dev/numa-node"],
                    "1",
                )
                lr0_n1_env = dict(
                    kv.split("=", 1)
                    for kv in lr0_by_name["numa1-pcie-0000-80-4"]["containerEdits"]["env"]
                )
                self.assertEqual(lr0_n1_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(lr0_n1_env["TPU_ACCELERATOR_TYPE"], "v6e-4")
                self.assertEqual(lr0_n1_env["TPU_VISIBLE_CHIPS"], "4,5,6,7")
                self.assertEqual(lr0_n1_env["TPU_VISIBLE_DEVICES"], "4,5,6,7")

                # Degraded v6e-8 with 4 surviving chips split cross-root (3 on NUMA 0 +
                # 1 on NUMA 1, OR 2 on NUMA 0 + 2 on NUMA 1): `all` MUST NOT inject
                # `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1` or `TPU_ACCELERATOR_TYPE=v6e-4`
                # because a valid v6e-4 2x2 slice must reside within one NUMA/PCIe root!
                for label, subset_indices in (
                    ("split_3_1", [0, 1, 2, 4]),
                    ("split_2_2", [0, 1, 4, 5]),
                ):
                    split_sys, split_dev = _make_mock_pci_vfio(
                        root / f"v6e8_degraded_{label}",
                        devices=[v6e8_verified[i] for i in subset_indices],
                        create_vfio_nodes=[str(i) for i in subset_indices],
                    )
                    split_chips, split_ctrl = freetpu_cdi.discover_tpu_chips(
                        sysfs_root=split_sys, dev_root=split_dev
                    )
                    split_spec = freetpu_cdi.build_cdi_spec(
                        split_chips, split_ctrl, single_host_env=True
                    )
                    split_by_name = {d["name"]: d for d in split_spec["devices"]}
                    self.assertNotIn("numa0-pcie-0000-00-4", split_by_name)
                    self.assertNotIn("numa1-pcie-0000-80-4", split_by_name)
                    split_all_env = dict(
                        kv.split("=", 1) for kv in split_by_name["all"]["containerEdits"]["env"]
                    )
                    self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", split_all_env)
                    self.assertNotIn("TPU_ACCELERATOR_TYPE", split_all_env)
                    self.assertNotIn("TPU_VISIBLE_CHIPS", split_all_env)
                    self.assertNotIn("TPU_VISIBLE_DEVICES", split_all_env)
                    self.assertEqual(
                        freetpu_cdi.verify_spec_against_host(
                            split_spec, sysfs_root=split_sys, dev_root=split_dev
                        ),
                        [],
                    )

                # Standard 4-chip `v5p` VM (`0x0062`) spans two NUMA nodes (2 chips on
                # NUMA 0 + 2 chips on NUMA 1): `--single-host-env` MUST recognize this
                # documented `[2, 2]` v5p host topology on `all` and emit `2,2,1` and
                # `v5p-8` (while still omitting unsupported 2-chip sub-host slices!).
                v5p_devices = [
                    {
                        "bdf": f"0000:00:0{4 + i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0062",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                        "numa_node": "0",
                        "chip_id": str(i),
                    }
                    for i in range(2)
                ] + [
                    {
                        "bdf": f"0000:80:0{i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0062",
                        "driver": "vfio-pci",
                        "iommu_group": str(2 + i),
                        "numa_node": "1",
                        "chip_id": str(2 + i),
                    }
                    for i in range(2)
                ]
                v5p_sys, v5p_dev = _make_mock_pci_vfio(
                    root / "v5p_4chip_dual_numa",
                    devices=v5p_devices,
                    create_vfio_nodes=["0", "1", "2", "3"],
                )
                v5p_chips, v5p_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=v5p_sys, dev_root=v5p_dev
                )
                v5p_spec = freetpu_cdi.build_cdi_spec(
                    v5p_chips, v5p_ctrl, single_host_env=True
                )
                v5p_by_name = {d["name"]: d for d in v5p_spec["devices"]}
                self.assertNotIn("numa0-2", v5p_by_name)
                self.assertNotIn("numa1-2", v5p_by_name)
                v5p_all_env = dict(
                    kv.split("=", 1) for kv in v5p_by_name["all"]["containerEdits"]["env"]
                )
                self.assertEqual(v5p_all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(v5p_all_env["TPU_ACCELERATOR_TYPE"], "v5p-8")
                self.assertEqual(v5p_all_env["TPU_VISIBLE_CHIPS"], "0,1,2,3")
                self.assertEqual(v5p_all_env["TPU_VISIBLE_DEVICES"], "0,1,2,3")
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        v5p_spec, sysfs_root=v5p_sys, dev_root=v5p_dev
                    ),
                    [],
                )

            # Verify tpu7x Ironwood 4-chip VM (8 PCI functions across 0x0076 / 0x0075):
            # sibling chiplets share PCI slots (`04.0/04.1`, `05.0/05.1` on NUMA 0;
            # `80:00.0/00.1`, `80:01.0/01.1` on NUMA 1); all 8 PCI functions remain
            # allocated in VFIO deviceNodes while host bounds use the 4 physical chips
            # (`2,2,1`, NOT `2,4,1`), and individual single-function `pci-*` selectors
            # are labeled `single-chiplet` rather than `single-chip`.
            root_7x = root / "tpu7x"
            ironwood_devices = [
                {
                    "bdf": f"0000:00:0{4 + (i // 2):x}.{i % 2}",
                    "vendor": "0x1ae0",
                    "device": "0x0076" if i < 3 else "0x0075",
                    "driver": "vfio-pci",
                    "iommu_group": str(i),
                    "numa_node": "0",
                    "chip_id": str(i),
                }
                for i in range(4)
            ] + [
                {
                    "bdf": f"0000:80:0{i // 2:x}.{i % 2}",
                    "vendor": "0x1ae0",
                    "device": "0x0076",
                    "driver": "vfio-pci",
                    "iommu_group": str(4 + i),
                    "numa_node": "1",
                    "chip_id": str(4 + i),
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
                # Individual Ironwood single-function `pci-*` selectors MUST be labeled
                # `single-chiplet` (never `single-chip`), and omit `TPU_SKIP_MDS_QUERY`.
                pci_7x = by_name_7x["pci-0000-00-04-0"]
                self.assertEqual(pci_7x["annotations"]["freetpu.dev/allocation-unit"], "single-chiplet")
                self.assertNotIn("TPU_SKIP_MDS_QUERY=true", pci_7x["containerEdits"]["env"])
                all_7x = by_name_7x["all"]
                # All 8 VFIO function nodes are attached.
                self.assertEqual(len(all_7x["containerEdits"]["deviceNodes"]), 8)
                self.assertEqual(all_7x["annotations"]["freetpu.dev/chip-count"], "4")
                self.assertEqual(all_7x["annotations"]["freetpu.dev/pci-function-count"], "8")
                self.assertNotIn("freetpu.dev/capacity-complete", all_7x["annotations"])
                self.assertNotIn("freetpu.dev/incomplete-pci-functions", all_7x["annotations"])
                all_7x_env = dict(kv.split("=", 1) for kv in all_7x["containerEdits"]["env"])
                self.assertEqual(all_7x_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(all_7x_env["TPU_ACCELERATOR_TYPE"], "tpu7x-8")
                self.assertEqual(all_7x_env["TPU_VISIBLE_CHIPS"], "0,1,2,3")
                self.assertEqual(all_7x_env["TPU_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                # PyTorch/XLA `torch_xla/_internal/tpu.py` version regex must match every emitted type.
                pytorch_xla_re = re.compile(r"^(?:v|tpu)(\d)([A-Za-z]?){7}-(\d+)$")
                for atype in ("v5litepod-1", "v6e-4", "v6e-8", "v5p-8", "tpu7x-8"):
                    self.assertRegex(atype, pytorch_xla_re)
                # Without a verified ICI mesh topology, 2-chip Ironwood sub-host slice
                # presets (`numa0-2`/`numa1-2`, `1,2,1`, `tpu7x-4`) MUST NOT be emitted;
                # raw per-chiplet selectors and full-host 4-chip `all` (`2,2,1`/`tpu7x-8`)
                # remain published!
                self.assertNotIn("numa0-2", by_name_7x)
                self.assertNotIn("numa1-2", by_name_7x)
                self.assertNotIn("0-1", by_name_7x)
                self.assertNotIn("2-3", by_name_7x)

                # Ironwood 4-chip host missing 1 PCI function/chiplet in NUMA 1 (only 7 VFIO nodes):
                # `all` MUST omit TPU_VISIBLE_CHIPS, TPU_VISIBLE_DEVICES, TPU_CHIPS_PER_HOST_BOUNDS,
                # TPU_ACCELERATOR_TYPE, and TPU_SKIP_MDS_QUERY (never advertising 0..7 or 2,2,1!).
                root_7x_deg = root / "tpu7x_missing_chiplet"
                sysfs_7x_deg, dev_7x_deg = _make_mock_pci_vfio(
                    root_7x_deg,
                    devices=ironwood_devices[:7],  # drop 1 chiplet from NUMA 1 -> 7 functions
                    create_vfio_nodes=[str(i) for i in range(7)],
                )
                chips_7x_deg, ctrl_7x_deg = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_7x_deg, dev_root=dev_7x_deg
                )
                spec_7x_deg = freetpu_cdi.build_cdi_spec(
                    chips_7x_deg, ctrl_7x_deg, single_host_env=True
                )
                by_name_7x_deg = {d["name"]: d for d in spec_7x_deg["devices"]}
                self.assertEqual(len(by_name_7x_deg["all"]["containerEdits"]["deviceNodes"]), 7)
                all_7x_deg_env = dict(
                    kv.split("=", 1) for kv in by_name_7x_deg["all"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_VISIBLE_CHIPS", all_7x_deg_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", all_7x_deg_env)
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", all_7x_deg_env)
                self.assertNotIn("TPU_ACCELERATOR_TYPE", all_7x_deg_env)
                self.assertNotIn("TPU_SKIP_MDS_QUERY", all_7x_deg_env)
                self.assertNotIn("numa1-2", by_name_7x_deg)
                self.assertNotIn("numa0-2", by_name_7x_deg)

                # Degraded 2-chip Ironwood host (only NUMA 1's 4 functions / 2 chips survive):
                # MUST NOT emit `numa1-2`, `1,2,1`, `tpu7x-4`, or `TPU_SKIP_MDS_QUERY=true`
                # without a verified ICI topology!
                root_7x_deg0 = root / "tpu7x_only_two_chips"
                sysfs_7x_deg0, dev_7x_deg0 = _make_mock_pci_vfio(
                    root_7x_deg0,
                    devices=ironwood_devices[4:],
                    create_vfio_nodes=[str(i) for i in range(4, 8)],
                )
                chips_7x_deg0, ctrl_7x_deg0 = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sysfs_7x_deg0, dev_root=dev_7x_deg0
                )
                spec_7x_deg0 = freetpu_cdi.build_cdi_spec(
                    chips_7x_deg0, ctrl_7x_deg0, single_host_env=True
                )
                by_name_7x_deg0 = {d["name"]: d for d in spec_7x_deg0["devices"]}
                self.assertNotIn("numa0-2", by_name_7x_deg0)
                self.assertNotIn("numa1-2", by_name_7x_deg0)
                self.assertNotIn("2-3", by_name_7x_deg0)
                all_7x_deg0_env = dict(
                    kv.split("=", 1) for kv in by_name_7x_deg0["all"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_VISIBLE_CHIPS", all_7x_deg0_env)
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", all_7x_deg0_env)
                self.assertNotIn("TPU_ACCELERATOR_TYPE", all_7x_deg0_env)
                self.assertNotIn("TPU_SKIP_MDS_QUERY", all_7x_deg0_env)

                # Degraded 1-chip Ironwood (`tpu7x`, 2 sibling PCI functions = 1 physical chip)
                # and degraded 1-chip `v5p` hosts: both generations are strictly 4-chip-per-VM
                # (`tpu7x-8`, `v5p-8`), so `all` is still published with its devices, but
                # `--single-host-env` MUST NOT inject unsupported `tpu7x-2`/`v5p-2`/`1,1,1`
                # bounds or `TPU_SKIP_MDS_QUERY=true`.
                one_chip_7x_devs = [
                    {
                        "bdf": "0000:00:04.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "numa_node": "0",
                    },
                    {
                        "bdf": "0000:00:04.1",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                        "numa_node": "0",
                    },
                ]
                o7x_sys, o7x_dev = _make_mock_pci_vfio(
                    root / "tpu7x_one_chip_degraded",
                    devices=one_chip_7x_devs,
                    create_vfio_nodes=["0", "1"],
                )
                o7x_chips, o7x_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=o7x_sys, dev_root=o7x_dev
                )
                o7x_spec = freetpu_cdi.build_cdi_spec(
                    o7x_chips, o7x_ctrl, single_host_env=True
                )
                o7x_all = {d["name"]: d for d in o7x_spec["devices"]}["all"]
                o7x_env = dict(
                    kv.split("=", 1) for kv in o7x_all["containerEdits"].get("env", [])
                )
                self.assertNotIn("TPU_ACCELERATOR_TYPE", o7x_env)
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", o7x_env)
                self.assertNotIn("TPU_SKIP_MDS_QUERY", o7x_env)

                one_chip_v5p_sys, one_chip_v5p_dev = _make_mock_pci_vfio(
                    root / "v5p_one_chip_degraded",
                    devices=[
                        {
                            "bdf": "0000:00:04.0",
                            "vendor": "0x1ae0",
                            "device": "0x0062",
                            "driver": "vfio-pci",
                            "iommu_group": "0",
                            "numa_node": "0",
                        }
                    ],
                    create_vfio_nodes=["0"],
                )
                ov5p_chips, ov5p_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=one_chip_v5p_sys, dev_root=one_chip_v5p_dev
                )
                ov5p_spec = freetpu_cdi.build_cdi_spec(
                    ov5p_chips, ov5p_ctrl, single_host_env=True
                )
                ov5p_all = {d["name"]: d for d in ov5p_spec["devices"]}["all"]
                ov5p_env = dict(
                    kv.split("=", 1) for kv in ov5p_all["containerEdits"].get("env", [])
                )
                self.assertNotIn("TPU_ACCELERATOR_TYPE", ov5p_env)
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", ov5p_env)
                self.assertNotIn("TPU_SKIP_MDS_QUERY", ov5p_env)

                # Orphaned Ironwood functions from TWO distinct degraded chips in the same
                # domain (1 intact chip + 1 orphan from chip 1 + 1 orphan from chip 2 = 4
                # functions total) MUST NOT be falsely paired into a 2nd complete chip!
                # Test both multifunction slots (no chip_id) AND verified chip_id cases.
                mf_orphan_devs = [
                    {
                        "bdf": "0000:00:04.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "numa_node": "0",
                    },
                    {
                        "bdf": "0000:00:04.1",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                        "numa_node": "0",
                    },
                    {
                        "bdf": "0000:00:05.0",  # orphan from chip 1 (05.1 missing)
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": "2",
                        "numa_node": "0",
                    },
                    {
                        "bdf": "0000:00:06.0",  # orphan from chip 2 (06.1 missing)
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                        "numa_node": "0",
                    },
                ]
                mfo_sys, mfo_dev = _make_mock_pci_vfio(
                    root / "tpu7x_multifunc_orphans",
                    devices=mf_orphan_devs,
                    create_vfio_nodes=["0", "1", "2", "3"],
                )
                mfo_chips, mfo_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=mfo_sys, dev_root=mfo_dev
                )
                mfo_spec = freetpu_cdi.build_cdi_spec(
                    mfo_chips, mfo_ctrl, single_host_env=True
                )
                mfo_by_name = {d["name"]: d for d in mfo_spec["devices"]}
                self.assertNotIn("numa0-2", mfo_by_name)
                mfo_all_env = dict(
                    kv.split("=", 1) for kv in mfo_by_name["all"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", mfo_all_env)
                self.assertNotIn("TPU_ACCELERATOR_TYPE", mfo_all_env)
                # 1 complete chip (04.0+04.1) + 2 orphaned functions (05.0, 06.0) -> chip-count=1,
                # NOT 3, with pci-function-count=4, incomplete-pci-functions=2, capacity-complete=false.
                mfo_all_ann = mfo_by_name["all"]["annotations"]
                self.assertEqual(mfo_all_ann["freetpu.dev/chip-count"], "1")
                self.assertEqual(mfo_all_ann["freetpu.dev/pci-function-count"], "4")
                self.assertEqual(mfo_all_ann["freetpu.dev/incomplete-pci-functions"], "2")
                self.assertEqual(mfo_all_ann["freetpu.dev/capacity-complete"], "false")

                # Separate-slot Ironwood with verified chip_ids {0, 1, 2, 4} (intact chip 0
                # + orphan 2 from chip 1 + orphan 4 from chip 2): MUST NOT pair 2 & 4!
                id_orphan_devs = [
                    {
                        "bdf": f"0000:00:0{4 + idx:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(idx),
                        "numa_node": "0",
                        "chip_id": str(cid),
                    }
                    for idx, cid in enumerate((0, 1, 2, 4))
                ]
                ido_sys, ido_dev = _make_mock_pci_vfio(
                    root / "tpu7x_id_orphans",
                    devices=id_orphan_devs,
                    create_vfio_nodes=["0", "1", "2", "3"],
                )
                ido_chips, ido_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=ido_sys, dev_root=ido_dev
                )
                ido_spec = freetpu_cdi.build_cdi_spec(
                    ido_chips, ido_ctrl, single_host_env=True
                )
                ido_by_name = {d["name"]: d for d in ido_spec["devices"]}
                self.assertNotIn("numa0-2", ido_by_name)
                ido_all_env = dict(
                    kv.split("=", 1) for kv in ido_by_name["all"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", ido_all_env)
                self.assertNotIn("TPU_ACCELERATOR_TYPE", ido_all_env)

                # Degraded separate-slot Ironwood with 1 surviving function per chip and
                # unique physical chip IDs {0, 1, 2, 3}: MUST NOT pair by `rid // 2` into
                # two bogus "complete" chips (and MUST omit bounds, accel type, and MDS skip)!
                id_unique_4_devs = [
                    {
                        "bdf": f"0000:00:0{4 + idx:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(idx),
                        "numa_node": "0",
                        "chip_id": str(idx),
                    }
                    for idx in range(4)
                ]
                idu_sys, idu_dev = _make_mock_pci_vfio(
                    root / "tpu7x_unique_physical_id_orphans",
                    devices=id_unique_4_devs,
                    create_vfio_nodes=["0", "1", "2", "3"],
                )
                idu_chips, idu_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=idu_sys, dev_root=idu_dev
                )
                idu_spec = freetpu_cdi.build_cdi_spec(
                    idu_chips, idu_ctrl, single_host_env=True
                )
                idu_by_name = {d["name"]: d for d in idu_spec["devices"]}
                self.assertNotIn("numa0-2", idu_by_name)
                idu_all_env = dict(
                    kv.split("=", 1) for kv in idu_by_name["all"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_CHIPS_PER_HOST_BOUNDS", idu_all_env)
                self.assertNotIn("TPU_ACCELERATOR_TYPE", idu_all_env)
                self.assertNotIn("TPU_SKIP_MDS_QUERY", idu_all_env)
                # 4 unpaired/orphaned Ironwood functions (0 complete chips) MUST advertise
                # chip-count=0, pci-function-count=4, incomplete-pci-functions=4, capacity-complete=false!
                idu_all_ann = idu_by_name["all"]["annotations"]
                self.assertEqual(idu_all_ann["freetpu.dev/chip-count"], "0")
                self.assertEqual(idu_all_ann["freetpu.dev/pci-function-count"], "4")
                self.assertEqual(idu_all_ann["freetpu.dev/incomplete-pci-functions"], "4")
                self.assertEqual(idu_all_ann["freetpu.dev/capacity-complete"], "false")

                # Separate-slot Ironwood with explicit SHARED physical chip IDs
                # (0,0,1,1 on NUMA 0; 2,2,3,3 on NUMA 1): pairs verified siblings cleanly!
                shared_id_7x_devs = [
                    {
                        "bdf": f"0000:00:0{4 + idx:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(idx),
                        "numa_node": "0",
                        "chip_id": str(idx // 2),
                    }
                    for idx in range(4)
                ] + [
                    {
                        "bdf": f"0000:80:0{idx:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(4 + idx),
                        "numa_node": "1",
                        "chip_id": str(2 + (idx // 2)),
                    }
                    for idx in range(4)
                ]
                sid_sys, sid_dev = _make_mock_pci_vfio(
                    root / "tpu7x_shared_physical_ids",
                    devices=shared_id_7x_devs,
                    create_vfio_nodes=[str(i) for i in range(8)],
                )
                sid_chips, sid_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sid_sys, dev_root=sid_dev
                )
                sid_spec = freetpu_cdi.build_cdi_spec(
                    sid_chips, sid_ctrl, single_host_env=True
                )
                sid_by_name = {d["name"]: d for d in sid_spec["devices"]}
                self.assertNotIn("numa0-2", sid_by_name)
                self.assertNotIn("numa1-2", sid_by_name)
                sid_all_env = dict(
                    kv.split("=", 1) for kv in sid_by_name["all"]["containerEdits"]["env"]
                )
                self.assertEqual(sid_all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(sid_all_env["TPU_ACCELERATOR_TYPE"], "tpu7x-8")
                self.assertEqual(sid_all_env["TPU_SKIP_MDS_QUERY"], "true")

                # Supplying a distinct logical `--chip-id-map` (e.g. logical device IDs 8..15)
                # on that same separate-slot Ironwood host MUST preserve sysfs physical sibling
                # pairing (chip-count=4, full isolated topology) while applying the logical map
                # to `TPU_VISIBLE_CHIPS` / `TPU_VISIBLE_DEVICES`!
                sid_logical_map = {
                    dev["bdf"]: 8 + idx for idx, dev in enumerate(shared_id_7x_devs)
                }
                sid_mapped_spec = freetpu_cdi.build_cdi_spec(
                    sid_chips,
                    sid_ctrl,
                    single_host_env=True,
                    chip_id_map=sid_logical_map,
                )
                sid_mapped_all = {d["name"]: d for d in sid_mapped_spec["devices"]}["all"]
                self.assertEqual(sid_mapped_all["annotations"]["freetpu.dev/chip-count"], "4")
                self.assertEqual(
                    sid_mapped_all["annotations"]["freetpu.dev/pci-function-count"], "8"
                )
                self.assertNotIn("freetpu.dev/capacity-complete", sid_mapped_all["annotations"])

                # Separate-slot 4-chip Ironwood host (8 PCI functions on NUMA 0 / 0000:00,
                # NO sysfs `chip_id`) where ONLY `--chip-id-map` pairs functions into 4 chips:
                # `validate --check-host --remove-if-stale` WITHOUT `--chip-id-map` MUST treat
                # missing chip pairing as unverifiable and PRESERVE the live spec (NOT unlink it!)
                # while validating cleanly with `--chip-id-map`!
                iw_pair_root = root / "tpu7x_map_only_pairing"
                iw_pair_devs = [
                    {
                        "bdf": f"0000:00:{4 + idx:02x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(idx),
                        "numa_node": "0",
                    }
                    for idx in range(8)
                ]
                iwp_sys, iwp_dev = _make_mock_pci_vfio(
                    iw_pair_root,
                    devices=iw_pair_devs,
                    create_vfio_nodes=[str(i) for i in range(8)],
                )
                iwp_map = {dev["bdf"]: idx // 2 for idx, dev in enumerate(iw_pair_devs)}
                iwp_map_file = iw_pair_root / "pair-map.json"
                iwp_map_file.write_text(json.dumps(iwp_map), encoding="utf-8")
                iwp_spec_path = iw_pair_root / "freetpu-tpu.json"
                rc_iwp_gen, _o_iwp, err_iwp = _run_cli_inproc(
                    [
                        "generate",
                        "--sysfs-root",
                        str(iwp_sys),
                        "--dev-root",
                        str(iwp_dev),
                        "--chip-id-map",
                        str(iwp_map_file),
                        "--output",
                        str(iwp_spec_path),
                        "--validate",
                    ]
                )
                self.assertEqual(rc_iwp_gen, 0, err_iwp)
                iwp_loaded = json.loads(iwp_spec_path.read_text(encoding="utf-8"))
                iwp_by_name = {d["name"]: d for d in iwp_loaded["devices"]}
                self.assertIn("numa0-pcie-0000-00-4", iwp_by_name)
                self.assertIn("0-3", iwp_by_name)
                self.assertEqual(
                    iwp_by_name["numa0-pcie-0000-00-4"]["annotations"].get(
                        "freetpu.dev/chip-id-source"
                    ),
                    "chip-id-map",
                )
                self.assertEqual(
                    iwp_by_name["all"]["annotations"].get("freetpu.dev/chip-id-source"),
                    "chip-id-map",
                )
                rc_iwp_nomap, _o_iwpnm, err_iwpnm = _run_cli_inproc(
                    [
                        "validate",
                        str(iwp_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--sysfs-root",
                        str(iwp_sys),
                        "--dev-root",
                        str(iwp_dev),
                    ]
                )
                self.assertNotEqual(rc_iwp_nomap, 0)
                self.assertIn("Preserved live CDI spec", err_iwpnm)
                self.assertIn("--chip-id-map", err_iwpnm)
                self.assertTrue(
                    iwp_spec_path.exists(),
                    "must preserve healthy separate-slot Ironwood spec when pairing --chip-id-map is omitted",
                )
                rc_iwp_withmap, _o_iwpwm, err_iwpwm = _run_cli_inproc(
                    [
                        "validate",
                        str(iwp_spec_path),
                        "--check-host",
                        "--remove-if-stale",
                        "--chip-id-map",
                        str(iwp_map_file),
                        "--sysfs-root",
                        str(iwp_sys),
                        "--dev-root",
                        str(iwp_dev),
                    ]
                )
                self.assertEqual(rc_iwp_withmap, 0, err_iwpwm)
                self.assertTrue(iwp_spec_path.exists())
                sid_mapped_env = dict(
                    kv.split("=", 1) for kv in sid_mapped_all["containerEdits"]["env"]
                )
                self.assertEqual(sid_mapped_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(sid_mapped_env["TPU_ACCELERATOR_TYPE"], "tpu7x-8")
                self.assertEqual(sid_mapped_env["TPU_SKIP_MDS_QUERY"], "true")
                self.assertEqual(sid_mapped_env["TPU_VISIBLE_CHIPS"], "4,5,6,7")
                self.assertEqual(sid_mapped_env["TPU_VISIBLE_DEVICES"], "8,9,10,11,12,13,14,15")

                # Long (>260-byte) formatted inline JSON `--chip-id-map` MUST parse
                # directly without calling Path.is_file() (preventing ENAMETOOLONG on
                # 255-byte NAME_MAX filesystems).
                import errno as _errno

                long_inline_json = json.dumps(
                    {f"0000:00:{4 + i:02x}.0": i for i in range(12)}, indent=4
                )
                self.assertGreater(len(long_inline_json), 260)
                with mock.patch.object(
                    Path,
                    "is_file",
                    side_effect=OSError(_errno.ENAMETOOLONG, "File name too long"),
                ):
                    parsed_long_map = freetpu_cdi._load_chip_id_map(long_inline_json)
                self.assertEqual(len(parsed_long_map or {}), 12)
                self.assertEqual((parsed_long_map or {})["0000:00:0f.0"], 11)

                # Healthy multifunction Ironwood (no chip_id required): 04.0/04.1 + 05.0/05.1
                # on NUMA 0 and 80:00.0/00.1 + 80:01.0/01.1 on NUMA 1 pair cleanly!
                mf_ok_devs = [
                    {
                        "bdf": f"0000:00:0{4 + slot:x}.{fn}",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(slot * 2 + fn),
                        "numa_node": "0",
                    }
                    for slot in range(2)
                    for fn in range(2)
                ] + [
                    {
                        "bdf": f"0000:80:0{slot:x}.{fn}",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(4 + slot * 2 + fn),
                        "numa_node": "1",
                    }
                    for slot in range(2)
                    for fn in range(2)
                ]
                mfok_sys, mfok_dev = _make_mock_pci_vfio(
                    root / "tpu7x_multifunc_healthy",
                    devices=mf_ok_devs,
                    create_vfio_nodes=[str(i) for i in range(8)],
                )
                mfok_chips, mfok_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=mfok_sys, dev_root=mfok_dev
                )
                mfok_spec = freetpu_cdi.build_cdi_spec(
                    mfok_chips, mfok_ctrl, single_host_env=True
                )
                mfok_by_name = {d["name"]: d for d in mfok_spec["devices"]}
                self.assertNotIn("numa0-2", mfok_by_name)
                self.assertNotIn("numa1-2", mfok_by_name)
                mfok_all_env = dict(
                    kv.split("=", 1) for kv in mfok_by_name["all"]["containerEdits"]["env"]
                )
                self.assertEqual(mfok_all_env["TPU_CHIPS_PER_HOST_BOUNDS"], "2,2,1")
                self.assertEqual(mfok_all_env["TPU_ACCELERATOR_TYPE"], "tpu7x-8")

            # Verify PCIe root resolution from sysfs ancestry when endpoints sit behind
            # PCIe switches on distinct downstream bus numbers (without numa_node files).
            # When sysfs supplies NO numa_node, `freetpu.dev/numa-node` MUST be omitted
            # and slices use `pcie0-4`/`pcie1-4` + `freetpu.dev/pcie-domain` (never inventing NUMA IDs).
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
                self.assertTrue(
                    {"pcie-0000-00-4", "pcie-0000-80-4", "all"} <= set(sw_by_name),
                    sw_by_name.keys(),
                )
                self.assertNotIn("pcie0-4", sw_by_name)
                self.assertNotIn("pcie1-4", sw_by_name)
                self.assertNotIn("0-3", sw_by_name)
                self.assertNotIn("4-7", sw_by_name)
                self.assertNotIn("numa0-4", sw_by_name)
                self.assertNotIn("numa1-4", sw_by_name)
                self.assertNotIn("freetpu.dev/numa-node", sw_by_name["pcie-0000-00-4"]["annotations"])
                self.assertNotIn("freetpu.dev/numa-node", sw_by_name["pcie-0000-80-4"]["annotations"])
                self.assertEqual(
                    sw_by_name["pcie-0000-00-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:00",
                )
                self.assertEqual(
                    sw_by_name["pcie-0000-80-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:80",
                )

            # On a 3-PCIe-root host (`0000:00`, `0000:40`, `0000:80`, no numa_node),
            # losing root `0000:00` MUST withdraw `pcie-0000-00-4` while keeping
            # `pcie-0000-40-4` and `pcie-0000-80-4` bound to their exact roots
            # (NEVER shifting ordinals to retarget root 0's selector onto root 1!).
            tri_root = root / "three_pcie_roots"
            tri_devs = (
                [
                    {
                        "bdf": f"0000:00:0{i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                    }
                    for i in range(4)
                ]
                + [
                    {
                        "bdf": f"0000:40:0{i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": str(4 + i),
                    }
                    for i in range(4)
                ]
                + [
                    {
                        "bdf": f"0000:80:0{i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": str(8 + i),
                    }
                    for i in range(4)
                ]
            )
            tri_sys, tri_dev = _make_mock_pci_vfio(
                tri_root / "all3",
                devices=tri_devs,
                create_vfio_nodes=[str(i) for i in range(12)],
            )
            tri_lost0_sys, tri_lost0_dev = _make_mock_pci_vfio(
                tri_root / "lost_root0",
                devices=tri_devs[4:],  # drop entire 0000:00 root, keep 0000:40 and 0000:80
                create_vfio_nodes=[str(i) for i in range(4, 12)],
            )
            with _mock_vfio_char_devices():
                tri_chips, tri_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=tri_sys, dev_root=tri_dev
                )
                tri_spec = freetpu_cdi.build_cdi_spec(tri_chips, tri_ctrl)
                tri_by_name = {d["name"]: d for d in tri_spec["devices"]}
                self.assertTrue(
                    {"pcie-0000-00-4", "pcie-0000-40-4", "pcie-0000-80-4"} <= set(tri_by_name),
                    tri_by_name.keys(),
                )
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        tri_spec, sysfs_root=tri_sys, dev_root=tri_dev
                    ),
                    [],
                )

                tri_lost_chips, tri_lost_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=tri_lost0_sys, dev_root=tri_lost0_dev
                )
                tri_lost_spec = freetpu_cdi.build_cdi_spec(tri_lost_chips, tri_lost_ctrl)
                tri_lost_by_name = {d["name"]: d for d in tri_lost_spec["devices"]}
                self.assertNotIn("pcie-0000-00-4", tri_lost_by_name)
                self.assertIn("pcie-0000-40-4", tri_lost_by_name)
                self.assertIn("pcie-0000-80-4", tri_lost_by_name)
                self.assertEqual(
                    tri_lost_by_name["pcie-0000-40-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:40",
                )
                self.assertEqual(
                    tri_lost_by_name["pcie-0000-80-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:80",
                )
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        tri_lost_spec, sysfs_root=tri_lost0_sys, dev_root=tri_lost0_dev
                    ),
                    [],
                )
                errs_lost0 = freetpu_cdi.verify_spec_against_host(
                    tri_spec, sysfs_root=tri_lost0_sys, dev_root=tri_lost0_dev
                )
                self.assertTrue(
                    any("device 'pcie-0000-00-4'" in e for e in errs_lost0),
                    errs_lost0,
                )

            # When two PCIe roots SHARE Linux NUMA node 0 (4 v6e chips per root), BOTH slices
            # must keep `freetpu.dev/numa-node == "0"` and use root-bound `pcie-<root>-4` names
            # (never writing domain ordinal "1" as NUMA node 1).
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
                for i in range(4)
            ] + [
                {
                    "bdf": f"0000:80:0{i:x}.0",
                    "vendor": "0x1ae0",
                    "device": "0x006f",
                    "driver": "vfio-pci",
                    "iommu_group": str(4 + i),
                    "numa_node": "0",
                }
                for i in range(4)
            ]
            sn_sys, sn_dev = _make_mock_pci_vfio(
                shared_numa_root,
                devices=shared_numa_devs,
                create_vfio_nodes=[str(i) for i in range(8)],
            )
            with _mock_vfio_char_devices():
                sn_chips, sn_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=sn_sys, dev_root=sn_dev
                )
                sn_spec = freetpu_cdi.build_cdi_spec(sn_chips, sn_ctrl)
                sn_by_name = {d["name"]: d for d in sn_spec["devices"]}
                self.assertIn("numa0-pcie-0000-00-4", sn_by_name)
                self.assertIn("numa0-pcie-0000-80-4", sn_by_name)
                self.assertNotIn("pcie-0000-00-4", sn_by_name)
                self.assertNotIn("pcie-0000-80-4", sn_by_name)
                self.assertNotIn("numa0-4", sn_by_name)
                self.assertNotIn("numa1-4", sn_by_name)
                self.assertEqual(
                    sn_by_name["numa0-pcie-0000-00-4"]["annotations"]["freetpu.dev/numa-node"],
                    "0",
                )
                self.assertEqual(
                    sn_by_name["numa0-pcie-0000-80-4"]["annotations"]["freetpu.dev/numa-node"],
                    "0",
                )
                self.assertEqual(
                    sn_by_name["numa0-pcie-0000-00-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:00",
                )
                self.assertEqual(
                    sn_by_name["numa0-pcie-0000-80-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:80",
                )

                # If the FIRST root (`0000:00`) disappears on a shared-NUMA-0 host,
                # `sync` MUST retain `numa0-pcie-0000-80-4` and NEVER invent scan-dependent
                # `numa0-4` or `pcie-0000-80-4` aliases that could flip binding across outages!
                sn_lost0_sys, sn_lost0_dev = _make_mock_pci_vfio(
                    root / "shared_numa0_lost_root0",
                    devices=shared_numa_devs[4:],
                    create_vfio_nodes=[str(i) for i in range(4, 8)],
                )
                sn_lost0_out = root / "shared_numa0_lost_root0.json"
                ok_snl, msg_snl = freetpu_cdi.reconcile_cdi_spec(
                    sn_lost0_out,
                    sysfs_root=sn_lost0_sys,
                    dev_root=sn_lost0_dev,
                )
                self.assertTrue(ok_snl, msg_snl)
                snl_spec = json.loads(sn_lost0_out.read_text(encoding="utf-8"))
                snl_by_name = {d["name"]: d for d in snl_spec["devices"]}
                self.assertNotIn("numa0-pcie-0000-00-4", snl_by_name)
                self.assertIn("numa0-pcie-0000-80-4", snl_by_name)
                self.assertNotIn("pcie-0000-80-4", snl_by_name)
                self.assertNotIn("numa0-4", snl_by_name)
                self.assertEqual(
                    snl_by_name["numa0-pcie-0000-80-4"]["annotations"]["freetpu.dev/pcie-root"],
                    "0000:80",
                )

                # `--single-host-env --omit-visible-env` (for PyTorch/XLA `xmp.spawn()`)
                # MUST omit TPU_VISIBLE_CHIPS / TPU_VISIBLE_DEVICES even when verified
                # chip IDs are supplied, while retaining isolated networking and topology
                # bounds (`TPU_CHIPS_PER_HOST_BOUNDS`, `TPU_ACCELERATOR_TYPE`, etc.)!
                sn_chip_map = {shared_numa_devs[i]["bdf"]: i for i in range(8)}
                rc_ov, out_ov, err_ov = _run_cli_inproc(
                    [
                        "generate",
                        "--stdout",
                        "--validate",
                        "--single-host-env",
                        "--omit-visible-env",
                        "--chip-id-map",
                        json.dumps(sn_chip_map),
                        "--sysfs-root",
                        str(sn_sys),
                        "--dev-root",
                        str(sn_dev),
                    ]
                )
                self.assertEqual(rc_ov, 0, err_ov)
                ov_spec = json.loads(out_ov)
                ov_by_name = {d["name"]: d for d in ov_spec["devices"]}
                self.assertIn("0-3", ov_by_name)
                for sel in ("numa0-pcie-0000-00-4", "0-3", "all"):
                    sel_env = dict(
                        kv.split("=", 1) for kv in ov_by_name[sel]["containerEdits"]["env"]
                    )
                    self.assertNotIn("TPU_VISIBLE_CHIPS", sel_env)
                    self.assertNotIn("TPU_VISIBLE_DEVICES", sel_env)
                    self.assertIn("TPU_CHIPS_PER_HOST_BOUNDS", sel_env)
                    self.assertIn("TPU_ACCELERATOR_TYPE", sel_env)
                    self.assertEqual(sel_env["TPU_WORKER_HOSTNAMES"], "localhost")
                    self.assertEqual(sel_env["TPU_WORKER_ID"], "0")
                    self.assertEqual(sel_env["TPU_SKIP_MDS_QUERY"], "true")

                ov_sync_out = root / "omit_visible_sync.json"
                rc_ovs, _o_ovs, err_ovs = _run_cli_inproc(
                    [
                        "sync",
                        "--once",
                        "--output",
                        str(ov_sync_out),
                        "--single-host-env",
                        "--omit-visible-env",
                        "--chip-id-map",
                        json.dumps(sn_chip_map),
                        "--sysfs-root",
                        str(sn_sys),
                        "--dev-root",
                        str(sn_dev),
                    ]
                )
                self.assertEqual(rc_ovs, 0, err_ovs)
                ovs_spec = json.loads(ov_sync_out.read_text(encoding="utf-8"))
                ovs_by_name = {d["name"]: d for d in ovs_spec["devices"]}
                ovs_env = dict(
                    kv.split("=", 1)
                    for kv in ovs_by_name["numa0-pcie-0000-00-4"]["containerEdits"]["env"]
                )
                self.assertNotIn("TPU_VISIBLE_CHIPS", ovs_env)
                self.assertNotIn("TPU_VISIBLE_DEVICES", ovs_env)
                self.assertEqual(ovs_env["TPU_ACCELERATOR_TYPE"], "v6e-4")

                # When ONE PCIe root (`0000:00`) spans TWO NUMA domains (NUMA 0 with 4 chips
                # and NUMA 1 with 4 chips), root-only `pcie-0000-00-4` MUST NEVER be emitted
                # (neither when both NUMA domains are up NOR when NUMA 0 disappears, so
                # `pcie-0000-00-4` can NEVER flip from NUMA 1 to NUMA 0 across refreshes!).
                one_root_two_numa_devs = [
                    {
                        "bdf": f"0000:00:0{i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                        "numa_node": "0",
                    }
                    for i in range(4)
                ] + [
                    {
                        "bdf": f"0000:00:0{4 + i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": str(4 + i),
                        "numa_node": "1",
                    }
                    for i in range(4)
                ]
                ortn_sys, ortn_dev = _make_mock_pci_vfio(
                    root / "one_root_two_numa",
                    devices=one_root_two_numa_devs,
                    create_vfio_nodes=[str(i) for i in range(8)],
                )
                ortn_chips, ortn_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=ortn_sys, dev_root=ortn_dev
                )
                ortn_spec = freetpu_cdi.build_cdi_spec(ortn_chips, ortn_ctrl)
                ortn_by_name = {d["name"]: d for d in ortn_spec["devices"]}
                self.assertNotIn("pcie-0000-00-4", ortn_by_name)
                self.assertNotIn("numa0-4", ortn_by_name)
                self.assertNotIn("numa1-4", ortn_by_name)
                self.assertIn("numa0-pcie-0000-00-4", ortn_by_name)
                self.assertIn("numa1-pcie-0000-00-4", ortn_by_name)
                self.assertEqual(
                    freetpu_cdi.verify_spec_against_host(
                        ortn_spec, sysfs_root=ortn_sys, dev_root=ortn_dev
                    ),
                    [],
                )
                # Drop 1 chip from NUMA 0: NUMA 0 withdraws, NUMA 1 stays permanently bound to
                # `numa1-pcie-0000-00-4`, and `pcie-0000-00-4` is STILL not emitted!
                ortn_deg_sys, ortn_deg_dev = _make_mock_pci_vfio(
                    root / "one_root_two_numa_deg",
                    devices=one_root_two_numa_devs[1:],
                    create_vfio_nodes=[str(i) for i in range(1, 8)],
                )
                ortn_deg_chips, ortn_deg_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=ortn_deg_sys, dev_root=ortn_deg_dev
                )
                ortn_deg_spec = freetpu_cdi.build_cdi_spec(ortn_deg_chips, ortn_deg_ctrl)
                ortn_deg_by_name = {d["name"]: d for d in ortn_deg_spec["devices"]}
                self.assertNotIn("pcie-0000-00-4", ortn_deg_by_name)
                self.assertNotIn("numa0-pcie-0000-00-4", ortn_deg_by_name)
                self.assertIn("numa1-pcie-0000-00-4", ortn_deg_by_name)

                # Drop ALL 4 chips on NUMA 0: `numa1-pcie-0000-00-4` MUST STILL be
                # preserved, and root-only `pcie-0000-00-4` MUST STILL NOT be emitted
                # so it can never later flip to NUMA 0 when NUMA 0 returns and NUMA 1 drops!
                ortn_lost0_sys, ortn_lost0_dev = _make_mock_pci_vfio(
                    root / "one_root_two_numa_lost_numa0",
                    devices=one_root_two_numa_devs[4:],
                    create_vfio_nodes=[str(i) for i in range(4, 8)],
                )
                ortn_lost0_out = root / "one_root_two_numa_lost_numa0.json"
                ok_ol0, msg_ol0 = freetpu_cdi.reconcile_cdi_spec(
                    ortn_lost0_out, sysfs_root=ortn_lost0_sys, dev_root=ortn_lost0_dev
                )
                self.assertTrue(ok_ol0, msg_ol0)
                ol0_spec = json.loads(ortn_lost0_out.read_text(encoding="utf-8"))
                ol0_by_name = {d["name"]: d for d in ol0_spec["devices"]}
                self.assertNotIn("numa0-pcie-0000-00-4", ol0_by_name)
                self.assertIn("numa1-pcie-0000-00-4", ol0_by_name)
                self.assertNotIn("pcie-0000-00-4", ol0_by_name)
                self.assertNotIn("numa1-4", ol0_by_name)

                # When NUMA locality is absent (`numa_node` unset / -1), root-bound
                # `pcie-<root>-<chips>` IS published cleanly!
                no_numa_devs = [
                    {
                        "bdf": f"0000:00:0{i:x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                    }
                    for i in range(4)
                ]
                nn_sys, nn_dev = _make_mock_pci_vfio(
                    root / "no_numa_pcie_root",
                    devices=no_numa_devs,
                    create_vfio_nodes=[str(i) for i in range(4)],
                )
                nn_chips, nn_ctrl = freetpu_cdi.discover_tpu_chips(
                    sysfs_root=nn_sys, dev_root=nn_dev
                )
                nn_spec = freetpu_cdi.build_cdi_spec(nn_chips, nn_ctrl)
                nn_by_name = {d["name"]: d for d in nn_spec["devices"]}
                self.assertIn("pcie-0000-00-4", nn_by_name)
                self.assertNotIn("freetpu.dev/numa-node", nn_by_name["pcie-0000-00-4"]["annotations"])

            # Verify Kubernetes DRA example manifest uses stable resource.k8s.io/v1 API
            # and accurately qualifies Kubernetes 1.34 DRA Topology Manager limitation.
            dra_path = CDI_DIR / "manifests" / "k8s-tpu-dra-example.yaml"
            dra_text = dra_path.read_text(encoding="utf-8")
            self.assertNotIn("resource.k8s.io/v1beta1", dra_text)
            self.assertEqual(dra_text.count("apiVersion: resource.k8s.io/v1"), 3)
            self.assertNotIn("basic:", dra_text)
            self.assertIn("exactly:", dra_text)
            self.assertIn('"generation" in device.attributes["freetpu.dev"]', dra_text)
            self.assertIn('"numa_node" in device.attributes["freetpu.dev"]', dra_text)
            self.assertIn('"chips" in device.attributes["freetpu.dev"]', dra_text)
            self.assertIn('"memoryGiB" in device.attributes["freetpu.dev"]', dra_text)
            self.assertIn("hint provider", dra_text)
            self.assertIn("separate placement mechanism", dra_text)
            self.assertIn("numactl", dra_text)
            self.assertIn("Guaranteed", dra_text)
            self.assertIn('cpu: "4"', dra_text)
            self.assertIn('memory: "32Gi"', dra_text)
            self.assertIn("TPU_WORKER_HOSTNAMES=localhost", dra_text)
            self.assertIn("TPU_WORKER_ID=0", dra_text)
            self.assertIn("TPU_SKIP_MDS_QUERY=true", dra_text)
            self.assertIn("freetpu.dev/tpu=numa0-pcie-0000-00-4", dra_text)
            self.assertIn("freetpu.dev/tpu=numa1-pcie-0000-80-4", dra_text)
            self.assertIn("--omit-visible-env", dra_text)
            self.assertIn("PJRT_LOCAL_PROCESS_COUNT", dra_text)

            # Verify README keeps containerd GHSA-33vj-92qq-66hc security prerequisite
            # before the general Quickstart section and documents --chip-id-map /
            # explicit TPU_VISIBLE_CHIPS + TPU_VISIBLE_DEVICES for isolated slices.
            readme_text = (CDI_DIR / "README.md").read_text(encoding="utf-8")
            qs_idx = readme_text.index("## Quickstart")
            sec_idx = readme_text.index("GHSA-33vj-92qq-66hc")
            self.assertLess(sec_idx, qs_idx)
            self.assertNotIn("2.1.x+", readme_text)
            self.assertIn("--chip-id-map", readme_text)
            self.assertIn("--omit-visible-env", readme_text)
            self.assertIn("PJRT_LOCAL_PROCESS_COUNT", readme_text)
            self.assertIn("slice-aware launcher", readme_text)
            self.assertIn("-e TPU_VISIBLE_CHIPS=", readme_text)
            self.assertIn("-e TPU_VISIBLE_DEVICES=", readme_text)
            self.assertIn("TPU_ACCELERATOR_TYPE", readme_text)
            self.assertIn("freetpu.dev/tpu=numa0-pcie-0000-00-4", readme_text)
            self.assertIn("--chip-id-map", dra_text)
            self.assertIn("TPU_ACCELERATOR_TYPE", dra_text)


if __name__ == "__main__":
    unittest.main()
