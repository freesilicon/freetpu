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
    """Patch Path.stat/lstat so synthetic non-symlink `<dev_root>/vfio/*` files behave as Linux char devices."""
    orig_stat = Path.stat
    orig_lstat = Path.lstat

    def _maybe_char(p: Path, st):
        if p.parent.name == "vfio" and not stat.S_ISLNK(st.st_mode):
            fake = mock.MagicMock()
            fake.st_mode = stat.S_IFCHR | 0o600
            if p.name == "vfio":
                fake.st_rdev = os.makedev(10, 196)
            elif p.name.isdigit():
                fake.st_rdev = os.makedev(242, int(p.name))
            else:
                fake.st_rdev = os.makedev(242, 250)
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

        # intelRdt was introduced in CDI v0.7.0 and must be rejected in v0.6.0 specs.
        intel_rdt_spec = {
            "cdiVersion": "0.6.0",
            "kind": "freetpu.dev/tpu",
            "containerEdits": {
                "deviceNodes": [{"path": "/dev/vfio/vfio"}],
                "intelRdt": {"closID": "tpu"},
            },
            "devices": [
                {
                    "name": "0",
                    "containerEdits": {"deviceNodes": [{"path": "/dev/vfio/0"}]},
                }
            ],
        }
        errs4 = freetpu_cdi.validate_cdi_spec(intel_rdt_spec)
        self.assertTrue(any("intelRdt" in e for e in errs4))

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
        self.assertIn("--cap-add=IPC_LOCK", res.stdout)
        self.assertIn("--ulimit memlock=-1:-1", res.stdout)

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

                # Invalid CDI kinds rejected by containerd's CDI parser (digit-leading vendor/class, trailing punctuation).
                invalid_kinds = [
                    "1vendor/tpu",
                    "freetpu.dev/1tpu",
                    "0example.com/0tpu",
                    "freetpu.dev-/tpu",
                    "freetpu.dev/tpu_",
                    ".freetpu.dev/tpu",
                    "freetpu.dev/.tpu",
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
                for good_kind in ["freetpu.dev/tpu", "Example_Org.io/TPU-v6e", "a/b", "v1.org/tpu0"]:
                    spec = freetpu_cdi.build_cdi_spec(chips, vfio_container, kind=good_kind)
                    self.assertEqual(freetpu_cdi.validate_cdi_spec(spec), [])

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


if __name__ == "__main__":
    unittest.main()
