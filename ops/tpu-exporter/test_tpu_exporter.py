#!/usr/bin/env python3
"""Deterministic offline self-tests for ops/tpu-exporter/tpu_exporter.py."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.request

EXPORTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORTER_DIR))

import tpu_exporter  # noqa: E402


@contextlib.contextmanager
def _mock_vfio_char_lstat():
    """Patch os.lstat so synthetic non-symlink `<dev_root>/vfio/*` files behave as char devices."""
    orig_lstat = os.lstat

    def _char_lstat(path_arg, *args, **kwargs):
        st = orig_lstat(path_arg, *args, **kwargs)
        p = Path(os.fspath(path_arg))
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

    with mock.patch("tpu_exporter.os.lstat", side_effect=_char_lstat):
        yield


def _make_mock_pci_tree(
    root: Path,
    devices: list[dict[str, str]],
    *,
    create_vfio_container: bool = True,
    create_vfio_nodes: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    sysfs_root = root / "sys"
    dev_root = root / "dev"
    proc_root = root / "proc"

    pci_devices = sysfs_root / "bus" / "pci" / "devices"
    pci_devices.mkdir(parents=True, exist_ok=True)
    iommu_groups = sysfs_root / "kernel" / "iommu_groups"
    iommu_groups.mkdir(parents=True, exist_ok=True)
    drivers_vfio = sysfs_root / "bus" / "pci" / "drivers" / "vfio-pci"
    drivers_vfio.mkdir(parents=True, exist_ok=True)

    vfio_dir = dev_root / "vfio"
    vfio_dir.mkdir(parents=True, exist_ok=True)
    if create_vfio_container:
        (vfio_dir / "vfio").write_text("", encoding="utf-8")
        misc_dev = sysfs_root / "class" / "misc" / "vfio" / "dev"
        misc_dev.parent.mkdir(parents=True, exist_ok=True)
        misc_dev.write_text("10:196\n", encoding="utf-8")

    for node in create_vfio_nodes or []:
        (vfio_dir / node).write_text("", encoding="utf-8")
        minor = int(node) if node.isdigit() else 250
        grp_dev = sysfs_root / "class" / "vfio" / node / "dev"
        grp_dev.parent.mkdir(parents=True, exist_ok=True)
        grp_dev.write_text(f"242:{minor}\n", encoding="utf-8")

    proc_root.mkdir(parents=True, exist_ok=True)

    for dev in devices:
        bdf = dev["bdf"]
        d = pci_devices / bdf
        d.mkdir(parents=True, exist_ok=True)
        (d / "vendor").write_text(dev["vendor"] + "\n", encoding="utf-8")
        (d / "device").write_text(dev["device"] + "\n", encoding="utf-8")
        if "subsystem_device" in dev:
            (d / "subsystem_device").write_text(dev["subsystem_device"] + "\n", encoding="utf-8")
        (d / "enable").write_text(dev.get("enable", "1") + "\n", encoding="utf-8")
        if "numa_node" in dev:
            (d / "numa_node").write_text(dev["numa_node"] + "\n", encoding="utf-8")
        if "chip_id" in dev:
            (d / "chip_id").write_text(dev["chip_id"] + "\n", encoding="utf-8")
        if dev.get("iommu_group"):
            grp = iommu_groups / dev["iommu_group"]
            grp_devs = grp / "devices"
            grp_devs.mkdir(parents=True, exist_ok=True)
            (grp_devs / bdf).write_text("", encoding="utf-8")
            os.symlink(str(grp), d / "iommu_group")
        if dev.get("driver"):
            drv_dir = sysfs_root / "bus" / "pci" / "drivers" / dev["driver"]
            drv_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(str(drv_dir), d / "driver")

    return sysfs_root, dev_root, proc_root


class TpuExporterTests(unittest.TestCase):
    def test_parse_tpu_info_unicode_and_ascii_tables(self) -> None:
        sample = """
TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ /dev/vfio/0 │ TPU v5e chip │ 1       │ 4242 │
│ /dev/vfio/1 │ TPU v6e chip │ 1       │ None │
└─────────────┴──────────────┴─────────┴──────┘

TPU Runtime Utilization
| Chip | HBM Usage (GiB)       | Duty cycle |
|------|-----------------------|------------|
| 0    | 8.00 GiB / 15.75 GiB  | 87.50%     |
| 1    | 16.00 GiB / 31.25 GiB | 0.00%      |

TensorCore Utilization
| Chip | TensorCore Utilization |
|------|------------------------|
| 0    | 99.00%                 |
| 1    | 75.50%                 |
"""
        rows = tpu_exporter.parse_tpu_info_output(sample)
        self.assertIn("/dev/vfio/0", rows)
        self.assertEqual(rows["/dev/vfio/0"].pids, [4242])
        self.assertEqual(rows["/dev/vfio/1"].pids, [])
        self.assertEqual(rows["0"].hbm_used_bytes, 8 * 1024**3)
        self.assertEqual(rows["0"].hbm_total_bytes, int(round(15.75 * 1024**3)))
        # Must NOT be overwritten by TensorCore Utilization percentages below it.
        self.assertAlmostEqual(rows["0"].duty_cycle_percent or 0.0, 87.5)
        self.assertEqual(rows["1"].hbm_used_bytes, 16 * 1024**3)
        self.assertAlmostEqual(rows["1"].duty_cycle_percent or 0.0, 0.0)

    def test_parse_tpu_info_dashboard_device_status_and_process_monitor(self) -> None:
        # Upstream tpu-info uses literal `|` characters inside `[|||||.....]` gauges
        # even inside ASCII `|` tables (tpu_info/cli_helper.py:ascii_bar_gauge).
        dashboard = """
\x1b[1mTPU Device Status\x1b[0m
+-----------+---------------------------+--------------------+--------------------+-----------------------+
| Device ID | Memory Usage (Used/Total) | Memory Bar Gauge   | HBM Bandwidth Util | TensorCore Duty Cycle |
+-----------+---------------------------+--------------------+--------------------+-----------------------+
| 0         | 8.00 GiB / 15.75 GiB      | [|||||.....] 50.8% | [||........] 20.0% | [|||||||||.] 87.5%    |
| 1         | N/A                       | [..........] N/A   | [..........] N/A   | [..........] N/A      |
| 2         | 15.75 GiB / 15.75 GiB     | [||||||||||] 100.0%| [||||||....] 60.0% | [||||||||||] 100.0%   |
+-----------+---------------------------+--------------------+--------------------+-----------------------+

TPU Process Monitor
+-----------+-------------------------+------+--------------+-------------------+
| Device ID | Device Path / VFIO Path | PID  | Process Name | Memory Allocation |
+-----------+-------------------------+------+--------------+-------------------+
| 0         | /dev/vfio/0             | 4242 | python3      | 8.00 GiB          |
| 1         | /dev/vfio/2             | None | N/A          | N/A               |
+-----------+-------------------------+------+--------------+-------------------+
"""
        rows = tpu_exporter.parse_tpu_info_output(dashboard)
        self.assertIn("0", rows)
        self.assertTrue(rows["0"].in_util_table)
        self.assertEqual(rows["0"].hbm_used_bytes, 8 * 1024**3)
        self.assertEqual(rows["0"].hbm_total_bytes, int(round(15.75 * 1024**3)))
        # Must read TensorCore Duty Cycle (87.5%), not Memory Bar Gauge (50.8%) or HBM BW (20.0%).
        self.assertAlmostEqual(rows["0"].duty_cycle_percent or 0.0, 87.5)

        # Fully-filled 10-pipe `[||||||||||]` gauges must also parse cleanly.
        self.assertIn("2", rows)
        self.assertAlmostEqual(rows["2"].duty_cycle_percent or 0.0, 100.0)

        # N/A rows must stay unset (None), never fabricated as 0.
        self.assertIn("1", rows)
        self.assertIsNone(rows["1"].hbm_used_bytes)
        self.assertIsNone(rows["1"].hbm_total_bytes)
        self.assertIsNone(rows["1"].duty_cycle_percent)

        self.assertIn("/dev/vfio/0", rows)
        self.assertTrue(rows["/dev/vfio/0"].in_chips_table)
        self.assertEqual(rows["/dev/vfio/0"].pids, [4242])
        self.assertIn("/dev/vfio/2", rows)
        self.assertEqual(rows["/dev/vfio/2"].pids, [])

        # Also verify Unicode box-drawing borders with literal `|` gauges inside cells.
        unicode_dash = """
TPU Device Status
│ Device ID │ Memory Usage (Used/Total) │ Memory Bar Gauge   │ HBM Bandwidth Util │ TensorCore Duty Cycle │
│ 0         │ 4.00 GiB / 16.00 GiB      │ [|||.......] 25.0% │ [|.........] 10.0% │ [||||||....] 62.5%    │
"""
        u_rows = tpu_exporter.parse_tpu_info_output(unicode_dash)
        self.assertEqual(u_rows["0"].hbm_used_bytes, 4 * 1024**3)
        self.assertAlmostEqual(u_rows["0"].duty_cycle_percent or 0.0, 62.5)

    def test_unparseable_and_na_tpu_info_leaves_utilization_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "enable": "1",
                    }
                ],
                create_vfio_nodes=["0"],
            )

            # Case 1: tpu-info exits 0 with unrecognized format (must NOT fabricate 0s).
            unknown_cli = root / "mock-unknown-tpu-info.sh"
            unknown_cli.write_text(
                "#!/bin/sh\necho 'Some Future Unrecognized Dashboard Format'\n",
                encoding="utf-8",
            )
            unknown_cli.chmod(0o755)

            snap_unknown = tpu_exporter.collect_snapshot(
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=str(unknown_cli),
                chip_id_map={"0000:00:06.0": "0"},
            )
            self.assertFalse(snap_unknown.tpu_info_available)
            self.assertIsNone(snap_unknown.devices[0].hbm_used_bytes)
            self.assertIsNone(snap_unknown.devices[0].duty_cycle_percent)
            prom_unknown = tpu_exporter.format_prometheus_metrics(snap_unknown)
            self.assertNotIn("freetpu_tpu_hbm_used_bytes{", prom_unknown)
            self.assertNotIn("freetpu_tpu_duty_cycle_percent{", prom_unknown)

            # Case 2: Dashboard outputs TPU Device Status with N/A readings (must stay unset).
            na_cli = root / "mock-na-tpu-info.sh"
            na_cli.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Device Status\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| Device ID | Memory Usage (Used/Total) | Memory Bar Gauge   | HBM Bandwidth Util | TensorCore Duty Cycle |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| 0         | N/A                       | [..........] N/A   | [..........] N/A   | [..........] N/A      |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "EOF\n",
                encoding="utf-8",
            )
            na_cli.chmod(0o755)

            snap_na = tpu_exporter.collect_snapshot(
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=str(na_cli),
                chip_id_map={"0000:00:06.0": "0"},
            )
            self.assertTrue(snap_na.tpu_info_available)
            self.assertIsNone(snap_na.devices[0].hbm_used_bytes)
            self.assertIsNone(snap_na.devices[0].duty_cycle_percent)
            self.assertEqual(snap_na.runtime_utilization, [])
            prom_na = tpu_exporter.format_prometheus_metrics(snap_na)
            self.assertNotIn("freetpu_tpu_hbm_used_bytes{", prom_na)
            self.assertNotIn("freetpu_tpu_duty_cycle_percent{", prom_na)

    def test_discovery_filters_gvnic_and_avoids_unverified_runtime_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:04.0",
                        "vendor": "0x1af4",
                        "device": "0x1041",
                        "driver": "virtio-pci",
                        "iommu_group": "1",
                    },
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0042",  # Google gVNIC NIC must be skipped!
                        "driver": "gve",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "enable": "1",
                        "numa_node": "0",
                    },
                    {
                        "bdf": "0000:00:07.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "2",
                        "enable": "0",  # idle vfio-pci TPU must still be healthy!
                        "numa_node": "0",
                    },
                ],
                create_vfio_nodes=["0", "2"],
            )

            # Synthetic /proc/9001/fd/3 and /proc/7777/fd/3 -> <dev_root>/vfio/0 (char device 242:0)
            # with real open flags (O_RDWR|O_CLOEXEC = 02000002, non-O_PATH).
            for pid_str in ("7777", "9001"):
                fd_dir = proc_root / pid_str / "fd"
                fdinfo_dir = proc_root / pid_str / "fdinfo"
                fd_dir.mkdir(parents=True, exist_ok=True)
                fdinfo_dir.mkdir(parents=True, exist_ok=True)
                os.symlink(str(dev_root / "vfio" / "0"), fd_dir / "3")
                (fdinfo_dir / "3").write_text("pos:\t0\nflags:\t02000002\n", encoding="utf-8")

            # Google v6e-style output: VFIO groups 0 & 2, runtime IDs 8 & 12 (unmapped)
            # plus TensorCore table that must not leak into duty cycle.
            mock_tpu_info = root / "mock-tpu-info.sh"
            mock_tpu_info.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Chips\n"
                "| Chip        | Type         | Devices | PID  |\n"
                "|-------------|--------------|---------|------|\n"
                "| /dev/vfio/0 | TPU v5e chip | 1       | 7777 |\n"
                "| /dev/vfio/2 | TPU v6e chip | 1       | None |\n"
                "TPU Runtime Utilization\n"
                "| Chip | HBM Usage (GiB)       | Duty cycle |\n"
                "|------|-----------------------|------------|\n"
                "| 8    | 4.00 GiB / 15.75 GiB  | 64.25%     |\n"
                "| 12   | 0.00 GiB / 31.25 GiB  | 0.00%      |\n"
                "TensorCore Utilization\n"
                "| Chip | TensorCore Utilization |\n"
                "|------|------------------------|\n"
                "| 8    | 99.00%                 |\n"
                "| 12   | 42.00%                 |\n"
                "EOF\n",
                encoding="utf-8",
            )
            mock_tpu_info.chmod(0o755)

            import stat as _stat
            from unittest import mock

            orig_os_stat = os.stat
            orig_os_lstat = os.lstat
            vfio0_st = orig_os_stat(dev_root / "vfio" / "0")
            vfio2_st = orig_os_stat(dev_root / "vfio" / "2")
            vfioc_st = orig_os_stat(dev_root / "vfio" / "vfio")
            vfio0_id = (vfio0_st.st_dev, vfio0_st.st_ino)
            vfio2_id = (vfio2_st.st_dev, vfio2_st.st_ino)
            vfioc_id = (vfioc_st.st_dev, vfioc_st.st_ino)

            def _maybe_char(st):
                if (st.st_dev, st.st_ino) == vfio0_id:
                    fake = mock.MagicMock()
                    fake.st_mode = _stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(242, 0)
                    return fake
                if (st.st_dev, st.st_ino) == vfio2_id:
                    fake = mock.MagicMock()
                    fake.st_mode = _stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(242, 2)
                    return fake
                if (st.st_dev, st.st_ino) == vfioc_id:
                    fake = mock.MagicMock()
                    fake.st_mode = _stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(10, 196)
                    return fake
                return st

            def _char_stat(path_arg, *args, **kwargs):
                return _maybe_char(orig_os_stat(path_arg, *args, **kwargs))

            def _char_lstat(path_arg, *args, **kwargs):
                st = orig_os_lstat(path_arg, *args, **kwargs)
                return st if _stat.S_ISLNK(st.st_mode) else _maybe_char(st)

            # Without verified mapping: runtime IDs 8/12 must NOT be attached or fabricated as 0.
            with (
                mock.patch("tpu_exporter.os.stat", side_effect=_char_stat),
                mock.patch("tpu_exporter.os.lstat", side_effect=_char_lstat),
            ):
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=str(mock_tpu_info),
                )
            self.assertTrue(snap.tpu_info_available)
            self.assertEqual(len(snap.devices), 2)

            v5e = snap.devices[0]
            self.assertEqual(v5e.pci_bdf, "0000:00:06.0")
            self.assertEqual(v5e.generation, "v5e")
            self.assertTrue(v5e.healthy)
            self.assertTrue(v5e.pci_enabled)
            self.assertEqual(v5e.active_pids, [7777, 9001])
            self.assertIsNone(v5e.hbm_used_bytes)
            self.assertIsNone(v5e.duty_cycle_percent)

            v6e = snap.devices[1]
            self.assertEqual(v6e.pci_bdf, "0000:00:07.0")
            self.assertEqual(v6e.generation, "v6e")
            self.assertTrue(v6e.healthy)
            self.assertFalse(v6e.pci_enabled)
            self.assertEqual(v6e.active_pids, [])
            self.assertIsNone(v6e.hbm_used_bytes)
            self.assertIsNone(v6e.duty_cycle_percent)

            prom = tpu_exporter.format_prometheus_metrics(snap)
            self._assert_valid_prometheus_text(prom)
            self.assertIn("freetpu_tpu_devices_discovered 2", prom)
            self.assertIn('runtime_chip="8",pci_bdf="unmapped"} 64.25', prom)
            self.assertNotIn('runtime_chip="8",pci_bdf="unmapped"} 99', prom)

            # With verified chip_id_map, runtime readings attach deterministically.
            snap_mapped = tpu_exporter.collect_snapshot(
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=str(mock_tpu_info),
                chip_id_map={"0000:00:06.0": "8", "0000:00:07.0": "12"},
            )
            self.assertEqual(snap_mapped.devices[0].hbm_used_bytes, 4 * 1024**3)
            self.assertAlmostEqual(snap_mapped.devices[0].duty_cycle_percent or 0.0, 64.25)
            self.assertAlmostEqual(snap_mapped.devices[1].duty_cycle_percent or 0.0, 0.0)

    def test_sysfs_fallback_and_unhealthy_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x005e",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "enable": "1",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0062",
                        "driver": "unbound",
                        "iommu_group": "1",
                        "enable": "0",
                    },
                ],
                create_vfio_nodes=["0"],  # group 1 node intentionally absent
            )

            with _mock_vfio_char_lstat():
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=str(root / "nonexistent-tpu-info"),
                )
            self.assertFalse(snap.tpu_info_available)
            self.assertEqual(len(snap.devices), 2)
            self.assertTrue(snap.devices[0].healthy)
            self.assertEqual(snap.devices[0].generation, "v4")
            self.assertEqual(snap.devices[0].hbm_total_bytes, 32 * 1024**3)
            self.assertFalse(snap.devices[1].healthy)
            self.assertEqual(snap.devices[1].generation, "v5p")

    def test_cli_once_json_and_http_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
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
                create_vfio_nodes=["0"],
            )

            script = EXPORTER_DIR / "tpu_exporter.py"
            res_once = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--once",
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                    "--proc-root",
                    str(proc_root),
                    "--tpu-info-bin",
                    "",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self._assert_valid_prometheus_text(res_once.stdout)
            self.assertIn("freetpu_tpu_devices_discovered 1", res_once.stdout)

            res_json = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--once",
                    "--json",
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                    "--proc-root",
                    str(proc_root),
                    "--tpu-info-bin",
                    "",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(res_json.stdout)
            self.assertEqual(len(payload["devices"]), 1)
            self.assertEqual(payload["devices"][0]["device_id"], "0x0063")

            server = tpu_exporter._ExporterServer(
                ("127.0.0.1", 0),
                tpu_exporter._ExporterHandler,
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=None,
                tpu_info_timeout_s=2.0,
                max_workers=2,
            )
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as resp:
                    self.assertEqual(resp.status, 200)
                    body = resp.read().decode("utf-8")
                    self._assert_valid_prometheus_text(body)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz") as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(json.loads(resp.read().decode("utf-8"))["status"], "ok")
            finally:
                server.shutdown()
                server.server_close()
                t.join(timeout=2.0)

    def test_unreadable_proc_fd_marks_pid_telemetry_unavailable(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "enable": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            (proc_root / "1234" / "fd").mkdir(parents=True)

            orig_scandir = os.scandir

            def _guarded_scandir(p):
                pp = Path(os.fspath(p))
                if pp.name == "fd" and pp.parent.name == "1234":
                    raise PermissionError("simulated EACCES without CAP_SYS_PTRACE")
                return orig_scandir(p)

            with (
                _mock_vfio_char_lstat(),
                mock.patch("tpu_exporter.os.scandir", side_effect=_guarded_scandir),
            ):
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                )

            self.assertFalse(snap.pid_telemetry_available)
            self.assertFalse(snap.devices[0].pid_telemetry_available)
            self.assertTrue(snap.devices[0].healthy)  # idle enable=0 still healthy
            prom = tpu_exporter.format_prometheus_metrics(snap)
            self.assertIn("freetpu_tpu_pid_telemetry_available{", prom)
            self.assertIn("} 0", prom)
            self.assertNotIn("freetpu_tpu_active_pids{", prom)

            # Bounded FD scan timeout marks PID telemetry unavailable without throwing,
            # even when a single `os.stat` blocks on an unresponsive filesystem (e.g. hung NFS).
            release_blocked = threading.Event()
            (proc_root / "2345" / "fd").mkdir(parents=True)
            os.symlink(str(root / "hung-nfs-file"), proc_root / "2345" / "fd" / "3")

            orig_stat_fn = os.stat

            def _hung_nfs_stat(path_arg, *args, **kwargs):
                if "2345" in os.fspath(path_arg):
                    release_blocked.wait(timeout=5.0)
                return orig_stat_fn(path_arg, *args, **kwargs)

            try:
                with (
                    _mock_vfio_char_lstat(),
                    mock.patch("tpu_exporter.os.stat", side_effect=_hung_nfs_stat),
                ):
                    _, ok_to, unrd_to, _ = tpu_exporter._scan_proc_vfio_pids(
                        proc_root,
                        [str(dev_root / "vfio" / "0")],
                        expected_devnums={str(dev_root / "vfio" / "0"): "242:0"},
                        scan_timeout_s=0.1,
                    )
                self.assertFalse(ok_to)
                self.assertGreaterEqual(unrd_to, 1)
            finally:
                release_blocked.set()

            # Even when tpu-info returns a matching chip row with PID=None, an
            # unreadable /proc scan MUST keep PID telemetry unavailable (no false 0).
            mock_none_pid = root / "mock-tpu-info-none-pid.sh"
            mock_none_pid.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Chips\n"
                "| Chip        | Type         | Devices | PID  |\n"
                "|-------------|--------------|---------|------|\n"
                "| /dev/vfio/0 | TPU v5e chip | 1       | None |\n"
                "EOF\n",
                encoding="utf-8",
            )
            mock_none_pid.chmod(0o755)
            with (
                _mock_vfio_char_lstat(),
                mock.patch("tpu_exporter.os.scandir", side_effect=_guarded_scandir),
            ):
                snap_with_cli = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=str(mock_none_pid),
                )
            self.assertTrue(snap_with_cli.tpu_info_available)
            self.assertFalse(snap_with_cli.pid_telemetry_available)
            self.assertFalse(snap_with_cli.devices[0].pid_telemetry_available)
            prom_with_cli = tpu_exporter.format_prometheus_metrics(snap_with_cli)
            self.assertNotIn("freetpu_tpu_active_pids{", prom_with_cli)

    def test_http_worker_cap_rejects_excess_scrapes_while_keeping_healthz_alive(self) -> None:
        import http.client
        import time as _time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "enable": "0",
                    }
                ],
                create_vfio_nodes=["0"],
            )
            server = tpu_exporter._ExporterServer(
                ("127.0.0.1", 0),
                tpu_exporter._ExporterHandler,
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=None,
                tpu_info_timeout_s=2.0,
                max_workers=1,
            )
            health_server = tpu_exporter._HealthServer(("127.0.0.1", 0))
            gate = threading.Event()
            release = threading.Event()
            orig_scrape = server.scrape

            def _slow_scrape():
                gate.set()
                release.wait(timeout=2.0)
                return orig_scrape()

            server.scrape = _slow_scrape  # type: ignore[method-assign]
            port = server.server_address[1]
            health_port = health_server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            th = threading.Thread(target=health_server.serve_forever, daemon=True)
            t.start()
            th.start()
            try:
                holder_status: list[int] = []

                def _hold() -> None:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
                    conn.request("GET", "/metrics")
                    holder_status.append(conn.getresponse().status)
                    conn.close()

                t1 = threading.Thread(target=_hold, daemon=True)
                t1.start()
                self.assertTrue(gate.wait(timeout=2.0))
                _time.sleep(0.05)

                # Excess concurrent /metrics scrape must be rejected with 503.
                conn_metrics = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn_metrics.request("GET", "/metrics")
                resp_metrics = conn_metrics.getresponse()
                self.assertEqual(resp_metrics.status, 503)
                conn_metrics.close()

                # /healthz on the main port stays 200 even when scrape slots are saturated.
                conn_hz = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn_hz.request("GET", "/healthz")
                resp_hz = conn_hz.getresponse()
                self.assertEqual(resp_hz.status, 200)
                conn_hz.close()

                # Dedicated health listener stays 200 independently of scrape traffic.
                conn_dedicated = http.client.HTTPConnection("127.0.0.1", health_port, timeout=2.0)
                conn_dedicated.request("GET", "/healthz")
                resp_dedicated = conn_dedicated.getresponse()
                self.assertEqual(resp_dedicated.status, 200)
                conn_dedicated.close()

                release.set()
                t1.join(timeout=2.0)
                self.assertEqual(holder_status, [200])
            finally:
                release.set()
                server.shutdown()
                server.server_close()
                health_server.shutdown()
                health_server.server_close()
                t.join(timeout=2.0)
                th.join(timeout=2.0)

    def test_shared_vfio_group_requires_pci_bdf_chip_id_map_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "12",
                        "enable": "1",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "12",  # shares /dev/vfio/12 with 0000:00:05.0
                        "enable": "1",
                    },
                ],
                create_vfio_nodes=["12"],
            )

            mock_tpu_info = root / "mock-tpu-info-shared.sh"
            mock_tpu_info.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Device Status\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| Device ID | Memory Usage (Used/Total) | Memory Bar Gauge   | HBM Bandwidth Util | TensorCore Duty Cycle |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| 0         | 10.00 GiB / 31.25 GiB     | [|||.......] 32.0% | [||........] 20.0% | [||||||....] 60.0%    |\n"
                "| 1         | 20.00 GiB / 31.25 GiB     | [||||||....] 64.0% | [||||......] 40.0% | [||||||||..] 80.0%    |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "EOF\n",
                encoding="utf-8",
            )
            mock_tpu_info.chmod(0o755)

            # Ambiguous VFIO-path mapping for shared /dev/vfio/12 must be rejected.
            with self.assertRaisesRegex(ValueError, "shared VFIO group"):
                tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=str(mock_tpu_info),
                    chip_id_map={"/dev/vfio/12": "0"},
                )

            # CLI must also fail cleanly (non-zero exit) on shared VFIO-path map.
            bad_map = root / "bad-map.json"
            bad_map.write_text(json.dumps({"/dev/vfio/12": "0"}), encoding="utf-8")
            script = EXPORTER_DIR / "tpu_exporter.py"
            cli_res = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--once",
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                    "--proc-root",
                    str(proc_root),
                    "--tpu-info-bin",
                    str(mock_tpu_info),
                    "--chip-id-map",
                    str(bad_map),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(cli_res.returncode, 0)
            self.assertIn("shared VFIO group", cli_res.stderr)

            # Explicit PCI BDF keys map each chip unambiguously without duplication.
            snap = tpu_exporter.collect_snapshot(
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=str(mock_tpu_info),
                chip_id_map={"0000:00:05.0": "0", "0000:00:06.0": "1"},
            )
            self.assertEqual(snap.devices[0].pci_bdf, "0000:00:05.0")
            self.assertEqual(snap.devices[0].hbm_used_bytes, 10 * 1024**3)
            self.assertAlmostEqual(snap.devices[0].duty_cycle_percent or 0.0, 60.0)
            self.assertEqual(snap.devices[1].pci_bdf, "0000:00:06.0")
            self.assertEqual(snap.devices[1].hbm_used_bytes, 20 * 1024**3)
            self.assertAlmostEqual(snap.devices[1].duty_cycle_percent or 0.0, 80.0)
            rt_map = {r.runtime_chip: r.mapped_pci_bdf for r in snap.runtime_utilization}
            self.assertEqual(rt_map, {"0": "0000:00:05.0", "1": "0000:00:06.0"})

    def test_legacy_0x0027_groups_dual_functions_and_resolves_v2_v3_subsystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    # One TPU v2 chip (subsystem 0x004e) exposing two PCI core functions (.0 and .1).
                    {
                        "bdf": "0000:00:04.0",
                        "vendor": "0x1ae0",
                        "device": "0x0027",
                        "subsystem_device": "0x004e",
                        "driver": "accel",
                        "enable": "1",
                    },
                    {
                        "bdf": "0000:00:04.1",
                        "vendor": "0x1ae0",
                        "device": "0x0027",
                        "subsystem_device": "0x004e",
                        "driver": "accel",
                        "enable": "1",
                    },
                    # One TPU v3 chip (subsystem 0x004f) exposing two PCI core functions (.0 and .1).
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0027",
                        "subsystem_device": "0x004f",
                        "driver": "accel",
                        "enable": "0",
                    },
                    {
                        "bdf": "0000:00:05.1",
                        "vendor": "0x1ae0",
                        "device": "0x0027",
                        "subsystem_device": "0x004f",
                        "driver": "accel",
                        "enable": "1",
                    },
                ],
                create_vfio_container=False,
                create_vfio_nodes=[],
            )
            with _mock_vfio_char_lstat():
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                )
            # 4 PCI functions -> exactly 2 physical chips (1 v2 + 1 v3).
            self.assertEqual(len(snap.devices), 2)

            v2 = snap.devices[0]
            self.assertEqual(v2.pci_bdf, "0000:00:04.0")
            self.assertEqual(v2.generation, "v2")
            self.assertEqual(v2.model_name, "TPU v2")
            self.assertEqual(v2.hbm_total_bytes, 16 * 1024**3)
            self.assertFalse(v2.healthy)

            v3 = snap.devices[1]
            self.assertEqual(v3.pci_bdf, "0000:00:05.0")
            self.assertEqual(v3.generation, "v3")
            self.assertEqual(v3.model_name, "TPU v3")
            self.assertEqual(v3.hbm_total_bytes, 32 * 1024**3)
            self.assertTrue(v3.pci_enabled)
            self.assertFalse(v3.healthy)

    def test_proc_vfio_pid_matches_by_char_dev_rdev_not_pathname(self) -> None:
        import stat as _stat
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "12",
                        "enable": "1",
                    }
                ],
                create_vfio_nodes=["12"],
            )
            host_vfio12 = dev_root / "vfio" / "12"

            # PID 5001: regular file at a decoy /dev/vfio/12 path (different mount ns) -> MUST NOT match.
            decoy_reg = root / "container_ns" / "dev" / "vfio" / "12"
            decoy_reg.parent.mkdir(parents=True, exist_ok=True)
            decoy_reg.write_text("not a vfio device", encoding="utf-8")
            (proc_root / "5001" / "fd").mkdir(parents=True, exist_ok=True)
            (proc_root / "5001" / "fdinfo").mkdir(parents=True, exist_ok=True)
            os.symlink(str(decoy_reg), proc_root / "5001" / "fd" / "3")
            (proc_root / "5001" / "fdinfo" / "3").write_text("pos:\t0\nflags:\t02000002\n", encoding="utf-8")

            # PID 5002: bind-mounted char device under a completely different path (/custom/tpu0)
            # with the SAME st_rdev (242:12) as host /dev/vfio/12 and real open flags -> MUST match!
            bind_mounted = root / "custom" / "tpu0"
            bind_mounted.parent.mkdir(parents=True, exist_ok=True)
            bind_mounted.write_text("", encoding="utf-8")
            (proc_root / "5002" / "fd").mkdir(parents=True, exist_ok=True)
            (proc_root / "5002" / "fdinfo").mkdir(parents=True, exist_ok=True)
            os.symlink(str(bind_mounted), proc_root / "5002" / "fd" / "4")
            (proc_root / "5002" / "fdinfo" / "4").write_text("pos:\t0\nflags:\t02000002\n", encoding="utf-8")

            # PID 5003: different char device (242:99) -> MUST NOT match.
            other_char = root / "dev" / "vfio" / "99"
            other_char.write_text("", encoding="utf-8")
            (proc_root / "5003" / "fd").mkdir(parents=True, exist_ok=True)
            (proc_root / "5003" / "fdinfo").mkdir(parents=True, exist_ok=True)
            os.symlink(str(other_char), proc_root / "5003" / "fd" / "5")
            (proc_root / "5003" / "fdinfo" / "5").write_text("pos:\t0\nflags:\t02000002\n", encoding="utf-8")

            # PID 5004: path-only O_PATH descriptor (010000000) to host /dev/vfio/12 -> MUST NOT match!
            (proc_root / "5004" / "fd").mkdir(parents=True, exist_ok=True)
            (proc_root / "5004" / "fdinfo").mkdir(parents=True, exist_ok=True)
            os.symlink(str(host_vfio12), proc_root / "5004" / "fd" / "3")
            (proc_root / "5004" / "fdinfo" / "3").write_text(
                "pos:\t0\nflags:\t010000000\n", encoding="utf-8"
            )

            # PID 5005: missing fdinfo for matching char device -> fail closed, MUST NOT match!
            (proc_root / "5005" / "fd").mkdir(parents=True, exist_ok=True)
            os.symlink(str(host_vfio12), proc_root / "5005" / "fd" / "3")

            # PID 5006: holds BOTH an O_PATH fd (3) AND a real open fd (4) -> MUST match!
            (proc_root / "5006" / "fd").mkdir(parents=True, exist_ok=True)
            (proc_root / "5006" / "fdinfo").mkdir(parents=True, exist_ok=True)
            os.symlink(str(host_vfio12), proc_root / "5006" / "fd" / "3")
            (proc_root / "5006" / "fdinfo" / "3").write_text(
                "pos:\t0\nflags:\t010000000\n", encoding="utf-8"
            )
            os.symlink(str(host_vfio12), proc_root / "5006" / "fd" / "4")
            (proc_root / "5006" / "fdinfo" / "4").write_text(
                "pos:\t0\nflags:\t02000002\n", encoding="utf-8"
            )

            orig_os_stat = os.stat
            st12 = orig_os_stat(host_vfio12)
            st_bind = orig_os_stat(bind_mounted)
            st_other = orig_os_stat(other_char)
            ids_12 = {(st12.st_dev, st12.st_ino), (st_bind.st_dev, st_bind.st_ino)}
            id_other = (st_other.st_dev, st_other.st_ino)

            def _mock_os_stat(path_arg, *args, **kwargs):
                st = orig_os_stat(path_arg, *args, **kwargs)
                key = (st.st_dev, st.st_ino)
                if key in ids_12:
                    fake = mock.MagicMock()
                    fake.st_mode = _stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(242, 12)
                    return fake
                if key == id_other:
                    fake = mock.MagicMock()
                    fake.st_mode = _stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(242, 99)
                    return fake
                return st

            orig_os_lstat = os.lstat

            def _mock_os_lstat(path_arg, *args, **kwargs):
                st = orig_os_lstat(path_arg, *args, **kwargs)
                if not _stat.S_ISLNK(st.st_mode):
                    key = (st.st_dev, st.st_ino)
                    if key in ids_12:
                        fake = mock.MagicMock()
                        fake.st_mode = _stat.S_IFCHR | 0o600
                        fake.st_rdev = os.makedev(242, 12)
                        return fake
                    if key == id_other:
                        fake = mock.MagicMock()
                        fake.st_mode = _stat.S_IFCHR | 0o600
                        fake.st_rdev = os.makedev(242, 99)
                        return fake
                return st

            with (
                mock.patch("tpu_exporter.os.stat", side_effect=_mock_os_stat),
                mock.patch("tpu_exporter.os.lstat", side_effect=_mock_os_lstat),
            ):
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                )

            self.assertTrue(snap.pid_telemetry_available)
            self.assertEqual(snap.devices[0].active_pids, [5002, 5006])

            # Even when tpu-info reports decoy PID 5001, O_PATH PID 5004, and missing-fdinfo
            # PID 5005 alongside real holders 5002/5006, only active non-O_PATH opens count.
            mock_cli = root / "mock-tpu-info-decoy.sh"
            mock_cli.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Chips\n"
                "| Chip         | Type         | Devices | PID                    |\n"
                "|--------------|--------------|---------|------------------------|\n"
                "| /dev/vfio/12 | TPU v5e chip | 1       | 5001, 5002, 5004, 5005 |\n"
                "EOF\n",
                encoding="utf-8",
            )
            mock_cli.chmod(0o755)
            with (
                mock.patch("tpu_exporter.os.stat", side_effect=_mock_os_stat),
                mock.patch("tpu_exporter.os.lstat", side_effect=_mock_os_lstat),
            ):
                snap_cli = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=str(mock_cli),
                )
            self.assertTrue(snap_cli.tpu_info_available)
            self.assertEqual(snap_cli.devices[0].active_pids, [5002, 5006])

    def test_ambiguous_combined_sysfs_and_chip_id_map_leaves_runtime_unmapped(self) -> None:
        self.assertEqual(tpu_exporter.build_arg_parser().parse_args([]).health_host, "127.0.0.1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "chip_id": "0",  # sysfs assigns runtime ID 0 to chip A
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
            mock_cli = root / "mock-tpu-info-ambig.sh"
            mock_cli.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Device Status\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| Device ID | Memory Usage (Used/Total) | Memory Bar Gauge   | HBM Bandwidth Util | TensorCore Duty Cycle |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| 0         | 10.00 GiB / 16.00 GiB     | [||||||....] 62.5% | [|||.......] 30.0% | [||||||||..] 75.0%    |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "EOF\n",
                encoding="utf-8",
            )
            mock_cli.chmod(0o755)

            # Explicit map assigns runtime ID 0 to chip B while sysfs assigns 0 to chip A:
            # collision must leave BOTH chips without per-PCI metrics and keep row 0 unmapped.
            snap = tpu_exporter.collect_snapshot(
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=str(mock_cli),
                chip_id_map={"0000:00:06.0": "0"},
            )
            self.assertIsNone(snap.devices[0].hbm_used_bytes)
            self.assertIsNone(snap.devices[0].duty_cycle_percent)
            self.assertIsNone(snap.devices[1].hbm_used_bytes)
            self.assertIsNone(snap.devices[1].duty_cycle_percent)
            self.assertEqual(len(snap.runtime_utilization), 1)
            self.assertEqual(snap.runtime_utilization[0].runtime_chip, "0")
            self.assertIsNone(snap.runtime_utilization[0].mapped_pci_bdf)
            prom = tpu_exporter.format_prometheus_metrics(snap)
            self.assertNotIn("freetpu_tpu_duty_cycle_percent{", prom)
            self.assertIn('freetpu_tpu_runtime_duty_cycle_percent{runtime_chip="0",pci_bdf="unmapped"} 75', prom)

    def test_sidecar_runtime_metrics_succeed_without_proc_or_vfio_access(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Sidecar pod: NO /dev/vfio mounted, NO shareProcessNamespace, NO SYS_PTRACE.
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "enable": "1",
                    }
                ],
                create_vfio_container=False,
                create_vfio_nodes=[],
            )
            (proc_root / "999" / "fd").mkdir(parents=True)

            captured_calls: list[tuple[object, str]] = []

            class _FakeTpuChip:
                V6E = "v6e_enum"

            def _fake_get_chip_usage(chip_type, addr="127.0.0.1:8431"):
                captured_calls.append((chip_type, addr))
                return [
                    SimpleNamespace(
                        device_id=0,
                        memory_usage=12 * 1024**3,
                        total_memory=32 * 1024**3,
                        duty_cycle_pct=71.5,
                    )
                ]

            fake_tpu_info = SimpleNamespace()
            fake_device_mod = SimpleNamespace(TpuChip=_FakeTpuChip)
            fake_metrics_mod = SimpleNamespace(get_chip_usage=_fake_get_chip_usage)

            # 1. --no-pid-scan sidecar mode: never scans /proc and succeeds via loopback gRPC.
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "tpu_info": fake_tpu_info,
                        "tpu_info.device": fake_device_mod,
                        "tpu_info.metrics": fake_metrics_mod,
                    },
                ),
                mock.patch.object(
                    tpu_exporter,
                    "_scan_proc_vfio_pids",
                    side_effect=AssertionError("must not scan /proc in --no-pid-scan mode"),
                ),
            ):
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    skip_pid_scan=True,
                    runtime_metrics_addr="127.0.0.1:8431",
                )

            self.assertTrue(snap.tpu_info_available)
            self.assertFalse(snap.pid_telemetry_available)
            self.assertEqual(captured_calls, [("v6e_enum", "127.0.0.1:8431")])
            self.assertEqual(len(snap.runtime_utilization), 1)
            self.assertEqual(snap.runtime_utilization[0].hbm_used_bytes, 12 * 1024**3)
            self.assertAlmostEqual(snap.runtime_utilization[0].duty_cycle_percent or 0.0, 71.5)
            prom = tpu_exporter.format_prometheus_metrics(snap)
            self.assertIn('freetpu_tpu_runtime_duty_cycle_percent{runtime_chip="0",pci_bdf="unmapped"} 71.5', prom)
            self.assertNotIn("freetpu_tpu_active_pids{", prom)

            # 2. Default mode fallback: even if `tpu-info` CLI aborts due to cross-profile
            #    `/proc` PermissionError under containerd AppArmor, fallback to `tpu_info.metrics`
            #    preserves live HBM and duty-cycle telemetry.
            failing_cli = root / "mock-apparmor-eacces-tpu-info.sh"
            failing_cli.write_text(
                "#!/bin/sh\necho 'PermissionError: [Errno 13] Permission denied: /proc/999/fd/3' >&2\nexit 1\n",
                encoding="utf-8",
            )
            failing_cli.chmod(0o755)
            captured_calls.clear()
            with mock.patch.dict(
                sys.modules,
                {
                    "tpu_info": fake_tpu_info,
                    "tpu_info.device": fake_device_mod,
                    "tpu_info.metrics": fake_metrics_mod,
                },
            ):
                snap_fb = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=str(failing_cli),
                    skip_pid_scan=False,
                    runtime_metrics_addr="127.0.0.1:8431",
                    chip_id_map={"0000:00:06.0": "0"},
                )

            self.assertTrue(snap_fb.tpu_info_available)
            self.assertEqual(captured_calls, [("v6e_enum", "127.0.0.1:8431")])
            self.assertEqual(snap_fb.devices[0].hbm_used_bytes, 12 * 1024**3)
            self.assertAlmostEqual(snap_fb.devices[0].duty_cycle_percent or 0.0, 71.5)

            # 3. In --no-pid-scan mode, when direct runtime metrics return empty/unavailable
            #    (e.g. before libtpu initializes), exporter MUST NOT fall back to `tpu-info` CLI.
            empty_metrics_mod = SimpleNamespace(get_chip_usage=lambda chip_type, addr="": [])
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "tpu_info": fake_tpu_info,
                        "tpu_info.device": fake_device_mod,
                        "tpu_info.metrics": empty_metrics_mod,
                    },
                ),
                mock.patch.object(
                    tpu_exporter,
                    "_run_tpu_info",
                    side_effect=AssertionError("must not invoke tpu-info CLI in --no-pid-scan mode"),
                ),
            ):
                snap_empty = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    skip_pid_scan=True,
                    runtime_metrics_addr="127.0.0.1:8431",
                )
            self.assertFalse(snap_empty.tpu_info_available)
            self.assertEqual(len(snap_empty.devices), 1)
            self.assertEqual(snap_empty.runtime_utilization, [])

            # 4. Hanging direct `tpu_info.metrics.get_chip_usage` (in a real subprocess package)
            #    must be terminated by hard deadline, return fallback snapshot, and release
            #    `_ExporterServer._scrape_lock` so subsequent scrapes do not hang.
            fake_pkg_root = root / "fake_site"
            fake_pkg = fake_pkg_root / "tpu_info"
            fake_pkg.mkdir(parents=True, exist_ok=True)
            (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
            (fake_pkg / "device.py").write_text(
                "class TpuChip:\n    V6E = 'v6e'\n", encoding="utf-8"
            )
            (fake_pkg / "metrics.py").write_text(
                "import time\ndef get_chip_usage(chip_type, addr='127.0.0.1:8431'):\n    time.sleep(30)\n",
                encoding="utf-8",
            )
            env_with_fake = dict(os.environ)
            prev_pp = env_with_fake.get("PYTHONPATH", "")
            env_with_fake["PYTHONPATH"] = (
                f"{fake_pkg_root}{os.pathsep}{prev_pp}" if prev_pp else str(fake_pkg_root)
            )
            import time as _time

            server = tpu_exporter._ExporterServer(
                ("127.0.0.1", 0),
                tpu_exporter._ExporterHandler,
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin="tpu-info",
                tpu_info_timeout_s=0.25,
                skip_pid_scan=True,
                runtime_metrics_addr="127.0.0.1:8431",
            )
            try:
                with (
                    mock.patch("tpu_exporter.importlib.util.find_spec", return_value=object()),
                    mock.patch.dict(os.environ, {"PYTHONPATH": env_with_fake["PYTHONPATH"]}),
                ):
                    t_start = _time.monotonic()
                    snap_to = server.scrape()
                    elapsed = _time.monotonic() - t_start
                self.assertLess(elapsed, 3.0)
                self.assertFalse(snap_to.tpu_info_available)
                self.assertIn("timed out", snap_to.tpu_info_error or "")
                self.assertEqual(len(snap_to.devices), 1)
                # Scrape lock MUST be released after timeout so a subsequent scrape succeeds.
                self.assertTrue(server._scrape_lock.acquire(blocking=False))
                server._scrape_lock.release()
            finally:
                server.server_close()

    def test_vfio_group_exclusive_requires_verified_full_iommu_group_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    # Chip 0 (group 0): sole member of /sys/kernel/iommu_groups/0/devices -> exclusive=True
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                    # Chip 1 (group 1): shares IOMMU group 1 with a non-TPU PCI device -> exclusive=False
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                    },
                    {
                        "bdf": "0000:00:06.1",
                        "vendor": "0x1af4",
                        "device": "0x1041",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                    },
                    # Chip 2 (group 2): missing /sys/kernel/iommu_groups/2/devices -> fail closed exclusive=False
                    {
                        "bdf": "0000:00:07.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "2",
                    },
                    # Chips 3 & 4 (group 3): two TPUs sharing IOMMU group 3 -> both exclusive=False
                    {
                        "bdf": "0000:00:08.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                        "enable": "2",
                    },
                    {
                        "bdf": "0000:00:09.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                    },
                    # Chip 5 (group 4): TPU bound to vfio-pci sharing IOMMU group with host-driver endpoint -> healthy=False
                    {
                        "bdf": "0000:00:0a.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "4",
                    },
                    {
                        "bdf": "0000:00:0a.1",
                        "vendor": "0x1af4",
                        "device": "0x1041",
                        "driver": "virtio-pci",
                        "iommu_group": "4",
                    },
                ],
                create_vfio_nodes=["0", "1", "2", "3", "4"],
            )
            # Remove /sys/kernel/iommu_groups/2/devices to simulate unreadable/missing group dir.
            grp2_devs = sysfs_root / "kernel" / "iommu_groups" / "2" / "devices"
            for child in list(grp2_devs.iterdir()):
                child.unlink()
            grp2_devs.rmdir()

            with _mock_vfio_char_lstat():
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                )
            by_bdf = {d.pci_bdf: d for d in snap.devices}
            self.assertTrue(by_bdf["0000:00:05.0"].vfio_group_exclusive)
            self.assertTrue(by_bdf["0000:00:05.0"].healthy)
            self.assertFalse(by_bdf["0000:00:06.0"].vfio_group_exclusive)
            self.assertTrue(by_bdf["0000:00:06.0"].healthy)
            self.assertFalse(by_bdf["0000:00:07.0"].vfio_group_exclusive)
            self.assertFalse(by_bdf["0000:00:07.0"].healthy)
            self.assertFalse(by_bdf["0000:00:08.0"].vfio_group_exclusive)
            self.assertTrue(by_bdf["0000:00:08.0"].healthy)
            # Sysfs PCI enable count > 1 (e.g. "2") must report pci_enabled=True.
            self.assertTrue(by_bdf["0000:00:08.0"].pci_enabled)
            self.assertFalse(by_bdf["0000:00:09.0"].vfio_group_exclusive)
            self.assertTrue(by_bdf["0000:00:09.0"].healthy)
            # Non-viable VFIO group (sibling endpoint bound to host driver) must report healthy=False.
            self.assertFalse(by_bdf["0000:00:0a.0"].vfio_group_exclusive)
            self.assertFalse(by_bdf["0000:00:0a.0"].healthy)

            prom = tpu_exporter.format_prometheus_metrics(snap)
            self._assert_valid_prometheus_text(prom)
            self.assertIn('pci_bdf="0000:00:05.0"', prom)
            self.assertIn('iommu_group="0",vfio_group_exclusive="true"', prom)
            self.assertIn('iommu_group="1",vfio_group_exclusive="false"', prom)
            self.assertIn('iommu_group="2",vfio_group_exclusive="false"', prom)
            self.assertIn('iommu_group="3",vfio_group_exclusive="false"', prom)

    def test_unverified_vfio_node_marks_pid_telemetry_unavailable(self) -> None:
        import stat
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                    {
                        "bdf": "0000:00:07.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                    },
                ],
                # Only group 0 has a host /dev/vfio/0 node; group 1 node is missing,
                # and group 0 is a regular file (non-char) unless mocked as S_ISCHR.
                create_vfio_nodes=["0"],
            )
            orig_stat = os.stat
            orig_lstat = os.lstat
            vfio0 = str(dev_root / "vfio" / "0")
            vfio_ctrl = str(dev_root / "vfio" / "vfio")

            def _only_group0_is_char_stat(path, *args, **kwargs):
                st = orig_stat(path, *args, **kwargs)
                if os.fspath(path) == vfio0:
                    fake = mock.MagicMock()
                    fake.st_mode = stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(242, 0)
                    return fake
                if os.fspath(path) == vfio_ctrl:
                    fake = mock.MagicMock()
                    fake.st_mode = stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(10, 196)
                    return fake
                return st

            def _only_group0_is_char_lstat(path, *args, **kwargs):
                st = orig_lstat(path, *args, **kwargs)
                if stat.S_ISLNK(st.st_mode):
                    return st
                if os.fspath(path) == vfio0:
                    fake = mock.MagicMock()
                    fake.st_mode = stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(242, 0)
                    return fake
                if os.fspath(path) == vfio_ctrl:
                    fake = mock.MagicMock()
                    fake.st_mode = stat.S_IFCHR | 0o600
                    fake.st_rdev = os.makedev(10, 196)
                    return fake
                return st

            with (
                mock.patch("tpu_exporter.os.stat", side_effect=_only_group0_is_char_stat),
                mock.patch("tpu_exporter.os.lstat", side_effect=_only_group0_is_char_lstat),
            ):
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                )

            by_bdf = {d.pci_bdf: d for d in snap.devices}
            self.assertTrue(by_bdf["0000:00:06.0"].healthy)
            self.assertFalse(by_bdf["0000:00:07.0"].healthy)
            self.assertTrue(by_bdf["0000:00:06.0"].pid_telemetry_available)
            self.assertFalse(by_bdf["0000:00:07.0"].pid_telemetry_available)
            self.assertFalse(snap.pid_telemetry_available)

            prom = tpu_exporter.format_prometheus_metrics(snap)
            self._assert_valid_prometheus_text(prom)
            self.assertIn('freetpu_tpu_pid_telemetry_available{chip="0",pci_bdf="0000:00:06.0",device_id="0x006f",generation="v6e",vfio_device="/dev/vfio/0"} 1', prom)
            self.assertIn('freetpu_tpu_pid_telemetry_available{chip="1",pci_bdf="0000:00:07.0",device_id="0x006f",generation="v6e",vfio_device="/dev/vfio/1"} 0', prom)
            self.assertIn('freetpu_tpu_active_pids{chip="0",pci_bdf="0000:00:06.0",device_id="0x006f",generation="v6e",vfio_device="/dev/vfio/0"} 0', prom)
            self.assertNotIn('freetpu_tpu_active_pids{chip="1",pci_bdf="0000:00:07.0"', prom)

            # Mismatched kernel sysfs /sys/class/vfio/0/dev (242:99 vs char st_rdev 242:0)
            # MUST mark the device unhealthy AND suppress PID telemetry and active PID counts.
            (sysfs_root / "class" / "vfio" / "0" / "dev").write_text("242:99\n", encoding="utf-8")
            with (
                mock.patch("tpu_exporter.os.stat", side_effect=_only_group0_is_char_stat),
                mock.patch("tpu_exporter.os.lstat", side_effect=_only_group0_is_char_lstat),
            ):
                snap_mm = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                )
            by_bdf_mm = {d.pci_bdf: d for d in snap_mm.devices}
            self.assertFalse(by_bdf_mm["0000:00:06.0"].healthy)
            self.assertFalse(by_bdf_mm["0000:00:06.0"].pid_telemetry_available)
            prom_mm = tpu_exporter.format_prometheus_metrics(snap_mm)
            self.assertNotIn('freetpu_tpu_active_pids{chip="0",pci_bdf="0000:00:06.0"', prom_mm)

    def test_runtime_telemetry_fallback_shares_single_scrape_deadline(self) -> None:
        import time as _time
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
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

            # When CLI exhausts the shared timeout budget, Python fallback MUST be skipped
            # rather than spending a second full timeout_s.
            py_called = []

            def _slow_cli(*args, **kwargs):
                _time.sleep(0.25)
                return False, "tpu-info timed out", {}

            def _record_py(*args, **kwargs):
                py_called.append(kwargs.get("timeout_s"))
                return True, None, {}

            with (
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_slow_cli),
                mock.patch.object(tpu_exporter, "_query_tpu_info_python_metrics", side_effect=_record_py),
            ):
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=0.2,
                    runtime_metrics_addr="127.0.0.1:8431",
                )
            self.assertEqual(py_called, [])
            self.assertFalse(snap.tpu_info_available)

            # 2. Direct gRPC MUST cap each individual RPC's timeout AND reserve scrape
            #    budget for CLI fallback so a stalled first RPC (dutycycle) neither
            #    starves independent HBM RPCs nor prevents CLI fallback when all RPCs fail.
            from types import SimpleNamespace

            seen_grpc_timeouts: list[tuple[str, float]] = []

            def _make_simple_item(chip_id: int, as_int: int) -> SimpleNamespace:
                val = SimpleNamespace(
                    int_attr=chip_id,
                    string_attr="",
                    WhichOneof=lambda _g: "int_attr",
                    HasField=lambda f: f == "int_attr",
                )
                gauge = SimpleNamespace(
                    as_int=as_int,
                    as_double=0.0,
                    WhichOneof=lambda _g: "as_int",
                    HasField=lambda f: f == "as_int",
                )
                return SimpleNamespace(
                    attribute=SimpleNamespace(key="device_id", value=val),
                    gauge=gauge,
                )

            class _SlowDutyFastHbmStub:
                def __init__(self, _channel):
                    pass

                def GetRuntimeMetric(self, req, timeout=None):
                    seen_grpc_timeouts.append((req.metric_name, float(timeout or 0.0)))
                    if req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        raise RuntimeError("DEADLINE_EXCEEDED")
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        return SimpleNamespace(
                            metric=SimpleNamespace(metrics=[_make_simple_item(0, 12 * 1024**3)])
                        )
                    if req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        return SimpleNamespace(
                            metric=SimpleNamespace(metrics=[_make_simple_item(0, 32 * 1024**3)])
                        )
                    return SimpleNamespace(metric=SimpleNamespace(metrics=[]))

            fake_grpc = SimpleNamespace(
                LocalConnectionType=SimpleNamespace(LOCAL_TCP="LOCAL_TCP", UDS="UDS"),
                local_channel_credentials=lambda conn_type: SimpleNamespace(conn_type=conn_type),
                secure_channel=lambda addr, creds, options=None: SimpleNamespace(addr=addr),
            )
            fake_pb2 = SimpleNamespace(
                MetricRequest=lambda metric_name: SimpleNamespace(metric_name=metric_name)
            )
            fake_pb2_grpc = SimpleNamespace(RuntimeMetricServiceStub=_SlowDutyFastHbmStub)

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(tpu_exporter._GRPC_CHANNELS, {}, clear=True),
                mock.patch.object(
                    tpu_exporter,
                    "_load_grpc_runtime_modules",
                    return_value=(fake_grpc, fake_pb2, fake_pb2_grpc),
                ),
            ):
                partial_grpc_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=2.0,
                    chip_id_map={"0000:00:06.0": "0"},
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            # First RPC (dutycycle) was capped well below full 2.0s budget (<= 2.0 * 0.65 / 4),
            # and HBM RPCs still succeeded despite dutycycle failing!
            self.assertEqual(len(seen_grpc_timeouts), 4)
            self.assertLessEqual(seen_grpc_timeouts[0][1], 0.5)
            self.assertEqual(partial_grpc_snap.devices[0].hbm_used_bytes, 12 * 1024**3)
            self.assertEqual(partial_grpc_snap.devices[0].hbm_total_bytes, 32 * 1024**3)

            # 2b. When direct gRPC returns HBM without duty-cycle (partial reply),
            #     fallback MUST run with the remaining scrape budget, fill in the
            #     missing duty-cycle, AND preserve direct gRPC's HBM readings
            #     (both in host CLI fallback mode and sidecar Python fallback mode)!
            partial_cli_calls: list[float] = []

            def _fill_duty_cli(*args, **kwargs):
                partial_cli_calls.append(float(kwargs.get("timeout_s", 0.0)))
                rows = tpu_exporter.parse_tpu_info_output(
                    "TPU Runtime Utilization\n"
                    "| Chip | HBM Usage (GiB)      | Duty cycle |\n"
                    "|------|----------------------|------------|\n"
                    "| 0    | 9.00 GiB / 32.00 GiB | 61.50%     |\n"
                )
                return True, None, rows

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(tpu_exporter._GRPC_CHANNELS, {}, clear=True),
                mock.patch.object(
                    tpu_exporter,
                    "_load_grpc_runtime_modules",
                    return_value=(fake_grpc, fake_pb2, fake_pb2_grpc),
                ),
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_fill_duty_cli),
            ):
                filled_host_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=2.0,
                    chip_id_map={"0000:00:06.0": "0"},
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertEqual(len(partial_cli_calls), 1)
            self.assertGreater(partial_cli_calls[0], 0.5)
            # Preserved direct-gRPC HBM (12 GiB, not overwritten by CLI's 9 GiB) + filled duty (61.5%).
            self.assertEqual(filled_host_snap.devices[0].hbm_used_bytes, 12 * 1024**3)
            self.assertEqual(filled_host_snap.devices[0].hbm_total_bytes, 32 * 1024**3)
            self.assertAlmostEqual(filled_host_snap.devices[0].duty_cycle_percent or 0.0, 61.5)

            partial_py_calls: list[float] = []

            def _fill_duty_py(*args, **kwargs):
                partial_py_calls.append(float(kwargs.get("timeout_s", 0.0)))
                rows = tpu_exporter.parse_tpu_info_output(
                    "TPU Runtime Utilization\n"
                    "| Chip | HBM Usage (GiB)      | Duty cycle |\n"
                    "|------|----------------------|------------|\n"
                    "| 0    | 7.00 GiB / 32.00 GiB | 58.00%     |\n"
                )
                return True, None, rows

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(tpu_exporter._GRPC_CHANNELS, {}, clear=True),
                mock.patch.object(
                    tpu_exporter,
                    "_load_grpc_runtime_modules",
                    return_value=(fake_grpc, fake_pb2, fake_pb2_grpc),
                ),
                mock.patch.object(
                    tpu_exporter, "_query_tpu_info_python_metrics", side_effect=_fill_duty_py
                ),
            ):
                filled_sidecar_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=2.0,
                    chip_id_map={"0000:00:06.0": "0"},
                    skip_pid_scan=True,
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertEqual(len(partial_py_calls), 1)
            self.assertEqual(filled_sidecar_snap.devices[0].hbm_used_bytes, 12 * 1024**3)
            self.assertEqual(filled_sidecar_snap.devices[0].hbm_total_bytes, 32 * 1024**3)
            self.assertAlmostEqual(filled_sidecar_snap.devices[0].duty_cycle_percent or 0.0, 58.0)

            # Conversely, when direct gRPC returns duty-cycle only (HBM timed out),
            # fallback fills missing HBM while preserving gRPC's duty-cycle reading.
            duty_only_row = tpu_exporter._TpuInfoRow(
                chip_key="0",
                duty_cycle_percent=73.0,
                in_util_table=True,
            )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.object(
                    tpu_exporter,
                    "_query_grpc_runtime_metrics",
                    return_value=(True, None, {"0": duty_only_row}),
                ),
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_fill_duty_cli),
            ):
                filled_hbm_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=2.0,
                    chip_id_map={"0000:00:06.0": "0"},
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertEqual(filled_hbm_snap.devices[0].hbm_used_bytes, 9 * 1024**3)
            self.assertEqual(filled_hbm_snap.devices[0].hbm_total_bytes, 32 * 1024**3)
            self.assertAlmostEqual(filled_hbm_snap.devices[0].duty_cycle_percent or 0.0, 73.0)

            # 2c. Per-device completeness check: one complete row (or HBM on chip 0 +
            #     duty on chip 1) MUST NOT suppress fallback when another runtime device
            #     is missing HBM or duty-cycle, AND fallback merges strictly missing
            #     fields while preserving already-collected gRPC readings per row!
            cross_row_partial = {
                "0": tpu_exporter._TpuInfoRow(
                    chip_key="0",
                    hbm_used_bytes=12 * 1024**3,
                    hbm_total_bytes=32 * 1024**3,
                    duty_cycle_percent=None,
                    in_util_table=True,
                ),
                "1": tpu_exporter._TpuInfoRow(
                    chip_key="1",
                    hbm_used_bytes=None,
                    hbm_total_bytes=None,
                    duty_cycle_percent=82.0,
                    in_util_table=True,
                ),
            }
            self.assertFalse(tpu_exporter._has_complete_utilization(cross_row_partial))

            one_complete_one_partial = {
                "0": tpu_exporter._TpuInfoRow(
                    chip_key="0",
                    hbm_used_bytes=12 * 1024**3,
                    hbm_total_bytes=32 * 1024**3,
                    duty_cycle_percent=40.0,
                    in_util_table=True,
                ),
                "1": tpu_exporter._TpuInfoRow(
                    chip_key="1",
                    hbm_used_bytes=18 * 1024**3,
                    hbm_total_bytes=32 * 1024**3,
                    duty_cycle_percent=None,
                    in_util_table=True,
                ),
            }
            self.assertFalse(tpu_exporter._has_complete_utilization(one_complete_one_partial))

            multi_cli_calls: list[float] = []

            def _fill_multi_cli(*args, **kwargs):
                multi_cli_calls.append(float(kwargs.get("timeout_s", 0.0)))
                rows = tpu_exporter.parse_tpu_info_output(
                    "TPU Runtime Utilization\n"
                    "| Chip | HBM Usage (GiB)      | Duty cycle |\n"
                    "|------|----------------------|------------|\n"
                    "| 0    | 5.00 GiB / 32.00 GiB | 25.00%     |\n"
                    "| 1    | 6.00 GiB / 32.00 GiB | 67.50%     |\n"
                )
                return True, None, rows

            with (
                _mock_vfio_char_lstat(),
                mock.patch.object(
                    tpu_exporter,
                    "_query_grpc_runtime_metrics",
                    return_value=(True, None, one_complete_one_partial),
                ),
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_fill_multi_cli),
            ):
                multi_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=2.0,
                    chip_id_map={"0000:00:06.0": "0"},
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertEqual(len(multi_cli_calls), 1)
            rt_by_chip = {r.runtime_chip: r for r in multi_snap.runtime_utilization}
            # Chip 0 keeps direct-gRPC HBM (12 GiB), duty (40%), and pure "libtpu-grpc" provenance;
            # chip 1 keeps gRPC HBM (18 GiB), fills missing duty (67.5%), and records "libtpu-grpc+tpu-info"!
            self.assertEqual(rt_by_chip["0"].hbm_used_bytes, 12 * 1024**3)
            self.assertAlmostEqual(rt_by_chip["0"].duty_cycle_percent or 0.0, 40.0)
            self.assertEqual(rt_by_chip["0"].telemetry_source, "libtpu-grpc")
            self.assertEqual(multi_snap.devices[0].telemetry_source, "libtpu-grpc")
            self.assertEqual(rt_by_chip["1"].hbm_used_bytes, 18 * 1024**3)
            self.assertAlmostEqual(rt_by_chip["1"].duty_cycle_percent or 0.0, 67.5)
            self.assertEqual(rt_by_chip["1"].telemetry_source, "libtpu-grpc+tpu-info")

            # Multi-device provenance check: when gRPC returns a partial row for chip 0
            # and NO row for chip 1, and fallback CLI supplies chip 1 completely,
            # chip 0 MUST record "libtpu-grpc+tpu-info" and chip 1 MUST record "tpu-info"
            # (NEVER stamping global "libtpu-grpc" onto fallback-only chip 1)!
            prov_root = root / "per_device_source_provenance"
            prov_sys, prov_dev, prov_proc = _make_mock_pci_tree(
                prov_root,
                devices=[
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "chip_id": "0",
                    },
                    {
                        "bdf": "0000:00:07.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "1",
                        "chip_id": "1",
                    },
                ],
                create_vfio_nodes=["0", "1"],
            )
            grpc_chip0_only_partial = {
                "0": tpu_exporter._TpuInfoRow(
                    chip_key="0",
                    hbm_used_bytes=12 * 1024**3,
                    hbm_total_bytes=32 * 1024**3,
                    duty_cycle_percent=None,
                    in_util_table=True,
                )
            }
            with (
                _mock_vfio_char_lstat(),
                mock.patch.object(
                    tpu_exporter,
                    "_query_grpc_runtime_metrics",
                    return_value=(True, None, grpc_chip0_only_partial),
                ),
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_fill_multi_cli),
            ):
                prov_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=prov_sys,
                    dev_root=prov_dev,
                    proc_root=prov_proc,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=2.0,
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=prov_root / "no-shm",
                )
            self.assertEqual(prov_snap.devices[0].telemetry_source, "libtpu-grpc+tpu-info")
            self.assertEqual(prov_snap.devices[1].telemetry_source, "tpu-info")
            prov_prom = tpu_exporter.format_prometheus_metrics(prov_snap)
            self.assertIn(
                'pci_bdf="0000:00:06.0",device_id="0x006f",generation="v6e",vfio_device="/dev/vfio/0",'
                'vendor="0x1ae0",model="TPU v6e (Trillium)",driver="vfio-pci",iommu_group="0",'
                'vfio_group_exclusive="true",telemetry_source="libtpu-grpc+tpu-info"',
                prov_prom,
            )
            self.assertIn(
                'pci_bdf="0000:00:07.0",device_id="0x006f",generation="v6e",vfio_device="/dev/vfio/1",'
                'vendor="0x1ae0",model="TPU v6e (Trillium)",driver="vfio-pci",iommu_group="1",'
                'vfio_group_exclusive="true",telemetry_source="tpu-info"',
                prov_prom,
            )

            # 3. When gRPC consumes its reserved 65% slice and fails completely, CLI
            #    fallback MUST still receive the reserved ~35% budget and succeed!
            cli_timeouts: list[float] = []

            def _stalled_grpc(*args, **kwargs):
                grpc_to = float(kwargs.get("timeout_s", 0.0))
                self.assertLessEqual(grpc_to, 0.27)
                _time.sleep(min(0.12, grpc_to))
                return False, "gRPC timed out", {}

            def _ok_cli(*args, **kwargs):
                cli_timeouts.append(float(kwargs.get("timeout_s", 0.0)))
                rows = tpu_exporter.parse_tpu_info_output(
                    "TPU Runtime Utilization\n"
                    "| Chip | HBM Usage (GiB)      | Duty cycle |\n"
                    "|------|----------------------|------------|\n"
                    "| 0    | 6.00 GiB / 32.00 GiB | 45.00%     |\n"
                )
                return True, None, rows

            with (
                _mock_vfio_char_lstat(),
                mock.patch.object(tpu_exporter, "_query_grpc_runtime_metrics", side_effect=_stalled_grpc),
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_ok_cli),
            ):
                fb_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=0.4,
                    chip_id_map={"0000:00:06.0": "0"},
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertEqual(len(cli_timeouts), 1)
            self.assertGreaterEqual(cli_timeouts[0], 0.1)
            self.assertTrue(fb_snap.tpu_info_available)
            self.assertEqual(fb_snap.devices[0].duty_cycle_percent, 45.0)

            # 4. Sidecar (`skip_pid_scan=True`) and non-default runtime-address modes
            #    MUST ALSO reserve ~35% of the shared budget for the Python fallback!
            sidecar_py_timeouts: list[float] = []

            def _ok_py(*args, **kwargs):
                sidecar_py_timeouts.append(float(kwargs.get("timeout_s", 0.0)))
                rows = tpu_exporter.parse_tpu_info_output(
                    "TPU Runtime Utilization\n"
                    "| Chip | HBM Usage (GiB)      | Duty cycle |\n"
                    "|------|----------------------|------------|\n"
                    "| 0    | 8.00 GiB / 32.00 GiB | 55.00%     |\n"
                )
                return True, None, rows

            with (
                _mock_vfio_char_lstat(),
                mock.patch.object(tpu_exporter, "_query_grpc_runtime_metrics", side_effect=_stalled_grpc),
                mock.patch.object(tpu_exporter, "_query_tpu_info_python_metrics", side_effect=_ok_py),
            ):
                sc_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=0.4,
                    chip_id_map={"0000:00:06.0": "0"},
                    skip_pid_scan=True,
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertEqual(len(sidecar_py_timeouts), 1)
            self.assertGreaterEqual(sidecar_py_timeouts[0], 0.1)
            self.assertTrue(sc_snap.tpu_info_available)
            self.assertEqual(sc_snap.devices[0].duty_cycle_percent, 55.0)

            # When CLI fails fast, Python fallback receives strictly the remaining shared budget.
            def _fast_fail_cli(*args, **kwargs):
                _time.sleep(0.05)
                return False, "PermissionError", {}

            with (
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_fast_fail_cli),
                mock.patch.object(tpu_exporter, "_query_tpu_info_python_metrics", side_effect=_record_py),
            ):
                snap2 = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=0.5,
                    runtime_metrics_addr="127.0.0.1:8431",
                )
            self.assertEqual(len(py_called), 1)
            self.assertLess(py_called[0], 0.5)
            self.assertGreater(py_called[0], 0.05)
            self.assertTrue(snap2.tpu_info_available)

            # When CLI exits 0 with ONLY inventory/N/A rows (no live utilization) on the default
            # runtime endpoint, preserve inventory rows and query runtime_metrics_addr within budget.
            py_called.clear()

            def _cli_inventory_only(*args, **kwargs):
                return (
                    True,
                    None,
                    {
                        "/dev/vfio/0": tpu_exporter._TpuInfoRow(
                            chip_key="/dev/vfio/0",
                            chip_type="TPU v6e chip",
                            pids=[],
                            in_chips_table=True,
                        )
                    },
                )

            def _py_live_util(*args, **kwargs):
                py_called.append(kwargs.get("addr"))
                return (
                    True,
                    None,
                    {
                        "0": tpu_exporter._TpuInfoRow(
                            chip_key="0",
                            hbm_used_bytes=4 * 1024**3,
                            hbm_total_bytes=32 * 1024**3,
                            duty_cycle_percent=55.0,
                            in_util_table=True,
                        )
                    },
                )

            with (
                mock.patch.object(tpu_exporter, "_run_tpu_info", side_effect=_cli_inventory_only),
                mock.patch.object(tpu_exporter, "_query_tpu_info_python_metrics", side_effect=_py_live_util),
            ):
                snap3 = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=0.5,
                    runtime_metrics_addr="127.0.0.1:8431",
                )
            self.assertEqual(py_called, ["127.0.0.1:8431"])
            self.assertTrue(snap3.tpu_info_available)
            self.assertEqual(len(snap3.runtime_utilization), 1)
            self.assertEqual(snap3.runtime_utilization[0].hbm_used_bytes, 4 * 1024**3)
            self.assertAlmostEqual(snap3.runtime_utilization[0].duty_cycle_percent or 0.0, 55.0)

            # When an explicit non-default `--runtime-metrics-addr` is supplied (with PID scanning
            # enabled), query it directly without invoking `tpu-info` CLI against default port 8431.
            py_called.clear()
            with (
                mock.patch.object(
                    tpu_exporter,
                    "_run_tpu_info",
                    side_effect=AssertionError("must not call CLI against default 8431 when explicit addr set"),
                ),
                mock.patch.object(tpu_exporter, "_query_tpu_info_python_metrics", side_effect=_py_live_util),
            ):
                snap4 = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    tpu_info_timeout_s=0.5,
                    runtime_metrics_addr="127.0.0.1:9999",
                )
            self.assertEqual(py_called, ["127.0.0.1:9999"])
            self.assertTrue(snap4.tpu_info_available)

            # Empty/None `tpu_info_bin` disables Python gRPC runtime telemetry even in
            # `--no-pid-scan` or explicit `--runtime-metrics-addr` mode.
            with mock.patch.object(
                tpu_exporter,
                "_query_tpu_info_python_metrics",
                side_effect=AssertionError("must not query Python metrics when tpu_info_bin is disabled"),
            ):
                snap_disabled = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                    skip_pid_scan=True,
                    runtime_metrics_addr="127.0.0.1:9999",
                )
            self.assertFalse(snap_disabled.tpu_info_available)

            # Negative classic/dashboard `tpu-info` readings must be omitted (never
            # parsed as positive magnitudes via unsigned regex search).
            neg_classic = """
TPU Runtime Utilization
+------+------------------------+------------+
| Chip | HBM Usage              | Duty Cycle |
+------+------------------------+------------+
| 0    | -1.00 GiB / 32.00 GiB  | -1.00%     |
| 1    | 4.00 GiB / -32.00 GiB  | 0.00%      |
| 2    | 8.00 GiB / 32.00 GiB   | 42.50%     |
+------+------------------------+------------+
"""
            neg_rows = tpu_exporter.parse_tpu_info_output(neg_classic)
            self.assertIsNone(neg_rows["0"].hbm_used_bytes)
            self.assertIsNone(neg_rows["0"].hbm_total_bytes)
            self.assertIsNone(neg_rows["0"].duty_cycle_percent)
            self.assertIsNone(neg_rows["1"].hbm_used_bytes)
            self.assertIsNone(neg_rows["1"].hbm_total_bytes)
            self.assertEqual(neg_rows["1"].duty_cycle_percent, 0.0)
            self.assertEqual(neg_rows["2"].hbm_used_bytes, 8 * 1024**3)
            self.assertEqual(neg_rows["2"].hbm_total_bytes, 32 * 1024**3)
            self.assertEqual(neg_rows["2"].duty_cycle_percent, 42.5)

            # `--pid-scan` cleanly overrides an earlier `--no-pid-scan` (e.g. Docker ENTRYPOINT).
            parser = tpu_exporter.build_arg_parser()
            self.assertTrue(parser.parse_args(["--no-pid-scan"]).no_pid_scan)
            self.assertFalse(parser.parse_args(["--no-pid-scan", "--pid-scan"]).no_pid_scan)

            # Timing out a `tpu-info` CLI process must kill its entire process group
            # (including grandchildren spawned without their own timeout).
            child_pid_file = root / "grandchild.pid"
            stalling_cli = root / "stalling-tpu-info"
            stalling_cli.write_text(
                f"#!{sys.executable}\n"
                "import subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                f"open({str(child_pid_file)!r}, 'w').write(str(child.pid))\n"
                "child.wait()\n",
                encoding="utf-8",
            )
            stalling_cli.chmod(0o755)
            ok_stall, err_stall, _rows_stall = tpu_exporter._run_tpu_info(
                str(stalling_cli), timeout_s=0.5
            )
            self.assertFalse(ok_stall)
            self.assertIn("timed out", (err_stall or "").lower())
            self.assertTrue(child_pid_file.exists())
            grandchild_pid = int(child_pid_file.read_text(encoding="utf-8").strip())
            # Wait briefly for kernel process reaping after SIGKILL to process group.
            # Treat both missing processes (ProcessLookupError) and un-reaped zombies
            # (e.g. in containers without an init reaper) as terminated.
            import time as _time2
            deadline_reap = _time2.monotonic() + 2.0
            alive = True
            while _time2.monotonic() < deadline_reap:
                try:
                    os.kill(grandchild_pid, 0)
                except ProcessLookupError:
                    alive = False
                    break
                proc_stat = Path(f"/proc/{grandchild_pid}/stat")
                if proc_stat.exists():
                    try:
                        stat_text = proc_stat.read_text(encoding="utf-8", errors="replace")
                        rparen = stat_text.rfind(")")
                        if rparen != -1:
                            state_fields = stat_text[rparen + 1 :].split()
                            if state_fields and state_fields[0] in ("Z", "X", "x"):
                                alive = False
                                break
                    except OSError:
                        alive = False
                        break
                else:
                    ps_res = subprocess.run(
                        ["ps", "-o", "stat=", "-p", str(grandchild_pid)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    st = ps_res.stdout.strip()
                    if ps_res.returncode != 0 or not st or st.startswith("Z"):
                        alive = False
                        break
                _time2.sleep(0.05)
            self.assertFalse(alive, f"grandchild pid {grandchild_pid} survived tpu-info timeout")

            # Procfs `hidepid=2` / `hidepid=invisible` must mark PID telemetry unavailable
            # unless the caller holds the configured bypass group (or root for default gid=0).
            self_dir = proc_root / "self"
            self_dir.mkdir(parents=True, exist_ok=True)
            mountinfo = self_dir / "mountinfo"
            mountinfo.write_text(
                "26 20 0:23 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw,hidepid=2,gid=999999\n",
                encoding="utf-8",
            )
            self.assertTrue(tpu_exporter._procfs_hides_pids(proc_root))
            mountinfo.write_text(
                "26 20 0:23 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw,hidepid=invisible,gid=999999\n",
                encoding="utf-8",
            )
            self.assertTrue(tpu_exporter._procfs_hides_pids(proc_root))
            mountinfo.write_text(
                f"26 20 0:23 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw,hidepid=2,gid={os.getegid()}\n",
                encoding="utf-8",
            )
            self.assertFalse(tpu_exporter._procfs_hides_pids(proc_root))
            # `hidepid=ptraceable` / `4` checks ptrace before `gid=`, so even matching gid must fail closed.
            mountinfo.write_text(
                f"26 20 0:23 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw,hidepid=ptraceable,gid={os.getegid()}\n",
                encoding="utf-8",
            )
            self.assertTrue(tpu_exporter._procfs_hides_pids(proc_root))
            mountinfo.write_text(
                f"26 20 0:23 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw,hidepid=4,gid={os.getegid()}\n",
                encoding="utf-8",
            )
            self.assertTrue(tpu_exporter._procfs_hides_pids(proc_root))
            mountinfo.write_text(
                "26 20 0:23 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw,hidepid=0\n",
                encoding="utf-8",
            )
            self.assertFalse(tpu_exporter._procfs_hides_pids(proc_root))
            mountinfo.unlink()

            # Server mode must handle SIGTERM cleanly (e.g. when running as container PID 1).
            srv_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(EXPORTER_DIR / "tpu_exporter.py"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--health-host",
                    "127.0.0.1",
                    "--health-port",
                    "0",
                    "--sysfs-root",
                    str(sysfs_root),
                    "--dev-root",
                    str(dev_root),
                    "--proc-root",
                    str(proc_root),
                    "--no-pid-scan",
                    "--tpu-info-bin",
                    "",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert srv_proc.stderr is not None
                ready_lines = []
                for _ in range(2):
                    line = srv_proc.stderr.readline()
                    ready_lines.append(line)
                    if "listening on" in line and "/metrics" in line:
                        break
                self.assertTrue(
                    any("/metrics" in ln for ln in ready_lines),
                    f"exporter did not report readiness: {ready_lines!r}",
                )
                srv_proc.terminate()
                rc = srv_proc.wait(timeout=3.0)
                self.assertEqual(rc, 0)
            finally:
                if srv_proc.poll() is None:
                    srv_proc.kill()
                    srv_proc.wait(timeout=2.0)
                for p in (srv_proc.stdout, srv_proc.stderr):
                    if p is not None:
                        p.close()

            # Oversized HTTP headers (> MAX_HTTP_HEADER_BYTES) are rejected with 431 before scrape.
            import socket as _socket

            srv = tpu_exporter._ExporterServer(
                ("127.0.0.1", 0),
                tpu_exporter._ExporterHandler,
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=None,
                tpu_info_timeout_s=1.0,
            )
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            try:
                port = srv.server_address[1]
                with _socket.create_connection(("127.0.0.1", port), timeout=3.0) as s:
                    big = b"GET /metrics HTTP/1.1\r\nHost: x\r\nX-Pad: " + (b"A" * (tpu_exporter.MAX_HTTP_HEADER_BYTES + 128)) + b"\r\n\r\n"
                    s.sendall(big)
                    resp = s.recv(1024)
                self.assertIn(b"431", resp)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_direct_grpc_runtime_metric_service_no_subprocess_and_graceful_fallback(self) -> None:
        from types import SimpleNamespace
        import tpu_exporter as _te

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",  # TPU v6e (devices_per_chip=1)
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                        "chip_id": "0",
                    },
                    {
                        "bdf": "0000:00:0c.0",
                        "vendor": "0x1ae0",
                        "device": "0x0062",  # TPU v5p (devices_per_chip=1)
                        "driver": "vfio-pci",
                        "iommu_group": "2",
                        "chip_id": "1",
                    },
                    {
                        "bdf": "0000:00:0d.0",
                        "vendor": "0x1ae0",
                        "device": "0x0063",  # TPU v5e (devices_per_chip=1)
                        "driver": "vfio-pci",
                        "iommu_group": "4",
                    },
                ],
                create_vfio_nodes=["3", "2", "4"],
            )

            created_channels: list[tuple[str, object, object]] = []
            rpc_calls: list[tuple[str, float]] = []

            class _FakeLocalConnType:
                LOCAL_TCP = "LOCAL_TCP"
                UDS = "UDS"

            class _FakeGrpc:
                LocalConnectionType = _FakeLocalConnType

                @staticmethod
                def local_channel_credentials(conn_type):
                    return ("local_creds", conn_type)

                @staticmethod
                def secure_channel(addr: str, creds, options=None):
                    created_channels.append((addr, creds, list(options or [])))
                    return SimpleNamespace(addr=addr, creds=creds, options=options)

                @staticmethod
                def insecure_channel(addr: str, options=None):
                    raise AssertionError(
                        f"must use local_channel_credentials + secure_channel, not insecure_channel({addr!r})"
                    )

            class _FakeMetricRequest:
                def __init__(self, metric_name: str):
                    self.metric_name = metric_name

            # Realistic proto3-semantics stubs matching published `tpu_metric_service.proto`:
            # - `AttrValue` uses `int_attr` / `string_attr` (oneof `attr`), NOT `int_value`/`string_value`.
            # - `Gauge` uses `as_int` / `as_double` (oneof `value`).
            # - Unset fields default to 0 / "" / 0.0 and WhichOneof returns None.
            class _ProtoAttrValue:
                __slots__ = ("string_attr", "int_attr", "_which")

                def __init__(self, *, string_attr=None, int_attr=None):
                    self.string_attr = ""
                    self.int_attr = 0
                    self._which = None
                    if string_attr is not None:
                        self.string_attr = string_attr
                        self._which = "string_attr"
                    elif int_attr is not None:
                        self.int_attr = int_attr
                        self._which = "int_attr"

                def WhichOneof(self, name: str):
                    if name == "attr":
                        return self._which
                    raise ValueError(name)

            class _ProtoAttribute:
                __slots__ = ("key", "value")

                def __init__(self, key: str, value: _ProtoAttrValue):
                    self.key = key
                    self.value = value

            class _ProtoGauge:
                __slots__ = ("as_double", "as_int", "_which")

                def __init__(self, *, as_double=None, as_int=None):
                    self.as_double = 0.0
                    self.as_int = 0
                    self._which = None
                    if as_double is not None:
                        self.as_double = float(as_double)
                        self._which = "as_double"
                    elif as_int is not None:
                        self.as_int = int(as_int)
                        self._which = "as_int"

                def WhichOneof(self, name: str):
                    if name == "value":
                        return self._which
                    raise ValueError(name)

            class _ProtoMetric:
                __slots__ = ("attribute", "gauge")

                def __init__(self, attribute: _ProtoAttribute, gauge: _ProtoGauge):
                    self.attribute = attribute
                    self.gauge = gauge

            def _make_item(chip_id, *, as_int=None, as_double=None):
                gauge = _ProtoGauge(as_int=as_int, as_double=as_double)
                if isinstance(chip_id, str):
                    val = _ProtoAttrValue(string_attr=chip_id)
                elif isinstance(chip_id, int):
                    val = _ProtoAttrValue(int_attr=chip_id)
                else:
                    val = _ProtoAttrValue()
                return _ProtoMetric(_ProtoAttribute("device_id", val), gauge)

            class _FakeStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    rpc_calls.append((req.metric_name, float(timeout or 0)))
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [
                            _make_item(0, as_int=16 * 1024**3),
                            _make_item("1", as_int=64 * 1024**3),
                            _make_item(None, as_int=999),  # unset AttrValue must be ignored
                            _make_item(99),  # unset Gauge must be ignored
                        ]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [
                            _make_item(0, as_int=32 * 1024**3),
                            _make_item("1", as_int=96 * 1024**3),
                        ]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [
                            _make_item(0, as_double=72.5),
                            _make_item("1", as_double=88.0),
                        ]
                    elif req.metric_name == "tpu.runtime.hbm.utilization.percent":
                        items = [
                            _make_item(0, as_double=63.2),
                            _make_item("1", as_double=41.0),
                        ]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            fake_pb2 = SimpleNamespace(MetricRequest=_FakeMetricRequest)
            fake_pb2_grpc = SimpleNamespace(RuntimeMetricServiceStub=_FakeStub)

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": fake_pb2_grpc,
                    },
                ),
                mock.patch.object(
                    _te,
                    "_run_tpu_info",
                    side_effect=AssertionError("must not spawn tpu-info subprocess when direct gRPC succeeds"),
                ),
            ):
                snap1 = _te.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
                snap2 = _te.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )

            self.assertTrue(snap1.tpu_info_available)
            self.assertTrue(snap2.tpu_info_available)
            # Persistent local-transport channel reused across scrapes with proxy disabled.
            self.assertEqual(
                created_channels,
                [
                    (
                        "127.0.0.1:8431",
                        ("local_creds", "LOCAL_TCP"),
                        [("grpc.enable_http_proxy", 0)],
                    )
                ],
            )
            # Non-local gRPC targets MUST be rejected before opening a channel.
            with (
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": fake_pb2_grpc,
                    },
                ),
            ):
                ok_remote, err_remote, rows_remote = _te._query_grpc_runtime_metrics(
                    snap1.devices, addr="10.0.0.5:8431"
                )
            self.assertFalse(ok_remote)
            self.assertIn("non-local", err_remote or "")
            self.assertEqual(rows_remote, {})
            self.assertEqual(snap1.devices[0].telemetry_source, "libtpu-grpc")
            self.assertEqual(snap1.devices[0].hbm_used_bytes, 16 * 1024**3)
            self.assertAlmostEqual(snap1.devices[0].duty_cycle_percent or 0.0, 72.5)
            self.assertAlmostEqual(
                snap1.devices[0].hbm_bandwidth_utilization_percent or 0.0, 63.2
            )
            self.assertEqual(snap1.devices[1].device_id, "0x0062")
            self.assertEqual(snap1.devices[1].generation, "v5p")
            self.assertEqual(snap1.devices[1].hbm_used_bytes, 64 * 1024**3)
            self.assertEqual(snap1.devices[1].hbm_total_bytes, 96 * 1024**3)
            self.assertAlmostEqual(
                snap1.devices[1].hbm_bandwidth_utilization_percent or 0.0, 41.0
            )
            self.assertEqual(snap1.devices[2].device_id, "0x0063")
            self.assertEqual(snap1.devices[2].generation, "v5e")
            self.assertEqual(snap1.devices[2].hbm_nominal_bytes, 16 * 1024**3)
            self.assertEqual(snap1.devices[2].hbm_total_bytes, 16 * 1024**3)
            # Unset AttrValue and unset Gauge entries must not appear in runtime_utilization.
            self.assertEqual(
                sorted(rt.runtime_chip for rt in snap1.runtime_utilization),
                ["0", "1"],
            )
            self.assertIn(
                "tpu.runtime.hbm.utilization.percent",
                [name for name, _ in rpc_calls],
            )

            prom = _te.format_prometheus_metrics(snap1)
            self._assert_valid_prometheus_text(prom)
            self.assertIn("freetpu_tpu_hbm_bandwidth_utilization_percent{", prom)
            self.assertIn("freetpu_tpu_runtime_hbm_bandwidth_utilization_percent{", prom)

            # When grpcio is absent, _query_grpc_runtime_metrics fails gracefully without raising.
            with mock.patch.object(_te, "_load_grpc_runtime_modules", return_value=None):
                ok_fb, err_fb, rows_fb = _te._query_grpc_runtime_metrics(snap1.devices)
            self.assertFalse(ok_fb)
            self.assertIn("unavailable", err_fb or "")
            self.assertEqual(rows_fb, {})

            # Crash-prone `tpu-info < 0.14.2` MUST NOT be imported in-process by
            # `_load_grpc_runtime_modules` (preventing import-time exporter termination).
            self.assertFalse(_te._parse_safe_tpu_info_version("0.14.1"))
            self.assertFalse(_te._parse_safe_tpu_info_version("0.14.2rc1"))
            self.assertFalse(_te._parse_safe_tpu_info_version("0.14.2.dev0"))
            self.assertTrue(_te._parse_safe_tpu_info_version("0.14.2"))
            self.assertTrue(_te._parse_safe_tpu_info_version("0.14.2.post1"))
            self.assertTrue(_te._parse_safe_tpu_info_version("0.15.0"))

            attempted_imports: list[str] = []

            def _guarded_import(name: str):
                attempted_imports.append(name)
                if name.startswith("tpu_info"):
                    raise AssertionError(f"must not import {name} in-process on tpu-info < 0.14.2")
                raise ModuleNotFoundError(name)

            with (
                mock.patch.dict(sys.modules, {"grpc": _FakeGrpc}),
                mock.patch("importlib.metadata.version", return_value="0.14.1"),
                mock.patch("importlib.import_module", side_effect=_guarded_import),
            ):
                # Ensure tpu_info.proto.* are absent from sys.modules during this check.
                sys.modules.pop("tpu_info.proto.tpu_metric_service_pb2", None)
                sys.modules.pop("tpu_info.proto.tpu_metric_service_pb2_grpc", None)
                self.assertIsNone(_te._load_grpc_runtime_modules())
            self.assertFalse(any(m.startswith("tpu_info") for m in attempted_imports))
            self.assertIn("tpu_metric_service_pb2", attempted_imports)

            # Ironwood (tpu7x, 0x0076) exposes 2 logical runtime cores per physical chip:
            # HBM is reported per core (0..7) while duty cycle is reported per physical chip
            # (0..3). Verify chip c's duty cycle maps onto cores (2*c, 2*c+1) rather than
            # misassigning chip 1's duty cycle onto core 1 of chip 0.
            iw_root = root / "ironwood_grpc"
            iw_sysfs, iw_dev, iw_proc = _make_mock_pci_tree(
                iw_root,
                devices=[
                    {
                        "bdf": f"0000:00:{4 + (i // 2):02x}.{i % 2}",
                        "vendor": "0x1ae0",
                        "device": "0x0075" if i == 7 else "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                        "chip_id": str(i),
                    }
                    for i in range(8)
                ],
                create_vfio_nodes=[str(i) for i in range(8)],
            )

            class _IronwoodStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 10 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        # 4 physical chips (0..3), NOT 8 logical cores!
                        items = [
                            _make_item(0, as_double=12.5),
                            _make_item(1, as_double=37.5),
                            _make_item(2, as_double=62.5),
                            _make_item(3, as_double=87.5),
                        ]
                    elif req.metric_name == "tpu.runtime.hbm.utilization.percent":
                        items = [_make_item(c, as_double=50.0 + c) for c in range(8)]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            iw_pb2_grpc = SimpleNamespace(RuntimeMetricServiceStub=_IronwoodStub)
            # In default `auto` mode on Ironwood (`tpu7x`), upstream `tpu-info` uses
            # a per-device duty-cycle path (`metrics.get_chip_usage_new`), so 4 duty
            # IDs `{0..3}` alongside 8 HBM IDs `{0..7}` is ambiguous with a partial
            # per-core reply. `auto` MUST keep them unjoined (NOT copy 4 readings
            # across 8 cores) and allow fallback to run!
            iw_auto_fallback_calls: List[str] = []

            def _failing_iw_fallback(*args, **kwargs):
                iw_auto_fallback_calls.append("cli")
                return False, "tpu-info unavailable", {}

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": iw_pb2_grpc,
                    },
                ),
                mock.patch.object(_te, "_run_tpu_info", side_effect=_failing_iw_fallback),
            ):
                iw_auto_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual(len(iw_auto_snap.devices), 8)
            self.assertEqual(iw_auto_fallback_calls, ["cli"])
            self.assertTrue(all(d.duty_cycle_percent is None for d in iw_auto_snap.devices))
            self.assertEqual(
                iw_auto_snap.runtime_unjoined_duty_cycle,
                {"0": 12.5, "1": 37.5, "2": 62.5, "3": 87.5},
            )
            self.assertEqual(
                [d.hbm_used_bytes for d in iw_auto_snap.devices],
                [(c + 1) * 10 * 1024**3 for c in range(8)],
            )

            # With explicit `--grpc-duty-layout=per-chip`, the operator opts in to
            # per-chip expansion (`chip c -> cores 2*c, 2*c+1`) directly over gRPC
            # without spawning fallback subprocesses.
            parsed_layout_args = _te.build_arg_parser().parse_args(
                ["--grpc-duty-layout", "per-chip"]
            )
            self.assertEqual(parsed_layout_args.grpc_duty_layout, "per-chip")
            with self.assertRaises(ValueError):
                _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin=None,
                    grpc_duty_layout="bogus",
                )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": iw_pb2_grpc,
                    },
                ),
                mock.patch.object(
                    _te,
                    "_run_tpu_info",
                    side_effect=AssertionError(
                        "must not spawn fallback when explicit --grpc-duty-layout=per-chip completes over gRPC"
                    ),
                ),
            ):
                iw_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    grpc_duty_layout="per-chip",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual(len(iw_snap.devices), 8)
            self.assertEqual(
                [d.duty_cycle_percent for d in iw_snap.devices],
                [12.5, 12.5, 37.5, 37.5, 62.5, 62.5, 87.5, 87.5],
            )
            self.assertEqual(iw_snap.runtime_unjoined_duty_cycle, {})
            self.assertEqual(
                [d.hbm_used_bytes for d in iw_snap.devices],
                [(c + 1) * 10 * 1024**3 for c in range(8)],
            )
            self.assertTrue(all(d.telemetry_source == "libtpu-grpc" for d in iw_snap.devices))

            # In `tpu_info.metrics` Python fallback on Ironwood (`V7X`), prefer
            # upstream `get_chip_usage_new` over legacy `get_chip_usage` when available.
            v7x_py_calls: List[str] = []

            def _v7x_get_chip_usage_new(chip_type, addr="localhost:8431"):
                v7x_py_calls.append(f"new:{chip_type}:{addr}")
                return [
                    SimpleNamespace(
                        device_id=c,
                        memory_usage=(c + 1) * 1024**3,
                        total_memory=96 * 1024**3,
                        duty_cycle_pct=10.0 + c,
                    )
                    for c in range(8)
                ]

            def _v7x_get_chip_usage_old(*args, **kwargs):
                raise AssertionError("must prefer get_chip_usage_new for V7X when available")

            with mock.patch.dict(
                sys.modules,
                {
                    "tpu_info.device": SimpleNamespace(TpuChip=SimpleNamespace(V7X="V7X")),
                    "tpu_info.metrics": SimpleNamespace(
                        get_chip_usage=_v7x_get_chip_usage_old,
                        get_chip_usage_new=_v7x_get_chip_usage_new,
                    ),
                },
            ):
                py_ok, py_err, py_rows = _te._query_tpu_info_python_metrics(
                    iw_snap.devices,
                    addr="127.0.0.1:8431",
                    timeout_s=1.0,
                )
            self.assertTrue(py_ok, py_err)
            self.assertEqual(v7x_py_calls, ["new:V7X:127.0.0.1:8431"])
            self.assertEqual(len(py_rows), 8)

            # Ironwood per-core duty-cycle response (8 entries 0..7 matching 8 HBM cores)
            # MUST preserve 1:1 per-core readings and NEVER double/fabricate rows 8..15!
            class _IronwoodPerCoreStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 8 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [_make_item(c, as_double=11.0 * (c + 1)) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.hbm.utilization.percent":
                        items = [_make_item(c, as_double=40.0 + c) for c in range(8)]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            iw_percore_pb2_grpc = SimpleNamespace(RuntimeMetricServiceStub=_IronwoodPerCoreStub)
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": iw_percore_pb2_grpc,
                    },
                ),
            ):
                iw_pc_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual(len(iw_pc_snap.runtime_utilization), 8)
            self.assertEqual(
                [d.duty_cycle_percent for d in iw_pc_snap.devices],
                [11.0 * (c + 1) for c in range(8)],
            )

            # Ironwood per-core duty-cycle response with 8 raw IDs (0..7) where only
            # cores 0..3 have valid readings and cores 4..7 have unset/NaN gauges MUST
            # stay per-core 1:1 (never doubling 0..3 into 8 cores or fabricating 4..7!).
            class _IronwoodPartialValidPerCoreStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 4 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [
                            _make_item(c, as_double=11.0 * (c + 1)) for c in range(4)
                        ] + [
                            _make_item(4),  # unset gauge
                            _make_item(5, as_double=float("nan")),
                            _make_item(6, as_double=-1.0),
                            _make_item(7),  # unset gauge
                        ]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodPartialValidPerCoreStub
                        ),
                    },
                ),
            ):
                iw_partial_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual(
                [d.duty_cycle_percent for d in iw_partial_snap.devices],
                [11.0, 22.0, 33.0, 44.0, None, None, None, None],
            )

            # Ambiguous partial response (only raw IDs {0, 1} on an 8-core / 4-chip host)
            # MUST omit ambiguous duty-cycle readings instead of guessing chip vs core.
            class _IronwoodAmbiguousDutyStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 4 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in range(8)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [_make_item(0, as_double=50.0), _make_item(1, as_double=60.0)]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodAmbiguousDutyStub
                        ),
                    },
                ),
            ):
                iw_ambig_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual([d.duty_cycle_percent for d in iw_ambig_snap.devices], [None] * 8)
            self.assertEqual(
                [d.hbm_used_bytes for d in iw_ambig_snap.devices],
                [(c + 1) * 4 * 1024**3 for c in range(8)],
            )

            # Ironwood sliced partition exposing duty-cycle IDs `{2, 3}` and HBM core
            # IDs `{4, 5, 6, 7}`: without an explicit layout flag, never guess per-chip
            # expansion across cores; keep `{2, 3}` in `runtime_unjoined_duty_cycle`
            # while preserving HBM cores `4..7` cleanly!
            class _IronwoodSlicedChips23Stub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 5 * 1024**3) for c in (4, 5, 6, 7)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in (4, 5, 6, 7)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [_make_item(2, as_double=42.5), _make_item(3, as_double=84.0)]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodSlicedChips23Stub
                        ),
                    },
                ),
            ):
                iw_slice_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual([d.duty_cycle_percent for d in iw_slice_snap.devices], [None] * 8)
            self.assertEqual(
                iw_slice_snap.runtime_unjoined_duty_cycle,
                {"2": 42.5, "3": 84.0},
            )
            self.assertEqual(
                sorted(rt.runtime_chip for rt in iw_slice_snap.runtime_utilization),
                ["4", "5", "6", "7"],
            )

            # Partial HBM response (`0..5`, missing cores `6..7`) alongside physical-chip
            # duty cycle (`0..3`) MUST be classified as ambiguous and omit duty cycle
            # rather than misclassifying `{0..3}` as per-core and assigning chip 1 to core 1!
            class _IronwoodPartialHbmDutyStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 4 * 1024**3) for c in range(6)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in range(6)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [_make_item(c, as_double=20.0 * (c + 1)) for c in range(4)]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodPartialHbmDutyStub
                        ),
                    },
                ),
            ):
                iw_partial_hbm_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual([d.duty_cycle_percent for d in iw_partial_hbm_snap.devices], [None] * 8)
            self.assertEqual(
                [d.hbm_used_bytes for d in iw_partial_hbm_snap.devices[:6]],
                [(c + 1) * 4 * 1024**3 for c in range(6)],
            )

            # Partial HBM response `{0, 1, 2, 3}` alongside EQUAL duty-cycle IDs `{0, 1, 2, 3}`
            # on an 8-core / 4-chip Ironwood host is STILL ambiguous (could be physical chips
            # 0..3 rather than cores 0..3!) and MUST omit duty cycle rather than assigning
            # chip 1's duty reading to core 1 of chip 0!
            class _IronwoodEqualPartialHbmDutyStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 4 * 1024**3) for c in range(4)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in range(4)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [_make_item(c, as_double=20.0 * (c + 1)) for c in range(4)]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodEqualPartialHbmDutyStub
                        ),
                    },
                ),
            ):
                iw_eq_partial_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual([d.duty_cycle_percent for d in iw_eq_partial_snap.devices], [None] * 8)
            self.assertEqual(
                [d.hbm_used_bytes for d in iw_eq_partial_snap.devices[:4]],
                [(c + 1) * 4 * 1024**3 for c in range(4)],
            )

            # Conclusive per-core duty-cycle response (8 distinct IDs 0..7 on a 4-chip /
            # 8-core Ironwood host) MUST keep all 8 duty readings even when HBM RPCs
            # return only 6 partial cores (0..5)!
            class _IronwoodFullPerCoreDutyPartialHbmStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [_make_item(c, as_int=(c + 1) * 4 * 1024**3) for c in range(6)]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [_make_item(c, as_int=96 * 1024**3) for c in range(6)]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [_make_item(c, as_double=10.0 * (c + 1)) for c in range(8)]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodFullPerCoreDutyPartialHbmStub
                        ),
                    },
                ),
            ):
                iw_full_duty_partial_hbm_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual(
                [d.duty_cycle_percent for d in iw_full_duty_partial_hbm_snap.devices],
                [10.0 * (c + 1) for c in range(8)],
            )
            self.assertEqual(
                [d.hbm_used_bytes for d in iw_full_duty_partial_hbm_snap.devices[:6]],
                [(c + 1) * 4 * 1024**3 for c in range(6)],
            )

            # Sidecar WITHOUT PCI inventory (`devices=[]`): MUST NOT default
            # `devices_per_chip=1` and misjoin 4 physical-chip Ironwood duty readings
            # `{0..3}` onto 8 logical-core HBM rows `{0..7}` (putting chip 1's duty
            # onto core 1 of chip 0)! HBM rows stay intact and duty is kept unjoined.
            self.assertIsNone(_te._infer_devices_per_chip([]))
            with (
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodStub
                        ),
                    },
                ),
            ):
                no_pci_ok, no_pci_err, no_pci_rows = _te._query_grpc_runtime_metrics(
                    [], addr="127.0.0.1:8431"
                )
            self.assertTrue(no_pci_ok, no_pci_err)
            self.assertEqual(sorted(no_pci_rows.keys(), key=int), [str(c) for c in range(8)])
            for core_idx in range(8):
                self.assertEqual(
                    no_pci_rows[str(core_idx)].hbm_used_bytes, (core_idx + 1) * 10 * 1024**3
                )
                self.assertIsNone(no_pci_rows[str(core_idx)].duty_cycle_percent)
            # Raw duty-cycle readings MUST still be preserved in the separate unjoined
            # runtime series keyed by reported ID so sidecars without PCI inventory or
            # `tpu-info` fallback never lose duty-cycle monitoring.
            empty_pci_root = root / "sidecar_no_pci"
            (empty_pci_root / "sys" / "bus" / "pci" / "devices").mkdir(parents=True)
            (empty_pci_root / "dev").mkdir(parents=True)
            (empty_pci_root / "proc").mkdir(parents=True)
            with (
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodStub
                        ),
                    },
                ),
            ):
                no_pci_snap = _te.collect_snapshot(
                    sysfs_root=empty_pci_root / "sys",
                    dev_root=empty_pci_root / "dev",
                    proc_root=empty_pci_root / "proc",
                    tpu_info_bin="tpu-info",
                    skip_pid_scan=True,
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=empty_pci_root / "no-shm",
                )
            self.assertTrue(no_pci_snap.tpu_info_available)
            self.assertEqual(len(no_pci_snap.runtime_utilization), 8)
            self.assertTrue(all(rt.duty_cycle_percent is None for rt in no_pci_snap.runtime_utilization))
            self.assertEqual(
                no_pci_snap.runtime_unjoined_duty_cycle,
                {"0": 12.5, "1": 37.5, "2": 62.5, "3": 87.5},
            )
            no_pci_prom = _te.format_prometheus_metrics(no_pci_snap)
            self._assert_valid_prometheus_text(no_pci_prom)
            self.assertNotIn("freetpu_tpu_runtime_duty_cycle_percent{", no_pci_prom)
            self.assertIn(
                'freetpu_tpu_runtime_unjoined_duty_cycle_percent{runtime_device_id="0"} 12.5',
                no_pci_prom,
            )
            self.assertIn(
                'freetpu_tpu_runtime_unjoined_duty_cycle_percent{runtime_device_id="3"} 87.5',
                no_pci_prom,
            )

            # Ironwood (8 PCI functions) with duty-cycle IDs `{0..3}` and NO HBM IDs:
            # MUST NOT assume physical-chip granularity and expand across all 8 cores
            # (a 4-core sliced runtime also reports per-core IDs 0..3). Keep unjoined!
            class _IronwoodDutyOnlyNoHbmStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [
                            _make_item(0, as_double=12.5),
                            _make_item(1, as_double=37.5),
                            _make_item(2, as_double=62.5),
                            _make_item(3, as_double=87.5),
                        ]
                        return SimpleNamespace(metric=SimpleNamespace(metrics=items))
                    return SimpleNamespace(metric=SimpleNamespace(metrics=[]))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodDutyOnlyNoHbmStub
                        ),
                    },
                ),
            ):
                iw_no_hbm_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertTrue(all(d.duty_cycle_percent is None for d in iw_no_hbm_snap.devices))
            self.assertEqual(
                iw_no_hbm_snap.runtime_unjoined_duty_cycle,
                {"0": 12.5, "1": 37.5, "2": 62.5, "3": 87.5},
            )

            # Offset Ironwood physical-chip duty IDs `{4, 5}` without HBM IDs MUST ALSO
            # stay unjoined rather than being treated as logical cores 4 & 5 just because
            # `4 >= expected_chip_count (4)`!
            class _IronwoodOffsetDutyNoHbmStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [
                            _make_item(4, as_double=22.5),
                            _make_item(5, as_double=77.5),
                        ]
                        return SimpleNamespace(metric=SimpleNamespace(metrics=items))
                    return SimpleNamespace(metric=SimpleNamespace(metrics=[]))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodOffsetDutyNoHbmStub
                        ),
                    },
                ),
            ):
                iw_offset_no_hbm_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertTrue(all(d.duty_cycle_percent is None for d in iw_offset_no_hbm_snap.devices))
            self.assertEqual(
                iw_offset_no_hbm_snap.runtime_unjoined_duty_cycle,
                {"4": 22.5, "5": 77.5},
            )

            # Incomplete 6-function Ironwood PCI inventory (2 intact pairs + 2 orphaned
            # functions spanning 4 physical chips!): MUST NOT divide `6 // 2 = 3` and
            # conclude 4 per-chip duty IDs `{0..3}` exceed chip count! Keep unjoined!
            iw6_root = root / "ironwood_incomplete_6fn"
            iw6_bdfs = [
                "0000:00:04.0",
                "0000:00:04.1",
                "0000:00:05.0",
                "0000:00:05.1",
                "0000:00:06.0",  # orphan from chip 2
                "0000:00:07.0",  # orphan from chip 3
            ]
            iw6_sysfs, iw6_dev, iw6_proc = _make_mock_pci_tree(
                iw6_root,
                devices=[
                    {
                        "bdf": bdf,
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                        "chip_id": str(i),
                    }
                    for i, bdf in enumerate(iw6_bdfs)
                ],
                create_vfio_nodes=[str(i) for i in range(6)],
            )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodPartialHbmDutyStub
                        ),
                    },
                ),
            ):
                iw6_snap = _te.collect_snapshot(
                    sysfs_root=iw6_sysfs,
                    dev_root=iw6_dev,
                    proc_root=iw6_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw6_root / "no-shm",
                )
            self.assertIsNone(_te._verified_dual_function_chip_count(iw6_snap.devices, 2))
            self.assertEqual([d.duty_cycle_percent for d in iw6_snap.devices], [None] * 6)
            self.assertEqual(
                iw6_snap.runtime_unjoined_duty_cycle,
                {"0": 20.0, "1": 40.0, "2": 60.0, "3": 80.0},
            )

            # Partial fallback after ambiguous gRPC duty-cycle: if CLI supplies joined
            # duty for ONLY 1 device ("0"), MUST retain the raw unjoined series so the
            # remaining 3 readings do not disappear; full joined coverage clears it.
            def _partial_1row_cli(*args, **kwargs):
                rows = _te.parse_tpu_info_output(
                    "TPU Runtime Utilization\n"
                    "| Chip | HBM Usage (GiB)      | Duty cycle |\n"
                    "|------|----------------------|------------|\n"
                    "| 0    | 10.00 GiB / 96.00 GiB | 12.50%    |\n"
                )
                return True, None, rows

            def _partial_4row_cli(*args, **kwargs):
                # Logical cores 0..3 cover only physical chips 0 and 1 on a
                # dual-function host (`devices_per_chip=2`), NOT chips 2 and 3!
                lines = [
                    "TPU Runtime Utilization",
                    "| Chip | HBM Usage (GiB)       | Duty cycle |",
                    "|------|-----------------------|------------|",
                ] + [
                    f"| {c}    | 10.00 GiB / 96.00 GiB | {12.5 + (c // 2) * 25.0:.2f}%     |"
                    for c in range(4)
                ]
                return True, None, _te.parse_tpu_info_output("\n".join(lines) + "\n")

            def _full_8row_cli(*args, **kwargs):
                lines = [
                    "TPU Runtime Utilization",
                    "| Chip | HBM Usage (GiB)       | Duty cycle |",
                    "|------|-----------------------|------------|",
                ] + [
                    f"| {c}    | 10.00 GiB / 96.00 GiB | {12.5 + (c // 2) * 25.0:.2f}%     |"
                    for c in range(8)
                ]
                return True, None, _te.parse_tpu_info_output("\n".join(lines) + "\n")

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodDutyOnlyNoHbmStub
                        ),
                    },
                ),
                mock.patch.object(_te, "_run_tpu_info", side_effect=_partial_1row_cli),
            ):
                iw_partial_fb_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual(
                iw_partial_fb_snap.runtime_unjoined_duty_cycle,
                {"0": 12.5, "1": 37.5, "2": 62.5, "3": 87.5},
            )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodDutyOnlyNoHbmStub
                        ),
                    },
                ),
                mock.patch.object(_te, "_run_tpu_info", side_effect=_partial_4row_cli),
            ):
                iw_partial4_fb_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            # Logical cores 0..3 cover raw IDs 0 (cores 0,1) and 1 (cores 2,3),
            # but NOT raw IDs 2 (cores 4,5) or 3 (cores 6,7) — MUST retain 2 & 3!
            self.assertEqual(
                iw_partial4_fb_snap.runtime_unjoined_duty_cycle,
                {"2": 62.5, "3": 87.5},
            )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_IronwoodDutyOnlyNoHbmStub
                        ),
                    },
                ),
                mock.patch.object(_te, "_run_tpu_info", side_effect=_full_8row_cli),
            ):
                iw_full_fb_snap = _te.collect_snapshot(
                    sysfs_root=iw_sysfs,
                    dev_root=iw_dev,
                    proc_root=iw_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_root / "no-shm",
                )
            self.assertEqual(iw_full_fb_snap.runtime_unjoined_duty_cycle, {})

            # Malformed gRPC device IDs (Unicode superscripts like "²", huge decimal
            # strings >4300 digits, negative integers, non-numeric strings) MUST be
            # skipped safely at decode time without raising ValueError or aborting scrapes.
            for bad_id in ("²", "9" * 5000, "-1", "chip-bad", 70000, -5, True):
                self.assertIsNone(_te._parse_runtime_device_id(bad_id))

            class _MalformedIdStub:
                def __init__(self, channel):
                    self.channel = channel

                def GetRuntimeMetric(self, req, timeout=None):
                    del timeout
                    if req.metric_name == "tpu.runtime.hbm.memory.usage.bytes":
                        items = [
                            _make_item("²", as_int=1024),
                            _make_item("9" * 5000, as_int=2048),
                            _make_item(0, as_int=16 * 1024**3),
                        ]
                    elif req.metric_name == "tpu.runtime.hbm.memory.total.bytes":
                        items = [
                            _make_item("²", as_int=4096),
                            _make_item(0, as_int=32 * 1024**3),
                        ]
                    elif req.metric_name == "tpu.runtime.tensorcore.dutycycle.percent":
                        items = [
                            _make_item("²", as_double=99.0),
                            _make_item("9" * 5000, as_double=88.0),
                            _make_item(0, as_double=44.5),
                        ]
                    else:
                        items = []
                    return SimpleNamespace(metric=SimpleNamespace(metrics=items))

            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": SimpleNamespace(
                            RuntimeMetricServiceStub=_MalformedIdStub
                        ),
                    },
                ),
            ):
                malformed_snap = _te.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertTrue(malformed_snap.tpu_info_available)
            self.assertEqual([rt.runtime_chip for rt in malformed_snap.runtime_utilization], ["0"])
            self.assertEqual(malformed_snap.devices[0].hbm_used_bytes, 16 * 1024**3)
            self.assertAlmostEqual(malformed_snap.devices[0].duty_cycle_percent or 0.0, 44.5)
            self._assert_valid_prometheus_text(_te.format_prometheus_metrics(malformed_snap))

            self.assertEqual(iw_snap.devices[0].device_id, "0x0076")
            self.assertEqual(iw_snap.devices[7].device_id, "0x0075")
            self.assertEqual(iw_snap.devices[7].generation, "tpu7x")
            self.assertEqual(iw_snap.devices[7].hbm_nominal_bytes, 96 * 1024**3)

            # Ironwood host where sibling PCI functions share a physical sysfs `chip_id`
            # (`0, 0, 1, 1, 2, 2, 3, 3`): without `--chip-id-map`, shared IDs stay
            # safely ambiguous/unmapped; WITH explicit per-BDF `--chip-id-map` core IDs
            # (`0..7`), explicit mappings override the shared sysfs ID before duplicate
            # detection so every core's HBM and duty cycle attaches cleanly!
            iw_shared_root = root / "ironwood_shared_sysfs_chip_id"
            iw_shared_sysfs, iw_shared_dev, iw_shared_proc = _make_mock_pci_tree(
                iw_shared_root,
                devices=[
                    {
                        "bdf": f"0000:00:{4 + (i // 2):02x}.{i % 2}",
                        "vendor": "0x1ae0",
                        "device": "0x0076",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                        "chip_id": str(i // 2),  # shared physical chip_id per sibling pair!
                    }
                    for i in range(8)
                ],
                create_vfio_nodes=[str(i) for i in range(8)],
            )
            iw_explicit_map = {
                f"0000:00:{4 + (i // 2):02x}.{i % 2}": str(i) for i in range(8)
            }
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": iw_percore_pb2_grpc,
                    },
                ),
            ):
                iw_shared_unmapped = _te.collect_snapshot(
                    sysfs_root=iw_shared_sysfs,
                    dev_root=iw_shared_dev,
                    proc_root=iw_shared_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_shared_root / "no-shm",
                )
                iw_shared_mapped = _te.collect_snapshot(
                    sysfs_root=iw_shared_sysfs,
                    dev_root=iw_shared_dev,
                    proc_root=iw_shared_proc,
                    tpu_info_bin="tpu-info",
                    chip_id_map=iw_explicit_map,
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=iw_shared_root / "no-shm",
                )
            # Without --chip-id-map, shared sysfs chip_id stays safely unmapped.
            self.assertTrue(all(d.hbm_used_bytes is None for d in iw_shared_unmapped.devices))
            # With explicit per-BDF --chip-id-map, all 8 cores attach to their exact PCI BDFs!
            self.assertEqual(
                [d.runtime_chip_id for d in iw_shared_mapped.devices],
                [str(i) for i in range(8)],
            )
            self.assertEqual(
                [d.hbm_used_bytes for d in iw_shared_mapped.devices],
                [(i + 1) * 8 * 1024**3 for i in range(8)],
            )
            self.assertEqual(
                [d.duty_cycle_percent for d in iw_shared_mapped.devices],
                [11.0 * (i + 1) for i in range(8)],
            )

            # Custom `--tpu-info-bin` specified by bare name (without `/`) MUST invoke
            # that configured executable via PATH instead of taking the default gRPC shortcut.
            custom_rows = _te.parse_tpu_info_output(
                "TPU Runtime Utilization\n"
                "| Chip | HBM Usage (GiB)      | Duty cycle |\n"
                "|------|----------------------|------------|\n"
                "| 0    | 5.00 GiB / 32.00 GiB | 33.00%     |\n"
            )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": fake_pb2_grpc,
                    },
                ),
                mock.patch.object(
                    _te, "_run_tpu_info", return_value=(True, None, custom_rows)
                ) as mock_custom_cli,
            ):
                custom_snap = _te.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="my-tpu-info-wrapper",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=root / "no-shm",
                )
            self.assertEqual(mock_custom_cli.call_count, 1)
            self.assertEqual(mock_custom_cli.call_args.kwargs["tpu_info_bin"], "my-tpu-info-wrapper")
            self.assertEqual(custom_snap.devices[0].hbm_used_bytes, 5 * 1024**3)
            self.assertAlmostEqual(custom_snap.devices[0].duty_cycle_percent or 0.0, 33.0)

            # Ironwood 4-chip VM (8 exposed PCI functions across 0x0076/0x0075) without
            # runtime totals MUST report 96 GiB per chiplet PCI row = 768 GiB aggregate
            # (not 8 * 192 GiB = 1,536 GiB).
            iw_inv_root = root / "ironwood_inventory"
            iw_inv_sysfs, iw_inv_dev, iw_inv_proc = _make_mock_pci_tree(
                iw_inv_root,
                devices=[
                    {
                        "bdf": f"0000:00:{4 + i:02x}.0",
                        "vendor": "0x1ae0",
                        "device": "0x0076" if i < 6 else "0x0075",
                        "driver": "vfio-pci",
                        "iommu_group": str(i),
                    }
                    for i in range(8)
                ],
                create_vfio_nodes=[str(i) for i in range(8)],
            )
            with _mock_vfio_char_lstat():
                iw_inv_snap = _te.collect_snapshot(
                    sysfs_root=iw_inv_sysfs,
                    dev_root=iw_inv_dev,
                    proc_root=iw_inv_proc,
                    tpu_info_bin=None,
                    shm_root=iw_inv_root / "no-shm",
                )
            self.assertEqual(len(iw_inv_snap.devices), 8)
            self.assertEqual(
                [d.hbm_total_bytes for d in iw_inv_snap.devices],
                [96 * 1024**3] * 8,
            )
            self.assertEqual(
                sum(d.hbm_total_bytes or 0 for d in iw_inv_snap.devices),
                768 * 1024**3,
            )

            # Multi-chip host WITHOUT sysfs `chip_id` or `--chip-id-map`:
            # 1) Direct gRPC MUST NOT spawn a redundant `tpu-info` CLI subprocess solely
            #    to guess VFIO mappings, and MUST keep runtime ordinals unmapped.
            # 2) Even when `tpu-info` CLI runs and prints a `TPU Process Monitor` table
            #    with guessed `/dev/vfio/<runtime_id>` paths, runtime metrics MUST stay
            #    unmapped (`pci_bdf="unmapped"`) unless sysfs `chip_id` or `--chip-id-map`
            #    authoritatively establishes PCI identity!
            nomap_root = root / "authoritative_mapping_only"
            nomap_sysfs, nomap_dev, nomap_proc = _make_mock_pci_tree(
                nomap_root,
                devices=[
                    {
                        "bdf": "0000:00:04.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                    },
                    {
                        "bdf": "0000:00:05.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "2",
                    },
                ],
                create_vfio_nodes=["0", "2"],
            )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.dict(_te._GRPC_CHANNELS, {}, clear=True),
                mock.patch.dict(
                    sys.modules,
                    {
                        "grpc": _FakeGrpc,
                        "tpu_info": SimpleNamespace(),
                        "tpu_info.proto": SimpleNamespace(),
                        "tpu_info.proto.tpu_metric_service_pb2": fake_pb2,
                        "tpu_info.proto.tpu_metric_service_pb2_grpc": fake_pb2_grpc,
                    },
                ),
                mock.patch.object(
                    _te,
                    "_run_tpu_info",
                    side_effect=AssertionError("must not spawn CLI when direct gRPC succeeded"),
                ),
            ):
                unmapped_grpc_snap = _te.collect_snapshot(
                    sysfs_root=nomap_sysfs,
                    dev_root=nomap_dev,
                    proc_root=nomap_proc,
                    tpu_info_bin="tpu-info",
                    runtime_metrics_addr="127.0.0.1:8431",
                    shm_root=nomap_root / "no-shm",
                )
            self.assertTrue(unmapped_grpc_snap.tpu_info_available)
            self.assertTrue(all(d.hbm_used_bytes is None for d in unmapped_grpc_snap.devices))
            self.assertTrue(all(d.duty_cycle_percent is None for d in unmapped_grpc_snap.devices))
            self.assertEqual(len(unmapped_grpc_snap.runtime_utilization), 2)
            self.assertTrue(all(rt.mapped_pci_bdf is None for rt in unmapped_grpc_snap.runtime_utilization))

            # Even when `tpu-info` CLI Process Monitor prints guessed `/dev/vfio/0` and
            # `/dev/vfio/2` alongside Device IDs 0 and 1, do NOT attach runtime metrics
            # to PCI devices without authoritative sysfs `chip_id` or `--chip-id-map`.
            cli_pm_script = nomap_root / "mock-tpu-info-pm.sh"
            cli_pm_script.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Device Status\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| Device ID | Memory Usage (Used/Total) | Memory Bar Gauge   | HBM Bandwidth Util | TensorCore Duty Cycle |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "| 0         | 15.00 GiB / 32.00 GiB     | [|||||.....] 46.9% | [||........] 20.0% | [|||||||...] 70.0%    |\n"
                "| 1         | 20.00 GiB / 32.00 GiB     | [||||||....] 62.5% | [||||......] 40.0% | [||||||||..] 80.0%    |\n"
                "+-----------+---------------------------+--------------------+--------------------+-----------------------+\n"
                "\n"
                "TPU Process Monitor\n"
                "+-----------+-------------------------+------+--------------+-------------------+\n"
                "| Device ID | Device Path / VFIO Path | PID  | Process Name | Memory Allocation |\n"
                "+-----------+-------------------------+------+--------------+-------------------+\n"
                "| 0         | /dev/vfio/0             | 4242 | python3      | 15.00 GiB         |\n"
                "| 1         | /dev/vfio/2             | 4242 | python3      | 20.00 GiB         |\n"
                "+-----------+-------------------------+------+--------------+-------------------+\n"
                "EOF\n",
                encoding="utf-8",
            )
            cli_pm_script.chmod(0o755)
            with _mock_vfio_char_lstat():
                pm_unmapped_snap = _te.collect_snapshot(
                    sysfs_root=nomap_sysfs,
                    dev_root=nomap_dev,
                    proc_root=nomap_proc,
                    tpu_info_bin=str(cli_pm_script),
                    shm_root=nomap_root / "no-shm",
                )
            self.assertTrue(pm_unmapped_snap.tpu_info_available)
            self.assertTrue(all(d.hbm_used_bytes is None for d in pm_unmapped_snap.devices))
            self.assertTrue(all(d.duty_cycle_percent is None for d in pm_unmapped_snap.devices))
            self.assertEqual(len(pm_unmapped_snap.runtime_utilization), 2)
            self.assertTrue(all(rt.mapped_pci_bdf is None for rt in pm_unmapped_snap.runtime_utilization))

    def test_shm_seqlock_mmap_reader_including_torn_read_retry(self) -> None:
        import struct
        import time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shm_root = root / "shm"
            shm_root.mkdir(parents=True, exist_ok=True)
            os.chmod(shm_root, 0o750)
            self.assertEqual(tpu_exporter.DEFAULT_SHM_ROOT, Path("/run/freetpu-metrics"))
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                    }
                ],
                create_vfio_nodes=["3"],
            )

            now_ns = time.time_ns()
            shm_file = shm_root / "freetpu_metrics_0000:00:0b.0"
            payload = struct.pack(
                "<IIQQQQQQ",
                tpu_exporter.FREETPU_SHM_MAGIC,
                tpu_exporter.FREETPU_SHM_VERSION,
                4,  # even seq
                32 * 1024**3,  # hbm_allocated_bytes
                18 * 1024**3,  # hbm_used_bytes
                987654321,  # mxu_active_cycles
                104857600,  # dma_bytes_transferred
                now_ns,  # fresh publisher heartbeat timestamp_ns
            )
            shm_file.write_bytes(payload)
            os.chmod(shm_file, 0o600)

            with _mock_vfio_char_lstat():
                snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                    shm_root=shm_root,
                )
            d = snap.devices[0]
            self.assertEqual(d.telemetry_source, "libopentpu-shm")
            self.assertEqual(d.hbm_used_bytes, 18 * 1024**3)
            self.assertEqual(d.hbm_total_bytes, 32 * 1024**3)
            self.assertEqual(d.mxu_active_cycles, 987654321)
            self.assertEqual(d.dma_bytes_transferred, 104857600)
            prom = tpu_exporter.format_prometheus_metrics(snap)
            self._assert_valid_prometheus_text(prom)
            self.assertIn("freetpu_tpu_shm_hbm_allocated_bytes{", prom)
            self.assertIn("freetpu_tpu_mxu_active_cycles_total{", prom)
            self.assertIn("freetpu_tpu_dma_bytes_transferred_total{", prom)

            # When an inventory-only `tpu-info` row or duty-cycle-only runtime sample sets
            # `telemetry_source` WITHOUT providing a measured runtime `hbm_total_bytes`,
            # SHM HBM usage (29 GiB of a 30 GiB pool on a nominal 32 GiB v6e chip) MUST
            # also adopt SHM pool capacity (30 GiB -> 96.7%, triggering TpuHbmOOMWarning).
            shm_file.write_bytes(
                struct.pack(
                    "<IIQQQQQQ",
                    tpu_exporter.FREETPU_SHM_MAGIC,
                    tpu_exporter.FREETPU_SHM_VERSION,
                    6,
                    30 * 1024**3,  # hbm_allocated_bytes (30 GiB pool)
                    29 * 1024**3,  # hbm_used_bytes (29 GiB -> 96.67%)
                    987654321,
                    104857600,
                    now_ns,
                )
            )
            inv_cli = root / "mock-inventory-only-tpu-info.sh"
            inv_cli.write_text(
                "#!/bin/sh\n"
                "cat <<'EOF'\n"
                "TPU Chips\n"
                "| Chip        | Type         | Devices | PID  |\n"
                "|-------------|--------------|---------|------|\n"
                "| /dev/vfio/3 | TPU v6e chip | 1       | None |\n"
                "TPU Runtime Utilization\n"
                "| Chip | HBM Usage (GiB) | Duty cycle |\n"
                "|------|-----------------|------------|\n"
                "| 0    | N/A             | 45.00%     |\n"
                "EOF\n",
                encoding="utf-8",
            )
            inv_cli.chmod(0o755)
            with _mock_vfio_char_lstat():
                inv_shm_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=str(inv_cli),
                    chip_id_map={"0000:00:0b.0": "0"},
                    shm_root=shm_root,
                )
            d_inv = inv_shm_snap.devices[0]
            self.assertEqual(d_inv.telemetry_source, "tpu-info")
            self.assertAlmostEqual(d_inv.duty_cycle_percent or 0.0, 45.0)
            self.assertEqual(d_inv.hbm_used_bytes, 29 * 1024**3)
            self.assertEqual(d_inv.hbm_total_bytes, 30 * 1024**3)
            ratio = d_inv.hbm_used_bytes / d_inv.hbm_total_bytes
            self.assertGreater(ratio, 0.95)

            # Even when runtime supplies a 32 GiB HBM total WITHOUT HBM usage,
            # SHM usage (29 GiB of a 30 GiB SHM pool) MUST pair with SHM pool
            # capacity (30 GiB -> 96.7%), not the runtime's 32 GiB total (90.6%).
            total_only_row = tpu_exporter._TpuInfoRow(
                chip_key="0",
                hbm_used_bytes=None,
                hbm_total_bytes=32 * 1024**3,
                duty_cycle_percent=50.0,
                in_util_table=True,
            )
            with (
                _mock_vfio_char_lstat(),
                mock.patch.object(
                    tpu_exporter,
                    "_query_grpc_runtime_metrics",
                    return_value=(True, None, {"0": total_only_row}),
                ),
            ):
                total_only_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin="tpu-info",
                    chip_id_map={"0000:00:0b.0": "0"},
                    shm_root=shm_root,
                )
            d_tot = total_only_snap.devices[0]
            self.assertEqual(d_tot.hbm_used_bytes, 29 * 1024**3)
            self.assertEqual(d_tot.hbm_total_bytes, 30 * 1024**3)
            self.assertGreater(d_tot.hbm_used_bytes / d_tot.hbm_total_bytes, 0.95)

            # Restore original payload for subsequent SHM security/torn-read tests.
            shm_file.write_bytes(payload)
            os.chmod(shm_file, 0o600)

            # Security checks: reject untrusted owner UID, group/world-writable modes,
            # symlinks, and world-writable directories without sticky bit.
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics(
                    "0000:00:0b.0",
                    shm_root=shm_root,
                    trusted_uids=[os.geteuid() + 1],
                )
            )
            os.chmod(shm_file, 0o660)
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics("0000:00:0b.0", shm_root=shm_root)
            )
            os.chmod(shm_file, 0o666)
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics("0000:00:0b.0", shm_root=shm_root)
            )
            os.chmod(shm_file, 0o600)

            sym_shm = shm_root / "freetpu_metrics_0000:00:0c.0"
            os.symlink(shm_file, sym_shm)
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics("0000:00:0c.0", shm_root=shm_root)
            )

            # Any group-writable or world-writable directory (including sticky 0o1777 like
            # shared `/dev/shm`), symlinked SHM directory, or untrusted-UID directory MUST
            # be rejected so unrelated local users cannot squat on predictable filenames.
            unsafe_dir = root / "unsafe_shm_dir"
            unsafe_dir.mkdir(parents=True, exist_ok=True)
            (unsafe_dir / "freetpu_metrics_0000:00:0b.0").write_bytes(payload)
            os.chmod(unsafe_dir / "freetpu_metrics_0000:00:0b.0", 0o600)
            try:
                for bad_mode in (0o777, 0o1777, 0o770):
                    os.chmod(unsafe_dir, bad_mode)
                    self.assertIsNone(
                        tpu_exporter.read_shm_seqlock_metrics("0000:00:0b.0", shm_root=unsafe_dir),
                        f"Expected mode {oct(bad_mode)} SHM dir to be rejected",
                    )
            finally:
                os.chmod(unsafe_dir, 0o700)

            sym_dir = root / "shm_dir_symlink"
            os.symlink(shm_root, sym_dir)
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics("0000:00:0b.0", shm_root=sym_dir)
            )
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics(
                    "0000:00:0b.0",
                    shm_root=shm_root,
                    trusted_uids=[os.geteuid() + 1],
                )
            )

            # Concurrent truncation during pread fails closed cleanly (no SIGBUS).
            orig_pread = os.pread
            pread_calls = {"n": 0}

            def _truncating_pread(fd: int, n: int, offset: int) -> bytes:
                pread_calls["n"] += 1
                if pread_calls["n"] == 2:
                    return b""  # simulated truncation between header and payload
                return orig_pread(fd, n, offset)

            with mock.patch.object(os, "pread", side_effect=_truncating_pread):
                self.assertIsNone(
                    tpu_exporter.read_shm_seqlock_metrics("0000:00:0b.0", shm_root=shm_root)
                )

            # Torn read on attempt 1 (seq1=4, seq2=6) must retry and succeed on attempt 2 (seq=6).
            seq_reads = iter([(6,), (4,)])
            orig_seq_struct = tpu_exporter._SHM_SEQ_STRUCT

            class _SeqProxy:
                size = orig_seq_struct.size

                def unpack_from(self, buf, offset=0):
                    try:
                        return next(seq_reads)
                    except StopIteration:
                        return orig_seq_struct.unpack_from(buf, offset)

            with mock.patch.object(tpu_exporter, "_SHM_SEQ_STRUCT", _SeqProxy()):
                res_torn = tpu_exporter.read_shm_seqlock_metrics("0000:00:0b.0", shm_root=shm_root)
            self.assertIsNotNone(res_torn)
            assert res_torn is not None
            self.assertEqual(res_torn.hbm_used_bytes, 18 * 1024**3)

            # Freshness/expiry checks: stale sample after publisher exit MUST be omitted
            # so crashed workloads cannot pin HBM OOM alerts indefinitely.
            stale_ts = now_ns - 60 * 1_000_000_000
            shm_file.write_bytes(
                struct.pack(
                    "<IIQQQQQQ",
                    tpu_exporter.FREETPU_SHM_MAGIC,
                    tpu_exporter.FREETPU_SHM_VERSION,
                    8,
                    32 * 1024**3,
                    31 * 1024**3,  # high HBM that would trigger OOM if not expired
                    987654321,
                    104857600,
                    stale_ts,
                )
            )
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics(
                    "0000:00:0b.0", shm_root=shm_root, max_age_s=30.0, now_ns=now_ns
                )
            )
            with _mock_vfio_char_lstat():
                stale_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                    shm_root=shm_root,
                    shm_max_age_s=30.0,
                )
            self.assertIsNone(stale_snap.devices[0].hbm_used_bytes)
            self.assertEqual(stale_snap.devices[0].telemetry_source, "sysfs-fallback")

            # Custom max_age_s=120 accepts 60s-old sample while rejecting 180s-old sample.
            res_ttl = tpu_exporter.read_shm_seqlock_metrics(
                "0000:00:0b.0", shm_root=shm_root, max_age_s=120.0, now_ns=now_ns
            )
            self.assertIsNotNone(res_ttl)
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics(
                    "0000:00:0b.0",
                    shm_root=shm_root,
                    max_age_s=120.0,
                    now_ns=now_ns + 121 * 1_000_000_000,
                )
            )

            # Verify the C/C++ SHM header documents GCC/Clang-compatible atomic operations
            # against the plain `uint64_t seq` ABI field (not C11 `atomic_fetch_add(&s->seq, 1)`).
            shm_hdr_path = Path(__file__).resolve().parent / "freetpu_metrics_shm.h"
            shm_hdr_text = shm_hdr_path.read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"(?<!_)atomic_fetch_add\(", shm_hdr_text))
            self.assertIn("__atomic_fetch_add(&s->seq", shm_hdr_text)
            self.assertIn("freetpu_shm_metrics_v1_publish", shm_hdr_text)
            self.assertIn('FREETPU_METRICS_SHM_DEFAULT_DIR', shm_hdr_text)
            self.assertIn('"/run/freetpu-metrics"', shm_hdr_text)

            # Missing heartbeat (timestamp_ns=0) and far-future timestamps fail closed.
            for bad_ts in (0, now_ns + 3600 * 1_000_000_000):
                shm_file.write_bytes(
                    struct.pack(
                        "<IIQQQQQQ",
                        tpu_exporter.FREETPU_SHM_MAGIC,
                        tpu_exporter.FREETPU_SHM_VERSION,
                        10,
                        32 * 1024**3,
                        18 * 1024**3,
                        1,
                        2,
                        bad_ts,
                    )
                )
                self.assertIsNone(
                    tpu_exporter.read_shm_seqlock_metrics(
                        "0000:00:0b.0", shm_root=shm_root, now_ns=now_ns
                    )
                )

            # Persistent odd seq (crashed writer) must fail closed within max_retries.
            shm_file.write_bytes(
                struct.pack(
                    "<IIQQQQQQ",
                    tpu_exporter.FREETPU_SHM_MAGIC,
                    tpu_exporter.FREETPU_SHM_VERSION,
                    5,  # odd = write in progress
                    32 * 1024**3,
                    18 * 1024**3,
                    1,
                    2,
                    now_ns,
                )
            )
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics(
                    "0000:00:0b.0", shm_root=shm_root, max_retries=4
                )
            )

            # Bad magic must fail closed.
            shm_file.write_bytes(
                struct.pack("<IIQQQQQQ", 0xDEADBEEF, 1, 2, 0, 0, 0, 0, now_ns)
            )
            self.assertIsNone(
                tpu_exporter.read_shm_seqlock_metrics("0000:00:0b.0", shm_root=shm_root)
            )

    def test_sysfs_pcie_aer_counters_gcp_preemption_and_alerts_yaml(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sysfs_root, dev_root, proc_root = _make_mock_pci_tree(
                root,
                devices=[
                    {
                        "bdf": "0000:00:0b.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "3",
                    },
                    {
                        "bdf": "0000:80:00.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "5",
                    },
                ],
                create_vfio_nodes=["3", "5"],
            )
            # Linux sysfs key-value format with TOTAL_ERR_* on chip 0, plain integer on chip 1.
            pci0 = sysfs_root / "bus" / "pci" / "devices" / "0000:00:0b.0"
            (pci0 / "aer_dev_correctable").write_text(
                "RxErr 2\nBadTLP 1\nTOTAL_ERR_COR 3\n", encoding="utf-8"
            )
            (pci0 / "aer_dev_fatal").write_text(
                "DLP 0\nTLP 1\nTOTAL_ERR_FATAL 1\n", encoding="utf-8"
            )
            pci1 = sysfs_root / "bus" / "pci" / "devices" / "0000:80:00.0"
            (pci1 / "aer_dev_correctable").write_text("0\n", encoding="utf-8")
            (pci1 / "aer_dev_fatal").write_text("0\n", encoding="utf-8")

            # Local mock GCP metadata server checking `Metadata-Flavor: Google`.
            preempt_state = {"val": "TRUE"}

            class _MetaHandler(BaseHTTPRequestHandler):
                def log_message(self, format, *args):  # noqa: A002
                    pass

                def do_GET(self):  # noqa: N802
                    if self.headers.get("Metadata-Flavor") != "Google":
                        self.send_response(403)
                        self.end_headers()
                        return
                    body = preempt_state["val"].encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            meta_srv = ThreadingHTTPServer(("127.0.0.1", 0), _MetaHandler)
            meta_t = threading.Thread(target=meta_srv.serve_forever, daemon=True)
            meta_t.start()
            try:
                meta_url = f"http://127.0.0.1:{meta_srv.server_address[1]}/computeMetadata/v1/instance/preempted"
                with _mock_vfio_char_lstat():
                    snap = tpu_exporter.collect_snapshot(
                        sysfs_root=sysfs_root,
                        dev_root=dev_root,
                        proc_root=proc_root,
                        tpu_info_bin=None,
                        shm_root=root / "no-shm",
                        gcp_metadata_preemption=True,
                        gcp_metadata_url=meta_url,
                        gcp_metadata_timeout_s=1.0,
                    )
                self.assertEqual(snap.devices[0].aer_correctable_total, 3)
                self.assertEqual(snap.devices[0].aer_fatal_total, 1)
                self.assertEqual(snap.devices[1].aer_correctable_total, 0)
                self.assertEqual(snap.devices[1].aer_fatal_total, 0)
                self.assertEqual(snap.gcp_preempted_status, 1)

                prom = tpu_exporter.format_prometheus_metrics(snap)
                self._assert_valid_prometheus_text(prom)
                # AER counters MUST use stable hardware PCI labels (no scan-order `chip`
                # label) so losing a lower-index TPU never resets surviving series identity.
                self.assertIn(
                    'freetpu_pcie_aer_correctable_total{pci_bdf="0000:00:0b.0",device_id="0x006f",generation="v6e"} 3',
                    prom,
                )
                self.assertIn(
                    'freetpu_pcie_aer_fatal_total{pci_bdf="0000:00:0b.0",device_id="0x006f",generation="v6e"} 1',
                    prom,
                )
                self.assertIn(
                    'freetpu_pcie_aer_fatal_total{pci_bdf="0000:80:00.0",device_id="0x006f",generation="v6e"} 0',
                    prom,
                )
                for ln in prom.splitlines():
                    if ln.startswith("freetpu_pcie_aer_"):
                        self.assertNotIn('chip="', ln)
                self.assertIn("gcp_metadata_preempted_status 1", prom)

                # Simulate lower-index TPU `0000:00:0b.0` disappearing and `0000:80:00.0`
                # recording a fatal AER fault: `0000:80:00.0` shifts from chip_index 1->0,
                # yet its `freetpu_pcie_aer_fatal_total` label set MUST remain identical!
                shifted_root = root / "aer_shifted"
                sh_sysfs, sh_dev, sh_proc = _make_mock_pci_tree(
                    shifted_root,
                    devices=[
                        {
                            "bdf": "0000:80:00.0",
                            "vendor": "0x1ae0",
                            "device": "0x006f",
                            "driver": "vfio-pci",
                            "iommu_group": "5",
                        }
                    ],
                    create_vfio_nodes=["5"],
                )
                (sh_sysfs / "bus/pci/devices/0000:80:00.0/aer_dev_fatal").write_text(
                    "DLP 1\nTOTAL_ERR_FATAL 1\n", encoding="utf-8"
                )
                with _mock_vfio_char_lstat():
                    sh_snap = tpu_exporter.collect_snapshot(
                        sysfs_root=sh_sysfs,
                        dev_root=sh_dev,
                        proc_root=sh_proc,
                        tpu_info_bin=None,
                        shm_root=root / "no-shm",
                    )
                self.assertEqual(sh_snap.devices[0].chip_index, 0)
                sh_prom = tpu_exporter.format_prometheus_metrics(sh_snap)
                self._assert_valid_prometheus_text(sh_prom)
                self.assertIn(
                    'freetpu_pcie_aer_fatal_total{pci_bdf="0000:80:00.0",device_id="0x006f",generation="v6e"} 1',
                    sh_prom,
                )
            finally:
                meta_srv.shutdown()
                meta_srv.server_close()
                meta_t.join(timeout=2.0)

            # Scheme-less / malformed --gcp-metadata-url must degrade gracefully to None
            # without aborting snapshot collection or dropping PCI/VFIO metrics.
            with _mock_vfio_char_lstat():
                bad_url_snap = tpu_exporter.collect_snapshot(
                    sysfs_root=sysfs_root,
                    dev_root=dev_root,
                    proc_root=proc_root,
                    tpu_info_bin=None,
                    shm_root=root / "no-shm",
                    gcp_metadata_preemption=True,
                    gcp_metadata_url="metadata.google.internal/computeMetadata/v1/instance/preempted",
                    gcp_metadata_timeout_s=0.5,
                )
            self.assertIsNone(bad_url_snap.gcp_preempted_status)
            self.assertEqual(len(bad_url_snap.devices), 2)
            self.assertEqual(bad_url_snap.devices[0].aer_correctable_total, 3)

            # Built-in GCP metadata endpoints (`metadata.google.internal` and
            # `169.254.169.254`) MUST bypass HTTP proxies via ProxyHandler({}).
            for direct_url in (
                tpu_exporter.DEFAULT_GCP_METADATA_PREEMPTED_URL,
                "http://169.254.169.254/computeMetadata/v1/instance/preempted",
            ):
                captured_handlers = []

                class _FakeResp:
                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        return False

                    def read(self, n=-1):
                        del n
                        return b"TRUE"

                class _FakeOpener:
                    def open(self, req, timeout=None):
                        del timeout
                        self.req = req
                        return _FakeResp()

                def _fake_build_opener(*handlers):
                    captured_handlers.extend(handlers)
                    return _FakeOpener()

                with (
                    mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9"}),
                    mock.patch.object(urllib.request, "build_opener", side_effect=_fake_build_opener),
                    mock.patch.object(
                        urllib.request,
                        "urlopen",
                        side_effect=AssertionError("must bypass proxy via ProxyHandler({}) for built-in metadata host"),
                    ),
                ):
                    status = tpu_exporter.query_gcp_preemption_status(direct_url, timeout_s=0.5)
                self.assertEqual(status, 1)
                self.assertEqual(len(captured_handlers), 1)
                self.assertIsInstance(captured_handlers[0], urllib.request.ProxyHandler)
                self.assertEqual(captured_handlers[0].proxies, {})

            # HTTP protocol errors (e.g. BadStatusLine / IncompleteRead from a malformed
            # metadata reply) MUST return None for the preemption gauge and never abort
            # the rest of the TPU scrape!
            import http.client

            for proto_exc in (
                http.client.BadStatusLine("HTTP/1.1"),
                http.client.IncompleteRead(b"tru"),
            ):
                with mock.patch.object(urllib.request, "urlopen", side_effect=proto_exc):
                    self.assertIsNone(
                        tpu_exporter.query_gcp_preemption_status(
                            "http://127.0.0.1:1/preempted", timeout_s=0.5
                        )
                    )
                with (
                    _mock_vfio_char_lstat(),
                    mock.patch.object(urllib.request, "urlopen", side_effect=proto_exc),
                ):
                    proto_snap = tpu_exporter.collect_snapshot(
                        sysfs_root=sysfs_root,
                        dev_root=dev_root,
                        proc_root=proc_root,
                        tpu_info_bin=None,
                        shm_root=root / "no-shm",
                        gcp_metadata_preemption=True,
                        gcp_metadata_url="http://127.0.0.1:1/preempted",
                    )
                self.assertIsNone(proto_snap.gcp_preempted_status)
                self.assertEqual(len(proto_snap.devices), 2)

            alerts_path = EXPORTER_DIR / "prometheus-alerts.yml"
            self.assertTrue(alerts_path.is_file())
            alerts_text = alerts_path.read_text(encoding="utf-8")
            for rule_name in (
                "TpuHbmOOMWarning",
                "TpuMxuDutyCycleCollapse",
                "TpuPcieAerFault",
                "GcpSpotPreemption",
            ):
                self.assertIn(rule_name, alerts_text)
            # Runtime fallback branches must restrict to unmapped chips to avoid duplicate alerts.
            self.assertIn('freetpu_tpu_runtime_hbm_used_bytes{pci_bdf="unmapped"}', alerts_text)
            self.assertIn(
                'freetpu_tpu_runtime_duty_cycle_percent{pci_bdf="unmapped",expected_tpu_activity="sustained"}',
                alerts_text,
            )
            # Duty cycle collapse must be gated on sustained activity + current instant duty sample + HBM allocation
            # across per-PCI, runtime-unmapped, AND direct-gRPC unjoined sidecar series.
            self.assertIn('expected_tpu_activity="sustained"', alerts_text)
            self.assertIn('freetpu_tpu_duty_cycle_percent{expected_tpu_activity="sustained"} >= 0', alerts_text)
            self.assertIn(
                'freetpu_tpu_runtime_duty_cycle_percent{pci_bdf="unmapped",expected_tpu_activity="sustained"} >= 0',
                alerts_text,
            )
            # Unjoined branch MUST aggregate at the scrape-target level (`max by (job, instance, node)`)
            # so idle unallocated devices on a partially-used host never trigger false critical pages,
            # and suppress when any joined duty series on the target is healthy (`>= 10`).
            self.assertIn("max by (job, instance, node) (", alerts_text)
            self.assertIn(
                'avg_over_time(freetpu_tpu_runtime_unjoined_duty_cycle_percent{expected_tpu_activity="sustained"}[5m])',
                alerts_text,
            )
            self.assertIn(
                'freetpu_tpu_runtime_unjoined_duty_cycle_percent{expected_tpu_activity="sustained"} >= 0',
                alerts_text,
            )
            self.assertIn("and on(job, instance)", alerts_text)
            self.assertIn("unless on(job, instance)", alerts_text)
            self.assertIn('freetpu_tpu_hbm_used_bytes{expected_tpu_activity="sustained"} > 0', alerts_text)
            self.assertIn("inference replicas", alerts_text)
            # PCIe AER alert must use a multi-scrape window ([5m], not [1m]).
            self.assertIn("increase(freetpu_pcie_aer_fatal_total[5m]) > 0", alerts_text)
            self.assertNotIn("freetpu_pcie_aer_fatal_total[1m]", alerts_text)
            # Spot preemption alert must clarify monitoring-only scope, workload-local checkpointing,
            # and retain observed TRUE notices via max_over_time past VM shutdown and Alertmanager's
            # default 5m group_interval ([6m]).
            self.assertIn("max_over_time(gcp_metadata_preempted_status[6m]) == 1", alerts_text)
            self.assertIn("group_interval", alerts_text)
            self.assertIn("Monitoring alert only", alerts_text)
            self.assertIn("workload-local metadata watcher or SIGTERM", alerts_text)

            # Long (>260-byte) inline JSON `--chip-id-map` MUST parse cleanly without
            # probing the filesystem (no ENAMETOOLONG on 255-byte NAME_MAX filesystems).
            long_inline_map = json.dumps(
                {f"0000:00:{4 + i:02x}.0": str(i) for i in range(12)}, indent=4
            )
            self.assertGreater(len(long_inline_map), 260)
            loaded_inline = tpu_exporter._load_chip_id_map(long_inline_map)
            self.assertEqual(len(loaded_inline or {}), 12)
            self.assertEqual((loaded_inline or {})["0000:00:0f.0"], "11")

    def _assert_valid_prometheus_text(self, text: str) -> None:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        self.assertGreater(len(lines), 0)
        sample_re = re.compile(
            r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$"
        )
        seen_samples: set[str] = set()
        for ln in lines:
            if ln.startswith("# HELP ") or ln.startswith("# TYPE "):
                continue
            self.assertRegex(ln, sample_re, f"Invalid Prometheus metric line: {ln}")
            metric_key = ln.rsplit(" ", 1)[0]
            self.assertNotIn(metric_key, seen_samples, f"Duplicate metric sample: {metric_key}")
            seen_samples.add(metric_key)


if __name__ == "__main__":
    unittest.main()
