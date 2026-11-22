#!/usr/bin/env python3
"""Deterministic offline self-tests for ops/tpu-exporter/tpu_exporter.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request

EXPORTER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORTER_DIR))

import tpu_exporter  # noqa: E402


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

    for node in create_vfio_nodes or []:
        (vfio_dir / node).write_text("", encoding="utf-8")

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
            grp.mkdir(parents=True, exist_ok=True)
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
            vfio0_st = orig_os_stat(dev_root / "vfio" / "0")
            vfio2_st = orig_os_stat(dev_root / "vfio" / "2")
            vfio0_id = (vfio0_st.st_dev, vfio0_st.st_ino)
            vfio2_id = (vfio2_st.st_dev, vfio2_st.st_ino)

            def _char_stat(path_arg, *args, **kwargs):
                st = orig_os_stat(path_arg, *args, **kwargs)
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
                return st

            # Without verified mapping: runtime IDs 8/12 must NOT be attached or fabricated as 0.
            with mock.patch("tpu_exporter.os.stat", side_effect=_char_stat):
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

            orig_iterdir = Path.iterdir

            def _guarded_iterdir(p: Path):
                if p.name == "fd" and p.parent.name == "1234":
                    raise PermissionError("simulated EACCES without CAP_SYS_PTRACE")
                return orig_iterdir(p)

            with mock.patch.object(Path, "iterdir", autospec=True, side_effect=_guarded_iterdir):
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
            with mock.patch.object(Path, "iterdir", autospec=True, side_effect=_guarded_iterdir):
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

            with mock.patch("tpu_exporter.os.stat", side_effect=_mock_os_stat):
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
            with mock.patch("tpu_exporter.os.stat", side_effect=_mock_os_stat):
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

            orig_run = tpu_exporter.subprocess.run

            def _run_with_fake_pkg(cmd, *args, **kwargs):
                kwargs.setdefault("env", env_with_fake)
                return orig_run(cmd, *args, **kwargs)

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
                    mock.patch("tpu_exporter.subprocess.run", side_effect=_run_with_fake_pkg),
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
