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
        (d / "enable").write_text(dev.get("enable", "1") + "\n", encoding="utf-8")
        if "numa_node" in dev:
            (d / "numa_node").write_text(dev["numa_node"] + "\n", encoding="utf-8")
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
        dashboard = """
\x1b[1mTPU Device Status\x1b[0m
+-----------+---------------------------+--------------------+--------------------+-----------------------+
| Device ID | Memory Usage (Used/Total) | Memory Bar Gauge   | HBM Bandwidth Util | TensorCore Duty Cycle |
+-----------+---------------------------+--------------------+--------------------+-----------------------+
| 0         | 8.00 GiB / 15.75 GiB      | [#####.....] 50.8% | [##........] 20.0% | [########..] 87.5%    |
| 1         | N/A                       | [..........] N/A   | [..........] N/A   | [..........] N/A      |
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

            # Synthetic /proc/9001/fd/3 -> <dev_root>/vfio/0
            fd_dir = proc_root / "9001" / "fd"
            fd_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(str(dev_root / "vfio" / "0"), fd_dir / "3")

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

            # Without verified mapping: runtime IDs 8/12 must NOT be attached or fabricated as 0.
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
