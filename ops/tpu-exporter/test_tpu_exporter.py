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
"""
        rows = tpu_exporter.parse_tpu_info_output(sample)
        self.assertIn("/dev/vfio/0", rows)
        self.assertEqual(rows["/dev/vfio/0"].pids, [4242])
        self.assertEqual(rows["/dev/vfio/1"].pids, [])
        self.assertEqual(rows["0"].hbm_used_bytes, 8 * 1024**3)
        self.assertEqual(rows["0"].hbm_total_bytes, int(round(15.75 * 1024**3)))
        self.assertAlmostEqual(rows["0"].duty_cycle_percent or 0.0, 87.5)
        self.assertEqual(rows["1"].hbm_used_bytes, 16 * 1024**3)
        self.assertAlmostEqual(rows["1"].duty_cycle_percent or 0.0, 0.0)

    def test_discovery_filters_non_google_and_scrapes_mock_tpu_info(self) -> None:
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
                        "device": "0x0063",
                        "driver": "vfio-pci",
                        "iommu_group": "0",
                        "numa_node": "0",
                    },
                    {
                        "bdf": "0000:00:06.0",
                        "vendor": "0x1ae0",
                        "device": "0x006f",
                        "driver": "vfio-pci",
                        "iommu_group": "2",
                        "numa_node": "0",
                    },
                ],
                create_vfio_nodes=["0", "2"],
            )

            # Synthetic /proc/9001/fd/3 -> <dev_root>/vfio/0
            fd_dir = proc_root / "9001" / "fd"
            fd_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(str(dev_root / "vfio" / "0"), fd_dir / "3")

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
                "| 0    | 4.00 GiB / 15.75 GiB  | 64.25%     |\n"
                "| 1    | 0.00 GiB / 31.25 GiB  | 0.00%      |\n"
                "EOF\n",
                encoding="utf-8",
            )
            mock_tpu_info.chmod(0o755)

            snap = tpu_exporter.collect_snapshot(
                sysfs_root=sysfs_root,
                dev_root=dev_root,
                proc_root=proc_root,
                tpu_info_bin=str(mock_tpu_info),
            )
            self.assertTrue(snap.tpu_info_available)
            self.assertEqual(len(snap.devices), 2)

            v5e = snap.devices[0]
            self.assertEqual(v5e.pci_bdf, "0000:00:05.0")
            self.assertEqual(v5e.generation, "v5e")
            self.assertTrue(v5e.healthy)
            self.assertEqual(v5e.active_pids, [7777, 9001])
            self.assertEqual(v5e.hbm_used_bytes, 4 * 1024**3)
            self.assertAlmostEqual(v5e.duty_cycle_percent or 0.0, 64.25)

            v6e = snap.devices[1]
            self.assertEqual(v6e.pci_bdf, "0000:00:06.0")
            self.assertEqual(v6e.generation, "v6e")
            self.assertTrue(v6e.healthy)
            self.assertEqual(v6e.active_pids, [])
            self.assertAlmostEqual(v6e.duty_cycle_percent or 0.0, 0.0)

            prom = tpu_exporter.format_prometheus_metrics(snap)
            self._assert_valid_prometheus_text(prom)
            self.assertIn("freetpu_tpu_devices_discovered 2", prom)
            self.assertIn('generation="v5e"', prom)
            self.assertIn('generation="v6e"', prom)
            self.assertIn('freetpu_tpu_duty_cycle_percent{chip="0",', prom)
            self.assertIn('pid="9001"', prom)

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
