#!/usr/bin/env python3
"""Offline unit tests for `tools/colab_fleet.py` (0 CUs, 0 network)."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import colab_fleet as cf  # noqa: E402


class TestColabFleet(unittest.TestCase):
    def test_self_test(self) -> None:
        res = cf.run_self_test()
        self.assertTrue(res["all_passed"])

    def test_public_only_guardrail_blocks_private_paths(self) -> None:
        for bad in (
            "/tmp/archive.sqlite",
            "/tmp/org-private/secrets.json",
            "/tmp/cloud/credentials.json",
            "/tmp/.env",
        ):
            with self.assertRaises(ValueError):
                cf.assert_public_path(bad)

    def test_s3_backend_put_get_and_no_secret_leak(self) -> None:
        store: dict[str, bytes] = {}

        def fake_http(req, _timeout):
            if req.get_method() == "PUT":
                store[req.full_url] = req.data
                return 200, b""
            if req.get_method() == "GET":
                return 200, store[req.full_url]
            return 404, b""

        cfg = cf.S3Config(
            endpoint="https://r2.example.invalid",
            bucket="public-artifacts",
            access_key_id="AK_TEST_999",
            secret_access_key="SUPER_SECRET_NEVER_LOG",
            region="auto",
            prefix="fleet-test",
        )
        bk = cf.S3CompatibleSyncBackend(cfg, http_opener=fake_http)
        desc = bk.describe()
        self.assertNotIn("SUPER_SECRET_NEVER_LOG", str(desc))
        self.assertNotIn("AK_TEST_999", str(desc))

        put_res = bk.put_bytes("wave1/adapter.bin", b"lora-bytes-123")
        self.assertEqual(put_res["size_bytes"], 14)
        self.assertEqual(bk.get_bytes("wave1/adapter.bin"), b"lora-bytes-123")

    def test_teardown_refuses_non_fleet_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="freetpu_fleet_refuse_") as td:
            disp = cf.ColabFleetDispatcher(
                colab_bin="colab",
                auth="adc",
                run_dir=Path(td),
                governor_cfg=cf.CuGovernorConfig(),
            )
            with self.assertRaises(RuntimeError):
                disp.stop_owned_session("gpu-a100-s-usc1f0-external")

    def test_session_and_usage_parsers(self) -> None:
        sample = (
            "[?] gpu-a100-s-usc1f0-abc | Hardware: A100 | Shape: Standard | Variant: GPU\n"
            "[fleet-j1-v5e1-123456] tpu-v5e1-s-usw4a0-xyz | Hardware: V5E1 | Shape: Standard | Variant: TPU\n"
            "[fleet-j2-cpu_hm-654321] m-hm-ase1a0-qwe | Hardware: CPU | Shape: High-RAM | Variant: DEFAULT\n"
        )
        parsed = cf.parse_sessions_output(sample)
        self.assertEqual(len(parsed), 3)
        self.assertEqual([s["kind"] for s in parsed], ["GPU", "TPU", "CPU"])
        self.assertEqual([s["zone"] for s in parsed], ["usc1f0", "usw4a0", "ase1a0"])

        u = cf.parse_usage_output("Current balance: 7323.47 compute units\nUsage rate: 11.42/hr\nActive assignments: 2\n")
        self.assertEqual(u["balance_cu"], 7323.47)
        self.assertEqual(u["usage_rate_hr"], 11.42)
        self.assertEqual(u["active_assignments"], 2)

    def test_classify_assign_error_ignores_rich_traceback_source_lines(self) -> None:
        rich_503 = (
            "│ 170 │ except TooManyAssignmentsError:\n"
            "╰────────────────────────╯\n"
            "ColabRequestError: Failed to issue request POST /tun/m/assign: Service Unavailable\n"
        )
        self.assertEqual(cf.classify_assign_error(rich_503), "503_UNAVAILABLE")
        rich_412 = "[colab] Allocation refused (precondition failed). This can mean too many active sessions."
        self.assertEqual(cf.classify_assign_error(rich_412), "412_TOO_MANY_ACTIVE")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unit tests for tools/colab_fleet.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
