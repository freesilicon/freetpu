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

    def test_persistent_vm_reuse_across_shards(self) -> None:
        """4 CPU shards with max_cpu_slots=1 should create ONLY 1 VM and reuse it 3 times."""
        import subprocess
        import threading

        with tempfile.TemporaryDirectory(prefix="freetpu_reuse_") as td:
            td_p = Path(td)
            worker = td_p / "w.py"
            worker.write_text("print('ok')\n", encoding="utf-8")
            sessions: dict[str, str] = {}
            new_calls: list[str] = []
            stop_calls: list[str] = []
            lock = threading.Lock()

            def runner(args, _to):
                cmd = args[0]
                with lock:
                    if cmd == "usage":
                        return subprocess.CompletedProcess(
                            args, 0, f"Current balance: 7000.00 compute units\nUsage rate: 0.26/hr\nActive assignments: {len(sessions)}\n", ""
                        )
                    if cmd == "sessions":
                        return subprocess.CompletedProcess(args, 0, "\n".join(sessions.values()) + "\n", "")
                    if cmd == "new":
                        sname = args[args.index("-s") + 1]
                        new_calls.append(sname)
                        sessions[sname] = f"[{sname}] m-hm-use4b0-x1 | Hardware: CPU | Shape: High-RAM | Variant: DEFAULT"
                        return subprocess.CompletedProcess(args, 0, "ok\n", "")
                    if cmd == "stop":
                        sname = args[args.index("-s") + 1]
                        stop_calls.append(sname)
                        sessions.pop(sname, None)
                        return subprocess.CompletedProcess(args, 0, "stopped\n", "")
                    if cmd == "exec":
                        return subprocess.CompletedProcess(args, 0, "ok\n", "")
                return subprocess.CompletedProcess(args, 0, "", "")

            disp = cf.ColabFleetDispatcher(
                colab_bin="colab",
                auth="adc",
                run_dir=td_p / "run",
                governor_cfg=cf.CuGovernorConfig(max_cpu_slots=1, max_total_slots=2),
                reuse_sessions=True,
                max_jobs_per_session=10,
                spawn_spacing_s=0.0,
                command_runner=runner,
            )
            jobs = [cf.FleetJob(f"shard_{i}", 1, "CPU", worker, sku_preference=("CPU_HM",)) for i in range(4)]
            proof = disp.run_queue(jobs)
            self.assertEqual(len(new_calls), 1, f"Expected 1 VM spawn across 4 shards, got {len(new_calls)}")
            self.assertEqual(proof["session_efficiency"]["sessions_created"], 1)
            self.assertEqual(proof["session_efficiency"]["sessions_reused"], 3)
            self.assertTrue(proof["teardown"]["zero_fleet_leaked"])
            self.assertEqual(len(stop_calls), 1)
            for i in range(4):
                self.assertEqual(proof["jobs"][f"shard_{i}"]["status"], "done")

    def test_exponential_backoff_and_polite_spawn_pacing(self) -> None:
        import random
        import subprocess
        import threading

        rng = random.Random(42)
        b1 = cf.compute_backoff_s(1, base_s=30.0, cap_s=300.0, jitter_frac=0.15, rng=rng)
        b2 = cf.compute_backoff_s(2, base_s=30.0, cap_s=300.0, jitter_frac=0.15, rng=rng)
        b3 = cf.compute_backoff_s(3, base_s=30.0, cap_s=300.0, jitter_frac=0.15, rng=rng)
        b6 = cf.compute_backoff_s(6, base_s=30.0, cap_s=300.0, jitter_frac=0.15, rng=rng)
        self.assertTrue(25.5 <= b1 <= 34.5, f"b1={b1}")
        self.assertTrue(51.0 <= b2 <= 69.0, f"b2={b2}")
        self.assertTrue(102.0 <= b3 <= 138.0, f"b3={b3}")
        self.assertTrue(255.0 <= b6 <= 300.0, f"b6={b6}")

        # Verify spawn spacing sleeps >= 15s between distinct VM spawns and 503 backoff triggers wait_on_cooldown
        sleeps: list[float] = []
        with tempfile.TemporaryDirectory(prefix="freetpu_pace_") as td:
            td_p = Path(td)
            worker = td_p / "w.py"
            worker.write_text("print('ok')\n", encoding="utf-8")
            sessions: dict[str, str] = {}
            attempts = {"n": 0}
            lock = threading.Lock()

            def runner(args, _to):
                cmd = args[0]
                with lock:
                    if cmd == "usage":
                        return subprocess.CompletedProcess(
                            args, 0, f"Current balance: 7000.00 compute units\nUsage rate: 0.0/hr\nActive assignments: {len(sessions)}\n", ""
                        )
                    if cmd == "sessions":
                        return subprocess.CompletedProcess(args, 0, "\n".join(sessions.values()) + "\n", "")
                    if cmd == "new":
                        attempts["n"] += 1
                        if attempts["n"] == 1:
                            return subprocess.CompletedProcess(args, 1, "", "503 Service Unavailable")
                        sname = args[args.index("-s") + 1]
                        sessions[sname] = f"[{sname}] m-hm-use4b0-p1 | Hardware: CPU | Shape: High-RAM | Variant: DEFAULT"
                        return subprocess.CompletedProcess(args, 0, "ok\n", "")
                    if cmd == "stop":
                        sname = args[args.index("-s") + 1]
                        sessions.pop(sname, None)
                        return subprocess.CompletedProcess(args, 0, "stopped\n", "")
                    if cmd == "exec":
                        return subprocess.CompletedProcess(args, 0, "ok\n", "")
                return subprocess.CompletedProcess(args, 0, "", "")

            disp = cf.ColabFleetDispatcher(
                colab_bin="colab",
                auth="adc",
                run_dir=td_p / "run",
                governor_cfg=cf.CuGovernorConfig(max_cpu_slots=2, max_total_slots=2),
                reuse_sessions=False,
                spawn_spacing_s=18.0,
                backoff_base_s=30.0,
                backoff_cap_s=300.0,
                wait_on_cooldown=True,
                max_cooldown_wait_s=120.0,
                command_runner=runner,
                sleep_fn=lambda s: sleeps.append(s),
                rng=random.Random(7),
            )
            # Clear cooldown clock when fake sleep fires so second attempt immediately proceeds
            orig_sleep = disp._sleep
            def _fake_sleep(sec: float) -> None:
                orig_sleep(sec)
                for k in list(disp._sku_cooldown_until.keys()):
                    disp._sku_cooldown_until[k] = 0.0
            disp._sleep = _fake_sleep

            jobs = [cf.FleetJob(f"j{i}", 1, "CPU", worker, sku_preference=("CPU_HM",)) for i in range(2)]
            proof = disp.run_queue(jobs)
            self.assertEqual(proof["jobs"]["j0"]["status"], "done")
            self.assertEqual(proof["jobs"]["j1"]["status"], "done")
            self.assertTrue(any(s >= 25.0 for s in sleeps), f"Expected 503 backoff sleep, got {sleeps}")
            self.assertTrue(any(15.0 <= s <= 18.5 for s in sleeps), f"Expected >=15s spawn spacing sleep, got {sleeps}")

    def test_s3_describe_redacts_r2_account_subdomain(self) -> None:
        cfg = cf.S3Config(
            endpoint="https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
            bucket="demo",
            access_key_id="AK",
            secret_access_key="SK",
        )
        desc = cf.S3CompatibleSyncBackend(cfg).describe()
        self.assertEqual(desc["endpoint"], "https://<redacted>.r2.cloudflarestorage.com")


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
