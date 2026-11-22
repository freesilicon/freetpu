"""Offline unit & parity test suite for FreeTPU Tile DSL (zero TPU/Colab credentials)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import freetpu_tile as freetpu
from freetpu_tile.backends.cpu_ref import quantize_w8a16_weights
from freetpu_tile.colab_runner import CmdRes, run_colab_tile_kernel


class TestTileDSLFrontendAndParity(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["OPENBUENOS_MOCK"] = "1"

    def test_ir_tracing_all_four_core_ops(self) -> None:
        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_mm(a, b, out):
            freetpu.store(out, freetpu.dot(a, b))

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="w8a16")
        def k_silu(a, b, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(a, b)))

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="int8")
        def k_gelu(a, b, out):
            freetpu.store(out, freetpu.gelu(a @ b))

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="w8a16")
        def k_swiglu(x, wg, wu, wd, out):
            g = freetpu.silu(freetpu.dot(x, wg))
            u = freetpu.dot(x, wu)
            freetpu.store(out, freetpu.dot(g * u, wd))

        self.assertEqual(k_mm.ir.op_kind, freetpu.OpKind.MATMUL)
        self.assertEqual(k_silu.ir.op_kind, freetpu.OpKind.FUSED_MATMUL_SILU)
        self.assertEqual(k_gelu.ir.op_kind, freetpu.OpKind.FUSED_MATMUL_GELU)
        self.assertEqual(k_swiglu.ir.op_kind, freetpu.OpKind.SWIGLU)
        self.assertTrue(k_swiglu.ir.has_down_proj)

    def test_buenos_a0_mock_and_vliw_0x640_parity_int8(self) -> None:
        qparams = freetpu.QuantParams(
            s_in=0.0625, zp_in=0, s_w=0.0625, zp_w=0, s_out=0.0625, zp_out=0
        )
        rng = np.random.default_rng(123)
        a = rng.integers(-24, 25, size=(16, 64), dtype=np.int8)
        b = rng.integers(-16, 17, size=(64, 64), dtype=np.int8)

        for act_fn in (freetpu.silu, freetpu.gelu):
            @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8", quant=qparams)
            def k_fused(lhs, rhs, out):
                freetpu.store(out, act_fn(freetpu.dot(lhs, rhs)))

            report = k_fused.parity_check(
                a,
                b,
                targets=["cpu_ref", "buenos_a0"],
                mock=True,
                max_abs_tol=1.0,
            )
            self.assertTrue(report.passed, msg=f"INT8 {act_fn.__name__} parity failed: {report.parity}")
            b_meta = report.results["buenos_a0"].metadata
            self.assertIsNotNone(b_meta["mode_a_0x640_spline"])
            self.assertEqual(b_meta["mode_a_0x640_spline"]["packet_bytes"], 208)
            self.assertLess(b_meta["mode_a_0x640_spline"]["max_lsb_err"], 0.1)
            self.assertEqual(len(b_meta["chunk_sizes"]), 7)

    def test_buenos_a0_mock_w8a16_and_bf16_all_ops(self) -> None:
        rng = np.random.default_rng(99)
        m, k, n = 64, 512, 512
        a = (rng.standard_normal((m, k)) * 0.4).astype(np.float32)
        b_f = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
        q_w, s_w = quantize_w8a16_weights(b_f, rhs_transposed=False)

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="w8a16")
        def k_mm_silu(lhs, rhs, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(lhs, rhs)))

        rep_w8 = k_mm_silu.parity_check(
            a,
            q_w,
            targets=["cpu_ref", "buenos_a0", "cloud_tpu_pallas"],
            dry_run_targets=["cloud_tpu_pallas"],
            scales={"rhs": s_w},
            mock=True,
            iters=2,
        )
        self.assertTrue(rep_w8.passed, msg=f"W8A16 parity failed: {rep_w8.parity}")
        self.assertLess(rep_w8.parity["buenos_a0"]["nmse"], 2e-3)
        self.assertTrue(rep_w8.results["buenos_a0"].mock)

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_mm_gelu(lhs, rhs, out):
            freetpu.store(out, freetpu.gelu(freetpu.dot(lhs, rhs)))

        rep_bf = k_mm_gelu.parity_check(
            a,
            b_f,
            targets=["cpu_ref", "buenos_a0", "cloud_tpu_pallas"],
            dry_run_targets=["cloud_tpu_pallas"],
            mock=True,
            iters=2,
        )
        self.assertTrue(rep_bf.passed, msg=f"BF16 parity failed: {rep_bf.parity}")
        self.assertLess(rep_bf.parity["buenos_a0"]["nmse"], 2e-3)

    def test_swiglu_mock_parity_and_pallas_lowering(self) -> None:
        rng = np.random.default_rng(7)
        m, k, n_inter = 64, 512, 512
        x = (rng.standard_normal((m, k)) * 0.35).astype(np.float32)
        wg, sg = quantize_w8a16_weights((rng.standard_normal((k, n_inter)) / np.sqrt(k)).astype(np.float32))
        wu, su = quantize_w8a16_weights((rng.standard_normal((k, n_inter)) / np.sqrt(k)).astype(np.float32))
        wd, sd = quantize_w8a16_weights((rng.standard_normal((n_inter, k)) / np.sqrt(n_inter)).astype(np.float32))

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="w8a16")
        def k_swiglu(inp, w_gate, w_up, w_down, out):
            g = freetpu.silu(freetpu.dot(inp, w_gate))
            u = freetpu.dot(inp, w_up)
            freetpu.store(out, freetpu.dot(g * u, w_down))

        rep = k_swiglu.parity_check(
            x,
            wg,
            wu,
            wd,
            targets=["cpu_ref", "buenos_a0", "cloud_tpu_pallas"],
            dry_run_targets=["cloud_tpu_pallas"],
            scales={"w_gate": sg, "w_up": su, "w_down": sd},
            mock=True,
            iters=2,
        )
        self.assertTrue(rep.passed, msg=f"SwiGLU parity failed: {rep.parity}")
        self.assertLess(rep.parity["buenos_a0"]["nmse"], 5e-3)

        lowered_pallas = rep.results["cloud_tpu_pallas"].lowered
        assert lowered_pallas is not None and lowered_pallas.source_code is not None
        self.assertIn("pl.pallas_call", lowered_pallas.source_code)
        self.assertIn("pl.BlockSpec", lowered_pallas.source_code)
        self.assertIn("pltpu.VMEM", lowered_pallas.source_code)
        self.assertIn("CompilerParams", lowered_pallas.source_code)
        self.assertTrue(lowered_pallas.summary["ast_verified"])

    def test_pallas_lowering_validates_alignment(self) -> None:
        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_bad(a, b, out):
            freetpu.store(out, freetpu.dot(a, b))

        with self.assertRaises(ValueError):
            k_bad.lower(shapes={"a": (64, 300), "b": (300, 512)}, target="cloud_tpu_pallas")

    def test_colab_runner_isolated_config_and_fallback_offline(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        example = repo_root / "examples" / "tile_dsl" / "fused_gemm_silu.py"
        cmds: list[list[str]] = []

        def fake_runner(cmd: list[str] | tuple[str, ...]) -> CmdRes:
            c = list(cmd)
            cmds.append(c)
            if "usage" in c:
                return CmdRes(0, "Current balance: 100.0 compute units\n", "")
            if "run" in c:
                if "V6E1" in c:
                    return CmdRes(1, "", "[colab] Allocation refused: REASON_NO_BACKENDS (412)")
                return CmdRes(0, '{"status":"ok"}\n', "")
            if "log" in c:
                out_p = Path(c[c.index("-o") + 1])
                out_p.write_text('{"nbformat":4,"cells":[]}', encoding="utf-8")
                return CmdRes(0, "ok", "")
            raise AssertionError(f"Unexpected cmd: {c}")

        with tempfile.TemporaryDirectory(prefix="freetpu-colab-test-") as tmp:
            tmp_p = Path(tmp)
            iso_cfg = tmp_p / "iso" / "sessions.json"
            rc, summary = run_colab_tile_kernel(
                example,
                config_path=iso_cfg,
                out_dir=tmp_p / "out",
                session_name="unit-test",
                runner=fake_runner,  # type: ignore[arg-type]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(summary["chosen_tpu"], "v5e1")
            self.assertTrue(Path(summary["proof_notebook"]).is_file())
            for c in cmds:
                self.assertIn("--config", c)
                self.assertIn(str(iso_cfg), c)

    def test_example_scripts_cli_offline(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        for rel in ("examples/tile_dsl/fused_gemm_silu.py", "examples/tile_dsl/swiglu.py"):
            proc = subprocess.run(
                [sys.executable, str(repo_root / rel), "--target", "all", "--iters", "2"],
                capture_output=True,
                text=True,
                env={**os.environ, "OPENBUENOS_MOCK": "1"},
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{rel} failed:\n{proc.stderr}\n{proc.stdout}")
            self.assertIn('"passed": true', proc.stdout)

    def test_freetpu_tile_runner_host_fused_matmul_silu_mock(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        runner = repo_root / "runtime" / "build" / "freetpu_tile_runner_host"
        if not runner.is_file():
            self.skipTest("runtime/build/freetpu_tile_runner_host not built")

        from freetpu_tile.backends.cpu_ref import silu_f32

        rng = np.random.default_rng(11)
        m, k, n = 16, 64, 64
        a = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
        w_nk = (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)
        ref = silu_f32(a @ w_nk.T)

        with tempfile.TemporaryDirectory(prefix="freetpu-runner-test-") as tmp:
            tp = Path(tmp)
            (tp / "a.bin").write_bytes(a.tobytes())
            (tp / "w.bin").write_bytes(w_nk.tobytes())
            out_p = tp / "out.bin"
            proc = subprocess.run(
                [
                    str(runner),
                    "--mock",
                    "--op",
                    "fused_matmul_silu",
                    "--dtype",
                    "w8a16",
                    "--m",
                    str(m),
                    "--k",
                    str(k),
                    "--n",
                    str(n),
                    "--in",
                    str(tp / "a.bin"),
                    "--w",
                    str(tp / "w.bin"),
                    "--out",
                    str(out_p),
                    "--warmup",
                    "1",
                    "--iters",
                    "2",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"runner failed:\n{proc.stderr}\n{proc.stdout}")
            got = np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(m, n)
            diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            nmse = float(np.sum(diff * diff) / max(float(np.sum(ref.astype(np.float64) ** 2)), 1e-30))
            self.assertLess(nmse, 5e-3, msg=f"runner fused_matmul_silu nmse too high: {nmse}")

    def test_bias_validation_and_rejection_across_backends(self) -> None:
        from freetpu_tile.libopenbuenos_bridge import OpenBuenosSession

        rng = np.random.default_rng(5)
        a = rng.integers(-8, 9, size=(16, 64), dtype=np.int8)
        b = rng.integers(-8, 9, size=(64, 64), dtype=np.int8)
        with OpenBuenosSession(mock=True) as sess:
            with self.assertRaises(ValueError):
                sess.matmul_int8(a, b, bias_n=np.zeros((1,), dtype=np.int32))

        @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8")
        def k_i8_silu(lhs, rhs, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(lhs, rhs)))

        bias_ok = np.full((64,), 10, dtype=np.int32)
        rep = k_i8_silu.parity_check(
            a, b, targets=["cpu_ref", "buenos_a0"], bias=bias_ok, mock=True, max_abs_tol=1.0
        )
        self.assertTrue(rep.passed, msg=f"INT8 bias parity failed: {rep.parity}")

        with self.assertRaises(ValueError):
            k_i8_silu.run(
                np.zeros((64, 512), dtype=np.int8),
                np.zeros((512, 512), dtype=np.int8),
                target="cloud_tpu_pallas",
                dry_run=True,
                bias=np.zeros((512,), dtype=np.int32),
            )

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_bf16_silu(lhs, rhs, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(lhs, rhs)))

        with self.assertRaises(ValueError):
            k_bf16_silu.run(
                np.zeros((64, 512), dtype=np.float32),
                np.zeros((512, 512), dtype=np.float32),
                target="buenos_a0",
                mock=True,
                bias=np.zeros((512,), dtype=np.float32),
            )

    def test_traced_operand_order_tile_indices_and_swiglu_constraints(self) -> None:
        @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8")
        def k_swapped(a, b, out):
            freetpu.store(out, freetpu.dot(b, a))

        rng = np.random.default_rng(17)
        a_km = rng.integers(-8, 9, size=(32, 16), dtype=np.int8)
        b_mk = rng.integers(-8, 9, size=(16, 32), dtype=np.int8)
        res_cpu = k_swapped.run(a_km, b_mk, target="cpu_ref")
        res_b = k_swapped.run(a_km, b_mk, target="buenos_a0", mock=True)
        self.assertEqual(res_cpu.output.shape, (16, 16))
        self.assertTrue(np.array_equal(res_cpu.output, res_b.output))

        # Invalid constant tile load index must be rejected
        with self.assertRaises(ValueError):
            @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
            def k_bad_idx(a, b, out):
                pid_n = freetpu.program_id(1)
                a_tile = freetpu.load(a, (0, None), shape=(64, 512))
                b_tile = freetpu.load(b, (None, pid_n), shape=(512, 512))
                freetpu.store(out, freetpu.dot(a_tile, b_tile))

        # Invalid store index must be rejected
        with self.assertRaises(ValueError):
            @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
            def k_bad_store(a, b, out):
                pid_m = freetpu.program_id(0)
                pid_n = freetpu.program_id(1)
                freetpu.store(out, freetpu.dot(a, b), (pid_n, pid_m))

        # Inconsistent SwiGLU per-projection transpose must be rejected
        with self.assertRaises(ValueError):
            @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
            def k_bad_swiglu_t(x, wg, wu, wd, out):
                g = freetpu.silu(freetpu.dot(x, wg, rhs_transposed=False))
                u = freetpu.dot(x, wu, rhs_transposed=False)
                freetpu.store(out, freetpu.dot(g * u, wd, rhs_transposed=True))

        # INT8 SwiGLU must be rejected consistently across targets
        with self.assertRaises(ValueError):
            @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="int8")
            def k_int8_swiglu(x, wg, wu, out):
                freetpu.store(out, freetpu.silu(freetpu.dot(x, wg)) * freetpu.dot(x, wu))

    def test_cloud_pallas_asymmetric_zp_and_colab_cli_separator(self) -> None:
        from freetpu_tile.colab_runner import main as colab_main

        q = freetpu.QuantParams(s_in=0.05, zp_in=-3, s_w=0.04, zp_w=5, s_out=0.05, zp_out=1)

        @freetpu.tile_kernel(block_m=64, block_n=128, block_k=128, dtype="int8", quant=q)
        def k_asym(a, b, out):
            freetpu.store(out, freetpu.dot(a, b))

        a = np.zeros((64, 128), dtype=np.int8)
        b = np.zeros((128, 128), dtype=np.int8)
        lowered = k_asym.lower(a, b, target="cloud_tpu_pallas", quant=q)
        assert lowered.source_code is not None
        self.assertIn("zp_in" if False else "jnp.sum(a_tile.astype(jnp.int32)", lowered.source_code)
        self.assertIn("jnp.sum(b_tile.astype(jnp.int32)", lowered.source_code)

        # Verify algebraic identity of emitted Pallas INT8 zp correction on random inputs
        rng = np.random.default_rng(77)
        a_r = rng.integers(-20, 21, size=(64, 128), dtype=np.int8)
        b_r = rng.integers(-20, 21, size=(128, 128), dtype=np.int8)
        exact_acc = (a_r.astype(np.int32) - q.zp_in) @ (b_r.astype(np.int32) - q.zp_w)
        alg_acc = (
            (a_r.astype(np.int32) @ b_r.astype(np.int32))
            - q.zp_w * np.sum(a_r.astype(np.int32), axis=1, keepdims=True)
            - q.zp_in * np.sum(b_r.astype(np.int32), axis=0, keepdims=True)
            + 128 * q.zp_in * q.zp_w
        )
        self.assertTrue(np.array_equal(exact_acc, alg_acc))

        repo_root = Path(__file__).resolve().parent.parent.parent
        example = repo_root / "examples" / "tile_dsl" / "fused_gemm_silu.py"
        with tempfile.TemporaryDirectory(prefix="freetpu-colab-cli-") as tmp:
            rc = colab_main([
                str(example),
                "--out-dir",
                tmp,
                "--dry-run",
                "--",
                "--target",
                "cloud_tpu_pallas",
                "--dtype",
                "w8a16",
                "--M",
                "64",
                "--K",
                "512",
                "--N",
                "512",
                "--iters",
                "20",
            ])
            self.assertEqual(rc, 0)

    def test_saturating_int8_s_out_iters_and_runner_edge_cases(self) -> None:
        from unittest import mock

        # 1. Saturating small s_out requantizes cleanly rather than widening output scale
        q_sat = freetpu.QuantParams(s_in=0.05, zp_in=0, s_w=0.05, zp_w=0, s_out=0.001, zp_out=0)

        @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8", quant=q_sat)
        def k_sat(a, b, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(a, b)))

        rng = np.random.default_rng(91)
        a_s = rng.integers(-20, 21, size=(16, 64), dtype=np.int8)
        b_s = rng.integers(-20, 21, size=(64, 64), dtype=np.int8)
        rep_sat = k_sat.parity_check(
            a_s, b_s, targets=["cpu_ref", "buenos_a0"], quant=q_sat, mock=True, max_abs_tol=1.0
        )
        self.assertTrue(rep_sat.passed, msg=f"Small s_out requantization failed: {rep_sat.parity}")

        # 2. Host INT8 actually invokes matmul_int8 warmup + iters times
        from freetpu_tile.libopenbuenos_bridge import OpenBuenosSession

        orig_mm = OpenBuenosSession.matmul_int8
        call_cnt = {"n": 0}

        def _counting_mm(self_s, *args, **kwargs):
            call_cnt["n"] += 1
            return orig_mm(self_s, *args, **kwargs)

        with mock.patch.object(OpenBuenosSession, "matmul_int8", _counting_mm):
            k_sat.run(a_s, b_s, target="buenos_a0", quant=q_sat, mock=True, warmup=1, iters=4)
        self.assertEqual(call_cnt["n"], 5)

        # 3. ADB runner checks: stops on upload failure + passes --zp-w + conditional --w-down
        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_2proj_swiglu(x, wg, wu, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(x, wg)) * freetpu.dot(x, wu))

        adb_cmds: list[list[str]] = []

        def _failing_push(cmd, **_kw):
            c = list(cmd)
            adb_cmds.append(c)
            if "push" in c and any("in.bin" in s for s in c):
                return subprocess.CompletedProcess(c, 1, "", "adb: device offline")
            return subprocess.CompletedProcess(c, 0, "", "")

        with mock.patch("freetpu_tile.backends.buenos_a0.subprocess.run", side_effect=_failing_push):
            repo_root = Path(__file__).resolve().parent.parent.parent
            runner_bin = repo_root / "runtime" / "build" / "freetpu_tile_runner"
            with mock.patch.object(Path, "is_file", lambda p: True if str(p) == str(runner_bin) else os.path.isfile(p)):
                with self.assertRaises(RuntimeError):
                    k_sat.run(a_s, b_s, target="buenos_a0", adb_serial="fake-serial", mock=False)
        self.assertFalse(any("su -c" in " ".join(c) for c in adb_cmds))

        # 4. Host C++ runner: 2-proj swiglu + INT8 asymmetric zp_w
        repo_root = Path(__file__).resolve().parent.parent.parent
        runner_host = repo_root / "runtime" / "build" / "freetpu_tile_runner_host"
        if runner_host.is_file():
            with tempfile.TemporaryDirectory(prefix="freetpu-runner-2proj-") as tmp:
                tp = Path(tmp)
                # 2-projection SwiGLU on runner_host
                x2 = (rng.standard_normal((16, 64)) * 0.4).astype(np.float32)
                wg2 = (rng.standard_normal((64, 64)) / np.sqrt(64)).astype(np.float32)
                wu2 = (rng.standard_normal((64, 64)) / np.sqrt(64)).astype(np.float32)
                ref2 = k_2proj_swiglu.run(x2, wg2, wu2, target="cpu_ref").output
                (tp / "x.bin").write_bytes(x2.tobytes())
                (tp / "wg.bin").write_bytes(wg2.T.copy().tobytes())
                (tp / "wu.bin").write_bytes(wu2.T.copy().tobytes())
                out2 = tp / "out2.bin"
                p2 = subprocess.run(
                    [
                        str(runner_host), "--mock", "--op", "swiglu", "--dtype", "w8a16",
                        "--m", "16", "--k", "64", "--n", "64", "--n-inter", "64",
                        "--in", str(tp / "x.bin"), "--w", str(tp / "wg.bin"),
                        "--w-up", str(tp / "wu.bin"), "--out", str(out2), "--warmup", "1", "--iters", "1",
                    ],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(p2.returncode, 0, msg=p2.stderr)
                got2 = np.frombuffer(out2.read_bytes(), dtype=np.float32).reshape(16, 64)
                nmse2 = float(np.sum((got2 - ref2) ** 2) / max(float(np.sum(ref2 ** 2)), 1e-30))
                self.assertLess(nmse2, 5e-3)

                # INT8 --zp-w forwarding
                q_zw = freetpu.QuantParams(s_in=0.05, zp_in=-2, s_w=0.04, zp_w=3, s_out=0.05, zp_out=1)
                @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8", quant=q_zw)
                def k_i8_mm(lhs, rhs, out):
                    freetpu.store(out, freetpu.dot(lhs, rhs))
                ref_zw = k_i8_mm.run(a_s, b_s, target="cpu_ref", quant=q_zw).output
                (tp / "ai.bin").write_bytes(a_s.tobytes())
                (tp / "bi.bin").write_bytes(b_s.T.copy().tobytes())
                out_zw = tp / "out_zw.bin"
                p_zw = subprocess.run(
                    [
                        str(runner_host), "--mock", "--op", "matmul", "--dtype", "int8",
                        "--m", "16", "--k", "64", "--n", "64",
                        "--in", str(tp / "ai.bin"), "--w", str(tp / "bi.bin"),
                        "--s-in", str(q_zw.s_in), "--zp-in", str(q_zw.zp_in),
                        "--s-w", str(q_zw.s_w), "--zp-w", str(q_zw.zp_w),
                        "--s-out", str(q_zw.s_out), "--zp-out", str(q_zw.zp_out),
                        "--out", str(out_zw), "--warmup", "1", "--iters", "1",
                    ],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(p_zw.returncode, 0, msg=p_zw.stderr)
                got_zw = np.frombuffer(out_zw.read_bytes(), dtype=np.int8).reshape(16, 64)
                self.assertTrue(np.array_equal(got_zw, ref_zw))


if __name__ == "__main__":
    unittest.main()
