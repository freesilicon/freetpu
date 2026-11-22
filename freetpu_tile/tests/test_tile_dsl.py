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


def _resolve_example_script(filename: str) -> Path:
    """Resolve example script from a repo checkout or installed wheel package data."""
    pkg_root = Path(__file__).resolve().parent.parent
    repo_candidate = pkg_root.parent / "examples" / "tile_dsl" / filename
    if repo_candidate.is_file():
        return repo_candidate
    packaged_candidate = pkg_root / "_packaged_examples" / filename
    if packaged_candidate.is_file():
        return packaged_candidate
    raise FileNotFoundError(
        f"Example fixture {filename!r} not found in {repo_candidate} or {packaged_candidate}"
    )


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

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="w8a16")
        def k_mm_gelu_w8(lhs, rhs, out):
            freetpu.store(out, freetpu.gelu(freetpu.dot(lhs, rhs)))

        rep_w8_gelu = k_mm_gelu_w8.parity_check(
            a,
            q_w,
            targets=["cpu_ref", "buenos_a0", "cloud_tpu_pallas"],
            dry_run_targets=["cloud_tpu_pallas"],
            scales={"rhs": s_w},
            mock=True,
            iters=2,
        )
        self.assertTrue(rep_w8_gelu.passed, msg=f"W8A16 GELU parity failed: {rep_w8_gelu.parity}")
        self.assertLess(rep_w8_gelu.parity["buenos_a0"]["nmse"], 2e-3)

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_mm_gelu(lhs, rhs, out):
            freetpu.store(out, freetpu.gelu(freetpu.dot(lhs, rhs)))

        # buenos_a0 must explicitly reject native BF16 rather than silently Q8_0-quantizing weights
        with self.assertRaises(ValueError):
            k_mm_gelu.lower(a, b_f, target="buenos_a0")
        with self.assertRaises(ValueError):
            k_mm_gelu.run(a, b_f, target="buenos_a0", mock=True)

        # cpu_ref and cloud_tpu_pallas continue to support BF16 cleanly
        rep_bf = k_mm_gelu.parity_check(
            a,
            b_f,
            targets=["cpu_ref", "cloud_tpu_pallas"],
            dry_run_targets=["cloud_tpu_pallas"],
            iters=2,
        )
        self.assertTrue(rep_bf.passed, msg=f"BF16 parity failed: {rep_bf.parity}")

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
        example = _resolve_example_script("fused_gemm_silu.py")
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
        for name in ("fused_gemm_silu.py", "swiglu.py"):
            script = _resolve_example_script(name)
            proc = subprocess.run(
                [sys.executable, str(script), "--target", "all", "--iters", "2"],
                capture_output=True,
                text=True,
                env={**os.environ, "OPENBUENOS_MOCK": "1"},
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{name} failed:\n{proc.stderr}\n{proc.stdout}")
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

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="w8a16")
        def k_w8_silu(lhs, rhs, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(lhs, rhs)))

        with self.assertRaises(ValueError):
            k_w8_silu.run(
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

        example = _resolve_example_script("fused_gemm_silu.py")
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

    def test_synthesize_dgc0_validates_weight_and_bias_shapes(self) -> None:
        from unittest import mock

        from freetpu_tile.libopenbuenos_bridge import OB_ACT_NONE, OpenBuenosSession

        with OpenBuenosSession(mock=True) as sess:
            sess.emitter_registered = True
            with mock.patch.object(
                sess.bridge.lib, "ob_gemm_synthesize_dgc0"
            ) as native_synth:
                # Wrong weights_nk shape (undersized rows)
                with self.assertRaises(ValueError):
                    sess.synthesize_dgc0(
                        m_dim=1,
                        k_dim=16,
                        n_dim=16,
                        weights_nk=np.zeros((8, 16), dtype=np.int8),
                        bias_n=np.zeros((16,), dtype=np.int32),
                        s_in=0.0625,
                        zp_in=0,
                        s_w=0.0625,
                        s_out=0.0625,
                        zp_out=0,
                        activation=OB_ACT_NONE,
                    )
                # Wrong weights_nk shape (undersized cols)
                with self.assertRaises(ValueError):
                    sess.synthesize_dgc0(
                        m_dim=1,
                        k_dim=16,
                        n_dim=16,
                        weights_nk=np.zeros((16, 8), dtype=np.int8),
                        bias_n=np.zeros((16,), dtype=np.int32),
                        s_in=0.0625,
                        zp_in=0,
                        s_w=0.0625,
                        s_out=0.0625,
                        zp_out=0,
                        activation=OB_ACT_NONE,
                    )
                # Wrong bias_n shape (undersized 1D and wrong rank 2D)
                with self.assertRaises(ValueError):
                    sess.synthesize_dgc0(
                        m_dim=1,
                        k_dim=16,
                        n_dim=16,
                        weights_nk=np.zeros((16, 16), dtype=np.int8),
                        bias_n=np.zeros((4,), dtype=np.int32),
                        s_in=0.0625,
                        zp_in=0,
                        s_w=0.0625,
                        s_out=0.0625,
                        zp_out=0,
                        activation=OB_ACT_NONE,
                    )
                with self.assertRaises(ValueError):
                    sess.synthesize_dgc0(
                        m_dim=1,
                        k_dim=16,
                        n_dim=16,
                        weights_nk=np.zeros((16, 16), dtype=np.int8),
                        bias_n=np.zeros((16, 1), dtype=np.int32),
                        s_in=0.0625,
                        zp_in=0,
                        s_w=0.0625,
                        s_out=0.0625,
                        zp_out=0,
                        activation=OB_ACT_NONE,
                    )
                # Non-positive dimensions
                with self.assertRaises(ValueError):
                    sess.synthesize_dgc0(
                        m_dim=0,
                        k_dim=16,
                        n_dim=16,
                        weights_nk=np.zeros((16, 16), dtype=np.int8),
                        bias_n=None,
                        s_in=0.0625,
                        zp_in=0,
                        s_w=0.0625,
                        s_out=0.0625,
                        zp_out=0,
                        activation=OB_ACT_NONE,
                    )
                native_synth.assert_not_called()

    def test_openbuenos_mock_env_prevents_adb_dispatch_unless_explicit_mock_false(self) -> None:
        from unittest import mock

        @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8")
        def k_mm(a, b, out):
            freetpu.store(out, freetpu.dot(a, b))

        a = np.zeros((16, 64), dtype=np.int8)
        b = np.zeros((64, 64), dtype=np.int8)

        # 1. OPENBUENOS_MOCK=1 + FREETPU_ADB_SERIAL set + mock=None -> MUST NOT dispatch over ADB
        with mock.patch.dict(
            os.environ,
            {"OPENBUENOS_MOCK": "1", "FREETPU_ADB_SERIAL": "fake-adb-device"},
            clear=False,
        ):
            with mock.patch(
                "freetpu_tile.backends.buenos_a0._run_adb_rango",
                side_effect=AssertionError("ADB must not be called when OPENBUENOS_MOCK=1"),
            ) as adb_mock:
                res = k_mm.run(a, b, target="buenos_a0", iters=1, warmup=0)
                self.assertTrue(res.mock)
                adb_mock.assert_not_called()

        # 2. OPENBUENOS_MOCK=1 + FREETPU_ADB_SERIAL set + explicit mock=False -> MUST dispatch over ADB
        sentinel = freetpu.ExecutionResult(
            target=freetpu.Target.BUENOS_A0,
            op_kind=freetpu.OpKind.MATMUL,
            dtype=freetpu.DType.INT8,
            output=np.zeros((16, 64), dtype=np.int8),
            hw_us=10.0,
            wall_us=12.0,
            mock=False,
        )
        with mock.patch.dict(
            os.environ,
            {"OPENBUENOS_MOCK": "1", "FREETPU_ADB_SERIAL": "fake-adb-device"},
            clear=False,
        ):
            with mock.patch(
                "freetpu_tile.backends.buenos_a0._run_adb_rango",
                return_value=sentinel,
            ) as adb_mock:
                res_live = k_mm.run(a, b, target="buenos_a0", mock=False, iters=1, warmup=0)
                self.assertFalse(res_live.mock)
                adb_mock.assert_called_once()

        # 3. OPENBUENOS_MOCK unset + FREETPU_ADB_SERIAL set + mock=None -> dispatches over ADB
        env_no_mock = dict(os.environ)
        env_no_mock.pop("OPENBUENOS_MOCK", None)
        env_no_mock["FREETPU_ADB_SERIAL"] = "fake-adb-device"
        with mock.patch.dict(os.environ, env_no_mock, clear=True):
            with mock.patch(
                "freetpu_tile.backends.buenos_a0._run_adb_rango",
                return_value=sentinel,
            ) as adb_mock:
                res_adb = k_mm.run(a, b, target="buenos_a0", iters=1, warmup=0)
                self.assertFalse(res_adb.mock)
                adb_mock.assert_called_once()

    def test_fused_int8_mode_b_prevents_preactivation_clipping(self) -> None:
        # Exact regression case: s_in=s_w=0.0625, s_out=0.01, zp=0
        # Preactivations near 1.27 (~code 99), 2.0 (~code 127), and -10.0 (~code 0) must stay distinct.
        q_clip = freetpu.QuantParams(
            s_in=0.0625, zp_in=0, s_w=0.0625, zp_w=0, s_out=0.01, zp_out=0
        )

        # Construct exact accumulators via row-0 of a @ b:
        #   s_in * s_w = 1/256 = 0.00390625
        #   acc = 325  -> x = 1.26953125 -> SiLU(x)/0.01 = 99.11 -> code 99
        #   acc = 512  -> x = 2.0        -> SiLU(x)/0.01 = 176.16 -> code 127
        #   acc = -2560 -> x = -10.0     -> SiLU(x)/0.01 = -0.045 -> code 0
        a = np.zeros((16, 64), dtype=np.int8)
        b = np.zeros((64, 64), dtype=np.int8)
        # Col 0: sum = 5 * 65 = 325
        a[0, :5] = 65
        b[:5, 0] = 1
        # Col 1: sum = 8 * 64 = 512
        a[0, 5:13] = 64
        b[5:13, 1] = 1
        # Col 2: sum = 32 * (-80) = -2560
        a[0, 13:45] = -80
        b[13:45, 2] = 1

        for act_fn in (freetpu.silu, freetpu.gelu):
            @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8", quant=q_clip)
            def k_act(lhs, rhs, out):
                freetpu.store(out, act_fn(freetpu.dot(lhs, rhs)))

            rep = k_act.parity_check(
                a, b, targets=["cpu_ref", "buenos_a0"], quant=q_clip, mock=True, max_abs_tol=1.0
            )
            self.assertTrue(rep.passed, msg=f"{act_fn.__name__} preactivation clip parity failed: {rep.parity}")
            b_res = rep.results["buenos_a0"]
            self.assertIsNone(b_res.metadata["mode_a_0x640_spline"])
            self.assertIsNotNone(b_res.metadata["mode_b_requantize"])
            out_b = b_res.output
            out_ref = rep.results["cpu_ref"].output
            if act_fn is freetpu.silu:
                self.assertEqual(int(out_ref[0, 0]), 99)
                self.assertEqual(int(out_ref[0, 1]), 127)
                self.assertEqual(int(out_ref[0, 2]), 0)
            self.assertLessEqual(abs(int(out_b[0, 0]) - int(out_ref[0, 0])), 1)
            self.assertLessEqual(abs(int(out_b[0, 1]) - int(out_ref[0, 1])), 1)
            self.assertLessEqual(abs(int(out_b[0, 2]) - int(out_ref[0, 2])), 1)
            self.assertNotEqual(int(out_b[0, 0]), int(out_b[0, 1]))

        # Large s_out where spline domain cap (+11.1) would clip before output saturation (+127 * 0.2 = 25.4)
        q_wide = freetpu.QuantParams(
            s_in=0.5, zp_in=0, s_w=0.5, zp_w=0, s_out=0.2, zp_out=0
        )

        @freetpu.tile_kernel(block_m=16, block_n=64, block_k=64, dtype="int8", quant=q_wide)
        def k_wide(lhs, rhs, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(lhs, rhs)))

        a_w = np.zeros((16, 64), dtype=np.int8)
        b_w = np.zeros((64, 64), dtype=np.int8)
        a_w[0, 0] = 60  # acc=60 -> x=15.0 -> SiLU(15)/0.2 = 75
        b_w[0, 0] = 1
        a_w[0, 1] = 100  # acc=100 -> x=25.0 -> SiLU(25)/0.2 = 125
        b_w[1, 1] = 1
        rep_w = k_wide.parity_check(
            a_w, b_w, targets=["cpu_ref", "buenos_a0"], quant=q_wide, mock=True, max_abs_tol=1.0
        )
        self.assertTrue(rep_w.passed, msg=f"Wide s_out Mode-B fallback failed: {rep_w.parity}")
        self.assertIsNotNone(rep_w.results["buenos_a0"].metadata["mode_b_requantize"])

    def test_pallas_rejects_mismatched_shapes_and_aligns_large_row_blocks(self) -> None:
        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_gemm(a, b, out):
            freetpu.store(out, freetpu.dot(a, b))

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16", rhs_transposed=True)
        def k_gemm_t(a, b, out):
            freetpu.store(out, freetpu.dot(a, b))

        # GEMM contracting dimension mismatch (normal and rhs_transposed)
        with self.assertRaises(ValueError):
            k_gemm.lower(shapes={"a": (64, 512), "b": (1024, 512)}, target="cloud_tpu_pallas")
        with self.assertRaises(ValueError):
            k_gemm.run(
                np.zeros((64, 512), dtype=np.float32),
                np.zeros((1024, 512), dtype=np.float32),
                target="cloud_tpu_pallas",
                dry_run=True,
            )
        with self.assertRaises(ValueError):
            k_gemm_t.lower(shapes={"a": (64, 512), "b": (512, 1024)}, target="cloud_tpu_pallas")

        # SwiGLU contracting & gate/up width mismatches (normal and rhs_transposed)
        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16")
        def k_swiglu(x, wg, wu, wd, out):
            g = freetpu.silu(freetpu.dot(x, wg))
            u = freetpu.dot(x, wu)
            freetpu.store(out, freetpu.dot(g * u, wd))

        # Gate contracting mismatch
        with self.assertRaises(ValueError):
            k_swiglu.lower(
                shapes={"x": (64, 512), "wg": (1024, 512), "wu": (512, 512), "wd": (512, 512)},
                target="cloud_tpu_pallas",
            )
        # Up contracting mismatch
        with self.assertRaises(ValueError):
            k_swiglu.lower(
                shapes={"x": (64, 512), "wg": (512, 512), "wu": (1024, 512), "wd": (512, 512)},
                target="cloud_tpu_pallas",
            )
        # Gate vs up width mismatch
        with self.assertRaises(ValueError):
            k_swiglu.lower(
                shapes={"x": (64, 512), "wg": (512, 512), "wu": (512, 1024), "wd": (512, 512)},
                target="cloud_tpu_pallas",
            )
        # Down contracting mismatch
        with self.assertRaises(ValueError):
            k_swiglu.lower(
                shapes={"x": (64, 512), "wg": (512, 512), "wu": (512, 512), "wd": (1024, 512)},
                target="cloud_tpu_pallas",
            )

        @freetpu.tile_kernel(block_m=64, block_n=512, block_k=512, dtype="bf16", rhs_transposed=True)
        def k_swiglu_t(x, wg, wu, wd, out):
            g = freetpu.silu(freetpu.dot(x, wg))
            u = freetpu.dot(x, wu)
            freetpu.store(out, freetpu.dot(g * u, wd))

        with self.assertRaises(ValueError):
            k_swiglu_t.lower(
                shapes={"x": (64, 512), "wg": (512, 512), "wu": (1024, 512), "wd": (512, 512)},
                target="cloud_tpu_pallas",
            )

        # Row block alignment above 128: block_m=130 with M=256 must align bm to 136 (divisible by 8)
        @freetpu.tile_kernel(block_m=130, block_n=512, block_k=512, dtype="bf16")
        def k_large_bm(a, b, out):
            freetpu.store(out, freetpu.dot(a, b))

        lowered = k_large_bm.lower(shapes={"a": (256, 512), "b": (512, 512)}, target="cloud_tpu_pallas")
        self.assertEqual(lowered.summary["block_m"], 136)
        self.assertEqual(lowered.summary["block_m"] % 8, 0)
        self.assertEqual(lowered.summary["m_pad"], 272)
        self.assertEqual(lowered.summary["m_pad"] % lowered.summary["block_m"], 0)
        self.assertEqual(lowered.summary["grid"][0], 2)
        assert lowered.source_code is not None
        self.assertIn("pl.BlockSpec((136, 512)", lowered.source_code)

    def test_runner_host_preserves_fp16_subnormal_weight_scales(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        runner_host = repo_root / "runtime" / "build" / "freetpu_tile_runner_host"
        if not runner_host.is_file():
            self.skipTest("runtime/build/freetpu_tile_runner_host not built")

        rng = np.random.default_rng(2026)
        m, k, n = 16, 64, 64
        a = (rng.standard_normal((m, k)) * 10.0).astype(np.float32)
        # Max weight magnitude ~0.005 -> Q8 scale ~3.94e-5 (IEEE-754 FP16 subnormal!)
        w_nk = rng.uniform(-0.005, 0.005, size=(n, k)).astype(np.float32)
        ref = (a @ w_nk.T).astype(np.float32)
        self.assertGreater(float(np.max(np.abs(ref))), 1e-2)

        with tempfile.TemporaryDirectory(prefix="freetpu-subnormal-test-") as tmp:
            tp = Path(tmp)
            (tp / "a.bin").write_bytes(a.tobytes())
            (tp / "w.bin").write_bytes(w_nk.tobytes())
            out_p = tp / "out.bin"
            proc = subprocess.run(
                [
                    str(runner_host),
                    "--mock",
                    "--op",
                    "matmul",
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
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"runner failed:\n{proc.stderr}\n{proc.stdout}")
            got = np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(m, n)
            self.assertGreater(float(np.max(np.abs(got))), 0.0, msg="Subnormal FP16 scale was flushed to zero!")
            diff = got.astype(np.float64) - ref.astype(np.float64)
            nmse = float(np.sum(diff * diff) / max(float(np.sum(ref.astype(np.float64) ** 2)), 1e-30))
            self.assertLess(nmse, 5e-3, msg=f"Subnormal weight scale nmse too high: {nmse}")

    def test_packaging_tpu_extra_and_example_resolution(self) -> None:
        self.assertTrue(_resolve_example_script("fused_gemm_silu.py").is_file())
        self.assertTrue(_resolve_example_script("swiglu.py").is_file())
        pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8")
            self.assertIn('tpu = ["jax[tpu]"]', text)
            self.assertIn("freetpu_tile._packaged_examples", text)

    def test_colab_embedded_args_and_adb_runner_path_and_2proj_flops(self) -> None:
        from unittest import mock

        # 1. Script args colliding with colab run flags (--keep, --timeout) stay inside bundle SCRIPT_ARGS
        example = _resolve_example_script("fused_gemm_silu.py")
        with tempfile.TemporaryDirectory(prefix="freetpu-colab-args-") as tmp:
            tp = Path(tmp)
            rc, summary = run_colab_tile_kernel(
                example,
                script_args=("--target", "cloud_tpu_pallas", "--keep", "--timeout", "999"),
                out_dir=tp,
                dry_run=True,
            )
            self.assertEqual(rc, 0)
            run_cmds = [c for c in summary["planned_commands"] if "run" in c]
            self.assertTrue(run_cmds)
            for c in run_cmds:
                self.assertNotIn("--keep", c)
                self.assertNotIn("999", c)
            bundle_src = Path(summary["bundle_path"]).read_text(encoding="utf-8")
            self.assertIn("'--keep'", bundle_src)
            self.assertIn("'999'", bundle_src)

        # 2. Prebuilt runner path (env/kwarg) works without runtime tree + 2-proj SwiGLU FLOPs exclude down_proj
        @freetpu.tile_kernel(block_m=64, block_n=64, block_k=64, dtype="w8a16")
        def k_2proj_w8(x, wg, wu, out):
            freetpu.store(out, freetpu.silu(freetpu.dot(x, wg)) * freetpu.dot(x, wu))

        with tempfile.TemporaryDirectory(prefix="freetpu-adb-2proj-") as tmp:
            tp = Path(tmp)
            fake_runner = tp / "custom_runner"
            fake_runner.write_bytes(b"#!/bin/sh\n")
            x = np.ones((16, 64), dtype=np.float32)
            wg = np.ones((64, 64), dtype=np.float32)
            wu = np.ones((64, 64), dtype=np.float32)

            def _fake_adb(cmd, **_kw):
                c = list(cmd)
                if "shell" in c and any("custom_runner" in s or "freetpu_tile_runner" in s for s in c):
                    return subprocess.CompletedProcess(
                        c, 0, '{"ok":true,"hw_us":1000.0,"min_hw_us":1000.0,"wall_us":1000.0}', ""
                    )
                if "pull" in c:
                    Path(c[-1]).write_bytes(np.zeros((16, 64), dtype=np.float32).tobytes())
                return subprocess.CompletedProcess(c, 0, "", "")

            with mock.patch("freetpu_tile.backends.buenos_a0.subprocess.run", side_effect=_fake_adb):
                res = k_2proj_w8.run(
                    x, wg, wu,
                    target="buenos_a0",
                    adb_serial="fake-serial",
                    runner_path=fake_runner,
                    mock=False,
                )
            # 2-proj SwiGLU FLOPs = 4 * M * K * N_inter = 4 * 16 * 64 * 64 = 262,144 FLOPs
            # At hw_us = 1000 us (1e-3 s), gflops = 262,144 / 1e-3 / 1e9 = 0.000262 -> rounds to 0.0
            # Verify directly with larger 64x512x512 -> 4*64*512*512 = 67,108,864 FLOPs -> 67.109 GFLOP/s (not 100.663!)
            x_big = np.ones((64, 512), dtype=np.float32)
            wg_big = np.ones((512, 512), dtype=np.float32)
            wu_big = np.ones((512, 512), dtype=np.float32)

            def _fake_adb_big(cmd, **_kw):
                c = list(cmd)
                if "shell" in c and any("freetpu_tile_runner" in s for s in c):
                    return subprocess.CompletedProcess(
                        c, 0, '{"ok":true,"hw_us":1000.0,"min_hw_us":1000.0,"wall_us":1000.0}', ""
                    )
                if "pull" in c:
                    Path(c[-1]).write_bytes(np.zeros((64, 512), dtype=np.float32).tobytes())
                return subprocess.CompletedProcess(c, 0, "", "")

            with mock.patch.dict(os.environ, {"FREETPU_TILE_RUNNER_PATH": str(fake_runner)}):
                with mock.patch("freetpu_tile.backends.buenos_a0.subprocess.run", side_effect=_fake_adb_big):
                    res_big = k_2proj_w8.run(
                        x_big, wg_big, wu_big,
                        target="buenos_a0",
                        adb_serial="fake-serial",
                        mock=False,
                    )
            self.assertAlmostEqual(res_big.gflops or 0.0, 67.109, places=2)
            self.assertIsNotNone(res.output)

        # 3. Host runner rejects --dtype bf16 explicitly
        runner_host = Path(__file__).resolve().parent.parent.parent / "runtime" / "build" / "freetpu_tile_runner_host"
        if runner_host.is_file():
            proc = subprocess.run(
                [
                    str(runner_host), "--mock", "--op", "matmul", "--dtype", "bf16",
                    "--m", "16", "--k", "64", "--n", "64",
                    "--in", "/dev/null", "--w", "/dev/null", "--out", "/dev/null",
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("Unsupported --dtype", proc.stderr)


if __name__ == "__main__":
    unittest.main()
