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


if __name__ == "__main__":
    unittest.main()
