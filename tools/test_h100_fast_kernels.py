#!/usr/bin/env python3
"""Offline unit & parity tests for `tools/h100_fast_kernels.py` (CI-safe with or without PyTorch)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import h100_fast_kernels as hfk  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class TestFastKernelsAlways(unittest.TestCase):
    """Tests that always pass even on stdlib-only CI runners without PyTorch."""

    def test_self_test_entrypoint(self) -> None:
        res = hfk.run_offline_self_test()
        self.assertTrue(res["all_passed"])

    def test_startup_artifact_cache_integrity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="freetpu_cache_ut_") as td:
            cache = hfk.StartupArtifactCache(td)
            src = Path(td) / "weights.bin"
            src.write_bytes(b"deterministic-public-payload-123456")
            entry = cache.put_file("pruned_base", src, metadata={"experts": 64})
            hit = cache.get_file("pruned_base", expected_sha256=entry["sha256"])
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual(hit.read_bytes(), b"deterministic-public-payload-123456")
            # Mismatched expected SHA fails closed
            self.assertIsNone(cache.get_file("pruned_base", expected_sha256="0" * 64))
            # Corrupted file on disk fails closed
            hit.write_bytes(b"corrupted")
            self.assertIsNone(cache.get_file("pruned_base"))

    def test_probes_never_crash(self) -> None:
        c1d = hfk.ensure_causal_conv1d(install=False)
        self.assertIn("backend", c1d)
        ro = hfk.probe_fast_rollout_backends()
        self.assertIn(ro["recommended_backend"], ("vllm_sleep_mode", "torch_compile_cudagraphs", "eager_generate"))


@unittest.skipIf(torch is None, "PyTorch not installed on this CI runner")
class TestFastKernelsTorch(unittest.TestCase):
    """Full numerical parity & zero-leakage tests when PyTorch is available."""

    def test_sequence_packing_zero_attn_and_sconv_leak(self) -> None:
        torch.manual_seed(20261126)
        exs = [
            {"prompt_ids": [10, 11, 12, 13], "completion_ids": [14, 15, 16]},
            {"prompt_ids": [20, 21, 22], "completion_ids": [23, 24, 25, 26]},
            {"prompt_ids": [30, 31], "completion_ids": [32, 33, 34]},
        ]
        batches, stats = hfk.pack_sft_sequences(exs, max_len=64, pad_id=0, eos_id=2, sconv_guard=3)
        self.assertEqual(len(batches), 1)
        self.assertEqual(stats["packing_speedup_factor"], 3.0)
        b = batches[0]

        d = 32
        seq_len = b.total_len
        q = torch.randn(1, 1, seq_len, d)
        k = torch.randn(1, 1, seq_len, d)
        v = torch.randn(1, 1, seq_len, d)

        def _sdpa(qt, kt, vt, m4d):
            return F.softmax((qt @ kt.transpose(-2, -1)) / math.sqrt(d) + m4d, dim=-1) @ vt

        seg1 = b.segment_ids[0] == 1
        other = b.segment_ids[0] > 1
        k_mut, v_mut = k.clone(), v.clone()
        k_mut[:, :, seg1, :] += 100.0
        v_mut[:, :, seg1, :] += 100.0
        attn_leak = float((_sdpa(q, k, v, b.attention_mask_4d)[:, :, other, :] - _sdpa(q, k_mut, v_mut, b.attention_mask_4d)[:, :, other, :]).abs().max().item())
        self.assertEqual(attn_leak, 0.0)

        conv = nn.Conv1d(d, d, kernel_size=4, groups=d, bias=False)
        x = torch.randn(1, d, seq_len)
        x_mut = x.clone()
        x_mut[:, :, seg1] += 100.0
        y0 = conv(F.pad(x, (3, 0)))
        y1 = conv(F.pad(x_mut, (3, 0)))
        sconv_leak = float((y0[:, :, other] - y1[:, :, other]).abs().max().item())
        self.assertEqual(sconv_leak, 0.0)

    def test_grouped_moe_and_backward_parity(self) -> None:
        torch.manual_seed(42)
        n_tok, h, inter, e, k, r = 16, 64, 32, 8, 4, 8
        x = torch.randn(n_tok, h, requires_grad=True)
        idx = torch.topk(torch.randn(n_tok, e), k=k, dim=-1).indices
        w = torch.softmax(torch.randn(n_tok, k), dim=-1) * 8.0
        w_gu = torch.randn(e, 2 * inter, h) * 0.02
        w_dn = torch.randn(e, h, inter) * 0.02
        la_gu = (torch.randn(e, r, h) * 0.02).requires_grad_(True)
        lb_gu = (torch.randn(e, 2 * inter, r) * 0.02).requires_grad_(True)
        la_dn = (torch.randn(e, r, inter) * 0.02).requires_grad_(True)
        lb_dn = (torch.randn(e, h, r) * 0.02).requires_grad_(True)

        # Reference naive token-by-token loop
        x_r = x.detach().clone().requires_grad_(True)
        out_r = torch.zeros((n_tok, h), dtype=torch.float32)
        for t in range(n_tok):
            xt = x_r[t : t + 1]
            for s in range(k):
                ei = int(idx[t, s].item())
                gu = F.linear(xt, w_gu[ei]) + F.linear(F.linear(xt, la_gu[ei]), lb_gu[ei]) * 2.0
                act = hfk.default_situ_activation(gu)
                dn = F.linear(act, w_dn[ei]) + F.linear(F.linear(act, la_dn[ei]), lb_dn[ei]) * 2.0
                out_r[t] = out_r[t] + dn[0] * w[t, s]

        out_f, _ = hfk.grouped_moe_dispatch(
            x, idx, w, w_gu, w_dn,
            lora_gu_A=la_gu, lora_gu_B=lb_gu, lora_dn_A=la_dn, lora_dn_B=lb_dn,
            lora_scaling=2.0, use_grouped_mm="auto",
        )
        self.assertLessEqual(float((out_r - out_f).abs().max().item()), 1e-4)
        g = torch.randn_like(out_r)
        out_r.backward(g)
        out_f.backward(g)
        self.assertLessEqual(float((x_r.grad - x.grad).abs().max().item()), 1e-4)

    def test_liger_guards_and_fp8_filter(self) -> None:
        class _DummyRMSNorm(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))

        class _UnweightedRMSNorm(nn.Module):
            pass

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1 = _DummyRMSNorm(256)
                self.v_norm = _UnweightedRMSNorm()
                self.q_proj = nn.Linear(256, 256)
                self.mlp_gate = nn.Linear(256, 64)
                self.lm_head = nn.Linear(256, 1024)

        m = _M()
        c80 = hfk.configure_liger_and_checkpointing(m, vram_gb=80.0, enable_liger=True)
        self.assertFalse(c80["gradient_checkpointing_enabled"])
        self.assertEqual(c80["liger"]["weighted_rmsnorm_candidates"], 1)
        self.assertEqual(c80["liger"]["skipped_unweighted_rmsnorm_count"], 1)

        fp8 = hfk.configure_hopper_fp8(m, enabled=True, sm_override=(8, 0))
        self.assertFalse(fp8["active"])
        self.assertEqual(fp8["eligible_linear_layers"], 1)  # q_proj only (lm_head & 64-out gate skipped)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unit tests for tools/h100_fast_kernels.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
