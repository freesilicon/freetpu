#!/usr/bin/env python3
"""Single-source FreeTPU Tile DSL example: Fused GEMM + SiLU.

The exact same `@freetpu.tile_kernel` definition lowers and runs on:
  1. `cpu_ref`:          Pure NumPy deterministic reference
  2. `buenos_a0`:        Pixel Tensor G5 Edge TPU (VLIW + Mode-A 0x640 spline + libopenbuenos)
  3. `cloud_tpu_pallas`: Cloud TPU v5e-1 / v6e-1 via JAX Pallas/Mosaic (`tpu_custom_call`)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import freetpu_tile as freetpu  # noqa: E402
from freetpu_tile.backends.cpu_ref import quantize_w8a16_weights  # noqa: E402


def build_fused_gemm_silu_kernel(
    *,
    block_m: int = 64,
    block_n: int = 512,
    block_k: int = 512,
    dtype: str = "w8a16",
    quant: freetpu.QuantParams | None = None,
) -> freetpu.TileKernel:
    """Construct the single-source @freetpu.tile_kernel for Fused GEMM + SiLU."""

    @freetpu.tile_kernel(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        dtype=dtype,
        quant=quant,
    )
    def fused_gemm_silu(a, b, out):
        pid_m = freetpu.program_id(0)
        pid_n = freetpu.program_id(1)
        a_tile = freetpu.load(a, (pid_m, None), shape=(block_m, block_k))
        b_tile = freetpu.load(b, (None, pid_n), shape=(block_k, block_n))
        acc = freetpu.dot(a_tile, b_tile)
        y = freetpu.silu(acc)
        freetpu.store(out, y, (pid_m, pid_n))

    return fused_gemm_silu


def make_inputs(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    dtype: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], freetpu.QuantParams]:
    rng = np.random.default_rng(seed)
    qparams = freetpu.QuantParams(
        s_in=0.0625, zp_in=0, s_w=0.0625, zp_w=0, s_out=0.0625, zp_out=0
    )
    if dtype == "int8":
        a = rng.integers(-24, 25, size=(m_dim, k_dim), dtype=np.int8)
        b = rng.integers(-16, 17, size=(k_dim, n_dim), dtype=np.int8)
        return a, b, {}, qparams

    scale = 1.0 / math.sqrt(k_dim)
    a = (rng.standard_normal((m_dim, k_dim)) * 0.5).astype(np.float32)
    b_f32 = (rng.standard_normal((k_dim, n_dim)) * scale).astype(np.float32)
    if dtype == "w8a16":
        q_w, s_w = quantize_w8a16_weights(b_f32, rhs_transposed=False)
        return a, q_w, {"b": s_w}, qparams
    return a, b_f32, {}, qparams


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Single-source FreeTPU Tile DSL Fused GEMM + SiLU")
    ap.add_argument("--target", default="all", choices=("cpu_ref", "buenos_a0", "cloud_tpu_pallas", "all"))
    ap.add_argument("--dtype", default="w8a16", choices=("bf16", "w8a16", "int8"))
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=512)
    ap.add_argument("--block-k", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--adb-serial", default=None, help="ADB serial for live Tensor G5 (rango)")
    ap.add_argument("--dry-run", action="store_true", help="Lower without executing on hardware")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    a, b, scales, qparams = make_inputs(args.M, args.K, args.N, args.dtype)
    kernel = build_fused_gemm_silu_kernel(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        dtype=args.dtype,
        quant=qparams,
    )

    if args.target == "all":
        targets: List[str] = (
            ["cpu_ref", "cloud_tpu_pallas"]
            if args.dtype == "bf16"
            else ["cpu_ref", "buenos_a0", "cloud_tpu_pallas"]
        )
        dry_targets: List[str] = ["cloud_tpu_pallas"] if args.dry_run else []
        if args.dry_run:
            dry_targets = [t for t in targets if t != "cpu_ref"]
        else:
            # Locally without JAX TPU, automatically dry-run cloud_tpu_pallas if JAX TPU absent
            try:
                import jax  # type: ignore

                if jax.default_backend() != "tpu":
                    dry_targets.append("cloud_tpu_pallas")
            except ImportError:
                dry_targets.append("cloud_tpu_pallas")
        report = kernel.parity_check(
            a,
            b,
            targets=targets,
            dry_run_targets=dry_targets,
            scales=scales,
            adb_serial=args.adb_serial,
            warmup=args.warmup,
            iters=args.iters,
        )
        payload: Dict[str, Any] = report.to_dict()
        payload["ir"] = kernel.ir.to_dict()
        ok = report.passed
    else:
        tgt = freetpu.Target.parse(args.target)
        dry = args.dry_run and tgt != freetpu.Target.CPU_REF
        if tgt == freetpu.Target.CPU_REF:
            res = kernel.run(a, b, target=tgt, scales=scales)
            payload = {"ir": kernel.ir.to_dict(), "result": res.summary_dict()}
            ok = True
        else:
            report = kernel.parity_check(
                a,
                b,
                targets=["cpu_ref", tgt.value],
                dry_run_targets=[tgt.value] if dry else [],
                scales=scales,
                adb_serial=args.adb_serial,
                warmup=args.warmup,
                iters=args.iters,
            )
            payload = report.to_dict()
            payload["ir"] = kernel.ir.to_dict()
            ok = report.passed

    out_text = json.dumps(payload, indent=2)
    print(out_text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(out_text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
